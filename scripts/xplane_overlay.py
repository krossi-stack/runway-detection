"""
Transparent runway polygon overlay drawn directly on top of the X-Plane window.

Auto-detects the X-Plane window, runs YOLO segmentation in a background thread,
and draws a see-through trapezoid overlay on top of X-Plane in real time.

Stability features:
  - Temporal smoothing (EMA) on corner positions
  - Hysteresis on confidence threshold (higher to appear, lower to disappear)
  - Consistent corner ordering (TL, TR, BR, BL) before smoothing

Usage:
  python scripts/xplane_overlay.py
  python scripts/xplane_overlay.py --model models/runway_seg_xplane_v1.pt
  python scripts/xplane_overlay.py --conf 0.3 --fps 10 --smooth 0.35

Controls:
  Ctrl+Q = quit
"""

import argparse
import ctypes
import ctypes.wintypes
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import keyboard
import mss
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CONFIDENCE_THRESHOLD, INFERENCE_SIZE, MODEL_PATH, XPLANE_SCREEN_REGION
from scripts.jetson_infer import RunwaySegmenter

TRANSPARENT_COLOR = "#000000"
POLY_COLOR = "#00ff00"
CONF_COLOR = "#ffffff"

# Hysteresis thresholds — polygon appears at ON, disappears below OFF
CONF_HYSTERESIS_ON  = 1.15  # multiplier above base conf to trigger appearance
CONF_HYSTERESIS_OFF = 0.75  # multiplier below base conf to trigger disappearance


# ---------------------------------------------------------------------------
# Window detection
# ---------------------------------------------------------------------------

def find_xplane_window():
    found = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if "X-Plane" in title or "X-System" in title:
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 100 and h > 100:
                found.append({
                    "left": rect.left, "top": rect.top,
                    "width": w, "height": h,
                    "title": title,
                })
        return True

    ctypes.windll.user32.EnumWindows(WNDENUMPROC(callback), 0)
    return found[0] if found else None


def parse_region(region_str):
    parts = [int(x) for x in region_str.split(",")]
    if len(parts) == 4:
        return {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    return None


# ---------------------------------------------------------------------------
# Quad fitting + corner ordering
# ---------------------------------------------------------------------------

def fit_quad(pts):
    """Reduce a polygon mask to exactly 4 corners (convex hull + approxPolyDP)."""
    p = pts.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(p).reshape(-1, 2)

    # Adaptive approximation — stop as soon as we reach exactly 4
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    for scale in np.arange(0.02, 0.5, 0.01):
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), scale * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)

    # Fallback: always return exactly 4 extreme points
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmin(-hull[:, 0] + hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmax(-hull[:, 0] + hull[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_quad(pts):
    """Sort 4 points into consistent order: TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    # arctan2 order: left, bottom, right, top -> rotate to TL, TR, BR, BL
    # Find the point closest to top-left (min x+y)
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = np.argmin(sums)
    # Reorder starting from TL going clockwise
    n = len(pts)
    # Sort by angle from center, starting from top-left quadrant
    angles_from_tl = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    # Shift angles so TL comes first (clockwise: TL, TR, BR, BL)
    ref_angle = angles_from_tl[tl_idx]
    shifted = (angles_from_tl - ref_angle) % (2 * np.pi)
    sorted_idx = np.argsort(shifted)
    return pts[sorted_idx]


# ---------------------------------------------------------------------------
# Temporal smoother
# ---------------------------------------------------------------------------

class QuadSmoother:
    """EMA smoother for a 4-corner trapezoid with hysteresis on visibility."""

    def __init__(self, alpha, conf_base, conf_on_mult, conf_off_mult):
        self.alpha = alpha          # EMA weight for new measurement (0=frozen, 1=no smoothing)
        self.conf_on  = conf_base * conf_on_mult
        self.conf_off = conf_base * conf_off_mult
        self._quad = None           # current smoothed quad (4x2 float)
        self._visible = False

    def update(self, quad, conf):
        """Feed a new detection. quad=None means no detection this frame."""
        if quad is not None:
            ordered = order_quad(quad.astype(np.float32))
            if self._quad is None:
                self._quad = ordered.copy()
            else:
                self._quad = self.alpha * ordered + (1.0 - self.alpha) * self._quad

            if not self._visible and conf >= self.conf_on:
                self._visible = True
        else:
            if self._visible and conf < self.conf_off:
                self._visible = False
            # Keep _quad at last known position (don't reset)

    @property
    def quad(self):
        return self._quad if self._visible else None


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    def __init__(self, segmenter, region, target_fps, smoother, state, lock, stop_event):
        super().__init__(daemon=True)
        self.segmenter = segmenter
        self.region = region
        self.interval = 1.0 / target_fps
        self.smoother = smoother
        self.state = state
        self.lock = lock
        self.stop_event = stop_event

    def run(self):
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()

                img = np.array(sct.grab(self.region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                result = self.segmenter.predict(frame)

                quad_raw, conf = self._best_quad(result)
                self.smoother.update(quad_raw, conf)

                with self.lock:
                    self.state["quad"] = self.smoother.quad  # None or 4x2 array
                    self.state["conf"] = conf
                    self.state["size"] = (frame.shape[1], frame.shape[0])

                elapsed = time.perf_counter() - t0
                wait = self.interval - elapsed
                if wait > 0:
                    time.sleep(wait)

    def _best_quad(self, result):
        if result is None or result.masks is None or not result.masks.xy:
            return None, 0.0
        conf = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes) else 0.0
        pts = result.masks.xy[0]
        if len(pts) < 3:
            return None, conf
        return fit_quad(pts.astype(np.float32)), conf


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------

class OverlayWindow:
    def __init__(self, root, region, state, lock, stop_event):
        self.root = root
        self.region = region
        self.state = state
        self.lock = lock
        self.stop_event = stop_event

        x, y, w, h = region["left"], region["top"], region["width"], region["height"]

        root.geometry(f"{w}x{h}+{x}+{y}")
        root.configure(bg=TRANSPARENT_COLOR)
        root.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-alpha", 1.0)
        root.overrideredirect(True)

        self.canvas = tk.Canvas(root, width=w, height=h,
                                bg=TRANSPARENT_COLOR, highlightthickness=0)
        self.canvas.pack()
        self._poll()

    def quit(self):
        self.stop_event.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _poll(self):
        if self.stop_event.is_set():
            try:
                self.root.destroy()
            except Exception:
                pass
            return

        with self.lock:
            quad = self.state.get("quad", None)
            conf = self.state.get("conf", 0.0)
            size = self.state.get("size", None)

        self.canvas.delete("all")

        if quad is not None and size is not None:
            img_w, img_h = size
            sx = self.region["width"] / img_w
            sy = self.region["height"] / img_h

            scaled = [(float(pt[0] * sx), float(pt[1] * sy)) for pt in quad]
            flat = [c for pt in scaled for c in pt]
            self.canvas.create_polygon(flat, outline=POLY_COLOR, fill="", width=2)

            if conf > 0:
                self.canvas.create_text(10, 30, anchor="nw",
                                        text=f"runway  {conf:.2f}",
                                        fill=CONF_COLOR,
                                        font=("Helvetica", 14, "bold"))

        self.root.after(33, self._poll)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Transparent runway overlay on X-Plane window")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=INFERENCE_SIZE)
    parser.add_argument("--region", default=XPLANE_SCREEN_REGION,
                        help="x,y,w,h — auto-detected from X-Plane window if omitted")
    parser.add_argument("--fps", type=int, default=10,
                        help="Inference frames per second (default: 10)")
    parser.add_argument("--smooth", type=float, default=0.35,
                        help="EMA weight for new detections, 0=frozen 1=no smoothing (default: 0.35)")
    args = parser.parse_args()

    if args.region:
        region = parse_region(args.region)
        if region is None:
            print("Invalid --region format, expected x,y,w,h")
            sys.exit(1)
    else:
        win = find_xplane_window()
        if win is None:
            print("X-Plane window not found. Start X-Plane or pass --region x,y,w,h manually.")
            sys.exit(1)
        region = win
        print(f"Found window: \"{win['title']}\"  at {win['left']},{win['top']}  {win['width']}x{win['height']}")

    print(f"Loading model: {args.model}")
    segmenter = RunwaySegmenter(args.model, args.conf, args.imgsz)

    smoother = QuadSmoother(
        alpha=args.smooth,
        conf_base=args.conf,
        conf_on_mult=CONF_HYSTERESIS_ON,
        conf_off_mult=CONF_HYSTERESIS_OFF,
    )

    state = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    infer_thread = InferenceThread(segmenter, region, args.fps, smoother, state, lock, stop_event)
    infer_thread.start()

    root = tk.Tk()
    overlay = OverlayWindow(root, region, state, lock, stop_event)

    keyboard.add_hotkey("ctrl+q", overlay.quit)
    print("Overlay running. Press Ctrl+Q anywhere to quit.\n")

    root.mainloop()
    stop_event.set()
    keyboard.unhook_all()


if __name__ == "__main__":
    main()
