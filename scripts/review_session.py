"""
Interactive reviewer for auto-labeled X-Plane capture sessions.

For each labeled frame you can:
  - Accept as-is, or after dragging vertices to correct positions
  - Deny  (label file is deleted)
  - Skip  (move on, no changes)

Usage:
  python scripts/review_session.py --session 20260424_125550
  python scripts/review_session.py --session 20260424_125550 --n 50
  python scripts/review_session.py --session 20260424_125550 --sequential

Controls:
  a / Enter   = accept (save polygon, possibly edited)
  d / Delete  = deny   (delete label file)
  Space / s   = skip   (no changes)
  r           = reset polygon to original
  q           = quit
  Left-click + drag near a vertex = move that vertex
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import LABELS_DIR, RAW_FRAMES_DIR

MAX_W = 1400
MAX_H = 900
HIT_RADIUS = 14  # hit detection radius (px) -- larger than drawn dot for easier grabbing
DOT_RADIUS = 4   # visual dot size


class Editor:
    def __init__(self, img, polygons):
        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        self.scale = scale
        self.orig_w, self.orig_h = w, h
        self.base = cv2.resize(img, (int(w * scale), int(h * scale)))

        self.polys = [p.astype(float).copy() for p in polygons]
        self.orig_polys = [p.astype(float).copy() for p in polygons]

        self._drag_pi = -1
        self._drag_vi = -1
        self.dragging = False
        self.result = None

    def _to_disp(self, pts):
        return (np.array(pts) * self.scale).astype(np.int32)

    def _to_orig(self, x, y):
        return (
            float(np.clip(x / self.scale, 0, self.orig_w - 1)),
            float(np.clip(y / self.scale, 0, self.orig_h - 1)),
        )

    def _nearest_vertex(self, xd, yd):
        best_d, best = HIT_RADIUS + 1, (-1, -1)
        for pi, poly in enumerate(self.polys):
            for vi, pt in enumerate(self._to_disp(poly)):
                d = float(np.hypot(pt[0] - xd, pt[1] - yd))
                if d < best_d:
                    best_d, best = d, (pi, vi)
        return best

    def mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pi, vi = self._nearest_vertex(x, y)
            if pi >= 0:
                self._drag_pi, self._drag_vi = pi, vi
                self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            ox, oy = self._to_orig(x, y)
            self.polys[self._drag_pi][self._drag_vi] = [ox, oy]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def render(self, title_text):
        disp = self.base.copy()
        dh, dw = disp.shape[:2]

        overlay = disp.copy()
        for poly in self.polys:
            cv2.fillPoly(overlay, [self._to_disp(poly)], (0, 255, 0))
        disp = cv2.addWeighted(disp, 0.55, overlay, 0.45, 0)

        for pi, poly in enumerate(self.polys):
            pts_d = self._to_disp(poly)
            cv2.polylines(disp, [pts_d], True, (0, 255, 0), 1)
            for vi, pt in enumerate(pts_d):
                is_active = self.dragging and pi == self._drag_pi and vi == self._drag_vi
                r = DOT_RADIUS + 2 if is_active else DOT_RADIUS
                cv2.circle(disp, tuple(pt), r + 1, (0, 0, 0), -1)      # black border
                cv2.circle(disp, tuple(pt), r, (0, 200, 255), -1)       # dot

        _shadow_text(disp, title_text, (10, 26), 0.65)
        _shadow_text(disp, "a/Enter=accept    d/Del=deny    Space=skip    r=reset    q=quit    arrows=nudge",
                     (10, dh - 12), 0.55)
        return disp

    def _nudge(self, dx, dy):
        for poly in self.polys:
            poly[:, 0] = np.clip(poly[:, 0] + dx, 0, self.orig_w - 1)
            poly[:, 1] = np.clip(poly[:, 1] + dy, 0, self.orig_h - 1)

    def reset(self):
        self.polys = [p.copy() for p in self.orig_polys]

    def run(self, window_name, title_text):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        dh, dw = self.base.shape[:2]
        cv2.resizeWindow(window_name, dw, dh)
        cv2.setMouseCallback(window_name, self.mouse)

        while self.result is None:
            cv2.imshow(window_name, self.render(title_text))
            key = cv2.waitKeyEx(16)  # waitKeyEx returns full extended keycodes for arrow keys
            if key == -1:
                continue

            k = key & 0xFF

            # Arrow keys (Windows extended keycodes via waitKeyEx)
            NUDGE = 1.0 / self.scale  # 1 display pixel in original image coords
            if key == 2490368:    # up
                self._nudge(0, -NUDGE)
            elif key == 2621440:  # down
                self._nudge(0, NUDGE)
            elif key == 2424832:  # left
                self._nudge(-NUDGE, 0)
            elif key == 2555904:  # right
                self._nudge(NUDGE, 0)
            elif k in (ord('a'), 13):
                self.result = 'accept'
            elif k in (ord('d'), 127, 8):
                self.result = 'deny'
            elif k in (ord('s'), 32):
                self.result = 'skip'
            elif k == ord('r'):
                self.reset()
            elif k == ord('q'):
                self.result = 'quit'

        return self.result, self.polys


def _shadow_text(img, text, pos, scale):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def load_label(label_path, img_w, img_h):
    polygons, class_ids = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_ids.append(int(parts[0]))
            coords = [float(x) for x in parts[1:]]
            pts = [[coords[i] * img_w, coords[i + 1] * img_h]
                   for i in range(0, len(coords) - 1, 2)]
            polygons.append(np.array(pts, dtype=float))
    return polygons, class_ids


def save_label(label_path, polygons, class_ids, img_w, img_h):
    with open(label_path, "w") as f:
        for poly, cid in zip(polygons, class_ids):
            parts = [str(cid)]
            for x, y in poly:
                parts.append(f"{x / img_w:.6f}")
                parts.append(f"{y / img_h:.6f}")
            f.write(" ".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Review auto-labeled X-Plane session frames")
    parser.add_argument("--session", required=True, help="Session folder name (e.g. 20260424_125550)")
    parser.add_argument("--n", type=int, default=None, help="Max number of frames to review")
    parser.add_argument("--sequential", action="store_true", help="Step through in frame order instead of random")
    args = parser.parse_args()

    frames_dir = Path(RAW_FRAMES_DIR) / args.session / "frames"
    labels_dir = Path(LABELS_DIR) / args.session

    if not frames_dir.exists():
        print(f"Session frames not found: {frames_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"No labels found for session: {labels_dir}")
        print("Run auto_label.py first to generate labels.")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"No label files in {labels_dir}")
        sys.exit(1)

    if not args.sequential:
        random.shuffle(label_files)
    if args.n is not None:
        label_files = label_files[:args.n]

    print(f"Session: {args.session}")
    print(f"Frames:  {frames_dir}")
    print(f"Labels:  {labels_dir}")
    print(f"Reviewing {len(label_files)} labeled frames")
    print("Controls: a/Enter=accept  d/Del=deny  Space=skip  r=reset  q=quit\n")

    accepted = denied = skipped = 0

    for i, label_path in enumerate(label_files):
        frame_file = frames_dir / label_path.with_suffix(".jpg").name
        if not frame_file.exists():
            continue

        img = cv2.imread(str(frame_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        polygons, class_ids = load_label(label_path, w, h)

        title = f"[{i+1}/{len(label_files)}]  {frame_file.name}"
        editor = Editor(img, polygons)
        result, edited_polys = editor.run("Session Review", title)

        if result == 'quit':
            break
        elif result == 'accept':
            save_label(label_path, edited_polys, class_ids, w, h)
            accepted += 1
        elif result == 'deny':
            label_path.unlink()
            denied += 1
        else:
            skipped += 1

    cv2.destroyAllWindows()
    print(f"\nDone.  Accepted: {accepted}  |  Denied: {denied}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
