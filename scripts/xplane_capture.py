"""
X-Plane screen capture + UDP telemetry recorder.

Captures screen frames and X-Plane telemetry simultaneously, saving them
with synchronized timestamps for training data generation.

Controls:
  1 = Start/resume capture
  2 = Pause capture
  3 = Stop and save session

X-Plane setup:
  Settings > Data Output > check "Internet via UDP" for:
    - Index 20: latitude, longitude, altitude (MSL & AGL)
    - Index 17: pitch, roll, heading
    - Index 3 : speeds (Vind, Vtrue, etc.)
  Set UDP rate and destination IP:port to match XPLANE_UDP_IP:XPLANE_UDP_PORT.
"""

import csv
import json
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import keyboard
import mss
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CAPTURE_FPS,
    RAW_FRAMES_DIR,
    XPLANE_SCREEN_REGION,
    XPLANE_UDP_IP,
    XPLANE_UDP_PORT,
)


def parse_xplane_udp(data: bytes) -> dict:
    """Parse an X-Plane DATA packet into a dict of index -> list of floats."""
    if data[:4] != b"DATA":
        return {}
    results = {}
    offset = 5  # skip "DATA" + padding byte
    while offset + 36 <= len(data):
        index = struct.unpack_from("<i", data, offset)[0]
        values = struct.unpack_from("<8f", data, offset + 4)
        results[index] = list(values)
        offset += 36
    return results


def telemetry_listener(udp_ip, udp_port, telemetry_state, lock, stop_event):
    """Background thread: continuously updates telemetry_state from UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(1.0)

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        parsed = parse_xplane_udp(data)
        if parsed:
            with lock:
                telemetry_state.update(parsed)

    sock.close()


def get_screen_region():
    """Parse XPLANE_SCREEN_REGION or return None for full primary monitor."""
    if not XPLANE_SCREEN_REGION:
        return None
    parts = [int(x) for x in XPLANE_SCREEN_REGION.split(",")]
    if len(parts) == 4:
        return {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    return None


def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(RAW_FRAMES_DIR) / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = session_dir / "telemetry.csv"
    metadata_path = session_dir / "metadata.json"

    telemetry_state = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    tel_thread = threading.Thread(
        target=telemetry_listener,
        args=(XPLANE_UDP_IP, XPLANE_UDP_PORT, telemetry_state, lock, stop_event),
        daemon=True,
    )
    tel_thread.start()

    region = get_screen_region()
    interval = 1.0 / CAPTURE_FPS
    frame_count = 0
    capturing = False
    running = True

    def on_start():
        nonlocal capturing
        if not capturing:
            capturing = True
            print("\n>> CAPTURING  (2=pause, 3=stop)")

    def on_pause():
        nonlocal capturing
        if capturing:
            capturing = False
            print("\n>> PAUSED     (1=resume, 3=stop)")

    def on_stop():
        nonlocal running
        running = False
        print("\n>> STOPPING...")

    keyboard.on_press_key("1", lambda _: on_start(), suppress=False)
    keyboard.on_press_key("2", lambda _: on_pause(), suppress=False)
    keyboard.on_press_key("3", lambda _: on_stop(), suppress=False)

    print(f"Session: {session_id}")
    print(f"Saving to: {session_dir}")
    print(f"Capture FPS: {CAPTURE_FPS}")
    print(f"Listening for X-Plane UDP on {XPLANE_UDP_IP}:{XPLANE_UDP_PORT}")
    print()
    print("Controls:")
    print("  1 = Start / resume capture")
    print("  2 = Pause capture")
    print("  3 = Stop and save session")
    print()
    print(">> PAUSED     (press 1 to start capturing)")

    csv_file = open(telemetry_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame", "timestamp",
        "lat", "lon", "alt_msl_ft", "alt_agl_ft",
        "pitch", "roll", "heading",
        "vind_kts",
    ])

    try:
        with mss.mss() as sct:
            monitor = region or sct.monitors[1]

            while running:
                t0 = time.perf_counter()

                if not capturing:
                    time.sleep(0.05)
                    continue

                timestamp = time.time()

                img = np.array(sct.grab(monitor))
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                fname = f"{frame_count:06d}.jpg"
                cv2.imwrite(str(frames_dir / fname), frame_bgr)

                with lock:
                    tel = dict(telemetry_state)

                pos = tel.get(20, [0] * 8)
                att = tel.get(17, [0] * 8)
                spd = tel.get(3, [0] * 8)

                alt_msl_ft = pos[2]
                terrain_ft = pos[4]
                alt_agl_ft = alt_msl_ft - terrain_ft

                writer.writerow([
                    frame_count, f"{timestamp:.6f}",
                    f"{pos[0]:.8f}", f"{pos[1]:.8f}", f"{alt_msl_ft:.2f}", f"{alt_agl_ft:.2f}",
                    f"{att[0]:.4f}", f"{att[1]:.4f}", f"{att[2]:.4f}",
                    f"{spd[0]:.2f}",
                ])

                frame_count += 1
                if frame_count % (CAPTURE_FPS * 5) == 0:
                    alt_agl = alt_agl_ft
                    print(f"  Frames: {frame_count}  |  AGL: {alt_agl:.0f} ft  |  HDG: {att[2]:.0f}")

                elapsed = time.perf_counter() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        csv_file.close()
        keyboard.unhook_all()

        metadata = {
            "session_id": session_id,
            "total_frames": frame_count,
            "capture_fps": CAPTURE_FPS,
            "screen_region": region,
            "udp_port": XPLANE_UDP_PORT,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSession complete: {frame_count} frames captured.")
        print(f"Frames:    {frames_dir}")
        print(f"Telemetry: {telemetry_path}")
        print(f"Metadata:  {metadata_path}")


if __name__ == "__main__":
    main()
