"""
Real-time runway segmentation on the Jetson Nano.

Reads RTSP from the Marshall CV574 and runs YOLOv8-seg inference,
overlaying the runway mask and estimated altitude/alignment.
"""

import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CAMERA_FPS,
    CONFIDENCE_THRESHOLD,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INFERENCE_SIZE,
    MODEL_PATH,
    RTSP_URL,
)


class RunwaySegmenter:
    def __init__(self, model_path: str, conf: float = 0.5, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame: np.ndarray):
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        return results[0] if results else None

    @staticmethod
    def draw_mask(frame: np.ndarray, result, color=(0, 255, 0), alpha=0.4):
        if result is None or result.masks is None:
            return frame
        overlay = frame.copy()
        for mask_data in result.masks.data:
            mask = mask_data.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay[mask > 0] = (
                overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
            ).astype(np.uint8)
        return overlay


class CameraStream:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def main():
    print(f"Connecting to {RTSP_URL}...")
    cam = CameraStream(RTSP_URL)

    print(f"Loading model from {MODEL_PATH}...")
    segmenter = RunwaySegmenter(MODEL_PATH, CONFIDENCE_THRESHOLD, INFERENCE_SIZE)

    frame_count = 0
    fps_start = time.perf_counter()

    print("Running inference. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Lost camera feed, reconnecting...")
                cam.release()
                time.sleep(1)
                cam = CameraStream(RTSP_URL)
                continue

            result = segmenter.predict(frame)
            display = RunwaySegmenter.draw_mask(frame, result)

            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()
                cv2.putText(
                    display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            cv2.imshow("Runway Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
