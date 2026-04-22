"""
Export a YOLO .pt model to a TensorRT engine optimized for Jetson.

Run this ON the Jetson -- the engine is hardware-specific.

Usage:
    python scripts/export_tensorrt.py                        # uses defaults from settings
    python scripts/export_tensorrt.py --model models/runway_seg.pt --imgsz 320 --fp16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import INFERENCE_SIZE, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=INFERENCE_SIZE)
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 (default: True, use --no-fp16 to disable)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"Loading {model_path}...")
    model = YOLO(str(model_path))

    print(f"Exporting to TensorRT (imgsz={args.imgsz}, fp16={args.fp16})...")
    print("This may take 10-30 minutes on Jetson Nano.\n")

    engine_path = model.export(format="engine", imgsz=args.imgsz, half=args.fp16)
    print(f"\nDone. Engine saved to: {engine_path}")
    print(f"Set MODEL_PATH={engine_path} or place it alongside the .pt file.")


if __name__ == "__main__":
    main()
