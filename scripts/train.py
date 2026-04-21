"""
Train YOLOv8-seg on runway segmentation data.

Expects a dataset in YOLO segmentation format at data/datasets/runway/
with a dataset.yaml file.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATASETS_DIR, INFERENCE_SIZE


def main():
    parser = argparse.ArgumentParser(description="Train runway segmentation model")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=INFERENCE_SIZE)
    parser.add_argument("--dataset", default=None, help="Path to dataset.yaml")
    parser.add_argument("--name", default="runway_seg", help="Run name")
    args = parser.parse_args()

    dataset_yaml = args.dataset or str(Path(DATASETS_DIR) / "runway" / "dataset.yaml")

    from ultralytics import YOLO
    model = YOLO(args.model)

    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        project="logs",
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        perspective=0.001,
    )

    print(f"\nTraining complete. Best weights: logs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
