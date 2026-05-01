"""
Assemble labeled X-Plane capture sessions into a YOLO-seg dataset.

Scans all sessions under data/labels/ and data/raw_frames/, collects frames
that have a matching label file, then splits them into train/val/test and
writes to data/datasets/xplane/.

Usage:
  python scripts/build_xplane_dataset.py
  python scripts/build_xplane_dataset.py --val 0.15 --test 0.05
  python scripts/build_xplane_dataset.py --out data/datasets/xplane_v2
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATASETS_DIR, LABELS_DIR, RAW_FRAMES_DIR


def collect_pairs(labels_root: Path, frames_root: Path):
    """Return list of (image_path, label_path) for all labeled frames."""
    pairs = []
    for label_dir in sorted(labels_root.iterdir()):
        if not label_dir.is_dir():
            continue
        session = label_dir.name
        frames_dir = frames_root / session / "frames"
        if not frames_dir.exists():
            print(f"  Warning: no frames dir for session {session}, skipping")
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            img_path = frames_dir / label_path.with_suffix(".jpg").name
            if img_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"  Warning: label {label_path.name} has no matching frame, skipping")
    return pairs


def split(pairs, val_frac, test_frac, seed=42):
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def copy_pairs(pairs, images_dir: Path, labels_dir: Path, session_prefix=True):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        session = img_path.parent.parent.name  # session id
        stem = f"{session}_{img_path.stem}" if session_prefix else img_path.stem
        shutil.copy2(img_path, images_dir / f"{stem}.jpg")
        shutil.copy2(label_path, labels_dir / f"{stem}.txt")


def main():
    parser = argparse.ArgumentParser(description="Build YOLO-seg xplane dataset from labeled sessions")
    parser.add_argument("--out", default=None, help="Output dataset dir (default: data/datasets/xplane)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.05, help="Test fraction (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_root = Path(LABELS_DIR)
    frames_root = Path(RAW_FRAMES_DIR)
    out_dir = Path(args.out) if args.out else Path(DATASETS_DIR) / "xplane"

    print(f"Scanning sessions in {labels_root} ...")
    pairs = collect_pairs(labels_root, frames_root)

    if not pairs:
        print("No labeled frames found.")
        sys.exit(1)

    print(f"Found {len(pairs)} labeled frames across {len(list(labels_root.iterdir()))} sessions")

    train, val, test = split(pairs, args.val, args.test, args.seed)
    print(f"Split -> train: {len(train)}  val: {len(val)}  test: {len(test)}")

    print(f"Writing dataset to {out_dir} ...")
    copy_pairs(train, out_dir / "images/train", out_dir / "labels/train")
    copy_pairs(val,   out_dir / "images/val",   out_dir / "labels/val")
    copy_pairs(test,  out_dir / "images/test",  out_dir / "labels/test")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": 1,
        "names": {0: "runway"},
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone.")
    print(f"  train: {len(train)} images")
    print(f"  val:   {len(val)} images")
    print(f"  test:  {len(test)} images")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
