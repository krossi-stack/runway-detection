"""
Download DEEL-AI/LARD_V2 from HuggingFace and convert to YOLO segmentation format.

Each image's 4-corner runway annotation (TR, TL, BL, BR) is treated as a
quadrilateral polygon mask -- exactly what YOLO-seg needs.

Output layout:
  <out_dir>/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    data.yaml

Usage:
  python scripts/prepare_lard.py
  python scripts/prepare_lard.py --out data/datasets/lard --val-split 0.1
  python scripts/prepare_lard.py --subsets xplane flightsim
"""

import argparse
import sys
from pathlib import Path
from math import floor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUBSETS = ["arcgis", "bingmaps", "flsim", "ges", "xplane"]


def has_valid_corners(row: dict) -> bool:
    keys = ["x_TR", "y_TR", "x_TL", "y_TL", "x_BL", "y_BL", "x_BR", "y_BR"]
    return all(row.get(k) is not None for k in keys)


def corners_to_yolo_seg(row: dict, img_w: int, img_h: int) -> str:
    """Return YOLO seg label line: '0 x1 y1 x2 y2 x3 y3 x4 y4' (normalised)."""
    pts = [
        (row["x_TL"], row["y_TL"]),
        (row["x_TR"], row["y_TR"]),
        (row["x_BR"], row["y_BR"]),
        (row["x_BL"], row["y_BL"]),
    ]
    coords = []
    for x, y in pts:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return "0 " + " ".join(coords)


def save_sample(image, label_line: str, stem: str, img_dir: Path, lbl_dir: Path):
    img_path = img_dir / f"{stem}.jpg"
    lbl_path = lbl_dir / f"{stem}.txt"
    image.save(img_path, quality=95)
    lbl_path.write_text(label_line)


def process_subset(subset: str, out_dir: Path, val_split: float):
    from datasets import load_dataset

    # Skip if already converted (check for at least one image in train)
    if any((out_dir / "images" / "train").glob(f"{subset}_*.jpg")):
        print(f"\n[{subset}] Already converted, skipping.")
        return

    print(f"\n[{subset}] Loading...")
    ds = load_dataset("DEEL-AI/LARD_V2", name=subset)

    # Collect all rows (train + test splits from HF) then re-split ourselves
    rows = []
    for split_name in ds:
        for row in ds[split_name]:
            rows.append((split_name, row))

    total = len(rows)
    print(f"[{subset}] {total} rows total")

    # Determine split boundaries: use HF test split as our test set,
    # carve val_split off the top of HF train, rest is train.
    hf_train = [(i, r) for i, (s, r) in enumerate(rows) if s == "train"]
    hf_test  = [(i, r) for i, (s, r) in enumerate(rows) if s == "test"]

    n_val = floor(len(hf_train) * val_split)
    val_rows   = [r for _, r in hf_train[:n_val]]
    train_rows = [r for _, r in hf_train[n_val:]]
    test_rows  = [r for _, r in hf_test]

    split_map = [
        ("train", train_rows),
        ("val",   val_rows),
        ("test",  test_rows),
    ]

    skipped = 0
    for split_name, split_rows in split_map:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(split_rows):
            if not has_valid_corners(row):
                skipped += 1
                continue

            image = row["image"]
            img_w, img_h = image.size
            label_line = corners_to_yolo_seg(row, img_w, img_h)
            stem = f"{subset}_{split_name}_{idx:06d}"
            save_sample(image, label_line, stem, img_dir, lbl_dir)

        print(f"[{subset}] {split_name}: {len(split_rows) - (skipped if split_name == 'train' else 0)} saved")

    if skipped:
        print(f"[{subset}] Skipped {skipped} rows with missing corner annotations")


def write_yaml(out_dir: Path):
    yaml_path = out_dir / "data.yaml"
    content = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: runway\n"
    )
    yaml_path.write_text(content)
    print(f"\nWrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LARD_V2 for YOLO-seg training")
    parser.add_argument("--out", default="data/datasets/lard", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of HF train split to use as val (default 0.1)")
    parser.add_argument("--subsets", nargs="+", default=SUBSETS,
                        choices=SUBSETS, metavar="SUBSET",
                        help=f"Which LARD subsets to include. Choices: {SUBSETS} (arcgis, bingmaps, flsim, ges, xplane)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir.resolve()}")
    print(f"Subsets: {args.subsets}")
    print(f"Val split: {args.val_split:.0%} of each subset's train rows")

    for subset in args.subsets:
        process_subset(subset, out_dir, args.val_split)

    write_yaml(out_dir)
    print("\nDone. Run training with:")
    print(f"  python scripts/train.py --dataset {out_dir}/data.yaml --model yolo11n-seg.pt")


if __name__ == "__main__":
    main()
