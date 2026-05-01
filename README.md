# Runway Detection

Real-time runway segmentation for aircraft approach and landing, using a Jetson Nano 8GB and Marshall CV574 camera.

**Goals:**
- Detect and outline the runway in real time using YOLOv11-seg
- Derive altitude and centerline offset from runway geometry + camera intrinsics
- Long-term: low-latency output suitable for autonomous landing guidance

## Hardware

| Component | Detail |
|-----------|--------|
| Compute   | NVIDIA Jetson Nano 8GB (JetPack, TensorRT) |
| Camera    | Marshall CV574 (RTSP, 4K, cockpit-mounted, forward-facing) |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python numpy python-dotenv mss keyboard pyyaml
```

Copy `.env.example` to `.env` and fill in your paths:

```
XPLANE_APT_DAT=C:/X-Plane 12/Global Scenery/Global Airports/Earth nav data/apt.dat
XPLANE_FOV_DEG=65.0
XPLANE_FOV_V_DEG=28.45
IMAGE_WIDTH=3440
IMAGE_HEIGHT=1440
```

### X-Plane UDP Setup

In X-Plane go to **Settings > Data Output**, enable **Send network data** for:
- Index 3 — speeds
- Index 17 — pitch, roll, heading
- Index 20 — latitude, longitude, altitude

Set destination IP to your PC's IP and port to `5501`.

---

## Scripts

### `scripts/xplane_capture.py` — Capture training data from X-Plane

Records synchronized screen frames and telemetry (lat, lon, alt, pitch, roll, heading) from a live X-Plane session.

```bash
python scripts/xplane_capture.py
```

Controls while flying:
- `1` — start / resume capture
- `2` — pause
- `3` — stop and save session

Output saved to `data/raw_frames/<timestamp>/frames/` + `telemetry.csv`.

---

### `scripts/auto_label.py` — Generate labels from telemetry

Projects the runway's 4 corners into each captured frame using aircraft pose and camera FOV, then saves YOLO segmentation labels.

```bash
# Using apt.dat to look up runway coordinates
python scripts/auto_label.py --session 20260424_125550 --airport KPBG --runway 17 \
    --pitch-bias 0.3 --acf "C:/X-Plane 12/Aircraft/Alia 250/ALIA_ctol.acf" --preview

# Using direct coordinates
python scripts/auto_label.py --session 20260424_125550 \
    --runway-coords 44.665833,-73.476771,44.636042,-73.459462
```

Key options:
- `--pitch-bias` — degrees to add to pitch to correct camera mounting angle (tune per aircraft)
- `--roll-bias` — degrees to add to roll to correct camera mounting angle
- `--acf` — path to aircraft `.acf` file for pilot eye position offset
- `--preview` — show labeled frames after generating labels

**Calibrated values for Alia 250 CTOL at KPBG:** `--pitch-bias 0.3 --roll-bias 0`

---

### `scripts/review_session.py` — Review and edit auto-generated labels

Interactive frame-by-frame reviewer. Drag vertices to correct polygon positions, accept or deny each frame.

```bash
python scripts/review_session.py --session 20260424_125550
python scripts/review_session.py --session 20260424_125550 --n 50
```

Controls:
- `a` / `Enter` — accept (saves polygon, with any edits)
- `d` / `Delete` — deny (deletes label file)
- `Space` — skip (no changes)
- `r` — reset polygon to original
- Arrow keys — nudge entire polygon 1px at a time
- `q` — quit

---

### `scripts/build_xplane_dataset.py` — Assemble labeled sessions into a dataset

Collects all labeled frames from `data/labels/` and `data/raw_frames/`, splits into train/val/test, and writes a YOLO-format dataset to `data/datasets/xplane/`.

```bash
python scripts/build_xplane_dataset.py
python scripts/build_xplane_dataset.py --val 0.15 --test 0.05
```

Output: `data/datasets/xplane/data.yaml` ready for training.

---

### `scripts/prepare_lard.py` — Download and convert LARD_V2 dataset

Downloads DEEL-AI/LARD_V2 (~128k images) from HuggingFace and converts 4-corner annotations to YOLO-seg format. One-time setup — dataset is already downloaded.

```bash
python scripts/prepare_lard.py --out data/datasets/lard
```

---

### `scripts/train.py` — Train the segmentation model

Wraps YOLOv11-seg training. Fine-tune from existing weights or train from scratch.

```bash
# Fine-tune LARD weights on X-Plane data (recommended)
python scripts/train.py --dataset data/datasets/xplane/data.yaml \
    --model models/runway_seg_lard.pt --epochs 50 --batch 8 --name runway_seg_xplane_v1

# After training, copy best weights
cp runs/segment/runway_seg_xplane_v1/weights/best.pt models/runway_seg_xplane_v1.pt
```

---

### `scripts/export_tensorrt.py` — Export to TensorRT for Jetson

Run this **on the Jetson** to convert a `.pt` model to a TensorRT `.engine` file. Takes 10–30 minutes but gives a large inference speedup.

```bash
python scripts/export_tensorrt.py --model models/runway_seg_xplane_v1.pt --imgsz 320 --fp16
```

---

### `scripts/jetson_infer.py` — Real-time inference on Jetson (RTSP camera)

Reads from the Marshall CV574 RTSP stream, runs segmentation, and draws the runway outline. Auto-loads `.engine` over `.pt` if available.

```bash
python scripts/jetson_infer.py
python scripts/jetson_infer.py --model models/runway_seg_xplane_v1.pt
```

---

### `scripts/xplane_overlay.py` — Live transparent overlay on X-Plane (Windows)

Runs inference on the X-Plane window in real time and draws a trapezoid overlay directly on top of the sim. Auto-detects the X-Plane window. Includes temporal smoothing and confidence hysteresis to reduce flicker.

```bash
python scripts/xplane_overlay.py --model models/runway_seg_xplane_v1.pt
python scripts/xplane_overlay.py --model models/runway_seg_xplane_v1.pt --fps 10 --smooth 0.35 --conf 0.3
```

Key options:
- `--fps` — inference rate (default: 10). Lower if GPU is struggling.
- `--smooth` — EMA smoothing weight 0–1 (default: 0.35). Lower = smoother but slower to track.
- `--conf` — minimum confidence to show polygon (default: 0.3)
- `--region` — manual screen region `x,y,w,h` if auto-detection fails

Press **Ctrl+Q** to quit.

---

## Model Weights

| File | Description |
|------|-------------|
| `models/runway_seg.pt` | Original baseline weights |
| `models/runway_seg_lard.pt` | Trained on DEEL-AI/LARD_V2 (128k images, mAP50=0.969) |
| `models/runway_seg_xplane_v1.pt` | Fine-tuned on X-Plane cockpit approach captures |

## Typical Workflow

1. Fly approaches in X-Plane, record with `xplane_capture.py`
2. Generate labels with `auto_label.py`
3. Review and correct labels with `review_session.py`
4. Build dataset with `build_xplane_dataset.py`
5. Fine-tune model with `train.py`
6. Copy best weights to `models/`
7. Test live with `xplane_overlay.py`
8. Deploy to Jetson: export with `export_tensorrt.py`, run with `jetson_infer.py`

## Project Structure

```
runway-detection/
  config/
    settings.py        -- all config via environment variables
  data/
    raw_frames/        -- captured X-Plane sessions (frames + telemetry)
    labels/            -- auto-generated YOLO label files per session
    datasets/
      lard/            -- DEEL-AI/LARD_V2 converted dataset
      xplane/          -- assembled X-Plane capture dataset
  models/              -- trained model weights (.pt, .engine)
  scripts/             -- all scripts (see above)
  runs/                -- training run outputs (weights, metrics, plots)
```
