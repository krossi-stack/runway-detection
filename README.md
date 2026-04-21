# Runway Detection

Real-time runway edge segmentation for aircraft approach and landing, using a
Jetson Nano (8 GB) and Marshall CV574 camera.

## Goals

1. **Runway segmentation** -- detect runway edges in real time via YOLOv8-seg.
2. **Altitude estimation** -- derive altitude from runway geometry + camera
   intrinsics, validated against X-Plane telemetry.
3. **Centerline alignment** -- infer lateral offset from runway outline geometry.

Long-term target: low-latency output suitable for autonomous landing guidance.

## Hardware

| Component | Detail |
|-----------|--------|
| Compute   | NVIDIA Jetson Nano 8 GB |
| Camera    | Marshall CV574 (RTSP, 4K) |
| Mount     | Cockpit, forward-facing |

## Setup

### Prerequisites

- Python 3.12+
- X-Plane (any version with UDP data output support)

### Install

```bash
cd "C:/Users/CarsonKurtz-Rossi/OneDrive - Beta Technologies/Documents/Claude Projects/runway_detection"

python -m venv venv

source venv/Scripts/activate

pip install opencv-python numpy python-dotenv mss keyboard
```

### X-Plane UDP Configuration

The capture script records flight telemetry alongside screen frames. X-Plane
needs to be configured to broadcast this data over UDP.

1. In X-Plane, go to **Settings > Data Output**
2. Check the **"Send network data"** column for these indices:
   - **Index 3** -- speeds (indicated airspeed, true airspeed, etc.)
   - **Index 17** -- pitch, roll, heading (true)
   - **Index 20** -- latitude, longitude, altitude (MSL and AGL)
3. Set the destination IP to `127.0.0.1` and port to `49003`

## Usage

### Collecting Training Data

The capture script grabs screen frames from X-Plane at 10 FPS and records
synchronized telemetry (altitude, position, attitude) to a CSV file.

```bash
python scripts/xplane_capture.py
```

The script starts **paused**. Use these hotkeys to control it while flying:

| Key  | Action |
|------|--------|
| F9   | Start / resume capture |
| F10  | Pause capture |
| F12  | Stop and save session |

**Workflow:** Launch the script, switch to X-Plane, fly your approach, and
press F9 when a runway is in view. Press F10 to pause when you lose sight of
the runway or go around. Press F9 again on the next approach. Press F12 when
you're done.

Each session saves to `data/raw_frames/<timestamp>/`:

```
data/raw_frames/20260421_143000/
  frames/
    000000.jpg
    000001.jpg
    ...
  telemetry.csv    # one row per frame: lat, lon, alt, pitch, roll, heading, speed
  metadata.json    # session info (fps, frame count, etc.)
```

### Labeling

Upload frames from `data/raw_frames/<session>/frames/` to
[Roboflow](https://roboflow.com) for auto-labeling and review. Export in YOLO
segmentation format to `data/datasets/`.

### Training

```bash
python scripts/train.py --epochs 100 --batch 8
```

### Inference (Jetson)

```bash
python scripts/jetson_infer.py
```

## Training Data Pipeline

Training data is collected from **X-Plane screen captures** with synchronized
telemetry via X-Plane's UDP data output. Labeling is done in Roboflow using
auto-label with manual review.

Domain-gap mitigation: augmentations (lens distortion, noise, color jitter,
blur) simulate real camera behavior. A small camera-filmed validation set
confirms transfer quality.

## Project Structure

```
runway_detection/
  config/          -- configuration files
  data/
    raw_frames/    -- captured frames from X-Plane
    labels/        -- segmentation masks / YOLO label files
    datasets/      -- assembled train/val/test splits
  models/          -- trained model weights
  scripts/         -- data collection, labeling, training utilities
  logs/            -- training and inference logs
```

## Phases

1. **Data collection** -- X-Plane screen capture + UDP telemetry recorder
2. **Labeling** -- Roboflow auto-label + manual review, export YOLO format
3. **Model training** -- YOLOv8-seg (nano) fine-tuned on runway data
4. **Jetson inference** -- real-time segmentation pipeline with TensorRT
5. **Altitude & alignment** -- geometry-based estimation from segmentation output
