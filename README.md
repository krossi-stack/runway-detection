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

## Training Data Pipeline

Training data is collected from **X-Plane screen captures** with synchronized
telemetry (altitude, position, heading) via X-Plane's UDP data output.

Auto-labeling leverages X-Plane's known runway geometry to generate
segmentation masks, minimizing manual annotation.

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
2. **Auto-labeling** -- generate runway segmentation masks from telemetry
3. **Model training** -- YOLOv8-seg (nano) fine-tuned on runway data
4. **Jetson inference** -- real-time segmentation pipeline with TensorRT
5. **Altitude & alignment** -- geometry-based estimation from segmentation output
