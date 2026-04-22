# Runway Detection Project

Real-time runway segmentation for aircraft approach/landing using Jetson Nano 8GB + Marshall CV574.

## Architecture

- `config/settings.py` -- all configuration via environment variables with defaults
- `scripts/xplane_capture.py` -- screen capture + X-Plane UDP telemetry recorder
- `scripts/train.py` -- YOLOv8-seg training wrapper
- `scripts/jetson_infer.py` -- real-time RTSP inference pipeline for Jetson (auto-prefers .engine over .pt)
- `scripts/export_tensorrt.py` -- export .pt to TensorRT engine (run on Jetson)

## Training Data

- Screen capture from X-Plane (not camera-filmed monitor)
- X-Plane UDP telemetry synced via timestamps for altitude ground truth
- Auto-labeling from X-Plane runway geometry (to be implemented)
- Domain gap bridged with augmentations; validated with camera-filmed set

## Key Decisions

- YOLOv8-seg (nano) for segmentation -- balances speed and accuracy on Jetson
- Segmentation approach (not bounding box) to get runway edges for geometry
- Altitude derived from runway geometry + camera intrinsics, not from the model
- Centerline alignment inferred from runway outline geometry
- Latency is critical -- this will eventually support autonomous landings

## Hardware

- Jetson Nano 8GB (JetPack, PyTorch pre-installed)
- Marshall CV574 (RTSP, cockpit-mounted, forward-facing)

## Commands

```bash
# Collect training data from X-Plane
python scripts/xplane_capture.py

# Train model
python scripts/train.py --epochs 100 --batch 8

# Export to TensorRT (run ON Jetson, takes 10-30 min)
python scripts/export_tensorrt.py --imgsz 320 --fp16

# Run inference on Jetson (auto-loads .engine if present)
python scripts/jetson_infer.py
```
