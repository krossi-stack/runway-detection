# Runway Detection Project

Real-time runway segmentation for aircraft approach/landing using Jetson Nano 8GB + Marshall CV574.

## Architecture

- `config/settings.py` -- all configuration via environment variables with defaults
- `scripts/xplane_capture.py` -- screen capture + X-Plane UDP telemetry recorder
- `scripts/prepare_lard.py` -- download DEEL-AI/LARD_V2 from HuggingFace and convert to YOLO-seg format
- `scripts/train.py` -- YOLOv11-seg training wrapper
- `scripts/jetson_infer.py` -- real-time RTSP inference pipeline for Jetson (auto-prefers .engine over .pt)
- `scripts/export_tensorrt.py` -- export .pt to TensorRT engine (run on Jetson)

## Training Data

- DEEL-AI/LARD_V2 (~128k images across ArcGIS, Bing Maps, FlightSim, Google Earth Studio, X-Plane)
- 4-corner runway annotations converted to YOLO-seg quadrilateral polygons by prepare_lard.py
- Screen capture from X-Plane (not camera-filmed monitor)
- X-Plane UDP telemetry synced via timestamps for altitude ground truth
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
# Download + convert LARD_V2 dataset (~128k images, takes a while)
python scripts/prepare_lard.py --out data/datasets/lard

# Collect training data from X-Plane
python scripts/xplane_capture.py

# Train model
python scripts/train.py --dataset data/datasets/lard/data.yaml --epochs 100 --batch 8

# Export to TensorRT (run ON Jetson, takes 10-30 min)
python scripts/export_tensorrt.py --imgsz 320 --fp16

# Run inference on Jetson (auto-loads .engine if present)
python scripts/jetson_infer.py
```
