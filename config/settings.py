import os
from dotenv import load_dotenv

load_dotenv()

# --- Camera ---
RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.1.100/stream1")
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# Camera intrinsics (Marshall CV574 defaults -- update after calibration)
FOCAL_LENGTH_MM = float(os.getenv("FOCAL_LENGTH_MM", "4.0"))
SENSOR_WIDTH_MM = float(os.getenv("SENSOR_WIDTH_MM", "6.17"))
SENSOR_HEIGHT_MM = float(os.getenv("SENSOR_HEIGHT_MM", "4.55"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1920"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1080"))

# --- X-Plane Data Collection ---
XPLANE_UDP_IP = os.getenv("XPLANE_UDP_IP", "0.0.0.0")
XPLANE_UDP_PORT = int(os.getenv("XPLANE_UDP_PORT", "49003"))
XPLANE_SEND_PORT = int(os.getenv("XPLANE_SEND_PORT", "49000"))  # X-Plane's listening port for RREF
XPLANE_SCREEN_REGION = os.getenv("XPLANE_SCREEN_REGION", "")  # x,y,w,h or empty for full screen
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "10"))
XPLANE_FOV_DEG = float(os.getenv("XPLANE_FOV_DEG", "80.0"))  # fallback if RREF read fails

# --- X-Plane Paths ---
XPLANE_APT_DAT = os.getenv("XPLANE_APT_DAT", "")  # path to X-Plane's apt.dat file

# --- Detection / Segmentation ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/runway_seg.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "640"))

# --- Runway Reference ---
RUNWAY_WIDTH_M = float(os.getenv("RUNWAY_WIDTH_M", "45.0"))  # standard runway width in meters
RUNWAY_LENGTH_M = float(os.getenv("RUNWAY_LENGTH_M", "2500.0"))

# --- Paths ---
RAW_FRAMES_DIR = os.getenv("RAW_FRAMES_DIR", "data/raw_frames")
LABELS_DIR = os.getenv("LABELS_DIR", "data/labels")
DATASETS_DIR = os.getenv("DATASETS_DIR", "data/datasets")
