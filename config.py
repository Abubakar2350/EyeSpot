"""
config.py — All default settings in one place.

Change the values here to alter the application's behaviour without touching
any detection code. CLI flags (see main.py) override these defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Saved screenshots, videos and CSV logs go here
OUTPUT_DIR = BASE_DIR / "outputs"

# Downloaded model weights are cached here
MODELS_DIR = BASE_DIR / "models"

# Saved camera profiles (name -> stream URL). This file is NOT committed to git
# because it holds CCTV passwords; `cameras.example.json` is the template.
CAMERAS_FILE = BASE_DIR / "cameras.json"


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

# yolov8n = nano (fastest, CPU friendly). For better accuracy use
# "yolov8s.pt" (small) or "yolov8m.pt" (medium). Ultralytics downloads the
# weights automatically on first run.
MODEL_NAME = "yolov8n.pt"

# "cpu", "cuda", "mps" — or None to auto-detect
DEVICE = None

# Inference resolution (smaller = faster, larger = more accurate)
IMAGE_SIZE = 640


# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------

# Detections below this confidence are discarded
CONFIDENCE_THRESHOLD = 0.5

# Non-Max Suppression IoU threshold (removes overlapping boxes)
IOU_THRESHOLD = 0.45

# Detect only these classes, e.g. ["person", "car"].
# An empty list or None means all 80 COCO classes.
CLASS_FILTER: list[str] = []

# Class groups — COCO has no single "animal" class, so these shortcuts expand
# to the relevant classes. `--classes animals` pulls in the whole group.
CLASS_GROUPS: dict[str, list[str]] = {
    "animals": [
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
    ],
    "pets": ["cat", "dog", "bird"],
    "people": ["person"],
    "vehicles": [
        "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat",
    ],
    "road": ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"],
}

# Maximum detections to keep in a single frame
MAX_DETECTIONS = 300


# ---------------------------------------------------------------------------
# Input settings
# ---------------------------------------------------------------------------

# Default webcam index (0 = the system's first camera)
CAMERA_INDEX = 0

# Resolution requested from the webcam (applied only if the camera supports it)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Network stream (phone camera / CCTV / RTSP) settings
# ---------------------------------------------------------------------------

# How many times to retry when a stream drops
STREAM_RECONNECT_ATTEMPTS = 5

# Seconds to wait between reconnect attempts
STREAM_RECONNECT_DELAY = 2.0

# Keep the buffer small on network streams, otherwise latency grows steadily
# (frames pile up in the queue and the video falls behind real time)
STREAM_BUFFER_SIZE = 1


# ---------------------------------------------------------------------------
# Tracking settings
# ---------------------------------------------------------------------------

# "bytetrack.yaml" — fast and lightweight (default)
# "botsort.yaml"  — slower but better at re-identification (recognises a target
#                    that disappeared behind something and came back)
TRACKER = "bytetrack.yaml"

# How many frames of the locked target's path (trail) to draw
TRAIL_LENGTH = 64

# Show "LOST" once the target has been missing for this many frames
TARGET_LOST_AFTER = 30


# ---------------------------------------------------------------------------
# UI / drawing settings
# ---------------------------------------------------------------------------

WINDOW_NAME = "EyeSpot — Real-Time Detection"

# Number of frames used for the FPS moving average
FPS_SMOOTHING_WINDOW = 30

# Show the overlay panel (FPS + object counts)
SHOW_OVERLAY = True

# Print the confidence percentage above each box
SHOW_CONFIDENCE = True

# Bounding box line thickness in pixels
BOX_THICKNESS = 2

FONT_SCALE = 0.55


# ---------------------------------------------------------------------------
# Output settings
# ---------------------------------------------------------------------------

# Write a CSV log of detections? (timestamp, object, confidence)
ENABLE_CSV_LOG = False

# Codec used when saving processed video ("mp4v" is the most portable for mp4)
VIDEO_CODEC = "mp4v"

# Output FPS when processing a video file (None = use the source video's FPS)
OUTPUT_FPS = None


# ---------------------------------------------------------------------------
# Keyboard shortcuts (in the OpenCV window)
# ---------------------------------------------------------------------------

KEY_QUIT = ("q", 27)        # 'q' or Escape
KEY_SCREENSHOT = ("s",)     # save the current frame
KEY_PAUSE = ("p", " ")      # pause / resume
KEY_TOGGLE_OVERLAY = ("o",)
KEY_CONF_UP = ("+", "=")    # raise the confidence threshold
KEY_CONF_DOWN = ("-", "_")  # lower the confidence threshold
KEY_TOGGLE_TRACK = ("t",)   # tracking on/off
KEY_LOCK_FIRST = ("l",)     # lock onto the first/most prominent object
KEY_CLEAR_LOCK = ("c",)     # clear the lock
