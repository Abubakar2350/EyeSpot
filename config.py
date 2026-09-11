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
# Accident detection settings
# ---------------------------------------------------------------------------
#
# Speeds and accelerations below are in **object lengths per second**, not
# pixels. Dividing by an object's own size cancels out camera distance and
# resolution, so one set of thresholds works on a dashcam, a CCTV pole and a
# phone at the roadside alike.

# Detection confidence used while accident detection is on.
#
# Lower than the normal default on purpose, and this is the single setting that
# decides whether the feature works at all. Everything here is derived from how
# an object moves between frames, which requires it to keep the same track ID.
# At the usual 0.5 threshold, vehicles in ordinary CCTV footage are detected
# marginally — around 0.5 confidence — so they flicker in and out, the tracker
# issues a fresh ID each time, and the motion history resets.
#
# Measured on real CCTV accident footage (640x360, 2298 frames):
#   confidence 0.50 -> 0 events; the score cleared the threshold on 2 isolated frames
#   confidence 0.35 -> 3 events, correctly placed at the collisions
#
# The fusion rules are what guard against false positives, so a lower detection
# threshold costs little here and buys the continuity the whole method needs.
ACCIDENT_CONFIDENCE = 0.35

# Fused score at which an event is considered. Lower catches more and cries
# wolf more; higher is quieter but misses marginal cases.
ACCIDENT_THRESHOLD = 0.55

# The score must clear the threshold on this many of the last
# ACCIDENT_PERSISTENCE_WINDOW frames. A sliding window rather than a
# consecutive run: real footage drops a detection for a frame at a time, and
# demanding an unbroken streak discarded evidence that was plainly there.
ACCIDENT_PERSISTENCE = 3
ACCIDENT_PERSISTENCE_WINDOW = 8

# Seconds to stay silent after an event. One crash produces many chaotic
# frames; without this a single accident would raise dozens of events.
ACCIDENT_COOLDOWN = 5.0

# How much each channel contributes to the fused score. They sum above 1.0 on
# purpose: no single channel can reach the threshold alone, but any two strong
# ones together will. That is the whole point of fusing.
ACCIDENT_WEIGHTS = {
    "impact": 0.45,       # boxes overlapping while closing — the clearest sign
    "sudden_stop": 0.25,  # moving one moment, stationary the next
    "person_down": 0.25,  # a pedestrian's box gone from tall to wide
    "hard_brake": 0.20,   # deceleration beyond normal traffic
    "swerve": 0.20,       # heading changing faster than steering explains
    "ttc": 0.15,          # a converging pair about to meet
}

# Deceleration beyond this counts as a hard brake (lengths/s²)
ACCIDENT_BRAKE_DECEL = 1.2

# Heading change faster than this counts as a swerve (radians/s)
ACCIDENT_YAW_RATE = 2.0

# Box overlap above this is treated as a possible impact. Kept low on purpose:
# vehicles that collide rarely overlap much in the image — they meet edge to
# edge, so their boxes graze rather than merge. Demanding a large overlap would
# miss most real impacts. The corroborating checks (were they closing, did one
# take a jolt) are what stop this from firing on every passing car.
ACCIDENT_IMPACT_IOU = 0.10

# Time-to-collision below this many seconds raises the ttc channel
ACCIDENT_TTC_SECONDS = 0.8

# An object slower than this is treated as stationary (lengths/s)
ACCIDENT_MOVING_SPEED = 0.5

# A person's box wider-than-tall by this ratio suggests they are on the ground
ACCIDENT_PERSON_DOWN_RATIO = 1.1

# ...and must stay that way this long, so crouching is not mistaken for falling
ACCIDENT_PERSON_DOWN_SECONDS = 1.0


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
