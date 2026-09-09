"""
config.py — Project ki saari default settings ek jagah.

Yahan values change kar ke aap poore app ka behaviour badal sakte hain,
bina detection code ko chhue. CLI flags (main.py) in defaults ko override karte hain.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Saved screenshots / videos / CSV logs yahan jaate hain
OUTPUT_DIR = BASE_DIR / "outputs"

# Downloaded model weights yahan cache hote hain
MODELS_DIR = BASE_DIR / "models"

# Saved camera profiles (naam -> stream URL). Ye file git mein NAHI jaati
# kyunke is mein CCTV ke passwords hote hain — `cameras.example.json` template hai.
CAMERAS_FILE = BASE_DIR / "cameras.json"


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

# yolov8n = nano (sabse fast, CPU friendly). Accuracy chahiye to:
# "yolov8s.pt" (small) ya "yolov8m.pt" (medium). Pehli baar run par
# Ultralytics ye weights khud download kar leta hai.
MODEL_NAME = "yolov8n.pt"

# "cpu", "cuda", "mps" — ya None rakhein to auto-detect hoga
DEVICE = None

# Model ko is size par inference karta hai (chhota = tez, bara = accurate)
IMAGE_SIZE = 640


# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------

# Is se kam confidence wali detections discard ho jaayengi
CONFIDENCE_THRESHOLD = 0.5

# Non-Max Suppression IoU threshold (overlapping boxes hatane ke liye)
IOU_THRESHOLD = 0.45

# Sirf in classes ko detect karein, e.g. ["person", "car"].
# Empty list ya None = saari 80 COCO classes.
CLASS_FILTER: list[str] = []

# Class groups — "animal" naam ki koi ek COCO class nahi hoti, isliye aam
# groups ke shortcuts bana diye hain. `--classes animals` likhein to ye
# saari classes khud khul jaati hain.
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

# Ek frame mein max kitni detections rakhni hain
MAX_DETECTIONS = 300


# ---------------------------------------------------------------------------
# Input settings
# ---------------------------------------------------------------------------

# Default webcam index (0 = system ka pehla camera)
CAMERA_INDEX = 0

# Webcam se ye resolution maangte hain (camera support kare to milegi)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Network stream (phone camera / CCTV / RTSP) settings
# ---------------------------------------------------------------------------

# Stream toot jaye to kitni baar dobara connect karne ki koshish karein
STREAM_RECONNECT_ATTEMPTS = 5

# Har koshish ke darmiyan kitne second rukein
STREAM_RECONNECT_DELAY = 2.0

# Network streams par buffer chhota rakhte hain warna latency barhti jaati hai
# (frames queue mein jama ho jaate hain aur video "peechhe" reh jaati hai)
STREAM_BUFFER_SIZE = 1


# ---------------------------------------------------------------------------
# Tracking settings
# ---------------------------------------------------------------------------

# "bytetrack.yaml" — tez, halka (default)
# "botsort.yaml"  — thoda slow lekin behtar re-identification (target chhup ke
#                    wapas aaye to usay pehchan leta hai)
TRACKER = "bytetrack.yaml"

# Locked target ka raasta (trail) kitne frames tak dikhana hai
TRAIL_LENGTH = 64

# Target itne frames tak na dikhe to "LOST" dikhana shuru karo
TARGET_LOST_AFTER = 30


# ---------------------------------------------------------------------------
# UI / drawing settings
# ---------------------------------------------------------------------------

WINDOW_NAME = "YOLO Real-Time Detection"

# FPS ka moving average kitne frames par nikalna hai
FPS_SMOOTHING_WINDOW = 30

# Overlay panel (FPS + object counts) dikhana hai ya nahi
SHOW_OVERLAY = True

# Box ke upar confidence % likhna hai ya nahi
SHOW_CONFIDENCE = True

# Box ki line ki motai (pixels)
BOX_THICKNESS = 2

FONT_SCALE = 0.55


# ---------------------------------------------------------------------------
# Output settings
# ---------------------------------------------------------------------------

# Detections ka CSV log banayein? (timestamp, object, confidence)
ENABLE_CSV_LOG = False

# Processed video save karne ka codec (mp4 ke liye "mp4v" sabse portable hai)
VIDEO_CODEC = "mp4v"

# Video file process karte waqt output FPS (None = source video ka FPS)
OUTPUT_FPS = None


# ---------------------------------------------------------------------------
# Keyboard shortcuts (OpenCV window mein)
# ---------------------------------------------------------------------------

KEY_QUIT = ("q", 27)        # 'q' ya Escape
KEY_SCREENSHOT = ("s",)     # frame save karo
KEY_PAUSE = ("p", " ")      # pause / resume
KEY_TOGGLE_OVERLAY = ("o",)
KEY_CONF_UP = ("+", "=")    # confidence threshold barhao
KEY_CONF_DOWN = ("-", "_")  # confidence threshold ghatao
KEY_TOGGLE_TRACK = ("t",)   # tracking on/off
KEY_LOCK_FIRST = ("l",)     # pehle/sabse aham object par lock karo
KEY_CLEAR_LOCK = ("c",)     # lock hatao
