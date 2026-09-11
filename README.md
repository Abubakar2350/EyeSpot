# 🎯 EyeSpot — Real-Time Object Detection & Tracking

Real-time object detection built on **YOLOv8 (Ultralytics)** and **OpenCV**.
Webcam, phone camera, CCTV, video file or a single image — all of them get live
detection with bounding boxes, labels, confidence scores and an FPS counter.

All 80 COCO classes are supported: `person`, `car`, `dog`, `cat`, `cell phone`,
`laptop`, `bottle`, `chair`, and so on.

> 📋 For the full project report, setup steps for another machine, and a
> complete answer to "what can it actually track?", see **[FINAL_REPORT.md](FINAL_REPORT.md)**
>
> 📹 For connecting a home CCTV camera, whether that setup persists, and what a
> multi-user version with Google sign-in would require, see
> **[USER_REPORT_HOME_CCTV.md](USER_REPORT_HOME_CCTV.md)**
>
> 🗺️ For what could be built next, sized and prioritised, see
> **[NEXT_PHASES.md](NEXT_PHASES.md)**

---

## ✨ Features

| Feature | Status |
|---|---|
| Real-time webcam detection | ✅ |
| **Phone camera / CCTV / IP camera (RTSP, HTTP)** | ✅ |
| **Saved camera profiles — `--camera gate`, no need to retype URLs** | ✅ |
| Video file detection (.mp4 / .avi / .mov / .mkv) | ✅ |
| Single image detection | ✅ |
| Bounding box + label + confidence % | ✅ |
| Live FPS counter and per-class object counts | ✅ |
| Confidence threshold adjustable at runtime (`+` / `-`) | ✅ |
| Class filter, including group shortcuts (`animals`, `vehicles`) | ✅ |
| **Object tracking — persistent IDs (ByteTrack / BoT-SORT)** | ✅ |
| **Target lock — click an object and follow it** | ✅ |
| **Road accident detection — kinematics + multi-signal fusion** | ✅ |
| Pause / resume / stop | ✅ |
| Screenshot capture (`s` key) | ✅ |
| Annotated video export | ✅ |
| CSV detection log (timestamp, track ID, object, confidence, box) | ✅ |
| Stream auto-reconnect when a network camera drops | ✅ |
| Streamlit web UI (cached results, video/CSV download) | ✅ |
| **Browser camera in the web UI — photo + live (WebRTC), works on phones** | ✅ |
| GPU auto-detection (CUDA / Apple MPS) | ✅ |

---

## 📦 Installation

### 1. Python 3.10 or newer

```bash
python3 --version
```

### 2. Create a virtual environment

```bash
cd EyeSpot
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

> On Ubuntu/Debian, if you see `ensurepip is not available`, run
> `sudo apt install python3-venv` first.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

On a CPU-only machine you can save a large download (~200 MB instead of ~2.5 GB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The model weights (`yolov8n.pt`, about 6 MB) download automatically on the first
run — all you need is an internet connection.

---

## 🚀 Usage

### Webcam (default)

```bash
python main.py
```

### Video file

```bash
python main.py --source myvideo.mp4
python main.py --source myvideo.mp4 --save        # export the annotated video
```

### Single image

```bash
python main.py --source photo.jpg
python main.py --source photo.jpg --save
```

### Detect only specific objects

```bash
python main.py --classes person                    # people only
python main.py --classes person car dog            # several classes
python main.py --list-classes                      # all groups and all 80 classes
```

**Groups** — COCO has no single "animal" class, so these shortcuts exist:

| Group | Expands to |
|---|---|
| `people` | person |
| `animals` | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| `pets` | cat, dog, bird |
| `vehicles` | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| `road` | person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign |

```bash
python main.py --classes animals                   # any animal
python main.py --classes person animals            # people and animals
```

To define your own group, add it to `CLASS_GROUPS` in `config.py`.

### Higher accuracy (slightly slower)

```bash
python main.py --model yolov8s.pt      # small
python main.py --model yolov8m.pt      # medium
```

### Detection log (CSV)

```bash
python main.py --log                   # outputs/detections_*.csv
```

### Headless / batch processing (no window)

```bash
python main.py --source video.mp4 --no-display --save --log
```

---

## 🚨 Road Accident Detection

Watches how tracked road users *move* and raises an event when the motion stops
looking like traffic. Works on video files, live cameras and CCTV streams alike.

```bash
python main.py --source traffic.mp4 --accident
python main.py --camera roadcam --accident --log --save
python main.py --source dashcam.mp4 --accident --accident-threshold 0.45
```

In the web UI, turn on **🚨 Accident detection** in the sidebar. It enables
tracking automatically, because the whole method depends on an object keeping
the same identity from frame to frame.

### The pipeline

```
Detection → Tracking → Kinematics → Signals → Fusion → Event
```

No extra model runs. The engine consumes the tracked detections that already
exist and does arithmetic on their positions, so the measured overhead is under
a millisecond per frame — within noise of running tracking alone.

### The six signals

| Signal | What it looks for |
|---|---|
| `impact` | Two road users whose boxes overlap while closing on each other |
| `sudden_stop` | Something moving one moment and stationary the next |
| `person_down` | A person's box turning from tall to wide, and staying wide |
| `hard_brake` | Deceleration far beyond normal traffic behaviour |
| `swerve` | Heading changing faster than steering can explain |
| `ttc` | Time-to-collision for a converging pair dropping below a threshold |

**No single signal fires an event.** Each one alone has an innocent explanation:
a car braking at a light, a pedestrian crouching, two vehicles overlapping only
because one is behind the other. The weights in `config.py` are set so that no
channel can reach the threshold by itself — several have to agree. That is what
separates a crash from ordinary traffic, and it is verified by a test that
staged hard braking on its own and asserts nothing fires.

### Scale normalisation

Speeds are measured in **object lengths per second**, not pixels per second. A
pixel means nothing on its own — the same car on the same road produces wildly
different pixel speeds depending on camera distance and resolution. Dividing by
the object's own diagonal cancels most of that, so one set of thresholds works
on a dashcam, a CCTV pole and a phone at the roadside.

### Output

Each event writes a row to `outputs/accidents_*.csv` — timestamp, frame, score,
every individual signal, the track IDs involved — and saves a snapshot of the
frame. Events are flushed immediately rather than buffered, so nothing is lost
if the process is killed. A cooldown stops one crash from producing dozens of
events.

### Tuning

```bash
python main.py --source clip.mp4 --accident --accident-threshold 0.45   # more sensitive
python main.py --source clip.mp4 --accident --accident-threshold 0.70   # quieter
```

Every threshold and weight lives in the accident section of `config.py`.

### What this is, and what it is not

This is a **kinematic anomaly detector**, and it is honest about that:

| Included | Not included |
|---|---|
| YOLO detection of vehicles and road users | A trained accident classifier |
| ByteTrack / BoT-SORT persistent tracking | A temporal model (GRU/LSTM) over behaviour |
| Speed, acceleration, yaw and TTC | Training or evaluation on IDD / BDD100K / CCD |
| Multi-channel fusion with debounce and cooldown | Any accuracy figure measured on a labelled benchmark |

A learned classifier trained on a labelled accident dataset would almost
certainly outperform hand-tuned thresholds, and adding one is a sensible Phase 3
item. But it needs the dataset and a training run, so it is not claimed here.
What *is* claimed is the geometry and the fusion, and those are covered by tests
that stage a collision, ordinary traffic, and isolated hard braking, and assert
which of the three fires.

---

## 📷 Camera Integration — Phone, CCTV, USB

The system does not care what the camera is; it only needs a **video stream**.
Anything that streams over RTSP or HTTP works with `--source`.

| Camera | Protocol | What the source looks like |
|---|---|---|
| Laptop / USB webcam | none — a direct device | `0`, `1`, `2` ... |
| Mobile phone | **HTTP** | `http://<phone-ip>:8080/video` |
| CCTV / IP camera | **RTSP** (some also offer HTTP/MJPEG) | `rtsp://user:pass@<ip>:554/...` |

---

### Camera settings — configure once, then use the name

Retyping the full URL every time is tedious, and it puts the password into your
shell history. Save the camera under a name instead:

```bash
# 1. Save it once (this also confirms the URL actually works)
python main.py --source http://192.168.1.5:8080/video --save-camera phone --camera-note "my Android"
python main.py --source "rtsp://admin:pass@192.168.1.10:554/stream2" --save-camera gate
python main.py --source 0 --save-camera laptop

# 2. From now on the name is enough
python main.py --camera phone
python main.py --camera gate --track --lock person
python main.py --camera laptop

# List every saved camera (passwords are masked)
python main.py --list-cameras
```

The name also works directly with `--source`: `python main.py --source gate`

To edit the file by hand, copy the template:

```bash
cp cameras.example.json cameras.json      # Windows: copy cameras.example.json cameras.json
```

```json
{
  "laptop": { "url": "0", "kind": "usb" },
  "phone":  { "url": "http://192.168.1.5:8080/video", "kind": "mobile" },
  "gate":   { "url": "rtsp://admin:pass@192.168.1.10:554/stream2", "kind": "cctv", "note": "front gate" }
}
```

> 🔒 **`cameras.json` is not committed to git** (it is in `.gitignore`) because it
> holds CCTV passwords. Only `cameras.example.json` ships with the repository —
> copy it on a new machine and fill in your own values.

**In the Streamlit UI**: the Live camera tab has a dropdown of saved cameras and
a **URL builder**. The builder first asks whether it is a **📱 mobile phone** or a
**📹 CCTV / IP camera**, then shows only the relevant list (IP Webcam / DroidCam /
Iriun for phones, brand names for CCTV). Enter the IP address and the complete
URL is built for you, with the password URL-encoded automatically.

---

### Option 1 — Phone camera (easiest, free)

Turn a phone into a wireless IP camera:

**Android — the "IP Webcam" app** (Play Store, free):
1. Install and open the app
2. Scroll down and tap **Start server**
3. The screen shows a URL, e.g. `http://192.168.1.5:8080`
4. Run detection — the trailing `/video` is required:

```bash
python main.py --source http://192.168.1.5:8080/video
```

**iPhone — "DroidCam" or "EpocCam"**: use the URL the app gives you.

| App | Default port | Path |
|---|---|---|
| IP Webcam (Android) | 8080 | `/video` |
| DroidCam | 4747 | `/video` |
| Iriun | 8080 | `/video` |

> ⚠️ The phone and the PC must be on the **same Wi-Fi** network. Mobile data will not work.

### Option 2 — CCTV / NVR / IP camera (RTSP)

Every CCTV camera has its own RTSP URL format. Common brands:

| Brand | RTSP URL format |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream1` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| ONVIF (generic) | `rtsp://user:pass@IP:554/onvif1` |

```bash
python main.py --source "rtsp://admin:mypass@192.168.1.10:554/stream1" --classes person
```

**Finding your exact URL:**
1. Get the camera's IP from your router's admin page (connected devices)
2. Look for "RTSP" or "Stream URL" in the camera's app or web settings — it is often printed there
3. Or search the web for your brand name plus "RTSP URL"

**Test it in VLC first** — this saves a lot of time: VLC → Media → Open Network
Stream → paste the URL. If it plays in VLC, it will work here.

> 💡 The `substream` variant (`subtype=1`, `stream2`, `_sub`, channel `102`) has a
> lower resolution and is **much faster** for detection. The 4K main stream is slow
> on CPU and adds nothing useful.

### Option 3 — USB webcam / laptop camera

```bash
python main.py --source 0      # first camera
python main.py --source 1      # second camera
```

On Linux, list available cameras with `ls /dev/video*`

### Which camera should you use?

| Use case | Recommendation |
|---|---|
| Learning and testing | Laptop webcam or a phone (IP Webcam) — free |
| Home security | Any RTSP-capable CCTV (Tapo C200 is inexpensive; Hikvision/Dahua) |
| Watching birds | A camera with optical zoom — birds are small and far away |
| Portable / outdoor | An old Android phone is plenty |

The one thing that matters: it must support **RTSP**. Some cheap "cloud-only"
cameras work solely through their own app and expose no local stream — those
will not work here.

### What if the stream drops?

Network cameras lose connection routinely. The system retries **five times**
automatically (`STREAM_RECONNECT_ATTEMPTS` in `config.py`) instead of crashing.

---

## 🎯 Tracking and Target Lock

### Detection vs tracking — what's the difference?

- **Detection** (default): each frame is examined independently. "There are 3
  people in this frame." Box #1 in frame 1 and box #1 in frame 2 are not
  necessarily the same person.
- **Tracking** (`--track`): every object gets a **persistent ID** that stays the
  same across frames. "This is the same person who walked in 5 seconds ago."

Tracking is **required** for target lock — without IDs the system has no way to
know which object to follow.

```bash
python main.py --track                          # IDs on — boxes show #1, #2, #3...
python main.py --track --tracker botsort.yaml   # better re-identification
```

| Tracker | When to use it |
|---|---|
| `bytetrack.yaml` (default) | Fast and lightweight. Best for most situations. |
| `botsort.yaml` | Objects that hide and reappear (occlusion) — recognises them better, slightly slower. |

### Three ways to lock a target

**1. Mouse click** (the most direct) — click the object in the video window:

```bash
python main.py --track
# ...then click whichever object you want to follow
```

**2. Auto-lock by class name** — the first object of that class:

```bash
python main.py --track --lock person
python main.py --track --lock bird
```

**3. By track ID** — run with `--track` first to see the IDs, then:

```bash
python main.py --track --lock-id 3
```

### What a lock looks like on screen

- An amber box with corner brackets around the locked target
- A crosshair at its centre
- A line from the centre of the frame to the target, with the offset
  (`L171 D47` = 171 px left, 47 px down)
- The target's trail (its recent path)
- **Every other object dimmed**
- **"TARGET LOST"** if it disappears — and automatic re-acquisition when it returns

Press `c` to clear the lock, `l` to auto-lock, `t` to toggle tracking.

### Tracking birds

`bird` is already one of the 80 COCO classes, so no training is required:

```bash
python main.py --source 0 --track --lock bird --classes bird --conf 0.3
```

Tips for birds:
- Use `--conf 0.25` to `0.35` — birds are small and get missed at the default 0.5
- `--imgsz 960` picks up small objects better (a little slower)
- `--model yolov8s.pt` or `yolov8m.pt` — much better than nano on small objects
- Use a camera with optical zoom; no model can find a bird that is 10 pixels wide
- `botsort.yaml` handles fast-moving birds better

> **Note:** the model only recognises "bird" — it cannot tell you *which* bird.
> Species-level identification needs custom training (Phase 2 in the PRD).

---

## ⌨️ Keyboard Shortcuts

While the video window has focus:

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `s` | Save a screenshot (to `outputs/`) |
| `p` / `Space` | Pause / resume |
| `o` | Toggle the overlay panel |
| `+` / `=` | Raise the confidence threshold |
| `-` | Lower the confidence threshold |
| `t` | Toggle tracking |
| **Mouse click** | Lock onto that object |
| `l` | Auto-lock (most prominent object) |
| `c` | Clear the lock |

---

## 📈 Analysing the results

### 1. Live, on screen

The overlay panel (top-left) updates FPS, the total object count, per-class
counts and the lock status in real time.

### 2. Afterwards, from the CSV log

```bash
python main.py --source cctv_footage.mp4 --track --log
```

This produces `outputs/detections_*.csv`:

| timestamp | frame | track_id | label | confidence | x1 | y1 | x2 | y2 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-08T14:19:21 | 0 | 2 | person | 0.8786 | 34 | 313 | 185 | 714 |

From this you can work out:
- **How many unique people arrived** — count the distinct `track_id` values
  (impossible without tracking, since the same person is recounted every frame)
- **How long an object stayed** — count that `track_id`'s frames ÷ FPS
- **What happened when** — filter by timestamp
- **Which route an object took** — plot the box centres for that track ID

The file opens directly in Excel or Google Sheets, or with pandas:

```python
import pandas as pd
df = pd.read_csv("outputs/detections_20260908_141920.csv")
print(df.groupby("label")["track_id"].nunique())          # unique objects per class
print(df[df.label == "person"].track_id.value_counts())   # frames per person
```

### 3. Annotated video

`--save` writes the full video with boxes and labels to `outputs/` — useful for
reports and presentations.

---

## 🌐 Web UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

This opens `http://localhost:8501` in your browser, with:
- **Image / Video / Live camera** — three tabs
- **Video tab**: once processing finishes the result is **cached**, so moving a
  slider or switching tabs does **not** reprocess the video. The same view holds
  the annotated video player, a whole-video summary of what was found, and
  **download buttons** for the video and the CSV log. Use "🔄 Process again" for a
  fresh run.
- **Live camera tab**: **the viewing device's own camera** (photo or live WebRTC —
  point a phone at something), a dropdown of saved cameras, the local webcam, or
  a stream URL — plus the URL builder and a "save this camera" form
- Confidence and IoU sliders, class filter, model selector, tracking toggle
- Download buttons for every result

> The webcam option uses the camera of the machine running the app; deploying it
> on a remote server will not give you the browser's camera.

---

## ☁️ Deployment

The web UI is deployed to **Streamlit Community Cloud**, which is free and
serves the app from this private repository — visitors get a public link, while
the source stays private.

### Deploying

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Authorise Streamlit to read this repository
3. Create the app with:

   | Field | Value |
   |---|---|
   | Repository | `<user>/EyeSpot` |
   | Branch | `main` |
   | Main file path | `ui/streamlit_app.py` |
   | Python version | 3.11 |

4. Under **Advanced settings → Secrets**, add:

   ```toml
   EYESPOT_CLOUD = "1"
   ```

That last line matters — it tells the app it is hosted, so the live camera tab
explains itself instead of failing (see below).

### Files that make hosting work

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies, with torch pinned to its `+cpu` build |
| `packages.txt` | apt packages installed before pip: `ffmpeg`, `libgl1`, `libglib2.0-0t64` |
| `.python-version` | Requests Python 3.11 |

`torch` is pinned to `+cpu` deliberately. The default PyPI wheel bundles the
CUDA runtime and pulls roughly 2.5 GB of `nvidia-*` packages that no CPU host can
use — too large for most free tiers. The `+cpu` build is 196 MB. Wheels are
published for CPython 3.10 through 3.14, so the pin holds whichever Python the
host chooses.

`ffmpeg` powers the H.264 transcode that makes result videos playable in the
browser. `libgl1` and `libglib2.0-0t64` supply `libGL.so.1` and
`libgthread-2.0.so.0`, which the full (non-headless) OpenCV build links against;
the full build is kept because `main.py` opens a real window when run locally.

Two traps worth knowing about `packages.txt`:

- **It does not support comments.** Every line is passed to `apt-get install`
  verbatim, so a `#` comment makes apt try to install each of its words as a
  package and the build fails with `E: Unable to locate package #`.
- **Package names track the host's Debian release.** Streamlit Cloud currently
  runs Debian trixie, where the glib runtime is `libglib2.0-0t64`; on bookworm
  and earlier it is `libglib2.0-0`. A wrong name fails the whole apt step, so
  check the build log's Debian release if this ever breaks.

### What works when hosted, and what does not

The **Image** and **Video** tabs work fully — upload, detect, track, download.

**The viewer's own camera also works** — open the app on a phone, pick
**This device's camera**, and the browser asks for camera permission. Photo mode
(`st.camera_input`) snaps a frame and detects it; Live mode (`streamlit-webrtc`)
streams continuous video to the server and returns it annotated. This works
hosted because the *browser* captures the video — the server never has to reach
the camera. Live mode runs on the free tier's two CPU cores, so expect a few
FPS there.

What still cannot work hosted: the server's own webcam (it has none) and
cameras on your home network — `192.168.x.x` and `10.x.x.x` are private
addresses with no route from the public internet. CCTV/RTSP input therefore
needs EyeSpot running on your own machine, which is also where it is fastest.
The app detects a hosted environment through `SPACE_ID`, `EYESPOT_CLOUD`, or
the same key in Streamlit secrets.

### Hugging Face Spaces

`deploy/` holds a working Docker Space setup (`Dockerfile`, `requirements-hf.txt`,
`README_hf.md`, and `deploy_to_hf.py`), including a `--mode private-source`
variant that keeps a public Space's source private by cloning it at build time
with a read-only GitHub deploy key.

It is not currently in use: Hugging Face now requires a **PRO subscription** to
host Gradio or Docker Spaces on free hardware, and rejects free-tier creation
with `402 Payment Required`. Only Static Spaces remain free. The setup is kept
because it works the moment an account has PRO.

### Why not Vercel, Netlify, or similar

Serverless platforms cannot host this app:

| Constraint | Reality |
|---|---|
| Vercel Python bundle limit | 250 MB |
| `torch` alone | ~770 MB |
| All required packages | ~930 MB |
| Streamlit needs | a long-lived WebSocket server; serverless functions are stateless |
| Function timeout | 60–300 s, while video processing runs for minutes |

A container or app platform is the right fit — Streamlit Community Cloud,
Hugging Face Spaces, Render, Railway or Fly.io.

---

## ⚙️ Configuration

Every default lives in [config.py](config.py) — model, thresholds, resolution,
colours, keyboard shortcuts and output paths. CLI flags override these defaults.

Some common ones:

```python
MODEL_NAME = "yolov8n.pt"       # yolov8s.pt / yolov8m.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.5      # lower it for more (but weaker) detections
IMAGE_SIZE = 640                # 320 = faster, 640 = more accurate
CLASS_FILTER = []               # e.g. ["person", "car"]
CAMERA_INDEX = 0                # 1 for a second camera
```

---

## 📁 Project Structure

```
EyeSpot/
├── main.py                  # Entry point — CLI, main loop, keyboard handling
├── config.py                # All default settings
├── requirements.txt
├── test_smoke.py            # Self-test suite
├── cameras.example.json     # Camera settings template (cameras.json is not committed)
│
├── detector/
│   ├── model_loader.py      # Model loading + device auto-detection (CPU/CUDA/MPS)
│   ├── detect.py            # Detector class — detect(), track(), draw()
│   └── target_lock.py       # Target lock — follow, trail, lost/re-acquire, HUD
│
├── utils/
│   ├── cameras.py           # Saved camera profiles + brand URL builder
│   ├── fps_counter.py       # Moving-average FPS
│   ├── overlay.py           # On-screen info panel
│   └── save_output.py       # Screenshot / video writer / CSV logger
│
├── ui/
│   └── streamlit_app.py     # Web interface
│
├── models/                  # Downloaded weights (automatic)
└── outputs/                 # Screenshots, videos, CSV logs
```

**Design:** the detection logic is completely separate from the UI.
`Detector.detect()` returns plain `Detection` dataclass objects, so the same
logic serves the CLI, Streamlit, or any future frontend (Flask/FastAPI).

---

## 🧪 Testing

```bash
python test_smoke.py
```

This checks model loading, image detection, the video pipeline, class groups,
camera profiles, tracking, target lock, output saving and H.264 transcoding.

---

## 📊 Performance

`yolov8n.pt` at 640 px on an Intel CPU: **~55 FPS** inference (the PRD target was
15+ FPS).

To go faster:
- `--imgsz 320` — considerably faster, slightly less accurate
- `--model yolov8n.pt` — the smallest model (already the default)
- Run on a GPU — `--device cuda` (30–100+ FPS)

For better accuracy: `--model yolov8s.pt` or `--imgsz 800`.

---

## 🔧 Troubleshooting

**"Could not open the webcam"**
- Is the camera connected? On Linux check with `ls /dev/video*`
- Is another app (Zoom/Teams) using the camera?
- Try another index: `--source 1`

**"Could not connect to the stream" (phone / CCTV)**
- Test the URL in **VLC** first (Media → Open Network Stream). If VLC cannot play
  it, the problem is the URL or the camera, not this code.
- Are the phone and the PC on the same Wi-Fi network?
- Don't forget the trailing `/video` on a phone URL
- Does the password contain special characters (`@`, `:`, `/`)? URL-encode them
  (`@` → `%40`) — the Streamlit URL builder does this automatically
- Is RTSP enabled in the camera's settings?

**Stream latency / the video lags behind**
- Use the camera's **substream** (lower resolution) — `subtype=1` or `stream2`
- Speed up inference with `--imgsz 320`

**Track IDs keep changing**
- Try `--tracker botsort.yaml` (better re-identification)
- Lower `--conf` slightly — if detection breaks up, the ID breaks with it
- Very low FPS weakens tracking, because objects move further between frames

**"Ultralytics is not installed"**
- Did you activate the venv? Run `pip install -r requirements.txt` again

**Model download fails**
- Check your internet connection (weights are downloaded on the first run)
- Or download `yolov8n.pt` manually and place it in the `models/` folder

**Video not saved / the file is empty**
- Try `VIDEO_CODEC = "XVID"` in `config.py` with an `.avi` output

**In-browser video preview is blank (Streamlit)**
- Install `ffmpeg` — OpenCV cannot write H.264, so the app transcodes with ffmpeg
  for browser playback. Downloads work either way.

**FPS is very low**
- Use `--imgsz 320`, or run on a GPU

---

## 🔮 Future Enhancements (Phase 2+)

The full roadmap, with each item sized and ordered, is in
**[NEXT_PHASES.md](NEXT_PHASES.md)**. The headlines:

- **An alert system** — the highest-value gap. EyeSpot shows you what it sees
  while you watch; alerts would make it watch on your behalf
- Event history you can review, instead of raw CSV rows
- Zone and line-crossing counting ("how many people passed the gate")
- Multiple cameras at once (currently one source at a time)
- Target lock in the web UI, where it is currently invisible
- Custom training, for objects outside the 80 COCO classes

> Tracking (ByteTrack/BoT-SORT), target lock and IP/mobile/CCTV camera support
> were listed as Phase 2 in the PRD — they are **already built**, see above.

---

## 📄 License & Credits

- Detection models: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Built to the specification in [YOLO_Realtime_Detection_PRD.md](YOLO_Realtime_Detection_PRD.md)
