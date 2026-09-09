# 📋 EyeSpot — Final Project Report
## Real-Time Object Detection & Tracking System (YOLOv8)

**Report date:** 9 September 2026
**Status:** ✅ Phase 1 complete — every PRD requirement met, plus four extra features
**Repository:** `EyeSpot`
**Specification:** [YOLO_Realtime_Detection_PRD.md](YOLO_Realtime_Detection_PRD.md)

---

## 1. Executive Summary

EyeSpot is a working real-time object detection **and tracking** system that:

- Recognises **80 object classes** live (person, car, dog, cat, bird, phone, laptop, bag, bottle…)
- Accepts **four kinds of input** — laptop/USB webcam, mobile phone (HTTP), CCTV/IP camera (RTSP), and video/image files
- Goes beyond detection: every object receives a **persistent ID**, and a single object can be **locked** and followed
- Ships **two interfaces** — a fast CLI with keyboard shortcuts, and a Streamlit web UI served over HTTP
- **Saves results** — screenshots, annotated video, and a CSV log for later analysis

Test suite: **18/18 passing**. Inference speed on CPU: **~55 FPS**.

---

## 2. PRD Requirements — Status

### In scope (PRD section 3)

| # | Requirement | Status | Where |
|---|---|---|---|
| 6.1 | Real-time webcam detection | ✅ | `main.py --source 0` |
| 6.1 | Bounding box + label + confidence % | ✅ | `detector/detect.py` → `draw()` |
| 6.1 | Live FPS counter | ✅ | `utils/fps_counter.py` |
| 6.2 | Video file input (.mp4/.avi/.mov/.mkv) | ✅ | `--source video.mp4` |
| 6.2 | Save the output video | ✅ | `--save` |
| 6.3 | Adjustable confidence threshold | ✅ | `--conf`, and `+`/`-` at runtime |
| 6.3 | Class filter | ✅ | `--classes person car` |
| 6.3 | Pause / resume / stop | ✅ | `p` / `Space` / `q` |
| 6.4 | On-screen object list and counts | ✅ | `utils/overlay.py` |
| 6.4 | Screenshot capture (`s` key) | ✅ | `utils/save_output.py` |
| 6.5 | CSV detection log | ✅ | `--log` |
| 6.5 | Streamlit web UI | ✅ | `ui/streamlit_app.py` |
| 7 | 15+ FPS on CPU | ✅ | ~55 FPS inference (see §4) |
| 7 | Windows and Linux portability | ✅ | no OS-specific code |
| 7 | Graceful errors, no crashes | ✅ | every failure gives a clear message and a fix |
| 7 | Modular code | ✅ | detector / utils / ui kept separate |

### Listed as "Phase 2 / out of scope" in the PRD — but delivered anyway

| Feature | Status | Why it matters |
|---|---|---|
| **IP camera / mobile / CCTV support** | ✅ | RTSP and HTTP streams, with auto-reconnect |
| **Object tracking (ByteTrack / BoT-SORT)** | ✅ | Persistent IDs — "this is the same person" |
| **Target lock** | ✅ | Pick one object and follow it (trail, LOST/re-acquire) |
| **Saved camera profiles** | ✅ | `--camera gate` — no need to remember full URLs |

### Still outstanding (Phase 2)

| Feature | Why it is not here |
|---|---|
| Custom dataset training | A project of its own — needs dataset labelling and a training pipeline |
| Alert / notification system | Currently screen and CSV only; email/SMS/buzzer would need adding |
| Cloud deployment | Designed for a local machine |
| Multiple simultaneous cameras | One source at a time; parallel capture would need threading |
| Face recognition | YOLO reports "person", not *which* person |

---

## 3. Test Results

```
$ python test_smoke.py

  ✅ Dependencies installed — opencv 5.0.0, numpy 2.5.2, ultralytics 8.4.143
  ✅ Config valid — model=yolov8n.pt, conf=0.5
  ✅ FPS counter
  ✅ Model loads — 80 classes on cpu
  ✅ Detection on blank frame
  ✅ Detection on sample image — person x3, bus x1
  ✅ Class filter
  ✅ Class groups (animals/vehicles)
  ✅ Overlay drawing
  ✅ Output saving — screenshot, video, CSV
  ✅ Video pipeline
  ✅ Browser transcode (H.264)
  ✅ CLI parsing
  ✅ Stream URLs (phone/CCTV) — 4 URL formats + password masking
  ✅ Camera profiles (cameras.json)
  ✅ Object tracking (persistent IDs)
  ✅ Target lock
  ✅ Target lock drawing

  18 passed, 0 failed
```

The Streamlit UI was verified separately, end to end, using Streamlit's own
`AppTest` harness — upload, processing, result caching, and both download
buttons — with no errors.

---

## 4. Measured Performance

Test machine: Linux 6.8, Python 3.12.3, x86_64 **CPU (16 cores)**, no GPU.
Input 1280×720, model `yolov8n.pt`, `imgsz=640`:

| Mode | Speed | Per frame |
|---|---|---|
| Detection | **55.2 FPS** | 18 ms |
| Tracking (ByteTrack) | **52.9 FPS** | 19 ms |

> These are **pure inference** figures. End-to-end throughput — including frame
> capture, drawing and display — is lower in practice (typically 20–30 FPS),
> which still clears the PRD's 15 FPS target comfortably.

A full 60-second 960×540 video was processed end to end at **52.8 FPS average**
(1440 frames in 27.3 s), producing an annotated video and a CSV log.

To go faster: `--imgsz 320`. For better accuracy: `--model yolov8s.pt` or
`--imgsz 800`. On a GPU (`--device cuda`), expect 30–100+ FPS.

---

## 5. What can it actually track?

### ✅ Supported

**All 80 COCO classes**, with no training required:

> person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
> fire hydrant, stop sign, parking meter, bench, **bird**, cat, dog, horse, sheep, cow,
> elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
> skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
> surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
> apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
> potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
> cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase,
> scissors, teddy bear, hair drier, toothbrush

Full list: `python main.py --list-classes`

For any of those classes:

| Task | How |
|---|---|
| Detect it | `python main.py --camera gate` |
| Watch one thing only | `--classes person` |
| Watch a whole group | `--classes animals` / `vehicles` / `pets` / `road` |
| Assign persistent IDs (tracking) | `--track` → boxes show `#1`, `#2`, … |
| Follow one object (lock) | **click** it in the window, or `--lock person` |
| Count unique people or vehicles | `--track --log` → distinct `track_id` values in the CSV |
| Measure how long something stayed | that `track_id`'s frame count ÷ FPS |
| Trace the route it took | plot the box centres (x1,y1,x2,y2 in the CSV) for that track ID |

While locked, the display shows an amber box with corner brackets, a crosshair,
the offset from the frame centre (`L171 D47`), a motion trail, every other object
dimmed, and **TARGET LOST** if it disappears — with automatic re-acquisition when
it returns.

### ❌ Not supported (Phase 2)

| Capability | Reason |
|---|---|
| **Which** person it is (face recognition) | The model only knows the class "person" |
| **Which** bird species | The model only knows "bird"; species need custom training |
| Reading licence plates | That is OCR, a separate problem |
| Custom objects (a specific machine part, say) | Requires training on your own dataset |
| Recognising the same person **across two cameras** | Track IDs are valid only within a single stream |
| Line-crossing or zone alerts | Detection and lock exist; counting logic would need adding |

> Also note: a track ID lasts only for **that session**. Restart the app and IDs
> begin again from 1.

---

## 6. Running it on another machine (after a git push)

### 6.1 Pushing from this machine

The repository is already published as `Abubakar2350/EyeSpot` (private). Day to
day:

```bash
git add .
git commit -m "what changed"
git push
```

To set it up from scratch:

```bash
cd EyeSpot
git init
git add .
git commit -m "EyeSpot — real-time detection, tracking, camera profiles"
git branch -M main
git remote add origin https://github.com/Abubakar2350/EyeSpot.git
git push -u origin main
```

**What goes into git and what does not** (`.gitignore`):

| Committed ✅ | Excluded ❌ | Why excluded |
|---|---|---|
| All source code (`main.py`, `detector/`, `utils/`, `ui/`) | `.venv/` | Each machine builds its own |
| `requirements.txt` | `models/*.pt` | Downloaded automatically on first run |
| `README.md`, `REPORT.md`, the PRD | `outputs/` | Generated results |
| `cameras.example.json` | **`cameras.json`** | **It contains CCTV passwords** |
| `test_smoke.py`, `sample.jpg` | `__pycache__/`, `.vscode/` | Machine-specific |

> 🔒 This matters: never push your CCTV password to git. That is exactly why
> `cameras.json` is ignored and only `cameras.example.json` is committed.

### 6.2 On the new machine — five commands

```bash
git clone https://github.com/Abubakar2350/EyeSpot.git
cd EyeSpot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python test_smoke.py               # confirm everything works
python main.py                     # starts on the webcam
```

The first `python main.py` downloads `yolov8n.pt` (about 6 MB), so **an internet
connection is needed once**. After that it runs offline.

### 6.3 Platform notes

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
- Use `python`, not `python3`
- If PowerShell refuses to activate: `Set-ExecutionPolicy -Scope Process RemoteSigned`

**Ubuntu / Debian**

```bash
sudo apt install python3-venv python3-pip     # if venv creation fails
sudo apt install libgl1 libglib2.0-0          # needed for the OpenCV window
sudo apt install ffmpeg                       # for in-browser video preview in the web UI
```

**CPU-only machine** — to avoid torch's 2.5 GB download, run this **before**
`pip install -r requirements.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU machine** — nothing extra is needed; it is detected automatically.
Confirm by checking that the run prints `device: cuda`.

### 6.4 Camera settings on the new machine

`cameras.json` does not come from git (because of the passwords), so create it
once:

```bash
cp cameras.example.json cameras.json      # Windows: copy cameras.example.json cameras.json
```

Then put your real IPs and passwords in the file — **or** save them from the
command line:

```bash
python main.py --source 0 --save-camera laptop
python main.py --source http://192.168.1.5:8080/video --save-camera phone
python main.py --source "rtsp://admin:pass@192.168.1.10:554/stream2" --save-camera gate

python main.py --list-cameras     # verify
python main.py --camera gate      # run
```

> ⚠️ IP addresses **change with the network**. A home address of `192.168.1.5`
> will not work on the office Wi-Fi — check the app's new IP and update
> `cameras.json`.

### 6.5 First-run checklist

| Check | Command | Expected |
|---|---|---|
| Python version | `python --version` | 3.10 or newer |
| Virtualenv active | look at the prompt | `(.venv)` is shown |
| Dependencies | `python test_smoke.py` | 18 passed |
| Model download | first `python main.py` | "Model ready. Classes: 80" |
| Camera list | `python main.py --list-cameras` | your cameras |
| Web UI | `streamlit run ui/streamlit_app.py` | opens at `localhost:8501` |

---

## 7. Camera setup — mobile, CCTV, laptop

The only difference between the three is **what you write as the source** —
everything else (detection, tracking, lock, saving) behaves identically.

| Camera | Protocol | Source |
|---|---|---|
| Laptop / USB webcam | direct device | `0`, `1`, `2` … |
| Mobile phone | **HTTP** | `http://<phone-ip>:8080/video` |
| CCTV / IP camera | **RTSP** | `rtsp://user:pass@<ip>:554/...` |

### Mobile phone (HTTP) — the easiest option

1. Install **"IP Webcam"** on Android (free), or DroidCam/Iriun on iPhone
2. Open the app and tap **Start server**
3. The screen shows an IP, e.g. `http://192.168.1.5:8080`
4. The trailing **`/video`** is required:

```bash
python main.py --source http://192.168.1.5:8080/video --save-camera phone
```

The phone and the PC must be on the **same Wi-Fi** network (mobile data will not work).

| App | Default port | Path |
|---|---|---|
| IP Webcam (Android) | 8080 | `/video` |
| DroidCam | 4747 | `/video` |
| Iriun | 8080 | `/video` |

### CCTV / IP camera (RTSP)

| Brand | URL format (`sub` = lower resolution, faster) |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| Dahua / CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream2` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_sub` |
| ONVIF (generic) | `rtsp://user:pass@IP:554/onvif2` |

There is no need to memorise these — use the **URL builder in the Streamlit UI**
(choose phone or CCTV, pick the brand, enter the IP; the URL is assembled and the
password URL-encoded for you), or call it from code:

```python
from utils.cameras import build_url
build_url("tplink", "192.168.1.10", user="admin", password="pa@ss")
# 'rtsp://admin:pa%40ss@192.168.1.10:554/stream2'
```

**Finding the camera's IP:** your router's admin page (connected devices), or the
camera's own app settings.

**Test in VLC first** — Media → Open Network Stream. If VLC plays it, this will
too; if it does not, the problem is the camera or the network, not the code.

> 💡 Always prefer the **substream** (`102`, `subtype=1`, `stream2`, `_sub`). The
> main stream is 4K — slow on CPU and unnecessary for detection.

### Laptop / USB webcam

```bash
python main.py --source 0        # first camera
python main.py --source 1        # second
```
On Linux, list them with `ls /dev/video*`

### Stream troubleshooting

| Problem | Fix |
|---|---|
| Will not connect at all | Test in VLC; same Wi-Fi?; is RTSP enabled? |
| Video lags behind | Use the substream and add `--imgsz 320` |
| Password contains `@` or `:` | URL-encode it (`@` → `%40`) — `build_url()` does this for you |
| Drops out mid-stream | Nothing to do — it reconnects automatically, five times |

---

## 7b. Hosted Deployment

The web UI is deployed to **Streamlit Community Cloud**: free, served from the
private GitHub repository, with a public app link and no public source.

| Setting | Value |
|---|---|
| Repository | `Abubakar2350/EyeSpot` (private) |
| Branch | `main` |
| Main file | `ui/streamlit_app.py` |
| Python | 3.11 (`.python-version`) |
| Secret | `EYESPOT_CLOUD = "1"` |

Three repository files make hosting work: `requirements.txt` (torch pinned to
`+cpu`), `packages.txt` (`ffmpeg`, `libgl1`, `libglib2.0-0`), and
`.python-version`.

### Three platform constraints found along the way

**Vercel cannot host this at all.** Its Python bundle limit is 250 MB while
torch alone is ~770 MB (~930 MB for all dependencies), Streamlit needs a
long-lived WebSocket server rather than stateless functions, and video
processing outruns the 60–300 s function timeout. The deploy fails at build.

**Hugging Face no longer offers a free Docker Space.** Creating one returns
`402 Payment Required`: "Static Spaces are free for everyone, but hosting Gradio
and Docker Spaces on free cpu-basic requires a PRO subscription." Separately,
`sdk: streamlit` is no longer accepted at all — Streamlit apps must be Docker
Spaces. A complete Docker Space setup is kept under `deploy/` for whenever an
account has PRO, including a `--mode private-source` variant that keeps a public
Space's source private by cloning it at build time with a read-only deploy key.

**Torch must be pinned to `+cpu`.** The default PyPI wheel bundles the CUDA
runtime and pulls roughly 2.5 GB of `nvidia-*` packages that no CPU host can
use. The `+cpu` build, which exists only on the PyTorch index, is 196 MB and
downloads in about 20 seconds.

### What a hosted demo can and cannot do

The **Image** and **Video** tabs work fully: upload, detect, track, download.

The **Live camera** tab cannot work on any hosted server, and the app now says
so rather than failing with a connection error. The server has no webcam, and it
cannot reach a camera on a home network — `192.168.x.x` and `10.x.x.x` are
private addresses with no route from the public internet. The app detects a
hosted environment through `SPACE_ID`, `EYESPOT_CLOUD`, or the same key in
Streamlit secrets.

Phone, CCTV and webcam input consequently require a local install, which is also
where performance is best.

---

## 8. File Map

```
EyeSpot/
├── main.py                   # Entry point — CLI, main loop, keyboard/mouse
├── config.py                 # All default settings
├── requirements.txt
├── test_smoke.py             # 18 self-tests
├── cameras.example.json      # Camera settings template (cameras.json is not committed)
│
├── detector/
│   ├── model_loader.py       # Model loading + device auto-detection (CPU/CUDA/MPS)
│   ├── detect.py             # Detector — detect(), track(), draw()
│   └── target_lock.py        # Target lock — follow, trail, LOST/re-acquire, HUD
│
├── utils/
│   ├── cameras.py            # Camera profiles + brand URL builder + password masking
│   ├── fps_counter.py        # Moving-average FPS
│   ├── overlay.py            # On-screen info panel
│   └── save_output.py        # Screenshot / video writer / CSV logger / H.264 transcode
│
├── ui/
│   └── streamlit_app.py      # Web UI (HTTP) — image/video/live tabs + camera settings
│
├── models/                   # Weights (auto-downloaded, not in git)
└── outputs/                  # Screenshots, videos, CSV logs (not in git)
```

**Design principle:** the detection logic is entirely decoupled from the UI.
`Detector.detect()` returns plain `Detection` objects, which is why the CLI, the
Streamlit app, and any future frontend (Flask/FastAPI/mobile) can all share it.

---

## 9. Everyday Commands

```bash
# Save a camera once
python main.py --source http://192.168.1.5:8080/video --save-camera phone
python main.py --list-cameras

# Run it
python main.py --camera phone
python main.py --camera gate --classes person          # people only
python main.py --camera gate --track                   # with persistent IDs
python main.py --camera gate --track --lock person     # lock the first person
python main.py --camera gate --track --log --save      # record CSV + video

# Process existing footage
python main.py --source cctv_footage.mp4 --track --log --no-display

# Web UI (in the browser, over HTTP)
streamlit run ui/streamlit_app.py

# Birds
python main.py --camera phone --track --lock bird --classes bird --conf 0.3 --imgsz 960
```

Keyboard (video window): `q` quit · `s` screenshot · `p` pause · `o` overlay ·
`+`/`-` confidence · `t` tracking · **click** lock · `l` auto-lock · `c` clear lock

---

## 10. Conclusion

Every in-scope PRD requirement is met, the non-functional targets (15+ FPS,
portability, graceful error handling, modular code) are satisfied, and three
features the PRD deferred to "Phase 2" — **IP/mobile/CCTV camera support**,
**object tracking**, and **target lock** — arrived in Phase 1, together with a
fourth, **saved camera profiles**.

The product is deployable today: `git clone` → `venv` → `pip install` →
`python main.py`.

The natural next step (Phase 2) is an alert system (notify when a person is
detected), zone and line-crossing counting, and custom training for anything
beyond the 80 COCO classes.
