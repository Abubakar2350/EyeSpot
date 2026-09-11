# EyeSpot — Final Project Report
## Real-Time Object Detection & Tracking System (YOLOv8)

| | |
|---|---|
| **Report date** | 10 September 2026 |
| **Status** | Phase 1 complete; deployed |
| **Repository** | `Abubakar2350/EyeSpot` (private) |
| **Live app** | `https://eyespot.streamlit.app` — deployed, currently set to private (see 7.1) |
| **Specification** | [YOLO_Realtime_Detection_PRD.md](YOLO_Realtime_Detection_PRD.md) |
| **Code size** | ~3,680 lines across 29 tracked files |
| **Tests** | 22 / 22 passing |

**Companion documents**

| Document | Covers |
|---|---|
| [USER_REPORT_HOME_CCTV.md](USER_REPORT_HOME_CCTV.md) | Connecting a home CCTV camera, whether that setup persists, and what a multi-user version with Google sign-in would require |
| [NEXT_PHASES.md](NEXT_PHASES.md) | What could be built next, sized and prioritised |
| [LINKEDIN_POSTS.md](LINKEDIN_POSTS.md) | Launch posts for the three demo videos |
| [README.md](README.md) | Installation, usage, camera setup, troubleshooting |

---

## 1. Executive Summary

EyeSpot is a real-time object detection **and tracking** system. It recognises
all 80 COCO classes, assigns every object a persistent identity across frames,
and can lock onto a single chosen object and follow it.

It accepts four kinds of input — a laptop or USB webcam, a mobile phone acting
as an IP camera over HTTP, a CCTV or IP camera over RTSP, and video or image
files — and offers two interfaces: a keyboard-driven CLI and a Streamlit web UI.
Results are saved as screenshots, annotated video, and a CSV detection log
suitable for later analysis.

Every in-scope requirement from the PRD is met. Three features the PRD deferred
to "Phase 2" — IP/mobile/CCTV camera support, object tracking, and target lock —
were delivered in Phase 1, along with a fourth that was not in the PRD at all:
saved camera profiles.

The web UI is deployed to Streamlit Community Cloud. Two other platforms were
evaluated and rejected for concrete, measured reasons; section 7 records them.

---

## 2. Requirements Status

### In scope (PRD section 3)

| # | Requirement | Status | Implementation |
|---|---|---|---|
| 6.1 | Real-time webcam detection | Done | `main.py --source 0` |
| 6.1 | Bounding box + label + confidence % | Done | `detector/detect.py` → `draw()` |
| 6.1 | Live FPS counter | Done | `utils/fps_counter.py` |
| 6.2 | Video file input (.mp4/.avi/.mov/.mkv) | Done | `--source video.mp4` |
| 6.2 | Save the output video | Done | `--save` |
| 6.3 | Adjustable confidence threshold | Done | `--conf`, and `+`/`-` at runtime |
| 6.3 | Class filter | Done | `--classes person car` |
| 6.3 | Pause / resume / stop | Done | `p` / `Space` / `q` |
| 6.4 | On-screen object list and counts | Done | `utils/overlay.py` |
| 6.4 | Screenshot capture | Done | `s` key |
| 6.5 | CSV detection log | Done | `--log` |
| 6.5 | Streamlit web UI | Done | `ui/streamlit_app.py` |
| 7 | 15+ FPS on CPU | Done | ~55 FPS inference (section 5) |
| 7 | Windows and Linux portability | Done | no OS-specific code |
| 7 | Graceful errors, no crashes | Done | every failure states a cause and a fix |
| 7 | Modular code | Done | detector / utils / ui separated |

### Delivered although the PRD deferred them

| Feature | Why it matters |
|---|---|
| IP camera / mobile / CCTV support | RTSP and HTTP streams, with automatic reconnection |
| Object tracking (ByteTrack / BoT-SORT) | Persistent IDs — "this is the same person" |
| Target lock | Follow one chosen object, with trail and re-acquisition |
| Saved camera profiles | `--camera gate` instead of retyping credentials |
| Class groups | `--classes animals` instead of naming ten species |
| Browser device camera | Photo and live WebRTC from the viewer's own phone — the only camera input that works on a hosted demo |
| Road accident detection | Kinematic anomaly detection over tracked road users, fusing six signals (see 5b) |

### Not built (Phase 2)

| Feature | Reason |
|---|---|
| Custom dataset training | A project of its own: labelling plus a training pipeline |
| Alert / notification system | Output is currently screen and CSV; email/SMS would need adding |
| Multiple simultaneous cameras | One source at a time; parallel capture needs threading |
| Face recognition | YOLO reports "person", never *which* person |
| Line-crossing / zone counting | Detection and lock exist; the counting logic does not |

---

## 3. What It Can and Cannot Track

### Supported

All **80 COCO classes**, with no training required:

> person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
> fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,
> elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
> skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
> surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
> apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
> potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
> cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase,
> scissors, teddy bear, hair drier, toothbrush

Full list: `python main.py --list-classes`

Because COCO has no single "animal" class, group shortcuts are provided:
`people`, `animals`, `pets`, `vehicles`, `road`.

| Task | How |
|---|---|
| Detect | `python main.py --camera gate` |
| Watch one class | `--classes person` |
| Watch a group | `--classes animals` |
| Persistent IDs | `--track` → boxes read `#1`, `#2`, … |
| Follow one object | click it in the window, or `--lock person` |
| Count unique objects | `--track --log`, then count distinct `track_id` in the CSV |
| Measure dwell time | that `track_id`'s frame count ÷ FPS |
| Trace a route | plot the box centres for that `track_id` |

When a target is locked the display shows an amber box with corner brackets, a
crosshair, the offset from frame centre (`L171 D47`), a motion trail, every other
object dimmed, and **TARGET LOST** when it disappears — with automatic
re-acquisition on return.

### Not supported

| Capability | Reason |
|---|---|
| *Which* person (face recognition) | The model knows only the class "person" |
| *Which* bird species | The model knows only "bird"; species need custom training |
| Reading licence plates | That is OCR, a separate problem |
| Custom objects | Requires training on your own dataset |
| The same person across two cameras | Track IDs are valid only within one stream |

Track IDs also last only for a single session: restart the app and numbering
begins again at 1.

---

## 4. Test Results

```
$ python test_smoke.py

  Dependencies installed — opencv 5.0.0, numpy 2.5.2, ultralytics 8.4.143
  Config valid — model=yolov8n.pt, conf=0.5
  FPS counter
  Model loads — 80 classes on cpu
  Detection on blank frame
  Detection on sample image — person x3, bus x1
  Class filter
  Class groups (animals/vehicles)
  Overlay drawing
  Output saving — screenshot, video, CSV
  Video pipeline
  Browser transcode (H.264)
  CLI parsing
  Stream URLs (phone/CCTV) — 4 URL formats + password masking
  Camera profiles (cameras.json)
  Object tracking (persistent IDs)
  Target lock
  Target lock drawing

  18 passed, 0 failed
```

The Streamlit UI was additionally verified end to end with Streamlit's own
`AppTest` harness, covering upload, processing, result caching, both download
buttons, and the phone/CCTV URL builder.

---

## 5. Measured Performance

Test machine: Linux 6.8, Python 3.12.3, x86_64 CPU (16 cores), no GPU.
Input 1280×720, model `yolov8n.pt`, `imgsz=640`:

| Mode | Speed | Per frame |
|---|---|---|
| Detection | **55.2 FPS** | 18 ms |
| Tracking (ByteTrack) | **52.9 FPS** | 19 ms |

These are pure inference figures. End-to-end throughput including capture,
drawing and display is lower — typically 20–30 FPS — which still clears the
PRD's 15 FPS target.

A real 60-second 960×540 video processed end to end at **52.8 FPS average**
(1440 frames in 27.3 s), producing an annotated video and a CSV log. Of note,
the nano model occasionally misclassified a car as `train` or `airplane` on
cinematic close-ups; `--model yolov8s.pt` with a higher `--conf` corrects this.

To go faster: `--imgsz 320`. For accuracy: `--model yolov8s.pt` or `--imgsz 800`.
On a GPU (`--device cuda`), expect 30–100+ FPS.

---

## 5b. Road Accident Detection

Added after the main build, on top of the tracking work rather than beside it.

```
Detection → Tracking → Kinematics → Signals → Fusion → Event
```

No additional model is loaded. The engine consumes the tracked detections that
already exist and does arithmetic on their positions. Measured overhead on this
machine: **-0.28 ms per frame** — that is, within measurement noise of running
tracking alone (85.2 FPS tracking only versus 87.2 FPS with the engine attached).

### Six signals, fused

| Signal | Trigger |
|---|---|
| `impact` | Boxes overlapping while the pair was closing, with a jolt within 0.6 s |
| `sudden_stop` | Speed collapsing to under 15% of its recent value |
| `person_down` | A person's box wider than tall, held for over a second |
| `hard_brake` | Deceleration past 1.2 object-lengths/s² |
| `swerve` | Heading changing faster than 2.0 rad/s |
| `ttc` | Time-to-collision under 0.8 s for a converging pair |

The weights are deliberately set so **no single channel can reach the threshold
alone**. This is the core design decision: every individual signal has an
innocent explanation, and only agreement between several distinguishes a crash
from ordinary traffic.

### Engineering decisions worth recording

**Speeds are normalised by object size.** Pixels per second is meaningless
across cameras — the same car yields wildly different values depending on
distance and resolution. Dividing by the object's own diagonal gives
"object-lengths per second", verified by a test asserting that a 160px and an
80px vehicle moving at proportionally equal rates report the same figure.

**Impact needs corroboration, not deep overlap.** The first implementation
scaled the impact signal by how much the boxes overlapped, and never fired:
colliding vehicles meet edge to edge, producing an IoU around 0.14, not 0.5.
The channel was restructured so that the corroborating facts — were they
closing, did one take a jolt — earn most of the strength, with overlap only
modulating the remainder.

**Deceleration is looked up over a window, not a frame.** An impact and the
speed change it causes rarely land in the same frame. Requiring both
simultaneously missed real collisions; a 0.6 s lookback catches them.

**The cooldown started at zero and silenced every clip's opening.** Because the
clock starts near zero on a fresh video, a zero initial value blocked events for
the cooldown's whole length — precisely where a dashcam accident usually sits.
Initialising to negative infinity fixed it, and a test now asserts the first
event arrives within four seconds.

### Measured on real footage

Tested against a 76-second CCTV accident compilation (640×360, 2298 frames)
after an initial report that the feature detected nothing:

| Detection confidence | Events |
|---|---|
| 0.50 (the app's default) | **0** |
| 0.35 | 5 |

The cause was not the scoring. At 0.50 the fused score peaked at 0.714, well
above the 0.55 threshold — but on only **2 isolated frames out of 2298**.
Vehicles in that footage are detected at around 0.5 confidence, so they flicker
in and out, the tracker issues a new ID on each reappearance, and the motion
history resets precisely when the collision makes them hardest to see.

Two changes followed. Persistence became a sliding window (3 of the last 8
frames) rather than a consecutive run, since demanding an unbroken streak
discards evidence that is plainly present. And `--accident` now lowers the
detection confidence to 0.35 by default, which is what actually fixed it.

A track-stability metric was also built as a warning signal and then demoted to
a diagnostic, because measurement showed it pointed the wrong way: lowering the
confidence produced *more* short-lived tracks while detecting *more* accidents.
It is reported, but nothing depends on it.

A follow-up report that many accidents were still being missed led to two more
defects, both found by measuring rather than guessing:

**The CLI measured speed against the wall clock.** A video file is processed as
fast as the CPU allows — this clip ran at 2.5x real time — so every velocity,
acceleration and time-to-collision was inflated by that factor, manufacturing
events that were not there. The Streamlit video tab already drove the clock from
the footage; the CLI now does the same, while live cameras correctly keep using
the wall clock.

**The sudden-stop channel almost never fired.** It compared the last 0.2 s
against the last 0.6 s, but the longer window *contained the stop*, so the
"before" speed was already dragged down and the collapse never looked like one.
Measured across eight clips it fired on exactly one. With disjoint windows —
now against a window ending 0.35 s ago — it fires where it should, and peak
scores on real collisions rose accordingly (one clip from 0.73 to 0.95).

A threshold sweep against scene-cut ground truth then set the defaults:

| Threshold | Persistence | Clips found (of 6) | Extra events |
|---|---|---|---|
| 0.55 | 3 of 8 | 3 | +2 |
| **0.50** | **2 of 8** | **5** | **+3** |
| 0.45 | 3 of 8 | 5 | +4 |
| 0.40 | 2 of 8 | 5 | +5 |

Below 0.50 the extra events accumulate without finding anything new. Three
sensitivity presets expose the trade-off, because sifting recorded footage
tolerates false alarms far better than a live alert does.

The one clip missed at every setting is a night-time junction filmed wide, with
vehicles a few pixels across — a detection limit, not a scoring one.

A further report — that some real accidents were missed, that a car merely
passing by was called a crash, and that a crashed vehicle should stay marked —
led to the most useful change of all: **post-event confirmation**.

The kinematics of a collision and a near miss are nearly identical at the moment
they occur. What separates them is what happens next: a vehicle that has crashed
stops and stays stopped, while one that passed close by drives on. So a
candidate is now held for 2.5 seconds, the vehicles are watched, and it is
confirmed only if at least one is still sitting there.

| Configuration | Clips found (of 6) | Extra events |
|---|---|---|
| Threshold 0.55, no confirmation | 3 | +2 |
| Threshold 0.50, no confirmation | 5 | +3 |
| **Threshold 0.50, with confirmation** | **5** | **+1** |

The same accidents, with a third of the false alarms — and no recall lost, since
a genuine crash leaves its vehicles stationary by definition.

Confirmation also answered the third request directly. The vehicles that came to
rest are marked `ACCIDENTED #id` in red with corner brackets and stay marked for
the rest of the run, so the crashed car remains identifiable long after the
banner fades. Only vehicles that actually stopped are marked — a candidate
sweeps in everything nearby, but the ones still sitting there are the ones that
took part.

### Closing the last gap

Asked for a setting that misses nothing, the honest answer was that perfect
recall is unattainable — but the specific gap was known and could be closed.

The one clip no threshold could reach was a night-time junction with vehicles a
few pixels across. Lowering the threshold never helped because the problem was
never the threshold: the vehicles were barely being detected. Raising the model
size and inference resolution roughly doubled the objects found there, from 5.7
to 11.0 per frame.

Sensitivity presets therefore now carry detection settings as well as scoring
ones, and a `max` preset was added:

| Preset | Accidents found (of 6) | False alarms | Speed |
|---|---|---|---|
| medium | 5 | +1 | 82 FPS |
| **max** | **6** | +3 | 11 FPS |

`max` found every accident in that footage, at roughly seven times the
processing cost. That is the right trade for reviewing recordings and the wrong
one for a live camera, so it is offered rather than made the default.

Confirmation was also relaxed to accept a vehicle that vanished while already
slow. A wrecked car is frequently hidden behind the vehicle that hit it, behind
a gathering crowd, or by the shot cutting away, and insisting it stay visible
discarded those cases.

What cannot be promised: an accident outside the frame, fully occluded, or
involving the camera vehicle itself is invisible to any of this. `max` is the
setting that misses least on the footage it was measured against, not a
guarantee about footage it has never seen.

Reviewing saved snapshots by eye: real collisions are found, and the
characteristic false positive is perspective. The characteristic false positive is perspective — on a dashcam, a
nearby vehicle passing in front of distant pedestrians yields overlapping boxes
that appear to close, which a flat 2D image cannot distinguish from an impact.

The honest framing is that this surfaces *frames worth a human's attention*,
which is worth a great deal across eight hours of footage and is not the same
thing as a verdict.

### What it is not

This is a kinematic anomaly detector. It does **not** include a trained accident
classifier, a temporal model such as a GRU over behaviour, or any training or
evaluation on IDD, BDD100K or CCD. Consequently no accuracy figure against a
labelled benchmark is claimed. A learned classifier would likely outperform
hand-tuned thresholds and is a reasonable Phase 3 item, but it requires the
dataset and a training run.

What *is* verified is the geometry and the fusion, by four tests that stage a
collision, ordinary traffic, isolated hard braking and a long crash sequence,
and assert which of them fires.

---

## 6. Architecture

```
EyeSpot/
├── main.py                   # Entry point — CLI, main loop, keyboard/mouse
├── config.py                 # All default settings
├── requirements.txt          # Python dependencies (torch pinned to +cpu)
├── packages.txt              # apt packages for the hosted build
├── .python-version           # Requests Python 3.11
├── test_smoke.py             # 18 self-tests
├── cameras.example.json      # Camera settings template
│
├── detector/
│   ├── model_loader.py       # Model loading, device auto-detection, class groups
│   ├── detect.py             # Detector — detect(), track(), draw()
│   ├── target_lock.py        # Target lock — follow, trail, LOST/re-acquire, HUD
│   └── accident.py           # Kinematics, six signals, fusion, event HUD
│
├── utils/
│   ├── cameras.py            # Camera profiles, brand URL builder, password masking
│   ├── fps_counter.py        # Moving-average FPS
│   ├── overlay.py            # On-screen info panel
│   └── save_output.py        # Screenshot / video writer / CSV logger / H.264 transcode
│
├── ui/
│   └── streamlit_app.py      # Web UI — image, video and live camera tabs
│
├── deploy/                   # Hugging Face Space setup (see 7.4)
│   ├── Dockerfile
│   ├── Dockerfile.private-source
│   ├── requirements-hf.txt
│   ├── README_hf.md
│   └── deploy_to_hf.py
│
├── FINAL_REPORT.md           # This document
├── USER_REPORT_HOME_CCTV.md  # Home CCTV, persistence, multi-user design
├── NEXT_PHASES.md            # What to build next
├── LINKEDIN_POSTS.md         # Launch posts
├── README.md
├── YOLO_Realtime_Detection_PRD.md
│
├── models/                   # Weights, downloaded automatically (not in git)
└── outputs/                  # Screenshots, videos, CSV logs (not in git)
```

The detection logic is fully decoupled from the UI. `Detector.detect()` returns
plain `Detection` dataclass objects, which is why the CLI, the Streamlit app and
any future frontend can share it unchanged.

**Secrets discipline:** `cameras.json` holds CCTV passwords and is gitignored;
only `cameras.example.json` is committed. Passwords are masked wherever a URL is
printed or logged, and URL-encoded automatically when a URL is built.

---

## 7. Deployment

### 7.1 Current state

The web UI is deployed to **Streamlit Community Cloud**, which is free, builds
directly from the private GitHub repository, and serves a public link without
exposing the source.

| Setting | Value |
|---|---|
| Repository | `Abubakar2350/EyeSpot` (private) |
| Branch | `main` |
| Main file | `ui/streamlit_app.py` |
| URL | `https://eyespot.streamlit.app` |
| Secret | `EYESPOT_CLOUD = "1"` |
| Visibility | **Private at the time of writing** — the URL redirects visitors to a Streamlit login page. Making it public is one setting: Share → "Make this app public". |

Three repository files make the hosted build work:

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies, torch pinned to `+cpu` |
| `packages.txt` | `ffmpeg`, `libgl1`, `libglib2.0-0t64` |
| `.python-version` | Requests Python 3.11 |

### 7.2 Free-tier limits

Published limits for Community Cloud, as stated in Streamlit's documentation:

| Resource | Limit |
|---|---|
| CPU | 0.078 cores minimum, **2 cores maximum** |
| Memory | 690 MB minimum, **2.7 GB maximum** |
| Storage | up to 50 GB |
| Private apps | **one at a time** |
| Public apps | not numerically limited |
| Inactivity | apps with **no traffic for 12 hours go to sleep** |
| Update rate | at most five app updates per minute from GitHub |
| Region | all apps are hosted in the United States |

There is no expiry date and no usage meter — the app runs indefinitely, for free,
so long as it stays within these limits. Sleeping is not deletion: a sleeping app
shows a wake button, and **anyone with view access can wake it**, not just the
owner. Waking takes a few seconds.

What this means for EyeSpot specifically:

- **Memory is comfortable.** Torch plus `yolov8n` sits well inside 2.7 GB.
- **CPU is the real constraint.** Two cores at best, against the 16 used for the
  55 FPS benchmark, so hosted inference is several times slower than local. Fine
  for images and short clips; long videos will take a while.
- **Uploads are capped** at Streamlit's default 200 MB per file.
- **Nothing persists.** Saved outputs live on ephemeral storage and disappear
  when the app sleeps or restarts, which is why the video tab offers downloads.

### 7.3 Why not Vercel

Vercel cannot host this application. Three independent blockers, each sufficient
on its own:

| Constraint | Measured reality |
|---|---|
| Python bundle limit | 250 MB |
| `torch` alone | **~770 MB** |
| All required packages | **~930 MB** |
| Streamlit requires | a long-lived WebSocket server; Vercel functions are stateless and end with the request |
| Function timeout | 60–300 s, while a 60-second video takes ~27 s and longer clips take minutes |

The deployment would fail at build time on size alone; even if it did not, a
Streamlit server cannot exist in a stateless function. Vercel is the right tool
for a frontend or a small API, not for a PyTorch application that holds a session
open. The same reasoning excludes Netlify and comparable serverless hosts.

If a Vercel presence is wanted, the workable pattern is a static landing page
there that links to the live app hosted elsewhere.

### 7.4 Why not Hugging Face Spaces

Hugging Face was the first choice and the setup was built and tested. It was
abandoned for a policy reason, not a technical one. Two obstacles appeared:

**The Streamlit SDK no longer exists.** Creating a Space with `sdk: streamlit`
is rejected:

```
Invalid option: expected one of "gradio"|"docker"|"static" at sdk
```

Streamlit apps must now be packaged as Docker Spaces. That was done, and works.

**Docker Spaces are no longer free.** Creating one returns:

```
402 Payment Required — Static Spaces are free for everyone, but hosting
Gradio and Docker Spaces on free cpu-basic requires a PRO subscription.
```

A PRO subscription costs about $9/month. The account in question does not have
one, so deployment stopped there.

The complete setup is retained under `deploy/` and works the moment an account
has PRO:

| File | Role |
|---|---|
| `Dockerfile` | Serves Streamlit on port 7860 |
| `Dockerfile.private-source` | Public Space whose source stays private |
| `requirements-hf.txt` | Headless OpenCV, CPU-pinned torch |
| `README_hf.md` | Space card and YAML configuration |
| `deploy_to_hf.py` | Creates the Space, stages files, uploads, sets secrets |

The `private-source` variant deserves a note, because it solves a real problem:
a **public** Hugging Face Space shows its entire source in the Files tab, while a
**private** one cannot be opened by anyone else at all. Neither gives "public
demo, private code". The variant uploads only a Dockerfile; the application is
cloned at build time from the private GitHub repository using a **read-only
deploy key** stored as a Space secret. A deploy key was chosen over a personal
access token because a PAT carries access to every repository on the account,
whereas a deploy key is scoped to one repository and can be marked read-only —
verified by confirming that a push attempt with it is rejected. Hugging Face
never exposes a Space's built image, so visitors get the running app and not the
code. The key was deleted after the platform change, since nothing uses it.

### 7.5 Platform comparison

| Platform | Cost | Public link | Private code | Verdict |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Free | Yes | Yes | **In use** |
| Hugging Face Spaces | $9/month (PRO) | Yes | Yes, via private-source | Ready, needs PRO |
| Render / Railway | ~$7–25/month | Yes | Yes | Free tier RAM too small |
| Vercel / Netlify | — | — | — | Technically impossible |

### 7.6 Hosted versus local

The **Image** and **Video** tabs work fully when hosted: upload, detect, track,
download.

**The viewer's own camera works too.** Opening the app on a phone and choosing
"This device's camera" makes the browser ask for camera permission; photo mode
(`st.camera_input`) captures a frame for detection, and live mode
(`streamlit-webrtc`) streams continuous video to the server and plays it back
annotated. This works on a hosted server precisely because the *browser* does
the capturing — the server never needs a route to the camera. A public STUN
server (`stun.l.google.com:19302`) lets the phone and the server connect across
NAT. Live throughput is bounded by the free tier's two CPU cores, so a few FPS
is the realistic expectation there; the same mode is much smoother locally.

What still cannot work hosted: the server's own webcam (it has none) and any
camera on a home network — `192.168.x.x` and `10.x.x.x` are private addresses
with no route from the public internet. CCTV/RTSP input therefore requires a
local install, which is also where performance is best. The app detects a hosted
environment through `SPACE_ID`, `EYESPOT_CLOUD`, or the same key in Streamlit
secrets, and adjusts the camera options it offers accordingly.

---

## 8. Running It Locally

```bash
git clone https://github.com/Abubakar2350/EyeSpot.git
cd EyeSpot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python test_smoke.py               # expect 18 passed
python main.py                     # starts on the webcam
```

The first run downloads `yolov8n.pt` (about 6 MB), so an internet connection is
needed once. After that it runs offline.

### Platform notes

**Windows** — use `python`, not `python3`. If PowerShell refuses to activate the
environment: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

**Ubuntu / Debian** —

```bash
sudo apt install python3-venv python3-pip     # if venv creation fails
sudo apt install libgl1 libglib2.0-0          # for the OpenCV window
sudo apt install ffmpeg                       # for browser-playable video output
```

**NVIDIA GPU** — `requirements.txt` pins the CPU build of torch, which is what
hosting needs. To use a GPU, install the CUDA build over it:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Confirm by checking that the run prints `device: cuda`.

### Camera settings on a new machine

`cameras.json` is not in git, because it holds passwords. Create it once:

```bash
cp cameras.example.json cameras.json
```

Or save cameras from the command line:

```bash
python main.py --source 0 --save-camera laptop
python main.py --source http://192.168.1.5:8080/video --save-camera phone
python main.py --source "rtsp://admin:pass@192.168.1.10:554/stream2" --save-camera gate

python main.py --list-cameras     # verify (passwords masked)
python main.py --camera gate      # run
```

IP addresses change with the network: a home address of `192.168.1.5` will not
work on a different Wi-Fi, so update `cameras.json` when the network changes.

---

## 9. Camera Setup

The only difference between camera types is what goes in `--source`; detection,
tracking, locking and saving behave identically.

| Camera | Protocol | Source |
|---|---|---|
| Laptop / USB webcam | direct device | `0`, `1`, `2` … |
| Mobile phone | HTTP | `http://<phone-ip>:8080/video` |
| CCTV / IP camera | RTSP | `rtsp://user:pass@<ip>:554/...` |

**Mobile phone.** Install "IP Webcam" on Android (free), or DroidCam/Iriun on
iPhone. Open the app, tap **Start server**, and use the IP it displays. The
trailing `/video` is required. The phone and the PC must share a Wi-Fi network;
mobile data will not work.

| App | Default port | Path |
|---|---|---|
| IP Webcam (Android) | 8080 | `/video` |
| DroidCam | 4747 | `/video` |
| Iriun | 8080 | `/video` |

**CCTV.** Substream URLs — lower resolution, considerably faster for detection:

| Brand | URL |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| Dahua / CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream2` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_sub` |
| ONVIF (generic) | `rtsp://user:pass@IP:554/onvif2` |

The Streamlit UI includes a URL builder that asks whether the camera is a phone
or a CCTV unit, shows only the relevant list, and assembles the URL — encoding
special characters in the password automatically. The same is available in code:

```python
from utils.cameras import build_url
build_url("tplink", "192.168.1.10", user="admin", password="pa@ss")
# 'rtsp://admin:pa%40ss@192.168.1.10:554/stream2'
```

Always test an unfamiliar URL in VLC first (Media → Open Network Stream). If VLC
cannot play it, the problem is the camera or the network, not this software.

| Problem | Fix |
|---|---|
| Will not connect | Test in VLC; same Wi-Fi?; is RTSP enabled on the camera? |
| Video lags behind | Use the substream, and add `--imgsz 320` |
| Password contains `@` or `:` | URL-encode it (`@` → `%40`); `build_url()` does this |
| Drops mid-stream | Nothing to do — it reconnects automatically, five times |

---

## 10. Engineering Notes

Decisions and defects worth recording, because each cost time to find.

**OpenCV cannot write H.264.** Only `mp4v` (MPEG-4 Part 2) is available, and no
browser's HTML5 player supports it, so result videos rendered as a black
rectangle. Output is now transcoded with `ffmpeg` after processing — 28 MB of
`mp4v` becomes 7.6 MB of H.264 in about 1.4 seconds — and the smaller file is
what the download button serves. When `ffmpeg` is absent the preview is skipped
and downloads still work.

**Streamlit re-runs the whole script on every interaction.** The video tab
therefore reprocessed the entire video whenever a slider moved. Results are now
cached in `st.session_state`, keyed by the uploaded file, with an explicit
"Process again" button.

**Torch must be pinned to `+cpu`.** A bare `torch` requirement resolves to the
PyPI wheel, which bundles the CUDA runtime and pulls roughly 2.5 GB of `nvidia-*`
packages that no CPU host can use. The `+cpu` build — published only on the
PyTorch index — is 196 MB and downloads in about 20 seconds. Wheels exist for
CPython 3.10 through 3.14, so the pin holds whichever Python a host selects.

**`packages.txt` does not support comments.** Streamlit passes every line
directly to `apt-get install`, so a header comment made apt attempt to install
each of its words and the build failed with `E: Unable to locate package #`.

**apt package names track the host's Debian release.** The same file named
`libglib2.0-0`, which does not exist on Debian trixie — the `time_t` transition
renamed it `libglib2.0-0t64`. One wrong name fails the entire apt step. Worth
noting that Streamlit's own documentation still says the base image is Debian 11
"bullseye"; the build log said trixie, and the build log is authoritative.

**A deploy key beats a personal access token.** For the Hugging Face
private-source build, a PAT would have carried write access to every repository
on the account. A read-only deploy key is scoped to one repository — confirmed by
verifying that pushing with it is refused.

**A `git add -A` once committed an accidental deletion.** The PRD file was
removed from the working tree and the sweep staged the deletion. It was restored
byte-for-byte from history. Staging everything blindly is convenient and
occasionally wrong; `git status` before committing catches it.

---

## 11. Conclusion

Phase 1 is complete and deployed. Every in-scope PRD requirement is met, the
non-functional targets — 15+ FPS, portability, graceful error handling, modular
code — are satisfied, and four features beyond the specification were delivered:
IP/mobile/CCTV camera support, object tracking, target lock, and saved camera
profiles.

The system is reproducible from a clean machine in five commands, and the web UI
is deployed on free hosting, serving from a private repository. The hosted demo
also accepts the viewer's own phone camera through the browser, which is the one
form of live input a hosted server can legitimately offer.

[NEXT_PHASES.md](NEXT_PHASES.md) sets out what could be built next, sized and
prioritised. The single highest-value item is an alert system: EyeSpot currently
shows you what it sees while you watch, and alerts are what would turn it into
something that watches on your behalf.
