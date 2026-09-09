# 📋 EyeSpot — Final Project Report
## Real-Time Object Detection & Tracking System (YOLOv8)

**Report date:** 9 September 2026
**Status:** ✅ Phase 1 complete — sab PRD requirements poori, plus 4 extra features
**Repo:** `EyeSpot`
**Reference:** [YOLO_Realtime_Detection_PRD.md](YOLO_Realtime_Detection_PRD.md)

---

## 1. Executive Summary

Ek working real-time object detection **aur tracking** system ban chuka hai jo:

- **80 objects** (person, car, dog, cat, bird, phone, laptop, bag, bottle …) live pehchanta hai
- **4 tarah ke input** leta hai — laptop/USB webcam, mobile phone (HTTP), CCTV/IP camera (RTSP), aur video/image files
- Sirf detect nahi karta — har object ko **persistent ID** deta hai aur ek object par **lock** kar ke usay follow karta hai
- Do interfaces deta hai — **CLI** (tez, keyboard shortcuts ke sath) aur **Streamlit web UI** (browser mein, HTTP par)
- Results **save** karta hai — screenshots, annotated video, aur analysis ke liye CSV log

Test suite: **16/16 pass**. Inference speed CPU par **~55 FPS** (PRD target: 15+ FPS).

---

## 2. PRD Requirements — Status

### In-scope (PRD section 3)

| # | Requirement | Status | Kahan |
|---|---|---|---|
| 6.1 | Webcam real-time detection | ✅ | `main.py --source 0` |
| 6.1 | Bounding box + label + confidence % | ✅ | `detector/detect.py` → `draw()` |
| 6.1 | Live FPS counter | ✅ | `utils/fps_counter.py` |
| 6.2 | Video file (.mp4/.avi/.mov/.mkv) | ✅ | `--source video.mp4` |
| 6.2 | Output video save | ✅ | `--save` |
| 6.3 | Confidence threshold adjust | ✅ | `--conf`, live `+`/`-` keys |
| 6.3 | Class filter | ✅ | `--classes person car` |
| 6.3 | Pause / resume / stop | ✅ | `p` / `Space` / `q` |
| 6.4 | On-screen object list + counts | ✅ | `utils/overlay.py` |
| 6.4 | Screenshot save (`s` key) | ✅ | `utils/save_output.py` |
| 6.4 | CSV detection log | ✅ | `--log` |
| 6.5 | Streamlit web UI | ✅ | `ui/streamlit_app.py` |
| 7 | 15+ FPS on CPU | ✅ | ~55 FPS inference (neeche §4) |
| 7 | Windows + Linux portability | ✅ | koi OS-specific code nahi |
| 7 | Graceful errors (crash na ho) | ✅ | har failure par saaf message + hal |
| 7 | Modular code | ✅ | detector / utils / ui alag |

### PRD mein "Phase 2 / out-of-scope" tha — lekin ban gaya

| Feature | Status | Kyun ahem hai |
|---|---|---|
| **IP camera / mobile / CCTV support** | ✅ | RTSP + HTTP streams, auto-reconnect ke sath |
| **Object tracking (ByteTrack / BoT-SORT)** | ✅ | Persistent IDs — "ye wohi banda hai" |
| **Target lock** | ✅ | Ek object chun kar usay follow karna (trail, LOST/re-acquire) |
| **Saved camera profiles** | ✅ | `--camera gate` — poora URL yaad rakhne ki zaroorat nahi |

### Ab bhi baaqi (Phase 2)

| Feature | Kyun nahi hai |
|---|---|
| Custom dataset training | Alag project hai — dataset labelling + training pipeline chahiye |
| Alert / notification system | Abhi sirf screen + CSV; email/SMS/buzzer add karna hoga |
| Cloud deployment | Local machine ke liye design kiya gaya |
| Multi-camera ek sath | Ek waqt mein ek source; parallel ke liye multi-threading chahiye |
| Face recognition | YOLO "person" batata hai, "kaun person" nahi |

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
  ✅ Overlay drawing
  ✅ Output saving — screenshot, video, CSV
  ✅ Video pipeline
  ✅ CLI parsing
  ✅ Stream URLs (phone/CCTV) — 4 URL formats + password masking
  ✅ Camera profiles (cameras.json)
  ✅ Object tracking (persistent IDs)
  ✅ Target lock
  ✅ Target lock drawing

  16 passed, 0 failed
```

Streamlit UI alag se boot kar ke check ki gayi — HTTP 200, koi error nahi.

---

## 4. Performance (measured)

Test machine: Linux 6.8, Python 3.12.3, x86_64 **CPU (16 cores)**, koi GPU nahi.
Input 1280×720, model `yolov8n.pt`, `imgsz=640`:

| Mode | Speed | Per frame |
|---|---|---|
| Detection | **55.2 FPS** | 18 ms |
| Tracking (ByteTrack) | **52.9 FPS** | 19 ms |

> Ye **pure inference** ka number hai. Camera se frame lena, drawing, aur window
> par dikhana milaa kar end-to-end FPS is se kam rehti hai (aam tor par 20–30),
> jo PRD ke 15 FPS target se phir bhi upar hai.

Tez karna ho: `--imgsz 320`. Accurate chahiye: `--model yolov8s.pt` ya `--imgsz 800`.
GPU (`--device cuda`) par 30–100+ FPS.

---

## 5. Abhi ye kya kya track kar sakta hai?

### ✅ Kar sakta hai

**80 COCO classes** — koi training nahi chahiye:

> person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
> fire hydrant, stop sign, parking meter, bench, **bird**, cat, dog, horse, sheep, cow,
> elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
> skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
> surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
> apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
> potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
> cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase,
> scissors, teddy bear, hair drier, toothbrush

Poori list: `python main.py --list-classes`

Har class ke liye:

| Kaam | Kaise |
|---|---|
| Detect karna | `python main.py --camera gate` |
| Sirf ek cheez dekhna | `--classes person` |
| Poora group dekhna | `--classes animals` / `vehicles` / `pets` / `road` |
| Persistent ID dena (tracking) | `--track` → har box par `#1`, `#2` … |
| Ek object ko follow karna (lock) | window mein us par **click**, ya `--lock person` |
| Kitne unique log/gaadiyan aayin | `--track --log` → CSV mein `track_id` ke unique values |
| Koi object kitni der raha | us `track_id` ke frames ÷ FPS |
| Kis raaste gaya | us track ID ke box centers (CSV mein x1,y1,x2,y2) |

Lock hone par: amber box + corner brackets, crosshair, frame-center se offset
(`L171 D47`), trail, baaki objects dim, aur ghayab hone par **TARGET LOST** —
wapas aane par khud dobara pakad leta hai.

### ❌ Nahi kar sakta (Phase 2)

| Cheez | Wajah |
|---|---|
| **Kaun sa banda** hai (face recognition) | Model sirf "person" janta hai |
| **Kaun sa parinda** (koyal / chirya / baaz) | Model sirf "bird" janta hai — species ke liye custom training |
| Number plate parhna | OCR alag cheez hai |
| Custom objects (apni machine ka part waghera) | Apna dataset train karna hoga |
| **Do cameras ke darmiyan** wohi banda pehchanna | Track IDs sirf ek stream ke andar valid hain |
| Line-crossing / zone alerts | Abhi sirf detection + lock; counting logic add karni hogi |

> Ek aur baat: track ID **usi session** ke liye hai. App band kar ke dobara chalayein
> to IDs 1 se shuru ho jayengi.

---

## 6. Doosre PC par kaise chalayein (git push ke baad)

### 6.1 Is PC se git par push karna

Repo `Abubakar2350/EyeSpot` (private) par push ho chuki hai. Aage se sirf:

```bash
git add .
git commit -m "kya change kiya"
git push
```

Kabhi shuru se karna ho to:

```bash
cd EyeSpot
git init
git add .
git commit -m "EyeSpot — real-time detection, tracking, camera profiles"
git branch -M main
git remote add origin https://github.com/Abubakar2350/EyeSpot.git
git push -u origin main
```

**Git mein kya jaata hai aur kya nahi** (`.gitignore`):

| Jaata hai ✅ | Nahi jaata ❌ | Kyun nahi |
|---|---|---|
| Saara code (`main.py`, `detector/`, `utils/`, `ui/`) | `.venv/` | Har PC apna banata hai |
| `requirements.txt` | `models/*.pt` | Pehli run par khud download ho jaate hain |
| `README.md`, `REPORT.md`, PRD | `outputs/` | Generated results |
| `cameras.example.json` | **`cameras.json`** | **Is mein CCTV ke passwords hote hain** |
| `test_smoke.py` | `__pycache__/`, `.vscode/` | Machine-specific |

> 🔒 Ye ahem hai: apne CCTV ka password kabhi git par push na karein. Isi liye
> `cameras.json` ignore ki gayi hai aur repo mein sirf `cameras.example.json` jaati hai.

### 6.2 Naye PC par — 5 commands

```bash
git clone https://github.com/Abubakar2350/EyeSpot.git
cd EyeSpot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python test_smoke.py               # sab theek hai ya nahi
python main.py                     # webcam par chal padega
```

Pehli baar `python main.py` chalane par `yolov8n.pt` (~6 MB) khud download hota
hai — **internet chahiye**. Uske baad offline bhi chalta rahega.

### 6.3 OS ke hisaab se choti si baatein

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
- `python3` nahi, sirf `python` likhein
- PowerShell activate na kare to: `Set-ExecutionPolicy -Scope Process RemoteSigned`

**Ubuntu / Debian**

```bash
sudo apt install python3-venv python3-pip     # venv error aaye to
sudo apt install libgl1 libglib2.0-0          # OpenCV window ke liye
```

**CPU-only machine** (torch ka 2.5 GB download bachane ke liye) — `pip install -r requirements.txt` se **pehle**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU (NVIDIA) machine** — kuch extra nahi chahiye, khud detect ho jaata hai.
Confirm: run par `device: cuda` likha aana chahiye.

### 6.4 Naye PC par camera settings

`cameras.json` git se nahi aati (passwords ki wajah se), isliye ek baar banayein:

```bash
cp cameras.example.json cameras.json      # Windows: copy cameras.example.json cameras.json
```

Phir file mein apne asli IP/password daal dein — **ya** command se save kar lein:

```bash
python main.py --source 0 --save-camera laptop
python main.py --source http://192.168.1.5:8080/video --save-camera phone
python main.py --source "rtsp://admin:pass@192.168.1.10:554/stream2" --save-camera gate

python main.py --list-cameras     # check
python main.py --camera gate      # chalao
```

> ⚠️ IP addresses **network ke sath badalte hain**. Ghar ka `192.168.1.5` office
> ke Wi-Fi par kaam nahi karega — naye network par phone app ka naya IP dekh kar
> `cameras.json` update kar lein.

### 6.5 Naye PC par pehli run — checklist

| Check | Command | Sahi jawab |
|---|---|---|
| Python version | `python --version` | 3.10 ya upar |
| venv on hai? | prompt par `(.venv)` | dikhna chahiye |
| Dependencies | `python test_smoke.py` | 16 passed |
| Model download | pehli `python main.py` | "Model ready. Classes: 80" |
| Camera list | `python main.py --list-cameras` | aapki cameras |
| Web UI | `streamlit run ui/streamlit_app.py` | browser par `localhost:8501` |

---

## 7. Camera setup — mobile, CCTV, laptop

Teeno ka farq sirf itna hai ke **source kya likhna hai** — baaqi sab (detection,
tracking, lock, save) bilkul same chalta hai.

| Camera | Protocol | Source |
|---|---|---|
| Laptop / USB webcam | seedha device | `0`, `1`, `2` … |
| Mobile phone | **HTTP** | `http://<phone-ip>:8080/video` |
| CCTV / IP camera | **RTSP** | `rtsp://user:pass@<ip>:554/...` |

### Mobile phone (HTTP) — sabse aasan

1. Android par **"IP Webcam"** app install karein (free) — iPhone par DroidCam/Iriun
2. App kholein → neeche **"Start server"**
3. Screen par IP aayega, e.g. `http://192.168.1.5:8080`
4. Aakhir mein **`/video`** lagana zaroori hai:

```bash
python main.py --source http://192.168.1.5:8080/video --save-camera phone
```

Phone aur PC **same Wi-Fi** par hone chahiye (mobile data par nahi chalega).

| App | Default port | Path |
|---|---|---|
| IP Webcam (Android) | 8080 | `/video` |
| DroidCam | 4747 | `/video` |
| Iriun | 8080 | `/video` |

### CCTV / IP camera (RTSP)

| Brand | URL format (`sub` = chhoti resolution, tez) |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| Dahua / CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream2` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_sub` |
| ONVIF (generic) | `rtsp://user:pass@IP:554/onvif2` |

URL yaad rakhne ki zaroorat nahi — **Streamlit UI ka URL builder** use karein
(brand chunein + IP daalein, URL khud ban jayega, password bhi khud encode ho jayega),
ya code mein:

```python
from utils.cameras import build_url
build_url("tplink", "192.168.1.10", user="admin", password="pa@ss")
# 'rtsp://admin:pa%40ss@192.168.1.10:554/stream2'
```

**Camera ka IP kahan se milega:** router ke admin page (connected devices) se, ya
camera ki apni app ki settings se.

**Pehle VLC mein test karein** — Media → Open Network Stream. VLC mein chal gaya
to yahan bhi chalega; na chale to masla camera/network ka hai, code ka nahi.

> 💡 Hamesha **substream** use karein (`102`, `subtype=1`, `stream2`, `_sub`).
> Main stream 4K hoti hai — CPU par slow aur detection ke liye zaroorat bhi nahi.

### Laptop / USB webcam

```bash
python main.py --source 0        # pehla camera
python main.py --source 1        # doosra
```
Linux par available cameras: `ls /dev/video*`

### Stream ke masail

| Masla | Hal |
|---|---|
| Connect hi nahi hota | VLC mein test karein; same Wi-Fi?; RTSP enabled hai? |
| Video peechhe reh rahi hai (lag) | substream use karein + `--imgsz 320` |
| Password mein `@` ya `:` hai | URL-encode karein (`@` → `%40`) — `build_url()` khud kar deta hai |
| Beech mein toot jaata hai | Kuch nahi karna — khud 5 baar reconnect karta hai |

---

## 8. Files ka naqsha

```
EyeSpot/
├── main.py                   # Entry point — CLI, main loop, keyboard/mouse
├── config.py                 # Saari default settings
├── requirements.txt
├── test_smoke.py             # 16 self-tests
├── cameras.example.json      # Camera settings ka template (cameras.json git mein nahi jaati)
│
├── detector/
│   ├── model_loader.py       # Model load + device auto-detect (CPU/CUDA/MPS)
│   ├── detect.py             # Detector — detect(), track(), draw()
│   └── target_lock.py        # Target lock — follow, trail, LOST/re-acquire, HUD
│
├── utils/
│   ├── cameras.py            # Camera profiles + brand URL builder + password masking
│   ├── fps_counter.py        # Moving-average FPS
│   ├── overlay.py            # On-screen info panel
│   └── save_output.py        # Screenshot / video writer / CSV logger
│
├── ui/
│   └── streamlit_app.py      # Web UI (HTTP) — image/video/live tabs + camera settings
│
├── models/                   # Weights (auto-download, git mein nahi)
└── outputs/                  # Screenshots, videos, CSV (git mein nahi)
```

**Design ka usool:** detection logic UI se bilkul alag hai. `Detector.detect()`
plain `Detection` objects deta hai — isi liye CLI, Streamlit, aur koi bhi future
frontend (Flask/FastAPI/mobile) same logic use kar sakta hai.

---

## 9. Rozana ke commands (cheat sheet)

```bash
# Camera ek baar save karo
python main.py --source http://192.168.1.5:8080/video --save-camera phone
python main.py --list-cameras

# Chalao
python main.py --camera phone
python main.py --camera gate --classes person          # sirf log
python main.py --camera gate --track                   # IDs ke sath
python main.py --camera gate --track --lock person     # pehle bande par lock
python main.py --camera gate --track --log --save      # CSV + video record

# Purani footage par
python main.py --source cctv_footage.mp4 --track --log --no-display

# Web UI (browser par, HTTP)
streamlit run ui/streamlit_app.py

# Parinde
python main.py --camera phone --track --lock bird --classes bird --conf 0.3 --imgsz 960
```

Keyboard (video window): `q` quit · `s` screenshot · `p` pause · `o` overlay ·
`+`/`-` confidence · `t` tracking · **click** lock · `l` auto-lock · `c` clear lock

---

## 10. Conclusion

PRD ki saari in-scope requirements poori hain, non-functional targets (15+ FPS,
portability, graceful errors, modular code) bhi meet hue hain, aur PRD ke "Phase 2"
mein likhe teen features — **IP/mobile/CCTV camera support**, **object tracking**,
aur **target lock** — pehle hi phase mein aa gaye hain, saath ek chauthi cheez
**saved camera profiles** bhi.

Product deployable hai: `git clone` → `venv` → `pip install` → `python main.py`.

Agla logical qadam (Phase 2): alert system (person detect ho to notification),
zone/line-crossing counting, aur custom training agar COCO ki 80 classes se aage
jaana ho.
