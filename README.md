# 🎯 EyeSpot — Real-Time Object Detection & Tracking

Real-time object detection system using **YOLOv8 (Ultralytics)** + **OpenCV**.
Webcam, video file, ya image — teeno par live detection ke saath bounding boxes,
labels, confidence scores, aur FPS counter.

COCO dataset ki 80 classes support hain: `person`, `car`, `dog`, `cat`,
`cell phone`, `laptop`, `bottle`, `chair`, waghera.

> 📋 Project ka final report, doosre PC par setup karne ke steps, aur "ye kya kya
> track kar sakta hai" ka poora jawab: **[REPORT.md](REPORT.md)**

---

## ✨ Features

| Feature | Status |
|---|---|
| Webcam par real-time detection | ✅ |
| **Phone camera / CCTV / IP camera (RTSP, HTTP)** | ✅ |
| **Saved camera profiles — `--camera gate` (poora URL type karne ki zaroorat nahi)** | ✅ |
| Video file (.mp4 / .avi / .mov / .mkv) detection | ✅ |
| Single image detection | ✅ |
| Bounding box + label + confidence % | ✅ |
| Live FPS counter + per-class object counts | ✅ |
| Confidence threshold — live adjustable (`+` / `-`) | ✅ |
| Class filter (e.g. sirf `person` detect karo) | ✅ |
| **Object tracking — persistent IDs (ByteTrack / BoT-SORT)** | ✅ |
| **Target lock — ek object ko click kar ke follow karna** | ✅ |
| Pause / resume / stop | ✅ |
| Screenshot save (`s` key) | ✅ |
| Annotated video save | ✅ |
| CSV detection log (timestamp, track ID, object, confidence, box) | ✅ |
| Stream auto-reconnect (network camera drop ho jaye to) | ✅ |
| Streamlit web UI (result cache + video/CSV download) | ✅ |
| GPU (CUDA / Apple MPS) auto-detect | ✅ |

---

## 📦 Installation

### 1. Python 3.10+ chahiye

```bash
python3 --version
```

### 2. Virtual environment banayein

```bash
cd EyeSpot
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

> **Ubuntu/Debian par error aaye** (`ensurepip is not available`) to pehle:
> `sudo apt install python3-venv`

### 3. Dependencies install karein

```bash
pip install -r requirements.txt
```

CPU-only machine par (chhota, tez download ~200MB instead of ~2.5GB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Model weights (`yolov8n.pt`, ~6MB) pehli baar run par khud download ho jaate hain —
sirf internet connection chahiye.

---

## 🚀 Usage

### Webcam (default)

```bash
python main.py
```

### Video file

```bash
python main.py --source myvideo.mp4
python main.py --source myvideo.mp4 --save        # annotated video save karo
```

### Single image

```bash
python main.py --source photo.jpg
python main.py --source photo.jpg --save
```

### Sirf specific objects detect karein

```bash
python main.py --classes person                    # sirf log
python main.py --classes person car dog            # ek se zyada
python main.py --list-classes                      # saare groups + 80 classes
```

**Groups** — "animal" naam ki koi ek COCO class nahi hoti, isliye shortcuts bane hue hain:

| Group | Kya kya |
|---|---|
| `people` | person |
| `animals` | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| `pets` | cat, dog, bird |
| `vehicles` | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| `road` | person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign |

```bash
python main.py --classes animals                   # koi bhi jaanwar
python main.py --classes person animals            # log + jaanwar dono
```

Apna group banana ho to `config.py` ke `CLASS_GROUPS` mein add kar dein.

### Zyada accuracy chahiye (thoda slow)

```bash
python main.py --model yolov8s.pt      # small
python main.py --model yolov8m.pt      # medium
```

### Detection log (CSV)

```bash
python main.py --log                   # outputs/detections_*.csv
```

### Headless / batch processing (window ke bagair)

```bash
python main.py --source video.mp4 --no-display --save --log
```

---

## 📷 Camera Integration — Phone, CCTV, USB

System ko camera ki parwah nahi — usay bas ek **video stream** chahiye. Jo bhi
device RTSP ya HTTP par stream de sakta hai, wo `--source` mein chal jayega.

Teeno cases ka khulasa:

| Camera | Protocol | Source kaisa hota hai |
|---|---|---|
| Laptop / USB webcam | (koi network nahi — seedha device) | `0`, `1`, `2` ... |
| Mobile phone | **HTTP** | `http://<phone-ip>:8080/video` |
| CCTV / IP camera | **RTSP** (kuch par HTTP/MJPEG bhi) | `rtsp://user:pass@<ip>:554/...` |

---

### Camera settings — ek baar set karein, phir sirf naam

Har baar poora URL type karna takleef deh hai, aur password shell history mein
chala jaata hai. Isliye camera ko **ek naam** de kar save kar lein:

```bash
# 1. Ek baar save karein (URL chal raha hai ye bhi confirm ho jayega)
python main.py --source http://192.168.1.5:8080/video --save-camera phone --camera-note "mera Android"
python main.py --source "rtsp://admin:pass@192.168.1.10:554/stream2" --save-camera gate
python main.py --source 0 --save-camera laptop

# 2. Ab bas naam kaafi hai
python main.py --camera phone
python main.py --camera gate --track --lock person
python main.py --camera laptop

# Saari saved cameras dekhein (passwords chhupe hue)
python main.py --list-cameras
```

`--source` mein bhi seedha naam chalta hai: `python main.py --source gate`

**File hath se edit karni ho** to `cameras.example.json` ko copy kar lein:

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

> 🔒 **`cameras.json` git mein nahi jaati** (`.gitignore` mein hai) kyunke us mein
> CCTV ke passwords hote hain. Repo mein sirf `cameras.example.json` template
> jaati hai — doosre PC par usay copy kar ke apni values daal lein.

**Streamlit UI mein**: Live camera tab par saved cameras ki dropdown, aur ek
**URL builder** hai. Wo pehle poochta hai **📱 Mobile phone** ya **📹 CCTV / IP
camera** — phir usi ki list dikhata hai (phone ke liye IP Webcam / DroidCam /
Iriun, CCTV ke liye brands). Bas IP daalein, poora URL khud ban jayega aur
password bhi khud URL-encode ho jayega.

---

### Option 1 — Phone camera (sabse aasan, muft)

Phone ko wireless IP camera bana lein:

**Android — "IP Webcam" app** (Play Store, free):
1. App install kar ke kholein
2. Neeche scroll kar ke **"Start server"** dabayein
3. Screen par URL aayega, e.g. `http://192.168.1.5:8080`
4. Detection chalayein — URL ke aakhir mein `/video` lagana zaroori hai:

```bash
python main.py --source http://192.168.1.5:8080/video
```

**iPhone — "DroidCam" ya "EpocCam"**: app ka diya hua URL wese hi use karein.

> ⚠️ Phone aur PC **same Wi-Fi** par hone chahiye. Mobile data par kaam nahi karega.

### Option 2 — CCTV / NVR / IP camera (RTSP)

Har CCTV camera ka apna RTSP URL format hota hai. Aam brands:

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

**Apna exact URL kaise nikalein:**
1. Camera ka IP router ke admin page (connected devices) se lein
2. Camera ki app/web settings mein "RTSP" ya "Stream URL" dhoondein — aksar wahin likha hota hai
3. Ya brand ka naam + "RTSP URL" Google karein

**Pehle VLC mein test karein** (bohot time bachata hai): VLC → Media → Open
Network Stream → URL paste karein. VLC mein chal gaya, to yahan bhi chalega.

> 💡 `substream` / `subtype=1` / `stream2` wala URL chhoti resolution deta hai —
> detection ke liye **kaafi tez** rehta hai. Main stream (4K) CPU par slow hoga.

### Option 3 — USB webcam / laptop camera

```bash
python main.py --source 0      # pehla camera
python main.py --source 1      # doosra camera
```

Linux par available cameras: `ls /dev/video*`

### Kaunsa camera behtar hai?

| Use case | Recommendation |
|---|---|
| Seekhne / testing | Laptop webcam ya phone (IP Webcam) — muft |
| Ghar ki security | Koi bhi RTSP-capable CCTV (Tapo C200 ~sasta, Hikvision/Dahua) |
| Parindon ki nigrani | Optical zoom wala camera — parinde door aur chhote hote hain |
| Portable / outdoor | Purana Android phone hi kaafi hai |

Sabse ahem baat: **RTSP support** ho. Kuch saste "cloud-only" cameras sirf apni
app se chalte hain aur local stream nahi dete — wo yahan kaam nahi karenge.

### Stream drop ho jaye to?

Network cameras ka connection tootna aam baat hai. System khud **5 baar reconnect**
ki koshish karta hai (`config.py` mein `STREAM_RECONNECT_ATTEMPTS`), crash nahi hota.

---

## 🎯 Tracking aur Target Lock

### Detection vs Tracking — farq kya hai?

- **Detection** (default): har frame alag se dekha jaata hai. "Is frame mein 3 log hain."
  Frame 1 ka box #1 aur frame 2 ka box #1 zaroori nahi ke **wohi banda** ho.
- **Tracking** (`--track`): har object ko ek **persistent ID** milti hai jo frames
  ke darmiyan same rehti hai. "Ye wohi banda hai jo 5 second pehle andar aaya tha."

Target lock ke liye tracking **zaroori** hai — bina ID ke system ko nahi pata
ke follow kis cheez ko karna hai.

```bash
python main.py --track                    # IDs on — har box par #1, #2, #3...
python main.py --track --tracker botsort.yaml   # behtar re-identification
```

| Tracker | Kab use karein |
|---|---|
| `bytetrack.yaml` (default) | Tez, halka. Aam cases ke liye behtareen. |
| `botsort.yaml` | Object chhup kar wapas aata ho (occlusion) — behtar pehchanta hai, thoda slow. |

### Target lock — 3 tareeqe

**1. Mouse click** (sabse seedha) — video window mein object par click karein:

```bash
python main.py --track
# ...phir jis object ko follow karna hai us par click kar dein
```

**2. Class ke naam se auto-lock** — jo pehla us class ka object mile:

```bash
python main.py --track --lock person
python main.py --track --lock bird
```

**3. Track ID se** — pehle `--track` chala kar IDs dekh lein, phir:

```bash
python main.py --track --lock-id 3
```

### Lock hone par screen par kya dikhta hai

- Amber box + corner brackets locked target ke around
- Center par crosshair
- Frame ke center se target tak line, aur offset (`L171 D47` = 171px left, 47px neeche)
- Target ka trail (pichhla raasta)
- **Baaki sab objects dim** ho jaate hain
- Target ghayab ho to **"TARGET LOST"** — aur wapas aane par khud dobara pakad leta hai

`c` dabayein lock hatane ke liye, `l` auto-lock ke liye, `t` tracking on/off.

### Parinde (bird) track karna

`bird` COCO ki 80 classes mein pehle se mojood hai — koi training nahi chahiye:

```bash
python main.py --source 0 --track --lock bird --classes bird --conf 0.3
```

Parindon ke liye tips:
- `--conf 0.25` se `0.35` rakhein — parinde chhote hote hain, default 0.5 par miss ho jaate hain
- `--imgsz 960` chhoti cheezein behtar pakadta hai (thoda slow)
- `--model yolov8s.pt` ya `yolov8m.pt` — chhote objects par nano se kaafi behtar
- Zoom wala camera use karein; door ka parinda 10 pixel ka ho to koi model nahi pakdega
- Tez udte parinde par `botsort.yaml` behtar chalta hai

> **Note:** Model sirf "bird" pehchanta hai — *kaunsa* parinda (koyal, chirya,
> baaz) ye nahi bata sakta. Species-level pehchan ke liye custom training chahiye
> hogi (PRD ka Phase 2).

---

## ⌨️ Keyboard Shortcuts

Video window active hone par:

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `s` | Screenshot save (`outputs/`) |
| `p` / `Space` | Pause / Resume |
| `o` | Overlay panel on/off |
| `+` / `=` | Confidence threshold barhao |
| `-` | Confidence threshold ghatao |
| `t` | Tracking on/off |
| **Mouse click** | Us object par target lock |
| `l` | Auto-lock (sabse aham object par) |
| `c` | Lock hatao |

---

## 📈 Results ka analysis kaise karein

### 1. Live — screen par

Overlay panel (top-left) mein FPS, total objects, per-class counts, aur lock
status live update hote rehte hain.

### 2. Baad mein — CSV log

```bash
python main.py --source cctv_footage.mp4 --track --log
```

`outputs/detections_*.csv` banti hai:

| timestamp | frame | track_id | label | confidence | x1 | y1 | x2 | y2 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-08T14:19:21 | 0 | 2 | person | 0.8786 | 34 | 313 | 185 | 714 |

Is se ye sab nikal sakta hai:
- **Unique log kitne aaye** — `track_id` ke unique values ginein (bina tracking ke
  ye mumkin nahi, kyunke ek hi banda har frame mein dobara gina jayega)
- **Koi object kitni der raha** — us `track_id` ke frames ginein ÷ FPS
- **Kis waqt kya hua** — timestamp par filter karein
- **Object kis raaste gaya** — us track ID ke box centers plot karein

Excel/Google Sheets mein seedha khul jaati hai. Ya pandas se:

```python
import pandas as pd
df = pd.read_csv("outputs/detections_20260908_141920.csv")
print(df.groupby("label")["track_id"].nunique())   # har class ke unique objects
print(df[df.label == "person"].track_id.value_counts())  # har banda kitne frames
```

### 3. Annotated video

`--save` se boxes/labels wali poori video `outputs/` mein save hoti hai — reports
ya presentation ke liye.

---

## 🌐 Web UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

Browser par `http://localhost:8501` khulega. Wahan:
- **Image / Video / Live camera** — teen tabs
- **Video tab**: process hone ke baad result **cache** ho jaata hai — slider hilane
  ya tab badalne par video **dobara process nahi hoti**. Wahin par annotated video
  ka player, poori video ka summary ("kya kya mila"), aur **download buttons**
  (video + CSV log) mil jaate hain. Naya result chahiye to "🔄 Dobara process karein".
- Live camera tab par: **saved cameras ki dropdown**, local webcam, ya stream URL
- **URL builder** — brand chunein + IP daalein, poora RTSP URL khud ban jayega
  (password bhi khud URL-encode ho jaata hai)
- Camera ko wahin se naam de kar **save** karne ka button
- Confidence + IoU sliders, class filter dropdown, model selector, tracking toggle
- Result download button

> Webcam tab us machine ka camera use karta hai jahan app chal rahi hai
> (remote server par deploy karein to browser ka camera nahi milega).

---

## ⚙️ Configuration

Saari default settings [config.py](config.py) mein hain — model, thresholds,
resolution, colors, keyboard shortcuts, output paths. CLI flags in defaults ko
override karte hain.

Kuch common cheezain:

```python
MODEL_NAME = "yolov8n.pt"       # yolov8s.pt / yolov8m.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.5      # kam karein to zyada (lekin weaker) detections
IMAGE_SIZE = 640                # 320 = tez, 640 = accurate
CLASS_FILTER = []               # e.g. ["person", "car"]
CAMERA_INDEX = 0                # doosra camera = 1
```

---

## 📁 Project Structure

```
EyeSpot/
├── main.py                  # Entry point — CLI, main loop, keyboard handling
├── config.py                # Saari default settings
├── requirements.txt
├── test_smoke.py            # Quick self-test
├── cameras.example.json     # Camera settings template (cameras.json git mein nahi jaati)
│
├── detector/
│   ├── model_loader.py      # Model load + device auto-detect (CPU/CUDA/MPS)
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
├── models/                  # Downloaded weights (auto)
└── outputs/                 # Screenshots, videos, CSV logs
```

**Design:** detection logic UI se bilkul alag hai. `Detector.detect()` plain
`Detection` dataclass objects return karta hai — isliye same logic CLI, Streamlit,
ya kisi bhi future frontend (Flask/FastAPI) ke saath chal sakti hai.

---

## 🧪 Testing

```bash
python test_smoke.py
```

Ye model loading, image detection, video pipeline, FPS counter, aur output
saving — sab check karta hai.

---

## 📊 Performance

`yolov8n.pt` @ 640px, Intel CPU par measured: **~23 FPS** (PRD target: 15+ FPS ✅)

Tez chahiye to:
- `--imgsz 320` — kaafi tez, thodi accuracy kam
- `--model yolov8n.pt` — sabse chhota model (already default)
- GPU par chalayein — `--device cuda` (30-100+ FPS)

Accuracy chahiye to: `--model yolov8s.pt` ya `--imgsz 800`.

---

## 🔧 Troubleshooting

**"Webcam open nahi ho saka"**
- Camera connected hai? Linux par check: `ls /dev/video*`
- Koi doosri app (Zoom/Teams) camera use to nahi kar rahi
- Doosra index try karein: `--source 1`

**"Stream connect nahi ho saka" (phone / CCTV)**
- URL pehle **VLC** mein test karein (Media → Open Network Stream). VLC mein na
  chale to masla URL/camera ka hai, code ka nahi.
- Phone aur PC same Wi-Fi par hain?
- Phone URL ke aakhir mein `/video` lagana na bhoolein
- Password mein special characters (`@`, `:`, `/`) hain? Unhe URL-encode karein
  (`@` → `%40`)
- Camera settings mein RTSP enabled hai?

**Stream mein latency / video peechhe reh rahi hai**
- Camera ka **substream** use karein (chhoti resolution) — `subtype=1` ya `stream2`
- `--imgsz 320` se inference tez karein

**Track IDs baar baar badal rahi hain**
- `--tracker botsort.yaml` try karein (behtar re-identification)
- `--conf` thoda kam karein — detection toot rahi hogi to ID bhi toot jaati hai
- FPS bohot kam ho to tracking kamzor ho jaati hai (objects frames ke darmiyan
  zyada hil jaate hain)

**"Ultralytics installed nahi hai"**
- venv activate kiya? `pip install -r requirements.txt` dobara chalayein

**Model download fail ho raha hai**
- Internet connection check karein (pehli baar weights download hoti hain)
- Ya manually `yolov8n.pt` download kar ke `models/` folder mein rakh dein

**Video save nahi ho rahi / khali file**
- `config.py` mein `VIDEO_CODEC = "XVID"` try karein aur output `.avi` rakhein

**FPS bohot kam hai**
- `--imgsz 320` use karein, ya GPU par chalayein

---

## 🔮 Future Enhancements (Phase 2+)

- Custom dataset par apna YOLO model train karna (COCO ki 80 classes se aage)
- Alert system (person detect ho to notification / alarm / email)
- Zone ya line-crossing counting ("gate se kitne log guzre")
- Multi-camera ek sath (abhi ek waqt mein ek source)
- Cloud deployment (AWS / GCP)
- Face recognition / number plate OCR

> Tracking (ByteTrack/BoT-SORT), target lock, aur IP/mobile/CCTV camera support
> PRD mein Phase 2 the — wo **ban chuke hain**, upar dekhein.

---

## 📄 License & Credits

- Detection models: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Built as per [YOLO_Realtime_Detection_PRD.md](YOLO_Realtime_Detection_PRD.md)
