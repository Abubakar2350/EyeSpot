# PRD (Product Requirements Document)
## Real-Time Object Detection System using YOLO (Webcam/Video)

**Document Version:** 1.0
**Prepared For:** Claude Code (AI Coding Agent)
**Project Type:** Computer Vision — Real-Time Object Detection
**Date:** September 2026

---

## 1. Project Overview (Project Ka Ta'aruf)

Ye project ek **Real-Time Object Detection System** hai jo YOLO (You Only Look Once) model ka use kar ke webcam ya video feed se objects (insaan, car, cat, dog, phone, laptop, etc.) ko detect aur identify karega, wo bhi live/real-time mein bohot fast speed ke saath.

**Core Idea:** User apna webcam on kare (ya video file upload kare), aur system real-time mein har frame ke andar objects ko detect kar ke unke around bounding boxes, labels, aur confidence scores dikhaye.

---

## 2. Objectives (Maqasid)

- Ek working real-time object detection application banana jo webcam ya video input le sake.
- Pretrained YOLO model (YOLOv8 ya YOLOv11) use karna taake training se bachein aur jaldi result milay.
- Detected objects ko bounding box + label + confidence % ke sath live screen pe show karna.
- System ko modular banana taake future mein custom training ya extra features add ho sakein.
- Beginner-friendly, clean code structure jo samajhna aasan ho.

---

## 3. Scope (Project Ka Daira)

### In-Scope (Ye Cheezain Include Hongi):
- Real-time webcam-based object detection
- Video file (.mp4, .avi) upload aur detection
- Pretrained YOLO model integration (COCO dataset — 80 classes: person, car, dog, cat, phone, bottle, etc.)
- Live FPS counter
- Detected objects ki list/count on screen
- Simple GUI ya web interface (Streamlit/OpenCV window)
- Screenshot/save detection result option

### Out-of-Scope (Is Phase Mein Nahi Hoga):
- Custom object training (apna khud ka dataset train karna) — ye Phase 2 mein hoga
- Mobile app version
- Cloud deployment (initial version sirf local machine pe chalega)
- Multi-camera support

---

## 4. Target User

- Beginner/Intermediate developer jo Computer Vision seekh raha ho
- Student jo apna AI/ML portfolio project banana chahta ho
- Kisi bhi chhoti application (security camera demo, attendance system base, etc.) ke liye base project

---

## 5. Tech Stack (Recommended)

| Component | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| Object Detection Model | YOLOv8 (Ultralytics) — best balance of speed & ease of use |
| Computer Vision Library | OpenCV (cv2) |
| Deep Learning Backend | PyTorch (Ultralytics dependency) |
| GUI/Interface (Optional) | Streamlit (web-based) ya sirf OpenCV window |
| Video/Webcam Input | OpenCV VideoCapture |
| Package Manager | pip / venv |

> **Note:** YOLOv8 is recommended over YOLOv5/v11 kyunke Ultralytics library bohot beginner-friendly hai aur `pip install ultralytics` se seedha kaam shuru ho jata hai. Agar latest chahiye to YOLOv11 bhi same library se available hai.

---

## 6. Functional Requirements

### 6.1 Webcam Real-Time Detection
- App laptop/PC ke default webcam se live feed lega
- Har frame par YOLO model run hoga
- Detected objects ke around bounding box draw honge
- Har box ke upar class name + confidence score (%) likha hoga
- Live video window mein FPS counter show hoga

### 6.2 Video File Detection
- User apni local video file (.mp4/.avi) select kar sakay
- System pura video process kare frame-by-frame
- Output video (with detections) save karne ka option ho

### 6.3 Detection Controls
- Confidence threshold adjust karne ka option (e.g. slider ya config value — default 0.5)
- Specific classes ko filter karne ka option (e.g. sirf "person" detect kare)
- Detection ko pause/resume/stop karne ka option

### 6.4 Output & Reporting
- Real-time detected objects ki list (naam + count) screen ke side mein show ho
- Screenshot save karne ka button/key (jaise 's' key press)
- Optional: Detection log CSV file mein save ho (timestamp, object, confidence)

### 6.5 (Optional/Phase 2) Web Interface
- Streamlit based simple web UI jahan user browser se webcam/video access kar sakay
- Start/Stop button, confidence slider, class filter dropdown

---

## 7. Non-Functional Requirements

- **Performance:** Kam se kam 15-20 FPS real-time detection (CPU pe), aur 30+ FPS agar GPU available ho
- **Usability:** Simple, clean interface — beginner bhi use kar sakay bina extra setup ke
- **Portability:** Windows aur Linux dono pe chalna chahiye
- **Reliability:** Webcam disconnect/error hone par graceful error message aaye, app crash na ho
- **Code Quality:** Modular structure (separate files for detection logic, UI, utils)

---

## 8. Suggested Project Structure

```
yolo-realtime-detection/
│
├── main.py                 # Entry point - webcam/video detection start
├── requirements.txt        # Dependencies list
├── config.py                # Confidence threshold, model path, class filters
│
├── detector/
│   ├── __init__.py
│   ├── model_loader.py      # YOLO model load karne wala code
│   └── detect.py            # Detection logic (frame process karna)
│
├── utils/
│   ├── __init__.py
│   ├── fps_counter.py       # FPS calculate karna
│   └── save_output.py       # Screenshot/video/log save karna
│
├── ui/
│   └── streamlit_app.py     # (Optional) Web interface
│
├── outputs/                 # Saved screenshots/videos/logs yahan store hongi
│
└── README.md
```

---

## 9. Development Milestones (Roadmap)

| Phase | Task | Deliverable |
|---|---|---|
| Phase 1 | Environment setup (Python, venv, dependencies install) | Working environment |
| Phase 2 | Pretrained YOLOv8 model load karna aur ek static image pe test karna | Successful detection on sample image |
| Phase 3 | Webcam se live video feed lena aur real-time detection lagana | Working real-time webcam detection |
| Phase 4 | FPS counter, bounding boxes, labels UI pe polish karna | Clean visual output |
| Phase 5 | Video file input support add karna | Video detection feature |
| Phase 6 | Screenshot/log saving feature | Output saving working |
| Phase 7 (Optional) | Streamlit web UI banana | Browser-based interface |
| Phase 8 | Testing, bug fixing, README documentation | Final polished project |

---

## 10. Dependencies (requirements.txt ka andaza)

```
ultralytics
opencv-python
numpy
streamlit   # (agar web UI banani ho to)
```

---

## 11. Success Criteria

- App bina crash hue webcam se live detection kare
- Common objects (person, car, cat, dog, phone, laptop, etc.) correctly detect hon
- Real-time performance acceptable ho (minimum ~15 FPS on CPU)
- Code clean, modular, aur samajhne mein aasan ho
- README mein clear instructions hon ke project kaise run karein

---

## 12. Future Enhancements (Phase 2+)

- Custom dataset pe apna khud ka YOLO model train karna (custom objects detect karne ke liye)
- Alert system (jaise: agar "person" detect ho to notification/alarm)
- Multi-camera / IP camera support
- Cloud deployment (AWS/GCP) taake remote se access ho sakay
- Object tracking add karna (sirf detect nahi, track bhi kare — jaise DeepSORT)
- Mobile app version (Android/iOS)

---

## 13. Instructions for Claude Code

Jab is PRD ko Claude Code mein use karein, to ye steps follow karayein:

1. Pehle Phase 1-4 (core webcam detection) implement karayein — ye MVP hai.
2. Har phase ke baad test karayein ke code chal raha hai ya nahi.
3. Pretrained `yolov8n.pt` (nano — sabse fast, lightweight) model se start karayein taake beginner machine pe bhi smoothly chale.
4. Agar accuracy chahiye speed ki jagah, to baad mein `yolov8s.pt` ya `yolov8m.pt` try karein.
5. Code comments Roman Urdu ya English, dono chalega — lekin consistent rakhna.

---

**End of PRD**
