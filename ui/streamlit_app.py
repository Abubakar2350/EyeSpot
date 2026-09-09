"""
streamlit_app.py — Optional browser-based UI (PRD Phase 7).

Chalane ka tareeqa (project root se):
    streamlit run ui/streamlit_app.py

Features:
  - Image ya video file upload kar ke detection
  - Local webcam par live detection (jis machine par app chal rahi hai)
  - Confidence slider, class filter dropdown, model selector
  - Result download karne ka button
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Project root ko import path mein daalein taake `config`/`detector` mil jayein
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from detector.detect import Detector, count_by_label  # noqa: E402
from detector.model_loader import ModelLoadError  # noqa: E402
from utils.cameras import (  # noqa: E402
    CAMERA_BRANDS,
    MOBILE_APPS,
    CameraStoreError,
    build_url,
    load_cameras,
    save_camera,
)
from utils.fps_counter import FPSCounter  # noqa: E402
from utils.save_output import (  # noqa: E402
    DetectionLogger,
    VideoWriter,
    transcode_for_browser,
)

st.set_page_config(page_title="YOLO Real-Time Detection", page_icon="🎯", layout="wide")

MODEL_CHOICES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"]


@st.cache_resource(show_spinner="Model load ho raha hai...")
def get_detector(model_name: str, device: str | None):
    """
    Model ko cache karo taake har slider change par dobara load na ho.
    (Streamlit poora script re-run karta hai har interaction par.)
    """
    return Detector(model_name=model_name, device=device)


def sidebar_controls() -> dict:
    st.sidebar.title("⚙️ Settings")

    model_name = st.sidebar.selectbox("Model", MODEL_CHOICES, index=0)
    device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

    st.sidebar.divider()
    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, config.CONFIDENCE_THRESHOLD, 0.05)
    iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.9, config.IOU_THRESHOLD, 0.05)
    imgsz = st.sidebar.select_slider("Image size", [320, 416, 512, 640, 800], value=config.IMAGE_SIZE)

    return {
        "model_name": model_name,
        "device": None if device == "auto" else device,
        "confidence": confidence,
        "iou": iou,
        "imgsz": imgsz,
    }


def apply_settings(detector: Detector, settings: dict, class_filter: list[str]) -> None:
    """Cached detector par sliders ki values lagao (model reload kiye bagair)."""
    detector.confidence = settings["confidence"]
    detector.iou = settings["iou"]
    detector.image_size = settings["imgsz"]
    detector.class_filter_names = class_filter
    detector.class_ids = (
        [i for i, name in detector.model.names.items() if name in class_filter] or None
    )


def show_counts(counts: dict[str, int], placeholder) -> None:
    if not counts:
        placeholder.info("Koi object detect nahi hua.")
        return
    placeholder.dataframe(
        {"Object": list(counts.keys()), "Count": list(counts.values())},
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_image(detector: Detector) -> None:
    uploaded = st.file_uploader("Image upload karein", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Detection shuru karne ke liye ek image upload karein.")
        return

    data = np.frombuffer(uploaded.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Ye image read nahi ho saki — file corrupt ho sakti hai.")
        return

    start = time.perf_counter()
    detections = detector.detect(frame)
    elapsed = time.perf_counter() - start
    counts = count_by_label(detections)
    detector.draw(frame, detections)

    left, right = st.columns([3, 1])
    with left:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
    with right:
        st.metric("Objects", len(detections))
        st.metric("Inference", f"{elapsed * 1000:.0f} ms")
        show_counts(counts, st.empty())

    ok, buffer = cv2.imencode(".png", frame)
    if ok:
        st.download_button(
            "⬇️ Download result",
            buffer.tobytes(),
            file_name=f"detected_{uploaded.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )


def show_video_result(result: dict) -> None:
    """
    Process ho chuki video ka result dikhao — player, summary, aur downloads.

    Ye session_state se aata hai, isliye slider hilane ya tab badalne par
    video **dobara process nahi hoti** — bas yehi result phir se dikh jaata hai.
    """
    st.success(
        f"✅ Ho gaya — {result['frames']} frames, average {result['fps']:.1f} FPS "
        f"({result['duration']:.0f}s mein)"
    )

    video_path = Path(result["video"])
    playable = result.get("playable")

    if playable and Path(playable).exists():
        st.video(str(playable))
    else:
        st.info(
            "Browser mein preview nahi ho sakti (H.264 encoder mojood nahi). "
            "Neeche se download kar ke VLC ya kisi bhi player mein dekh lein."
        )

    st.subheader("Poori video mein kya kya mila")
    summary = result["summary"]
    if summary:
        st.dataframe(
            {
                "Object": [row["label"] for row in summary],
                "Kitne frames mein": [row["frames"] for row in summary],
                "Ek frame mein max": [row["max_in_frame"] for row in summary],
                "Unique objects": [row["unique"] for row in summary],
            },
            width="stretch",
            hide_index=True,
        )
        if not result["tracking"]:
            st.caption(
                "💡 **Unique objects** ginne ke liye sidebar se **Object tracking** on karein — "
                "warna ek hi cheez har frame mein dobara gini jaati hai."
            )
    else:
        st.info("Poori video mein koi object detect nahi hua. Confidence threshold kam kar ke dekhein.")

    # H.264 version chhoti hoti hai aur har player/browser mein chalti hai,
    # isliye download ke liye wohi behtar hai. Na bani ho to asli file de do.
    download_path = Path(playable) if playable and Path(playable).exists() else video_path

    col1, col2 = st.columns(2)
    with col1:
        if download_path.exists():
            size_mb = download_path.stat().st_size / 1e6
            st.download_button(
                f"⬇️ Detected video download karein ({size_mb:.1f} MB)",
                download_path.read_bytes(),
                file_name=download_path.name,
                mime="video/mp4",
                width="stretch",
            )
    with col2:
        csv_path = Path(result["csv"])
        if csv_path.exists():
            st.download_button(
                "⬇️ Detection log (CSV)",
                csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                width="stretch",
            )

    st.caption(f"Files yahan bhi mehfooz hain: `{video_path.parent}`")


def tab_video(detector: Detector, tracking: bool = False) -> None:
    uploaded = st.file_uploader("Video upload karein", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("Ek video file upload karein (.mp4 / .avi).")
        return

    # Har upload ki apni pehchan. Nayi file aaye to purana result nahi dikhana.
    job_key = f"{uploaded.name}:{uploaded.size}"
    cached = st.session_state.get("video_result")

    # Ek baar process ho chuki hai? To dobara mat chalao — wohi result dikhao.
    # (Streamlit har interaction par poora script dobara chalata hai; is check
    #  ke bagair har slider par video shuru se process hone lagti hai.)
    if cached and cached.get("key") == job_key:
        show_video_result(cached)
        st.divider()
        st.caption("Settings badli hain aur naya result chahiye?")
        if st.button("🔄 Dobara process karein"):
            st.session_state.pop("video_result", None)
            st.rerun()
        return

    if not st.button("▶️ Start detection", type="primary"):
        return

    # Streamlit upload memory mein hota hai; OpenCV ko file path chahiye
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        Path(video_path).unlink(missing_ok=True)
        st.error("Video open nahi ho saki (unsupported codec?).")
        return

    if tracking:
        detector.reset_tracker()  # nayi video = IDs 1 se dobara shuru

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    stem = Path(uploaded.name).stem

    frame_slot = st.empty()
    stats_slot = st.sidebar.empty()
    counts_slot = st.sidebar.empty()
    progress = st.progress(0.0, text="Processing...")
    fps_counter = FPSCounter()

    writer: VideoWriter | None = None
    det_logger = DetectionLogger(prefix=f"{stem}_detections")

    # Poori video ka hisaab — sirf aakhri frame ka nahi
    frames_seen: dict[str, int] = {}
    max_in_frame: dict[str, int] = {}
    unique_ids: dict[str, set] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.track(frame) if tracking else detector.detect(frame)
            counts = count_by_label(detections)
            detector.draw(frame, detections)
            fps_counter.update()

            for label, count in counts.items():
                frames_seen[label] = frames_seen.get(label, 0) + count
                max_in_frame[label] = max(max_in_frame.get(label, 0), count)
            for det in detections:
                if det.track_id is not None:
                    unique_ids.setdefault(det.label, set()).add(det.track_id)

            det_logger.log(fps_counter.frame_count, detections)

            if writer is None:
                h, w = frame.shape[:2]
                writer = VideoWriter(
                    fps=src_fps if src_fps and src_fps > 0 else 25.0,
                    frame_size=(w, h),
                    prefix=f"{stem}_detected",
                )
            writer.write(frame)

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            stats_slot.metric("FPS", f"{fps_counter.value:.1f}", f"{len(detections)} objects")
            show_counts(counts, counts_slot)

            if total:
                progress.progress(
                    min(fps_counter.frame_count / total, 1.0),
                    text=f"Frame {fps_counter.frame_count}/{total}",
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        det_logger.close()
        Path(video_path).unlink(missing_ok=True)

    progress.empty()
    frame_slot.empty()

    if writer is None:
        st.error("Video se ek bhi frame nahi mila.")
        return

    with st.spinner("Video ko browser ke liye tayyar kiya ja raha hai..."):
        playable = transcode_for_browser(writer.path)

    summary = sorted(
        (
            {
                "label": label,
                "frames": count,
                "max_in_frame": max_in_frame.get(label, 0),
                "unique": len(unique_ids.get(label, ())) if tracking else "—",
            }
            for label, count in frames_seen.items()
        ),
        key=lambda row: -row["frames"],
    )

    st.session_state["video_result"] = {
        "key": job_key,
        "video": str(writer.path),
        "playable": str(playable) if playable else None,
        "csv": str(det_logger.path),
        "frames": fps_counter.frame_count,
        "fps": fps_counter.average,
        "duration": fps_counter.elapsed,
        "tracking": tracking,
        "summary": summary,
    }
    st.rerun()  # ab result view dikhao (aur loop dobara na chale)


def camera_url_builder() -> str | None:
    """
    Brand chun kar URL bana lene wala helper — RTSP ka format yaad rakhne ki
    zaroorat nahi. Bana hua URL return karta hai (ya None).
    """
    with st.expander("🔧 URL builder — brand se URL bana lein"):
        brands = ["-- select --", *sorted(MOBILE_APPS), *sorted(CAMERA_BRANDS)]
        brand = st.selectbox(
            "Camera / app",
            brands,
            help="Mobile apps (ip-webcam, droidcam, iriun) HTTP par stream deti hain; CCTV brands RTSP par.",
        )
        if brand == "-- select --":
            return None

        is_mobile = brand in MOBILE_APPS
        col1, col2 = st.columns(2)
        with col1:
            ip = st.text_input("Camera / phone IP", placeholder="192.168.1.5").strip()
            user = st.text_input("Username", value="" if is_mobile else "admin").strip()
        with col2:
            default_port = (MOBILE_APPS if is_mobile else CAMERA_BRANDS)[brand]["port"]
            port = int(st.number_input("Port", min_value=1, max_value=65535, value=default_port))
            password = st.text_input("Password", type="password")

        quality = "sub"
        if not is_mobile:
            quality = st.radio(
                "Stream quality",
                ["sub", "main"],
                horizontal=True,
                help="sub = chhoti resolution (detection ke liye kaafi tez). main = full resolution.",
            )

        if not ip:
            st.caption("IP daalein to URL ban jayega.")
            return None

        try:
            url = build_url(brand, ip, user=user or None, password=password or None,
                            port=port, quality=quality)
        except ValueError as exc:
            st.error(str(exc))
            return None

        st.code(url, language="text")
        st.caption("Ye URL neeche box mein copy karein — ya pehle VLC mein test kar lein.")
        return url


def save_camera_form(url: str, default_kind: str) -> None:
    """URL ko naam de kar cameras.json mein save karne ka chhota form."""
    with st.expander("💾 Is camera ko naam de kar save karein"):
        col1, col2 = st.columns([1, 2])
        with col1:
            name = st.text_input("Naam", placeholder="gate", key=f"savename_{default_kind}").strip()
        with col2:
            note = st.text_input("Note (optional)", placeholder="front gate ka camera",
                                 key=f"savenote_{default_kind}").strip()
        if st.button("Save camera", key=f"savebtn_{default_kind}"):
            if not name:
                st.warning("Pehle naam daalein.")
                return
            try:
                profile = save_camera(name, url, kind=default_kind, note=note)
            except CameraStoreError as exc:
                st.error(str(exc))
                return
            st.success(f"Saved: **{profile.name}** → `{profile.safe_url}`  •  CLI: `python main.py --camera {profile.name}`")


def tab_live(detector: Detector, tracking: bool) -> None:
    """Saved camera, local webcam, ya network camera (phone / CCTV / IP camera)."""
    try:
        saved = load_cameras()
    except CameraStoreError as exc:
        st.warning(str(exc))
        saved = {}

    choices = ["Local webcam", "Phone / CCTV / IP camera (URL)"]
    if saved:
        choices.insert(0, "Saved camera")
    source_type = st.radio("Camera type", choices, horizontal=True)

    if source_type == "Saved camera":
        name = st.selectbox(
            "Camera",
            list(saved),
            format_func=lambda n: f"{n}  ·  {saved[n].kind}",
        )
        profile = saved[name]
        st.caption(f"`{profile.safe_url}`" + (f"  —  {profile.note}" if profile.note else ""))
        # "0" jaisi entry USB webcam hai — OpenCV ko int chahiye
        source: int | str = int(profile.url) if profile.url.isdigit() else profile.url

    elif source_type == "Local webcam":
        st.caption(
            "Note: webcam us machine ka use hota hai jahan ye app chal rahi hai "
            "(remote server par deploy karein to browser ka camera nahi milega)."
        )
        source = int(
            st.number_input("Camera index", min_value=0, max_value=10, value=config.CAMERA_INDEX)
        )
        save_camera_form(str(source), "usb")

    else:
        built = camera_url_builder()
        source = st.text_input(
            "Stream URL",
            value=built or "",
            placeholder="http://192.168.1.5:8080/video   ya   rtsp://user:pass@192.168.1.10:554/stream1",
            help="Phone: IP Webcam app ka URL. CCTV/NVR: RTSP URL.",
        ).strip()
        st.caption(
            "Phone aur PC same Wi-Fi par hone chahiye. URL pehle VLC mein test kar lein "
            "(Media → Open Network Stream)."
        )
        if not source:
            st.info("Stream URL daalein.")
            return
        save_camera_form(source, "cctv" if source.lower().startswith("rtsp://") else "mobile")

    run = st.checkbox("🎥 Start live detection")
    if not run:
        st.info("Live detection shuru karne ke liye checkbox on karein.")
        return

    is_stream = isinstance(source, str)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_stream else cv2.VideoCapture(source)
    if is_stream:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.STREAM_BUFFER_SIZE)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        cap.release()
        if is_stream:
            st.error(
                "Stream connect nahi ho saka. URL, Wi-Fi network, aur "
                "username/password check karein — pehle VLC mein test karein."
            )
        else:
            st.error(
                f"Webcam (index {source}) open nahi hui. Doosra index try karein, "
                "ya check karein ke koi aur app camera use to nahi kar rahi."
            )
        return

    frame_slot = st.empty()
    stats_slot = st.sidebar.empty()
    counts_slot = st.sidebar.empty()
    fps_counter = FPSCounter()

    try:
        # Checkbox off hote hi Streamlit script re-run karta hai aur loop khatam ho jaata hai
        while True:
            ok, frame = cap.read()
            if not ok:
                st.warning("Frame nahi mila — camera disconnect ho gaya?")
                break

            detections = detector.track(frame) if tracking else detector.detect(frame)
            counts = count_by_label(detections)
            detector.draw(frame, detections)
            fps_counter.update()

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            stats_slot.metric("FPS", f"{fps_counter.value:.1f}", f"{len(detections)} objects")
            show_counts(counts, counts_slot)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🎯 YOLO Real-Time Object Detection")

    settings = sidebar_controls()

    try:
        detector = get_detector(settings["model_name"], settings["device"])
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.divider()
    class_filter = st.sidebar.multiselect(
        "Class filter (khali = saari classes)",
        options=detector.class_names,
        default=[],
    )
    apply_settings(detector, settings, class_filter)

    tracking = st.sidebar.toggle(
        "Object tracking",
        value=False,
        help="Har object ko persistent ID deta hai jo frames ke darmiyan same rehti hai.",
    )
    st.sidebar.caption(f"Device: `{detector.device}`  •  Classes: {len(detector.class_names)}")

    image_tab, video_tab, live_tab = st.tabs(["🖼️ Image", "🎬 Video", "📷 Live camera"])
    with image_tab:
        tab_image(detector)
    with video_tab:
        tab_video(detector, tracking)
    with live_tab:
        tab_live(detector, tracking)


if __name__ == "__main__":
    main()
