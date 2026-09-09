"""
streamlit_app.py — Optional browser-based UI.

Run it from the project root:
    streamlit run ui/streamlit_app.py

Features:
  - Detection on an uploaded image or video
  - Live detection from a local webcam, a saved camera, or a network stream
  - Confidence slider, class filter, model selector, tracking toggle
  - Downloadable results (annotated video and CSV log)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Put the project root on the import path so `config`/`detector` resolve
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
    describe,
    load_cameras,
    save_camera,
)
from utils.fps_counter import FPSCounter  # noqa: E402
from utils.save_output import (  # noqa: E402
    DetectionLogger,
    VideoWriter,
    transcode_for_browser,
)

st.set_page_config(page_title="EyeSpot — Real-Time Detection", page_icon="🎯", layout="wide")

MODEL_CHOICES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"]


def is_cloud() -> bool:
    """
    True when the app is running on a hosted server rather than a local machine.

    This matters for the live camera tab: a hosted server has no webcam of its
    own, and it cannot reach a camera sitting on the user's home network —
    192.168.x.x and 10.x.x.x are private addresses that do not exist on the
    public internet. Detecting the environment lets us explain that up front
    instead of failing with a confusing connection error.

    SPACE_ID is set by Hugging Face Spaces; EYESPOT_CLOUD can be set by hand on
    any other host.
    """
    return bool(os.getenv("SPACE_ID") or os.getenv("EYESPOT_CLOUD"))


@st.cache_resource(show_spinner="Loading model...")
def get_detector(model_name: str, device: str | None):
    """
    Cache the model so it is not reloaded on every slider change.
    (Streamlit re-runs the whole script on each interaction.)
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
    """Apply the slider values to the cached detector without reloading the model."""
    detector.confidence = settings["confidence"]
    detector.iou = settings["iou"]
    detector.image_size = settings["imgsz"]
    detector.class_filter_names = class_filter
    detector.class_ids = (
        [i for i, name in detector.model.names.items() if name in class_filter] or None
    )


def show_counts(counts: dict[str, int], placeholder) -> None:
    if not counts:
        placeholder.info("No objects detected.")
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
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload an image to start detection.")
        return

    data = np.frombuffer(uploaded.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read this image — the file may be corrupt.")
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
    Show a processed video's result — player, summary and downloads.

    This comes from session_state, so moving a slider or switching tabs does
    **not** reprocess the video; the same result is simply rendered again.
    """
    st.success(
        f"✅ Done — {result['frames']} frames, average {result['fps']:.1f} FPS "
        f"(in {result['duration']:.0f}s)"
    )

    video_path = Path(result["video"])
    playable = result.get("playable")

    if playable and Path(playable).exists():
        st.video(str(playable))
    else:
        st.info(
            "In-browser preview is unavailable (no H.264 encoder found). "
            "Download the file below and play it in VLC or any other player."
        )

    st.subheader("What was found across the whole video")
    summary = result["summary"]
    if summary:
        st.dataframe(
            {
                "Object": [row["label"] for row in summary],
                "Detections": [row["frames"] for row in summary],
                "Max in one frame": [row["max_in_frame"] for row in summary],
                "Unique objects": [row["unique"] for row in summary],
            },
            width="stretch",
            hide_index=True,
        )
        if not result["tracking"]:
            st.caption(
                "💡 Turn on **Object tracking** in the sidebar to count **unique objects** — "
                "without it, the same object is counted again in every frame."
            )
    else:
        st.info("No objects were detected in this video. Try lowering the confidence threshold.")

    # The H.264 version is smaller and plays anywhere, so prefer it for the
    # download. Fall back to the original file if it was not produced.
    download_path = Path(playable) if playable and Path(playable).exists() else video_path

    col1, col2 = st.columns(2)
    with col1:
        if download_path.exists():
            size_mb = download_path.stat().st_size / 1e6
            st.download_button(
                f"⬇️ Download detected video ({size_mb:.1f} MB)",
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

    st.caption(f"These files are also saved in: `{video_path.parent}`")


def tab_video(detector: Detector, tracking: bool = False) -> None:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("Upload a video file (.mp4 / .avi).")
        return

    # Identify each upload, so a different file does not show a stale result
    job_key = f"{uploaded.name}:{uploaded.size}"
    cached = st.session_state.get("video_result")

    # Already processed? Then do not run again — show the same result.
    # (Streamlit re-runs the whole script on every interaction; without this
    #  check the video would restart from frame 0 on each slider move.)
    if cached and cached.get("key") == job_key:
        show_video_result(cached)
        st.divider()
        st.caption("Changed the settings and want a fresh result?")
        if st.button("🔄 Process again"):
            st.session_state.pop("video_result", None)
            st.rerun()
        return

    if not st.button("▶️ Start detection", type="primary"):
        return

    # Streamlit keeps uploads in memory; OpenCV needs a real file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        Path(video_path).unlink(missing_ok=True)
        st.error("Could not open the video (unsupported codec?).")
        return

    if tracking:
        detector.reset_tracker()  # a new video means IDs restart from 1

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

    # Totals for the whole video, not just the last frame
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
        st.error("Not a single frame could be read from this video.")
        return

    with st.spinner("Preparing the video for playback in the browser..."):
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
    st.rerun()  # render the result view (and stop the loop from running again)


def camera_url_builder() -> str | None:
    """
    Builds a stream URL from a brand name, so nobody has to memorise RTSP
    paths. Returns the URL it built, or None.
    """
    with st.expander("🔧 URL builder — build the URL for me", expanded=True):
        # Ask for the type first. Otherwise the phone apps (ip-webcam,
        # droidcam) get lost among the CCTV brands and it looks as though only
        # CCTV is supported.
        kind = st.radio(
            "What kind of camera is it?",
            ["📱 Mobile phone", "📹 CCTV / IP camera"],
            horizontal=True,
        )
        is_mobile = kind.startswith("📱")

        if is_mobile:
            st.caption(
                "Turning a phone into a camera needs an app on the phone. Pick yours "
                "below — if you have none, install **IP Webcam** from the Play Store "
                "or App Store (it is free)."
            )
            options = list(MOBILE_APPS)
        else:
            st.caption("Choose your CCTV or IP camera brand.")
            options = list(CAMERA_BRANDS)

        brand = st.selectbox(
            "App" if is_mobile else "Brand",
            options,
            format_func=describe,
        )

        if is_mobile:
            st.info(
                f"**{describe(brand)}** — open the app on your phone, tap **Start server**, "
                "then enter the IP address it shows (for example `192.168.1.5`) below. "
                "The phone and this PC must be on the **same Wi-Fi** network."
            )

        col1, col2 = st.columns(2)
        with col1:
            ip = st.text_input(
                "Phone IP address" if is_mobile else "Camera IP address",
                placeholder="192.168.1.5",
            ).strip()
            user = st.text_input("Username", value="" if is_mobile else "admin").strip()
        with col2:
            default_port = (MOBILE_APPS if is_mobile else CAMERA_BRANDS)[brand]["port"]
            port = int(st.number_input("Port", min_value=1, max_value=65535, value=default_port))
            password = st.text_input("Password", type="password")

        if is_mobile:
            st.caption("Phone apps usually need no username or password — leave them empty.")

        quality = "sub"
        if not is_mobile:
            quality = st.radio(
                "Stream quality",
                ["sub", "main"],
                horizontal=True,
                help="sub = lower resolution (plenty for detection, and much faster). main = full resolution.",
            )

        if not ip:
            st.caption("Enter the IP address and the URL will be built for you.")
            return None

        try:
            url = build_url(brand, ip, user=user or None, password=password or None,
                            port=port, quality=quality)
        except ValueError as exc:
            st.error(str(exc))
            return None

        st.code(url, language="text")
        st.caption("Copy this URL into the box below — or test it in VLC first.")
        return url


def save_camera_form(url: str, default_kind: str) -> None:
    """A small form for saving a URL under a name in cameras.json."""
    with st.expander("💾 Save this camera under a name"):
        col1, col2 = st.columns([1, 2])
        with col1:
            name = st.text_input("Name", placeholder="gate", key=f"savename_{default_kind}").strip()
        with col2:
            note = st.text_input("Note (optional)", placeholder="front gate camera",
                                 key=f"savenote_{default_kind}").strip()
        if st.button("Save camera", key=f"savebtn_{default_kind}"):
            if not name:
                st.warning("Enter a name first.")
                return
            try:
                profile = save_camera(name, url, kind=default_kind, note=note)
            except CameraStoreError as exc:
                st.error(str(exc))
                return
            st.success(f"Saved: **{profile.name}** → `{profile.safe_url}`  •  CLI: `python main.py --camera {profile.name}`")


def tab_live(detector: Detector, tracking: bool) -> None:
    """A saved camera, the local webcam, or a network camera (phone / CCTV / IP)."""
    try:
        saved = load_cameras()
    except CameraStoreError as exc:
        st.warning(str(exc))
        saved = {}

    hosted = is_cloud()
    if hosted:
        st.info(
            "**This is a hosted demo.** The server has no webcam of its own, and it "
            "cannot reach a camera on your home network — `192.168.x.x` addresses are "
            "private and do not exist on the public internet.\n\n"
            "Try the **🖼️ Image** and **🎬 Video** tabs instead. To use a phone camera "
            "or CCTV, run EyeSpot on your own machine — see the README for the steps. "
            "A publicly reachable stream URL will still work below."
        )

    choices = ["Phone / CCTV / IP camera (URL)"]
    if not hosted:
        choices.insert(0, "Local webcam")
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
        # An entry like "0" is a USB webcam, and OpenCV needs an int for those
        source: int | str = int(profile.url) if profile.url.isdigit() else profile.url

    elif source_type == "Local webcam":
        st.caption(
            "Note: this uses the webcam of the machine running the app "
            "(deploy it on a remote server and you will not get the browser's camera)."
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
            placeholder="http://192.168.1.5:8080/video   or   rtsp://user:pass@192.168.1.10:554/stream1",
            help="Phone: the URL from the IP Webcam app. CCTV/NVR: the RTSP URL.",
        ).strip()
        st.caption(
            "The phone and this PC must be on the same Wi-Fi network. Test the URL in "
            "VLC first (Media → Open Network Stream)."
        )
        if not source:
            st.info("Enter a stream URL.")
            return
        save_camera_form(source, "cctv" if source.lower().startswith("rtsp://") else "mobile")

    run = st.checkbox("🎥 Start live detection")
    if not run:
        st.info("Tick the checkbox to start live detection.")
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
                "Could not connect to the stream. Check the URL, the Wi-Fi network and "
                "the username/password — and test it in VLC first."
            )
        else:
            st.error(
                f"Could not open the webcam (index {source}). Try another index, or "
                "check that no other application is using the camera."
            )
        return

    frame_slot = st.empty()
    stats_slot = st.sidebar.empty()
    counts_slot = st.sidebar.empty()
    fps_counter = FPSCounter()

    try:
        # Unticking the checkbox re-runs the script, which ends this loop
        while True:
            ok, frame = cap.read()
            if not ok:
                st.warning("No frame received — did the camera disconnect?")
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
    st.title("🎯 EyeSpot — Real-Time Object Detection")

    settings = sidebar_controls()

    try:
        detector = get_detector(settings["model_name"], settings["device"])
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.divider()
    class_filter = st.sidebar.multiselect(
        "Class filter (empty = all classes)",
        options=detector.class_names,
        default=[],
    )
    apply_settings(detector, settings, class_filter)

    tracking = st.sidebar.toggle(
        "Object tracking",
        value=False,
        help="Gives every object a persistent ID that stays the same across frames.",
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
