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
from detector.accident import AccidentDetector  # noqa: E402
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
    AccidentLogger,
    DetectionLogger,
    VideoWriter,
    transcode_for_browser,
)

# Optional: live streaming from the viewer's own browser camera (WebRTC).
# The photo mode below works without it, so a missing package only hides the
# live option instead of breaking the app.
try:  # noqa: E402
    import av  # noqa: E402
    from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

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

    Three signals, in order of specificity:
      - SPACE_ID       set automatically by Hugging Face Spaces
      - EYESPOT_CLOUD  set by hand, e.g. in a Dockerfile
      - the same key in Streamlit secrets, which is how Streamlit Community
        Cloud passes configuration (it also exports secrets as env vars, but
        reading st.secrets works whichever way round it happens)
    """
    if os.getenv("SPACE_ID") or os.getenv("EYESPOT_CLOUD"):
        return True
    try:
        return bool(st.secrets.get("EYESPOT_CLOUD"))
    except Exception:
        # No secrets file at all is the normal local case, and older Streamlit
        # versions raise rather than return empty here.
        return False


def ice_servers() -> list[dict]:
    """
    ICE servers for the browser-camera WebRTC connection.

    Why a TURN server is needed and not just STUN: STUN only helps two peers
    discover their public addresses so they can talk directly. That works when
    at least one side is reachable. It fails when both sides sit behind
    restrictive NAT — which is exactly the common case here, because mobile
    networks put subscribers behind carrier-grade NAT, and a hosted server has
    its own restrictions. Neither end can accept an incoming connection, so the
    direct path never forms and the handshake simply times out.

    TURN solves it by relaying the media through a third machine that both sides
    *can* reach. It costs bandwidth, which is why it is not the default in most
    examples, and why leaving it out is such a common mistake.

    Several transports are listed on purpose. UDP is fastest; TCP on port 443
    is the one that survives firewalls strict enough to block everything that
    does not look like HTTPS.

    To use your own TURN service instead of the shared free one, put it in
    Streamlit secrets:

        [webrtc]
        turn_url = "turn:your.server:3478"
        turn_username = "user"
        turn_credential = "pass"
    """
    servers: list[dict] = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
    ]

    # A private TURN service, when one is configured
    try:
        webrtc_secrets = st.secrets.get("webrtc", {})
    except Exception:
        webrtc_secrets = {}

    if webrtc_secrets.get("turn_url"):
        servers.append({
            "urls": [webrtc_secrets["turn_url"]],
            "username": webrtc_secrets.get("turn_username", ""),
            "credential": webrtc_secrets.get("turn_credential", ""),
        })
        return servers

    # Otherwise fall back to Open Relay, a free shared TURN service. It is rate
    # limited and shared with everyone, so it is fine for a demo and not for
    # production traffic.
    servers.append({
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:80?transport=tcp",
            "turn:openrelay.metered.ca:443",
            "turns:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    })
    return servers


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

    accidents = result.get("accidents") or []
    if result.get("accident_csv") is not None:
        st.subheader("🚨 Accident detection")
        if accidents:
            st.error(f"**{len(accidents)} accident event(s) detected.**")
            st.dataframe(
                {
                    "Time": [f"{a['time']:.1f}s" for a in accidents],
                    "Frame": [a["frame"] for a in accidents],
                    "Score": [f"{a['score']:.2f}" for a in accidents],
                    "Involved": [", ".join(a["labels"]) or "—" for a in accidents],
                    "Evidence": [a["evidence"] for a in accidents],
                },
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The **Evidence** column lists the signals that fired and how strongly. "
                "No single one triggers an event on its own — several have to agree."
            )
        else:
            st.success("No accidents detected in this video.")

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

    accident_csv = result.get("accident_csv")
    if accident_csv and Path(accident_csv).exists() and accidents:
        st.download_button(
            f"⬇️ Accident report ({len(accidents)} event(s), CSV)",
            Path(accident_csv).read_bytes(),
            file_name=Path(accident_csv).name,
            mime="text/csv",
        )

    st.caption(f"These files are also saved in: `{video_path.parent}`")


def tab_video(detector: Detector, tracking: bool = False, accident_mode: bool = False) -> None:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("Upload a video file (.mp4 / .avi).")
        return

    # Identify each upload, so a different file does not show a stale result
    job_key = f"{uploaded.name}:{uploaded.size}:{tracking}:{accident_mode}"
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
    accidents = AccidentDetector() if accident_mode else None
    accident_logger = AccidentLogger(prefix=f"{stem}_accidents") if accident_mode else None

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

            if accidents is not None:
                # Drive the clock from the video's own frame rate rather than
                # the wall clock: a video processed slower or faster than real
                # time must still yield the speeds the footage actually shows.
                stamp = fps_counter.frame_count / (src_fps if src_fps and src_fps > 0 else 25.0)
                event = accidents.update(detections, fps_counter.frame_count, now=stamp)
                if event is not None:
                    accident_logger.log(event, frame)
                accidents.draw(frame, now=stamp)

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
        if accident_logger is not None:
            accident_logger.close()
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
        "accidents": [
            {
                "frame": e.frame,
                "time": e.frame / (src_fps if src_fps and src_fps > 0 else 25.0),
                "score": e.score,
                "labels": e.labels,
                "evidence": e.summary,
            }
            for e in (accidents.events if accidents else [])
        ],
        "accident_csv": str(accident_logger.path) if accident_logger else None,
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


def device_camera_section(detector: Detector, tracking: bool = False, accident_mode: bool = False) -> None:
    """
    Detection on the camera of the device *viewing* the page — a phone's camera,
    a tablet's, a laptop's webcam.

    This is fundamentally different from every other source in this tab: those
    are opened by the server, while here the **browser** captures the video and
    sends it up. That is why it works even on a hosted demo — the server never
    needs to reach the camera, the visitor's browser hands the frames over
    after asking for camera permission. HTTPS is required for that permission
    prompt, which both streamlit.app and localhost satisfy.
    """
    st.caption(
        "Uses the camera of the device you are reading this on — your browser "
        "will ask for permission."
    )

    modes = ["📸 Photo — point and shoot"]
    if WEBRTC_AVAILABLE:
        modes.append("🎥 Live — continuous video")
    mode = st.radio("Mode", modes, horizontal=True)

    if mode.startswith("🎥"):
        st.caption(
            "Press START, allow camera access, and the annotated stream comes "
            "back live. Frames are analysed on the server CPU, so on free "
            "hosting expect a few FPS — lower **Image size** in the sidebar "
            "for a smoother stream."
        )
        with st.expander("Connection taking a long time?"):
            st.markdown(
                "Live video has to negotiate a route between your browser and "
                "the server, and some networks make that hard.\n\n"
                "- **Mobile data** is the usual culprit. Carriers place "
                "subscribers behind a shared NAT, so no direct route exists and "
                "everything has to be relayed. Wi-Fi normally connects faster.\n"
                "- **Strict company or campus firewalls** can block the relay too.\n"
                "- **📸 Photo mode always works.** It sends a single still image "
                "over ordinary HTTPS with no negotiation at all, so it is "
                "unaffected by any of this.\n\n"
                "If live never connects on your network, use Photo mode — you "
                "lose the continuous stream, not the detection."
            )

        # One detector instance for the whole stream, so track IDs and the
        # motion history behind accident detection survive between frames.
        accidents = AccidentDetector() if accident_mode else None
        counter = {"frame": 0}

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                if tracking or accidents is not None:
                    detections = detector.track(img)
                else:
                    detections = detector.detect(img)
                detector.draw(img, detections)

                if accidents is not None:
                    counter["frame"] += 1
                    accidents.update(detections, counter["frame"])
                    accidents.draw(img)
            except Exception:
                # A failed frame should show as unannotated video, not kill
                # the stream.
                pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            webrtc_streamer(
                key="device-camera-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": ice_servers()},
                # Ask for a modest resolution. A phone will happily offer 1080p,
                # which is wasted here — inference runs at 640px anyway — and on
                # mobile data the extra bitrate is what makes the stream stall.
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15},
                    },
                    "audio": False,
                },
                video_frame_callback=video_frame_callback,
            )
        except Exception as exc:
            # The component needs a real Streamlit runtime; if it cannot start
            # (unusual host, test harness), fail as one clear message instead
            # of taking the whole tab down — photo mode still works.
            st.error(f"Live streaming could not start here: {exc}")
            st.caption("Use 📸 Photo mode instead — it needs no streaming support.")
        return

    if accident_mode:
        st.caption(
            "🚨 Accident detection needs motion over time, so it applies to **live** "
            "mode only — a single photo has no speed to measure."
        )

    photo = st.camera_input("Point the camera and take a photo")
    if photo is None:
        st.info("Take a photo to run detection on it.")
        return

    frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read the captured photo — please try again.")
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
            file_name="detected_photo.png",
            mime="image/png",
        )


def tab_live(detector: Detector, tracking: bool, accident_mode: bool = False) -> None:
    """A saved camera, the local webcam, or a network camera (phone / CCTV / IP)."""
    try:
        saved = load_cameras()
    except CameraStoreError as exc:
        st.warning(str(exc))
        saved = {}

    hosted = is_cloud()
    if hosted:
        st.info(
            "**This is a hosted demo.** Use **This device's camera** below — your "
            "browser will ask for permission and EyeSpot will analyse what *your* "
            "camera sees.\n\n"
            "What does not work here: the server's own webcam (it has none) and "
            "cameras on your home network — `192.168.x.x` addresses are private "
            "and do not exist on the public internet. For CCTV/RTSP input, run "
            "EyeSpot on your own machine — see the README."
        )

    choices = ["This device's camera", "Phone / CCTV / IP camera (URL)"]
    if not hosted:
        choices.insert(1, "Local webcam")
    if saved:
        choices.insert(1, "Saved camera")
    source_type = st.radio("Camera type", choices, horizontal=True)

    if source_type == "This device's camera":
        device_camera_section(detector, tracking, accident_mode)
        return

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
    alert_slot = st.empty()
    stats_slot = st.sidebar.empty()
    counts_slot = st.sidebar.empty()
    fps_counter = FPSCounter()
    accidents = AccidentDetector() if accident_mode else None

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

            if accidents is not None:
                event = accidents.update(detections, fps_counter.frame_count)
                if event is not None:
                    alert_slot.error(
                        f"🚨 **Accident detected** at frame {event.frame} — "
                        f"score {event.score:.2f} · {event.summary}"
                    )
                accidents.draw(frame)

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
    accident_mode = st.sidebar.toggle(
        "🚨 Accident detection",
        value=False,
        help=(
            "Watches how tracked road users move and raises an event when the "
            "motion stops looking like traffic. Needs tracking, so it turns it on."
        ),
    )
    if accident_mode:
        tracking = True
        # Accident detection reads motion between frames, which needs objects to
        # hold the same track ID. Marginal detections flicker and break that, so
        # the confidence threshold is lowered here unless the user has already
        # moved it below the recommended ceiling themselves.
        if settings["confidence"] > config.ACCIDENT_CONFIDENCE:
            settings["confidence"] = config.ACCIDENT_CONFIDENCE
            apply_settings(detector, settings, class_filter)
            st.sidebar.info(
                f"Confidence lowered to **{config.ACCIDENT_CONFIDENCE}** and tracking "
                "enabled — accident detection needs objects to keep the same ID "
                "between frames, and detections near the usual 0.5 threshold "
                "flicker in and out, resetting their motion history."
            )
        else:
            st.sidebar.caption("Tracking enabled automatically — accident detection needs persistent IDs.")

    st.sidebar.caption(f"Device: `{detector.device}`  •  Classes: {len(detector.class_names)}")

    image_tab, video_tab, live_tab = st.tabs(["🖼️ Image", "🎬 Video", "📷 Live camera"])
    with image_tab:
        tab_image(detector)
    with video_tab:
        tab_video(detector, tracking, accident_mode)
    with live_tab:
        tab_live(detector, tracking, accident_mode)


if __name__ == "__main__":
    main()
