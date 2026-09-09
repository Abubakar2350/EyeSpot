#!/usr/bin/env python3
"""
main.py — Entry point. Webcam ya video file par real-time YOLO detection chalata hai.

Examples:
    python main.py                              # default webcam
    python main.py --source 1                   # doosra camera
    python main.py --source video.mp4           # video file
    python main.py --source pic.jpg             # single image
    python main.py --source video.mp4 --save    # output video save karo
    python main.py --classes person car         # sirf ye classes
    python main.py --conf 0.35 --model yolov8s.pt
    python main.py --source video.mp4 --no-display --save   # headless batch

Phone camera / CCTV (network stream):
    python main.py --source http://192.168.1.5:8080/video     # phone (IP Webcam app)
    python main.py --source rtsp://user:pass@192.168.1.10:554/stream1   # CCTV

Saved cameras (poora URL baar baar type na karna pade):
    python main.py --source http://192.168.1.5:8080/video --save-camera phone
    python main.py --camera phone               # ab sirf naam kaafi hai
    python main.py --list-cameras               # saari saved cameras dekhein

Tracking aur target lock:
    python main.py --track                      # har object ko persistent ID do
    python main.py --track --lock person        # pehle person par khud lock karo
    python main.py --track --lock bird          # kisi parinde ko follow karo
    python main.py --track --lock-id 3          # jis ki track ID 3 hai

Keyboard (video window mein):
    q / Esc  quit        s  screenshot        p / Space  pause-resume
    o        overlay on/off                   + / -      confidence up/down
    t        tracking on/off                  click      us object par lock karo
    l        auto-lock                        c          lock hatao
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

import config
from detector.detect import Detector, count_by_label
from detector.model_loader import ModelLoadError
from detector.target_lock import TargetLock
from utils.cameras import (
    CameraStoreError,
    format_camera_table,
    get_camera,
    load_cameras,
    mask_credentials,
    save_camera,
)
from utils.fps_counter import FPSCounter
from utils.overlay import draw_help_bar, draw_overlay
from utils.save_output import DetectionLogger, VideoWriter, save_screenshot

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Network stream URLs — phone camera, CCTV, IP camera
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")

HELP_BAR = "q quit | s shot | p pause | o overlay | +/- conf | t track | click lock | c clear"

logger = logging.getLogger("yolo")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time object detection using YOLO (webcam / video / image).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default=str(config.CAMERA_INDEX),
        help=(
            "Camera index (0, 1, ...), video/image file path, ya network stream URL "
            "(rtsp:// CCTV, http:// phone camera). Default: 0"
        ),
    )
    # -- saved camera profiles (cameras.json) --
    parser.add_argument(
        "--camera", metavar="NAME", default=None,
        help="Saved camera ka naam use karo (cameras.json). --source ko override karta hai.",
    )
    parser.add_argument(
        "--save-camera", metavar="NAME", default=None,
        help="Mojooda --source ko is naam se cameras.json mein save kar do",
    )
    parser.add_argument(
        "--camera-note", metavar="TEXT", default="",
        help="--save-camera ke sath ek chhota note (e.g. 'front gate')",
    )
    parser.add_argument("--list-cameras", action="store_true", help="Saari saved cameras dikhao aur exit")

    parser.add_argument("--model", default=config.MODEL_NAME, help="YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--device", default=config.DEVICE, help="cpu / cuda / mps (default: auto)")
    parser.add_argument(
        "--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou", type=float, default=config.IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {config.IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMAGE_SIZE,
        help=f"Inference image size (default: {config.IMAGE_SIZE})",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None, metavar="NAME",
        help=(
            "Sirf ye classes detect karo, e.g. --classes person car. "
            f"Groups bhi chalte hain: {', '.join(sorted(config.CLASS_GROUPS))}"
        ),
    )
    # -- tracking / target lock --
    parser.add_argument(
        "--track", action="store_true",
        help="Tracking on karo — har object ko persistent ID milegi (lock ke liye zaroori)",
    )
    parser.add_argument(
        "--tracker", default=config.TRACKER, choices=["bytetrack.yaml", "botsort.yaml"],
        help="bytetrack = tez, botsort = behtar re-identification",
    )
    parser.add_argument(
        "--lock", metavar="CLASS", default=None,
        help='Is class ke pehle object par khud lock karo, e.g. --lock person / --lock bird',
    )
    parser.add_argument(
        "--lock-id", type=int, default=None, metavar="ID",
        help="Is track ID par lock karo (e.g. --lock-id 3)",
    )

    parser.add_argument("--save", action="store_true", help="Annotated video/image save karo")
    parser.add_argument("--log", action="store_true", help="Detections ko CSV mein log karo")
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Kahan save karna hai (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Window mat kholo (headless server / batch processing ke liye)",
    )
    parser.add_argument("--list-classes", action="store_true", help="Model ki saari classes print kar ke exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed logs")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_camera(name: str):
    """
    Saved camera dhoondo (cameras.json). Store corrupt ho to crash nahi karte —
    warning de kar aage barh jaate hain, kyunke ho sakta hai user ne file path
    diya ho aur camera store ka is se koi taalluq hi na ho.
    """
    try:
        return get_camera(name)
    except CameraStoreError as exc:
        logger.warning("%s", exc)
        return None


def resolve_source(source: str, _follow_camera: bool = True) -> tuple[int | str, str]:
    """
    User ke --source ko (opencv_source, kind) mein badlo.
    kind = "webcam" | "stream" | "video" | "image"

    File path aur URL ke ilawa `cameras.json` mein saved camera ka naam bhi
    chalta hai (e.g. --source phone).
    """
    if source.isdigit():
        return int(source), "webcam"

    # Network stream: phone camera app, IP camera, ya CCTV (RTSP)
    if source.lower().startswith(STREAM_PREFIXES):
        return source, "stream"

    path = Path(source)
    if path.exists():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return str(path), "image"
        return str(path), "video"

    # Na file hai na URL — shayad kisi saved camera ka naam ho
    if _follow_camera:
        profile = _lookup_camera(source)
        # `_follow_camera=False`: saved URL dobara naam ki tarah resolve na ho
        # (warna "phone" -> "phone" wali entry infinite loop bana deti)
        if profile is not None:
            return resolve_source(profile.url, _follow_camera=False)

    raise FileNotFoundError(
        f"Source samajh nahi aaya: {source}\n"
        "- Webcam ke liye number dein (e.g. --source 0)\n"
        "- File ke liye sahi path dein\n"
        "- Phone/CCTV ke liye poora URL dein (rtsp://... ya http://...)\n"
        "- Saved camera ka naam dein — dekhne ke liye: python main.py --list-cameras"
    )


def open_capture(source: int | str, kind: str) -> cv2.VideoCapture:
    """VideoCapture kholo aur saaf error do agar na khule."""
    if kind == "stream":
        # FFMPEG backend network streams ke liye sabse reliable hai
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Buffer chhota rakho warna latency barhti jaati hai (frames jama ho kar
        # video "peechhe" reh jaati hai — CCTV par ye bohot mehsoos hota hai)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.STREAM_BUFFER_SIZE)
    else:
        cap = cv2.VideoCapture(source)

    if kind == "webcam":
        # Ye sirf requests hain — camera support kare to apply hongi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        cap.release()
        if kind == "webcam":
            raise RuntimeError(
                f"Webcam (index {source}) open nahi ho saka.\n"
                "- Check karein ke camera connected hai aur koi doosri app use to nahi kar rahi\n"
                "- Doosra index try karein: --source 1\n"
                "- Linux par `ls /dev/video*` se available cameras dekhein"
            )
        if kind == "stream":
            raise RuntimeError(
                f"Stream connect nahi ho saka: {mask_credentials(source)}\n"
                "- URL browser/VLC mein test karein (VLC: Media > Open Network Stream)\n"
                "- Phone aur PC same Wi-Fi par hone chahiye\n"
                "- RTSP par username/password sahi hain? Special characters ko URL-encode karein\n"
                "- Firewall ya camera ka 'RTSP enabled' setting check karein"
            )
        raise RuntimeError(f"Video file open nahi ho saki: {source} (corrupt ya unsupported codec?)")

    return cap


def reconnect_stream(source: int | str, kind: str) -> cv2.VideoCapture | None:
    """
    Stream toot jaye to dobara connect karne ki koshish karo.

    Network cameras (phone/CCTV) par connection drop hona normal hai — Wi-Fi
    weak ho jaye ya camera reboot ho. Isliye crash karne ke bajaye retry karte hain.
    """
    for attempt in range(1, config.STREAM_RECONNECT_ATTEMPTS + 1):
        logger.warning(
            "Stream disconnect ho gaya — reconnect attempt %d/%d...",
            attempt, config.STREAM_RECONNECT_ATTEMPTS,
        )
        time.sleep(config.STREAM_RECONNECT_DELAY)
        try:
            cap = open_capture(source, kind)
        except RuntimeError:
            continue
        if cap.isOpened():
            logger.info("Stream wapas connect ho gaya.")
            return cap
        cap.release()

    logger.error("Stream reconnect nahi ho saka (%d koshishon ke baad).",
                 config.STREAM_RECONNECT_ATTEMPTS)
    return None


def key_matches(key: int, keys: tuple[str, ...]) -> bool:
    """OpenCV ka waitKey code compare karo config ke shortcut list se."""
    if key == -1:
        return False
    for k in keys:
        if isinstance(k, int):
            if key == k:
                return True
        elif key == ord(k):
            return True
    return False


def print_summary(counts: dict[str, int], fps: FPSCounter, saved: list[Path]) -> None:
    print("\n" + "=" * 46)
    print("  Session Summary")
    print("=" * 46)
    print(f"  Frames processed : {fps.frame_count}")
    print(f"  Duration         : {fps.elapsed:.1f}s")
    print(f"  Average FPS      : {fps.average:.1f}")
    if counts:
        print("  Last frame objects:")
        for label, count in counts.items():
            print(f"    - {label}: {count}")
    for path in saved:
        print(f"  Saved            : {path}")
    print("=" * 46 + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_image(detector: Detector, path: str, args: argparse.Namespace) -> int:
    """Single image par detection — Phase 2 ka test mode."""
    frame = cv2.imread(path)
    if frame is None:
        logger.error("Image read nahi ho saki: %s", path)
        return 1

    detections = detector.detect(frame)
    counts = count_by_label(detections)
    detector.draw(frame, detections)

    print(f"\n{len(detections)} object(s) detected in {Path(path).name}:")
    for label, count in counts.items():
        print(f"  - {label}: {count}")
    for det in detections:
        print(f"    {det.label:<15} {det.confidence * 100:5.1f}%  box={det.box}")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if args.save:
        saved = save_screenshot(frame, prefix=f"{Path(path).stem}_detected", directory=out_dir)
        print(f"\nSaved: {saved}")

    if not args.no_display:
        cv2.imshow(config.WINDOW_NAME, frame)
        print("\nKoi bhi key dabayein window band karne ke liye...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def run_stream(detector: Detector, source: int | str, kind: str, args: argparse.Namespace) -> int:
    """Webcam / network stream / video file ka main loop."""
    cap = open_capture(source, kind)
    out_dir = Path(args.output_dir) if args.output_dir else None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "video" else 0

    fps_counter = FPSCounter()
    writer: VideoWriter | None = None
    det_logger: DetectionLogger | None = None
    saved_paths: list[Path] = []
    show_overlay = config.SHOW_OVERLAY
    paused = False
    counts: dict[str, int] = {}
    detections: list = []

    # Tracking / target lock state
    tracking = args.track or args.lock is not None or args.lock_id is not None
    target_lock = TargetLock(
        trail_length=config.TRAIL_LENGTH,
        lost_after=config.TARGET_LOST_AFTER,
    )
    if args.lock_id is not None:
        target_lock.lock(args.lock_id)
    pending_auto_lock = args.lock  # pehle frame par is class ko dhoond kar lock karenge

    if args.log or config.ENABLE_CSV_LOG:
        det_logger = DetectionLogger(
            path=(out_dir / f"detections_{Path(str(source)).stem or 'webcam'}.csv") if out_dir else None
        )

    if kind == "webcam":
        label = "Webcam"
    elif kind == "stream":
        label = mask_credentials(str(source))
    else:
        label = Path(str(source)).name

    print(f"\nRunning on: {label}  |  model: {args.model}  |  device: {detector.device}")
    if detector.class_filter_names:
        print(f"Class filter: {', '.join(detector.class_filter_names)}")
    if tracking:
        print(f"Tracking: ON ({detector.tracker})")
    if args.lock:
        print(f"Auto-lock: first '{args.lock}'")
    if args.lock_id is not None:
        print(f"Locked on track ID: {args.lock_id}")
    if not args.no_display:
        print(HELP_BAR.replace(" | ", "   "))
    print()

    # Mouse click se target lock — callback ko current detections chahiye,
    # isliye ek mutable holder use karte hain jo har frame update hota hai.
    click_state: dict = {"detections": detections}

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not tracking:
            print("  Lock ke liye pehle tracking on karein ('t' key ya --track).")
            return
        hit = target_lock.lock_at_point(click_state["detections"], x, y)
        if hit is not None:
            print(f"  Locked: #{hit.track_id} {hit.label} ({hit.confidence * 100:.0f}%)")
        else:
            print("  Us jagah koi trackable object nahi mila.")

    if not args.no_display:
        cv2.namedWindow(config.WINDOW_NAME)
        cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if kind == "video":
                        logger.info("Video khatam.")
                        break
                    if kind == "stream":
                        # Network stream toot gaya — reconnect ki koshish karo
                        cap.release()
                        new_cap = reconnect_stream(source, kind)
                        if new_cap is None:
                            break
                        cap = new_cap
                        continue
                    logger.warning("Webcam se frame nahi aaya (disconnect ho gaya?).")
                    break

                # Tracking mode persistent IDs deta hai — target lock isi par chalta hai
                detections = detector.track(frame) if tracking else detector.detect(frame)
                click_state["detections"] = detections
                counts = count_by_label(detections)

                # --lock CLASS: pehle frame par jahan wo class mile, lock kar do
                if pending_auto_lock and not target_lock.is_active:
                    hit = target_lock.lock_first(detections, pending_auto_lock)
                    if hit is not None:
                        print(f"  Auto-locked: #{hit.track_id} {hit.label}")
                        pending_auto_lock = None

                target = target_lock.update(detections)

                # Lock active ho to baaki objects dim kar do
                dim_ids = None
                if target_lock.is_active:
                    dim_ids = {d.track_id for d in detections if d.track_id != target_lock.track_id}

                detector.draw(frame, detections, dim_ids=dim_ids)
                target_lock.draw(frame, target)

                if det_logger is not None:
                    det_logger.log(fps_counter.frame_count, detections)

                fps_counter.update()

                if args.save:
                    if writer is None:
                        h, w = frame.shape[:2]
                        out_fps = config.OUTPUT_FPS or (src_fps if src_fps and src_fps > 0 else 20.0)
                        stem = "webcam" if kind == "webcam" else Path(str(source)).stem
                        writer = VideoWriter(
                            path=(out_dir / f"{stem}_detected.mp4") if out_dir else None,
                            fps=out_fps,
                            frame_size=(w, h),
                            prefix=f"{stem}_detected",
                        )
                        saved_paths.append(writer.path)
                    writer.write(frame)

                # Headless mode mein video ka progress dikhao
                if args.no_display and total_frames and fps_counter.frame_count % 30 == 0:
                    pct = 100 * fps_counter.frame_count / total_frames
                    print(
                        f"\r  {fps_counter.frame_count}/{total_frames} frames "
                        f"({pct:5.1f}%)  {fps_counter.value:.1f} FPS",
                        end="",
                        flush=True,
                    )

                display_frame = frame
            else:
                # Pause ke doran wohi aakhri frame dikhate rahein
                display_frame = frame.copy()

            if args.no_display:
                continue

            if show_overlay:
                extra = []
                if tracking:
                    extra.append(target_lock.status_line())
                draw_overlay(
                    display_frame,
                    fps_counter.value,
                    counts,
                    detector.confidence,
                    paused=paused,
                    extra_lines=extra,
                )
                draw_help_bar(display_frame, HELP_BAR)

            cv2.imshow(config.WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            # User ne X button se window band kar di
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key_matches(key, config.KEY_QUIT):
                break
            if key_matches(key, config.KEY_SCREENSHOT):
                path = save_screenshot(display_frame, directory=out_dir)
                saved_paths.append(path)
                print(f"  Screenshot: {path}")
            elif key_matches(key, config.KEY_PAUSE):
                paused = not paused
                if not paused:
                    fps_counter.reset()  # pause ka time FPS mein count na ho
                print("  " + ("Paused" if paused else "Resumed"))
            elif key_matches(key, config.KEY_TOGGLE_OVERLAY):
                show_overlay = not show_overlay
            elif key_matches(key, config.KEY_CONF_UP):
                print(f"  Confidence: {detector.set_confidence(detector.confidence + 0.05):.2f}")
            elif key_matches(key, config.KEY_CONF_DOWN):
                print(f"  Confidence: {detector.set_confidence(detector.confidence - 0.05):.2f}")
            elif key_matches(key, config.KEY_TOGGLE_TRACK):
                tracking = not tracking
                if not tracking:
                    target_lock.clear()  # bina IDs ke lock ka koi matlab nahi
                print(f"  Tracking: {'ON' if tracking else 'OFF'}")
            elif key_matches(key, config.KEY_LOCK_FIRST):
                if not tracking:
                    print("  Pehle tracking on karein ('t').")
                else:
                    hit = target_lock.lock_first(detections, args.lock)
                    print(
                        f"  Locked: #{hit.track_id} {hit.label}" if hit
                        else "  Lock karne ke liye koi object nahi mila."
                    )
            elif key_matches(key, config.KEY_CLEAR_LOCK):
                target_lock.clear()
                print("  Lock cleared.")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if det_logger is not None:
            det_logger.close()
            saved_paths.append(det_logger.path)
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.no_display and total_frames:
        print()  # progress line ke baad newline
    print_summary(counts, fps_counter, saved_paths)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --list-cameras ka jawab model load kiye bagair de dete hain (fauran)
    if args.list_cameras:
        try:
            print("\n" + format_camera_table(load_cameras()) + "\n")
        except CameraStoreError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    # --camera NAME: saved profile ka URL --source ki jagah le leta hai
    source_text = args.source
    if args.camera:
        try:
            profile = get_camera(args.camera)
        except CameraStoreError as exc:
            logger.error("%s", exc)
            return 1
        if profile is None:
            logger.error(
                "Camera '%s' save nahi hai. Saved cameras dekhein: python main.py --list-cameras",
                args.camera,
            )
            return 1
        source_text = profile.url
        print(f"Camera '{profile.name}' -> {profile.safe_url}")

    try:
        detector = Detector(
            model_name=args.model,
            device=args.device,
            confidence=args.conf,
            iou=args.iou,
            class_filter=args.classes,
            image_size=args.imgsz,
            tracker=args.tracker,
        )
    except ModelLoadError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:  # galat class name
        logger.error("%s", exc)
        return 1

    if args.list_classes:
        print("\nGroups (shortcut — e.g. --classes animals):\n")
        for group, members in config.CLASS_GROUPS.items():
            print(f"  {group:<10} = {', '.join(members)}")

        print(f"\n{len(detector.class_names)} classes:\n")
        for i, name in enumerate(detector.class_names):
            print(f"  {i:>2}. {name}")
        print()
        return 0

    try:
        source, kind = resolve_source(source_text)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # --save-camera NAME: source samajh mein aa gaya, ab usay naam de kar mehfooz karo
    if args.save_camera:
        if kind == "stream":
            # http = phone app (IP Webcam/DroidCam), rtsp = CCTV — sirf label ke liye
            camera_kind = "cctv" if source_text.lower().startswith(("rtsp://", "rtmp://")) else "mobile"
        elif kind == "webcam":
            camera_kind = "usb"
        else:
            camera_kind = "other"
        try:
            saved = save_camera(args.save_camera, source_text, kind=camera_kind, note=args.camera_note)
        except CameraStoreError as exc:
            logger.error("%s", exc)
            return 1
        print(f"Camera saved: '{saved.name}' -> {saved.safe_url}")
        print(f"Ab sirf ye kaafi hai:  python main.py --camera {saved.name}\n")

    try:
        if kind == "image":
            return run_image(detector, str(source), args)
        return run_stream(detector, source, kind, args)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
