#!/usr/bin/env python3
"""
main.py — Entry point. Runs real-time YOLO detection on a camera or video file.

Examples:
    python main.py                              # default webcam
    python main.py --source 1                   # second camera
    python main.py --source video.mp4           # video file
    python main.py --source pic.jpg             # single image
    python main.py --source video.mp4 --save    # save the annotated video
    python main.py --classes person car         # only these classes
    python main.py --conf 0.35 --model yolov8s.pt
    python main.py --source video.mp4 --no-display --save   # headless batch

Phone camera / CCTV (network stream):
    python main.py --source http://192.168.1.5:8080/video     # phone (IP Webcam app)
    python main.py --source rtsp://user:pass@192.168.1.10:554/stream1   # CCTV

Saved cameras (so the full URL need not be retyped):
    python main.py --source http://192.168.1.5:8080/video --save-camera phone
    python main.py --camera phone               # the name is now enough
    python main.py --list-cameras               # show every saved camera

Accident detection (road video, CCTV, live camera):
    python main.py --source traffic.mp4 --accident
    python main.py --camera roadcam --accident --log
    python main.py --source dashcam.mp4 --accident --accident-threshold 0.45

Tracking and target lock:
    python main.py --track                      # give every object a persistent ID
    python main.py --track --lock person        # lock onto the first person seen
    python main.py --track --lock bird          # follow a bird
    python main.py --track --lock-id 3          # lock onto track ID 3

Keyboard (in the video window):
    q / Esc  quit        s  screenshot        p / Space  pause-resume
    o        overlay on/off                   + / -      confidence up/down
    t        tracking on/off                  click      lock onto that object
    l        auto-lock                        c          clear the lock
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

import config
from detector.accident import AccidentDetector
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
from utils.save_output import AccidentLogger, DetectionLogger, VideoWriter, save_screenshot

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Network stream URLs — phone camera, CCTV, IP camera
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")

HELP_BAR = "q quit | s shot | p pause | o overlay | +/- conf | t track | click lock | c clear"

logger = logging.getLogger("eyespot")


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
            "Camera index (0, 1, ...), a video/image file path, or a network stream "
            "URL (rtsp:// for CCTV, http:// for a phone camera). Default: 0"
        ),
    )
    # -- saved camera profiles (cameras.json) --
    parser.add_argument(
        "--camera", metavar="NAME", default=None,
        help="Use a saved camera by name (cameras.json). Overrides --source.",
    )
    parser.add_argument(
        "--save-camera", metavar="NAME", default=None,
        help="Save the current --source under this name in cameras.json",
    )
    parser.add_argument(
        "--camera-note", metavar="TEXT", default="",
        help="An optional note to store with --save-camera (e.g. 'front gate')",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List every saved camera and exit")

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
            "Detect only these classes, e.g. --classes person car. "
            f"Groups also work: {', '.join(sorted(config.CLASS_GROUPS))}"
        ),
    )
    # -- tracking / target lock --
    parser.add_argument(
        "--track", action="store_true",
        help="Enable tracking — every object gets a persistent ID (required for locking)",
    )
    parser.add_argument(
        "--tracker", default=config.TRACKER, choices=["bytetrack.yaml", "botsort.yaml"],
        help="bytetrack = faster, botsort = better re-identification",
    )
    parser.add_argument(
        "--lock", metavar="CLASS", default=None,
        help="Automatically lock onto the first object of this class, e.g. --lock person",
    )
    parser.add_argument(
        "--lock-id", type=int, default=None, metavar="ID",
        help="Lock onto this track ID (e.g. --lock-id 3)",
    )

    # -- accident detection --
    parser.add_argument(
        "--accident", action="store_true",
        help="Detect road accidents from motion (implies --track, since it needs persistent IDs)",
    )
    parser.add_argument(
        "--accident-threshold", type=float, default=None, metavar="SCORE",
        help=f"Fused score that triggers an event (default: {config.ACCIDENT_THRESHOLD})",
    )

    parser.add_argument("--save", action="store_true", help="Save the annotated video/image")
    parser.add_argument("--log", action="store_true", help="Write detections to a CSV log")
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Where to save output (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not open a window (for headless servers and batch processing)",
    )
    parser.add_argument("--list-classes", action="store_true", help="Print every class the model knows and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed logs")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_camera(name: str):
    """
    Look up a saved camera (cameras.json). A corrupt store must not crash the
    run — we log a warning and move on, since the user may well have passed a
    file path that has nothing to do with the camera store.
    """
    try:
        return get_camera(name)
    except CameraStoreError as exc:
        logger.warning("%s", exc)
        return None


def resolve_source(source: str, _follow_camera: bool = True) -> tuple[int | str, str]:
    """
    Turn the user's --source into an (opencv_source, kind) pair.
    kind = "webcam" | "stream" | "video" | "image"

    Besides file paths and URLs, the name of a camera saved in `cameras.json`
    also works (e.g. --source phone).
    """
    if source.isdigit():
        return int(source), "webcam"

    # Network stream: phone camera app, IP camera, or CCTV (RTSP)
    if source.lower().startswith(STREAM_PREFIXES):
        return source, "stream"

    path = Path(source)
    if path.exists():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return str(path), "image"
        return str(path), "video"

    # Neither a file nor a URL — it may be the name of a saved camera
    if _follow_camera:
        profile = _lookup_camera(source)
        # `_follow_camera=False` stops the saved URL from being resolved as a
        # name again; otherwise an entry pointing at itself would loop forever.
        if profile is not None:
            return resolve_source(profile.url, _follow_camera=False)

    raise FileNotFoundError(
        f"Could not make sense of the source: {source}\n"
        "- For a webcam, pass a number (e.g. --source 0)\n"
        "- For a file, pass a valid path\n"
        "- For a phone or CCTV, pass the full URL (rtsp://... or http://...)\n"
        "- For a saved camera, pass its name — list them with: python main.py --list-cameras"
    )


def open_capture(source: int | str, kind: str) -> cv2.VideoCapture:
    """Open a VideoCapture, raising a clear error if it fails."""
    if kind == "stream":
        # The FFMPEG backend is the most reliable one for network streams
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Keep the buffer small, otherwise latency grows as frames pile up and
        # the video falls behind — very noticeable on CCTV feeds
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.STREAM_BUFFER_SIZE)
    else:
        cap = cv2.VideoCapture(source)

    if kind == "webcam":
        # These are only requests; they apply if the camera supports them
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        cap.release()
        if kind == "webcam":
            raise RuntimeError(
                f"Could not open the webcam (index {source}).\n"
                "- Check that the camera is connected and no other app is using it\n"
                "- Try another index: --source 1\n"
                "- On Linux, list available cameras with `ls /dev/video*`"
            )
        if kind == "stream":
            raise RuntimeError(
                f"Could not connect to the stream: {mask_credentials(source)}\n"
                "- Test the URL in a browser or VLC (Media > Open Network Stream)\n"
                "- The phone and the PC must be on the same Wi-Fi network\n"
                "- For RTSP, check the username and password; URL-encode special characters\n"
                "- Check your firewall and that the camera has RTSP enabled"
            )
        raise RuntimeError(f"Could not open the video file: {source} (corrupt or unsupported codec?)")

    return cap


def reconnect_stream(source: int | str, kind: str) -> cv2.VideoCapture | None:
    """
    Try to reconnect after a stream drops.

    Dropped connections are normal on network cameras (phone or CCTV) — the
    Wi-Fi weakens, or the camera reboots. So we retry instead of crashing.
    """
    for attempt in range(1, config.STREAM_RECONNECT_ATTEMPTS + 1):
        logger.warning(
            "Stream disconnected — reconnect attempt %d/%d...",
            attempt, config.STREAM_RECONNECT_ATTEMPTS,
        )
        time.sleep(config.STREAM_RECONNECT_DELAY)
        try:
            cap = open_capture(source, kind)
        except RuntimeError:
            continue
        if cap.isOpened():
            logger.info("Stream reconnected.")
            return cap
        cap.release()

    logger.error("Could not reconnect the stream after %d attempts.",
                 config.STREAM_RECONNECT_ATTEMPTS)
    return None


def key_matches(key: int, keys: tuple[str, ...]) -> bool:
    """Compare OpenCV's waitKey code against a shortcut list from config."""
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
    """Detection on a single image."""
    frame = cv2.imread(path)
    if frame is None:
        logger.error("Could not read the image: %s", path)
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
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def run_stream(detector: Detector, source: int | str, kind: str, args: argparse.Namespace) -> int:
    """The main loop for a webcam, network stream or video file."""
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
    # Accident detection is built on kinematics, and kinematics need an object
    # to keep the same identity between frames — so it implies tracking.
    tracking = args.track or args.lock is not None or args.lock_id is not None or args.accident
    target_lock = TargetLock(
        trail_length=config.TRAIL_LENGTH,
        lost_after=config.TARGET_LOST_AFTER,
    )
    if args.lock_id is not None:
        target_lock.lock(args.lock_id)
    pending_auto_lock = args.lock  # on the first frame, find this class and lock onto it

    accidents: AccidentDetector | None = None
    accident_logger: AccidentLogger | None = None
    if args.accident:
        accidents = AccidentDetector(threshold=args.accident_threshold)
        accident_logger = AccidentLogger(directory=out_dir)
        saved_paths.append(accident_logger.path)

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
    if accidents is not None:
        print(f"Accident detection: ON (threshold {accidents.threshold:.2f})")
    if args.lock:
        print(f"Auto-lock: first '{args.lock}'")
    if args.lock_id is not None:
        print(f"Locked on track ID: {args.lock_id}")
    if not args.no_display:
        print(HELP_BAR.replace(" | ", "   "))
    print()

    # Mouse-click target lock. The callback needs the current detections, so a
    # mutable holder is updated on every frame.
    click_state: dict = {"detections": detections}

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not tracking:
            print("  Turn tracking on before locking ('t' key or --track).")
            return
        hit = target_lock.lock_at_point(click_state["detections"], x, y)
        if hit is not None:
            print(f"  Locked: #{hit.track_id} {hit.label} ({hit.confidence * 100:.0f}%)")
        else:
            print("  No trackable object at that point.")

    if not args.no_display:
        cv2.namedWindow(config.WINDOW_NAME)
        cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if kind == "video":
                        logger.info("End of video.")
                        break
                    if kind == "stream":
                        # The network stream dropped — try to reconnect
                        cap.release()
                        new_cap = reconnect_stream(source, kind)
                        if new_cap is None:
                            break
                        cap = new_cap
                        continue
                    logger.warning("No frame from the webcam (did it disconnect?).")
                    break

                # Tracking mode assigns persistent IDs, which target lock relies on
                detections = detector.track(frame) if tracking else detector.detect(frame)
                click_state["detections"] = detections
                counts = count_by_label(detections)

                # --lock CLASS: lock onto that class the first time it appears
                if pending_auto_lock and not target_lock.is_active:
                    hit = target_lock.lock_first(detections, pending_auto_lock)
                    if hit is not None:
                        print(f"  Auto-locked: #{hit.track_id} {hit.label}")
                        pending_auto_lock = None

                target = target_lock.update(detections)

                # While a lock is active, dim every other object
                dim_ids = None
                if target_lock.is_active:
                    dim_ids = {d.track_id for d in detections if d.track_id != target_lock.track_id}

                detector.draw(frame, detections, dim_ids=dim_ids)
                target_lock.draw(frame, target)

                if accidents is not None:
                    event = accidents.update(detections, fps_counter.frame_count)
                    if event is not None:
                        # Snapshot the annotated frame before the banner is drawn
                        # over it, so the saved image shows the scene clearly.
                        snapshot = accident_logger.log(event, frame)
                        print(
                            f"\n  ⚠ ACCIDENT  frame {event.frame}  score {event.score:.2f}"
                            f"  [{event.summary}]"
                        )
                        if snapshot:
                            print(f"    snapshot: {snapshot}")
                    accidents.draw(frame)

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

                # Show progress while processing a video headlessly
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
                # While paused, keep showing the last frame
                display_frame = frame.copy()

            if args.no_display:
                continue

            if show_overlay:
                extra = []
                if tracking:
                    extra.append(target_lock.status_line())
                if accidents is not None:
                    extra.append(accidents.status_line())
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

            # The user closed the window with the X button
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
                    fps_counter.reset()  # don't count paused time towards FPS
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
                    target_lock.clear()  # a lock is meaningless without IDs
                print(f"  Tracking: {'ON' if tracking else 'OFF'}")
            elif key_matches(key, config.KEY_LOCK_FIRST):
                if not tracking:
                    print("  Turn tracking on first ('t').")
                else:
                    hit = target_lock.lock_first(detections, args.lock)
                    print(
                        f"  Locked: #{hit.track_id} {hit.label}" if hit
                        else "  No object available to lock onto."
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
        if accident_logger is not None:
            accident_logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.no_display and total_frames:
        print()  # newline after the progress line
    if accidents is not None:
        print(f"\n  Accident events: {len(accidents.events)}")
        for event in accidents.events:
            print(f"    frame {event.frame:>6}  score {event.score:.2f}  [{event.summary}]")
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

    # --list-cameras is answered immediately, without loading the model
    if args.list_cameras:
        try:
            print("\n" + format_camera_table(load_cameras()) + "\n")
        except CameraStoreError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    # --camera NAME: the saved profile's URL takes the place of --source
    source_text = args.source
    if args.camera:
        try:
            profile = get_camera(args.camera)
        except CameraStoreError as exc:
            logger.error("%s", exc)
            return 1
        if profile is None:
            logger.error(
                "No saved camera called '%s'. List them with: python main.py --list-cameras",
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
    except ValueError as exc:  # invalid class name
        logger.error("%s", exc)
        return 1

    if args.list_classes:
        print("\nGroups (shortcuts — e.g. --classes animals):\n")
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

    # --save-camera NAME: the source resolved, so store it under that name
    if args.save_camera:
        if kind == "stream":
            # http = phone app (IP Webcam/DroidCam), rtsp = CCTV — a label only
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
        print(f"From now on this is enough:  python main.py --camera {saved.name}\n")

    try:
        if kind == "image":
            return run_image(detector, str(source), args)
        return run_stream(detector, source, kind, args)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
