#!/usr/bin/env python3
"""
test_smoke.py — A quick self-test that everything is installed and wired up.

Run it with:
    python test_smoke.py

No webcam is required: the tests use synthetic frames and, when it is present,
sample.jpg.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASSED = 0
FAILED = 0


def check(name: str, fn) -> None:
    """Run one test and print PASS/FAIL, so an exception cannot abort the run."""
    global PASSED, FAILED
    try:
        detail = fn()
        PASSED += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:
        FAILED += 1
        print(f"  ❌ {name} — {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports() -> str:
    import cv2
    import numpy
    import ultralytics

    return f"opencv {cv2.__version__}, numpy {numpy.__version__}, ultralytics {ultralytics.__version__}"


def test_config() -> str:
    import config

    assert 0 < config.CONFIDENCE_THRESHOLD < 1, "confidence must be between 0 and 1"
    assert config.IMAGE_SIZE % 32 == 0, "image size must be a multiple of 32"
    return f"model={config.MODEL_NAME}, conf={config.CONFIDENCE_THRESHOLD}"


def test_fps_counter() -> str:
    import time

    from utils.fps_counter import FPSCounter

    fps = FPSCounter(window=5)
    for _ in range(6):
        time.sleep(0.01)
        fps.update()

    assert fps.frame_count == 6, "wrong frame count"
    assert fps.value > 0, "FPS should not be zero"
    fps.reset()
    assert fps.value == 0.0, "FPS should be 0 after a reset"
    return f"{fps.average:.0f} FPS avg"


def test_model_load() -> str:
    from detector.detect import Detector

    detector = Detector()
    assert len(detector.class_names) == 80, "a COCO model should have 80 classes"
    assert "person" in detector.class_names
    globals()["_DETECTOR"] = detector  # reused by the later tests
    return f"{len(detector.class_names)} classes on {detector.device}"


def test_detect_synthetic() -> str:
    import numpy as np

    detector = globals()["_DETECTOR"]
    # A blank frame must not crash — it should simply yield 0 detections
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    detector.draw(frame, detections)
    return f"{len(detections)} detections on blank frame"


def test_detect_sample() -> str:
    import cv2

    sample = ROOT / "sample.jpg"
    if not sample.exists():
        return "skipped (sample.jpg not present)"

    from detector.detect import count_by_label

    detector = globals()["_DETECTOR"]
    frame = cv2.imread(str(sample))
    detections = detector.detect(frame)
    counts = count_by_label(detections)

    assert detections, "sample.jpg should have produced some detections"
    assert all(0 <= d.confidence <= 1 for d in detections), "confidence outside 0-1"
    for d in detections:
        x1, y1, x2, y2 = d.box
        assert x1 < x2 and y1 < y2, f"invalid box: {d.box}"
    return ", ".join(f"{k}x{v}" for k, v in counts.items())


def test_class_filter() -> str:
    from detector.model_loader import class_name_to_id

    detector = globals()["_DETECTOR"]
    ids = class_name_to_id(detector.model, ["person", "car"])
    assert ids == [0, 2], f"expected [0, 2], got {ids}"

    try:
        class_name_to_id(detector.model, ["unicorn"])
        raise AssertionError("an invalid class name should have raised")
    except ValueError:
        pass
    return "person=0, car=2; unknown class rejected"


def test_class_groups() -> str:
    """Group shortcuts such as `--classes animals` should expand."""
    import config
    from detector.model_loader import class_name_to_id, expand_class_groups

    assert "dog" in expand_class_groups(["animals"]), "animals should include dog"
    assert expand_class_groups(["person"]) == ["person"], "a plain class must pass through"

    # Groups mixed with plain classes, and duplicates collapsed
    mixed = expand_class_groups(["person", "pets", "dog"])
    assert mixed[0] == "person" and mixed.count("dog") == 1, mixed

    detector = globals()["_DETECTOR"]
    ids = class_name_to_id(detector.model, ["animals"])
    assert len(ids) == len(config.CLASS_GROUPS["animals"]), f"got {len(ids)} IDs"
    assert len(set(ids)) == len(ids), "IDs should not repeat"

    try:
        class_name_to_id(detector.model, ["no_such_class"])
        raise AssertionError("an unknown class should have raised ValueError")
    except ValueError as exc:
        assert "Groups:" in str(exc), "the error should mention the groups"

    return f"{len(config.CLASS_GROUPS)} groups; animals -> {len(ids)} classes"


def test_overlay() -> str:
    import numpy as np

    from utils.overlay import draw_help_bar, draw_overlay

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_overlay(frame, 24.5, {"person": 3, "car": 1}, 0.5, paused=True)
    draw_help_bar(frame, "q quit")
    assert frame.any(), "the overlay drew nothing"
    return "panel + help bar drawn"


def test_save_output() -> str:
    import numpy as np

    from utils.save_output import DetectionLogger, VideoWriter, save_screenshot
    from detector.detect import Detection

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)

        shot = save_screenshot(frame, prefix="test", directory=tmp_path)
        assert shot.exists() and shot.stat().st_size > 0, "the screenshot is empty"

        with VideoWriter(tmp_path / "test.mp4", fps=15, frame_size=(320, 240)) as writer:
            for _ in range(5):
                writer.write(frame)
        assert (tmp_path / "test.mp4").stat().st_size > 0, "the video is empty"

        det = Detection(class_id=0, label="person", confidence=0.9, box=(1, 2, 3, 4))
        with DetectionLogger(tmp_path / "log.csv") as log:
            log.log(0, [det, det])
        lines = (tmp_path / "log.csv").read_text().strip().splitlines()
        assert len(lines) == 3, f"expected a header plus 2 rows, got {len(lines)}"

    return "screenshot, video and CSV all OK"


def test_browser_transcode() -> str:
    """An H.264 version is produced for the Streamlit player (when ffmpeg exists)."""
    import shutil

    import cv2
    import numpy as np

    from utils.save_output import VideoWriter, transcode_for_browser

    if shutil.which("ffmpeg") is None:
        return "skipped (ffmpeg not installed — downloads still work)"

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "clip.mp4"
        with VideoWriter(path=source, fps=10, frame_size=(160, 120)) as writer:
            for i in range(10):
                frame = np.full((120, 160, 3), i * 20, dtype=np.uint8)
                writer.write(frame)

        out = transcode_for_browser(source)
        assert out is not None and out.exists(), "no H.264 file was produced"
        assert out.stat().st_size > 0, "the H.264 file is empty"

        # It must also be readable, otherwise the browser cannot play it either
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened(), "the H.264 video would not open"
        ok, _ = cap.read()
        cap.release()
        assert ok, "no frame could be read from the H.264 video"

    return "mp4v -> H.264 (browser-playable)"


def test_video_pipeline() -> str:
    import cv2
    import numpy as np

    detector = globals()["_DETECTOR"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
        )
        for i in range(5):
            writer.write(np.full((240, 320, 3), i * 40, dtype=np.uint8))
        writer.release()

        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), "the test video would not open"
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detector.detect(frame)
            frames += 1
        cap.release()

    assert frames == 5, f"expected 5 frames, got {frames}"
    return f"{frames} frames processed"


def test_cli_parsing() -> str:
    from main import parse_args, resolve_source

    args = parse_args(["--source", "0", "--conf", "0.3", "--classes", "person", "car"])
    assert args.conf == 0.3 and args.classes == ["person", "car"]
    assert resolve_source("0") == (0, "webcam")

    sample = ROOT / "sample.jpg"
    if sample.exists():
        assert resolve_source(str(sample))[1] == "image"

    try:
        resolve_source("does_not_exist.mp4")
        raise AssertionError("a missing file should have raised")
    except FileNotFoundError:
        pass
    return "flags, source resolution, missing-file error"


def test_stream_urls() -> str:
    """Phone camera and CCTV URLs must be recognised as streams."""
    from main import resolve_source
    from utils.cameras import mask_credentials

    urls = [
        "rtsp://admin:pass123@192.168.1.10:554/stream1",   # CCTV
        "http://192.168.1.5:8080/video",                   # phone (IP Webcam app)
        "https://example.com/live.m3u8",
        "udp://239.0.0.1:1234",
    ]
    for url in urls:
        src, kind = resolve_source(url)
        assert kind == "stream", f"{url} -> {kind} (should have been a stream)"
        assert src == url, "the URL must not be altered"

    masked = mask_credentials("rtsp://admin:pass123@192.168.1.10:554/stream1")
    assert "pass123" not in masked, "the password is leaking into the logs"
    assert "admin" in masked, "the username should stay visible"
    return f"{len(urls)} URL formats + password masking"


def test_camera_profiles() -> str:
    """cameras.json save/load, resolving a source by name, and the URL builder."""
    import config
    import main
    from utils.cameras import build_url, delete_camera, load_cameras, save_camera

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "cameras.json"

        save_camera("phone", "http://192.168.1.5:8080/video", kind="mobile", note="test", path=store)
        save_camera("laptop", "0", kind="usb", path=store)
        cams = load_cameras(store)
        assert set(cams) == {"phone", "laptop"}, f"saved cameras: {set(cams)}"
        assert cams["phone"].note == "test", "the note was not saved"

        # `--source phone` should resolve through the saved camera
        original = config.CAMERAS_FILE
        config.CAMERAS_FILE = store
        try:
            assert main.resolve_source("phone") == ("http://192.168.1.5:8080/video", "stream")
            assert main.resolve_source("laptop") == (0, "webcam"), "a USB entry should become an int"
            try:
                main.resolve_source("no_such_camera")
                raise AssertionError("an unknown camera should have raised")
            except FileNotFoundError:
                pass
        finally:
            config.CAMERAS_FILE = original

        assert delete_camera("phone", path=store), "delete failed"
        assert set(load_cameras(store)) == {"laptop"}

    # Building a URL from a brand — the password must be URL-encoded (@ -> %40)
    url = build_url("tplink", "192.168.1.10", user="admin", password="pa@ss", quality="sub")
    assert url == "rtsp://admin:pa%40ss@192.168.1.10:554/stream2", url
    assert build_url("ip-webcam", "192.168.1.5") == "http://192.168.1.5:8080/video"
    assert "Channels/101" in build_url("hikvision", "1.2.3.4", quality="main")
    try:
        build_url("no_such_brand", "1.2.3.4")
        raise AssertionError("an unknown brand should have raised ValueError")
    except ValueError:
        pass

    return "save/load/delete, name resolution, brand URL builder"


def test_tracking() -> str:
    """Track IDs must persist from frame to frame."""
    import cv2

    sample = ROOT / "sample.jpg"
    if not sample.exists():
        return "skipped (sample.jpg not present)"

    detector = globals()["_DETECTOR"]
    detector.reset_tracker()
    frame = cv2.imread(str(sample))

    ids_per_frame = []
    for shift in range(4):
        # A slight pan, so objects move and the tracker has real work to do
        moved = frame[:, shift * 5 : frame.shape[1] - 40 + shift * 5]
        dets = detector.track(moved)
        assert all(d.track_id is not None for d in dets), "track mode should assign IDs"
        ids_per_frame.append({d.track_id for d in dets})

    persistent = set.intersection(*ids_per_frame) if ids_per_frame else set()
    assert persistent, f"no ID persisted across frames: {ids_per_frame}"

    # detect() mode must not carry IDs
    assert all(d.track_id is None for d in detector.detect(frame)), "detect() must not assign IDs"
    return f"{len(persistent)} IDs persisted across 4 frames"


def test_target_lock() -> str:
    from detector.detect import Detection
    from detector.target_lock import TargetLock

    lock = TargetLock(trail_length=5, lost_after=3)
    assert not lock.is_active, "nothing should be locked initially"

    a = Detection(0, "person", 0.9, (10, 10, 50, 90), track_id=1)
    b = Detection(2, "car", 0.8, (100, 100, 200, 180), track_id=2)

    # Click to lock — the smaller (more specific) object should win
    hit = lock.lock_at_point([a, b], 30, 50)
    assert hit is not None and lock.track_id == 1, "the click should have locked the person"

    # A point inside no box at all
    lock.clear()
    assert lock.lock_at_point([a, b], 500, 500) is None, "empty space must not lock"

    # Auto-lock by class name
    assert lock.lock_first([a, b], "car") is not None and lock.track_id == 2

    # While the target is visible, the trail should grow
    lock.lock(1)
    for _ in range(3):
        assert lock.update([a, b]) is not None
    assert len(lock.trail) == 3 and not lock.is_lost

    # Target disappears -> LOST, then it is re-acquired when it returns
    for _ in range(4):
        assert lock.update([b]) is None
    assert lock.is_lost and lock.is_stale, "should be stale after 4 missing frames"
    assert lock.update([a, b]) is not None, "it should be re-acquired on return"
    assert not lock.is_lost, "it must not still read LOST after re-acquisition"

    return "click-lock, auto-lock, trail, lost + re-acquire"


def test_lock_drawing() -> str:
    import numpy as np

    from detector.detect import Detection
    from detector.target_lock import TargetLock

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lock = TargetLock()
    target = Detection(0, "bird", 0.77, (100, 60, 160, 140), track_id=7)
    lock.lock(7, "bird")
    lock.update([target])
    lock.draw(frame, target)
    assert frame.any(), "the lock HUD was not drawn"

    # The LOST banner must not crash either
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    lock.update([])
    lock.draw(blank, None)
    assert blank.any(), "the LOST banner was not drawn"
    return "lock HUD + lost banner"


# ---------------------------------------------------------------------------

TESTS = [
    ("Dependencies installed", test_imports),
    ("Config valid", test_config),
    ("FPS counter", test_fps_counter),
    ("Model loads", test_model_load),
    ("Detection on blank frame", test_detect_synthetic),
    ("Detection on sample image", test_detect_sample),
    ("Class filter", test_class_filter),
    ("Class groups (animals/vehicles)", test_class_groups),
    ("Overlay drawing", test_overlay),
    ("Output saving", test_save_output),
    ("Video pipeline", test_video_pipeline),
    ("Browser transcode (H.264)", test_browser_transcode),
    ("CLI parsing", test_cli_parsing),
    ("Stream URLs (phone/CCTV)", test_stream_urls),
    ("Camera profiles (cameras.json)", test_camera_profiles),
    ("Object tracking (persistent IDs)", test_tracking),
    ("Target lock", test_target_lock),
    ("Target lock drawing", test_lock_drawing),
]


def main() -> int:
    print("\n" + "=" * 52)
    print("  EyeSpot — Smoke Test")
    print("=" * 52 + "\n")

    for name, fn in TESTS:
        check(name, fn)

    print("\n" + "-" * 52)
    print(f"  {PASSED} passed, {FAILED} failed")
    print("-" * 52 + "\n")

    if FAILED == 0:
        print("All good. Now run:  python main.py\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
