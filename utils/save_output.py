"""
save_output.py — Saving screenshots, output video and the CSV detection log.

Everything lands inside config.OUTPUT_DIR with a timestamped filename, so
earlier files are never overwritten.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def transcode_for_browser(path: Path | str, remove_source: bool = False) -> Path | None:
    """
    Re-encode a video to H.264 so a browser (Streamlit) can play it.

    Why this is needed: OpenCV cannot write H.264 on most builds — only `mp4v`
    (MPEG-4 Part 2) — and the HTML5 player does not support `mp4v`. Downloading
    the file still works fine (VLC and friends play it); only the in-browser
    preview stays blank.

    Returns None when `ffmpeg` is not installed, in which case the caller
    should simply skip the preview; downloads keep working either way.
    """
    source = Path(path)
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not found — skipping browser preview; download still works.")
        return None

    target = source.with_name(f"{source.stem}_h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(source),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",   # required by older players
                "-movflags", "+faststart",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg transcode failed: %s", exc)
        return None

    if remove_source:
        source.unlink(missing_ok=True)

    logger.info("Browser-playable H.264 version: %s", target)
    return target


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_output_dir(directory: Path | None = None) -> Path:
    target = Path(directory) if directory else config.OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_screenshot(
    frame: np.ndarray,
    prefix: str = "detection",
    directory: Path | None = None,
) -> Path:
    """
    Save the current frame as a PNG and return its path.
    """
    out_dir = _ensure_output_dir(directory)
    path = out_dir / f"{prefix}_{_timestamp()}.png"

    # Two screenshots in the same second must not overwrite each other
    counter = 1
    while path.exists():
        path = out_dir / f"{prefix}_{_timestamp()}_{counter}.png"
        counter += 1

    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Could not save screenshot: {path}")

    logger.info("Screenshot saved: %s", path)
    return path


class VideoWriter:
    """
    A small wrapper for writing processed frames to a video file.

    Use it as a context manager so the file is always closed properly:
        with VideoWriter(path, fps, (w, h)) as writer:
            writer.write(frame)
    """

    def __init__(
        self,
        path: Path | str | None = None,
        fps: float = 30.0,
        frame_size: tuple[int, int] = (1280, 720),
        codec: str | None = None,
        prefix: str = "output",
    ) -> None:
        if path is None:
            path = _ensure_output_dir() / f"{prefix}_{_timestamp()}.mp4"

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Some webcams report 0 or a nonsensical FPS — fall back to something sane
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_size = frame_size
        self.codec = codec or config.VIDEO_CODEC

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(
                f"Could not open video writer: {self.path}\n"
                f"Codec '{self.codec}' may be unsupported. Try setting "
                "config.VIDEO_CODEC to 'XVID' with an .avi output."
            )

        self.frames_written = 0
        logger.info("Writing video to %s (%.1f FPS, %s)", self.path, self.fps, frame_size)

    def write(self, frame: np.ndarray) -> None:
        """Write one frame, resizing it first if the dimensions do not match."""
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video saved: %s (%d frames)", self.path, self.frames_written)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class AccidentLogger:
    """
    Records accident events: one CSV row per event, plus a snapshot of the frame.

    Kept separate from DetectionLogger because the two answer different
    questions. The detection log is a firehose — every object in every frame —
    while this is an incident report, and there should be very few of them. You
    read one by opening the snapshot, not by filtering forty thousand rows.
    """

    HEADER = [
        "timestamp", "frame", "score", "visual", "labels", "track_ids",
        "impact", "sudden_stop", "person_down", "hard_brake", "swerve", "ttc",
        "snapshot",
    ]

    def __init__(
        self,
        path: Path | str | None = None,
        prefix: str = "accidents",
        directory: Path | None = None,
    ) -> None:
        self.directory = _ensure_output_dir(directory)
        if path is None:
            path = self.directory / f"{prefix}_{_timestamp()}.csv"

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events_written = 0

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._file.flush()

        logger.info("Accident log: %s", self.path)

    def log(self, event, frame: np.ndarray | None = None) -> Path | None:
        """
        Write one event. When a frame is supplied, a snapshot is saved alongside
        and its path recorded in the row.
        """
        snapshot: Path | None = None
        if frame is not None:
            snapshot = save_screenshot(
                frame,
                prefix=f"accident_f{event.frame}",
                directory=self.directory,
            )

        signals = event.signals
        self._writer.writerow([
            event.timestamp.isoformat(timespec="milliseconds"),
            event.frame,
            f"{event.score:.3f}",
            f"{event.visual_score:.3f}" if getattr(event, "visual_score", None) is not None else "",
            "|".join(event.labels),
            "|".join(str(t) for t in event.track_ids),
            f"{signals.get('impact', 0.0):.3f}",
            f"{signals.get('sudden_stop', 0.0):.3f}",
            f"{signals.get('person_down', 0.0):.3f}",
            f"{signals.get('hard_brake', 0.0):.3f}",
            f"{signals.get('swerve', 0.0):.3f}",
            f"{signals.get('ttc', 0.0):.3f}",
            str(snapshot) if snapshot else "",
        ])
        # Events are rare and important — flush immediately rather than
        # buffering, so nothing is lost if the process is killed.
        self._file.flush()
        self.events_written += 1
        return snapshot

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("Accident log saved: %s (%d event(s))", self.path, self.events_written)

    def __enter__(self) -> "AccidentLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class DetectionLogger:
    """
    Writes every detection to a CSV: timestamp, frame, object, confidence, box.

    Rows are buffered and flushed periodically rather than written on every
    frame, which would otherwise drag the frame rate down.
    """

    # track_id is only populated in tracking mode; it stays empty otherwise
    HEADER = ["timestamp", "frame", "track_id", "label", "confidence", "x1", "y1", "x2", "y2"]

    def __init__(
        self,
        path: Path | str | None = None,
        prefix: str = "detections",
        flush_every: int = 50,
    ) -> None:
        if path is None:
            path = _ensure_output_dir() / f"{prefix}_{_timestamp()}.csv"

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.rows_written = 0

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._since_flush = 0

        logger.info("Detection log: %s", self.path)

    def log(self, frame_index: int, detections) -> None:
        """Write every detection from one frame."""
        now = datetime.now().isoformat(timespec="milliseconds")
        for det in detections:
            x1, y1, x2, y2 = det.box
            self._writer.writerow(
                [
                    now, frame_index,
                    det.track_id if det.track_id is not None else "",
                    det.label, f"{det.confidence:.4f}", x1, y1, x2, y2,
                ]
            )
            self.rows_written += 1
            self._since_flush += 1

        if self._since_flush >= self.flush_every:
            self._file.flush()
            self._since_flush = 0

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("Log saved: %s (%d rows)", self.path, self.rows_written)

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
