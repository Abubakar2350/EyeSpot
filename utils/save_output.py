"""
save_output.py — Screenshots, output video, aur CSV detection log save karna.

Sab kuch config.OUTPUT_DIR ke andar jaata hai, timestamped filenames ke sath
taake purani files overwrite na hon.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


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
    Current frame ko PNG file mein save karo aur uska path return karo.
    """
    out_dir = _ensure_output_dir(directory)
    path = out_dir / f"{prefix}_{_timestamp()}.png"

    # Same second mein do screenshots lein to overwrite na ho
    counter = 1
    while path.exists():
        path = out_dir / f"{prefix}_{_timestamp()}_{counter}.png"
        counter += 1

    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Screenshot save nahi ho saka: {path}")

    logger.info("Screenshot saved: %s", path)
    return path


class VideoWriter:
    """
    Processed frames ko ek video file mein likhne ke liye chhota wrapper.

    Context manager ki tarah use karein taake file hamesha properly close ho:
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

        # Kuch webcams 0 ya ajeeb FPS report karte hain — safe fallback
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_size = frame_size
        self.codec = codec or config.VIDEO_CODEC

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(
                f"Video writer open nahi hua: {self.path}\n"
                f"Codec '{self.codec}' shayad support nahi. config.VIDEO_CODEC "
                "ko 'XVID' (.avi ke sath) try karein."
            )

        self.frames_written = 0
        logger.info("Writing video to %s (%.1f FPS, %s)", self.path, self.fps, frame_size)

    def write(self, frame: np.ndarray) -> None:
        """Ek frame likho. Size mismatch ho to resize kar deta hai."""
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


class DetectionLogger:
    """
    Har detection ko CSV mein likhta hai: timestamp, frame, object, confidence, box.

    Rows buffer mein jama hote hain aur thodi thodi der baad flush hote hain
    taake har frame par disk write na ho (warna FPS gir jaata hai).
    """

    # track_id sirf tracking mode mein bharta hai, warna khali rehta hai
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
        """Ek frame ki saari detections likho."""
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
