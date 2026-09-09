"""
detect.py — Detection logic: take a frame, find objects, draw boxes.

Design: the `Detector` class holds the model and `detect()` is called once per
frame. Drawing lives in a separate method (`draw`) so that a Streamlit UI, or
any other frontend, can consume the raw detections instead.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

import config
from .model_loader import class_name_to_id, load_model


@dataclass(frozen=True)
class Detection:
    """A single detected object — plain data, nothing YOLO-specific."""

    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels

    # In tracking mode every object gets a persistent ID that stays the same
    # across frames. It is None in detection-only mode.
    track_id: int | None = None

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def contains(self, x: int, y: int) -> bool:
        """Is this point inside the box? (used to lock a target by mouse click)"""
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


def _build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """
    Build a distinct BGR colour per class by stepping evenly around the HSV
    wheel, so that objects of the same class always share a colour.
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(max(num_classes, 1)):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio gives a good spread
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))  # OpenCV = BGR
    return palette


class Detector:
    """
    A thin wrapper around the YOLO model.

    Usage:
        det = Detector()
        detections = det.detect(frame)
        frame = det.draw(frame, detections)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        class_filter: Iterable[str] | None = None,
        image_size: int | None = None,
        tracker: str | None = None,
    ) -> None:
        self.model, self.device = load_model(model_name, device)
        self.confidence = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        self.iou = iou if iou is not None else config.IOU_THRESHOLD
        self.image_size = image_size or config.IMAGE_SIZE
        self.tracker = tracker or config.TRACKER

        names = list(class_filter) if class_filter else list(config.CLASS_FILTER or [])
        self.class_filter_names = names
        # An empty list means all classes; Ultralytics expects None in that case.
        self.class_ids = class_name_to_id(self.model, names) or None

        self._palette = _build_palette(len(self.model.names))

    # -- properties ---------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Every class label the model knows, sorted by class ID."""
        return [self.model.names[i] for i in sorted(self.model.names)]

    def set_confidence(self, value: float) -> float:
        """Clamp the confidence threshold to 0.05–0.95 and apply it."""
        self.confidence = float(min(0.95, max(0.05, value)))
        return self.confidence

    # -- core ---------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLO on a single BGR frame and return a list of Detection objects.

        This is stateless — every frame is independent. Use `track()` to follow
        objects across frames.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.class_ids,
            max_det=config.MAX_DETECTIONS,
            device=self.device,
            verbose=False,
        )
        # with_ids=False matters here: once track() has run, Ultralytics leaves
        # its tracker callbacks attached to the predictor, so predict() starts
        # returning IDs too. detect() must stay stateless, otherwise stale IDs
        # keep appearing even after tracking is turned off.
        return self._parse_results(results, frame, with_ids=False)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        Same as `detect()`, but each object also gets a **persistent track ID**
        that stays the same from frame to frame.

        That is what makes target lock possible: you hold on to one ID, and the
        object keeps that identity for as long as it is in frame — even as it
        moves, hides behind something, or briefly disappears.

        persist=True tells Ultralytics to keep the tracker's state between
        calls, so new IDs are not minted on every frame.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.class_ids,
            max_det=config.MAX_DETECTIONS,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        return self._parse_results(results, frame)

    def reset_tracker(self) -> None:
        """
        Clear the tracker's state so IDs restart from 1.
        Call this when starting a new video or stream.
        """
        for predictor_attr in ("predictor",):
            predictor = getattr(self.model, predictor_attr, None)
            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    tracker.reset()

    def _parse_results(self, results, frame: np.ndarray, with_ids: bool = True) -> list[Detection]:
        """Convert Ultralytics' raw output into our own Detection objects."""
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        h, w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # boxes.id exists in tracking mode and is None in detection mode
        if with_ids and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = [None] * len(xyxy)

        for (x1, y1, x2, y2), conf, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
            # Clamp the box to the frame — the model occasionally spills outside
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    label=self.model.names.get(int(cls_id), str(cls_id)),
                    confidence=float(conf),
                    box=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )

        return detections

    # -- drawing ------------------------------------------------------------

    def color_for(self, class_id: int) -> tuple[int, int, int]:
        return self._palette[class_id % len(self._palette)]

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        show_confidence: bool | None = None,
        dim_ids: set[int] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame, in place, and return it.

        dim_ids: draw these track IDs faded. Target-lock mode uses this to dim
        everything except the locked target.
        """
        if show_confidence is None:
            show_confidence = config.SHOW_CONFIDENCE

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = config.BOX_THICKNESS
        scale = config.FONT_SCALE

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self.color_for(det.class_id)

            dimmed = dim_ids is not None and det.track_id in dim_ids
            if dimmed:
                # Faded colour and a thinner line, so the eye stays on the target
                color = tuple(int(c * 0.35 + 60) for c in color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            text = f"{det.label} {det.confidence * 100:.0f}%" if show_confidence else det.label
            if det.track_id is not None:
                text = f"#{det.track_id} {text}"  # ID first — easiest to spot
            (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)

            # Label background sits above the box, or inside it when there is no room
            label_bottom = y1 if y1 - th - baseline - 4 >= 0 else y2 + th + baseline + 4
            label_top = label_bottom - th - baseline - 4

            cv2.rectangle(frame, (x1, label_top), (x1 + tw + 6, label_bottom), color, -1)
            cv2.putText(
                frame,
                text,
                (x1 + 3, label_bottom - baseline - 2),
                font,
                scale,
                _contrast_color(color),
                1,
                cv2.LINE_AA,
            )

        return frame


def _contrast_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text depending on the background's luminance."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def count_by_label(detections: list[Detection]) -> dict[str, int]:
    """
    Count detections per label, most frequent first.
    Used by both the overlay panel and reporting.
    """
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
