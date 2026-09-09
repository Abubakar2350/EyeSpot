"""
detect.py — Detection logic: ek frame lo, objects detect karo, boxes draw karo.

Design: `Detector` class model ko hold karti hai aur har frame par
`detect()` call hota hai. Drawing alag method (`draw`) mein hai taake
Streamlit UI ya koi aur frontend sirf raw detections bhi le sakay.
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
    """Ek detected object — plain data, koi YOLO-specific cheez nahi."""

    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels mein

    # Tracking mode mein har object ko ek persistent ID milti hai jo frames ke
    # darmiyan same rehti hai. Detection-only mode mein ye None hoti hai.
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
        """Kya ye point is box ke andar hai? (mouse click se target lock karne ke liye)"""
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


def _build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """
    Har class ke liye ek distinct BGR color banao (HSV wheel ko evenly ghumaa ke).
    Isse similar objects hamesha same color ke rehte hain.
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(max(num_classes, 1)):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio = achhi spread
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))  # OpenCV = BGR
    return palette


class Detector:
    """
    YOLO detector ka wrapper.

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
        # Empty list => saari classes. Ultralytics ko None chahiye is case mein.
        self.class_ids = class_name_to_id(self.model, names) or None

        self._palette = _build_palette(len(self.model.names))

    # -- properties ---------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Model ki saari class labels (sorted)."""
        return [self.model.names[i] for i in sorted(self.model.names)]

    def set_confidence(self, value: float) -> float:
        """Confidence threshold ko 0.05–0.95 ke beech clamp kar ke set karo."""
        self.confidence = float(min(0.95, max(0.05, value)))
        return self.confidence

    # -- core ---------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Ek BGR frame par YOLO chalao aur Detection objects ki list return karo.

        Ye stateless hai — har frame independent. Objects ko frames ke darmiyan
        follow karna ho to `track()` use karein.
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
        # with_ids=False zaroori hai: ek baar track() chalne ke baad Ultralytics
        # apne tracker callbacks predictor par lagi rehne deta hai, is liye
        # predict() bhi IDs return karne lagta hai. detect() stateless rehna
        # chahiye, warna tracking off karne par bhi purani IDs aati rahengi.
        return self._parse_results(results, frame, with_ids=False)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        `detect()` jaisa hi, lekin har object ko ek **persistent track ID** milti hai
        jo frames ke darmiyan same rehti hai.

        Isi wajah se "target lock" mumkin hota hai: aap ek ID pakad lete hain
        aur wo object jab tak frame mein hai, usi ID se pehchana jaata hai —
        chahe wo hile, doosron ke peeche chhup jaye, ya thodi der ke liye ghayab ho.

        persist=True ka matlab: Ultralytics tracker ki state calls ke darmiyan
        yaad rakhta hai (isliye har frame par nayi IDs nahi banti).
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
        Tracker ki state saaf karo (IDs 1 se dobara shuru).
        Naya video/stream shuru karte waqt call karein.
        """
        for predictor_attr in ("predictor",):
            predictor = getattr(self.model, predictor_attr, None)
            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    tracker.reset()

    def _parse_results(self, results, frame: np.ndarray, with_ids: bool = True) -> list[Detection]:
        """Ultralytics ke raw output ko humare Detection objects mein badlo."""
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

        # Tracking mode mein boxes.id mojood hoti hai; detect mode mein None
        if with_ids and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = [None] * len(xyxy)

        for (x1, y1, x2, y2), conf, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
            # Box ko frame ke andar clamp karo (kabhi kabhi model bahar nikal jaata hai)
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
        Frame par bounding boxes + labels draw karo (in-place, wohi frame return hota hai).

        dim_ids: in track IDs ko halka (faded) draw karo — target lock mode mein
        baaki objects ko dim kar ke locked target ko numaya karne ke liye.
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
                # Faded color + patli line — nazar locked target par rahe
                color = tuple(int(c * 0.35 + 60) for c in color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            text = f"{det.label} {det.confidence * 100:.0f}%" if show_confidence else det.label
            if det.track_id is not None:
                text = f"#{det.track_id} {text}"  # ID sabse pehle — pehchanne mein aasan
            (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)

            # Label ka background: box ke upar, lekin agar upar jagah na ho to box ke andar
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
    """Background color ke hisaab se text kaala ya safed choose karo."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def count_by_label(detections: list[Detection]) -> dict[str, int]:
    """
    Detections ko label ke hisaab se count karo, sabse zyada wala pehle.
    Overlay panel aur reporting dono isko use karte hain.
    """
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
