"""
target_lock.py — Locking onto one specific object and following it.

Detection tells you what is in the frame. Tracking gives every object a
persistent ID. **Target lock** is the step beyond that: you pick one ID and the
system follows only that object —

  - a bold highlighted box with corner brackets around the locked target
  - a crosshair, plus the offset from the centre of the frame (which way it is)
  - a trail showing the last N frames of its path
  - every other object dimmed
  - a "LOST" status if the target disappears, and automatic re-acquisition
    when it comes back

Three ways to lock on:
  1. **Click** the object in the video window
  2. `--lock person` — lock the first person that appears
  3. `--lock-id 3` — lock whichever object has track ID 3
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .detect import Detection

# Lock HUD colours (BGR)
_LOCKED = (60, 240, 255)     # amber/yellow — locked target
_LOST = (60, 60, 240)        # red — target missing
_TRAIL = (200, 220, 255)
_TEXT = (20, 20, 20)


class TargetLock:
    """
    Holds the state of a single locked target.

    Usage:
        lock = TargetLock()
        lock.lock(track_id=3)
        target = lock.update(detections)     # once per frame
        lock.draw(frame, target)
    """

    def __init__(self, trail_length: int = 64, lost_after: int = 30) -> None:
        self.track_id: int | None = None
        self.label: str | None = None
        self.trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
        self.lost_after = lost_after      # frames missing before showing "LOST"
        self.frames_missing = 0
        self.frames_locked = 0

    # -- state --------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.track_id is not None

    @property
    def is_lost(self) -> bool:
        """A target is locked, but is not visible in the current frame."""
        return self.is_active and self.frames_missing > 0

    @property
    def is_stale(self) -> bool:
        """Missing long enough that it may not come back at all."""
        return self.frames_missing > self.lost_after

    def lock(self, track_id: int, label: str | None = None) -> None:
        """Lock onto a track ID (this clears any previous trail)."""
        self.track_id = int(track_id)
        self.label = label
        self.trail.clear()
        self.frames_missing = 0
        self.frames_locked = 0

    def clear(self) -> None:
        """Release the lock."""
        self.track_id = None
        self.label = None
        self.trail.clear()
        self.frames_missing = 0
        self.frames_locked = 0

    # -- ways to lock on ----------------------------------------------------

    def lock_at_point(self, detections: list[Detection], x: int, y: int) -> Detection | None:
        """
        Lock onto whatever was clicked. When boxes overlap, the smallest (and
        therefore most specific) object wins.
        """
        hits = [d for d in detections if d.contains(x, y) and d.track_id is not None]
        if not hits:
            return None

        target = min(hits, key=lambda d: d.area)
        self.lock(target.track_id, target.label)
        return target

    def lock_first(self, detections: list[Detection], label: str | None = None) -> Detection | None:
        """
        Lock onto the most prominent object automatically — of the given class
        if a label is supplied, otherwise of any class. The largest and most
        confident object is chosen, since that is usually the main subject.
        """
        pool = [d for d in detections if d.track_id is not None]
        if label:
            pool = [d for d in pool if d.label.lower() == label.lower()]
        if not pool:
            return None

        target = max(pool, key=lambda d: d.area * d.confidence)
        self.lock(target.track_id, target.label)
        return target

    # -- per frame ----------------------------------------------------------

    def update(self, detections: list[Detection]) -> Detection | None:
        """
        Look for the locked target among this frame's detections. If found,
        update the trail and return that Detection; otherwise return None.
        """
        if not self.is_active:
            return None

        for det in detections:
            if det.track_id == self.track_id:
                self.trail.append(det.center)
                self.frames_missing = 0
                self.frames_locked += 1
                self.label = det.label
                return det

        self.frames_missing += 1
        return None

    # -- drawing ------------------------------------------------------------

    def draw(self, frame: np.ndarray, target: Detection | None) -> np.ndarray:
        """Draw the full lock HUD: brackets, crosshair, trail and status."""
        if not self.is_active:
            return frame

        self._draw_trail(frame)

        if target is None:
            self._draw_lost_banner(frame)
            return frame

        x1, y1, x2, y2 = target.box
        cx, cy = target.center

        # Bold box plus corner brackets — the classic "lock-on" look
        cv2.rectangle(frame, (x1, y1), (x2, y2), _LOCKED, 2)
        self._draw_brackets(frame, x1, y1, x2, y2)

        # Crosshair at the centre of the target
        cv2.drawMarker(frame, (cx, cy), _LOCKED, cv2.MARKER_CROSS, 22, 2)

        # A line from the frame centre to the target — shows which way it lies
        fh, fw = frame.shape[:2]
        frame_center = (fw // 2, fh // 2)
        cv2.line(frame, frame_center, (cx, cy), _LOCKED, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, frame_center, (180, 180, 180), cv2.MARKER_CROSS, 12, 1)

        self._draw_label(frame, target, x1, y1, x2, y2, frame_center)
        return frame

    def _draw_brackets(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """L-shaped brackets at all four corners."""
        arm = max(12, min(x2 - x1, y2 - y1) // 5)
        t = 3
        for (px, py), (dx, dy) in (
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ):
            cv2.line(frame, (px, py), (px + arm * dx, py), _LOCKED, t)
            cv2.line(frame, (px, py), (px, py + arm * dy), _LOCKED, t)

    def _draw_trail(self, frame: np.ndarray) -> None:
        """The target's recent path — older points thinner and more faded."""
        points = list(self.trail)
        for i in range(1, len(points)):
            weight = i / len(points)  # newer points stand out more
            cv2.line(
                frame,
                points[i - 1],
                points[i],
                tuple(int(c * weight) for c in _TRAIL),
                max(1, int(3 * weight)),
                cv2.LINE_AA,
            )

    def _draw_label(
        self,
        frame: np.ndarray,
        target: Detection,
        x1: int, y1: int, x2: int, y2: int,
        frame_center: tuple[int, int],
    ) -> None:
        """An info chip above the target: ID, label and offset from centre."""
        cx, cy = target.center
        dx, dy = cx - frame_center[0], cy - frame_center[1]

        horizontal = "R" if dx > 0 else "L"
        vertical = "D" if dy > 0 else "U"
        text = (
            f"LOCK #{target.track_id} {target.label} "
            f"{target.confidence * 100:.0f}%  "
            f"{horizontal}{abs(dx)} {vertical}{abs(dy)}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, 0.5, 1)

        # Chip above the box, or below it when there is no room
        chip_bottom = y1 - 6 if y1 - th - baseline - 10 >= 0 else y2 + th + baseline + 10
        chip_top = chip_bottom - th - baseline - 6
        chip_x = max(0, min(x1, frame.shape[1] - tw - 12))

        cv2.rectangle(frame, (chip_x, chip_top), (chip_x + tw + 12, chip_bottom), _LOCKED, -1)
        cv2.putText(
            frame, text, (chip_x + 6, chip_bottom - baseline - 3),
            font, 0.5, _TEXT, 1, cv2.LINE_AA,
        )

    def _draw_lost_banner(self, frame: np.ndarray) -> None:
        """A warning across the centre of the screen when the target is missing."""
        h, w = frame.shape[:2]
        text = (
            f"TARGET #{self.track_id} LOST ({self.frames_missing}f)"
            if not self.is_stale
            else f"TARGET #{self.track_id} LOST - press c to clear"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        x, y = (w - tw) // 2, h // 2

        cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), (30, 30, 30), -1)
        cv2.putText(frame, text, (x, y), font, 0.6, _LOST, 2, cv2.LINE_AA)

    # -- reporting ----------------------------------------------------------

    def status_line(self) -> str:
        """A short status line for the overlay panel."""
        if not self.is_active:
            return "Lock: none (click on object)"
        if self.is_lost:
            return f"Lock: #{self.track_id} LOST ({self.frames_missing}f)"
        return f"Lock: #{self.track_id} {self.label} ({self.frames_locked}f)"
