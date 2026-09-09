"""
target_lock.py — Ek specific object par "lock" kar ke usay follow karna.

Detection batata hai ke frame mein kya kya hai. Tracking har object ko ek
persistent ID deti hai. **Target lock** us se agla qadam hai: aap ek ID chun
lete hain, aur system sirf usi object ko follow karta hai —

  - locked target ke around motta highlighted box + corner brackets
  - crosshair aur frame ke center se offset (kis taraf ja raha hai)
  - trail (pichhle N frames ka raasta)
  - baaki sab objects dim ho jaate hain
  - target ghayab ho jaye to "LOST" status, aur wapas aane par khud dobara pakad leta hai

Lock karne ke tareeqe:
  1. Video window mein object par **mouse click**
  2. `--lock person` — pehle nazar aane wale person par khud lock kar lo
  3. `--lock-id 3` — jis ki track ID 3 ho
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .detect import Detection

# Lock UI ke colors (BGR)
_LOCKED = (60, 240, 255)     # amber/yellow — locked target
_LOST = (60, 60, 240)        # red — target ghayab
_TRAIL = (200, 220, 255)
_TEXT = (20, 20, 20)


class TargetLock:
    """
    Ek locked target ki state sambhalta hai.

    Usage:
        lock = TargetLock()
        lock.lock(track_id=3)
        target = lock.update(detections)     # har frame par
        lock.draw(frame, target)
    """

    def __init__(self, trail_length: int = 64, lost_after: int = 30) -> None:
        self.track_id: int | None = None
        self.label: str | None = None
        self.trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
        self.lost_after = lost_after      # itne frames na dikhe to "LOST"
        self.frames_missing = 0
        self.frames_locked = 0

    # -- state --------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.track_id is not None

    @property
    def is_lost(self) -> bool:
        """Target locked to hai, lekin filhal frame mein nazar nahi aa raha."""
        return self.is_active and self.frames_missing > 0

    @property
    def is_stale(self) -> bool:
        """Itni der se ghayab hai ke shayad wapas na aaye."""
        return self.frames_missing > self.lost_after

    def lock(self, track_id: int, label: str | None = None) -> None:
        """Kisi track ID par lock karo (purani trail saaf ho jaati hai)."""
        self.track_id = int(track_id)
        self.label = label
        self.trail.clear()
        self.frames_missing = 0
        self.frames_locked = 0

    def clear(self) -> None:
        """Lock hata do."""
        self.track_id = None
        self.label = None
        self.trail.clear()
        self.frames_missing = 0
        self.frames_locked = 0

    # -- lock karne ke tareeqe ----------------------------------------------

    def lock_at_point(self, detections: list[Detection], x: int, y: int) -> Detection | None:
        """
        Mouse click par lock karo. Agar overlapping boxes hon to sabse chhota
        (yani sabse specific) object chunte hain.
        """
        hits = [d for d in detections if d.contains(x, y) and d.track_id is not None]
        if not hits:
            return None

        target = min(hits, key=lambda d: d.area)
        self.lock(target.track_id, target.label)
        return target

    def lock_first(self, detections: list[Detection], label: str | None = None) -> Detection | None:
        """
        Sabse "aham" object par khud lock karo — agar label diya hai to us class ka,
        warna kisi bhi class ka. Sabse bara + confident object chunte hain
        (ke wo aam tor par main subject hota hai).
        """
        pool = [d for d in detections if d.track_id is not None]
        if label:
            pool = [d for d in pool if d.label.lower() == label.lower()]
        if not pool:
            return None

        target = max(pool, key=lambda d: d.area * d.confidence)
        self.lock(target.track_id, target.label)
        return target

    # -- har frame ----------------------------------------------------------

    def update(self, detections: list[Detection]) -> Detection | None:
        """
        Is frame ki detections mein se locked target dhoondo.
        Mil jaye to trail update kar ke wo Detection return karo, warna None.
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
        """Locked target ka pura HUD draw karo (brackets, crosshair, trail, status)."""
        if not self.is_active:
            return frame

        self._draw_trail(frame)

        if target is None:
            self._draw_lost_banner(frame)
            return frame

        x1, y1, x2, y2 = target.box
        cx, cy = target.center

        # Motta box + corner brackets (classic "lock-on" look)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _LOCKED, 2)
        self._draw_brackets(frame, x1, y1, x2, y2)

        # Crosshair target ke center par
        cv2.drawMarker(frame, (cx, cy), _LOCKED, cv2.MARKER_CROSS, 22, 2)

        # Frame ke center se target tak line — "kis taraf hai" ka andaza
        fh, fw = frame.shape[:2]
        frame_center = (fw // 2, fh // 2)
        cv2.line(frame, frame_center, (cx, cy), _LOCKED, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, frame_center, (180, 180, 180), cv2.MARKER_CROSS, 12, 1)

        self._draw_label(frame, target, x1, y1, x2, y2, frame_center)
        return frame

    def _draw_brackets(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """Chaar corners par L-shape brackets."""
        arm = max(12, min(x2 - x1, y2 - y1) // 5)
        t = 3
        for (px, py), (dx, dy) in (
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ):
            cv2.line(frame, (px, py), (px + arm * dx, py), _LOCKED, t)
            cv2.line(frame, (px, py), (px, py + arm * dy), _LOCKED, t)

    def _draw_trail(self, frame: np.ndarray) -> None:
        """Target ka pichhla raasta — purane points patle aur faded."""
        points = list(self.trail)
        for i in range(1, len(points)):
            weight = i / len(points)  # naye points zyada numaya
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
        """Target ke upar ek info chip: ID, label, aur center se offset."""
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

        # Chip box ke upar, ya jagah na ho to neeche
        chip_bottom = y1 - 6 if y1 - th - baseline - 10 >= 0 else y2 + th + baseline + 10
        chip_top = chip_bottom - th - baseline - 6
        chip_x = max(0, min(x1, frame.shape[1] - tw - 12))

        cv2.rectangle(frame, (chip_x, chip_top), (chip_x + tw + 12, chip_bottom), _LOCKED, -1)
        cv2.putText(
            frame, text, (chip_x + 6, chip_bottom - baseline - 3),
            font, 0.5, _TEXT, 1, cv2.LINE_AA,
        )

    def _draw_lost_banner(self, frame: np.ndarray) -> None:
        """Target ghayab ho to screen ke center mein warning."""
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
        """Overlay panel ke liye ek chhoti si status line."""
        if not self.is_active:
            return "Lock: none (click on object)"
        if self.is_lost:
            return f"Lock: #{self.track_id} LOST ({self.frames_missing}f)"
        return f"Lock: #{self.track_id} {self.label} ({self.frames_locked}f)"
