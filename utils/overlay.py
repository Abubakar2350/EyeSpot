"""
overlay.py — Screen ke upar info panel draw karna (FPS, object counts, status).

Ye purely cosmetic hai; detection logic se bilkul alag rakha gaya hai.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_BG = (28, 28, 28)
_TEXT = (240, 240, 240)
_ACCENT = (80, 220, 120)
_WARN = (60, 180, 255)


def _fps_color(fps: float) -> tuple[int, int, int]:
    """Achha FPS = hara, kam FPS = orange, bohot kam = laal."""
    if fps >= 20:
        return _ACCENT
    if fps >= 10:
        return _WARN
    return (60, 60, 240)


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    counts: dict[str, int],
    confidence: float,
    paused: bool = False,
    extra_lines: list[str] | None = None,
    max_items: int = 8,
) -> np.ndarray:
    """
    Frame ke top-left par ek semi-transparent panel banao jismein:
    FPS, total objects, per-class counts, aur current confidence threshold ho.
    """
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"FPS: {fps:5.1f}", _fps_color(fps)),
        (f"Objects: {sum(counts.values())}   Conf: {confidence:.2f}", _TEXT),
    ]

    for label, count in list(counts.items())[:max_items]:
        lines.append((f"  {label} x{count}", _TEXT))

    hidden = len(counts) - max_items
    if hidden > 0:
        lines.append((f"  +{hidden} more...", _TEXT))

    for extra in extra_lines or []:
        lines.append((extra, _TEXT))

    if paused:
        lines.append(("[ PAUSED ]", _WARN))

    # Panel ka size text ke hisaab se
    pad = 10
    line_h = 22
    width = max(
        cv2.getTextSize(text, _FONT, 0.5, 1)[0][0] for text, _ in lines
    ) + pad * 2
    height = line_h * len(lines) + pad

    # Semi-transparent background (frame ke saath blend kar ke)
    panel_w = min(width, frame.shape[1] - 20)
    panel_h = min(height, frame.shape[0] - 20)
    roi = frame[10 : 10 + panel_h, 10 : 10 + panel_w]
    if roi.size:
        overlay = np.full(roi.shape, _PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

    y = 10 + pad + 8
    for text, color in lines:
        if y > 10 + panel_h:
            break
        cv2.putText(frame, text, (10 + pad, y), _FONT, 0.5, color, 1, cv2.LINE_AA)
        y += line_h

    return frame


def draw_help_bar(frame: np.ndarray, text: str) -> np.ndarray:
    """Frame ke neeche keyboard shortcuts ki ek line dikhao."""
    h, w = frame.shape[:2]
    bar_h = 26
    roi = frame[h - bar_h : h, 0:w]
    if roi.size:
        overlay = np.full(roi.shape, _PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)
    cv2.putText(frame, text, (10, h - 8), _FONT, 0.45, _TEXT, 1, cv2.LINE_AA)
    return frame
