"""
fps_counter.py — FPS (frames per second) calculate karne ka simple helper.

Har frame ka time note karte hain aur last N frames ka moving average
nikalte hain — isse number jhatke nahi khaata aur padhne mein aasan rehta hai.
"""

from __future__ import annotations

import time
from collections import deque

import config


class FPSCounter:
    """
    Usage:
        fps = FPSCounter()
        while True:
            ...frame process karo...
            fps.update()
            print(fps.value)
    """

    def __init__(self, window: int | None = None) -> None:
        self.window = window or config.FPS_SMOOTHING_WINDOW
        self._durations: deque[float] = deque(maxlen=self.window)
        self._last_tick: float | None = None
        self._start = time.perf_counter()
        self.frame_count = 0

    def update(self) -> float:
        """Ek frame complete hone par call karein. Current FPS return karta hai."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._durations.append(now - self._last_tick)
        self._last_tick = now
        self.frame_count += 1
        return self.value

    @property
    def value(self) -> float:
        """Moving-average FPS. Shuru mein (data kam ho to) 0.0 return karta hai."""
        if not self._durations:
            return 0.0
        avg = sum(self._durations) / len(self._durations)
        return 1.0 / avg if avg > 0 else 0.0

    @property
    def average(self) -> float:
        """Poore session ka overall average FPS."""
        elapsed = time.perf_counter() - self._start
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def elapsed(self) -> float:
        """Session ko shuru hue kitne seconds ho gaye."""
        return time.perf_counter() - self._start

    def reset(self) -> None:
        """Pause ke baad resume karte waqt useful — stale timings hata deta hai."""
        self._durations.clear()
        self._last_tick = None
