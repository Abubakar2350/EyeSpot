"""
fps_counter.py — A small helper for measuring frames per second.

Each frame's duration is recorded and a moving average over the last N frames
is reported, so the number stays steady instead of jumping around.
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
            ...process a frame...
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
        """Call once per completed frame. Returns the current FPS."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._durations.append(now - self._last_tick)
        self._last_tick = now
        self.frame_count += 1
        return self.value

    @property
    def value(self) -> float:
        """Moving-average FPS. Returns 0.0 until enough samples are collected."""
        if not self._durations:
            return 0.0
        avg = sum(self._durations) / len(self._durations)
        return 1.0 / avg if avg > 0 else 0.0

    @property
    def average(self) -> float:
        """Overall average FPS for the whole session."""
        elapsed = time.perf_counter() - self._start
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def elapsed(self) -> float:
        """Seconds since the session started."""
        return time.perf_counter() - self._start

    def reset(self) -> None:
        """Drop stale timings — useful when resuming after a pause."""
        self._durations.clear()
        self._last_tick = None
