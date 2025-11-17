"""
src/core/timer.py

High-resolution timer & simple profiler utilities for the Chess3D engine.

Provides:
- Timer: class with update(), delta_time (seconds), total_time
- FPSCounter: rolling average FPS measurement
- profile() context manager for quick timing

Usage:
    timer = Timer()
    while running:
        timer.update()         # updates delta_time
        dt = timer.delta_time
        # game logic...
        fps = timer.fps.get_fps()
"""

from __future__ import annotations
import time
from collections import deque
from contextlib import contextmanager
from typing import Optional


class Timer:
    def __init__(self):
        self._last = time.perf_counter()
        self.delta_time: float = 1.0 / 60.0
        self.total_time: float = 0.0
        self.fps = FPSCounter(window=1.0)

    def update(self) -> None:
        """Call once per frame to update delta_time and fps."""
        now = time.perf_counter()
        dt = now - self._last
        # clamp unreasonable dt (e.g., when debugger pauses)
        if dt <= 0.0:
            dt = 1e-6
        if dt > 1.0:
            # if a long pause occurred, cap to 1 second to avoid physics blowups
            dt = 1.0
        self.delta_time = dt
        self.total_time += dt
        self._last = now
        self.fps.update(dt)

    @property
    def elapsed(self) -> float:
        """Alias for total_time"""
        return self.total_time


class FPSCounter:
    """
    Rolling FPS average. Keeps samples for a sliding window (seconds).
    Default window = 1.0s (approximate recent FPS).
    """

    def __init__(self, window: float = 1.0):
        self.window = max(0.1, float(window))
        self._samples = deque()  # stores (timestamp, dt)
        self._acc_dt = 0.0
        self._now = time.perf_counter()

    def update(self, dt: float) -> None:
        """Append a new frame dt and remove old ones outside window."""
        self._now = time.perf_counter()
        self._samples.append((self._now, dt))
        self._acc_dt += dt

        # Drop samples older than the window from the left
        cutoff = self._now - self.window
        while self._samples and self._samples[0][0] < cutoff:
            _, old_dt = self._samples.popleft()
            self._acc_dt -= old_dt
            if self._acc_dt < 0:
                self._acc_dt = 0.0

    def get_fps(self) -> float:
        """Return the rolling FPS (frames per second)."""
        if not self._samples:
            return 0.0
        total_dt = self._acc_dt
        if total_dt <= 0.0:
            return 0.0
        return len(self._samples) / total_dt

    def get_frame_time_ms(self) -> float:
        """Return average frame time in milliseconds over the window."""
        fps = self.get_fps()
        if fps <= 0.0:
            return 0.0
        return 1000.0 / fps


@contextmanager
def profile(label: Optional[str] = None):
    """
    Simple context manager for quick profiling of code blocks.
    Example:
        with profile("update_physics"):
            update_physics(dt)
    Prints elapsed time in ms to stdout.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        if label:
            print(f"[profile] {label}: {elapsed_ms:.3f} ms")
        else:
            print(f"[profile] {elapsed_ms:.3f} ms")
