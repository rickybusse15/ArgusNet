"""Timestamp utilities for ArgusNet."""

from __future__ import annotations

import time as _time

__all__ = [
    "now_s",
    "now_ms",
    "sim_to_epoch",
    "epoch_to_sim",
    "format_duration",
    "TimestampedMixin",
]


def now_s() -> float:
    """Wall-clock time in seconds (Unix epoch)."""
    return _time.time()


def now_ms() -> float:
    """Wall-clock time in milliseconds (Unix epoch)."""
    return _time.time() * 1e3


def sim_to_epoch(sim_s: float, epoch_origin_s: float | None = None) -> float:
    """Convert simulation-relative seconds to Unix epoch seconds.

    If *epoch_origin_s* is None, the current wall-clock time is used as the
    origin so that t=0 in simulation maps to "now".
    """
    origin = epoch_origin_s if epoch_origin_s is not None else now_s()
    return origin + sim_s


def epoch_to_sim(epoch_s: float, epoch_origin_s: float) -> float:
    """Convert Unix epoch seconds to simulation-relative seconds."""
    return epoch_s - epoch_origin_s


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.mmm."""
    negative = seconds < 0
    seconds = abs(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    result = f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"-{result}" if negative else result


class TimestampedMixin:
    """Mixin providing a ``timestamp_s`` attribute and age computation."""

    timestamp_s: float

    def age_s(self, now: float | None = None) -> float:
        """Seconds elapsed since this object's timestamp."""
        t = now if now is not None else now_s()
        return t - self.timestamp_s
