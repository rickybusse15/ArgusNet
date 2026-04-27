"""Sensor frame synchronisation for ArgusNet.

Aligns observations from multiple sensors to a common target timestamp
using nearest-neighbour or interpolation strategies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = [
    "SyncBuffer",
    "NearestNeighbourSync",
    "FrameSyncResult",
]

T = TypeVar("T")


@dataclass
class _TimestampedItem(Generic[T]):
    timestamp_s: float
    data: T


@dataclass
class FrameSyncResult(Generic[T]):
    """Result of a synchronisation query."""

    target_s: float
    """The requested target timestamp."""

    items: dict[str, T]
    """Per-sensor data items, keyed by sensor ID. Missing sensors are omitted."""

    latencies: dict[str, float]
    """Time difference |item.timestamp - target_s| per sensor ID."""


class SyncBuffer(Generic[T]):
    """Rolling time-ordered buffer for a single sensor stream.

    Stores the last *max_history* items and supports nearest-neighbour
    lookup by timestamp.
    """

    def __init__(self, sensor_id: str, max_history: int = 64) -> None:
        self.sensor_id = sensor_id
        self._buf: deque[_TimestampedItem[T]] = deque(maxlen=max_history)

    def push(self, timestamp_s: float, data: T) -> None:
        """Add a new item to the buffer (must be monotonically increasing)."""
        self._buf.append(_TimestampedItem(timestamp_s=timestamp_s, data=data))

    def nearest(self, target_s: float, max_age_s: float = 0.5) -> tuple[float, T] | None:
        """Return the (timestamp, data) item closest to *target_s*.

        Returns None if the buffer is empty or the closest item is older
        than *max_age_s*.
        """
        if not self._buf:
            return None
        best = min(self._buf, key=lambda it: abs(it.timestamp_s - target_s))
        if abs(best.timestamp_s - target_s) > max_age_s:
            return None
        return best.timestamp_s, best.data

    def latest(self) -> tuple[float, T] | None:
        if not self._buf:
            return None
        it = self._buf[-1]
        return it.timestamp_s, it.data


class NearestNeighbourSync(Generic[T]):
    """Multi-sensor nearest-neighbour synchroniser.

    Maintains one :class:`SyncBuffer` per registered sensor.  On each
    call to :meth:`sync`, returns the nearest available item from each
    sensor within *max_age_s* of the requested target timestamp.
    """

    def __init__(self, max_age_s: float = 0.2, max_history: int = 64) -> None:
        self.max_age_s = max_age_s
        self.max_history = max_history
        self._buffers: dict[str, SyncBuffer[T]] = {}

    def register(self, sensor_id: str) -> None:
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = SyncBuffer(sensor_id, self.max_history)

    def push(self, sensor_id: str, timestamp_s: float, data: T) -> None:
        if sensor_id not in self._buffers:
            self.register(sensor_id)
        self._buffers[sensor_id].push(timestamp_s, data)

    def sync(self, target_s: float) -> FrameSyncResult[T]:
        """Retrieve nearest-neighbour items from all registered sensors."""
        items: dict[str, T] = {}
        latencies: dict[str, float] = {}
        for sid, buf in self._buffers.items():
            result = buf.nearest(target_s, self.max_age_s)
            if result is not None:
                ts, data = result
                items[sid] = data
                latencies[sid] = abs(ts - target_s)
        return FrameSyncResult(target_s=target_s, items=items, latencies=latencies)
