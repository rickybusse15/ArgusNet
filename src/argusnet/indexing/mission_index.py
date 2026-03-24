"""Mission event index for ArgusNet.

Records per-mission metadata: keyframe sets, coverage statistics,
flight paths, and timing.  Supports query by mission ID and
summary aggregation across missions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from argusnet.core.ids import new_mission_id
from argusnet.core.types import Vector3

__all__ = [
    "MissionRecord",
    "MissionIndex",
]


@dataclass
class MissionRecord:
    """Summary of a single mission's data collection."""

    mission_id: str
    """Unique mission identifier (``msn-<uuid>``)."""

    start_time_s: float = 0.0
    """Mission start timestamp (seconds)."""

    end_time_s: float = 0.0
    """Mission end timestamp (seconds)."""

    keyframe_ids: List[str] = field(default_factory=list)
    """Ordered list of keyframe IDs captured during this mission."""

    flight_path: List[Vector3] = field(default_factory=list)
    """Waypoints traversed (metres, world frame)."""

    total_coverage_m2: float = 0.0
    """Cumulative area covered (m²)."""

    total_distance_m: float = 0.0
    """Total distance flown (m)."""

    sensor_ids: List[str] = field(default_factory=list)
    """Sensors used during the mission."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata (weather, operator notes, etc.)."""

    @property
    def duration_s(self) -> float:
        return max(self.end_time_s - self.start_time_s, 0.0)

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframe_ids)

    def add_keyframe(self, keyframe_id: str) -> None:
        self.keyframe_ids.append(keyframe_id)

    def add_waypoint(self, position: Vector3) -> None:
        """Append a waypoint and update total distance."""
        if self.flight_path:
            prev = np.asarray(self.flight_path[-1], dtype=float)
            curr = np.asarray(position, dtype=float)
            self.total_distance_m += float(np.linalg.norm(curr - prev))
        self.flight_path.append(position)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "keyframe_ids": self.keyframe_ids,
            "flight_path": [list(p) for p in self.flight_path],
            "total_coverage_m2": self.total_coverage_m2,
            "total_distance_m": self.total_distance_m,
            "sensor_ids": self.sensor_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MissionRecord:
        path = data.get("flight_path", [])
        return cls(
            mission_id=data.get("mission_id", new_mission_id()),
            start_time_s=data.get("start_time_s", 0.0),
            end_time_s=data.get("end_time_s", 0.0),
            keyframe_ids=list(data.get("keyframe_ids", [])),
            flight_path=[tuple(p) for p in path],  # type: ignore[arg-type]
            total_coverage_m2=data.get("total_coverage_m2", 0.0),
            total_distance_m=data.get("total_distance_m", 0.0),
            sensor_ids=list(data.get("sensor_ids", [])),
            metadata=data.get("metadata", {}),
        )


class MissionIndex:
    """In-memory index of mission records.

    Provides lookup by mission ID and aggregation across missions.
    """

    def __init__(self) -> None:
        self._missions: Dict[str, MissionRecord] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, record: MissionRecord) -> None:
        """Register a mission record."""
        self._missions[record.mission_id] = record

    def create_mission(self, mission_id: Optional[str] = None, **kwargs: Any) -> MissionRecord:
        """Create and register a new mission record."""
        mid = mission_id or new_mission_id()
        record = MissionRecord(mission_id=mid, **kwargs)
        self.add(record)
        return record

    def remove(self, mission_id: str) -> Optional[MissionRecord]:
        return self._missions.pop(mission_id, None)

    def clear(self) -> None:
        self._missions.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, mission_id: str) -> Optional[MissionRecord]:
        return self._missions.get(mission_id)

    def __len__(self) -> int:
        return len(self._missions)

    def __contains__(self, mission_id: str) -> bool:
        return mission_id in self._missions

    def all(self) -> List[MissionRecord]:
        """Return all missions sorted by start time."""
        return sorted(self._missions.values(), key=lambda m: m.start_time_s)

    def query_time_range(self, t_min: float, t_max: float) -> List[MissionRecord]:
        """Return missions overlapping the time range ``[t_min, t_max]``."""
        return [
            m
            for m in self._missions.values()
            if m.start_time_s <= t_max and m.end_time_s >= t_min
        ]

    def missions_with_keyframe(self, keyframe_id: str) -> List[MissionRecord]:
        """Return missions containing a given keyframe ID."""
        return [
            m for m in self._missions.values() if keyframe_id in m.keyframe_ids
        ]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def total_coverage_m2(self) -> float:
        """Sum of coverage across all missions (may double-count overlap)."""
        return sum(m.total_coverage_m2 for m in self._missions.values())

    def total_distance_m(self) -> float:
        return sum(m.total_distance_m for m in self._missions.values())

    def total_keyframe_count(self) -> int:
        return sum(m.keyframe_count for m in self._missions.values())

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary of the index."""
        return {
            "mission_count": len(self._missions),
            "total_keyframes": self.total_keyframe_count(),
            "total_coverage_m2": self.total_coverage_m2(),
            "total_distance_m": self.total_distance_m(),
            "missions": [m.to_dict() for m in self.all()],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.all()]

    @classmethod
    def from_dicts(cls, records: Sequence[Dict[str, Any]]) -> MissionIndex:
        index = cls()
        for rec in records:
            index.add(MissionRecord.from_dict(rec))
        return index
