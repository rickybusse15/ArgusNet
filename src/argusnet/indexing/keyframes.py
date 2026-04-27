"""Keyframe store for ArgusNet.

Manages a collection of observation keyframes selected by coverage delta
or novelty score.  Supports spatial and temporal range queries for
downstream retrieval, mapping, and evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from argusnet.core.ids import new_keyframe_id
from argusnet.core.types import Vector3

try:
    from scipy.spatial import KDTree as _KDTree

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

__all__ = [
    "Keyframe",
    "KeyframeStore",
    "select_keyframes_by_coverage",
]


@dataclass(frozen=True)
class Keyframe:
    """A single observation keyframe."""

    keyframe_id: str
    """Unique ID (``kf-<uuid>``)."""

    timestamp_s: float
    """Simulation / capture time (seconds)."""

    position: Vector3
    """Camera / sensor position in world frame (metres)."""

    orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Orientation as (roll, pitch, yaw) in radians."""

    coverage_delta: float = 0.0
    """Incremental coverage added by this keyframe (m²)."""

    novelty_score: float = 0.0
    """Novelty metric — higher means more new information."""

    sensor_id: str = ""
    """Which sensor captured this keyframe."""

    mission_id: str = ""
    """Mission this keyframe belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata (feature count, image path, etc.)."""

    descriptor: np.ndarray | None = field(default=None, compare=False, hash=False)
    """Optional appearance descriptor vector for loop-closure verification.

    Typically a unit-normalised embedding from a visual place-recognition
    model.  When present, :class:`~argusnet.localization.loop_closure.LoopClosureDetector`
    uses cosine similarity against candidate keyframes as a secondary gate.
    """


class KeyframeStore:
    """In-memory keyframe collection with spatial and temporal queries.

    For persistent storage, serialize via :meth:`to_dicts` /
    :meth:`from_dicts`.
    """

    def __init__(self) -> None:
        self._keyframes: dict[str, Keyframe] = {}
        self._by_time: list[Keyframe] = []  # kept sorted by timestamp
        self._kdtree: Any | None = None
        self._kdtree_coords: np.ndarray | None = None
        self._kdtree_ids: list[str] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, kf: Keyframe) -> None:
        """Insert a keyframe into the store."""
        self._keyframes[kf.keyframe_id] = kf
        self._by_time.append(kf)
        self._by_time.sort(key=lambda k: k.timestamp_s)
        self._invalidate_kdtree()

    def remove(self, keyframe_id: str) -> Keyframe | None:
        """Remove and return a keyframe by ID, or ``None``."""
        kf = self._keyframes.pop(keyframe_id, None)
        if kf is not None:
            self._by_time = [k for k in self._by_time if k.keyframe_id != keyframe_id]
        self._invalidate_kdtree()
        return kf

    def clear(self) -> None:
        self._keyframes.clear()
        self._by_time.clear()
        self._invalidate_kdtree()

    # ------------------------------------------------------------------
    # KDTree helpers
    # ------------------------------------------------------------------

    def _invalidate_kdtree(self) -> None:
        self._kdtree = None

    def _ensure_kdtree(self) -> None:
        if self._kdtree is not None:
            return
        if not _HAS_SCIPY or not self._by_time:
            return
        coords = np.array([list(kf.position) for kf in self._by_time], dtype=float)
        self._kdtree_coords = coords
        self._kdtree_ids = [kf.keyframe_id for kf in self._by_time]
        self._kdtree = _KDTree(coords)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, keyframe_id: str) -> Keyframe | None:
        return self._keyframes.get(keyframe_id)

    def __len__(self) -> int:
        return len(self._keyframes)

    def __contains__(self, keyframe_id: str) -> bool:
        return keyframe_id in self._keyframes

    def all(self) -> list[Keyframe]:
        """Return all keyframes sorted by timestamp."""
        return list(self._by_time)

    def query_temporal(self, t_min: float, t_max: float) -> list[Keyframe]:
        """Return keyframes in ``[t_min, t_max]``."""
        return [kf for kf in self._by_time if t_min <= kf.timestamp_s <= t_max]

    def query_spatial(self, center: Vector3, radius_m: float) -> list[Keyframe]:
        """Return keyframes within *radius_m* of *center* (Euclidean).

        Uses a KDTree for O(log N) queries when scipy is available;
        falls back to an O(n) linear scan otherwise.
        """
        center_arr = np.asarray(center, dtype=float)
        self._ensure_kdtree()
        if _HAS_SCIPY and self._kdtree is not None:
            indices = self._kdtree.query_ball_point(center_arr, radius_m)
            return [
                self._keyframes[self._kdtree_ids[i]]
                for i in indices
                if self._kdtree_ids[i] in self._keyframes
            ]
        # O(n) fallback
        return [
            kf
            for kf in self._by_time
            if float(np.linalg.norm(np.asarray(kf.position, dtype=float) - center_arr)) <= radius_m
        ]

    def query_spatial_box(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> list[Keyframe]:
        """Return keyframes within a 2-D bounding box.

        Uses a KDTree for an initial candidate filter when scipy is available;
        falls back to an O(n) linear scan otherwise.
        """
        self._ensure_kdtree()
        if _HAS_SCIPY and self._kdtree is not None:
            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
            # Use the midpoint in z of the stored data, defaulting to 0.
            cz = (
                float(np.mean(self._kdtree_coords[:, 2]))
                if self._kdtree_coords is not None and len(self._kdtree_coords) > 0
                else 0.0
            )
            half_diag = float(np.hypot((x_max - x_min) * 0.5, (y_max - y_min) * 0.5))
            center_arr = np.array([cx, cy, cz], dtype=float)
            indices = self._kdtree.query_ball_point(center_arr, half_diag)
            result = []
            for i in indices:
                kid = self._kdtree_ids[i]
                kf = self._keyframes.get(kid)
                if (
                    kf is not None
                    and x_min <= kf.position[0] <= x_max
                    and y_min <= kf.position[1] <= y_max
                ):
                    result.append(kf)
            return result
        # O(n) fallback
        return [
            kf
            for kf in self._by_time
            if x_min <= kf.position[0] <= x_max and y_min <= kf.position[1] <= y_max
        ]

    def query_mission(self, mission_id: str) -> list[Keyframe]:
        """Return all keyframes for a given mission."""
        return [kf for kf in self._by_time if kf.mission_id == mission_id]

    def top_by_novelty(self, n: int = 10) -> list[Keyframe]:
        """Return the *n* keyframes with highest novelty score."""
        return sorted(self._by_time, key=lambda kf: kf.novelty_score, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all keyframes to a list of JSON-friendly dicts."""
        return [
            {
                "keyframe_id": kf.keyframe_id,
                "timestamp_s": kf.timestamp_s,
                "position": list(kf.position),
                "orientation_rad": list(kf.orientation_rad),
                "coverage_delta": kf.coverage_delta,
                "novelty_score": kf.novelty_score,
                "sensor_id": kf.sensor_id,
                "mission_id": kf.mission_id,
                "metadata": kf.metadata,
                "descriptor": kf.descriptor.tolist() if kf.descriptor is not None else None,
            }
            for kf in self._by_time
        ]

    @classmethod
    def from_dicts(cls, records: Sequence[dict[str, Any]]) -> KeyframeStore:
        """Deserialize from a list of dicts."""
        store = cls()
        for rec in records:
            pos = rec.get("position", [0.0, 0.0, 0.0])
            orient = rec.get("orientation_rad", [0.0, 0.0, 0.0])
            raw_desc = rec.get("descriptor")
            descriptor = np.asarray(raw_desc, dtype=float) if raw_desc is not None else None
            kf = Keyframe(
                keyframe_id=rec.get("keyframe_id", new_keyframe_id()),
                timestamp_s=rec.get("timestamp_s", 0.0),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                orientation_rad=(float(orient[0]), float(orient[1]), float(orient[2])),
                coverage_delta=rec.get("coverage_delta", 0.0),
                novelty_score=rec.get("novelty_score", 0.0),
                sensor_id=rec.get("sensor_id", ""),
                mission_id=rec.get("mission_id", ""),
                metadata=rec.get("metadata", {}),
                descriptor=descriptor,
            )
            store.add(kf)
        return store


def select_keyframes_by_coverage(
    candidates: Sequence[Keyframe],
    min_coverage_delta: float = 1.0,
    max_keyframes: int = 1000,
) -> list[Keyframe]:
    """Greedy keyframe selection based on coverage delta.

    Iterates candidates in time order, accepting those whose
    ``coverage_delta`` exceeds *min_coverage_delta*, up to
    *max_keyframes*.
    """
    sorted_candidates = sorted(candidates, key=lambda kf: kf.timestamp_s)
    selected: list[Keyframe] = []
    for kf in sorted_candidates:
        if len(selected) >= max_keyframes:
            break
        if kf.coverage_delta >= min_coverage_delta:
            selected.append(kf)
    return selected
