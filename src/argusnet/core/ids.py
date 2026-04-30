"""ID generation utilities for ArgusNet."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

__all__ = [
    "new_track_id",
    "new_node_id",
    "new_mission_id",
    "new_keyframe_id",
    "new_tile_id",
    "make_id",
    "IdTable",
]


def make_id(prefix: str = "") -> str:
    """Return a new UUID4-based ID, optionally prefixed."""
    uid = str(uuid.uuid4())
    return f"{prefix}{uid}" if prefix else uid


def new_track_id() -> str:
    return make_id("trk-")


def new_node_id() -> str:
    return make_id("node-")


def new_mission_id() -> str:
    return make_id("msn-")


def new_keyframe_id() -> str:
    return make_id("kf-")


def new_tile_id(zoom: int, x: int, y: int) -> str:
    """Deterministic tile ID in slippy-map notation."""
    return f"tile-{zoom}-{x}-{y}"


@dataclass
class IdTable(Generic[T]):
    """Dense id/index mapping for hot loops while preserving stable external IDs."""

    id_to_index: dict[T, int] = field(default_factory=dict)
    index_to_id: list[T] = field(default_factory=list)

    def get_or_insert(self, item_id: T) -> int:
        existing = self.id_to_index.get(item_id)
        if existing is not None:
            return existing
        index = len(self.index_to_id)
        self.id_to_index[item_id] = index
        self.index_to_id.append(item_id)
        return index

    def get_index(self, item_id: T) -> int | None:
        return self.id_to_index.get(item_id)

    def get_id(self, index: int) -> T:
        return self.index_to_id[index]

    def __len__(self) -> int:
        return len(self.index_to_id)
