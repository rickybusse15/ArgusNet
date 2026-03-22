"""ID generation utilities for ArgusNet."""

from __future__ import annotations

import uuid

__all__ = [
    "new_track_id",
    "new_node_id",
    "new_mission_id",
    "new_keyframe_id",
    "new_tile_id",
    "make_id",
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
