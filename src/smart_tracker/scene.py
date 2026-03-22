"""Backward-compatibility shim — imports from argusnet.world.scene_loader."""
from argusnet.world.scene_loader import *  # noqa: F401, F403
from argusnet.world.scene_loader import (
    SCENE_FORMAT_VERSION,
    build_scene_from_gis,
    build_scene_from_replay,
    build_scene_package,
    load_scene_manifest,
    validate_scene_manifest,
    _build_obstacle_meshes,
)
