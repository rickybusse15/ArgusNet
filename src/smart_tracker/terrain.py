"""Backward-compatibility shim — imports from argusnet.world.terrain."""
from argusnet.world.terrain import *  # noqa: F401, F403
from argusnet.world.terrain import (
    KNOWN_TERRAIN_PRESETS,
    MountainRange,
    NoiseLayer,
    OccludingObject,
    Plateau,
    RidgeLine,
    TerrainFeature,
    TerrainModel,
    Valley,
    terrain_model_from_preset,
    xy_bounds,
    alpine_terrain,
)
