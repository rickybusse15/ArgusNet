"""Backward-compatibility shim — imports from argusnet.world.environment."""
from argusnet.world.environment import *  # noqa: F401, F403
from argusnet.world.environment import (
    Bounds2D,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverClass,
    LandCoverLayer,
    SeasonalVariation,
    SensorVisibilityModel,
    TerrainLayer,
    TerrainTile,
    VisibilityResult,
    load_environment_bundle,
    write_environment_bundle,
    compute_effective_noise,
    compute_weather_factor,
    free_space_path_loss,
    identify_dominant_loss,
    DetectionResult,
    EnvironmentQuery,
)
