"""Backward-compatibility shim — imports from argusnet.core.config."""
from argusnet.core.config import *  # noqa: F401, F403
from argusnet.core.config import (
    DEFAULT_MAP_PRESET_SCALES,
    DEFAULT_PLATFORM_PRESETS,
    DynamicsConfig,
    GroundStationLayoutConfig,
    PlatformPresetProfile,
    SensorConfig,
    SimulationConstants,
    TargetTrajectoryConfig,
)
