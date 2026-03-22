"""Backward-compatibility shim — imports from argusnet.simulation.behaviors."""
from argusnet.simulation.behaviors import *  # noqa: F401, F403
from argusnet.simulation.behaviors import (
    BEHAVIOR_PRESETS,
    FlightEnvelope,
    build_target_trajectory,
)
