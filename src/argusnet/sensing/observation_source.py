"""Typed boundary for producing observations consumed by the runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from argusnet.core.config import SimulationConstants
from argusnet.core.types import BearingObservation, ObservationRejection, TruthState
from argusnet.sensing.models.noise import SensorModel
from argusnet.world.environment import EnvironmentModel, SensorVisibilityModel
from argusnet.world.weather import WeatherModel


@dataclass(frozen=True)
class ObservationBatch:
    """Observations and generation diagnostics for one runtime frame."""

    observations: list[BearingObservation]
    attempted_count: int
    rejection_counts: dict[str, int]
    accepted_by_target: dict[str, int]
    rejected_by_target: dict[str, int]
    accepted_by_node_target: dict[tuple[str, str], int]
    generation_rejections: list[ObservationRejection] = field(default_factory=list)


@dataclass(frozen=True)
class ObservationRequest:
    """Immutable inputs supplied to an observation source for a frame."""

    rng: np.random.Generator
    nodes: Sequence[Any]
    truths: Sequence[TruthState]
    timestamp_s: float
    terrain: Any
    occluding_objects: Sequence[Any] = ()
    min_elevation_deg: float = 0.5
    environment: EnvironmentModel | None = None
    sensor_profile: SensorVisibilityModel | None = None
    weather: WeatherModel | None = None
    sensor_models: Mapping[str, SensorModel] | None = None
    constants: SimulationConstants = field(default_factory=SimulationConstants.default)
    seed: int | None = None


@runtime_checkable
class ObservationSource(Protocol):
    """Version-one runtime contract for frame-level observation production."""

    contract_version: str

    def generate(self, request: ObservationRequest) -> ObservationBatch:
        """Produce observations and diagnostics for one frame."""


@dataclass(frozen=True)
class FunctionObservationSource:
    """Adapt an existing keyword-based generator to ``ObservationSource``."""

    generator: Callable[..., ObservationBatch]
    contract_version: str = "argusnet.observation-source/v1"

    def generate(self, request: ObservationRequest) -> ObservationBatch:
        return self.generator(
            rng=request.rng,
            nodes=request.nodes,
            truths=request.truths,
            timestamp_s=request.timestamp_s,
            terrain=request.terrain,
            occluding_objects=request.occluding_objects,
            min_elevation_deg=request.min_elevation_deg,
            environment=request.environment,
            sensor_profile=request.sensor_profile,
            weather=request.weather,
            sensor_models=request.sensor_models,
            constants=request.constants,
            seed=request.seed,
        )


__all__ = [
    "FunctionObservationSource",
    "ObservationBatch",
    "ObservationRequest",
    "ObservationSource",
]
