"""Versioned observation-source contract.

Observation synthesis (turning ground-truth object/platform state into noisy
sensor observations) is the one place in the platform that is *allowed* to read
simulation ground truth. Formalizing it behind a small, versioned interface lets
alternative backends — a higher-fidelity geometry/physics model, a recorded-data
replayer, or an approved live adapter — be swapped in without the mission loop
reaching into truth directly or knowing how observations are produced.

The default backend is :class:`AnalyticObservationSource`, which delegates to the
existing deterministic analytic generator (``build_observations`` in
``argusnet.simulation.sim``). It is injected as a plain callable so this module
does not import the simulation loop (preserving the one-way dependency direction
core/types <- subsystems <- simulation).

Contract:

* Input is an :class:`ObservationRequest` describing one simulation step.
* Output is an :class:`ObservationBatch` (``argusnet.core.types``): the accepted
  observations plus per-step generation statistics.
* Every source exposes a ``source_id`` and a ``version`` for provenance/lineage.
* ``OBSERVATION_SOURCE_CONTRACT_VERSION`` versions the interface shape itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from argusnet.core.types import ObservationBatch, TruthState

# Version of the ObservationSource interface shape (request/response contract),
# distinct from an individual source's ``version``. Bump on breaking changes to
# ObservationRequest / ObservationBatch semantics.
OBSERVATION_SOURCE_CONTRACT_VERSION = "1.0"

__all__ = [
    "OBSERVATION_SOURCE_CONTRACT_VERSION",
    "ObservationRequest",
    "ObservationSource",
    "AnalyticObservationSource",
]


@dataclass(frozen=True)
class ObservationRequest:
    """Typed inputs for synthesizing observations for a single step.

    Fields that belong to the simulation/world layers (``nodes``, ``terrain``,
    ``environment``, ``weather``) are typed structurally (``object``) so this
    contract does not depend on the simulation loop. Core-owned inputs (``truths``,
    ``rng``, ``timestamp_s``) keep their precise types.
    """

    rng: np.random.Generator
    nodes: Sequence[object]
    truths: Sequence[TruthState]
    timestamp_s: float
    terrain: object
    occluding_objects: Sequence[object] = ()
    environment: object | None = None
    weather: object | None = None
    sensor_models: Mapping[str, object] | None = None
    constants: object | None = None
    seed: int | None = None


@runtime_checkable
class ObservationSource(Protocol):
    """A versioned producer of per-step :class:`ObservationBatch` outputs."""

    source_id: str
    version: str

    def observe(self, request: ObservationRequest) -> ObservationBatch:
        """Synthesize observations for the step described by *request*."""
        ...


class AnalyticObservationSource:
    """Default deterministic backend delegating to an analytic generator.

    The generator is injected (rather than imported) to keep this module free of a
    simulation-loop dependency. ``argusnet.simulation.sim`` constructs the default
    instance with its ``build_observations`` function.
    """

    def __init__(
        self,
        generate_fn: Callable[..., ObservationBatch],
        *,
        source_id: str = "analytic",
        version: str = "1.0",
    ) -> None:
        self._generate_fn = generate_fn
        self.source_id = source_id
        self.version = version

    def observe(self, request: ObservationRequest) -> ObservationBatch:
        kwargs = dict(
            rng=request.rng,
            nodes=request.nodes,
            truths=request.truths,
            timestamp_s=request.timestamp_s,
            terrain=request.terrain,
            occluding_objects=request.occluding_objects,
            environment=request.environment,
            weather=request.weather,
            sensor_models=request.sensor_models,
            seed=request.seed,
        )
        # Only forward constants when supplied so the generator's own default
        # applies otherwise (it requires a concrete SimulationConstants).
        if request.constants is not None:
            kwargs["constants"] = request.constants
        return self._generate_fn(**kwargs)
