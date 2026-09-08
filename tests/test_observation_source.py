from __future__ import annotations

import numpy as np

from argusnet.core.types import TruthState
from argusnet.sensing.observation_source import (
    FunctionObservationSource,
    ObservationBatch,
    ObservationRequest,
    ObservationSource,
)
from argusnet.world.terrain import TerrainModel


def test_function_observation_source_forwards_immutable_request() -> None:
    captured: dict[str, object] = {}
    expected = ObservationBatch([], 0, {}, {}, {}, {})

    def generator(**kwargs: object) -> ObservationBatch:
        captured.update(kwargs)
        return expected

    truth = TruthState("target-1", np.ones(3), np.zeros(3), 2.5)
    request = ObservationRequest(
        rng=np.random.default_rng(7),
        nodes=(),
        truths=(truth,),
        timestamp_s=2.5,
        terrain=TerrainModel(),
        seed=7,
    )
    source = FunctionObservationSource(generator)

    assert isinstance(source, ObservationSource)
    assert source.generate(request) is expected
    assert captured["truths"] == (truth,)
    assert captured["timestamp_s"] == 2.5
    assert captured["seed"] == 7
    assert source.contract_version == "argusnet.observation-source/v1"
