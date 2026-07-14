"""Tests for the versioned observation-source contract.

Covers the three things the seam must guarantee:

1. The default :class:`AnalyticObservationSource` is behavior-equivalent to calling
   ``build_observations`` directly (the refactor changed no observations).
2. An alternative source injected into ``run_simulation`` is actually used — the
   contract is a real runtime path, not a dangling interface.
3. Source and interface versions are exposed for provenance.
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import ObservationBatch, TruthState
from argusnet.sensing.observation_source import (
    OBSERVATION_SOURCE_CONTRACT_VERSION,
    AnalyticObservationSource,
    ObservationRequest,
    ObservationSource,
)
from argusnet.simulation.sim import (
    SimNode,
    SimulationConfig,
    build_default_scenario,
    build_observations,
    run_simulation,
    tracking_demo_options,
)
from argusnet.world.terrain import TerrainModel


def _flat_terrain() -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=0.0,
        slope_x_m_per_m=0.0,
        slope_y_m_per_m=0.0,
        wave_amplitude_m=0.0,
        ridge_amplitude_m=0.0,
        basin_depth_m=0.0,
    )


def _simple_node_and_truth() -> tuple[SimNode, TruthState]:
    node = SimNode(
        node_id="ground-a",
        is_mobile=False,
        bearing_std_rad=0.002,
        dropout_probability=0.0,
        max_range_m=500.0,
        trajectory=lambda _: (
            np.array([0.0, 0.0, 10.0], dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    truth = TruthState(
        target_id="asset-a",
        position=np.array([100.0, 0.0, 12.0], dtype=float),
        velocity=np.zeros(3, dtype=float),
        timestamp_s=0.0,
    )
    return node, truth


class AnalyticObservationSourceEquivalenceTest(unittest.TestCase):
    def test_observe_matches_direct_build_observations(self) -> None:
        node, truth = _simple_node_and_truth()
        terrain = _flat_terrain()

        direct = build_observations(
            rng=np.random.default_rng(4),
            nodes=[node],
            truths=[truth],
            timestamp_s=0.0,
            terrain=terrain,
        )
        source = AnalyticObservationSource(build_observations)
        via_source = source.observe(
            ObservationRequest(
                rng=np.random.default_rng(4),
                nodes=[node],
                truths=[truth],
                timestamp_s=0.0,
                terrain=terrain,
            )
        )

        self.assertIsInstance(via_source, ObservationBatch)
        self.assertEqual(direct.attempted_count, via_source.attempted_count)
        self.assertEqual(direct.rejection_counts, via_source.rejection_counts)
        self.assertEqual(direct.accepted_by_target, via_source.accepted_by_target)
        self.assertEqual(len(direct.observations), len(via_source.observations))
        for a, b in zip(direct.observations, via_source.observations):
            self.assertEqual(a.node_id, b.node_id)
            self.assertEqual(a.target_id, b.target_id)
            self.assertTrue(np.allclose(a.direction, b.direction))
            self.assertEqual(a.bearing_std_rad, b.bearing_std_rad)


class _EmptyObservationSource:
    """A conforming source that emits nothing — used to prove substitutability."""

    source_id = "test-empty"
    version = "0.1"

    def __init__(self) -> None:
        self.call_count = 0

    def observe(self, request: ObservationRequest) -> ObservationBatch:
        self.call_count += 1
        return ObservationBatch(
            observations=[],
            attempted_count=0,
            rejection_counts={},
            accepted_by_target={},
            rejected_by_target={},
            accepted_by_node_target={},
        )


class ObservationSourceInjectionTest(unittest.TestCase):
    def test_injected_source_is_used_by_run_simulation(self) -> None:
        source = _EmptyObservationSource()
        self.assertIsInstance(source, ObservationSource)  # structural conformance

        scenario = build_default_scenario(options=tracking_demo_options(), seed=7)
        result = run_simulation(
            scenario,
            SimulationConfig(steps=40, dt_s=0.25, seed=7),
            observation_source=source,
        )

        # The source was driven once per step, and with no observations no track
        # can confirm — proving the loop routes through the injected source.
        self.assertGreater(source.call_count, 0)
        self.assertEqual(max((f.metrics.active_track_count for f in result.frames), default=0), 0)
        self.assertEqual(result.summary["observation_source_id"], "test-empty")
        self.assertEqual(result.summary["observation_source_version"], "0.1")

    def test_default_source_reports_analytic_provenance(self) -> None:
        scenario = build_default_scenario(options=tracking_demo_options(), seed=7)
        result = run_simulation(scenario, SimulationConfig(steps=20, dt_s=0.25, seed=7))
        self.assertEqual(result.summary["observation_source_id"], "analytic")
        self.assertEqual(
            result.summary["observation_source_contract_version"],
            OBSERVATION_SOURCE_CONTRACT_VERSION,
        )


class ObservationSourceContractTest(unittest.TestCase):
    def test_analytic_source_exposes_versions(self) -> None:
        source = AnalyticObservationSource(build_observations)
        self.assertEqual(source.source_id, "analytic")
        self.assertTrue(source.version)
        self.assertTrue(OBSERVATION_SOURCE_CONTRACT_VERSION)

    def test_request_is_frozen(self) -> None:
        request = ObservationRequest(
            rng=np.random.default_rng(0),
            nodes=[],
            truths=[],
            timestamp_s=0.0,
            terrain=_flat_terrain(),
        )
        with self.assertRaises(Exception):
            request.timestamp_s = 1.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
