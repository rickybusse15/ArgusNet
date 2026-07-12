"""Runtime belief-query interface: contract, summary, and the viewer terrain
reconstruction as its first consumer (truth-isolated).

Milestone: "runtime belief-query interface for mapping (reconstruction is first
consumer)" — the seam that lets belief consumers read observed height/coverage
without touching simulation truth.
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.mapping import (
    BELIEF_QUERY_CONTRACT_VERSION,
    BeliefQuery,
    BeliefSummary,
    WorldBeliefQuery,
)
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid
from argusnet.mapping.world_map import WorldMap
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


def _linear_terrain(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return 0.1 * pts[:, 0] + 0.2 * pts[:, 1] + 5.0


def _scan_options(**overrides) -> ScenarioOptions:
    base = dict(
        map_preset="small",
        terrain_preset="alpine",
        drone_count=3,
        mission_mode="scan_map_inspect",
        scan_coverage_threshold=0.99,
        coverage_resolution_m=30.0,
    )
    base.update(overrides)
    return ScenarioOptions(**base)


def _run_scan(options: ScenarioOptions, *, duration_s: float = 24.0, **kwargs):
    scenario = build_default_scenario(options=options, seed=7)
    result = run_simulation(
        scenario=scenario,
        simulation_config=SimulationConfig.from_duration(duration_s, dt_s=0.5, seed=7),
        tracker_config=None,
        **kwargs,
    )
    return scenario, result


class BeliefContractTests(unittest.TestCase):
    def test_worldbeliefquery_satisfies_protocol_and_exposes_provenance(self) -> None:
        wm = WorldMap(GridBounds(0.0, 100.0, 0.0, 100.0, 10.0))
        query = WorldBeliefQuery(wm, source_id="grid", version="1.0")
        self.assertIsInstance(query, BeliefQuery)
        self.assertEqual(query.source_id, "grid")
        self.assertEqual(query.version, "1.0")
        self.assertEqual(BELIEF_QUERY_CONTRACT_VERSION, "1.0")

    def test_requires_a_map(self) -> None:
        with self.assertRaises(ValueError):
            WorldBeliefQuery()


class ObservedHeightLayerTests(unittest.TestCase):
    """The observed-height belief layer is dense over covered cells and matches
    the observed terrain — so the reconstruction never falls back to datum."""

    def setUp(self) -> None:
        self.bounds = GridBounds(0.0, 300.0, 0.0, 300.0, 30.0)
        self.wm = WorldMap(self.bounds)
        rng = np.random.default_rng(0)
        for _ in range(30):
            pos = rng.uniform(20.0, 280.0, size=2)
            self.wm.add_scan_observation(
                drone_position=np.array([pos[0], pos[1], 80.0]),
                terrain_height=float(_linear_terrain(pos.reshape(1, 2))[0]),
                footprint_radius_m=60.0,
                terrain_height_at_many=_linear_terrain,
            )

    def test_observed_height_defined_exactly_on_covered_cells(self) -> None:
        covered = self.wm.coverage_map.count_grid > 0
        finite = np.isfinite(self.wm.observed_height_grid)
        self.assertGreater(int(covered.sum()), 0)
        np.testing.assert_array_equal(finite, covered)

    def test_observed_height_matches_the_terrain_sampler(self) -> None:
        obs = self.wm.observed_height_grid
        ii, jj = np.where(np.isfinite(obs))
        centers = np.array(
            [self.bounds.ij_to_xy(int(i), int(j)) for i, j in zip(ii, jj, strict=False)]
        )
        expected = _linear_terrain(centers)
        np.testing.assert_allclose(obs[ii, jj], expected, rtol=0, atol=1e-9)

    def test_height_estimate_grid_defined_on_all_covered_cells(self) -> None:
        # The grid unions the observed-height layer with the legacy nadir mean
        # grid, so it is defined wherever either source has data — in particular
        # on every covered cell, so the reconstruction never hits a NaN there.
        query = WorldBeliefQuery(self.wm)
        grid = query.height_estimate_grid()
        covered = self.wm.coverage_map.count_grid > 0
        self.assertTrue(np.all(np.isfinite(grid[covered])))
        # Where covered, the estimate is the observed-height layer verbatim.
        obs = self.wm.observed_height_grid
        np.testing.assert_array_equal(grid[covered], obs[covered])

    def test_scalar_path_falls_back_to_nadir_mean_grid(self) -> None:
        # Without a per-cell sampler the observed layer stays empty; the query
        # must still return a height from the legacy nadir mean grid.
        wm = WorldMap(self.bounds)
        wm.add_scan_observation(np.array([150.0, 150.0, 80.0]), 42.0, 60.0)
        query = WorldBeliefQuery(wm)
        self.assertIsNotNone(query.height_estimate_at(150.0, 150.0))


class BeliefSummaryTests(unittest.TestCase):
    def test_summary_counts_and_confidence(self) -> None:
        bounds = GridBounds(0.0, 100.0, 0.0, 100.0, 10.0)  # 10x10 = 100 cells
        wm = WorldMap(bounds)
        for _ in range(6):  # revisit so confidence saturates
            wm.add_scan_observation(
                np.array([25.0, 25.0, 80.0]), 4.0, 12.0, terrain_height_at_many=_linear_terrain
            )
        query = WorldBeliefQuery(wm)
        s = query.belief_summary()
        self.assertIsInstance(s, BeliefSummary)
        self.assertEqual(s.total_cells, 100)
        self.assertGreater(s.observed_cells, 0)
        self.assertEqual(s.unknown_cells, s.total_cells - s.observed_cells)
        self.assertEqual(s.unsafe_cells, 0)  # no occupancy grid
        self.assertGreater(s.frontier_cells, 0)  # observed region has a boundary
        self.assertGreater(s.mean_belief_confidence, 0.0)
        self.assertIsNotNone(s.mean_height_uncertainty_m)

    def test_summary_marks_obstacles_unsafe(self) -> None:
        bounds = GridBounds(0.0, 100.0, 0.0, 100.0, 10.0)
        wm = WorldMap(bounds)
        wm.add_scan_observation(
            np.array([25.0, 25.0, 80.0]), 4.0, 20.0, terrain_height_at_many=_linear_terrain
        )
        occ = OccupancyGrid(bounds)
        occ.mark_occupied(25.0, 25.0, height_m=8.0)
        query = WorldBeliefQuery(wm, occupancy_grid=occ)
        self.assertGreaterEqual(query.belief_summary().unsafe_cells, 1)


class ReconstructionConsumerTests(unittest.TestCase):
    """The viewer reconstruction (newly_scanned_cells) is the belief query's
    first consumer: it reads believed heights, never simulation truth."""

    def test_reconstruction_reads_belief_query_not_truth(self) -> None:
        # Inject a belief query whose height grid is a sentinel constant. If the
        # reconstruction still consulted truth, cell heights would not be 777.0.
        sentinel = 777.0

        class SentinelBelief:
            source_id = "sentinel"
            version = "9.9"

            def __init__(self, world_map: WorldMap) -> None:
                self._shape = (world_map.bounds.nx, world_map.bounds.ny)

            def height_estimate_grid(self) -> np.ndarray:
                return np.full(self._shape, sentinel, dtype=float)

            def belief_summary(self) -> BeliefSummary:
                return BeliefSummary(0, 0, 0, 0, 0, 0.0, 0.0, None)

        _scenario, result = _run_scan(
            _scan_options(), belief_query_factory=lambda wm: SentinelBelief(wm)
        )
        heights = [
            z
            for frame in result.frames
            if frame.scan_mission_state is not None
            for _x, _y, z in frame.scan_mission_state.newly_scanned_cells
        ]
        self.assertTrue(heights, "expected the scan to reconstruct some cells")
        self.assertTrue(all(z == sentinel for z in heights))

    def test_reconstruction_heights_match_observed_terrain(self) -> None:
        scenario, result = _run_scan(_scan_options())
        terrain = scenario.terrain
        checked = 0
        for frame in result.frames:
            if frame.scan_mission_state is None:
                continue
            for x, y, z in frame.scan_mission_state.newly_scanned_cells:
                # Believed height equals observed terrain at the cell centre.
                self.assertAlmostEqual(z, round(float(terrain.height_at(x, y)), 1), delta=0.2)
                checked += 1
        self.assertGreater(checked, 0)

    def test_mapping_state_belief_fields_are_populated(self) -> None:
        _scenario, result = _run_scan(_scan_options())
        populated = 0
        for frame in result.frames:
            ms = frame.mapping_state
            if ms is None or ms.observed_cells is None:
                continue
            populated += 1
            self.assertEqual(ms.observed_cells, ms.covered_cells)
            self.assertEqual(ms.unknown_cells, ms.total_cells - ms.observed_cells)
            self.assertIsNotNone(ms.mean_belief_confidence)
        self.assertGreater(populated, 0)

    def test_reconstruction_is_deterministic(self) -> None:
        _s1, r1 = _run_scan(_scan_options())
        _s2, r2 = _run_scan(_scan_options())
        cells1 = [
            tuple(f.scan_mission_state.newly_scanned_cells)
            for f in r1.frames
            if f.scan_mission_state is not None
        ]
        cells2 = [
            tuple(f.scan_mission_state.newly_scanned_cells)
            for f in r2.frames
            if f.scan_mission_state is not None
        ]
        self.assertEqual(cells1, cells2)


if __name__ == "__main__":
    unittest.main()
