"""Tests for cooperative radial search: one grounded origin, disjoint angular
wedges, and team co-localization against the shared coverage map."""

from __future__ import annotations

import math
import unittest

import numpy as np

from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    compute_wedge_coverage,
    cooperative_wedge_waypoints,
    run_simulation,
    select_cooperative_wedge_cell,
)


class _Bounds:
    x_min_m = -100.0
    y_min_m = -100.0
    resolution_m = 20.0


class _Cov:
    def __init__(self, grid):
        self.count_grid = grid
        self.bounds = _Bounds()


class _NoClaims:
    def others(self, _drone_id):
        return set()

_BOUNDS = {"x_min_m": -2000.0, "x_max_m": 2000.0, "y_min_m": -2000.0, "y_max_m": 2000.0}


class WedgeWaypointTests(unittest.TestCase):
    def test_starts_at_origin_and_stays_in_bounds(self) -> None:
        origin = np.array([0.0, 0.0])
        wp = cooperative_wedge_waypoints(origin, _BOUNDS, 0, 4, lane_spacing_m=120.0)
        self.assertGreaterEqual(len(wp), 2)
        # First waypoint is the grounded origin.
        self.assertAlmostEqual(wp[0][0], 0.0, places=6)
        self.assertAlmostEqual(wp[0][1], 0.0, places=6)
        # Every waypoint is within the search area (plus the small clamp margin).
        margin = 40.0
        self.assertTrue(
            np.all(wp[:, 0] >= _BOUNDS["x_min_m"] - margin)
            and np.all(wp[:, 0] <= _BOUNDS["x_max_m"] + margin)
            and np.all(wp[:, 1] >= _BOUNDS["y_min_m"] - margin)
            and np.all(wp[:, 1] <= _BOUNDS["y_max_m"] + margin)
        )

    def test_wedges_are_angularly_disjoint(self) -> None:
        origin = np.array([0.0, 0.0])
        # Small lane spacing keeps radii modest so nothing clamps to the bounds,
        # leaving the pure angular assignment intact.
        w0 = cooperative_wedge_waypoints(origin, _BOUNDS, 0, 4, lane_spacing_m=200.0)
        w2 = cooperative_wedge_waypoints(origin, _BOUNDS, 2, 4, lane_spacing_m=200.0)

        def median_angle(wp: np.ndarray) -> float:
            pts = wp[1:]  # drop the origin point
            angles = np.arctan2(pts[:, 1], pts[:, 0]) % (2.0 * math.pi)
            return float(np.median(angles))

        # Wedge 0 spans [0, pi/2]; wedge 2 spans [pi, 3pi/2].
        self.assertTrue(0.0 <= median_angle(w0) <= math.pi / 2 + 0.2)
        self.assertTrue(math.pi - 0.2 <= median_angle(w2) <= 1.5 * math.pi + 0.2)


class AdaptiveWedgeTests(unittest.TestCase):
    def test_wedge_coverage_reflects_covered_quadrant(self) -> None:
        # 10x10 grid at 20 m, origin (0,0). Cover only the +x/+y quadrant.
        grid = np.zeros((10, 10), dtype=int)
        grid[5:, 5:] = 1  # cell centers here are x>0, y>0 → wedge 0 of 4
        cov = compute_wedge_coverage(_Cov(grid), np.array([0.0, 0.0]), 4)
        self.assertGreater(cov[0], 0.5)  # +x+y wedge well covered
        self.assertAlmostEqual(cov[2], 0.0, places=6)  # opposite wedge untouched

    def test_selects_home_wedge_then_rebalances(self) -> None:
        grid = np.zeros((10, 10), dtype=int)  # nothing covered
        cov = np.zeros(4)
        # Home wedge 0 (angles [0, pi/2] → +x/+y): drone at origin should pick a
        # cell in that quadrant.
        cell = select_cooperative_wedge_cell(
            _Cov(grid), np.array([0.0, 0.0]), _NoClaims(), "d", np.array([0.0, 0.0]), 0, 4, cov
        )
        cx = _Bounds.x_min_m + (cell[0] + 0.5) * _Bounds.resolution_m
        cy = _Bounds.y_min_m + (cell[1] + 0.5) * _Bounds.resolution_m
        self.assertGreater(cx, 0.0)
        self.assertGreater(cy, 0.0)

        # Home wedge "done" → reallocate toward the most under-covered wedge (2).
        cov_done = np.array([1.0, 0.9, 0.0, 0.9])
        cell2 = select_cooperative_wedge_cell(
            _Cov(grid), np.array([0.0, 0.0]), _NoClaims(), "d", np.array([0.0, 0.0]), 0, 4, cov_done
        )
        cx2 = _Bounds.x_min_m + (cell2[0] + 0.5) * _Bounds.resolution_m
        cy2 = _Bounds.y_min_m + (cell2[1] + 0.5) * _Bounds.resolution_m
        self.assertLess(cx2, 0.0)  # wedge 2 is the -x/-y quadrant
        self.assertLess(cy2, 0.0)


class CooperativeSearchRunTests(unittest.TestCase):
    def test_run_grounds_origin_and_fuses_team_localization(self) -> None:
        options = ScenarioOptions(
            map_preset="medium",
            drone_count=6,
            mission_mode="scan_map_inspect",
            cooperative_search=True,
            coverage_resolution_m=50.0,
        )
        scenario = build_default_scenario(options=options, seed=7)
        result = run_simulation(
            scenario=scenario,
            simulation_config=SimulationConfig.from_duration(90.0, dt_s=0.25, seed=7),
            tracker_config=None,
        )

        self.assertTrue(result.summary.get("cooperative_search"))
        origin = result.summary.get("search_origin_m")
        self.assertIsInstance(origin, list)
        self.assertEqual(len(origin), 2)

        # All drones launch clustered around the single grounded origin.
        first = result.frames[0]
        drone_xy = [
            (n.position[0], n.position[1])
            for n in first.nodes
            if "drone" in n.node_id
        ]
        self.assertGreater(len(drone_xy), 1)
        xy = np.array(drone_xy)
        spread = float(np.linalg.norm(xy - xy.mean(axis=0), axis=1).max())
        self.assertLess(spread, 120.0)

        # Some coverage accrued (the search actually ran).
        last_scan = next(
            (f.scan_mission_state for f in reversed(result.frames) if f.scan_mission_state),
            None,
        )
        self.assertIsNotNone(last_scan)
        self.assertGreater(last_scan.scan_coverage_fraction, 0.1)

    def test_adaptive_search_runs_and_records(self) -> None:
        options = ScenarioOptions(
            map_preset="medium",
            drone_count=6,
            mission_mode="scan_map_inspect",
            cooperative_search=True,
            adaptive_search=True,
            coverage_resolution_m=50.0,
        )
        scenario = build_default_scenario(options=options, seed=7)
        result = run_simulation(
            scenario=scenario,
            simulation_config=SimulationConfig.from_duration(90.0, dt_s=0.25, seed=7),
            tracker_config=None,
        )
        self.assertTrue(result.summary.get("cooperative_search"))
        self.assertTrue(result.summary.get("adaptive_search"))
        last_scan = next(
            (f.scan_mission_state for f in reversed(result.frames) if f.scan_mission_state),
            None,
        )
        self.assertIsNotNone(last_scan)
        # Adaptive redirection should still cover meaningful ground.
        self.assertGreater(last_scan.scan_coverage_fraction, 0.3)


if __name__ == "__main__":
    unittest.main()
