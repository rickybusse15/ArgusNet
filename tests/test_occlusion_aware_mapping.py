"""Tests for opt-in obstacle-aware scan mapping and mission redirects."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.planning.planner_base import PathPlanner2D, PlannerConfig, _point_on_polygon_boundary
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)
from argusnet.world.environment import Bounds2D, SensorVisibilityModel
from argusnet.world.obstacles import _point_in_polygon


def _urban_options(*, occlusion_aware_mapping: bool) -> ScenarioOptions:
    return ScenarioOptions(
        map_preset="small",
        terrain_preset="urban_flat",
        drone_count=2,
        mission_mode="scan_map_inspect",
        scan_coverage_threshold=0.99,
        coverage_resolution_m=20.0,
        occlusion_aware_mapping=occlusion_aware_mapping,
    )


def _shadow_cell_count(terrain_preset: str) -> int:
    """Cells stamped by legacy (LOS-blind) coverage but withheld once occlusion
    gating is on — i.e. the occlusion "shadow" for this terrain."""

    def _scanned(occlusion_aware_mapping: bool) -> set[tuple[float, float]]:
        scenario = build_default_scenario(
            options=ScenarioOptions(
                map_preset="small",
                terrain_preset=terrain_preset,
                drone_count=3,
                mission_mode="scan_map_inspect",
                scan_coverage_threshold=0.99,
                coverage_resolution_m=20.0,
                occlusion_aware_mapping=occlusion_aware_mapping,
            ),
            seed=7,
        )
        result = run_simulation(
            scenario=scenario,
            simulation_config=SimulationConfig.from_duration(30.0, dt_s=0.5, seed=7),
            tracker_config=None,
        )
        return _newly_scanned_xy(result)

    return len(_scanned(False) - _scanned(True))


def _run_urban(*, occlusion_aware_mapping: bool):
    scenario = build_default_scenario(
        options=_urban_options(occlusion_aware_mapping=occlusion_aware_mapping),
        seed=7,
    )
    result = run_simulation(
        scenario=scenario,
        simulation_config=SimulationConfig.from_duration(24.0, dt_s=0.5, seed=7),
        tracker_config=None,
    )
    return scenario, result


def _newly_scanned_xy(result) -> set[tuple[float, float]]:
    cells: set[tuple[float, float]] = set()
    for frame in result.frames:
        state = frame.scan_mission_state
        if state is None:
            continue
        cells.update((float(x), float(y)) for x, y, _z in state.newly_scanned_cells)
    return cells


def _hard_blockers(scenario):
    bounds = scenario.environment.bounds_xy_m
    return [
        obstacle
        for obstacle in scenario.environment.obstacles.query_obstacles(bounds)
        if obstacle.blocker_type in {"building", "wall"}
    ]


class OcclusionAwareCoverageTests(unittest.TestCase):
    def test_occluded_cells_are_not_reconstructed_when_flag_is_on(self) -> None:
        scenario_on, result_on = _run_urban(occlusion_aware_mapping=True)
        _scenario_off, result_off = _run_urban(occlusion_aware_mapping=False)

        scanned_on = _newly_scanned_xy(result_on)
        scanned_off = _newly_scanned_xy(result_off)
        off_only = sorted(scanned_off - scanned_on)
        self.assertGreater(len(off_only), 0)

        query = scenario_on.environment.query
        terrain = scenario_on.terrain
        profile = SensorVisibilityModel.optical_default()
        hard_blockers = _hard_blockers(scenario_on)
        self.assertGreater(len(hard_blockers), 0)

        fully_occluded_cells: list[tuple[float, float]] = []
        for cx, cy in off_only:
            target = np.array(
                [cx, cy, float(terrain.height_at(cx, cy)) + 1.0],
                dtype=float,
            )
            visible_from_any_drone = False
            for frame in result_on.frames:
                for node in frame.nodes:
                    if not node.is_mobile:
                        continue
                    if query.los(
                        np.asarray(node.position, dtype=float),
                        target,
                        profile,
                        terrain_clearance_m=1.0,
                    ).visible:
                        visible_from_any_drone = True
                        break
                if visible_from_any_drone:
                    break
            if not visible_from_any_drone:
                fully_occluded_cells.append((cx, cy))

        self.assertGreater(
            len(fully_occluded_cells),
            0,
            "expected at least one cell stamped by legacy coverage but blocked by urban LOS",
        )
        self.assertTrue(scanned_on.isdisjoint(fully_occluded_cells))

    def test_occlusion_aware_new_cells_are_deterministic(self) -> None:
        _scenario_a, result_a = _run_urban(occlusion_aware_mapping=True)
        _scenario_b, result_b = _run_urban(occlusion_aware_mapping=True)

        by_frame_a = [
            tuple(frame.scan_mission_state.newly_scanned_cells)
            for frame in result_a.frames
            if frame.scan_mission_state is not None
        ]
        by_frame_b = [
            tuple(frame.scan_mission_state.newly_scanned_cells)
            for frame in result_b.frames
            if frame.scan_mission_state is not None
        ]
        self.assertEqual(by_frame_a, by_frame_b)


class OcclusionShowcaseTests(unittest.TestCase):
    """Occlusion is small for near-nadir mapping over open/flat terrain but grows
    with tall, dense obstacles — the "mapping through dense obstacles" payoff."""

    def test_tall_dense_obstacles_produce_more_occlusion_than_open_terrain(self) -> None:
        open_shadow = _shadow_cell_count("coastal")  # flat, effectively no blockers
        dense_shadow = _shadow_cell_count("military_compound")  # tall buildings + walls

        # Dense obstacles must shadow substantially more ground than open terrain.
        self.assertGreaterEqual(dense_shadow, 12)
        self.assertGreater(dense_shadow, 2 * open_shadow)


class ObstacleRoutedRedirectTests(unittest.TestCase):
    def test_route_points_avoid_expanded_hard_blocker_interiors(self) -> None:
        scenario = build_default_scenario(
            options=_urban_options(occlusion_aware_mapping=True),
            seed=7,
        )
        planner = PathPlanner2D(
            bounds_xy_m=scenario.environment.bounds_xy_m,
            obstacle_layer=scenario.environment.obstacles,
            config=PlannerConfig(),
        )

        route = None
        blocker = None
        for candidate in _hard_blockers(scenario):
            bounds = candidate.bounds_xy_m()
            center_y = 0.5 * (bounds.y_min_m + bounds.y_max_m)
            for margin in (40.0, 80.0, 120.0):
                start = np.array([bounds.x_min_m - margin, center_y], dtype=float)
                goal = np.array([bounds.x_max_m + margin, center_y], dtype=float)
                route = planner.plan_route(
                    start,
                    goal,
                    clearance_m=planner.config.drone_clearance_m,
                )
                if route is not None:
                    blocker = candidate
                    break
            if route is not None:
                break

        self.assertIsNotNone(route)
        self.assertIsNotNone(blocker)
        expanded = planner.expanded_polygon_for_primitive(
            blocker,
            planner.config.drone_clearance_m,
        )
        for point in route.points_xy_m:
            inside = _point_in_polygon(point, expanded) and not _point_on_polygon_boundary(
                point,
                expanded,
            )
            self.assertFalse(inside, f"route point {point.tolist()} is inside expanded blocker")

    def test_occlusion_aware_run_samples_do_not_collide_with_hard_blockers(self) -> None:
        scenario, result = _run_urban(occlusion_aware_mapping=True)
        bounds = scenario.environment.bounds_xy_m
        self.assertIsInstance(bounds, Bounds2D)
        self.assertGreater(len(_hard_blockers(scenario)), 0)

        for frame in result.frames:
            for node in frame.nodes:
                if not node.is_mobile:
                    continue
                pos = np.asarray(node.position, dtype=float)
                collision = scenario.environment.obstacles.point_collides(
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                )
                self.assertIsNone(collision, f"{node.node_id} collided at t={frame.timestamp_s}")


if __name__ == "__main__":
    unittest.main()
