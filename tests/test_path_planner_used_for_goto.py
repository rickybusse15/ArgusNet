"""Verify the executor's goto trajectories detour around obstacles.

Phase C2 of the post-Phase-B follow-up plan: ``_make_goto_hover``'s straight
line is replaced by a ``PathPlanner2D``-derived multi-waypoint trajectory.
The integration in ``run_simulation`` is deterministic — it constructs a
single ``PathPlanner2D(bounds, obstacles, PlannerConfig())`` and calls
``plan_route(start, goal, clearance_m=...)`` from ``_make_planned_goto``.

This test exercises the same planner contract directly: with a building
sitting between two free points, ``plan_route`` must return a route with more
than two vertices. If this contract holds, the executor's planned trajectory
will detour rather than cut through the building. The fallback path
(``plan_route`` returning ``None``) is also verified for blocked endpoints.
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.planning.planner_base import PathPlanner2D, PlannerConfig
from argusnet.world.environment import Bounds2D, ObstacleLayer
from argusnet.world.obstacles import BuildingPrism


def _planner_with_building_in_middle() -> PathPlanner2D:
    bounds = Bounds2D(x_min_m=-100.0, x_max_m=100.0, y_min_m=-100.0, y_max_m=100.0)
    # Square building footprint centered at (0, 0), 30 m on a side.
    building = BuildingPrism(
        primitive_id="bldg-1",
        footprint_xy_m=[[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]],
        base_z_m=0.0,
        top_z_m=30.0,
    )
    obstacles = ObstacleLayer(
        bounds_xy_m=bounds,
        tile_size_m=50.0,
        primitives=[building],
    )
    return PathPlanner2D(bounds_xy_m=bounds, obstacle_layer=obstacles, config=PlannerConfig())


class PlannerDetourContractTest(unittest.TestCase):
    def test_route_around_building_has_more_than_two_waypoints(self) -> None:
        planner = _planner_with_building_in_middle()
        # Start west of the building, goal east of it — straight line goes
        # through the obstacle, so a detour is required.
        route = planner.plan_route(
            (-80.0, 0.0), (80.0, 0.0), clearance_m=planner.config.drone_clearance_m
        )
        self.assertIsNotNone(route, "Planner must find a route around a single building")
        self.assertGreater(
            route.vertex_count,
            2,
            f"Expected detour with >2 waypoints, got {route.vertex_count}",
        )
        # Length must exceed the straight-line distance of 160 m.
        self.assertGreater(route.length_m, 160.0)

    def test_route_returns_none_when_start_is_inside_obstacle(self) -> None:
        planner = _planner_with_building_in_middle()
        # Start point (0, 0) is inside the building; planner refuses the route.
        route = planner.plan_route(
            (0.0, 0.0), (80.0, 0.0), clearance_m=planner.config.drone_clearance_m
        )
        self.assertIsNone(route)

    def test_clear_segment_returns_two_vertex_route(self) -> None:
        planner = _planner_with_building_in_middle()
        # Both points are well clear of the building — direct line is fine.
        route = planner.plan_route(
            (-80.0, 60.0), (80.0, 60.0), clearance_m=planner.config.drone_clearance_m
        )
        self.assertIsNotNone(route)
        self.assertEqual(route.vertex_count, 2)
        self.assertAlmostEqual(route.length_m, 160.0, places=3)


class PlannedRouteShapeTest(unittest.TestCase):
    """Sanity checks on the geometry the executor's helper consumes."""

    def test_route_points_are_2d(self) -> None:
        planner = _planner_with_building_in_middle()
        route = planner.plan_route(
            (-80.0, 0.0), (80.0, 0.0), clearance_m=planner.config.drone_clearance_m
        )
        self.assertIsNotNone(route)
        pts = np.asarray(route.points_xy_m)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 2)

    def test_cumulative_length_matches_segments(self) -> None:
        planner = _planner_with_building_in_middle()
        route = planner.plan_route(
            (-80.0, 0.0), (80.0, 0.0), clearance_m=planner.config.drone_clearance_m
        )
        self.assertIsNotNone(route)
        pts = np.asarray(route.points_xy_m)
        seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self.assertAlmostEqual(float(seg_lens.sum()), route.length_m, places=4)


if __name__ == "__main__":
    unittest.main()
