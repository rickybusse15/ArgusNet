from __future__ import annotations

import unittest

import numpy as np

from smart_tracker.environment import Bounds2D, ObstacleLayer
from smart_tracker.obstacles import BuildingPrism, CylinderObstacle, WallSegment
from smart_tracker.planning import PathPlanner2D, PlannerConfig


class PathPlannerTest(unittest.TestCase):
    def test_direct_route_stays_straight_without_obstacles(self) -> None:
        planner = PathPlanner2D(
            bounds_xy_m=Bounds2D(0.0, 200.0, 0.0, 200.0),
            obstacle_layer=ObstacleLayer.empty(bounds_xy_m=Bounds2D(0.0, 200.0, 0.0, 200.0), tile_size_m=50.0),
        )

        route = planner.plan_route([20.0, 20.0], [180.0, 160.0], clearance_m=planner.config.drone_clearance_m)

        self.assertIsNotNone(route)
        self.assertEqual(2, route.vertex_count)
        self.assertTrue(np.allclose(route.points_xy_m[0], [20.0, 20.0]))
        self.assertTrue(np.allclose(route.points_xy_m[-1], [180.0, 160.0]))

    def test_route_detours_around_hard_obstacles(self) -> None:
        bounds = Bounds2D(0.0, 220.0, 0.0, 220.0)
        building = BuildingPrism(
            primitive_id="building-a",
            footprint_xy_m=[[90.0, 70.0], [130.0, 70.0], [130.0, 150.0], [90.0, 150.0]],
            base_z_m=0.0,
            top_z_m=40.0,
        )
        wall = WallSegment(
            primitive_id="wall-a",
            blocker_type="wall",
            start_xy_m=np.array([150.0, 20.0], dtype=float),
            end_xy_m=np.array([150.0, 110.0], dtype=float),
            thickness_m=6.0,
            base_z_m=0.0,
            top_z_m=30.0,
        )
        cylinder = CylinderObstacle(
            primitive_id="tower-a",
            blocker_type="building",
            center_x_m=60.0,
            center_y_m=150.0,
            radius_m=10.0,
            base_z_m=0.0,
            top_z_m=30.0,
        )
        planner = PathPlanner2D(
            bounds_xy_m=bounds,
            obstacle_layer=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=55.0, primitives=(building, wall, cylinder)),
        )

        route = planner.plan_route([20.0, 110.0], [200.0, 110.0], clearance_m=planner.config.drone_clearance_m)

        self.assertIsNotNone(route)
        self.assertGreater(route.vertex_count, 2)
        self.assert_path_clears_obstacles(route.points_xy_m, (building, wall, cylinder))

    def test_route_smoothing_removes_redundant_vertices(self) -> None:
        bounds = Bounds2D(0.0, 200.0, 0.0, 200.0)
        planner = PathPlanner2D(
            bounds_xy_m=bounds,
            obstacle_layer=ObstacleLayer.empty(bounds_xy_m=bounds, tile_size_m=50.0),
        )
        points = np.array(
            [
                [10.0, 10.0],
                [40.0, 40.0],
                [70.0, 70.0],
                [190.0, 190.0],
            ],
            dtype=float,
        )

        smoothed = planner._smooth_path(points, planner.config.target_clearance_m)

        self.assertEqual(2, len(smoothed))
        self.assertTrue(np.allclose(smoothed[0], [10.0, 10.0]))
        self.assertTrue(np.allclose(smoothed[-1], [190.0, 190.0]))

    def test_route_cache_marks_repeated_requests(self) -> None:
        bounds = Bounds2D(0.0, 200.0, 0.0, 200.0)
        building = BuildingPrism(
            primitive_id="building-a",
            footprint_xy_m=[[90.0, 80.0], [120.0, 80.0], [120.0, 140.0], [90.0, 140.0]],
            base_z_m=0.0,
            top_z_m=30.0,
        )
        planner = PathPlanner2D(
            bounds_xy_m=bounds,
            obstacle_layer=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=50.0, primitives=(building,)),
            config=PlannerConfig(snap_m=10.0),
        )

        first = planner.plan_route([20.0, 100.0], [180.0, 100.0], clearance_m=planner.config.drone_clearance_m)
        second = planner.plan_route([20.0, 100.0], [180.0, 100.0], clearance_m=planner.config.drone_clearance_m)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def assert_path_clears_obstacles(self, points_xy_m: np.ndarray, primitives: tuple[object, ...]) -> None:
        for start_xy, end_xy in zip(points_xy_m[:-1], points_xy_m[1:]):
            origin = np.array([start_xy[0], start_xy[1], 10.0], dtype=float)
            target = np.array([end_xy[0], end_xy[1], 10.0], dtype=float)
            for primitive in primitives:
                self.assertEqual(0.0, primitive.path_length_inside(origin, target))


if __name__ == "__main__":
    unittest.main()
