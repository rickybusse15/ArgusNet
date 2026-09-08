from __future__ import annotations

import unittest

import numpy as np

from argusnet.planning.planner_base import PathPlanner2D
from argusnet.simulation.sim import collision_aware_position
from argusnet.world.environment import (
    Bounds2D,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverLayer,
    ObstacleLayer,
    TerrainLayer,
)
from argusnet.world.obstacles import (
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    OrientedBox,
    WallSegment,
    _point_in_polygon,
)


class ObstacleCollisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.building = BuildingPrism(
            primitive_id="building-a",
            footprint_xy_m=[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
            base_z_m=0.0,
            top_z_m=20.0,
        )
        self.box = OrientedBox(
            primitive_id="box-a",
            blocker_type="building",
            center_x_m=60.0,
            center_y_m=20.0,
            length_m=20.0,
            width_m=10.0,
            yaw_rad=0.0,
            base_z_m=0.0,
            top_z_m=25.0,
        )
        self.wall = WallSegment(
            primitive_id="wall-a",
            blocker_type="wall",
            start_xy_m=np.array([80.0, 10.0], dtype=float),
            end_xy_m=np.array([80.0, 30.0], dtype=float),
            thickness_m=4.0,
            base_z_m=0.0,
            top_z_m=18.0,
        )
        self.cylinder = CylinderObstacle(
            primitive_id="tower-a",
            blocker_type="building",
            center_x_m=110.0,
            center_y_m=20.0,
            radius_m=6.0,
            base_z_m=0.0,
            top_z_m=30.0,
        )
        self.forest = ForestStand(
            primitive_id="forest-a",
            footprint_xy_m=[[130.0, 10.0], [150.0, 10.0], [150.0, 30.0], [130.0, 30.0]],
            canopy_base_z_m=2.0,
            canopy_top_z_m=22.0,
            density=0.5,
        )

    def test_point_inside_covers_all_primitive_types(self) -> None:
        self.assertTrue(self.building.point_inside(20.0, 20.0, 10.0))
        self.assertTrue(self.box.point_inside(60.0, 20.0, 10.0))
        self.assertTrue(self.wall.point_inside(80.0, 20.0, 10.0))
        self.assertTrue(self.cylinder.point_inside(110.0, 20.0, 10.0))
        self.assertTrue(self.forest.point_inside(140.0, 20.0, 10.0))

        self.assertFalse(self.building.point_inside(35.0, 20.0, 10.0))
        self.assertFalse(self.box.point_inside(75.0, 20.0, 10.0))
        self.assertFalse(self.wall.point_inside(90.0, 20.0, 10.0))
        self.assertFalse(self.cylinder.point_inside(118.0, 20.0, 10.0))
        self.assertFalse(self.forest.point_inside(160.0, 20.0, 10.0))

    def test_push_outside_xy_moves_point_outside(self) -> None:
        pushed_building = self.building.push_outside_xy(20.0, 20.0)
        pushed_cylinder = self.cylinder.push_outside_xy(110.0, 20.0)

        self.assertFalse(
            self.building.point_inside(float(pushed_building[0]), float(pushed_building[1]), 10.0)
        )
        self.assertFalse(
            self.cylinder.point_inside(float(pushed_cylinder[0]), float(pushed_cylinder[1]), 10.0)
        )

    def test_obstacle_layer_point_collides_finds_hard_blockers(self) -> None:
        bounds = Bounds2D(0.0, 200.0, 0.0, 60.0)
        layer = ObstacleLayer(
            bounds_xy_m=bounds,
            tile_size_m=32.0,
            primitives=(self.building, self.wall, self.forest),
        )

        collision = layer.point_collides(20.0, 20.0, 10.0)
        vegetation_collision = layer.point_collides(140.0, 20.0, 10.0)

        self.assertIsNotNone(collision)
        self.assertEqual("building-a", collision.primitive_id)
        self.assertIsNone(vegetation_collision)

    def test_planning_polygons_respect_clearance_margin(self) -> None:
        bounds = Bounds2D(0.0, 200.0, 0.0, 60.0)
        planner = PathPlanner2D(
            bounds_xy_m=bounds,
            obstacle_layer=ObstacleLayer(
                bounds_xy_m=bounds,
                tile_size_m=32.0,
                primitives=(self.building, self.box, self.wall, self.cylinder),
            ),
        )

        building_polygon = planner.expanded_polygon_for_primitive(
            self.building, planner.config.target_clearance_m
        )
        cylinder_polygon = planner.expanded_polygon_for_primitive(
            self.cylinder, planner.config.target_clearance_m
        )

        self.assertTrue(_point_in_polygon(np.array([20.0, 20.0], dtype=float), building_polygon))
        self.assertGreater(
            float(np.max(cylinder_polygon[:, 0])), self.cylinder.center_x_m + self.cylinder.radius_m
        )
        self.assertLess(
            float(np.min(cylinder_polygon[:, 0])), self.cylinder.center_x_m - self.cylinder.radius_m
        )

    def test_collision_aware_position_pushes_entity_outside_building(self) -> None:
        bounds = Bounds2D(0.0, 120.0, 0.0, 120.0)
        terrain = TerrainLayer.from_height_grid(
            environment_id="flat",
            bounds_xy_m=bounds,
            heights_m=np.zeros((3, 3), dtype=float),
            resolution_m=60.0,
            tile_size_cells=2,
        )
        obstacle_layer = ObstacleLayer(
            bounds_xy_m=bounds,
            tile_size_m=60.0,
            primitives=(self.building,),
        )
        environment = EnvironmentModel(
            environment_id="collision-scene",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=obstacle_layer,
            land_cover=LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=30.0),
        )

        adjusted_position, _ = collision_aware_position(
            np.array([20.0, 20.0, 12.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            terrain=environment.terrain,
            min_agl_m=5.0,
            obstacle_layer=environment.obstacles,
        )

        self.assertIsNone(environment.obstacles.point_collides(*adjusted_position))
        self.assertGreaterEqual(adjusted_position[2], 5.0)

    def test_collision_push_outside_remains_available_when_planning_fails(self) -> None:
        bounds = Bounds2D(0.0, 120.0, 0.0, 120.0)
        planner = PathPlanner2D(
            bounds_xy_m=bounds,
            obstacle_layer=ObstacleLayer(
                bounds_xy_m=bounds,
                tile_size_m=60.0,
                primitives=(
                    self.building,
                    WallSegment(
                        primitive_id="wall-block",
                        blocker_type="wall",
                        start_xy_m=np.array([60.0, 0.0], dtype=float),
                        end_xy_m=np.array([60.0, 120.0], dtype=float),
                        thickness_m=8.0,
                        base_z_m=0.0,
                        top_z_m=25.0,
                    ),
                ),
            ),
        )

        route = planner.plan_route(
            [20.0, 20.0], [100.0, 20.0], clearance_m=planner.config.drone_clearance_m
        )

        self.assertIsNone(route)
        pushed_xy = self.building.push_outside_xy(20.0, 20.0)
        self.assertFalse(self.building.point_inside(float(pushed_xy[0]), float(pushed_xy[1]), 10.0))


def _reference_point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    """Independent even-odd ray crossing, written from scratch as a test oracle.

    Deliberately does not share code with ``argusnet.world.obstacles`` so that it can
    detect errors in the production implementation rather than mirroring them.
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    count = len(polygon_xy)
    for index in range(count):
        x0, y0 = float(polygon_xy[index][0]), float(polygon_xy[index][1])
        x1, y1 = (
            float(polygon_xy[(index + 1) % count][0]),
            float(polygon_xy[(index + 1) % count][1]),
        )
        if (y0 > y) != (y1 > y) and x < ((x1 - x0) * (y - y0) / (y1 - y0)) + x0:
            inside = not inside
    return inside


def _distance_to_boundary_m(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    best = float("inf")
    count = len(polygon_xy)
    for index in range(count):
        start = np.asarray(polygon_xy[index], dtype=float)
        end = np.asarray(polygon_xy[(index + 1) % count], dtype=float)
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq <= 1.0e-12:
            nearest = start
        else:
            t_value = float(np.clip(np.dot(point_xy - start, edge) / length_sq, 0.0, 1.0))
            nearest = start + (edge * t_value)
        best = min(best, float(np.linalg.norm(point_xy - nearest)))
    return best


class RotatedFootprintPointInPolygonTest(unittest.TestCase):
    """Non-axis-aligned footprints.

    Axis-aligned rectangles cannot detect a sign error in the edge-crossing
    denominator, because the ``(b.x - a.x)`` numerator is zero for their vertical
    edges. These cases use footprints whose edges genuinely descend in Y.
    """

    TRIANGLE = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]], dtype=float)
    DIAMOND = np.array([[5.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]], dtype=float)

    def _assert_matches_reference_on_grid(self, polygon_xy: np.ndarray) -> None:
        """Compare against the oracle across a grid covering the footprint's neighbourhood."""
        x_min, x_max = float(np.min(polygon_xy[:, 0])), float(np.max(polygon_xy[:, 0]))
        y_min, y_max = float(np.min(polygon_xy[:, 1])), float(np.max(polygon_xy[:, 1]))
        pad_x = max((x_max - x_min) * 0.25, 1.0)
        pad_y = max((y_max - y_min) * 0.25, 1.0)

        checked = 0
        for x_m in np.linspace(x_min - pad_x, x_max + pad_x, 47):
            for y_m in np.linspace(y_min - pad_y, y_max + pad_y, 47):
                point_xy = np.array([float(x_m), float(y_m)], dtype=float)
                # Points on the boundary are legitimately ambiguous: the production
                # implementation reports them as inside, the oracle picks a side.
                if _distance_to_boundary_m(point_xy, polygon_xy) <= 1.0e-3:
                    continue
                checked += 1
                self.assertEqual(
                    _point_in_polygon(point_xy, polygon_xy),
                    _reference_point_in_polygon(point_xy, polygon_xy),
                    msg=f"disagreement at ({point_xy[0]:.4f}, {point_xy[1]:.4f})",
                )
        self.assertGreater(checked, 2000)

    def test_triangle_matches_reference_on_grid(self) -> None:
        self._assert_matches_reference_on_grid(self.TRIANGLE)

    def test_diamond_matches_reference_on_grid(self) -> None:
        self._assert_matches_reference_on_grid(self.DIAMOND)

    def test_rotated_oriented_box_matches_reference_on_grid(self) -> None:
        for yaw_rad in (0.12, 0.18, 0.7853981633974483, 2.3, -1.1):
            box = OrientedBox(
                primitive_id="box-rotated",
                blocker_type="building",
                center_x_m=60.0,
                center_y_m=20.0,
                length_m=20.0,
                width_m=10.0,
                yaw_rad=yaw_rad,
                base_z_m=0.0,
                top_z_m=25.0,
            )
            with self.subTest(yaw_rad=yaw_rad):
                self._assert_matches_reference_on_grid(box.footprint_xy_m())

    def test_triangle_and_diamond_interiors(self) -> None:
        """Explicit spot checks: the centroid is inside, the bounding-box corner is not."""
        self.assertTrue(_point_in_polygon(np.array([5.0, 5.0], dtype=float), self.TRIANGLE))
        self.assertFalse(_point_in_polygon(np.array([1.0, 8.0], dtype=float), self.TRIANGLE))
        self.assertTrue(_point_in_polygon(np.array([5.0, 5.0], dtype=float), self.DIAMOND))
        self.assertFalse(_point_in_polygon(np.array([9.0, 9.0], dtype=float), self.DIAMOND))

    def test_rotated_box_point_inside_matches_centre_and_corner(self) -> None:
        box = OrientedBox(
            primitive_id="box-rotated",
            blocker_type="building",
            center_x_m=60.0,
            center_y_m=20.0,
            length_m=20.0,
            width_m=10.0,
            yaw_rad=0.6,
            base_z_m=0.0,
            top_z_m=25.0,
        )
        self.assertTrue(box.point_inside(60.0, 20.0, 10.0))
        bounds = box.bounds_xy_m()
        # The rotated box does not fill its own axis-aligned bounding box.
        self.assertFalse(box.point_inside(bounds.x_min_m, bounds.y_min_m, 10.0))

    def test_rotated_box_push_outside_escapes_footprint(self) -> None:
        box = OrientedBox(
            primitive_id="box-rotated",
            blocker_type="building",
            center_x_m=60.0,
            center_y_m=20.0,
            length_m=20.0,
            width_m=10.0,
            yaw_rad=0.6,
            base_z_m=0.0,
            top_z_m=25.0,
        )
        pushed_xy = box.push_outside_xy(60.0, 20.0)
        self.assertFalse(box.point_inside(float(pushed_xy[0]), float(pushed_xy[1]), 10.0))

    def test_rotated_box_path_length_inside_is_positive_for_crossing_ray(self) -> None:
        """Occlusion through a rotated footprint must register a non-zero traversal."""
        box = OrientedBox(
            primitive_id="box-rotated",
            blocker_type="building",
            center_x_m=60.0,
            center_y_m=20.0,
            length_m=20.0,
            width_m=10.0,
            yaw_rad=0.6,
            base_z_m=0.0,
            top_z_m=25.0,
        )
        origin = np.array([30.0, 20.0, 10.0], dtype=float)
        target = np.array([90.0, 20.0, 10.0], dtype=float)
        self.assertGreater(box.path_length_inside(origin, target), 1.0)
        first_hit_t = box.first_hit_t(origin, target)
        self.assertIsNotNone(first_hit_t)
        assert first_hit_t is not None
        self.assertTrue(0.0 < first_hit_t < 1.0)


if __name__ == "__main__":
    unittest.main()
