from __future__ import annotations

import unittest

import numpy as np

from argusnet.world.environment import Bounds2D, EnvironmentCRS, EnvironmentModel, LandCoverLayer, ObstacleLayer, TerrainLayer
from argusnet.world.obstacles import BuildingPrism, CylinderObstacle, ForestStand, OrientedBox, WallSegment, _point_in_polygon
from argusnet.planning.planner_base import PathPlanner2D
from argusnet.simulation.sim import collision_aware_position


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

        self.assertFalse(self.building.point_inside(float(pushed_building[0]), float(pushed_building[1]), 10.0))
        self.assertFalse(self.cylinder.point_inside(float(pushed_cylinder[0]), float(pushed_cylinder[1]), 10.0))

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

        building_polygon = planner.expanded_polygon_for_primitive(self.building, planner.config.target_clearance_m)
        cylinder_polygon = planner.expanded_polygon_for_primitive(self.cylinder, planner.config.target_clearance_m)

        self.assertTrue(_point_in_polygon(np.array([20.0, 20.0], dtype=float), building_polygon))
        self.assertGreater(float(np.max(cylinder_polygon[:, 0])), self.cylinder.center_x_m + self.cylinder.radius_m)
        self.assertLess(float(np.min(cylinder_polygon[:, 0])), self.cylinder.center_x_m - self.cylinder.radius_m)

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

        route = planner.plan_route([20.0, 20.0], [100.0, 20.0], clearance_m=planner.config.drone_clearance_m)

        self.assertIsNone(route)
        pushed_xy = self.building.push_outside_xy(20.0, 20.0)
        self.assertFalse(self.building.point_inside(float(pushed_xy[0]), float(pushed_xy[1]), 10.0))


if __name__ == "__main__":
    unittest.main()
