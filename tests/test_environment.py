from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from smart_tracker.environment import (
    Bounds2D,
    BuildingPrism,
    EnvironmentCRS,
    EnvironmentModel,
    ForestStand,
    LandCoverClass,
    LandCoverLayer,
    ObstacleLayer,
    SensorVisibilityModel,
    TerrainLayer,
    load_environment_bundle,
    write_environment_bundle,
)
from smart_tracker.terrain import TerrainModel


class TerrainLayerTest(unittest.TestCase):
    def test_bilinear_height_interpolation(self) -> None:
        terrain = TerrainLayer.from_height_grid(
            environment_id="interp",
            bounds_xy_m=Bounds2D(0.0, 2.0, 0.0, 2.0),
            heights_m=np.array(
                [
                    [0.0, 2.0, 4.0],
                    [2.0, 4.0, 6.0],
                    [4.0, 6.0, 8.0],
                ],
                dtype=float,
            ),
            resolution_m=1.0,
            tile_size_cells=2,
        )

        self.assertAlmostEqual(4.0, terrain.height_at(1.0, 1.0))
        self.assertAlmostEqual(3.0, terrain.height_at(0.5, 1.0))

    def test_tile_boundary_continuity(self) -> None:
        heights = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
            ],
            dtype=float,
        )
        terrain = TerrainLayer.from_height_grid(
            environment_id="boundary",
            bounds_xy_m=Bounds2D(0.0, 4.0, 0.0, 4.0),
            heights_m=heights,
            resolution_m=1.0,
            tile_size_cells=2,
        )

        self.assertAlmostEqual(4.0, terrain.height_at(2.0, 2.0))
        self.assertAlmostEqual(3.99, terrain.height_at(1.99, 2.0), places=2)
        self.assertAlmostEqual(4.01, terrain.height_at(2.01, 2.0), places=2)

    def test_from_geotiff_promotes_dem_to_public_api(self) -> None:
        heights = np.array(
            [
                [120.0, 125.0, 130.0],
                [122.0, 127.0, 132.0],
                [124.0, 129.0, 134.0],
            ],
            dtype=np.float32,
        )
        with TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "terrain.tif"
            tifffile.imwrite(
                dem_path,
                heights,
                extratags=[
                    (33550, "d", 3, (10.0, 10.0, 0.0), False),
                    (33922, "d", 6, (0.0, 0.0, 0.0, 500000.0, 4100000.0, 0.0), False),
                ],
            )
            terrain = TerrainLayer.from_geotiff(dem_path, source_crs="EPSG:32611")

        summary = terrain.terrain_summary()
        self.assertAlmostEqual(summary["min_height_m"], 120.0)
        self.assertAlmostEqual(summary["max_height_m"], 134.0)
        self.assertGreater(terrain.bounds_xy_m.width_m, 0.0)
        self.assertGreater(terrain.bounds_xy_m.height_m, 0.0)


class EnvironmentQueryTest(unittest.TestCase):
    def _flat_environment(self, obstacle) -> EnvironmentModel:
        bounds = Bounds2D(0.0, 200.0, 0.0, 200.0)
        terrain_model = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        terrain = TerrainLayer.from_analytic(
            environment_id="flat",
            terrain_model=terrain_model,
            bounds_xy_m=bounds,
            resolution_m=5.0,
        )
        land_cover = LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=10.0)
        return EnvironmentModel(
            environment_id="flat-scene",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=ObstacleLayer(
                bounds_xy_m=bounds,
                tile_size_m=terrain.tile_size_cells * terrain.base_resolution_m,
                primitives=(obstacle,),
            ),
            land_cover=land_cover,
        )

    def test_building_blocks_line_of_sight(self) -> None:
        environment = self._flat_environment(
            BuildingPrism(
                primitive_id="hangar",
                footprint_xy_m=[[90.0, 80.0], [110.0, 80.0], [110.0, 120.0], [90.0, 120.0]],
                base_z_m=0.0,
                top_z_m=35.0,
            )
        )

        visibility = environment.query.los(
            np.array([20.0, 100.0, 12.0], dtype=float),
            np.array([180.0, 100.0, 12.0], dtype=float),
            sensor_profile=SensorVisibilityModel.optical_default(),
        )

        self.assertFalse(visibility.visible)
        self.assertEqual("building", visibility.blocker_type)
        self.assertIsNotNone(visibility.closest_point)
        self.assertGreaterEqual(float(visibility.closest_point[0]), 89.0)
        self.assertLessEqual(float(visibility.closest_point[0]), 111.0)

    def test_forest_attenuates_visibility(self) -> None:
        environment = self._flat_environment(
            ForestStand(
                primitive_id="trees",
                footprint_xy_m=[[70.0, 70.0], [130.0, 70.0], [130.0, 130.0], [70.0, 130.0]],
                canopy_base_z_m=2.0,
                canopy_top_z_m=28.0,
                density=0.35,
            )
        )

        visibility = environment.query.los(
            np.array([20.0, 100.0, 14.0], dtype=float),
            np.array([180.0, 100.0, 14.0], dtype=float),
            sensor_profile=SensorVisibilityModel.optical_default(),
        )

        self.assertTrue(visibility.visible)
        self.assertLess(visibility.transmittance, 1.0)
        self.assertLess(visibility.detection_multiplier, 1.0)
        self.assertGreater(visibility.noise_multiplier, 1.0)
        self.assertIsNotNone(visibility.closest_point)

    def test_out_of_coverage_is_reported(self) -> None:
        environment = self._flat_environment(
            BuildingPrism(
                primitive_id="hangar",
                footprint_xy_m=[[90.0, 80.0], [110.0, 80.0], [110.0, 120.0], [90.0, 120.0]],
                base_z_m=0.0,
                top_z_m=35.0,
            )
        )

        visibility = environment.query.los(
            np.array([-20.0, 50.0, 12.0], dtype=float),
            np.array([180.0, 100.0, 12.0], dtype=float),
        )

        self.assertFalse(visibility.visible)
        self.assertEqual("out_of_coverage", visibility.blocker_type)
        self.assertIsNotNone(visibility.closest_point)
        self.assertGreaterEqual(float(visibility.closest_point[0]), 0.0)


class EnvironmentBundleTest(unittest.TestCase):
    def test_bundle_round_trip_preserves_height_and_land_cover(self) -> None:
        bounds = Bounds2D(0.0, 20.0, 0.0, 20.0)
        terrain = TerrainLayer.from_height_grid(
            environment_id="bundle",
            bounds_xy_m=bounds,
            heights_m=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ],
                dtype=float,
            ),
            resolution_m=10.0,
            tile_size_cells=2,
        )
        classes = np.array([[int(LandCoverClass.OPEN), int(LandCoverClass.FOREST)]], dtype=np.uint8)
        density = np.array([[0, 200]], dtype=np.uint8)
        land_cover = LandCoverLayer.from_rasters(
            bounds_xy_m=bounds,
            classes=classes,
            density=density,
            resolution_m=10.0,
            tile_size_cells=2,
        )
        obstacle = BuildingPrism(
            primitive_id="shed",
            footprint_xy_m=[[5.0, 5.0], [8.0, 5.0], [8.0, 8.0], [5.0, 8.0]],
            base_z_m=1.0,
            top_z_m=6.0,
        )
        environment = EnvironmentModel(
            environment_id="bundle-scene",
            crs=EnvironmentCRS(runtime_crs_id="local-enu"),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=20.0, primitives=(obstacle,)),
            land_cover=land_cover,
        )

        with TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "map"
            write_environment_bundle(str(bundle_path), environment)
            loaded = load_environment_bundle(str(bundle_path))

        self.assertEqual("bundle-scene", loaded.environment_id)
        self.assertAlmostEqual(environment.terrain.height_at(10.0, 10.0), loaded.terrain.height_at(10.0, 10.0))
        self.assertEqual(LandCoverClass.FOREST, loaded.land_cover.land_cover_at(15.0, 5.0))
        self.assertEqual(1, len(loaded.obstacles.primitives))


if __name__ == "__main__":
    unittest.main()
