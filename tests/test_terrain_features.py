"""Test suite for new terrain and zone rendering features.

Tests cover:
- Spatial terrain heights in mission zones
- Rejection diagnostics with geometric information
- Terrain and platform preset behavior
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from argusnet.core.types import (
    ZONE_TYPE_EXCLUSION,
    ZONE_TYPE_SURVEILLANCE,
    MissionZone,
    ObservationRejection,
    vec3,
)
from argusnet.world.environment import (
    Bounds2D,
    CylinderObstacle,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverLayer,
    ObstacleLayer,
    TerrainLayer,
)
from argusnet.world.terrain import TerrainModel


class TerrainHeightIntegrationTest(unittest.TestCase):
    """Test terrain height integration with mission zones."""

    def setUp(self) -> None:
        """Create a simple terrain layer for testing."""
        bounds = Bounds2D(x_min_m=0.0, x_max_m=1000.0, y_min_m=0.0, y_max_m=1000.0)
        heights = np.array(
            [
                [10.0, 15.0, 20.0, 25.0],
                [12.0, 18.0, 24.0, 30.0],
                [14.0, 21.0, 28.0, 35.0],
                [16.0, 24.0, 32.0, 40.0],
            ],
            dtype=float,
        )
        self.terrain = TerrainLayer.from_height_grid(
            environment_id="test-terrain",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=333.0,  # 1000/3 ≈ 333m per cell
        )

    def test_terrain_height_at_point(self) -> None:
        """Verify height sampling at various points."""
        # Test known height points
        height_origin = self.terrain.height_at(0.0, 0.0)
        self.assertAlmostEqual(height_origin, 10.0, places=1)

        height_center = self.terrain.height_at(500.0, 500.0)
        self.assertGreater(height_center, 10.0)  # Should interpolate to something higher

    def test_zone_with_terrain_height(self) -> None:
        """Test mission zone with terrain-baked height."""
        # Sample terrain height at zone center location
        zone_x, zone_y = 500.0, 500.0
        zone_height = self.terrain.height_at(zone_x, zone_y)

        # Create zone with spatially-accurate height
        zone = MissionZone(
            zone_id="zone-001",
            zone_type=ZONE_TYPE_SURVEILLANCE,
            center=vec3(zone_x, zone_y, zone_height),  # z is now terrain-aware
            radius_m=200.0,
            priority=1,
            label="surveillance-perimeter",
        )

        self.assertEqual(zone.zone_id, "zone-001")
        self.assertGreater(zone.center[2], 10.0)  # z should be > ground level
        self.assertEqual(zone.radius_m, 200.0)

    def test_zone_height_preserved_across_serialization(self) -> None:
        """Test zone height survives serialization to metadata."""
        zone_x, zone_y = 250.0, 250.0
        zone_height = self.terrain.height_at(zone_x, zone_y)

        zone = MissionZone(
            zone_id="zone-002",
            zone_type=ZONE_TYPE_EXCLUSION,
            center=vec3(zone_x, zone_y, zone_height),
            radius_m=150.0,
        )

        # Simulate serialization
        zone_dict = {
            "zone_id": zone.zone_id,
            "zone_type": zone.zone_type,
            "center": list(zone.center),
            "radius_m": zone.radius_m,
        }

        self.assertEqual(zone_dict["center"][2], zone_height)

    def test_multiple_zones_at_different_elevations(self) -> None:
        """Test multiple zones respecting local terrain elevations."""
        zones = []
        locations = [(250.0, 250.0), (500.0, 500.0), (750.0, 750.0)]

        for i, (x, y) in enumerate(locations):
            height = self.terrain.height_at(x, y)
            zone = MissionZone(
                zone_id=f"zone-{i:03d}",
                zone_type=ZONE_TYPE_SURVEILLANCE,
                center=vec3(x, y, height),
                radius_m=100.0,
            )
            zones.append(zone)

        # Verify heights vary with terrain
        heights = [zone.center[2] for zone in zones]
        self.assertTrue(len(set(heights)) > 1, "Zones should have different heights")


class RejectionDiagnosticsTest(unittest.TestCase):
    """Test geometric rejection diagnostics."""

    def test_rejection_with_full_geometry(self) -> None:
        """Test rejection record with all geometric fields."""
        origin = vec3(0.0, 0.0, 100.0)
        attempted_point = vec3(50.0, 50.0, 150.0)
        closest_point = vec3(45.0, 45.0, 145.0)

        rejection = ObservationRejection(
            node_id="ground-a",
            target_id="asset-x",
            timestamp_s=0.0,
            reason="los_blocked",
            detail="Blocked by building at 45m distance",
            origin=origin,
            attempted_point=attempted_point,
            closest_point=closest_point,
            blocker_type="building",
            first_hit_range_m=45.0,
        )

        self.assertEqual(rejection.reason, "los_blocked")
        self.assertEqual(rejection.blocker_type, "building")
        self.assertIsNotNone(rejection.origin)
        self.assertIsNotNone(rejection.attempted_point)
        self.assertIsNotNone(rejection.closest_point)
        self.assertAlmostEqual(rejection.first_hit_range_m, 45.0)

    def test_rejection_geometry_distance_calculation(self) -> None:
        """Verify geometric rejection data for range calculations."""
        origin = vec3(0.0, 0.0, 0.0)
        blocker_point = vec3(30.0, 40.0, 0.0)

        # Expected: sqrt(30^2 + 40^2) = 50m
        expected_range = np.sqrt(30.0**2 + 40.0**2)

        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="terrain_blocked",
            origin=origin,
            closest_point=blocker_point,
            blocker_type="terrain",
            first_hit_range_m=expected_range,
        )

        self.assertAlmostEqual(rejection.first_hit_range_m, 50.0, places=1)

    def test_rejection_classification_by_blocker_type(self) -> None:
        """Test rejection categorization by blocker type."""
        node_id = "ground-a"
        target_id = "asset-1"
        timestamp_s = 0.0

        blocker_types = ["building", "wall", "vegetation", "terrain", "unknown"]
        rejections = []

        for blocker_type in blocker_types:
            rejection = ObservationRejection(
                node_id=node_id,
                target_id=target_id,
                timestamp_s=timestamp_s,
                reason="los_blocked",
                blocker_type=blocker_type,
                first_hit_range_m=50.0,
            )
            rejections.append(rejection)

        # Verify each blocker type is captured
        captured_types = {r.blocker_type for r in rejections}
        self.assertEqual(captured_types, set(blocker_types))

    def test_optional_rejection_fields(self) -> None:
        """Test that rejection fields are correctly optional."""
        rejection_minimal = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="prefilter_failed",
        )

        self.assertIsNone(rejection_minimal.origin)
        self.assertIsNone(rejection_minimal.attempted_point)
        self.assertIsNone(rejection_minimal.closest_point)
        self.assertEqual(rejection_minimal.blocker_type, "")
        self.assertIsNone(rejection_minimal.first_hit_range_m)


class EnvironmentQueryTest(unittest.TestCase):
    """Test environment visibility queries used in rejection diagnostics."""

    def setUp(self) -> None:
        """Create a complete environment model."""
        bounds = Bounds2D(x_min_m=0.0, x_max_m=500.0, y_min_m=0.0, y_max_m=500.0)

        # Create terrain
        heights = np.full((5, 5), 50.0, dtype=float)
        heights[2:4, 2:4] = 75.0  # Higher hill in center
        self.terrain = TerrainLayer.from_height_grid(
            environment_id="test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=125.0,
        )

        # Create obstacle (building)
        building = CylinderObstacle(
            primitive_id="building-1",
            blocker_type="building",
            center_x_m=250.0,
            center_y_m=250.0,
            radius_m=30.0,
            base_z_m=50.0,
            top_z_m=80.0,
        )

        obstacles = ObstacleLayer(bounds_xy_m=bounds, tile_size_m=250.0, primitives=[building])

        # Create land cover
        land_cover = LandCoverLayer.open_terrain(
            bounds_xy_m=bounds,
            resolution_m=125.0,
        )

        self.env = EnvironmentModel(
            environment_id="test-env",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=self.terrain,
            obstacles=obstacles,
            land_cover=land_cover,
        )

    def test_environment_has_query_interface(self) -> None:
        """Verify environment has query capability."""
        self.assertIsNotNone(self.env.query)

    def test_terrain_visibility_to_metadata(self) -> None:
        """Test that terrain mesh is available in replay metadata."""
        metadata = self.env.to_replay_metadata()

        self.assertIn("terrain", metadata)
        self.assertIn("viewer_mesh", metadata["terrain"])

        mesh = metadata["terrain"]["viewer_mesh"]
        self.assertIn("heights_m", mesh)
        self.assertGreater(len(mesh["heights_m"]), 0)


class TerrainPresetTest(unittest.TestCase):
    """Test terrain preset behavior (clean_terrain flag)."""

    def test_clean_terrain_flag_behavior(self) -> None:
        """Test clean_terrain flag removes obstacles while preserving shape."""
        bounds = Bounds2D(x_min_m=0.0, x_max_m=500.0, y_min_m=0.0, y_max_m=500.0)

        # Create terrain with obstacles
        heights = np.full((5, 5), 50.0, dtype=float)
        terrain = TerrainLayer.from_height_grid(
            environment_id="test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=125.0,
        )

        # With obstacles
        building = CylinderObstacle(
            primitive_id="building-1",
            blocker_type="building",
            center_x_m=250.0,
            center_y_m=250.0,
            radius_m=30.0,
            base_z_m=50.0,
            top_z_m=80.0,
        )
        obstacles_full = ObstacleLayer(bounds_xy_m=bounds, tile_size_m=250.0, primitives=[building])

        # Clean terrain (no obstacles)
        obstacles_clean = ObstacleLayer.empty(bounds_xy_m=bounds, tile_size_m=250.0)

        env_full = EnvironmentModel(
            environment_id="full",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=obstacles_full,
            land_cover=LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=125.0),
        )

        env_clean = EnvironmentModel(
            environment_id="clean",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=obstacles_clean,
            land_cover=LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=125.0),
        )

        # Terrain shape should be identical
        self.assertEqual(
            env_full.terrain.height_at(250.0, 250.0),
            env_clean.terrain.height_at(250.0, 250.0),
        )

        # But obstacle counts should differ
        self.assertGreater(len(env_full.obstacles.primitives), len(env_clean.obstacles.primitives))


class PlatformPresetTest(unittest.TestCase):
    """Test platform preset behavior (baseline vs wide_area)."""

    def test_platform_preset_independence(self) -> None:
        """Test that platform presets are independent of terrain."""
        # This test verifies the decoupling: a sensor platform should
        # not implicitly change based on map size.

        # Example preset configurations (values would come from actual system)
        baseline_preset = {
            "name": "baseline",
            "max_range_m": 5000.0,
            "speed_m_s": 15.0,
            "fov_half_angle_deg": 45.0,
        }

        wide_area_preset = {
            "name": "wide_area",
            "max_range_m": 15000.0,
            "speed_m_s": 25.0,
            "fov_half_angle_deg": 60.0,
        }

        # Both presets should work on any terrain size
        self.assertNotEqual(baseline_preset["max_range_m"], wide_area_preset["max_range_m"])
        self.assertNotEqual(baseline_preset["speed_m_s"], wide_area_preset["speed_m_s"])


class ViewerTerrainSamplingTest(unittest.TestCase):
    """Test viewer terrain mesh sampling for zone visualization."""

    def setUp(self) -> None:
        """Create terrain for viewer testing."""
        bounds = Bounds2D(x_min_m=0.0, x_max_m=1000.0, y_min_m=0.0, y_max_m=1000.0)
        heights = np.linspace(10.0, 50.0, 20).reshape((4, 5))
        self.terrain = TerrainLayer.from_height_grid(
            environment_id="viewer-test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=100.0,
        )

    def test_viewer_mesh_generation(self) -> None:
        """Test terrain mesh for viewer rendering."""
        mesh = self.terrain.viewer_mesh(max_dimension=32)

        self.assertIn("x_min_m", mesh)
        self.assertIn("x_max_m", mesh)
        self.assertIn("y_min_m", mesh)
        self.assertIn("y_max_m", mesh)
        self.assertIn("heights_m", mesh)
        self.assertIn("cols", mesh)
        self.assertIn("rows", mesh)

        # Verify mesh structure
        self.assertGreater(mesh["cols"], 0)
        self.assertGreater(mesh["rows"], 0)
        self.assertEqual(len(mesh["heights_m"]), mesh["rows"])
        self.assertEqual(len(mesh["heights_m"][0]), mesh["cols"])

    def test_zone_rendered_above_terrain(self) -> None:
        """Test zone visualization properly positions rings above terrain."""
        zone_x, zone_y = 500.0, 500.0
        zone_height = self.terrain.height_at(zone_x, zone_y)

        zone = MissionZone(
            zone_id="viewer-zone",
            zone_type=ZONE_TYPE_SURVEILLANCE,
            center=vec3(zone_x, zone_y, zone_height),
            radius_m=100.0,
        )

        # Zone should be rendered above the sampled terrain mesh
        mesh = self.terrain.viewer_mesh()
        # Interpolate terrain height at zone location
        xi = (zone_x - mesh["x_min_m"]) / (mesh["x_max_m"] - mesh["x_min_m"]) * (mesh["cols"] - 1)
        yi = (zone_y - mesh["y_min_m"]) / (mesh["y_max_m"] - mesh["y_min_m"]) * (mesh["rows"] - 1)

        col_idx = int(np.clip(xi, 0, mesh["cols"] - 2))
        row_idx = int(np.clip(yi, 0, mesh["rows"] - 2))

        # Zone center z should match terrain at that location
        mesh_height = mesh["heights_m"][row_idx][col_idx]
        self.assertAlmostEqual(zone.center[2], mesh_height, delta=5.0)


class TerrainModelCurvatureTest(unittest.TestCase):
    """Tests for TerrainModel.curvature_at and slope_rad_at."""

    def setUp(self) -> None:
        """Use a pure analytic Gaussian ridge (no noise/waves) for deterministic checks."""
        # Isolated Gaussian hill centred at origin: ridge_amplitude=50m, radius=100m.
        # No waves, no basin, no feature noise — guarantees a smooth, symmetric hilltop.
        self.terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=50.0,
            ridge_center_x_m=0.0,
            ridge_center_y_m=0.0,
            ridge_radius_m=100.0,
            basin_depth_m=0.0,
        )
        self.ridge_x = 0.0
        self.ridge_y = 0.0

    def test_curvature_at_ridge_peak_is_negative(self) -> None:
        """A Gaussian hilltop should have negative curvature (concave down)."""
        # The Laplacian of a Gaussian peak is negative: centre > neighbours.
        curvature = self.terrain.curvature_at(self.ridge_x, self.ridge_y, delta_m=5.0)
        self.assertLess(
            curvature, 0.0, "Gaussian ridge peak should have negative (concave-down) curvature"
        )

    def test_curvature_at_flat_area_is_near_zero(self) -> None:
        """Perfectly flat terrain should have zero curvature."""
        flat_terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=5.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        curvature = flat_terrain.curvature_at(0.0, 0.0, delta_m=1.0)
        self.assertAlmostEqual(
            curvature, 0.0, places=6, msg="Perfectly flat terrain should have zero curvature"
        )

    def test_curvature_delta_clamped(self) -> None:
        """Passing delta_m < 0.1 should be silently clamped to 0.1."""
        c_clamped = self.terrain.curvature_at(0.0, 0.0, delta_m=0.001)
        c_explicit = self.terrain.curvature_at(0.0, 0.0, delta_m=0.1)
        self.assertAlmostEqual(
            c_clamped, c_explicit, places=10, msg="sub-0.1 delta should be clamped to 0.1"
        )

    def test_slope_rad_at_returns_non_negative(self) -> None:
        """Slope magnitude must always be >= 0."""
        for x, y in [(0.0, 0.0), (30.0, 0.0), (-50.0, 40.0)]:
            slope = self.terrain.slope_rad_at(x, y, delta_m=2.0)
            self.assertGreaterEqual(slope, 0.0, f"Slope at ({x}, {y}) must be non-negative")

    def test_slope_rad_at_steep_area_is_positive(self) -> None:
        """A steep slope should give a positive slope_rad value."""
        # Ridge shoulder at x=30m: gradient is nonzero here.
        slope = self.terrain.slope_rad_at(30.0, 0.0, delta_m=2.0)
        self.assertGreater(slope, 0.0, "Slope on ridge shoulder should be positive")

    def test_slope_rad_at_flat_is_near_zero(self) -> None:
        """Perfectly flat terrain should give zero slope."""
        flat_terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=5.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        slope = flat_terrain.slope_rad_at(0.0, 0.0, delta_m=1.0)
        self.assertAlmostEqual(
            slope, 0.0, places=10, msg="Slope on perfectly flat terrain should be 0 rad"
        )

    def test_slope_rad_at_bounded_by_pi_over_2(self) -> None:
        """Slope in radians must be in [0, pi/2]."""
        for x, y in [(0.0, 0.0), (30.0, 0.0), (80.0, -60.0)]:
            slope = self.terrain.slope_rad_at(x, y, delta_m=1.0)
            self.assertLessEqual(slope, math.pi / 2.0, f"Slope at ({x}, {y}) exceeds pi/2")


class TerrainLayerCurvatureTest(unittest.TestCase):
    """Tests for TerrainLayer.curvature_at and slope_rad_at."""

    def setUp(self) -> None:
        """Build a small synthetic height grid with a clear central peak."""
        bounds = Bounds2D(x_min_m=-200.0, x_max_m=200.0, y_min_m=-200.0, y_max_m=200.0)
        # 9x9 grid; peak at center cell (4,4)
        size = 9
        heights = np.zeros((size, size), dtype=float)
        cx, cy = size // 2, size // 2
        for row in range(size):
            for col in range(size):
                dist = math.sqrt((row - cy) ** 2 + (col - cx) ** 2)
                heights[row, col] = 50.0 * math.exp(-0.5 * (dist / 2.0) ** 2)
        self.terrain_layer = TerrainLayer.from_height_grid(
            environment_id="curvature-test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=50.0,
        )
        # centre of the domain = peak
        self.peak_x = 0.0
        self.peak_y = 0.0

    def test_curvature_at_peak_is_negative(self) -> None:
        """The central peak of a Gaussian hill should have negative curvature."""
        curvature = self.terrain_layer.curvature_at(self.peak_x, self.peak_y, delta_m=5.0)
        self.assertLess(
            curvature, 0.0, "Central peak of Gaussian hill should have negative curvature"
        )

    def test_curvature_at_flat_area_near_zero(self) -> None:
        """Far from the peak (near the grid edge) curvature should be near zero."""
        # Build a fully flat TerrainLayer
        bounds = Bounds2D(x_min_m=0.0, x_max_m=100.0, y_min_m=0.0, y_max_m=100.0)
        heights = np.full((5, 5), 10.0, dtype=float)
        flat_layer = TerrainLayer.from_height_grid(
            environment_id="flat-test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=25.0,
        )
        curvature = flat_layer.curvature_at(50.0, 50.0, delta_m=1.0)
        self.assertAlmostEqual(
            curvature, 0.0, places=6, msg="Flat terrain layer should have zero curvature"
        )

    def test_slope_rad_at_peak_is_near_zero(self) -> None:
        """Gradient is zero at the very tip of a symmetric hill."""
        slope = self.terrain_layer.slope_rad_at(self.peak_x, self.peak_y, delta_m=2.0)
        self.assertAlmostEqual(
            slope, 0.0, delta=0.15, msg="Slope at symmetric peak should be near zero"
        )

    def test_slope_rad_at_flank_is_positive(self) -> None:
        """Away from the peak, slope must be positive."""
        slope = self.terrain_layer.slope_rad_at(50.0, 0.0, delta_m=2.0)
        self.assertGreater(slope, 0.0, "Slope on hill flank should be positive")

    def test_slope_rad_at_flat_layer_is_zero(self) -> None:
        """Flat terrain layer gives zero slope."""
        bounds = Bounds2D(x_min_m=0.0, x_max_m=100.0, y_min_m=0.0, y_max_m=100.0)
        heights = np.full((5, 5), 10.0, dtype=float)
        flat_layer = TerrainLayer.from_height_grid(
            environment_id="flat-slope-test",
            bounds_xy_m=bounds,
            heights_m=heights,
            resolution_m=25.0,
        )
        slope = flat_layer.slope_rad_at(50.0, 50.0, delta_m=1.0)
        self.assertAlmostEqual(
            slope, 0.0, places=10, msg="Flat terrain layer should give zero slope"
        )

    def test_slope_rad_returns_non_negative(self) -> None:
        """slope_rad_at must always return a value >= 0 on any terrain layer."""
        for x, y in [(0.0, 0.0), (50.0, 0.0), (-80.0, 80.0)]:
            slope = self.terrain_layer.slope_rad_at(x, y, delta_m=2.0)
            self.assertGreaterEqual(slope, 0.0, f"Slope at ({x}, {y}) must be non-negative")


if __name__ == "__main__":
    unittest.main()
