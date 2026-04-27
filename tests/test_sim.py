from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import PlatformFrame, PlatformMetrics, TruthState
from argusnet.sensing.models.noise import SensorErrorConfig, SensorModel
from argusnet.simulation.sim import (
    REJECT_OBJECT_OCCLUSION,
    REJECT_OUT_OF_COVERAGE,
    ObservationBatch,
    ObservationTriggeredFollowController,
    SimNode,
    build_observations,
    orbital_path,
)
from argusnet.world.environment import (
    Bounds2D,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverLayer,
    ObstacleLayer,
    TerrainLayer,
)
from argusnet.world.terrain import OccludingObject, TerrainModel
from argusnet.world.weather import weather_from_preset


class SimulationEnvironmentTest(unittest.TestCase):
    def test_object_occlusion_rejects_blocked_observations(self) -> None:
        terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
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
        occluder = OccludingObject(
            object_id="warehouse-1",
            center_x_m=50.0,
            center_y_m=0.0,
            radius_m=8.0,
            height_agl_m=30.0,
        )

        batch = build_observations(
            rng=np.random.default_rng(4),
            nodes=[node],
            truths=[truth],
            timestamp_s=0.0,
            terrain=terrain,
            occluding_objects=[occluder],
        )

        self.assertEqual([], batch.observations)
        self.assertEqual(1, batch.attempted_count)
        self.assertEqual(1, batch.rejection_counts.get(REJECT_OBJECT_OCCLUSION))
        self.assertEqual(1, batch.rejected_by_target.get("asset-a"))
        self.assertEqual(1, len(batch.generation_rejections))
        rejection = batch.generation_rejections[0]
        self.assertTrue(np.allclose(rejection.origin, np.array([0.0, 0.0, 10.0], dtype=float)))
        self.assertTrue(np.allclose(rejection.attempted_point, truth.position))
        self.assertEqual("building", rejection.blocker_type)
        self.assertIsNotNone(rejection.closest_point)

    def test_out_of_coverage_generation_rejection_includes_closest_point(self) -> None:
        bounds = Bounds2D(0.0, 100.0, 0.0, 100.0)
        terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        environment = EnvironmentModel(
            environment_id="coverage-test",
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds,
            terrain=TerrainLayer.from_analytic(
                environment_id="coverage-test",
                terrain_model=terrain,
                bounds_xy_m=bounds,
                resolution_m=5.0,
            ),
            obstacles=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=25.0, primitives=()),
            land_cover=LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=10.0),
        )
        node = SimNode(
            node_id="ground-a",
            is_mobile=False,
            bearing_std_rad=0.002,
            dropout_probability=0.0,
            max_range_m=500.0,
            trajectory=lambda _: (
                np.array([10.0, 10.0, 5.0], dtype=float),
                np.zeros(3, dtype=float),
            ),
        )
        truth = TruthState(
            target_id="asset-a",
            position=np.array([140.0, 40.0, 12.0], dtype=float),
            velocity=np.zeros(3, dtype=float),
            timestamp_s=0.0,
        )

        batch = build_observations(
            rng=np.random.default_rng(4),
            nodes=[node],
            truths=[truth],
            timestamp_s=0.0,
            terrain=terrain,
            environment=environment,
        )

        self.assertEqual([], batch.observations)
        self.assertEqual(1, batch.rejection_counts.get(REJECT_OUT_OF_COVERAGE))
        self.assertEqual(1, len(batch.generation_rejections))
        rejection = batch.generation_rejections[0]
        self.assertEqual(REJECT_OUT_OF_COVERAGE, rejection.reason)
        self.assertIsNotNone(rejection.closest_point)
        self.assertLessEqual(float(rejection.closest_point[0]), 100.0)

    def test_orbital_path_stays_above_ground_plane(self) -> None:
        terrain = TerrainModel(
            ground_plane_m=25.0,
            base_elevation_m=-20.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        trajectory = orbital_path(
            center_xy=np.array([0.0, 0.0], dtype=float),
            radii_xy=np.array([10.0, 20.0], dtype=float),
            base_agl_m=-5.0,
            vertical_amplitude_m=0.0,
            omega=0.0,
            phase=0.0,
            terrain=terrain,
            min_agl_m=18.0,
        )

        position, _ = trajectory(3.0)

        self.assertGreaterEqual(position[2], terrain.ground_plane_m + 18.0)

    def test_adaptive_drone_follow_requires_hit_from_that_drone(self) -> None:
        class DummyFollowTrajectory:
            def __init__(self) -> None:
                self.seed_calls = []

            def __call__(self, timestamp_s: float):
                return np.array([0.0, 0.0, 200.0], dtype=float), np.array(
                    [0.0, 0.0, 0.0], dtype=float
                )

            def seed(
                self, *, position: np.ndarray, velocity: np.ndarray, timestamp_s: float
            ) -> None:
                self.seed_calls.append((position.copy(), velocity.copy(), float(timestamp_s)))

            def reset_state(self) -> None:
                return

        follow_trajectory = DummyFollowTrajectory()
        controller = ObservationTriggeredFollowController(
            node_id="drone-east",
            search_trajectory=lambda _: (
                np.array([5.0, 0.0, 180.0], dtype=float),
                np.zeros(3, dtype=float),
            ),
            follow_trajectory=follow_trajectory,
            preferred_target_id="asset-a",
        )
        controller(2.0)

        frame = PlatformFrame(
            timestamp_s=2.0,
            nodes=[],
            observations=[],
            rejected_observations=[],
            tracks=[],
            truths=[],
            metrics=PlatformMetrics(
                mean_error_m=None,
                max_error_m=None,
                active_track_count=0,
                observation_count=0,
                accepted_observation_count=0,
                rejected_observation_count=0,
                mean_measurement_std_m=None,
            ),
        )

        controller.update_from_frame(
            frame,
            ObservationBatch(
                observations=[],
                attempted_count=1,
                rejection_counts={},
                accepted_by_target={"asset-a": 1},
                rejected_by_target={},
                accepted_by_node_target={("ground-a", "asset-a"): 1},
            ),
        )
        self.assertFalse(controller.engaged)
        self.assertEqual([], follow_trajectory.seed_calls)

        controller.update_from_frame(
            frame,
            ObservationBatch(
                observations=[],
                attempted_count=1,
                rejection_counts={},
                accepted_by_target={"asset-a": 1},
                rejected_by_target={},
                accepted_by_node_target={("drone-east", "asset-a"): 1},
            ),
        )
        self.assertTrue(controller.engaged)
        self.assertEqual(1, len(follow_trajectory.seed_calls))

    def test_airborne_node_can_observe_lower_air_target(self) -> None:
        terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        drone = SimNode(
            node_id="drone-east",
            is_mobile=True,
            bearing_std_rad=0.002,
            dropout_probability=0.0,
            max_range_m=500.0,
            trajectory=lambda _: (
                np.array([0.0, 0.0, 220.0], dtype=float),
                np.zeros(3, dtype=float),
            ),
        )
        truth = TruthState(
            target_id="asset-a",
            position=np.array([120.0, 0.0, 150.0], dtype=float),
            velocity=np.zeros(3, dtype=float),
            timestamp_s=0.0,
        )

        batch = build_observations(
            rng=np.random.default_rng(4),
            nodes=[drone],
            truths=[truth],
            timestamp_s=0.0,
            terrain=terrain,
        )

        self.assertEqual(1, len(batch.observations))
        self.assertEqual("drone-east", batch.observations[0].node_id)
        self.assertEqual(1, batch.accepted_by_node_target.get(("drone-east", "asset-a")))

    def test_observations_use_sensor_models_weather_and_clutter_deterministically(self) -> None:
        terrain = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        node = SimNode(
            node_id="ground-a",
            is_mobile=False,
            bearing_std_rad=0.002,
            dropout_probability=0.0,
            max_range_m=1200.0,
            trajectory=lambda _: (
                np.array([0.0, 0.0, 10.0], dtype=float),
                np.zeros(3, dtype=float),
            ),
        )
        truth = TruthState(
            target_id="asset-a",
            position=np.array([35.0, 0.0, 40.0], dtype=float),
            velocity=np.zeros(3, dtype=float),
            timestamp_s=0.0,
        )
        sensor_model = SensorModel(
            config=SensorErrorConfig(
                base_bearing_std_rad=0.002,
                detection_range_knee_m=5000.0,
                detection_range_falloff_m=5000.0,
                min_detection_probability=1.0,
                false_alarm_rate_per_scan=3.0,
                bias_drift_rate_rad_per_s=0.0,
            )
        )
        sensor_model.initialize(seed=17)

        batch_a = build_observations(
            rng=np.random.default_rng(4),
            nodes=[node],
            truths=[truth],
            timestamp_s=3.0,
            terrain=terrain,
            weather=weather_from_preset("light_rain"),
            sensor_models={"ground-a": sensor_model},
            seed=17,
        )
        batch_b = build_observations(
            rng=np.random.default_rng(4),
            nodes=[node],
            truths=[truth],
            timestamp_s=3.0,
            terrain=terrain,
            weather=weather_from_preset("light_rain"),
            sensor_models={"ground-a": sensor_model},
            seed=17,
        )

        self.assertTrue(any(obs.target_id == "asset-a" for obs in batch_a.observations))
        self.assertTrue(any(obs.target_id == "clutter" for obs in batch_a.observations))
        self.assertEqual(
            [(obs.target_id, round(float(obs.confidence), 4)) for obs in batch_a.observations],
            [(obs.target_id, round(float(obs.confidence), 4)) for obs in batch_b.observations],
        )


if __name__ == "__main__":
    unittest.main()
