"""Test suite for observation rejection diagnostics in simulation.

Tests cover:
- Rejection tracking in simulation (generation_rejections)
- Rejection tracking in fusion (rejected_observations)
- Geometric context in rejection records
- Viewer placement of rejection markers
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import (
    BearingObservation,
    NodeState,
    TruthState,
    ObservationRejection,
    PlatformFrame,
    vec3,
)
from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService
from argusnet.world.environment import (
    Bounds2D,
    TerrainLayer,
    EnvironmentModel,
    EnvironmentCRS,
    ObstacleLayer,
    LandCoverLayer,
    CylinderObstacle,
)


def create_bearing_observation(
    node_id: str,
    target_id: str,
    origin: np.ndarray,
    target: np.ndarray,
    timestamp_s: float,
    confidence: float = 1.0,
    bearing_std_rad: float = 0.002,
) -> BearingObservation:
    """Helper to create bearing observation."""
    direction = target - origin
    direction = direction / np.linalg.norm(direction)
    return BearingObservation(
        node_id=node_id,
        target_id=target_id,
        origin=origin,
        direction=direction,
        bearing_std_rad=bearing_std_rad,
        timestamp_s=timestamp_s,
        confidence=confidence,
    )


class RejectionDiagnosticsIntegrationTest(unittest.TestCase):
    """Test rejection diagnostics in complete tracking pipeline."""

    def setUp(self) -> None:
        """Set up environment and platform."""
        self.bounds = Bounds2D(x_min_m=0.0, x_max_m=1000.0, y_min_m=0.0, y_max_m=1000.0)

        # Create terrain with a blocking hill
        heights = np.linspace(10.0, 50.0, 9).reshape((3, 3))
        self.terrain = TerrainLayer.from_height_grid(
            environment_id="test",
            bounds_xy_m=self.bounds,
            heights_m=heights,
            resolution_m=500.0,
        )

        # Create building obstacle
        self.building = CylinderObstacle(
            primitive_id="building-1",
            blocker_type="building",
            center_x_m=500.0,
            center_y_m=500.0,
            radius_m=50.0,
            base_z_m=10.0,
            top_z_m=60.0,
        )

        obstacles = ObstacleLayer(bounds_xy_m=self.bounds, tile_size_m=500.0, primitives=[self.building])
        land_cover = LandCoverLayer.open_terrain(bounds_xy_m=self.bounds, resolution_m=500.0)

        self.env = EnvironmentModel(
            environment_id="test-env",
            crs=EnvironmentCRS(),
            bounds_xy_m=self.bounds,
            terrain=self.terrain,
            obstacles=obstacles,
            land_cover=land_cover,
        )

        self.platform = TrackingService(config=TrackerConfig())

    def test_fusion_rejection_with_confidence_threshold(self) -> None:
        """Test rejection when observation fails confidence threshold."""
        timestamp_s = 0.0
        nodes = [
            NodeState("sensor-1", vec3(0.0, 0.0, 10.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truths = [
            TruthState("target-1", vec3(100.0, 100.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s),
        ]

        # Low-confidence observation
        observations = [
            create_bearing_observation(
                "sensor-1",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
                confidence=0.05,  # Below typical threshold
            ),
        ]

        frame = self.platform.ingest_frame(
            timestamp_s,
            node_states=nodes,
            observations=observations,
            truths=truths,
        )

        # Observation should be rejected
        self.assertEqual(frame.metrics.rejected_observation_count, 1)

    def test_fusion_rejection_tracking_metrics(self) -> None:
        """Test rejection metrics are properly tracked."""
        timestamp_s = 0.0
        nodes = [
            NodeState("sensor-1", vec3(0.0, 0.0, 10.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("sensor-2", vec3(100.0, 0.0, 10.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truths = [
            TruthState("target-1", vec3(50.0, 50.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s),
        ]

        observations = [
            # Valid observation 1
            create_bearing_observation(
                "sensor-1",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
            ),
            # Valid observation 2 (needed for track acceptance)
            create_bearing_observation(
                "sensor-2",
                "target-1",
                nodes[1].position,
                truths[0].position,
                timestamp_s,
            ),
            # Invalid: unknown node
            create_bearing_observation(
                "ghost-node",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
            ),
            # Invalid: low confidence
            create_bearing_observation(
                "sensor-1",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
                confidence=0.1,
            ),
        ]

        frame = self.platform.ingest_frame(
            timestamp_s,
            node_states=nodes,
            observations=observations,
            truths=truths,
        )

        # Verify metrics
        self.assertEqual(frame.metrics.accepted_observation_count, 2)
        self.assertEqual(frame.metrics.rejected_observation_count, 2)
        self.assertIn("unknown_node", frame.metrics.rejection_counts)
        self.assertIn("low_confidence", frame.metrics.rejection_counts)

    def test_rejected_observations_list_accessible(self) -> None:
        """Test rejected observations are stored in frame."""
        timestamp_s = 0.0
        nodes = [
            NodeState("sensor-1", vec3(0.0, 0.0, 10.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truths = [
            TruthState("target-1", vec3(100.0, 100.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s),
        ]

        observations = [
            create_bearing_observation(
                "sensor-1",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
                confidence=0.05,  # Will be rejected
            ),
        ]

        frame = self.platform.ingest_frame(
            timestamp_s,
            node_states=nodes,
            observations=observations,
            truths=truths,
        )

        # rejected_observations should be present
        self.assertIsNotNone(frame.rejected_observations)
        self.assertEqual(len(frame.rejected_observations), 1)

    def test_generation_rejections_present_in_frame(self) -> None:
        """Test generation_rejections field exists for simulation-side rejections."""
        timestamp_s = 0.0
        nodes = [
            NodeState("sensor-1", vec3(0.0, 0.0, 10.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truths = [
            TruthState("target-1", vec3(100.0, 100.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s),
        ]
        observations = [
            create_bearing_observation(
                "sensor-1",
                "target-1",
                nodes[0].position,
                truths[0].position,
                timestamp_s,
            ),
        ]

        frame = self.platform.ingest_frame(
            timestamp_s,
            node_states=nodes,
            observations=observations,
            truths=truths,
        )

        # generation_rejections should exist (may be empty if no LOS checks)
        self.assertIsNotNone(frame.generation_rejections)


class RejectionGeometryTest(unittest.TestCase):
    """Test geometric information in rejection records."""

    def test_rejection_with_origin_point(self) -> None:
        """Test rejection records sensor origin."""
        origin = vec3(10.0, 20.0, 30.0)
        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            origin=origin,
            blocker_type="building",
        )

        self.assertIsNotNone(rejection.origin)
        np.testing.assert_array_almost_equal(rejection.origin, origin)

    def test_rejection_with_attempted_point(self) -> None:
        """Test rejection records attempted target point."""
        attempted = vec3(100.0, 150.0, 50.0)
        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            attempted_point=attempted,
            blocker_type="terrain",
        )

        self.assertIsNotNone(rejection.attempted_point)
        np.testing.assert_array_almost_equal(rejection.attempted_point, attempted)

    def test_rejection_with_closest_point(self) -> None:
        """Test rejection records point on blocker closest to ray."""
        closest = vec3(95.0, 145.0, 55.0)
        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            closest_point=closest,
            blocker_type="building",
        )

        self.assertIsNotNone(rejection.closest_point)
        np.testing.assert_array_almost_equal(rejection.closest_point, closest)

    def test_rejection_with_range_to_blocker(self) -> None:
        """Test rejection records distance to first hit."""
        origin = vec3(0.0, 0.0, 0.0)
        blocker = vec3(50.0, 0.0, 0.0)
        range_m = 50.0

        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            origin=origin,
            closest_point=blocker,
            first_hit_range_m=range_m,
            blocker_type="building",
        )

        self.assertIsNotNone(rejection.first_hit_range_m)
        self.assertAlmostEqual(rejection.first_hit_range_m, 50.0)

    def test_blocker_type_classification(self) -> None:
        """Test rejection blocker type is recorded."""
        blocker_types = [
            "building",
            "wall",
            "vegetation",
            "terrain",
            "unknown",
        ]

        for blocker_type in blocker_types:
            rejection = ObservationRejection(
                node_id="sensor-1",
                target_id="target-1",
                timestamp_s=0.0,
                reason="los_blocked",
                blocker_type=blocker_type,
            )

            self.assertEqual(rejection.blocker_type, blocker_type)


class RejectionViewerPlacementTest(unittest.TestCase):
    """Test rejection markers for viewer visualization."""

    def test_rejection_origin_for_sensor_position(self) -> None:
        """Test using rejection origin as sensor marker position."""
        sensor_pos = vec3(100.0, 100.0, 50.0)
        target_pos = vec3(300.0, 300.0, 50.0)

        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            origin=sensor_pos,
            attempted_point=target_pos,
            blocker_type="building",
        )

        # Viewer can use origin for sensor marker
        self.assertIsNotNone(rejection.origin)

    def test_rejection_closest_point_for_blocker_marker(self) -> None:
        """Test using rejection closest_point as blocker marker position."""
        blocker_pos = vec3(200.0, 200.0, 60.0)

        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            closest_point=blocker_pos,
            blocker_type="building",
        )

        # Viewer can use closest_point for blocker marker
        self.assertIsNotNone(rejection.closest_point)

    def test_rejection_geometry_describes_los_path(self) -> None:
        """Test rejection geometry describes the blocked ray."""
        origin = vec3(0.0, 0.0, 50.0)
        attempted = vec3(300.0, 300.0, 50.0)
        closest = vec3(150.0, 150.0, 80.0)
        range_m = np.sqrt(150.0**2 + 150.0**2)

        rejection = ObservationRejection(
            node_id="sensor-1",
            target_id="target-1",
            timestamp_s=0.0,
            reason="los_blocked",
            origin=origin,
            attempted_point=attempted,
            closest_point=closest,
            first_hit_range_m=range_m,
            blocker_type="building",
        )

        # Viewer can draw line from origin through closest_point to attempted
        self.assertIsNotNone(rejection.origin)
        self.assertIsNotNone(rejection.closest_point)
        self.assertIsNotNone(rejection.attempted_point)


class RejectionReasonCategoriesTest(unittest.TestCase):
    """Test standard rejection reason categories."""

    def test_prefilter_rejection_reasons(self) -> None:
        """Test pre-filter rejection reasons."""
        reasons = [
            "unknown_node",
            "low_confidence",
            "bearing_noise_too_high",
            "duplicate_node_observation",
        ]

        for reason in reasons:
            rejection = ObservationRejection(
                node_id="sensor-1",
                target_id="target-1",
                timestamp_s=0.0,
                reason=reason,
            )

            self.assertEqual(rejection.reason, reason)

    def test_los_coverage_rejection_reasons(self) -> None:
        """Test LOS and coverage rejection reasons."""
        reasons = [
            "los_blocked",
            "terrain_blocked",
            "outside_fov",
            "out_of_range",
        ]

        for reason in reasons:
            rejection = ObservationRejection(
                node_id="sensor-1",
                target_id="target-1",
                timestamp_s=0.0,
                reason=reason,
                blocker_type="terrain" if "terrain" in reason else "building",
            )

            self.assertEqual(rejection.reason, reason)


if __name__ == "__main__":
    unittest.main()
