"""Python-side proto round-trip tests for gRPC adapter conversion functions.

Each test serialises a Python domain object to proto and then deserialises it back,
asserting that every field survives the round-trip unchanged.

Known limitation (Bug #3): fields in TrackerConfig that use zero-detection rather than
proto3 `optional` cannot distinguish "explicitly set to 0.0" from "not set".  A client
that wants a zero-value noise parameter will have it silently replaced with the server
default.  This is documented here rather than tested, because 0.0 is also invalid for
all affected fields (enforced by TrackerConfig.__post_init__).
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.adapters.argusnet_grpc import (
    TrackerConfig,
    _node_from_proto,
    _node_to_proto,
    _observation_from_proto,
    _observation_to_proto,
    _tracker_config_from_proto,
    _tracker_config_to_proto,
)
from argusnet.core.types import BearingObservation, NodeState


class TrackerConfigRoundTripTest(unittest.TestCase):
    def test_default_config_round_trips_intact(self) -> None:
        original = TrackerConfig()
        recovered = _tracker_config_from_proto(_tracker_config_to_proto(original))
        self.assertEqual(original, recovered)

    def test_non_default_scalar_fields_survive(self) -> None:
        original = TrackerConfig(
            min_observations=5,
            max_stale_steps=12,
            retain_history=True,
            min_confidence=0.3,
            max_bearing_std_rad=0.05,
            max_timestamp_skew_s=2.0,
            min_intersection_angle_deg=5.0,
            cv_process_accel_std=1.5,
            ct_process_accel_std=6.0,
            ct_turn_rate_std=0.2,
            innovation_window=8,
            innovation_scale_factor=2.0,
            innovation_max_scale=6.0,
            adaptive_measurement_noise=True,
            chi_squared_gate_threshold=12.0,
            cluster_distance_threshold_m=150.0,
            near_parallel_rejection_angle_deg=3.0,
            confirmation_m=4,
            confirmation_n=7,
            max_coast_frames=15,
            max_coast_seconds=8.0,
            min_quality_score=0.2,
        )
        recovered = _tracker_config_from_proto(_tracker_config_to_proto(original))
        self.assertEqual(original, recovered)

    def test_data_association_mode_gnn_survives(self) -> None:
        original = TrackerConfig(data_association_mode="gnn")
        recovered = _tracker_config_from_proto(_tracker_config_to_proto(original))
        self.assertEqual(recovered.data_association_mode, "gnn")

    def test_data_association_mode_jpda_survives(self) -> None:
        original = TrackerConfig(data_association_mode="jpda")
        recovered = _tracker_config_from_proto(_tracker_config_to_proto(original))
        self.assertEqual(recovered.data_association_mode, "jpda")

    def test_boolean_retain_history_true_survives(self) -> None:
        original = TrackerConfig(retain_history=True)
        recovered = _tracker_config_from_proto(_tracker_config_to_proto(original))
        self.assertTrue(recovered.retain_history)


class BearingObservationRoundTripTest(unittest.TestCase):
    def _obs(
        self,
        *,
        node_id: str = "node-a",
        target_id: str = "target-1",
        confidence: float = 0.85,
    ) -> BearingObservation:
        return BearingObservation(
            node_id=node_id,
            target_id=target_id,
            origin=np.array([10.0, 20.0, 30.0], dtype=float),
            direction=np.array([0.0, 1.0, 0.0], dtype=float),
            bearing_std_rad=0.005,
            timestamp_s=1.23,
            confidence=confidence,
        )

    def test_basic_observation_round_trips_intact(self) -> None:
        original = self._obs()
        recovered = _observation_from_proto(_observation_to_proto(original))
        self.assertEqual(original.node_id, recovered.node_id)
        self.assertEqual(original.target_id, recovered.target_id)
        np.testing.assert_array_almost_equal(original.origin, recovered.origin)
        np.testing.assert_array_almost_equal(original.direction, recovered.direction)
        self.assertAlmostEqual(original.bearing_std_rad, recovered.bearing_std_rad)
        self.assertAlmostEqual(original.timestamp_s, recovered.timestamp_s)
        self.assertAlmostEqual(original.confidence, recovered.confidence)

    def test_multiple_observations_preserve_identity(self) -> None:
        obs_list = [self._obs(node_id=f"node-{i}", target_id=f"tgt-{i}") for i in range(5)]
        recovered_list = [_observation_from_proto(_observation_to_proto(o)) for o in obs_list]
        for orig, recov in zip(obs_list, recovered_list):
            self.assertEqual(orig.node_id, recov.node_id)
            self.assertEqual(orig.target_id, recov.target_id)


class NodeStateRoundTripTest(unittest.TestCase):
    def _node(self, node_id: str = "n-1", is_mobile: bool = True) -> NodeState:
        return NodeState(
            node_id=node_id,
            position=np.array([1.0, 2.0, 50.0], dtype=float),
            velocity=np.array([3.0, 0.0, 0.0], dtype=float),
            is_mobile=is_mobile,
            timestamp_s=5.0,
            health=0.9,
            sensor_type="optical",
            fov_half_angle_deg=45.0,
            max_range_m=600.0,
        )

    def test_mobile_node_round_trips_intact(self) -> None:
        original = self._node(is_mobile=True)
        recovered = _node_from_proto(_node_to_proto(original))
        self.assertEqual(original.node_id, recovered.node_id)
        self.assertTrue(recovered.is_mobile)
        np.testing.assert_array_almost_equal(original.position, recovered.position)
        np.testing.assert_array_almost_equal(original.velocity, recovered.velocity)
        self.assertAlmostEqual(original.health, recovered.health)
        self.assertEqual(original.sensor_type, recovered.sensor_type)
        self.assertAlmostEqual(original.fov_half_angle_deg, recovered.fov_half_angle_deg)
        self.assertAlmostEqual(original.max_range_m, recovered.max_range_m)

    def test_stationary_node_round_trips_intact(self) -> None:
        original = self._node(is_mobile=False)
        recovered = _node_from_proto(_node_to_proto(original))
        self.assertFalse(recovered.is_mobile)


if __name__ == "__main__":
    unittest.main()
