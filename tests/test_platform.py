from __future__ import annotations

import unittest

import numpy as np

from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService
from argusnet.core.types import BearingObservation, NodeState, TruthState, vec3


def bearing(
    node_id: str,
    target_id: str,
    origin: np.ndarray,
    target: np.ndarray,
    timestamp_s: float,
    confidence: float = 1.0,
    bearing_std_rad: float = 0.002,
) -> BearingObservation:
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


class SensorFusionPlatformTest(unittest.TestCase):
    def test_multi_track_frame_contains_expected_tracks(self) -> None:
        platform = TrackingService(config=TrackerConfig())
        timestamp_s = 0.0
        nodes = [
            NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("drone-c", vec3(40.0, 80.0, 40.0), vec3(0.0, 0.0, 0.0), True, timestamp_s),
        ]
        truths = [
            TruthState("asset-a", vec3(42.0, 35.0, 20.0), vec3(0.0, 0.0, 0.0), timestamp_s),
            TruthState("asset-b", vec3(58.0, 22.0, 12.0), vec3(0.0, 0.0, 0.0), timestamp_s),
        ]
        observations = [
            bearing("ground-a", "asset-a", nodes[0].position, truths[0].position, timestamp_s),
            bearing("ground-b", "asset-a", nodes[1].position, truths[0].position, timestamp_s),
            bearing("drone-c", "asset-a", nodes[2].position, truths[0].position, timestamp_s),
            bearing("ground-a", "asset-b", nodes[0].position, truths[1].position, timestamp_s),
            bearing("ground-b", "asset-b", nodes[1].position, truths[1].position, timestamp_s),
            bearing("drone-c", "asset-b", nodes[2].position, truths[1].position, timestamp_s),
        ]

        frame = platform.ingest_frame(
            timestamp_s, node_states=nodes, observations=observations, truths=truths
        )

        self.assertEqual({"asset-a", "asset-b"}, {track.track_id for track in frame.tracks})
        self.assertIsNotNone(frame.metrics.mean_error_m)
        self.assertLess(frame.metrics.mean_error_m, 0.5)

    def test_track_persists_through_short_dropout(self) -> None:
        platform = TrackingService(config=TrackerConfig(max_stale_steps=2))
        nodes = [
            NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, 0.0),
            NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, 0.0),
        ]
        truth = TruthState("asset-a", vec3(42.0, 18.0, 12.0), vec3(3.0, 0.0, 0.0), 0.0)
        observations = [
            bearing("ground-a", "asset-a", nodes[0].position, truth.position, 0.0),
            bearing("ground-b", "asset-a", nodes[1].position, truth.position, 0.0),
        ]
        platform.ingest_frame(0.0, node_states=nodes, observations=observations, truths=[truth])

        next_truth = TruthState("asset-a", vec3(45.0, 18.0, 12.0), vec3(3.0, 0.0, 0.0), 1.0)
        frame = platform.ingest_frame(1.0, observations=[], truths=[next_truth])

        self.assertEqual(1, len(frame.tracks))
        self.assertEqual(1, frame.tracks[0].stale_steps)

    def test_rejection_diagnostics_capture_prefilter_failures(self) -> None:
        platform = TrackingService(config=TrackerConfig(min_confidence=0.3))
        timestamp_s = 0.0
        nodes = [
            NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truth = TruthState("asset-a", vec3(52.0, 16.0, 15.0), vec3(0.0, 0.0, 0.0), timestamp_s)

        observations = [
            bearing("ground-a", "asset-a", nodes[0].position, truth.position, timestamp_s),
            bearing("ground-b", "asset-a", nodes[1].position, truth.position, timestamp_s),
            bearing("ghost-node", "asset-a", nodes[0].position, truth.position, timestamp_s),
            bearing(
                "ground-a",
                "asset-a",
                nodes[0].position,
                truth.position,
                timestamp_s,
                confidence=0.1,
            ),
            bearing(
                "ground-b",
                "asset-a",
                nodes[1].position,
                truth.position,
                timestamp_s,
                bearing_std_rad=0.2,
            ),
        ]

        frame = platform.ingest_frame(
            timestamp_s, node_states=nodes, observations=observations, truths=[truth]
        )

        self.assertEqual(1, len(frame.tracks))
        self.assertEqual(2, frame.metrics.accepted_observation_count)
        self.assertEqual(3, frame.metrics.rejected_observation_count)
        self.assertIn("unknown_node", frame.metrics.rejection_counts)
        self.assertIn("low_confidence", frame.metrics.rejection_counts)
        self.assertIn("bearing_noise_too_high", frame.metrics.rejection_counts)

    def test_duplicate_observations_are_rejected_per_node(self) -> None:
        platform = TrackingService(config=TrackerConfig())
        timestamp_s = 0.0
        nodes = [
            NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]
        truth = TruthState("asset-a", vec3(51.0, 20.0, 16.0), vec3(0.0, 0.0, 0.0), timestamp_s)

        observations = [
            bearing(
                "ground-a",
                "asset-a",
                nodes[0].position,
                truth.position,
                timestamp_s,
                confidence=0.7,
            ),
            bearing(
                "ground-a",
                "asset-a",
                nodes[0].position,
                truth.position,
                timestamp_s,
                confidence=0.95,
            ),
            bearing("ground-b", "asset-a", nodes[1].position, truth.position, timestamp_s),
        ]
        frame = platform.ingest_frame(
            timestamp_s, node_states=nodes, observations=observations, truths=[truth]
        )

        self.assertEqual(1, len(frame.tracks))
        self.assertEqual(2, frame.metrics.accepted_observation_count)
        self.assertEqual(1, frame.metrics.rejected_observation_count)
        self.assertEqual(1, frame.metrics.rejection_counts.get("duplicate_node_observation"))


if __name__ == "__main__":
    unittest.main()
