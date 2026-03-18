from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from smart_tracker import TrackerConfig, TrackingService
from smart_tracker.models import BearingObservation, NodeState, TruthState


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "runtime_parity_fixture.json"


def _vector(value: list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _node_from_json(value: dict[str, Any]) -> NodeState:
    return NodeState(
        node_id=value["node_id"],
        position=_vector(value["position"]),
        velocity=_vector(value["velocity"]),
        is_mobile=bool(value["is_mobile"]),
        timestamp_s=float(value["timestamp_s"]),
        health=float(value["health"]),
    )


def _observation_from_json(value: dict[str, Any]) -> BearingObservation:
    return BearingObservation(
        node_id=value["node_id"],
        target_id=value["target_id"],
        origin=_vector(value["origin"]),
        direction=_vector(value["direction"]),
        bearing_std_rad=float(value["bearing_std_rad"]),
        timestamp_s=float(value["timestamp_s"]),
        confidence=float(value["confidence"]),
    )


def _truth_from_json(value: dict[str, Any]) -> TruthState:
    return TruthState(
        target_id=value["target_id"],
        position=_vector(value["position"]),
        velocity=_vector(value["velocity"]),
        timestamp_s=float(value["timestamp_s"]),
    )


def _assert_json_close(test_case: unittest.TestCase, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        test_case.assertAlmostEqual(expected, float(actual), delta=5.0e-4)
        return
    if isinstance(expected, list):
        if expected and isinstance(expected[0], (int, float)) and actual and isinstance(actual[0], list):
            actual = [value for row in actual for value in row]
        if actual and isinstance(actual[0], (int, float)) and expected and isinstance(expected[0], list):
            expected = [value for row in expected for value in row]
        test_case.assertEqual(len(expected), len(actual))
        for actual_item, expected_item in zip(actual, expected):
            _assert_json_close(test_case, actual_item, expected_item)
        return
    if isinstance(expected, dict):
        test_case.assertEqual(set(expected), set(actual))
        for key, expected_value in expected.items():
            _assert_json_close(test_case, actual[key], expected_value)
        return
    test_case.assertEqual(expected, actual)


class RustRuntimeParityTest(unittest.TestCase):
    def test_tracking_service_matches_frozen_python_fixture(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        config = TrackerConfig(**fixture["tracker_config"])
        service = TrackingService(config=config, retain_history=True)
        self.addCleanup(service.close)

        for request, expected_response in zip(fixture["requests"], fixture["responses"]):
            frame = service.ingest_frame(
                request["timestamp_s"],
                node_states=[_node_from_json(node) for node in request["node_states"]],
                observations=[_observation_from_json(observation) for observation in request["observations"]],
                truths=[_truth_from_json(truth) for truth in request["truths"]],
            )

            _assert_json_close(self, frame.timestamp_s, expected_response["timestamp_s"])
            _assert_json_close(self, len(frame.nodes), len(expected_response["nodes"]))
            _assert_json_close(self, len(frame.tracks), len(expected_response["tracks"]))
            _assert_json_close(
                self,
                {track.track_id: track.stale_steps for track in frame.tracks},
                {track["track_id"]: track["stale_steps"] for track in expected_response["tracks"]},
            )
            _assert_json_close(
                self,
                frame.metrics.rejection_counts,
                expected_response["metrics"]["rejection_counts"],
            )
            _assert_json_close(
                self,
                frame.metrics.track_errors_m,
                expected_response["metrics"]["track_errors_m"],
            )
            _assert_json_close(
                self,
                [track.position.tolist() for track in frame.tracks],
                [track["position"] for track in expected_response["tracks"]],
            )
            _assert_json_close(
                self,
                [track.velocity.tolist() for track in frame.tracks],
                [track["velocity"] for track in expected_response["tracks"]],
            )
            _assert_json_close(
                self,
                [track.covariance.tolist() for track in frame.tracks],
                [track["covariance"] for track in expected_response["tracks"]],
            )
            self.assertIs(frame, service.latest_frame())

        self.assertEqual(len(fixture["responses"]), len(service.history))

    def test_tracking_service_spawns_and_closes_local_daemon(self) -> None:
        service = TrackingService()
        process = service._owned_process

        self.assertIsNotNone(process)
        self.assertIsNone(process.poll())

        service.close()

        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
