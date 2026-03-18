from __future__ import annotations

import json
import os
import tempfile
import unittest

from smart_tracker.models import BearingObservation, NodeState, TruthState, vec3
from smart_tracker.replay import build_replay_document, load_replay_document, validate_replay_document
from smart_tracker.service import TrackerConfig, TrackingService


class ReplayDocumentTest(unittest.TestCase):
    def test_replay_document_serializes_frames(self) -> None:
        platform = TrackingService(config=TrackerConfig())
        timestamp_s = 0.0
        node_a = NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s)
        node_b = NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s)
        truth = TruthState("asset-a", vec3(50.0, 15.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s)
        observations = [
            BearingObservation("ground-a", "asset-a", node_a.position, truth.position - node_a.position, 0.002, timestamp_s, 1.0),
            BearingObservation("ground-b", "asset-a", node_b.position, truth.position - node_b.position, 0.002, timestamp_s, 1.0),
        ]
        frame = platform.ingest_frame(timestamp_s, node_states=[node_a, node_b], observations=observations, truths=[truth])

        document = build_replay_document(
            [frame],
            scenario_name="test-range",
            dt_s=0.5,
            seed=4,
            extra_meta={"schema_version": "2.0"},
        )

        self.assertEqual(1, document["meta"]["frame_count"])
        self.assertEqual(["asset-a"], document["meta"]["track_ids"])
        self.assertEqual("2.0", document["meta"]["schema_version"])
        self.assertIsInstance(document["frames"][0]["tracks"][0]["position"], list)
        self.assertIn("rejected_observations", document["frames"][0])
        self.assertIn("generation_rejections", document["frames"][0])
        self.assertIn("observation_rejection_rate", document["summary"])

    def test_validate_replay_document_rejects_missing_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty frames list"):
            validate_replay_document({"meta": {"dt_s": 0.5}, "frames": []})

    def test_validate_replay_document_rejects_invalid_dt(self) -> None:
        with self.assertRaisesRegex(ValueError, "meta.dt_s"):
            validate_replay_document({"meta": {"dt_s": 0.0}, "frames": [{}]})

    def test_validate_replay_document_rejects_missing_required_frame_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "nodes"):
            validate_replay_document(
                {
                    "meta": {
                        "scenario_name": "invalid",
                        "generated_at_utc": "2025-01-01T00:00:00+00:00",
                        "frame_count": 1,
                        "dt_s": 0.5,
                        "seed": 7,
                        "node_ids": [],
                        "track_ids": [],
                    },
                    "frames": [{"timestamp_s": 0.0}],
                }
            )

    def test_load_replay_document_rejects_missing_required_frame_fields(self) -> None:
        invalid_document = {
            "meta": {
                "scenario_name": "invalid",
                "generated_at_utc": "2025-01-01T00:00:00+00:00",
                "frame_count": 1,
                "dt_s": 0.5,
                "seed": 7,
                "node_ids": [],
                "track_ids": [],
            },
            "frames": [{"timestamp_s": 0.0}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(invalid_document, handle)
            handle.flush()
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "nodes"):
                load_replay_document(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
