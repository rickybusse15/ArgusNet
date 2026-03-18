"""Tests for the replay document JSON Schema and schema-based validation."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from smart_tracker.replay import _SCHEMA_PATH, validate_replay_with_schema


def _minimal_valid_document() -> dict:
    """Return a minimal replay document that satisfies the schema."""
    return {
        "meta": {
            "scenario_name": "unit-test",
            "generated_at_utc": "2025-01-15T12:00:00+00:00",
            "frame_count": 1,
            "dt_s": 0.5,
            "seed": 42,
            "node_ids": ["node-a"],
            "track_ids": ["track-1"],
        },
        "summary": {
            "mean_error_m": 1.5,
            "peak_error_m": 3.0,
            "mean_observations_per_frame": 2.0,
            "mean_active_tracks": 1.0,
            "mean_accepted_observations_per_frame": 2.0,
            "mean_rejected_observations_per_frame": 0.0,
            "total_accepted_observations": 2,
            "total_rejected_observations": 0,
            "observation_rejection_rate": 0.0,
            "rejection_counts": {},
        },
        "frames": [
            {
                "timestamp_s": 0.0,
                "nodes": [
                    {
                        "node_id": "node-a",
                        "position": [0.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "is_mobile": False,
                        "timestamp_s": 0.0,
                        "health": 1.0,
                        "sensor_type": "optical",
                        "fov_half_angle_deg": 180.0,
                        "max_range_m": 0.0,
                    }
                ],
                "observations": [
                    {
                        "node_id": "node-a",
                        "target_id": "target-1",
                        "origin": [0.0, 0.0, 0.0],
                        "direction": [1.0, 0.0, 0.0],
                        "bearing_std_rad": 0.002,
                        "timestamp_s": 0.0,
                        "confidence": 1.0,
                    }
                ],
                "rejected_observations": [],
                "tracks": [
                    {
                        "track_id": "track-1",
                        "timestamp_s": 0.0,
                        "position": [50.0, 15.0, 10.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "covariance": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "measurement_std_m": 5.0,
                        "update_count": 1,
                        "stale_steps": 0,
                    }
                ],
                "truths": [
                    {
                        "target_id": "target-1",
                        "position": [50.0, 15.0, 10.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "timestamp_s": 0.0,
                    }
                ],
                "metrics": {
                    "mean_error_m": 1.5,
                    "max_error_m": 1.5,
                    "active_track_count": 1,
                    "observation_count": 2,
                    "accepted_observation_count": 2,
                    "rejected_observation_count": 0,
                    "mean_measurement_std_m": 5.0,
                    "track_errors_m": {},
                    "rejection_counts": {},
                    "accepted_observations_by_target": {},
                    "rejected_observations_by_target": {},
                },
                "generation_rejections": [],
                "launch_events": [],
            }
        ],
    }


class TestSchemaFileValidity(unittest.TestCase):
    """Verify the schema file itself is well-formed JSON Schema."""

    def test_schema_file_exists(self) -> None:
        self.assertTrue(_SCHEMA_PATH.exists(), f"Schema file not found at {_SCHEMA_PATH}")

    def test_schema_is_valid_json(self) -> None:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertIsInstance(schema, dict)

    def test_schema_has_draft_declaration(self) -> None:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertIn("$schema", schema)
        self.assertIn("draft-07", schema["$schema"])

    def test_schema_declares_required_top_level_keys(self) -> None:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertIn("required", schema)
        self.assertIn("meta", schema["required"])
        self.assertIn("frames", schema["required"])

    def test_schema_has_definitions(self) -> None:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertIn("definitions", schema)
        expected_defs = {"vector3", "frame", "node_state", "bearing_observation", "track_state", "truth_state", "platform_metrics"}
        self.assertTrue(expected_defs.issubset(set(schema["definitions"].keys())))


class TestValidDocumentPassesValidation(unittest.TestCase):
    """A well-formed replay document should produce zero errors."""

    def test_minimal_valid_document(self) -> None:
        doc = _minimal_valid_document()
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Expected no errors but got: {errors}")

    def test_document_with_null_summary_errors(self) -> None:
        """Null-able summary fields (mean_error_m, peak_error_m) should be accepted."""
        doc = _minimal_valid_document()
        doc["summary"]["mean_error_m"] = None
        doc["summary"]["peak_error_m"] = None
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Expected no errors but got: {errors}")


class TestMissingRequiredFields(unittest.TestCase):
    """Missing required properties should produce validation errors."""

    def test_missing_meta(self) -> None:
        doc = _minimal_valid_document()
        del doc["meta"]
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors for missing meta")
        self.assertTrue(
            any("meta" in e and "required" in e for e in errors),
            f"Expected a 'required' error mentioning 'meta', got: {errors}",
        )

    def test_missing_frames(self) -> None:
        doc = _minimal_valid_document()
        del doc["frames"]
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors for missing frames")
        self.assertTrue(
            any("frames" in e and "required" in e for e in errors),
            f"Expected a 'required' error mentioning 'frames', got: {errors}",
        )

    def test_missing_dt_s_in_meta(self) -> None:
        doc = _minimal_valid_document()
        del doc["meta"]["dt_s"]
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors for missing dt_s")
        self.assertTrue(
            any("dt_s" in e and "required" in e for e in errors),
            f"Expected a 'required' error mentioning 'dt_s', got: {errors}",
        )

    def test_empty_frames_array(self) -> None:
        doc = _minimal_valid_document()
        doc["frames"] = []
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors for empty frames")


class TestWrongTypes(unittest.TestCase):
    """Wrong types for fields should produce validation errors."""

    def test_meta_not_object(self) -> None:
        doc = _minimal_valid_document()
        doc["meta"] = "not-an-object"
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when meta is a string")

    def test_frames_not_array(self) -> None:
        doc = _minimal_valid_document()
        doc["frames"] = "not-an-array"
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when frames is a string")

    def test_dt_s_is_string(self) -> None:
        doc = _minimal_valid_document()
        doc["meta"]["dt_s"] = "half"
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when dt_s is a string")

    def test_dt_s_zero_fails(self) -> None:
        doc = _minimal_valid_document()
        doc["meta"]["dt_s"] = 0
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when dt_s is 0")

    def test_dt_s_negative_fails(self) -> None:
        doc = _minimal_valid_document()
        doc["meta"]["dt_s"] = -1.0
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when dt_s is negative")

    def test_frame_timestamp_wrong_type(self) -> None:
        doc = _minimal_valid_document()
        doc["frames"][0]["timestamp_s"] = "zero"
        errors = validate_replay_with_schema(doc)
        self.assertTrue(len(errors) > 0, "Expected errors when timestamp_s is a string")


class TestExtraFieldsAllowed(unittest.TestCase):
    """Extra/unknown fields should be allowed for forward compatibility."""

    def test_extra_top_level_field(self) -> None:
        doc = _minimal_valid_document()
        doc["custom_extension"] = {"version": "1.0"}
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Extra top-level fields should be allowed, got: {errors}")

    def test_extra_meta_field(self) -> None:
        doc = _minimal_valid_document()
        doc["meta"]["experiment_id"] = "exp-42"
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Extra meta fields should be allowed, got: {errors}")

    def test_extra_frame_field(self) -> None:
        doc = _minimal_valid_document()
        doc["frames"][0]["custom_data"] = [1, 2, 3]
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Extra frame fields should be allowed, got: {errors}")

    def test_extra_track_field(self) -> None:
        doc = _minimal_valid_document()
        doc["frames"][0]["tracks"][0]["lifecycle_state"] = "confirmed"
        errors = validate_replay_with_schema(doc)
        self.assertEqual(errors, [], f"Extra track fields should be allowed, got: {errors}")


class TestFallbackValidation(unittest.TestCase):
    """Test the manual fallback path directly, regardless of jsonschema availability."""

    def test_manual_fallback_catches_missing_meta(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        del doc["meta"]
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("meta" in e for e in errors))

    def test_manual_fallback_catches_missing_frames(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        del doc["frames"]
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("frames" in e for e in errors))

    def test_manual_fallback_catches_empty_frames(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["frames"] = []
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)

    def test_manual_fallback_catches_wrong_meta_type(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["meta"] = "bad"
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)

    def test_manual_fallback_catches_wrong_dt_s_type(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["meta"]["dt_s"] = "bad"
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)

    def test_manual_fallback_catches_negative_dt_s(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["meta"]["dt_s"] = -1.0
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)

    def test_manual_fallback_passes_valid_document(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        errors = _manual_schema_validation(doc, schema)
        self.assertEqual(errors, [])

    def test_manual_fallback_allows_extra_fields(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["custom_extension"] = True
        doc["meta"]["extra_key"] = "fine"
        errors = _manual_schema_validation(doc, schema)
        self.assertEqual(errors, [])

    def test_manual_fallback_catches_missing_dt_s(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        del doc["meta"]["dt_s"]
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("dt_s" in e for e in errors))

    def test_manual_fallback_catches_frame_missing_required_fields(self) -> None:
        from smart_tracker.replay import _load_replay_schema, _manual_schema_validation
        schema = _load_replay_schema()
        doc = _minimal_valid_document()
        doc["frames"] = [{"timestamp_s": 0.0}]  # missing most required frame fields
        errors = _manual_schema_validation(doc, schema)
        self.assertTrue(len(errors) > 0)


if __name__ == "__main__":
    unittest.main()
