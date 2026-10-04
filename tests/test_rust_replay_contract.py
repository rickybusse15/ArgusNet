"""The cross-language replay contract.

`docs/replay-schema.json` is what CLAUDE.md calls "the interface between
simulation and viewer". Until now only the *Python* writer answered to it, so
a Rust-written replay could drift arbitrarily and nothing would notice until a
viewer failed to load one.

`tests/fixtures/rust_replay_fixture.json` is emitted by `argusnet-sim`'s
`replay` module and byte-compared against that module by
`the_committed_fixture_is_what_this_writer_emits` in Rust. This file closes
the loop from the other side: the same bytes are validated here with
`jsonschema`, through the same `validate_replay_with_schema` the Python writer
answers to, and then loaded through the ordinary reader.

Neither language can silently rebaseline the contract: changing the Rust
writer without regenerating the fixture fails in Rust CI, and regenerating it
into a shape the schema rejects fails here.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "rust_replay_fixture.json"

# Mirrors `argusnet_sim::replay::BATTERY_NOT_APPLICABLE` and
# `argusnet.adapters.argusnet_grpc.BATTERY_NOT_APPLICABLE`.
BATTERY_NOT_APPLICABLE = -1.0


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class RustReplayContractTest(unittest.TestCase):
    def test_the_fixture_exists(self) -> None:
        """A missing fixture is a failure, not a skip.

        Skipping would turn the whole cross-language contract into a no-op the
        moment someone moved the file, which is the failure mode this test
        exists to prevent.
        """
        self.assertTrue(
            FIXTURE.is_file(),
            f"{FIXTURE} is missing. Regenerate with "
            "ARGUSNET_BLESS_REPLAY_FIXTURE=1 cargo test -p argusnet-sim",
        )

    def test_jsonschema_is_installed_so_this_is_a_strict_check(self) -> None:
        """`validate_replay_with_schema` falls back to a permissive manual
        validator when `jsonschema` is absent, and the fallback checks only
        required keys and a few types. A strict check that silently degraded
        to a loose one would be worse than no check, so assert the dependency
        rather than trusting it.
        """
        import jsonschema  # noqa: F401

    def test_the_rust_writers_output_validates_against_the_schema(self) -> None:
        from argusnet.evaluation.replay import validate_replay_with_schema

        errors = validate_replay_with_schema(_load())
        self.assertEqual(errors, [], f"Rust replay fails the schema: {errors}")

    def test_the_document_loads_through_the_ordinary_reader(self) -> None:
        """Schema-valid is necessary, not sufficient: `validate_replay_document`
        is what every consumer actually calls.
        """
        from argusnet.evaluation.replay import validate_replay_document

        validate_replay_document(_load())

    def test_the_writer_declares_its_schema_version_and_producer(self) -> None:
        meta = _load()["meta"]
        self.assertEqual(meta["schema_version"], "2.0")
        self.assertEqual(
            meta["producer"],
            "argusnet-sim",
            "a Rust-written replay must say so, so a parity investigation "
            "starts from fact rather than from a filename",
        )

    def test_the_schema_version_matches_the_python_writers(self) -> None:
        """Two producers, one schema, one version string.

        If these diverge, a consumer branching on the version takes different
        paths for documents that are identical in shape.
        """
        from argusnet.evaluation.replay import REPLAY_SCHEMA_VERSION

        source = (
            Path(__file__).parent.parent / "src" / "argusnet" / "simulation" / "sim.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"schema_version": REPLAY_SCHEMA_VERSION',
            source,
            "sim.py no longer writes REPLAY_SCHEMA_VERSION",
        )
        self.assertEqual(
            _load()["meta"]["schema_version"],
            REPLAY_SCHEMA_VERSION,
            "the Rust writer's SCHEMA_VERSION must move with the Python one",
        )

    def test_a_platform_with_no_battery_carries_the_sentinel(self) -> None:
        """The distinction the whole `battery_fraction` design exists for.

        `-1.0` is "this platform has no battery"; `0.0` is "this battery is
        flat". The fixture contains one of each so a writer that collapsed
        them would fail here rather than render a ground station as a drone
        about to come down.
        """
        nodes = _load()["frames"][0]["nodes"]
        by_id = {node["node_id"]: node for node in nodes}
        self.assertEqual(by_id["tower"]["battery_fraction"], BATTERY_NOT_APPLICABLE)
        self.assertEqual(by_id["node-a"]["battery_fraction"], 1.0)

    def test_every_node_carries_the_four_fields_added_by_the_widening(self) -> None:
        for frame in _load()["frames"]:
            for node in frame["nodes"]:
                for field in (
                    "sensor_type",
                    "fov_half_angle_deg",
                    "max_range_m",
                    "battery_fraction",
                ):
                    self.assertIn(field, node, f"node {node['node_id']} dropped {field}")

    def test_covariance_reshapes_to_six_by_six(self) -> None:
        """`evaluation/replay.py` reshapes unconditionally, so a covariance of
        any other length crashes the Python reader rather than degrading.
        """
        import numpy as np

        for frame in _load()["frames"]:
            for track in frame["tracks"]:
                flat = np.asarray(track["covariance"], dtype=float)
                self.assertEqual(flat.size, 36)
                self.assertEqual(flat.reshape(6, 6).shape, (6, 6))

    def test_the_covariance_arrived_row_major(self) -> None:
        """The fixture's matrix is `row * 6 + column`, which is asymmetric, so
        a transposed document reads back differently. A symmetric fixture
        could not tell the two orders apart.
        """
        import numpy as np

        covariance = np.asarray(
            _load()["frames"][0]["tracks"][0]["covariance"], dtype=float
        ).reshape(6, 6)
        expected = np.arange(36, dtype=float).reshape(6, 6)
        np.testing.assert_array_equal(covariance, expected)
        self.assertFalse(
            np.array_equal(covariance, expected.T),
            "fixture is symmetric -- this test cannot detect a transpose",
        )

    def test_every_sensor_type_the_runtime_models_is_in_the_schema(self) -> None:
        """Decision D-14 closed the gap this test used to pin.

        `SensorBand::from_sensor_type` in `argusnet-sim`'s `observe` module
        classifies ``"thermal"``, ``"lwir"``, ``"mwir"`` and ``"ir"`` into the
        thermal band; the schema once allowed only ``"optical"`` and
        ``"radar"``, so a thermal platform wrote a replay that failed strict
        validation. Now the vocabulary is one list
        (`argusnet.core.types.SENSOR_TYPES`, `argusnet_types::SENSOR_TYPES`),
        the schema accepts every member, configs and bundles refuse anything
        else, and a name outside it is still refused here.
        """
        from argusnet.core.types import SENSOR_TYPES
        from argusnet.evaluation.replay import validate_replay_with_schema

        for sensor_type in SENSOR_TYPES:
            document = _load()
            document["frames"][0]["nodes"][0]["sensor_type"] = sensor_type
            self.assertEqual(validate_replay_with_schema(document), [], sensor_type)
        document = _load()
        document["frames"][0]["nodes"][0]["sensor_type"] = "sonar"
        self.assertTrue(
            any("sensor_type" in error for error in validate_replay_with_schema(document)),
            "a sensor type outside the vocabulary must still be refused",
        )

    def test_a_tampered_document_is_rejected(self) -> None:
        """The validator is actually rejecting things.

        A schema check that passes everything is indistinguishable from no
        check at all, and this suite would look just as green.
        """
        from argusnet.evaluation.replay import validate_replay_with_schema

        document = _load()
        document["frames"][0]["nodes"][0]["battery_fraction"] = -0.5
        self.assertTrue(
            validate_replay_with_schema(document),
            "-0.5 is neither a reading nor the sentinel and must be rejected",
        )

        document = _load()
        del document["frames"][0]["metrics"]
        self.assertTrue(
            validate_replay_with_schema(document),
            "a frame without metrics must be rejected",
        )

        document = _load()
        document["frames"][0]["nodes"][0]["position"] = [1.0, 2.0]
        self.assertTrue(
            validate_replay_with_schema(document),
            "a two-element position must be rejected",
        )


if __name__ == "__main__":
    unittest.main()
