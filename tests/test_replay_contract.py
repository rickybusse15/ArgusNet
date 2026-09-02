"""Writer half of the cross-language replay contract test.

Replay JSON is the interface between the Python simulation and the Rust viewer,
and nothing previously checked that the two agreed. They did not — see
``tests/fixtures/README-replay-contract.md``.

This module asserts that what the simulation writes is (a) valid against the
formal schema and (b) still carries every field the viewer reads. The reader half
lives in ``rust/argusnet-viewer/src/replay.rs`` (``contract_fixture_tests``) and
deserializes the same fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argusnet.evaluation.replay import validate_replay_with_schema

FIXTURE = Path(__file__).parent / "fixtures" / "replay_contract_fixture.json"

#: Fields on each replay node that `rust/argusnet-viewer/src/replay.rs::NodeState`
#: reads. Dropping one silently blanks part of the operator UI.
REQUIRED_NODE_FIELDS = (
    "node_id",
    "position",
    "velocity",
    "is_mobile",
    "health",
    "sensor_type",
    "battery_fraction",
)

#: Fields on each replay track that the viewer's track tables and overlays read.
REQUIRED_TRACK_FIELDS = (
    "track_id",
    "position",
    "velocity",
    "covariance",
    "lifecycle_state",
    "stale_steps",
    "update_count",
    "quality_score",
)

REQUIRED_FRAME_FIELDS = (
    "timestamp_s",
    "nodes",
    "observations",
    "rejected_observations",
    "tracks",
    "truths",
    "metrics",
)


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_valid_against_the_formal_schema(document: dict) -> None:
    """The strict `jsonschema` path, not the permissive manual fallback.

    ``jsonschema`` is a declared dependency precisely so this runs for real; the
    fallback validator does not check nested vector shapes at all.
    """
    pytest.importorskip("jsonschema")
    assert validate_replay_with_schema(document) == []


def test_fixture_carries_frames_with_tracks(document: dict) -> None:
    """Guards the fixture itself: an empty fixture would pass everything below."""
    frames = document["frames"]
    assert frames, "fixture must contain frames"
    assert any(frame["tracks"] for frame in frames), "fixture must exercise tracks"
    assert any(frame["nodes"] for frame in frames), "fixture must exercise nodes"


@pytest.mark.parametrize("field", REQUIRED_FRAME_FIELDS)
def test_every_frame_carries_the_required_fields(document: dict, field: str) -> None:
    for index, frame in enumerate(document["frames"]):
        assert field in frame, f"frame {index} is missing {field!r}"


@pytest.mark.parametrize("field", REQUIRED_NODE_FIELDS)
def test_every_node_carries_the_fields_the_viewer_reads(document: dict, field: str) -> None:
    seen = 0
    for frame in document["frames"]:
        for node in frame["nodes"]:
            assert field in node, f"node {node.get('node_id')!r} is missing {field!r}"
            seen += 1
    assert seen, "no nodes in fixture"


@pytest.mark.parametrize("field", REQUIRED_TRACK_FIELDS)
def test_every_track_carries_the_fields_the_viewer_reads(document: dict, field: str) -> None:
    seen = 0
    for frame in document["frames"]:
        for track in frame["tracks"]:
            assert field in track, f"track {track.get('track_id')!r} is missing {field!r}"
            seen += 1
    assert seen, "no tracks in fixture"


def test_mobile_nodes_report_a_battery_fraction(document: dict) -> None:
    """`docs/VIEWER_UI.md` documents a per-drone battery readout.

    It was undisplayable for as long as the viewer had no field for this, which
    is exactly the kind of drift this module exists to catch.
    """
    mobile = [
        node for frame in document["frames"] for node in frame["nodes"] if node.get("is_mobile")
    ]
    assert mobile, "fixture must contain mobile nodes"
    for node in mobile:
        fraction = node["battery_fraction"]
        assert isinstance(fraction, (int, float))
        assert 0.0 <= float(fraction) <= 1.0


def test_schema_version_is_recorded(document: dict) -> None:
    """`meta.schema_version` is how a future reader will detect an old replay."""
    assert document["meta"]["schema_version"]


def test_track_conversion_carries_imm_mode_and_contributing_nodes() -> None:
    """The gRPC response fields that drive two viewer overlays must survive.

    ``_track_from_proto`` previously dropped ``mode_probability_cv`` and
    ``contributing_nodes``, so the viewer's IMM readout and contributing-node
    overlay -- both of which declare the fields -- were permanently blank in
    replay mode while working over the live stream.
    """
    from argusnet.adapters.argusnet_grpc import _track_from_proto
    from argusnet.core.types import to_jsonable
    from argusnet.v1 import world_model_pb2 as pb

    message = pb.TrackState(
        track_id="t1",
        timestamp_s=1.0,
        covariance_row_major=[0.0] * 36,
        measurement_std_m=1.0,
        update_count=3,
        stale_steps=0,
        mode_probability_cv=0.73,
    )
    message.contributing_nodes.extend(["drone-0", "ground-2"])

    track = _track_from_proto(message)
    assert track.mode_probability_cv == pytest.approx(0.73)
    assert track.contributing_node_ids == ("drone-0", "ground-2")

    serialized = to_jsonable(track)
    assert serialized["mode_probability_cv"] == pytest.approx(0.73)
    assert serialized["contributing_node_ids"] == ["drone-0", "ground-2"]
