"""G0 (E1.2): every string the Python producers write into a replay's closed
vocabularies is one docs/replay-schema.json accepts. The Rust producers have
the same check in rust/argusnet-sim/tests/replay_vocabulary.rs; the table of
vocabularies is docs/migration/CONTRACT_VOCABULARIES.md."""

import json
from pathlib import Path

import pytest

from argusnet.adapters import argusnet_grpc
from argusnet.core.config import SensorConfig
from argusnet.core.types import (
    SENSOR_TYPES,
    BeliefCellStatus,
    LocalizationStatus,
    MissionEventType,
)
from argusnet.localization import state
from argusnet.simulation import sim

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "docs/replay-schema.json").read_text())


def schema_enum(*path: str) -> set:
    node = SCHEMA["definitions"]
    for key in path:
        node = node[key]
    return set(node["enum"])


def constants(module, prefix: str) -> set[str]:
    return {
        value
        for name, value in vars(module).items()
        if name.startswith(prefix) and isinstance(value, str)
    }


@pytest.mark.parametrize(
    ("what", "written", "path"),
    [
        (
            "simulation rejection reasons",
            constants(sim, "REJECT_"),
            ("observation_rejection", "properties", "reason"),
        ),
        (
            "fusion rejection reasons",
            constants(argusnet_grpc, "REJECT_"),
            ("observation_rejection", "properties", "reason"),
        ),
        (
            "track lifecycle states",
            constants(state, "TRACK_STATE_"),
            ("track_state", "properties", "lifecycle_state"),
        ),
        (
            "belief cell statuses",
            {status.value for status in BeliefCellStatus},
            ("belief_cell", "properties", "status"),
        ),
        (
            "pose estimate statuses",
            {status.value for status in LocalizationStatus},
            ("pose_estimate", "properties", "status"),
        ),
        (
            "mission event types",
            {event.value for event in MissionEventType},
            ("mission_event", "properties", "event_type"),
        ),
    ],
)
def test_python_producers_stay_within_the_schema(what, written, path):
    assert written, f"no {what} found: the check would pass vacuously"
    outside = written - schema_enum(*path)
    assert not outside, f"{what}: {sorted(outside)} are refused by the replay schema"


def test_the_sensor_vocabulary_is_exactly_the_schemas():
    assert set(SENSOR_TYPES) == schema_enum("node_state", "properties", "sensor_type")


@pytest.mark.parametrize("sensor_type", ["sonar", "", "OPTICAL"])
def test_a_config_naming_an_unknown_sensor_type_is_refused(sensor_type):
    with pytest.raises(ValueError, match="sensor type"):
        SensorConfig(drone_sensor_type=sensor_type)
    with pytest.raises(ValueError, match="sensor type"):
        SensorConfig(ground_station_sensor_type=sensor_type)


@pytest.mark.parametrize("sensor_type", SENSOR_TYPES)
def test_every_known_sensor_type_configures(sensor_type):
    assert SensorConfig(drone_sensor_type=sensor_type).drone_sensor_type == sensor_type
