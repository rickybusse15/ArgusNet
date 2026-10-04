"""G0 (E1): every versioned payload a Python reader takes is refused, by name,
when it declares a version the reader was not written for. The table of every
versioned payload and its readers is docs/migration/CONTRACT_VERSIONS.md."""

import copy
import json
from pathlib import Path

import pytest

from argusnet.evaluation.replay import (
    REPLAY_SCHEMA_VERSION,
    SUPPORTED_REPLAY_SCHEMA_MAJOR,
    validate_replay_document,
)
from argusnet.world.environment import load_environment_bundle
from argusnet.world.environment_io import ENVIRONMENT_BUNDLE_FORMAT
from argusnet.world.scene_loader import SCENE_FORMAT_VERSION, validate_scene_manifest

ROOT = Path(__file__).resolve().parents[1]


def _replay() -> dict:
    return json.loads((ROOT / "tests/fixtures/replay_contract_fixture.json").read_text())


def test_the_written_replay_version_is_the_one_read():
    assert REPLAY_SCHEMA_VERSION.split(".")[0] == SUPPORTED_REPLAY_SCHEMA_MAJOR


@pytest.mark.parametrize("version", [None, "2.0", "2.7"])
def test_replays_of_the_supported_major_or_undeclared_are_read(version):
    document = _replay()
    if version is None:
        document["meta"].pop("schema_version", None)
    else:
        document["meta"]["schema_version"] = version
    validate_replay_document(document)


@pytest.mark.parametrize("version", ["3.0", "1.9", "", 2])
def test_replays_of_another_major_are_refused_by_name(version):
    document = copy.deepcopy(_replay())
    document["meta"]["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        validate_replay_document(document)


def test_a_scene_manifest_of_another_format_is_refused():
    manifest = json.loads((ROOT / "scene.smartscene/scene_manifest.json").read_text())
    assert manifest["format_version"] == SCENE_FORMAT_VERSION
    validate_scene_manifest(manifest)
    manifest["format_version"] = "smartscene-v2"
    with pytest.raises(ValueError, match="format_version"):
        validate_scene_manifest(manifest)


def test_an_environment_bundle_of_another_format_is_refused(tmp_path):
    (tmp_path / "environment_manifest.json").write_text(
        json.dumps({"format_version": "smartmap-v2"})
    )
    with pytest.raises(ValueError, match="smartmap-v2"):
        load_environment_bundle(str(tmp_path))
    assert ENVIRONMENT_BUNDLE_FORMAT == "smartmap-v1"
