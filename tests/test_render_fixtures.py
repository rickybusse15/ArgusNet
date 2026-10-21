"""The G2 fixture scenes (E2): the high-entity variant adds distinct tracks."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_viewer_smoke_scene", ROOT / "scripts/prepare_viewer_smoke_scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_entity_scene_adds_distinct_tracks(tmp_path):
    prepare = _prepare_module()
    base, load = tmp_path / "base", tmp_path / "load"
    prepare.prepare(base)
    prepare.prepare(load, 100)
    base_frames = json.loads((base / "replay/replay.json").read_text())["frames"]
    load_frames = json.loads((load / "replay/replay.json").read_text())["frames"]
    for before, after in zip(base_frames, load_frames, strict=True):
        assert len(after["tracks"]) == len(before["tracks"]) + 100
        ids = [track["track_id"] for track in after["tracks"]]
        assert len(set(ids)) == len(ids)
        assert after["nodes"] == before["nodes"]


def test_render_fixtures_declares_the_g2_and_realism_fixtures():
    spec = importlib.util.spec_from_file_location(
        "render_fixtures", ROOT / "scripts/render_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.FIXTURES) == {
        "overview",
        "close",
        "low_pitch",
        "split",
        "high_entity",
        "realism_dense",
        "realism_dense_close",
        "realism_dense_low_pitch",
        "realism_large",
        "realism_large_close",
    }
    assert module.TIERS == ("low", "standard", "high")
