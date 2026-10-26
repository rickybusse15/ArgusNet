"""The canonical real-world scenario (R5): West Pleasant View, Colorado.

The committed sources (a USGS 3DEP DEM and an OSM Overpass extract; see the
fixture's ATTRIBUTION.md) are imported with the parameters in
`canonical.json`, and the import and the run must give the counts recorded
there. Any change to an importer or the builder that changes this world
shows here, to be accepted by updating `canonical.json` deliberately.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("pyproj")

from argusnet.world.world_import import import_world  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real" / "west_pleasant_view"
CANONICAL = json.loads((FIXTURE / "canonical.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("canonical") / CANONICAL["name"]
    manifest = import_world(
        FIXTURE / CANONICAL["dem"],
        FIXTURE / CANONICAL["osm"],
        out,
        map_preset=CANONICAL["map_preset"],
        origin=tuple(CANONICAL["origin"]),  # type: ignore[arg-type]
        terrain_preset=CANONICAL["terrain_preset"],
        seed=CANONICAL["seed"],
        duration_s=CANONICAL["duration_s"],
    )
    return out, manifest


def test_the_import_gives_the_recorded_world(world: tuple[Path, dict]) -> None:
    _, manifest = world
    expected = CANONICAL["expected"]["import"]
    summaries = manifest["summaries"]
    assert summaries["buildings"]["buildings"] == expected["buildings"]
    assert (
        summaries["buildings"]["estimated_heights"] == expected["buildings_with_estimated_height"]
    )
    assert summaries["features"]["trees"] == expected["trees"]
    assert summaries["features"]["power_lines"] == expected["power_lines"]
    assert summaries["features"]["features_by_class"] == expected["features_by_class"]
    low, high = expected["elevation_m"]
    assert summaries["terrain"]["elevation_min_m"] == pytest.approx(low, abs=0.01)
    assert summaries["terrain"]["elevation_max_m"] == pytest.approx(high, abs=0.01)
    assert manifest["origin"] == {
        "latitude_deg": CANONICAL["origin"][0],
        "longitude_deg": CANONICAL["origin"][1],
    }


def test_the_sources_are_the_attributed_ones() -> None:
    import hashlib

    attribution = (FIXTURE / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for name in (CANONICAL["dem"], CANONICAL["osm"]):
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert digest in attribution, f"{name}'s hash is not the one ATTRIBUTION.md records"
    assert "Open Database License" in attribution and "public domain" in attribution


def test_the_canonical_world_runs(world: tuple[Path, dict]) -> None:
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    out, _ = world
    replay = out / "replay.json"
    subprocess.run(
        [
            str(binary),
            "sim",
            "--quiet",
            "--config",
            str(out / "scenario.json"),
            "--replay",
            str(replay),
        ],
        check=True,
        capture_output=True,
    )
    meta = json.loads(replay.read_text(encoding="utf-8"))["meta"]
    expected = CANONICAL["expected"]["run"]
    assert meta["buildings"]["buildings"] == expected["buildings"]
    assert meta["buildings"]["left_out_outside_map"] == expected["buildings_outside_map"]
    assert meta["features"]["trees"] == expected["trees"]
    assert meta["features"]["trees_left_out"]["outside_map"] == expected["trees_outside_map"]
    lines = {line["id"]: len(line["supports"]) for line in meta["power_lines"]}
    assert lines == expected["power_lines"]
    assert meta["enu_origin"]["altitude_m"] == expected["origin_h_m"]
    low, high = expected["terrain_m"]
    assert meta["terrain_summary"]["min_height_m"] == pytest.approx(low, abs=0.01)
    assert meta["terrain_summary"]["max_height_m"] == pytest.approx(high, abs=0.01)
