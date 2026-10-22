"""One real-world import (R5): a DEM and an OSM extract to a runnable world.

The place is synthetic but shaped like the data: a DEM in UTM zone 11 (the
terrain importer's tests' tilted plane) and an extract in degrees with two
buildings, a forest, a lake, a road, a power line on towers and two trees.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("pyproj")

from argusnet.world.world_import import import_world, split_buildings  # noqa: E402
from tests.test_terrain_import import ORIGIN, aligned, write_dem  # noqa: E402

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(ORIGIN[0]))


def lonlat(x: float, y: float) -> list[float]:
    return [ORIGIN[1] + x / M_PER_DEG_LON, ORIGIN[0] + y / M_PER_DEG_LAT]


def square(cx: float, cy: float, half: float) -> list[list[list[float]]]:
    ring = [
        lonlat(cx - half, cy - half),
        lonlat(cx + half, cy - half),
        lonlat(cx + half, cy + half),
        lonlat(cx - half, cy + half),
    ]
    return [[*ring, ring[0]]]


def feature(tags: dict, kind: str, coordinates: list, fid: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"@id": fid, **tags},
        "geometry": {"type": kind, "coordinates": coordinates},
    }


def extract(path: Path, *, with_buildings: bool = True, only_buildings: bool = False) -> Path:
    features = (
        []
        if only_buildings
        else [
            feature({"landuse": "forest"}, "Polygon", square(-150, 150, 100), "way/1"),
            feature({"natural": "water"}, "Polygon", square(150, 150, 50), "way/2"),
            feature(
                {"highway": "secondary"}, "LineString", [lonlat(-300, 0), lonlat(300, 0)], "way/3"
            ),
            feature(
                {"power": "line", "voltage": "132000"},
                "LineString",
                [lonlat(-300, -250), lonlat(0, -250), lonlat(300, -250)],
                "way/4",
            ),
            feature({"power": "tower", "height": "36"}, "Point", lonlat(0, -250), "node/5"),
            feature({"natural": "tree", "height": "14"}, "Point", lonlat(40, 60), "node/6"),
            feature(
                {"natural": "tree", "leaf_type": "needleleaved"}, "Point", lonlat(60, 60), "node/7"
            ),
        ]
    )
    if with_buildings:
        features += [
            feature(
                {"building": "house", "building:levels": "2"},
                "Polygon",
                square(100, -100, 8),
                "way/8",
            ),
            feature(
                {"building": "yes", "height": "18"}, "Polygon", square(-100, -100, 15), "way/9"
            ),
            feature(
                {"building": "no", "amenity": "parking"},
                "Polygon",
                square(-200, -100, 10),
                "way/10",
            ),
        ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def place(
    tmp_path: Path, name: str = "world", terrain_preset: str = "urban_flat", **kwargs: object
) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    osm = extract(tmp_path / f"{name}.geojson", **kwargs)  # type: ignore[arg-type]
    out = tmp_path / name
    return out, import_world(
        dem,
        osm,
        out,
        map_preset="small",
        origin=ORIGIN,
        terrain_preset=terrain_preset,
        duration_s=6.0,
    )


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_layer_is_imported_around_one_origin(tmp_path: Path) -> None:
    out, manifest = place(tmp_path)
    assert manifest["format"] == "argusnet.world-import"
    assert set(manifest["files"]) == {"terrain", "buildings", "features"}
    for name, entry in manifest["files"].items():
        document = load(out / entry["path"])
        georeference = document["georeference"]
        assert (georeference["origin_lat_deg"], georeference["origin_lon_deg"]) == ORIGIN, name
        assert entry["sha256"] == hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest()
    assert manifest["origin"] == {"latitude_deg": ORIGIN[0], "longitude_deg": ORIGIN[1]}
    assert not (out / "osm-buildings.geojson").exists()


def test_buildings_go_to_the_buildings_importer_and_nothing_else(tmp_path: Path) -> None:
    out, _ = place(tmp_path)
    buildings = load(out / "buildings.json")["buildings"]
    assert sorted(b["id"] for b in buildings) == ["way/8", "way/9"]
    features = load(out / "features.json")
    assert sorted({e["class"] for e in features["land_cover"]}) == ["forest", "road", "water"]
    assert len(features["trees"]) == 2 and len(features["power_lines"]) == 1
    # The two buildings and the `building=no` parking are skipped there.
    assert features["summary"]["skipped"]["untagged"] == 3
    split = tmp_path / "split.geojson"
    assert split_buildings(tmp_path / "world.geojson", split) == 2


def test_the_same_data_is_the_same_world(tmp_path: Path) -> None:
    _, first = place(tmp_path / "a")
    _, again = place(tmp_path / "b")
    assert first["world_id"] == again["world_id"]
    _, other = place(tmp_path / "c", with_buildings=False)
    assert other["world_id"] != first["world_id"]
    assert "buildings" not in other["files"]
    scenario = load(tmp_path / "c" / "world" / "scenario.json")
    assert "buildings_path" not in scenario["options"]


def test_the_scenario_names_every_file(tmp_path: Path) -> None:
    out, _ = place(tmp_path)
    options = load(out / "scenario.json")["options"]
    assert options["map_preset"] == "small" and options["trees"] is True
    for name in ("terrain", "buildings", "features"):
        assert Path(options[f"{name}_path"]) == (out / f"{name}.json").resolve()


def test_the_imported_world_runs(tmp_path: Path) -> None:
    """End to end: the scenario runs in `argusnet-rs`, standing every layer
    on the one origin; the replay records all three."""
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    out, manifest = place(tmp_path)
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
    meta = load(replay)["meta"]
    assert meta["terrain_summary"]["source"]["source"] == "dem-import"
    assert meta["buildings"]["buildings"] == 2
    assert meta["features"]["power_lines"] == 1
    assert meta["features"]["trees"] == 2
    assert meta["enu_origin"]["latitude_deg"] == ORIGIN[0]
    (line,) = meta["power_lines"]
    assert [s["height_m"] for s in line["supports"]] == [30.0, 36.0, 30.0]
    assert line["clearance_m"] == 10.0
    mapped = [t for t in meta["trees"] if t["id"].startswith("node/")]
    assert sorted(t["id"] for t in mapped) == ["node/6", "node/7"]
    # The run says which DEM it stands on: the manifest's.
    assert meta["terrain_import"]["source_sha256"] == manifest["sources"]["dem_sha256"]


def test_without_an_origin_every_layer_takes_the_dems_centre(tmp_path: Path) -> None:
    # The DEM's centre is 100 m east of the extract's, so the two defaults
    # differ and only one can win.
    a, b, c, d, e, f = aligned()
    dem = write_dem(tmp_path / "dem.tif", affine=(a, b, c + 100.0, d, e, f), shape=(120, 120))
    osm = extract(tmp_path / "town.geojson")
    out = tmp_path / "world"
    manifest = import_world(dem, osm, out, map_preset="small")
    origins = {
        name: (
            load(out / entry["path"])["georeference"]["origin_lat_deg"],
            load(out / entry["path"])["georeference"]["origin_lon_deg"],
        )
        for name, entry in manifest["files"].items()
    }
    assert len(set(origins.values())) == 1, origins
    east_m = (origins["terrain"][1] - ORIGIN[1]) * M_PER_DEG_LON
    assert east_m == pytest.approx(100.0, abs=2.0)


def test_the_world_id_covers_what_shapes_the_world(tmp_path: Path) -> None:
    _, first = place(tmp_path / "a")
    _, alpine = place(tmp_path / "b", terrain_preset="alpine")
    assert alpine["world_id"] != first["world_id"]
    assert alpine["files"] == first["files"]


def test_an_extract_of_only_buildings_still_says_what_is_there(tmp_path: Path) -> None:
    """No land cover, trees or lines mapped: an empty features file is
    written and named, so the builder paints open ground and invents no
    forest, walls or trees."""
    out, manifest = place(tmp_path, only_buildings=True)
    features = load(out / "features.json")
    assert features["land_cover"] == features["trees"] == features["power_lines"] == []
    assert Path(load(out / "scenario.json")["options"]["features_path"]).exists()
    assert "features" in manifest["files"]
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
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
    meta = load(replay)
    assert meta["meta"].get("trees", []) == []
    assert meta["meta"]["buildings"]["buildings"] == 2
