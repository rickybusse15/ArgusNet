"""The land-cover importer (R5): OSM GeoJSON to `argusnet.features`.

A synthetic extract around the synthetic town's origin, in degrees, with
the tags OSM writes: a park with a lake in it (and an island), a forest, a
residential area, a river, roads with and without widths, a tunnel, a
bridge and a footpath.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")

from argusnet.core.frames import ENUOrigin, wgs84_to_enu  # noqa: E402
from argusnet.world.features_import import import_features, segment_rectangles  # noqa: E402

ORIGIN = (45.999892, 7.0)
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(ORIGIN[0]))


def lonlat(x: float, y: float) -> list[float]:
    return [ORIGIN[1] + x / M_PER_DEG_LON, ORIGIN[0] + y / M_PER_DEG_LAT]


def square(cx: float, cy: float, half: float) -> list[list[float]]:
    ring = [
        lonlat(cx - half, cy - half),
        lonlat(cx + half, cy - half),
        lonlat(cx + half, cy + half),
        lonlat(cx - half, cy + half),
    ]
    return [*ring, ring[0]]


def feature(tags: dict, kind: str, coordinates: list, fid: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"@id": fid, **tags},
        "geometry": {"type": kind, "coordinates": coordinates},
    }


def extract(path: Path) -> Path:
    features = [
        # Roads first in the file: painted last all the same.
        feature(
            {"highway": "primary", "lanes": "2"},
            "LineString",
            [lonlat(-300, -100), lonlat(0, -100), lonlat(0, 200)],
            "way/10",
        ),
        feature(
            {"highway": "residential", "width": "9 m"},
            "LineString",
            [lonlat(-300, 250), lonlat(300, 250)],
            "way/11",
        ),
        feature(
            {"highway": "service", "tunnel": "yes"},
            "LineString",
            [lonlat(0, 0), lonlat(50, 0)],
            "way/12",
        ),
        feature(
            {"highway": "tertiary", "bridge": "yes"},
            "LineString",
            [lonlat(100, 0), lonlat(150, 0)],
            "way/13",
        ),
        feature({"highway": "footway"}, "LineString", [lonlat(0, 0), lonlat(9, 9)], "way/14"),
        feature(
            {"waterway": "river", "width": "20"},
            "LineString",
            [lonlat(-300, -300), lonlat(300, -300)],
            "way/20",
        ),
        feature(
            {"natural": "water"}, "Polygon", [square(200, 150, 40), square(200, 150, 5)], "way/21"
        ),
        feature({"leisure": "park"}, "Polygon", [square(200, 150, 90)], "way/22"),
        feature({"landuse": "forest"}, "Polygon", [square(-150, 100, 100)], "way/23"),
        feature({"landuse": "residential"}, "Polygon", [square(150, -150, 80)], "relation/24"),
        feature({"building": "yes"}, "Polygon", [square(0, 0, 5)], "way/25"),
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_tags_become_land_cover_in_painting_order(tmp_path: Path) -> None:
    document = import_features(extract(tmp_path / "town.geojson"), origin=ORIGIN)
    assert document["format"] == "argusnet.features" and document["version"] == 1
    cover = document["land_cover"]
    classes = [entry["class"] for entry in cover]
    # open, then urban, then natural, then water, then roads; file order within.
    assert classes == ["open", "urban", "forest", "water", "water", "road", "road", "road"]
    ids = [entry["id"] for entry in cover]
    assert ids == [
        "way/22",
        "relation/24",
        "way/23",
        "way/20",
        "way/21",
        "way/10/0",
        "way/10/1",
        "way/11",
    ]
    summary = document["summary"]
    assert summary["skipped"] == {
        "untagged": 2,
        "tunnel_or_bridge": 2,
        "degenerate": 0,
        "underground_cable": 0,
    }
    assert summary["holes_ignored"] == 1
    assert summary["features_by_class"] == {
        "forest": 1,
        "open": 1,
        "road": 2,
        "urban": 1,
        "water": 2,
    }
    assert (
        document["source_sha256"]
        == hashlib.sha256((tmp_path / "town.geojson").read_bytes()).hexdigest()
    )


def test_polygons_land_in_the_run_frame(tmp_path: Path) -> None:
    document = import_features(extract(tmp_path / "town.geojson"), origin=ORIGIN)
    forest = next(e for e in document["land_cover"] if e["class"] == "forest")
    enu = ENUOrigin(latitude_deg=ORIGIN[0], longitude_deg=ORIGIN[1], altitude_m=0.0)
    expected = [wgs84_to_enu(lat, lon, 0.0, enu)[:2] for lon, lat in square(-150, 100, 100)[:-1]]
    np.testing.assert_allclose(forest["polygon_xy_m"], expected, atol=1e-3)
    # And near the square it was drawn as (the degree conversion is rough).
    np.testing.assert_allclose(forest["polygon_xy_m"][0], [-250, 0], atol=1.0)


def test_lines_are_drawn_at_their_width(tmp_path: Path) -> None:
    document = import_features(extract(tmp_path / "town.geojson"), origin=ORIGIN)
    by_id = {entry["id"]: entry for entry in document["land_cover"]}
    # `lanes=2` -> 7 m; `width=9 m`; a river's `width=20`.
    assert by_id["way/10/0"]["width_m"] == 7.0
    assert by_id["way/11"]["width_m"] == 9.0
    assert by_id["way/20"]["width_m"] == 20.0
    for key, width in (("way/10/0", 7.0), ("way/11", 9.0), ("way/20", 20.0)):
        polygon = np.array(by_id[key]["polygon_xy_m"])
        across = np.linalg.norm(polygon[3] - polygon[0])
        assert across == pytest.approx(width, abs=1e-2)


def test_segments_overlap_at_their_joints() -> None:
    first, second = segment_rectangles([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], 4.0)
    # Each runs half the width past its ends: the corner square is covered.
    xs = [p[0] for p in first]
    assert min(xs) == -2.0 and max(xs) == 12.0
    ys = [p[1] for p in second]
    assert min(ys) == -2.0 and max(ys) == 12.0
    assert segment_rectangles([[1.0, 1.0], [1.0, 1.0]], 3.0) == []


def test_the_default_origin_is_the_extracts_centre(tmp_path: Path) -> None:
    document = import_features(extract(tmp_path / "town.geojson"))
    georeference = document["georeference"]
    # The extent's points run x -300..300 and y -300..250 (the river's
    # centre line, not its edge): the centre is 25 m south of the origin.
    assert georeference["origin_lon_deg"] == pytest.approx(ORIGIN[1], abs=1e-7)
    assert georeference["origin_lat_deg"] == pytest.approx(
        ORIGIN[0] - 25.0 / M_PER_DEG_LAT, abs=1e-6
    )


def test_an_extract_with_no_land_cover_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "none.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    feature({"building": "yes"}, "Polygon", [square(0, 0, 5)], "way/1"),
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="no land-cover, tree or power-line features"):
        import_features(path, origin=ORIGIN)


def test_a_run_paints_the_imported_land_cover(tmp_path: Path) -> None:
    """End to end: the importer's document, run by `argusnet-rs sim`. The
    replay records it, its polygons by class and its origin. (What the land
    cover becomes is the builder's tests' to check: the replay carries no
    raster.)"""
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    document = import_features(extract(tmp_path / "town.geojson"), origin=ORIGIN)
    features_path = tmp_path / "features.json"
    features_path.write_text(json.dumps(document), encoding="utf-8")
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 7,
                "duration_s": 4.0,
                "options": {
                    "map_preset": "small",
                    "terrain_preset": "urban_flat",
                    "features_path": str(features_path),
                },
            }
        ),
        encoding="utf-8",
    )
    replay = tmp_path / "replay.json"
    subprocess.run(
        [str(binary), "sim", "--quiet", "--config", str(config), "--replay", str(replay)],
        check=True,
        capture_output=True,
    )
    meta = json.loads(replay.read_text(encoding="utf-8"))["meta"]
    record = meta["features"]
    assert record["source_sha256"] == document["source_sha256"]
    assert record["land_cover_polygons"] == {
        "forest": 1,
        "open": 1,
        "road": 3,
        "urban": 1,
        "water": 2,
    }
    assert meta["enu_origin"]["latitude_deg"] == ORIGIN[0]


def trees_and_lines(path: Path) -> Path:
    tower = lonlat(0, 300)
    features = [
        feature(
            {
                "natural": "tree",
                "leaf_type": "needleleaved",
                "height": "12 m",
                "diameter_crown": "6",
                "circumference": "1.2",
            },
            "Point",
            lonlat(10, 20),
            "node/1",
        ),
        feature({"natural": "tree", "genus": "Phoenix"}, "Point", lonlat(-10, 20), "node/2"),
        feature({"natural": "tree"}, "Point", lonlat(0, -20), "node/3"),
        feature({"natural": "tree_row"}, "LineString", [lonlat(0, 50), lonlat(21, 50)], "way/4"),
        feature({"natural": "tree_row"}, "LineString", [lonlat(0, 60), lonlat(18, 60)], "way/5"),
        feature(
            {"power": "line", "cables": "3", "voltage": "110000;20000"},
            "LineString",
            [lonlat(-200, 300), tower, lonlat(200, 300)],
            "way/6",
        ),
        feature({"power": "tower", "height": "42"}, "Point", tower, "node/7"),
        feature({"power": "pole"}, "Point", lonlat(-200, 300), "node/8"),
        feature({"power": "minor_line"}, "LineString", [lonlat(0, 0), lonlat(0, 0)], "way/9"),
        feature({"power": "cable"}, "LineString", [lonlat(0, 0), lonlat(9, 0)], "way/10"),
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_mapped_trees_keep_their_tags_and_leave_the_rest_to_the_builder(tmp_path: Path) -> None:
    document = import_features(trees_and_lines(tmp_path / "t.geojson"), origin=ORIGIN)
    trees = {tree["id"]: tree for tree in document["trees"]}
    conifer = trees["node/1"]
    assert conifer["species"] == "conifer"
    assert conifer["height_m"] == 12.0 and conifer["crown_radius_m"] == 3.0
    assert conifer["trunk_radius_m"] == pytest.approx(1.2 / (2 * math.pi), abs=1e-3)
    np.testing.assert_allclose(conifer["xy_m"], [10, 20], atol=0.5)
    assert trees["node/2"]["species"] == "palm"
    plain = trees["node/3"]
    assert plain["species"] == "broadleaf"
    assert set(plain) == {"id", "xy_m", "species"}
    # A 21 m row: 0, 8, 16 and its end; an 18 m one ends 2 m past 16: not
    # a tree closer than half the spacing.
    row = [t for t in document["trees"] if t["id"].startswith("way/4/")]
    xs = [t["xy_m"][0] for t in row]
    np.testing.assert_allclose(xs, [0, 8, 16, 21], atol=0.05)
    assert len([t for t in document["trees"] if t["id"].startswith("way/5/")]) == 3


def test_mapped_lines_take_tower_heights_cables_and_voltage(tmp_path: Path) -> None:
    document = import_features(trees_and_lines(tmp_path / "t.geojson"), origin=ORIGIN)
    (line,) = document["power_lines"]
    assert line["id"] == "way/6" and line["kind"] == "line"
    assert line["cables"] == 3 and line["voltage_v"] == 110000.0
    supports = line["supports"]
    assert [s.get("height_m") for s in supports] == [None, 42.0, None]
    np.testing.assert_allclose(supports[1]["xy_m"], [0, 300], atol=0.5)
    summary = document["summary"]
    assert summary["power_lines"] == 1 and summary["power_line_supports_with_height"] == 1
    # A line with one distinct support, and an underground cable.
    assert summary["skipped"]["degenerate"] == 1
    assert summary["skipped"]["underground_cable"] == 1
    assert document["land_cover"] == []
