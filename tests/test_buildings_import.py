"""R1: building footprints to `argusnet.buildings` (scripts/import_buildings.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argusnet.world.buildings_import import import_buildings, parse_length_m, ridge_azimuth_rad

FIXTURES = Path(__file__).parent / "fixtures"
TOWN = FIXTURES / "buildings_town.geojson"


@pytest.fixture(scope="module")
def town() -> dict:
    return import_buildings(TOWN)


def by_id(document: dict) -> dict:
    return {b["id"]: b for b in document["buildings"]}


def test_the_committed_document_is_what_the_importer_writes(town: dict) -> None:
    # The native builder's tests read the committed file; it must not drift
    # from the importer (regenerate with scripts/import_buildings.py).
    committed = json.loads((FIXTURES / "buildings_town.buildings.json").read_text())
    assert committed == json.loads(json.dumps(town))


def test_the_document_is_georeferenced(town: dict) -> None:
    georeference = town["georeference"]
    assert georeference["source_crs_id"] == "EPSG:4326"
    assert georeference["runtime_crs_id"] == "local-enu"
    assert abs(georeference["origin_lat_deg"] - 46.0) < 1e-3
    assert abs(georeference["origin_lon_deg"] - 7.0) < 1e-3
    # Projection keeps every footprint's area to the ellipsoid's.
    assert town["summary"]["worst_footprint_area_error"] < 1e-6
    # A 12 x 8 m house is 12 x 8 m in local metres.
    xs = [p[0] for p in by_id(town)["way/1"]["footprint_xy_m"]]
    ys = [p[1] for p in by_id(town)["way/1"]["footprint_xy_m"]]
    assert abs((max(xs) - min(xs)) - 12.0) < 0.05
    assert abs((max(ys) - min(ys)) - 8.0) < 0.05


def test_an_explicit_origin_moves_the_frame() -> None:
    shifted = import_buildings(TOWN, origin=(46.0, 7.0))
    assert shifted["georeference"]["origin_lat_deg"] == 46.0
    # The footprints were laid out around (46.0, 7.0): house 1 starts at -80 m east.
    first = by_id(shifted)["way/1"]["footprint_xy_m"][0]
    assert abs(first[0] + 80.0) < 0.2 and abs(first[1] + 40.0) < 0.2


def test_heights_follow_the_data_and_say_when_they_do_not(town: dict) -> None:
    b = by_id(town)
    assert (b["way/1"]["height_m"], b["way/1"]["height_source"]) == (9.0, "measured")
    assert (b["way/2"]["height_m"], b["way/2"]["height_source"]) == (6.0, "levels")
    assert (b["way/5"]["height_m"], b["way/5"]["height_source"]) == (10.0, "default")
    assert abs(b["way/6"]["height_m"] - 40 * 0.3048) < 1e-9
    assert town["summary"]["estimated_heights"] == 4


def test_roofs_follow_the_data_and_unmodelled_ones_are_flat_at_full_height(town: dict) -> None:
    b = by_id(town)
    assert (b["way/1"]["roof_shape"], b["way/1"]["eave_height_m"]) == ("gabled", 6.0)
    assert b["way/3"]["roof_shape"] == "pyramidal" and b["way/3"]["eave_height_m"] == 12.0
    # A hip roof over an L, and a mansard: flat at the full height.
    for key, why in [("way/6", "unsupported:non-convex-hipped"), ("way/7", "unsupported:mansard")]:
        assert b[key]["roof_shape"] == "flat"
        assert b[key]["eave_height_m"] == b[key]["height_m"]
        assert b[key]["roof_source"] == why
    # Holes are left out (recorded); each polygon of a multipolygon is a building.
    assert town["summary"]["holes_ignored"] == 1
    assert {"relation/9", "relation/9#1"} <= set(b)


def test_ridges_follow_roof_direction_and_orientation(town: dict) -> None:
    b = by_id(town)
    # A slope facing south: the ridge runs east-west.
    assert abs(math.sin(b["way/1"]["ridge_azimuth_rad"])) < 1e-12
    # Across the longest edge (east-west): the ridge runs north-south, to
    # within the few microradians a tangent-plane frame turns a meridian.
    assert abs(abs(b["way/8"]["ridge_azimuth_rad"]) - math.pi / 2) < 1e-5
    assert ridge_azimuth_rad({"roof:direction": "90"}, [[0, 0], [1, 0], [1, 1]]) is not None
    assert "ridge_azimuth_rad" not in b["way/2"]


def test_lengths_are_read_in_metres() -> None:
    assert parse_length_m("12") == 12.0
    assert parse_length_m("12.5 m") == 12.5
    assert abs(parse_length_m("40 ft") - 12.192) < 1e-9
    assert parse_length_m("tall") is None
    assert parse_length_m("-3") is None


def test_materials_are_classed_from_the_data_and_generic_otherwise(town: dict) -> None:
    b = by_id(town)
    assert (b["way/1"]["facade_material"], b["way/1"]["roof_material"]) == ("brick", "tiles")
    assert (b["way/4"]["facade_material"], b["way/4"]["roof_material"]) == ("glass", "concrete")
    assert (b["way/5"]["facade_material"], b["way/5"]["roof_material"]) == ("metal", "metal")
    # An unknown material, and none at all: generic, never a guess.
    assert b["way/7"]["facade_material"] == "generic"
    assert (b["way/3"]["facade_material"], b["way/3"]["roof_material"]) == ("generic", "generic")


def test_levels_are_kept_only_when_the_data_gives_them(town: dict) -> None:
    b = by_id(town)
    assert b["way/4"]["levels"] == 8.0
    assert b["way/2"]["levels"] == 2.0
    assert "levels" not in b["way/3"]
