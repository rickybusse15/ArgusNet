"""Overpass `out geom` answers to GeoJSON (R5)."""

from __future__ import annotations

from argusnet.world.overpass import is_area, stitch, to_geojson


def g(*points: tuple[float, float]) -> list[dict[str, float]]:
    return [{"lon": x, "lat": y} for x, y in points]


SQUARE = g((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))


def test_nodes_ways_and_areas() -> None:
    answer = {
        "elements": [
            {"type": "node", "id": 1, "lat": 2.0, "lon": 3.0, "tags": {"natural": "tree"}},
            {"type": "node", "id": 2, "lat": 0.0, "lon": 0.0},
            {"type": "way", "id": 3, "tags": {"building": "yes"}, "geometry": SQUARE},
            {"type": "way", "id": 4, "tags": {"highway": "residential"}, "geometry": SQUARE},
            {"type": "way", "id": 5, "tags": {"natural": "tree_row"}, "geometry": SQUARE},
            {"type": "way", "id": 6, "tags": {"power": "line"}, "geometry": g((0, 0), (5, 5))},
            {"type": "way", "id": 7, "tags": {"highway": "path"}, "geometry": g((0, 0))},
        ]
    }
    collection = to_geojson(answer)
    by_id = {f["properties"]["@id"]: f for f in collection["features"]}
    assert set(by_id) == {"node/1", "way/3", "way/4", "way/5", "way/6"}
    assert by_id["node/1"]["geometry"] == {"type": "Point", "coordinates": [3.0, 2.0]}
    assert by_id["way/3"]["geometry"]["type"] == "Polygon"
    # A closed road is still a road; a tree row is a line.
    assert by_id["way/4"]["geometry"]["type"] == "LineString"
    assert by_id["way/5"]["geometry"]["type"] == "LineString"
    assert by_id["way/6"]["properties"]["power"] == "line"
    assert collection["argusnet_overpass"]["skipped"] == 1


def test_area_tags() -> None:
    assert is_area({"landuse": "forest"}) and is_area({"natural": "water"})
    assert not is_area({"natural": "coastline"}) and not is_area({"highway": "service"})
    assert is_area({"highway": "pedestrian", "area": "yes"})
    assert not is_area({"building": "yes", "area": "no"})


def test_ways_are_stitched_into_rings_either_way_round() -> None:
    parts = [
        [[0, 0], [1, 0]],
        [[1, 1], [1, 0]],  # reversed
        [[1, 1], [0, 1]],
        [[0, 0], [0, 1]],  # reversed, closes it
        [[5, 5], [6, 6]],  # never closes
    ]
    rings, unclosed = stitch(parts)
    assert unclosed == 1
    (ring,) = rings
    assert ring[0] == ring[-1] and len(ring) == 5
    assert sorted(map(tuple, ring[:-1])) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_a_multipolygon_keeps_its_holes_in_the_right_ring() -> None:
    big = g((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
    other = g((20, 0), (30, 0), (30, 10), (20, 10), (20, 0))
    hole = g((2, 2), (4, 2), (4, 4), (2, 4), (2, 2))
    answer = {
        "elements": [
            {
                "type": "relation",
                "id": 9,
                "tags": {"type": "multipolygon", "natural": "water"},
                "members": [
                    {"role": "outer", "geometry": big[:3]},
                    {"role": "outer", "geometry": big[2:]},
                    {"role": "outer", "geometry": other},
                    {"role": "inner", "geometry": hole},
                ],
            },
            {
                "type": "relation",
                "id": 10,
                "tags": {"type": "multipolygon", "landuse": "forest"},
                "members": [{"role": "outer", "geometry": big[:3]}],
            },
        ]
    }
    collection = to_geojson(answer)
    (lake,) = collection["features"]
    assert lake["properties"]["@id"] == "relation/9"
    polygons = lake["geometry"]["coordinates"]
    assert len(polygons) == 2
    with_hole = [p for p in polygons if len(p) == 2]
    assert len(with_hole) == 1 and with_hole[0][0][0] == [0.0, 0.0]
    assert collection["argusnet_overpass"]["skipped"] == 1
