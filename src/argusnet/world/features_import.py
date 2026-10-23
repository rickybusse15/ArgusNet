"""Real land cover for a run (R5): OSM features to `argusnet.features`.

Offline tooling, like the buildings and terrain importers. It reads a GeoJSON
FeatureCollection, such as an OSM export with the tags as properties, and
writes an `argusnet.features` v1 document. The native builder's
`features_path` option paints the run's land cover from it.

Georeferencing is the buildings importer's: footprints in a source CRS
(EPSG:4326 unless the file names another), projected to local east-north-up
metres around a geodetic origin, which defaults to the centre of the
features' extent. Pass the same `--origin` to every importer: the builder
refuses layers whose origins differ.

What becomes what. The tags are read as OSM writes them, and anything else
is skipped and counted, never guessed:

* water: `natural=water`, any `water=*`, `waterway=riverbank|dock`,
  `landuse=reservoir|basin`; rivers, canals, streams and ditches drawn as
  lines, at their `width` or a default by kind.
* wetland: `natural=wetland`. forest: `landuse=forest`, `natural=wood`.
  scrub: `natural=scrub|heath`. rocky: `natural=bare_rock|scree|shingle`.
  snow: `natural=glacier`.
* urban: `landuse=residential|commercial|industrial|retail|construction|
  garages|railway`.
* open: grass, meadow, farmland, parks, pitches and the like, which says
  "open" explicitly, over whatever was under it.
* road: `highway=*` lines for vehicles (motorway to track), at `width`,
  else `lanes` x 3.5 m, else a default by class. Tunnels and bridges are
  left out, so a road does not paint over a river it crosses.

Lines are written as their segments (`line_xy_m`) with their width: the
builder paints each as a rectangle at least as wide as the land cover's
cells need, so a 3 m track still marks every cell it crosses. Polygons keep
their outer ring. Holes are ignored and counted: an island in a mapped lake
reads as water.

Painting order is the document's order: open, then urban, then the natural
classes, then water, then roads, each in the file's order. So a road runs
through a forest, and a lake inside a park stays a lake.

Trees and power lines (R5.2b):

* trees: `natural=tree` points, and `natural=tree_row` lines with a tree
  every 8 m along them. Species from `leaf_type` (`needleleaved` is a
  conifer) or a palm genus or family, otherwise broadleaf. `height`,
  `diameter_crown` and `circumference` are kept when tagged. Anything
  untagged is left for the builder to fill from its species table, never
  guessed here.
* power lines: `power=line` and `power=minor_line` ways, with a support at
  every vertex. A support's height comes from a `power=tower|pole` point
  on that vertex with a `height`, else from the builder's default for the
  kind. `cables` and `voltage` are kept. Underground cables
  (`power=cable`) are skipped and counted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from argusnet.core.frames import ENUOrigin, wgs84_to_enu
from argusnet.core.io_limits import sha256_file

from ._scene_gis import lazy_import_pyproj, load_geojson_layer
from .buildings_import import _ring, centre_longitude, parse_length_m

FORMAT = "argusnet.features"
VERSION = 1
LANE_WIDTH_M = 3.5

# Painting order: a later class paints over an earlier one.
LAYER_ORDER = ("open", "urban", "natural", "water", "road")
CLASS_LAYER = {
    "open": "open",
    "urban": "urban",
    "forest": "natural",
    "scrub": "natural",
    "wetland": "natural",
    "rocky": "natural",
    "snow": "natural",
    "water": "water",
    "road": "road",
}

AREA_TAGS: dict[tuple[str, str], str] = {
    ("natural", "water"): "water",
    ("waterway", "riverbank"): "water",
    ("waterway", "dock"): "water",
    ("landuse", "reservoir"): "water",
    ("landuse", "basin"): "water",
    ("natural", "wetland"): "wetland",
    ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest",
    ("natural", "scrub"): "scrub",
    ("natural", "heath"): "scrub",
    ("natural", "bare_rock"): "rocky",
    ("natural", "scree"): "rocky",
    ("natural", "shingle"): "rocky",
    ("natural", "glacier"): "snow",
    **{
        ("landuse", value): "urban"
        for value in (
            "residential",
            "commercial",
            "industrial",
            "retail",
            "construction",
            "garages",
            "railway",
        )
    },
    **{
        (key, value): "open"
        for key, value in (
            ("landuse", "grass"),
            ("landuse", "meadow"),
            ("landuse", "farmland"),
            ("landuse", "farmyard"),
            ("landuse", "recreation_ground"),
            ("landuse", "village_green"),
            ("landuse", "cemetery"),
            ("natural", "grassland"),
            ("leisure", "park"),
            ("leisure", "pitch"),
            ("leisure", "golf_course"),
        )
    },
}

# Default widths, metres, for lines without a usable `width` (or `lanes`).
WATERWAY_WIDTH_M = {"river": 12.0, "canal": 8.0, "stream": 2.5, "ditch": 1.5, "drain": 1.5}
ROAD_WIDTH_M = {
    "motorway": 14.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 7.0,
    "motorway_link": 6.0,
    "trunk_link": 6.0,
    "primary_link": 6.0,
    "secondary_link": 6.0,
    "tertiary_link": 6.0,
    "residential": 6.0,
    "unclassified": 5.0,
    "living_street": 5.0,
    "service": 4.0,
    "track": 3.0,
}


def _text(properties: Mapping[str, object], key: str) -> str:
    return str(properties.get(key, "")).strip().lower()


def _yes(properties: Mapping[str, object], key: str) -> bool:
    value = _text(properties, key)
    return value not in ("", "no", "false", "0")


def area_class(properties: Mapping[str, object]) -> str | None:
    """The land-cover class a polygon's tags give, or None."""
    if _text(properties, "water"):
        return "water"
    for (key, value), cover in AREA_TAGS.items():
        if _text(properties, key) == value:
            return cover
    return None


def line_class(properties: Mapping[str, object]) -> tuple[str, float] | None:
    """(class, width in metres) for a line the tags make land cover, or None."""
    waterway = _text(properties, "waterway")
    if waterway in WATERWAY_WIDTH_M:
        return "water", parse_length_m(properties.get("width")) or WATERWAY_WIDTH_M[waterway]
    highway = _text(properties, "highway")
    if highway in ROAD_WIDTH_M:
        width: float | None = parse_length_m(properties.get("width"))
        if width is None:
            try:
                lanes = float(_text(properties, "lanes"))
            except ValueError:
                lanes = 0.0
            width = lanes * LANE_WIDTH_M if lanes > 0 and math.isfinite(lanes) else None
        return "road", width or ROAD_WIDTH_M[highway]
    return None


def segments(points: Sequence[Sequence[float]]) -> list[list[list[float]]]:
    """The line's segments of non-zero length, as [start, end] pairs."""
    return [
        [[float(x0), float(y0)], [float(x1), float(y1)]]
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False)
        if math.hypot(x1 - x0, y1 - y0) > 0.0
    ]


def _lines(geometry: Mapping[str, object]) -> list[list[list[float]]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if kind == "LineString":
        return [[list(map(float, p[:2])) for p in coordinates]]
    if kind == "MultiLineString":
        return [[list(map(float, p[:2])) for p in line] for line in coordinates]
    return []


def _areas(geometry: Mapping[str, object]) -> list[tuple[list[list[float]], int]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if kind == "Polygon":
        return [(_ring(coordinates[0]), len(coordinates) - 1)] if coordinates else []
    if kind == "MultiPolygon":
        return [(_ring(polygon[0]), len(polygon) - 1) for polygon in coordinates if polygon]
    return []


TREE_ROW_SPACING_M = 8.0
PALM_GENERA = frozenset(
    {"phoenix", "washingtonia", "cocos", "roystonea", "trachycarpus", "sabal", "chamaerops"}
)


def tree_species(properties: Mapping[str, object]) -> str:
    """The species class a tree's tags give: palm, conifer or broadleaf."""
    genus = _text(properties, "genus")
    family = _text(properties, "taxon:family")
    if genus in PALM_GENERA or family == "arecaceae":
        return "palm"
    if _text(properties, "leaf_type") == "needleleaved":
        return "conifer"
    return "broadleaf"


def tree_sizes(properties: Mapping[str, object]) -> dict[str, float]:
    """The sizes a tree's tags give, metres; none when not tagged."""
    sizes: dict[str, float] = {}
    height = parse_length_m(properties.get("height"))
    if height is not None:
        sizes["height_m"] = height
    crown = parse_length_m(properties.get("diameter_crown"))
    if crown is not None:
        sizes["crown_radius_m"] = crown * 0.5
    girth = parse_length_m(properties.get("circumference"))
    if girth is not None:
        sizes["trunk_radius_m"] = round(girth / (2.0 * math.pi), 3)
    return sizes


def along(points: Sequence[Sequence[float]], spacing_m: float) -> list[list[float]]:
    """Points every `spacing_m` along a polyline by the distance walked,
    from its first point, and its last point when that is at least half a
    spacing past the one before."""
    if not points:
        return []
    out = [[float(points[0][0]), float(points[0][1])]]
    walked = 0.0
    next_at = spacing_m
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        length = math.hypot(x1 - x0, y1 - y0)
        while length > 0.0 and next_at <= walked + length:
            t = (next_at - walked) / length
            out.append([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
            next_at += spacing_m
        walked += length
    end = [float(points[-1][0]), float(points[-1][1])]
    if math.hypot(end[0] - out[-1][0], end[1] - out[-1][1]) >= 0.5 * spacing_m:
        out.append(end)
    return out


def _voltage(properties: Mapping[str, object]) -> float | None:
    first = _text(properties, "voltage").split(";")[0]
    try:
        value = float(first)
    except ValueError:
        return None
    return value if value > 0 and math.isfinite(value) else None


def _points(geometry: Mapping[str, object]) -> list[list[float]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if kind == "Point":
        return [list(map(float, coordinates[:2]))]
    if kind == "MultiPoint":
        return [list(map(float, p[:2])) for p in coordinates]
    return []


def _id(properties: Mapping[str, object], index: int) -> str:
    return str(
        properties.get("@id") or properties.get("id") or properties.get("osm_id") or f"f{index}"
    )


def import_features(
    geojson_path: str | Path,
    *,
    origin: tuple[float, float] | None = None,
    source_crs: str | None = None,
) -> dict[str, object]:
    """The `argusnet.features` document for `geojson_path`. `origin` is the
    local frame's (latitude, longitude) in degrees; the centre of the
    features' extent when not given."""
    CRS, Transformer = lazy_import_pyproj()
    source_sha256 = sha256_file(geojson_path)
    layer = load_geojson_layer(geojson_path, default_source_crs_id=source_crs)
    source_crs_id = source_crs or layer.source_crs_id
    to_wgs84 = Transformer.from_crs(
        CRS.from_user_input(source_crs_id), CRS.from_epsg(4326), always_xy=True
    )

    # (layer rank, file order, id, class, lon/lat ring or line, width or None)
    pending: list[tuple[int, int, str, str, list[tuple[float, float]], float | None]] = []
    skipped = {"untagged": 0, "tunnel_or_bridge": 0, "degenerate": 0}
    holes_ignored = 0
    # (id, species, lon/lat, sizes); (id, species, lon/lat line, sizes); and
    # (id, kind, the line's source coordinates, cables, voltage).
    tree_points: list[tuple[str, str, tuple[float, float], dict[str, float]]] = []
    tree_rows: list[tuple[str, str, list[tuple[float, float]], dict[str, float]]] = []
    lines_pending: list[tuple[str, str, list[list[float]], int | None, float | None]] = []
    # Tagged support heights by source coordinate.
    support_heights: dict[tuple[float, float], float] = {}
    skipped["underground_cable"] = 0
    for index, feature in enumerate(layer.features):
        raw = feature.get("properties")
        properties: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            skipped["untagged"] += 1
            continue
        base_id = _id(properties, index)
        located = _points(geometry)
        natural = _text(properties, "natural")
        power = _text(properties, "power")
        if located and natural == "tree":
            for part, (x, y) in enumerate(located):
                lon, lat = to_wgs84.transform(x, y)
                part_id = base_id if part == 0 else f"{base_id}#{part}"
                tree_points.append(
                    (part_id, tree_species(properties), (lon, lat), tree_sizes(properties))
                )
            continue
        if located and power in ("tower", "pole"):
            height = parse_length_m(properties.get("height"))
            if height is not None:
                for x, y in located:
                    support_heights[(round(x, 9), round(y, 9))] = height
            continue
        if power == "cable":
            skipped["underground_cable"] += 1
            continue
        if power in ("line", "minor_line") and _lines(geometry):
            cables_text = _text(properties, "cables").split(";")[0]
            cables = int(cables_text) if cables_text.isdigit() and int(cables_text) > 0 else None
            for part, line_points in enumerate(_lines(geometry)):
                part_id = base_id if part == 0 else f"{base_id}#{part}"
                lines_pending.append((part_id, power, line_points, cables, _voltage(properties)))
            continue
        if natural == "tree_row" and _lines(geometry):
            for part, line_points in enumerate(_lines(geometry)):
                lonlat_line = [to_wgs84.transform(x, y) for x, y in line_points]
                part_id = base_id if part == 0 else f"{base_id}#{part}"
                tree_rows.append(
                    (part_id, tree_species(properties), lonlat_line, tree_sizes(properties))
                )
            continue
        areas = _areas(geometry)
        lines = _lines(geometry)
        cover = area_class(properties) if areas else None
        line = line_class(properties) if lines else None
        if cover is None and line is None:
            skipped["untagged"] += 1
            continue
        if line is not None and (_yes(properties, "tunnel") or _yes(properties, "bridge")):
            skipped["tunnel_or_bridge"] += 1
            continue
        if cover is not None:
            for part, (ring, holes) in enumerate(areas):
                holes_ignored += holes
                lonlat = [tuple(to_wgs84.transform(x, y)) for x, y in ring]
                part_id = base_id if part == 0 else f"{base_id}#{part}"
                rank = LAYER_ORDER.index(CLASS_LAYER[cover])
                pending.append((rank, len(pending), part_id, cover, lonlat, None))  # type: ignore[arg-type]
        if line is not None:
            cover_class, width = line
            for part, points in enumerate(lines):
                lonlat = [tuple(to_wgs84.transform(x, y)) for x, y in points]
                part_id = base_id if part == 0 else f"{base_id}#{part}"
                rank = LAYER_ORDER.index(CLASS_LAYER[cover_class])
                pending.append((rank, len(pending), part_id, cover_class, lonlat, width))  # type: ignore[arg-type]
    if not (pending or tree_points or tree_rows or lines_pending):
        raise ValueError(f"{geojson_path}: no land-cover, tree or power-line features")
    if origin is None:
        everything: list[tuple[float, float]] = [p for *_, ps, _ in pending for p in ps]
        everything += [p for _, _, p, _ in tree_points]
        everything += [p for _, _, ps, _ in tree_rows for p in ps]
        everything += [to_wgs84.transform(x, y) for _, _, ps, _, _ in lines_pending for x, y in ps]
        lons = [lon for lon, _ in everything]
        lats = [lat for _, lat in everything]
        origin = (round(0.5 * (min(lats) + max(lats)), 7), round(centre_longitude(lons), 7))
    enu_origin = ENUOrigin(latitude_deg=origin[0], longitude_deg=origin[1], altitude_m=0.0)

    def local(points: Sequence[tuple[float, float]]) -> list[list[float]]:
        return [
            [round(float(v), 3) for v in wgs84_to_enu(lat, lon, 0.0, enu_origin)[:2]]
            for lon, lat in points
        ]

    land_cover: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for _, _, feature_id, cover, source, line_width in sorted(pending, key=lambda p: (p[0], p[1])):
        xy = local(source)
        shapes = (
            ([xy] if len(xy) >= 3 and abs(_area(xy)) > 0.0 else [])
            if line_width is None
            else segments(xy)
        )
        if not shapes:
            skipped["degenerate"] += 1
            continue
        for k, shape in enumerate(shapes):
            entry: dict[str, object] = {
                "id": feature_id if len(shapes) == 1 else f"{feature_id}/{k}",
                "class": cover,
            }
            rounded = [[round(x, 3), round(y, 3)] for x, y in shape]
            if line_width is None:
                entry["polygon_xy_m"] = rounded
            else:
                entry["line_xy_m"] = rounded
                entry["width_m"] = line_width
            land_cover.append(entry)
        counts[cover] = counts.get(cover, 0) + 1

    trees: list[dict[str, object]] = []
    for tree_id, species, point, sizes in tree_points:
        trees.append({"id": tree_id, "xy_m": local([point])[0], "species": species, **sizes})
    for row_id, species, line_lonlat, sizes in tree_rows:
        for k, row_xy in enumerate(along(local(line_lonlat), TREE_ROW_SPACING_M)):
            trees.append(
                {
                    "id": f"{row_id}/{k}",
                    "xy_m": [round(v, 3) for v in row_xy],
                    "species": species,
                    **sizes,
                }
            )
    power_lines: list[dict[str, object]] = []
    supports_with_height = 0
    for line_id, kind, line_points, cables, voltage in lines_pending:
        supports: list[dict[str, object]] = []
        for x, y in line_points:
            xy_m = local([to_wgs84.transform(x, y)])[0]
            if supports and supports[-1]["xy_m"] == xy_m:
                continue
            support: dict[str, object] = {"xy_m": xy_m}
            height = support_heights.get((round(x, 9), round(y, 9)))
            if height is not None:
                support["height_m"] = height
                supports_with_height += 1
            supports.append(support)
        if len(supports) < 2:
            skipped["degenerate"] += 1
            continue
        power_line: dict[str, object] = {"id": line_id, "kind": kind, "supports": supports}
        if cables is not None:
            power_line["cables"] = cables
        if voltage is not None:
            power_line["voltage_v"] = voltage
        power_lines.append(power_line)

    return {
        "format": FORMAT,
        "version": VERSION,
        "georeference": {
            "source_crs_id": source_crs_id,
            "runtime_crs_id": "local-enu",
            "origin_lat_deg": origin[0],
            "origin_lon_deg": origin[1],
            "origin_h_m": 0.0,
        },
        "land_cover": land_cover,
        "trees": trees,
        "power_lines": power_lines,
        "source_sha256": source_sha256,
        "summary": {
            "features_by_class": dict(sorted(counts.items())),
            "polygons": len(land_cover),
            "trees": len(trees),
            "power_lines": len(power_lines),
            "power_line_supports_with_height": supports_with_height,
            "skipped": skipped,
            "holes_ignored": holes_ignored,
        },
    }


def _area(points: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, [*points[1:], points[0]], strict=True)
    )


def write_features(document: Mapping[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
