"""Real buildings for a run (R1): GeoJSON or OSM footprints to `argusnet.buildings`.

Offline tooling. It reads a GeoJSON FeatureCollection of building footprints
(OSM exports carry the tags as properties) and writes an `argusnet.buildings`
v1 document, which the native builder's `buildings_path` option places in a
run as `building-v2` obstacles.

Georeferencing first. Footprints come in a source CRS (EPSG:4326 unless the
file names another) and are projected to the run's local east-north-up metres
around a geodetic origin, by default the centre of the footprints' extent.
The document records the source CRS and the origin, so the run says where on
Earth it stands; the same transform serves R2, R3 and R5.

Heights and roofs follow the data and say when they do not:

* height: `height` (metres; "12", "12 m", "40 ft"), else `building:levels`
  (plus `roof:levels`) x 3 m, else a default by the `building` tag. Marked
  `measured`, `levels` or `default`.
* roof: `roof:shape` when it is flat, gabled, hipped or pyramidal. Any other
  shape, or a hipped or pyramidal roof over a footprint that is not convex,
  is written flat at the full height: it over-states the solid, never
  under-states it. Marked `tagged`, `default` or `unsupported:<why>`.
* roof height: `roof:height`, else `roof:levels` x 3 m, else 40 % of the
  height up to 3 m.
* materials: `building:material` and `roof:material` sorted into the classes
  the viewer draws (facade: concrete, brick, glass, metal, wood, stone,
  plaster; roof: tiles, slate, metal, concrete, membrane, glass, green), and
  `generic` when the data does not say: nothing is drawn as brick that the
  data did not call brick.
* levels: `building:levels`, kept so the viewer can draw window rows; none
  are drawn on a building whose levels the data does not give.
* ridge: `roof:direction` (the way the slope faces, compass degrees or
  letters) gives the ridge square to it; `roof:orientation=across` turns it
  across the longest edge; otherwise the builder runs it along the longest
  edge.

Validation: every footprint's planar area after projection is compared with
its geodesic area on the WGS84 ellipsoid, and the worst relative error is in
the summary.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from argusnet.core.frames import ENUOrigin, wgs84_to_enu
from argusnet.core.io_limits import sha256_file

from ._scene_gis import lazy_import_pyproj, load_geojson_layer

FORMAT = "argusnet.buildings"
VERSION = 1
LEVEL_HEIGHT_M = 3.0
ROOF_SHAPES = ("flat", "gabled", "hipped", "pyramidal")
DEFAULT_HEIGHT_BY_TAG = {
    "house": 8.0,
    "detached": 8.0,
    "semidetached_house": 8.0,
    "terrace": 9.0,
    "residential": 10.0,
    "apartments": 15.0,
    "commercial": 12.0,
    "retail": 8.0,
    "office": 15.0,
    "industrial": 10.0,
    "warehouse": 10.0,
    "garage": 3.0,
    "garages": 3.0,
    "shed": 3.0,
    "church": 15.0,
    "school": 10.0,
}
DEFAULT_HEIGHT_M = 10.0
FACADE_CLASSES = {
    "concrete": "concrete",
    "reinforced_concrete": "concrete",
    "cement_block": "concrete",
    "brick": "brick",
    "glass": "glass",
    "mirror": "glass",
    "metal": "metal",
    "steel": "metal",
    "aluminium": "metal",
    "wood": "wood",
    "timber_framing": "wood",
    "stone": "stone",
    "sandstone": "stone",
    "limestone": "stone",
    "plaster": "plaster",
    "render": "plaster",
}
ROOF_CLASSES = {
    "roof_tiles": "tiles",
    "tiles": "tiles",
    "tile": "tiles",
    "clay": "tiles",
    "slate": "slate",
    "metal": "metal",
    "metal_sheet": "metal",
    "tin": "metal",
    "copper": "metal",
    "zinc": "metal",
    "concrete": "concrete",
    "tar_paper": "membrane",
    "asphalt": "membrane",
    "bitumen": "membrane",
    "glass": "glass",
    "grass": "green",
    "plants": "green",
}
COMPASS = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def parse_length_m(value: object) -> float | None:
    """A length tag in metres: "12", "12.5 m", "40 ft", "40'". None if unreadable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) and value > 0 else None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(m|ft|'|)\s*", str(value))
    if not match:
        return None
    number = float(match.group(1))
    if match.group(2) in ("ft", "'"):
        number *= 0.3048
    return number if number > 0 else None


def parse_levels(value: object) -> float | None:
    try:
        levels = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return levels if levels > 0 and math.isfinite(levels) else None


def is_convex(points: Sequence[Sequence[float]]) -> bool:
    n = len(points)
    sign = 0.0
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) <= 1e-12:
            continue
        if sign == 0.0:
            sign = math.copysign(1.0, cross)
        elif math.copysign(1.0, cross) != sign:
            return False
    return True


def planar_area(points: Sequence[Sequence[float]]) -> float:
    n = len(points)
    return 0.5 * abs(
        sum(
            points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
    )


def longest_edge_azimuth(points: Sequence[Sequence[float]]) -> float:
    n = len(points)
    best = (0.0, 0.0)
    for i in range(n):
        p, q = points[i], points[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        if length > best[0]:
            best = (length, math.atan2(q[1] - p[1], q[0] - p[0]))
    return best[1]


def ridge_azimuth_rad(properties: Mapping[str, object], footprint) -> float | None:
    """The ridge's direction, radians from +x (east), when the tags say."""
    direction = properties.get("roof:direction")
    if direction is not None:
        text = str(direction).strip().upper()
        degrees = COMPASS.get(text)
        if degrees is None:
            try:
                degrees = float(text)
            except ValueError:
                degrees = None
        if degrees is not None:
            # The slope faces compass `degrees` (0 north, clockwise): that is
            # pi/2 - d from east, and the ridge runs square to it.
            facing = math.pi / 2.0 - math.radians(degrees)
            return math.remainder(facing + math.pi / 2.0, math.pi)
    if str(properties.get("roof:orientation", "")).strip() == "across":
        return math.remainder(longest_edge_azimuth(footprint) + math.pi / 2.0, math.pi)
    return None


def centre_longitude(lons: Sequence[float]) -> float:
    """The middle of the footprints' longitudes, in [-180, 180).

    Unwrapped around the first longitude, so a town astride the antimeridian
    (179.999 and -179.999) centres at 180, not at 0 on the far side of the
    Earth. Footprints span far less than half the globe, so unwrapping each
    to within 180 degrees of the first keeps them contiguous.
    """
    first = lons[0]
    unwrapped = [first + math.remainder(lon - first, 360.0) for lon in lons]
    middle = 0.5 * (min(unwrapped) + max(unwrapped))
    return (middle + 180.0) % 360.0 - 180.0


def _ring(coordinates: Sequence[Sequence[float]]) -> list[list[float]]:
    ring = [list(map(float, point[:2])) for point in coordinates]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    deduped: list[list[float]] = []
    for point in ring:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return deduped


def _polygons(geometry: Mapping[str, object]) -> list[tuple[list[list[float]], int]]:
    """Each polygon's outer ring in source coordinates, with its hole count."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if kind == "Polygon":
        return [(_ring(coordinates[0]), len(coordinates) - 1)] if coordinates else []
    if kind == "MultiPolygon":
        return [(_ring(polygon[0]), len(polygon) - 1) for polygon in coordinates if polygon]
    return []


def _height(properties: Mapping[str, object]) -> tuple[float, str]:
    measured = parse_length_m(properties.get("height"))
    if measured is not None:
        return measured, "measured"
    levels = parse_levels(properties.get("building:levels"))
    if levels is not None:
        roof_levels = parse_levels(properties.get("roof:levels")) or 0.0
        return (levels + roof_levels) * LEVEL_HEIGHT_M, "levels"
    tag = str(properties.get("building", "")).strip()
    return DEFAULT_HEIGHT_BY_TAG.get(tag, DEFAULT_HEIGHT_M), "default"


def _roof(properties: Mapping[str, object], height_m: float, footprint) -> tuple[str, float, str]:
    """(shape, roof height, source)."""
    raw = properties.get("roof:shape")
    if raw is None:
        return "flat", 0.0, "default"
    shape = str(raw).strip().lower()
    if shape not in ROOF_SHAPES:
        return "flat", 0.0, f"unsupported:{shape}"
    if shape in ("hipped", "pyramidal") and not is_convex(footprint):
        return "flat", 0.0, f"unsupported:non-convex-{shape}"
    if shape == "flat":
        return "flat", 0.0, "tagged"
    roof_height = parse_length_m(properties.get("roof:height"))
    if roof_height is None:
        roof_levels = parse_levels(properties.get("roof:levels"))
        roof_height = roof_levels * LEVEL_HEIGHT_M if roof_levels else min(3.0, 0.4 * height_m)
    # The walls keep at least half the height.
    return shape, min(roof_height, 0.5 * height_m), "tagged"


def import_buildings(
    geojson_path: str | Path,
    *,
    origin: tuple[float, float] | None = None,
    source_crs: str | None = None,
) -> dict[str, object]:
    """The `argusnet.buildings` document for the footprints in `geojson_path`.

    `origin` is the local frame's (latitude, longitude) in degrees; the
    centre of the footprints' extent when not given.
    """
    CRS, Transformer = lazy_import_pyproj()
    from pyproj import Geod

    source_sha256 = sha256_file(geojson_path)
    layer = load_geojson_layer(geojson_path, default_source_crs_id=source_crs)
    source_crs_id = source_crs or layer.source_crs_id
    to_wgs84 = Transformer.from_crs(
        CRS.from_user_input(source_crs_id), CRS.from_epsg(4326), always_xy=True
    )
    features: list[tuple[int, int, Mapping[str, object], list[tuple[float, float]], int]] = []
    for index, feature in enumerate(layer.features):
        # The layer types its features loosely; narrow them here.
        raw_properties = feature.get("properties")
        properties: Mapping[str, object] = (
            raw_properties if isinstance(raw_properties, Mapping) else {}
        )
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
        for part, (ring, holes) in enumerate(_polygons(geometry)):
            lonlat = [to_wgs84.transform(x, y) for x, y in ring]
            features.append((index, part, properties, lonlat, holes))
    if not features:
        raise ValueError(f"{geojson_path}: no building polygons")
    if origin is None:
        lons = [lon for *_, lonlat, _ in features for lon, _ in lonlat]
        lats = [lat for *_, lonlat, _ in features for _, lat in lonlat]
        origin = (round(0.5 * (min(lats) + max(lats)), 7), round(centre_longitude(lons), 7))
    enu_origin = ENUOrigin(latitude_deg=origin[0], longitude_deg=origin[1], altitude_m=0.0)
    geod = Geod(ellps="WGS84")

    buildings = []
    skipped = 0
    holes_ignored = 0
    worst_area_error = 0.0
    for index, part, properties, lonlat, holes in features:
        footprint = [
            [float(v) for v in wgs84_to_enu(lat, lon, 0.0, enu_origin)[:2]] for lon, lat in lonlat
        ]
        if len(footprint) < 3 or planar_area(footprint) <= 0.0:
            skipped += 1
            continue
        holes_ignored += holes
        geodesic_area, _ = geod.polygon_area_perimeter(
            [lon for lon, _ in lonlat], [lat for _, lat in lonlat]
        )
        geodesic_area = abs(geodesic_area)
        if geodesic_area > 0.0:
            worst_area_error = max(
                worst_area_error, abs(planar_area(footprint) - geodesic_area) / geodesic_area
            )
        height_m, height_source = _height(properties)
        shape, roof_height_m, roof_source = _roof(properties, height_m, footprint)
        base_id = str(
            properties.get("@id")
            or properties.get("id")
            or properties.get("osm_id")
            or f"building-{index}"
        )
        building = {
            "id": base_id if part == 0 else f"{base_id}#{part}",
            "footprint_xy_m": footprint,
            "height_m": height_m,
            "eave_height_m": height_m - roof_height_m,
            "roof_shape": shape,
            "height_source": height_source,
            "roof_source": roof_source,
        }
        facade = FACADE_CLASSES.get(str(properties.get("building:material", "")).strip().lower())
        roof_material = ROOF_CLASSES.get(str(properties.get("roof:material", "")).strip().lower())
        building["facade_material"] = facade or "generic"
        building["roof_material"] = roof_material or "generic"
        levels = parse_levels(properties.get("building:levels"))
        if levels is not None:
            building["levels"] = levels
        if shape == "gabled":
            azimuth = ridge_azimuth_rad(properties, footprint)
            if azimuth is not None:
                building["ridge_azimuth_rad"] = azimuth
        buildings.append(building)

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
        "source_sha256": source_sha256,
        "summary": {
            "buildings": len(buildings),
            "skipped_degenerate": skipped,
            "holes_ignored": holes_ignored,
            "estimated_heights": sum(1 for b in buildings if b["height_source"] != "measured"),
            "roofs_written_flat": sum(
                1 for b in buildings if str(b["roof_source"]).startswith("unsupported")
            ),
            "worst_footprint_area_error": worst_area_error,
        },
        "buildings": buildings,
    }


def write_buildings(document: Mapping[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


__all__ = ["import_buildings", "write_buildings", "parse_length_m", "ridge_azimuth_rad"]
