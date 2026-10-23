"""Overpass API JSON (`out geom`) to GeoJSON, for the R5 importers.

Offline tooling: the importers read GeoJSON with OSM's tags as properties,
and Overpass answers in its own JSON. This converts one answer:

* a tagged node becomes a Point (untagged nodes are a way's vertices);
* a way becomes a Polygon when it is closed and its tags make it an area
  (`building`, `landuse`, `leisure`, `amenity`, `water`, most `natural`, or
  `area=yes`), otherwise a LineString;
* a `type=multipolygon` relation becomes a MultiPolygon. Its outer ways are
  stitched end to end into rings, and inner ways are kept as holes of the
  ring they fall in.

Every feature carries `@id` (`node/1`, `way/2`, `relation/3`) so the
importers name what they made. Coordinates are WGS84 (EPSG:4326), as
Overpass gives them. Anything that cannot be built (an unclosed outer ring,
a way with under two points) is skipped and counted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# Natural features that are lines, not areas, even when closed.
LINEAR_NATURAL = frozenset({"tree_row", "coastline", "cliff", "ridge", "arete"})
AREA_KEYS = ("building", "landuse", "leisure", "amenity", "water")


def is_area(tags: Mapping[str, str]) -> bool:
    if tags.get("area") == "yes":
        return True
    if tags.get("area") == "no":
        return False
    if any(key in tags for key in AREA_KEYS):
        return True
    natural = tags.get("natural")
    return natural is not None and natural not in LINEAR_NATURAL


def _coords(geometry: Sequence[Mapping[str, float]]) -> list[list[float]]:
    return [[float(p["lon"]), float(p["lat"])] for p in geometry]


def stitch(parts: list[list[list[float]]]) -> tuple[list[list[list[float]]], int]:
    """Join way geometries end to end into closed rings; return the rings
    and how many parts could not be closed into one."""
    pending = [list(part) for part in parts if len(part) >= 2]
    rings: list[list[list[float]]] = []
    unclosed = 0
    while pending:
        ring = pending.pop(0)
        progress = True
        while ring[0] != ring[-1] and progress:
            progress = False
            for i, part in enumerate(pending):
                if part[0] == ring[-1]:
                    ring += part[1:]
                elif part[-1] == ring[-1]:
                    ring += part[::-1][1:]
                elif part[-1] == ring[0]:
                    ring = part[:-1] + ring
                elif part[0] == ring[0]:
                    ring = part[::-1][:-1] + ring
                else:
                    continue
                pending.pop(i)
                progress = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
        else:
            unclosed += 1
    return rings, unclosed


def _inside(point: Sequence[float], ring: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:], strict=False):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def to_geojson(answer: Mapping[str, object]) -> dict[str, object]:
    """The FeatureCollection for an Overpass `out geom` answer."""
    features: list[dict[str, object]] = []
    skipped = 0
    elements = answer.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("an Overpass answer's `elements` must be a list")
    for element in elements:
        kind = element.get("type")
        tags = dict(element.get("tags") or {})
        properties = {"@id": f"{kind}/{element.get('id')}", **tags}
        if kind == "node":
            if not tags:
                continue
            geometry: dict[str, object] = {
                "type": "Point",
                "coordinates": [float(element["lon"]), float(element["lat"])],
            }
        elif kind == "way":
            points = _coords(element.get("geometry") or [])
            if len(points) < 2:
                skipped += 1
                continue
            closed = points[0] == points[-1] and len(points) >= 4
            if closed and is_area(tags):
                geometry = {"type": "Polygon", "coordinates": [points]}
            else:
                geometry = {"type": "LineString", "coordinates": points}
        elif kind == "relation" and tags.get("type") == "multipolygon":
            members = element.get("members") or []
            outer = [_coords(m.get("geometry") or []) for m in members if m.get("role") == "outer"]
            inner = [_coords(m.get("geometry") or []) for m in members if m.get("role") == "inner"]
            outers, unclosed_outer = stitch(outer)
            inners, unclosed_inner = stitch(inner)
            # An inner ring that cannot be closed would fill an island or a
            # courtyard: the relation is skipped, never drawn without it.
            if not outers or unclosed_outer or unclosed_inner:
                skipped += 1
                continue
            polygons: list[list[list[list[float]]]] = [[ring] for ring in outers]
            for hole in inners:
                for polygon in polygons:
                    if _inside(hole[0], polygon[0]):
                        polygon.append(hole)
                        break
            geometry = {"type": "MultiPolygon", "coordinates": polygons}
        else:
            continue
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})
    return {
        "type": "FeatureCollection",
        "features": features,
        "argusnet_overpass": {"skipped": skipped, "features": len(features)},
    }


def convert(answer_path: str | Path, out_path: str | Path) -> dict[str, object]:
    collection = to_geojson(json.loads(Path(answer_path).read_text(encoding="utf-8")))
    Path(out_path).write_text(json.dumps(collection), encoding="utf-8")
    return collection
