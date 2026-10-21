"""One real-world import (R5): a DEM and an OSM extract to a runnable world.

Offline tooling that runs the three importers with one origin and writes
what a run needs into one folder:

* `terrain.json` (`argusnet.terrain`): the DEM on the map preset's grid;
* `buildings.json` (`argusnet.buildings`): the extract's buildings, when it
  has any;
* `features.json` (`argusnet.features`): its land cover, trees and power
  lines, when it has any;
* `scenario.json`: a scenario config naming all three, with trees on;
* `world.json` (`argusnet.world-import`): the manifest. It holds the origin,
  each file's SHA-256 and the sources', each importer's summary, and a
  `world_id` hashed from the files' hashes, so two imports of the same data
  are the same world.

The origin is the one given, or the DEM's centre. Every file is imported
around it, so the builder's one-origin check holds by construction. The
extract is split by the `building` tag: features with one (other than
`no`) go to the buildings importer, which reads every polygon it is given
as a building; everything goes to the features importer, which skips
buildings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .buildings_import import import_buildings, write_buildings
from .features_import import import_features, write_features
from .terrain_import import import_terrain, write_terrain

FORMAT = "argusnet.world-import"
VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_building(properties: Mapping[str, object]) -> bool:
    value = str(properties.get("building", "")).strip().lower()
    return value not in ("", "no")


def split_buildings(osm_path: str | Path, out_path: Path) -> int:
    """Write the extract's building features to `out_path`, keeping its CRS
    member; return how many there are."""
    document = json.loads(Path(osm_path).read_text(encoding="utf-8"))
    features = document.get("features", []) if document.get("type") == "FeatureCollection" else []
    buildings = [
        feature
        for feature in features
        if isinstance(feature, Mapping)
        and isinstance(feature.get("properties"), Mapping)
        and is_building(feature["properties"])
    ]
    split: dict[str, object] = {"type": "FeatureCollection", "features": buildings}
    if "crs" in document:
        split["crs"] = document["crs"]
    out_path.write_text(json.dumps(split), encoding="utf-8")
    return len(buildings)


def import_world(
    dem_path: str | Path,
    osm_path: str | Path,
    out_dir: str | Path,
    *,
    map_preset: str,
    origin: tuple[float, float] | None = None,
    dem_crs: str | None = None,
    osm_crs: str | None = None,
    terrain_preset: str = "urban_flat",
    seed: int = 7,
    duration_s: float = 120.0,
) -> dict[str, object]:
    """Import `dem_path` and `osm_path` into `out_dir` around one origin and
    return the manifest (also written as `world.json`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    terrain = import_terrain(dem_path, map_preset=map_preset, origin=origin, source_crs=dem_crs)
    georeference = terrain["georeference"]
    assert isinstance(georeference, Mapping)
    shared = (float(georeference["origin_lat_deg"]), float(georeference["origin_lon_deg"]))
    files: dict[str, Path] = {"terrain": out / "terrain.json"}
    write_terrain(terrain, files["terrain"])
    summaries: dict[str, object] = {"terrain": terrain["summary"]}

    building_source = out / "osm-buildings.geojson"
    if split_buildings(osm_path, building_source):
        buildings = import_buildings(building_source, origin=shared, source_crs=osm_crs)
        files["buildings"] = out / "buildings.json"
        write_buildings(buildings, files["buildings"])
        summaries["buildings"] = buildings["summary"]
    building_source.unlink()

    try:
        features = import_features(osm_path, origin=shared, source_crs=osm_crs)
    except ValueError as error:
        if "no land-cover" not in str(error):
            raise
        features = None
    if features is not None:
        files["features"] = out / "features.json"
        write_features(features, files["features"])
        summaries["features"] = features["summary"]

    options: dict[str, object] = {
        "map_preset": map_preset,
        "terrain_preset": terrain_preset,
        "terrain_path": str(files["terrain"].resolve()),
        "trees": True,
    }
    for name in ("buildings", "features"):
        if name in files:
            options[f"{name}_path"] = str(files[name].resolve())
    scenario = {
        "format": "argusnet.scenario-config",
        "version": 1,
        "seed": seed,
        "duration_s": duration_s,
        "options": options,
    }
    (out / "scenario.json").write_text(json.dumps(scenario, indent=1) + "\n", encoding="utf-8")

    hashes = {name: _sha256(path) for name, path in files.items()}
    world_id = hashlib.sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes)).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "format": FORMAT,
        "version": VERSION,
        "world_id": world_id,
        "origin": {"latitude_deg": shared[0], "longitude_deg": shared[1]},
        "map_preset": map_preset,
        "files": {
            name: {"path": path.name, "sha256": hashes[name]} for name, path in files.items()
        },
        "sources": {"dem_sha256": _sha256(Path(dem_path)), "osm_sha256": _sha256(Path(osm_path))},
        "scenario": "scenario.json",
        "summaries": summaries,
    }
    (out / "world.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
