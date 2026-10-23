#!/usr/bin/env python3
"""Import a real-world place for a run (R5): a DEM and an OSM extract.

    python3 scripts/import_world.py dem.tif town.geojson --map-preset medium --out worlds/town
    python3 scripts/import_world.py dem.tif town.geojson --map-preset medium --out worlds/town \\
        --origin 47.6,-122.3 --argusnet-rs target/release/argusnet-rs

Writes terrain, buildings and features documents around one origin, a
scenario config naming them (`scenario.json`) and a manifest (`world.json`).
With --argusnet-rs it also runs the scenario (`replay.json`) and packages
its scene (`scene/`), which the viewer opens with --scene. See
`argusnet.world.world_import`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argusnet.world.world_import import import_world  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dem", type=Path, help="the DEM, a single-band GeoTIFF")
    parser.add_argument("osm", type=Path, help="the extract, a GeoJSON FeatureCollection")
    parser.add_argument("--out", type=Path, required=True, help="the folder to write")
    parser.add_argument("--map-preset", required=True, help="the run's map preset")
    parser.add_argument("--origin", help="LAT,LON in degrees (default: the DEM's centre)")
    parser.add_argument("--dem-crs", help="the DEM's CRS when the file does not say")
    parser.add_argument("--osm-crs", help="the extract's CRS when it does not say")
    parser.add_argument("--terrain-preset", default="urban_flat")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--argusnet-rs", type=Path, help="run the scenario and build its scene")
    args = parser.parse_args(argv)
    origin = None
    if args.origin:
        lat, lon = (float(v) for v in args.origin.split(","))
        origin = (lat, lon)
    manifest = import_world(
        args.dem,
        args.osm,
        args.out,
        map_preset=args.map_preset,
        origin=origin,
        dem_crs=args.dem_crs,
        osm_crs=args.osm_crs,
        terrain_preset=args.terrain_preset,
        seed=args.seed,
        duration_s=args.duration_s,
    )
    if args.argusnet_rs:
        replay = args.out / "replay.json"
        scene = args.out / "scene"
        subprocess.run(
            [
                str(args.argusnet_rs),
                "sim",
                "--quiet",
                "--config",
                str(args.out / "scenario.json"),
                "--replay",
                str(replay),
            ],
            check=True,
        )
        subprocess.run(
            [str(args.argusnet_rs), "build-scene", "--replay", str(replay), "--output", str(scene)],
            check=True,
        )
    print(json.dumps({"out": str(args.out), "world_id": manifest["world_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
