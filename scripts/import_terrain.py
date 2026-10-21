#!/usr/bin/env python3
"""Import a DEM for a run (R5): GeoTIFF -> argusnet.terrain.

    python3 scripts/import_terrain.py dem.tif --map-preset medium --out t.json
    python3 scripts/import_terrain.py dem.tif --map-preset medium --out t.json \\
        --origin 47.6,-122.3

Use the same --origin as import_buildings.py so the layers line up, then
name the output in a scenario config's `terrain_path` option (with the same
`map_preset`). See `argusnet.world.terrain_import` for how the DEM is
placed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argusnet.world.terrain_import import import_terrain, write_terrain  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dem", type=Path, help="the DEM, a single-band GeoTIFF")
    parser.add_argument("--out", type=Path, required=True, help="the argusnet.terrain file")
    parser.add_argument(
        "--map-preset", required=True, help="the run's map preset, whose grid the DEM fills"
    )
    parser.add_argument(
        "--origin",
        help="the local frame's origin as LAT,LON in degrees (default: the DEM's centre)",
    )
    parser.add_argument(
        "--source-crs", help="the DEM's CRS when the file does not say, e.g. EPSG:32611"
    )
    args = parser.parse_args(argv)
    origin = None
    if args.origin:
        lat, lon = (float(v) for v in args.origin.split(","))
        origin = (lat, lon)
    document = import_terrain(
        args.dem, map_preset=args.map_preset, origin=origin, source_crs=args.source_crs
    )
    write_terrain(document, args.out)
    print(json.dumps({"out": str(args.out), **document["summary"]}, indent=2))  # type: ignore[dict-item]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
