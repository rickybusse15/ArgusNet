#!/usr/bin/env python3
"""Import building footprints for a run (R1): GeoJSON/OSM -> argusnet.buildings.

    python3 scripts/import_buildings.py town.geojson --out town.buildings.json
    python3 scripts/import_buildings.py town.geojson --out b.json --origin 47.6,-122.3

Then name the output in a scenario config's `buildings_path` option. See
`argusnet.world.buildings_import` for the height and roof rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argusnet.world.buildings_import import import_buildings, write_buildings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("geojson", type=Path, help="footprints, a GeoJSON FeatureCollection")
    parser.add_argument("--out", type=Path, required=True, help="the argusnet.buildings file")
    parser.add_argument(
        "--origin",
        help="the local frame's origin as LAT,LON in degrees (default: the footprints' centre)",
    )
    parser.add_argument(
        "--source-crs", help="the footprints' CRS when the file does not name one (EPSG:4326)"
    )
    args = parser.parse_args(argv)
    origin = None
    if args.origin:
        lat, lon = (float(v) for v in args.origin.split(","))
        origin = (lat, lon)
    document = import_buildings(args.geojson, origin=origin, source_crs=args.source_crs)
    write_buildings(document, args.out)
    print(json.dumps(document["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
