#!/usr/bin/env python3
"""Import land cover for a run (R5): GeoJSON/OSM -> argusnet.features.

    python3 scripts/import_features.py town.geojson --out town.features.json --origin 47.6,-122.3

Use the same --origin as the other importers so the layers line up, then
name the output in a scenario config's `features_path` option. See
`argusnet.world.features_import` for which tags become which land cover.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argusnet.world.features_import import import_features, write_features  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("geojson", type=Path, help="features, a GeoJSON FeatureCollection")
    parser.add_argument("--out", type=Path, required=True, help="the argusnet.features file")
    parser.add_argument(
        "--origin",
        help="the local frame's origin as LAT,LON in degrees (default: the features' centre)",
    )
    parser.add_argument("--source-crs", help="the file's CRS when it does not say")
    args = parser.parse_args(argv)
    origin = None
    if args.origin:
        lat, lon = (float(v) for v in args.origin.split(","))
        origin = (lat, lon)
    document = import_features(args.geojson, origin=origin, source_crs=args.source_crs)
    write_features(document, args.out)
    print(json.dumps({"out": str(args.out), **document["summary"]}, indent=2))  # type: ignore[dict-item]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
