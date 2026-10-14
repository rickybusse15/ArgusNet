#!/usr/bin/env python3
"""Record the world `build_default_scenario` builds, across presets (P4.9b.1).

The Rust-native builder must produce the same `world` and `presentation` as
`export_bundle` of a Python-built scenario. The committed scenario bundles pin
one world -- the small alpine map, two obstacles. This fixture drives the same
calls `build_default_scenario` makes (`build_terrain_layer`, then
`build_default_environment`) over a matrix chosen to reach every branch:

* every terrain preset's `_preset_environment_profile` branch, including the
  unknown-name fall-through (`default`, `mountain_pass`);
* the map presets that add obstacle clusters (large, xlarge, regional) and the
  regional corridor walls;
* every land-cover patch -- urban rectangles, the river corridor, the coastal
  strip, and the lake ellipses (`np.cos`/`np.sin` over an angle array);
* `clean_terrain` (no obstacles, terrain-only patches, no vegetation), a
  winter month (snow), an explicit terrain seed and detail strength.

Recorded per case: the obstacle documents in full; the land-cover class and
density rasters as compact strings; the terrain summary; and every 37th
viewer-mesh height. Terrain heights differ from a Rust build by numpy's `exp`
(P4.3), so the Rust test compares heights and what derives from them within
that bound, and everything else exactly.

Usage:
    python3 scripts/capture_scenario_world_fixture.py --out tests/fixtures/scenario_world.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixture_platform import captured_on  # noqa: E402

from argusnet.simulation.scenario_bundle import _obstacles_document  # noqa: E402
from argusnet.simulation.sim import (  # noqa: E402
    build_default_environment,
    map_bounds_for_scale,
    map_bounds_object,
    map_scale_for_preset,
    terrain_resolution_for_scale,
)
from argusnet.world.procedural import TerrainBuildConfig, build_terrain_layer  # noqa: E402

MESH_STRIDE = 37

CASES = [
    dict(map_preset="small", terrain_preset=preset)
    for preset in (
        "alpine",
        "arctic_tundra",
        "coastal",
        "default",
        "desert_canyon",
        "jungle_canopy",
        "lake_district",
        "military_compound",
        "mountain_pass",
        "river_valley",
        "rolling_highlands",
        "urban_flat",
    )
] + [
    dict(map_preset="large", terrain_preset="military_compound"),
    dict(map_preset="xlarge", terrain_preset="rolling_highlands"),
    dict(map_preset="regional", terrain_preset="military_compound"),
    dict(map_preset="medium", terrain_preset="alpine", clean_terrain=True),
    dict(map_preset="small", terrain_preset="river_valley", clean_terrain=True),
    dict(
        map_preset="small",
        terrain_preset="lake_district",
        season_month=1,
        terrain_seed=42,
        detail_strength=0.5,
    ),
    # Seeds found by search for the landscape filter's rarer branches: the
    # hangar dropped for touching water alone (35), and for water and slope
    # together (1).
    dict(map_preset="small", terrain_preset="alpine", terrain_seed=35),
    dict(map_preset="small", terrain_preset="alpine", terrain_seed=1),
    # A forest stand dropped for standing at least half in water.
    dict(map_preset="small", terrain_preset="river_valley", terrain_seed=0),
    # The one searched seed whose tower has its lowest ground at the centre
    # sample, below the whole rim.
    dict(map_preset="small", terrain_preset="urban_flat", terrain_seed=33),
    # Walls and boxes at every remaining map scale.
    dict(map_preset="medium", terrain_preset="military_compound"),
    dict(map_preset="xlarge", terrain_preset="military_compound"),
    dict(map_preset="theater", terrain_preset="military_compound"),
    dict(map_preset="operational", terrain_preset="military_compound"),
    # A terrain finer than 5 m, where the land cover's 10 m floor applies.
    dict(map_preset="small", terrain_preset="alpine", terrain_resolution_m=2.5),
    # The dense block grid, at a scale of one and at a scale that multiplies
    # every block. These two were appended on linux-x86_64 to a fixture
    # captured on macos-aarch64; heights are compared within tolerance either
    # way, and a full recapture records them like the rest.
    dict(map_preset="small", terrain_preset="urban_dense"),
    dict(map_preset="medium", terrain_preset="urban_dense"),
]


def capture_geometry() -> dict:
    """Footprints from the reference's own classes over random shapes: the
    arithmetic (numpy's fused matmul, `np.linalg.norm`, the offset's
    association) checked far beyond the hand-placed preset obstacles."""
    import numpy as np

    from argusnet.world.obstacles import OrientedBox, WallSegment

    rng = np.random.default_rng(2026)
    boxes = []
    for index in range(300):
        cx, cy = (float(v) for v in rng.uniform(-5000.0, 5000.0, size=2))
        length, width = (float(v) for v in rng.uniform(0.5, 400.0, size=2))
        yaw = float(rng.uniform(-7.0, 7.0))
        box = OrientedBox(
            primitive_id=f"box-{index}",
            blocker_type="building",
            center_x_m=cx,
            center_y_m=cy,
            length_m=length,
            width_m=width,
            yaw_rad=yaw,
            base_z_m=0.0,
            top_z_m=1.0,
        )
        boxes.append(
            {"args": [cx, cy, length, width, yaw], "footprint": box.footprint_xy_m().tolist()}
        )
    walls = []
    for index in range(300):
        start = rng.uniform(-5000.0, 5000.0, size=2)
        end = start + rng.uniform(-900.0, 900.0, size=2)
        thickness = float(rng.uniform(0.1, 20.0))
        wall = WallSegment(
            primitive_id=f"wall-{index}",
            blocker_type="wall",
            start_xy_m=start,
            end_xy_m=end,
            thickness_m=thickness,
            base_z_m=0.0,
            top_z_m=1.0,
        )
        walls.append(
            {
                "args": [*start.tolist(), *end.tolist(), thickness],
                "footprint": wall.footprint_xy_m().tolist(),
            }
        )
    return {"boxes": boxes, "walls": walls}


def raster(layer, attribute: str) -> str:
    """A tiled uint8 raster, stitched and cropped, as hex -- two characters a
    cell, row by row (row = y)."""
    size = layer.tile_size_cells
    width = layer.bounds_xy_m.width_m
    height = layer.bounds_xy_m.height_m
    cols = max(int(-(-width // layer.base_resolution_m)), 1)
    rows = max(int(-(-height // layer.base_resolution_m)), 1)
    out = []
    for row in range(rows):
        for col in range(cols):
            tile = layer._tiles[(col // size, row // size)]
            out.append(f"{int(getattr(tile, attribute)[row % size, col % size]):02x}")
    return "".join(out)


def capture_case(case: dict) -> dict:
    map_preset = case["map_preset"]
    terrain_preset = case["terrain_preset"]
    clean_terrain = case.get("clean_terrain", False)
    season_month = case.get("season_month", 7)
    scale = map_scale_for_preset(map_preset)
    bounds = map_bounds_for_scale(scale)
    config = TerrainBuildConfig(
        terrain_source="procedural",
        terrain_preset=terrain_preset,
        terrain_seed=case.get("terrain_seed"),
        detail_strength=case.get("detail_strength", 1.0),
        terrain_resolution_m=case.get("terrain_resolution_m")
        or terrain_resolution_for_scale(scale),
        season_month=season_month,
    )
    terrain = build_terrain_layer(
        config,
        map_bounds_object(bounds),
        environment_id=f"procedural-{terrain_preset}",
    )
    environment = build_default_environment(
        scale,
        terrain,
        bounds,
        map_preset=map_preset,
        terrain_preset=terrain_preset,
        clean_terrain=clean_terrain,
        season_month=season_month,
    )
    land_cover = environment.land_cover
    mesh = environment.terrain.viewer_mesh()
    flat_mesh = [h for row in mesh["heights_m"] for h in row]
    return {
        "options": {
            "map_preset": map_preset,
            "terrain_preset": terrain_preset,
            "clean_terrain": clean_terrain,
            "season_month": season_month,
            "terrain_seed": case.get("terrain_seed"),
            "detail_strength": case.get("detail_strength", 1.0),
            "terrain_resolution_m": case.get("terrain_resolution_m"),
        },
        "environment_id": environment.environment_id,
        "obstacles": _obstacles_document(environment.obstacles),
        "land_cover": {
            "resolution": land_cover.base_resolution_m,
            "classes": raster(land_cover, "classes"),
            "density": raster(land_cover, "density"),
            "metadata": land_cover.to_metadata(),
        },
        "terrain_summary": json.loads(json.dumps(environment.terrain.terrain_summary())),
        "viewer_mesh": {
            "rows": mesh["rows"],
            "cols": mesh["cols"],
            "stride": MESH_STRIDE,
            "sample": flat_mesh[::MESH_STRIDE],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for case in CASES:
        captured = capture_case(case)
        kinds = sorted({p["kind"] for p in captured["obstacles"]["primitives"]})
        print(
            f"{case['map_preset']}/{case['terrain_preset']}"
            f"{' clean' if case.get('clean_terrain') else ''}: "
            f"{len(captured['obstacles']['primitives'])} obstacles {kinds}"
        )
        cases.append(captured)
    kinds = {p["kind"] for c in cases for p in c["obstacles"]["primitives"]}
    expected = {"box-v1", "cylinder-v1", "polygon-prism-v1", "wall-v1", "forest-stand-v1"}
    assert kinds == expected, kinds
    args.out.write_text(
        json.dumps(
            {
                "_source": "scripts/capture_scenario_world_fixture.py",
                "_captured_on": captured_on(),
                "cases": cases,
                "geometry": capture_geometry(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
