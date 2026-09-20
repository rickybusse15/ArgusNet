#!/usr/bin/env python3
"""Record build_terrain_layer and build_land_cover_layer end to end.

P4.3a-f each verified one piece against its own fixture, in isolation. This
one drives the two real builders and records what they produce, so the port is
checked on the thing nobody had tested yet: whether the pieces are WIRED UP
correctly.

FIVE SEAMS THIS PINS THAT NO EARLIER FIXTURE COULD

1. `effective_seed`. When `terrain_seed` is None the seed is
   `_stable_seed("terrain", preset)` -- a DIFFERENT derivation from the
   `_stable_seed("profile", preset, seed)` that `procedural_terrain_grid`
   applies to it, and different again from `_stable_seed("land-cover",
   preset, seed)` for the moisture field. Three stable seeds in one build,
   feeding each other. Get the order wrong and every scene is plausible and
   wrong.

2. The land-cover resolution is DERIVED, not configured:
   `max(10.0, terrain_resolution * 2.0)`. Terrain and cover are rasterised at
   different resolutions, which is what makes the mask resampler do real work.
   A port that shared one resolution passes every isolated fixture and builds
   the wrong world.

3. Heights and slopes come from the TILED field, not from the source grid.
   `TerrainLayer.from_height_grid` resamples into padded tiles, so reading a
   height back is not bit-identical to the grid it was built from. Sampling
   the source grid instead would be within a hair everywhere and wrong at
   every tile seam.

4. The gradient step. `slope_rad_at_many` uses
   `max(base_resolution * 0.5, 0.25)` -- while `TerrainModel.gradient_at`
   defaults to 0.5 and `slope_rad_at` to 1.0. Three steps for "the gradient"
   in one package; this records which one land cover actually gets.

5. `SeasonalVariation.from_month`. The table lives in environment.py, far from
   everything else here, and winter's 0.15 is the one factor whose product
   with 190 is not an integer.

Usage:
    python3 scripts/capture_landscape_fixture.py --out tests/fixtures/landscape.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.world.environment import Bounds2D  # noqa: E402
from argusnet.world.procedural import (  # noqa: E402
    TerrainBuildConfig,
    _stable_seed,
    build_land_cover_layer,
    build_terrain_layer,
)


def fstr(value: float) -> str:
    return repr(float(value))


def bits(values) -> str:
    return "".join("1" if bool(v) else "0" for v in values)


def scenes() -> list[dict]:
    """Chosen against the seams above, not by varying a seed."""
    return [
        # The default path: no explicit seed, so `effective_seed` is derived.
        # alpine's derived seed is what a port that forgets the derivation
        # cannot reproduce.
        dict(name="alpine_derived_seed", preset="alpine", month=7),
        # An EXPLICIT seed, so the derived branch is not the only one covered.
        dict(name="coastal_explicit_seed", preset="coastal", month=7, seed=4242),
        # Terrain at 20 m -> cover at max(10, 40) = 40. The multiplier binds.
        dict(
            name="cover_from_multiplier", preset="river_valley", month=7, terrain_resolution_m=20.0
        ),
        # Terrain at 3 m -> cover at max(10, 6) = 10. The FLOOR binds instead,
        # which is the other half of the rule and a different code path.
        dict(
            name="cover_from_floor", preset="rolling_highlands", month=7, terrain_resolution_m=3.0
        ),
        # Winter: foliage 0.15, snow on. The only season whose density scaling
        # is not a whole number.
        dict(name="winter_snow", preset="alpine", month=1),
        # Autumn, jungle: the highest vegetation_bias against a 0.4 factor.
        dict(name="jungle_autumn", preset="jungle_canopy", month=10),
        # A ground plane well above zero, so the water rule and the tile floor
        # both move off their defaults together.
        dict(name="raised_ground_plane", preset="lake_district", month=7, ground_plane_m=35.0),
        # MULTIPLE TILES, reached by shrinking the tile rather than growing
        # the grid. At 256 cells a tile spans 1 km at 4 m resolution, so a
        # multi-tile scene at the default would need a 2.4 km world and a
        # 4.7 MB fixture -- 240,000 mask cells per mask, six masks, to test one
        # seam. `tile_size_cells` is a config field with a floor of 2, so 32
        # cells gives 3.6 tiles across on the same small world. The seam is
        # what is under test, not the acreage.
        dict(
            name="multi_tile",
            preset="mountain_pass",
            month=7,
            terrain_resolution_m=4.0,
            tile_size_cells=32,
        ),
        # detail_strength away from 1.0, which scales the detail octave only.
        dict(name="low_detail", preset="desert_canyon", month=7, detail_strength=0.25),
    ]


def capture_scene(scene: dict) -> dict:
    preset = scene["preset"]
    month = int(scene["month"])
    x_min = float(scene.get("x_min_m", -120.0))
    y_min = float(scene.get("y_min_m", 75.0))
    bounds = Bounds2D(
        x_min_m=x_min,
        x_max_m=x_min + float(scene.get("width_m", 460.0)),
        y_min_m=y_min,
        y_max_m=y_min + float(scene.get("height_m", 300.0)),
    )
    config = TerrainBuildConfig(
        terrain_preset=preset,
        terrain_seed=scene.get("seed"),
        detail_strength=float(scene.get("detail_strength", 1.0)),
        terrain_resolution_m=scene.get("terrain_resolution_m"),
        season_month=month,
        tile_size_cells=int(scene.get("tile_size_cells", 256)),
        ground_plane_m=float(scene.get("ground_plane_m", 0.0)),
    )
    terrain = build_terrain_layer(config, bounds)

    # The derived rule, evaluated here exactly as build_landscape does.
    cover_resolution = float(max(10.0, terrain.base_resolution_m * 2.0))
    layer = build_land_cover_layer(
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=(),
        patches=(),
        resolution_m=cover_resolution,
        season=config.season,
        terrain_preset=preset,
        seed=config.effective_seed,
        suppress_vegetation=bool(scene.get("suppress_vegetation", False)),
        terrain_masks=terrain.semantic_masks,
    )

    cols = max(1, int(math.ceil(bounds.width_m / cover_resolution)))
    rows = max(1, int(math.ceil(bounds.height_m / cover_resolution)))
    x_values = bounds.x_min_m + ((np.arange(cols, dtype=float) + 0.5) * cover_resolution)
    y_values = bounds.y_min_m + ((np.arange(rows, dtype=float) + 0.5) * cover_resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    # Sampled through the TILED layer -- seam 3.
    heights = terrain.height_at_many(points)
    slopes = terrain.slope_rad_at_many(points)

    classes, density = grids_from_layer(layer, rows, cols)

    return {
        "name": scene["name"],
        "preset": preset,
        "season_month": month,
        "terrain_seed": scene.get("seed"),
        "effective_seed": int(config.effective_seed),
        "detail_strength": fstr(config.detail_strength),
        "terrain_resolution_m": fstr(terrain.base_resolution_m),
        "cover_resolution_m": fstr(cover_resolution),
        "tile_size_cells": int(config.tile_size_cells),
        "ground_plane_m": fstr(config.ground_plane_m),
        "suppress_vegetation": bool(scene.get("suppress_vegetation", False)),
        "season": {
            "foliage_density_factor": fstr(config.season.foliage_density_factor),
            "snow_cover": bool(config.season.snow_cover),
        },
        "bounds": {
            "x_min_m": fstr(bounds.x_min_m),
            "x_max_m": fstr(bounds.x_max_m),
            "y_min_m": fstr(bounds.y_min_m),
            "y_max_m": fstr(bounds.y_max_m),
        },
        "rows": rows,
        "cols": cols,
        "sampled_heights_m": [fstr(v) for v in np.asarray(heights).reshape(-1)],
        "sampled_slopes_rad": [fstr(v) for v in np.asarray(slopes).reshape(-1)],
        # Coverage fractions, not bitmaps. The full masks are already pinned
        # cell-for-cell by terrain_grid.json (P4.3e); repeating them here would
        # add megabytes to re-assert something proven. What is NOT proven
        # elsewhere is that these masks reach the classifier at all, and the
        # recorded `classes` below carry that -- a mis-wired resample changes
        # the map. The fractions are a cheap cross-check that the Rust side
        # generated the same masks before resampling them.
        "mask_coverage": {
            name: fstr(float(np.mean(np.asarray(mask, dtype=bool))))
            for name, mask in sorted(terrain.semantic_masks.items())
        },
        "mask_shape": {
            name: [int(np.asarray(mask).shape[0]), int(np.asarray(mask).shape[1])]
            for name, mask in sorted(terrain.semantic_masks.items())
        },
        "classes": [int(v) for v in classes.reshape(-1)],
        "density": [int(v) for v in density.reshape(-1)],
    }


def grids_from_layer(layer, rows: int, cols: int):
    size = layer.tile_size_cells
    classes = np.zeros((rows, cols), dtype=np.uint8)
    density = np.zeros((rows, cols), dtype=np.uint8)
    for (tx, ty), tile in layer._tiles.items():  # noqa: SLF001
        row_start, col_start = ty * size, tx * size
        row_end = min(row_start + size, rows)
        col_end = min(col_start + size, cols)
        classes[row_start:row_end, col_start:col_end] = tile.classes[
            : row_end - row_start, : col_end - col_start
        ]
        if tile.density is not None:
            density[row_start:row_end, col_start:col_end] = tile.density[
                : row_end - row_start, : col_end - col_start
            ]
    return classes, density


def check_discriminating(cases: list[dict]) -> None:
    """Refuse to write a fixture that proves nothing.

    Unlike the earlier slices there is no formula to mutate here -- this slice
    is wiring. So the guards assert that each SEAM is actually exercised by at
    least one scene, and that scenes differ where the seam says they must.
    """
    problems = []

    derived = [c for c in cases if c["terrain_seed"] is None]
    explicit = [c for c in cases if c["terrain_seed"] is not None]
    if not derived or not explicit:
        problems.append("need both a derived and an explicit terrain_seed")
    for case in derived:
        expected = _stable_seed("terrain", case["preset"])
        if case["effective_seed"] != expected:
            problems.append(f"{case['name']}: recorded seed is not the derived one")
        if case["effective_seed"] == 0:
            problems.append(f"{case['name']}: a zero seed would not discriminate")

    # Seam 2: both halves of max(10, res*2) must bind somewhere.
    multiplier = [c for c in cases if float(c["cover_resolution_m"]) > 10.0]
    floor = [
        c
        for c in cases
        if float(c["cover_resolution_m"]) == 10.0 and float(c["terrain_resolution_m"]) * 2.0 < 10.0
    ]
    if not multiplier:
        problems.append("no scene where the *2 multiplier binds")
    if not floor:
        problems.append("no scene where the 10 m floor binds")

    # Seam 3: at least one scene must span more than one tile, or the tiling
    # round-trip is never exercised at a seam.
    spans = []
    for case in cases:
        res = float(case["terrain_resolution_m"])
        width = float(case["bounds"]["x_max_m"]) - float(case["bounds"]["x_min_m"])
        spans.append(width / res / case["tile_size_cells"])
    if max(spans) <= 1.0:
        problems.append("every scene fits in a single tile; tiling is untested")

    # Seam 5: seasons must actually differ.
    factors = {c["season"]["foliage_density_factor"] for c in cases}
    snow = {c["season"]["snow_cover"] for c in cases}
    if len(factors) < 3:
        problems.append(f"only {len(factors)} distinct foliage factors")
    if len(snow) < 2:
        problems.append("snow_cover never varies")

    # And the outputs must not be degenerate: a fixture of uniform maps would
    # pass a port that returned a constant.
    for case in cases:
        if len(set(case["classes"])) < 2:
            problems.append(f"{case['name']}: land cover is a single class")
        if len(set(case["sampled_heights_m"])) < 8:
            problems.append(f"{case['name']}: terrain is nearly flat")

    # Every pair of scenes must differ, or one of them is redundant.
    for i, a in enumerate(cases):
        for b in cases[i + 1 :]:
            if a["classes"] == b["classes"] and a["sampled_heights_m"] == b["sampled_heights_m"]:
                problems.append(f"{a['name']} and {b['name']} are identical")

    if problems:
        raise SystemExit("fixture does not discriminate:\n  " + "\n  ".join(problems))
    print(
        f"  seams covered: {len(derived)} derived / {len(explicit)} explicit seeds, "
        f"{len(multiplier)} multiplier / {len(floor)} floor, "
        f"max tile span {max(spans):.1f}, {len(factors)} foliage factors"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [capture_scene(scene) for scene in scenes()]
    print(f"captured {len(cases)} scenes")
    check_discriminating(cases)

    payload = {
        "description": (
            "build_terrain_layer and build_land_cover_layer, end to end. "
            "Pins the wiring between the parts P4.3a-f ported separately."
        ),
        "source": "src/argusnet/world/procedural.py",
        "generator": "scripts/capture_landscape_fixture.py",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
