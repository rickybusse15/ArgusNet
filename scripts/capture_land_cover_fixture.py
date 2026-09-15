#!/usr/bin/env python3
"""Record build_land_cover_layer's classification pass, bit-for-bit.

This covers everything `build_land_cover_layer` does *before* the obstacle and
patch polygon overlay, which is P4.3e. Driving it with `obstacles=()` and
`patches=()` makes those trailing loops no-ops, so the layer the reference
returns IS the classification result and can be compared directly.

WHAT IS ACTUALLY BEING PINNED

Seven class assignments run in order -- water, wetland, snow, rocky, forest,
scrub, road -- and each overwrites the last. The masks are NOT mutually
exclusive: `snow_mask` excludes only water, so snow lands on top of wetland
wherever both hold. Reordering the blocks changes the map while leaving every
individual predicate correct, so the order is the contract and the cases below
are chosen to make each overwrite observable.

Density follows two different rules in the same function. Wetland, snow, rocky
and road ASSIGN; forest and scrub take `max(existing, new)`. Road assigns
ZERO -- the only rule in the pass that lowers a density, so a road over forest
clears the foliage rather than keeping the larger value.

THREE TRAPS THAT PRODUCE PLAUSIBLE, WRONG MAPS

1. `np.meshgrid(x_values, y_values)` with no `indexing="ij"`, so axis 0 is Y.
   This module is the transpose of `grid_ops` and `semantic`, which are
   documented `(i = x, j = y)`. Getting it wrong yields a land-cover map that
   is self-consistent and rotated -- the same class of bug that put the lake
   ellipse 0.013 m out in P4.3c. Cases use NON-SQUARE grids so a transposed
   port cannot even produce the right shape.

2. `np.rint` in `_sample_mask_to_grid` is BANKER'S rounding, ties to even.
   Rust's `f64::round()` is half-away-from-zero. Third appearance of this trap
   in the migration. The source masks are resampled onto a DIFFERENT grid
   resolution here, so the resampler does real work rather than an identity
   map, and ties are searched for rather than hoped for.

3. The water-bias clamp here is `[0.03, 0.30]` and applies to four presets
   only. `_semantic_masks` clamps the SAME profile field to `[0.01, 0.35]`,
   forty lines away. Two different clamps, one field.

Usage:
    python3 scripts/capture_land_cover_fixture.py --out tests/fixtures/land_cover.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.world.environment import Bounds2D, SeasonalVariation  # noqa: E402
from argusnet.world.procedural import (  # noqa: E402
    _PRESET_PROFILES,
    TerrainBuildConfig,
    _fbm,
    _sample_mask_to_grid,
    _stable_seed,
    build_land_cover_layer,
    build_terrain_layer,
    terrain_profile_for_preset,
)

MASK_NAMES = ("water", "lowland", "valley", "steep_slope", "ridge", "road")


def fstr(value: float) -> str:
    """Floats are stored as strings, matching `semantic.json` and `noise.json`.

    Python's `repr` and Rust's `str::parse::<f64>` are both shortest
    round-trip, so the value survives exactly. Consistency with the fixtures
    beside it matters more than the two bytes of quotes.
    """
    return repr(float(value))


def bits(values) -> str:
    """Masks are stored as "0"/"1" strings, not JSON boolean arrays.

    `true`/`false` costs five or six bytes a cell, and the masks are most of
    this fixture: as arrays the file was 469 KB, of which 320 KB was booleans.
    A character a cell brings it under the threshold in
    PERFORMANCE_AND_BENCHMARKING.md 5.5 while staying readable in a diff --
    a flipped cell shows as a flipped character.
    """
    return "".join("1" if bool(v) else "0" for v in values)


class _Profile:
    """The recorded profile, rehydrated. Attribute access only -- the guard
    reads it exactly as the reference reads a `TerrainProfile`."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def scenes() -> list[dict]:
    """Seven scenes, each present to reach something the others cannot.

    P4.3c's first mutation pass caught only 46 of 63 because every case ran on
    one grid at one scale. The lesson was structural: a single configuration
    cannot exercise parameters that only matter at other scales. These are
    chosen against the branch list, not by varying a seed.
    """
    return [
        # The four second-quantile presets, and one that is not on the list.
        dict(name="coastal", preset="coastal", month=7),
        dict(name="lake_district", preset="lake_district", month=7),
        dict(name="river_valley", preset="river_valley", month=7),
        # arctic_tundra takes the second quantile AND snows unconditionally,
        # so snow overwrites wetland on every non-water cell.
        dict(name="arctic_tundra", preset="arctic_tundra", month=7),
        dict(name="rolling_highlands", preset="rolling_highlands", month=7),
        # alpine in January: snow_cover is on, so the `elev_norm > snow_bias`
        # branch is live rather than short-circuited by the preset test. It is
        # also one of the two presets with the separate `elev_norm > 0.82` rule.
        dict(name="alpine_winter", preset="alpine", month=1),
        # Autumn foliage: foliage_density_factor is below 1, so 190*f and 105*f
        # truncate to something other than 190 and 105 and the uint8 cast is
        # observable. Jungle has the highest vegetation_bias, which moves the
        # forest moisture threshold off its 0.35 floor.
        dict(name="jungle_autumn", preset="jungle_canopy", month=10),
        # Vegetation suppressed: wetland, forest and scrub must all stay
        # unassigned while snow and rocky still apply.
        dict(
            name="suppressed",
            preset="jungle_canopy",
            month=7,
            suppress_vegetation=True,
        ),
        # A geometry SEARCHED FOR, not stumbled on. `np.rint` only differs from
        # half-away-from-zero on an exact tie, and none of the scenes above
        # produce one: their cell centres never land on a half-integer source
        # index. Scanning bounds against terrain and cover resolutions found
        # 212 combinations with ties on both axes; this is one of them, at
        # 8 m into 24 m so the resampler is doing a real 3:1 downsample rather
        # than an identity map. "Not found by sampling" is not "unreachable" --
        # the same lesson as the bit-exact slope thresholds in P4.3c.
        dict(
            name="rint_ties",
            preset="lake_district",
            month=7,
            width_m=480.0,
            height_m=384.0,
            terrain_resolution_m=8.0,
            cover_resolution_m=24.0,
        ),
        # The two water clamps are `[0.03, 0.30]` here and `[0.01, 0.35]` in
        # `_semantic_masks`, and NO PRESET CAN TELL THEM APART: the only four
        # presets that reach this code have water_bias 0.08, 0.16, 0.2 and
        # 0.28, all strictly interior to both ranges. Both clamps are therefore
        # no-ops on every real preset. Rather than declare that an equivalence
        # and move on -- a future preset edit would silently break it -- the
        # profile is overridden to a bias that lands between the two lower
        # bounds, where 0.02 clamps to 0.03 under one rule and stays 0.02 under
        # the other.
        # alpine in SUMMER. The winter case above cannot test the
        # `elev_norm > 0.82` rule at all: with snow_cover on, the
        # `elev_norm > snow_bias` term at 0.45 already covers everything the
        # 0.82 term would add, so both the threshold and the two-preset list
        # are dead. With snow_cover off, the 0.82 branch is the only thing
        # putting snow on the map.
        dict(name="alpine_summer", preset="alpine", month=7),
        dict(name="mountain_pass_summer", preset="mountain_pass", month=7),
        # The wetland predicate reads `lowland | valley | (moist & low)`, and
        # the valley term is astonishingly hard to make observable. A cell has
        # to be valley but NOT lowland, survive the water and slope gates, NOT
        # also qualify through the moisture path, and then not be overwritten
        # by the road rule that runs last -- and roads follow valleys, so most
        # candidates are roads.
        #
        # A first search over six presets found zero. That was not evidence of
        # an equivalence, only of an incomplete search: it did not include
        # mountain_pass, which supplies one such cell on an 8x12 grid.
        dict(
            name="valley_only",
            preset="mountain_pass",
            month=1,
            seed=7,
            cover_resolution_m=40.0,
        ),
        # Winter foliage is the only season whose density scaling is not a
        # whole number: 190 * 0.15 = 28.5. numpy's uint8 cast TRUNCATES to 28;
        # Rust's `.round()` would give 29. Every other season lands on an
        # integer, so no summer or autumn case can tell the two apart.
        dict(
            name="winter_forest",
            preset="rolling_highlands",
            month=1,
            cover_resolution_m=34.0,
        ),
        # Two survivors at once. This geometry is the only one found that puts
        # a ROAD ON A WATER CELL -- `road_mask & ~water_mask` is otherwise
        # indistinguishable from `road_mask` -- and it also lands a cell in the
        # narrow moisture window between 0.60 and 0.68, which is what makes the
        # wetland moisture threshold a number rather than a direction.
        #
        # It was caught before the bounds origin moved off (0, 0) and was lost
        # when the terrain changed under it. Fixture coverage is a property of
        # the geometry, not just of the case count.
        dict(
            name="road_over_water",
            preset="river_valley",
            month=7,
            seed=7,
            cover_resolution_m=34.0,
        ),
        # The wetland moisture threshold, likewise lost to the origin move.
        # The first attempt at this scene searched for cells whose moisture
        # falls in the window between 0.60 and 0.68 -- and found plenty, on a
        # scene where the threshold still changed nothing, because later rules
        # overwrote every one of them. Searching the FINAL CLASSIFICATION
        # instead found 15 scenes; this is the smallest.
        dict(
            name="wetland_moisture_window",
            preset="rolling_highlands",
            month=7,
            seed=7,
            cover_resolution_m=40.0,
        ),
        # The forest moisture floor `max(0.35, 0.55 - veg * 0.18)` binds only
        # when vegetation_bias exceeds 1.1111. The highest in the preset table
        # is jungle_canopy's 0.95, giving 0.379 -- so THE FLOOR IS A NO-OP ON
        # EVERY PRESET, exactly like the water clamps. Driven directly instead.
        # Overriding the bias is not enough on its own: the window the floor
        # opens is `[0.55 - veg * 0.18, 0.35]`, and a cell has to have its
        # moisture INSIDE that window for the floor to change its class.
        # jungle_canopy's moisture of 0.95 can never fall to 0.35 -- fbm would
        # have to return about -2.3 -- so the first attempt at this scene
        # caught nothing. The second, on desert_canyon, put cells in the window
        # and still caught nothing, because they failed the other forest tests.
        # alpine at 40 m clears all of them: the floor moves the threshold from
        # 0.350 to 0.280 and takes 23 cells with it.
        dict(
            name="vegetation_floor_override",
            preset="alpine",
            month=7,
            cover_resolution_m=40.0,
            profile_overrides={"vegetation_bias": 1.5},
        ),
        # `elev_norm > snow_bias` versus `>=` needs a cell whose elev_norm is
        # bit-exactly the bias. elev_norm is `(h - h_min) / span`, so the
        # minimum-height cell is exactly 0.0 -- and a snow_bias of 0.0 puts the
        # boundary right there. The lowest bias in the table is 0.2, which no
        # cell can land on, so this too is driven directly.
        dict(
            name="snow_bias_zero",
            preset="rolling_highlands",
            month=1,
            profile_overrides={"snow_bias": 0.0},
        ),
        # ... and the UPPER bound, where 0.5 clamps to 0.30 under one rule and
        # 0.35 under the other. Two scenes, because a single combined mutant
        # would be "caught" by a case that exercises only one bound.
        dict(
            name="water_clamp_upper",
            preset="arctic_tundra",
            month=7,
            cover_resolution_m=40.0,
            profile_overrides={"water_bias": 0.5},
        ),
        # The LOWER bound: 0.02 clamps to 0.03 under this pass's rule and stays
        # 0.02 under `_semantic_masks`'. This one took a search, and two
        # separate things were absorbing it.
        #
        # `coastal` at the default geometry cannot show it at all: its low tail
        # is a flat flooded sea, so quantile(h, 0.02) and quantile(h, 0.03)
        # return the SAME height and the two rules agree exactly. And where the
        # levels DO differ, the cells between them sit near the minimum -- which
        # is where `heights <= ground_plane + 1.0` has already marked them
        # water, so the second quantile changes nothing observable.
        #
        # Scanning presets, seeds and cover resolutions is what found these.
        # Both scenes are the smallest hit of their search, at 8x12. They also
        # had to be re-searched once: moving the default bounds origin off
        # (0, 0) -- needed to catch a port that confuses source_x_min with
        # source_y_min -- changed the terrain and cost both scenes their
        # discrimination. The guard caught that immediately, which is the point
        # of running it on every capture rather than once.
        dict(
            name="water_clamp_lower",
            preset="arctic_tundra",
            month=7,
            seed=99,
            cover_resolution_m=40.0,
            profile_overrides={"water_bias": 0.02},
        ),
    ]


def grids_from_layer(layer, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble the class and density rasters from the layer's tiles.

    `LandCoverLayer.from_rasters` pads each tile out to `tile_size_cells` and
    keeps no copy of the original grids, so the only way to read back exactly
    what the classification produced is to undo the tiling. Sampling through
    `land_cover_at` would go through the native handle instead -- part of what
    is being ported, so not a usable reference.
    """
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


def capture_scene(scene: dict) -> dict:
    preset = scene["preset"]
    seed = scene.get("seed", 20260911)
    month = scene["month"]
    suppress = bool(scene.get("suppress_vegetation", False))
    ground_plane_m = float(scene.get("ground_plane_m", 0.0))

    overrides = scene.get("profile_overrides")
    restore = None
    if overrides:
        # The preset table cannot reach some parameters at all -- see the
        # water-bias note in main(). The Rust port takes the profile as data,
        # so the fixture drives it directly; here that means patching the table
        # for the duration of one scene.
        original = _PRESET_PROFILES[preset]
        _PRESET_PROFILES[preset] = dataclasses.replace(original, **overrides)
        restore = original
    try:
        return _capture_scene_inner(scene, preset, seed, month, suppress, ground_plane_m)
    finally:
        if restore is not None:
            _PRESET_PROFILES[preset] = restore


def _capture_scene_inner(scene, preset, seed, month, suppress, ground_plane_m):
    # Deliberately non-square, and deliberately not a multiple of the terrain
    # resolution: a square grid cannot catch a transposed port, and matching
    # resolutions would turn the mask resampler into an identity map.
    # The origin is DELIBERATELY not (0, 0) by default, and x and y differ.
    # With both at zero, `source_x_min_m` and `source_y_min_m` are the same
    # number and a port that uses one where it means the other is invisible.
    x_min = float(scene.get("x_min_m", -120.0))
    y_min = float(scene.get("y_min_m", 75.0))
    bounds = Bounds2D(
        x_min_m=x_min,
        x_max_m=x_min + float(scene.get("width_m", 460.0)),
        y_min_m=y_min,
        y_max_m=y_min + float(scene.get("height_m", 300.0)),
    )
    terrain_resolution_m = float(scene.get("terrain_resolution_m", 20.0))
    cover_resolution_m = float(scene.get("cover_resolution_m", 26.0))

    terrain = build_terrain_layer(
        TerrainBuildConfig(
            terrain_preset=preset,
            terrain_seed=seed,
            terrain_resolution_m=terrain_resolution_m,
            season_month=month,
            ground_plane_m=ground_plane_m,
        ),
        bounds,
    )
    season = SeasonalVariation.from_month(month)

    # The reference, with nothing for the polygon overlay to do. This is the
    # answer every recomputation below is checked against.
    layer = build_land_cover_layer(
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=(),
        patches=(),
        resolution_m=cover_resolution_m,
        season=season,
        terrain_preset=preset,
        seed=seed,
        suppress_vegetation=suppress,
    )

    # Rebuild the inputs exactly as build_land_cover_layer does. Every one of
    # these is an input to the Rust function under test, so getting them from
    # the same expressions -- and then proving the result matches the layer
    # above -- is what makes the fixture a test of the port rather than of this
    # script.
    cols = max(1, int(math.ceil(bounds.width_m / cover_resolution_m)))
    rows = max(1, int(math.ceil(bounds.height_m / cover_resolution_m)))
    x_values = bounds.x_min_m + ((np.arange(cols, dtype=float) + 0.5) * cover_resolution_m)
    y_values = bounds.y_min_m + ((np.arange(rows, dtype=float) + 0.5) * cover_resolution_m)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    heights = terrain.height_at_many(points).reshape(rows, cols)
    slopes = terrain.slope_rad_at_many(points).reshape(rows, cols)

    moisture_noise = _fbm(
        xx,
        yy,
        seed=_stable_seed("land-cover", preset, seed),
        base_wavelength_m=max(bounds.width_m, bounds.height_m) * 0.22,
        octaves=4,
    )

    source_masks = dict(terrain.semantic_masks)
    sampled = {
        name: _sample_mask_to_grid(
            mask, source_bounds=terrain.bounds_xy_m, target_xx=xx, target_yy=yy
        )
        for name, mask in source_masks.items()
    }

    classes, density = grids_from_layer(layer, rows, cols)
    assert classes.shape == (rows, cols), f"{classes.shape} != {(rows, cols)}"

    return {
        "name": scene["name"],
        "preset": preset,
        "seed": seed,
        # Recorded as DATA rather than looked up by name in the parity test:
        # some scenes override it, because the preset table cannot reach every
        # parameter (see the water-bias note in main()).
        "profile": {
            key: (fstr(value) if isinstance(value, float) else value)
            for key, value in dataclasses.asdict(terrain_profile_for_preset(preset)).items()
        },
        "rows": rows,
        "cols": cols,
        "suppress_vegetation": suppress,
        "ground_plane_m": fstr(terrain.ground_plane_m),
        "season": {
            "foliage_density_factor": fstr(season.foliage_density_factor),
            "snow_cover": bool(season.snow_cover),
        },
        "cover_bounds": {
            "x_min_m": fstr(bounds.x_min_m),
            "y_min_m": fstr(bounds.y_min_m),
            "width_m": fstr(bounds.width_m),
            "height_m": fstr(bounds.height_m),
        },
        "source_bounds": {
            "x_min_m": fstr(terrain.bounds_xy_m.x_min_m),
            "y_min_m": fstr(terrain.bounds_xy_m.y_min_m),
            "width_m": fstr(terrain.bounds_xy_m.width_m),
            "height_m": fstr(terrain.bounds_xy_m.height_m),
        },
        "cover_resolution_m": fstr(cover_resolution_m),
        "heights_m": [fstr(v) for v in heights.reshape(-1)],
        "slopes_rad": [fstr(v) for v in slopes.reshape(-1)],
        "moisture_noise": [fstr(v) for v in np.asarray(moisture_noise).reshape(-1)],
        "source_masks": {
            name: {
                "rows": int(np.asarray(mask).shape[0]),
                "cols": int(np.asarray(mask).shape[1]),
                "values": bits(np.asarray(mask, dtype=bool).reshape(-1)),
            }
            for name, mask in sorted(source_masks.items())
        },
        "sampled_masks": {
            name: bits(np.asarray(mask, dtype=bool).reshape(-1))
            for name, mask in sorted(sampled.items())
        },
        "classes": [int(v) for v in classes.reshape(-1)],
        "density": [int(v) for v in density.reshape(-1)],
    }


# ---------------------------------------------------------------------------
# Discrimination guards
#
# Three times in this migration a guard has asserted a PROXY -- "are the
# offsets non-round", "is a divergent sample count present", "does the inner
# noise value change" -- passed, and caught nothing. The rule that came out of
# it is the one applied here: implement the wrong version and require it to
# disagree on the FINAL OUTPUT. Anything less is not an assertion.
# ---------------------------------------------------------------------------


def wrong_wetland_without_valley(case: dict) -> tuple[list[int], list[int]]:
    """The wetland predicate with its `valley` term dropped.

    `valley_mask` is a snapshot of `lowland_mask`, so it is USUALLY a subset
    and the term usually redundant -- but not always, and where it is not, the
    road rule that runs last tends to overwrite the difference.
    """
    blinded = dict(case)
    blinded["sampled_masks"] = dict(case["sampled_masks"])
    if "valley" in blinded["sampled_masks"]:
        blinded["sampled_masks"]["valley"] = "0" * len(blinded["sampled_masks"]["valley"])
    return reference_classification(blinded)


def _unbits(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) == ord("1")


def _rint_half_away(values: np.ndarray) -> np.ndarray:
    """What Rust's `f64::round()` would do: ties away from zero."""
    return np.floor(np.abs(values) + 0.5) * np.sign(values)


def wrong_resampler_ties_away(case: dict) -> dict[str, str]:
    """`_sample_mask_to_grid` with half-away rounding instead of banker's."""
    rows, cols = case["rows"], case["cols"]
    res = float(case["cover_resolution_m"])
    cb = {k: float(v) for k, v in case["cover_bounds"].items()}
    sb = {k: float(v) for k, v in case["source_bounds"].items()}
    x_values = cb["x_min_m"] + ((np.arange(cols, dtype=float) + 0.5) * res)
    y_values = cb["y_min_m"] + ((np.arange(rows, dtype=float) + 0.5) * res)
    xx, yy = np.meshgrid(x_values, y_values)

    out: dict[str, str] = {}
    for name, mask in case["source_masks"].items():
        m = _unbits(mask["values"]).reshape(mask["rows"], mask["cols"])
        src_rows, src_cols = m.shape
        col = np.clip(
            _rint_half_away(
                (xx - sb["x_min_m"]) / max(sb["width_m"], 1.0e-9) * max(src_cols - 1, 1)
            ).astype(int),
            0,
            src_cols - 1,
        )
        row = np.clip(
            _rint_half_away(
                (yy - sb["y_min_m"]) / max(sb["height_m"], 1.0e-9) * max(src_rows - 1, 1)
            ).astype(int),
            0,
            src_rows - 1,
        )
        out[name] = bits(m[row, col].reshape(-1))
    return out


def wrong_resampler_transposed(case: dict) -> dict[str, str]:
    """The source mask read as (row = x, col = y) -- `grid_ops`' convention."""
    rows, cols = case["rows"], case["cols"]
    res = float(case["cover_resolution_m"])
    cb = {k: float(v) for k, v in case["cover_bounds"].items()}
    sb = {k: float(v) for k, v in case["source_bounds"].items()}
    x_values = cb["x_min_m"] + ((np.arange(cols, dtype=float) + 0.5) * res)
    y_values = cb["y_min_m"] + ((np.arange(rows, dtype=float) + 0.5) * res)
    xx, yy = np.meshgrid(x_values, y_values)

    out: dict[str, str] = {}
    for name, mask in case["source_masks"].items():
        m = _unbits(mask["values"]).reshape(mask["rows"], mask["cols"]).T
        src_rows, src_cols = m.shape
        col = np.clip(
            np.rint(
                (xx - sb["x_min_m"]) / max(sb["width_m"], 1.0e-9) * max(src_cols - 1, 1)
            ).astype(int),
            0,
            src_cols - 1,
        )
        row = np.clip(
            np.rint(
                (yy - sb["y_min_m"]) / max(sb["height_m"], 1.0e-9) * max(src_rows - 1, 1)
            ).astype(int),
            0,
            src_rows - 1,
        )
        out[name] = bits(m[row, col].reshape(-1))
    return out


def _classify(
    case: dict,
    *,
    snow_before_wetland: bool,
    road_keeps_density: bool,
    forest_assigns: bool,
    water_clamp: tuple[float, float],
    wetland_moisture: float = 0.68,
    road_ignores_water: bool = False,
) -> tuple[list[int], list[int]]:
    """The reference classification, with four behaviours switchable.

    Every switch defaults (in `reference_classification`) to the reference's
    own choice, so this doubles as the check that the capture reproduced the
    layer it recorded.
    """
    n = case["rows"] * case["cols"]
    profile = _Profile(**case["profile"])
    heights = np.asarray([float(v) for v in case["heights_m"]])
    slopes = np.asarray([float(v) for v in case["slopes_rad"]])
    noise = np.asarray([float(v) for v in case["moisture_noise"]])
    season = case["season"]
    suppress = case["suppress_vegetation"]
    masks = {k: _unbits(v) for k, v in case["sampled_masks"].items()}
    zero = np.zeros(n, dtype=bool)

    elev_norm = (heights - heights.min()) / max(heights.max() - heights.min(), 1.0)
    moisture = np.clip(float(profile.moisture) + (0.28 * noise), 0.0, 1.0)

    classes = np.full(n, 0, dtype=np.uint8)
    density = np.zeros(n, dtype=np.uint8)

    water = (heights <= float(case["ground_plane_m"]) + 1.0) | masks.get("water", zero)
    if case["preset"] in {"coastal", "lake_district", "river_valley", "arctic_tundra"}:
        level = np.quantile(
            heights, min(max(float(profile.water_bias), water_clamp[0]), water_clamp[1])
        )
        water = water | ((heights <= level) & (slopes < 0.16))
    classes[water] = 3

    wetland = (
        (~water)
        & (
            masks.get("lowland", zero)
            | masks.get("valley", zero)
            | ((moisture > wetland_moisture) & (elev_norm < 0.30))
        )
        & (slopes < 0.14)
    )
    snow = (~water) & (
        (season["snow_cover"] & (elev_norm > float(profile.snow_bias)))
        | (case["preset"] == "arctic_tundra")
        | ((case["preset"] in {"alpine", "mountain_pass"}) & (elev_norm > 0.82))
    )

    def apply_wetland() -> None:
        if not suppress:
            classes[wetland] = 5
            density[wetland] = 120

    def apply_snow() -> None:
        classes[snow] = 7
        density[snow] = 40

    # The reference runs wetland first, so snow overwrites it. Swapping these
    # leaves both predicates correct and changes the map.
    if snow_before_wetland:
        apply_snow()
        apply_wetland()
    else:
        apply_wetland()
        apply_snow()

    rocky = (
        (~water)
        & (~snow)
        & (
            masks.get("steep_slope", zero)
            | masks.get("ridge", zero)
            | (slopes > 0.52)
            | (elev_norm > 0.78)
        )
    )
    classes[rocky] = 6
    density[rocky] = 60

    forest = np.zeros(n, dtype=bool)
    if not suppress:
        forest = (
            (~water)
            & (~snow)
            & (~rocky)
            & (moisture > max(0.35, 0.55 - (float(profile.vegetation_bias) * 0.18)))
            & (elev_norm > 0.18)
            & (slopes < 0.45)
        )
        classes[forest] = 2
        fd = np.uint8(np.clip(190 * float(season["foliage_density_factor"]), 0, 255))
        density[forest] = fd if forest_assigns else np.maximum(density[forest], fd)

        scrub = (~water) & (~snow) & (~rocky) & (~forest) & ((moisture < 0.42) | (slopes > 0.24))
        classes[scrub] = 4
        sd = np.uint8(np.clip(105 * float(season["foliage_density_factor"]), 0, 255))
        density[scrub] = np.maximum(density[scrub], sd)

    road = masks.get("road", zero)
    if not road_ignores_water:
        road = road & (~water)
    classes[road] = 8
    # The reference assigns zero here. Keeping the larger value is the obvious
    # "consistency" fix and is wrong.
    density[road] = np.maximum(density[road], 0) if road_keeps_density else 0
    return [int(v) for v in classes], [int(v) for v in density]


def reference_classification(case: dict) -> tuple[list[int], list[int]]:
    return _classify(
        case,
        snow_before_wetland=False,
        road_keeps_density=False,
        forest_assigns=False,
        water_clamp=(0.03, 0.30),
    )


WRONG_VERSIONS = {
    "resampler/ties-away-from-zero": lambda c: ("sampled", wrong_resampler_ties_away(c)),
    "resampler/source-transposed": lambda c: ("sampled", wrong_resampler_transposed(c)),
    "order/snow-before-wetland": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=True,
            road_keeps_density=False,
            forest_assigns=False,
            water_clamp=(0.03, 0.30),
        ),
    ),
    "road/keeps-the-larger-density": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=True,
            forest_assigns=False,
            water_clamp=(0.03, 0.30),
        ),
    ),
    "forest/assigns-instead-of-max": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=False,
            forest_assigns=True,
            water_clamp=(0.03, 0.30),
        ),
    ),
    # Split into the two bounds. A single combined mutant would be "caught" by
    # a case that only exercises one of them, leaving the other untested.
    "wetland/valley-term-dropped": lambda c: ("grid", wrong_wetland_without_valley(c)),
    "wetland/moisture-threshold-0.60": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=False,
            forest_assigns=False,
            water_clamp=(0.03, 0.30),
            wetland_moisture=0.60,
        ),
    ),
    "road/does-not-exclude-water": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=False,
            forest_assigns=False,
            water_clamp=(0.03, 0.30),
            road_ignores_water=True,
        ),
    ),
    "water/lower-clamp-is-0.01": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=False,
            forest_assigns=False,
            water_clamp=(0.01, 0.30),
        ),
    ),
    "water/upper-clamp-is-0.35": lambda c: (
        "grid",
        _classify(
            c,
            snow_before_wetland=False,
            road_keeps_density=False,
            forest_assigns=False,
            water_clamp=(0.03, 0.35),
        ),
    ),
}


def check_discriminating(cases: list[dict]) -> dict[str, int]:
    """Refuse to write a fixture that cannot tell right from wrong.

    Each wrong version is recomputed over every case and must disagree with
    what was recorded on at least one of them. A wrong version that no case
    can distinguish is a hole in the fixture, not an equivalence -- proving
    equivalence is a separate argument that has to be made explicitly.
    """
    scores: dict[str, int] = {}
    for label, build in WRONG_VERSIONS.items():
        hits = 0
        for case in cases:
            kind, wrong = build(case)
            if kind == "sampled":
                if any(wrong[name] != case["sampled_masks"][name] for name in wrong):
                    hits += 1
            else:
                if list(wrong[0]) != case["classes"] or list(wrong[1]) != case["density"]:
                    hits += 1
        scores[label] = hits
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [capture_scene(scene) for scene in scenes()]

    # The capture rebuilt the inputs by hand. This proves it rebuilt them
    # correctly, by running the reference's own logic over them and requiring
    # the answer the real `build_land_cover_layer` gave.
    for case in cases:
        classes, density = reference_classification(case)
        if classes != case["classes"] or density != case["density"]:
            print(f"ABORT: {case['name']}: recomputed inputs do not reproduce the layer")
            return 2
    print(f"{len(cases)} cases reproduce build_land_cover_layer exactly\n")

    scores = check_discriminating(cases)
    for label, hits in sorted(scores.items()):
        print(f"{'OK  ' if hits else 'GAP '} {label}: {hits}/{len(cases)} cases disagree")
    missing = [label for label, hits in scores.items() if hits == 0]
    if missing:
        print(f"\nABORT: {len(missing)} wrong version(s) indistinguishable: {missing}")
        return 2

    document = {
        "description": (
            "build_land_cover_layer's classification pass, before the obstacle "
            "and patch overlay. Regenerate with "
            "scripts/capture_land_cover_fixture.py."
        ),
        "mask_names": list(MASK_NAMES),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True))
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
