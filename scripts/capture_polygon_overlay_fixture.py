#!/usr/bin/env python3
"""Record the obstacle and patch polygon overlay that closes `build_land_cover_layer`.

P4.3d pinned the seven mask rules; this pins the two loops that run after them.
`capture_land_cover_fixture.py` deliberately drives the reference with
`obstacles=()` and `patches=()` so those loops are no-ops -- everything they do
is recorded here instead.

HOW THE ANSWER IS OBTAINED WITHOUT TRUSTING THIS SCRIPT

Each scene calls the real `build_land_cover_layer` TWICE on identical inputs,
once with the obstacles and patches and once without. Everything upstream of
the overlay is deterministic, so the two rasters differ by exactly the overlay
and nothing else. The fixture stores the empty run as `base_*` and the full run
as `final_*`; the port's job is to turn one into the other. No part of the
overlay is recomputed here to produce the answer.

WHAT THE OVERLAY ACTUALLY DOES, AND THE FOUR WAYS TO GET IT WRONG

1. OBSTACLES RUN BEFORE PATCHES. A patch exists so a caller can override what
   the generator produced, and it can only do that by running last. Swapping
   the two loops leaves every individual rule correct and changes the map
   wherever they overlap.

2. CLASS ASSIGNS, DENSITY TAKES `max`. That pairing is what separates the
   overlay from the ROAD rule twenty lines above it, which assigns density
   ZERO. So there are two ways to put a road on the map with two different
   densities: the road mask clears the foliage, a ROAD *patch* keeps it.

3. A FOREST STAND IS SKIPPED UNDER `suppress_vegetation`; A BUILDING IS NOT.
   The flag removes the trees and leaves the town. A patch is not gated at all,
   so a FOREST patch lands even with vegetation suppressed -- the flag
   suppresses *generated* vegetation, not vegetation the caller named.

4. AN OBSTACLE WITH NO FOOTPRINT IS SILENTLY DROPPED. The reference's
   `else: continue` catches exactly the cylinders, which have no
   `footprint_xy_m` at all. A cylindrical obstacle occludes sight and blocks
   movement and contributes nothing whatever to land cover. Every other
   primitive either stores a footprint array or exposes it as a CALLABLE
   (`OrientedBox`, `WallSegment`), which the reference handles with
   `footprint_source() if callable(footprint_source) else footprint_source`.

Two numeric traps, both of which round the wrong way if transcribed casually:
`int(np.clip(obstacle.density, 0.0, 1.0) * 255)` TRUNCATES -- 0.5 gives 127,
not 128 -- and `int(np.clip(patch.density, 0, 255))` clamps a density that an
`as u8` cast would wrap.

Usage:
    python3 scripts/capture_polygon_overlay_fixture.py --out tests/fixtures/polygon_overlay.json
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

from argusnet.world.environment import Bounds2D, SeasonalVariation  # noqa: E402
from argusnet.world.obstacles import (  # noqa: E402
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    OrientedBox,
    PolygonPrism,
    WallSegment,
    _point_in_polygon,
)
from argusnet.world.procedural import (  # noqa: E402
    LandCoverClass,
    LandCoverPatch,
    TerrainBuildConfig,
    build_land_cover_layer,
    build_terrain_layer,
)


def fstr(value: float) -> str:
    """Floats as shortest-round-trip strings, matching the fixtures beside this one."""
    return repr(float(value))


def poly(points) -> list[list[str]]:
    return [[fstr(p[0]), fstr(p[1])] for p in np.asarray(points, dtype=float)]


# ---------------------------------------------------------------------------
# Obstacle specs
#
# Recorded as CONSTRUCTOR ARGUMENTS, not as a derived polygon, so the parity
# test builds each primitive through the same constructor the Rust port
# exposes. Two of the five derive their footprint rather than storing one, and
# in the reference that difference shows up as a CALLABLE attribute -- a
# distinction that does not survive into Rust, where every prism is already a
# polygon. Recording the arguments keeps the derivation under test; the derived
# footprint is recorded alongside so a mismatch localises to the constructor
# instead of to the overlay.
# ---------------------------------------------------------------------------


# Per-kind defaults for the fields the scenes above do not set. Applied ONCE,
# up front, so the recorded spec is self-contained: the parity test rebuilds
# each primitive from the JSON alone and never has to know a default this
# script chose.
_DEFAULTS = {
    "forest-stand-v1": dict(base_z_m=0.0, top_z_m=18.0, blocker_type="vegetation"),
    "building-v1": dict(base_z_m=0.0, top_z_m=24.0, blocker_type="building"),
    "polygon-prism-v1": dict(base_z_m=0.0, top_z_m=24.0, blocker_type="building"),
    "box-v1": dict(base_z_m=0.0, top_z_m=20.0, blocker_type="building"),
    "wall-v1": dict(base_z_m=0.0, top_z_m=6.0, blocker_type="wall"),
    "cylinder-v1": dict(base_z_m=0.0, top_z_m=30.0, blocker_type="building"),
}


def normalize(spec: dict) -> dict:
    filled = dict(_DEFAULTS[spec["kind"]])
    filled.update(spec)
    return filled


def build_obstacle(spec: dict):
    kind = spec["kind"]
    if kind == "forest-stand-v1":
        return ForestStand(
            primitive_id=spec["id"],
            footprint_xy_m=spec["footprint_xy_m"],
            canopy_base_z_m=spec.get("base_z_m", 0.0),
            canopy_top_z_m=spec.get("top_z_m", 18.0),
            density=spec["density"],
        )
    if kind == "building-v1":
        return BuildingPrism(
            primitive_id=spec["id"],
            footprint_xy_m=spec["footprint_xy_m"],
            base_z_m=spec.get("base_z_m", 0.0),
            top_z_m=spec.get("top_z_m", 24.0),
        )
    if kind == "polygon-prism-v1":
        return PolygonPrism(
            primitive_id=spec["id"],
            blocker_type=spec.get("blocker_type", "building"),
            footprint_xy_m=np.asarray(spec["footprint_xy_m"], dtype=float),
            base_z_m=spec.get("base_z_m", 0.0),
            top_z_m=spec.get("top_z_m", 24.0),
        )
    if kind == "box-v1":
        return OrientedBox(
            primitive_id=spec["id"],
            blocker_type=spec.get("blocker_type", "building"),
            center_x_m=spec["center_x_m"],
            center_y_m=spec["center_y_m"],
            length_m=spec["length_m"],
            width_m=spec["width_m"],
            yaw_rad=spec["yaw_rad"],
            base_z_m=spec.get("base_z_m", 0.0),
            top_z_m=spec.get("top_z_m", 20.0),
        )
    if kind == "wall-v1":
        return WallSegment(
            primitive_id=spec["id"],
            blocker_type=spec.get("blocker_type", "wall"),
            start_xy_m=np.asarray(spec["start_xy_m"], dtype=float),
            end_xy_m=np.asarray(spec["end_xy_m"], dtype=float),
            thickness_m=spec["thickness_m"],
            base_z_m=spec.get("base_z_m", 0.0),
            top_z_m=spec.get("top_z_m", 6.0),
        )
    if kind == "cylinder-v1":
        return CylinderObstacle(
            primitive_id=spec["id"],
            blocker_type=spec.get("blocker_type", "building"),
            center_x_m=spec["center_x_m"],
            center_y_m=spec["center_y_m"],
            radius_m=spec["radius_m"],
            base_z_m=spec.get("base_z_m", 0.0),
            top_z_m=spec.get("top_z_m", 30.0),
        )
    raise ValueError(f"unknown obstacle kind {kind!r}")


def derived_footprint(obstacle) -> list[list[str]] | None:
    source = getattr(obstacle, "footprint_xy_m", None)
    if source is None:
        return None
    return poly(source() if callable(source) else source)


def build_patch(spec: dict) -> LandCoverPatch:
    return LandCoverPatch(
        polygon_xy_m=np.asarray(spec["polygon_xy_m"], dtype=float),
        land_cover_class=LandCoverClass(spec["land_cover_class"]),
        density=spec["density"],
    )


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

# One geometry for every scene unless a scene says otherwise: 18 x 12 cells at
# 26 m, on an origin that is neither (0, 0) nor square in x and y. Both choices
# are inherited from the land-cover fixture for the same reasons -- a square
# grid cannot catch a transposed port, and equal origins hide a port that
# confuses x_min with y_min.
DEFAULT_X_MIN = -120.0
DEFAULT_Y_MIN = 75.0
DEFAULT_WIDTH = 460.0
DEFAULT_HEIGHT = 300.0
DEFAULT_RESOLUTION = 26.0


def rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def scenes() -> list[dict]:
    """Each scene is here to reach a branch or a boundary the others cannot."""
    return [
        # -------------------------------------------------------------------
        # Obstacles before patches. A FOREST stand and a ROAD patch cover the
        # SAME cells. Correct: those cells end ROAD, at the stand's density
        # because `max` kept it. Swap the loops and they end FOREST; drop the
        # `max` and the density falls to the patch's.
        # -------------------------------------------------------------------
        dict(
            name="obstacle_then_patch_overlap",
            preset="rolling_highlands",
            month=7,
            obstacles=[
                dict(
                    kind="forest-stand-v1",
                    id="stand",
                    footprint_xy_m=rect(-60.0, 100.0, 120.0, 260.0),
                    density=0.92,
                )
            ],
            patches=[
                dict(
                    polygon_xy_m=rect(-60.0, 100.0, 120.0, 260.0),
                    land_cover_class=int(LandCoverClass.ROAD),
                    density=0,
                )
            ],
        ),
        # -------------------------------------------------------------------
        # The same overlap with the classes swapped, so "obstacles last" cannot
        # be caught by one scene and then quietly right in the other. Also the
        # cleanest `max` case in the fixture: an URBAN building at 180 under a
        # WETLAND patch at 5. Correct gives class WETLAND, density 180.
        # -------------------------------------------------------------------
        dict(
            name="patch_over_building_keeps_density",
            preset="rolling_highlands",
            month=7,
            obstacles=[
                dict(
                    kind="building-v1",
                    id="block",
                    footprint_xy_m=rect(0.0, 120.0, 200.0, 300.0),
                )
            ],
            patches=[
                dict(
                    polygon_xy_m=rect(50.0, 150.0, 260.0, 340.0),
                    land_cover_class=int(LandCoverClass.WETLAND),
                    density=5,
                )
            ],
        ),
        # -------------------------------------------------------------------
        # Vegetation suppressed, with a stand and a building on the same cells
        # and a FOREST patch beside them. Correct: the stand vanishes, the
        # building paints URBAN 180, and the patch paints FOREST anyway.
        #
        # Three separate wrong ports fail here and nowhere else -- skipping all
        # obstacles under the flag, skipping none, and gating the patch loop.
        # -------------------------------------------------------------------
        dict(
            name="suppressed_drops_only_the_stand",
            preset="jungle_canopy",
            month=7,
            suppress_vegetation=True,
            obstacles=[
                dict(
                    kind="forest-stand-v1",
                    id="stand",
                    footprint_xy_m=rect(-80.0, 90.0, 60.0, 250.0),
                    density=0.8,
                ),
                dict(
                    kind="building-v1",
                    id="block",
                    footprint_xy_m=rect(-80.0, 90.0, 60.0, 250.0),
                ),
            ],
            patches=[
                dict(
                    polygon_xy_m=rect(150.0, 120.0, 320.0, 300.0),
                    land_cover_class=int(LandCoverClass.FOREST),
                    density=210,
                )
            ],
        ),
        # The same obstacles with the flag OFF, so the stand's contribution is
        # observable rather than only its absence. The building is listed
        # SECOND, so it overwrites the stand: last obstacle wins, exactly as
        # last patch does.
        dict(
            name="unsuppressed_building_overwrites_stand",
            preset="jungle_canopy",
            month=7,
            obstacles=[
                dict(
                    kind="forest-stand-v1",
                    id="stand",
                    footprint_xy_m=rect(-80.0, 90.0, 60.0, 250.0),
                    density=0.8,
                ),
                dict(
                    kind="building-v1",
                    id="block",
                    footprint_xy_m=rect(-30.0, 150.0, 60.0, 250.0),
                ),
            ],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # `int(clip(d, 0, 1) * 255)` truncates. 0.5 -> 127.5 -> 127, and a
        # port that rounds gives 128. Searched rather than guessed: only a
        # density whose product lands on a non-integer can tell them apart, and
        # 0.5 is the one that lands exactly on a tie -- where `round()` and
        # `round_ties_even()` also disagree with each other.
        #
        # The second stand is ABOVE 1.0, which the clip caps at 255; the third
        # is negative, which `ForestStand.__init__` has already floored at 0.0
        # before the clip ever sees it.
        # -------------------------------------------------------------------
        dict(
            name="forest_density_truncates",
            preset="rolling_highlands",
            month=7,
            obstacles=[
                dict(
                    kind="forest-stand-v1",
                    id="half",
                    footprint_xy_m=rect(-100.0, 90.0, -10.0, 180.0),
                    density=0.5,
                ),
                dict(
                    kind="forest-stand-v1",
                    id="over",
                    footprint_xy_m=rect(20.0, 90.0, 140.0, 180.0),
                    density=3.5,
                ),
                dict(
                    kind="forest-stand-v1",
                    id="zero",
                    footprint_xy_m=rect(180.0, 90.0, 320.0, 180.0),
                    density=-2.0,
                ),
            ],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # Patch density outside [0, 255]. 300 clamps to 255; an `as u8` cast
        # without the clamp wraps it to 44, which is both wrong and plausible.
        # -400 clamps to 0, where a wrapping cast gives 112.
        # -------------------------------------------------------------------
        dict(
            name="patch_density_clamps",
            preset="rolling_highlands",
            month=7,
            obstacles=[],
            patches=[
                dict(
                    polygon_xy_m=rect(-100.0, 90.0, 40.0, 220.0),
                    land_cover_class=int(LandCoverClass.URBAN),
                    density=300,
                ),
                dict(
                    polygon_xy_m=rect(120.0, 90.0, 320.0, 220.0),
                    land_cover_class=int(LandCoverClass.OPEN),
                    density=-400,
                ),
            ],
        ),
        # -------------------------------------------------------------------
        # A cylinder, and nothing else. Correct: the raster is UNCHANGED --
        # `final_*` equals `base_*` cell for cell. The cylinder is centred on
        # the grid and wide enough to cover 20-odd cells, so a port that gives
        # it a bounding-box footprint (the obvious wrong reading of "it has no
        # `footprint_xy_m`") repaints every one of them.
        #
        # A second obstacle follows it to prove the loop does not simply stop
        # at the `continue`.
        # -------------------------------------------------------------------
        dict(
            name="cylinder_contributes_nothing",
            preset="rolling_highlands",
            month=7,
            obstacles=[
                dict(
                    kind="cylinder-v1",
                    id="tower",
                    center_x_m=40.0,
                    center_y_m=200.0,
                    radius_m=90.0,
                ),
                dict(
                    kind="building-v1",
                    id="after",
                    footprint_xy_m=rect(220.0, 250.0, 330.0, 360.0),
                ),
            ],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # The two primitives whose footprint is DERIVED, not stored -- and both
        # ROTATED, because an axis-aligned footprint is exactly the case the
        # old clamped-denominator polygon test happened to get right. That
        # defect misclassified 14-16% of probes around each rotated building
        # and went unseen for months because every fixture was axis-aligned.
        # -------------------------------------------------------------------
        dict(
            name="rotated_derived_footprints",
            preset="rolling_highlands",
            month=7,
            obstacles=[
                dict(
                    kind="box-v1",
                    id="hangar",
                    center_x_m=-10.0,
                    center_y_m=180.0,
                    length_m=190.0,
                    width_m=70.0,
                    yaw_rad=0.6981317007977318,
                ),
                dict(
                    kind="wall-v1",
                    id="perimeter",
                    start_xy_m=[150.0, 110.0],
                    end_xy_m=[330.0, 330.0],
                    thickness_m=44.0,
                ),
            ],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # A `PolygonPrism` declared `blocker_type="vegetation"` that is NOT a
        # `ForestStand`. The reference asks `isinstance(obstacle, ForestStand)`,
        # so this takes the URBAN branch at 180 -- and a port that dispatches on
        # the blocker type instead of on the kind paints FOREST here.
        #
        # It is also present under SUPPRESSION, where the same wrong port drops
        # it entirely.
        # -------------------------------------------------------------------
        dict(
            name="vegetation_prism_is_not_a_stand",
            preset="rolling_highlands",
            month=7,
            suppress_vegetation=True,
            obstacles=[
                dict(
                    kind="polygon-prism-v1",
                    id="hedge",
                    blocker_type="vegetation",
                    footprint_xy_m=rect(-90.0, 110.0, 130.0, 290.0),
                )
            ],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # A tall, narrow polygon: 2 cells wide and 9 tall. Transposing the grid
        # cannot even map it onto itself, so a port that indexes `[col, row]`
        # paints a 9-wide, 2-tall band somewhere else entirely. The grid is 18
        # by 12, so the transposed shape does not fit either.
        # -------------------------------------------------------------------
        dict(
            name="tall_narrow_polygon_pins_the_axes",
            preset="rolling_highlands",
            month=7,
            obstacles=[],
            patches=[
                dict(
                    polygon_xy_m=rect(-95.0, 95.0, -45.0, 330.0),
                    land_cover_class=int(LandCoverClass.SNOW),
                    density=77,
                )
            ],
        ),
        # -------------------------------------------------------------------
        # A patch whose edges pass exactly through cell centres. Cell centres
        # sit at x = -107 + 26k and y = 88 + 26k; the rectangle below has all
        # four edges on one. `_point_in_polygon` counts the boundary as INSIDE,
        # so the border cells are painted, and a port that tests strict
        # interior loses the entire outline.
        # -------------------------------------------------------------------
        dict(
            name="boundary_cells_count_as_inside",
            preset="rolling_highlands",
            month=7,
            obstacles=[],
            patches=[
                dict(
                    polygon_xy_m=rect(-81.0, 114.0, 97.0, 296.0),
                    land_cover_class=int(LandCoverClass.ROCKY),
                    density=33,
                )
            ],
        ),
        # -------------------------------------------------------------------
        # A ROAD patch over the generated forest. The road MASK assigns density
        # zero -- the only rule in the whole pass that lowers one -- while a
        # road PATCH takes `max` and keeps the foliage. Two roads, two
        # densities, and this is the scene that makes the difference visible.
        # jungle_canopy in July puts forest at 190 nearly everywhere.
        # -------------------------------------------------------------------
        dict(
            name="road_patch_keeps_foliage_density",
            preset="jungle_canopy",
            month=7,
            obstacles=[],
            patches=[
                dict(
                    polygon_xy_m=rect(-110.0, 140.0, 340.0, 200.0),
                    land_cover_class=int(LandCoverClass.ROAD),
                    density=0,
                )
            ],
        ),
        # -------------------------------------------------------------------
        # Nothing at all: empty obstacle and patch lists. `final_*` must equal
        # `base_*`, which pins the one case where the overlay has to be a
        # complete no-op rather than merely harmless.
        # -------------------------------------------------------------------
        dict(
            name="no_polygons_is_identity",
            preset="coastal",
            month=7,
            obstacles=[],
            patches=[],
        ),
        # -------------------------------------------------------------------
        # A polygon entirely OUTSIDE the cover bounds, and one that straddles
        # the edge. Neither may reach for a cell that is not there, and the
        # straddling one must paint the half that is.
        # -------------------------------------------------------------------
        dict(
            name="polygons_outside_and_across_the_bounds",
            preset="rolling_highlands",
            month=7,
            obstacles=[],
            patches=[
                dict(
                    polygon_xy_m=rect(2000.0, 2000.0, 2400.0, 2400.0),
                    land_cover_class=int(LandCoverClass.URBAN),
                    density=200,
                ),
                dict(
                    polygon_xy_m=rect(-400.0, -200.0, 0.0, 160.0),
                    land_cover_class=int(LandCoverClass.WATER),
                    density=11,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def grids_from_layer(layer, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble the rasters from the layer's padded tiles.

    Lifted from `capture_land_cover_fixture.py` for the same reason it exists
    there: `from_rasters` keeps no copy of the grids it was given, and reading
    them back through `land_cover_at` would go through the native handle --
    which is part of what is being ported, so not a usable reference.
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
    seed = int(scene.get("seed", 20260915))
    month = int(scene["month"])
    suppress = bool(scene.get("suppress_vegetation", False))
    resolution = float(scene.get("cover_resolution_m", DEFAULT_RESOLUTION))

    x_min = float(scene.get("x_min_m", DEFAULT_X_MIN))
    y_min = float(scene.get("y_min_m", DEFAULT_Y_MIN))
    bounds = Bounds2D(
        x_min_m=x_min,
        x_max_m=x_min + float(scene.get("width_m", DEFAULT_WIDTH)),
        y_min_m=y_min,
        y_max_m=y_min + float(scene.get("height_m", DEFAULT_HEIGHT)),
    )

    terrain = build_terrain_layer(
        TerrainBuildConfig(
            terrain_preset=preset,
            terrain_seed=seed,
            terrain_resolution_m=float(scene.get("terrain_resolution_m", 20.0)),
            season_month=month,
            ground_plane_m=0.0,
        ),
        bounds,
    )
    season = SeasonalVariation.from_month(month)

    obstacle_specs = [normalize(spec) for spec in scene["obstacles"]]
    patch_specs = list(scene["patches"])
    obstacles = [build_obstacle(spec) for spec in obstacle_specs]
    patches = [build_patch(spec) for spec in patch_specs]

    def run(obs, pat):
        layer = build_land_cover_layer(
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=obs,
            patches=pat,
            resolution_m=resolution,
            season=season,
            terrain_preset=preset,
            seed=seed,
            suppress_vegetation=suppress,
        )
        return layer

    cols = max(1, int(math.ceil(bounds.width_m / resolution)))
    rows = max(1, int(math.ceil(bounds.height_m / resolution)))

    # The overlay is the ONLY difference between these two calls.
    base_classes, base_density = grids_from_layer(run((), ()), rows, cols)
    final_classes, final_density = grids_from_layer(run(obstacles, patches), rows, cols)

    recorded_obstacles = []
    for spec, obstacle in zip(obstacle_specs, obstacles, strict=True):
        entry = dict(spec)
        for key in (
            "density",
            "center_x_m",
            "center_y_m",
            "radius_m",
            "length_m",
            "width_m",
            "yaw_rad",
            "thickness_m",
            "base_z_m",
            "top_z_m",
        ):
            if key in entry:
                entry[key] = fstr(entry[key])
        if "footprint_xy_m" in entry:
            entry["footprint_xy_m"] = poly(entry["footprint_xy_m"])
        for key in ("start_xy_m", "end_xy_m"):
            if key in entry:
                entry[key] = [fstr(v) for v in entry[key]]
        entry["derived_footprint_xy_m"] = derived_footprint(obstacle)
        recorded_obstacles.append(entry)

    return {
        "name": scene["name"],
        "preset": preset,
        "seed": seed,
        "month": month,
        "rows": rows,
        "cols": cols,
        "suppress_vegetation": suppress,
        "cover_bounds": {
            "x_min_m": fstr(bounds.x_min_m),
            "y_min_m": fstr(bounds.y_min_m),
            "width_m": fstr(bounds.width_m),
            "height_m": fstr(bounds.height_m),
        },
        "cover_resolution_m": fstr(resolution),
        "obstacles": recorded_obstacles,
        "patches": [
            {
                "polygon_xy_m": poly(spec["polygon_xy_m"]),
                "land_cover_class": int(spec["land_cover_class"]),
                "density": int(spec["density"]),
            }
            for spec in patch_specs
        ],
        "base_classes": [int(v) for v in base_classes.reshape(-1)],
        "base_density": [int(v) for v in base_density.reshape(-1)],
        "final_classes": [int(v) for v in final_classes.reshape(-1)],
        "final_density": [int(v) for v in final_density.reshape(-1)],
    }


# ---------------------------------------------------------------------------
# Discrimination guards
#
# Four times in this migration a guard has asserted a PROXY -- "are the offsets
# non-round", "is a divergent sample count present", "does the inner noise
# value change", "are there valley cells outside lowland" -- passed, and caught
# nothing. The rule that came out of it: implement the WRONG version and
# require it to disagree on the FINAL OUTPUT. Anything less is not an
# assertion.
#
# `reference_overlay` below is the only re-implementation here, and it is
# itself checked against the recorded `final_*` before any mutant is run. If it
# drifts from the reference, every guard becomes meaningless -- so that check
# runs first and hard.
# ---------------------------------------------------------------------------


def _polygons_for(
    case: dict,
    *,
    obstacles_first=True,
    suppress_stands=None,
    suppress_all=False,
    suppress_patches=False,
    forest_rounds=False,
    clamp_patch_density=True,
    cylinder_has_footprint=False,
    dispatch_on_blocker_type=False,
    reverse_obstacles=False,
):
    suppress = case["suppress_vegetation"] if suppress_stands is None else suppress_stands
    obstacle_polygons = []
    specs = list(case["obstacles"])
    if reverse_obstacles:
        specs = list(reversed(specs))
    for spec in specs:
        kind = spec["kind"]
        is_stand = (
            spec.get("blocker_type") == "vegetation"
            if dispatch_on_blocker_type
            else kind == "forest-stand-v1"
        )
        if kind == "forest-stand-v1" and dispatch_on_blocker_type:
            is_stand = True
        if is_stand:
            if suppress:
                continue
            raw = max(float(spec.get("density", 0.0)), 0.0)
            scaled = min(max(raw, 0.0), 1.0) * 255
            value = int(round(scaled)) if forest_rounds else int(scaled)
            obstacle_polygons.append(
                (spec["derived_footprint_xy_m"], int(LandCoverClass.FOREST), value)
            )
            continue
        if suppress_all and case["suppress_vegetation"]:
            continue
        footprint = spec["derived_footprint_xy_m"]
        if footprint is None:
            if not cylinder_has_footprint:
                continue
            cx, cy, r = (
                float(spec["center_x_m"]),
                float(spec["center_y_m"]),
                float(spec["radius_m"]),
            )
            footprint = [
                [repr(cx - r), repr(cy - r)],
                [repr(cx + r), repr(cy - r)],
                [repr(cx + r), repr(cy + r)],
                [repr(cx - r), repr(cy + r)],
            ]
        obstacle_polygons.append((footprint, int(LandCoverClass.URBAN), 180))

    patch_polygons = []
    if not (suppress_patches and case["suppress_vegetation"]):
        for spec in case["patches"]:
            raw = int(spec["density"])
            value = min(max(raw, 0), 255) if clamp_patch_density else (raw & 0xFF)
            patch_polygons.append(
                (
                    [[repr(p[0]), repr(p[1])] for p in spec["polygon_xy_m"]]
                    if not isinstance(spec["polygon_xy_m"][0][0], str)
                    else spec["polygon_xy_m"],
                    int(spec["land_cover_class"]),
                    value,
                )
            )

    return (
        obstacle_polygons + patch_polygons
        if obstacles_first
        else patch_polygons + obstacle_polygons
    )


def cell_centres(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """`x_min + (i + 0.5) * resolution`, exactly as the reference builds them.

    Not recorded in the fixture: they are a closed-form function of the bounds
    and the resolution, which ARE recorded, and 6,000 float literals of
    redundancy would be a third of the file. `land_cover_parity.rs` derives
    them the same way from the same two fields.
    """
    rows, cols = case["rows"], case["cols"]
    res = float(case["cover_resolution_m"])
    x_min = float(case["cover_bounds"]["x_min_m"])
    y_min = float(case["cover_bounds"]["y_min_m"])
    x_values = x_min + ((np.arange(cols, dtype=float) + 0.5) * res)
    y_values = y_min + ((np.arange(rows, dtype=float) + 0.5) * res)
    return np.meshgrid(x_values, y_values)


def reference_overlay(
    case: dict, *, transpose=False, strict_interior=False, density_assigns=False, **kwargs
) -> tuple[list[int], list[int]]:
    rows, cols = case["rows"], case["cols"]
    classes = np.array(case["base_classes"], dtype=np.uint8).reshape(rows, cols)
    density = np.array(case["base_density"], dtype=np.uint8).reshape(rows, cols)
    xx, yy = cell_centres(case)

    for footprint, cover_class, cover_density in _polygons_for(case, **kwargs):
        polygon = np.asarray([[float(p[0]), float(p[1])] for p in footprint], dtype=float)
        for row in range(rows):
            for col in range(cols):
                r, c = (col, row) if transpose else (row, col)
                if transpose and (r >= rows or c >= cols):
                    continue
                point = np.array([xx[r, c], yy[r, c]], dtype=float)
                inside = _point_in_polygon(point, polygon)
                if inside and strict_interior:
                    inside = _strictly_inside(point, polygon)
                if inside:
                    classes[r, c] = cover_class
                    density[r, c] = (
                        cover_density if density_assigns else max(int(density[r, c]), cover_density)
                    )
    return [int(v) for v in classes.reshape(-1)], [int(v) for v in density.reshape(-1)]


def _strictly_inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    """`_point_in_polygon` minus its boundary short-circuit."""
    n = len(polygon)
    for index in range(n):
        a, b = polygon[index], polygon[(index + 1) % n]
        segment = b - a
        rel = point - a
        if abs(segment[0] * rel[1] - segment[1] * rel[0]) <= 1.0e-6:
            dot = float(np.dot(rel, segment))
            if -1.0e-6 <= dot <= float(np.dot(segment, segment)) + 1.0e-6:
                return False
    return _point_in_polygon(point, polygon)


MUTANTS = {
    "patches-before-obstacles": dict(obstacles_first=False),
    "obstacles-in-reverse": dict(reverse_obstacles=True),
    "density-assigns-instead-of-max": dict(density_assigns=True),
    "suppression-skips-every-obstacle": dict(suppress_all=True),
    "suppression-skips-no-stand": dict(suppress_stands=False),
    "suppression-gates-the-patch-loop": dict(suppress_patches=True),
    "forest-density-rounds": dict(forest_rounds=True),
    "patch-density-wraps": dict(clamp_patch_density=False),
    "cylinder-gets-a-bounding-box": dict(cylinder_has_footprint=True),
    "dispatch-on-blocker-type": dict(dispatch_on_blocker_type=True),
    "grid-transposed": dict(transpose=True),
    "boundary-is-outside": dict(strict_interior=True),
}


def check_discriminating(cases: list[dict]) -> None:
    """Refuse to write a fixture that proves nothing."""
    for case in cases:
        got_classes, got_density = reference_overlay(case)
        if got_classes != case["final_classes"] or got_density != case["final_density"]:
            raise SystemExit(
                f"{case['name']}: the guard's own overlay does not reproduce the "
                "reference. Every mutation below would be meaningless -- fix this first."
            )

    missed = []
    for label, kwargs in MUTANTS.items():
        catchers = []
        for case in cases:
            wrong_classes, wrong_density = reference_overlay(case, **kwargs)
            if wrong_classes != case["final_classes"] or wrong_density != case["final_density"]:
                catchers.append(case["name"])
        if not catchers:
            missed.append(label)
        else:
            print(f"  {label}: caught by {', '.join(catchers)}")
    if missed:
        raise SystemExit("no scene distinguishes these wrong ports: " + ", ".join(missed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [capture_scene(scene) for scene in scenes()]
    print(f"captured {len(cases)} scenes")
    check_discriminating(cases)

    payload = {
        "description": (
            "build_land_cover_layer's obstacle and patch polygon overlay. "
            "base_* is the reference run with no obstacles and no patches; "
            "final_* is the same run with them. The port turns one into the other."
        ),
        "source": "src/argusnet/world/procedural.py::build_land_cover_layer",
        "generator": "scripts/capture_polygon_overlay_fixture.py",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
