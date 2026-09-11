#!/usr/bin/env python3
"""Record _semantic_masks and the preset table, bit-for-bit.

`_semantic_masks` rewrites the height field while it runs, and the ORDER of
those rewrites is the contract. Four reads are stale by design -- they use a
value computed before an edit that came after -- and each is recorded here
because reproducing the staleness is the whole job:

* `elev_norm` is normalised from the ORIGINAL heights and then used at the very
  end for `ridge_mask`, after rivers, lakes, coast, water levelling, smoothing
  and the ground-plane floor have all edited the field. Recomputing it from the
  final heights is the obvious cleanup and changes every ridge.
* `lowland_mask` uses the FIRST slope field; `steep_slope_mask` uses a second
  one computed after smoothing.
* `valley_mask` is a SNAPSHOT of `lowland_mask` taken before the preset
  branches; `lake_district` and `coastal` widen `lowland_mask` only.
* `_dilate_mask(water_mask, 0)` dilates nothing.

Every preset is run, so every branch is exercised, and the whole height field
plus all six masks are recorded for each. The preset table itself is dumped
field by field: 12 profiles x 15 fields is exactly the kind of table where a
transcription typo survives review.

Usage:
    python3 scripts/capture_semantic_fixture.py --out tests/fixtures/semantic.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.world.procedural import (  # noqa: E402
    _PRESET_PROFILES,
    _semantic_masks,
    _slope_grid,
    terrain_profile_for_preset,
)

MASK_NAMES = ("water", "ridge", "valley", "steep_slope", "lowland", "road")


def exact(value: float) -> str:
    return repr(float(value))


def flat(array: np.ndarray) -> list[str]:
    return [exact(v) for v in np.asarray(array, dtype=float).ravel()]


def flat_bool(array: np.ndarray) -> list[bool]:
    return [bool(v) for v in np.asarray(array, dtype=bool).ravel()]


class Bounds:
    """The attributes `_semantic_masks` reads off a Bounds2D."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.x_min_m = x_min
        self.x_max_m = x_max
        self.y_min_m = y_min
        self.y_max_m = y_max

    @property
    def width_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        return self.y_max_m - self.y_min_m


SCENES: dict[str, dict] = {
    # The workhorse: real relief, a 25 m grid, heights well above the ground
    # plane so the water branches stay quiet unless a preset carves them.
    "base": {
        "nx": 14,
        "ny": 13,
        "resolution": 25.0,
        "ground": 2.0,
        "lift": 30.0,
        "tilt": 40.0,
        "rough": 9.0,
    },
    # Heights pressed down ONTO the ground plane, so `heights <= ground + 1`
    # actually selects cells. Without this the whole water-levelling path is
    # dead for every preset that does not carve its own water.
    "low": {
        "nx": 12,
        "ny": 12,
        "resolution": 20.0,
        "ground": 2.0,
        "lift": 2.0,
        "tilt": 3.0,
        "rough": 1.2,
    },
    # A span under one metre, so `max(h_max - h_min, 1.0)` in the elevation
    # normalisation stops being a no-op.
    "flat": {
        "nx": 10,
        "ny": 10,
        "resolution": 15.0,
        "ground": 0.0,
        "lift": 40.0,
        "tilt": 0.3,
        "rough": 0.12,
    },
    # A scene only a few metres across, where the `max(..., 1.0)` floors on the
    # river wavelength and widths bite.
    "tiny": {
        "nx": 9,
        "ny": 9,
        "resolution": 0.4,
        "ground": 0.0,
        "lift": 5.0,
        "tilt": 2.0,
        "rough": 0.8,
    },
    # A river scene fine enough that the amplitude difference between the two
    # presets (0.035 vs 0.015 of the scene scale) exceeds one cell. That ratio
    # is 0.02 * (n - 1) cells regardless of resolution, so it needs n >= 60 on
    # the axis the river is measured along -- a 13-row grid cannot see it.
    "river_fine": {
        "nx": 8,
        "ny": 64,
        "resolution": 5.0,
        "ground": 0.0,
        "lift": 40.0,
        "tilt": 25.0,
        "rough": 4.0,
    },
    # Small enough that `max(scene_scale * 0.23, 1.0)` clamps the river
    # wavelength (scale 4.0 -> 0.92), AND sampled finely enough that the
    # resulting phase shift actually moves the river band across cells. The
    # dimensions were SEARCHED for -- and then searched again for the
    # SMALLEST grid that still discriminates, because the fixture is committed:
    # on a coarse tiny scene the floor moves the river centreline by a few
    # millimetres, far under one cell, and the mutant that drops it survives.
    "tiny_fine": {
        "nx": 30,
        "ny": 12,
        "resolution": 0.13793103448275862,
        "ground": 0.0,
        "lift": 5.0,
        "tilt": 2.0,
        "rough": 0.5,
    },
    # A tall, finely sampled y axis so the coastal 0.26 and 0.34 fractions fall
    # on different rows. On a coarse grid both land between the same two.
    "fine_coast": {
        "nx": 6,
        "ny": 40,
        "resolution": 4.0,
        "ground": 1.0,
        "lift": 6.0,
        "tilt": 9.0,
        "rough": 1.1,
    },
}


def make_field(scene: dict, seed: int, layout: str = "ij") -> tuple:
    """A height field plus its coordinate grids, for one scene configuration.

    `layout` selects which meshgrid convention the grids use:

    * `"ij"` -- `meshgrid(..., indexing="ij")`, so axis 0 is x.
    * `"xy"` -- numpy's DEFAULT, so axis 0 is y. This is what
      `procedural_terrain_grid` actually passes, and it is not equivalent:
      `_smooth_heights` sums its four neighbours in a fixed POSITIONAL order,
      which transposition permutes, and floating-point addition is not
      associative. The two layouts agree on every mask and differ on the
      height field by up to ~7e-15 -- small, but not zero, and this fixture
      compares bit patterns.
    """
    nx, ny = scene["nx"], scene["ny"]
    resolution = scene["resolution"]
    rng = np.random.default_rng(seed)
    xs = np.arange(nx, dtype=float) * resolution
    ys = np.arange(ny, dtype=float) * resolution
    if layout == "ij":
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
    else:
        xx, yy = np.meshgrid(xs, ys)
    base = (
        scene["lift"]
        + (xx / max(xs[-1], 1.0)) * scene["tilt"]
        - (yy / max(ys[-1], 1.0)) * (scene["tilt"] * 0.375)
    )
    heights = base + rng.normal(0.0, scene["rough"], size=xx.shape)
    bounds = Bounds(0.0, float(xs[-1]), 0.0, float(ys[-1]))
    return heights, xx, yy, bounds


# Ramp steps whose slope lands BIT-EXACTLY on each steep-slope threshold,
# found by scanning ULPs around tan(threshold). Without these, `slope > t` and
# `slope >= t` are indistinguishable -- 2,000,000 random gradients produced no
# exact hit, so the case has to be constructed rather than sampled.
EXACT_SLOPE_STEPS = {0.52: 0.5725618302516684, 0.62: 0.7139090066592402}


def make_ramp(step: float, nx: int, ny: int) -> tuple:
    """A pure ramp along x at 1 m spacing, so every derivative equals `step`."""
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    heights = xx * step
    return heights, xx, yy, Bounds(0.0, float(xs[-1]), 0.0, float(ys[-1]))


def profile_with(base, **overrides):
    """A profile with some fields replaced.

    The preset table alone cannot reach several parameters: no preset has a
    water_bias above 0.35 or strictly between 0 and 0.01, so both clamps in
    `min(max(water_bias, 0.01), 0.35)` are no-ops on every real preset. The
    port takes the profile as data, so the fixture drives those directly.
    """
    return dataclasses.replace(base, **overrides)


def profile_record(profile) -> dict:
    fields = [f.name for f in dataclasses.fields(profile)]
    record = {}
    for field in fields:
        value = getattr(profile, field)
        if field == "name":
            record[field] = value
        elif field in ("ridge_count", "valley_count", "smoothing_passes"):
            record[field] = int(value)
        else:
            record[field] = exact(value)
    return record


def build() -> dict:
    presets = []
    for name in sorted(_PRESET_PROFILES):
        record = profile_record(_PRESET_PROFILES[name])
        record["lookup_name"] = name
        presets.append(record)

    fallbacks = [
        {"lookup_name": probe, "resolved_name": terrain_profile_for_preset(probe).name}
        for probe in ("no_such_preset", "", "Default", "alpine")
    ]

    # Explicit quantile and median RESULTS. An earlier version of the parity
    # test only asserted these were finite, which is not an assertion at all --
    # all three quantile mutants survived it.
    rng = np.random.default_rng(99)
    quantile_cases = []
    median_cases = []
    for n in (1, 2, 3, 4, 5, 8, 13, 40):
        values = list(rng.normal(20.0, 30.0, size=n))
        # 0.5 lands exactly on numpy's lerp switch; the others straddle it, and
        # the ones the reference actually uses are 0.18, 0.28 and the clamped
        # water bias.
        for q in (0.0, 0.01, 0.18, 0.25, 0.28, 0.35, 0.5, 0.72, 0.99, 1.0):
            quantile_cases.append(
                {
                    "values": [exact(v) for v in values],
                    "q": exact(q),
                    "result": exact(np.quantile(np.array(values), q)),
                }
            )
        median_cases.append(
            {
                "values": [exact(v) for v in values],
                "result": exact(np.median(np.array(values))),
            }
        )

    default_profile = _PRESET_PROFILES["default"]
    runs = []

    def add_run(
        label: str, scene_name: str, preset_name: str, profile, field=None, layout: str = "ij"
    ) -> None:
        if field is None:
            scene = SCENES[scene_name]
            heights, xx, yy, bounds = make_field(scene, seed=20260911, layout=layout)
        else:
            heights, xx, yy, bounds, scene = field
        assert_coordinates_are_reconstructible(xx, yy, scene["resolution"], layout)
        result_heights, masks = _semantic_masks(
            preset_name=preset_name,
            profile=profile,
            heights_m=heights.copy(),
            xx=xx,
            yy=yy,
            bounds_xy_m=bounds,
            resolution_m=scene["resolution"],
            seed=4242,
            ground_plane_m=scene["ground"],
        )
        runs.append(
            {
                "label": label,
                "scene": scene_name,
                "preset_name": preset_name,
                "profile": profile_record(profile),
                "layout": layout,
                "shape": [int(heights.shape[0]), int(heights.shape[1])],
                "resolution_m": exact(scene["resolution"]),
                "seed": 4242,
                "ground_plane_m": exact(scene["ground"]),
                "bounds": {
                    "x_min_m": exact(bounds.x_min_m),
                    "x_max_m": exact(bounds.x_max_m),
                    "y_min_m": exact(bounds.y_min_m),
                    "y_max_m": exact(bounds.y_max_m),
                },
                # `xx` and `yy` are omitted deliberately: they are exactly
                # `i * resolution` and `j * resolution`, so storing them would
                # double the fixture to no purpose. The parity test rebuilds
                # them and `check_discriminating` asserts that is valid.
                "heights_in_m": flat(heights),
                "heights_out_m": flat(result_heights),
                "masks": {name: flat_bool(masks[name]) for name in MASK_NAMES},
            }
        )

    # Every preset on the workhorse scene, plus the unknown-name fallback.
    for preset_name in sorted(_PRESET_PROFILES) + ["no_such_preset"]:
        add_run(f"base/{preset_name}", "base", preset_name, terrain_profile_for_preset(preset_name))

    # Every preset again pressed onto the ground plane, so the water-levelling
    # and ground-floor paths are live rather than dead.
    for preset_name in sorted(_PRESET_PROFILES):
        add_run(f"low/{preset_name}", "low", preset_name, terrain_profile_for_preset(preset_name))

    # Scenes that reach a specific floor or fraction.
    add_run("flat/default", "flat", "default", default_profile)
    add_run("flat/alpine", "flat", "alpine", terrain_profile_for_preset("alpine"))
    for preset_name in ("river_valley", "mountain_pass", "lake_district"):
        add_run(f"tiny/{preset_name}", "tiny", preset_name, terrain_profile_for_preset(preset_name))
    for preset_name in ("river_valley", "mountain_pass"):
        add_run(
            f"river_fine/{preset_name}",
            "river_fine",
            preset_name,
            terrain_profile_for_preset(preset_name),
        )
        add_run(
            f"tiny_fine/{preset_name}",
            "tiny_fine",
            preset_name,
            terrain_profile_for_preset(preset_name),
        )
    add_run("fine_coast/coastal", "fine_coast", "coastal", terrain_profile_for_preset("coastal"))
    add_run(
        "fine_coast/river_valley",
        "fine_coast",
        "river_valley",
        terrain_profile_for_preset("river_valley"),
    )

    # PROFILE OVERRIDES, for parameters no preset reaches.
    add_run(
        "override/zero_water_bias", "low", "default", profile_with(default_profile, water_bias=0.0)
    )
    add_run(
        "override/tiny_water_bias",
        "low",
        "default",
        profile_with(default_profile, water_bias=0.005),
    )
    add_run(
        "override/huge_water_bias", "low", "default", profile_with(default_profile, water_bias=0.5)
    )
    add_run(
        "override/bias_at_upper_clamp",
        "low",
        "default",
        profile_with(default_profile, water_bias=0.35),
    )
    add_run(
        "override/rough_below_branch",
        "base",
        "default",
        profile_with(default_profile, roughness=0.69),
    )
    add_run(
        "override/rough_at_branch", "base", "default", profile_with(default_profile, roughness=0.7)
    )
    add_run(
        "override/no_smoothing",
        "base",
        "default",
        profile_with(default_profile, smoothing_passes=0),
    )
    add_run(
        "override/heavy_smoothing",
        "base",
        "default",
        profile_with(default_profile, smoothing_passes=6),
    )
    # A zero-water-bias preset on a scene where water cells DO exist, so the
    # `> 0.0` vs `>= 0.0` branch on the water level is observable.
    add_run(
        "override/zero_bias_urban", "low", "urban_flat", terrain_profile_for_preset("urban_flat")
    )

    # THE LAYOUT THE REAL CALLER USES. `procedural_terrain_grid` builds its
    # coordinate grids with numpy's DEFAULT meshgrid, so axis 0 is y -- the
    # transpose of every run above. `_semantic_masks` is layout-agnostic for
    # the masks, but NOT for the heights: `_smooth_heights` sums its four
    # neighbours in a fixed positional order, transposition permutes that
    # order, and floating-point addition is not associative. The two layouts
    # differ by up to ~7e-15, which is nothing physically and everything to a
    # fixture that compares bit patterns. Without these runs the port is only
    # pinned against a layout production never uses.
    for preset_name in ("default", "coastal", "river_valley", "lake_district", "urban_flat"):
        add_run(
            f"xy/{preset_name}",
            "base",
            preset_name,
            terrain_profile_for_preset(preset_name),
            layout="xy",
        )

    # Slopes landing EXACTLY on each steep-slope threshold. Smoothing is turned
    # off and the ground plane put well below the ramp so the height field
    # reaches the second slope computation untouched; roughness selects which
    # threshold applies (< 0.7 -> 0.52, otherwise 0.62). Without these,
    # `slope > t` and `slope >= t` are indistinguishable.
    for threshold, roughness in ((0.52, 0.6), (0.62, 0.9)):
        ramp = make_ramp(EXACT_SLOPE_STEPS[threshold], 6, 5)
        scene = {"nx": 6, "ny": 5, "resolution": 1.0, "ground": -10.0}
        add_run(
            f"exact_slope/{threshold}",
            "exact_slope",
            "default",
            profile_with(default_profile, roughness=roughness, smoothing_passes=0),
            field=(*ramp, scene),
        )

    return {
        "presets": presets,
        "fallbacks": fallbacks,
        "quantiles": quantile_cases,
        "medians": median_cases,
        "runs": runs,
    }


def assert_coordinates_are_reconstructible(xx, yy, resolution: float, layout: str) -> None:
    """The fixture omits xx/yy; prove they follow from the shape and layout."""
    rows, cols = xx.shape
    for a in range(rows):
        for b in range(cols):
            # "ij": axis 0 is x. "xy": axis 0 is y -- numpy's default, and what
            # the real caller uses.
            want_x = a * resolution if layout == "ij" else b * resolution
            want_y = b * resolution if layout == "ij" else a * resolution
            if xx[a, b] != want_x or yy[a, b] != want_y:
                raise SystemExit(
                    f"coordinate grids are not reconstructible at ({a}, {b}) "
                    f"under layout {layout!r}; the fixture cannot omit them"
                )


def check_discriminating(document: dict) -> None:
    """Implement each stale-by-design read the "clean" way and require it to
    disagree. Not "do the masks look plausible"."""
    nx_ny = {}

    # (a) elev_norm recomputed from the FINAL heights must change some ridge.
    ridge_hits = 0
    for run in document["runs"]:
        nx, ny = run["shape"]
        nx_ny[run["label"]] = (nx, ny)
        final = np.array([float(v) for v in run["heights_out_m"]]).reshape(nx, ny)
        expected = np.array(run["masks"]["ridge"]).reshape(nx, ny)
        steep = np.array(run["masks"]["steep_slope"]).reshape(nx, ny)
        h_min, h_max = float(final.min()), float(final.max())
        fresh_norm = (final - h_min) / max(h_max - h_min, 1.0)
        wrong = (fresh_norm > 0.72) | (steep & (fresh_norm > 0.55))
        if not np.array_equal(wrong, expected):
            ridge_hits += 1
    if not ridge_hits:
        raise SystemExit(
            "no run distinguishes elev_norm-from-original-heights from "
            "elev_norm-recomputed-from-final-heights; the staleness is untested"
        )

    # (b) lowland_mask recomputed with the FINAL slope field must disagree.
    slope_hits = 0
    for run in document["runs"]:
        nx, ny = nx_ny[run["label"]]
        final = np.array([float(v) for v in run["heights_out_m"]]).reshape(nx, ny)
        late_slopes = _slope_grid(final, float(run["resolution_m"]))
        expected = np.array(run["masks"]["lowland"]).reshape(nx, ny)
        initial = np.array([float(v) for v in run["heights_in_m"]]).reshape(nx, ny)
        profile = run["profile"]
        ground = float(run["ground_plane_m"])
        if float(profile["water_bias"]) > 0.0:
            water_level = float(
                np.quantile(initial, min(max(float(profile["water_bias"]), 0.01), 0.35))
            )
        else:
            water_level = ground
        wrong = (initial <= water_level + (float(profile["relief_m"]) * 0.05)) & (
            late_slopes < 0.22
        )
        if not np.array_equal(wrong, expected):
            slope_hits += 1
    if not slope_hits:
        raise SystemExit(
            "no run distinguishes the first slope field from the second; "
            "lowland_mask's staleness is untested"
        )

    # (c) valley_mask must differ from lowland_mask somewhere, or the snapshot
    #     could be an alias.
    snapshot_hits = sum(
        1 for run in document["runs"] if run["masks"]["valley"] != run["masks"]["lowland"]
    )
    if not snapshot_hits:
        raise SystemExit(
            "valley_mask equals lowland_mask in every run; the pre-branch "
            "snapshot is indistinguishable from an alias"
        )

    # (d) every preset branch must actually fire: each named preset must
    #     produce a height field different from the fallback run's.
    fallback = next(r for r in document["runs"] if r["label"] == "base/no_such_preset")
    branching = [
        run["preset_name"]
        for run in document["runs"]
        if run["scene"] == "base"
        and run["preset_name"]
        in {
            "river_valley",
            "mountain_pass",
            "lake_district",
            "coastal",
            "urban_flat",
            "military_compound",
        }
        and run["heights_out_m"] != fallback["heights_out_m"]
    ]
    missing = {
        "river_valley",
        "mountain_pass",
        "lake_district",
        "coastal",
        "urban_flat",
        "military_compound",
    } - set(branching)
    if missing:
        raise SystemExit(
            "these preset branches changed nothing versus the fallback, so "
            f"they are untested: {sorted(missing)}"
        )

    # (e) the xy-layout runs must DIFFER from their ij twins, or they are
    #     redundant and the layout sensitivity is untested.
    by_label = {run["label"]: run for run in document["runs"]}
    layout_hits = 0
    for label, run in by_label.items():
        if not label.startswith("xy/"):
            continue
        twin = by_label.get("base/" + label[len("xy/") :])
        if twin is None:
            continue
        rows, cols = run["shape"]
        mine = np.array([float(v) for v in run["heights_out_m"]]).reshape(rows, cols)
        theirs = np.array([float(v) for v in twin["heights_out_m"]]).reshape(
            twin["shape"][0], twin["shape"][1]
        )
        if not np.array_equal(mine, theirs.T):
            layout_hits += 1
    if not layout_hits:
        raise SystemExit(
            "every xy-layout run is a bit-exact transpose of its ij twin, so "
            "the layout sensitivity of _smooth_heights is untested"
        )

    print(
        f"discriminating: {ridge_hits} runs pin elev_norm staleness, "
        f"{slope_hits} pin the two slope fields, {snapshot_hits} pin the "
        f"valley snapshot, {len(branching)} preset branches fire, "
        f"{layout_hits} xy-layout runs differ from their ij twins"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = build()
    check_discriminating(document)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators, not indent=2: this fixture carries whole grids, and
    # a line per float more than doubles it. The repo's own rule
    # (PERFORMANCE_AND_BENCHMARKING.md 5.5) is to keep large numeric artifacts
    # out of the tree, so the file is kept as small as the coverage allows.
    args.out.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(
        f"wrote {args.out}: {len(document['presets'])} presets, "
        f"{len(document['fallbacks'])} fallbacks, "
        f"{len(document['quantiles'])} quantiles, {len(document['medians'])} medians, "
        f"{len(document['runs'])} runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
