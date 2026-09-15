#!/usr/bin/env python3
"""Record procedural_terrain_grid, bit-for-bit.

This is the top of the terrain builder: the function that turns a preset name
and a seed into the height field and semantic masks everything downstream reads.
It composes the P4.3a noise primitives, the P4.3b grid operations and the P4.3c
semantic pass over a single numpy `default_rng` stream.

WHY IT IS COMPARED BIT-FOR-BIT RATHER THAN STATISTICALLY

ADR-006 recommends re-baselining rather than reproducing numpy's PRNG, on the
grounds of cost. That reasoning was written about `build_observations`, which
builds a generator per (node, target) pair per tick and uses SeedSequence.spawn.
Here there is ONE generator and 12 to 46 scalar `uniform(low, high)` draws, and
`scripts/capture_numpy_rng_fixture.py` proves that stream reproducible. The
benefit is concrete: terrain the Rust runtime generates is identical to terrain
the Python scene baker writes, so the two agree about the world during the
transition. An envelope cannot give that.

THE DRAW ORDER IS THE CONTRACT

One generator serves the whole function, so every draw shifts the ones after it.
The valley loop is the sharpest case: its first iteration takes ONE uniform for
`river_valley` and `mountain_pass` (a near-axis-aligned angle, offset hardcoded
to zero) and TWO for every other preset. That single missing draw realigns the
entire remaining stream, so the preset silently decides every later value.

Usage:
    python3 scripts/capture_terrain_grid_fixture.py --out tests/fixtures/terrain_grid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.world.environment import Bounds2D  # noqa: E402
from argusnet.world.procedural import (  # noqa: E402
    _PRESET_PROFILES,
    procedural_terrain_grid,
)

MASK_NAMES = ("water", "lowland", "valley", "steep_slope", "ridge", "road")


def f(value: float) -> str:
    return repr(float(value))


def bits(values) -> str:
    return "".join("1" if bool(v) else "0" for v in values)


def scenes() -> list[dict]:
    """Every preset, plus the cases the preset table cannot reach on its own.

    Every preset is run because `ridge_count` and `valley_count` differ between
    them and so does the draw count -- 12 for `urban_flat`, 46 for
    `mountain_pass` -- which means each preset exercises a different alignment
    of the stream.
    """
    base = dict(width_m=260.0, height_m=180.0, resolution_m=20.0, seed=20260911)
    cases = [dict(name=f"preset/{p}", preset=p, **base) for p in sorted(_PRESET_PROFILES)]
    cases += [
        # The two presets whose first valley takes one draw instead of two,
        # at a second seed: the branch is what aligns the rest of the stream,
        # so it is worth more than one sample.
        dict(name="river_valley/seed7", preset="river_valley", **{**base, "seed": 7}),
        dict(name="mountain_pass/seed7", preset="mountain_pass", **{**base, "seed": 7}),
        # detail_strength is a plain multiplier on the detail term and nothing
        # else; zero must remove it entirely rather than merely shrink it.
        dict(name="detail_zero", preset="alpine", **base, detail_strength=0.0),
        dict(name="detail_double", preset="alpine", **base, detail_strength=2.0),
        # A raised ground plane, so the final `maximum(heights, ground_plane)`
        # actually clamps something. It has to sit INSIDE the preset's height
        # range: the first attempt used 40 m on `coastal`, whose terrain tops
        # out at 16.5 m, and flattened the entire grid to a single value -- a
        # scene that proves the clamp fires and nothing else. `rolling_highlands`
        # spans 15 to 91 m, so 27 m clamps roughly the lowest quarter and leaves
        # the rest to be compared.
        dict(name="ground_plane_high", preset="rolling_highlands", **base, ground_plane_m=27.0),
        # water_bias == 0 skips the lowland deepening entirely, which is not
        # the same as clamping zero up to 0.01.
        dict(name="no_water_bias", preset="urban_flat", **base),
        # A different aspect ratio and resolution: the grid shape feeds the
        # scene scale, which feeds every wavelength in the function.
        dict(
            name="tall_fine",
            preset="alpine",
            width_m=120.0,
            height_m=300.0,
            resolution_m=12.0,
            seed=20260911,
        ),
        dict(
            name="wide_coarse",
            preset="desert_canyon",
            width_m=400.0,
            height_m=110.0,
            resolution_m=34.0,
            seed=20260911,
        ),
        # Two floors in this function only bite on absurdly small scenes, and
        # nothing else in the fixture can reach them:
        #   * the macro wavelengths, `max(scene_scale * 0.42, 1.0)`, need a
        #     scene under 2.38 m;
        #   * the ridge crenulation wavelength, `max(width * 1.8, 10.0)`, needs
        #     a ridge width under 5.56 m, and width is `scene_scale * [0.045,
        #     0.13]`, so a scene under about 123 m.
        # A 2 m scene reaches both at once. It is not a realistic site, but it is
        # legal input, and a floor that no test can reach is a floor nobody can
        # prove is wired up.
        dict(
            name="micro_scene",
            preset="alpine",
            width_m=2.0,
            height_m=1.6,
            resolution_m=0.5,
            seed=20260911,
        ),
        # A small-but-sane site, so the crenulation floor is also exercised
        # somewhere a real caller might actually go.
        dict(
            name="small_site",
            preset="rolling_highlands",
            width_m=90.0,
            height_m=64.0,
            resolution_m=8.0,
            seed=20260911,
        ),
    ]
    return cases


def capture_scene(scene: dict) -> dict:
    x_min, y_min = -120.0, 75.0
    bounds = Bounds2D(
        x_min_m=x_min,
        x_max_m=x_min + scene["width_m"],
        y_min_m=y_min,
        y_max_m=y_min + scene["height_m"],
    )
    grid = procedural_terrain_grid(
        bounds_xy_m=bounds,
        preset_name=scene["preset"],
        seed=scene["seed"],
        resolution_m=scene["resolution_m"],
        detail_strength=float(scene.get("detail_strength", 1.0)),
        ground_plane_m=float(scene.get("ground_plane_m", 0.0)),
    )
    heights = np.asarray(grid.heights_m)
    rows, cols = heights.shape
    return {
        "name": scene["name"],
        "preset": scene["preset"],
        "seed": int(scene["seed"]),
        "resolution_m": f(scene["resolution_m"]),
        "detail_strength": f(scene.get("detail_strength", 1.0)),
        "ground_plane_m": f(scene.get("ground_plane_m", 0.0)),
        "bounds": {
            "x_min_m": f(bounds.x_min_m),
            "x_max_m": f(bounds.x_max_m),
            "y_min_m": f(bounds.y_min_m),
            "y_max_m": f(bounds.y_max_m),
        },
        "rows": int(rows),
        "cols": int(cols),
        "x_values_m": [f(v) for v in grid.x_values_m],
        "y_values_m": [f(v) for v in grid.y_values_m],
        "heights_m": [f(v) for v in heights.reshape(-1)],
        "masks": {
            name: bits(np.asarray(mask, dtype=bool).reshape(-1))
            for name, mask in sorted(grid.masks.items())
        },
        "mask_coverage_fraction": {
            name: f(float(np.mean(mask))) for name, mask in sorted(grid.masks.items())
        },
    }


def check_discriminating(cases: list[dict]) -> list[str]:
    """Refuse to write a fixture that cannot tell right from wrong.

    Every check here implements or detects a specific wrong version. Asserting
    that the heights merely look like terrain would pass for any smooth field.
    """
    problems = []
    by_name = {c["name"]: c for c in cases}

    # Two presets differing only in draw COUNT must give different terrain, or
    # the stream alignment is not observable.
    urban = by_name["preset/urban_flat"]["heights_m"]
    military = by_name["preset/military_compound"]["heights_m"]
    if urban == military:
        problems.append("urban_flat and military_compound produced identical heights")

    # detail_strength must actually scale the detail term.
    if by_name["detail_zero"]["heights_m"] == by_name["preset/alpine"]["heights_m"]:
        problems.append("detail_strength=0 changed nothing")
    if by_name["detail_double"]["heights_m"] == by_name["preset/alpine"]["heights_m"]:
        problems.append("detail_strength=2 changed nothing")

    # The ground plane must clamp, and must clamp only PART of the grid: a
    # plane above the preset's whole range flattens everything to one value and
    # proves only that the clamp fires.
    plane = float(by_name["ground_plane_high"]["ground_plane_m"])
    floored = [float(v) for v in by_name["ground_plane_high"]["heights_m"]]
    if min(floored) < plane:
        problems.append(f"a cell sits below the ground plane {plane}")
    if not any(v == plane for v in floored):
        problems.append("no cell sits ON the ground plane, so the clamp is untested")
    if all(v == plane for v in floored):
        problems.append("every cell is on the ground plane; the scene is degenerate")

    # The river/pass one-draw branch must matter: same preset at two seeds must
    # differ, and the branch presets must differ from their non-branch peers.
    if by_name["river_valley/seed7"]["heights_m"] == by_name["preset/river_valley"]["heights_m"]:
        problems.append("river_valley is insensitive to its seed")

    # Masks must not be uniformly empty or full, or the semantic pass is inert.
    interesting = 0
    for case in cases:
        for _name, mask in case["masks"].items():
            ones = mask.count("1")
            if 0 < ones < len(mask):
                interesting += 1
    if interesting < 20:
        problems.append(f"only {interesting} masks are non-degenerate")

    # Heights must vary within a scene.
    for case in cases:
        values = {v for v in case["heights_m"]}
        if len(values) < 10:
            problems.append(f"{case['name']}: only {len(values)} distinct heights")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [capture_scene(s) for s in scenes()]
    problems = check_discriminating(cases)
    for problem in problems:
        print(f"GAP  {problem}")
    if problems:
        print(f"\nABORT: {len(problems)} gap(s)")
        return 2

    total_cells = sum(c["rows"] * c["cols"] for c in cases)
    print(f"OK   {len(cases)} scenes, {total_cells} cells")
    document = {
        "description": (
            "procedural_terrain_grid heights and semantic masks. Regenerate "
            "with scripts/capture_terrain_grid_fixture.py."
        ),
        "mask_names": list(MASK_NAMES),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
