#!/usr/bin/env python3
"""Record FrontierPlanner's cell selection and gap detection.

Two things make this fixture what it is.

**The chosen cell is decided by `np.argmax`, which breaks ties on the FIRST
index.** Candidates come from `np.where(grid == 0)`, which is row-major: i
ascending, then j ascending. A port that iterated the grid column-major, or
that kept the last maximum rather than the first, would pick a different cell
on any grid with a tie — and ties are the normal case at low coverage, where
the gradient term is zero and equidistant cells score identically. Several
cases below are deliberately symmetric for that reason.

**The coverage gradient goes through `scipy.ndimage.uniform_filter`,** which
computes a 3x3 mean as two separable passes of `1/3` weights and is then
multiplied by 9. The code's comment calls the result a neighbour count, and it
is one mathematically, but the float arithmetic need not land exactly on an
integer. The gradient values are recorded so a port computing the count as an
integer can be checked against what the reference actually produced rather
than against what it meant.

Usage:
    python3 scripts/capture_frontier_fixture.py --out tests/fixtures/frontier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.mapping.coverage import CoverageMap  # noqa: E402
from argusnet.mapping.occupancy import GridBounds  # noqa: E402
from argusnet.planning.frontier import (  # noqa: E402
    ClaimedCells,
    FrontierConfig,
    FrontierPlanner,
)


def exact(value: float) -> str:
    return repr(float(value))


def build(nx: int, ny: int, covered: list[tuple[int, int]]) -> CoverageMap:
    """A coverage map of `nx` x `ny` unit cells with the listed cells marked."""
    bounds = GridBounds(
        x_min_m=0.0, x_max_m=float(nx), y_min_m=0.0, y_max_m=float(ny), resolution_m=1.0
    )
    cmap = CoverageMap(bounds)
    # Mark by cell centre, so the mapping from (i, j) to a marked cell is the
    # grid's own and not this script's arithmetic.
    cmap.mark([(i + 0.5, j + 0.5) for i, j in covered])
    return cmap


def grid_of(cmap: CoverageMap) -> list[list[int]]:
    return [[int(v) for v in row] for row in cmap.count_grid]


SCENES: dict[str, tuple[int, int, list[tuple[int, int]]]] = {
    # Nothing covered: every cell is a candidate and the gradient term is off,
    # so the choice is distance alone -- and symmetric, so argmax's
    # first-index rule decides it.
    "empty_5x5": (5, 5, []),
    # A single covered cell: still below the 25% gradient floor.
    "one_covered_5x5": (5, 5, [(2, 2)]),
    # Just over half covered, so the gradient ramp is partly on.
    "half_covered_6x6": (
        6,
        6,
        [(i, j) for i in range(6) for j in range(3)] + [(0, 3), (1, 3)],
    ),
    # Above 75%: the ramp is saturated and the gradient dominates.
    "mostly_covered_6x6": (
        6,
        6,
        [(i, j) for i in range(6) for j in range(6) if not (i in (2, 3) and j in (2, 3))],
    ),
    # One enclosed hole, surrounded on all sides.
    "single_hole_5x5": (
        5,
        5,
        [(i, j) for i in range(5) for j in range(5) if (i, j) != (2, 2)],
    ),
    # A hole touching the border is NOT enclosed: binary_fill_holes only fills
    # background that does not reach the edge, so this must yield no gaps.
    "edge_notch_5x5": (
        5,
        5,
        [(i, j) for i in range(5) for j in range(5) if (i, j) != (0, 2)],
    ),
    # Two separate enclosed holes, one of them two cells across.
    "two_holes_7x7": (
        7,
        7,
        [(i, j) for i in range(7) for j in range(7) if (i, j) not in {(1, 1), (4, 4), (4, 5)}],
    ),
    # Fully covered: no candidates at all, so selection must return None.
    "full_4x4": (4, 4, [(i, j) for i in range(4) for j in range(4)]),
    # A non-square grid, so a port that transposed i and j fails.
    "wide_3x7": (3, 7, [(0, 0), (1, 0), (2, 0)]),
    # Only the centre is covered. A drone standing on it is equidistant from
    # the four orthogonal neighbours and the gradient is off, so those four
    # score IDENTICALLY -- the only thing choosing between them is argmax's
    # first-index rule over row-major candidates. Every other scene here has a
    # clear winner, so without this one that rule is never exercised.
    "tie_3x3": (3, 3, [(1, 1)]),
    # A hole whose ONLY route to the border is diagonal. Under the 4-connected
    # fill the reference uses it is enclosed; an 8-connected fill would reach
    # it from outside and report no hole at all.
    "diagonal_leak_5x5": (
        5,
        5,
        [(i, j) for i in range(5) for j in range(5) if (i, j) not in {(2, 2), (1, 1), (0, 0)}],
    ),
    # The uncovered background touches ONLY the bottom and right edges, so a
    # flood fill seeded from the top and left alone would call it enclosed.
    "bottom_right_opening_5x5": (
        5,
        5,
        [(i, j) for i in range(5) for j in range(5) if not (i >= 3 and j >= 3)],
    ),
    # Large and mostly covered. NEAR the origin is a 2x2 uncovered block,
    # whose cells each see only five covered neighbours; FAR away is a single
    # uncovered cell seeing eight. Normalising distance by the grid diagonal
    # shrinks the distance term to [0, 1], so the three-neighbour advantage
    # (worth 6 at a saturated weight) carries the far cell; without the
    # normalisation the raw eleven-cell distance gap buries it and the near
    # block wins instead.
    "scale_12x12": (
        12,
        12,
        [
            (i, j)
            for i in range(12)
            for j in range(12)
            if not (1 <= i <= 2 and 1 <= j <= 2) and (i, j) != (9, 9)
        ],
    ),
    # Big enough that a claim at one corner excludes part of the grid without
    # excluding all of it, so the radius comparison itself decides rather than
    # the all-excluded fallback.
    "empty_9x9": (9, 9, []),
    # The uncovered strip reaches the border only through the INTERIOR of the
    # bottom row. Seeding a flood fill from the left and right edges plus the
    # top row would leave it unreached and report three holes where there are
    # none, and the four corners the other loops touch do not save it.
    "bottom_row_interior_5x5": (
        5,
        5,
        [(i, j) for i in range(5) for j in range(5) if (i, j) not in {(4, 2), (3, 2), (2, 2)}],
    ),
}


def _score_margin(planner, cmap, drone_xy, claimed, drone_id) -> float:
    """The gap between the best and second-best scores, or 0.0 for a tie.

    Recomputes the reference's scoring to measure it. That is a reimplementation
    and would be unacceptable for an EXPECTED value -- it is used only to size
    the float noise the port has to survive, never as a target.
    """
    from scipy.ndimage import uniform_filter  # noqa: PLC0415

    cfg = planner.cfg
    grid = cmap.count_grid
    nx, ny = grid.shape
    ii, jj = np.where(grid == 0)
    if len(ii) == 0:
        return 0.0
    other = claimed.others(drone_id)
    if other:
        ei = np.array([c[0] for c in other], dtype=int)
        ej = np.array([c[1] for c in other], dtype=int)
        r = cfg.exclusion_radius_cells
        keep = np.array(
            [
                not np.any((np.abs(ei - ii[k]) <= r) & (np.abs(ej - jj[k]) <= r))
                for k in range(len(ii))
            ]
        )
        ci, cj = (ii[keep], jj[keep]) if keep.any() else (ii, jj)
    else:
        ci, cj = ii, jj
    gi, gj = cmap.bounds.xy_to_ij(float(drone_xy[0]), float(drone_xy[1]))
    dist = np.hypot(ci - gi, cj - gj) / max(float(np.hypot(nx, ny)), 1.0)
    fraction = cmap.stats.coverage_fraction
    weight = cfg.coverage_gradient_weight * max(0.0, min(1.0, (fraction - 0.25) / 0.50))
    if weight > 0.0:
        grad_map = uniform_filter((grid > 0).astype(float), size=3, mode="constant") * 9.0
        grad = grad_map[ci, cj]
    else:
        grad = np.zeros(len(ci))
    score = -cfg.distance_weight * dist + weight * grad
    if len(score) < 2:
        return 0.0
    ordered = np.sort(score)[::-1]
    return float(ordered[0] - ordered[1])


def capture() -> dict:
    cfg = FrontierConfig()
    planner = FrontierPlanner(cfg)

    scenes = []
    for name, (nx, ny, covered) in SCENES.items():
        cmap = build(nx, ny, covered)
        stats = cmap.stats
        scenes.append(
            {
                "name": name,
                "nx": nx,
                "ny": ny,
                "grid": grid_of(cmap),
                "coverage_fraction": exact(stats.coverage_fraction),
                "covered_cells": stats.covered_cells,
                "total_cells": stats.total_cells,
                "gap_cells": [list(c) for c in planner.find_gap_cells(cmap)],
            }
        )

    # The gradient map, recorded wherever the ramp is actually on. See the
    # module docstring: this is what `uniform_filter` produced, not what the
    # neighbour count "should" be.
    from scipy.ndimage import uniform_filter  # noqa: PLC0415

    gradients = []
    for name, (nx, ny, covered) in SCENES.items():
        cmap = build(nx, ny, covered)
        fraction = cmap.stats.coverage_fraction
        weight = cfg.coverage_gradient_weight * max(0.0, min(1.0, (fraction - 0.25) / 0.50))
        grad = uniform_filter((cmap.count_grid > 0).astype(float), size=3, mode="constant") * 9.0
        gradients.append(
            {
                "scene": name,
                "gradient_weight": exact(weight),
                "map": [[exact(v) for v in row] for row in grad],
            }
        )
    if not any(float(g["gradient_weight"]) > 0.0 for g in gradients):
        raise AssertionError("no scene turns the gradient term on")
    if not any(float(g["gradient_weight"]) == 0.0 for g in gradients):
        raise AssertionError("no scene leaves the gradient term off")

    selections = []
    for scene, drone_xy, claims, drone_id, note in [
        ("empty_5x5", (0.5, 0.5), {}, "d1", "corner start: the nearest cell is the corner itself"),
        (
            "empty_5x5",
            (2.5, 2.5),
            {},
            "d1",
            "centre start: the drone's own cell wins at distance 0",
        ),
        ("tie_3x3", (1.5, 1.5), {}, "d1", "a genuine four-way tie, broken by argmax"),
        (
            "empty_5x5",
            (2.5, 2.5),
            {"d2": (2, 2)},
            "d1",
            "another drone's claim excludes a 3-cell radius",
        ),
        # Every candidate excluded, so the fallback ignores exclusion entirely.
        (
            "empty_5x5",
            (2.5, 2.5),
            {"d2": (2, 2), "d3": (0, 0), "d4": (4, 4)},
            "d1",
            "all candidates excluded: the fallback ignores the exclusion",
        ),
        # A drone's OWN claim must not exclude it.
        ("empty_5x5", (0.5, 0.5), {"d1": (0, 0)}, "d1", "a drone's own claim does not exclude it"),
        ("one_covered_5x5", (4.5, 4.5), {}, "d1", "gradient still off below 25%"),
        ("half_covered_6x6", (0.5, 5.5), {}, "d1", "gradient partly on"),
        ("mostly_covered_6x6", (0.5, 0.5), {}, "d1", "gradient saturated"),
        ("full_4x4", (0.5, 0.5), {}, "d1", "nothing uncovered: None"),
        ("wide_3x7", (1.5, 6.5), {}, "d1", "non-square grid"),
        ("diagonal_leak_5x5", (0.5, 4.5), {}, "d1", "diagonal leak scene"),
        ("bottom_right_opening_5x5", (0.5, 0.5), {}, "d1", "opening on the far edges"),
        # On a large grid the normalised distance lets a far cell with many
        # covered neighbours beat a near one with few.
        ("scale_12x12", (0.5, 0.5), {}, "d1", "normalised distance lets the gradient compete"),
        # Exactly at the exclusion radius: Chebyshev distance 3 from the claim.
        # Inclusive excludes it, exclusive does not.
        # The drone stands exactly `exclusion_radius_cells` from the claim.
        # Inclusive excludes its own cell and sends it one step away;
        # exclusive keeps it and it stays put.
        ("empty_9x9", (3.5, 3.5), {"d2": (0, 0)}, "d1", "a claim exactly at the exclusion radius"),
    ]:
        nx, ny, covered = SCENES[scene]
        cmap = build(nx, ny, covered)
        claimed = ClaimedCells()
        for did, cell in claims.items():
            claimed.claim(did, cell)
        cell = planner.select_frontier_cell(
            cmap, np.asarray(drone_xy, dtype=float), claimed, drone_id
        )
        selections.append(
            {
                "scene": scene,
                "note": note,
                "drone_xy": [exact(drone_xy[0]), exact(drone_xy[1])],
                "drone_id": drone_id,
                "claims": [[did, list(cell_)] for did, cell_ in claims.items()],
                "expected": None if cell is None else list(cell),
                # How far ahead the winner was. The port computes the
                # neighbour count as an integer, where the reference's
                # `uniform_filter` carries ~1e-15 of float noise, so a margin
                # narrower than that noise would make the two disagree. The
                # Rust test checks every margin against it rather than
                # assuming the noise is harmless. A tie (margin 0.0) is fine
                # and expected -- argmax's first-index rule decides those, and
                # both implementations order candidates the same way.
                "score_margin": exact(_score_margin(planner, cmap, drone_xy, claimed, drone_id)),
            }
        )
    if not any(s["expected"] is None for s in selections):
        raise AssertionError("no selection case returns None")
    chosen = {tuple(s["expected"]) for s in selections if s["expected"] is not None}
    if len(chosen) < 4:
        raise AssertionError(
            f"only {len(chosen)} distinct cells chosen; the cases are not discriminating"
        )
    # A genuine tie must be present, or argmax's first-index rule -- the thing
    # this fixture claims to pin -- is never exercised.
    ties = [s for s in selections if s["expected"] is not None and float(s["score_margin"]) == 0.0]
    if not ties:
        raise AssertionError(
            "no selection case is an exact tie, so the first-index tie-break is "
            "untested and a port keeping the LAST maximum would pass"
        )

    with_gaps = [s for s in scenes if s["gap_cells"]]
    without = [s for s in scenes if not s["gap_cells"]]
    if not with_gaps:
        raise AssertionError("no scene has an enclosed hole")
    if not without:
        raise AssertionError("no scene is hole-free")
    notch = next(s for s in scenes if s["name"] == "edge_notch_5x5")
    if notch["gap_cells"]:
        raise AssertionError(
            "the edge notch was reported as an enclosed hole; the border rule "
            "is not being exercised"
        )

    return {
        "source": "src/argusnet/planning/frontier.py",
        "note": (
            "FrontierPlanner read back from the reference. The chosen cell is "
            "an np.argmax over row-major candidates, so ties go to the lowest "
            "(i, j); several scenes are symmetric to pin that."
        ),
        "config": {
            "distance_weight": exact(cfg.distance_weight),
            "coverage_gradient_weight": exact(cfg.coverage_gradient_weight),
            "exclusion_radius_cells": cfg.exclusion_radius_cells,
            "gap_fill_min_fraction": exact(cfg.gap_fill_min_fraction),
        },
        "scenes": scenes,
        "gradients": gradients,
        "selections": selections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    counts = {k: len(v) for k, v in document.items() if isinstance(v, list)}
    print(f"wrote {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
