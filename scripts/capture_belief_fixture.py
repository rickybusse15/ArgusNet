#!/usr/bin/env python3
"""Record WorldBeliefQuery over a set of grids: cells, frontiers, corridors.

Every scenario records the FULL cell table -- every cell of the grid, with all
twelve fields -- plus the frontier ranking and a set of corridor probes. The
cell table is what makes the ragged-grid cases worth having: the divergence
they expose is one cell reporting itself outside the geofence, which no scalar
query returns and no aggregate notices.

Three things the cases are built to catch, none of which a "reasonable" port
gets right by accident:

* `coverage_at` tests the RAW inclusive bounds and returns 0 outside them,
  while `height_estimate_at` does not test them at all and clamps. On a grid
  whose span is not a whole number of cells the two disagree, and the last
  cell's own centre can fall past the bound -- so the cell reports itself
  OUTSIDE_GEOFENCE with zero coverage while the grid holds counts for it.
* `expected_coverage_gain` divides by the PATCH size, which shrinks to 4 at a
  corner and 6 along a side. A port that always divided by 9 agrees on every
  interior cell.
* `safe_corridor_between` walks `np.linspace`, which is `k * (1/(n-1))` with
  the last value forced to 1.0 -- not `k/(n-1)`. Those differ for n = 7 and
  n = 101 and AGREE at the default n = 16, so the sample counts below include
  7 and 101 deliberately. A fixture that only used the default would pass a
  port with the naive spelling.

Occupancy probabilities are read back out of the real OccupancyGrid rather
than being chosen, so the fixture carries the exact floats Python produced and
the Rust side is handed those same floats rather than re-deriving them from
log-odds.

Usage:
    python3 scripts/capture_belief_fixture.py --out tests/fixtures/belief.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.mapping.belief import WorldBeliefQuery  # noqa: E402
from argusnet.mapping.coverage import CoverageMap  # noqa: E402
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid  # noqa: E402


def exact(value: float) -> str:
    return repr(float(value))


def opt_exact(value: float | None) -> str | None:
    return None if value is None else exact(value)


@dataclass
class WorldMapStub:
    """Stands in for WorldMap.

    `WorldBeliefQuery` reads exactly two attributes off its world map --
    `mean_height_grid` and `coverage_map` — which `grep -n 'self.world_map'
    src/argusnet/mapping/belief.py` confirms. Using the real WorldMap would
    mean driving its scan-accumulation path to reach a chosen height field,
    which adds a second thing under test without changing what the facade sees.
    """

    mean_height_grid: np.ndarray
    coverage_map: CoverageMap


def coverage_map_with(bounds: GridBounds, counts: np.ndarray) -> CoverageMap:
    cmap = CoverageMap(bounds)
    cmap._count = counts.astype(np.int32)  # noqa: SLF001
    return cmap


def occupancy_with(bounds: GridBounds, log_odds: np.ndarray) -> OccupancyGrid:
    grid = OccupancyGrid(bounds)
    grid._log_odds = log_odds.astype(float)  # noqa: SLF001
    return grid


def occupancy_readback(bounds: GridBounds, grid: OccupancyGrid) -> list[str]:
    """The probabilities Python actually produces, cell by cell, i-major."""
    out = []
    for i in range(bounds.nx):
        for j in range(bounds.ny):
            x, y = bounds.ij_to_xy(i, j)
            out.append(exact(grid.occupancy_at(x, y)))
    return out


def corridor_with_alphas(query, start, end, samples, allow_unknown, spelling: str) -> bool:
    """`safe_corridor_between` with one of the two wrong alpha spellings.

    `naive` uses `k/(n-1)`; `unsnapped` keeps `k * (1/(n-1))` but drops the
    forced `1.0` at the end. Both are recorded per probe so the guard can
    require each to return a DIFFERENT ANSWER somewhere.

    Recorded per probe so the guard can require that the wrong spelling
    actually returns a DIFFERENT ANSWER somewhere, rather than merely that a
    sample count exists where the two alpha sequences differ in the last bit.
    An earlier version of the guard checked only the latter and let a port with
    the naive spelling through.
    """
    n = max(2, samples)
    step = 1.0 / (n - 1)
    sx, sy = start
    ex, ey = end
    for k in range(n):
        alpha = k / (n - 1) if spelling == "naive" else k * step
        x_m = sx + (ex - sx) * alpha
        y_m = sy + (ey - sy) * alpha
        if not query.is_inside_geofence(x_m, y_m):
            return False
        if allow_unknown and not query.is_observed(x_m, y_m):
            continue
        if not query.is_known_safe(x_m, y_m):
            return False
    return True


def cell_record(cell) -> dict:
    return {
        "cell_id": cell.cell_id,
        "ij": [int(cell.ij[0]), int(cell.ij[1])],
        "center_xy_m": [exact(cell.center_xy_m[0]), exact(cell.center_xy_m[1])],
        "height_estimate_m": opt_exact(cell.height_estimate_m),
        "height_uncertainty_m": opt_exact(cell.height_uncertainty_m),
        "obstacle_probability": exact(cell.obstacle_probability),
        "terrain_confidence": exact(cell.terrain_confidence),
        "coverage_count": int(cell.coverage_count),
        "last_observed_s": opt_exact(cell.last_observed_s),
        "inside_geofence": bool(cell.inside_geofence),
        "status": cell.status,
        "source_ids": list(cell.source_ids),
    }


# ----------------------------------------------------------------------
# Geofences and semantic labellers, named so the Rust side can rebuild them
# ----------------------------------------------------------------------

GEOFENCES = {
    "none": None,
    # A half-plane that cuts the grid, so some cells are excluded by the
    # predicate rather than by the bounds.
    "x_below_5": lambda x, y: x < 5.0,
    # A predicate that excludes nothing, to separate "no geofence" from "a
    # geofence that always passes" -- the reference treats them the same and a
    # port could easily not.
    "always": lambda x, y: True,
    "never": lambda x, y: False,
}

SEMANTICS = {
    "none": None,
    "band": lambda x, y: "scrub" if y < 3.0 else None,
    # An empty label is falsy and contributes no source id.
    "empty": lambda x, y: "",
}


def scenarios() -> list[dict]:
    out = []

    def add(
        name: str,
        bounds: GridBounds,
        counts: np.ndarray,
        *,
        heights: np.ndarray | None = None,
        log_odds: np.ndarray | None = None,
        last_observed: np.ndarray | None = None,
        geofence: str = "none",
        semantics: str = "none",
        origins: list[tuple[float, float]] | None = None,
        max_candidates: int = 10,
        corridors: list[tuple] | None = None,
        probes: list[tuple[float, float]] | None = None,
    ) -> None:
        cmap = coverage_map_with(bounds, counts)
        occupancy = None if log_odds is None else occupancy_with(bounds, log_odds)
        world_map = None if heights is None else WorldMapStub(heights.astype(float), cmap)

        query = WorldBeliefQuery(
            world_map,
            coverage_map=cmap,
            occupancy_grid=occupancy,
            geofence=GEOFENCES[geofence],
            last_observed_s_grid=last_observed,
            semantic_label_at=SEMANTICS[semantics],
        )

        rankings = []
        for origin in origins or [(0.0, 0.0)]:
            ranked = query.score_frontiers(origin, max_candidates=max_candidates)
            rankings.append(
                {
                    "origin_xy_m": [exact(origin[0]), exact(origin[1])],
                    "max_candidates": max_candidates,
                    "candidates": [
                        {
                            "ij": [int(c.cell.ij[0]), int(c.cell.ij[1])],
                            "score": exact(c.score),
                            "expected_coverage_gain": exact(c.expected_coverage_gain),
                            "uncertainty_reduction": exact(c.uncertainty_reduction),
                            "travel_cost_m": exact(c.travel_cost_m),
                            "risk_cost": exact(c.risk_cost),
                        }
                        for c in ranked
                    ],
                }
            )

        corridor_records = []
        for start, end, samples, allow_unknown in corridors or []:
            corridor_records.append(
                {
                    "start_xy_m": [exact(start[0]), exact(start[1])],
                    "end_xy_m": [exact(end[0]), exact(end[1])],
                    "sample_count": samples,
                    "allow_unknown": allow_unknown,
                    "safe": bool(
                        query.safe_corridor_between(
                            start, end, sample_count=samples, allow_unknown=allow_unknown
                        )
                    ),
                    "safe_under_naive_alphas": bool(
                        corridor_with_alphas(query, start, end, samples, allow_unknown, "naive")
                    ),
                    "safe_under_unsnapped_alphas": bool(
                        corridor_with_alphas(query, start, end, samples, allow_unknown, "unsnapped")
                    ),
                }
            )

        # Scalar probes at arbitrary points, including points OUTSIDE the raw
        # bounds. The cell table cannot reach these: a cell is always evaluated
        # at its own centre, which is never below x_min, so `xy_to_ij`'s lower
        # clamp and the inclusive bound tests are only observable from here.
        probe_records = []
        for px, py in probes or []:
            probe_records.append(
                {
                    "xy_m": [exact(px), exact(py)],
                    "coverage_at": int(query.coverage_at(px, py)),
                    "confidence_at": exact(query.confidence_at(px, py)),
                    "obstacle_probability_at": exact(query.obstacle_probability_at(px, py)),
                    "height_estimate_at": opt_exact(query.height_estimate_at(px, py)),
                    "height_uncertainty_at": opt_exact(query.height_uncertainty_at(px, py)),
                    "is_observed": bool(query.is_observed(px, py)),
                    "is_inside_geofence": bool(query.is_inside_geofence(px, py)),
                    "is_known_safe": bool(query.is_known_safe(px, py)),
                    "cell_at_ij": [int(v) for v in query.cell_at(px, py).ij],
                }
            )

        out.append(
            {
                "name": name,
                "bounds": {
                    "x_min_m": exact(bounds.x_min_m),
                    "x_max_m": exact(bounds.x_max_m),
                    "y_min_m": exact(bounds.y_min_m),
                    "y_max_m": exact(bounds.y_max_m),
                    "resolution_m": exact(bounds.resolution_m),
                },
                "dimensions": [bounds.nx, bounds.ny],
                "geofence": geofence,
                "semantics": semantics,
                "coverage": [int(v) for v in counts.ravel()],
                "mean_height": None if heights is None else [exact(v) for v in heights.ravel()],
                "occupancy": (None if occupancy is None else occupancy_readback(bounds, occupancy)),
                "last_observed_s": (
                    None if last_observed is None else [exact(v) for v in last_observed.ravel()]
                ),
                "cells": [
                    cell_record(query.cell_ij(i, j))
                    for i in range(bounds.nx)
                    for j in range(bounds.ny)
                ],
                "frontier_ij": [[int(c.ij[0]), int(c.ij[1])] for c in query.frontier_cells()],
                "expected_gain": [
                    exact(query._expected_coverage_gain((i, j)))  # noqa: SLF001
                    for i in range(bounds.nx)
                    for j in range(bounds.ny)
                ],
                "rankings": rankings,
                "corridors": corridor_records,
                "probes": probe_records,
            }
        )

    rng = np.random.default_rng(20260911)

    # 1. A plain aligned grid with a single covered cell: the frontier ring.
    counts = np.zeros((3, 3), dtype=np.int32)
    counts[1, 1] = 2
    add(
        "aligned_single_seed",
        GridBounds(0.0, 3.0, 0.0, 3.0, 1.0),
        counts,
        origins=[(1.5, 1.5), (0.0, 0.0)],
        corridors=[((0.5, 0.5), (2.5, 2.5), 16, False), ((0.5, 0.5), (2.5, 2.5), 16, True)],
    )

    # 2. A RAGGED grid: span 10 at resolution 3 gives 4 cells, and the last
    #    centre (10.5) falls past x_max. The cell table shows it reporting
    #    itself outside the geofence with zero coverage.
    add(
        "ragged_grid_last_cell_outside",
        GridBounds(0.0, 10.0, 0.0, 10.0, 3.0),
        np.full((4, 4), 5, dtype=np.int32),
        origins=[(0.0, 0.0)],
        corridors=[((1.5, 1.5), (10.5, 10.5), 16, False)],
    )

    # 3. Heights present, coverage absent: the two disagree out of bounds.
    heights = np.arange(16, dtype=float).reshape(4, 4) * 1.5
    heights[0, 0] = np.nan
    heights[3, 3] = np.inf
    add(
        "heights_including_non_finite",
        GridBounds(0.0, 4.0, 0.0, 4.0, 1.0),
        rng.integers(0, 4, size=(4, 4)).astype(np.int32),
        heights=heights,
        origins=[(2.0, 2.0)],
    )

    # 4. Occupancy across every status threshold. The log-odds are chosen so the
    #    read-back probabilities straddle 0.4 and 0.6 in both directions.
    log_odds = np.array(
        [
            [-4.0, -1.0, -0.405465108108164, 0.0],
            [0.405465108108164, 0.4054651081081646, 1.0, 4.0],
            [-0.5, 0.5, 0.85, -0.85],
            [0.0, 0.0, 2.0, -2.0],
        ]
    )
    add(
        "occupancy_across_thresholds",
        GridBounds(0.0, 4.0, 0.0, 4.0, 1.0),
        np.full((4, 4), 2, dtype=np.int32),
        log_odds=log_odds,
        origins=[(0.0, 0.0)],
        corridors=[((0.5, 0.5), (3.5, 3.5), 16, False)],
    )

    # 5. Geofence half-plane plus a semantic band, so source_ids populate and
    #    some cells are excluded by the predicate rather than by the bounds.
    add(
        "geofence_and_semantics",
        GridBounds(0.0, 8.0, 0.0, 8.0, 1.0),
        rng.integers(0, 3, size=(8, 8)).astype(np.int32),
        geofence="x_below_5",
        semantics="band",
        origins=[(0.0, 0.0), (7.5, 7.5)],
        max_candidates=5,
        corridors=[((0.5, 0.5), (7.5, 0.5), 16, True)],
    )

    # 6. A geofence that always passes, distinguished from having none.
    counts6 = np.zeros((3, 3), dtype=np.int32)
    counts6[0, 0] = 1
    add(
        "geofence_always_true",
        GridBounds(0.0, 3.0, 0.0, 3.0, 1.0),
        counts6,
        geofence="always",
        origins=[(0.0, 0.0)],
    )
    add(
        "geofence_never_true",
        GridBounds(0.0, 3.0, 0.0, 3.0, 1.0),
        counts6,
        geofence="never",
        origins=[(0.0, 0.0)],
    )

    # 7. An empty semantic label: falsy, so no source id.
    add(
        "empty_semantic_label",
        GridBounds(0.0, 2.0, 0.0, 2.0, 1.0),
        np.full((2, 2), 1, dtype=np.int32),
        semantics="empty",
        origins=[(0.0, 0.0)],
    )

    # 8. last_observed_s with non-finite entries, which read back as absent.
    last = np.array([[1.0, np.nan], [np.inf, 42.5]])
    add(
        "last_observed_with_non_finite",
        GridBounds(0.0, 2.0, 0.0, 2.0, 1.0),
        np.full((2, 2), 3, dtype=np.int32),
        last_observed=last,
        origins=[(0.0, 0.0)],
    )

    # 9. CORRIDOR SAMPLE COUNTS. 7 and 101 are the counts where numpy's
    #    linspace differs from k/(n-1); 16 is the default, where it does not.
    #    A hole in the middle of an otherwise-covered field makes the sample
    #    positions decide the answer.
    counts9 = np.full((12, 12), 4, dtype=np.int32)
    counts9[5:7, 5:7] = 0
    add(
        "corridor_sample_counts",
        GridBounds(0.0, 12.0, 0.0, 12.0, 1.0),
        counts9,
        origins=[(0.0, 0.0)],
        max_candidates=20,
        corridors=[
            ((0.5, 0.5), (11.5, 11.5), n, unknown)
            for n in (0, 1, 2, 3, 7, 16, 17, 101)
            for unknown in (False, True)
        ],
    )

    # 10. A fully covered field: no frontiers at all.
    add(
        "fully_covered_has_no_frontiers",
        GridBounds(0.0, 4.0, 0.0, 4.0, 1.0),
        np.full((4, 4), 9, dtype=np.int32),
        origins=[(0.0, 0.0)],
    )

    # 11. A fully uncovered field: no frontiers either, since a frontier must
    #     touch a covered cell.
    add(
        "fully_uncovered_has_no_frontiers",
        GridBounds(0.0, 4.0, 0.0, 4.0, 1.0),
        np.zeros((4, 4), dtype=np.int32),
        origins=[(0.0, 0.0)],
    )

    # 12. A single-cell grid, where every patch is the whole grid.
    add(
        "single_cell_grid",
        GridBounds(0.0, 1.0, 0.0, 1.0, 1.0),
        np.array([[0]], dtype=np.int32),
        origins=[(0.0, 0.0)],
        corridors=[((0.5, 0.5), (0.5, 0.5), 2, False)],
    )

    # 13. A non-square grid, so an i/j transposition is caught.
    counts13 = np.zeros((2, 7), dtype=np.int32)
    counts13[0, 3] = 1
    add(
        "non_square_grid",
        GridBounds(0.0, 2.0, 0.0, 7.0, 1.0),
        counts13,
        heights=np.arange(14, dtype=float).reshape(2, 7),
        origins=[(0.0, 0.0)],
    )

    # 14. Ties in the frontier ranking, resolved by (i, j) ascending.
    counts14 = np.zeros((5, 5), dtype=np.int32)
    counts14[2, 2] = 1
    add(
        "frontier_ties_break_by_ij",
        GridBounds(0.0, 5.0, 0.0, 5.0, 1.0),
        counts14,
        origins=[(2.5, 2.5)],
        max_candidates=8,
    )

    # 15. EDGE PROBES. A cell is always evaluated at its own centre, which is
    #     never below x_min and never outside the grid, so the cell table alone
    #     can never reach `xy_to_ij`'s lower clamp or either inclusive bound
    #     test. These probes do: points exactly ON each bound, just inside, just
    #     outside, and far below the origin.
    heights15 = np.arange(9, dtype=float).reshape(3, 3) + 100.0
    add(
        "edge_probes",
        GridBounds(0.0, 3.0, 0.0, 3.0, 1.0),
        np.full((3, 3), 2, dtype=np.int32),
        heights=heights15,
        origins=[(0.0, 0.0)],
        probes=[
            (0.0, 0.0),  # exactly on the lower bound: inside
            (3.0, 3.0),  # exactly on the upper bound: inside
            (np.nextafter(0.0, -np.inf), 1.5),  # one ulp below: outside
            (np.nextafter(3.0, np.inf), 1.5),  # one ulp above: outside
            (1.5, 0.0),
            (1.5, 3.0),
            (-40.0, -40.0),  # far below the origin: exercises the lower clamp
            (99.0, 99.0),  # far above: exercises the upper clamp
            (-0.5, 1.5),  # truncates toward zero to -0, clamps to 0
        ],
    )

    # 16. is_known_safe ACROSS ITS THRESHOLD, probed directly. It is strict
    #     `< 0.4` while cell status calls `>= 0.4` uncertain; only a probe at
    #     exactly 0.4 separates `<` from `<=`, and only one in [0.4, 0.6)
    #     separates 0.4 from 0.6.
    #     The log-odds are chosen so the read-back probabilities land on 0.4,
    #     just under it, and midway between the two thresholds.
    log_odds16 = np.array(
        [
            [-0.4054651081081646, -0.4054651081081645, 0.0],
            [0.2006706954621511, 0.4054651081081646, 1.0],
            # (2,2) is left UNOBSERVED below, and is given a very low obstacle
            # probability here on purpose: with p ~= 0.018 the obstacle test
            # passes, so the observation test is the only thing that can make
            # `is_known_safe` false there. Give it p >= 0.4 instead and dropping
            # the observation test entirely becomes invisible.
            [-4.0, 4.0, -4.0],
        ]
    )
    counts16 = np.full((3, 3), 2, dtype=np.int32)
    counts16[2, 2] = 0
    add(
        "known_safe_thresholds",
        GridBounds(0.0, 3.0, 0.0, 3.0, 1.0),
        counts16,
        log_odds=log_odds16,
        origins=[(0.0, 0.0)],
        probes=[(i + 0.5, j + 0.5) for i in range(3) for j in range(3)],
        corridors=[
            ((0.5, 0.5), (2.5, 2.5), 7, False),
            ((0.5, 0.5), (2.5, 2.5), 7, True),
            ((0.5, 1.5), (1.5, 1.5), 3, False),
        ],
    )

    # 17. THE LINSPACE CASE. Solved for, not stumbled on: at n = 7 the sample
    #     k = 5 along this segment lands in cell 7 under numpy's
    #     `k * (1/(n-1))` and in cell 8 under the naive `k/(n-1)`. Every other
    #     sample lands in 2, 3, 4, 6 or 9. Cell 7 is left uncovered and every
    #     other visited cell is covered, so the two spellings return DIFFERENT
    #     answers rather than merely different floats.
    counts17 = np.zeros((11, 2), dtype=np.int32)
    for i in (2, 3, 4, 6, 8, 9):
        counts17[i, :] = 4
    add(
        "linspace_sample_positions",
        GridBounds(0.0, 11.0, 0.0, 2.0, 1.0),
        counts17,
        origins=[(0.0, 0.0)],
        max_candidates=30,
        corridors=[
            ((2.999999999999999, 0.5), (9.0, 0.5), 7, False),
            # The same segment at the default 16 samples, where the two
            # spellings agree -- so the pair shows the divergence is about the
            # sample count, not about this geometry.
            ((2.999999999999999, 0.5), (9.0, 0.5), 16, False),
        ],
    )

    # 18. THE SAMPLE-COUNT FLOOR. With n forced to 2 both endpoints are
    #     sampled; with n left at 1 the single sample is the ENDPOINT (k == n-1
    #     snaps alpha to 1.0), so an unsafe start would go unnoticed. The start
    #     here is unobserved and the end is covered, which separates them.
    counts18 = np.full((4, 2), 3, dtype=np.int32)
    counts18[0, :] = 0
    add(
        "sample_count_floor",
        GridBounds(0.0, 4.0, 0.0, 2.0, 1.0),
        counts18,
        origins=[(0.0, 0.0)],
        corridors=[
            ((0.5, 0.5), (3.5, 0.5), 0, False),
            ((0.5, 0.5), (3.5, 0.5), 1, False),
            ((0.5, 0.5), (3.5, 0.5), 2, False),
        ],
    )

    # 19. THE ENDPOINT SNAP. numpy forces the final linspace value to exactly
    #     1.0; without that, n = 50 ends at 0.9999999999999999 because
    #     49 * (1/49) != 1.0. This segment ends exactly on the 7|8 cell
    #     boundary, so the snap decides whether the last sample reads cell 8 or
    #     cell 7 -- and only k = 49 differs. Cell 8 is left uncovered and 0..7
    #     are covered, so the two spellings return different answers.
    counts19 = np.zeros((10, 2), dtype=np.int32)
    counts19[0:8, :] = 4
    add(
        "endpoint_snap",
        GridBounds(0.0, 10.0, 0.0, 2.0, 1.0),
        counts19,
        origins=[(0.0, 0.0)],
        max_candidates=30,
        corridors=[
            ((0.5, 0.5), (8.0, 0.5), 50, False),
            # n = 49 instead, where 48 * (1/48) == 1.0 and the snap is a no-op.
            ((0.5, 0.5), (8.0, 0.5), 49, False),
        ],
    )

    return out


def check_discriminating(cases: list[dict]) -> None:
    """Refuse to write a fixture that cannot catch the known divergences.

    Each check recomputes the property the wrong way and requires a real
    disagreement, rather than asserting that the numbers look interesting.
    """
    # (a) Some cell must report itself outside the geofence with a NONZERO
    #     grid count -- the ragged-grid divergence.
    ragged = 0
    for case in cases:
        ny = case["dimensions"][1]
        for cell in case["cells"]:
            i, j = cell["ij"]
            if not cell["inside_geofence"] and case["coverage"][i * ny + j] > 0:
                ragged += 1
    if not ragged:
        raise SystemExit(
            "no cell is reported outside the geofence while the grid holds a "
            "count for it; the ragged-grid divergence is untested"
        )

    # (b) Some expected_gain entry must differ from the always-divide-by-9
    #     spelling -- the shrinking-patch divergence.
    patchy = 0
    for case in cases:
        nx, ny = case["dimensions"]
        for i in range(nx):
            for j in range(ny):
                zeros = sum(
                    1
                    for pi in range(max(0, i - 1), min(i + 2, nx))
                    for pj in range(max(0, j - 1), min(j + 2, ny))
                    if case["coverage"][pi * ny + pj] == 0
                )
                if float(case["expected_gain"][i * ny + j]) != zeros / 9.0:
                    patchy += 1
    if not patchy:
        raise SystemExit(
            "every expected_coverage_gain equals the divide-by-9 spelling; the "
            "shrinking-patch denominator is untested"
        )

    # (c) Some corridor must return a DIFFERENT ANSWER under the naive
    #     `k/(n-1)` spelling. Requiring only that a divergent sample count is
    #     present is the weaker property, and an earlier version of this guard
    #     checked exactly that -- it passed while a port with the naive
    #     spelling survived the whole mutation set.
    linspace_hits = {}
    for key in ("safe_under_naive_alphas", "safe_under_unsnapped_alphas"):
        linspace_hits[key] = [
            (case["name"], probe["sample_count"])
            for case in cases
            for probe in case["corridors"]
            if probe["safe"] != probe[key]
        ]
        if not linspace_hits[key]:
            raise SystemExit(
                f"no corridor returns a different answer under {key}; a "
                "divergent sample count alone is not enough -- the samples "
                "must land in cells that differ in safety"
            )

    # (d) Some probe must sit outside the raw bounds, which the cell table
    #     cannot reach: a cell is always evaluated at its own centre.
    outside = sum(
        1
        for case in cases
        for probe in case["probes"]
        if not probe["is_inside_geofence"] and case["geofence"] in ("none", "always")
    )
    if not outside:
        raise SystemExit(
            "no probe falls outside the raw bounds under a permissive "
            "geofence; the inclusive bound tests and the lower clamp are untested"
        )

    print(
        f"discriminating: {ragged} out-of-fence cells with counts, "
        f"{patchy} shrinking-patch gains, "
        + "".join(f"{len(v)} corridors flip under {k} {v}, " for k, v in linspace_hits.items())
        + (f"{outside} out-of-bounds probes")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = scenarios()
    check_discriminating(cases)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scenarios": cases}, indent=2) + "\n")
    total_cells = sum(len(c["cells"]) for c in cases)
    total_probes = sum(len(c["corridors"]) for c in cases)
    print(
        f"wrote {args.out}: {len(cases)} scenarios, {total_cells} cells, "
        f"{total_probes} corridor probes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
