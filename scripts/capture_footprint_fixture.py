#!/usr/bin/env python3
"""Record both of Python's circular-footprint rasterizations, cell for cell.

P4.6 asks for *"one footprint rasterizer whose indices/weights feed both coverage
and belief updates"*. There are currently two, and they disagree -- see
`docs/migration/FOOTPRINT_RASTERIZATION.md`. This records what each one actually
produces so the Rust port can be compared exactly rather than approximately, and
so the eventual choice between them is made against measured behaviour.

Hits are recorded **with multiplicity and in accumulation order**, because
`np.add.at` increments a cell once per hit: a fixture that deduplicated them
would pass against a rasterizer that silently lost revisit counts.

Cases cover the ratio `radius / resolution` on both sides of an integer, the
sub-cell phase of the centre, centres outside the grid, and the degenerate
radii -- the parameters that separate the two rasterizations.

Usage:
    python3 scripts/capture_footprint_fixture.py --out tests/fixtures/footprint_cells.json
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


def exact(value: float) -> str:
    """Render a float as a string so it survives the trip into Rust.

    `serde_json`'s float parser is off by one ULP on some shortest-round-trip
    literals, which would turn a sub-cell phase into a different cell at a
    boundary. Rust's `str::parse::<f64>` is exact, so every float crosses as a
    string. This bit the localization fixture already; not repeating it here.
    """
    return repr(float(value))


def world_anchored_hits(bounds: GridBounds, center, radius_m: float):
    """Replay `CoverageMap.mark_circular`'s lattice, keeping every hit.

    Rebuilt rather than read off `count_grid`, because the grid only records
    totals -- the order and multiplicity that `np.add.at` consumes would be
    lost. Verified against a real `CoverageMap` below.
    """
    cx, cy = center
    res = bounds.resolution_m
    xs = np.arange(cx - radius_m, cx + radius_m + res, res)
    ys = np.arange(cy - radius_m, cy + radius_m + res, res)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    inside = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius_m**2
    mark_x = grid_x[inside]
    mark_y = grid_y[inside]
    keep = (
        (mark_x >= bounds.x_min_m)
        & (mark_x <= bounds.x_max_m)
        & (mark_y >= bounds.y_min_m)
        & (mark_y <= bounds.y_max_m)
    )
    mark_x = mark_x[keep]
    mark_y = mark_y[keep]
    nx, ny = bounds.nx, bounds.ny
    i = ((mark_x - bounds.x_min_m) / res).astype(np.int64)
    j = ((mark_y - bounds.y_min_m) / res).astype(np.int64)
    np.clip(i, 0, nx - 1, out=i)
    np.clip(j, 0, ny - 1, out=j)
    return [[int(a), int(b)] for a, b in zip(i, j, strict=True)]


def cell_disc_hits(bounds: GridBounds, center, radius_m: float):
    """Replay `WorldMap.add_scan_observation`'s index-space disc."""
    cx, cy = center
    res = bounds.resolution_m
    nx, ny = bounds.nx, bounds.ny
    r_cells = max(1, int(radius_m / res))
    ci = int((cx - bounds.x_min_m) / res)
    cj = int((cy - bounds.y_min_m) / res)
    i_range = np.arange(max(0, ci - r_cells), min(nx, ci + r_cells + 1))
    j_range = np.arange(max(0, cj - r_cells), min(ny, cj + r_cells + 1))
    if i_range.size == 0 or j_range.size == 0:
        return []
    gi, gj = np.meshgrid(i_range, j_range, indexing="ij")
    mask = (gi - ci) ** 2 + (gj - cj) ** 2 <= r_cells**2
    return [[int(a), int(b)] for a, b in zip(gi[mask], gj[mask], strict=True)]


def check_against_real_coverage_map(bounds: GridBounds, center, radius_m: float, hits) -> None:
    """Prove the replayed lattice equals what `CoverageMap` itself accumulates.

    The replay above duplicates `mark_circular`'s body to recover per-hit
    ordering. That duplication is only trustworthy if it agrees with the real
    thing, so every case is checked against a live `CoverageMap`'s count grid --
    totals, not just membership, so a lost or doubled hit shows up.
    """
    coverage = CoverageMap(bounds)
    coverage.mark_circular(center_xy=(float(center[0]), float(center[1])), radius_m=radius_m)
    expected = coverage.count_grid
    actual = np.zeros_like(expected)
    for i, j in hits:
        actual[i, j] += 1
    if not np.array_equal(expected, actual):
        raise AssertionError(
            f"replayed lattice disagrees with CoverageMap at centre={center} radius={radius_m}"
        )


def capture() -> dict:
    cases: list[dict] = []

    def add(name: str, bounds: GridBounds, center, radius_m: float, note: str) -> None:
        world = world_anchored_hits(bounds, center, radius_m)
        check_against_real_coverage_map(bounds, center, radius_m, world)
        disc = cell_disc_hits(bounds, center, radius_m)
        cases.append(
            {
                "name": name,
                "note": note,
                "bounds": {
                    "x_min_m": exact(bounds.x_min_m),
                    "x_max_m": exact(bounds.x_max_m),
                    "y_min_m": exact(bounds.y_min_m),
                    "y_max_m": exact(bounds.y_max_m),
                    "resolution_m": exact(bounds.resolution_m),
                },
                "dimensions": [bounds.nx, bounds.ny],
                "center_xy": [exact(center[0]), exact(center[1])],
                "radius_m": exact(radius_m),
                "world_anchored": world,
                "cell_disc": disc,
            }
        )

    regional = GridBounds(0.0, 4300.0, 0.0, 4100.0, 50.0)
    fine = GridBounds(0.0, 2000.0, 0.0, 2000.0, 25.0)
    small = GridBounds(0.0, 240.0, 0.0, 240.0, 12.0)
    offset = GridBounds(-1000.0, 1000.0, -500.0, 1500.0, 40.0)

    # The shipping default: radius is an exact multiple of resolution, which is
    # the case where the two rasterizations very nearly agree.
    add(
        "regional_integer_ratio",
        regional,
        (1237.0, 981.0),
        350.0,
        "radius/resolution = 7 exactly; the configuration the sim normally runs",
    )
    add(
        "regional_grid_aligned",
        regional,
        (1025.0, 1025.0),
        350.0,
        "centre on a cell centre, so the lattice has no sub-cell phase",
    )
    # Non-integer ratios, where truncating the radius to whole cells bites.
    add(
        "fine_fractional_ratio",
        fine,
        (1013.7, 987.4),
        120.0,
        "radius/resolution = 4.8; cell disc truncates to 4 cells",
    )
    add(
        "fine_just_under_integer",
        fine,
        (1013.7, 987.4),
        124.9,
        "radius/resolution just below 5, the worst case for truncation",
    )
    add(
        "fine_just_over_integer",
        fine,
        (1013.7, 987.4),
        125.1,
        "radius/resolution just above 5, where truncation barely bites",
    )
    # The floor-binding configuration from the divergence measurement.
    add(
        "small_footprint_floor",
        small,
        (124.4, 118.9),
        50.0,
        "the 50 m footprint floor against a 12 m grid: ratio 4.17",
    )
    # Boundary and degenerate behaviour.
    add(
        "edge_clamped",
        regional,
        (60.0, 4060.0),
        350.0,
        "footprint overhanging two edges; exercises the inclusive bounds test and clamping",
    )
    add(
        "centre_outside_grid",
        regional,
        (-200.0, 2000.0),
        350.0,
        "centre left of the grid, so the cell-disc centre index goes negative",
    )
    add(
        "negative_origin",
        offset,
        (-137.5, 62.5),
        200.0,
        "non-zero, negative grid origin",
    )
    add(
        "radius_below_one_cell",
        regional,
        (1237.0, 981.0),
        10.0,
        "radius under one cell; cell disc floors to 1 cell, lattice keeps the true circle",
    )
    add(
        "radius_zero",
        regional,
        (1237.0, 981.0),
        0.0,
        "degenerate zero radius",
    )
    # The lattice lands exactly on x_max_m here, so the far edge's inclusive test
    # admits a sample whose truncated index is nx -- which the clamp folds onto
    # the last cell alongside its neighbour. That fold is a real double
    # increment, and it is the only case that pins both the inclusive test and
    # the clamp.
    add(
        "sample_exactly_on_far_edge",
        regional,
        (4000.0, 3800.0),
        350.0,
        "lattice sample lands exactly on x_max_m/y_max_m; index nx folds onto nx-1",
    )
    # A resolution that is not exactly representable in binary, so the lattice
    # separates numpy's `start + i*step` from a running sum.
    add(
        "inexact_resolution",
        GridBounds(0.0, 900.0, 0.0, 900.0, 7.3),
        (413.7, 388.1),
        90.0,
        "resolution 7.3 m is inexact in binary; ratio 12.33",
    )

    arange_cases = []
    for start, stop, step in [
        (-89.7, 97.3, 7.3),
        (323.7, 511.1, 7.3),
        (0.1, 1.0, 0.1),
        (887.0, 1637.0, 50.0),
        (-100.0, 125.0, 25.0),
        (0.3, 12.0, 0.7),
        (-1000.0, -700.0, 33.3),
    ]:
        values = np.arange(start, stop, step)
        arange_cases.append(
            {
                "start": exact(start),
                "stop": exact(stop),
                "step": exact(step),
                "values": [exact(v) for v in values],
            }
        )

    return {
        "arange_cases": arange_cases,
        "arange_note": (
            "numpy computes arange as start + i*step, not as a running sum; the "
            "two diverge once step is inexact in binary. The lattice is built "
            "from this, so the Rust port reproduces the multiply form. Recorded "
            "directly because the resulting ULP differences seldom move a sample "
            "across a cell boundary, and so would not be caught by cell parity "
            "alone."
        ),
        "source": (
            "src/argusnet/mapping/coverage.py::CoverageMap.mark_circular and "
            "src/argusnet/mapping/world_map.py::WorldMap.add_scan_observation"
        ),
        "note": (
            "Cell hits with multiplicity, in accumulation order, from each of "
            "Python's two circular-footprint rasterizations. The two disagree "
            "whenever radius/resolution is not an integer; see "
            "docs/migration/FOOTPRINT_RASTERIZATION.md. Floats cross as strings "
            "so the sub-cell phase survives parsing exactly."
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    diverging = sum(
        1
        for case in document["cases"]
        if sorted(map(tuple, case["world_anchored"])) != sorted(map(tuple, case["cell_disc"]))
    )
    print(
        f"wrote {args.out}: {len(document['cases'])} cases, "
        f"{diverging} of them where the two rasterizations disagree"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
