#!/usr/bin/env python3
"""Record procedural.py's grid operations bit-for-bit.

`_slope_grid`, `_dilate_mask`, `_smooth_heights` and `_ellipse_mask` sit
directly on the noise primitives and feed every mask the terrain builders use.
Whole grids are compared, not sampled: an edge-handling divergence shows up on
the border and nowhere else, so a fixture that checked interior cells would
pass a port that got the boundaries wrong.

Three things the cases are built to catch:

* `np.gradient` is central in the interior and ONE-SIDED at both edges. A port
  that clamped the index instead -- reading f[0] twice at the border -- gets
  half the correct derivative there and is right everywhere else. The guard
  below implements that wrong version and requires it to disagree.

* `_smooth_heights` pads by REPLICATION and its kernel is
  (4*centre + 4 neighbours) / 8: a weight sum of 8 over 5 cells, which is not
  an average. On a constant field the two facts cancel exactly, so a constant
  test case proves nothing; every smoothing case below uses a field with
  structure.

* `_ellipse_mask` includes its boundary (`<= 1.0`) and floors both radii at one
  metre. The cases place samples exactly on the boundary.

Usage:
    python3 scripts/capture_grid_ops_fixture.py --out tests/fixtures/grid_ops.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.world.procedural import (  # noqa: E402
    _dilate_mask,
    _ellipse_mask,
    _slope_grid,
    _smooth_heights,
)


def exact(value: float) -> str:
    return repr(float(value))


def flat(array: np.ndarray) -> list[str]:
    return [exact(v) for v in np.asarray(array, dtype=float).ravel()]


def flat_bool(array: np.ndarray) -> list[bool]:
    return [bool(v) for v in np.asarray(array, dtype=bool).ravel()]


def ramp(nx: int, ny: int, along_axis: int, step: float) -> np.ndarray:
    grid = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            grid[i, j] = (i if along_axis == 0 else j) * step
    return grid


def structured(nx: int, ny: int, seed: int) -> np.ndarray:
    """A field with real structure, so smoothing actually moves values."""
    rng = np.random.default_rng(seed)
    return rng.normal(50.0, 15.0, size=(nx, ny))


SLOPE_CASES: list[tuple[str, np.ndarray, float]] = [
    ("flat", np.full((4, 5), 12.5), 10.0),
    ("ramp_along_axis0", ramp(4, 3, 0, 1.0), 2.0),
    ("ramp_along_axis1", ramp(3, 6, 1, 2.5), 5.0),
    # Sub-metre resolution: clamped to 1.0, so this must equal the 1.0 case.
    ("sub_metre_resolution", ramp(4, 3, 0, 3.0), 0.1),
    ("unit_resolution", ramp(4, 3, 0, 3.0), 1.0),
    ("structured", structured(6, 7, 20260911), 12.5),
    ("minimum_shape", np.array([[0.0, 1.0], [4.0, 9.0]]), 1.0),
    ("negative_heights", structured(5, 5, 7) - 200.0, 3.0),
]

DILATE_CASES: list[tuple[str, np.ndarray, int]] = [
    ("single_seed_one_pass", None, 1),
    ("single_seed_two_passes", None, 2),
    ("single_seed_three_passes", None, 3),
    ("zero_iterations", None, 0),
    ("negative_iterations", None, -3),
]

SMOOTH_CASES: list[dict] = [
    {"name": "structured_one_pass", "passes": 1, "blend": 0.5, "preserve": "none"},
    {"name": "structured_four_passes", "passes": 4, "blend": 0.35, "preserve": "none"},
    {"name": "preserve_centre", "passes": 4, "blend": 0.9, "preserve": "centre"},
    {"name": "preserve_border", "passes": 3, "blend": 0.75, "preserve": "border"},
    {"name": "zero_blend", "passes": 5, "blend": 0.0, "preserve": "none"},
    {"name": "full_blend", "passes": 2, "blend": 1.0, "preserve": "none"},
    {"name": "zero_passes", "passes": 0, "blend": 1.0, "preserve": "none"},
    {"name": "negative_passes", "passes": -2, "blend": 1.0, "preserve": "none"},
    # A blend above one extrapolates rather than interpolating; the reference
    # does not guard it and the port must not either.
    {"name": "overshooting_blend", "passes": 2, "blend": 1.4, "preserve": "none"},
]

ELLIPSE_CASES: list[dict] = [
    {
        "name": "boundary_included",
        "xs": [0.0, 10.0, 20.0],
        "ys": [0.0],
        "cx": 10.0,
        "cy": 0.0,
        "rx": 10.0,
        "ry": 10.0,
    },
    # Both axes must VARY here. An earlier version held ys = [0.0], so dy was
    # always 0 whatever ry was, and the y-radius floor went untested -- the
    # mutant that dropped it survived.
    {
        "name": "sub_metre_radii_floored",
        "xs": [0.0, 0.5, 1.0],
        "ys": [0.0, 0.5, 1.0],
        "cx": 0.0,
        "cy": 0.0,
        "rx": 0.01,
        "ry": 0.01,
    },
    {
        "name": "unit_radii",
        "xs": [0.0, 0.5, 1.0],
        "ys": [0.0, 0.5, 1.0],
        "cx": 0.0,
        "cy": 0.0,
        "rx": 1.0,
        "ry": 1.0,
    },
    # A sub-metre radius on ONE axis only, so the two floors cannot be
    # confused for each other.
    {
        "name": "sub_metre_y_radius_only",
        "xs": [0.0, 2.0, 4.0],
        "ys": [0.0, 0.5, 1.0],
        "cx": 0.0,
        "cy": 0.0,
        "rx": 4.0,
        "ry": 0.02,
    },
    {
        "name": "anisotropic",
        "xs": [0.0, 30.0],
        "ys": [0.0, 30.0],
        "cx": 0.0,
        "cy": 0.0,
        "rx": 40.0,
        "ry": 10.0,
    },
    {
        "name": "offset_centre",
        "xs": [float(v) for v in range(-40, 41, 10)],
        "ys": [float(v) for v in range(-40, 41, 10)],
        "cx": 12.5,
        "cy": -7.5,
        "rx": 27.0,
        "ry": 18.0,
    },
]


def build() -> dict:
    slopes = []
    for name, heights, resolution in SLOPE_CASES:
        result = _slope_grid(heights, resolution)
        slopes.append(
            {
                "name": name,
                "shape": [int(heights.shape[0]), int(heights.shape[1])],
                "resolution_m": exact(resolution),
                "heights_m": flat(heights),
                "slope_rad": flat(result),
            }
        )

    dilations = []
    seed_mask = np.zeros((9, 9), dtype=bool)
    seed_mask[4, 4] = True
    corner_mask = np.zeros((3, 3), dtype=bool)
    corner_mask[0, 0] = True
    for name, _, iterations in DILATE_CASES:
        result = _dilate_mask(seed_mask, iterations)
        dilations.append(
            {
                "name": name,
                "shape": [9, 9],
                "iterations": iterations,
                "mask": flat_bool(seed_mask),
                "dilated": flat_bool(result),
            }
        )
    # A corner seed, which is where "padded with False" differs from any
    # wrap-around or replicate-the-edge reading.
    dilations.append(
        {
            "name": "corner_seed",
            "shape": [3, 3],
            "iterations": 1,
            "mask": flat_bool(corner_mask),
            "dilated": flat_bool(_dilate_mask(corner_mask, 1)),
        }
    )

    smooths = []
    base = structured(7, 6, 4242)
    for case in SMOOTH_CASES:
        preserve = np.zeros(base.shape, dtype=bool)
        if case["preserve"] == "centre":
            preserve[3, 3] = True
        elif case["preserve"] == "border":
            preserve[0, :] = True
            preserve[-1, :] = True
            preserve[:, 0] = True
            preserve[:, -1] = True
        result = _smooth_heights(
            base, preserve_mask=preserve, passes=case["passes"], blend=case["blend"]
        )
        smooths.append(
            {
                "name": case["name"],
                "shape": [int(base.shape[0]), int(base.shape[1])],
                "passes": case["passes"],
                "blend": exact(case["blend"]),
                "heights_m": flat(base),
                "preserve_mask": flat_bool(preserve),
                "smoothed_m": flat(result),
            }
        )

    ellipses = []
    for case in ELLIPSE_CASES:
        xs = np.array(case["xs"], dtype=float)
        ys = np.array(case["ys"], dtype=float)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        result = _ellipse_mask(
            xx,
            yy,
            center_xy=(case["cx"], case["cy"]),
            radius_x_m=case["rx"],
            radius_y_m=case["ry"],
        )
        ellipses.append(
            {
                "name": case["name"],
                "xs": [exact(v) for v in xs],
                "ys": [exact(v) for v in ys],
                "center_xy": [exact(case["cx"]), exact(case["cy"])],
                "radius_x_m": exact(case["rx"]),
                "radius_y_m": exact(case["ry"]),
                "mask": flat_bool(result),
            }
        )

    return {
        "slope": slopes,
        "dilate": dilations,
        "smooth": smooths,
        "ellipse": ellipses,
    }


def check_discriminating(document: dict) -> None:
    """Implement each wrong version and require it to disagree.

    Not "do the numbers look interesting" -- three earlier fixtures in this
    migration shipped guards that asked the weaker question and let a broken
    port through.
    """

    # (a) gradient edges: a port that clamped the index instead of taking a
    #     one-sided difference halves the derivative at the border.
    def clamped_gradient(values: np.ndarray, spacing: float) -> np.ndarray:
        nx, ny = values.shape
        d0 = np.zeros_like(values)
        d1 = np.zeros_like(values)
        for i in range(nx):
            for j in range(ny):
                lo = values[max(i - 1, 0), j]
                hi = values[min(i + 1, nx - 1), j]
                d0[i, j] = (hi - lo) / (2.0 * spacing)
                lo = values[i, max(j - 1, 0)]
                hi = values[i, min(j + 1, ny - 1)]
                d1[i, j] = (hi - lo) / (2.0 * spacing)
        return np.arctan(np.sqrt(d0 * d0 + d1 * d1))

    edge_hits = 0
    for case in document["slope"]:
        nx, ny = case["shape"]
        heights = np.array([float(v) for v in case["heights_m"]]).reshape(nx, ny)
        expected = np.array([float(v) for v in case["slope_rad"]]).reshape(nx, ny)
        wrong = clamped_gradient(heights, max(float(case["resolution_m"]), 1.0))
        if not np.array_equal(wrong, expected):
            edge_hits += 1
    if not edge_hits:
        raise SystemExit(
            "no slope case distinguishes numpy's one-sided edge difference "
            "from an index-clamping port; the boundary handling is untested"
        )

    # (b) smoothing: a port using a true 5-cell average (sum / 5) instead of
    #     the reference's /8 must disagree. A constant field cannot tell them
    #     apart, so this also proves the cases carry structure.
    def true_mean_smooth(
        heights: np.ndarray, preserve: np.ndarray, passes: int, blend: float
    ) -> np.ndarray:
        smoothed = heights.copy()
        for _ in range(max(passes, 0)):
            padded = np.pad(smoothed, 1, mode="edge")
            mean = (
                padded[1:-1, 1:-1]
                + padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            ) / 5.0
            candidate = (smoothed * (1.0 - blend)) + (mean * blend)
            smoothed = np.where(preserve, smoothed, candidate)
        return smoothed

    kernel_hits = 0
    for case in document["smooth"]:
        nx, ny = case["shape"]
        heights = np.array([float(v) for v in case["heights_m"]]).reshape(nx, ny)
        preserve = np.array(case["preserve_mask"]).reshape(nx, ny)
        expected = np.array([float(v) for v in case["smoothed_m"]]).reshape(nx, ny)
        wrong = true_mean_smooth(heights, preserve, case["passes"], float(case["blend"]))
        if not np.array_equal(wrong, expected):
            kernel_hits += 1
    if not kernel_hits:
        raise SystemExit(
            "no smoothing case distinguishes the /8 kernel from a true 5-cell "
            "mean; the weighting is untested"
        )

    # (c) ellipse: the boundary must be attained somewhere, or `<= 1.0` and
    #     `< 1.0` are indistinguishable.
    boundary_hits = 0
    for case in document["ellipse"]:
        rx = max(float(case["radius_x_m"]), 1.0)
        ry = max(float(case["radius_y_m"]), 1.0)
        cx, cy = float(case["center_xy"][0]), float(case["center_xy"][1])
        for x in case["xs"]:
            for y in case["ys"]:
                dx = (float(x) - cx) / rx
                dy = (float(y) - cy) / ry
                if (dx * dx) + (dy * dy) == 1.0:
                    boundary_hits += 1
    if not boundary_hits:
        raise SystemExit(
            "no ellipse sample lands exactly on the boundary; `<= 1.0` and "
            "`< 1.0` are indistinguishable"
        )

    # (d) both ellipse radius floors must change an answer. Holding one axis
    #     constant makes that axis's floor invisible, which is exactly how the
    #     y floor went untested in the first version of these cases.
    floors_exercised = {"rx": 0, "ry": 0}
    for case in document["ellipse"]:
        raw_rx, raw_ry = float(case["radius_x_m"]), float(case["radius_y_m"])
        cx, cy = float(case["center_xy"][0]), float(case["center_xy"][1])
        for axis, raw in (("rx", raw_rx), ("ry", raw_ry)):
            if raw >= 1.0:
                continue
            rx_ref, ry_ref = max(raw_rx, 1.0), max(raw_ry, 1.0)
            rx_bad = raw_rx if axis == "rx" else rx_ref
            ry_bad = raw_ry if axis == "ry" else ry_ref
            for index, (x, y) in enumerate(
                (float(x), float(y)) for x in case["xs"] for y in case["ys"]
            ):
                good = ((x - cx) / rx_ref) ** 2 + ((y - cy) / ry_ref) ** 2 <= 1.0
                bad = ((x - cx) / rx_bad) ** 2 + ((y - cy) / ry_bad) ** 2 <= 1.0
                assert good == case["mask"][index], "reference disagrees with itself"
                if good != bad:
                    floors_exercised[axis] += 1
    missing = [axis for axis, hits in floors_exercised.items() if hits == 0]
    if missing:
        raise SystemExit(
            "these ellipse radius floors change no answer: "
            + ", ".join(missing)
            + "; hold neither axis constant, or its floor is untested"
        )

    print(
        f"discriminating: {edge_hits} slope cases vs an index-clamping "
        f"gradient, {kernel_hits} smoothing cases vs a true 5-cell mean, "
        f"{boundary_hits} ellipse samples exactly on the boundary, "
        f"radius floors exercised {floors_exercised}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = build()
    check_discriminating(document)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    counts = ", ".join(f"{len(v)} {k}" for k, v in document.items())
    print(f"wrote {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
