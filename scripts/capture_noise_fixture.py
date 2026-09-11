#!/usr/bin/env python3
"""Record procedural.py's deterministic primitives bit-for-bit.

These sit at the bottom of the terrain dependency graph -- height shaping,
hydrology masks and land cover all run through them -- so the fixture compares
raw bit patterns, never a tolerance.

Three cases are chosen for what they catch rather than for coverage:

* `_stable_seed` here is **blake2b-64, little-endian, masked to 31 bits**, and
  is a DIFFERENT FUNCTION from `sim.py::_stable_seed`, which is an FNV-1 walk
  over characters and already has a Rust port. They share a name. The fixture
  records both so a port that reaches for the existing one fails loudly instead
  of producing plausible, deterministic, wrong terrain.

* `_hash_noise` multiplies int64 values by 0x45D9F3B twice and the products
  exceed int64 range for ordinary grid coordinates. numpy wraps silently; a
  Rust `*` panics in a debug build. The coordinates below are chosen to
  overflow. (The SHIFTS, by contrast, are provably immaterial: swapping
  arithmetic for logical cannot change the result, because only bits 51+
  differ and the final & 0xFFFF keeps bits that depend on operand bits 0..28.
  Checked over 300,000 inputs, 149,893 with a negative h: zero disagreements.
  An earlier version of this note claimed the opposite.)

* `np.arange` is NOT `start + i*step`. numpy's DOUBLE_fill writes the first two
  elements, recomputes `delta = buf[1] - buf[0]` -- which generally differs
  from `step` -- and fills the rest with `start + i*delta`. The guard at the
  bottom refuses to write a fixture whose arange cases cannot tell that apart
  from the two plausible wrong spellings.

Usage:
    python3 scripts/capture_noise_fixture.py --out tests/fixtures/noise.json
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

from argusnet.world.procedural import (  # noqa: E402
    _fbm,
    _grid_values,
    _hash_noise,
    _line_gaussian,
    _smoothstep,
    _stable_seed,
    _value_noise,
)


def exact(value: float) -> str:
    return repr(float(value))


class Bounds:
    """The two attributes `_grid_values` reads off a Bounds2D."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.x_min_m = x_min
        self.x_max_m = x_max
        self.y_min_m = y_min
        self.y_max_m = y_max


SEED_CASES: list[list[str]] = [
    [],
    [""],
    ["argusnet"],
    ["argusnet", "7"],
    ["7", "argusnet"],
    ["a", "b"],
    ["b", "a"],
    # The join is on "|", so these two payloads differ even though the
    # concatenation of their parts does not.
    ["a|b"],
    ["a", "b", ""],
    ["terrain", "regional", "1234"],
    # Non-ASCII: the payload is UTF-8 encoded, so a port hashing code points
    # rather than bytes diverges here and nowhere else.
    ["fjörd", "42"],
    ["日本", "0"],
    ["x" * 200, "long"],
]

HASH_CASES: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 1, 1),
    # Ordinary grid coordinates that still overflow the multiply chain.
    (1000, 2000, 123456789),
    (12345, 54321, 123456789),
    (-7, 13, 123456789),
    (-99999, -88888, 5),
    (32767, 32768, 0),
    (2147483647, -2147483648, 987654321),
    # Seeds at the edge of the 31-bit mask _stable_seed produces.
    (5, 5, 0x7FFFFFFF),
    (-1, -1, 0x7FFFFFFF),
]

VALUE_NOISE_CASES: list[tuple[float, float, int]] = [
    (0.0, 0.0, 7),
    (4.0, 9.0, 7),  # exactly on a lattice point
    (0.5, 0.5, 7),
    (3.0 - 1e-9, 1.25, 99),
    (-2.75, -8.125, 11),
    (1234.567, -8910.111, 2024),
    (1e6, -1e6, 3),
    (0.9999999999, 0.0000000001, 5),
    # Found by search on the FINAL output: the two lerp spellings --
    # `a + t*(b - a)` and `(1 - t)*a + t*b` -- give a different value_noise
    # here. An earlier version of these cases was searched on the INNER `nx0`
    # instead, and every one of them was absorbed by the outer lerp, so the
    # interpolation form stayed unpinned. The rate is about 2e-6 per random
    # point: these four came out of two million tries.
    (-42.900168911572564, -313.4428345493791, 40187),
    (-299.17603123890126, -549.8741868266475, 35016),
    (-681.209262138125, 93.60833183915292, 13159),
    (1964.4863062395307, -360.50173206593286, 13603),
]

FBM_CASES: list[dict] = [
    {"x": 0.0, "y": 0.0, "seed": 1, "wavelength": 500.0, "octaves": 4},
    {"x": 10.0, "y": 20.0, "seed": 5, "wavelength": 0.25, "octaves": 3},  # clamped
    {"x": 10.0, "y": 20.0, "seed": 5, "wavelength": 1.0, "octaves": 3},
    {"x": 3.0, "y": 4.0, "seed": 1, "wavelength": 500.0, "octaves": 0},  # floored
    {"x": 3.0, "y": 4.0, "seed": 1, "wavelength": 500.0, "octaves": 1},
    {"x": 12.0, "y": 34.0, "seed": 3, "wavelength": 100.0, "octaves": 1},
    {"x": -1234.5, "y": 6789.0, "seed": 77, "wavelength": 850.0, "octaves": 7},
    # Non-default persistence and lacunarity, so a port that hardcoded the
    # defaults is caught.
    {
        "x": 50.0,
        "y": -50.0,
        "seed": 9,
        "wavelength": 250.0,
        "octaves": 5,
        "persistence": 0.31,
        "lacunarity": 1.77,
    },
    {
        "x": 50.0,
        "y": -50.0,
        "seed": 9,
        "wavelength": 250.0,
        "octaves": 5,
        "persistence": 0.71,
        "lacunarity": 2.61,
    },
]

LINE_CASES: list[dict] = [
    {
        "xx": 0.0,
        "yy": 0.0,
        "angle": 0.0,
        "offset": 0.0,
        "width": 100.0,
        "length": 100.0,
        "cx": 0.0,
        "cy": 0.0,
    },
    # Two length units along: where the quartic and quadratic envelopes part.
    {
        "xx": 200.0,
        "yy": 0.0,
        "angle": 0.0,
        "offset": 0.0,
        "width": 100.0,
        "length": 100.0,
        "cx": 0.0,
        "cy": 0.0,
    },
    {
        "xx": 10.0,
        "yy": -20.0,
        "angle": math.pi / 4,
        "offset": 0.0,
        "width": 50.0,
        "length": 500.0,
        "cx": 10.0,
        "cy": -20.0,
    },
    {
        "xx": 137.0,
        "yy": 42.0,
        "angle": -2.4011,
        "offset": 25.0,
        "width": 80.0,
        "length": 600.0,
        "cx": -10.0,
        "cy": 5.0,
    },
    # Sub-metre width and length, both floored at 1.0.
    {
        "xx": 0.3,
        "yy": 0.2,
        "angle": 0.0,
        "offset": 0.0,
        "width": 0.01,
        "length": 0.01,
        "cx": 0.0,
        "cy": 0.0,
    },
    {
        "xx": 0.3,
        "yy": 0.2,
        "angle": 0.0,
        "offset": 0.0,
        "width": 1.0,
        "length": 1.0,
        "cx": 0.0,
        "cy": 0.0,
    },
]

GRID_CASES: list[tuple[float, float, float, float, float]] = [
    # (x_min, x_max, y_min, y_max, resolution)
    (0.0, 100.0, 0.0, 100.0, 10.0),
    (0.1, 5.1, 0.0, 1.0, 0.3),  # the recomputed-delta case from the docs
    (-262.0353729081086, 737.9646270918914, 0.0, 50.0, 18.52926056907656),
    (0.0, 0.0, 0.0, 1000.0, 50.0),  # degenerate x span
    (-500.0, 500.0, -500.0, 500.0, 7.0),
    (0.0, 1.0, 0.0, 1.0, 10.0),  # resolution wider than the span
    (1000.0, 4300.0, 2000.0, 6100.0, 12.5),
]


def smoothstep_cases() -> list[dict]:
    values = [0.0, 0.5, 1.0, 0.25, 0.75, 1e-9, 0.9999999999, -0.1, 1.1, 0.3333333333333333]
    return [{"value": exact(v), "result": exact(float(_smoothstep(np.float64(v))))} for v in values]


def build() -> dict:
    seeds = []
    for parts in SEED_CASES:
        seeds.append(
            {
                "parts": list(parts),
                "seed": str(_stable_seed(*parts)),
            }
        )

    hashes = []
    for ix, iy, seed in HASH_CASES:
        value = _hash_noise(np.array([ix], dtype=np.int64), np.array([iy], dtype=np.int64), seed)
        hashes.append({"ix": str(ix), "iy": str(iy), "seed": str(seed), "result": exact(value[0])})

    noises = []
    for x, y, seed in VALUE_NOISE_CASES:
        value = _value_noise(np.array([x]), np.array([y]), seed)
        noises.append({"x": exact(x), "y": exact(y), "seed": str(seed), "result": exact(value[0])})

    fbms = []
    for case in FBM_CASES:
        kwargs = {
            "seed": case["seed"],
            "base_wavelength_m": case["wavelength"],
            "octaves": case["octaves"],
        }
        if "persistence" in case:
            kwargs["persistence"] = case["persistence"]
        if "lacunarity" in case:
            kwargs["lacunarity"] = case["lacunarity"]
        value = _fbm(np.array([case["x"]]), np.array([case["y"]]), **kwargs)
        fbms.append(
            {
                "x": exact(case["x"]),
                "y": exact(case["y"]),
                "seed": str(case["seed"]),
                "base_wavelength_m": exact(case["wavelength"]),
                "octaves": str(case["octaves"]),
                "persistence": exact(case.get("persistence", 0.52)),
                "lacunarity": exact(case.get("lacunarity", 2.03)),
                "result": exact(value[0]),
            }
        )

    lines = []
    for case in LINE_CASES:
        value = _line_gaussian(
            np.array([case["xx"]]),
            np.array([case["yy"]]),
            angle_rad=case["angle"],
            offset_m=case["offset"],
            width_m=case["width"],
            length_m=case["length"],
            center_xy=(case["cx"], case["cy"]),
        )
        lines.append(
            {
                "xx": exact(case["xx"]),
                "yy": exact(case["yy"]),
                "angle_rad": exact(case["angle"]),
                "offset_m": exact(case["offset"]),
                "width_m": exact(case["width"]),
                "length_m": exact(case["length"]),
                "center_xy": [exact(case["cx"]), exact(case["cy"])],
                "result": exact(value[0]),
            }
        )

    grids = []
    for x_min, x_max, y_min, y_max, resolution in GRID_CASES:
        xs, ys = _grid_values(Bounds(x_min, x_max, y_min, y_max), resolution)
        grids.append(
            {
                "x_min_m": exact(x_min),
                "x_max_m": exact(x_max),
                "y_min_m": exact(y_min),
                "y_max_m": exact(y_max),
                "resolution_m": exact(resolution),
                "x_values": [exact(v) for v in xs],
                "y_values": [exact(v) for v in ys],
            }
        )

    return {
        "stable_seeds": seeds,
        "smoothsteps": smoothstep_cases(),
        "hash_noise": hashes,
        "value_noise": noises,
        "fbm": fbms,
        "line_gaussian": lines,
        "grid_values": grids,
    }


def check_discriminating(document: dict) -> None:
    """Refuse to write a fixture that cannot catch the three known divergences.

    Each check implements the WRONG version and requires it to disagree, rather
    than asserting that the numbers look interesting. Two earlier fixtures in
    this migration shipped guards that checked the weaker property and let a
    broken port through.
    """
    # (a) arange: the recomputed delta must matter somewhere.
    naive_mul = 0
    naive_acc = 0
    for case in document["grid_values"]:
        resolution = float(case["resolution_m"])
        for axis, lo_key in (("x_values", "x_min_m"), ("y_values", "y_min_m")):
            values = [float(v) for v in case[axis]]
            if len(values) < 3:
                continue
            lo = float(case[lo_key])
            if any(values[i] != lo + i * resolution for i in range(len(values))):
                naive_mul += 1
            acc = lo
            hit = False
            for v in values:
                if v != acc:
                    hit = True
                    break
                acc += resolution
            if hit:
                naive_acc += 1
    if not naive_mul or not naive_acc:
        raise SystemExit(
            f"grid axes do not discriminate the arange spellings "
            f"(start+i*step: {naive_mul}, cumulative add: {naive_acc}); "
            "both must disagree with numpy on at least one axis"
        )

    # (b) hash_noise: the multiply chain must actually overflow int64
    #     somewhere, or the wrapping requirement is untested.
    overflowing = 0
    for case in document["hash_noise"]:
        ix, iy, seed = int(case["ix"]), int(case["iy"]), int(case["seed"])
        h = (ix * 73856093) ^ (iy * 19349663) ^ seed
        step1 = ((h >> 13) ^ h) * 0x45D9F3B
        if not (-(2**63) <= step1 < 2**63):
            overflowing += 1
    if not overflowing:
        raise SystemExit(
            "no hash_noise case overflows int64; the wrapping multiply is "
            "untested and a non-wrapping port would pass"
        )

    # (c) the two _stable_seed functions must be distinguishable: at least one
    #     case where blake2b and the FNV-1 walk disagree. (They disagree on
    #     essentially everything, but assert it rather than assume it.)
    def fnv1(parts: list[str]) -> int:
        seed = 0
        for part in parts:
            for character in str(part):
                seed = ((seed * 16777619) ^ ord(character)) & 0xFFFFFFFF
        return seed

    if not any(int(case["seed"]) != fnv1(case["parts"]) for case in document["stable_seeds"]):
        raise SystemExit(
            "every stable_seed case agrees with the FNV-1 walk; the two "
            "same-named functions are not being distinguished"
        )

    # (d) value_noise: the two lerp spellings must disagree somewhere, or the
    #     interpolation form is unpinned. They agree almost everywhere.
    def lerp_alt_differs(case: dict) -> bool:
        """Does the wrong spelling change the FINAL value_noise?

        Measuring the inner `nx0` instead is the trap: the outer lerp absorbs
        a last-bit difference there almost every time, so a fixture built that
        way pins nothing. This recomputes the whole function both ways.
        """
        x, y, seed = float(case["x"]), float(case["y"]), int(case["seed"])
        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx = float(_smoothstep(np.float64(x - ix)))
        fy = float(_smoothstep(np.float64(y - iy)))

        def h(dx: int, dy: int) -> float:
            return float(
                _hash_noise(np.array([ix + dx], np.int64), np.array([iy + dy], np.int64), seed)[0]
            )

        n00, n10, n01, n11 = h(0, 0), h(1, 0), h(0, 1), h(1, 1)
        nx1 = n01 + fx * (n11 - n01)
        reference = (n00 + fx * (n10 - n00)) + fy * (nx1 - (n00 + fx * (n10 - n00)))
        alternate_nx0 = (1.0 - fx) * n00 + fx * n10
        alternate = alternate_nx0 + fy * (nx1 - alternate_nx0)
        return reference != alternate

    lerp_hits = sum(1 for case in document["value_noise"] if lerp_alt_differs(case))
    if not lerp_hits:
        raise SystemExit(
            "no value_noise case distinguishes `a + t*(b-a)` from "
            "`(1-t)*a + t*b`; the interpolation form is unpinned"
        )

    print(
        f"discriminating: arange (start+i*step {naive_mul} axes, "
        f"cumulative {naive_acc} axes), {overflowing} overflowing hash cases, "
        f"{lerp_hits} lerp-discriminating noise cases"
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
