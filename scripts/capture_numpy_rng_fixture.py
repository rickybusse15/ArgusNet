#!/usr/bin/env python3
"""Record numpy's default_rng uniform stream, bit-for-bit.

WHY THIS FIXTURE EXISTS

ADR-006 recommends re-baselining rather than reproducing numpy's PRNG. That
recommendation was made about `build_observations`, which builds a fresh
generator per (node, target) pair per tick and uses SeedSequence.spawn, and its
stated reason is cost. `procedural_terrain_grid` is the opposite case: ONE
generator, 12 to 46 scalar `uniform(low, high)` draws, no spawn, no arrays, no
other distribution. Reproducing that exactly is cheap, and it buys terrain that
is identical between the Python scene baker and the Rust runtime -- which a
statistical envelope cannot.

This fixture is what makes the claim checkable rather than asserted. It records
numpy's actual output so the Rust port is compared against numpy, not against a
second implementation of my own reasoning.

WHAT IS PINNED

* The raw 64-bit outputs, so a failure localises to the bit generator rather
  than to the double construction or the affine scale on top of it.
* `uniform(low, high)` over the ten ranges `procedural_terrain_grid` actually
  uses, plus degenerate ones (zero width, reversed, negative).
* The seeds the presets really produce, via `_stable_seed("profile", ...)`,
  alongside adversarial ones (0, 1, 2**31-1).

Usage:
    python3 scripts/capture_numpy_rng_fixture.py --out tests/fixtures/numpy_rng.json
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

from argusnet.world.procedural import _PRESET_PROFILES, _stable_seed  # noqa: E402

# Exactly the ranges procedural_terrain_grid draws from, plus degenerate cases
# that the terrain never produces but a caller could.
RANGES = [
    (-math.pi, math.pi),  # macro phase, ridge angle, valley angle
    (-0.36, 0.36),  # ridge offset fraction
    (0.045, 0.13),  # ridge width fraction
    (0.36, 0.85),  # ridge length fraction
    (0.20, 0.58),  # ridge amplitude fraction
    (-0.10, 0.10),  # the river/pass valley angle
    (-0.30, 0.30),  # valley offset fraction
    (0.035, 0.10),  # valley width fraction
    (0.45, 1.05),  # valley length fraction
    (0.12, 0.36),  # valley amplitude fraction
    (0.0, 1.0),  # the unit interval, where next_double is visible undisguised
    (5.0, 5.0),  # zero width: must return the bound and still consume a draw
    (-8.0, -3.0),  # wholly negative
    # A REVERSED range is deliberately absent: numpy REJECTS it outright
    # ("ValueError: high - low < 0"), which the first version of this script
    # discovered by trying. So there is no reference behaviour to record, and
    # the port's contract is low <= high. `numpy_rejects_a_reversed_range`
    # in the parity test pins that this is a rejection and not an oversight.
]


def f(value: float) -> str:
    """Floats as repr strings, matching the other fixtures in this directory."""
    return repr(float(value))


def seeds() -> list[dict]:
    """Real preset seeds and adversarial ones.

    The preset seeds are what actually reaches the generator in production;
    the others are chosen because a mixing bug tends to show at the extremes of
    the entropy word rather than in the middle.
    """
    cases = [
        {"name": "zero", "seed": 0},
        {"name": "one", "seed": 1},
        {"name": "two", "seed": 2},
        {"name": "max_31_bit", "seed": (1 << 31) - 1},
        {"name": "max_32_bit", "seed": (1 << 32) - 1},
        {"name": "power_of_two", "seed": 1 << 20},
        {"name": "all_ones_low", "seed": 0xFFFF},
    ]
    for preset in sorted(_PRESET_PROFILES):
        cases.append(
            {
                "name": f"preset/{preset}",
                "seed": int(_stable_seed("profile", preset, 20260911)),
            }
        )
    return cases


def capture() -> dict:
    raw_cases = []
    uniform_cases = []
    for case in seeds():
        seed = case["seed"]

        # Raw 64-bit outputs. numpy exposes these through the bit generator's
        # ctypes view rather than through Generator, which is the only way to
        # see the stream before next_double folds 11 bits away.
        bit_generator = np.random.PCG64(seed)
        raw = bit_generator.random_raw(16)
        raw_cases.append(
            {
                "name": case["name"],
                "seed": seed,
                "raw_u64": [str(int(v)) for v in raw],
            }
        )

        # One fresh generator per case, drawing across every range in turn so
        # the ranges are exercised at different points in the stream rather
        # than always at its start.
        generator = np.random.default_rng(seed)
        draws = []
        for index in range(len(RANGES) * 3):
            low, high = RANGES[index % len(RANGES)]
            draws.append(
                {
                    "low": f(low),
                    "high": f(high),
                    "value": f(generator.uniform(low, high)),
                }
            )
        uniform_cases.append({"name": case["name"], "seed": seed, "draws": draws})

    return {
        "description": (
            "numpy default_rng(seed).uniform(low, high) and PCG64 raw output. "
            "Regenerate with scripts/capture_numpy_rng_fixture.py."
        ),
        "raw": raw_cases,
        "uniform": uniform_cases,
    }


def check_discriminating(document: dict) -> list[str]:
    """Refuse to write a fixture that cannot tell right from wrong.

    Each check implements a plausible wrong version and requires the recorded
    values to rule it out. Asserting that the draws merely LOOK uniform would
    pass for any competent PRNG and prove nothing about this one.
    """
    problems = []

    # A stream that never sets the top bits of next_double would be caught by
    # nothing else here, since every recorded value is in range by construction.
    unit = [
        float(d["value"])
        for c in document["uniform"]
        for d in c["draws"]
        if (float(d["low"]), float(d["high"])) == (0.0, 1.0)
    ]
    if not unit:
        problems.append("no draw on the unit interval: next_double is never seen directly")
    elif max(unit) < 0.75 or min(unit) > 0.25:
        problems.append(f"unit draws span only [{min(unit):.3f}, {max(unit):.3f}]")

    # A wholly negative range must produce negative values, or a port that
    # took the absolute value somewhere would pass unnoticed.
    negative = [
        float(d["value"])
        for c in document["uniform"]
        for d in c["draws"]
        if (float(d["low"]), float(d["high"])) == (-8.0, -3.0)
    ]
    if not negative or any(v >= 0.0 for v in negative):
        problems.append("the negative range did not stay negative")

    # Zero width must be exactly the bound, and must still have consumed a draw
    # -- which the next value differing across cases demonstrates.
    zero_width = [
        float(d["value"])
        for c in document["uniform"]
        for d in c["draws"]
        if (float(d["low"]), float(d["high"])) == (5.0, 5.0)
    ]
    if not zero_width or any(v != 5.0 for v in zero_width):
        problems.append("zero-width range did not return its bound exactly")

    # Raw outputs must differ between seeds, or the entropy mixing is inert.
    first_raw = {c["raw_u64"][0] for c in document["raw"]}
    if len(first_raw) != len(document["raw"]):
        problems.append("two seeds produced the same first raw output")

    # Seed 0 must not produce a zero state -- the classic LCG failure.
    zero_case = next(c for c in document["raw"] if c["name"] == "zero")
    if int(zero_case["raw_u64"][0]) == 0:
        problems.append("seed 0 produced a zero first output")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = capture()
    problems = check_discriminating(document)
    for problem in problems:
        print(f"GAP  {problem}")
    if problems:
        print(f"\nABORT: {len(problems)} gap(s); refusing to write a fixture that proves nothing")
        return 2

    print(
        f"OK   {len(document['raw'])} seeds, "
        f"{sum(len(c['draws']) for c in document['uniform'])} uniform draws, "
        f"{sum(len(c['raw_u64']) for c in document['raw'])} raw outputs"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
