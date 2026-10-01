#!/usr/bin/env python3
"""Record `build_target_trajectory` for every preset (P4.9b.2).

Each case builds a behaviour from `(preset, bounds, altitude, speed, seed)`
and samples its position and velocity at times chosen to cross every internal
boundary: transit segments and the post-transit loiter, search-pattern legs
and wrap-around, racetrack straights and turns, the `mixed` composite's
switches at 30 s and 90 s and their 2 s blends, and the evasive behaviour's
windows (with a synthetic, analytic terrain so the dips are exact and
terrain-independent).

THE UNFUSED UNIFORM

numpy's `Generator.uniform` computes `low + (high - low) * u`. Some builds
(arm64, numpy 2.5) contract that to a fused multiply-add and others do not, so
the "reference value" differs in the last place between machines. This capture
wraps `default_rng` so `uniform` is the unfused form on every platform --
`integers` and `random` go straight to numpy -- which pins the behaviour logic
bit for bit. The FMA effect is measured separately, in the builder's own
target-truth comparison.

Usage:
    python3 scripts/capture_behaviors_fixture.py --out tests/fixtures/behaviors.json
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

from argusnet.simulation import behaviors  # noqa: E402


class UnfusedGenerator:
    """`default_rng(seed)` with `uniform` computed unfused."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.Generator(np.random.PCG64(seed))

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(low) + (float(high) - float(low)) * self._generator.random()
        values = [
            float(low) + (float(high) - float(low)) * self._generator.random()
            for _ in range(int(np.prod(size)))
        ]
        return np.asarray(values, dtype=float).reshape(size)

    def integers(self, low, high=None, size=None):
        return self._generator.integers(low, high, size=size)

    def random(self, size=None):
        return self._generator.random(size)


def terrain(x: float, y: float) -> float:
    """An analytic terrain for the evasive dips: exact in both languages."""
    return 40.0 + 0.05 * x - 0.03 * y


BOUNDS = [
    {"x_min_m": -360.0, "x_max_m": 360.0, "y_min_m": -330.0, "y_max_m": 360.0},
    {"x_min_m": -2160.0, "x_max_m": 2160.0, "y_min_m": -1980.0, "y_max_m": 2160.0},
    {"x_min_m": 100.0, "x_max_m": 400.0, "y_min_m": -50.0, "y_max_m": 900.0},
]
TIMES = sorted(
    {round(t, 3) for t in np.arange(0.0, 240.0, 2.5)}
    | {29.0, 29.999, 30.0, 30.5, 31.0, 31.999, 32.0, 89.999, 90.0, 91.0, 92.0, 92.5}
    | {0.001, 600.0, 900.0, 1234.5}
)

BLEND_TIMES = [30.0 + k / 37.0 for k in range(74)] + [90.0 + k / 37.0 for k in range(74)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    behaviors.np.random.default_rng = UnfusedGenerator  # type: ignore[assignment]
    cases = []
    for preset in sorted(behaviors.BEHAVIOR_PRESETS):
        for index, bounds in enumerate(BOUNDS):
            for seed in (1, 42, 701002222 + index, 2**31 - 1):
                altitude_m = 120.0 + 17.5 * index
                speed_mps = [7.0, 18.0, 24.5][index]
                trajectory = behaviors.build_target_trajectory(
                    preset,
                    bounds,
                    altitude_m=altitude_m,
                    speed_mps=speed_mps,
                    seed=seed,
                    terrain_height_fn=terrain,
                )
                samples = []
                # The composite's blends run through `3 a**2 - 2 a**3` with C
                # `pow`; sweep both 2 s windows densely where they exist.
                times = TIMES + (BLEND_TIMES if preset == "mixed" else [])
                for t in times:
                    position, velocity = trajectory(float(t))
                    samples.append(
                        [float(t)] + [float(v) for v in position] + [float(v) for v in velocity]
                    )
                cases.append(
                    {
                        "preset": preset,
                        "bounds": bounds,
                        "altitude_m": altitude_m,
                        "speed_mps": speed_mps,
                        "seed": seed,
                        "samples": samples,
                    }
                )
    evasive_dips = sum(
        1
        for case in cases
        if case["preset"] == "evasive"
        for sample in case["samples"]
        if not math.isclose(sample[3], case["altitude_m"], abs_tol=15.5)
    )
    assert evasive_dips > 50, f"the evasive windows barely fired ({evasive_dips})"
    args.out.write_text(json.dumps({"_source": Path(__file__).name, "cases": cases}) + "\n")
    print(f"{len(cases)} cases, {len(TIMES)} times each, {evasive_dips} evasive dip samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
