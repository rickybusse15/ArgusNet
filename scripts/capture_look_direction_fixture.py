#!/usr/bin/env python3
"""Record the reference's sensor look direction and FOV angle at their edges.

`sensor_look_direction` and `fov_angle_deg` were lifted verbatim out of
`build_observations` (P4.8e.2) so they can be called directly. The cases sit
on the edges a port gets wrong: horizontal speed exactly at, just under and
just over 0.5 m/s; a pure vertical climb (no horizontal speed, so no look
direction, however fast it climbs); a climbing platform whose vertical
velocity is *replaced* by the tilt; tilts of 0, 30 and 89 degrees; and sight
lines ahead, abeam, behind and straight down.

Usage:
    python3 scripts/capture_look_direction_fixture.py --out tests/fixtures/look_direction.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixture_platform import captured_on  # noqa: E402

from argusnet.core.config import SimulationConstants  # noqa: E402
from argusnet.simulation.sim import fov_angle_deg, sensor_look_direction  # noqa: E402


def exact(value: float) -> str:
    return repr(float(value))


def vec(values) -> list[str]:
    return [exact(v) for v in np.asarray(values, dtype=float).ravel()]


VELOCITIES = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],  # exactly at the gate: no look direction (strict >)
    [0.3, 0.4, 0.0],  # |xy| = 0.5 by a 3-4-5 triangle, also exactly at it
    [0.5000001, 0.0, 0.0],
    [0.4999999, 0.0, 30.0],
    [0.0, 0.0, 25.0],  # pure climb: fast, but no horizontal heading
    [12.0, -5.0, 8.0],  # climbing: vz is replaced by the tilt
    [-3.7, 9.1, -2.2],
    [22.0, 0.001, 0.0],
]
TILTS = [0.0, 30.0, 89.0]
SIGHT_LINES = [
    [100.0, 0.0, 0.0],
    [0.0, 100.0, 0.0],
    [-100.0, 0.0, 0.0],
    [0.0, 0.0, -100.0],
    [37.5, -12.25, -60.0],
    [1.0e-3, 2.0e-3, -1.0e-3],
]


def capture() -> dict:
    directions = []
    angles = []
    for velocity in VELOCITIES:
        for tilt in TILTS:
            look = sensor_look_direction(np.array(velocity, dtype=float), tilt)
            directions.append(
                {
                    "velocity": vec(velocity),
                    "tilt_deg": exact(tilt),
                    "look": None if look is None else vec(look),
                }
            )
            if look is None:
                continue
            for los in SIGHT_LINES:
                los = np.array(los, dtype=float)
                range_m = float(np.linalg.norm(los))
                angles.append(
                    {
                        "line_of_sight": vec(los),
                        "range_m": exact(range_m),
                        "look": vec(look),
                        "angle_deg": exact(fov_angle_deg(los, range_m, look)),
                    }
                )
    none_count = sum(1 for d in directions if d["look"] is None)
    assert none_count >= 5 * len(TILTS), "the speed gate never closed"
    return {
        "_source": "scripts/capture_look_direction_fixture.py",
        "_captured_on": captured_on(),
        # The tilt every directional sensor gets by default.
        "reference_look_down_angle_deg": exact(
            SimulationConstants().sensor.drone_look_down_angle_deg
        ),
        "directions": directions,
        "angles": angles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {len(fixture['directions'])} directions, "
        f"{len(fixture['angles'])} angles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
