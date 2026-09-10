#!/usr/bin/env python3
"""Record BatteryModel's cost functions and BatteryState.consume.

Values come from calling the reference, not from reimplementing its formulas
here -- a generator that recomputed them would agree with a port that shared
its author's misreading while both disagreed with Python.

Inputs straddle every branch: both sides of the climb/descent power split,
the cruise- and climb-speed floors, headwind and tailwind, the groundspeed
floor, the static-reserve floor, and the consume clamp at empty. A coverage
check asserts the fixture actually reaches them, so a port that collapsed a
branch nothing exercises could not pass.

Usage:
    python3 scripts/capture_battery_fixture.py --out tests/fixtures/battery_costs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.planning.battery import BatteryModel, BatteryState  # noqa: E402


def exact(value: float) -> str:
    """Render a float as a string; `serde_json`'s parser is off by an ULP on
    some literals, which broke an exact comparison in the sensor error model."""
    return repr(float(value))


def model_dict(model: BatteryModel) -> dict:
    return {
        "capacity_wh": exact(model.capacity_wh),
        "hover_power_w": exact(model.hover_power_w),
        "travel_power_w": exact(model.travel_power_w),
        "climb_power_w": exact(model.climb_power_w),
        "cruise_speed_m_per_s": exact(model.cruise_speed_m_per_s),
        "climb_speed_m_per_s": exact(model.climb_speed_m_per_s),
        "reserve_fraction": exact(model.reserve_fraction),
    }


def _ground_speed(model: BatteryModel, pos, home, wind) -> float:
    """The groundspeed the reference resolves for this case.

    Reimplemented here only so the fixture can *report* which branch each case
    lands in -- the coverage checks read it to prove the floor and the wind
    guard are exercised. It is never compared against the port; the costs are.
    """
    delta_xy = home[:2] - pos[:2]
    distance_m = float(np.linalg.norm(delta_xy))
    if wind is None or distance_m <= 1e-6:
        return max(model.cruise_speed_m_per_s, 0.01)
    track = delta_xy / distance_m
    along = float(np.dot(np.asarray(wind, dtype=float)[:2], track))
    return max(model.cruise_speed_m_per_s + along, 0.25 * model.cruise_speed_m_per_s)


def capture() -> dict:
    default = BatteryModel()
    # A model whose speeds are below the 0.01 floors, so the floors bite.
    floored = BatteryModel(cruise_speed_m_per_s=0.0, climb_speed_m_per_s=0.0)
    # A tiny capacity, so the static reserve floor cannot dominate.
    small = BatteryModel(capacity_wh=1.0, reserve_fraction=0.01)

    travel = [
        {"model": model_dict(m), "distance_m": exact(d), "expected": exact(m.travel_cost_wh(d))}
        for m, d in [
            (default, 0.0),
            (default, 100.0),
            (default, 5000.0),
            (default, -50.0),  # negative distance is not guarded against
            (floored, 100.0),  # speed floor
        ]
    ]

    climb = [
        {"model": model_dict(m), "delta_z_m": exact(dz), "expected": exact(m.climb_cost_wh(dz))}
        for m, dz in [
            (default, 0.0),  # exactly zero takes the DESCENT branch
            (default, 50.0),  # climb
            (default, -50.0),  # descent, at hover_power * 0.8
            (default, 1e-9),  # just above zero: climb
            (default, -1e-9),  # just below: descent
            (floored, 30.0),  # climb-speed floor
        ]
    ]

    segment = [
        {
            "model": model_dict(default),
            "start": [exact(v) for v in s],
            "end": [exact(v) for v in e],
            "expected": exact(default.segment_cost_wh(np.array(s), np.array(e))),
        }
        for s, e in [
            ((0.0, 0.0, 100.0), (0.0, 0.0, 100.0)),  # no movement
            ((0.0, 0.0, 100.0), (300.0, 400.0, 100.0)),  # pure horizontal, 500 m
            ((0.0, 0.0, 100.0), (0.0, 0.0, 160.0)),  # pure climb
            ((0.0, 0.0, 160.0), (0.0, 0.0, 100.0)),  # pure descent
            ((10.0, -20.0, 50.0), (-130.0, 90.0, 175.0)),  # both
        ]
    ]

    winds = [
        None,
        (5.0, 0.0, 0.0),  # headwind (home lies in -x)
        (-5.0, 0.0, 0.0),  # tailwind
        (-40.0, 0.0, 0.0),  # strong tailwind
        (60.0, 0.0, 0.0),  # headwind past the groundspeed floor
        (0.0, 7.0, 0.0),  # pure crosswind
    ]
    return_home = []
    for m in (default, small):
        for pos, home in [
            ((1000.0, 0.0, 120.0), (0.0, 0.0, 20.0)),
            ((1000.0, 0.0, 20.0), (0.0, 0.0, 20.0)),  # level: cost is all travel
            ((0.0, 0.0, 120.0), (0.0, 0.0, 20.0)),  # no horizontal separation
            # A track shorter than the 1e-6 guard but not zero. Loosening the
            # guard to `> 0` would resolve wind onto this track and change the
            # cost; both guards reject the exactly-zero case above identically,
            # so that case alone cannot pin the threshold.
            ((5e-7, 0.0, 120.0), (0.0, 0.0, 20.0)),
            ((0.0, 500.0, 20.0), (0.0, 0.0, 120.0)),  # climbing home
        ]:
            for wind in winds:
                cost = m.return_home_cost_wh(
                    np.array(pos), np.array(home), None if wind is None else np.array(wind)
                )
                reserve = m.dynamic_reserve_wh(
                    np.array(pos), np.array(home), None if wind is None else np.array(wind)
                )
                return_home.append(
                    {
                        "model": model_dict(m),
                        "position": [exact(v) for v in pos],
                        "home": [exact(v) for v in home],
                        "wind": None if wind is None else [exact(v) for v in wind],
                        "expected_return_cost_wh": exact(cost),
                        "expected_dynamic_reserve_wh": exact(reserve),
                        "static_reserve_wh": exact(m.capacity_wh * m.reserve_fraction),
                        # Recorded so the Rust coverage checks can assert the
                        # fixture reaches the groundspeed floor and the wind
                        # guard, without recomputing either from the model.
                        "horizontal_distance_m": exact(
                            float(np.linalg.norm(np.array(home)[:2] - np.array(pos)[:2]))
                        ),
                        "ground_speed_m_per_s": exact(
                            _ground_speed(m, np.array(pos), np.array(home), wind)
                        ),
                    }
                )

    consume = []
    state = BatteryState(remaining_wh=10.0, timestamp_s=3.5)
    for cost in (0.0, 1.0, 4.0, 100.0, -2.0):
        after = state.consume(cost)
        consume.append(
            {
                "before_remaining_wh": exact(state.remaining_wh),
                "before_timestamp_s": exact(state.timestamp_s),
                "cost_wh": exact(cost),
                "after_remaining_wh": exact(after.remaining_wh),
                "after_timestamp_s": exact(after.timestamp_s),
            }
        )

    return {
        "source": "src/argusnet/planning/battery.py",
        "note": (
            "Cost-function values read back from the reference. Inputs straddle "
            "the climb/descent split, the speed floors, head/tailwind, the "
            "groundspeed floor, the static-reserve floor and the consume clamp. "
            "One return-home track is shorter than the 1e-6 wind guard without "
            "being zero, so the guard's threshold is observable."
        ),
        "default_safety_margin": exact(1.25),
        "travel": travel,
        "climb": climb,
        "segment": segment,
        "return_home": return_home,
        "consume": consume,
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
