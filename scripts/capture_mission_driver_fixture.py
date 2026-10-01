#!/usr/bin/env python3
"""Record the reference primitives the P4.8e.3 mission driver leans on.

The full-run comparison (`capture_scenario_bundle_fixture.py`, `scan` case)
proves the driver end to end, but a single scenario leaves some branches
undecided: whether the rescore's density read ever changes which POI wins,
and which side of a boundary a last-bit rounding lands on. These are recorded
directly from the reference:

* ``coverage_in_region`` -- a real ``WorldMap``, marked with real footprints,
  queried at points across it, edges included, and at radii below one cell
  (the ``max(1, ...)`` floor) and above it.
* the 30 m dwell boundary -- a real ``POIManager`` crediting dwell to a drone
  whose offset from its POI has ``sqrt(x^2+y^2) == 30.0`` while
  ``hypot(x, y) > 30.0``. Which one the reference computes decides whether the
  dwell counts.
* ``np.linalg.norm`` on the same pair, which is the redirect's distance
  (``_make_redirect_goto``).

Usage:
    python3 scripts/capture_mission_driver_fixture.py --out tests/fixtures/mission_driver.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.core.types import InspectionPOI, vec3  # noqa: E402
from argusnet.mapping.occupancy import GridBounds  # noqa: E402
from argusnet.mapping.world_map import WorldMap  # noqa: E402
from argusnet.planning.poi import POIAssignmentContext, POIManager  # noqa: E402


def exact(value: float) -> str:
    return repr(float(value))


# An offset found by search: sqrt(x^2 + y^2) rounds to exactly 30.0, while
# hypot(x, y) rounds to 30.000000000000004.
BOUNDARY_OFFSET = (10.0007, 28.284023750343586)


def capture_coverage_in_region() -> dict:
    bounds = GridBounds(
        x_min_m=-500.0, x_max_m=700.0, y_min_m=-300.0, y_max_m=450.0, resolution_m=50.0
    )
    world_map = WorldMap(bounds)
    for position, radius in [
        ((-300.0, -100.0), 120.0),
        ((0.0, 0.0), 180.0),
        ((250.0, 300.0), 90.0),
        ((600.0, -250.0), 150.0),
        ((-480.0, 420.0), 60.0),
    ]:
        world_map.add_scan_observation(
            drone_position=np.array([position[0], position[1], 100.0]),
            terrain_height=0.0,
            footprint_radius_m=radius,
        )
    counts = np.asarray(world_map.coverage_map.count_grid, dtype=int)
    queries = []
    rng = np.random.default_rng(3)
    points = [(-500.0, -300.0), (699.9, 449.9), (0.0, 0.0), (-510.0, 0.0), (26.0, -24.0)]
    points += [
        (float(rng.uniform(-520.0, 720.0)), float(rng.uniform(-320.0, 470.0))) for _ in range(40)
    ]
    for x, y in points:
        for radius in (20.0, 50.0, 75.0, 140.0):
            queries.append(
                {
                    "center": [exact(x), exact(y)],
                    "radius_m": exact(radius),
                    "fraction": exact(world_map.coverage_in_region((x, y), radius_m=radius)),
                }
            )
    fractions = {q["fraction"] for q in queries}
    assert len(fractions) > 5, "coverage_in_region barely varied"
    return {
        "bounds": {
            "x_min_m": exact(bounds.x_min_m),
            "x_max_m": exact(bounds.x_max_m),
            "y_min_m": exact(bounds.y_min_m),
            "y_max_m": exact(bounds.y_max_m),
            "resolution_m": exact(bounds.resolution_m),
        },
        "counts": counts.tolist(),
        "queries": queries,
    }


def capture_dwell_boundary() -> dict:
    # The POI sits at the origin so the drone's position IS the offset:
    # subtracting positions of any larger magnitude would round the offset's
    # low bits away and the pair would no longer straddle 30 m.
    poi = InspectionPOI(
        poi_id="poi-00", name="Site A", position=vec3(0.0, 0.0, 40.0), required_dwell_s=15.0
    )
    manager = POIManager([poi])
    drone_at = np.array([BOUNDARY_OFFSET[0], BOUNDARY_OFFSET[1], 40.0])
    manager.assign_energy_aware(
        POIAssignmentContext(
            drone_id="drone-1",
            drone_pos=drone_at,
            battery_remaining_wh=500.0,
            battery_capacity_wh=500.0,
            timestamp_s=0.0,
        )
    )
    manager.accumulate_dwell("drone-1", drone_at, 20.0)
    completed = manager.check_completions(20.0)
    return {
        "poi_position": [exact(v) for v in poi.position],
        "drone_position": [exact(v) for v in drone_at],
        "dwell_s": exact(20.0),
        "completed": list(completed),
    }


def capture_norm() -> dict:
    offset = np.array(BOUNDARY_OFFSET, dtype=float)
    return {
        "offset": [exact(v) for v in offset],
        "norm": exact(float(np.linalg.norm(offset))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = {
        "_source": "scripts/capture_mission_driver_fixture.py",
        "coverage_in_region": capture_coverage_in_region(),
        "dwell_boundary": capture_dwell_boundary(),
        "norm": capture_norm(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {len(fixture['coverage_in_region']['queries'])} region queries; "
        f"dwell at the boundary completed={fixture['dwell_boundary']['completed']}; "
        f"norm={fixture['norm']['norm']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
