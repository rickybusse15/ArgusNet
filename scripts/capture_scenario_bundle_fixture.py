#!/usr/bin/env python3
"""Export a scenario bundle, and record the reference running the same scenario.

Two files:

* ``tests/fixtures/scenario_bundle_tt.json`` -- the bundle, exactly as
  ``export_bundle`` writes it, for the Rust assembly to run;
* ``tests/fixtures/scenario_bundle_tt_reference.json`` -- ``run_simulation``
  on the same built scenario: per tick, every node's position and velocity,
  every truth, the deconfliction events, the accepted-observation count and
  whether each search-then-follow drone had engaged.

WHAT CAN BE COMPARED EXACTLY, AND WHAT CANNOT

Stage 6 draws from keyed ChaCha streams (ADR-006), not NumPy's, so which
observations are accepted differs by design, and so does anything downstream
of an observation: engagement, fusion. Kinematics that no observation feeds
are compared bit for bit -- stations, launch, climb, weather drift, lanes,
truth-fed pursuit, deconfliction of those -- up to the first tick on which the
two runs disagree about an engagement. Engagement state is recorded per tick
(through the progress callback, which runs after each tick's
``update_from_frame``) so the Rust test can find that tick instead of
guessing it.

Usage:
    python3 scripts/capture_scenario_bundle_fixture.py --out-dir tests/fixtures
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.simulation.scenario_bundle import export_bundle, write_bundle  # noqa: E402
from argusnet.simulation.sim import (  # noqa: E402
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


def exact(value: float) -> str:
    return repr(float(value))


def vec(values) -> list[str]:
    return [exact(v) for v in np.asarray(values, dtype=float).ravel()]


SEED = 11
CONFIG = SimulationConfig(steps=80, dt_s=0.5, seed=SEED)
CASES = {
    # Mixed drone modes: one search-then-follow drone, whose engagement is
    # observation-fed, bounds the exact window.
    "tt": ScenarioOptions(
        map_preset="small",
        mission_mode="target_tracking",
        target_count=2,
        drone_count=3,
        ground_station_count=3,
    ),
    # Every drone truth-fed and four of them on one target: nothing an
    # observation decides can move anything, so the whole run -- including
    # every deconfliction resolution over real terrain -- must match.
    "tt_follow": ScenarioOptions(
        map_preset="small",
        mission_mode="target_tracking",
        drone_mode_preset="follow",
        target_count=1,
        drone_count=4,
        ground_station_count=3,
    ),
}


def capture_los(scenario) -> dict:
    """The reference's line of sight over this scenario's world.

    Sight lines from low and high origins across the map, so terrain blocks
    some, land cover varies along others, and the obstacles are crossed.
    Recorded at the default clearance (which the follow controller uses,
    because it passes none) and the sensing clearance. Every field is
    recorded, so land cover -- which only reaches the detection and noise
    multipliers -- is pinned too.
    """
    query = scenario.environment.query
    terrain = scenario.terrain
    bounds = scenario.map_bounds_m
    rng = np.random.default_rng(5)
    default_clearance = float(
        inspect.signature(query.los).parameters["terrain_clearance_m"].default
    )
    sensing_clearance = float(scenario.constants.sensor.los_terrain_clearance_m)

    def point(agl: float) -> np.ndarray:
        x = float(rng.uniform(bounds["x_min_m"], bounds["x_max_m"]))
        y = float(rng.uniform(bounds["y_min_m"], bounds["y_max_m"]))
        return np.array([x, y, float(terrain.height_at(x, y)) + agl])

    pairs = []
    for _ in range(150):
        pairs.append(
            (point(float(rng.choice([2.0, 10.0]))), point(float(rng.choice([2.0, 30.0, 150.0]))))
        )
    for primitive in scenario.environment.obstacles.primitives:
        b = primitive.bounds_xy_m()
        centre = np.array([(b.x_min_m + b.x_max_m) / 2, (b.y_min_m + b.y_max_m) / 2])
        for _ in range(10):
            a = point(5.0)
            far = centre + (centre - a[:2])
            z = float(terrain.height_at(far[0], far[1])) + 5.0
            pairs.append((a, np.array([far[0], far[1], z])))
    queries = []
    for a, b in pairs:
        for clearance in (default_clearance, sensing_clearance):
            r = query.los(a, b, terrain_clearance_m=clearance)
            queries.append(
                {
                    "from": vec(a),
                    "to": vec(b),
                    "clearance_m": exact(clearance),
                    "visible": bool(r.visible),
                    "blocker_type": str(r.blocker_type),
                    "transmittance": exact(r.transmittance),
                    "detection_multiplier": exact(r.detection_multiplier),
                    "noise_multiplier": exact(r.noise_multiplier),
                }
            )
    blocked = sum(1 for q in queries if not q["visible"])
    multipliers = {q["noise_multiplier"] for q in queries if q["visible"]}
    assert blocked > 10, "almost nothing was blocked; the LOS set tests little"
    assert len(multipliers) > 2, "land cover never varied the noise multiplier"
    return {"default_clearance_m": exact(default_clearance), "queries": queries}


def capture(out_dir: Path, name: str, options: ScenarioOptions) -> None:
    scenario = build_default_scenario(options, seed=SEED)
    bundle = export_bundle(scenario, CONFIG)
    write_bundle(out_dir / f"scenario_bundle_{name}.json", bundle)

    adaptive = scenario.adaptive_drone_controllers
    engaged_log: list[dict[str, bool]] = []

    def progress(_step: int, _total: int) -> None:
        engaged_log.append({node_id: bool(c.engaged) for node_id, c in adaptive.items()})

    result = run_simulation(scenario, CONFIG, progress=progress)
    frames = []
    for index, frame in enumerate(result.frames):
        frames.append(
            {
                "timestamp_s": exact(frame.timestamp_s),
                "nodes": {
                    n.node_id: {
                        "position": vec(n.position),
                        "velocity": vec(n.velocity),
                        "health": exact(n.health),
                        "battery_fraction": exact(n.battery_fraction),
                    }
                    for n in frame.nodes
                },
                "truths": {
                    t.target_id: {"position": vec(t.position), "velocity": vec(t.velocity)}
                    for t in frame.truths
                },
                "accepted_observations": len(frame.observations),
                "tracks": len(frame.tracks),
                "engaged": engaged_log[index] if index < len(engaged_log) else {},
                # Both parties to every resolution: either one's position is
                # then a function of the other's.
                "deconfliction_drones": sorted(
                    {e.yielding_drone_id for e in frame.deconfliction_events}
                    | {e.conflicting_drone_id for e in frame.deconfliction_events}
                ),
            }
        )
    reference = {
        "_source": "scripts/capture_scenario_bundle_fixture.py",
        "los": capture_los(scenario),
        "seed": SEED,
        "steps": CONFIG.steps,
        "dt_s": exact(CONFIG.dt_s),
        "frames": frames,
    }
    assert len(frames) == CONFIG.steps, f"expected {CONFIG.steps} frames, got {len(frames)}"
    if name == "tt":
        assert adaptive, "the scenario has no search-then-follow drone; the handover is untested"
    else:
        assert not adaptive, "the all-follow case grew an observation-fed drone"
        assert any(f["deconfliction_drones"] for f in frames), "no deconfliction to compare"
    (out_dir / f"scenario_bundle_{name}_reference.json").write_text(
        json.dumps(reference, indent=1, sort_keys=True) + "\n"
    )
    engaged_ticks = [i for i, e in enumerate(engaged_log) if any(e.values())]
    print(
        f"{name}: wrote bundle ({len(bundle['nodes'])} nodes, "
        f"{len(bundle['targets'])} targets) and reference ({len(frames)} frames); "
        f"first engagement at tick {engaged_ticks[0] if engaged_ticks else 'never'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, options in CASES.items():
        capture(args.out_dir, name, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
