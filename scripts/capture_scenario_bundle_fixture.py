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

THE REPLAY'S OPTIONAL PAYLOADS (P4.9a.2)

Each frame also records, under ``replay``, the reference's own serialisation
of the payloads the Rust writer now produces -- generation rejections,
mapping and localization state, inspection and deconfliction events, the scan
mission state and target metadata -- and the reference ``meta``'s run keys
under ``meta_run``. The Rust test decides per field what is exact and what is
draw-dependent; the capture only records.

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

from argusnet.simulation.scenario_bundle import (  # noqa: E402
    export_bundle,
    motion_weather_document,
    write_bundle,
)
from argusnet.simulation.sim import (  # noqa: E402
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    build_replay_document_from_result,
    cloud_detection_factor,
    run_simulation,
    scenario_replay_metadata,
)
from argusnet.world.weather import WEATHER_PRESETS  # noqa: E402


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
    # The default mission mode: lawnmower scan, localization, POI inspection,
    # egress. A lowered coverage threshold lets 400 one-second ticks reach
    # every phase, through `complete`.
    "scan": ScenarioOptions(
        map_preset="small",
        drone_count=2,
        ground_station_count=3,
        poi_count=3,
        scan_coverage_threshold=0.3,
    ),
    # Stage 4 (P4.9d). The all-follow fleet again, under the hard safety
    # floor: interceptor envelopes, and nothing an observation decides, so
    # every clamp and every event must match on every tick.
    "tt_safety": ScenarioOptions(
        map_preset="small",
        mission_mode="target_tracking",
        drone_mode_preset="follow",
        target_count=1,
        drone_count=4,
        ground_station_count=3,
        safety_blocking=True,
    ),
    # Scan drones take the tracker envelope, and with localization replayed
    # nothing an observation decides moves them either.
    "scan_safety": ScenarioOptions(
        map_preset="small",
        drone_count=2,
        ground_station_count=3,
        poi_count=3,
        scan_coverage_threshold=0.3,
        safety_blocking=True,
    ),
}
CONFIGS = {
    "scan": SimulationConfig(steps=400, dt_s=1.0, seed=5),
    "scan_safety": SimulationConfig(steps=400, dt_s=1.0, seed=5),
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
    # The world's detection model on the same sight lines, as stage 6 calls
    # it: the sensing clearance, the optical profile, and a weather -- this
    # scenario's, and two with cloud slabs and precipitation, so every factor
    # (path loss against two sensor ranges, weather visibility, weather noise,
    # cloud occlusion, the dominant loss) takes more than one value.
    weathers = {
        "scenario": scenario.weather,
        "fog": WEATHER_PRESETS["fog"],
        "heavy_rain": WEATHER_PRESETS["heavy_rain"],
    }
    # Sight lines that cross a whole cloud slab (from near the ground to
    # 2.5 km up), so the full-occlusion branch is reached; the map's own
    # sight lines never climb that high.
    tall_pairs = []
    for _ in range(6):
        low = point(3.0)
        high = point(0.0)
        high[2] = 2500.0
        tall_pairs.append((low, high))
    detections = []
    for weather_name, weather in weathers.items():
        for max_range_m in (1200.0, 400.0):
            for a, b in pairs[:60] + tall_pairs:
                d = query.compute_detection_probability(
                    a,
                    b,
                    weather=weather,
                    max_range_m=max_range_m,
                    terrain_clearance_m=sensing_clearance,
                )
                detections.append(
                    {
                        "weather": weather_name,
                        "from": vec(a),
                        "to": vec(b),
                        "max_range_m": exact(max_range_m),
                        "p_d": exact(d.p_d),
                        "effective_noise_multiplier": exact(d.effective_noise_multiplier),
                        "dominant_loss_factor": str(d.dominant_loss_factor),
                        "cloud_factor": exact(
                            cloud_detection_factor(weather, float(a[2]), float(b[2]))
                            if weather is not None and weather.cloud_layers
                            else 1.0
                        ),
                    }
                )
    factors = {d["dominant_loss_factor"] for d in detections}
    assert len(factors) >= 3, f"dominant loss barely varied: {factors}"
    assert any(d["cloud_factor"] == "0.0" for d in detections), "no cloud ever blocked"
    blocked = sum(1 for q in queries if not q["visible"])
    multipliers = {q["noise_multiplier"] for q in queries if q["visible"]}
    assert blocked > 10, "almost nothing was blocked; the LOS set tests little"
    assert len(multipliers) > 2, "land cover never varied the noise multiplier"
    return {
        "default_clearance_m": exact(default_clearance),
        "queries": queries,
        "weathers": {name: motion_weather_document(w) for name, w in weathers.items()},
        "detections": detections,
    }


# The frame payloads P4.9a.2 adds to the Rust replay, by their replay keys.
REPLAY_PAYLOADS = (
    "generation_rejections",
    "mapping_state",
    "localization_state",
    "inspection_events",
    "deconfliction_events",
    "scan_mission_state",
    "target_metadata",
)
# Recorded only for the stage-4 cases, so the other references stay as they
# were captured.
SAFETY_PAYLOADS = ("safety_events",)
# The keys of the reference `meta` that describe the run rather than the
# scenario, and that the Rust writer computes itself.
META_RUN_KEYS = ("platform", "requested_duration_s", "actual_duration_s", "observation_generation")


def capture(out_dir: Path, name: str, options: ScenarioOptions) -> None:
    config = CONFIGS.get(name, CONFIG)
    scenario = build_default_scenario(options, seed=config.seed)
    bundle = export_bundle(scenario, config)
    write_bundle(out_dir / f"scenario_bundle_{name}.json", bundle)

    adaptive = scenario.adaptive_drone_controllers
    engaged_log: list[dict[str, bool]] = []

    def progress(_step: int, _total: int) -> None:
        engaged_log.append({node_id: bool(c.engaged) for node_id, c in adaptive.items()})

    result = run_simulation(scenario, config, progress=progress)
    # Exactly what `write_replay_document` puts on disk: frames through
    # `to_jsonable`, `meta` dumped as it is.
    document = json.loads(json.dumps(build_replay_document_from_result(result)))
    # The bundle's presentation IS the scenario half of this replay's meta:
    # checked here, where both are in hand, so the Rust side can trust it.
    presentation = json.loads(json.dumps(scenario_replay_metadata(scenario)))
    for key, value in presentation.items():
        assert document["meta"][key] == value, f"presentation key {key} differs from meta"
    assert bundle["presentation"] == presentation, "the bundle's presentation drifted"
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
                # The pair gates that run before the pair's first random
                # draw: which (node, target) pairs each rejected is
                # deterministic, so it is compared exactly while the
                # kinematics agree, even though later draws differ (ADR-006).
                "draw_free_rejections": sorted(
                    [r.node_id, r.target_id, r.reason]
                    for r in frame.generation_rejections
                    if r.reason in ("out_of_range", "outside_fov")
                ),
                # Every generation rejection by reason, for statistical
                # comparison of the draw-dependent gates.
                "rejection_counts": {
                    reason: sum(1 for r in frame.generation_rejections if r.reason == reason)
                    for reason in sorted({r.reason for r in frame.generation_rejections})
                },
                "generated_observations": len(frame.observations)
                + len(frame.rejected_observations),
                "tracks": len(frame.tracks),
                "mission": None
                if frame.scan_mission_state is None
                else {
                    "phase": frame.scan_mission_state.phase,
                    "coverage_fraction": exact(frame.scan_mission_state.scan_coverage_fraction),
                    "pois": [
                        [p.poi_id, p.status, p.assigned_drone_id or ""]
                        for p in frame.scan_mission_state.poi_statuses
                    ],
                    "confidences": [
                        exact(e.confidence) for e in frame.scan_mission_state.localization_estimates
                    ],
                    "localization_timed_out": bool(frame.scan_mission_state.localization_timed_out),
                },
                "engaged": engaged_log[index] if index < len(engaged_log) else {},
                # Both parties to every resolution: either one's position is
                # then a function of the other's.
                "deconfliction_drones": sorted(
                    {e.yielding_drone_id for e in frame.deconfliction_events}
                    | {e.conflicting_drone_id for e in frame.deconfliction_events}
                ),
                "replay": {
                    key: document["frames"][index].get(key)
                    for key in REPLAY_PAYLOADS
                    + (SAFETY_PAYLOADS if options.safety_blocking else ())
                },
            }
        )
    reference = {
        "_source": "scripts/capture_scenario_bundle_fixture.py",
        "los": capture_los(scenario),
        "seed": config.seed,
        "steps": config.steps,
        "dt_s": exact(config.dt_s),
        "frames": frames,
        "meta_run": {key: document["meta"][key] for key in META_RUN_KEYS},
    }
    assert len(frames) == config.steps, f"expected {config.steps} frames, got {len(frames)}"
    if name in ("scan", "scan_safety"):
        phases = {f["mission"]["phase"] for f in frames}
        assert phases >= {"scanning", "inspecting", "egress", "complete"}, phases
    elif name == "tt":
        assert adaptive, "the scenario has no search-then-follow drone; the handover is untested"
    elif name == "tt_follow":
        assert not adaptive, "the all-follow case grew an observation-fed drone"
        assert any(f["deconfliction_drones"] for f in frames), "no deconfliction to compare"
    if options.safety_blocking:
        events = [e for f in frames for e in f["replay"]["safety_events"] or []]
        reasons = {e["reason"] for e in events}
        assert "abort" in reasons, f"stage 4 never aborted: {reasons}"
        if name == "tt_safety":
            assert "warning" in reasons, f"stage 4 never warned first: {reasons}"
        assert any(e["clamped"] for e in events), "stage 4 never clamped"
    if name == "tt_safety":
        assert not adaptive, "the all-follow safety case grew an observation-fed drone"
        assert set(scenario.drone_roles.values()) == {"interceptor"}, scenario.drone_roles
    elif name == "scan_safety":
        assert set(scenario.drone_roles.values()) == {"tracker"}, scenario.drone_roles
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
    parser.add_argument("--cases", default="", help="comma-separated names (default: all)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name for name in args.cases.split(",") if name}
    unknown = wanted - set(CASES)
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    for name, options in CASES.items():
        if not wanted or name in wanted:
            capture(args.out_dir, name, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
