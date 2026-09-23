#!/usr/bin/env python3
"""Record the last two stage-1 controllers the Rust tick needs.

`looped_waypoint_path` is the search lane every search-then-follow drone flies,
and `ObservationTriggeredFollowController` is the wrapper that switches it to
`FollowPathController` when the drone's sensor first accepts an observation of
its target. With these two, every controller the default scenario composes
(`Launch(ObservationTriggered(search, follow))`) exists in Rust.

Values come from calling the reference, never from recomputing it here.

TERRAIN AND OBSTACLES ARE ORACLES

Same recorders as `capture_follow_path_fixture.py`, imported rather than
copied, so the analytic terrain and the single blocking cylinder are the same
ones FollowPath's fixture used. The Rust test replays every read and panics on
a miss, so which points the search lane probed is part of the assertion.

THE WRAPPER IS RECORDED AS A CALL LOG

`ObservationTriggeredFollowController` computes nothing: it chooses which half
to call, when to seed the follow half, and from what. So its halves here are
fakes that record every call -- `search(t)`, `follow(t)`, `seed(p, v, t)`,
`reset_state()` -- and return fixed functions of `t`. The Rust test drives the
same script of events through the port with fakes that replay those returns
and compares the call logs exactly. A port that seeded from the wrong sample,
sampled the inactive half, or reset the search half fails on the log.

Usage:
    python3 scripts/capture_controller_fixture.py --out tests/fixtures/controllers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_follow_path_fixture import (  # noqa: E402
    RecordingObstacles,
    RecordingTerrain,
    exact,
    vec,
)

from argusnet.sensing.observation_source import ObservationBatch  # noqa: E402
from argusnet.simulation.sim import (  # noqa: E402
    ObservationTriggeredFollowController,
    looped_waypoint_path,
)

LANE = [[-300.0, -200.0], [300.0, -200.0], [300.0, -100.0], [-300.0, -100.0]]


def capture_waypoint_case(
    name: str,
    *,
    waypoints,
    timestamps,
    speed_mps: float = 12.0,
    base_agl_m: float = 80.0,
    vertical_amplitude_m: float = 0.0,
    vertical_frequency_rad_s: float = 0.0,
    phase: float = 0.0,
    min_agl_m: float = 18.0,
    obstacles: RecordingObstacles | None = None,
) -> dict:
    terrain = RecordingTerrain()
    trajectory = looped_waypoint_path(
        waypoints_xy=np.asarray(waypoints, dtype=float),
        speed_mps=speed_mps,
        base_agl_m=base_agl_m,
        vertical_amplitude=vertical_amplitude_m,
        vertical_frequency=vertical_frequency_rad_s,
        phase=phase,
        terrain=terrain,
        obstacle_layer=obstacles,
        min_agl_m=min_agl_m,
    )
    samples = []
    for t in timestamps:
        position, velocity = trajectory(float(t))
        samples.append({"t": exact(t), "position": vec(position), "velocity": vec(velocity)})
    return {
        "name": name,
        "waypoints_xy_m": [vec(p) for p in waypoints],
        "speed_mps": exact(speed_mps),
        "base_agl_m": exact(base_agl_m),
        "vertical_amplitude_m": exact(vertical_amplitude_m),
        "vertical_frequency_rad_s": exact(vertical_frequency_rad_s),
        "phase": exact(phase),
        "min_agl_m": exact(min_agl_m),
        "avoid_obstacles": obstacles is not None,
        "samples": samples,
        "terrain_oracle": terrain.oracle(),
        "obstacle_oracle": [] if obstacles is None else obstacles.oracle(),
        "obstacle_push_oracle": [] if obstacles is None else obstacles.push_oracle(),
    }


def capture_waypoint_cases() -> list[dict]:
    """Every branch of the per-tick closure.

    * the wrap: a timestamp past one full loop lands back on segment 0;
    * `searchsorted(side="right")`: a distance exactly on a vertex resolves to
      the segment that starts there;
    * the phase offset, including a phase above tau (the `% tau`);
    * the vertical oscillation and its analytic rate;
    * the speed floor at 0.5 m/s;
    * the terrain clamp raising a low lane (min_agl binds);
    * the obstacle push, and the no-layer path that never probes.
    """
    # Loop length: 600 + 100 + 600 + 100 = 1400 m. At 12 m/s the vertex at
    # 600 m is reached at t=50 exactly, and one full loop takes 116.67 s.
    ts = [0.0, 1.0, 25.0, 50.0, 55.5, 58.3333333333, 110.0, 120.0, 240.25]
    return [
        capture_waypoint_case("lane_basic", waypoints=LANE, timestamps=ts),
        capture_waypoint_case(
            "phase_offset_above_tau", waypoints=LANE, timestamps=ts[:5], phase=7.5
        ),
        capture_waypoint_case(
            "vertical_oscillation",
            waypoints=LANE,
            timestamps=ts[:6],
            vertical_amplitude_m=15.0,
            vertical_frequency_rad_s=0.21,
            phase=1.3,
        ),
        capture_waypoint_case(
            "speed_floor", waypoints=LANE, timestamps=[0.0, 3.0, 40.0], speed_mps=0.1
        ),
        capture_waypoint_case(
            "low_lane_clamped",
            waypoints=LANE,
            timestamps=[0.0, 20.0, 60.0],
            base_agl_m=4.0,
            vertical_amplitude_m=10.0,
            vertical_frequency_rad_s=0.5,
            min_agl_m=18.0,
        ),
        capture_waypoint_case(
            "obstacle_on_lane",
            waypoints=LANE,
            # Lane 0 runs along y=-200; the cylinder sits on it at x=0.
            timestamps=[0.0, 20.0, 24.0, 25.0, 26.0, 30.0, 60.0],
            base_agl_m=40.0,
            obstacles=RecordingObstacles(center_xy=[0.0, -200.0], radius_m=30.0, top_z_m=500.0),
        ),
        # Two diagonal segments whose lengths `hypot` and `sqrt(x*x + y*y)`
        # round differently (found by search: 100.10000000000001, 153.4).
        # Every other case has exactly representable lengths, where the two
        # agree, so a port using `hypot` passed them all. Added after a
        # mutation run: `wp_hypot` survived.
        capture_waypoint_case(
            "diagonal_lengths_use_numpy_norm",
            waypoints=[[0.0, 0.0], [100.10000000000001, 153.4], [-80.0, 153.4]],
            timestamps=[0.0, 7.0, 16.0, 21.0, 33.0, 41.0],
        ),
        capture_waypoint_case(
            "degenerate_segment_stepped_over",
            waypoints=[[0.0, 0.0], [0.0, 0.0], [200.0, 0.0], [200.0, 150.0]],
            timestamps=[0.0, 5.0, 16.6666666667, 30.0],
        ),
    ]


class RecordingHalf:
    """A fake trajectory half: returns a fixed function of t, logs calls."""

    def __init__(self, label: str, log: list, offset: float) -> None:
        self.label = label
        self.log = log
        self.offset = offset
        self.returns: dict[str, dict] = {}

    def __call__(self, timestamp_s: float):
        t = float(timestamp_s)
        position = np.array([self.offset + 3.0 * t, -2.0 * t + self.offset, 100.0 + 0.5 * t])
        velocity = np.array([3.0, -2.0, 0.5 + self.offset * 0.01])
        self.log.append({"call": self.label, "t": exact(t)})
        self.returns[exact(t)] = {"position": vec(position), "velocity": vec(velocity)}
        return position, velocity


class RecordingFollow(RecordingHalf):
    def seed(self, *, position, velocity, timestamp_s) -> None:
        self.log.append(
            {
                "call": "seed",
                "position": vec(position),
                "velocity": vec(velocity),
                "t": exact(timestamp_s),
            }
        )

    def reset_state(self) -> None:
        self.log.append({"call": "reset_follow"})


def batch(counts: dict[tuple[str, str], int]) -> ObservationBatch:
    return ObservationBatch(
        observations=[],
        attempted_count=0,
        rejection_counts={},
        accepted_by_target={},
        rejected_by_target={},
        accepted_by_node_target=dict(counts),
    )


def capture_observation_case(name: str, script: list[dict]) -> dict:
    log: list = []
    search = RecordingHalf("search", log, 0.0)
    follow = RecordingFollow("follow", log, 1000.0)
    controller = ObservationTriggeredFollowController(
        node_id="drone-1",
        search_trajectory=search,
        follow_trajectory=follow,
        preferred_target_id="target-a",
    )
    outputs = []
    for event in script:
        kind = event["event"]
        if kind == "sample":
            position, velocity = controller(event["t"])
            outputs.append(
                {"event": "sample", "position": vec(position), "velocity": vec(velocity)}
            )
        elif kind == "observe":
            counts = {tuple(k.split("|")): v for k, v in event["counts"].items()}
            controller.update_from_frame(None, batch(counts))
            outputs.append({"event": "observe", "engaged": controller.engaged})
        elif kind == "reset":
            controller.reset_state()
            outputs.append({"event": "reset", "engaged": controller.engaged})
        else:
            raise ValueError(kind)
    serial_script = [
        {**e, "t": exact(e["t"])} if "t" in e else dict(e) for e in script
    ]
    return {
        "name": name,
        "script": serial_script,
        "outputs": outputs,
        "calls": log,
        "search_returns": search.returns,
        "follow_returns": follow.returns,
    }


def capture_observation_cases() -> list[dict]:
    mine = {"drone-1|target-a": 2}
    return [
        # Search, see the target, then follow -- seeded from the last search
        # sample at ITS timestamp, not the next one.
        capture_observation_case(
            "engage_after_search",
            [
                {"event": "sample", "t": 0.0},
                {"event": "sample", "t": 1.0},
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 2.0},
                {"event": "sample", "t": 3.0},
            ],
        ),
        # Observations that are not this node's, or not this target, or zero:
        # none engages.
        capture_observation_case(
            "wrong_pair_does_not_engage",
            [
                {"event": "sample", "t": 0.0},
                {"event": "observe", "counts": {"drone-2|target-a": 5}},
                {"event": "observe", "counts": {"drone-1|target-b": 5}},
                {"event": "observe", "counts": {"drone-1|target-a": 0}},
                {"event": "sample", "t": 1.0},
            ],
        ),
        # Engaged is one-way: a later observation does not re-seed.
        capture_observation_case(
            "second_observation_ignored",
            [
                {"event": "sample", "t": 0.0},
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 1.0},
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 2.0},
            ],
        ),
        # Engaging before any sample: engages, but nothing to seed from.
        capture_observation_case(
            "engage_before_any_sample_does_not_seed",
            [
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 0.0},
            ],
        ),
        # Reset returns to search and resets only the follow half; a
        # re-engagement then seeds from the post-reset sample.
        capture_observation_case(
            "reset_then_reengage",
            [
                {"event": "sample", "t": 0.0},
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 1.0},
                {"event": "reset"},
                {"event": "sample", "t": 5.0},
                {"event": "observe", "counts": mine},
                {"event": "sample", "t": 6.0},
            ],
        ),
    ]


def check_coverage(fixture: dict) -> None:
    waypoint = {c["name"]: c for c in fixture["looped_waypoint_path"]}
    pushed = waypoint["obstacle_on_lane"]
    assert any(p["collides"] for p in pushed["obstacle_oracle"]), "no sample hit the obstacle"
    assert pushed["obstacle_push_oracle"], "the obstacle never pushed"
    assert waypoint["lane_basic"]["obstacle_oracle"] == [], "the no-layer path probed"
    clamped = waypoint["low_lane_clamped"]
    assert any(
        float(c["clamped_z_m"]) > float(c["z_m"]) for c in clamped["terrain_oracle"]["clamps"]
    ), "min_agl never bound in the low-lane case"
    floor = waypoint["speed_floor"]["samples"]
    speed = float(np.hypot(float(floor[1]["velocity"][0]), float(floor[1]["velocity"][1])))
    assert abs(speed - 0.5) < 1e-12, f"speed floor not applied: {speed}"
    # t=50 is exactly the 600 m vertex: the direction must be the SECOND
    # segment's (+y), which is what side="right" means.
    vertex = waypoint["lane_basic"]["samples"][3]
    assert vertex["t"] == "50.0"
    assert float(vertex["velocity"][1]) > 0.0 and float(vertex["velocity"][0]) == 0.0, vertex

    observation = {c["name"]: c for c in fixture["observation_triggered_follow"]}
    engage = observation["engage_after_search"]["calls"]
    seeds = [c for c in engage if c["call"] == "seed"]
    assert len(seeds) == 1 and seeds[0]["t"] == "1.0", seeds
    assert not any(
        c["call"] == "seed" for c in observation["engage_before_any_sample_does_not_seed"]["calls"]
    )
    assert not any(c["call"] == "follow" for c in observation["wrong_pair_does_not_engage"]["calls"])
    reset_calls = observation["reset_then_reengage"]["calls"]
    assert {"call": "reset_follow"} in reset_calls
    assert [c for c in reset_calls if c["call"] == "seed"][-1]["t"] == "5.0"


def capture() -> dict:
    fixture = {
        "_source": "scripts/capture_controller_fixture.py",
        "_reference": "src/argusnet/simulation/sim.py",
        "looped_waypoint_path": capture_waypoint_cases(),
        "observation_triggered_follow": capture_observation_cases(),
    }
    check_coverage(fixture)
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        f"  {len(fixture['looped_waypoint_path'])} waypoint cases, "
        f"{len(fixture['observation_triggered_follow'])} observation-follow cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
