#!/usr/bin/env python3
"""Record the two trajectory wrappers every controller is built out of.

`LaunchableTrajectoryController` and `_apply_weather_adjustments` (with its
`WeatherAdjustedTrajectory` wrapper) are the outermost layers of a controller:
one decides whether the platform is still on the ground, the other applies
wind and weather to whatever the inner trajectory produced. Both are on the
reference's own list of helpers that block controller extraction, at
`sim.py` line 136.

Values come from calling the reference, not from reimplementing its formulas
here -- a generator that recomputed them would agree with a port that shared
its author's misreading while both disagreed with Python.

TERRAIN AND WEATHER ARE ORACLES, NOT SUBJECTS

Same technique as `capture_motion_fixture.py`, for the same reason. The
terrain is already Rust; the weather model is a separate port that belongs to
sensing, not to motion. So every `height_at`, every `flight_speed_penalty`
and every `wind_at` the reference performed is recorded with its arguments,
and the Rust test replays them through a stub that panics on an unrecorded
read.

That makes the *call pattern* part of the assertion, which matters more here
than it did for the motion substrate, because `_apply_weather_adjustments`
reads the terrain **twice** -- once before the speed penalty so the penalty
sees a true AGL, and once after the wind drift has moved the platform
somewhere else. A port that clamped once would produce a plausible altitude
and replay one fewer read.

Usage:
    python3 scripts/capture_trajectory_fixture.py --out tests/fixtures/trajectory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.simulation.sim import (  # noqa: E402
    LaunchableTrajectoryController,
    WeatherAdjustedTrajectory,
    _apply_weather_adjustments,
)
from argusnet.world.weather import (  # noqa: E402
    PrecipitationModel,
    WeatherModel,
    WindModel,
)


def exact(value: float) -> str:
    """Render a float as a string; `serde_json`'s parser is off by an ULP on
    some literals, which broke an exact comparison in the sensor error model."""
    return repr(float(value))


def vec(values) -> list[str]:
    return [exact(v) for v in np.asarray(values, dtype=float).ravel()]


class RecordingTerrain:
    """Records every height read, keyed by the point asked for."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.heights: dict[tuple[str, str], str] = {}
        self.calls: list[list[str]] = []

    def height_at(self, x_m: float, y_m: float) -> float:
        value = float(self._inner(x_m, y_m))
        self.heights[(exact(x_m), exact(y_m))] = exact(value)
        self.calls.append([exact(x_m), exact(y_m)])
        return value


class RecordingWeather:
    """Records the two weather reads the motion path makes.

    Delegates to a real `WeatherModel`, so the numbers are the reference's.
    The keys are what pin the call pattern: `flight_speed_penalty` is asked
    for an altitude, and getting that altitude wrong -- absolute instead of
    AGL, or before the clamp instead of after -- is a lookup miss rather than
    a plausible wrong answer.
    """

    def __init__(self, inner: WeatherModel) -> None:
        self._inner = inner
        self.penalties: dict[str, str] = {}
        self.winds: dict[tuple[str, str], list[str]] = {}

    def flight_speed_penalty(self, altitude_m: float) -> float:
        value = float(self._inner.flight_speed_penalty(altitude_m))
        self.penalties[exact(altitude_m)] = exact(value)
        return value

    @property
    def wind(self) -> RecordingWind:
        return RecordingWind(self._inner.wind, self.winds)


class RecordingWind:
    def __init__(self, inner: WindModel, sink: dict) -> None:
        self._inner = inner
        self._sink = sink

    def wind_at(self, altitude_m: float, time_s: float) -> np.ndarray:
        value = np.asarray(self._inner.wind_at(altitude_m, time_s), dtype=float)
        self._sink[(exact(altitude_m), exact(time_s))] = vec(value)
        return value


def terrain_surface(x_m: float, y_m: float) -> float:
    """A tilted plane with relief.

    Flat ground would make the two terrain reads in
    `_apply_weather_adjustments` return the same number, and a port that did
    only one of them would pass.
    """
    return 40.0 + 0.08 * x_m + 0.05 * y_m


BOUNDS = {"x_min_m": -400.0, "x_max_m": 600.0, "y_min_m": -200.0, "y_max_m": 300.0}


def weather_model() -> WeatherModel:
    """Wind strong enough to move the platform, and rain so the precipitation
    branch of `flight_speed_penalty` contributes as well as the wind branch."""
    return WeatherModel(
        wind=WindModel(
            base_speed_mps=9.0,
            base_heading_rad=0.7,
            gust_amplitude_mps=3.0,
            gust_period_s=12.0,
            altitude_scaling=0.3,
        ),
        precipitation=PrecipitationModel(precip_type="rain", rate_mmph=6.0),
    )


def capture_weather_adjustments() -> list[dict]:
    """`_apply_weather_adjustments`, every branch.

    Five things a port can get wrong and still look right:

    * **Weather `None` returns copies.** Not the inputs themselves.
    * **The terrain is read twice.** Before the penalty and after the drift.
    * **Altitude means different things** depending on whether terrain is
      supplied: AGL when it is, the raw `z` when it is not. That is the
      reference's, and it is the value handed to both weather reads.
    * **Only the horizontal components move.** `vz` is never touched by the
      penalty or the wind.
    * **`wind_at` is given the simulation timestamp**, while
      `flight_speed_penalty` internally asks for `wind_at(altitude, 0.0)` --
      so the gust phase applies to the drift and not to the penalty.
    """
    cases = []

    def case(
        name: str,
        position,
        velocity,
        *,
        timestamp_s: float,
        dt_s: float = 1.0,
        weather: bool = True,
        terrain: bool = True,
        bounds: bool = True,
        min_agl_m: float = 18.0,
        wind_scale: float = 0.35,
    ) -> None:
        recording_terrain = RecordingTerrain(terrain_surface) if terrain else None
        recording_weather = RecordingWeather(weather_model()) if weather else None
        adjusted_position, adjusted_velocity = _apply_weather_adjustments(
            np.asarray(position, dtype=float),
            np.asarray(velocity, dtype=float),
            timestamp_s=timestamp_s,
            dt_s=dt_s,
            weather=recording_weather,
            terrain=recording_terrain,
            map_bounds_m=BOUNDS if bounds else None,
            min_agl_m=min_agl_m,
            wind_scale=wind_scale,
        )
        cases.append(
            {
                "name": name,
                "position": vec(position),
                "velocity": vec(velocity),
                "timestamp_s": exact(timestamp_s),
                "dt_s": exact(dt_s),
                "has_weather": weather,
                "has_terrain": terrain,
                "has_bounds": bounds,
                "min_agl_m": exact(min_agl_m),
                "wind_scale": exact(wind_scale),
                "terrain_calls": recording_terrain.calls if recording_terrain else [],
                "terrain_oracle": (
                    [
                        {"x_m": k[0], "y_m": k[1], "height_m": v}
                        for k, v in sorted(recording_terrain.heights.items())
                    ]
                    if recording_terrain
                    else []
                ),
                "penalty_oracle": (
                    [
                        {"altitude_m": k, "penalty": v}
                        for k, v in sorted(recording_weather.penalties.items())
                    ]
                    if recording_weather
                    else []
                ),
                "wind_oracle": (
                    [
                        {"altitude_m": k[0], "time_s": k[1], "wind": v}
                        for k, v in sorted(recording_weather.winds.items())
                    ]
                    if recording_weather
                    else []
                ),
                "adjusted_position": vec(adjusted_position),
                "adjusted_velocity": vec(adjusted_velocity),
            }
        )

    # No weather at all: a passthrough, and the terrain is not read either.
    case(
        "no_weather_is_a_passthrough",
        [100.0, 50.0, 200.0],
        [12.0, -4.0, 1.0],
        timestamp_s=30.0,
        weather=False,
    )
    # The ordinary case: clamp, penalty, drift, bounds, clamp again.
    case("full_path", [100.0, 50.0, 200.0], [12.0, -4.0, 1.0], timestamp_s=30.0)
    # Below the terrain floor: the FIRST clamp raises it, and the penalty
    # therefore sees the clamped AGL rather than the underground one.
    case(
        "below_floor_is_clamped_before_the_penalty",
        [100.0, 50.0, 10.0],
        [12.0, -4.0, 0.0],
        timestamp_s=30.0,
    )
    # No terrain: altitude is the raw z, not an AGL, and neither clamp runs.
    case(
        "without_terrain_altitude_is_absolute",
        [100.0, 50.0, 200.0],
        [12.0, -4.0, 1.0],
        timestamp_s=30.0,
        terrain=False,
    )
    # No bounds: the drift is not clamped horizontally.
    case(
        "without_bounds_drift_is_unclamped",
        [580.0, 280.0, 200.0],
        [12.0, -4.0, 0.0],
        timestamp_s=30.0,
        bounds=False,
    )
    # At the edge the wind actually pushes toward, with a long dt so the
    # drift certainly crosses it. The margin is 1.0, hardcoded at the call
    # site, so the clamped value is the bound inset by exactly that.
    #
    # The first attempt at this case sat at the x_max edge -- and this
    # wind blows west and south, so the drift moved the platform AWAY from
    # the boundary and the clamp never ran. A mutation run caught it:
    # replacing the margin with 0.0 survived.
    case(
        "drift_past_the_low_edge_is_clamped",
        [-395.0, -190.0, 200.0],
        [12.0, -4.0, 0.0],
        timestamp_s=30.0,
        dt_s=4.0,
    )
    # A different timestamp moves the gust and therefore the drift, while the
    # penalty (which asks for wind at t=0) does not move.
    case("gust_phase_moves_the_drift", [100.0, 50.0, 200.0], [12.0, -4.0, 1.0], timestamp_s=33.0)
    # dt and wind_scale both scale the drift, and they are separate knobs.
    case(
        "longer_dt_drifts_further",
        [100.0, 50.0, 200.0],
        [12.0, -4.0, 1.0],
        timestamp_s=30.0,
        dt_s=4.0,
    )
    case(
        "wind_scale_zero_removes_the_drift",
        [100.0, 50.0, 200.0],
        [12.0, -4.0, 1.0],
        timestamp_s=30.0,
        wind_scale=0.0,
    )
    # Zero horizontal velocity: the penalty has nothing to scale, so the
    # position still moves (wind) while the velocity does not.
    case("still_platform_still_drifts", [100.0, 50.0, 200.0], [0.0, 0.0, 0.0], timestamp_s=30.0)
    return cases


class RecordingInner:
    """A stand-in operational trajectory that records when it was asked.

    `LaunchableTrajectoryController` must not call the wrapped trajectory at
    all before the climb finishes -- a controller that consulted it during
    the climb would be evaluating a path the platform is not on yet, and on a
    stateful inner trajectory that is a side effect, not just waste.
    """

    def __init__(self) -> None:
        self.timestamps: list[str] = []
        self.reset_count = 0

    def __call__(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        self.timestamps.append(exact(timestamp_s))
        # Deterministic and distinctive, so a delegated sample is
        # recognisable in the output.
        return (
            np.array([300.0 + timestamp_s, 120.0 - timestamp_s, 260.0], dtype=float),
            np.array([7.0, -3.0, 0.5], dtype=float),
        )

    def reset_state(self) -> None:
        self.reset_count += 1


def capture_launch() -> list[dict]:
    """`LaunchableTrajectoryController`, phase by phase.

    The smooth-step is `3f^2 - 2f^3` and its rate is the analytic derivative
    `A * (6f - 6f^2) / duration`, which is **zero at both ends** -- the climb
    starts and finishes at zero vertical speed. A port using a linear ramp
    would be close in position and wrong in velocity everywhere.
    """
    cases = []

    def case(
        name: str,
        *,
        launch_at: float | None,
        sample_at: float,
        climb_duration_s: float = 8.0,
        operational_altitude_agl: float = 230.0,
        weather: bool = False,
        trigger_twice_at: float | None = None,
    ) -> None:
        inner = RecordingInner()
        recording_weather = RecordingWeather(weather_model()) if weather else None
        recording_terrain = RecordingTerrain(terrain_surface) if weather else None
        controller = LaunchableTrajectoryController(
            station_position=np.array([50.0, -25.0, 12.0], dtype=float),
            operational_trajectory=inner,
            climb_duration_s=climb_duration_s,
            operational_altitude_agl=operational_altitude_agl,
            weather=recording_weather,
            terrain=recording_terrain,
            map_bounds_m=BOUNDS if weather else None,
        )
        if launch_at is not None:
            controller.trigger_launch(launch_at, "t-1")
        if trigger_twice_at is not None:
            controller.trigger_launch(trigger_twice_at, "t-2")
        position, velocity = controller(sample_at)
        cases.append(
            {
                "name": name,
                "station_position": vec([50.0, -25.0, 12.0]),
                "launch_at": None if launch_at is None else exact(launch_at),
                "trigger_twice_at": (None if trigger_twice_at is None else exact(trigger_twice_at)),
                "sample_at": exact(sample_at),
                "climb_duration_s": exact(climb_duration_s),
                "operational_altitude_agl": exact(operational_altitude_agl),
                "has_weather": weather,
                "inner_calls": inner.timestamps,
                "terrain_oracle": (
                    [
                        {"x_m": k[0], "y_m": k[1], "height_m": v}
                        for k, v in sorted(recording_terrain.heights.items())
                    ]
                    if recording_terrain
                    else []
                ),
                "penalty_oracle": (
                    [
                        {"altitude_m": k, "penalty": v}
                        for k, v in sorted(recording_weather.penalties.items())
                    ]
                    if recording_weather
                    else []
                ),
                "wind_oracle": (
                    [
                        {"altitude_m": k[0], "time_s": k[1], "wind": v}
                        for k, v in sorted(recording_weather.winds.items())
                    ]
                    if recording_weather
                    else []
                ),
                "launched": bool(controller.launched),
                "launch_target_id": controller.launch_target_id,
                "position": vec(position),
                "velocity": vec(velocity),
            }
        )

    # Never launched: parked, and the inner trajectory is never consulted.
    case("unlaunched_sits_at_the_station", launch_at=None, sample_at=40.0)
    # Launched, but sampled BEFORE the launch time. `elapsed < 0` is its own
    # guard -- without it the smooth-step runs on a negative fraction and
    # produces a descent below the station.
    case("sampled_before_launch_time", launch_at=50.0, sample_at=20.0)
    # Exactly at launch: fraction 0, so zero climb rate and no altitude yet.
    case("at_launch_instant", launch_at=20.0, sample_at=20.0)
    # Quarter, half and three-quarters through the climb.
    case("quarter_through_climb", launch_at=20.0, sample_at=22.0)
    case("half_through_climb", launch_at=20.0, sample_at=24.0)
    case("three_quarters_through_climb", launch_at=20.0, sample_at=26.0)
    # Exactly at the end of the climb: `elapsed < climb_duration_s` is
    # strict, so this is the FIRST operational sample, not the last climb
    # one. A port using `<=` delegates one tick late.
    case("exactly_at_climb_end_is_operational", launch_at=20.0, sample_at=28.0)
    case("well_into_operation", launch_at=20.0, sample_at=60.0)
    # Launching twice keeps the first launch time and the first target.
    case("relaunch_is_ignored", launch_at=20.0, trigger_twice_at=25.0, sample_at=24.0)
    # A zero-length climb: every sample is operational, and the fraction is
    # never evaluated (which would divide by zero).
    case(
        "zero_climb_duration_goes_straight_to_operation",
        launch_at=20.0,
        sample_at=20.0,
        climb_duration_s=0.0,
    )
    # The operational phase with weather on: the delegated sample is then
    # pushed through the full adjustment path.
    case("operational_sample_is_weather_adjusted", launch_at=20.0, sample_at=60.0, weather=True)
    return cases


def capture_weather_wrapper() -> list[dict]:
    """`WeatherAdjustedTrajectory` delegates and adjusts, and resets through.

    Thin, but it carries one thing worth pinning: its `dt_s` is a field with
    its own default, whereas `LaunchableTrajectoryController` never passes
    one and therefore always gets `_apply_weather_adjustments`' default of
    1.0 -- regardless of the simulation timestep. Two defaults, one of which
    is unreachable from the launch path.
    """
    cases = []
    for name, dt_s in [("default_dt", 1.0), ("explicit_dt", 0.25)]:
        inner = RecordingInner()
        recording_weather = RecordingWeather(weather_model())
        recording_terrain = RecordingTerrain(terrain_surface)
        wrapper = WeatherAdjustedTrajectory(
            trajectory=inner,
            weather=recording_weather,
            terrain=recording_terrain,
            map_bounds_m=BOUNDS,
            min_agl_m=18.0,
            dt_s=dt_s,
        )
        position, velocity = wrapper(30.0)
        wrapper.reset_state()
        cases.append(
            {
                "name": name,
                "dt_s": exact(dt_s),
                "sample_at": exact(30.0),
                "inner_calls": inner.timestamps,
                "reset_reached_inner": inner.reset_count == 1,
                "terrain_oracle": [
                    {"x_m": k[0], "y_m": k[1], "height_m": v}
                    for k, v in sorted(recording_terrain.heights.items())
                ],
                "penalty_oracle": [
                    {"altitude_m": k, "penalty": v}
                    for k, v in sorted(recording_weather.penalties.items())
                ],
                "wind_oracle": [
                    {"altitude_m": k[0], "time_s": k[1], "wind": v}
                    for k, v in sorted(recording_weather.winds.items())
                ],
                "position": vec(position),
                "velocity": vec(velocity),
            }
        )
    return cases


def check_coverage(fixture: dict) -> None:
    """A fixture that misses a branch blesses a port that dropped it."""
    weather = {c["name"]: c for c in fixture["apply_weather_adjustments"]}

    passthrough = weather["no_weather_is_a_passthrough"]
    assert passthrough["adjusted_position"] == passthrough["position"]
    assert passthrough["terrain_calls"] == [], "the no-weather path read terrain"

    full = weather["full_path"]
    assert len(full["terrain_calls"]) == 2, (
        f"the terrain was read {len(full['terrain_calls'])} times, not twice"
    )
    assert full["terrain_calls"][0] != full["terrain_calls"][1], (
        "both terrain reads were at the same point, so a port that clamped once would pass"
    )
    # The horizontal velocity was scaled down and the vertical was not.
    for axis in (0, 1):
        assert abs(float(full["adjusted_velocity"][axis])) < abs(float(full["velocity"][axis])), (
            f"the speed penalty did not scale axis {axis}"
        )
    assert full["adjusted_velocity"][2] == full["velocity"][2], "vz was touched"
    # The platform moved horizontally even though the clamp only moves z.
    assert full["adjusted_position"][:2] != full["position"][:2], "no wind drift"

    clamped = weather["below_floor_is_clamped_before_the_penalty"]
    penalty_altitudes = [p["altitude_m"] for p in clamped["penalty_oracle"]]
    assert penalty_altitudes == [exact(18.0)], (
        f"the penalty saw {penalty_altitudes}, not the clamped 18 m AGL -- so "
        "the clamp does not precede it"
    )

    absolute = weather["without_terrain_altitude_is_absolute"]
    assert absolute["penalty_oracle"][0]["altitude_m"] == exact(200.0), (
        "without terrain the altitude should be the raw z"
    )
    assert absolute["terrain_calls"] == []

    # Both axes pinned to the bound inset by exactly the 1.0 margin. An
    # inequality would pass for a margin of zero, which is what let that
    # mutant through.
    edge = weather["drift_past_the_low_edge_is_clamped"]
    assert edge["adjusted_position"][0] == exact(BOUNDS["x_min_m"] + 1.0), (
        f"x was not clamped to the inset bound: {edge['adjusted_position'][0]}"
    )
    assert edge["adjusted_position"][1] == exact(BOUNDS["y_min_m"] + 1.0), (
        f"y was not clamped to the inset bound: {edge['adjusted_position'][1]}"
    )

    assert (
        weather["gust_phase_moves_the_drift"]["adjusted_position"]
        != weather["full_path"]["adjusted_position"]
    ), "changing the timestamp did not move the drift"
    assert (
        weather["longer_dt_drifts_further"]["adjusted_position"]
        != weather["full_path"]["adjusted_position"]
    ), "dt_s did not scale the drift"
    assert (
        weather["wind_scale_zero_removes_the_drift"]["adjusted_position"][:2]
        == weather["wind_scale_zero_removes_the_drift"]["position"][:2]
    ), "wind_scale=0 still drifted"

    launch = {c["name"]: c for c in fixture["launch_controller"]}
    assert launch["unlaunched_sits_at_the_station"]["inner_calls"] == [], (
        "an unlaunched controller consulted its operational trajectory"
    )
    assert launch["quarter_through_climb"]["inner_calls"] == [], (
        "a climbing controller consulted its operational trajectory"
    )
    assert launch["exactly_at_climb_end_is_operational"]["inner_calls"] == [exact(28.0)], (
        "the strict `<` boundary did not delegate at exactly the climb end"
    )

    # Zero climb rate at both ends of the smooth step, non-zero in between.
    assert float(launch["at_launch_instant"]["velocity"][2]) == 0.0
    assert float(launch["half_through_climb"]["velocity"][2]) > 0.0
    # Symmetric: a quarter and three-quarters through climb at the same rate.
    assert (
        launch["quarter_through_climb"]["velocity"][2]
        == launch["three_quarters_through_climb"]["velocity"][2]
    ), "the smooth-step rate is not symmetric, so it is not 6f - 6f^2"

    assert launch["relaunch_is_ignored"]["launch_target_id"] == "t-1", (
        "a second trigger_launch overwrote the first"
    )
    assert (
        launch["sampled_before_launch_time"]["position"]
        == launch["unlaunched_sits_at_the_station"]["position"]
    ), "a negative elapsed did not park the platform"

    wrapper = {c["name"]: c for c in fixture["weather_adjusted_trajectory"]}
    assert wrapper["default_dt"]["position"] != wrapper["explicit_dt"]["position"], (
        "dt_s made no difference through the wrapper"
    )
    assert all(c["reset_reached_inner"] for c in wrapper.values())


def capture() -> dict:
    fixture = {
        "_source": "scripts/capture_trajectory_fixture.py",
        "_reference": "src/argusnet/simulation/sim.py",
        "bounds": {k: exact(v) for k, v in BOUNDS.items()},
        "apply_weather_adjustments": capture_weather_adjustments(),
        "launch_controller": capture_launch(),
        "weather_adjusted_trajectory": capture_weather_wrapper(),
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
        f"  {len(fixture['apply_weather_adjustments'])} weather, "
        f"{len(fixture['launch_controller'])} launch, "
        f"{len(fixture['weather_adjusted_trajectory'])} wrapper"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
