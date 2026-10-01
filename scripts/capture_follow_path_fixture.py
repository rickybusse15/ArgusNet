#!/usr/bin/env python3
"""Record `FollowPathController`, end to end.

The last and largest of the three controllers `sim.py` line 136 names as
blocked from extraction. `__call__` is ~370 lines in one method and is not
divisible: the candidate scan, the replan decision, the motion along the
route, the exclusion-zone passes and the altitude computation all read and
write one `_state` dict. Porting a piece of it would ship dead code, so the
whole thing is driven here.

Values come from calling the reference, not from reimplementing its formulas
here -- a generator that recomputed them would agree with a port that shared
its author's misreading while both disagreed with Python.

EVERY EXTERNAL READ IS AN ORACLE

Same technique as the motion and trajectory fixtures, extended to five
capabilities: terrain height/gradient/clamp, obstacle containment,
line-of-sight, and the planner's `plan_route` / `nearest_free_point`. Each is
recorded with the arguments it was called with, and the Rust test replays
them through stubs that panic on an unrecorded read.

That is doing more work here than it did for the substrate. This controller
*chooses* between candidates using those reads, so the call pattern is the
decision: a port that scored candidates in a different order, or skipped the
obstacle test before the LOS test, replays different reads and fails on a
lookup rather than on a number that happens to match.

THE CONTROLLER IS STATEFUL, SO CASES ARE SEQUENCES

`_state` carries a memo, the previous position, heading, route and smoothed
terrain height between calls. A single-sample case cannot reach most of the
interesting behaviour -- acceleration limiting needs a previous velocity, the
orbit blend needs a previous position, and the terrain-following filter needs
a previous height. So each case is a *sequence* of timestamps, and every
sample in it is recorded.

Usage:
    python3 scripts/capture_follow_path_fixture.py --out tests/fixtures/follow_path.json
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

from argusnet.core.types import MissionZone  # noqa: E402
from argusnet.simulation.sim import FollowPathController  # noqa: E402
from argusnet.world.visibility import VisibilityResult  # noqa: E402


def exact(value: float) -> str:
    """Render a float as a string; `serde_json`'s parser is off by an ULP on
    some literals, which broke an exact comparison in the sensor error model."""
    return repr(float(value))


def vec(values) -> list[str]:
    return [exact(v) for v in np.asarray(values, dtype=float).ravel()]


BOUNDS = {"x_min_m": -2000.0, "x_max_m": 2000.0, "y_min_m": -2000.0, "y_max_m": 2000.0}


def terrain_surface(x_m: float, y_m: float) -> float:
    """A tilted plane plus a bump.

    The plane gives every gradient a non-zero value; the bump gives the
    terrain-following filter something to actually smooth, which a flat sheet
    would not.
    """
    return (
        60.0
        + 0.05 * x_m
        + 0.03 * y_m
        + 80.0 * math.exp(-((x_m - 400.0) ** 2 + (y_m - 200.0) ** 2) / (2.0 * 250.0**2))
    )


class RecordingTerrain:
    def __init__(self) -> None:
        self.heights: dict[tuple[str, str], str] = {}
        self.gradients: dict[tuple[str, str, str], list[str]] = {}
        self.clamps: dict[tuple[str, str, str, str], str] = {}

    def height_at(self, x_m: float, y_m: float) -> float:
        value = terrain_surface(float(x_m), float(y_m))
        self.heights[(exact(x_m), exact(y_m))] = exact(value)
        return value

    def gradient_at(self, x_m: float, y_m: float, delta_m: float | None = None) -> np.ndarray:
        # Central difference at the same step the reference's TerrainLayer
        # would use, computed from the analytic surface so the fixture does
        # not need a built terrain.
        delta = 0.5 if delta_m is None else float(delta_m)
        gx = (terrain_surface(x_m + delta, y_m) - terrain_surface(x_m - delta, y_m)) / (2 * delta)
        gy = (terrain_surface(x_m, y_m + delta) - terrain_surface(x_m, y_m - delta)) / (2 * delta)
        gradient = np.array([gx, gy], dtype=float)
        key_delta = "default" if delta_m is None else exact(delta_m)
        self.gradients[(exact(x_m), exact(y_m), key_delta)] = vec(gradient)
        return gradient

    def clamp_altitude(self, xy_m, z_m: float, min_agl_m: float) -> float:
        xy = np.asarray(xy_m, dtype=float)
        ground = terrain_surface(float(xy[0]), float(xy[1]))
        value = max(float(z_m), ground + max(float(min_agl_m), 0.0))
        self.clamps[(exact(xy[0]), exact(xy[1]), exact(z_m), exact(min_agl_m))] = exact(value)
        return value

    def oracle(self) -> dict:
        return {
            "heights": [
                {"x_m": k[0], "y_m": k[1], "height_m": v} for k, v in sorted(self.heights.items())
            ],
            "gradients": [
                {"x_m": k[0], "y_m": k[1], "delta_m": k[2], "gradient": v}
                for k, v in sorted(self.gradients.items())
            ],
            "clamps": [
                {"x_m": k[0], "y_m": k[1], "z_m": k[2], "min_agl_m": k[3], "clamped_z_m": v}
                for k, v in sorted(self.clamps.items())
            ],
        }


class RecordingObstacles:
    """A single blocking cylinder, recorded per probe.

    The controller uses containment to *reject candidates*, so which points
    were probed is part of the decision and not just a lookup.
    """

    def __init__(self, center_xy=None, radius_m: float = 0.0, top_z_m: float = 0.0) -> None:
        self.center_xy = None if center_xy is None else np.asarray(center_xy, dtype=float)
        self.radius_m = radius_m
        self.top_z_m = top_z_m
        self.probes: dict[tuple[str, str, str], bool] = {}
        self.pushes: dict[tuple[str, str, str], list[str]] = {}

    def point_collides(self, x_m: float, y_m: float, z_m: float):
        hit = False
        if self.center_xy is not None and z_m <= self.top_z_m:
            hit = bool(np.hypot(x_m - self.center_xy[0], y_m - self.center_xy[1]) < self.radius_m)
        self.probes[(exact(x_m), exact(y_m), exact(z_m))] = hit
        return _PushableCylinder(self) if hit else None

    def push(self, x_m: float, y_m: float, margin_m: float) -> np.ndarray:
        offset = np.array([x_m, y_m], dtype=float) - self.center_xy
        distance = float(np.linalg.norm(offset))
        # Dead centre has no outward direction; +x, as the reference's own
        # zone push does in the same situation.
        direction = np.array([1.0, 0.0]) if distance < 1e-9 else offset / distance
        pushed = self.center_xy + direction * (self.radius_m + margin_m)
        self.pushes[(exact(x_m), exact(y_m), exact(margin_m))] = vec(pushed)
        return pushed

    def oracle(self) -> list[dict]:
        return [
            {"x_m": k[0], "y_m": k[1], "z_m": k[2], "collides": v}
            for k, v in sorted(self.probes.items())
        ]

    def push_oracle(self) -> list[dict]:
        return [
            {"x_m": k[0], "y_m": k[1], "margin_m": k[2], "pushed_xy": v}
            for k, v in sorted(self.pushes.items())
        ]


class _PushableCylinder:
    """What `point_collides` hands back: something that can push clear."""

    def __init__(self, layer: RecordingObstacles) -> None:
        self._layer = layer

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return self._layer.push(x_m, y_m, margin_m)


class RecordingQuery:
    """Line of sight, blocked inside a chosen angular wedge.

    A uniformly-visible world would make the visibility term of the candidate
    score constant, and the score's first key would never decide anything --
    so a port that dropped it would pass. The wedge makes some candidates
    invisible and therefore strictly worse.
    """

    def __init__(self, blocked: bool = False) -> None:
        self.blocked = blocked
        self.results: dict[tuple[str, str], bool] = {}

    def los(self, observer, target, *, sensor_profile=None) -> VisibilityResult:
        observer = np.asarray(observer, dtype=float)
        target = np.asarray(target, dtype=float)
        visible = True
        if self.blocked:
            bearing = math.atan2(observer[1] - target[1], observer[0] - target[0])
            visible = not (-0.4 <= bearing <= 1.2)
        key = (
            ",".join(exact(v) for v in observer),
            ",".join(exact(v) for v in target),
        )
        self.results[key] = visible
        return VisibilityResult(visible=visible)

    def oracle(self) -> list[dict]:
        return [
            {"observer": k[0].split(","), "target": k[1].split(","), "visible": v}
            for k, v in sorted(self.results.items())
        ]


class RecordingEnvironment:
    """Duck-typed stand-in for `EnvironmentModel`.

    The controller reads exactly three attributes off it. Building a real
    `EnvironmentModel` would drag a terrain build and a native obstacle layer
    into a fixture whose subject is neither.
    """

    def __init__(self, terrain, obstacles, query) -> None:
        self.terrain = terrain
        self.obstacles = obstacles
        self.query = query


def target_trajectory(timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
    """A target on a slow straight run, so the lead sample differs from the
    current one and the two are distinguishable in the output."""
    return (
        np.array([200.0 + 6.0 * timestamp_s, 100.0 + 2.0 * timestamp_s, 140.0], dtype=float),
        np.array([6.0, 2.0, 0.0], dtype=float),
    )


def descending_target(timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
    """A target losing height fast.

    Needed for the follow floor and for nothing else. The floor compares the
    composed altitude against `current_target_z + offset * 0.3`, and the
    composition uses the *lead* sample -- so with a level target the two
    agree and the floor is unreachable at any offset (checked: it never
    fires for anything from -250 m to +250 m). A descending target separates
    the current altitude from the lead one, which is the only way in.
    """
    return (
        np.array(
            [200.0 + 6.0 * timestamp_s, 100.0 + 2.0 * timestamp_s, 400.0 - 30.0 * timestamp_s],
            dtype=float,
        ),
        np.array([6.0, 2.0, -30.0], dtype=float),
    )


def controller_kwargs(**overrides) -> dict:
    base = dict(
        target_trajectory=target_trajectory,
        map_bounds_m=BOUNDS,
        base_agl_m=120.0,
        standoff_radius_m=180.0,
        map_scale=1.0,
        max_speed_mps=22.0,
        phase=0.3,
        lead_s=4.0,
        candidate_count=12,
        rotation_rate_rad_s=0.08,
        planner=None,
        min_agl_m=18.0,
        vertical_amplitude_m=0.0,
        vertical_frequency_rad_s=0.0,
        target_altitude_offset_m=None,
        max_accel_mps2=None,
        terrain_following=False,
        terrain_following_agl_m=30.0,
        terrain_following_smoothing_s=1.5,
        drone_role="interceptor",
        slot_index=0,
        slot_count=1,
        mission_zones=(),
    )
    base.update(overrides)
    return base


def capture_case(
    name: str,
    timestamps: list[float],
    *,
    target_kind: str = "level",
    obstacles: RecordingObstacles | None = None,
    blocked_los: bool = False,
    zones: tuple[MissionZone, ...] = (),
    seed_with: tuple[list[float], list[float], float] | None = None,
    **overrides,
) -> dict:
    terrain = RecordingTerrain()
    obstacles = obstacles or RecordingObstacles()
    query = RecordingQuery(blocked=blocked_los)
    environment = RecordingEnvironment(terrain, obstacles, query)

    kwargs = controller_kwargs(**overrides)
    kwargs["mission_zones"] = zones
    kwargs["target_trajectory"] = (
        descending_target if target_kind == "descending" else target_trajectory
    )
    controller = FollowPathController(environment=environment, **kwargs)
    if seed_with is not None:
        position, velocity, seeded_at = seed_with
        controller.seed(
            position=np.asarray(position, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            timestamp_s=seeded_at,
        )

    samples = []
    for timestamp_s in timestamps:
        sample_position, sample_velocity = controller(timestamp_s)
        samples.append(
            {
                "timestamp_s": exact(timestamp_s),
                "position": vec(sample_position),
                "velocity": vec(sample_velocity),
            }
        )

    return {
        "name": name,
        "config": {
            k: (
                None
                if v is None
                else (exact(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            )
            for k, v in controller_kwargs(**overrides).items()
            if k not in {"target_trajectory", "map_bounds_m", "planner"}
        },
        "has_planner": overrides.get("planner") is not None,
        "target_kind": target_kind,
        "seed_with": (
            None
            if seed_with is None
            else {
                "position": vec(seed_with[0]),
                "velocity": vec(seed_with[1]),
                "timestamp_s": exact(seed_with[2]),
            }
        ),
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "center": vec(z.center),
                "radius_m": exact(z.radius_m),
            }
            for z in zones
        ],
        "obstacle": (
            None
            if obstacles.center_xy is None
            else {
                "center_xy": vec(obstacles.center_xy),
                "radius_m": exact(obstacles.radius_m),
                "top_z_m": exact(obstacles.top_z_m),
            }
        ),
        "samples": samples,
        "terrain_oracle": terrain.oracle(),
        "obstacle_oracle": obstacles.oracle(),
        "obstacle_push_oracle": obstacles.push_oracle(),
        "los_oracle": query.oracle(),
    }


# The handover seed every comparison case shares, so a pair differs only in
# the one thing it is testing. Unseeded cases freeze (see
# `unseeded_never_flies`) and would compare equal for the wrong reason.
HANDOVER = ([500.0, 300.0, 200.0], [15.0, -5.0, 0.0], -1.0)


def capture_cases() -> list[dict]:
    cases = []

    # The plain orbit. Long enough to cross the replan threshold at least
    # once -- see `unseeded_start_holds_until_the_replan_threshold` for why a
    # short span would record a stationary platform.
    #
    # Seeded, because that is the realistic entry: a launched drone hands
    # over to its follow controller mid-flight, and `seed` is how. An
    # unseeded controller does not fly at all (see the case below).
    cases.append(
        capture_case(
            "plain_orbit",
            [float(t) for t in range(0, 9)],
            seed_with=HANDOVER,
        )
    )

    # A FINDING, pinned rather than corrected: an UNSEEDED controller never
    # flies at all.
    #
    # Tick 0 has `delta_t == 0` and no `last_xy`, so it takes the branch that
    # jumps straight to `active_route_points[1]` -- the platform teleports to
    # its first goal -- and collapses the route to `[chosen, chosen]`. From
    # then on `advance_along_polyline` walks a zero-length route, so the
    # position is frozen unless a replan rebuilds it.
    #
    # The replan never comes, and the reason is self-stabilising: the
    # candidate score's second key is `|candidate - current_xy|`, the
    # distance from where the drone IS. A frozen drone therefore keeps
    # choosing the candidate nearest itself, which stays well inside the
    # `max(30, standoff * 0.5)` replan threshold no matter how long the run.
    # Verified over thirteen seconds: one position, thirteen times.
    #
    # The velocity keeps changing throughout, so this does not look like a
    # stuck controller from the outside -- only the position gives it away.
    cases.append(
        capture_case(
            "unseeded_never_flies",
            [float(t) for t in range(0, 13)],
        )
    )

    # The memo: the same timestamp twice returns the cached pair without
    # recomputing. A port that recomputed would produce the same numbers --
    # so the assertion is on the recorded READS, which stay flat.
    cases.append(capture_case("repeated_timestamp_is_memoised", [2.0, 2.0, 2.0]))

    # Time going backwards clears the route and heading state. The reference
    # treats it as a new run rather than interpolating backwards.
    cases.append(capture_case("backwards_time_resets_state", [0.0, 1.0, 2.0, 0.5, 1.5]))

    # `seed` primes last_xy/heading without a sample, which is how a launched
    # drone hands over to its follow controller mid-flight.
    cases.append(
        capture_case(
            "seeded_from_a_handover",
            [10.0, 11.0, 12.0],
            seed_with=([500.0, 300.0, 200.0], [15.0, -5.0, 0.0], 9.0),
        )
    )

    # Acceleration limiting, on and off with everything else equal. The limit
    # applies to the SCALAR speed only, so a turn is not penalised -- the pair
    # is what isolates that.
    # Long enough for the speed to actually vary: saturated at max speed the
    # limit has nothing to limit, and the pair would be identical.
    for name, accel in [("accel_unlimited", None), ("accel_limited", 1.5)]:
        cases.append(
            capture_case(
                name,
                [float(t) for t in range(0, 14)],
                seed_with=([200.0, 100.0, 200.0], [0.0, 0.0, 0.0], -1.0),
                max_accel_mps2=accel,
            )
        )

    # The orbit blend needs the drone near the standoff radius, which the
    # seed puts it at. Compare against a seed far outside, where the blend
    # is below its 0.05 threshold and must not fire.
    cases.append(
        capture_case(
            "orbit_blend_at_the_radius",
            [0.0, 1.0, 2.0, 3.0],
            seed_with=([200.0 + 180.0, 100.0, 200.0], [0.0, 20.0, 0.0], -1.0),
        )
    )
    # Just inside the band but below the 0.05 threshold: the orbit error is
    # 60 m against a 63 m band, so the blend computes to 0.0476 and is
    # discarded. Dropping the threshold entirely would apply it. Added after
    # a mutation run.
    cases.append(
        capture_case(
            "orbit_blend_just_below_the_threshold",
            [0.0, 1.0, 2.0],
            seed_with=([224.0 + 240.0, 108.0, 200.0], [0.0, 20.0, 0.0], -1.0),
        )
    )
    cases.append(
        capture_case(
            "orbit_blend_far_outside_does_not_fire",
            [0.0, 1.0, 2.0, 3.0],
            seed_with=([1200.0, 900.0, 200.0], [0.0, 20.0, 0.0], -1.0),
        )
    )
    # A negative rotation rate flips the tangent's sign, and zero falls back
    # to +1 rather than to no tangent at all.
    cases.append(
        capture_case(
            "orbit_blend_reverses_with_rotation",
            [0.0, 1.0, 2.0],
            seed_with=([200.0 + 180.0, 100.0, 200.0], [0.0, 20.0, 0.0], -1.0),
            rotation_rate_rad_s=-0.08,
        )
    )
    cases.append(
        capture_case(
            "zero_rotation_still_orbits_positively",
            [0.0, 1.0, 2.0],
            seed_with=([200.0 + 180.0, 100.0, 200.0], [0.0, 20.0, 0.0], -1.0),
            rotation_rate_rad_s=0.0,
        )
    )
    # A standoff radius at or below 1.0 disables the blend entirely.
    cases.append(
        capture_case(
            "tiny_standoff_disables_the_blend",
            [0.0, 1.0, 2.0],
            seed_with=([201.0, 100.0, 200.0], [0.0, 20.0, 0.0], -1.0),
            standoff_radius_m=1.0,
        )
    )
    # The gate is `standoff <= 1.0`, and the case above sits far from its
    # tiny radius so the blend would be below threshold anyway. This one
    # parks the platform exactly on a half-metre radius, where the blend
    # would be 1.0 if the gate let it through. Added after a mutation run:
    # moving the gate to `<= 0.0` survived without it.
    cases.append(
        capture_case(
            "sub_metre_standoff_is_gated_not_merely_small",
            [0.0, 1.0],
            seed_with=([224.5, 108.0, 200.0], [0.0, 20.0, 0.0], -1.0),
            standoff_radius_m=0.5,
        )
    )

    # Terrain following: the smoothed height is an EMA whose alpha depends on
    # dt, and the first sample (or any zero-dt sample) takes the raw height.
    cases.append(
        capture_case(
            "terrain_following",
            [0.0, 1.0, 2.0, 3.0, 4.0],
            terrain_following=True,
            terrain_following_agl_m=45.0,
            terrain_following_smoothing_s=1.5,
        )
    )
    # A FINDING: `terrain_following_smoothing_s` does nothing.
    #
    # The reference computes an exponential moving average of the terrain
    # height under the platform, stores it in `_state["smoothed_terrain_z"]`
    # -- and never reads it again except to compute the next average. The
    # altitude on this branch is `terrain_following_agl_m`, flat. So the
    # whole filter is dead code and the constant is an inert knob.
    #
    # These two cases sit at opposite ends of its range (0.01 s and 60 s) and
    # are recorded as IDENTICAL, which is the evidence. Ported anyway, with a
    # property test, rather than deleted: the state is real, a future fix
    # would wire it up, and this slice ports rather than tidies.
    cases.append(
        capture_case(
            "terrain_following_fast_filter",
            [0.0, 1.0, 2.0, 3.0],
            terrain_following=True,
            terrain_following_smoothing_s=0.01,
        )
    )
    cases.append(
        capture_case(
            "terrain_following_slow_filter",
            [0.0, 1.0, 2.0, 3.0],
            terrain_following=True,
            terrain_following_smoothing_s=60.0,
        )
    )

    # Vertical oscillation, which also feeds the altitude-rate term.
    cases.append(
        capture_case(
            "vertical_oscillation",
            [0.0, 1.0, 2.0, 3.0],
            vertical_amplitude_m=25.0,
            vertical_frequency_rad_s=0.6,
        )
    )

    # Target-relative altitude hold, which switches `compose_air_state` to its
    # second branch and adds the follow floor at the end of the tick.
    cases.append(
        capture_case(
            "target_altitude_hold",
            [0.0, 1.0, 2.0, 3.0],
            target_altitude_offset_m=60.0,
        )
    )
    # A deep negative offset drives the position under the follow floor, which
    # is the only case that exercises the floor's velocity correction.
    cases.append(
        capture_case(
            "follow_floor_raises_the_platform",
            [0.0, 1.0, 2.0],
            target_kind="descending",
            target_altitude_offset_m=-50.0,
            seed_with=HANDOVER,
        )
    )

    # An obstacle in the orbit rejects candidates, so the chosen one differs
    # from the unobstructed case.
    cases.append(
        capture_case(
            "obstacle_rejects_candidates",
            [0.0, 1.0, 2.0, 3.0],
            obstacles=RecordingObstacles(center_xy=[371.0, 210.0], radius_m=140.0, top_z_m=500.0),
            seed_with=HANDOVER,
        )
    )

    # Blocked line of sight makes some candidates strictly worse, which is the
    # first key of the candidate score.
    cases.append(
        capture_case(
            "blocked_line_of_sight", [0.0, 1.0, 2.0, 3.0], blocked_los=True, seed_with=HANDOVER
        )
    )

    # Cooperative slots spread drones around the orbit. Two slots of four is
    # enough to show the offset without recording four whole runs.
    for slot in (0, 1):
        cases.append(
            capture_case(
                f"cooperative_slot_{slot}_of_4",
                [0.0, 1.0, 2.0],
                slot_index=slot,
                slot_count=4,
                seed_with=HANDOVER,
            )
        )

    # Exclusion zones: the waypoint pass pushes to radius * 1.1 and the final
    # pass to radius + 1.0. Two different margins, and a case that lands on
    # the zone centre exactly, where the push direction is degenerate.
    zone = MissionZone(
        zone_id="keep-out",
        zone_type="exclusion",
        center=(380.0, 200.0, 0.0),
        radius_m=150.0,
    )
    cases.append(
        capture_case("exclusion_zone", [0.0, 1.0, 2.0, 3.0], zones=(zone,), seed_with=HANDOVER)
    )
    # Two overlapping exclusions. The waypoint pass walks the zones in
    # order: the push out of the first can land the waypoint inside the
    # second, and the push out of the second can land it back inside the
    # first -- which the loop does not revisit. So the waypoint can end up
    # inside a zone, and the FINAL pass is what catches it, at its own
    # margin of `radius + 1.0` rather than `radius * 1.1`.
    #
    # Added after a mutation run: with one zone the final pass never fired,
    # and changing its margin to `+ 0.0` survived.
    overlapping = (
        MissionZone(
            zone_id="keep-out-a",
            zone_type="exclusion",
            center=(380.0, 200.0, 0.0),
            radius_m=150.0,
        ),
        MissionZone(
            zone_id="keep-out-b",
            zone_type="exclusion",
            center=(470.0, 260.0, 0.0),
            radius_m=150.0,
        ),
    )
    cases.append(
        capture_case(
            "overlapping_exclusion_zones",
            [0.0, 1.0, 2.0, 3.0],
            zones=overlapping,
            seed_with=HANDOVER,
        )
    )

    # The FINAL zone pass, which the waypoint pass normally makes
    # unreachable: pass one pushes the waypoint to `radius * 1.1`, and
    # nothing between there and the end moves it back in -- except an
    # obstacle. This places one just outside the zone, on the side the
    # platform approaches from, sized so that being pushed clear of it lands
    # the platform 144 m from the zone centre, inside the 150 m boundary.
    # The final pass then pushes it back out, to its own `radius + 1.0`.
    #
    # So the two passes really do have different margins and both are
    # reachable. Added after a mutation run: with only the waypoint pass
    # firing, changing the final margin to `+ 0.0` survived.
    cases.append(
        capture_case(
            "obstacle_pushes_the_platform_back_into_a_zone",
            [0.0, 1.0],
            zones=(zone,),
            obstacles=RecordingObstacles(center_xy=[530.5, 339.2], radius_m=60.0, top_z_m=600.0),
            seed_with=HANDOVER,
        )
    )

    # A non-exclusion zone of the same geometry must be ignored entirely.
    soft = MissionZone(
        zone_id="watch",
        zone_type="surveillance",
        center=(380.0, 200.0, 0.0),
        radius_m=150.0,
    )
    cases.append(
        capture_case(
            "surveillance_zone_is_not_enforced",
            [0.0, 1.0, 2.0],
            zones=(soft,),
            seed_with=HANDOVER,
        )
    )

    # A candidate count of 1 removes the scan's choice, isolating the rest of
    # the pipeline from the selection.
    cases.append(capture_case("single_candidate", [0.0, 1.0, 2.0], candidate_count=1))

    # A map scale above 1 widens the bounds margin (15 * scale), which is what
    # rejects candidates near the edge.
    #
    # A scale of 8 leaves the margin at 120 m against 2 km bounds, which
    # rejects nothing -- a mutation run caught that, with `15 * map_scale`
    # replaced by a bare `15` surviving. At 120 the margin is 1800 m, the
    # usable box shrinks to +/-200 m, and part of the candidate ring falls
    # outside it.
    cases.append(capture_case("scaled_bounds_margin", [0.0, 1.0, 2.0], map_scale=120.0))

    return cases


def check_coverage(cases: list[dict]) -> None:
    """A fixture that misses a branch blesses a port that dropped it."""
    by_name = {c["name"]: c for c in cases}

    def positions(name: str) -> list[list[str]]:
        return [s["position"] for s in by_name[name]["samples"]]

    def velocities(name: str) -> list[list[str]]:
        return [s["velocity"] for s in by_name[name]["samples"]]

    # The controller actually moves, or every comparison below is vacuous.
    moved = positions("plain_orbit")
    assert moved[0] != moved[-1], "the controller never moved"
    # ... and it moved by REPLANNING, not merely by drifting: the held
    # position has to break at some point in the sequence.
    assert len({tuple(p) for p in moved}) > 2, (
        "the plain orbit visited fewer than three distinct positions, so the "
        "replan path is untested"
    )

    # The unseeded freeze is total, and the velocity keeps moving while the
    # position does not -- both halves of the finding.
    held = positions("unseeded_never_flies")
    assert len({tuple(p) for p in held}) == 1, (
        f"the unseeded freeze no longer holds: {len({tuple(p) for p in held})} distinct positions"
    )
    held_velocities = velocities("unseeded_never_flies")
    assert len({tuple(v) for v in held_velocities}) > 1, (
        "the frozen controller also reports a constant velocity, so the "
        "finding is no longer that the two disagree"
    )

    # The memo returns the same pair every time.
    memo = positions("repeated_timestamp_is_memoised")
    assert memo[0] == memo[1] == memo[2], "the memo returned different results"

    # Acceleration limiting changed the trajectory, and the limit it enforces
    # is on the speed CHANGE -- in both directions. At the tick where it
    # bites here the limited platform is *faster* than the unlimited one,
    # because what was clamped was a sudden deceleration. An assertion that
    # the limited run is never faster would be wrong, and was: it failed
    # against the reference before this comment existed.
    unlimited_speeds = [math.hypot(float(v[0]), float(v[1])) for v in velocities("accel_unlimited")]
    limited_speeds = [math.hypot(float(v[0]), float(v[1])) for v in velocities("accel_limited")]
    assert limited_speeds != unlimited_speeds, "acceleration limiting had no effect"
    paired = zip(limited_speeds, unlimited_speeds, strict=True)
    assert any(limited > unlimited + 1e-9 for limited, unlimited in paired), (
        "the limit never clamped a deceleration, so only half of it is tested"
    )
    # NOT asserted: that every tick's speed change respects accel * dt. It
    # does not, and the reason is an ordering the port has to keep. The limit
    # is applied to `xy_velocity` and the result stored in
    # `last_velocity_xy` -- and THEN the orbit tangent blend rewrites
    # `xy_velocity` without updating that store. So the blend can move the
    # speed past the limit, and the next tick limits against a value the
    # platform never actually flew.
    #
    # Measured here: a 5.75 m/s change on a 1.5 m/s/s limit. A port that
    # applied the limit after the blend, or that stored the blended value,
    # would be more defensible and would disagree with the reference from
    # that tick onward. The recorded samples pin the reference's order.
    assert (
        max(abs(limited_speeds[i] - limited_speeds[i - 1]) for i in range(2, len(limited_speeds)))
        > 1.5
    ), (
        "no tick now exceeds the acceleration limit, so the blend-after-limit "
        "ordering is no longer pinned by this case"
    )

    # The orbit blend fired in one case and not the other.
    near = velocities("orbit_blend_at_the_radius")
    far = velocities("orbit_blend_far_outside_does_not_fire")
    assert near != far, "the orbit blend produced the same result near and far"
    assert velocities("orbit_blend_reverses_with_rotation") != near, (
        "reversing the rotation rate did not change the tangent"
    )

    # The dead smoothing filter: opposite ends of the constant's range give
    # byte-identical output. If this ever starts failing, someone has wired
    # the EMA up and the port has to follow.
    fast = positions("terrain_following_fast_filter")
    slow = positions("terrain_following_slow_filter")
    assert fast == slow, (
        "the smoothing constant now changes the output -- the EMA is no "
        "longer dead code and the port must consume it"
    )
    # Terrain following itself IS observable: it replaces base_agl_m with
    # terrain_following_agl_m.
    assert positions("terrain_following") != positions("plain_orbit")[:5], (
        "terrain following changed nothing at all"
    )

    # Vertical oscillation moved z and gave it a rate.
    heights = [float(p[2]) for p in positions("vertical_oscillation")]
    assert len(set(heights)) > 1, "the vertical oscillation is flat"

    # The follow floor actually raised something.
    floor = positions("follow_floor_raises_the_platform")
    assert floor, "no follow-floor samples"

    # The obstacle was probed and rejected at least one candidate.
    obstacle_case = by_name["obstacle_rejects_candidates"]
    assert any(p["collides"] for p in obstacle_case["obstacle_oracle"]), (
        "the obstacle never collided, so no candidate was rejected"
    )
    assert positions("obstacle_rejects_candidates") != positions("plain_orbit")[:4], (
        "the obstacle did not change the chosen path"
    )

    # Blocked LOS was actually blocked somewhere, and changed the choice.
    los_case = by_name["blocked_line_of_sight"]
    assert any(not r["visible"] for r in los_case["los_oracle"]), (
        "no candidate was ever invisible, so the score's first key is untested"
    )
    assert positions("blocked_line_of_sight") != positions("plain_orbit")[:4], (
        "blocking line of sight did not change the chosen path"
    )

    # Cooperative slots differ from each other.
    assert positions("cooperative_slot_0_of_4") != positions("cooperative_slot_1_of_4"), (
        "two cooperative slots flew the same path"
    )

    # The exclusion zone moved the platform and the soft zone did not.
    assert positions("exclusion_zone") != positions("plain_orbit")[:4], (
        "the exclusion zone was not enforced"
    )
    assert positions("surveillance_zone_is_not_enforced") == positions("plain_orbit")[:3], (
        "a surveillance zone was enforced as if it were an exclusion"
    )

    # Every exclusion sample ends up outside the zone.
    for sample in by_name["exclusion_zone"]["samples"]:
        x, y = float(sample["position"][0]), float(sample["position"][1])
        assert math.hypot(x - 380.0, y - 200.0) >= 150.0 - 1e-6, (
            f"a sample stayed inside the exclusion zone at ({x}, {y})"
        )

    # The follow floor fires, and lands exactly on `target_z + 0.3 * offset`
    # -- which is what separates the 0.3 from any other fraction.
    floor_case = by_name["follow_floor_raises_the_platform"]
    for sample in floor_case["samples"]:
        t = float(sample["timestamp_s"])
        expected = (400.0 - 30.0 * t) + (-50.0 * 0.3)
        assert abs(float(sample["position"][2]) - expected) < 1e-9, (
            f"the follow floor did not bind at t={t}: {sample['position'][2]} vs {expected}"
        )

    # The bounds margin rejected part of the candidate ring, rather than
    # being a number nothing ever compared against.
    assert positions("scaled_bounds_margin") != positions("unseeded_never_flies")[:3], (
        "the scaled bounds margin changed nothing, so 15 * map_scale is untested"
    )

    # The orbit blend threshold: just inside the band, just under 0.05.
    assert (
        positions("orbit_blend_just_below_the_threshold")
        != positions("orbit_blend_at_the_radius")[:3]
    ), "the sub-threshold case blends the same as the on-radius one"

    # The sub-metre standoff is gated, not merely small.
    assert positions("sub_metre_standoff_is_gated_not_merely_small"), "no samples"

    # The final pass fired: the platform sits at exactly `radius + 1.0`,
    # which is where only that pass puts it (the waypoint pass uses 1.1x).
    trapped = by_name["obstacle_pushes_the_platform_back_into_a_zone"]["samples"]
    assert any(
        abs(math.hypot(float(s["position"][0]) - 380.0, float(s["position"][1]) - 200.0) - 151.0)
        < 1e-6
        for s in trapped
    ), (
        "no sample landed at radius + 1, so the final zone pass never fired "
        "and its margin is untested"
    )

    # Overlapping zones leave the platform outside BOTH of them.
    for sample in by_name["overlapping_exclusion_zones"]["samples"]:
        x, y = float(sample["position"][0]), float(sample["position"][1])
        for cx, cy in ((380.0, 200.0), (470.0, 260.0)):
            assert math.hypot(x - cx, y - cy) >= 150.0 - 1e-6, (
                f"a sample stayed inside the zone at ({cx}, {cy})"
            )

    # Backwards time produced a different trajectory than monotonic time.
    assert positions("backwards_time_resets_state") != positions("plain_orbit")[:5], (
        "going backwards in time changed nothing"
    )


def capture() -> dict:
    cases = capture_cases()
    check_coverage(cases)
    return {
        "_source": "scripts/capture_follow_path_fixture.py",
        "_reference": "src/argusnet/simulation/sim.py",
        "bounds": {k: exact(v) for k, v in BOUNDS.items()},
        "target_trajectory": "position = [200 + 6t, 100 + 2t, 140], velocity = [6, 2, 0]",
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fixture = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    samples = sum(len(c["samples"]) for c in fixture["cases"])
    reads = sum(
        len(c["terrain_oracle"]["heights"])
        + len(c["terrain_oracle"]["gradients"])
        + len(c["terrain_oracle"]["clamps"])
        + len(c["obstacle_oracle"])
        + len(c["los_oracle"])
        for c in fixture["cases"]
    )
    print(f"wrote {args.out}")
    print(f"  {len(fixture['cases'])} cases, {samples} samples, {reads} recorded reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
