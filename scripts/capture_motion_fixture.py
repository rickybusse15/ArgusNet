#!/usr/bin/env python3
"""Record the motion substrate stage 1's controllers are built on.

`compose_terrain_state`, `compose_air_state`, `collision_aware_position`,
`advance_along_polyline`, `within_bounds` and `clamp_to_bounds` are the
module-level helpers every controller in `sim.py` calls. The reference notes at
its own line 136 that these are exactly what blocks controller extraction in
Python -- the controllers call helpers defined later in the file, so a
`controllers.py` would import in a circle. Rust has no such constraint, which
is why the substrate moves first and the controller bodies follow.

Values come from calling the reference, not from reimplementing its formulas
here -- a generator that recomputed them would agree with a port that shared
its author's misreading while both disagreed with Python.

WHAT THIS FIXTURE DOES NOT TEST, DELIBERATELY

The terrain and the obstacle geometry are *already* Rust. `TerrainLayer`
samples through the native library, and `CylinderObstacle.push_outside_xy`
is a `call("obstacle", ..., method="push")` into `argusnet-world`. Re-checking
them here would test P4.3 and Phase 3 over again and say nothing about this
slice.

So the terrain and obstacle reads are recorded as an **oracle**: every
`(x, y)` height, every `(x, y, delta)` gradient and every collision probe the
reference made, with what came back. The Rust test replays them through a stub.
What is under test is the composition around those reads -- which is the part
that is new.

This is stronger than driving a real terrain, not weaker: the gradient oracle
is keyed on the delta, and there are **three** different gradient steps in this
codebase (`gradient_at` defaults to `base_resolution * 0.5`, `slope_rad_at`
uses 1.0, `slope_rad_at_many` uses `max(base_resolution * 0.5, 0.25)`). A port
that asks for the wrong one gets a lookup miss and fails, where a real terrain
would have answered it with a plausible number.

Usage:
    python3 scripts/capture_motion_fixture.py --out tests/fixtures/motion.json
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
    advance_along_polyline,
    clamp_to_bounds,
    collision_aware_position,
    compose_air_state,
    compose_terrain_state,
    within_bounds,
)
from argusnet.world.environment import Bounds2D, ObstacleLayer, TerrainLayer  # noqa: E402
from argusnet.world.obstacles import CylinderObstacle  # noqa: E402


def exact(value: float) -> str:
    """Render a float as a string; `serde_json`'s parser is off by an ULP on
    some literals, which broke an exact comparison in the sensor error model."""
    return repr(float(value))


def vec(values) -> list[str]:
    return [exact(v) for v in np.asarray(values, dtype=float).ravel()]


class RecordingTerrain:
    """Wraps a real terrain and records every read the reference makes.

    Delegation, not simulation: the numbers are the native library's. The
    point is the *key* -- recording which `(x, y, delta)` was asked for is what
    pins the gradient step, and recording `clamp_altitude`'s `min_agl_m`
    separately from its result is what stops a port folding the clamp into the
    composition and passing anyway.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.heights: dict[tuple[str, str], str] = {}
        self.gradients: dict[tuple[str, str, str], list[str]] = {}
        self.clamps: dict[tuple[str, str, str, str], str] = {}

    @property
    def base_resolution_m(self) -> float:
        return self._inner.base_resolution_m

    def height_at(self, x_m: float, y_m: float) -> float:
        value = float(self._inner.height_at(x_m, y_m))
        self.heights[(exact(x_m), exact(y_m))] = exact(value)
        return value

    def gradient_at(self, x_m: float, y_m: float, delta_m: float | None = None):
        gradient = np.asarray(self._inner.gradient_at(x_m, y_m, delta_m), dtype=float)
        # `None` is recorded as the literal string "default" rather than
        # resolved here. Resolving it would bake this script's reading of the
        # default into the fixture, and the default is one of the three steps
        # the port could plausibly get wrong.
        key_delta = "default" if delta_m is None else exact(delta_m)
        self.gradients[(exact(x_m), exact(y_m), key_delta)] = vec(gradient)
        return gradient

    def clamp_altitude(self, xy_m, z_m: float, min_agl_m: float) -> float:
        value = float(self._inner.clamp_altitude(xy_m, z_m, min_agl_m))
        xy = np.asarray(xy_m, dtype=float)
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
    """Records every collision probe and every push the reference performed.

    `collision_aware_position` loops up to eight times, and the loop is the new
    code. Recording the probes in *order* is what pins the iteration: a port
    that pushed once and returned would replay fewer probes than were recorded.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.probes: list[dict] = []

    def point_collides(self, x_m: float, y_m: float, z_m: float):
        hit = self._inner.point_collides(x_m, y_m, z_m)
        record = {
            "point": [exact(x_m), exact(y_m), exact(z_m)],
            "collides": hit is not None,
        }
        if hit is not None:
            pushed = np.asarray(hit.push_outside_xy(x_m, y_m, margin_m=1.0), dtype=float)
            record["pushed_xy"] = vec(pushed)
            # Wrap so the port cannot get the margin wrong unnoticed: the
            # recorded push is the one taken at margin 1.0 and no other.
            record["margin_m"] = exact(1.0)
            hit = _FixedPush(pushed)
        self.probes.append(record)
        return hit


class _FixedPush:
    def __init__(self, pushed) -> None:
        self._pushed = pushed

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0):
        return self._pushed


def build_terrain() -> TerrainLayer:
    """A small analytic terrain with real relief.

    Flat ground would make every gradient zero, and a zero gradient hides the
    `vz` coupling in `compose_terrain_state` completely -- the dot product
    vanishes and a port that dropped the term entirely would pass.
    """
    bounds = Bounds2D(x_min_m=-500.0, x_max_m=500.0, y_min_m=-500.0, y_max_m=500.0)
    xs = np.arange(-500.0, 501.0, 25.0)
    ys = np.arange(-500.0, 501.0, 25.0)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    # A tilted plane plus a bump: the plane gives a constant non-zero gradient
    # everywhere, the bump gives somewhere the gradient reverses sign.
    heights = (
        40.0
        + 0.08 * grid_x
        + 0.05 * grid_y
        + 60.0 * np.exp(-((grid_x - 100.0) ** 2 + (grid_y + 50.0) ** 2) / (2.0 * 120.0**2))
    )
    return TerrainLayer.from_height_grid(
        environment_id="motion-fixture",
        bounds_xy_m=bounds,
        heights_m=heights,
        resolution_m=25.0,
    )


def capture_compose_terrain(terrain) -> list[dict]:
    """Every branch of `compose_terrain_state`.

    The interesting one is the descent guard. `vz` is
    `dot(gradient, xy_velocity) + agl_rate`, and it is forced to zero only when
    the terrain clamp *raised* the altitude AND the composed vz is negative --
    a drone already being pushed up by the ground must not also be descending.
    Both halves of that conjunction need a case, or a port that dropped either
    condition passes.
    """
    cases = []

    def case(name: str, **kwargs) -> None:
        position, velocity = compose_terrain_state(**kwargs)
        cases.append(
            {
                "name": name,
                "xy_m": vec(kwargs["xy_m"]),
                "xy_velocity_mps": vec(kwargs["xy_velocity_mps"]),
                "agl_m": exact(kwargs["agl_m"]),
                "agl_rate_mps": exact(kwargs["agl_rate_mps"]),
                "min_agl_m": exact(kwargs["min_agl_m"]),
                "terrain_speed_adaptation": bool(kwargs.get("terrain_speed_adaptation", False)),
                "position": vec(position),
                "velocity": vec(velocity),
            }
        )

    base = {"terrain": terrain, "min_agl_m": 20.0}

    # Climbing over rising ground: gradient dotted with velocity is positive,
    # so vz exceeds agl_rate and the guard does not fire.
    case(
        "uphill_climb",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[12.0, 0.0],
        agl_m=60.0,
        agl_rate_mps=1.5,
        **base,
    )
    # Flying down the slope with a descent rate: vz is negative and the clamp
    # is NOT active, so the guard must leave it alone.
    case(
        "downhill_descent_unclamped",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[-12.0, 0.0],
        agl_m=60.0,
        agl_rate_mps=-2.0,
        **base,
    )
    # agl below min_agl: the clamp raises z above nominal. With a negative vz
    # this is the conjunction -- both halves true -- and vz must become 0.
    case(
        "clamped_and_descending",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[-12.0, 0.0],
        agl_m=1.0,
        agl_rate_mps=-2.0,
        **base,
    )
    # Clamped but ASCENDING: clamp active, vz positive, guard must not fire.
    case(
        "clamped_and_ascending",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[12.0, 0.0],
        agl_m=1.0,
        agl_rate_mps=2.0,
        **base,
    )
    # Speed adaptation off vs on at the same point, so the pair isolates the
    # steepness factor from everything else.
    case(
        "adaptation_off_on_slope",
        xy_m=[100.0, -50.0],
        xy_velocity_mps=[15.0, 15.0],
        agl_m=80.0,
        agl_rate_mps=0.0,
        terrain_speed_adaptation=False,
        **base,
    )
    case(
        "adaptation_on_slope",
        xy_m=[100.0, -50.0],
        xy_velocity_mps=[15.0, 15.0],
        agl_m=80.0,
        agl_rate_mps=0.0,
        terrain_speed_adaptation=True,
        **base,
    )
    # On the bump's flank the gradient magnitude is largest; adaptation should
    # bite hardest here. Near the 0.7 floor if the slope is steep enough.
    case(
        "adaptation_on_steep_flank",
        xy_m=[220.0, -50.0],
        xy_velocity_mps=[20.0, 0.0],
        agl_m=80.0,
        agl_rate_mps=0.0,
        terrain_speed_adaptation=True,
        **base,
    )
    # Zero horizontal velocity: the dot product is zero and vz is purely the
    # agl rate. Separates the two contributions.
    case(
        "stationary_pure_agl_rate",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[0.0, 0.0],
        agl_m=60.0,
        agl_rate_mps=3.0,
        **base,
    )
    # min_agl of zero: the clamp floor collapses to the ground itself.
    case(
        "zero_min_agl",
        xy_m=[-200.0, 200.0],
        xy_velocity_mps=[5.0, -5.0],
        agl_m=30.0,
        agl_rate_mps=0.0,
        terrain=terrain,
        min_agl_m=0.0,
    )
    return cases


def capture_compose_air(terrain) -> list[dict]:
    """`compose_air_state`'s two paths, and the `max` inside the second.

    The delegating path (no target) is not a trivial passthrough: it forwards
    `nominal_agl_m` as `agl_m` and drops `terrain_speed_adaptation` entirely,
    so a target-following drone never gets speed adaptation. That is the
    reference's behaviour and the port has to keep it.
    """
    cases = []

    def case(name: str, **kwargs) -> None:
        position, velocity = compose_air_state(**kwargs)
        target_position = kwargs.get("target_position")
        target_velocity = kwargs.get("target_velocity")
        offset = kwargs.get("target_altitude_offset_m")
        rate_offset = kwargs.get("target_altitude_rate_offset_mps")
        cases.append(
            {
                "name": name,
                "xy_m": vec(kwargs["xy_m"]),
                "xy_velocity_mps": vec(kwargs["xy_velocity_mps"]),
                "min_agl_m": exact(kwargs["min_agl_m"]),
                "nominal_agl_m": exact(kwargs["nominal_agl_m"]),
                "agl_rate_mps": exact(kwargs["agl_rate_mps"]),
                "target_position": None if target_position is None else vec(target_position),
                "target_velocity": None if target_velocity is None else vec(target_velocity),
                "target_altitude_offset_m": None if offset is None else exact(offset),
                "target_altitude_rate_offset_mps": (
                    None if rate_offset is None else exact(rate_offset)
                ),
                "position": vec(position),
                "velocity": vec(velocity),
            }
        )

    base = {
        "terrain": terrain,
        "min_agl_m": 20.0,
        "nominal_agl_m": 70.0,
        "agl_rate_mps": 0.5,
    }

    # No target at all -- delegates.
    case("no_target_delegates", xy_m=[0.0, 0.0], xy_velocity_mps=[10.0, 4.0], **base)
    # A target position but NO offset: the guard is `or`, so this delegates too.
    # A port that tested only `target_position is None` would take the other
    # branch here and read `target_altitude_offset_m` as None.
    case(
        "target_without_offset_delegates",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_position=np.array([50.0, 50.0, 200.0]),
        **base,
    )
    # An offset but no position: the other half of the same `or`.
    case(
        "offset_without_target_delegates",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_altitude_offset_m=30.0,
        **base,
    )
    # Target high above terrain: the target term wins the max.
    case(
        "target_term_wins_max",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_position=np.array([50.0, 50.0, 400.0]),
        target_velocity=np.array([1.0, 2.0, 3.0]),
        target_altitude_offset_m=40.0,
        target_altitude_rate_offset_mps=0.25,
        **base,
    )
    # Target underground: the terrain floor wins the max instead.
    case(
        "terrain_floor_wins_max",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_position=np.array([50.0, 50.0, -100.0]),
        target_velocity=np.array([1.0, 2.0, -3.0]),
        target_altitude_offset_m=5.0,
        target_altitude_rate_offset_mps=0.0,
        **base,
    )
    # Target given, velocity omitted: vz is the rate offset alone. Separates
    # the target's own vz from the offset.
    case(
        "target_velocity_defaults_to_zero",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_position=np.array([50.0, 50.0, 400.0]),
        target_altitude_offset_m=40.0,
        target_altitude_rate_offset_mps=1.5,
        **base,
    )
    # Rate offset omitted entirely: the `or 0.0` at the end. Note it is `or`,
    # not a None check, so a rate offset of 0.0 and of None are the same --
    # which is fine here only because the fallback is also 0.0.
    case(
        "rate_offset_none_is_zero",
        xy_m=[0.0, 0.0],
        xy_velocity_mps=[10.0, 4.0],
        target_position=np.array([50.0, 50.0, 400.0]),
        target_velocity=np.array([1.0, 2.0, 3.0]),
        target_altitude_offset_m=40.0,
        **base,
    )
    # The target branch does NOT apply the terrain clamp -- it takes a plain
    # max against `ground + min_agl`. Same thing here, but it reaches it by a
    # different route, and this case is the one that would expose a port that
    # called clamp_altitude instead (the clamp would be recorded, and no
    # clamp is recorded for this case).
    case(
        "target_branch_uses_max_not_clamp",
        xy_m=[-300.0, 300.0],
        xy_velocity_mps=[0.0, 0.0],
        target_position=np.array([0.0, 0.0, 10.0]),
        target_altitude_offset_m=0.0,
        **base,
    )
    return cases


def capture_collision(terrain) -> list[dict]:
    """`collision_aware_position`: the loop, not the push.

    The push is `argusnet-world`'s already. What is new is: return early with
    a *copy* when there is no layer at all, iterate at most eight times,
    re-clamp altitude after every push, and return best-effort rather than
    raising when it never converges.
    """
    cases = []

    def case(name: str, position, velocity, obstacles) -> None:
        layer = None if obstacles is None else RecordingObstacles(obstacles)
        result_position, result_velocity = collision_aware_position(
            np.asarray(position, dtype=float),
            np.asarray(velocity, dtype=float),
            terrain=terrain,
            min_agl_m=20.0,
            obstacle_layer=layer,
        )
        cases.append(
            {
                "name": name,
                "position": vec(position),
                "velocity": vec(velocity),
                "min_agl_m": exact(20.0),
                "has_obstacle_layer": obstacles is not None,
                "probes": [] if layer is None else layer.probes,
                "result_position": vec(result_position),
                "result_velocity": vec(result_velocity),
            }
        )

    layer_bounds = Bounds2D(x_min_m=-500.0, x_max_m=500.0, y_min_m=-500.0, y_max_m=500.0)
    empty = ObstacleLayer.empty(bounds_xy_m=layer_bounds, tile_size_m=250.0)

    def cylinder(primitive_id: str, blocker_type: str) -> CylinderObstacle:
        return CylinderObstacle(
            primitive_id=primitive_id,
            blocker_type=blocker_type,
            center_x_m=0.0,
            center_y_m=0.0,
            radius_m=60.0,
            base_z_m=0.0,
            top_z_m=300.0,
        )

    def layer(*primitives) -> ObstacleLayer:
        return ObstacleLayer(bounds_xy_m=layer_bounds, tile_size_m=250.0, primitives=primitives)

    tower = layer(cylinder("tower", "building"))
    # `point_collides` filters on `blocker_type not in {"building", "wall"}`.
    # A forest stand occupies exactly the same space and is NOT a collision.
    # Found the hard way: this fixture's first draft used "structure" and
    # recorded a tower that nothing ever hit, and the coverage check below is
    # the only reason that did not become the port's specification.
    forest = layer(cylinder("stand", "vegetation"))

    # No layer: returns the inputs untouched, and -- the detail worth pinning --
    # returns `velocity` itself on this path while every other path returns
    # `velocity.copy()`. Recorded so a port that copies everywhere is at least
    # a deliberate divergence rather than an accident.
    case("no_layer_returns_inputs", [10.0, 10.0, 150.0], [5.0, 0.0, 0.0], None)
    # A layer with nothing in it still costs one probe.
    case("empty_layer_probes_once", [10.0, 10.0, 150.0], [5.0, 0.0, 0.0], empty)
    # Clear of the tower: one probe, no push.
    case("clear_of_obstacle", [200.0, 200.0, 150.0], [5.0, 0.0, 0.0], tower)
    # Inside the tower: pushed out, then a second probe confirms clear.
    case("inside_obstacle_pushed_out", [10.0, 10.0, 150.0], [5.0, 1.0, 0.0], tower)
    # Dead centre. The push direction is degenerate here -- there is no
    # "outward" at the axis -- and whatever the native code picks, the port
    # must pick the same thing, which is why the pushed point is recorded.
    case("dead_centre_push", [0.0, 0.0, 150.0], [0.0, 0.0, 0.0], tower)
    # Low over the tower's foot. The push moves the drone onto higher ground
    # and the re-clamp RAISES it; in every case above the drone is high
    # enough that the clamp is a no-op, so without this one a port that
    # pushed in xy and never revisited z passes -- and CLAUDE.md's "physical
    # collision must never push entities below terrain" goes untested.
    # Added after a mutation run: `collision drops re-clamp` survived.
    case("push_over_higher_ground_reclamps", [10.0, 10.0, 30.0], [5.0, 1.0, 0.0], tower)
    # Above the tower's top: not a collision at all, despite the xy overlap.
    case("above_obstacle_top", [0.0, 0.0, 400.0], [5.0, 0.0, 0.0], tower)
    # Same geometry, non-blocking type: the blocker_type filter, not the
    # geometry, is what makes this a miss.
    case("vegetation_is_not_a_collision", [10.0, 10.0, 150.0], [5.0, 1.0, 0.0], forest)
    return cases


def capture_polyline() -> list[dict]:
    """`advance_along_polyline`, including the shapes that look like bugs.

    Three behaviours a tidy-up would break:

    * The returned polyline is `vstack([new_xy, rest])` -- the *cursor becomes
      the first vertex*, so the route shortens as it is walked and the caller
      can feed the result straight back in.
    * Running off the end returns `[last, last]` -- a two-point degenerate
      polyline, not a one-point one -- because the caller's next call would
      raise on `n < 2`.
    * The direction returned when the walk terminates early is
      `last_direction`, which is the zero vector if every segment was
      degenerate. Not the final heading.
    """
    cases = []

    def case(name: str, points, distance: float) -> None:
        polyline = np.asarray(points, dtype=float)
        remaining, new_xy, direction = advance_along_polyline(polyline, distance)
        cases.append(
            {
                "name": name,
                "points_xy": [vec(p) for p in polyline],
                "travel_distance_m": exact(distance),
                "remaining_xy": [vec(p) for p in np.asarray(remaining, dtype=float)],
                "new_xy": vec(new_xy),
                "direction": vec(direction),
            }
        )

    line = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]

    case("zero_distance_stays_put", line, 0.0)
    # Negative distance is floored to zero, not walked backwards.
    case("negative_distance_is_floored", line, -50.0)
    case("partway_along_first_segment", line, 40.0)
    # Exactly on a vertex. `remaining <= segment_length` is inclusive, so this
    # resolves on the FIRST segment and the direction is that segment's, not
    # the next one's. A port using `<` would report the second segment's
    # direction and be plausible.
    case("exactly_on_vertex", line, 100.0)
    case("into_second_segment", line, 150.0)
    # Exactly at the end of the last segment -- still resolves inside the loop.
    case("exactly_at_end", line, 200.0)
    # Past the end: the degenerate two-point return.
    case("past_the_end", line, 500.0)
    # A zero-length segment in the middle is skipped without consuming
    # distance and without setting `last_direction`.
    case("degenerate_segment_skipped", [[0.0, 0.0], [0.0, 0.0], [60.0, 0.0]], 30.0)
    # Every segment degenerate and the walk runs off the end: direction is the
    # zero vector, which is the `last_direction` initialiser showing through.
    case("all_degenerate_zero_direction", [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], 10.0)
    # Two points only -- the minimum the validator accepts.
    case("two_point_polyline", [[0.0, 0.0], [30.0, 40.0]], 25.0)
    return cases


def capture_bounds() -> dict:
    """`within_bounds` and `clamp_to_bounds`, including the collapse.

    `clamp_to_bounds` has an explicit guard for a margin wider than half the
    span: it collapses to the midpoint rather than letting `np.clip` see an
    inverted range. The comment says np.clip "handles [it] inconsistently",
    so the midpoint is the defined behaviour and the port needs it.
    """
    # Deliberately asymmetric: the x span is 600 m and the y span 200 m, so a
    # margin can collapse one axis while the other still clamps normally. The
    # midpoints differ too (200 vs 100) -- with equal midpoints a port that
    # collapsed both axes together would produce the same numbers as one that
    # collapsed the right one, and the case would prove nothing.
    bounds = {"x_min_m": -100.0, "x_max_m": 500.0, "y_min_m": 0.0, "y_max_m": 200.0}

    within_cases = []
    for name, xy, margin in [
        ("inside", [50.0, 100.0], 0.0),
        ("on_min_edge_inclusive", [-100.0, 0.0], 0.0),
        ("on_max_edge_inclusive", [500.0, 200.0], 0.0),
        ("outside_x_low", [-101.0, 100.0], 0.0),
        ("outside_y_high", [50.0, 201.0], 0.0),
        ("inside_but_within_margin", [-90.0, 100.0], 25.0),
        ("on_margin_edge_inclusive", [-75.0, 25.0], 25.0),
        # One case per edge, each failing on THAT edge alone. Without these
        # the x-min test decides every rejection and a port that dropped the
        # margin from the other three comparisons passes. Added after a
        # mutation run: `contains ignores margin` survived.
        ("outside_x_max_margin_only", [490.0, 100.0], 25.0),
        ("outside_y_min_margin_only", [50.0, 10.0], 25.0),
        ("outside_y_max_margin_only", [50.0, 190.0], 25.0),
        ("margin_wider_than_span_rejects_all", [100.0, 100.0], 500.0),
    ]:
        within_cases.append(
            {
                "name": name,
                "xy_m": vec(xy),
                "margin_m": exact(margin),
                "within": bool(within_bounds(xy, bounds, margin)),
            }
        )

    clamp_cases = []
    for name, xy, margin in [
        ("inside_unchanged", [50.0, 100.0], 0.0),
        ("clamps_x_low", [-500.0, 100.0], 0.0),
        ("clamps_both_high", [900.0, 900.0], 0.0),
        ("clamps_with_margin", [-500.0, 900.0], 30.0),
        # Margin wider than half the y span but not the x span: y collapses to
        # its midpoint while x still clamps normally. A port that collapsed
        # both axes together would pass a symmetric case and fail this one.
        ("y_collapses_x_clamps", [900.0, 900.0], 150.0),
        ("both_collapse", [900.0, 900.0], 500.0),
    ]:
        clamp_cases.append(
            {
                "name": name,
                "xy_m": vec(xy),
                "margin_m": exact(margin),
                "clamped_xy_m": vec(clamp_to_bounds(xy, bounds, margin)),
            }
        )

    return {
        "bounds": {k: exact(v) for k, v in bounds.items()},
        "within": within_cases,
        "clamp": clamp_cases,
    }


def check_coverage(fixture: dict) -> None:
    """A fixture that misses a branch is a fixture that blesses a port which
    dropped it. These assert the cases above actually reach what they claim."""
    terrain_cases = {c["name"]: c for c in fixture["compose_terrain_state"]}

    # The descent guard fired exactly where it should and nowhere else.
    clamped = terrain_cases["clamped_and_descending"]
    assert float(clamped["velocity"][2]) == 0.0, "the descent guard did not fire"
    descending = terrain_cases["downhill_descent_unclamped"]
    assert float(descending["velocity"][2]) < 0.0, "unclamped descent was zeroed anyway"
    ascending = terrain_cases["clamped_and_ascending"]
    assert float(ascending["velocity"][2]) > 0.0, "a clamped ascent was zeroed"

    # Speed adaptation actually changed the speed, and by less than the floor.
    off = np.array([float(v) for v in terrain_cases["adaptation_off_on_slope"]["velocity"][:2]])
    on = np.array([float(v) for v in terrain_cases["adaptation_on_slope"]["velocity"][:2]])
    ratio = float(np.linalg.norm(on) / np.linalg.norm(off))
    assert ratio < 1.0, "terrain_speed_adaptation had no effect -- is the terrain flat?"
    assert ratio >= 0.7 - 1e-12, f"steepness factor {ratio} fell below the 0.7 floor"

    # A non-zero gradient somewhere, or the vz coupling is untested.
    magnitudes = [
        float(np.hypot(float(g["gradient"][0]), float(g["gradient"][1])))
        for g in fixture["terrain_oracle"]["gradients"]
    ]
    assert max(magnitudes) > 0.05, "every recorded gradient is ~flat; vz coupling untested"

    air_cases = {c["name"]: c for c in fixture["compose_air_state"]}
    # Both sides of the max, or the max is untested.
    target_wins = air_cases["target_term_wins_max"]
    floor_wins = air_cases["terrain_floor_wins_max"]
    assert float(target_wins["position"][2]) > 300.0, "the target term did not win the max"
    assert float(floor_wins["position"][2]) > 0.0, "the terrain floor did not win the max"

    collision_cases = {c["name"]: c for c in fixture["collision_aware_position"]}
    pushed = collision_cases["inside_obstacle_pushed_out"]
    assert len(pushed["probes"]) >= 2, "the push loop did not iterate"
    assert pushed["probes"][0]["collides"], "the first probe did not collide"
    assert not pushed["probes"][-1]["collides"], "the loop exited while still colliding"
    assert collision_cases["no_layer_returns_inputs"]["probes"] == [], (
        "the no-layer path probed the obstacle layer"
    )
    assert len(collision_cases["above_obstacle_top"]["probes"]) == 1
    assert not collision_cases["above_obstacle_top"]["probes"][0]["collides"], (
        "a point above the tower's top was treated as a collision"
    )
    # The same point, same geometry, different blocker_type. If this ever
    # collides, the filter has changed and the port needs to change with it.
    # The re-clamp did real work somewhere: the push landed the drone on
    # ground high enough that its altitude had to rise.
    reclamped = collision_cases["push_over_higher_ground_reclamps"]
    assert float(reclamped["result_position"][2]) > float(reclamped["position"][2]), (
        "the re-clamp never raised an altitude; a port that skipped it would pass"
    )

    vegetation = collision_cases["vegetation_is_not_a_collision"]
    assert not vegetation["probes"][0]["collides"], (
        "vegetation collided; the blocker_type filter is no longer {building, wall}"
    )
    assert vegetation["result_position"] == pushed["position"], (
        "the vegetation case did not start where the building case did, so the "
        "pair no longer isolates blocker_type"
    )

    polyline_cases = {c["name"]: c for c in fixture["advance_along_polyline"]}
    # The inclusive boundary: at exactly 100 m the direction is the FIRST
    # segment's (+x), not the second's (+y).
    vertex = polyline_cases["exactly_on_vertex"]
    assert vertex["direction"] == ["1.0", "0.0"], (
        f"exactly_on_vertex resolved to the wrong segment: {vertex['direction']}"
    )
    # Past the end returns two points, not one.
    assert len(polyline_cases["past_the_end"]["remaining_xy"]) == 2
    assert polyline_cases["all_degenerate_zero_direction"]["direction"] == ["0.0", "0.0"]

    # Each edge rejects on its own. Four cases, four different comparisons --
    # so no single one of them carries every rejection.
    bounds_within = {c["name"]: c for c in fixture["bounds"]["within"]}
    for edge in (
        "inside_but_within_margin",
        "outside_x_max_margin_only",
        "outside_y_min_margin_only",
        "outside_y_max_margin_only",
    ):
        assert not bounds_within[edge]["within"], f"{edge} no longer rejects"

    bounds_clamp = {c["name"]: c for c in fixture["bounds"]["clamp"]}
    collapsed = bounds_clamp["y_collapses_x_clamps"]
    assert collapsed["clamped_xy_m"][1] == "100.0", (
        f"y did not collapse to its midpoint: {collapsed['clamped_xy_m']}"
    )
    assert collapsed["clamped_xy_m"][0] == "350.0", (
        f"x collapsed when it should have clamped to 350: {collapsed['clamped_xy_m']}"
    )
    # Both axes collapse, to their own midpoints, which are not the same number.
    both = bounds_clamp["both_collapse"]["clamped_xy_m"]
    assert both == ["200.0", "100.0"], f"collapse midpoints wrong: {both}"


def capture() -> dict:
    terrain = RecordingTerrain(build_terrain())
    fixture = {
        "_source": "scripts/capture_motion_fixture.py",
        "_reference": "src/argusnet/simulation/sim.py",
        "compose_terrain_state": capture_compose_terrain(terrain),
        "compose_air_state": capture_compose_air(terrain),
        "collision_aware_position": capture_collision(terrain),
        "advance_along_polyline": capture_polyline(),
        "bounds": capture_bounds(),
    }
    # Recorded last: the oracle has to contain every read the cases above made.
    fixture["terrain_oracle"] = terrain.oracle()
    check_coverage(fixture)
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fixture = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    oracle = fixture["terrain_oracle"]
    print(f"wrote {args.out}")
    print(
        f"  {len(fixture['compose_terrain_state'])} compose_terrain_state, "
        f"{len(fixture['compose_air_state'])} compose_air_state, "
        f"{len(fixture['collision_aware_position'])} collision, "
        f"{len(fixture['advance_along_polyline'])} polyline, "
        f"{len(fixture['bounds']['within'])}+{len(fixture['bounds']['clamp'])} bounds"
    )
    print(
        f"  oracle: {len(oracle['heights'])} heights, {len(oracle['gradients'])} gradients, "
        f"{len(oracle['clamps'])} clamps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
