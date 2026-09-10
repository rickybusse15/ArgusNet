#!/usr/bin/env python3
"""Record PathPlanner2D's public routing surface: nearest_free_point,
plan_route, route_waypoints, and the route cache.

Cases are TAGGED, because one behaviour deliberately diverges:

  agreement  -- the port must reproduce these values.
  divergence -- the port must NOT. The recording proves the reference really
                behaves as described, so the change is demonstrable rather
                than asserted, and a regression to the old behaviour fails.

The divergence is the route cache key. `docs/migration/PLANNING_INVENTORY.md`
establishes that the reference's key is quantized start, quantized goal and
clearance -- with no obstacle or scene identity of any kind -- and that
`clear_cache()` has no caller anywhere in `src/` or `tests/`. The counter it
needs already exists: `Environment.add_obstacle()` bumps `scene_version`, and
`EnvironmentQuery.los` already keys its own cache on it. The port keys on the
revision; the reference cannot, so the fixture records what the reference does
and the Rust test asserts the new contract.

Usage:
    python3 scripts/capture_plan_route_fixture.py --out tests/fixtures/plan_route.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.planning.planner_base import PathPlanner2D, PlannerConfig  # noqa: E402
from argusnet.world.environment import Bounds2D, ObstacleLayer  # noqa: E402
from argusnet.world.obstacles import (  # noqa: E402
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    OrientedBox,
    WallSegment,
)


def exact(value: float) -> str:
    """`serde_json`'s float parser is an ULP off `str::parse::<f64>` on some
    literals, which broke an exact comparison in the sensor error model."""
    return repr(float(value))


def pair(xy) -> list[str]:
    return [exact(xy[0]), exact(xy[1])]


def polygon(points) -> list[list[str]]:
    return [pair(p) for p in np.asarray(points, dtype=float)]


BOUNDS = Bounds2D(x_min_m=-500.0, x_max_m=500.0, y_min_m=-500.0, y_max_m=500.0)


def building(pid: str, cx: float, cy: float, half: float) -> BuildingPrism:
    footprint = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=float,
    )
    return BuildingPrism(primitive_id=pid, footprint_xy_m=footprint, base_z_m=0.0, top_z_m=40.0)


def wall(pid: str, x0: float, y0: float, x1: float, y1: float) -> WallSegment:
    return WallSegment(
        primitive_id=pid,
        blocker_type="wall",
        start_xy_m=np.array([x0, y0], dtype=float),
        end_xy_m=np.array([x1, y1], dtype=float),
        thickness_m=4.0,
        base_z_m=0.0,
        top_z_m=20.0,
    )


def box(pid: str, cx: float, cy: float, yaw: float) -> OrientedBox:
    return OrientedBox(
        primitive_id=pid,
        blocker_type="building",
        center_x_m=cx,
        center_y_m=cy,
        length_m=90.0,
        width_m=30.0,
        yaw_rad=yaw,
        base_z_m=0.0,
        top_z_m=25.0,
    )


def cylinder(pid: str, cx: float, cy: float, radius: float) -> CylinderObstacle:
    return CylinderObstacle(
        primitive_id=pid,
        blocker_type="building",
        center_x_m=cx,
        center_y_m=cy,
        radius_m=radius,
        base_z_m=0.0,
        top_z_m=30.0,
    )


SCENES = {
    "empty": [],
    # Sits on the x=-150 edge, so it blocks the CLOSING segment of the loop
    # below and nothing else. Without it, `closed` is unobservable: with every
    # segment free the point list is identical either way, and smoothing
    # collapses a four-corner loop to two points regardless.
    "edge_blocker": [building("b1", -150.0, 0.0, 40.0)],
    "one_building": [building("b1", 0.0, 0.0, 40.0)],
    # An oriented box and a wall reach the same expansion path as a polygon in
    # the reference, but arrive as different Python classes. The Rust world
    # crate converts both to polygons at construction, so this checks the two
    # agree rather than assuming Phase 3's conversion was faithful.
    "box_and_wall": [box("x1", -60.0, 40.0, 0.6), wall("w1", 40.0, -150.0, 40.0, 150.0)],
    # A cylinder is the one shape that is NOT a stored footprint: the reference
    # samples `cylinder_vertex_count` points around it at expansion time.
    "cylinder": [cylinder("c1", 0.0, 0.0, 45.0)],
    "cluster": [
        building("b1", -70.0, 0.0, 30.0),
        building("b2", 0.0, 0.0, 30.0),
        building("b3", 70.0, 0.0, 30.0),
    ],
    # Vegetation is NOT a hard blocker: routes fly through it and only
    # visibility attenuates. Without a scene containing some, a port that
    # treated every obstacle as solid would pass everything.
    "forest": [
        ForestStand(
            primitive_id="f1",
            footprint_xy_m=np.array(
                [[-60.0, -60.0], [60.0, -60.0], [60.0, 60.0], [-60.0, 60.0]], dtype=float
            ),
            canopy_base_z_m=0.0,
            canopy_top_z_m=25.0,
        )
    ],
    # A building hard against the western boundary, so pushing a point out of
    # it drives it past the bounds margin and the re-clamp matters.
    "corner_block": [building("b1", -470.0, 0.0, 40.0)],
    # Tall and wide, straddling the western boundary. The nearest way out of
    # it is WEST, which is outside the bounds margin -- so the clamp after
    # each push drags the point straight back inside, the loop makes no
    # progress, and the ten-iteration cap is what ends it. Both the re-clamp
    # and the cap are unobservable without a shape like this.
    "boundary_slab": [building("b1", -480.0, 0.0, 200.0)],
    # Two concentric-ish rings: a point in the middle needs several push
    # iterations to escape, so the ten-iteration cap is reachable.
    "nested": [
        building("b1", 0.0, 0.0, 30.0),
        building("b2", 0.0, 0.0, 60.0),
        building("b3", 0.0, 0.0, 90.0),
        building("b4", 0.0, 0.0, 120.0),
    ],
}


def obstacles_of(name: str):
    return SCENES[name]


def make_planner(scene: str, config: PlannerConfig | None = None) -> PathPlanner2D:
    layer = ObstacleLayer(bounds_xy_m=BOUNDS, tile_size_m=100.0, primitives=obstacles_of(scene))
    return PathPlanner2D(bounds_xy_m=BOUNDS, obstacle_layer=layer, config=config or PlannerConfig())


def scene_metadata(scene: str, clearance: float) -> dict:
    """The expanded polygons the reference builds for this scene, in the order
    its own candidate query returns them. Recorded as INPUT so a disagreement
    is localised to routing rather than to footprint construction."""
    planner = make_planner(scene)
    return {
        "name": scene,
        "clearance_m": exact(clearance),
        # The scene as `to_metadata()` emits it. Rust rebuilds it through
        # `ObstacleMetadata::into_obstacle`, the same conversion the native
        # bridge uses, so the footprint construction for oriented boxes and
        # walls is exercised rather than assumed.
        "obstacles": [p.to_metadata() for p in planner.obstacle_layer.primitives],
        "tile_size_m": exact(planner.obstacle_layer.tile_size_m),
        "bounds": {
            "x_min_m": exact(BOUNDS.x_min_m),
            "x_max_m": exact(BOUNDS.x_max_m),
            "y_min_m": exact(BOUNDS.y_min_m),
            "y_max_m": exact(BOUNDS.y_max_m),
        },
        "polygons": [
            {
                "primitive_id": p.primitive_id,
                "blocker_type": p.blocker_type,
                "vertices": polygon(planner.expanded_polygon_for_primitive(p, clearance)),
            }
            for p in planner.obstacle_layer.primitives
        ],
    }


def route_dict(route) -> dict | None:
    if route is None:
        return None
    return {
        "points": polygon(route.points_xy_m),
        "length_m": exact(route.length_m),
        "vertex_count": route.vertex_count,
        "cache_hit": bool(route.cache_hit),
        "skipped_segments": route.skipped_segments,
    }


def capture() -> dict:
    config = PlannerConfig()
    clearance = config.drone_clearance_m

    # --- nearest_free_point ---------------------------------------------
    nearest = []
    for scene, point, cl, note in [
        ("empty", (100.0, 100.0), 8.0, "already free"),
        ("empty", (10000.0, 0.0), 8.0, "clamped to bounds, margin applied"),
        ("empty", (-500.0, -500.0), 25.0, "exactly on the corner, margin applied"),
        ("one_building", (0.0, 0.0), 8.0, "dead centre of a building"),
        ("one_building", (-45.0, -45.0), 8.0, "just inside a corner"),
        ("one_building", (60.0, 0.0), 8.0, "just outside, already free"),
        ("cluster", (0.0, 0.0), 8.0, "inside overlapping expanded footprints"),
        ("cylinder", (5.0, 5.0), 8.0, "inside a sampled circle"),
        # Pushing out of this one runs into the bounds margin, so the re-clamp
        # after each push is observable.
        ("corner_block", (-470.0, 0.0), 8.0, "pushed out against the boundary"),
        ("corner_block", (-470.0, 0.0), 25.0, "pushed out, wider margin"),
        # Nested footprints: escaping needs more than one iteration.
        ("nested", (0.0, 0.0), 8.0, "nested rings, several push iterations"),
        ("forest", (0.0, 0.0), 8.0, "vegetation does not block, so unchanged"),
        ("boundary_slab", (-500.0, 0.0), 8.0, "escape lies outside bounds; clamp fights push"),
        ("boundary_slab", (-450.0, 30.0), 8.0, "same slab, off-centre"),
        # The push step is floored at 0.5 m, which every clearance above that
        # hides. At 0.2 the floor is what decides how far each push moves, so
        # dropping it changes the answer.
        ("one_building", (0.0, 0.0), 0.2, "clearance below the 0.5 m push floor"),
        ("one_building", (-39.0, -39.0), 0.2, "same, from a corner"),
        ("nested", (0.0, 0.0), 0.2, "nested rings at a sub-floor clearance"),
    ]:
        planner = make_planner(scene)
        result = planner.nearest_free_point(np.asarray(point, dtype=float), cl)
        nearest.append(
            {
                "scene": scene,
                "note": note,
                "point": pair(np.asarray(point, dtype=float)),
                "clearance_m": exact(cl),
                "expected": pair(result),
                # Recorded so a coverage check can prove the fixture reaches
                # both the "already free" and the "pushed out" branches.
                "moved": bool(
                    float(np.linalg.norm(result - np.asarray(point, dtype=float))) > 1e-9
                ),
            }
        )
    if not any(c["moved"] for c in nearest):
        raise AssertionError("no nearest_free_point case is actually pushed out")
    if all(c["moved"] for c in nearest):
        raise AssertionError("no nearest_free_point case is already free")

    # --- plan_route ------------------------------------------------------
    routes = []
    for scene, start, goal, cl, note in [
        ("empty", (-200.0, 0.0), (200.0, 0.0), 8.0, "free straight line"),
        ("one_building", (-200.0, 0.0), (200.0, 0.0), 8.0, "detour"),
        ("one_building", (0.0, 0.0), (200.0, 0.0), 8.0, "start inside an obstacle"),
        ("one_building", (-200.0, 0.0), (0.0, 0.0), 8.0, "goal inside an obstacle"),
        ("empty", (-499.0, 0.0), (200.0, 0.0), 8.0, "start inside the clearance margin"),
        ("empty", (-200.0, 0.0), (499.0, 0.0), 8.0, "goal inside the clearance margin"),
        ("cluster", (-250.0, 0.0), (250.0, 0.0), 8.0, "unroutable cluster"),
        ("box_and_wall", (-250.0, -150.0), (250.0, 150.0), 8.0, "box and wall"),
        ("cylinder", (-200.0, 0.0), (200.0, 0.0), 8.0, "around a cylinder"),
        ("one_building", (-200.0, 0.0), (200.0, 0.0), 25.0, "wider clearance"),
        # Exactly ON the expanded footprint's boundary. `blocking_primitive`
        # uses point-in-polygon, which counts a boundary point as inside, so
        # this is REJECTED -- while `segment_is_free` exempts boundary points,
        # so without the guard the graph would happily route from here. This
        # is the only case that separates the two.
        ("one_building", (-48.0, 0.0), (200.0, 0.0), 8.0, "start on the expanded boundary"),
        ("one_building", (-200.0, 0.0), (-48.0, 0.0), 8.0, "goal on the expanded boundary"),
        # Vegetation is flown through, not around: a straight line.
        ("forest", (-200.0, 0.0), (200.0, 0.0), 8.0, "straight through vegetation"),
        # On the boundary AND with somewhere free to go: the segment away from
        # the square is unobstructed, so without the start guard this would
        # route. The earlier boundary case has no free edge at all and so
        # returns None either way -- it cannot separate the guard from a
        # failed search.
        (
            "one_building",
            (-48.0, -20.0),
            (-200.0, -200.0),
            8.0,
            "start on the boundary with a free way out",
        ),
        # Exactly on the bounds margin: x_min + clearance. The guard is
        # inclusive, so this is accepted; an exclusive one would reject it.
        ("empty", (-492.0, 0.0), (200.0, 0.0), 8.0, "start exactly on the bounds margin"),
        ("empty", (-200.0, 0.0), (492.0, 0.0), 8.0, "goal exactly on the bounds margin"),
    ]:
        planner = make_planner(scene)
        route = planner.plan_route(
            np.asarray(start, dtype=float), np.asarray(goal, dtype=float), clearance_m=cl
        )
        routes.append(
            {
                "scene": scene,
                "note": note,
                "start": pair(np.asarray(start, dtype=float)),
                "goal": pair(np.asarray(goal, dtype=float)),
                "clearance_m": exact(cl),
                "expected": route_dict(route),
            }
        )
    if not any(c["expected"] is None for c in routes):
        raise AssertionError("no plan_route case is rejected; the guards are untested")
    if not any(c["expected"] is not None for c in routes):
        raise AssertionError("no plan_route case succeeds")

    # --- route_waypoints -------------------------------------------------
    waypoints = []
    for scene, anchors, cl, closed, note in [
        (
            "empty",
            [(-150.0, -150.0), (150.0, -150.0), (150.0, 150.0), (-150.0, 150.0)],
            8.0,
            True,
            "closed loop, nothing in the way",
        ),
        ("empty", [(-150.0, -150.0), (150.0, -150.0), (150.0, 150.0)], 8.0, False, "open path"),
        (
            "one_building",
            [(-150.0, -150.0), (150.0, -150.0), (150.0, 150.0), (-150.0, 150.0)],
            8.0,
            True,
            "closed loop around a building",
        ),
        # An anchor inside a building: route_waypoints runs every anchor
        # through nearest_free_point first, so this must still route.
        (
            "one_building",
            [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)],
            8.0,
            False,
            "an anchor starts inside an obstacle",
        ),
        (
            "cluster",
            [(-250.0, 0.0), (250.0, 0.0), (250.0, 200.0), (-250.0, 200.0)],
            8.0,
            True,
            "segments that cannot be routed are skipped",
        ),
        # The same anchors twice, differing only in `closed`. The obstacle sits
        # on the closing edge, so the loop has to route around it and the open
        # path never touches it. Asserted below to actually differ.
        (
            "edge_blocker",
            [(-150.0, -150.0), (150.0, -150.0), (150.0, 150.0), (-150.0, 150.0)],
            8.0,
            True,
            "closed: the closing segment must detour",
        ),
        (
            "edge_blocker",
            [(-150.0, -150.0), (150.0, -150.0), (150.0, 150.0), (-150.0, 150.0)],
            8.0,
            False,
            "open: the same anchors never traverse the closing edge",
        ),
        ("empty", [(0.0, 0.0), (0.0, 0.0)], 8.0, True, "coincident anchors"),
        # Exactly at the threshold: four closed segments, two of which cross
        # the unroutable cluster. The reference gives up at MORE than
        # max(1, count // 2) == 2, so two survives and the route comes back
        # with skipped_segments == 2. At `>=` it would give up instead, and
        # that is the only thing separating the two comparisons.
        (
            "cluster",
            [(-250.0, 0.0), (250.0, 0.0), (-250.0, 10.0), (-250.0, 200.0)],
            8.0,
            True,
            "skips exactly at the threshold",
        ),
    ]:
        planner = make_planner(scene)
        route = planner.route_waypoints(
            np.asarray(anchors, dtype=float), clearance_m=cl, closed=closed
        )
        waypoints.append(
            {
                "scene": scene,
                "note": note,
                "anchors": polygon(np.asarray(anchors, dtype=float)),
                "clearance_m": exact(cl),
                "closed": closed,
                "expected": route_dict(route),
            }
        )
    if not any(
        c["expected"] is not None and c["expected"]["skipped_segments"] > 0 for c in waypoints
    ):
        raise AssertionError("no route_waypoints case skips a segment")
    if not any(c["expected"] is None for c in waypoints):
        raise AssertionError("no route_waypoints case returns None")
    threshold = next(c for c in waypoints if c["note"] == "skips exactly at the threshold")
    if threshold["expected"] is None or threshold["expected"]["skipped_segments"] != 2:
        got = "None" if threshold["expected"] is None else threshold["expected"]["skipped_segments"]
        raise AssertionError(
            f"the threshold case skipped {got}, not exactly 2; it cannot "
            "separate `> max(1, n // 2)` from `>=`"
        )
    # `closed` must change something, or a port that ignored it would pass.
    closed_pair = [c for c in waypoints if c["scene"] == "edge_blocker"]
    if len(closed_pair) != 2:
        raise AssertionError("the closed/open pair is missing")
    if closed_pair[0]["expected"] == closed_pair[1]["expected"]:
        raise AssertionError(
            "closed=True and closed=False produced identical routes; the flag "
            "is unobservable and a port ignoring it would pass"
        )

    # --- the cache -------------------------------------------------------
    #
    # A sequence of calls against ONE planner, recording cache_hit each time.
    # The quantization matters: `snap_m` defaults to 10, so two starts 3 m
    # apart share a key and the second is served the FIRST one's route --
    # points and all, not merely a similar route.
    planner = make_planner("one_building")
    sequence = []
    for start, goal, cl, note in [
        ((-200.0, 0.0), (200.0, 0.0), 8.0, "cold"),
        ((-200.0, 0.0), (200.0, 0.0), 8.0, "same request, hit"),
        ((-197.0, 0.0), (200.0, 0.0), 8.0, "3 m away: snaps to the same key, hit"),
        ((-160.0, 0.0), (200.0, 0.0), 8.0, "40 m away: a different key, miss"),
        ((-200.0, 0.0), (200.0, 0.0), 25.0, "different clearance, miss"),
        ((-200.0, 0.0), (200.0, 0.0), 8.0, "back to the first key, still hit"),
    ]:
        route = planner.plan_route(
            np.asarray(start, dtype=float), np.asarray(goal, dtype=float), clearance_m=cl
        )
        sequence.append(
            {
                "start": pair(np.asarray(start, dtype=float)),
                "goal": pair(np.asarray(goal, dtype=float)),
                "clearance_m": exact(cl),
                "note": note,
                "expected": route_dict(route),
            }
        )
    if not any(s["expected"]["cache_hit"] for s in sequence):
        raise AssertionError("no cache hit in the sequence")
    if not any(not s["expected"]["cache_hit"] for s in sequence[1:]):
        raise AssertionError("no cache miss after the first call")
    # The snap case must genuinely be served the earlier route, not merely
    # produce an equal one: assert the returned points start at the ORIGINAL
    # start, not the one asked for.
    snapped = sequence[2]
    if snapped["expected"]["points"][0] != sequence[0]["expected"]["points"][0]:
        raise AssertionError(
            "the 3 m case did not reuse the first route; the snap is not exercised"
        )

    # Eviction is invisible at the default capacity of 512. A capacity of two
    # makes the order observable: the reference moves an entry to the end on a
    # hit and evicts from the FRONT, so touching the oldest entry saves it.
    tiny = make_planner("one_building", PlannerConfig(max_cache_entries=2))
    eviction = []
    for start, goal, note in [
        ((-200.0, 0.0), (200.0, 0.0), "A: cold"),
        ((-200.0, 100.0), (200.0, 100.0), "B: cold, cache now [A, B]"),
        ((-200.0, 0.0), (200.0, 0.0), "A again: hit, moves A to the end -> [B, A]"),
        ((-200.0, 200.0), (200.0, 200.0), "C: cold, evicts B -> [A, C]"),
        ((-200.0, 0.0), (200.0, 0.0), "A: still cached because the hit saved it"),
        ((-200.0, 100.0), (200.0, 100.0), "B: evicted, so a miss"),
    ]:
        route = tiny.plan_route(
            np.asarray(start, dtype=float), np.asarray(goal, dtype=float), clearance_m=8.0
        )
        eviction.append(
            {
                "start": pair(np.asarray(start, dtype=float)),
                "goal": pair(np.asarray(goal, dtype=float)),
                "note": note,
                "cache_hit": bool(route.cache_hit),
            }
        )
    if eviction[4]["cache_hit"] is not True or eviction[5]["cache_hit"] is not False:
        raise AssertionError(
            "the eviction sequence does not distinguish front- from back-eviction: "
            f"got {[e['cache_hit'] for e in eviction]}"
        )

    # Quantization. Two distinct properties, and the first is a trap.
    #
    # Python's `round()` is BANKER'S rounding: `round(0.5) == 0` and
    # `round(-0.5) == 0`, so +5 and -5 at snap 10 share a key. Rust's
    # `f64::round()` rounds half away from zero and would give +1 and -1 --
    # two different keys for requests the reference treats as one. The port
    # uses `round_ties_even()`; this case is what forces it.
    #
    # +6 then separates rounding from truncation: it keys to 1 under either
    # rounding rule and to 0 under `floor`, which is +5's key.
    quant = make_planner("empty")
    quantization = []
    for start, goal, cl, note in [
        ((5.0, 0.0), (300.0, 0.0), 8.0, "+5 cold"),
        ((-5.0, 0.0), (300.0, 0.0), 8.0, "-5: banker's rounding shares +5's key"),
        ((6.0, 0.0), (300.0, 0.0), 8.0, "+6: a different key under rounding, +5's under floor"),
        ((5.0, 0.0), (300.0, 0.0), 8.0000001, "clearance differs below 1e-3: same key"),
    ]:
        route = quant.plan_route(
            np.asarray(start, dtype=float), np.asarray(goal, dtype=float), clearance_m=cl
        )
        quantization.append(
            {
                "start": pair(np.asarray(start, dtype=float)),
                "goal": pair(np.asarray(goal, dtype=float)),
                "clearance_m": exact(cl),
                "note": note,
                "cache_hit": bool(route.cache_hit),
            }
        )
    if not quantization[1]["cache_hit"]:
        raise AssertionError(
            "-5 missed +5's entry, so this build does not use banker's rounding "
            "and the case cannot catch a port that rounds half away from zero"
        )
    if quantization[2]["cache_hit"]:
        raise AssertionError("+6 shared +5's key; truncation is not distinguishable")
    if not quantization[3]["cache_hit"]:
        raise AssertionError("the sub-millimetre clearance missed; rounding is not applied")

    # Two floors that the default configuration hides. `snap_m` is floored at
    # 1.0 and `cylinder_vertex_count` at 6, and the defaults (10 and 12) sit
    # above both, so neither floor does anything unless the config pushes
    # below it.
    fine_snap = PlannerConfig(snap_m=0.1)
    snap_planner = make_planner("empty", fine_snap)
    snap_calls = []
    for start, note in [
        ((0.0, 0.0), "cold"),
        # 0.4 m away. Under the 1.0 floor both key to 0 and this hits; at the
        # configured 0.1 they are four keys apart and it misses.
        ((0.4, 0.0), "0.4 m away: the 1.0 floor makes this the same key"),
    ]:
        route = snap_planner.plan_route(
            np.asarray(start, dtype=float), np.array([300.0, 0.0]), clearance_m=8.0
        )
        snap_calls.append(
            {
                "start": pair(np.asarray(start, dtype=float)),
                "goal": pair(np.array([300.0, 0.0])),
                "note": note,
                "cache_hit": bool(route.cache_hit),
            }
        )
    if not snap_calls[1]["cache_hit"]:
        raise AssertionError(
            "the 0.4 m call missed, so snap_m was not floored at 1.0 and the floor cannot be tested"
        )

    coarse = PlannerConfig(cylinder_vertex_count=3)
    coarse_planner = make_planner("cylinder", coarse)
    coarse_polygon = coarse_planner.expanded_polygon_for_primitive(
        coarse_planner.obstacle_layer.primitives[0], 8.0
    )
    if len(coarse_polygon) != 6:
        raise AssertionError(
            f"a vertex count of 3 produced {len(coarse_polygon)} vertices; the "
            "floor of 6 is not being applied"
        )

    return {
        "source": "src/argusnet/planning/planner_base.py",
        "note": (
            "PathPlanner2D's public routing surface, read back from the "
            "reference. Cases are tagged agreement/divergence: the route "
            "cache key is a deliberate change, so the reference is recorded "
            "as evidence of what was changed, not as the target."
        ),
        "config": {
            "drone_clearance_m": exact(config.drone_clearance_m),
            "target_clearance_m": exact(config.target_clearance_m),
            "snap_m": exact(config.snap_m),
            "cylinder_vertex_count": config.cylinder_vertex_count,
            "turn_penalty_m": exact(config.turn_penalty_m),
            "max_cache_entries": config.max_cache_entries,
        },
        "default_clearance_m": exact(clearance),
        "scenes": [scene_metadata(name, 8.0) for name in SCENES],
        "scenes_wide": [scene_metadata(name, 25.0) for name in SCENES],
        "tag": "agreement",
        "nearest_free_point": nearest,
        "plan_route": routes,
        "route_waypoints": waypoints,
        "cache_sequence": {
            "tag": "agreement",
            "note": (
                "The reference's cache behaviour, which the port reproduces "
                "for a FIXED scene. What the port adds is a revision in the "
                "key, so a scene change misses instead of serving a route "
                "planned around obstacles that have moved -- the reference "
                "cannot express that, so there is nothing to record for it."
            ),
            "calls": sequence,
        },
        "cache_eviction": {
            "max_cache_entries": 2,
            "clearance_m": exact(8.0),
            "scene": "one_building",
            "calls": eviction,
        },
        "cache_quantization": {"scene": "empty", "calls": quantization},
        "snap_floor": {"scene": "empty", "snap_m": exact(0.1), "calls": snap_calls},
        "cylinder_vertex_floor": {
            "scene": "cylinder",
            "cylinder_vertex_count": 3,
            "clearance_m": exact(8.0),
            "expected": polygon(coarse_polygon),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    counts = {k: len(v) for k, v in document.items() if isinstance(v, list)}
    counts["cache_calls"] = len(document["cache_sequence"]["calls"])
    print(f"wrote {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
