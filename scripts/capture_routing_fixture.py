#!/usr/bin/env python3
"""Record PathPlanner2D's visibility-graph routing against the reference.

Scope is the ordering-sensitive core: polygon expansion, segment freedom, the
visibility graph, the A* search and the path smoothing. `plan_route`'s cache
and `route_waypoints` are P4.7d.

Two things make this fixture unusual, and both are the point.

**Adjacency is recorded in insertion order, not as a set.** A* iterates
`adjacency[node]` in list order and rejects a neighbour whose cost is not
strictly better (`next_cost + 1e-9 >= best_cost`), so between two equal-cost
routes the one reached first wins. A port that stored adjacency in a HashSet,
or sorted it, would produce a different path on a symmetric obstacle while
every individual edge still agreed. The Rust test asserts order.

**Expanded polygons are recorded as inputs, not derived.** The Rust side is
given the same polygons the reference built, so a disagreement is localised to
the routing rather than to footprint construction, which Phase 3 already
ported and tests separately.

Values come from calling the reference and reading its internals back, never
from recomputing the formulas here.

Usage:
    python3 scripts/capture_routing_fixture.py --out tests/fixtures/routing_graph.json
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

from argusnet.planning.planner_base import (  # noqa: E402
    PathPlanner2D,
    PlannerConfig,
    _expand_polygon,
)
from argusnet.world.environment import Bounds2D, ObstacleLayer  # noqa: E402
from argusnet.world.obstacles import BuildingPrism, WallSegment  # noqa: E402


def exact(value: float) -> str:
    """Floats cross as strings: `serde_json`'s parser is an ULP off
    `str::parse::<f64>` on some literals, which broke an exact comparison in
    the sensor error model."""
    return repr(float(value))


def pair(xy) -> list[str]:
    return [exact(xy[0]), exact(xy[1])]


def polygon(points) -> list[list[str]]:
    return [pair(p) for p in np.asarray(points, dtype=float)]


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
    return BuildingPrism(
        primitive_id=pid,
        footprint_xy_m=footprint,
        base_z_m=0.0,
        top_z_m=40.0,
    )


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


BOUNDS = Bounds2D(x_min_m=-500.0, x_max_m=500.0, y_min_m=-500.0, y_max_m=500.0)


def make_planner(primitives) -> PathPlanner2D:
    layer = ObstacleLayer(bounds_xy_m=BOUNDS, tile_size_m=100.0, primitives=primitives)
    return PathPlanner2D(bounds_xy_m=BOUNDS, obstacle_layer=layer, config=PlannerConfig())


def capture_case(name: str, primitives, start, goal, clearance: float, expect: str) -> dict:
    """Drive the reference and read its graph, path and smoothing back out.

    `expect` is "direct", "detour" or "unroutable", and the generator FAILS if
    reality disagrees. Two cases in the first version of this fixture claimed
    to force a detour and did not -- the obstacles left a corridor the straight
    line went through, so they recorded a two-node no-op that would have passed
    against a port with no visibility graph at all.
    """
    planner = make_planner(primitives)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    prims = planner._candidate_primitives(start, goal)
    polys = [(p.primitive_id, planner.expanded_polygon_for_primitive(p, clearance)) for p in prims]
    nodes: list[np.ndarray] = [start, goal]
    polygon_node_indices: list[list[int]] = []
    for _, poly in polys:
        indices: list[int] = []
        for vertex in poly:
            nodes.append(np.asarray(vertex, dtype=float))
            indices.append(len(nodes) - 1)
        polygon_node_indices.append(indices)

    adjacency = planner._build_visibility_graph(nodes, polygon_node_indices, clearance)
    path = planner._astar_path(nodes, adjacency)
    smoothed = None if path is None else planner._smooth_path(path, clearance)

    direct = planner._segment_is_free(start, goal, clearance)
    if expect == "detour" and direct:
        raise AssertionError(
            f"case {name!r} claims to force a detour, but the straight line is "
            "free: it would record a two-node no-op"
        )
    if expect == "direct" and not direct:
        raise AssertionError(f"case {name!r} claims a direct line that is blocked")
    if expect == "detour" and (smoothed is None or len(smoothed) < 3):
        count = 0 if smoothed is None else len(smoothed)
        raise AssertionError(
            f"case {name!r} forced a detour but produced {count} points; "
            "a route around an obstacle needs at least three"
        )
    if expect == "unroutable" and smoothed is not None:
        raise AssertionError(f"case {name!r} expected no route but found one")

    return {
        "name": name,
        "clearance_m": exact(clearance),
        "start": pair(start),
        "goal": pair(goal),
        "polygons": [{"primitive_id": pid, "vertices": polygon(poly)} for pid, poly in polys],
        "nodes": [pair(n) for n in nodes],
        "polygon_node_indices": polygon_node_indices,
        # Insertion order preserved deliberately -- see the module docstring.
        "adjacency": [adjacency[i] for i in range(len(nodes))],
        "astar_path": None if path is None else [pair(p) for p in path],
        "smoothed": None if smoothed is None else [pair(p) for p in smoothed],
        "straight_line_is_free": bool(planner._segment_is_free(start, goal, clearance)),
    }


def capture() -> dict:
    one = [building("b1", 0.0, 0.0, 40.0)]
    # Two buildings leaving a gap. The start and goal are offset in y so the
    # direct line clips the upper block and the route has to thread the gap.
    gate = [building("b1", 0.0, 70.0, 50.0), building("b2", 0.0, -70.0, 50.0)]
    # Three blocks whose EXPANDED footprints overlap (10 m gaps, 8 m
    # clearance). The reference finds no route at all, even though one plainly
    # exists around the cluster: the visibility graph's only nodes are obstacle
    # corners, and the corners along the cluster's shared boundary each fall
    # inside a neighbour's expanded polygon, so every segment joining them is
    # rejected as interior. Start reaches the first block's corners and stops.
    # Recorded because the port must reproduce the None, not because it is
    # desirable -- and because P4.7 asks for unreachable-goal coverage.
    row = [
        building("b1", -70.0, 0.0, 30.0),
        building("b2", 0.0, 0.0, 30.0),
        building("b3", 70.0, 0.0, 30.0),
    ]
    walled = [wall("w1", -120.0, 0.0, 120.0, 0.0)]
    mixed = [building("b1", -60.0, 20.0, 35.0), wall("w1", 40.0, -120.0, 40.0, 120.0)]

    cases = []
    for spec in [
        ("clear_line_no_obstacles", [], (-200.0, -200.0), (200.0, 200.0), 8.0, "direct"),
        # Symmetric about the start-goal axis: the mirrored detour is exactly
        # as long, so which side A* returns is decided purely by the order
        # neighbours appear in the adjacency lists.
        ("single_building_detour", one, (-200.0, 0.0), (200.0, 0.0), 8.0, "detour"),
        ("single_building_larger_clearance", one, (-200.0, 0.0), (200.0, 0.0), 25.0, "detour"),
        ("threaded_gate", gate, (-250.0, -120.0), (250.0, 120.0), 8.0, "detour"),
        ("overlapping_cluster_is_unroutable", row, (-250.0, 0.0), (250.0, 0.0), 8.0, "unroutable"),
        # Also symmetric, in the other axis.
        ("wall_crossing", walled, (0.0, -200.0), (0.0, 200.0), 8.0, "detour"),
        ("mixed_building_and_wall", mixed, (-250.0, -150.0), (250.0, 150.0), 8.0, "detour"),
        ("diagonal_past_corner", one, (-150.0, -150.0), (150.0, 150.0), 12.0, "detour"),
        ("start_adjacent_to_obstacle", one, (-55.0, 0.0), (200.0, 0.0), 8.0, "detour"),
    ]:
        cases.append(capture_case(*spec))

    # Polygon expansion on its own, including the miter-limit floor: a sharp
    # vertex is where `scale = max(dot, 0.2)` actually binds.
    expansions = []
    shapes = {
        "square": [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        "square_clockwise": [[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]],
        "sharp_spike": [[0.0, 0.0], [100.0, 2.0], [100.0, -2.0]],
        "triangle": [[0.0, 20.0], [-20.0, -10.0], [20.0, -10.0]],
    }
    for label, verts in shapes.items():
        arr = np.asarray(verts, dtype=float)
        # 0.5 sits above the 1e-9 early-out but below 1.0, so widening
        # that threshold to 1.0 becomes observable. Without it every
        # case is either far below the cutoff or far above it.
        for clearance in (0.0, 1.0e-12, 0.5, 5.0, 30.0):
            expansions.append(
                {
                    "label": label,
                    "clearance_m": exact(clearance),
                    "polygon": polygon(arr),
                    "expected": polygon(_expand_polygon(arr, clearance)),
                }
            )

    # `_smooth_path` on an A* result is nearly always a no-op: the path is
    # already taut, because the graph only contains visible edges. So it was
    # completely untested until these cases were added -- mutating it to take
    # the NEAREST visible point rather than the farthest changed nothing. The
    # reference feeds it slack polylines through `route_waypoints`, which
    # concatenates per-segment routes; these mirror that.
    smoothing = []
    smoothing_specs = [
        # A detour around nothing: the middle points are all directly visible
        # from the first, so a correct smoother collapses to two points and a
        # nearest-first smoother keeps them all.
        (
            "slack_line_no_obstacles",
            [],
            [(-200.0, 0.0), (-100.0, 40.0), (0.0, -40.0), (100.0, 40.0), (200.0, 0.0)],
        ),
        # Slack around a real obstacle: some points can be skipped, others
        # cannot, so neither "keep all" nor "keep two" passes.
        (
            "slack_around_one_building",
            one,
            [
                (-200.0, 0.0),
                (-150.0, -30.0),
                (-100.0, -60.0),
                (-48.0, -48.0),
                (0.0, -60.0),
                (48.0, -48.0),
                (100.0, -60.0),
                (200.0, 0.0),
            ],
        ),
        # Exactly two points: the early-out path.
        ("two_points_pass_through", one, [(-200.0, 0.0), (-150.0, 0.0)]),
        # Three collinear points: the middle one is redundant.
        ("three_collinear_points", [], [(-200.0, 0.0), (0.0, 0.0), (200.0, 0.0)]),
    ]
    for label, primitives, raw in smoothing_specs:
        planner = make_planner(primitives)
        clearance = 8.0
        prims = planner._candidate_primitives(
            np.asarray(raw[0], dtype=float), np.asarray(raw[-1], dtype=float)
        )
        polys = [
            (p.primitive_id, planner.expanded_polygon_for_primitive(p, clearance)) for p in prims
        ]
        arr = np.asarray(raw, dtype=float)
        smoothed = planner._smooth_path(arr, clearance)
        smoothing.append(
            {
                "label": label,
                "clearance_m": exact(clearance),
                "polygons": [
                    {"primitive_id": pid, "vertices": polygon(poly)} for pid, poly in polys
                ],
                "input": polygon(arr),
                "expected": polygon(smoothed),
            }
        )

    # The turn penalty decides between a shorter, kinkier route and a longer,
    # straighter one. No obstacle layout in this fixture can pin it: going
    # further around a building is also going *straighter* around it, so the
    # two terms never trade off and every recorded path was identical at
    # penalties 0, 3 and 40.
    #
    # A synthetic graph separates them. Route A is a shallow zigzag -- shorter
    # but with four reversals; route B is a single wide V -- longer but with
    # one gentle turn. A wins on length alone, B wins once turning is priced.
    # The generator FAILS if the recorded paths do not actually differ.
    zig = [(20.0, 3.0), (40.0, -3.0), (60.0, 3.0), (80.0, -3.0)]
    turn_nodes = [np.array([0.0, 0.0]), np.array([100.0, 0.0])]
    turn_nodes += [np.array(p) for p in zig]
    turn_nodes.append(np.array([50.0, 25.0]))
    zig_ids = [2, 3, 4, 5]
    vee_id = 6
    turn_adjacency = {i: [] for i in range(len(turn_nodes))}

    def link(a: int, b: int) -> None:
        turn_adjacency[a].append(b)
        turn_adjacency[b].append(a)

    chain = [0, *zig_ids, 1]
    for a, b in zip(chain[:-1], chain[1:], strict=False):
        link(a, b)
    link(0, vee_id)
    link(vee_id, 1)

    turn_variants = []
    for penalty in (0.0, 3.0, 200.0):
        planner = make_planner([])
        planner.config = PlannerConfig(turn_penalty_m=penalty)
        path = planner._astar_path(turn_nodes, turn_adjacency)
        turn_variants.append(
            {
                "turn_penalty_m": exact(penalty),
                "path": None if path is None else [pair(p) for p in path],
            }
        )
    distinct = {str(v["path"]) for v in turn_variants}
    if len(distinct) < 2:
        raise AssertionError(
            "the turn-penalty variants all chose the same path, so the penalty "
            "is unpinned and dropping it from the port would go unnoticed"
        )

    # `_segment_is_free` on its own. The graph cases only ever hand it
    # polygon CORNERS, which lie on a boundary and so take the exemption; the
    # strictly-inside endpoint check was untested until these, and deleting it
    # changed nothing.
    freedom = []
    freedom_planner = make_planner(one)
    freedom_clearance = 8.0
    freedom_prims = freedom_planner._candidate_primitives(
        np.array([-200.0, 0.0]), np.array([200.0, 0.0])
    )
    freedom_polys = [
        (p.primitive_id, freedom_planner.expanded_polygon_for_primitive(p, freedom_clearance))
        for p in freedom_prims
    ]
    corner = freedom_polys[0][1][0]
    for label, a, b in [
        ("both_outside_clear", (-200.0, 200.0), (200.0, 200.0)),
        ("both_outside_crossing", (-200.0, 0.0), (200.0, 0.0)),
        ("start_strictly_inside", (0.0, 0.0), (200.0, 0.0)),
        ("end_strictly_inside", (-200.0, 0.0), (0.0, 0.0)),
        ("both_strictly_inside", (-10.0, -10.0), (10.0, 10.0)),
        # A corner is ON the boundary, which is exempt -- this is the case
        # that makes the graph work at all, since every vertex edge starts
        # on a boundary.
        ("start_on_boundary", tuple(float(v) for v in corner), (-200.0, 200.0)),
        ("grazes_a_corner", (-200.0, 48.0), (200.0, 48.0)),
    ]:
        freedom.append(
            {
                "label": label,
                "start": pair(np.asarray(a, dtype=float)),
                "end": pair(np.asarray(b, dtype=float)),
                "expected": bool(
                    freedom_planner._segment_is_free(
                        np.asarray(a, dtype=float),
                        np.asarray(b, dtype=float),
                        freedom_clearance,
                    )
                ),
            }
        )
    freedom_polygons = [
        {"primitive_id": pid, "vertices": polygon(poly)} for pid, poly in freedom_polys
    ]
    if not any(c["expected"] for c in freedom):
        raise AssertionError("every segment-freedom case is blocked; none proves the clear path")
    if all(c["expected"] for c in freedom):
        raise AssertionError("every segment-freedom case is free; none proves a block")

    # Two routes of EXACTLY equal cost reaching the same A* state, so the
    # search's tie-breaking is what decides the answer. 0 -> A -> X and
    # 0 -> B -> X are mirror images, so state (C, X) is reached twice at the
    # same cost; the original keeps the FIRST parent, and anything that
    # accepts an equal-cost relaxation keeps the second, flipping the path
    # between the +y and -y limbs.
    tie_nodes = [
        np.array([0.0, 0.0]),
        np.array([200.0, 0.0]),
        np.array([50.0, 40.0]),
        np.array([50.0, -40.0]),
        np.array([100.0, 0.0]),
        np.array([150.0, 0.0]),
    ]
    tie_adjacency = {i: [] for i in range(len(tie_nodes))}

    def tie_link(a: int, b: int) -> None:
        tie_adjacency[a].append(b)
        tie_adjacency[b].append(a)

    for a, b in [(0, 2), (0, 3), (2, 4), (3, 4), (4, 5), (5, 1)]:
        tie_link(a, b)
    tie_planner = make_planner([])
    tie_path = tie_planner._astar_path(tie_nodes, tie_adjacency)
    if tie_path is None:
        raise AssertionError("the tie graph must be routable")
    via = [float(p[1]) for p in tie_path]
    if not any(abs(v) > 1.0 for v in via):
        raise AssertionError("the tie graph route never leaves the axis; the limbs are not used")

    # Two graphs that isolate the A* frontier's ORDERING, which nothing else
    # in this fixture reaches.
    #
    # `collinear` puts three exactly-collinear nodes on the path. For a
    # collinear pair the cosine is dot / (|a| * |b|), and those are computed
    # differently enough that the quotient can round ABOVE 1 -- 23% of random
    # collinear pairs do, up to 1.0000000000000002 for these coordinates,
    # which `acos` rejects outright. So the clamp is load-bearing, not
    # decorative, and this is the case that proves it.
    #
    # `tiebreak` was found by search rather than construction: 40,000 random
    # graphs were run through the reference with the g tie-break reversed, and
    # this is one where the answer changes.
    ordering_graphs = {}

    # Found by search, not placed by hand: the vectors A* actually compares
    # come out of SUBTRACTION, which rounds differently from the
    # multiplication a hand-built collinear pair uses -- the first triple
    # tried here gave a ratio of exactly 1.0 once the subtraction was done,
    # and proved nothing. The check below refuses to emit a triple that does
    # not still exceed 1 after subtraction.
    collinear_nodes = [
        np.array([0.0, 0.0]),
        np.array([40.8, -111.0]),
        np.array([13.6, -37.0]),
    ]
    _incoming = collinear_nodes[2] - collinear_nodes[0]
    _outgoing = collinear_nodes[1] - collinear_nodes[2]
    _dot = float(_incoming[0] * _outgoing[0] + _incoming[1] * _outgoing[1])
    _na = math.sqrt(float(_incoming[0]) ** 2 + float(_incoming[1]) ** 2)
    _nb = math.sqrt(float(_outgoing[0]) ** 2 + float(_outgoing[1]) ** 2)
    _ratio = _dot / max(_na * _nb, 1.0e-9)
    if not _ratio > 1.0:
        raise AssertionError(
            f"the collinear triple gives a cosine of {_ratio!r} after subtraction; "
            "it must exceed 1 or the clamp before acos is not exercised"
        )
    collinear_adjacency = {0: [2], 1: [2], 2: [0, 1]}
    collinear_planner = make_planner([])
    collinear_path = collinear_planner._astar_path(collinear_nodes, collinear_adjacency)
    if collinear_path is None or len(collinear_path) != 3:
        raise AssertionError("the collinear graph must route through the middle node")
    ordering_graphs["collinear"] = {
        "nodes": [pair(n) for n in collinear_nodes],
        "adjacency": [collinear_adjacency[i] for i in range(len(collinear_nodes))],
        "expected_path": [pair(p) for p in collinear_path],
    }

    tiebreak_nodes = [
        np.array([0.0, 0.0]),
        np.array([200.0, 0.0]),
        np.array([0.0, 100.0]),
        np.array([200.0, -40.0]),
        np.array([200.0, -80.0]),
        np.array([-40.0, -20.0]),
        np.array([140.0, 20.0]),
    ]
    tiebreak_adjacency = {
        0: [2, 4, 5],
        1: [3, 4],
        2: [0, 4, 6],
        3: [1, 4],
        4: [0, 1, 2, 3, 5],
        5: [0, 4],
        6: [2],
    }
    tiebreak_planner = make_planner([])
    tiebreak_path = tiebreak_planner._astar_path(tiebreak_nodes, tiebreak_adjacency)
    if tiebreak_path is None:
        raise AssertionError("the tie-break graph must route")
    ordering_graphs["tiebreak"] = {
        "nodes": [pair(n) for n in tiebreak_nodes],
        "adjacency": [tiebreak_adjacency[i] for i in range(len(tiebreak_nodes))],
        "expected_path": [pair(p) for p in tiebreak_path],
    }

    turn_graph = {
        "nodes": [pair(n) for n in turn_nodes],
        "adjacency": [turn_adjacency[i] for i in range(len(turn_nodes))],
        "variants": turn_variants,
    }

    return {
        "source": "src/argusnet/planning/planner_base.py",
        "note": (
            "Visibility-graph routing read back from the reference. Adjacency "
            "is recorded in INSERTION ORDER: A* iterates it in order and keeps "
            "the first of two equal-cost routes, so a port storing adjacency "
            "in a set would diverge on a symmetric obstacle while every edge "
            "still agreed."
        ),
        "turn_penalty_m": exact(PlannerConfig().turn_penalty_m),
        "cylinder_vertex_count": PlannerConfig().cylinder_vertex_count,
        "expansions": expansions,
        "cases": cases,
        "smoothing": smoothing,
        "segment_freedom": {
            "clearance_m": exact(freedom_clearance),
            "polygons": freedom_polygons,
            "cases": freedom,
        },
        "ordering_graphs": ordering_graphs,
        "tie_graph": {
            "nodes": [pair(n) for n in tie_nodes],
            "adjacency": [tie_adjacency[i] for i in range(len(tie_nodes))],
            "expected_path": [pair(p) for p in tie_path],
        },
        "turn_penalty_graph": turn_graph,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    counts = {k: len(v) for k, v in document.items() if isinstance(v, list)}
    counts["turn_variants"] = len(document["turn_penalty_graph"]["variants"])
    routed = sum(1 for c in document["cases"] if c["smoothed"] is not None)
    detoured = sum(
        1 for c in document["cases"] if not c["straight_line_is_free"] and c["smoothed"] is not None
    )
    unroutable = sum(1 for c in document["cases"] if c["smoothed"] is None)
    print(
        f"wrote {args.out}: {counts}; routed={routed}, detoured={detoured}, unroutable={unroutable}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
