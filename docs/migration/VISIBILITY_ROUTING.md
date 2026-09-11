# Visibility-graph routing — P4.7c

Port of `PathPlanner2D`'s routing core from `src/argusnet/planning/planner_base.py`
into `rust/argusnet-planning/src/route.rs`.

## Scope

Polygon expansion, segment freedom, the visibility graph, the A\* search and
the path smoothing. `plan_route`'s cache and `route_waypoints` are P4.7d — the
revision-keyed cache that P4.7 asks for lands with them, since
`PLANNING_INVENTORY.md` establishes that the reference's cache key carries no
scene revision at all.

Geometry is reused from `argusnet-world`, which Phase 3 already ported and
tests separately: `point_in_polygon`, `segment_polygon_intervals`,
`nearest_point_on_segment`, `signed_polygon_area`, `edge_outward_normal` and
`point_on_segment`. Nothing here reimplements them.

## The ordering contract

A\* iterates a node's neighbours in the order they were appended to its
adjacency list, and rejects a neighbour whose cost is not *strictly* better
(`next_cost + 1e-9 >= best`). Between two routes of equal cost, the one reached
first wins.

So **adjacency is a `Vec`, never a set**, and is built in the reference's
order: ascending `right_index` for visible pairs, then polygon ring edges. The
fixture records adjacency in insertion order and the test compares it in order.
A `HashSet`, or a `sort`, returns the mirror-image detour on a symmetric
obstacle while every individual edge still agrees with Python — and the
mutation that sorts adjacency is caught only by that ordered comparison.

`BinaryHeap` is a max-heap, so `Frontier`'s `Ord` is the **reverse** of
Python's `heapq` tuple `(f, g, node, previous)`, field by field in the same
order. The float fields use `total_cmp`.

The search state is `(node, previous)`, not `node`. The turn penalty makes the
cost of leaving a node depend on how it was entered, so two arrivals by
different predecessors are genuinely different states.

## Two behaviours worth knowing about

**A cluster of obstacles can be unroutable.** Three buildings 10 m apart with
8 m of clearance have overlapping expanded footprints. The visibility graph's
only nodes are obstacle corners, and the corners along the cluster's shared
boundary each fall inside a neighbour's expanded polygon, so every segment
joining them is rejected as interior. The start reaches the first block's
corners and stops; `plan_route` returns `None` even though a route plainly
exists around the cluster. Recorded as a fixture case
(`overlapping_cluster_is_unroutable`) because the port must reproduce it, and
because P4.7 asks for unreachable-goal coverage — not because it is desirable.

**A segment lying along a polygon edge is never free.** `segment_polygon_intervals`
reports the whole edge as an interior interval. This is what makes the
`contains` guard before appending a ring edge unreachable — see the equivalent
mutants below — and it is asserted directly rather than assumed.

## Verification

Fixture: `tests/fixtures/routing_graph.json`, from
`scripts/capture_routing_fixture.py`, which drives the reference and reads its
internals back. It regenerates byte-identically.

Geometry is compared at `1e-9` relative rather than bit-exactly: the reference
measures distance with `np.linalg.norm`, whose summation need not match
`sqrt(dx*dx + dy*dy)`. Adjacency and path **indices** are integers and are
compared exactly.

**The generator refuses to emit a case that does not do what it claims.** Each
route case declares `direct`, `detour` or `unroutable`, and the generator
raises if reality disagrees. Two cases in its first version claimed to force a
detour and did not — the obstacles left a corridor the straight line went
through — so they recorded a two-node no-op that would have passed against a
port with no visibility graph at all. The same guard rejects a turn-penalty set
whose variants all choose the same path, and a segment-freedom set that is
entirely free or entirely blocked.

### Mutation testing: 18 mutations, every one verified to have reached disk

**13 caught by a named test, 1 detected as a hang, 4 survived.** The first pass
caught only 8 of 18 — the tests passed on the first run and were much weaker
than that suggested. What the ten survivors exposed:

- **`smooth_path` was completely untested.** On an A\* result it is nearly
  always a no-op, because the graph holds only visible edges and the path
  arrives taut. Making it take the *nearest* visible point rather than the
  farthest changed no recorded output. Closed with slack polylines of the kind
  `route_waypoints` actually feeds it (5→2, 8→4 and 3→2 points).
- **The turn penalty was unpinned**, and no obstacle layout in this fixture can
  pin it: routing further around a building is also routing *straighter* around
  it, so the two terms never trade off — all three recorded penalties chose
  identical paths in every case. A synthetic graph separates them: a short
  zigzag against a long wide V. The reference switches between them at a
  penalty of 200.
- **The strict-improvement epsilon was untested**, despite the module docstring
  calling it load-bearing. Closed with a graph where `0 → A → X` and
  `0 → B → X` are mirror images, so the state `(C, X)` is reached twice at
  exactly equal cost: the reference keeps the first parent, and anything that
  accepts an equal-cost relaxation returns the opposite limb.
- **The `g` tie-break was untested.** Found by *searching* 40,000 random graphs
  for one where reversing it changes the reference's answer, rather than by
  hand-construction. The mirror-image ties elsewhere have equal `g`, so the
  ordering among them is settled by node index before `g` is consulted.
- **The zero-clearance early-out threshold was unpinned.** Every expansion case
  was either far below `1e-9` or far above `1.0`, so widening the cutoff to
  `1.0` passed. Closed by adding a 0.5 m clearance.
- **`segment_is_free`'s strictly-inside endpoint check was untested**, because
  the graph only ever hands it polygon *corners*, which lie on a boundary and
  take the exemption. Closed with a dedicated segment-freedom section.

### The clamp before `acos` is load-bearing, and its absence is a hang

For a collinear pair the cosine is `dot / (|a| * |b|)`, and numerator and
denominator round differently often enough that the quotient exceeds 1 — 23% of
random collinear pairs do, reaching `1.0000000000000002`. `acos` of that is
`NaN`.

Removing the clamp does not produce a wrong route. It produces a `NaN` turn
penalty, and every pruning comparison against `NaN` is false, so states
re-relax forever and **the search never terminates**. The mutation is detected
by a 240-second timeout, not by an assertion — which is a poor way to find out,
and is why the collinear case is in the fixture and why a named test asserts
that the unclamped quotient really is `NaN` for it.

Finding a triple that reaches it took a second attempt. The vectors A\*
compares come out of **subtraction**, which rounds differently from the
multiplication a hand-built collinear pair uses: the first triple tried gave a
quotient of exactly `1.0` once placed in a graph and proved nothing. The
generator now refuses to emit a triple that does not still exceed 1 after
subtraction.

### Two proven equivalent mutants

1. `smooth_path`'s early-out is `points.len() <= 2`. Lowering it to `<= 1`
   changes nothing: with two points the loop runs once, `next` starts at
   `len - 1 == 1`, the inner `while next > index + 1` is immediately false, and
   `points[1]` is pushed — reproducing the input exactly.
2. The `contains` guard before appending a polygon ring edge can never fire. A
   segment along a polygon edge is never free, so the visible-pair loop cannot
   have already added a ring edge; and two rings never share a node index, so
   no other ring can have added it. The guard is dead code in the reference.

Both are asserted in `two_recorded_equivalent_mutants` rather than closed with
a contrived test.

### Two survivors that are not proven equivalent

Stated as open gaps rather than dismissed:

- **Dropping `segment_is_free`'s `end` endpoint check.** 300,000 randomised
  segments failed to produce one that ends strictly inside a polygon without
  also registering a crossing interval — which the geometry argues must be so,
  since reaching the interior means crossing the boundary, and the *start*
  check covers starting inside. Strong evidence of redundancy; not a proof.
- **Dropping the A\* heuristic**, turning it into Dijkstra. No discriminating
  graph was found in 40,000 random trials. With a consistent heuristic and
  identical `(g, node, previous)` tie-breaks after `f`, both appear to reach
  the same first-arrival parents. Not proven equivalent, and not closed.
