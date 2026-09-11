# Route planning — P4.7d

Ports `PathPlanner2D`'s public surface — `nearest_free_point`, `plan_route`,
`route_waypoints` and the route cache — onto P4.7c's routing core. Completes
the planner half of P4.7.

## The cache carries a scene revision, and the reference's does not

The one deliberate divergence, under ADR-006.

`PLANNING_INVENTORY.md` established that the reference's route cache key is
quantized start, quantized goal and clearance — no obstacle or scene identity
of any kind — and that `PathPlanner2D.clear_cache()` has **no caller anywhere**
in `src/` or `tests/`. Nothing invalidates it. Meanwhile
`Environment.add_obstacle()` already bumps `scene_version`, and
`EnvironmentQuery.los` already keys its own cache on that counter: the planner
is the only cache in the world/planning stack that ignores it.

Reproducing that would mean handing a platform a route around a no-fly volume
that has since moved, silently, because the start and goal happened to quantize
the same way. So `Scene` carries a revision and it is part of both the route
and the expanded-footprint keys.

The reference cannot express this, so there is nothing to record for it: the
fixture captures the reference's cache behaviour for a **fixed** scene as
agreement and the port reproduces it exactly, while two Rust tests assert the
new contract directly. The revision **partitions** the cache rather than
clearing it — an entry from revision 1 survives a detour through revision 2.

Latent rather than live in the reference: `add_obstacle` has no callers today.

## A port bug the fixture caught: `round()` is not `round()`

Python's built-in `round()` is **banker's rounding**. `round(0.5) == 0` and
`round(-0.5) == 0`, so a start at +5 and one at −5 with `snap_m = 10` share a
cache key and the second is served the first's route. Rust's `f64::round()`
rounds half away from zero and would key them +1 and −1 — two entries for
requests the reference treats as one.

The port uses `round_ties_even()`. This was not spotted by reading: the
fixture generator refuses to emit a quantization case that cannot tell the two
rules apart, and it failed on its first run with

```
AssertionError: +5 and -5 shared a key; rounding is not observable
```

which was the generator telling me my *expectation* was wrong, and the port
with it. The same applies to `round(clearance, 3)`.

## Behaviours reproduced deliberately

- **Vegetation does not block a route.** Only `building` and `wall` are hard
  blockers; a route flies straight through a forest stand. Without a scene
  containing vegetation, a port that treated every obstacle as solid would
  pass the entire suite.
- **`nearest_free_point` gives up after ten pushes.** Against a slab
  straddling the western boundary the nearest escape is *outside* the bounds
  margin, so the clamp after each push drags the point straight back inside
  and the loop makes no progress; the cap is what ends it, and the point
  comes back still inside the obstacle. Recorded rather than fixed — callers
  already treat the result as advisory.
- **A cluster of obstacles can be unroutable**, per `VISIBILITY_ROUTING.md`.
- **`route_waypoints` gives up at *more* than `max(1, segments / 2)` skips**,
  so exactly half may be skipped and the route is still returned.
- Two floors the defaults hide: `snap_m` is floored at 1.0 (default 10) and
  `cylinder_vertex_count` at 6 (default 12). Both are pinned with non-default
  configurations, because nothing else reaches them.

## One optimisation taken deliberately

The reference re-queries the obstacle layer inside *every* `_segment_is_free`
call — O(n²) queries per route. The port computes the candidate set once for
the whole start–goal window and reuses it.

That is equivalent, not a shortcut: an obstacle the per-segment query would
exclude has bounds that do not overlap the segment, so it can neither contain
an endpoint nor cross it, and a superset only ever adds blocking. Asserted
rather than argued — `a_superset_of_candidate_obstacles_changes_nothing` adds
a far-away obstacle and checks no route moves.

## Verification

Fixture: `tests/fixtures/plan_route.json`, from
`scripts/capture_plan_route_fixture.py`, which drives the reference and reads
its results back. Scenes cross as the metadata `to_metadata()` emits and are
rebuilt in Rust through `ObstacleMetadata::into_obstacle` — the same
conversion the native bridge uses — so an oriented box's and a wall's
footprint construction is exercised rather than assumed. Regenerates
byte-identically.

**The generator refuses to emit a case that does not do what it claims.** It
raises if no `nearest_free_point` case is actually pushed out (or if none is
already free), if no `plan_route` case is rejected, if no `route_waypoints`
case skips a segment or returns `None`, if the closed/open pair produces
identical routes, if the threshold case does not skip *exactly*
`max(1, segments / 2)`, if the eviction sequence cannot distinguish
front- from back-eviction, or if the quantization cases cannot separate
banker's rounding from truncation. Five of those guards fired during
development.

### Mutation testing: 28 mutations, each verified to have reached disk

**24 caught by a named test, 4 proven equivalent.**

The first pass caught **10 of 28**, with all 18 tests passing on the first run
— the same pattern as P4.7c, and the reason the first green run is treated as
a warning rather than a result. What the survivors exposed:

- Both endpoint guards were untested in a way that separated them from a
  failed search. A start strictly *inside* an obstacle yields no route either
  way — every edge leaving it is rejected, so A\* finds nothing regardless.
  Only a start exactly **on** the expanded boundary distinguishes them:
  `blocking_primitive` uses point-in-polygon, which counts a boundary point as
  inside, while `segment_is_free` exempts boundary points, so without the
  guard the graph routes happily.
- The bounds guard's inclusivity, the LRU's touch-on-hit and its
  eviction-from-the-front, the `snap_m` and `cylinder_vertex_count` floors,
  the ten-push cap and the re-clamp, vegetation, and the skip threshold were
  all unreachable by any existing case.
- The cache-key mutations are what turned up the `round()` bug above.

### Four proven equivalent mutants

Asserted in `four_recorded_equivalent_mutants_in_the_planner`, not tested
around:

1. **The empty-query fallback** in `candidate_primitives`. The window is
   padded by `max(clearance * 4, 80)` and a footprint expands by only
   `clearance`; since `4c >= c` for every non-negative `c`, an empty query
   means no obstacle can reach the segment, so the straight line is free, the
   early return fires, and no graph is ever built from the fallback's extra
   polygons.
2. **The `max(80.0)` floor on that window**, by the same inequality.
3. **De-duplicating the join point** between consecutive segments. The
   duplicate is coincident with the previous point, so it is trivially visible
   from it and `smooth_path` skips straight past it.
4. **`nearest_free_point`'s `margin.max(0.5)` at the call site.**
   `push_outside_polygon` floors its own margin at 0.5 as well, so the outer
   floor can never change the step. The reference double-floors identically —
   `planner_base.py:207` passes `max(margin_m, 0.5)` into a function whose
   first act is `max(margin_m, 0.5)` — so this is faithful redundancy rather
   than a port mistake, and is left in place.
