# P4.6d: publishing belief, and the one update that does not commute

**Status:** delivered 2026-09-16. Completes P4.6 alongside P4.6a (footprint
rasterizer), P4.6b (coverage aggregates) and P4.6c (the query facade).

P4.6's second paragraph asks for the publication side:

> Publish immutable belief revisions and dirty tile patches for planners and
> P2.1. Carry observed/unknown, uncertainty, timestamp and evidence provenance.
> Define out-of-bounds/missing-tile behavior and deterministic update order when
> multiple sensors touch a cell. Test commutative updates where promised,
> deterministic ordered updates where not, tile seams and no-truth-access.

Unlike every other P4.6 slice, **this one has no Python reference to port.**
`WorldMap` and `OccupancyGrid` expose live mutable arrays; nothing publishes.
`belief_cells` is `[]` in every replay in the repo and `mapping_state`'s belief
aggregates are all `null`. So the contract below is *stated*, and this document
is where it is stated.

What *is* ported is the fusion arithmetic, cell for cell. That turned out to
matter more than expected.

## The finding: `OccupancyGrid` does not commute, and the clamp is why

P4.6 says to test "commutative updates where promised, deterministic ordered
updates where not". Deciding which is which needed measurement, not reading.

| quantity | fusion | commutes? |
|---|---|---|
| `coverage_count` | saturating `+= 1` | **yes**, exactly |
| `last_observed_s` | `max` | **yes**, exactly |
| `height_sum` / `height_count` | `+=` | yes in exact arithmetic; **4.1e-13 m** apart over 4,000 f64 observations |
| `log_odds` | `clip(lo + delta, -4, 4)` | **no** |

`OccupancyGrid.mark_free` and `mark_occupied` clamp **on every observation**,
and a clamp that binds throws away evidence a different order would still have
had room for. Six observations of one cell — five `mark_occupied`, one
`mark_free`:

| order | log-odds | obstacle probability |
|---|---|---|
| free **last** | 3.6 | **0.9734** |
| free **first** | 3.85 | **0.9792** |

Same evidence, two answers. Five `+0.85` reach the `+4.0` ceiling and lose 0.25
before the `-0.4` lands; put the `-0.4` first and there is headroom for all of
it. The difference is 0.6 percentage points here and can be far larger near the
clamp — enough to move a cell across a planner threshold.

That single fact is the reason `BeliefStore` **stages rather than applies**.

## Deterministic order: sorted between sources, arrival order within one

`BeliefStore::apply` stages. `BeliefStore::commit` sorts by
`(source_id, arrival index within that source)` and then fuses.

- **Between sources**, lexicographic by `source_id`. So the result does not
  depend on which drone's gRPC response landed first, which is the thing the
  mapping layer does not control and should not be sensitive to.
- **Within one source**, arrival order, preserved. A sensor's own frames are
  genuinely sequential and reordering them would be a lie about time.

Applying at call time would also have been deterministic — deterministically
dependent on the caller's scheduling. That is the distinction worth having.

Both halves are pinned: `commit_order_does_not_depend_on_arrival_order` shuffles
the arrival order of two sources and requires the same answer;
`one_sources_own_frames_keep_their_arrival_order` requires **named values**, not
merely a difference, because asserting only `!=` passes just as happily when the
frames are reversed. `a_shuffled_tick_always_commits_to_the_same_belief` runs 16
shuffles of 42 observations from 6 sources across a tiled grid.

## Unknown is a value, not an absence

Every patch carries an explicit `observed` mask, and **nothing else in the patch
is meaningful where that mask is false.** Unobserved cells carry `NaN` heights
and `NaN` timestamps, not `0.0`.

That is deliberate, and it is P2.1b's constraint made mechanical: *"a replay
without belief data renders an explicit unknown state rather than fabricated
confidence."* A consumer that ignores the mask produces visible garbage rather
than a plausible lie about flat, freshly-scanned ground at the datum. Since
**no replay in the repo carries belief data**, the unknown path is the only one
that runs on real data today — so it is the one built to be loud.

`obstacle_probability` is `0.5` with no occupancy evidence. That is the unknown
prior, and it is only ever read alongside `observed == false`, so it cannot be
mistaken for a measured coin flip.

`tile_patch` answers for a tile nothing has ever touched, returning a full patch
of explicit unknown rather than `None`. A missing tile and an unobserved tile
are the same thing to a renderer, and only one of them can be drawn.

## Out of bounds is dropped and counted, never clamped

`OccupancyGrid.mark_free` routes through `bounds.xy_to_ij`, which **clamps**.
Clamping an observation from outside the grid piles that evidence onto an edge
cell, which then reports confident belief about ground nobody looked at.

Here an out-of-range index is discarded, counted on the store
(`dropped_out_of_bounds`) and reported on the revision that dropped it. A commit
whose cells are *all* out of bounds publishes nothing and does not move the
revision counter — but still lists the source, so a caller can tell "nothing
arrived" from "everything arrived off-grid".

## Two uncertainties, and only one of them looks at the heights

`WorldBeliefQuery::height_uncertainty_at` returns
`resolution / sqrt(coverage_count)`. That is a **geometric proxy**: it never
reads a height. It shrinks with revisits whether the revisits agree or not.

The patch carries it under that name, so the query and the renderer agree. It
also carries a new field, `height_dispersion_m` — the sample standard deviation
of the heights actually observed in the cell, `NaN` below two observations.
The reference has no dispersion measure at all, so this is an addition, named
separately rather than quietly redefining a field the replay schema already
has.

`height_dispersion_m` is maintained with Welford's M2, but its running mean is
taken from the ported `sum / count` rather than Welford's own. The two differ in
the last place — `[287.78, 285.65, 267.49, 188.44]` gives
`257.33999999999997` one way and `257.34000000000003` the other — and
`mean_height_m` must stay bit-identical to `WorldMap.mean_height_grid`, or a
future divergence would read as a floating-point bug in P4.6b rather than a
design choice here. `mean_height_is_bit_identical_to_sum_over_count` pins it
with those four values.

## Tiles

Ceiling division, so the high-edge tile is a partial tile and not a remainder to
drop: 20 cells at tile size 8 is 8 + 8 + **4**, and the corner tile answers at
4×4 rather than padding to 8×8. `tiles_partition_a_ragged_grid_exactly_once`
walks every cell of a 9×9 grid at tile size 4 and requires each to appear in
exactly one patch, at the tile `tile_of` names.

`patches_since(revision)` is generated from current state, not replayed from a
log. A consumer that falls behind by many revisions receives each changed tile
**once, at its latest value**, which bounds the catch-up cost and means no
history has to be retained. `Revision::INITIAL` gives a full resync of
everything ever touched; `all_patches()` gives every tile including the
untouched ones, which is what a cold renderer's first frame needs.

## Provenance

Per-cell, as a `u64` bitmask over an interned source table — 8 bytes per cell
regardless of how long a run is. `MAX_TRACKED_SOURCES` is 64.

A 65th distinct source still has its **evidence applied in full**; only its
provenance bit is unavailable, and `source_overflow_count` reports that rather
than letting it pass silently.
`a_source_past_the_provenance_limit_still_fuses_and_is_counted` asserts the
coverage count reaches 65, which is the part that would be easy to get wrong.

## The footprint join

`Observation::from_footprint` is P4.6's *"one footprint rasterizer whose
indices/weights feed both coverage and belief updates"*. The reference calls two
rasterizers on the same disc with two different samplings — see
`FOOTPRINT_RASTERIZATION.md` for the measurement, where the height disc covers
as little as 59% of the coverage disc. Coming through one call here, coverage
and height necessarily cover the **same cells**.

Which sampling to use stays the caller's visible decision, as P4.6a established.
It is a behaviour change from the reference either way, and it should be one
someone chose.

## No truth access

Structural. `BeliefStore` holds grids and a source table and nothing else: no
terrain, no obstacle layer, no entity list. Every path in is an `Observation`.
There is no method that *could* consult truth, which is a stronger guarantee
than a test that one currently does not.

`the_store_can_only_know_what_was_observed` pins the consequence: the cell next
to an observed obstacle reports the unknown prior, not free ground, and its
neighbour's evidence does not leak sideways.

## Verification

- 57 unit tests in `revision.rs`, 4 integration tests in
  `tests/belief_publication.rs` crossing into `WorldBeliefQuery`.
- **19 mutants, 18 caught.** The survivor — replacing the `height_count > 1`
  guard with `> 0` — is equivalent: at `n == 1`, `m2` is `0.0` and `count - 1`
  is `0`, so `0.0 / 0.0` is already `NaN`. The guard is defensive, not
  load-bearing.
- Mutants that a weaker suite would have passed, and what caught them:

  | mutant | caught by |
  |---|---|
  | sort key ignores `source_id` | `commit_order_does_not_depend_on_arrival_order` |
  | a source's own frames reversed | `one_sources_own_frames_keep_their_arrival_order` |
  | log-odds clamped at read time | `the_occupancy_log_odds_update_does_not_commute` |
  | out-of-bounds clamped, not dropped | `an_out_of_bounds_cell_is_dropped_and_counted_never_clamped` |
  | tiling floors instead of ceils | `tiles_partition_a_ragged_grid_exactly_once` |
  | unobserved mean height reads `0.0` | `an_untouched_tile_patches_as_explicit_unknown_not_as_absence` |
  | coverage only counted where a verdict exists | `a_cell_is_covered_even_when_the_sensor_concluded_nothing` |
  | Welford's own running mean | `mean_height_is_bit_identical_to_sum_over_count` |

  The first three survived an earlier revision of the suite. They are the
  reason the tests assert named values rather than differences.

## What this unblocks

**P2.1b** (the belief render channel) was deferred in the graphics lane with the
note: *"revisit when the runtime lane emits belief, and let the shape it emits
drive the snapshot type rather than guessing it from the schema."* That shape is
now `BeliefTilePatch`: grid origin, dims, resolution via `bounds()`, an explicit
observed mask, confidence, last-observed time and a revision — which is the
`BeliefRenderSnapshot` that section describes, minus the guessing.

Discrete channels (`observed`, `source_mask`) are separate from numeric ones
(`obstacle_probability`, `height_*`, `last_observed_s`), so the nearest-sampling
rule P2.1b requires has something to apply to.

**P4.8** needs a producer for `belief_cells` in the Rust replay writer.
`WorldBeliefQuery` over `BeliefStore`'s grids is that producer, and
`belief_publication.rs` is the test that the two agree.

## Still open

`BeliefTilePatch` derives `PartialEq`, which is `false` for two identical
patches containing `NaN`. That is ordinary f64 behaviour and the tests compare
by bits where it matters, but a consumer using `==` for change detection should
use `revision` instead.

Nothing yet *calls* `BeliefStore` from the transitional Python loop. P4.6's exit
condition — *"mapping is load-bearing in the transitional Python loop and shared
with the renderer"* — needs P4.8's tick composition for the first half and
P2.1b for the second. This slice supplies what both consume.
