# Coverage statistics: from a full-grid reduction to a maintained aggregate

**Status:** ported 2026-09-10 (P4.6b).

P4.6 asks to *"replace repeated full-grid stats with revision-aware aggregates;
prove incremental totals against full reductions, including reset and removal."*

## What the reference does

`CoverageStats` recomputes two whole-grid reductions on every access — the
cells with a non-zero count, and the sum of all counts. The simulation reads it
from inside a doubly nested loop:

```python
for zone in scenario.mission_zones:
    for ns in mobile_states:
        inside = ...
        if inside:
            zmap.mark_circular(...)
        zcov = zmap.stats.coverage_fraction     # outside the `if`
```

The read sits **outside** the `if inside:` that guards the write. So the grid
changes only when a platform is inside the zone, but the statistics are
recomputed for every (zone, platform) pair on every tick regardless.

## Measured

`CoverageMap.stats`, on this repository:

| grid | cells | `stats` | `count_grid` copy |
|---|---|---|---|
| 86 × 86 (default preset) | 7,396 | 14.3 µs | 1.2 µs |
| 172 × 172 | 29,584 | 33.4 µs | 3.7 µs |
| 430 × 430 | 184,900 | 180.9 µs | 22.2 µs |

Multiplied by the loop:

| grid | 3 zones × 3 platforms | 5 zones × 4 platforms |
|---|---|---|
| 86 × 86 | 0.13 ms/tick | 0.29 ms/tick |
| 172 × 172 | 0.30 ms/tick | 0.67 ms/tick |
| 430 × 430 | 1.63 ms/tick | 3.62 ms/tick |

For context, the Phase 0 profile put the whole gRPC round-trip at 1.73 ms/frame
and called it 5.9% of a representative run. At a fine grid this one read is the
same order, and most of it recomputes a value that cannot have changed.

## What was ported

`argusnet_mapping::CoverageGrid` maintains `covered_cells` and
`total_observations` as cells are marked, so `stats()` is O(1) and touches no
cell. `full_reduction_stats()` recomputes from the grid, and exists so the
equality of the two can be asserted rather than assumed.

A `revision` counter bumps on every mutation that changes the grid — and only
then. That is what lets a caller in the zone loop's position skip work between
writes without a cache that can go stale silently.

Two details that are easy to get wrong and invisible until they matter:

- **`covered_cells` rises only on the 0 → 1 transition.** Counting it on every
  hit makes it a hit counter rather than a cell counter, and the two agree
  exactly as long as nothing is ever revisited — which is to say, until the
  first overlapping footprint.
- **Clearing a cell returns *all* of its observations**, not one. A cell
  observed five times must give back five.

Counts saturate rather than wrapping. A wrap at `u32::MAX` would take a
heavily observed cell to zero and thereby report it as *uncovered*.

## How the equality is proved

The safety argument is that the maintained aggregates always equal a full
reduction, so that is asserted as a property over randomised sequences of
marks, clears, footprint rasterisations and resets — 4,000 operations,
concentrated in a small area so revisits and clears of already-covered cells
are common rather than rare. A tidy scripted sequence tends to avoid exactly
the transitions where this breaks.

Seven mutations are each caught by a named test. One is worth recording,
because it shows how the property test was initially too weak:

> **A reset that cleared `covered_cells` but not `total_observations` passed.**

With no covered cells, `mean_revisits` short-circuits to zero whatever the
total holds — so the corruption is invisible in the empty state, and the test
checked only the empty state. It appears on the *next* mark, when one covered
cell is divided into a stale total. The property test now continues past the
reset with further marks and clears, and the mutation fails immediately.

Removal is covered for the same reason P4.6 names it: an aggregate that is only
ever incremented is trivially correct and proves nothing. Nothing in the
reference removes coverage today, which is precisely why the behaviour needs to
be right before something does.

## Not included

The belief half — `WorldMap`'s height accumulation, immutable belief revisions
and dirty-tile patches for planners and P2.1. This is the coverage side only,
which is what the zone loop reads.
