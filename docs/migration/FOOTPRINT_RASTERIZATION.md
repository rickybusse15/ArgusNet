# Circular footprint rasterization: the two Python implementations disagree

**Status:** measured, 2026-09-10. Ported to `argusnet-mapping` as two named
samplings; the choice between them is still open and belongs to P4.6.

## Why this document exists

`DELIVERY_PLAN.md` §P4.6 says:

> Introduce one footprint rasterizer whose indices/weights feed both coverage
> and belief updates, while preserving their distinct statistical meanings.

and §4 describes the present state as:

> `CoverageMap.mark_circular` and `WorldMap.add_scan_observation` independently
> rasterize the *same* circular footprint with their own meshgrid and
> `np.add.at`, called back-to-back on the same position every tick.

The premise is wrong in one word. They do not rasterize the *same* footprint.
Taken at face value, "introduce one rasterizer" is a deduplication; in fact it
is a **behaviour change**, and doing it without noticing would silently move the
goldens.

## The two rasterizations

| | `CoverageMap.mark_circular` | `WorldMap.add_scan_observation` |
|---|---|---|
| sample lattice | metres, `arange(c - r, c + r + res, res)` | cell indices, `c_idx - r_cells ..= c_idx + r_cells` |
| anchored at | `centre - radius` (moves with the drone) | the centre cell (grid-aligned) |
| membership test | `dx² + dy² <= radius²`, in metres | `di² + dj² <= r_cells²`, in cells |
| effective radius | `radius` | `max(1, trunc(radius / resolution)) · resolution` |

The last row is the divergence. Truncating the radius to whole cells makes the
height disc **strictly smaller** whenever `radius / resolution` is not an
integer, and the `max(1, …)` floor makes it **larger** when the radius is under
one cell.

## Measured

From `tests/fixtures/footprint_cells.json`, comparing distinct cells touched:

| case | resolution | radius | ratio | coverage | height | height / coverage |
|---|---|---|---|---|---|---|
| `regional_integer_ratio` | 50 m | 350 m | 7.00 | 149 | 149 | 100% |
| `fine_fractional_ratio` | 25 m | 120 m | 4.80 | 71 | 49 | **69%** |
| `fine_just_under_integer` | 25 m | 124.9 m | 5.00⁻ | 71 | 49 | **69%** |
| `fine_just_over_integer` | 25 m | 125.1 m | 5.00⁺ | 77 | 81 | 105% |
| `small_footprint_floor` | 12 m | 50 m | 4.17 | 52 | 49 | 94% |
| `inexact_resolution` | 7.3 m | 90 m | 12.33 | 474 | 441 | 93% |
| `radius_below_one_cell` | 50 m | 10 m | 0.20 | **0** | 5 | — |

The `fine_just_under_integer` / `fine_just_over_integer` pair is the shape of the
problem: 0.2 m of radius moves the height footprint from 49 cells to 81, a 65%
jump, because it crosses an integer number of cells.

`radius_below_one_cell` is the opposite failure. A footprint smaller than a cell
records **no coverage at all**: the lattice is anchored at `centre - radius` and
steps by the resolution, so with `radius < resolution` its samples sit at offsets
`-radius` and `resolution - radius` from the centre — at 10 m radius on a 50 m
grid, 10 m and 40 m out. Both are outside the 10 m circle, so nothing is marked.
Meanwhile the cell disc's `max(1, …)` floor still writes terrain height into 5
cells. Height is recorded where coverage says nothing was observed.

### Is it reachable?

Yes, though not on the default path. `sim.py` derives

```python
_r_cells = max(1, ceil(sqrt(_target_cells / pi)))
_FOOTPRINT_RADIUS_M = max(50.0, float(_r_cells) * _coverage_bounds.resolution_m)
```

so the radius is normally `r_cells · resolution` — an exact multiple, ratio
integral, and the two agree. That is why this has gone unnoticed.

But `--coverage-resolution-m` is user-settable with only a `> 0` guard, and the
`max(50.0, …)` floor binds whenever `r_cells · resolution < 50`. There the
radius is 50 m against an arbitrary resolution, and the ratio is arbitrary.
Sweeping ten such configurations, **all ten diverged**, with the height disc
covering 59–94% of the coverage disc.

The consequence is that `WorldMap` records terrain height over less ground than
it records coverage over: cells counted toward `coverage_fraction` have no height
sample, so `mean_height_grid` is NaN there and `extract_features` cannot see
them. Coverage and belief drift apart, which is exactly what P4.6's shared
rasterizer is meant to prevent.

## Two details the port had to reproduce exactly

**`np.arange` is not `start + i·step`.** numpy's `DOUBLE_fill` writes the first
two entries, derives `delta = (start + step) - start`, and computes the rest as
`start + i·delta`. Rounding makes that `delta` differ from `step` when `start` is
large relative to `step`: `arange(-89.7, 97.3, 7.3)[4]` is `-60.500000000000014`,
not `-60.5`. Two ULP is ignorable in the coordinate and is *not* ignorable after
truncation to a cell index, where either side of a boundary is a different cell.
`lattice_reproduces_numpy_arange_bit_for_bit` pins it.

**Cells can be hit twice by one call, and the second hit counts.** The lattice
pitch equals the cell size, so distinct samples normally land in distinct cells.
At the far edge they do not: the bounds test is inclusive of `x_max_m`, a sample
landing exactly there truncates to index `nx`, and the clamp folds it onto
`nx - 1` alongside its neighbour. `np.add.at` increments both. The
`sample_exactly_on_far_edge` case carries **14 such folds**, so the rasterizer
returns hits with multiplicity and the parity test compares sequences rather than
sets — a set comparison would pass while `mean_revisits` quietly changed.

## What was ported, and what was not

`argusnet-mapping::footprint` implements one rasterizer with both behaviours as
an explicit `Sampling` mode — `WorldAnchored` and `CellDisc`. Callers can migrate
mode-for-mode with behaviour preserved, and both are pinned cell-for-cell against
Python by `tests/footprint_parity.rs`.

Unifying them is deliberately **not** done here. That is a semantic decision with
a golden re-baseline attached, and it should be made with the numbers above in
view rather than as a side effect of a port.

When P4.6 makes it, the recommendation from these measurements is
`WorldAnchored`: it honours the radius it is given, it does not quantise
behaviour to integer cell counts, and it is the one the coverage statistics —
the numbers the mission FSM actually gates on — already use. The cost is that
belief updates change wherever the ratio is non-integral, which needs the
statistical envelopes ADR-006 provides rather than a bit-exact golden.

Two loose ends that belong with that decision:

- The sub-cell radius case needs a defined answer. Neither current behaviour is
  obviously right: coverage recording nothing is defensible, height recording 5
  cells is not.
- `add_scan_observation` calls `mark_circular` internally, so in
  `scan_map_inspect` the lattice is rasterized **twice per drone per tick** on
  two different `CoverageMap` instances, with the LOS predicate evaluated twice.
  One rasterizer should also mean one evaluation.
