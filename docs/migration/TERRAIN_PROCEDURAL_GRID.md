# Procedural terrain grid (P4.3e)

`src/argusnet/world/procedural.py` → `rust/argusnet-world/src/numpy_rng.rs` and
`terrain_grid.rs`.

Fifth slice of P4.3, and the top of the terrain builder: the function that turns
a preset name and a seed into the height field and the six semantic masks that
everything downstream reads. It composes P4.3a's noise primitives, P4.3b's grid
operations and P4.3c's semantic pass over a single numpy `default_rng` stream.

Two fixtures: `tests/fixtures/numpy_rng.json` (19 seeds, 741 uniform draws, 304
raw outputs) and `tests/fixtures/terrain_grid.json` (22 scenes, 2,986 cells).

## numpy's PRNG is reproducible, and ADR-006 is amended rather than ignored

ADR-006 recommends **not** reproducing numpy's PRNG bit-for-bit. Its stated
reason is cost — *"possible but fragile and costly ... a hand-verified
reimplementation of two algorithms"* — and it was written about
`build_observations`, which builds a **fresh generator per (node, target) pair
per tick** and uses `SeedSequence.spawn`.

`procedural_terrain_grid` is the opposite case. **One** generator, **12 to 46**
scalar `uniform(low, high)` draws, no `spawn`, no arrays, no other distribution.
A pure-Python `SeedSequence` + `PCG64` reproduced numpy exactly on the first
attempt, and then on **24,000 of 24,000** draws across 2,000 random seeds and
ten ranges. The cost premise does not hold here.

The benefit does. Bit-exact draws mean terrain the Rust runtime builds is the
same terrain the Python scene baker writes, so the two agree about the world
while both exist. A statistical envelope cannot give that, and a disagreement
would be invisible until a replay recorded against one was viewed against the
other.

ADR-006 carries this as a **scoped amendment**, not a departure: it applies the
ADR's own cost reasoning to a case where the cost is different.

## What is reproduced, and what deliberately is not

Only `default_rng(int)` → `Generator.uniform(low, high)`: entropy mixing, PCG64
seeding, the XSL-RR output, the 53-bit double, and the affine scale. `spawn`,
`integers`, `normal` and the array paths are absent because nothing here uses
them and each would need its own proof.

Three details are easy to get wrong and are pinned individually:

- `pcg64_random_r` **steps and then outputs**. Reading first gives a stream
  shifted by one, which is uniform, deterministic and wrong.
- `next_double` is `(raw >> 11) * 2^-53` — the **top** 53 bits. Masking the
  bottom 53 gives a value in the same range from the same stream.
- `uniform` is `low + (high - low) * t`, not the algebraically equal
  `low * (1 - t) + high * t`. The two round differently and the fixture tells
  them apart.

`uniform` is also only defined for `low <= high`: numpy **rejects** a reversed
range with `ValueError: high - low < 0` rather than sorting or scaling by a
negative. The first version of the capture script tried to record one and found
out. There is therefore no reference behaviour to match, and a test says so, so
the gap in the fixture is not read later as an oversight.

**Mutation on the RNG: 38 of 38 caught.** Every SeedSequence constant, the
mixing order, the diagonal skip, the pool indexing, the little-endian word
order, the PCG multiplier, both seeding steps, the fold, the rotation and the
double construction.

## Heights cannot be bit-exact, and the reason is numpy's `exp`

This is the one place in the migration where a tolerance replaces a bit
comparison, so the reason is measured rather than asserted.

`_line_gaussian` uses `np.exp`, and **numpy's `exp` is not the platform's**.
Over 300,000 inputs drawn from the exact ranges `line_gaussian` evaluates —
`-0.5 * (across/width)²` and `-(along/length)⁴` — it differs from glibc's
scalar `exp` by 1 ULP on **4.39%** and **5.02%** of them. That is not a SIMD
dispatch artifact; `NPY_DISABLE_CPU_FEATURES` changes nothing. It is numpy's own
implementation, and matching it would mean porting numpy's `exp` — a larger job
than the whole of P4.3, to be kept correct against future numpy releases.

What that costs, measured across all 22 scenes:

| | |
|---|---|
| cells differing from numpy | **150 of 2,986 (5.25%)** |
| worst disagreement | **32 ULP**, **5.7e-14 m** |
| semantic mask cells differing | **0** |

The 5.25% matches the measured `exp` disagreement rate almost exactly, which is
what **identifies** the cause rather than merely being consistent with it. 5.7e-14 m
is 57 femtometres; the masks, which are what the rest of the system actually
branches on, are bit-identical on every cell of every scene.

The parity test asserts `1e-9 m` — four orders looser than observed, so a
different libm cannot turn the suite red, and eleven orders below anything a
consumer could care about. A tolerance is only honest if a wrong port fails it,
so a guard requires a *different RNG stream* to miss by a factor of a million,
and a second assertion fails if the observed gap ever grows past `1e-12 m`.

### A related finding about already-merged code

`noise.rs`'s `line_gaussian` parity passes bit-exactly on **6 cases**. At ~95%
per-point agreement between numpy's `exp` and the platform's, that is partly
luck rather than a guarantee. It is not wrong today and nothing here changes it,
but it is thinner than it looks and would be worth widening if that fixture is
revisited.

## The draw ORDER is the contract

One generator serves the whole function, so every draw shifts the ones after it,
and reordering two draws used in the same expression still changes the terrain.

The sharpest case is the valley loop: its first iteration takes **one** uniform
for `river_valley` and `mountain_pass` — a near-axis-aligned angle, offset
hardcoded to zero — and **two** for every other preset. That single missing draw
realigns the entire remaining stream, so the preset name silently decides every
later value. All twelve presets are in the fixture, and they draw between 12 and
46 times, so each exercises a different alignment.

The same reasoning is why the terms are accumulated into **separate arrays** and
combined once, in the reference's `base + slope + macro + ridges - valleys +
detail` order. Folding them into a running total as they are computed is the
natural way to write it and was the first version here; it was wrong by 1 ULP at
cell 0 of `alpine`, which is how it was found.

## Mutation: 50 of 54 caught

The first pass caught 48. Two floors survived because nothing in the fixture was
small enough to reach them: the macro wavelengths need a scene under 2.38 m and
the ridge crenulation wavelength one under about 123 m. A 2 m scene reaches both
at once — not a realistic site, but legal input, and *a floor no test can reach
is a floor nobody can prove is wired up*. A 90 m scene covers the crenulation
floor somewhere a real caller might go.

The four remaining survivors, each pinned by a unit test rather than left
unexplained:

**Two are below the tolerance floor, not equivalents.** `combine/base-added-last`
and `combine/detail-before-valleys` move the heights by about 1 ULP — roughly
`1e-14 m` — and the parity tolerance is `1e-9 m`, so it cannot see them.
`the_final_sum_is_not_reassociable` pins the requirement directly.

Choosing values for that test needed a search, and the first attempt asserted
something false: a plausible-looking alpine cell with a large base and small
detail, on the reasoning that small terms lose bits against large ones. All
three orderings agreed exactly. Reassociation shows up where the terms nearly
**cancel** — six parts as large as 28 summing to 4.47 — not where they differ in
magnitude.

**Two are unreachable from the preset table.** `procedural_terrain_grid` takes a
preset *name* and looks the profile up itself, so unlike `build_land_cover_layer`
there is no hook for a caller to drive these:

- `water_bias.clamp(0.01, 0.35)` — the biases are 0.0, 0.01, 0.02, 0.03, 0.06,
  0.08, 0.16, 0.2 and 0.28. None exceeds 0.30, and the two zeroes skip the block
  entirely rather than clamping up to 0.01.
- `max(relief_m * 0.18, 1.0)` — the floor bites below 5.56 m of relief. The two
  presets with less (urban_flat at 12 m, military_compound at 24 m) both have
  `water_bias == 0` and never reach the line; every preset that does reach it
  has at least 28 m, giving 5.04.

## One defect found by composing

`Bounds2D` was **defined twice** — once in `noise`, once in `semantic` — with
identical fields and different types. No caller could pass one where the other
was expected, and nothing noticed until this slice called `grid_values` and
`semantic_masks` from the same function and the compiler refused.

It is now one definition in `noise`, re-exported from `semantic` so existing
imports keep working. CLAUDE.md's "one authority per concern" applies to value
types too, and this is the kind of duplication that only surfaces when two
modules are finally used together.

## Not in this slice

The **obstacle and patch polygon overlay** at the end of `build_land_cover_layer`
— `ForestStand` and footprint polygons rasterised with `_point_in_polygon`,
which `argusnet-world/src/obstacles.rs` already ports. That is deterministic,
has no RNG, and is bit-exact portable; it is a small slice of its own.

`build_terrain_layer`, `build_land_cover_layer`'s caller and `build_landscape`
remain, and they are what P4.8 actually calls.
