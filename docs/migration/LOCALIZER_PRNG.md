# The localizer's particle draw: two deliberate divergences

**Status:** ported 2026-09-10 (P4.2b), under ADR-006.

## What ADR-006 licenses, and what it doesn't

ADR-006 accepts that the Rust runtime cannot be bit-identical to numpy, and
redefines the contract as *deterministic for a fixed seed and config*, validated
against declared statistical envelopes. It also names a property to **preserve**:
the per-`(node, target, tick)` seed keying that makes `build_observations`
independent of the order nodes and targets are visited in.

`GridLocalizer` does not have that property to preserve. It draws from a single
generator — `np.random.default_rng(42)`, created once in `__init__` — consumed
sequentially by every platform on every step. Two consequences follow, and both
were measured on the reference before anything was written.

## 1. The reference is not order-independent

Running the same four steps with the platform order swapped:

| visit order | drone-1 estimate x | drone-2 estimate x |
|---|---|---|
| `[1, 2, 1, 2]` | 498.900434 | 498.636080 |
| `[2, 1, 2, 1]` | 498.636080 | 498.900434 |

The estimates swap outright. Each platform gets whichever block of the shared
stream its turn lands on, so the identity of the result belongs to the *slot*,
not to the platform.

This is not a rounding-scale difference that envelopes could absorb. It means a
scheduler change, a dictionary iteration order, or adding a platform silently
reassigns every other platform's noise.

## 2. `reset()` does not restore the draw sequence

`reset()` clears `_estimates`, `_step_counts` and `_timed_out`. It never touches
`_rng`. So a reset localizer is not equivalent to a fresh one:

| | estimate x on the first step |
|---|---|
| fresh instance | `500.275561590426` |
| after `reset()` | `498.27161014760924` |

Two metres apart. P4.1 defines reset as returning the runtime to a reproducible
state, and `argusnet-sim`'s reset semantics were built on that definition. The
reference does not meet it.

## What was ported

`argusnet_sim::particles::particle_offsets` keys a fresh stream per
`(platform_id, step)`, namespaced under `localizer.particles`, using the
`DeterministicRng` and FNV-1 seed derivation P4.1 established.

Both properties above follow by construction rather than by discipline:

- A platform's offsets depend on its own identity, its own step count and the
  run seed. Nothing another platform does can move them, so visit order is
  irrelevant.
- Reset clears step counts, and step counts are part of the key, so a reset
  localizer draws exactly what a fresh one draws.

**This is a behaviour change, taken deliberately rather than inherited.** It is
the change ADR-006's envelopes exist to license: the numbers differ from
Python's, the statistical character does not, and the resulting behaviour is
the one the rest of the simulation already has.

The scale still comes from `GridLocalizer::particle_scale`, which owns annealing
and coverage shrinkage — this module only draws. Nothing about the estimator
formulas ported in P4.2a changes.

## The declared envelopes, and how their limits were set

For `n` draws from `N(0, s)`, the sample mean has standard error `s/√n` and the
sample standard deviation roughly `s/√(2n)`. The tests bound both at five
standard errors, so a correct sampler fails about once in 3.5 million runs.

`n` sets the **detection floor**, and it was chosen by measuring rather than by
picking a round number. At `n = 20 000` the mean bound is `0.035 s`, and a
deliberately injected bias of `0.02 s` **passed the test unnoticed**. Raising to
`n = 200 000` tightens the bound to `0.0112 s` for the mean and `0.0079 s` for
the spread, which catches that bias and any spread error above about 1%.

Recording this because a statistical envelope that cannot fail is worse than no
envelope: it reads like verification while asserting nothing. The envelope is
only as good as the effect size it can resolve, and that number belongs in the
open.

Beyond the moments, the tests pin what moments alone cannot see:

- **The two coordinates are uncorrelated.** Box–Muller pairs a radius with an
  angle, so reusing one variate for both axes leaves them perfectly correlated
  while each still passes the mean and spread checks.
- **Scale is a pure multiplier on a fixed key**, which catches a scale applied
  to one axis only, or applied twice.
- **The namespace is part of the key**, so another subsystem keyed on the same
  platform and tick cannot draw this stream's numbers.

Seven mutations of the sampler are each caught by a named test. Two of them —
the `0.02 s` bias and dropping the namespace — survived the first version, and
are why the envelope was tightened and the namespace test written.

## What this does not do

The transitional native estimator handle is **not** included here. This module
supplies offsets; wiring Python's `GridLocalizer` to consume Rust-drawn offsets
through `argusnet-native` is a separate change with its own golden re-baseline,
and it should land only once someone has decided whether the two divergences
above are corrections or regressions for the scenarios the goldens cover.

That decision is worth making explicitly. Both changes make the localizer behave
*more* like the rest of the simulation, but they will move any golden that
depends on the current draw sequence.
