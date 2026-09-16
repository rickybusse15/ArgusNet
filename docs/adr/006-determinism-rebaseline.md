# ADR-006: Determinism Re-baseline for the Rust Runtime

**Status:** accepted
**Date:** 2026-09-02

## Context

Deterministic simulation for a fixed seed and configuration is a stated project
invariant (CLAUDE.md), and two tests depend on it: `tests/test_runtime.py::RustRuntimeParityTest`
and `rust/argusnet-server/tests/grpc_parity.rs` both replay the frozen fixture
`tests/fixtures/runtime_parity_fixture.json`.

What makes the current simulation reproducible under reordering is a specific
numpy construction. `build_observations` creates a fresh generator **per
(node, target) pair per tick**:

```python
rng = np.random.default_rng(_stable_seed(seed, node_id, target_id, round(t * 1000)))
```

Each pair therefore draws from its own independent stream keyed by identity and
time, so the result does not depend on iteration order. This is a good design.
It is also unreproducible outside numpy: `default_rng` seeds a `SeedSequence`,
whose entropy-mixing algorithm and PCG64 output function would have to be
reimplemented bit-exactly in Rust to yield the same numbers.

ADR-005 moves this code to Rust. A decision is therefore needed **before** the
port rather than discovered during it: preserve bit-parity, or re-baseline.

## Decision

**Re-baseline. Do not attempt bit-parity with numpy.**

1. The determinism contract becomes: *the Rust runtime is deterministic for a
   fixed seed and configuration, and reproducible across platforms.* It is no
   longer "identical to the Python implementation's output".
2. The seed-derivation *scheme* is preserved: a per-(node, target, tick) stream
   keyed by a stable hash of the same inputs, so order-independence — the
   property that actually matters — is retained.
3. The Rust PRNG is named and pinned in the port, and its choice is documented
   alongside the seed derivation.
4. Golden files under `tests/golden/` are regenerated at the port, in a commit
   that changes nothing else, so the diff is reviewable as a pure re-baseline.
5. During the transition both implementations run against the same scenarios and
   are compared on **statistical envelopes** — acceptance and rejection rates,
   rejection-reason distributions, track position error, coverage fraction —
   rather than on equality. A port that shifts a distribution is a bug; a port
   that shifts individual draws is expected.
6. "Reproducible across platforms" means **no draw may call the platform's C
   maths library.** See below; this is not free and it is not automatic.
7. The recommendation in 1 is about **cost**, not principle. Where reproducing
   numpy's stream is cheap and the benefit is concrete, reproduce it. See the
   amendment below.

## Amendment: bit-parity where it is cheap (2026-09-15, P4.3e)

Decision 1 says not to reproduce numpy's PRNG, and the Rationale gives the
reason plainly: bit-parity *"costs a hand-verified reimplementation of two
algorithms"* and *"would then have to be kept correct forever"*. That reasoning
was written about `build_observations`, which constructs a **fresh generator per
(node, target) pair per tick** and uses `SeedSequence.spawn`.

`procedural_terrain_grid` is not that shape. It creates **one** generator and
draws **12 to 46 scalars** from it — `uniform(low, high)` and nothing else. No
`spawn`, no arrays, no other distribution. A `SeedSequence` + `PCG64` port for
that path is about 130 lines, and it was verified against numpy on **24,000 of
24,000** draws across 2,000 seeds and ten ranges, then mutation-tested at
**38 of 38**.

So the amendment is narrow:

> Where a call site uses **one** generator, **no `spawn`**, and a **scalar-only**
> distribution surface, reproduce numpy's stream exactly and keep the goldens.
> Everywhere else, Decision 1 stands and the re-baseline applies.

**Why it is worth the 130 lines.** Python still bakes scene packages during the
transition. If Rust generated statistically-similar-but-different terrain for
the same seed, the `.smartscene` GLBs and the Rust runtime would describe
*different worlds*, and a replay recorded against one would be quietly wrong
against the other. An envelope cannot detect that; bit-parity makes it
impossible.

### The limit of parity is numpy's `exp`, not the PRNG

Worth recording, because it bounds what "bit-exact" can mean anywhere in this
migration, not only here.

`np.exp` is **not** the platform's `exp`. Measured over 300,000 inputs in the
ranges `_line_gaussian` uses, it differs from glibc's scalar `exp` by 1 ULP on
**4.4%–5.0%** of them, and `NPY_DISABLE_CPU_FEATURES` does not change that — it
is numpy's own implementation, not SIMD dispatch.

The consequence for terrain: **5.25% of cells differ, by at most 32 ULP and
5.7e-14 m**, while all six semantic masks are bit-identical on every cell of
every scene. The 5.25% matching the measured `exp` rate is what identifies the
cause rather than merely being consistent with it.

So the rule is: **the PRNG stream is bit-exact; floating-point results that pass
through a numpy transcendental are compared on a tolerance with a named cause
and a measured bound.** That is not the statistical envelope of Decision 5 —
it is a deterministic bound, and a guard requires a wrong port to miss it by a
factor of a million.

### Interaction with Decision 6

Decisions 6 and this amendment pull in opposite directions wherever a
transcendental appears, and it is better to say so than to leave it implicit:

* Matching **numpy** wants the platform's libm, since that is what numpy is
  mostly calling.
* Cross-platform reproducibility wants the `libm` crate, which calls nothing.

They cannot both hold. `argusnet-sim`'s draw stream takes Decision 6, because
nothing compares it to Python. `argusnet-world`'s terrain takes Python parity,
because the scene baker is Python and the two must agree. Neither is a mistake;
they are different contracts, and the parity fixtures for `argusnet-world` are
therefore **valid for one platform at a time**. If those fixtures are ever run
on a second platform, expect ULP-level differences and widen the comparison
rather than treating it as a regression.

### The other limit is BLAS, and it is not worth chasing (2026-09-15, P4.3f)

A second instance of the same shape, recorded because the conclusion goes the
other way from the PRNG amendment above and the difference is the point.

`OrientedBox.footprint_xy_m()` rotates four corners with `corners @ rot.T`.
numpy hands that to BLAS, which accumulates the two-term dot product with a
**fused multiply-add** — one rounding where a plain `x * cos - y * sin` has two.
The derived vertices differ from the Rust constructor's by up to 2.8e-14 m.

One `mul_add` reproduces it exactly; it is deliberately not used. Whether
numpy's matmul fuses is a property of the BLAS **vendor**, the matrix size and
the CPU's feature set — not of a published algorithm — so matching it would pin
this crate to one build of one library. That is the fragile bit-parity Decision 1
declines, and it is what distinguishes this case from PCG64, where the stream is
specified.

The divergence is accepted with a **stated mechanism for why it cannot matter**,
which is the standard this ADR asks for and not merely a tolerance: obstacle
footprints are consumed by `point_in_polygon`, which answers anything within
1e-6 m of an edge from its boundary short-circuit, identically under both
spellings. A 2.8e-14 m vertex shift is eight orders of magnitude inside that
band. `polygon_overlay_parity` carries a compile-time check that the tolerance
stays three orders of magnitude inside it, so the argument cannot be widened out
of existence by a later edit.

## "Across platforms" costs one dependency, and the cost is real

Added 2026-09-11, after the clause in Decision 1 was violated the first time CI
ran on something other than x86_64 Linux.

IEEE-754 requires `sqrt`, `fma` and the arithmetic operators to be **correctly
rounded** — one answer, everywhere, for every input. It requires nothing at all
of `ln`, `cos`, `exp` or any other transcendental, and implementations use that
freedom: glibc on x86_64 and Apple's libm on arm64 disagree by 1 ULP on some
inputs.

`DeterministicRng::normal` is a Box–Muller transform, so it multiplies `ln` and
`cos` results straight into the draw. Spelled `u1.ln()` it is a call into
whichever libm the binary happens to link, and the recorded stream becomes a
property of the build machine. The conformance fixture caught it the first time
CI ran on a Mac — element 0 of the very first normal draw, off by exactly one
ULP, and nothing else in the fixture moved.

The rule that follows is narrow and mechanical:

- Correctly-rounded operations (`+ - * /`, `sqrt`, `fma`, comparisons, integer
  work) stay on `f64`. They are already portable and the hardware is faster.
- Every transcendental in a stream whose values are a contract goes through the
  **`libm` crate** — a pure-Rust port of MUSL's libm with no platform calls,
  version-pinned exactly like the PRNG, because a version bump there can move
  the draws just as a platform change can. `libm`'s architecture-specific
  routines cover only `fma`, `rint` and `sqrt`, all correctly rounded, so its
  `log` and `cos` run identical generic code everywhere.

**No behavioural test on a single machine can enforce this.** `f64::ln` and
`libm::log` agree bit-for-bit on x86_64 glibc across the whole conformance
corpus — the mistake is invisible on the platform that would have to catch it,
and only becomes a red fixture on someone else's machine. So the rule is
enforced as a **source rule**:
`deterministic::tests::no_draw_routes_through_the_platform_libm` scans the
module and fails on any `f64` transcendental method. It is a source-text
assertion on purpose, and the alternative is no guard at all.

The "Tests Required" item below — *reproduces a run exactly, on more than one
platform* — was written at this ADR and first actually exercised here.

## Rationale

Bit-parity is achievable — `SeedSequence` and PCG64 are both specified — but it
costs a hand-verified reimplementation of two algorithms whose only purpose would
be to reproduce numbers nobody has a reason to prefer. It would then have to be
kept correct forever, including against future numpy changes, and it would
constrain the Rust implementation to numpy's exact draw *order* as well as its
values, which conflicts with the vectorisation and restructuring the port exists
to enable.

The property worth protecting is that a given seed reproduces a given run, not
that Rust reproduces Python.

## Consequences

### Positive
- The Rust implementation is free to restructure the tick loop, which is most of
  the point of ADR-005.
- No permanent obligation to track numpy's internals.

### Negative
- **Historical replays are not bit-comparable across the boundary.** Any archived
  replay compared against a fresh run will differ in individual values.
- The parity fixture and every golden file must be regenerated once, and that
  commit cannot be distinguished from a real behavioural regression by diff
  alone — hence the requirement that it change nothing else.
- Statistical comparison is weaker than equality and needs thresholds that are
  themselves a judgement call.

### Migration
1. Before porting a module, record its statistical envelope on the Python
   implementation across the canonical scenarios and seeds.
2. Port. Compare envelopes. Investigate any shift.
3. Regenerate `tests/golden/` and `tests/fixtures/runtime_parity_fixture.json` in
   an isolated commit citing this ADR.
4. Update both parity tests together — they are two halves of one contract.

The deterministic **work counters** in `tests/test_performance_regression.py`
(cache hit/miss/eviction totals, observation counts) are unaffected in kind: they
stay exact-comparison gates. Their values are re-baselined along with everything
else.

## Affected Modules

| Module | Change type |
|--------|------------|
| `src/argusnet/simulation/sim.py` (`build_observations`, `_stable_seed`) | ported |
| `tests/fixtures/runtime_parity_fixture.json` | regenerated |
| `tests/golden/` | regenerated |
| `tests/test_runtime.py`, `rust/argusnet-server/tests/grpc_parity.rs` | updated together |

## Tests Required

- Rust: same seed and configuration reproduces a run exactly, on more than one
  platform.
- Rust: reordering nodes or targets within a tick does not change the result —
  the property the per-pair seeding exists to guarantee.
- Cross-implementation statistical envelope comparison for the transition.

## References

- `src/argusnet/simulation/sim.py:4258` — the per-pair generator construction
- `docs/PERFORMANCE_AND_BENCHMARKING.md` §4 — `repeated_seed_diff_count` gate
- ADR-005 (Rust as the runtime authority)
