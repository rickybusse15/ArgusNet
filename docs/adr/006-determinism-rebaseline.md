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
