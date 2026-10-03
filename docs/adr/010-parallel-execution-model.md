# ADR-010: Parallel execution model for the scheduled runtime

**Status:** proposed
**Date:** 2026-10-03
**Author:** ArgusNet maintainers (drafted by an agent; not accepted until reviewed)
**Extends:** ADR-009; retains ADR-006

## Context

ADR-009 replaces the single-rate eleven-stage tick with an integer-time
scheduler: independent sensor and task deadlines, a fixed phase order at one
timestamp, bounded workers, and a determinism contract that worker count,
pacing, render rate and recording must not change committed state. It does
not say how work inside one phase may run concurrently, how multi-rate lanes
may overlap, or which kinds of rate change are allowed in deterministic
simulation.

[PARALLEL_TICK_PLAN.md](../migration/PARALLEL_TICK_PLAN.md) proposes those
rules. They need a durable decision before P4.10a.2 (fork-join inside phases)
and P4.10d (background jobs) are built, because both change the executor's
concurrency contract and every later scheduler slice depends on it.

Three facts constrain the choice:

- Observation draws are already order-independent: ADR-006 keys them by
  `(node, target, tick)`. Per-platform estimator streams have not been
  audited for the same property.
- Ten stages still share state with external callers through
  `Arc<Mutex<Shared>>` links (`rust/argusnet-sim/src`), so write sets are not
  yet provably disjoint.
- No Rust per-stage profile exists. The only profile is from the Python
  runtime and does not rank Rust costs.

## Decision

1. **Phases are barriers.** At one timestamp the phases run in ADR-009's order.
   Inside a phase, tasks whose write sets are disjoint state partitions may run
   concurrently. Their results are merged **in canonical order** (stable
   source id, then sequence), never in completion order.
2. **One executor path.** A session owns one bounded compute pool (not a
   global pool). `workers = 1` runs the same scheduler serially, and that
   serial result is the reference every parallel run must equal.
3. **No unordered floating-point reduction.** Every reduction that feeds
   committed state combines partial results in a fixed order (by tile,
   platform or source id).
4. **Randomness is keyed, not sequenced.** A task that draws random numbers
   uses a stream keyed by its identity and tick (as ADR-006 does for
   observations). A stage whose stream is shared may not be parallelised
   until a separately versioned slice re-keys it.
5. **Background work is revisioned.** A job takes an immutable input revision,
   returns a proposal tagged with it, and is applied only at a recorded
   application boundary. If the proposal is not ready, simulated time waits
   while commands stay serviceable. Stale revisions are rejected, and safety
   revalidates any applied trajectory.
6. **Three rate mechanisms, with different determinism status:**

   | Mechanism | Deterministic simulation | Live |
   | --- | --- | --- |
   | Static schedule (fixed or rational rate) | allowed | allowed |
   | Data-triggered (batch size, max age, state change, expiry deadline) | allowed; the trigger reason is logged | allowed |
   | Load-adaptive (skip or slow work under CPU pressure) | **forbidden**: wall-clock load lowers achieved speed only | only under a qualified fallback policy, recorded as an event |

7. **Parallelism is earned by measurement.** A stage is parallelised only when
   a published Rust profile predicts a tick-latency gain larger than
   run-to-run noise on named hardware. Below a size threshold the serial path
   is used. "Not parallelised, with the reason recorded" is an acceptable
   outcome.
8. **`legacy_stages_v1` is unchanged.** Everything here applies to
   `scheduled_v1` only, which stays opt-in until P4.10f qualification and a
   human promotion decision.

## Consequences

### Positive
- Worker count and completion order cannot change results, so parallel runs
  stay replayable and checkpoints stay comparable.
- One execution path: the serial run is the oracle, and no separate parallel
  code path needs its own acceptance.
- Load-adaptive behaviour cannot leak into deterministic simulation.

### Negative
- Ordered merges and keyed streams cost some throughput compared with
  unordered parallelism.
- Small fleets may see no gain. The measurement rule may leave most stages
  serial.
- Removing the shared stage links (A1 in the agent execution plan) becomes a
  prerequisite for parallel phases, not an optional cleanup.

### Migration
- P4.10-pre: per-stage timers and work counters, and a published Rust profile.
- P4.10c, pulled forward: work reduction before parallelism.
- P4.10a: serial scheduler with phase barriers; serial-equivalence gate against `legacy_stages_v1`.
- P4.10a.2: fork-join inside phases behind `workers = n`.
- P4.10b, then P4.10d to P4.10f, as in [DELIVERY_PLAN.md](../migration/DELIVERY_PLAN.md).

## Affected Modules

| Module | Change type |
|--------|------------|
| `rust/argusnet-sim` (scheduler, executor, compute pool) | new |
| `rust/argusnet-sim/src/{vio,observe,mapping,controller,battery}.rs` | modified (keyed streams, ordered merge) |
| `rust/argusnet-mapping` (footprint reduction) | modified |
| `docs/RUST_RUNTIME_ARCHITECTURE.md` §3/§5 | modified when accepted |

## Tests Required

- Identical committed state and replay at 1, 2 and N workers.
- A test executor with randomized task delays and reversed completion order.
- A mutation test that swaps a reduction's combine order and must fail.
- Serial equivalence: `scheduled_v1` at legacy rates with one worker equals `legacy_stages_v1`.
- Stale-revision rejection, and restore with pending jobs equal to an uninterrupted run.
- Below the size threshold, the serial path is taken.

## References

- [ADR-009](009-independently-scheduled-runtime.md), [ADR-006](006-determinism-rebaseline.md)
- [PARALLEL_TICK_PLAN.md](../migration/PARALLEL_TICK_PLAN.md) §4–§9
- [RUST_RUNTIME_ARCHITECTURE.md](../RUST_RUNTIME_ARCHITECTURE.md) §3–§5
- [AGENT_EXECUTION_PLAN.md](../migration/AGENT_EXECUTION_PLAN.md) §5 (WS-B)
