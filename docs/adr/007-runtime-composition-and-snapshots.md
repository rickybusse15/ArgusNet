# ADR-007: Deterministic Runtime Composition and Snapshot Boundaries

> Extended by [ADR-009](009-independently-scheduled-runtime.md), 2026-09-23.
> The fixed stage order below remains the compatibility-port contract. The final
> runtime uses independently scheduled subsystems while preserving one owner,
> explicit causal order, complete checkpoints and coherent committed publication.

**Status:** accepted (architecture plan; implementation staged)
**Date:** 2026-09-08
**Supersedes:** none; operationalizes ADR-005 and retains ADR-006

## Context

World/safety now have Rust consumers, but the Python tick still owns mission,
localization and mapping state. Partial shared types and marker entities do not
provide a coherent mutable-state boundary. Porting each module independently
without a tick/ownership contract risks two mission authorities, hidden truth
inputs and render scheduling changing simulation results.

## Decision

A single `argusnet-sim::Simulation` owner composes domain crates. Batch callers
step it directly; interactive callers use a dedicated worker with acknowledged,
bounded commands and immutable committed snapshots. The renderer never owns a
second simulation. Domain crates do not depend on Bevy or transport code.

Preserve the legacy causal/clamp order in the first port, with machine-readable
trace fixtures before restructuring behavior. Planning uses a declared belief
capability; synthetic sensing and physical collision have separately scoped
truth access. Unknown data is explicit. Compatibility profiles expose historical
behavior instead of silently treating it as a new physical-mode guarantee.

Snapshots, full causal checkpoints and replay interchange are distinct types.
Checkpoint restore must reproduce uninterrupted execution, including RNG,
controller, estimator, safety and pending command state. Partial ticks are never
published. A faulted partially mutated run requires reset/restore before resuming.

Use `argusnet-localization` for estimator state, retain the existing shared kernel
in `argusnet-mapping`, and use `argusnet-replay` for long-lived DTO compatibility.
Types and conversion policies remain shared; a separate DTO is justified by
representation/compatibility, not permission to drop fields.

The normative details are [Architecture and protocols](../migration/ARCHITECTURE_AND_PROTOCOLS.md)
sections 1-6 and 8-9. Initial resource ceilings are configurable starting points
and must be measured before acceptance.

## Consequences

- One writer per concern and a responsive renderer despite slow ticks/builds.
- Explicit bounded memory, fault handling, command semantics and truth isolation.
- Checkpoint completeness and compatibility require more work than serializing
  visible frames; that work is a Phase 5 dependency.
- The current per-frame marker snapshot and Python/native estimator wrappers are
  transitional, not exceptions to the final authority model.

## Migration and tests required

P4.0/P4.1 establish types, trace and interfaces; P4.2-P4.8 port consumers;
P5.0/P5.2 connect the worker and checkpoints. Require direct/transport parity,
truth perturbation tests, command saturation/dedup, restore equivalence, epoch
interpolation resets and bounded lifecycle tests. Cross-platform determinism
remains governed by ADR-006 and the explicit gate in the verification plan.

## References

- [Delivery plan](../migration/DELIVERY_PLAN.md)
- [Verification and rollout](../migration/VERIFICATION_AND_ROLLOUT.md)
- [Current state ownership](../STATE_OWNERSHIP.md)
