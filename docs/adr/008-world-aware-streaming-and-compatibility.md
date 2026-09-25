# ADR-008: World-Aware Streaming and Versioned Compatibility

> Timing extension: [ADR-009](009-independently-scheduled-runtime.md) requires
> component as-of times and freshness within coherent revision sets. A coherent
> stream does not imply simultaneous acquisition or grant control authority.

**Status:** accepted (architecture plan; no new RPC implemented)
**Date:** 2026-09-08
**Supersedes:** none

## Context

The current V2 frame stream has sequence/session/drop metadata but no complete
world/belief revision protocol. Mission-zone messages exist, yet MissionStatus
returns no stored zones. A live viewer cannot construct a coherent world from
frames alone, and a resume filter is not a durable replay service.

## Decision

Retain existing protobuf fields/RPCs and add a capability-negotiated world-aware
stream plus snapshot/world fetches. A session/epoch/sequence identifies frame
order; manifests, chunks and patches carry immutable world/belief revisions.
Missing bases, gaps or unsupported revisions trigger explicit resynchronization.
World transfer is bounded and separate from high-rate frame publication.

The server stores actual mission/world metadata at the declared authority;
clients install a complete verified world revision before presenting dependent
frames as coherent. Slow display subscribers may drop frames with telemetry;
event journals/recordings have separate loss semantics. Truth is filtered at the
boundary. TLS/loopback policy and signed ingest checks remain intact.

Use distinct versions for protobuf protocol, replay schema, C ABI capability,
checkpoint format, algorithm/seed contract and world generation. New glTF
packages declare spec-conforming axes with one loader transform; absent metadata
retains legacy Z-up package behavior. Do not change the meaning of an old field
or artifact to simplify new code.

The command protocol starts in-process. Remote control/physical actuation is not
part of a read-stream upgrade. Its authorization cannot be inferred from a
client subscribing to frames.

[Architecture and protocols](../migration/ARCHITECTURE_AND_PROTOCOLS.md) sections
2 and 7-9 define message semantics, limits and transition behavior. Exact proto
field numbers and dependency pins belong in the implementing, regenerated proto
PR, not in a prose promise that an RPC already exists.

## Consequences

- Live mode can show an attributable world and real mission status.
- Bounded transfer/revision state makes reconnect behavior explicit and testable.
- Additional protocol and backward-compatibility tests are required; V2 cannot
  be deleted at the same time the new stream is introduced.
- Offline replay, checkpoint restore and live resume remain distinct operations.

## Migration and tests required

G0/G1 inventory/regeneration precede wire changes. P4.0/P4.8 provide DTOs and
snapshot semantics; P5.3 implements new stream consumers. Test old/new clients,
missing/oversized/corrupt chunks, session restart, revision gaps, TLS, truth
capabilities and actual mission-zone state. Upgrade readers before defaulting
writers. See [verification and rollout](../migration/VERIFICATION_AND_ROLLOUT.md).

## References

- [Delivery plan](../migration/DELIVERY_PLAN.md)
- [ADR-007](007-runtime-composition-and-snapshots.md)
- [Runtime migration index](../RUST_RUNTIME_MIGRATION.md)
