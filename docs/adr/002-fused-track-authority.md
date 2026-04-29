# ADR-002: Rust Sensor Fusion Output Authority

**Status:** proposed
**Date:** 2026-03-15
**Author:** architecture-update

## Context

ArgusNet has Python helper code and a Rust runtime service that can both reason over observations.
The current product direction is map, localize, inspect, revisit, and coordinate drones. Sensor
fusion is a support layer for that workflow.

## Decision

Rust `argusnet-core` remains the runtime authority for fused object-state output returned across the
gRPC boundary. Python fusion helpers are reference/preprocessing utilities unless an ADR approves a
different runtime path.

## Consequences

- Mapping, localization, and inspection consumers read fused state from `PlatformFrame`.
- Python helpers may support tests or preprocessing, but should not create a parallel runtime
  authority.
- Future schema migrations should prefer product-neutral names such as object state, observation,
  POI, map region, and evidence.

## Affected Modules

| Module | Change type |
|--------|-------------|
| `proto/argusnet/v1/world_model.proto` | modified only during explicit schema migrations |
| `rust/argusnet-core/src/lib.rs` | runtime fusion authority |
| `rust/argusnet-proto/src/lib.rs` | conversion boundary |
| `src/argusnet/adapters/argusnet_grpc.py` | Python service adapter |
| `src/argusnet/core/types.py` | replay/runtime dataclasses |

## Tests Required

- Protobuf round-trip tests.
- Runtime parity tests.
- Tests that mapping/localization/inspection consumers use the service boundary rather than a
  parallel Python authority.
