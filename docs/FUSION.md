# Sensor Fusion State Contract

This document describes the Rust sensor-fusion boundary that converts observations into fused
object-state estimates for replay, evaluation, and downstream planning. The map/localize/inspect
workflow should treat this as a low-level sensing service, not as the product framing.

## Current Boundary

- Python simulation or ingestion produces node states and bearing observations.
- `src/argusnet/adapters/argusnet_grpc.py` sends each frame to the Rust service.
- `rust/argusnet-core` performs filtering, association, rejection validation, and health updates.
- Python receives fused object-state records and per-frame metrics in `PlatformFrame`.
- Mapping, localization, and inspection logic should consume only the fused state needed for the
  current workflow.

## Authority Rules

- Rust `argusnet-core` is the runtime authority for fused object-state output.
- Python fusion helpers are reference or preprocessing utilities unless explicitly wired through an
  approved architecture decision.
- Replay fields should remain additive and backward-compatible.
- Planning should not depend on raw observations when a fused state is available.

## Current Python Types

The current runtime still uses established code names in `argusnet.core.types`:

- `BearingObservation`
- `ObservationRejection`
- `TrackState`
- `PlatformMetrics`
- `PlatformFrame`

These names are implementation details of the current service boundary. Product docs should frame
the system around mapping, localization, inspection, POIs, and spatial memory.

## Consumer Expectations

Consumers should expect each fused object-state record to provide:

- stable ID;
- timestamp;
- position and velocity;
- covariance or uncertainty;
- update count;
- stale count;
- lifecycle/quality when available.

Consumers should handle missing or stale fused state by degrading confidence, requesting additional
observations, or holding/returning according to mission safety policy.

## Roadmap

1. Keep Rust as the single runtime authority for fused object-state output.
2. Clarify replay and proto field names when a schema migration is explicitly approved.
3. Add mapping/localization/inspection metrics that do not depend on old scenario framing.
4. Keep sensor-fusion benchmarks under `PERFORMANCE_AND_BENCHMARKING.md`.
