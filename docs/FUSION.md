# Sensor Fusion State Contract

This document describes the Rust sensor-fusion boundary that converts observations into fused
object-state estimates for replay, evaluation, and downstream planning. The map/localize/inspect
workflow should treat this as a low-level sensing service, not as the product framing.

## Scheduled Rust integration (target, 2026-09-23)

Fusion remains solely in `argusnet-core`, called directly by the Rust runtime.
Measurement arrivals and prediction/expiry deadlines are separate triggers;
no-observation intervals must still age tracks. Preserve acquisition time,
source/frame/calibration identity and uncertainty through ingestion. Unsupported
late/out-of-order measurements are explicitly rejected, never retimestamped.
Prediction/correction history and association state belong in full checkpoints.
Internal batching must preserve temporal order and existing weighting semantics.

The execution contract is [Rust runtime architecture](RUST_RUNTIME_ARCHITECTURE.md);
[delivery](migration/DELIVERY_PLAN.md), [verification](migration/VERIFICATION_AND_ROLLOUT.md)
and [hardware integration](HARDWARE_INTEGRATION.md) define promotion gates.

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

## Track identity, lifecycle and time contract (target)

A platform carries sensors; a scenario target describes an observed object; an
observation is evidence; a track is an estimate with its own lifecycle. An
inspection POI is a work/evidence objective, not automatically a track. Preserve
these typed identities rather than matching them by a shared display label.
Current `argusnet-core` lifecycle names are `tentative`, `confirmed`, `coasting`
and `deleted`; expose supported states explicitly through future contracts.

Track identity is scoped to run/source epoch, stable while that track lives and
never silently reused within the epoch. Reacquisition after deletion creates a
new track; any relationship to an old track is explicit evidence, not an inferred
identity guarantee. If merge/split is later supported, emit lineage events and
preserve historical references; do not pretend those operations exist today.
Deletion publishes a tombstone/event so skipped display frames do not leave
phantom tracks or silently transfer selection to a new object.

Separate last measurement acquisition time, filter state/prediction time and
publication time. Predictions must not refresh the age of supporting evidence.
Covariance, contributing source/sample identities and provenance refer to the
stated estimate/frame. Unknown quality is not zero uncertainty; a scalar quality
score is not automatically a calibrated probability. Avoid double-counting the
same relayed measurement or correlated shared estimate.

For scheduled execution, track expiration uses elapsed source time. Confirmation
windows identify actual accepted observation opportunities rather than viewer
frames or arbitrary scheduler wakeups. Retain frame-count semantics only in the
legacy compatibility profile; conversion requires a versioned behavior test.
Missing inputs still trigger prediction, coasting and expiry deadlines.

Operational association cannot depend on hidden truth identity in a profile
claiming unlabeled perception. Preserve explicitly labeled/cooperative inputs as
a separate source capability. Evaluation owns truth matching and reports false
tracks, identity switches, fragmented/reacquired tracks, error and calibration.

Acceptance: missed observations at unequal rates, duplicate relay, out-of-order
inputs under the declared policy, ambiguous crossing trajectories, reacquisition,
expiry without rendering, tombstone loss/resync and checkpointed lifecycle state.

## Roadmap

1. Keep Rust as the single runtime authority for fused object-state output.
2. Clarify replay and proto field names when a schema migration is explicitly approved.
3. Add mapping/localization/inspection metrics that do not depend on old scenario framing.
4. Keep sensor-fusion benchmarks under `PERFORMANCE_AND_BENCHMARKING.md`.

## Next delivery integration (planned)

Keep Rust fusion as the sole operational fused-state producer and call it
directly in the composed simulation. Independent evaluation matches unlabeled
tracks to truth and reports unmatched objects, false tracks, identity changes
and covariance calibration. F1/F4 consume these scores; operational viewers do
not receive truth-derived metrics without the evaluation capability.

See [CAPABILITY_DELIVERY.md](CAPABILITY_DELIVERY.md) for F1–F7 dependencies and
acceptance, [ROADMAP.md](ROADMAP.md) for sequencing, and
[VIEWER_UI.md](VIEWER_UI.md) for the consuming operator workflows.
