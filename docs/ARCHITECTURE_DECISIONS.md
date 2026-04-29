# Architecture Decisions — ArgusNet

This document records three explicit decisions that resolve contradictions identified in
`docs/CRITICAL_REVIEW.md`. All subsequent implementation work cites these decisions as
authoritative.

---

## Decision 1: Python Fusion Helpers vs. Rust Sensor Fusion Service

**Date:** 2026-03-22
**Status:** Accepted

### Context

`src/argusnet/mapping/fusion.py` and related localization utilities implement
a full IMM Kalman filter (`IMMTrack3D`, `ManagedTrack`, `CoordinatedTurnTrack3D`). The Rust
`argusnet-core` service is also a full sensor-fusion engine exposed via gRPC. Both produce fused
object estimates from bearing observations. The runtime simulation loop calls Rust via
`TrackingService.ingest_frame()`; the Python filter is not on the critical path.

### Decision

**Rust `argusnet-core` is the sole runtime fused-state producer.** Python fusion/localization helpers are
designated as:

1. A **test utility** — correctness reference for filter math (`test_fusion_advanced.py`).
2. A **triangulation/initialization helper** — `triangulate_bearings`, `fuse_bearing_cluster` may
   be used by Python code before passing observations to Rust.

**`IMMTrack3D` must not be instantiated from production code paths.** It carries a
`DeprecationWarning` guard to enforce this.

### Consequences

- Any code reading fused object state must read from the Rust gRPC response, not
  from a Python `ManagedTrack`.
- `test_fusion_advanced.py` tests are correctness reference tests, not integration tests.
- The Python filter may be removed in a future cleanup phase after test coverage migrates to Rust.

---

## Decision 2: Waypoints-as-Advisory vs. Direct Execution in `FollowPathController`

**Date:** 2026-03-22
**Status:** Accepted

### Context

ADR-003 requires all drone motion to flow through:

```
mission intent → candidate route → feasibility validation → executable trajectory → safety monitor → execution
```

`FollowPathController.__call__()` in `simulation/sim.py` directly computes position, velocity, and
terrain-clamped Z at each timestep. It does not pass through the `safety-engine` Rust crate. The
`DronePhysicalLimits` and `ConstraintValidator` from `rust/safety-engine/` are not wired into the
Python loop.

### Decision

**Adopt Posture A for Phases 1–5:** The safety engine validates but does NOT block motion.

Specifically:

- A Python `DroneConstraintChecker` (in `src/argusnet/safety/checker.py`) mirrors the
  `DronePhysicalLimits` from Rust and is called after each drone step.
- Violations are **logged** to the replay JSON as `safety_events` but do not alter motion.
- `FollowPathController` behavior is grandfathered via a `legacy_mode` flag when the full pipeline
  is introduced.

Phase 6 (and future work) will enforce the full ADR-003 pipeline. Until then, new drone-motion
code must:
1. Call `DroneConstraintChecker.check_state()` after each commanded step.
2. Log violations — do not silently ignore them.
3. Not introduce new blocking constraint logic outside the Rust safety engine.

---

## Decision 3: Single `TerrainLayer` Serving Physical + LOS + Visual Roles

**Date:** 2026-03-22
**Status:** Accepted

### Context

ADR-001 proposes separating visual terrain (for rendering) from analytic terrain (for LOS
raycasting and altitude clamping). Currently `TerrainLayer` in `argusnet.world.environment` serves
all three roles. The Rust `terrain-engine` crate exists but is not depended on by any other crate.

### Decision

**Formalize the current reality as the boundary:**

| Role | Authority |
|------|-----------|
| Physical altitude clamping in Python sim loop | Python `TerrainLayer.height()` |
| LOS raycasting in Python visibility queries | Python `TerrainLayer` + `visibility.py` |
| Visual mesh for the viewer | Replay JSON `meta.terrain.viewer_mesh` (pre-baked by scene builder) |
| New Rust-side consumers (safety validator, forward prediction) | `terrain-engine` crate |

**The viewer never routes rendering through Python analytic APIs.** The `viewer_mesh` in
`replay.rs` `TerrainViewerMesh` is the viewer's terrain data source. This separation already holds
in practice; this decision formalizes it so no future viewer work accidentally calls Python terrain
queries.

The `terrain-engine` crate is reserved for Rust-side analytic consumers. No changes to the Python
`TerrainLayer` API are required.

---

## Summary

| Decision | Rule |
|----------|------|
| 1 | Rust is sole runtime fused-state producer; Python Kalman = test utility |
| 2 | Safety engine logs violations, does not block (Posture A through Phase 5) |
| 3 | Viewer uses replay `viewer_mesh`; Rust crate serves new Rust consumers only |
