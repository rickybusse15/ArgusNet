# ADR-002: Unified Fused Track as Authoritative Target State

**Status:** proposed
**Date:** 2026-03-15
**Author:** architecture-update
**Supersedes:** none

## Context

The current system has tracking logic in two places:
- `rust/tracker-core/src/lib.rs`: Kalman filter with GNN/JPDA association (authoritative per CLAUDE.md)
- `src/smart_tracker/fusion.py`: Python Kalman filter used for triangulation and local fusion

The architecture update (Section 7) requires a single authoritative fusion resource. Planners must consume fused tracks, not raw radar reports.

Currently, `tracker-core` maintains `TrackState` per track but the schema lacks explicit covariance representation in the protobuf interface, confidence scores, source history, and lifecycle state beyond basic confirmation/coasting.

## Decision

Define a unified `FusedTrack` schema as the single authoritative target belief state:

```
FusedTrack {
    track_id: u64,
    position: Vector3,          // meters
    velocity: Vector3,          // m/s
    acceleration: Option<Vector3>, // m/s², optional
    covariance_6x6: [f64; 36], // position+velocity covariance
    confidence: f64,            // 0.0–1.0
    source_ids: Vec<u64>,       // contributing sensor IDs
    last_seen: f64,             // simulation time
    lifecycle: TrackLifecycle,  // Tentative | Confirmed | Coasting | Lost
    hits: u32,                  // total measurement associations
    misses: u32,                // consecutive missed updates
}
```

Rust `tracker-core` remains the sole producer of `FusedTrack`. Python `fusion.py` becomes a utility for pre-processing (triangulation) before handing to Rust, not an alternative authority.

## Consequences

### Positive
- Single source of truth eliminates state drift between Python and Rust
- Planners get consistent, covariance-rich target estimates
- Track lifecycle is explicit and auditable

### Negative
- Proto file must be updated with new fields
- Python fusion.py role changes (may break tests relying on Python-only fusion)

### Migration
1. Extend `tracker.proto` with FusedTrack message
2. Update `tracker-core` to produce FusedTrack
3. Update `tracker-proto` conversions
4. Update `service.py` to consume FusedTrack
5. Refactor `fusion.py` to triangulation-only role
6. Update tests

## Affected Modules

| Module | Change type |
|--------|------------|
| `proto/smarttracker/v1/tracker.proto` | modified |
| `rust/tracker-core/src/lib.rs` | modified |
| `rust/tracker-proto/src/lib.rs` | modified |
| `src/smart_tracker/service.py` | modified |
| `src/smart_tracker/fusion.py` | modified (scope reduction) |
| `src/smart_tracker/models.py` | modified (new FusedTrack dataclass) |

## Tests Required

- FusedTrack schema round-trip through protobuf
- Track lifecycle state machine tests
- Planner receives FusedTrack (not raw observations)
- Python fusion.py triangulation-only tests

## References

- Architecture update Section 7: Sensing, Fusion, and Track Authority
- Architecture update Section 7.3: Architectural Rule
