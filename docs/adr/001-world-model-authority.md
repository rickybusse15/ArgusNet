# ADR-001: Separation of Visual and Analytic Terrain

**Status:** proposed
**Date:** 2026-03-15
**Author:** architecture-update
**Supersedes:** none

## Context

The current terrain stack (`terrain.py`, `environment.py`) serves both rendering and planning/sensing queries through the same data structures. The architecture update (Section 5.3) requires that visual terrain must not be the authoritative planning terrain. Planning and sensing must rely on a dedicated analytic terrain service with stable semantics and predictable performance.

Currently `TerrainModel` in `terrain.py` provides analytic height queries, while `TerrainTile` and `TerrainLayer` in `environment.py` provide tiled LOD terrain for both visual and analytic use. The `EnvironmentQuery.los()` path in `visibility.py` already uses analytic queries, but there is no formal contract separating the two roles.

## Decision

Introduce a `TerrainQuery` trait (Rust) and protocol (Python) that provides:
- `elevation_at(x, y) -> f64` — height above datum
- `slope_at(x, y) -> (f64, f64)` — gradient in x and y
- `los_clear(from, to) -> bool` — line-of-sight check
- `raycast(origin, direction, max_range) -> Option<HitResult>` — occlusion query
- `comms_shadow(from, to) -> f64` — communications attenuation factor

The analytic terrain service will be the authoritative source for planning, sensing, and constraint validation. The visual terrain (Bevy mesh, LOD pyramids) reads from the same underlying data but has no authority over planning decisions.

## Consequences

### Positive
- Planning and sensing get predictable, testable terrain queries
- Visual terrain can use lower LOD without affecting planning accuracy
- Terrain cache can be optimized independently for each use case

### Negative
- Two terrain representations must stay consistent
- Additional abstraction layer to maintain

### Migration
1. Extract `TerrainQuery` trait from existing `terrain.py` analytic functions
2. Implement Rust-side `TerrainQuery` in new `terrain-engine` crate
3. Route all planning/sensing queries through the trait
4. Visual terrain continues using `TerrainLayer` for rendering

## Affected Modules

| Module | Change type |
|--------|------------|
| `rust/terrain-engine/` | new crate |
| `src/smart_tracker/terrain.py` | modified (extract trait interface) |
| `src/smart_tracker/visibility.py` | modified (use trait) |
| `src/smart_tracker/planning.py` | modified (use trait) |
| `rust/tracker-viewer/` | modified (visual terrain only) |

## Tests Required

- Analytic terrain query unit tests (height, slope, LOS)
- Visual vs analytic consistency check
- Planning uses analytic terrain only (integration test)

## References

- Architecture update Section 5: World Model and Terrain Update
- Architecture update Section 5.3: Architectural Rule
