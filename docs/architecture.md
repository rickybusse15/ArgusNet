# Architecture & Module Map

Guide for engineers and AI agents. Covers module responsibilities, data flow, invariants, subsystem boundaries, and common change workflows.

## Six-Subsystem Architecture

The platform is organized into six subsystems. Each has a clear responsibility, authoritative state ownership, and defined interfaces to the others. See `docs/STATE_OWNERSHIP.md` for the full state ownership map.

```
+------------------+     +------------------+     +------------------+
|  1. WORLD MODEL  |     | 2. SENSING/FUSION|     | 3. MISSION GEN   |
|  terrain, clutter|---->| radar, sensors,  |     | scenarios, seeds, |
|  visibility, CRS |     | fused tracks     |     | objectives, zones |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
| 4. PLANNING      |<----| 5. TRAJECTORY    |<----| 6. EVALUATION    |
| roles, routes,   |     | physical limits, |     | metrics, replay, |
| replanning       |---->| safety monitor   |     | benchmarks       |
+------------------+     +------------------+     +------------------+
```

### Subsystem → Code Mapping

| Subsystem | Python modules | Rust crates | Docs |
|-----------|---------------|-------------|------|
| 1. World Model | `terrain.py`, `environment.py`, `obstacles.py`, `visibility.py`, `weather.py`, `environment_io.py` | `terrain-engine` (planned) | `TERRAIN.md` |
| 2. Sensing/Fusion | `sensor_models.py`, `fusion.py`, `service.py` | `tracker-core`, `tracker-server`, `tracker-proto` | `FUSION.md` |
| 3. Mission Gen | `sim.py` (scenario builders), `models.py` (MissionZone) | — (planned: `mission-gen`) | `MISSION_MODEL.md` |
| 4. Planning | `planning.py`, `sim.py` (controllers) | — (planned: `planner-engine`) | `PLANNING.md` |
| 5. Trajectory | `behaviors.py`, `sim.py` (FollowPathController) | — (planned: `trajectory-engine`, `safety-engine`) | `SAFETY.md` |
| 6. Evaluation | `replay.py`, `export.py` | `tracker-viewer` | `SCENARIOS.md` |

### Core Architectural Rule

> Do not execute mission intent directly. All mission actions must pass through:
> `mission intent → candidate route → feasibility validation → executable trajectory → safety monitor → execution`

See `docs/adr/003-mission-intent-pipeline.md`.

## Entry points

- `src/smart_tracker/cli.py` — CLI dispatcher (`sim`, `build-scene`, `ingest`, `export`)
- `src/smart_tracker/sim.py` — Scenario construction, observation synthesis, replay generation
- `src/smart_tracker/service.py` — Python gRPC proxy to Rust daemon (`TrackingService`)
- `proto/smarttracker/v1/tracker.proto` — Authoritative service contract
- `rust/tracker-core/src/lib.rs` — Native tracking math and lifecycle
- `rust/tracker-viewer/src/main.rs` — Bevy viewer binary

## Data flow

```
CLI args → ScenarioOptions / SimulationConfig
→ build_default_scenario()
→ build_observations() via EnvironmentQuery.los()
→ TrackingService.ingest_frame() [gRPC to Rust]
→ PlatformFrame (FusedTrack + metrics)
→ replay.py → replay JSON
→ scene.py → smartscene-v1 package
→ tracker-viewer loads and renders
```

## Python modules

| Module | Subsystem | Responsibility |
|--------|-----------|---------------|
| `sim.py` | 3, 4 | Scenario builders, observation generation, path planning, replay metadata |
| `service.py` | 2 | gRPC client proxy. `TrackingService.ingest_frame()` is the runtime handle |
| `environment.py` | 1 | Tiled terrain/obstacle/land-cover layers wrapped in `EnvironmentModel` |
| `terrain.py` | 1 | Analytic terrain model, features, presets |
| `obstacles.py` | 1 | Obstacle primitives, collision, footprint logic |
| `visibility.py` | 1 | LOS queries. `EnvironmentQuery.los()` is the single entry point |
| `planning.py` | 4 | Obstacle-aware 2D route planning, visibility-graph detours |
| `behaviors.py` | 5 | Target trajectory behaviors (loiter, transit, evasive, search, etc.) |
| `sensor_models.py` | 2 | Range-dependent noise, detection probability, atmospheric attenuation |
| `fusion.py` | 2 | Python Kalman filter (triangulation utility; Rust is authoritative for tracking) |
| `models.py` | all | Shared dataclasses (`NodeState`, `BearingObservation`, `TrackState`, `PlatformFrame`, `MissionZone`) |
| `config.py` | all | Simulation constants (`SensorConfig`, `DynamicsConfig`, `SimulationConstants`) |
| `weather.py` | 1 | Weather models (visibility, precipitation, wind) |
| `replay.py` | 6 | Replay JSON validation, writing, loading |
| `export.py` | 6 | GeoJSON, CZML, Foxglove MCAP, KML, Shapefile export |
| `scene.py` | 6 | `smartscene-v1` compiler (replay and GIS modes) |
| `_glb.py`, `_scene_*.py` | 6 | Internal scene compiler helpers |
| `ingest.py` | 2 | MQTT and file replay ingestion adapters |
| `coordinates.py` | 1 | WGS84/ECEF/ENU transforms |
| `environment_io.py` | 1 | Environment bundle serialization |

## Rust workspace

| Crate | Subsystem | Responsibility |
|-------|-----------|---------------|
| `tracker-core` | 2 | Fusion runtime: IMM Kalman filter, GNN/JPDA association, rejection validation, health tracking |
| `tracker-server` | 2 | gRPC daemon (`smart-trackerd`), config loading |
| `tracker-viewer` | 6 | Bevy app: scene loading, orbit camera, egui UI, replay playback |
| `tracker-proto` | 2 | Protobuf bindings and Rust conversion helpers |
| `terrain-engine` (planned) | 1 | Analytic terrain queries (`TerrainQuery` trait) |
| `trajectory-engine` (planned) | 5 | Feasible path generation, constraint validation |
| `safety-engine` (planned) | 5 | Safety monitor, abort/lost-link behaviors |

## Environment stack

```
terrain.py → analytic TerrainModel and presets
environment.py → tiled layers → EnvironmentModel
obstacles.py → geometry, point containment, push-out
planning.py → expanded footprints, local route search
visibility.py → LOS queries over EnvironmentModel
environment_io.py → bundle read/write
```

Flow: `sim.py` picks a terrain preset → `EnvironmentModel.from_legacy()` converts to tiled layers → `build_observations()` calls `EnvironmentQuery.los()` → collision-aware pathing uses `ObstacleLayer.point_collides()`.

**Visual vs Analytic terrain:** See `docs/adr/001-world-model-authority.md`. Visual terrain (viewer mesh) may differ in resolution from analytic terrain (planning/sensing queries). The `TerrainQuery` trait (docs/TERRAIN.md) formalizes the analytic interface.

## State authority

| Concept | Authority | Details |
|---------|-----------|---------|
| Terrain, obstacles, weather | Python | Static for simulation duration |
| Bearing observations | Python (generated) → Rust (filtered) | Split rejection ownership |
| Tracks (FusedTrack) | **Rust** | Single source of truth (ADR-002) |
| Platform metrics | **Rust** | Computed per frame |
| Node health | **Rust** | Accumulated, polled on demand |
| Mission zones | Python | Cosmetic only; enforcement planned |
| Replay document | Python (writes) → Rust viewer (reads) | Sole sim↔viewer interface |

See `docs/STATE_OWNERSHIP.md` for the complete map.

## Coordinates and units

- XY: meters in local projected map frame
- Z: meters above terrain datum
- Angles: radians (CLI flags explicitly labeled when degrees)
- `smartscene-v1`: meter-based

## Invariants

- `environment.py` must continue re-exporting moved symbols for backward compatibility
- Simulation is deterministic for fixed seed + config
- Replay metadata changes should be additive
- Physical collision never pushes entities below terrain
- Drones stay inside map bounds
- Rust is the source of truth for tracking output
- All mission actions pass through the feasibility pipeline (ADR-003)
- Fused tracks are the sole target belief consumed by planners (ADR-002)

## Common change paths

### Add a terrain preset
Edit: `terrain.py`, `sim.py`, `tests/test_terrain_features.py`, `tests/test_service.py`

### Add an obstacle primitive
Edit: `obstacles.py`, `visibility.py`, `environment_io.py`, `tests/test_collision.py`, `tests/test_environment.py`

### Change drone pathing
Edit: `planning.py`, `sim.py`. Test: `test_collision.py`, `test_service.py`, `test_sim.py`

### Change observation acceptance/rejection
Edit: `sim.py`, `visibility.py`. Inspect rejection constants, `build_observations()`, `SensorVisibilityModel`.

### Change live runtime behavior
Edit: `proto/tracker.proto`, `rust/tracker-core/src/lib.rs`, `rust/tracker-server/src/lib.rs`, `service.py`, `tests/test_runtime.py`

### Add a new Rust crate
1. Create `rust/<crate-name>/Cargo.toml` and `src/lib.rs`
2. Add to workspace `members` in root `Cargo.toml`
3. Wire dependencies from `[workspace.dependencies]`
4. Add to this architecture doc
5. Run `cargo test` to verify workspace builds

### Change proto schema
1. Edit `proto/smarttracker/v1/tracker.proto`
2. Rebuild Rust bindings (`cargo build` triggers `build.rs`)
3. Regenerate Python bindings (`protoc`)
4. Update `service.py` serializers/deserializers
5. Update `rust/tracker-server/src/lib.rs` conversions
6. Run both `cargo test` and `pytest`

## Architecture Decision Records

All nontrivial architecture changes require an ADR in `docs/adr/`. See `docs/adr/000-template.md`.

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Separation of Visual and Analytic Terrain | proposed |
| 002 | Unified Fused Track as Authoritative Target State | proposed |
| 003 | Mission Intent Must Pass Through Feasibility Pipeline | proposed |

## Debug checklist

1. No observations → check `sim.py`, `terrain.py`, `visibility.py`, environment bounds
2. Wrong tracks → check Rust runtime or protobuf/service conversion
3. Replay correct but viewer wrong → check Bevy viewer code
4. Constraint violation → check `DronePhysicalLimits` in `SAFETY.md`
5. Mission zone not enforced → zone enforcement not yet implemented (see `KNOWN_GAPS.md`)

## Related documentation

| Document | Content |
|----------|---------|
| `TERRAIN.md` | Analytic terrain query interface and caching strategy |
| `FUSION.md` | Fused track schema, lifecycle state machine, staleness rules |
| `SAFETY.md` | Drone physical limits, constraint validation, safety monitor |
| `MISSION_MODEL.md` | Mission generation schema, templates, difficulty scaling |
| `PLANNING.md` | Drone roles, planning objectives, planner-to-trajectory contract |
| `SCENARIOS.md` | Evaluation metrics, benchmark families, regression integration |
| `STATE_OWNERSHIP.md` | Runtime state ownership map with timing assumptions |
| `KNOWN_GAPS.md` | Gap analysis: what exists vs what the architecture requires |
| `CRITICAL_REVIEW.md` | Critical review of the architecture update plan |
| `SESSION_STATE.md` | Execution progress for resumability |
