# Architecture & Module Map

Guide for engineers and AI agents. Covers module responsibilities, data flow, invariants, subsystem boundaries, and common change workflows.

## Current Requirement Set

The current ArgusNet requirement set is a **closed-loop, belief-world mission system**. The project should not be treated as only a tracker, replay viewer, or simulation harness.

The current top-level requirement documents are:

| Requirement document | Role |
|----------------------|------|
| `TERRAIN.md` | Analytic terrain query interface and terrain/runtime boundary |
| `MAPPING.md` | Belief-world mapping, geofence-bounded exploration, uncertainty-aware world reconstruction |
| `LOCALIZATION.md` | Pose recovery, map-relative localization, relocalization after restart or battery swap |
| `INDEXING.md` | Spatial memory, keyframes, evidence, reconstructions, and cross-mission retrieval |
| `INSPECTION.md` | Map-relative inspection targets, multi-view capture, local reconstruction, repeat inspection |
| `MISSION_EXECUTION.md` | Closed-loop runtime mission coordination, task model, safety-gated execution |
| `PLANNING.md` | Drone roles, route planning, planner-to-trajectory contract |
| `SAFETY.md` | Safety validation, physical limits, abort/hold/return behavior |

Older subsystem docs remain useful implementation references, but the documents above now define the authoritative direction for new work.

## Eight-Subsystem Architecture

The platform is organized into eight cooperating subsystems. Each has clear responsibility, authoritative state ownership, and defined interfaces to the others. See `docs/STATE_OWNERSHIP.md` for the full state ownership map.

```text
+------------------+     +------------------+     +------------------+
|  1. WORLD MODEL  |     | 2. SENSING/FUSION|     | 3. LOCALIZATION  |
| terrain, clutter |---->| sensors, tracks, |---->| pose, covariance,|
| visibility, CRS  |     | observations     |     | relocalization   |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
| 4. MAPPING       |---->| 5. INDEXING      |---->| 6. INSPECTION    |
| belief world,    |     | spatial memory,  |     | targets, evidence|
| coverage, unknown|     | keyframes, data  |     | reconstructions  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+
| 7. PLANNING /    |---->| 8. MISSION       |
| TRAJECTORY/SAFETY|     | EXECUTION / EVAL |
| routes, commands |     | loop, replay     |
+------------------+     +------------------+
```

### Subsystem → Code Mapping

| Subsystem | Python modules | Rust crates | Docs |
|-----------|---------------|-------------|------|
| 1. World Model | `terrain.py`, `environment.py`, `obstacles.py`, `visibility.py`, `weather.py`, `environment_io.py` | `terrain-engine` / `world-engine` (planned) | `TERRAIN.md` |
| 2. Sensing/Fusion | `sensor_models.py`, `fusion.py`, `service.py` | `tracker-core`, `tracker-server`, `tracker-proto` | `FUSION.md` |
| 3. Localization | current replay/localization state plus planned localization modules | planned: `localization-engine` | `LOCALIZATION.md` |
| 4. Mapping | `src/argusnet/core/types.py`, coverage/mapping state, planned belief-world modules | planned: `mapping-engine` | `MAPPING.md` |
| 5. Indexing | planned keyframe/evidence/map tile stores | planned: `indexing-engine` or storage adapters | `INDEXING.md` |
| 6. Inspection | planned inspection target/evidence/reconstruction modules | planned: `inspection-engine` | `INSPECTION.md` |
| 7. Planning / Trajectory / Safety | `planning.py`, `behaviors.py`, `sim.py` controllers | planned: `planner-engine`, `trajectory-engine`, `safety-engine` | `PLANNING.md`, `SAFETY.md` |
| 8. Mission Execution / Evaluation | `sim.py`, `replay.py`, `export.py`, CLI orchestration | `tracker-viewer` plus planned runtime services | `MISSION_EXECUTION.md`, `SCENARIOS.md` |

### Core Architectural Rule

> Do not execute mission intent directly. All mission actions must pass through:
> `mission intent → candidate task → candidate route/viewpoint → trajectory proposal → safety validation → executable command`.

Mission execution is closed-loop: sensor ingestion updates localization and mapping; mapping updates the belief world; indexing stores reusable memory; planning chooses actions from that belief; safety validates commands; replay/evaluation records what happened.

See `docs/MISSION_EXECUTION.md` and `docs/adr/003-mission-intent-pipeline.md`.

## Entry points

- `src/smart_tracker/cli.py` — CLI dispatcher (`sim`, `build-scene`, `ingest`, `export`)
- `src/smart_tracker/sim.py` — Scenario construction, observation synthesis, replay generation
- `src/smart_tracker/service.py` — Python gRPC proxy to Rust daemon (`TrackingService`)
- `proto/smarttracker/v1/tracker.proto` — Current tracking service contract
- `rust/tracker-core/src/lib.rs` — Native tracking math and lifecycle
- `rust/tracker-viewer/src/main.rs` — Bevy viewer binary

## Current Data Flow

The current implementation path is still primarily simulation/replay oriented:

```text
CLI args → ScenarioOptions / SimulationConfig
→ build_default_scenario()
→ build_observations() via EnvironmentQuery.los()
→ TrackingService.ingest_frame() [gRPC to Rust]
→ PlatformFrame (FusedTrack + metrics)
→ replay.py → replay JSON
→ scene.py → smartscene-v1 package
→ tracker-viewer loads and renders
```

## Target Data Flow

The required direction is a closed-loop mission runtime:

```text
Mission constraints / geofence
   ↓
Sensor ingestion
   ↓
LocalizationState update
   ↓
BeliefWorldModel / mapping update
   ↓
Indexing update and retrieval
   ↓
Planning / inspection task selection
   ↓
Trajectory proposal
   ↓
Safety validation
   ↓
Command execution
   ↓
Replay, metrics, evidence, and mission log
```

Ground truth may exist in simulation for scoring, but planning and inspection must use the belief world and localization state, not hidden truth.

## Python modules

| Module | Subsystem | Responsibility |
|--------|-----------|---------------|
| `sim.py` | Mission execution, scenario generation | Current scenario builders, observation generation, path planning, replay metadata |
| `service.py` | Sensing/Fusion | gRPC client proxy. `TrackingService.ingest_frame()` is the runtime handle |
| `environment.py` | World Model | Tiled terrain/obstacle/land-cover layers wrapped in `EnvironmentModel` |
| `terrain.py` | World Model | Analytic terrain model, features, presets |
| `obstacles.py` | World Model | Obstacle primitives, collision, footprint logic |
| `visibility.py` | World Model / Sensing | LOS queries. `EnvironmentQuery.los()` is current single entry point |
| `planning.py` | Planning | Obstacle-aware 2D route planning, visibility-graph detours |
| `behaviors.py` | Trajectory | Target and drone trajectory behaviors |
| `sensor_models.py` | Sensing | Range-dependent noise, detection probability, atmospheric attenuation |
| `fusion.py` | Sensing/Fusion | Python Kalman/triangulation utility; Rust is authoritative for tracking |
| `models.py` | Shared | Dataclasses such as `NodeState`, `BearingObservation`, `TrackState`, `PlatformFrame`, `MissionZone` |
| `config.py` | Shared | Simulation constants and sensor/dynamics config |
| `weather.py` | World Model | Weather models |
| `replay.py` | Evaluation / UI | Replay JSON validation, writing, loading |
| `export.py` | Evaluation / UI | GeoJSON, CZML, Foxglove MCAP, KML, Shapefile export |
| `scene.py` | Evaluation / UI | `smartscene-v1` compiler |
| `ingest.py` | Sensing | MQTT and file replay ingestion adapters |
| `coordinates.py` | World / Localization | WGS84/ECEF/ENU transforms |
| `environment_io.py` | World | Environment bundle serialization |

## Rust workspace

| Crate | Subsystem | Responsibility |
|-------|-----------|---------------|
| `tracker-core` | Sensing/Fusion | Fusion runtime: IMM Kalman filter, GNN/JPDA association, rejection validation, health tracking |
| `tracker-server` | Sensing/Fusion | gRPC daemon (`smart-trackerd`), config loading |
| `tracker-viewer` | Evaluation / UI | Bevy app: scene loading, orbit camera, egui UI, replay playback |
| `tracker-proto` | Sensing/Fusion | Protobuf bindings and Rust conversion helpers |
| `terrain-engine` / `world-engine` (planned) | World / Mapping | Terrain and belief-world queries |
| `localization-engine` (planned) | Localization | Map-relative pose recovery and confidence state |
| `planner-engine` (planned) | Planning | Route/viewpoint/task planning from belief state |
| `trajectory-engine` (planned) | Trajectory | Feasible path generation and motion constraints |
| `safety-engine` (planned) | Safety | Safety monitor, abort/lost-link/return-home behavior |
| `indexing-engine` (planned) | Indexing | Spatial memory and artifact retrieval |

## Environment and belief-world stack

Current implementation:

```text
terrain.py → analytic TerrainModel and presets
environment.py → tiled layers → EnvironmentModel
obstacles.py → geometry, point containment, push-out
planning.py → expanded footprints, local route search
visibility.py → LOS queries over EnvironmentModel
environment_io.py → bundle read/write
```

Target requirement:

```text
Prior world / geofence / live observations
   ↓
BeliefWorldModel
   ↓
WorldBeliefQuery
   ↓
Localization, planning, inspection, safety
```

**Visual vs Analytic terrain:** See `docs/adr/001-world-model-authority.md`. Visual terrain may differ in resolution from analytic/belief terrain. Viewer meshes are display artifacts; planning, safety, and inspection must query the analytic/belief interfaces.

## State authority

| Concept | Authority | Details |
|---------|-----------|---------|
| Ground truth terrain/objects | Simulation only | Used for synthetic observations and scoring; not planner input in physical-mode tests |
| Prior world/geofence | Mission / World | Initial constraints and optional starting map |
| Belief world | Mapping | Authoritative planning input for unknown/known/unsafe regions |
| Localization state | Localization | Authoritative platform pose estimate and confidence |
| Bearing observations | Sensing/Python generated → Rust filtered | Split rejection ownership remains current implementation detail |
| Tracks (FusedTrack) | **Rust** | Single source of truth for tracked target state |
| Platform metrics | **Rust** | Computed per frame in current tracker path |
| Node health | **Rust** | Accumulated, polled on demand |
| Mission zones / geofence | Mission Execution | Hard constraints, not merely cosmetic |
| Inspection targets/evidence | Inspection + Indexing | Persistent map-relative mission artifacts |
| Replay document | Evaluation/UI | Current sim↔viewer interface; target runtime should also log mission/evidence artifacts |

See `docs/STATE_OWNERSHIP.md` for the complete map.

## Coordinates and units

- XY: meters in local projected map frame
- Z: meters above terrain datum
- Angles: radians internally
- `smartscene-v1`: meter-based
- Ground truth and belief-world layers must use explicit frame IDs when multiple sessions or relocalization are involved

## Invariants

- `environment.py` must continue re-exporting moved symbols for backward compatibility
- Simulation is deterministic for fixed seed + config
- Replay metadata changes should be additive
- Physical collision never pushes entities below terrain
- Drones stay inside geofence/map bounds unless operator explicitly changes mission constraints
- Rust is the source of truth for tracking output
- BeliefWorldModel is the source of truth for physical-mode planning input
- LocalizationState gates map-relative planning and inspection
- Mission zones/geofences are safety constraints, not only viewer annotations
- All mission actions pass through the feasibility/safety pipeline
- Fused tracks are the sole target belief consumed by planners for dynamic tracked objects
- Ground truth must not be used by planning, localization, inspection, or safety in physical-mode tests

## Common change paths

### Add or change belief-world mapping
Edit planned mapping modules plus `MAPPING.md`, `STATE_OWNERSHIP.md`, tests for truth isolation and coverage behavior.

### Add or change localization behavior
Edit localization modules plus `LOCALIZATION.md`, `coordinates.py`, replay/viewer schemas, and tests for startup/relocalization modes.

### Add or change inspection behavior
Edit inspection modules plus `INSPECTION.md`, indexing adapters, planner integration, viewer/replay schema, and evaluation metrics.

### Add or change mission execution behavior
Edit runtime loop/orchestration plus `MISSION_EXECUTION.md`, safety validation, task state, and replay/mission logs.

### Add or change indexing behavior
Edit index storage/query modules plus `INDEXING.md`, artifact schema, keyframe/evidence persistence, and retrieval tests.

### Add a terrain preset
Edit: `terrain.py`, `sim.py`, `tests/test_terrain_features.py`, `tests/test_service.py`.

### Add an obstacle primitive
Edit: `obstacles.py`, `visibility.py`, `environment_io.py`, `tests/test_collision.py`, `tests/test_environment.py`.

### Change drone pathing
Edit: `planning.py`, `sim.py`, and eventually planner/trajectory/safety modules. Test collision, safety, localization-confidence gating, and mission execution flow.

### Change observation acceptance/rejection
Edit: `sim.py`, `visibility.py`, `sensor_models.py`, and Rust prefiltering as needed. Preserve documented generation-vs-runtime rejection ownership.

### Change live runtime behavior
Edit: `proto/tracker.proto`, Rust runtime crates, `service.py`, mission execution modules, and `tests/test_runtime.py`.

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
5. Update Rust server conversions
6. Run both `cargo test` and `pytest`

## Architecture Decision Records

All nontrivial architecture changes require an ADR in `docs/adr/`. See `docs/adr/000-template.md`.

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Separation of Visual and Analytic Terrain | proposed |
| 002 | Unified Fused Track as Authoritative Target State | proposed |
| 003 | Mission Intent Must Pass Through Feasibility Pipeline | proposed |

Recommended new ADRs:

| Proposed ADR | Purpose |
|--------------|---------|
| Belief World as Planning Authority | Planning/safety use belief-world state, not simulation truth |
| Localization Gating for Inspection | Inspection routing requires localization confidence threshold |
| Persistent Spatial Memory | Indexing is the source of cross-mission retrieval |

## Debug checklist

1. No observations → check sensing, terrain, visibility, environment bounds
2. Wrong tracks → check Rust runtime or protobuf/service conversion
3. Bad map update → check localization state, sensor pose, belief-world frame, and mapping uncertainty
4. Planner uses impossible information → check for accidental ground-truth access
5. Replay correct but viewer wrong → check Bevy viewer code and replay schema
6. Constraint violation → check `SAFETY.md`, mission geofence, and safety gate
7. Inspection failed → check localization confidence, LOS, viewpoint safety, evidence quality, and target state
8. Relocalization failed → check indexing/keyframes, map region candidates, and coordinate frames

## Related documentation

| Document | Content |
|----------|---------|
| `TERRAIN.md` | Analytic terrain query interface and caching strategy |
| `MAPPING.md` | Belief-world mapping, geofence-bounded exploration, uncertainty handling |
| `LOCALIZATION.md` | Pose recovery, map-relative localization, relocalization modes |
| `INDEXING.md` | Spatial memory, keyframes, artifact retrieval, mission persistence |
| `INSPECTION.md` | Inspection targets, multi-view capture, local reconstruction, change detection |
| `MISSION_EXECUTION.md` | Closed-loop runtime, task model, mission phases, safety-gated execution |
| `FUSION.md` | Fused track schema, lifecycle state machine, staleness rules |
| `SAFETY.md` | Drone physical limits, constraint validation, safety monitor |
| `MISSION_MODEL.md` | Mission generation schema, templates, difficulty scaling |
| `PLANNING.md` | Drone roles, planning objectives, planner-to-trajectory contract |
| `SCENARIOS.md` | Evaluation metrics, benchmark families, regression integration |
| `STATE_OWNERSHIP.md` | Runtime state ownership map with timing assumptions |
| `KNOWN_GAPS.md` | Gap analysis: what exists vs what the architecture requires |
| `CRITICAL_REVIEW.md` | Critical review of the architecture update plan |
| `SESSION_STATE.md` | Execution progress for resumability |
