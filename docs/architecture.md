# Architecture & Module Map

Guide for engineers and AI agents. This document maps the current ArgusNet codebase to the
documentation set, identifies authoritative state owners, and separates implemented behavior from
roadmap architecture.

## Current Requirement Set

ArgusNet is a closed-loop, belief-world mission system. It should not be treated as only a
replay viewer, or simulation harness.

| Requirement document | Status | Role |
|----------------------|--------|------|
| `TERRAIN.md` | Current + roadmap | Runtime terrain query interface, terrain construction, and viewer/runtime boundary |
| `MAPPING.md` | Current + roadmap | Current coverage/world-map state plus roadmap belief-world contract |
| `LOCALIZATION.md` | Current + roadmap | Current grid localizer/VIO/relocalization modules plus roadmap pose-recovery contract |
| `INDEXING.md` | Roadmap contract | Spatial memory, keyframes, evidence, reconstructions, and cross-mission retrieval |
| `INSPECTION.md` | Current + roadmap | Current POI inspection runtime plus roadmap evidence/reconstruction contract |
| `MISSION_EXECUTION.md` | Roadmap contract with current bridge | Closed-loop runtime contract and the current `scan_map_inspect` bridge |
| `PLANNING.md` | Current + roadmap | Current planner modules, library capabilities, and planner-to-trajectory contract |
| `SAFETY.md` | Current + roadmap | Constraint validation, safety logging, and future blocking safety gates |
| `PERFORMANCE_AND_BENCHMARKING.md` | Current + standard | Current evaluation harness plus benchmarking and regression standards |

Current code wins when a roadmap contract and implementation disagree. Roadmap-only concepts must
be labeled as planned until the referenced code exists.

## Eight-Subsystem Architecture

```text
+------------------+     +------------------+     +------------------+
|  1. WORLD MODEL  |     | 2. SENSING/FUSION|     | 3. LOCALIZATION  |
| terrain, clutter |---->| sensors, fusion, |---->| pose, confidence |
| visibility, CRS  |     | observations     |     | relocalization   |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
| 4. MAPPING       |---->| 5. INDEXING      |---->| 6. INSPECTION    |
| coverage, world  |     | keyframes, data  |     | POIs, evidence   |
| map, belief      |     | spatial memory   |     | reconstruction   |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+
| 7. PLANNING /    |---->| 8. MISSION       |
| TRAJECTORY/SAFETY|     | EXECUTION / EVAL |
| routes, commands |     | replay, metrics  |
+------------------+     +------------------+
```

### Subsystem To Code Mapping

| Subsystem | Python modules | Rust crates | Docs |
|-----------|----------------|-------------|------|
| World Model | `src/argusnet/world/*` | `terrain-engine`, planned `world-engine` | `TERRAIN.md` |
| Sensing/Fusion | `src/argusnet/sensing/*`, `src/argusnet/adapters/argusnet_grpc.py` | `argusnet-core`, `argusnet-server`, `argusnet-proto` | `FUSION.md` |
| Localization | `src/argusnet/localization/*`, replay state in `argusnet.core.types` | planned `localization-engine` | `LOCALIZATION.md` |
| Mapping | `src/argusnet/mapping/*`, replay state in `argusnet.core.types` | planned `mapping-engine` | `MAPPING.md` |
| Indexing | `src/argusnet/indexing/*` | planned `indexing-engine` | `INDEXING.md` |
| Inspection | `src/argusnet/planning/poi.py`, `src/argusnet/planning/inspection.py`, inspection state in `argusnet.core.types` | planned `inspection-engine` | `INSPECTION.md` |
| Planning / Trajectory / Safety | `src/argusnet/planning/*`, `src/argusnet/simulation/behaviors.py`, `src/argusnet/safety/*` | `safety-engine`, planned `planner-engine` / `trajectory-engine` | `PLANNING.md`, `SAFETY.md` |
| Mission Execution / Evaluation | `src/argusnet/simulation/sim.py`, `src/argusnet/mission/*`, `src/argusnet/evaluation/*` | `argusnet-viewer`, planned runtime services | `MISSION_EXECUTION.md`, `SCENARIOS.md`, `PERFORMANCE_AND_BENCHMARKING.md` |

## Core Architectural Rule

Do not execute mission intent directly. All mission actions should pass through:

```text
mission intent
  -> candidate task
  -> candidate route/viewpoint
  -> trajectory proposal
  -> safety validation
  -> executable command
```

The current simulation still contains legacy direct motion paths. New work should either use the
pipeline or clearly document why it is bridging legacy behavior.

## Entry Points

- `src/argusnet/cli/main.py` - top-level CLI dispatcher.
- `src/argusnet/cli/sim.py` and `src/argusnet/simulation/sim.py` - simulation CLI and runtime.
- `src/argusnet/cli/scene.py` and `src/argusnet/world/scene_loader.py` - scene package build/load.
- `src/argusnet/evaluation/export.py` - replay export CLI.
- `src/argusnet/adapters/argusnet_grpc.py` - Python gRPC proxy to the Rust service.
- `proto/argusnet/v1/world_model.proto` - current gRPC service contract.
- `rust/argusnet-core/src/lib.rs` - native sensor-fusion math and lifecycle.
- `rust/argusnet-server/src/main.rs` - gRPC daemon.
- `rust/argusnet-viewer/src/main.rs` - Bevy viewer binary.

## Current Data Flow

The current implementation path is primarily simulation/replay oriented:

```text
CLI args -> ScenarioOptions / SimulationConfig
-> build_default_scenario()
-> build_observations() via EnvironmentQuery.los()
-> TrackingService.ingest_frame() [gRPC to Rust]
-> PlatformFrame (fused object states + metrics + optional mission state)
-> replay JSON
-> smartscene package
-> argusnet-viewer loads and renders
```

For `scan_map_inspect`, `src/argusnet/simulation/sim.py` also populates:

- `MappingState` from `CoverageMap` / `WorldMap`.
- `LocalizationState` and `LocalizationEstimate` from `GridLocalizer`.
- `InspectionEvent` from mission-zone coverage and violations.
- `DeconflictionEvent` from the deconfliction planner.
- `ScanMissionState` for `scanning -> localizing -> inspecting -> egress -> complete`.

## Target Data Flow

The required direction is a closed-loop mission runtime:

```text
Mission constraints / geofence
   -> Sensor ingestion
   -> LocalizationState update
   -> Mapping / belief-world update
   -> Indexing update and retrieval
   -> Planning / inspection task selection
   -> Trajectory proposal
   -> Safety validation
   -> Command execution
   -> Replay, metrics, evidence, and mission log
```

Ground truth may exist in simulation for scoring and synthetic observations, but planning,
localization, inspection, and safety must use belief state and localization state rather than
hidden truth in physical-mode tests.

## Python Module Map

| Module | Subsystem | Responsibility |
|--------|-----------|----------------|
| `src/argusnet/core/types.py` | Shared state | Frozen dataclasses for replay/runtime frames, mission zones, mapping/localization/inspection state |
| `src/argusnet/core/config.py` | Shared config | Simulation constants |
| `src/argusnet/core/frames.py` | Coordinates | WGS84/ECEF/ENU transforms and origin handling |
| `src/argusnet/world/environment.py` | World | Bounds, terrain/obstacle/land-cover layers, environment model, bundle helpers |
| `src/argusnet/world/terrain.py` | World | Analytic terrain models, features, presets |
| `src/argusnet/world/procedural.py` | World | Terrain build configuration and procedural/DEM/hybrid terrain creation |
| `src/argusnet/world/obstacles.py` | World | Obstacle primitives, collision, footprint logic |
| `src/argusnet/world/visibility.py` | World / Sensing | LOS queries over `EnvironmentModel` |
| `src/argusnet/world/weather.py` | World | Weather models |
| `src/argusnet/sensing/*` | Sensing | Sensor abstractions, noise/FOV/latency models, ingestion primitives |
| `src/argusnet/adapters/argusnet_grpc.py` | Sensing/Fusion | gRPC client proxy; `TrackingService.ingest_frame()` is the runtime fusion handle |
| `src/argusnet/localization/*` | Localization | Grid localizer, pose/VIO/relocalization/pose-graph foundations |
| `src/argusnet/mapping/*` | Mapping | Coverage map, world map, elevation/occupancy/semantic/uncertainty layers |
| `src/argusnet/indexing/*` | Indexing | Keyframe, landmark, tile, observation, and mission indexes |
| `src/argusnet/planning/planner_base.py` | Planning | `PathPlanner2D`, `PlannerConfig`, `PlannerRoute` |
| `src/argusnet/planning/frontier.py` | Planning | Frontier cell scoring and enclosed-gap detection |
| `src/argusnet/planning/coordination.py` | Planning | Coordinator election, claimed-cell message helpers, formation offsets |
| `src/argusnet/planning/poi.py` | Inspection planning | POI assignment, dwell tracking, handoff, team assignment, rescoring |
| `src/argusnet/planning/deconfliction.py` | Planning / Safety | Drone separation and corridor/exclusion-zone deconfliction |
| `src/argusnet/planning/inspection.py` | Mission generation | Mission spec generation and validation |
| `src/argusnet/simulation/sim.py` | Mission execution | Scenario construction, observation synthesis, scan-map-inspect runtime, replay generation |
| `src/argusnet/simulation/behaviors.py` | Trajectory | Drone and synthetic-object behavior helpers |
| `src/argusnet/evaluation/*` | Evaluation | Replay validation, export, metrics, benchmark aggregation, benchmark scenarios, performance summaries |
| `src/argusnet/mission/execution.py` | Mission execution | Mission loop skeleton and task model foundations |

## Rust Workspace

| Crate | Subsystem | Responsibility |
|-------|-----------|----------------|
| `argusnet-core` | Sensing/Fusion | Runtime sensor-fusion engine, IMM filter, GNN/JPDA association, rejection validation, health state |
| `argusnet-server` | Sensing/Fusion | gRPC daemon and config loading |
| `argusnet-proto` | Sensing/Fusion | Protobuf bindings and Rust conversion helpers |
| `argusnet-viewer` | Evaluation / UI | Bevy app, replay loading, terrain/mission-zone display, egui UI |
| `terrain-engine` | World | Rust terrain query/interpolation helpers |
| `safety-engine` | Safety | Rust constraint validator and monitor foundations |
| planned `world-engine` | World / Mapping | Obstacle/world query acceleration |
| planned `mapping-engine` | Mapping | Dense belief tile updates |
| planned `localization-engine` | Localization | Map-relative pose recovery and confidence state |
| planned `planner-engine` | Planning | Route/viewpoint/task planning from belief state |
| planned `trajectory-engine` | Trajectory | Feasible path generation and smoothing |
| planned `indexing-engine` | Indexing | Spatial memory and artifact retrieval |

## Environment And Belief-World Stack

Current implementation:

```text
world/terrain.py -> analytic TerrainModel and presets
world/procedural.py -> procedural / DEM / hybrid terrain builder
world/environment.py -> TerrainLayer / ObstacleLayer / LandCoverLayer / EnvironmentModel
world/visibility.py -> LOS queries over EnvironmentModel
planning/planner_base.py -> obstacle-aware local route search
mapping/coverage.py + mapping/world_map.py -> coverage and scan-derived world map state
evaluation/replay.py -> replay JSON
argusnet-viewer -> replay-driven rendering
```

Roadmap requirement:

```text
Prior world / geofence / live observations
   -> BeliefWorldModel
   -> WorldBeliefQuery
   -> Localization, planning, inspection, safety
```

`BeliefWorldModel` and `WorldBeliefQuery` are roadmap terms. The current runtime bridge is
`CoverageMap`, `WorldMap`, and `MappingState`.

## State Authority

| Concept | Authority | Details |
|---------|-----------|---------|
| Ground truth terrain/objects | Simulation only | Used for synthetic observations and scoring, not planner input in physical-mode tests |
| Prior terrain/geofence | World / Mission | Initial constraints and optional starting map |
| Runtime terrain queries | `argusnet.world.environment.TerrainLayer` | Python authority for sim LOS, altitude clamping, and terrain-derived replay data |
| Belief/coverage map | Mapping | Current implementation uses `CoverageMap`, `WorldMap`, and `MappingState` |
| Localization state | Localization | Current implementation uses `GridLocalizer`, `LocalizationEstimate`, and `LocalizationState` |
| Bearing observations | Python generated -> Rust filtered | Python creates observations; Rust performs runtime fusion acceptance/update |
| Fused object states | Rust `argusnet-core` | Single runtime source of truth for dynamic object estimates |
| Platform metrics | Rust `argusnet-core` | Computed per frame in current fusion path |
| Node health | Rust `argusnet-core` | Accumulated and polled on demand |
| Mission zones / geofence | Mission execution | Definitions live in `argusnet.core.types`; current sim uses them for replay, coverage, violations, and route constraints in supported paths |
| Inspection POIs | `argusnet.planning.poi` + `argusnet.core.types` | Current scan-map-inspect POI lifecycle and status authority |
| Replay document | Evaluation/UI | Current sim-to-viewer contract |

See `docs/STATE_OWNERSHIP.md` for the complete map.

## Coordinates And Units

- XY: meters in local projected/ENU frame.
- Z: meters above terrain datum or local frame as documented by the owning type.
- Angles: radians internally.
- `smartscene-v1`: meter-based scene package format.
- Multi-session and relocalization work should carry explicit frame IDs.

## Invariants

- `src/argusnet/world/environment.py` must continue re-exporting moved symbols for backward compatibility.
- Simulation is deterministic for fixed seed and config.
- Replay metadata changes should be additive unless viewer and tests are updated together.
- Physical collision must never push entities below terrain.
- Drones stay inside geofence/map bounds unless the mission constraints explicitly change.
- Rust `argusnet-core` is the source of truth for fused object-state output.
- Current mapping state is produced in `src/argusnet/simulation/sim.py`; roadmap physical-mode planning should use belief-world state.
- Localization state gates map-relative planning and inspection.
- Mission zones/geofences are safety constraints, not only viewer annotations.
- All new mission actions should pass through the feasibility/safety pipeline.
- Ground truth must not be used by planning, localization, inspection, or safety in physical-mode tests.

## Common Change Paths

### Add Or Change Belief-World Mapping

Edit `src/argusnet/mapping/*`, `src/argusnet/core/types.py`, `src/argusnet/simulation/sim.py`
where the current bridge is needed, plus `MAPPING.md`, `STATE_OWNERSHIP.md`, replay/viewer schema,
and truth-isolation or coverage tests.

### Add Or Change Localization Behavior

Edit `src/argusnet/localization/*`, `src/argusnet/core/frames.py`, `src/argusnet/core/types.py`,
replay/viewer schema, `LOCALIZATION.md`, and tests for startup/relocalization modes.

### Add Or Change Inspection Behavior

Edit `src/argusnet/planning/poi.py`, `src/argusnet/planning/inspection.py`,
`src/argusnet/planning/deconfliction.py`, relevant indexing adapters, replay/viewer schema,
`INSPECTION.md`, and inspection/evaluation tests.

### Add Or Change Mission Execution Behavior

Edit `src/argusnet/mission/*`, `src/argusnet/simulation/sim.py`, safety validation, replay/mission
logs, and `MISSION_EXECUTION.md`.

### Add Or Change Indexing Behavior

Edit `src/argusnet/indexing/*`, artifact schemas, keyframe/evidence persistence, retrieval tests,
and `INDEXING.md`.

### Add A Terrain Preset Or Terrain Source

Edit `src/argusnet/world/terrain.py`, `src/argusnet/world/procedural.py`,
`src/argusnet/simulation/sim.py`, `docs/TERRAIN.md`, and terrain tests.

### Change Observation Acceptance Or Rejection

Edit `src/argusnet/simulation/sim.py`, `src/argusnet/world/visibility.py`,
`src/argusnet/sensing/models/noise.py`, Rust prefiltering if needed, `FUSION.md`, and parity tests.

### Change Proto Schema

1. Edit `proto/argusnet/v1/world_model.proto`.
2. Rebuild Rust bindings through `rust/argusnet-proto/build.rs`.
3. Regenerate Python bindings under `src/argusnet/v1/`.
4. Update `src/argusnet/adapters/argusnet_grpc.py` serializers/deserializers.
5. Update `rust/argusnet-server` conversions.
6. Run Rust and Python proto/runtime tests.

## Architecture Decision Records

All nontrivial architecture changes require an ADR in `docs/adr/`.

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Separation of Visual and Analytic Terrain | proposed; partially implemented by terrain/viewer separation |
| 002 | Unified Fused Object State Authority | proposed; current runtime already uses Rust as fusion authority |
| 003 | Mission Intent Must Pass Through Feasibility Pipeline | proposed; current sim has partial safety logging and legacy paths |

Recommended new ADRs:

| Proposed ADR | Purpose |
|--------------|---------|
| Belief World as Planning Authority | Planning/safety use belief-world state, not simulation truth |
| Localization Gating for Inspection | Inspection routing requires localization confidence threshold |
| Persistent Spatial Memory | Indexing is the source of cross-mission retrieval |

## Debug Checklist

1. No observations -> check sensing, terrain, visibility, environment bounds.
2. Wrong fused object states -> check Rust runtime or protobuf/service conversion.
3. Bad map update -> check localization state, sensor pose, belief-world frame, and mapping uncertainty.
4. Planner uses impossible information -> check for accidental ground-truth access.
5. Replay correct but viewer wrong -> check Bevy viewer code and replay schema.
6. Constraint violation -> check `SAFETY.md`, mission geofence, and safety gate.
7. Inspection failed -> check localization confidence, LOS, viewpoint safety, evidence quality, and POI state.
8. Relocalization failed -> check indexing/keyframes, map region candidates, and coordinate frames.

## Benchmark Operations

Fast accepted performance baselines are stored in `tests/golden/performance/`. Pull request CI runs
the fast Python smoke benchmarks and compiles Rust Criterion benchmarks; `.github/workflows/nightly-bench.yml`
runs the slow multi-seed scenario sweep and uploads the generated benchmark artifacts.

## Related Documentation

| Document | Content |
|----------|---------|
| `TERRAIN.md` | Analytic terrain query interface and caching strategy |
| `MAPPING.md` | Belief-world mapping, geofence-bounded exploration, uncertainty handling |
| `LOCALIZATION.md` | Pose recovery, map-relative localization, relocalization modes |
| `INDEXING.md` | Spatial memory, keyframes, artifact retrieval, mission persistence |
| `INSPECTION.md` | Inspection POIs, multi-view capture, local reconstruction, change detection |
| `MISSION_EXECUTION.md` | Closed-loop runtime, task model, mission phases, safety-gated execution |
| `FUSION.md` | Fused object-state schema, lifecycle state machine, staleness rules |
| `SAFETY.md` | Drone physical limits, constraint validation, safety monitor |
| `MISSION_MODEL.md` | Mission generation schema, templates, difficulty scaling |
| `PLANNING.md` | Mapping, localization, inspection, and coordination planning |
| `SCENARIOS.md` | Evaluation scenarios and metric definitions |
| `PERFORMANCE_AND_BENCHMARKING.md` | Benchmark levels, data-layout, caching, CI, and regression standards |
