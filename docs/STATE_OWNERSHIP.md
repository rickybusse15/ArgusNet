# State Ownership Map

This document describes current ArgusNet state ownership. It intentionally uses the current
`src/argusnet` package, `argusnet-*` Rust crates, and split mapping/localization/planning modules.
Roadmap contracts belong in the subsystem docs; this file should reflect implemented ownership.

## Ownership Rules

- Runtime mutable state is owned by the subsystem that mutates it.
- Replay data is an immutable snapshot of runtime state.
- Rust `argusnet-core` owns fused object-state output and fusion health.
- Python simulation owns scenario construction, synthetic observations, mapping/localization mission
  state, and replay assembly.
- Mapping, localization, inspection, deconfliction, and scan mission replay types live in
  `argusnet.core.types` and are populated in `argusnet.simulation.sim`.
- The Rust viewer reads replay/scene data from disk; it is not a live authority for planning,
  safety, terrain, or fusion state.

## Current Runtime State

| State | Defined In | Written By | Read By | Boundary Notes |
|-------|------------|------------|---------|----------------|
| Terrain model features | `src/argusnet/world/terrain.py` | Terrain preset/build code | `world/procedural.py`, environment creation, tests | Analytic source for procedural terrain. |
| Terrain layer | `src/argusnet/world/environment.py` (`TerrainLayer`) | Scenario/environment builders, `build_terrain_layer()` | sim altitude clamping, LOS, replay/scene builders | Python runtime terrain authority; Rust terrain consumers use `terrain-engine` only where wired. |
| Terrain viewer mesh | Replay `meta.terrain.viewer_mesh`, scene packages | replay/scene builders | `rust/argusnet-viewer/src/replay.rs` and viewer state | Display artifact, not planning authority. |
| Obstacles | `src/argusnet/world/obstacles.py`, `EnvironmentModel.obstacles` | Scenario/environment builders | LOS, planner, collision/deconfliction, scene export | Python-owned geometry. |
| Land cover | `src/argusnet/world/environment.py`, `world/procedural.py` | Environment builders | visibility/sensing, replay/scene export | Seasonal masks are current; production tuning is scenario-specific. |
| Weather | `src/argusnet/world/weather.py` | Scenario builders | visibility/sensing and scenario metadata | Weather affects sensor factors; full vehicle dynamics effects remain partial. |
| Scenario config | `src/argusnet/simulation/sim.py`, `src/argusnet/core/config.py` | CLI/scenario builders | sim runtime, replay metadata, tests | `ScenarioOptions` and `SimulationConfig` are Python authorities. |
| Node state | `argusnet.core.types.NodeState` | Python simulation loop and ingestion adapters | Rust service request conversion, replay, viewer | Includes sensor capability and battery fraction fields for replay/sim use. |
| Bearing observations | `argusnet.core.types.BearingObservation` | Python observation generation / ingestion | Rust `TrackingService.ingest_frame()`, replay | Python creates observations; Rust owns fusion update. |
| Observation rejections | `argusnet.core.types.ObservationRejection` | Python generation and Rust runtime rejection conversion | replay/evaluation/viewer | `generation_rejections` are Python-side; `rejected_observations` are runtime frame data. |
| Truth state | `argusnet.core.types.TruthState` | Python simulation only | metrics/evaluation/replay | Not available to live runtime planning. |
| Fused object state | `argusnet.core.types.TrackState`, `rust/argusnet-core` | Rust `argusnet-core` through gRPC response | sim replay assembly, evaluation, viewer, planners that consume dynamic estimates | Rust is the runtime fusion authority. |
| Platform metrics | `argusnet.core.types.PlatformMetrics`, `rust/argusnet-core` | Rust `argusnet-core` | sim, replay, evaluation, viewer | Per-frame fusion/observation metrics. |
| Node health | `argusnet.core.types.NodeHealthMetrics`, `HealthReport`, `rust/argusnet-core` | Rust health monitor | service users, tests | Polled service state, not replay frame state unless explicitly exported. |
| Mission zones | `argusnet.core.types.MissionZone` | Scenario/mission generation | sim, planner/deconfliction, replay, viewer | Exclusion and objective zones are runtime constraints where supported, not only UI annotations. |
| Launch events | `argusnet.core.types.LaunchEvent` | Mission generation / scenario setup | sim/replay/viewer | Replayable mission event state. |

## Scan-Map-Inspect State

These current replay/runtime types are the main source of the updated mapping, localization, and
inspection documentation.

| State | Defined In | Populated By | Consumers | Notes |
|-------|------------|--------------|-----------|-------|
| `MappingState` | `argusnet.core.types` | `src/argusnet/simulation/sim.py` from `CoverageMap` / `WorldMap` | replay schema, viewer, evaluation/docs | Contains coverage fraction, covered cells, total cells, mean revisits. |
| `CoverageMap` | `src/argusnet/mapping/coverage.py` | `WorldMap.add_scan_observation()` and scan loop | frontier gap gate, POI rescoring, mapping metrics | Mutable mapping runtime owned by Python sim. |
| `WorldMap` | `src/argusnet/mapping/world_map.py` | scan loop | feature extraction, coverage region queries, mapping state | Current bridge toward a richer belief world. |
| `LocalizationState` | `argusnet.core.types` | `src/argusnet/simulation/sim.py` from `GridLocalizer` estimates | replay schema, viewer, docs | Aggregate active localizations, mean position std, mean confidence. |
| `LocalizationEstimate` | `argusnet.core.types` | `GridLocalizer.update()` results copied into `ScanMissionState` | replay/viewer/evaluation | Per-drone map-relative estimate for `scan_map_inspect`. |
| `GridLocalizer` | `src/argusnet/localization/engine.py` | Python sim loop | mission phase gate, replay state | Uses `LocalizationConfig.localization_timeout_steps`; timeout can force confidence for mission progress. |
| `InspectionPOI` | `argusnet.core.types` | scenario defaults / mission setup | `POIManager`, replay scan mission state | Map-relative point of interest for the current inspection runtime. |
| `POIStatus` | `argusnet.core.types` | `POIManager` | replay/viewer/evaluation | Status is `pending`, `active`, or `complete`. |
| `InspectionEvent` | `argusnet.core.types` | `src/argusnet/simulation/sim.py` mission-zone coverage loop | replay schema, viewer/evaluation | Includes entered/coverage/exited style events and current violation events. |
| `DeconflictionEvent` | `argusnet.core.types` | `src/argusnet/planning/deconfliction.py` | replay schema, viewer/evaluation | Captures lateral/vertical/corridor conflict resolutions. |
| `ScanMissionState` | `argusnet.core.types` | `src/argusnet/simulation/sim.py` | replay schema, viewer/evaluation | Current phases are `scanning`, `localizing`, `inspecting`, `egress`, `complete`. |
| `EgressDroneProgress` | `argusnet.core.types` | egress phase in `sim.py` | `ScanMissionState` / viewer | Reports per-drone return-home distance and home position. |

## Planning And Coordination State

| State | Owner | Runtime Status |
|-------|-------|----------------|
| `PathPlanner2D`, `PlannerConfig`, `PlannerRoute` | `src/argusnet/planning/planner_base.py` | Current 2D visibility-graph route planning. |
| `FrontierPlanner.find_gap_cells()` | `src/argusnet/planning/frontier.py` | Wired into `scan_map_inspect` as the scanning phase transition gate. |
| `FrontierPlanner.select_frontier_cell()` | `src/argusnet/planning/frontier.py` | Library capability; not currently called by `sim.py`. |
| `ClaimedCells` | `src/argusnet/planning/frontier.py` | Instantiated by `sim.py`; not currently used for runtime frontier selection. |
| `CoordinationManager.elect_coordinator()` | `src/argusnet/planning/coordination.py` | Wired into scan-map-inspect coordinator election. |
| `CoordinationManager.update_claimed()` / `flush_messages()` | `src/argusnet/planning/coordination.py` | Library capability; not currently called by `sim.py`. |
| `CoordinationManager.formation_offsets()` | `src/argusnet/planning/coordination.py` | Library capability; not currently called by `sim.py`. |
| `POIManager` | `src/argusnet/planning/poi.py` | Wired into inspection assignment, dwell completion, handoff/team assignment, and rescoring. |
| `DeconflictionPlanner` | `src/argusnet/planning/deconfliction.py` | Wired into current sim deconfliction events. |

## Evaluation, Replay, And UI State

| State | Defined In | Written By | Read By | Notes |
|-------|------------|------------|---------|-------|
| Replay document | `src/argusnet/evaluation/replay.py`, `docs/replay-schema.json` | simulation and replay helpers | evaluation, export, viewer | Current sim-to-viewer contract. |
| Evaluation report | `src/argusnet/evaluation/metrics.py` | `evaluate_replay()` | benchmark/report helpers, tests | Includes coverage, localization, mission completion, safety, and fusion metrics. |
| Benchmark suite state | `src/argusnet/evaluation/benchmarks.py` | benchmark runner callers | reports/tests | Aggregates `EvaluationReport` across seeds. |
| Export artifacts | `src/argusnet/evaluation/export.py` | CLI/export helpers | external tools | GeoJSON/CZML/Foxglove and related exports. |
| Viewer runtime state | `rust/argusnet-viewer/src/state.rs` and related viewer modules | viewer replay loading/playback | UI/rendering | Viewer state is derived from replay/scene files. |

## Cross-Boundary Contracts

| Boundary | Contract | Authority |
|----------|----------|-----------|
| Python sim -> Rust fusion service | `proto/argusnet/v1/world_model.proto` via `argusnet.adapters.argusnet_grpc` | Rust owns returned fused object states and metrics. |
| Python sim -> replay JSON | `src/argusnet/evaluation/replay.py` and `docs/replay-schema.json` | Python owns serialization; schema is additive. |
| Replay JSON -> Rust viewer | `rust/argusnet-viewer/src/replay.rs` / schema types | Viewer reads and displays; it does not mutate mission truth. |
| Python terrain -> Rust terrain helpers | No universal live boundary | `terrain-engine` is available for Rust-side consumers but not the Python sim authority. |
| Mapping/localization/inspection -> gRPC | Not currently in proto | These states are replay/runtime Python state today, not Rust gRPC state. |

## Current Gaps To Keep Visible

- `BeliefWorldModel` and `WorldBeliefQuery` are roadmap contracts; the current bridge is
  `CoverageMap`, `WorldMap`, and `MappingState`.
- Persistent inspection site records, evidence sets, reconstructions, and change records are roadmap
  inspection contracts; the current runtime uses `InspectionPOI`, `POIStatus`, and
  `InspectionEvent`.
- RF latency and formation coordination helpers exist as library code but are not wired into the
  current simulation loop.
- Rich localization status/covariance/pose-graph semantics are roadmap work; current replay state is
  aggregate confidence/std plus per-drone `LocalizationEstimate`.
- Mission execution has a skeleton package and a partial `scan_map_inspect` runtime path; the full
  closed-loop executive is still roadmap architecture.
