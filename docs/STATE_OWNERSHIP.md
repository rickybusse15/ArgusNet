# State Ownership Map

## Reviewed state and target (2026-09-23)

The reviewed code baseline is `ec9c2f7` (including main through `45390b8`), not a
claim about the latest remote or release. Rust stage bodies, direct fusion,
belief publication and replay writing exist. Command recovery/dedup and format-2
control-state checkpoints exist locally in this baseline. Complete production
composition, subsystem reset/restore, full causal checkpoints and viewer-worker
acceptance remain open. No scheduler or aircraft qualification is claimed.

**Update 2026-10-03 (`main` at `0b1032c`).** P4.8d to P4.9d and viewer P2.0 to
P2.3 are merged (#122, `badb1e1`): the controller stage, world-backed stage 1,
the scenario bundle and native builder, the mission driver, the causal-state
audit of every assembled stateful stage, and an `argusnet-rs` CLI that builds
and runs all seven canonical scenarios with no Python (six within every
declared envelope; see [migration status](migration/STATUS.md)), owned
subsystem state in every stage (A1.2) and a production constructor that
rejects a missing required stage by name (A1.3). Reset equal to fresh and full
restore (A2), a durable event journal with crash recovery (A4.1), 30-seed
statistical qualification (A3.2) and the viewer's in-process simulation worker
(D1.1, D1.2) are merged too. Still open: the independent scheduler (P4.10:
scheduler not yet wired into `Simulation`, pending-job restore), cross-platform
determinism (A3.3) and the Python-free scene build (D2/D5). No scheduler or
aircraft qualification is claimed.

The [Rust-only runtime target](RUST_RUNTIME_ARCHITECTURE.md) now specifies
independent sensor/task deadlines, one session owner, timestamped measurements,
bounded workers and separate virtual/live time. P4.10a–f adds this after P4.9;
P5.0 basic controls ships alongside it. [Hardware integration](HARDWARE_INTEGRATION.md)
remains a separate S3–S7 qualification track.

This document describes current ArgusNet state ownership. It intentionally uses the current
`src/argusnet` package, `argusnet-*` Rust crates, and split mapping/localization/planning modules.
Roadmap contracts belong in the subsystem docs; this file should reflect implemented ownership.

Planned ownership and communication boundaries are specified separately in
[migration/ARCHITECTURE_AND_PROTOCOLS.md](migration/ARCHITECTURE_AND_PROTOCOLS.md).
This file continues to describe implemented behavior; accepted design is not a
claim that the existing `argusnet-sim` shell and domain ports compose a complete
mission, or that world-aware live RPCs exist.

## Historical integration status at `ba163ac`

This plan now includes the merged P4.3g landscape builders, P4.6d belief
publication and P4.8a–c mission/world/stage work. Evidence is recorded in
[landscape builders](migration/LANDSCAPE_BUILDERS.md),
[belief revisions](migration/BELIEF_REVISIONS.md),
[mission state](migration/MISSION_STATE.md),
[tick world](migration/TICK_WORLD.md) and [stage bodies](migration/TICK_STAGES.md).
Rust can publish populated platform/track snapshots from `TickWorld`; it is no
longer accurate to describe all snapshots as necessarily empty. The default
constructor still registers no domain handlers. Complete production scenario
composition, remaining controllers, full causal checkpoints, replay/CLI
qualification and viewer integration remain acceptance work. Existing slice
test reports are not a fresh full-system test result.

The inventory below is retained as the original review baseline, not current
completion status. P4.3 procedural landscape construction and P4.6 immutable
belief publication now exist; the active Python consumer boundary and complete
Rust mission acceptance must still be checked separately.

## Migration inventory at reviewed checkout `388f74b`

| Concern | Implemented library | Active consumer / integration boundary | Legacy retirement |
| --- | --- | --- | --- |
| World / safety | `argusnet-world`, `safety-engine` | Native Python runtime and viewer sampling | Python authoring/orchestration remain |
| Fusion | `argusnet-core` | `argusnet-server`, Python gRPC client | Direct Rust simulation call pending |
| Localization / sensing | `argusnet-localization`, `argusnet-sensing`; deterministic streams in `argusnet-sim` | Port fixtures exist; Python still runs mission orchestration | No broad runtime deletion |
| Mapping / planning | `argusnet-mapping`, `argusnet-planning` | Native scoring active; other ports await composed tick; belief query facade is not a revisioned mutable owner | Python mission consumers remain |
| Simulation | `argusnet-sim`: assembled eleven-stage production run (`ProductionBuilder`), `argusnet-rs sim` | Runs every canonical scenario with no Python; the Python tick remains the reference the replays are checked against | Python simulation still ships and is still what the viewer's Tools → Scenario launches |
| Replay / viewer | `argusnet-replay`, `argusnet-viewer` | Replay, V2 live presentation, and `--run` (D1.2): the Rust run in-process on a worker thread | Scene building still spawns Python (`spawn_build_scene_child`), except for `--world-knowledge none` runs, which need no scene (D2/D5 open) |

Evidence: [integration record](migration/P4_INTEGRATION.md),
[source review](NEXT_REFACTOR_REVIEW.md), and the referenced crates. Historical
integration test totals are not a fresh verification of this checkout.
The [viewer specification](VIEWER_UI.md) describes the target, while
[feature delivery](CAPABILITY_DELIVERY.md) assigns future feature ownership.

## Ownership Rules

- Runtime mutable state is owned by the subsystem that mutates it.
- Replay data is an immutable snapshot of runtime state.
- Rust `argusnet-core` owns fused object-state output and fusion health.
- Rust `argusnet-world` owns terrain, obstacle and visibility queries; `safety-engine`
  owns physical limits and simulation safety decisions. Python invokes them through
  persistent `argusnet-native` handles.
- Python simulation owns scenario construction, synthetic observations, mapping/localization mission
  state, and replay assembly.
- Mapping, localization, inspection, deconfliction, and scan mission replay types live in
  `argusnet.core.types` and are populated in `argusnet.simulation.sim`.
- The Rust viewer reads replay/scene data and live frame streams; it is not an authority for planning,
  safety, terrain, or fusion state.

## Current Runtime State

| State | Defined In | Written By | Read By | Boundary Notes |
|-------|------------|------------|---------|----------------|
| Terrain model features | `src/argusnet/world/terrain.py` | Terrain preset/build code | `world/procedural.py`, environment creation, tests | Preset/configuration data; analytic evaluation is owned by `argusnet-world`. |
| Terrain layer | `src/argusnet/world/environment.py` (`TerrainLayer`) | Scenario/environment builders, `build_terrain_layer()` | sim altitude clamping, LOS, replay/scene builders | Rust `argusnet-world` owns evaluation through `argusnet-native`; Python retains configuration/export metadata. |
| Terrain viewer mesh | Replay `meta.terrain.viewer_mesh`, scene packages | replay/scene builders | `rust/argusnet-viewer/src/replay.rs` and viewer state | Display artifact, not planning authority. |
| Obstacles | `src/argusnet/world/obstacles.py`, `EnvironmentModel.obstacles` | Scenario/environment builders | LOS, planner, collision/deconfliction, scene export | Rust `argusnet-world` owns geometry and indexing; Python retains metadata wrappers. |
| Land cover | `src/argusnet/world/environment.py`, `world/procedural.py` | Environment builders | visibility/sensing, replay/scene export | Rust owns categorical sampling and density queries; Python authors seasonal raster data. |
| Simulation safety state | `safety-engine::simulation`, Python `safety/native.py` wrappers | Native monitor retained per drone | Python simulation clamps, replay safety events | Rust policy/limits; Python retains opt-in orchestration and clamp ordering. |
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
| `GridLocalizer` | Python `localization/engine.py` state + Rust `argusnet-mapping` scoring | Python sim loop with native batched footprint queries | mission phase gate, replay state | Uses `LocalizationConfig.localization_timeout_steps`; timeout can force confidence for mission progress. |
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
| `FrontierPlanner.select_frontier_cell()` | `src/argusnet/planning/frontier.py` | Wired into `scan_map_inspect` scanning redirects behind `--frontier-exploration` (`sim.py`). The default scan remains the sector lawnmower. |
| `ClaimedCells` | `src/argusnet/planning/frontier.py` | Claimed/released per drone around `select_frontier_cell()` so two drones do not target the same cell (`sim.py`). |
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
| Persistent marker entities | `rust/argusnet-viewer/src/marker_meshes.rs` | Interpolated replay marker sync | Bevy mesh rendering | Typed identities survive motion/toggles; absent IDs despawn. |
| Viewer runtime state | `rust/argusnet-viewer/src/state.rs` and related viewer modules | viewer replay loading/playback | UI/rendering | Viewer state is derived from replay/scene files. |

## Cross-Boundary Contracts

| Boundary | Contract | Authority |
|----------|----------|-----------|
| Python sim -> Rust fusion service | `proto/argusnet/v1/world_model.proto` via `argusnet.adapters.argusnet_grpc` | Rust owns returned fused object states and metrics. |
| Python sim -> replay JSON | `src/argusnet/evaluation/replay.py` and `docs/replay-schema.json` | Python owns serialization; schema is additive. |
| Replay JSON -> Rust viewer | `rust/argusnet-viewer/src/replay.rs` / schema types | Viewer reads and displays; it does not mutate mission truth. |
| Python world -> Rust world | Versioned native C ABI, owned handles and batched f64 buffers | `argusnet-world` evaluates terrain/obstacles/LOS; `safety-engine` makes safety decisions. No Python fallback. |
| Mapping/localization/inspection -> gRPC | Not currently in proto | These states are replay/runtime Python state today, not Rust gRPC state. |

## Current Gaps To Keep Visible

- `WorldBeliefQuery` and P4.6d immutable belief revisions exist in Rust. Full
  production composition remains open; the existing Python consumer still uses
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
