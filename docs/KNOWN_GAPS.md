# Known Gaps — ArgusNet

This document records current implementation status and unresolved gaps. It replaces older
Smart-Trajectory-Tracker-era wording with the current ArgusNet module layout.

Status labels:

- **Implemented**: wired into the current runtime or supported by tested modules.
- **Partial**: code exists, but integration, validation, or production behavior is incomplete.
- **Planned**: documented roadmap architecture with little or no runtime wiring.

## World And Terrain

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Analytic terrain presets | **Implemented** | `src/argusnet/world/terrain.py` |
| Runtime terrain layer | **Implemented** | `src/argusnet/world/environment.py` `TerrainLayer` |
| Procedural / DEM / hybrid terrain construction | **Implemented** | `src/argusnet/world/procedural.py`, `TerrainBuildConfig`, `build_terrain_layer()` |
| Terrain batch/cached query path | **Implemented** | `TerrainLayer.height_at_many()` and viewer mesh caching |
| Obstacle primitives and collision checks | **Implemented** | `src/argusnet/world/obstacles.py`, `EnvironmentModel` |
| Land-cover and seasonal masks | **Partial** | Current procedural masks exist; production sensor-specific seasonal tuning remains future work. |
| Weather affecting sensing | **Implemented** | `src/argusnet/world/visibility.py`, `src/argusnet/world/weather.py` |
| Weather affecting vehicle dynamics | **Planned** | Wind/weather are not fully coupled to drone dynamics. |
| CRS/geodetic transforms | **Partial** | `src/argusnet/core/frames.py` and `EnvironmentCRS`; most sim scenarios remain local-frame. |
| Rust world/terrain runtime integration | **Partial** | `terrain-engine` exists, but Python sim terrain remains authoritative for current runtime. |

## Sensing And Fusion

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Bearing observation synthesis | **Implemented** | `src/argusnet/simulation/sim.py` |
| FOV/range/elevation generation gating | **Implemented** | Python generation path and Rust runtime validation. |
| LOS terrain/obstacle occlusion | **Implemented** | `src/argusnet/world/visibility.py` |
| Sensor noise model | **Implemented** | `src/argusnet/sensing/models/noise.py` |
| Rust IMM fusion and association | **Implemented** | `rust/argusnet-core/src/lib.rs`, `association.rs` |
| Labeled/GNN/JPDA association | **Implemented** | `rust/argusnet-core` |
| Python Kalman/triangulation utilities | **Partial** | `src/argusnet/mapping/fusion.py` and localization helpers are utility/reference paths, not runtime fusion authority. |
| Clutter / false-alarm generation | **Partial** | Parameters exist; scenario-wide clutter generation remains limited. |
| Versioned observation-source interface | **Implemented** | `argusnet.sensing.observation_source` (`ObservationSource` protocol, `ObservationRequest`, `AnalyticObservationSource`); `run_simulation()` synthesizes every step through it and accepts an injectable `observation_source`. `build_observations()` remains the default analytic backend behind the seam. |

## Mapping

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Coverage map | **Implemented** | `src/argusnet/mapping/coverage.py` |
| World map from scan observations | **Implemented** | `src/argusnet/mapping/world_map.py` |
| Occlusion-aware scan coverage | **Implemented (opt-in)** | `--occlusion-aware-mapping` gates coverage/reconstruction through `EnvironmentQuery.los()` |
| Replay `MappingState` | **Implemented** | `argusnet.core.types.MappingState`, populated in `src/argusnet/simulation/sim.py` |
| Elevation/occupancy/semantic/uncertainty modules | **Partial** | Modules exist under `src/argusnet/mapping/*`; closed-loop runtime integration is incomplete. |
| Runtime belief-query interface | **Implemented** | `argusnet.mapping.belief` (`BeliefQuery` protocol, `WorldBeliefQuery`, `BeliefSummary`, `BELIEF_QUERY_CONTRACT_VERSION`). `run_simulation()` builds it over the live world map; the viewer terrain reconstruction reads believed heights through it and `MappingState` belief fields are populated from `belief_summary()`. Sensor ingest stamps a dense observed-height layer so belief consumers never read terrain truth. |
| Belief-world planning authority | **Partial** | The belief query is the runtime authority for the terrain reconstruction and mapping summary; planners still read the coverage map directly (next belief consumer). |
| Truth-isolation tests for physical-mode mapping | **Implemented (opt-in)** | `tests/test_occlusion_aware_mapping.py` covers LOS-gated reconstruction and obstacle-routed redirects. |

## Localization

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Grid localization for scan-map-inspect | **Implemented** | `src/argusnet/localization/engine.py` `GridLocalizer` |
| Localization timeout gate | **Implemented** | `LocalizationConfig.localization_timeout_steps`; mission state records team-level timeout. |
| Replay `LocalizationState` and estimates | **Implemented** | `argusnet.core.types.LocalizationState`, `LocalizationEstimate`, populated in `sim.py` |
| VIO interfaces and simple backends | **Partial** | `src/argusnet/localization/vio.py` |
| Relocalization helpers | **Partial** | `src/argusnet/localization/relocalization.py`; not fully wired into mission execution. |
| Pose graph / loop closure modules | **Partial** | Foundations exist; full runtime pose graph is not the current localization authority. |
| Rich pose/covariance/status localization contract | **Planned** | `LOCALIZATION.md` documents the roadmap model. |

## Inspection And Mission Execution

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Scan-map-inspect mission phases | **Implemented** | `src/argusnet/simulation/sim.py`: `scanning`, `localizing`, `inspecting`, `egress`, `complete` |
| Curated in-range target-tracking demo | **Implemented (opt-in)** | `--demo tracking` / `tracking_demo_options()` in `sim.py` place targets inside sensor range so fused tracks confirm out of the box; the default large-map behavior (no tracks) is unchanged. |
| POI model and lifecycle | **Implemented** | `InspectionPOI`, `POIStatus`, `POIManager` |
| Energy-aware POI assignment | **Implemented** | `src/argusnet/planning/poi.py` |
| POI handoff, team assignment, rescoring | **Implemented** | `POIManager` and current sim wiring |
| Inspection events | **Implemented** | `InspectionEvent` populated by mission-zone coverage/violation loop |
| Exclusion-zone route/deconfliction handling | **Partial** | Current sim and deconfliction paths handle supported cases; full mission-executive hard gating remains future work. |
| Evidence sets, reconstructions, change records | **Planned** | Contract exists in `INSPECTION.md`; no full runtime/index-backed evidence pipeline yet. |
| Full closed-loop mission executive | **Planned** | `src/argusnet/mission/execution.py` is a foundation; `scan_map_inspect` is the current partial runtime bridge. |

## Planning, Coordination, And Safety

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| 2D visibility-graph planner | **Implemented** | `src/argusnet/planning/planner_base.py` |
| Route cache and obstacle expansion | **Implemented** | `PathPlanner2D`, `PlannerConfig` |
| Frontier enclosed-gap gate | **Implemented** | `FrontierPlanner.find_gap_cells()` is wired into `scan_map_inspect`. |
| Frontier cell selection | **Implemented (opt-in)** | `select_frontier_cell()` redirects scanning drones when `--frontier-exploration` is set; the default scan remains sector lawnmower. |
| Obstacle-routed mission redirects | **Implemented (opt-in)** | `--occlusion-aware-mapping` routes frontier/adaptive/POI/egress redirects through `PathPlanner2D` with straight-line fallback. |
| Claimed-cell RF latency helpers | **Partial** | `CoordinationManager.update_claimed()` / `flush_messages()` exist but are not called by `sim.py`. |
| Formation offsets | **Partial** | `CoordinationManager.formation_offsets()` exists but is not called by `sim.py`. |
| Coordinator election | **Implemented** | `CoordinationManager.elect_coordinator()` is wired into `scan_map_inspect`. |
| Drone deconfliction events | **Implemented** | `src/argusnet/planning/deconfliction.py` and sim replay events |
| Blocking safety gate | **Implemented (opt-in)** | `--safety-blocking` runs after deconfliction, clamps commands, emits events, and holds Abort-state drones. |
| 3D path planning | **Planned** | Current route planner is 2D; altitude is handled separately. |

## Evaluation, Benchmarking, Replay, And UI

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Replay schema | **Implemented** | `docs/replay-schema.json`, `src/argusnet/evaluation/replay.py` |
| Evaluation report metrics | **Implemented** | `src/argusnet/evaluation/metrics.py` |
| Benchmark aggregation | **Implemented** | `src/argusnet/evaluation/benchmarks.py` |
| Reports/export helpers | **Implemented** | `src/argusnet/evaluation/reports.py`, `export.py` |
| Benchmark standards | **Implemented as documentation** | `docs/PERFORMANCE_AND_BENCHMARKING.md` |
| CI benchmark markers and golden performance files | **Planned** | Standard is documented; not all commands/files are wired into CI. |
| Viewer replay playback | **Implemented** | `rust/argusnet-viewer` |
| Viewer live streaming | **Implemented** | Legacy `WatchFrames` plus operator-focused `WatchFramesV2`, reconnect, bounded queues/history, truth filtering, sequence/drop telemetry |
| Viewer headless CI/render path | **Partial** | Headless module exists; CI/render workflow is not fully documented or gated. |

## Interface Boundaries That Need Formalization

1. ~~Observation generation should move behind a versioned observation-source contract.~~
   **Done** — `argusnet.sensing.observation_source`, wired as the default synthesis
   path in `run_simulation()` (see the Sensing And Fusion table).
2. ~~Mapping should expose a runtime belief query interface before planners depend on
   belief-world semantics in physical-mode tests.~~
   **Done** — `argusnet.mapping.belief` (`BeliefQuery` protocol + `WorldBeliefQuery`
   backend), constructed in `run_simulation()` and consumed by the viewer terrain
   reconstruction and the `MappingState` belief summary (see the Mapping table).
   Planner consumption of belief-world semantics is the next step behind the same seam.
3. Localization should expose a richer pose/covariance/status interface before precision
   inspection routing depends on it.
4. Inspection evidence and reconstruction artifacts should be persisted through indexing before
   repeat-inspection/change-detection claims are made.
5. The mission executive should own task state and route all motion through planning, trajectory,
   and safety gates.
6. Rust-side world/terrain interfaces should be explicit before Rust prediction or safety consumers
   depend on terrain/obstacle data.

## Proto And Schema State

| Concern | Status |
|---------|--------|
| Proto file | `proto/argusnet/v1/world_model.proto` |
| Python generated bindings | `src/argusnet/v1/world_model_pb2.py`, `world_model_pb2_grpc.py` |
| Rust generated/conversion boundary | `rust/argusnet-proto`, `rust/argusnet-server`, `rust/argusnet-core` |
| Replay schema | `docs/replay-schema.json`; additive changes preferred |
| Mapping/localization/inspection over gRPC | Not currently part of the proto runtime contract |
| Operational target metadata and safety events | Additive fields on live/replay frames; supported by V2 live clients |
