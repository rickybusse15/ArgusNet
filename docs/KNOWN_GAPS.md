# Known Gaps — ArgusNet

## Scheduled-runtime status (reviewed 2026-10-12)

| Capability | Status | Remaining evidence |
| --- | --- | --- |
| Rust stage composition, direct fusion and replay writer | Partial | 2026-10-03: `argusnet-rs` builds and runs all seven canonical scenarios with no Python, six within every envelope (#122). Owned subsystem state (A1.2) and a production constructor that rejects missing stages (A1.3, `tests/production.rs`) are done. Acceptance scores against a stored reference (D-10). Multi-seed and held-out qualification done (A3.2, `docs/performance/a32-qualification-2026-10-03.md`): everything passes except `covariance_reduction_mean`, whose sampling noise is of the order of its envelope. D-12 judges it by its median (envelope unchanged); the confirmation run on 30 fresh seeds passes 7/7 (`docs/performance/d12-confirmation-2026-10-04.md`) |
| Command recovery and format-2 control checkpoints | Implemented in reviewed `ec9c2f7` | Not full causal checkpointing or proof of remote release |
| Subsystem-wide reset/restore | **Implemented** (A2, 2026-10-03) | `AssembledRun::reset` reloads the run as assembled; checkpoint format 4 binds to the scenario's content hash and refuses another scenario before mutating. Evidence: `scripts/replay_hashes.py --lifecycle` (all 7 canonical scenarios × seeds 7, 42: reset at a seeded tick replays byte-identical to fresh; resume from an on-disk checkpoint equals the uninterrupted frames), `tests/assembled_resume.rs` (`a_reset_assembled_run_is_a_fresh_run`, `a_checkpoint_from_a_different_scenario_is_refused_before_anything_changes`, `truncated_or_corrupt_checkpoint_text_fails_cleanly`, per-phase resume). Not covered: pending scheduler jobs (P4.10) |
| Independent/rational task and sensor rates | **Partial** (P4.10a) | Scheduler types (integer `SimTime`, rational schedules, phase order, `EventQueue`) and the legacy-order compound-lane and phased-lane executors are merged and replay byte-identical (B2.1–B2.3). The mission lane can run at its own rate under `scheduled_v1` (B2.4, `--executor lanes --mission-rate`), recorded in the replay; a stage not due is traced as skipped and its next run is given the elapsed time. The viewer's in-process run takes the same rate (`--run ... --mission-rate`). The mapping lane can run at its own rate too (B2.5, `--mapping-rate`), its skipped ticks buffered and integrated when it runs. Not yet: fusion off the legacy rate, stages 1–6 off the compound lane, real per-sensor rates, work pending across a boundary |
| Timestamped measurement delivery | **Partial** (P4.10b, #196, #199) | Delivery envelopes with acquisition/availability/receipt ticks, seeded per-source delay and jitter, fusion at a fixed lag, explicit drops; track error scored at evidence time with present error and evidence age reported (`docs/migration/DELIVERY_TIMING.md`). Not yet: buffered mapping inputs, per-source capture schedules, clock quality, evidence age on the track itself, delayed-measurement smoothing |
| Bounded causal jobs and complete scheduler history | Planned P4.10d–f | Deterministic application, stale result checks and restore during pending work |
| Starting with no world knowledge | **First slice** (#201–#208) | `--world-knowledge none`: range scanners, a checkpointed belief grid, stages 1, 3 and 4 planning on it, a physics guard; the map exports as a scene package. Pursuit now uses fused tracks (a tasking cue before the first). Search drones explore the frontier and scan progress reads the map. Still reading truth: the bundle builder's lane and POI planning (`docs/migration/UNKNOWN_WORLD.md`) |
| Operate/Plan/Review visual redesign and freshness | Planned/partial foundations | P2/P5 acceptance; current replay controls remain distinct |
| Qualified hardware adapters and deadlines | Planned S3–S7 | Actual SIL/HIL and device/clock/fault evidence; telemetry viewing grants no control |

See [target architecture](RUST_RUNTIME_ARCHITECTURE.md),
[viewer specification](VIEWER_UI.md) and [hardware integration](HARDWARE_INTEGRATION.md).
The older domain inventory below must not be read as completed Rust production
integration merely because a Python path or ported library exists.

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
| Rust world/terrain runtime integration | **Implemented (Phase 3)** | `argusnet-world` owns analytic/tiled sampling, obstacle geometry/indexing, land-cover queries and LOS. Python calls the native library; the viewer uses its replay-raster adapter. Python scenario authoring/export and the tick loop remain until later phases. |

## Sensing And Fusion

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Bearing observation synthesis | **Implemented** | `src/argusnet/simulation/sim.py` |
| FOV/range/elevation generation gating | **Implemented** | Python generation path and Rust runtime validation. |
| LOS terrain/obstacle occlusion | **Implemented** | `src/argusnet/world/visibility.py` |
| Sensor noise model | **Implemented** | `src/argusnet/sensing/models/noise.py` |
| Rust IMM fusion and association | **Implemented** | `rust/argusnet-core/src/lib.rs`, `association.rs` |
| Labeled/GNN/JPDA association | **Implemented** | `rust/argusnet-core` |
| Poorly conditioned two-ray triangulation | **Gap** | A cluster of two nearly parallel rays that still clears `min_intersection_angle_deg` (2.5°) can triangulate hundreds of metres off, and the fix is admitted with a small measurement std. P4.9c saw one such run-off at 3.6° (intercept_stress, seed 31415). The fix is tracker tuning: a larger minimum angle, or a range-aware measurement std. See `docs/migration/PHASE_ACCEPTANCE.md`. |
| Python Kalman/triangulation utilities | **Partial** | `src/argusnet/mapping/fusion.py` and localization helpers are utility/reference paths, not runtime fusion authority. |
| Clutter / false-alarm generation | **Partial** | Parameters exist; scenario-wide clutter generation remains limited. |
| Formal observation-source interface | **Implemented** | `argusnet.sensing.observation_source.ObservationSource` is the versioned, injectable runtime boundary; the deterministic simulator generator remains available as `build_observations()`. |

## Mapping

| Capability | Status | Current Location / Gap |
|------------|--------|------------------------|
| Coverage map | **Implemented** | `src/argusnet/mapping/coverage.py` |
| World map from scan observations | **Implemented** | `src/argusnet/mapping/world_map.py` |
| Occlusion-aware scan coverage | **Implemented (opt-in)** | `--occlusion-aware-mapping` gates coverage/reconstruction through `EnvironmentQuery.los()` |
| Replay `MappingState` | **Implemented** | `argusnet.core.types.MappingState`, populated in `src/argusnet/simulation/sim.py` |
| Elevation/occupancy/semantic/uncertainty modules | **Partial** | Modules exist under `src/argusnet/mapping/*`; closed-loop runtime integration is incomplete. |
| Belief-world planning authority | **Partial** | Rust queries and P4.6d revisioned belief publication exist; mandatory planning use and complete production ownership remain unqualified. |
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
| Blocking safety monitor after an abort | **Reference behaviour, ported** | An abort zeroes the allowed velocity, which is stored as `previous_velocity`. The controller's next normal command then reads as an acceleration violation and aborts again: in the scan fixture every tick of both drones aborts. The clamp does not accumulate, so missions still complete. Whether the policy should difference against the commanded velocity after an abort is a policy decision (`docs/migration/SAFETY_AND_PROFILES.md`). |
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
| Replay schema | **Implemented** | `docs/replay-schema.json`, `src/argusnet/evaluation/replay.py`. Enforced with `jsonschema` and guarded end-to-end by `tests/test_replay_contract.py` plus `replay.rs::contract_fixture_tests`. |
| Evaluation report metrics | **Implemented** | `src/argusnet/evaluation/metrics.py`. `check_pass_fail`'s `track_continuity_mean` threshold fails every canonical scenario, including the target-free scan scenarios where continuity is 0 by construction (P4.9c). `covariance_reduction_mean` now uses each track's covariance at its last update and is scored in acceptance only with at least five qualifying tracks a side (D-1, 2026-10-03); `sar_person_search` passes it. The reference's draws are platform-dependent, so acceptance now scores against a stored reference (D-10). A3.2 (30 + 10 held-out seeds): `localisation_rmse_m` passes everywhere; `covariance_reduction_mean` is underpowered at these seed counts as a mean; judged by its median (D-12) it passes the 30-seed confirmation set. |
| Benchmark aggregation | **Implemented** | `src/argusnet/evaluation/benchmarks.py` |
| Reports/export helpers | **Implemented** | `src/argusnet/evaluation/reports.py`, `export.py` |
| Benchmark standards | **Implemented as documentation** | `docs/PERFORMANCE_AND_BENCHMARKING.md` |
| CI benchmark markers and golden performance files | **Implemented** | Deterministic work-counter gate runs on every PR; timing comparison is opt-in (`ARGUSNET_RUN_PERF_TIMING`). Baseline profile in `docs/performance/baseline-profile-2026-09.md`. |
| Viewer replay playback | **Implemented** | `rust/argusnet-viewer` |
| Viewer live streaming | **Implemented** | Legacy `WatchFrames` plus operator-focused `WatchFramesV2`, reconnect, bounded queues/history, truth filtering, sequence/drop telemetry |
| Scene GLB triangle winding | **Fixed (2026-10-12)** | Every terrain triangle used to be wound facing down while its normals faced up, so the double-sided material lit it as a back face: the sun contributed nothing, and the camera's exposure had been opened three stops to make up for it, which washed sunlit surfaces (buildings, the drones' self-built map) out to white. Both scene writers now wind the terrain counter-clockwise from above, the committed `scene.smartscene` terrain was rewritten the same way, and both cameras use EV100 12 (`Exposure::OVERCAST`, matching the 12 000 lux sun). `tests/test_rust_scene_build.py` fails if any layer of a built package has a triangle facing away from its normals. Before/after renders: [`performance/terrain-winding-2026-10-12/`](performance/terrain-winding-2026-10-12/). The lit ground exposed two compositing faults, both fixed in the same change: sensor volumes and zone walls were not additive (they covered 85% of what was behind them), and the belief overlay took the ground's hue. The picker still casts occluders from both faces, for packages built before the fix. |
| Viewer headless CI/render path | **Implemented smoke coverage** | CI renders through Bevy and checks a non-empty image. Fixed-image regression and steady-state frame-budget gates remain open. |

## Interface Boundaries That Need Formalization

1. Observation generation is behind the versioned `ObservationSource` contract; future dataset and physical adapters should implement that interface.
2. Mapping should expose a runtime belief query interface before planners depend on
   belief-world semantics in physical-mode tests.
3. Localization should expose a richer pose/covariance/status interface before precision
   inspection routing depends on it.
4. Inspection evidence and reconstruction artifacts should be persisted through indexing before
   repeat-inspection/change-detection claims are made.
5. The mission executive should own task state and route all motion through planning, trajectory,
   and safety gates.
6. Rust world queries and native safety are now load-bearing (migration Phase 3).
   Remaining runtime consumers should use `argusnet-world` directly as they migrate;
   Python still owns the simulation loop. See [migration plan](RUST_RUNTIME_MIGRATION.md).

## Proto And Schema State

| Concern | Status |
|---------|--------|
| Proto file | `proto/argusnet/v1/world_model.proto` |
| Python generated bindings | `src/argusnet/v1/world_model_pb2.py`, `world_model_pb2_grpc.py` |
| Rust generated/conversion boundary | `rust/argusnet-proto`, `rust/argusnet-server`, `rust/argusnet-core` |
| Replay schema | `docs/replay-schema.json`; additive changes preferred |
| Mapping/localization/inspection over gRPC | Not currently part of the proto runtime contract |
| Operational target metadata and safety events | Additive fields on live/replay frames; supported by V2 live clients |

## Expanded viewer requirements (planned)

The [latest feature UI plan](VIEWER_FEATURE_INTEGRATION.md) adds Perspective,
Top-down and Section over coherent 3D spatial revisions, localization integrity,
onboard contingencies, SIL/HIL qualification and client-product review/export.
The [autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md) defines their S0–S8 gates.
These are requirements, not claims that volumetric mapping, aircraft autonomy,
section rendering or all export formats are implemented.

## Latest merged phase status

At `ba163ac`, P4.3g landscapes, P4.6d belief publication and P4.8a–c mission/world/
stage integration are present. Rust snapshots can contain platforms and tracks.
References below to missing builders/publication or an empty shell describe the
older review baseline. Complete scenario/controller, checkpoint/replay/CLI and
viewer integration remain open; see [the phase plan](migration/DELIVERY_PLAN.md).

## Next delivery boundary

The [ownership inventory](STATE_OWNERSHIP.md) distinguishes Rust libraries from
active runtime consumers. Localization/VIO, sensing, coverage/belief queries and
planning ports exist, but the Rust simulation still needs complete domain-stage
composition, runtime world assembly and revisioned belief ownership. Its command
and checkpoint foundations require the corrections in the
[source review](NEXT_REFACTOR_REVIEW.md) before viewer control integration.

[VIEWER_UI.md](VIEWER_UI.md) is the new target specification, not the current UI.
Operate/Plan/Review, acknowledged runtime controls and complete world-aware live
state remain staged work. Existing V2 live rendering is not evidence of P5.3.

F1 scorecards, F3 explanations and F4 comparisons have shipped as replay and
mission slices, and F2 failure schedules has its first slice (sensor outages, link
delays, source restarts). F5 continuous reserve replanning, F6 persistent memory
and F7 evidence-based adaptive inspection are planned extensions of existing modules. See [their acceptance matrix](CAPABILITY_DELIVERY.md).
Energy-aware assignment, benchmark aggregation and POI dwell already exist; they
must not be counted as completion of these richer features.
