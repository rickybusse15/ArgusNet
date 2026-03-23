# State Ownership Map — Smart Trajectory Tracker

> Stage 0 Architecture Audit — 2026-03-15
>
> This document maps every major runtime concept to its owner, readers, writers, and
> timing assumptions.  Line numbers reference the codebase at the time of writing.

---

## 1. Coordinate System Convention

All coordinates are **meter-based** in a local projected (XY) frame with Z above
the terrain datum.  Angles are radians internally; CLI flags are labeled when degrees
are used.  There is a stub `EnvironmentCRS` struct in
`src/smart_tracker/environment.py` (lines 66-86) that holds a `local-enu` CRS
identifier but no active geodetic projection is wired up for simulation.

---

## 2. Runtime State Inventory

### 2.1 Terrain Model

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/terrain.py` — `TerrainModel` (line 334), tiled representation in `src/smart_tracker/environment.py` — `TerrainLayer` (line 205) |
| **Owner** | Python (created once at scenario build time) |
| **Writers** | `sim.py` via `terrain_model_from_preset()` or `EnvironmentModel.from_legacy()` at scenario construction |
| **Readers** | `sim.py` (AGL clamping, collision checks), `visibility.py` (LOS terrain sampling), `planning.py` (path obstacles), `replay.py` / `scene.py` (exported to replay JSON as `viewer_mesh`), `rust/tracker-viewer` (reads from replay JSON, not live) |
| **Cross-boundary** | Exported as a 2-D height grid in replay JSON (`meta.terrain.viewer_mesh`).  The Rust viewer (`rust/tracker-viewer/src/replay.rs` `TerrainViewerMesh`) reads it from disk.  Rust tracking engine never sees terrain. |
| **Timing** | Static for the duration of a simulation run.  Created once in `build_default_scenario()` / `ScenarioDefinition.__post_init__`. |
| **Gap** | Terrain is purely analytic in Python (on-the-fly `height_at()`).  Tiled `TerrainLayer` exists but is pre-sampled from the analytic model at 5 m resolution (`environment.py` line 294).  The Rust runtime has no access to terrain; collision avoidance and AGL enforcement are Python-only. |

---

### 2.2 Obstacles / Environment Model

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/obstacles.py` (primitives: `BuildingPrism`, `CylinderObstacle`, `OrientedBox`, `PolygonPrism`, `WallSegment`, `ForestStand`), `src/smart_tracker/environment.py` — `ObstacleLayer`, `EnvironmentModel` (line 664) |
| **Owner** | Python |
| **Writers** | `sim.py` scenario builders, `_scene_*.py` scene compiler |
| **Readers** | `visibility.py` (`EnvironmentQuery.los()`), `planning.py` (`ObstacleLayer.query_obstacles()`), `sim.py` `FollowPathController` (collision check, line 478), `scene.py` (serialised to `.smartscene`) |
| **Cross-boundary** | Serialised to replay JSON `meta.occluding_objects` as a flat list of geometry metadata.  Rust viewer renders them; Rust tracking engine does not use obstacle data. |
| **Timing** | Static for a simulation run. |
| **Gap** | The new `EnvironmentModel` (tiled layers) and legacy `OccludingObject` (cylinder stub) coexist via `EnvironmentModel.from_legacy()`.  `ForestStand` land-cover objects are present but not yet fully wired into `ObstacleLayer` spatial queries (only hard-blocker types `building` / `wall` are checked in `ObstacleLayer.point_collides()`). |

---

### 2.3 Land Cover

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/environment.py` — `LandCoverLayer`, `LandCoverClass` (lines 89, 471) |
| **Owner** | Python |
| **Writers** | Scenario builders (defaults to `LandCoverLayer.open_terrain()` unless a GIS bundle is loaded) |
| **Readers** | `visibility.py` — `EnvironmentQuery.los()` uses `LandCoverClass` for attenuation and noise multiplier lookup |
| **Cross-boundary** | Legend exported to replay JSON `meta.land_cover_legend`.  Rust viewer does not yet render land cover. |
| **Timing** | Static. |
| **Gap** | All simulations default to all-open land cover unless an external GIS bundle is provided.  No terrain preset automatically generates a matching land-cover map. |

---

### 2.4 Weather Model

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/weather.py` — `WeatherModel`, `WindModel`, `PrecipitationModel`, `CloudModel`, `AtmosphericProfile` |
| **Owner** | Python |
| **Writers** | `sim.py` via `weather_from_preset()` at scenario construction |
| **Readers** | `sim.py` `build_observations()` (sensor noise and attenuation scaling), `visibility.py` `compute_weather_factor()` |
| **Cross-boundary** | Weather parameters are **not** exported to replay JSON or transmitted to Rust.  They influence only observation generation on the Python side. |
| **Timing** | Static preset for the simulation.  `WindModel.wind_at(altitude_m, time_s)` returns a time-varying vector used for target/drone position adjustments, but this is evaluated at each step in Python. |
| **Gap** | Weather is Python-only and invisible to the Rust tracking engine.  There is no weather-state field in `IngestFrameRequest` or the proto.  Wind affects target positions but is not represented in `TruthState`. |

---

### 2.5 Target Truth State (Ground Truth Trajectories)

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `TruthState` (line 55); trajectory functions in `src/smart_tracker/behaviors.py` |
| **Owner** | Python |
| **Writers** | `SimTarget.truth(timestamp_s)` evaluated per-step in `sim.py` |
| **Readers** | Python `build_observations()` (to generate bearing rays), Rust `TrackingEngine.ingest_frame()` (receives `truths` list via proto for accuracy metrics only) |
| **Cross-boundary** | Sent to Rust via `IngestFrameRequest.truths` (proto field 4, `tracker.proto` line 117).  Rust uses them only to compute `track_errors_m` in `PlatformMetrics`.  Returned in `PlatformFrame.truths`. |
| **Timing** | Evaluated at each simulation step (default 0.25 s, `DynamicsConfig.default_dt_s`).  Trajectory functions are deterministic closures parameterised by `timestamp_s`. |
| **Gap** | The complete trajectory function is only available in Python.  Rust only sees the current truth snapshot at the time of the frame.  Trajectory type (sinusoid, racetrack, etc.) is not exported to replay or transmitted. |

---

### 2.6 Drone / Node State

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `NodeState` (line 17); simulation controllers in `sim.py` — `SimNode`, `FollowPathController` |
| **Owner** | Python (position, velocity computed per-step) |
| **Writers** | `SimNode.state(timestamp_s)` evaluates trajectory; `FollowPathController.__call__()` computes follow-mode position; `LaunchController` manages climb phase |
| **Readers** | Python `build_observations()` (sensor origin for bearing generation), Rust `TrackingEngine.ingest_frame()` (stores in `nodes: HashMap<String, NodeState>`) |
| **Cross-boundary** | Sent per-frame via `IngestFrameRequest.node_states` (proto line 115).  Rust stores latest node state for observation validation (range checks, duplicate detection).  Returned in `PlatformFrame.nodes`. |
| **Timing** | Evaluated at each simulation step.  Drone state is stateful via `FollowPathController._state` dict (mutable, contains `last_xy`, `last_velocity_xy`, `smoothed_terrain_z`, `route_points`). |
| **Gap** | Drone role (`"interceptor"` vs `"tracker"`), slot index, and slot count are Python-only fields on `FollowPathController`.  These are not transmitted to Rust or stored in `NodeState`.  Sensor type and FOV are in `NodeState` in Python but stripped before proto conversion (not in `NodeState` proto message). |

---

### 2.7 Bearing Observations

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `BearingObservation` (line 30) |
| **Owner** | Python (generated); Rust (filtered and consumed) |
| **Writers** | `sim.py` `build_observations()` synthesises observations from truth positions and sensor models |
| **Readers** | Python serialises to proto; Rust `TrackingEngine.prefilter_observations()` validates and rejects; Rust fusion pipeline triangulates to 3D position |
| **Cross-boundary** | Sent via `IngestFrameRequest.observations` (proto lines 46-54).  Python also stores simulation-side rejections (`generation_rejections` — obs rejected before reaching Rust) separately from Rust-side rejections (`rejected_observations` — obs rejected inside Rust). |
| **Timing** | Generated at each simulation step for every (node, target) pair that passes visibility and dropout tests. |
| **Gap** | Python-side generation rejections (terrain occlusion, FOV, range, dropout) are collected in `ObservationBatch.generation_rejections` and sent as a separate field in the proto (`generation_rejections` in `PlatformFrame`).  The split between Python rejection and Rust rejection is an implicit protocol that is not formally documented or versioned. |

---

### 2.8 Tracks (Estimated State)

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `TrackState` (line 63); Rust `tracker-core/src/lib.rs` — `ManagedTrack`, `TrackState` (lines 372-384, 956-967) |
| **Owner** | **Rust** (authoritative).  Python holds a read-only cache in `TrackingService.tracks: Dict[str, TrackState]` (service.py line 383) |
| **Writers** | Rust `TrackingEngine.ingest_frame()` only |
| **Readers** | Python `TrackingService.ingest_frame()` deserialises from proto response; `replay.py` serialises to JSON; `scene.py` packages for viewer |
| **Cross-boundary** | Returned in `IngestFrameResponse.frame.tracks` (proto line 105). Python cache is updated after each `ingest_frame()` call. |
| **Timing** | Updated once per `ingest_frame()` call (once per simulation step).  Rust internally predicts on every frame for all alive tracks, regardless of update. |
| **Track lifecycle states** | `Tentative` → `Confirmed` → `Coasting` → `Deleted` (managed in `rust/tracker-core/src/lib.rs` lines 927-943, `ManagedTrack.update_lifecycle()`) |
| **Filter model** | IMM (Interacting Multiple Model): CV (Constant Velocity, 6-state) + CT (Coordinated Turn, 7-state), blended by mode probabilities.  Defined in `tracker-core/src/lib.rs` lines 680-904. |
| **Gap** | The full covariance matrix (6×6) is transmitted but Python-side `TrackState` stores it as a `np.ndarray` with no structured access.  There is no Python code that reads covariance for any downstream purpose (planning, fusion, etc.). |

---

### 2.9 Observation Rejections

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `ObservationRejection` (line 41) |
| **Owner** | Split: Python owns `generation_rejections`; Rust owns `rejected_observations` |
| **Writers** | Python `build_observations()` for generation rejections; Rust `prefilter_observations()` and association logic for runtime rejections |
| **Readers** | Both sides serialise to replay JSON.  Viewer renders rejection markers. |
| **Timing** | Per simulation step. |

---

### 2.10 Platform Metrics

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `PlatformMetrics` (line 77); proto `PlatformMetrics` (line 89) |
| **Owner** | **Rust** (computed); Python receives read-only |
| **Writers** | Rust `TrackingEngine.ingest_frame()` computes accuracy vs truth, acceptance counts, rejection breakdown |
| **Readers** | Python `sim.py` writes to `metrics_rows` for CSV output; `replay.py` serialises per-frame metrics to JSON |
| **Timing** | Computed once per frame by Rust. |

---

### 2.11 Mission Zones

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `MissionZone` (line 143); Rust viewer `rust/tracker-viewer/src/replay.rs` — `MissionZone` (line 89) |
| **Owner** | Python (definitions); Rust viewer (rendering) |
| **Writers** | `sim.py` `build_default_scenario()` creates zones; stored in `ScenarioDefinition.mission_zones` |
| **Readers** | `replay.py` serialises to `meta.zones`; Rust viewer reads from replay JSON |
| **Cross-boundary** | Serialised as JSON in replay metadata.  Not part of the gRPC proto — zones are **not** transmitted to the tracking engine. |
| **Timing** | Static — defined at scenario construction. |
| **Gap** | Mission zones are purely cosmetic from the tracking engine's perspective.  There is no enforcement logic: drones do not respect exclusion zones, and the tracker does not prioritise surveillance zones.  Zone data exists only in replay metadata. |

---

### 2.12 Mapping State

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/argusnet/core/types.py` — `MappingState` |
| **Owner** | Python |
| **Writers** | `sim.py` `run_simulation` — increments `CoverageMap` per mobile node footprint each step |
| **Readers** | Serialised to replay JSON `frames[*].mapping_state`; Rust viewer shows coverage % in Mapping panel |
| **Cross-boundary** | Not in gRPC proto. |
| **Timing** | One `MappingState` per frame (cumulative, not per-step delta). |

---

### 2.12b Localization State

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/argusnet/core/types.py` — `LocalizationState` |
| **Owner** | Python |
| **Writers** | `sim.py` `run_simulation` — derived from active track covariances and observation confidences |
| **Readers** | Serialised to replay JSON `frames[*].localization_state`; Rust viewer shows in Localization panel |
| **Cross-boundary** | Not in gRPC proto. |
| **Timing** | One `LocalizationState` per frame when active tracks exist; `null` otherwise. |

---

### 2.12c Inspection Events

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/argusnet/core/types.py` — `InspectionEvent` |
| **Owner** | Python |
| **Writers** | `sim.py` `run_simulation` — generated when mobile nodes enter/exit mission zone radii |
| **Readers** | Serialised to replay JSON `frames[*].inspection_events`; Rust viewer shows in Inspection Events panel |
| **Cross-boundary** | Not in gRPC proto. |
| **Timing** | Zero or more events per frame, one per (node, zone) boundary crossing or coverage update. |

---

### 2.13 Tracker Configuration

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/service.py` — `TrackerConfig` (line 52); Rust `tracker-core/src/lib.rs` — `TrackerConfig` (line 105); proto `TrackerConfig` (line 11) |
| **Owner** | Both — Python constructs and transmits; Rust is authoritative at runtime |
| **Writers** | Python constructs `TrackerConfig`, serialises to YAML for daemon startup, or sends via `GetConfig` RPC |
| **Readers** | Rust `TrackingEngine.new()` initialises from config; Python reads back via `GetConfig` RPC |
| **Cross-boundary** | Sent at startup via YAML file (`service.py` line 432, `_render_tracker_config_yaml()`).  Python reads back the remote config via `GetConfigResponse`. |
| **Timing** | Set at daemon startup.  No live reconfiguration is supported. |

---

### 2.14 Node Health

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/models.py` — `NodeHealthMetrics`, `HealthReport` (lines 105, 116); Rust `tracker-core/src/lib.rs` — `NodeHealthTracker` (line 11) |
| **Owner** | **Rust** (authoritative); Python receives via `Health` RPC |
| **Writers** | Rust `TrackingEngine.ingest_frame()` updates `NodeHealthTracker` per observation |
| **Readers** | Python `TrackingService.health()` fetches on demand (not per-frame) |
| **Timing** | Accumulated across all frames.  Polled on demand, not streamed. |

---

### 2.15 Replay Document

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/replay.py` — `ReplayDocument` (type alias, line 15); JSON schema at `docs/replay-schema.json` |
| **Owner** | Python (writes); Rust viewer (reads) |
| **Writers** | `sim.py` → `replay.py` `build_replay_document()` then `write_replay_document()` |
| **Readers** | `rust/tracker-viewer/src/replay.rs` `ReplayDocument` deserialiser |
| **Cross-boundary** | Replay JSON is the **sole interface** between simulation and viewer.  It carries: `meta` (terrain, zones, occluding objects, scenario config, CRS), `frames` (per-step snapshots of tracks/truths/nodes/observations/rejections/metrics/launch events), `summary`. |
| **Timing** | Written once after simulation completes.  Viewer reads entire file at load time. |
| **Gap** | The Rust viewer's `ReplayFrame` uses `f32` for positions and timestamps, while the Python writer uses `f64`.  This is a precision mismatch with no documented rationale. |

---

### 2.16 Simulation Configuration (SimulationConfig, ScenarioOptions, SimulationConstants)

| Attribute | Value |
|-----------|-------|
| **Defined in** | `src/smart_tracker/sim.py` — `SimulationConfig` (line 302), `ScenarioOptions` (line 172); `src/smart_tracker/config.py` — `SimulationConstants` |
| **Owner** | Python (read-only once constructed) |
| **Writers** | CLI / caller at startup |
| **Readers** | `sim.py` `run_simulation()` |
| **Cross-boundary** | Not transmitted to Rust.  Rust only receives the subset relevant to tracking via `TrackerConfig`. |
| **Timing** | Static for a simulation run. |

---

## 3. Cross-Boundary State Flows

### 3.1 Python → Rust (per-frame, via gRPC `IngestFrame`)

```
Python step                Rust step
─────────────────────────────────────────────────────────────────
NodeState list             → nodes: HashMap<String, NodeState>    (replaces/updates)
BearingObservation list    → prefilter → association → Kalman update
TruthState list            → accuracy metrics only
float timestamp_s          → frame clock
```

**Proto message:** `IngestFrameRequest` (`tracker.proto` line 114)
**Serialisation:** `service.py` lines 559-563 (`_node_to_proto`, `_observation_to_proto`, `_truth_to_proto`)

### 3.2 Rust → Python (per-frame, via gRPC `IngestFrameResponse`)

```
Rust output                Python receives
─────────────────────────────────────────────────────────────────
TrackState list            → TrackingService.tracks dict (cache)
ObservationRejection list  → PlatformFrame.rejected_observations
NodeState list             → PlatformFrame.nodes (echo)
TruthState list            → PlatformFrame.truths (echo)
PlatformMetrics            → PlatformFrame.metrics
generation_rejections      → PlatformFrame.generation_rejections (Python-set before call)
```

**Deserialisation:** `service.py` lines 326-340 (`_frame_from_proto`, `_track_from_proto`)

### 3.3 Python → Viewer (offline, via replay JSON)

All simulation state that the viewer needs is written to the replay JSON file.  The
Rust viewer does **not** communicate with the Rust tracking daemon at runtime — it
only reads the replay file.

Key viewer-side deserialisers: `rust/tracker-viewer/src/replay.rs`

---

## 4. State Not Crossing the gRPC Boundary

The following state exists only in Python and is never seen by Rust:

| Concept | Location |
|---------|----------|
| Terrain model | `terrain.py`, `environment.py` |
| Obstacle geometry | `obstacles.py`, `environment.py` |
| Land cover | `environment.py` |
| Weather | `weather.py` |
| Drone flight controller state | `sim.py` `FollowPathController._state` |
| Target trajectory functions | `behaviors.py` |
| Mission zones (definitions) | `models.py` `MissionZone`, `sim.py` |
| Launch events | `models.py` `LaunchEvent` |
| Path planner cache | `planning.py` `PathPlanner2D` |
| Sensor noise parameters | `config.py` `SensorConfig`, `sensor_models.py` |
| Scenario options / map preset | `sim.py` `ScenarioOptions` |

---

## 5. Timing Assumptions

| Concept | Rate / Assumption |
|---------|-------------------|
| Simulation step | `DynamicsConfig.default_dt_s` = 0.25 s (configurable via `--dt-s`) |
| Default duration | `DynamicsConfig.default_duration_s` = 180 s |
| gRPC `IngestFrame` call | Once per simulation step (synchronous, blocking) |
| Track prediction | Every frame for all alive tracks inside Rust (even without observations) |
| Stale track deletion | After `max_stale_steps` (default 8) missed frames or `max_coast_seconds` (default 5.0 s) |
| Track confirmation | M=3 updates in N=5 frames (M/N confirmation) |
| Health polling | On demand via `Health` RPC; not streamed per-frame |
| Viewer replay | No timing constraint; driven by UI scrubbing or auto-play |

---

## 6. Known Ownership Ambiguities

1. **`generation_rejections` field ownership**: Python generates them and sets them on
   `PlatformFrame`, but they travel through the proto response path
   (`PlatformFrame.generation_rejections` in proto line 111) as if Rust generated
   them.  In practice Python fills this field in `build_replay_document()` by merging
   the `ObservationBatch.generation_rejections` into the frame before the frame is
   serialised.  The gRPC `IngestFrame` response carries them back only because
   `PlatformFrame` proto includes that field — it is currently a pass-through
   populated by Python before the call.

2. **Sensor type / FOV on NodeState**: These fields exist on the Python `NodeState`
   dataclass (`models.py` lines 24-26) and are exported to replay JSON, but the proto
   `NodeState` message does not include `sensor_type` or `fov_half_angle_deg`.  Rust
   never sees these fields.

3. **`TrackingService.tracks` cache**: Python keeps a `Dict[str, TrackState]` on the
   service object (service.py line 384) as a convenience cache.  It is updated
   synchronously after each `ingest_frame()` call.  If `retain_history=False`, there
   is no persistent track history in Python; only the latest frame is available.

4. **Fusion module (`fusion.py`)**: A Python `KalmanTrack3D` and
   `triangulate_bearings()` function exist in `fusion.py`.  These duplicate the Rust
   tracking logic but are not called from `sim.py` or `service.py` in the normal
   simulation path.  Their relationship to the Rust engine is unclear — they may be
   vestiges of an earlier pure-Python implementation or testing utilities.
