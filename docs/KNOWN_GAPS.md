# Known Gaps — Smart Trajectory Tracker

> Stage 0 Architecture Audit — 2026-03-15
>
> This document catalogues features that are absent, partially implemented, or whose
> boundaries are not yet formalised, mapped against the six proposed subsystems:
> **World, Sensing/Fusion, Mission, Planning, Trajectory, Evaluation**.

---

## 1. Subsystem Mapping — Current Code vs Proposed Architecture

### 1.1 World Subsystem

Covers: terrain, obstacles, land cover, weather, CRS/geodetics.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Analytic terrain model (procedural) | **Exists** | `terrain.py` `TerrainModel` |
| Tiled heightmap (runtime query) | **Exists** | `environment.py` `TerrainLayer` |
| Real DEM ingest (GeoTIFF) | **Exists** (partial) | `environment.py` `TerrainLayer.from_geotiff()`, `_scene_gis.py` |
| Obstacle primitives (buildings, walls, forests) | **Exists** | `obstacles.py` |
| Land cover layer | **Exists** (stub default) | `environment.py` `LandCoverLayer` |
| Seasonal variation | **Exists** (data structure only) | `environment.py` `SeasonalVariation` |
| Weather model (wind, precipitation, cloud) | **Exists** | `weather.py` `WeatherModel` |
| Weather affecting sensor noise | **Exists** | `visibility.py` `compute_weather_factor()` |
| Weather affecting flight dynamics (wind drift on targets/drones) | **Absent** | — |
| Formal CRS / geodetic projection pipeline | **Stub only** | `environment.py` `EnvironmentCRS` holds identifiers but no projection math for simulation |
| World state serialisation to Rust tracking engine | **Absent** | Terrain/obstacles not transmitted over gRPC |
| World state API (queryable at runtime by Rust) | **Absent** | — |
| Environment bundle persistence (save/load) | **Exists** | `environment_io.py` `load_environment_bundle` / `write_environment_bundle` |

**Critical gap**: The Rust tracking engine has zero knowledge of world geometry.  Terrain AGL enforcement and obstacle avoidance are Python-only.  Any future Rust-side trajectory prediction that needs terrain clearance would have to re-implement this or receive world data over gRPC.

---

### 1.2 Sensing / Fusion Subsystem

Covers: sensor models, observation generation, bearing triangulation, Kalman filter, IMM, data association.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Bearing observation synthesis (sim) | **Exists** | `sim.py` `build_observations()` |
| FOV / range / elevation gating | **Exists** | `sim.py` (Python-side rejection); also replicated in Rust `prefilter_observations()` |
| LOS occlusion (terrain, buildings, walls) | **Exists** | `visibility.py` `EnvironmentQuery.los()` |
| Vegetation attenuation model | **Exists** (partial) | `visibility.py` — transmittance computed but hard-block threshold is approximate |
| Sensor noise model (range-dependent, atmospheric, bias drift, false alarms) | **Exists** | `sensor_models.py` `SensorErrorConfig`, `SensorModel` |
| IMM Kalman filter (CV + CT modes) | **Exists** (Rust, authoritative) | `rust/tracker-core/src/lib.rs` `IMMTrack3D` |
| Python Kalman filter (duplicate) | **Exists** (orphaned) | `fusion.py` `KalmanTrack3D` — not called from sim path; relationship to Rust engine is unclear |
| Bearing triangulation (Python) | **Exists** (orphaned) | `fusion.py` `triangulate_bearings()` — not called from sim path |
| Data association: Labeled (sim default) | **Exists** | `rust/tracker-core/src/association.rs` |
| Data association: GNN | **Exists** | `rust/tracker-core/src/lib.rs` `ingest_gnn()` |
| Data association: JPDA | **Exists** | `rust/tracker-core/src/lib.rs` `ingest_jpda()` |
| Chi-squared gating | **Exists** | Rust `TrackingEngine` (`chi_squared_gate_threshold`) |
| Track lifecycle (Tentative/Confirmed/Coasting/Deleted) | **Exists** | Rust `ManagedTrack.update_lifecycle()` |
| Adaptive measurement noise | **Exists** (flag, not default) | `TrackerConfig.adaptive_measurement_noise` |
| Live MQTT ingestion | **Exists** (stub) | `ingest.py` `MqttIngestionAdapter` — functional but not integrated with `TrackingService` in a documented way |
| Clutter / false alarm generation | **Exists** (model params) | `sensor_models.py` `SensorErrorConfig.false_alarm_rate_per_scan` — params exist but it is unclear if false alarms are wired into `build_observations()` |
| Formal Sensing/Fusion subsystem API boundary | **Absent** | No interface class; `build_observations()` in `sim.py` is a 400+ line function |

---

### 1.3 Mission Subsystem

Covers: mission zone definitions, objectives, rules of engagement, drone assignment.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Mission zone data model (surveillance, exclusion, patrol, objective) | **Exists** | `models.py` `MissionZone`, zone type constants (lines 136-154) |
| Zone creation in scenario builder | **Exists** | `sim.py` `build_default_scenario()` |
| Zone serialisation to replay / viewer | **Exists** | `replay.py`, Rust viewer `replay.rs` `MissionZone` |
| Zone enforcement logic (drones avoid exclusion zones) | **Absent** | Drones have no awareness of `MissionZone` objects at runtime |
| Zone-driven task assignment (assign drone to patrol zone) | **Absent** | Drone-to-target assignments are hardcoded or simple index-based |
| Objective completion detection | **Absent** | — |
| Rules of engagement (ROE) | **Absent** | — |
| Mission replanning on event (new target, drone lost) | **Absent** | — |
| Mission state in proto | **Absent** | `tracker.proto` has no mission concepts |
| Formal Mission subsystem interface | **Absent** | — |

---

### 1.4 Planning Subsystem

Covers: 2D path planning, obstacle avoidance, drone routing.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Visibility-graph 2D path planner | **Exists** | `planning.py` `PathPlanner2D` |
| Obstacle polygon expansion (drone clearance) | **Exists** | `planning.py` `_expand_polygon()` |
| A* / Dijkstra on visibility graph | **Exists** | `planning.py` |
| Route cache (LRU) | **Exists** | `planning.py` `max_cache_entries` |
| Cooperative orbit geometry (multi-drone slots) | **Exists** | `sim.py` `FollowPathController` `slot_index` / `slot_count` |
| Terrain-following flight mode | **Exists** | `sim.py` `FollowPathController.terrain_following` |
| Lawnmower search pattern | **Exists** | `sim.py` (search drone path generation) |
| 3D path planning (altitude-aware) | **Absent** | Planner is 2D only; altitude is handled separately by AGL logic |
| Dynamic obstacle avoidance (other drones) | **Absent** | — |
| Mission-zone-aware routing | **Absent** | Planner does not know about `MissionZone` |
| Formal Planning subsystem interface | **Absent** | `PathPlanner2D` is used inline in `FollowPathController` with no abstract interface |

---

### 1.5 Trajectory Subsystem

Covers: target motion models, drone trajectory generation, trajectory prediction.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Target behavior presets (loiter, transit, evasive, sinusoid, racetrack, patrol) | **Exists** | `behaviors.py` `BEHAVIOR_PRESETS`, `build_target_trajectory()` |
| Turbulence perturbation model | **Exists** | `behaviors.py` `TurbulenceModel` |
| Flight envelope constraints | **Exists** (data model) | `behaviors.py` `FlightEnvelope` — defined but applied inconsistently across behavior classes |
| Drone follow-path trajectory | **Exists** | `sim.py` `FollowPathController` |
| Drone search (lawnmower) trajectory | **Exists** | `sim.py` |
| Drone launch / climb trajectory | **Exists** | `sim.py` `LaunchController` |
| Trajectory prediction used by Rust for association | **Absent** | Rust only uses last known track state; no forward prediction from trajectory model |
| Trajectory representation in proto | **Absent** | No trajectory or waypoint message in `tracker.proto` |
| Target intent / maneuver detection | **Absent** | IMM CT mode detects turns implicitly but there is no explicit maneuver classification |
| Formal Trajectory subsystem interface | **Absent** | `TrajectoryFn = Callable[[float], Tuple[np.ndarray, np.ndarray]]` is a type alias, not a class hierarchy |

---

### 1.6 Evaluation Subsystem

Covers: accuracy metrics, track quality scoring, performance reporting.

| Component | Status | Current Location |
|-----------|--------|-----------------|
| Per-track RMSE vs truth | **Exists** | Rust `PlatformMetrics.track_errors_m`; Python `sim.py` `metrics_rows` CSV |
| Mean / max error across active tracks | **Exists** | `PlatformMetrics.mean_error_m`, `max_error_m` |
| Observation acceptance / rejection counters | **Exists** | `PlatformMetrics` |
| Track lifecycle state reporting | **Exists** | `TrackState.lifecycle_state` |
| Quality score per track | **Exists** | `TrackState.quality_score` (M/N update ratio) |
| Node health scoring | **Exists** | Rust `NodeHealthTracker.health_score()` |
| Export to GeoJSON / CZML / KML / Foxglove MCAP / GPX / GeoPackage / Shapefile | **Exists** | `export.py` |
| Per-scenario summary statistics | **Exists** (basic) | `sim.py` `build_summary()` — mean/max error, accepted/rejected counts |
| Track continuity metrics (false tracks, missed tracks, ID switches) | **Absent** | — |
| OSPA (Optimal Sub-Pattern Assignment) or similar multi-target metric | **Absent** | — |
| Formal evaluation harness / benchmark runner | **Absent** | — |
| Evaluation results persisted independently of replay | **Absent** | Metrics only live in replay JSON frames |

---

## 2. Interface Boundaries That Need Formalisation

### 2.1 Python Observation Generator → Rust Tracking Engine

**Current state**: `sim.py` `build_observations()` is a monolithic ~400-line function
that mixes sensor geometry, noise application, rejection recording, and dropout
simulation.  Its contract with the Rust engine is entirely implicit — whatever
`BearingObservation` objects it produces are sent over gRPC.

**What is missing**:
- No interface class or abstract type for "observation source"
- No versioned schema for what constitutes a valid observation before reaching Rust
- Duplicate validation: Python gates on range/FOV/elevation; Rust re-gates on
  confidence/std/skew.  The boundary between these two layers is not documented.
- `generation_rejections` are Python-side but travel through the proto `PlatformFrame`
  field, creating confusion about who generates what.

### 2.2 Drone Controller → Path Planner

**Current state**: `FollowPathController` directly instantiates and calls
`PathPlanner2D`.  There is no controller interface — drone controllers are plain
Python dataclasses with a `__call__` method.

**What is missing**:
- No abstract `DroneController` protocol
- No way to swap planning strategies at runtime
- No integration of mission zones into planning decisions

### 2.3 Environment Model → Tracking Engine

**Current state**: Zero interface.  Rust has no world knowledge.

**What is missing**:
- If Rust is to perform terrain-aware prediction or provide drift-corrected coasting, a
  terrain query API must exist (either via gRPC extension or shared memory).
- Obstacle avoidance for drone trajectories generated by Rust (not currently done) would
  require obstacle data in Rust.

### 2.4 Mission → Planning → Trajectory Chain

**Current state**: These three subsystems are interleaved in `sim.py` with no clear
boundaries.  Mission zones are defined in `models.py` but never read by planning or
trajectory code.

**What is missing**:
- A `MissionExecutive` that holds current zone assignments and drone tasks
- A planning layer that queries mission state before computing routes
- A trajectory layer that enforces mission constraints (altitude corridors, exclusion
  zones)

### 2.5 Live Ingestion → Tracking Service

**Current state**: `ingest.py` defines `MqttIngestionAdapter` and a file-replay
adapter.  They produce `(timestamp_s, node_states, observations, truths)` tuples
matching the `OnFrameCallback` signature.  There is no documented wiring from these
adapters into `TrackingService.ingest_frame()`.

**What is missing**:
- A top-level ingestion loop (or async bridge) connecting adapters to the service
- Rate limiting / frame alignment between MQTT messages and the 4 Hz sim clock
- Truth state handling for live data (truths are typically absent in real deployments)

---

## 3. Features Mentioned in Architecture Plans That Do Not Exist

The following concepts appear in the codebase architecture documentation
(`docs/architecture.md`, `CLAUDE.md`) or in AGENT_TEAM.md's implied six-subsystem
model but have no implementation:

| Feature | Notes |
|---------|-------|
| Drone launch decisions driven by track quality | Currently `LaunchController` fires on raw range threshold, not track quality |
| Exclusion zone enforcement during flight | `MissionZone` exists but is not enforced anywhere in flight controllers |
| Mission replanning on drone loss | No drone failure model; drones run forever |
| Wind effect on drone / target dynamics | `WeatherModel.WindModel` is computed but not applied to position updates in `sim.py` |
| Cooperative triangulation geometry optimisation | Slot geometry exists in `FollowPathController` but optimisation criterion is heuristic (visibility + route cost), not information-theoretic |
| Chi-squared track gating tied to FOV | Chi-squared gate is in tracker config but not spatially filtered by sensor FOV |
| ROS 2 ingestion | `ingest.py` references ROS 2 in docstring but has no ROS 2 implementation |
| Formal replay versioning / migration | Replay schema v1 exists (`docs/replay-schema.json`) but no migration tooling |

---

## 4. Partially Implemented Features

| Feature | What Exists | What Is Missing |
|---------|-------------|----------------|
| `fusion.py` Python Kalman tracker | Full implementation of `KalmanTrack3D`, `triangulate_bearings()` | Not called from any sim or production path; may be dead code |
| `SeasonalVariation` in environment | Data class with `foliage_density_factor` and `snow_cover` | Not read by `LandCoverLayer`, `SensorVisibilityModel`, or terrain models |
| `SensorErrorConfig` false alarm model | Parameters `false_alarm_rate_per_scan`, `clutter_bearing_std_rad`, `clutter_min/max_range_m` defined | Not wired into `build_observations()` in `sim.py`; clutter observations are never generated |
| `SensorErrorConfig` bias drift model | Parameters `bias_drift_rate_rad_per_s`, `bias_drift_max_rad` defined | Not wired into `build_observations()`; sensors have no persistent bias state |
| `adaptive_measurement_noise` in tracker | Flag exists and code path exists in Rust (`IMMTrack3D.update_position()`) | Disabled by default; never enabled in standard scenario builders |
| `coordinates.py` geodetic transforms | `wgs84_to_enu`, `enu_to_wgs84`, `ENUOrigin` | ENU origin is never set in simulation scenarios; all coordinates are purely local |
| GeoTIFF DEM ingest | `TerrainLayer.from_geotiff()` and `_scene_gis.py` | No CLI flag to select an external DEM; requires manual API use |
| `tracker-viewer` headless mode | `rust/tracker-viewer/src/headless.rs` exists | Not tested or documented as a CI/render path |

---

## 5. Proto / Interface Version State

| Concern | Status |
|---------|--------|
| Proto file version | No version field in `tracker.proto` |
| `NodeState` in proto missing `sensor_type`, `fov_half_angle_deg`, `max_range_m` | These fields exist in Python `NodeState` and replay JSON but are absent from the gRPC message |
| `PlatformFrame.generation_rejections` | Present in proto but semantically these are Python-generated, not Rust-generated |
| `TrackerConfig` has no version field | Any field addition is a breaking change if the YAML config format diverges |
| Replay JSON schema | `docs/replay-schema.json` Draft 7 schema exists; version pinned in `meta.schema_version` at Python write time |
