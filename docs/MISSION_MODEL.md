# MISSION_MODEL.md — Mission Generation Schema

This document defines the procedural mission generation schema (Section 8 of the architecture
update plan) and the scenario taxonomy used throughout the Smart Trajectory Tracker platform.
All field names match the Python dataclass fields used in `sim.py`, `models.py`, and `config.py`
unless explicitly noted as new additions.

---

## 1. Mission Generation Inputs

A `MissionSpec` is the seed document consumed by the mission generator to produce a fully
resolved `ScenarioDefinition`. Every field is typed and validated before scenario construction
begins.

```
MissionSpec
  seed:               int          # Deterministic RNG seed; propagates to ScenarioDefinition
  terrain_preset:     str          # One of KNOWN_TERRAIN_PRESETS (terrain.py)
  weather_preset:     str          # One of KNOWN_WEATHER_PRESETS (weather.py)
  map_preset:         str          # Key into DEFAULT_MAP_PRESET_SCALES (config.py)
  platform_preset:    str          # "baseline" | "wide_area"
  drone_count:        int          # 1..12
  ground_station_count: int        # 1..12
  target_count:       int          # 1..8
  difficulty:         float        # 0.0 (easiest) .. 1.0 (hardest); drives scaling model
  mission_type:       str          # See Section 3: template types
  tags:               list[str]    # Free-form taxonomy tags; see Section 5
  timing:             MissionTiming
  constraints:        MissionConstraints
```

`MissionTiming` captures the temporal bounds of the generated scenario:

```
MissionTiming
  duration_s:         float        # Total mission wall time (positive, finite)
  dt_s:               float        # Simulation step size; default 0.25 s (DynamicsConfig)
  start_offset_s:     float        # Simulation time at which objectives become active
  deadline_s:         float | None # Latest acceptable mission completion time; None = open
```

`MissionConstraints` captures hard physical and sensor requirements that must hold throughout:

```
MissionConstraints
  min_sensor_baseline_m:    float  # Minimum triangulation baseline for localisation quality
  max_target_covariance_m2: float  # Upper bound on 3x3 position covariance trace at handoff
  min_active_tracks:        int    # At least this many targets must be tracked continuously
  exclusion_zones:          list[MissionZone]  # MissionZone with zone_type="exclusion"
  terrain_clearance_m:      float  # Minimum AGL for all drone trajectories
  comms_range_m:            float  # Maximum inter-drone and drone-to-station link range
```

`MissionZone` is the existing frozen dataclass from `models.py`:

```
MissionZone
  zone_id:    str      # Unique identifier
  zone_type:  str      # "surveillance" | "exclusion" | "patrol" | "objective"
  center:     Vector3  # XY local projected + Z above datum (metres)
  radius_m:   float    # Must be > 0
  priority:   int      # Higher value = higher scheduling priority
  label:      str      # Human-readable label for viewer display
```

---

## 2. Mission Generation Outputs

The generator returns a `GeneratedMission`, which wraps a `ScenarioDefinition` with additional
mission-level metadata that survives into the replay document.

```
GeneratedMission
  scenario_def:         ScenarioDefinition     # Direct input to run_simulation()
  spec:                 MissionSpec            # The original spec (for reproducibility)
  launch_points:        list[LaunchPoint]
  objectives:           list[MissionObjective]
  corridors:            list[FlightCorridor]
  timing:               MissionTiming          # May differ from spec if generator clamped values
  tags:                 list[str]
  validity_report:      ValidityReport
```

### 2.1 Launch Points

```
LaunchPoint
  launch_id:            str
  station_id:           str        # References a SimNode.node_id with is_mobile=False
  position:             Vector3    # 3D ENU position of the launch pad
  assigned_drone_ids:   list[str]  # Which drone node IDs launch from here
  earliest_launch_s:    float      # Earliest activation time
  latest_launch_s:      float | None
```

### 2.2 Mission Objectives

```
MissionObjective
  objective_id:         str
  objective_type:       str        # "acquire" | "maintain" | "handoff" | "neutralize" | "survey"
  target_ids:           list[str]  # Which SimTarget.target_id values this covers
  zone:                 MissionZone | None
  start_s:              float      # Earliest time this objective becomes evaluable
  end_s:                float | None
  priority:             int
  required:             bool       # False = bonus objective
  success_condition:    ObjectiveCondition
```

```
ObjectiveCondition
  track_continuity_fraction:  float  # Fraction of [start_s, end_s] target must be tracked
  max_position_error_m:       float  # 3-sigma bound on position error at objective time
  min_observation_count:      int    # Minimum bearing measurements in the window
  covariance_trace_max_m2:    float  # Maximum trace of 3x3 position covariance
```

### 2.3 Flight Corridors

```
FlightCorridor
  corridor_id:          str
  waypoints_xy_m:       list[[float, float]]  # 2D XY vertices; same format as PlannerRoute
  width_m:              float                 # Half-width on each side of the centerline
  min_agl_m:            float
  max_agl_m:            float
  direction:            str        # "inbound" | "outbound" | "bidirectional"
  assigned_drone_ids:   list[str]  # Empty = any drone may use
  active_window:        [float, float]  # [start_s, end_s]
```

---

## 3. Mission Template Types

The `mission_type` field in `MissionSpec` selects a template that pre-populates defaults for
objectives, corridors, and zone layout before difficulty scaling is applied.

### 3.1 `surveillance`

- One or more `MissionZone` with `zone_type="surveillance"` covering a fixed area.
- Objective type: `maintain` — track all targets that enter the zone for at least
  `track_continuity_fraction >= 0.85` of their dwell time.
- Drone roles biased toward `primary_observer` and `secondary_baseline`.
- Target motion: loiter or patrol within the zone.
- Baseline drone count: 2, scales with difficulty.

### 3.2 `intercept`

- Target trajectory is `transit` from an entry point to an exit point across the map.
- Objective type: `acquire` within the first 15 % of mission duration, then `maintain`.
- Optional terminal `handoff` objective when the target exits sensor coverage.
- Drone roles: at least one `follow` mode drone, one `corridor_watcher`.
- `FlightCorridor` generated along the predicted intercept path.

### 3.3 `persistent_observation`

- Long-duration scenario (`duration_s >= 300 s`).
- Multiple overlapping `surveillance` zones with different priorities.
- Objective: `maintain` track continuity >= 0.90 across all high-priority zones
  simultaneously.
- Includes at least one `relay` drone when map_preset is `large` or above.
- Target count >= 2; at least one uses `evasive` behavior.
- Energy constraint active: `energy_reserve_fraction >= 0.15` at mission end.

### 3.4 `search`

- No prior track; objective is `acquire` with `min_observation_count` triggering initial track.
- Zone type: `objective` marking the suspected search area.
- Drone roles: `search` mode with lawnmower pattern from `DynamicsConfig.drone_search_lane_spacing_scale`.
- Success condition: track established and `covariance_trace_max_m2` drops below threshold
  within `deadline_s`.

---

## 4. Validity Requirements

The generator runs `ValidityReport` checks after scenario construction and before returning
`GeneratedMission`. A scenario is rejected and regenerated (up to 8 attempts) if any hard
requirement fails.

```
ValidityReport
  physically_valid:    bool   # All trajectories above terrain; no obstacle penetration
  sensor_valid:        bool   # At least one node covers each target at mission start
  solvable:            bool   # Objectives are reachable given platform capabilities
  corridor_clear:      bool   # All FlightCorridors free of hard obstacles (PathPlanner2D)
  baseline_adequate:   bool   # max inter-drone separation >= min_sensor_baseline_m
  failures:            list[str]  # Human-readable descriptions of any failed checks
```

**Physically valid** — checked by:
- All drone AGL values >= `MissionConstraints.terrain_clearance_m` at t=0.
- No SimNode or SimTarget position inside an obstacle from `ObstacleLayer`.
- `physical collision must never push entities below terrain` (CLAUDE.md critical rule).

**Sensor valid** — checked by:
- For each `SimTarget`, at least one `SimNode` has LOS at t=0 with range <=
  `NodeState.max_range_m`.
- Uses `SensorVisibilityModel` from `environment.py`.

**Solvable** — checked by:
- `PathPlanner2D.plan_route()` returns a non-None route for every drone's initial
  assigned corridor.
- Required objective `success_condition.track_continuity_fraction` does not exceed what is
  achievable given drone count and sensor range.

---

## 5. Difficulty Scaling Model

`difficulty: float` in [0.0, 1.0] drives a family of monotonic scaling functions applied
after template defaults are resolved.

| Parameter | At difficulty=0.0 | At difficulty=1.0 | Scaling function |
|---|---|---|---|
| `target_speed_scale` | 0.5 × baseline | 1.8 × baseline | linear |
| `drone_search_speed_scale` | 1.2 × baseline | 0.8 × baseline | linear (drones slower relative to targets) |
| `bearing_std_rad` (all sensors) | 0.5 × baseline | 2.0 × baseline | linear |
| `dropout_probability` (all sensors) | 0.3 × baseline | 2.5 × baseline | linear |
| Target behavior | loiter only | evasive with high probability | step at 0.6 |
| Exclusion zone count | 0 | 3 | floor(difficulty × 3) |
| Target count cap | 1 | target_count from spec | linear |
| Mission duration multiplier | 1.5 × | 0.7 × | linear (less time at high difficulty) |
| Terrain clearance requirement | 30 m | 60 m | linear |

The baseline values for sensor noise and speed are the defaults in `SensorConfig` and
`DynamicsConfig` (config.py). All scaling is applied symmetrically so that `seed + difficulty`
fully determines the scenario.

---

## 6. Three-Phase ISR Mission (`scan_map_inspect`)

The `scan_map_inspect` mission mode runs a state machine with four phases:
`scanning → localizing → inspecting → complete`. This section documents the transition
conditions and associated design decisions.

### 6.1 Phase Transitions

```
scanning  →  localizing    when: scan_coverage_fraction >= scan_coverage_threshold
                                  AND gap_fill_fraction < gap_fill_min_fraction
                                  AND NOT all drones already converged
                                  (if already converged: skip to inspecting directly)

localizing  →  inspecting  when: all(drone.confidence >= loc_confidence_threshold)
                                  OR any drone triggers localization timeout

inspecting  →  complete    when: all POIs reach status "complete"
```

**Gap-fill gate** (`gap_fill_min_fraction`, default 2 %): even after reaching the coverage
threshold, the transition is blocked if more than 2 % of grid cells are enclosed interior
holes (fully surrounded by covered cells). This prevents declaring coverage complete when
a pocket of terrain remains inside the mapped perimeter, which would corrupt localization.
See `FrontierPlanner.find_gap_cells()` in `src/argusnet/planning/frontier.py`.

**Localization timeout**: if any drone has not converged after `LocalizationConfig.timeout_steps`
(default 200 steps ≈ 50 s at dt=0.25 s), its confidence is forced to 1.0 so the mission
can proceed. The `localization_timed_out` field in `ScanMissionState` is set to `True` when
this occurs. This is a **team-level flag** — it indicates that at least one drone advanced
via timeout rather than genuine convergence, but does not identify which drone(s).

Forced convergence is acceptable because: (a) the localization estimate is still used for
POI assignment and does not require sub-metre accuracy; (b) indefinite blocking on
localization convergence would stall the mission if a drone's coverage contribution is poor.

### 6.2 Coordinator Election

One drone is elected coordinator at the start of the `scanning` phase using the highest
battery-fraction criterion (see `CoordinationManager.elect_coordinator()` in
`src/argusnet/planning/coordination.py`). Election is one-shot; no re-election occurs
because drone failure is not modelled in the current simulation.

---

## 7. Scenario Taxonomy and Tagging

Tags are free-form strings stored in `MissionSpec.tags` and propagated to the replay metadata.
The following canonical prefixes are reserved:

| Prefix | Meaning | Example values |
|---|---|---|
| `type:` | Mission template type | `type:surveillance`, `type:intercept` |
| `terrain:` | Terrain preset | `terrain:alpine`, `terrain:urban` |
| `weather:` | Weather preset | `weather:clear`, `weather:fog` |
| `diff:` | Difficulty band | `diff:easy` (0–0.33), `diff:medium` (0.33–0.67), `diff:hard` (0.67–1.0) |
| `size:` | Map preset | `size:small`, `size:regional` |
| `role:` | Drone role emphasis | `role:relay_heavy`, `role:search_only` |
| `eval:` | Evaluation suite membership | `eval:regression`, `eval:benchmark` |
| `behavior:` | Target behavior class | `behavior:evasive`, `behavior:loiter` |

Multiple tags of the same prefix are allowed. The viewer and evaluation harness filter by
prefix:value pairs.
