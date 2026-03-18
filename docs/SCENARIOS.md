# SCENARIOS.md — Evaluation Model, Benchmark Scenarios, and Regression Integration

This document defines the evaluation model (Section 19 of the architecture update plan),
the benchmark scenario families, the regression test integration approach, and the replay
and experiment logging requirements.

All metric names are canonical identifiers used in replay metadata, CSV exports
(`METRICS_CSV_FIELDS` in `sim.py`), and the Rust evaluation harness.

---

## 1. Evaluation Metrics

Each metric has a canonical name, unit, computation source, threshold direction, and the
scenario phases over which it is computed.

### 1.1 Track Performance Metrics

```
Metric: time_to_reacquire_s
  Unit:             seconds
  Direction:        lower is better
  Definition:       After a track enters stale_steps > 0, the elapsed simulation time
                    until stale_steps returns to 0 for that track_id.
                    If reacquisition never occurs, value = mission_duration_s.
  Computed from:    TrackState.stale_steps across consecutive frames
  Aggregation:      mean and 95th percentile across all track-loss events in the replay
  Threshold:        mean <= 10.0 s (pass), mean > 30.0 s (fail)
  CSV column:       stale_steps (from METRICS_CSV_FIELDS in sim.py)
```

```
Metric: track_continuity_fraction
  Unit:             fraction [0.0, 1.0]
  Direction:        higher is better
  Definition:       For each (target_id, evaluation_window) pair defined in a
                    MissionObjective, the fraction of simulation steps in the window
                    during which the target has at least one accepted bearing observation.
                    Computed as: steps_with_observations / total_steps_in_window.
  Computed from:    BearingObservation.target_id and PlatformFrame timestamps
  Aggregation:      per-target and mission-wide mean
  Threshold:        >= 0.80 for required objectives, >= 0.60 for bonus objectives
```

```
Metric: localisation_error_m
  Unit:             metres (RMSE)
  Direction:        lower is better
  Definition:       Root mean square of ||TrackState.position - TruthState.position||
                    for matched (track_id, target_id) pairs at each frame.
                    Track-to-truth matching by minimum distance at mission start.
  Computed from:    TrackState.position, TruthState.position
  Aggregation:      per-track time series; mission RMSE = sqrt(mean of squared errors)
  Threshold:        mission RMSE <= 15.0 m (pass), > 50.0 m (fail)
  CSV column:       error_m (from METRICS_CSV_FIELDS in sim.py)
```

```
Metric: covariance_reduction_fraction
  Unit:             fraction [0.0, 1.0]
  Direction:        higher is better
  Definition:       (trace(P_initial) - trace(P_final)) / trace(P_initial)
                    where P is the 3x3 position block of TrackState.covariance (first 9
                    elements of the flattened covariance array, reshaped to 3x3).
                    Computed per track from first update to mission end.
  Computed from:    TrackState.covariance, TrackState.update_count
  Aggregation:      mean across all tracks that received >= 5 updates
  Threshold:        mean >= 0.50 (50 % covariance reduction from initial)
```

### 1.2 System Reliability Metrics

```
Metric: false_handoff_rate
  Unit:             events per minute
  Direction:        lower is better
  Definition:       A handoff is false when drone role reassignment occurs but the
                    incoming drone's first observation of the target has localisation_error_m
                    > 2 × the outgoing drone's last localisation_error_m.
                    Rate = false_handoff_count / (mission_duration_s / 60.0).
  Computed from:    LaunchEvent records + TrackState error sequence
  Aggregation:      scalar per replay
  Threshold:        <= 0.5 false handoffs/minute
```

```
Metric: infeasible_path_rejection_count
  Unit:             integer count
  Direction:        lower is better
  Definition:       Number of times PathPlanner2D.plan_route() returned None OR the
                    safety gate in PlannedTrajectory conversion rejected a plan.
                    (See PLANNING.md Section 3.2 step 5.)
  Computed from:    Planner invocation log; stored in replay metadata
  Aggregation:      total count per replay, plus breakdown by rejection cause
  Threshold:        <= 2 per drone per mission (warning at > 0)
```

```
Metric: safety_override_count
  Unit:             integer count
  Direction:        lower is better
  Definition:       Number of times the PlannedTrajectory safety gate fired and forced
                    a drone to retain its previous trajectory.
                    Each event is logged with drone_id, timestamp_s, and override_reason.
  Computed from:    PlannedTrajectory.override_reason field (non-None entries)
  Aggregation:      total per replay
  Threshold:        0 is target; any non-zero value is flagged in the evaluation report
```

```
Metric: comms_dropout_count
  Unit:             integer count
  Direction:        lower is better
  Definition:       Number of simulation steps where any drone exceeds
                    MissionConstraints.comms_range_m from its nearest ground station or
                    peer drone. Counts consecutive steps as separate events only if
                    connectivity was restored in between.
  Computed from:    NodeState.position for all mobile nodes + ground station positions
  Aggregation:      total events and total dropout duration (seconds) per replay
  Threshold:        total dropout duration <= 5.0 s per drone per mission
```

### 1.3 Mission Outcome Metrics

```
Metric: mission_completion_rate
  Unit:             fraction [0.0, 1.0]
  Direction:        higher is better
  Definition:       Fraction of required MissionObjectives whose ObjectiveCondition
                    was fully satisfied by mission end.
                    Bonus objectives contribute 0.1 × their weight to the rate when satisfied.
  Computed from:    MissionObjective records + per-objective evaluation against replay frames
  Aggregation:      scalar per replay
  Threshold:        >= 1.0 (all required objectives) for a passing run
```

```
Metric: energy_reserve_fraction
  Unit:             fraction [0.0, 1.0]
  Direction:        higher is better
  Definition:       Minimum energy reserve across all active drones at mission end.
                    Energy model: linear depletion proportional to distance flown.
                    energy_reserve = 1.0 - (total_distance_m / (speed_mps × max_endurance_s))
                    where max_endurance_s is a platform parameter (default: 600 s for baseline).
  Computed from:    NodeState.position time series for mobile nodes
  Aggregation:      minimum across all drones; also reported per-drone
  Threshold:        >= 0.10 (10 % reserve) for all drones at mission end
```

### 1.4 Metric Summary Schema

The `EvaluationReport` is written to `replay_metadata["evaluation"]` in the replay JSON
and is also emitted as a separate `<scenario_name>_eval.json` file.

```
EvaluationReport
  scenario_name:                  str
  mission_type:                   str
  difficulty:                     float
  seed:                           int
  duration_s:                     float

  # Track performance
  time_to_reacquire_mean_s:       float | None
  time_to_reacquire_p95_s:        float | None
  track_continuity_mean:          float
  track_continuity_per_target:    dict[str, float]    # keyed by target_id
  localisation_rmse_m:            float | None
  localisation_rmse_per_track:    dict[str, float]    # keyed by track_id
  covariance_reduction_mean:      float | None

  # Reliability
  false_handoff_rate:             float
  infeasible_path_rejection_count: int
  safety_override_count:          int
  comms_dropout_count:            int
  comms_dropout_duration_s:       float

  # Mission outcome
  mission_completion_rate:        float
  required_objectives_met:        int
  required_objectives_total:      int
  energy_reserve_min:             float
  energy_reserve_per_drone:       dict[str, float]    # keyed by node_id

  # Pass/fail summary
  passed:                         bool
  failure_reasons:                list[str]
  tags:                           list[str]
  generated_at_utc:               str                 # ISO 8601
```

---

## 2. Benchmark Scenario Families

Each family is a named collection of `MissionSpec` instances that collectively stress a
specific subsystem. All benchmark runs use a fixed set of seeds (7, 42, 137, 9999, 31415)
to ensure reproducibility.

### Family A: `baseline_coverage`

Purpose: Verify basic tracking coverage on the simplest scenario configuration.

| Name | map_preset | terrain | drone_count | target_count | difficulty | mission_type |
|---|---|---|---|---|---|---|
| `baseline_small_1t` | small | default | 2 | 1 | 0.1 | surveillance |
| `baseline_medium_2t` | medium | default | 3 | 2 | 0.2 | surveillance |
| `baseline_alpine_2t` | medium | alpine | 3 | 2 | 0.3 | surveillance |

Pass criterion: `track_continuity_mean >= 0.80`, `localisation_rmse_m <= 20.0`.

### Family B: `intercept_stress`

Purpose: Exercise the intercept template, follow-mode planner, and corridor_watcher role.

| Name | map_preset | terrain | drone_count | target_count | difficulty | mission_type |
|---|---|---|---|---|---|---|
| `intercept_easy` | medium | alpine | 2 | 1 | 0.2 | intercept |
| `intercept_medium` | large | alpine | 3 | 2 | 0.5 | intercept |
| `intercept_hard` | large | alpine | 4 | 2 | 0.8 | intercept |
| `intercept_evasive` | large | alpine | 4 | 2 | 0.9 | intercept |

Pass criterion: `track_continuity_mean >= 0.75`, `false_handoff_rate <= 0.5`.

### Family C: `persistent_long`

Purpose: Stress energy management, comms, and long-horizon replanning.

| Name | map_preset | duration_s | drone_count | target_count | difficulty | mission_type |
|---|---|---|---|---|---|---|
| `persist_300s` | large | 300 | 3 | 2 | 0.4 | persistent_observation |
| `persist_600s` | regional | 600 | 4 | 3 | 0.5 | persistent_observation |
| `persist_relay` | regional | 600 | 5 (1 relay) | 3 | 0.6 | persistent_observation |

Pass criterion: `energy_reserve_min >= 0.10`, `comms_dropout_duration_s <= 5.0`,
`track_continuity_mean >= 0.85`.

### Family D: `search_acquisition`

Purpose: Exercise search mode, lawnmower pattern, and initial track acquisition.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|---|---|---|---|---|---|
| `search_clear` | medium | default | 2 | 0.2 | search |
| `search_urban` | medium | urban | 2 | 0.5 | search |
| `search_fog` | large | default | 3 | 0.6 | search |

Pass criterion: `time_to_reacquire_mean_s <= 30.0`, `mission_completion_rate >= 1.0`.

### Family E: `planner_adversarial`

Purpose: Force infeasible path rejection and safety overrides; verify graceful degradation.

| Name | Description |
|---|---|
| `blocked_corridor` | FlightCorridor inserted through a dense building cluster |
| `shrinking_exclusion` | Exclusion zone radius grows at runtime, triggering repeated replanning |
| `drone_failure` | Reserve drone activation; primary_observer disabled at t=30 s |

Pass criterion: `safety_override_count = 0` (plans degrade but never install unsafe routes),
`infeasible_path_rejection_count` logged but does not prevent mission progress.

---

## 3. Regression Test Integration

### 3.1 Test Structure

Regression tests live in `tests/` and use `pytest`. Each benchmark family maps to a
parametrized test module:

```
tests/test_eval_baseline.py       # Family A
tests/test_eval_intercept.py      # Family B
tests/test_eval_persistent.py     # Family C
tests/test_eval_search.py         # Family D
tests/test_eval_adversarial.py    # Family E
```

Each test:
1. Constructs the `MissionSpec` with a fixed seed.
2. Calls the mission generator to produce a `GeneratedMission`.
3. Calls `run_simulation()` with `SimulationConfig.from_duration()`.
4. Calls the evaluation harness to produce an `EvaluationReport`.
5. Asserts the family-specific pass criteria.

### 3.2 Determinism Requirement

All benchmark scenarios must produce identical `EvaluationReport` values across runs
with the same seed. This is guaranteed by:
- Using `SimulationConfig.seed` as the sole RNG seed.
- Propagating the seed to `TurbulenceModel`, `EvasiveBehavior`, and all procedural
  generators in the mission generator.
- The existing `DynamicsConfig.default_seed = 7` convention is the default; benchmark
  suites override with the fixed seed list `[7, 42, 137, 9999, 31415]`.

Determinism is tested by the existing `tests/test_determinism.py` pattern: run the same
scenario twice and assert `SimulationResult.frames` are element-wise identical.

### 3.3 Threshold Pinning

Each benchmark scenario stores a `golden_eval.json` file in `tests/golden/` containing
the expected `EvaluationReport` values from the last accepted run. The regression test
asserts:
- `passed == True` (hard).
- All scalar metrics within ±5 % of the golden value (soft; prints a warning but does
  not fail the test unless the deviation exceeds ±20 %).
- `failure_reasons` list is empty.

Golden files are regenerated by running `pytest --update-golden` and committing the
resulting diffs.

### 3.4 CI Integration

The benchmark suite is split into fast (`duration_s <= 60`) and slow (`duration_s > 60`)
groups via pytest marks:

```
@pytest.mark.benchmark_fast   # Families A, D, subset of B
@pytest.mark.benchmark_slow   # Families C, E, long intercepts
```

CI runs `benchmark_fast` on every PR. `benchmark_slow` runs nightly and on release branches.

---

## 4. Replay and Experiment Logging Requirements

### 4.1 Replay Metadata Extensions

The existing `replay-schema.json` (docs/) is extended with an `evaluation` block at the
top level. The extension is additive (the schema uses `"additionalProperties": true`):

```json
"evaluation": {
  "type": "object",
  "description": "EvaluationReport for this replay, if computed.",
  "properties": {
    "scenario_name":                   { "type": "string" },
    "mission_type":                    { "type": "string" },
    "difficulty":                      { "type": "number", "minimum": 0, "maximum": 1 },
    "seed":                            { "type": "integer" },
    "duration_s":                      { "type": "number" },
    "time_to_reacquire_mean_s":        { "type": ["number", "null"] },
    "time_to_reacquire_p95_s":         { "type": ["number", "null"] },
    "track_continuity_mean":           { "type": "number", "minimum": 0, "maximum": 1 },
    "track_continuity_per_target":     { "type": "object" },
    "localisation_rmse_m":             { "type": ["number", "null"] },
    "localisation_rmse_per_track":     { "type": "object" },
    "covariance_reduction_mean":       { "type": ["number", "null"] },
    "false_handoff_rate":              { "type": "number", "minimum": 0 },
    "infeasible_path_rejection_count": { "type": "integer", "minimum": 0 },
    "safety_override_count":           { "type": "integer", "minimum": 0 },
    "comms_dropout_count":             { "type": "integer", "minimum": 0 },
    "comms_dropout_duration_s":        { "type": "number", "minimum": 0 },
    "mission_completion_rate":         { "type": "number", "minimum": 0, "maximum": 1 },
    "required_objectives_met":         { "type": "integer", "minimum": 0 },
    "required_objectives_total":       { "type": "integer", "minimum": 0 },
    "energy_reserve_min":              { "type": "number", "minimum": 0, "maximum": 1 },
    "energy_reserve_per_drone":        { "type": "object" },
    "passed":                          { "type": "boolean" },
    "failure_reasons":                 { "type": "array", "items": { "type": "string" } },
    "tags":                            { "type": "array", "items": { "type": "string" } },
    "generated_at_utc":                { "type": "string" }
  }
}
```

The `evaluation` block is populated by `build_replay_document()` in `replay.py` when an
`EvaluationReport` is provided.

### 4.2 Mission Spec in Replay Metadata

The `meta` block of the replay is extended with a `mission_spec` field:

```json
"mission_spec": {
  "type": "object",
  "description": "MissionSpec used to generate this scenario.",
  "properties": {
    "seed":               { "type": "integer" },
    "terrain_preset":     { "type": "string" },
    "weather_preset":     { "type": "string" },
    "map_preset":         { "type": "string" },
    "platform_preset":    { "type": "string" },
    "drone_count":        { "type": "integer" },
    "ground_station_count": { "type": "integer" },
    "target_count":       { "type": "integer" },
    "difficulty":         { "type": "number" },
    "mission_type":       { "type": "string" },
    "tags":               { "type": "array", "items": { "type": "string" } }
  }
}
```

### 4.3 Planner Event Log

A `planner_events` array is added to the replay document. Each entry records a planning
decision for auditing:

```
PlannerEvent (stored in replay as list under "planner_events")
  timestamp_s:          float
  drone_id:             str
  event_type:           str    # "plan_issued" | "plan_rejected" | "safety_override"
                               # | "replan_trigger" | "role_change"
  trigger:              str    # Trigger reason (e.g., "track_loss", "staleness_expiry")
  generation:           int    # PlannedTrajectory.generation
  route_length_m:       float | None
  override_reason:      str | None
```

### 4.4 Experiment Logging

For multi-run experiment sweeps (e.g., sweeping `difficulty` across the full [0, 1] range),
results are written to `mlruns/` using the existing MLflow directory layout already present
at the project root. Each run logs:

- All scalar fields from `EvaluationReport` as MLflow metrics.
- `MissionSpec` fields as MLflow params.
- The replay `.json` file as an artifact.
- The `_eval.json` report as an artifact.

The experiment name matches the benchmark family name (e.g., `intercept_stress`).
Run names use the format `<scenario_name>_seed<seed>`.
