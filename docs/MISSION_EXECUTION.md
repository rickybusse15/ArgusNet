# MISSION_EXECUTION.md — Closed-Loop Mission Runtime Contract

This document defines how ArgusNet should execute missions by connecting mapping, localization, inspection, planning, trajectory generation, safety validation, indexing, and evaluation into one closed-loop runtime.

The purpose of this document is to prevent ArgusNet from becoming a set of disconnected modules. A real mission should operate as a supervised, bounded loop where the system updates its belief about the world, localizes itself inside that belief, chooses useful actions, validates those actions, executes them, and records evidence.

---

## 1. Purpose

The mission execution subsystem answers:

- What is the current mission objective?
- What state is authoritative right now?
- Is the platform localized well enough to act?
- What part of the world needs mapping, inspection, or revisit?
- Which action should be attempted next?
- Is the planned action safe and inside constraints?
- What should be logged, indexed, and evaluated?
- When should the mission stop, pause, return home, or ask for operator approval?

Mission execution is the runtime layer that coordinates the other ArgusNet subsystems.

---

## 2. Core runtime loop

ArgusNet should execute missions as a closed loop:

```text
Mission constraints / operator objective
        ↓
Sensor ingestion
        ↓
Localization update
        ↓
Mapping update
        ↓
Indexing / memory update
        ↓
Planning decision
        ↓
Trajectory generation
        ↓
Safety validation
        ↓
Command execution
        ↓
Evidence / metrics / replay logging
        ↓
Repeat
```

The loop should continue until a stop condition is reached.

---

## Current implementation bridge

The full closed-loop executive below is roadmap architecture. The current runtime bridge is
`scan_map_inspect` inside `src/argusnet/simulation/sim.py`:

- Mapping state is populated from `CoverageMap` / `WorldMap`.
- Localization state is populated from `GridLocalizer`.
- Inspection state is handled through `InspectionPOI`, `POIStatus`, and `POIManager`.
- Coordinator election is wired through `CoordinationManager.elect_coordinator()`.
- Frontier completion uses `FrontierPlanner.find_gap_cells()`.
- Deconfliction events are emitted through `src/argusnet/planning/deconfliction.py`.
- The mission phase sequence is `scanning -> localizing -> inspecting -> egress -> complete`.

`src/argusnet/mission/execution.py` is a foundation for the future executive. It is not yet the
single authority for all simulation motion.

---

## 3. Authority rule

Mission execution must not directly command motion from high-level intent.

All actions should pass through:

```text
mission intent
   → candidate task
   → candidate route/viewpoint
   → trajectory proposal
   → safety validation
   → executable command
```

This preserves the architecture rule that mission intent must pass through planning, trajectory, and safety layers before execution.

---

## 4. Runtime state inputs

Mission execution consumes state from several subsystems.

| State | Source | Use |
|------|--------|-----|
| Mission constraints | Operator / mission config | Defines geofence, limits, goals |
| BeliefWorldModel | Mapping | Target authority for known, unknown, unsafe, and uncertain regions; current bridge is `CoverageMap` / `WorldMap` |
| LocalizationState | Localization | Determines whether platform can act safely; current bridge is aggregate replay state plus `LocalizationEstimate` |
| Inspection POI/site | Inspection | Roadmap persistent task model; current bridge is `InspectionPOI` / `POIStatus` |
| Indexed memory | Indexing | Retrieves prior maps, keyframes, POIs, and evidence |
| Platform health | Runtime / safety | Determines energy, comms, sensor health |
| Detections / fused object states | Sensing and fusion | Dynamic objects, if used |
| Weather / environment | World model | Affects sensing, safety, and execution |

Mission execution should never treat any single subsystem as globally authoritative for all decisions.

---

## 5. Mission state model

ArgusNet should maintain an explicit mission state.

```text
MissionState
  mission_id: str
  status: created | initializing | active | paused | returning_home | completed | aborted | failed
  objective_type: map | inspect | search | revisit | mixed
  geofence_id: str
  launch_position_m: [x, y, z]
  return_home_position_m: [x, y, z]
  active_platform_ids: list[str]
  active_task_id: optional str
  active_plan_id: optional str
  active_trajectory_id: optional str
  active_safety_state: optional str
  coverage_goal: optional float
  battery_reserve_fraction: float
  localization_required: bool
  operator_approval_required: bool
```

Mission state should be logged so that replay and evaluation can reconstruct why the system acted the way it did.

---

## 6. Mission constraints

Mission constraints are hard or soft bounds on behavior.

```text
MissionConstraints
  geofence: polygon or radius
  max_altitude_m: float
  min_terrain_clearance_m: float
  max_range_from_launch_m: optional float
  battery_reserve_fraction: float
  max_speed_mps: optional float
  comms_required: bool
  allowed_sensors: list[str]
  allowed_operation_modes: list[str]
  human_approval_required_for_boundary_change: bool
```

Hard constraints must be enforced by the safety layer. Soft constraints may be used by the planner as costs.

---

## 7. Task model

The mission executive should convert high-level objectives into explicit tasks.

```text
MissionTask
  task_id: str
  task_type: map_frontier | inspect_poi | relocalize | return_home | hold | revisit | operator_review
  priority: int
  poi_ref: optional str
  required_localization_confidence: float
  required_world_confidence: optional float
  status: pending | active | blocked | complete | failed
  reason: optional str
```

Examples:

- map the next frontier cell;
- relocalize after startup;
- inspect a roof vent POI;
- revisit a previously inspected tower;
- return home due to battery reserve;
- hold because localization is lost.

---

## 8. Mission phases

A mission may move through several phases.

| Phase | Purpose |
|------|---------|
| initialize | Load config, geofence, prior world model, and platform state |
| localize | Establish or recover map-relative pose |
| map | Build or update the belief world |
| inspect | Capture evidence for known POIs |
| index | Store observations, evidence, keyframes, and reconstructions |
| evaluate | Compute progress and quality metrics |
| return_home | Safely return to launch or recovery zone |
| complete | Close out mission logs and results |

Not all missions use every phase. Mixed missions may alternate between mapping and inspection.

---

## 9. Decision priority order

When multiple actions are possible, mission execution should use a priority order.

1. Safety emergency or abort condition.
2. Return-home requirement due to battery or comms.
3. Localization recovery if pose is insufficient.
4. Active inspection capture that is already in progress.
5. High-priority inspection or revisit task.
6. Frontier mapping to improve world belief.
7. Additional uncertainty reduction.
8. Hold or request operator review.

Safety and localization gates should be checked before normal mission progress.

---

## 10. Safety gate

Every proposed action must pass through safety validation.

The safety layer should check:

- geofence compliance;
- terrain clearance;
- obstacle collision risk;
- localization confidence;
- battery reserve and return-home feasibility;
- communications constraints;
- altitude limits;
- route continuity;
- unknown-space risk policy;
- platform health.

The safety layer may reject a plan even if the planner selected it.

---

## 11. Operator supervision

ArgusNet should remain human-supervised for real-world deployment.

Operator approval should be required for:

- expanding or changing the geofence;
- entering a higher-risk mode;
- ignoring a blocked POI;
- continuing after repeated localization failure;
- accepting low-confidence inspection results;
- using real-world adapters outside simulation.

The system should preserve the reason why approval was requested.

---

## 12. Runtime outputs

Mission execution should produce:

- mission log;
- replay frames;
- mapping updates;
- localization history;
- inspection evidence and results;
- indexed keyframes and artifacts;
- safety events;
- task history;
- evaluation metrics;
- operator decisions.

These outputs should support replay, debugging, benchmarking, and future mission reuse.

---

## 13. Failure and fallback behavior

The mission executive should handle failures explicitly.

| Failure | Expected response |
|--------|-------------------|
| localization lost | hold, climb if safe, relocalize, or return using fallback |
| battery low | stop new tasks and return home |
| POI blocked | mark blocked with reason, choose alternate POI |
| route unsafe | request alternate plan or hold |
| geofence boundary reached | stop expansion or request operator approval |
| comms degraded | return, relay, or hold depending on policy |
| sensor failure | switch sensor, reduce task scope, or abort |
| no safe frontier | complete mapping or request review |

Failures should be recorded as structured events, not silent state transitions.

---

## 14. Integration with indexing

Mission execution should write important runtime products into the indexing subsystem:

- keyframes;
- localization landmarks;
- inspection evidence;
- local reconstructions;
- mission events;
- safety events;
- task decisions;
- map tiles;
- coverage history.

Indexing makes the mission reusable instead of disposable.

---

## 15. Evaluation metrics

Mission-level evaluation should include:

- mission completion status;
- coverage achieved;
- inspection POIs completed;
- time spent localized;
- time spent mapping vs inspecting;
- number of safety rejections;
- number of operator interventions;
- route efficiency;
- battery margin at completion;
- failed or blocked tasks;
- evidence quality;
- reconstruction completeness;
- return-home success.

Ground truth may be used in simulation for scoring after actions are taken, but it must not be used as planner input for physical-mode tests.

---

## 16. Implementation phases

### Phase 1 — Mission state and task model

- Add `MissionState`.
- Add `MissionTask`.
- Add mission status transitions.
- Log task decisions.

### Phase 2 — Runtime loop

- Connect sensor ingestion, localization, mapping, planning, safety, and logging.
- Add a deterministic simulation loop using the same interfaces expected in physical runtime.

### Phase 3 — Safety and fallback integration

- Require safety validation before execution.
- Add structured failure responses.
- Add return-home and hold states.

### Phase 4 — Inspection and indexing integration

- Connect inspection tasks to mission execution.
- Store evidence, reconstructions, and task results in the index.

### Phase 5 — Real-world adapter readiness

- Add operator supervision hooks.
- Add adapter boundary for physical platforms.
- Add physical-mode tests that prevent ground truth access.

---

## 17. Success criteria

Mission execution is successful when ArgusNet can:

- start from a mission boundary and objective;
- localize or relocalize before acting;
- map unknown space inside the geofence;
- inspect map-relative POIs;
- store evidence and reconstructions;
- reject unsafe routes;
- return home when required;
- log why decisions were made;
- replay and evaluate the full mission;
- run the same high-level loop in simulation and physical adapters.

At that point, ArgusNet becomes a closed-loop world-modeling mission system rather than a collection of isolated simulation tools.
