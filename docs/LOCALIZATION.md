# LOCALIZATION.md — Pose Recovery and Map-Relative Localization Contract

This document defines the localization step for ArgusNet: how a drone or other mobile system should estimate its position relative to a known or partially known world model, especially when it starts with an uncertain position.

Localization is the counterpart to mapping. Mapping answers, "What does the world look like?" Localization answers, "Where am I within that world?"

This capability matters when:

- a drone powers on in a known area but does not know its exact position;
- a drone loses GNSS or starts with poor GNSS accuracy;
- a drone returns after a battery swap and must rejoin an existing mission;
- a replacement drone is launched into a partially mapped environment;
- an operator wants to send a system to a previously inspected location;
- ArgusNet must align live observations with stored spatial memory.

---

## 1. Purpose

The localization subsystem estimates the current pose of a platform relative to the active world model.

It should answer:

- Where is the drone now?
- How certain is that pose estimate?
- Which prior map region does the drone appear to be observing?
- Has the drone seen this place before?
- Is the current pose estimate good enough to plan safely?
- Can the drone navigate to a known inspection location?

Localization should produce a pose estimate with uncertainty, not just a single position.

---

## 2. Why this matters to ArgusNet

ArgusNet is designed around persistent world models and spatial memory. That requires systems to localize against prior maps, not only fly from a single continuous mission timeline.

Without localization, ArgusNet can map an area once but cannot reliably reuse that map later.

Localization enables:

- battery swap continuity;
- multi-session mapping;
- returning to known inspection points;
- persistent site memory;
- map-relative navigation;
- relocalization after GNSS dropouts;
- handoff between drones;
- safe planning from a partially known world model.

This is especially important for inspection workflows. A later inspection system should be able to say:

```text
Go to the previously mapped roof vent on Building A.
```

That command only works if the system can localize itself relative to the stored world model.

---

## 3. Relationship to mapping

Mapping and localization form a closed loop.

```text
Live observations
        ↓
Localization against current/prior world model
        ↓
Pose estimate with uncertainty
        ↓
Mapping update in the correct map frame
        ↓
Improved world model
        ↓
Better future localization
```

Mapping builds the world. Localization places the drone inside that world.

In simulation, ground truth may be used to evaluate localization error. In physical deployments, ground truth is not available and must not be assumed.

---

## 4. Core distinction: local pose, global pose, and map-relative pose

ArgusNet should distinguish several pose concepts.

| Pose type | Meaning | Reliability |
|----------|---------|-------------|
| Local odometry pose | Motion estimate since startup from IMU/visual odometry | Drifts over time |
| GNSS/global pose | Latitude/longitude/altitude or ENU pose from GNSS | Useful but may be noisy or unavailable |
| Map-relative pose | Pose aligned to the stored ArgusNet world model | Needed for spatial memory and inspection |
| Ground truth pose | True simulated pose | Simulation/evaluation only |

The runtime planner should use the best fused localization estimate, not raw ground truth.

---

## 5. Startup localization modes

ArgusNet should support multiple startup cases.

### 5.1 Known launch point

The drone starts from a known launch pad or operator-defined origin.

Initial state:

- position uncertainty is low;
- heading uncertainty may be moderate;
- map frame is already known;
- system can begin mapping or inspection quickly.

### 5.2 Known area, unknown exact position

The drone powers on somewhere inside a previously mapped area but does not know its exact position.

Initial state:

- position uncertainty is high;
- system must compare live observations to stored map features;
- movement should be cautious until localization confidence improves.

### 5.3 Weak GNSS prior

The drone has GNSS or coarse operator input but needs map-relative correction.

Initial state:

- broad search area is known;
- localization should use GNSS as a prior, not as absolute truth;
- visual/terrain/landmark matching can refine the estimate.

### 5.4 No useful prior inside geofence

The drone knows only the mission geofence and launch constraints.

Initial state:

- localization begins in a local frame;
- mapping and localization occur together;
- map-relative localization becomes meaningful after enough observations are collected.

### 5.5 Battery swap / system restart

A drone or replacement system is turned on after a previous mission has already mapped part of the environment.

Initial state:

- prior world model exists;
- current exact pose may be unknown;
- system must relocalize before accepting high-risk navigation tasks.

---

## 6. Localization inputs

The localization subsystem may use several evidence sources.

| Input | Use |
|------|-----|
| IMU | short-term motion and attitude propagation |
| GNSS | coarse global prior when available |
| barometer / altimeter | altitude constraint and terrain-relative height |
| camera keyframes | visual landmark matching and loop closure |
| depth / lidar | terrain and obstacle shape matching |
| magnetometer | heading prior, if reliable |
| known geofence | limits search area |
| prior terrain model | terrain-relative pose correction |
| prior obstacle/structure map | landmark and shape matching |
| stored inspection points | map-relative navigation targets |

No single sensor should be treated as universally authoritative. Localization should fuse evidence and track uncertainty.

---

## 7. Localization outputs

The output should be a structured localization state.

```text
LocalizationState
  platform_id: str
  timestamp_s: float
  pose_estimate:
    position_m: [x, y, z]
    orientation: quaternion or yaw/pitch/roll
    frame_id: str
  covariance: matrix or compact uncertainty representation
  confidence: float
  status: unlocalized | initializing | localized | degraded | lost
  map_region_id: optional str
  matched_landmarks: list
  relocalization_score: float
  failure_reason: optional str
```

The planner should not only know where the drone probably is. It must also know whether the estimate is trustworthy enough for the requested action.

---

## 8. Localization status model

ArgusNet should use explicit localization states.

| State | Meaning | Planner behavior |
|------|---------|------------------|
| unlocalized | No reliable pose estimate | Do not execute map-relative navigation |
| initializing | Collecting evidence and narrowing pose candidates | Permit cautious observation behavior only |
| localized | Pose confidence is high enough for normal planning | Permit mission planning |
| degraded | Pose exists but uncertainty is growing | Slow down, increase clearance, seek landmarks |
| lost | Pose estimate failed or diverged | Stop, hover/hold, relocalize, or return using fallback |

Localization state should directly affect safety and planning.

---

## 9. Relocalization against prior world model

When a prior map exists, the system should compare live observations against stored world features.

Possible matching layers:

- visual landmarks;
- keyframe descriptors;
- terrain height profile;
- obstacle outlines;
- structure edges;
- semantic objects;
- inspection markers;
- coverage map regions.

Relocalization should produce candidate poses, score them, and converge only when confidence is high enough.

```text
Live observation
        ↓
Feature extraction
        ↓
Candidate retrieval from spatial memory
        ↓
Pose hypothesis generation
        ↓
Geometric consistency check
        ↓
Pose estimate + uncertainty
```

This should connect to the future indexing subsystem. The index stores prior keyframes, landmarks, map tiles, and inspection points. Localization retrieves from that memory to determine where the drone is.

---

## 10. Localization for battery swap and replacement drones

Battery swap continuity is a primary reason to add localization.

A typical flow:

```text
Mission maps part of area
        ↓
Drone lands or battery runs low
        ↓
World model is saved
        ↓
Same or replacement drone powers on
        ↓
Drone receives prior world model and geofence
        ↓
Drone performs relocalization behavior
        ↓
Pose confidence passes threshold
        ↓
Mission resumes from previous map state
```

Until the new drone localizes, it should not be sent to a precise inspection point or through a tight corridor. It may perform low-risk observation maneuvers to improve localization.

---

## 11. Localization for future inspection routing

A later inspection system will depend on localization.

Inspection targets should be stored in the world model as map-relative objects:

```text
InspectionTarget
  target_id: str
  world_position_m: [x, y, z]
  required_view_angle: optional
  required_standoff_m: optional
  required_resolution: optional
  last_inspected_at: timestamp
  confidence: float
```

To fly to an inspection target, the system must first know its own pose relative to that target.

```text
Current live observations
        ↓
LocalizationState
        ↓
Map-relative current pose
        ↓
Planner route to inspection target
        ↓
Trajectory and safety validation
```

If localization confidence is too low, the system should first plan a relocalization route, not an inspection route.

---

## 12. Planning behavior under localization uncertainty

Localization uncertainty must affect guidance.

| Localization condition | Planning response |
|------------------------|------------------|
| High confidence | Normal planning allowed |
| Moderate uncertainty | Increase clearance and reduce speed |
| Uncertain heading | Prefer observation or rotation before translation |
| Large position uncertainty | Avoid tight corridors and nearby obstacles |
| Lost localization | Stop/hold, climb if safe, relocalize, or return using fallback |
| Conflicting map match | Continue gathering observations before committing |

The planner should never treat a low-confidence pose as exact.

---

## 13. Query interface

Planning and mapping should access localization through a formal interface.

```text
LocalizationQuery
  current_pose(platform_id) -> pose estimate
  current_covariance(platform_id) -> uncertainty
  localization_status(platform_id) -> status
  confidence(platform_id) -> float
  is_localized(platform_id, threshold) -> bool
  candidate_regions(platform_id) -> list
  matched_landmarks(platform_id) -> list
  needs_relocalization(platform_id) -> bool
```

The system should also expose a relocalization command:

```text
RelocalizationRequest
  platform_id: str
  prior_region: optional region/geofence
  allowed_motion: hover | rotate | local_search | return_home
  confidence_threshold: float
```

---

## 14. Interaction with geofence and belief world

Localization should be constrained by the active mission boundary.

The geofence limits possible pose hypotheses. If the system knows the drone is inside a mission area, candidate poses outside that area should be rejected or heavily penalized.

The belief world also helps localization:

- terrain height can constrain altitude;
- obstacle layout can support shape matching;
- coverage history tells where recognizable features exist;
- prior inspection targets provide known landmarks;
- uncertainty layers indicate where localization is likely weak.

Mapping improves localization, and localization improves mapping.

---

## 15. Safety rules

Minimum safety rules:

1. If localization status is `unlocalized`, the drone may not execute map-relative navigation.
2. If localization status is `initializing`, only cautious observation motions are allowed.
3. If localization becomes `degraded`, the planner must increase safety margins or seek relocalization.
4. If localization becomes `lost`, the drone should hold, climb if safe, relocalize, or return using the best available fallback.
5. Inspection routing requires localization confidence above a configured threshold.
6. Battery swap continuation requires successful relocalization before mission resumption.
7. Ground truth pose must only be used for simulation scoring, never for planner input in physical-mode tests.

---

## 16. Implementation priorities

### Phase 1 — Localization state model

- Add `LocalizationState` as a first-class runtime object.
- Track pose estimate, uncertainty, confidence, and status.
- Serialize localization state into replay and viewer state.

### Phase 2 — Startup mode handling

- Support known launch point.
- Support weak GNSS prior.
- Support known area / unknown exact position.
- Support battery-swap relocalization mode.

### Phase 3 — Map-relative relocalization

- Store keyframes, landmarks, or compact descriptors in the world model.
- Retrieve candidate regions from spatial memory.
- Score candidate pose hypotheses.
- Produce pose estimate with uncertainty.

### Phase 4 — Planner integration

- Make planner consume `LocalizationState`.
- Prevent map-relative routing when localization is insufficient.
- Add relocalization behavior as a planner mode.
- Add uncertainty-aware speed and clearance adjustment.

### Phase 5 — Inspection preparation

- Define map-relative inspection targets.
- Require localization confidence before routing to inspection points.
- Add evaluation metrics for revisit accuracy.

---

## 17. Evaluation metrics

Simulation should measure localization separately from mapping and tracking.

Useful metrics:

- position error over time;
- heading error over time;
- time to localize after startup;
- time to relocalize after simulated battery swap;
- false localization confidence;
- localization lost events;
- percentage of mission spent localized;
- revisit error to known inspection targets;
- map alignment error after relocalization;
- planning violations caused by pose uncertainty.

These metrics should be computed using ground truth only after the planner has acted, not as planner input.

---

## 18. Success criteria

Localization is successful when ArgusNet can:

- start with a known launch point and maintain pose through a mission;
- start in a previously mapped area with uncertain position and recover its map-relative pose;
- resume a mission after battery swap or replacement drone launch;
- distinguish localized, degraded, and lost states;
- prevent inspection routing until pose confidence is sufficient;
- use localization uncertainty to adjust planning behavior;
- evaluate localization accuracy in simulation without allowing the planner to use ground truth.

At that point, ArgusNet can reuse world models across missions and support future inspection routing to known locations.
