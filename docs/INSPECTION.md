# INSPECTION.md — Inspection Targets, Multi-View Evidence, and Local Reconstruction Contract

This document defines the inspection step for ArgusNet: how the system should route to meaningful map-relative targets, capture sufficient evidence, optionally reconstruct the local inspection area, and store results for review, comparison, and future revisit.

Inspection builds on the previous architecture layers:

```text
Mapping       → builds the belief world
Localization  → places the platform inside that world
Inspection    → sends the platform to meaningful places in that world
```

The inspection subsystem is not only a photo-taking step. It should support multi-image capture, local 3D reconstruction, quality scoring, repeat inspection, and change detection.

---

## 1. Purpose

The inspection subsystem answers:

- What needs to be inspected?
- Where is it in the world model?
- What view angle, distance, or sensor quality is required?
- How many images or viewpoints are needed?
- Does the inspection require a local reconstruction?
- Has this area already been inspected?
- Is the drone localized well enough to inspect it?
- Was the inspection successful?
- Does the target need to be revisited?

Inspection turns mapping and localization into useful mission action.

---

## 2. Why inspection matters to ArgusNet

ArgusNet is intended for autonomous aerial mapping, inspection, localization, and spatial memory. Mapping creates a model of the world. Localization places the platform inside that model. Inspection uses both to perform a specific, useful task.

Important use cases include:

- roof inspection;
- infrastructure inspection;
- construction progress capture;
- repeated site monitoring;
- damage or change detection;
- environmental survey;
- revisit of known points of interest;
- local reconstruction of high-value target regions.

Inspection should not be limited to a single image. One image may miss damage, deformation, occluded surfaces, or the geometric context needed for measurement. Multiple viewpoints and local reconstructions improve confidence, repeatability, and comparison across time.

---

## 3. Relationship to mapping and localization

Inspection depends on both the belief world and localization state.

```text
BeliefWorldModel
        +
LocalizationState
        ↓
Inspection Planner
        ↓
Viewpoint / capture-plan selection
        ↓
Trajectory + safety validation
        ↓
Evidence capture
        ↓
Optional local reconstruction
        ↓
Inspection result stored in world model and index
```

Rules:

- If the target is not mapped, map the area first.
- If the platform is not localized, relocalize first.
- If the target is outside the geofence, reject the task or require operator authorization.
- If the required viewpoint is unsafe, choose an alternate viewpoint or mark the target blocked.
- If reconstruction is required, collect sufficient overlapping views before declaring success.

---

## 4. Inspection target model

An inspection target is a persistent map-relative object. It may be created by an operator, imported asset, prior mission, or automatic map detection.

```text
InspectionTarget
  target_id: str
  target_type: roof | wall | tower | pipe | vent | structure | terrain_region | custom
  world_position_m: [x, y, z]
  region_geometry: optional polygon / bounding box / mesh reference
  priority: int
  required_sensor: rgb | thermal | depth | lidar | multispectral
  required_standoff_m: optional float
  required_view_angle: optional
  required_resolution_cm_per_px: optional float
  required_overlap_fraction: optional float
  required_lighting: optional
  reconstruction_required: bool
  required_view_count: optional int
  required_view_angles: optional list
  required_surface_coverage_fraction: optional float
  required_reconstruction_resolution: optional float
  prior_reconstruction_id: optional str
  confidence: float
  created_from: operator | map_detection | prior_mission | imported_asset
  last_inspected_at: optional timestamp
  status: pending | planned | inspected | blocked | failed | needs_revisit
```

The target itself is stable across missions. Individual inspection requests may ask for different evidence quality or reconstruction requirements.

---

## 5. Inspection request model

An inspection request is a command to inspect one or more targets. It is separate from the stored target because the same target may be inspected many times under different mission requirements.

```text
InspectionRequest
  request_id: str
  target_ids: list[str]
  mission_id: str
  priority: int
  deadline: optional timestamp
  required_outputs:
    images
    video
    thermal_frames
    depth_scan
    point_cloud
    local_reconstruction
    notes / metadata
  success_criteria:
    coverage
    clarity
    resolution
    view_angle
    localization_confidence
    reconstruction_quality
```

This separation allows repeat inspections, scheduled revisits, and changing evidence requirements over time.

---

## 6. Inspection planning

The inspection planner chooses:

- which target to inspect next;
- which sensor to use;
- which viewpoint or capture plan to use;
- how many views are required;
- whether localization is sufficient;
- whether extra mapping is needed first;
- how to route safely to the target area.

Target selection should consider:

- target priority;
- distance and energy cost;
- localization confidence;
- route safety;
- expected inspection quality;
- expected reconstruction quality;
- lighting and weather;
- whether another drone is closer;
- whether the target is overdue for revisit;
- whether the target has prior reconstruction gaps.

---

## 7. Viewpoint generation

For each target, ArgusNet should generate candidate viewpoints.

```text
InspectionViewpoint
  viewpoint_id: str
  target_id: str
  position_m: [x, y, z]
  camera_orientation: yaw / pitch / roll or quaternion
  standoff_m: float
  expected_resolution: float
  expected_occlusion: float
  expected_coverage: float
  expected_overlap_with_other_views: float
  safety_margin_m: float
  localization_requirement: float
  score: float
```

A viewpoint should be rejected if it:

- is outside the geofence;
- violates terrain clearance;
- intersects an obstacle;
- is in a poor localization region;
- lacks line of sight to the target;
- has insufficient expected resolution;
- lacks a safe return path;
- violates altitude, speed, or standoff constraints.

---

## 8. Multi-view capture plan

Many inspections require multiple images or scans rather than one viewpoint.

```text
MultiViewCapturePlan
  capture_plan_id: str
  target_id: str
  viewpoints: list[InspectionViewpoint]
  required_min_images: int
  required_angle_diversity: float
  required_overlap_fraction: float
  required_surface_coverage_fraction: float
  reconstruction_required: bool
  completion_rule: all_views | coverage_threshold | reconstruction_quality
```

The plan should ensure that captured views are not merely redundant. It should balance:

- surface coverage;
- angle diversity;
- image overlap;
- reconstruction quality;
- route cost;
- safety;
- localization confidence.

Multi-view capture is important because local inspection quality depends on seeing enough of the target from enough angles to support reliable review and reconstruction.

---

## 9. Evidence capture

Inspection must produce stored evidence, not just a visited-target flag.

Individual evidence item:

```text
InspectionEvidence
  evidence_id: str
  target_id: str
  timestamp_s: float
  platform_id: str
  sensor_type: str
  platform_pose: pose + covariance
  camera_pose: pose + covariance
  file_uri: str
  quality_score: float
  coverage_fraction: float
  resolution_estimate: float
  view_angle: float
  view_angle_error: float
  overlap_with_other_views: float
  localization_confidence: float
```

Evidence set:

```text
InspectionEvidenceSet
  evidence_set_id: str
  target_id: str
  evidence_ids: list[str]
  reconstruction_id: optional str
  capture_plan_id: str
  coverage_fraction: float
  view_diversity_score: float
  reconstruction_quality_score: optional float
```

Evidence should be indexed into spatial memory so it can be retrieved later by target, location, time, viewpoint, or mission.

---

## 10. Local inspection reconstruction

Some inspection tasks should produce a local reconstruction of the target area.

```text
InspectionReconstruction
  reconstruction_id: str
  target_id: str
  source_evidence_ids: list[str]
  reconstruction_type: photogrammetry | depth_fusion | lidar_scan | hybrid
  local_mesh_uri: optional str
  point_cloud_uri: optional str
  texture_uri: optional str
  confidence_map_uri: optional str
  coverage_fraction: float
  reconstruction_error_estimate: optional float
  resolution_estimate: float
  created_at: timestamp
  aligned_to_world_model: bool
  alignment_confidence: float
```

A local reconstruction is useful for:

- measuring shape or deformation;
- checking surfaces not clear in a single view;
- comparing the same area across missions;
- identifying missing coverage;
- building higher-quality world-model detail around important targets.

The reconstruction should be aligned to the ArgusNet world model and should preserve its own confidence and error estimates.

---

## 11. Inspection result model

An inspection result records whether the request succeeded and why.

```text
InspectionResult
  result_id: str
  target_id: str
  request_id: str
  status: success | partial | failed | blocked
  evidence_set_id: optional str
  evidence_ids: list[str]
  reconstruction_id: optional str
  coverage_score: float
  quality_score: float
  reconstruction_score: optional float
  reason: optional str
  needs_revisit: bool
  next_revisit_after: optional timestamp
```

Partial and failed inspections should preserve reason codes. This prevents silent failure and helps future planners choose better routes or viewpoints.

---

## 12. Revisit and repeat inspection

ArgusNet should support repeated inspection over time.

Use cases:

- compare roof condition over months;
- monitor construction progress;
- detect vegetation growth;
- check infrastructure degradation;
- compare storm or fire damage before and after an event.

Repeat inspection should prefer:

- the same target-relative viewpoint when safe;
- the same standoff distance;
- the same camera angle;
- the same sensor type;
- the same reconstruction coverage;
- the same target-relative frame.

It may add new views if:

- the previous reconstruction had gaps;
- occlusion changed;
- damage is suspected;
- localization confidence was low;
- operator requests higher detail;
- the target geometry changed.

---

## 13. Change detection across evidence and reconstructions

Inspection should support comparison between previous and current evidence.

```text
InspectionChangeRecord
  change_record_id: str
  target_id: str
  previous_evidence_set_id: optional str
  current_evidence_set_id: optional str
  previous_reconstruction_id: optional str
  current_reconstruction_id: optional str
  detected_changes: list
  change_confidence: float
  severity_score: optional float
  requires_human_review: bool
```

Change detection may compare:

- old images vs new images;
- old local reconstruction vs new local reconstruction;
- surface deformation;
- missing or damaged objects;
- cracks, corrosion, leaks, vegetation growth;
- construction progress.

The system should distinguish detected change from uncertain change. Low-confidence differences should be marked for review instead of treated as confirmed findings.

---

## 14. Safety and authority rules

Minimum rules:

1. No inspection route if localization confidence is below threshold.
2. No target outside the geofence unless the operator extends the mission boundary.
3. No inspection if the required viewpoint violates safety margins.
4. If the target is unmapped or occluded, map first.
5. If reconstruction is required, enough overlapping views must be captured before declaring success.
6. The planner must use `BeliefWorldModel`, not ground truth.
7. The safety engine may reject any route, viewpoint, or capture plan.
8. Failed or blocked inspections must preserve reason codes.

---

## 15. Query and command interfaces

Inspection query interface:

```text
InspectionQuery
  pending_targets()
  target_by_id(target_id)
  target_status(target_id)
  evidence_for_target(target_id)
  last_inspection(target_id)
  required_viewpoints(target_id)
  capture_plan_for_target(target_id)
  reconstruction_for_target(target_id)
  change_records_for_target(target_id)
  is_inspectable(target_id, localization_state, world_belief)
```

Inspection command interface:

```text
InspectionCommand
  create_target(...)
  request_inspection(...)
  create_capture_plan(...)
  mark_blocked(...)
  schedule_revisit(...)
  attach_evidence(...)
  attach_reconstruction(...)
  record_change(...)
```

---

## 16. Integration with existing ArgusNet subsystems

| Subsystem | Inspection relationship |
|-----------|-------------------------|
| Mapping | Provides target geometry, coverage, and local world belief |
| Localization | Confirms platform pose relative to target |
| Planning | Selects target, viewpoint, and route |
| Trajectory | Executes approach, hold, orbit, or multi-view path |
| Safety | Validates route, viewpoint, and capture plan |
| Evaluation | Scores inspection success and reconstruction quality |
| Indexing | Stores evidence, reconstructions, targets, and history |
| Viewer | Shows targets, routes, evidence, reconstruction, and status |

Inspection is the first major mission-use layer built directly on top of mapping and localization.

---

## 17. Viewer and UI requirements

The viewer should show:

- inspection targets;
- target status;
- target priority;
- planned viewpoint or multi-view capture path;
- achieved viewpoint;
- target-relative view cones;
- evidence markers;
- image thumbnails by viewpoint;
- inspection evidence sets;
- local reconstructed mesh or point cloud;
- reconstruction confidence map;
- previous vs current reconstruction comparison;
- changed regions highlighted;
- missing coverage regions;
- failed or blocked reason codes;
- revisit timeline.

The viewer should make it clear whether a target was truly inspected, partially inspected, blocked, or only passed nearby.

---

## 18. Evaluation metrics

Simulation and replay evaluation should measure:

- target reached / not reached;
- inspection success rate;
- coverage fraction;
- image/view quality;
- number of useful images captured;
- viewpoint diversity score;
- image overlap quality;
- localization confidence at capture;
- revisit accuracy;
- viewpoint repeatability;
- reconstruction coverage fraction;
- reconstruction resolution estimate;
- reconstruction alignment error;
- reconstruction completeness;
- repeat reconstruction difference;
- false change detections;
- missed change detections;
- failed due to localization;
- failed due to occlusion;
- failed due to safety;
- energy cost per inspected target.

Ground truth may be used to score inspection performance in simulation, but it must not be used by the planner during physical-mode tests.

---

## 19. Implementation phases

### Phase 1 — Target model

- Add `InspectionTarget`.
- Add pending, planned, inspected, blocked, failed, and needs-revisit states.
- Store targets in the world model.

### Phase 2 — Request and result model

- Add `InspectionRequest`.
- Add `InspectionResult`.
- Store evidence metadata.
- Preserve failed and blocked reason codes.

### Phase 3 — Viewpoint and capture-plan planner

- Generate candidate viewpoints.
- Generate multi-view capture plans.
- Check LOS, safety, geofence, localization, and expected resolution.
- Select best viewpoint or capture path.

### Phase 4 — Evidence and reconstruction support

- Store individual evidence items.
- Group evidence into evidence sets.
- Add optional local reconstruction records.
- Track coverage, overlap, and reconstruction quality.

### Phase 5 — Planner integration

- Route the drone to the target or capture path.
- Require localization confidence.
- Support blocked, retry, and map-first states.
- Support multi-view capture trajectories.

### Phase 6 — Repeat inspection and change detection

- Store previous viewpoints and reconstructions.
- Compare current and previous evidence.
- Add change records.
- Support revisit scheduling.

### Phase 7 — Viewer and evaluation

- Render targets, capture plans, evidence sets, and reconstructions.
- Add inspection metrics.
- Add replay support for inspection results and change records.

---

## 20. Success criteria

Inspection is successful when ArgusNet can:

- store map-relative inspection targets;
- localize before routing to them;
- choose safe viewpoints;
- generate multi-view capture plans when needed;
- collect sufficient overlapping evidence;
- create or attach local reconstructions;
- score quality, coverage, and reconstruction completeness;
- mark failed or blocked cases with reason codes;
- revisit the same target later;
- compare evidence or reconstructions over time;
- highlight likely changes for human review.

At that point, ArgusNet can support persistent inspection workflows rather than only general mapping or tracking.
