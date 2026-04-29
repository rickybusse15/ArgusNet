# INSPECTION.md — Inspection POIs, Evidence, and Local Reconstruction Contract

This document defines ArgusNet inspection: routing to meaningful map-relative POIs, collecting
enough evidence, optionally reconstructing the local area, and storing results for review,
comparison, and future revisit.

```text
Mapping      -> builds the belief world
Localization -> places each drone inside that world
Inspection   -> sends drones to useful map-relative POIs
```

Inspection is not only a photo-taking step. It should support multi-view capture, local 3D
reconstruction, quality scoring, repeat inspection, and change detection.

## Current Implementation

Current `scan_map_inspect` inspection is a POI-based runtime bridge:

- `argusnet.core.types.InspectionPOI` defines current map-relative POIs.
- `argusnet.core.types.POIStatus` records `pending`, `active`, and `complete` lifecycle state.
- `src/argusnet/planning/poi.py` owns `POIManager`, energy-aware assignment, handoff, team
  assignment, dwell accumulation, and coverage-based rescoring.
- `argusnet.core.types.InspectionEvent` records replay-visible zone coverage and violation events.
- `src/argusnet/planning/deconfliction.py` emits `DeconflictionEvent` records used by replay and
  evaluation.
- `src/argusnet/simulation/sim.py` transitions inspection completion into `egress` before
  `complete`.

Evidence sets, local reconstructions, change detection, and indexing-backed repeat inspection are
roadmap contracts. They should build on the current POI runtime.

## Inspection POI Model

An inspection POI is a persistent map-relative object. It may be created by an operator, imported
asset, prior mission, or automatic map detection.

```text
InspectionSite
  site_id: str
  site_type: roof | wall | tower | pipe | vent | structure | terrain_region | custom
  world_position_m: [x, y, z]
  region_geometry: optional polygon / bounding box / mesh reference
  priority: int
  required_sensor: rgb | thermal | depth | lidar | multispectral
  required_standoff_m: optional float
  required_view_angle: optional
  required_resolution_cm_per_px: optional float
  required_overlap_fraction: optional float
  reconstruction_required: bool
  required_view_count: optional int
  confidence: float
  created_from: operator | map_detection | prior_mission | imported_asset
  last_inspected_at: optional timestamp
  status: pending | planned | inspected | blocked | failed | needs_revisit
```

The site is stable across missions. Individual inspection requests may ask for different evidence
quality or reconstruction requirements.

## Inspection Request Model

```text
InspectionRequest
  request_id: str
  site_ids: list[str]
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
    metadata
  success_criteria:
    coverage
    clarity
    resolution
    view_angle
    localization_confidence
    reconstruction_quality
```

This separation allows repeat inspections, scheduled revisits, and changing evidence requirements
over time.

## Planning Rules

The inspection planner chooses:

- which POI to inspect next;
- which sensor to use;
- which viewpoint or capture plan to use;
- how many views are required;
- whether localization is sufficient;
- whether extra mapping is needed first;
- how to route safely to the local area.

Rules:

1. If the POI is not mapped, map the area first.
2. If the drone is not localized, relocalize first.
3. If the POI is outside the geofence, reject the task or require operator authorization.
4. If the required viewpoint is unsafe, choose an alternate viewpoint or mark the POI blocked.
5. If reconstruction is required, collect sufficient overlapping views before declaring success.

## Viewpoint And Capture Plan

```text
InspectionViewpoint
  viewpoint_id: str
  site_id: str
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
- lacks line of sight to the POI;
- has insufficient expected resolution;
- lacks a safe return path;
- violates altitude, speed, or standoff constraints.

Multi-view capture should balance:

- surface coverage;
- angle diversity;
- image overlap;
- reconstruction quality;
- route cost;
- safety;
- localization confidence.

## Evidence And Reconstruction

```text
InspectionEvidence
  evidence_id: str
  site_id: str
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
  localization_confidence: float
```

```text
InspectionReconstruction
  reconstruction_id: str
  site_id: str
  source_evidence_ids: list[str]
  reconstruction_type: photogrammetry | depth_fusion | lidar_scan | hybrid
  local_mesh_uri: optional str
  point_cloud_uri: optional str
  confidence_map_uri: optional str
  coverage_fraction: float
  reconstruction_error_estimate: optional float
  resolution_estimate: float
  aligned_to_world_model: bool
  alignment_confidence: float
```

Evidence should be indexed into spatial memory so it can be retrieved later by site, location, time,
viewpoint, or mission.

## Result And Revisit Model

```text
InspectionResult
  result_id: str
  site_id: str
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

Repeat inspection should prefer the same site-relative viewpoint, standoff distance, camera angle,
sensor type, and reconstruction coverage when safe. It may add new views when localization was weak,
coverage had gaps, occlusion changed, or the site geometry changed.

## Safety And Authority Rules

1. No inspection route if localization confidence is below threshold.
2. No POI outside the geofence unless the operator extends the mission boundary.
3. No inspection if the required viewpoint violates safety margins.
4. If the POI is unmapped or occluded, map first.
5. If reconstruction is required, enough overlapping views must be captured before declaring
   success.
6. Planning must use mapping/localization state, not hidden simulation truth.
7. The safety engine may reject any route, viewpoint, or capture plan.
8. Failed or blocked inspections must preserve reason codes.

## Viewer And Evaluation

The viewer should show:

- inspection POIs and status;
- planned viewpoint or multi-view path;
- achieved viewpoint;
- evidence markers and thumbnails;
- local reconstructed mesh or point cloud;
- previous/current comparison;
- changed regions and missing coverage;
- failed or blocked reason codes;
- revisit timeline.

Evaluation should measure:

- POI reached / not reached;
- inspection completion rate;
- coverage fraction;
- image/view quality;
- useful image count;
- viewpoint diversity;
- localization confidence at capture;
- revisit accuracy;
- reconstruction coverage and alignment error;
- failed due to localization, occlusion, or safety;
- energy cost per inspected POI.

## Implementation Phases

1. Extend the current `InspectionPOI` / `POIStatus` runtime bridge into persistent inspection site
   records.
2. Add request/result records and preserve blocked/failed reason codes.
3. Generate candidate viewpoints and multi-view capture plans.
4. Store evidence, evidence sets, and optional local reconstructions.
5. Route drones to POIs only after localization and safety checks pass.
6. Add repeat-inspection and change-detection history.
7. Render POIs, capture plans, evidence, and reconstructions in the viewer.
