# MISSION_MODEL.md — Map, Localize, Inspect Mission Model

This document defines ArgusNet mission structure for mapping, localization, inspection, revisit,
coordination, and safe return. Multi-drone behavior is a coordination tool for map and inspection
work.

## Mission Inputs

```text
MissionSpec
  seed: int
  terrain_preset: str
  weather_preset: str
  map_preset: str
  platform_preset: str
  drone_count: int
  difficulty: float
  mission_type: str
  tags: list[str]
  timing: MissionTiming
  constraints: MissionConstraints
```

Supported mission types should use this vocabulary:

- `mapping`: build or extend coverage of a bounded area.
- `localization`: establish or recover map-relative platform pose.
- `inspection`: visit and document map-relative POIs.
- `revisit`: return to previously stored POIs or regions.
- `map_localize_inspect`: perform the full staged workflow.

## Mission Constraints

```text
MissionConstraints
  geofence: polygon or radius
  terrain_clearance_m: float
  max_altitude_m: float
  battery_reserve_fraction: float
  comms_range_m: float
  required_localization_confidence: float
  inspection_quality_threshold: float
  exclusion_zones: list[MissionZone]
```

`MissionZone` lives in `argusnet.core.types`. Use zone labels around civil mapping semantics:

- `mapping_area`;
- `inspection_area`;
- `exclusion`;
- `return_home`;
- `revisit_area`.

Mission labels should stay aligned with mapping, localization, inspection, revisit, exclusion, and
safe return.

## Mission Outputs

```text
GeneratedMission
  scenario_def: ScenarioDefinition
  spec: MissionSpec
  launch_points: list[LaunchPoint]
  map_regions: list[MapRegion]
  inspection_pois: list[InspectionPOI]
  timing: MissionTiming
  tags: list[str]
  validity_report: ValidityReport
```

Current replay/runtime state for the full workflow is defined in `argusnet.core.types`:

- `MappingState`;
- `LocalizationState`;
- `LocalizationEstimate`;
- `InspectionPOI`;
- `POIStatus`;
- `InspectionEvent`;
- `DeconflictionEvent`;
- `EgressDroneProgress`;
- `ScanMissionState`.

The state is populated in `src/argusnet/simulation/sim.py`.

## Scan-Map-Inspect Runtime

The current `scan_map_inspect` runtime is the implemented bridge for the product direction. Its
phase sequence is:

```text
scanning -> localizing -> inspecting -> egress -> complete
```

Transitions:

```text
scanning -> localizing
  when scan_coverage_fraction >= scan_coverage_threshold
  and enclosed coverage gaps are below FrontierConfig.gap_fill_min_fraction
  and localization still needs convergence

scanning -> inspecting
  when scan coverage is sufficient and all active localization estimates already pass confidence

localizing -> inspecting
  when all localization estimates pass confidence
  or timeout handling allows progress

inspecting -> egress
  when all POIs reach status "complete"

egress -> complete
  when active mobile drones return within the home threshold
```

Localization timeout uses `LocalizationConfig.localization_timeout_steps`. When timeout is reached,
the mission records `ScanMissionState.localization_timed_out = True` as a team-level flag.

## Mission Validity

A generated map/localize/inspect mission is valid when:

- the initial drone positions are inside the mission boundary;
- planned mapping and inspection areas are inside the geofence;
- exclusion zones are respected;
- terrain clearance can be maintained;
- the required POIs are reachable with battery reserve;
- the localization requirement is appropriate for the inspection task;
- replay and evaluation metadata can explain the chosen phase transitions.

## Difficulty Scaling

Difficulty should scale civil autonomy constraints:

| Parameter | Easier | Harder |
|-----------|--------|--------|
| Map size | smaller bounded area | larger bounded area |
| Coverage threshold | lower | higher |
| Terrain roughness | smoother | more varied |
| Obstacle density | lower | higher |
| Localization prior | known launch or strong prior | weak prior / relocalization required |
| POI count | fewer | more |
| Required dwell/quality | lower | higher |
| Battery margin | generous | tighter |
| Weather/sensing degradation | lower | higher |

Difficulty should only adjust mapping, localization, inspection, coordination, and safety
constraints.

## Tags

Canonical tag prefixes:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `type:` | Mission type | `type:mapping`, `type:inspection`, `type:map_localize_inspect` |
| `terrain:` | Terrain preset | `terrain:alpine`, `terrain:urban` |
| `weather:` | Weather preset | `weather:clear`, `weather:fog` |
| `diff:` | Difficulty band | `diff:easy`, `diff:medium`, `diff:hard` |
| `size:` | Map size | `size:small`, `size:regional` |
| `workflow:` | Workflow emphasis | `workflow:coverage`, `workflow:revisit`, `workflow:multi_drone` |
| `eval:` | Evaluation suite membership | `eval:regression`, `eval:benchmark` |
