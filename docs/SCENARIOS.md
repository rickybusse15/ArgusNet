# SCENARIOS.md — Map, Localize, Inspect Scenario and Evaluation Model

This document defines benchmark scenario families for ArgusNet’s current purpose: mapping,
localization, inspection, revisit, multi-drone coordination, safety, and replay evaluation.

## Metric Families

### Mapping Metrics

```text
Metric: map_coverage_fraction
  Unit: fraction [0.0, 1.0]
  Direction: higher is better
  Definition: Fraction of required map cells observed at least once.
  Source: MappingState.coverage_fraction
```

```text
Metric: mean_revisits
  Unit: count
  Direction: scenario-dependent
  Definition: Mean number of observations per covered map cell.
  Source: MappingState.mean_revisits
```

```text
Metric: enclosed_gap_fraction
  Unit: fraction [0.0, 1.0]
  Direction: lower is better
  Definition: Fraction of map cells that remain uncovered inside the covered perimeter.
  Source: FrontierPlanner.find_gap_cells() and CoverageMap dimensions
```

### Localization Metrics

```text
Metric: localization_confidence_mean
  Unit: fraction [0.0, 1.0]
  Direction: higher is better
  Definition: Mean confidence across active localization estimates.
  Source: LocalizationState.mean_observation_confidence
```

```text
Metric: localization_position_std_mean_m
  Unit: meters
  Direction: lower is better
  Definition: Mean 1-sigma position uncertainty across active localization estimates.
  Source: LocalizationState.mean_position_std_m
```

```text
Metric: localization_timeout_count
  Unit: count
  Direction: lower is better
  Definition: Number of frames or runs where localization advanced through timeout handling.
  Source: ScanMissionState.localization_timed_out
```

### Inspection Metrics

```text
Metric: inspection_completion_fraction
  Unit: fraction [0.0, 1.0]
  Direction: higher is better
  Definition: Completed POIs divided by total POIs.
  Source: ScanMissionState.completed_poi_count / total_poi_count
```

```text
Metric: inspection_dwell_completion_s
  Unit: seconds
  Direction: lower is better for equivalent quality
  Definition: Time from inspection phase entry to all POIs complete.
  Source: POIStatus.completion_time_s and ScanMissionState.phase_started_at_s
```

```text
Metric: inspection_event_count
  Unit: count
  Direction: scenario-dependent
  Definition: Number of replay-visible inspection events.
  Source: PlatformFrame.inspection_events
```

### Coordination, Safety, And Reliability Metrics

```text
Metric: deconfliction_event_count
  Unit: count
  Direction: lower is better for baseline scenarios
  Definition: Number of drone-to-drone or corridor deconfliction events.
  Source: PlatformFrame.deconfliction_events
```

```text
Metric: comms_dropout_count
  Unit: count
  Direction: lower is better
  Definition: Number of coordination link dropout events.
  Source: replay mission/evaluation metadata
```

```text
Metric: return_home_completion_fraction
  Unit: fraction [0.0, 1.0]
  Direction: higher is better
  Definition: Fraction of active drones that complete egress.
  Source: ScanMissionState.egress_progress and phase transition to complete
```

```text
Metric: energy_reserve_min
  Unit: fraction [0.0, 1.0]
  Direction: higher is better
  Definition: Minimum battery reserve across active drones at mission end.
  Source: NodeState.battery_fraction
```

## Benchmark Scenario Families

Each family is a named collection of deterministic runs. Seeds should be fixed for repeatability.

### Family A: `mapping_coverage`

Purpose: verify map coverage, gap detection, terrain handling, and replay output.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|------|------------|---------|-------------|------------|--------------|
| `mapping_small` | small | default | 1 | 0.1 | mapping |
| `mapping_medium_multi` | medium | default | 3 | 0.3 | mapping |
| `mapping_alpine` | medium | alpine | 3 | 0.5 | mapping |

Primary pass criteria:

- `map_coverage_fraction >= scan_coverage_threshold`
- `enclosed_gap_fraction <= gap_fill_min_fraction`
- no unsafe route acceptance

### Family B: `localization_recovery`

Purpose: exercise startup localization, weak-prior localization, and timeout handling.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|------|------------|---------|-------------|------------|--------------|
| `localize_known_launch` | small | default | 1 | 0.1 | localization |
| `localize_weak_prior` | medium | default | 2 | 0.4 | localization |
| `localize_after_revisit` | medium | alpine | 2 | 0.6 | localization |

Primary pass criteria:

- localization confidence reaches configured threshold;
- position uncertainty trends downward;
- timeout use is reported when convergence does not happen naturally.

### Family C: `inspection_workflow`

Purpose: verify map-relative POI assignment, dwell completion, egress, and inspection events.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|------|------------|---------|-------------|------------|--------------|
| `inspect_single_poi` | small | default | 1 | 0.2 | inspection |
| `inspect_multi_poi` | medium | default | 3 | 0.4 | inspection |
| `inspect_occluded_area` | medium | urban_flat | 3 | 0.6 | inspection |

Primary pass criteria:

- all required POIs complete;
- inspection does not start before localization confidence is sufficient;
- egress reaches complete.

### Family D: `coordination_stress`

Purpose: verify multi-drone coverage distribution, POI workload sharing, deconfliction, and
communication assumptions.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|------|------------|---------|-------------|------------|--------------|
| `coordination_two_drone` | medium | default | 2 | 0.3 | map_localize_inspect |
| `coordination_dense_pois` | medium | urban_flat | 4 | 0.6 | map_localize_inspect |
| `coordination_large_area` | regional | alpine | 5 | 0.7 | map_localize_inspect |

Primary pass criteria:

- no unresolved separation violations;
- POI assignments remain explainable;
- completed replay includes mapping, localization, inspection, and egress state.

### Family E: `revisit_and_change`

Purpose: prepare future repeat-inspection and change-detection work.

| Name | map_preset | terrain | drone_count | difficulty | mission_type |
|------|------------|---------|-------------|------------|--------------|
| `revisit_known_poi` | small | default | 1 | 0.2 | revisit |
| `revisit_multi_session` | medium | default | 2 | 0.5 | revisit |
| `revisit_changed_area` | medium | urban_flat | 3 | 0.7 | revisit |

Primary pass criteria:

- prior POIs or map regions are retrieved from the index;
- localization confidence is established before revisit routing;
- evidence/change records are persisted when that roadmap capability is enabled.

## Replay Metadata

Mission metadata should describe map, localization, inspection, revisit, and coordination work:

```json
{
  "mission_spec": {
    "seed": 7,
    "terrain_preset": "alpine",
    "weather_preset": "clear",
    "map_preset": "medium",
    "platform_preset": "baseline",
    "drone_count": 3,
    "difficulty": 0.4,
    "mission_type": "map_localize_inspect",
    "tags": ["type:map_localize_inspect", "workflow:multi_drone"]
  }
}
```

## Acceptance Policy

Scenario results should be accepted only when:

- mapping, localization, inspection, and egress states are visible in replay when relevant;
- safety/deconfliction events are logged rather than hidden;
- deterministic seeds produce repeatable summary metrics;
- benchmark reports include command, seed, environment, and commit SHA;
- scenario objectives stay within mapping, localization, inspection, revisit, coordination, and
  safety.
