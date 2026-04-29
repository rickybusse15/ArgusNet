# INDEXING.md — Spatial Memory, Retrieval, and Data Persistence Contract

This document defines the indexing subsystem for ArgusNet: how observations, maps, inspection data, and mission artifacts are stored, retrieved, and reused across time.

Indexing is what turns ArgusNet from a one-time mission system into a persistent world-model platform.

---

## 1. Purpose

The indexing subsystem answers:

- Have we seen this place before?
- Where was this observation captured?
- What data exists for this POI or region?
- What changed between missions?
- Which areas are well-covered or under-observed?
- What evidence supports a given inspection result?

---

## 2. Why indexing matters

Without indexing, ArgusNet cannot:

- reuse maps across missions
- relocalize effectively
- revisit inspection POIs
- compare data over time
- build spatial memory

Indexing is required for:

- localization (retrieving landmarks/keyframes)
- mapping (storing world updates)
- inspection (storing evidence/reconstructions)
- evaluation (retrieving mission outputs)

---

## 3. Core data types

Indexing should store structured artifacts:

- keyframes (images + pose)
- landmarks / descriptors
- map tiles (terrain, occupancy, semantic layers)
- inspection POIs
- inspection evidence sets
- local reconstructions
- mission logs
- safety events
- change detection records

---

## 4. Keyframe model

```text
Keyframe
  keyframe_id: str
  timestamp_s: float
  platform_id: str
  pose: pose + covariance
  sensor_data_refs: list[str]
  descriptors: optional
  world_region_id: optional
```

Keyframes are the primary unit for relocalization and map alignment.

---

## 5. Spatial indexing

Data should be queryable by:

- spatial region (x, y, z bounds)
- geofence or mission region
- poi_id
- timestamp range
- sensor type
- confidence thresholds

Possible structures:

- grid index
- k-d tree
- R-tree
- tile-based map storage

---

## 6. Retrieval interface

```text
IndexQuery
  keyframes_in_region(bounds)
  nearest_keyframes(position)
  observations_for_poi(poi_id)
  reconstructions_for_poi(poi_id)
  coverage_map(region)
  recent_missions(region)
  change_records(poi_id)
```

---

## 7. Write interface

```text
IndexWrite
  add_keyframe(...)
  add_observation(...)
  add_map_update(...)
  add_inspection_poi(...)
  add_evidence_set(...)
  add_reconstruction(...)
  add_change_record(...)
  add_mission_log(...)
```

---

## 8. Temporal dimension

Indexing must support time-based queries:

- historical comparison
- revisit planning
- change detection
- mission playback

---

## 9. Data integrity

Indexing should preserve:

- source of data
- uncertainty
- timestamps
- versioning
- alignment to world model

---

## 10. Integration with other subsystems

| Subsystem | Role in indexing |
|-----------|------------------|
| Mapping | writes world updates |
| Localization | reads keyframes and landmarks |
| Inspection | writes evidence and reconstructions |
| Planning | reads coverage and POI data |
| Mission execution | logs events and decisions |
| Evaluation | reads mission outputs |

---

## 11. Implementation phases

### Phase 1
- keyframe store
- simple spatial queries

### Phase 2
- map tile storage
- observation indexing

### Phase 3
- inspection data + reconstructions

### Phase 4
- change detection history
- time-series queries

### Phase 5
- optimization + scaling

---

## 12. Success criteria

Indexing is successful when ArgusNet can:

- store and retrieve keyframes and observations
- support relocalization queries
- persist inspection evidence and reconstructions
- compare data across missions
- support planning decisions based on memory

At that point, ArgusNet becomes a persistent spatial memory system rather than a stateless mission runner.
