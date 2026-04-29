# Viewer UI Reference

The viewer presents replay state for mapping, localization, inspection, safety, and multi-drone
coordination. The interface should make it clear what has been mapped, where drones are localized,
which POIs are being inspected, and why safety or deconfliction events occurred.

## Panel Layout

### Scene Header

- Scene ID and coordinate reference system
- Terrain bounds and elevation range
- Terrain mesh resolution
- Replay duration and frame count

### Scenario Controls

- Map, terrain, platform, workflow, and drone mode presets
- Drone count, duration, and seed controls
- Run simulation button for synthetic scenes
- Progress bar during simulation pipeline

### Playback

- Play/Pause
- Step forward/back
- Jump to start/end
- Speed slider
- Frame scrubber
- Current timestamp and frame index

### Mapping

- Coverage fraction
- Covered/total cells
- Mean revisit count
- Newly scanned cells
- Coverage and uncertainty overlays

### Localization

- Active localization count
- Mean position uncertainty
- Mean confidence
- Per-drone localization estimate and confidence
- Timeout indicator when the mission advanced through timeout handling

### Inspection

- POI list and status
- Assigned drone
- Dwell accumulated and completion time
- Planned/achieved viewpoints when available
- Inspection events and blocked reasons

### Drones

- ID, health, mobile status, altitude, battery fraction
- Sensor details: type, FOV, and range
- Current route/phase when available

### Coordination And Safety

- Coordinator drone ID
- Deconfliction events
- Egress progress
- Return-home distance
- Safety/rejection events

### Frame Events

- Mapping updates
- Localization updates
- Inspection events
- Deconfliction events
- Observation/rejection summaries

### Layers

- Terrain
- Map coverage
- Localization ellipses
- Inspection POIs
- Routes and egress paths
- Mission zones
- Sensor FOV cones

### Mission Zones

- Mapping area
- Inspection area
- Exclusion zone
- Return-home area
- Revisit area

## Controls

| Input | Action |
|-------|--------|
| Space | Play / Pause |
| Left Arrow | Previous frame |
| Right Arrow | Next frame |
| Home | Jump to first frame |
| End | Jump to last frame |
| Right-drag | Orbit camera |
| Middle-drag or Shift+right-drag | Pan camera |
| Scroll wheel | Zoom in/out |
| Left-click marker | Select for inspection |

## Verification

1. Playback updates mapping, localization, inspection, and egress panels each frame.
2. Selecting a drone shows live position, battery, and localization state.
3. Selecting a POI shows assignment, dwell, and completion state.
4. Mapping overlays match `MappingState` and `ScanMissionState.newly_scanned_cells`.
5. Localization overlays match `LocalizationEstimate` uncertainty and confidence.
6. Deconfliction and inspection event lists match replay frame contents.

## Known Limitations

- Panel width is limited; very long IDs may be truncated.
- Event display is per-frame only unless replay history is explicitly accumulated.
- Some roadmap inspection evidence/reconstruction panels depend on future indexing support.
