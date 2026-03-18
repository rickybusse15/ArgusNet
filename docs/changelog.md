# Changelog

## March 15, 2026 — Viewer UI overhaul

### Usability
- Wrapped entire side panel in vertical scroll area; all content reachable via scrolling
- Reorganized UI into clear collapsible sections for operator workflow
- Added keyboard shortcut reference panel

### New panels
- **Scene header**: terrain bounds, elevation range, mesh resolution
- **Node summary table**: all nodes with sensor type, health, mobility, altitude
- **Track summary table**: all tracks with error, update count, staleness, measurement std
- **Alerts panel**: dynamic safety alerts (stale tracks, low health, untracked truths, high rejection)
- **Frame events panel**: launch events, generation/tracker rejections per frame with display caps
- **Covariance diagonals**: per-track position uncertainty (sigma_x/y/z)
- **Per-track error breakdown**: individual track localization errors

### Enhanced selection inspector
- **Tracks**: covariance diagonal, heading angle, color-coded staleness
- **Nodes**: sensor type, FOV angle, max range, mobile status
- **Truths**: distance to nearest track
- All types: full velocity vector, heading angle

### State extensions
- `SelectionState` extended with node sensor fields, covariance diagonal, truth-to-track distance
- Event lists capped at 20-30 displayed entries with overflow indicator

### Documentation
- Added `docs/VIEWER_UI.md`: panel layout, controls, verification checklist, known limitations

### Tests
- Added 4 new unit tests: covariance diagonal extraction (4x4, 3x3, empty), health color mapping
- All 43 tests pass

## March 15, 2026 — Project cleanup

- Consolidated 7 documentation files into `README.md` + `docs/`
- Removed root `smart_tracker/` shim package (use `pip install -e .` instead)
- Removed legacy HTML viewer (`viewer.py`, `static/`) and `platform.py` compat wrapper
- Added `CLAUDE.md` for AI agent context
- Hardened `.gitignore`

## March 14, 2026 — Native viewer, scene pipeline, terrain expansion

### Scene compilation pipeline
- Split scene compiler into modular helpers (`_glb.py`, `_scene_geometry.py`, `_scene_gis.py`, `_scene_style.py`)
- Added `smart-tracker build-scene` for replay-to-scene and GIS-to-scene compilation
- `smartscene-v1` package format: manifest, terrain GLB chunks, overlays, metadata

### Native Bevy viewer
- Rust viewer binary (`tracker-viewer`) replaces browser/canvas viewer
- Scene loading from disk, orbit/pan/zoom camera, egui controls
- Metrics panel, rejection markers, mission zones, layer toggles
- Fixed startup camera pitch to stay above terrain

### Terrain-aware mission zones
- `MissionZone.center.z` now stores terrain elevation baked at generation time
- Viewer samples `terrain.viewer_mesh()` to position zone rings above terrain
- Zone types: surveillance (blue), exclusion (red), patrol (yellow), objective (green)

### Rejection diagnostics with geometry
- `ObservationRejection` carries `origin`, `attempted_point`, `closest_point`, `blocker_type`, `first_hit_range_m`
- Two rejection streams: `rejected_observations` (tracker-side) and `generation_rejections` (simulation-side)
- Viewer draws rejection markers and LOS lines

### Separated terrain and platform presets
- `terrain_preset`: controls obstacles (`default` vs `clean_terrain`)
- `platform_preset`: controls sensor capabilities (`baseline` vs `wide_area`)
- Presets are independent of map size

### New terrain presets
- `jungle_canopy`, `arctic_tundra`, `military_compound`, `river_valley`

### New map scales
- `theater` (10x base), `operational` (15x base)

### Configurable counts
- `--target-count N`, `--drone-count N` CLI flags
- Dynamic generation instead of hardcoded 2+2

### Simulation behavior fixes
- Targets stay airborne with dynamic altitude variation
- Search interceptors switch to follow only after own-drone hit
- Airborne nodes can look downward at lower-altitude targets
- Follow drones hold standoff ring above assigned target

### Ingestion improvements
- `FileReplayIngestionAdapter` for offline replay through the ingestion pipeline
- Corrected `mean_ingest_latency_s` measurement in Rust daemon
- Documentation corrected: GNN and MQTT adapter were already fully implemented
- Approximate JPDA data association implemented (`tracker-core/src/association.rs:JPDAAssociator`)

### Verification
- 17 Rust tests pass, 115+ Python tests pass
- 28 new tests for terrain features and rejection diagnostics

## March 13, 2026 — Initial commit

- End-to-end workflow: Python simulation → Rust gRPC tracking → replay export
- Regional map with alpine terrain, 7 ground stations, 2 targets, 2 drones
- Kalman filtering, GNN data association, rejection validation
- Browser-based 3D viewer (HTML/canvas)
- MQTT and file replay ingestion adapters
