# Viewer UI Reference

## Panel Layout

The viewer uses a single resizable left side panel with vertical scrolling.
All sections are organized top-to-bottom by operational priority.

### Scene Header
- Scene ID and coordinate reference system
- Terrain bounds (X/Y extents in meters)
- Elevation range (min/max height in meters)
- Terrain mesh resolution (if loaded)

### Scenario Controls (collapsible)
- Map, terrain, platform, motion, and drone mode presets
- Target/drone/station count sliders
- Duration and seed controls
- "Run Simulation" button (synthetic scenes only)
- Progress bar during simulation pipeline

### Playback
- Replay summary: frame count, total duration
- Play/Pause, step forward/back, jump to start/end buttons
- Speed slider (0.25x to 4x)
- Frame scrubber slider
- Current timestamp and frame indicator

### Tracking Metrics (collapsible)
- Active track count
- Mean and max localization error (meters)
- Observation count, acceptance rate with progress bar
- Rejection breakdown by reason (collapsible)
- Mean measurement standard deviation
- Per-track error breakdown (collapsible)

### Nodes / Drones (collapsible)
Summary table of all sensor nodes in the current frame:
- ID, sensor type, health %, mobile status, altitude
- Sensor details sub-section: FOV and max range per node

### Tracks (collapsible)
Summary table of all active tracks in the current frame:
- Track ID, localization error, update count, stale steps, measurement std
- Covariance diagonals sub-section (sigma_x, sigma_y, sigma_z in meters)

### Selection Inspector
Click any marker in the 3D viewport to inspect it:

**All marker types:** position (x/y/z meters), velocity vector and speed, heading angle

**Tracks:** measurement std, update count, stale steps (color-coded), track error, covariance diagonal

**Nodes:** health (color-coded), sensor type, mobile status, FOV angle, max range

**Truths:** distance to nearest track

### Alerts (collapsible)
Dynamically computed from current frame state:
- Stale tracks (>5 steps without update)
- Low health nodes (<30%)
- Untracked truths (no track within 50m)
- High rejection rate (>50%)

Green "No active alerts" when clean.

### Frame Events (collapsible)
Events from the current replay frame:
- Launch events: drone, station, target, launch time
- Generation rejections: reason, node, target, blocker type (capped at 30 displayed)
- Tracker rejections: reason, node, target, detail (capped at 30 displayed)
- Observation count summary

### Layers (collapsible)
Toggle visibility of base GIS/terrain layers loaded from the scene package.

### Runtime Overlays (collapsible)
Toggle visibility of 9 overlay categories:
- Tracks, Truths, Nodes
- Observation Rays, Rejection Markers
- Mission Zones, FOV Cones, Radar Rings, Launch Lines

### Mission Zones (collapsible)
- Type legend: S=surveillance, X=exclusion, P=patrol, O=objective
- Zone groups with chip summary (type counts)
- Click a group to focus the camera/zone rendering on it
- Expand groups to see individual zone details (type, label, radius, priority)

### Keyboard Shortcuts (collapsible)
Reference table shown at the bottom of the panel.

## Keyboard Controls

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| Left Arrow | Previous frame |
| Right Arrow | Next frame |
| Home | Jump to first frame |
| End | Jump to last frame |

## Mouse Controls

| Input | Action |
|-------|--------|
| Right-drag | Orbit camera |
| Middle-drag (or Shift+right-drag) | Pan camera |
| Scroll wheel | Zoom in/out |
| Left-click on marker | Select for inspection |

## Overflow and Scrolling

The entire side panel is wrapped in a vertical scroll area. All content is reachable
by scrolling even when the window is too small to show everything at once.

Lists with potentially unbounded content (rejections, events) are capped at display
limits (20-30 items) with a "... and N more" indicator.

## Verifying UI is Connected to Real State

1. Start playback — all value fields should update each frame
2. Click a track marker — selection panel should show live position updates
3. Check Tracking Metrics — values should change as the scenario progresses
4. Open Alerts — stale track warnings should appear/disappear as tracks age
5. Check Node summary — health values should reflect actual node degradation
6. Step frame by frame — all panels should update consistently

## Known Limitations

- Panel width is limited; very long IDs may be truncated
- Covariance display assumes 4x4 or 3x3 matrix layout
- Safety alerts use fixed thresholds (stale=5, health=30%, untracked=50m, rejection=50%)
- Event display is per-frame only (no cross-frame event history)
- Zone badges overlay may overlap with the side panel at extreme window sizes
