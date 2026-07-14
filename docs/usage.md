# Usage Guide

## CLI Commands

### Simulation

```bash
argusnet sim \
  --duration-s 180 --dt 0.25 \
  --map-preset regional --terrain-preset alpine \
  --platform-preset baseline \
  --mission-mode scan_map_inspect --drone-mode mixed \
  --csv metrics.csv --replay replay.json
```

Key flags:
- `--map-preset`: `small`, `medium`, `large`, `xlarge`, `regional`, `theater`, `operational`
- `--terrain-preset`: `alpine`, `coastal`, `urban_flat`, `desert_canyon`, `rolling_highlands`, `lake_district`, `jungle_canopy`, `arctic_tundra`, `river_valley`
- `--platform-preset`: `baseline`, `wide_area`
- `--mission-mode`: `scan_map_inspect` for the map/localize/inspect workflow, or
  `target_tracking` for the legacy fused-track workflow
- `--drone-mode`: `follow`, `search`, `mixed`
- `--drone-count N`
- `--clean-terrain`: terrain geometry without buildings/walls/vegetation
- `--demo tracking`: curated in-range target-tracking scene that confirms fused
  tracks out of the box (explicit flags still override)

The default `argusnet sim` runs `scan_map_inspect` with `--target-count 0`, so it
reports mapping coverage and POI inspection rather than fused tracks — that is the
intended workflow, not an empty result.

### Target-tracking demo

To see the Rust fusion engine produce fused tracks out of the box, use the curated
demo shortcut:

```bash
argusnet sim --demo tracking --duration-s 60 --replay tracking-replay.json
```

`--demo tracking` seeds an in-range `target_tracking` scene (`--map-preset small`,
3 targets, 4 drones, `loiter` motion) whose targets stay inside drone sensor range,
so tracks confirm. Any explicit flag still wins, e.g. `--demo tracking
--drone-count 6` or `--demo tracking --map-preset medium`. The equivalent manual
form is:

```bash
argusnet sim \
  --mission-mode target_tracking \
  --map-preset small --target-count 3 --drone-count 4 \
  --target-motion loiter --duration-s 60 --seed 7 \
  --replay tracking-replay.json
```

The run ends with a `Track RMSE` summary. On a large map (`regional` and up) a
handful of drones will not close the distance to scattered targets, so most
observations are rejected `out_of_range` and no tracks confirm; the CLI prints a
diagnostic hint (pointing at `--demo tracking`) when that happens.

### Mapping run with split-view 3-D reconstruction

Run a scan/map/inspect mission, package it, and open the viewer with the real
terrain on the left and a 3-D reconstruction of the believed terrain on the right:

```bash
argusnet sim --mission-mode scan_map_inspect --map-preset xlarge \
  --terrain-resolution-m 25 --coverage-resolution-m 30 --drone-count 8 \
  --frontier-exploration --duration-s 180 --seed 7 --replay mapping.json
argusnet build-scene --replay mapping.json --output scenes/mapping-scene
argusnet-viewer --scene scenes/mapping-scene --view-mode split --autoplay
```

Cooperative search (opt-in): add `--cooperative-search` to launch every drone from
one grounded origin and fan them out into disjoint angular wedges of the search
area, then fuse their per-drone localizations (all referenced to the shared
coverage map) into a team estimate. `--search-origin X,Y` sets the common origin
(meters; defaults to the search-area center). The CLI summary reports the grounded
origin and the team co-localization convergence.

Add `--adaptive-search` to refine it: drones redirect via wedge-aware,
coverage-balanced frontier selection (reallocating from finished wedges to the
most under-covered one), and the wedge origin re-anchors on the fused team estimate
— a localization→planning feedback loop that covers more ground more evenly.

Add `--occlusion-aware-mapping` for dense obstacle mapping runs. The scan map and
split-view reconstruction only mark cells with line of sight from a drone, and
frontier/POI/egress redirects use the 2-D planner to route around hard blockers.
The flag is opt-in so default fixed-seed replays keep their legacy behavior.

The team estimate is recorded per frame in the replay
(`scan_mission_state.team_localization`) and rendered by the viewer (a cyan beacon
with spokes to each contributing drone), including in headless `--view-mode split`
stills and sequences.

```bash
argusnet sim --mission-mode scan_map_inspect --map-preset medium --drone-count 6 \
  --cooperative-search --coverage-resolution-m 40 --duration-s 120 --seed 7 \
  --replay coop.json
```

Resolution and size:
- `--map-preset` sets map size: `small`, `medium`, `large`, `xlarge`, `theater`,
  `operational` (increasing extent) — larger maps exercise the 2-D path planner
  harder.
- `--terrain-resolution-m` sets terrain grid fidelity (smaller = higher-res).
- `--coverage-resolution-m` sets the scan/reconstruction grid cell size (default
  50 m; smaller gives a denser reconstruction). It is recorded additively under
  `meta.scenario_options.coverage_resolution_m`.

Viewer:
- `--view-mode` accepts `real-world` (default), `scan-map` (reconstruction only),
  or `split`. In split mode the left half shows the real terrain and the right
  half is an empty area that fills with a 3-D reconstruction of the believed
  terrain, rendered from an angled perspective camera on its own render layer
  (no real terrain, drones, or gizmos leak into it). The `M` key cycles the modes.
- `--autoplay` starts the timeline immediately. The reconstruction accumulates
  from the mission's per-frame `newly_scanned_cells`, so it builds up as the
  drones cover the map (rewinding resets it).

Independent controls in split view: mouse input is routed to whichever region the
cursor is over, so the two panes and the drawer each have their own scroll wheel.
- Cursor over the **left** pane: right-drag orbits, middle-drag (or shift +
  right-drag) pans, and the scroll wheel zooms the real-terrain view.
- Cursor over the **right** pane: the same controls orbit / pan / zoom the 3-D
  reconstruction independently, without moving the left view.
- Cursor over the **drawer / tabs**: the scroll wheel scrolls the panel content
  and the cameras stay put.

### Build scene

```bash
# From replay
argusnet build-scene --replay replay.json --output scenes/demo-scene

# From GIS data
argusnet build-scene \
  --dem path/to/dem.tif --source-crs EPSG:32611 \
  --roads roads.geojson --water water.geojson --buildings buildings.geojson \
  --output scenes/gis-scene
```

### Ingest (live or replay)

```bash
# File replay
argusnet ingest --replay-file replay.json --enu-origin "47.0,11.0,600"

# MQTT live
argusnet ingest --mqtt-broker localhost --enu-origin "47.0,11.0,600"
```

### Export

```bash
argusnet export --replay replay.json --format geojson \
  --enu-origin "47.0,11.0,600" --output observations.geojson
```

Formats: `geojson`, `czml`, `foxglove`

## Python API

### Run a simulation

```python
from argusnet import (
    ScenarioOptions, SimulationConfig, build_default_scenario, run_simulation,
)

scenario = build_default_scenario(ScenarioOptions(
    map_preset="regional", terrain_preset="alpine",
    mission_mode="scan_map_inspect",
))
result = run_simulation(scenario, SimulationConfig.from_duration(180.0, dt_s=0.25, seed=7))
```

### Live sensor fusion

```python
from argusnet import TrackingService, TrackerConfig
from argusnet.core.types import BearingObservation, NodeState, vec3

service = TrackingService(config=TrackerConfig())
frame = service.ingest_frame(
    timestamp_s=0.0,
    node_states=[NodeState("sensor-a", vec3(0, 0, 50), vec3(0, 0, 0), False, 0.0)],
    observations=[...],
    truths=[],
)

print(frame.metrics.accepted_observation_count, frame.metrics.rejected_observation_count)

# Rejection diagnostics
for rej in frame.rejected_observations:
    print(f"{rej.reason}: {rej.blocker_type} at {rej.first_hit_range_m}m")

service.close()
```

### Terrain and zones

```python
from argusnet.world.environment import Bounds2D, TerrainLayer
from argusnet.core.types import MissionZone, vec3, ZONE_TYPE_OBJECTIVE

terrain = TerrainLayer.from_height_grid(
    environment_id="test", bounds_xy_m=bounds, heights_m=heights, resolution_m=10.0,
)
height = terrain.height_at(500.0, 500.0)
zone = MissionZone(
    zone_id="inspect-north", zone_type=ZONE_TYPE_OBJECTIVE,
    center=vec3(500.0, 500.0, height), radius_m=150.0,
)
```

### Presets

Terrain and platform presets are independent:
- `terrain_preset` controls obstacle composition (buildings, walls, vegetation)
- `platform_preset` controls sensor capabilities (range, speed)
- Both work on any map size

## Testing

```bash
python3 -m pytest tests/ -q          # All Python tests
cargo test                            # All Rust tests
python3 tests/example_integration.py  # End-to-end example
```
