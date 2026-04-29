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
- `--mission-mode`: `scan_map_inspect` for the map/localize/inspect workflow
- `--drone-mode`: `follow`, `search`, `mixed`
- `--drone-count N`
- `--clean-terrain`: terrain geometry without buildings/walls/vegetation

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
