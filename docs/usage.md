# Usage Guide

## Start here: current checkout

The installed `argusnet` CLI is currently Python orchestration with Rust
world/safety/fusion components. The following commands are existing entrypoints;
the [Rust-only scheduled runtime](RUST_RUNTIME_ARCHITECTURE.md) is a target,
not an additional available CLI flag. Use Python 3.10+ and the repository-pinned
Rust toolchain; source installation builds the native library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p runs
argusnet sim --map-preset small --duration-s 30 --seed 7 --replay runs/demo.json
argusnet build-scene --replay runs/demo.json --output runs/demo.smartscene
cargo run -p argusnet-viewer --release -- --scene runs/demo.smartscene --autoplay
```

Run from the repository root. Building the viewer needs the platform's graphics
and windowing dependencies. First builds are not startup-performance measurements.
For more options use `argusnet --help`, `argusnet sim --help`, and
`cargo run -p argusnet-viewer -- --help`. The current viewer opens a scene package,
not a bare replay through a `--replay` argument.

The planned [fidelity/time profiles](SIMULATION_FIDELITY.md) are not current CLI
flags. More sensor samples or a higher visual tier do not establish a more
accurate model. Current `target_count` configures scenario objects; a fused track
is the estimator's output, while an inspection POI identifies required work.
See [track semantics](FUSION.md#track-identity-lifecycle-and-time-contract-target).

## What is changing in the viewer (planned)

Current `real-world` / `scan-map` / `split` controls below remain current behavior.
The new [viewer specification](VIEWER_UI.md) introduces Operate/Plan/Review,
Perspective/Top-down/Section and independent layer presets with flat, restrained
instrument styling. Those names describe planned UI, not existing CLI values.

Basic Rust-worker Run/Pause/Step/Speed/Reset controls ship before the independent
scheduler is fully qualified. The target then shows source identity, pose/map/plan
freshness, pending commands, achieved speed, recording health and history bounds.
Diagnostics contains rates, deadline misses and queue details. Simulation Step
processes one control interval; playback on a live source changes only the viewed
history. Unsupported controls display a reason.

## Interpreting results and common problems

- Default mapping missions have no targets: an empty fused-track list is expected.
  Use coverage and inspection results, or the target-tracking example below.
- A scene is the current viewer input; build it from the replay before opening it.
- Rust/native loading errors require a working pinned toolchain and a successful
  source install; there is no silent Python world/safety fallback.
- Current split-view and team-localization graphics describe existing legacy
  behavior; they are not proof of qualified cooperative SLAM.
- Existing live sequence/drop counters describe transport delivery. Planned
  sensor-loss, freshness and recording-gap diagnostics distinguish different
  causes; do not infer one from another.
- SIL/HIL or flight capability is not enabled by opening a live stream. See
  [hardware qualification](HARDWARE_INTEGRATION.md) and [live operations](LIVE_OPERATIONS.md).

When reporting a problem, include command/config, seed, build/commit, OS/hardware,
source mode, execution profile if available, logs and the replay/scene manifest.
For performance reports also state cold/warm state, sensor/entity counts and
viewport/quality; compare quality/error as well as speed.

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

The default `argusnet sim` runs `scan_map_inspect` with `--target-count 0`, so it
reports mapping coverage and POI inspection rather than fused tracks — that is the
intended workflow, not an empty result.

### Target-tracking demo

To see the Rust fusion engine produce fused tracks, run the legacy target-tracking
mode with targets placed inside sensor range (a smaller map keeps drones close to
their assigned targets):

```bash
argusnet sim \
  --mission-mode target_tracking \
  --map-preset small --target-count 2 --drone-count 4 \
  --target-motion loiter --duration-s 60 --seed 7 \
  --replay tracking-replay.json
```

The run ends with a `Track RMSE` summary. On a large map (`regional` and up) a
handful of drones will not close the distance to scattered targets, so most
observations are rejected `out_of_range` and no tracks confirm; the CLI prints a
diagnostic hint when that happens.

### Run a simulation inside the viewer

The viewer can run a scenario itself instead of playing a recorded replay (P5.0,
D1). Export the scenario's configuration and build a scene for its world once:

```bash
argusnet sim --mission-mode target_tracking --map-preset small --target-count 3 \
  --drone-count 4 --target-motion mixed --duration-s 120 --seed 7 \
  --replay runs/demo.json --export-config runs/demo-config.json
argusnet build-scene --replay runs/demo.json --output runs/demo.smartscene
cargo run -p argusnet-viewer -- --scene runs/demo.smartscene --run runs/demo-config.json
```

`--run` also accepts a scenario bundle (`argusnet-rs build`). The run ticks on
its own thread, in Rust, with no Python. It opens paused at tick 0. The
simulation strip over the map has these controls:

- **Run / Pause** starts and stops it.
- **Step** runs exactly one tick.
- **Speed** sets wall-clock pacing from 0.5× to 10×. The physics timestep
  never changes.
- **Reset** starts the run again from tick 0, paused.

`--run` takes the same run options as `argusnet-rs sim`, with the same
meanings. They are applied by the same code, so the viewer's run equals the
CLI's byte for byte:

```bash
cargo run -p argusnet-viewer -- --scene runs/demo.smartscene --run runs/demo-config.json \
  --fusion-lag-s 1.5 --delivery-delay-s 0.25 --delivery-jitter-s 0.25 \
  --energy-profile physical_v1 --faults tests/fixtures/fault_manifest_example.json
```

An option the run cannot support (for example, a delay longer than the fusion
lag) is refused before the window opens. Reset keeps the options.

Each control shows as pending until the run confirms it took effect, and a
refusal says why (for example, Run on a finished run). The timeline holds
every tick so far; scrubbing it reviews history without rewinding the run.
Rewinding the run itself needs retained checkpoints and is shown disabled.

### Start from no world knowledge

By default the drones are given the true terrain and buildings. With
`--world-knowledge none` they know nothing but their launch pads: each carries
a range scanner, builds its own map, and plans only on that map.

```bash
argusnet-rs sim --config runs/demo-config.json --world-knowledge none --replay runs/scratch.json
cargo run -p argusnet-viewer -- --run runs/demo-config.json --world-knowledge none
```

No `--scene` is needed. Without one, the viewer opens on the run's map bounds
with nothing drawn, and the terrain and buildings appear as the drones measure
them. Pass `--scene` to have the true world available for comparison; it is
hidden when the run opens and stays in the Layers panel.

The replay's `mapping_state.belief` reports how much of the map is known and
how far its ground is from the truth. `world_contacts` lists any time the map
led a drone somewhere physics had to correct. See
`docs/migration/UNKNOWN_WORLD.md`.

To keep the map, save it as a scene package and open it like any other world:

```bash
argusnet-rs sim --config runs/demo-config.json --world-knowledge none --export-scene runs/demo-map
cargo run -p argusnet-viewer -- --scene runs/demo-map
```

The package holds the measured ground and the buildings as 3D models, the map
itself, and the run that built it. Opening it plays that run on the timeline.
In the viewer, **Save map as scene** in the map readout saves the map at the
cursor to `./argusnet-maps/`.

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
