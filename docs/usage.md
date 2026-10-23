# Usage Guide

## Start here: current checkout

Two simulation entrypoints ship: `argusnet-rs` builds and runs scenarios in Rust,
and the Python `argusnet` CLI remains the reference and offline tooling entrypoint.
The viewer runs the Rust simulation in process with `--run`, without requiring
Python or a prebuilt scene. Independent scheduling is partially implemented:
`--executor lanes` with `--mission-rate`, `--mapping-rate` and `--fusion-rate`,
plus per-sensor rates and delayed delivery (see below). The complete
[scheduled runtime](RUST_RUNTIME_ARCHITECTURE.md) remains a target.

The following Python workflow uses Python 3.10+ and the repository-pinned Rust
toolchain; source installation builds the native library.

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

## Viewer shell and remaining work

Operate/Plan/Review workspace controls, Perspective/Top-down/Section projections
and Operational/Reconstruction/Safety layer presets are implemented in the shell.
Plan and Review display unavailable reasons until their canonical mission,
journal and evidence contracts exist. These are UI controls; `--view-mode`
continues to accept `real-world`, `scan-map` and `split`.

Rust-worker Run/Pause/Step/Speed/Reset controls are implemented (see below).
The [viewer specification](VIEWER_UI.md) describes the full target, including
complete source freshness, achieved speed, recording health and history bounds.
Full scheduler qualification and pending-job history remain open.

## Interpreting results and common problems

- Default mapping missions have no targets: an empty fused-track list is expected.
  Use coverage and inspection results, or the target-tracking example below.
- Replay playback requires a scene package; an in-process `--run` builds its
  own scene, or starts empty under `--world-knowledge none`.
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
- `--terrain-preset`: `alpine`, `coastal`, `urban_flat`, `urban_dense`, `desert_canyon`, `rolling_highlands`, `lake_district`, `jungle_canopy`, `arctic_tundra`, `river_valley`, `military_compound`, `mountain_pass`. `urban_dense` is a flat city of 30 blocks 70–110 m tall on a 6 × 5 grid with streets at least 42 m wide (scaled with the map); POIs and ground stations land in the streets.
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

A checked-in bundle can run directly, with no Python or prebuilt scene:

```bash
cargo run --release -p argusnet-viewer -- --run tests/fixtures/scenario_bundle_scan.json
```

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

`--planning-policy efficient_v1` (on `argusnet-rs sim` and the viewer's
`--run`) allocates a scan mission's POIs across the whole fleet at once
(PLANNING_EFFICIENCY_PLAN P1) instead of the reference's one drone at a time
in fleet order. The default, `reference`, is unchanged. The replay records a
non-reference policy in `meta.planning_policy`, and a checkpoint resumes only
under the policy it was taken with.

`--mission-rate NUM/DEN` runs the mission lane at its own rate (B2.4,
`scheduled_v1`), `--mapping-rate NUM/DEN` the mapping lane (B2.5) and
`--fusion-rate NUM/DEN` the fusion lane (B2.6): the
viewer's run is then the CLI's `--executor lanes` run with the same rates,
frame for frame, and a reset starts the lanes' deadlines over.

`--sensor-rate NODE_ID=NUM/DEN` (on `argusnet-rs sim` and the viewer's `--run`,
repeatable) has that sensor capture at its own rate instead of every tick: at
the first tick at or after each of its deadlines, and nothing in between. The
replay records it in `meta.sensor_rates`; the tracks then carry
`evidence_time_s`, so the track summary shows how old each track's last
measurement is.

Each control shows as pending until the run confirms it took effect, and a
refusal says why (for example, Run on a finished run). The timeline holds
every tick so far; scrubbing it reviews history without rewinding the run.
Rewinding the run itself needs retained checkpoints and is shown disabled.

### Build a scene package without Python

```bash
argusnet-rs build-scene --replay runs/demo.json --output runs/demo.smartscene
```

The package is the one `argusnet build-scene` writes: terrain, buildings,
walls and vegetation as GLB models, the styles, the environment metadata and
the replay. A replay that breaks the replay contract (no frames, an
unsupported `schema_version`, a `frame_count` that does not match) is refused,
as Python refuses it. `argusnet-viewer --run <scenario>` builds one itself
when no `--scene` is given, except under `--world-knowledge none`, which starts
with only the map bounds. Pass `--scene` to include the true world for comparison.

### Real buildings in a run (R1)

Import building footprints (GeoJSON, or an OSM export as GeoJSON) into the
run's local metres, then name the result in a scenario config:

```bash
python3 scripts/import_buildings.py town.geojson --out town.buildings.json
```

The importer projects the footprints around an origin (the centre of their
extent, or `--origin LAT,LON`) and records it with the source CRS. Heights
come from `height`, then `building:levels` x 3 m, then a default by building
type, and roofs from `roof:shape` (flat, gabled, hipped, pyramidal). Each
building says which; a roof it cannot model is written flat at the full
height. In the config's `options`, set `"buildings_path": "town.buildings.json"`.
The run then uses those buildings in place of the procedural ones, with
their roofs, in line of sight, collision and the viewer, and its replay
records the georeference.

### Real terrain in a run (R5)

A DEM (a single-band GeoTIFF) becomes the run's terrain the same way. Import
it for the run's map preset, with the buildings' origin so the two line up:

```bash
python3 scripts/import_terrain.py dem.tif --map-preset medium --out town.terrain.json --origin 47.6,-122.3
```

The importer samples the DEM at each cell of that preset's height grid,
placing every cell through the DEM's own CRS, so a rotated raster or a CRS
whose grid north is not true north still lines up. It refuses a DEM that
does not cover the whole map. Heights are the DEM's elevations less an
offset that puts the lowest ground 2 m above the ground plane; the offset
is recorded as the origin's altitude. In the config's `options`, set
`"terrain_path": "town.terrain.json"` with the same `map_preset`. Trees,
power lines and buildings then stand on the real ground. With
`buildings_path` too, both files must share an origin, or the build fails
and says so.

### Real land cover in a run (R5)

Water, land use and roads come from OSM (or any GeoJSON with OSM's tags):

```bash
python3 scripts/import_features.py town.geojson --out town.features.json --origin 47.6,-122.3
```

Polygons tagged as water, wetland, forest, scrub, rock, glacier, urban land
use or open ground become that land cover. Rivers, canals and streams are
drawn at their `width`, and vehicle roads at their `width`, their `lanes` x
3.5 m, or a default by class. Tunnels and bridges are left out. Anything
else is skipped and counted in the summary. In the config's `options`, set
`"features_path": "town.features.json"`. The run's land cover is then the
data's: open where the data says nothing, rock where the ground is steeper
than about 30 degrees, and nothing invented. Roads run through forests,
buildings stay urban, and asked-for trees grow only in the mapped forests.
The procedural environment's walls and forest stands are dropped. Every
real layer must share one origin.

The same file carries mapped trees (`natural=tree` points, `natural=tree_row`
lines) and power lines (`power=line|minor_line`, a support at each vertex).
A tree keeps its tagged `height`, `diameter_crown` and `circumference`, and
otherwise takes its species' typical size. A support takes the `height` of a
tower or pole mapped on it, or 30 m (line) or 10 m (minor line). `cables`
sets the conductors and `voltage` the clearance a drone keeps. With a
features file, these lines replace the procedural one. Asked-for scattered
trees keep 8 m from mapped ones. Anything outside the map or inside a
building is left out and counted in the replay's `features` record.

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

Follow drones in such a run chase the run's own fused tracks, not the targets'
true positions, and search drones fly to the blank parts of the map rather than
along pre-planned lanes. `--pursuit truth` and `--explore lanes` bring back the
old behaviour for comparison.

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
- `--map-preset` sets map size: `small`, `medium`, `large`, `xlarge`, `regional`, `theater`,
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

### Replay-derived client assurance package (S1)

```bash
argusnet client-package --replay mapping.json --output reports/mapping-assurance
```

The new output directory contains `manifest.json`, `assurance.json`,
`assurance.md` and an annotated `replay.json`. An existing package directory is
refused, so a reviewed package is never overwritten. The manifest lists the
SHA-256 of the exact delivered bytes, and the source replay's canonical JSON
hash: the input as given, as canonical JSON (sorted keys, indent 2, trailing
newline), less any previous assurance extension, and less its `summary` when that
held nothing else. It is unsigned. The package's status never contradicts the
F1 outcome it carries: a run is completed only when it reached its terminal
phase and F1 finds it completed. The replay is read under the replay-input size
cap.

`assurance.json` (`argusnet.mission-assurance/v1`) records:
- the outcome: its own status, plus the F1 record (`argusnet.mission-outcome/1`)
  with every reason and the frames and events each rests on;
- recorded POI results, each pointing at its frame;
- the minimum measured battery fraction;
- localization timeouts;
- events, each with a frame reference.

Missing provenance, evidence acceptance, landing, recording completeness and
physical localization integrity stay unavailable. Legacy POI dwell completion
and timeout confidence do not establish them. The package does not include
reconstruction, raw capture, change detection or qualification products.

Evaluation metrics write missing values as JSON `null`, and text reports show
`N/A`. Energy comes from the recorded `battery_fraction`, not from trajectory
estimates, and zero remains a flat battery. A run with no required objectives
has no completion rate. It is never 100%, and the pass/fail check treats it as
not applicable rather than failing it. Unmatched position-error data fails
qualification with an unavailable reason. Benchmark means exclude unavailable
values.

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
