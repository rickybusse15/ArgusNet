# Smart Trajectory Tracker

Civilian 3D sensor fusion platform for research and observation workloads.

## Capabilities

- Multi-node bearing-only 3D tracking with fixed ground and mobile aerial nodes
- Rust gRPC tracking daemon with Kalman filtering, GNN and JPDA data association
- Environment-aware observation synthesis with terrain, obstacles, and LOS checks
- Replay-driven 3D visualization via native Bevy viewer
- `smartscene-v1` scene packaging for synthetic and GIS-backed terrain
- Export to GeoJSON, CZML, and Foxglove MCAP

## Quickstart

```bash
# Install Python dependencies
pip install -e .

# Run a simulation (generates replay.json + metrics.csv)
smart-tracker sim --duration-s 180 --dt 0.25 --terrain-preset alpine

# Build a scene package from replay
smart-tracker build-scene --replay replay.json --output scenes/demo-scene

# Export to GeoJSON
smart-tracker export --replay replay.json --format geojson --enu-origin "47.0,11.0,600" --output tracks.geojson
```

### Rust daemon

```bash
cargo run -p tracker-server --bin smart-trackerd -- serve --listen 127.0.0.1:50051
cargo test   # Run Rust tests
```

### Python tests

```bash
python3 -m pytest tests/ -q
```

## Architecture

```
Python (simulation, terrain, replay)     Rust (tracking, filtering, viewer)
─────────────────────────────────────    ────────────────────────────────────
sim.py          → scenario + observations   tracker-core   → Kalman + GNN
environment.py  → terrain, obstacles, LOS   tracker-server → gRPC daemon
scene.py        → smartscene-v1 compiler    tracker-viewer → Bevy 3D viewer
service.py      → gRPC client proxy         tracker-proto  → protobuf bindings
                        │
                        └── proto/smarttracker/v1/tracker.proto (contract)
```

**Boundary:** Python owns simulation/terrain/replay. Rust owns live tracking/filtering. The replay JSON format is the contract between them.

## Project layout

```
src/smart_tracker/       Python package (sim, terrain, fusion, export, scene)
rust/                    Rust workspace (tracker-core, tracker-server, tracker-viewer, tracker-proto)
proto/                   Protobuf service definition
tests/                   Python test suite
docs/                    Architecture, usage guide, changelog
```

## Use as a library

```python
from smart_tracker import (
    ScenarioOptions, SimulationConfig, TrackerConfig,
    TrackingService, build_default_scenario, run_simulation,
)

service = TrackingService(config=TrackerConfig())
scenario = build_default_scenario(ScenarioOptions(
    map_preset="regional", terrain_preset="alpine",
    ground_station_count=7, target_motion_preset="mixed",
))
result = run_simulation(scenario, SimulationConfig.from_duration(180.0, dt_s=0.25, seed=7))
```

## Documentation

- [Architecture & module map](docs/architecture.md)
- [Usage guide & CLI reference](docs/usage.md)
- [Changelog](docs/changelog.md)

## Status

Working prototype (March 2026). End-to-end local workflow functional. See [docs/changelog.md](docs/changelog.md) for recent work.

### Known gaps

1. Real sensor hardware driver wrappers (MQTT adapter is production-ready)
2. Per-node packet-loss and clock-drift alerting
3. Live streaming to native viewer (gRPC RPC exists server-side)
