# CLAUDE.md — AI Agent Reference

## Project

Civilian 3D sensor fusion platform. Python simulation + Rust tracking engine + Bevy 3D viewer.

## Structure

```
src/smart_tracker/   Python package (install with pip install -e .)
rust/                Rust workspace (tracker-core, tracker-server, tracker-viewer, tracker-proto)
proto/               Protobuf service definition (tracker.proto)
tests/               Python tests
docs/                Architecture, usage, changelog
```

## Build & run

```bash
pip install -e .                    # Python deps
smart-tracker sim --duration-s 60   # Run simulation
cargo test                          # Rust tests
python3 -m pytest tests/ -q         # Python tests
```

## Key conventions

- **Frozen dataclasses** for all data models (`models.py`)
- **gRPC boundary**: Python calls Rust via `TrackingService.ingest_frame()`
- **Replay-as-contract**: replay JSON is the interface between simulation and viewer
- **Deterministic simulation** for fixed seed + config
- **Meter-based coordinates** throughout (XY local projected, Z above datum)
- **Radians** for angles internally (CLI flags labeled when degrees)

## Critical rules

- **Proto changes** require updating both Python bindings (`src/smarttracker/v1/`) AND Rust (`tracker-proto`, `tracker-core`, `tracker-server`)
- **Rust is source of truth** for tracking output — do not put tracking math in Python
- `environment.py` must continue re-exporting moved symbols for backward compatibility
- Replay metadata changes should be **additive** unless updating viewer + tests together
- Physical collision must never push entities below terrain

## Key files for common tasks

| Task | Start here |
|------|-----------|
| Simulation behavior | `src/smart_tracker/sim.py` |
| Terrain/obstacles | `terrain.py`, `environment.py`, `obstacles.py`, `visibility.py` |
| Tracking/filtering | `rust/tracker-core/src/lib.rs` |
| gRPC service | `proto/tracker.proto`, `service.py`, `rust/tracker-server/` |
| Scene packaging | `scene.py`, `_scene_*.py`, `_glb.py` |
| Viewer | `rust/tracker-viewer/src/` |

## Multi-agent coordination

See `AGENT_TEAM.md` for the multi-agent execution team protocol (Manager/Worker/Critical Reviewer structure).
