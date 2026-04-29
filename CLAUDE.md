# CLAUDE.md — AI Agent Reference

## Project

Civilian 3D world-modeling platform. Python simulation + Rust sensor-fusion service + Bevy 3D viewer.

## Structure

```
src/argusnet/        Python package (install with pip install -e .)
rust/                Rust workspace (argusnet-core, argusnet-server, argusnet-viewer, argusnet-proto)
proto/               Protobuf service definition (argusnet/v1/world_model.proto)
tests/               Python tests
docs/                Architecture, usage, changelog
```

## Build & run

```bash
pip install -e .                    # Python deps
argusnet sim --duration-s 60        # Run simulation
cargo test                          # Rust tests
python3 -m pytest tests/ -q         # Python tests
```

## Key conventions

- **Frozen dataclasses** for shared runtime/replay state (`src/argusnet/core/types.py`)
- **gRPC boundary**: Python calls Rust via `TrackingService.ingest_frame()`
- **Replay-as-contract**: replay JSON is the interface between simulation and viewer
- **Deterministic simulation** for fixed seed + config
- **Meter-based coordinates** throughout (XY local projected, Z above datum)
- **Radians** for angles internally (CLI flags labeled when degrees)

## Critical rules

- **Proto changes** require updating both Python bindings (`src/argusnet/v1/`) AND Rust (`argusnet-proto`, `argusnet-core`, `argusnet-server`)
- **Rust is source of truth** for fused object-state output — do not put duplicated fusion math in Python
- `environment.py` must continue re-exporting moved symbols for backward compatibility
- Replay metadata changes should be **additive** unless updating viewer + tests together
- Physical collision must never push entities below terrain

## Key files for common tasks

| Task | Start here |
|------|-----------|
| Simulation behavior | `src/argusnet/simulation/sim.py` |
| Terrain/obstacles | `src/argusnet/world/terrain.py`, `environment.py`, `obstacles.py`, `visibility.py` |
| Tracking/filtering | `rust/argusnet-core/src/lib.rs` |
| gRPC service | `proto/argusnet/v1/world_model.proto`, `src/argusnet/adapters/argusnet_grpc.py`, `rust/argusnet-server/` |
| Scene packaging | `src/argusnet/world/scene_loader.py`, `_glb.py` |
| Viewer | `rust/argusnet-viewer/src/` |
