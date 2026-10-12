# CLAUDE.md — AI Agent Reference

## Project

Civilian 3D world-modeling platform. Current entry paths: the Rust CLI `argusnet-rs sim` (assembled eleven-stage run, no Python) and the Python simulation it is checked against; Rust world/safety/fusion; the Bevy viewer, which can run the Rust simulation in-process (`--run`). Target: Rust-only production execution with independently scheduled subsystems; Python remains optional offline tooling.

Read `docs/RUST_RUNTIME_ARCHITECTURE.md` and ADR-009 for the target, and
`docs/STATE_OWNERSHIP.md` for implemented versus planned behavior. Preserve the
legacy stage order during port acceptance; the independent scheduler is P4.10.
Hardware timing and actuation require the separate S3–S7 qualification gates.

## Structure

```
src/argusnet/        Python package (install with pip install -e .)
rust/                Rust workspace (argusnet-core, argusnet-server, argusnet-viewer, argusnet-proto, argusnet-world, argusnet-native, argusnet-types, argusnet-mapping, safety-engine)
proto/               Protobuf service definition (argusnet/v1/world_model.proto)
tests/               Python tests
docs/                Architecture, usage, changelog
```

## Build & run

```bash
pip install -e .                    # Python + native world library (requires Rust)
argusnet sim --duration-s 60        # Run simulation
cargo test                          # Rust tests
python3 -m pytest tests/ -q         # Python tests
cargo run --release -p argusnet-cli -- sim --config <cfg> --replay out.json   # Rust run (argusnet-rs)
scripts/ci-local.sh                 # every CI job locally (ruff, mypy, pytest, cargo, clippy, viewer); use before a PR
```

## Key conventions

- **Frozen dataclasses** for shared runtime/replay state (`src/argusnet/core/types.py`)
- **gRPC boundary**: Python calls Rust via `TrackingService.ingest_frame()`
- **Replay-as-contract**: replay JSON remains the persisted interchange contract; the target in-process viewer consumes immutable snapshots, not a second mutable simulation
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
| Scene packaging | `rust/argusnet-sim/src/scene_build.rs` (Rust, used by the viewer and `argusnet-rs build-scene`), `src/argusnet/world/scene_loader.py`, `_glb.py` (Python reference) |
| Viewer | `rust/argusnet-viewer/src/` |
