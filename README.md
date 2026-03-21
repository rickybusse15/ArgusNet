# ArgusNet

**ArgusNet: world models for autonomous aerial mapping, inspection, localization, and spatial memory.**

ArgusNet is an open-source platform for building and using **world models** from drone observations. It is designed to support **search, mapping, inspection, localization, and structured data collection** in simulation and authorized real-world research environments.

Rather than treating a drone flight as a one-time stream of sensor data, ArgusNet treats each mission as part of a growing spatial understanding of the environment. The platform is meant to help answer four core questions:

- **Where am I?**
- **What does the world look like?**
- **What have I already seen?**
- **Where should I go next?**

## Overview

ArgusNet combines environment modeling, sensing, localization, mapping, indexing, planning, and evaluation into one modular system. It is intended for research, education, infrastructure inspection, survey workflows, benchmarking, and controlled field testing.

Today, this repository contains an actively evolving prototype whose current implementation centers on simulation, terrain, sensing, tracking, planning, replay, export, and visualization workflows. The existing `smart_tracker` Python package and Rust tracking crates remain the implementation backbone while the project evolves toward the broader ArgusNet architecture.

## Core Goal

ArgusNet aims to provide a platform where drones can:

- build structured representations of environments from aerial observations
- localize themselves relative to those environments
- index and retrieve previous observations, landmarks, and map regions
- plan search, mapping, and inspection actions based on uncertainty and coverage
- generate replayable, measurable mission outputs for training and evaluation

## Design Principles

- **World-model first** — the environment should be a structured, queryable internal representation, not just a background map
- **Localization-aware** — the platform must continuously estimate where the drone is relative to both local motion and persistent maps
- **Memory and indexing** — observations should be stored, retrieved, compared, and reused across missions
- **Simulation-first** — all major capabilities should work in realistic simulation before field use
- **Modular architecture** — sensing, localization, mapping, planning, and evaluation should remain separable
- **Human-supervised use** — real-world deployment should remain controlled, reviewable, and safety-bounded

## Primary Use Cases

- aerial mapping
- infrastructure and roof inspection
- construction progress capture
- environmental survey
- agricultural observation
- search coverage of unobserved regions
- repeated inspection over time
- synthetic dataset generation
- localization and world-model benchmarking

## Scope Boundaries

ArgusNet is intended for:

- mapping
- inspection
- localization
- survey
- scene understanding
- coverage planning
- authorized research and field testing

ArgusNet is **not** intended for:

- weaponization
- autonomous engagement
- unlawful surveillance
- cyber intrusion or takeover
- harmful pursuit workflows
- any deployment that violates applicable aviation, privacy, or local laws

Users are responsible for ensuring that any real-world use complies with all applicable laws, regulations, permissions, and safety requirements. See [SAFETY.md](SAFETY.md) and [LEGAL.md](LEGAL.md).

## Conceptual Architecture

```text
                           +----------------------+
                           |      ArgusNet        |
                           |  World Model System  |
                           +----------+-----------+
                                      |
      -----------------------------------------------------------------
      |                |                |               |              |
      v                v                v               v              v

+-------------+  +-------------+  +-------------+  +-------------+  +-------------+
|    World    |  |   Sensing   |  | Localization|  |   Mapping   |  |  Indexing   |
| Environment |  |  Perception |  |  Pose Est.  |  | World State |  | Spatial Mem.|
+------+------+  +------+------+  +------+------+  +------+------+  +------+------+
       |                |                |               |              |
       |                |                |               |              |
       +----------------+----------------+---------------+--------------+
                                      |
                                      v
                             +------------------+
                             |     Planning     |
                             | Search / Inspect |
                             | Coverage / Reuse |
                             +--------+---------+
                                      |
                                      v
                             +------------------+
                             | Evaluation / UI  |
                             | Replay / Metrics |
                             | Training Export  |
                             +------------------+
```

## Subsystems

### 1. World

The **World** subsystem represents terrain, elevation, land cover, structures, obstacles, weather, visibility, and coordinate systems.

It answers:

- What exists in the environment?
- What is visible from a location?
- What physical constraints affect flight, sensing, and mapping?

### 2. Sensing

The **Sensing** subsystem models or ingests drone observations from RGB, thermal, depth, IMU, GNSS, altimeter, and optional lidar-style sources, including timing, latency, noise, and field-of-view behavior.

It answers:

- What is the drone seeing right now?
- How reliable is the measurement?
- What part of the world is observable from this pose?

### 3. Localization

The **Localization** subsystem estimates the drone pose over time using GNSS, inertial, visual, and map-relative methods.

It answers:

- Where am I right now?
- How certain is that estimate?
- How does this pose align with previous maps and observations?

### 4. Mapping

The **Mapping** subsystem updates the internal world model using occupancy, elevation, semantic, obstacle, confidence, and temporal layers.

It answers:

- What does the environment look like?
- Which areas are well understood?
- What has changed since the last mission?

### 5. Indexing

The **Indexing** subsystem is ArgusNet's persistent memory layer for keyframes, landmarks, embeddings, observations, mission events, and retrieval.

It answers:

- Have I seen this place before?
- Where was this image or feature captured?
- Which areas still need more coverage?

### 6. Planning

The **Planning** subsystem turns the world model into action through search, frontier exploration, inspection routing, revisit planning, battery-aware updates, and next-best-view logic.

It answers:

- Where should the drone go next?
- Which area gives the most value if observed now?
- What path best improves coverage, map quality, or inspection completeness?

### 7. Evaluation

The **Evaluation** subsystem makes missions measurable and reproducible through replay, benchmarks, metrics, diagnostics, and export workflows.

It answers:

- How well did the mission perform?
- How accurate was localization?
- How complete or useful was the map?
- Which planning or sensing strategy worked best?

## Recommended Internal Data Flow

```text
Environment State
   ↓
Sensor Simulation / Ingestion
   ↓
Observation Processing
   ↓
Localization Update
   ↓
World Model Update
   ↓
Index + Memory Update
   ↓
Planning / Action Selection
   ↓
Mission Log / Replay / Evaluation
```

## Suggested Feature Priorities

### Phase 1 — World Model Core

- coordinate frames
- terrain and scene ingestion
- basic sensor simulation
- pose representation
- world model container
- mission logs and replay format

### Phase 2 — Localization Foundation

- GNSS + IMU fusion
- local odometry pipeline
- pose graph structure
- relocalization hooks
- keyframe generation

### Phase 3 — Mapping and Memory

- occupancy and elevation layers
- coverage map
- keyframe store
- landmark store
- observation index
- low-confidence region tracking

### Phase 4 — Planning

- frontier exploration
- inspection routing
- revisit planning
- next-best-view scoring
- battery-aware mission planning

### Phase 5 — Evaluation and Benchmarks

- replay viewer
- localization drift metrics
- map completeness metrics
- inspection coverage scoring
- dataset export
- benchmark scenarios

### Phase 6 — Controlled Real-World Adapters

- telemetry ingest
- real sensor sync
- field mission logging
- human-supervised execution tools

## Current Implementation Snapshot

The repository already includes working building blocks that map into this roadmap:

- Python simulation, terrain, weather, export, and replay tooling in `src/smart_tracker/`
- Rust tracking, gRPC service, and 3D viewer crates in `rust/`
- Architecture and subsystem documentation under `docs/`
- Automated Python and Rust test suites under `tests/` and the Cargo workspace

### Quickstart

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
cargo test
```

### Python tests

```bash
python3 -m pytest tests/ -q
```

## Documentation

- [Architecture & module map](docs/architecture.md)
- [Usage guide & CLI reference](docs/usage.md)
- [Changelog](docs/changelog.md)
- [Detailed safety guidance](docs/SAFETY.md)

## Non-Goals

ArgusNet does not aim to be:

- a weapon system
- an autonomous engagement platform
- a harmful surveillance toolkit
- a cyber intrusion or drone takeover framework
- a product for unlawful or unsafe deployment

The project focuses on world modeling, localization, mapping, inspection, coverage planning, and mission evaluation in simulation and controlled research settings.

## First Milestone Target

**Milestone 1: Persistent Mapping Demo**

A single drone flies a simulated mission over terrain and structures, collects camera and IMU/GNSS data, builds a basic world model with occupancy and elevation layers, stores keyframes and landmarks, and later relocalizes against that stored map during a second mission. The mission can be replayed visually and scored for localization drift, map completeness, and revisit success.
