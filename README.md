# ArgusNet

**ArgusNet: world models for autonomous aerial mapping, inspection, localization, and spatial memory.**

ArgusNet is an open-source platform for building and using **world models** from drone observations. It is designed to support **search, mapping, inspection, localization, and structured data collection** in simulation and authorized real-world research environments.

Rather than treating a drone flight as a one-time stream of sensor data, ArgusNet treats each mission as part of a growing spatial understanding of the environment. The platform is intended to help answer four core questions:

- **Where am I?**
- **What does the world look like?**
- **What have I already seen?**
- **Where should I go next?**

ArgusNet combines environment modeling, sensing, localization, mapping, indexing, planning, and evaluation into one modular system. It is intended for research, education, infrastructure inspection, survey workflows, benchmarking, and controlled field testing.

## Overview

ArgusNet is built around the idea that an aerial system should not only sense the world, but also remember it, organize it, and use that memory to make better decisions over time.

The long-term goal is a simulation-first drone world-modeling platform that can:

- build structured representations of environments from aerial observations
- localize relative to both live sensor streams and prior world models
- index and retrieve previous observations, landmarks, and map regions
- plan search, mapping, and inspection actions based on uncertainty and coverage
- generate replayable, measurable mission outputs for training and evaluation

## Current Repository Status

This repository currently contains the foundations of that broader vision, including:

- Python-based simulation, terrain, replay, export, and mission tooling
- Rust sensor-fusion and filtering services with gRPC boundaries
- Environment-aware observation synthesis with terrain, obstacle, and line-of-sight reasoning
- Replay-driven visualization and scene packaging workflows
- CLI and test infrastructure for simulation and evaluation workflows

In other words: **the current codebase is a working prototype and foundation for the larger ArgusNet architecture described below**.

## Design Principles

- **World-model first** — the environment is a structured, queryable internal representation, not just a background map
- **Localization-aware** — the platform must continuously estimate where the drone is relative to both local motion and persistent maps
- **Memory and indexing** — observations should be stored, retrieved, compared, and reused across missions
- **Simulation-first** — all major capabilities should work in realistic simulation before field use
- **Modular architecture** — sensing, localization, mapping, planning, and evaluation should remain separable
- **Human-supervised use** — real-world deployment should remain controlled, reviewable, and safety-bounded

## Primary Use Cases

ArgusNet is intended to support:

- aerial mapping
- infrastructure and roof inspection
- construction progress capture
- environmental survey
- agricultural observation
- search coverage of unobserved regions
- repeated inspection over time
- synthetic dataset generation
- localization and world-model benchmarking

## Safety and Scope Boundary

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
- unsafe autonomous operation
- privacy-invasive monitoring
- cyber intrusion or takeover
- harmful pursuit workflows
- any deployment that violates applicable aviation, privacy, or local laws

Users are responsible for ensuring that any real-world use complies with all applicable laws, regulations, permissions, and safety requirements.

## System Vision

ArgusNet should support:

- simulated and imported environments
- terrain and obstacle-aware world modeling
- drone sensor simulation and sensor ingestion
- localization through GNSS, inertial, visual, and map-relative methods
- mapping through occupancy, elevation, semantic, and inspection layers
- persistent indexing of keyframes, landmarks, and observations
- search and coverage planning
- inspection route planning and repeat capture
- replay, benchmarking, and evaluation
- synthetic data generation for training perception and navigation systems

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

The **World** subsystem defines the environment in which the drone operates.

It should represent:

- terrain
- elevation
- land cover
- structures
- obstacles
- weather and visibility
- coordinate systems and geodesy
- static and dynamic environment layers

This subsystem answers:

- What exists in the environment?
- What is visible from a location?
- What physical constraints affect flight, sensing, and mapping?

### 2. Sensing

The **Sensing** subsystem models or ingests drone observations.

It should support:

- RGB cameras
- thermal cameras
- depth sensors
- IMU
- GNSS
- altimeters
- optional lidar or acoustic extensions
- sensor timing, noise, latency, and field-of-view constraints

This subsystem answers:

- What is the drone seeing right now?
- How reliable is the measurement?
- What part of the world is observable from this pose?

### 3. Localization

The **Localization** subsystem estimates the drone pose in space over time.

It should combine:

- GNSS
- IMU
- visual odometry
- visual-inertial odometry
- map-relative correction
- loop closure
- relocalization against prior missions
- coordinate frame transforms

This subsystem answers:

- Where am I right now?
- How certain is that estimate?
- How does this pose align with previous maps and observations?

### 4. Mapping

The **Mapping** subsystem updates the internal world model.

It should maintain multiple possible layers:

- occupancy map
- elevation map
- semantic map
- obstacle map
- inspection history layer
- confidence / uncertainty layer
- observed vs unobserved regions
- temporal change layer

This subsystem answers:

- What does the environment look like?
- Which areas are well understood?
- What has changed since the last mission?

### 5. Indexing

The **Indexing** subsystem is ArgusNet's persistent memory and retrieval layer.

It should store and retrieve:

- keyframes
- landmarks
- descriptors and embeddings
- geotagged observations
- map tiles
- mission events
- region metadata
- inspection findings
- time-linked site history

This subsystem answers:

- Have I seen this place before?
- Where was this image or feature captured?
- What observations exist for this structure or region?
- Which areas still need more coverage?

### 6. Planning

The **Planning** subsystem turns the world model into action.

It should support:

- search planning
- frontier exploration
- inspection route generation
- revisit planning
- battery-aware mission updates
- uncertainty reduction planning
- adaptive sensing decisions
- multi-pass coverage logic

This subsystem answers:

- Where should the drone go next?
- Which area gives the most value if observed now?
- What path best improves coverage, map quality, or inspection completeness?

### 7. Evaluation

The **Evaluation** subsystem makes the platform measurable and reproducible.

It should support:

- mission replay
- scenario comparison
- benchmark suites
- localization error metrics
- map completeness metrics
- coverage scoring
- inspection success metrics
- dataset export for downstream training

This subsystem answers:

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

## Target Module Tree

```text
argusnet/
├── README.md
├── LICENSE
├── SAFETY.md
├── LEGAL.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── configs/
│   ├── environments/
│   ├── sensors/
│   ├── missions/
│   ├── planners/
│   └── evaluation/
├── docs/
│   ├── architecture/
│   ├── design/
│   ├── examples/
│   └── api/
├── data/
│   ├── terrain/
│   ├── scenes/
│   ├── sample_missions/
│   └── benchmarks/
├── argusnet/
│   ├── __init__.py
│   ├── core/
│   ├── world/
│   ├── sensing/
│   ├── localization/
│   ├── mapping/
│   ├── indexing/
│   ├── planning/
│   ├── evaluation/
│   ├── ui/
│   ├── adapters/
│   └── cli/
├── tests/
└── examples/
```

## Suggested Feature Priorities

### Phase 1 — World Model Core

Build the minimum foundation:

- coordinate frames
- terrain and scene ingestion
- basic sensor simulation
- pose representation
- world model container
- mission logs and replay format

### Phase 2 — Localization Foundation

Add pose estimation and correction:

- GNSS + IMU fusion
- local odometry pipeline
- pose graph structure
- relocalization hooks
- keyframe generation

### Phase 3 — Mapping and Memory

Turn flights into persistent world knowledge:

- occupancy and elevation layers
- coverage map
- keyframe store
- landmark store
- observation index
- low-confidence region monitoring

### Phase 4 — Planning

Use the world model to drive action:

- frontier exploration
- inspection routing
- revisit planning
- next-best-view scoring
- battery-aware mission planning

### Phase 5 — Evaluation and Benchmarks

Make the platform measurable:

- replay viewer
- localization drift metrics
- map completeness metrics
- inspection coverage scoring
- dataset export
- benchmark scenarios

### Phase 6 — Controlled Real-World Adapters

Only after simulation is strong:

- telemetry ingest
- real sensor sync
- field mission logging
- human-supervised execution tools

## Non-Goals

ArgusNet does not aim to be:

- an unsafe autonomous operation platform
- a privacy-invasive monitoring toolkit
- a cyber intrusion or drone takeover framework
- a product for unlawful or unsafe deployment

The project focuses on world modeling, localization, mapping, inspection, coverage planning, and mission evaluation in simulation and controlled research settings.

## Suggested First Milestone

**Milestone 1: Persistent Mapping Demo**

A single drone flies a simulated mission over terrain and structures, collects camera and IMU/GNSS data, builds a basic world model with occupancy and elevation layers, stores keyframes and landmarks, and later relocalizes against that stored map during a second mission. The mission can be replayed visually and scored for localization drift, map completeness, and revisit success.

## Existing Documentation

For implementation details already present in this repository, see:

- [Architecture & module map](docs/architecture.md)
- [Usage guide & CLI reference](docs/usage.md)
- [Terrain notes](docs/TERRAIN.md)
- [Mapping contract](docs/MAPPING.md)
- [Localization contract](docs/LOCALIZATION.md)
- [Indexing contract](docs/INDEXING.md)
- [Inspection contract](docs/INSPECTION.md)
- [Mission execution contract](docs/MISSION_EXECUTION.md)
- [Planning notes](docs/PLANNING.md)
- [Safety notes](docs/SAFETY.md)
- [Benchmark scenarios](docs/SCENARIOS.md)
- [Performance and benchmarking standard](docs/PERFORMANCE_AND_BENCHMARKING.md)
- [Changelog](docs/changelog.md)
