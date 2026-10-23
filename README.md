# ArgusNet

**World models for civilian aerial mapping, inspection, localization and spatial memory.**

ArgusNet combines simulation, sensing, localization, mapping, planning and replay
so a mission can establish where a vehicle is, what it has observed, what remains
uncertain and what it should inspect next. Its intended uses include research,
infrastructure inspection, survey, environmental observation and search coverage.

## Current status

The Rust CLI `argusnet-rs sim` builds and runs scenarios with no Python runtime.
The Bevy viewer runs the same simulation on an in-process worker (`--run`),
with acknowledged Run/Pause/Step/Speed/Reset controls, and builds scene packages
in Rust. The Python `argusnet` CLI remains available as the simulation reference
and for offline GIS, analysis and export workflows.

Reviewed against checkout `79ca540` on **2026-10-23**. Opt-in scheduling supports
separate mission, mapping and fusion rates, per-sensor capture rates on tick
boundaries, and delayed measurement delivery. Full independent scheduling and
pending-job history remain incomplete. Real buildings, DEM terrain, OSM land
cover, mapped trees and power lines can feed the Rust run; the unified import
pipeline and canonical real-world scenario remain planned.
Merged #280 hides true trees and power lines in unknown-world views. Merged
#281 adds an offline client assurance package with a Rust report reader; the
Review workspace integration remains an open PR. See the dated
[recent PR snapshot](docs/migration/STATUS.md#recent-pr-snapshot-2026-10-23)
for merged versus proposed work.
[State ownership](docs/STATE_OWNERSHIP.md) and [known gaps](docs/KNOWN_GAPS.md)
distinguish implemented behavior from remaining qualification.

## Run the current system

Use Python 3.10+ and the repository-pinned Rust toolchain. Source installation
builds the native world/safety library; the viewer additionally needs your
platform's graphics/windowing dependencies. Run these commands from the repository
root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p runs
argusnet sim --map-preset small --duration-s 30 --seed 7 --replay runs/demo.json
argusnet build-scene --replay runs/demo.json --output runs/demo.smartscene
cargo run -p argusnet-viewer --release -- --scene runs/demo.smartscene --autoplay
```

The default simulation is a mapping/inspection mission with no targets; empty
fused-track output is expected. Evaluate coverage and inspection results. For
replay playback, the viewer opens a scene package built from a replay.
For an in-process run, pass `--run` a scenario config or bundle; no scene is required.
First compilation is separate from runtime startup performance.

See [usage](docs/usage.md) for existing modes, split-view controls, target-tracking
examples, ingestion and exports. Use `argusnet --help` or `argusnet sim --help`
for current options. [Live operations](docs/LIVE_OPERATIONS.md) describes the
existing live-enabled build and TLS connection workflow.

A Rust-only run from a checked-in scenario bundle needs no Python setup:

```bash
cargo run --release -p argusnet-cli -- sim --bundle tests/fixtures/scenario_bundle_scan.json --replay runs/rust-demo.json
cargo run --release -p argusnet-cli -- build-scene --replay runs/rust-demo.json --output runs/rust-demo.smartscene
cargo run --release -p argusnet-viewer -- --run tests/fixtures/scenario_bundle_scan.json
```

The viewer opens paused at tick 0. See [usage](docs/usage.md) for run options,
unknown-world mapping and real-data imports.

## The remaining Rust runtime target

Starting, running, controlling, viewing and recording simulations already have Rust
entrypoints; the remaining target is complete independent subsystem scheduling.
Python remains optional for offline analysis, experiment
orchestration, GIS and export workflows. Existing Rust domain crates are reused;
this is an incremental migration within this repository.

The [target runtime architecture](docs/RUST_RUNTIME_ARCHITECTURE.md) replaces an
all-subsystems tick with **independently scheduled work**:

- Physics and vehicle control follow fixed integration/control deadlines.
- Sensors sample at their own rates, preserving acquisition and delivery times.
- Estimation reacts to measurements and prediction/expiry deadlines.
- Mapping updates affected regions from accumulated evidence.
- Planning runs on relevant changes or deadlines, using current belief.
- Rendering, meshing and recording have independent bounded execution paths.

One session owns causal state. Modules communicate through typed local calls,
immutable revisioned results and explicit worker queues. gRPC serves external
boundaries. Deterministic simulation uses virtual time; hardware uses separately
qualified deadlines and clock conversion. Neither rendering nor faster playback
changes physical dt. See [ADR-009](docs/adr/009-independently-scheduled-runtime.md).

[Simulation fidelity](docs/SIMULATION_FIDELITY.md) keeps physical/sensor realism,
execution speed and visual quality separate. Paced simulation can report slower
achieved speed under load; hardware-coupled operation instead needs explicit
synchronization limits and qualified deadline responses.

## Viewer status and remaining work

The viewer has an Operate/Plan/Review shell, Perspective/Top-down/Section
projections, Operational/Reconstruction/Safety layer presets, inspectors and a
source-time timeline. Plan and Review show explicit unavailable reasons pending
their canonical mission, journal and evidence contracts. The existing
`real-world`, `scan-map` and `split` CLI view modes still work independently of
these shell controls.

The [viewer specification](docs/VIEWER_UI.md) remains the full target, including
complete freshness, recording health, durable history and qualified live-world
presentation. The [feature integration guide](docs/VIEWER_FEATURE_INTEGRATION.md)
maps workflows to backend contracts and explicit unavailable states.

## Delivery sequence

| Track | Current state / next outcome |
| --- | --- |
| P4.8–P4.9 | Rust production composition, reset/restore and statistical acceptance implemented; remaining qualification is tracked in migration status |
| P5.0 | In-process viewer worker and acknowledged basic controls implemented; complete remaining operator acceptance |
| P4.10a–f | Independent schedules, sensor timing, incremental work, bounded jobs and complete history; opt-in until qualified |
| Remaining P2/P5 | Visual system, history/seek, recording and coherent live-world presentation |
| P6 | Retire superseded execution paths after consumer and packaging acceptance |
| S3–S7 | Actual flight-software SIL, HIL, contained aircraft, qualified field envelope and coordinated fleet |

P IDs are migration packages; S IDs are autonomous-system gates; F1–F7 are
capability features. They are not interchangeable phase numbers. The
[delivery plan](docs/migration/DELIVERY_PLAN.md), [product roadmap](docs/ROADMAP.md)
and [autonomous system plan](docs/AUTONOMOUS_SYSTEM_PLAN.md) define their dependencies.

## Python boundary and remaining migration

The Rust CLI and viewer simulation require no Python runtime. Python still ships
the reference simulation, GIS import, evaluation/reporting, exports and test
orchestration. The installed `argusnet` command remains Python; `argusnet-rs` is
the Rust executable. Python source installation still builds `argusnet-native`,
and the Python tracking adapter can build/start `argusnetd`.

P6 will separate lightweight analysis packaging from native dependencies,
transfer production command ownership to Rust and retire migrated Python runtime
code after caller/compatibility checks. Optional offline Python tooling remains
part of the target; finishing the migration does not require rewriting it all.
See [Python ownership](docs/STATE_OWNERSHIP.md#remaining-python-code) and the
[P6 delivery contracts](docs/migration/DELIVERY_PLAN.md#8-phase-6--reduce-python-without-breaking-analysis).

## Performance and testing

Track startup, simulation CPU/throughput, task deadlines, sensor-to-estimate age,
render CPU/GPU time, memory, queues and recording separately. Speed improvements
must preserve accuracy, mission outcomes, safety and repeatability. Published
historical profiles are not measurements of the future scheduler.

Existing test and benchmark entrypoints:

```bash
cargo test --workspace --exclude argusnet-viewer
cargo test -p argusnet-viewer
python -m pytest tests/ -q
argusnet benchmark --suite fast --seeds 7 --output runs/benchmarks/fast
```

Install test/development dependencies with `python -m pip install -e ".[dev]"`. See
[local CI](docs/LOCAL_CI.md) for the complete existing checks and
[performance tracking](docs/PERFORMANCE_AND_BENCHMARKING.md) for measurement rules.
The [verification plan](docs/migration/VERIFICATION_AND_ROLLOUT.md) adds mixed-rate,
worker-order, delayed-data, full checkpoint, stale-result and overload acceptance.
Those planned gates do not imply the corresponding jobs already exist.

## Hardware integration

Live telemetry viewing is not aircraft control. Future adapters must qualify
sensor clocks, calibration, frames, capabilities, deadlines and failure behavior.
Stabilization, safety and contingencies remain onboard despite ground-link loss.
The Rust-only application goal does not require rewriting autopilot firmware.

[Hardware integration](docs/HARDWARE_INTEGRATION.md) defines the adapter contract
and SIL/HIL progression. No board, autopilot, nominal sensor rate or flight
readiness is selected or established by this documentation. Hardware profiles
require actual timing and fault evidence before promotion.

## Repository and documentation

| Location | Role |
| --- | --- |
| `rust/` | Domain libraries, assembled simulation runtime and CLI, server, replay and viewer |
| `src/argusnet/` | Python reference simulation, adapters and offline tooling |
| `proto/` | External service contracts; binding changes update both languages |
| `tests/` | Python/contract fixtures and evaluation checks; Rust crates also contain tests |
| `docs/` | Current behavior, accepted designs, delivery and evidence |

Start with [architecture](docs/architecture.md), [state ownership](docs/STATE_OWNERSHIP.md),
[usage](docs/usage.md), [contributing](CONTRIBUTING.md) and
[capability delivery](docs/CAPABILITY_DELIVERY.md). Domain contracts cover
[terrain](docs/TERRAIN.md), [fusion](docs/FUSION.md),
[localization](docs/LOCALIZATION.md), [mapping](docs/MAPPING.md),
[planning](docs/PLANNING.md), [mission execution](docs/MISSION_EXECUTION.md),
[safety](docs/SAFETY.md), [inspection](docs/INSPECTION.md),
[spatial memory](docs/INDEXING.md) and [scenarios](docs/SCENARIOS.md).
See [changelog](docs/changelog.md) for changes and [security](SECURITY.md) for
supported security practices.

## Scope

ArgusNet is intended for civilian mapping, inspection, localization, survey,
research and controlled testing. It is not intended for weaponization, harmful
pursuit, privacy-invasive monitoring, cyber intrusion or unsafe autonomous use.
Real-world operation requires the applicable permissions and qualified operating
envelope. Simulation results and a functioning viewer do not establish that
qualification.
