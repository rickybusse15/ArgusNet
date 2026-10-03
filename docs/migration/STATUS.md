# Migration status: everything before P5.0

**Status:** opened 2026-09-23. The live index of what is left before Phase 5
(the in-process viewer runtime) can start. [DELIVERY_PLAN.md](DELIVERY_PLAN.md)
holds the acceptance contracts; this file holds the order and the state. Each
row links the slice doc that proves it. A row changes only when its slice's
verification table does.

States: **done** (merged), **local** (committed and locally verified but not pushed),
**open**, **blocked** (with the reason). **done** rows cite the merging PR and
its merge commit on `main`.

## Why P5.0 is not startable yet

P5.0 needs P4.9, a Rust CLI running supported scenarios with no Python runtime,
and P2.x, a viewer engine that presents the runtime's snapshots. At 2026-09-23
`argusnet-rs sim --config` builds and runs every canonical scenario with no
Python, mission profiles and the blocking safety monitor included (P4.9d). Six
of the seven pass every declared envelope. One envelope,
`sar_person_search`'s `covariance_reduction_mean`, fails for a reason in the
metric's definition, which needs a decision; see
[SAFETY_AND_PROFILES.md](SAFETY_AND_PROFILES.md).

At 2026-09-25 the viewer track has P2.0 to P2.3 locally: the snapshot
adapter, the terrain and belief material, persistent sensor, zone and trail
geometry, and picking and selection. P2.4 to P2.6 (camera and time
controls, the mission-control shell and quality tiers, graphics acceptance)
are open, as are guardrails G0, G2 and G3.

At 2026-10-03 everything above that was *local* is merged: P4.8d.1 through
P4.9d and P2.0 to P2.3 reached `main` together in #122 (merge `badb1e1`).
P2.4a and P2.4b are merged (#128, #129). The `sar_person_search` covariance
metric was redefined (D-1, #137) and now passes; on Linux the scenario fails
its RMSE envelope by 0.11 m instead, a platform effect of the reference, so
whether acceptance fixes a reference platform or stores the reference is the
open decision ([PHASE_ACCEPTANCE.md](PHASE_ACCEPTANCE.md#after-d-1-the-covariance-metric-redefined-2026-10-03)).

## Baseline run (Wave 0.2)

`main` at `0b1032c`, 2026-10-03, in a Linux container (x86_64, glibc 2.39,
4 cores, Python 3.11, cargo 1.98), through `scripts/ci-local.sh`. GitHub CI
could not run: the self-hosted runner had taken no job since 2026-10-01.
A skip is not a pass.

| Job | Result | Detail |
| --- | --- | --- |
| python | fail → fixed | 1040 passed, 17 skipped, 1 failed: `test_run_simulation_has_no_stdout_side_effects`, because the daemon auto-build printed to stdout on a cold machine. Fixed in #139 |
| rust-fast | fail → fixed | With `--no-fail-fast`: 9 parity tests in 6 suites failed by 1 ULP of libm rounding against fixtures captured on macOS aarch64. Fixed in #138 ([PARITY_PORTABILITY.md](PARITY_PORTABILITY.md)). Without the flag cargo reported only the first; #139 adds it. 5 `argusnet-server` tests also failed when `<repo>/target` was absent (fixed in #139) |
| rust-lint | pass | `cargo fmt --check`, `clippy --workspace --all-targets -D warnings` |
| rust-viewer | pass | viewer tests, binary, headless render (10,596 colours, 259 green pixels) |
| GPU probes | pass | `belief_probes`, `geometry_probes`, `picking_probes` with `ARGUSNET_GPU_TESTS=1` on lavapipe: shader correctness, not frame rate |
| packaging | pass | wheel, clean venv, installed-package contract |
| security | **skipped** | pip-audit, bandit and cargo-audit not installed in the container |
| phase acceptance | 6/7 | see [PHASE_ACCEPTANCE.md](PHASE_ACCEPTANCE.md#after-d-1-the-covariance-metric-redefined-2026-10-03) (run with D-1, #137) |

## Track A — runtime (critical path)

| ID | Slice | State | Evidence |
| --- | --- | --- | --- |
| P4.1 | Command lifecycle and control checkpoints (#119) | done | `ec9c2f7` |
| P4.8d.1–2b | Motion substrate, world adapter, trajectory wrappers, FollowPath | done | #122 (`badb1e1`); [MOTION_SUBSTRATE.md](MOTION_SUBSTRATE.md) |
| P4.8d.3 | Controller stage; stage-state checkpoint contract | done | #122 (`badb1e1`); [CONTROLLER_STAGE.md](CONTROLLER_STAGE.md) |
| P4.8d.4 | World-backed stage 1 and a Python-vs-Rust stage-1 scenario parity run | done inside P4.8e.1 (#122): stage 1 runs on the owned `runtime_world`; `an_all_follow_scenario_matches_on_every_tick` (7 nodes × 80 ticks bit-exact) and `a_python_scenario_runs_in_the_rust_tick_and_matches_where_it_must` in `rust/argusnet-sim/tests/scenario_bundle.rs` | [SCENARIO_BUNDLE.md](SCENARIO_BUNDLE.md) |
| P4.8e.1 | ScenarioBundle v1, runtime assembly (stages 1, 3, 6, 7), replay frames, Python exporter; target-tracking parity | done | #122 (`badb1e1`); [SCENARIO_BUNDLE.md](SCENARIO_BUNDLE.md) |
| P4.8e.2 | Stage 6 senses as the reference does: nominal pose, tilted look direction, the world detection model (`p_d`, effective noise, dominant loss), sensing weather | done | #122 (`badb1e1`); [STAGE6_REFERENCE_SENSING.md](STAGE6_REFERENCE_SENSING.md) |
| P4.8e.3 | Mission driver (POI assignment, redirects, dwell, egress); stages 2, 8 and 9 assembled; the default `scan_map_inspect` run end to end | done | #122 (`badb1e1`); [MISSION_DRIVER.md](MISSION_DRIVER.md) |
| P4.8e.4 | Causal-state audit: every assembled stateful stage on the checkpoint contract; whole-run resume differential | done | #122 (`badb1e1`); [CAUSAL_STATE.md](CAUSAL_STATE.md) |
| P4.9a | `argusnet-rs` CLI runs a bundle and writes a replay, checkpoint/resume from the command line; `argusnet sim --export-bundle` adapter refuses what it cannot carry | done | #122 (`badb1e1`); [ARGUSNET_RS_CLI.md](ARGUSNET_RS_CLI.md) |
| P4.9a.2 | Additive replay enrichment: every frame payload and the scenario `meta` a Python replay carries, viewer-readable; deconfliction `hypot` fix | done | #122 (`badb1e1`); [REPLAY_ENRICHMENT.md](REPLAY_ENRICHMENT.md) |
| P4.9b.1 | Native builder, world half: `argusnet.scenario-config` → terrain, environment obstacles and patches, land cover → bundle `world` + world `presentation`; exact except heights (numpy `exp`, ≤1e-12 m) | done | #122 (`badb1e1`); [NATIVE_BUILDER.md](NATIVE_BUILDER.md) |
| P4.9b.2a | numpy `integers` (PCG64 half-word buffer, Lemire) and the eight target behaviours, bit for bit | done | #122 (`badb1e1`); [NATIVE_BUILDER.md](NATIVE_BUILDER.md) |
| P4.9b.2b | Native builder, scenario half: stations, targets, drones and controllers, sensors, zones, POIs, weather; whole bundles; `argusnet-rs build` and `sim --config` | done | #122 (`badb1e1`); [NATIVE_BUILDER.md](NATIVE_BUILDER.md) |
| P4.9c | Phase acceptance: 5/5 supported canonical scenarios within the declared envelopes, release timing gates, direct fusion parity; fixed an IMM mode-probability collapse (`argusnet-core`) and the RMSE metric's track pairing | done | #122 (`badb1e1`); [PHASE_ACCEPTANCE.md](PHASE_ACCEPTANCE.md) |
| P4.9d | Stage 4 (hard safety monitor) assembled from bundles, three reference-argument defects fixed, checkpointed; `safety_blocking` and mission profiles through exporter and native builder; all 7 canonical scenarios scored, 6 within every envelope (`sar_person_search` fails `covariance_reduction_mean`, analysed, not widened) | done | #122 (`badb1e1`); [SAFETY_AND_PROFILES.md](SAFETY_AND_PROFILES.md) |
| P4.10-pre | Per-stage wall time in the Rust tick (`--stage-timing`), profile script, first Rust profile of the canonical scenarios | done: timers #136 (`3a0ca89`); profile [rust-tick-profile-2026-10-03.md](../performance/rust-tick-profile-2026-10-03.md). Work counters and allocation counts open | [rust-tick-profile-2026-10-03.md](../performance/rust-tick-profile-2026-10-03.md) |
| P4.10c (first item) | Stage 9 searches for enclosed gaps only on the scanning gate, as the reference does, under one lock | done in #141: `persistent_long` tick 0.516 → 0.304 ms (stage 9 0.211 → 0.003 ms), replays byte-identical | [rust-tick-profile-2026-10-03.md](../performance/rust-tick-profile-2026-10-03.md) |
| A1.1 | Inventory of the ten shared stage links and the owned-state design (one owner, `TickWorld` for same-tick data, consumer inboxes for next-tick data, `SessionView` for observers); slice order from the dependency graph | done ([SESSION_OWNERSHIP.md](SESSION_OWNERSHIP.md)); A1.2 done: all ten links replaced (slices 1–10), 14/14 replays byte-identical; A1 exit met: no `Arc<Mutex>` in stage modules; A1.3 done: `Simulation::production` → `ProductionBuilder::build` refuses a missing required stage by name (core 1, 3, 6, 7; mission adds 2, 8, 9; `safety_blocking` adds 4; VIO not required), `Simulation::bare` is tests-only |

## Track B — viewer engine

| ID | Slice | State |
| --- | --- | --- |
| P2.0 | Snapshot → ECS adapter, ordered SystemSets, persistent presentation state: typed interned keys, epochs, allocations per frame 1,108 → ~5 | done (#122, `badb1e1`; [SNAPSHOT_ADAPTER.md](SNAPSHOT_ADAPTER.md)) |
| P2.1 | Terrain/belief WGSL material, `BeliefRenderSnapshot`, land-cover tint, tile uploads, GPU pixel probes; headless renders fail on shader errors | done (#122, `badb1e1`; [TERRAIN_BELIEF_MATERIAL.md](TERRAIN_BELIEF_MATERIAL.md)) |
| P2.2 | Sensor volumes gated by the runtime's look, draped radar rings, extruded zone walls, path-independent trail ribbons; no volume encloses the camera; GPU probes | done (#122, `badb1e1`; [PRESENTATION_GEOMETRY.md](PRESENTATION_GEOMETRY.md)) |
| P2.3 | Occlusion-aware mesh ray picking with drone pick proxies, viewport- and UI-aware; hover and selection rings with no through-terrain draw; Tab/Esc and an entity list; GPU click probes. Found and fixed: the terrain's inverted winding let picks through; blended trails made transparent order flip | done (#122, `badb1e1`; [PICKING_AND_SELECTION.md](PICKING_AND_SELECTION.md)) |
| P2.4a | Time: a source-time cursor (irregular timestamps play and blend as recorded), gaps and teleports not interpolated, loop, jump-to-event (`[` `]`), trimmed live history and gaps shown as unavailable | done (#128; [TIME_CONTROLS.md](TIME_CONTROLS.md)) |
| P2.4b | Camera: damped free-fly, follow/chase with loss to the last known position, `F` focus, `1`-`9` bookmarks as per-scene preferences, one remappable binding table gated on typing; 0 allocations a frame | done (#129; [CAMERA_CONTROLS.md](CAMERA_CONTROLS.md)) |
| P2.5a | Visual tokens (palette, spacing, type roles), bundled Arimo (OFL, D-2), status vocabulary, `Field<T>`; zero-radius flat visuals | done in this PR ([VIEWER_TOKENS.md](VIEWER_TOKENS.md)) |
| P2.5b–g | Shell, inspectors, timeline, projections, presets and tiers, failure states | open |
| P2.6 | Graphics acceptance across tiers | open |

## Track C — guardrails

| ID | Slice | State |
| --- | --- | --- |
| G0 | Semantic contract inventory and conversion | open (field parity done) |
| G1 | Proto regeneration and no-diff check | done |
| G2 | Renderer fixtures, CPU/GPU frame instrumentation baselines | open |
| G3 | Scanner triage, artifact and provenance checks | open |

## Decisions recorded here

**Bundle first, native builder second (P4.8e.1 before P4.9b).** Porting every
closure-built trajectory before Rust could run anything would delay the first
end-to-end run by weeks and leave no way to diff a scenario as a whole. A
versioned, data-only `ScenarioBundle` is the "documented preprocessed world
bundle" P4.3 already names. Python exports it from a built scenario and Rust
runs it. The Rust-native builder then has an exact parity target: it must
produce the same bundle.

**Target truth crosses as a sampled table in v1.** Every target trajectory in
the reference is a pure function of time: randomness is drawn at construction
or keyed on the timestamp. A table sampled at every time the tick asks for is
therefore exact, whatever order it is read in. Porting the behaviours is
P4.9b's work.

**Observation parity is statistical, kinematics are exact.** Stage 6 draws from
keyed ChaCha streams (ADR-006), not NumPy's, so accepted-observation sets
differ by design. Anything not fed by an observation is compared bit for bit:
stations, launch, lanes, truth-fed pursuit, coverage from positions. Anything
downstream of one (engagement, fusion, localization) is compared against
envelopes.
