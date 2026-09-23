# Migration status: everything before P5.0

**Status:** opened 2026-09-23. The live index of what is left before Phase 5
(the in-process viewer runtime) can start. [DELIVERY_PLAN.md](DELIVERY_PLAN.md)
holds the acceptance contracts; this file holds the order and the state. Each
row links the slice doc that proves it. A row changes only when its slice's
verification table does.

States: **done** (merged), **local** (committed and locally verified but not pushed),
**open**, **blocked** (with the reason).

## Why P5.0 is not startable yet

P5.0 needs P4.9, a Rust CLI running supported scenarios with no Python runtime,
and P2.x, a viewer engine that presents the runtime's snapshots. At 2026-09-23
every tick stage has a Rust body, and stage 1 moves the fleet (P4.8d.3). Nothing
yet **composes** a scenario into a configured `Simulation`, drives the mission
side-effects that `MissionStage` emits as events, or runs from a command line.

## Track A — runtime (critical path)

| ID | Slice | State | Evidence |
| --- | --- | --- | --- |
| P4.1 | Command lifecycle and control checkpoints (#119) | done | `ec9c2f7` |
| P4.8d.1–2b | Motion substrate, world adapter, trajectory wrappers, FollowPath | local | [MOTION_SUBSTRATE.md](MOTION_SUBSTRATE.md) |
| P4.8d.3 | Controller stage; stage-state checkpoint contract | local | [CONTROLLER_STAGE.md](CONTROLLER_STAGE.md) |
| P4.8e.1 | ScenarioBundle v1, runtime assembly (stages 1, 3, 6, 7), replay frames, Python exporter; target-tracking parity | local | [SCENARIO_BUNDLE.md](SCENARIO_BUNDLE.md) |
| P4.8e.2 | Stage 6 senses as the reference does: nominal pose, tilted look direction, the world detection model (`p_d`, effective noise, dominant loss), sensing weather | local | [STAGE6_REFERENCE_SENSING.md](STAGE6_REFERENCE_SENSING.md) |
| P4.8e.3 | Mission driver (POI assignment, redirects, dwell, egress); stages 2, 8 and 9 assembled; the default `scan_map_inspect` run end to end | local | [MISSION_DRIVER.md](MISSION_DRIVER.md) |
| P4.8e.4 | Causal-state audit: every assembled stateful stage on the checkpoint contract; whole-run resume differential | local | [CAUSAL_STATE.md](CAUSAL_STATE.md) |
| P4.9a | `argusnet-rs` CLI runs a bundle and writes a replay, checkpoint/resume from the command line; `argusnet sim --export-bundle` adapter refuses what it cannot carry | local | [ARGUSNET_RS_CLI.md](ARGUSNET_RS_CLI.md) |
| P4.9a.2 | Additive replay enrichment: every frame payload and the scenario `meta` a Python replay carries, viewer-readable; deconfliction `hypot` fix | local | [REPLAY_ENRICHMENT.md](REPLAY_ENRICHMENT.md) |
| P4.9b.1 | Native builder, world half: `argusnet.scenario-config` → terrain, environment obstacles and patches, land cover → bundle `world` + world `presentation`; exact except heights (numpy `exp`, ≤1e-12 m) | local | [NATIVE_BUILDER.md](NATIVE_BUILDER.md) |
| P4.9b.2 | Native builder, scenario half: stations, targets and behaviours, drones and controllers, zones, POIs, weather, sensors; `argusnet-rs build` | open | [NATIVE_BUILDER.md](NATIVE_BUILDER.md) |
| P4.9c | Phase acceptance: canonical scenarios, multi-seed envelopes, startup/tick timing | open | — |

## Track B — viewer engine

| ID | Slice | State |
| --- | --- | --- |
| P2.0 | Snapshot → ECS adapter, ordered SystemSets, persistent presentation state | open |
| P2.1 | Terrain/belief WGSL material, `BeliefRenderSnapshot` | open (terrain material exists; belief does not) |
| P2.2 | Sensor frusta, zone volumes, trail ribbons | open (zones exist) |
| P2.3 | Raycast picking, hover, selection outline | open |
| P2.4 | Free-fly, follow/chase, focus, bookmarks, time scrubbing | open (orbit camera exists) |
| P2.5 | Mission-control shell, quality tiers | open |
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
