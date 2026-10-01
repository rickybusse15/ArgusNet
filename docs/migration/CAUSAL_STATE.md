# P4.8e.4: every stage on the checkpoint contract

**Status:** delivered 2026-09-23, locally verified only (not pushed). Package ID
`P4.8e.4`. See [STATUS.md](STATUS.md).

P4.8d.3 gave `StageHandler` its state contract (`reset`, `checkpoint_state`,
`check_state`, `restore_state`) and put stage 1 on it. Every other stateful
stage still held state the checkpoint could not see. This slice closes that
audit for everything the scenario bundle assembles, and proves it with a
resume differential over whole assembled runs.

## The inventory

| Stage | State that decides a future tick | Checkpointed as |
| --- | --- | --- |
| 1 controllers + battery | controller memory, launch, engagement; energy ledger | P4.8d.3 |
| 2 overrides | POI, egress and frontier redirects in flight | the three maps, ordered |
| 3 deconfliction | none (configuration only) | nothing |
| 6 observations | each sensor's bias walk and its stream position | `SensorState` per sensor |
| 7 fusion | the whole tracking engine: tracks, IMM filters, lifecycle, health | `TrackingEngine`, serialised |
| 8 mapping + localization | coverage counts and aggregates, the localizer, last estimates, timeout latch | all four |
| 9 mission driver | the phase machine, POI manager, redirect map, rescore latch | all four |

Each stage's `check_state` validates against the run it is restored into,
**before anything is mutated**: the sensor table (stage 6), the tracker
configuration (stage 7), the coverage grid's area and resolution, and whether
localization is on (stage 8), and the POI identities (stage 9). The POIs, homes,
sensor table and world are scenario, rebuilt by assembly, and are checked
against rather than carried.

**`TickWorld` is not persisted, and does not need to be for assembled runs.**
Stage 1 rewrites every controlled platform's position, velocity and timestamp,
and stage 1's battery rewrites every battery fraction, before any other stage
reads them. Tracks are written by stage 7 before anything reads them. A run
with *driver-prescribed* platforms (outside a bundle) would need the world in
the checkpoint; that is recorded under *Not in this slice*.

## Serialisation

The engine and the other state types gained `serde` derives, which is
additive. It needed `nalgebra`'s `serde-serialize` feature (IMM filters are
nalgebra vectors and matrices), serde's `rc` feature (contributor lists behind
`Arc`), `rand_chacha`'s `serde` feature (bias-walk stream positions), and
`serde` promoted from a dev- to a real dependency of `argusnet-mapping` and
`argusnet-planning`. The engine's latest output frame and its dedup scratch map
are `#[serde(skip)]`: output and scratch, not state.

The engine keys tracks in `HashMap`s, and a restored map iterates in a
different order. That would matter if any output depended on iteration order.
The differential shows none does: every fused track matches bit for bit across
resumes.

## Verification

`rust/argusnet-sim/tests/assembled_resume.rs` flies a bundle straight through,
then again with a checkpoint that is serialised to JSON text and restored into
a **freshly assembled run first sent down a wrong history**. Every later tick
is compared bit for bit: node kinematics, health and battery; every fused
track; every observation's bearing and noise (the bias walk's output); and in
scan mode the phase and coverage. Nothing is statistical: the Rust run is
fully deterministic, so any difference is missing state.

| Test | Checkpoints |
| --- | --- |
| `a_target_tracking_run_resumes_exactly` | ticks 3, 12, 40, 70: fused tracks, engagement, bias walks, deconfliction |
| `a_scan_map_inspect_run_resumes_exactly_in_every_phase` | tick 10 and three ticks into every phase: scanning, localizing, inspecting (POIs assigned, redirects in flight), egress, complete |
| ...both, immediately after each restore | the run's own checkpoint reproduces every stage's state **field for field**, which sees a dropped field even when the scenario never makes it decisive |
| `a_checkpoint_for_a_different_scenario_is_refused_stage_by_stage` | a real checkpoint tampered four ways (a sensor the run lacks, a different tracker config, a different grid, different POIs): each refused by the right stage, the run unchanged |
| `a_reset_assembled_run_is_a_fresh_run` | reset mid-run, then equal to a fresh run on every tick |
| `mapping::tests::the_localizer_timeout_latches` | the timeout latch survives a checkpoint round trip |

### Mutation round

18 mutants: **18 caught**, after one strengthening pass. The first pass caught
12. The survivors were the four consistency checks (no refusal test existed)
and two fields no checkpoint in these scenarios made decisive: the rescore
latch and the localization timeout. The round-trip assertion and the latch
test close them.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1041 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests` | clean |
| `pytest tests/` | 1010 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

## Not in this slice

* **`TickWorld` persistence**, for runs whose platforms are prescribed by a
  driver rather than a controller (see above).
* **The durable event journal** P4.1 names. Checkpoints carry state, not
  history.
* **Stages 4 and 5** (safety, VIO) are not assembled yet. Each needs the
  contract when it is: safety holds per-drone monitor escalation, and VIO holds
  filter state.

## Next

**P4.9a**: the `argusnet-rs` CLI. It runs a bundle and writes a replay, with
checkpoint/resume available from the command line, plus the
`argusnet export-bundle` adapter.
