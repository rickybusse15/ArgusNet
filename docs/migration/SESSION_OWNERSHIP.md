# A1.1: the shared stage links, and owned state to replace them

**Status:** design, 2026-10-03, against `main` at `8e62e3a`. Read-only: no code
changes. First package of A1 (production ownership) in the agent execution
plan; [RUST_RUNTIME_ARCHITECTURE.md](../RUST_RUNTIME_ARCHITECTURE.md) asks for
one session owner and no interior-mutable aliasing between stages and callers.
ADR-010 (proposed in #134) makes disjoint write sets a precondition for
parallel phases, which these links currently defeat.

## 1. Inventory

Ten stages expose an `Arc<Mutex<Shared>>` handle (`…Link`). Each is cloned
into other stages and into the scenario's `RunHandle`, so three kinds of code
reach into a stage's state: other stages during the tick, the assembly code
that builds the run, and observers after the tick (CLI, replay writer, tests).
The edges below are production code (`#[cfg(test)]` uses excluded), each read
at the cited line.

### 1.1 Stage → stage, during a tick

| From (stage) | To (link) | Call | Direction in the tick | Where |
| --- | --- | --- | --- | --- |
| 2 overrides | `MissionLink` | `phase()`: frontier open while scanning/localizing | **backward**: reads stage 9's result from the previous tick | `overrides.rs:340` |
| 4 safety | `BatteryLink` | `fraction(node)` | backward: stage 1's result this tick (1 runs before 4) | `safety.rs:369` |
| 4 safety | `FusionLink` | `submit_safety_events` | **forward**: consumed by stage 7 this tick | `safety.rs:455` |
| 6 observe | `FusionLink` | `submit_observations` | forward: stage 7 this tick | `observe.rs:1156` |
| 6 observe | `ControllerLink` | `observe(accepted)`: engagement feed | backward: takes effect at stage 1 next tick | `observe.rs:1165` |
| 8 mapping | `MissionLink` | `phase()` gate | backward: stage 9's previous result | `mapping.rs:684` |
| 9 mission driver | `BatteryLink` | `model()`, `remaining_wh()` | backward: stage 1 this tick | `mission_driver.rs:258, 390-398` |
| 9 mission driver | `MappingLink` | `coverage_fraction`, `coverage_counts`, `bounds`, `confidences`, `localizer_timed_out` | backward: stage 8 this tick | `mission_driver.rs` (`run`) |
| 9 mission driver | `OverrideLink` | `set_poi`, `clear_poi`, `set_egress`, `clear_all_egress` | backward: takes effect at stage 2 next tick | `mission_driver.rs:544-560` |
| 9 mission driver | `MissionLink` | `set_inputs`, `take_events`, `state`, `restore_state`, `reset` | own state, held through a link | `mission_driver.rs` |

### 1.2 Assembly (configuration before the first tick)

`scenario.rs::assemble` configures stages through their links:
`BatteryLink::{set_model, register}`, `ControllerLink::{install,
set_environment}`, `DeconflictLink::{set_roles, set_zones}`,
`SafetyLink::{set_enabled, set_role, set_terrain, use_battery, feed_fusion}`,
`ObserveLink::{install_sensor, set_terrain, set_visibility_oracle,
set_pose_source, feed_fusion, feed_controllers}`,
`MappingLink::{set_scan_mode, set_zones, set_cooperative_search,
enable_localization, watch_mission}`. Several of these *create* the edges in
§1.1 (`feed_fusion`, `feed_controllers`, `use_battery`, `watch_mission`).

### 1.3 Observers (after a tick)

The `RunHandle` keeps the links and reads them between ticks: the replay
writer and CLI read `FusionLink::frame`, `ObserveLink::{observations, tally,
generation_rejections}`, `MappingLink::{stats, inspection_events,
newly_covered_cells, team_localization, last_estimates}`,
`MissionLink::{phase, state, take_events}`,
`BatteryLink::{remaining_wh, take_events}`, `DeconflictLink::take_events`;
tests also read `ControllerLink::is_engaged`. `take_events` *drains* state, so
an observer is also a writer.

### 1.4 Findings

1. **Mission is the hub.** It is read by stages 2 and 8 and written by stage 9
   through `MissionLink`. Converting it first would touch three stages at
   once; it goes last.
2. **`VioLink` has no production user.** Stage 5 is not assembled from
   bundles (known: [SCENARIO_BUNDLE.md](SCENARIO_BUNDLE.md#not-in-this-slice), "Not in this slice"),
   so production runs execute `UnportedStage` there, which reports `NoWork`.
3. **A1.3 cannot land as specified.** "Reject a missing required stage" would
   reject every production run today, because of finding 2. Stage 5 needs
   either a real assembly or an explicit *declared-absent* handler that
   records `Skipped { reason }`, which `UnportedStage` must never stand in
   for. That choice belongs in A1.3, and it is the reason A1.3 has to follow
   this design.
4. **Two timings, both load-bearing.** Forward edges (4→7, 6→7) take effect
   in the same tick. Backward edges (6→1, 9→2, and the reads of stage 9's
   phase by 2 and 8) take effect on the next tick, because the consumer has
   already run. The trace-equality tests depend on exactly this, so the
   replacement must reproduce both.
5. **Observers drain.** `take_events` on the run handle mutates stage state
   between ticks. Under owned state, draining has to become an explicit
   session operation (events move into the ordered journal of A4) rather than
   a side effect of reading.

## 2. Target design

One owner, explicit data flow, the same timing.

1. **`Subsystems`, owned by `Simulation`.** Each stage's state moves out of
   its `Arc<Mutex<Shared>>` into a plain struct owned by the stage handler. No
   handle is cloned out.
2. **Configuration is construction.** §1.2's setters become stage constructor
   arguments or a builder consumed by `assemble`. Nothing configures a stage
   through a retained handle.
3. **Same-tick forward data travels in `TickWorld`.** A producing stage writes a
   typed slot on the tick's world (`observations`, `safety_events`), and a
   later stage reads it. The stage order makes the dependency visible and
   checkable, which is what P4.10's phase barriers need (ADR-010).
4. **Next-tick backward data is a typed inbox owned by the consumer.**
   Engagement observations (6→1) and override commands (9→2) are queued into
   the consumer's inbox and applied when the consumer runs, which is the next
   tick, exactly as now. Stage 9's phase reaches stages 2 and 8 the same way:
   committed state of the previous tick, read-only.
5. **Observers read a `SessionView`.** After a commit,
   `Simulation::view()` lends read-only access to each subsystem's published
   state (`frame()`, `stats()`, `phase()`). Draining events becomes a session
   call that moves them into the journal (finding 5).
6. **Checkpoints do not change.** Each stage's `checkpoint_state` payload stays
   byte-identical, so existing checkpoints still load.

## 2a. Amendment after slices 1–4 (2026-10-03)

Sizing slice 5 showed two edges that §2's mechanisms do not cover as written:

- **6→1 is not purely next-tick.** Stage 6 calls `ControllerLink::observe`
  *during* stage 6. The engagement it sets is visible at once: to
  `is_engaged` after the tick, to stage 6's trace note, and to the
  per-tick engagement comparison in `tests/scenario_bundle.rs`. Only its
  effect on motion waits for stage 1 of the next tick. A next-tick inbox
  (§2.4) would move the state change by a tick and break parity.
- **Stage 6 reads stage 1's controllers in the same tick.** `ControllerLink`
  is the observer's `SensorPoseSource` and VIO's `NominalTrajectory`, so
  stage 6 samples controller poses mid-tick.

Two ways to keep the timing exact without a shared handle:

1. **Session-mediated handoff.** After a stage runs, `Simulation` applies
   its typed output to the consumer stage it owns
   (`controllers.observe(&observe.accepted())`), between stages and in a
   fixed order. Timing is identical; the edge becomes explicit in
   `Simulation::step` rather than hidden in a link.
2. **Lent views in `TickContext`.** Stage 6 receives `&ControllerStage` (the
   pose source) as a borrowed view for its run. This needs a split borrow
   of the handler map, which is safe because stages run one at a time.

Recommendation: (1) for the writes (6→1, 9→2, 4→7, 6→7), (2) for the
same-tick reads (poses, battery fractions, mapping results, mission phase).
**Decided 2026-10-03 (maintainer): adopt the recommendation**, (1) for writes and (2) for same-tick reads. Slices 5–10 follow it.

## 3. Slices, in order

One link per PR, each with the stage's existing tests unchanged, plus a
trace- and replay-hash equality check over the canonical scenarios: the
before hashes are captured in the first slice and reused by every one after
it.

| # | Link | Why here |
| --- | --- | --- |
| 1 | `DeconflictLink` | Assembly and observers only: no stage-to-stage edge. Establishes the pattern and the hash harness. **Done:** `Simulation::stage`/`stage_mut` (typed access to an installed handler), builders for assembly, `take_events` through the session; `scripts/replay_hashes.py` shows 14/14 canonical replays byte-identical |
| 2 | `VioLink` | No production user (finding 2). **Done:** `VioStage` owns its filters; tests reach it through `Simulation::stage_mut` |
| 3 | `SafetyLink` | One forward edge (→ fusion) and one same-tick read (battery). **Done:** `SafetyStage` owns its monitors and configuration; assembly configures it before installing; `AssembledRun::safety()`. It still holds the battery and fusion links until slices 6 and 7 |
| 4 | `ObserveLink` | Forward edge (→ fusion) and the backward 6→1 inbox. **Done:** `ObserveStage` owns its sensor table, feeds and last batch; assembly configures it before installing; `AssembledRun::observe()`. The fusion and controller links it holds go with slices 5 and 6 |
| 5 | `ControllerLink` | Receives 6→1; assembly only otherwise |
| 6 | `FusionLink` | Receives 4→7 and 6→7 through `TickWorld` once 3 and 4 have moved |
| 7 | `BatteryLink` | Read by 4 and 9 (same tick); observers drain events |
| 8 | `OverrideLink` | Receives 9→2 |
| 9 | `MappingLink` | Read by 9; reads mission phase |
| 10 | `MissionLink` | The hub (finding 1): last, once every reader takes it as input |

This differs from the stage-order sequence in the agent execution plan's
A1.2: that order put the hub (mission) last but the receivers of backward
edges (controllers) before their senders. The order above follows the edges
found in §1.

Exit for A1 as a whole: no `Arc<Mutex<…>>` shared between a stage and anything
outside it (`grep -rn 'Arc<Mutex' rust/argusnet-sim/src` finds none in stage
modules); trace and replay hashes unchanged across the canonical scenarios;
stage checkpoints byte-identical.
