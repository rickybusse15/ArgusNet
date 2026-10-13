# B2.2 design note: putting `legacy_stages_v1` on the scheduler

**Status:** 2026-10-03. Design for the second B2 slice (P4.10a). Follows the
scheduler foundations (#160, `rust/argusnet-sim/src/scheduler/`). Decided on
the recommendation below, as the maintainer asked; open to reversal.

The plan's serial-equivalence gate asks that `scheduled_v1` with every lane at
the legacy rate and one worker reproduce `legacy_stages_v1`
([PARALLEL_TICK_PLAN.md](PARALLEL_TICK_PLAN.md) §7.3,
[AGENT_EXECUTION_PLAN.md](AGENT_EXECUTION_PLAN.md) B2). Sizing it found two
facts that decide how.

## 1. No order-preserving map from the eleven stages to the nine phases

Bit-exact equivalence needs each legacy stage mapped to a phase with the
legacy order kept (a monotone map). ADR-009's phase order is propagate →
commands → captures → deliveries → estimators → mapping → mission/planning →
control/safety → publication. The legacy order is commands (0) → controllers
and battery (1) → overrides (2) → deconfliction (3) → safety (4) → VIO (5) →
observations (6) → fusion (7) → mapping (8) → mission (9) → commit (10).

- Stage 0 (commands) runs before stage 1 (propagation); the phases put
  propagation first.
- Stages 2–4 (overrides, deconfliction, safety) act on this tick's positions
  before anything is sensed; the phases put control/safety after mission, and
  their outputs affect the *next* interval.
- Stage 5 (VIO, an estimator) runs before stage 6 (observations, captures);
  the phases put captures before estimators.

So any per-stage phase assignment reorders legacy work, and a reorder is a
behaviour change: not bit-exact, only statistically comparable.

## 2. Float time derived from nanoseconds is not the legacy timestamp

Legacy time is `tick as f64 * dt_s` (`Clock::timestamp_s`). The scheduler's
seconds are `nanos / 1e9`. They are equal for `dt_s` = 0.5 and 1.0, but at
`dt_s` = 0.1 they differ on 3 595 of the first 10 000 ticks
(`3 * 0.1 = 0.30000000000000004`, `300_000_000 / 1e9 = 0.3`). Every
seed key, truth lookup and replay timestamp hangs off that value.

## Recommendation (taken)

1. **B2.2: the legacy pipeline is one lane.** The serial executor schedules
   one compound task per legacy boundary, in the `Propagate` phase, from
   `Schedule::Period(dt_ns)`; the task runs the eleven stages in their legacy
   order and keeps the legacy float time (`tick * dt_s`), not `nanos / 1e9`.
   `dt_s` must be a whole number of nanoseconds, or configuration refuses the
   run. Serial equivalence is then bit-exact by construction, and what the gate
   proves is the plumbing: integer deadlines, boundary commands, epochs and
   reset through the queue.
2. **B2.3 onward: lanes come out one at a time**, each with its own evidence.
   A lane whose extraction keeps the legacy order (fusion after observations;
   mapping after fusion) stays bit-exact. A lane that has to move (VIO after
   captures; control/safety after mission) is a declared behaviour change of
   `scheduled_v1` only, judged by the statistical envelopes against
   `legacy_stages_v1`, never by widening them. `legacy_stages_v1` itself does
   not change.
3. **Float time under `scheduled_v1`** is `nanos / 1e9` once a lane leaves the
   compound task. Where a lane still reads legacy-keyed seeds or truth, it uses
   the legacy derivation until its own extraction says otherwise; mixing the two
   inside one lane is refused.

## B2.3: the first lanes out (done)

Fusion (7), mapping (8), mission (9) and publication (10) leave the compound
task. Their legacy order is the phases' order (estimators → mapping →
mission/planning → publication), so each runs where it always ran and the run
stays bit-exact. `PhasedLaneExecutor` (`scheduler/lanes.rs`) queues five tasks
per legacy boundary at its integer deadline:

| Lane (source) | Phase | Runs |
|---|---|---|
| compound (0) | `Propagate` | stage 0 and the boundary checks, then stages 1–6 |
| fusion (1) | `Estimators` | stage 7 |
| mapping (2) | `Mapping` | stage 8 |
| mission (3) | `MissionPlanning` | stage 9 |
| publication (4) | `Publication` | stage 10: commit |

`Simulation` runs a tick in parts — `begin_tick`, `run_through(stage)`,
`commit_tick` — and refuses parts out of order (`TickError::OutOfOrder`): a
stage run twice or skipped, a commit before stage 9, a tick from another epoch.
`step` is those three parts in sequence. A failed lane discards the rest of its
boundary; a reset applied by stage 0 commits at once and the executor follows
the run back to tick 0. Every lane still reads the legacy float time, so no
lane mixes the two derivations.

Stages 1–6 stay compound: moving any of them is a reorder (§1), and so a
declared behaviour change of `scheduled_v1`, judged statistically.

Evidence: `argusnet-rs sim --executor lanes` reproduces the legacy replays byte
for byte (`scripts/replay_hashes.py --executor lanes --check`, 14/14) and
passes `--lifecycle`.

## B2.4: the mission lane at its own rate

The first lane off the legacy rate, and the first `scheduled_v1` behaviour.
`PhasedLaneExecutor::with_mission_rate(schedule)` gives the mission lane
(stage 9) its own `Schedule`; every other lane still runs every boundary, and
without a rate nothing changes.

- **When it runs.** At the first legacy boundary at or after each of its
  deadlines. A deadline between two boundaries is served by the later one;
  several deadlines inside one boundary interval are served by one run. The
  next occurrence is computed from the schedule's `n`th deadline, never by
  accumulating, so the lane cannot drift (a million 10 Hz boundaries at 3 Hz:
  exactly 300 000 runs, each on the first boundary at or after its deadline).
- **When it does not.** The executor queues a `NotDue` part in the lane's
  phase, and `Simulation::skip_through` records the stage as `Skipped`, with
  the reason (`not due: lane at 1/1 Hz (scheduled_v1)`), in the trace. A stage
  is never left out silently. Its handler clears what the last run left as
  this tick's output (`StageHandler::not_due`): the driver's decisions and
  override commands, the bare mission stage's events and inputs. The default
  `not_due` **refuses**, so a stage that has not said what its per-tick output
  is faults the run rather than repeating stale output.
- **Elapsed time.** A stage that sat out `k` boundaries is given
  `(k + 1) * dt` as its `dt_s` on its next run. With `k = 0` this is the
  timestep, bit for bit. A quantity integrated per boundary is not credited
  from it in bulk: `not_due` is handed the tick as the stages before it left
  it, and `TickContext::boundary_dt_s` is the boundary's own interval.
  Inspect dwell is residence, so the driver credits it at every boundary,
  sat out or run, where each drone is then, to the POI it holds then; a drone
  that arrived or left between runs, or a POI assigned at the next run, is
  credited only for the boundaries it was actually there. Assignment and
  completion wait for the lane's run.
- **Time.** The lane runs at a legacy boundary, so it reads that boundary's
  legacy float time, like every other lane; no lane mixes the two derivations.
- **Checkpoint.** `SimulationCheckpoint.stage_lag` carries each stage's
  boundaries since it last ran (absent when empty, so legacy checkpoints are
  unchanged). The executor derives its next occurrence from the tick it starts
  at, so a resumed run serves the same boundaries as an uninterrupted one.
  The driver declares each lane's own rate on the simulation
  (`Simulation::declare_lane_rate`); only a declared stage may sit out a
  boundary, and the checkpoint carries the declared rates (`lane_rates`). A
  restore into a run that declared other rates, or none, is refused, as is
  lag for an undeclared stage or more lag than the checkpoint's tick. A
  checkpoint with rates or lag is format 5 (`FORMAT_VERSION_STAGE_LAG`), so a
  reader that predates the fields refuses it; every other checkpoint stays
  format 4, so legacy replays and their recorded `checkpoint_format` do not
  change; a scheduled run's replay records 5, and `version` names both. A run directory records the rate in `run.json`, so `--recover`
  continues the scheduled run; `--resume` needs the same `--mission-rate`.
- **Deadlines are used up by commits.** A boundary that does not commit
  (paused, failed, reset) consumes no occurrence: the executor moves the lane
  to its next occurrence only when the boundary commits.
  This is the baseline P4.10e builds on: there is still no work pending across
  a boundary.
- **Label.** `argusnet-rs sim --executor lanes --mission-rate NUM/DEN` records
  `meta.execution_model: "scheduled_v1"` and `meta.lane_rates` in the replay
  (additive; absent at the legacy rate).

Why mission first: its inputs are cumulative (the map and the fleet), so a
boundary it sits out loses no evidence. Fusion cannot sit one out without
breaking the replay's one-fusion-frame-per-tick invariant, and mapping would
drop evidence until P4.10b buffers its inputs.

Evidence: with no rate, `--executor lanes` reproduces the legacy replays byte
for byte (`scripts/replay_hashes.py --executor lanes --check`, 14/14) and
passes `--lifecycle`. Tests: `tests/lane_rates.rs` (deadlines, elapsed time,
the trace, resume equal to uninterrupted, refusal of an undeclared stage) and
`scheduler::lanes` (no drift, resume derivation, not-due parts).
`scripts/run_phase_acceptance.py --rust-arg ...` judges a `scheduled_v1` run
against the same envelopes, unwidened; the result at 1 Hz is in
[`../performance/b24-mission-rate-2026-10-12.md`](../performance/b24-mission-rate-2026-10-12.md).

## B2.5: the mapping lane at its own rate, inputs buffered

`--mapping-rate NUM/DEN` runs stage 8 at its own rate, like the mission lane
(both can be set). Mapping is the first lane whose inputs are *not*
cumulative: every boundary brings the platforms' footprints, zone entries and
the range scans of a run with no prior map. So a boundary it sits out loses
nothing:

- **Buffered.** `MappingStage::not_due` keeps the boundary: its tick, its
  timestamp, the mobile platforms' positions and the scans stage 6 handed
  over. The next run integrates every kept boundary in order, each at its own
  tick and timestamp, before its own: coverage counts, newly covered cells,
  zone inspection (entry, exit and coverage events, stamped with their own
  times) and the self-built map. At every boundary the lane runs, stage 8's
  state equals the every-boundary run's exactly
  (`mapping::tests::a_mapping_lane_at_its_own_rate_catches_up_exactly`).
- **Between runs** the rest of the tick reads the map as the lane last left
  it: the mission's coverage gate and the planners' belief see it late, by up
  to one lane period. Localization runs at the lane's rate, and its last
  estimates stay current (the mission reads them every tick). Per-tick
  outputs are cleared on a skipped boundary (coverage deltas, inspection
  events, fused scans), so a boundary that did not run reports none.
- **Bounded.** A mapping rate that would leave the lane idle for more than
  `MAX_BUFFERED_BOUNDARIES` (1024) ticks in a row is refused at
  configuration (`TimeError::BacklogTooLong`), since everything it keeps is
  held in memory and written into every checkpoint. 1024 ticks is over 17
  minutes at the 1 s canonical timestep. The mission lane buffers nothing and
  has no such cap.
- **Checkpointed.** The kept boundaries are part of stage 8's checkpoint
  (`pending`, absent when empty, so legacy checkpoints are unchanged), and a
  resumed run integrates them as the uninterrupted one does
  (`mapping::tests::pending_boundaries_survive_a_checkpoint`; the CLI recovers
  a crash whose last checkpoint holds a pending boundary to the uninterrupted
  journal byte for byte).
- **One executor.** `PhasedLaneExecutor` keeps a rate per lane
  (`RATED_LANES`: mapping and mission), and `LaneRates` carries them through
  `RunOptions`, the run directory, the worker and the viewer.
  `meta.lane_rates` records each rated lane by name.

Fusion stays every boundary: the replay holds one fusion frame per tick.

## Not decided here

Rates for the other lanes, `nanos / 1e9` time for a lane at its own rate,
per-sensor rates, fork-join (B3) and background jobs (B5) follow the
extraction of the lanes they touch.

## B2.6: the fusion lane at its own rate, observations buffered

`--fusion-rate NUM/DEN` runs stage 7 at its own rate, beside the mapping and
mission lanes (any of the three can be set). Fusion's inputs, like
mapping's, are not cumulative: every boundary brings that boundary's bearing
observations and platform states. So a boundary it sits out loses nothing:

- **Buffered.** `FusionStage::not_due` keeps the boundary: its platforms and
  the batch it would fuse, at the time it would fuse it. With timestamped
  delivery modelled, the boundary's observations go through the delivery
  network at once, at its own tick, so link faults applied tick by tick (delay
  overrides, source restarts) act on them as they would have, and what the
  network drops is reported on that boundary's own frame. The next run fuses
  every kept batch in order before its own. The tracker therefore ages and updates exactly as it
  would have every boundary: at every boundary the lane runs, the tracker's
  state and the tracks equal the every-boundary run's
  (`fusion::tests::a_fusion_lane_at_its_own_rate_catches_up_exactly`;
  `tests/fusion_rate.rs` on the canonical target-tracking scenario, without
  delivery, with it, and with source restarts and a link delay landing while
  boundaries are pending). Target metadata and safety events are not kept: the
  tracker only passes them through to its frame, so the skipped boundary
  publishes them itself.
- **Between runs** the rest of the tick reads the tracks the lane last fused,
  and a skipped boundary's frame shows them with this boundary's platforms
  and no fused observations. The frame of the boundary the lane runs at
  reports what it fused for the kept boundaries before its own. In a run
  without a prior world, pursuit follows the tracks late by up to one lane
  period.
- **Bounded and checkpointed** as for mapping: a rate that would leave the
  lane idle for more than 1024 ticks in a row is refused, and the kept
  boundaries are part of stage 7's checkpoint (`pending`, beside `engine`;
  absent when empty, so legacy checkpoints are unchanged). The CLI recovers a
  crash whose last checkpoint holds pending boundaries to the uninterrupted
  journal byte for byte, and the run directory records the rate
  (`fusion_rate`).
- **Rated lanes** are now fusion, mapping and mission (`RATED_LANES`).
  Publication stays at the legacy rate.

