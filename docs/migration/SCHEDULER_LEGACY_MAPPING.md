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
  `(k + 1) * dt` as its `dt_s` on its next run: mission's one use of it,
  inspect dwell, accumulates the time that actually passed. With `k = 0` this
  is the timestep, bit for bit.
- **Time.** The lane runs at a legacy boundary, so it reads that boundary's
  legacy float time, like every other lane; no lane mixes the two derivations.
- **Checkpoint.** `SimulationCheckpoint.stage_lag` carries each stage's
  boundaries since it last ran (absent when empty, so legacy checkpoints are
  unchanged). The executor derives its next occurrence from the tick it starts
  at, so a resumed run serves the same boundaries as an uninterrupted one.
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

## Not decided here

Rates for the other lanes, `nanos / 1e9` time for a lane at its own rate,
per-sensor rates, fork-join (B3) and background jobs (B5) follow the
extraction of the lanes they touch.
