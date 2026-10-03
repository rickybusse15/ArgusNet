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

## Not decided here

Rates other than the legacy rate, per-subsystem elapsed time, fork-join (B3)
and background jobs (B5) follow the extraction of the lanes they touch.
