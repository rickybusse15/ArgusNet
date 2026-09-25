# Cross-tick state in `run_simulation`

> Scope: this document records compatibility-port behavior and its dated evidence.
> The final [independently scheduled Rust runtime](../RUST_RUNTIME_ARCHITECTURE.md)
> is delivered separately by P4.10 in [the delivery plan](DELIVERY_PLAN.md).
> Do not reinterpret stage-fixture results as full runtime or hardware acceptance.

P4.1 requires this before code moves: *"Identify every mutable
controller/cache and future-tick side effect before moving code."*

It matters for two separate reasons.

**Causal order.** [`ARCHITECTURE_AND_PROTOCOLS.md`](ARCHITECTURE_AND_PROTOCOLS.md)
§4 says the first Rust port *"MUST preserve the Python causal order, including
decisions that take effect on the next tick"* and must not silently "improve"
it. A next-tick effect is invisible in a single tick's stage trace — the stage
that writes it and the stage that reads it never run in the same tick — so it
cannot be found by the ordering comparison in
[`test_stage_trace_reference.py`](../../tests/test_stage_trace_reference.py). It
has to be found by reading state.

**Checkpoints.** §6 requires a checkpoint to persist all causal state and to
exclude only caches whose rebuild is *proven* behaviourally neutral. A port that
forgets one of these resumes into a subtly different run.

---

## Method — and the mistake it made first

`run_simulation` spans lines 4829–6131 of
[`sim.py`](../../src/argusnet/simulation/sim.py); the tick loop is 5273–6131.

The first pass enumerated locals bound before the loop and **rebound or
subscript-assigned** inside it. That found 26 items and felt complete.

It was not. It missed three of the most important pieces of state in the loop:

| missed | why |
|---|---|
| `rng` | never rebound; `np.random.Generator` advances its own position on every draw |
| `sensor_models` | never rebound; each `SensorModel` owns an OU `SensorBiasDrift` with `_current_bias` and its own generator, advanced by `step()` |
| `adaptive_drone_controllers` | never rebound; the loop calls `controller.update_from_frame(frame, batch)` every tick, and `LaunchableTrajectoryController` keeps `launched` / `launch_time_s` |

**The lesson generalises past this repo.** Searching for assignments finds state
that is *replaced*. It is blind to state that is *mutated through a method call
on a name that never changes* — and that second category is the more dangerous
one, because it is also invisible in a diff. It was caught here only by checking
a specific claim ("sensing keeps nothing across ticks") against the code instead
of asserting it, and that claim turned out to be false.

So the inventory below is in two parts, and a future re-run must do both passes.

### What the ordering test does and does not prove

For each name, every read and write position inside the loop was compared. If
every *read* precedes every *write*, nothing in that tick can observe that
tick's write, so the value can only land on a later tick. That is a proof.

The converse is not. A name with reads on both sides of a write may be a
same-tick effect, a next-tick effect, or both — `_scan_phase` is both. Those
were resolved by reading the code.

Line numbers are relative to the loop start. They will drift; treat them as
pointers, not citations.

---

## Strict next-tick effects

Every read precedes every write. A decision made in stage 9 is acted on in
stage 2 **of the following tick**, and nothing in between can see it.

| state | read at | written at | what it carries |
|---|---|---|---|
| `_frontier_trajectories` | 56, 59 | 455, 467, 485, 497 | per-platform frontier redirect |
| `_poi_trajectories` | 67, 70 | 559, 595, 599 | per-platform POI redirect |
| `_egress_trajectories` | 78, 81 | 603, 624 | per-platform return-to-home |

These are the concrete instances of what §4 means, and why it asks stage 9 to
*"record which future tick controller changes take effect"*.
[`argusnet-sim`](../../rust/argusnet-sim/src/stage.rs)'s `StageRecord::notes` is
where that record goes.

**The trap.** Stage 2 applies these overrides *before* stage 9 computes them. A
port that computed a redirect and applied it in the same tick would be one tick
"better" than the reference — and would diverge from every golden file, from the
replay fixtures, and from the mission timing the scenario suite is calibrated
against. That is precisely the silent improvement §4 forbids.

---

## Mixed: same-tick *and* next-tick

`_scan_phase`, the mission state machine.

- read at 56 (stage 2, before any write)
- written at 517, 526, 597, 622
- read again at 520, 529, 611, 669, 673 — after writes, same tick

A phase transition computed in stage 9 is visible to the rest of stage 9 and to
frame assembly, but the movement overrides in stage 2 act on the phase as it
stood at the **start** of the tick.

This is the most delicate item here. Modelling it as purely next-tick loses the
within-stage reads that drive event emission; modelling it as purely same-tick
makes movement react a tick early. `_phase_started_at_s` moves with it.

---

## Previous-value carries

Written near the end of the stage that reads them, so the value read is always
the prior tick's. Mechanically simple; still checkpoint state.

| state | effect of losing it on restore |
|---|---|
| `_battery_prev_positions` | first post-restore tick charges no battery segment |
| `_safety_previous_velocities` | first acceleration check has no baseline |
| `_vio_prev_positions`, `_vio_prev_velocities` | first synthetic IMU sample is wrong |
| `_prev_coverage_count_grid` | every already-covered cell reports as newly covered |

`_prev_coverage_count_grid` deserves a note. It feeds only the `newly_covered`
delta in `ScanMissionState`, which is *replay output* rather than an input to any
decision — so it looks like a reconstructible cache worth dropping from the
checkpoint. That would be wrong. Losing it does not change what the simulation
decides; it changes what the simulation **reports**, producing a replay that
claims the whole map was discovered in one tick. §6 requires an excluded cache
to be proven behaviourally neutral, and this one is not.

---

## Inventory: state held in rebound names

| state | kind |
|---|---|
| `_scan_phase`, `_phase_started_at_s` | mission FSM — mixed, see above |
| `_frontier_trajectories`, `_frontier_targets` | next-tick controller override |
| `_poi_trajectories`, `_drone_poi_assignments`, `_poi_rescored` | next-tick override + assignment memory |
| `_egress_trajectories` | next-tick override |
| `_battery_states`, `_battery_prev_positions`, `_battery_low_warned` | energy accounting; the warned-set suppresses repeat warnings |
| `_grid_localizer` | estimator internals — estimates, per-platform step counts, timeout set, RNG position |
| `_loc_estimates`, `_loc_timed_out`, `_team_loc_best`, `_team_loc_current` | localization results that gate mission phases |
| `_safety_monitors`, `_safety_previous_velocities` | per-platform monitor state and escalation |
| `_vio_prev_positions`, `_vio_prev_velocities` | VIO integration baseline |
| `_coop_wedge_index`, `_coordinator_elected` | lazily assigned search wedges; election is once-only |
| `_node_zone_membership` | prior membership; zone entry/exit events are edges, not levels |
| `_prev_coverage_count_grid` | coverage delta baseline — see note above |
| `_vio_estimates_per_frame` | **output accumulation**, not causal |
| `_frame_times_ms` | **diagnostic**, not causal |

## Inventory: state held in objects that are never rebound

The pass that matters more, and the one that is easy to skip.

| state | what it holds | why it is causal |
|---|---|---|
| `rng` | the run's main `np.random.Generator` | its stream position advances on every draw. §6 names random-stream state explicitly; a restore without it re-draws numbers the original run already consumed |
| `sensor_models` | per-sensor `SensorBiasDrift` — `_current_bias` plus its own generator | bias is an Ornstein–Uhlenbeck walk. Restoring without it snaps every sensor's bias back to zero, which is a visible discontinuity in bearing error, not a rounding difference |
| `adaptive_drone_controllers` | per-platform controller state, fed `update_from_frame` every tick | `LaunchableTrajectoryController` keeps `launched` and `launch_time_s`; a controller that forgets it launched returns to its station position |
| `frames`, `metrics_rows`, `per_track_errors`, `generation_*` counters | run-level accumulation for the summary and replay | **output**, not causal — but a restore that keeps them produces a doubled artifact, so they are reset-on-restore rather than persisted |

`argusnet-sim`'s `SimulationCheckpoint` currently carries the clock, the seed and
the pending command queue. ADR-006's per-key stream scheme is what keeps the
`rng` row cheap: because each stream is re-derived from `(seed, key, tick)`, no
generator's internal position needs storing. That is a direct benefit of the
seed-derivation decision, and it does **not** extend to `sensor_models` or the
controllers, which hold state that is not a function of the tick.

---

## Consequences for the remaining ports

- **P4.2 (localization).** `_grid_localizer` holds estimates, per-platform step
  counts and the timeout set. The Rust `GridLocalizer` models all three, and its
  `reset` clears them together for this reason.
- **P4.5 (sensing).** `build_observations` itself is a pure function of its
  arguments — no globals, no module cache. But it is handed `sensor_models`,
  and those are stateful. **The port must move the OU bias walk with the
  function, or sensing will silently lose its drift.** This is the correction
  described at the top; an earlier draft of this document claimed sensing had no
  cross-tick state at all.
- **P4.7 (planning and safety).** The three strict next-tick overrides, both
  safety dictionaries and the controllers are all in scope at once. This is the
  stage where timing is most likely to be got wrong and hardest to see.
- **P4.8 (compose tick).** Owns `_scan_phase`. The mixed reads above are the
  specification it has to satisfy.

## Where this goes stale

This is a snapshot of `sim.py` at this commit, and nothing enforces it. It is
deliberately not wired into CI: a check that a hand-written table matches a
heuristic parse would fail on formatting changes while still missing the thing
that matters — a *new* piece of cross-tick state, especially one held in an
object rather than a rebound name.

The durable guard is the port itself. Once a stage's state lives in Rust behind
`SimulationCheckpoint`, the compiler knows the fields exist and the
restore-reproduces-execution test in
[`tick_transaction.rs`](../../rust/argusnet-sim/tests/tick_transaction.rs)
exercises them. Until then, re-run **both** passes when touching the tick loop.
