# P4.8a: the mission machine, and three asymmetries it has to keep

> Scope: this document records compatibility-port behavior and its dated evidence.
> The final [independently scheduled Rust runtime](../RUST_RUNTIME_ARCHITECTURE.md)
> is delivered separately by P4.10 in [the delivery plan](DELIVERY_PLAN.md).
> Do not reinterpret stage-fixture results as full runtime or hardware acceptance.

**Status:** delivered 2026-09-16. First slice of P4.8.

P4.8 asks to *"replace Python orchestration with the P4.1 stages and explicit
mission state"* and to *"preserve scan/localize/inspect/egress transitions"*.
This slice does the mission-state half. The tick composition, the replay writer
and the direct-core/gRPC comparison are the remaining P4.8 work.

## There is no state machine to port, only an `if/elif` chain

In `run_simulation` the transitions are a chain over a `_scan_phase` **string**,
about 120 lines interleaved with POI assignment, battery handoff and trajectory
construction. Deciding to advance and doing the work of advancing are the same
code, which is why the transitions have never been testable on their own.

`MissionState::advance` decides only what the phase should be, and emits
`MissionEvent`s. Everything the reference does inline — rescoring POIs, clearing
POI redirects, routing drones home — is the caller's, driven by those events.
`MissionStage` is the stage-9 handler; the machine itself never sees a world.

## One transition per tick, and why that is a correctness property

The `elif` chain tests `_scan_phase` as it stood at the **top** of the tick, so
at most one branch runs, and a phase entered this tick does no work until the
next one. `ARCHITECTURE_AND_PROTOCOLS.md` §4:

> *"The first Rust port MUST preserve the Python causal order, including
> decisions that take effect on the next tick. It MUST NOT silently 'improve'
> that order."*

A machine that cascaded — scanning straight through to complete in one call when
every gate happens to be open — would finish missions several ticks early and
would look better while doing it. `at_most_one_transition_happens_per_tick`
drives exactly that input and requires one step.

## Three asymmetries, preserved and pinned

Each was found by reading the two branches against each other, and each is the
kind of thing a port "cleans up" without noticing it changed behaviour.

### 1. An empty fleet converges out of `Scanning` and is stuck in `Localizing`

```python
# scanning branch
loc_converged = (not mobile_states) or all(e.confidence >= threshold for e in ...)
# localizing branch
if mobile_states and all(e.confidence >= threshold for e in ...):
```

With no mobile platforms the first is `True` and the second is `False`. The same
emptiness means "converged" in one branch and "not converged" in the other, so a
run that loses its fleet while scanning proceeds to `Inspecting`, and one that
loses it while localizing never leaves. `an_empty_fleet_leaves_scanning_but_can_never_leave_localizing`
pins both halves, and mutants that "fix" the asymmetry in either direction are
caught.

### 2. A platform with no recorded home counts as having arrived

```python
if home is None:
    arrived += 1
    continue
```

So a fleet whose homes were never recorded completes egress immediately — but
only if it is non-empty, because the outer guard is
`if mobile_states and arrived >= len(mobile_states)`.

Note also that `arrived` is counted over a derived per-platform list while the
comparison is against `len(mobile_states)`. The port carries `mobile_count`
explicitly for that reason: inferring the fleet size from the distance list
would make a short list complete the mission, reporting a fleet home when one
of them was never looked at.
`a_short_home_distance_list_cannot_complete_egress` binds it.

### 3. The localization timeout latches, in any phase, and never clears

`_loc_timed_out` is raised from the **localization stage**, which runs every
tick regardless of phase, and nothing ever resets it. A run that converges
during scanning and skips `Localizing` entirely can still report
`localization_timed_out: true`.

## Gate directions

Getting one backwards changes when a mission advances and nothing crashes, so
each is a named constant with the reference's own comparison:

| gate | comparison | default |
|---|---|---|
| scan coverage | `coverage_fraction >= threshold` | 0.70 |
| gap fill | `gap_fraction < gap_fill_min_fraction` | 0.02 |
| localization confidence | `confidence >= threshold`, **all** platforms | 0.65 |
| egress arrival | `distance < radius` | 15.0 m |

The gap gate is the only strict one. All four are tested at the boundary and one
ULP either side of it.

## The one-shot POI rescore

`_poi_rescored` is set on first entry to `Inspecting` and **never cleared**, so
it is once per run rather than once per entry. No phase path re-enters
`Inspecting` today, which is exactly why the latch has to be preserved rather
than reasoned away — the test constructs the second entry rather than declaring
it untestable, so if a future phase ever loops back there is already an
assertion saying what should happen.

## Wiring: a shared handle, not a boxed stage

`Simulation::set_stage_handler` takes ownership of a `Box<dyn StageHandler>`,
so once installed the caller cannot reach the stage again. The tick still has
to push this tick's inputs in and pull this tick's events out, so `MissionStage`
hands out a `MissionLink` before it is installed.

Two failures are refused rather than absorbed:

- **An enabled stage with no inputs** is a wiring bug — stage 8 never ran, or
  never pushed. Reporting `NoWork` would make it look like a mission with
  nothing to do, so it raises `StageError::Invariant`, which faults the run and
  publishes no snapshot, per §4.
- **Inputs stamped for a different tick** are refused, because
  `phase_started_at_s` is taken from the inputs and stale ones would date a
  transition to the wrong tick.

A *disabled* machine — the reference's `mission_mode != "scan_map_inspect"` —
reports `Skipped`, which is a different trace entry from `NoWork`. "This run has
no mission" and "the mission had nothing to do" are different facts and a trace
that conflated them would hide one.

## Verification

- 21 unit tests on the machine, 6 integration tests driving it through the real
  tick transaction (`tests/mission_tick.rs`).
- **16 mutants, 16 caught.** Three survived the first pass and are the reason
  three tests exist that otherwise would not:

  | survivor | what was missing |
  |---|---|
  | `Complete` runs the egress test again | the terminal test fed *different* inputs after completing, so the mutant had nothing to transition on |
  | `Inspecting` leaves without all POIs complete | every test set `all_pois_complete` before checking `Egress`; none held `Inspecting` while POIs were pending |
  | arrivals counted against the list length | every test had `home_distances_m.len() == mobile_count` |

- `cargo test`/`clippy --all-targets -D warnings`/`fmt` clean.

## What is not here

`MissionEvent` names the work a transition requires; nothing yet performs it,
because the POI manager, the frontier planner and the trajectory builders are
ported but not yet composed into the tick. That composition is the next P4.8
slice, and it is what turns these events from a contract into behaviour.

`ScanMissionState` — the replay payload — also carries `newly_scanned_cells`,
`poi_statuses`, `team_localization` and `egress_progress`, which are assembled
from other subsystems rather than decided by the phase machine. They belong
with the replay writer.
