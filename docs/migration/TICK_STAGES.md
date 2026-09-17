# P4.8c: the first tick stage bodies

**Status:** delivered 2026-09-17. The first stage bodies wired into the tick, after
P4.8a (the mission machine, stage 9) and P4.8b (`TickWorld` and the truth
capability).

§4 names stage 1 *"sample nominal controllers and account battery **using the
prior positions**. The ordering relative to overrides is a compatibility
constraint, not an accident."* This is the battery half; the controllers are a
separate slice.

The arithmetic is `argusnet_planning::battery`, ported in P4.7a and already
fixture-tested. What is new is the **accounting loop** around it, which the
reference keeps inline in `run_simulation` and which carries four behaviours
worth naming — plus one finding.

## The finding: nothing charges for hovering

`segment_cost_wh` is `travel_cost_wh(horizontal) + climb_cost_wh(dz)`, and both
derive their energy from a **duration** computed from a distance. A zero-length
segment has zero duration, so it costs **exactly zero**.

There is no hover term anywhere in the per-tick accounting. `hover_power_w` is
in the model, but it is used only as the **descent** power inside
`climb_cost_wh` (`hover_power_w * 0.8` on the `delta_z_m > 0` false branch).

So **a drone that stops moving has unlimited endurance in this simulation.**
That is preserved rather than corrected: charging hover would move every
existing golden run, and it belongs in a decision of its own rather than
arriving inside a port.
`a_hovering_platform_consumes_nothing_because_there_is_no_hover_term` pins it,
and asserts that the model does carry a hover power — it is simply never
charged, which is the part that would otherwise read as a porting slip.

## The first tick is free, and only the first

```python
prev_pos = _battery_prev_positions[ns.node_id]
if prev_pos is not None:
    cost_wh = _battery_model.segment_cost_wh(prev_pos, ns.position)
    ...
_battery_prev_positions[ns.node_id] = np.array(ns.position, dtype=float)
```

Energy is charged for the **segment between the previous position and this
one**, so on the first tick there is no previous position and nothing is drawn.

Two ways to get this wrong, both tested:

- **Seeding `prev_pos` with the spawn position** is indistinguishable on a
  drone that starts where it spawns — and silently wrong for one that does not.
- **Moving the assignment inside the `if`** makes the first *two* ticks free.
  `only_the_first_tick_is_free` catches it by requiring two equal segments to
  cost equally.

## Not every platform has a battery

The loop skips a platform that is not mobile *or* not in the ledger, and
`battery_fraction` for anything unregistered is the sentinel **-1.0** — the
reference's own comment reads *"0.0 (empty) → 1.0 (full); -1.0 = not
applicable"*. A ground station is not a drone at 0%.

Note the test for this has to **move** the fixed platform. A first attempt
parked it and asserted no draw, which passes whether or not the skip works,
because a zero-length segment costs nothing either way. The mutation run caught
that.

## The reserve is dynamic, with the static fraction as a floor

`dynamic_reserve_wh` estimates the energy to actually fly home from here, into
the wind, and takes `max` with `capacity * reserve_fraction` — so the estimate
can only make the reserve **more** conservative, never less. With no recorded
home there is no estimate to make and the floor is used directly.

The comparison is `remaining <= reserve`, not `<`. The two differ only when the
remaining energy lands exactly on the reserve, so
`a_battery_exactly_at_its_reserve_warns` constructs that: with
`reserve_fraction = 0.0` and no home the reserve is exactly `0.0`, and
`consume` clamps a drained battery to exactly `0.0`.

## The warning latches per platform

The reference guards on a `_battery_low_warned` set, so a drone below its
reserve warns **once per run**, not once per tick — and climbing back above
does not re-arm it, because nothing removes from that set.

`clear` does drop the latch, so a reset can warn again in the new epoch;
without that a fresh run would inherit the previous one's silence.

## Wiring

`BatteryLink`, the same shared-handle shape P4.8b established for
`MissionLink`: the stage is owned by the `Simulation` once installed, and the
caller still has to seed the fleet, read fractions and take events.

Events are cleared at the top of each run, so an event nobody took is **dropped
rather than replayed a tick late** — a caller that polls late gets the current
tick's events, never a backlog dated to an earlier one.

`Simulation::world_mut` is new: platform positions are prescribed by the
scenario rather than integrated by a stage (the reference evaluates
`scenario.node_states(timestamp_s)` at the top of each loop iteration), so
something outside the pipeline has to write them.

## `battery_fraction` does not reach the snapshot yet, deliberately

The ledger is the authority for energy, and nothing joins it onto
`NodeState` — which has no `battery_fraction` field, nor `sensor_type`,
`fov_half_angle_deg` or `max_range_m`. The replay DTO in `argusnet-replay`
*does* carry `battery_fraction` (Phase 0b closed that half), so the join is
real work that is simply not in this slice.

Widening `argusnet-types::NodeState` touches 31 construction sites across nine
files **and** the proto — and CLAUDE.md is explicit that a proto change means
updating the Python bindings and three Rust crates together. That is its own
slice, and pretending otherwise would have made this one a contract change
wearing a stage handler's clothes.

## Verification

- 19 unit tests, driving the stage through the real tick transaction.
- **12 mutants, 12 caught.** Four survived the first pass and are the reason
  four tests changed:

  | survivor | what was too weak |
  |---|---|
  | fixed platforms charged too | the test's tower never moved, so it cost nothing either way |
  | reserve trips on `<` not `<=` | no test put a battery exactly on its reserve |
  | `clear` forgets the warning latch | the test cleared a ledger that had never warned |
  | events not cleared between ticks | the test took events every tick, so nothing could go stale |

- `cargo test`/`clippy --all-targets -D warnings`/`fmt` clean.

---

# Stage 3: cooperative deconfliction

Second stage body of P4.8c. §4 places stage 3 **after** the mission overrides
of stage 2 and **before** the safety monitor of stage 4, and the reference says
why at the call site: the resolution *"runs after all controller and trajectory
overrides but before observation generation, so the tracker and replay see the
resolved positions."*

The resolution itself is `argusnet_planning::deconfliction::resolve_step`,
ported in P4.7g. This is the loop around it, which carries three behaviours the
reference states by construction rather than by comment.

## The write-back is guarded on the event list, not on the proposals changing

```python
if _step_deconfliction_events:
    node_states = [ ... _resolved[ns.node_id] ... ]
```

Today the two agree — `resolve_step` only moves a platform when it also emits
an event — so the guard is invisible. It stops being invisible the moment a
resolution is added that adjusts without reporting, and at that point the
reference **silently discards the adjustment**.

Preserved rather than simplified to "write back whatever came out": a port that
dropped the guard would apply an adjustment the reference does not, and no
existing test would notice. Reaching the case needs a stub resolver, so the
stage takes a `ConflictResolver` and `an_adjustment_without_an_event_is_discarded`
installs one that moves both platforms a kilometre while reporting nothing.

## Fewer than two mobile platforms is not a call at all

`resolve_step` also returns early below two proposals, which makes the
reference's `if len(_mobile_proposed) >= 2` look redundant. It is not redundant
to the **trace**: with one drone the reference never enters the block, so there
is no call to record. `the_resolver_is_not_called_with_fewer_than_two_proposals`
counts calls, which is the only way to see the difference.

## Only mobile platforms are proposed

A fixed node is never in the proposal map, so it can neither yield nor be
yielded to, and the write-back leaves anything absent from the resolved map
untouched — tested with a tower parked *between* two converging drones.

## Verification

- 14 unit tests driving the stage through the real tick transaction.
- **12 mutants, 12 caught** — but **five survived the first pass**, and the
  suite passing first time was the warning rather than the reassurance:

  | survivor | what was missing |
  |---|---|
  | write-back dropped the velocity | no test asserted a velocity change landed |
  | write-back dropped `z` | same, for altitude |
  | roles never reached the resolver | no test let role priority decide who yields |
  | the resolver was called with one proposal | no test counted calls |
  | events not cleared between ticks | `shared.events = events` masks it on the success path; only an **early return** can leave a stale list, and no test had one follow a tick that produced events |

  The roles gap is the sharpest: without roles the two platforms tie on
  `DEFAULT_ROLE_PRIORITY` and the tie-break — *the larger id yields* — still
  produces a plausible answer. The test now pins both the tie-break and the
  priority override, so a resolver that never sees the roles gets caught.

---

# Stage 4: the hard safety monitor

Third stage body of P4.8c. §4 places it **after** deconfliction, and the
reference is explicit about why it is opt-in:

> *"Hard safety floor runs after cooperative deconfliction. It is opt-in so
> legacy seeded simulations remain bit-identical."*

So a run without `safety_blocking` must not merely produce the same numbers —
the monitor must not run at all. `Skipped` says that; `NoWork` would claim it
ran and found nothing.

The policy is `safety_engine::simulation::SimulationSafetyMonitor`. This is the
loop around it.

## The role match is a substring, not an equality

```python
DronePhysicalLimits.interceptor_default()
if "interceptor" in role
else DronePhysicalLimits.tracker_default()
```

`"interceptor-lead"` and `"backup_interceptor"` both get the interceptor
envelope. The two envelopes differ — 42 m/s against 35 m/s top speed, and a
higher minimum AGL — so reading it as an equality silently gives part of the
fleet the wrong limits, with nothing to show for it. `limits_for_role` is a free
function so the rule has a name, and the test drives the same speed through both
roles to prove the choice reaches the monitor rather than only the table.

## The monitor is created once and kept

It carries the consecutive-violation count that escalates `warning` to `abort`.
Rebuilding it each tick would mean a platform could never escalate — and the
run would look calmer than it is.

## `previous_velocity` is the clamped velocity, not the proposal

```python
_safety_previous_velocities[ns.node_id] = decision.velocity.copy()
```

What is stored is what the monitor **allowed**. A platform commanded far past
its limit therefore shows a plausible acceleration next tick rather than an
enormous one, because the clamp already absorbed the difference. Storing the
proposal instead would manufacture an acceleration violation out of the clamp's
own work.

## The energy sentinel reaches the monitor

The reference passes `ns.battery_fraction` straight through, and that field is
**-1.0** for a platform with no battery in the ledger. `check` flags
`min_energy_reserve` for any fraction below the limit, so a **mobile** platform
with no battery would violate its energy reserve on every tick, for ever.

It does not bite today only because the reference registers a battery for every
mobile node at start-up. Preserved rather than papered over with a `None`, and
`a_mobile_platform_with_no_battery_violates_its_energy_reserve` records what it
costs. This is also the first place two stage bodies compose: stage 4 reads
stage 1's ledger through `SafetyLink::use_battery`.

## Verification

- 20 unit tests through the real tick transaction.
- **14 mutants, 13 caught. The 14th is equivalent, and that was established
  rather than assumed.**

  The mutant made the write-back conditional on the decision having violated
  something. Reading `check` against the clamp, every quantity the clamp
  touches — `min_agl`, `max_agl`, `max_speed`, `max_climb_rate`,
  `max_descent_rate` — is a constraint tested on the same condition, so a clamp
  that changes anything always coincides with a violation. A 200,000-state
  randomised probe over altitude, all three velocity components and terrain
  height found **zero** clamp-without-violation cases, which is the evidence
  for calling it equivalent rather than a test gap.

  Of the 13 caught, two needed tests that did not exist:

  | survivor | what was missing |
  |---|---|
  | validations not cleared between ticks | they are **pushed** and nothing reassigns the list, so only two consecutive violating ticks with nothing taken in between can see it |
  | (the equivalent one) | — |

---

# Stage 8: coverage, and the loop closing

Fourth stage body of P4.8c, and the first place two ported subsystems drive
each other.

## The reference keeps two identical coverage maps

`run_simulation` builds a top-level `CoverageMap` **and** a `WorldMap`, which
builds a second `CoverageMap` of its own. Both are marked from the same drone
positions, with the same radius, the same visibility predicate and the same
sample cap:

```python
_coverage_map.mark_circular(center_xy=(cx, cy), radius_m=_FOOTPRINT_RADIUS_M,
                            visibility_predicate=..., los_max_samples=...)
# ... and, inside WorldMap.add_scan_observation:
self._coverage.mark_circular(center_xy=(cx, cy), radius_m=footprint_radius_m,
                             visibility_predicate=..., los_max_samples=...)
```

Identical arguments to the same method on two instances. So
`MappingState.coverage_fraction` (read from the first) and the mission
machine's `scan_frac` (read from the second) are **the same number computed
twice**, every tick, for every drone — which is the duplicate rasterization the
delivery plan names.

The asymmetry worth keeping: the second map is only fed under
`mission_mode == "scan_map_inspect"`. In any other mode `WorldMap` stays empty
while the top-level map fills, so the two agree only in scan mode.

Here there is one grid, and `coverage_fraction` is read from it — so it cannot
drift from `stats()` the way the reference's two could.

## The footprint radius is derived, and the floor is the interesting part

```python
_target_cells = max(5, int(0.05 * nx * ny / _n_mobile))
_r_cells = max(1, int(ceil(sqrt(_target_cells / pi))))
_FOOTPRINT_RADIUS_M = max(50.0, float(_r_cells) * resolution_m)
```

Each drone is scaled to cover about 5% of the map per pass, then floored at
50 m. `footprint_radius_m` reproduces all three floors, and every one of them
is mutation-tested — including `or 1` on the fleet size, which is what stops a
fleet of zero dividing by zero.

The 50 m floor matters downstream: it breaks the integral `radius / resolution`
the rest of the configuration maintains, and that is exactly when P4.6a's two
rasterizations stop agreeing.
`the_two_samplings_disagree_at_a_non_integral_radius` pins that the choice is a
visible decision rather than a detail.

## A correction the mutation run forced

The multiplicity contract — a cell the lattice folds onto twice must be
incremented twice — had no test. Writing one, my first attempt parked a drone
near the corner on the assumption that out-of-range samples clamp onto the last
cell. **They do not: a sample past the bounds is skipped entirely.** The fold is
much narrower — only a sample landing *exactly* on the inclusive far edge
truncates to index `nx` and clamps back onto `nx - 1`, the cell that
`x_max - res` already hit.

The test now places the centre so the lattice lands on both `950` and exactly
`1000` on a grid whose `x_max` is 1000, and requires that cell to count **2**
from a single footprint while its neighbour counts 1.

## The loop

`tests/mapping_drives_mission.rs` flies a lawnmower pattern and requires the
mission machine's own coverage gate to trip — no test reaching in to fake the
fraction. Its negative is what makes it mean anything: a **stationary** drone
covers its own footprint and never reaches the threshold, so the positive test
cannot be passing because coverage climbs for some unrelated reason.

This is the first evidence that P4.6's coverage and P4.8a's mission machine
compose through the real tick transaction.

## What is not here

`BeliefStore` (P4.6d) is the second consumer those same rasterized hits are
meant to feed, and it is deliberately not wired: it lands with its own PR, and
stacking to reach one accessor would make a three-deep chain across two crates
for no reviewer's benefit. Nothing consumes belief yet — P2.1b is still
deferred — whereas the coverage fraction has a consumer today.

# Stage 8, second half: localization

The `GridLocalizer` ported in P4.2 is now driven by the same stage, after the
footprints are marked. It is opt-in — `MappingLink::enable_localization`
installs it — so a scenario that only wants coverage pays nothing.

Four things about the wiring are decisions rather than details, and each is
pinned by a test that fails if it is reversed.

## The phase gate reads the *previous* tick's phase

Localization runs only in `Scanning` or `Localizing`. Stage 8 runs **before**
stage 9, where the mission machine advances, so the phase it reads is the one
committed at the end of the last tick, never one this tick produced. That is
the correct reading — the phase is an input to this tick's work, not an output
of it — but it means a transition into `Egress` stops localization on the tick
*after* the transition, not the same one.
`localization_stops_outside_the_scanning_and_localizing_phases` and
`the_localizing_phase_still_localizes` pin both sides of the gate; dropping it
entirely is caught too.

## Below the speed floor the heading is zero, not the last heading

`atan2(vy, vx)` is only meaningful when the platform is actually moving, so
below `MIN_SPEED_FOR_HEADING_MPS` (0.1 m/s) the stage feeds the estimator
exactly `0.0`. The tempting alternative — hold the last heading — would make
the estimator's input depend on history the stage does not own.
`heading_is_zero_below_the_speed_floor_not_the_last_heading` requires the
zero, and `the_heading_speed_floor_is_strict` pins that the comparison is `>`
rather than `>=`.

## The estimator sees this tick's own footprint

Stage 8 marks coverage and *then* localizes, so the count grid the scorer reads
already includes the footprint this tick just stamped. This turned up while
writing the contract test below — an earlier draft captured
`coverage_fraction` before `step()` and disagreed. It is pinned deliberately
now rather than left to be rediscovered.

## The timeout budget counts localization steps, not ticks

`localizer_timed_out` latches once, and it latches from the localizer's own
step counter. A platform that never localizes — because it sits below the
coverage floor, so the stage returns `MapUnavailable` before a step is
recorded — therefore **never** times out, however many ticks pass.
`the_localizer_timeout_latches` and
`a_platform_below_the_coverage_floor_never_times_out` are two tests because
they are two different behaviours, and an earlier single test conflated them
and failed.

## The contract test

Two mutants survived the first pass — keying the particle stream on the
previous step, and passing the raw offsets as candidates instead of
`nominal + offset`. Both leave the estimate plausibly near the platform, so no
assertion about *where* the estimate lands can separate them.

`the_estimator_receives_the_step_scale_and_candidates_the_contract_specifies`
separates them by reconstructing the stage's inputs independently — a second
`GridLocalizer` driven by hand — and requiring an exact match on step, scale,
offsets and candidates. That is the test that took the localization half from
12/14 to 14/14, and it is also what pins the three properties above.

## Verification

- 15 unit tests plus 2 integration tests for coverage; 18 more for
  localization, through the real tick transaction.
- Coverage: **13 mutants, 13 caught** — one at compile time. The single
  first-pass survivor was the multiplicity contract above.
- Localization: **14 mutants, 14 caught**, from 7/14 on the first pass.

# Stage 2: mission overrides

§4 names stage 2 *"frontier, then POI, then egress overrides, in that
precedence"*. The reference is three near-identical blocks in a row, each
rebuilding the list the next one reads:

```python
if _frontier_trajectories and _scan_phase in ("scanning", "localizing"):
    for ns in node_states: ...          # rebuilds node_states
if _poi_trajectories:
    for ns in node_states: ...          # rebuilds it again
if _egress_trajectories:
    for ns in node_states: ...          # and again
```

So a platform named by two tables is flown by the **last** one to touch it.
Egress beats POI beats frontier, and that is a real decision rather than an
artefact: a drone recalled to base does not keep inspecting.

## This stage applies; it does not select

The reference sets `_frontier_trajectories` about 350 lines *later* in the same
tick, beside the mission transitions. A trajectory chosen on tick `n` therefore
first moves a platform on tick `n + 1` — exactly §4's *"decisions that take
effect on the next tick"*. Selection is stage 9's work and is deliberately not
in this slice.

Splitting them the way the reference already splits them buys something
concrete: the precedence is testable without a planner, a coverage grid or a
terrain sampler. A test installs three trajectories by hand and asks which one
flew the drone.

## The frontier gate reads the previous tick's phase

Only the frontier block is gated, on `scanning` or `localizing`. The reference's
own comment says why: *"Kept active through the localizing phase so drones hold
position rather than jumping back to their lawnmower trajectories."*

Stage 2 runs long before stage 9, so the phase it reads is the one committed at
the end of the previous tick — the same causal shape as stage 8's localization
gate, and `the_frontier_override_is_dropped_once_the_mission_leaves_localizing`
walks all three ticks of it: the tick the machine transitions on still flies the
override, because stage 2 ran first.

POI and egress are ungated in the reference and are ungated here.

**One case the reference does not have:** a link with no mission machine
watching it. The gate is left **open** there, because a run with no phases has
nothing to gate on. That is a decision, not a default, so it is pinned by
`an_unwatched_link_leaves_the_frontier_gate_open` — a future run that acquires
phases without acquiring a link fails there rather than silently flying its
drones through egress.

The gate suppresses the pass; it does not empty the table. The same trajectory
flies again when the phase comes back into range.

## Three things about `_make_goto_hover` that are easy to tidy away

- **Velocity is constant while in flight**, not scaled by progress: the whole
  displacement over the whole interval, so it reads as an average velocity.
- **On arrival, velocity is exactly zero and the position is exactly the
  target**, from a separate branch. Evaluating the lerp at `frac == 1` would
  land on the target too — and would keep reporting the flight velocity.
- **Before `t_start` the position clamps but the velocity does not.** The
  reference clamps `frac` into `[0, 1]` and then computes velocity from the
  interval alone, so an early sample reports the start position at full speed.
  Nothing samples early today. It is reproduced because it is what is written,
  and `sampling_before_the_start_holds_position_but_not_velocity` says so out
  loud rather than leaving a future reader to wonder whether it was a port bug.

`_make_redirect_goto` measures its distance in the **XY plane only**, so a
redirect that also climbs 500 m takes exactly as long as one that does not
climb at all, and floors the flight time at a millisecond so a drone standing on
its own target still gets a finite interval to divide by.

## What is not here

`_make_routed_goto` — the branch `_make_redirect_goto` takes when a route
planner is installed — is not ported in this slice. `PathPlanner2D` landed in
P4.7c and wiring it in is stage 9's business, next to the selection that decides
where the platform is going in the first place.

## Verification

- 21 unit tests, all through the real tick transaction.
- **29 mutants, 29 caught**, across two rounds.

The first round scored 19/19, which is a warning rather than a result: a suite
that catches everything first time is usually a suite whose mutants were chosen
to match it. The second round was written to be harder and found three
survivors — all three in places the first round's tests happened to agree on by
accident:

| survivor | why nothing caught it |
|---|---|
| `redirect` ignores its `speed_mps` and uses the constant | every test passed `REDIRECT_SPEED_MPS` |
| the zero-length-interval epsilon guard is dropped | reachable only by sampling a zero-length trajectory *before* it starts, which nothing did — the velocity becomes an infinity |
| `clear_frontier` reaches into the POI table | six near-identical accessors over three maps, and only the bulk clearer was tested |

# Stage 5: synthetic IMU, the VIO update, and the pose select

§4 names stage 5 *"truth/synthetic IMU features, VIO update, observation pose
select"*. The estimator itself landed in P4.4; this is the wiring around it —
and the wiring turned out to be where the interesting behaviour lives.

## The VIO integrates the nominal path, not the flown one

`node.trajectory(timestamp_s)` is the scenario's undisturbed path. Stages 2, 3,
4 and 10 have all run by the time stage 5 does, so the platform in the world may
have been redirected to a frontier cell, nudged by deconfliction and clamped by
the safety monitor — and the VIO sees **none of it**.

That is load-bearing. It is what makes the estimate drift away from the world
position in a way the fusion stage can see; reading the world here instead would
make the estimate track the truth it is supposed to be independent of.

So the nominal path arrives through an injected `NominalTrajectory` rather than
from `TickContext::world`. §3's truth capability covers *target* truth, which is
a different thing — a platform's own prescribed path is not in
`TickContext::truth` at all. With no source installed the stage reports
`Skipped` rather than quietly falling back to the world, because the fallback
would be a different estimator.

The test compares two runs, identical but for a world position thrown a
kilometre off course every tick, and requires bit-identical estimates. Comparing
runs rather than asserting a range means it does not have to know where the
estimate ought to land — only that the world cannot reach it.

## Parity: value-for-value against the reference, within 1 ULP

Six ticks are pinned against values taken from running `EKFVIO` itself, driven
with the inputs `sim.py` gives it. The worst disagreement across those ticks is
**1.42e-14 m**, which at a state magnitude of 80 m is exactly one ULP, and comes
from numpy and nalgebra ordering the covariance products differently.

`PARITY_TOLERANCE_M = 1e-9` is ~70,000x that, and the fixture asserts the
measured worst stays four orders of magnitude *below* the tolerance — so if a
future change starts eating the headroom, the fixture says so rather than
absorbing it. The covariance trace agrees exactly and is pinned exactly; a
tolerance there would hide a real divergence.

A correction worth recording: the first reading of this was "bit-for-bit
identical", taken from output rounded to six decimals. It is not. Full-precision
comparison is what turned a claim into a measurement.

## ⚠ A defect in the reference, reproduced deliberately

**The synthetic IMU feeds kinematic acceleration where the filter expects
specific force, so a level cruise makes the VIO estimate free-fall.**

The reference synthesises `accel = (v - v_prev) / dt` — the derivative of
velocity. A real accelerometer measures *specific force*, which at rest reads
`+g` along up, not zero. `_predict` then computes `R @ accel_body -
_GRAVITY_WORLD`, so on a straight, level, constant-velocity path `accel_body` is
zero and the estimate accelerates downward at 9.80665 m/s².

Measured, on a drone cruising east at 10 m/s from 80 m:

| tick | estimated z (m) | estimated vz (m/s) |
|---|---|---|
| 0 | 80.000 | 0.000 |
| 1 | 75.097 | −9.807 |
| 2 | 60.387 | −19.613 |
| 3 | 35.870 | −29.420 |
| 4 | 1.547 | −39.227 |
| 5 | **−42.583** | −49.033 |

The estimate is through the terrain datum inside five ticks and still
accelerating. Verified by running `EKFVIO` directly with the same inputs — this
is the reference's behaviour, not a port artefact — and carried across because
§4 forbids silently "improving" the port.

It matters because `use_vio_position` substitutes this position into the state
the tracker sees. **This needs a decision** and is flagged rather than fixed: the
candidate fix is to synthesise specific force (`a_kinematic + g` in the body
frame), which would change every `use_vio_position` run and is not this slice's
call to make.

## Three dead stores, and one that took measuring to find

- `_vio_prev_positions` is written every tick and never read.
- `_vio_prev_velocities.get(node_id, np.zeros(3))` names a zeros default that
  cannot fire — the setup loop that creates a backend fills both maps.
- The setup loop's seeding of `_vio_prev_velocities` from `vel0` has **no
  observable effect either**. The only tick that would consume it is the one at
  `t = 0`, where the filter was also initialised: its prediction interval is
  empty and its IMU sample is skipped as not newer than the clock. By tick 2 the
  value has been overwritten with tick 1's nominal velocity either way.

The third was not read off the source. A first attempt asserted that the seed
*avoided a startup spike*; it failed, and running the reference with the seed and
with zeros produced identical output at every tick. The test now records the
finding instead of asserting a false property. The seeding is still reproduced —
`VioLink::track` mirrors the reference's signature, and initialising the filter
at a non-zero timestamp would make it observable again.

## Two orderings that are easy to get wrong

**`classify` and `dt_total` are taken before `predict_to`**, because
`predict_to` advances the filter's clock. Taking `dt_total` afterwards makes it
identically zero, which zeroes the measurement Jacobian and silences the update
every tick — leaving prediction's estimate standing while still reporting
`Applied`.

**The feature store is refreshed after the update, not before.** Refreshing
first makes every frame match against itself, so the measured flow is zero by
construction rather than by the features happening not to move.

## The out-of-order path, and how a reset reaches it

`Simulation::reset` rewinds the clock and clears the world, but the stage's
filters live behind the link and keep the clock they reached. So a reset run
drops **every** VIO frame until the sim clock catches back up — and per finding 3
in `VIO_CONTRACT.md` the feature store is deliberately left stale while it does.

Whoever wires reset has to re-`track` on the new epoch.
`a_reset_leaves_the_filters_ahead_of_the_clock_and_every_frame_is_dropped` is
where that requirement is written down, and it is also what makes the
otherwise-unreachable drop branch testable.

## Verification

- 22 unit tests, all through the real tick transaction.
- **23 mutants, 22 caught, 1 proven equivalent.**

The first round scored 19/23. Of the four survivors, one was a badly written
mutant of mine — it computed a per-frame offset and then discarded it, so it
varied nothing. The other three were real gaps:

| survivor | why nothing caught it |
|---|---|
| the synthetic acceleration is not divided by `dt` | every test ran at `dt_s = 1.0`, where dividing by the interval is the identity |
| platforms are visited in insertion order | only one platform was ever in the `dropped` list, so its order could not differ |
| `classify` taken after `predict_to` | equivalent — see below |

The `dt` gap is closed on the **vertical** axis, because the optical-flow update
touches only `vx` and `vy` — its Jacobian has entries at state indices 3 and 4
and nowhere else — so `vz` carries the IMU's integral unopposed and a wrong
scaling shows there.

The last survivor is **proven equivalent, not assumed**. Moving `classify` after
`predict_to` looks like it should break the out-of-order drop, and does not,
because calling `predict_to` twice with the same target timestamp is a no-op the
second time: every IMU sample is skipped as not newer than the clock, and the
tail interval is empty. That property is not obvious from the signature, so
rather than rest on the reading, a test drives 5,000 randomised states through
both orderings and requires bit-identical state, covariance and clock. If a
future change breaks the idempotence, that test fails and the mutant stops being
equivalent.

# Stage 7: the fusion update, in process

§4 names stage 7 *"fusion update, in-process rather than over gRPC"*, and that
is the whole change in shape. The reference's `_ingest_stream.ingest_frame` puts
a request on a queue and blocks on `next(self._responses)` — a synchronous gRPC
round-trip on the sim thread, every tick. The stage calls
`TrackingEngine::ingest_frame` directly instead.

Worth restating from the Phase 0 profile, because it corrects the obvious
assumption: that round-trip measured **1.73 ms/frame, 5.9% of a representative
run**. Removing it is a minor throughput win. The reason to remove it is that it
is what stands between the world and an in-process viewer, not the microseconds.

## ⚠ Truth: fusion does not use it, but the tracker's metrics do

`ingest_frame` takes a `truths` argument, and `Stage::UpdateFusion` is not one
of the two stages `Stage::may_read_truth` admits. That is not an oversight —
inside `ingest_frame`, `truths` reaches exactly two places:

- `build_metrics`, which matches each track against the truth of the same id and
  computes a position error, and
- the returned `PlatformFrame`, as a passthrough for the replay.

**Nothing in the estimator sees it.** Fusion is *measured* against truth, not
*informed* by it — and `the_fused_tracks_are_identical_with_and_without_truth`
proves that rather than taking it on trust: two engines, one given the truth of
every target and one given none, fused bit-identically over eight ticks.

So the stage passes an empty truth vector, and the truth-referenced metrics are
not computed here. **That leaves a real gap:** under the capability list as it
stands, *no* stage can compute them — stage 11 (`CommitSnapshot`), where the
replay is assembled, is not admitted either.

The gap is left open rather than closed by widening the list. Handing fusion a
truth handle so it can compute an error metric would also let it fuse with
truth, and nothing structural would stop that — which is the one thing §3's
capability model exists to prevent. Separating measurement from estimation is a
design decision and it is **flagged, not taken**.

## The tracker runs on every tick, including barren ones

The reference calls `ingest_frame` unconditionally. That is load-bearing: the
track lifecycle ages toward `Coasting` and `Lost` on the strength of *ticks
passing*, so a stage that skipped a tick with no observations would keep a stale
track alive forever. `an_unobserved_track_ages_out_over_barren_ticks` watches
a real track disappear over 40 empty ticks rather than asserting it from the
tracker's documentation.

## The inbox is taken, not copied

A pending batch left in place would be re-fused next tick, and the tracker would
treat the same bearing as independent evidence twice — which is how a
triangulation gets falsely confident. Each `submit_*` also *replaces* rather
than appends, for the same reason.

## Verification

- 12 unit tests, all through the real tick transaction.
- **18 mutants, 16 caught, 2 proven equivalent.**

Both survivors are equivalent for reasons worth writing down, and both are now
pinned by a test that fails if the equivalence stops holding:

**"the node states are written back from the frame."** An earlier version of the
module comment said writing them back "can only lose". That is an overclaim, and
the mutation run is what exposed it. The tracker stores what it is handed and
never modifies it, and this stage hands it the world's own platforms, so the
round trip is an **identity** — the write-back would change nothing. It is
skipped because it is redundant, not because it is destructive.
`the_frame_round_trips_the_worlds_platforms` pins that identity field by field;
if the tracker ever starts editing node states, that test fails and the decision
has to be made again on new evidence.

**"truth is forwarded after all."** A mutant that reaches for `context.truth`
gets `None`, because §3 hands the capability out by stage and stage 7 is not on
the list. The emptiness is enforced by the capability model rather than by this
stage remembering to pass `Vec::new()` — the stronger guarantee, and the one
`stage_seven_cannot_reach_truth_at_all` pins, along with the two stages that
*are* admitted, so a future widening of the list has to come past an assertion.

# Stage 6a: generating the observation batch

§4 names stage 6 *"generate the observation batch, preserving rejection
stages"*. The cascade landed in P4.5b and the error model in P4.5a; this is the
loop around them.

## Stage 6 was not blocked on the proto change, and a first reading that said so
## is corrected here

The reference reads `max_range_m`, `fov_half_angle_deg`, `dropout_probability`
and `sensor_type` off its scenario node. `argusnet_types::NodeState` carries
none of those, and I initially read that as blocking stage 6 behind a proto
change that has to move together with the Python bindings, `argusnet-proto`,
`argusnet-core` and `argusnet-server`.

That widening is still wanted — the viewer cannot draw a sensor cone it cannot
see — but it is a **replay and transport** concern. `PairInput` already takes
`max_range_m` and the FOV as plain values, so this stage takes a `SensorSpec`
table the same way stage 5 takes its nominal trajectory. Nothing about
generating observations needs the field on the wire.

Line of sight is injected the same way, as a `VisibilityOracle`. The two-level
DDA lives in `argusnet-world`, and wiring it in would pull terrain, obstacles
and land cover into this crate's graph for one call; the cascade already models
visibility as an *input* rather than something it computes, so honouring that
boundary costs nothing.

**The default oracle sees through terrain**, and that is dangerous to leave
implicit — acceptance rates from such a run mean something quite different.
`ObserveLink::visibility_is_modelled` reports whether a real oracle was
installed, so a caller cannot mistake one for the other.

## The per-pair stream is opened after the draw-free gates

```python
# Per-pair RNG streams are seeded independently, so deferring
# construction past the draw-free rejects above is draw-order safe.
```

Range and FOV consume no draws, so where the stream is opened cannot change any
value — the reference says so itself. It is reproduced anyway, because "this is
safe *because* the streams are independent" is exactly the kind of reasoning
that stops being true when someone later threads one generator through the loop.

The key is `(node_id, target_id, round(t * 1000))`, which is what makes the batch
independent of the order pairs are visited in — ADR-006's central property.

## What 6a deferred — and one it deferred without saying so

Three quality modifiers were **explicitly 1.0**: FOV-edge degradation, altitude
falloff above 200 m AGL, and the bearing bias random walk. The thermal duplicate
was also absent. All four are now in 6b.

**6a's list was incomplete**, which is worth recording rather than quietly
fixing. The reference also multiplies in `detection.effective_noise_multiplier`
— partial vegetation widens the bearing noise without blocking the observation —
and 6a neither implemented it nor named it among its deferrals. It is in 6b, on
the visibility oracle's result, which is where the reference computes it.

Still outstanding after 6b, and named here so the next slice does not have to
rediscover it: **clutter**. `SensorModel.generate_clutter` emits false-alarm
observations from a separate `(node_id, "clutter", tick)` stream, capped at
`confidence_floor * 0.9`. That is a second observation source rather than a
modifier, and it is stage 6c.

## Verification

- 28 unit tests, all through the real tick transaction.
- **27 mutants, 26 caught, 1 proven equivalent.**

The 19 tests written first caught only **16 of 26** on the first pass — the
worst first-round score of any stage in this PR, and the round that most earned
its keep. What the survivors had in common is that the test rig's *convenient*
settings made distinct formulas agree:

| survivor | why nothing caught it |
|---|---|
| elevation is the raw height difference | at a 30° gate, `45°` and `100 m` both pass and `63°` and `20 m` both… did not disagree either |
| elevation uses horizontal range, not slant | `45°` vs `90°` are on the same side of every gate the tests used |
| every sensor is treated as directional | every omnidirectional test used a *stationary* sensor, where the gate is skipped for a different reason |
| the stream key drops the target id | the only "same place" test used two **nodes**, so the node id still separated them |
| confidence drops the detection probability | the permissive spec put the knee past every range, so `pd` was exactly 1.0 |
| the confidence floor is dropped | the range slope alone bottoms out at 0.3, so the 0.15 floor never bound |
| the tally is not reset between ticks | nothing stepped twice and re-read it |
| only two noise draws are taken | nothing looked at the z component or the draw count |
| the degenerate branch returns zero | unreachable from a random stream |

The elevation pair needed a threshold at which **all three** candidate formulas
disagree, which took a table to find:

| target | true angle | raw `dz` | `dz`/horizontal |
|---|---|---|---|
| `(100, 0, 100)` | 45° | 100 | 90° |
| `(10, 0, 20)` | 63.4° | 20 | `asin(2)` = NaN |

At a 60° gate the true answer rejects the first and accepts the second, and each
wrong answer gets at least one backwards — the NaN case because `NaN < 60.0` is
false, so it accepts by default.

The last two survivors drove a design change rather than another test.
`noisy_unit_vector` is now split into a pure `perturb_unit_vector` plus a
`bearing_perturbation` that takes the draws — the same shape
`argusnet_sensing::BearingBiasDrift` already uses for its increment, and for the
same stated reason: it makes the function checkable value-for-value. It also
makes the degenerate branch reachable at all. A random stream will not produce a
perturbation that cancels the unit direction on demand, so with the draw inside
the function that branch was untestable and would have sat there unverified.

The one equivalent mutant turns the missing-truth-capability `Invariant` into
`NoWork`. Stage 6 is one of the two stages §3 admits to truth, so that branch is
unreachable through the pipeline — which is the point. `stage_six_is_admitted_to_truth`
is what would notice if the list changed.

# Stage 6b: the quality modifiers

Four things 6a left at `1.0`, plus the thermal duplicate.

## FOV edge and altitude

Noise grows and confidence falls toward the FOV boundary
(`1 + f·1.5`, `1 − f·0.4`), and again above 200 m AGL, ramping over the next
400 m and then **saturating** — a sensor at 10 km is treated exactly like one at
600 m.

Two details that only show with the right test:

- **AGL is above *ground*, not above the datum.** A drone at 300 m over a 250 m
  hill is 50 m up and outside the ramp; the same drone over flat ground is
  inside it. The terrain sampler is injected, like everything else this stage
  needs from `argusnet-world`.
- **The ramp applies only to a *mobile* sensor.** A mast-mounted camera at 600 m
  is not treated as airborne, which is the reference's `if node.is_mobile`.

## The bias walk runs on the sensor's own stream

`BearingBiasDrift` gets its own generator in the reference, so the walk is **per
sensor and continuous across ticks**, while the pair streams are opened fresh
for every `(node, target, tick)`. Driving the bias from the pair stream would
make a drone's accumulated bias a function of how many targets happened to be in
view — which is why the two are kept apart here and why a mutant that re-keys
the bias stream per tick is caught.

**The rotation axis is not random**, although `apply_bias_to_direction`'s
docstring says it is *"a rotation about a random axis perpendicular to the
direction vector"* — and the function takes an `rng` it never reads. The axis is
the cross product with world X, falling back to world Y within about 25° of X.
The code is the contract; the docstring is wrong, and the unused generator is
not reproduced.

## The thermal duplicate

A `thermal`/`lwir`/`mwir`/`ir` sensor publishes a **second** observation of
every target it accepts: 1.3× the noise, 0.9× the confidence, and a separately
drawn direction taken from the *same* pair stream immediately after the primary.
That ordering matters — the thermal draws come last, so adding or removing a
thermal sensor cannot shift any other observation's noise.

## Verification

- 49 unit tests in the stage, all through the real tick transaction.
- **32 mutants, 29 caught, 3 proven equivalent.**

The first round scored 22/32, and one survivor was a genuine self-inflicted
wound worth naming:

**The thermal assertions referenced the constants they were testing.**
`assert_eq!(thermal.bearing_std_rad, primary.bearing_std_rad * THERMAL_NOISE_FACTOR)`
passes for *any* value of `THERMAL_NOISE_FACTOR`, so mutating the constant moved
the expectation with it. Both thermal factors are now asserted against literals,
with a separate assertion pinning the constants themselves.

Two more survivors were confounded comparisons:

- the FOV-edge test put its edge target **further away**, so the range slope
  alone explained the confidence drop. It now sits at the same range, 40° off
  axis.
- nothing tested altitude's effect on **confidence** at all — only on noise.

And one was a test that measured the wrong thing: every bias test read
`bias_rad()`, which a stage that computed the walk and then discarded it would
still satisfy. The new test requires the published direction to equal
`apply_bias_to_direction(undrifted, bias)` exactly.

The three equivalent mutants are each pinned by a test that says *why*, so the
equivalence is checked rather than assumed:

| mutant | why it is equivalent | what would notice |
|---|---|---|
| the altitude threshold is `<` not `<=` | at exactly the threshold the excess is zero, so the ramp returns `(1.0, 1.0)` — the early return's own answer | `the_altitude_threshold_boundary_is_neutral_either_way` |
| the altitude confidence floor is dropped | the ramp saturates at `0.5`, well above the `0.3` floor, so the floor never binds **with the current slope** | `the_altitude_confidence_floor_is_currently_unreachable` fails the moment it does |
| AGL is not floored at zero | a negative AGL is below the threshold, and the ramp is neutral there | `a_negative_agl_is_already_neutral_without_the_floor` |

# Stage 6c: clutter

`SensorModel.generate_clutter` emits Poisson-distributed false alarms tagged
`target_id = "clutter"`. It is a second observation *source* rather than a
modifier on an existing one, which is why it is its own slice.

Off by default — `false_alarm_rate_per_scan = 0.0` — so a scenario that never
asks for clutter never pays for it.

## Three things that are not obvious from the shape

**Clutter is emitted even with no targets at all.** The reference's clutter loop
sits inside the per-node loop *after* the per-target one, so an empty world still
produces false alarms. An earlier draft of this stage returned `NoWork` as soon
as the truth list was empty, which silently dropped them; the early return is now
on sensors alone.

**Its stream is its own**, keyed `(node_id, "clutter", tick)`, so the number of
real targets in view cannot shift the false-alarm draws.
`the_clutter_stream_is_independent_of_the_targets_in_view` runs two worlds —
empty and crowded — and requires bit-identical clutter.

**The confidence is capped at `confidence_floor * 0.9`** by the caller, so no
false alarm is ever more trusted than the least-trusted real observation. The
draw itself runs to 0.4, well above the 0.15 floor, so without the cap most
clutter would outrank a long-range real return.

`clutter_min_range_m` and `clutter_max_range_m` are declared on the reference's
config and never read by `generate_false_alarms`. They are not carried across.

## The Poisson draw is not numpy's, deliberately

numpy switches between PTRS transformed-rejection and an inversion loop at
`lambda = 10`, and reproducing either bit-for-bit would mean porting its branch
structure. ADR-006 already re-baselined every stream in this crate — the
contract is *"deterministic for a fixed seed and config"*, not *"identical to
numpy"* — so `DeterministicRng::poisson` uses Knuth's multiplication method and
the fixtures pin it.

`exp(-lambda)` goes through **`libm::exp`, not `f64::exp`**. Neither is required
by IEEE-754 to be correctly rounded and glibc disagrees with Apple's libm by
1 ULP, which is what broke the conformance fixture when CI moved to a Mac.

## Two failures worth recording

**The crate's own guard caught the libm slip.**
`no_draw_routes_through_the_platform_libm` scans this module's source for bare
`.exp()`/`.log()`/`.cos()` calls and fails on them. It named the exact line. I
had run the stage's own tests, which passed, and would have shipped the slip if
I had not run the whole crate — a reminder that a filtered test run is not a
verification.

**The mutation harness left a mutant in the tree.** The "poisson compares the
wrong way" mutant turns Knuth's loop into one that almost never terminates
early, so a test that draws 20,000 samples ran until the harness's 900-second
subprocess timeout, which killed the script before its restore step. The source
was left mutated and the buffered output was lost.

Nothing was committed, because the working tree is diffed against the
pre-mutation backup before any commit — that check is what caught it. Two
lessons for the harness itself: restore in a `finally`, and run Python
unbuffered so a killed run still yields its partial results.

## Verification

- 58 unit tests in the stage; 316 in the crate.
