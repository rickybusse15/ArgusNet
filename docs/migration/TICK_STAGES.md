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
