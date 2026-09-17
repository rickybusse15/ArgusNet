# P4.8c: stage 1's battery accounting

**Status:** delivered 2026-09-17. First stage body wired into the tick, after
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
