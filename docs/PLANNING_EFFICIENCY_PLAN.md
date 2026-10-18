# Planning efficiency plan: teamwork, allocation and routing

**Status:** proposed 2026-10-12. Shipped (#223–#239): the P0 metrics
(duplicate coverage still open), and behind the opt-in `efficient_v1`
policy P1's allocation, teams and safe handoff (hysteresis and the
acceptance run still open), P2's first slice (return home at the reserve)
and three P3 slices. P4–P6 are not started. Each phase below says what has
shipped.
**Workstream:** WS-P (new). Ties into F5 and F7
([AGENT_EXECUTION_PLAN.md](migration/AGENT_EXECUTION_PLAN.md)), P4.7
([DELIVERY_PLAN.md](migration/DELIVERY_PLAN.md)) and
[PLANNING.md](PLANNING.md).

## Where planning stands

The Rust port reproduces the Python reference, quirks included. That was the
right call for parity, but it leaves the fleet working much less well
together than it could. File and line references are as of `main` on
2026-10-12.

| Area | Today | Cost |
| --- | --- | --- |
| POI allocation | Greedy, one drone at a time in fleet order (`mission_driver.rs` `inspect`). Earlier drones take the best POIs first, and nothing optimises across the fleet | Drones cross paths; a far drone can take a near POI |
| Allocation cost | Straight-line horizontal distance at a hardcoded 350 W and 28 m/s (`poi.rs:26,91,244`). It ignores the trip home, hover during dwell, obstacles and the dynamic reserve | Drones accept POIs they cannot afford to finish and get home from |
| Team assignment | Never assigns anything (reference; `efficient_v1` teams since P1 slice 2): it is called on a *pending* POI and only accepts an *active* one (pinned by `the_reference_team_assignment_branch_assigns_nothing`) | Idle drones sit out while the last POI is serviced |
| Handoff | Overwrites whatever the receiving drone held, without checking (`poi.rs:305`) | A POI can be silently dropped |
| Coordinator | Elected once and then shown in the viewer, but it decides nothing; `CoordinationManager` is wired only into parity tests | No shared claims, so drones duplicate work |
| Routing | No planner at runtime (`ControllerEnvironment.planner: None`). POI, egress and frontier moves are straight lines that ignore obstacles (`GotoHover::redirect`). A* exists, but is used only at build time | Drones depend on collision push-out and deconfliction instead of planned paths |
| Return home | `ReserveReached` is emitted and nothing reads it. Egress happens only once every POI is done | No drone goes home individually on low battery |
| Scan coverage | Equal-width x strips, one per drone, flown as boustrophedon lanes on a loop until the phase changes (`builder/scenario.rs:161`) | Uneven strips; a drone that finishes keeps re-flying its own strip instead of helping |
| Frontier exploration | Greedy in controller order; a block claimed by another drone costs a soft +1000 m (`frontier.rs`). `ClaimedCells` and `select_frontier_cell` are unused | Goals can duplicate, and allocation depends on drone order |
| Pursuit | Targets are assigned at build time as `target_ids[i % n]` and never reassigned | Lost and found targets are not rebalanced |
| Deconfliction | Reactive offsets each tick; nothing feeds back into routes | Hovering conflicts persist |

**Efficiency is not measured today.** Acceptance checks accuracy, continuity,
completion and the energy floor. Nothing measures mission time, distance flown,
idle time or duplicated coverage.

## Rules for every step

1. **Measure first.** Metrics land before any behaviour change, so every change
   can be judged against a baseline.
2. **Opt in, versioned.** Each change is a new planning policy, for example
   `--planning-policy efficient_v1`. It is recorded in replay `meta` additively,
   the same way `energy_profile` and `execution_model` are. The reference
   policy stays byte for byte: all 14 legacy replays identical, `--lifecycle`
   equal (AGENT_EXECUTION_PLAN §12.3).
3. **Developed on one seed set, judged on another.** A policy is developed
   and tuned on the A3.2 qualification seeds, run through
   `run_phase_acceptance.py --rust-arg ...`. The final decision for each policy
   is made once, on a fresh held-out seed set declared before the run and never
   used for tuning. The D-12 confirmation set is not reused for this, since it
   was itself a one-time fresh test. The policy must not regress any envelope,
   and must show its efficiency gain with variance.
4. **Promote by decision.** A policy becomes the default only through a recorded
   decision (D-n) and a fixture-rebaseline commit, never as a side effect.
5. **Deterministic.** Every solver iterates in a fixed order and breaks ties by
   id. No hash-map iteration and no wall-clock time limits.
6. **Checkpointed.** New planner state (claims, assignments, hysteresis timers)
   goes into the driver checkpoint and passes the A2 reset and restore checks.

## Phases

### P0: efficiency metrics (behaviour-neutral)

**First, record every assignment.** Today the replay cannot support these
metrics: `ReplayPoiStatus` holds one `assigned_drone_id`, while
`PoiManager::request_team_assign` keeps a second drone only in its internal
map. So P0 starts with an additive per-drone **assignment event history** in
the replay. It covers assignments, team joins, handoffs, releases and
reassignments, each with a time, the drone, the task and the reason. It is
seeded at tick 0 with the tasks `builder/scenario.rs` hands out when the
scenario is built: scan strips, search lanes, follow and pursuit targets.
From then on it records every change, from the driver's `Decision` records
and from the controllers that switch a drone's task (frontier goals, egress).
A drone flying its scan strip is working, not idle. Every metric below is
derived from that
history, never from POI status. Otherwise, once P1 turns on teams, a drone
helping on a POI would count as idle, and handoff and thrash counts would
leave it out.

*Done (history):* the Rust writer records `meta.assignment_history`, drone id
to its events in time order, each `{timestamp_s, kind, task, previous,
reason, to}`. A task is what the drone flies on the tick: a recall stage 2
flew (`egress`, `poi:<id>`, `frontier`), else its own controller (`search`,
`follow:<target>`, `explore`; a search-then-follow drone is `search` until its
sensor engages). Kinds are `assign` (its first task), `reassign`, and
`handoff` and `team_join` from stage 9's applied decisions, since the
reference leaves a drone that hands its POI on flying to it. Reasons come from
the transition (`poi_assigned`, `poi_released`, `egress`, `egress_cleared`,
`target_engaged`, ...). It is read after each committed tick, cleared by a
reset (so a reset run records exactly a fresh run's history) and, like the
frames, starts at the checkpoint in a resumed run. The scorecard reports
`task_changes`, `poi_handoffs` and `assignment_thrash` (a change within 30 s
of that drone's previous one) from it; None for a replay without one.

*Done (time and effort):* the scorecard reports, from the replay, the mission
makespan (first frame to the last POI completed, when every POI was), the
time to reach the scan's coverage threshold, the mean POI service time (first
time a drone was on the POI, from the history, to its completion), distance
flown per drone and per fleet, battery used per drone, and idle time (time on
`hold`). Duplicate coverage needs the belief map's measurement counts and is
still open.

The metrics are computed from the replay, in Rust (`argusnet-sim`
evaluation) and mirrored in the Python scorecard:

- **Time:** mission makespan, time to reach scan coverage, and per-POI time
  from assignment to completion.
- **Effort:** distance flown and energy used per drone and per fleet, idle time
  (hovering with no task), and duplicate coverage (cells measured twice).
- **Reassignments:** handoffs, reassignments and thrash (an assignment changed
  within N seconds).

**Make completion observable.** `mission_completion_rate` is currently
vacuous for Rust scan-map-inspect replays. `evaluate_replay` reads a top-level
`mission.required_objectives_*`, the Rust replay writer has none, and
`metrics.py` then reports 1.0. P0 adds an additive objective summary to the
replay (POIs required, completed, unmet) and derives the rate from it, with a
test that a run leaving a POI unfinished scores below 1.0. Only after that
can the metric gate any policy.

*Done:* the replays already carry the summary. Every frame's
`scan_mission_state` has `total_poi_count`, `completed_poi_count` and the
per-POI statuses, so `metrics.py` now takes the objectives from the last
frame when a replay has no top-level `mission`, and no replay change was
needed. A run with an unfinished POI scores below 1.0, and one with no POIs
still has no objective to miss. The stored acceptance reference was
re-recorded with it (525 runs).

Reported by phase acceptance, for reporting only (no envelope), on the
canonical and qualification seeds. This baseline is what every later phase
is measured against.

### P1: fleet-wide POI allocation (`efficient_v1`)

- **Cost that tells the truth.** Each drone's cost for a POI covers the trip
  there, hover energy for the dwell (PhysicalV1), and the trip home afterwards
  against `dynamic_reserve_wh`. An assignment that would break the reserve is
  infeasible, not merely discouraged.
- **Global assignment.** Every time the mission lane runs, solve
  drones × POIs together. Use a deterministic Hungarian solver for small fleets
  and a sequential auction above a size limit. Ties go to the lower id.
- **Hysteresis.** Keep the current assignment unless a new one is better by a
  set margin, so drones don't switch back and forth.
- **Teams that work.** When there are more free drones than pending POIs,
  assign the extra drones to *active* POIs, so several drones credit dwell
  together. This is the reference team branch with its bug fixed, and the
  reference policy keeps the bug.
- **Safe handoff.** A drone receives a handoff only if it is free and can afford
  the POI and the trip home. If no drone qualifies, the POI goes back to the
  pool and is reported as unassigned (F5).

**Acceptance:** shorter makespan and less distance flown on scan-map-inspect,
with `mission_completion_rate` and `energy_reserve_min` within their envelopes.
The policy is judged with the PhysicalV1 energy profile, where hover costs
energy. `energy_reserve_min` is estimated from distance alone and cannot see
hover during dwell or a team waiting at a POI. So acceptance also requires
`battery_remaining_min`, which reads the battery ledger and includes hover. It
gets an envelope of its own, declared before the policy's first acceptance
run.

*First slice done:* `--planning-policy efficient_v1` (CLI and viewer `--run`)
solves free drones × pending POIs with a deterministic Hungarian solver
(`argusnet-planning::allocation::min_cost_assignment`, checked against brute
force). The cost is the trip's energy, plus hover for the dwell, less 5 Wh
per priority level. A pair that would leave the drone below
`dynamic_reserve_wh` home (the static reserve without a home) is infeasible.
A drone keeps the POI it holds, so there is no preemption and so no
hysteresis yet. The reference is byte-identical (14/14 replays; reset and
resume equal). Development measurement (qualification seeds, PhysicalV1,
paired over 30 seeds, `scripts/compare_scenarios.py`; not the acceptance
run):

| Scenario | `mission_completion_rate` | `fleet_distance_m` | `battery_remaining_min` |
| --- | --- | --- | --- |
| `planner_adversarial` | +0.10 (0.04 to 0.16), better on 12, worse on 1 | +108 (−3 to 219), no detectable difference | identical |
| `persistent_long` | +0.17 (0.09 to 0.24), better on 15, worse on 0 | +363 (−3 to 728), no detectable difference | identical |
| `baseline_coverage` | identical (its runs end before the POIs) | identical | identical |

The measurement found a defect in P0's `assignment_thrash`: it counted
finishing a POI and taking the next one as thrash. Leaving a completed POI
now restarts the window. With that fixed, thrash is 0 under both policies.
*Second slice: teams.* Under `efficient_v1`, a drone still free after the
pending POIs are allocated joins a POI being inspected, so both credit its
dwell. It joins only once every drone holding the POI is in dwell range, only
when it would arrive before they finish it alone, and only if it can afford the trip, its share of the dwell
and the trip home above its reserve. The same solver picks the joins, at most
one new teammate per POI per run, on trip-plus-hover energy. The POI's status
keeps naming the drone that took it. Under `efficient_v1` only that drone
hands the POI on when it runs low; a teammate keeps its share. Each join is an
applied `team_assignment` decision (`fleet_team_join`) and a `team_join` in
the assignment history. The reference keeps its dead team branch and is
unchanged byte for byte.

Measured against the first slice on the same qualification seeds (PhysicalV1,
paired, `efficient_v1` before and after teams; measured before the dwell-range
rule, which can only make joins rarer):

| Scenario | Seeds with a team join | `mission_completion_rate` | `fleet_distance_m` |
| --- | --- | --- | --- |
| `planner_adversarial` | 8 of 30 | identical on every seed | +9 m (0 to 18), worse on 8, better on 0 |
| `persistent_long` | 6 of 30 | one more POI on one seed | +14 m (−40 to 69), no detectable difference |

So teams work as specified (two drones finish a POI in half the dwell), but
on these scenarios they rarely pay. Their POIs need 15 s of dwell, so by the
time a drone is free little dwell is left to share, and the trip costs
distance. Teams should matter for long dwells. Whether they stay on inside
`efficient_v1`, or need a minimum time saved before a drone joins, is open.

*Third slice: safe handoff.* Under `efficient_v1`, every drone below the
handoff threshold gives up the POI it holds before allocation. A lead's POI
goes to the cheapest receiver that can afford the dwell still to do and the
trip home above its reserve. A teammate already on the POI needs no trip; a
free drone above the threshold needs the trip. The solver picks receivers
across all low leads. A POI no drone can afford goes back to the pool
(`poi_release`, `no_affordable_receiver`) instead of being handed to a drone
that would run short, and a low teammate leaves its team
(`low_battery_teammate`). A drone that gives a POI up stops flying to it and
takes nothing new while it is below the threshold. Getting it home is P2. The
reference's handoff, which overwrites the receiver without checking, is
unchanged.

Still to do in P1: hysteresis once preemption exists, and the acceptance run (held-out seeds and a `battery_remaining_min` envelope, both
declared first).

### P2: per-drone return home and reserve feasibility (F5)

- Consume `ReserveReached`. A drone that reaches its reserve gets a
  `SetEgress` home of its own, and its POI goes back to the allocator.
- Re-check reserve feasibility continuously, not only when assigning. No team
  assignment may weaken one drone's reserve.

*First slice done (`efficient_v1`):* every mission run, a drone whose battery
is at or below its reserve home from where it is (`dynamic_reserve_wh` to its
station, the battery's own rule) is routed home on its own (`SetEgress`, a
`return_home` decision with reason `reserve_reached`), whatever the mission
phase. Its POI goes through the safe handoff (P1): to a drone that can afford
it, or back to the pool. The journal says it was the reserve, not a low battery:
`reserve_handoff` instead of `safe_handoff`, `reserve_teammate` instead of
`low_battery_teammate`, and `at_reserve` in the inputs of each. A homeward drone takes nothing again; the set is
checkpointed, and absent from checkpoints that have none. A drone with no
station stays. Team joins already require the reserve (P1), which covers the
second bullet at join time. Not yet measured on the qualification seeds.

### P3: obstacle-aware routing at runtime

- Wire `RoutePlanner` (visibility-graph A* with smoothing, `route.rs`) into
  `ControllerEnvironment`.
- POI, egress and frontier moves follow planned routes instead of
  `GotoHover::redirect` straight lines. The allocation cost from P1 then uses
  route length instead of straight-line distance.
- Cache routes by obstacle-layer revision (P4.7), so a route is not recomputed
  every tick.
- In runs with no prior map, plan on the belief grid's structures and replan
  when the belief changes. This is the mode UNKNOWN_WORLD.md asks for.
- Later: 3D routing that climbs over low structures, where the energy cost
  says it is worth it (KNOWN_GAPS "3D path planning").

*First slice done (`efficient_v1`):* the mission driver (stage 9) is given
the prior map's obstacle layer at assembly and plans with `RoutePlanner`
(visibility-graph A* with smoothing, 8 m clearance, buildings and walls
only). A trip is planned only around the obstacles that reach its height:
those whose top plus the clearance is above the lowest height the flight
passes through, rounded down to a 10 m band. A drone flying over a low
building collides with nothing (collision is 3D), so it is not sent round
it. A POI redirect, a reserve return home and the mission-end egress
whose straight line is blocked fly the planned route instead: `GotoHover`
carries the route's corners (`via_xy`, absent from the serialised form when
there are none, so every existing checkpoint is unchanged) and flies the
polyline at the redirect speed, with height changing linearly over the
whole flight and an exact stop on the target. The fleet-wide allocation,
the team-join arrival and cost, and the safe-handoff trip cost use the
route's length when the route turns, and exactly the straight-line cost
otherwise. The redirect decisions record `route_length_m` and
`route_waypoints` (and the egress decision a `routes` map by drone), only
when a route turns; reason codes are unchanged. A trip the planner cannot
route (an endpoint within clearance of an obstacle or out of bounds, or no
way through) flies straight, exactly as before. A run with no prior map
(`set_world_knowledge` with `NoPrior`) withdraws the layer, so it never
routes on truth. The reference never reads the layer and is byte-identical.

The planner's own route cache is deliberately not used at runtime: it keys
on endpoints snapped to 10 m, so a hit returns a route from where an earlier
request started, and a resumed run (empty cache) would fly differently. The
driver uses `RoutePlanner::plan_route_uncached` and keeps each run's routes
by exact endpoints only for that run.

Measured against `efficient_v1` before routing, on the same qualification
seeds (PhysicalV1, paired over 30 seeds; development measurement, not the
acceptance run):

| Scenario | Seeds with a routed redirect | `mission_completion_rate` | `fleet_distance_m` | `battery_remaining_min` |
| --- | --- | --- | --- | --- |
| `planner_adversarial` | 11 of 30 | identical on every seed | +1.9 m (−0.8 to 4.6), worse on 7, better on 5 | identical |
| `persistent_long` | 6 of 30 | identical on every seed | +0.06 m (−0.07 to 0.20), no detectable difference | identical |

So the routing works but these scenarios barely exercise it: the urban
preset places a handful of buildings, and drones redirect at operating
height, mostly above them. A scenario with tall, dense structures (the
realism plan's) is what will show whether it pays.

Still to do in P3: the reserve (`dynamic_reserve_wh`) still prices the trip
home as a straight line; routes are not cached across runs by obstacle-layer
revision (P4.7), so every mission run re-plans the blocked pairs it costs;
frontier moves still fly straight; planning on the belief grid for runs with
no prior map; and 3D routing.

*Measured on a dense city.* On `planner_adversarial` the first routing slice
(`efficient_v1` routes POI and egress redirects around known buildings and
walls; PR #236) had little to route: `urban_flat` has a handful of buildings
and drones mostly fly above them. The `urban_dense` preset is a flat city of
30 blocks, 70–110 m tall, on a 6 × 5 grid with streets of at least 42 m
(73 m on the medium map). POIs hover 40 m above the streets, so most
POI-to-POI flights cross a block. The scenario `urban_dense_inspection` is
`planner_adversarial` on it, run for 150 s: at 120 s most runs are still
scanning, and by 180 s every run has finished its POIs.

`efficient_v1` before routing (main, with teams, safe handoff and return
home) against the same with PR #236 merged, both with the preset, on the
qualification seeds (PhysicalV1, paired over 30 seeds, identical exported
configs; development measurement, not the acceptance run). The scorecard has
no collision metric, and a run with a prior map has no physical guard, so a
straight flight passes through a block unrecorded. "Samples in a block" counts
replay samples (every 0.25 s) where a drone is inside a block's box.

| Metric | Before | After | Paired difference (95% CI) | Seeds better / worse |
| --- | --- | --- | --- | --- |
| `mission_completion_rate` | 0.744 | 0.733 | −0.011 (−0.034 to 0.012), no detectable difference | 1 / 3 |
| `fleet_distance_m` | 18,793 | 18,744 | −49 m (−132 to 35), no detectable difference | 15 / 13 |
| `battery_remaining_min` | 0.947 | 0.947 | identical on every seed | 0 / 0 |
| Samples in a block | 21.5 | 13.4 | −8.1 (−11.4 to −4.8), better | 21 / 1 |

Routing turned 73 of the 187 POI redirects (on 28 of 30 seeds; the other two
never leave the scan in 150 s), adding 21 m to a trip on average (median
18 m, at most 81 m), and 8 of the 9 egress decisions carry a route. No
routed redirect entered a block. The completion rate falls by one POI on
three seeds, probably where a longer trip moves a POI's finish past the
150 s cut-off.

Routing removes only 44 of the 100 passes through a block (a pass is a run of
consecutive samples inside one), because 85 of the 187 redirects cannot be
routed: the planner refuses an endpoint within its 8 m clearance of an
obstacle or of the map's edge. The POI generator rejects only points inside a
block, so 56 redirects start or end within 8 m of a wall, and 35 start or end
within 8 m of the map's edge, where the scan's lanes end (6 are both). Those
caused 48 of the 56 remaining passes. Flights home after an egress caused 7,
6 of them by the drone whose ground station stands 5 m from a block. So the
next P3 step that would pay here is letting a route start or end inside the
clearance (planning from the nearest clear point, or shrinking the clearance
at the endpoints), before reserve pricing or caching.
Reproduce with `scripts/compare_scenarios.py --scenario urban_dense_inspection`
(one binary per run; it now accepts any scenario in `scenarios.py`).

*Second slice: route ends inside the clearance.* A trip whose start or end
lies within the planner's clearance of a wall or of the map's edge is now
planned from and to the nearest clear points
(`RoutePlanner::plan_route_between`), joined to the true ends by short
straight legs that must cross no obstacle's own footprint (an end on a
face counts as inside). Only those legs are inside the clearance; when
nothing is in the way and no end moves, it is still the straight line. Measured on `urban_dense_inspection`
against main with the first slice (same seeds and setup as above, binaries
built in separate target directories):

| Metric | First slice | Route ends | Paired difference (95% CI) | Seeds better / worse |
| --- | --- | --- | --- | --- |
| `mission_completion_rate` | 0.733 | 0.744 | +0.011 (−0.005 to 0.027), no detectable difference | 2 / 0 |
| `fleet_distance_m` | 18,740 | 18,751 | +11 m (−87 to 109), no detectable difference | 15 / 12 |
| `battery_remaining_min` | 0.947 | 0.947 | identical on every seed | 0 / 0 |
| Samples in a block | 14.7 | 5.0 | −9.7 (−13.2 to −6.2), better | 20 / 0 |

Routed POI redirects rise from 73 to 147 of 187. The passes left come from
trips with no way round inside the clearance and from deconfliction, whose
lateral push is not checked against obstacles in a run with a prior map.

*Deconfliction kept out of structures.* Under `efficient_v1` stage 3's
adjusted positions now go through the same `PhysicalGuard` a run with no
prior map uses (out of the ground and out of structures, contacts recorded in
the replay). The reference keeps its unguarded push, so its replays are
unchanged. On `urban_dense_inspection` it moves little: samples in a block
5.0 to 4.7 (−0.3, −0.8 to 0.1; 2 seeds better, none worse), everything else
identical. Deconfliction rarely pushed a drone into a block; the passes left
are trips the planner still cannot route.

*Reserve priced by route.* The reserve home that return home (P2), safe
handoff and allocation check (the battery's `dynamic_reserve_wh`: the energy
home at a safety margin of 1.25, never below the 20 % floor) is priced along
the route home when that route turns, each leg at the wind along it. A trip
home that does not turn keeps the battery's straight-line reserve exactly.
Planning a route from every drone's position on every run cost 13 times the
run, so a drone with enough energy to fly `MAX_PRICED_DETOUR` (2) times the
straight line home at the slowest ground speed the wind allows is not planned
one: a longer detour, behind a very long wall, is priced only once the
battery falls that far. Routes between a POI and a station, whose ends do not
move, are kept across runs. On `urban_dense_inspection` (30 seeds, 150 s, and
one seed at 450 s) no drone comes near its reserve, so every replay is
byte-identical to main; a run costs about 0.4 s more, once, for the
POI-to-station routes. The effect is in the unit tests: a drone with more than
the straight-line reserve but less than the routed one goes home when the
wall is known, and not otherwise.

### P4: coverage that rebalances

- **Balanced split.** Size each drone's scan region by flight time and the
  drone's distance from home, not by equal width.
- **Finish, then help.** A drone that has covered its region stops looping it
  and takes the largest uncovered area (measured from coverage cells), instead
  of re-flying covered ground.
- **Frontier claims.** Replace the soft +1000 m penalty with hard claims through
  `ClaimedCells`. Assign frontier blocks across all explorers at once, with the
  same deterministic solver as P1.
- **Adaptive revisits (F7):** limited re-flights of cells whose revisit age or
  uncertainty is high.

*First slice: fill the seams (`efficient_v1`).* Measured first: on
`planner_adversarial` (120 s) the scan takes a median 94 s and on
`persistent_long` (300 s) 256 s, so completion (25 % and 20 %) is decided by
the scan. Each drone loops its own equal-width sector, and the cells between
sectors are not on any lane: once coverage passes the threshold, those
seams are enclosed holes, and at 2 % of the grid or more the gap gate keeps
the mission scanning to the end of the run (4 of 30 seeds on
`planner_adversarial`). Now, while the gate holds, each hole (4-connected
gap cells) gets the free drone that reaches it soonest (the same solver,
on route length), which sweeps its cells column by column from the nearer
end, at its own height above the highest ground it crosses, as a frontier
override (`gap_fill`). It goes back to its lane once the hole is covered or
its sweep ends (`gap_fill_release`); past the scan the frontier gate ends
the flight. A drone whose override ends now flies back to its own path at
the redirect speed rather than appearing on it (stage 2, every policy but
the reference).

| `planner_adversarial`, 30 seeds | main | gap fill |
| --- | --- | --- |
| Runs that leave the scan | 26 | 29 |
| Median time they leave it | 94.0 s | 93.75 s |
| POIs inspected | 46 / 180 | 56 / 180 |

Three stuck seeds now leave the scan at 92–103 s, one leaves at 90 s
instead of 116.5 s, and no seed got worse. `persistent_long` is unchanged:
its scan reaches the coverage threshold only at 256 s, so its time goes on
coverage, not holes. That is the balanced split and "finish, then help"
below.

### P5: pursuit that reallocates

- Reassign pursuers from fused-track state: lost tracks, covariance and time
  since last seen. Use the same solver and hysteresis.

### P6: coordination over a real link

Decide what the coordinator is for:

- **Option A:** it becomes the allocation authority. Claims and assignments
  travel over the B4 delivery network with delay and loss. Drones act on claims
  as old as their link makes them, and on a link loss each falls back to its
  local greedy plan.
- **Option B:** remove it.

Option A is the realistic one. It also uses the scheduler: the planning lane
runs at its own rate (B2.4) and its results arrive over delayed links (B4).

## Order and size

| Phase | Depends on | Size | Behaviour change |
| --- | --- | --- | --- |
| P0 metrics | — | small | none |
| P1 allocation | P0 | medium | `efficient_v1`, opt-in |
| P2 return home | P1 | small | part of `efficient_v1` |
| P3 routing | P0 | medium–large | `efficient_v1` (route cost), opt-in |
| P4 coverage | P1, P3 | medium | `efficient_v1` |
| P5 pursuit | P1 | small | `efficient_v1` |
| P6 coordination | P1, B4 | large | `coordinated_v1` |

P0 and P1 give the biggest, cheapest gain. P3 is what makes the platforms
believable around buildings, and the realism plan
([WORLD_REALISM_PLAN.md](WORLD_REALISM_PLAN.md)) relies on it: power lines and
taller buildings mean nothing if drones fly straight through them.
