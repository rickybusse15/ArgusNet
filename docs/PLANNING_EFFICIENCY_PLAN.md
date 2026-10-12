# Planning efficiency plan: teamwork, allocation and routing

**Status:** proposed 2026-10-12. Nothing here is implemented yet.
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
| Team assignment | Never assigns anything: it is called on a *pending* POI and only accepts an *active* one (pinned by `the_reference_team_assignment_branch_assigns_nothing`) | Idle drones sit out while the last POI is serviced |
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
reassignments, each with a time, the drone, the task and the reason, taken
from the driver's `Decision` records. Every metric below is derived from that
history, never from POI status. Otherwise, once P1 turns on teams, a drone
helping on a POI would count as idle, and handoff and thrash counts would
leave it out.

The metrics are computed from the replay, in Rust (`argusnet-sim`
evaluation) and mirrored in the Python scorecard:

- **Time:** mission makespan, time to reach scan coverage, and per-POI time
  from assignment to completion.
- **Effort:** distance flown and energy used per drone and per fleet, idle time
  (hovering with no task), and duplicate coverage (cells measured twice).
- **Reassignments:** handoffs, reassignments and thrash (an assignment changed
  within N seconds).

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

### P2: per-drone return home and reserve feasibility (F5)

- Consume `ReserveReached`. A drone that reaches its reserve gets a
  `SetEgress` home of its own, and its POI goes back to the allocator.
- Re-check reserve feasibility continuously, not only when assigning. No team
  assignment may weaken one drone's reserve.

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
