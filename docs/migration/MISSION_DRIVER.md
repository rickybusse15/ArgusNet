# P4.8e.3: the mission driver — the default run in the Rust tick

**Status:** delivered 2026-09-23, locally verified, then merged 2026-10-01 in #122 (`badb1e1`). Package ID
`P4.8e.3`. See [STATUS.md](STATUS.md).

`scan_map_inspect` is the default mission mode, the one `argusnet sim` runs:
lawnmower scan, localization, POI inspection, egress, complete. Every piece
had a Rust body: coverage and localization (stage 8), the phase machine
(P4.8a), the override stage (stage 2), the POI manager (P4.7). What was
missing was the **caller** P4.8a deliberately left the side effects to: the
code that turns phases into assignments, redirects and routes home. This slice
adds it, assembles stages 2, 8 and 9 from the scenario bundle, and runs the
default mission end to end.

## `MissionDriverStage` (stage 9)

One tick, in the reference's order. `run_simulation`'s `elif` chain reads the
phase as it stood at the top of the tick, and in the inspecting branch it does
all its POI work *before* asking `all_complete`:

1. read the phase the machine settled on last tick;
2. if `Inspecting`: rescore POIs (once per run), then for each drone in fleet
   order assign energy-aware, redirect on a *new* assignment, hand off below
   30% battery to a free drone above 50%, and credit dwell; then team-assign
   and complete;
3. build the machine's inputs from **this** tick's stage 8 results (coverage,
   gap fraction, localization) and the POI manager as step 2 left it;
4. advance the machine and turn its egress events into overrides, which stage
   2 flies from the next tick.

**The rescore belongs to the driver, not to the event.** The machine emits
`RescorePois` on the tick it *enters* `Inspecting`. The reference rescores at
the top of the first tick *spent* inspecting, after that tick's coverage update
has changed what the density read sees.

`WorldMap.coverage_in_region`, the rescore's density read, is ported here.

## Findings

1. **Two `hypot`s where the reference uses `np.linalg.norm`.** One in the
   redirect's flight time (`GotoHover::redirect`), one in the POI manager's
   distance, which decides the 30 m dwell boundary and energy-aware scoring.
   Both are now the plain square root of the sum of squares. This is the same
   one-ULP trap P4.8d.2b found in `FollowPathController`. The scan run never
   lands on a boundary, so it cannot tell them apart. A searched offset can:
   `(10.0007, 28.284023750343586)` has `sqrt` = 30.0 exactly and `hypot` =
   30.000000000000004. The reference's `POIManager` **credits** the dwell
   there and its `np.linalg.norm` returns 30.0, so `hypot` would have denied
   a dwell the reference grants. Both are now pinned
   (`tests/mission_driver.rs`).
2. **The reference's team-assignment branch is dead code.** `run_simulation`
   calls `request_team_assign(pending_pois[0], ...)` on a POI it has just
   filtered as *pending*, and `request_team_assign` refuses anything not
   already *active*. It can never assign anything. The driver keeps the call,
   and `the_reference_team_assignment_branch_assigns_nothing` pins that it is
   inert, so "fixing" it would show up as a behaviour change.
3. **`MappingLink::set_scan_mode` stores a flag nothing reads.** Rust feeds one
   coverage grid in either mode, and localization is gated by
   `enable_localization`. The flag is inert; the mutant that flips it is
   equivalent.
4. **The existing mapping → mission integration test pushed inputs a tick
   late.** `mapping_drives_mission.rs` sets the machine's inputs before the
   tick, so it carries the previous tick's coverage. Its own comment claims the
   reference's order, which is the current tick's. The driver computes inputs
   inside stage 9, from stage 8's results for the same tick. The old test is
   kept, since it tests the machine and the gate, but its ordering claim was
   not the reference's.
5. The POI lift is behaviour-preserving. `make_poi_positions` and
   `build_run_inspection_pois` were lifted verbatim out of `run_simulation`,
   and a 400-tick scan run through committed `HEAD` and through the lifted code
   produced byte-identical frames through every phase.

## What is compared

`scripts/capture_scenario_bundle_fixture.py` gains a `scan` case (small map, 2
drones, 3 stations, 3 POIs, coverage threshold 0.3, 400 one-second ticks,
reaching every phase through `complete`). It records per tick every node's
kinematics, health and battery fraction, and the mission phase, coverage
fraction, POI statuses and localization confidences.

* **Free-running:** the phase follows localization, and localization draws from
  ADR-006 streams, so the two runs first disagree on the phase at tick 34.
  Until then every node, the coverage fraction and the phase match bit for bit;
  stations match for all 400 ticks. The Rust run then finishes the mission on
  its own, through `complete`.
* **With the reference's localization replayed** (`LocalizationReplay`, a seam
  on the driver that makes stage 9 read recorded confidences instead of stage
  8's), the mission has no stochastic input left, and **every node on every
  tick, the phase and the coverage fraction match bit for bit for the whole
  400-tick run**. That covers every POI assignment, redirect, dwell, egress
  route and transition.

## Verification

| Test | What it proves |
| --- | --- |
| `a_scan_map_inspect_run_with_replayed_localization_matches_throughout` | 5 nodes × 400 ticks, position, velocity, health and battery bit-exact; phase, coverage fraction and the localization-timeout latch on every tick; through `complete` |
| `a_scan_map_inspect_run_matches_until_localization_decides` | free-running: exact to the first localization-driven phase difference (tick 34); stations for all 400 ticks; the Rust run completes the mission on its own |
| `the_scan_assembly_wires_the_mission_as_the_bundle_describes` | every threshold, speed and POI handed to the driver is the bundle's |
| `coverage_in_region_matches_the_reference_world_map` | 180 queries against a real `WorldMap`, edges and sub-cell radii included |
| `the_dwell_boundary_is_decided_by_numpy_norm`, `the_redirect_flight_time_uses_numpy_norm` | the two distances whose last bit decides a boundary |
| `an_enclosed_hole_holds_the_mission_in_scanning` | the gap-fill gate holds at a 4%, 3% and exactly 2% hole and releases at 1% |
| `a_replayed_localization_timeout_latches`, `the_reference_team_assignment_branch_assigns_nothing` | the timeout latch; the dead team branch stays dead |
| `test_scenario_bundle.py` (9) | scan export carries the reference's own POIs; opt-in mission features refused by name |

### Mutation round

20 mutants: **17 caught, 3 proven equivalent**, after two strengthening
passes. The first honest pass caught 9. (An earlier run looked like 20 of 20,
but one test was failing on its own and "caught" every mutant; it was fixed
before the count above.) The survivors were closed by the reference fixture
for `coverage_in_region` and the two norm boundaries, by the gap-gate and
timeout simulations, and by wiring checks against the bundle. Two of my own
tests were too weak on the first try. One added 10 s before comparing a flight
time, which rounded the last bit away. The other put the POI far from the
origin, so the offset no longer straddled 30 m. Both were fixed.

The equivalents: the team branch (dead in the reference); clearing POI
overrides at egress (stage 2 applies egress after POI, so leftovers are
unobservable while every drone has a home, and every launched drone does);
the mapping scan-mode flag (never read).

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1037 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (after removing one `clone` on a `Copy` type) |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests` | clean |
| `pytest tests/` | 1008 passed, 9 skipped, 2 failed. Both are the pre-existing `argusnetd` startup flake (`Connection refused`, the daemon never bound). On re-run both passed and a different test hit the same refusal. |
| CI | **not run**: nothing is pushed |

## Not in this slice

* **Frontier redirects, occlusion-aware mapping and adaptive cooperative
  search** are refused by the exporter. They are opt-in and each has its own
  planner path.
* **Coordinator election** and the replay's scan-mission fields
  (`newly_scanned_cells`, `team_localization`, `egress_progress`,
  `coordinator_drone_id`) do not move anything, so they belong with the replay
  work in P4.9a.
* **The inspection zone events** (entered / coverage_updated / exited) are
  replay output too. Same.
* **`MissionDriverStage` holds state** (POI manager, redirect map, rescore
  latch) that the checkpoint does not yet carry. That is P4.8e.4, the
  causal-state audit, next.

## Next

**P4.8e.4**: every stateful handler on the checkpoint contract (observe bias,
fusion, mapping and localizer, deconfliction, overrides, the mission driver),
`TickWorld` persisted, and the resume differential over a full default run.
