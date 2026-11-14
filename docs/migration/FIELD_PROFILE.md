# Field profile v1: one composed, scored mission

`argusnet-rs sim --profile field_v1` (and `argusnet-viewer --run … --profile
field_v1`) runs the no-prior capabilities as one named set, instead of a flag
each, and that set is what is scored against thresholds declared beforehand.
Without the profile nothing changes: 14 of 14 canonical replay hashes are
identical to `main`, every lifecycle check is equal, and the existing no-prior
flag combinations give the same bytes on the scan fixture and West Pleasant
View.

## Why

Each slice (rig, LiDAR returns, navigation camera, measured power lines,
quality requirements, contingency planning, flown energy) was evaluated
alone. Run together on West Pleasant View they did things none did alone:

| Stack, seed 7, 300 s | Known | Structure recall | World contacts | Mission |
| --- | --- | --- | --- | --- |
| Range scanner only | 99.5% | 99.9% | 0 | complete at 140 s |
| Rig | 95.6% | 95.7% | 45 | complete at 289 s |
| Rig + LiDAR returns | 90.3% | 91.2% | 35 | complete at 234 s |
| Rig + wires | 95.6% | 94.8% | 0 | still inspecting |
| Everything | 90.2% | 95.3% | 1 | complete at 218 s |

The 45 contacts were one drone. Its flight home was a straight descending
line through a building; the physical guard held it against the wall for the
rest of the egress, the mission was declared complete with it still there,
and it was then put back on its lane 810 m away. The comment in
`set_world_knowledge` said why: mission flights planned on the belief grid
were still to do.

The high "known" figures also came from drones flying their lanes again
after the mission completed. The mission itself ended its scan at 70% of
cells seen.

## What the profile is

| Part | Setting |
| --- | --- |
| World knowledge | none: the platforms map for themselves |
| Sensing | the provisional rig, its pod LiDAR with modelled returns, its navigation camera |
| Power lines | measured, tracked, poles and corridors, wire-aware flight |
| Mission flights | over the believed surface (`--belief-flights`, below) |
| Map requirement | 0.03 ground returns per m², 0.5 m vertical sigma, 40° incidence, 0.05 m declared bias |
| Scan gate | 0.9 of cells resolved against that requirement (`--scan-coverage-threshold`) |
| Energy and planning | `flown_v1`, `contingency_v1` |
| Setup | built blind (`blind_setup`, below) when run from a config |

It is recorded as `meta.run_profile: {"id": "field_v1"}`. The flags it sets
are refused with it; faults, delivery, sensor rates and lane rates still
apply. Changed options are not the profile and do not say they are.

## Mission flights over the believed surface

Stage 2's height governor (`wire_governor.rs`), which W4 added for wires, now
also governs the flights a mission flies: to a POI, home, and back to a
platform's own path.

- It looks along the flight for what the map holds: measured ground,
  believed structures, the navigation camera's obstacles, and the stand-in
  for unseen ground (never free). It keeps `clearance_m` (8 m) above the
  highest of them in a strip `corridor_m` (5 m) either side.
- Height changes at `climb_mps` (15 m/s). A flight not yet above what its
  next step crosses waits where it is and climbs, and its plan is delayed by
  the tick. Nothing is put on its plan further on.
- Within `approach_radius_m` (25 m) of its target a flight may come down:
  only believed structures on the way in are kept above, not the ground and
  not the target's own cell.
- The physical guard sweeps each step from where the platform is. A flight
  it stops at something the map does not hold climbs twice the clearance
  before trying again.
- Rejoins go through the same rules and are never ended by taking a place
  on a path that is running away.
- In such a run every other change of height is flown at the climb rate
  too: a controller that asks for a new height at once is flown there.
- A platform an override flies touched nothing at the place stage 1 worked
  out for it: that contact is not recorded.

Deconfliction (stage 3) is flown the same way. The yielder of a pair whose
ways come within 16 m in plan inside a 4 s look ahead takes a height offset
that makes up the 20 m vertical separation, flown in and out at the climb
rate, kept in the direction it started, and dropped when a new flight is
planned from where the platform is. A drone joining a POI another holds
stops 22 m short of it and 20 m above its hover height, inside the 30 m a
dwell counts within and outside the 16 m separation. Flights planned together that would pass within
18 m of each other leave apart: one is put off, whichever settles it sooner
(see the campaigns below).

Once the mission is complete every drone stays at its station
(`egress_held`), where the reference drops the recalls and each flies its
lanes again.

It is a reactive floor, not a planned three-dimensional route: a flight goes
over what is in its way, never round it.

## Setup without the true world

`blind_setup` (a scenario option; a profile applies it to configs):

- search lanes are left as laid out, neither routed round true obstacles
  nor pushed out of them;
- POIs are placed without asking whether a building stands there, and each
  declares `hover_agl_m`. The run plans a POI at that height over its launch
  floor and resolves it against the map when a flight goes there, no lower
  than the clearance over a structure believed in its cell. The absolute
  height a scenario took from the true ground is not kept;
- stations are sited outside each power line's clearance plus 15 m. On West
  Pleasant View the layout put one 12 m from the line, and the drone
  climbing from it passed 8 m from a conductor.

The replay says which was used
(`meta.world_knowledge.mission_setup.poi_hover_agl`: declared, or recovered
from the true ground for a bundle that declares none).

Still from the scenario generator, and so not earned by the platforms:
station siting itself, the POIs' horizontal places, and, in tracking
scenarios, follow routes planned on the true world. Localization starts from
the true pose plus noise.

## Scoring

`docs/contracts/field_v1.thresholds.json` holds the gates, each with its
reason. `argusnet.evaluation.field_qualification` scores a replay against
them; a gate whose value the replay does not hold fails.
`scripts/qualify_field_profile.py` imports the site, runs qualification,
held-out and tuning seeds twice each, and exits non-zero unless every counted
seed passes every gate. `scripts/real_site_inspection.py --profile field_v1`
does the same for the seeds given, with the client package and Review scene.

### First campaign (2026-11-13): not qualified

West Pleasant View, 600 s runs, 10 qualification and 5 held-out seeds, each
run twice. 11 of 15 passed every gate.

| Measure | Threshold | Min | Median | Max |
| --- | --- | --- | --- | --- |
| Mission complete, POIs done, drones home | all | 15/15 | | |
| World contacts | 0 | 0 | 0 | 0 |
| Known fraction | ≥ 0.90 | 0.950 | 0.968 | 0.989 |
| Cells meeting the map requirement | ≥ 0.90 | 0.908 | 0.919 | 0.950 |
| Structure recall | ≥ 0.90 | 0.926 | 0.978 | 0.991 |
| Structure precision | ≥ 0.90 | 0.967 | 0.970 | 0.975 |
| Ground RMSE | ≤ 1.0 m | 0.716 | 0.729 | 0.738 |
| Closest approach to a conductor | ≥ 10 m | 16.4 | 22.3 | 25.1 |
| Energy over reserve, worst tick | ≥ 0 Wh | 301 | 314 | 320 |
| Mission complete at | — | 319 s | 397 s | 542 s |
| **Largest move in one tick** | ≤ 12 m | 9.0 | 9.4 | **33.6** |
| Repeat run identical | yes | 15/15 | | |

Four seeds (707, 1010, 9002, 9005) failed the step gate, and so did tuning
seed 42. In each, one drone joined a POI the other held; deconfliction
applied its separation offset at once and changed sides from tick to tick,
moving the joiner 17 to 34 m in a tick. That led to the ramped offsets and
the join standoff above, and to a governor fix: the look ahead and the
single-step check sampled the map apart, so a cell only the step clipped
made a flight wait for a height nothing asked of it.

Flying the offset in trades jumps for closer passes, which no gate
measured, so `drone_separation_min_m ≥ 16` was added to the thresholds (the
amendment is recorded in the file; nothing existing was loosened). The fix
was made on those seeds' results, so none of them counts again: the script
now draws a fresh qualification and held-out set, and re-runs the four
failing seeds as tuning seeds.

### Second campaign (2026-11-13): not qualified

Fresh seeds, the same site and thresholds with the separation gate. 14 of
15 passed every gate. The step gate passed on all 15 (largest move in a
tick 10.6 m), and every measure in the table above stayed inside its
threshold (known 0.952 to 0.983, cells meeting the requirement 0.907 to
0.939, structure recall 0.943 to 0.988, ground RMSE 0.720 to 0.741 m,
conductor approach 14.9 to 25.0 m, no contact, every mission complete
between 321 and 478 s).

Seed 4451 failed `drone_separation_min_m`: 3.9 m. One drone had joined the
other's POI and stood 24 m from it; the mission then released both for home
on the same tick, on ways that crossed 2 ticks later. Closing at over
50 m/s, a separation flown in at 10 m/s with a 1 s look ahead cannot open
in time. That is the cost of not putting a drone somewhere, and it is paid
at planning time instead: flights planned in one mission run are checked
against each other, and one that would pass within 30 m of another is put
off a second at a time, up to 30 s (`departure_staggered` in the
decisions). On that seed the closest approach is now 18.9 m.

### Third campaign (2026-11-14): not qualified

Fresh seeds again. 14 of 15 passed every gate; seed 13127 failed
`drone_separation_min_m` (8.8 m), and so did three of the re-run tuning
seeds. Every other measure stayed inside its threshold on all 15. The
failures were three cases, and the changes they led to:

- **A drone waiting in the other's way.** Putting the later flight off does
  not help when the first flies past where the second is standing. The
  stagger now tries both: the flight is put off, or the flights it conflicts
  with are.
- **Two explorers crossing mid-scan**, closing at about 50 m/s. A 1 s look
  ahead suits a resolver that can put a drone aside at once, not one that
  flies its separation in. In a governed run deconfliction looks 4 s ahead
  (`separation_look_ahead_s`).
- **A resolver whose answer changes every tick.** The reference's resolver
  chooses afresh each tick between a sideways push, its mirror and a height
  change. Flown in at a rate, that is a target that keeps moving. A
  governed run asks one thing instead: the yielder takes a height offset
  that makes up the vertical separation, keeps the direction it has
  started in, and goes up when the ground leaves no room below
  (`vertical_separation` in `deconflict.rs`).
- **Two drones at one POI whose stations lie on opposite sides.** Each one's
  way home passes beside the other whichever leaves first, within two
  ticks. A drone joining a POI now holds 20 m above its hover height as
  well as 22 m short of it, so such a crossing is a height apart from the
  start.

All eight seeds that had failed any gate, in any campaign, pass every gate
after these.

### Fourth campaign

In progress when this was written; its results replace this line.

## Under the scheduler and under a fault

- The composed mission is byte-identical under the phased-lane executor and
  at 1 or 4 workers.
- With the mission lane at 1/2 Hz and the mapping lane at 1/3 Hz on the
  fixture's 1 s tick (`scheduled_v1`) it completes with no contact, and a
  run checkpointed while scanning, on a mission flight or on the way home
  finishes exactly as the uninterrupted one. The governor's state, flight
  delays and deconfliction offsets are in the checkpoint.
- `mapping_outage` (F2, [FAILURE_SCENARIOS.md](FAILURE_SCENARIOS.md)): one
  drone's mapping LiDAR out for 80 s. The mission completes later, with no
  contact, every POI done and every drone home.

Bounded causal jobs and restore with pending jobs (P4.10d–e) are not part of
this.

## Tests

`rust/argusnet-sim/tests/field_profile.rs` (15), the governor's and the
deconfliction stage's unit tests, `tests/test_field_qualification.py` (18).

## What it does not claim

- One real site. A second needs OSM and elevation sources fetched.
- The rig, LiDAR and camera models are assumed, not calibrated.
- Ground RMSE sits near 0.72 m in every run while the map requirement asks
  for 0.5 m vertical sigma and counts over 90% of cells as meeting it: the
  requirement is judged on the returns' own spread and a declared 0.05 m
  bias, and the pod georeferences by an estimated pose whose error it does
  not carry. Predicted and achieved quality disagree, and nothing flags it
  yet (the sensing guide's SR5d).
- No navigation-camera or gimbal fault, no pointing arbiter, no view
  planning, no hold or return response to a fault.
- No landing: home is a hover at the station.
