# Navigation: the estimator in the loop

**Status:** two slices, on plane heights and wire-limited separation, behind `--navigation {rtk,standalone,FILE}` in
`--world-knowledge none` runs and with a profile. Off, nothing changes. Code:
`rust/argusnet-localization/src/nav.rs` (the filter),
`rust/argusnet-sim/src/navigation.rs` (the simulated sensors and the placing
of scans), `quality.rs` and `belief_world.rs` (what the map keeps of it).

## The defect it closes

Every return was placed from the aircraft's true position. The map had no
position error, so its quality could not have any either: with a position
error injected, ground RMSE went from 0.12 m to 1.91 m while the share of cells
meeting the requirement stayed at 0.91.

## How it works

```text
truth ─► IMU, GNSS, heading (noise, biases, shared errors) ─► filter ─► pose + covariance
  └────► sensing from the true pose ──────────────────────────────────► placed by the pose
```

- **The filter** is an extended Kalman filter of 17 states: position, velocity,
  heading, accelerometer and gyroscope biases, the attitude sensor's tilt error,
  and the parts of the GNSS and heading errors that successive readings share. It takes measurements and
  nothing else. The shared errors are states because averaging fixes does not
  remove them: without them it would soon claim millimetres.
- **The sensors** draw from streams keyed on the run's seed, the aircraft and
  the tick, so an outage changes no draw after it, and their state is in stage
  6's checkpoint.
- **A scan** is made from the true pose, then moved as a body to the estimated
  one: pod LiDAR, navigation camera, wire points and lines, image poses. An
  aircraft with no fix yet has nowhere to put a return, so its scan is not
  evidence.
- **Each return** has a vertical and a horizontal sigma from its pose: the
  position's, with the tilt and heading errors acting over the offset to it.
- **Each cell** keeps the weighted mean of its accepted returns' pose sigmas.
  That is the most the weighted mean height's own sigma can be, however the
  returns' pose errors are related; no count reduces it. A horizontal error
  reaches the height through the ground's slope. A well-placed return
  outweighs a badly placed one, in the cell's height too.

## What the map says

| Reason | Meaning | Planner response |
| --- | --- | --- |
| `pose_uncertain` | Its poses alone leave it short; some aircraft now knows better | Revisit |
| `pose_limited` | And none does | None: unreachable, reported, never met |

`max_horizontal_sigma_m` in a quality requirement asks for where the cell's
returns are; it needs navigation. The replay gains `frames[].navigation` (each
aircraft's estimate and sigmas), `pose_uncertain_cells` and
`pose_limited_cells`; the map report gains `horizontal_sigma_m`. A run with a
profile and navigation is recorded as `field_v2+navigation`: it is not the
profile, which is qualified without it.

`gnss_outage` (F2) takes an aircraft's fixes away for a time.

## When it is lost

With `fault_return.lost_position` (on in `field_v3`) a drone that knows where it
is to worse than `hold_sigma_m` (5 m, half a cell: past it a capture is refused,
so flying on maps nothing) brakes to a hold where it believes it is, above what
may be there, as one whose navigation camera is out does. It goes on when it
knows again. `home_after_s` (240 s) after its last fix it goes home for good, at
the blind speed. A drone let go from a hold is flown back to where its
controller ran on to while it held, at 1.25 times the speed asked; before, it
was put there, up to three steps in one tick (the camera's hold did the same).

Measured on `field_v3`, the seven tuning seeds on each site, no GNSS
60–240 s:

| | Both drones lose it | One does |
| --- | --- | --- |
| Complete in 600 s | 13 of 14 | 14 of 14 |
| at | 514–574 s | 446–569 s |
| Every gate passed | 13 of 14 | 14 of 14 |
| Closest drones | 20.2 m | 20.8 m |

Flying on completed as many (13 of 14), about 20 s later on average. The hold
is for what is under the drone, not for time.

## Is the filter honest?

Over 400 seeded runs of a survey each, the position error's squared length in
the filter's own covariance averages 3, within the 0.5 declared before the
test was run: for `rtk` and `standalone`, with GNSS throughout and after 30 s
without it. Gravity leaking through the attitude sensor's tilt error is in both
the simulated IMU and the filter; left out, 180 s without GNSS cost 5 m at worst
in place of 60 to 90 m.

## Flying the estimate

An autopilot holds its *estimate* on its path, so the true path is off by the
error. Positions in the run are where each aircraft believes it is, which is
all its controller, overrides, separation and safety monitor have. Physics,
sensing and the replay's `nodes` are where it is (`FlownPositions`): last
tick's estimate error, approached at 4 m/s at most, so an error it has only now
learned of is flown off and never jumped. Over its launch pad it is held to the
pad, as a precision landing is. `correction_mps` absent flies the truth.

How well it knows is its own to use:

- mission flights clear the believed surface by 2 sigma more, over a strip
  2 sigma wider, and wires and poles likewise (at most 40 m sideways, 20 m up);
- drones keep apart by 2 sigma more, both drones' combined;
- a wire track's sigma is at least the mean pose sigma of its points, so its
  clearance (3 sigma over the base) grows with how lost the aircraft that saw it
  was;
- a mapping capture placed worse than half a cell is **refused**
  (`misplaced_scans`): it cannot say which cell it saw. The navigation camera
  is not: it looks where the aircraft is going, which is off by the same error.

## Measured (2026-11-23, `field_v2` on both sites, 600 s)

The tuning seeds (7, 42, 707, 1010, 4451, 9002, 13127) on West Pleasant View
and Montague City, on this branch: plane heights, wire-limited separation and
the estimate together. Ranges over the 14 runs; where the sites differ, West
Pleasant View / Montague City.

| | `rtk` | `standalone` | `rtk`, no GNSS 60–240 s, both drones |
| --- | --- | --- | --- |
| Mission complete in 600 s | 14 of 14 | 14 of 14 | 13 of 14 |
| at | 372–455 s | 298–381 s | 542–596 s |
| Cells meeting the requirement | 0.90–0.92 | **0.00** | 0.90–0.92 |
| Ground RMSE | 0.09–0.10 / 0.16–0.17 m | 0.89–2.44 m | 0.09–0.14 / 0.16–0.22 m |
| Error over claim (RMS) | 0.74–0.80 / 0.94–1.02 | 0.43–1.16 | 0.70–0.77 / 0.84–0.94 |
| Closest to a conductor | 17.1 m | 22.1 m | 18.5 m |
| Closest drones | 21.0 m | 21.7 m | 22.6 m |
| World contacts | 0 | 0 | 0 |
| Worst position error, one axis | 0.08 m | 4.4 m | 77 m |
| Position within 2 sigma, per axis | 0.94–0.97 | 0.96–1.00 | 0.95–0.97 |

- With a survey-grade receiver every run passes every gate of `field_v2`.
- With an uncorrected receiver no cell meets a 0.5 m vertical requirement, and
  the map says so, and that a revisit would not help. The mission still
  completes: an unreachable cell is resolved, as a bias-limited one is, so the
  scan gate opens. The qualification's met-fraction gate is what fails.
- In the outage the drones drift up to 77 m. Refusing what they then map keeps
  the ground where it is. It costs about 150 s, and one run (Montague City,
  707) is still out at 600 s: a lost drone flies on while it maps nothing.
- Zero contacts is these sites' doing: the survey flies 60 m over low houses. A
  test puts a 120 m vertical error on the scan fixture's hills, and it hits them.
- Navigation off changes nothing: 47 of 47 `field_v2` replays on the two sites
  and 14 of 14 canonical ones are byte-identical to the base.

## Plane heights under an estimated pose

A cell's plane ([FIELD_PROFILE.md](FIELD_PROFILE.md)) is fitted with each
return weighted as its pose is known, so a good pass over a bad one sets it.
The weighted sums stand for `(sum w)^2 / sum w^2` returns of equal weight. The
centre height is a weighted sum of the returns' heights, and Cauchy-Schwarz
bounds what their pose errors can do to it however they are related: where the
returns lie to one side and the plane reaches out to the centre, the claim
grows by as much. A test moves every return by its own sigma, in the worst
sense, and the claim covers it.

Separation keeps its peers' wires by the same 2 sigma as flights do, sideways
and in height, and the peer that makes up the rest of a separation makes up
the widened one.

## What it does not claim

- The field profiles ask for no horizontal sigma, so a flat site hides a
  sideways error of up to half a cell.
- Structures and poles carry no pose sigma of their own; wires do.
- Lanes and explorers keep their own height: only mission flights, wires and
  separation widen.
- A lost aircraft holds and goes home only where that is asked for
  (`fault_return.lost_position`, on in `field_v3`); otherwise it flies on.
- Control is perfect but for the error: no gusts, no lag.
- Bearing observations of targets, and the grid localizer, are unchanged. The
  in-process viewer's live snapshot shows believed positions; the replay, true.
- The filter is told the sensors' true noise. No outliers, multipath jumps,
  spoofing, clock error or lever arms between sensors; the GNSS velocity is
  white; the aircraft is level.
- Every sensor number is assumed. No filter has seen a recorded flight.

## Tests

`nav.rs` (7), `navigation.rs` (10), `quality.rs`, `belief_world.rs`,
`wire_governor.rs`, `deconflict.rs` and `wires.rs` (9), and
`rust/argusnet-sim/tests/navigation.rs` (7). Ten mutants of the first slice
each fail at least one.
