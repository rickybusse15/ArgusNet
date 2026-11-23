# Mission plan and the Plan workspace (first slice)

**Status:** shipped. `argusnet-rs plan --config SCENARIO --profile field_v3 --output
plan.json` works out and checks a mission before it is flown;
`argusnet-viewer --plan plan.json` shows it in the Plan workspace, read only. Code:
`rust/argusnet-sim/src/plan_report.rs`, `rust/argusnet-viewer/src/ui/plan.rs`.

Everything is worked out from what is declared: the map's extent and requirement, one
pass of the declared sensors, the energy model, where the pads and points are, and the
run's length. Nothing of the world is in it. A plan that checks out is not one that will
succeed. One that does not is refused, with the limit it meets.

| Check | Fails when |
| --- | --- |
| `drones` | There are none |
| `points_on_the_map` | A point is off it |
| `points_within_reach` | No drone can go, dwell and come home with its reserve |
| `vertical_accuracy` | One pass, or the declared bias, is worse than asked |
| `image_detail` | No flight gets an image as fine as asked |
| `density` | (A warning) more than one pass is needed |
| `time` | The least flying there is takes longer than the run |
| `scan_energy` | A drone's share of the scan takes more than it has over its reserve |

Within 80% of a limit a check passes as `warning`. What nothing declared says is
`unknown`, never `ok`. The command exits 2 when a check fails.

The viewer works nothing out: it shows the backend's checks, scan, points and drones, or
says how to make a plan. Writing one there, and checking a proposal, wait for the
canonical site, mission and trajectory contracts (S1).

## Against what was flown (2026-11-23)

`field_v3` on both sites, five tuning seeds each:

| | Planned | Flown |
| --- | --- | --- |
| Scan | at least 140 s | 294–361 s |
| Mission | at least about 200 s | 486–534 s, 120 s of it the follow up |
| Energy a drone | at least 14 Wh for the scan | 75–97 Wh for the mission |

The plan's figures are floors: no overlap, turn, transit, wire or hill is in them, and
legs are straight. A scan took 2.1 to 2.6 times its floor. So `time` and `scan_energy` find
what is out of reach, not what is comfortable.

## Not in this slice

- No preview in the world: pads and points are listed, not drawn from the plan.
- No airspace, zone or weather check; no contingency is planned.
- The panel's text and statuses are tested; the window was not looked at.

## Tests

`plan_report.rs` (4): each limit alone, warnings, what is unknown, the round trip.
`rust/argusnet-cli/tests/plan.rs`: a mission planned, and one too short refused.
`ui/plan.rs` (2): the backend's words are the panel's; what is not a plan says so.
