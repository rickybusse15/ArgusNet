# Inspection evidence (F7, first slice)

**Status:** shipped, opt in: `argusnet-rs sim --profile field_v3 --inspection-evidence`
(profile id `field_v3+evidence`). Code: `rust/argusnet-sim/src/evidence.rs`. Off,
nothing changes.

A point was inspected when a drone had hovered over it for its dwell, which says
nothing of what was seen. With a requirement it also asks for three views at least 80
degrees apart round it, from within 60 m, 15 to 75 degrees above level, with nothing
in the way. Once every dwell is done, drones with energy go round the points still
short, one viewpoint at a time, and then the map is followed up and the fleet goes
home.

| Verdict | When |
| --- | --- |
| `met` | The views asked for |
| `no_viewpoint` | No place left would add a view, as the map has it |
| `tries_spent` | 8 viewpoints flown to |
| `time_spent` | 120 s for all the points |
| `fleet_homeward` | No drone is left to go |

A point is never called met on fewer views, and never holds the mission past those
bounds. Reserves are kept as ever: a drone at its own gives its viewpoint up. Each
choice is a decision (`evidence_revisit`, `evidence_view`, `evidence_settled`); the
replay ends with `meta.inspection_evidence`.

A viewpoint is chosen on the drones' own map. Whether a view from it *shows* the point
is asked of the image taken there. Here that is the true world, as an image would be:
a ray from where the drone is to the point. Range, elevation and the way round are the
image's too, from where it was taken and not from where the drone believes it was. A view that shows nothing is not counted,
and that place is not tried again.

## Measured (2026-11-23)

`field_v3+evidence`, five tuning seeds on each site, three points each.

| | West Pleasant View | Montague City |
| --- | --- | --- |
| Points met | 13 of 15 | 15 of 15 |
| Unmet | 2, both `no_viewpoint` (0 and 2 views) | 0 |
| Viewpoints flown to for a point that was met | 3 | 3 (one 4) |
| Mission complete | 516–553 s, two still out at 600 s | 532–586 s |
| World contacts | 0 | 0 |

It costs about 20 to 60 s a mission, so it is not in `field_v3`, whose runs are 600 s.

## What it does not do

- The camera is taken to point at the subject: its field of regard, focus and exposure
  are not asked. Sharpness at range is not either, beyond the range itself.
- The subject is the top of what the map has at the point. A wall's face or an
  asset's side is not a subject yet.
- A dwell's own look straight down is no view.
- One drone a point at a time; viewpoints are the nearest first, not a planned round.
- Nothing is revisited across missions (with F6).

## Tests

`evidence.rs` (9): all round, a view judged from where it was taken, a hold of one stay, a view that shows nothing, seen from one side only, out
of time, no fleet, sharing, a requirement nothing could meet.
`tests/field_v3.rs`: the composed mission settles every point and resumes exactly on
the way to a viewpoint, at two points.
