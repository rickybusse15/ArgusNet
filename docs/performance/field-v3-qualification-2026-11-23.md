# `field_v3` qualified on two sites (2026-11-23)

15 of 15 unseen seeds on West Pleasant View and 15 of 15 on Montague City, on one
build (`4ec476d`), against [limits](../contracts/field_v3.thresholds.json) frozen
before any seed was run. Every replay was run twice and was byte-identical.
Results: [West Pleasant View](field-v3-qualification-west-pleasant-view-2026-11-23-macos-arm64.json),
[Montague City](field-v3-qualification-montague-city-2026-11-23-macos-arm64.json).

What it is: `field_v2` flown on a survey-grade receiver's estimated positions,
with cells placed to 0.5 m, a lost drone's hold and return, separation over the
believed surface, a follow up on the map, mission flights routed in three
dimensions and flown handovers ([FIELD_PROFILE.md](../migration/FIELD_PROFILE.md)).

## Measured (lowest, median, highest of the 15 counted seeds)

| | Limit | West Pleasant View | Montague City |
| --- | --- | --- | --- |
| Cells meeting the requirement | at least 0.90 | 0.937, 0.948, 0.954 | 0.958, 0.973, 0.977 |
| Ground RMSE, m | at most 1.0 | 0.087, 0.090, 0.094 | 0.150, 0.152, 0.164 |
| Error over claim | at most 1.5 | 0.66, 0.70, 0.73 | 0.84, 0.86, 0.92 |
| Map error within 2 sigma | at least 0.88 | 0.983, 0.986, 0.988 | 0.964, 0.970, 0.974 |
| Position error within 2 sigma | at least 0.88 | 0.930, 0.952, 0.973 | 0.912, 0.956, 0.977 |
| Closest to a conductor, m | at least 10 | 21.8, 25.0, 25.0 | 19.3, 61.2, 91.8 |
| Closest drones, m | at least 16 | 20.9, 28.6, 407.7 | 20.3, 129.5, 303.0 |
| Longest step in a tick, m | at most 12 | 8.0, 8.5, 10.3 | 8.3, 8.5, 8.6 |
| Furthest from home after, m | at most 15 | 8.1, 12.0, 14.7 | 8.2, 10.4, 13.0 |
| Structure recall, precision | at least 0.9 each | 0.992, 0.966 (lowest) | 0.987, 0.925 (lowest) |
| World contacts | 0 | 0 | 0 |
| Mission complete, s | (600 s run) | 488, 530, 580 | 449, 504, 528 |

Against `field_v2` the maps are better (0.948 and 0.973 of cells against about
0.91) and the missions about 115 s longer: the follow up's time.

## What it took

Four campaigns, each on seeds nothing had been tried on.

1. **Failed**, 29 of 30. Seed 119107 on West Pleasant View moved a drone 17 m in one
   tick. An explorer runs up to three steps ahead while its drone climbs from its pad,
   and the handover put the drone there. It is flown there now. `field_v2` has the same
   handover, and its own qualifying seeds did not meet it.
2. **Passed**, 30 of 30, on a build that no longer exists. Review then found that a
   steep leg of a route could step over a wire between two samples. Fixed, 13 of the
   campaign's 46 replays changed, so it describes code that no longer runs.
3. **Failed**, 29 of 30. Seed 158701 on West Pleasant View: corners a route kept about
   its pad, for the way down past a wire's margin, had a drone 13.9 m from it, and so
   home, and then 15.5 m. Run on that build, three more of the retired seeds went by or
   round their pads. A route comes in over its end now.
4. **Passed**, 30 of 30: this record. Every retired and tuning seed was run on its
   build first, 164 runs, and passed.

A failed seed is a tuning seed from then on, and both sites' sets are retired with it.
Before the first campaign, three faults of the routes were found and fixed on tuning
seeds and by review: a spiral descent at the pad, planning to the bare wire clearance,
and a reroute flag that routed flights never set.

## What it does not say

- Every sensor number is assumed, and the filter is told the true ones. No
  recorded flight has been through it.
- The closest pass to a wire counted is 19.3 m, against 10 m.
- `standalone` GNSS is not this profile and meets no cell's requirement. A
  three-minute loss of GNSS takes both drones to about 700 s, past this run.
- One machine (macOS arm64); the two sites are mild: low houses, gentle ground.
