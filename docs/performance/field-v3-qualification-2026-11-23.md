# `field_v3` qualified on two sites (2026-11-23)

15 of 15 unseen seeds on West Pleasant View and 15 of 15 on Montague City, on one
build (`e44879d`), against [limits](../contracts/field_v3.thresholds.json) frozen
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
| Cells meeting the requirement | at least 0.90 | 0.908, 0.947, 0.953 | 0.959, 0.970, 0.975 |
| Ground RMSE, m | at most 1.0 | 0.086, 0.090, 0.095 | 0.150, 0.154, 0.171 |
| Error over claim | at most 1.5 | 0.67, 0.69, 0.74 | 0.83, 0.87, 0.95 |
| Map error within 2 sigma | at least 0.88 | 0.981, 0.985, 0.989 | 0.965, 0.969, 0.976 |
| Position error within 2 sigma | at least 0.88 | 0.909, 0.958, 0.978 | 0.947, 0.963, 0.971 |
| Closest to a conductor, m | at least 10 | 19.0, 25.0, 25.0 | 11.8, 73.2, 94.7 |
| Closest drones, m | at least 16 | 17.8, 33.3, 351 | 19.5, 29.8, 303 |
| Longest step in a tick, m | at most 12 | 8.0, 8.5, 8.5 | 8.3, 8.5, 8.5 |
| Furthest from home after, m | at most 15 | 9.5, 12.2, 14.0 | 8.8, 12.7, 14.8 |
| Structure recall, precision | at least 0.9 each | 0.993, 0.965 (lowest) | 0.984, 0.925 (lowest) |
| World contacts | 0 | 0 | 0 |
| Mission complete, s | (600 s run) | 461, 520, 565 | 457, 508, 547 |

Against `field_v2` the maps are better (0.947 and 0.970 of cells against about
0.91) and the missions about 115 s longer: the follow up's time.

## What it took

- **The first campaign failed**, 29 of 30: seed 119107 on West Pleasant View moved
  a drone 17 m in one tick. An explorer runs up to three steps ahead while its
  drone climbs from its pad, and the handover put the drone there. It is flown
  there now. The seed is a tuning seed, both sites' first sets are retired, and
  these are second sets. `field_v2` has the same handover, and its own qualifying
  seeds did not meet it.
- Before any seed was scored, three faults of the 3D routes were found and fixed
  on tuning seeds and by review: a spiral descent at the pad, planning to the bare
  wire clearance, and a reroute flag that routed flights never set.

## What it does not say

- Every sensor number is assumed, and the filter is told the true ones. No
  recorded flight has been through it.
- 11.8 m from a wire on Montague City is inside the 10 m limit by 1.8 m.
- `standalone` GNSS is not this profile and meets no cell's requirement. A
  three-minute loss of GNSS takes both drones to about 700 s, past this run.
- One machine (macOS arm64); the two sites are mild: low houses, gentle ground.
