# field_v2 qualification: two sites (2026-11-23)

**Qualified on both**: 15 of 15 unseen seeds pass every gate on Montague City,
Massachusetts, and 15 of 15 on West Pleasant View, Colorado, with one build and
the thresholds frozen on 2026-11-22, unchanged. It replaces the
[2026-11-22 record](field-v2-qualification-2026-11-22.md), whose build no
longer exists.

## Montague City

| Measure | Threshold | Min | Median | Max |
| --- | --- | --- | --- | --- |
| `pois_complete_fraction` | min 1.0 | 1.000 | 1.000 | 1.000 |
| `home_radius_m` | max 15.0 | 8.180 | 11.320 | 14.584 |
| `world_contacts` | max 0 | 0.000 | 0.000 | 0.000 |
| `known_fraction` | min 0.9 | 0.969 | 0.978 | 0.987 |
| `quality_met_fraction` | min 0.9 | 0.905 | 0.908 | 0.920 |
| `structure_recall` | min 0.9 | 0.933 | 0.990 | 1.000 |
| `structure_precision` | min 0.9 | 0.925 | 0.937 | 0.947 |
| `ground_rmse_m` | max 1.0 | 0.157 | 0.166 | 0.171 |
| `map_error_over_claim` | max 1.5 | 0.950 | 1.027 | 1.083 |
| `map_error_within_2_sigma` | min 0.88 | 0.945 | 0.948 | 0.962 |
| `image_gsd_p90_m` | max 0.04 | 0.036 | 0.036 | 0.036 |
| `wire_clearance_min_m` | min 10.0 | 20.261 | 85.085 | 92.629 |
| `drone_separation_min_m` | min 16.0 | 20.031 | 39.888 | 362.566 |
| `energy_reserve_margin_min_wh` | min 0.0 | 319.782 | 324.956 | 328.728 |
| `max_step_m` | max 12.0 | 8.902 | 9.255 | 9.325 |
| `mission_complete_s` | — | 355.000 | 389.750 | 444.500 |

Commit `c79d7cc5edc5c2f59ac3eb7765b9350d492e27d5`, binary `e807609b6a4d5c68`, world `0a2af0946f03e2a6ba2c175a50a795b1d6dd5f5111da72d0bceb5e1d89e9b775`, 600 s runs.

## West Pleasant View

| Measure | Threshold | Min | Median | Max |
| --- | --- | --- | --- | --- |
| `pois_complete_fraction` | min 1.0 | 1.000 | 1.000 | 1.000 |
| `home_radius_m` | max 15.0 | 8.807 | 11.187 | 14.144 |
| `world_contacts` | max 0 | 0.000 | 0.000 | 0.000 |
| `known_fraction` | min 0.9 | 0.980 | 0.989 | 0.996 |
| `quality_met_fraction` | min 0.9 | 0.901 | 0.908 | 0.929 |
| `structure_recall` | min 0.9 | 0.981 | 0.995 | 1.000 |
| `structure_precision` | min 0.9 | 0.963 | 0.969 | 0.972 |
| `ground_rmse_m` | max 1.0 | 0.092 | 0.094 | 0.097 |
| `map_error_over_claim` | max 1.5 | 0.806 | 0.825 | 0.850 |
| `map_error_within_2_sigma` | min 0.88 | 0.968 | 0.972 | 0.976 |
| `image_gsd_p90_m` | max 0.04 | 0.036 | 0.036 | 0.036 |
| `wire_clearance_min_m` | min 10.0 | 17.759 | 19.741 | 25.000 |
| `drone_separation_min_m` | min 16.0 | 22.316 | 29.732 | 399.423 |
| `energy_reserve_margin_min_wh` | min 0.0 | 308.680 | 312.999 | 318.186 |
| `max_step_m` | max 12.0 | 8.067 | 9.255 | 9.327 |
| `mission_complete_s` | — | 373.750 | 401.000 | 460.500 |

Commit `c79d7cc5edc5c2f59ac3eb7765b9350d492e27d5`, binary `e807609b6a4d5c68`, world `a6b4fb6cd062c4a5d2df4c19a37ea10ea55e1cb96a7107e9f534cfc39d390d1b`, 600 s runs.

Both:

- A simulated qualification: the rig, LiDAR and camera models are assumed, not
  calibrated against hardware.
- Truth is read to score the map and wire clearance, never by the run.
- Station siting and the POIs' horizontal places come from the scenario
  generator.
- Localization starts from the true pose plus noise; landing is a hover at the
  station.

## How it got here

Montague City was chosen before any run was made on it, for its differences: a
river, 14 power lines instead of one, and 55 m of relief instead of 39 m
([ATTRIBUTION.md](../../tests/fixtures/real/montague_city/ATTRIBUTION.md)).

| Step | Seeds | Result |
| --- | --- | --- |
| The qualified build, unchanged, on the new site | tuning 7, 42, 4451 | None passed. All three claimed 1.64 to 1.73 times less map error than they had (85% of cells within two sigma); 4451 also came 8.2 m from a wire |
| Plane heights; separation that keeps wires | 8 tuning seeds | All pass |
| Campaign 1 | 67421..77339; 9701..9705 | **14 of 15: not qualified.** Seed 74027 did not finish: a strip between two lines held its scan open until 499 s |
| Wires the cruise height clears are flown over | the 8, and 74027 | All pass, on both sites |
| Campaign 2 (this record) | 100403..110311; 10001..10005 | 15 of 15 |
| West Pleasant View (this record) | 89417..99317; 9901..9905 | 15 of 15. Its third set was stopped before any seed was scored, when campaign 1 failed |

The tuning seeds pass every gate on both sites; they are reported, not counted.
What changed: [MAP_REPORT.md](../migration/MAP_REPORT.md),
[WIRE_CORRIDORS.md](../migration/WIRE_CORRIDORS.md).

## What it is evidence of, and what it is not

- **The profile carries to a second site, and did not before.** Three defects
  one site had never shown were found by flying another. Nothing says a third
  would find none.
- **The map's claim now holds on both**: error over claim 0.95 to 1.08 on
  Montague City. On West Pleasant View it is under 1: the map claims somewhat
  more error than it has.
- **The share of cells meeting the requirement still sits just over its gate**:
  it is the scan's own stopping rule, so it is no margin.
- Montague City's extract maps no trees, so nothing here tests canopy over
  relief.
- One machine (macOS, arm64). Faulted runs are not part of the campaign.

`field-v2-qualification-<site>-2026-11-23-macos-arm64.json` are the campaigns'
`results.json`.

    cargo build --release --locked -p argusnet-cli
    python scripts/qualify_field_profile.py --output runs/field-v2 --profile field_v2
    python scripts/qualify_field_profile.py --output runs/field-v2-montague \
        --profile field_v2 --site tests/fixtures/real/montague_city
