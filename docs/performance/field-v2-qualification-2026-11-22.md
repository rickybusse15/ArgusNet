# field_v2 qualification: West Pleasant View (2026-11-22)

**Superseded** by the [two-site record](field-v2-qualification-2026-11-23.md):
`field_v2` was changed for what a second site showed, so this is evidence for a
build that no longer exists.

**Qualified**: 15 of 15 runs passed every gate.


| Measure | Threshold | Min | Median | Max |
| --- | --- | --- | --- | --- |
| `pois_complete_fraction` | min 1.0 | 1.000 | 1.000 | 1.000 |
| `home_radius_m` | max 15.0 | 8.520 | 11.327 | 14.154 |
| `world_contacts` | max 0 | 0.000 | 0.000 | 0.000 |
| `known_fraction` | min 0.9 | 0.975 | 0.989 | 0.993 |
| `quality_met_fraction` | min 0.9 | 0.902 | 0.905 | 0.918 |
| `structure_recall` | min 0.9 | 0.966 | 0.989 | 0.997 |
| `structure_precision` | min 0.9 | 0.966 | 0.969 | 0.973 |
| `ground_rmse_m` | max 1.0 | 0.103 | 0.105 | 0.108 |
| `map_error_over_claim` | max 1.5 | 1.177 | 1.201 | 1.244 |
| `map_error_within_2_sigma` | min 0.88 | 0.913 | 0.919 | 0.924 |
| `image_gsd_p90_m` | max 0.04 | 0.036 | 0.036 | 0.036 |
| `wire_clearance_min_m` | min 10.0 | 17.299 | 21.166 | 25.000 |
| `drone_separation_min_m` | min 16.0 | 22.592 | 29.434 | 183.092 |
| `energy_reserve_margin_min_wh` | min 0.0 | 298.876 | 312.679 | 320.553 |
| `max_step_m` | max 12.0 | 8.489 | 9.173 | 9.327 |
| `mission_complete_s` | — | 390.250 | 418.000 | 456.250 |

Commit `942e35e66378a8669e9db3ce21dbf6ed0103213d`, binary `8f757cc00ad04793`, world `a6b4fb6cd062c4a5d2df4c19a37ea10ea55e1cb96a7107e9f534cfc39d390d1b`, 600 s runs.

- A simulated qualification: the rig, LiDAR and camera models are assumed, not calibrated against hardware.
- Truth is read to score the map and wire clearance, never by the run.
- Station siting and the POIs' horizontal places come from the scenario generator.
- Localization starts from the true pose plus noise; landing is a hover at the station.

## How it got here

| Campaign | Counted seeds | Result |
| --- | --- | --- |
| 1 | 45329..55291; 9501..9505 | Stopped with five seeds scored, all passing, when code review found two imaging defects (a sharp-speed cap that used the wrong height, and images dropped over unseen ground). Counts for nothing; its seeds are not used again |
| 2 | 56393..66343; 9601..9605 | This record: 15 of 15 |

The thresholds were committed before the first seed of campaign 1 and have not
changed: `field_v1`'s gates, and three on the map's own claims and its images.
The three were set with two tuning seeds in view. The eight tuning seeds pass
every gate too; they are reported, not counted.

## What it is evidence of, and what it is not

On this site, `field_v2` finishes its mission with every POI done and every
drone home, touching nothing, with a map that meets its requirement, its 4 cm
image included, and with an error about 1.2 times what it claims.

- **The share of cells meeting the requirement sits just over its gate** (0.902
  to 0.918 against 0.90). It is the scan's own stopping rule, so it says the
  scan reached its end, not that there was margin.
- **It takes longer than `field_v1`**: median 418 s against 376 s, on different
  seeds. The images cost more than view planning saves.
- One site, one machine (macOS, arm64), assumed sensor models. The world ID
  differs from `field_v1`'s Linux record; the two records are not compared
  hash by hash.
- Faulted runs are not part of the campaign
  ([FAILURE_SCENARIOS.md](../migration/FAILURE_SCENARIOS.md) has them).

`field-v2-qualification-macos-arm64.json` is the campaign's `results.json`.

    cargo build --release --locked -p argusnet-cli
    python scripts/qualify_field_profile.py --output runs/field-v2 --profile field_v2
