# field_v1 qualification: West Pleasant View (2026-11-14)

**Qualified**: 15 of 15 runs passed every gate.


| Measure | Threshold | Min | Median | Max |
| --- | --- | --- | --- | --- |
| `pois_complete_fraction` | min 1.0 | 1.000 | 1.000 | 1.000 |
| `home_radius_m` | max 15.0 | 8.204 | 11.674 | 14.592 |
| `world_contacts` | max 0 | 0.000 | 0.000 | 0.000 |
| `known_fraction` | min 0.9 | 0.944 | 0.960 | 0.980 |
| `quality_met_fraction` | min 0.9 | 0.903 | 0.912 | 0.938 |
| `structure_recall` | min 0.9 | 0.917 | 0.975 | 0.997 |
| `structure_precision` | min 0.9 | 0.965 | 0.970 | 0.974 |
| `ground_rmse_m` | max 1.0 | 0.719 | 0.731 | 0.742 |
| `wire_clearance_min_m` | min 10.0 | 12.498 | 21.935 | 25.017 |
| `drone_separation_min_m` | min 16.0 | 29.732 | 164.057 | 421.345 |
| `energy_reserve_margin_min_wh` | min 0.0 | 304.263 | 315.655 | 319.347 |
| `max_step_m` | max 12.0 | 8.339 | 9.349 | 9.520 |
| `mission_complete_s` | — | 302.250 | 363.000 | 508.000 |

Commit `1f6505c7e532c3a3aef013178ade61b4ec60d24e`, binary `609431714d06fdd2`, world `0fd4c2b70d34ba37645cb74a85714e8be3d216c09999e71e3d02a740a01e7714`, 600 s runs.

- A simulated qualification: the rig, LiDAR and camera models are assumed, not calibrated against hardware.
- Truth is read to score the map and wire clearance, never by the run.
- Station siting and the POIs' horizontal places come from the scenario generator.
- Localization starts from the true pose plus noise; landing is a hover at the station.

## How it got here

Four campaigns on West Pleasant View, each on seeds the one before had not
seen. A seed that informed a fix never counts again.

| Campaign | Counted seeds | Passed | Gate failed | What it led to |
| --- | --- | --- | --- | --- |
| 1 | 101..1010 by 101; 9001..9005 | 11 of 15 | `max_step_m` (4) | Deconfliction offsets flown in; a drone joining a POI stands off; a governor deadlock fixed; `drone_separation_min_m` added to the thresholds |
| 2 | 1117..11003; 9101..9105 | 14 of 15 | `drone_separation_min_m` (1) | Departures that would pass too close are staggered |
| 3 | 12007..22073; 9201..9205 | 14 of 15 | `drone_separation_min_m` (1) | Separation as one flown height rule with a 4 s look ahead; the stagger can send the waiting drone first; joiners hold 20 m above |
| 4 | 23131..33149; 9301..9305 | 15 of 15 | none | This record |

In every campaign every counted seed completed its mission with every POI
done and every drone home, with no world contact, and inside every map, wire
and energy threshold. What failed was how drones were kept apart.

The eight tuning seeds in this run (7, 42, and the six that failed a gate in
an earlier campaign) all pass every gate too. They are reported, not counted.

## What this is evidence of, and what it is not

It is evidence that, on this site, the composed profile completes its mission
and meets the declared thresholds on seeds it was not adjusted on. It is one
site: the profile's parameters were chosen while looking at it. The sensor
models are assumed. See [FIELD_PROFILE.md](../migration/FIELD_PROFILE.md).

`field-v1-qualification-linux-x86_64.json` is the campaign's `results.json`:
every run's measurements and gate verdicts, the thresholds as scored, and the
commit, binary and world identities. Replays are not kept; each run's replay
hash is, and each seed was run twice with identical bytes.

To reproduce:

```bash
cargo build --release --locked -p argusnet-cli
python scripts/qualify_field_profile.py --output runs/field-v1
```
