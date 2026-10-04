# Scenario comparison: intercept_stress

- Seeds: 30 paired (qualification)
- Code: `b914086717e4bb477df26b91f3022b15c28ba53e`
- Runtime binary sha256: `74a7c58f14d17de3…`
- World: procedural terrain, preset alpine, seed 701002222 (procedural-landscape-v1); environment procedural-1.75

## Variants

| Variant | Arguments | Runs | Failed | Outside envelope | Wall s (median) | Tick ms (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | `(scenario defaults)` | 30 | 0 | 30 | 0.525 | 0.41 |
| five-drones | `--drone-count 5` | 30 | 0 | 30 | 0.6 | 0.511 |

## Metrics

### `localisation_rmse_m` (lower is better)

Root-mean-square distance in metres between each fused track and the truth it was first paired with, over every frame the track exists.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 25.61 | 22.13 | 9.796 | 14.61 | 53.41 |
| five-drones | 30 | 25.04 | 21.64 | 10.51 | 12.08 | 53.66 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | -0.5698 | -1.452 to 0.3121 | 16 / 14 / 0 | no detectable difference |

### `track_continuity_mean` (higher is better)

Mean over targets of the fraction of steps in which the target has at least one accepted bearing observation.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0.6287 | 0.6326 | 0.06033 | 0.4855 | 0.7327 |
| five-drones | 30 | 0.7471 | 0.7507 | 0.03528 | 0.6676 | 0.8255 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | 0.1184 | 0.1064 to 0.1303 | 30 / 0 / 0 | better |

### `covariance_reduction_mean` (higher is better)

Mean over tracks of (trace(P_first) - trace(P_last update)) / trace(P_first), the 3x3 position block; tracks qualify only with enough updates (D-1).

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0.9 | 0.9351 | 0.09513 | 0.5428 | 0.9859 |
| five-drones | 30 | 0.8982 | 0.9422 | 0.1187 | 0.3886 | 0.9874 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | -0.001833 | -0.04633 to 0.04266 | 22 / 8 / 0 | no detectable difference |

### `time_to_reacquire_mean_s` (lower is better)

Mean duration in seconds of track-loss events (stale steps > 0 until 0 again); an event never recovered counts to the end of the run.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0.5742 | 0.563 | 0.0692 | 0.4862 | 0.7794 |
| five-drones | 30 | 0.5329 | 0.5202 | 0.05945 | 0.4363 | 0.65 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | -0.0413 | -0.05787 to -0.02473 | 24 / 6 / 0 | better |

### `mission_completion_rate` (higher is better)

Required objectives met over required objectives (1.0 when none are required).

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 1 | 1 | 0 | 1 | 1 |
| five-drones | 30 | 1 | 1 | 0 | 1 | 1 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | 0 | 0 to 0 | 0 / 0 / 30 | no detectable difference |

### `energy_reserve_min` (higher is better)

Lowest battery fraction left on any drone at the end of the run.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0.85 | 0.85 | 6.981e-17 | 0.85 | 0.85 |
| five-drones | 30 | 0.85 | 0.85 | 6.981e-17 | 0.85 | 0.85 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | 0 | 0 to 0 | 0 / 0 / 30 | no detectable difference |

### `false_handoff_rate` (lower is better)

Fraction of track handoffs that went to the wrong target.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0 | 0 | 0 | 0 | 0 |
| five-drones | 30 | 0 | 0 | 0 | 0 | 0 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | 0 | 0 to 0 | 0 / 0 / 30 | no detectable difference |

### `safety_override_count`

Times the safety monitor overrode a commanded motion. Context-dependent: fewer can mean a safer plan or a less exercised monitor.

| Variant | n | Mean | Median | SD | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| three-drones | 30 | 0 | 0 | 0 | 0 | 0 |
| five-drones | 30 | 0 | 0 | 0 | 0 | 0 |

Paired against **three-drones** (variant − baseline):

| Variant | Pairs | Mean diff | 95% interval | Improved / worsened / tied | Verdict |
| --- | --- | --- | --- | --- | --- |
| five-drones | 30 | 0 | 0 to 0 | – | not scored |

