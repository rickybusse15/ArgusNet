# LiDAR returns (SR3)

`--lidar-returns`, with `--rig-profile`, models what the pod LiDAR actually gets
back (`rust/argusnet-sim/src/lidar_returns.rs`). A rig profile file can declare
its own model under the LiDAR's `returns`. Without either, every ray that meets
a surface in range returns, at the capture's instant, with Gaussian range noise
only, as before: 14 of 14 canonical replay hashes are unchanged and every
lifecycle check is equal.

This follows section 5 of the
[sensing realism guide](../SENSING_REALISM_IMPLEMENTATION_GUIDE.md).

## The model

| Part | What it does | Default (assumed) |
| --- | --- | --- |
| Per-firing time | Ray `k` of `n` fires `k / n` of the scan period after the capture, from where the aircraft has flown to; the estimate is deskewed with the same motion | Period = the capture interval |
| Admission | Detected with a logistic chance in returned power, reflectivity × cos(incidence) / range², 50% for a 10%-reflective target facing the beam at the reference range | Reference 250 m, steepness 4 |
| Reflectivity | Per surface kind | Terrain 0.25, structure 0.3, canopy 0.4, trunk 0.15 |
| Incidence | From the true ground's normal; structures and crowns are taken as facing the beam | |
| Range bias | Every return, along its ray | 0.02 m |
| Range quantization | Reported ranges on a grid | 0.005 m |
| Angular noise | In where the estimator believes the ray pointed, each axis | 0.5 mrad |
| Dropout | A detected return lost anyway | 0.2% |
| Footprint | Range × divergence, for analysis | 2 mrad |

Range noise stays the LiDAR's own (`range_noise_m`). The new draws come from a
stream of their own, so the existing draws do not shift with the model on.

**Motion across a scan** follows the aircraft's horizontal velocity, capped at
60 m/s. Carrying a climb or a descent through a whole capture put ray origins
underground at take-off and landing. That returned "ground" up to a kilometre
down, and pushed the fixture's ground RMSE from 0.75 m to 17 m. Attitude is held
over one capture, and the estimate knows the true velocity, so deskew residuals
show only when attitude or the scan period is wrong.

## Validation

| Experiment (guide section 10) | Test | Result |
| --- | --- | --- |
| Surveyed plane, nadir and oblique rays | `modelled_rays_fire_across_the_capture_and_still_land_on_the_plane` | Every ray lands on the plane at 30 m/s; each lands where the untimed ray does, moved by speed × offset |
| Dark and bright targets, range edge, grazing | `return_fractions_follow_the_declared_curve` | Observed returns within 4 sigma of the model's expectation; a 5% surface returns under half what a 25% one does |
| Separate errors | `a_range_bias_lowers_the_map_and_ranges_sit_on_their_grid` | A 0.5 m bias lowers a flat map by 0.4–0.5 m; ranges sit on a 0.25 m grid to 1 µm |
| Climb, descent and sampled jumps | `a_climb_or_a_jump_never_puts_a_ray_underground` | No return off the plane |
| Checkpoint during a scan | `modelled_returns_run_deterministically_and_cost_returns` | Deterministic; resume from tick 150 equals the straight run |

## Results

The scan fixture, `efficient_v1`, provisional rig, 400 s:

| LiDAR | Returns | Map known | Ground RMSE | Worst ground error |
| --- | ---: | ---: | ---: | ---: |
| Exact | 20,883 | 99.3% | 0.75 m | 6.1 m |
| Modelled | 22,318 | 100.0% | 0.77 m | 7.9 m |

The two runs fly differently once their maps differ, so the totals are not like
for like: the share of rays returning was 10.0% exact and 11.0% modelled, because
the modelled run flew where more rays met ground in range. The like-for-like
check is the capture test above, which holds the flight fixed.

West Pleasant View, Colorado, 300 s, `contingency_v1`, `flown_v1`, provisional
rig, seeds 7 and 42:

| Run | LiDAR | Rays returning | Map known | Ground RMSE | Structure recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Rig only | Exact | 15.5%, 14.9% | 95.6%, 95.8% | 0.74, 0.72 m | 95.7%, 95.5% |
| Rig only | Modelled | 15.1%, 15.0% | 90.3%, 93.5% | 0.76, 0.75 m | 91.2%, 90.8% |
| W4 wires | Exact | 15.0%, 15.1% | 87.1%, 83.7% | 0.72, 0.72 m | 93.7%, 89.9% |
| W4 wires | Modelled | 14.8%, 14.6% | 97.5%, 94.2% | 0.74, 0.74 m | 95.9%, 96.0% |

With the wire stack on and returns modelled, no platform came within 10 m of a
conductor (closest 17.1 and 23.2 m). On the rig alone, modelled returns cost 2
to 5 points of map, 0.02 to 0.03 m of ground RMSE and about 5 points of
structure recall: dark, far and grazing surfaces are missed. With W4 the flights
differ enough between the two runs that the map figures move the other way, by
more than the sensor could explain, so two seeds cannot rank them.

## Not modelled

- False returns and multiple returns per pulse. The guide asks for an actual
  model of each, not duplicated or invented points.
- Intensity, wavelength and ambient light, and surface normals on structures.
- Attitude change within a capture, and velocity error in the deskew.
- Every figure is assumed until a device is chosen, and tagged so.
