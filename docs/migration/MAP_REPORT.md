# Map report: asked against achieved

**Status:** shipped. `argusnet-rs sim --map-report FILE` writes it for any run
with a quality requirement; `--profile field_v2` also records it as
`meta.map_report`. Code: `rust/argusnet-sim/src/map_report.rs`. It is computed
from the finished map when the run ends and nothing a platform does depends on
it; without the flag and the profile nothing changes.

The sensing guide (§8.5) asks for predicted against achieved quality. The
replay's counts say how much of the map is done; this says whether the flight
did what its sensors promised, whether the map's own error can be believed, and
what is left and why.

| Part | What it says |
| --- | --- |
| `asked` | The quality requirement |
| `predicted` | One nadir pass at the survey height and the explorers' speed: swath, accepted returns per m², vertical sigma, image sample distance, from the declared sensors alone |
| `achieved` | Counts by reason; 10th, 50th and 90th percentile density, sigma and sample distance; the median density in passes' worth |
| `calibration` | The map's claimed sigma against its true error. `normalized_rms` is 1 for an honest map, above it for one that claims more than it knows |
| `missed` | Cells not met by the map's reason (`unseen`, `sparse`, …) and what the true world has there: a structure in the cell or beside it, a crown over it, the map's edge, or open ground |
| `areas` | The ten largest connected areas flying could still fix, with bounds |

`calibration` and the causes read the true world. They are evaluation.

## Measured (2026-11-22, West Pleasant View)

| | v1, seed 7 | v1, seed 4451 | v2, seed 7 | v2, seed 4451 |
| --- | --- | --- | --- | --- |
| Predicted returns per m², one pass | 0.165 | 0.165 | 0.196 | 0.196 |
| Median achieved | 0.29 | 0.31 | 0.33 | 0.38 |
| In passes' worth | 1.8 | 1.9 | 1.7 | 1.9 |
| RMS error, m | 0.114 | 0.115 | 0.107 | 0.105 |
| RMS claimed sigma, m | 0.099 | 0.093 | 0.089 | 0.087 |
| Error over claim | 1.17 | 1.25 | 1.19 | 1.20 |
| Within 2 sigma | 92% | 91% | 92% | 92% |

- **The map is mildly overconfident**: its error is about 1.2 times its claim,
  with 92% of cells within two sigma rather than 95%.
- **It reads about 2 cm low** everywhere, less than the 5 cm bias the
  requirement declares.
- **Most unseen cells in a v2 map are a building's**: 469 of 585 on seed 7 have
  a structure in or beside them. Flying longer would not fix those; 71 are open
  ground.
- The claimed sigma's per-return noise is still the legacy scanner's 0.3 m,
  not the rig's 0.05 m. The map is overconfident despite that, because returns
  scatter over a sloping cell.

## Plane heights (2026-11-23, what a second site showed)

On Montague City, 55 m of relief with river banks, the same report read 1.64 to
1.73 times the claim, with 85% of cells within two sigma. A cell's height was
the mean of its accepted returns, and those fall 0.8 m from its centre at the
median: on a slope that alone is the error, and the mean's sigma says nothing
of it.

With `--plane-heights` (on in `field_v2`) a cell keeps where its accepted
returns fell, and:

- **Its height is a plane's at its centre.** `plane_fit` in
  `argusnet_mapping::quality`; the published ground uses it too. No plane is
  fitted to fewer than four returns, to a line of them, or where it would say
  less of the centre than one return there; the mean stands in those.
- **Its claim is that plane's.** The scatter about the plane over what the fit
  leaves at the centre, and the ground that is no plane: what the returns
  scatter beyond their own noise, times 1.25 (180/144, the squared error of a
  plane at the centre of an evenly sampled ridge or valley over the variance
  it leaves). No count of returns reduces that part. The residual does not say
  what shape it is: that ratio is 2.5 for a bowl or a dome, whose error is then
  1.4 times the claim, nothing for a saddle or a twist, and 0.90 on the mean of
  isotropic ground. 1.25 is for the mix, not the worst.
- **A cell with no plane claims its spread whole**, or the relief its
  neighbours show if that is more: one return has no spread.
- **A return's noise is the rig's own LiDAR's** (0.05 m), not the legacy
  scanner's 0.3 m, which the old claim needed to cover the slope.

Eight tuning seeds each, on the build that was campaigned:

| | Montague City, before | after | West Pleasant View, before | after |
| --- | --- | --- | --- | --- |
| Error over claim | 1.64–1.73 (3 seeds) | 1.00–1.07 | 1.19–1.20 (2 seeds) | 0.81–0.86 |
| Within 2 sigma | 85% | 94–95% | 92% | 97–98% |
| Ground RMSE, m | 0.19–0.21 | 0.16–0.18 | 0.10–0.11 | 0.092–0.096 |

On the flatter site the map now claims somewhat more error than it has. The
constants are the geometry's and the sensor's: none was set from these numbers.

## What it does not do

- The report itself is written when the run ends. What it would name as left to
  fix is followed up in flight (below), from the map and not from the report's
  evaluation parts, which read the true world.
- `predicted` is flat ground, nadir, one pass. It is not a coverage plan.
- A cell's error is its height less the true ground at its centre. Without
  plane heights that includes where in the cell the returns fell.
- Rough ground is called `sparse` when its claim exceeds the requirement, though
  flying more does not smooth it: fewer than 0.3% of Montague City's cells
  scatter that much.
- The true platform pose still places every return: no part of the claim is a
  position error, and none of the measured error is.

## Followed up in flight (`follow_up`, on in `field_v3`)

The scan ends at its gate, 0.9 of the map. Once every point is inspected the
fleet then stays out for what still falls short and flying could fix. Each
drone with energy goes on exploring, its goals the view planner's, until one of:

| Reason | Meaning |
| --- | --- |
| `little_left` | At most 2% of cells are short and reachable |
| `little_gained` | Under 20 cells a drone-minute brought to the requirement over 20 s |
| `time_up` | 120 s |
| `fleet_homeward` | Every drone is going home |

Both ends are decisions in the journal, with the counts. A drone at its reserve
goes home as ever. Measured on the seven tuning seeds at each site:

| | Without | With |
| --- | --- | --- |
| Cells meeting the requirement, West Pleasant View | 0.903–0.917 | 0.932–0.950 |
| Montague City | 0.904–0.915 | 0.948–0.973 |
| Ground RMSE, Montague City | 0.157–0.174 m | 0.151–0.166 m |
| Mission complete | 372–455 s | 486–542 s |

Every run used its 120 s. On Montague City that is half to two thirds of the
cells left short.

`end_when_all_home` (on in `field_v3`) ends a scan whose every drone is homeward
for good (a lost sensor, its reserve). The mission used to scan on with nothing
to scan. It closes the points no drone can reach, completes, and its coverage
stays short of its gate in the replay, so a score on the map still fails it.

## In the viewer and the scores

The viewer's map readout shows the share met and what the rest lack.
`argusnet.evaluation.field_qualification` measures `map_error_over_claim`,
`map_error_within_2_sigma`, `passes_at_median` and `image_gsd_p90_m` from the
report, for a threshold file to gate on.

## Tests

`map_report.rs` (5), `rust/argusnet-sim/tests/field_v2.rs` (the replay's report
agrees with its last frame), the viewer's quality line.
