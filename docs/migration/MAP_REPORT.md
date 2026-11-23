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

## What it does not do

- Nothing replans on it: it is written when the run ends.
- `predicted` is flat ground, nadir, one pass. It is not a coverage plan.
- A cell's error is its mean height less the true ground at its centre, so it
  includes where in the cell the returns fell.

## In the viewer and the scores

The viewer's map readout shows the share met and what the rest lack.
`argusnet.evaluation.field_qualification` measures `map_error_over_claim`,
`map_error_within_2_sigma`, `passes_at_median` and `image_gsd_p90_m` from the
report, for a threshold file to gate on.

## Tests

`map_report.rs` (5), `rust/argusnet-sim/tests/field_v2.rs` (the replay's report
agrees with its last frame), the viewer's quality line.
