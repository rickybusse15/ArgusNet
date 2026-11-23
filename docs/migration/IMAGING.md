# Imaging (SR4, first slice)

**Status:** first slice shipped behind `--imaging` (or `--imaging-profile FILE`)
in a `--world-knowledge none` run that flies a rig (`--rig-profile`), and in
`--profile field_v2`. Code: `rust/argusnet-sim/src/imaging.rs`, with the capture
in `pod.rs` and the evidence in `belief_world.rs`. Without the flag nothing
changes.

The rig always carried two visible cameras and a thermal camera that nothing
modelled. This models what decides whether an image is usable. It renders no
pixels and no temperatures, and every number is assumed.

## What a capture records

`range_scans[].pod.images`, from the pose the LiDAR was fired from:

| | |
| --- | --- |
| Ground sample distance | Slant range over the focal length in pixels, stretched by obliquity |
| Blur | Pixels the scene moves in the exposure: speed across the boresight, plus the pod's rotation |
| Refusals | Stowed, unsettled, no ground, too oblique (45°), blurred (over 1 px) |
| Stereo | Depth error of the two visible cameras at that range |
| Thermal | Blur over an 8 ms time constant; smallest temperature difference shown, `NETD × pixels smeared` |

A settled pod turns as fast as its command does. A sweep is judged at that
rate, not at its residual jitter.

## An image requirement

`max_image_gsd_m` in a quality requirement asks each cell for a visible image
at least that fine.

- The map places an accepted image by the estimated pose and its own measured
  ground. Each measured cell it covers keeps the finest sample distance.
- A cell lacking only that is `image_coarse`. It is checked last, so a cell
  names what else it lacks first.
- A requirement finer than the floor height allows (`min_image_agl_m`, 40 m) is
  `image_limited`: unreachable, never met, and not flown for.
- Otherwise the whole flight comes down to 0.9 of the height that gives the
  sample distance straight down, and slows to keep the image sharp.

## Measured (2026-11-22)

West Pleasant View, 600 s, the field profile's flags with view planning:

- **Watching changes nothing.** With no requirement the mission is the same
  frame for frame. From the survey height the visible sample distance is 4.3 cm
  at the median, and no settled visible image was refused.
- **Low passes lose to one low flight.** Surveying high, then coming down only
  for blocks that mostly lacked an image, was built and removed. Scan time for
  a 0.9 gate:

| Requirement, seed | Passes | Whole flight low |
| --- | --- | --- |
| 4 cm, 7 | 589 s | 308 s |
| 3 cm, 7 | not in 600 s (0.82 met) | 470 s |
| 3 cm, 42 | not in 600 s (0.85 met) | 491 s |
| 3 cm, 4451 | not in 600 s (0.70 met) | 489 s |
| 3 cm, 707 | not in 600 s (0.84 met) | 448 s |

  A requirement on every cell means a low flight over every cell, and that
  flight meets the LiDAR's requirement too; the high survey is then extra.
  Passes might pay for a requirement on part of the map, which there is no way
  to declare yet.
- **Thermal is nearly all blurred at survey speeds**: 904 of 1,908 settled
  images from the survey height, 2,353 of 2,392 at 4 cm.

## What it does not claim

- Assumed optics, exposure and time constant; no calibration.
- What stands between the camera and the ground is not considered when the map
  places an image: a cell behind a building counts as imaged.
- One visible camera meets a requirement; stereo and thermal are reported only.
- Ground more than twice as far below the aircraft as what is under it is left
  out of an image.
- No trigger interval, overlap, light or weather.

## Tests

`rust/argusnet-sim/tests/imaging.rs` (5) and the unit tests in `imaging.rs` and
`quality.rs`.
