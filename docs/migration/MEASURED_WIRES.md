# Measured power lines (W1)

**Status:** first slice shipped behind `--measured-wires` (or
`--wire-sensor-profile FILE`) in `--world-knowledge none` runs. Without the
flag, the oracle is unchanged. Code: `rust/argusnet-sim/src/wire_sensing.rs`,
the evidence in `belief_world.rs`, and the sensors in `range_scan.rs`, `pod.rs`
and `nav_camera.rs`.

## What it replaces

By default, a sensor near a conductor reports the whole true span: exact
geometry and real id, from one sighting. Planners then avoid the exact wire.
The sensing guide (§5) names this oracle assistance. With the model, sensors
report only measurements, and no output names a wire.

## The LiDAR model

A conductor is far thinner than the simulated ray spacing, so rays are not cast
at it. For each conductor piece (at most 5 m), expected returns per capture are:

```text
λ = (points per capture / field of view) · (apparent length / r) · divergence · p
```

The first bracket is rays per steradian; the next two are the solid angle within
which a beam touches the piece, so a wire seen end-on earns few points. The hit
count is a Poisson draw. A hit blocked by terrain, a structure, a trunk or a
crown (with the same leaf draw as ground rays) is dropped. Each point has range and
angular noise and, on a rig, goes through the estimated pose like ground
returns.

`p` is the chance a return clears the detector. Return power goes with
reflectivity, the share of the footprint the wire fills, and 1/r². Past the
point where the wire is thinner than the footprint, that is 1/r³. The threshold
is what a 10%-reflective, beam-filling target returns at the device's quoted
range, and the chance falls steeply there (a logistic of exponent 4).

| Assumed | Value |
| --- | --- |
| Point rate | 240,000 /s |
| Beam divergence | 2 mrad |
| Quoted range | 190 m |
| Conductor reflectivity | 20% |

With these, a 3 cm conductor is detected at 50% around 100 m, against 190 m for
a surface, and is gone by 200 m.

## The camera model

A camera line is two sight directions bounding the part of a span in view, each
off by about a pixel. It has no depth and no id, and is kept with a declared
per-frame probability, within the span's pixel range. W1 records them; W2
triangulates them. Contrast against sky or ground is not modelled yet.

## The map

Points build per-cell evidence: a count and the lowest and highest point. A cell
with `min_hazard_points` (2) becomes a height band, `hazard_margin_m` (5 m)
above and below, that avoidance reads (`BeliefSnapshot::collides`). A platform
can pass over or under a band, unlike a structure. Points never count as
measured ground. Refused pod captures add nothing.

`BeliefSnapshot::wires()` stays empty: no spans exist until W2 fits them, so
planners that route around spans have none. The band has holes wherever a cell
has too few points.

Truth never enters this path. The tests compare points with the true lines
themselves.

## Measured (2026-11-04)

Scan fixture with a two-span, two-conductor line (3 cm) across the map at flight
height, `efficient_v1`, provisional rig:

| Line top above datum | LiDAR points | Range p10 / p50 / p90 | Structure contacts, oracle → measured | Closest to a conductor, oracle → measured |
| ---: | ---: | --- | --- | --- |
| 220 m | 27,737 | 16 / 60 / 95 m | 7 → 7 | 0.5 → 0.5 m |
| 260 m | 24,207 | 10 / 65 / 94 m | 8 → 8 | 2.0 → 1.0 m |
| 300 m | 16,339 | 27 / 48 / 108 m | 6 → 6 | 2.0 → 2.0 m |
| 340 m | 10,459 | 7 / 69 / 72 m | 6 → 6 | 1.4 → 1.4 m |

**Measuring is not worse than the oracle here, and neither is good.** Aircraft
come within 0.5–2 m of a conductor in every run, far inside the 10 m clearance.
Only the physical guard stops contact. Keeping clearance needs a fitted wire
with uncertainty (W2) and planning on it (W4).

On built `medium` scenarios (8 drones, 120 s, 3 seeds) the LiDAR touched 88–98%
of the line's pieces. From the air, in time, a line gets found; the hard part is
finding it soon enough at flight height, ahead of the aircraft.

## Not in this slice

- Fitting points into wires (W2), pole detection (W3), wire-aware planning (W4),
  uncertain priors and the evaluation suite (W5).
- Camera contrast and background.
- False wire points.
