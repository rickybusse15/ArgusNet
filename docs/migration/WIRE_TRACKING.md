# Wire tracking (W2)

`--track-wires` fits the points and lines from `--measured-wires`
([MEASURED_WIRES.md](MEASURED_WIRES.md)) into wire tracks with uncertainty, and
flies the fitted wires. It needs `--measured-wires` or `--wire-sensor-profile`,
in a `--world-knowledge none` run. Without it nothing changes.

## The model

A track is one conductor over one span. It is a straight line across the ground
and a parabola for the sag, fitted by weighted least squares from evidence
binned every 5 m along the wire.

- **Association.** A point joins a track within 0.75 m across and 1 m in height,
  up to 60 m past its end. Others wait in a bounded pending list.
- **Seeding.** Deterministic RANSAC over the pending points. A seed needs 24
  points whose bins cover at least 60% of the chord, which refuses a chance line
  between two conductors.
- **Merging.** Two tracks merge when they are parallel, in gate, and a joint fit
  stays within 0.3 m.
- **Uncertainty.** Each height and lateral sigma comes from the fit's covariance
  and its misfit. It grows past the evidence and is capped at 3 m.
- **Cameras.** A camera line that agrees with a track counts as support. Two
  lines from views at least 5 m apart and within 60 m triangulate a track with a
  2 m sigma, confirmed only with further camera support.
- **Confirmation.** 30 points over 30 m.

Nothing tells the tracker which wire a point came from.

## What flies it

Confirmed tracks enter the belief as `PowerLine` spans named `wire-track-N`, so
route planning and the replay's `mapped_wires` read them as they read any
mapped wire. A span moves as evidence lands, and a move replans the POI and
egress flights the span now lies across, as a new span does. Its clearance is
10 m plus three of its largest sigma.

Explorers take fitted wires by height, looking 4 s ahead:

- at or above a wire's height, they fly over it, a floor of its top plus the
  clearance;
- clear below it, they stay below, a ceiling of its bottom less the clearance;
- already beneath it, they never climb through it.

Height changes at 15 m/s, so a platform reaches the clearance before the wire.
A fitted span is never routed round: its end is only where the wire stopped
being seen, and the wire goes on past it. Evidence further from a fitted wire
than its clearance keeps W1's height band.

## Results

The scan fixture with a double line, under `efficient_v1`, at 13 placements of
the line. Platform-ticks within 10 m of a conductor:

| Line x (m) | Line top (m) | Oracle | W1 band | W2 |
| --- | --- | --- | --- | --- |
| 0 | 220 | 9 | 9 | 7 |
| 0 | 260 | 14 | 13 | 5 |
| 0 | 300 | 13 | 13 | 6 |
| 0 | 340 | 14 | 14 | **70** |
| -120 | 260 | 11 | 11 | 5 |
| -120 | 300 | 2 | 2 | 2 |
| -120 | 340 | 12 | 13 | 13 |
| 120 | 260 | 1 | 1 | 1 |
| 120 | 300 | 4 | 4 | 5 |
| 120 | 340 | 3 | 3 | 3 |
| 60 | 260 | 10 | 10 | 9 |
| 60 | 300 | 5 | 5 | 2 |
| 60 | 340 | 4 | 4 | 1 |

W2 is better at 7 placements, equal at 4, slightly worse at 1 and much worse at
1. The closest pass
barely changes: 1 to 3.5 m at most placements in all three modes. The ones
traced came before a track was confirmed or in a mission flight. The fit is the
stronger result: mid-span error was 0.01 to 0.1 m where inspected, and 1 of 42
confirmed tracks was off, by 4.5 m.

## A real place

West Pleasant View, Colorado (`tests/fixtures/real/west_pleasant_view`): USGS
elevation and OSM buildings, trees and a transmission line of 5 spans with 6
conductors 1.6 m apart. 300 s, `contingency_v1`, `flown_v1`, no world knowledge.

| Seed | Mode | Ticks within 10 m | Closest (m) | Tracks | Map known |
| --- | --- | --- | --- | --- | --- |
| 7 | Oracle | 57 | 1.0 | - | 95.6% |
| 7 | W1 band | 57 | 1.0 | - | 95.6% |
| 7 | W2 | 17 | 1.0 | 34 | 95.7% |
| 42 | Oracle | 13 | 1.1 | - | 95.8% |
| 42 | W1 band | 13 | 1.1 | - | 95.8% |
| 42 | W2 | 20 | 7.9 | 26 | 92.5% |

At seed 7, 32 of 35 fitted tracks lie within 0.2 m of a true conductor across
the line, so neighbouring conductors stay apart; the other 3 sit between two.
Track heights were not checked to better than a metre. Seed 42 mapped less of
the place and finished its flight home, where the other two did not.

## Known limits

- **Pinning at the guard.** The 70 above is one drone meeting a stretch of wire
  no track had confirmed. The guard holds it, while the explorer's commanded
  position flies on, so it stays until a straight line to that position is
  clear. This is not W2's: the oracle and W1 record 21 to 22 contacts the same
  way at x = -120. W2 changes which runs meet it. The explorer needs to hear
  that the aircraft was stopped.
- **Tracks across a pole.** A track seen on both sides of an unseen pole can fit
  through it. That is the 4.5 m miss, and its reported sigma did not show it.
  Pole detection (W3) splits them.
- **Short tracking range.** A wire at flight height is seen late: the camera
  sees a 3 cm conductor to under 40 m, about 2 s at cruise.
- **Mission flights.** Inspection flights near a wire are not changed by the
  explorer's bounds and rely on route planning.

## Replay

`mapping_state.belief.wire_tracks` counts confirmed tracks, absent when none.
Each `mapped_wires` entry for a track is the span at that tick, and the latest
row for an id wins. A frame's `retired_wires` lists the spans dropped that tick
when tracks merged, so applying the frames gives the wires the map holds.
