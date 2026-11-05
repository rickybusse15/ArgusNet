# Poles, corridors, wire-aware flight and uncertain priors (W3 to W5)

Three opt-in flags, each building on [W2](WIRE_TRACKING.md). Without them
nothing changes: 14 of 14 replay hashes match the base, and every lifecycle
check is equal.

| Flag | Needs | What it adds |
| --- | --- | --- |
| `--detect-poles` | `--track-wires` | W3: poles, track ends, corridors |
| `--wire-planning` | `--detect-poles` | W4: every flight held clear of what is believed |
| `--wire-prior-file FILE` | `--track-wires` | W5: a wire map with declared errors |
| `--degraded-wire-prior` | `--track-wires` | W5, for evaluation: the true lines, spoiled |

## W3: poles and corridors

A 3 cm conductor fades at about 100 m. A pole fills the beam, so it is seen as
far as a surface. Poles were not visible to any sensor before this: supports
were only drawn. Now a pole returns LiDAR points through the wire model, with a
radius assumed at 0.2 m (`POLE_RADIUS_M`). Its returns arrive among the wire
points, and nothing says which object a return came from.

- **Detection.** Returns are kept by 4 m ground cell. A cell whose returns stand
  at least 8 m tall, with at least half their 2 m height bins filled, is a pole.
  A wire stays at one height, and stacked conductors leave gaps.
- **Tracks.** A pole inside a track's evidence splits it, and the pieces cannot
  merge again. A pole within 120 m past a track's end is where the track goes.
- **Corridors.** A track between two poles, or a straight row of poles, links
  them. The span past a row's last pole is predicted, and a free track end is
  carried on 120 m. Predicted stretches get 3 m of sigma, so 19 m of clearance.
  They are mapped as `wire-corridor-*` and read as fitted wires.
- **Heights.** A pole's highest return is often not its top: a downward LiDAR
  sees nothing above itself and a pod's view clips what it does see. In one
  case a corridor sat 70 m below the wire. So a top is open until a scan from
  above reaches it, an open top is planned 30 m higher, and a connected row
  plans for the highest evidence anywhere in it.

**Explorers rejoin their aircraft.** With detection on, an explorer stays with
an aircraft a mission is flying, and goes on from where it is released. It used
to fly on unseen, and the aircraft jumped to it across the map. Because honest
explorers fly the same lines again, and the scan's rings leave the same gap
under each line, a reached goal that still needs work now counts against that
area.

## W4: wire-aware flight

`rust/argusnet-sim/src/wire_governor.rs` runs after mission overrides, so it
sees what every mobile platform is about to fly, whoever is flying it. It looks
6 s ahead for fitted wires, corridors, poles, prior spans and unconfirmed
evidence, then:

- goes over from at or above a wire's height, and stays under from clear below;
- goes over when there is no room under, as with a wire over a hilltop;
- changes height at 15 m/s from where the platform was, never as a jump;
- flies any offset off at the same rate once clear, and leaves a flight with no
  offset untouched;
- climbs 20 m off anything the guard holds a mission flight at.

It reads the belief only. Goal choice also counts crossings (300 m each) and
each metre flown along a line (5 m).

**Post-mission recalls.** A mapped wire could send a drone home after a mission
completed, with a recall nothing ever released, and it stayed parked. Corridors
change often, so this fired often. Fitted wires no longer recall anyone after
completion. Surveyed lines still do, as before.

## W5: uncertain priors

A prior span is a hypothesis with a declared error, mapped from the first frame
as `wire-prior-*`, with `10 m + 3 sigma` of clearance. Flight changes it:

- **Confirmed:** a confirmed track within 3 sigma and parallel moves the span
  onto what was measured and shrinks its error to 1 m.
- **Refuted:** within 60 m and above a stretch, with no return, counts against
  it. With half its stretches at 3 such scans and no return anywhere, the span
  is dropped, and `retired_wires` records it.
- **Open:** everything else stays a wide hazard.

A line missing from the prior is found as any unknown wire is.
`--degraded-wire-prior` makes a test map from truth: a shared 6 m offset, 3 m of
height error per support, 25% of spans dropped and one phantom 180 m away.

## Results

Scan fixture, 13 line placements by 3 seeds, 39 runs each, double conductor on
five poles, `efficient_v1`, provisional rig. `examples/wire_eval.rs` wrote
[the full report](../performance/wire-eval-2026-11-04.jsonl). The fixture's seed
changes little, so these are close to 13 independent cases.

| Mode | Ticks within 10 m | Median closest (m) | Runs under 5 m | Runs 10 m or more | Map known |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 7.7 | 3.6 | 25 | 5 | 99.6% |
| W1 band | 8.7 | 3.6 | 26 | 7 | 99.6% |
| W2 tracks | 8.5 | 5.5 | 19 | 8 | 98.5% |
| W3 poles | 6.1 | 4.0 | 24 | 6 | 99.9% |
| W4 planning | 4.2 | 7.6 | 14 | 8 | 99.8% |
| W5 prior | 9.6 | 8.0 | 13 | 16 | 99.9% |

W5's 9.6 is one pinned run of 279 ticks; the other 38 average 2.5. W3 alone
does not improve closest passes: it adds knowledge and honest explorers, and
W4 is what flies it. Poles: 5 of 5 on average, worst error 0.41 m. Priors, per
run: 2.9 confirmed, 0.67 of the 1 phantom refuted, 1.1 still open.

West Pleasant View, Colorado: real elevation and buildings, a transmission line
of 5 spans with 6 conductors, 300 s, `contingency_v1`, `flown_v1`, no world
knowledge. Distances are from the conductors, checked in 2 m steps.

| Seed | Mode | Ticks within 10 m | Closest (m) | Poles of 6 | Map known |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | Oracle | 57 | 1.0 | - | 95.6% |
| 7 | W2 | 17 | 1.0 | - | 95.7% |
| 7 | W3 | 14 | 1.0 | 5 | 93.8% |
| 7 | W4 | 0 | 17.1 | 5 | 87.1% |
| 7 | W5 | 0 | 17.1 | 5 | 94.3% |
| 42 | Oracle | 13 | 1.5 | - | 95.8% |
| 42 | W2 | 15 | 8.1 | - | 93.5% |
| 42 | W3 | 1 | 8.9 | 6 | 93.9% |
| 42 | W4 | 0 | 24.1 | 5 | 83.7% |
| 42 | W5 | 0 | 18.1 | 4 | 80.3% |

Clearing the line costs map: up to 13 points in the 300 s, because flights take
more height over the line and cross it less. Two seeds are too few to rank W4
against W5 here.

## Known limits

- **A pinned run.** With a prior, line at x = -120, top 340 m, seed 1234: a
  mission-era drone stays frozen 1 m from a conductor for 279 ticks. Two causes
  were tried and ruled out (the guard holding a mission flight, and a rejoin
  refusing steps into a wide clearance). The cause is not known.
- **Hills.** 14 fixture runs still pass within 5 m. The worst are flights along
  a line over rising ground, where the only safe move is sideways, and the
  governor moves only height.
- **Before detection.** A wire or pole not yet seen is not avoided. The closest
  pass at some placements is in the first ticks.
- **Clipped tops.** A row seen only from a distance can still sit low, within
  30 m of margin.
- **Assumed numbers.** The pole radius, height margin, clearances, crossing
  costs and all sensor figures are assumed. Lattice towers, camera poles and
  false returns are not modelled. This is simulation evidence, not a claim
  about flight.
- **W2 contacts.** W2 alone averages 135 contacts on the fixture because two
  runs at one placement (line x = 120, top 260 m) record 2,382 each. Not
  investigated here. With poles on, the same placement does not.

## Replay

Additive, absent when empty: `mapped_poles` on a frame (id, position, low and
high returns, sigma, `top_open`), and `wire_poles`, `wire_corridors`,
`wire_prior_open`, `wire_prior_confirmed` and `wire_prior_refuted` under
`mapping_state.belief`. Corridors and prior spans use `mapped_wires` and
`retired_wires`, so the viewer draws them as mapped wires. It does not draw
poles yet.
