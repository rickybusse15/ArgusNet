# Poles, corridors, wire-aware flight and uncertain priors (W3 to W5)

Three opt-in flags, each building on [W2](WIRE_TRACKING.md). Without them the
canonical scenarios are unchanged: 14 of 14 replay hashes match the base, and
every lifecycle check is equal. One fix is not behind a flag: the escape floor
below applies in any `--world-knowledge none` run where the guard stops an
explorer.

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
each metre flown along a line (5 m), and an explorer running along a wire turns
out of its corridor and carries on beside it. Height alone cannot leave a
corridor: a line over rising ground has no room under it and little over it.
Unverified prior spans are the exception. They are passed by height and repel
nothing, because flying near a span is how it gets settled.

**Escapes.** A drone the guard stops backs away and changes height. Its survey
height could stand above the wire, in which case both the explorer and the
launch wrapper lifted every descent back up through it, and the drone stayed
pinned 1 m from a conductor for the rest of the run. An escape now keeps a 15 m
floor over the ground, and the guard still holds the true ground. Without it,
W2 at line x = -120, top 260 m comes within 10 m for 35 ticks, not 7. Because
the escape is not behind a flag, oracle, W1 and W2 runs change slightly too.

**Post-mission recalls.** A mapped wire could send a drone home after a mission
completed, with a recall nothing ever released, and it stayed parked. Corridors
change often, so this fired often. Fitted wires no longer recall anyone after
completion. Surveyed lines still do, as before.

## W5: uncertain priors

A prior span is a hypothesis with a declared error, mapped from the first frame
as `wire-prior-*`, with `10 m + 3 sigma` of clearance. Flight changes it:

- **Confirmed:** a confirmed track within 3 sigma, in plan and in height, and
  parallel moves the span
  onto what was measured and shrinks its error to 1 m.
- **Refuted:** within 60 m and above a stretch, a scan with a ray within 20 m of
  it, going on past it, and no return counts against it. A stretch behind an
  obstacle or outside a pod's view says nothing. With half its stretches at 3 such scans and no return anywhere, the span
  is dropped, and `retired_wires` records it.
- **Open:** everything else stays a wide hazard.

A line missing from the prior is found as any unknown wire is.
`--degraded-wire-prior` makes a test map from truth: a shared 6 m offset, 3 m of
height error per support, 25% of spans dropped and one phantom 180 m away.

## Results

Scan fixture, 13 line placements by 3 seeds, 39 runs each, double conductor on
five poles, `efficient_v1`, provisional rig. `examples/wire_eval.rs` wrote
[the full report](../performance/wire-eval-2026-11-07.jsonl). The fixture's seed
changes little, so these are close to 13 independent cases.

| Mode | Ticks within 10 m | Median closest (m) | Runs under 5 m | Runs 10 m or more | Map known |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 7.8 | 3.6 | 25 | 5 | 99.6% |
| W1 band | 8.7 | 3.6 | 26 | 7 | 99.6% |
| W2 tracks | 7.2 | 5.5 | 19 | 8 | 99.1% |
| W3 poles | 5.6 | 4.0 | 24 | 6 | 99.9% |
| W4 planning | 3.8 | 9.5 | 11 | 19 | 99.7% |
| W5 prior | 1.9 | 11.4 | 7 | 22 | 99.8% |

W3 alone does not improve closest passes: it adds knowledge and honest
explorers, and W4 is what flies it. One placement got worse under W4 (line x =
0, top 340 m: 22 ticks against 13 with W2). Poles: 5 of 5 on average, worst
error 0.51 m. Priors, per run: 3.1 confirmed, 0.26 of the 1 phantom refuted,
1.3 still open.

West Pleasant View, Colorado: real elevation and buildings, a transmission line
of 5 spans with 6 conductors, 300 s, `contingency_v1`, `flown_v1`, no world
knowledge. Distances are from the conductors, checked in 2 m steps.

| Seed | Mode | Ticks within 10 m | Closest (m) | Poles of 6 | Map known |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | Oracle | 57 | 1.0 | - | 95.6% |
| 7 | W2 | 17 | 1.0 | - | 95.7% |
| 7 | W3 | 14 | 1.0 | 5 | 93.8% |
| 7 | W4 | 0 | 17.1 | 6 | 91.9% |
| 7 | W5 | 0 | 21.4 | 6 | 91.2% |
| 42 | Oracle | 13 | 1.5 | - | 95.8% |
| 42 | W2 | 15 | 8.1 | - | 93.5% |
| 42 | W3 | 1 | 8.9 | 6 | 93.9% |
| 42 | W4 | 0 | 38.0 | 6 | 96.5% |
| 42 | W5 | 0 | 22.4 | 6 | 93.0% |

Keeping off the line cost at most 4.5 points of map in these 300 s, and at seed
42 W4 mapped more than W2. Two seeds are too few to rank W4 against W5 here.

## Follow-ups (2026-11-08)

**Where the close passes came from.** The evaluator now records what was
flying each drone at every pass within 10 m. Of W4's 147, 137 were explorers,
4 POI flights and 6 egress, so the work went to explorers, not mission flights.
Two causes were traced and fixed:

- **Dithering at a corridor's edge.** A flight crossing a line at up to 50
  degrees was treated as running along it, turned out, and came back in, tick
  after tick, never holding a heading the climb over the wire could use. Only
  flights within 25 degrees of a wire are turned out now; a crossing is the
  governor's.
- **A lane down the wire.** With no frontier goal left an explorer flies its
  search lane, and steering spared any goal inside a corridor. Lanes are now
  turned out wherever they run; only a frontier goal inside a corridor is
  approached on purpose.

**Settling priors.** Each stretch of an unverified prior span that no return
and no run of silent scans has settled is now an explorer goal, alongside
unmapped ground. Refuted phantoms per run rose from 0.31 to 0.72 and confirmed
spans from 3.1 to 3.6, at the cost of 1.2 to 1.7 ticks within 10 m and 0.4
points of map: drones now fly near prior spans on purpose, true ones included.

**A drone pinned on its pad.** After a mission a smooth rejoin steps the drone
back toward its path, and each step was stopped at its first ground contact.
A drone on its pad starts inside the guard's ground clearance, so the contact
was at once, every tick: 2,382 contacts in one W2 run. A step that starts
inside the clearance is now left to `enforce`, which lifts its end clear, as a
step starting inside a structure already was. This is not behind a flag. The 14
canonical scenarios are unchanged, and the lifecycle checks are equal.

**Viewer.** Poles are drawn upright from their lowest return to their highest,
and corridors and prior spans (`wire-corridor-*`, `wire-prior-*`) in a paler
blue than measured wires, all under the Mapped Wires toggle.

| Mode | Ticks within 10 m | Median closest (m) | Runs under 5 m | Map known | Contacts, mean / worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 7.8 | 3.6 | 25 | 99.6% | 11.0 / 16 |
| W2 tracks | 7.2 | 5.5 | 19 | 99.7% | 12.1 / 40 |
| W3 poles | 5.6 | 4.0 | 24 | 99.9% | 16.9 / 22 |
| W4 planning | 3.0 | 9.5 | 14 | 99.8% | 16.7 / 21 |
| W5 prior | 1.7 | 11.4 | 8 | 99.4% | 13.3 / 25 |

[Report](../performance/wire-eval-2026-11-08.jsonl). W2's worst run had 2,382
contacts before the rejoin fix.

West Pleasant View, seeds 7 and 42, against the W3 to W5 results above:

| Mode | Ticks within 10 m | Closest (m) | Map known, before | Map known, now |
| --- | ---: | ---: | ---: | ---: |
| W4 | 0, 0 | 17.1, 24.1 | 91.9%, 96.5% | 87.1%, 83.7% |
| W5 | 0, 0 | 17.1, 22.4 | 91.2%, 93.0% | 95.1%, 94.8% |

No pass came within 10 m either way. W4 maps 5 to 13 points less here: turning
search lanes out of corridors keeps explorers off ground under the line that
they used to map. W5 maps more, as prior goals send drones across the line on
purpose. Two seeds are too few to say which holds in general.

## Known limits

- **Mission flights along a line.** The governor moves only height. Mission
  flights were 7% of the close passes, so this was left.
- **Phantoms.** 28% of phantom spans are still open after a 400 s fixture run.
  An open phantom costs map, not safety.
- **Before detection.** A wire or pole not yet seen is not avoided. The closest
  pass at some placements is in the first ticks.
- **Clipped tops.** A row seen only from a distance can still sit low, within
  30 m of margin.
- **Assumed numbers.** The pole radius, height margin, clearances, crossing
  costs and all sensor figures are assumed. Lattice towers, camera poles and
  false returns are not modelled. This is simulation evidence, not a claim
  about flight.

## Replay

Additive, absent when empty: `mapped_poles` on a frame (id, position, low and
high returns, sigma, `top_open`), and `wire_poles`, `wire_corridors`,
`wire_prior_open`, `wire_prior_confirmed` and `wire_prior_refuted` under
`mapping_state.belief`. Corridors and prior spans use `mapped_wires` and
`retired_wires`. The viewer draws poles, and corridors and prior spans in their
own colour.
