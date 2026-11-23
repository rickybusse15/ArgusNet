# Routes in three dimensions

**Status:** shipped behind `ExplorationConfig.route_3d`, on in `field_v3`. Code:
`rust/argusnet-planning/src/route3d.rs` (the search, pure) and
`rust/argusnet-sim/src/route_air.rs` (the air over a self-built map).

Mission flights (to a point to inspect, home) were flown straight in plan, or
round a mapped wire, and a height governor climbed over what came into its
reach. Nothing weighed going over a hill against going round it, and the climb
began when the hill was near.

## What it does

A* over plan position (20 m) and height (10 m), costed with the run's battery
model. The space is asked for one column at a time, as the search reaches it:

- **a floor**: the believed surface (ground, structures, the navigation camera's
  obstacles, the stand-in for unseen ground) plus the mission flights' clearance,
  and poles;
- **a closed band** where wires hang: a route goes over it, or under it where the
  ground leaves `min_agl_m`. Wires are kept 10 m further off than flights keep
  them (`wire_margin_m`), since a mapped wire is a fit.

A leg is tested every quarter cell in plan and every half layer in height, and across
no closed band between two samples, so a steep one does not step over a wire.

Within 40 m of either end only wires close the air: a pad and a point are the
end's own business, and a drone comes down over its end and does not circle down to
it. Corners within 40 m of the end are dropped for one over it: they
are for the way down, and had a drone pass its pad by, or go round it, and be taken to
be home. The way found is pulled straight wherever a straight leg is open, gentle
enough and no dearer.

| Outcome | Flown |
| --- | --- |
| `Straight`: the line is open | Straight |
| `Route`: saves 5% of the straight flight's energy as the governor would climb it, or the straight flight meets a wire and the route costs no more | The route |
| `NotWorth`, `None` | As without it |

The router's air holds the wires the map measured. A plan, straight or routed,
that comes within a wire the map was *given* is not flown; those are kept off as
they always were. A wire mapped later replans a routed flight too.

The governor and the physical guard still have the last word on every flight.
Ties go to the earlier node, so a run is deterministic and resumes exactly.

## Measured (2026-11-23)

The hilly scan fixture, seven missions that differ by when the scan ends
(`--ignored`, `routes_in_three_dimensions_over_seven_missions`):

| Scan gate | Done at, without → with | Energy left (two drones), without → with |
| --- | --- | --- |
| 0.20 | 104 → 95 s | 897.3 → 903.2 Wh |
| 0.25 | 98 → 95 s | 899.9 → 903.0 Wh |
| 0.30 | 111 → 111 s | 894.2 → 896.3 Wh |
| 0.35 | 119 → 143 s | 889.5 → 890.5 Wh |
| 0.40 | 130 → 137 s | 882.1 → 881.9 Wh |
| 0.45, 0.50 (one mission) | 170 → 170 s | 873.1 → 873.9 Wh |

It saves energy in five of the six distinct missions. It is slower in two: the
search minimises energy, not time, and a route round is longer.

On the real sites, whose ground is gentle and buildings low, mission times and
maps hardly move. The closest passes to a conductor on West Pleasant View's
tuning seeds were 17.1 m and up, as without it. Before the two rules about wires
(the margin, and the route taken where the straight flight meets one) two seeds
passed at 12.7 and 16.5 m, governed but close; they are 23.7 and 20.0 m.

## What it does not do

- The allocator still prices a trip by the plan route, not by this one.
- It plans once, when a flight starts, on the map as it is then.
- Explorers keep their own heights; only mission flights are routed.
- It does not trade energy against time.
- The gradient limit is the same up and down, and the aircraft's own lag is not
  modelled.

## Tests

`route3d.rs` (12): over, round and under; a wire not stepped over by a steep leg; an end not
passed by; a pad under the floor; coming down over
the end; what it costs; nothing found; determinism. `route_air.rs`: a structure
gone round and a wire kept 20 m off on a self-built map. `tests/field_v3.rs`: less
energy, no contacts and exact resume on the fixture.
