# View planning (SR5d, first slice)

**Status:** first slice shipped behind `--view-planning` (or
`--view-planning-profile FILE`) in a `--world-knowledge none` run that flies a
rig (`--rig-profile`) against a quality requirement (`--quality-requirements`).
Spec: sensing realism guide §8.3. Code: `rust/argusnet-planning/src/views.rs`
(the pure planner), with the explorer's goal choice in
`argusnet-sim/src/frontier.rs` and the pod's planned roll in `pod.rs`.

Stacked on #304 (rig and pod), with #302 (quality deficits) merged in.

## What it decides

**Where to cruise next.** An explorer's next goal is the block whose
improvable deficit is worth the most per watt-hour, not the nearest one.

- **Worth:** each cell's deficit weight (unseen 1.0, too oblique 0.8, stale 0.7,
  sparse 0.5; met and unreachable 0). A block another aircraft has claimed
  keeps a quarter of its worth (`claimed_worth`): a crowded fleet that never
  shares a goal sends aircraft far for little.
- **Cost:** the run's own battery model. Cruise for the horizontal leg, plus
  every climb and descent over the believed ground every 50 m, since the
  aircraft holds its height above it. Turning costs `turn_wh_per_rad`.
- **Cruise only.** Hover is for take-off and landing, so the planner never plans
  a station. An explorer re-goals on arrival and keeps flying, and the pod is
  stowed only below cruise.

**Where the pod looks (opt-in, `plan_pointing`).** The cross-track roll is
chosen per leg, not per capture. It is re-planned every `roll_replan_s` (8 s)
or when the heading turns more than 15°. Each candidate roll is valued over
the footprints the leg will sweep, sampled at four points ahead. Only rays
that the requirement accepts and that the LiDAR reaches count. A switch must
repay the captures it loses to slewing and settling, plus a 15% margin.
Planned pointing never lowers the flight height: rolls whose rays would fall
out of range are simply worth less.

## Measured (2026-10-27)

Regional scenario: 16 drones, ~179,000 cells, 120 s, provisional rig, 0.03/m²
density, 0.5 m sigma, 40° incidence, 0.05 m bias. Met cells at 120 s:

| Seed | Nearest-goal | View planning | Gain | Battery used (summed over drones) |
| --- | ---: | ---: | ---: | --- |
| 7 | 43,761 | 55,254 | +26% | 0.89 → 0.89 |
| 11 | 44,913 | 49,068 | +9% | 0.86 → 0.83 |
| 23 | 50,449 | 55,556 | +10% | 0.92 → 0.88 |

`cargo test --release -p argusnet-sim --test view_planning -- --ignored` pins
this: more than 5% more met cells, and no more than 2% more energy, on every
seed.

Not every variant tried helped:

- **Small maps are noise.** On `small` and `medium` maps (4–12 drones, 3 seeds
  each) the median change is about 0% and single runs swing ±16%. With a map
  only a few swaths wide, any change in an early goal sends the run somewhere
  else. The planner's advantage shows where goals are a real choice.
- **Excluding claimed blocks outright** was −5% to −6% on every 12-drone run.
  The claim penalty replaced it.
- **Counting what a leg sweeps on the way** (`en_route_weight`) averaged +12.5%
  on the regional seeds against +15% without it, at slightly higher energy. It
  is off by default.
- **Planned pointing** did not repay itself on any configuration measured:
  - at a 40° limit the 70° swath already fills what the requirement accepts,
    so the planner correctly holds nadir;
  - at 60°: −1%;
  - with a 30°-wide LiDAR: −16%.

  The early version, which flew lower so that rolled rays stayed in range, was
  worse still. It is off by default; enable it with `plan_pointing: true` in a
  profile once the LiDAR is chosen.

## Not in this slice

- Routing around believed obstacles on the way to a goal: legs are straight
  (P3).
- Cruise-only POI inspection: dwell is still a hover, so the pod stows.
- A calibrated aircraft energy model, and replanning on predicted vs achieved
  quality (guide §8.5).
