# Quality requirements and deficit-driven mapping (SR5a, SR5b first slice)

**Status:** first slice shipped behind `--quality-requirements FILE` in
`--world-knowledge none` runs. Spec: sensing realism guide §8.1, §8.2 and §8.6.
Code: `rust/argusnet-mapping/src/quality.rs` (the pure contract), and the
evidence, the layer and the queries in `rust/argusnet-sim/src/belief_world.rs`.

## The requirement

A `QualityRequirement` is declared before the run and recorded in
`meta.world_knowledge.exploration.quality`:

| Field | Meaning |
| --- | --- |
| `min_ground_density_per_m2` | Accepted ground returns per square metre |
| `max_vertical_sigma_m` | One-sigma vertical uncertainty of the cell's height |
| `max_age_s` | How old the newest accepted return may be |
| `max_incidence_deg` | The steepest-from-nadir ray whose ground return is accepted |
| `bias_floor_m` | The declared systematic vertical error every return shares |

The density, sigma and age fields are optional, but at least one must be
present. A requirement with none of them would mark every seen cell met, so it
is refused.

## Evidence and reasons

With a requirement, the belief grid keeps each cell's **accepted** ground
returns: the count, the sum and sum of squares of their heights, and the tick
of the newest. Returns from shallower rays are seen but not accepted. The
vertical sigma is the spread of those returns, never taken below the scanner's
declared range noise, divided by √n and combined with the bias floor. No count
reduces the bias floor.

Each published snapshot carries every cell's reason, the first that applies:

| Reason | Meaning | Planner response |
| --- | --- | --- |
| `unseen` | No ground return | Go there |
| `too_oblique` | Ground returns, none steep enough | New geometry |
| `bias_limited` | The bias floor alone exceeds the sigma requirement | None: **unreachable** |
| `sparse` | Too few accepted returns for the density or the sigma | More passes |
| `stale` | Met once, too old now | Revisit |
| `met` | — | — |

`occluded` and `rejected` are in the guide's table but need evidence this map
does not keep yet (per-ray occlusion, an admission step). No cell reports
them.

## How the run uses it (SR5b)

- **Explorers** go to blocks where at least half the cells are improvable
  (unseen, too oblique, sparse or stale), not just unmeasured
  (`BeliefSnapshot::work_centres` / `needs_work`). Without a requirement these
  are exactly `unexplored_centres` / `is_unexplored`.
- **The scan gate** reads `resolved_fraction`: met **or** unreachable over all
  cells, meaning nothing is left for flying to do. Unreachable cells never hold
  the scan open, and they are never counted as met. `met_fraction` and
  `unreachable_cells` are reported beside it, in
  `mapping_state.belief.quality`.
- **The scan decision** (F3 journal) gains `quality_met_cells` and
  `quality_unreachable_cells` in its inputs.

## Measured (2026-10-23, `scan` fixture, density 0.03/m², sigma 0.5 m, 40° incidence, 0.05 m bias)

| Run | Scan phase ends | At the gate |
| --- | ---: | --- |
| No requirement (known fraction) | tick 23 | at least 30% known (the threshold) |
| Requirement | tick 211 | 30% met; 2697 cells sparse, 418 too oblique |
| Same, bias floor 0.6 m | tick 35 | 0 met, 1460 unreachable |

By the end of the requirement run the map is 99% known but only 46% met.

## Not in this slice

- Occlusion and admission reasons.
- Reasoned views and pointing (SR5c/SR5d).
- Evidence deduplication: the mapping lane integrates each boundary's scans
  exactly once, but nothing yet refuses a duplicated scan from a source.
- Incremental quality. The layer is recomputed over the grid at each publish:
  about 5 000 cells here, linear in the grid.
