# Starting from no world knowledge (`world_knowledge: none`)

Every run before this one gave the platforms the true terrain and obstacle
layer. Controllers flew terrain-following altitudes from it, deconfliction and
safety clamped against it, and follow controllers planned sight lines over it.
A real fleet arrives somewhere it has never seen.

`--world-knowledge none` runs the same scenario with the platforms knowing
nothing about the world but their own launch pads. They build the map with
their own sensors, and every planning decision reads only that map.

```bash
argusnet-rs sim --config runs/demo-config.json --world-knowledge none --replay runs/scratch.json
argusnet-viewer --scene runs/demo.smartscene --run runs/demo-config.json --world-knowledge none
```

`prior`, the default, is the run it always was. All 14 canonical replays are
byte-identical, and the lifecycle checks are equal.

## Design

```
            true world (terrain, buildings)
             │                       │
   stage 6: range scanner      physics guard (stages 1, 3)
   (a sensor model)             keeps platforms out of it, records contacts
             │ RangeScansToMapping
   stage 8: BeliefGrid  ──publish──▶  BeliefSnapshot (immutable, Arc)
   (state, checkpointed)                  │
                        next tick's stage 1 (controllers), 3 (deconfliction),
                        4 (safety) read it through BeliefTerrain,
                        BeliefObstacles and BeliefVisibility
```

1. **Sensing** (`range_scan.rs`). Each mobile platform carries a range
   scanner. The default fan has 226 rays:
   - a nadir ray;
   - six rings of 36 rays, at 80° to 20° below the horizon;
   - a 9-ray forward fan at 4° below the horizon, spanning ±60° around the
     heading, which sees rising ground and walls before the platform reaches
     them.

   Range is 400 m, with 0.3 m seeded range noise keyed on node and tick. A
   return is a point plus a class, `ground` or `structure`. The class models
   ground segmentation of the point cloud, taken from the world exactly. Like
   the bearing sensors, the scanner is a sensor model, so it is allowed to
   read the true world.
2. **Mapping** (`belief_world.rs`, stage 8). A grid at 10 m holds, per cell:
   - the mean of its ground returns;
   - log-odds structure evidence (+0.85 per structure return, −0.4 per ground
     return);
   - the highest structure return;
   - the tick it was last seen.

   It starts empty. A cell counts as occupied at p ≥ 0.75, which takes two
   structure returns.
3. **Publication.** After each tick stage 8 publishes an immutable snapshot.
   The next tick's stages 1, 3 and 4 read it through samplers that implement
   the same traits as the true world (`TerrainSampler`, `ObstacleSampler`,
   `VisibilitySampler`, `TerrainHeight`). The stages can't tell the
   difference.
4. **Unknown is not free** (ARCHITECTURE_AND_PROTOCOLS §3). Ground in an
   unseen cell is the highest ground seen within 200 m, plus a 30 m margin.
   With nothing seen nearby it is the launch floor plus the margin. The
   radius is counted in 5×5-cell blocks, so it can reach up to a block
   further, which errs high. Platforms therefore climb out over unseen ground
   and settle as the map fills under them.
5. **Physics stays true.** `PhysicalGuard` holds the true terrain and
   obstacles for one purpose: a platform cannot be inside them. It runs after
   stage 1 samples and after stage 3 resolves. If a belief led a platform
   below the true ground or into a building, the guard puts it back and the
   frame records a `world_contacts` entry. The contact count measures how
   safe flying on the belief was.

The grid, together with what the replay last showed of it, is stage 8
checkpoint state (an optional key, absent in prior runs). A resumed run
publishes the same snapshot and writes the same replay rows. Reset empties the
grid. The snapshot is incremental: cell arrays are shared, not copied, and the
stand-in is recomputed only for blocks that received returns.

## Replay

All additions are optional:

- **Frame `belief_updates`**: rows `[i, j, ground_cm, structure_top_cm,
  occupancy_pct, confidence]` for the cells whose shown state changed. That
  means first seen, ground moved ≥ 10 cm, occupancy moved ≥ 5 points,
  structure appeared or moved, or the confidence level stepped (1 + log₂
  returns, capped at 4). Applying the rows in frame order rebuilds the map.
- **Frame `range_scans`**: at most 32 returns per platform scan, for showing
  the sensing.
- **Frame `world_contacts`**: as above.
- **`mapping_state.belief`**: known cells and fraction; ground RMSE and
  maximum error against the true terrain at known cell centres; structure
  cells, with structure precision (believed cells with a true building or
  wall within one cell) and recall (true building cells with a believed
  structure within one cell). These are evaluation figures, computed by the
  writer and never read by the run.
- **`meta.world_knowledge`**: the exploration config, the grid layout
  (origin, resolution, dimensions) and the launch floor.

## The viewer

A replay or `--run` from a `none` run shows the map the drones had at the
cursor. The scene's own layers (the true terrain and buildings) are hidden
when such a run opens, and stay in the Layers panel as truth for comparison.

- **Known ground**: a surface over every cell with a ground return, its
  corners averaged from neighbouring cells. Few returns show washed toward
  slate; many show as solid terrain colours.
- **New cells**: they glow cyan for 2.5 s, so the edge of the map visibly
  moves.
- **Structures**: extruded in magenta to their highest return.
- **Unknown**: black, inside a faint outline of the map bounds.
- **Range returns**: each drone's returns this frame, drawn as rays.
- **Contacts**: red rings.
- **Readout**: known %, ground RMSE, structure found and right %, and
  contacts.

Scrubbing back rebuilds the map from the start. Mesh rebuilds are confined
to the 32×32-cell chunks that changed.

## Evidence

Release build, 4 workers, seed 7, 120 s at dt 0.25 s:

| Scenario | Known at 1 s / 10 s / 30 s / 60 s / 120 s | Ground RMSE at 120 s (max) | Structures at 120 s: found / right | Contacts | Tick p50 (prior → none) | Replay (prior → none) |
|---|---|---|---|---|---|---|
| Demo: `small` map, 4 drones, target tracking | 8% / 60% / 78% / 82% / 86% | 0.98 m | 88% / 100% | 0 | 0.26 → 3.4 ms | 9.4 → 11.9 MB |
| Regional: 16 drones, scan/map/inspect, 178,848 cells | 1% / 9% / 27% / 45% / 74% | 0.28 m (2.2 m) | 13% / 100% | 0 | 0.60 → 9.8 ms | 5.8 → 21.3 MB |

The regional recall is a coverage figure, not a mapping error. The scan lanes
were planned around the large building (432 × 252 m) by the bundle builder,
which still uses the true obstacle layer (see below). No drone ever flies over
it, and at working altitude the 400 m scan range does not reach its roof.
Everything that has been seen is mapped right.

A range return reports the nearer of the terrain crossing and the first
building or wall face. An earlier cut used the visibility test, which checks
terrain first, and so missed nearly every building.

The worst ground errors are steep cells, where a 10 m cell's mean return
differs from the height at its centre. Over the regional run, stage 6 (the
scans) is about 59% of the tick and stage 8 about 39%. The replay is the same
at 1 and 4 workers: scans fork per platform.

Tests (`rust/argusnet-sim/tests/world_knowledge.rs`):

- the map starts nearly empty and fills monotonically, with ground RMSE under
  3 m;
- `prior` changes nothing;
- `none` is deterministic and differs from `prior`;
- no platform is ever below the true ground (zero contacts on the scan/safety
  fixture, which also exercises stage 4);
- a checkpoint taken at tick 30 resumes exactly, and a prior run refuses it;
- reset equals a fresh run;
- the replay rows rebuild the map, one row per known cell and matching
  structure counts;
- a run without stage 8, a bad config, or a run already started is refused.

## What still knows more than it should

This slice removes the terrain and obstacle prior from planning. Three other
sources of prior knowledge remain, and are named here so they are not
mistaken for done:

- **Target pursuit reads truth.** Follow controllers get target positions from
  `TablePursuit`, the truth table, in prior and no-prior runs alike. The stage
  notes say so every tick ("legacy pursuit read N true target states"). The
  fix is a pursuit oracle over the previous tick's fused tracks.
- **The bundle builder used the true world.** Search lanes were pushed out of
  true obstacles, follow routes planned around them, and POIs placed at true
  heights, all before the run starts. In a no-prior run these are mission
  inputs the platforms did not earn. The builder needs a mode that leaves the
  lanes straight and lets the belief obstacles deflect them.
- **Unseen buildings stay unknown.** Unknown ground already stands in high,
  so planning stays safe. But a mapping mission should go and look, which is
  frontier exploration again.
- **Mission progress counts footprint coverage.** The scan mission's
  `coverage_fraction` is still the 50 m footprint bookkeeping, not the
  belief's known fraction, so a scan mission completes on the old measure.
  The natural next step is to drive the scan phase from the belief, and send
  idle platforms to frontier cells (known ground bordering unknown).

The structure classes are exact, and vegetation returns are dropped. A
segmentation error model and canopy handling come later.
