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
cursor. `--run` needs no scene package. Without one, the viewer writes a
package holding only the run's bounds and the marker styles, and opens on
that; the world it draws is the one the drones build. With `--scene`, the
scene's own layers (the true terrain and buildings) are hidden when such a
run opens, and stay in the Layers panel as truth for comparison.

- **Known ground**: one continuous lit surface over every cell with a ground
  return. Its vertices are the cell corners, shared between cells, with
  normals taken from the corner heights. Few returns show washed toward
  slate; many show as solid terrain colours. A darker skirt drops from every
  edge where the known map meets the unknown, so the map reads as a cut-out
  of the world.
- **New cells**: they glow cyan for 2.5 s from first being sensed, so the
  edge of the map visibly moves.
- **Structures**: one solid, with a roof over each occupied cell at its own
  top and a wall wherever the neighbour is lower or empty. Adjacent cells
  merge into a single building outline.
- **Unknown**: black, inside a faint outline of the map bounds.
- **Range returns**: each drone's returns this frame, drawn as rays.
- **Contacts**: red rings.
- **Readout**: known %, ground RMSE, structure found and right %, and
  contacts.

It is part of the viewer, not an overlay on top of it:

- **Terrain material.** The ground uses the same material as a scene's
  terrain: the terrain style's base colour, the elevation tint, contour lines
  and slope shading. Cells with few returns, and cells sensed in the last
  2.5 s, are tinted on top of that.
- **Belief overlay.** Each changed tile is republished to the viewer's belief
  pipeline (`belief_render`), in the store's own patch format. The
  Reconstruction preset then draws the classes (safe, obstacle, unknown),
  confidence and staleness over the drones' terrain, through the overlay the
  pipeline was built for.
- **Terrain heights.** Marker footprints, zone walls and the camera's ground
  read the drones' measured ground, refreshed once a second. An unknown cell
  is no data, not zero.
- **Layers panel.** "Drones' map" is a layer of its own. The scene's layers,
  when a scene was given, are labelled as truth.
- **Split view.** The right pane shows the drones' map alone, in 3D.
- **Picking.** The drones' terrain and buildings hide what is behind them, as
  a scene's own geometry does.

Scrubbing back rebuilds the map from the start. Mesh rebuilds are confined
to the 32×32-cell chunks that changed.

## Saving the map

A map the drones built can be saved as a scene package (`smartscene-v1`) and
opened again like any other world:

```bash
argusnet-rs sim --config runs/demo-config.json --world-knowledge none --export-scene runs/demo-map
argusnet-viewer --scene runs/demo-map
```

In the viewer, **Save map as scene** in the map readout saves the map at the
cursor, live or replayed, to `./argusnet-maps/self-built-t<seconds>s`.

The package is what `argusnet build-scene` writes, and the Python manifest
validator accepts it:

| Path | Contents |
|---|---|
| `terrain/terrain-base.glb` | The measured ground, as one surface with a skirt. |
| `overlays/buildings.glb` | The believed structures, when there are any. |
| `metadata/belief_map.json` | The map itself: the grid layout and one row per known cell. |
| `metadata/environment.json`, `metadata/style.json` | The bounds, the measured height range and the styles. |
| `replay/replay.json` | The run that built the map (CLI export only). |
| `scene_manifest.json` | `build.source_kind = "self-built"`, with the run's provenance. |

Opening the CLI's package therefore shows the run on the timeline, with the
map growing from nothing. The saved end state is in the Layers panel as
"saved map". The viewer's chunks and the saved meshes come from the same
code (`argusnet_sim::belief_mesh`), so the saved model is the one the viewer
drew.

The export refuses a run that built no map (`--world-knowledge prior`), a
map with no measured ground, and a directory that is not empty. The CLI
refuses before the run starts.

## Evidence

Release build, 4 workers, seed 7, 120 s at dt 0.25 s:

| Scenario | Known at 1 s / 10 s / 30 s / 60 s / 120 s | Ground RMSE at 120 s (max) | Structures at 120 s: found / right | Contacts | Tick p50 (prior → none) | Replay (prior → none) |
|---|---|---|---|---|---|---|
| Demo: `small` map, 4 drones, target tracking | 8% / 61% / 78% / 83% / 87% | 0.98 m | 100% / 100% | 0 | 0.26 → 3.4 ms | 9.4 → 11.9 MB |
| Regional: 16 drones, scan/map/inspect, 178,848 cells | 1% / 9% / 27% / 45% / 74% | 0.28 m (2.2 m) | 44% / 100% | 0 | 0.60 → 9.8 ms | 5.8 → 21.3 MB |

Believed roofs on the demo map are 137.5–138.1 m, against a true 137.2 m; the
spread is the 0.3 m range noise.

The regional recall is a coverage figure, not a mapping error. The scan lanes
were planned around the large building (432 × 252 m) by the bundle builder,
which still uses the true obstacle layer (see below). No drone flies over it,
so only the parts of its roof and walls within scan range of the lanes are
seen. Everything that has been seen is mapped right.

A range return reports the nearer of the terrain crossing and the first
building or wall surface, in 3D (`Obstacle::surface_hit_t`). Two earlier
cuts got this wrong:

- The first used the visibility test, which checks terrain first, and so
  missed nearly every building.
- The second used `first_hit_t`, which reports where a ray enters the
  footprint in plan view. That is right for "is the line blocked", but a ray
  coming down onto a roof enters the footprint above it. Roofs came out up
  to 150 m high (291 m against a true 137 m), and roof cells that only a
  steep ray reaches were missed. Saving the map as a model showed it.

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
- the map saves as a scene package that carries its replay, and a prior run
  has nothing to save;
- follow drones pursue fused tracks and stage 1 never reads truth, while
  `--pursuit truth` keeps the legacy oracle and differs;
- a checkpoint does not cross between pursuit sources;
- a run without stage 8, a bad config, or a run already started is refused.

## Pursuit from fused tracks

Follow controllers in a no-prior run steer off the run's own fused tracks, not
the targets' true trajectories (`pursuit.rs`, `TrackPursuit`). A track's id is
its target's id, so each follow drone looks up its target's latest track:

- **A track exists.** The estimate is extrapolated at the track's velocity from
  the track's timestamp to the time asked, for at most 3 s. Past that it is held
  where the extrapolation stopped, with zero velocity. Tracks arrive late by the
  fusion lag (B4), so extrapolation is the usual case.
- **The tracker dropped the track.** The last estimate is kept, so the drone
  goes to where the target was last believed to be.
- **No track yet.** The drone uses a tasking cue: the target's position at the
  start of the run, held still. This is the operator's report a follow
  drone is launched with, and it is the only truth the oracle uses.

The run driver hands each committed tick's tracks to the oracle between ticks
(`AssembledRun::commit_tick`), so its checkpoint holds them and a resumed run
pursues exactly what an uninterrupted one would. The stage notes say where each
answer came from ("pursuit read N fused-track estimates and M tasking cues (no
truth)"). `--pursuit truth` keeps the legacy oracle, to measure what truth was
worth. A prior run always pursues truth, as before.

The demo scenario has two follow drones (east on Lynx, west on Fox). The table
compares the two oracles over five seeds (7, 42, 137, 9999, 31415): the mean of
the per-seed difference, tracks minus truth, ± its standard deviation.

| | No fusion lag | 1.5 s fusion lag |
|---|---|---|
| Track RMSE at evidence time | −1.0 ± 4.5 m (truth 17.3 m) | −1.9 ± 3.6 m (truth 17.4 m) |
| Follow drone to target, median | −18.9 ± 1.5 m (truth 54.0 m) | −4.1 ± 0.9 m (truth 54.0 m) |
| Follow drone to target, 90th percentile | +12.8 ± 7.0 m (truth 74.6 m) | +53.3 ± 9.9 m (truth 74.6 m) |
| Bearings from the follow drones | −84 ± 134 (truth 1,284) | −106 ± 87 (truth 1,263) |
| Map known at 120 s | +5 ± 2 points | +7 ± 2 points |
| Contacts | 0 | 0 |

Tracking accuracy does not change measurably. With truth, a follow drone holds
its standoff from the real target. With tracks, it holds it from the estimate,
so it sits closer on average but follows the track's own error in its worst
moments. At 1.5 s lag a track can be 170–230 m off a manoeuvring target for a
few seconds, and the drone goes with it. Turning the extrapolation off made
following worse (at 1.5 s lag, mean distance 86 m against 71 m) and raised the
evidence-time RMSE to 22.4 m, so it stays on.

## What still knows more than it should

The terrain and obstacle prior is gone from planning, and so is target truth
from pursuit. These remain, named so they are not mistaken for done:

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
- **Follow targets are checked against the truth table at setup.** Assembly
  confirms each follow target exists (`scenario.rs`, `operational`). It is a
  bundle validity check, not a position read, but it is a truth read.
- **Localization particles start at the true position.** Stage 8's particle
  localizer centres its draws on each platform's true position. That is ego
  truth plus noise, the localizer's own model, and it is separate from the
  world map. It matters once positioning is degraded (F2).

The structure classes are exact, and vegetation returns are dropped. A
segmentation error model and canopy handling come later.
