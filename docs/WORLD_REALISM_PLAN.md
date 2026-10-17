# World and viewer realism plan: buildings, vegetation, power lines

**Status:** proposed 2026-10-12. R0 is done: the opacity fix (#222) and
typed structure matching. R1–R5 are not started.
**Workstream:** WS-R (new). Ties into ROADMAP Tier C (scene ingestion and
semantic materials), VIEWER_UI §6.3a, §15 and §18, and AGENT_EXECUTION_PLAN C6,
C8 (P2.6) and D2 (P5.1).

## Where the world stands

| Feature | Today | Gap |
| --- | --- | --- |
| Buildings | Flat-roof footprint prisms (`ObstaclePrimitive::Polygon`, `obstacles.rs:104`), extruded with an ear-clipped roof in the scene (`scene_build.rs:389`) | No roofs, overhangs or levels |
| Buildings from real data | GeoJSON footprints load into the **viewer scene only** (`scene_loader.py:609`), with a default height of 24 m | The simulation never sees them, so the scene and the run disagree. No OSM, LiDAR or CityGML support |
| Vegetation | One `ForestStand` prism with a single density, plus Forest/Scrub land-cover cells. It attenuates line of sight (`visibility.rs:246`) | No individual trees. Drones fly through canopy, and the range scanner ignores it |
| Power lines | **Nothing.** A "tower" is a solid cylinder | No wires, no catenary sag, no clearance rules, nothing for thin obstacles anywhere (planner, safety, deconfliction, scanner) |
| Scene packaging | One GLB per layer, white material with vertex colours. No textures, UVs, LOD or instancing | Layer `alpha` styles are never applied, because the GLB has no `alphaMode` (a likely bug) |
| Viewer | Bevy PBR with 4 shadow cascades, fog and bloom by quality tier. Custom terrain material with land-cover tint and belief overlay | No instancing, mesh LOD or impostors; buildings and trees have no material classes |

## Principles

1. **One world, two consumers.** Every structure the viewer draws is the same
   primitive the simulation collides with, senses and plans around. No
   scene-only geometry.
2. **Opt-in and versioned.** New primitive kinds (`building-v2`, `tree-v1`,
   `powerline-v1`) and new world sources are new versions. The canonical
   scenarios keep their current worlds, so legacy replays stay byte-identical.
   New canonical scenarios exercise the new kinds.
3. **Deterministic placement.** Procedural trees and poles are placed from the
   scenario seed with a fixed-order Poisson-disk sampler. Imported data is
   content-hashed into the bundle.
4. **Safety geometry is never culled.** A lower quality tier may reduce detail.
   It may never hide a wire, a tower or a building (VIEWER_UI §15). Wires keep a
   minimum on-screen width.
5. **No plausible lies.** Detail that is not in the data is drawn as unknown or
   generic, never invented to look real (VIEWER_UI §18).
6. **Collision rules hold.** Physical collision never pushes an entity below
   terrain (`collision_aware_position` keeps clamping), and every new shape
   goes through the same guard.

## Phases

### R0: small fixes (behaviour-neutral)

- **Done.** Write `alphaMode: BLEND` on the overlay GLB materials, so layer
  `alpha` styles actually show. glTF allows only `OPAQUE`, `MASK` and
  `BLEND`, and its colours are stored straight, not premultiplied. Any
  premultiplication the renderer wants happens after import. A headless probe
  checks building translucency
  (`geometry_probes::a_building_is_drawn_at_its_styled_opacity`).
- **Done.** Name the world-model shapes explicitly, so each consumer matches
  on a shape, not on a `kind` string. Shapes were already the
  `ObstacleShape` enum (polygon, cylinder) for `los`, `first_hit`,
  `point_collides`, the planner and `scene_build`. What was still a string
  match was the hard-structure rule: `point_collides` and the range
  scanner's `first_hit` tested `blocker_type.as_str()` against
  `"building" | "wall"`, and the planner had its own copy. All three now ask
  `BlockerType::is_hard()`. The new kinds below add variants there. The one
  `kind` check left, `forest-stand-v1` in land-cover painting, is the
  reference's own rule (a stand is told from other prisms only by its tag)
  and stays.

### R1: real buildings

- **Georeferencing first.** OSM and GeoJSON come in external coordinate
  systems (GeoJSON defaults to EPSG:4326), and the simulation works in local
  meters. The first conversion therefore fixes, and records in the world
  bundle, the source CRS, the projection and the local origin, and aligns them
  with the terrain. R2, R3 and R5 reuse the same transform.
- **Data in.** An offline tool in Python (allowed as tooling) converts OSM or
  GeoJSON footprints, `height`/`building:levels`, `roof:shape` and
  `roof:height` into a versioned **world bundle** section. When the data
  has no height, use the tagged levels × 3 m, then a land-use default, and mark
  that height as *estimated*.
- **Model.** `building-v2`: a footprint prism plus a roof (flat, gabled, hipped
  or pyramidal, with ridge direction and roof height). Line of sight,
  `first_hit` and collision intersect the roof triangles as well as the prism.
  The old `building-v1` stays as it is.
- **The simulation sees what the viewer sees.** GIS buildings become
  `ObstaclePrimitive`s in the run, not just meshes.
- **Rendering.** Generate roof meshes and per-face normals. Add semantic
  material classes (concrete, brick, metal roof, glass) with a procedural
  facade shader for window rows from the level count. This needs no textures,
  so the bundle stays small and deterministic.
- **Validation.** Footprint area and height error against the source data. Line
  of sight against roofs in a parity fixture. A probe that a gabled roof
  occludes a ray that a flat-roof prism would let through.

### R2: vegetation

- **Model.** `tree-v1`: position, height, crown base, crown radius and a species
  class (conifer, broadleaf, palm, shrub). Trees are placed from data (OSM
  `natural=tree`, or a LiDAR canopy-height model) or scattered procedurally
  over Forest and Scrub land cover by density, seeded.
- **Sensing.** Each crown is an ellipsoid with a leaf density. Line-of-sight
  transmittance falls as the ray crosses crowns, as forest stands do now.
  Stands remain the cheap aggregate for large forests.
- **Range scanner.** Canopy gives partial returns with a probability set by the
  crown density, so belief mapping sees tree tops as soft structure instead of
  dropping them (UNKNOWN_WORLD.md:357).
- **Flight.** Trunks are hard. Crowns are a soft keep-out with vertical
  clearance in the planner (planning plan P3). Collision with a crown is a
  recorded contact, not a push-out.
- **Rendering.** Instanced low-poly meshes per species, with impostors at
  distance. Seasonal foliage reuses `foliage_density_factor`. Wind sway is off
  by default (C6: effects need G2 evidence).

### R3: power lines

- **Model.** `powerline-v1` has two parts:
  - **Supports:** poles or lattice towers, at a position, height and
    cross-arm layout.
  - **Conductors:** each span is a catenary between attachment points, with sag
    (or tension and temperature), conductor radius and count.
  
  Data comes from OSM (`power=line|minor_line|tower|pole`, `cables`,
  `voltage`) or is generated along roads, seeded.
- **Collision and clearance.** The catenary is approximated as a chain of
  capsules, with a clearance buffer that depends on voltage. The test is
  **swept**: the segment a drone flies between two ticks is checked against
  every capsule, not just its position at each tick. Otherwise a drone moving
  more than a capsule's width per tick passes through a wire unseen. A test
  covers a wire crossed between ticks, for both physical collision and
  clearance. Add a `structure_clearance` violation to the safety engine; today
  `ConstraintViolation` has no structure clearance at all. Deconfliction and the
  planner treat the corridor as a hard keep-out with vertical clearance.
- **Sensing.** Wires do not occlude. Their weakness is that sensors barely see
  them: the range scanner detects a wire only within a short, density-limited
  range. A run with no prior map then has to treat mapped corridors or a
  wire-hazard prior as dangerous. This is the realism that matters for
  civilian drone operations.
- **Rendering.** Towers as instanced meshes. Wires as screen-space lines with a
  minimum pixel width, never culled at any quality tier. Clearance corridors can
  be shown as an optional overlay.

### R4: viewer realism

- Semantic PBR materials per class (roof, facade, road, water, vegetation,
  metal), driven by the existing land-cover texture and the new material
  classes.
- GPU instancing for trees, poles and repeated buildings. Mesh LOD and
  impostors with camera-driven selection (D2 / P5.1 terrain LOD, extended to
  structures).
- Tune lighting, fog and shadow cascades against real scenes. Shadows stay at
  the Standard tier and above.
- **Evidence:**
  - frame budget p95 ≤ 16.7 ms on named hardware with a dense urban scene
    (VIEWER_UI §14);
  - perceptual-diff probes at LOD seams (C8);
  - a probe that wires and towers are present at every quality tier.

### R5: real-world scene ingestion

- One offline pipeline: DEM (GeoTIFF) + OSM (buildings, trees, power, roads,
  water, land use) + an optional LiDAR canopy and building-height model. It
  produces a content-hashed world bundle and its `.smartscene`.
- Georeferencing: the transform R1 defined and recorded, applied to the DEM
  and every layer, so the layers line up.
- **One canonical real-world scenario** (a small town edge with a transmission
  line), used for parity, performance and visual probes.

## Order and size

| Phase | Depends on | Size | Changes canonical replays? |
| --- | --- | --- | --- |
| R0 fixes | — | small | no |
| R1 buildings | R0 | medium | no (opt-in kind; new scenario) |
| R2 vegetation | R0 | medium | no (opt-in kind) |
| R3 power lines | R0, planning P3 | medium–large | no (opt-in kind) |
| R4 viewer | R1, R2, R3 (its probes cover wires and towers) | medium | no |
| R5 ingestion | R1–R3 | medium | no (new scenario only) |

R0 and R1 come first: real buildings matter to every mission, and making the
scene and the simulation agree removes a correctness gap that exists today.
R3 is the most valuable for civilian realism, but it needs runtime routing
(planning plan P3) before drones can actually avoid wires.
