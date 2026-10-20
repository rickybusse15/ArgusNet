# World and viewer realism plan: buildings, vegetation, power lines

**Status:** proposed 2026-10-12; updated 2026-10-19. R0 is done: the
opacity fix (#222) and typed structure matching. R3 is in progress: the
model, the safety clearance, lines in a run, drawing in the viewer and the
planner's corridor are done. R2 is in progress: the tree model and trees in a run are done.
R1, R4 and R5 are not started.
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

  *Done (model):* `argusnet_world::trees` holds `Tree` (`tree-v1`): the
  trunk's position and ground level, the tree's height, its crown base and
  crown radius, its trunk radius, its species and its leaf density (0 bare
  to 1 in full leaf). It is validated on construction and as read, so a
  tree the model refuses is an error, never one that blocks nothing. The
  trunk is a vertical cylinder from the ground to the middle of the crown:
  hard. The crown is an ellipsoid between the crown base and the top:
  soft. `BlockerType::is_hard()` stays buildings and walls; trees answer
  for themselves, part by part, so no new `BlockerType` was needed. Light
  through a crown falls off as through a leaf canopy (Beer–Lambert):
  `exp(-k · d · L)` over a chord `L`, with leaf density `d` and
  `k = 0.35 /m` (`G · LAD` for a spherical leaf-angle distribution and a
  leaf area density of 0.7 m²/m³). A 6 m chord at density 0.9 lets 15 %
  through.

  `TreeLayer` indexes trees on a 16 m grid, in id order so every answer is
  the same whatever order they came in. It answers what a run asks: the
  transmittance along a ray, the first trunk or crown a ray meets (or the
  first trunk only), and the trunk or crown at a point. A ray walks the
  grid cell by cell along its length and skips any cell it passes above
  the highest tree in, so a scanner's 400 m rays over a forest cost what
  the trees near them cost; a first-hit walk stops at the first cell past
  which nothing nearer can lie. A test checks the walk against testing
  every tree, on 400 rays in all directions over 60 random trees.

  `poisson_disk` is the sampler procedural placement uses: points no two
  closer than a spacing, seeded, drawn in a fixed order. Cells of side
  `spacing / √2` are visited row by row, each drawing a fixed number of
  candidates and keeping the first that is clear, so the stream of draws
  depends on the seed and the grid alone and thinning the sample by land
  cover moves no point kept.

  *Done (in a run):* the bundle's `world` carries `trees`, each read
  through the model's check (a tree it refuses, or two with one id, is a
  bundle error), absent when there are none and left out of the bundle's
  hash then. A bundle from data carries whatever trees it lists; the
  builder's opt-in `trees` option scatters them. Over the land cover's
  Forest cells every point of an 8 m Poisson-disk sample is kept, over
  Scrub about one in three (mostly shrubs); none stands in a forest stand
  (the stand is the aggregate for the trees inside it, so no crown is
  dimmed twice) or in a building. Species follow the terrain preset
  (alpine mostly conifer, jungle broadleaf and palm, the rest conifer and
  broadleaf), and each tree's height, crown and trunk are drawn from its
  species' ranges. Deciduous crowns (broadleaf, shrub) take the season's
  `foliage_density_factor`; conifers and palms keep theirs. Snow paints
  over forest in winter, but the trees are placed on the cover as it is
  without snow, so they stay. The forest is part of the world, like the
  land cover: seeded from the presets and the terrain seed, not the run's
  seed, so seeds 7 and 42 fly over one forest. A medium default map holds
  about 7,900 trees; a map whose forest would hold more than 15,000 is
  thinned evenly to that. The trees go into the replay's meta as `trees`.
  Nothing in a run reads them yet: with `trees` on, every frame of a run is
  the frame it was without them.

  To measure the behaviour slices, the builder's opt-in `tree_inspection`
  option stands the mission's POIs inside the canopy: beside the crowns of
  tall forest trees, spread through the forest, at 16 m over the ground and
  in no crown, so a drone flying between them passes through crowns. The
  scenario: medium map, default terrain, 3 drones, 8 POIs, coverage
  threshold 0.3, 240 s, `trees`, `tree_inspection`, `safety_blocking`. All
  8 POIs are inspected (seed 7, efficient_v1).

### R3: power lines

- **Model.** `powerline-v1` has two parts:
  - **Supports:** poles or lattice towers, at a position, height and
    cross-arm layout.
  - **Conductors:** each span is a catenary between attachment points, with sag
    (or tension and temperature), conductor radius and count.
  
  Data comes from OSM (`power=line|minor_line|tower|pole`, `cables`,
  `voltage`) or is generated along roads, seeded.

  *Done (model):* `argusnet_world::powerline` holds `PowerLine` (supports in
  order, conductor offsets across the cross-arm, sag ratio, conductor radius
  and clearance), validated on construction. Each span's conductors hang as
  parabolas (a catenary of real sag is one to within a millimetre) cut into
  pieces of at most 5 m. `clearance_along(from, to)` is the swept test: the
  distance from the segment flown between two ticks to the nearest
  conductor's surface, never more than the true one (each piece's distance
  is cut by how far the wire can sag below it).

  *Done (safety):* the safety engine has a `structure_clearance` violation,
  judged on the swept flight. With power lines set on stage 4
  (`SafetyStage::set_power_lines`), it keeps where it left each platform and
  checks the segment flown since then against every line near it, each by
  its own clearance. A run without lines keeps nothing and checks nothing,
  and its checkpoint is unchanged.

  *Done (in a run):* the bundle's `world` carries `power_lines` (validated
  as it is built, absent when there are none, and left out of the bundle's
  hash then), and stage 4 checks them under `safety_blocking`. The builder's
  `power_lines` option strings one transmission line across the map (30 m
  supports every 200 m, three conductors, 10 m clearance), seated on the
  ground and skipping a support that would stand in a building; it is laid
  out from the map alone, so it is the same for every seed. The line goes
  into the replay's meta as `power_lines`.

  *Done (drawing):* the viewer draws `meta.power_lines` (`power_lines.rs`).
  Each support is a lattice tower on its own ground height, its cross-arm
  along `PowerLine::cross_arm`. Towers of one height share one mesh. Every
  conductor piece is a quad expanded in screen space by its own vertex
  shader to at least 1.6 px, or to its true thickness when that is wider,
  so a 3 cm wire stays visible kilometres off. Wires are never culled, at
  any tier. The geometry is built once per run and not rebuilt as the
  camera moves. A Power Lines toggle in Runtime Overlays shows and hides
  them; no preset owns it. Each line is read on its own: one the model
  refuses is logged and not drawn, and the others still are. GPU probes check that the wire and tower are where the model puts
  them and that nothing is drawn 150 m off the line, at Low, Standard and
  High.

  *Measured before choosing the next slice:* on `planner_adversarial`,
  `baseline_coverage`, `industrial_asset_survey` and `persistent_long`
  (seeds 7 and 42, both planning policies, builder line on), no flight
  published in a replay came within 10 m of a conductor; the closest was
  26 m. Drones cruise at 100–260 m. So the physical guard against
  conductors (deconfliction pushes and override flights) would change
  nothing measurable yet. It matters once something flies low near a line:
  inspection of the line itself, or the R5 town-edge scenario.

  *Done (planner's corridor):* under efficient_v1 the POI, egress and
  gap-fill flights keep every line's clearance (`mission_driver.rs`). Each
  span is a corridor from above: a rectangle holding its conductors and
  their clearance (plus 1 m), with a crossing height that clearance above
  its higher support's attachment. A flight whose every leg already keeps
  the clearance, by the swept test (`PowerLine::violated_by` on each leg,
  never sampled), is flown as it was. Otherwise it is flown **over** the
  wires: at least the crossing height over each corridor it crosses,
  climbing to it at 1 in 8 where there is room (steeper when an end is
  nearer) and coming down after, never lower than the straight flight
  anywhere. A flight now carries a height at each corner for this
  (`GotoHover::routed_over`; serialised only when set). Or it goes
  **around** the line's ends: the planner routes round the corridors as
  walls. A POI or egress flight takes whichever costs less energy, and the
  journal records which (`route_power_lines`). A gap sweep only climbs
  over; one that cannot keep the clearance is not flown. A route that can
  do neither is flown as it was. Lines are part of the prior map: a
  `--world-knowledge none` run forgets them with the obstacles. The
  reference never reads them.

  To measure it, the builder's opt-in `power_line_inspection` option puts
  the mission's POIs along the line, 35 m off it on alternate sides at
  32 m over the ground, so a drone flying between neighbours crosses the
  wires low. On that scenario (medium map, default terrain, 3 drones,
  8 POIs, coverage threshold 0.3, 240 s, `safety_blocking`),
  `structure_clearance` events under efficient_v1 went from 69 to 2
  (seed 7) and from 26 to 0 (seed 42). Both remaining events are
  deconfliction pushing one drone sideways off another at a shared POI,
  outside any planned route. All 8 POIs are inspected before and after.
  The reference is byte-identical. With the line on in the canonical
  scenarios, the only change is `persistent_long` seed 7 under
  efficient_v1: one POI flight now climbs over, and its 8 events go to 0.
  (These events judge the proposed flight. The blocking monitor's own
  climbs keep every published position 19 m or more from a wire.) Next:
  the guard swept against conductors (deconfliction pushes, rejoins and
  override flights), then an optional clearance-corridor overlay.
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
