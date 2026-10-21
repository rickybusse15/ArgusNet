# World and viewer realism plan: buildings, vegetation, power lines

**Status:** proposed 2026-10-12; updated 2026-10-20. R0 is done: the
opacity fix (#222) and typed structure matching. R1 is done: building-v2
roofs, the georeferenced import and material facades. R3 is done: the
model, the safety clearance, lines in a run, drawing in the viewer, the
planner's corridor, the guard against wires, sensing wires and the
clearance-corridor overlay. R2 is done: the tree model, trees in a run,
sensing through trees, flight round trees and drawing trees. R4 is done
but for lighting tuned against a real scene, which waits for R5. R5 is
not started.
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
  (`geometry_probes::a_building_is_solid_unless_drawn_see_through`). Since
  the R4 look pass, buildings are drawn solid by default, and the styled
  translucency is what the See-through Buildings toggle shows.
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
  *Done (model, R1.1):* `building-v2` in `argusnet_world::obstacles`
  (`Obstacle::building_v2`, metadata `eave_elevation_m`, `roof_shape`,
  `ridge_azimuth_rad`). The roof is a set of planes and the solid ends at
  the lowest: a gable is two, a hip or pyramid one per edge (convex
  footprints only; others are refused). Point, ray, path-length and surface
  queries clip against the planes exactly. The scene builder draws the same
  solid: walls rising to the roof line (split where it turns) and one facet
  per plane. Every other kind is unchanged, field and bytes (14/14 replays).
- **The simulation sees what the viewer sees.** GIS buildings become
  `ObstaclePrimitive`s in the run, not just meshes.

  *Done (import, R1.2):* `scripts/import_buildings.py`
  (`argusnet.world.buildings_import`) writes an `argusnet.buildings` v1
  document. It projects in float64 to local ENU around a recorded origin,
  takes heights and roofs from the tags as above, and marks what it
  estimated or could not model. It also checks each projected footprint's
  area against its geodesic area (worst 3e-10 on the test town). The
  native builder's `buildings_path` option replaces the procedural
  buildings with the imported ones, each standing on the run's own ground,
  and leaves out (and counts) any outside the map. The model refuses a bad
  one by name. The bundle and the replay carry the georeference, the
  document's hash and the counts, and the replay's `enu_origin` is the
  buildings' own. DEM terrain under them is R5.

  *Done (materials, R1.3):* the importer sorts `building:material` and
  `roof:material` into classes (facade: concrete, brick, glass, metal,
  wood, stone, plaster; roof: tiles, slate, metal, concrete, membrane,
  glass, green), with `generic` when the data does not say. It keeps
  `building:levels`. The scene builder colours walls and roofs by class
  and gives walls window coordinates: storey above the building's ground,
  metres along the wall, and the level count. Those are texture
  coordinates written only on roofed parts, so every other GLB is byte for
  byte what it was. The viewer's facade material (an extension of the
  baked `StandardMaterial`, so light, shadow, fog and translucency stay)
  draws window rows from them, only between the ground and the eave of a
  building whose storeys the data gives. A GPU probe checks windows appear
  on facades with levels, and not on roofs or on buildings without levels.
  R1 is done; LOS parity fixtures against roofs are covered by the model's
  tests.
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
  tree the model refuses is an error, never one that blocks nothing; every
  value must be within what a tree can be (at most 200 m tall, a crown
  radius of 100 m, a trunk radius of 20 m, 10,000 km from the origin), so
  no derived value overflows. The
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
  Scrub about one in three (mostly shrubs, the rest broadleaves no taller
  than 12 m, so only forest has tall trees); none stands in a forest stand
  (the stand is the aggregate for the trees inside it, so no crown is
  dimmed twice) or in a building. Species follow the terrain preset
  (alpine mostly conifer, jungle broadleaf and palm, the rest conifer and
  broadleaf), and each tree's height, crown and trunk are drawn from its
  species' ranges. Deciduous crowns (broadleaf, shrub) take the season's
  `foliage_density_factor`; conifers and palms keep theirs. Snow paints
  over forest in winter, but the trees are placed on the cover as it is
  without snow, so they stay. The forest is part of the world, like the
  land cover: seeded from the presets and the terrain's effective seed, not the run's
  seed, so seeds 7 and 42 fly over one forest. A medium default map holds
  about 7,800 trees; a map whose forest would hold more than 15,000 is
  thinned evenly to that. The trees go into the replay's meta as `trees`.
  Nothing in a run reads them yet: with `trees` on, every frame of a run is
  the frame it was without them.

  To measure the behaviour slices, the builder's opt-in `tree_inspection`
  option stands the mission's POIs inside the canopy: beside the crowns of
  tall forest trees, spread through the forest, at 16 m over the ground,
  in no crown or building and at least 10 m inside the map, so a drone flying between
  them passes through crowns. It cannot be combined with
  `power_line_inspection`; each places the POIs. The
  scenario: medium map, default terrain, 3 drones, 8 POIs, coverage
  threshold 0.3, 240 s, `trees`, `tree_inspection`, `safety_blocking`. All
  8 POIs are inspected (seed 7, efficient_v1).

  *Done (sensing):* line of sight goes through trees
  (`visibility::los_with_trees`, which is `los` exactly without them). A
  trunk stops the ray like a building (`blocker_type` `"trunk"`). Each
  crown crossed dims it by its optical depth, multiplied into the forest
  stands' transmittance, and counts as vegetation for the first hit, so
  three dense crowns in a row fall under the hard-block threshold. The run's
  `SharedVisibility` carries the world's trees, so bearing sensors, follow
  controllers and every other sight line see them. With trees added to the
  canonical target scenarios, observations over the run fell:
  `intercept_stress` 2810 → 2776 and 2888 → 2852 (seeds 7, 42),
  `sar_person_search` 205 → 193 and 104 → 52; `search_acquisition`
  (urban, its trees on scrub away from the targets) is unchanged.

  The range scanner returns from trees (`SharedVisibility::first_return`).
  A trunk returns as structure. A crown is leaves with gaps: at each crown
  a ray enters, in order, one uniform draw `u` stops it `-ln(1 − u)/(k·d)`
  metres in, and if that is inside the crown and short of anything harder
  it returns a **canopy** point there; otherwise the ray goes on. So a crown
  returns with probability one minus the transmittance `los` gives it
  (checked over 20,000 draws to within 1 %). The draws come from a stream of
  their own, keyed by node and tick, and are never made in a world without
  trees. The belief map keeps canopy as soft structure: per cell, the
  highest canopy return and a count. It marks the cell seen but never
  occupied, so nothing collides with a crown. Canopy rows go to the replay
  as `canopy_updates` (`[i, j, top_cm]`, each replacing the cell's last,
  written when first seen or moved 10 cm), the frame's range-scan sample
  carries `canopy` points, and `mapping_state.belief.canopy_cells` counts
  them. Each is written only when there are some, and the canopy arrays
  are allocated at the first canopy return, so a run without trees
  checkpoints as before. A believed structure on a trunk's cell counts as
  right in `structure_precision`; recall stays the buildings' and walls'.

  On the tree scenario with no prior map (`--world-knowledge none`,
  efficient_v1), the belief went from dropping the canopy to holding 6,936
  and 7,012 canopy cells (seeds 7, 42) of 15,246, with 40 and 30 trunk cells
  believed structure (precision 1.0). Fewer rays reach the ground under
  trees, so the ground error rose a little (RMSE 0.17 → 0.22 m and 0.18 →
  0.21 m). All 8 POIs are inspected before and after, with no world
  contacts. With a prior map and no targets, every frame is unchanged.

  *Done (flight):* a trunk is as hard as a building to physics. The run's
  `SharedObstacles` carries the trees, so stage 1's collision, the
  physical guard and guarded deconfliction push a platform out of a trunk,
  straight away from its axis to its surface plus the margin, and clamp it
  to the terrain after, as for any structure (never below it). The push is
  by point, as for buildings: a platform passing a trunk between ticks is
  recorded, not pushed. A crown is soft: nothing collides with it or is
  pushed out of it. Each frame records every tree a mobile platform touched
  in the last tick as `tree_contacts` (`node_id`, `tree_id`, `part`): the
  segment it flew, from where its velocity puts it a tick ago, swept
  against each trunk and crown. It is stateless, so a resumed run records
  what a straight one does, and written only when there are some.

  Under efficient_v1 a crown is a soft keep-out with vertical clearance:
  its radius plus 1 m from above, crossed at least 3 m over its top. POI,
  egress and gap-fill flights that would cross one lower climb over it,
  with the power lines' "over" profile (a climb at 1 in 8 where there is
  room, steeper when an end is near, never lower than the straight
  flight), and the journal records `route_trees: "over"`. A keep-out that
  holds a flight's end is left out: an end beside a crown cannot be above
  it. A route round a line's ends must keep above the crowns too. Trees
  are part of the prior map, forgotten with it in a `--world-knowledge
  none` run. The reference never reads them.

  On the tree scenario without `safety_blocking` (with it, the safety
  floor holds every drone 50 m up, over every crown, so no platform
  touches a tree under any policy), crown contacts under efficient_v1 went
  from 254 to 22 (seed 7) and from 243 to 30 (seed 42); all 8 POI flights
  climb over, and all 8 POIs are inspected. The contacts left are drones
  still on launch pads under crowns, return flights after the mission,
  egress, and hovering beside a POI's crown. Under the reference every
  position is identical before and after, with the safety floor or
  without it.

  *Done (drawing):* the viewer draws `meta.trees` (`trees.rs`). Each tree
  is a trunk and a crown, from one unit mesh per species, shared so the
  renderer batches them. Crowns are coloured from bare to full leaf in
  four steps of leaf density, which already carries the season. Past
  2 km a tree is a crossed-quad impostor. Bevy's `VisibilityRange`
  switches between the two, so nothing is recomputed as the camera
  moves. Trees are never hidden by a quality tier. A Trees toggle sits in
  Runtime Overlays, and a tree the model refuses is reported, not drawn.
  A GPU probe checks trees are drawn where the model stands them, at
  Close and Isometric on every tier, and a unit test checks meshes draw
  near and impostors far.

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
  climbs keep every published position 19 m or more from a wire.)

  *Done (guard and deconfliction):* the physical guard knows the wires
  (`PhysicalGuard::with_power_lines`). Each conductor piece is a capsule:
  the conductor's radius, plus the guard's contact clearance, plus the
  piece's sag error, round its chord, so it holds the true sagging wire
  (`PowerLine::envelope`). `enforce` pushes a platform inside one straight
  away from the wire to just past it. Where that would put it under the
  floor, it lifts it straight up over the wire instead; physics never
  pushes into the ground. `first_structure_hit` also meets the wires,
  solved against each capsule (`capsule_entry`: ray against cylinder and
  end spheres, not sampled), so a rejoin step stops short of a wire as it
  does at a wall. Under efficient_v1 two planning rules use the lines'
  own clearance. Deconfliction refuses a push whose flight since last
  tick (from where the safety monitor left the platform, swept) comes
  within a line's clearance, and tries the next position the reference
  would take: the mirror, the vertical offset, then the vertical offset
  the other way, or the one that keeps the most when all are refused
  (`resolve_step_keeping`; `resolve_step` is it with no margin, so the
  reference's parity cases are unchanged). A rejoin step that would come
  within a known line's clearance stops short of it and climbs over. Both
  are the planner's rules, not physics, so they record no contact. (A
  rejoin holding the height it climbed to also gives up a path running
  away faster than it flies, as one that did not climb does; it used to
  chase it for ever.) The
  reference and a run without a prior map know no lines; the physical
  guard keeps off the wires in every run that has one.

  Measured on `power_line_inspection` (the configuration above):
  efficient_v1 seed 7 goes from 2 `structure_clearance` events to 0, all
  8 POIs still inspected. Seed 42 and both reference runs are
  byte-identical. With the alpine terrain preset instead, efficient_v1
  seed 42 stays at 11: the 2 deconfliction pushes are gone, and a rejoin
  starting where the monitor's abort clamp left it, not where its own
  flight was, adds 2. That run and seed 7 (43 events, before and after)
  also show what this slice does not cover. A drone's own **scan lane**
  runs along the line at the wires' height, through them: stage 1's
  lanes are not routed round lines, and a prior run never puts stage 1
  through the guard. That is the next gap for the planner.

  *Done (sensing):* the range scanner sees wires (`range_scan.rs`). A wire
  never stops a ray: the ground behind it still returns. A scanner puts
  points on a conductor only while its points are no further apart than
  the wire is thick. So a span is reported (a `WireReturn`: the span as
  fitted, `PowerLine::span_line`, and its nearest point) only while one of
  its conductors is within `2 r / WIRE_RESOLUTION_RAD` (1 mrad, so 30 m for
  the builder's 3 cm wire) and below the scanner, whose fan looks down. A
  `--world-knowledge none` run maps a span the first time a scan sees it
  (`BeliefGrid::wires`, checkpointed, published with each snapshot), and
  learns wires late: on the inspection scenario the first is seen 58–85 s
  in. Under efficient_v1 the mapped spans are hazards, as the prior map's
  lines are in a prior run. POI, egress and gap-fill routes keep their
  clearance, deconfliction and rejoins keep out of it, and a POI flight a
  newly mapped wire lies across is flown again from where the drone is
  (`poi_reroute`), as is a flight home, on a drone's own reserve or in the
  mission's egress (`egress_reroute`). The detection range never exceeds
  the scanner's own `max_range_m`. The opt-in wire-hazard prior (`--wire-prior`,
  `ExplorationConfig::wire_prior`) gives the map every span from the
  start, as a utility's map of its corridors would. A span alone hangs
  as in its line, but where the line turns its outer conductors sit a
  little off (their offset times one minus the cosine of half the turn).
  The replay's range scans carry each wire's nearest point (`wire`).

  Measured with no prior map on `power_line_inspection` (the configuration
  above), efficient_v1, seed 42: `structure_clearance` events go from 47
  (wires never sensed) to 8 (sensed and mapped), and to 0 with the
  wire-hazard prior, which is as many as the same run with a prior map
  has. The 8 left are one drone crossing a span 1.25 s after a scan
  first mapped it, on stage 1's own exploration flight, already 23 m off
  and descending: learned too late to turn, and on the stage 1 path the
  previous slice found the planner does not route. Seed 7 and every
  reference run stay at 0, with all 8 POIs inspected and no physical
  contact in any run. Runs without lines are unchanged.

  *Done (clearance-corridor overlay):* a Clearance Corridors toggle beside
  Power Lines in Runtime Overlays, off by default and owned by no preset,
  shows each span's clearance volume from `meta.power_lines`: everywhere
  within the line's clearance of a conductor's surface: the union of a tube
  round each conductor, of its radius plus the clearance, since the safety
  test measures to each conductor on its own (the gap between two widely
  spaced wires is not drawn as inside). `clearance_mesh` sweeps each tube's
  circular section along every span, a section at each piece joint; at a
  support both spans share one section (its plane square to the mean of
  their directions), so a tube is unbroken round a turn, and it is closed at
  the line's ends. It is one translucent amber mesh per run, unlit, both
  faces, casting no shadow. Unit tests check that every section point is
  exactly that far from its own conductor, in one flat section, and that
  both spans' sections are the same ring at a turn. A GPU probe checks, at Low, Standard and High, that
  nothing is drawn by default, that the corridor is drawn 6 m over the
  centre conductor at mid-span when the overlay is on, and that nothing is
  drawn 150 m off the line. Nothing is left of R3 as planned. Two gaps found
  on the way are recorded above: stage 1's own lanes and exploration flights
  are not routed round lines, and a prior run never puts stage 1 through
  the guard.
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

  *Done (materials, R4.2):* each material class sets how its surface
  responds to light, not only its colour. The scene builder writes a stable
  class code on every roofed building's walls and roof (`material_code`).
  The facade material looks up roughness and metalness by it: glass is
  smooth, masonry rough, and metal moderately metallic. Metalness stays at
  or below 0.5, because the scene has no environment map and a fully
  metallic surface with nothing to reflect renders black. Generic keeps
  the baked surface. Water in the land cover (the legend's `water` code)
  is drawn smooth, so it takes the sky's highlight. Trees (R2) and towers
  (R3) already share one mesh per species and per height, and the trees
  switch to impostors past 2 km.

  *Done (the impostor seam, R4.3, C8):* a GPU probe stands two identical
  trees toward the scene's centre, 1850 m and 2150 m from the Close
  camera, either side of the 2 km switch from meshes to impostor. Both are
  drawn, and they differ by under a quarter of what either adds to its
  window (measured 0.09 against 0.66). An impostor a third its size, or in
  the trunk's colour, fails it. The headless probe output now records the
  camera's position, so probes can place things by distance.

  *Done (frame budget, R4.1):* on the D-5 reference machine (Apple M5 Pro),
  the realism fixtures stay under the budget in every tier: worst p95
  6.3 ms, p99 8.0 ms, no stall, over 75 quiesced runs at 1280x720,
  including a regional map with 15,000 trees
  ([record](performance/r4-frame-budget-2026-10-21.md)). Shadows stay on
  at Standard and above. Not done: lighting and fog tuned against a real
  scene, which needs R5's real-world scene.

  *Done (the look pass, R4.4, after a maintainer review):*
  - **Trees.** Each species has its own shape inside the model's crown:
    lobed broadleaf clusters, five serrated conifer tiers, palm fronds with
    leaflets, low shrub mounds. Trunks taper. Facets are flat-shaded and
    jittered, lighter towards the top. Each tree takes one of four mesh
    variants, one of three tints and its own turn from its id
    (deterministic), and impostors are cut to each species' silhouette.
    A test holds every vertex of every species and variant inside the
    crown.
  - **Buildings.** Plain procedural buildings take the same flat-roofed
    path as imported ones, so every building gets the facade surface, with
    a slight tint of its own. The facade shader adds a soft mottle, walls
    darker towards the ground, and, where storeys are known, floor slabs, a
    cornice and paned windows (taller on the ground floor), plus tile,
    slate and metal-seam courses on pitched roofs. Wall data is now
    TEXCOORD_0 = (height above the building's ground, along the wall) and
    TEXCOORD_1 = (storey height, or -1 on a roof; class + 32 x levels).
  - **Solid buildings.** Buildings are opaque by default, so the detail is
    not washed out and they cast shadows. A See-through Buildings toggle in
    Runtime Overlays draws them at the scene's styled translucency, to see
    drones behind them. Scene files are unchanged: the viewer switches each
    building material's alpha mode, once per part and change. A GPU probe
    checks the default matches the opaque building and the toggle shows the
    ground through it.
  - **Budget after the look pass:** the same realism fixtures, solid
    buildings and new trees, stay under budget on the D-5 machine: worst p95
    7.0 ms, p99 8.2 ms, max 9.3 ms, no stall, over 45 runs.

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
