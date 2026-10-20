# World and viewer realism plan: buildings, vegetation, power lines

**Status:** proposed 2026-10-12; updated 2026-10-19. R0 is done: the
opacity fix (#222) and typed structure matching. R3 is done: the model,
the safety clearance, lines in a run, drawing in the viewer, the planner's
corridor, the guard against wires, sensing wires and the clearance-corridor
overlay. R1, R2, R4 and R5 are not started.
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
  within the line's clearance of a conductor's surface. A span's conductors
  lie across the line on one level segment, so the volume's cross-section is
  that segment widened by the conductor radius plus the clearance: a
  stadium. `clearance_mesh` sweeps it along each span, a section at every
  piece joint holding the conductors' segment there, closed at the line's
  ends. It is one translucent amber mesh per run, unlit, both faces, casting
  no shadow. Unit tests check that every section point is exactly that far
  from the conductors' segment, in one flat section, and that the spans meet
  at their supports. A GPU probe checks, at Low, Standard and High, that
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
