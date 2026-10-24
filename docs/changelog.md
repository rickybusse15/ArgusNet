# Changelog

## Unreleased — Rust runtime, scheduled lanes and world realism

- Sensor rig and moving pod (SR1, SR2). `--rig-profile provisional|FILE` in
  a `--world-knowledge none` run flies a declared rig.
  - The rig is validated before the run, against both mass limits, and
    recorded with its identity.
  - The pod's fixed rectangular LiDAR maps from a 3-axis gimbal with no joint
    limits: slew, settle, airframe occlusion, and truth vs estimated pose, so
    boresight bias shows in the map.
  - `--pointing nadir|sweep:A:P|body:Y,P,R`. The pod stows below cruise, and
    stowed or unsettled captures are refused by the map.
  - Every controller flies low enough for the LiDAR's furthest ray to reach
    the ground.

  Also fixed: stage 2's POI and egress overrides could leave a platform below
  the true ground in a no-prior run. Runs without a rig are unchanged. See
  `docs/migration/RIG_AND_POD.md`.
- Forward navigation camera as the primary obstacle-avoidance sensor (SR5c,
  first slice). `--nav-camera` in a `--world-knowledge none` run (or
  `--nav-camera-profile FILE`) adds a body-fixed camera:
  - its structure detections become safety obstacles the planners avoid, and
    never count as mapped ground or structure;
  - every controller is capped to the speed that stops short within its
    detection range, which fog shortens;
  - the range scanner drops its forward fan.

  The replay records the profile and the envelope in `meta.world_knowledge`,
  and the camera's frames in `range_scans` with `sensor: "nav_camera"` (drawn
  green in the viewer). Runs without the flag are unchanged. See
  `docs/migration/NAV_CAMERA_SAFETY.md`; the planner integration it starts is
  §8 of `docs/SENSING_REALISM_IMPLEMENTATION_GUIDE.md`.
- Client assurance package (S1.0, ported from the maintainer's 2026-10-12
  work). `argusnet client-package --replay R --output DIR` writes:
  - an unsigned, hash-listed manifest;
  - `assurance.json` and `assurance.md` (`argusnet.mission-assurance/v1`),
    carrying the F1 outcome record with its reasons;
  - the replay annotated with `summary.mission_assurance`, which
    `argusnet-replay` reads.

  Evaluation no longer reads battery from motion; it uses the recorded
  `battery_fraction`. Missing metrics stay `null` (shown as N/A).
  Impossible objective counts are refused. A run with no required
  objectives has no completion rate: never 100%, and not applicable to the
  pass/fail check. The stored P4.9c reference was migrated for that one
  field (225 runs).

- Real land cover from OSM (R5). `scripts/import_features.py` turns water,
  land use, natural cover and roads into an `argusnet.features` document;
  the scenario option `features_path` paints the run's land cover from it.
  Where the data says nothing the ground is open, or rock where steep:
  nothing is invented. Roads run through forests, buildings stay urban,
  trees grow only in mapped forests, and the procedural walls and forest
  stands are dropped. Every real layer must share one origin.
- Mapped trees and power lines (R5). The features document carries OSM
  trees (points and tree rows) and power lines, with their tagged sizes,
  tower heights, cables and voltage. The run stands them on its ground in
  place of the procedural line, keeps scattered trees 8 m from mapped ones,
  and counts anything it leaves out.
- Real terrain from a DEM (R5). `scripts/import_terrain.py` turns a GeoTIFF
  into an `argusnet.terrain` document on a map preset's height grid,
  around the same origin as the buildings importer. Each cell is placed
  through the DEM's own CRS, so rotated rasters and grid convergence line
  up. The new scenario option `terrain_path` builds the run on it. Trees,
  power lines and buildings stand on the real ground; buildings and
  terrain must share an origin. The replay records the import and its
  georeference. Without the option nothing changes.

- Viewer look pass (R4), after a maintainer review:
  - Trees are no longer a ball on a stick. Each species has its own shape
    (lobed broadleaf, tiered conifer, palm fronds, shrub mounds) inside
    the model's crown, on a tapered trunk, flat-shaded for texture. Each
    tree has one of four variants, three tints and its own turn, all from
    its id. Impostors follow each species' silhouette.
  - Buildings: every building gets the facade surface and a tint of its
    own, with weathering, and, where storeys are known, floor slabs, a
    cornice and paned windows. Pitched roofs show tile, slate or metal
    seams.
  - Buildings are solid by default. A See-through Buildings toggle in
    Runtime Overlays draws them at the styled translucency, to see drones
    behind them. Scene files are unchanged.

- R4 frame budget recorded on the D-5 reference machine (Apple M5 Pro):
  the realism fixtures, including 15,000 trees, stay under p95 16.7 ms in
  every tier (worst 6.3 ms; p99 8.0 ms; no stall), over 75 quiesced runs
  (docs/performance/r4-frame-budget-2026-10-21.md).
- Viewer: material classes set the surface, not only the colour (R4).
  Glass facades are smooth, masonry rough, and metal cladding and roofs
  moderately metallic. Generic keeps the layer's surface. Water in the
  land cover is drawn smooth. A unit test keeps every class the scene
  builder writes in the viewer's table, and keeps metalness low enough
  that, with no environment map, nothing renders black.
- Viewer: trees are drawn (R2). Trunks and crowns from one shared mesh
  per species, crowns coloured by leaf density (bare to full leaf), and
  crossed-quad impostors past 2 km, switched by the renderer's visibility
  ranges. Never hidden by a quality tier; a Trees toggle in Runtime
  Overlays. A replay without trees draws nothing new.

- Flight among trees (R2). A trunk is hard to physics: collision, the
  physical guard and guarded deconfliction push out of it as out of a
  building, never below the terrain. A crown is soft, and each frame
  records the trees a platform touched in the last tick, swept, as
  `tree_contacts`. Under efficient_v1 a crown is a keep-out from above
  (radius + 1 m) that POI, egress and gap-fill flights cross at least 3 m
  over its top, climbing over as over power lines (`route_trees: "over"`).
  On the tree scenario without the safety floor, crown contacts under
  efficient_v1 fell from 254 to 22 and 243 to 30 (seeds 7, 42). The
  reference flies as before, and the 14 canonical replays are unchanged.
- Sensing through trees (R2). Line of sight crosses crowns: each dims it
  by Beer–Lambert on its leaf density, and a trunk stops it
  (`blocker_type` `"trunk"`). The range scanner returns from a crown with
  the probability its leaves stop the ray (one minus its transmittance),
  from a stream of draws of its own, and from a trunk as structure. The
  no-prior belief keeps canopy as soft structure (seen, never occupied),
  written to the replay as `canopy_updates`, range-scan `canopy` points and
  `belief.canopy_cells`, each only when there are some. With trees on,
  observations in the canonical target scenarios fell by up to half
  (`sar_person_search` seed 42); a no-prior run over the tree scenario maps
  about 7,000 canopy cells. The 14
  canonical replays are unchanged.
- Trees in a run (R2). The bundle's `world` carries `trees` (`tree-v1`),
  each validated as read, absent when there are none and left out of the
  bundle's hash then. The builder's opt-in `trees` option scatters them
  over Forest and Scrub land cover from a seeded Poisson-disk sample (none
  in forest stands or buildings), species by terrain preset, deciduous
  crowns thinned by the season; seeded from the world, not the run. They go
  into the replay's meta as `trees`. The opt-in `tree_inspection` option
  puts the mission's POIs inside the canopy beside tall trees, for
  measuring the behaviour slices. Nothing reads the trees yet: frames are
  unchanged with them on, and the 14 canonical replays are unchanged.
- World: a tree model (R2, first slice). `argusnet_world::trees` holds
  `Tree` (`tree-v1`): position, ground level, height, crown base, crown
  radius, trunk radius, species (conifer, broadleaf, palm, shrub) and leaf
  density, validated on construction and as read. The trunk is a vertical
  cylinder to the middle of the crown (hard); the crown is an ellipsoid
  (soft) whose light falls off by Beer–Lambert, `exp(-0.35 · density ·
  chord)`. `TreeLayer` indexes trees on a 16 m grid and walks a ray through
  it cell by cell (transmittance, first trunk or crown met, what is at a
  point), skipping cells the ray passes above. `poisson_disk` is the
  seeded, fixed-order sampler procedural placement will use. Nothing in a
  run reads it yet; the 14 canonical replays are unchanged.
- `efficient_v1` fills the holes that hold a scan open (P4, first slice).
  Each drone loops its own sector, so the seams between sectors stay
  uncovered, and once coverage passes the threshold those enclosed holes
  keep the gap gate shut. Now, while they do, each hole gets the free drone
  that reaches it soonest, which sweeps it (decision `gap_fill`, a frontier
  override) and goes back to its lane once it is covered
  (`gap_fill_release`). The way there and every leg of the sweep go round
  the known obstacles at the sweep's height (a cell inside a building is
  left out, a sweep with a leg that has no route is not flown), and the
  sweep climbs to clear the known ground along all of it. On `planner_adversarial` (30 qualification seeds,
  120 s) runs that leave the scan go from 26 to 29, and POIs inspected from
  46 to 56 of 180. No seed got worse.
- `efficient_v1` flies a drone back to its own path when an override ends
  (a POI released, a recall dropped, a hole filled) at the redirect speed,
  instead of moving it there in one tick, at the mission's redirect speed,
  every step checked every metre along its length to keep it out of the
  ground and out of structures; a structure that stops it is climbed over rather
  than leaving it at its face. The reference moved drones
  thousands of metres in a frame. On `persistent_long` frames faster than
  60 m/s after launch go from 4 to 0. Checkpoint policy version 11. The
  reference is unchanged (14/14 replays).

- Viewer: a clearance-corridor overlay (R3). The Clearance Corridors toggle,
  beside Power Lines in Runtime Overlays and off by default, shows each
  span's clearance volume from `meta.power_lines`: a translucent amber tube
  round each conductor, of its radius plus the clearance, unbroken round
  the line's turns. A GPU probe checks it is drawn round the wires and nowhere off
  the line, at every quality tier. No simulation change: 14/14 replays
  identical.

- Wires are sensed (R3). The range scanner reports a power-line span
  only while one of its conductors is near enough for the scan's points to
  land on it (30 m for a 3 cm wire at 1 mrad) and below the scanner; wires
  never occlude. A `--world-knowledge none` run maps each span the first
  time it is seen and, under efficient_v1, plans round the mapped spans as a
  prior run plans round its lines (routes, deconfliction, rejoins), and flies
  again a POI flight a newly mapped wire lies across (`poi_reroute`).
  `--wire-prior` gives the map every span from the start. With no prior map
  on `power_line_inspection` seed 42, efficient_v1's `structure_clearance`
  events go from 47 to 8, and to 0 with the prior. The replay's range scans
  carry each wire's nearest point (`wire`, absent when none). The reference,
  runs without lines and the 14 canonical replays are unchanged.

- The physical guard keeps platforms off power lines' wires, and
  `efficient_v1` keeps deconfliction pushes and rejoins out of each line's
  clearance (R3). Each conductor piece is a capsule holding the true
  sagging wire; a platform inside is pushed straight off it, or lifted over
  it where a push would go into the ground, and a guarded step stops at it,
  solved ray against capsule. Deconfliction tries the mirror push, then
  the vertical offsets, before one that comes within a line's clearance
  (swept from where the safety monitor left the platform); a rejoin stops
  short of a line's clearance and climbs over. A rejoin holding the height
  it climbed to now also gives up a path running away faster than it flies
  (it chased one for ever). On `power_line_inspection` seed 7 the last 2
  `structure_clearance` events go to 0. The reference, efficient_v1 runs
  without lines, the canonical scenarios with the builder line on, and the
  14 canonical replays are unchanged.

- `efficient_v1` routes keep power lines' clearance (R3, the planner's
  corridor). A POI, egress or gap-fill flight that would pass a line closer
  than its clearance, by the swept test on each leg, is flown over the
  wires (climbing to the clearance over each span it crosses and down
  again; a flight now carries a height at each corner, serialised only when
  set) or round the line's ends, whichever costs less energy. The journal
  records which (`route_power_lines`). A gap sweep only climbs over, or is
  not flown. The builder's opt-in `power_line_inspection` option puts the
  POIs low beside the line, on alternate sides. On that scenario (seeds 7
  and 42) `structure_clearance` events under efficient_v1 go from 69 and 26
  to 2 and 0; the 2 left are deconfliction pushes. The reference is
  unchanged, and so are the 14 canonical replays.

- Buildings drawn by material, with window rows (R1). Imported buildings
  carry facade and roof material classes and their storeys. The scene
  colours walls and roofs by class, and the viewer draws window rows on
  facades whose storeys the data gives, none elsewhere. Unknown materials
  draw in the layer's own colour. Nothing else changes: other GLBs are
  byte-identical, and 14/14 replays are identical.

- Real buildings in a run (R1). `scripts/import_buildings.py` turns
  GeoJSON or OSM footprints into an `argusnet.buildings` document: local
  metres around a recorded origin, heights and roofs from the tags, and
  every estimate marked. A scenario config's `buildings_path` option puts
  them in the run in place of the procedural buildings, with their roofs,
  standing on the run's terrain. The bundle and the replay record where on
  Earth they are. Without the option nothing changes: 14/14 replays
  identical.

- World: `building-v2`, buildings with roofs (R1, first slice). Flat,
  gabled, hipped and pyramidal roofs as planes over the footprint; line of
  sight, collision and the range scanner's hits clip against them exactly,
  and the scene builder draws the same solid. No run produces one yet:
  14/14 replays identical.

- Viewer: power lines are drawn (R3). The replay's `meta.power_lines`
  becomes lattice towers on their own ground, plus conductors drawn as
  screen-space lines at least 1.6 px wide, so a 3 cm wire never vanishes
  with distance or at any quality tier. A Power Lines toggle sits in
  Runtime Overlays. A replay without lines draws nothing new. GPU probes
  cover the wire, the tower and an empty point beside them at all three
  tiers. `PowerLine::cross_arm` is now public. No simulation change: 14/14
  replays identical.

- World: a power-line model (`powerline-v1`, R3, first slice). Supports in
  order, conductors strung between them with their sag, and a swept
  clearance test: how close the segment flown between two ticks comes to a
  wire, so a wire crossed between two positions that are each clear is
  still caught. Model only; nothing in a run reads it yet.
- Safety: a `structure_clearance` violation (R3). Stage 4, given power
  lines, checks the segment each platform flew since the last tick against
  them, so a wire crossed between two clear positions is a violation.
- Power lines in a run (R3): the bundle's `world` carries `power_lines`, and
  the builder's opt-in `power_lines` option strings a transmission line
  across the map (same for every seed), recorded in the replay's meta.
  Without the option nothing changes: no key in the bundle, the replay or
  the bundle's hash (14/14 replays identical).

- World: what a platform collides with and a route goes around is asked of
  the obstacle's type (`BlockerType::is_hard()`: a building or a wall), not
  matched on its `blocker_type` string in `point_collides`, the range
  scanner's structure hits and the planner (R0). No behaviour change: 14/14
  replays identical.

- Viewer on Bevy 0.20 and bevy_egui 0.43 (supersedes Dependabot #244 and
  #247). Bevy 0.20 preprocesses shaders as WESL, not with naga_oil, so the
  sky, terrain and hologram shaders are now `.wesl` files with `import`
  statements, and the terrain shader calls the new three-argument
  `alpha_discard`. Other API moves: `MeshAabb::get_aabb` for
  `compute_aabb`, `get_mapped_range` returning a `Result` in the headless
  capture, and `anyhow::Context` imported under another name where the
  prelude's own `ContextExt` would clash. The GPU pixel probes, the headless
  render and the windowed smoke test all pass on it.

- Viewer: a drone's inspector reads its task from the replay's recorded
  assignment history (V4): the task at the frame's time, its last change
  with kind and reason, and how many changes came before. Without a history
  the task is still read off the POI statuses and the change rows read as
  unavailable. A live run drops the history of the replay loaded before it.

- `efficient_v1` prices a drone's reserve home along the route it would fly
  home when that route turns round a known obstacle (P3), each leg with the
  wind along it, instead of along the straight line. Return home at the
  reserve, safe handoff and the fleet-wide allocation all use it. A drone that
  could fly even twice the straight line home is not planned a route at all
  (`MAX_PRICED_DETOUR`), and routes between a POI and a station are kept
  across runs, so a run costs about 0.4 s more, once. On
  `urban_dense_inspection` no drone nears its reserve in 150 or 450 s, and
  every replay is unchanged. Checkpoint policy version 10. The reference is
  unchanged.

- `efficient_v1` keeps deconfliction's adjusted positions out of the ground
  and out of structures in runs with a prior map, as runs with no prior map
  already do; contacts are recorded in the replay. Checkpoint policy version
  9. The reference is unchanged.

- `efficient_v1` routes trips that start or end beside a wall or at the
  map's edge (P3, second slice): planned from and to the nearest clear points
  and joined by short legs that cross no obstacle. On `urban_dense_inspection`
  time spent inside buildings falls by two thirds; completion, distance and
  battery are unchanged. Checkpoint policy version 8. The reference is
  unchanged.

- `--planning-policy efficient_v1` routes around known obstacles
  (PLANNING_EFFICIENCY_PLAN P3, first slice): a POI or egress redirect whose
  straight line crosses a building or wall tall enough to hit flies a planned
  route around it (visibility-graph A*), and the fleet-wide allocation, team joins and safe
  handoffs cost the route's length. The redirect decision records
  `route_length_m` and `route_waypoints`. A run with no prior map, or a trip
  the planner cannot route, flies straight as before. The reference is
  unchanged.

- Terrain preset `urban_dense` and the measurement scenario
  `urban_dense_inspection` (`planner_adversarial` on it, 150 s): a flat city
  of 30 blocks, 70–110 m tall, on a 6 × 5 grid with streets of at least 42 m,
  so a POI-to-POI flight at 40 m usually crosses one. Built the same by the
  Rust builder and the Python reference (parity fixtures extended). The
  scenario is outside the canonical suite, so no recorded baseline changes.
  Every existing preset is unchanged.

- `--planning-policy efficient_v1` sends each drone home on its own when its
  battery reaches its reserve home (PLANNING_EFFICIENCY_PLAN P2), handing its
  POI on safely first. The reference still only egresses at mission end.

- `--planning-policy efficient_v1` hands POIs off safely (P1, third slice):
  a low drone's POI goes to the cheapest drone that can afford the dwell left
  and the trip home (a teammate first, needing no trip), or back to the pool
  if none can; the low drone stops flying to it. The reference is unchanged.

- `--planning-policy efficient_v1` forms teams (PLANNING_EFFICIENCY_PLAN P1,
  second slice): a drone left free after allocation joins a POI being
  inspected when it arrives in time to help and can afford the dwell and the
  trip home, and both credit the dwell. Recorded as a `team_assignment`
  decision and a `team_join` in the assignment history. On the qualification
  seeds teams rarely trigger and change little (short dwells); see the plan.
  The reference is unchanged.

- Sensors can capture at their own rates (P4.10b,
  `--sensor-rate NODE_ID=NUM/DEN` on `argusnet-rs sim` and the viewer's
  `--run`). A rated sensor captures at the first tick at or after each of its
  deadlines and takes nothing in between; every other sensor's observations
  are unchanged. Recorded in `meta.sensor_rates`, the stage's checkpoint and
  `run.json` (`argusnet-run/3`); tracks carry `evidence_time_s` in such a run.
  A rate faster than the tick is refused. Legacy runs are unchanged.

- Tracks record their evidence time (P4.10b). In a run whose tracks can be
  older than their frame (delivery modelled, or fusion at its own rate),
  every replay track carries `evidence_time_s`, the time it last took a
  measurement. The scorecard reports `track_measurement_age_mean_s`, the
  evidence age plus any coasting, and the viewer's track summary shows each
  track's age. Legacy replays carry no such key and are byte for byte
  unchanged.

- `--planning-policy efficient_v1` (PLANNING_EFFICIENCY_PLAN P1, first
  slice, opt-in) allocates a scan mission's POIs across the whole fleet at
  once: a deterministic minimum-cost assignment (Hungarian method,
  `argusnet-planning::allocation`) of every free drone against every pending
  POI. The cost is the trip's energy plus hover for the dwell, less a worth
  per priority level. A pair that would leave the drone below its reserve
  home (`dynamic_reserve_wh`) is never assigned, and a drone keeps a POI it
  holds. The default, `reference`, is unchanged: all 14 replays are
  byte-identical, and reset and resume are equal on every canonical run. A
  non-reference policy is recorded in `meta.planning_policy` and in the
  driver's checkpoint, which resumes only under the same policy. A
  `--run-dir` run records it in `run.json`, so `--recover` continues under
  it. Available on `argusnet-rs sim` and the viewer's `--run`.

- `assignment_thrash` no longer counts finishing a POI and taking the next
  as thrash: leaving a POI the replay reports completed restarts the drone's
  30 s window. Scenario comparisons now report the P0 time and effort
  figures and the assignment counts.

- The fusion lane can run at its own rate under `scheduled_v1` (B2.6,
  `argusnet-rs sim --executor lanes --fusion-rate NUM/DEN`, and the viewer's
  `--run ... --fusion-rate`). The ticks it sits out are kept (platforms and
  observations) and fused in order when it next runs, through the delivery
  network when one is modelled, so the tracks equal the every-tick run's at
  every tick it runs; between its runs the rest of the tick reads the tracks
  it last fused. Kept ticks are part of stage 7's checkpoint, the run
  directory records the rate, and a rate that would leave it idle for more
  than 1024 ticks is refused. Legacy runs are unchanged.

- The scorecard reports time and effort (PLANNING_EFFICIENCY_PLAN P0):
  `mission_makespan_s`, `scan_coverage_time_s`, `poi_service_time_mean_s`,
  `fleet_distance_m` and per-drone distance, battery used per drone and
  `idle_time_s`, all read from the replay and its assignment history. Reporting
  only; nothing a run does changes.

- Every Rust replay records each drone's task over the run
  (`meta.assignment_history`, PLANNING_EFFICIENCY_PLAN P0): the task it flies
  on each tick (a search lane, a followed target, an explorer, a POI, egress),
  each change with its time and reason, and the POI handoffs stage 9 applied.
  The scorecard reports `task_changes`, `poi_handoffs` and `assignment_thrash`
  from it. Nothing else in a replay changes:
  `scripts/replay_hashes.py --ignore-meta assignment_history` hashes the rest.

- Buildings, vegetation and walls are now drawn at their styled opacity
  (WORLD_REALISM_PLAN R0). The overlay GLBs carried translucent vertex
  colours, but their material had no `alphaMode`, so glTF's default `OPAQUE`
  drew them solid. Both GLB writers (`scene_export.rs`, `_glb.py`) now write
  `alphaMode: BLEND` for a mesh with any translucent colour, and leave opaque
  meshes as they were. Blending draws without depth writes, so each overlay
  object is its own glTF mesh and node (the viewer sorts them back to front),
  and translucent materials are single-sided with clockwise footprints'
  walls reversed to face out, so a building's far wall never draws over its
  near one. GIS scenes split per feature too, and road and water strips are
  drawn as the exact union of their segments and bevel wedges, facing up, so
  no part of a line is drawn twice -- not at a bend, nor where it retraces or
  crosses itself. A GPU probe checks the roof of a built building against the
  same building drawn opaque.

- `--mapping-rate NUM/DEN` (`argusnet-rs sim --executor lanes`, and the
  viewer's `--run`) runs the mapping lane at its own rate (B2.5). The ticks it
  sits out are kept -- positions, scans -- and integrated in order when it
  next runs, so coverage, zone inspection and the self-built map equal the
  every-tick run's whenever it has run; a checkpoint carries the kept ticks.
  The replay's `meta.lane_rates` records it as `mapping`.

- `mission_completion_rate` is observable (PLANNING_EFFICIENCY_PLAN P0). It
  read 1.0 for every Rust scan-map-inspect replay, because neither replay
  writer emits a top-level `mission` object; it now comes from the last
  frame's `scan_mission_state` POI counts. The stored acceptance reference
  (`docs/performance/p49c-reference.json`) is re-recorded with it on all four
  seed sets.

- Plans: [PLANNING_EFFICIENCY_PLAN.md](PLANNING_EFFICIENCY_PLAN.md) (fleet-wide
  allocation, reserve-feasible return home, runtime obstacle-aware routing,
  rebalancing coverage, coordination over delayed links) and
  [WORLD_REALISM_PLAN.md](WORLD_REALISM_PLAN.md) (buildings with roofs from
  real data, individual trees, power lines with catenary wires and clearance,
  instanced and LOD rendering, real-world ingestion). Proposed; nothing
  implemented.

- `argusnet-viewer --run ... --mission-rate NUM/DEN` runs the viewer's
  in-process run with the mission lane at its own rate: `SimWorker` steps it
  through the lanes executor (`SimWorker::spawn_with_rates`), so its frames are
  the CLI's `--executor lanes --mission-rate` run's. The rate is a run option
  (`RunOptions::lane_rates`): `apply` declares it on the run and records
  `scheduled_v1` in the replay, for the CLI and the viewer alike.

- `argusnet-rs sim --executor lanes --mission-rate NUM/DEN` runs the mission
  stage at its own rate in Hz instead of every tick (B2.4, the first lane off
  the legacy rate). A boundary it sits out is in the trace as skipped, with the
  reason; its next run is given the time since it last ran; a checkpoint
  carries the lag, so a resumed run matches an uninterrupted one. The replay
  records `meta.execution_model: "scheduled_v1"` and `meta.lane_rates`.
  Without the flag the run is the legacy run, byte for byte.
- `scripts/run_phase_acceptance.py --rust-arg ...` passes extra arguments to
  `argusnet-rs`, so a declared behaviour change is judged against the same
  envelopes.

## Unreleased — The terrain faces up, and the sun lights it

- Scene terrain was wound facing down, so the viewer lit it as a back face:
  no sunlight, only ambient, with the camera opened three stops to compensate.
  Sunlit surfaces washed out to white as a result, including the drones'
  self-built map. Both scene writers (Python and `argusnet-rs build-scene`)
  now wind the terrain facing up, the committed `scene.smartscene` terrain is
  rewritten to match, and the viewer's cameras use EV100 12, matching the
  12 000 lux sun. Terrain shows its relief, buildings show lit and shaded
  faces, and the self-built map is no longer pale. Renders:
  `docs/performance/terrain-winding-2026-10-12/`.
- Sensor volumes and zone walls are additive light, as documented: the
  hologram shader returned its weight as alpha, so under `AlphaMode::Add` they
  covered 85% of the ground behind them. Marker glyphs keep covering what is
  behind them, so they read over sunlit ground.
- The belief overlay is drawn over the terrain's brightness, not its colour,
  so sunlit ground no longer pulls the classes toward its hue.

## Unreleased — Tools → Scenario runs without Python

- The viewer's Tools → Scenario run now runs the Rust simulation on a thread
  in the viewer (D5) instead of starting `python -m argusnet sim`. The panel's
  options become an `argusnet.scenario-config`, flag for flag what the Python
  command was given, so the run is the one `argusnet-rs sim --config` makes.
  The progress bar follows the ticks actually run. `ARGUSNET_PYTHON` and
  `SMART_TRACKER_PYTHON` are no longer read: the viewer starts no process.

## Unreleased — Scenes without Python

- `argusnet-rs build-scene --replay R --output DIR` builds the same scene
  package as `argusnet build-scene`, in Rust. Over the same replay the two
  packages match vertex for vertex, and the viewer draws them identically.
  Like the Python builder it first checks the replay (non-empty frames, a
  supported `schema_version`, a positive `dt_s`, a matching `frame_count`) and
  writes nothing for a replay that fails.
- `argusnet-viewer --run <scenario>` no longer needs `--scene` for a run with a
  prior world: it builds the scene from the run itself, in process. After a
  Tools → Scenario run the viewer builds the new scene the same way instead of
  starting a second Python process.

## Unreleased — Search drones explore the edge of their own map

- In a `--world-knowledge none` run, search drones now fly toward the nearest
  part of the map that is still mostly blank, spread out so they map different
  ground, instead of flying lanes planned before the run. `--explore lanes`
  keeps the old lanes.
- The scan mission measures its progress on the drones' own map. On the
  regional scenario the map reaches 83% at 120 s (74% with lanes), the large
  building is fully mapped (44% with lanes), and the scan phase finishes at
  97.5 s and moves on to inspection. With lanes it never finished.

## Unreleased — Follow drones chase their own tracks

- In a `--world-knowledge none` run, follow drones now steer off the run's own
  fused tracks instead of the targets' true positions. Before the first track
  they head for where the target was when they were tasked; when a track is
  dropped they go to where it was last believed to be. `--pursuit truth` brings
  back the old behaviour for comparison.
- Over five seeds tracking accuracy is unchanged, follow drones sit closer to
  their targets on average, and they map 5–7 points more of the world. Their
  worst moments are worse: when a track wanders off a manoeuvring target, the
  drone follows it.

## Unreleased — Save the drones' map as a 3D world

- `argusnet-rs sim --world-knowledge none --export-scene DIR` saves the map
  the drones built as a scene package. It holds the measured ground and the
  buildings as 3D models (GLB), the map itself, and the run that built it.
  `argusnet-viewer --scene DIR` opens it like any other world, with the run on
  the timeline.
- The viewer's map readout has **Save map as scene**, which saves the map at
  the cursor, live or replayed.
- Buildings now come out at their real height. A range return that came down
  onto a roof was reported where it crossed the building's outline in plan
  view, up to 150 m above the roof. Roofs now match the truth to within the
  scanner's noise, and more of each building is found: 100% of the demo
  building (was 88%), and 44% of the regional one (was 13%).

## Unreleased — The drones' map is part of the viewer

- The terrain the drones build is drawn with the same terrain material as a
  real scene: the same colours, elevation bands and contour lines.
- The Reconstruction preset's belief overlay (safe, obstacle, unknown,
  confidence, staleness) now draws the drones' own map. Before this it only
  ever showed test data.
- Markers, zone walls and the camera rest on the drones' measured ground, and
  the "terrain data unavailable" alert no longer appears in a run with no
  scene.
- "Drones' map" has its own entry in the Layers panel, Split view shows it
  alone in the right pane, and it blocks picking like real terrain.

## Unreleased — The drones' map as a 3D world, with no scene needed

- `argusnet-viewer --run <scenario> --world-knowledge none` no longer needs a
  scene package. The viewer opens on the scenario's map bounds with nothing
  in them, and the world appears as the drones measure it.
- The map is drawn as real terrain: one continuous, lit surface, with a
  darker cut edge where the known area ends. Each building is a single solid
  traced from the cells the drones found, not a stack of per-cell boxes.

## Unreleased — See the map being built

- The viewer shows a no-prior run's map as the drones build it. The true
  terrain and buildings are hidden, and in their place is the ground the
  drones have measured: pale where it has few returns, solid where it has
  many, glowing where it is new. Buildings the drones have found are raised
  in magenta. Each drone's range returns are drawn as rays. A readout tracks
  how much is known, how close it is to the truth, and how many buildings
  were found.
- Range returns now report a building that stands in front of the ground. An
  earlier cut reported the ground behind it, so buildings were mostly missed.

## Unreleased — Start from no world knowledge

- `--world-knowledge none` (in `argusnet-rs sim` and `argusnet-viewer --run`)
  runs a scenario with the platforms knowing nothing about the terrain or the
  buildings. Each drone carries a range scanner; the returns build a map, and
  every planning decision reads only that map. Ground nobody has seen is
  treated as high, never as open air, so drones climb out cautiously and
  settle as the map fills. Physics still uses the real world, and each time a
  drone's map put it somewhere it could not be is recorded as a contact.
- Replays from such runs carry the map's changes (`belief_updates`), samples
  of each drone's returns (`range_scans`), contacts, and the map's progress
  and accuracy (`mapping_state.belief`). On the demo scenario the map is 87%
  complete after two minutes, within 1 m of the true ground, with no contacts.
  See `docs/migration/UNKNOWN_WORLD.md`.

## Unreleased — The viewer's in-process run takes the run options

- `argusnet-viewer --run` accepts `--fusion-lag-s`, `--delivery-delay-s`,
  `--source-delay`, `--delivery-jitter-s`, `--delivery-queue-capacity`,
  `--energy-profile` and `--faults`, as `argusnet-rs sim` does. Both apply them
  through `argusnet_sim::run_options::RunOptions`, so the same options give the
  same run. Unsupported options are refused before the window opens.

## Unreleased — Track error scored at evidence time (B4)

- Evaluation reports score each track against truth at the track's own time.
  Under a fusion lag, tracks describe the moment their evidence was taken, so
  `localisation_rmse_m` now measures the estimate rather than its age. It is
  unchanged for runs without delivery modelling.
- New `present_error_rmse_m` (the error if the track is taken as the present)
  and `track_evidence_age_mean_s` show what the lag costs. On the demo
  scenario the estimate holds at 13.5 m at every lag, while the present error
  grows from 13.5 m to 47.6 m at 1.5 s. Projecting tracks forward with the
  tracker's velocity made it worse, so tracks stay at their evidence time (see
  `docs/migration/DELIVERY_TIMING.md`).

## Unreleased — Sensor and link failure scenarios (F2)

- `argusnet-rs sim --faults manifest.json` schedules sensor outages, link
  delays and sensor restarts at given times. The run stays deterministic, can
  resume from a checkpoint taken mid-fault, and records the manifest in the
  replay. Invalid or impossible manifests are refused before the run.

## Unreleased — Hover costs energy, as an opt-in profile (D-3)

- `argusnet-rs sim --energy-profile physical_v1` charges hover power while a
  drone is airborne but not moving. Previously a stationary drone had unlimited
  endurance. The default (`legacy`) is unchanged, so existing results stand.
- Evaluation reports gain `battery_remaining_min`, the lowest battery level any
  drone reached as the battery ledger recorded it.

## Unreleased — Timestamped sensor delivery (B4)

- `argusnet-rs sim --fusion-lag-s ... --delivery-delay-s ...` models sensor
  evidence arriving later than it was captured, per source, with seeded
  jitter. Fusion uses each observation's capture time and keeps tracks ageing
  through gaps. Evidence it cannot use (late, duplicate, from before a sensor
  restart, or overflowing a queue) is recorded with the reason. Without these
  flags runs are unchanged.

## Unreleased — Scenario comparisons (F4)

- `scripts/compare_scenarios.py` compares variants of a scenario over the same
  seeds. The report shows each metric's spread per variant and the paired
  difference against the baseline with a 95% interval. Failed runs, runtime
  cost and everything needed to reproduce the comparison are recorded.

## Unreleased — Run a simulation inside the viewer (P5.0)

- `argusnet-viewer --scene <scene> --run <config>` runs the scenario in
  process, in Rust, starting paused. Run, Pause, Step, Speed and Reset drive
  it. Each control shows as pending until the run confirms it, and refusals
  say why. The timeline keeps every tick for review.

## Unreleased — Parallel observation generation (B3)

- `argusnet-rs sim --workers N` evaluates sensor/target pairs on N worker
  threads. Output is identical at every worker count. It pays off on large
  fleets (2.7x faster ticks at 24 drones and 48 targets on 4 cores) but not
  on small ones, so the default stays 1.
- With `--workers`, scan missions of 12 or more drones also run each drone's
  localization on the worker threads (1.5x faster ticks at 32 drones).
  Smaller fleets are unaffected.

## Unreleased — Security scan triage (G3)

- Dependency updates fix three reported vulnerabilities in the viewer's
  libraries (XML parsing, browser launching) and two unsound crates.
- The Python static analysis check is now required to pass.
- Mission validity checks now log a platform they could not evaluate
  instead of skipping it silently.

## Unreleased — Sensor types and tracker modes are checked (G0)

- A configuration or scenario naming a sensor type other than optical,
  radar, thermal, lwir, mwir or ir is refused; replays may now say thermal.
  An unknown tracker association mode is refused instead of quietly running
  as labeled association.

## Unreleased — Version checks (G0)

- Python now refuses a replay written in a different major schema version
  instead of reading it silently, and refuses an environment bundle of an
  unknown format. Recovering a run directory written by a build with a
  different checkpoint format is refused with both formats named.

## Unreleased — Resource growth checked (C8)

- `argusnet-viewer --headless --lifecycle N` reloads, seeks, toggles
  layers, changes quality and projection N times over and records what the
  viewer holds after each round; a test requires those counts to stay flat.

## Unreleased — Renderer fixtures and frame statistics (E2)

- `argusnet-viewer --headless --frame-stats N` times N frames after the
  still and writes their distribution, the warm-up and the graphics adapter
  beside it. New `--camera close` and `--camera low-pitch` presets.
- `scripts/render_fixtures.py` renders the five standard views in every
  quality tier and writes one manifest; a software-renderer baseline is in
  `docs/performance/`.

## Unreleased — Preferences: scale, motion, keys (C7.2)

- Tools → Preferences… sets the interface scale (75–200 %) without
  changing the map, turns on reduced motion (camera moves arrive at once,
  no interface animations), and lists every key with a Rebind button. A key
  already in use is refused with the action that has it; Reset restores the
  defaults. Everything is remembered.

## Unreleased — Failure and degraded states (C7.1)

- Problems now show on the map as a banner that says what is affected,
  what you are still looking at, and what to do next, most serious first:
  no source, a live link reconnecting or dropping frames, belief data
  missing under the belief overlay, a failed simulation run, and a replay
  written in a newer format than this viewer reads. Where the viewer can
  act (hide the overlay, open the Scenario panel) the banner has a button.

## Unreleased — Layer presets and quality tiers (C6)

- The Operational, Reconstruction and Safety presets now show what the
  design says each is for; the viewer starts in Operational.
- Tools → Quality chooses Low, Standard or High rendering (also
  `--quality`), and the choice is remembered. Tiers change shadows, haze and
  glow only, never what is shown. Glow (bloom) is now off unless High is
  chosen.
- The belief overlay looks the same at every tier: its colours no longer
  fade with haze, and the "unknown" hatch is bluer and more clearly striped.

## Unreleased — Section view (C5.2)

- Section shows a vertical slice through the world, looking north or east:
  the terrain's profile and the buildings it cuts are drawn, and everything
  outside the slice is hidden and cannot be clicked. PgUp/PgDn move the
  slice, X turns it, - and = make it thinner or thicker; the overlay on the
  map has the same controls and says that hidden is not the same as empty.

## Unreleased — D-12 confirmed

- The Rust runtime's covariance-reduction metric, now judged by its median
  across seeds, matches the reference on 30 fresh seeds: all seven canonical
  scenarios pass acceptance.

## Unreleased — Top-down view (C5.1)

- Top-down is now a true plan view: straight down, north up, at the same
  scale the 3D view had. V switches between it and the perspective view;
  pan and zoom work in both.
- Fixed a start-up race that could leave the whole operator interface
  invisible.

## Unreleased — One timeline (C4)

- Time is controlled in the bottom timeline only. Its buttons now include
  previous / next event, and every button does exactly what its key does.
- The timeline shows the run's epoch, elapsed time and frame, marks events
  in lanes (mission, safety, command, evidence, fault), and names the
  current frame's events next to the scrubber.

## Unreleased — Map-cell inspector (C3.4)

- Clicking empty ground shows what the belief map says about that spot: its
  classification, whether it was ever observed and when, and the confidence
  the terrain is drawn with. Places never observed say so rather than
  showing a confidence of zero.

## Unreleased — Safety-event inspector (C3.3)

- The mission rail lists each frame's safety events. Opening one shows its
  decision, the constraints it violated, what it changed and when, with
  buttons for the snapshot before and after and for selecting the platform.
  Esc closes it.
- Clicking in the side panels no longer reaches the 3D world behind them,
  so it no longer clears the selection.

## Unreleased — Fleet index; Tab works every time (C3.2)

- The mission rail's fleet is a numbered list: click a platform to select it,
  as if picked in the world.
- Tab steps through entities every time it is pressed. Before, the second
  Tab moved focus onto a header button instead, and Space then pressed that
  button.

## Unreleased — Platform inspector (C3.1)

- Selecting a platform shows its state in eleven ordered sections: identity,
  actual and commanded state, task, localization, dynamics, battery, link,
  sensors, safety and onboard executive. Anything the replay does not carry
  says so and why, instead of showing zero. Legacy estimator outputs are
  marked as such.

## Unreleased — Shell fit and finish

- The timeline holds the play, step and speed controls, and fits its region;
  the header wraps instead of overlapping in narrow windows.
- Sliders show their track again, and the status line says "no mission
  state" rather than "no replay" when a replay simply has none.
- CI runs the real viewer window on a virtual display, so a crash in the
  operator shell can no longer pass.

## Unreleased — Workspaces, projections and presets (C2.3)

- The header has three separate controls: the workspace (Operate, Plan,
  Review), the projection (Perspective now; Top-down and Section later) and
  the layer preset (Operational, Reconstruction, Safety). Each has a key: O,
  V and P. None of them changes what is selected.
- Plan and Review say what they are waiting for instead of showing
  placeholder data.
- The Mission / Replay / Diagnostics / Scenario panels moved into a Tools
  menu; T opens and closes the panel.

## Unreleased — Source indicator (C2.2)

- The header always says what the window is showing: `REPLAY`, `LIVE` or
  `LIVE / HISTORY`, or `EVALUATION / TRUTH ENABLED` when ground truth is on.
  Hover it for what that source permits.

## Unreleased — Operate shell layout (C2.1)

- The viewer is laid out as the operate shell: a header, a status line with
  the mission phase, time and alert counts, a mission rail (phase, fleet,
  alerts) on the left, the selection inspector on the right and the timeline
  below, leaving the world as the dominant region. In narrow windows the rail
  folds away first, then the tool panel, then the inspector's secondary
  fields; alert counts never disappear.
- The Mission / Replay / Diagnostics / Scenario panel now opens on demand
  (the mode buttons or Show Panel) instead of at start-up.

## Unreleased — Viewer UI split by responsibility (C2.0)

- The viewer's single 3 100-line UI file is now shell, timeline, inspector,
  command and widget modules. Nothing on screen changes.

## Unreleased — Fusion, mapping, mission and publication as lanes (B2.3)

- `argusnet-rs sim --executor lanes` runs fusion, mapping, mission and
  publication as their own scheduled tasks, each in its own phase, after the
  compound lane. The replay is byte-identical to the legacy executor's.

## Unreleased — Why the mission did what it did (F3)

- The mission driver records each decision it makes -- coordinator, POI
  assignments and handoffs, completions, egress, phase changes -- with what
  it chose from, why, on which inputs, and the command it issued. Recorded
  runs (`--run-dir`) journal them every tick.

## Unreleased — Mission outcomes with reasons (F1)

- Scorecards now say how a run ended -- completed, aborted, incomplete, or
  objectives unmet -- and why, with each reason citing the ticks, safety
  events and POIs behind it. Metrics a replay cannot support are marked
  unavailable instead of reading as zero.

## Unreleased — Crash-recoverable runs (A4)

- `argusnet-rs sim --run-dir DIR` records a run so that it survives a crash:
  its configuration, a durable journal of every tick, checkpoints and a
  completion marker. `--recover DIR` finishes a crashed run, and its journal
  then matches an uninterrupted run's exactly.

## Unreleased — Truth isolation (A3.4)

- A stage that may not read truth is no longer lent the stages that may
  (VIO and observation generation), whose state is derived from truth.
- New tests hold the measurements fixed, change where the targets really
  are, and require everything downstream to stay bit-identical.

## Unreleased — Reset equals fresh, restore equals uninterrupted (A2)

- `argusnet-rs sim --reset-at K` resets the run after K ticks and runs it
  again; the replay is byte-identical to a straight run.
- Checkpoints are format 4: the header carries the scenario's content hash,
  and a checkpoint from a different scenario is refused before anything
  changes. Format-3 checkpoints are refused with a version error. Replays
  record `checkpoint_format: 4` in their provenance.
- `scripts/replay_hashes.py --lifecycle` checks reset and resume on every
  canonical scenario.
- A production run is built with `Simulation::production`, which refuses a
  missing required stage by name (A1.3).

## Unreleased — Visual tokens and the bundled face (P2.5a)

- The viewer's UI uses one palette, spacing scale and set of type roles from
  the specification. Corners are square, rules are 1 px, panels are flat, and
  selection and focus use the amber accent.
- The UI font is Arimo (SIL Open Font License 1.1), bundled in the binary:
  metric-compatible with Helvetica, with tabular digits.
- Muted text is `#767F83`, not the reference `#687175`, so that all UI text
  meets WCAG AA contrast.

## Unreleased — Cheaper mission transitions (P4.10c)

- Stage 9 searched the whole coverage grid for enclosed gaps on every tick,
  locking the mapping state once per cell. It now searches only on the
  scanning gate, where the reference does and the only place the result is
  read: `persistent_long`'s tick drops 41 % (0.516 → 0.304 ms), with
  byte-identical replays.

## Unreleased — Per-stage timing for the Rust tick (P4.10-pre)

- `argusnet-rs sim --stage-timing` prints each stage's p50/p95/p99 and share
  of tick time; `--timing-json` carries the same per stage, plus replay
  write time. Off by default; the replay is byte-identical with it on.
- `scripts/profile_rust_tick.py` profiles the canonical scenarios. The first
  profile (`docs/performance/rust-tick-profile-2026-10-03.md`) puts
  observation generation at 79 % of the tracking scenarios' tick and
  mission transitions at 46 % of `persistent_long`'s.

## Unreleased — Parity tests off their capture platform

- The Rust parity suites passed only where their fixtures were captured
  (macOS aarch64). On Linux, nine tests in six suites failed by one ULP of
  libm rounding. Fixtures now record `_captured_on`; comparisons stay bit
  for bit there and allow 2 ULPs elsewhere, recorded oracles included
  (`docs/migration/PARITY_PORTABILITY.md`).

## Unreleased — Camera controls and bindings (P2.4b)

- `F` focuses the selection, `L` follows it and `C` chases it, and `G` flies
  freely (`W A S D`, `E`/`Q`, `Shift`, right-drag to look, scroll for speed).
  Transitions are damped, on render time. Losing a followed entity holds its
  last known position and says so.
- `1`-`9` recall camera bookmarks and `Ctrl`/`Cmd`+digit saves them, per
  scene, in `~/.config/argusnet/viewer.json` (or `$ARGUSNET_VIEWER_PREFS`).
- Every key is remappable in that file. No binding fires while a text field
  has the keyboard, and `Esc` leaves a camera mode before it clears the
  selection.
- The top bar shows the camera's mode and notices.
- Fixed: `M` did nothing in a scene without a replay.

## Unreleased — Time controls over source time (P2.4a)

- Replays play and scrub by source time, not frame count. A recording with
  unevenly spaced frames now plays at the rate it was recorded, and blends
  between frames by the true fraction of each step.
- Nothing is interpolated across a recording gap or a teleport; the frame
  before is shown as recorded.
- The timeline scrubs in seconds on a time axis. It shades recording gaps,
  says when live history was trimmed, and labels live history as read-only.
- Loop playback, and `[` / `]` to jump to the previous or next event
  (safety events now count as events).

## Unreleased — Picking and selection (P2.3)

- Clicks pick by ray against the scene's meshes, not by an 18-pixel radius:
  the nearest visible marker under the pointer, and nothing when terrain or
  a building is in front of it. Sensor volumes, trails and zone walls never
  take a pick. A drone is picked across its whole glyph, including the gaps
  between its arms.
- In split view only the main viewport picks; a pointer over the UI never
  reaches the world.
- The selected marker has an amber ring and the hovered one a blue ring,
  hidden by terrain like the glyph itself.
- Tab and Shift+Tab step through the shown markers, Escape clears, and the
  inspector lists every shown entity by kind.
- Fixed: a click on a paused frame showed empty details until the frame
  moved.
- Fixed: picks went through the ground. The committed terrain is wound
  facing down (recorded in KNOWN_GAPS), so the picker now tests both faces
  of terrain and buildings.
- Trails are additive, like every other overlay, so their draw order can no
  longer flip.

## Unreleased — Sensor, zone and trail geometry (P2.2)

- Fields of view are persistent meshes: the region stage 6 actually accepts,
  a spherical sector along the runtime's own look (the look rule moved into
  `argusnet_sensing::look`, shared by the sim and the viewer), or the range
  sphere when the sensor does not gate. An unset range draws nothing,
  instead of an invented 400 m. A volume that encloses the camera is hidden.
- Radar ranges are rings draped on the terrain, built once per station.
- Mission zones are extruded walls along each overlap group's contour,
  taller for exclusion zones; focus changes restyle them without a rebuild.
  The gizmo outline now draws only with the diagnostic glyphs.
- Trails are ribbons sampled by simulation time and distance, bounded in
  memory, restarted on teleports and removed with their entity. Playing to a
  frame and seeking to it give the same trail. A new **Trails** toggle.
- Fixed: a zone contour could come out as half a ring, when the circle's
  padded edge fell on the marching-squares grid's last column.
- Pixel probes check the volumes, walls and trails in real renders; CI's
  viewer job runs them.

## Unreleased — Belief and land cover on the terrain (P2.1)

- The terrain material draws belief from `argusnet-mapping`'s tile patches
  through one adapter (`BeliefRenderSnapshot`), whose classes and confidence
  are `WorldBeliefQuery`'s, cell for cell. Categorical channels are never
  filtered. Unknown cells are hatched and stale ones faded. With no belief
  data the whole terrain is drawn as unavailable, and the overlay panel says
  why. Changed tiles reach the GPU as tile writes; uploads are capped and
  counted.
- Scenes built from an environment carry `metadata/land_cover.json`
  (additive), and the terrain is tinted by the runtime layer's class.
- Pixel probes read real headless renders back against the CPU queries; CI's
  viewer job runs them. They caught the belief mode written into the wrong
  uniform lane.
- Headless renders now exit non-zero when a shader fails to compile or a PNG
  cannot be written, instead of saving a plausible image.

## Unreleased — The viewer presents frames without rebuilding them (P2.0)

- A snapshot adapter (`argusnet-viewer::presentation`) replaces the per-frame
  marker rebuild. Entities have typed, interned keys; membership is rebuilt
  only when the source frame changes, poses are re-blended in place, and a
  paused frame does nothing. On a 240-entity replay, allocations per rendered
  frame fell from 1,108 to about 5 playing, and from 1,210 to about 2 paused.
- Marker entities persist, keyed and indexed, and are written only where
  something changed. Presentation epochs (reload, or a live session change)
  void keys, entities, selection and interpolation.
- Fixed: selecting a truth resolved to the track that shares its label.
- The viewer's `Update` runs in ordered `ViewerSet`s: Input, SnapshotApply,
  Presentation, Picking, Overlays.

## Unreleased — Blocking safety and mission profiles, with no Python (P4.9d)

- Stage 4, the hard safety monitor, is assembled when a bundle says
  `"safety_blocking": true`. Both bundle writers set that key only when the
  option is on. Assembling it found three arguments that were not the
  reference's: `dt_s` was the tick's timestamp, the caution fraction was 0.85
  where `SafetyMonitor` defaults to 0.9, and the terrain was one constant.
  The stage is now on the checkpoint contract, its validations reach fusion
  as `safety_events`, and replays carry them.
- Parity: an all-follow run and a whole scan mission under the hard floor
  match the reference bit for bit on every tick, every safety event included.
- The native builder builds mission profiles: the profile's zones and POIs,
  exported from the reference on a unit map, the profile's scenario name, and
  `meta.mission_profile`. It validates profiles as `ScenarioOptions` does.
- All seven canonical scenarios are now scored. Six pass every envelope.
  `sar_person_search` fails `covariance_reduction_mean`, which here measures a
  single coasting track per seed at most; it is analysed and not widened.
- Finding, kept for parity: after an abort, the reference monitor keeps
  aborting, because it differences the next command against the zeroed
  velocity it allowed.

## Unreleased — Phase acceptance for the Rust CLI (P4.9c)

- `scripts/run_phase_acceptance.py` runs the seven canonical scenarios on five
  seeds. The reference is `argusnet sim`; Rust is `argusnet sim
  --export-config` then a release `argusnet-rs sim --config`. It applies the
  envelopes `docs/migration/PHASE_ACCEPTANCE.md` declared before the first
  run. All five supported scenarios pass. The tick p95 is at most 0.21 ms,
  startup at most 43 ms, and the tick loop 16–81x faster than the reference's.
  The two mission profiles are refused (blocking safety; P4.9d).
- **Fusion fix (`argusnet-core`, both runtimes):** the IMM filter normalised
  its mode probabilities by `max(sum, 1e-12)`. When both likelihoods
  underflowed, the probabilities fell to about 1e-20, and the track collapsed
  to the origin for good. They are now divided by the true sum.
- **Metric fix:** `compute_localisation_rmse` paired tracks with truths only
  in the replay's first tracked frame, by distance. Later tracks went
  unscored, and one passing another target was scored against it. Each track
  is now paired at its own first appearance, and by target ID where the track
  is named for one.
- Direct fusion parity: `tests/test_direct_fusion_parity.py` replays the
  daemon's recorded requests through a hidden `argusnet-rs fuse`. Every
  response must be the same protobuf message.
- `argusnet sim --export-config PATH` writes the versioned scenario config the
  flags resolve to. `argusnet-rs sim` records `provenance` (runtime, version,
  formats, input and resolved config) in the replay's meta, and writes
  `--timing-json`.
- Rust replay tracks now carry `lifecycle_state` and `quality_score`, as the
  reference's do. Both are declared, optional, in `docs/replay-schema.json`.

## Unreleased — Scenarios from configs, with no Python at all (P4.9b.2b)

- `argusnet-rs build --config C --bundle B` writes the bundle `argusnet sim
  --export-bundle` would, and `argusnet-rs sim --config C` builds and runs it
  in process. The native builder ports the scenario half of
  `build_default_scenario`: targets, ground stations, drones and their
  controllers, sensors, mission zones, POIs and the mission section.
- Structurally identical to Python's bundles (every key, string, integer and
  length) and numerically within 1e-12 m, over the three committed bundles,
  an 11-scenario preset matrix, and a Python-vs-Rust round trip through both
  CLIs.
- The builder's constant tables are exported from the reference
  (`scripts/export_builder_tables.py`) and checked for drift.

## Unreleased — numpy's `integers` and the target behaviours in Rust (P4.9b.2a)

- `NumpyRng` reproduces `Generator.integers` (PCG64's `next_uint32` half-word
  buffer and Lemire's bounded method) and `random()`, proven against 684
  interleaved draws.
- `builder::behaviors` ports `build_target_trajectory`'s eight presets,
  bit for bit over 10,464 samples.
- Recorded: numpy 2.5 on arm64 fuses `uniform`'s affine step into an FMA, so
  the reference's `uniform` is platform-dependent in the last place.

## Unreleased — The native builder builds the world (P4.9b.1)

- `argusnet_sim::builder` reads an `argusnet.scenario-config` v1 document
  (`ScenarioOptions` field for field, unknown options refused) and builds the
  world Python's `build_default_scenario` builds: terrain, the environment's
  obstacles and land-cover patches for every terrain and map preset, and the
  land cover. It writes the bundle's `world` section and the world keys of
  `presentation`.
- Exact against the reference except for terrain heights and what derives
  from them, which are within 1e-12 m (numpy's `exp`, P4.3), across a
  27-case preset matrix, 600 random box and wall footprints, and the three
  committed bundles.

## Unreleased — `argusnet-rs` replays carry what the viewer shows (P4.9a.2)

- Rust replay frames now carry generation rejections (with the reference's
  detail text and sight-line points), mapping state, zone inspection events,
  deconfliction events, localization state, target metadata and, under
  `scan_map_inspect`, the full scan mission state (POIs with positions,
  coordinator, newly scanned cells, egress progress, team localization).
- Replay `meta` carries the scenario half of the reference's metadata
  (terrain mesh, obstacles, land cover, zones, assignments), exported in the
  bundle as `presentation` by the newly lifted `scenario_replay_metadata`,
  plus the tracker thresholds, durations and generation totals.
- Stage 8 runs in every mission mode and checkpoints per-zone coverage and
  membership; stage 9 elects and checkpoints the coordinator.
- Bundle v1 gains optional `coverage_bounds`, `targets[].behavior`,
  `cooperative_search`, `requested_duration_s` and `presentation`.
- CI builds `argusnet-rs` before pytest and sets `ARGUSNET_REQUIRE_RS`, so
  the Rust round-trip tests fail rather than skip when the binary is missing.
  Until now they were skipped in CI.
- Fix: deconfliction measured predicted separation with `hypot`; it now uses
  `np.linalg.norm`'s arithmetic, as the reference does. The value sizes the
  lateral push.

## Unreleased — `argusnet-rs`: the Rust simulation from a command line (P4.9a)

- New crate `rust/argusnet-cli` builds the `argusnet-rs` binary. `argusnet-rs
  sim --bundle B --replay R` runs a scenario bundle in the Rust tick and writes
  its replay, with no Python at run time. `--checkpoint-at N --checkpoint C`
  and `--resume C` checkpoint and resume from the command line. It prints
  provenance and tick-time percentiles. Exit codes: 1 fault (no replay
  written), 2 invalid input, 3 unsupported.
- `argusnet sim ... --export-bundle PATH` writes the scenario those flags
  describe as a bundle, with the command's own tracker config, instead of
  running it. Options the Rust runtime cannot honour are refused by name,
  including the run-time flags `--continuous`, `--realtime`, `--streaming`,
  `--compact` and `--replay-subsample`.

## Unreleased — Every assembled stage resumes exactly (P4.8e.4)

- Stages 2 (overrides), 6 (bias walks), 7 (the tracking engine), 8 (coverage,
  localizer) and 9 (mission driver) implement the checkpoint contract, each
  validating the checkpoint against the run before anything is mutated.
- A whole assembled run, checkpointed to JSON in any phase and restored into a
  fresh process, matches the uninterrupted run bit for bit: kinematics, fused
  tracks, observation bearings, phase and coverage.
- `argusnet-core`'s engine and filter types, and the mapping, localization and
  planning state types, are serde-serialisable (additive).

## Unreleased — The default mission runs in the Rust tick (P4.8e.3)

- `MissionDriverStage` (stage 9) turns the mission machine's phases into POI
  rescoring, energy-aware assignment, redirects, handoffs, dwell and egress
  routes, in the reference's in-tick order. The scenario bundle carries a
  `mission` section, and `assemble` builds stages 2, 8 and 9 for
  `scan_map_inspect`.
- With the reference's localization replayed, a 400-tick default mission
  matches `run_simulation` bit for bit on every node and tick, through
  `complete`. Free-running, it is exact until localization diverges and then
  completes the mission on its own.
- Distances in the POI manager and redirects use `np.linalg.norm`'s arithmetic,
  not `hypot`. At a searched boundary offset the reference credits a dwell that
  `hypot` would deny.
- Python: POI construction is lifted verbatim out of `run_simulation`
  (`make_poi_positions`, `build_run_inspection_pois`); a before/after 400-tick
  run is byte-identical.

## Unreleased — Stage 6 senses as the reference does (P4.8e.2)

- Observations originate at each node's nominal controller pose (before
  overrides and deconfliction), and directional sensors look along the nominal
  velocity tilted down 30° above 0.5 m/s of horizontal speed. Both match
  `build_observations`.
- The detection draw now includes the world detection model's `p_d` (path loss,
  transmittance, weather visibility, land cover) and cloud layers, and bearing
  noise uses the effective noise multiplier. Without them the Rust run accepted
  ~60% more observations than the reference. The totals are now within
  statistical noise of it.
- Weather's sensing half is ported (`visibility_at_range`, `bearing_noise_scale`,
  cloud layers) and carried in the scenario bundle.
- Python: `sensor_look_direction`, `fov_angle_deg` and `cloud_detection_factor`
  are lifted verbatim out of `build_observations`; recaptured references are
  unchanged.

## Unreleased — The Rust tick runs a Python-built scenario (P4.8e.1)

- `argusnet.scenario-bundle/v1`: `argusnet.simulation.scenario_bundle.export_bundle`
  describes a built target-tracking scenario as resolved data (world payloads
  exactly as the native ABI takes them, controllers, sensors, truth tables).
  Unsupported features are refused by name.
- `argusnet-sim::scenario::assemble` builds a configured `Simulation` from a
  bundle (stages 1, 3, 6 and 7 over one shared world) and runs it to a replay.
  An all-follow scenario matches `run_simulation` bit for bit on every node
  and tick, including 61 frames of deconfliction.
- Deconfliction now clamps against real terrain (`RoleTable::with_terrain`),
  not a constant floor. Motion weather (`WindModel`, flight-speed penalty) is
  ported.
- `serde_json/float_roundtrip` is enabled on the workspace dependency, so every
  build parses floats identically, including `cargo build -p argusnet-native`.

## Unreleased — The Rust tick moves the fleet (P4.8d.3)

- Stage 1 (`SampleControllersAndBattery`) now samples platform controllers and
  then accounts battery (`argusnet-sim::controller`). Ports
  `looped_waypoint_path` (search lanes) and `ObservationTriggeredFollowController`
  bit-exact against recorded Python fixtures; stage 6 feeds engagement to
  stage 1 for the next tick.
- `StageHandler` gains `reset`, `checkpoint_state`, `check_state` and
  `restore_state`. Checkpoint format 3 adds `stage_states`; every handler's
  state is validated before anything is mutated. `Simulation::reset` now resets
  handlers. `BatteryStage` checkpoints its ledger, previous positions and
  warning latch.
- `argusnet-sim` enables `serde_json/float_roundtrip`. Without it a JSON
  checkpoint could return an f64 one ULP off, and a resumed run diverged from
  an uninterrupted one.
- The follow controller's pursuit of true target state (the reference's
  behaviour) is now an explicit `PursuitOracle` capability, noted in the stage
  trace every tick it is read.

## Unreleased — Scheduled Rust System Documentation (2026-09-23)

- Complete the architecture audit with world-generation manifests, track identity
  and expiry, fidelity/time profiles, frame-consistent viewer resources and
  verified backend recovery requirements.
- Define the final Rust-only runtime: independently scheduled sensors/tasks,
  virtual versus live time, lifecycle ownership and bounded communication.
- Add ADR-009 and planned P4.10a–f while retaining port-parity gates and early
  P5.0 viewer delivery; reconcile seed documentation with the implemented RNG.
- Align viewer visual/freshness workflows, performance tracking, hardware
  qualification, scenario/testing goals and subsystem contracts.
- Refresh README, usage and contributor guidance; separate runnable current
  commands from planned controls. This entry records documentation only, not a
  scheduler implementation, measured speedup or hardware readiness.

## Unreleased — Runtime command recovery and control checkpoints

- Checkpoint format 2 preserves pause/speed, committed revision, pending controls,
  retry receipts, request identities and retired-ID watermark. Restore replaces
  destination command history instead of mixing timelines. Rejected nonfinite
  speed requests retain exact identity through JSON using encoded float bits.
- Validate receipt bounds/identity, pending-command consistency and control state
  before mutation. Old checkpoint formats are rejected explicitly. This completes
  control-state persistence only; domain-stage state and publication epochs for
  interactive branching remain separate work.

- Recover faulted/completed simulations through queued Reset, including tick zero;
  reset cancels old-epoch commands already due and those scheduled for later.
- SingleStep executes one tick and pauses. Scheduled controls revalidate revision
  preconditions; invalid pacing and past-tick requests are rejected.
- In-process command protocol v2 preserves retry outcomes/reasons, detects reused
  IDs with different payloads, and exposes applied/cancelled results.
- Checkpoint restore validates tick/seed/queue identity, bounds and payloads before
  mutation and restores the clock in constant time. Runtime algorithm version 2
  rejects old control semantics; complete domain-state checkpointing remains open.

## Unreleased — Point-In-Polygon Correctness For Rotated Footprints

- World: fixed the edge-crossing test in `_point_in_polygon` (`src/argusnet/world/obstacles.py`).
  The denominator was clamped with `max((b.y - a.y), 1.0e-12)`, which replaced the real signed
  height of every *downward* edge with a tiny positive number. For those edges `x_cross` blew up to
  ±1e12 and the crossing test degenerated into "toggle iff the edge moves leftward in x",
  independent of the query point. The enclosing `(a.y > p.y) != (b.y > p.y)` test already
  guarantees a non-zero denominator, so the clamp was never needed.
- Impact: any footprint with a non-axis-aligned edge was mis-classified. Axis-aligned rectangles
  were unaffected (their vertical edges have a zero `(b.x - a.x)` numerator), which is why the
  existing suite missed it. Measured on the shipped `scene.smartscene` over a 60x60 grid per
  footprint: rotated `box-v1` hangars (yaw 0.12/0.18) disagreed with a reference even-odd crossing
  on 16.0–18.5% of probes and diagonal `wall-v1` segments on 35.6–35.8%, for 7,766 of 54,000 probes
  overall; axis-aligned `polygon-prism-v1` warehouses disagreed on 0%. After the fix, 0 of 54,000.
  The yaw=0.18 hangar footprint measured 2260 m² against its true 1792 m².
- Blast radius: `_point_in_polygon` also backs `_segment_polygon_intervals`, so `path_length_inside`
  and `first_hit_t` — line-of-sight occlusion and path collision — were affected, as was land-cover
  rasterisation via `build_land_cover_layer`. The bug also defeated `push_outside_xy`, which emitted
  "still inside an obstacle after 8 push iterations" warnings on `military_compound`; those are gone.
- Tests: added `RotatedFootprintPointInPolygonTest` covering a triangle, a diamond, and `OrientedBox`
  at five non-zero yaws, each compared against an independent even-odd oracle across a ~2,000-point
  grid (boundary-adjacent probes skipped as legitimately ambiguous), plus rotated-box coverage for
  `point_inside`, `push_outside_xy`, `path_length_inside`, and `first_hit_t`.
- Goldens (ADR-005 re-baseline): `benchmark_fast_seed7_intercept_stress`
  (`total_rejected_observations` 3 → 5, `terrain.height_at.hits` 1275 → 1276) and
  `benchmark_fast_seed7_search_acquisition` (`total_accepted_observations` 15 → 16) moved because
  correct footprints change occlusion and collision. `benchmark_fast_seed7_baseline_coverage` is
  unchanged. The canonical `sim --duration-s 60 --seed 7` replay hash is also unchanged
  (`cc66136b…`) — the default scenario has no rotated footprints in its line-of-sight paths.
- Tests: re-baselined the occlusion showcase thresholds, which were propped up by the inflated
  footprints (`military_compound` shadow 28 → 6 cells, `coastal` 6 → 4).

## Unreleased — Obstacle-Aware Mapping And Redirects

- Sim: new opt-in `--occlusion-aware-mapping` gates scan coverage and the split-view reconstruction
  through `EnvironmentQuery.los()`, so cells hidden behind buildings/walls are no longer stamped as
  scanned or populated with ground-truth height.
- Sim: with the same flag, adaptive/cooperative frontier redirects, frontier exploration, POI
  inspection redirects, and egress return-home trajectories use `PathPlanner2D` routes around hard
  blockers, falling back to the legacy straight-line hover trajectory when no route is available.
- Tests: added truth-isolation coverage for blocked reconstruction cells, obstacle-routed redirect
  geometry, collision-free sampled drone positions, and fixed-seed determinism for the opt-in mode.
- Perf: occlusion-aware coverage caps per-footprint LOS evaluations (`_COVERAGE_LOS_MAX_SAMPLES`,
  deterministic striding via `los_max_samples` on `CoverageMap.mark_circular` /
  `WorldMap.add_scan_observation`). Typical resolutions test every cell exactly; the cap only bites at
  very fine `--coverage-resolution-m`, bounding worst-case cost. Marked cells are still genuinely
  visible (invariant preserved).
- Tests: showcase test confirming occlusion scales with obstacle height/density — `military_compound`
  (tall buildings + walls) shadows more ground than open `coastal` terrain; near-nadir mapping
  over flat/undulating terrain is barely occluded (expected). (The previously recorded ~4–5× ratio
  was an artifact of the `_point_in_polygon` footprint bug fixed below; the true margin is smaller.)

## Unreleased — Adaptive Cooperative Search And Headless View Modes

- Sim: `--adaptive-search` refines `--cooperative-search` — drones redirect via wedge-aware,
  coverage-balanced frontier selection (`select_cooperative_wedge_cell`), reallocating from finished
  wedges to the most under-covered one, and the wedge origin re-anchors on the fused team estimate
  (a localization→planning feedback loop). Measured vs plain cooperative on a medium map: coverage
  92% vs 81%, more even per-wedge balance, tighter team fix.
- Viewer: `live-stream` is now a default cargo feature, so a plain `cargo build --workspace` keeps
  the viewer's `--live` flag (build `--no-default-features` for a lean replay-only viewer).
- Viewer (headless): `render_headless` honors `--view-mode` — scan-map/split stills and PNG
  sequences now draw the accumulated reconstruction (height-colored) and the team co-localization
  beacon with spokes, and a single `--output` still renders the final frame. Makes the reconstruction
  and cooperative results CI-renderable (covered by a new headless test).

## Unreleased — Team Co-Localization In The Viewer

- Replay: cooperative runs now record the fused team position per frame as the additive
  `scan_mission_state.team_localization` (position, 1-sigma std, confidence, contributing drone IDs);
  documented in `docs/replay-schema.json` and modelled by `argusnet.core.types.TeamLocalization`.
- Viewer: renders the team estimate as a cyan beacon (stalk + sphere + true uncertainty ring) with
  spokes to each drone whose estimate was fused in, so cooperative co-localization is visible.
  Fixed the per-drone localization ellipses, which had used a stale `(x, z, -y)` mapping and drew in
  the wrong place — now Z-up `(x, y, z)` like every other overlay.

## Unreleased — Cooperative Radial Search

- Sim: new opt-in `--cooperative-search` (scan_map_inspect mode) launches every drone from one
  grounded origin and fans them out into disjoint angular wedges of the search area — a smarter,
  datum-anchored coverage pattern than independent lawnmower sectors. `--search-origin X,Y` sets the
  common origin (defaults to the search-area center). `cooperative_wedge_waypoints()` builds each
  drone's expanding polar boustrophedon.
- Sim: in cooperative mode the drones' independent per-drone localizations (all referenced to the
  shared coverage map they build together) are inverse-variance fused into a team estimate via the
  now-wired `GridLocalizer.fuse_estimates()`. The grounded origin, and the team co-localization
  convergence, are reported in the CLI summary and recorded additively in `meta.scenario_options`
  (`cooperative_search`, `search_origin_m`) and the run summary.

## Unreleased — Independent Split-View Camera Controls

- Viewer: mouse input is now routed to the region under the cursor. The reconstruction pane has its
  own `OrbitCamera`, so in split view the right pane can be orbited, panned, and zoomed independently
  of the left real-terrain view. Scroll and drag over the drawer/tabs are left to egui, so the panel
  scrolls without moving the cameras — each region (left view, reconstruction, drawer) has its own
  scroll wheel.

## Unreleased — Split-View 3-D Reconstruction And Coverage Resolution

- Sim: new `--coverage-resolution-m` flag sets the scan/reconstruction grid cell size (default 50 m,
  previously hard-coded), so mapping runs can produce a higher-resolution coverage map and denser
  reconstruction. Recorded additively as `meta.scenario_options.coverage_resolution_m`.
- Viewer: Split mode is now a genuine side-by-side — the right half is an empty area that fills with
  a 3-D reconstruction of the believed terrain, rendered by an angled perspective camera on a
  dedicated render layer (real terrain, drones, and gizmos no longer leak into it) over its own dark
  clear color. Reconstruction tiles auto-size to the coverage cell spacing so the surface reads as
  near-continuous at any resolution.
- Docs: `docs/usage.md` mapping recipe updated for a large, high-resolution path-planning run and the
  3-D split view.

## Unreleased — Viewer Split-View And Startup Options

- Viewer: `argusnet-viewer` gained `--view-mode {real-world,scan-map,split}` to choose the initial
  render (split shows real terrain on the left and the accumulated scan-map reconstruction on the
  right; the `M` key still cycles the modes at runtime) and `--autoplay` to start the replay
  timeline immediately so the reconstruction fills in without pressing Space. `ViewMode` is now
  re-exported from the crate root.
- Docs: `docs/usage.md` adds a mapping-run recipe that packages a scan/map/inspect mission and opens
  it directly in split view.

## Unreleased — Front-Door Reporting

- CLI: `argusnet sim` no longer prints the misleading "No fused tracks were produced." for
  mission modes where tracks are not the point. A `scan_map_inspect` run now reports final phase,
  map coverage, and POIs inspected; a `target_tracking` run that produced no tracks reports the
  observation acceptance rate and dominant rejection reason, with an actionable hint (e.g. use a
  smaller `--map-preset`) when targets are `out_of_range`. Report-only change — replay output and
  the fixed-seed determinism hash are unchanged.
- Docs: `docs/usage.md` documents that the default sim is a mapping mission and adds a curated
  target-tracking demo command that reliably produces fused tracks.
- Live: `argusnet live` gained a `--map-preset` flag (default `medium`) and passes it to the
  continuous sim, so the streamed target-tracking run keeps drones within sensor range of their
  targets and the viewer shows fused tracks instead of a nearly empty scene.
- Fixed: the bundled `scene.smartscene` manifest referenced `overlays/walls.glb`, which is not in
  the package; the dangling overlay layer was removed so the viewer no longer logs a missing-asset
  error on every load (`validate-scene` still reports the separately-missing replay payload).

## Unreleased — Usability, Realism, And Robustness Pass

- CLI: live progress bars for `sim` and `benchmark` runs (TTY-only, `--quiet`-aware); export
  formats listed in `--help`; `--dry-run` now fails fast on bad config files and warns about
  unrecognized config keys (`unknown_config_keys` in `core/config.py`).
- Performance: the simulation loop now ingests frames over the bidirectional `TrackStream` RPC
  (`TrackingService.open_ingest_stream()`) instead of per-frame unary calls; coverage-map marking
  is vectorized; per-pair RNG construction and sensor look-direction evaluation are hoisted out of
  the observation hot loop. Fixed-seed replay output is bit-identical to the previous behavior.
- Realism (opt-in flags, default off): `--enforce-flight-envelope` clamps drone motion to
  `FlightEnvelope` acceleration, bank-derived turn-radius, and climb/descent limits
  (`EnvelopeLimitedTrajectory`); `--frontier-exploration` drives scanning drones with
  `FrontierPlanner.select_frontier_cell()` instead of fixed lawnmower sectors.
- Realism (additive): `wind_drift_scale` exposed in `DynamicsConfig`/`dump-config`; wind- and
  climb-aware return-home battery reserve (`BatteryModel.dynamic_reserve_wh`, static reserve as
  floor); optional GNSS Gauss-Markov colored noise and deterministic outage windows
  (`GNSSSimulator`, `OutageSchedule`).
- Robustness: Rust tracker reconditions degenerate innovation covariances (symmetrize + jitter
  retry) instead of erroring, bounds covariance growth during pathological coasts, and gains a
  multi-track long-duration stress test; collision push loops log a warning when they exhaust
  their iteration budget.
- Fixed: `MissionExecutor` inspect-task completion crashed any run reaching the inspect phase
  (`all_complete` property was called as a method).

## Unreleased — Benchmarking And Performance Harness

- Added the `argusnet benchmark` CLI, canonical fast/slow scenario suites, performance summaries,
  and fast-suite golden regression checks.
- Added Python hot-path benchmark tests, Rust Criterion benchmark compile coverage, and nightly
  benchmark workflow artifact upload.
- Added terrain, gradient, LOS, and obstacle cache metrics plus per-frame simulation timing fields.

## 2026-04-29 — Documentation Reframing

- Reframed documentation around ArgusNet as a map, localize, inspect, revisit, and coordination
  system.
- Updated architecture and state ownership docs to use `src/argusnet` and `argusnet-*` crate names.
- Clarified current `scan_map_inspect` behavior: mapping coverage, localization confidence,
  inspection POI completion, deconfliction, egress, and replay state.
- Replaced outdated scenario and planning vocabulary with mapping, localization, inspection,
  revisit, POI assignment, and multi-drone coordination terminology.

## 2026-04-27 — Scan, Localize, Inspect, Egress

- Added scan → localize → inspect → egress mission state.
- Added scan coverage thresholds, POI localization ellipses, and inspection event logging.
- Added egress/RTH with per-drone home progress and arrival detection.
- Hardened multi-drone deconfliction.
- Improved viewer panels, selection details, event display, and scan overlay color.

## March 2026 — Viewer, Scene, Terrain, And Ingestion Foundations

- Consolidated documentation into `README.md` and `docs/`.
- Added native Bevy viewer and `smartscene-v1` package loading.
- Split scene compilation into terrain, geometry, GIS, and styling helpers.
- Added terrain presets, terrain-aware mission zones, rejection diagnostics, and export formats.
- Added MQTT and file replay ingestion adapters.
- Added Rust fusion service, replay export, and Python simulation workflow.

## Rust world authority (Phase 3)

- Grew `terrain-engine` into `argusnet-world`, including tiled/analytic terrain,
  obstacle indexing, land cover, DDA visibility and detection models.
- Wired Python world queries through packaged native bindings and removed the
  Python safety checker. Simulation safety now uses `safety-engine` defaults
  and policy, preserving opt-in blocking and clamp ordering.
- Moved replay terrain sampling from the viewer into the shared world crate.
- Added cross-language analytic fixtures and regression coverage for terrain
  seams, LODs, negative ray directions, missing coverage and combined violations.
