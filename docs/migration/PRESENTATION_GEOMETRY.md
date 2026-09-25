# P2.2: sensor, zone and trail geometry

**Status:** local, 2026-09-24. See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.2: *"Keep the existing hologram material, but move sensor FOV
into cone/frustum meshes, mission zones into extruded boundary/volume geometry
and trails into persistent ribbons. ... Derive sensor axes from the same
sensing pose and limits as the runtime. Unknown/unset range keeps adapter
semantics. ... no effect may enclose the camera and make the world invisible.
Move remaining diagnostic gizmos behind a toggle once an equivalent readable
mesh path exists."*

Before this slice, fields of view, radar rings, trails and zone outlines were
gizmo lines, redrawn from scratch every rendered frame.

## Sensor volumes (`sensor_volumes.rs`)

**The drawn volume is the gated one.** Stage 6 accepts a target when it is
within range and, only when the sensor has both a half-angle and a look
direction, within that half-angle of the look. The look rule moved out of
`argusnet-sim` into `argusnet_sensing::look`, verbatim, and `argusnet-sim`
re-exports it. The viewer calls the same function, so the axis is the
runtime's: along the velocity, tilted down 30 degrees, and none below
0.5 m/s of horizontal speed. The drawn region is therefore:

* a spherical sector (a cone capped by the range sphere) when the sensor
  gates, at any half-angle, including past 90 degrees;
* the range sphere when it does not: no look (hovering, or zeroed by
  safety), no half-angle, or a half-angle of 179.9 degrees or more;
* nothing when the range is unset. The adapter already reads
  `max_range_m: 0` as unset. The old gizmo invented 400 m.

`VolumeSpec::contains` is the gate's own test, and the unit tests hold it to
the runtime's arithmetic.

**Topology cached by shape.** One unit mesh per shape (the sphere, or a sector
with its half-angle quantised to half a degree). The entity's transform
scales it by range and turns it onto the look axis. An entity is spawned or
despawned only when a platform's volume appears, disappears or changes
shape. Transforms are written only where they changed. The whole pass is
skipped unless the presentation epoch, the pose revision, the camera or the
toggle moved.

**Volumes stop at the ground.** A range reaches below the terrain, where
nothing can be. The low-pitch review showed the buried half of a radar's
sphere hanging visibly under the map's edge. The hologram material now has
a floor, and sensor volumes discard fragments below the scene's lowest
terrain point (`terrain_summary.min_height_m`). Markers and walls have no
floor. Between that floor and the local terrain, a volume is hidden by the
terrain itself, except in a sliver at most the terrain's relief (18 m on the
bundled site) seen from under the edge.

**Radar range rings** stay as rings, since a hemisphere over each station
would bury the site. Each is one line-list mesh draped 1.5 m above the
replay's terrain mesh and clipped to the scene. It is rebuilt only when the
radar moves or its range changes, which for a station means once.

## Zone walls (`zone_walls.rs`)

Each overlap group's merged outer contour is extruded into a wall, from the
terrain under each vertex up to 1.2% of the scene diagonal (about 70 m on
the bundled site), and 1.5 times that for exclusion zones. It is clipped to
the scene as the old outline was. One mesh per group, rebuilt only when the
overlap model's `generation` changes. A change of focus restyles the
materials and rebuilds nothing. Walls are double-sided, so they read from
inside the zone as well as out.

**A pre-existing contour bug, fixed.** The first wall render showed half a
ring. The marching-squares grid was sized `span / cell + 2` samples from a
cell before the minimum. When `span / cell` came out just under an integer,
the last sample column sat exactly on the padded edge of the circle, which
counts as inside. The contour was then open there, and the walk, which only
goes forward from its first segment, kept one half. The gizmo outline drew
the same half-contour. The grid now has `+ 3`, a full empty cell beyond the
extent on every side. `a_single_circle_contour_goes_all_the_way_round`
checks coverage and closure for three circles, including the one that
failed.

## Trails (`trail_ribbons.rs`)

**Sampling that does not depend on how you got here.** Every decision looks
at one recorded frame and the one before it, never at the samples kept so
far. So playing to a frame and seeking straight to it give the same trail. A
test compares the two across random seeks. For each entity in each frame:

* absent: its trail is removed;
* teleported (a step faster than 200 m/s, or absent from the frame before):
  the trail restarts;
* otherwise sampled on entering a new 0.5 s bucket of simulation time, or
  after moving more than 25 m.

The horizon is 60 s, with at most 240 samples a trail, and a global budget
of 20,000 samples, oldest first. A new epoch clears everything. **The one
exception to path independence** is that budget. If it bound earlier in the
window and later stopped binding (entities left), playback has already
dropped samples that a rebuild keeps.

**Ribbons.** Each trail is one mesh, two crossed strips, so it reads from
overhead and from low pitch without being rebuilt as the camera turns.
Colour follows altitude (blue low, red high) and alpha fades from old to new,
as the gizmo trails did. The width is 3 screen pixels at the trail's head,
recomputed only when the view scale moves by more than a quarter. A mesh is
rewritten when its trail changes, which happens only when the source frame
does. A trail ends at the newest recorded sample, at most one frame behind
the interpolated glyph. A new overlay toggle, **Trails**, defaults to on.

## Depth, transparency and picking

| Geometry | Material | Depth | Faces |
| --- | --- | --- | --- |
| Sensor volumes, zone walls | the existing additive hologram | tested, not written | volumes cull back faces; walls are double-sided |
| Trail ribbons | unlit, alpha-blended, vertex colour | tested, not written | both |
| Range rings | unlit, alpha-blended lines | tested, not written | n/a |

Additive geometry commutes, so ordering between volumes and walls does not
matter. Nothing here writes depth, so none of it hides what is behind it.
Everything carries `PresentationOnly`, which P2.3's picking will exclude.
Today's picker is the 18-pixel marker proximity search, which never looked at
these entities.

**No volume encloses the camera.** A volume whose region contains the camera
eye is hidden outright. For a convex volume (a sphere, or a sector up to
90 degrees) this matches what back-face culling already does: from inside,
every face points away. A sector wider than 90 degrees is concave, though,
and from inside it the far side of its cone can face the camera. The hide
covers that case. The probe renders a follow view inside a 20 km radar
sphere; at every probe point it must match the render without volumes to
within 0.01. Because that volume is convex, the probe cannot tell the hide
from culling (the mutation round showed this). A CPU test,
`a_volume_around_the_camera_is_hidden`, pins the decision itself.

**Gizmos behind the toggle.** The zone outline gizmo now draws only with
*Diagnostic glyphs* on, because the wall is its readable mesh equivalent. The
field-of-view, ring and trail gizmos are gone. Zone centre glyphs and badges
have no mesh equivalent yet and stay as they were.

**LOD and instancing: none.** The counts do not justify them: one entity per
sensing platform, per trail and per zone group, and a handful of shared unit
meshes.

## Known limits

* **Recorded pose, not nominal.** The runtime senses from the nominal
  controller pose, before deconfliction and the safety clamp move a
  platform. The replay carries only the recorded pose. So a drone the safety
  monitor has zeroed reads as having no look here and draws a sphere, while
  the runtime still gated along its nominal heading.
* **Default tilt.** The replay carries no per-sensor tilt. Every writer uses
  `DEFAULT_LOOK_DOWN_ANGLE_DEG`, and so does the viewer.
* **Draping needs the replay's terrain mesh.** Rings, walls and the zone
  outline sample `meta.terrain.viewer_mesh`, which simulated replays carry.
  Without it, a ring sits at its station's height and a wall at its zones'
  average height, as the outline did. The rendered terrain comes from the
  scene's GLB, and nothing on the CPU samples that. A ring can then end up
  under the terrain, where depth testing hides it; the probe test found this.
* **The floor is a plane.** The clip is at the scene's lowest terrain point,
  not at the terrain under each fragment (see above).

## Verification

### Pixel probes

`tests/geometry_probes.rs`: real headless renders of the committed scene with
a synthetic replay (an optical drone flying east, a radar station, one
exclusion zone). Each overlay is rendered on and off, and the 9×9 windows at
world points are compared.

| Probe | What must hold | Measured |
| --- | --- | --- |
| Halfway along the look axis | the sector is drawn, and only adds light | Δ 0.185 |
| Behind the drone, off to the side | no sector, no wall, no trail | Δ 0.000 |
| Exclusion wall, south, north and east faces | the wall is drawn all the way round, red | Δ 0.43–0.45 |
| The drone's position 20 s ago | the trail is drawn | Δ 0.108 |
| Follow view inside a 20 km radar sphere | identical to the render without volumes | Δ < 0.01 at all points |
| On a radar's ring (south, east), top-down, rings on vs off | a blue ring at the range, nothing at the station | Δ 0.04, 0.04; 0.000 |
| On the zone boundary, *Diagnostic glyphs* on vs off | the outline gizmo appears only with them | Δ up to 0.020 |

The north and east wall probes were added after the half-ring bug. Before
the fix the east face was missing. The ring probe gives its replay a flat
terrain mesh above the bundled terrain's peak. Without one, the ring sat
under the terrain (see *Known limits*).

**Visual review.** Top-down, isometric, close follow, and a 0.1 rad low
pitch (a temporary change to the isometric preset, reverted). The low pitch
found the buried volume showing under the map's edge, which is now fixed by
the floor. Nothing else stood out: at close follow the sector reads as a
translucent cone from its apex, and the trail as a thin line behind the
drone.

### Mutation round

33 mutants across volumes, rings, trails, walls and the contour grid. **20
were caught at first.** The 13 survivors were all test gaps, now closed:

| Survivor | Gap | Now caught by |
| --- | --- | --- |
| hiding a volume around the camera removed | the probe's volume is convex, where culling already hides it | `a_volume_around_the_camera_is_hidden` (a 150-degree sector, concave) |
| ring at half radius; not draped; not clipped | no ring tests | `a_range_ring_is_draped_at_the_range_and_clipped` |
| ring ignores its toggle | every probe had rings off | GPU `a_range_ring_is_drawn_at_the_range` |
| teleport by distance, not speed | the fixture's 5 km jump is both | `a_teleport_is_a_speed_not_a_distance` |
| no distance sampling; no per-trail cap | never exercised | `a_long_step_is_sampled_within_a_bucket` |
| altitude hue inverted | colour direction unchecked | the ribbon test, low blue and high red |
| width hysteresis off | the camera never moved | `widths_are_recomputed_past_a_quarter_change_in_scale` (the rule is now `needs_rescale`) |
| exclusion walls not taller | the rule sat inside the system | `exclusion_walls_are_taller_and_height_follows_the_scene` (now `wall_height`) |
| walls never rebuilt | nothing drove a new generation | `walls_rebuild_per_generation_and_restyle_on_focus`, an app test |
| outline gizmo drawn without the diagnostic toggle | walls and outline overlap | GPU `the_zone_outline_is_a_diagnostic` |

All 33 are caught now. **Not mutated:** the floor clip in the shader. Its
value is pinned on the CPU (`volumes_stop_at_the_terrains_lowest_point`), and
the discard was checked by the low-pitch review render, but no probe looks
from under the terrain's edge.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`; GPU probes (belief, geometry) | 1156 passed, 0 failed; 4 + 4 passed |
| clippy `-D warnings` (workspace, and viewer without `live-stream`); `cargo fmt --check` | clean |
| `ruff check`, `ruff format --check`; `pytest tests/` (`ARGUSNET_REQUIRE_RS=1`) | clean; 1049 passed, 9 skipped |
| CI's headless smoke, locally | exit 0, 10,561 colours, 258 green track pixels (P2.1: 10,698 and 257); the radar rings now draw on the terrain as meshes |
| CI | **not run**: nothing is pushed. The viewer job gains a *Sensor, zone and trail geometry probes* step |
