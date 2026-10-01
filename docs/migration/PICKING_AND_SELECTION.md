# P2.3: picking, hover and focus identity

**Status:** local, 2026-09-25. See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.3: *"Replace the 18-pixel proximity picker with mesh
raycasting filtered to visible, selectable entities in the active viewport.
Terrain/buildings must occlude picks; transparent sensor volumes, trails, sky
and zones must not steal a platform pick ... Decide pick proxy geometry for
thin platform arms; it must be bounded by the visible glyph footprint. Add
hover feedback and a selection outline/material pass independent of the
normal hologram material, with no through-terrain outline by default.
UI-captured pointer input does not pick the world. Tests cover depth
ordering, hidden truth, split viewport coordinates, camera movement, despawn
and same-label/different-kind IDs. Inspect both pointer and keyboard
selection workflows."*

## What the old picker did

It projected every visible marker to the screen and took the nearest within
18 pixels. So a marker behind a hill was picked through it, and the nearer of
two overlapping markers was not necessarily the one picked. In split view it
compared window coordinates against projections that could land in the
reconstruction viewport, so a click there could pick a marker the main
camera had off-screen. Nothing in the world showed what was selected. And on
a paused frame a click selected with empty details, until the frame moved.

## Picking (`picking.rs`)

**A ray, not a radius.** The pointer becomes a ray through the main camera,
but only when it is inside that camera's viewport (a half-open rectangle, so
two split viewports never both claim a pixel). The ray is cast with Bevy's
`MeshRayCast`, and the nearest hit wins, among exactly two sets:

* **markers** whose kind is shown and whose entity is visible. The kind is
  checked as well as the entity's visibility, because a toggle flipped this
  frame reaches `InheritedVisibility` only after propagation;
* **occluders**: every mesh under a base layer (terrain, buildings), tagged
  `PickOccluder` as the glTF spawns.

If an occluder is nearest, nothing is picked. Everything else is invisible to
the ray: sensor volumes, trails, zone walls, rings and the sky. All of those
are `PresentationOnly`, and no interaction mode selects them yet.

**Occluders block from both faces.** Every triangle of the committed
terrain is wound facing down while its normals face up. It draws correctly
only because its material is double-sided. With the ray cast's default
back-face culling, a ray from above passed straight through the ground: the
first click probe picked a node buried 250 m under it. Occluders now carry
`RayCastBackfaces`, which matches how they are drawn. The winding itself is
a scene-writer defect, recorded in `KNOWN_GAPS.md`.

**Pick proxies.** A drone glyph is a body and two thin arms. It is picked by
its bounding box (a `SimplifiedMesh`): the whole symbol, gaps between the
arms included, and nothing past the arm tips. A test casts through the gap
(the drawn glyph misses it, the proxy hits) and just past the tips (both
miss). Stations, tracks and truths are compact and are picked as drawn.
Glyphs are sized in screen pixels, so the proxy is too, whatever the zoom.

**Hover** is recomputed only when one of its inputs moves: the pointer, the
camera's transform or viewport, the markers' epoch, membership or poses, the
kind toggles, or the occluders (new meshes tagged, a layer shown or hidden).
A still pointer over a paused frame casts nothing.

**The UI keeps the pointer.** The pointer reaches the world through a
`PointerInput` resource. In the interactive viewer it is filled from the
window, and marked as the UI's when egui has the pointer. A pointer the UI
owns hovers nothing, and a click there is consumed without touching the
selection. So is a click outside the main viewport.

**Selection** is by typed key (`EntityKey`), as P2.0 made it. A click
selects what is hovered, or clears on empty world. A new selection is
resolved against the current frame at once, paused or not.

**Keyboard and list.** Every pointer path has a non-pointer one:

* Tab and Shift+Tab step through the shown markers in presentation order,
  wrapping; Escape clears (not while egui has the keyboard);
* the inspector has an **Entities** list, grouped as Platforms, Tracks and
  Truth (evaluation). Rows are keyed, so a track and a truth that share an
  id are two rows under two headings. The panel reads the selection, so a
  row click is a `SelectRequest` that the picking stage applies.

## Hover and selection rings (`selection_outline.rs`)

The selected marker and the hovered one each get a ring: a camera-facing
annulus just outside the glyph, 3 px clear of it. The selection ring is
amber and 2.5 px wide; the hover ring is blue and 1.5 px. The hover ring is
not drawn on the selected marker.

**Why a ring, not a silhouette outline.** An inverted-hull outline needs
something opaque in front of the hull to hide its interior. The glyphs are
additive holograms that write no depth, so the hull would show through them
as a filled disc. A screen-space outline would need a separate mask pass.
The ring is its own mesh and material (unlit, opaque): it changes no marker
material, which markers share by colour. It is bounded by the glyph, as
VIEWER_UI §6.4 asks.

**No through-terrain outline.** The ring is depth-tested like any mesh, so
terrain hides it where terrain hides the glyph. The probe that proves it
compares, pixel for pixel, a render with a buried node selected against one
with nothing selected. A ring on a hidden marker (for example a truth with
truths off) is not drawn; the selection stays, and the snapshot adapter
clears it when the entity leaves or the epoch changes. Rings are
`PresentationOnly` and cast no shadow.

## A P2.2 defect this slice found

The ring probes are pixel-exact, and at first a ring hidden under the terrain
still changed 24 pixels around the drone. The cause was trails. They were
alpha-blended while every other presentation layer was additive, and at a
trail's head the ribbon and the glyph sort by almost the same distance.
Blend-then-add and add-then-blend differ, so an unrelated mesh entering the
world could flip them. Trails are now additive, so every transparent layer
commutes. The P2.2 doc is corrected, and `trail_blended_again` is in the
mutation round below.

## Verification

### CPU tests (`picking.rs`, `selection_outline.rs`, `marker_meshes.rs`)

Ray casts use the real `MeshRayCast` against real meshes and AABBs, and the
pointer-to-selection tests run `pick_world_system` with a real camera:

| Test | Holds |
| --- | --- |
| `the_nearest_marker_on_the_ray_is_picked` | depth order, from above and from below |
| `an_occluder_in_front_blocks_the_pick` | an occluder in front blocks; one behind does not |
| `base_layer_meshes_occlude_from_either_face` | a base-layer mesh is tagged, and a ground plane wound facing down still blocks; a loose mesh is not tagged |
| `hidden_markers_are_passed_through` | a kind toggle, or a hidden entity, lets the ray through to the next marker; truths are hidden by default |
| `presentation_geometry_never_steals_a_pick` | a volume in front of the drone is ignored |
| `a_despawned_marker_is_gone_from_the_ray` | despawn |
| `same_label_different_kind_are_different_selections` | a track and a truth sharing `x` select as themselves, with their own detail kinds |
| `hover_and_click_follow_the_pointer_and_the_camera` | hover follows the pointer; nothing is cast while nothing moves; moving the camera changes the hover; a click selects or clears |
| `the_ui_keeps_the_pointer_from_the_world` | UI capture |
| `a_split_viewport_picks_only_inside_itself` | split viewport coordinates |
| `keyboard_and_list_select_without_the_pointer` | Tab, Shift+Tab, Escape, list requests (and that they are consumed) |
| `tab_cycles_the_shown_markers_in_order_and_wraps`, `split_viewports_do_not_overlap` | the cycle and the half-open viewport rule |
| `a_drone_is_picked_across_its_whole_footprint`, `markers_carry_their_pick_proxies` | the proxy's extent, and its wiring |
| `a_selection_on_a_paused_frame_gets_its_details_at_once` | the paused-details fix |
| ring tests | radii, facing the camera at pixel scale |

### GPU probes (`tests/picking_probes.rs`)

Real headless renders of the committed scene. A probe can now select by kind
and id, click, and hover. A click names a world point: the live camera
projects it to a pixel, and that pixel goes through the real picker (pointer
to ray to cast), against the committed terrain's own meshes. Each click's
outcome is written to the probe JSON.

| Probe | Result |
| --- | --- |
| Clicks with volumes, trail, zone walls and a radar sphere around the drone all on: drone, open ground, station, a node 250 m under the terrain, drone | drone-1, cleared, station-1, **nothing** (the terrain occludes), drone-1 |
| Clicks on a truth and a track that share `t-1`; then the truth with truths hidden | truth `t-1`, track `t-1`; then nothing |
| Split view: the drone in the left viewport, then the same click shifted into the right one | drone-1; then ignored, with the selection unchanged |
| Rings: selected drone, hovered drone, selected buried node, selected hidden truth, selected and hovered drone | 268 amber pixels, all within 20 px of the drone; 179 blue; **0 changed**; 0 changed; identical to selected-only |

### Mutation round

26 mutants over the picker, rings, proxies, the paused-details fix and the
trail blend mode. **23 were caught at first.** Every mutant got a rebuild,
the CPU tests, and the GPU probes when the CPU tests passed. The three
survivors were test gaps, now closed:

| Survivor | Gap | Now caught by |
| --- | --- | --- |
| ring gap set to 0 | the test recomputed the radius from the same constant | `the_ring_sits_just_outside_the_glyph` bounds the clearance, 2 to 4 px |
| proxy not updated when a glyph changes | no marker changed glyph | `a_marker_that_changes_glyph_changes_proxy` (a node that lands) |
| trails blended again | only an incidental order flip showed it | `ribbons_are_additive_like_every_other_layer` |

The other 23 include: no viewport check, a closed viewport edge, no
occluders, occluders without both faces, tagging every mesh, no kind check,
`RayCastVisibility::Any`, hover ignoring the camera, hover casting every
frame, clicks outside the viewport or on the UI selecting, an unconsumed
press, Shift+Tab stepping forward, an empty pick not clearing, an
unconsumed list request, Escape ignored, a ring on a hidden marker, a hover
ring on the selected marker, a ring not facing the camera, swapped ring
colours, a stale-epoch selection ringed, drones picked by their drawn mesh,
and the paused-details fix removed. All 26 are caught now.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`; GPU probes (belief, geometry, picking) | 1181 passed, 0 failed; 4 + 4 + 4 passed |
| clippy `-D warnings` (workspace, and viewer without `live-stream`); `cargo fmt --check` | clean |
| CI's headless smoke, locally | exit 0, 10,561 colours, 258 green track pixels: identical to P2.2, since nothing is selected or hovered there |
| Python | not rerun: this slice changes no Python and no crate the native library builds from |
| The interactive window | **not driven.** Pointer to ray to pick runs for real in the headless probes, and UI capture, keyboard and list selection run in CPU app tests; egui's own hit-testing in a live window has not been exercised by hand |
| CI | **not run** before this PR. The viewer job gains a *Picking and selection probes* step |

## Found after the PR opened: a trail pixel that is sometimes missing

Repeating the ring probe showed it failing about one run in three, always
on the same 13 pixels at the drone's centre, never on a ring. The drone's
trail in that two-frame scene is a ribbon about a pixel long, ending under
the glyph. In roughly one render in twelve that has probe interactions, it
is not drawn at all. With trails off the pixel never varies (12 of 12);
with trails on and no interactions it never varied either (6 of 6).

What was ruled out: draw order (trails are additive now), and the ribbon's
data (its samples, colour and width inputs are the same in good and bad
runs). A per-trail width re-cut was tried and did not help, so it was not
kept. The cause is still **open**. The ring probes now render without
trails, since rings do not involve them, and pass repeatedly (4 of 4 runs
of all six renders).

## Not in this slice

* **Selecting presentation geometry.** No interaction mode selects a zone,
  a volume or a trail yet. When one does, it must opt those entities into
  the ray explicitly for that mode.
* **An x-ray outline.** Only the default (no through-terrain outline) is
  built. A mode that shows occluded selections would be a second ring
  material with the depth test off.
* **Focus.** Framing the camera on the selection is P2.4's.
