# P2.4b: camera controls and bindings

**Status:** local, 2026-09-29, on `claude/p24b-camera` (stacked on
`claude/p24a-time-controls`, itself on PR #122). See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.4, camera half: *"Retain orbit and add damped free-fly,
selected-entity follow/chase, F-to-focus and 1-9 bookmarks. Camera controls
use render delta time, not simulation dt. Scale speed to altitude/scene
size, clamp pitch/far planes for the Z-up world and apply movement within
the focused viewport. Follow loss returns to a stable camera rather than
snapping to origin. Remappable bindings cannot fire while typing. ... Save
camera bookmarks as presentation preferences, not deterministic runtime
state."* The time half is [TIME_CONTROLS.md](TIME_CONTROLS.md).

## Before

The main camera orbited (right-drag), panned (middle or Shift+right-drag)
and zoomed (scroll), and `R` snapped it home. Nothing else moved it. Keys
were hard-coded across three systems, each asking egui separately whether a
text field had the keyboard. `M` (cycle view) sat after the "no replay
loaded" return, so it did nothing in a scene without one.

## Behaviour (`camera_control.rs`)

The whole behaviour is one function, `CameraControl::step`. It takes the
frame's actions, pointer look and scroll, the render delta, and what the
world says: the selection, where tracked entities are, the ground, and the
saved bookmarks. It updates the orbit and returns the camera's transform.
The system around it only gathers those inputs, so every behaviour below is
tested without a window.

| Key | Does |
| --- | --- |
| right-drag, middle-drag, scroll | orbit, pan, zoom, as before |
| `F` | move the orbit onto the selection, closing in to at most 600 m |
| `L` | follow the selection: the focus stays on it; orbit and zoom still work around it (pan does not) |
| `C` | chase: follow, and swing behind its direction of travel at a low pitch (yaw is the chase's) |
| `G` | free-fly: `W A S D` along the view, `E`/`Q` up/down, right-drag to look, scroll for speed, `Shift` for four times |
| `1`-`9` | recall a bookmark; with `Ctrl` or `Cmd`, save one |
| `R` | home |
| `Esc` | leave free-fly, follow or chase; only an `Esc` the camera did not use clears the selection (VIEWER_UI §12's order) |

* **Render time.** Everything runs on the render delta (`Time<Real>`), so
  pausing or speeding up a replay neither freezes nor races the camera.
* **Damped.** Focus, follow, chase, bookmarks and home move the orbit toward
  a goal by `1 - exp(-dt / 0.18 s)` a frame, which is the same path at any
  frame rate. Direct mouse orbiting stays immediate. Free-fly accelerates
  the same way (0.15 s).
* **Speed** follows height above the ground: from 5 m/s up to a quarter of
  the scene's span per second, times the scroll multiplier (0.1 to 10).
* **Clamps, Z-up.** Free-fly keeps the eye 2 m over the ground, within one
  scene span of the bounds, and under the orbit's ceiling. Its pitch stops
  5.2 degrees short of straight up and down. The orbit keeps its own pitch
  range. The far plane is sized from the orbit's ceiling and the scene, and
  a test holds that it reaches the far corner from the worst eye free-fly
  allows.
* **Leaving free-fly** makes an orbit around the ground point the view was
  on. The eye does not move, unless it is closer to the ground than the
  orbit's minimum radius (40 m). Then the focus stays on that point and the
  eye backs off along the view: the first version clamped the distance
  first and put the focus underground, and a test caught it.
* **Follow loss.** When the followed entity is missing from the frame, or
  the run changes (a new epoch), the mode ends. The camera settles on the
  entity's last known position, never the origin, and the status bar says
  *"Lost drone-1: holding its last known position"*. Chase keeps its last
  heading while the entity is slower than 0.5 m/s, so a hover does not spin
  it.
* **Focused viewport.** Keys and look move the main camera only while the
  pointer is over it or off the window, and never while the UI has the
  pointer. In split view the reconstruction pane keeps its own orbit input,
  as before.
* **Status.** The top bar shows the camera's mode (*Orbit*, *Free-fly ×1.0*,
  *Following drone-1*, *Chasing drone-1*) and its last notice for four
  seconds: nothing to focus, an empty bookmark slot, a lost entity.

## Bindings (`bindings.rs`)

Every keyboard action, playback, selection and camera, is an `Action` bound
to one key in one `Bindings` resource. The defaults are the table above
plus VIEWER_UI's. A preferences file may remap by name
(`"bindings": {"focus": "KeyZ"}`). An unknown action or key, or a key
another action already has, is refused, reported in the status bar and the
log, and the default stays. Every default key is distinct (tested).

**Nothing fires while typing.** `KeyboardGate` asks egui once a frame, first
thing, whether a text field has the keyboard. Every binding is read through
it, so the three separate egui checks are now one.

## Presentation preferences (`prefs.rs`)

Bookmarks and remaps live in one JSON file: `$ARGUSNET_VIEWER_PREFS`, else
`~/.config/argusnet/viewer.json`. They are the operator's, not the run's.
They never enter a replay, a bundle or a checkpoint, and nothing
deterministic reads them.

* **Bookmarks are per scene id**, so one site's bookmark never flies the
  camera somewhere meaningless on another.
* **Headless reads and writes none of it**, so a render does not depend on
  who ran it.
* **A malformed file** is reported and left untouched; that session's
  bookmarks stay in memory.
* **Writes** go to a temporary file and are renamed into place.

## Allocation

`tests/presentation_budget.rs` now also measures the camera system,
against the same app without it, with the allocator serialised between the
two tests (they ran in parallel at first and counted each other's
allocations, which read as 38 per frame):

| | allocations per frame, net |
| --- | --- |
| idle | 0.00 |
| following a drone | 0.00 |
| free-flying with a key held | 0.00 |

Two things were changed to get there: the list of actions is a constant
array, and the held and pressed actions are a bitset. As a `BTreeSet`,
holding a movement key cost two allocations a frame. The followed entity's
label is shared with the marker table, not copied. P2.0's path is
unchanged: 5.03 playing, 2.03 paused.

## Verification

### Tests

`camera_control.rs`: damping frame-rate independence and the short way round;
`orbit_from_eye` inverting `eye_position`; speed and clamps; focus settling
(and nothing selected); follow, a loss holding the last known position, a
new epoch as a loss; chase behind the heading and holding it on a hover;
Escape's order; free-fly entering without a jump, the damped first step,
flying forward, staying over the ground, keys doing nothing elsewhere, the
pitch limits, and leaving without a jump (and near the ground, without an
underground focus); scroll speed; bookmarks saved, recalled (leaving
free-fly) and empty; home; notices expiring; the far plane.
`bindings.rs`: distinct defaults, name round trips, refused remaps, typing
blocking every binding, the bitset. `prefs.rs`: per-scene persistence, a
malformed file, no path. `picking.rs`: a camera Escape keeps the selection.
`app.rs`: `M` with no replay.

### Mutation round

23 mutants, against a checked green baseline. **18 were caught at first.**
The five survivors were test gaps, now closed:

| Survivor | Gap | Fixed by |
| --- | --- | --- |
| free-fly without damping | the bound sat exactly at an undamped first step | the first step must be the damped fraction, about 10% |
| keys move the camera from another viewport | leftover momentum hid it | from rest, holding a key elsewhere moves nothing, exactly |
| Escape clears the selection even when the camera used it | not tested in the selection system | `keyboard_and_list_select_without_the_pointer` |
| `M` needs a replay again | no test | `cycle_view_works_without_a_replay` |
| free-fly pitch unclamped | no test | the pitch limits, both ways |

All 23 are caught now.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`; GPU probes (belief, geometry, picking) | 1211 passed, 0 failed; 4 + 4 + 4 passed |
| clippy `-D warnings` (workspace, and viewer without `live-stream`); `cargo fmt --check` | clean |
| mypy (as CI runs it), ruff | clean |
| CI's headless smoke, locally | exit 0, 10,561 colours, 258 green track pixels: identical, so the camera's new placement path renders the presets exactly as before |
| The live window | **not driven.** Keys, look and the status bar have not been exercised by hand; the behaviour is tested through `step`, and the system through the allocation test's app |
| CI | not run: this branch is not pushed |

## Not in this slice

* **A bindings editor.** Remapping is by preferences file only.
* **Pick-to-focus.** Double-click to focus is not bound; `F` after a click
  does the same.
* **Section projection** (VIEWER_UI §6.1) is P2.5's shell work.
