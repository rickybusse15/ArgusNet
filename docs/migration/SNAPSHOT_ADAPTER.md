# P2.0: the snapshot adapter and persistent presentation state

**Status:** local, 2026-09-23. See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.0: *"Replace per-frame `CurrentRuntimeMarkers`
reconstruction/string cloning with a single snapshot-to-ECS adapter. Keep typed
platform/track/truth IDs ... Stable IDs survive pause, seek, filtering and live
reconnect only within a valid run/epoch. Spawn/despawn on membership changes;
update transforms/quality only on changed data. Clear selection on removal and
invalidate incompatible interpolation when epoch/world changes."*

## What the viewer did per rendered frame before

With 240 presented entities:

* `interpolated_markers` rebuilt the whole marker list from the frame, cloning
  every label. It then searched the next frame once per marker: quadratic.
* `trail_points` rescanned 32 frames for every marker, with a linear search
  per frame, in the overlays and again in the minimap.
* `marker_appearance` searched the frame's tracks for every track marker, in
  both the mesh sync and the overlays: quadratic again.
* The selection was re-resolved **by label**. A track is keyed by its
  target's id, so it shares a label with that target's truth. Selecting a
  truth resolved to the track, whose entry comes first.

Measured on the allocation harness below: **1,108 allocations per frame
playing, 1,210 paused.**

## The adapter

`crate::presentation`. `CurrentRuntimeMarkers` is now persistent state,
updated in place by `apply(&ReplayState)`:

* **Identity.** `EntityKey::{Platform, Track, Truth}(IdIndex)`: the kind plus
  an id interned in the epoch's `IdTable`. Labels are `Arc<str>` shared with
  the table, so presenting a frame clones no strings.
* **Membership** (entities, recorded poses, track quality, trails, and the
  next-frame match through a key index) is rebuilt only when the *source
  frame* changes. The source is identified by frame index, frame count and
  timestamp. Trimming live history shifts the index under an unchanged frame;
  that still counts as a new source, never as a stale index reused.
* **Poses** are re-blended in place when the interpolation alpha changes, and
  never otherwise. A paused frame does nothing.
* Each `apply` reports what changed: epoch, membership, frame, poses. The
  membership and pose revision counters let downstream systems skip work.
* Track quality (`measurement_std_m`) rides on the marker, so appearance
  needs no second search.

### Epochs

`ReplayState::epoch`. A new epoch begins only when the document is replaced
by a different run. A reload uses `succeed`, and a live frame from a different
daemon session (`WatchFramesV2Response.session_id`) uses `begin_epoch`, which
drops the old session's history. Pausing, seeking, filtering, a reconnect to
the same session, and trimming live history all keep the epoch. A new epoch:

* starts a new id table, so a stale key cannot alias the new table's entity;
* despawns every marker entity, so none outlives its run, even where the new
  run's ids intern to the same numbers;
* voids the selection, which now carries its key and epoch;
* drops interpolation state keyed to the old frames.

### Marker entities

`marker_meshes::sync_marker_meshes`, with `MarkerEntityIndex` mapping each key
to its entity:

* Entities are spawned and despawned only on a membership change, found
  through the index rather than a scan.
* Presentation (transform, visibility, mesh and material) is recomputed only
  when the pose revision, the view (`GlyphProjection`, now `PartialEq`), the
  overlay settings or the scene package changed.
* It is written with `set_if_neq`, so an unchanged marker does not register
  as changed for anything downstream.

### Ordered system sets

`app::ViewerSet`, chained in `Update`: **Input** (playback, operator input,
live ingest) → **SnapshotApply** (take in new runs, then the adapter) →
**Presentation** (entities, visibility, fog, reconstruction) → **Picking** →
**Overlays**. The operator UI runs in egui's own pass after `Update`, so it
reads a finished frame. Reconstruction keeps running when its overlay is off,
as before: it owns real state.

## Verification

| Check | Result |
| --- | --- |
| `tests/presentation_budget.rs`: allocations per rendered frame, net of Bevy's scheduler (an identical app with no-op systems), 240 entities at 60 fps over quarter-second frames | **playing: 1,108 → about 5; paused: 1,210 → about 2**. The paused 2 is Bevy fetching the marker system's parameters: returning at the very top of the system body measures the same 2.03. Asserted: < 8 playing, < 3 paused |
| Long seek/reload loop: 2,000 random seeks, a reload every 97 | exactly one entity per presented marker every round; at most 7 materials and 4 meshes |
| Adapter unit tests | interpolation (exact seek, midpoint, no overshoot, holds when absent); a track and truth sharing a label get distinct keys and blend to their own poses; keys stable across seeks; a new epoch voids keys; trails equal the recorded window; revisions; in-place blending; trimmed live history |
| Selection | a selected truth stays the truth (the label bug); a new epoch clears it |
| Live | only a new session begins an epoch and replaces history; trimming keeps the epoch and the frame |
| Headless render, as CI runs it (650×480, isometric) | 10,698 distinct colours, 257 green track pixels (CI requires > 50 and > 30) |
| Mutation round | 17 mutants. 14 caught at first; the 3 survivors (first-occurrence next match, epoch flush of entities, the dirty gate) got tests, and all 17 are now caught |
| `cargo test -p argusnet-viewer`; clippy `-D warnings` with and without `live-stream`; `cargo fmt --check` | 115 passed; clean; clean |

Writing the mutants also found a defect of this slice's own: frame metrics
refreshed only on a *membership* change, so they went stale whenever the same
entities persisted from frame to frame. The adapter now reports a `frame`
change separately, and
`frame_metrics_follow_the_frame_not_the_membership` pins it.

## Not in this slice

* **`RuntimeSnapshot` as a source.** The adapter reads the replay cursor.
  Reading the in-process runtime's snapshot is P5.0's worker protocol; the
  adapter's `apply` is the seam it plugs into.
* **Raycast picking** (P2.3). Picking still takes the nearest marker within
  18 px on screen, now by key.
