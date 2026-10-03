# P2.4a: time controls over source time

**Status:** local, 2026-09-28, on `claude/p24a-time-controls` (stacked on PR
#122). See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.4 is split in two. This half is the time half: *"Use
time-based scrubbing over irregular replay timestamps, with interpolation,
loop and jump-to-event. Pause/step acts on the runtime command channel in
Phase 5; replay mode implements the same UI affordances locally. Display
unavailable seek ranges and live/read-only history separately from
simulation rewind."* P2.4b, the camera half (free-fly, follow and chase,
focus, bookmarks, remappable bindings), is still open.

## What playback did before

It counted frames. A cursor (`frame_index`) advanced one frame per fixed
step: `meta.dt_s`, or the first interval when a replay had none. The blend
toward the next frame was the time accumulated divided by that same step.
So a replay whose frames were not evenly spaced played at the wrong speed,
and blended with the wrong fraction. A live stream, whose frames arrive when
they arrive, was the common case. The timeline slider was also a frame
number, and its plot's x-axis was the frame index, so an irregular recording
was drawn distorted. Nothing was said about trimmed live history or gaps in
a recording.

## The cursor is a source time (`replay.rs`)

`ReplayState` now holds a cursor in source seconds. `frame_index` is derived
from it: the last frame recorded at or before the cursor, found by binary
search. Playing advances the cursor by render time times the speed.

* **The blend is the true fraction** of the current step, between the two
  timestamps around the cursor.
* **The nominal step** is `meta.dt_s`, or else the median of up to 31
  intervals spread over the recording. The median is taken over a
  fixed-size array on the stack: this runs every presented frame, and a
  first version that collected the intervals into a `Vec` broke P2.0's
  allocation budget (9 allocations on a paused frame, against a limit of
  fewer than 3). The budget test caught it.
* **Recording gaps.** A step longer than three nominal steps is a gap.
  Nothing is blended across it: seeking into one shows the frame before it,
  as recorded. The timeline shades it and reports its length.
* **Teleports.** The presentation adapter no longer blends an entity whose
  step implies more than 200 m/s (the threshold trails already used). It
  holds its recorded pose instead of sliding through everything between.
  Entities missing from the next frame already held; epochs never blended.
* **Seeking** clamps to the recorded range, and a non-finite time goes to the
  start. `step_to` puts the cursor exactly on a frame, so a stepped or paused
  frame is shown as recorded. A cursor out of step with a `frame_index`
  written directly follows the index, so older writers stay correct.
* **Loop.** Playback wraps to the start at the end, keeping the overshoot.
  Live history never loops: it stops at its tail.
* **Jump to event** steps to the previous or next frame that carries events
  (rejections, deconfliction, inspection, and now safety events), wrapping,
  and pauses. The keys are `[` and `]`, and the buttons are in the replay
  drawer.
* **Live history** is marked `live` by ingestion. Trimming it from the front
  records the first retained time, and the cursor keeps its source time
  where that time survives, or moves to the first retained frame.

## The timeline (`ui/timeline.rs`, `ui/shell.rs`)

The bottom timeline is the single source-time control:

* a time slider in seconds, with *t*, the recorded range and the frame
  number;
* a **Replay** or **Live history (read-only)** label;
* speed, and a **Loop** checkbox (replay only);
* a warning line when history before a time was trimmed, and how many gaps,
  and how many seconds, are unavailable;
* a standing note: seeking a replay is local and changes no runtime, and
  reviewing live history never pauses or rewinds one;
* the observation and event plot on a source-time axis, with gaps shaded,
  phase changes, and the cursor.

Pause and step still act on the viewer's own replay. In Phase 5 they act on
the runtime's command channel; the affordances here are the same.

## Verification

### Tests

| Test | Holds |
| --- | --- |
| `irregular_timestamps_play_at_the_recorded_rate` | one second of play is one second of recording; the blend is the fraction of the current step, including a step shorter than the nominal one |
| `recording_gaps_are_reported_and_not_blended` | the median step; the gap listed; no blend inside it; blending resumes after it |
| `seeking_clamps_to_the_recording_and_step_is_exact` | clamping, NaN, exact steps, a directly written index |
| `loop_wraps_and_live_history_never_loops` | wrap with overshoot; stop without loop; live never loops |
| `trimmed_history_is_reported_and_the_cursor_keeps_its_time` | trimming |
| `jump_to_event_steps_between_event_frames_and_wraps` | order, wrap, pause, safety events count |
| `a_teleport_is_not_interpolated` | a 5 km jump holds; a walk still blends; a 50 m hop in 0.1 s is a teleport too |
| `a_recording_gap_is_not_interpolated` | the adapter shows the frame before a gap across it |
| live `trimming_history_keeps_the_epoch_and_the_frame` | live history is flagged, trimmed and keeps its time |
| `presentation_budget` | 5.03 allocations per frame playing, 2.03 paused: unchanged from P2.0 |

### Mutation round

13 mutants over the cursor, the blend, the nominal step, gaps, loop, trim,
events, the teleport guard and the live flag. The harness now refuses to run
unless the tree is green first: an earlier pass here ran against a failing
budget test, which made every mutant read as caught, and it was discarded.
Against a green baseline, **11 were caught at first**. The two survivors were
test gaps, now closed:

| Survivor | Gap | Fixed by |
| --- | --- | --- |
| blend divided by the nominal step | the test's step equalled the nominal step | a cursor inside a 0.1 s step of a 0.5 s recording |
| teleport judged by distance | the test's step was 1 s, where 200 m/s and 200 m coincide | a 50 m hop in 0.1 s |

All 13 are caught now.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`; GPU probes (belief, geometry, picking) | 1189 passed, 0 failed; 4 + 4 + 4 passed |
| clippy `-D warnings` (workspace, and viewer without `live-stream`); `cargo fmt --check` | clean |
| CI's headless smoke, locally | exit 0, 10,561 colours, 258 green track pixels: unchanged, since the still is taken at the last frame, exactly |
| The timeline panel | **not seen rendered.** egui does not run headless, and the interactive window was not driven. It builds, and every number it shows comes from tested `ReplayState` methods |
| CI | not run: this branch is not pushed |

## Not in this slice

* **Discrete mission transitions and world revisions** do not yet block the
  blend. The replay carries no world revision, and the mission phase is
  only in scan-mission state. VIEWER_UI §8 asks for both.
* **Durable event journal.** Events come from the frames. VIEWER_UI §8
  wants them from the runtime's ordered journal, which arrives with P5.0's
  snapshot protocol.
