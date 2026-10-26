//! Replay playback state, and the viewer's re-export of the replay DTOs.
//!
//! The document types themselves live in `argusnet-replay`, which is free of
//! Bevy so anything can read a replay without linking a renderer. What stays
//! here is what is actually about *viewing* one: the frame cursor, its
//! interpolation alpha and its epoch. The markers built from frames are
//! [`crate::presentation`]'s.
//!
//! The DTOs are re-exported rather than referenced through their crate so the
//! rest of the viewer, and its tests, keep importing `crate::replay::*` as
//! before.

use std::sync::Arc;

use bevy::prelude::Resource;
use bevy::prelude::Vec3;

use crate::presentation::EntityKey;

pub use argusnet_replay::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerKind {
    Track,
    Truth,
    Node,
}

/// One presented domain entity. Built by [`crate::presentation`].
#[derive(Debug, Clone)]
pub struct RuntimeMarker {
    pub kind: MarkerKind,
    /// Kind and interned id: the identity. `label` alone is not one, since a
    /// track shares its target's id.
    pub key: EntityKey,
    /// Shared with the epoch's id table, so presenting a frame clones no
    /// strings.
    pub label: Arc<str>,
    pub position: Vec3,
    pub velocity: Vec3,
    /// For `Node` markers: whether this node is a mobile platform (drone).
    /// `None` for tracks and truths.
    pub is_mobile: Option<bool>,
    /// For `Track` markers: the track's measurement std, which sets its
    /// quality colour and size. Carried here so appearance needs no second
    /// search of the frame.
    pub measurement_std_m: Option<f32>,
}

/// A timestamp step longer than this many nominal steps is a recording gap:
/// nothing is interpolated across it, and the timeline shows it as
/// unavailable.
pub const GAP_STEPS: f32 = 3.0;

/// How many intervals the nominal step is the median of.
pub const STEP_SAMPLES: usize = 31;

/// Playback over source time (P2.4a).
///
/// The cursor is a **source time**, not a frame count: playing advances it by
/// render time times the speed, and the frame shown is the last one recorded
/// at or before it. So a replay with irregular timestamps plays at the rate
/// it was recorded, and the blend toward the next frame is the true fraction
/// of the step between their timestamps. Before P2.4a playback counted frames
/// at a fixed step (`meta.dt_s`, or the first interval), which played an
/// irregular replay at the wrong speed and blended with the wrong fraction.
///
/// Nothing is interpolated across a recording gap (see [`GAP_STEPS`]); the
/// presentation adapter also holds an entity that teleports. `frame_index`
/// stays public for readers; move the cursor with [`Self::step_to`] or
/// [`Self::seek_time`].
#[derive(Debug, Clone, Resource)]
pub struct ReplayState {
    pub document: Option<ReplayDocument>,
    pub playing: bool,
    pub playback_speed: f32,
    /// The frame at or before the cursor.
    pub frame_index: usize,
    /// Wrap to the start at the end instead of stopping. Not for live history.
    pub looping: bool,
    /// The frames are a live stream's history: read-only, and trimmed from
    /// the front as it grows. Seeking reviews it; it never rewinds a runtime.
    pub live: bool,
    /// The cursor, in source seconds.
    time_s: f32,
    /// Live history dropped before this source time.
    trimmed_before_s: Option<f32>,
    /// Which run the frames belong to. See [`Self::epoch`].
    epoch: u64,
    /// The mission's decisions (F3), when the source reports them: an
    /// in-process run does, in tick order. `None` for a replay, which never
    /// records them (they are journaled), so the inspector says so.
    pub decisions: Option<Vec<RecordedDecision>>,
}

/// One mission decision (F3) as an in-process run reported it, stamped with
/// its tick's time.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedDecision {
    pub timestamp_s: f32,
    pub kind: String,
    /// `applied`, `queued_for_next_tick` or `rejected`.
    pub status: String,
    pub reason: String,
    /// The platform it is about.
    pub subject: Option<String>,
    /// What it chose: a POI, a receiving drone, a phase.
    pub selected: Option<String>,
    pub detail: String,
}

impl ReplayState {
    /// Record a tick's decisions, if this state takes them.
    pub fn push_decisions(&mut self, decisions: impl IntoIterator<Item = RecordedDecision>) {
        if let Some(log) = self.decisions.as_mut() {
            log.extend(decisions);
        }
    }
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            document: None,
            playing: false,
            playback_speed: 1.0,
            frame_index: 0,
            looping: false,
            live: false,
            time_s: 0.0,
            trimmed_before_s: None,
            epoch: 0,
            decisions: None,
        }
    }
}

/// A frame's events, as the timeline counts them: rejections, deconfliction,
/// inspection and safety events.
pub fn frame_event_count(frame: &ReplayFrame) -> usize {
    frame.generation_rejections.len()
        + frame.rejected_observations.len()
        + frame.deconfliction_events.len()
        + frame.inspection_events.len()
        + frame.safety_events.len()
}

impl ReplayState {
    pub fn new(document: Option<ReplayDocument>) -> Self {
        let mut state = Self {
            document,
            ..Self::default()
        };
        state.step_to(0);
        state
    }

    /// Which run the frames belong to.
    ///
    /// Presented identities are valid only within an epoch. A new epoch
    /// begins when the document is replaced by a different run (a reload), or
    /// when a live stream reconnects to a different session. Pausing, seeking,
    /// filtering and trimming live history do not begin one.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Replace the document with a different run's, from its first frame.
    pub fn begin_epoch(&mut self, document: Option<ReplayDocument>) {
        self.document = document;
        self.trimmed_before_s = None;
        // Another run's decisions are not this one's; a source that reports
        // them keeps reporting them.
        if let Some(log) = self.decisions.as_mut() {
            log.clear();
        }
        self.epoch += 1;
        self.step_to(0);
    }

    /// A seek branched the run (P5.2): keep the first `keep` frames, which
    /// the branch shares with the history it left, drop the rest, and begin a
    /// new epoch, so what was presented from the abandoned branch is not
    /// carried into this one. The cursor stays where it was, within the kept
    /// frames.
    pub fn branch(&mut self, keep: usize) {
        if let Some(document) = self.document.as_mut() {
            document.frames.truncate(keep);
        }
        self.trimmed_before_s = None;
        self.epoch += 1;
        self.step_to(self.frame_index);
    }

    /// Carry `previous`'s epoch forward, to one after it. For a state built
    /// fresh to replace `previous` with another run.
    pub fn succeed(&mut self, previous: &ReplayState) {
        self.epoch = previous.epoch + 1;
    }

    /// Append a frame, as live streaming does. Same epoch.
    pub fn push_frame(&mut self, frame: ReplayFrame) {
        let first = self.document.as_ref().is_none_or(|d| d.frames.is_empty());
        self.document
            .get_or_insert_with(|| ReplayDocument {
                meta: None,
                summary: None,
                frames: Vec::new(),
                mission_assurance: None,
            })
            .frames
            .push(frame);
        if first {
            self.step_to(0);
        }
    }

    /// Drop the oldest `count` frames, keeping the cursor on the same frame
    /// where it survives. Same epoch. What was dropped is reported as
    /// unavailable history.
    pub fn drain_front(&mut self, count: usize) {
        let time = self.time_s();
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let count = count.min(document.frames.len());
        if count == 0 {
            return;
        }
        document.frames.drain(..count);
        self.trimmed_before_s = document.frames.first().map(|f| f.timestamp_s);
        self.frame_index = self.frame_index.saturating_sub(count);
        // The cursor keeps its time where that time survives.
        match self.time_range() {
            Some((first, _)) if time < first => self.step_to(0),
            Some(_) => self.time_s = time,
            None => {}
        }
    }

    /// Stale-step threshold for the loaded run.
    ///
    /// Falls back to [`DEFAULT_MAX_STALE_STEPS`] when the replay predates the
    /// `meta.platform` block or when running live without a loaded document.
    pub fn max_stale_steps(&self) -> u32 {
        self.document
            .as_ref()
            .and_then(|document| document.max_stale_steps())
            .unwrap_or(DEFAULT_MAX_STALE_STEPS)
    }

    pub fn frame_count(&self) -> usize {
        self.document
            .as_ref()
            .map(|document| document.frames.len())
            .unwrap_or(0)
    }

    pub fn current_frame(&self) -> Option<&ReplayFrame> {
        let document = self.document.as_ref()?;
        document.frames.get(
            self.frame_index
                .min(document.frames.len().saturating_sub(1)),
        )
    }

    pub fn current_timestamp_s(&self) -> f32 {
        self.current_frame()
            .map(|frame| frame.timestamp_s)
            .unwrap_or(0.0)
    }

    fn timestamp(&self, index: usize) -> Option<f32> {
        self.document
            .as_ref()?
            .frames
            .get(index)
            .map(|frame| frame.timestamp_s)
    }

    /// The first and last recorded times.
    pub fn time_range(&self) -> Option<(f32, f32)> {
        let frames = &self.document.as_ref()?.frames;
        Some((frames.first()?.timestamp_s, frames.last()?.timestamp_s))
    }

    /// The cursor, in source seconds. Always within the current frame's step:
    /// if `frame_index` was set directly, the cursor is that frame's time.
    pub fn time_s(&self) -> f32 {
        let Some(at) = self.timestamp(self.frame_index) else {
            return 0.0;
        };
        let next = self.timestamp(self.frame_index + 1);
        let within = self.time_s >= at && next.is_none_or(|next| self.time_s < next);
        if within {
            self.time_s
        } else {
            at
        }
    }

    /// The last frame recorded at or before `time_s` (the first frame for a
    /// time before it). Timestamps are non-decreasing.
    pub fn frame_at(&self, time_s: f32) -> usize {
        let Some(document) = self.document.as_ref() else {
            return 0;
        };
        document
            .frames
            .partition_point(|frame| frame.timestamp_s <= time_s)
            .saturating_sub(1)
    }

    /// Move the cursor to `time_s`, clamped to the recorded range.
    pub fn seek_time(&mut self, time_s: f32) {
        let Some((first, last)) = self.time_range() else {
            self.frame_index = 0;
            self.time_s = 0.0;
            return;
        };
        let time_s = if time_s.is_finite() {
            time_s.clamp(first, last)
        } else {
            first
        };
        self.frame_index = self.frame_at(time_s);
        self.time_s = time_s;
    }

    /// Move the cursor to a frame, exactly as recorded.
    pub fn step_to(&mut self, frame_index: usize) {
        let frame_count = self.frame_count();
        self.frame_index = frame_index.min(frame_count.saturating_sub(1));
        self.time_s = self.timestamp(self.frame_index).unwrap_or(0.0);
    }

    /// Play forward by `delta_s` of render time, times the speed.
    pub fn advance(&mut self, delta_s: f32) {
        if !self.playing || self.frame_count() <= 1 {
            return;
        }
        let Some((first, last)) = self.time_range() else {
            return;
        };
        let mut time = self.time_s() + delta_s.max(0.0) * self.playback_speed.max(0.1);
        if time >= last {
            if self.looping && !self.live && last > first {
                time = first + (time - last) % (last - first);
            } else {
                time = last;
                self.playing = false;
            }
        }
        self.seek_time(time);
    }

    /// How far the cursor is into the current frame's step, in `0.0..=1.0`:
    /// the blend toward the next frame. 0 on the last frame, and across a
    /// recording gap, which is not interpolated.
    pub fn interpolation_alpha(&self) -> f32 {
        let Some(step) = self.blend_step_s() else {
            return 0.0;
        };
        let at = self.timestamp(self.frame_index).unwrap_or(0.0);
        ((self.time_s() - at) / step).clamp(0.0, 1.0)
    }

    /// The length of the step from the current frame to the next, when it
    /// may be interpolated across: positive, and not a recording gap.
    pub fn blend_step_s(&self) -> Option<f32> {
        let at = self.timestamp(self.frame_index)?;
        let next = self.timestamp(self.frame_index + 1)?;
        let step = next - at;
        (step > 0.0 && step <= GAP_STEPS * self.frame_step_s()).then_some(step)
    }

    /// The nominal step: the recorded `meta.dt_s`, else the median of up to
    /// [`STEP_SAMPLES`] positive intervals spread evenly over the recording.
    /// Sampled on the stack: this runs every presented frame, and a paused
    /// frame allocates nothing (P2.0's budget).
    pub fn frame_step_s(&self) -> f32 {
        let Some(document) = self.document.as_ref() else {
            return 0.25;
        };
        if let Some(meta) = &document.meta {
            if let Some(value) = meta.get("dt_s").and_then(|value| value.as_f64()) {
                return value.max(0.01) as f32;
            }
        }
        let frames = &document.frames;
        let intervals = frames.len().saturating_sub(1);
        let stride = intervals.div_ceil(STEP_SAMPLES).max(1);
        let mut samples = [0.0_f32; STEP_SAMPLES];
        let mut n = 0;
        for i in (0..intervals).step_by(stride) {
            let step = frames[i + 1].timestamp_s - frames[i].timestamp_s;
            if step > 0.0 && n < STEP_SAMPLES {
                samples[n] = step;
                n += 1;
            }
        }
        if n == 0 {
            return 0.25;
        }
        let samples = &mut samples[..n];
        samples.sort_by(f32::total_cmp);
        samples[n / 2].max(0.01)
    }

    /// Recording gaps: steps longer than [`GAP_STEPS`] nominal steps, as
    /// `(from, to)` source times. Seeking into one shows the frame before it.
    pub fn gaps(&self) -> Vec<(f32, f32)> {
        let Some(document) = self.document.as_ref() else {
            return Vec::new();
        };
        let limit = GAP_STEPS * self.frame_step_s();
        document
            .frames
            .windows(2)
            .map(|pair| (pair[0].timestamp_s, pair[1].timestamp_s))
            .filter(|(from, to)| to - from > limit)
            .collect()
    }

    /// Live history dropped before this source time, if any was.
    pub fn trimmed_before_s(&self) -> Option<f32> {
        self.trimmed_before_s
    }

    /// Source times of the frames that carry events, in order.
    pub fn event_times(&self) -> Vec<f32> {
        self.document
            .as_ref()
            .map(|document| {
                document
                    .frames
                    .iter()
                    .filter(|frame| frame_event_count(frame) > 0)
                    .map(|frame| frame.timestamp_s)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Jump to the next event frame after the current frame (or the previous
    /// one before it), wrapping. Pauses. Returns whether there was one.
    pub fn jump_to_event(&mut self, forward: bool) -> bool {
        let events = self.event_times();
        let now = self.current_timestamp_s();
        let target = if forward {
            events
                .iter()
                .copied()
                .find(|t| *t > now)
                .or(events.first().copied())
        } else {
            events
                .iter()
                .rev()
                .copied()
                .find(|t| *t < now)
                .or(events.last().copied())
        };
        let Some(target) = target else {
            return false;
        };
        self.playing = false;
        self.seek_time(target);
        true
    }
}
#[cfg(test)]
mod replay_state_tests {
    use serde_json::json;

    use super::{MarkerKind, ReplayDocument, ReplayState, DEFAULT_MAX_STALE_STEPS};

    #[test]
    fn replay_state_advances_and_stops_at_last_frame() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.5},
            "frames": [
                {"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []},
                {"timestamp_s": 0.5, "tracks": [], "truths": [], "nodes": []},
                {"timestamp_s": 1.0, "tracks": [], "truths": [], "nodes": []}
            ]
        }))
        .unwrap();
        let mut replay = ReplayState::new(Some(document));
        replay.playing = true;
        replay.advance(1.1);
        assert_eq!(replay.frame_index, 2);
        assert!(!replay.playing);
    }

    /// Frames at the given times, no `meta.dt_s`, and a safety event on the
    /// frames listed in `events`.
    fn timed(times: &[f32], events: &[usize]) -> ReplayState {
        let frames: Vec<_> = times
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let safety = if events.contains(&i) {
                    json!([{"event_id": format!("e{i}"), "subject_id": "d", "timestamp_s": t, "state": "caution",
                            "violations": [], "command_clamped": false, "blocked": false, "reason": ""}])
                } else {
                    json!([])
                };
                json!({"timestamp_s": t, "tracks": [], "truths": [], "nodes": [], "safety_events": safety})
            })
            .collect();
        ReplayState::new(Some(
            serde_json::from_value(json!({ "frames": frames })).unwrap(),
        ))
    }

    /// Playback runs on source time: one second of play at 1x is one second
    /// of the recording, however irregular its frames. Counting frames at a
    /// fixed step (the first interval, 0.1 s) would have run ten frames on.
    #[test]
    fn irregular_timestamps_play_at_the_recorded_rate() {
        let mut replay = timed(&[0.0, 0.1, 0.5, 0.9, 1.3, 1.7], &[]);
        replay.playing = true;
        replay.advance(1.0);
        assert_eq!(replay.frame_index, 3, "the frame at or before t = 1.0");
        assert!((replay.time_s() - 1.0).abs() < 1e-6);
        // The blend is the true fraction of this step: (1.0 - 0.9) / 0.4.
        assert!((replay.interpolation_alpha() - 0.25).abs() < 1e-5);

        // Inside a step shorter than the nominal one (0.5 s), the fraction
        // is of that step: halfway through 1.0 -> 1.1 is 0.5, not 0.1.
        let mut replay = timed(&[0.0, 0.5, 1.0, 1.1, 1.6, 2.1], &[]);
        assert_eq!(replay.frame_step_s(), 0.5);
        replay.seek_time(1.05);
        assert_eq!(replay.frame_index, 2);
        assert!(
            (replay.interpolation_alpha() - 0.5).abs() < 1e-4,
            "{}",
            replay.interpolation_alpha()
        );
    }

    /// A step longer than GAP_STEPS nominal steps is a recording gap: it is
    /// reported, and nothing is blended across it.
    #[test]
    fn recording_gaps_are_reported_and_not_blended() {
        let mut replay = timed(&[0.0, 0.5, 1.0, 5.0, 5.5], &[]);
        assert_eq!(replay.frame_step_s(), 0.5, "the median step");
        assert_eq!(replay.gaps(), vec![(1.0, 5.0)]);
        replay.seek_time(3.0);
        assert_eq!(replay.frame_index, 2, "inside the gap: the frame before it");
        assert_eq!(replay.interpolation_alpha(), 0.0);
        assert_eq!(replay.blend_step_s(), None);
        replay.seek_time(0.25);
        assert!((replay.interpolation_alpha() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn seeking_clamps_to_the_recording_and_step_is_exact() {
        let mut replay = timed(&[2.0, 3.0, 4.0], &[]);
        replay.seek_time(-10.0);
        assert_eq!((replay.frame_index, replay.time_s()), (0, 2.0));
        replay.seek_time(99.0);
        assert_eq!((replay.frame_index, replay.time_s()), (2, 4.0));
        replay.seek_time(f32::NAN);
        assert_eq!(replay.frame_index, 0);
        replay.step_to(1);
        assert_eq!(replay.time_s(), 3.0);
        assert_eq!(
            replay.interpolation_alpha(),
            0.0,
            "a stepped frame is shown as recorded"
        );
        // A cursor out of step with a directly written frame index follows the index.
        replay.frame_index = 2;
        assert_eq!(replay.time_s(), 4.0);
    }

    #[test]
    fn loop_wraps_and_live_history_never_loops() {
        let mut replay = timed(&[0.0, 1.0, 2.0], &[]);
        replay.looping = true;
        replay.playing = true;
        replay.step_to(2);
        replay.advance(0.5);
        assert!(replay.playing, "a loop keeps playing");
        assert!(
            (replay.time_s() - 0.5).abs() < 1e-6,
            "wrapped to {}",
            replay.time_s()
        );
        replay.looping = false;
        replay.advance(5.0);
        assert!(!replay.playing);
        assert_eq!(replay.time_s(), 2.0);
        replay.looping = true;
        replay.live = true;
        replay.playing = true;
        replay.advance(5.0);
        assert!(!replay.playing, "live history stops at its tail");
    }

    /// Trimming live history keeps the cursor's time where it survives and
    /// reports what was dropped.
    #[test]
    fn trimmed_history_is_reported_and_the_cursor_keeps_its_time() {
        let mut replay = timed(&[0.0, 1.0, 2.0, 3.0], &[]);
        replay.seek_time(2.5);
        replay.drain_front(1);
        assert_eq!(replay.trimmed_before_s(), Some(1.0));
        assert!((replay.time_s() - 2.5).abs() < 1e-6);
        assert_eq!(replay.frame_index, 1);
        replay.drain_front(2);
        assert_eq!(replay.trimmed_before_s(), Some(3.0));
        assert_eq!(
            replay.time_s(),
            3.0,
            "a cursor in dropped history moves to the first frame kept"
        );
    }

    #[test]
    fn jump_to_event_steps_between_event_frames_and_wraps() {
        let mut replay = timed(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 3]);
        assert_eq!(replay.event_times(), vec![1.0, 3.0], "safety events count");
        replay.playing = true;
        assert!(replay.jump_to_event(true));
        assert!(!replay.playing, "a jump pauses");
        assert_eq!(replay.time_s(), 1.0);
        assert!(replay.jump_to_event(true));
        assert_eq!(replay.time_s(), 3.0);
        assert!(replay.jump_to_event(true));
        assert_eq!(replay.time_s(), 1.0, "wraps to the first");
        assert!(replay.jump_to_event(false));
        assert_eq!(replay.time_s(), 3.0, "wraps to the last");
        let mut quiet = timed(&[0.0, 1.0], &[]);
        assert!(!quiet.jump_to_event(true));
    }

    #[test]
    fn replay_state_builds_runtime_markers() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "t1", "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0],
                            "measurement_std_m": 12.5}],
                "truths": [{"target_id": "truth-1", "position": [4.0, 5.0, 6.0], "velocity": [1.0, 0.0, 0.0]}],
                "nodes": [{"node_id": "node-1", "position": [7.0, 8.0, 9.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}]
            }]
        }))
        .unwrap();
        let replay = ReplayState::new(Some(document));
        let mut presented = crate::presentation::CurrentRuntimeMarkers::default();
        presented.apply(&replay);
        let markers = &presented.markers;
        assert_eq!(markers.len(), 3);
        assert_eq!(markers[0].kind, MarkerKind::Track);
        assert_eq!(markers[0].measurement_std_m, Some(12.5));
        assert_eq!(markers[1].kind, MarkerKind::Truth);
        assert_eq!(markers[2].kind, MarkerKind::Node);
        assert_eq!(markers[2].is_mobile, Some(true));
    }

    #[test]
    fn only_a_different_run_begins_an_epoch() {
        let document = || -> ReplayDocument {
            serde_json::from_value(json!({"frames": [
                {"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []},
                {"timestamp_s": 1.0, "tracks": [], "truths": [], "nodes": []}
            ]}))
            .unwrap()
        };
        let mut replay = ReplayState::new(Some(document()));
        let epoch = replay.epoch();
        replay.step_to(1);
        replay.playing = true;
        replay.advance(0.3);
        replay.push_frame(document().frames.remove(0));
        replay.drain_front(1);
        assert_eq!(
            replay.epoch(),
            epoch,
            "seeking, playing and live history keep the epoch"
        );
        replay.begin_epoch(Some(document()));
        assert_eq!(replay.epoch(), epoch + 1);
        assert_eq!(replay.frame_index, 0);
        let mut next = ReplayState::new(Some(document()));
        next.succeed(&replay);
        assert_eq!(next.epoch(), epoch + 2);
    }

    #[test]
    fn stale_threshold_comes_from_the_replays_own_tracker_config() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25, "platform": {"min_observations": 2, "max_stale_steps": 6}},
            "frames": [{"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []}]
        }))
        .unwrap();
        let replay = ReplayState::new(Some(document));
        // The value the run recorded wins over any constant compiled into the viewer.
        assert_eq!(replay.max_stale_steps(), 6);
    }

    #[test]
    fn stale_threshold_falls_back_when_replay_predates_platform_meta() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25},
            "frames": [{"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []}]
        }))
        .unwrap();
        let replay = ReplayState::new(Some(document));
        assert_eq!(replay.max_stale_steps(), DEFAULT_MAX_STALE_STEPS);
        // Guards the drift this replaced: the viewer hardcoded 5 against a core default of 8.
        assert_eq!(DEFAULT_MAX_STALE_STEPS, 8);
    }
}
