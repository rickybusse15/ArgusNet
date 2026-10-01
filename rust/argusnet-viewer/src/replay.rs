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

#[derive(Debug, Clone, Resource)]
pub struct ReplayState {
    pub document: Option<ReplayDocument>,
    pub playing: bool,
    pub playback_speed: f32,
    pub frame_index: usize,
    pub accumulator_s: f32,
    /// Which run the frames belong to. See [`Self::epoch`].
    epoch: u64,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            document: None,
            playing: false,
            playback_speed: 1.0,
            frame_index: 0,
            accumulator_s: 0.0,
            epoch: 0,
        }
    }
}

impl ReplayState {
    pub fn new(document: Option<ReplayDocument>) -> Self {
        Self {
            document,
            ..Self::default()
        }
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
        self.frame_index = 0;
        self.accumulator_s = 0.0;
        self.epoch += 1;
    }

    /// Carry `previous`'s epoch forward, to one after it. For a state built
    /// fresh to replace `previous` with another run.
    pub fn succeed(&mut self, previous: &ReplayState) {
        self.epoch = previous.epoch + 1;
    }

    /// Append a frame, as live streaming does. Same epoch.
    pub fn push_frame(&mut self, frame: ReplayFrame) {
        self.document
            .get_or_insert_with(|| ReplayDocument {
                meta: None,
                summary: None,
                frames: Vec::new(),
            })
            .frames
            .push(frame);
    }

    /// Drop the oldest `count` frames, keeping the cursor on the same frame
    /// where it survives. Same epoch.
    pub fn drain_front(&mut self, count: usize) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let count = count.min(document.frames.len());
        document.frames.drain(..count);
        self.frame_index = self.frame_index.saturating_sub(count);
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

    pub fn step_to(&mut self, frame_index: usize) {
        let frame_count = self.frame_count();
        self.frame_index = frame_index.min(frame_count.saturating_sub(1));
        self.accumulator_s = 0.0;
    }

    pub fn advance(&mut self, delta_s: f32) {
        if !self.playing || self.frame_count() <= 1 {
            return;
        }

        self.accumulator_s += delta_s.max(0.0) * self.playback_speed.max(0.1);
        let frame_step = self.frame_step_s();
        while self.accumulator_s >= frame_step {
            self.accumulator_s -= frame_step;
            if self.frame_index + 1 >= self.frame_count() {
                self.frame_index = self.frame_count().saturating_sub(1);
                self.playing = false;
                self.accumulator_s = 0.0;
                break;
            }
            self.frame_index += 1;
            if self.frame_index + 1 >= self.frame_count() {
                self.playing = false;
                self.accumulator_s = 0.0;
                break;
            }
        }
    }

    /// How far playback has advanced into the current frame, in `0.0..=1.0`.
    ///
    /// `advance` already tracks this as `accumulator_s`; expressing it as a
    /// fraction of the frame step turns it into the blend factor between the
    /// current frame and the next. Seeking zeroes the accumulator, so a paused
    /// or scrubbed timeline reports 0 and shows the frame exactly as recorded.
    pub fn interpolation_alpha(&self) -> f32 {
        let frame_step = self.frame_step_s();
        if frame_step <= 0.0 {
            return 0.0;
        }
        (self.accumulator_s / frame_step).clamp(0.0, 1.0)
    }

    pub fn frame_step_s(&self) -> f32 {
        let Some(document) = self.document.as_ref() else {
            return 0.25;
        };
        if let Some(meta) = &document.meta {
            if let Some(value) = meta.get("dt_s").and_then(|value| value.as_f64()) {
                return value.max(0.01) as f32;
            }
        }
        if document.frames.len() >= 2 {
            return (document.frames[1].timestamp_s - document.frames[0].timestamp_s)
                .abs()
                .max(0.01);
        }
        0.25
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
