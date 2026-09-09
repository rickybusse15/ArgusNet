//! Replay playback state, and the viewer's re-export of the replay DTOs.
//!
//! The document types themselves live in `argusnet-replay`, which is free of
//! Bevy so anything can read a replay without linking a renderer. What stays
//! here is what is actually about *viewing* one: the frame cursor, the
//! interpolation between frames, and the camera-facing markers built from them
//! -- all of which speak Bevy's `Vec3` and one of which is a `Resource`.
//!
//! The DTOs are re-exported rather than referenced through their crate so the
//! rest of the viewer, and its tests, keep importing `crate::replay::*` as
//! before.

use bevy::prelude::Resource;
use bevy::prelude::Vec3;

pub use argusnet_replay::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerKind {
    Track,
    Truth,
    Node,
}

#[derive(Debug, Clone)]
pub struct RuntimeMarker {
    pub kind: MarkerKind,
    pub label: String,
    pub position: Vec3,
    pub velocity: Vec3,
    /// For `Node` markers: whether this node is a mobile platform (drone).
    /// `None` for tracks and truths.
    pub is_mobile: Option<bool>,
}

#[derive(Debug, Clone, Resource)]
pub struct ReplayState {
    pub document: Option<ReplayDocument>,
    pub playing: bool,
    pub playback_speed: f32,
    pub frame_index: usize,
    pub accumulator_s: f32,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            document: None,
            playing: false,
            playback_speed: 1.0,
            frame_index: 0,
            accumulator_s: 0.0,
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

    /// Markers positioned between the current frame and the next.
    ///
    /// Replay frames arrive at the simulation's tick rate — a quarter second by
    /// default — while the viewer renders far faster, so drawing frame contents
    /// directly makes every platform jump on a tick boundary and freeze between
    /// them. Blending toward the next frame's pose spreads that motion across
    /// the frames actually rendered.
    ///
    /// An entity present now but absent from the next frame holds its current
    /// pose rather than blending toward nothing, so a track that ends does not
    /// slide to the origin on its way out.
    pub fn interpolated_markers(&self) -> Vec<RuntimeMarker> {
        let mut markers = self.current_markers();
        let alpha = self.interpolation_alpha();
        if alpha <= 0.0 {
            return markers;
        }
        let Some(document) = self.document.as_ref() else {
            return markers;
        };
        let Some(next_frame) = document.frames.get(self.frame_index + 1) else {
            return markers;
        };

        let next = Self::frame_markers(next_frame);
        for marker in markers.iter_mut() {
            let Some(target) = next
                .iter()
                .find(|candidate| candidate.kind == marker.kind && candidate.label == marker.label)
            else {
                continue;
            };
            marker.position = marker.position.lerp(target.position, alpha);
            marker.velocity = marker.velocity.lerp(target.velocity, alpha);
        }
        markers
    }

    pub fn current_markers(&self) -> Vec<RuntimeMarker> {
        let Some(frame) = self.current_frame() else {
            return Vec::new();
        };
        Self::frame_markers(frame)
    }

    /// Markers for one frame exactly as recorded, with no blending.
    fn frame_markers(frame: &ReplayFrame) -> Vec<RuntimeMarker> {
        let mut markers =
            Vec::with_capacity(frame.tracks.len() + frame.truths.len() + frame.nodes.len());
        for track in &frame.tracks {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Track,
                label: track.track_id.clone(),
                position: Vec3::from_array(track.position),
                velocity: Vec3::from_array(track.velocity),
                is_mobile: None,
            });
        }
        for truth in &frame.truths {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Truth,
                label: truth.target_id.clone(),
                position: Vec3::from_array(truth.position),
                velocity: Vec3::from_array(truth.velocity),
                is_mobile: None,
            });
        }
        for node in &frame.nodes {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Node,
                label: node.node_id.clone(),
                position: Vec3::from_array(node.position),
                velocity: Vec3::from_array(node.velocity),
                is_mobile: Some(node.is_mobile),
            });
        }
        markers
    }

    pub fn trail_points(&self, marker: &RuntimeMarker, max_samples: usize) -> Vec<Vec3> {
        let Some(document) = self.document.as_ref() else {
            return Vec::new();
        };
        let start = self
            .frame_index
            .saturating_sub(max_samples.saturating_sub(1));
        let mut points = Vec::new();
        for frame in &document.frames[start
            ..=self
                .frame_index
                .min(document.frames.len().saturating_sub(1))]
        {
            match marker.kind {
                MarkerKind::Track => {
                    if let Some(track) = frame
                        .tracks
                        .iter()
                        .find(|item| item.track_id == marker.label)
                    {
                        points.push(Vec3::from_array(track.position));
                    }
                }
                MarkerKind::Truth => {
                    if let Some(truth) = frame
                        .truths
                        .iter()
                        .find(|item| item.target_id == marker.label)
                    {
                        points.push(Vec3::from_array(truth.position));
                    }
                }
                MarkerKind::Node => {
                    if let Some(node) = frame.nodes.iter().find(|item| item.node_id == marker.label)
                    {
                        points.push(Vec3::from_array(node.position));
                    }
                }
            }
        }
        points
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
mod interpolation_tests {
    use super::*;

    /// Two frames a second apart, one track moving 100 m along +X.
    fn moving_track_replay() -> ReplayState {
        let document: ReplayDocument = serde_json::from_value(serde_json::json!({
            "frames": [
                {
                    "timestamp_s": 0.0,
                    "tracks": [{"track_id": "t1", "position": [0.0, 0.0, 0.0], "velocity": [100.0, 0.0, 0.0]}],
                    "truths": [], "nodes": []
                },
                {
                    "timestamp_s": 1.0,
                    "tracks": [{"track_id": "t1", "position": [100.0, 0.0, 0.0], "velocity": [100.0, 0.0, 0.0]}],
                    "truths": [], "nodes": []
                }
            ]
        }))
        .expect("fixture parses");
        ReplayState::new(Some(document))
    }

    #[test]
    fn a_seeked_frame_is_shown_exactly() {
        let mut replay = moving_track_replay();
        replay.step_to(0);
        assert_eq!(replay.interpolation_alpha(), 0.0);
        assert_eq!(replay.interpolated_markers()[0].position.x, 0.0);
    }

    #[test]
    fn playback_blends_toward_the_next_frame() {
        let mut replay = moving_track_replay();
        replay.playing = true;
        // Half a frame step in: halfway between the two recorded positions.
        replay.advance(0.5);
        assert!((replay.interpolation_alpha() - 0.5).abs() < 1e-6);
        let x = replay.interpolated_markers()[0].position.x;
        assert!((x - 50.0).abs() < 1e-3, "expected midpoint, got {x}");
    }

    #[test]
    fn interpolation_never_overshoots_the_next_frame() {
        let mut replay = moving_track_replay();
        replay.playing = true;
        // Far more than one frame step; alpha must saturate rather than run past
        // the next frame's pose.
        replay.accumulator_s = 10.0;
        assert_eq!(replay.interpolation_alpha(), 1.0);
        let x = replay.interpolated_markers()[0].position.x;
        assert!(
            (x - 100.0).abs() < 1e-3,
            "expected next-frame pose, got {x}"
        );
    }

    #[test]
    fn an_entity_absent_from_the_next_frame_holds_its_pose() {
        let document: ReplayDocument = serde_json::from_value(serde_json::json!({
            "frames": [
                {
                    "timestamp_s": 0.0,
                    "tracks": [{"track_id": "ending", "position": [10.0, 20.0, 30.0], "velocity": [0.0, 0.0, 0.0]}],
                    "truths": [], "nodes": []
                },
                {"timestamp_s": 1.0, "tracks": [], "truths": [], "nodes": []}
            ]
        }))
        .expect("fixture parses");
        let mut replay = ReplayState::new(Some(document));
        replay.playing = true;
        replay.advance(0.5);
        // Must not slide toward the origin on the way out.
        let marker = &replay.interpolated_markers()[0];
        assert_eq!(marker.position, Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn the_final_frame_has_nothing_to_blend_toward() {
        let mut replay = moving_track_replay();
        replay.step_to(1);
        replay.accumulator_s = 0.5;
        let x = replay.interpolated_markers()[0].position.x;
        assert!((x - 100.0).abs() < 1e-3);
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
                "tracks": [{"track_id": "t1", "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0]}],
                "truths": [{"target_id": "truth-1", "position": [4.0, 5.0, 6.0], "velocity": [1.0, 0.0, 0.0]}],
                "nodes": [{"node_id": "node-1", "position": [7.0, 8.0, 9.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}]
            }]
        }))
        .unwrap();
        let replay = ReplayState::new(Some(document));
        let markers = replay.current_markers();
        assert_eq!(markers.len(), 3);
        assert_eq!(markers[0].kind, MarkerKind::Track);
        assert_eq!(markers[1].kind, MarkerKind::Truth);
        assert_eq!(markers[2].kind, MarkerKind::Node);
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
