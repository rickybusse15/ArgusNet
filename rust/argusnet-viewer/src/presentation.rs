//! The snapshot adapter: replay frames in, persistent presentation state out
//! (P2.0).
//!
//! Before this module the viewer rebuilt its whole marker list every rendered
//! frame: every label cloned, the next frame searched once per marker
//! (quadratic), and each marker's trail rescanned across 32 frames with a
//! linear search per frame. A track's quality colour searched the frame's
//! tracks again for every track. With 240 markers that was about 1,100
//! allocations a frame, playing or paused.
//!
//! Now [`CurrentRuntimeMarkers`] is updated in place by [`apply`]:
//!
//! * **Membership** (which entities exist, and their recorded poses, quality
//!   and trails) is rebuilt only when the *source frame* changes: a new frame
//!   index, a new frame count (live history), or a new epoch. The next frame
//!   is matched through a key index, so the rebuild is linear.
//! * **Poses** are re-blended only when the interpolation alpha changes, in
//!   place, with no allocation.
//! * Nothing happens at all on a paused frame.
//!
//! Each change bumps a revision counter, so downstream systems skip work
//! whose inputs did not change.
//!
//! # Identity
//!
//! An entity is an [`EntityKey`]: its kind and an interned id. A track and
//! a truth share a label (a track is keyed by its target's id), so a label
//! alone is not an identity; that confusion once let selecting a truth
//! resolve to its track. Keys are only meaningful within one [`epoch`]: a new
//! replay, or a live stream from a different session, starts a new id table,
//! and everything keyed to the old one (entities, selection, interpolation)
//! is dropped.
//!
//! [`apply`]: CurrentRuntimeMarkers::apply
//! [`epoch`]: crate::replay::ReplayState::epoch

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;

use crate::replay::{MarkerKind, ReplayFrame, ReplayState, RuntimeMarker};

/// How many recorded frames a trail spans.
pub const TRAIL_MAX_FRAMES: usize = 32;

/// A step faster than this is a teleport, not motion: nothing is
/// interpolated across it, and a trail restarts at it.
pub const TELEPORT_SPEED_MPS: f32 = 200.0;

/// An interned id, valid within one epoch's [`IdTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IdIndex(u32);

impl IdIndex {
    /// An index for tests that need a key without an id table.
    #[cfg(test)]
    pub fn for_tests(n: usize) -> Self {
        Self(n as u32)
    }
}

/// Which domain entity a marker presents: its kind, and its interned id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Component)]
pub enum EntityKey {
    Platform(IdIndex),
    Track(IdIndex),
    Truth(IdIndex),
}

impl EntityKey {
    pub fn new(kind: MarkerKind, id: IdIndex) -> Self {
        match kind {
            MarkerKind::Node => Self::Platform(id),
            MarkerKind::Track => Self::Track(id),
            MarkerKind::Truth => Self::Truth(id),
        }
    }

    pub fn kind(self) -> MarkerKind {
        match self {
            Self::Platform(_) => MarkerKind::Node,
            Self::Track(_) => MarkerKind::Track,
            Self::Truth(_) => MarkerKind::Truth,
        }
    }

    pub fn id(self) -> IdIndex {
        match self {
            Self::Platform(id) | Self::Track(id) | Self::Truth(id) => id,
        }
    }
}

/// Every id seen in an epoch, each stored once.
///
/// Interning an id already seen allocates nothing, so a rebuilt marker list
/// shares its labels with the table rather than cloning strings.
#[derive(Debug, Default, Clone)]
pub struct IdTable {
    names: Vec<Arc<str>>,
    index: HashMap<Arc<str>, IdIndex>,
}

impl IdTable {
    pub fn intern(&mut self, name: &str) -> IdIndex {
        if let Some(id) = self.index.get(name) {
            return *id;
        }
        let id = IdIndex(u32::try_from(self.names.len()).expect("fewer than 2^32 ids"));
        let name: Arc<str> = Arc::from(name);
        self.names.push(Arc::clone(&name));
        self.index.insert(name, id);
        id
    }

    pub fn get(&self, name: &str) -> Option<IdIndex> {
        self.index.get(name).copied()
    }

    pub fn name(&self, id: IdIndex) -> &Arc<str> {
        &self.names[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// The frame the presented markers were built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Source {
    frame_index: usize,
    frame_count: usize,
    timestamp_bits: u32,
}

/// What one [`CurrentRuntimeMarkers::apply`] changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentationChange {
    /// A new epoch began: every key from the previous one is void.
    pub epoch: bool,
    /// Entities appeared or disappeared.
    pub membership: bool,
    /// The source frame changed (or went away): anything read from the frame
    /// itself, such as its metrics, is stale.
    pub frame: bool,
    /// Some pose, quality or trail changed.
    pub poses: bool,
}

/// The presented domain entities: persistent, updated in place.
#[derive(Debug, Default, Resource)]
pub struct CurrentRuntimeMarkers {
    /// In frame order: tracks, then truths, then nodes.
    pub markers: Vec<RuntimeMarker>,
    epoch: u64,
    membership_revision: u64,
    pose_revision: u64,
    ids: IdTable,
    source: Option<Source>,
    /// Each marker's recorded pose in the source frame.
    recorded: Vec<(Vec3, Vec3)>,
    /// Each marker's pose in the next frame, where it has one.
    next: Vec<Option<(Vec3, Vec3)>>,
    /// Each marker's recent recorded positions, oldest first, with the frame
    /// each came from.
    trails: Vec<Vec<(usize, Vec3)>>,
    /// `key -> index into markers`, for the source frame.
    index: HashMap<EntityKey, usize>,
    /// The previous membership, to tell a pose change from a membership one.
    previous_keys: Vec<EntityKey>,
    alpha_bits: Option<u32>,
}

impl CurrentRuntimeMarkers {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Bumped whenever an entity appears or disappears, or the epoch turns.
    pub fn membership_revision(&self) -> u64 {
        self.membership_revision
    }

    /// Bumped whenever any presented pose, quality or trail changes.
    pub fn pose_revision(&self) -> u64 {
        self.pose_revision
    }

    pub fn ids(&self) -> &IdTable {
        &self.ids
    }

    /// Where `key` sits in [`Self::markers`], in the source frame.
    pub fn index_of(&self, key: EntityKey) -> Option<usize> {
        self.index.get(&key).copied()
    }

    pub fn get(&self, key: EntityKey) -> Option<&RuntimeMarker> {
        self.index_of(key).map(|index| &self.markers[index])
    }

    /// The key an id of `kind` has in this epoch, if it has been seen.
    pub fn key_of(&self, kind: MarkerKind, id: &str) -> Option<EntityKey> {
        self.ids.get(id).map(|id| EntityKey::new(kind, id))
    }

    /// Marker `index`'s recorded positions over the last `frames` frames up to
    /// the source frame, oldest first. At most [`TRAIL_MAX_FRAMES`].
    pub fn trail(&self, index: usize, frames: usize) -> impl Iterator<Item = Vec3> + '_ {
        let trail = self.trails.get(index).map(Vec::as_slice).unwrap_or(&[]);
        let oldest = self
            .source
            .map(|source| source.frame_index + 1)
            .unwrap_or(0)
            .saturating_sub(frames);
        let start = trail.partition_point(|(frame, _)| *frame < oldest);
        trail[start..].iter().map(|(_, position)| *position)
    }

    /// Bring the presentation up to date with `replay`.
    pub fn apply(&mut self, replay: &ReplayState) -> PresentationChange {
        let mut change = PresentationChange::default();
        if replay.epoch() != self.epoch {
            self.begin_epoch(replay.epoch());
            change.epoch = true;
            change.membership = true;
        }

        let Some(frame) = replay.current_frame() else {
            if !self.markers.is_empty() || change.epoch {
                self.clear_frame();
                change.membership = true;
                change.poses = true;
            }
            change.frame |= self.source.is_some();
            self.source = None;
            self.finish(change);
            return change;
        };

        let source = Source {
            frame_index: replay
                .frame_index
                .min(replay.frame_count().saturating_sub(1)),
            frame_count: replay.frame_count(),
            timestamp_bits: frame.timestamp_s.to_bits(),
        };
        if self.source != Some(source) {
            if self.rebuild(replay, frame, source.frame_index) {
                change.membership = true;
            }
            self.source = Some(source);
            self.alpha_bits = None;
            change.frame = true;
            change.poses = true;
        }

        let alpha = replay.interpolation_alpha();
        if self.alpha_bits != Some(alpha.to_bits()) {
            self.blend(alpha);
            self.alpha_bits = Some(alpha.to_bits());
            change.poses = true;
        }
        self.finish(change);
        change
    }

    fn finish(&mut self, change: PresentationChange) {
        if change.membership {
            self.membership_revision += 1;
        }
        if change.poses || change.membership {
            self.pose_revision += 1;
        }
    }

    fn begin_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.ids = IdTable::default();
        self.clear_frame();
        self.previous_keys.clear();
        self.source = None;
        self.alpha_bits = None;
    }

    fn clear_frame(&mut self) {
        self.markers.clear();
        self.recorded.clear();
        self.next.clear();
        self.index.clear();
        for trail in &mut self.trails {
            trail.clear();
        }
    }

    /// Rebuild membership from `frame`. Returns whether the set of entities,
    /// in order, changed.
    fn rebuild(&mut self, replay: &ReplayState, frame: &ReplayFrame, frame_index: usize) -> bool {
        self.previous_keys.clear();
        self.previous_keys
            .extend(self.markers.iter().map(|marker| marker.key));
        self.clear_frame();

        for (kind, id, position, velocity, is_mobile, std_m) in frame_entities(frame) {
            let key = EntityKey::new(kind, self.ids.intern(id));
            let index = self.markers.len();
            // First occurrence wins, as a linear search would find it.
            self.index.entry(key).or_insert(index);
            self.markers.push(RuntimeMarker {
                kind,
                key,
                label: Arc::clone(self.ids.name(key.id())),
                position,
                velocity,
                is_mobile,
                measurement_std_m: std_m,
            });
            self.recorded.push((position, velocity));
        }

        self.next.resize(self.markers.len(), None);
        // Only a step that may be interpolated at all: not the last frame,
        // not across a recording gap (P2.4a).
        let next_frame = replay.blend_step_s().and_then(|step| {
            let document = replay.document.as_ref()?;
            Some((document.frames.get(frame_index + 1)?, step))
        });
        if let Some((next_frame, step)) = next_frame {
            for (kind, id, position, velocity, _, _) in frame_entities(next_frame) {
                let Some(id) = self.ids.get(id) else {
                    continue;
                };
                if let Some(&index) = self.index.get(&EntityKey::new(kind, id)) {
                    // A teleport holds its recorded pose rather than sliding
                    // through everything between.
                    let teleport =
                        position.distance(self.recorded[index].0) > TELEPORT_SPEED_MPS * step;
                    if !teleport {
                        self.next[index].get_or_insert((position, velocity));
                    }
                }
            }
        }

        if self.trails.len() < self.markers.len() {
            self.trails.resize_with(self.markers.len(), Vec::new);
        }
        if let Some(document) = replay.document.as_ref() {
            let start = frame_index.saturating_sub(TRAIL_MAX_FRAMES - 1);
            for (offset, past) in document.frames[start..=frame_index].iter().enumerate() {
                let past_index = start + offset;
                for (kind, id, position, _, _, _) in frame_entities(past) {
                    let Some(id) = self.ids.get(id) else {
                        continue;
                    };
                    if let Some(&index) = self.index.get(&EntityKey::new(kind, id)) {
                        let trail = &mut self.trails[index];
                        if trail.last().map(|(frame, _)| *frame) != Some(past_index) {
                            trail.push((past_index, position));
                        }
                    }
                }
            }
        }

        !self
            .markers
            .iter()
            .map(|marker| marker.key)
            .eq(self.previous_keys.iter().copied())
    }

    /// Blend every marker toward its next-frame pose, in place. A marker
    /// absent from the next frame holds its recorded pose rather than sliding
    /// toward nothing.
    fn blend(&mut self, alpha: f32) {
        for ((marker, recorded), next) in
            self.markers.iter_mut().zip(&self.recorded).zip(&self.next)
        {
            let (position, velocity) = match next {
                Some((next_position, next_velocity)) if alpha > 0.0 => (
                    recorded.0.lerp(*next_position, alpha),
                    recorded.1.lerp(*next_velocity, alpha),
                ),
                _ => *recorded,
            };
            marker.position = position;
            marker.velocity = velocity;
        }
    }
}

type FrameEntity<'a> = (MarkerKind, &'a str, Vec3, Vec3, Option<bool>, Option<f32>);

/// A frame's entities in marker order: tracks, truths, nodes.
fn frame_entities(frame: &ReplayFrame) -> impl Iterator<Item = FrameEntity<'_>> {
    let tracks = frame.tracks.iter().map(|track| {
        (
            MarkerKind::Track,
            track.track_id.as_str(),
            Vec3::from_array(track.position),
            Vec3::from_array(track.velocity),
            None,
            Some(track.measurement_std_m),
        )
    });
    let truths = frame.truths.iter().map(|truth| {
        (
            MarkerKind::Truth,
            truth.target_id.as_str(),
            Vec3::from_array(truth.position),
            Vec3::from_array(truth.velocity),
            None,
            None,
        )
    });
    let nodes = frame.nodes.iter().map(|node| {
        (
            MarkerKind::Node,
            node.node_id.as_str(),
            Vec3::from_array(node.position),
            Vec3::from_array(node.velocity),
            Some(node.is_mobile),
            None,
        )
    });
    tracks.chain(truths).chain(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::ReplayDocument;

    fn document(frames: serde_json::Value) -> ReplayDocument {
        serde_json::from_value(serde_json::json!({ "frames": frames })).expect("fixture parses")
    }

    /// Two frames a second apart, one track moving 100 m along +X.
    fn moving_track() -> ReplayState {
        ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [],
             "tracks": [{"track_id": "t1", "position": [0.0, 0.0, 0.0], "velocity": [100.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "truths": [], "nodes": [],
             "tracks": [{"track_id": "t1", "position": [100.0, 0.0, 0.0], "velocity": [100.0, 0.0, 0.0]}]},
        ]))))
    }

    fn presented(replay: &ReplayState) -> CurrentRuntimeMarkers {
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(replay);
        markers
    }

    #[test]
    fn a_seeked_frame_is_shown_exactly() {
        let mut replay = moving_track();
        replay.step_to(0);
        assert_eq!(presented(&replay).markers[0].position.x, 0.0);
    }

    #[test]
    fn playback_blends_toward_the_next_frame() {
        let mut replay = moving_track();
        replay.playing = true;
        replay.advance(0.5);
        let x = presented(&replay).markers[0].position.x;
        assert!((x - 50.0).abs() < 1e-3, "expected the midpoint, got {x}");
    }

    /// An entity that jumps faster than a teleport holds its recorded pose
    /// instead of sliding through everything between; one that moves
    /// normally in the same frames still blends.
    #[test]
    fn a_teleport_is_not_interpolated() {
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [], "tracks": [
                {"track_id": "jump", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
                {"track_id": "walk", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "truths": [], "nodes": [], "tracks": [
                {"track_id": "jump", "position": [5000.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
                {"track_id": "walk", "position": [10.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
        ]))));
        replay.seek_time(0.5);
        let markers = presented(&replay);
        assert_eq!(markers.markers[0].position.x, 0.0, "the teleport slid");
        assert!((markers.markers[1].position.x - 5.0).abs() < 1e-4);

        // A teleport is a speed: 50 m in 0.1 s is 500 m/s, though 50 m is
        // short.
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [], "tracks": [
                {"track_id": "hop", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 0.1, "truths": [], "nodes": [], "tracks": [
                {"track_id": "hop", "position": [50.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
        ]))));
        replay.seek_time(0.05);
        assert_eq!(
            presented(&replay).markers[0].position.x,
            0.0,
            "a 500 m/s hop slid"
        );
    }

    /// Nothing blends across a recording gap: the frame before it is shown
    /// as recorded all the way across.
    #[test]
    fn a_recording_gap_is_not_interpolated() {
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [], "tracks": [{"track_id": "t", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "truths": [], "nodes": [], "tracks": [{"track_id": "t", "position": [1.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 2.0, "truths": [], "nodes": [], "tracks": [{"track_id": "t", "position": [2.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 60.0, "truths": [], "nodes": [], "tracks": [{"track_id": "t", "position": [60.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
        ]))));
        replay.seek_time(30.0);
        assert_eq!(presented(&replay).markers[0].position.x, 2.0);
        replay.seek_time(1.5);
        assert!((presented(&replay).markers[0].position.x - 1.5).abs() < 1e-5);
    }

    #[test]
    fn interpolation_never_overshoots_the_next_frame() {
        let mut replay = moving_track();
        replay.playing = true;
        replay.seek_time(1e6);
        assert_eq!(presented(&replay).markers[0].position.x, 100.0);
    }

    #[test]
    fn an_entity_missing_from_the_next_frame_holds_its_pose() {
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [],
             "tracks": [{"track_id": "gone", "position": [7.0, 0.0, 0.0], "velocity": [1.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "truths": [], "nodes": [], "tracks": []},
        ]))));
        replay.playing = true;
        replay.advance(0.5);
        let markers = presented(&replay);
        assert_eq!(markers.markers[0].position.x, 7.0);
        assert_eq!(markers.markers[0].velocity.x, 1.0);
    }

    /// A track is keyed by its target's id, so it shares a label with that
    /// truth. The keys must still differ, and each must blend toward its own
    /// kind's next pose.
    #[test]
    fn a_track_and_its_truth_share_a_label_but_not_a_key() {
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "nodes": [],
             "tracks": [{"track_id": "asset-a", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}],
             "truths": [{"target_id": "asset-a", "position": [10.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "nodes": [],
             "tracks": [{"track_id": "asset-a", "position": [2.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}],
             "truths": [{"target_id": "asset-a", "position": [30.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
        ]))));
        replay.playing = true;
        replay.advance(0.5);
        let markers = presented(&replay);
        let (track, truth) = (&markers.markers[0], &markers.markers[1]);
        assert_eq!(&*track.label, &*truth.label);
        assert_ne!(track.key, truth.key);
        assert_eq!(track.key.id(), truth.key.id(), "one interned id, two kinds");
        assert!((track.position.x - 1.0).abs() < 1e-5);
        assert!((truth.position.x - 20.0).abs() < 1e-5);
        assert_eq!(markers.get(truth.key).unwrap().position.x, truth.position.x);
    }

    fn walking(frames: usize) -> ReplayState {
        let frames: Vec<serde_json::Value> = (0..frames)
            .map(|f| {
                let mut nodes = vec![serde_json::json!({"node_id": "drone-a",
                    "position": [f as f64, 0.0, 0.0], "velocity": [1.0, 0.0, 0.0], "is_mobile": true})];
                // A second drone that exists only on even frames.
                if f % 2 == 0 {
                    nodes.push(serde_json::json!({"node_id": "drone-b",
                        "position": [0.0, f as f64, 0.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}));
                }
                serde_json::json!({"timestamp_s": f as f64, "tracks": [], "truths": [], "nodes": nodes})
            })
            .collect();
        ReplayState::new(Some(document(serde_json::Value::Array(frames))))
    }

    /// Keys survive seeking, forward and back, within an epoch.
    #[test]
    fn keys_are_stable_across_seeks() {
        let mut replay = walking(60);
        let mut markers = CurrentRuntimeMarkers::default();
        replay.step_to(10);
        markers.apply(&replay);
        let a = markers.key_of(MarkerKind::Node, "drone-a").unwrap();
        for frame in [50, 3, 59, 0, 22] {
            replay.step_to(frame);
            markers.apply(&replay);
            assert_eq!(markers.key_of(MarkerKind::Node, "drone-a"), Some(a));
            assert_eq!(markers.markers[0].key, a);
            assert_eq!(markers.markers[0].position.x, frame as f32);
        }
        assert_eq!(markers.ids().len(), 2, "each id interned once");
    }

    /// A new epoch voids every key: a stale one must not alias whatever the
    /// new table gives the same number.
    #[test]
    fn a_new_epoch_starts_a_new_id_table() {
        let mut replay = walking(4);
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(&replay);
        let old_epoch = markers.epoch();
        replay.begin_epoch(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "tracks": [], "truths": [],
             "nodes": [{"node_id": "other", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": false}]}
        ]))));
        let change = markers.apply(&replay);
        assert!(change.epoch && change.membership);
        assert_ne!(markers.epoch(), old_epoch);
        assert_eq!(markers.key_of(MarkerKind::Node, "drone-a"), None);
        assert_eq!(markers.ids().len(), 1);
    }

    #[test]
    fn trails_are_the_recorded_positions_of_the_window() {
        let mut replay = walking(60);
        let mut markers = CurrentRuntimeMarkers::default();
        replay.step_to(40);
        markers.apply(&replay);
        let a = markers
            .index_of(markers.key_of(MarkerKind::Node, "drone-a").unwrap())
            .unwrap();
        let trail: Vec<f32> = markers.trail(a, TRAIL_MAX_FRAMES).map(|p| p.x).collect();
        assert_eq!(trail, (9..=40).map(|f| f as f32).collect::<Vec<_>>());
        let short: Vec<f32> = markers.trail(a, 16).map(|p| p.x).collect();
        assert_eq!(short, (25..=40).map(|f| f as f32).collect::<Vec<_>>());
        // drone-b is present on even frames only, and its trail says so.
        let b = markers
            .index_of(markers.key_of(MarkerKind::Node, "drone-b").unwrap())
            .unwrap();
        let trail: Vec<f32> = markers.trail(b, 16).map(|p| p.y).collect();
        assert_eq!(
            trail,
            (26..=40).step_by(2).map(|f| f as f32).collect::<Vec<_>>()
        );
    }

    /// Revisions move only when their inputs do; a paused frame changes
    /// nothing.
    #[test]
    fn revisions_track_membership_and_poses_separately() {
        let mut replay = walking(8);
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(&replay);
        let (m0, p0) = (markers.membership_revision(), markers.pose_revision());

        assert_eq!(markers.apply(&replay), PresentationChange::default());
        assert_eq!(
            (markers.membership_revision(), markers.pose_revision()),
            (m0, p0)
        );

        // Frame 0 -> 1 drops drone-b: membership changes.
        replay.step_to(1);
        let change = markers.apply(&replay);
        assert!(change.membership && change.poses);
        assert!(markers.membership_revision() > m0);

        // Blending within frame 1 moves poses, not membership, and reads
        // nothing new from the frame.
        let m1 = markers.membership_revision();
        replay.playing = true;
        replay.advance(0.25);
        let change = markers.apply(&replay);
        assert!(change.poses && !change.membership && !change.frame);
        assert_eq!(markers.membership_revision(), m1);

        // Frame 1 -> 3: the same entities, but a different frame.
        replay.step_to(3);
        let change = markers.apply(&replay);
        assert!(change.frame && !change.membership);
    }

    /// Blending in place reuses the marker storage: no allocation on a frame
    /// where only the interpolation alpha moved.
    #[test]
    fn blending_reuses_the_marker_storage() {
        let mut replay = walking(8);
        replay.playing = true;
        let mut markers = CurrentRuntimeMarkers::default();
        replay.advance(0.1);
        markers.apply(&replay);
        let storage = markers.markers.as_ptr();
        replay.advance(0.1);
        markers.apply(&replay);
        assert_eq!(markers.markers.as_ptr(), storage);
    }

    /// A duplicated id is malformed, but the answer is pinned anyway: the
    /// first occurrence wins, as the linear search it replaced found it.
    #[test]
    fn a_duplicated_id_blends_toward_its_first_occurrence() {
        let mut replay = ReplayState::new(Some(document(serde_json::json!([
            {"timestamp_s": 0.0, "truths": [], "nodes": [],
             "tracks": [{"track_id": "t", "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
            {"timestamp_s": 1.0, "truths": [], "nodes": [],
             "tracks": [{"track_id": "t", "position": [10.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
                        {"track_id": "t", "position": [99.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}]},
        ]))));
        replay.playing = true;
        replay.advance(0.5);
        assert!((presented(&replay).markers[0].position.x - 5.0).abs() < 1e-5);
    }

    /// Live history is trimmed from the front, which shifts the frame index
    /// under an unchanged frame. The source is the frame, not the index.
    #[test]
    fn trimming_live_history_rebuilds_rather_than_reusing_a_stale_index() {
        let mut replay = walking(10);
        replay.step_to(9);
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(&replay);
        let key = markers.markers[0].key;
        replay.drain_front(4);
        assert_eq!(replay.frame_index, 5);
        markers.apply(&replay);
        assert_eq!(markers.markers[0].key, key, "same epoch, same key");
        assert_eq!(markers.markers[0].position.x, 9.0);
        let trail: Vec<f32> = markers.trail(0, TRAIL_MAX_FRAMES).map(|p| p.x).collect();
        assert_eq!(trail, (4..=9).map(|f| f as f32).collect::<Vec<_>>());
    }
}
