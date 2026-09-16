//! The tick's mutable world, and truth as a capability.
//!
//! Until now [`TickContext`] carried a tick number, a timestamp, a seed and a
//! notes buffer — everything a stage needs to *describe* what it did and
//! nothing it needs to *do* anything. So no domain stage could be wired in at
//! all, and `commit_current` published `RuntimeSnapshot::empty` every tick.
//! This is the state the stages share.
//!
//! [`TickContext`]: crate::simulation::TickContext
//!
//! # One list, mutated in place, frozen once
//!
//! P4.8: *"Eliminate repeated dataclass-style list rebuilds; freeze once at the
//! snapshot boundary."* The reference rebuilds its `node_states` list **up to
//! seven times per tick** — battery, then frontier, then POI, then egress, then
//! deconfliction, then safety, then the frame — each one a full
//! `dataclasses.replace` over every platform. A faithful transcription would
//! preserve that shape and waste the migration.
//!
//! Here there is one `Vec<NodeState>`. Stages take `&mut` and edit the
//! platforms they are changing; nothing rebuilds the list. The single clone
//! happens at [`TickWorld::freeze`], which is stage 10 and nowhere else, and
//! [`TickWorld::freezes_this_tick`] counts them so "once" is a tested property
//! rather than an intention.
//!
//! # Truth is a capability, and the capability is a type
//!
//! `ARCHITECTURE_AND_PROTOCOLS.md` §3 makes truth a capability rather than a
//! convention. A comment asking stages not to read truth is worth nothing; a
//! stage that *cannot name* truth is worth something.
//!
//! So truth does not live on [`TickWorld`] at all. It reaches a stage only as
//! [`TickContext::truth`], which the simulation populates **only** for the
//! stages [`Stage::may_read_truth`] admits — the synthetic IMU/VIO stage and
//! the observation generator, which exist to sense the true world. Every other
//! handler sees `None`, whatever it does, because there is no other path.
//!
//! That also makes the snapshot's `truth_filtered` flag honest: the snapshot
//! is built from [`TickWorld`], which never held truth, so the flag is a
//! statement about the type rather than about a filtering step someone has to
//! remember to run.
//!
//! [`Stage::may_read_truth`]: crate::stage::Stage::may_read_truth
//! [`TickContext::truth`]: crate::simulation::TickContext::truth

use argusnet_types::{NodeState, TrackState};

/// Platform and track state for the tick in progress.
///
/// Holds no truth — see the module docs. Holds no terrain or obstacles either:
/// those are immutable for a run and belong to the world the stages are handed,
/// not to the per-tick state they mutate.
#[derive(Debug, Clone, Default)]
pub struct TickWorld {
    platforms: Vec<NodeState>,
    tracks: Vec<TrackState>,
    revision: u64,
    freezes_this_tick: u32,
}

impl TickWorld {
    pub fn new(platforms: Vec<NodeState>, tracks: Vec<TrackState>) -> Self {
        Self {
            platforms,
            tracks,
            revision: 0,
            freezes_this_tick: 0,
        }
    }

    pub fn platforms(&self) -> &[NodeState] {
        &self.platforms
    }

    pub fn tracks(&self) -> &[TrackState] {
        &self.tracks
    }

    /// Mobile platforms, in list order — the reference's `mobile_states`.
    ///
    /// Order is the fleet's declaration order and is load-bearing: §4 asks for
    /// "stable platform ordering" in deconfliction, and several reference
    /// stages break ties by position in this list.
    pub fn mobile_platforms(&self) -> impl Iterator<Item = &NodeState> {
        self.platforms.iter().filter(|p| p.is_mobile)
    }

    pub fn mobile_count(&self) -> usize {
        self.platforms.iter().filter(|p| p.is_mobile).count()
    }

    /// Edit every platform in place.
    ///
    /// This is the replacement for `dataclasses.replace` over the whole list:
    /// the caller mutates what it is changing and leaves the rest alone, so a
    /// stage that touches one field of one platform costs one field write
    /// rather than a rebuilt list.
    pub fn platforms_mut(&mut self) -> &mut [NodeState] {
        self.revision += 1;
        &mut self.platforms
    }

    /// Edit one platform by id, if it exists.
    ///
    /// Returns `None` rather than inserting: a stage naming a platform that is
    /// not in the fleet has a bug, and silently creating one would hide it.
    pub fn platform_mut(&mut self, node_id: &str) -> Option<&mut NodeState> {
        let found = self.platforms.iter_mut().find(|p| p.node_id == node_id);
        if found.is_some() {
            self.revision += 1;
        }
        found
    }

    pub fn platform(&self, node_id: &str) -> Option<&NodeState> {
        self.platforms.iter().find(|p| p.node_id == node_id)
    }

    /// Replace the track list. Tracks come back wholesale from the fuser, so
    /// unlike platforms there is nothing to edit in place.
    pub fn set_tracks(&mut self, tracks: Vec<TrackState>) {
        self.tracks = tracks;
        self.revision += 1;
    }

    /// Bumped on every mutation. Feeds `SnapshotHeader::world_revision`, so a
    /// consumer can tell two snapshots apart without diffing them.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// How many times this tick has frozen. Must be exactly one by the end of
    /// a committed tick.
    pub fn freezes_this_tick(&self) -> u32 {
        self.freezes_this_tick
    }

    /// Reset the freeze counter. Called by the simulation at the top of a tick.
    pub fn begin_tick(&mut self) {
        self.freezes_this_tick = 0;
    }

    /// The single clone, at the snapshot boundary.
    ///
    /// Everything before this point mutates; this is where the tick's state
    /// becomes an immutable value. Calling it twice in a tick is the defect
    /// P4.8 names, so it is counted rather than merely discouraged.
    pub fn freeze(&mut self) -> (Vec<NodeState>, Vec<TrackState>) {
        self.freezes_this_tick += 1;
        (self.platforms.clone(), self.tracks.clone())
    }

    /// Drop all state, keeping the revision counter.
    ///
    /// The revision keeps counting for the same reason the run's does: it
    /// orders snapshots a consumer may receive out of order, and restarting it
    /// would make a stale snapshot look newer than a current one.
    pub fn clear(&mut self) {
        self.platforms.clear();
        self.tracks.clear();
        self.revision += 1;
        self.freezes_this_tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_types::Vector3;

    fn node(id: &str, mobile: bool) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: mobile,
            timestamp_s: 0.0,
            health: 1.0,
        }
    }

    fn world() -> TickWorld {
        TickWorld::new(
            vec![
                node("drone-1", true),
                node("tower", false),
                node("drone-2", true),
            ],
            Vec::new(),
        )
    }

    #[test]
    fn a_platform_is_edited_in_place_not_rebuilt() {
        let mut w = world();
        let before = w.revision();
        w.platform_mut("drone-2").unwrap().health = 0.4;
        assert!(w.revision() > before);
        assert_eq!(w.platform("drone-2").unwrap().health, 0.4);
        // The others are untouched, which is the whole point: the reference
        // replaces every element of the list to change one field of one.
        assert_eq!(w.platform("drone-1").unwrap().health, 1.0);
        assert_eq!(w.platforms().len(), 3);
    }

    #[test]
    fn naming_a_platform_that_does_not_exist_is_none_not_an_insert() {
        let mut w = world();
        let before = w.revision();
        assert!(w.platform_mut("ghost").is_none());
        assert_eq!(w.platforms().len(), 3);
        assert_eq!(w.revision(), before, "a failed lookup is not a mutation");
    }

    #[test]
    fn mobile_platforms_keep_declaration_order() {
        let w = world();
        let ids: Vec<&str> = w.mobile_platforms().map(|p| p.node_id.as_str()).collect();
        assert_eq!(ids, vec!["drone-1", "drone-2"]);
        assert_eq!(w.mobile_count(), 2);
    }

    #[test]
    fn freezing_is_counted() {
        let mut w = world();
        w.begin_tick();
        assert_eq!(w.freezes_this_tick(), 0);
        let (platforms, tracks) = w.freeze();
        assert_eq!(w.freezes_this_tick(), 1);
        assert_eq!(platforms.len(), 3);
        assert!(tracks.is_empty());
        w.freeze();
        assert_eq!(w.freezes_this_tick(), 2, "a second freeze is visible");
        w.begin_tick();
        assert_eq!(w.freezes_this_tick(), 0);
    }

    #[test]
    fn a_frozen_snapshot_does_not_move_when_the_world_does() {
        let mut w = world();
        let (platforms, _) = w.freeze();
        w.platform_mut("drone-1").unwrap().health = 0.1;
        assert_eq!(platforms[0].health, 1.0, "the frozen copy is immutable");
        assert_eq!(w.platform("drone-1").unwrap().health, 0.1);
    }

    #[test]
    fn clearing_keeps_the_revision_climbing() {
        let mut w = world();
        w.platform_mut("drone-1").unwrap().health = 0.5;
        let before = w.revision();
        w.clear();
        assert!(w.platforms().is_empty());
        assert!(
            w.revision() > before,
            "a restarted revision would make a stale snapshot look newer"
        );
    }

    #[test]
    fn tracks_are_replaced_wholesale() {
        let mut w = world();
        assert!(w.tracks().is_empty());
        let before = w.revision();
        w.set_tracks(Vec::new());
        assert!(w.revision() > before);
    }
}
