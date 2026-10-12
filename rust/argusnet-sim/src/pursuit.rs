//! Pursuit over the run's own fused tracks.
//!
//! The legacy follow controllers steer off the target's true trajectory
//! ([`crate::controller::PursuitOracle`], answered from the truth table). In a
//! run that starts with no world knowledge that is the one remaining way for
//! the platforms to know where a target is without having sensed it.
//! [`TrackPursuit`] answers from what fusion actually produced instead:
//!
//! - **A track.** The latest fused estimate of the target (its track id is the
//!   target id), extrapolated at its velocity from the track's timestamp to the
//!   time asked. Tracks arrive late by the fusion lag (B4), so extrapolation
//!   is the normal case, not the exception. It is capped at
//!   [`TrackPursuit::max_extrapolation_s`]; past that the estimate is held
//!   where the extrapolation stopped, with zero velocity, because a coasting
//!   track's velocity is not worth chasing further.
//! - **A dropped track.** The tracker deletes lost tracks. The last estimate
//!   is kept (and checkpointed), so a follow drone goes to where the target
//!   was last believed to be rather than nowhere.
//! - **No track yet.** The tasking cue: where the target was when the run
//!   started, held still. A follow drone is tasked with *some* starting point
//!   (an operator's report), and this is the only truth the oracle uses.
//!
//! The run driver hands each committed tick's tracks to the oracle
//! ([`crate::scenario::AssembledRun::commit_tick`]); stage 1 runs before
//! fusion, so a tick pursues the previous tick's tracks. Handing them over
//! between ticks, rather than reading the world inside stage 1, means the
//! oracle's checkpoint already holds them and a resumed run pursues exactly
//! what an uninterrupted one would.

use std::cell::Cell;
use std::collections::BTreeMap;

use argusnet_types::TrackState;
use serde::{Deserialize, Serialize};

use crate::controller::PursuitOracle;

/// How far a track is extrapolated before it is held, seconds.
pub const DEFAULT_MAX_EXTRAPOLATION_S: f64 = 3.0;

/// A target's last fused estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub timestamp_s: f64,
}

/// Where each target is believed to be, from fused tracks and tasking cues.
#[derive(Debug, Clone)]
pub struct TrackPursuit {
    cues: BTreeMap<String, [f64; 3]>,
    latest: BTreeMap<String, Estimate>,
    pub max_extrapolation_s: f64,
    /// Reads answered from a track, and from a cue, since the last stage
    /// note. Counted through `&self`, since the controllers read the oracle
    /// while it is lent to them.
    from_tracks: Cell<u64>,
    from_cues: Cell<u64>,
}

impl TrackPursuit {
    /// `cues`: each target's position at the start of the run.
    pub fn new(cues: impl IntoIterator<Item = (String, [f64; 3])>) -> Self {
        Self {
            cues: cues.into_iter().collect(),
            latest: BTreeMap::new(),
            max_extrapolation_s: DEFAULT_MAX_EXTRAPOLATION_S,
            from_tracks: Cell::new(0),
            from_cues: Cell::new(0),
        }
    }

    /// The estimate held for `target_id`, if a track has ever been seen.
    pub fn estimate(&self, target_id: &str) -> Option<&Estimate> {
        self.latest.get(target_id)
    }

    fn answer(&self, target_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3], bool)> {
        if let Some(e) = self.latest.get(target_id) {
            let age = (timestamp_s - e.timestamp_s).max(0.0);
            let dt = age.min(self.max_extrapolation_s);
            let position = std::array::from_fn(|i| e.position[i] + e.velocity[i] * dt);
            let velocity = if age <= self.max_extrapolation_s {
                e.velocity
            } else {
                [0.0; 3]
            };
            return Some((position, velocity, true));
        }
        self.cues.get(target_id).map(|cue| (*cue, [0.0; 3], false))
    }
}

impl PursuitOracle for TrackPursuit {
    fn target_state(&self, target_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        let (p, v, from_track) = self.answer(target_id, timestamp_s)?;
        let counter = if from_track {
            &self.from_tracks
        } else {
            &self.from_cues
        };
        counter.set(counter.get() + 1);
        Some((p, v))
    }

    fn observe_tracks(&mut self, tracks: &[TrackState]) {
        for track in tracks {
            let estimate = Estimate {
                position: [track.position.x, track.position.y, track.position.z],
                velocity: [track.velocity.x, track.velocity.y, track.velocity.z],
                timestamp_s: track.timestamp_s,
            };
            match self.latest.get(&track.track_id) {
                Some(held) if held.timestamp_s > estimate.timestamp_s => {}
                _ => {
                    self.latest.insert(track.track_id.clone(), estimate);
                }
            }
        }
    }

    fn take_note(&mut self) -> Option<String> {
        let (tracks, cues) = (self.from_tracks.take(), self.from_cues.take());
        (tracks + cues > 0).then(|| {
            format!(
                "pursuit read {tracks} fused-track estimates and {cues} tasking cues (no truth)"
            )
        })
    }

    fn reads_truth(&self) -> bool {
        false
    }

    fn reset(&mut self) {
        self.latest.clear();
        self.from_tracks.set(0);
        self.from_cues.set(0);
    }

    fn checkpoint(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(&self.latest).expect("estimates serialise"))
    }

    fn restore(&mut self, state: &serde_json::Value) -> Result<(), String> {
        self.latest = serde_json::from_value(state.clone()).map_err(|error| error.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_types::Vector3;

    fn track(id: &str, t: f64, p: [f64; 3], v: [f64; 3]) -> TrackState {
        TrackState {
            track_id: id.into(),
            timestamp_s: t,
            position: Vector3::new(p[0], p[1], p[2]),
            velocity: Vector3::new(v[0], v[1], v[2]),
            covariance: Default::default(),
            measurement_std_m: 1.0,
            update_count: 3,
            stale_steps: 0,
            lifecycle_state: None,
            quality_score: None,
            mode_probability_cv: None,
            contributing_node_ids: Default::default(),
        }
    }

    #[test]
    fn before_any_track_the_tasking_cue_is_held_still() {
        let pursuit = TrackPursuit::new([("fox".to_string(), [10.0, 20.0, 30.0])]);
        assert_eq!(
            pursuit.target_state("fox", 5.0),
            Some(([10.0, 20.0, 30.0], [0.0; 3]))
        );
        assert_eq!(pursuit.target_state("lynx", 5.0), None);
    }

    #[test]
    fn a_late_track_is_extrapolated_to_the_time_asked_then_held() {
        let mut pursuit = TrackPursuit::new([("fox".to_string(), [0.0; 3])]);
        pursuit.observe_tracks(&[track("fox", 10.0, [100.0, 0.0, 50.0], [2.0, -1.0, 0.0])]);
        let (p, v) = pursuit.target_state("fox", 11.5).unwrap();
        assert_eq!(p, [103.0, -1.5, 50.0]);
        assert_eq!(v, [2.0, -1.0, 0.0]);
        // Past the cap the estimate stops where the extrapolation stopped.
        let (p, v) = pursuit.target_state("fox", 30.0).unwrap();
        assert_eq!(p, [106.0, -3.0, 50.0]);
        assert_eq!(v, [0.0; 3]);
    }

    #[test]
    fn a_dropped_track_is_remembered_and_an_older_one_never_replaces_a_newer() {
        let mut pursuit = TrackPursuit::new([("fox".to_string(), [0.0; 3])]);
        pursuit.observe_tracks(&[track("fox", 10.0, [5.0, 5.0, 5.0], [0.0; 3])]);
        pursuit.observe_tracks(&[]);
        assert_eq!(
            pursuit.target_state("fox", 10.0).unwrap().0,
            [5.0, 5.0, 5.0]
        );
        pursuit.observe_tracks(&[track("fox", 9.0, [1.0, 1.0, 1.0], [0.0; 3])]);
        assert_eq!(
            pursuit.target_state("fox", 10.0).unwrap().0,
            [5.0, 5.0, 5.0]
        );
    }

    #[test]
    fn checkpoint_restore_and_reset() {
        let mut pursuit = TrackPursuit::new([("fox".to_string(), [0.0; 3])]);
        pursuit.observe_tracks(&[track("fox", 10.0, [5.0, 5.0, 5.0], [1.0, 0.0, 0.0])]);
        let saved = pursuit.checkpoint().unwrap();
        let mut other = TrackPursuit::new([("fox".to_string(), [0.0; 3])]);
        other.restore(&saved).unwrap();
        assert_eq!(
            other.target_state("fox", 11.0),
            pursuit.target_state("fox", 11.0)
        );
        other.reset();
        assert_eq!(other.target_state("fox", 11.0), Some(([0.0; 3], [0.0; 3])));
    }

    #[test]
    fn the_note_counts_where_answers_came_from() {
        let mut pursuit = TrackPursuit::new([
            ("fox".to_string(), [0.0; 3]),
            ("lynx".to_string(), [0.0; 3]),
        ]);
        pursuit.observe_tracks(&[track("fox", 1.0, [0.0; 3], [0.0; 3])]);
        pursuit.target_state("fox", 1.0);
        pursuit.target_state("fox", 1.0);
        pursuit.target_state("lynx", 1.0);
        let note = pursuit.take_note().unwrap();
        assert!(
            note.contains("2 fused-track") && note.contains("1 tasking"),
            "{note}"
        );
        assert_eq!(pursuit.take_note(), None);
    }
}
