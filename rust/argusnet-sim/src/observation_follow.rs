//! `ObservationTriggeredFollowController`: search until you see it, then follow.
//!
//! Forty-five lines in the reference, and the only controller whose mode is
//! changed by something *outside* stage 1. It samples one of two inner
//! trajectories -- search while disengaged, follow once engaged -- and it is
//! engaged by the observation batch: the first tick on which this node's
//! sensor accepted an observation of its preferred target.
//!
//! # The engagement is a next-tick effect
//!
//! The reference calls `update_from_frame` after fusion, at the end of the
//! tick; the new mode is first *flown* on the following tick's stage 1. That
//! is §4's "decisions that take effect on the next tick", and it is why the
//! engaged flag and the last sample are checkpoint state: a checkpoint taken
//! between the observation and the next sample must carry the decision
//! across.
//!
//! # The handover seeds the follow controller from the last *search* sample
//!
//! Not from wherever the platform is when the follow first runs -- from what
//! this controller last returned, at the timestamp it returned it. That is
//! how the follow controller avoids the "unseeded controller never flies"
//! finding in [`crate::follow_path`]: the search trajectory hands over a real
//! position and heading.
//!
//! Two quirks preserved rather than corrected:
//!
//! * **Engagement without a sample does not seed.** If an observation arrives
//!   before this controller has ever been sampled, it engages anyway, and the
//!   follow controller starts unseeded.
//! * **Reset resets the follow half only.** The reference guards on
//!   `hasattr(self.follow_trajectory, "reset_state")` and never touches the
//!   search trajectory, which is a pure function of time in every scenario
//!   that builds one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::follow_path::FollowPathController;

/// The two things the wrapper does to its follow half besides sampling it.
pub trait Seedable {
    fn seed(&mut self, position_m: [f64; 3], velocity_mps: [f64; 3], timestamp_s: f64);
    fn reset_state(&mut self);
}

impl Seedable for FollowPathController {
    fn seed(&mut self, position_m: [f64; 3], velocity_mps: [f64; 3], timestamp_s: f64) {
        FollowPathController::seed(self, position_m, velocity_mps, timestamp_s);
    }

    fn reset_state(&mut self) {
        FollowPathController::reset_state(self);
    }
}

/// Everything this wrapper carries across ticks.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EngagementState {
    pub engaged: bool,
    /// `(timestamp_s, position, velocity)` of the last sample returned,
    /// whichever half produced it.
    pub last_sample: Option<(f64, [f64; 3], [f64; 3])>,
}

/// Accepted-observation counts keyed `(node_id, target_id)` -- the shape of
/// `GenerationTally::accepted_by_node_target` and of the reference's
/// `ObservationBatch.accepted_by_node_target`.
pub type AcceptedByNodeTarget = BTreeMap<(String, String), u64>;

/// Search with `S`, follow with `F`.
///
/// Generic over both halves and sampled through closures, because the
/// halves need different per-tick context (the search lane reads terrain and
/// obstacles, the follow controller also reads line of sight, zones and the
/// target) and the wrapper's job is to choose between them, not to know
/// what either needs.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationTriggeredFollow<S, F> {
    node_id: String,
    preferred_target_id: String,
    search: S,
    follow: F,
    state: EngagementState,
}

impl<S, F: Seedable> ObservationTriggeredFollow<S, F> {
    pub fn new(
        node_id: impl Into<String>,
        preferred_target_id: impl Into<String>,
        search: S,
        follow: F,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            preferred_target_id: preferred_target_id.into(),
            search,
            follow,
            state: EngagementState::default(),
        }
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn preferred_target_id(&self) -> &str {
        &self.preferred_target_id
    }

    pub fn is_engaged(&self) -> bool {
        self.state.engaged
    }

    pub fn search(&self) -> &S {
        &self.search
    }

    pub fn follow(&self) -> &F {
        &self.follow
    }

    pub fn follow_mut(&mut self) -> &mut F {
        &mut self.follow
    }

    pub fn engagement(&self) -> &EngagementState {
        &self.state
    }

    /// Take a checkpoint's engagement back. The follow half's own state is
    /// restored separately, by whoever owns its type.
    pub fn restore_engagement(&mut self, state: EngagementState) {
        self.state = state;
    }

    /// One sample, from whichever half is active.
    pub fn sample(
        &mut self,
        timestamp_s: f64,
        sample_search: impl FnOnce(&mut S, f64) -> ([f64; 3], [f64; 3]),
        sample_follow: impl FnOnce(&mut F, f64) -> ([f64; 3], [f64; 3]),
    ) -> ([f64; 3], [f64; 3]) {
        let (position, velocity) = if self.state.engaged {
            sample_follow(&mut self.follow, timestamp_s)
        } else {
            sample_search(&mut self.search, timestamp_s)
        };
        self.state.last_sample = Some((timestamp_s, position, velocity));
        (position, velocity)
    }

    /// `update_from_frame`. Returns whether this call engaged the controller.
    ///
    /// One-way: once engaged, nothing here disengages it, and a later batch
    /// is ignored without being looked at.
    pub fn update_from_observations(&mut self, accepted: &AcceptedByNodeTarget) -> bool {
        if self.state.engaged {
            return false;
        }
        let seen = accepted
            .get(&(self.node_id.clone(), self.preferred_target_id.clone()))
            .copied()
            .unwrap_or(0)
            > 0;
        if !seen {
            return false;
        }
        if let Some((timestamp_s, position, velocity)) = self.state.last_sample {
            self.follow.seed(position, velocity, timestamp_s);
        }
        self.state.engaged = true;
        true
    }

    /// Back to searching. Resets the follow half and not the search half --
    /// see the module comment.
    pub fn reset_state(&mut self) {
        self.state = EngagementState::default();
        self.follow.reset_state();
    }
}
