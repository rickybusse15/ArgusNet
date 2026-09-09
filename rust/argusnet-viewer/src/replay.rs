//! Bevy integration for the renderer-independent replay model.

use std::ops::{Deref, DerefMut};

use bevy::prelude::Resource;

pub use argusnet_replay::*;

/// Viewer resource wrapper around the renderer-independent replay state.
///
/// Keeping the Bevy derive here lets `argusnet-replay` remain usable without
/// pulling an ECS or renderer into downstream crates.
#[derive(Debug, Clone, Default, Resource)]
pub struct ReplayState(argusnet_replay::ReplayState);

impl ReplayState {
    pub fn new(document: Option<ReplayDocument>) -> Self {
        Self(argusnet_replay::ReplayState::new(document))
    }

    pub fn step_to(&mut self, frame_index: usize) {
        self.0.step_to(frame_index);
    }
}

impl Deref for ReplayState {
    type Target = argusnet_replay::ReplayState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ReplayState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
