//! Safety constraint engine for the Smart Trajectory Tracker.
//!
//! This crate provides:
//! - [`limits`]: `DronePhysicalLimits` — complete physical envelope for one drone platform.
//! - [`validator`]: constraint validation pipeline that checks a commanded state against limits.
//! - [`monitor`]: stateful `SafetyMonitor` that tracks per-drone safety state and escalates violations.

pub mod limits;
pub mod monitor;
pub mod validator;
