//! Shared runtime values. No simulation, transport or renderer dependencies.
use nalgebra::{SMatrix, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub type Matrix6 = SMatrix<f64, 6, 6>;

#[derive(Clone, Debug, PartialEq)]
pub struct NodeState {
    pub node_id: String,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub is_mobile: bool,
    pub timestamp_s: f64,
    pub health: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BearingObservation {
    pub node_id: String,
    pub target_id: String,
    pub origin: Vector3<f64>,
    pub direction: Vector3<f64>,
    pub bearing_std_rad: f64,
    pub timestamp_s: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObservationRejection {
    pub node_id: String,
    pub target_id: String,
    pub timestamp_s: f64,
    pub reason: String,
    pub detail: String,
    pub origin: Option<Vector3<f64>>,
    pub attempted_point: Option<Vector3<f64>>,
    pub closest_point: Option<Vector3<f64>>,
    pub blocker_type: String,
    pub first_hit_range_m: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TruthState {
    pub target_id: String,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub timestamp_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackState {
    pub track_id: String,
    pub timestamp_s: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub covariance: Matrix6,
    pub measurement_std_m: f64,
    pub update_count: u32,
    pub stale_steps: u32,
    pub lifecycle_state: Option<String>,
    pub quality_score: Option<f64>,
    /// IMM CV-model weight in [0, 1]. None when not using an IMM filter.
    pub mode_probability_cv: Option<f64>,
    /// Node IDs whose observations contributed to the most recent update.
    /// Shared with the owning track so per-frame snapshots don't deep-clone.
    pub contributing_node_ids: std::sync::Arc<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlatformMetrics {
    pub mean_error_m: Option<f64>,
    pub max_error_m: Option<f64>,
    pub active_track_count: u32,
    pub observation_count: u32,
    pub accepted_observation_count: u32,
    pub rejected_observation_count: u32,
    pub mean_measurement_std_m: Option<f64>,
    pub track_errors_m: BTreeMap<String, f64>,
    pub rejection_counts: BTreeMap<String, u32>,
    pub accepted_observations_by_target: BTreeMap<String, u32>,
    pub rejected_observations_by_target: BTreeMap<String, u32>,
}

/// Operator-supplied annotation for a target.
///
/// Carried through the runtime rather than around it: before P4.0 the server
/// grafted the wire copies onto the ingest response, so `LatestFrame` -- which
/// rebuilds from `PlatformFrame` -- returned an empty list for the same frame.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetMetadata {
    pub target_id: String,
    pub operator_label: String,
    pub target_type: String,
    pub priority: i32,
    pub behavior: String,
    pub status: String,
    pub first_seen_s: f64,
    pub last_seen_s: f64,
    pub confidence: f64,
    pub notes: String,
}

/// A safety-envelope event raised against one subject during a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct SafetyEvent {
    pub event_id: String,
    pub subject_id: String,
    pub timestamp_s: f64,
    pub state: String,
    pub violations: Vec<String>,
    pub command_clamped: bool,
    pub blocked: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlatformFrame {
    pub timestamp_s: f64,
    pub nodes: Vec<NodeState>,
    pub observations: Vec<BearingObservation>,
    pub rejected_observations: Vec<ObservationRejection>,
    pub tracks: Vec<TrackState>,
    pub truths: Vec<TruthState>,
    pub metrics: PlatformMetrics,
    pub generation_rejections: Vec<ObservationRejection>,
    pub target_metadata: Vec<TargetMetadata>,
    pub safety_events: Vec<SafetyEvent>,
}

/// Axis-aligned horizontal extent of a footprint, metres.
///
/// Vertical extent belongs to the terrain or obstacle that owns this footprint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds2d {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

impl Bounds2d {
    pub fn new(x_min_m: f64, x_max_m: f64, y_min_m: f64, y_max_m: f64) -> Self {
        Self {
            x_min_m,
            x_max_m,
            y_min_m,
            y_max_m,
        }
    }

    #[inline]
    pub fn contains_xy(&self, x_m: f64, y_m: f64) -> bool {
        x_m >= self.x_min_m && x_m <= self.x_max_m && y_m >= self.y_min_m && y_m <= self.y_max_m
    }

    /// Widen by `margin_m` on every side. Used to pre-filter candidates before
    /// running the exact test.
    pub fn expanded(&self, margin_m: f64) -> Self {
        Self {
            x_min_m: self.x_min_m - margin_m,
            x_max_m: self.x_max_m + margin_m,
            y_min_m: self.y_min_m - margin_m,
            y_max_m: self.y_max_m + margin_m,
        }
    }
}
