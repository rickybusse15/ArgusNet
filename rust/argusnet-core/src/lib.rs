pub mod association;

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};

pub type Matrix6 = SMatrix<f64, 6, 6>;
pub type Vector6 = SVector<f64, 6>;

#[derive(Clone, Debug)]
pub struct NodeHealthTracker {
    pub node_id: String,
    pub first_seen_s: f64,
    pub last_seen_s: f64,
    pub accepted_count: u32,
    pub rejected_count: u32,
    pub total_observations: u32,
}

impl NodeHealthTracker {
    pub fn new(node_id: &str, timestamp_s: f64) -> Self {
        Self {
            node_id: node_id.to_string(),
            first_seen_s: timestamp_s,
            last_seen_s: timestamp_s,
            accepted_count: 0,
            rejected_count: 0,
            total_observations: 0,
        }
    }

    pub fn record_accepted(&mut self, timestamp_s: f64) {
        self.last_seen_s = timestamp_s;
        self.accepted_count += 1;
        self.total_observations += 1;
    }

    pub fn record_rejected(&mut self, timestamp_s: f64) {
        self.last_seen_s = timestamp_s;
        self.rejected_count += 1;
        self.total_observations += 1;
    }

    pub fn observation_rate_hz(&self) -> f64 {
        let elapsed = self.last_seen_s - self.first_seen_s;
        if elapsed > 0.0 {
            self.total_observations as f64 / elapsed
        } else {
            0.0
        }
    }

    pub fn health_score(&self, current_timestamp_s: f64, stale_threshold_s: f64) -> f64 {
        let staleness = current_timestamp_s - self.last_seen_s;
        if staleness > stale_threshold_s {
            0.0
        } else {
            let freshness = 1.0 - (staleness / stale_threshold_s).clamp(0.0, 1.0);
            let acceptance_ratio = if self.total_observations > 0 {
                self.accepted_count as f64 / self.total_observations as f64
            } else {
                0.0
            };
            freshness * 0.5 + acceptance_ratio * 0.5
        }
    }
}

#[derive(Clone, Debug)]
pub struct NodeHealthSnapshot {
    pub node_id: String,
    pub last_seen_s: f64,
    pub observation_rate_hz: f64,
    pub accepted_count: u32,
    pub rejected_count: u32,
    pub health_score: f64,
}

pub const REJECT_UNKNOWN_NODE: &str = "unknown_node";
pub const REJECT_INVALID_TARGET: &str = "invalid_target_id";
pub const REJECT_INVALID_DIRECTION: &str = "invalid_direction";
pub const REJECT_LOW_CONFIDENCE: &str = "low_confidence";
pub const REJECT_INVALID_BEARING_STD: &str = "invalid_bearing_std";
pub const REJECT_EXCESS_BEARING_STD: &str = "bearing_noise_too_high";
pub const REJECT_TIMESTAMP_SKEW: &str = "timestamp_skew";
pub const REJECT_DUPLICATE_NODE: &str = "duplicate_node_observation";
pub const REJECT_INSUFFICIENT_CLUSTER: &str = "insufficient_cluster_observations";
pub const REJECT_WEAK_GEOMETRY: &str = "weak_intersection_geometry";
pub const REJECT_FUSION_FAILURE: &str = "fusion_failure";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AssociationMode {
    Labeled,
    GNN,
    JPDA,
}

impl Default for AssociationMode {
    fn default() -> Self {
        Self::Labeled
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackerConfig {
    pub min_observations: u32,
    pub max_stale_steps: u32,
    pub retain_history: bool,
    pub min_confidence: f64,
    pub max_bearing_std_rad: f64,
    pub max_timestamp_skew_s: f64,
    pub min_intersection_angle_deg: f64,
    #[serde(default)]
    pub data_association_mode: AssociationMode,
    #[serde(default = "default_cv_process_accel_std")]
    pub cv_process_accel_std: f64,
    #[serde(default = "default_ct_process_accel_std")]
    pub ct_process_accel_std: f64,
    #[serde(default = "default_ct_turn_rate_std")]
    pub ct_turn_rate_std: f64,
    #[serde(default = "default_innovation_window")]
    pub innovation_window: u32,
    #[serde(default = "default_innovation_scale_factor")]
    pub innovation_scale_factor: f64,
    #[serde(default = "default_innovation_max_scale")]
    pub innovation_max_scale: f64,
    #[serde(default)]
    pub adaptive_measurement_noise: bool,
    #[serde(default = "default_chi_squared_gate_threshold")]
    pub chi_squared_gate_threshold: f64,
    #[serde(default = "default_cluster_distance_threshold_m")]
    pub cluster_distance_threshold_m: f64,
    #[serde(default = "default_near_parallel_rejection_angle_deg")]
    pub near_parallel_rejection_angle_deg: f64,
    #[serde(default = "default_confirmation_m")]
    pub confirmation_m: u32,
    #[serde(default = "default_confirmation_n")]
    pub confirmation_n: u32,
    #[serde(default = "default_max_coast_frames")]
    pub max_coast_frames: u32,
    #[serde(default = "default_max_coast_seconds")]
    pub max_coast_seconds: f64,
    #[serde(default = "default_min_quality_score")]
    pub min_quality_score: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_observations: 2,
            max_stale_steps: 8,
            retain_history: false,
            min_confidence: 0.15,
            max_bearing_std_rad: 0.08,
            max_timestamp_skew_s: 1.5,
            min_intersection_angle_deg: 2.5,
            data_association_mode: AssociationMode::Labeled,
            cv_process_accel_std: default_cv_process_accel_std(),
            ct_process_accel_std: default_ct_process_accel_std(),
            ct_turn_rate_std: default_ct_turn_rate_std(),
            innovation_window: default_innovation_window(),
            innovation_scale_factor: default_innovation_scale_factor(),
            innovation_max_scale: default_innovation_max_scale(),
            adaptive_measurement_noise: false,
            chi_squared_gate_threshold: default_chi_squared_gate_threshold(),
            cluster_distance_threshold_m: default_cluster_distance_threshold_m(),
            near_parallel_rejection_angle_deg: default_near_parallel_rejection_angle_deg(),
            confirmation_m: default_confirmation_m(),
            confirmation_n: default_confirmation_n(),
            max_coast_frames: default_max_coast_frames(),
            max_coast_seconds: default_max_coast_seconds(),
            min_quality_score: default_min_quality_score(),
        }
    }
}

const fn default_cv_process_accel_std() -> f64 {
    3.0
}

const fn default_ct_process_accel_std() -> f64 {
    8.0
}

const fn default_ct_turn_rate_std() -> f64 {
    0.1
}

const fn default_innovation_window() -> u32 {
    5
}

const fn default_innovation_scale_factor() -> f64 {
    1.5
}

const fn default_innovation_max_scale() -> f64 {
    4.0
}

const fn default_chi_squared_gate_threshold() -> f64 {
    16.0
}

const fn default_cluster_distance_threshold_m() -> f64 {
    200.0
}

const fn default_near_parallel_rejection_angle_deg() -> f64 {
    2.5
}

const fn default_confirmation_m() -> u32 {
    3
}

const fn default_confirmation_n() -> u32 {
    5
}

const fn default_max_coast_frames() -> u32 {
    10
}

const fn default_max_coast_seconds() -> f64 {
    5.0
}

const fn default_min_quality_score() -> f64 {
    0.1
}

impl TrackerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.min_observations < 2 {
            return Err("min_observations must be at least 2.".to_string());
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err("min_confidence must be within [0.0, 1.0].".to_string());
        }
        if !self.max_bearing_std_rad.is_finite() || self.max_bearing_std_rad <= 0.0 {
            return Err("max_bearing_std_rad must be finite and > 0.".to_string());
        }
        if !self.max_timestamp_skew_s.is_finite() || self.max_timestamp_skew_s < 0.0 {
            return Err("max_timestamp_skew_s must be finite and >= 0.".to_string());
        }
        if !self.min_intersection_angle_deg.is_finite() || self.min_intersection_angle_deg <= 0.0 {
            return Err("min_intersection_angle_deg must be finite and > 0.".to_string());
        }
        for (name, value) in [
            ("cv_process_accel_std", self.cv_process_accel_std),
            ("ct_process_accel_std", self.ct_process_accel_std),
            ("ct_turn_rate_std", self.ct_turn_rate_std),
            ("innovation_scale_factor", self.innovation_scale_factor),
            ("innovation_max_scale", self.innovation_max_scale),
            ("chi_squared_gate_threshold", self.chi_squared_gate_threshold),
            (
                "cluster_distance_threshold_m",
                self.cluster_distance_threshold_m,
            ),
            (
                "near_parallel_rejection_angle_deg",
                self.near_parallel_rejection_angle_deg,
            ),
            ("max_coast_seconds", self.max_coast_seconds),
            ("min_quality_score", self.min_quality_score),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite."));
            }
        }
        if self.cv_process_accel_std <= 0.0 || self.ct_process_accel_std <= 0.0 {
            return Err("CV/CT process noise must be > 0.".to_string());
        }
        if self.ct_turn_rate_std <= 0.0 {
            return Err("ct_turn_rate_std must be > 0.".to_string());
        }
        if self.innovation_window < 1 {
            return Err("innovation_window must be at least 1.".to_string());
        }
        if self.innovation_scale_factor < 1.0 || self.innovation_max_scale < 1.0 {
            return Err("innovation adaptation scale factors must be >= 1.0.".to_string());
        }
        if self.chi_squared_gate_threshold <= 0.0 {
            return Err("chi_squared_gate_threshold must be > 0.".to_string());
        }
        if self.cluster_distance_threshold_m <= 0.0 {
            return Err("cluster_distance_threshold_m must be > 0.".to_string());
        }
        if self.near_parallel_rejection_angle_deg <= 0.0 {
            return Err("near_parallel_rejection_angle_deg must be > 0.".to_string());
        }
        if self.confirmation_m < 1 || self.confirmation_n < self.confirmation_m {
            return Err("confirmation_n must be >= confirmation_m >= 1.".to_string());
        }
        if self.max_coast_seconds < 0.0 {
            return Err("max_coast_seconds must be >= 0.".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_quality_score) {
            return Err("min_quality_score must be within [0.0, 1.0].".to_string());
        }
        Ok(())
    }

    fn adaptive_filter_config(&self) -> AdaptiveFilterConfig {
        AdaptiveFilterConfig {
            cv_accel_std: self.cv_process_accel_std,
            ct_accel_std: self.ct_process_accel_std,
            ct_turn_rate_std: self.ct_turn_rate_std,
            innovation_window: self.innovation_window as usize,
            innovation_scale_factor: self.innovation_scale_factor,
            innovation_max_scale: self.innovation_max_scale,
            adaptive_measurement_noise: self.adaptive_measurement_noise,
            ..AdaptiveFilterConfig::default()
        }
    }

    fn lifecycle_config(&self) -> TrackLifecycleConfig {
        TrackLifecycleConfig {
            confirmation_m: self.confirmation_m,
            confirmation_n: self.confirmation_n,
            max_coast_frames: self.max_coast_frames,
            max_coast_seconds: self.max_coast_seconds,
            min_quality_score: self.min_quality_score,
        }
    }
}

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
}

#[derive(Clone, Debug)]
pub struct TriangulatedEstimate {
    pub position: Vector3<f64>,
    pub measurement_std_m: f64,
}

pub type Matrix7 = SMatrix<f64, 7, 7>;
pub type Vector7 = SVector<f64, 7>;

#[derive(Clone, Debug)]
pub(crate) struct KalmanTrack3D {
    pub(crate) timestamp_s: f64,
    pub(crate) state: Vector6,
    pub(crate) covariance: Matrix6,
    process_accel_std: f64,
}

impl KalmanTrack3D {
    fn initialize(
        timestamp_s: f64,
        position: Vector3<f64>,
        position_std_m: f64,
        velocity_std_mps: f64,
    ) -> Self {
        let mut state = Vector6::zeros();
        state.fixed_rows_mut::<3>(0).copy_from(&position);

        let covariance = Matrix6::from_diagonal(&Vector6::from_row_slice(&[
            position_std_m.powi(2),
            position_std_m.powi(2),
            position_std_m.powi(2),
            velocity_std_mps.powi(2),
            velocity_std_mps.powi(2),
            velocity_std_mps.powi(2),
        ]));

        Self {
            timestamp_s,
            state,
            covariance,
            process_accel_std: default_cv_process_accel_std(),
        }
    }

    #[inline]
    fn predict(&mut self, timestamp_s: f64) {
        let dt = timestamp_s - self.timestamp_s;
        if dt <= 0.0 {
            return;
        }

        let mut transition = Matrix6::identity();
        transition[(0, 3)] = dt;
        transition[(1, 4)] = dt;
        transition[(2, 5)] = dt;

        let accel_var = self.process_accel_std.powi(2);
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;

        let process_block = Matrix3::new(
            dt4 / 4.0,
            dt3 / 2.0,
            0.0,
            dt3 / 2.0,
            dt2,
            0.0,
            0.0,
            0.0,
            0.0,
        ) * accel_var;

        let mut process_noise = Matrix6::zeros();
        for axis in 0..3 {
            let velocity = axis + 3;
            process_noise[(axis, axis)] = process_block[(0, 0)];
            process_noise[(axis, velocity)] = process_block[(0, 1)];
            process_noise[(velocity, axis)] = process_block[(1, 0)];
            process_noise[(velocity, velocity)] = process_block[(1, 1)];
        }

        self.state = transition * self.state;
        self.covariance = transition * self.covariance * transition.transpose() + process_noise;
        self.timestamp_s = timestamp_s;
    }

    #[inline]
    fn update_position(
        &mut self,
        position: Vector3<f64>,
        measurement_std_m: f64,
    ) -> Result<(), String> {
        let mut measurement_matrix = SMatrix::<f64, 3, 6>::zeros();
        measurement_matrix[(0, 0)] = 1.0;
        measurement_matrix[(1, 1)] = 1.0;
        measurement_matrix[(2, 2)] = 1.0;

        let measurement_cov = Matrix3::identity() * measurement_std_m.powi(2);
        let innovation = position - (measurement_matrix * self.state);
        let innovation_cov =
            measurement_matrix * self.covariance * measurement_matrix.transpose() + measurement_cov;
        let innovation_cov_decomp = innovation_cov
            .cholesky()
            .ok_or_else(|| "innovation covariance is not positive-definite".to_string())?;
        let kalman_gain = innovation_cov_decomp
            .solve(&(measurement_matrix * self.covariance.transpose()))
            .transpose();

        self.state += kalman_gain * innovation;
        let ikh = Matrix6::identity() - kalman_gain * measurement_matrix;
        self.covariance = ikh * self.covariance * ikh.transpose()
            + kalman_gain * measurement_cov * kalman_gain.transpose();
        Ok(())
    }

}

#[derive(Clone, Debug)]
pub struct AdaptiveFilterConfig {
    pub cv_accel_std: f64,
    pub ct_accel_std: f64,
    pub ct_turn_rate_std: f64,
    pub init_position_std_m: f64,
    pub init_velocity_std_mps: f64,
    pub cv_to_ct_prob: f64,
    pub ct_to_cv_prob: f64,
    pub innovation_window: usize,
    pub innovation_scale_factor: f64,
    pub innovation_max_scale: f64,
    pub adaptive_measurement_noise: bool,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            cv_accel_std: default_cv_process_accel_std(),
            ct_accel_std: default_ct_process_accel_std(),
            ct_turn_rate_std: default_ct_turn_rate_std(),
            init_position_std_m: 30.0,
            init_velocity_std_mps: 15.0,
            cv_to_ct_prob: 0.05,
            ct_to_cv_prob: 0.10,
            innovation_window: default_innovation_window() as usize,
            innovation_scale_factor: default_innovation_scale_factor(),
            innovation_max_scale: default_innovation_max_scale(),
            adaptive_measurement_noise: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CoordinatedTurnTrack3D {
    timestamp_s: f64,
    state: Vector7,
    covariance: Matrix7,
    accel_std: f64,
    turn_rate_std: f64,
}

impl CoordinatedTurnTrack3D {
    fn initialize(
        timestamp_s: f64,
        position: Vector3<f64>,
        velocity: Option<Vector3<f64>>,
        config: &AdaptiveFilterConfig,
    ) -> Self {
        let mut state = Vector7::zeros();
        state.fixed_rows_mut::<3>(0).copy_from(&position);
        if let Some(velocity) = velocity {
            state.fixed_rows_mut::<3>(3).copy_from(&velocity);
        }
        let covariance = Matrix7::from_diagonal(&Vector7::from_row_slice(&[
            config.init_position_std_m.powi(2),
            config.init_position_std_m.powi(2),
            config.init_position_std_m.powi(2),
            config.init_velocity_std_mps.powi(2),
            config.init_velocity_std_mps.powi(2),
            config.init_velocity_std_mps.powi(2),
            config.ct_turn_rate_std.powi(2),
        ]));
        Self {
            timestamp_s,
            state,
            covariance,
            accel_std: config.ct_accel_std,
            turn_rate_std: config.ct_turn_rate_std,
        }
    }

    #[inline]
    fn predict(&mut self, timestamp_s: f64) {
        let dt = timestamp_s - self.timestamp_s;
        if dt <= 0.0 {
            return;
        }

        let omega = self.state[6];
        let vx = self.state[3];
        let vy = self.state[4];
        let dt2 = dt * dt;

        if omega.abs() < 1.0e-6 {
            self.state[0] += vx * dt;
            self.state[1] += vy * dt;
        } else {
            let sin_wt = (omega * dt).sin();
            let cos_wt = (omega * dt).cos();
            self.state[0] += (vx * sin_wt - vy * (1.0 - cos_wt)) / omega;
            self.state[1] += (vx * (1.0 - cos_wt) + vy * sin_wt) / omega;
            self.state[3] = vx * cos_wt - vy * sin_wt;
            self.state[4] = vx * sin_wt + vy * cos_wt;
        }
        self.state[2] += self.state[5] * dt;

        let mut transition = Matrix7::identity();
        if omega.abs() < 1.0e-6 {
            transition[(0, 3)] = dt;
            transition[(1, 4)] = dt;
        } else {
            let sin_wt = (omega * dt).sin();
            let cos_wt = (omega * dt).cos();
            transition[(0, 3)] = sin_wt / omega;
            transition[(0, 4)] = -(1.0 - cos_wt) / omega;
            transition[(1, 3)] = (1.0 - cos_wt) / omega;
            transition[(1, 4)] = sin_wt / omega;
            transition[(3, 3)] = cos_wt;
            transition[(3, 4)] = -sin_wt;
            transition[(4, 3)] = sin_wt;
            transition[(4, 4)] = cos_wt;
        }
        transition[(2, 5)] = dt;

        let accel_var = self.accel_std.powi(2);
        let mut process_noise = Matrix7::zeros();
        for axis in 0..3 {
            let velocity = axis + 3;
            process_noise[(axis, axis)] = dt2 * dt2 / 4.0 * accel_var;
            process_noise[(axis, velocity)] = dt2 * dt / 2.0 * accel_var;
            process_noise[(velocity, axis)] = dt2 * dt / 2.0 * accel_var;
            process_noise[(velocity, velocity)] = dt2 * accel_var;
        }
        process_noise[(6, 6)] = self.turn_rate_std.powi(2) * dt;

        self.covariance = transition * self.covariance * transition.transpose() + process_noise;
        self.timestamp_s = timestamp_s;
    }

    #[inline]
    fn update_position(
        &mut self,
        position: Vector3<f64>,
        measurement_std_m: f64,
    ) -> Result<f64, String> {
        let mut measurement_matrix = SMatrix::<f64, 3, 7>::zeros();
        measurement_matrix[(0, 0)] = 1.0;
        measurement_matrix[(1, 1)] = 1.0;
        measurement_matrix[(2, 2)] = 1.0;

        let measurement_cov = Matrix3::identity() * measurement_std_m.powi(2);
        let innovation = position - (measurement_matrix * self.state);
        let innovation_cov = measurement_matrix * self.covariance * measurement_matrix.transpose()
            + measurement_cov;
        let innovation_cov_decomp = innovation_cov
            .cholesky()
            .ok_or_else(|| "innovation covariance is not positive-definite".to_string())?;
        let kalman_gain = innovation_cov_decomp
            .solve(&(measurement_matrix * self.covariance.transpose()))
            .transpose();

        self.state += kalman_gain * innovation;
        let ikh = Matrix7::identity() - kalman_gain * measurement_matrix;
        self.covariance = ikh * self.covariance * ikh.transpose()
            + kalman_gain * measurement_cov * kalman_gain.transpose();

        let nis = innovation
            .dot(&innovation_cov_decomp.solve(&innovation))
            .max(0.0);
        Ok(nis)
    }

    fn as_cv_state(&self) -> (Vector6, Matrix6) {
        let mut state = Vector6::zeros();
        state.fixed_rows_mut::<6>(0)
            .copy_from(&self.state.fixed_rows::<6>(0));
        let mut covariance = Matrix6::zeros();
        covariance.copy_from(&self.covariance.fixed_view::<6, 6>(0, 0));
        (state, covariance)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IMMTrack3D {
    pub(crate) cv_track: KalmanTrack3D,
    pub(crate) ct_track: CoordinatedTurnTrack3D,
    mode_probabilities: [f64; 2],
    config: AdaptiveFilterConfig,
    innovation_history: Vec<f64>,
    q_scale: f64,
}

impl IMMTrack3D {
    fn initialize(
        timestamp_s: f64,
        position: Vector3<f64>,
        position_std_m: f64,
        velocity_std_mps: f64,
        config: &TrackerConfig,
    ) -> Self {
        let adaptive = config.adaptive_filter_config();
        let mut cv_track = KalmanTrack3D::initialize(
            timestamp_s,
            position,
            position_std_m,
            velocity_std_mps,
        );
        cv_track.process_accel_std = adaptive.cv_accel_std;
        let mut ct_config = adaptive.clone();
        ct_config.init_position_std_m = position_std_m;
        ct_config.init_velocity_std_mps = velocity_std_mps;
        let ct_track =
            CoordinatedTurnTrack3D::initialize(timestamp_s, position, None, &ct_config);
        Self {
            cv_track,
            ct_track,
            mode_probabilities: [0.8, 0.2],
            config: adaptive,
            innovation_history: Vec::new(),
            q_scale: 1.0,
        }
    }

    fn timestamp_s(&self) -> f64 {
        self.cv_track.timestamp_s
    }

    fn position(&self) -> Vector3<f64> {
        self.state().fixed_rows::<3>(0).into_owned()
    }

    fn velocity(&self) -> Vector3<f64> {
        self.state().fixed_rows::<3>(3).into_owned()
    }

    fn state(&self) -> Vector6 {
        let (p_cv, p_ct) = (self.mode_probabilities[0], self.mode_probabilities[1]);
        let (ct_state, _) = self.ct_track.as_cv_state();
        (self.cv_track.state * p_cv) + (ct_state * p_ct)
    }

    fn covariance(&self) -> Matrix6 {
        let combined_state = self.state();
        let (ct_state, ct_covariance) = self.ct_track.as_cv_state();
        let cv_diff = self.cv_track.state - combined_state;
        let ct_diff = ct_state - combined_state;
        (self.cv_track.covariance + (cv_diff * cv_diff.transpose())) * self.mode_probabilities[0]
            + (ct_covariance + (ct_diff * ct_diff.transpose())) * self.mode_probabilities[1]
    }

    fn predict(&mut self, timestamp_s: f64) {
        let transition = [
            [1.0 - self.config.cv_to_ct_prob, self.config.cv_to_ct_prob],
            [self.config.ct_to_cv_prob, 1.0 - self.config.ct_to_cv_prob],
        ];

        let mut c_bar = [0.0; 2];
        c_bar[0] = transition[0][0] * self.mode_probabilities[0]
            + transition[1][0] * self.mode_probabilities[1];
        c_bar[1] = transition[0][1] * self.mode_probabilities[0]
            + transition[1][1] * self.mode_probabilities[1];
        let total = (c_bar[0] + c_bar[1]).max(1.0e-12);
        c_bar[0] /= total;
        c_bar[1] /= total;

        let mut mix = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                mix[i][j] = transition[i][j] * self.mode_probabilities[i] / c_bar[j].max(1.0e-12);
            }
        }

        let (ct_state, _) = self.ct_track.as_cv_state();
        self.cv_track.state = self.cv_track.state * mix[0][0] + ct_state * mix[1][0];

        let mut mixed_ct_state = Vector7::zeros();
        let cv_state = self.cv_track.state;
        let previous_ct_state = self.ct_track.state;
        mixed_ct_state
            .fixed_rows_mut::<6>(0)
            .copy_from(&(cv_state * mix[0][1] + ct_state * mix[1][1]));
        mixed_ct_state[6] = previous_ct_state[6] * mix[1][1];
        self.ct_track.state = mixed_ct_state;

        let original_cv_std = self.cv_track.process_accel_std;
        let original_ct_std = self.ct_track.accel_std;
        self.cv_track.process_accel_std *= self.q_scale;
        self.ct_track.accel_std *= self.q_scale;
        self.cv_track.predict(timestamp_s);
        self.ct_track.predict(timestamp_s);
        self.cv_track.process_accel_std = original_cv_std;
        self.ct_track.accel_std = original_ct_std;
        self.mode_probabilities = c_bar;
    }

    fn update_position(
        &mut self,
        position: Vector3<f64>,
        measurement_std_m: f64,
    ) -> Result<(), String> {
        let mut effective_measurement_std = measurement_std_m.max(1.0);
        if self.config.adaptive_measurement_noise && self.innovation_history.len() >= 3 {
            let mean_nis = self.innovation_history.iter().copied().sum::<f64>()
                / self.innovation_history.len() as f64;
            let scale = (mean_nis / 3.0)
                .clamp(1.0, self.config.innovation_max_scale)
                .sqrt();
            effective_measurement_std *= scale;
        }

        let mut measurement_matrix_cv = SMatrix::<f64, 3, 6>::zeros();
        measurement_matrix_cv[(0, 0)] = 1.0;
        measurement_matrix_cv[(1, 1)] = 1.0;
        measurement_matrix_cv[(2, 2)] = 1.0;
        let measurement_cov = Matrix3::identity() * effective_measurement_std.powi(2);

        let innovation_cv = position - (measurement_matrix_cv * self.cv_track.state);
        let innovation_cov_cv = measurement_matrix_cv
            * self.cv_track.covariance
            * measurement_matrix_cv.transpose()
            + measurement_cov;
        let cv_likelihood = gaussian_likelihood(&innovation_cv, &innovation_cov_cv);
        self.cv_track
            .update_position(position, effective_measurement_std)?;

        let nis_ct = self
            .ct_track
            .update_position(position, effective_measurement_std)?;
        let mut measurement_matrix_ct = SMatrix::<f64, 3, 7>::zeros();
        measurement_matrix_ct[(0, 0)] = 1.0;
        measurement_matrix_ct[(1, 1)] = 1.0;
        measurement_matrix_ct[(2, 2)] = 1.0;
        let innovation_ct = position - (measurement_matrix_ct * self.ct_track.state);
        let innovation_cov_ct = measurement_matrix_ct
            * self.ct_track.covariance
            * measurement_matrix_ct.transpose()
            + measurement_cov;
        let ct_likelihood = gaussian_likelihood(&innovation_ct, &innovation_cov_ct);

        let posterior = [
            self.mode_probabilities[0] * cv_likelihood.max(1.0e-30),
            self.mode_probabilities[1] * ct_likelihood.max(1.0e-30),
        ];
        let total = (posterior[0] + posterior[1]).max(1.0e-12);
        self.mode_probabilities = [posterior[0] / total, posterior[1] / total];

        self.innovation_history.push(nis_ct);
        if self.innovation_history.len() > self.config.innovation_window {
            let keep_from = self.innovation_history.len() - self.config.innovation_window;
            self.innovation_history.drain(0..keep_from);
        }
        if self.innovation_history.len() >= 3 {
            let mean_nis = self.innovation_history.iter().copied().sum::<f64>()
                / self.innovation_history.len() as f64;
            if mean_nis > self.config.innovation_scale_factor * 3.0 {
                self.q_scale = (mean_nis / 3.0).min(self.config.innovation_max_scale);
            } else {
                self.q_scale = (self.q_scale * 0.9).max(1.0);
            }
        }

        Ok(())
    }

    fn snapshot(
        &self,
        track_id: &str,
        measurement_std_m: f64,
        update_count: u32,
        stale_steps: u32,
        lifecycle_state: Option<String>,
        quality_score: Option<f64>,
    ) -> TrackState {
        TrackState {
            track_id: track_id.to_string(),
            timestamp_s: self.timestamp_s(),
            position: self.position(),
            velocity: self.velocity(),
            covariance: self.covariance(),
            measurement_std_m,
            update_count,
            stale_steps,
            lifecycle_state,
            quality_score,
        }
    }
}

#[inline]
fn gaussian_likelihood(innovation: &Vector3<f64>, covariance: &Matrix3<f64>) -> f64 {
    covariance
        .cholesky()
        .map(|decomp| {
            let logdet = 2.0
                * decomp
                    .l()
                    .diagonal()
                    .iter()
                    .copied()
                    .map(|value| value.abs().ln())
                    .sum::<f64>();
            let maha = innovation.dot(&decomp.solve(innovation));
            let n = innovation.len() as f64;
            (-0.5 * (n * (2.0 * std::f64::consts::PI).ln() + logdet + maha))
                .exp()
                .max(1.0e-30)
        })
        .unwrap_or(1.0e-30)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleState {
    Tentative,
    Confirmed,
    Coasting,
    Deleted,
}

impl LifecycleState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tentative => "tentative",
            Self::Confirmed => "confirmed",
            Self::Coasting => "coasting",
            Self::Deleted => "deleted",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackLifecycleConfig {
    pub confirmation_m: u32,
    pub confirmation_n: u32,
    pub max_coast_frames: u32,
    pub max_coast_seconds: f64,
    pub min_quality_score: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ManagedTrack {
    pub(crate) track_id: String,
    pub(crate) filter_state: IMMTrack3D,
    pub(crate) lifecycle_state: LifecycleState,
    config: TrackLifecycleConfig,
    pub(crate) measurement_std_m: f64,
    pub(crate) update_count: u32,
    pub(crate) stale_steps: u32,
    last_update_time_s: f64,
    update_history: Vec<bool>,
    quality_score: f64,
}

impl ManagedTrack {
    pub(crate) fn new(
        track_id: String,
        timestamp_s: f64,
        position: Vector3<f64>,
        measurement_std_m: f64,
        config: &TrackerConfig,
    ) -> Self {
        Self {
            track_id,
            filter_state: IMMTrack3D::initialize(
                timestamp_s,
                position,
                measurement_std_m.max(5.0),
                20.0,
                config,
            ),
            lifecycle_state: LifecycleState::Tentative,
            config: config.lifecycle_config(),
            measurement_std_m,
            update_count: 1,
            stale_steps: 0,
            last_update_time_s: timestamp_s,
            update_history: vec![true],
            quality_score: 0.5,
        }
    }

    pub(crate) fn predict(&mut self, timestamp_s: f64) {
        self.filter_state.predict(timestamp_s);
    }

    pub(crate) fn update(
        &mut self,
        position: Vector3<f64>,
        measurement_std_m: f64,
        timestamp_s: f64,
    ) -> Result<(), String> {
        self.filter_state.update_position(position, measurement_std_m)?;
        self.measurement_std_m = measurement_std_m;
        self.update_count += 1;
        self.stale_steps = 0;
        self.last_update_time_s = timestamp_s;
        self.update_history.push(true);
        self.trim_update_history();
        self.update_lifecycle();
        Ok(())
    }

    pub(crate) fn mark_missed(&mut self, timestamp_s: f64) {
        self.stale_steps += 1;
        self.update_history.push(false);
        self.trim_update_history();
        self.update_lifecycle();
        let time_since_update = timestamp_s - self.last_update_time_s;
        if self.stale_steps >= self.config.max_coast_frames
            || time_since_update >= self.config.max_coast_seconds
        {
            self.lifecycle_state = LifecycleState::Deleted;
        }
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.lifecycle_state != LifecycleState::Deleted
    }

    pub(crate) fn predicted_measurement(
        &self,
        timestamp_s: f64,
    ) -> (Vector3<f64>, Matrix3<f64>) {
        let mut predicted = self.filter_state.clone();
        predicted.predict(timestamp_s);
        let covariance = predicted.covariance();
        (
            predicted.position(),
            covariance.fixed_view::<3, 3>(0, 0).into_owned(),
        )
    }

    fn trim_update_history(&mut self) {
        if self.update_history.len() > self.config.confirmation_n as usize {
            let keep_from = self.update_history.len() - self.config.confirmation_n as usize;
            self.update_history.drain(0..keep_from);
        }
    }

    fn update_lifecycle(&mut self) {
        if self.lifecycle_state == LifecycleState::Deleted {
            return;
        }

        let recent_updates = self.update_history.iter().filter(|updated| **updated).count() as u32;
        let history_len = self.update_history.len().max(1) as f64;
        self.quality_score = recent_updates as f64 / history_len;

        match self.lifecycle_state {
            LifecycleState::Tentative => {
                if recent_updates >= self.config.confirmation_m {
                    self.lifecycle_state = LifecycleState::Confirmed;
                } else if self.update_history.len() >= self.config.confirmation_n as usize
                    && recent_updates < 2
                {
                    self.lifecycle_state = LifecycleState::Deleted;
                }
            }
            LifecycleState::Confirmed => {
                if self.stale_steps > 0 {
                    self.lifecycle_state = LifecycleState::Coasting;
                }
            }
            LifecycleState::Coasting => {
                if self.stale_steps == 0 {
                    self.lifecycle_state = LifecycleState::Confirmed;
                } else if self.quality_score < self.config.min_quality_score {
                    self.lifecycle_state = LifecycleState::Deleted;
                }
            }
            LifecycleState::Deleted => {}
        }
    }

    fn snapshot(&self) -> TrackState {
        self.filter_state.snapshot(
            &self.track_id,
            self.measurement_std_m,
            self.update_count,
            self.stale_steps,
            Some(self.lifecycle_state.as_str().to_string()),
            Some(self.quality_score),
        )
    }
}

pub struct TrackingEngine {
    config: TrackerConfig,
    min_intersection_angle_rad: f64,
    near_parallel_rejection_angle_rad: f64,
    nodes: HashMap<String, NodeState>,
    tracks: HashMap<String, ManagedTrack>,
    latest_frame: Option<PlatformFrame>,
    node_health: HashMap<String, NodeHealthTracker>,
    frame_timestamps: Vec<f64>,
    next_track_index: u64,
    /// Scratch HashMap reused across frames to avoid per-cluster allocation in deduplicate_cluster.
    dedup_scratch: HashMap<String, BearingObservation>,
}

impl TrackingEngine {
    pub fn new(config: TrackerConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            min_intersection_angle_rad: config.min_intersection_angle_deg.to_radians(),
            near_parallel_rejection_angle_rad: config
                .near_parallel_rejection_angle_deg
                .to_radians(),
            config,
            nodes: HashMap::new(),
            tracks: HashMap::new(),
            latest_frame: None,
            node_health: HashMap::new(),
            frame_timestamps: Vec::new(),
            next_track_index: 0,
            dedup_scratch: HashMap::new(),
        })
    }

    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }

    pub fn latest_frame(&self) -> Option<&PlatformFrame> {
        self.latest_frame.as_ref()
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.tracks.clear();
        self.latest_frame = None;
        self.node_health.clear();
        self.frame_timestamps.clear();
        self.next_track_index = 0;
        self.dedup_scratch.clear();
    }

    pub fn node_health_snapshot(&self, current_timestamp_s: f64) -> Vec<NodeHealthSnapshot> {
        let stale_threshold = self.config.max_stale_steps as f64 * 0.25;
        let mut snapshots: Vec<_> = self
            .node_health
            .values()
            .map(|tracker| NodeHealthSnapshot {
                node_id: tracker.node_id.clone(),
                last_seen_s: tracker.last_seen_s,
                observation_rate_hz: tracker.observation_rate_hz(),
                accepted_count: tracker.accepted_count,
                rejected_count: tracker.rejected_count,
                health_score: tracker.health_score(current_timestamp_s, stale_threshold.max(2.0)),
            })
            .collect();
        snapshots.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        snapshots
    }

    pub fn mean_frame_rate_hz(&self) -> f64 {
        if self.frame_timestamps.len() < 2 {
            return 0.0;
        }
        let first = self.frame_timestamps.first().unwrap();
        let last = self.frame_timestamps.last().unwrap();
        let elapsed = last - first;
        if elapsed > 0.0 {
            (self.frame_timestamps.len() - 1) as f64 / elapsed
        } else {
            0.0
        }
    }

    pub fn stale_node_count(&self, current_timestamp_s: f64) -> u32 {
        let stale_threshold = (self.config.max_stale_steps as f64 * 0.25).max(2.0);
        self.node_health
            .values()
            .filter(|tracker| current_timestamp_s - tracker.last_seen_s > stale_threshold)
            .count() as u32
    }

    pub fn ingest_frame(
        &mut self,
        timestamp_s: f64,
        node_states: Vec<NodeState>,
        observations: Vec<BearingObservation>,
        truths: Vec<TruthState>,
    ) -> PlatformFrame {
        for node in node_states {
            self.nodes.insert(node.node_id.clone(), node);
        }

        self.frame_timestamps.push(timestamp_s);
        // Keep at most 1000 timestamps for rolling frame rate computation
        if self.frame_timestamps.len() > 1000 {
            self.frame_timestamps.drain(..500);
        }

        let (prefiltered_observations, mut rejections) =
            self.prefilter_observations(timestamp_s, observations);

        let (accepted_observations, updated_track_ids) = match self.config.data_association_mode {
            AssociationMode::GNN => {
                self.ingest_gnn(timestamp_s, prefiltered_observations, &mut rejections)
            }
            AssociationMode::JPDA => {
                self.ingest_jpda(timestamp_s, prefiltered_observations, &mut rejections)
            }
            AssociationMode::Labeled => {
                self.ingest_labeled(timestamp_s, prefiltered_observations, &mut rejections)
            }
        };

        let updated_track_set = updated_track_ids
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let n_tracks = self.tracks.len();
        let mut existing_track_ids = Vec::with_capacity(n_tracks);
        existing_track_ids.extend(self.tracks.keys().cloned());
        for track_id in existing_track_ids {
            if updated_track_set.contains(&track_id) {
                continue;
            }
            let should_remove = if let Some(managed_track) = self.tracks.get_mut(&track_id) {
                managed_track.predict(timestamp_s);
                managed_track.mark_missed(timestamp_s);
                !managed_track.is_alive()
            } else {
                false
            };

            if should_remove {
                self.tracks.remove(&track_id);
            }
        }

        let mut track_ids = Vec::with_capacity(self.tracks.len());
        track_ids.extend(self.tracks.keys().cloned());
        track_ids.sort();
        let mut track_states = Vec::with_capacity(track_ids.len());
        track_states.extend(
            track_ids
                .iter()
                .filter_map(|track_id| self.tracks.get(track_id).map(ManagedTrack::snapshot)),
        );

        let mut node_ids = Vec::with_capacity(self.nodes.len());
        node_ids.extend(self.nodes.keys().cloned());
        node_ids.sort();
        let mut nodes = Vec::with_capacity(node_ids.len());
        nodes.extend(node_ids.iter().filter_map(|node_id| self.nodes.get(node_id).cloned()));

        // Update per-node health trackers
        for obs in &accepted_observations {
            self.node_health
                .entry(obs.node_id.clone())
                .or_insert_with(|| NodeHealthTracker::new(&obs.node_id, timestamp_s))
                .record_accepted(timestamp_s);
        }
        for rej in &rejections {
            self.node_health
                .entry(rej.node_id.clone())
                .or_insert_with(|| NodeHealthTracker::new(&rej.node_id, timestamp_s))
                .record_rejected(timestamp_s);
        }

        let metrics = build_metrics(&track_states, &accepted_observations, &rejections, &truths);
        let frame = PlatformFrame {
            timestamp_s,
            nodes,
            observations: accepted_observations,
            rejected_observations: rejections,
            tracks: track_states,
            truths,
            metrics,
            generation_rejections: Vec::new(),
        };

        self.latest_frame = Some(frame.clone());
        frame
    }

    fn allocate_track_id(&mut self) -> String {
        let track_id = format!("track-{}", self.next_track_index);
        self.next_track_index += 1;
        track_id
    }

    fn minimum_geometry_angle_rad(&self) -> f64 {
        self.min_intersection_angle_rad
            .max(self.near_parallel_rejection_angle_rad)
    }

    fn cluster_has_weak_geometry(&self, observations: &[BearingObservation]) -> bool {
        self.max_pairwise_angle(observations) < self.minimum_geometry_angle_rad()
    }

    fn ingest_labeled(
        &mut self,
        timestamp_s: f64,
        prefiltered_observations: Vec<BearingObservation>,
        rejections: &mut Vec<ObservationRejection>,
    ) -> (Vec<BearingObservation>, SmallVec<[String; 8]>) {
        let mut grouped_observations: HashMap<String, Vec<BearingObservation>> = HashMap::new();
        for observation in prefiltered_observations {
            grouped_observations
                .entry(observation.target_id.clone())
                .or_default()
                .push(observation);
        }

        let mut accepted_observations = Vec::new();
        let mut updated_track_ids: SmallVec<[String; 8]> = SmallVec::new();
        let mut sorted_track_ids = grouped_observations.keys().cloned().collect::<Vec<_>>();
        sorted_track_ids.sort();

        for track_id in sorted_track_ids {
            let cluster = grouped_observations.remove(&track_id).unwrap_or_default();
            let (deduped_cluster, duplicate_rejections) = self.deduplicate_cluster(cluster);
            rejections.extend(duplicate_rejections);

            if deduped_cluster.len() < self.config.min_observations as usize {
                rejections.extend(self.reject_cluster(
                    &deduped_cluster,
                    REJECT_INSUFFICIENT_CLUSTER,
                    &format!(
                        "Requires at least {} observations.",
                        self.config.min_observations
                    ),
                ));
                continue;
            }

            if self.cluster_has_weak_geometry(&deduped_cluster) {
                rejections.extend(self.reject_cluster(
                    &deduped_cluster,
                    REJECT_WEAK_GEOMETRY,
                    "Observation rays are nearly parallel.",
                ));
                continue;
            }

            let estimate = match fuse_bearing_cluster(&deduped_cluster) {
                Ok(estimate) => estimate,
                Err(error) => {
                    rejections.extend(self.reject_cluster(
                        &deduped_cluster,
                        REJECT_FUSION_FAILURE,
                        &error,
                    ));
                    continue;
                }
            };

            accepted_observations.extend(deduped_cluster.iter().cloned());
            if let Some(managed_track) = self.tracks.get_mut(&track_id) {
                managed_track.predict(timestamp_s);
                if managed_track
                    .update(estimate.position, estimate.measurement_std_m, timestamp_s)
                    .is_err()
                {
                    rejections.extend(self.reject_cluster(
                        &deduped_cluster,
                        REJECT_FUSION_FAILURE,
                        "Failed to update IMM track state.",
                    ));
                    continue;
                }
            } else {
                self.tracks.insert(
                    track_id.clone(),
                    ManagedTrack::new(
                        track_id.clone(),
                        timestamp_s,
                        estimate.position,
                        estimate.measurement_std_m,
                        &self.config,
                    ),
                );
            }
            updated_track_ids.push(track_id);
        }

        (accepted_observations, updated_track_ids)
    }

    fn ingest_gnn(
        &mut self,
        timestamp_s: f64,
        prefiltered_observations: Vec<BearingObservation>,
        rejections: &mut Vec<ObservationRejection>,
    ) -> (Vec<BearingObservation>, SmallVec<[String; 8]>) {
        use crate::association::{cluster_observations, GNNAssociator, TrackAssignment};

        let clusters = cluster_observations(
            &prefiltered_observations,
            self.config.cluster_distance_threshold_m,
        );
        let mut valid_clusters = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            let (deduped, dup_rejections) = self.deduplicate_cluster(cluster);
            rejections.extend(dup_rejections);

            if deduped.len() < self.config.min_observations as usize {
                rejections.extend(self.reject_cluster(
                    &deduped,
                    REJECT_INSUFFICIENT_CLUSTER,
                    &format!(
                        "Requires at least {} observations.",
                        self.config.min_observations
                    ),
                ));
                continue;
            }

            if self.cluster_has_weak_geometry(&deduped) {
                rejections.extend(self.reject_cluster(
                    &deduped,
                    REJECT_WEAK_GEOMETRY,
                    "Observation rays are nearly parallel.",
                ));
                continue;
            }

            valid_clusters.push(deduped);
        }

        let associator = GNNAssociator {
            gate_threshold: self.config.chi_squared_gate_threshold,
        };
        let assignments = associator.associate(&self.tracks, &valid_clusters, timestamp_s);

        let mut accepted_observations = Vec::new();
        let mut updated_track_ids: SmallVec<[String; 8]> = SmallVec::new();

        for assignment in assignments {
            let cluster = &valid_clusters[assignment.cluster_index];
            accepted_observations.extend(cluster.iter().cloned());

            let track_id = match assignment.track_id {
                TrackAssignment::Existing(ref id) => id.clone(),
                TrackAssignment::NewTrack => self.allocate_track_id(),
            };

            match assignment.track_id {
                TrackAssignment::Existing(ref id) => {
                    if let Some(managed_track) = self.tracks.get_mut(id) {
                        managed_track.predict(timestamp_s);
                        if managed_track
                            .update(assignment.position, assignment.measurement_std_m, timestamp_s)
                            .is_err()
                        {
                            rejections.extend(self.reject_cluster(
                                cluster,
                                REJECT_FUSION_FAILURE,
                                "Failed to update IMM track state.",
                            ));
                            continue;
                        }
                    }
                }
                TrackAssignment::NewTrack => {
                    self.tracks.insert(
                        track_id.clone(),
                        ManagedTrack::new(
                            track_id.clone(),
                            timestamp_s,
                            assignment.position,
                            assignment.measurement_std_m,
                            &self.config,
                        ),
                    );
                }
            }

            updated_track_ids.push(track_id);
        }

        (accepted_observations, updated_track_ids)
    }

    fn ingest_jpda(
        &mut self,
        timestamp_s: f64,
        prefiltered_observations: Vec<BearingObservation>,
        rejections: &mut Vec<ObservationRejection>,
    ) -> (Vec<BearingObservation>, SmallVec<[String; 8]>) {
        use crate::association::{cluster_observations, JPDAAssociator};

        let clusters = cluster_observations(
            &prefiltered_observations,
            self.config.cluster_distance_threshold_m,
        );
        let mut valid_clusters = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            let (deduped, dup_rejections) = self.deduplicate_cluster(cluster);
            rejections.extend(dup_rejections);

            if deduped.len() < self.config.min_observations as usize {
                rejections.extend(self.reject_cluster(
                    &deduped,
                    REJECT_INSUFFICIENT_CLUSTER,
                    &format!(
                        "Requires at least {} observations.",
                        self.config.min_observations
                    ),
                ));
                continue;
            }

            if self.cluster_has_weak_geometry(&deduped) {
                rejections.extend(self.reject_cluster(
                    &deduped,
                    REJECT_WEAK_GEOMETRY,
                    "Observation rays are nearly parallel.",
                ));
                continue;
            }

            valid_clusters.push(deduped);
        }

        let associator = JPDAAssociator {
            gate_threshold: self.config.chi_squared_gate_threshold,
            new_track_probability_threshold: 0.45,
            update_probability_threshold: 0.20,
        };
        let result = associator.associate(&self.tracks, &valid_clusters, timestamp_s);

        let mut accepted_observations = Vec::new();
        for cluster in &valid_clusters {
            accepted_observations.extend(cluster.iter().cloned());
        }

        let mut updated_track_ids: SmallVec<[String; 8]> = SmallVec::new();
        for update in result.track_updates {
            if let Some(track) = self.tracks.get_mut(&update.track_id) {
                track.predict(timestamp_s);
                if track
                    .update(update.position, update.measurement_std_m, timestamp_s)
                    .is_ok()
                {
                    updated_track_ids.push(update.track_id);
                }
            }
        }

        for candidate in result.new_tracks {
            let track_id = self.allocate_track_id();
            self.tracks.insert(
                track_id.clone(),
                ManagedTrack::new(
                    track_id.clone(),
                    timestamp_s,
                    candidate.position,
                    candidate.measurement_std_m,
                    &self.config,
                ),
            );
            updated_track_ids.push(track_id);
        }

        (accepted_observations, updated_track_ids)
    }

    fn prefilter_observations(
        &self,
        timestamp_s: f64,
        observations: Vec<BearingObservation>,
    ) -> (Vec<BearingObservation>, Vec<ObservationRejection>) {
        let mut accepted = Vec::new();
        let mut rejections = Vec::new();

        for observation in observations {
            if !self.nodes.contains_key(&observation.node_id) {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_UNKNOWN_NODE,
                    "Node state was not available for this frame.",
                ));
                continue;
            }
            if observation.target_id.trim().is_empty() {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_INVALID_TARGET,
                    "target_id must be non-empty.",
                ));
                continue;
            }
            if observation.confidence < self.config.min_confidence {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_LOW_CONFIDENCE,
                    &format!(
                        "Confidence {:.3} is below minimum {:.3}.",
                        observation.confidence, self.config.min_confidence
                    ),
                ));
                continue;
            }
            if !observation.bearing_std_rad.is_finite() || observation.bearing_std_rad <= 0.0 {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_INVALID_BEARING_STD,
                    "bearing_std_rad must be finite and > 0.",
                ));
                continue;
            }
            if observation.bearing_std_rad > self.config.max_bearing_std_rad {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_EXCESS_BEARING_STD,
                    &format!(
                        "bearing_std_rad {:.4} exceeds limit {:.4}.",
                        observation.bearing_std_rad, self.config.max_bearing_std_rad
                    ),
                ));
                continue;
            }

            let norm = observation.direction.norm();
            if !norm.is_finite() || norm <= 1.0e-8 {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_INVALID_DIRECTION,
                    "Direction vector norm must be finite and non-zero.",
                ));
                continue;
            }

            let timestamp_delta_s = (timestamp_s - observation.timestamp_s).abs();
            if timestamp_delta_s > self.config.max_timestamp_skew_s {
                rejections.push(reject_observation(
                    &observation,
                    REJECT_TIMESTAMP_SKEW,
                    &format!(
                        "Observation skew {:.3}s exceeds limit {:.3}s.",
                        timestamp_delta_s, self.config.max_timestamp_skew_s
                    ),
                ));
                continue;
            }

            let mut normalized = observation.clone();
            normalized.origin = observation.origin;
            normalized.direction = observation.direction / norm;
            accepted.push(normalized);
        }

        (accepted, rejections)
    }

    fn deduplicate_cluster(
        &mut self,
        cluster: Vec<BearingObservation>,
    ) -> (Vec<BearingObservation>, Vec<ObservationRejection>) {
        // Reuse the scratch HashMap to avoid a per-cluster allocation.
        let selected = &mut self.dedup_scratch;
        selected.clear();
        let mut rejections = Vec::new();

        for observation in cluster {
            if let Some(existing) = selected.get(&observation.node_id).cloned() {
                if observation_score(&observation) > observation_score(&existing) {
                    selected.insert(observation.node_id.clone(), observation);
                    rejections.push(reject_observation(
                        &existing,
                        REJECT_DUPLICATE_NODE,
                        "Replaced by higher-scoring observation from same node.",
                    ));
                } else {
                    rejections.push(reject_observation(
                        &observation,
                        REJECT_DUPLICATE_NODE,
                        "Lower-scoring duplicate from same node.",
                    ));
                }
            } else {
                selected.insert(observation.node_id.clone(), observation);
            }
        }

        let mut node_ids = selected.keys().cloned().collect::<Vec<_>>();
        node_ids.sort();
        let deduped = node_ids
            .into_iter()
            .filter_map(|node_id| selected.remove(&node_id))
            .collect::<Vec<_>>();
        (deduped, rejections)
    }

    fn max_pairwise_angle(&self, cluster: &[BearingObservation]) -> f64 {
        let mut max_angle = 0.0_f64;
        for index in 0..cluster.len() {
            let direction_a = cluster[index].direction;
            for inner in (index + 1)..cluster.len() {
                let direction_b = cluster[inner].direction;
                let dot = direction_a.dot(&direction_b).clamp(-1.0, 1.0);
                let angle = dot.acos();
                if angle > max_angle {
                    max_angle = angle;
                }
            }
        }
        max_angle
    }

    fn reject_cluster(
        &self,
        observations: &[BearingObservation],
        reason: &str,
        detail: &str,
    ) -> Vec<ObservationRejection> {
        observations
            .iter()
            .map(|observation| reject_observation(observation, reason, detail))
            .collect()
    }
}

fn reject_observation(
    observation: &BearingObservation,
    reason: &str,
    detail: &str,
) -> ObservationRejection {
    ObservationRejection {
        node_id: observation.node_id.clone(),
        target_id: observation.target_id.clone(),
        timestamp_s: observation.timestamp_s,
        reason: reason.to_string(),
        detail: detail.to_string(),
        origin: Some(observation.origin),
        attempted_point: None,
        closest_point: None,
        blocker_type: String::new(),
        first_hit_range_m: None,
    }
}

fn observation_score(observation: &BearingObservation) -> f64 {
    observation.confidence / observation.bearing_std_rad.powi(2).max(1.0e-6)
}

pub fn triangulate_bearings(observations: &[BearingObservation]) -> Result<Vector3<f64>, String> {
    let mut lhs = Matrix3::zeros();
    let mut rhs = Vector3::zeros();
    let mut used = 0_usize;

    for observation in observations {
        let direction = normalize(&observation.direction)?;
        let projector = Matrix3::identity() - (direction * direction.transpose());
        let weight =
            observation.confidence.max(0.05) / observation.bearing_std_rad.powi(2).max(1.0e-6);
        lhs += projector * weight;
        rhs += (projector * observation.origin) * weight;
        used += 1;
    }

    if used < 2 {
        return Err("At least two observations are required for triangulation.".to_string());
    }

    let svd = lhs.svd(true, true);
    svd.solve(&rhs, 1.0e-12).map_err(|error| error.to_string())
}

pub fn infer_measurement_std(
    observations: &[BearingObservation],
    estimate: &Vector3<f64>,
) -> Result<f64, String> {
    if observations.is_empty() {
        return Err("At least one observation is required.".to_string());
    }

    let mut per_node_std: SmallVec<[f64; 8]> = SmallVec::new();
    for observation in observations {
        let range_m = (estimate - observation.origin).norm();
        per_node_std.push((range_m * observation.bearing_std_rad).max(1.0));
    }

    Ok(mean(&per_node_std).unwrap_or(1.0))
}

pub fn fuse_bearing_cluster(
    observations: &[BearingObservation],
) -> Result<TriangulatedEstimate, String> {
    if observations.len() < 2 {
        return Err("At least two observations are required for fusion.".to_string());
    }

    let triangulated_position = triangulate_bearings(observations)?;
    let measurement_std_m = infer_measurement_std(observations, &triangulated_position)?;
    Ok(TriangulatedEstimate {
        position: triangulated_position,
        measurement_std_m,
    })
}

fn normalize(vector: &Vector3<f64>) -> Result<Vector3<f64>, String> {
    let norm = vector.norm();
    if norm == 0.0 {
        return Err("Cannot normalize a zero-length vector.".to_string());
    }
    Ok(vector / norm)
}

fn build_metrics(
    tracks: &[TrackState],
    observations: &[BearingObservation],
    rejections: &[ObservationRejection],
    truths: &[TruthState],
) -> PlatformMetrics {
    let truth_by_id = truths
        .iter()
        .map(|truth| (truth.target_id.clone(), truth))
        .collect::<HashMap<_, _>>();

    let mut errors = BTreeMap::new();
    for track in tracks {
        if let Some(truth) = truth_by_id.get(&track.track_id) {
            errors.insert(
                track.track_id.clone(),
                (track.position - truth.position).norm(),
            );
        }
    }

    let mut accepted_by_target = BTreeMap::new();
    for observation in observations {
        *accepted_by_target
            .entry(observation.target_id.clone())
            .or_insert(0) += 1;
    }

    let mut rejected_by_target = BTreeMap::new();
    let mut rejection_counts = BTreeMap::new();
    for rejection in rejections {
        *rejected_by_target
            .entry(rejection.target_id.clone())
            .or_insert(0) += 1;
        *rejection_counts
            .entry(rejection.reason.clone())
            .or_insert(0) += 1;
    }

    let mut measurement_stds = Vec::with_capacity(tracks.len());
    measurement_stds.extend(tracks.iter().map(|track| track.measurement_std_m));
    let mut error_values = Vec::with_capacity(errors.len());
    error_values.extend(errors.values().copied());

    let accepted_count = observations.len() as u32;
    let rejected_count = rejections.len() as u32;
    PlatformMetrics {
        mean_error_m: mean(&error_values),
        max_error_m: max_value(&error_values),
        active_track_count: tracks.len() as u32,
        observation_count: accepted_count,
        accepted_observation_count: accepted_count,
        rejected_observation_count: rejected_count,
        mean_measurement_std_m: mean(&measurement_stds),
        track_errors_m: errors,
        rejection_counts,
        accepted_observations_by_target: accepted_by_target,
        rejected_observations_by_target: rejected_by_target,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn max_value(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    fn bearing(
        node_id: &str,
        target_id: &str,
        origin: Vector3<f64>,
        target: Vector3<f64>,
        timestamp_s: f64,
    ) -> BearingObservation {
        let direction = (target - origin).normalize();
        BearingObservation {
            node_id: node_id.to_string(),
            target_id: target_id.to_string(),
            origin,
            direction,
            bearing_std_rad: 0.002,
            timestamp_s,
            confidence: 1.0,
        }
    }

    #[test]
    fn triangulation_recovers_truth() {
        let truth = vec3(50.0, 15.0, 10.0);
        let observations = vec![
            bearing("ground-a", "asset-a", vec3(0.0, 0.0, 0.0), truth, 0.0),
            bearing("ground-b", "asset-a", vec3(100.0, 0.0, 0.0), truth, 0.0),
        ];

        let estimate = fuse_bearing_cluster(&observations).unwrap();
        assert!((estimate.position - truth).norm() < 1.0e-6);
    }

    #[test]
    fn duplicate_node_prefers_higher_score() {
        let mut engine = TrackingEngine::new(TrackerConfig::default()).unwrap();
        engine.nodes.insert(
            "ground-a".to_string(),
            NodeState {
                node_id: "ground-a".to_string(),
                position: vec3(0.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
        );
        engine.nodes.insert(
            "ground-b".to_string(),
            NodeState {
                node_id: "ground-b".to_string(),
                position: vec3(100.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
        );

        let truth = vec3(50.0, 15.0, 10.0);
        let frame = engine.ingest_frame(
            0.0,
            Vec::new(),
            vec![
                BearingObservation {
                    confidence: 0.7,
                    ..bearing("ground-a", "asset-a", vec3(0.0, 0.0, 0.0), truth, 0.0)
                },
                BearingObservation {
                    confidence: 0.95,
                    ..bearing("ground-a", "asset-a", vec3(0.0, 0.0, 0.0), truth, 0.0)
                },
                bearing("ground-b", "asset-a", vec3(100.0, 0.0, 0.0), truth, 0.0),
            ],
            vec![TruthState {
                target_id: "asset-a".to_string(),
                position: truth,
                velocity: Vector3::zeros(),
                timestamp_s: 0.0,
            }],
        );

        assert_eq!(1, frame.tracks.len());
        assert_eq!(
            Some(&1),
            frame.metrics.rejection_counts.get(REJECT_DUPLICATE_NODE)
        );
    }

    #[test]
    fn weak_geometry_cluster_is_rejected() {
        let mut config = TrackerConfig::default();
        config.min_intersection_angle_deg = 20.0;
        let mut engine = TrackingEngine::new(config).unwrap();
        let origin = vec3(0.0, 0.0, 0.0);
        let same_dir = vec3(1.0, 0.0, 0.0);
        engine.nodes.insert(
            "a".to_string(),
            NodeState {
                node_id: "a".to_string(),
                position: origin,
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
        );
        engine.nodes.insert(
            "b".to_string(),
            NodeState {
                node_id: "b".to_string(),
                position: vec3(1.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
        );

        let frame = engine.ingest_frame(
            0.0,
            Vec::new(),
            vec![
                BearingObservation {
                    node_id: "a".to_string(),
                    target_id: "asset-a".to_string(),
                    origin,
                    direction: same_dir,
                    bearing_std_rad: 0.002,
                    timestamp_s: 0.0,
                    confidence: 1.0,
                },
                BearingObservation {
                    node_id: "b".to_string(),
                    target_id: "asset-a".to_string(),
                    origin: vec3(1.0, 0.0, 0.0),
                    direction: same_dir,
                    bearing_std_rad: 0.002,
                    timestamp_s: 0.0,
                    confidence: 1.0,
                },
            ],
            Vec::new(),
        );

        assert_eq!(0, frame.tracks.len());
        assert_eq!(
            Some(&2),
            frame.metrics.rejection_counts.get(REJECT_WEAK_GEOMETRY)
        );
    }

    #[test]
    fn stale_tracks_expire_after_limit() {
        let mut config = TrackerConfig::default();
        config.max_stale_steps = 1;
        config.max_coast_frames = 2;
        let mut engine = TrackingEngine::new(config).unwrap();
        let nodes = vec![
            NodeState {
                node_id: "ground-a".to_string(),
                position: vec3(0.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
            NodeState {
                node_id: "ground-b".to_string(),
                position: vec3(100.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s: 0.0,
                health: 1.0,
            },
        ];
        let truth = TruthState {
            target_id: "asset-a".to_string(),
            position: vec3(50.0, 15.0, 10.0),
            velocity: Vector3::zeros(),
            timestamp_s: 0.0,
        };
        let observations = vec![
            bearing(
                "ground-a",
                "asset-a",
                nodes[0].position,
                truth.position,
                0.0,
            ),
            bearing(
                "ground-b",
                "asset-a",
                nodes[1].position,
                truth.position,
                0.0,
            ),
        ];

        let first = engine.ingest_frame(0.0, nodes.clone(), observations, vec![truth.clone()]);
        assert_eq!(1, first.tracks.len());

        let second = engine.ingest_frame(1.0, Vec::new(), Vec::new(), vec![truth.clone()]);
        assert_eq!(1, second.tracks.len());
        let third = engine.ingest_frame(2.0, Vec::new(), Vec::new(), vec![truth]);
        assert_eq!(0, third.tracks.len());
    }
}
