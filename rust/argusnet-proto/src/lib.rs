use std::borrow::Cow;

use argusnet_core::{
    AssociationMode, NodeHealthSnapshot, SafetyEvent, TargetMetadata, TrackerConfig,
};
use argusnet_types::{
    BearingObservation, Matrix6, NodeState, ObservationRejection, PlatformFrame, PlatformMetrics,
    TrackState, TruthState,
};

pub mod pb {
    tonic::include_proto!("argusnet.v1");
}

fn vec3_to_pb(value: nalgebra::Vector3<f64>) -> pb::Vector3 {
    pb::Vector3 {
        x_m: value.x,
        y_m: value.y,
        z_m: value.z,
    }
}

fn vec3_from_pb(
    value: Option<pb::Vector3>,
    field_name: &str,
) -> Result<nalgebra::Vector3<f64>, String> {
    let value = value.ok_or_else(|| format!("{field_name} is required"))?;
    Ok(nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m))
}

/// For the optional geometry on `ObservationRejection`, where "absent" is a
/// value rather than an error and the caller has already matched on `Some`.
fn vec3_from_pb_infallible(value: pb::Vector3) -> nalgebra::Vector3<f64> {
    nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)
}

fn association_mode_to_string(mode: &AssociationMode) -> String {
    match mode {
        AssociationMode::Labeled => "labeled".to_string(),
        AssociationMode::GNN => "gnn".to_string(),
        AssociationMode::JPDA => "jpda".to_string(),
    }
}

fn association_mode_from_string(value: &str) -> AssociationMode {
    match value.to_lowercase().as_str() {
        "gnn" => AssociationMode::GNN,
        "jpda" => AssociationMode::JPDA,
        _ => AssociationMode::Labeled,
    }
}

pub fn tracker_config_to_pb(config: &TrackerConfig) -> pb::TrackerConfig {
    pb::TrackerConfig {
        min_observations: config.min_observations,
        max_stale_steps: config.max_stale_steps,
        retain_history: config.retain_history,
        min_confidence: config.min_confidence,
        max_bearing_std_rad: config.max_bearing_std_rad,
        max_timestamp_skew_s: config.max_timestamp_skew_s,
        min_intersection_angle_deg: config.min_intersection_angle_deg,
        data_association_mode: association_mode_to_string(&config.data_association_mode),
        cv_process_accel_std: config.cv_process_accel_std,
        ct_process_accel_std: config.ct_process_accel_std,
        ct_turn_rate_std: config.ct_turn_rate_std,
        innovation_window: config.innovation_window,
        innovation_scale_factor: config.innovation_scale_factor,
        innovation_max_scale: config.innovation_max_scale,
        adaptive_measurement_noise: config.adaptive_measurement_noise,
        chi_squared_gate_threshold: config.chi_squared_gate_threshold,
        cluster_distance_threshold_m: config.cluster_distance_threshold_m,
        near_parallel_rejection_angle_deg: config.near_parallel_rejection_angle_deg,
        confirmation_m: config.confirmation_m,
        confirmation_n: config.confirmation_n,
        max_coast_frames: config.max_coast_frames,
        max_coast_seconds: config.max_coast_seconds,
        min_quality_score: config.min_quality_score,
    }
}

pub fn tracker_config_from_pb(config: pb::TrackerConfig) -> Result<TrackerConfig, String> {
    let config = TrackerConfig {
        min_observations: config.min_observations,
        max_stale_steps: config.max_stale_steps,
        retain_history: config.retain_history,
        min_confidence: config.min_confidence,
        max_bearing_std_rad: config.max_bearing_std_rad,
        max_timestamp_skew_s: config.max_timestamp_skew_s,
        min_intersection_angle_deg: config.min_intersection_angle_deg,
        data_association_mode: association_mode_from_string(&config.data_association_mode),
        cv_process_accel_std: if config.cv_process_accel_std == 0.0 {
            TrackerConfig::default().cv_process_accel_std
        } else {
            config.cv_process_accel_std
        },
        ct_process_accel_std: if config.ct_process_accel_std == 0.0 {
            TrackerConfig::default().ct_process_accel_std
        } else {
            config.ct_process_accel_std
        },
        ct_turn_rate_std: if config.ct_turn_rate_std == 0.0 {
            TrackerConfig::default().ct_turn_rate_std
        } else {
            config.ct_turn_rate_std
        },
        innovation_window: if config.innovation_window == 0 {
            TrackerConfig::default().innovation_window
        } else {
            config.innovation_window
        },
        innovation_scale_factor: if config.innovation_scale_factor == 0.0 {
            TrackerConfig::default().innovation_scale_factor
        } else {
            config.innovation_scale_factor
        },
        innovation_max_scale: if config.innovation_max_scale == 0.0 {
            TrackerConfig::default().innovation_max_scale
        } else {
            config.innovation_max_scale
        },
        adaptive_measurement_noise: config.adaptive_measurement_noise,
        chi_squared_gate_threshold: if config.chi_squared_gate_threshold == 0.0 {
            TrackerConfig::default().chi_squared_gate_threshold
        } else {
            config.chi_squared_gate_threshold
        },
        cluster_distance_threshold_m: if config.cluster_distance_threshold_m == 0.0 {
            TrackerConfig::default().cluster_distance_threshold_m
        } else {
            config.cluster_distance_threshold_m
        },
        near_parallel_rejection_angle_deg: if config.near_parallel_rejection_angle_deg == 0.0 {
            TrackerConfig::default().near_parallel_rejection_angle_deg
        } else {
            config.near_parallel_rejection_angle_deg
        },
        confirmation_m: if config.confirmation_m == 0 {
            TrackerConfig::default().confirmation_m
        } else {
            config.confirmation_m
        },
        confirmation_n: if config.confirmation_n == 0 {
            TrackerConfig::default().confirmation_n
        } else {
            config.confirmation_n
        },
        max_coast_frames: if config.max_coast_frames == 0 {
            TrackerConfig::default().max_coast_frames
        } else {
            config.max_coast_frames
        },
        max_coast_seconds: if config.max_coast_seconds == 0.0 {
            TrackerConfig::default().max_coast_seconds
        } else {
            config.max_coast_seconds
        },
        min_quality_score: if config.min_quality_score == 0.0 {
            TrackerConfig::default().min_quality_score
        } else {
            config.min_quality_score
        },
    };
    config.validate()?;
    Ok(config)
}

// ---------------------------------------------------------------------------
// Runtime type -> wire message.
//
// Each type has exactly ONE conversion body, reached by two public entry
// points: `x_to_pb` for a caller that owns the value and `x_to_pb_ref` for one
// that only borrows it (the server keeps an `Arc<PlatformFrame>` for
// latest-frame queries and cannot move out of it).
//
// The two used to be two hand-written bodies per type, and they drifted: when
// `PlatformFrame` grew `target_metadata` and `safety_events`, and when the
// covariance transpose was corrected, each fix had to be applied twice and
// there was nothing to catch it if it were not. `Cow` collapses them.
//
// `Cow::into_owned` moves in the `Owned` case and clones the whole struct in
// the `Borrowed` case -- which is what the old `_ref` bodies already did field
// by field, since every non-`Copy` field was cloned individually. The one
// difference is at collection level: cloning a `Vec<NodeState>` allocates the
// vector's spine as well as cloning its elements, so the borrowed frame path
// pays a handful of extra allocations against the hundreds of `String` clones
// it was already making. That is the price of the field list existing once.
// ---------------------------------------------------------------------------

/// `TrackState::mode_probability_cv` is `Option<f64>` in the runtime type and a
/// bare `double` on the wire, so "unset" needs an in-band representation.
///
/// Zero is that representation. This is not a new interpretation of the field:
/// `track_from_pb` and the viewer's replay reader (`replay.rs`) both already
/// decode `0.0` as absent, and `NodeState.max_range_m` uses the same convention
/// in the same proto. Only the *encoder* disagreed -- it sent `1.0` for `None`,
/// so a `None` came back as `Some(1.0)`, silently reporting maximum confidence
/// in the constant-velocity mode where the truth was "no IMM estimate".
///
/// The cost is that a genuine `Some(0.0)` cannot be distinguished from absent.
/// That is acceptable because the only producer, `argusnet-core`'s IMM filter,
/// cannot emit it: the mixed mode probability is a normalized posterior whose
/// numerator is floored at `1.0e-30`, so it is strictly positive.
const UNSET_MODE_PROBABILITY_CV: f64 = 0.0;

pub fn node_state_to_pb(node: NodeState) -> pb::NodeState {
    node_state_pb(Cow::Owned(node))
}

pub fn node_state_to_pb_ref(node: &NodeState) -> pb::NodeState {
    node_state_pb(Cow::Borrowed(node))
}

/// In-band "unset" for the two sensor fields that have no wire presence.
///
/// `fov_half_angle_deg` and `max_range_m` are bare `double`s in the proto and
/// document `0` as unset, so the runtime `Option` has to collapse to zero on
/// the way out and re-expand on the way in. `battery_fraction` deliberately
/// does *not* use this: it is `optional` on the wire precisely because a flat
/// battery is a real reading that zero would swallow.
const UNSET_SENSOR_FIELD: f64 = 0.0;

fn node_state_pb(node: Cow<'_, NodeState>) -> pb::NodeState {
    let node = node.into_owned();
    pb::NodeState {
        node_id: node.node_id,
        position: Some(vec3_to_pb(node.position)),
        velocity: Some(vec3_to_pb(node.velocity)),
        is_mobile: node.is_mobile,
        timestamp_s: node.timestamp_s,
        health: node.health,
        sensor_type: node.sensor_type,
        fov_half_angle_deg: node.fov_half_angle_deg.unwrap_or(UNSET_SENSOR_FIELD),
        max_range_m: node.max_range_m.unwrap_or(UNSET_SENSOR_FIELD),
        battery_fraction: node.battery_fraction,
        // Signed-envelope identity is verified and stripped at the Python
        // ingestion boundary before frames reach the core tracker; it is
        // never reconstructed on the way back out.
        device_id: String::new(),
        sequence: 0,
        signature: Vec::new(),
        signer_pubkey_id: String::new(),
    }
}

/// Re-expand a wire field that encodes "unset" as zero.
fn optional_sensor_field(value: f64) -> Option<f64> {
    if value == UNSET_SENSOR_FIELD {
        None
    } else {
        Some(value)
    }
}

pub fn node_state_from_pb(node: pb::NodeState) -> Result<NodeState, String> {
    // An empty `sensor_type` is what a producer that never set the field
    // sends -- proto3 strings have no presence -- so it means "unspecified",
    // not "a sensor whose name is the empty string".
    let sensor_type = if node.sensor_type.is_empty() {
        argusnet_types::DEFAULT_SENSOR_TYPE.to_string()
    } else {
        node.sensor_type
    };
    Ok(NodeState {
        node_id: node.node_id,
        position: vec3_from_pb(node.position, "node.position")?,
        velocity: vec3_from_pb(node.velocity, "node.velocity")?,
        is_mobile: node.is_mobile,
        timestamp_s: node.timestamp_s,
        health: node.health,
        sensor_type,
        fov_half_angle_deg: optional_sensor_field(node.fov_half_angle_deg),
        max_range_m: optional_sensor_field(node.max_range_m),
        battery_fraction: node.battery_fraction,
    })
}

pub fn observation_to_pb(observation: BearingObservation) -> pb::BearingObservation {
    observation_pb(Cow::Owned(observation))
}

pub fn observation_to_pb_ref(observation: &BearingObservation) -> pb::BearingObservation {
    observation_pb(Cow::Borrowed(observation))
}

fn observation_pb(observation: Cow<'_, BearingObservation>) -> pb::BearingObservation {
    let observation = observation.into_owned();
    pb::BearingObservation {
        node_id: observation.node_id,
        target_id: observation.target_id,
        origin: Some(vec3_to_pb(observation.origin)),
        direction: Some(vec3_to_pb(observation.direction)),
        bearing_std_rad: observation.bearing_std_rad,
        timestamp_s: observation.timestamp_s,
        confidence: observation.confidence,
        device_id: String::new(),
        sequence: 0,
        signature: Vec::new(),
        signer_pubkey_id: String::new(),
    }
}

pub fn observation_from_pb(
    observation: pb::BearingObservation,
) -> Result<BearingObservation, String> {
    Ok(BearingObservation {
        node_id: observation.node_id,
        target_id: observation.target_id,
        origin: vec3_from_pb(observation.origin, "observation.origin")?,
        direction: vec3_from_pb(observation.direction, "observation.direction")?,
        bearing_std_rad: observation.bearing_std_rad,
        timestamp_s: observation.timestamp_s,
        confidence: observation.confidence,
    })
}

pub fn rejection_to_pb(rejection: ObservationRejection) -> pb::ObservationRejection {
    rejection_pb(Cow::Owned(rejection))
}

pub fn rejection_to_pb_ref(rejection: &ObservationRejection) -> pb::ObservationRejection {
    rejection_pb(Cow::Borrowed(rejection))
}

fn rejection_pb(rejection: Cow<'_, ObservationRejection>) -> pb::ObservationRejection {
    let rejection = rejection.into_owned();
    pb::ObservationRejection {
        node_id: rejection.node_id,
        target_id: rejection.target_id,
        timestamp_s: rejection.timestamp_s,
        reason: rejection.reason,
        detail: rejection.detail,
        origin: rejection.origin.map(vec3_to_pb),
        attempted_point: rejection.attempted_point.map(vec3_to_pb),
        closest_point: rejection.closest_point.map(vec3_to_pb),
        blocker_type: rejection.blocker_type,
        first_hit_range_m: rejection.first_hit_range_m,
    }
}

/// The inverse of [`rejection_to_pb`].
///
/// Infallible where the others are not: every geometry field on
/// `ObservationRejection` is already optional, so a missing one is a legitimate
/// value rather than a malformed message. `frame_from_pb` used to inline this
/// body twice -- once for `rejected_observations` and once for
/// `generation_rejections` -- which made a third copy of the field list.
pub fn rejection_from_pb(rejection: pb::ObservationRejection) -> ObservationRejection {
    ObservationRejection {
        node_id: rejection.node_id,
        target_id: rejection.target_id,
        timestamp_s: rejection.timestamp_s,
        reason: rejection.reason,
        detail: rejection.detail,
        origin: rejection.origin.map(vec3_from_pb_infallible),
        attempted_point: rejection.attempted_point.map(vec3_from_pb_infallible),
        closest_point: rejection.closest_point.map(vec3_from_pb_infallible),
        blocker_type: rejection.blocker_type,
        first_hit_range_m: rejection.first_hit_range_m,
    }
}

pub fn truth_to_pb(truth: TruthState) -> pb::TruthState {
    truth_pb(Cow::Owned(truth))
}

pub fn truth_to_pb_ref(truth: &TruthState) -> pb::TruthState {
    truth_pb(Cow::Borrowed(truth))
}

fn truth_pb(truth: Cow<'_, TruthState>) -> pb::TruthState {
    let truth = truth.into_owned();
    pb::TruthState {
        target_id: truth.target_id,
        position: Some(vec3_to_pb(truth.position)),
        velocity: Some(vec3_to_pb(truth.velocity)),
        timestamp_s: truth.timestamp_s,
    }
}

pub fn truth_from_pb(truth: pb::TruthState) -> Result<TruthState, String> {
    Ok(TruthState {
        target_id: truth.target_id,
        position: vec3_from_pb(truth.position, "truth.position")?,
        velocity: vec3_from_pb(truth.velocity, "truth.velocity")?,
        timestamp_s: truth.timestamp_s,
    })
}

fn covariance_to_vec(matrix: Matrix6) -> Vec<f64> {
    // The wire field is `covariance_row_major`, and every reader honours that:
    // `covariance_from_vec` uses `from_row_slice`, and the Python adapter uses
    // numpy's C-order `reshape(6, 6)`. nalgebra stores column-major, so
    // `matrix.iter()` would emit the transpose under a row-major name.
    //
    // A covariance matrix is symmetric, so this was invisible in practice --
    // but the Kalman update never explicitly symmetrizes, so rounding leaves
    // small asymmetries, and any genuinely non-symmetric matrix would have been
    // corrupted silently. Transposing first yields true row-major order.
    matrix.transpose().iter().copied().collect()
}

fn covariance_from_vec(values: Vec<f64>) -> Result<Matrix6, String> {
    if values.len() != 36 {
        return Err(format!(
            "track covariance must contain 36 row-major values, got {}",
            values.len()
        ));
    }
    Ok(Matrix6::from_row_slice(&values))
}

pub fn track_to_pb(track: TrackState) -> pb::TrackState {
    track_pb(Cow::Owned(track))
}

pub fn track_to_pb_ref(track: &TrackState) -> pb::TrackState {
    track_pb(Cow::Borrowed(track))
}

fn track_pb(track: Cow<'_, TrackState>) -> pb::TrackState {
    let track = track.into_owned();
    pb::TrackState {
        track_id: track.track_id,
        timestamp_s: track.timestamp_s,
        position: Some(vec3_to_pb(track.position)),
        velocity: Some(vec3_to_pb(track.velocity)),
        covariance_row_major: covariance_to_vec(track.covariance),
        measurement_std_m: track.measurement_std_m,
        update_count: track.update_count,
        stale_steps: track.stale_steps,
        lifecycle_state: track.lifecycle_state,
        quality_score: track.quality_score,
        // Wire-only fields the runtime type does not carry. FIELD_PARITY.md
        // records `acceleration_mps2` as `inert` and `confidence` /
        // `last_seen_s` as `derived` for exactly this reason: they are written
        // here and never read back, so they carry no information of their own.
        acceleration_mps2: vec![0.0, 0.0, 0.0],
        confidence: track.quality_score.unwrap_or(0.0),
        last_seen_s: track.timestamp_s,
        mode_probability_cv: track
            .mode_probability_cv
            .unwrap_or(UNSET_MODE_PROBABILITY_CV),
        // Move out of the Arc when this TrackState holds the last reference;
        // otherwise fall back to cloning the shared list.
        contributing_nodes: std::sync::Arc::try_unwrap(track.contributing_node_ids)
            .unwrap_or_else(|shared| (*shared).clone()),
    }
}

pub fn track_from_pb(track: pb::TrackState) -> Result<TrackState, String> {
    Ok(TrackState {
        track_id: track.track_id,
        timestamp_s: track.timestamp_s,
        position: vec3_from_pb(track.position, "track.position")?,
        velocity: vec3_from_pb(track.velocity, "track.velocity")?,
        covariance: covariance_from_vec(track.covariance_row_major)?,
        measurement_std_m: track.measurement_std_m,
        update_count: track.update_count,
        stale_steps: track.stale_steps,
        lifecycle_state: track.lifecycle_state,
        quality_score: track.quality_score,
        mode_probability_cv: if track.mode_probability_cv == UNSET_MODE_PROBABILITY_CV {
            None
        } else {
            Some(track.mode_probability_cv)
        },
        contributing_node_ids: std::sync::Arc::new(track.contributing_nodes),
    })
}

pub fn metrics_to_pb(metrics: PlatformMetrics) -> pb::PlatformMetrics {
    metrics_pb(Cow::Owned(metrics))
}

pub fn metrics_to_pb_ref(metrics: &PlatformMetrics) -> pb::PlatformMetrics {
    metrics_pb(Cow::Borrowed(metrics))
}

fn metrics_pb(metrics: Cow<'_, PlatformMetrics>) -> pb::PlatformMetrics {
    let metrics = metrics.into_owned();
    pb::PlatformMetrics {
        mean_error_m: metrics.mean_error_m,
        max_error_m: metrics.max_error_m,
        active_track_count: metrics.active_track_count,
        observation_count: metrics.observation_count,
        accepted_observation_count: metrics.accepted_observation_count,
        rejected_observation_count: metrics.rejected_observation_count,
        mean_measurement_std_m: metrics.mean_measurement_std_m,
        track_errors_m: metrics.track_errors_m.into_iter().collect(),
        rejection_counts: metrics.rejection_counts.into_iter().collect(),
        accepted_observations_by_target: metrics
            .accepted_observations_by_target
            .into_iter()
            .collect(),
        rejected_observations_by_target: metrics
            .rejected_observations_by_target
            .into_iter()
            .collect(),
    }
}

pub fn metrics_from_pb(metrics: pb::PlatformMetrics) -> PlatformMetrics {
    PlatformMetrics {
        mean_error_m: metrics.mean_error_m,
        max_error_m: metrics.max_error_m,
        active_track_count: metrics.active_track_count,
        observation_count: metrics.observation_count,
        accepted_observation_count: metrics.accepted_observation_count,
        rejected_observation_count: metrics.rejected_observation_count,
        mean_measurement_std_m: metrics.mean_measurement_std_m,
        track_errors_m: metrics.track_errors_m.into_iter().collect(),
        rejection_counts: metrics.rejection_counts.into_iter().collect(),
        accepted_observations_by_target: metrics
            .accepted_observations_by_target
            .into_iter()
            .collect(),
        rejected_observations_by_target: metrics
            .rejected_observations_by_target
            .into_iter()
            .collect(),
    }
}

fn target_metadata_to_pb(value: TargetMetadata) -> pb::TargetMetadata {
    pb::TargetMetadata {
        target_id: value.target_id,
        operator_label: value.operator_label,
        target_type: value.target_type,
        priority: value.priority,
        behavior: value.behavior,
        status: value.status,
        first_seen_s: value.first_seen_s,
        last_seen_s: value.last_seen_s,
        confidence: value.confidence,
        notes: value.notes,
    }
}

pub fn target_metadata_from_pb(value: pb::TargetMetadata) -> TargetMetadata {
    TargetMetadata {
        target_id: value.target_id,
        operator_label: value.operator_label,
        target_type: value.target_type,
        priority: value.priority,
        behavior: value.behavior,
        status: value.status,
        first_seen_s: value.first_seen_s,
        last_seen_s: value.last_seen_s,
        confidence: value.confidence,
        notes: value.notes,
    }
}

fn safety_event_to_pb(value: SafetyEvent) -> pb::SafetyEvent {
    pb::SafetyEvent {
        event_id: value.event_id,
        subject_id: value.subject_id,
        timestamp_s: value.timestamp_s,
        state: value.state,
        violations: value.violations,
        command_clamped: value.command_clamped,
        blocked: value.blocked,
        reason: value.reason,
    }
}

pub fn safety_event_from_pb(value: pb::SafetyEvent) -> SafetyEvent {
    SafetyEvent {
        event_id: value.event_id,
        subject_id: value.subject_id,
        timestamp_s: value.timestamp_s,
        state: value.state,
        violations: value.violations,
        command_clamped: value.command_clamped,
        blocked: value.blocked,
        reason: value.reason,
    }
}

pub fn frame_to_pb(frame: PlatformFrame) -> pb::PlatformFrame {
    frame_pb(Cow::Owned(frame))
}

pub fn frame_to_pb_ref(frame: &PlatformFrame) -> pb::PlatformFrame {
    frame_pb(Cow::Borrowed(frame))
}

fn frame_pb(frame: Cow<'_, PlatformFrame>) -> pb::PlatformFrame {
    let frame = frame.into_owned();
    pb::PlatformFrame {
        timestamp_s: frame.timestamp_s,
        nodes: frame.nodes.into_iter().map(node_state_to_pb).collect(),
        observations: frame
            .observations
            .into_iter()
            .map(observation_to_pb)
            .collect(),
        rejected_observations: frame
            .rejected_observations
            .into_iter()
            .map(rejection_to_pb)
            .collect(),
        tracks: frame.tracks.into_iter().map(track_to_pb).collect(),
        truths: frame.truths.into_iter().map(truth_to_pb).collect(),
        metrics: Some(metrics_to_pb(frame.metrics)),
        generation_rejections: frame
            .generation_rejections
            .into_iter()
            .map(rejection_to_pb)
            .collect(),
        target_metadata: frame
            .target_metadata
            .into_iter()
            .map(target_metadata_to_pb)
            .collect(),
        safety_events: frame
            .safety_events
            .into_iter()
            .map(safety_event_to_pb)
            .collect(),
    }
}

pub fn node_health_to_pb(snapshot: &NodeHealthSnapshot) -> pb::NodeHealthMetrics {
    pb::NodeHealthMetrics {
        node_id: snapshot.node_id.clone(),
        last_seen_s: snapshot.last_seen_s,
        observation_rate_hz: snapshot.observation_rate_hz,
        mean_latency_s: 0.0,
        accepted_count: snapshot.accepted_count,
        rejected_count: snapshot.rejected_count,
        health_score: snapshot.health_score,
    }
}

pub fn frame_from_pb(frame: pb::PlatformFrame) -> Result<PlatformFrame, String> {
    Ok(PlatformFrame {
        timestamp_s: frame.timestamp_s,
        nodes: frame
            .nodes
            .into_iter()
            .map(node_state_from_pb)
            .collect::<Result<Vec<_>, _>>()?,
        observations: frame
            .observations
            .into_iter()
            .map(observation_from_pb)
            .collect::<Result<Vec<_>, _>>()?,
        rejected_observations: frame
            .rejected_observations
            .into_iter()
            .map(rejection_from_pb)
            .collect(),
        tracks: frame
            .tracks
            .into_iter()
            .map(track_from_pb)
            .collect::<Result<Vec<_>, _>>()?,
        truths: frame
            .truths
            .into_iter()
            .map(truth_from_pb)
            .collect::<Result<Vec<_>, _>>()?,
        metrics: metrics_from_pb(frame.metrics.unwrap_or_default()),
        generation_rejections: frame
            .generation_rejections
            .into_iter()
            .map(rejection_from_pb)
            .collect(),
        target_metadata: frame
            .target_metadata
            .into_iter()
            .map(target_metadata_from_pb)
            .collect(),
        safety_events: frame
            .safety_events
            .into_iter()
            .map(safety_event_from_pb)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_core::PlatformMetrics;
    use nalgebra::Vector3;

    #[test]
    fn rejection_round_trip_preserves_optional_geometry() {
        let rejection = ObservationRejection {
            node_id: "node-1".into(),
            target_id: "asset-a".into(),
            timestamp_s: 12.0,
            reason: "terrain_occlusion".into(),
            detail: "blocked".into(),
            origin: Some(Vector3::new(1.0, 2.0, 3.0)),
            attempted_point: Some(Vector3::new(4.0, 5.0, 6.0)),
            closest_point: Some(Vector3::new(7.0, 8.0, 9.0)),
            blocker_type: "terrain".into(),
            first_hit_range_m: Some(25.0),
        };

        let pb = rejection_to_pb(rejection.clone());
        let frame = frame_from_pb(pb::PlatformFrame {
            timestamp_s: 0.0,
            nodes: Vec::new(),
            observations: Vec::new(),
            rejected_observations: vec![pb],
            tracks: Vec::new(),
            truths: Vec::new(),
            metrics: Some(metrics_to_pb(PlatformMetrics {
                mean_error_m: None,
                max_error_m: None,
                active_track_count: 0,
                observation_count: 0,
                accepted_observation_count: 0,
                rejected_observation_count: 0,
                mean_measurement_std_m: None,
                track_errors_m: Default::default(),
                rejection_counts: Default::default(),
                accepted_observations_by_target: Default::default(),
                rejected_observations_by_target: Default::default(),
            })),
            generation_rejections: Vec::new(),
            target_metadata: Vec::new(),
            safety_events: Vec::new(),
        })
        .expect("frame");

        assert_eq!(frame.rejected_observations[0], rejection);
        assert!(frame.generation_rejections.is_empty());
    }

    #[test]
    fn frame_from_pb_defaults_generation_rejections_to_empty() {
        let frame = frame_from_pb(pb::PlatformFrame {
            timestamp_s: 0.0,
            nodes: Vec::new(),
            observations: Vec::new(),
            rejected_observations: Vec::new(),
            tracks: Vec::new(),
            truths: Vec::new(),
            metrics: None,
            generation_rejections: Vec::new(),
            target_metadata: Vec::new(),
            safety_events: Vec::new(),
        })
        .expect("frame");

        assert!(frame.generation_rejections.is_empty());
    }
}
