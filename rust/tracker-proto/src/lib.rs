use tracker_core::{
    AssociationMode, BearingObservation, Matrix6, NodeHealthSnapshot, NodeState,
    ObservationRejection, PlatformFrame, PlatformMetrics, TrackState, TrackerConfig, TruthState,
};

pub mod pb {
    tonic::include_proto!("smarttracker.v1");
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

pub fn node_state_to_pb(node: NodeState) -> pb::NodeState {
    pb::NodeState {
        node_id: node.node_id,
        position: Some(vec3_to_pb(node.position)),
        velocity: Some(vec3_to_pb(node.velocity)),
        is_mobile: node.is_mobile,
        timestamp_s: node.timestamp_s,
        health: node.health,
    }
}

pub fn node_state_from_pb(node: pb::NodeState) -> Result<NodeState, String> {
    Ok(NodeState {
        node_id: node.node_id,
        position: vec3_from_pb(node.position, "node.position")?,
        velocity: vec3_from_pb(node.velocity, "node.velocity")?,
        is_mobile: node.is_mobile,
        timestamp_s: node.timestamp_s,
        health: node.health,
    })
}

pub fn observation_to_pb(observation: BearingObservation) -> pb::BearingObservation {
    pb::BearingObservation {
        node_id: observation.node_id,
        target_id: observation.target_id,
        origin: Some(vec3_to_pb(observation.origin)),
        direction: Some(vec3_to_pb(observation.direction)),
        bearing_std_rad: observation.bearing_std_rad,
        timestamp_s: observation.timestamp_s,
        confidence: observation.confidence,
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

pub fn truth_to_pb(truth: TruthState) -> pb::TruthState {
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
    matrix.iter().copied().collect()
}

fn covariance_from_vec(values: Vec<f64>) -> Result<Matrix6, String> {
    if values.len() != 36 {
        return Err("track covariance must contain 36 row-major values".to_string());
    }
    Ok(Matrix6::from_row_slice(&values))
}

pub fn track_to_pb(track: TrackState) -> pb::TrackState {
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
        // FusedTrack extension fields — populated with defaults until
        // tracker-core produces them.
        acceleration_mps2: vec![0.0, 0.0, 0.0],
        confidence: track.quality_score.unwrap_or(0.0),
        mode_probability_cv: 1.0,
        last_seen_s: track.timestamp_s,
        contributing_nodes: vec![],
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
    })
}

pub fn metrics_to_pb(metrics: PlatformMetrics) -> pb::PlatformMetrics {
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

pub fn frame_to_pb(frame: PlatformFrame) -> pb::PlatformFrame {
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
            .map(|rejection| ObservationRejection {
                node_id: rejection.node_id,
                target_id: rejection.target_id,
                timestamp_s: rejection.timestamp_s,
                reason: rejection.reason,
                detail: rejection.detail,
                origin: rejection
                    .origin
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                attempted_point: rejection
                    .attempted_point
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                closest_point: rejection
                    .closest_point
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                blocker_type: rejection.blocker_type,
                first_hit_range_m: rejection.first_hit_range_m,
            })
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
            .map(|rejection| ObservationRejection {
                node_id: rejection.node_id,
                target_id: rejection.target_id,
                timestamp_s: rejection.timestamp_s,
                reason: rejection.reason,
                detail: rejection.detail,
                origin: rejection
                    .origin
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                attempted_point: rejection
                    .attempted_point
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                closest_point: rejection
                    .closest_point
                    .map(|value| nalgebra::Vector3::new(value.x_m, value.y_m, value.z_m)),
                blocker_type: rejection.blocker_type,
                first_hit_range_m: rejection.first_hit_range_m,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use tracker_core::PlatformMetrics;

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
        })
        .expect("frame");

        assert!(frame.generation_rejections.is_empty());
    }
}
