use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use tracker_proto::pb::tracker_service_client::TrackerServiceClient;
use tracker_proto::pb::{
    BearingObservation, GetConfigRequest, HealthRequest, IngestFrameRequest, LatestFrameRequest,
    NodeState, ResetRequest, TruthState, Vector3,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("repo root")
        .to_path_buf()
}

fn allocate_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test port");
    let address = listener.local_addr().expect("local addr");
    drop(listener);
    address.to_string()
}

fn fixture_path() -> PathBuf {
    repo_root().join("tests/fixtures/runtime_parity_fixture.json")
}

fn vector_from_json(value: &Value) -> Vector3 {
    let values = value.as_array().expect("vector");
    Vector3 {
        x_m: values[0].as_f64().expect("x"),
        y_m: values[1].as_f64().expect("y"),
        z_m: values[2].as_f64().expect("z"),
    }
}

fn node_from_json(value: &Value) -> NodeState {
    NodeState {
        node_id: value["node_id"].as_str().expect("node_id").to_string(),
        position: Some(vector_from_json(&value["position"])),
        velocity: Some(vector_from_json(&value["velocity"])),
        is_mobile: value["is_mobile"].as_bool().expect("is_mobile"),
        timestamp_s: value["timestamp_s"].as_f64().expect("timestamp_s"),
        health: value["health"].as_f64().expect("health"),
    }
}

fn observation_from_json(value: &Value) -> BearingObservation {
    BearingObservation {
        node_id: value["node_id"].as_str().expect("node_id").to_string(),
        target_id: value["target_id"].as_str().expect("target_id").to_string(),
        origin: Some(vector_from_json(&value["origin"])),
        direction: Some(vector_from_json(&value["direction"])),
        bearing_std_rad: value["bearing_std_rad"].as_f64().expect("bearing_std_rad"),
        timestamp_s: value["timestamp_s"].as_f64().expect("timestamp_s"),
        confidence: value["confidence"].as_f64().expect("confidence"),
    }
}

fn truth_from_json(value: &Value) -> TruthState {
    TruthState {
        target_id: value["target_id"].as_str().expect("target_id").to_string(),
        position: Some(vector_from_json(&value["position"])),
        velocity: Some(vector_from_json(&value["velocity"])),
        timestamp_s: value["timestamp_s"].as_f64().expect("timestamp_s"),
    }
}

fn request_from_json(value: &Value) -> IngestFrameRequest {
    IngestFrameRequest {
        timestamp_s: value["timestamp_s"].as_f64().expect("timestamp_s"),
        node_states: value["node_states"]
            .as_array()
            .expect("node_states")
            .iter()
            .map(node_from_json)
            .collect(),
        observations: value["observations"]
            .as_array()
            .expect("observations")
            .iter()
            .map(observation_from_json)
            .collect(),
        truths: value["truths"]
            .as_array()
            .expect("truths")
            .iter()
            .map(truth_from_json)
            .collect(),
    }
}

fn vector_to_json(value: &Vector3) -> Value {
    json!([value.x_m, value.y_m, value.z_m])
}

fn message_metrics_to_json(metrics: &tracker_proto::pb::PlatformMetrics) -> Value {
    json!({
        "mean_error_m": metrics.mean_error_m,
        "max_error_m": metrics.max_error_m,
        "active_track_count": metrics.active_track_count,
        "observation_count": metrics.observation_count,
        "accepted_observation_count": metrics.accepted_observation_count,
        "rejected_observation_count": metrics.rejected_observation_count,
        "mean_measurement_std_m": metrics.mean_measurement_std_m,
        "track_errors_m": metrics.track_errors_m,
        "rejection_counts": metrics.rejection_counts,
        "accepted_observations_by_target": metrics.accepted_observations_by_target,
        "rejected_observations_by_target": metrics.rejected_observations_by_target,
    })
}

fn frame_to_json(frame: &tracker_proto::pb::PlatformFrame) -> Value {
    let nodes = frame
        .nodes
        .iter()
        .map(|node| {
            json!({
                "node_id": node.node_id,
                "position": vector_to_json(node.position.as_ref().expect("position")),
                "velocity": vector_to_json(node.velocity.as_ref().expect("velocity")),
                "is_mobile": node.is_mobile,
                "timestamp_s": node.timestamp_s,
                "health": node.health,
                "sensor_type": "optical",
                "fov_half_angle_deg": 180.0,
                "max_range_m": 0.0,
            })
        })
        .collect::<Vec<_>>();
    let observations = frame
        .observations
        .iter()
        .map(|observation| {
            json!({
                "node_id": observation.node_id,
                "target_id": observation.target_id,
                "origin": vector_to_json(observation.origin.as_ref().expect("origin")),
                "direction": vector_to_json(observation.direction.as_ref().expect("direction")),
                "bearing_std_rad": observation.bearing_std_rad,
                "timestamp_s": observation.timestamp_s,
                "confidence": observation.confidence,
            })
        })
        .collect::<Vec<_>>();
    let rejected_observations = frame
        .rejected_observations
        .iter()
        .map(|rejection| {
            json!({
                "node_id": rejection.node_id,
                "target_id": rejection.target_id,
                "timestamp_s": rejection.timestamp_s,
                "reason": rejection.reason,
                "detail": rejection.detail,
                "origin": rejection.origin.as_ref().map(vector_to_json),
                "attempted_point": rejection.attempted_point.as_ref().map(vector_to_json),
                "closest_point": rejection.closest_point.as_ref().map(vector_to_json),
                "blocker_type": rejection.blocker_type,
                "first_hit_range_m": rejection.first_hit_range_m,
            })
        })
        .collect::<Vec<_>>();
    let generation_rejections = frame
        .generation_rejections
        .iter()
        .map(|rejection| {
            json!({
                "node_id": rejection.node_id,
                "target_id": rejection.target_id,
                "timestamp_s": rejection.timestamp_s,
                "reason": rejection.reason,
                "detail": rejection.detail,
                "origin": rejection.origin.as_ref().map(vector_to_json),
                "attempted_point": rejection.attempted_point.as_ref().map(vector_to_json),
                "closest_point": rejection.closest_point.as_ref().map(vector_to_json),
                "blocker_type": rejection.blocker_type,
                "first_hit_range_m": rejection.first_hit_range_m,
            })
        })
        .collect::<Vec<_>>();
    let tracks = frame
        .tracks
        .iter()
        .map(|track| {
            let covariance = track
                .covariance_row_major
                .iter()
                .map(|value| json!(value))
                .collect::<Vec<_>>();
            json!({
                "track_id": track.track_id,
                "timestamp_s": track.timestamp_s,
                "position": vector_to_json(track.position.as_ref().expect("position")),
                "velocity": vector_to_json(track.velocity.as_ref().expect("velocity")),
                "covariance": covariance,
                "measurement_std_m": track.measurement_std_m,
                "update_count": track.update_count,
                "stale_steps": track.stale_steps,
                "lifecycle_state": track.lifecycle_state,
                "quality_score": track.quality_score,
            })
        })
        .collect::<Vec<_>>();
    let truths = frame
        .truths
        .iter()
        .map(|truth| {
            json!({
                "target_id": truth.target_id,
                "position": vector_to_json(truth.position.as_ref().expect("position")),
                "velocity": vector_to_json(truth.velocity.as_ref().expect("velocity")),
                "timestamp_s": truth.timestamp_s,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "timestamp_s": frame.timestamp_s,
        "nodes": nodes,
        "observations": observations,
        "rejected_observations": rejected_observations,
        "generation_rejections": generation_rejections,
        "tracks": tracks,
        "truths": truths,
        "metrics": message_metrics_to_json(frame.metrics.as_ref().expect("metrics")),
        "launch_events": [],
    })
}

fn assert_json_close(expected: &Value, actual: &Value) {
    match (expected, actual) {
        (Value::Null, Value::Null) => {}
        (Value::Bool(a), Value::Bool(b)) => assert_eq!(a, b),
        (Value::String(a), Value::String(b)) => assert_eq!(a, b),
        (Value::Number(a), Value::Number(b)) => {
            let lhs = a.as_f64().expect("lhs float");
            let rhs = b.as_f64().expect("rhs float");
            assert!(
                (lhs - rhs).abs() < 5.0e-4,
                "float mismatch: expected {lhs}, got {rhs}"
            );
        }
        (Value::Array(expected_items), Value::Array(actual_items)) => {
            assert_eq!(
                expected_items.len(),
                actual_items.len(),
                "array length mismatch"
            );
            for (expected_item, actual_item) in expected_items.iter().zip(actual_items.iter()) {
                assert_json_close(expected_item, actual_item);
            }
        }
        (Value::Object(expected_map), Value::Object(actual_map)) => {
            assert_eq!(
                expected_map.len(),
                actual_map.len(),
                "object length mismatch"
            );
            for (key, expected_value) in expected_map {
                let actual_value = actual_map
                    .get(key)
                    .unwrap_or_else(|| panic!("missing key {key}"));
                assert_json_close(expected_value, actual_value);
            }
        }
        _ => panic!("shape mismatch: expected {expected:?}, got {actual:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_daemon_matches_golden_fixture_and_streaming() {
    let fixture: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path()).expect("fixture"))
            .expect("parse fixture");
    let tracker_config = &fixture["tracker_config"];
    let config_path = repo_root().join("target/runtime-parity-config.yaml");
    fs::write(
        &config_path,
        format!(
            "min_observations: {}\nmax_stale_steps: {}\nretain_history: false\nmin_confidence: {}\nmax_bearing_std_rad: {}\nmax_timestamp_skew_s: {}\nmin_intersection_angle_deg: {}\n",
            tracker_config["min_observations"].as_u64().expect("min_observations"),
            tracker_config["max_stale_steps"].as_u64().expect("max_stale_steps"),
            tracker_config["min_confidence"].as_f64().expect("min_confidence"),
            tracker_config["max_bearing_std_rad"].as_f64().expect("max_bearing_std_rad"),
            tracker_config["max_timestamp_skew_s"].as_f64().expect("max_timestamp_skew_s"),
            tracker_config["min_intersection_angle_deg"].as_f64().expect("min_intersection_angle_deg"),
        ),
    )
    .expect("write config");

    let address = allocate_endpoint();
    let server_handle = tokio::spawn(tracker_server::serve(tracker_server::ServeArgs {
        listen: address.clone(),
        config: Some(config_path.clone()),
        min_observations: None,
        max_stale_steps: None,
        min_confidence: None,
        max_bearing_std_rad: None,
        max_timestamp_skew_s: None,
        min_intersection_angle_deg: None,
        data_association_mode: None,
        cv_process_accel_std: None,
        ct_process_accel_std: None,
        ct_turn_rate_std: None,
        innovation_window: None,
        innovation_scale_factor: None,
        innovation_max_scale: None,
        adaptive_measurement_noise: None,
        chi_squared_gate_threshold: None,
        cluster_distance_threshold_m: None,
        near_parallel_rejection_angle_deg: None,
        confirmation_m: None,
        confirmation_n: None,
        max_coast_frames: None,
        max_coast_seconds: None,
        min_quality_score: None,
    }));

    let mut client = loop {
        match TrackerServiceClient::connect(format!("http://{address}")).await {
            Ok(client) => break client,
            Err(_) => sleep(Duration::from_millis(100)).await,
        }
    };

    let health = client
        .health(HealthRequest {})
        .await
        .expect("health")
        .into_inner();
    assert_eq!("SERVING", health.status);

    let requests = fixture["requests"].as_array().expect("requests");
    let expected_responses = fixture["responses"].as_array().expect("responses");
    let unary_requests = requests.iter().map(request_from_json).collect::<Vec<_>>();
    for (request, expected_response) in unary_requests
        .iter()
        .cloned()
        .zip(expected_responses.iter())
    {
        let response = client
            .ingest_frame(request)
            .await
            .expect("unary ingest")
            .into_inner();
        let actual = frame_to_json(response.frame.as_ref().expect("frame"));
        assert_json_close(expected_response, &actual);
    }

    let latest = client
        .latest_frame(LatestFrameRequest {})
        .await
        .expect("latest")
        .into_inner();
    assert!(latest.frame.is_some());

    client.reset(ResetRequest {}).await.expect("reset");
    let latest_after_reset = client
        .latest_frame(LatestFrameRequest {})
        .await
        .expect("latest after reset")
        .into_inner();
    assert!(latest_after_reset.frame.is_none());

    let request_stream = tokio_stream::iter(unary_requests.clone());
    let mut stream = client
        .track_stream(request_stream)
        .await
        .expect("track stream")
        .into_inner();
    for expected_response in expected_responses {
        let response = stream
            .message()
            .await
            .expect("stream message status")
            .expect("stream message");
        let actual = frame_to_json(response.frame.as_ref().expect("frame"));
        assert_json_close(expected_response, &actual);
    }

    let config = client
        .get_config(GetConfigRequest {})
        .await
        .expect("config")
        .into_inner();
    assert_eq!(
        tracker_config["min_observations"]
            .as_u64()
            .expect("min_observations"),
        u64::from(config.config.expect("config").min_observations)
    );

    server_handle.abort();
    let _ = fs::remove_file(config_path);
}
