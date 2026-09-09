//! All-field round trips for the proto <-> runtime conversions.
//!
//! The G0 field matrix records what is *supposed* to happen to each wire field,
//! and it was built partly by searching for field names. That is enough to
//! catch a field nobody handles, but it cannot tell `TrackState.confidence`
//! from `BearingObservation.confidence`, and it cannot see that a field is
//! assigned a constant rather than carried. These tests close that gap by
//! moving values through the conversions and comparing whole structs, so a
//! dropped or defaulted field fails without anyone having to list it.
//!
//! Every value below is deliberately non-default: a field that is silently
//! replaced by `0.0`, `""` or `None` shows up as a mismatch rather than
//! coinciding with the value it was meant to carry.

use argusnet_types::{
    BearingObservation, Matrix6, NodeState, PlatformFrame, PlatformMetrics, SafetyEvent,
    TargetMetadata, TrackState, TruthState,
};
use nalgebra::Vector3;
use std::collections::BTreeMap;
use std::sync::Arc;

fn distinctive_covariance() -> Matrix6 {
    // Every entry distinct, so a transpose or a row/column mix-up in the
    // row-major flattening cannot survive.
    Matrix6::from_fn(|row, column| (row * 6 + column) as f64 + 0.5)
}

#[test]
fn node_state_round_trips_every_field() {
    let original = NodeState {
        node_id: "node-alpha".to_string(),
        position: Vector3::new(1.5, -2.5, 3.5),
        velocity: Vector3::new(-4.5, 5.5, -6.5),
        is_mobile: true,
        timestamp_s: 12.25,
        health: 0.75,
    };
    let restored =
        argusnet_proto::node_state_from_pb(argusnet_proto::node_state_to_pb(original.clone()))
            .expect("node state round trip");
    assert_eq!(restored, original);
}

#[test]
fn observation_round_trips_every_field() {
    let original = BearingObservation {
        node_id: "node-bravo".to_string(),
        target_id: "target-7".to_string(),
        origin: Vector3::new(7.5, 8.5, 9.5),
        direction: Vector3::new(0.25, -0.5, 0.75),
        bearing_std_rad: 0.125,
        timestamp_s: 33.75,
        confidence: 0.625,
    };
    let restored =
        argusnet_proto::observation_from_pb(argusnet_proto::observation_to_pb(original.clone()))
            .expect("observation round trip");
    assert_eq!(restored, original);
}

#[test]
fn truth_round_trips_every_field() {
    let original = TruthState {
        target_id: "target-9".to_string(),
        position: Vector3::new(-11.5, 12.5, -13.5),
        velocity: Vector3::new(14.5, -15.5, 16.5),
        timestamp_s: 44.5,
    };
    let restored = argusnet_proto::truth_from_pb(argusnet_proto::truth_to_pb(original.clone()))
        .expect("truth round trip");
    assert_eq!(restored, original);
}

#[test]
fn track_round_trips_every_field() {
    let original = TrackState {
        track_id: "track-3".to_string(),
        timestamp_s: 55.25,
        position: Vector3::new(21.5, -22.5, 23.5),
        velocity: Vector3::new(-24.5, 25.5, -26.5),
        covariance: distinctive_covariance(),
        measurement_std_m: 1.75,
        update_count: 17,
        stale_steps: 4,
        lifecycle_state: Some("confirmed".to_string()),
        quality_score: Some(0.875),
        mode_probability_cv: Some(0.375),
        contributing_node_ids: Arc::new(vec!["node-alpha".to_string(), "node-bravo".to_string()]),
    };
    let restored = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original.clone()))
        .expect("track round trip");
    assert_eq!(restored, original);
}

/// `mode_probability_cv` is `Option<f64>` in the runtime type but a bare `double`
/// on the wire, so "unset" needs an in-band representation. Zero is it.
///
/// Both decoders already read `0.0` as absent -- `track_from_pb` here, and the
/// viewer's replay reader in `replay.rs` -- and `NodeState.max_range_m` uses the
/// same convention in the same proto. Only the encoder disagreed: it sent `1.0`
/// for `None`, so `None` came back as `Some(1.0)`, which reports total
/// confidence in the constant-velocity mode where the truth is "no IMM estimate
/// at all". That was pinned as a known defect before P4.0; this test now pins
/// the fix.
///
/// The residual cost is that a genuine `Some(0.0)` cannot be told from absent.
/// `argusnet-core`'s IMM filter cannot emit it -- the mixed mode probability is
/// a normalized posterior with its numerator floored at `1.0e-30`, so it is
/// strictly positive -- and that is asserted below rather than asserted here.
#[test]
fn mode_probability_cv_round_trips_through_the_unset_sentinel() {
    let mut original = TrackState {
        track_id: "track-4".to_string(),
        timestamp_s: 1.0,
        position: Vector3::zeros(),
        velocity: Vector3::zeros(),
        covariance: distinctive_covariance(),
        measurement_std_m: 1.0,
        update_count: 1,
        stale_steps: 0,
        lifecycle_state: None,
        quality_score: None,
        mode_probability_cv: None,
        contributing_node_ids: Arc::new(Vec::new()),
    };

    let encoded = argusnet_proto::track_to_pb(original.clone());
    assert_eq!(
        encoded.mode_probability_cv, 0.0,
        "None must encode as the 0.0 sentinel, not as 1.0"
    );
    let restored = argusnet_proto::track_from_pb(encoded).expect("track round trip");
    assert_eq!(restored.mode_probability_cv, None);

    // Some(1.0) is now a real value, distinct from unset. Before the fix this
    // was the value None masqueraded as.
    original.mode_probability_cv = Some(1.0);
    let restored_one = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original.clone()))
        .expect("track round trip");
    assert_eq!(restored_one.mode_probability_cv, Some(1.0));

    // The documented residual: 0.0 is the sentinel, so it cannot also be a value.
    original.mode_probability_cv = Some(0.0);
    let restored_zero = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original))
        .expect("track round trip");
    assert_eq!(
        restored_zero.mode_probability_cv, None,
        "0.0 means unset; no reader can recover a genuine Some(0.0)"
    );
}

/// The claim the sentinel rests on: the only producer of `mode_probability_cv`
/// never emits exactly zero, so nothing real is lost by spending `0.0` on
/// "unset". Asserted by running the filter rather than by reading it -- the
/// numerator floor is `1.0e-30`, which is small enough that "obviously
/// positive" deserves a check.
#[test]
fn the_imm_filter_never_emits_a_zero_mode_probability() {
    let mut engine = argusnet_core::TrackingEngine::new(argusnet_core::TrackerConfig::default())
        .expect("default config is valid");
    let mut saw_a_track = false;

    for step in 0..40 {
        let timestamp_s = step as f64 * 0.1;
        // Two widely separated nodes bearing on one crossing target, so the
        // intersection is well conditioned and tracks actually form.
        let target = Vector3::new(150.0 + 8.0 * timestamp_s, 90.0, 20.0);
        let observations = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 400.0, 0.0),
            Vector3::new(400.0, 0.0, 0.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, origin)| argusnet_types::BearingObservation {
            node_id: format!("node-{index}"),
            target_id: "asset-a".to_string(),
            origin,
            direction: (target - origin).normalize(),
            bearing_std_rad: 0.01,
            timestamp_s,
            confidence: 1.0,
        })
        .collect::<Vec<_>>();

        let nodes = observations
            .iter()
            .map(|observation| argusnet_types::NodeState {
                node_id: observation.node_id.clone(),
                position: observation.origin,
                velocity: Vector3::zeros(),
                is_mobile: false,
                timestamp_s,
                health: 1.0,
            })
            .collect::<Vec<_>>();

        let frame = engine.ingest_frame(
            timestamp_s,
            nodes,
            observations,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        for track in &frame.tracks {
            saw_a_track = true;
            let probability = track
                .mode_probability_cv
                .expect("the core filter always reports a mode probability");
            assert!(
                probability > 0.0,
                "mode_probability_cv={probability} collides with the unset sentinel"
            );
        }
    }

    assert!(
        saw_a_track,
        "no track formed, so the assertion above never ran"
    );
}

/// `acceleration_mps2`, `confidence` and `last_seen_s` exist on the wire but not
/// on the runtime `TrackState`. The first is a hardcoded zero vector; the other
/// two are derived from fields that are carried. None of them survives a round
/// trip because none is ever read back -- which is why the matrix records them
/// as `inert` and `derived` rather than `carried`.
#[test]
fn track_wire_only_fields_are_constant_or_derived() {
    let track = TrackState {
        track_id: "track-5".to_string(),
        timestamp_s: 99.5,
        position: Vector3::zeros(),
        velocity: Vector3::zeros(),
        covariance: distinctive_covariance(),
        measurement_std_m: 1.0,
        update_count: 1,
        stale_steps: 0,
        lifecycle_state: None,
        quality_score: Some(0.25),
        mode_probability_cv: Some(0.5),
        contributing_node_ids: Arc::new(Vec::new()),
    };
    let encoded = argusnet_proto::track_to_pb(track.clone());

    assert_eq!(
        encoded.acceleration_mps2,
        vec![0.0, 0.0, 0.0],
        "acceleration is a constant, not a measurement"
    );
    assert_eq!(
        encoded.confidence, 0.25,
        "confidence is an alias of quality_score"
    );
    assert_eq!(
        encoded.last_seen_s, 99.5,
        "last_seen_s is an alias of timestamp_s"
    );
}

#[test]
fn metrics_round_trip_every_field() {
    let original = PlatformMetrics {
        mean_error_m: Some(1.25),
        max_error_m: Some(9.75),
        active_track_count: 5,
        observation_count: 40,
        accepted_observation_count: 31,
        rejected_observation_count: 9,
        mean_measurement_std_m: Some(2.5),
        track_errors_m: BTreeMap::from([("track-1".to_string(), 0.5)]),
        rejection_counts: BTreeMap::from([("occlusion".to_string(), 3u32)]),
        accepted_observations_by_target: BTreeMap::from([("target-1".to_string(), 12u32)]),
        rejected_observations_by_target: BTreeMap::from([("target-2".to_string(), 4u32)]),
    };
    let restored = argusnet_proto::metrics_from_pb(argusnet_proto::metrics_to_pb(original.clone()));
    assert_eq!(restored, original);
}

/// The frame-level annotations P4.0 gave `PlatformFrame` real fields for.
/// Before that they had no runtime home, so `frame_to_pb` emitted empty vectors
/// and only the server's ingest path carried them, by grafting the request's
/// copies onto the response.
#[test]
fn frame_annotations_round_trip_every_field() {
    let original = PlatformFrame {
        timestamp_s: 7.5,
        nodes: Vec::new(),
        observations: Vec::new(),
        rejected_observations: Vec::new(),
        tracks: Vec::new(),
        truths: Vec::new(),
        metrics: PlatformMetrics {
            mean_error_m: None,
            max_error_m: None,
            active_track_count: 0,
            observation_count: 0,
            accepted_observation_count: 0,
            rejected_observation_count: 0,
            mean_measurement_std_m: None,
            track_errors_m: BTreeMap::new(),
            rejection_counts: BTreeMap::new(),
            accepted_observations_by_target: BTreeMap::new(),
            rejected_observations_by_target: BTreeMap::new(),
        },
        generation_rejections: Vec::new(),
        target_metadata: vec![TargetMetadata {
            target_id: "tgt-1".to_string(),
            operator_label: "Alpha".to_string(),
            target_type: "rotary".to_string(),
            priority: 3,
            behavior: "loiter".to_string(),
            status: "tracked".to_string(),
            first_seen_s: 1.25,
            last_seen_s: 2.75,
            confidence: 0.875,
            notes: "operator annotation".to_string(),
        }],
        safety_events: vec![SafetyEvent {
            event_id: "evt-1".to_string(),
            subject_id: "tgt-1".to_string(),
            timestamp_s: 1.5,
            state: "warn".to_string(),
            violations: vec!["floor".to_string(), "geofence".to_string()],
            command_clamped: true,
            blocked: false,
            reason: "below floor".to_string(),
        }],
    };

    let restored = argusnet_proto::frame_from_pb(argusnet_proto::frame_to_pb(original.clone()))
        .expect("frame round trip");
    assert_eq!(restored, original);

    // frame_to_pb_ref must agree with the owning variant, not drift from it.
    let by_ref = argusnet_proto::frame_to_pb_ref(&original);
    assert_eq!(by_ref, argusnet_proto::frame_to_pb(original));
}

// ---------------------------------------------------------------------------
// The cases P4.0 names explicitly: every optional payload present *and* absent,
// and invalid covariance lengths.
// ---------------------------------------------------------------------------

/// A 6x6 covariance is 36 values. Anything else is a malformed message, and the
/// conversion has to say so rather than reshape whatever it was given -- Python's
/// adapter hardcodes `reshape(6, 6)` and raises on a mismatch, so a Rust side
/// that quietly accepted 35 would produce frames Python cannot read.
#[test]
fn covariance_lengths_other_than_36_are_rejected() {
    let template = argusnet_proto::track_to_pb(TrackState {
        track_id: "track-len".to_string(),
        timestamp_s: 1.0,
        position: Vector3::zeros(),
        velocity: Vector3::zeros(),
        covariance: distinctive_covariance(),
        measurement_std_m: 1.0,
        update_count: 1,
        stale_steps: 0,
        lifecycle_state: None,
        quality_score: None,
        mode_probability_cv: Some(0.5),
        contributing_node_ids: Arc::new(Vec::new()),
    });
    assert_eq!(template.covariance_row_major.len(), 36);

    for length in [0usize, 1, 9, 35, 37, 72] {
        let mut malformed = template.clone();
        malformed.covariance_row_major = vec![1.0; length];
        let error = argusnet_proto::track_from_pb(malformed)
            .expect_err("a covariance that is not 36 values must be rejected");
        assert!(
            error.contains("36") && error.contains(&length.to_string()),
            "error {error:?} should name both the expected and the actual length"
        );
    }

    // The boundary itself still passes, so the check is a length test rather
    // than a blanket rejection.
    argusnet_proto::track_from_pb(template).expect("36 values is valid");
}

/// Every optional payload absent at once. The all-present case is covered by the
/// per-type round trips above; this is the other half, and it is the half that
/// catches a conversion which substitutes a default for `None` -- a substitution
/// the present-case tests cannot see.
#[test]
fn every_optional_payload_survives_being_absent() {
    let track = TrackState {
        track_id: "track-empty".to_string(),
        timestamp_s: 3.5,
        position: Vector3::new(1.0, 2.0, 3.0),
        velocity: Vector3::zeros(),
        covariance: distinctive_covariance(),
        measurement_std_m: 0.5,
        update_count: 0,
        stale_steps: 0,
        lifecycle_state: None,
        quality_score: None,
        mode_probability_cv: None,
        contributing_node_ids: Arc::new(Vec::new()),
    };
    let rejection = argusnet_types::ObservationRejection {
        node_id: "node-empty".to_string(),
        target_id: "target-empty".to_string(),
        timestamp_s: 4.5,
        reason: "out_of_range".to_string(),
        detail: String::new(),
        origin: None,
        attempted_point: None,
        closest_point: None,
        blocker_type: String::new(),
        first_hit_range_m: None,
    };
    let original = PlatformFrame {
        timestamp_s: 5.5,
        nodes: Vec::new(),
        observations: Vec::new(),
        rejected_observations: vec![rejection.clone()],
        tracks: vec![track],
        truths: Vec::new(),
        metrics: PlatformMetrics {
            mean_error_m: None,
            max_error_m: None,
            active_track_count: 0,
            observation_count: 0,
            accepted_observation_count: 0,
            rejected_observation_count: 0,
            mean_measurement_std_m: None,
            track_errors_m: BTreeMap::new(),
            rejection_counts: BTreeMap::new(),
            accepted_observations_by_target: BTreeMap::new(),
            rejected_observations_by_target: BTreeMap::new(),
        },
        generation_rejections: vec![rejection],
        target_metadata: Vec::new(),
        safety_events: Vec::new(),
    };

    let restored = argusnet_proto::frame_from_pb(argusnet_proto::frame_to_pb(original.clone()))
        .expect("frame round trip");
    assert_eq!(restored, original);
}

/// `metrics` is the one submessage `frame_from_pb` tolerates missing entirely,
/// because a producer that reports no metrics is not malformed. It must default
/// to the empty metrics rather than fail or fabricate counts.
#[test]
fn a_frame_without_metrics_decodes_to_empty_metrics() {
    let mut encoded = argusnet_proto::frame_to_pb(PlatformFrame {
        timestamp_s: 6.5,
        nodes: Vec::new(),
        observations: Vec::new(),
        rejected_observations: Vec::new(),
        tracks: Vec::new(),
        truths: Vec::new(),
        metrics: PlatformMetrics {
            mean_error_m: Some(9.0),
            max_error_m: Some(11.0),
            active_track_count: 4,
            observation_count: 5,
            accepted_observation_count: 3,
            rejected_observation_count: 2,
            mean_measurement_std_m: Some(1.5),
            track_errors_m: BTreeMap::new(),
            rejection_counts: BTreeMap::new(),
            accepted_observations_by_target: BTreeMap::new(),
            rejected_observations_by_target: BTreeMap::new(),
        },
        generation_rejections: Vec::new(),
        target_metadata: Vec::new(),
        safety_events: Vec::new(),
    });
    encoded.metrics = None;

    let restored = argusnet_proto::frame_from_pb(encoded).expect("frame round trip");
    assert_eq!(restored.metrics.mean_error_m, None);
    assert_eq!(restored.metrics.active_track_count, 0);
}

/// A required `Vector3` that is absent is a malformed message, not a zero
/// vector. Proto3 cannot distinguish "absent" from "all zeros" for a scalar, but
/// it can for a submessage, and these conversions use that: a decoder that
/// defaulted to the origin would place a node at (0, 0, 0) with no signal.
#[test]
fn a_missing_required_vector_is_rejected_rather_than_defaulted() {
    let node = NodeState {
        node_id: "node-missing".to_string(),
        position: Vector3::new(1.0, 2.0, 3.0),
        velocity: Vector3::new(4.0, 5.0, 6.0),
        is_mobile: true,
        timestamp_s: 1.0,
        health: 1.0,
    };

    let mut without_position = argusnet_proto::node_state_to_pb(node.clone());
    without_position.position = None;
    let error = argusnet_proto::node_state_from_pb(without_position)
        .expect_err("a node with no position must be rejected");
    assert!(
        error.contains("node.position"),
        "error {error:?} should name the missing field"
    );

    let mut without_velocity = argusnet_proto::node_state_to_pb(node);
    without_velocity.velocity = None;
    argusnet_proto::node_state_from_pb(without_velocity)
        .expect_err("a node with no velocity must be rejected");
}

/// The owned and borrowed encoders share one body per type. This asserts the two
/// public entry points stay wired to it: before P4.0 they were two hand-written
/// bodies, and they drifted -- `target_metadata` reached one and not the other,
/// and the covariance transpose had to be fixed twice.
#[test]
fn owned_and_borrowed_encoders_agree_for_every_type() {
    let node = NodeState {
        node_id: "node-pair".to_string(),
        position: Vector3::new(1.5, 2.5, 3.5),
        velocity: Vector3::new(4.5, 5.5, 6.5),
        is_mobile: true,
        timestamp_s: 7.5,
        health: 0.5,
    };
    assert_eq!(
        argusnet_proto::node_state_to_pb_ref(&node),
        argusnet_proto::node_state_to_pb(node)
    );

    let observation = BearingObservation {
        node_id: "node-pair".to_string(),
        target_id: "target-pair".to_string(),
        origin: Vector3::new(1.0, 2.0, 3.0),
        direction: Vector3::new(0.0, 0.0, 1.0),
        bearing_std_rad: 0.02,
        timestamp_s: 8.5,
        confidence: 0.9,
    };
    assert_eq!(
        argusnet_proto::observation_to_pb_ref(&observation),
        argusnet_proto::observation_to_pb(observation)
    );

    let rejection = argusnet_types::ObservationRejection {
        node_id: "node-pair".to_string(),
        target_id: "target-pair".to_string(),
        timestamp_s: 9.5,
        reason: "terrain_occlusion".to_string(),
        detail: "ridge".to_string(),
        origin: Some(Vector3::new(1.0, 1.0, 1.0)),
        attempted_point: Some(Vector3::new(2.0, 2.0, 2.0)),
        closest_point: None,
        blocker_type: "terrain".to_string(),
        first_hit_range_m: Some(12.5),
    };
    assert_eq!(
        argusnet_proto::rejection_to_pb_ref(&rejection),
        argusnet_proto::rejection_to_pb(rejection)
    );

    let truth = TruthState {
        target_id: "target-pair".to_string(),
        position: Vector3::new(3.0, 4.0, 5.0),
        velocity: Vector3::new(6.0, 7.0, 8.0),
        timestamp_s: 10.5,
    };
    assert_eq!(
        argusnet_proto::truth_to_pb_ref(&truth),
        argusnet_proto::truth_to_pb(truth)
    );

    let track = TrackState {
        track_id: "track-pair".to_string(),
        timestamp_s: 11.5,
        position: Vector3::new(9.0, 10.0, 11.0),
        velocity: Vector3::new(12.0, 13.0, 14.0),
        covariance: distinctive_covariance(),
        measurement_std_m: 2.5,
        update_count: 6,
        stale_steps: 1,
        lifecycle_state: Some("tentative".to_string()),
        quality_score: Some(0.25),
        mode_probability_cv: Some(0.75),
        contributing_node_ids: Arc::new(vec!["node-pair".to_string()]),
    };
    assert_eq!(
        argusnet_proto::track_to_pb_ref(&track),
        argusnet_proto::track_to_pb(track)
    );

    let metrics = PlatformMetrics {
        mean_error_m: Some(1.25),
        max_error_m: Some(3.75),
        active_track_count: 2,
        observation_count: 8,
        accepted_observation_count: 6,
        rejected_observation_count: 2,
        mean_measurement_std_m: Some(0.5),
        track_errors_m: BTreeMap::from([("track-pair".to_string(), 1.5)]),
        rejection_counts: BTreeMap::from([("out_of_range".to_string(), 2u32)]),
        accepted_observations_by_target: BTreeMap::from([("target-pair".to_string(), 6u32)]),
        rejected_observations_by_target: BTreeMap::from([("target-pair".to_string(), 2u32)]),
    };
    assert_eq!(
        argusnet_proto::metrics_to_pb_ref(&metrics),
        argusnet_proto::metrics_to_pb(metrics)
    );
}

/// `TrackerConfig` is the one conversion whose decoder substitutes defaults on
/// purpose: a zero on the wire means "unspecified", so an older client that
/// leaves a tuning field unset gets the current default instead of a zero that
/// would fail validation. Round-tripping a fully non-default config proves the
/// substitution only fires where it is meant to.
#[test]
fn tracker_config_round_trips_every_field() {
    let original = argusnet_core::TrackerConfig {
        min_observations: 3,
        max_stale_steps: 9,
        retain_history: true,
        min_confidence: 0.35,
        max_bearing_std_rad: 0.09,
        max_timestamp_skew_s: 0.45,
        min_intersection_angle_deg: 7.5,
        data_association_mode: argusnet_core::AssociationMode::JPDA,
        cv_process_accel_std: 1.25,
        ct_process_accel_std: 2.25,
        ct_turn_rate_std: 0.15,
        innovation_window: 11,
        innovation_scale_factor: 1.75,
        innovation_max_scale: 4.5,
        adaptive_measurement_noise: true,
        chi_squared_gate_threshold: 12.5,
        cluster_distance_threshold_m: 22.5,
        near_parallel_rejection_angle_deg: 3.5,
        confirmation_m: 2,
        confirmation_n: 4,
        max_coast_frames: 6,
        max_coast_seconds: 2.5,
        min_quality_score: 0.15,
    };
    let restored =
        argusnet_proto::tracker_config_from_pb(argusnet_proto::tracker_config_to_pb(&original))
            .expect("config round trip");
    assert_eq!(restored, original);
}
