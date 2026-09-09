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
    BearingObservation, Matrix6, NodeState, PlatformMetrics, TrackState, TruthState,
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
/// on the wire, and the two directions disagree about which value means "unset":
/// `track_to_pb` sends `unwrap_or(1.0)`, while `track_from_pb` treats only `0.0`
/// as `None`. So `None` comes back as `Some(1.0)`.
///
/// This is the one conversion in the file that does not round-trip. It is pinned
/// rather than fixed because changing either direction changes what an existing
/// peer sees on the wire, which belongs with P4.0's ownership of the DTO
/// contract -- and because `Some(1.0)` and `None` are genuinely
/// indistinguishable once encoded, so no reader can recover the difference.
#[test]
fn mode_probability_cv_none_does_not_survive_the_wire() {
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

    let restored = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original.clone()))
        .expect("track round trip");
    assert_eq!(
        restored.mode_probability_cv,
        Some(1.0),
        "None is encoded as 1.0 and read back as Some(1.0)"
    );

    // Some(1.0) is indistinguishable from None on the wire, by construction.
    original.mode_probability_cv = Some(1.0);
    let restored_one = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original.clone()))
        .expect("track round trip");
    assert_eq!(restored_one.mode_probability_cv, Some(1.0));

    // 0.0 is the value that does map back to None.
    original.mode_probability_cv = Some(0.0);
    let restored_zero = argusnet_proto::track_from_pb(argusnet_proto::track_to_pb(original))
        .expect("track round trip");
    assert_eq!(restored_zero.mode_probability_cv, None);
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
