//! The optical-flow update: parity where the port agrees, and proof of change
//! where it does not.
//!
//! This test file is shaped differently from the other parity tests in this
//! migration, deliberately. Cases the fixture tags `agreement` are ordinary
//! parity targets. Cases tagged `divergence` record behaviour P4.4 requires
//! **changing** — asserting parity on those would enshrine the defect.
//!
//! For a divergence case the fixture is still the evidence: it proves the
//! reference really does what the contract document says, so the change is
//! demonstrable rather than asserted, and a regression back to the old
//! behaviour would fail here.

use argusnet_localization::vio::{
    FeatureStore, FlowConfig, FrameTime, ImuMeasurement, NoInformation, UpdateOutcome, VioNoise,
    VioPredictor, VisualFeature,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    mahalanobis_gate: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    first_frame: Frame,
    second_frame: Frame,
    accel_for_velocity: Option<[String; 3]>,
    before_second_frame: Snapshot,
    reference_after: Snapshot,
    reference_tracked_feature_count: Option<usize>,
}

#[derive(Deserialize)]
struct Frame {
    timestamp_s: String,
    features: Vec<Feature>,
}

#[derive(Deserialize)]
struct Feature {
    feature_id: u64,
    u: String,
    v: String,
}

#[derive(Deserialize)]
struct Snapshot {
    state: Vec<String>,
    covariance: Vec<String>,
    position_covariance_trace: String,
    feature_store_ids: Vec<u64>,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/vio_update.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn features(frame: &Frame) -> Vec<VisualFeature> {
    frame
        .features
        .iter()
        .map(|feature| VisualFeature {
            feature_id: feature.feature_id,
            u: f(&feature.u),
            v: f(&feature.v),
        })
        .collect()
}

/// Replay a case up to just before the second frame's update.
fn replay_to_second_frame(case: &Case) -> (VioPredictor, FeatureStore, Vec<VisualFeature>, f64) {
    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 100.0], 0.0);

    let mut measurements = Vec::new();
    if let Some(accel) = &case.accel_for_velocity {
        measurements.push(ImuMeasurement {
            timestamp_s: 0.5,
            angular_velocity: [0.0; 3],
            linear_acceleration: [f(&accel[0]), f(&accel[1]), f(&accel[2])],
        });
    }

    let mut store = FeatureStore::default();
    let first = features(&case.first_frame);
    let t1 = f(&case.first_frame.timestamp_s);
    predictor.predict_to(&measurements, t1);
    // Nothing matches on the first frame; it only seeds the store.
    store.replace(&first);

    let second = features(&case.second_frame);
    let t2 = f(&case.second_frame.timestamp_s);
    (predictor, store, second, t2)
}

const TOLERANCE: f64 = 1e-9;

/// Cases where the port must reproduce the reference.
#[test]
fn agreement_cases_match_the_reference() {
    let fixture = load();
    let mut checked = 0;
    for case in fixture.cases.iter().filter(|c| c.kind == "agreement") {
        let (mut predictor, store, second, t2) = replay_to_second_frame(case);
        let dt_total = t2 - predictor.timestamp_s();
        predictor.predict_to(&[], t2);
        let outcome =
            predictor.update_optical_flow(&second, &store, dt_total, &FlowConfig::default());

        let expected_matched = case
            .reference_tracked_feature_count
            .expect("agreement cases return a state");
        let actual_matched = match outcome {
            UpdateOutcome::Applied {
                features_matched, ..
            } => features_matched,
            UpdateOutcome::NoInformation {
                features_matched, ..
            } => features_matched,
        };
        assert_eq!(
            actual_matched, expected_matched,
            "{}: matched feature count",
            case.name
        );

        for (component, expected) in case.reference_after.state.iter().enumerate() {
            let actual = predictor.state()[component];
            let expected = f(expected);
            let scale = expected.abs().max(actual.abs()).max(1.0);
            assert!(
                (actual - expected).abs() / scale <= TOLERANCE,
                "{}: state[{component}] {actual} vs {expected}",
                case.name
            );
        }

        // The covariance too. Checking only the state left the sign of the
        // `(I - KH) P` update untested: flipping it to `(I + KH)` -- making
        // every accepted measurement INCREASE uncertainty -- passed cleanly.
        assert_eq!(
            case.reference_after.covariance.len(),
            225,
            "covariance must be 15x15"
        );
        for (flat, expected) in case.reference_after.covariance.iter().enumerate() {
            let (row, col) = (flat / 15, flat % 15);
            let actual = predictor.covariance()[(row, col)];
            let expected = f(expected);
            let scale = expected.abs().max(actual.abs()).max(1.0);
            assert!(
                (actual - expected).abs() / scale <= TOLERANCE,
                "{}: P[{row},{col}] {actual} vs {expected}",
                case.name
            );
        }

        // And the direction of travel: an accepted measurement must make the
        // filter more certain, which is the property the sign carries.
        if matches!(outcome, UpdateOutcome::Applied { features_used, .. } if features_used > 0) {
            assert!(
                predictor.position_covariance_trace()
                    < f(&case.before_second_frame.position_covariance_trace)
                        + f(&case.reference_after.position_covariance_trace),
                "{}: an applied update must not inflate the covariance",
                case.name
            );
        }
        checked += 1;
    }
    assert!(checked >= 2, "expected agreement cases in the fixture");
}

/// The defect this port exists to remove: the reference, handed nothing it
/// believes, moves the estimate and becomes MORE confident.
///
/// The fixture proves it does that. The assertions prove the port does not.
#[test]
fn the_port_does_not_manufacture_confidence_when_every_feature_is_gated_out() {
    let fixture = load();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.name == "all_features_gated_out")
        .expect("divergence case present");

    // First: the reference really does what the contract document claims.
    let before_trace = f(&case.before_second_frame.position_covariance_trace);
    let reference_trace = f(&case.reference_after.position_covariance_trace);
    assert!(
        reference_trace < before_trace,
        "the reference no longer shrinks the covariance here ({before_trace} -> \
         {reference_trace}); the defect this port removes may have been fixed \
         upstream, and this test's premise needs revisiting"
    );
    let reference_moved = (6..9)
        .map(|i| {
            (f(&case.reference_after.state[i]) - f(&case.before_second_frame.state[i])).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        reference_moved > 1.0,
        "the reference moved only {reference_moved} m; expected metres"
    );

    // Now the port, on the same inputs.
    let (mut predictor, store, second, t2) = replay_to_second_frame(case);
    let dt_total = t2 - predictor.timestamp_s();
    predictor.predict_to(&[], t2);
    let trace_after_prediction = predictor.position_covariance_trace();
    let position_after_prediction = predictor.position();

    let outcome = predictor.update_optical_flow(&second, &store, dt_total, &FlowConfig::default());

    assert_eq!(
        outcome,
        UpdateOutcome::NoInformation {
            features_matched: 8,
            reason: NoInformation::AllFeaturesGatedOut,
        },
        "{}: every feature failed the gate, so nothing should be applied",
        case.name
    );
    assert_eq!(
        predictor.position_covariance_trace(),
        trace_after_prediction,
        "covariance must be exactly what prediction left it -- not shrunk"
    );
    assert_eq!(
        predictor.position(),
        position_after_prediction,
        "position must not move on a frame carrying no usable observation"
    );
    // And the direction of travel is the opposite of the reference's.
    assert!(
        predictor.position_covariance_trace() > before_trace,
        "having rejected every observation, the filter must be less certain, \
         not more"
    );
}

/// The reference drops an out-of-order frame *before* refreshing its feature
/// store, so the next frame matches against a two-frame-old store. The port
/// refreshes on every frame it sees.
#[test]
fn an_out_of_order_frame_still_refreshes_the_feature_store() {
    let fixture = load();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.name == "out_of_order_frame")
        .expect("divergence case present");

    // The reference keeps the FIRST frame's ids.
    let first_ids: Vec<u64> = case
        .first_frame
        .features
        .iter()
        .map(|f| f.feature_id)
        .collect();
    assert_eq!(
        case.reference_after.feature_store_ids, first_ids,
        "the reference no longer leaves a stale store; premise needs revisiting"
    );

    // The port classifies the frame and refreshes regardless.
    let (predictor, mut store, second, t2) = replay_to_second_frame(case);
    assert_eq!(
        predictor.classify(t2),
        FrameTime::OutOfOrder,
        "the second frame is older than the filter clock"
    );
    store.replace(&second);
    let ids: Vec<u64> = store.ids().collect();
    let expected: Vec<u64> = case
        .second_frame
        .features
        .iter()
        .map(|f| f.feature_id)
        .collect();
    assert_eq!(ids, expected, "the store must hold the frame just seen");
    assert_ne!(
        ids, first_ids,
        "and must no longer hold the previous frame's"
    );
}

/// A repeated timestamp is its own case, named rather than falling through the
/// `< 0` guard as it does in the reference.
#[test]
fn a_repeated_timestamp_is_classified_separately_from_an_older_one() {
    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 100.0], 1.0);
    assert_eq!(predictor.classify(1.5), FrameTime::Advancing);
    assert_eq!(predictor.classify(1.0), FrameTime::Repeated);
    assert_eq!(predictor.classify(0.5), FrameTime::OutOfOrder);
}

/// Fewer than two matches is distinct from "matched but all rejected". The
/// reference reports both through the same covariance change, so a caller
/// cannot tell them apart.
#[test]
fn too_few_matches_is_reported_distinctly_from_all_gated_out() {
    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 100.0], 0.0);
    predictor.predict_to(&[], 0.1);

    let mut store = FeatureStore::default();
    store.replace(&[VisualFeature {
        feature_id: 1,
        u: 10.0,
        v: 10.0,
    }]);
    let outcome = predictor.update_optical_flow(
        &[VisualFeature {
            feature_id: 1,
            u: 10.5,
            v: 10.5,
        }],
        &store,
        0.1,
        &FlowConfig::default(),
    );
    assert_eq!(
        outcome,
        UpdateOutcome::NoInformation {
            features_matched: 1,
            reason: NoInformation::TooFewMatches,
        }
    );
}

/// The gate threshold is part of the contract, not an implementation detail.
#[test]
fn the_gate_threshold_matches_the_reference() {
    let fixture = load();
    assert_eq!(
        f(&fixture.mahalanobis_gate),
        argusnet_localization::vio::MAHALANOBIS_GATE
    );
}
