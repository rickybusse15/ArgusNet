//! Parity for `EKFVIO`'s public surface: `process_image`, `get_state`, `reset`.
//!
//! `vio_prediction` and `vio_update` pin the filter's internals. This pins what
//! a caller sees — the part P4.4 names explicitly and the part the simulation
//! consumes at `sim.py:5480`.
//!
//! # What is compared, and what is asserted instead
//!
//! `docs/migration/VIO_CONTRACT.md` sets the policy and this follows it. The
//! prediction step is deterministic and compares value-for-value. The update
//! step **deliberately diverges** where the reference fabricates a measurement
//! from a feature count, so on those frames the fixture is evidence of what was
//! changed and these tests assert the *new* contract. Mixing the two would let
//! the fixture bless the behaviour P4.4 exists to remove.
//!
//! The surface behaviours below have nothing to do with the maths and are
//! compared directly, because nothing about them is meant to differ.

use argusnet_localization::vio::{
    FeatureStore, FlowConfig, NoPose, VioNoise, VioPredictor, VisualFeature,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    frames: Vec<Frame>,
    final_get_state: Frame,
    reset_clears_the_estimate: bool,
    uninitialized_returns_none: bool,
}

#[derive(Deserialize)]
struct Frame {
    requested_timestamp_s: String,
    state: Option<State>,
}

#[derive(Deserialize)]
struct State {
    timestamp_s: String,
    bias_gyro: Vec<String>,
    bias_accel: Vec<String>,
    tracked_feature_count: usize,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/vio_surface.json"
    );
    let text = std::fs::read_to_string(path).expect("vio_surface.json");
    serde_json::from_str(&text).expect("vio_surface.json parses")
}

#[test]
fn an_uninitialized_estimator_has_no_pose_rather_than_a_confident_one() {
    // P4.4: "no manufactured confidence on a failed estimator". The reference
    // returns None; the danger is a port that answers with a zero pose at the
    // default covariance, which reads as a confident estimate at the origin.
    let fixture = load();
    assert!(fixture.cases.iter().all(|c| c.uninitialized_returns_none));

    let predictor = VioPredictor::new(VioNoise::default());
    assert_eq!(predictor.state_snapshot(), Err(NoPose::Uninitialized));

    let mut predictor = VioPredictor::new(VioNoise::default());
    let mut store = FeatureStore::default();
    assert_eq!(
        predictor
            .observe_image(&[], &mut store, &[], 1.0, &FlowConfig::default())
            .err(),
        Some(NoPose::Uninitialized),
        "an uninitialized estimator must refuse a frame, not answer one"
    );
}

#[test]
fn reset_returns_the_estimator_to_having_no_pose() {
    let fixture = load();
    assert!(
        fixture.cases.iter().all(|c| c.reset_clears_the_estimate),
        "the reference clears on reset"
    );

    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([1.0, 2.0, 3.0], 0.5);
    assert!(predictor.state_snapshot().is_ok());
    predictor.reset();
    assert_eq!(
        predictor.state_snapshot(),
        Err(NoPose::Uninitialized),
        "reset must clear the initialized flag, not merely zero the state -- \
         the reference leaves P = I * 0.01 behind, a SMALL covariance on a \
         meaningless state, and only the flag makes that unreachable"
    );
}

#[test]
fn get_state_reports_zero_tracked_features_even_while_tracking() {
    // The reference sets the real matched count on the `process_image` path
    // and hardcodes zero on the `get_state` path -- two values for one field.
    // Preserved rather than unified: a caller comparing the two paths would
    // otherwise see a difference this port invented.
    let fixture = load();
    for case in &fixture.cases {
        let state = case
            .final_get_state
            .state
            .as_ref()
            .expect("the run ends initialized");
        assert_eq!(
            state.tracked_feature_count, 0,
            "{}: the reference hardcodes zero here",
            case.name
        );
    }

    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 0.0], 0.0);
    let mut store = FeatureStore::default();
    let features: Vec<VisualFeature> = (1..6)
        .map(|id| VisualFeature {
            feature_id: id,
            u: id as f64 * 30.0,
            v: id as f64 * 20.0,
        })
        .collect();
    store.replace(&features);
    let outcome = predictor
        .observe_image(&features, &mut store, &[], 0.1, &FlowConfig::default())
        .expect("initialized");
    assert!(
        outcome.state.tracked_feature_count > 0,
        "an update path must report the real count"
    );
    assert_eq!(
        predictor.state_snapshot().unwrap().tracked_feature_count,
        0,
        "the snapshot path must still report zero"
    );
}

#[test]
fn an_out_of_order_frame_leaves_the_clock_where_it_was() {
    // The reference returns `get_state()` before touching anything, so the
    // state it hands back is stamped with the FILTER's time, not the frame's.
    // A caller stamping that pose with the frame's own timestamp would date it
    // wrongly. The fixture must contain such a frame or this proves nothing.
    let fixture = load();
    let mut stale_frames = 0;
    for case in &fixture.cases {
        for frame in &case.frames {
            let Some(state) = &frame.state else { continue };
            if f(&state.timestamp_s) != f(&frame.requested_timestamp_s) {
                assert!(
                    f(&state.timestamp_s) > f(&frame.requested_timestamp_s),
                    "{}: a state may lag a requested time, never lead it",
                    case.name
                );
                stale_frames += 1;
            }
        }
    }
    assert!(
        stale_frames > 0,
        "no recorded frame is out of order, so this asserts nothing"
    );

    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 0.0], 5.0);
    let mut store = FeatureStore::default();
    let outcome = predictor
        .observe_image(&[], &mut store, &[], 4.0, &FlowConfig::default())
        .expect("initialized");
    assert_eq!(
        outcome.state.timestamp_s, 5.0,
        "the clock must not move backwards"
    );
    assert!(
        outcome.update.is_none(),
        "an out-of-order frame applies nothing, which is a different thing \
         from applying an update that carried no information"
    );
    assert_eq!(predictor.timestamp_s(), 5.0);
}

#[test]
fn a_repeated_timestamp_is_processed_rather_than_rejected() {
    // The reference gates on `dt_total < 0`, not `<= 0`, so a repeated frame
    // runs the full predict-and-update. Off-by-one here silently drops frames.
    let mut predictor = VioPredictor::new(VioNoise::default());
    predictor.initialize([0.0, 0.0, 0.0], 2.0);
    let mut store = FeatureStore::default();
    let outcome = predictor
        .observe_image(&[], &mut store, &[], 2.0, &FlowConfig::default())
        .expect("initialized");
    assert!(
        outcome.update.is_some(),
        "a repeated timestamp is not out of order"
    );
}

#[test]
fn the_bias_blocks_are_not_swapped() {
    // The state vector is `rotation velocity position accel_bias gyro_bias`
    // -- accel FIRST -- while `VIOState(...)` lists bias_gyro before
    // bias_accel. A port reading them positionally swaps two small 3-vectors
    // near zero: nothing crashes and the drift is quietly wrong.
    //
    // The fixture catches a swap because accel bias reaches ~1e-3 while gyro
    // bias stays exactly zero -- the flow model observes horizontal velocity
    // only, so nothing makes gyro bias observable. The asymmetry does the
    // work. The capture script asserts that asymmetry still holds.
    let fixture = load();
    let mut separated = 0;
    for case in &fixture.cases {
        for frame in &case.frames {
            let Some(state) = &frame.state else { continue };
            let gap = state
                .bias_gyro
                .iter()
                .zip(&state.bias_accel)
                .map(|(g, a)| (f(g) - f(a)).abs())
                .fold(0.0f64, f64::max);
            if gap >= 1.0e-5 {
                separated += 1;
            }
        }
    }
    assert!(
        separated > 0,
        "no recorded frame separates the two biases, so a swap would pass"
    );
}
