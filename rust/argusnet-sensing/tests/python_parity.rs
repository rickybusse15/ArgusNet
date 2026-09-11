//! Value-for-value parity with the Python sensor error model.
//!
//! `tests/fixtures/sensor_noise.json` records what
//! `src/argusnet/sensing/models/noise.py` produces: the four closed-form
//! functions across ranges that straddle every branch, and the bias-drift state
//! machine driven step by step with the increment numpy drew recorded alongside
//! the resulting bias.
//!
//! Because the increments are recorded rather than redrawn, the drift is a pure
//! function of the fixture and can be compared exactly — the same trick that
//! makes the localizer checkable in `argusnet-localization`.

use argusnet_sensing::{
    atmospheric_attenuation, detection_probability, range_dependent_bearing_noise, snr_at_range,
    BearingBiasDrift, SensorErrorConfig,
};
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;

/// A float the fixture wrote as a string, parsed exactly.
///
/// `serde_json`'s number parser is off by one ULP on values like
/// `0.9999000049998333`, which it reads as `...332`. Rust's own
/// `str::parse::<f64>` is exact, so every value these tests compare travels as
/// a string and is parsed here. Without this the comparison below could only be
/// approximate, and an approximate comparison of closed forms would hide
/// exactly the kind of transcription error it exists to catch.
fn exact<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(serde::de::Error::custom)
}

fn exact_optional<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    let text = Option::<String>::deserialize(deserializer)?;
    match text {
        None => Ok(None),
        Some(text) => text.parse().map(Some).map_err(serde::de::Error::custom),
    }
}

fn exact_map<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, f64>, D::Error> {
    let raw = BTreeMap::<String, String>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|(key, value)| {
            value
                .parse()
                .map(|v| (key, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

#[derive(Deserialize)]
struct Fixture {
    closed_form: Vec<ClosedFormCase>,
    drift: Vec<DriftCase>,
}

#[derive(Deserialize)]
struct ClosedFormCase {
    name: String,
    config: ConfigFields,
    rows: Vec<Row>,
}

/// Mirrors `SensorErrorConfig`, with every field string-encoded for exactness.
#[derive(Deserialize)]
struct ConfigFields {
    #[serde(deserialize_with = "exact")]
    base_bearing_std_rad: f64,
    #[serde(deserialize_with = "exact")]
    range_noise_exponent: f64,
    #[serde(deserialize_with = "exact")]
    range_noise_reference_m: f64,
    #[serde(deserialize_with = "exact")]
    max_bearing_std_rad: f64,
    #[serde(deserialize_with = "exact")]
    atmospheric_attenuation_coeff: f64,
    #[serde(deserialize_with = "exact")]
    min_detection_probability: f64,
    #[serde(deserialize_with = "exact")]
    detection_range_knee_m: f64,
    #[serde(deserialize_with = "exact")]
    detection_range_falloff_m: f64,
    #[serde(deserialize_with = "exact")]
    bias_drift_rate_rad_per_s: f64,
    #[serde(deserialize_with = "exact")]
    bias_drift_max_rad: f64,
    #[serde(deserialize_with = "exact")]
    snr_reference_db: f64,
    #[serde(deserialize_with = "exact")]
    snr_range_exponent: f64,
}

impl ConfigFields {
    fn to_config(&self) -> SensorErrorConfig {
        SensorErrorConfig {
            base_bearing_std_rad: self.base_bearing_std_rad,
            range_noise_exponent: self.range_noise_exponent,
            range_noise_reference_m: self.range_noise_reference_m,
            max_bearing_std_rad: self.max_bearing_std_rad,
            atmospheric_attenuation_coeff: self.atmospheric_attenuation_coeff,
            min_detection_probability: self.min_detection_probability,
            detection_range_knee_m: self.detection_range_knee_m,
            detection_range_falloff_m: self.detection_range_falloff_m,
            bias_drift_rate_rad_per_s: self.bias_drift_rate_rad_per_s,
            bias_drift_max_rad: self.bias_drift_max_rad,
            snr_reference_db: self.snr_reference_db,
            snr_range_exponent: self.snr_range_exponent,
        }
    }
}

#[derive(Deserialize)]
struct Row {
    #[serde(deserialize_with = "exact")]
    range_m: f64,
    #[serde(deserialize_with = "exact")]
    bearing_std_rad: f64,
    #[serde(deserialize_with = "exact")]
    transmittance: f64,
    #[serde(deserialize_with = "exact")]
    snr_db: f64,
    /// Keyed by the transmittance the value was computed at, as a string
    /// because JSON object keys are strings.
    #[serde(deserialize_with = "exact_map")]
    detection_probability: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct DriftCase {
    name: String,
    #[serde(deserialize_with = "exact")]
    drift_rate_rad_per_s: f64,
    #[serde(deserialize_with = "exact")]
    max_bias_rad: f64,
    steps: Vec<DriftStep>,
}

#[derive(Deserialize)]
struct DriftStep {
    #[serde(deserialize_with = "exact")]
    timestamp_s: f64,
    /// `None` when the step drew nothing: no baseline yet, zero elapsed time,
    /// or drift disabled.
    #[serde(deserialize_with = "exact_optional")]
    increment: Option<f64>,
    #[serde(deserialize_with = "exact")]
    expected_bias_rad: f64,
}

/// These are closed forms over a handful of operations, not 200-element sums,
/// so there is no pairwise-summation difference to absorb. Exact equality is
/// the right assertion, and it holds — see the localizer's parity test for the
/// case where it does not and why.
#[track_caller]
fn assert_exact(actual: f64, expected: f64, what: &str) {
    assert_eq!(
        actual, expected,
        "{what}: rust {actual:?} vs python {expected:?}"
    );
}

fn fixture() -> Fixture {
    let path = format!(
        "{}/../../tests/fixtures/sensor_noise.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

#[test]
fn the_closed_form_functions_reproduce_python_exactly() {
    let fixture = fixture();
    let mut rows_checked = 0usize;

    for case in &fixture.closed_form {
        let config = case.config.to_config();
        for row in &case.rows {
            let where_ = format!("{} at {} m", case.name, row.range_m);

            assert_exact(
                range_dependent_bearing_noise(config.base_bearing_std_rad, row.range_m, &config),
                row.bearing_std_rad,
                &format!("{where_} bearing_std_rad"),
            );
            assert_exact(
                atmospheric_attenuation(row.range_m, &config),
                row.transmittance,
                &format!("{where_} transmittance"),
            );
            assert_exact(
                snr_at_range(row.range_m, &config),
                row.snr_db,
                &format!("{where_} snr_db"),
            );

            for (key, expected) in &row.detection_probability {
                let transmittance: f64 = key.parse().expect("numeric key");
                assert_exact(
                    detection_probability(row.range_m, &config, transmittance),
                    *expected,
                    &format!("{where_} detection_probability at t={transmittance}"),
                );
            }
            rows_checked += 1;
        }
    }

    assert!(rows_checked >= 50, "only {rows_checked} rows checked");
}

#[test]
fn the_bias_drift_state_machine_replays_exactly() {
    let fixture = fixture();
    let mut steps_checked = 0usize;
    let mut increments_applied = 0usize;

    for case in &fixture.drift {
        let mut drift = BearingBiasDrift::new(case.drift_rate_rad_per_s, case.max_bias_rad);
        for (index, step) in case.steps.iter().enumerate() {
            // Zero stands in where Python drew nothing; the state machine must
            // ignore it on exactly those steps, which is what makes this a test
            // of the branching rather than of the arithmetic alone.
            let increment = step.increment.unwrap_or(0.0);
            let bias = drift.step(step.timestamp_s, increment);
            assert_exact(
                bias,
                step.expected_bias_rad,
                &format!("{} step {index} (t={})", case.name, step.timestamp_s),
            );
            steps_checked += 1;
            if step.increment.is_some() {
                increments_applied += 1;
            }
        }
    }

    assert!(steps_checked >= 30, "only {steps_checked} steps checked");
    assert!(
        increments_applied >= 20,
        "only {increments_applied} steps actually drew an increment; the rest \
         were baseline or no-op steps, which exercise the branching but not the walk"
    );
}

/// The fixture must disagree with a wrong implementation, not just with itself.
///
/// Each of these is a plausible port error that produces a *shaped* result — it
/// would look fine in isolation. If the fixture cannot distinguish them, it is
/// not evidence of anything.
#[test]
fn the_fixture_covers_the_branches_a_wrong_port_would_collapse() {
    let fixture = fixture();

    let default = fixture
        .closed_form
        .iter()
        .find(|case| case.name == "default")
        .expect("default case present");

    // Both clamps on the bearing noise must be exercised.
    let default_config = default.config.to_config();
    let base = default_config.base_bearing_std_rad;
    assert!(
        default
            .rows
            .iter()
            .any(|row| row.range_m > 0.0 && row.bearing_std_rad == base),
        "no row sits on the lower clamp, so dropping it would go unnoticed"
    );
    assert!(
        default
            .rows
            .iter()
            .any(|row| row.bearing_std_rad == default_config.max_bearing_std_rad),
        "no row sits on the upper clamp"
    );

    // Both sides of the detection knee.
    assert!(
        default.rows.iter().any(|row| row.range_m < 800.0),
        "no row inside the detection knee"
    );
    assert!(
        default.rows.iter().any(|row| row.range_m > 800.0),
        "no row past the detection knee"
    );

    // The drift cases must include a step that draws nothing and one that
    // clamps, or the branching is untested.
    let disabled = fixture
        .drift
        .iter()
        .find(|case| case.name == "disabled")
        .expect("disabled case present");
    assert!(
        disabled.steps.iter().all(|step| step.increment.is_none()),
        "the disabled case drew an increment, so drift was not actually off"
    );
    assert!(
        disabled
            .steps
            .iter()
            .all(|step| step.expected_bias_rad == 0.0),
        "disabled drift produced a non-zero bias"
    );

    let clamped = fixture
        .drift
        .iter()
        .find(|case| case.name == "tight_clamp")
        .expect("tight_clamp case present");
    assert!(
        clamped
            .steps
            .iter()
            .any(|step| step.expected_bias_rad.abs() == clamped.max_bias_rad),
        "the tight_clamp case never reached its clamp, so the clamp is untested"
    );

    let non_monotonic = fixture
        .drift
        .iter()
        .find(|case| case.name == "non_monotonic")
        .expect("non_monotonic case present");
    assert!(
        non_monotonic
            .steps
            .iter()
            .filter(|step| step.increment.is_none())
            .count()
            >= 2,
        "the non-monotonic case must contain repeated and backwards timestamps \
         that draw nothing"
    );
}
