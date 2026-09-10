//! Value-for-value parity with the Python estimator.
//!
//! P4.2: *"First validate all non-random formulas using recorded offsets."*
//!
//! `tests/fixtures/localization_steps.json` records, for every step, the
//! particle offsets numpy drew, the footprint scores the native scorer
//! returned, and the estimate Python produced from them. With the random draw
//! recorded rather than repeated, the estimator is a pure function of the
//! fixture — so this compares numbers, not distributions.
//!
//! That is the whole point of doing P4.2 in two halves. A statistical
//! comparison can only tell you the port is *plausible*; this tells you the
//! arithmetic is the same one.

use argusnet_localization::{GridLocalizer, LocalizationConfig, LocalizationStatus, StepInput};
use serde::{Deserialize, Deserializer};

/// A float the fixture wrote as a string, parsed exactly.
///
/// `serde_json`'s float parser is off by one ULP on many values -- measured at
/// **15.25% of the 11,061 literals** in an earlier, number-encoded version of
/// this fixture. Rust's own `str::parse::<f64>` is exact, so every value
/// travels as a string and is parsed here.
///
/// This matters for what the tolerance below *means*: before the change, its
/// residual mixed the parser's error with any real numerical difference, so
/// neither could be attributed.
fn exact<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    String::deserialize(deserializer)?
        .parse()
        .map_err(serde::de::Error::custom)
}

fn exact_vec<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
    Vec::<String>::deserialize(deserializer)?
        .into_iter()
        .map(|text| text.parse().map_err(serde::de::Error::custom))
        .collect()
}

fn exact_triple<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f64; 3], D::Error> {
    let values = exact_vec(deserializer)?;
    values.try_into().map_err(|v: Vec<f64>| {
        serde::de::Error::custom(format!("expected 3 values, got {}", v.len()))
    })
}

fn exact_pairs<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<[f64; 2]>>, D::Error> {
    let raw = Option::<Vec<Vec<String>>>::deserialize(deserializer)?;
    let Some(raw) = raw else { return Ok(None) };
    raw.into_iter()
        .map(|pair| {
            let parsed: Result<Vec<f64>, _> = pair
                .into_iter()
                .map(|text| text.parse().map_err(serde::de::Error::custom))
                .collect();
            parsed.and_then(|values| {
                values.try_into().map_err(|v: Vec<f64>| {
                    serde::de::Error::custom(format!("expected 2 values, got {}", v.len()))
                })
            })
        })
        .collect::<Result<Vec<[f64; 2]>, _>>()
        .map(Some)
}

fn exact_optional_vec<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<f64>>, D::Error> {
    let raw = Option::<Vec<String>>::deserialize(deserializer)?;
    match raw {
        None => Ok(None),
        Some(values) => values
            .into_iter()
            .map(|text| text.parse().map_err(serde::de::Error::custom))
            .collect::<Result<Vec<f64>, _>>()
            .map(Some),
    }
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    config: ConfigFields,
    steps: Vec<Step>,
    expected_fused: Option<Expected>,
    expected_any_timed_out: bool,
}

#[derive(Deserialize)]
struct ConfigFields {
    #[serde(deserialize_with = "exact")]
    search_radius_m: f64,
    particle_count: usize,
    #[serde(deserialize_with = "exact")]
    convergence_threshold_m: f64,
    #[serde(deserialize_with = "exact")]
    min_coverage_to_localize: f64,
    #[serde(deserialize_with = "exact")]
    confidence_decay: f64,
    #[serde(deserialize_with = "exact")]
    annealing_rate: f64,
    localization_timeout_steps: u32,
}

#[derive(Deserialize)]
struct Step {
    platform_id: String,
    #[serde(deserialize_with = "exact_triple")]
    nominal_position: [f64; 3],
    #[serde(deserialize_with = "exact")]
    heading_rad: f64,
    #[serde(deserialize_with = "exact")]
    timestamp_s: f64,
    #[serde(deserialize_with = "exact")]
    coverage_fraction: f64,
    /// `None` on the no-map path, where neither is reached.
    #[serde(deserialize_with = "exact_pairs")]
    offsets: Option<Vec<[f64; 2]>>,
    #[serde(deserialize_with = "exact_optional_vec")]
    scores: Option<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    #[serde(deserialize_with = "exact_triple")]
    position_estimate: [f64; 3],
    #[serde(deserialize_with = "exact")]
    position_std_m: f64,
    #[serde(deserialize_with = "exact")]
    confidence: f64,
}

/// Relative tolerance for a value that went through a 200-element sum.
///
/// Not bit-equality, and the reason is specific: numpy switches to pairwise
/// summation for arrays of this size, so its rounding differs from a sequential
/// accumulation even when every input is identical.
///
/// **That attribution took two measurements to earn.** The first read 7.86e-16
/// with the fixture encoding floats as JSON numbers — and `serde_json`'s parser
/// is off by one ULP on 15.25% of them, so the inputs were not provably the same
/// on both sides and the residual could not be attributed to anything. Encoding
/// them as strings and parsing with `str::parse` (see `exact`) makes the inputs
/// bit-identical, and the gap becomes:
///
/// | particles | agreement |
/// |---|---|
/// | 4 | bit-identical (pinned by a unit test in `lib.rs`) |
/// | 200 | **8.98e-16 relative**, about 4 ULP |
///
/// Slightly *larger* than the confounded figure, which is worth noting: the
/// parser error was partly cancelling the real difference rather than inflating
/// it. With identical inputs the residual can only come from the arithmetic, so
/// summation order is now the supported explanation rather than a plausible one.
///
/// `1e-12` sits about 1,100x above that and orders of magnitude below anything a
/// formula error would produce. Setting it to `0.0` reproduces the 8.98e-16
/// failure, which is how the number was obtained.
const RELATIVE_TOLERANCE: f64 = 1e-12;

#[track_caller]
fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    let error = (actual - expected).abs() / scale;
    assert!(
        error <= RELATIVE_TOLERANCE,
        "{what}: rust {actual:?} vs python {expected:?} (relative error {error:e})"
    );
}

fn fixture() -> Fixture {
    let path = format!(
        "{}/../../tests/fixtures/localization_steps.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

#[test]
fn every_recorded_step_reproduces_pythons_estimate() {
    let fixture = fixture();
    assert!(!fixture.cases.is_empty(), "fixture has no cases");

    let mut steps_checked = 0usize;
    let mut particle_steps = 0usize;

    for case in &fixture.cases {
        let config = LocalizationConfig {
            search_radius_m: case.config.search_radius_m,
            particle_count: case.config.particle_count,
            convergence_threshold_m: case.config.convergence_threshold_m,
            min_coverage_to_localize: case.config.min_coverage_to_localize,
            confidence_decay: case.config.confidence_decay,
            annealing_rate: case.config.annealing_rate,
            localization_timeout_steps: case.config.localization_timeout_steps,
        };
        let mut localizer = GridLocalizer::new(config);

        for (index, step) in case.steps.iter().enumerate() {
            let input = StepInput {
                platform_id: &step.platform_id,
                nominal_position: step.nominal_position,
                heading_rad: step.heading_rad,
                timestamp_s: step.timestamp_s,
                coverage_fraction: step.coverage_fraction,
            };
            // The no-map path records no offsets, because it returns before
            // drawing any. Empty slices exercise the same early return.
            let empty_offsets: Vec<[f64; 2]> = Vec::new();
            let empty_scores: Vec<f64> = Vec::new();
            let offsets = step.offsets.as_ref().unwrap_or(&empty_offsets);
            let scores = step.scores.as_ref().unwrap_or(&empty_scores);

            let estimate = localizer
                .update(input, offsets, scores)
                .unwrap_or_else(|error| panic!("{} step {index}: {error}", case.name));

            let where_ = format!("{} step {index}", case.name);
            for axis in 0..3 {
                assert_close(
                    estimate.position_estimate[axis],
                    step.expected.position_estimate[axis],
                    &format!("{where_} position[{axis}]"),
                );
            }
            assert_close(
                estimate.position_std_m,
                step.expected.position_std_m,
                &format!("{where_} position_std_m"),
            );
            assert_close(
                estimate.confidence,
                step.expected.confidence,
                &format!("{where_} confidence"),
            );

            steps_checked += 1;
            if !offsets.is_empty() {
                particle_steps += 1;
            }
        }

        assert_eq!(
            localizer.any_timed_out(),
            case.expected_any_timed_out,
            "{}: timeout state diverged",
            case.name
        );

        let ids: Vec<&str> = case.steps.iter().map(|s| s.platform_id.as_str()).collect();
        match (&case.expected_fused, localizer.fuse_estimates(&ids)) {
            (None, None) => {}
            (Some(expected), Some(actual)) => {
                for axis in 0..3 {
                    assert_close(
                        actual.position_estimate[axis],
                        expected.position_estimate[axis],
                        &format!("{} fused position[{axis}]", case.name),
                    );
                }
                assert_close(
                    actual.position_std_m,
                    expected.position_std_m,
                    &format!("{} fused position_std_m", case.name),
                );
                assert_close(
                    actual.confidence,
                    expected.confidence,
                    &format!("{} fused confidence", case.name),
                );
            }
            (expected, actual) => panic!(
                "{}: fusion disagreed about whether a result exists (python: {}, rust: {})",
                case.name,
                expected.is_some(),
                actual.is_some()
            ),
        }
    }

    // Guards the way this test could pass for the wrong reason: a fixture that
    // silently lost its particle data would make every comparison trivial.
    assert!(steps_checked >= 15, "only {steps_checked} steps checked");
    assert!(
        particle_steps >= 10,
        "only {particle_steps} steps exercised the matching path; the rest were \
         no-map early returns, which validate almost none of the formula"
    );
}

#[test]
fn the_fixture_covers_every_behavioural_path() {
    // Named separately from the parity assertions, because "the numbers match"
    // is only worth as much as the paths the numbers came from.
    let fixture = fixture();
    let names: Vec<&str> = fixture
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect();
    for required in [
        "no_map_then_matching",
        "annealing",
        "two_platforms_fused",
        "timeout",
    ] {
        assert!(
            names.contains(&required),
            "fixture lost the {required} case"
        );
    }

    let timeout_case = fixture
        .cases
        .iter()
        .find(|case| case.name == "timeout")
        .expect("present");
    assert!(
        timeout_case.expected_any_timed_out,
        "the timeout case must actually reach its timeout"
    );

    let no_map = fixture
        .cases
        .iter()
        .find(|case| case.name == "no_map_then_matching")
        .expect("present");
    assert!(
        no_map.steps.iter().any(|step| step.offsets.is_none()),
        "the no-map case must contain a step that returned before drawing"
    );
    assert!(
        no_map.steps.iter().any(|step| step.offsets.is_some()),
        "and a step that went on to match, so the stored zero-confidence \
         estimate feeds the next blend"
    );
}

#[test]
fn the_status_the_port_adds_agrees_with_pythons_numbers() {
    // `LocalizationStatus` is new in the port -- Python has no equivalent -- so
    // nothing in the fixture states it directly. What can be checked is that it
    // never contradicts the numbers Python did record.
    let fixture = fixture();
    for case in &fixture.cases {
        let config = LocalizationConfig {
            search_radius_m: case.config.search_radius_m,
            particle_count: case.config.particle_count,
            convergence_threshold_m: case.config.convergence_threshold_m,
            min_coverage_to_localize: case.config.min_coverage_to_localize,
            confidence_decay: case.config.confidence_decay,
            annealing_rate: case.config.annealing_rate,
            localization_timeout_steps: case.config.localization_timeout_steps,
        };
        let mut localizer = GridLocalizer::new(config);

        for step in &case.steps {
            let empty_offsets: Vec<[f64; 2]> = Vec::new();
            let empty_scores: Vec<f64> = Vec::new();
            let estimate = localizer
                .update(
                    StepInput {
                        platform_id: &step.platform_id,
                        nominal_position: step.nominal_position,
                        heading_rad: step.heading_rad,
                        timestamp_s: step.timestamp_s,
                        coverage_fraction: step.coverage_fraction,
                    },
                    step.offsets.as_ref().unwrap_or(&empty_offsets),
                    step.scores.as_ref().unwrap_or(&empty_scores),
                )
                .expect("step");

            match estimate.status {
                LocalizationStatus::MapUnavailable => {
                    assert_eq!(
                        step.expected.confidence, 0.0,
                        "{}: an unavailable map must report zero confidence",
                        case.name
                    );
                    assert!(step.offsets.is_none(), "no particles were drawn");
                }
                LocalizationStatus::TimedOut => {
                    assert_eq!(
                        estimate.confidence, 1.0,
                        "{}: the timeout path forces confidence to 1.0",
                        case.name
                    );
                }
                LocalizationStatus::Converged => assert!(
                    estimate.position_std_m <= config.convergence_threshold_m,
                    "{}: converged with a spread above the threshold",
                    case.name
                ),
                LocalizationStatus::Converging => assert!(
                    estimate.position_std_m > config.convergence_threshold_m,
                    "{}: still converging with a spread inside the threshold",
                    case.name
                ),
            }
        }
    }
}
