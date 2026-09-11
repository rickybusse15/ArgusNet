//! Sensor error model: range-dependent noise, atmospheric attenuation, SNR,
//! detection probability, and bearing bias drift.
//!
//! Ported from `src/argusnet/sensing/models/noise.py`. This is the first half
//! of P4.5, split the same way P4.2 was and for the same reason: the
//! deterministic formulas are validated against recorded Python values before
//! anything stochastic moves.
//!
//! What is here:
//!
//! - the four closed-form functions — [`range_dependent_bearing_noise`],
//!   [`atmospheric_attenuation`], [`snr_at_range`], [`detection_probability`];
//! - [`BearingBiasDrift`], the one piece of *state* in the sensing path.
//!
//! What is not: `build_observations`' eight-stage rejection cascade, false-alarm
//! generation, and the bias application itself. Those consume random draws, and
//! moving them belongs with ADR-006's statistical envelopes.
//!
//! # Why the bias drift matters more than it looks
//!
//! [`TICK_SIDE_EFFECTS.md`](../../../docs/migration/TICK_SIDE_EFFECTS.md) found
//! it the hard way. `build_observations` is a pure function of its arguments —
//! no globals, no module cache — so sensing looks stateless. But it is handed
//! `sensor_models`, and each of those owns a bias walk with its own accumulated
//! value and generator. A checkpoint that omits it snaps every sensor's bias
//! back to zero on restore: a visible discontinuity in bearing error, not a
//! rounding difference.
//!
//! That is why the drift is in this half rather than deferred with the rest of
//! the stochastic work — the *state machine* around the draw is deterministic
//! and can be validated now, even though the draw itself cannot.

use serde::{Deserialize, Serialize};

/// Sensor error model parameters. Defaults match `SensorErrorConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorErrorConfig {
    // Range-dependent bearing noise.
    /// Bearing noise standard deviation at the reference range, radians.
    pub base_bearing_std_rad: f64,
    /// Growth exponent: `sigma ∝ (R / R_ref)^exponent`.
    pub range_noise_exponent: f64,
    /// Range at which noise equals `base_bearing_std_rad`.
    pub range_noise_reference_m: f64,
    /// Hard cap, so a distant target cannot produce a degenerate observation.
    pub max_bearing_std_rad: f64,

    /// Beer–Lambert extinction coefficient, per metre.
    pub atmospheric_attenuation_coeff: f64,

    // Detection.
    /// Floor on detection probability: even at extreme range there is a chance.
    pub min_detection_probability: f64,
    /// Range below which detection probability is 1.
    pub detection_range_knee_m: f64,
    /// Distance over which detection probability halves past the knee.
    pub detection_range_falloff_m: f64,

    // Bias drift.
    /// Random-walk standard deviation per second, rad/√s. Zero disables drift.
    pub bias_drift_rate_rad_per_s: f64,
    /// Absolute clamp on accumulated bias.
    pub bias_drift_max_rad: f64,

    // Signal-to-noise.
    pub snr_reference_db: f64,
    /// SNR falls as `(R_ref / R)^exponent`; 2.0 is the inverse-square law.
    pub snr_range_exponent: f64,
}

impl Default for SensorErrorConfig {
    fn default() -> Self {
        Self {
            base_bearing_std_rad: 0.015,
            range_noise_exponent: 1.5,
            range_noise_reference_m: 1000.0,
            max_bearing_std_rad: 0.15,
            atmospheric_attenuation_coeff: 0.0001,
            min_detection_probability: 0.1,
            detection_range_knee_m: 800.0,
            detection_range_falloff_m: 400.0,
            bias_drift_rate_rad_per_s: 0.0,
            bias_drift_max_rad: 0.01,
            snr_reference_db: 30.0,
            snr_range_exponent: 2.0,
        }
    }
}

/// Bearing noise standard deviation at a given range.
///
/// `sigma(R) = base * (R / R_ref)^exponent`, clamped to
/// `[base, max_bearing_std_rad]`.
///
/// The **lower** clamp is the surprising half: noise never falls below the base
/// value even at very short range, so a target at 1 m is not modelled as
/// arbitrarily precise. Removing it would make close-range observations
/// dominate every fusion update.
pub fn range_dependent_bearing_noise(
    base_std_rad: f64,
    range_m: f64,
    config: &SensorErrorConfig,
) -> f64 {
    if range_m <= 0.0 || config.range_noise_reference_m <= 0.0 {
        return base_std_rad;
    }
    let ratio = range_m / config.range_noise_reference_m;
    let scaled = base_std_rad * ratio.powf(config.range_noise_exponent);
    scaled.clamp(base_std_rad, config.max_bearing_std_rad)
}

/// Beer–Lambert transmittance over a path, in `[0, 1]` where 1 is clear.
pub fn atmospheric_attenuation(range_m: f64, config: &SensorErrorConfig) -> f64 {
    if config.atmospheric_attenuation_coeff <= 0.0 || range_m <= 0.0 {
        return 1.0;
    }
    (-config.atmospheric_attenuation_coeff * range_m).exp()
}

/// Signal-to-noise ratio in dB at a given range.
///
/// `SNR(R) = SNR_ref - exponent * 10 * log10(R / R_ref)`.
pub fn snr_at_range(range_m: f64, config: &SensorErrorConfig) -> f64 {
    if range_m <= 0.0 || config.range_noise_reference_m <= 0.0 {
        return config.snr_reference_db;
    }
    let ratio = range_m / config.range_noise_reference_m;
    // Guards log10 of a value that has underflowed toward zero. The reference
    // uses this exact threshold; it is not a general epsilon.
    if ratio < 1e-12 {
        return config.snr_reference_db;
    }
    config.snr_reference_db - config.snr_range_exponent * 10.0 * ratio.log10()
}

/// Probability of detecting a target at a given range.
///
/// Flat at 1.0 out to the knee, then halving every `detection_range_falloff_m`,
/// scaled by atmospheric transmittance and floored at
/// `min_detection_probability`.
///
/// Note the falloff is exponential (`exp(-ln2 * excess / falloff)`) despite the
/// Python docstring calling it a "sigmoid_falloff". The code is the contract;
/// the docstring is wrong and is not reproduced here.
pub fn detection_probability(
    range_m: f64,
    config: &SensorErrorConfig,
    atmospheric_transmittance: f64,
) -> f64 {
    let base = if range_m <= config.detection_range_knee_m {
        1.0
    } else if config.detection_range_falloff_m <= 0.0 {
        config.min_detection_probability
    } else {
        let excess = range_m - config.detection_range_knee_m;
        (-std::f64::consts::LN_2 * excess / config.detection_range_falloff_m).exp()
    };
    (base * atmospheric_transmittance).clamp(config.min_detection_probability, 1.0)
}

/// Per-sensor bearing bias, evolving as a clamped random walk.
///
/// # It is not an Ornstein–Uhlenbeck process
///
/// The delivery plan calls this "OU bias drift", and an earlier draft of the
/// side-effect audit repeated that. It is not OU: there is **no mean reversion**.
/// The bias accumulates `N(0, rate * sqrt(dt))` increments and is held inside
/// `[-max, +max]` by clamping alone, so it can sit against a bound
/// indefinitely rather than being pulled back toward zero. A reader who assumed
/// OU would expect the bias to decay on its own, and it does not.
///
/// The random increment is supplied by the caller, for the same reason the
/// localizer takes its particle offsets as an argument: it makes the state
/// machine a pure function and therefore checkable against Python
/// value-for-value. Generating it belongs with ADR-006's envelopes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BearingBiasDrift {
    drift_rate_rad_per_s: f64,
    max_bias_rad: f64,
    current_bias_rad: f64,
    /// `None` until the first step. The first call establishes a baseline and
    /// applies no increment, because there is no interval to integrate over.
    last_time_s: Option<f64>,
}

impl BearingBiasDrift {
    pub fn new(drift_rate_rad_per_s: f64, max_bias_rad: f64) -> Self {
        Self {
            drift_rate_rad_per_s,
            max_bias_rad,
            current_bias_rad: 0.0,
            last_time_s: None,
        }
    }

    pub fn from_config(config: &SensorErrorConfig) -> Self {
        Self::new(config.bias_drift_rate_rad_per_s, config.bias_drift_max_rad)
    }

    pub fn current_bias_rad(&self) -> f64 {
        self.current_bias_rad
    }

    pub fn last_time_s(&self) -> Option<f64> {
        self.last_time_s
    }

    /// The standard deviation of the increment for a step to `timestamp_s`.
    ///
    /// Exposed so the generator and a recorded fixture agree on the scale
    /// rather than each deriving it. Returns `None` when no increment is due —
    /// drift disabled, no baseline yet, or zero elapsed time — which is
    /// precisely when [`step`](Self::step) ignores the increment it is given.
    pub fn increment_std(&self, timestamp_s: f64) -> Option<f64> {
        if self.drift_rate_rad_per_s <= 0.0 {
            return None;
        }
        let last = self.last_time_s?;
        let dt = (timestamp_s - last).max(0.0);
        if dt <= 0.0 {
            return None;
        }
        Some(self.drift_rate_rad_per_s * dt.sqrt())
    }

    /// Advance to `timestamp_s` and return the resulting bias.
    ///
    /// `increment` is a draw from `N(0, increment_std)`; it is ignored whenever
    /// [`increment_std`](Self::increment_std) returns `None`.
    ///
    /// # A quirk carried across deliberately
    ///
    /// When drift is disabled the reference returns early **without recording
    /// the timestamp**, so `last_time_s` stays `None`. That is preserved. It is
    /// unreachable today, because the rate is fixed at construction — but if a
    /// rate ever became settable mid-run, the first enabled step would integrate
    /// over the whole elapsed run rather than one tick. Recorded here rather
    /// than silently corrected, because correcting it would change the
    /// reference's behaviour under a configuration nobody has yet built.
    pub fn step(&mut self, timestamp_s: f64, increment: f64) -> f64 {
        if self.drift_rate_rad_per_s <= 0.0 {
            return 0.0;
        }
        if self.last_time_s.is_some() && self.increment_std(timestamp_s).is_some() {
            self.current_bias_rad =
                (self.current_bias_rad + increment).clamp(-self.max_bias_rad, self.max_bias_rad);
        }
        self.last_time_s = Some(timestamp_s);
        self.current_bias_rad
    }

    /// Forget the accumulated bias and the baseline together.
    pub fn reset(&mut self) {
        self.current_bias_rad = 0.0;
        self.last_time_s = None;
    }
}

pub mod cascade;

pub use cascade::{
    evaluate_pair, Blocker, DominantLoss, FovInput, GenerationTally, Outcome, PairInput,
    RejectReason, Visibility,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SensorErrorConfig {
        SensorErrorConfig::default()
    }

    #[test]
    fn bearing_noise_equals_the_base_at_the_reference_range() {
        let value = range_dependent_bearing_noise(0.015, 1000.0, &config());
        assert!((value - 0.015).abs() < 1e-15);
    }

    #[test]
    fn bearing_noise_is_floored_at_the_base_not_reduced_at_short_range() {
        // The lower clamp is the easy one to drop. Without it a target at 10 m
        // would be modelled as ~300x more precise than one at 1 km and would
        // dominate every fusion update it touched.
        let close = range_dependent_bearing_noise(0.015, 10.0, &config());
        assert_eq!(close, 0.015);
    }

    #[test]
    fn bearing_noise_is_capped_at_extreme_range() {
        let far = range_dependent_bearing_noise(0.015, 1_000_000.0, &config());
        assert_eq!(far, config().max_bearing_std_rad);
    }

    #[test]
    fn a_non_positive_range_returns_the_base_unchanged() {
        for range in [0.0, -1.0, -1000.0] {
            assert_eq!(
                range_dependent_bearing_noise(0.015, range, &config()),
                0.015
            );
        }
    }

    #[test]
    fn attenuation_is_one_when_disabled_or_at_zero_range() {
        let clear = SensorErrorConfig {
            atmospheric_attenuation_coeff: 0.0,
            ..config()
        };
        assert_eq!(atmospheric_attenuation(1000.0, &clear), 1.0);
        assert_eq!(atmospheric_attenuation(0.0, &config()), 1.0);
    }

    #[test]
    fn attenuation_falls_monotonically_with_range() {
        let mut previous = 1.0;
        for range in [100.0, 500.0, 1000.0, 5000.0, 20_000.0] {
            let value = atmospheric_attenuation(range, &config());
            assert!(value < previous, "attenuation rose at {range} m");
            assert!((0.0..=1.0).contains(&value));
            previous = value;
        }
    }

    #[test]
    fn snr_is_the_reference_value_at_the_reference_range() {
        assert!((snr_at_range(1000.0, &config()) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn snr_drops_by_the_inverse_square_law() {
        // Doubling the range costs 2 * 10 * log10(2) = 6.02 dB at exponent 2.
        let near = snr_at_range(1000.0, &config());
        let far = snr_at_range(2000.0, &config());
        assert!(
            (near - far - 6.0206).abs() < 1e-3,
            "drop was {}",
            near - far
        );
    }

    #[test]
    fn detection_is_certain_inside_the_knee_and_floored_outside_it() {
        let clear = SensorErrorConfig {
            atmospheric_attenuation_coeff: 0.0,
            ..config()
        };
        assert_eq!(detection_probability(0.0, &clear, 1.0), 1.0);
        assert_eq!(detection_probability(800.0, &clear, 1.0), 1.0);
        // Far past the knee the exponential underflows below the floor.
        assert_eq!(
            detection_probability(100_000.0, &clear, 1.0),
            clear.min_detection_probability
        );
    }

    #[test]
    fn detection_halves_every_falloff_distance_past_the_knee() {
        let clear = config();
        let at_knee = detection_probability(800.0, &clear, 1.0);
        let one_falloff = detection_probability(800.0 + 400.0, &clear, 1.0);
        assert!((at_knee - 1.0).abs() < 1e-12);
        assert!((one_falloff - 0.5).abs() < 1e-12, "was {one_falloff}");
    }

    #[test]
    fn transmittance_scales_detection_but_never_below_the_floor() {
        let clear = config();
        let half_visibility = detection_probability(400.0, &clear, 0.5);
        assert!((half_visibility - 0.5).abs() < 1e-12);
        // Even zero transmittance cannot drive it under the floor.
        assert_eq!(
            detection_probability(400.0, &clear, 0.0),
            clear.min_detection_probability
        );
    }

    #[test]
    fn a_zero_falloff_collapses_straight_to_the_floor() {
        let sharp = SensorErrorConfig {
            detection_range_falloff_m: 0.0,
            ..config()
        };
        assert_eq!(
            detection_probability(801.0, &sharp, 1.0),
            sharp.min_detection_probability
        );
        assert_eq!(detection_probability(800.0, &sharp, 1.0), 1.0);
    }

    // -- bias drift ------------------------------------------------------

    fn drifting() -> BearingBiasDrift {
        BearingBiasDrift::new(1e-4, 0.01)
    }

    #[test]
    fn the_first_step_only_establishes_a_baseline() {
        let mut drift = drifting();
        assert_eq!(drift.last_time_s(), None);
        // A large increment on the first step must be ignored: there is no
        // interval to integrate over yet.
        assert_eq!(drift.step(10.0, 0.005), 0.0);
        assert_eq!(drift.current_bias_rad(), 0.0);
        assert_eq!(drift.last_time_s(), Some(10.0));
    }

    #[test]
    fn increments_accumulate_after_the_baseline() {
        let mut drift = drifting();
        drift.step(0.0, 0.0);
        drift.step(1.0, 0.002);
        drift.step(2.0, 0.001);
        assert!((drift.current_bias_rad() - 0.003).abs() < 1e-15);
    }

    #[test]
    fn the_bias_is_clamped_and_stays_against_the_bound() {
        // The clamp, not mean reversion, is what bounds this walk. Pushing hard
        // in one direction leaves the bias pinned rather than pulled back --
        // which is exactly how it differs from an Ornstein-Uhlenbeck process.
        let mut drift = drifting();
        drift.step(0.0, 0.0);
        for tick in 1..=20 {
            drift.step(tick as f64, 0.005);
        }
        assert_eq!(drift.current_bias_rad(), 0.01);
        // And it does not relax on its own when increments stop.
        for tick in 21..=40 {
            drift.step(tick as f64, 0.0);
        }
        assert_eq!(
            drift.current_bias_rad(),
            0.01,
            "a clamped walk holds its value; an OU process would decay"
        );
    }

    #[test]
    fn a_repeated_timestamp_applies_no_increment() {
        let mut drift = drifting();
        drift.step(0.0, 0.0);
        drift.step(1.0, 0.001);
        let before = drift.current_bias_rad();
        drift.step(1.0, 0.004);
        assert_eq!(drift.current_bias_rad(), before, "dt of zero adds nothing");
    }

    #[test]
    fn time_running_backwards_applies_no_increment() {
        // The reference clamps dt at zero rather than rejecting it.
        let mut drift = drifting();
        drift.step(10.0, 0.0);
        let before = drift.current_bias_rad();
        drift.step(5.0, 0.004);
        assert_eq!(drift.current_bias_rad(), before);
        assert_eq!(
            drift.last_time_s(),
            Some(5.0),
            "the baseline still moves, matching the reference"
        );
    }

    #[test]
    fn disabled_drift_returns_zero_and_records_no_baseline() {
        // The quirk documented on `step`: the early return skips the timestamp
        // update. Pinned so a future edit cannot quietly change it.
        let mut disabled = BearingBiasDrift::new(0.0, 0.01);
        assert_eq!(disabled.step(5.0, 0.5), 0.0);
        assert_eq!(disabled.current_bias_rad(), 0.0);
        assert_eq!(
            disabled.last_time_s(),
            None,
            "the reference does not record a baseline when drift is off"
        );
    }

    #[test]
    fn increment_std_scales_with_the_square_root_of_the_interval() {
        let mut drift = drifting();
        assert_eq!(
            drift.increment_std(1.0),
            None,
            "no baseline yet, so no increment is due"
        );
        drift.step(0.0, 0.0);
        let one_second = drift.increment_std(1.0).expect("due");
        let four_seconds = drift.increment_std(4.0).expect("due");
        assert!((one_second - 1e-4).abs() < 1e-18);
        assert!(
            (four_seconds / one_second - 2.0).abs() < 1e-12,
            "four seconds must scale by sqrt(4) = 2"
        );
    }

    #[test]
    fn reset_clears_the_bias_and_the_baseline_together() {
        let mut drift = drifting();
        drift.step(0.0, 0.0);
        drift.step(1.0, 0.003);
        drift.reset();
        assert_eq!(drift.current_bias_rad(), 0.0);
        assert_eq!(drift.last_time_s(), None);
        // So the next step is a baseline again, not an increment.
        assert_eq!(drift.step(2.0, 0.004), 0.0);
    }

    #[test]
    fn drift_state_round_trips_through_json() {
        // It is checkpoint state -- see TICK_SIDE_EFFECTS.md -- so it has to
        // serialize, and the baseline has to survive with the bias.
        let mut drift = drifting();
        drift.step(0.0, 0.0);
        drift.step(1.0, 0.002);
        let json = serde_json::to_string(&drift).expect("serializable");
        let restored: BearingBiasDrift = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(restored, drift);
        assert_eq!(restored.last_time_s(), Some(1.0));
    }
}
