//! Grid-based platform self-localization.
//!
//! Ported from `src/argusnet/localization/engine.py::GridLocalizer`, which the
//! Phase 0 profile puts at **40% of the Python tick loop** — the largest single
//! item in it, and the reason P4.2 comes first among the stage ports.
//!
//! # What this crate does and does not do yet
//!
//! P4.2 splits the port deliberately: *"First validate all non-random formulas
//! using recorded offsets. Then migrate PRNG explicitly under ADR-006 with
//! predeclared statistical envelopes."*
//!
//! This is the first half. Every formula is here — annealing, particle
//! weighting, the weighted mean and spread, the confidence blend, the timeout
//! path and inverse-variance fusion — and the particle offsets arrive as an
//! argument rather than being drawn. That makes the whole estimator a pure
//! function of its inputs, which is what lets it be checked against Python
//! value-for-value instead of statistically.
//!
//! The offsets parameter is **a deterministic test input, not a second RNG
//! authority** — the delivery plan is explicit about that. The generator that
//! fills it belongs to P4.2b, alongside the statistical envelopes ADR-006 asks
//! for.
//!
//! # Compatibility notes carried across on purpose
//!
//! Several of these look like bugs and are not; they are the reference's
//! behaviour, and P4.2 requires preserving *"reset behavior, no-map early
//! returns, elevation handling and duplicate/missing platform semantics"*.
//! Each is commented where it appears.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Parameters controlling estimator behaviour. Defaults match `engine.py`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalizationConfig {
    /// Uncertainty radius the particle cloud starts from, metres.
    pub search_radius_m: f64,
    pub particle_count: usize,
    /// Position spread below which the estimate counts as converged.
    pub convergence_threshold_m: f64,
    /// Map coverage needed before localizing at all.
    pub min_coverage_to_localize: f64,
    /// How much of the previous step's confidence carries over.
    pub confidence_decay: f64,
    /// Per-step fractional shrink of the search radius.
    pub annealing_rate: f64,
    /// Steps before confidence is forced to 1.0. Zero disables it.
    pub localization_timeout_steps: u32,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            search_radius_m: 80.0,
            particle_count: 200,
            convergence_threshold_m: 15.0,
            min_coverage_to_localize: 0.15,
            confidence_decay: 0.50,
            annealing_rate: 0.02,
            localization_timeout_steps: 200,
        }
    }
}

/// Why an estimate reads the way it does.
///
/// New in the port, and required by P4.2: *"Add `LocalizationStatus` to the
/// runtime/DTO contract so timeout, unavailable map and actual convergence are
/// distinguishable."*
///
/// The Python estimator conflates them. A caller sees `confidence == 1.0` and
/// cannot tell whether the filter converged or whether the timeout simply
/// fired and overwrote the number — and those mean opposite things about
/// whether the position can be trusted. Likewise `confidence == 0.0` from the
/// no-map path is not a failed match; it is "no map yet".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalizationStatus {
    /// Coverage is below `min_coverage_to_localize`; nothing was matched and
    /// the reported position is the nominal one.
    MapUnavailable,
    /// Matching ran; spread is still above the convergence threshold.
    Converging,
    /// Matching ran and the spread is within the convergence threshold.
    Converged,
    /// The step budget ran out and confidence was forced to 1.0. **This is not
    /// convergence.** It is the compatibility path, retained because mission
    /// gating depends on it; P4.2 keeps it *"only under the compatibility
    /// profile until a separate mission-gating change is reviewed."*
    TimedOut,
}

/// One platform's estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalizationEstimate {
    pub platform_id: String,
    pub timestamp_s: f64,
    /// Metres. `z` is copied from the nominal position — the estimator only
    /// searches in the horizontal plane, and P4.2 lists elevation handling
    /// among the semantics to preserve.
    pub position_estimate: [f64; 3],
    pub heading_rad: f64,
    /// One-sigma horizontal spread.
    pub position_std_m: f64,
    pub confidence: f64,
    pub status: LocalizationStatus,
}

/// What the estimator needs to know about one step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepInput<'a> {
    pub platform_id: &'a str,
    pub nominal_position: [f64; 3],
    pub heading_rad: f64,
    pub timestamp_s: f64,
    /// Fraction of the map covered so far.
    pub coverage_fraction: f64,
}

/// Particle offsets for one step, in metres, relative to the nominal position.
///
/// Passed in rather than drawn. See the module comment: this is the deterministic
/// input that makes the estimator checkable against Python value-for-value, and
/// it is explicitly not a second RNG authority.
///
/// The caller is responsible for having scaled these correctly;
/// [`GridLocalizer::particle_scale`] computes the scale the reference uses, so a
/// generator and a fixture agree on it.
pub type ParticleOffsets = [[f64; 2]];

/// Why a step could not be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalizationError {
    /// Offsets and scores must describe the same particles.
    ParticleCountMismatch {
        offsets: usize,
        scores: usize,
    },
    /// A step with no particles has no cloud to take a mean over.
    NoParticles,
    NotFinite(&'static str),
}

impl std::fmt::Display for LocalizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalizationError::ParticleCountMismatch { offsets, scores } => write!(
                formatter,
                "{offsets} particle offsets but {scores} scores; they must describe the same particles"
            ),
            LocalizationError::NoParticles => {
                write!(formatter, "a localization step needs at least one particle")
            }
            LocalizationError::NotFinite(what) => write!(formatter, "{what} was not finite"),
        }
    }
}

impl std::error::Error for LocalizationError {}

/// Particle-filter-inspired pose estimator over an accumulated coverage map.
///
/// As the map fills in, matching a sensor footprint against it becomes more
/// discriminative, so the spread narrows and confidence rises.
#[derive(Debug, Clone, Default)]
pub struct GridLocalizer {
    config: LocalizationConfig,
    /// `BTreeMap` rather than a hash map so iteration order is deterministic.
    /// `all_estimates` feeds fusion and replay output, and a nondeterministic
    /// order there would make runs differ for no reason.
    estimates: BTreeMap<String, LocalizationEstimate>,
    step_counts: BTreeMap<String, u32>,
    timed_out: BTreeSet<String>,
}

impl GridLocalizer {
    pub fn new(config: LocalizationConfig) -> Self {
        Self {
            config,
            estimates: BTreeMap::new(),
            step_counts: BTreeMap::new(),
            timed_out: BTreeSet::new(),
        }
    }

    pub fn config(&self) -> LocalizationConfig {
        self.config
    }

    /// The spread the particle cloud should be drawn with on this step.
    ///
    /// Exposed so the generator in P4.2b and the recorded fixtures agree on it
    /// rather than each computing it. Two shrinking factors compose: annealing
    /// over steps, and coverage — a better-known map needs a smaller search.
    ///
    /// Both floors (`0.05` and `0.1`) are the reference's and are load-bearing:
    /// without them the cloud collapses to a point and the estimate freezes.
    pub fn particle_scale(&self, step: u32, coverage_fraction: f64) -> f64 {
        let annealing = (1.0 - self.config.annealing_rate)
            .powi(step as i32)
            .max(0.05);
        self.config.search_radius_m * (1.0 - coverage_fraction).max(0.1) * annealing
    }

    /// How many steps this platform has taken, including no-map ones.
    ///
    /// The reference increments **before** the no-map early return, so steps
    /// spent waiting for coverage still count toward the timeout. Preserved:
    /// changing it would move when the timeout fires for every existing run.
    pub fn step_count(&self, platform_id: &str) -> u32 {
        self.step_counts.get(platform_id).copied().unwrap_or(0)
    }

    /// Advance one platform by one step.
    ///
    /// `offsets` and `scores` are parallel: `scores[i]` is the footprint score
    /// at `nominal_position + offsets[i]`, as produced by
    /// [`argusnet_mapping::score_footprints`].
    pub fn update(
        &mut self,
        input: StepInput<'_>,
        offsets: &ParticleOffsets,
        scores: &[f64],
    ) -> Result<LocalizationEstimate, LocalizationError> {
        let step = self
            .step_counts
            .entry(input.platform_id.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        let step = *step;

        if input.coverage_fraction < self.config.min_coverage_to_localize {
            // No-map early return. The estimate is still stored, so the next
            // step's confidence blend sees this 0.0 rather than treating the
            // platform as unseen -- which is the reference's behaviour and
            // materially slows the confidence ramp after coverage arrives.
            let estimate = LocalizationEstimate {
                platform_id: input.platform_id.to_string(),
                timestamp_s: input.timestamp_s,
                position_estimate: input.nominal_position,
                heading_rad: input.heading_rad,
                position_std_m: self.config.search_radius_m,
                confidence: 0.0,
                status: LocalizationStatus::MapUnavailable,
            };
            self.estimates
                .insert(input.platform_id.to_string(), estimate.clone());
            return Ok(estimate);
        }

        if offsets.len() != scores.len() {
            return Err(LocalizationError::ParticleCountMismatch {
                offsets: offsets.len(),
                scores: scores.len(),
            });
        }
        if offsets.is_empty() {
            return Err(LocalizationError::NoParticles);
        }

        // Softmax over scores, shifted by the maximum so `exp` cannot overflow.
        // The `1e-12` in the denominator is the reference's guard against every
        // weight underflowing to zero.
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max_score.is_finite() {
            return Err(LocalizationError::NotFinite("particle score"));
        }
        let unnormalized: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let total: f64 = unnormalized.iter().sum();
        let weights: Vec<f64> = unnormalized.iter().map(|w| w / (total + 1e-12)).collect();

        // Weighted mean of the cloud, in the horizontal plane only.
        let mut mean = [0.0f64; 2];
        for (offset, weight) in offsets.iter().zip(&weights) {
            mean[0] += (input.nominal_position[0] + offset[0]) * weight;
            mean[1] += (input.nominal_position[1] + offset[1]) * weight;
        }

        // Weighted variance per axis, then the spread is the root of their mean.
        // Not the root of the sum: the reference takes `var.mean()`, which makes
        // this an average per-axis sigma rather than a radial one.
        let mut variance = [0.0f64; 2];
        for (offset, weight) in offsets.iter().zip(&weights) {
            let dx = input.nominal_position[0] + offset[0] - mean[0];
            let dy = input.nominal_position[1] + offset[1] - mean[1];
            variance[0] += weight * dx * dx;
            variance[1] += weight * dy * dy;
        }
        let position_std = ((variance[0] + variance[1]) / 2.0).sqrt();

        // Confidence rises with coverage and falls with spread. The `max(_, 1.0)`
        // floors the divisor at one metre, so a very tight cloud cannot drive
        // the ratio above the `min(1.0, _)` cap anyway -- it is there to stop a
        // near-zero spread producing a division blow-up.
        let coverage_term = (input.coverage_fraction / 0.8).min(1.0);
        let spread_term = (self.config.convergence_threshold_m / position_std.max(1.0)).min(1.0);
        let base_confidence = coverage_term * spread_term;

        let confidence = match self.estimates.get(input.platform_id) {
            Some(previous) => {
                self.config.confidence_decay * previous.confidence
                    + (1.0 - self.config.confidence_decay) * base_confidence
            }
            None => base_confidence,
        };

        let converged = position_std <= self.config.convergence_threshold_m;
        let mut status = if converged {
            LocalizationStatus::Converged
        } else {
            LocalizationStatus::Converging
        };
        let mut confidence = confidence;

        if self.config.localization_timeout_steps > 0
            && step >= self.config.localization_timeout_steps
        {
            self.timed_out.insert(input.platform_id.to_string());
            confidence = 1.0;
            // Timeout wins over convergence in the status, because the forced
            // confidence is what a caller is looking at and it must not be
            // mistaken for a real result.
            status = LocalizationStatus::TimedOut;
        }

        let estimate = LocalizationEstimate {
            platform_id: input.platform_id.to_string(),
            timestamp_s: input.timestamp_s,
            position_estimate: [mean[0], mean[1], input.nominal_position[2]],
            heading_rad: input.heading_rad,
            position_std_m: position_std,
            confidence: confidence.clamp(0.0, 1.0),
            status,
        };
        self.estimates
            .insert(input.platform_id.to_string(), estimate.clone());
        Ok(estimate)
    }

    /// True once any platform has hit the step budget.
    pub fn any_timed_out(&self) -> bool {
        !self.timed_out.is_empty()
    }

    pub fn timed_out(&self, platform_id: &str) -> bool {
        self.timed_out.contains(platform_id)
    }

    pub fn estimate(&self, platform_id: &str) -> Option<&LocalizationEstimate> {
        self.estimates.get(platform_id)
    }

    pub fn all_estimates(&self) -> Vec<LocalizationEstimate> {
        self.estimates.values().cloned().collect()
    }

    /// Inverse-variance fusion of the platforms that have converged enough.
    ///
    /// "Enough" is a spread below half the search radius, which is the
    /// reference's admission test and is deliberately looser than
    /// `convergence_threshold_m`. Returns `None` when nothing qualifies —
    /// distinct from a fused estimate with low confidence, and the caller must
    /// keep them distinct.
    ///
    /// A platform id that is unknown, or listed twice, is handled the way the
    /// reference handles it: unknown ids are skipped, and a duplicate id
    /// contributes twice. P4.2 lists duplicate/missing platform semantics among
    /// what to preserve, so both are pinned by tests rather than tidied.
    pub fn fuse_estimates(&self, platform_ids: &[&str]) -> Option<LocalizationEstimate> {
        let half_radius = self.config.search_radius_m * 0.5;
        let candidates: Vec<&LocalizationEstimate> = platform_ids
            .iter()
            .filter_map(|id| self.estimates.get(*id))
            .filter(|estimate| estimate.position_std_m < half_radius)
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let precisions: Vec<f64> = candidates
            .iter()
            .map(|estimate| 1.0 / (estimate.position_std_m * estimate.position_std_m).max(1e-6))
            .collect();
        let precision_sum: f64 = precisions.iter().sum();
        let weights: Vec<f64> = precisions.iter().map(|p| p / precision_sum).collect();

        let mut fused_position = [0.0f64; 3];
        let mut fused_confidence = 0.0f64;
        for (estimate, weight) in candidates.iter().zip(&weights) {
            for (fused_axis, estimate_axis) in
                fused_position.iter_mut().zip(estimate.position_estimate)
            {
                *fused_axis += estimate_axis * weight;
            }
            fused_confidence += estimate.confidence * weight;
        }

        // The fused spread comes from the *unnormalized* precision sum, which
        // is why `precision_sum` is kept rather than folded into `weights`.
        let fused_std = (1.0 / precision_sum).sqrt();
        let reference = candidates[0];

        Some(LocalizationEstimate {
            platform_id: "fused".to_string(),
            timestamp_s: reference.timestamp_s,
            position_estimate: fused_position,
            heading_rad: reference.heading_rad,
            position_std_m: fused_std,
            confidence: fused_confidence.clamp(0.0, 1.0),
            // A fusion of converged inputs is a converged result; a timeout on
            // one input does not make the fusion a timeout, because the fused
            // number is a real inverse-variance combination either way.
            status: LocalizationStatus::Converged,
        })
    }

    /// Forget one platform, or all of them.
    ///
    /// Clears estimate, step count and timeout together — leaving any of the
    /// three behind would make a reset platform behave as though it had a
    /// history it does not.
    pub fn reset(&mut self, platform_id: Option<&str>) {
        match platform_id {
            None => {
                self.estimates.clear();
                self.step_counts.clear();
                self.timed_out.clear();
            }
            Some(id) => {
                self.estimates.remove(id);
                self.step_counts.remove(id);
                self.timed_out.remove(id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(platform_id: &'a str, coverage: f64) -> StepInput<'a> {
        StepInput {
            platform_id,
            nominal_position: [100.0, 200.0, 30.0],
            heading_rad: 0.5,
            timestamp_s: 1.0,
            coverage_fraction: coverage,
        }
    }

    fn uniform(count: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
        let offsets = (0..count)
            .map(|i| [i as f64 - count as f64 / 2.0, 0.0])
            .collect();
        (offsets, vec![1.0; count])
    }

    #[test]
    fn below_the_coverage_floor_the_nominal_position_is_returned_unchanged() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let (offsets, scores) = uniform(4);
        let estimate = localizer
            .update(input("drone-1", 0.05), &offsets, &scores)
            .expect("no-map path");
        assert_eq!(estimate.status, LocalizationStatus::MapUnavailable);
        assert_eq!(estimate.position_estimate, [100.0, 200.0, 30.0]);
        assert_eq!(estimate.confidence, 0.0);
        assert_eq!(estimate.position_std_m, 80.0);
    }

    #[test]
    fn a_no_map_step_still_counts_toward_the_timeout() {
        // The reference increments before the early return. If this changed,
        // every existing run's timeout would fire at a different step.
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let (offsets, scores) = uniform(4);
        for _ in 0..3 {
            localizer
                .update(input("drone-1", 0.0), &offsets, &scores)
                .expect("no-map path");
        }
        assert_eq!(localizer.step_count("drone-1"), 3);
    }

    #[test]
    fn equal_scores_put_the_estimate_at_the_centre_of_the_cloud() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        // Offsets symmetric about zero, all scored equally: the weighted mean
        // must land on the nominal position.
        let offsets = vec![[-10.0, -5.0], [10.0, 5.0], [-4.0, 8.0], [4.0, -8.0]];
        let scores = vec![2.0; 4];
        let estimate = localizer
            .update(input("drone-1", 0.5), &offsets, &scores)
            .expect("step");
        // Not exactly centred, and that is the reference's arithmetic rather
        // than a port error -- see `the_weight_guard_biases_the_mean_toward_the_origin`.
        assert!((estimate.position_estimate[0] - 100.0).abs() < 1e-9);
        assert!((estimate.position_estimate[1] - 200.0).abs() < 1e-9);
        assert_eq!(
            estimate.position_estimate[2], 30.0,
            "elevation is carried from the nominal position, not searched"
        );
    }

    #[test]
    fn a_dominant_score_pulls_the_estimate_onto_that_particle() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[-10.0, 0.0], [10.0, 0.0], [30.0, 40.0]];
        // The softmax is over score differences, so a 50-unit lead is total.
        let scores = vec![0.0, 0.0, 50.0];
        let estimate = localizer
            .update(input("drone-1", 0.5), &offsets, &scores)
            .expect("step");
        assert!((estimate.position_estimate[0] - 130.0).abs() < 1e-6);
        assert!((estimate.position_estimate[1] - 240.0).abs() < 1e-6);
    }

    #[test]
    fn the_softmax_shift_survives_scores_that_would_overflow_exp() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[-1.0, 0.0], [1.0, 0.0]];
        let scores = vec![1000.0, 1000.0];
        let estimate = localizer
            .update(input("drone-1", 0.5), &offsets, &scores)
            .expect("step");
        assert!(
            estimate.position_estimate[0].is_finite(),
            "exp(1000) would be inf without the max-shift"
        );
        assert!((estimate.position_estimate[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn the_weight_guard_biases_the_mean_and_does_so_identically_to_python() {
        // `weights /= weights.sum() + 1e-12` is the reference's guard against
        // every weight underflowing to zero. The cost is that the weights sum
        // to slightly *under* one, so the weighted mean is scaled down --
        // pulling the estimate toward the coordinate origin by a fraction of
        // its own magnitude rather than by a fixed distance.
        //
        // The numbers below are Python's, to full precision:
        //
        //     w = np.exp(s - s.max()); w /= w.sum() + 1e-12
        //     (positions * w[:, None]).sum(axis=0) - [100.0, 200.0]
        //     -> (-2.5011104298755527e-11, -4.999378688808065e-11)
        //
        // and this port reproduces them bit-for-bit, accumulated rounding
        // included. That is worth asserting rather than approximating: it is
        // the strongest evidence available that the formula was carried across
        // rather than re-derived into something that merely looks similar.
        //
        // Caveat, stated because it will matter later: bit-identity here rests
        // on the summation order matching. numpy switches to pairwise summation
        // for large arrays, so agreement at four particles does not promise
        // agreement at the configured 200. The recorded-offset fixtures are
        // what settle that, and they compare with a tolerance for exactly this
        // reason.
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[-10.0, -5.0], [10.0, 5.0], [-4.0, 8.0], [4.0, -8.0]];
        let scores = vec![2.0; 4];
        let estimate = localizer
            .update(input("drone-1", 0.5), &offsets, &scores)
            .expect("step");

        assert_eq!(
            estimate.position_estimate[0] - 100.0,
            -2.5011104298755527e-11
        );
        assert_eq!(
            estimate.position_estimate[1] - 200.0,
            -4.999378688808065e-11
        );
    }

    #[test]
    fn confidence_blends_with_the_previous_step() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[-1.0, 0.0], [1.0, 0.0]];
        let scores = vec![1.0, 1.0];

        let first = localizer
            .update(input("drone-1", 0.8), &offsets, &scores)
            .expect("step");
        let second = localizer
            .update(input("drone-1", 0.8), &offsets, &scores)
            .expect("step");
        // With decay 0.5 and an unchanged base, the blend is a no-op -- so the
        // check that matters is that it *is* a blend, which a differing first
        // step shows.
        assert!((second.confidence - first.confidence).abs() < 1e-12);

        let mut fresh = GridLocalizer::new(LocalizationConfig::default());
        let cold = fresh
            .update(input("drone-2", 0.4), &offsets, &scores)
            .expect("step");
        let warm = fresh
            .update(input("drone-2", 0.8), &offsets, &scores)
            .expect("step");
        assert!(
            warm.confidence > cold.confidence && warm.confidence < 1.0,
            "a rising base must pull confidence up but not all the way in one step"
        );
    }

    #[test]
    fn the_timeout_forces_confidence_and_says_so() {
        let config = LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        };
        let mut localizer = GridLocalizer::new(config);
        let offsets = vec![[-40.0, 0.0], [40.0, 0.0]];
        let scores = vec![1.0, 1.0];

        for _ in 0..2 {
            let estimate = localizer
                .update(input("drone-1", 0.3), &offsets, &scores)
                .expect("step");
            assert_ne!(estimate.status, LocalizationStatus::TimedOut);
            assert!(estimate.confidence < 1.0);
        }
        let timed_out = localizer
            .update(input("drone-1", 0.3), &offsets, &scores)
            .expect("step");
        assert_eq!(timed_out.status, LocalizationStatus::TimedOut);
        assert_eq!(timed_out.confidence, 1.0);
        assert!(localizer.any_timed_out());
    }

    #[test]
    fn a_zero_timeout_disables_it() {
        let config = LocalizationConfig {
            localization_timeout_steps: 0,
            ..LocalizationConfig::default()
        };
        let mut localizer = GridLocalizer::new(config);
        let offsets = vec![[-40.0, 0.0], [40.0, 0.0]];
        let scores = vec![1.0, 1.0];
        for _ in 0..50 {
            localizer
                .update(input("drone-1", 0.3), &offsets, &scores)
                .expect("step");
        }
        assert!(!localizer.any_timed_out());
    }

    #[test]
    fn timeout_and_convergence_are_distinguishable() {
        // The gap LocalizationStatus exists to close: both report confidence
        // 1.0-ish, and they mean opposite things about the position.
        let config = LocalizationConfig {
            localization_timeout_steps: 1,
            ..LocalizationConfig::default()
        };
        let mut localizer = GridLocalizer::new(config);
        let tight = vec![[-0.01, 0.0], [0.01, 0.0]];
        let forced = localizer
            .update(input("drone-1", 0.9), &tight, &[1.0, 1.0])
            .expect("step");
        assert_eq!(forced.status, LocalizationStatus::TimedOut);

        let mut honest = GridLocalizer::new(LocalizationConfig::default());
        let converged = honest
            .update(input("drone-2", 0.9), &tight, &[1.0, 1.0])
            .expect("step");
        assert_eq!(converged.status, LocalizationStatus::Converged);
        // Same confidence, different meaning -- which is the whole point.
        assert!(converged.confidence > 0.0);
    }

    #[test]
    fn annealing_and_coverage_both_shrink_the_search() {
        let localizer = GridLocalizer::new(LocalizationConfig::default());
        let early = localizer.particle_scale(1, 0.2);
        let later = localizer.particle_scale(50, 0.2);
        assert!(later < early, "annealing must shrink the cloud over steps");

        let sparse = localizer.particle_scale(1, 0.2);
        let dense = localizer.particle_scale(1, 0.9);
        assert!(dense < sparse, "better coverage needs a smaller search");
    }

    #[test]
    fn the_scale_floors_stop_the_cloud_collapsing() {
        let localizer = GridLocalizer::new(LocalizationConfig::default());
        // Coverage of 1.0 would zero the coverage term without its 0.1 floor.
        assert!(localizer.particle_scale(1, 1.0) > 0.0);
        // And 10,000 steps of annealing would underflow without the 0.05 floor.
        let annealed = localizer.particle_scale(10_000, 0.0);
        assert!(annealed > 0.0, "annealing floored at 0.05 of the radius");
        assert!((annealed - 80.0 * 0.05).abs() < 1e-9);
    }

    #[test]
    fn fusion_weights_the_tighter_estimate_more() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        localizer.estimates.insert(
            "a".to_string(),
            LocalizationEstimate {
                platform_id: "a".to_string(),
                timestamp_s: 5.0,
                position_estimate: [0.0, 0.0, 10.0],
                heading_rad: 0.0,
                position_std_m: 1.0,
                confidence: 0.9,
                status: LocalizationStatus::Converged,
            },
        );
        localizer.estimates.insert(
            "b".to_string(),
            LocalizationEstimate {
                platform_id: "b".to_string(),
                timestamp_s: 5.0,
                position_estimate: [100.0, 0.0, 10.0],
                heading_rad: 0.0,
                position_std_m: 10.0,
                confidence: 0.5,
                status: LocalizationStatus::Converged,
            },
        );

        let fused = localizer.fuse_estimates(&["a", "b"]).expect("both qualify");
        // Precisions 1 and 0.01, so a carries 100/101 of the weight.
        assert!(
            fused.position_estimate[0] < 1.0,
            "pulled toward the tight one"
        );
        assert!(
            fused.position_std_m < 1.0,
            "fusing two estimates must be tighter than either"
        );
    }

    #[test]
    fn fusion_ignores_estimates_that_have_not_converged_enough() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        localizer.estimates.insert(
            "loose".to_string(),
            LocalizationEstimate {
                platform_id: "loose".to_string(),
                timestamp_s: 1.0,
                position_estimate: [0.0, 0.0, 0.0],
                heading_rad: 0.0,
                // Half the 80 m search radius is the admission test, so 40 is out.
                position_std_m: 40.0,
                confidence: 1.0,
                status: LocalizationStatus::Converging,
            },
        );
        assert!(localizer.fuse_estimates(&["loose"]).is_none());
    }

    #[test]
    fn fusion_skips_unknown_ids_and_counts_a_duplicate_twice() {
        // Both are the reference's behaviour, pinned rather than tidied: P4.2
        // lists duplicate/missing platform semantics among what to preserve.
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        localizer.estimates.insert(
            "a".to_string(),
            LocalizationEstimate {
                platform_id: "a".to_string(),
                timestamp_s: 1.0,
                position_estimate: [10.0, 0.0, 0.0],
                heading_rad: 0.0,
                position_std_m: 2.0,
                confidence: 0.8,
                status: LocalizationStatus::Converged,
            },
        );
        let once = localizer
            .fuse_estimates(&["a", "missing"])
            .expect("a alone");
        let twice = localizer.fuse_estimates(&["a", "a"]).expect("a twice");

        assert_eq!(once.position_estimate, [10.0, 0.0, 0.0]);
        // Counting it twice halves the fused variance, which is exactly what
        // treating one measurement as two independent ones does.
        assert!(twice.position_std_m < once.position_std_m);
    }

    #[test]
    fn reset_clears_estimate_step_count_and_timeout_together() {
        let config = LocalizationConfig {
            localization_timeout_steps: 1,
            ..LocalizationConfig::default()
        };
        let mut localizer = GridLocalizer::new(config);
        let offsets = vec![[-1.0, 0.0], [1.0, 0.0]];
        localizer
            .update(input("drone-1", 0.5), &offsets, &[1.0, 1.0])
            .expect("step");
        assert!(localizer.any_timed_out());

        localizer.reset(Some("drone-1"));
        assert_eq!(localizer.step_count("drone-1"), 0);
        assert!(localizer.estimate("drone-1").is_none());
        assert!(
            !localizer.any_timed_out(),
            "a reset platform must not stay flagged as timed out"
        );
    }

    #[test]
    fn resetting_one_platform_leaves_the_others_alone() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[-1.0, 0.0], [1.0, 0.0]];
        localizer
            .update(input("drone-1", 0.5), &offsets, &[1.0, 1.0])
            .expect("step");
        localizer
            .update(input("drone-2", 0.5), &offsets, &[1.0, 1.0])
            .expect("step");
        localizer.reset(Some("drone-1"));
        assert!(localizer.estimate("drone-1").is_none());
        assert!(localizer.estimate("drone-2").is_some());
    }

    #[test]
    fn mismatched_particle_inputs_are_rejected() {
        let mut localizer = GridLocalizer::new(LocalizationConfig::default());
        let offsets = vec![[0.0, 0.0], [1.0, 1.0]];
        assert_eq!(
            localizer.update(input("drone-1", 0.5), &offsets, &[1.0]),
            Err(LocalizationError::ParticleCountMismatch {
                offsets: 2,
                scores: 1
            })
        );
        assert_eq!(
            localizer.update(input("drone-1", 0.5), &[], &[]),
            Err(LocalizationError::NoParticles)
        );
    }
}
