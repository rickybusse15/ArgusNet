//! Quality requirements and the deficit each map cell has against them
//! (SR5a; SENSING_REALISM_IMPLEMENTATION_GUIDE §8.1–8.2).
//!
//! A count of hits is not a map. A cell is done when the evidence accepted
//! for it meets what the mission declared it needs: enough ground returns
//! per square metre from rays steep enough to trust, a vertical uncertainty
//! small enough, recent enough. Each cell that falls short says **why**,
//! because each reason asks the planner for something different:
//!
//! | Reason | What would fix it |
//! | --- | --- |
//! | [`DeficitReason::Unseen`] | Going there: frontier exploration |
//! | [`DeficitReason::TooOblique`] | New geometry: nearer nadir |
//! | [`DeficitReason::Sparse`] | More of the same geometry |
//! | [`DeficitReason::Stale`] | A revisit |
//! | [`DeficitReason::BiasLimited`] | Nothing a revisit can do |
//!
//! **Bias-limited** is the case that matters most for planning. A declared
//! systematic error (boresight, calibration) is shared by every return, so
//! averaging more returns cannot shrink it below its floor. If the floor
//! alone exceeds the requirement, no number of passes meets it, and a
//! planner that kept revisiting would never finish. Such a cell is
//! **unreachable**, reported with its reason, never complete.
//!
//! Occlusion and rejected evidence are named by the guide too. They need
//! evidence this map does not yet keep (per-ray occlusion, an admission
//! step), so no cell reports them yet; an unmet cell that is neither of the
//! above is reported by the reason that actually applies.
//!
//! Unknown is not met. A requirement field left out is not checked; a
//! requirement with every field left out is refused, since it would mark
//! every seen cell complete.

use serde::{Deserialize, Serialize};

/// What the mission declares each cell of the map needs. Frozen before the
/// run; recorded in the replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityRequirement {
    /// Accepted ground returns per square metre.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_ground_density_per_m2: Option<f64>,
    /// One-sigma vertical uncertainty of the cell's height, metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_vertical_sigma_m: Option<f64>,
    /// How old the newest accepted return may be, seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_age_s: Option<f64>,
    /// The steepest-from-nadir ray whose ground return is accepted, degrees.
    /// Returns from shallower rays are seen but not accepted.
    pub max_incidence_deg: f64,
    /// The declared systematic vertical error every return shares, metres:
    /// what averaging cannot remove.
    #[serde(default)]
    pub bias_floor_m: f64,
}

impl QualityRequirement {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: Option<f64>| match value {
            Some(v) if !(v.is_finite() && v > 0.0) => Err(format!(
                "quality requirement {name} must be positive, got {v}"
            )),
            _ => Ok(()),
        };
        positive("min_ground_density_per_m2", self.min_ground_density_per_m2)?;
        positive("max_vertical_sigma_m", self.max_vertical_sigma_m)?;
        positive("max_age_s", self.max_age_s)?;
        if !(self.max_incidence_deg.is_finite()
            && self.max_incidence_deg > 0.0
            && self.max_incidence_deg <= 90.0)
        {
            return Err(format!(
                "quality requirement max_incidence_deg must be in (0, 90], got {}",
                self.max_incidence_deg
            ));
        }
        if !(self.bias_floor_m.is_finite() && self.bias_floor_m >= 0.0) {
            return Err(format!(
                "quality requirement bias_floor_m must not be negative, got {}",
                self.bias_floor_m
            ));
        }
        if self.min_ground_density_per_m2.is_none()
            && self.max_vertical_sigma_m.is_none()
            && self.max_age_s.is_none()
        {
            return Err(
                "a quality requirement must name a density, a vertical sigma or an age".into(),
            );
        }
        Ok(())
    }

    /// Whether the declared bias alone exceeds the vertical requirement:
    /// then no cell can ever meet it. A bias equal to the requirement is
    /// not: a noiseless cell's sigma reaches it exactly, and the sigma
    /// check accepts equality.
    pub fn bias_exceeds_sigma(&self) -> bool {
        self.max_vertical_sigma_m
            .is_some_and(|sigma| self.bias_floor_m > sigma)
    }
}

/// Why a cell does not meet its requirement, or that it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeficitReason {
    Met,
    /// No ground return at all.
    Unseen,
    /// Ground returns, none from a ray steep enough to accept.
    TooOblique,
    /// The declared bias alone exceeds the vertical requirement.
    BiasLimited,
    /// Too few accepted returns for the density or the vertical sigma.
    Sparse,
    /// Met once, but the newest accepted return is too old.
    Stale,
}

impl DeficitReason {
    pub const ALL: [DeficitReason; 6] = [
        DeficitReason::Met,
        DeficitReason::Unseen,
        DeficitReason::TooOblique,
        DeficitReason::BiasLimited,
        DeficitReason::Sparse,
        DeficitReason::Stale,
    ];

    /// Whether flying more could fix it: everything but met and bias-limited.
    pub fn is_improvable(self) -> bool {
        !matches!(self, DeficitReason::Met | DeficitReason::BiasLimited)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DeficitReason::Met => "met",
            DeficitReason::Unseen => "unseen",
            DeficitReason::TooOblique => "too_oblique",
            DeficitReason::BiasLimited => "bias_limited",
            DeficitReason::Sparse => "sparse",
            DeficitReason::Stale => "stale",
        }
    }
}

/// One cell's accepted evidence, as the map keeps it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CellEvidence {
    /// Every ground return, accepted or not.
    pub ground_returns: u32,
    /// Ground returns from rays within the incidence limit.
    pub accepted_returns: u32,
    /// Sum and sum of squares of the accepted returns' heights.
    pub accepted_height_sum_m: f64,
    pub accepted_height_sq_sum_m2: f64,
    /// Seconds since the newest accepted return; `None` for none.
    pub accepted_age_s: Option<f64>,
}

/// One cell's quality against a requirement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellQuality {
    pub reason: DeficitReason,
    /// Accepted returns per square metre.
    pub density_per_m2: f64,
    /// Estimated one-sigma vertical uncertainty; `None` with no accepted
    /// return.
    pub vertical_sigma_m: Option<f64>,
}

/// One-sigma vertical uncertainty of a cell's mean height: the spread of its
/// accepted returns (never taken below `per_return_sigma_m`, the sensor's
/// declared range noise) over the root of their count, combined with the
/// shared bias, which no count reduces.
pub fn vertical_sigma_m(
    evidence: &CellEvidence,
    per_return_sigma_m: f64,
    bias_floor_m: f64,
) -> Option<f64> {
    let n = evidence.accepted_returns;
    if n == 0 {
        return None;
    }
    let count = n as f64;
    let spread_sq = if n >= 2 {
        let mean = evidence.accepted_height_sum_m / count;
        ((evidence.accepted_height_sq_sum_m2 - count * mean * mean) / (count - 1.0)).max(0.0)
    } else {
        0.0
    };
    let per_return_sq = spread_sq.max(per_return_sigma_m * per_return_sigma_m);
    Some((per_return_sq / count + bias_floor_m * bias_floor_m).sqrt())
}

/// A cell's quality against `requirement`. The first reason that applies,
/// in the order the guide's table lists them: unseen, too oblique,
/// bias-limited, sparse, stale.
pub fn assess(
    evidence: &CellEvidence,
    cell_area_m2: f64,
    per_return_sigma_m: f64,
    requirement: &QualityRequirement,
) -> CellQuality {
    let density_per_m2 = evidence.accepted_returns as f64 / cell_area_m2.max(f64::MIN_POSITIVE);
    let vertical_sigma_m = vertical_sigma_m(evidence, per_return_sigma_m, requirement.bias_floor_m);
    let reason = if evidence.ground_returns == 0 {
        DeficitReason::Unseen
    } else if evidence.accepted_returns == 0 {
        DeficitReason::TooOblique
    } else if requirement.bias_exceeds_sigma() {
        DeficitReason::BiasLimited
    } else if requirement
        .min_ground_density_per_m2
        .is_some_and(|min| density_per_m2 < min)
        || requirement
            .max_vertical_sigma_m
            .zip(vertical_sigma_m)
            .is_some_and(|(max, sigma)| sigma > max)
    {
        DeficitReason::Sparse
    } else if requirement
        .max_age_s
        .zip(evidence.accepted_age_s)
        .is_some_and(|(max, age)| age > max)
    {
        DeficitReason::Stale
    } else {
        DeficitReason::Met
    };
    CellQuality {
        reason,
        density_per_m2,
        vertical_sigma_m,
    }
}

/// How many cells give each reason.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeficitCounts {
    pub met: u64,
    pub unseen: u64,
    pub too_oblique: u64,
    pub bias_limited: u64,
    pub sparse: u64,
    pub stale: u64,
}

impl DeficitCounts {
    pub fn add(&mut self, reason: DeficitReason) {
        *match reason {
            DeficitReason::Met => &mut self.met,
            DeficitReason::Unseen => &mut self.unseen,
            DeficitReason::TooOblique => &mut self.too_oblique,
            DeficitReason::BiasLimited => &mut self.bias_limited,
            DeficitReason::Sparse => &mut self.sparse,
            DeficitReason::Stale => &mut self.stale,
        } += 1;
    }

    pub fn total(&self) -> u64 {
        self.met + self.unseen + self.too_oblique + self.bias_limited + self.sparse + self.stale
    }

    /// The share of cells that meet the requirement.
    pub fn met_fraction(&self) -> f64 {
        self.met as f64 / self.total().max(1) as f64
    }

    /// Cells no amount of flying can bring to the requirement.
    pub fn unreachable(&self) -> u64 {
        self.bias_limited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement() -> QualityRequirement {
        QualityRequirement {
            min_ground_density_per_m2: Some(0.05),
            max_vertical_sigma_m: Some(0.5),
            max_age_s: Some(60.0),
            max_incidence_deg: 45.0,
            bias_floor_m: 0.1,
        }
    }

    /// `n` accepted returns all at `height`, the newest `age_s` ago.
    fn evidence(n: u32, height: f64, age_s: f64) -> CellEvidence {
        CellEvidence {
            ground_returns: n,
            accepted_returns: n,
            accepted_height_sum_m: n as f64 * height,
            accepted_height_sq_sum_m2: n as f64 * height * height,
            accepted_age_s: (n > 0).then_some(age_s),
        }
    }

    #[test]
    fn each_reason_is_the_first_that_applies() {
        let r = requirement();
        let area = 100.0;
        let reason = |e: &CellEvidence| assess(e, area, 0.3, &r).reason;
        assert_eq!(reason(&CellEvidence::default()), DeficitReason::Unseen);
        let oblique = CellEvidence {
            ground_returns: 12,
            ..CellEvidence::default()
        };
        assert_eq!(reason(&oblique), DeficitReason::TooOblique);
        assert_eq!(reason(&evidence(2, 10.0, 0.0)), DeficitReason::Sparse);
        assert_eq!(reason(&evidence(8, 10.0, 0.0)), DeficitReason::Met);
        assert_eq!(reason(&evidence(8, 10.0, 61.0)), DeficitReason::Stale);
        let biased = QualityRequirement {
            bias_floor_m: 0.6,
            ..r.clone()
        };
        assert_eq!(
            assess(&evidence(1000, 10.0, 0.0), area, 0.3, &biased).reason,
            DeficitReason::BiasLimited,
            "no count beats a bias above the requirement"
        );
        // A bias equal to the requirement is reachable: with no noise and
        // identical heights the sigma is exactly the bias.
        let equal = QualityRequirement {
            bias_floor_m: 0.5,
            ..r.clone()
        };
        let quality = assess(&evidence(8, 10.0, 0.0), area, 0.0, &equal);
        assert_eq!(quality.vertical_sigma_m, Some(0.5));
        assert_eq!(quality.reason, DeficitReason::Met);
    }

    #[test]
    fn sigma_falls_with_count_but_never_below_the_bias() {
        let sigma = |n| vertical_sigma_m(&evidence(n, 5.0, 0.0), 0.3, 0.1).unwrap();
        assert!((sigma(1) - (0.09f64 + 0.01).sqrt()).abs() < 1e-12);
        assert!(sigma(4) < sigma(1));
        assert!(sigma(10_000) > 0.1 && sigma(10_000) < 0.1001);
        assert_eq!(vertical_sigma_m(&CellEvidence::default(), 0.3, 0.1), None);
        // The returns' own spread counts once it exceeds the sensor's noise.
        let spread = CellEvidence {
            ground_returns: 2,
            accepted_returns: 2,
            accepted_height_sum_m: 0.0 + 4.0,
            accepted_height_sq_sum_m2: 0.0 + 16.0,
            accepted_age_s: Some(0.0),
        };
        // Sample variance 8, over 2, plus bias²: sqrt(4.01).
        let s = vertical_sigma_m(&spread, 0.3, 0.1).unwrap();
        assert!((s - 4.01f64.sqrt()).abs() < 1e-12, "{s}");
    }

    #[test]
    fn met_and_unreachable_are_counted_apart() {
        let mut counts = DeficitCounts::default();
        for reason in DeficitReason::ALL {
            counts.add(reason);
        }
        counts.add(DeficitReason::Met);
        assert_eq!(counts.total(), 7);
        assert!((counts.met_fraction() - 2.0 / 7.0).abs() < 1e-12);
        assert_eq!(counts.unreachable(), 1);
        assert!(!DeficitReason::BiasLimited.is_improvable());
        assert!(DeficitReason::Stale.is_improvable());
        assert_eq!(DeficitCounts::default().met_fraction(), 0.0);
    }

    #[test]
    fn empty_or_invalid_requirements_are_refused() {
        assert!(requirement().validate().is_ok());
        let nothing = QualityRequirement {
            min_ground_density_per_m2: None,
            max_vertical_sigma_m: None,
            max_age_s: None,
            ..requirement()
        };
        assert!(
            nothing.validate().is_err(),
            "would mark every seen cell met"
        );
        let bad = [
            QualityRequirement {
                max_incidence_deg: 0.0,
                ..requirement()
            },
            QualityRequirement {
                min_ground_density_per_m2: Some(-1.0),
                ..requirement()
            },
            QualityRequirement {
                bias_floor_m: f64::NAN,
                ..requirement()
            },
        ];
        for requirement in bad {
            assert!(requirement.validate().is_err(), "{requirement:?}");
        }
    }
}
