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
//! | [`DeficitReason::ImageCoarse`] | An image from nearer (SR4) |
//! | [`DeficitReason::ImageLimited`] | Nothing: no flight gets that near |
//! | [`DeficitReason::PoseUncertain`] | A revisit that knows better where it is |
//! | [`DeficitReason::PoseLimited`] | Nothing, while no aircraft knows better |
//!
//! **Bias-limited** is the case that matters most for planning. A declared
//! systematic error (boresight, calibration) is shared by every return, so
//! averaging more returns cannot shrink it below its floor. If the floor
//! alone exceeds the requirement, no number of passes meets it, and a
//! planner that kept revisiting would never finish. Such a cell is
//! **unreachable**, reported with its reason, never complete.
//!
//! **The pose behind a return** is shared the same way, by every return of a
//! pass, so a count does not shrink it either. Passes share it in part, by
//! how long a navigation error lasts, which the map does not know. Its part
//! of a cell's sigma is the weighted mean of its returns' own pose sigmas:
//! the most the weighted mean's can be, however their errors are related.
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
    /// The coarsest ground sample distance of a cell's best image, metres
    /// (SR4). Needs the imaging model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_image_gsd_m: Option<f64>,
    /// One-sigma horizontal uncertainty of where the cell's returns are,
    /// metres. Needs the pose behind them to be estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_horizontal_sigma_m: Option<f64>,
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
        positive("max_image_gsd_m", self.max_image_gsd_m)?;
        positive("max_horizontal_sigma_m", self.max_horizontal_sigma_m)?;
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
            && self.max_image_gsd_m.is_none()
            && self.max_horizontal_sigma_m.is_none()
        {
            return Err(
                "a quality requirement must name a density, a vertical or horizontal sigma, \
                 an age or an image sample distance"
                    .into(),
            );
        }
        Ok(())
    }

    /// Whether a pose known to `[vertical, horizontal]` one sigma leaves a
    /// cell of this slope short on its own, with nothing but the declared
    /// bias beside it.
    fn pose_falls_short(&self, [vertical_m, horizontal_m]: [f64; 2], slope: f64) -> bool {
        self.max_vertical_sigma_m.is_some_and(|max| {
            vertical_m
                .hypot(slope * horizontal_m)
                .hypot(self.bias_floor_m)
                > max
        }) || self
            .max_horizontal_sigma_m
            .is_some_and(|max| horizontal_m > max)
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
    /// Nothing else is missing, but no accepted image is fine enough (SR4).
    ImageCoarse,
    /// The image requirement is finer than any flight gets (SR4).
    ImageLimited,
    /// The pose its returns were placed by was too uncertain.
    PoseUncertain,
    /// And no aircraft's is good enough now.
    PoseLimited,
}

impl DeficitReason {
    pub const ALL: [DeficitReason; 10] = [
        DeficitReason::Met,
        DeficitReason::Unseen,
        DeficitReason::TooOblique,
        DeficitReason::BiasLimited,
        DeficitReason::Sparse,
        DeficitReason::Stale,
        DeficitReason::ImageCoarse,
        DeficitReason::ImageLimited,
        DeficitReason::PoseUncertain,
        DeficitReason::PoseLimited,
    ];

    /// Whether flying more could fix it: everything but met and the
    /// limits.
    pub fn is_improvable(self) -> bool {
        !matches!(
            self,
            DeficitReason::Met
                | DeficitReason::BiasLimited
                | DeficitReason::ImageLimited
                | DeficitReason::PoseLimited
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DeficitReason::Met => "met",
            DeficitReason::Unseen => "unseen",
            DeficitReason::TooOblique => "too_oblique",
            DeficitReason::BiasLimited => "bias_limited",
            DeficitReason::Sparse => "sparse",
            DeficitReason::Stale => "stale",
            DeficitReason::ImageCoarse => "image_coarse",
            DeficitReason::ImageLimited => "image_limited",
            DeficitReason::PoseUncertain => "pose_uncertain",
            DeficitReason::PoseLimited => "pose_limited",
        }
    }
}

/// Where a cell's accepted returns fell, as sums over them: east and north of
/// the cell's centre, metres, and their products with each other and with the
/// height. With [`CellEvidence`]'s count and height sums, enough to fit a
/// plane to the cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f64; 7]", into = "[f64; 7]")]
pub struct PlaneMoments {
    pub u: f64,
    pub v: f64,
    pub uu: f64,
    pub uv: f64,
    pub vv: f64,
    pub uz: f64,
    pub vz: f64,
}

impl From<[f64; 7]> for PlaneMoments {
    fn from([u, v, uu, uv, vv, uz, vz]: [f64; 7]) -> Self {
        Self {
            u,
            v,
            uu,
            uv,
            vv,
            uz,
            vz,
        }
    }
}

impl From<PlaneMoments> for [f64; 7] {
    fn from(m: PlaneMoments) -> Self {
        [m.u, m.v, m.uu, m.uv, m.vv, m.uz, m.vz]
    }
}

impl PlaneMoments {
    /// One more return, `u`, `v` from the cell's centre, at height `z`.
    pub fn add(&mut self, u: f64, v: f64, z: f64) {
        self.add_weighted(1.0, u, v, z);
    }

    /// One that counts for `w` ([`pose_weight`]): with an estimated pose, the
    /// moments are weighted as [`PoseEvidence`]'s sums are.
    pub fn add_weighted(&mut self, w: f64, u: f64, v: f64, z: f64) {
        let (wu, wv) = (w * u, w * v);
        self.u += wu;
        self.v += wv;
        self.uu += wu * u;
        self.uv += wu * v;
        self.vv += wv * v;
        self.uz += wu * z;
        self.vz += wv * z;
    }
}

/// A plane fitted to a cell's accepted returns, at the cell's centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneFit {
    /// The height at the centre.
    pub height_m: f64,
    /// The returns' variance about the plane.
    pub residual_m2: f64,
    /// The centre height's variance over the returns': `1/n` with the
    /// returns spread evenly about the centre, more the further from it
    /// they fell.
    pub leverage: f64,
    /// The variance the poses leave the centre height, over the returns'
    /// weighted sum of squared pose sigmas: the most it can be, however the
    /// poses' errors are related. Zero without estimated poses.
    pub pose_gain: f64,
}

/// The plane through a cell's accepted returns, at the centre. `None` without
/// moments, with fewer returns than leave a residual, or with the returns so
/// placed that a plane says less about the centre than one return there would
/// (a single scan line off to one side).
pub fn plane_fit(evidence: &CellEvidence) -> Option<PlaneFit> {
    let m = evidence.plane.as_ref()?;
    if evidence.accepted_returns < 4 {
        return None;
    }
    // With estimated poses every sum is weighted: `n` is the weight, and
    // `count` the returns of equal weight that would say as much.
    let posed = evidence.pose.filter(|pose| pose.weight > 0.0);
    let (n, z_sum, zz_sum, count) = match posed {
        Some(pose) => (
            pose.weight,
            pose.height_m,
            pose.height_sq_m2,
            pose.weight * pose.weight / pose.weight_sq,
        ),
        None => {
            let n = evidence.accepted_returns as f64;
            (
                n,
                evidence.accepted_height_sum_m,
                evidence.accepted_height_sq_sum_m2,
                n,
            )
        }
    };
    if count < 3.5 {
        return None;
    }
    // What one return of that many counts for.
    let each = if posed.is_some() { n / count } else { 1.0 };
    let (u, v) = (m.u / n, m.v / n);
    let z = z_sum / n;
    let suu = m.uu - n * u * u;
    let suv = m.uv - n * u * v;
    let svv = m.vv - n * v * v;
    let det = suu * svv - suv * suv;
    // Returns in a line fix no slope across it.
    if det <= 1e-9 * (suu + svv) * (suu + svv) {
        return None;
    }
    let gain = 1.0 / n + (svv * u * u - 2.0 * suv * u * v + suu * v * v) / det;
    let leverage = gain * each;
    if !(leverage.is_finite() && leverage <= 1.0) {
        return None;
    }
    let suz = m.uz - n * u * z;
    let svz = m.vz - n * v * z;
    let east = (svv * suz - suv * svz) / det;
    let north = (suu * svz - suv * suz) / det;
    let szz = zz_sum - n * z * z;
    Some(PlaneFit {
        height_m: z - east * u - north * v,
        residual_m2: ((szz - east * suz - north * svz) / each / (count - 3.0)).max(0.0),
        leverage,
        pose_gain: if posed.is_some() { gain } else { 0.0 },
    })
}

/// A cell's height at its centre: its accepted returns' plane there, or their
/// mean where no plane is fitted. `None` with no accepted return.
pub fn centre_height_m(evidence: &CellEvidence) -> Option<f64> {
    let mean = evidence.mean_height_m()?;
    Some(plane_fit(evidence).map_or(mean, |fit| fit.height_m))
}

/// What the poses behind a cell's accepted returns leave uncertain. Sums over
/// those returns, each weighted by [`pose_weight`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PoseEvidence {
    pub weight: f64,
    pub height_m: f64,
    pub vertical_sigma_m: f64,
    pub horizontal_sigma_m: f64,
    /// For a plane through them ([`plane_fit`]): the weighted sums of their
    /// squared heights and sigmas, and the sum of their squared weights.
    pub height_sq_m2: f64,
    pub vertical_sq_m2: f64,
    pub horizontal_sq_m2: f64,
    pub weight_sq: f64,
    /// How steep the ground is here, rise over run: what a return in the
    /// wrong place does to the height.
    pub slope: f64,
    /// The best `[vertical, horizontal]` one sigma any aircraft is placing
    /// returns by now; `None` when none is.
    pub best_now_m: Option<[f64; 2]>,
}

impl PoseEvidence {
    /// The cell's `[vertical, horizontal]` one sigma from its poses.
    fn sigmas_m(&self) -> Option<[f64; 2]> {
        (self.weight > 0.0).then(|| {
            [
                self.vertical_sigma_m / self.weight,
                self.horizontal_sigma_m / self.weight,
            ]
        })
    }
}

/// How much a return counts for, placed by a pose whose vertical one sigma
/// is this: a well-placed return outweighs a badly placed one, and below
/// 5 cm they are all alike.
pub fn pose_weight(vertical_sigma_m: f64) -> f64 {
    1.0 / (vertical_sigma_m * vertical_sigma_m + 0.05 * 0.05)
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
    /// The finest accepted image's ground sample distance here, metres;
    /// `None` for none (SR4).
    pub image_gsd_m: Option<f64>,
    /// Where the accepted returns fell; `None` when the map does not keep it.
    pub plane: Option<PlaneMoments>,
    /// With `plane` and none fitted: the variance of the ground over the cell
    /// that its neighbours' heights show, m². Zero for none.
    pub relief_m2: f64,
    /// The poses its accepted returns were placed by; `None` when they were
    /// placed by the true one.
    pub pose: Option<PoseEvidence>,
}

impl CellEvidence {
    /// The accepted returns' mean height; `None` with none.
    pub fn mean_height_m(&self) -> Option<f64> {
        if self.accepted_returns == 0 {
            return None;
        }
        Some(match self.pose.filter(|pose| pose.weight > 0.0) {
            Some(pose) => pose.height_m / pose.weight,
            None => self.accepted_height_sum_m / f64::from(self.accepted_returns),
        })
    }
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
    /// Estimated one-sigma horizontal uncertainty of where its accepted
    /// returns are; `None` without an estimated pose behind them.
    pub horizontal_sigma_m: Option<f64>,
}

/// A plane's squared error at a cell's centre, over the variance the ground's
/// curvature leaves about the plane, for evenly spread returns and a claim of
/// one sigma over curved ground of every shape.
///
/// Along one axis, `a u²` over a cell of side `L` has mean `a L²/12` (the
/// plane's error at the centre) and variance `a² L⁴/180`: a ridge or a valley
/// gives 180/144. The errors of two axes add and their variances add, so a
/// bowl or a dome gives twice that and a saddle or a twist nothing: the
/// residual does not say which it is. Over isotropic ground the mean is 0.90,
/// so a ridge's is no understatement of it, and a bowl's own error is 1.4
/// times the claim.
pub const CURVATURE_VARIANCE: f64 = 1.25;

/// One-sigma vertical uncertainty of a cell's height: the spread of its
/// accepted returns (never taken below `per_return_sigma_m`, the sensor's
/// declared range noise) over the root of their count, combined with the
/// shared bias and the poses' own uncertainty, which no count reduces.
///
/// With [`CellEvidence::plane`] it is the uncertainty of the height at the
/// centre ([`centre_height_m`]). Where a plane is fitted, the spread is about
/// the plane, the count is what the fit leaves at the centre, and what the
/// returns scatter beyond their own noise is ground that is no plane
/// ([`CURVATURE_VARIANCE`]), which no count reduces either. Where none is,
/// the returns lie together somewhere in the cell, so their whole spread
/// stands, or the relief its neighbours show if that is more: one return has
/// no spread, and is no less a place on a slope.
pub fn vertical_sigma_m(
    evidence: &CellEvidence,
    per_return_sigma_m: f64,
    bias_floor_m: f64,
) -> Option<f64> {
    let n = evidence.accepted_returns;
    if n == 0 {
        return None;
    }
    let floor_sq = per_return_sigma_m * per_return_sigma_m;
    if let Some(fit) = plane_fit(evidence) {
        let per_return_sq = fit.residual_m2.max(floor_sq);
        let curved_sq = CURVATURE_VARIANCE * (fit.residual_m2 - floor_sq).max(0.0);
        // The centre height is a weighted sum of the returns' heights, and
        // Cauchy-Schwarz bounds what their pose errors do to it: a plane
        // leaning on badly placed returns to one side makes more of them.
        let pose_sq = evidence.pose.as_ref().map_or(0.0, |pose| {
            fit.pose_gain * (pose.vertical_sq_m2 + pose.slope * pose.slope * pose.horizontal_sq_m2)
        });
        return Some(
            (per_return_sq * fit.leverage + curved_sq + bias_floor_m * bias_floor_m + pose_sq)
                .sqrt(),
        );
    }
    let count = n as f64;
    let spread_sq = if n >= 2 {
        let mean = evidence.accepted_height_sum_m / count;
        ((evidence.accepted_height_sq_sum_m2 - count * mean * mean) / (count - 1.0)).max(0.0)
    } else {
        0.0
    };
    let (per_return_sq, count) = if evidence.plane.is_some() {
        (spread_sq.max(floor_sq).max(evidence.relief_m2), 1.0)
    } else {
        (spread_sq.max(floor_sq), count)
    };
    let pose_sq = evidence.pose.as_ref().map_or(0.0, |pose| {
        pose.sigmas_m().map_or(0.0, |[vertical, horizontal]| {
            vertical * vertical + (pose.slope * horizontal).powi(2)
        })
    });
    Some((per_return_sq / count + bias_floor_m * bias_floor_m + pose_sq).sqrt())
}

/// A cell's quality against `requirement`. The first reason that applies,
/// in the order the guide's table lists them: unseen, too oblique,
/// bias-limited, image-limited, the pose behind them, sparse, stale,
/// image-coarse.
/// `finest_image_gsd_m` is the finest image any flight gets.
pub fn assess(
    evidence: &CellEvidence,
    cell_area_m2: f64,
    per_return_sigma_m: f64,
    finest_image_gsd_m: Option<f64>,
    requirement: &QualityRequirement,
) -> CellQuality {
    let density_per_m2 = evidence.accepted_returns as f64 / cell_area_m2.max(f64::MIN_POSITIVE);
    let vertical_sigma_m = vertical_sigma_m(evidence, per_return_sigma_m, requirement.bias_floor_m);
    let pose_sigmas = evidence.pose.as_ref().and_then(PoseEvidence::sigmas_m);
    let pose_short = evidence
        .pose
        .filter(|pose| pose_sigmas.is_some_and(|s| requirement.pose_falls_short(s, pose.slope)));
    let reason = if evidence.ground_returns == 0 {
        DeficitReason::Unseen
    } else if evidence.accepted_returns == 0 {
        DeficitReason::TooOblique
    } else if requirement.bias_exceeds_sigma() {
        DeficitReason::BiasLimited
    } else if requirement
        .max_image_gsd_m
        .zip(finest_image_gsd_m)
        .is_some_and(|(max, finest)| finest > max)
    {
        DeficitReason::ImageLimited
    } else if let Some(pose) = pose_short {
        // A revisit helps only if some aircraft now knows better.
        match pose.best_now_m {
            Some(best) if requirement.pose_falls_short(best, pose.slope) => {
                DeficitReason::PoseLimited
            }
            _ => DeficitReason::PoseUncertain,
        }
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
    } else if requirement
        .max_image_gsd_m
        .is_some_and(|max| evidence.image_gsd_m.is_none_or(|gsd| gsd > max))
    {
        DeficitReason::ImageCoarse
    } else {
        DeficitReason::Met
    };
    CellQuality {
        reason,
        density_per_m2,
        vertical_sigma_m,
        horizontal_sigma_m: pose_sigmas.map(|[_, horizontal]| horizontal),
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
    #[serde(default)]
    pub image_coarse: u64,
    #[serde(default)]
    pub image_limited: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pose_uncertain: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pose_limited: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
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
            DeficitReason::ImageCoarse => &mut self.image_coarse,
            DeficitReason::ImageLimited => &mut self.image_limited,
            DeficitReason::PoseUncertain => &mut self.pose_uncertain,
            DeficitReason::PoseLimited => &mut self.pose_limited,
        } += 1;
    }

    pub fn total(&self) -> u64 {
        self.met
            + self.unseen
            + self.too_oblique
            + self.bias_limited
            + self.sparse
            + self.stale
            + self.image_coarse
            + self.image_limited
            + self.pose_uncertain
            + self.pose_limited
    }

    /// The share of cells that meet the requirement.
    pub fn met_fraction(&self) -> f64 {
        self.met as f64 / self.total().max(1) as f64
    }

    /// Cells no amount of flying can bring to the requirement.
    pub fn unreachable(&self) -> u64 {
        self.bias_limited + self.image_limited + self.pose_limited
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
            max_image_gsd_m: None,
            max_horizontal_sigma_m: None,
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
            image_gsd_m: None,
            plane: None,
            relief_m2: 0.0,
            pose: None,
        }
    }

    #[test]
    fn each_reason_is_the_first_that_applies() {
        let r = requirement();
        let area = 100.0;
        let reason = |e: &CellEvidence| assess(e, area, 0.3, None, &r).reason;
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
            assess(&evidence(1000, 10.0, 0.0), area, 0.3, None, &biased).reason,
            DeficitReason::BiasLimited,
            "no count beats a bias above the requirement"
        );
        // A bias equal to the requirement is reachable: with no noise and
        // identical heights the sigma is exactly the bias.
        let equal = QualityRequirement {
            bias_floor_m: 0.5,
            ..r.clone()
        };
        let quality = assess(&evidence(8, 10.0, 0.0), area, 0.0, None, &equal);
        assert_eq!(quality.vertical_sigma_m, Some(0.5));
        assert_eq!(quality.reason, DeficitReason::Met);
    }

    /// Returns on a 20% slope, gathered on its uphill side: their mean is the
    /// height where they fell, their plane the height at the centre, and the
    /// plane's sigma is wider for being read away from them.
    #[test]
    fn a_plane_gives_the_centre_of_a_slope_the_returns_missed() {
        let (mut plane, mut sum, mut sq, mut n) = (PlaneMoments::default(), 0.0, 0.0, 0u32);
        for i in 0..4 {
            for j in 0..4 {
                let (u, v) = (1.0 + i as f64, j as f64 - 1.5);
                let z = 50.0 + 0.2 * u;
                plane.add(u, v, z);
                sum += z;
                sq += z * z;
                n += 1;
            }
        }
        let mut cell = CellEvidence {
            accepted_height_sum_m: sum,
            accepted_height_sq_sum_m2: sq,
            ..evidence(n, 0.0, 0.0)
        };
        let mean_sigma = vertical_sigma_m(&cell, 0.3, 0.0).unwrap();
        assert!((centre_height_m(&cell).unwrap() - 50.5).abs() < 1e-9);
        cell.plane = Some(plane);
        let fit = plane_fit(&cell).expect("sixteen returns over an area");
        assert!((fit.height_m - 50.0).abs() < 1e-9, "{fit:?}");
        assert!(fit.residual_m2 < 1e-9, "all on the plane: {fit:?}");
        // 1/16 for the mean, 2.5² over the returns' 20 m² about theirs.
        assert!((fit.leverage - (1.0 / 16.0 + 6.25 / 20.0)).abs() < 1e-9);
        assert_eq!(centre_height_m(&cell), Some(fit.height_m));
        let sigma = vertical_sigma_m(&cell, 0.3, 0.0).unwrap();
        assert!((sigma - 0.3 * fit.leverage.sqrt()).abs() < 1e-9);
        assert!(sigma > mean_sigma);
    }

    /// Curved ground sampled evenly: the plane reads its centre off by the
    /// claim for a valley, 1.4 times it for a bowl and not at all for a saddle
    /// or a twist, however many returns there are.
    #[test]
    fn curved_ground_is_claimed_for_and_does_not_average_away() {
        type Shape = fn(f64, f64) -> f64;
        let shapes: [(&str, Shape, f64); 5] = [
            ("a valley", |u, _| u * u, 1.0),
            ("a ridge", |_, v| -v * v, 1.0),
            ("a bowl", |u, v| u * u + v * v, 2f64.sqrt()),
            ("a saddle", |u, v| u * u - v * v, 0.0),
            ("a twist", |u, v| u * v, 0.0),
        ];
        for per_side in [8, 40] {
            for (name, shape, errors_over_claim) in shapes {
                let (mut plane, mut sum, mut sq) = (PlaneMoments::default(), 0.0, 0.0);
                for i in 0..per_side {
                    for j in 0..per_side {
                        let at = |k: usize| 10.0 * ((k as f64 + 0.5) / per_side as f64 - 0.5);
                        let z = 20.0 + 0.02 * shape(at(i), at(j));
                        plane.add(at(i), at(j), z);
                        sum += z;
                        sq += z * z;
                    }
                }
                let cell = CellEvidence {
                    accepted_height_sum_m: sum,
                    accepted_height_sq_sum_m2: sq,
                    plane: Some(plane),
                    ..evidence(per_side as u32 * per_side as u32, 0.0, 0.0)
                };
                let error = (centre_height_m(&cell).unwrap() - 20.0).abs();
                let sigma = vertical_sigma_m(&cell, 0.0, 0.0).unwrap();
                assert!(
                    (error / sigma - errors_over_claim).abs() < 0.04,
                    "{name}: {error} m against a claimed {sigma} m"
                );
            }
        }
    }

    /// One scan line fixes no slope across it, three returns leave no
    /// residual, and a far corner says less of the centre than one return:
    /// the mean stands in each.
    #[test]
    fn no_plane_is_fitted_where_the_returns_fix_none() {
        let cell = |points: &[(f64, f64)]| {
            let mut plane = PlaneMoments::default();
            for (u, v) in points {
                plane.add(*u, *v, 10.0);
            }
            CellEvidence {
                plane: Some(plane),
                ..evidence(points.len() as u32, 10.0, 0.0)
            }
        };
        let line: Vec<_> = (0..8).map(|k| (k as f64 - 3.5, 2.0)).collect();
        let corner: Vec<_> = (0..8)
            .map(|k| (4.0 + 0.1 * (k % 4) as f64, 4.0 + 0.1 * (k / 4) as f64))
            .collect();
        for (why, points) in [
            ("a line", &line[..]),
            ("three", &[(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0)][..]),
            ("a corner", &corner[..]),
        ] {
            let cell = cell(points);
            assert_eq!(plane_fit(&cell), None, "{why}");
            assert_eq!(centre_height_m(&cell), Some(10.0), "{why}");
            // Level returns, however many, claim the slope beside them.
            assert_eq!(vertical_sigma_m(&cell, 0.3, 0.0), Some(0.3), "{why}");
            let sloping = CellEvidence {
                relief_m2: 4.0,
                ..cell
            };
            assert_eq!(vertical_sigma_m(&sloping, 0.3, 0.0), Some(2.0), "{why}");
        }
        assert_eq!(centre_height_m(&CellEvidence::default()), None);
    }

    /// An image requirement is checked last, so a cell says what else it
    /// lacks first; one no flight can meet is a limit, not a deficit.
    #[test]
    fn a_coarse_image_is_the_last_reason_and_an_unreachable_one_a_limit() {
        let r = QualityRequirement {
            max_image_gsd_m: Some(0.03),
            ..requirement()
        };
        let reason = |gsd, floor| {
            let e = CellEvidence {
                image_gsd_m: gsd,
                ..evidence(8, 10.0, 0.0)
            };
            assess(&e, 100.0, 0.3, Some(floor), &r).reason
        };
        assert_eq!(reason(None, 0.01), DeficitReason::ImageCoarse);
        assert_eq!(reason(Some(0.031), 0.01), DeficitReason::ImageCoarse);
        assert_eq!(reason(Some(0.03), 0.01), DeficitReason::Met);
        assert_eq!(reason(Some(0.03), 0.04), DeficitReason::ImageLimited);
        let sparse = CellEvidence {
            image_gsd_m: None,
            ..evidence(2, 10.0, 0.0)
        };
        assert_eq!(
            assess(&sparse, 100.0, 0.3, Some(0.01), &r).reason,
            DeficitReason::Sparse
        );
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
            image_gsd_m: None,
            plane: None,
            relief_m2: 0.0,
            pose: None,
        };
        // Sample variance 8, over 2, plus bias²: sqrt(4.01).
        let s = vertical_sigma_m(&spread, 0.3, 0.1).unwrap();
        assert!((s - 4.01f64.sqrt()).abs() < 1e-12, "{s}");
    }

    /// `n` returns, each placed by a pose known to these sigmas.
    fn posed(n: u32, vertical: f64, horizontal: f64, slope: f64) -> CellEvidence {
        let weight = pose_weight(vertical);
        let total = weight * f64::from(n);
        CellEvidence {
            pose: Some(PoseEvidence {
                weight: total,
                height_m: total * 10.0,
                vertical_sigma_m: total * vertical,
                horizontal_sigma_m: total * horizontal,
                slope,
                ..PoseEvidence::default()
            }),
            ..evidence(n, 10.0, 0.0)
        }
    }

    #[test]
    fn no_count_averages_away_the_pose_behind_the_returns() {
        let many = vertical_sigma_m(&posed(100_000, 0.4, 0.0, 0.0), 0.3, 0.1).unwrap();
        assert!((many - 0.4f64.hypot(0.1)).abs() < 1e-3, "{many}");
        // A return in the wrong place on a slope is at the wrong height.
        let steep = vertical_sigma_m(&posed(100_000, 0.0, 2.0, 0.25), 0.3, 0.1).unwrap();
        assert!((steep - 0.5f64.hypot(0.1)).abs() < 1e-3, "{steep}");
        // And a pose known exactly changes nothing.
        assert_eq!(
            vertical_sigma_m(&posed(8, 0.0, 0.0, 0.5), 0.3, 0.1),
            vertical_sigma_m(&evidence(8, 10.0, 0.0), 0.3, 0.1)
        );
    }

    /// However the returns' pose errors are related, the claim is no less
    /// than the weighted mean's true sigma: checked at the worst, all the
    /// same error, and for two passes that differ.
    #[test]
    fn the_pose_s_part_is_never_claimed_too_small() {
        let (bad, good) = (2.0f64, 0.02f64);
        let (wb, wg) = (pose_weight(bad), pose_weight(good));
        let pose = PoseEvidence {
            weight: 30.0 * (wb + wg),
            vertical_sigma_m: 30.0 * (wb * bad + wg * good),
            ..PoseEvidence::default()
        };
        let [claimed, _] = pose.sigmas_m().unwrap();
        for correlation in [-1.0, 0.0, 0.5, 1.0] {
            let (a, b) = (wb * bad / (wb + wg), wg * good / (wb + wg));
            let truth = (a * a + b * b + 2.0 * correlation * a * b).sqrt();
            assert!(claimed >= truth - 1e-12, "{claimed} under {truth}");
        }
        // A good pass over a bad one is what the cell becomes.
        assert!(claimed < 0.025, "{claimed}");
    }

    /// `(u, v, z, [vertical, horizontal] sigma)` returns, kept as the map
    /// keeps them with estimated poses and plane heights.
    fn posed_plane(returns: &[(f64, f64, f64, [f64; 2])]) -> CellEvidence {
        let mut plane = PlaneMoments::default();
        let mut pose = PoseEvidence::default();
        let (mut sum, mut sq) = (0.0, 0.0);
        for &(u, v, z, [vertical, horizontal]) in returns {
            let w = pose_weight(vertical);
            plane.add_weighted(w, u, v, z);
            pose.weight += w;
            pose.height_m += w * z;
            pose.height_sq_m2 += w * z * z;
            pose.vertical_sigma_m += w * vertical;
            pose.horizontal_sigma_m += w * horizontal;
            pose.vertical_sq_m2 += w * vertical * vertical;
            pose.horizontal_sq_m2 += w * horizontal * horizontal;
            pose.weight_sq += w * w;
            sum += z;
            sq += z * z;
        }
        CellEvidence {
            ground_returns: returns.len() as u32,
            accepted_returns: returns.len() as u32,
            accepted_height_sum_m: sum,
            accepted_height_sq_sum_m2: sq,
            accepted_age_s: Some(0.0),
            plane: Some(plane),
            pose: Some(pose),
            ..CellEvidence::default()
        }
    }

    fn grid(f: impl Fn(f64, f64) -> (f64, [f64; 2])) -> Vec<(f64, f64, f64, [f64; 2])> {
        let mut returns = Vec::new();
        for i in -2..=2 {
            for j in -2..=2 {
                let (u, v) = (f64::from(i), f64::from(j) * 0.9 + 0.2);
                let (z, sigmas) = f(u, v);
                returns.push((u, v, z, sigmas));
            }
        }
        returns
    }

    /// Returns all placed alike weigh alike, whatever that weight is: the
    /// plane is the one fitted without poses, and their sigma is added whole.
    #[test]
    fn a_plane_through_returns_placed_alike_is_the_unposed_one() {
        for sigma in [0.0, 0.02, 0.8] {
            let returns = grid(|u, v| (50.0 + 0.3 * u - 0.2 * v + 0.1 * (u * v).sin(), [sigma; 2]));
            let posed = posed_plane(&returns);
            let mut plane = PlaneMoments::default();
            for &(u, v, z, _) in &returns {
                plane.add(u, v, z);
            }
            let plain = CellEvidence {
                plane: Some(plane),
                pose: None,
                ..posed
            };
            let (a, b) = (plane_fit(&posed).unwrap(), plane_fit(&plain).unwrap());
            assert!((a.height_m - b.height_m).abs() < 1e-9);
            assert!((a.residual_m2 - b.residual_m2).abs() < 1e-9);
            assert!((a.leverage - b.leverage).abs() < 1e-12);
            let (a, b) = (
                vertical_sigma_m(&posed, 0.05, 0.1).unwrap(),
                vertical_sigma_m(&plain, 0.05, 0.1).unwrap(),
            );
            let gain = (a * a - b * b) / (sigma * sigma).max(1e-300);
            assert!(
                sigma == 0.0 && (a - b).abs() < 1e-9 || (1.0..1.2).contains(&gain),
                "{sigma}: {gain}"
            );
        }
    }

    /// A pass placed 3 m high by an aircraft that knows it is lost barely
    /// moves a plane through a good pass; the good pass alone sets the claim.
    #[test]
    fn a_well_placed_pass_sets_the_plane() {
        let truth = |u: f64, v: f64| 50.0 + 0.3 * u - 0.2 * v;
        let mut returns = grid(|u, v| (truth(u, v), [0.02, 0.02]));
        returns.extend(grid(|u, v| (truth(u, v) + 3.0, [3.0, 3.0])));
        let cell = posed_plane(&returns);
        let height = centre_height_m(&cell).unwrap();
        assert!((height - 50.0).abs() < 0.01, "{height}");
        let sigma = vertical_sigma_m(&cell, 0.05, 0.1).unwrap();
        assert!(sigma < 0.2, "{sigma}");
    }

    /// Worst case: the poses' errors conspire. Whatever signs they take, the
    /// centre height moves by no more than the claim's pose part, even where
    /// the returns lie to one side and the plane reaches out to the centre.
    #[test]
    fn no_pose_errors_move_a_plane_by_more_than_is_claimed() {
        let sigma_of = |u: f64| if u > 0.0 { 0.6 } else { 0.1 };
        for side in [0.0, 1.5] {
            let sound = grid(|u, v| (10.0 + 0.1 * u + 0.1 * v, [sigma_of(u), 0.0]))
                .into_iter()
                .map(|(u, v, z, s)| (u + side, v, z, s))
                .collect::<Vec<_>>();
            let base = posed_plane(&sound);
            let fit = plane_fit(&base).unwrap();
            let claim = (fit.pose_gain * base.pose.unwrap().vertical_sq_m2).sqrt();
            // The sign of each return's pull on the centre, so the worst sum.
            let mut worst = 0.0;
            for k in 0..sound.len() {
                let mut moved = sound.clone();
                moved[k].2 += moved[k].3[0];
                worst += (plane_fit(&posed_plane(&moved)).unwrap().height_m - fit.height_m).abs();
            }
            assert!(claim >= worst - 1e-9, "{claim} under {worst}");
            assert!(claim < 2.5 * worst, "{claim} over {worst}");
        }
    }

    #[test]
    fn a_badly_placed_cell_is_worth_a_revisit_only_by_a_better_pose() {
        let r = requirement();
        let reason = |best| {
            let mut e = posed(50, 0.8, 0.5, 0.0);
            e.pose.as_mut().unwrap().best_now_m = best;
            assess(&e, 100.0, 0.3, None, &r).reason
        };
        assert_eq!(reason(Some([0.02, 0.02])), DeficitReason::PoseUncertain);
        assert_eq!(reason(Some([0.6, 0.4])), DeficitReason::PoseLimited);
        assert_eq!(reason(None), DeficitReason::PoseUncertain);
        assert!(DeficitReason::PoseUncertain.is_improvable());
        assert!(!DeficitReason::PoseLimited.is_improvable());
        // Horizontally, only when the requirement asks.
        let sideways = posed(50, 0.02, 1.5, 0.0);
        assert_eq!(
            assess(&sideways, 100.0, 0.3, None, &r).reason,
            DeficitReason::Met
        );
        let placed = QualityRequirement {
            max_horizontal_sigma_m: Some(1.0),
            ..r
        };
        let quality = assess(&sideways, 100.0, 0.3, None, &placed);
        assert_eq!(quality.reason, DeficitReason::PoseUncertain);
        assert!((quality.horizontal_sigma_m.unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn met_and_unreachable_are_counted_apart() {
        let mut counts = DeficitCounts::default();
        for reason in DeficitReason::ALL {
            counts.add(reason);
        }
        counts.add(DeficitReason::Met);
        assert_eq!(counts.total(), 11);
        assert!((counts.met_fraction() - 2.0 / 11.0).abs() < 1e-12);
        assert_eq!(counts.unreachable(), 3);
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
