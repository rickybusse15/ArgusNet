//! LiDAR returns (SR3): what a pod LiDAR actually gets back from a surface,
//! in place of a perfect return from every ray that meets one.
//!
//! Without a [`LidarReturnModel`] every ray that hits something within range
//! returns, from one instant, with Gaussian range noise and nothing else.
//! With one, following the sensing guide's section 5:
//!
//! - **Per-firing time.** A LiDAR fires its rays across the capture, not at
//!   once. Ray `k` of `n` is acquired `k / n` of the scan period after the
//!   capture's timestamp, from where the aircraft has moved to by then. The
//!   estimate is deskewed with the aircraft's velocity, so the timing shows
//!   only where the velocity or attitude estimate is wrong. Attitude is held
//!   over one capture.
//! - **Admission.** A return clears the detector with a probability set by
//!   the power it carries back: reflectivity times the cosine of incidence,
//!   over range squared. The threshold is what a 10%-reflective target
//!   facing the beam returns at the reference range, and the chance falls
//!   steeply there (a logistic in power, as the wire model's,
//!   [`crate::wire_sensing`]). A dark surface, a far one and one met at a
//!   grazing angle are all missed more often.
//! - **Separate errors.** Range noise (the LiDAR's own), a persistent range
//!   bias, range quantization, angular noise in where the estimator believes
//!   the ray pointed, and dropout, each declared on its own: precision at
//!   short range says nothing about the rest of the envelope.
//!
//! Not modelled: false returns and multiple returns per pulse (the guide asks
//! for an actual model, not duplicated points), intensity, wavelength and
//! ambient light effects, and surface normals on structures, which are taken
//! as facing the beam. Every figure is assumed until a device is chosen.

use serde::{Deserialize, Serialize};

use crate::nav_camera::EvidenceTag;

/// Diffuse reflectivity of each kind of surface the world has, in `(0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceReflectivity {
    pub terrain: f64,
    pub structure: f64,
    pub canopy: f64,
    pub trunk: f64,
}

impl Default for SurfaceReflectivity {
    /// Near-infrared, assumed: soil and grass, concrete and roofing, leaves,
    /// bark.
    fn default() -> Self {
        Self {
            terrain: 0.25,
            structure: 0.3,
            canopy: 0.4,
            trunk: 0.15,
        }
    }
}

/// What the pod LiDAR's returns are declared to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarReturnModel {
    /// The range at which a 10%-reflective target facing the beam is
    /// detected half the time, metres: the quoted range.
    pub reference_range_m: f64,
    /// How sharply detection falls at the threshold: the exponent of the
    /// logistic in return power.
    pub detection_steepness: f64,
    pub reflectivity: SurfaceReflectivity,
    /// Full beam divergence, radians: the footprint grows by this per metre.
    pub beam_divergence_rad: f64,
    /// One-sigma error in where the estimator believes a ray pointed,
    /// radians, each axis.
    pub angular_noise_rad: f64,
    /// A range error every return carries, metres.
    pub range_bias_m: f64,
    /// The range resolution, metres; 0 for none.
    pub range_quantization_m: f64,
    /// The chance a return that cleared the detector is lost anyway.
    pub dropout_probability: f64,
    /// How long the rays of one capture take to fire, seconds; `None` is the
    /// whole interval between captures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_period_s: Option<f64>,
    #[serde(default)]
    pub evidence: EvidenceTag,
}

impl Default for LidarReturnModel {
    fn default() -> Self {
        Self {
            reference_range_m: 250.0,
            detection_steepness: 4.0,
            reflectivity: SurfaceReflectivity::default(),
            beam_divergence_rad: 2.0e-3,
            angular_noise_rad: 5.0e-4,
            range_bias_m: 0.02,
            range_quantization_m: 0.005,
            dropout_probability: 0.002,
            scan_period_s: None,
            evidence: EvidenceTag::Assumed,
        }
    }
}

impl LidarReturnModel {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "lidar returns {name} must be positive, got {value}"
                ))
            }
        };
        let non_negative = |name: &str, value: f64| {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "lidar returns {name} must not be negative, got {value}"
                ))
            }
        };
        positive("reference_range_m", self.reference_range_m)?;
        positive("detection_steepness", self.detection_steepness)?;
        positive("beam_divergence_rad", self.beam_divergence_rad)?;
        non_negative("angular_noise_rad", self.angular_noise_rad)?;
        non_negative("range_quantization_m", self.range_quantization_m)?;
        if !self.range_bias_m.is_finite() {
            return Err("lidar returns range_bias_m must be finite".into());
        }
        if !(0.0..1.0).contains(&self.dropout_probability) {
            return Err("lidar returns dropout_probability must be in [0, 1)".into());
        }
        if let Some(period) = self.scan_period_s {
            positive("scan_period_s", period)?;
        }
        let r = &self.reflectivity;
        for (name, value) in [
            ("terrain", r.terrain),
            ("structure", r.structure),
            ("canopy", r.canopy),
            ("trunk", r.trunk),
        ] {
            if !(value > 0.0 && value <= 1.0) {
                return Err(format!(
                    "lidar returns {name} reflectivity must be in (0, 1], got {value}"
                ));
            }
        }
        Ok(())
    }

    /// The reflectivity of a surface of `kind`, as the world names it.
    pub fn reflectivity_of(&self, kind: &str) -> Option<f64> {
        let r = &self.reflectivity;
        match kind {
            "terrain" => Some(r.terrain),
            "building" | "wall" => Some(r.structure),
            "canopy" => Some(r.canopy),
            "trunk" => Some(r.trunk),
            _ => None,
        }
    }

    /// The chance a return from a surface of `reflectivity`, met at
    /// `cos_incidence` from its normal, `range_m` away, clears the detector.
    pub fn detection_probability(
        &self,
        reflectivity: f64,
        cos_incidence: f64,
        range_m: f64,
    ) -> f64 {
        let range_m = range_m.max(1e-3);
        let power = reflectivity * cos_incidence.clamp(0.0, 1.0) / (range_m * range_m);
        let threshold = 0.1 / (self.reference_range_m * self.reference_range_m);
        let rising = (power / threshold).powf(self.detection_steepness);
        rising / (1.0 + rising)
    }

    /// The beam's footprint on a surface square to it, `range_m` away,
    /// metres: not range precision, and not the spacing between rays.
    pub fn footprint_m(&self, range_m: f64) -> f64 {
        range_m * self.beam_divergence_rad
    }

    /// A measured range as the device reports it: biased and quantized.
    pub fn reported_range_m(&self, range_m: f64) -> f64 {
        let biased = range_m + self.range_bias_m;
        if self.range_quantization_m > 0.0 {
            (biased / self.range_quantization_m).round() * self.range_quantization_m
        } else {
            biased
        }
    }
}

/// Unit `ray` turned by small angles `[about y, about z]`, radians: where
/// the estimator believes a ray pointed, off by its angular noise.
pub fn perturbed(ray: [f64; 3], angles: [f64; 2]) -> [f64; 3] {
    let turned = [
        ray[0] - angles[1] * ray[1] + angles[0] * ray[2],
        ray[1] + angles[1] * ray[0],
        ray[2] - angles[0] * ray[0],
    ];
    let n = (turned[0] * turned[0] + turned[1] * turned[1] + turned[2] * turned[2]).sqrt();
    turned.map(|v| v / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LidarReturnModel {
        LidarReturnModel::default()
    }

    /// A 10%-reflective target facing the beam is detected half the time at
    /// the reference range, and nearly always well inside it.
    #[test]
    fn a_reference_target_is_seen_half_the_time_at_the_reference_range() {
        let m = model();
        assert!((m.detection_probability(0.1, 1.0, 250.0) - 0.5).abs() < 1e-9);
        assert!(m.detection_probability(0.1, 1.0, 100.0) > 0.99);
        assert!(m.detection_probability(0.1, 1.0, 400.0) < 0.05);
    }

    /// Darker, further and more grazing all lose returns.
    #[test]
    fn dark_far_and_grazing_surfaces_return_less() {
        let m = model();
        let p = |r: f64, c: f64, d: f64| m.detection_probability(r, c, d);
        assert!(p(0.05, 1.0, 250.0) < p(0.25, 1.0, 250.0));
        assert!(p(0.25, 1.0, 400.0) < p(0.25, 1.0, 250.0));
        assert!(p(0.25, 0.1, 250.0) < p(0.25, 1.0, 250.0));
        assert_eq!(p(0.25, 0.0, 50.0), 0.0, "edge-on returns nothing");
    }

    #[test]
    fn a_report_is_biased_and_quantized() {
        let m = model();
        let reported = m.reported_range_m(100.0012);
        assert!((reported - 100.02).abs() < 1e-9, "{reported}");
        assert!(((reported / 0.005).round() * 0.005 - reported).abs() < 1e-12);
        let exact = LidarReturnModel {
            range_bias_m: 0.0,
            range_quantization_m: 0.0,
            ..model()
        };
        assert_eq!(exact.reported_range_m(100.0012), 100.0012);
    }

    /// For small angles the footprint is range times divergence.
    #[test]
    fn the_footprint_grows_with_range() {
        let m = model();
        assert!((m.footprint_m(100.0) - 0.2).abs() < 1e-12);
        assert!((m.footprint_m(250.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_perturbed_ray_stays_unit_and_turns_by_the_angle() {
        let ray = [1.0, 0.0, 0.0];
        let turned = perturbed(ray, [0.0, 1e-3]);
        let n: f64 = turned.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12);
        let angle = (turned[0]).acos();
        assert!((angle - 1e-3).abs() < 1e-6, "{angle}");
        assert_eq!(perturbed(ray, [0.0, 0.0]), ray);
    }

    #[test]
    fn surfaces_are_named_as_the_world_names_them() {
        let m = model();
        assert_eq!(m.reflectivity_of("terrain"), Some(0.25));
        assert_eq!(m.reflectivity_of("wall"), Some(0.3));
        assert_eq!(m.reflectivity_of("water"), None);
    }

    #[test]
    fn nonsense_is_refused() {
        assert!(model().validate().is_ok());
        let bad = LidarReturnModel {
            dropout_probability: 1.0,
            ..model()
        };
        assert!(bad.validate().is_err());
        let mut dark = model();
        dark.reflectivity.canopy = 0.0;
        assert!(dark.validate().is_err());
        let bad = LidarReturnModel {
            scan_period_s: Some(0.0),
            ..model()
        };
        assert!(bad.validate().is_err());
    }
}
