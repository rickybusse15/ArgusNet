//! Imaging (SR4, first slice): what each pod image is worth, without
//! rendering it.
//!
//! The rig's pod carries two visible cameras and a thermal camera
//! ([`crate::rig`]) that nothing modelled. This models what decides whether
//! an image is usable, per capture, from the same pose the LiDAR was fired
//! from:
//!
//! - **Ground sample distance.** The size of a pixel where the boresight
//!   meets the world: slant range over the focal length in pixels,
//!   stretched along the look direction by the obliquity.
//! - **Motion blur.** How many pixels the scene moves across during the
//!   exposure: the aircraft's speed across the boresight plus the pod's own
//!   rotation, at that range.
//! - **Refusals.** A stowed or unsettled pod, a boresight that meets nothing,
//!   a look too oblique, a blur past the limit.
//! - **Stereo.** The depth error the two visible cameras' baseline gives at
//!   that range.
//! - **Thermal.** A microbolometer integrates over its time constant, so its
//!   blur is that of an exposure as long; a target smeared across `n`
//!   pixels needs `n` times the contrast to be detected.
//!
//! No pixels and no temperatures: an image here is its geometry and its
//! timing. Every number is assumed ([`ImagingConfig::evidence`]).

use serde::{Deserialize, Serialize};

use crate::nav_camera::EvidenceTag;
use crate::rig::CameraOptics;

/// The imaging model's parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagingConfig {
    /// The visible cameras' exposure, seconds.
    pub visible_exposure_s: f64,
    /// The most blur a visible image may carry, pixels.
    pub visible_max_blur_px: f64,
    /// The furthest from straight down a boresight may meet the ground and
    /// its image still count, degrees.
    pub max_incidence_deg: f64,
    /// The thermal detector's time constant, seconds: its effective
    /// exposure.
    pub thermal_time_constant_s: f64,
    /// The most blur a thermal image may carry, pixels.
    pub thermal_max_blur_px: f64,
    /// The thermal camera's noise-equivalent temperature difference, kelvin.
    pub thermal_netd_k: f64,
    /// One-sigma disparity error of a stereo match, pixels.
    pub stereo_disparity_sigma_px: f64,
    /// The lowest a flight comes over the ground for an image requirement,
    /// metres: below it the requirement is unreachable, not flown.
    pub min_image_agl_m: f64,
    /// The share of the height an image requirement allows that is flown:
    /// room for terrain the map does not hold yet.
    pub gsd_margin: f64,
    #[serde(default)]
    pub evidence: EvidenceTag,
}

impl Default for ImagingConfig {
    fn default() -> Self {
        Self {
            visible_exposure_s: 1.0e-3,
            visible_max_blur_px: 1.0,
            max_incidence_deg: 45.0,
            thermal_time_constant_s: 8.0e-3,
            thermal_max_blur_px: 1.0,
            thermal_netd_k: 0.05,
            stereo_disparity_sigma_px: 0.25,
            min_image_agl_m: 40.0,
            gsd_margin: 0.9,
            evidence: EvidenceTag::Assumed,
        }
    }
}

impl ImagingConfig {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("imaging {name} must be positive, got {value}"))
            }
        };
        positive("visible_exposure_s", self.visible_exposure_s)?;
        positive("visible_max_blur_px", self.visible_max_blur_px)?;
        positive("thermal_time_constant_s", self.thermal_time_constant_s)?;
        positive("thermal_max_blur_px", self.thermal_max_blur_px)?;
        positive("thermal_netd_k", self.thermal_netd_k)?;
        positive("stereo_disparity_sigma_px", self.stereo_disparity_sigma_px)?;
        positive("min_image_agl_m", self.min_image_agl_m)?;
        if !(self.max_incidence_deg.is_finite()
            && self.max_incidence_deg > 0.0
            && self.max_incidence_deg < 90.0)
        {
            return Err(format!(
                "imaging max_incidence_deg must be in (0, 90), got {}",
                self.max_incidence_deg
            ));
        }
        if !(self.gsd_margin.is_finite() && self.gsd_margin > 0.0 && self.gsd_margin <= 1.0) {
            return Err(format!(
                "imaging gsd_margin must be in (0, 1], got {}",
                self.gsd_margin
            ));
        }
        Ok(())
    }

    /// The height over the ground to fly for images no coarser than
    /// `gsd_m` straight down, with the margin; never below the floor.
    pub fn agl_for_gsd_m(&self, optics: &CameraOptics, gsd_m: f64) -> f64 {
        (self.gsd_margin * gsd_m * focal_px(optics)).max(self.min_image_agl_m)
    }

    /// The finest pixel on the ground any flight gets: straight down from
    /// the floor.
    pub fn finest_gsd_m(&self, optics: &CameraOptics) -> f64 {
        self.min_image_agl_m / focal_px(optics)
    }

    /// The fastest a flight at `agl_m` moves and its nadir visible image
    /// stays within the blur limit, m/s.
    pub fn max_sharp_speed_mps(&self, optics: &CameraOptics, agl_m: f64) -> f64 {
        self.visible_max_blur_px * agl_m / focal_px(optics) / self.visible_exposure_s
    }
}

/// A camera's focal length in pixels, from its horizontal field of view.
pub fn focal_px(optics: &CameraOptics) -> f64 {
    (optics.width_px as f64 / 2.0) / (optics.horizontal_fov_deg.to_radians() / 2.0).tan()
}

/// Why an image does not count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageRefusal {
    /// The pod was stowed: below cruise.
    Stowed,
    /// The pod had not settled on its command.
    Unsettled,
    /// The boresight met nothing.
    NoGround,
    /// The boresight met the ground too far from straight down.
    TooOblique,
    /// The scene moved too many pixels during the exposure.
    Blurred,
}

/// One camera's image at one capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageCapture {
    /// Absent for an image that counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<ImageRefusal>,
    /// Slant range to where the boresight meets the world, metres. This and
    /// the rest are absent when the pod was not looking or met nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_m: Option<f64>,
    /// The pixel's size on the ground along the look direction, metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gsd_m: Option<f64>,
    /// Scene motion during the exposure, pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blur_px: Option<f64>,
    /// The boresight's angle from straight down, degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incidence_deg: Option<f64>,
}

impl ImageCapture {
    pub fn refused(refusal: ImageRefusal) -> Self {
        Self {
            refusal: Some(refusal),
            range_m: None,
            gsd_m: None,
            blur_px: None,
            incidence_deg: None,
        }
    }

    pub fn accepted(&self) -> bool {
        self.refusal.is_none()
    }
}

/// What the pod's cameras got at one capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodImages {
    pub visible: ImageCapture,
    /// One-sigma depth error of the visible pair at the boresight's range,
    /// metres. Absent without a pair or a range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stereo_depth_sigma_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<ImageCapture>,
    /// The smallest temperature difference of a pixel-sized target the
    /// thermal image shows, kelvin: the NETD times the pixels it is smeared
    /// across.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_min_contrast_k: Option<f64>,
    /// Where the estimator has the visible camera: how the map places the
    /// image. Absent when the image does not count, and in replays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible_pose: Option<ImagePose>,
}

impl PodImages {
    /// Rounded to what a replay shows, without the pose the map used.
    pub fn round_for_replay(&mut self) {
        let round = |value: &mut Option<f64>, scale: f64| {
            *value = value.map(|v| (v * scale).round() / scale);
        };
        self.visible_pose = None;
        round(&mut self.stereo_depth_sigma_m, 1e3);
        round(&mut self.thermal_min_contrast_k, 1e4);
        for image in std::iter::once(&mut self.visible).chain(self.thermal.as_mut()) {
            round(&mut image.range_m, 1e2);
            round(&mut image.gsd_m, 1e5);
            round(&mut image.blur_px, 1e3);
            round(&mut image.incidence_deg, 1e2);
        }
    }
}

/// A camera's estimated pose at capture.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImagePose {
    pub origin_m: [f64; 3],
    /// The camera frame (x along the boresight, y left, z up) in the world.
    pub world_camera: crate::pod::Quat,
}

/// How the map turns an accepted visible image into evidence: which ground it
/// covers and at what sample distance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImageFootprint {
    focal_px: f64,
    tan_half_h: f64,
    tan_half_v: f64,
    max_incidence_deg: f64,
}

impl ImageFootprint {
    pub fn new(optics: &CameraOptics, config: &ImagingConfig) -> Self {
        Self {
            focal_px: focal_px(optics),
            tan_half_h: (optics.horizontal_fov_deg.to_radians() / 2.0).tan(),
            tan_half_v: (optics.vertical_fov_deg.to_radians() / 2.0).tan(),
            max_incidence_deg: config.max_incidence_deg,
        }
    }

    /// The furthest across the ground from the camera an image counts, a
    /// drop of `drop_m` below it, metres.
    pub fn reach_m(&self, drop_m: f64) -> f64 {
        drop_m.max(0.0) * self.max_incidence_deg.to_radians().tan()
    }

    /// The ground sample distance at `point_m`, metres: `None` when it is
    /// outside the image or seen too obliquely.
    pub fn gsd_at(&self, pose: &ImagePose, point_m: [f64; 3]) -> Option<f64> {
        let ray: [f64; 3] = std::array::from_fn(|k| point_m[k] - pose.origin_m[k]);
        let drop = -ray[2];
        let horizontal = ray[0].hypot(ray[1]);
        if drop <= 0.0 || horizontal.atan2(drop).to_degrees() > self.max_incidence_deg {
            return None;
        }
        let local = pose.world_camera.conj().rotate(ray);
        let depth = local[0];
        if depth <= 0.0
            || local[1].abs() > depth * self.tan_half_h
            || local[2].abs() > depth * self.tan_half_v
        {
            return None;
        }
        // Depth over focal length on a plane facing the camera; on the
        // ground, stretched by how much further that falls away than the
        // plane does. Straight down it is the height over the focal length
        // everywhere in the image.
        Some(depth / self.focal_px * (depth / drop).max(1e-9))
    }
}

/// The view one camera has: how far its boresight runs before it meets the
/// world, how far from straight down, how fast the aircraft crosses it and
/// how fast the pod turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub range_m: f64,
    /// The boresight's angle from straight down, radians.
    pub incidence_rad: f64,
    /// The aircraft's speed across the boresight, m/s.
    pub cross_speed_mps: f64,
    /// The pod's rotation rate, rad/s.
    pub turn_rate_rad_s: f64,
}

/// One camera's image of `view`, exposed for `exposure_s`.
pub fn assess(
    optics: &CameraOptics,
    view: View,
    exposure_s: f64,
    max_blur_px: f64,
    max_incidence_deg: f64,
) -> ImageCapture {
    let focal = focal_px(optics);
    // Across the look direction a pixel is range over focal length; along
    // it the ground falls away, so it covers more. The worse is reported.
    let cross_gsd_m = view.range_m / focal;
    let gsd_m = cross_gsd_m / view.incidence_rad.cos().max(1e-6);
    // Motion in the image plane: translation across the boresight and the
    // pod's rotation, both at the pixel's angular size.
    let blur_px =
        (view.cross_speed_mps / view.range_m.max(1e-9) + view.turn_rate_rad_s) * exposure_s * focal;
    let incidence_deg = view.incidence_rad.to_degrees();
    let refusal = if incidence_deg > max_incidence_deg {
        Some(ImageRefusal::TooOblique)
    } else if blur_px > max_blur_px {
        Some(ImageRefusal::Blurred)
    } else {
        None
    };
    ImageCapture {
        refusal,
        range_m: Some(view.range_m),
        gsd_m: Some(gsd_m),
        blur_px: Some(blur_px),
        incidence_deg: Some(incidence_deg),
    }
}

/// One-sigma depth error of a stereo pair `baseline_m` apart at `range_m`.
pub fn stereo_depth_sigma_m(
    optics: &CameraOptics,
    baseline_m: f64,
    range_m: f64,
    disparity_sigma_px: f64,
) -> f64 {
    range_m * range_m * disparity_sigma_px / (focal_px(optics) * baseline_m.max(1e-9))
}

/// The smallest temperature difference of a pixel-sized target an image
/// blurred across `blur_px` pixels shows.
pub fn thermal_min_contrast_k(netd_k: f64, blur_px: f64) -> f64 {
    netd_k * blur_px.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible() -> CameraOptics {
        CameraOptics {
            horizontal_fov_deg: 60.0,
            vertical_fov_deg: 45.0,
            width_px: 4000,
            height_px: 3000,
        }
    }

    fn nadir(range_m: f64, speed_mps: f64) -> View {
        View {
            range_m,
            incidence_rad: 0.0,
            cross_speed_mps: speed_mps,
            turn_rate_rad_s: 0.0,
        }
    }

    #[test]
    fn a_pixel_is_range_over_focal_length_and_stretches_when_oblique() {
        let focal = focal_px(&visible());
        assert!((focal - 2000.0 / 30f64.to_radians().tan()).abs() < 1e-9);
        let config = ImagingConfig::default();
        let down = assess(&visible(), nadir(100.0, 0.0), 1e-3, 1.0, 45.0);
        assert!((down.gsd_m.unwrap() - 100.0 / focal).abs() < 1e-12);
        assert!(down.accepted());
        let oblique = assess(
            &visible(),
            View {
                incidence_rad: 40f64.to_radians(),
                ..nadir(100.0, 0.0)
            },
            1e-3,
            1.0,
            config.max_incidence_deg,
        );
        assert!((oblique.gsd_m.unwrap() - 100.0 / focal / 40f64.to_radians().cos()).abs() < 1e-12);
        let too = assess(
            &visible(),
            View {
                incidence_rad: 50f64.to_radians(),
                ..nadir(100.0, 0.0)
            },
            1e-3,
            1.0,
            config.max_incidence_deg,
        );
        assert_eq!(too.refusal, Some(ImageRefusal::TooOblique));
    }

    #[test]
    fn blur_is_the_pixels_crossed_in_the_exposure() {
        let focal = focal_px(&visible());
        // 20 m/s at 100 m for 1 ms: 2 cm of ground, in pixels of 100/focal.
        let image = assess(&visible(), nadir(100.0, 20.0), 1e-3, 1.0, 45.0);
        let expected = 0.02 / (100.0 / focal);
        assert!((image.blur_px.unwrap() - expected).abs() < 1e-9);
        assert!(image.accepted(), "{expected} px is within a pixel");
        // The same flight from half the height blurs twice as much.
        let low = assess(&visible(), nadir(50.0, 20.0), 1e-3, 1.0, 45.0);
        assert!((low.blur_px.unwrap() - 2.0 * expected).abs() < 1e-9);
        assert_eq!(low.refusal, Some(ImageRefusal::Blurred));
        // A turning pod blurs at any range.
        let turning = assess(
            &visible(),
            View {
                turn_rate_rad_s: 0.4,
                ..nadir(100.0, 0.0)
            },
            1e-3,
            1.0,
            45.0,
        );
        assert!((turning.blur_px.unwrap() - 0.4 * 1e-3 * focal).abs() < 1e-9);
        assert_eq!(turning.refusal, Some(ImageRefusal::Blurred));
    }

    #[test]
    fn the_sharp_speed_and_the_height_for_a_pixel_agree_with_the_model() {
        let config = ImagingConfig::default();
        let optics = visible();
        let agl = config.agl_for_gsd_m(&optics, 0.02);
        assert!((agl - 0.9 * 0.02 * focal_px(&optics)).abs() < 1e-9);
        let speed = config.max_sharp_speed_mps(&optics, agl);
        let at_limit = assess(&optics, nadir(agl, speed), 1e-3, 1.0, 45.0);
        assert!((at_limit.blur_px.unwrap() - 1.0).abs() < 1e-9);
        assert!(at_limit.gsd_m.unwrap() <= 0.02);
        // A pixel finer than the floor allows is not flown for.
        assert_eq!(config.agl_for_gsd_m(&optics, 0.001), config.min_image_agl_m);
        assert!(config.finest_gsd_m(&optics) > 0.001);
    }

    /// Straight down, every pixel on flat ground is the same size; past the
    /// image's edge or the incidence limit there is none.
    #[test]
    fn a_nadir_footprint_has_one_sample_distance_to_its_edge() {
        let optics = visible();
        let footprint = ImageFootprint::new(&optics, &ImagingConfig::default());
        let pose = ImagePose {
            origin_m: [0.0, 0.0, 100.0],
            world_camera: crate::pod::Quat::from_ypr(0.0, std::f64::consts::FRAC_PI_2, 0.0),
        };
        let expected = 100.0 / focal_px(&optics);
        for at in [[0.0, 0.0, 0.0], [30.0, 50.0, 0.0], [-40.0, -55.0, 0.0]] {
            let gsd = footprint.gsd_at(&pose, at).unwrap();
            assert!((gsd - expected).abs() < 1e-9, "{at:?}: {gsd}");
        }
        // 45 degrees along the track reaches 41.4 m, 60 across it 57.7 m.
        assert_eq!(footprint.gsd_at(&pose, [42.0, 0.0, 0.0]), None);
        assert_eq!(footprint.gsd_at(&pose, [0.0, 58.0, 0.0]), None);
        assert_eq!(footprint.gsd_at(&pose, [0.0, 0.0, 101.0]), None);
        assert!(footprint.reach_m(100.0) > 58.0);
        // Tilted 30 degrees forward, the boresight's pixel is stretched.
        let tilted = ImagePose {
            world_camera: crate::pod::Quat::from_ypr(0.0, 60f64.to_radians(), 0.0),
            ..pose
        };
        let ahead = 100.0 * 30f64.to_radians().tan();
        let gsd = footprint.gsd_at(&tilted, [ahead, 0.0, 0.0]).unwrap();
        let slant = 100.0 / 30f64.to_radians().cos();
        let along = slant / focal_px(&optics) / 30f64.to_radians().cos();
        assert!((gsd - along).abs() < 1e-9, "{gsd} against {along}");
    }

    #[test]
    fn stereo_error_grows_with_the_square_of_range() {
        let near = stereo_depth_sigma_m(&visible(), 0.08, 50.0, 0.25);
        let far = stereo_depth_sigma_m(&visible(), 0.08, 100.0, 0.25);
        assert!((far / near - 4.0).abs() < 1e-9);
        // 8 cm at 100 m: metres, not a height measurement.
        assert!(far > 5.0, "{far}");
    }

    #[test]
    fn thermal_contrast_scales_with_the_smear() {
        assert_eq!(thermal_min_contrast_k(0.05, 0.3), 0.05);
        assert!((thermal_min_contrast_k(0.05, 6.0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn a_bad_config_is_refused() {
        assert!(ImagingConfig::default().validate().is_ok());
        for bad in [
            ImagingConfig {
                visible_exposure_s: 0.0,
                ..Default::default()
            },
            ImagingConfig {
                max_incidence_deg: 90.0,
                ..Default::default()
            },
            ImagingConfig {
                gsd_margin: 1.5,
                ..Default::default()
            },
            ImagingConfig {
                min_image_agl_m: -1.0,
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
