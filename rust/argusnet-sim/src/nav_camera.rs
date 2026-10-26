//! The forward navigation camera: the primary obstacle-avoidance sensor of a
//! run without a prior map (SENSING_REALISM_IMPLEMENTATION_GUIDE §8.4).
//!
//! The camera is body-fixed and looks along the direction of flight, so its
//! coverage follows the flight path and never competes with mapping for
//! pointing. The range scanner ([`crate::range_scan`]) is the mapping
//! payload; with the camera installed it no longer needs the forward fan that
//! stood in for forward sensing, and avoidance does not depend on where the
//! scanner looks.
//!
//! # What the camera reports, and what it does not
//!
//! The camera is a perception model, not an image renderer. Each frame casts
//! a grid of sight lines over its field of view and reports a detection where
//! one meets a structure (building, wall, trunk) or a crown within that
//! class's **declared detection range**, with the declared per-frame
//! detection probability and a depth error growing with the square of range
//! (stereo or learned depth both degrade that way). It reports:
//!
//! - **Structure** and **canopy** detections, as points;
//! - **wires** whose nearest conductor point is inside the field of view and
//!   within the range where the wire still spans
//!   [`NavCameraConfig::wire_min_pixels`] pixels.
//!
//! It never reports ground: terrain clearance stays the belief terrain's job.
//! It has no false-alarm model (assumed zero; the profile says so in
//! [`NavCameraConfig::evidence`]). Which object a sight line met comes from
//! the world exactly, as the scanner's return class does: a real system
//! detects it from pixels.
//!
//! # The envelope bounds the flight, not the other way round
//!
//! [`NavCameraEnvelope`] is what the camera can be relied on to see in this
//! run's weather: each class's detection range, shortened where the
//! atmosphere's contrast falls below the camera's threshold first. The
//! fastest safe speed follows from it: the aircraft must stop within the
//! bounding range after the camera's latency, at the declared braking
//! deceleration, with a margin. A run with the camera caps every controller
//! to that speed. Thin wires are outside the bound unless the profile opts
//! in ([`NavCameraConfig::bound_speed_by_wires`]); the envelope records the
//! wire detection range either way, so a run that flies faster than its
//! wire range allows says so.
//!
//! # Heading
//!
//! The sim has no yaw separate from velocity, so the camera looks along the
//! horizontal velocity, and east below 0.5 m/s, exactly as the scanner's
//! forward fan did.

use std::sync::Arc;

use argusnet_world::powerline::PowerLine;
use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::range_scan::{
    span_pieces, wires_seen, RangeReturn, RangeScan, ReturnClass, ScanSensor, SpanPieces,
};
use crate::runtime_world::SharedVisibility;
use crate::weather::MotionWeather;

/// Bound work and allocation from externally supplied per-capture profiles.
pub const MAX_CAPTURE_RAYS: usize = 1_000_000;

/// Where a profile's numbers come from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTag {
    /// Placeholder values, not tied to a device.
    #[default]
    Assumed,
    /// Taken from a vendor datasheet.
    VendorSupported,
    /// Measured on the installed device.
    Measured,
}

/// A navigation camera's declared capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NavCameraConfig {
    /// Full horizontal field of view, degrees.
    pub horizontal_fov_deg: f64,
    /// Full vertical field of view, degrees.
    pub vertical_fov_deg: f64,
    /// How far below the horizon the boresight points, degrees.
    pub boresight_depression_deg: f64,
    /// Sight lines across the field of view.
    pub columns: usize,
    /// Sight lines up the field of view.
    pub rows: usize,
    /// How far a building, wall or trunk is detected, metres.
    pub structure_range_m: f64,
    /// How far a tree's crown is detected, metres.
    pub canopy_range_m: f64,
    /// One pixel's angular size, radians.
    pub pixel_ifov_rad: f64,
    /// How many pixels a wire must span to be detected.
    pub wire_min_pixels: f64,
    /// Per-frame probability a sight line within range detects what it meets.
    pub detection_probability: f64,
    /// Depth error coefficient: one sigma is this times range squared,
    /// metres per square metre.
    pub depth_noise_per_m: f64,
    /// Capture to usable detection, seconds.
    pub latency_s: f64,
    /// The braking deceleration the speed bound assumes, m/s².
    pub braking_mps2: f64,
    /// How far short of a detected obstacle the aircraft must stop, metres.
    pub stop_margin_m: f64,
    /// Whether the speed bound also covers wires of [`Self::wire_radius_m`].
    #[serde(default)]
    pub bound_speed_by_wires: bool,
    /// The thinnest wire the envelope reports a detection range for, metres
    /// of conductor radius.
    pub wire_radius_m: f64,
    /// An object's contrast against its background, before the atmosphere.
    pub intrinsic_contrast: f64,
    /// The apparent contrast below which nothing is detected.
    pub contrast_threshold: f64,
    /// Where these numbers come from.
    #[serde(default)]
    pub evidence: EvidenceTag,
}

impl Default for NavCameraConfig {
    fn default() -> Self {
        Self {
            horizontal_fov_deg: 90.0,
            vertical_fov_deg: 60.0,
            boresight_depression_deg: 10.0,
            columns: 19,
            rows: 9,
            structure_range_m: 150.0,
            canopy_range_m: 100.0,
            // About 1920 pixels across 90 degrees.
            pixel_ifov_rad: 8.0e-4,
            wire_min_pixels: 1.5,
            detection_probability: 0.9,
            depth_noise_per_m: 5.0e-4,
            latency_s: 0.15,
            braking_mps2: 3.0,
            stop_margin_m: 10.0,
            bound_speed_by_wires: false,
            wire_radius_m: 0.01,
            intrinsic_contrast: 0.5,
            contrast_threshold: 0.05,
            evidence: EvidenceTag::Assumed,
        }
    }
}

impl NavCameraConfig {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("nav camera {name} must be positive, got {value}"))
            }
        };
        let non_negative = |name: &str, value: f64| {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "nav camera {name} must not be negative, got {value}"
                ))
            }
        };
        positive("horizontal_fov_deg", self.horizontal_fov_deg)?;
        positive("vertical_fov_deg", self.vertical_fov_deg)?;
        if self.horizontal_fov_deg >= 360.0 || self.vertical_fov_deg >= 180.0 {
            return Err("nav camera field of view must be under 360 x 180 degrees".into());
        }
        if !self.boresight_depression_deg.is_finite() || self.boresight_depression_deg.abs() > 90.0
        {
            return Err(format!(
                "nav camera boresight_depression_deg must be within ±90, got {}",
                self.boresight_depression_deg
            ));
        }
        if self.columns < 2 || self.rows < 2 {
            return Err("nav camera needs at least 2 columns and 2 rows".into());
        }
        if self
            .columns
            .checked_mul(self.rows)
            .is_none_or(|rays| rays > MAX_CAPTURE_RAYS)
        {
            return Err(format!(
                "nav camera exceeds the {MAX_CAPTURE_RAYS} rays per capture limit"
            ));
        }
        positive("structure_range_m", self.structure_range_m)?;
        positive("canopy_range_m", self.canopy_range_m)?;
        positive("pixel_ifov_rad", self.pixel_ifov_rad)?;
        positive("wire_min_pixels", self.wire_min_pixels)?;
        if !(self.detection_probability > 0.0 && self.detection_probability <= 1.0) {
            return Err(format!(
                "nav camera detection_probability must be in (0, 1], got {}",
                self.detection_probability
            ));
        }
        non_negative("depth_noise_per_m", self.depth_noise_per_m)?;
        non_negative("latency_s", self.latency_s)?;
        positive("braking_mps2", self.braking_mps2)?;
        non_negative("stop_margin_m", self.stop_margin_m)?;
        positive("wire_radius_m", self.wire_radius_m)?;
        positive("intrinsic_contrast", self.intrinsic_contrast)?;
        positive("contrast_threshold", self.contrast_threshold)?;
        if self.contrast_threshold >= self.intrinsic_contrast {
            return Err("nav camera contrast_threshold must be below intrinsic_contrast".into());
        }
        Ok(())
    }

    /// How far a wire of conductor radius `radius_m` is detected in clear
    /// air: where its width shrinks to [`Self::wire_min_pixels`] pixels.
    pub fn wire_range_m(&self, radius_m: f64) -> f64 {
        2.0 * radius_m / (self.pixel_ifov_rad * self.wire_min_pixels)
    }

    /// The camera's envelope under `weather` (clear air when `None`).
    pub fn envelope(&self, weather: Option<&MotionWeather>) -> NavCameraEnvelope {
        // Koschmieder: apparent contrast is intrinsic contrast times
        // exp(-3.912 r / V), so the camera loses an object where that falls
        // to its threshold.
        let contrast_range_m = weather.map(|weather| {
            let visibility_m = weather.effective_visibility_m();
            (self.intrinsic_contrast / self.contrast_threshold).ln() * visibility_m / 3.912
        });
        let limit = |range_m: f64| contrast_range_m.map_or(range_m, |c| range_m.min(c));
        let structure_range_m = limit(self.structure_range_m);
        let canopy_range_m = limit(self.canopy_range_m);
        let wire_range_m = limit(self.wire_range_m(self.wire_radius_m));
        let (bounding_range_m, bounded_by) =
            if self.bound_speed_by_wires && wire_range_m < structure_range_m {
                (wire_range_m, "wire")
            } else {
                (structure_range_m, "structure")
            };
        let stop_m = (bounding_range_m - self.stop_margin_m).max(0.0);
        // v * latency + v² / (2 a) = stop distance, for v.
        let a = self.braking_mps2;
        let l = self.latency_s;
        let max_speed_mps = a * (-l + (l * l + 2.0 * stop_m / a).sqrt());
        NavCameraEnvelope {
            structure_range_m,
            canopy_range_m,
            wire_range_m,
            wire_radius_m: self.wire_radius_m,
            contrast_range_m,
            bounding_range_m,
            bounded_by: bounded_by.to_string(),
            max_speed_mps,
        }
    }

    /// Unit sight lines for heading `heading_rad` (east = 0), row by row from
    /// the top, each row left to right.
    pub fn directions(&self, heading_rad: f64) -> Vec<[f64; 3]> {
        let half_h = self.horizontal_fov_deg.to_radians() / 2.0;
        let half_v = self.vertical_fov_deg.to_radians() / 2.0;
        let boresight = -self.boresight_depression_deg.to_radians();
        let mut rays = Vec::with_capacity(self.columns * self.rows);
        for r in 0..self.rows {
            let elevation = boresight + half_v - 2.0 * half_v * r as f64 / (self.rows - 1) as f64;
            for c in 0..self.columns {
                let azimuth =
                    heading_rad + half_h - 2.0 * half_h * c as f64 / (self.columns - 1) as f64;
                rays.push([
                    elevation.cos() * azimuth.cos(),
                    elevation.cos() * azimuth.sin(),
                    elevation.sin(),
                ]);
            }
        }
        rays
    }

    /// Whether direction `d` (not necessarily unit) from the camera lies in
    /// its field of view at heading `heading_rad`.
    fn in_view(&self, heading_rad: f64, d: [f64; 3]) -> bool {
        let horizontal = d[0].hypot(d[1]);
        if horizontal < 1e-9 {
            return false;
        }
        let mut offset = d[1].atan2(d[0]) - heading_rad;
        offset = (offset + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        let elevation = d[2].atan2(horizontal);
        let boresight = -self.boresight_depression_deg.to_radians();
        offset.abs() <= self.horizontal_fov_deg.to_radians() / 2.0
            && (elevation - boresight).abs() <= self.vertical_fov_deg.to_radians() / 2.0
    }
}

/// What the camera can be relied on to see in this run, and the speed that
/// follows. Recorded in `meta.world_knowledge.nav_camera_envelope`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NavCameraEnvelope {
    pub structure_range_m: f64,
    pub canopy_range_m: f64,
    /// For a wire of [`Self::wire_radius_m`].
    pub wire_range_m: f64,
    pub wire_radius_m: f64,
    /// Where the atmosphere's contrast runs out; `None` in clear air.
    pub contrast_range_m: Option<f64>,
    /// The range the speed bound stops within.
    pub bounding_range_m: f64,
    /// `structure`, or `wire` when the profile bounds speed by wires.
    pub bounded_by: String,
    /// The fastest the aircraft may fly and still stop short.
    pub max_speed_mps: f64,
}

/// The camera: a grid of sight lines over the true world.
#[derive(Debug, Clone)]
pub struct NavCamera {
    config: NavCameraConfig,
    envelope: NavCameraEnvelope,
    world: SharedVisibility,
    spans: Arc<Vec<SpanPieces>>,
    /// Where the camera sits on the body and how it is turned (a rig's
    /// mount): lever arm in the body frame, and the mount's rotation. `None`
    /// is a camera at the platform's origin looking along its heading.
    mount: Option<([f64; 3], crate::pod::Quat)>,
}

impl NavCamera {
    pub fn new(
        config: NavCameraConfig,
        world: SharedVisibility,
        weather: Option<&MotionWeather>,
    ) -> Result<Self, String> {
        config.validate()?;
        let envelope = config.envelope(weather);
        Ok(Self {
            config,
            envelope,
            world,
            spans: Arc::new(Vec::new()),
            mount: None,
        })
    }

    /// The camera on a rig's body mount: `lever_arm_m` from the platform's
    /// origin in the body frame (FLU), turned by `rotation_ypr_deg` (yaw,
    /// then pitch nose down, then roll). Its field of view and boresight
    /// depression are the profile's, about the mounted axis.
    pub fn with_mount(mut self, lever_arm_m: [f64; 3], rotation_ypr_deg: [f64; 3]) -> Self {
        self.mount = Some((
            lever_arm_m,
            crate::pod::Quat::from_ypr_deg(rotation_ypr_deg),
        ));
        self
    }

    /// The camera's position and rotation in the world for a platform at
    /// `position_m` heading `heading_rad`; `None` for the rotation when the
    /// camera is unmounted (it looks along the heading).
    fn pose(&self, position_m: [f64; 3], heading_rad: f64) -> ([f64; 3], Option<crate::pod::Quat>) {
        match &self.mount {
            None => (position_m, None),
            Some((lever, rotation)) => {
                let body = crate::pod::Quat::from_axis_angle([0.0, 0.0, 1.0], heading_rad);
                let offset = body.rotate(*lever);
                (
                    std::array::from_fn(|k| position_m[k] + offset[k]),
                    Some(body.then(*rotation)),
                )
            }
        }
    }

    /// The camera, also detecting `lines`' wires.
    pub fn with_power_lines(mut self, lines: &[PowerLine]) -> Self {
        let config = self.config.clone();
        let contrast = self.envelope.contrast_range_m;
        self.spans = Arc::new(span_pieces(lines, |line| {
            let range_m = config.wire_range_m(line.conductor_radius_m);
            contrast.map_or(range_m, |c| range_m.min(c))
        }));
        self
    }

    pub fn config(&self) -> &NavCameraConfig {
        &self.config
    }

    pub fn envelope(&self) -> &NavCameraEnvelope {
        &self.envelope
    }

    /// One frame from `origin_m`, looking along `velocity`. Keyed on the node
    /// and the tick, like the scanner, so a frame is the same however the
    /// run got to that tick.
    pub fn frame(
        &self,
        node_id: &str,
        tick: u64,
        timestamp_s: f64,
        origin_m: [f64; 3],
        velocity: [f64; 3],
    ) -> RangeScan {
        let speed = velocity[0].hypot(velocity[1]);
        let heading = if speed > 0.5 {
            velocity[1].atan2(velocity[0])
        } else {
            0.0
        };
        let key = |stream: &'static str| {
            DeterministicRng::keyed(
                0,
                &[
                    SeedPart::Str(node_id),
                    SeedPart::Str(stream),
                    SeedPart::Int(tick as i64),
                ],
            )
        };
        let mut rng = key("nav-camera");
        let mut canopy = key("nav-camera-canopy");
        let envelope = &self.envelope;
        let cast_m = envelope.structure_range_m.max(envelope.canopy_range_m);
        let mut returns = Vec::new();
        let mut misses = 0u32;
        let (origin_m, rotation) = self.pose(origin_m, heading);
        let directions = match rotation {
            None => self.config.directions(heading),
            Some(rotation) => self
                .config
                .directions(0.0)
                .into_iter()
                .map(|d| rotation.rotate(d))
                .collect(),
        };
        for direction in directions {
            // Drawn for every sight line, so one line's outcome never shifts
            // another's.
            let detect = rng.uniform();
            let noise = rng.normal(0.0, 1.0);
            let end = std::array::from_fn(|k| origin_m[k] + direction[k] * cast_m);
            let Some((kind, point)) = self
                .world
                .first_return(origin_m, end, &mut || canopy.uniform())
            else {
                misses += 1;
                continue;
            };
            let (class, range_limit_m) = match kind.as_str() {
                "building" | "wall" | "trunk" => {
                    (ReturnClass::Structure, envelope.structure_range_m)
                }
                "canopy" => (ReturnClass::Canopy, envelope.canopy_range_m),
                // Ground, or anything else: not the camera's to report.
                _ => {
                    misses += 1;
                    continue;
                }
            };
            let range_m = (0..3)
                .map(|k| (point[k] - origin_m[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            if range_m > range_limit_m || detect >= self.config.detection_probability {
                misses += 1;
                continue;
            }
            let depth_error_m = noise * self.config.depth_noise_per_m * range_m * range_m;
            returns.push(RangeReturn {
                point_m: std::array::from_fn(|k| point[k] + direction[k] * depth_error_m),
                class,
            });
        }
        let wires = wires_seen(&self.spans, origin_m, |point| {
            let d = std::array::from_fn(|k| point[k] - origin_m[k]);
            match rotation {
                None => self.config.in_view(heading, d),
                Some(rotation) => self.config.in_view(0.0, rotation.conj().rotate(d)),
            }
        });
        RangeScan {
            node_id: node_id.to_string(),
            timestamp_s,
            origin_m,
            returns,
            misses,
            wires,
            sensor: ScanSensor::NavCamera,
            pod: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weather::{Atmosphere, Precipitation, WindModel};
    use argusnet_world::index::ObstacleLayer;
    use argusnet_world::obstacles::{BlockerType, Bounds2d, Obstacle};
    use argusnet_world::tiled::TiledTerrain;
    use argusnet_world::visibility::VisibilityProfile;

    fn world_with_wall(x_m: f64) -> SharedVisibility {
        let bounds = Bounds2d::new(0.0, 1000.0, 0.0, 1000.0);
        let terrain = Arc::new(
            TiledTerrain::from_grid(bounds, &vec![vec![0.0; 101]; 101], 10.0, 4, 0.0, &[]).unwrap(),
        );
        let wall = Obstacle::polygon_prism(
            "wall",
            BlockerType::Building,
            vec![
                [x_m, 300.0],
                [x_m + 10.0, 300.0],
                [x_m + 10.0, 700.0],
                [x_m, 700.0],
            ],
            0.0,
            80.0,
        )
        .unwrap();
        SharedVisibility::new(
            terrain,
            Arc::new(ObstacleLayer::new(bounds, 50.0, vec![wall]).unwrap()),
            None,
            Arc::new(VisibilityProfile::default()),
            0.0,
        )
    }

    fn foggy(visibility_m: f64) -> MotionWeather {
        MotionWeather {
            wind: WindModel::default(),
            precipitation: Precipitation::default(),
            atmosphere: Atmosphere {
                visibility_m,
                ..Atmosphere::default()
            },
            cloud_layers: Vec::new(),
        }
    }

    #[test]
    fn the_field_of_view_is_centred_on_the_heading_and_below_the_horizon() {
        let config = NavCameraConfig::default();
        let rays = config.directions(0.0);
        assert_eq!(rays.len(), 19 * 9);
        // The middle sight line is the boresight: east, 10 degrees down.
        let middle = rays[4 * 19 + 9];
        assert!((middle[0] - 10f64.to_radians().cos()).abs() < 1e-12);
        assert!(middle[1].abs() < 1e-12);
        assert!((middle[2] + 10f64.to_radians().sin()).abs() < 1e-12);
        assert!(config.in_view(0.0, [1.0, 0.0, 0.0]));
        assert!(!config.in_view(0.0, [-1.0, 0.0, 0.0]), "never behind");
        assert!(!config.in_view(0.0, [0.0, 1.0, 0.0]), "never abeam");
        assert!(
            !config.in_view(0.0, [0.1, 0.0, -1.0]),
            "never straight down"
        );
    }

    #[test]
    fn invalid_profiles_are_refused() {
        let bad = [
            NavCameraConfig {
                detection_probability: 0.0,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                columns: 1,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                columns: usize::MAX,
                rows: usize::MAX,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                columns: 1001,
                rows: 1000,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                structure_range_m: f64::NAN,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                contrast_threshold: 0.6,
                ..NavCameraConfig::default()
            },
            NavCameraConfig {
                horizontal_fov_deg: 400.0,
                ..NavCameraConfig::default()
            },
        ];
        for config in bad {
            assert!(config.validate().is_err(), "{config:?}");
        }
        assert!(NavCameraConfig::default().validate().is_ok());
    }

    #[test]
    fn the_speed_bound_stops_within_the_bounding_range() {
        let config = NavCameraConfig::default();
        let envelope = config.envelope(None);
        assert_eq!(envelope.bounded_by, "structure");
        assert_eq!(envelope.bounding_range_m, 150.0);
        let v = envelope.max_speed_mps;
        let stop = v * config.latency_s + v * v / (2.0 * config.braking_mps2);
        assert!((stop - (150.0 - config.stop_margin_m)).abs() < 1e-9);
        // A 1 cm wire spans 1.5 pixels at about 16.7 m.
        assert!((envelope.wire_range_m - 0.02 / (8.0e-4 * 1.5)).abs() < 1e-9);

        let by_wires = NavCameraConfig {
            bound_speed_by_wires: true,
            ..config.clone()
        }
        .envelope(None);
        assert_eq!(by_wires.bounded_by, "wire");
        assert!(by_wires.max_speed_mps < v / 3.0, "{by_wires:?}");
    }

    #[test]
    fn fog_shrinks_the_envelope_and_the_speed() {
        let config = NavCameraConfig::default();
        let clear = config.envelope(Some(&foggy(10_000.0)));
        assert_eq!(clear.structure_range_m, 150.0, "10 km air changes nothing");
        let fog = config.envelope(Some(&foggy(150.0)));
        // ln(10) * 150 / 3.912
        let expected = 10f64.ln() * 150.0 / 3.912;
        assert!((fog.structure_range_m - expected).abs() < 1e-9, "{fog:?}");
        assert!(fog.max_speed_mps < clear.max_speed_mps);
    }

    #[test]
    fn a_wall_ahead_is_detected_within_range_and_not_beyond() {
        let camera =
            NavCamera::new(NavCameraConfig::default(), world_with_wall(500.0), None).unwrap();
        // Flying east at 60 m, 100 m short of the wall.
        let near = camera.frame("d", 3, 1.0, [400.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        assert_eq!(near.sensor, ScanSensor::NavCamera);
        assert!(
            near.returns.len() > 20,
            "a wall 100 m ahead fills much of the view: {}",
            near.returns.len()
        );
        assert!(near
            .returns
            .iter()
            .all(|r| r.class == ReturnClass::Structure));
        // On the wall's face, within four sigma of depth error at the
        // furthest sight line's range.
        let sigma_m = NavCameraConfig::default().depth_noise_per_m * 150.0 * 150.0;
        for hit in &near.returns {
            assert!((hit.point_m[0] - 500.0).abs() < 4.0 * sigma_m, "{hit:?}");
        }
        // 200 m short: past the structure range, nothing.
        let far = camera.frame("d", 3, 1.0, [300.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        assert!(far.returns.is_empty(), "{:?}", far.returns);
        // Flying west, away from it: nothing.
        let away = camera.frame("d", 3, 1.0, [400.0, 500.0, 60.0], [-10.0, 0.0, 0.0]);
        assert!(away.returns.is_empty());
        // Keyed on node and tick: the same frame again.
        assert_eq!(
            near,
            camera.frame("d", 3, 1.0, [400.0, 500.0, 60.0], [10.0, 0.0, 0.0])
        );
    }

    /// A rig's mount moves the camera and turns it: mounted backwards, it
    /// sees the wall behind the platform and not the one ahead; a zero
    /// mount is the unmounted camera.
    #[test]
    fn the_mount_places_and_turns_the_camera() {
        let flying_east = |camera: &NavCamera, x: f64| {
            camera.frame("d", 3, 1.0, [x, 500.0, 60.0], [10.0, 0.0, 0.0])
        };
        let plain =
            NavCamera::new(NavCameraConfig::default(), world_with_wall(500.0), None).unwrap();
        let zero = plain.clone().with_mount([0.0; 3], [0.0; 3]);
        let a = flying_east(&plain, 400.0);
        let b = flying_east(&zero, 400.0);
        assert_eq!(a.returns.len(), b.returns.len());
        for (p, q) in a.returns.iter().zip(&b.returns) {
            assert!((0..3).all(|k| (p.point_m[k] - q.point_m[k]).abs() < 1e-9));
        }
        let backwards = plain
            .clone()
            .with_mount([0.45, 0.0, 0.05], [180.0, 0.0, 0.0]);
        assert!(
            flying_east(&backwards, 400.0).returns.is_empty(),
            "the wall is ahead"
        );
        let past = flying_east(&backwards, 610.0);
        assert!(!past.returns.is_empty(), "the wall is behind");
        // The lever arm moves the origin: 0.45 m forward of the platform,
        // which faces east.
        assert!(
            (past.origin_m[0] - 610.45).abs() < 1e-9 && (past.origin_m[2] - 60.05).abs() < 1e-9
        );
    }

    #[test]
    fn detection_is_probabilistic_per_sight_line() {
        let config = NavCameraConfig {
            detection_probability: 0.5,
            ..NavCameraConfig::default()
        };
        let all = NavCamera::new(NavCameraConfig::default(), world_with_wall(500.0), None)
            .unwrap()
            .frame("d", 7, 1.0, [400.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        let half = NavCamera::new(config, world_with_wall(500.0), None)
            .unwrap()
            .frame("d", 7, 1.0, [400.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        assert!(half.returns.len() < all.returns.len());
        assert!(!half.returns.is_empty());
    }

    #[test]
    fn a_wire_is_seen_ahead_within_its_pixel_range_only() {
        use argusnet_world::powerline::{PowerLine, Support};
        let support = |y: f64| Support {
            xy_m: [520.0, y],
            base_z_m: 0.0,
            height_m: 62.0,
        };
        let line = PowerLine {
            id: "line".into(),
            supports: vec![support(0.0), support(1000.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.0,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap();
        let camera = NavCamera::new(NavCameraConfig::default(), world_with_wall(900.0), None)
            .unwrap()
            .with_power_lines(std::slice::from_ref(&line));
        // 3 cm across spans 1.5 pixels at 25 m.
        let range = NavCameraConfig::default().wire_range_m(0.015);
        assert!((range - 25.0).abs() < 1e-9);
        let seen = camera.frame("d", 1, 1.0, [500.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        assert_eq!(seen.wires.len(), 1, "20 m ahead");
        let too_far = camera.frame("d", 1, 1.0, [490.0, 500.0, 60.0], [10.0, 0.0, 0.0]);
        assert!(too_far.wires.is_empty(), "30 m ahead");
        let behind = camera.frame("d", 1, 1.0, [500.0, 500.0, 60.0], [-10.0, 0.0, 0.0]);
        assert!(behind.wires.is_empty(), "behind the camera");
    }
}
