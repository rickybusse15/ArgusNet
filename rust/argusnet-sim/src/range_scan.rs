//! A drone-mounted range scanner: the sensor a run without a prior map builds
//! its map from.
//!
//! Each mobile platform casts a fixed fan of rays every tick: rings below the
//! horizon at several depression angles, a nadir ray, and a forward fan just
//! under the horizon that sees rising ground and walls before the platform
//! reaches them. A ray returns the first surface it meets within range, with
//! seeded range noise, or nothing.
//!
//! The scanner is a sensor model, so it is the one place outside physics that
//! reads the true world, as stage 6's bearing sensors do. What it reports is a
//! point and a class, nothing more:
//!
//! - **Ground**: the ray met terrain.
//! - **Structure**: it met a building or wall.
//!
//! The class models ground segmentation on the scanner's point cloud, which a
//! real system computes from the cloud itself. This model takes it from the
//! world exactly. Vegetation returns are dropped: they are neither ground nor
//! something a platform must not fly into (the obstacle model already lets
//! platforms pass through foliage).

use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::runtime_world::SharedVisibility;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeScanConfig {
    /// Rays per ring around the platform.
    pub azimuth_count: usize,
    /// Depression angles of the rings, degrees below the horizon.
    pub ring_depressions_deg: Vec<f64>,
    /// Rays in the forward fan.
    pub forward_count: usize,
    /// The forward fan's depression, degrees below the horizon.
    pub forward_depression_deg: f64,
    /// Half-width of the forward fan about the heading, degrees.
    pub forward_half_angle_deg: f64,
    pub max_range_m: f64,
    /// One-sigma range noise, metres.
    pub range_noise_m: f64,
}

impl Default for RangeScanConfig {
    fn default() -> Self {
        Self {
            azimuth_count: 36,
            ring_depressions_deg: vec![80.0, 65.0, 50.0, 38.0, 28.0, 20.0],
            forward_count: 9,
            forward_depression_deg: 4.0,
            forward_half_angle_deg: 60.0,
            max_range_m: 400.0,
            range_noise_m: 0.3,
        }
    }
}

impl RangeScanConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.max_range_m.is_finite() && self.max_range_m > 0.0) {
            return Err(format!("max range {} m is not positive", self.max_range_m));
        }
        if !(self.range_noise_m.is_finite() && self.range_noise_m >= 0.0) {
            return Err(format!("range noise {} m is negative", self.range_noise_m));
        }
        if self.ring_depressions_deg.is_empty() && self.forward_count == 0 {
            return Err("the scan casts no rays".into());
        }
        Ok(())
    }

    /// Unit ray directions for a platform heading `heading_rad` (east = 0).
    pub fn directions(&self, heading_rad: f64) -> Vec<[f64; 3]> {
        let unit = |azimuth: f64, depression_deg: f64| {
            let d = depression_deg.to_radians();
            [d.cos() * azimuth.cos(), d.cos() * azimuth.sin(), -d.sin()]
        };
        let mut rays = vec![[0.0, 0.0, -1.0]];
        for depression in &self.ring_depressions_deg {
            for k in 0..self.azimuth_count {
                let azimuth =
                    heading_rad + std::f64::consts::TAU * k as f64 / self.azimuth_count as f64;
                rays.push(unit(azimuth, *depression));
            }
        }
        let half = self.forward_half_angle_deg.to_radians();
        for k in 0..self.forward_count {
            let offset = if self.forward_count == 1 {
                0.0
            } else {
                -half + 2.0 * half * k as f64 / (self.forward_count - 1) as f64
            };
            rays.push(unit(heading_rad + offset, self.forward_depression_deg));
        }
        rays
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReturnClass {
    Ground,
    Structure,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RangeReturn {
    pub point_m: [f64; 3],
    pub class: ReturnClass,
}

/// One platform's scan at one tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeScan {
    pub node_id: String,
    pub timestamp_s: f64,
    pub origin_m: [f64; 3],
    pub returns: Vec<RangeReturn>,
    /// Rays that met nothing within range (or left the world).
    pub misses: u32,
}

/// The scanner: a fan of rays over the true world.
#[derive(Debug, Clone)]
pub struct RangeScanner {
    config: RangeScanConfig,
    world: SharedVisibility,
}

impl RangeScanner {
    pub fn new(config: RangeScanConfig, world: SharedVisibility) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config, world })
    }

    pub fn config(&self) -> &RangeScanConfig {
        &self.config
    }

    /// Scan from `origin_m`, heading from `velocity` (east when slower than
    /// 0.5 m/s). Noise is keyed on the node and the tick, so a scan is the
    /// same however the run got to that tick.
    pub fn scan(
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
        let mut rng = DeterministicRng::keyed(
            0,
            &[
                SeedPart::Str(node_id),
                SeedPart::Str("range-scan"),
                SeedPart::Int(tick as i64),
            ],
        );
        let max_range = self.config.max_range_m;
        let mut returns = Vec::new();
        let mut misses = 0u32;
        for direction in self.config.directions(heading) {
            let end = [
                origin_m[0] + direction[0] * max_range,
                origin_m[1] + direction[1] * max_range,
                origin_m[2] + direction[2] * max_range,
            ];
            // Drawn for every ray, hit or not, so one ray's outcome never
            // shifts another's noise.
            let noise = rng.normal(0.0, self.config.range_noise_m.max(0.0));
            let Some((kind, point)) = self.world.first_hit(origin_m, end) else {
                misses += 1;
                continue;
            };
            let class = match kind.as_str() {
                "terrain" => ReturnClass::Ground,
                "building" | "wall" => ReturnClass::Structure,
                _ => {
                    misses += 1;
                    continue;
                }
            };
            // Range noise moves the point along the ray.
            returns.push(RangeReturn {
                point_m: std::array::from_fn(|k| point[k] + direction[k] * noise),
                class,
            });
        }
        RangeScan {
            node_id: node_id.to_string(),
            timestamp_s,
            origin_m,
            returns,
            misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_fan_points_down_and_forward() {
        let config = RangeScanConfig::default();
        let rays = config.directions(0.0);
        assert_eq!(rays.len(), 1 + 36 * 6 + 9);
        for ray in &rays {
            let norm = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
            assert!(ray[2] < 0.0, "every ray looks below the horizon");
        }
        // The forward fan's middle ray points along the heading.
        let middle = rays[1 + 216 + 4];
        assert!(middle[0] > 0.99 && middle[1].abs() < 1e-12);
    }

    #[test]
    fn an_empty_or_invalid_scan_is_refused() {
        let config = RangeScanConfig {
            max_range_m: 0.0,
            ..RangeScanConfig::default()
        };
        assert!(config.validate().is_err());
        let config = RangeScanConfig {
            ring_depressions_deg: Vec::new(),
            forward_count: 0,
            ..RangeScanConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
