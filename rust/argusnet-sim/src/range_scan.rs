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
//! - **Structure**: it met a building, a wall or a tree's trunk.
//! - **Canopy**: it was stopped by leaves in a tree's crown (R2): soft
//!   structure, something to keep clear of but not a wall.
//!
//! The class models ground segmentation on the scanner's point cloud, which a
//! real system computes from the cloud itself. This model takes it from the
//! world exactly. A crown returns with the probability its leaves stop the
//! ray (one minus its transmittance, [`SharedVisibility::first_return`]), so
//! some rays reach the ground under a tree and some come back from its top.
//! Forest stands' returns are still dropped: a stand is an aggregate with no
//! surface to return from.
//!
//! **Wires** (R3) are seen differently. A conductor a few centimetres thick
//! does not stop the fan's rays (wires never occlude: the ground behind
//! still returns), and a scanner puts points on it only while its points
//! are no further apart than the wire is thick: within
//! [`wire_detection_range_m`], about 30 m for a 3 cm wire at the scanner's
//! [`WIRE_RESOLUTION_RAD`]. A span with a conductor that near, below the
//! scanner's horizon (the fan looks down), is reported as a [`WireReturn`]:
//! the line, the span, and the nearest point. Which span the points belong
//! to is taken from the world exactly, as the ground class is; a real
//! system fits it from the points. So a platform without a prior map learns
//! a wire only when it is already close.

use std::sync::Arc;

use argusnet_world::powerline::{segment_distance, ConductorPiece, PowerLine};
use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::runtime_world::SharedVisibility;

/// The scanner's angular resolution for wires, radians: how far apart its
/// points are at a metre's range.
pub const WIRE_RESOLUTION_RAD: f64 = 1e-3;

/// How near a conductor of `radius_m` must be for the scanner's points to
/// land on it: where their spacing grows to the wire's thickness.
pub fn wire_detection_range_m(radius_m: f64) -> f64 {
    2.0 * radius_m / WIRE_RESOLUTION_RAD
}

/// A span of a power line the scanner put points on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireReturn {
    /// The span, as fitted to its points: [`PowerLine::span_line`], whose id
    /// names the line and the span (`line#k`).
    pub span: PowerLine,
    /// The nearest point of its conductors.
    pub point_m: [f64; 3],
}

/// One span's conductor pieces, and the box holding them.
#[derive(Debug, Clone)]
pub(crate) struct SpanPieces {
    span: PowerLine,
    pieces: Vec<ConductorPiece>,
    range_m: f64,
    low_m: [f64; 3],
    high_m: [f64; 3],
}

/// `lines`' spans, each detectable within `range_m(line)` of a sensor.
pub(crate) fn span_pieces(
    lines: &[PowerLine],
    range_m: impl Fn(&PowerLine) -> f64,
) -> Vec<SpanPieces> {
    let mut spans = Vec::new();
    for line in lines {
        let range_m = range_m(line);
        let per_span = line.pieces_by_span();
        for (span, pieces) in per_span.into_iter().enumerate() {
            let mut low_m = [f64::INFINITY; 3];
            let mut high_m = [f64::NEG_INFINITY; 3];
            for piece in &pieces {
                for axis in 0..3 {
                    low_m[axis] = low_m[axis]
                        .min(piece.start_m[axis].min(piece.end_m[axis]) - piece.sag_error_m);
                    high_m[axis] = high_m[axis]
                        .max(piece.start_m[axis].max(piece.end_m[axis]) + piece.sag_error_m);
                }
            }
            spans.push(SpanPieces {
                span: line.span_line(span),
                pieces,
                range_m,
                low_m,
                high_m,
            });
        }
    }
    spans
}

/// The spans of `spans` with a conductor within detection range of
/// `origin_m` whose nearest point `sees` admits, each with that point (in
/// line, then span order). The nearest point is on the pieces, which the
/// wire hangs within millimetres of.
pub(crate) fn wires_seen(
    spans: &[SpanPieces],
    origin_m: [f64; 3],
    sees: impl Fn([f64; 3]) -> bool,
) -> Vec<WireReturn> {
    let mut wires = Vec::new();
    for span in spans {
        let outside = (0..3).any(|axis| {
            origin_m[axis] < span.low_m[axis] - span.range_m
                || origin_m[axis] > span.high_m[axis] + span.range_m
        });
        if outside {
            continue;
        }
        let nearest = span
            .pieces
            .iter()
            .map(|piece| {
                let point = argusnet_world::powerline::closest_on_segment(
                    origin_m,
                    piece.start_m,
                    piece.end_m,
                );
                (segment_distance(origin_m, origin_m, point, point), point)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((distance, point)) = nearest {
            if distance <= span.range_m && sees(point) {
                wires.push(WireReturn {
                    span: span.span.clone(),
                    point_m: point,
                });
            }
        }
    }
    wires
}

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
    Canopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RangeReturn {
    pub point_m: [f64; 3],
    pub class: ReturnClass,
}

/// Which sensor made a [`RangeScan`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanSensor {
    /// The mapping payload's range scanner: evidence for the map.
    #[default]
    RangeScanner,
    /// The body-fixed forward navigation camera
    /// ([`crate::nav_camera`]): evidence for obstacle avoidance only.
    NavCamera,
}

impl ScanSensor {
    pub fn is_range_scanner(&self) -> bool {
        *self == Self::RangeScanner
    }
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
    /// The power-line spans near enough to put points on (R3). Absent when
    /// there are none, so a scan in a world without lines is unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wires: Vec<WireReturn>,
    /// The sensor: the range scanner unless written. A navigation camera's
    /// frame never counts as mapping evidence.
    #[serde(default, skip_serializing_if = "ScanSensor::is_range_scanner")]
    pub sensor: ScanSensor,
}

/// The scanner: a fan of rays over the true world.
#[derive(Debug, Clone)]
pub struct RangeScanner {
    config: RangeScanConfig,
    world: SharedVisibility,
    /// The world's power lines, span by span (R3).
    spans: Arc<Vec<SpanPieces>>,
}

impl RangeScanner {
    pub fn new(config: RangeScanConfig, world: SharedVisibility) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            world,
            spans: Arc::new(Vec::new()),
        })
    }

    /// The scanner, also detecting `lines`' wires (R3).
    pub fn with_power_lines(mut self, lines: &[PowerLine]) -> Self {
        // No further than the scanner reaches at all.
        let max_range_m = self.config.max_range_m;
        self.spans = Arc::new(span_pieces(lines, |line| {
            wire_detection_range_m(line.conductor_radius_m).min(max_range_m)
        }));
        self
    }

    /// How many power-line spans the scanner can detect.
    pub fn wire_span_count(&self) -> usize {
        self.spans.len()
    }

    /// The spans with a conductor within detection range of `origin_m` and
    /// below its horizon (the fan looks down).
    fn wires_near(&self, origin_m: [f64; 3]) -> Vec<WireReturn> {
        wires_seen(&self.spans, origin_m, |point| point[2] <= origin_m[2])
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
        // Crowns draw from a stream of their own, keyed the same way, so a
        // world without trees draws exactly what it drew before them.
        let mut canopy = DeterministicRng::keyed(
            0,
            &[
                SeedPart::Str(node_id),
                SeedPart::Str("range-scan-canopy"),
                SeedPart::Int(tick as i64),
            ],
        );
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
            let Some((kind, point)) = self
                .world
                .first_return(origin_m, end, &mut || canopy.uniform())
            else {
                misses += 1;
                continue;
            };
            let class = match kind.as_str() {
                "terrain" => ReturnClass::Ground,
                "building" | "wall" | "trunk" => ReturnClass::Structure,
                "canopy" => ReturnClass::Canopy,
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
            wires: self.wires_near(origin_m),
            sensor: ScanSensor::RangeScanner,
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

    /// A scan over a world with trees: a trunk returns as structure, a
    /// crown as canopy, each crown's draw from its own stream keyed by node
    /// and tick.
    #[test]
    fn trees_return_as_structure_and_canopy_from_their_own_stream() {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::Bounds2d;
        use argusnet_world::tiled::TiledTerrain;
        use argusnet_world::trees::{Species, Tree, TreeLayer, CROWN_EXTINCTION_PER_M};
        use argusnet_world::visibility::VisibilityProfile;
        use std::sync::Arc;
        let bounds = Bounds2d::new(0.0, 200.0, 0.0, 200.0);
        let terrain = Arc::new(
            TiledTerrain::from_grid(bounds, &vec![vec![0.0; 21]; 21], 10.0, 4, 0.0, &[]).unwrap(),
        );
        let world = SharedVisibility::new(
            terrain,
            Arc::new(ObstacleLayer::new(bounds, 50.0, vec![]).unwrap()),
            None,
            Arc::new(VisibilityProfile::default()),
            0.0,
        );
        let tree = |id: &str, x: f64, density: f64| Tree {
            id: id.into(),
            xy_m: [x, 100.0],
            base_z_m: 0.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 5.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: density,
        };
        let world = world.with_trees(Arc::new(
            TreeLayer::new(vec![tree("trunk", 60.0, 0.0), tree("crown", 102.0, 1.0)]).unwrap(),
        ));
        // Level ahead at 5 m: the trunk. Straight down 2 m off the crown's
        // axis: canopy (a crown of density 1 over 11 m stops 98 % of rays),
        // or, in the 2 % of draws past it, the ground.
        let level = RangeScanConfig {
            ring_depressions_deg: Vec::new(),
            forward_count: 1,
            forward_depression_deg: 0.0,
            ..RangeScanConfig::default()
        };
        let scanner = RangeScanner::new(level, world.clone()).unwrap();
        let scan = scanner.scan("d", 3, 0.75, [20.0, 100.0, 5.0], [10.0, 0.0, 0.0]);
        assert_eq!(scan.returns.len(), 2, "{scan:?}");
        assert_eq!(scan.returns[1].class, ReturnClass::Structure);
        // (The scan needs a ray besides the nadir: a second straight down.)
        let down = RangeScanConfig {
            ring_depressions_deg: Vec::new(),
            forward_count: 1,
            forward_depression_deg: 90.0,
            range_noise_m: 0.0,
            ..RangeScanConfig::default()
        };
        let scanner = RangeScanner::new(down, world).unwrap();
        let scan = scanner.scan("d", 3, 0.75, [100.0, 100.0, 60.0], [0.0; 3]);
        let mut canopy = DeterministicRng::keyed(
            0,
            &[
                SeedPart::Str("d"),
                SeedPart::Str("range-scan-canopy"),
                SeedPart::Int(3),
            ],
        );
        let u = canopy.uniform();
        let top = 14.0 + 6.0 * (1.0_f64 - 0.16).sqrt();
        let expected = top - (-(1.0 - u).ln() / CROWN_EXTINCTION_PER_M);
        let hit = scan.returns[0];
        if expected > 14.0 - 6.0 * (1.0_f64 - 0.16).sqrt() {
            assert_eq!(hit.class, ReturnClass::Canopy, "{scan:?}");
            assert!(
                (hit.point_m[2] - expected).abs() < 1e-9,
                "{hit:?} vs {expected}"
            );
        } else {
            assert_eq!(hit.class, ReturnClass::Ground, "{scan:?}");
        }
    }

    /// Flat ground at 0 over 400 m square, nothing on it, and a scanner
    /// that knows `lines`.
    fn scanner_over_flat(lines: &[PowerLine]) -> RangeScanner {
        use argusnet_types::Bounds2d;
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::tiled::{Tile, TiledTerrain};
        use argusnet_world::visibility::VisibilityProfile;
        let bounds = Bounds2d::new(-200.0, 200.0, -200.0, 200.0);
        let tile = Tile {
            tx: 0,
            ty: 0,
            lod: 0,
            x_min_m: -200.0,
            y_min_m: -200.0,
            cell_size_m: 5.0,
            heights_m: vec![vec![0.0; 81]; 81],
            min_pyramid: Vec::new(),
            max_pyramid: Vec::new(),
        };
        let world = SharedVisibility::new(
            Arc::new(TiledTerrain::new(bounds, 80, 5.0, 0.0, vec![tile]).unwrap()),
            Arc::new(ObstacleLayer::new(bounds, 400.0, Vec::new()).unwrap()),
            None,
            Arc::new(VisibilityProfile::default()),
            1.0,
        );
        RangeScanner::new(RangeScanConfig::default(), world)
            .unwrap()
            .with_power_lines(lines)
    }

    /// A 3 cm single wire 20 m up, along x from -100 to 100, sagging 3 %:
    /// two spans, the support at x = 0.
    fn wire() -> PowerLine {
        use argusnet_world::powerline::Support;
        let support = |x: f64| Support {
            xy_m: [x, 0.0],
            base_z_m: 0.0,
            height_m: 20.0,
        };
        PowerLine {
            id: "line".into(),
            supports: vec![support(-100.0), support(0.0), support(100.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.03,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap()
    }

    #[test]
    fn a_three_centimetre_wire_is_resolved_out_to_thirty_metres() {
        assert!((wire_detection_range_m(0.015) - 30.0).abs() < 1e-9);
    }

    /// A span is reported while a conductor is within detection range below
    /// the scanner, with its nearest point on the pieces; not beyond the
    /// range, and not above the scanner, whose fan looks down.
    #[test]
    fn a_wire_is_detected_only_near_and_below() {
        let scanner = scanner_over_flat(&[wire()]);
        let scan = |origin: [f64; 3]| scanner.scan("d", 0, 0.0, origin, [0.0; 3]);
        // Over mid-span of the second span, 25 m over the wire's lowest
        // point (17 m): seen, its nearest point just off straight below
        // (the pieces either side tilt toward the scanner).
        let seen = scan([50.0, 0.0, 42.0]);
        assert_eq!(seen.wires.len(), 1, "{:?}", seen.wires);
        assert_eq!(seen.wires[0].span.id, "line#1");
        let point = seen.wires[0].point_m;
        assert!(
            (point[0] - 50.0).abs() < 0.2 && (point[2] - 17.0).abs() < 1e-3,
            "{point:?}"
        );
        let near = segment_distance([50.0, 0.0, 42.0], [50.0, 0.0, 42.0], point, point);
        assert!(near <= 25.0, "{near}");
        // The fitted span is the line's second span alone.
        assert_eq!(seen.wires[0].span, wire().span_line(1));
        // 31 m over it: too far for the points to land on a 3 cm wire.
        assert!(scan([50.0, 0.0, 48.0]).wires.is_empty());
        // Beside the line, 29 m off, level with it: below the horizon by a
        // hair, and seen; 1 m higher than the wire's top seen too.
        assert_eq!(scan([50.0, 29.0, 17.5]).wires.len(), 1);
        // Past the line's end, 20 m out along it and 10 m up: the end span
        // is seen, from outside the box its pieces fill.
        let past = scan([-120.0, 0.0, 30.0]);
        let ids: Vec<&str> = past.wires.iter().map(|w| w.span.id.as_str()).collect();
        assert_eq!(ids, ["line#0"]);
        // A scanner that reaches only 10 m sees the wire only that near.
        let short = RangeScanner {
            config: RangeScanConfig {
                max_range_m: 10.0,
                ..RangeScanConfig::default()
            },
            ..scanner_over_flat(&[])
        }
        .with_power_lines(&[wire()]);
        let short_scan = |origin: [f64; 3]| short.scan("d", 0, 0.0, origin, [0.0; 3]).wires.len();
        assert_eq!(short_scan([50.0, 0.0, 28.0]), 0, "20 m over the wire");
        assert_eq!(short_scan([50.0, 0.0, 26.5]), 1, "9.5 m over the wire");
        // Under the wire, looking down: not seen.
        assert!(scan([50.0, 0.0, 10.0]).wires.is_empty());
        // Over the support, both spans are near.
        let both = scan([0.0, 0.0, 30.0]);
        let ids: Vec<&str> = both.wires.iter().map(|w| w.span.id.as_str()).collect();
        assert_eq!(ids, ["line#0", "line#1"]);
    }

    /// Wires never stop a ray: the ground under one still returns, exactly
    /// as with no line at all.
    #[test]
    fn a_wire_does_not_occlude_the_ground() {
        let with = scanner_over_flat(&[wire()]).scan("d", 3, 0.0, [50.0, 0.0, 42.0], [0.0; 3]);
        let without = scanner_over_flat(&[]).scan("d", 3, 0.0, [50.0, 0.0, 42.0], [0.0; 3]);
        assert_eq!(with.returns, without.returns);
        assert_eq!(with.misses, without.misses);
        assert!(without.wires.is_empty() && !with.wires.is_empty());
        // The nadir ray lands under the wire.
        assert!(with.returns.iter().any(|r| r.class == ReturnClass::Ground
            && (r.point_m[0] - 50.0).abs() < 1.5
            && r.point_m[1].abs() < 1.5));
    }
}
