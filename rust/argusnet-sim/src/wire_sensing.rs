//! Measured wire returns (W1): what a LiDAR and a camera actually get from a
//! power line, in place of the oracle that hands over the whole true span.
//!
//! By default a sensor near a conductor reports the span itself
//! ([`crate::range_scan::WireReturn`]): exact geometry and its true id, from
//! one sighting. With a [`WireSensorModel`] it reports only measurements:
//!
//! - **LiDAR points** on a conductor, with no id. A conductor is far thinner
//!   than the simulated ray spacing, so rays are not cast at it. The sim
//!   fires hundreds of rays a capture where a device fires hundreds of
//!   thousands, and almost none would ever land on a wire. Instead the
//!   expected returns from each short conductor piece are worked out from
//!   the device's point rate:
//!
//!   ```text
//!   λ = (points per capture / field of view) · (apparent length / r) · divergence · p
//!   ```
//!
//!   The first bracket is the rays per steradian; the next two are the solid
//!   angle within which a beam touches the piece. `p` is the chance the
//!   return clears the detector ([`WireSensorModel::return_strength`]).
//!   Return power falls with range squared, and a wire thinner than the
//!   footprint fills less of the beam as the footprint grows with range, so
//!   past that point a wire's return falls with range cubed. A detector
//!   either clears its noise floor or does not, so the chance falls steeply
//!   at the threshold, not gradually: about 100 m for a 3 cm conductor on a
//!   device quoted at 190 m. A wire seen end-on has little apparent length
//!   and earns few points. The hit count is a Poisson draw.
//! - **Camera lines**, with no id and no depth: two sight directions bounding
//!   the part of a span in view, each off by about a pixel. A single camera
//!   sees which way a wire lies, not how far away it is.
//!
//! Which sensor saw which span is never reported. Evaluation compares the
//! measurements with the true lines itself. Fitting them into a wire is W2.
//!
//! Every number is assumed until a device is chosen.

use serde::{Deserialize, Serialize};

use crate::deterministic::DeterministicRng;
use crate::nav_camera::EvidenceTag;
use crate::range_scan::SpanPieces;

/// What the wire sensors are declared to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireSensorModel {
    /// The LiDAR's point rate, returns per second.
    pub points_per_s: f64,
    /// Full beam divergence, radians.
    pub beam_divergence_rad: f64,
    /// The range at which the device just detects a 10%-reflective target
    /// that fills its beam, metres: the quoted range.
    pub reference_range_m: f64,
    /// A conductor's diffuse reflectivity, in `(0, 1]`.
    pub wire_reflectivity: f64,
    /// How sharply detection falls at the threshold: the exponent of the
    /// logistic in return power. 4 goes from 94% to 6% across a factor of
    /// four in power, or 6 dB.
    pub detection_steepness: f64,
    /// One-sigma range noise on a wire return, metres.
    pub range_noise_m: f64,
    /// One-sigma angular noise on a wire return, radians.
    pub angular_noise_rad: f64,
    /// Per-frame probability the camera picks out a wire within its pixel
    /// range: contrast against sky or ground is not modelled yet.
    pub camera_detection_probability: f64,
    /// Points a map cell needs before it counts as a wire hazard.
    pub min_hazard_points: u32,
    /// How far above and below its points a hazard reaches, metres.
    pub hazard_margin_m: f64,
    #[serde(default)]
    pub evidence: EvidenceTag,
}

impl Default for WireSensorModel {
    fn default() -> Self {
        Self {
            points_per_s: 240_000.0,
            beam_divergence_rad: 2.0e-3,
            reference_range_m: 190.0,
            wire_reflectivity: 0.2,
            detection_steepness: 4.0,
            range_noise_m: 0.05,
            angular_noise_rad: 5.0e-4,
            camera_detection_probability: 0.7,
            min_hazard_points: 2,
            hazard_margin_m: 5.0,
            evidence: EvidenceTag::Assumed,
        }
    }
}

impl WireSensorModel {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(format!("wire sensor {name} must be positive, got {v}"))
            }
        };
        let non_negative = |name: &str, v: f64| {
            if v.is_finite() && v >= 0.0 {
                Ok(())
            } else {
                Err(format!("wire sensor {name} must not be negative, got {v}"))
            }
        };
        positive("points_per_s", self.points_per_s)?;
        positive("beam_divergence_rad", self.beam_divergence_rad)?;
        positive("reference_range_m", self.reference_range_m)?;
        positive("wire_reflectivity", self.wire_reflectivity)?;
        positive("detection_steepness", self.detection_steepness)?;
        non_negative("range_noise_m", self.range_noise_m)?;
        non_negative("angular_noise_rad", self.angular_noise_rad)?;
        non_negative("hazard_margin_m", self.hazard_margin_m)?;
        if self.wire_reflectivity > 1.0 {
            return Err("wire sensor wire_reflectivity must be at most 1".into());
        }
        if !(self.camera_detection_probability >= 0.0 && self.camera_detection_probability <= 1.0) {
            return Err("wire sensor camera_detection_probability must be in [0, 1]".into());
        }
        if self.min_hazard_points == 0 {
            return Err("wire sensor min_hazard_points must be at least 1".into());
        }
        Ok(())
    }

    /// The chance a conductor of `radius_m` at `range_m` returns, in
    /// `[0, 1]`: 50% where its return power meets the device's threshold,
    /// falling steeply past it. Power goes with the reflectivity, the share
    /// of the footprint the wire fills and one over range squared; the
    /// threshold is what a 10%-reflective, beam-filling target returns at
    /// the reference range.
    pub fn return_strength(&self, radius_m: f64, range_m: f64) -> f64 {
        let range_m = range_m.max(1e-3);
        let footprint_m = range_m * self.beam_divergence_rad;
        let fill = (2.0 * radius_m / footprint_m).min(1.0);
        let power = self.wire_reflectivity * fill / (range_m * range_m);
        let threshold = 0.1 / (self.reference_range_m * self.reference_range_m);
        let snr = power / threshold;
        let rising = snr.powf(self.detection_steepness);
        rising / (1.0 + rising)
    }
}

/// Measured wire sensing as a sensor holds it: the model, and how long one
/// capture lasts, which sets how many LiDAR points it fires.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredWires {
    pub model: WireSensorModel,
    pub capture_s: f64,
}

/// One camera detection of a wire: the sight directions bounding the part
/// seen, from the frame's origin. No id, and no depth.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WireLine {
    pub from_dir: [f64; 3],
    pub to_dir: [f64; 3],
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// A Poisson count of mean `lambda`, by Knuth's product of uniforms: mean
/// values here are small, far under the point where that loses precision.
fn poisson(rng: &mut DeterministicRng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let limit = (-lambda.min(30.0)).exp();
    let mut product = 1.0;
    let mut count = 0;
    loop {
        product *= rng.uniform();
        if product <= limit {
            return count;
        }
        count += 1;
    }
}

/// The LiDAR's returns from `spans` for one capture from `origin_m`.
/// `field_of_view_sr` is the solid angle the points are spread over, and
/// `in_view` says whether a direction is in it. `clear` says whether a sight
/// line from the origin reaches a point unblocked.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lidar_points(
    spans: &[SpanPieces],
    model: &WireSensorModel,
    origin_m: [f64; 3],
    capture_s: f64,
    field_of_view_sr: f64,
    max_range_m: f64,
    in_view: impl Fn([f64; 3]) -> bool,
    clear: impl Fn([f64; 3]) -> bool,
    rng: &mut DeterministicRng,
) -> Vec<[f64; 3]> {
    let per_steradian = model.points_per_s * capture_s / field_of_view_sr.max(1e-9);
    let mut points = Vec::new();
    for span in spans {
        if span.outside_reach(origin_m, max_range_m) {
            continue;
        }
        let radius_m = span.radius_m();
        for piece in span.pieces() {
            let middle: [f64; 3] =
                std::array::from_fn(|k| (piece.start_m[k] + piece.end_m[k]) / 2.0);
            let to_middle = sub(middle, origin_m);
            let range_m = norm(to_middle);
            if range_m < 1e-6 || range_m > max_range_m || !in_view(to_middle) {
                continue;
            }
            let along = sub(piece.end_m, piece.start_m);
            let apparent_m = norm(cross(along, to_middle)) / range_m;
            let lambda = per_steradian
                * (apparent_m / range_m)
                * model.beam_divergence_rad
                * model.return_strength(radius_m, range_m);
            for _ in 0..poisson(rng, lambda) {
                let t = rng.uniform();
                let on_wire: [f64; 3] = std::array::from_fn(|k| piece.start_m[k] + along[k] * t);
                if !clear(on_wire) {
                    continue;
                }
                // Noise along the ray and across it, as the device reports.
                let ray = sub(on_wire, origin_m);
                let r = norm(ray);
                let along_ray = rng.normal(0.0, model.range_noise_m) / r;
                let sigma = model.angular_noise_rad * r;
                let across = [
                    rng.normal(0.0, sigma),
                    rng.normal(0.0, sigma),
                    rng.normal(0.0, sigma),
                ];
                points.push(std::array::from_fn(|k| {
                    on_wire[k] + ray[k] * along_ray + across[k]
                }));
            }
        }
    }
    points
}

/// The camera's line detections from `spans` for one frame from `origin_m`.
/// Each span's pieces within its detection range and `in_view` make one
/// line, kept with the model's detection probability.
pub(crate) fn camera_lines(
    spans: &[SpanPieces],
    model: &WireSensorModel,
    origin_m: [f64; 3],
    pixel_ifov_rad: f64,
    in_view: impl Fn([f64; 3]) -> bool,
    rng: &mut DeterministicRng,
) -> Vec<WireLine> {
    let unit_with_noise = |v: [f64; 3], rng: &mut DeterministicRng| -> [f64; 3] {
        let n = norm(v);
        let noisy: [f64; 3] = std::array::from_fn(|k| v[k] / n + rng.normal(0.0, pixel_ifov_rad));
        let m = norm(noisy);
        noisy.map(|x| x / m)
    };
    let mut lines = Vec::new();
    for span in spans {
        let range_m = span.range_m();
        if span.outside_reach(origin_m, range_m) {
            continue;
        }
        let seen: Vec<[f64; 3]> = span
            .pieces()
            .iter()
            .flat_map(|piece| [piece.start_m, piece.end_m])
            .map(|point| sub(point, origin_m))
            .filter(|d| norm(*d) <= range_m && norm(*d) > 1e-6 && in_view(*d))
            .collect();
        let (Some(first), Some(last)) = (seen.first().copied(), seen.last().copied()) else {
            continue;
        };
        // Drawn for every span in view, so one's outcome never shifts
        // another's noise.
        let detected = rng.uniform() < model.camera_detection_probability;
        let from_dir = unit_with_noise(first, rng);
        let to_dir = unit_with_noise(last, rng);
        if detected {
            lines.push(WireLine { from_dir, to_dir });
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic::SeedPart;
    use crate::range_scan::span_pieces;
    use argusnet_world::powerline::{closest_on_segment, PowerLine, Support};

    /// A 300 m span along y at x = 0, 50 m up, no sag.
    fn line(radius_m: f64) -> PowerLine {
        let support = |y: f64| Support {
            xy_m: [0.0, y],
            base_z_m: 0.0,
            height_m: 50.0,
        };
        PowerLine {
            id: "line".into(),
            supports: vec![support(-150.0), support(150.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.0,
            conductor_radius_m: radius_m,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap()
    }

    fn rng(tick: i64) -> DeterministicRng {
        DeterministicRng::keyed(0, &[SeedPart::Str("wire-test"), SeedPart::Int(tick)])
    }

    /// Mean returns per capture over many captures from `standoff_m`.
    fn mean_points(radius_m: f64, standoff_m: f64) -> f64 {
        let lines = [line(radius_m)];
        let spans = span_pieces(&lines, |_| 1000.0);
        let model = WireSensorModel::default();
        let total: usize = (0..200)
            .map(|k| {
                lidar_points(
                    &spans,
                    &model,
                    [standoff_m, 0.0, 50.0],
                    0.1,
                    1.0,
                    1000.0,
                    |_| true,
                    |_| true,
                    &mut rng(k),
                )
                .len()
            })
            .sum();
        total as f64 / 200.0
    }

    #[test]
    fn returns_fade_with_range_and_with_thinner_wires() {
        let near = mean_points(0.015, 40.0);
        let edge = mean_points(0.015, 110.0);
        let far = mean_points(0.015, 240.0);
        let thin_edge = mean_points(0.005, 110.0);
        assert!(near > 3.0 * edge, "near {near} against edge {edge}");
        assert!(
            edge > 2.0 * thin_edge,
            "edge {edge} against thin {thin_edge}"
        );
        assert!(far < 0.01 * edge, "a wire at 240 m all but vanishes: {far}");
    }

    /// A 3 cm conductor at 20% reflectivity on a device quoted at 190 m:
    /// near certain at 50 m, an even chance around 100 m, gone by 200 m,
    /// long before the quoted range for a surface. A thinner wire goes
    /// sooner.
    #[test]
    fn detection_falls_away_at_the_threshold_near_100_m() {
        let model = WireSensorModel::default();
        assert!(model.return_strength(0.015, 50.0) > 0.99);
        let at_100 = model.return_strength(0.015, 100.0);
        assert!((0.4..0.7).contains(&at_100), "{at_100}");
        assert!(model.return_strength(0.015, 200.0) < 0.001);
        assert!(model.return_strength(0.005, 100.0) < 0.1, "a thin wire");
        // A wider target fills the beam: the same power falls with r².
        assert!(model.return_strength(1.0, 150.0) > 0.99);
    }

    #[test]
    fn a_wire_seen_end_on_earns_few_points() {
        let lines = [line(0.015)];
        let spans = span_pieces(&lines, |_| 1000.0);
        let model = WireSensorModel::default();
        let count = |origin: [f64; 3]| -> usize {
            (0..100)
                .map(|k| {
                    lidar_points(
                        &spans,
                        &model,
                        origin,
                        0.1,
                        1.0,
                        1000.0,
                        |_| true,
                        |_| true,
                        &mut rng(k),
                    )
                    .len()
                })
                .sum()
        };
        let abeam = count([100.0, 0.0, 50.0]);
        let end_on = count([0.0, 400.0, 50.0]);
        assert!(abeam > 10 * end_on.max(1), "abeam {abeam}, end-on {end_on}");
    }

    #[test]
    fn points_land_on_the_wire_within_the_noise_and_blocked_ones_are_dropped() {
        let lines = [line(0.015)];
        let spans = span_pieces(&lines, |_| 1000.0);
        let model = WireSensorModel::default();
        let origin = [60.0, 0.0, 50.0];
        let points: Vec<[f64; 3]> = (0..50)
            .flat_map(|k| {
                lidar_points(
                    &spans,
                    &model,
                    origin,
                    0.1,
                    1.0,
                    1000.0,
                    |_| true,
                    |_| true,
                    &mut rng(k),
                )
            })
            .collect();
        assert!(!points.is_empty());
        let (a, b) = ([0.0, -150.0, 50.0], [0.0, 150.0, 50.0]);
        for p in &points {
            let on = closest_on_segment(*p, a, b);
            assert!(norm(sub(*p, on)) < 0.5, "{p:?}");
        }
        let blocked = lidar_points(
            &spans,
            &model,
            origin,
            0.1,
            1.0,
            1000.0,
            |_| true,
            |_| false,
            &mut rng(1),
        );
        assert!(blocked.is_empty());
    }

    #[test]
    fn a_camera_line_has_direction_but_no_depth_or_id() {
        let lines = [line(0.015)];
        let spans = span_pieces(&lines, |_| 25.0);
        let model = WireSensorModel {
            camera_detection_probability: 1.0,
            ..WireSensorModel::default()
        };
        let frame = |origin: [f64; 3]| {
            camera_lines(
                &spans,
                &model,
                origin,
                8.0e-4,
                |d: [f64; 3]| d[0] < 0.0,
                &mut rng(3),
            )
        };
        // 20 m east of the wire, looking west: in range and in view.
        let seen = frame([20.0, 0.0, 50.0]);
        assert_eq!(seen.len(), 1);
        for dir in [seen[0].from_dir, seen[0].to_dir] {
            assert!((norm(dir) - 1.0).abs() < 1e-9, "a direction, not a point");
            assert!(dir[0] < 0.0, "toward the wire");
        }
        assert_ne!(seen[0].from_dir, seen[0].to_dir, "the part seen has extent");
        // Out of pixel range, nothing.
        assert!(frame([40.0, 0.0, 50.0]).is_empty());
    }

    #[test]
    fn invalid_models_are_refused() {
        assert!(WireSensorModel::default().validate().is_ok());
        let bad = [
            WireSensorModel {
                points_per_s: 0.0,
                ..WireSensorModel::default()
            },
            WireSensorModel {
                wire_reflectivity: 1.5,
                ..WireSensorModel::default()
            },
            WireSensorModel {
                camera_detection_probability: 2.0,
                ..WireSensorModel::default()
            },
            WireSensorModel {
                min_hazard_points: 0,
                ..WireSensorModel::default()
            },
        ];
        for model in bad {
            assert!(model.validate().is_err(), "{model:?}");
        }
    }
}
