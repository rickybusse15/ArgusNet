//! Wire tracking (W2): fit measured points and camera lines into wires, each
//! with what is known about it and how well.
//!
//! The sensors report points on conductors and lines in images, and nothing
//! says which wire any of them is on (`argusnet_sim::wire_sensing`). This
//! turns them into hypotheses:
//!
//! - **A track is one conductor over one span**: a straight line across the
//!   ground, `d(s) = e + f s`, and a parabola over it, `z(s) = a s² + b s + c`.
//!   For the sag of a real span a parabola is a catenary to within a
//!   millimetre (`argusnet_world::powerline`).
//! - **Evidence is bounded.** Points are kept as means in [`BIN_M`] bins along
//!   the wire, so a wire scanned for an hour costs what one scanned for a
//!   minute does, and the fit is a weighted least squares over the bins.
//! - **Uncertainty is the fit's, and its misfit.** The parameter covariance
//!   gives errors across the line and in height at any point along it, small
//!   where the wire was seen and growing past the ends of what was. That
//!   alone shrinks with every point, so a fit that does not follow the wire
//!   would claim to know it to millimetres: the residuals are added, so a bad
//!   fit says so. The curvature has a prior around zero, so a short track does
//!   not invent a sag.
//! - **New wires are seeded** by RANSAC over the points no track claimed. The
//!   draws come from the tracker's own checkpointed generator, so a run is
//!   deterministic and resumes exactly.
//! - **Conductors stay apart.** A line's conductors a couple of metres apart
//!   fall outside each other's gates and are tracked separately.
//! - **Camera lines** have no depth. A line agreeing with a track supports it.
//!   Two lines from different places, whose planes cross along a level
//!   direction, triangulate a new track, whose uncertainty is that of a
//!   camera's depth, not a LiDAR's.
//!
//! A track is **confirmed** once it has enough points over enough length, or
//! came from a triangulation that a further line agrees with: two lines on
//! two conductors also cross, so two alone prove nothing. Only confirmed
//! tracks are hazards.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

/// How long a bin of evidence is along the wire, metres.
pub const BIN_M: f64 = 5.0;

/// What the tracker is declared to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireTrackerConfig {
    /// A point is a track's within this across the line, metres.
    pub lateral_gate_m: f64,
    /// A point is a track's within this in height, metres.
    pub vertical_gate_m: f64,
    /// A track reaches for points this far past its ends, metres.
    pub extension_gate_m: f64,
    /// Points a track needs to be confirmed.
    pub confirm_points: u32,
    /// Length of wire a track needs to be confirmed, metres.
    pub confirm_length_m: f64,
    /// Unclaimed points kept for seeding.
    pub pending_capacity: usize,
    /// Fewest unclaimed points worth a seeding pass.
    pub seed_min_points: usize,
    /// RANSAC draws per seeding pass.
    pub seed_draws: usize,
    /// Smallest one-sigma error a track claims, metres: the sensor's noise,
    /// and the attitude and boresight errors a fit cannot see.
    pub floor_sigma_m: f64,
    /// The prior on curvature, per metre: a sag ratio of 3% over a 100 m span
    /// is about 1.2e-3.
    pub curvature_prior: f64,
    /// A track's reported hazard reaches this far past what was seen, metres.
    pub extend_m: f64,
    /// The most error a track's hazard claims, metres: past that a track
    /// says nothing useful, and a huge clearance would close off the map.
    pub max_sigma_m: f64,
    /// Sigmas of uncertainty a hazard's clearance is grown by.
    pub envelope_sigmas: f64,
    /// A camera line supports a track within this angle, radians.
    pub camera_gate_rad: f64,
    /// Two tracks join only if their joint fit is this good, metres: the
    /// corner at a support shows as misfit.
    pub merge_sigma_m: f64,
    /// Unclaimed camera lines kept for triangulation.
    pub line_capacity: usize,
    /// Two camera lines triangulate only from this far apart, metres.
    pub min_baseline_m: f64,
    /// A triangulation further than this from either view is refused,
    /// metres: nearly parallel planes cross anywhere.
    pub max_triangulation_range_m: f64,
    /// The share of a seed's chord its points must cover: a wire's points
    /// run along it, where a shallow chord across two conductors collects
    /// points only where it crosses each.
    pub seed_coverage: f64,
    /// One-sigma error of a triangulated track, metres.
    pub triangulated_sigma_m: f64,
    /// The clearance a wire of unknown voltage is given, metres.
    pub clearance_m: f64,
    /// The conductor radius assumed for a fitted wire, metres.
    pub assumed_radius_m: f64,
}

impl Default for WireTrackerConfig {
    fn default() -> Self {
        Self {
            lateral_gate_m: 0.75,
            vertical_gate_m: 1.0,
            extension_gate_m: 60.0,
            confirm_points: 30,
            confirm_length_m: 30.0,
            pending_capacity: 2048,
            seed_min_points: 24,
            seed_draws: 64,
            floor_sigma_m: 0.15,
            curvature_prior: 2.0e-3,
            extend_m: 20.0,
            max_sigma_m: 3.0,
            envelope_sigmas: 3.0,
            camera_gate_rad: 0.02,
            merge_sigma_m: 0.3,
            line_capacity: 64,
            min_baseline_m: 5.0,
            max_triangulation_range_m: 60.0,
            seed_coverage: 0.6,
            triangulated_sigma_m: 2.0,
            clearance_m: 10.0,
            assumed_radius_m: 0.015,
        }
    }
}

impl WireTrackerConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("lateral_gate_m", self.lateral_gate_m),
            ("vertical_gate_m", self.vertical_gate_m),
            ("confirm_length_m", self.confirm_length_m),
            ("floor_sigma_m", self.floor_sigma_m),
            ("curvature_prior", self.curvature_prior),
            ("camera_gate_rad", self.camera_gate_rad),
            ("min_baseline_m", self.min_baseline_m),
            ("max_triangulation_range_m", self.max_triangulation_range_m),
            ("seed_coverage", self.seed_coverage),
            ("triangulated_sigma_m", self.triangulated_sigma_m),
            ("max_sigma_m", self.max_sigma_m),
            ("merge_sigma_m", self.merge_sigma_m),
            ("clearance_m", self.clearance_m),
            ("assumed_radius_m", self.assumed_radius_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("wire tracker {name} must be positive, got {value}"));
            }
        }
        for (name, value) in [
            ("extension_gate_m", self.extension_gate_m),
            ("extend_m", self.extend_m),
            ("envelope_sigmas", self.envelope_sigmas),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!(
                    "wire tracker {name} must not be negative, got {value}"
                ));
            }
        }
        if self.confirm_points == 0 || self.seed_min_points < 2 || self.seed_draws == 0 {
            return Err("wire tracker confirm and seed counts are too small".into());
        }
        if self.pending_capacity < self.seed_min_points {
            return Err("wire tracker pending_capacity is below seed_min_points".into());
        }
        Ok(())
    }
}

/// A bin's accumulated points.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct Bin {
    n: u32,
    sum: [f64; 3],
}

impl Bin {
    fn mean(&self) -> [f64; 3] {
        self.sum.map(|v| v / self.n as f64)
    }
}

/// A track's fit: line and parabola in its own frame, with their inverse
/// normal matrices, which scale to covariances by the residual variances.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct Fit {
    line: [f64; 2],
    parabola: [f64; 3],
    line_inverse: [[f64; 2]; 2],
    parabola_inverse: [[f64; 3]; 3],
    sigma_lateral_m: f64,
    sigma_vertical_m: f64,
    s_min_m: f64,
    s_max_m: f64,
    points: u32,
}

/// One conductor over one span, as far as the evidence goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireTrack {
    id: u32,
    origin_m: [f64; 2],
    direction: [f64; 2],
    bins: BTreeMap<i32, Bin>,
    fit: Fit,
    /// The least error this track claims.
    floor_sigma_m: f64,
    camera_support: u32,
    triangulated: bool,
    /// Where a pole ends this track on each side (W3), as (pole, distance
    /// along): the wire has a corner there, so nothing past it is this
    /// track's.
    #[serde(default, skip_serializing_if = "both_none")]
    ends: [Option<TrackEnd>; 2],
}

/// A pole a track ends at.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct TrackEnd {
    pole: u32,
    s_m: f64,
}

fn both_none(ends: &[Option<TrackEnd>; 2]) -> bool {
    ends[0].is_none() && ends[1].is_none()
}

/// A track's estimate, as planners read it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireEstimate {
    pub id: u32,
    /// The ends of the hazard: what was seen, extended, on the fitted curve.
    pub from_m: [f64; 3],
    pub to_m: [f64; 3],
    /// How far the wire hangs below its chord at mid-span, metres.
    pub sag_m: f64,
    /// Worst one-sigma error along the hazard, across the line and in height.
    pub sigma_lateral_m: f64,
    pub sigma_vertical_m: f64,
    /// Length of wire with evidence, metres.
    pub observed_length_m: f64,
    pub points: u32,
    pub camera_support: u32,
    pub triangulated: bool,
    pub confirmed: bool,
    /// The pole at each end, when one was found (W3).
    #[serde(default)]
    pub pole_ends: [Option<u32>; 2],
}

/// `value` of `x` under the 2-vector `row`.
fn quadratic_2(inverse: &[[f64; 2]; 2], row: [f64; 2]) -> f64 {
    let mut total = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            total += row[i] * inverse[i][j] * row[j];
        }
    }
    total
}

fn quadratic_3(inverse: &[[f64; 3]; 3], row: [f64; 3]) -> f64 {
    let mut total = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            total += row[i] * inverse[i][j] * row[j];
        }
    }
    total
}

/// The inverse of a symmetric 3×3 matrix, `None` when it is singular.
fn invert_3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ];
    let det = m[0][0] * c[0][0] + m[0][1] * c[1][0] + m[0][2] * c[2][0];
    if det.abs() < 1e-18 {
        return None;
    }
    Some(c.map(|row| row.map(|v| v / det)))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

impl WireTrack {
    fn new(id: u32, origin_m: [f64; 2], direction: [f64; 2], floor_sigma_m: f64) -> Self {
        Self {
            id,
            origin_m,
            direction,
            bins: BTreeMap::new(),
            fit: Fit::default(),
            floor_sigma_m,
            camera_support: 0,
            triangulated: false,
            ends: [None, None],
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    /// Distance along the track and across it of `(x, y)`.
    fn frame(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let (dx, dy) = (x_m - self.origin_m[0], y_m - self.origin_m[1]);
        (
            dx * self.direction[0] + dy * self.direction[1],
            dy * self.direction[0] - dx * self.direction[1],
        )
    }

    fn add(&mut self, point: [f64; 3]) {
        let (s, _) = self.frame(point[0], point[1]);
        let bin = self.bins.entry((s / BIN_M).floor() as i32).or_default();
        bin.n += 1;
        for (sum, value) in bin.sum.iter_mut().zip(point) {
            *sum += value;
        }
    }

    /// Weighted least squares over the bins. The curvature's prior is a
    /// measurement of zero with `curvature_prior` of error.
    fn refit(&mut self, config: &WireTrackerConfig) {
        let bins: Vec<(f64, f64, f64, f64)> = self
            .bins
            .values()
            .map(|bin| {
                let [x, y, z] = bin.mean();
                let (s, d) = self.frame(x, y);
                (s, d, z, f64::from(bin.n))
            })
            .collect();
        let floor = self.floor_sigma_m;
        let weight: f64 = bins.iter().map(|b| b.3).sum();
        let (s_min, s_max) = bins.iter().fold((f64::MAX, f64::MIN), |(lo, hi), b| {
            (lo.min(b.0), hi.max(b.0))
        });
        // Line: d = e + f s.
        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        // Parabola: z = a s² + b s + c.
        let mut m = [[0.0; 3]; 3];
        let mut v = [0.0; 3];
        for &(s, d, z, w) in &bins {
            a11 += w;
            a12 += w * s;
            a22 += w * s * s;
            b1 += w * d;
            b2 += w * s * d;
            let x = [s * s, s, 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += w * x[i] * x[j];
                }
                v[i] += w * x[i] * z;
            }
        }
        // Slope and height are weakly held too, so a track of one bin stays
        // solvable: it says little, and its covariance shows that.
        a22 += 1e-6;
        m[1][1] += 1e-6;
        let ridge = (floor / config.curvature_prior).powi(2);
        m[0][0] += ridge;
        let det = a11 * a22 - a12 * a12;
        let line_inverse = [[a22 / det, -a12 / det], [-a12 / det, a11 / det]];
        let line = [
            line_inverse[0][0] * b1 + line_inverse[0][1] * b2,
            line_inverse[1][0] * b1 + line_inverse[1][1] * b2,
        ];
        let parabola_inverse = invert_3(m).unwrap_or([[0.0; 3]; 3]);
        let parabola: [f64; 3] =
            std::array::from_fn(|i| (0..3).map(|j| parabola_inverse[i][j] * v[j]).sum::<f64>());
        let (mut lateral, mut vertical) = (0.0, 0.0);
        for &(s, d, z, w) in &bins {
            lateral += w * (d - line[0] - line[1] * s).powi(2);
            let at = parabola[0] * s * s + parabola[1] * s + parabola[2];
            vertical += w * (z - at).powi(2);
        }
        let dof = (weight - 3.0).max(1.0);
        self.fit = Fit {
            line,
            parabola,
            line_inverse,
            parabola_inverse,
            sigma_lateral_m: (lateral / dof).sqrt().max(floor),
            sigma_vertical_m: (vertical / dof).sqrt().max(floor),
            s_min_m: s_min,
            s_max_m: s_max,
            points: weight as u32,
        };
    }

    fn lateral_at(&self, s: f64) -> f64 {
        self.fit.line[0] + self.fit.line[1] * s
    }

    fn height_at(&self, s: f64) -> f64 {
        let p = self.fit.parabola;
        p[0] * s * s + p[1] * s + p[2]
    }

    /// One-sigma errors across the line and in height at `s`: the error of
    /// the curve there, and the misfit of the curve to its evidence, which
    /// more points do not reduce.
    fn sigmas_at(&self, s: f64) -> (f64, f64) {
        let fit = &self.fit;
        let lateral =
            fit.sigma_lateral_m * (1.0 + quadratic_2(&fit.line_inverse, [1.0, s]).max(0.0)).sqrt();
        let vertical = fit.sigma_vertical_m
            * (1.0 + quadratic_3(&fit.parabola_inverse, [s * s, s, 1.0]).max(0.0)).sqrt();
        (lateral, vertical)
    }

    fn world_at(&self, s: f64) -> [f64; 3] {
        let d = self.lateral_at(s);
        let [ux, uy] = self.direction;
        [
            self.origin_m[0] + ux * s - uy * d,
            self.origin_m[1] + uy * s + ux * d,
            self.height_at(s),
        ]
    }

    /// Whether `point` is this track's.
    fn claims(&self, point: [f64; 3], config: &WireTrackerConfig) -> bool {
        let (s, d) = self.frame(point[0], point[1]);
        if s < self.fit.s_min_m - config.extension_gate_m
            || s > self.fit.s_max_m + config.extension_gate_m
        {
            return false;
        }
        if self.ends[0].is_some_and(|end| s < end.s_m)
            || self.ends[1].is_some_and(|end| s > end.s_m)
        {
            return false;
        }
        (d - self.lateral_at(s)).abs() <= config.lateral_gate_m
            && (point[2] - self.height_at(s)).abs() <= config.vertical_gate_m
    }

    fn confirmed(&self, config: &WireTrackerConfig) -> bool {
        (self.triangulated && self.camera_support > 0)
            || (self.fit.points >= config.confirm_points
                && self.fit.s_max_m - self.fit.s_min_m >= config.confirm_length_m)
    }

    fn estimate(&self, config: &WireTrackerConfig) -> WireEstimate {
        // A pole is where the wire goes, however far past the evidence.
        let (s0, s1) = (
            self.ends[0].map_or(self.fit.s_min_m - config.extend_m, |end| end.s_m),
            self.ends[1].map_or(self.fit.s_max_m + config.extend_m + BIN_M, |end| end.s_m),
        );
        let (from_m, to_m) = (self.world_at(s0), self.world_at(s1));
        let middle = (s0 + s1) / 2.0;
        let chord = (from_m[2] + to_m[2]) / 2.0;
        let (lateral_0, vertical_0) = self.sigmas_at(s0);
        let (lateral_1, vertical_1) = self.sigmas_at(s1);
        WireEstimate {
            id: self.id,
            from_m,
            to_m,
            sag_m: (chord - self.height_at(middle)).max(0.0),
            sigma_lateral_m: lateral_0.max(lateral_1).min(config.max_sigma_m),
            sigma_vertical_m: vertical_0.max(vertical_1).min(config.max_sigma_m),
            observed_length_m: self.bins.len() as f64 * BIN_M,
            points: self.fit.points,
            camera_support: self.camera_support,
            triangulated: self.triangulated,
            confirmed: self.confirmed(config),
            pole_ends: self.ends.map(|end| end.map(|end| end.pole)),
        }
    }

    /// Whether a pole at `s_m` stands between this track and `other`.
    fn walled_from(&self, other: &WireTrack) -> bool {
        let [x, y, _] = other.world_at((other.fit.s_min_m + other.fit.s_max_m) / 2.0);
        let (s, _) = self.frame(x, y);
        self.ends[0].is_some_and(|end| s < end.s_m) || self.ends[1].is_some_and(|end| s > end.s_m)
    }

    /// Note a pole at `s_m` as this track's end on the side it lies, if it
    /// is nearer than the end known.
    fn end_at(&mut self, pole: u32, s_m: f64) {
        let middle = (self.fit.s_min_m + self.fit.s_max_m) / 2.0;
        let side = usize::from(s_m > middle);
        let nearer = self.ends[side]
            .is_none_or(|end| end.pole == pole || (end.s_m - middle).abs() > (s_m - middle).abs());
        if nearer {
            self.ends[side] = Some(TrackEnd { pole, s_m });
        }
    }
}

/// A camera line with no track yet, kept for a second view.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct PendingLine {
    origin_m: [f64; 3],
    from_dir: [f64; 3],
    to_dir: [f64; 3],
}

/// All the wires as far as the evidence goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireTracker {
    config: WireTrackerConfig,
    tracks: Vec<WireTrack>,
    next_id: u32,
    pending: VecDeque<[f64; 3]>,
    since_seeding: usize,
    lines: VecDeque<PendingLine>,
    /// The seeding generator: xorshift64, checkpointed with the rest.
    draw: u64,
}

impl WireTracker {
    pub fn new(config: WireTrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 0,
            pending: VecDeque::new(),
            since_seeding: 0,
            lines: VecDeque::new(),
            draw: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn config(&self) -> &WireTrackerConfig {
        &self.config
    }

    pub fn tracks(&self) -> &[WireTrack] {
        &self.tracks
    }

    /// Back to nothing known.
    pub fn clear(&mut self) {
        *self = Self::new(self.config.clone());
    }

    fn next_draw(&mut self) -> u64 {
        let mut x = self.draw;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.draw = x;
        x
    }

    /// Take one scan's points: each to the first track that claims it, the
    /// rest to the seeding pool.
    pub fn add_points(&mut self, points: &[[f64; 3]]) {
        if points.is_empty() {
            return;
        }
        let mut touched = vec![false; self.tracks.len()];
        for &point in points {
            let claimed = self
                .tracks
                .iter()
                .position(|track| track.claims(point, &self.config));
            match claimed {
                Some(k) => {
                    self.tracks[k].add(point);
                    touched[k] = true;
                }
                None => {
                    if self.pending.len() == self.config.pending_capacity {
                        self.pending.pop_front();
                    }
                    self.pending.push_back(point);
                    self.since_seeding += 1;
                }
            }
        }
        for (k, touched) in touched.into_iter().enumerate() {
            if touched {
                self.tracks[k].refit(&self.config);
            }
        }
        if self.pending.len() >= self.config.seed_min_points
            && self.since_seeding >= self.config.seed_min_points
        {
            self.since_seeding = 0;
            while self.seed() {}
        }
        self.merge();
    }

    /// One RANSAC pass over the unclaimed points. The inliers of the best
    /// chord, if enough, seed a track; true when one did.
    fn seed(&mut self) -> bool {
        let n = self.pending.len();
        if n < self.config.seed_min_points {
            return false;
        }
        let mut best: Option<(usize, usize, usize)> = None;
        for _ in 0..self.config.seed_draws {
            let (i, j) = (
                (self.next_draw() % n as u64) as usize,
                (self.next_draw() % n as u64) as usize,
            );
            let (p, q) = (self.pending[i], self.pending[j]);
            let span = (q[0] - p[0]).hypot(q[1] - p[1]);
            if span < self.config.confirm_length_m / 2.0 {
                continue;
            }
            let along: Vec<f64> = self
                .pending
                .iter()
                .filter(|point| near_chord(**point, p, q, &self.config))
                .map(|point| {
                    ((point[0] - p[0]) * (q[0] - p[0]) + (point[1] - p[1]) * (q[1] - p[1])) / span
                })
                .collect();
            let bins: std::collections::BTreeSet<i32> = along
                .iter()
                .filter(|s| **s >= 0.0 && **s <= span)
                .map(|s| (s / BIN_M).floor() as i32)
                .collect();
            let covered = bins.len() as f64 * BIN_M / span.max(BIN_M);
            if covered < self.config.seed_coverage {
                continue;
            }
            let count = along.len();
            if best.is_none_or(|(most, _, _)| count > most) {
                best = Some((count, i, j));
            }
        }
        let Some((count, i, j)) = best else {
            return false;
        };
        if count < self.config.seed_min_points / 2 {
            return false;
        }
        let (p, q) = (self.pending[i], self.pending[j]);
        let span = (q[0] - p[0]).hypot(q[1] - p[1]);
        let direction = [(q[0] - p[0]) / span, (q[1] - p[1]) / span];
        let mut track = WireTrack::new(
            self.next_id,
            [p[0], p[1]],
            direction,
            self.config.floor_sigma_m,
        );
        self.next_id += 1;
        let mut kept = VecDeque::with_capacity(n);
        for point in self.pending.drain(..) {
            if near_chord(point, p, q, &self.config) {
                track.add(point);
            } else {
                kept.push_back(point);
            }
        }
        track.refit(&self.config);
        // The fit reaches further than the chord did: what it now claims is
        // this wire's, and left in the pool would seed false wires later.
        self.pending = VecDeque::with_capacity(kept.len());
        for point in kept {
            if track.claims(point, &self.config) {
                track.add(point);
            } else {
                self.pending.push_back(point);
            }
        }
        track.refit(&self.config);
        self.tracks.push(track);
        true
    }

    /// Join tracks that turn out to be one conductor: nearly parallel, in
    /// each other's lateral gates, and fitting one curve together. Two spans
    /// of a line fail that last test at the support between them, where the
    /// wire has a corner no parabola follows.
    fn merge(&mut self) {
        let mut a = 0;
        while a < self.tracks.len() {
            let mut b = a + 1;
            while b < self.tracks.len() {
                match self.merged(a, b) {
                    Some(track) => {
                        self.tracks[a] = track;
                        self.tracks.remove(b);
                    }
                    None => b += 1,
                }
            }
            a += 1;
        }
    }

    /// Tracks `a` and `b` as one, if they are one wire.
    fn merged(&self, a: usize, b: usize) -> Option<WireTrack> {
        let (ta, tb) = (&self.tracks[a], &self.tracks[b]);
        if ta.walled_from(tb) || tb.walled_from(ta) {
            return None;
        }
        let cos = ta.direction[0] * tb.direction[0] + ta.direction[1] * tb.direction[1];
        if cos.abs() < 0.9995 {
            return None;
        }
        let reach = self.config.extension_gate_m;
        let in_line = [tb.fit.s_min_m, tb.fit.s_max_m].iter().all(|&s| {
            let point = tb.world_at(s);
            let (along, across) = ta.frame(point[0], point[1]);
            (across - ta.lateral_at(along)).abs() <= self.config.lateral_gate_m
                && along >= ta.fit.s_min_m - reach - (tb.fit.s_max_m - tb.fit.s_min_m)
                && along <= ta.fit.s_max_m + reach + (tb.fit.s_max_m - tb.fit.s_min_m)
        });
        if !in_line {
            return None;
        }
        let mut joined = ta.clone();
        for bin in tb.bins.values() {
            let mean = bin.mean();
            let (s, _) = joined.frame(mean[0], mean[1]);
            let into = joined.bins.entry((s / BIN_M).floor() as i32).or_default();
            into.n += bin.n;
            for k in 0..3 {
                into.sum[k] += bin.sum[k];
            }
        }
        joined.camera_support += tb.camera_support;
        joined.triangulated &= tb.triangulated;
        joined.floor_sigma_m = ta.floor_sigma_m.min(tb.floor_sigma_m);
        for end in tb.ends.into_iter().flatten() {
            let [x, y, _] = tb.world_at(end.s_m);
            let (s, _) = joined.frame(x, y);
            joined.end_at(end.pole, s);
        }
        joined.refit(&self.config);
        let fits = joined.fit.sigma_lateral_m <= self.config.merge_sigma_m
            && joined.fit.sigma_vertical_m <= self.config.merge_sigma_m;
        fits.then_some(joined)
    }

    /// Take one camera line: its origin and the sight directions bounding
    /// the part seen. It supports a track lying in its plane, or waits for a
    /// second view.
    pub fn add_camera_line(&mut self, origin_m: [f64; 3], from_dir: [f64; 3], to_dir: [f64; 3]) {
        let normal = cross(from_dir, to_dir);
        let length = norm(normal);
        if length < 1e-9 {
            return;
        }
        let normal = normal.map(|v| v / length);
        let gate = self.config.camera_gate_rad;
        let reach = self.config.extend_m;
        let supported = self.tracks.iter_mut().find(|track| {
            let (s, _) = track.frame(origin_m[0], origin_m[1]);
            let s = s.clamp(track.fit.s_min_m - reach, track.fit.s_max_m + reach);
            [s - 10.0, s + 10.0].iter().all(|&at| {
                let to = track.world_at(at);
                let d = [
                    to[0] - origin_m[0],
                    to[1] - origin_m[1],
                    to[2] - origin_m[2],
                ];
                dot(d, normal).abs() <= gate * norm(d).max(1.0)
            })
        });
        if let Some(track) = supported {
            track.camera_support += 1;
            return;
        }
        let line = PendingLine {
            origin_m,
            from_dir,
            to_dir,
        };
        let partner = self
            .lines
            .iter()
            .rev()
            .find_map(|other| triangulate(other, &line, &self.config));
        match partner {
            Some((start, end)) => self.seed_triangulated(start, end),
            None => {
                if self.lines.len() == self.config.line_capacity {
                    self.lines.pop_front();
                }
                self.lines.push_back(line);
            }
        }
    }

    /// A track from a triangulated stretch of wire: points along it stand
    /// for the two sight lines, and its error is a camera's.
    fn seed_triangulated(&mut self, start: [f64; 3], end: [f64; 3]) {
        let span = (end[0] - start[0]).hypot(end[1] - start[1]);
        // Under two bins a stretch fixes no direction.
        if span < 2.0 * BIN_M {
            return;
        }
        let direction = [(end[0] - start[0]) / span, (end[1] - start[1]) / span];
        let mut track = WireTrack::new(
            self.next_id,
            [start[0], start[1]],
            direction,
            self.config.triangulated_sigma_m,
        );
        self.next_id += 1;
        let steps = ((span / BIN_M).ceil() as usize).max(2);
        for k in 0..=steps {
            let t = k as f64 / steps as f64;
            track.add(std::array::from_fn(|i| start[i] + (end[i] - start[i]) * t));
        }
        track.triangulated = true;
        track.refit(&self.config);
        self.tracks.push(track);
    }

    /// Bring the tracks up to date with the poles known (W3). A pole on a
    /// track's line ends it there. One inside its evidence splits it: the
    /// wire has a corner at a support that a single curve fits through, and
    /// more evidence would not show that. Past the end of a track, a pole
    /// within `reach_m` is where the wire goes.
    pub fn set_poles(
        &mut self,
        poles: &[crate::poles::PoleAnchor],
        lateral_gate_m: f64,
        reach_m: f64,
    ) {
        for pole in poles {
            let mut k = 0;
            while k < self.tracks.len() {
                let (s, d) = self.tracks[k].frame(pole.xy_m[0], pole.xy_m[1]);
                let track = &self.tracks[k];
                let off = (d - track.lateral_at(s)).abs();
                let inside = s > track.fit.s_min_m + BIN_M && s < track.fit.s_max_m;
                let near = s >= track.fit.s_min_m - reach_m && s <= track.fit.s_max_m + reach_m;
                if off <= lateral_gate_m && near {
                    if inside
                        && !self.tracks[k]
                            .ends
                            .iter()
                            .flatten()
                            .any(|e| e.pole == pole.id)
                    {
                        self.split(k, pole.id, s);
                    } else {
                        self.tracks[k].end_at(pole.id, s);
                    }
                }
                k += 1;
            }
        }
    }

    /// Cut track `k` at `s_m`, at pole `pole`. A side with too little
    /// evidence for a track of its own is the corner's misfit, and goes.
    fn split(&mut self, k: usize, pole: u32, s_m: f64) {
        /// Fewest bins a side of a split needs to stand alone.
        const MIN_BINS: usize = 3;
        let track = &mut self.tracks[k];
        let cut = (s_m / BIN_M).floor() as i32;
        let far: BTreeMap<i32, Bin> = track.bins.split_off(&cut);
        let mut other = track.clone();
        other.bins = far;
        let (near_bins, far_bins) = (track.bins.len(), other.bins.len());
        match (near_bins >= MIN_BINS, far_bins >= MIN_BINS) {
            (true, true) => {
                other.id = self.next_id;
                self.next_id += 1;
                track.ends[1] = Some(TrackEnd { pole, s_m });
                other.ends[0] = Some(TrackEnd { pole, s_m });
                track.refit(&self.config);
                other.refit(&self.config);
                self.tracks.push(other);
            }
            (true, false) => {
                track.ends[1] = Some(TrackEnd { pole, s_m });
                track.refit(&self.config);
            }
            (false, true) => {
                other.ends[0] = Some(TrackEnd { pole, s_m });
                other.refit(&self.config);
                self.tracks[k] = other;
            }
            (false, false) => {
                let bins = other.bins;
                track.bins.extend(bins);
            }
        }
    }

    /// Every track's estimate, in track order.
    pub fn estimates(&self) -> Vec<WireEstimate> {
        self.tracks
            .iter()
            .map(|track| track.estimate(&self.config))
            .collect()
    }
}

/// Whether `point` lies near the level chord through `p` and `q`, taken
/// out past both ends by half their span.
fn near_chord(point: [f64; 3], p: [f64; 3], q: [f64; 3], config: &WireTrackerConfig) -> bool {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let span = dx.hypot(dy);
    let (ux, uy) = (dx / span, dy / span);
    let (rx, ry) = (point[0] - p[0], point[1] - p[1]);
    let along = rx * ux + ry * uy;
    if along < -span / 2.0 || along > span * 1.5 {
        return false;
    }
    let across = ry * ux - rx * uy;
    let height = p[2] + (q[2] - p[2]) * along / span;
    across.abs() <= config.lateral_gate_m / 2.0
        && (point[2] - height).abs() <= config.vertical_gate_m
}

/// Where two camera lines' planes cross, if they see the same wire: the
/// stretch of that crossing both saw, or `None` when they are too close
/// together, too parallel, or their crossing is not a level line.
fn triangulate(
    a: &PendingLine,
    b: &PendingLine,
    config: &WireTrackerConfig,
) -> Option<([f64; 3], [f64; 3])> {
    let baseline: [f64; 3] = std::array::from_fn(|k| b.origin_m[k] - a.origin_m[k]);
    if norm(baseline) < config.min_baseline_m {
        return None;
    }
    let unit = |v: [f64; 3]| {
        let n = norm(v);
        (n > 1e-9).then(|| v.map(|x| x / n))
    };
    let na = unit(cross(a.from_dir, a.to_dir))?;
    let nb = unit(cross(b.from_dir, b.to_dir))?;
    let along = cross(na, nb);
    let direction = unit(along)?;
    if norm(along) < 0.05 || direction[2].abs() > 0.3 {
        return None;
    }
    // A point on both planes: dot(x - oa, na) = 0, dot(x - ob, nb) = 0, on
    // the plane through oa square to the crossing.
    let m = [na, nb, direction];
    let inverse = invert_3(m)?;
    let rhs = [
        dot(a.origin_m, na),
        dot(b.origin_m, nb),
        dot(a.origin_m, direction),
    ];
    let point: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| inverse[i][j] * rhs[j]).sum());
    // How far along the crossing each sight direction lands, as seen from
    // its own origin: where its ray comes nearest the crossing.
    let along_of = |origin: [f64; 3], dir: [f64; 3]| -> Option<f64> {
        let w: [f64; 3] = std::array::from_fn(|k| origin[k] - point[k]);
        let (b1, d1, e1) = (dot(dir, direction), dot(dir, w), dot(direction, w));
        let denominator = 1.0 - b1 * b1;
        if denominator < 1e-6 {
            return None;
        }
        let forward = (b1 * e1 - d1) / denominator;
        (forward > 0.0).then_some((e1 - b1 * d1) / denominator)
    };
    let a0 = along_of(a.origin_m, a.from_dir)?;
    let a1 = along_of(a.origin_m, a.to_dir)?;
    let b0 = along_of(b.origin_m, b.from_dir)?;
    let b1 = along_of(b.origin_m, b.to_dir)?;
    let (a_lo, a_hi) = (a0.min(a1), a0.max(a1));
    let (b_lo, b_hi) = (b0.min(b1), b0.max(b1));
    let (lo, hi) = (a_lo.max(b_lo), a_hi.min(b_hi));
    if hi - lo < 1.0 {
        return None;
    }
    let start: [f64; 3] = std::array::from_fn(|k| point[k] + direction[k] * lo);
    let end: [f64; 3] = std::array::from_fn(|k| point[k] + direction[k] * hi);
    let in_range = |at: [f64; 3], origin: [f64; 3]| {
        let d: [f64; 3] = std::array::from_fn(|k| at[k] - origin[k]);
        norm(d) <= config.max_triangulation_range_m
    };
    (in_range(start, a.origin_m)
        && in_range(end, a.origin_m)
        && in_range(start, b.origin_m)
        && in_range(end, b.origin_m))
    .then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small repeatable generator for test noise.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % 10_000) as f64 / 10_000.0
        }

        /// Roughly normal, of one-sigma `sigma`.
        fn normal(&mut self, sigma: f64) -> f64 {
            ((0..12).map(|_| self.next()).sum::<f64>() - 6.0) * sigma
        }
    }

    /// A wire along x at y = `y`, from x = 0 to 200, 40 m up at its ends,
    /// sagging 6 m.
    fn wire_z(x: f64) -> f64 {
        let t = x / 200.0;
        40.0 - 4.0 * 6.0 * t * (1.0 - t)
    }

    fn scan(noise: &mut Noise, y: f64, from_x: f64, to_x: f64, n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|_| {
                let x = from_x + (to_x - from_x) * noise.next();
                [x, y + noise.normal(0.1), wire_z(x) + noise.normal(0.1)]
            })
            .collect()
    }

    fn tracker() -> WireTracker {
        WireTracker::new(WireTrackerConfig::default())
    }

    #[test]
    fn points_on_a_wire_seed_a_confirmed_track_with_its_sag() {
        let mut noise = Noise(11);
        let mut wires = tracker();
        wires.add_points(&scan(&mut noise, 50.0, 0.0, 200.0, 300));
        assert_eq!(wires.tracks().len(), 1);
        let estimate = &wires.estimates()[0];
        assert!(estimate.confirmed);
        // The fitted curve is the wire, within the noise.
        let track = &wires.tracks()[0];
        for x in [20.0, 100.0, 180.0] {
            let (s, _) = track.frame(x, 50.0);
            let at = track.world_at(s);
            assert!((at[1] - 50.0).abs() < 0.2, "{at:?}");
            assert!((at[2] - wire_z(at[0])).abs() < 0.3, "{at:?}");
        }
        assert!(estimate.sag_m > 4.0, "{estimate:?}");
        assert!(estimate.observed_length_m > 150.0);
    }

    #[test]
    fn a_short_sighting_is_not_confirmed_and_uncertainty_grows_past_it() {
        let mut noise = Noise(12);
        let mut wires = tracker();
        wires.add_points(&scan(&mut noise, 50.0, 90.0, 112.0, 25));
        let estimate = &wires.estimates()[0];
        assert!(!estimate.confirmed, "{estimate:?}");
        let track = &wires.tracks()[0];
        let (seen, _) = track.frame(97.0, 50.0);
        let (near_lateral, near_vertical) = track.sigmas_at(seen);
        let (far_lateral, far_vertical) = track.sigmas_at(seen + 100.0);
        assert!(
            far_lateral > 3.0 * near_lateral,
            "{far_lateral} {near_lateral}"
        );
        assert!(
            far_vertical > 3.0 * near_vertical,
            "{far_vertical} {near_vertical}"
        );
        // More of the wire, and it firms up.
        wires.add_points(&scan(&mut noise, 50.0, 40.0, 160.0, 200));
        assert_eq!(wires.tracks().len(), 1);
        assert!(wires.estimates()[0].confirmed);
        let track = &wires.tracks()[0];
        let (s, _) = track.frame(150.0, 50.0);
        assert!(track.sigmas_at(s).1 < 0.5);
    }

    #[test]
    fn two_conductors_four_metres_apart_stay_two_tracks() {
        let mut noise = Noise(13);
        let mut wires = tracker();
        let mut points = scan(&mut noise, 48.0, 0.0, 200.0, 200);
        points.extend(scan(&mut noise, 52.0, 0.0, 200.0, 200));
        wires.add_points(&points);
        assert_eq!(wires.tracks().len(), 2, "{:?}", wires.estimates());
    }

    #[test]
    fn scattered_points_seed_nothing() {
        let mut noise = Noise(14);
        let mut wires = tracker();
        let points: Vec<[f64; 3]> = (0..300)
            .map(|_| {
                [
                    noise.next() * 500.0,
                    noise.next() * 500.0,
                    20.0 + noise.next() * 60.0,
                ]
            })
            .collect();
        wires.add_points(&points);
        assert!(
            wires.estimates().iter().all(|e| !e.confirmed),
            "{:?}",
            wires.estimates()
        );
    }

    #[test]
    fn sightings_of_one_wire_from_two_ends_merge() {
        let mut noise = Noise(15);
        let mut wires = tracker();
        wires.add_points(&scan(&mut noise, 50.0, 0.0, 70.0, 80));
        wires.add_points(&scan(&mut noise, 50.0, 110.0, 190.0, 80));
        wires.add_points(&scan(&mut noise, 50.0, 60.0, 120.0, 80));
        assert_eq!(wires.tracks().len(), 1, "{:?}", wires.estimates());
    }

    /// Two spans of one line, meeting at a support with a corner in height,
    /// are two wires: no parabola follows the corner.
    #[test]
    fn two_spans_of_a_line_stay_two_tracks() {
        let mut noise = Noise(19);
        let mut wires = tracker();
        let span = |noise: &mut Noise, from: f64, n: usize| -> Vec<[f64; 3]> {
            (0..n)
                .map(|_| {
                    let x = from + 200.0 * noise.next();
                    [
                        x,
                        50.0 + noise.normal(0.1),
                        wire_z(x - from) + noise.normal(0.1),
                    ]
                })
                .collect()
        };
        wires.add_points(&span(&mut noise, 0.0, 300));
        wires.add_points(&span(&mut noise, 200.0, 300));
        assert_eq!(wires.tracks().len(), 2, "{:?}", wires.estimates());
    }

    /// A fit that does not follow its points says so: more points cannot
    /// make a bad curve look certain.
    #[test]
    fn misfit_shows_in_the_uncertainty_however_many_points() {
        let mut track = WireTrack::new(0, [0.0, 0.0], [1.0, 0.0], 0.15);
        let config = WireTrackerConfig::default();
        // A wire with a corner in the middle: 3 m up and down every 20 m.
        for k in 0..4000 {
            let x = (k % 200) as f64;
            let z = if (x / 20.0).floor() as i32 % 2 == 0 {
                40.0
            } else {
                43.0
            };
            track.add([x, 0.0, z]);
        }
        track.refit(&config);
        let (_, vertical) = track.sigmas_at(100.0);
        assert!(vertical > 1.0, "{vertical}");
    }

    #[test]
    fn evidence_is_bounded_however_long_the_wire_is_scanned() {
        let mut noise = Noise(16);
        let mut wires = tracker();
        for _ in 0..200 {
            wires.add_points(&scan(&mut noise, 50.0, 0.0, 200.0, 50));
        }
        assert_eq!(wires.tracks().len(), 1);
        assert!(wires.tracks()[0].bins.len() <= (200.0 / BIN_M) as usize + 2);
    }

    #[test]
    fn a_camera_line_supports_a_track_it_lies_along() {
        let mut noise = Noise(17);
        let mut wires = tracker();
        wires.add_points(&scan(&mut noise, 50.0, 0.0, 200.0, 300));
        // From 20 m south of the wire, seeing 80..120 of it.
        let origin = [100.0, 30.0, 34.0];
        let dir = |x: f64| {
            let d = [x - origin[0], 50.0 - origin[1], wire_z(x) - origin[2]];
            let n = norm(d);
            d.map(|v| v / n)
        };
        wires.add_camera_line(origin, dir(80.0), dir(120.0));
        assert_eq!(wires.estimates()[0].camera_support, 1);
        // A line nowhere near it waits.
        wires.add_camera_line(origin, [1.0, 0.0, 0.0], [0.9, 0.0, 0.4359]);
        assert_eq!(wires.estimates()[0].camera_support, 1);
    }

    #[test]
    fn two_camera_views_triangulate_a_wire_with_a_cameras_uncertainty() {
        let mut wires = tracker();
        let view = |origin: [f64; 3]| {
            let dir = |x: f64| {
                let d = [x - origin[0], 50.0 - origin[1], 35.0 - origin[2]];
                let n = norm(d);
                d.map(|v| v / n)
            };
            (dir(90.0), dir(110.0))
        };
        // One line has no depth: nothing yet.
        let (a, b) = view([60.0, 20.0, 40.0]);
        wires.add_camera_line([60.0, 20.0, 40.0], a, b);
        assert!(wires.tracks().is_empty());
        // A second view from elsewhere crosses it along the wire.
        let (a, b) = view([140.0, 25.0, 45.0]);
        wires.add_camera_line([140.0, 25.0, 45.0], a, b);
        assert_eq!(wires.tracks().len(), 1);
        let estimate = &wires.estimates()[0];
        assert!(estimate.triangulated, "{estimate:?}");
        assert!(!estimate.confirmed, "two lines also cross on two wires");
        assert!(
            estimate.sigma_lateral_m >= 2.0,
            "a camera's error, not a LiDAR's"
        );
        assert!(estimate.sigma_lateral_m <= 3.0, "capped: {estimate:?}");
        // A third view agreeing with it confirms it.
        let (a, b) = view([100.0, 15.0, 42.0]);
        wires.add_camera_line([100.0, 15.0, 42.0], a, b);
        assert_eq!(wires.tracks().len(), 1);
        assert!(wires.estimates()[0].confirmed, "{:?}", wires.estimates());
        let track = &wires.tracks()[0];
        let (s, _) = track.frame(100.0, 50.0);
        let at = track.world_at(s);
        assert!(
            (at[1] - 50.0).abs() < 1.0 && (at[2] - 35.0).abs() < 1.0,
            "{at:?}"
        );
    }

    #[test]
    fn a_run_is_deterministic_and_survives_a_checkpoint() {
        let feed = |wires: &mut WireTracker, seed: u64, rounds: usize| {
            let mut noise = Noise(seed);
            for _ in 0..rounds {
                wires.add_points(&scan(&mut noise, 50.0, 0.0, 200.0, 40));
            }
        };
        let mut straight = tracker();
        feed(&mut straight, 18, 6);
        let mut first = tracker();
        feed(&mut first, 18, 3);
        let text = serde_json::to_string(&first).unwrap();
        let mut resumed: WireTracker = serde_json::from_str(&text).unwrap();
        // The same noise onward: replay the first half to reach it.
        let mut noise = Noise(18);
        for _ in 0..3 {
            scan(&mut noise, 50.0, 0.0, 200.0, 40);
        }
        for _ in 0..3 {
            resumed.add_points(&scan(&mut noise, 50.0, 0.0, 200.0, 40));
        }
        assert_eq!(straight, resumed);
    }

    #[test]
    fn invalid_configs_are_refused() {
        assert!(WireTrackerConfig::default().validate().is_ok());
        let bad = [
            WireTrackerConfig {
                lateral_gate_m: 0.0,
                ..WireTrackerConfig::default()
            },
            WireTrackerConfig {
                confirm_points: 0,
                ..WireTrackerConfig::default()
            },
            WireTrackerConfig {
                pending_capacity: 1,
                ..WireTrackerConfig::default()
            },
        ];
        for config in bad {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }

    /// Points along a wire at y = 0, z = 50, from `x0` to `x1`.
    fn along(x0: f64, x1: f64) -> Vec<[f64; 3]> {
        let n = ((x1 - x0) / 1.0) as usize;
        (0..n)
            .map(|k| [x0 + k as f64, 0.01 * ((k * 7 % 5) as f64), 50.0])
            .collect()
    }

    fn anchor(id: u32, x: f64) -> crate::poles::PoleAnchor {
        crate::poles::PoleAnchor { id, xy_m: [x, 3.0] }
    }

    /// W3: a pole inside a track's evidence cuts it in two, each ending at
    /// the pole, and more evidence neither joins them nor crosses it.
    #[test]
    fn a_pole_splits_a_track_and_keeps_it_split() {
        let mut tracker = WireTracker::new(WireTrackerConfig::default());
        tracker.add_points(&along(0.0, 200.0));
        assert_eq!(tracker.tracks().len(), 1);
        tracker.set_poles(&[anchor(7, 100.0)], 8.0, 120.0);
        assert_eq!(tracker.tracks().len(), 2);
        let estimates = tracker.estimates();
        for estimate in &estimates {
            assert_eq!(estimate.pole_ends.iter().flatten().count(), 1);
            assert!(estimate.pole_ends.contains(&Some(7)));
        }
        tracker.add_points(&along(0.0, 200.0));
        tracker.set_poles(&[anchor(7, 100.0)], 8.0, 120.0);
        assert_eq!(tracker.tracks().len(), 2);
    }

    #[test]
    fn a_track_runs_to_a_pole_past_its_evidence() {
        let mut tracker = WireTracker::new(WireTrackerConfig::default());
        tracker.add_points(&along(0.0, 100.0));
        tracker.set_poles(&[anchor(1, 180.0)], 8.0, 120.0);
        let estimate = &tracker.estimates()[0];
        let end = estimate
            .pole_ends
            .iter()
            .position(|end| *end == Some(1))
            .expect("ends at the pole");
        let reached = if end == 0 {
            estimate.from_m
        } else {
            estimate.to_m
        };
        assert!((reached[0] - 180.0).abs() < 1.0, "{reached:?}");
    }

    #[test]
    fn a_pole_off_the_line_changes_nothing() {
        let mut tracker = WireTracker::new(WireTrackerConfig::default());
        tracker.add_points(&along(0.0, 100.0));
        let before = tracker.clone();
        tracker.set_poles(
            &[crate::poles::PoleAnchor {
                id: 1,
                xy_m: [50.0, 30.0],
            }],
            8.0,
            120.0,
        );
        assert_eq!(tracker, before);
    }

    #[test]
    fn a_pole_near_a_track_end_ends_it_without_a_split() {
        let mut tracker = WireTracker::new(WireTrackerConfig::default());
        tracker.add_points(&along(0.0, 100.0));
        tracker.set_poles(&[anchor(2, 99.0)], 8.0, 120.0);
        assert_eq!(tracker.tracks().len(), 1);
        assert!(tracker.estimates()[0].pole_ends.contains(&Some(2)));
    }
}
