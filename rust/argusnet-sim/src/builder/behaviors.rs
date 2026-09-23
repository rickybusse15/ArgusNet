//! Target behaviours: `behaviors.build_target_trajectory` and the eight
//! presets it dispatches to (P4.9b.2).
//!
//! Ported line for line, with the reference's evaluation order kept wherever
//! it can change the last bit: numpy's elementwise array expressions are
//! separate ufunc passes (never fused), Python's `**` is C `pow` (so
//! `powf`, not `powi`), and Python's float `%` takes the divisor's sign
//! ([`py_mod`]).
//!
//! # The generator
//!
//! Every draw goes through [`NumpyRng`]: `uniform`, `integers` and `random`,
//! on the exact `default_rng(seed)` stream. `uniform` is the **unfused**
//! `low + (high - low) * u`, which is what numpy computes on a build that
//! does not contract it to a fused multiply-add. Some builds do (arm64,
//! numpy 2.5), and their values differ in the last place; the parity fixture
//! is therefore recorded with the unfused form, so the behaviour logic is
//! pinned bit for bit and the platform's FMA stays a separately measured
//! effect.

use std::f64::consts::PI;

use argusnet_world::numpy_rng::NumpyRng;

use crate::waypoint_path::py_mod;

type Vec3 = [f64; 3];

/// `(position, velocity)` at a time.
pub type Sample = (Vec3, Vec3);

/// A terrain height query, for the evasive behaviour's nap-of-earth dips.
pub type TerrainHeightFn<'a> = &'a dyn Fn(f64, f64) -> f64;

/// `BEHAVIOR_PRESETS`.
pub const BEHAVIOR_PRESETS: [&str; 8] = [
    "loiter",
    "transit",
    "evasive",
    "search_pattern",
    "sinusoid",
    "racetrack",
    "waypoint_patrol",
    "mixed",
];

fn norm3(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// `d / norm * speed`, elementwise and in that order.
fn unit_times(d: Vec3, norm: f64, speed: f64) -> Vec3 {
    [
        d[0] / norm * speed,
        d[1] / norm * speed,
        d[2] / norm * speed,
    ]
}

/// `h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1`, left to right.
fn hermite_sum(h: [f64; 4], p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3) -> Vec3 {
    let mut out = [0.0; 3];
    for k in 0..3 {
        out[k] = h[0] * p0[k] + h[1] * m0[k] + h[2] * p1[k] + h[3] * m1[k];
    }
    out
}

fn hermite_basis(u: f64) -> ([f64; 4], [f64; 4]) {
    let u2 = u * u;
    let u3 = u2 * u;
    (
        [
            2.0 * u3 - 3.0 * u2 + 1.0,
            u3 - 2.0 * u2 + u,
            -2.0 * u3 + 3.0 * u2,
            u3 - u2,
        ],
        [
            6.0 * u2 - 6.0 * u,
            3.0 * u2 - 4.0 * u + 1.0,
            -6.0 * u2 + 6.0 * u,
            3.0 * u2 - 2.0 * u,
        ],
    )
}

/// Catmull-Rom-style tangents: the neighbours' difference, normalised to the
/// speed, or zero for a degenerate one.
fn tangents(points: &[Vec3], speed: f64) -> Vec<Vec3> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let d = if i == 0 {
                sub(points[1], points[0])
            } else if i == n - 1 {
                sub(points[n - 1], points[n - 2])
            } else {
                sub(points[i + 1], points[i - 1])
            };
            let norm = norm3(d);
            if norm > 1e-9 {
                unit_times(d, norm, speed)
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

/// `LoiterBehavior`.
#[derive(Debug, Clone)]
pub struct Loiter {
    center: Vec3,
    radius_m: f64,
    alt_var_m: f64,
    alt_period_s: f64,
    omega: f64,
}

impl Loiter {
    pub fn new(center: Vec3, radius_m: f64, speed_mps: f64, clockwise: bool) -> Self {
        let radius_m = radius_m.max(1.0);
        let speed = speed_mps.max(0.5);
        let direction = if clockwise { -1.0 } else { 1.0 };
        Self {
            center,
            radius_m,
            alt_var_m: 20.0,
            alt_period_s: 60.0f64.max(1.0),
            omega: direction * speed / radius_m,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let angle = self.omega * t;
        let (cx, cy, base_alt) = (self.center[0], self.center[1], self.center[2]);
        let x = cx + self.radius_m * angle.cos();
        let y = cy + self.radius_m * angle.sin();
        let alt_phase = 2.0 * PI * t / self.alt_period_s;
        let z = base_alt + self.alt_var_m * alt_phase.sin();
        let vx = -self.radius_m * self.omega * angle.sin();
        let vy = self.radius_m * self.omega * angle.cos();
        let vz = self.alt_var_m * (2.0 * PI / self.alt_period_s) * alt_phase.cos();
        ([x, y, z], [vx, vy, vz])
    }
}

/// `TransitBehavior`: Hermite through the waypoints at constant speed, then
/// a 20 m loiter around the last one.
#[derive(Debug, Clone)]
pub struct Transit {
    waypoints: Vec<Vec3>,
    speed: f64,
    segment_lengths: Vec<f64>,
    cumulative_times: Vec<f64>,
    total_time: f64,
    tangents: Vec<Vec3>,
}

impl Transit {
    pub fn new(waypoints: Vec<Vec3>, speed_mps: f64) -> Self {
        assert!(
            waypoints.len() >= 2,
            "TransitBehavior requires at least 2 waypoints"
        );
        let speed = speed_mps.max(0.5);
        let mut segment_lengths = Vec::new();
        let mut cumulative_times = vec![0.0];
        for pair in waypoints.windows(2) {
            let length = norm3(sub(pair[1], pair[0]));
            segment_lengths.push(length.max(1e-6));
            let last = *cumulative_times.last().expect("seeded with 0");
            cumulative_times.push(last + length.max(1e-6) / speed);
        }
        let total_time = *cumulative_times.last().expect("non-empty");
        let tangents = tangents(&waypoints, speed);
        Self {
            waypoints,
            speed,
            segment_lengths,
            cumulative_times,
            total_time,
            tangents,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let t = if t < 0.0 { 0.0 } else { t };
        if t >= self.total_time {
            let last = *self.waypoints.last().expect("non-empty");
            let dt = t - self.total_time;
            let loiter_radius = 20.0;
            let omega = self.speed / loiter_radius;
            let angle = omega * dt;
            let x = last[0] + loiter_radius * angle.cos();
            let y = last[1] + loiter_radius * angle.sin();
            let vx = -loiter_radius * omega * angle.sin();
            let vy = loiter_radius * omega * angle.cos();
            return ([x, y, last[2]], [vx, vy, 0.0]);
        }
        let mut index = self.segment_lengths.len() - 1;
        for i in 0..self.cumulative_times.len() - 1 {
            if t < self.cumulative_times[i + 1] {
                index = i;
                break;
            }
        }
        let start = self.cumulative_times[index];
        let duration = self.cumulative_times[index + 1] - start;
        let u = ((t - start) / duration.max(1e-9)).clamp(0.0, 1.0);
        let m0 = scale(self.tangents[index], duration);
        let m1 = scale(self.tangents[index + 1], duration);
        let (h, dh) = hermite_basis(u);
        let p0 = self.waypoints[index];
        let p1 = self.waypoints[index + 1];
        let position = hermite_sum(h, p0, m0, p1, m1);
        let tangent_du = hermite_sum(dh, p0, m0, p1, m1);
        let d = duration.max(1e-9);
        (
            position,
            [tangent_du[0] / d, tangent_du[1] / d, tangent_du[2] / d],
        )
    }
}

/// `SearchPatternBehavior` with the `expanding_square` pattern, which is the
/// only one the presets build.
#[derive(Debug, Clone)]
pub struct SearchPattern {
    center: Vec3,
    speed: f64,
    waypoints: Vec<Vec3>,
    segment_lengths: Vec<f64>,
    cumulative_dists: Vec<f64>,
    total_dist: f64,
    tangents: Vec<Vec3>,
}

impl SearchPattern {
    pub fn expanding_square(
        center: Vec3,
        leg_length_m: f64,
        speed_mps: f64,
        altitude_m: f64,
    ) -> Self {
        let leg = leg_length_m.max(10.0);
        let speed = speed_mps.max(0.5);
        let (cx, cy) = (center[0], center[1]);
        let mut waypoints = vec![[cx, cy, altitude_m]];
        let directions: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for i in 0..20usize {
            let multiplier = (i / 2 + 1) as f64 * leg;
            let (dx, dy) = directions[i % 4];
            let last = *waypoints.last().expect("seeded");
            waypoints.push([
                last[0] + dx * multiplier,
                last[1] + dy * multiplier,
                altitude_m,
            ]);
        }
        let mut segment_lengths = Vec::new();
        let mut cumulative_dists = vec![0.0];
        for pair in waypoints.windows(2) {
            let d = norm3(sub(pair[1], pair[0]));
            segment_lengths.push(d.max(1e-6));
            let last = *cumulative_dists.last().expect("seeded");
            cumulative_dists.push(last + d.max(1e-6));
        }
        let total_dist = *cumulative_dists.last().expect("non-empty");
        let tangents = tangents(&waypoints, speed);
        Self {
            center,
            speed,
            waypoints,
            segment_lengths,
            cumulative_dists,
            total_dist,
            tangents,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        if self.waypoints.len() < 2 {
            return (self.center, [0.0; 3]);
        }
        let dist = py_mod(self.speed * t.max(0.0), self.total_dist);
        let mut index = self.segment_lengths.len().saturating_sub(1);
        for i in 0..self.cumulative_dists.len() - 1 {
            if dist < self.cumulative_dists[i + 1] {
                index = i;
                break;
            }
        }
        let start = self.cumulative_dists[index];
        let length = self.segment_lengths[index].max(1e-9);
        let u = ((dist - start) / length).clamp(0.0, 1.0);
        let p0 = self.waypoints[index];
        let p1 = self.waypoints[index + 1];
        let s = length / self.speed;
        let m0 = scale(self.tangents[index], s);
        let m1 = scale(self.tangents[index + 1], s);
        let (h, dh) = hermite_basis(u);
        let position = hermite_sum(h, p0, m0, p1, m1);
        let tangent_du = hermite_sum(dh, p0, m0, p1, m1);
        let tnorm = norm3(tangent_du);
        let velocity = if tnorm > 1e-9 {
            unit_times(tangent_du, tnorm, self.speed)
        } else {
            [0.0; 3]
        };
        (position, velocity)
    }
}

/// `_sinusoid_trajectory`.
#[derive(Debug, Clone)]
pub struct Sinusoid {
    start_xy: [f64; 2],
    vx: f64,
    vy: f64,
    lateral_amp: f64,
    freq: f64,
    phase: f64,
    altitude_m: f64,
}

impl Sinusoid {
    pub fn new(bounds: Bounds, altitude_m: f64, speed_mps: f64, seed: u32) -> Self {
        let mut rng = NumpyRng::new(seed);
        let (center, span) = bounds.center_and_span();
        let draw = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)];
        let start_xy = [center[0] + draw[0] * span[0], center[1] + draw[1] * span[1]];
        let angle = rng.uniform(0.0, 2.0 * PI);
        let span_min = span[0].min(span[1]);
        Self {
            start_xy,
            vx: speed_mps * angle.cos() * 0.3,
            vy: speed_mps * angle.sin() * 0.3,
            lateral_amp: span_min * 0.08,
            freq: speed_mps / span_min.max(100.0) * 2.0,
            phase: rng.uniform(0.0, 2.0 * PI),
            altitude_m,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let x =
            self.start_xy[0] + self.vx * t + self.lateral_amp * (self.freq * t + self.phase).sin();
        let y = self.start_xy[1] + self.vy * t;
        let z = self.altitude_m + 15.0 * (0.08 * t + self.phase * 0.5).sin();
        let dx = self.vx + self.lateral_amp * self.freq * (self.freq * t + self.phase).cos();
        let dz = 15.0 * 0.08 * (0.08 * t + self.phase * 0.5).cos();
        ([x, y, z], [dx, self.vy, dz])
    }
}

/// `_racetrack_trajectory`.
#[derive(Debug, Clone)]
pub struct Racetrack {
    cx: f64,
    cy: f64,
    straight: f64,
    turn_r: f64,
    track_len: f64,
    phase_offset: f64,
    speed: f64,
    altitude_m: f64,
}

impl Racetrack {
    pub fn new(bounds: Bounds, altitude_m: f64, speed_mps: f64, seed: u32) -> Self {
        let mut rng = NumpyRng::new(seed);
        let (center, span) = bounds.center_and_span();
        let cx = center[0] + rng.uniform(-0.15, 0.15) * span[0];
        let cy = center[1] + rng.uniform(-0.15, 0.15) * span[1];
        let span_min = span[0].min(span[1]);
        let straight = span_min * 0.2;
        let turn_r = span_min * 0.06;
        let track_len = 2.0 * straight + 2.0 * PI * turn_r;
        let phase_offset = rng.uniform(0.0, track_len);
        Self {
            cx,
            cy,
            straight,
            turn_r,
            track_len,
            phase_offset,
            speed: speed_mps,
            altitude_m,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let (cx, cy, straight, turn_r, speed) =
            (self.cx, self.cy, self.straight, self.turn_r, self.speed);
        let d = py_mod(speed * t + self.phase_offset, self.track_len);
        let (x, y, vx, vy);
        if d < straight {
            x = cx - straight * 0.5 + d;
            y = cy - turn_r;
            (vx, vy) = (speed, 0.0);
        } else if d < straight + PI * turn_r {
            let arc = d - straight;
            let ang = -PI / 2.0 + arc / turn_r;
            x = cx + straight * 0.5 + turn_r * ang.cos();
            y = cy + turn_r * ang.sin();
            vx = -speed * ang.sin();
            vy = speed * ang.cos();
        } else if d < 2.0 * straight + PI * turn_r {
            let top_d = d - straight - PI * turn_r;
            x = cx + straight * 0.5 - top_d;
            y = cy + turn_r;
            (vx, vy) = (-speed, 0.0);
        } else {
            let arc = d - 2.0 * straight - PI * turn_r;
            let ang = PI / 2.0 + arc / turn_r;
            x = cx - straight * 0.5 + turn_r * ang.cos();
            y = cy + turn_r * ang.sin();
            vx = -speed * ang.sin();
            vy = speed * ang.cos();
        }
        let z = self.altitude_m + 10.0 * (0.06 * t).sin();
        let dz = 10.0 * 0.06 * (0.06 * t).cos();
        ([x, y, z], [vx, vy, dz])
    }
}

/// `_waypoint_patrol_trajectory`.
#[derive(Debug, Clone)]
pub struct WaypointPatrol {
    waypoints: Vec<[f64; 2]>,
    seg_lens: Vec<f64>,
    cum_dists: Vec<f64>,
    total_dist: f64,
    speed: f64,
    altitude_m: f64,
}

impl WaypointPatrol {
    pub fn new(bounds: Bounds, altitude_m: f64, speed_mps: f64, seed: u32) -> Self {
        let mut rng = NumpyRng::new(seed);
        let (center, span) = bounds.center_and_span();
        let half_w = span[0] * 0.15;
        let half_h = span[1] * 0.15;
        let draw = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)];
        let (cx, cy) = (center[0] + draw[0] * span[0], center[1] + draw[1] * span[1]);
        let mut waypoints = vec![
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ];
        waypoints.push(waypoints[0]);
        let mut seg_lens = Vec::new();
        let mut cum_dists = vec![0.0];
        for pair in waypoints.windows(2) {
            let (dx, dy) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            let d = (dx * dx + dy * dy).sqrt();
            seg_lens.push(d.max(1e-6));
            let last = *cum_dists.last().expect("seeded");
            cum_dists.push(last + d.max(1e-6));
        }
        let total_dist = *cum_dists.last().expect("non-empty");
        Self {
            waypoints,
            seg_lens,
            cum_dists,
            total_dist,
            speed: speed_mps,
            altitude_m,
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let dist = py_mod(self.speed * t.max(0.0), self.total_dist);
        let mut index = self.seg_lens.len() - 1;
        for i in 0..self.cum_dists.len() - 1 {
            if dist < self.cum_dists[i + 1] {
                index = i;
                break;
            }
        }
        let frac =
            ((dist - self.cum_dists[index]) / self.seg_lens[index].max(1e-9)).clamp(0.0, 1.0);
        let p0 = self.waypoints[index];
        let p1 = self.waypoints[index + 1];
        let xy = [
            p0[0] + (p1[0] - p0[0]) * frac,
            p0[1] + (p1[1] - p0[1]) * frac,
        ];
        let direction = [p1[0] - p0[0], p1[1] - p0[1]];
        let norm = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        let velocity = if norm > 1e-9 {
            let d = norm.max(1e-9);
            [direction[0] / d * self.speed, direction[1] / d * self.speed]
        } else {
            [0.0, 0.0]
        };
        let z = self.altitude_m + 12.0 * (0.05 * t).sin();
        let dz = 12.0 * 0.05 * (0.05 * t).cos();
        ([xy[0], xy[1], z], [velocity[0], velocity[1], dz])
    }
}

/// `EvasiveBehavior` over a sinusoid: nap-of-earth dips in windows drawn
/// every 5 s over the first 600 s.
#[derive(Debug, Clone)]
pub struct Evasive {
    base: Sinusoid,
    noe_agl: f64,
    windows: Vec<(f64, f64)>,
}

impl Evasive {
    pub fn new(base: Sinusoid, seed: u32) -> Self {
        let (noe_agl, probability, duration) =
            (40.0f64.max(5.0), 0.3f64.clamp(0.0, 1.0), 15.0f64.max(1.0));
        let mut rng = NumpyRng::new(seed);
        let check_interval = 5.0;
        let mut t = 0.0;
        let mut windows = Vec::new();
        while t < 600.0 {
            if rng.random() < probability * (check_interval / duration) {
                let end = t + duration;
                windows.push((t, end));
                t = end;
            } else {
                t += check_interval;
            }
        }
        Self {
            base,
            noe_agl,
            windows,
        }
    }

    fn evading(&self, t: f64) -> Option<f64> {
        let ramp_s = 2.0;
        for &(start, end) in &self.windows {
            if start <= t && t <= end {
                let ramp_in = ((t - start) / ramp_s).min(1.0);
                let ramp_out = ((end - t) / ramp_s).min(1.0);
                return Some(ramp_in.min(ramp_out));
            }
        }
        None
    }

    pub fn sample(&self, t: f64, terrain: TerrainHeightFn<'_>) -> Sample {
        let (position, velocity) = self.base.sample(t);
        let Some(blend) = self.evading(t) else {
            return (position, velocity);
        };
        if blend <= 0.0 {
            return (position, velocity);
        }
        let noe_z = terrain(position[0], position[1]) + self.noe_agl;
        let target_z = position[2] * (1.0 - blend) + noe_z * blend;
        let dz = target_z - position[2];
        (
            [position[0], position[1], target_z],
            [velocity[0], velocity[1], velocity[2] + dz * 0.5],
        )
    }
}

/// `CompositeTrajectory` for the `mixed` preset: loiter from 0 s, transit
/// from 30 s, search from 90 s, each switch blended over 2 s.
#[derive(Debug, Clone)]
pub struct Mixed {
    loiter: Loiter,
    transit: Transit,
    search: SearchPattern,
}

impl Mixed {
    fn segment(&self, index: usize, t: f64) -> Sample {
        match index {
            0 => self.loiter.sample(t),
            1 => self.transit.sample(t),
            _ => self.search.sample(t),
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        const STARTS: [f64; 3] = [0.0, 30.0, 90.0];
        const BLEND_DURATION_S: f64 = 2.0;
        let mut index = 0;
        for (i, start) in STARTS.iter().enumerate() {
            if t >= *start {
                index = i;
            }
        }
        if index > 0 && t < STARTS[index] + BLEND_DURATION_S {
            let alpha = ((t - STARTS[index]) / BLEND_DURATION_S).clamp(0.0, 1.0);
            // `3 * alpha**2 - 2 * alpha**3`: C `pow`, as Python's `**` is.
            let alpha = 3.0 * alpha.powf(2.0) - 2.0 * alpha.powf(3.0);
            let (p_old, v_old) = self.segment(index - 1, t);
            let (p_new, v_new) = self.segment(index, t);
            let mix = |a: Vec3, b: Vec3| {
                [
                    a[0] * (1.0 - alpha) + b[0] * alpha,
                    a[1] * (1.0 - alpha) + b[1] * alpha,
                    a[2] * (1.0 - alpha) + b[2] * alpha,
                ]
            };
            return (mix(p_old, p_new), mix(v_old, v_new));
        }
        self.segment(index, t)
    }
}

/// Map bounds as `_bounds_center_and_span` reads them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

impl Bounds {
    fn center_and_span(self) -> ([f64; 2], [f64; 2]) {
        (
            [
                (self.x_min_m + self.x_max_m) * 0.5,
                (self.y_min_m + self.y_max_m) * 0.5,
            ],
            [self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m],
        )
    }
}

/// A built target behaviour.
#[derive(Debug, Clone)]
pub enum Behavior {
    Loiter(Loiter),
    Transit(Transit),
    Evasive(Evasive),
    SearchPattern(SearchPattern),
    Sinusoid(Sinusoid),
    Racetrack(Racetrack),
    WaypointPatrol(WaypointPatrol),
    Mixed(Mixed),
}

impl Behavior {
    /// The behaviour's position and velocity at `t`. `terrain` is read only
    /// by the evasive behaviour.
    pub fn sample(&self, t: f64, terrain: TerrainHeightFn<'_>) -> Sample {
        match self {
            Behavior::Loiter(b) => b.sample(t),
            Behavior::Transit(b) => b.sample(t),
            Behavior::Evasive(b) => b.sample(t, terrain),
            Behavior::SearchPattern(b) => b.sample(t),
            Behavior::Sinusoid(b) => b.sample(t),
            Behavior::Racetrack(b) => b.sample(t),
            Behavior::WaypointPatrol(b) => b.sample(t),
            Behavior::Mixed(b) => b.sample(t),
        }
    }
}

/// `build_target_trajectory(preset, bounds, altitude_m, speed_mps, seed)`.
pub fn build_target_behavior(
    preset: &str,
    bounds: Bounds,
    altitude_m: f64,
    speed_mps: f64,
    seed: u32,
) -> Result<Behavior, String> {
    if !BEHAVIOR_PRESETS.contains(&preset) {
        return Err(format!("unknown behaviour preset {preset:?}"));
    }
    // One generator for the factory's own draws; the sinusoid, racetrack,
    // patrol and evasive windows each open their own from the same seed.
    let mut rng = NumpyRng::new(seed);
    let (center, span) = bounds.center_and_span();
    let span_min = span[0].min(span[1]);
    Ok(match preset {
        "loiter" => {
            let x = center[0] + rng.uniform(-0.15, 0.15) * span[0];
            let y = center[1] + rng.uniform(-0.15, 0.15) * span[1];
            let clockwise = rng.integers(0, 2) != 0;
            Behavior::Loiter(Loiter::new(
                [x, y, altitude_m],
                span_min * 0.08,
                speed_mps,
                clockwise,
            ))
        }
        "transit" => {
            let count = rng.integers(3, 6);
            let waypoints = (0..count)
                .map(|_| {
                    let x = center[0] + rng.uniform(-0.3, 0.3) * span[0];
                    let y = center[1] + rng.uniform(-0.3, 0.3) * span[1];
                    let z = altitude_m + rng.uniform(-20.0, 20.0);
                    [x, y, z]
                })
                .collect();
            Behavior::Transit(Transit::new(waypoints, speed_mps))
        }
        "evasive" => Behavior::Evasive(Evasive::new(
            Sinusoid::new(bounds, altitude_m, speed_mps, seed),
            seed,
        )),
        "search_pattern" => {
            let x = center[0] + rng.uniform(-0.1, 0.1) * span[0];
            let y = center[1] + rng.uniform(-0.1, 0.1) * span[1];
            Behavior::SearchPattern(SearchPattern::expanding_square(
                [x, y, altitude_m],
                span_min * 0.05,
                speed_mps,
                altitude_m,
            ))
        }
        "sinusoid" => Behavior::Sinusoid(Sinusoid::new(bounds, altitude_m, speed_mps, seed)),
        "racetrack" => Behavior::Racetrack(Racetrack::new(bounds, altitude_m, speed_mps, seed)),
        "waypoint_patrol" => {
            Behavior::WaypointPatrol(WaypointPatrol::new(bounds, altitude_m, speed_mps, seed))
        }
        "mixed" => {
            let x = center[0] + rng.uniform(-0.1, 0.1) * span[0];
            let y = center[1] + rng.uniform(-0.1, 0.1) * span[1];
            let loiter_center = [x, y, altitude_m];
            let loiter = Loiter::new(loiter_center, span_min * 0.06, speed_mps, true);
            let wp1 = [
                loiter_center[0] + span[0] * 0.15,
                loiter_center[1] + span[1] * 0.1,
                loiter_center[2] + 0.0,
            ];
            let wp2 = [
                loiter_center[0] + span[0] * 0.25,
                loiter_center[1] + -span[1] * 0.05,
                loiter_center[2] + 0.0,
            ];
            Behavior::Mixed(Mixed {
                loiter,
                transit: Transit::new(vec![loiter_center, wp1, wp2], speed_mps),
                search: SearchPattern::expanding_square(
                    wp2,
                    span_min * 0.04,
                    speed_mps,
                    altitude_m,
                ),
            })
        }
        _ => unreachable!("checked against BEHAVIOR_PRESETS"),
    })
}
