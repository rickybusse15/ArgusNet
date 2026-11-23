//! A navigation filter: where an aircraft believes it is, and how well it
//! knows it.
//!
//! An extended Kalman filter over an IMU, a GNSS receiver and a heading
//! sensor. It takes measurements and nothing else, so a recorded flight's
//! measurements run through it as a simulated flight's do.
//!
//! # State
//!
//! | Index | State | Frame, unit |
//! | --- | --- | --- |
//! | 0..3 | position | world, m |
//! | 3..6 | velocity | world, m/s |
//! | 6 | heading | east = 0, counter-clockwise, rad |
//! | 7..10 | accelerometer bias | body (forward, left, up), m/s² |
//! | 10 | gyroscope bias, about up | rad/s |
//! | 11..14 | GNSS correlated position error | world, m |
//! | 14 | heading sensor's correlated error | rad |
//! | 15..17 | attitude sensor's roll and pitch error, about the world's x and y | rad |
//!
//! The last five rows are what keep the filter honest. A GNSS position error does
//! not change much from one fix to the next (atmosphere, orbits,
//! multipath), so averaging fixes does not remove it. A filter that took
//! each fix's error as new would soon claim millimetres. Modelled as a
//! state, its part of the position stays uncertain however many fixes come.
//!
//! The body has a heading and no roll or pitch, as the simulation's has, but
//! its attitude sensor has an error in both. That error matters most without
//! GNSS: an IMU that thinks it is level when it is not reads some of gravity
//! as a sideways acceleration, 1 cm/s² for each milliradian.

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

const N: usize = 17;
const P: usize = 0;
const V: usize = 3;
const PSI: usize = 6;
const BA: usize = 7;
const BG: usize = 10;
const GP: usize = 11;
const HP: usize = 14;
const T: usize = 15;

/// Gravity, m/s².
pub const GRAVITY_MPS2: f64 = 9.80665;

type Vector = SVector<f64, N>;
type Matrix = SMatrix<f64, N, N>;

/// `angle` in `(-π, π]`.
pub fn wrap_angle(angle: f64) -> f64 {
    let turn = std::f64::consts::TAU;
    let a = angle.rem_euclid(turn);
    if a > std::f64::consts::PI {
        a - turn
    } else {
        a
    }
}

/// What the filter is told of its sensors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NavModel {
    /// Accelerometer white noise, m/s² per √Hz.
    pub accel_noise: f64,
    /// Accelerometer bias at power-on, one sigma per axis, m/s².
    pub accel_bias_sigma: f64,
    /// Accelerometer bias drift, m/s² per √s.
    pub accel_bias_walk: f64,
    /// Gyroscope white noise, rad/s per √Hz.
    pub gyro_noise: f64,
    /// Gyroscope bias at power-on, one sigma, rad/s.
    pub gyro_bias_sigma: f64,
    /// Gyroscope bias drift, rad/s per √s.
    pub gyro_bias_walk: f64,
    /// A fix's independent position error, one sigma, horizontal per axis
    /// and vertical, m.
    pub gnss_white_sigma_m: [f64; 2],
    /// The position error fixes share: one sigma, horizontal per axis and
    /// vertical, m, and how long it takes to change, s.
    pub gnss_correlated_sigma_m: [f64; 2],
    pub gnss_correlation_s: f64,
    /// A fix's velocity error, one sigma per axis, m/s.
    pub gnss_velocity_sigma_mps: f64,
    /// A heading's independent error, one sigma, rad.
    pub heading_white_sigma_rad: f64,
    /// The error headings share: one sigma, rad, and how long it takes to
    /// change, s.
    pub heading_correlated_sigma_rad: f64,
    pub heading_correlation_s: f64,
    /// The attitude sensor's roll and pitch error, one sigma per axis, rad,
    /// and how long it takes to change, s.
    pub tilt_sigma_rad: f64,
    pub tilt_correlation_s: f64,
}

impl NavModel {
    pub fn validate(&self) -> Result<(), String> {
        let check = |name: &str, value: f64, zero: bool| {
            let ok = value.is_finite() && (value > 0.0 || (zero && value == 0.0));
            if ok {
                Ok(())
            } else {
                Err(format!("navigation {name} is invalid: {value}"))
            }
        };
        // A sensor with no noise at all makes a filter that cannot be wrong.
        check("accel_noise", self.accel_noise, false)?;
        check("gyro_noise", self.gyro_noise, false)?;
        check("accel_bias_sigma", self.accel_bias_sigma, true)?;
        check("accel_bias_walk", self.accel_bias_walk, true)?;
        check("gyro_bias_sigma", self.gyro_bias_sigma, true)?;
        check("gyro_bias_walk", self.gyro_bias_walk, true)?;
        for k in 0..2 {
            check("gnss_white_sigma_m", self.gnss_white_sigma_m[k], false)?;
            check(
                "gnss_correlated_sigma_m",
                self.gnss_correlated_sigma_m[k],
                true,
            )?;
        }
        check("gnss_correlation_s", self.gnss_correlation_s, false)?;
        check(
            "gnss_velocity_sigma_mps",
            self.gnss_velocity_sigma_mps,
            false,
        )?;
        check(
            "heading_white_sigma_rad",
            self.heading_white_sigma_rad,
            false,
        )?;
        check(
            "heading_correlated_sigma_rad",
            self.heading_correlated_sigma_rad,
            true,
        )?;
        check("heading_correlation_s", self.heading_correlation_s, false)?;
        check("tilt_sigma_rad", self.tilt_sigma_rad, true)?;
        check("tilt_correlation_s", self.tilt_correlation_s, false)
    }

    fn gnss_axis(&self, sigma: [f64; 2], axis: usize) -> f64 {
        sigma[usize::from(axis == 2)]
    }
}

/// What the IMU measured over one interval, with gravity taken out: the
/// mean specific force in the body frame and the mean turn rate about up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuInterval {
    pub accel_body: [f64; 3],
    pub yaw_rate: f64,
    pub dt_s: f64,
}

/// One GNSS fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssFix {
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
}

/// The filter's state, as a checkpoint keeps it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NavFilter {
    model: NavModel,
    x: Vec<f64>,
    /// Row by row.
    p: Vec<f64>,
}

impl NavFilter {
    /// Start from a first fix and heading: nothing else is known.
    pub fn new(model: NavModel, fix: &GnssFix, heading_rad: f64) -> Self {
        let mut x = Vector::zeros();
        let mut p = Matrix::zeros();
        let mut set = |k: usize, sigma: f64| p[(k, k)] = sigma * sigma;
        for axis in 0..3 {
            x[P + axis] = fix.position_m[axis];
            x[V + axis] = fix.velocity_mps[axis];
            let white = model.gnss_axis(model.gnss_white_sigma_m, axis);
            let shared = model.gnss_axis(model.gnss_correlated_sigma_m, axis);
            set(P + axis, white.hypot(shared));
            set(V + axis, model.gnss_velocity_sigma_mps);
            set(BA + axis, model.accel_bias_sigma);
            set(GP + axis, shared);
        }
        x[PSI] = wrap_angle(heading_rad);
        set(
            PSI,
            model
                .heading_white_sigma_rad
                .hypot(model.heading_correlated_sigma_rad),
        );
        set(BG, model.gyro_bias_sigma);
        set(HP, model.heading_correlated_sigma_rad);
        set(T, model.tilt_sigma_rad);
        set(T + 1, model.tilt_sigma_rad);
        // The fix is the position plus the error it shares with later fixes,
        // so the two errors are opposite.
        for axis in 0..3 {
            let shared = model.gnss_axis(model.gnss_correlated_sigma_m, axis);
            p[(P + axis, GP + axis)] = -shared * shared;
            p[(GP + axis, P + axis)] = -shared * shared;
        }
        let shared = model.heading_correlated_sigma_rad;
        p[(PSI, HP)] = -shared * shared;
        p[(HP, PSI)] = -shared * shared;
        let mut filter = Self {
            model,
            x: Vec::new(),
            p: Vec::new(),
        };
        filter.put(&x, &p);
        filter
    }

    fn take(&self) -> (Vector, Matrix) {
        (
            Vector::from_row_slice(&self.x),
            Matrix::from_row_slice(&self.p),
        )
    }

    fn put(&mut self, x: &Vector, p: &Matrix) {
        self.x = x.iter().copied().collect();
        // Kept exactly symmetric, whatever rounding did.
        self.p = (0..N * N)
            .map(|k| 0.5 * (p[(k / N, k % N)] + p[(k % N, k / N)]))
            .collect();
    }

    /// Carry the state over one IMU interval.
    pub fn predict(&mut self, imu: &ImuInterval) {
        let dt = imu.dt_s;
        if dt <= 0.0 {
            return;
        }
        let (mut x, p) = self.take();
        let m = self.model;
        x[PSI] += (imu.yaw_rate - x[BG]) * dt;
        let (sin, cos) = x[PSI].sin_cos();
        let f: [f64; 3] = std::array::from_fn(|k| imu.accel_body[k] - x[BA + k]);
        let accel = [
            cos * f[0] - sin * f[1] - GRAVITY_MPS2 * x[T + 1],
            sin * f[0] + cos * f[1] + GRAVITY_MPS2 * x[T],
            f[2],
        ];
        for k in 0..3 {
            x[V + k] += accel[k] * dt;
            x[P + k] += x[V + k] * dt;
        }
        let keep_gp = (-dt / m.gnss_correlation_s).exp();
        let keep_hp = (-dt / m.heading_correlation_s).exp();
        for k in 0..3 {
            x[GP + k] *= keep_gp;
        }
        x[HP] *= keep_hp;
        let keep_t = (-dt / m.tilt_correlation_s).exp();
        x[T] *= keep_t;
        x[T + 1] *= keep_t;

        // How each error carries over. Velocity first, then position from
        // the new velocity, then heading's part in both.
        let mut step = Matrix::identity();
        // d(velocity)/d(heading after the turn).
        let turned = [
            (-sin * f[0] - cos * f[1]) * dt,
            (cos * f[0] - sin * f[1]) * dt,
            0.0,
        ];
        // The velocity rows, as functions of the old error.
        let mut velocity = SMatrix::<f64, 3, N>::zeros();
        for k in 0..3 {
            velocity[(k, V + k)] = 1.0;
            velocity[(k, PSI)] = turned[k];
            velocity[(k, BG)] = -turned[k] * dt;
        }
        velocity[(0, BA)] = -cos * dt;
        velocity[(0, BA + 1)] = sin * dt;
        velocity[(1, BA)] = -sin * dt;
        velocity[(1, BA + 1)] = -cos * dt;
        velocity[(2, BA + 2)] = -dt;
        velocity[(0, T + 1)] = -GRAVITY_MPS2 * dt;
        velocity[(1, T)] = GRAVITY_MPS2 * dt;
        for row in 0..3 {
            for column in 0..N {
                step[(V + row, column)] = velocity[(row, column)];
                step[(P + row, column)] += velocity[(row, column)] * dt;
            }
        }
        step[(PSI, BG)] = -dt;
        for k in 0..3 {
            step[(GP + k, GP + k)] = keep_gp;
        }
        step[(HP, HP)] = keep_hp;
        step[(T, T)] = keep_t;
        step[(T + 1, T + 1)] = keep_t;

        let mut q = Matrix::zeros();
        // White acceleration noise enters velocity, and through it position.
        let accel_var = m.accel_noise * m.accel_noise * dt;
        // White turn-rate noise enters heading, and through it both.
        let yaw_var = m.gyro_noise * m.gyro_noise * dt;
        let mut yaw_in = Vector::zeros();
        yaw_in[PSI] = 1.0;
        for k in 0..3 {
            q[(V + k, V + k)] += accel_var;
            q[(P + k, P + k)] += accel_var * dt * dt;
            q[(P + k, V + k)] += accel_var * dt;
            q[(V + k, P + k)] += accel_var * dt;
            yaw_in[V + k] = turned[k];
            yaw_in[P + k] = turned[k] * dt;
            q[(BA + k, BA + k)] += m.accel_bias_walk * m.accel_bias_walk * dt;
            let shared = m.gnss_axis(m.gnss_correlated_sigma_m, k);
            q[(GP + k, GP + k)] += shared * shared * (1.0 - keep_gp * keep_gp);
        }
        q += yaw_in * yaw_in.transpose() * yaw_var;
        q[(BG, BG)] += m.gyro_bias_walk * m.gyro_bias_walk * dt;
        let shared = m.heading_correlated_sigma_rad;
        q[(HP, HP)] += shared * shared * (1.0 - keep_hp * keep_hp);
        for k in 0..2 {
            let shared = m.tilt_sigma_rad;
            q[(T + k, T + k)] += shared * shared * (1.0 - keep_t * keep_t);
        }

        x[PSI] = wrap_angle(x[PSI]);
        let p = step * p * step.transpose() + q;
        self.put(&x, &p);
    }

    /// One scalar measurement of the states `h` sums, with variance `r`.
    fn update(&mut self, h: &[usize], innovation: f64, r: f64) {
        let (mut x, p) = self.take();
        let mut row = Vector::zeros();
        for &k in h {
            row[k] = 1.0;
        }
        let ph = p * row;
        let s = row.dot(&ph) + r;
        let gain = ph / s;
        x += gain * innovation;
        x[PSI] = wrap_angle(x[PSI]);
        // Joseph form: symmetric and positive however the gain rounded.
        let keep = Matrix::identity() - gain * row.transpose();
        let p = keep * p * keep.transpose() + gain * gain.transpose() * r;
        self.put(&x, &p);
    }

    pub fn update_gnss(&mut self, fix: &GnssFix) {
        let m = self.model;
        for k in 0..3 {
            let innovation = fix.position_m[k] - self.x[P + k] - self.x[GP + k];
            let sigma = m.gnss_axis(m.gnss_white_sigma_m, k);
            self.update(&[P + k, GP + k], innovation, sigma * sigma);
        }
        for k in 0..3 {
            let innovation = fix.velocity_mps[k] - self.x[V + k];
            let sigma = m.gnss_velocity_sigma_mps;
            self.update(&[V + k], innovation, sigma * sigma);
        }
    }

    pub fn update_heading(&mut self, heading_rad: f64) {
        let innovation = wrap_angle(heading_rad - self.x[PSI] - self.x[HP]);
        let sigma = self.model.heading_white_sigma_rad;
        self.update(&[PSI, HP], innovation, sigma * sigma);
    }

    pub fn position_m(&self) -> [f64; 3] {
        std::array::from_fn(|k| self.x[P + k])
    }

    pub fn velocity_mps(&self) -> [f64; 3] {
        std::array::from_fn(|k| self.x[V + k])
    }

    pub fn heading_rad(&self) -> f64 {
        self.x[PSI]
    }

    /// The position's covariance, m².
    pub fn position_covariance(&self) -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| self.p[(P + i) * N + P + j]))
    }

    /// What the filter makes of the attitude sensor's roll and pitch error.
    pub fn tilt_rad(&self) -> [f64; 2] {
        [self.x[T], self.x[T + 1]]
    }

    /// How uncertain that leaves the tilt, per axis at worst, rad.
    pub fn tilt_sigma_rad(&self) -> f64 {
        (0..2)
            .map(|k| self.p[(T + k) * N + T + k])
            .fold(0.0f64, f64::max)
            .sqrt()
    }

    pub fn heading_sigma_rad(&self) -> f64 {
        self.p[PSI * N + PSI].max(0.0).sqrt()
    }

    /// The vertical one sigma, and the horizontal along its worst axis, m.
    pub fn position_sigmas_m(&self) -> PositionSigmas {
        let c = self.position_covariance();
        let mean = 0.5 * (c[0][0] + c[1][1]);
        let half = 0.5 * (c[0][0] - c[1][1]);
        let worst = mean + half.hypot(c[0][1]);
        PositionSigmas {
            horizontal_m: worst.max(0.0).sqrt(),
            vertical_m: c[2][2].max(0.0).sqrt(),
        }
    }

    /// How surprised the filter would be by `truth`: the position error's
    /// squared length in its own covariance, 3 on average when it is honest.
    /// Evaluation only.
    pub fn position_nees(&self, truth_m: [f64; 3]) -> Option<f64> {
        let c = self.position_covariance();
        let covariance = SMatrix::<f64, 3, 3>::from_fn(|i, j| c[i][j]);
        let error = SVector::<f64, 3>::from_fn(|k, _| self.x[P + k] - truth_m[k]);
        covariance
            .try_inverse()
            .map(|inverse| (error.transpose() * inverse * error)[(0, 0)])
    }
}

/// One-sigma position errors, metres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PositionSigmas {
    /// Along the worst horizontal axis.
    pub horizontal_m: f64,
    pub vertical_m: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> NavModel {
        NavModel {
            accel_noise: 1e-3,
            accel_bias_sigma: 0.02,
            accel_bias_walk: 1e-4,
            gyro_noise: 1e-4,
            gyro_bias_sigma: 1e-4,
            gyro_bias_walk: 1e-6,
            gnss_white_sigma_m: [0.01, 0.02],
            gnss_correlated_sigma_m: [0.01, 0.02],
            gnss_correlation_s: 60.0,
            gnss_velocity_sigma_mps: 0.05,
            heading_white_sigma_rad: 2e-3,
            heading_correlated_sigma_rad: 1e-3,
            heading_correlation_s: 120.0,
            tilt_sigma_rad: 4e-4,
            tilt_correlation_s: 60.0,
        }
    }

    /// A path with turns, climbs and abrupt changes of velocity.
    fn path(k: usize) -> ([f64; 3], f64) {
        let t = k as f64 * 0.25;
        let turn = 0.3 * (t * 0.2).sin() + if (k / 40).is_multiple_of(2) { 0.0 } else { 2.0 };
        let position = [
            12.0 * t + 20.0 * (t * 0.3).sin(),
            30.0 * (t * 0.2).cos() + if k % 80 > 40 { 3.0 } else { 0.0 },
            80.0 + 5.0 * (t * 0.1).sin(),
        ];
        (position, turn)
    }

    /// The interval an IMU without error would report between two steps.
    fn exact(k: usize, velocity: &mut [f64; 3]) -> ImuInterval {
        let dt = 0.25;
        let ((a, psi_a), (b, psi_b)) = (path(k - 1), path(k));
        let next: [f64; 3] = std::array::from_fn(|i| (b[i] - a[i]) / dt);
        let accel: [f64; 3] = std::array::from_fn(|i| (next[i] - velocity[i]) / dt);
        *velocity = next;
        let (sin, cos) = psi_b.sin_cos();
        ImuInterval {
            accel_body: [
                cos * accel[0] + sin * accel[1],
                -sin * accel[0] + cos * accel[1],
                accel[2],
            ],
            yaw_rate: wrap_angle(psi_b - psi_a) / dt,
            dt_s: dt,
        }
    }

    #[test]
    fn exact_measurements_are_followed_exactly() {
        let (start, psi) = path(0);
        let fix = GnssFix {
            position_m: start,
            velocity_mps: [0.0; 3],
        };
        let mut filter = NavFilter::new(model(), &fix, psi);
        let mut velocity = [0.0; 3];
        for k in 1..400 {
            filter.predict(&exact(k, &mut velocity));
            let (at, psi) = path(k);
            let p = filter.position_m();
            for i in 0..3 {
                assert!((p[i] - at[i]).abs() < 1e-6, "step {k}: {p:?} {at:?}");
            }
            assert!(wrap_angle(filter.heading_rad() - psi).abs() < 1e-9);
        }
    }

    /// The carried covariance is the one the equations imply: it matches the
    /// step's own differences.
    #[test]
    fn the_jacobian_is_the_step_s_own() {
        let fix = GnssFix {
            position_m: [1.0, 2.0, 3.0],
            velocity_mps: [4.0, -3.0, 0.5],
        };
        let imu = ImuInterval {
            accel_body: [3.0, -7.0, 1.0],
            yaw_rate: 0.7,
            dt_s: 0.25,
        };
        // Two filters a small step apart in one state, with that state alone
        // uncertain, must end where the carried covariance says.
        for k in 0..N {
            let h = 1e-6;
            let mut quiet = model();
            quiet.accel_noise = 1e-12;
            quiet.gyro_noise = 1e-12;
            quiet.accel_bias_walk = 0.0;
            quiet.gyro_bias_walk = 0.0;
            quiet.gnss_correlated_sigma_m = [0.0; 2];
            quiet.heading_correlated_sigma_rad = 0.0;
            quiet.tilt_sigma_rad = 0.0;
            let mut base = NavFilter::new(quiet, &fix, 0.4);
            base.x[BA] = 0.1;
            base.x[BG] = 0.01;
            base.p = vec![0.0; N * N];
            base.p[k * N + k] = 1.0;
            let mut moved = base.clone();
            moved.x[k] += h;
            base.predict(&imu);
            moved.predict(&imu);
            let slope: Vec<f64> = (0..N).map(|r| (moved.x[r] - base.x[r]) / h).collect();
            // One uncertain state: the carried covariance is the slopes'
            // product, signs included.
            for row in 0..N {
                for column in 0..N {
                    let carried = base.p[row * N + column];
                    let expected = slope[row] * slope[column];
                    assert!(
                        (carried - expected).abs() < 1e-5,
                        "state {k} into {row}, {column}: {carried}, not {expected}"
                    );
                }
            }
        }
    }

    /// The error fixes share is not averaged away, however many come.
    #[test]
    fn fixes_do_not_average_away_the_error_they_share() {
        let fix = GnssFix {
            position_m: [0.0; 3],
            velocity_mps: [0.0; 3],
        };
        let hover = ImuInterval {
            accel_body: [0.0; 3],
            yaw_rate: 0.0,
            dt_s: 0.25,
        };
        let mut shared = model();
        shared.gnss_correlated_sigma_m = [1.0, 2.0];
        shared.gnss_correlation_s = 1e6;
        let mut filter = NavFilter::new(shared, &fix, 0.0);
        for _ in 0..2000 {
            filter.predict(&hover);
            filter.update_gnss(&fix);
            filter.update_heading(0.0);
        }
        let sigma = filter.position_sigmas_m();
        assert!((sigma.horizontal_m - 1.0).abs() < 0.05, "{sigma:?}");
        assert!((sigma.vertical_m - 2.0).abs() < 0.05, "{sigma:?}");
    }

    #[test]
    fn without_fixes_the_uncertainty_grows_and_with_them_it_comes_back() {
        let fix = GnssFix {
            position_m: [0.0; 3],
            velocity_mps: [0.0; 3],
        };
        let hover = ImuInterval {
            accel_body: [0.0; 3],
            yaw_rate: 0.0,
            dt_s: 0.25,
        };
        let mut filter = NavFilter::new(model(), &fix, 0.0);
        for _ in 0..40 {
            filter.predict(&hover);
            filter.update_gnss(&fix);
            filter.update_heading(0.0);
        }
        let held = filter.position_sigmas_m().horizontal_m;
        let mut last = held;
        for _ in 0..240 {
            filter.predict(&hover);
            let now = filter.position_sigmas_m().horizontal_m;
            assert!(now > last);
            last = now;
        }
        assert!(last > 1.0, "a minute on the IMU alone: {last} m");
        for _ in 0..40 {
            filter.predict(&hover);
            filter.update_gnss(&fix);
        }
        assert!(filter.position_sigmas_m().horizontal_m < 2.0 * held);
    }

    /// A tilt error is a sideways acceleration, and most of what a minute
    /// without GNSS costs.
    #[test]
    fn gravity_leaks_through_a_tilt_error() {
        let fix = GnssFix {
            position_m: [0.0; 3],
            velocity_mps: [0.0; 3],
        };
        let hover = ImuInterval {
            accel_body: [0.0; 3],
            yaw_rate: 0.0,
            dt_s: 0.25,
        };
        let after = |tilt_sigma_rad| {
            let model = NavModel {
                tilt_sigma_rad,
                accel_bias_sigma: 1e-4,
                ..model()
            };
            let mut filter = NavFilter::new(model, &fix, 0.0);
            for _ in 0..40 {
                filter.predict(&hover);
                filter.update_gnss(&fix);
            }
            for _ in 0..240 {
                filter.predict(&hover);
            }
            filter.position_sigmas_m()
        };
        let (level, tilted) = (after(0.0), after(2e-3));
        assert!(tilted.horizontal_m > 3.0 * level.horizontal_m);
        assert!((tilted.vertical_m - level.vertical_m).abs() < 0.2);
    }

    #[test]
    fn a_checkpoint_goes_on_as_the_filter_would_have() {
        let (start, psi) = path(0);
        let fix = GnssFix {
            position_m: start,
            velocity_mps: [0.0; 3],
        };
        let mut filter = NavFilter::new(model(), &fix, psi);
        let mut velocity = [0.0; 3];
        for k in 1..20 {
            filter.predict(&exact(k, &mut velocity));
        }
        let text = serde_json::to_string(&filter).unwrap();
        let mut restored: NavFilter = serde_json::from_str(&text).unwrap();
        assert_eq!(filter, restored);
        let imu = exact(20, &mut velocity);
        filter.predict(&imu);
        restored.predict(&imu);
        assert_eq!(filter, restored);
    }

    #[test]
    fn invalid_models_are_refused() {
        assert!(model().validate().is_ok());
        for bad in [
            NavModel {
                accel_noise: 0.0,
                ..model()
            },
            NavModel {
                gnss_correlation_s: 0.0,
                ..model()
            },
            NavModel {
                heading_white_sigma_rad: f64::NAN,
                ..model()
            },
            NavModel {
                gnss_correlated_sigma_m: [0.1, -0.1],
                ..model()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
