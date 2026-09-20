//! The 15-state EKF VIO prediction step.
//!
//! State layout, as the reference orders it:
//! `rotation(0:3) velocity(3:6) position(6:9) accel_bias(9:12) gyro_bias(12:15)`.
//!
//! # Scope
//!
//! Prediction only. `docs/migration/VIO_CONTRACT.md` records three defects in
//! the reference's *update* step that P4.4 requires changing rather than
//! copying -- chiefly a scalar fallback that manufactures a measurement out of
//! the matched-feature count, moving the position 23 m and *shrinking* the
//! covariance when every feature has failed the outlier gate. Porting the
//! update alongside a parity fixture would enshrine that, so it is deliberately
//! separate work with its own contract tests.
//!
//! The prediction path has no such problem: it is a deterministic function of
//! the buffered IMU, so it is checked value-for-value against recorded Python.

use nalgebra::{Matrix3, Matrix3x1, SMatrix, SVector};

/// 15x15 covariance.
pub type Covariance = SMatrix<f64, 15, 15>;
/// The 15-element state vector.
pub type StateVector = SVector<f64, 15>;

/// Gravity in the world frame, matching `_GRAVITY_WORLD`.
pub const GRAVITY_WORLD: [f64; 3] = [0.0, 0.0, 9.806_65];

/// One inertial measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuMeasurement {
    pub timestamp_s: f64,
    pub angular_velocity: [f64; 3],
    pub linear_acceleration: [f64; 3],
}

/// Noise parameters. Defaults match the reference's constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VioNoise {
    pub accel_noise_std: f64,
    pub gyro_noise_std: f64,
    pub accel_bias_drift: f64,
    pub gyro_bias_drift: f64,
}

impl Default for VioNoise {
    fn default() -> Self {
        Self {
            accel_noise_std: 0.05,
            gyro_noise_std: 0.005,
            accel_bias_drift: 0.001,
            gyro_bias_drift: 0.0001,
        }
    }
}

/// Rodrigues' formula, matching `_rotvec_to_matrix`.
///
/// The `angle < 1e-9` short circuit is the reference's, and it matters: below
/// that threshold it returns exactly the identity rather than a series
/// expansion, so the two agree bit-for-bit at small angles instead of merely
/// closely.
pub fn rotvec_to_matrix(rvec: &Matrix3x1<f64>) -> Matrix3<f64> {
    let angle = rvec.norm();
    if angle < 1e-9 {
        return Matrix3::identity();
    }
    let axis = rvec / angle;
    let k = Matrix3::new(
        0.0, -axis[2], axis[1], //
        axis[2], 0.0, -axis[0], //
        -axis[1], axis[0], 0.0,
    );
    Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * (k * k)
}

/// The prediction half of the filter.
#[derive(Debug, Clone)]
pub struct VioPredictor {
    state: StateVector,
    covariance: Covariance,
    timestamp_s: f64,
    noise: VioNoise,
    initialized: bool,
}

impl VioPredictor {
    pub fn new(noise: VioNoise) -> Self {
        Self {
            state: StateVector::zeros(),
            covariance: Covariance::identity() * 0.01,
            timestamp_s: 0.0,
            noise,
            initialized: false,
        }
    }

    /// Seed the filter at a known position.
    ///
    /// The initial covariance is the reference's block diagonal: rotation
    /// 0.01, velocity 0.1, position 1.0, accel bias 0.001, gyro bias 0.01.
    /// Note position starts the *least* certain of the three pose blocks.
    pub fn initialize(&mut self, position: [f64; 3], timestamp_s: f64) {
        self.state = StateVector::zeros();
        self.state[6] = position[0];
        self.state[7] = position[1];
        self.state[8] = position[2];
        let diag: [f64; 15] = [
            0.01, 0.01, 0.01, // rotation
            0.1, 0.1, 0.1, // velocity
            1.0, 1.0, 1.0, // position
            0.001, 0.001, 0.001, // accel bias
            0.01, 0.01, 0.01, // gyro bias
        ];
        self.covariance = Covariance::from_diagonal(&SVector::from_column_slice(&diag));
        self.timestamp_s = timestamp_s;
        self.initialized = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn state(&self) -> &StateVector {
        &self.state
    }

    pub fn covariance(&self) -> &Covariance {
        &self.covariance
    }

    pub fn timestamp_s(&self) -> f64 {
        self.timestamp_s
    }

    pub fn position(&self) -> [f64; 3] {
        [self.state[6], self.state[7], self.state[8]]
    }

    pub fn velocity(&self) -> [f64; 3] {
        [self.state[3], self.state[4], self.state[5]]
    }

    /// The confidence signal `sim.py` actually reads: the trace of the
    /// position block of the covariance.
    pub fn position_covariance_trace(&self) -> f64 {
        self.covariance[(6, 6)] + self.covariance[(7, 7)] + self.covariance[(8, 8)]
    }

    /// Propagate to `timestamp_s` using the supplied measurements.
    ///
    /// `measurements` need not be sorted; they are ordered here as the
    /// reference sorts its buffer. A measurement at or before the current
    /// time, or after `timestamp_s`, is skipped -- and skipped means *not
    /// consumed*, so it cannot silently vanish from a later interval.
    ///
    /// Returns `false` without changing anything if the filter is
    /// uninitialised or time runs backwards, which is the reference's
    /// behaviour for both.
    pub fn predict_to(&mut self, measurements: &[ImuMeasurement], timestamp_s: f64) -> bool {
        if !self.initialized {
            return false;
        }
        let dt_total = timestamp_s - self.timestamp_s;
        if dt_total < 0.0 {
            return false;
        }

        let mut ordered: Vec<&ImuMeasurement> = measurements.iter().collect();
        ordered.sort_by(|a, b| {
            a.timestamp_s
                .partial_cmp(&b.timestamp_s)
                .expect("IMU timestamps must be ordered; NaN is rejected on entry")
        });

        let gravity = Matrix3x1::from_column_slice(&GRAVITY_WORLD);
        let mut t_prev = self.timestamp_s;

        for meas in ordered {
            if meas.timestamp_s <= t_prev || meas.timestamp_s > timestamp_s {
                continue;
            }
            let dt = meas.timestamp_s - t_prev;

            let gyro = Matrix3x1::from_column_slice(&meas.angular_velocity)
                - Matrix3x1::new(self.state[12], self.state[13], self.state[14]);
            let accel_body = Matrix3x1::from_column_slice(&meas.linear_acceleration)
                - Matrix3x1::new(self.state[9], self.state[10], self.state[11]);
            let rotation =
                rotvec_to_matrix(&Matrix3x1::new(self.state[0], self.state[1], self.state[2]));
            let accel_world = rotation * accel_body - gravity;

            // Order matters: position uses the OLD velocity, so it is updated
            // before velocity is. Swapping these two lines integrates the new
            // velocity over the interval it was not yet travelling at.
            for axis in 0..3 {
                self.state[6 + axis] +=
                    self.state[3 + axis] * dt + 0.5 * accel_world[axis] * dt * dt;
            }
            for axis in 0..3 {
                self.state[3 + axis] += accel_world[axis] * dt;
            }
            for axis in 0..3 {
                self.state[axis] += gyro[axis] * dt;
            }

            let mut f = Covariance::identity();
            f[(6, 3)] = dt;
            f[(7, 4)] = dt;
            f[(8, 5)] = dt;
            for row in 0..3 {
                for col in 0..3 {
                    f[(3 + row, 9 + col)] = -rotation[(row, col)] * dt;
                    f[(6 + row, 9 + col)] = -0.5 * rotation[(row, col)] * dt * dt;
                }
                f[(row, 12 + row)] = -dt;
            }

            let q_a = self.noise.accel_noise_std.powi(2) * dt;
            let q_g = self.noise.gyro_noise_std.powi(2) * dt;
            let q_ba = self.noise.accel_bias_drift.powi(2) * dt;
            let q_bg = self.noise.gyro_bias_drift.powi(2) * dt;
            // Note the position block is `q_a * dt`, not `q_a` -- an extra
            // factor of dt the reference applies only there.
            let q_diag: [f64; 15] = [
                q_g,
                q_g,
                q_g,
                q_a,
                q_a,
                q_a,
                q_a * dt,
                q_a * dt,
                q_a * dt,
                q_ba,
                q_ba,
                q_ba,
                q_bg,
                q_bg,
                q_bg,
            ];
            let q = Covariance::from_diagonal(&SVector::from_column_slice(&q_diag));
            self.covariance = f * self.covariance * f.transpose() + q;
            t_prev = meas.timestamp_s;
        }

        // Any interval no measurement covered is closed with a single
        // constant-velocity step. Its process noise differs from the loop's:
        // the position block is `accel_noise^2 * dt^2` here, against
        // `accel_noise^2 * dt * dt` above -- the same value, reached by a
        // different expression, and worth not "simplifying" into one.
        if t_prev < timestamp_s && dt_total > 0.0 {
            let dt = timestamp_s - t_prev;
            for axis in 0..3 {
                self.state[6 + axis] += self.state[3 + axis] * dt;
            }
            let mut f = Covariance::identity();
            f[(6, 3)] = dt;
            f[(7, 4)] = dt;
            f[(8, 5)] = dt;
            let q_diag: [f64; 15] = [
                self.noise.gyro_noise_std.powi(2) * dt,
                self.noise.gyro_noise_std.powi(2) * dt,
                self.noise.gyro_noise_std.powi(2) * dt,
                self.noise.accel_noise_std.powi(2) * dt,
                self.noise.accel_noise_std.powi(2) * dt,
                self.noise.accel_noise_std.powi(2) * dt,
                self.noise.accel_noise_std.powi(2) * dt * dt,
                self.noise.accel_noise_std.powi(2) * dt * dt,
                self.noise.accel_noise_std.powi(2) * dt * dt,
                self.noise.accel_bias_drift.powi(2) * dt,
                self.noise.accel_bias_drift.powi(2) * dt,
                self.noise.accel_bias_drift.powi(2) * dt,
                self.noise.gyro_bias_drift.powi(2) * dt,
                self.noise.gyro_bias_drift.powi(2) * dt,
                self.noise.gyro_bias_drift.powi(2) * dt,
            ];
            let q = Covariance::from_diagonal(&SVector::from_column_slice(&q_diag));
            self.covariance = f * self.covariance * f.transpose() + q;
        }

        self.timestamp_s = timestamp_s;
        true
    }
}

// ---------------------------------------------------------------------------
// The optical-flow update
// ---------------------------------------------------------------------------

/// One tracked image feature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualFeature {
    pub feature_id: u64,
    pub u: f64,
    pub v: f64,
}

/// Why an update carried no information.
///
/// P4.4 requires that a failed estimator not manufacture confidence, so each
/// of these leaves the state and covariance exactly as prediction left them —
/// which is to say less certain than before, never more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoInformation {
    /// Fewer than two features were tracked from the previous frame.
    TooFewMatches,
    /// Features matched, but every one failed the outlier gate.
    ///
    /// This is the case the reference answers by fabricating a measurement
    /// from the matched-feature count. See `docs/migration/VIO_CONTRACT.md`:
    /// measured, it moved the position 23 m and *shrank* the covariance while
    /// rejecting every observation it was given.
    AllFeaturesGatedOut,
    /// The innovation covariance could not be inverted.
    SingularInnovation,
}

/// The result of one image update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateOutcome {
    Applied {
        features_matched: usize,
        features_used: usize,
    },
    /// Nothing was applied. The caller can distinguish "no information" from
    /// "confident" — which the reference cannot, because it reports both as a
    /// smaller covariance.
    NoInformation {
        features_matched: usize,
        reason: NoInformation,
    },
}

/// How a frame's timestamp relates to the filter's clock.
///
/// P4.4 asks for invalid and stale pose behaviour to be defined explicitly.
/// The reference has one guard, `dt_total < 0`, which returns early *before*
/// refreshing the feature store — so an out-of-order frame is dropped and the
/// next frame matches against a store two frames old while dividing by its own
/// one-frame interval. `dt_total == 0` is not caught at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTime {
    /// Strictly newer than the filter's clock: the ordinary case.
    Advancing,
    /// Exactly equal. Predicted flow is identically zero, so the residual is
    /// the raw displacement. Accepted, and named, rather than silently taken.
    Repeated,
    /// Older than the filter's clock.
    OutOfOrder,
}

/// The estimator's answer, as the caller sees it.
///
/// Mirrors `VIOState`, and the field names carry a trap worth stating. The
/// state vector is laid out
/// `rotation(0:3) velocity(3:6) position(6:9) accel_bias(9:12) gyro_bias(12:15)`
/// — **accel before gyro** — while `VIOState` lists `bias_gyro` before
/// `bias_accel` in its constructor. The two orderings are opposite, and a port
/// that reads them positionally swaps the biases silently: both are small
/// 3-vectors near zero, so nothing crashes and the drift is subtly wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct VioState {
    /// World-to-body rotation, from the rotation vector at `x[0:3]`.
    pub rotation: Matrix3<f64>,
    /// `x[6:9]`.
    pub position_m: [f64; 3],
    /// `x[3:6]`.
    pub velocity_mps: [f64; 3],
    /// `x[12:15]`. Note the index: gyro bias is the LAST block.
    pub bias_gyro: [f64; 3],
    /// `x[9:12]`. Accel bias comes first in the vector.
    pub bias_accel: [f64; 3],
    pub covariance: Covariance,
    pub timestamp_s: f64,
    /// How many features were matched into this update.
    ///
    /// The reference reports two different values for the same field depending
    /// on which path produced the state: `process_image` sets the real matched
    /// count, while `get_state` hardcodes **zero** even though the filter is
    /// tracking features. Preserved rather than unified — a caller comparing
    /// the two paths would otherwise see a difference this port invented.
    pub tracked_feature_count: usize,
}

/// What one image frame produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageOutcome {
    pub state: VioState,
    /// `None` when the frame was out of order, so nothing was applied and the
    /// state is the filter's previous one rather than a new estimate. That is
    /// a different thing from `Some(NoInformation)`, which means the frame was
    /// processed and carried nothing usable.
    pub update: Option<UpdateOutcome>,
}

/// Why the estimator has no pose to give.
///
/// P4.4 requires that invalid and stale pose behaviour be defined explicitly
/// rather than left to a `None` a caller has to guess at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoPose {
    /// `initialize` has not been called. The reference returns `None`, and the
    /// whole point of P4.4's "no manufactured confidence on a failed
    /// estimator" rule is that this must not be answered with a zero pose at
    /// some default covariance — which would read as a confident estimate at
    /// the origin.
    Uninitialized,
}

/// The gate threshold: chi-squared at 2 degrees of freedom, 0.997.
pub const MAHALANOBIS_GATE: f64 = 11.8;

/// Camera and feature-noise parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowConfig {
    pub focal_length_px: f64,
    pub feature_noise_px: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            focal_length_px: 500.0,
            feature_noise_px: 1.5,
        }
    }
}

/// The feature store carried between frames.
#[derive(Debug, Clone, Default)]
pub struct FeatureStore {
    previous: std::collections::BTreeMap<u64, VisualFeature>,
}

impl FeatureStore {
    pub fn len(&self) -> usize {
        self.previous.len()
    }

    pub fn is_empty(&self) -> bool {
        self.previous.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.previous.keys().copied()
    }

    /// Replace the store with this frame's features.
    ///
    /// Called on **every** frame the filter accepts or rejects for timing
    /// reasons, which is the fix for the reference's stale-store defect: there,
    /// an out-of-order frame returns before this happens, so the next frame
    /// matches against features two frames old.
    pub fn replace(&mut self, features: &[VisualFeature]) {
        self.previous.clear();
        for feature in features {
            self.previous.insert(feature.feature_id, *feature);
        }
    }
}

impl VioPredictor {
    /// The current estimate, or why there is none.
    ///
    /// `Err(NoPose::Uninitialized)` where the reference returns `None`. A
    /// `Result` rather than an `Option` so the reason is carried rather than
    /// reconstructed at the call site.
    ///
    /// `tracked_feature_count` is **zero** here, matching `get_state`. The
    /// count from an update is reported by [`Self::observe_image`] instead.
    pub fn state_snapshot(&self) -> Result<VioState, NoPose> {
        if !self.is_initialized() {
            return Err(NoPose::Uninitialized);
        }
        Ok(self.snapshot_with_matches(0))
    }

    fn snapshot_with_matches(&self, tracked_feature_count: usize) -> VioState {
        let x = self.state();
        let block = |i: usize| [x[i], x[i + 1], x[i + 2]];
        VioState {
            rotation: rotvec_to_matrix(&Matrix3x1::new(x[0], x[1], x[2])),
            position_m: block(6),
            velocity_mps: block(3),
            bias_gyro: block(12),
            bias_accel: block(9),
            covariance: *self.covariance(),
            timestamp_s: self.timestamp_s(),
            tracked_feature_count,
        }
    }

    /// Return to the uninitialized state, as `reset` does.
    ///
    /// The reference zeroes the state and sets `P = I * 0.01` — a *small*
    /// covariance on a state that means nothing. That combination is only safe
    /// because `_initialized` is false and every accessor checks it first, so
    /// the confident-looking covariance is unreachable. This keeps the same
    /// values so a restarted filter behaves identically, and keeps the flag as
    /// the thing that gates them.
    pub fn reset(&mut self) {
        *self = Self::new(self.noise);
    }

    /// One image frame: predict over the buffered IMU, then apply optical
    /// flow. The composition `process_image` performs.
    ///
    /// # The three ways this answers
    ///
    /// * **Uninitialized** → `Err(NoPose::Uninitialized)`. The reference's
    ///   `return None`.
    /// * **Out of order** (`timestamp_s` older than the filter's clock) → the
    ///   *current* state, unchanged: no prediction, no update, the clock does
    ///   not move. The reference does this by returning `get_state()` before
    ///   touching anything, and the consequence is easy to miss — a late frame
    ///   is not merely ignored, it yields a state whose `timestamp_s` is the
    ///   filter's, not the frame's. A caller stamping that pose with the
    ///   frame's own time would date it wrongly.
    /// * **Otherwise** → predict, update, and report the outcome alongside the
    ///   new state, so "moved a lot because it saw a lot" and "moved a lot
    ///   because it saw nothing" stay distinguishable. That distinction is the
    ///   whole of P4.4's no-manufactured-confidence rule.
    ///
    /// `Repeated` frames (`dt == 0`) are *not* out of order and do run: the
    /// reference gates on `dt_total < 0`, not `<= 0`.
    pub fn observe_image(
        &mut self,
        features: &[VisualFeature],
        store: &mut FeatureStore,
        measurements: &[ImuMeasurement],
        timestamp_s: f64,
        config: &FlowConfig,
    ) -> Result<ImageOutcome, NoPose> {
        if !self.is_initialized() {
            return Err(NoPose::Uninitialized);
        }
        if self.classify(timestamp_s) == FrameTime::OutOfOrder {
            return Ok(ImageOutcome {
                state: self.snapshot_with_matches(0),
                update: None,
            });
        }

        let dt_total = timestamp_s - self.timestamp_s();
        self.predict_to(measurements, timestamp_s);
        let outcome = self.update_optical_flow(features, store, dt_total, config);
        store.replace(features);

        let matched = match outcome {
            UpdateOutcome::Applied {
                features_matched, ..
            } => features_matched,
            UpdateOutcome::NoInformation {
                features_matched, ..
            } => features_matched,
        };
        Ok(ImageOutcome {
            state: self.snapshot_with_matches(matched),
            update: Some(outcome),
        })
    }

    /// Classify a frame timestamp against the filter's clock.
    pub fn classify(&self, timestamp_s: f64) -> FrameTime {
        let dt = timestamp_s - self.timestamp_s();
        if dt > 0.0 {
            FrameTime::Advancing
        } else if dt == 0.0 {
            FrameTime::Repeated
        } else {
            FrameTime::OutOfOrder
        }
    }

    /// Apply an optical-flow update from features matched against `store`.
    ///
    /// The measurement model is the reference's: world-frame horizontal
    /// velocity drives image flow, so the predicted displacement of every
    /// feature is `(vx * dt, vy * dt)` divided by the focal length. Each
    /// feature gets a 2x2 Mahalanobis gate before entering the batch.
    ///
    /// **Where this deliberately differs from the reference:** when fewer than
    /// two features survive the gate, nothing is applied. The reference instead
    /// synthesises `y = (matched / (min_features * 2) - 0.5) * 0.1` and treats
    /// it as a position measurement, which both moves the estimate and reduces
    /// its covariance on the strength of a feature count. P4.4 forbids exactly
    /// that, so this returns [`NoInformation::AllFeaturesGatedOut`] and leaves
    /// prediction's (larger) covariance standing.
    pub fn update_optical_flow(
        &mut self,
        features: &[VisualFeature],
        store: &FeatureStore,
        dt_total: f64,
        config: &FlowConfig,
    ) -> UpdateOutcome {
        let f_px = config.focal_length_px;
        let r_var = (config.feature_noise_px / f_px).powi(2);

        let current: std::collections::BTreeMap<u64, VisualFeature> =
            features.iter().map(|f| (f.feature_id, *f)).collect();

        let mut rows: Vec<(SVector<f64, 15>, f64)> = Vec::new();
        let mut matched = 0usize;

        // BTreeMap iteration is ordered, so the batch is assembled in feature-id
        // order on every run. The reference iterates a Python set, whose order
        // is not part of its contract; making it deterministic here costs
        // nothing and removes a source of run-to-run drift.
        for (id, previous) in store.previous.iter() {
            let Some(observed) = current.get(id) else {
                continue;
            };
            matched += 1;

            let delta_u = (observed.u - previous.u) / f_px;
            let delta_v = (observed.v - previous.v) / f_px;
            let predicted_u = self.state[3] * dt_total / f_px;
            let predicted_v = self.state[4] * dt_total / f_px;
            let residual = [delta_u - predicted_u, delta_v - predicted_v];

            let mut h_u = SVector::<f64, 15>::zeros();
            h_u[3] = dt_total / f_px;
            let mut h_v = SVector::<f64, 15>::zeros();
            h_v[4] = dt_total / f_px;

            // 2x2 innovation covariance for this feature alone.
            let s00 = h_u.dot(&(self.covariance * h_u)) + r_var;
            let s01 = h_u.dot(&(self.covariance * h_v));
            let s10 = h_v.dot(&(self.covariance * h_u));
            let s11 = h_v.dot(&(self.covariance * h_v)) + r_var;
            let determinant = s00 * s11 - s01 * s10;
            if determinant.abs() < f64::EPSILON || !determinant.is_finite() {
                // The reference calls np.linalg.solve here, which raises.
                // Skipping the feature keeps one degenerate feature from
                // taking down the whole frame.
                continue;
            }
            let mahalanobis = (residual[0] * (s11 * residual[0] - s01 * residual[1])
                + residual[1] * (-s10 * residual[0] + s00 * residual[1]))
                / determinant;
            if !mahalanobis.is_finite() || mahalanobis > MAHALANOBIS_GATE {
                continue;
            }

            rows.push((h_u, residual[0]));
            rows.push((h_v, residual[1]));
        }

        if matched < 2 {
            return UpdateOutcome::NoInformation {
                features_matched: matched,
                reason: NoInformation::TooFewMatches,
            };
        }
        if rows.len() < 4 {
            // Fewer than two features survived the gate.
            return UpdateOutcome::NoInformation {
                features_matched: matched,
                reason: NoInformation::AllFeaturesGatedOut,
            };
        }

        // Batch update. Every H row is zero except one column, so H P H^T is
        // assembled directly rather than by forming a (2K x 15) matrix.
        let count = rows.len();
        let mut innovation = nalgebra::DMatrix::<f64>::zeros(count, count);
        let mut residuals = nalgebra::DVector::<f64>::zeros(count);
        let mut gain_left = nalgebra::DMatrix::<f64>::zeros(15, count);
        for (i, (h_i, z_i)) in rows.iter().enumerate() {
            residuals[i] = *z_i;
            let p_h = self.covariance * h_i;
            for row in 0..15 {
                gain_left[(row, i)] = p_h[row];
            }
            for (j, (h_j, _)) in rows.iter().enumerate() {
                innovation[(i, j)] = h_j.dot(&p_h);
            }
            innovation[(i, i)] += r_var;
        }

        let Some(inverse) = innovation.try_inverse() else {
            return UpdateOutcome::NoInformation {
                features_matched: matched,
                reason: NoInformation::SingularInnovation,
            };
        };

        let gain = &gain_left * inverse;
        let correction = &gain * &residuals;
        for row in 0..15 {
            self.state[row] += correction[row];
        }

        // P = (I - K H) P, with K H assembled column by column.
        let mut kh = Covariance::zeros();
        for (i, (h_i, _)) in rows.iter().enumerate() {
            for row in 0..15 {
                for col in 0..15 {
                    kh[(row, col)] += gain[(row, i)] * h_i[col];
                }
            }
        }
        self.covariance = (Covariance::identity() - kh) * self.covariance;

        UpdateOutcome::Applied {
            features_matched: matched,
            features_used: count / 2,
        }
    }
}
