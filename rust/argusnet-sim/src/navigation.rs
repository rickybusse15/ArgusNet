//! Where each aircraft believes it is, and placing what it senses there.
//!
//! Until this, every return was placed from the true position: the map had
//! no position error, so its quality could not have any either. Here the
//! truth makes an IMU's, a GNSS receiver's and a heading sensor's
//! measurements, with their noise, biases and shared errors; a navigation
//! filter ([`argusnet_localization::nav`]) makes a pose and its covariance
//! from the measurements alone; and the scan is placed by that pose.
//!
//! ```text
//! truth ──► NavSensors ──► NavFilter ──► PoseEstimate ──► the map
//!   └────────► sensing from the true pose ─────────┘  (placed by the estimate)
//! ```
//!
//! # Placing a scan
//!
//! A scan is made from the true pose and its points are given in the world.
//! The estimated pose puts the same body-frame measurement somewhere else:
//! [`PoseError`] is the rigid motion from one to the other, and
//! [`PoseError::place`] applies it to everything in the scan. That is
//! exactly georeferencing each ray from the estimate, the pod's own measured
//! attitude and nominal mount included.
//!
//! # Flying the estimate
//!
//! An aircraft's autopilot holds its *estimate* on its path, so its true path
//! is off by the error ([`FlownPositions`]). Positions in the run are where
//! each aircraft believes it is; physics, sensing and the replay's `nodes`
//! are where it is. An error it has only now learned of, as when GNSS comes
//! back, it flies off at a declared rate; until it has, it is not where the
//! planners have it either.
//!
//! # What this does not do
//!
//! - Control holds the estimate on the planned path to within a step, as if
//!   by a perfect autopilot: no gusts, no lag but [`Navigation::fly`]'s.
//! - The oracle's wires (`--measured-wires` off) are named by the world and
//!   stay where they are.
//! - Bearing observations of targets still come from the true position.
//! - Every sensor number is assumed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use argusnet_localization::nav::{
    wrap_angle, GnssFix, ImuInterval, NavFilter, NavModel, GRAVITY_MPS2,
};
use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::pod::Quat;
use crate::range_scan::RangeScan;

/// How an aircraft's navigation sensors behave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NavigationConfig {
    /// Named in the replay: `rtk`, `standalone`, or a file's own.
    pub id: String,
    /// The sensors' errors. The filter is told the same numbers, so a run is
    /// consistent by construction; what it is told being wrong is a
    /// different experiment.
    pub sensors: NavModel,
    /// Seconds between GNSS fixes; no finer than the tick.
    pub gnss_period_s: f64,
    /// How fast an aircraft flies off a position error it has just learned
    /// of, m/s. Absent flies the true position, whatever the estimate says:
    /// what navigation would cost the map alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_mps: Option<f64>,
}

impl NavigationConfig {
    /// A survey-grade receiver with a fixed carrier-phase solution and a
    /// dual-antenna heading.
    pub fn rtk() -> Self {
        Self {
            id: "rtk".into(),
            sensors: NavModel {
                accel_noise: 1.0e-3,
                accel_bias_sigma: 0.02,
                accel_bias_walk: 1.0e-4,
                gyro_noise: 1.0e-4,
                gyro_bias_sigma: 5.0e-5,
                gyro_bias_walk: 1.0e-6,
                gnss_white_sigma_m: [0.008, 0.015],
                gnss_correlated_sigma_m: [0.010, 0.015],
                gnss_correlation_s: 60.0,
                gnss_velocity_sigma_mps: 0.03,
                heading_white_sigma_rad: 2.0e-3,
                heading_correlated_sigma_rad: 1.0e-3,
                heading_correlation_s: 120.0,
                tilt_sigma_rad: 4.4e-4,
                tilt_correlation_s: 60.0,
            },
            gnss_period_s: 1.0,
            correction_mps: Some(CORRECTION_MPS),
        }
    }

    /// A receiver alone, uncorrected, with a magnetic heading.
    pub fn standalone() -> Self {
        Self {
            id: "standalone".into(),
            sensors: NavModel {
                accel_noise: 2.0e-3,
                accel_bias_sigma: 0.05,
                accel_bias_walk: 3.0e-4,
                gyro_noise: 2.0e-4,
                gyro_bias_sigma: 2.0e-4,
                gyro_bias_walk: 3.0e-6,
                gnss_white_sigma_m: [0.5, 0.8],
                gnss_correlated_sigma_m: [1.2, 2.0],
                gnss_correlation_s: 300.0,
                gnss_velocity_sigma_mps: 0.05,
                heading_white_sigma_rad: 1.7e-2,
                heading_correlated_sigma_rad: 1.7e-2,
                heading_correlation_s: 300.0,
                tilt_sigma_rad: 3.5e-3,
                tilt_correlation_s: 60.0,
            },
            gnss_period_s: 1.0,
            correction_mps: Some(CORRECTION_MPS),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.sensors.validate()?;
        if let Some(rate) = self.correction_mps {
            if !(rate.is_finite() && rate > 0.0) {
                return Err(format!("navigation correction_mps is invalid: {rate}"));
            }
        }
        if !(self.gnss_period_s.is_finite() && self.gnss_period_s > 0.0) {
            return Err(format!(
                "navigation gnss_period_s is invalid: {}",
                self.gnss_period_s
            ));
        }
        Ok(())
    }
}

/// How fast an aircraft flies off a position error, m/s, unless told.
const CORRECTION_MPS: f64 = 4.0;
/// Over its launch pad an aircraft is held to it by what it sees of it, as a
/// precision landing is: fully within this far of it in plan and at its
/// height, not at all this much further out or this high above it, metres.
const PAD_HELD_M: f64 = 10.0;
const PAD_FREE_M: f64 = 30.0;
const PAD_FREE_HEIGHT_M: f64 = 15.0;

/// Where each aircraft really is, for where it believes it is.
///
/// Everything that flies an aircraft (its controller, its overrides, its
/// separation, its safety monitor) works on where it believes it is: that is
/// all it has. What it *is* differs by the error it does not know it has.
/// Physics and sensing ask here.
#[derive(Debug, Clone, Default)]
pub struct FlownPositions(Arc<RwLock<BTreeMap<String, Flown>>>);

/// Where an aircraft is, and how well it knows: the second is the aircraft's
/// own, the first is not.
#[derive(Debug, Clone, Copy, Default)]
struct Flown {
    offset: FlownOffset,
    /// Its `[horizontal, vertical]` one sigma, metres.
    sigma_m: [f64; 2],
    /// Seconds since its last fix.
    fix_age_s: f64,
}

/// How many sigma an aircraft keeps clear by.
const CLEAR_SIGMAS: f64 = 2.0;
/// The most its being lost widens what it keeps clear of, metres, horizontal
/// and vertical: past it nothing near is clear, and it has to fly somewhere.
const MAX_MARGIN_M: [f64; 2] = [40.0, 20.0];

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct FlownOffset {
    /// True less believed, away from its pad, metres.
    offset_m: [f64; 3],
    pad_m: [f64; 3],
}

impl FlownOffset {
    fn at(&self, believed_m: [f64; 3]) -> [f64; 3] {
        let out = (believed_m[0] - self.pad_m[0]).hypot(believed_m[1] - self.pad_m[1]);
        let free = ((out - PAD_HELD_M) / (PAD_FREE_M - PAD_HELD_M))
            .max((believed_m[2] - self.pad_m[2]) / PAD_FREE_HEIGHT_M)
            .clamp(0.0, 1.0);
        self.offset_m.map(|v| v * free)
    }
}

impl FlownPositions {
    /// Where `node_id` is when it believes it is at `believed_m`.
    pub fn truth(&self, node_id: &str, believed_m: [f64; 3]) -> [f64; 3] {
        let offsets = self.0.read().expect("offsets lock");
        let offset = offsets
            .get(node_id)
            .map_or([0.0; 3], |flown| flown.offset.at(believed_m));
        std::array::from_fn(|k| believed_m[k] + offset[k])
    }

    /// How much further than usual, `[horizontal, vertical]` metres,
    /// `node_id` keeps clear for not knowing quite where it is: nothing about
    /// where it really is.
    pub fn margin_m(&self, node_id: &str) -> [f64; 2] {
        let flown = self.0.read().expect("offsets lock");
        let sigma = flown.get(node_id).map_or([0.0; 2], |flown| flown.sigma_m);
        std::array::from_fn(|k| (CLEAR_SIGMAS * sigma[k]).min(MAX_MARGIN_M[k]))
    }

    /// Whether `node_id` knows where it is, sideways, to no better than
    /// `sigma_m`, and how long since its last fix if so.
    pub fn lost_for_s(&self, node_id: &str, sigma_m: f64) -> Option<f64> {
        let flown = self.0.read().expect("offsets lock");
        flown
            .get(node_id)
            .filter(|flown| flown.sigma_m[0] > sigma_m)
            .map(|flown| flown.fix_age_s)
    }

    fn set(&self, node_id: &str, offset: FlownOffset, sigma_m: [f64; 2], fix_age_s: f64) {
        self.0.write().expect("offsets lock").insert(
            node_id.to_string(),
            Flown {
                offset,
                sigma_m,
                fix_age_s,
            },
        );
    }

    /// `node_id` this far off wherever it believes it is, its pad far away.
    #[cfg(test)]
    pub(crate) fn always_unsure(node_id: &str, sigma_m: [f64; 2]) -> Self {
        let positions = Self::default();
        positions.set(node_id, FlownOffset::default(), sigma_m, 0.0);
        positions
    }

    #[cfg(test)]
    pub(crate) fn always_off(node_id: &str, offset_m: [f64; 3]) -> Self {
        let positions = Self::default();
        positions.set(
            node_id,
            FlownOffset {
                offset_m,
                pad_m: [1.0e9, 1.0e9, 0.0],
            },
            [0.0; 2],
            0.0,
        );
        positions
    }

    fn clear(&self) {
        self.0.write().expect("offsets lock").clear();
    }
}

/// How well a pose was known when a scan was placed by it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseQuality {
    /// The position's one sigma: vertical, and along the worst horizontal
    /// axis, metres.
    pub vertical_sigma_m: f64,
    pub horizontal_sigma_m: f64,
    /// One sigma, rad. Tilt is per axis and includes a pod's own measured
    /// attitude.
    pub heading_sigma_rad: f64,
    pub tilt_sigma_rad: f64,
}

impl PoseQuality {
    /// What the pose alone leaves uncertain of a point `offset_m` from the
    /// sensor: its vertical one sigma, and its horizontal, metres. Tilt
    /// lifts a point by how far out it is and moves it by how far down;
    /// heading swings it by how far out.
    pub fn at(&self, offset_m: [f64; 3]) -> [f64; 2] {
        let out = offset_m[0].hypot(offset_m[1]);
        let vertical = self.vertical_sigma_m.hypot(out * self.tilt_sigma_rad);
        let horizontal = self
            .horizontal_sigma_m
            .hypot(out * self.heading_sigma_rad)
            .hypot(offset_m[2] * self.tilt_sigma_rad);
        [vertical, horizontal]
    }
}

/// An aircraft's pose as its navigation has it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseEstimate {
    pub node_id: String,
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub heading_rad: f64,
    pub vertical_sigma_m: f64,
    pub horizontal_sigma_m: f64,
    pub heading_sigma_rad: f64,
    /// Seconds since the last GNSS fix.
    pub fix_age_s: f64,
}

/// The rigid motion from an aircraft's true pose to its estimated one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseError {
    from_m: [f64; 3],
    to_m: [f64; 3],
    rotation: Quat,
    velocity_error_mps: [f64; 3],
}

impl PoseError {
    /// From a true position to an estimated one, turned `heading_rad` about
    /// up and tilted `tilt_rad` about the world's x and y.
    pub fn new(
        from_m: [f64; 3],
        to_m: [f64; 3],
        heading_rad: f64,
        tilt_rad: [f64; 2],
        velocity_error_mps: [f64; 3],
    ) -> Self {
        let rotation = Quat::from_ypr(heading_rad, tilt_rad[1], tilt_rad[0]);
        Self {
            from_m,
            to_m,
            rotation,
            velocity_error_mps,
        }
    }

    pub fn direction(&self, d: [f64; 3]) -> [f64; 3] {
        self.rotation.rotate(d)
    }

    pub fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let d = self.direction(std::array::from_fn(|k| p[k] - self.from_m[k]));
        std::array::from_fn(|k| self.to_m[k] + d[k])
    }

    /// Move everything `scan` reports to where the estimate puts it. Spans
    /// the oracle names ([`RangeScan::wires`]) stay where the world has them.
    pub fn place(&self, scan: &mut RangeScan) {
        scan.origin_m = self.point(scan.origin_m);
        for hit in &mut scan.returns {
            // A ray fired later in the capture is placed from where the
            // estimated velocity had carried the aircraft by then. A
            // capture's sweep is carried level (`crate::pod`).
            let late = hit.acquired_offset_s.unwrap_or(0.0);
            let [x, y, _] = self.velocity_error_mps.map(|v| v * late);
            for point in std::iter::once(&mut hit.point_m).chain(hit.acquired_from_m.as_mut()) {
                let p = self.point(*point);
                *point = [p[0] + x, p[1] + y, p[2]];
            }
        }
        for point in &mut scan.wire_points {
            *point = self.point(*point);
        }
        for line in &mut scan.wire_lines {
            line.from_dir = self.direction(line.from_dir);
            line.to_dir = self.direction(line.to_dir);
        }
        if let Some(pod) = scan.pod.as_mut() {
            pod.boresight_world = self.direction(pod.boresight_world);
            if let Some(pose) = pod
                .images
                .as_mut()
                .and_then(|images| images.visible_pose.as_mut())
            {
                pose.origin_m = self.point(pose.origin_m);
                pose.world_camera = self.rotation.then(pose.world_camera);
            }
        }
    }
}

/// One aircraft's true sensor errors and its filter. The errors are the
/// simulation's; only the filter's side is what the aircraft knows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Aircraft {
    timestamp_s: f64,
    position_m: [f64; 3],
    velocity_mps: [f64; 3],
    heading_rad: f64,
    accel_bias: [f64; 3],
    gyro_bias: f64,
    gnss_error_m: [f64; 3],
    heading_error_rad: f64,
    tilt_rad: [f64; 2],
    last_fix_s: Option<f64>,
    next_fix_s: f64,
    filter: Option<NavFilter>,
    /// What it flies off by: none when it flies the true position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flown: Option<FlownOffset>,
}

/// Every aircraft's navigation, as a checkpoint keeps it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NavigationState {
    aircraft: BTreeMap<String, Aircraft>,
}

/// Every aircraft's navigation.
#[derive(Debug, Clone)]
pub struct Navigation {
    config: NavigationConfig,
    seed: u32,
    /// The platforms with no GNSS now.
    gnss_out: BTreeSet<String>,
    state: NavigationState,
    /// Shared with what physics and the replay ask.
    flown: FlownPositions,
}

fn sigmas_of(filter: &NavFilter) -> [f64; 2] {
    let sigmas = filter.position_sigmas_m();
    [sigmas.horizontal_m, sigmas.vertical_m]
}

/// An aircraft this slow holds its heading.
const MIN_HEADING_SPEED_MPS: f64 = 0.5;

impl Navigation {
    pub fn new(config: NavigationConfig, seed: u32) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            seed,
            gnss_out: BTreeSet::new(),
            state: NavigationState::default(),
            flown: FlownPositions::default(),
        })
    }

    pub fn config(&self) -> &NavigationConfig {
        &self.config
    }

    pub fn set_gnss_out(&mut self, nodes: BTreeSet<String>) {
        self.gnss_out = nodes;
    }

    pub fn reset(&mut self) {
        self.state = NavigationState::default();
        self.flown.clear();
    }

    /// Where each aircraft is for where it believes it is: shared, and
    /// current as of the last [`Self::fly`].
    pub fn flown_positions(&self) -> FlownPositions {
        self.flown.clone()
    }

    /// Whether aircraft fly their estimates ([`NavigationConfig`]).
    pub fn flies_estimates(&self) -> bool {
        self.config.correction_mps.is_some()
    }

    /// Where `node_id` really was at the last step; what the replay shows.
    pub fn true_position_m(&self, node_id: &str) -> Option<[f64; 3]> {
        Some(self.state.aircraft.get(node_id)?.position_m)
    }

    /// Set how far off `node_id` flies from now: by the error its estimate
    /// has, which holding the estimate on its path puts into its true path,
    /// approached no faster than it can fly it off.
    pub fn fly(&mut self, node_id: &str, dt_s: f64) {
        let Some(aircraft) = self.state.aircraft.get_mut(node_id) else {
            return;
        };
        let Some(filter) = aircraft.filter.as_ref() else {
            return;
        };
        let sigmas = sigmas_of(filter);
        let fix_age_s = aircraft
            .last_fix_s
            .map_or(0.0, |at| aircraft.timestamp_s - at);
        let (Some(rate), Some(flown)) = (self.config.correction_mps, aircraft.flown.as_mut())
        else {
            self.flown
                .set(node_id, FlownOffset::default(), sigmas, fix_age_s);
            return;
        };
        let estimate = filter.position_m();
        let step: [f64; 3] =
            std::array::from_fn(|k| (aircraft.position_m[k] - estimate[k]) - flown.offset_m[k]);
        let length = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        let share = if length > rate * dt_s {
            rate * dt_s / length
        } else {
            1.0
        };
        for (held, step) in flown.offset_m.iter_mut().zip(step) {
            *held += step * share;
        }
        self.flown.set(node_id, *flown, sigmas, fix_age_s);
    }

    pub fn state(&self) -> &NavigationState {
        &self.state
    }

    pub fn restore(&mut self, state: NavigationState) {
        self.state = state;
        self.flown.clear();
        for (node_id, aircraft) in &self.state.aircraft {
            if let Some(filter) = &aircraft.filter {
                let offset = aircraft.flown.unwrap_or_default();
                let fix_age_s = aircraft
                    .last_fix_s
                    .map_or(0.0, |at| aircraft.timestamp_s - at);
                self.flown
                    .set(node_id, offset, sigmas_of(filter), fix_age_s);
            }
        }
    }

    /// Carry `node_id` to its true pose at `timestamp_s`: its sensors
    /// measure the way there, and its filter follows.
    pub fn step(
        &mut self,
        node_id: &str,
        tick: u64,
        timestamp_s: f64,
        position_m: [f64; 3],
        velocity_mps: [f64; 3],
    ) {
        let model = self.config.sensors;
        let seed = self.seed;
        let key = |stream: &'static str| {
            DeterministicRng::keyed(
                seed,
                &[
                    SeedPart::Str(node_id),
                    SeedPart::Str(stream),
                    SeedPart::Int(tick as i64),
                ],
            )
        };
        let axis = |pair: [f64; 2], k: usize| pair[usize::from(k == 2)];
        let heading = |held: f64, v: [f64; 3]| {
            if v[0].hypot(v[1]) > MIN_HEADING_SPEED_MPS {
                v[1].atan2(v[0])
            } else {
                held
            }
        };
        let aircraft = match self.state.aircraft.get_mut(node_id) {
            Some(aircraft) => {
                let dt = timestamp_s - aircraft.timestamp_s;
                if dt <= 0.0 {
                    return;
                }
                // What the IMU felt: the true change of motion between the
                // two steps, as its own errors make it.
                let mut imu = key("nav-imu");
                let mut walk = key("nav-walk");
                let flown: [f64; 3] =
                    std::array::from_fn(|k| (position_m[k] - aircraft.position_m[k]) / dt);
                let turned = heading(aircraft.heading_rad, flown);
                let (sin, cos) = turned.sin_cos();
                let mut accel: [f64; 3] =
                    std::array::from_fn(|k| (flown[k] - aircraft.velocity_mps[k]) / dt);
                // Believing itself level, it reads some of gravity sideways.
                accel[0] += GRAVITY_MPS2 * aircraft.tilt_rad[1];
                accel[1] -= GRAVITY_MPS2 * aircraft.tilt_rad[0];
                let body = [
                    cos * accel[0] + sin * accel[1],
                    -sin * accel[0] + cos * accel[1],
                    accel[2],
                ];
                let interval = ImuInterval {
                    accel_body: std::array::from_fn(|k| {
                        body[k]
                            + aircraft.accel_bias[k]
                            + imu.normal(0.0, model.accel_noise / dt.sqrt())
                    }),
                    yaw_rate: wrap_angle(turned - aircraft.heading_rad) / dt
                        + aircraft.gyro_bias
                        + imu.normal(0.0, model.gyro_noise / dt.sqrt()),
                    dt_s: dt,
                };
                if let Some(filter) = aircraft.filter.as_mut() {
                    filter.predict(&interval);
                }
                aircraft.timestamp_s = timestamp_s;
                aircraft.position_m = position_m;
                aircraft.velocity_mps = flown;
                aircraft.heading_rad = turned;
                // The errors drift on.
                let drift = |value: &mut f64, sigma: f64, over_s: f64, draw: f64| {
                    let keep = (-dt / over_s).exp();
                    *value = *value * keep + sigma * (1.0 - keep * keep).sqrt() * draw;
                };
                for k in 0..3 {
                    aircraft.accel_bias[k] += walk.normal(0.0, model.accel_bias_walk * dt.sqrt());
                    drift(
                        &mut aircraft.gnss_error_m[k],
                        axis(model.gnss_correlated_sigma_m, k),
                        model.gnss_correlation_s,
                        walk.normal(0.0, 1.0),
                    );
                }
                aircraft.gyro_bias += walk.normal(0.0, model.gyro_bias_walk * dt.sqrt());
                drift(
                    &mut aircraft.heading_error_rad,
                    model.heading_correlated_sigma_rad,
                    model.heading_correlation_s,
                    walk.normal(0.0, 1.0),
                );
                for tilt in &mut aircraft.tilt_rad {
                    drift(
                        tilt,
                        model.tilt_sigma_rad,
                        model.tilt_correlation_s,
                        walk.normal(0.0, 1.0),
                    );
                }
                aircraft
            }
            None => {
                let mut rng = DeterministicRng::keyed(
                    seed,
                    &[SeedPart::Str(node_id), SeedPart::Str("nav-power-on")],
                );
                let fresh = Aircraft {
                    timestamp_s,
                    position_m,
                    velocity_mps,
                    heading_rad: heading(0.0, velocity_mps),
                    accel_bias: std::array::from_fn(|_| rng.normal(0.0, model.accel_bias_sigma)),
                    gyro_bias: rng.normal(0.0, model.gyro_bias_sigma),
                    gnss_error_m: std::array::from_fn(|k| {
                        rng.normal(0.0, axis(model.gnss_correlated_sigma_m, k))
                    }),
                    heading_error_rad: rng.normal(0.0, model.heading_correlated_sigma_rad),
                    tilt_rad: std::array::from_fn(|_| rng.normal(0.0, model.tilt_sigma_rad)),
                    last_fix_s: None,
                    next_fix_s: timestamp_s,
                    filter: None,
                    flown: self.config.correction_mps.map(|_| FlownOffset {
                        offset_m: [0.0; 3],
                        pad_m: position_m,
                    }),
                };
                self.state
                    .aircraft
                    .entry(node_id.to_string())
                    .or_insert(fresh)
            }
        };
        let mut fixes = key("nav-fix");
        let heading_read = aircraft.heading_rad
            + aircraft.heading_error_rad
            + fixes.normal(0.0, model.heading_white_sigma_rad);
        if let Some(filter) = aircraft.filter.as_mut() {
            filter.update_heading(heading_read);
        }
        // Drawn every step, so an outage changes no draw after it.
        let fix = GnssFix {
            position_m: std::array::from_fn(|k| {
                aircraft.position_m[k]
                    + aircraft.gnss_error_m[k]
                    + fixes.normal(0.0, axis(model.gnss_white_sigma_m, k))
            }),
            velocity_mps: std::array::from_fn(|k| {
                aircraft.velocity_mps[k] + fixes.normal(0.0, model.gnss_velocity_sigma_mps)
            }),
        };
        if timestamp_s + 1e-9 < aircraft.next_fix_s {
            return;
        }
        while aircraft.next_fix_s <= timestamp_s + 1e-9 {
            aircraft.next_fix_s += self.config.gnss_period_s;
        }
        if self.gnss_out.contains(node_id) {
            return;
        }
        match aircraft.filter.as_mut() {
            Some(filter) => filter.update_gnss(&fix),
            None => aircraft.filter = Some(NavFilter::new(model, &fix, heading_read)),
        }
        aircraft.last_fix_s = Some(timestamp_s);
    }

    /// `node_id`'s pose as its navigation has it; `None` before its first
    /// fix.
    pub fn estimate(&self, node_id: &str) -> Option<PoseEstimate> {
        let aircraft = self.state.aircraft.get(node_id)?;
        let filter = aircraft.filter.as_ref()?;
        let sigmas = filter.position_sigmas_m();
        Some(PoseEstimate {
            node_id: node_id.to_string(),
            position_m: filter.position_m(),
            velocity_mps: filter.velocity_mps(),
            heading_rad: filter.heading_rad(),
            vertical_sigma_m: sigmas.vertical_m,
            horizontal_sigma_m: sigmas.horizontal_m,
            heading_sigma_rad: filter.heading_sigma_rad(),
            fix_age_s: aircraft.timestamp_s - aircraft.last_fix_s?,
        })
    }

    /// Every aircraft's estimate, in node order.
    pub fn estimates(&self) -> Vec<PoseEstimate> {
        self.state
            .aircraft
            .keys()
            .filter_map(|node| self.estimate(node))
            .collect()
    }

    /// Place `scan` by its aircraft's estimate, and say how well that is
    /// known. An aircraft with no pose yet has nowhere to put a return:
    /// `false`, and the scan is not evidence.
    /// `attitude_sigma_rad` is a pod's own measured attitude.
    pub fn place(&self, scan: &mut RangeScan, attitude_sigma_rad: f64) -> bool {
        let Some((aircraft, filter)) = self
            .state
            .aircraft
            .get(&scan.node_id)
            .and_then(|aircraft| Some((aircraft, aircraft.filter.as_ref()?)))
        else {
            return false;
        };
        let (believed, tilt) = (filter.velocity_mps(), filter.tilt_rad());
        PoseError::new(
            aircraft.position_m,
            filter.position_m(),
            wrap_angle(filter.heading_rad() - aircraft.heading_rad),
            // What is left of the tilt error once the filter's own idea of it
            // is taken out.
            std::array::from_fn(|k| aircraft.tilt_rad[k] - tilt[k]),
            std::array::from_fn(|k| believed[k] - aircraft.velocity_mps[k]),
        )
        .place(scan);
        let sigmas = filter.position_sigmas_m();
        scan.pose = Some(PoseQuality {
            vertical_sigma_m: sigmas.vertical_m,
            horizontal_sigma_m: sigmas.horizontal_m,
            heading_sigma_rad: filter.heading_sigma_rad(),
            tilt_sigma_rad: filter.tilt_sigma_rad().hypot(attitude_sigma_rad),
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_scan::{RangeReturn, ReturnClass, ScanSensor};

    const DT_S: f64 = 0.25;

    /// A survey: legs with a turn at each end, a climb, and a hold.
    fn flown(k: u64) -> ([f64; 3], [f64; 3]) {
        let at = |k: f64| {
            let t = k * DT_S;
            let leg = (t / 40.0).floor();
            let along = t - leg * 40.0;
            let x = if leg % 2.0 == 0.0 {
                along
            } else {
                40.0 - along
            } * 12.0;
            [x, leg * 30.0, 70.0 + 8.0 * (t * 0.05).sin()]
        };
        let (now, before) = (at(k as f64), at(k as f64 - 1.0));
        (now, std::array::from_fn(|i| (now[i] - before[i]) / DT_S))
    }

    fn run(navigation: &mut Navigation, node: &str, ticks: std::ops::Range<u64>) {
        for k in ticks {
            let (position, velocity) = flown(k);
            navigation.step(node, k, k as f64 * DT_S, position, velocity);
        }
    }

    /// Over many runs the position error is what the filter says it is: its
    /// squared length in the filter's own covariance averages 3, with and
    /// without GNSS.
    fn mean_nees(config: &NavigationConfig, ticks: u64, out: Option<u64>) -> f64 {
        let runs = 400u32;
        let total: f64 = (0..runs)
            .map(|seed| {
                let mut navigation = Navigation::new(config.clone(), seed).unwrap();
                let split = out.unwrap_or(ticks);
                run(&mut navigation, "drone", 0..split);
                navigation.set_gnss_out(BTreeSet::from(["drone".to_string()]));
                run(&mut navigation, "drone", split..ticks);
                let aircraft = &navigation.state.aircraft["drone"];
                aircraft
                    .filter
                    .as_ref()
                    .unwrap()
                    .position_nees(aircraft.position_m)
                    .unwrap()
            })
            .sum();
        total / f64::from(runs)
    }

    #[test]
    fn the_filter_s_errors_are_what_it_says_they_are() {
        // The mean of 400 draws of chi-squared with 3 degrees of freedom is
        // 3 ± 0.12, one sigma; declared before the run, four of them.
        for config in [NavigationConfig::rtk(), NavigationConfig::standalone()] {
            for (ticks, out) in [(600, None), (600, Some(480))] {
                let mean = mean_nees(&config, ticks, out);
                assert!(
                    (mean - 3.0).abs() < 0.5,
                    "{} over {ticks} ticks, GNSS out from {out:?}: {mean}",
                    config.id
                );
            }
        }
    }

    #[test]
    fn an_outage_is_known_about_and_ends() {
        let mut navigation = Navigation::new(NavigationConfig::rtk(), 7).unwrap();
        run(&mut navigation, "drone", 0..400);
        let held = navigation.estimate("drone").unwrap();
        assert!(held.horizontal_sigma_m < 0.05, "{held:?}");
        navigation.set_gnss_out(BTreeSet::from(["drone".to_string()]));
        run(&mut navigation, "drone", 400..640);
        let lost = navigation.estimate("drone").unwrap();
        assert!(lost.horizontal_sigma_m > 0.5, "{lost:?}");
        assert!((lost.fix_age_s - 60.0).abs() < 1.1, "{lost:?}");
        navigation.set_gnss_out(BTreeSet::new());
        run(&mut navigation, "drone", 640..700);
        let back = navigation.estimate("drone").unwrap();
        assert!(back.horizontal_sigma_m < 0.05, "{back:?}");
    }

    #[test]
    fn no_fix_yet_is_no_pose_and_no_evidence() {
        let mut navigation = Navigation::new(NavigationConfig::rtk(), 7).unwrap();
        navigation.set_gnss_out(BTreeSet::from(["drone".to_string()]));
        run(&mut navigation, "drone", 0..8);
        assert!(navigation.estimate("drone").is_none());
        let mut scan = scan_from([0.0, 0.0, 70.0]);
        assert!(!navigation.place(&mut scan, 0.0));
    }

    /// Each aircraft's errors are its own, and the run's seed's.
    #[test]
    fn errors_differ_by_aircraft_and_seed_and_nothing_else() {
        let position = |seed, node: &str| {
            let mut navigation = Navigation::new(NavigationConfig::rtk(), seed).unwrap();
            run(&mut navigation, node, 0..40);
            navigation.estimate(node).unwrap().position_m
        };
        assert_eq!(position(7, "a"), position(7, "a"));
        assert_ne!(position(7, "a"), position(7, "b"));
        assert_ne!(position(7, "a"), position(8, "a"));
    }

    #[test]
    fn a_checkpoint_goes_on_as_the_run_would_have() {
        let mut navigation = Navigation::new(NavigationConfig::standalone(), 11).unwrap();
        run(&mut navigation, "drone", 0..50);
        let text = serde_json::to_string(navigation.state()).unwrap();
        let mut restored = Navigation::new(NavigationConfig::standalone(), 11).unwrap();
        restored.restore(serde_json::from_str(&text).unwrap());
        run(&mut navigation, "drone", 50..90);
        run(&mut restored, "drone", 50..90);
        assert_eq!(navigation.state(), restored.state());
    }

    fn scan_from(origin_m: [f64; 3]) -> RangeScan {
        RangeScan {
            node_id: "drone".into(),
            timestamp_s: 0.0,
            origin_m,
            returns: vec![
                RangeReturn {
                    point_m: [origin_m[0] + 30.0, origin_m[1], 0.0],
                    class: ReturnClass::Ground,
                    acquired_offset_s: None,
                    acquired_from_m: None,
                },
                RangeReturn {
                    point_m: [origin_m[0], origin_m[1] + 40.0, 5.0],
                    class: ReturnClass::Structure,
                    acquired_offset_s: Some(0.2),
                    acquired_from_m: Some(origin_m),
                },
            ],
            misses: 0,
            wires: Vec::new(),
            sensor: ScanSensor::RangeScanner,
            pod: None,
            wire_points: vec![[origin_m[0] + 10.0, origin_m[1], 20.0]],
            wire_lines: Vec::new(),
            pose: None,
        }
    }

    /// A scan placed by a pose that is turned, tilted and elsewhere is the
    /// body's own measurement put there: ranges from the sensor are kept.
    #[test]
    fn placing_a_scan_moves_it_as_a_body() {
        let origin = [100.0, 200.0, 70.0];
        let error = PoseError::new(
            origin,
            [101.0, 198.0, 70.5],
            std::f64::consts::FRAC_PI_2,
            [0.0, 0.0],
            [0.5, 0.0, 0.0],
        );
        let mut scan = scan_from(origin);
        error.place(&mut scan);
        let near = |a: [f64; 3], b: [f64; 3]| {
            assert!((0..3).all(|k| (a[k] - b[k]).abs() < 1e-9), "{a:?} {b:?}");
        };
        near(scan.origin_m, [101.0, 198.0, 70.5]);
        // 30 m east of it is now 30 m north of where it thinks it is.
        near(scan.returns[0].point_m, [101.0, 228.0, 0.5]);
        // Fired 0.2 s in, from where 0.5 m/s of velocity error had carried it.
        near(scan.returns[1].point_m, [61.0 + 0.1, 198.0, 5.5]);
        near(
            scan.returns[1].acquired_from_m.unwrap(),
            [101.0 + 0.1, 198.0, 70.5],
        );
        near(scan.wire_points[0], [101.0, 208.0, 20.5]);

        let tilted = PoseError::new(origin, origin, 0.0, [0.01, 0.0], [0.0; 3]);
        let mut scan = scan_from(origin);
        tilted.place(&mut scan);
        // Tilted about east: what is 40 m north of it rises.
        let lift = scan.returns[1].point_m[2] - 5.0;
        assert!((lift - 40.0 * 0.01f64.sin()).abs() < 0.01, "{lift}");
    }

    #[test]
    fn a_pose_s_uncertainty_at_a_point_is_where_it_acts() {
        let pose = PoseQuality {
            vertical_sigma_m: 0.03,
            horizontal_sigma_m: 0.04,
            heading_sigma_rad: 0.01,
            tilt_sigma_rad: 0.002,
        };
        assert_eq!(pose.at([0.0, 0.0, 0.0]), [0.03, 0.04]);
        let [vertical, horizontal] = pose.at([30.0, 40.0, -80.0]);
        assert!((vertical - 0.03f64.hypot(0.1)).abs() < 1e-12);
        assert!((horizontal - 0.04f64.hypot(0.5).hypot(0.16)).abs() < 1e-12);
    }

    /// Over its pad an aircraft is held to it; away from it, off by its
    /// error, and in between in part, without a jump.
    #[test]
    fn an_aircraft_is_held_to_its_pad_and_off_everywhere_else() {
        let flown = FlownPositions::default();
        assert_eq!(flown.truth("drone", [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        flown.set(
            "drone",
            FlownOffset {
                offset_m: [4.0, 0.0, -2.0],
                pad_m: [100.0, 100.0, 10.0],
            },
            [3.0, 0.5],
            0.0,
        );
        assert_eq!(flown.margin_m("drone"), [6.0, 1.0]);
        assert_eq!(flown.margin_m("other"), [0.0; 2]);
        assert_eq!(
            flown.truth("drone", [100.0, 100.0, 10.0]),
            [100.0, 100.0, 10.0]
        );
        assert_eq!(
            flown.truth("drone", [100.0, 100.0, 25.0]),
            [104.0, 100.0, 23.0]
        );
        assert_eq!(
            flown.truth("drone", [140.0, 100.0, 10.0]),
            [144.0, 100.0, 8.0]
        );
        assert_eq!(flown.truth("other", [0.0; 3]), [0.0; 3]);
        let mut last = flown.truth("drone", [100.0, 100.0, 10.0])[2] - 10.0;
        for k in 1..=300 {
            let z = 10.0 + 0.1 * f64::from(k);
            let off = flown.truth("drone", [100.0, 100.0, z])[2] - z;
            assert!((off - last).abs() < 0.02, "{z}: {last} to {off}");
            last = off;
        }
    }

    /// Flown off by its error, an aircraft's estimate stays on its path; and
    /// an error it has just learned of is flown off no faster than declared.
    #[test]
    fn an_aircraft_holds_its_estimate_on_its_path() {
        let config = NavigationConfig::rtk();
        let rate = config.correction_mps.unwrap();
        let mut navigation = Navigation::new(config, 5).unwrap();
        let flown_positions = navigation.flown_positions();
        let mut previous: Option<[f64; 3]> = None;
        let mut worst_out: f64 = 0.0;
        for k in 0..1200u64 {
            let lost = (200..800).contains(&k);
            navigation.set_gnss_out(if lost {
                BTreeSet::from(["drone".to_string()])
            } else {
                BTreeSet::new()
            });
            let (believed, velocity) = flown(k);
            let truth = flown_positions.truth("drone", believed);
            navigation.step("drone", k, k as f64 * DT_S, truth, velocity);
            navigation.fly("drone", DT_S);
            assert_eq!(navigation.true_position_m("drone"), Some(truth));
            let off: [f64; 3] = std::array::from_fn(|i| truth[i] - believed[i]);
            let size = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if let Some(before) = previous {
                let moved = size(std::array::from_fn(|i| off[i] - before[i]));
                assert!(moved <= rate * DT_S + 1e-9, "tick {k}: {moved}");
            }
            previous = Some(off);
            let estimate = navigation.estimate("drone").unwrap().position_m;
            let held = size(std::array::from_fn(|i| estimate[i] - believed[i]));
            worst_out = worst_out.max(size(off));
            // Except while flying off what an outage left it with.
            if !(800..=1100).contains(&k) {
                assert!(
                    held < 0.6,
                    "tick {k}: the estimate is {held} m off its path"
                );
            }
        }
        assert!(worst_out > 5.0, "150 s without GNSS: {worst_out} m");
    }
}
