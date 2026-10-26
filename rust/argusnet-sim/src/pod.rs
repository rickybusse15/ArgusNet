//! The moving sensor pod (SR2; SENSING_REALISM_IMPLEMENTATION_GUIDE §4, §8.4):
//! a 3-axis gimbal, the pose chain from body to sensor, the pointing the
//! planner asks for and the pointing the pod actually holds, and the pod's
//! fixed rectangular LiDAR.
//!
//! # Attitude, not joint angles
//!
//! The maintainer set no joint limits, so every attitude is reachable and the
//! gimbal is modelled by the attitude it holds, a quaternion in the body
//! frame, rather than by three joint angles. That also avoids the singularity
//! a yaw–pitch–roll decomposition has exactly at nadir, where mapping points
//! most. The pod turns toward its commanded attitude along the shortest
//! rotation at [`GimbalEnvelope::max_rate_rad_s`]. It is **settled** once it
//! has held within [`GimbalEnvelope::settle_tolerance_rad`] for
//! [`GimbalEnvelope::settle_s`]. A capture taken unsettled is flagged, and
//! the map refuses it.
//!
//! # Commanded is never measured; truth is never the estimate
//!
//! Three attitudes are kept apart:
//!
//! - **commanded**: what the arbiter asked for;
//! - **actual**: where the gimbal truly is;
//! - **measured**: actual plus the encoder and IMU's declared noise.
//!
//! The LiDAR's rays are cast from the **true** pose: actual attitude, true
//! mount, and the calibration's boresight bias. Its returns are then
//! georeferenced with the **estimated** pose: measured attitude, nominal
//! mount, no bias. So the boresight bias and the attitude noise show in the
//! map as they would in flight, instead of cancelling out.
//!
//! The platform's position and heading are still the truth's: there is no
//! position estimator in the loop yet, so localization error is not in this
//! chain.
//!
//! # Pointing
//!
//! The planner proposes a [`PointingPolicy`] and the arbiter commands the
//! pod. The forward navigation camera is the avoidance sensor
//! ([`crate::nav_camera`]), so no safety rule ever takes the pod. The arbiter
//! has two sources:
//!
//! 1. **stow** while the aircraft is below cruise (horizontal speed under
//!    0.5 m/s). Hover is for take-off and landing only, so a hovering
//!    aircraft is not mapping;
//! 2. **mission** otherwise: the policy's attitude.
//!
//! # Housing occlusion
//!
//! The pod hangs below the airframe. A ray that leaves at or above
//! [`GimbalEnvelope::airframe_mask_depression_deg`] below the body's horizon
//! meets the airframe and is counted as occluded, not as a miss.

use std::collections::BTreeMap;
use std::sync::Arc;

use argusnet_world::powerline::PowerLine;
use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::range_scan::{
    span_pieces, wire_detection_range_m, wires_seen, RangeReturn, RangeScan, ReturnClass,
    ScanSensor, SpanPieces,
};
use crate::rig::{GimbalEnvelope, LidarModel, RigProfile, SensorMount, SensorRole};
use crate::runtime_world::SharedVisibility;

/// A unit quaternion, `[w, x, y, z]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quat(pub [f64; 4]);

impl Quat {
    pub const IDENTITY: Quat = Quat([1.0, 0.0, 0.0, 0.0]);

    pub fn from_axis_angle(axis: [f64; 3], angle_rad: f64) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-15 {
            return Self::IDENTITY;
        }
        let (s, c) = (angle_rad / 2.0).sin_cos();
        Quat([
            c,
            axis[0] / norm * s,
            axis[1] / norm * s,
            axis[2] / norm * s,
        ])
    }

    /// Yaw about z, then pitch about the new y (positive nose down in a
    /// z-up frame), then roll about the new x.
    pub fn from_ypr(yaw_rad: f64, pitch_rad: f64, roll_rad: f64) -> Self {
        Self::from_axis_angle([0.0, 0.0, 1.0], yaw_rad)
            .then(Self::from_axis_angle([0.0, 1.0, 0.0], pitch_rad))
            .then(Self::from_axis_angle([1.0, 0.0, 0.0], roll_rad))
    }

    pub fn from_ypr_deg(ypr: [f64; 3]) -> Self {
        Self::from_ypr(
            ypr[0].to_radians(),
            ypr[1].to_radians(),
            ypr[2].to_radians(),
        )
    }

    /// `self` then `other`: the composed rotation.
    pub fn then(self, other: Quat) -> Quat {
        let [a1, b1, c1, d1] = self.0;
        let [a2, b2, c2, d2] = other.0;
        Quat([
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ])
        .normalized()
    }

    pub fn conj(self) -> Quat {
        let [w, x, y, z] = self.0;
        Quat([w, -x, -y, -z])
    }

    fn normalized(self) -> Quat {
        let n = self.0.iter().map(|v| v * v).sum::<f64>().sqrt();
        Quat(self.0.map(|v| v / n))
    }

    pub fn rotate(self, v: [f64; 3]) -> [f64; 3] {
        let p = Quat([0.0, v[0], v[1], v[2]]);
        // Not normalised: p is a vector, not a rotation.
        let [a1, b1, c1, d1] = self.0;
        let [a2, b2, c2, d2] = p.0;
        let q = [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ];
        let [a1, b1, c1, d1] = q;
        let [a2, b2, c2, d2] = self.conj().0;
        [
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ]
    }

    /// The rotation angle from `self` to `other`, radians in `[0, π]`.
    pub fn angle_to(self, other: Quat) -> f64 {
        let dot: f64 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        2.0 * dot.abs().min(1.0).acos()
    }

    /// Turn from `self` toward `target` by at most `max_rad`, along the
    /// shortest rotation.
    pub fn toward(self, target: Quat, max_rad: f64) -> Quat {
        let angle = self.angle_to(target);
        if angle <= max_rad || angle < 1e-12 {
            return target;
        }
        let mut dot: f64 = self.0.iter().zip(target.0.iter()).map(|(a, b)| a * b).sum();
        let mut target = target.0;
        if dot < 0.0 {
            target = target.map(|v| -v);
            dot = -dot;
        }
        // Slerp by the fraction of the angle allowed.
        let t = max_rad / angle;
        let theta = dot.min(1.0).acos();
        let s = theta.sin();
        let (wa, wb) = if s < 1e-12 {
            (1.0 - t, t)
        } else {
            (((1.0 - t) * theta).sin() / s, (t * theta).sin() / s)
        };
        Quat(std::array::from_fn(|k| wa * self.0[k] + wb * target[k])).normalized()
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// What the planner asks the pod to hold while mapping.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PointingPolicy {
    /// Straight down, image up along the track, stabilised in the world.
    #[default]
    Nadir,
    /// Nadir rolled side to side about the track: `amplitude_deg` each way,
    /// one full cycle every `period_s`. Widens the swath at the cost of
    /// steeper incidence at its edges.
    CrossTrackSweep { amplitude_deg: f64, period_s: f64 },
    /// A fixed attitude in the body frame (yaw, pitch, roll in degrees).
    BodyFixed { ypr_deg: [f64; 3] },
}

impl PointingPolicy {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Nadir => Ok(()),
            Self::CrossTrackSweep {
                amplitude_deg,
                period_s,
            } => {
                if !(amplitude_deg.is_finite() && (0.0..=90.0).contains(amplitude_deg)) {
                    return Err(format!(
                        "sweep amplitude must be within [0, 90] degrees, got {amplitude_deg}"
                    ));
                }
                if !(period_s.is_finite() && *period_s > 0.0) {
                    return Err(format!("sweep period must be positive, got {period_s}"));
                }
                Ok(())
            }
            Self::BodyFixed { ypr_deg } => {
                if ypr_deg.iter().all(|v| v.is_finite()) {
                    Ok(())
                } else {
                    Err("body-fixed pointing must be finite".into())
                }
            }
        }
    }

    /// The commanded attitude in the body frame at `t_s`, the aircraft
    /// heading `heading_rad`. World-stabilised modes come back to the body
    /// through the heading: the sim has no body roll or pitch.
    fn commanded_body(&self, heading_rad: f64, t_s: f64) -> Quat {
        let down = Quat::from_ypr(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        match self {
            Self::Nadir => down,
            Self::CrossTrackSweep {
                amplitude_deg,
                period_s,
            } => {
                let phase = std::f64::consts::TAU * t_s / period_s;
                let roll = amplitude_deg.to_radians() * phase.sin();
                // Roll about the track axis (body x), then point down.
                Quat::from_axis_angle([1.0, 0.0, 0.0], roll).then(down)
            }
            Self::BodyFixed { ypr_deg } => {
                let _ = heading_rad;
                Quat::from_ypr_deg(*ypr_deg)
            }
        }
    }
}

/// Who commanded the pod this capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointingSource {
    /// Below cruise: take-off, landing, parked.
    Stow,
    /// The planner's policy.
    Mission,
}

/// One platform's gimbal, as it carries between ticks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GimbalState {
    /// The pod's true attitude in the body frame.
    pub actual: Quat,
    /// When the pod came within tolerance of its command and has stayed
    /// there; `None` while it is off target.
    pub on_target_since_s: Option<f64>,
    pub last_t_s: Option<f64>,
    /// The body's heading, kept while the aircraft is too slow to have one.
    pub heading_rad: f64,
}

impl GimbalState {
    fn stowed(gimbal: &GimbalEnvelope) -> Self {
        Self {
            actual: Quat::from_ypr_deg(gimbal.stow_ypr_deg),
            on_target_since_s: None,
            last_t_s: None,
            heading_rad: 0.0,
        }
    }
}

/// What a capture records about the pod.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodCapture {
    pub source: PointingSource,
    /// Whether the pod had settled on its command: an unsettled capture is
    /// refused by the map.
    pub settled: bool,
    /// How far the actual attitude was from the commanded one, radians.
    pub command_error_rad: f64,
    /// The LiDAR's boresight in the world as the estimator has it (unit).
    pub boresight_world: [f64; 3],
    /// Rays the airframe blocked.
    pub occluded: u32,
}

/// The pod: its gimbals, its LiDAR, its pointing.
#[derive(Debug, Clone)]
pub struct PodSensing {
    rig: RigProfile,
    lidar_mount: SensorMount,
    lidar: LidarModel,
    policy: PointingPolicy,
    world: SharedVisibility,
    spans: Arc<Vec<SpanPieces>>,
    gimbals: BTreeMap<String, GimbalState>,
}

/// The pod's checkpointed state: every platform's gimbal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodsState {
    pub gimbals: BTreeMap<String, GimbalState>,
}

/// Below this horizontal speed the aircraft is taking off, landing or parked.
pub const CRUISE_THRESHOLD_MPS: f64 = 0.5;

impl PodSensing {
    pub fn new(
        rig: RigProfile,
        policy: PointingPolicy,
        world: SharedVisibility,
    ) -> Result<Self, String> {
        rig.validate()?;
        policy.validate()?;
        let lidar_mount = rig
            .mount(SensorRole::Lidar)
            .cloned()
            .ok_or("the rig carries no LiDAR")?;
        let lidar = lidar_mount
            .lidar
            .clone()
            .ok_or("the rig LiDAR has no model")?;
        Ok(Self {
            rig,
            lidar_mount,
            lidar,
            policy,
            world,
            spans: Arc::new(Vec::new()),
            gimbals: BTreeMap::new(),
        })
    }

    /// The pod, also putting points on `lines`' wires within the range its
    /// points are dense enough for them.
    pub fn with_power_lines(mut self, lines: &[PowerLine]) -> Self {
        let max_range_m = self.lidar.max_range_m;
        self.spans = Arc::new(span_pieces(lines, |line| {
            wire_detection_range_m(line.conductor_radius_m).min(max_range_m)
        }));
        self
    }

    pub fn rig(&self) -> &RigProfile {
        &self.rig
    }

    pub fn policy(&self) -> &PointingPolicy {
        &self.policy
    }

    pub fn state(&self) -> PodsState {
        PodsState {
            gimbals: self.gimbals.clone(),
        }
    }

    pub fn restore(&mut self, state: PodsState) {
        self.gimbals = state.gimbals;
    }

    pub fn reset(&mut self) {
        self.gimbals.clear();
    }

    /// The highest the aircraft may fly above the ground it believes and
    /// still put every ray of the pod's pointing on the ground within the
    /// LiDAR's range (§5: the furthest required ray, not the boresight):
    /// `max_range · cos(θ_max) · terrain_margin − unknown_margin_m`, where
    /// `θ_max` is the steepest ray from nadir over the policy's whole
    /// cycle. `None` when a ray of the policy points within 10° of the
    /// horizon, where no height reaches the ground everywhere.
    pub fn max_mapping_agl_m(&self, unknown_margin_m: f64) -> Option<f64> {
        /// Room for ground that rises within the swath.
        const TERRAIN_MARGIN: f64 = 0.9;
        let attitudes: Vec<Quat> = match &self.policy {
            PointingPolicy::Nadir => vec![self.policy.commanded_body(0.0, 0.0)],
            PointingPolicy::CrossTrackSweep { period_s, .. } => (0..64)
                .map(|k| self.policy.commanded_body(0.0, *period_s * k as f64 / 64.0))
                .collect(),
            PointingPolicy::BodyFixed { .. } => vec![self.policy.commanded_body(0.0, 0.0)],
        };
        let mount = Quat::from_ypr_deg(self.lidar_mount.rotation_ypr_deg);
        let rays = self.sensor_rays();
        let steepest = attitudes
            .iter()
            .flat_map(|attitude| {
                let sensor = attitude.then(mount);
                rays.iter().map(move |ray| {
                    let down = -sensor.rotate(*ray)[2];
                    down.clamp(-1.0, 1.0).acos()
                })
            })
            .fold(0.0, f64::max);
        if steepest >= 80f64.to_radians() {
            return None;
        }
        Some((self.lidar.max_range_m * steepest.cos() * TERRAIN_MARGIN - unknown_margin_m).max(0.0))
    }

    /// Unit rays across the LiDAR's rectangle in its own frame, row by row
    /// from the top, left to right.
    pub fn sensor_rays(&self) -> Vec<[f64; 3]> {
        let l = &self.lidar;
        let half_h = l.horizontal_fov_deg.to_radians() / 2.0;
        let half_v = l.vertical_fov_deg.to_radians() / 2.0;
        let mut rays = Vec::with_capacity(l.columns * l.rows);
        for r in 0..l.rows {
            let el = half_v - 2.0 * half_v * r as f64 / (l.rows - 1) as f64;
            for c in 0..l.columns {
                let az = half_h - 2.0 * half_h * c as f64 / (l.columns - 1) as f64;
                rays.push([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]);
            }
        }
        rays
    }

    /// Step `node_id`'s gimbal to `t_s` and say what holds it there: the
    /// arbiter's source, its command, and whether it has settled.
    fn step_gimbal(
        &mut self,
        node_id: &str,
        t_s: f64,
        velocity: [f64; 3],
    ) -> (PointingSource, Quat, GimbalState) {
        let gimbal = &self.rig.gimbal;
        let state = self
            .gimbals
            .entry(node_id.to_string())
            .or_insert_with(|| GimbalState::stowed(gimbal));
        let speed = velocity[0].hypot(velocity[1]);
        if speed > CRUISE_THRESHOLD_MPS {
            state.heading_rad = velocity[1].atan2(velocity[0]);
        }
        let (source, command) = if speed > CRUISE_THRESHOLD_MPS {
            (
                PointingSource::Mission,
                self.policy.commanded_body(state.heading_rad, t_s),
            )
        } else {
            (
                PointingSource::Stow,
                Quat::from_ypr_deg(gimbal.stow_ypr_deg),
            )
        };
        let dt = state.last_t_s.map_or(0.0, |last| (t_s - last).max(0.0));
        let start_error = state.actual.angle_to(command);
        let reach = gimbal.max_rate_rad_s * dt;
        state.actual = state.actual.toward(command, reach);
        let end_error = state.actual.angle_to(command);
        state.on_target_since_s = if end_error > gimbal.settle_tolerance_rad {
            None
        } else if start_error <= gimbal.settle_tolerance_rad {
            // Held on target through the step, unless never stamped.
            state.on_target_since_s.or(Some(t_s))
        } else {
            // Arrived within the step: when it came within tolerance.
            let travel = (start_error - gimbal.settle_tolerance_rad).max(0.0);
            Some(state.last_t_s.unwrap_or(t_s) + travel / gimbal.max_rate_rad_s)
        };
        state.last_t_s = Some(t_s);
        (source, command, state.clone())
    }

    /// Step every platform's gimbal to `t_s`, in platform order, and
    /// capture from each.
    pub fn capture_all(
        &mut self,
        tick: u64,
        t_s: f64,
        poses: &[(&str, [f64; 3], [f64; 3])],
        compute: &crate::compute::ComputePool,
    ) -> Vec<RangeScan> {
        let stepped: Vec<_> = poses
            .iter()
            .map(|(id, position, velocity)| {
                let (source, command, state) = self.step_gimbal(id, t_s, *velocity);
                (*id, *position, source, command, state)
            })
            .collect();
        let this = &*self;
        compute.map_ordered(&stepped, |(id, position, source, command, state)| {
            this.capture(id, tick, t_s, *position, *source, *command, state)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn capture(
        &self,
        node_id: &str,
        tick: u64,
        t_s: f64,
        position: [f64; 3],
        source: PointingSource,
        command: Quat,
        gimbal: &GimbalState,
    ) -> RangeScan {
        let envelope = &self.rig.gimbal;
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
        let mut attitude_rng = key("pod-attitude");
        let mut rng = key("pod-lidar");
        let mut canopy = key("pod-lidar-canopy");

        let settled = gimbal
            .on_target_since_s
            .is_some_and(|since| t_s - since >= envelope.settle_s - 1e-9);
        let body = Quat::from_axis_angle([0.0, 0.0, 1.0], gimbal.heading_rad);
        let base = add(position, body.rotate(envelope.base_lever_arm_m));
        let mount = Quat::from_ypr_deg(self.lidar_mount.rotation_ypr_deg);
        let bias = self.rig.calibration.lidar_boresight_bias_rad;
        let biased = mount.then(Quat::from_ypr(bias[0], bias[1], bias[2]));
        // Truth: actual attitude, true mount.
        let world_pod_true = body.then(gimbal.actual);
        let world_sensor_true = world_pod_true.then(biased);
        let origin_true = add(base, world_pod_true.rotate(self.lidar_mount.lever_arm_m));
        // Estimate: measured attitude, nominal mount.
        let sigma = envelope.attitude_noise_rad;
        let noise = [
            attitude_rng.normal(0.0, sigma),
            attitude_rng.normal(0.0, sigma),
            attitude_rng.normal(0.0, sigma),
        ];
        let measured = gimbal
            .actual
            .then(Quat::from_ypr(noise[0], noise[1], noise[2]));
        let world_pod_est = body.then(measured);
        let world_sensor_est = world_pod_est.then(mount);
        let origin_est = add(base, world_pod_est.rotate(self.lidar_mount.lever_arm_m));

        let mask = -envelope.airframe_mask_depression_deg.to_radians().sin();
        let max_range = self.lidar.max_range_m;
        let mut returns = Vec::new();
        let mut misses = 0u32;
        let mut occluded = 0u32;
        for ray in self.sensor_rays() {
            // Drawn for every ray, hit or not.
            let range_noise = rng.normal(0.0, self.lidar.range_noise_m);
            let direction = world_sensor_true.rotate(ray);
            // The body has no roll or pitch, so its horizon is the world's.
            if direction[2] > mask {
                occluded += 1;
                continue;
            }
            let end = add(origin_true, scale(direction, max_range));
            let Some((kind, point)) = self
                .world
                .first_return(origin_true, end, &mut || canopy.uniform())
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
            let range = (0..3)
                .map(|k| (point[k] - origin_true[k]).powi(2))
                .sum::<f64>()
                .sqrt()
                + range_noise;
            // Georeferenced by the estimate: the range along the ray as the
            // estimator believes it points.
            let believed = world_sensor_est.rotate(ray);
            returns.push(RangeReturn {
                point_m: add(origin_est, scale(believed, range)),
                class,
            });
        }
        let half_h = self.lidar.horizontal_fov_deg.to_radians() / 2.0;
        let half_v = self.lidar.vertical_fov_deg.to_radians() / 2.0;
        let inverse = world_sensor_true.conj();
        let wires = wires_seen(&self.spans, origin_true, |point| {
            let local = inverse.rotate([
                point[0] - origin_true[0],
                point[1] - origin_true[1],
                point[2] - origin_true[2],
            ]);
            local[0] > 0.0
                && local[1].atan2(local[0]).abs() <= half_h
                && local[2].atan2(local[0].hypot(local[1])).abs() <= half_v
        });
        RangeScan {
            node_id: node_id.to_string(),
            timestamp_s: t_s,
            origin_m: origin_est,
            returns,
            misses,
            wires,
            sensor: ScanSensor::RangeScanner,
            pod: Some(PodCapture {
                source,
                settled,
                command_error_rad: gimbal.actual.angle_to(command),
                boresight_world: world_sensor_est.rotate([1.0, 0.0, 0.0]),
                occluded,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_world::index::ObstacleLayer;
    use argusnet_world::obstacles::Bounds2d;
    use argusnet_world::tiled::TiledTerrain;
    use argusnet_world::visibility::VisibilityProfile;

    /// Flat ground at 0 on a 1 m grid: the world's terrain test samples
    /// along a ray, so a fine grid keeps its hits close to the surface.
    fn flat_world() -> SharedVisibility {
        let bounds = Bounds2d::new(-150.0, 150.0, -150.0, 150.0);
        let terrain = Arc::new(
            TiledTerrain::from_grid(bounds, &vec![vec![0.0; 301]; 301], 1.0, 4, 0.0, &[]).unwrap(),
        );
        SharedVisibility::new(
            terrain,
            Arc::new(ObstacleLayer::new(bounds, 100.0, vec![]).unwrap()),
            None,
            Arc::new(VisibilityProfile::default()),
            0.0,
        )
    }

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < tol)
    }

    #[test]
    fn quaternions_compose_and_point_nadir_without_a_singularity() {
        let down = Quat::from_ypr(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        assert!(close(down.rotate([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], 1e-12));
        let east_to_north = Quat::from_ypr(std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        assert!(close(
            east_to_north.rotate([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            1e-12
        ));
        assert!((down.angle_to(Quat::IDENTITY) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // Halfway along the shortest rotation.
        let half = Quat::IDENTITY.toward(down, std::f64::consts::FRAC_PI_4);
        assert!((half.angle_to(down) - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
        assert_eq!(Quat::IDENTITY.toward(down, 10.0), down);
        // Through nadir in any heading: the boresight stays straight down.
        for heading in [0.0, 1.0, 2.5, -3.0] {
            let body = Quat::from_axis_angle([0.0, 0.0, 1.0], heading);
            assert!(close(
                body.then(down).rotate([1.0, 0.0, 0.0]),
                [0.0, 0.0, -1.0],
                1e-12
            ));
        }
    }

    fn rig_without_noise() -> RigProfile {
        let mut rig = RigProfile::provisional();
        rig.calibration.lidar_boresight_bias_rad = [0.0; 3];
        rig.gimbal.attitude_noise_rad = 0.0;
        for mount in &mut rig.sensors {
            if let Some(lidar) = mount.lidar.as_mut() {
                lidar.range_noise_m = 0.0;
            }
        }
        rig
    }

    #[test]
    fn the_gimbal_slews_at_its_rate_and_settles_before_it_is_trusted() {
        let mut pod =
            PodSensing::new(rig_without_noise(), PointingPolicy::Nadir, flat_world()).unwrap();
        let pool = crate::compute::ComputePool::serial();
        let fly = [10.0, 0.0, 0.0];
        let at = |pod: &mut PodSensing, tick: u64, t: f64| {
            pod.capture_all(tick, t, &[("d", [0.0, 0.0, 100.0], fly)], &pool)
                .remove(0)
        };
        // From stow (forward) to nadir is 90 degrees: 0.785 s at 2 rad/s.
        let first = at(&mut pod, 0, 0.0);
        assert!(!first.pod.as_ref().unwrap().settled);
        let half = at(&mut pod, 1, 0.5);
        let capture = half.pod.as_ref().unwrap();
        assert!(!capture.settled, "still slewing at 0.5 s");
        assert!((capture.command_error_rad - (std::f64::consts::FRAC_PI_2 - 1.0)).abs() < 1e-9);
        let near = at(&mut pod, 2, 0.9);
        assert!(
            !near.pod.as_ref().unwrap().settled,
            "on target at ~0.78 s, but not held for 0.2 s yet"
        );
        let settled = at(&mut pod, 3, 1.1);
        let capture = settled.pod.as_ref().unwrap();
        assert!(capture.settled);
        assert_eq!(capture.source, PointingSource::Mission);
        assert!(close(capture.boresight_world, [0.0, 0.0, -1.0], 1e-9));
    }

    #[test]
    fn nadir_returns_land_under_the_aircraft_and_the_bias_shows_in_the_map() {
        let pool = crate::compute::ComputePool::serial();
        let fly = [10.0, 0.0, 0.0];
        let settle = |pod: &mut PodSensing| {
            for (k, t) in [0.0, 1.0, 2.0].into_iter().enumerate() {
                pod.capture_all(k as u64, t, &[("d", [0.0, 0.0, 100.0], fly)], &pool);
            }
            pod.capture_all(3, 3.0, &[("d", [0.0, 0.0, 100.0], fly)], &pool)
                .remove(0)
        };
        let mut clean =
            PodSensing::new(rig_without_noise(), PointingPolicy::Nadir, flat_world()).unwrap();
        let scan = settle(&mut clean);
        assert_eq!(scan.returns.len(), 24 * 10, "every ray meets flat ground");
        let spread_x = scan
            .returns
            .iter()
            .map(|r| r.point_m[0].abs())
            .fold(0.0, f64::max);
        // 30 degrees along the track and 70 across, from ~99.75 m: the
        // rectangle's corner rays reach furthest along it,
        // h·sin15° / (cos15°·cos35°), about 32.7 m, plus the lever arms.
        // 30 degrees along the track and 70 across, from 99.74 m: the
        // rectangle's corner rays reach furthest along it,
        // h·tan15° / cos35°, about 32.6 m.
        let corner = 99.74 * 15f64.to_radians().tan() / 35f64.to_radians().cos();
        assert!(
            (spread_x - corner).abs() < 1.0,
            "{spread_x} against {corner}"
        );
        for hit in &scan.returns {
            assert!(
                hit.point_m[2].abs() < 0.5,
                "georeferenced on the ground: {hit:?}"
            );
        }
        // With a 2 mrad pitch bias the estimator does not know of, the
        // ground maps tilted: 2 mrad times ~100 m, about 0.2 m out.
        let mut biased_rig = rig_without_noise();
        biased_rig.calibration.lidar_boresight_bias_rad = [0.0, 2.0e-3, 0.0];
        let mut biased = PodSensing::new(biased_rig, PointingPolicy::Nadir, flat_world()).unwrap();
        let tilted = settle(&mut biased);
        // The centre ray, straight down: a pitch bias moves its true
        // footprint ~0.2 m along the track, but the estimator puts the
        // return straight below, so the map is out by the bias times the
        // range.
        let centre = |scan: &RangeScan| {
            scan.returns
                .iter()
                .min_by(|a, b| {
                    (a.point_m[0].hypot(a.point_m[1])).total_cmp(&b.point_m[0].hypot(b.point_m[1]))
                })
                .unwrap()
                .point_m
        };
        let (clean_centre, tilted_centre) = (centre(&scan), centre(&tilted));
        assert!(
            (clean_centre[0] - tilted_centre[0]).abs() < 0.05,
            "georeferenced along the nominal ray either way"
        );
        // Off nadir the bias changes the true range, which the estimator
        // places along the nominal ray: the far edge maps out of the plane.
        let worst = tilted
            .returns
            .iter()
            .zip(&scan.returns)
            .map(|(t, c)| (t.point_m[2] - c.point_m[2]).abs())
            .fold(0.0, f64::max);
        assert!(worst > 0.05 && worst < 2.0, "{worst}");
    }

    #[test]
    fn a_hovering_aircraft_stows_and_the_airframe_blocks_level_rays() {
        let pool = crate::compute::ComputePool::serial();
        let mut pod =
            PodSensing::new(rig_without_noise(), PointingPolicy::Nadir, flat_world()).unwrap();
        let mut last = None;
        for (k, t) in [0.0, 1.0, 2.0].into_iter().enumerate() {
            last = Some(
                pod.capture_all(k as u64, t, &[("d", [0.0, 0.0, 100.0], [0.0; 3])], &pool)
                    .remove(0),
            );
        }
        let scan = last.unwrap();
        let capture = scan.pod.unwrap();
        assert_eq!(capture.source, PointingSource::Stow);
        // Stowed forward and level: the upper half of the rectangle is in
        // the airframe's mask.
        assert!(capture.occluded > 0);
        assert!(capture.occluded as usize >= 24 * 10 / 2);
    }

    #[test]
    fn a_sweep_widens_the_swath() {
        let pool = crate::compute::ComputePool::serial();
        let cross_extent = |policy: PointingPolicy| {
            let mut pod = PodSensing::new(rig_without_noise(), policy, flat_world()).unwrap();
            let mut widest: f64 = 0.0;
            for k in 0..40u64 {
                let t = k as f64 * 0.5;
                let scan = pod
                    .capture_all(k, t, &[("d", [0.0, 0.0, 100.0], [10.0, 0.0, 0.0])], &pool)
                    .remove(0);
                if scan.pod.as_ref().unwrap().settled {
                    for hit in &scan.returns {
                        widest = widest.max(hit.point_m[1].abs());
                    }
                }
            }
            widest
        };
        let nadir = cross_extent(PointingPolicy::Nadir);
        let sweep = cross_extent(PointingPolicy::CrossTrackSweep {
            amplitude_deg: 20.0,
            period_s: 8.0,
        });
        assert!(sweep > nadir * 1.3, "sweep {sweep} against nadir {nadir}");
    }
}
