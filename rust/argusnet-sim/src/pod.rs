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
    /// View planning (SR5d): nadir rolled across the track toward the
    /// deficits the footprint can fix, chosen each capture from the
    /// published quality map ([`argusnet_planning::views::choose_roll`]).
    Planned,
}

impl PointingPolicy {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Nadir | Self::Planned => Ok(()),
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

    /// How fast the commanded attitude turns at `t_s`, rad/s.
    fn command_rate_rad_s(&self, t_s: f64) -> f64 {
        match self {
            Self::CrossTrackSweep {
                amplitude_deg,
                period_s,
            } => {
                let omega = std::f64::consts::TAU / period_s;
                (amplitude_deg.to_radians() * omega * (omega * t_s).cos()).abs()
            }
            Self::Nadir | Self::Planned | Self::BodyFixed { .. } => 0.0,
        }
    }

    /// The commanded attitude in the body frame at `t_s`, the aircraft
    /// heading `heading_rad`. World-stabilised modes come back to the body
    /// through the heading: the sim has no body roll or pitch.
    fn commanded_body(&self, heading_rad: f64, t_s: f64) -> Quat {
        let down = Quat::from_ypr(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        match self {
            // The planned roll is applied by the pod, which holds it.
            Self::Nadir | Self::Planned => down,
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
    /// None: the gimbal is jammed (F2's `gimbal_jam`) and holds where it is,
    /// whatever is commanded. Its encoders still measure where that is.
    Jammed,
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
    /// The cross-track roll view planning holds (SR5d), radians.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub planned_roll_rad: f64,
    /// When the roll was last planned, and on what heading: it is planned
    /// again after [`argusnet_planning::views::ViewPlanningConfig::roll_replan_s`]
    /// or on a new leg.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roll_planned_at: Option<(f64, f64)>,
}

fn is_zero_f64(value: &f64) -> bool {
    *value == 0.0
}

/// What view planning needs to point the pod (SR5d): the published map, the
/// planner's parameters, and the steepest ray the requirement accepts.
#[derive(Debug, Clone)]
pub struct PodPlanning {
    pub belief: crate::belief_world::BeliefHandle,
    pub config: argusnet_planning::views::ViewPlanningConfig,
    pub max_incidence_rad: f64,
}

impl GimbalState {
    fn stowed(gimbal: &GimbalEnvelope) -> Self {
        Self {
            actual: Quat::from_ypr_deg(gimbal.stow_ypr_deg),
            on_target_since_s: None,
            last_t_s: None,
            heading_rad: 0.0,
            planned_roll_rad: 0.0,
            roll_planned_at: None,
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
    /// What the pod's cameras got (SR4). Absent without the imaging model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<crate::imaging::PodImages>,
}

/// The pod's cameras, as the imaging model needs them (SR4).
#[derive(Debug, Clone)]
struct PodCameras {
    config: crate::imaging::ImagingConfig,
    visible: SensorMount,
    visible_optics: crate::rig::CameraOptics,
    /// The visible pair's baseline, metres; `None` with one camera.
    baseline_m: Option<f64>,
    thermal: Option<(SensorMount, crate::rig::CameraOptics)>,
}

/// How far a camera's boresight is followed, metres.
const IMAGE_RANGE_M: f64 = 2000.0;

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
    /// Report wires as measured points instead of whole spans (W1).
    measured: Option<crate::wire_sensing::MeasuredWires>,
    /// The world's poles, measured as returns (W3).
    poles: Arc<Vec<crate::range_scan::SpanPieces>>,
    /// The time between captures, seconds (SR3).
    scan_interval_s: f64,
    /// Platforms whose gimbal is jammed this tick (F2). Set before every
    /// tick from the fault schedule; not checkpointed, since the attitude
    /// it holds is the gimbal's own, which is.
    jammed: std::collections::BTreeSet<String>,
    /// Set for [`PointingPolicy::Planned`].
    planning: Option<PodPlanning>,
    /// The steepest ray from nadir a quality requirement accepts, radians:
    /// no ray past it is required, so the height cap need not reach with it.
    required_incidence_rad: Option<f64>,
    /// The cameras' imaging model (SR4).
    cameras: Option<PodCameras>,
}

/// The furthest from nadir a jammed pod's boresight may point and its
/// captures still count as fixed-pointing survey evidence, degrees. Past it
/// the rays graze the ground far off, as a stowed pod's do, and the map
/// refuses them.
pub const JAMMED_MAX_OFF_NADIR_DEG: f64 = 45.0;

/// The pod's checkpointed state: every platform's gimbal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodsState {
    pub gimbals: BTreeMap<String, GimbalState>,
}

/// The fastest an aircraft moves across one scan, m/s (SR3): a faster
/// velocity is a sampled jump, not flight.
pub const MAX_DESKEW_SPEED_MPS: f64 = 60.0;

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
            jammed: std::collections::BTreeSet::new(),
            measured: None,
            poles: Arc::new(Vec::new()),
            scan_interval_s: 1.0,
            planning: None,
            required_incidence_rad: None,
            cameras: None,
        })
    }

    /// The time between captures, seconds: a modelled LiDAR's rays fire
    /// across it unless its model declares a shorter scan (SR3).
    pub fn with_scan_interval(mut self, interval_s: f64) -> Self {
        self.scan_interval_s = interval_s;
        self
    }

    /// Get returns from `lines`' poles too (W3), among the wire points.
    pub fn with_measured_poles(mut self, lines: &[PowerLine]) -> Self {
        self.poles = Arc::new(crate::range_scan::pole_pieces(
            lines,
            crate::wire_sensing::POLE_RADIUS_M,
        ));
        self
    }

    /// Report wires as measured points, never as whole spans (W1).
    pub fn with_measured_wires(mut self, measured: crate::wire_sensing::MeasuredWires) -> Self {
        self.measured = Some(measured);
        self
    }

    /// Model what the pod's cameras get (SR4). Needs a visible camera with
    /// declared optics.
    pub fn with_imaging(mut self, config: crate::imaging::ImagingConfig) -> Result<Self, String> {
        config.validate()?;
        let of = |role| {
            self.rig
                .sensors
                .iter()
                .filter(move |mount| mount.role == role)
                .filter_map(|mount| mount.camera.clone().map(|optics| (mount.clone(), optics)))
        };
        let mut pair = of(SensorRole::VisibleCamera);
        let (visible, visible_optics) = pair
            .next()
            .ok_or("imaging needs a visible camera with declared optics on the rig")?;
        let baseline_m = pair.next().map(|(second, _)| {
            (0..3)
                .map(|k| (second.lever_arm_m[k] - visible.lever_arm_m[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        });
        drop(pair);
        let thermal = of(SensorRole::ThermalCamera).next();
        self.cameras = Some(PodCameras {
            config,
            visible,
            visible_optics,
            baseline_m,
            thermal,
        });
        Ok(self)
    }

    /// The optics of the camera image requirements are met with, when the
    /// imaging model is on.
    pub fn visible_optics(&self) -> Option<&crate::rig::CameraOptics> {
        self.cameras.as_ref().map(|cameras| &cameras.visible_optics)
    }

    /// What the cameras get: from the pod's true attitude, with the pose the
    /// estimator would place the visible image by.
    #[allow(clippy::too_many_arguments)]
    fn image(
        &self,
        cameras: &PodCameras,
        source: PointingSource,
        settled: bool,
        velocity: [f64; 3],
        t_s: f64,
        base: [f64; 3],
        world_pod_true: Quat,
        world_pod_est: Quat,
        draw: &mut dyn FnMut() -> f64,
    ) -> crate::imaging::PodImages {
        use crate::imaging::{
            assess, stereo_depth_sigma_m, thermal_min_contrast_k, ImageCapture, ImagePose,
            ImageRefusal, PodImages, View,
        };
        let config = &cameras.config;
        let refusal = if source == PointingSource::Stow {
            Some(ImageRefusal::Stowed)
        } else if !settled {
            Some(ImageRefusal::Unsettled)
        } else {
            None
        };
        if let Some(refusal) = refusal {
            return PodImages {
                visible: ImageCapture::refused(refusal),
                stereo_depth_sigma_m: None,
                thermal: cameras
                    .thermal
                    .as_ref()
                    .map(|_| ImageCapture::refused(refusal)),
                thermal_min_contrast_k: None,
                visible_pose: None,
            };
        }
        // A settled pod still turns as fast as its command does: judging it
        // at its residual jitter passed every image of a sweep.
        let turn_rate_rad_s = if source == PointingSource::Jammed {
            0.0
        } else {
            self.policy
                .command_rate_rad_s(t_s)
                .min(self.rig.gimbal.max_rate_rad_s)
        };
        let mut view_of = |mount: &SensorMount| -> Option<View> {
            let world_camera = world_pod_true.then(Quat::from_ypr_deg(mount.rotation_ypr_deg));
            let origin = add(base, world_pod_true.rotate(mount.lever_arm_m));
            let boresight = world_camera.rotate([1.0, 0.0, 0.0]);
            let end = add(origin, scale(boresight, IMAGE_RANGE_M));
            let (_, point) = self.world.first_return(origin, end, draw)?;
            let range_m = (0..3)
                .map(|k| (point[k] - origin[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let along: f64 = (0..3).map(|k| velocity[k] * boresight[k]).sum();
            let speed_sq: f64 = velocity.iter().map(|v| v * v).sum();
            Some(View {
                range_m,
                incidence_rad: (-boresight[2]).clamp(-1.0, 1.0).acos(),
                cross_speed_mps: (speed_sq - along * along).max(0.0).sqrt(),
                turn_rate_rad_s,
            })
        };
        let visible_view = view_of(&cameras.visible);
        let visible = visible_view.map_or(ImageCapture::refused(ImageRefusal::NoGround), |view| {
            assess(
                &cameras.visible_optics,
                view,
                config.visible_exposure_s,
                config.visible_max_blur_px,
                config.max_incidence_deg,
            )
        });
        let thermal = cameras.thermal.as_ref().map(|(mount, optics)| {
            view_of(mount).map_or(ImageCapture::refused(ImageRefusal::NoGround), |view| {
                assess(
                    optics,
                    view,
                    config.thermal_time_constant_s,
                    config.thermal_max_blur_px,
                    config.max_incidence_deg,
                )
            })
        });
        PodImages {
            stereo_depth_sigma_m: cameras
                .baseline_m
                .zip(visible_view)
                .map(|(baseline, view)| {
                    stereo_depth_sigma_m(
                        &cameras.visible_optics,
                        baseline,
                        view.range_m,
                        config.stereo_disparity_sigma_px,
                    )
                }),
            thermal_min_contrast_k: thermal
                .as_ref()
                .and_then(|image| image.blur_px)
                .map(|blur| thermal_min_contrast_k(config.thermal_netd_k, blur)),
            visible_pose: visible.accepted().then(|| ImagePose {
                origin_m: add(base, world_pod_est.rotate(cameras.visible.lever_arm_m)),
                world_camera: world_pod_est
                    .then(Quat::from_ypr_deg(cameras.visible.rotation_ypr_deg)),
            }),
            visible,
            thermal,
        }
    }

    /// Only rays within `max_incidence_rad` of nadir are required (a quality
    /// requirement's limit): the height cap reaches with those. Before the
    /// first tick.
    pub fn set_required_incidence(&mut self, max_incidence_rad: f64) {
        self.required_incidence_rad = Some(max_incidence_rad);
    }

    /// Point by the published map (SR5d). Needed by
    /// [`PointingPolicy::Planned`]; before the first tick.
    pub fn set_planning(&mut self, planning: PodPlanning) {
        self.planning = Some(planning);
    }

    /// What the footprint at cross-track roll `roll_rad` would fix, from
    /// `position_m` heading `heading_rad`: each ray cast onto the believed
    /// ground below the aircraft (a plane at its height there), counted at
    /// its cell's deficit weight when the requirement would accept it and
    /// the LiDAR reaches it.
    fn footprint_gain(
        &self,
        planning: &PodPlanning,
        snapshot: &crate::belief_world::BeliefSnapshot,
        position_m: [f64; 3],
        heading_rad: f64,
        roll_rad: f64,
    ) -> f64 {
        let body = Quat::from_axis_angle([0.0, 0.0, 1.0], heading_rad);
        let down = Quat::from_ypr(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        let sensor = body
            .then(Quat::from_axis_angle([1.0, 0.0, 0.0], roll_rad))
            .then(down)
            .then(Quat::from_ypr_deg(self.lidar_mount.rotation_ypr_deg));
        let ground = snapshot.ground_at(position_m[0], position_m[1]);
        let height = position_m[2] - ground;
        if height <= 0.0 {
            return 0.0;
        }
        let min_down = planning.max_incidence_rad.cos();
        let mut gain = 0.0;
        for ray in self.sensor_rays() {
            let d = sensor.rotate(ray);
            let down_component = -d[2];
            if down_component < min_down.max(1e-6) {
                continue;
            }
            let range = height / down_component;
            if range > self.lidar.max_range_m {
                continue;
            }
            let x = position_m[0] + d[0] * range;
            let y = position_m[1] + d[1] * range;
            gain += snapshot.deficit_weight_at(x, y, &planning.config.weights);
        }
        gain
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

    /// The platforms whose gimbal is jammed from now on (F2), until set
    /// again.
    pub fn set_jammed(&mut self, nodes: std::collections::BTreeSet<String>) {
        self.jammed = nodes;
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
            // Nadir's: the planner rolls only where rolled rays still reach
            // the ground from the height nadir allows ([`Self::footprint_gain`]
            // scores a ray past the LiDAR's range as nothing), rather than
            // flying every leg lower in case it rolls.
            PointingPolicy::Planned => vec![self.policy.commanded_body(0.0, 0.0)],
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
        // The furthest *required* ray (§5): a ray steeper than a quality
        // requirement accepts maps nothing it counts.
        let steepest = self
            .required_incidence_rad
            .map_or(steepest, |limit| steepest.min(limit));
        if steepest >= 80f64.to_radians() {
            return None;
        }
        Some((self.lidar.max_range_m * steepest.cos() * TERRAIN_MARGIN - unknown_margin_m).max(0.0))
    }

    /// The width of ground the pod maps across the track at nadir from
    /// `height_m` above it, metres.
    pub fn nadir_swath_m(&self, height_m: f64) -> f64 {
        2.0 * height_m.max(0.0) * (self.lidar.horizontal_fov_deg.to_radians() / 2.0).tan()
    }

    /// What one nadir pass at `agl_m` and `speed_mps`, a capture every
    /// `interval_s`, should leave on flat ground: rays the requirement
    /// accepts and the LiDAR reaches, each counted by its chance of
    /// returning.
    pub fn predicted_pass(
        &self,
        agl_m: f64,
        speed_mps: f64,
        interval_s: f64,
        requirement: &argusnet_mapping::quality::QualityRequirement,
        per_return_sigma_m: f64,
        cell_area_m2: f64,
    ) -> Option<crate::map_report::PassPrediction> {
        if !(agl_m > 0.0 && speed_mps > 0.0 && interval_s > 0.0) {
            return None;
        }
        let limit = requirement.max_incidence_deg.to_radians();
        let sensor = PointingPolicy::Nadir
            .commanded_body(0.0, 0.0)
            .then(Quat::from_ypr_deg(self.lidar_mount.rotation_ypr_deg));
        // As the LiDAR is mounted: the swath is as wide as the rays that
        // count reach across the track (body y).
        let (mut left, mut right) = (f64::NEG_INFINITY, f64::INFINITY);
        let returns: f64 = self
            .sensor_rays()
            .into_iter()
            .filter_map(|ray| {
                let direction = sensor.rotate(ray);
                let down = -direction[2];
                if down < limit.cos().max(1e-6) {
                    return None;
                }
                let range = agl_m / down;
                if range > self.lidar.max_range_m {
                    return None;
                }
                let across = direction[1] * range;
                left = left.max(across);
                right = right.min(across);
                Some(self.lidar.returns.as_ref().map_or(1.0, |model| {
                    let reflectivity = model.reflectivity_of("terrain").unwrap_or(0.1);
                    model.detection_probability(reflectivity, down, range)
                        * (1.0 - model.dropout_probability)
                }))
            })
            .sum();
        let swath_m = left - right;
        if !(swath_m.is_finite() && swath_m > 0.0) {
            return None;
        }
        let density_per_m2 = returns / (swath_m * speed_mps * interval_s);
        let count = (density_per_m2 * cell_area_m2).max(1.0);
        let bias = requirement.bias_floor_m;
        Some(crate::map_report::PassPrediction {
            agl_m,
            speed_mps,
            swath_m,
            density_per_m2,
            vertical_sigma_m: (per_return_sigma_m * per_return_sigma_m / count + bias * bias)
                .sqrt(),
            image_gsd_m: self
                .visible_optics()
                .map(|optics| agl_m / crate::imaging::focal_px(optics)),
        })
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
        position_m: [f64; 3],
        velocity: [f64; 3],
    ) -> (PointingSource, Quat, GimbalState) {
        let speed = velocity[0].hypot(velocity[1]);
        let mut held = self
            .gimbals
            .get(node_id)
            .cloned()
            .unwrap_or_else(|| GimbalState::stowed(&self.rig.gimbal));
        if speed > CRUISE_THRESHOLD_MPS {
            held.heading_rad = velocity[1].atan2(velocity[0]);
        }
        // View planning chooses the roll before the gimbal steps toward it.
        if let (PointingPolicy::Planned, Some(planning), true) = (
            &self.policy,
            self.planning.as_ref(),
            speed > CRUISE_THRESHOLD_MPS,
        ) {
            let config = &planning.config;
            let new_leg = held.roll_planned_at.is_none_or(|(at_s, heading)| {
                let turn = (held.heading_rad - heading).rem_euclid(std::f64::consts::TAU);
                let turn = turn.min(std::f64::consts::TAU - turn);
                t_s - at_s >= config.roll_replan_s - 1e-9 || turn > config.leg_turn_deg.to_radians()
            });
            let snapshot = planning.belief.current();
            if new_leg && snapshot.quality().is_some() && config.plan_pointing {
                // The leg's ground: footprints at four points along the
                // track over the hold, at this speed.
                let reach_m = speed * config.roll_replan_s;
                let (ux, uy) = (held.heading_rad.cos(), held.heading_rad.sin());
                let ahead: Vec<[f64; 3]> = (0..4)
                    .map(|k| {
                        let d = reach_m * k as f64 / 3.0;
                        [
                            position_m[0] + ux * d,
                            position_m[1] + uy * d,
                            position_m[2],
                        ]
                    })
                    .collect();
                let gimbal = &self.rig.gimbal;
                let interval = held.last_t_s.map_or(0.0, |last| (t_s - last).max(0.0));
                let current = held.planned_roll_rad;
                // Held for the leg: as many captures as the hold spans.
                let leg = argusnet_planning::views::ViewPlanningConfig {
                    roll_horizon_captures: if interval > 0.0 {
                        (config.roll_replan_s / interval).max(1.0)
                    } else {
                        config.roll_horizon_captures
                    },
                    ..config.clone()
                };
                held.planned_roll_rad = argusnet_planning::views::choose_roll(
                    current,
                    &leg,
                    |roll| {
                        ahead
                            .iter()
                            .map(|at| {
                                self.footprint_gain(
                                    planning,
                                    &snapshot,
                                    *at,
                                    held.heading_rad,
                                    roll,
                                )
                            })
                            .sum()
                    },
                    |roll| {
                        if interval <= 0.0 {
                            return 0.0;
                        }
                        let busy_s =
                            (roll - current).abs() / gimbal.max_rate_rad_s + gimbal.settle_s;
                        (busy_s / interval).ceil()
                    },
                );
                held.roll_planned_at = Some((t_s, held.heading_rad));
            } else if snapshot.quality().is_none() || !config.plan_pointing {
                held.planned_roll_rad = 0.0;
            }
        }
        let gimbal = &self.rig.gimbal;
        let state = self
            .gimbals
            .entry(node_id.to_string())
            .or_insert(held.clone());
        *state = held;
        let (source, command) = if speed > CRUISE_THRESHOLD_MPS {
            let base = self.policy.commanded_body(state.heading_rad, t_s);
            let command = match self.policy {
                PointingPolicy::Planned => {
                    Quat::from_axis_angle([1.0, 0.0, 0.0], state.planned_roll_rad).then(base)
                }
                _ => base,
            };
            (PointingSource::Mission, command)
        } else {
            (
                PointingSource::Stow,
                Quat::from_ypr_deg(gimbal.stow_ypr_deg),
            )
        };
        if self.jammed.contains(node_id) {
            // Jammed: it stays where it is. On target or not is still the
            // attitude against the command, so a pod jammed where it was
            // already pointing does not have to settle again once freed.
            state.on_target_since_s =
                if state.actual.angle_to(command) <= gimbal.settle_tolerance_rad {
                    state.on_target_since_s.or(Some(t_s))
                } else {
                    None
                };
            state.last_t_s = Some(t_s);
            return (PointingSource::Jammed, command, state.clone());
        }
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
                let (source, command, state) = self.step_gimbal(id, t_s, *position, *velocity);
                (*id, *position, *velocity, source, command, state)
            })
            .collect();
        let this = &*self;
        compute.map_ordered(
            &stepped,
            |(id, position, velocity, source, command, state)| {
                this.capture(
                    id, tick, t_s, *position, *velocity, *source, *command, state,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn capture(
        &self,
        node_id: &str,
        tick: u64,
        t_s: f64,
        position: [f64; 3],
        velocity: [f64; 3],
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
        // A jammed pod is still and its attitude measured: its capture is
        // fixed-pointing evidence while it looks down steeply enough.
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
        let model = self.lidar.returns.as_ref();
        // SR3 draws from a stream of its own, so the legacy draws are
        // unchanged with it off.
        let mut admit = key("pod-lidar-returns");
        let rays = self.sensor_rays();
        let period = model.map_or(0.0, |m| m.scan_period_s.unwrap_or(self.scan_interval_s));
        // How the aircraft moves across the scan: along its horizontal
        // velocity, never faster than an aircraft flies. A climb or descent
        // carried on for a whole capture put origins underground at take-off
        // and landing, and a sampled jump is not flight.
        let deskew_velocity = {
            let horizontal = velocity[0].hypot(velocity[1]);
            let scale_down = if horizontal > MAX_DESKEW_SPEED_MPS {
                MAX_DESKEW_SPEED_MPS / horizontal
            } else {
                1.0
            };
            [velocity[0] * scale_down, velocity[1] * scale_down, 0.0]
        };
        let count = rays.len().max(1) as f64;
        for (k, ray) in rays.into_iter().enumerate() {
            // Drawn for every ray, hit or not.
            let range_noise = rng.normal(0.0, self.lidar.range_noise_m);
            // When this ray fires, and where both the aircraft and the
            // estimate of it have moved to by then.
            let offset_s = period * k as f64 / count;
            let moved = scale(deskew_velocity, offset_s);
            let origin_true = add(origin_true, moved);
            let origin_est = add(origin_est, moved);
            // A ray carried along the track into rising ground would fire
            // from inside it: not fired, as one the airframe blocks.
            if offset_s > 0.0
                && origin_true[2] < self.world.terrain_height(origin_true[0], origin_true[1]) + 0.5
            {
                occluded += 1;
                continue;
            }
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
            let true_range = (0..3)
                .map(|k| (point[k] - origin_true[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let mut range = true_range + range_noise;
            let mut believed_ray = ray;
            if let Some(model) = model {
                // Admitted by the power it carries back, then dropped out or
                // not, all drawn for every hit so one's outcome never shifts
                // another's.
                let reflectivity = model.reflectivity_of(kind.as_str()).unwrap_or(0.1);
                let cos_incidence = if kind == "terrain" {
                    let normal = self.world.terrain_normal(point[0], point[1]);
                    -(direction[0] * normal[0]
                        + direction[1] * normal[1]
                        + direction[2] * normal[2])
                } else {
                    1.0
                };
                let detected = admit.uniform()
                    < model.detection_probability(reflectivity, cos_incidence, true_range);
                let dropped = admit.uniform() < model.dropout_probability;
                let angles = [
                    admit.normal(0.0, model.angular_noise_rad),
                    admit.normal(0.0, model.angular_noise_rad),
                ];
                if !detected || dropped {
                    misses += 1;
                    continue;
                }
                range = model.reported_range_m(range);
                believed_ray = crate::lidar_returns::perturbed(ray, angles);
            }
            // Georeferenced by the estimate: the range along the ray as the
            // estimator believes it points.
            let believed = world_sensor_est.rotate(believed_ray);
            returns.push(RangeReturn {
                point_m: add(origin_est, scale(believed, range)),
                class,
                acquired_offset_s: model.map(|_| offset_s),
                acquired_from_m: model.map(|_| origin_est),
            });
        }
        let half_h = self.lidar.horizontal_fov_deg.to_radians() / 2.0;
        let half_v = self.lidar.vertical_fov_deg.to_radians() / 2.0;
        let inverse = world_sensor_true.conj();
        let in_rectangle = |d: [f64; 3]| {
            let local = inverse.rotate(d);
            local[0] > 0.0
                && local[1].atan2(local[0]).abs() <= half_h
                && local[2].atan2(local[0].hypot(local[1])).abs() <= half_v
        };
        let (wires, wire_points) = match &self.measured {
            None => (
                wires_seen(&self.spans, origin_true, |point| {
                    in_rectangle([
                        point[0] - origin_true[0],
                        point[1] - origin_true[1],
                        point[2] - origin_true[2],
                    ])
                }),
                Vec::new(),
            ),
            Some(measured) => {
                let mut wire_rng = key("pod-lidar-wire");
                // Crowns draw from a stream of their own, as the scan's do.
                let mut wire_canopy = key("pod-lidar-wire-canopy");
                let mut points = crate::wire_sensing::lidar_points(
                    &self.spans,
                    &measured.model,
                    origin_true,
                    measured.capture_s,
                    2.0 * half_h * 2.0 * half_v,
                    self.lidar.max_range_m,
                    in_rectangle,
                    |point| {
                        self.world
                            .first_return(origin_true, point, &mut || wire_canopy.uniform())
                            .is_none()
                    },
                    &mut wire_rng,
                );
                if !self.poles.is_empty() {
                    let mut pole_rng = key("pod-lidar-pole");
                    let mut pole_canopy = key("pod-lidar-pole-canopy");
                    points.extend(crate::wire_sensing::lidar_points(
                        &self.poles,
                        &measured.model,
                        origin_true,
                        measured.capture_s,
                        2.0 * half_h * 2.0 * half_v,
                        self.lidar.max_range_m,
                        in_rectangle,
                        |point| {
                            self.world
                                .first_return(origin_true, point, &mut || pole_canopy.uniform())
                                .is_none()
                        },
                        &mut pole_rng,
                    ));
                }
                // Georeferenced by the estimate, as the ground returns are.
                let believe = world_sensor_est.then(inverse);
                let points = points
                    .into_iter()
                    .map(|point| {
                        let d = [
                            point[0] - origin_true[0],
                            point[1] - origin_true[1],
                            point[2] - origin_true[2],
                        ];
                        add(origin_est, believe.rotate(d))
                    })
                    .collect();
                (Vec::new(), points)
            }
        };
        let settled = if source == PointingSource::Jammed {
            let down = -world_sensor_est.rotate([1.0, 0.0, 0.0])[2];
            down >= JAMMED_MAX_OFF_NADIR_DEG.to_radians().cos()
        } else {
            settled
        };
        let images = self.cameras.as_ref().map(|cameras| {
            // A stream of its own: the LiDAR's draws are the same with it on.
            let mut leaves = key("pod-image-canopy");
            self.image(
                cameras,
                source,
                settled,
                velocity,
                t_s,
                base,
                world_pod_true,
                world_pod_est,
                &mut || leaves.uniform(),
            )
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
                images,
            }),
            wire_points,
            wire_lines: Vec::new(),
            pose: None,
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

    /// The predicted swath is the mounted sensor's: rolled a quarter turn
    /// about its boresight, its short side lies across the track.
    #[test]
    fn a_predicted_pass_is_as_wide_as_the_mounted_sensor_sees() {
        let requirement = argusnet_mapping::quality::QualityRequirement {
            min_ground_density_per_m2: Some(0.03),
            max_vertical_sigma_m: None,
            max_age_s: None,
            max_image_gsd_m: None,
            max_horizontal_sigma_m: None,
            max_incidence_deg: 40.0,
            bias_floor_m: 0.0,
        };
        let predict = |roll_deg: f64| {
            let mut rig = RigProfile::provisional();
            for sensor in &mut rig.sensors {
                if sensor.role == SensorRole::Lidar {
                    sensor.rotation_ypr_deg = [0.0, 0.0, roll_deg];
                }
            }
            PodSensing::new(rig, PointingPolicy::Nadir, flat_world())
                .unwrap()
                .predicted_pass(100.0, 20.0, 0.25, &requirement, 0.05, 25.0)
                .unwrap()
        };
        let (as_built, rolled) = (predict(0.0), predict(90.0));
        let wide = 200.0 * 35f64.to_radians().tan();
        assert!((as_built.swath_m - wide).abs() < 1e-6, "{as_built:?}");
        assert!(rolled.swath_m < 0.6 * wide, "{rolled:?}");
        assert!(rolled.density_per_m2 > as_built.density_per_m2);
        assert!(as_built.density_per_m2 > 0.0);
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

    /// A climb is not carried across the scan, and a jump faster than an
    /// aircraft flies is cut to that speed: no ray is fired from underground.
    #[test]
    fn a_climb_or_a_jump_never_puts_a_ray_underground() {
        let rig = rig_without_noise().with_lidar_returns(exact_returns());
        let climbing = settled_captures(rig.clone(), 30.0, [10.0, 0.0, -40.0], 1).remove(0);
        for hit in &climbing.returns {
            assert!(hit.point_m[2].abs() < 0.5, "{hit:?}");
        }
        let jump = settled_captures(rig, 100.0, [500.0, 0.0, 0.0], 1).remove(0);
        let reach = jump
            .returns
            .iter()
            .map(|r| r.point_m[0] - jump.origin_m[0])
            .fold(f64::MIN, f64::max);
        assert!(reach < MAX_DESKEW_SPEED_MPS + 40.0, "{reach}");
    }

    /// Ground rising as `z = x` on a 1 m grid, x from -150 to 150.
    fn ramp_world() -> SharedVisibility {
        let bounds = Bounds2d::new(-150.0, 150.0, -150.0, 150.0);
        let row: Vec<f64> = (0..=300).map(|i| -150.0 + f64::from(i)).collect();
        let terrain = Arc::new(
            TiledTerrain::from_grid(bounds, &vec![row; 301], 1.0, 4, -150.0, &[]).unwrap(),
        );
        SharedVisibility::new(
            terrain,
            Arc::new(ObstacleLayer::new(bounds, 100.0, vec![]).unwrap()),
            None,
            Arc::new(VisibilityProfile::default()),
            0.0,
        )
    }

    /// Flying fast and low up a slope, a ray carried along the track would
    /// fire from inside the rising ground: it is not fired, and no return
    /// lands under the ground.
    #[test]
    fn a_ray_carried_into_rising_ground_is_not_fired() {
        let pool = crate::compute::ComputePool::serial();
        let rig = rig_without_noise().with_lidar_returns(exact_returns());
        let mut pod = PodSensing::new(rig, PointingPolicy::Nadir, ramp_world())
            .unwrap()
            .with_scan_interval(1.0);
        let fly = [30.0, 0.0, 0.0];
        let at = [-60.0, 0.0, -50.0];
        for k in 0..4u64 {
            pod.capture_all(k, k as f64, &[("d", at, fly)], &pool);
        }
        let scan = pod.capture_all(4, 4.0, &[("d", at, fly)], &pool).remove(0);
        assert!(!scan.returns.is_empty());
        assert!(scan.returns.len() < 240, "some rays were not fired");
        for hit in &scan.returns {
            let below = hit.point_m[0] - hit.point_m[2];
            assert!(below < 0.6, "{below} m under the slope: {hit:?}");
        }
    }

    /// Each timed return carries where its ray started, so incidence
    /// measured from there is the ray's own, as for the same ray fired with
    /// no motion.
    #[test]
    fn a_timed_return_carries_its_own_ray_origin() {
        let fly = [30.0, 0.0, 0.0];
        let timed = settled_captures(
            rig_without_noise().with_lidar_returns(exact_returns()),
            30.0,
            fly,
            1,
        )
        .remove(0);
        let still = settled_captures(
            rig_without_noise().with_lidar_returns(crate::lidar_returns::LidarReturnModel {
                scan_period_s: Some(1.0e-9),
                ..exact_returns()
            }),
            30.0,
            fly,
            1,
        )
        .remove(0);
        let incidence = |hit: &RangeReturn, origin: [f64; 3]| {
            let from = hit.ray_origin_m(origin);
            (hit.point_m[0] - from[0])
                .hypot(hit.point_m[1] - from[1])
                .atan2(from[2] - hit.point_m[2])
        };
        for (a, b) in timed.returns.iter().zip(&still.returns) {
            let (ia, ib) = (incidence(a, timed.origin_m), incidence(b, still.origin_m));
            assert!((ia - ib).abs() < 0.02, "{ia} against {ib}");
        }
    }

    /// SR3: a model with every error off and a reference range far past
    /// anything in view, so every ray returns exactly.
    fn exact_returns() -> crate::lidar_returns::LidarReturnModel {
        crate::lidar_returns::LidarReturnModel {
            reference_range_m: 1.0e6,
            angular_noise_rad: 0.0,
            range_bias_m: 0.0,
            range_quantization_m: 0.0,
            dropout_probability: 0.0,
            ..Default::default()
        }
    }

    /// Settled nadir captures at `height_m`, flying `fly`, ticks 4 onward.
    fn settled_captures(
        rig: RigProfile,
        height_m: f64,
        fly: [f64; 3],
        captures: usize,
    ) -> Vec<RangeScan> {
        let pool = crate::compute::ComputePool::serial();
        let mut pod = PodSensing::new(rig, PointingPolicy::Nadir, flat_world())
            .unwrap()
            .with_scan_interval(1.0);
        let at = [0.0, 0.0, height_m];
        for k in 0..4u64 {
            pod.capture_all(k, k as f64, &[("d", at, fly)], &pool);
        }
        (4..4 + captures as u64)
            .map(|k| {
                pod.capture_all(k, k as f64, &[("d", at, fly)], &pool)
                    .remove(0)
            })
            .collect()
    }

    /// The rays fire across the capture, each from where the aircraft has
    /// flown to, and the estimate moves with it: at 30 m/s every return
    /// still lands on the plane, every ray returns, and the offsets climb
    /// through the scan period.
    #[test]
    fn modelled_rays_fire_across_the_capture_and_still_land_on_the_plane() {
        let rig = rig_without_noise().with_lidar_returns(exact_returns());
        let scan = settled_captures(rig, 100.0, [30.0, 0.0, 0.0], 1).remove(0);
        assert_eq!(scan.returns.len(), 24 * 10);
        let offsets: Vec<f64> = scan
            .returns
            .iter()
            .map(|r| r.acquired_offset_s.expect("timed"))
            .collect();
        assert_eq!(offsets[0], 0.0);
        assert!(offsets.windows(2).all(|w| w[1] > w[0]));
        assert!(*offsets.last().unwrap() < 1.0);
        for hit in &scan.returns {
            assert!(hit.point_m[2].abs() < 0.5, "{hit:?}");
        }
        // Each ray lands where the same ray of an untimed capture does,
        // moved along the track by the speed times its offset.
        let untimed = settled_captures(
            rig_without_noise().with_lidar_returns(crate::lidar_returns::LidarReturnModel {
                scan_period_s: Some(1.0e-9),
                ..exact_returns()
            }),
            100.0,
            [30.0, 0.0, 0.0],
            1,
        )
        .remove(0);
        for (timed, still) in scan.returns.iter().zip(&untimed.returns) {
            let moved = timed.point_m[0] - still.point_m[0];
            let expected = 30.0 * timed.acquired_offset_s.unwrap();
            assert!((moved - expected).abs() < 0.6, "{moved} against {expected}");
        }
        // Without a model nothing is timed, as before.
        let legacy = settled_captures(rig_without_noise(), 100.0, [30.0, 0.0, 0.0], 1).remove(0);
        assert!(legacy.returns.iter().all(|r| r.acquired_offset_s.is_none()));
    }

    /// Over flat ground a ray's range and incidence follow from its angle
    /// alone, so the returns expected from a capture are the model's chance
    /// summed over the rays an exact model returns. The counts agree within
    /// four standard deviations, the curve bites (well under every ray),
    /// and a dark surface returns less than a bright one.
    #[test]
    fn return_fractions_follow_the_declared_curve() {
        let height = 140.0;
        let model = |terrain: f64| crate::lidar_returns::LidarReturnModel {
            reference_range_m: 110.0,
            dropout_probability: 0.0,
            reflectivity: crate::lidar_returns::SurfaceReflectivity {
                terrain,
                ..Default::default()
            },
            ..exact_returns()
        };
        let exact = settled_captures(
            rig_without_noise().with_lidar_returns(exact_returns()),
            height,
            [10.0, 0.0, 0.0],
            1,
        )
        .remove(0);
        let expected = |m: &crate::lidar_returns::LidarReturnModel| -> (f64, f64) {
            exact.returns.iter().fold((0.0, 0.0), |(sum, var), hit| {
                let d: [f64; 3] = std::array::from_fn(|k| hit.point_m[k] - exact.origin_m[k]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let p = m.detection_probability(m.reflectivity.terrain, -d[2] / r, r);
                (sum + p, var + p * (1.0 - p))
            })
        };
        let captures = 8;
        let count = |m: &crate::lidar_returns::LidarReturnModel| -> f64 {
            settled_captures(
                rig_without_noise().with_lidar_returns(m.clone()),
                height,
                [10.0, 0.0, 0.0],
                captures,
            )
            .iter()
            .map(|scan| scan.returns.len())
            .sum::<usize>() as f64
        };
        let bright = model(0.25);
        let (mean, var) = expected(&bright);
        let (mean, sigma) = (mean * captures as f64, (var * captures as f64).sqrt());
        let observed = count(&bright);
        assert!(
            (observed - mean).abs() < 4.0 * sigma,
            "{observed} returns against {mean:.1} ± {sigma:.1}"
        );
        assert!(
            mean < 0.9 * (240 * captures) as f64,
            "the curve bites: {mean}"
        );
        assert!(count(&model(0.05)) < 0.5 * observed, "dark returns less");
    }

    /// A persistent range bias moves every return along its ray, so a flat
    /// plane maps that much low at nadir and less off it; quantization
    /// leaves each range on its grid.
    #[test]
    fn a_range_bias_lowers_the_map_and_ranges_sit_on_their_grid() {
        let model = crate::lidar_returns::LidarReturnModel {
            range_bias_m: 0.5,
            ..exact_returns()
        };
        let scan = settled_captures(
            rig_without_noise().with_lidar_returns(model),
            100.0,
            [10.0, 0.0, 0.0],
            1,
        )
        .remove(0);
        let exact = settled_captures(
            rig_without_noise().with_lidar_returns(exact_returns()),
            100.0,
            [10.0, 0.0, 0.0],
            1,
        )
        .remove(0);
        let mean_z = |scan: &RangeScan| {
            scan.returns.iter().map(|r| r.point_m[2]).sum::<f64>() / scan.returns.len() as f64
        };
        // Along each ray, so 0.5 m at nadir and the cosine of the angle off it.
        let lowered = mean_z(&exact) - mean_z(&scan);
        assert!(lowered > 0.4 && lowered < 0.5, "{lowered}");
        let gridded = crate::lidar_returns::LidarReturnModel {
            range_quantization_m: 0.25,
            ..exact_returns()
        };
        let scan = settled_captures(
            rig_without_noise().with_lidar_returns(gridded),
            100.0,
            [0.6, 0.0, 0.0],
            1,
        )
        .remove(0);
        for hit in &scan.returns {
            let offset = hit.acquired_offset_s.unwrap();
            let origin: [f64; 3] =
                std::array::from_fn(|k| scan.origin_m[k] + [0.6, 0.0, 0.0][k] * offset);
            let r: f64 = (0..3)
                .map(|k| (hit.point_m[k] - origin[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let off_grid = (r / 0.25 - (r / 0.25).round()).abs() * 0.25;
            assert!(
                off_grid < 1e-6,
                "range {r} is off the 0.25 m grid by {off_grid}"
            );
        }
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
