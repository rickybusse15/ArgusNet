//! The sensor rig profile (SR1; SENSING_REALISM_IMPLEMENTATION_GUIDE §3): what
//! the aircraft carries, where, on which mount, and what it weighs.
//!
//! A run that declares a rig flies it or is refused before any tick. It is
//! never accepted while the run silently keeps the legacy sensors, so the
//! profile also says which of its sensors the simulation models
//! ([`SensorMount::simulated`]). Today that is the pod LiDAR
//! ([`crate::pod`]) and the forward navigation camera
//! ([`crate::nav_camera`]). The visible cameras, the thermal camera and the
//! mapping IMU are declared (their mounts, mass and fields of view are
//! carried and validated) but produce nothing until SR4. The replay says
//! which is which.
//!
//! # Frames
//!
//! - **World** is ENU: x east, y north, z up, metres.
//! - **Body** is FLU: x forward, y left, z up. The sim has no attitude beyond
//!   heading, so body roll and pitch are zero and its yaw is the horizontal
//!   velocity's direction.
//! - **Pod** is the gimbal's moving frame. It equals the body frame when the
//!   gimbal is at rest: x forward, y left, z up.
//! - **Sensor** frames look along their own +x, with +y left and +z up in the
//!   image.
//!
//! `T_A_B` maps coordinates from B into A, so a pod sensor's pose is
//! `T_world_body(t) · T_body_base · R_base_pod(t) · T_pod_sensor`. Each mount
//! carries its own lever arm, so stacked apertures are never coincident.
//!
//! # Mass
//!
//! The ledger is checked against the guide's two limits: the scanning
//! assembly (≤ 10 lb), which excludes compute, storage, cooling and batteries,
//! and the whole aircraft at take-off (≤ 55 lb), which includes them.
//!
//! # Identity
//!
//! The canonical JSON of the validated profile is hashed (FNV-1a 64) into
//! `meta.world_knowledge.rig_identity`, so two runs with different rigs never
//! look like the same run.

use serde::{Deserialize, Serialize};

use crate::nav_camera::{EvidenceTag, NavCameraConfig};

/// 55 lb in kilograms: the aircraft's take-off mass limit.
pub const MAX_TAKEOFF_MASS_KG: f64 = 24.947_580_35;
/// 10 lb in kilograms: the scanning assembly's mass limit.
pub const MAX_SCANNING_MASS_KG: f64 = 4.535_923_7;
/// The only rig profile version this build reads.
pub const RIG_PROFILE_VERSION: u32 = 1;

/// What a sensor is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorRole {
    Lidar,
    VisibleCamera,
    ThermalCamera,
    MappingImu,
    NavCamera,
}

/// Where a sensor is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attachment {
    /// To the airframe: moves with the body only.
    Body,
    /// To the gimbal's carrier: moves with the pod.
    Pod,
}

/// A fixed rectangular LiDAR's declared capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarModel {
    /// Full horizontal and vertical field of view, degrees.
    pub horizontal_fov_deg: f64,
    pub vertical_fov_deg: f64,
    /// Rays across and up the field of view in one capture.
    pub columns: usize,
    pub rows: usize,
    pub max_range_m: f64,
    /// One-sigma range noise, metres.
    pub range_noise_m: f64,
}

/// A camera's declared optics, for a mount the simulation does not yet
/// render (SR4): enough to carry and check its field of view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraOptics {
    pub horizontal_fov_deg: f64,
    pub vertical_fov_deg: f64,
    pub width_px: u32,
    pub height_px: u32,
}

/// One sensor on the rig.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorMount {
    pub id: String,
    pub role: SensorRole,
    pub attachment: Attachment,
    /// The sensor's origin in its attachment's frame, metres.
    pub lever_arm_m: [f64; 3],
    /// The sensor frame's rotation in its attachment's frame: yaw about z,
    /// then pitch about the new y, then roll about the new x, degrees. Pitch
    /// is positive nose down.
    #[serde(default)]
    pub rotation_ypr_deg: [f64; 3],
    pub model_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lidar: Option<LidarModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraOptics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nav_camera: Option<NavCameraConfig>,
}

impl SensorMount {
    /// Whether this build models the sensor's output.
    pub fn simulated(&self) -> bool {
        matches!(self.role, SensorRole::Lidar | SensorRole::NavCamera)
    }
}

/// A 3-axis gimbal's declared envelope. The maintainer set no joint limits:
/// every axis turns continuously, so the pod can reach any attitude, and the
/// limits that remain are how fast it gets there and how long it takes to
/// settle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GimbalEnvelope {
    /// Always 3: yaw, pitch and roll.
    pub axes: u8,
    /// The carrier's pivot in the body frame, metres.
    pub base_lever_arm_m: [f64; 3],
    /// The fastest the pod's attitude turns, radians per second, about any
    /// axis.
    pub max_rate_rad_s: f64,
    /// How long the pod must hold within the tolerance before a capture is
    /// settled, seconds.
    pub settle_s: f64,
    /// How near the commanded attitude counts as on target, radians.
    pub settle_tolerance_rad: f64,
    /// One-sigma error of the measured attitude (encoders and the mapping
    /// IMU together), radians per axis.
    pub attitude_noise_rad: f64,
    /// The pod's attitude when stowed: yaw, pitch, roll in degrees, in the
    /// body frame.
    pub stow_ypr_deg: [f64; 3],
    /// Rays this far above the body's horizontal plane or higher are blocked
    /// by the airframe above the pod: the housing's occlusion, degrees below
    /// the horizon (negative is above it).
    pub airframe_mask_depression_deg: f64,
}

/// The rig's calibration, as the simulation's truth knows it and the
/// estimator does not.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RigCalibration {
    /// An identity for the calibration the estimator assumes.
    pub epoch: String,
    /// The LiDAR's true misalignment from its nominal mount, yaw, pitch and
    /// roll in radians. The estimator georeferences with the nominal mount,
    /// so this shows in the map as boresight error.
    #[serde(default)]
    pub lidar_boresight_bias_rad: [f64; 3],
}

/// What each item weighs, by the category the limits treat differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MassCategory {
    /// Counts toward the 10 lb scanning assembly: sensors, carrier, gimbal,
    /// housing, windows.
    Scanning,
    Compute,
    Storage,
    Cooling,
    Battery,
    Airframe,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassItem {
    pub name: String,
    pub category: MassCategory,
    pub mass_kg: f64,
}

/// The rig, declared in full.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigProfile {
    pub version: u32,
    pub id: String,
    pub sensors: Vec<SensorMount>,
    pub gimbal: GimbalEnvelope,
    #[serde(default)]
    pub calibration: RigCalibration,
    pub mass: Vec<MassItem>,
    #[serde(default)]
    pub evidence: EvidenceTag,
}

/// The mass ledger's totals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MassTotals {
    pub scanning_kg: f64,
    pub takeoff_kg: f64,
}

impl RigProfile {
    /// The provisional rig: the maintainer's decisions (3-axis gimbal with no
    /// joint limits; LiDAR, two visible cameras, thermal and the mapping IMU
    /// on the pod; a fixed rectangular LiDAR; a forward navigation camera on
    /// the body) with every number assumed.
    pub fn provisional() -> Self {
        let pod = |id: &str, role, lever_arm_m, model: &str| SensorMount {
            id: id.into(),
            role,
            attachment: Attachment::Pod,
            lever_arm_m,
            rotation_ypr_deg: [0.0; 3],
            model_version: model.into(),
            lidar: None,
            camera: None,
            nav_camera: None,
        };
        let optics = |h, v, w, px_h| CameraOptics {
            horizontal_fov_deg: h,
            vertical_fov_deg: v,
            width_px: w,
            height_px: px_h,
        };
        // The LiDAR below two side-by-side visible cameras, thermal above.
        let lidar = SensorMount {
            lidar: Some(LidarModel {
                horizontal_fov_deg: 70.0,
                vertical_fov_deg: 30.0,
                columns: 24,
                rows: 10,
                max_range_m: 250.0,
                range_noise_m: 0.05,
            }),
            ..pod(
                "lidar",
                SensorRole::Lidar,
                [0.06, 0.0, -0.05],
                "assumed-rect-70x30",
            )
        };
        let visible_left = SensorMount {
            camera: Some(optics(60.0, 45.0, 4000, 3000)),
            ..pod(
                "visible-left",
                SensorRole::VisibleCamera,
                [0.06, 0.04, 0.02],
                "assumed",
            )
        };
        let visible_right = SensorMount {
            camera: Some(optics(60.0, 45.0, 4000, 3000)),
            ..pod(
                "visible-right",
                SensorRole::VisibleCamera,
                [0.06, -0.04, 0.02],
                "assumed",
            )
        };
        let thermal = SensorMount {
            camera: Some(optics(45.0, 34.0, 640, 512)),
            ..pod(
                "thermal",
                SensorRole::ThermalCamera,
                [0.06, 0.0, 0.07],
                "assumed",
            )
        };
        let imu = pod(
            "mapping-imu",
            SensorRole::MappingImu,
            [0.0, 0.0, 0.0],
            "assumed",
        );
        let nav = SensorMount {
            id: "nav-camera".into(),
            role: SensorRole::NavCamera,
            attachment: Attachment::Body,
            lever_arm_m: [0.45, 0.0, 0.05],
            // Its boresight depression is its own profile's.
            rotation_ypr_deg: [0.0, 0.0, 0.0],
            model_version: "assumed".into(),
            lidar: None,
            camera: None,
            nav_camera: Some(NavCameraConfig::default()),
        };
        let item = |name: &str, category, mass_kg| MassItem {
            name: name.into(),
            category,
            mass_kg,
        };
        Self {
            version: RIG_PROFILE_VERSION,
            id: "provisional-v1".into(),
            sensors: vec![lidar, visible_left, visible_right, thermal, imu, nav],
            gimbal: GimbalEnvelope {
                axes: 3,
                base_lever_arm_m: [0.1, 0.0, -0.2],
                max_rate_rad_s: 2.0,
                settle_s: 0.2,
                settle_tolerance_rad: 0.5f64.to_radians(),
                attitude_noise_rad: 5.0e-4,
                stow_ypr_deg: [0.0, 0.0, 0.0],
                airframe_mask_depression_deg: 5.0,
            },
            calibration: RigCalibration {
                epoch: "assumed-nominal".into(),
                lidar_boresight_bias_rad: [1.0e-3, 2.0e-3, 0.0],
            },
            mass: vec![
                item("lidar", MassCategory::Scanning, 0.5),
                item("visible cameras", MassCategory::Scanning, 0.3),
                item("thermal camera", MassCategory::Scanning, 0.2),
                item("mapping imu", MassCategory::Scanning, 0.05),
                item(
                    "gimbal, carrier, housing, windows",
                    MassCategory::Scanning,
                    1.6,
                ),
                item("compute", MassCategory::Compute, 0.4),
                item("storage", MassCategory::Storage, 0.1),
                item("cooling", MassCategory::Cooling, 0.2),
                item("batteries", MassCategory::Battery, 9.0),
                item("airframe and propulsion", MassCategory::Airframe, 10.0),
                item("navigation camera", MassCategory::Other, 0.1),
                item("wiring and margin", MassCategory::Other, 1.0),
            ],
            evidence: EvidenceTag::Assumed,
        }
    }

    pub fn mass_totals(&self) -> MassTotals {
        let scanning_kg = self
            .mass
            .iter()
            .filter(|item| item.category == MassCategory::Scanning)
            .map(|item| item.mass_kg)
            .sum();
        let takeoff_kg = self.mass.iter().map(|item| item.mass_kg).sum();
        MassTotals {
            scanning_kg,
            takeoff_kg,
        }
    }

    /// The first mount with `role`.
    pub fn mount(&self, role: SensorRole) -> Option<&SensorMount> {
        self.sensors.iter().find(|mount| mount.role == role)
    }

    /// Everything the run relies on, checked before it mutates anything.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != RIG_PROFILE_VERSION {
            return Err(format!(
                "rig profile version {} is not supported (this build reads {RIG_PROFILE_VERSION})",
                self.version
            ));
        }
        if self.id.is_empty() {
            return Err("the rig profile has no id".into());
        }
        let finite3 = |name: &str, v: [f64; 3]| {
            if v.iter().all(|x| x.is_finite()) {
                Ok(())
            } else {
                Err(format!("rig {name} has a non-finite value: {v:?}"))
            }
        };
        let mut ids = std::collections::BTreeSet::new();
        for mount in &self.sensors {
            if mount.id.is_empty() || !ids.insert(mount.id.as_str()) {
                return Err(format!("rig sensor id {:?} is empty or repeated", mount.id));
            }
            finite3(&format!("{} lever arm", mount.id), mount.lever_arm_m)?;
            finite3(&format!("{} rotation", mount.id), mount.rotation_ypr_deg)?;
            if mount.model_version.is_empty() {
                return Err(format!("rig sensor {} names no model version", mount.id));
            }
            let expected = match mount.role {
                SensorRole::NavCamera => Attachment::Body,
                _ => Attachment::Pod,
            };
            if mount.attachment != expected {
                return Err(format!(
                    "rig sensor {} ({:?}) must be on the {:?}",
                    mount.id, mount.role, expected
                ));
            }
            match mount.role {
                SensorRole::Lidar => {
                    let lidar = mount
                        .lidar
                        .as_ref()
                        .ok_or_else(|| format!("rig LiDAR {} has no model", mount.id))?;
                    validate_lidar(&mount.id, lidar)?;
                }
                SensorRole::VisibleCamera | SensorRole::ThermalCamera => {
                    let optics = mount
                        .camera
                        .as_ref()
                        .ok_or_else(|| format!("rig camera {} has no optics", mount.id))?;
                    if !(fov_ok(optics.horizontal_fov_deg) && fov_ok(optics.vertical_fov_deg))
                        || optics.width_px == 0
                        || optics.height_px == 0
                    {
                        return Err(format!("rig camera {} has invalid optics", mount.id));
                    }
                }
                SensorRole::NavCamera => {
                    mount
                        .nav_camera
                        .as_ref()
                        .ok_or_else(|| {
                            format!("rig navigation camera {} has no profile", mount.id)
                        })?
                        .validate()?;
                }
                SensorRole::MappingImu => {}
            }
        }
        for role in [SensorRole::Lidar, SensorRole::NavCamera] {
            let count = self.sensors.iter().filter(|m| m.role == role).count();
            if count != 1 {
                return Err(format!(
                    "the rig must carry exactly one {role:?}, found {count}"
                ));
            }
        }
        let gimbal = &self.gimbal;
        if gimbal.axes != 3 {
            return Err(format!(
                "the gimbal has {} axes; this rig is 3-axis",
                gimbal.axes
            ));
        }
        finite3("gimbal base lever arm", gimbal.base_lever_arm_m)?;
        finite3("gimbal stow attitude", gimbal.stow_ypr_deg)?;
        let positive = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(format!("gimbal {name} must be positive, got {v}"))
            }
        };
        positive("max_rate_rad_s", gimbal.max_rate_rad_s)?;
        positive("settle_tolerance_rad", gimbal.settle_tolerance_rad)?;
        if !(gimbal.settle_s.is_finite() && gimbal.settle_s >= 0.0)
            || !(gimbal.attitude_noise_rad.is_finite() && gimbal.attitude_noise_rad >= 0.0)
        {
            return Err("gimbal settle_s and attitude_noise_rad must not be negative".into());
        }
        if !(gimbal.airframe_mask_depression_deg.is_finite()
            && gimbal.airframe_mask_depression_deg.abs() <= 90.0)
        {
            return Err("gimbal airframe_mask_depression_deg must be within ±90".into());
        }
        finite3(
            "LiDAR boresight bias",
            self.calibration.lidar_boresight_bias_rad,
        )?;
        if self.calibration.epoch.is_empty() {
            return Err("the rig calibration names no epoch".into());
        }
        if self
            .mass
            .iter()
            .any(|item| !(item.mass_kg.is_finite() && item.mass_kg >= 0.0))
        {
            return Err("every rig mass item must be a non-negative weight".into());
        }
        let totals = self.mass_totals();
        if totals.scanning_kg > MAX_SCANNING_MASS_KG {
            return Err(format!(
                "the scanning assembly weighs {:.3} kg, over the 10 lb ({MAX_SCANNING_MASS_KG} kg) limit",
                totals.scanning_kg
            ));
        }
        if totals.takeoff_kg > MAX_TAKEOFF_MASS_KG {
            return Err(format!(
                "the aircraft weighs {:.3} kg at take-off, over the 55 lb ({MAX_TAKEOFF_MASS_KG} kg) limit",
                totals.takeoff_kg
            ));
        }
        Ok(())
    }

    /// The profile's identity: FNV-1a 64 of its canonical JSON, as hex.
    pub fn identity(&self) -> String {
        let text = serde_json::to_string(self).expect("a rig profile serialises");
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("fnv1a64:{hash:016x}")
    }

    /// Which sensors the run models and which it only carries.
    pub fn sensor_summary(&self) -> serde_json::Value {
        let ids = |simulated: bool| -> Vec<&str> {
            self.sensors
                .iter()
                .filter(|m| m.simulated() == simulated)
                .map(|m| m.id.as_str())
                .collect()
        };
        serde_json::json!({
            "simulated": ids(true),
            "declared_not_simulated": ids(false),
        })
    }
}

fn fov_ok(deg: f64) -> bool {
    deg.is_finite() && deg > 0.0 && deg < 180.0
}

fn validate_lidar(id: &str, lidar: &LidarModel) -> Result<(), String> {
    if !(fov_ok(lidar.horizontal_fov_deg) && fov_ok(lidar.vertical_fov_deg)) {
        return Err(format!(
            "rig LiDAR {id} field of view must be within (0, 180) degrees"
        ));
    }
    if lidar.columns < 2 || lidar.rows < 2 {
        return Err(format!(
            "rig LiDAR {id} needs at least 2 columns and 2 rows"
        ));
    }
    if lidar
        .columns
        .checked_mul(lidar.rows)
        .is_none_or(|rays| rays > crate::nav_camera::MAX_CAPTURE_RAYS)
    {
        return Err(format!("rig LiDAR {id} exceeds the rays per capture limit"));
    }
    if !(lidar.max_range_m.is_finite() && lidar.max_range_m > 0.0) {
        return Err(format!("rig LiDAR {id} max range must be positive"));
    }
    if !(lidar.range_noise_m.is_finite() && lidar.range_noise_m >= 0.0) {
        return Err(format!("rig LiDAR {id} range noise must not be negative"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_and_overflowing_lidar_grids_are_refused_before_capture() {
        for columns in [usize::MAX, 1_000_001] {
            let mut rig = RigProfile::provisional();
            let lidar = rig
                .sensors
                .iter_mut()
                .find_map(|mount| mount.lidar.as_mut())
                .unwrap();
            lidar.columns = columns;
            lidar.rows = 2;
            assert!(rig.validate().unwrap_err().contains("rays per capture"));
        }
    }

    #[test]
    fn the_provisional_rig_is_valid_and_within_both_mass_limits() {
        let rig = RigProfile::provisional();
        rig.validate().unwrap();
        let totals = rig.mass_totals();
        assert!((totals.scanning_kg - 2.65).abs() < 1e-9, "{totals:?}");
        assert!(totals.takeoff_kg <= MAX_TAKEOFF_MASS_KG, "{totals:?}");
        let summary = rig.sensor_summary();
        assert_eq!(
            summary["simulated"],
            serde_json::json!(["lidar", "nav-camera"])
        );
        assert_eq!(
            summary["declared_not_simulated"],
            serde_json::json!(["visible-left", "visible-right", "thermal", "mapping-imu"])
        );
    }

    #[test]
    fn a_rig_over_either_mass_limit_is_refused() {
        let mut heavy_pod = RigProfile::provisional();
        heavy_pod.mass[4].mass_kg = 4.0;
        let error = heavy_pod.validate().unwrap_err();
        assert!(error.contains("10 lb"), "{error}");
        let mut heavy_aircraft = RigProfile::provisional();
        heavy_aircraft.mass[8].mass_kg = 12.0;
        let error = heavy_aircraft.validate().unwrap_err();
        assert!(error.contains("55 lb"), "{error}");
        // Batteries do not count toward the scanning assembly.
        let totals = heavy_aircraft.mass_totals();
        assert!(totals.scanning_kg < MAX_SCANNING_MASS_KG);
    }

    #[test]
    fn malformed_rigs_are_refused() {
        type Mutation = Box<dyn Fn(&mut RigProfile)>;
        let cases: Vec<(&str, Mutation)> = vec![
            ("version", Box::new(|r| r.version = 2)),
            ("repeated", Box::new(|r| r.sensors[1].id = "lidar".into())),
            (
                "must be on the",
                Box::new(|r| r.sensors[0].attachment = Attachment::Body),
            ),
            (
                "exactly one",
                Box::new(|r| r.sensors.retain(|m| m.role != SensorRole::Lidar)),
            ),
            (
                "non-finite",
                Box::new(|r| r.sensors[0].lever_arm_m[2] = f64::NAN),
            ),
            ("3-axis", Box::new(|r| r.gimbal.axes = 2)),
            ("max_rate", Box::new(|r| r.gimbal.max_rate_rad_s = 0.0)),
            ("no model", Box::new(|r| r.sensors[0].lidar = None)),
            ("epoch", Box::new(|r| r.calibration.epoch.clear())),
        ];
        for (expected, mutate) in cases {
            let mut rig = RigProfile::provisional();
            mutate(&mut rig);
            let error = rig.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn identity_changes_with_the_rig_and_survives_a_round_trip() {
        let rig = RigProfile::provisional();
        let again: RigProfile =
            serde_json::from_str(&serde_json::to_string(&rig).unwrap()).unwrap();
        assert_eq!(rig.identity(), again.identity());
        let mut other = rig.clone();
        other.calibration.lidar_boresight_bias_rad[0] = 0.0;
        assert_ne!(rig.identity(), other.identity());
    }
}
