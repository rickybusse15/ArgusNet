//! A mission before it is flown: what its declared sensors, batteries and limits say
//! of it, with nothing of the world in it.
//!
//! Every number is worked out from what is declared: the map's extent and the
//! requirement, one pass of the sensors, the aircraft's energy model, where the pads
//! and points are, and the run's length. Whether a point is hidden, what the ground
//! does and where wires hang is for the mission to find out, so a plan that checks out
//! is not one that will succeed. One that does not is refused with the limit it meets.

use serde::{Deserialize, Serialize};

use argusnet_mapping::quality::QualityRequirement;
use argusnet_planning::battery::BatteryModel;

use crate::map_report::PassPrediction;

pub const PLAN_FORMAT: &str = "argusnet.mission-plan";
pub const PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Ok,
    /// Allowed, and close.
    Warning,
    /// Not possible as declared.
    Failed,
    /// Nothing declared says.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedDrone {
    pub drone_id: String,
    pub home_m: [f64; 3],
    pub energy_wh: f64,
    /// What it must keep at its pad.
    pub reserve_wh: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedPoint {
    pub poi_id: String,
    pub position_m: [f64; 3],
    pub dwell_s: f64,
    pub priority: i64,
    /// The drone that could do it for least, from its pad and back.
    pub nearest_drone: Option<String>,
    pub out_m: Option<f64>,
    pub energy_wh: Option<f64>,
    /// What that drone would have left over its reserve.
    pub margin_wh: Option<f64>,
    pub feasible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedScan {
    pub area_m2: f64,
    /// The share of it the scan ends at.
    pub gate_fraction: f64,
    pub pass: PassPrediction,
    /// Passes one over the other the density asked for takes.
    pub passes: f64,
    /// The track there is to fly, and each drone's share in time and energy,
    /// with no overlap, turn or transit: the least it can take.
    pub track_m: f64,
    pub least_s: f64,
    pub least_wh_per_drone: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionPlan {
    pub format: String,
    pub version: u32,
    pub scenario_name: String,
    pub seed: u32,
    pub run_profile: Option<String>,
    pub duration_s: f64,
    /// `[x0, y0, x1, y1]`.
    pub bounds_m: [f64; 4],
    pub requirement: Option<QualityRequirement>,
    pub drones: Vec<PlannedDrone>,
    pub points: Vec<PlannedPoint>,
    pub scan: Option<PlannedScan>,
    pub checks: Vec<Check>,
    /// The worst of the checks.
    pub status: CheckStatus,
}

impl MissionPlan {
    /// Refuse a document this build does not read.
    pub fn check_format(&self) -> Result<(), String> {
        if self.format != PLAN_FORMAT || self.version != PLAN_VERSION {
            return Err(format!(
                "not a mission plan this build reads: {} version {}",
                self.format, self.version
            ));
        }
        Ok(())
    }
}

/// What a plan is worked out from.
pub struct PlanInput<'a> {
    pub scenario_name: &'a str,
    pub seed: u32,
    pub run_profile: Option<String>,
    pub duration_s: f64,
    pub bounds_m: [f64; 4],
    pub battery: &'a BatteryModel,
    /// Each drone, its pad and its charge.
    pub drones: Vec<(String, [f64; 3], f64)>,
    /// Each point: id, place, dwell, priority.
    pub points: Vec<(String, [f64; 3], f64, i64)>,
    pub redirect_speed_mps: f64,
    pub requirement: Option<QualityRequirement>,
    pub gate_fraction: f64,
    pub pass: Option<PassPrediction>,
    pub finest_image_gsd_m: Option<f64>,
}

/// Past this share of a limit a plan is close to it.
const CLOSE: f64 = 0.8;

pub fn build(input: PlanInput<'_>) -> MissionPlan {
    let battery = input.battery;
    let drones: Vec<PlannedDrone> = input
        .drones
        .iter()
        .map(|(drone_id, home_m, energy_wh)| PlannedDrone {
            drone_id: drone_id.clone(),
            home_m: *home_m,
            energy_wh: *energy_wh,
            reserve_wh: battery.dynamic_reserve_wh(
                *home_m,
                *home_m,
                None,
                BatteryModel::DEFAULT_SAFETY_MARGIN,
            ),
        })
        .collect();
    let points: Vec<PlannedPoint> = input
        .points
        .iter()
        .map(|(poi_id, at, dwell_s, priority)| {
            let trip = |drone: &PlannedDrone| {
                let out = battery.segment_cost_wh(drone.home_m, *at);
                let dwell = battery.hover_power_w * dwell_s / 3600.0;
                out + dwell + battery.return_home_cost_wh(*at, drone.home_m, None)
            };
            let best = drones
                .iter()
                .map(|drone| (trip(drone), drone))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.drone_id.cmp(&b.1.drone_id)));
            let margin = best.map(|(cost, drone)| drone.energy_wh - cost - drone.reserve_wh);
            PlannedPoint {
                poi_id: poi_id.clone(),
                position_m: *at,
                dwell_s: *dwell_s,
                priority: *priority,
                nearest_drone: best.map(|(_, drone)| drone.drone_id.clone()),
                out_m: best
                    .map(|(_, drone)| (at[0] - drone.home_m[0]).hypot(at[1] - drone.home_m[1])),
                energy_wh: best.map(|(cost, _)| cost),
                margin_wh: margin,
                feasible: margin.is_some_and(|margin| margin >= 0.0),
            }
        })
        .collect();
    let [x0, y0, x1, y1] = input.bounds_m;
    let area_m2 = (x1 - x0) * (y1 - y0);
    let scan = input.pass.clone().map(|pass| {
        let asked = input
            .requirement
            .as_ref()
            .and_then(|r| r.min_ground_density_per_m2)
            .unwrap_or(0.0);
        let passes = (asked / pass.density_per_m2.max(1e-12)).max(1.0);
        let track_m = area_m2 * input.gate_fraction * passes / pass.swath_m.max(1e-9);
        let share_m = track_m / drones.len().max(1) as f64;
        PlannedScan {
            area_m2,
            gate_fraction: input.gate_fraction,
            passes,
            track_m,
            least_s: share_m / pass.speed_mps.max(1e-9),
            least_wh_per_drone: battery.travel_cost_wh(share_m),
            pass,
        }
    });

    let mut checks = Vec::new();
    let mut check = |name: &str, status: CheckStatus, detail: String| {
        checks.push(Check {
            name: name.into(),
            status,
            detail,
        });
    };
    check(
        "drones",
        if drones.is_empty() {
            CheckStatus::Failed
        } else {
            CheckStatus::Ok
        },
        format!("{} drones", drones.len()),
    );
    let outside: Vec<&str> = points
        .iter()
        .filter(|p| {
            let [x, y, _] = p.position_m;
            x < x0 || x > x1 || y < y0 || y > y1
        })
        .map(|p| p.poi_id.as_str())
        .collect();
    check(
        "points_on_the_map",
        if outside.is_empty() {
            CheckStatus::Ok
        } else {
            CheckStatus::Failed
        },
        if outside.is_empty() {
            format!("{} points", points.len())
        } else {
            format!("off the map: {}", outside.join(", "))
        },
    );
    let unreachable: Vec<&str> = points
        .iter()
        .filter(|p| !p.feasible)
        .map(|p| p.poi_id.as_str())
        .collect();
    let least_margin = points
        .iter()
        .filter_map(|p| p.margin_wh)
        .fold(f64::INFINITY, f64::min);
    check(
        "points_within_reach",
        if !unreachable.is_empty() {
            CheckStatus::Failed
        } else if points.is_empty() {
            CheckStatus::Unknown
        } else if least_margin < (1.0 - CLOSE) * battery.capacity_wh {
            CheckStatus::Warning
        } else {
            CheckStatus::Ok
        },
        if !unreachable.is_empty() {
            format!(
                "no drone can go, dwell and come home with its reserve: {}",
                unreachable.join(", ")
            )
        } else if points.is_empty() {
            "no points".into()
        } else {
            format!("the least left over a reserve is {least_margin:.0} Wh")
        },
    );
    match (&input.requirement, &scan) {
        (Some(requirement), Some(scan)) => {
            let sigma = requirement.max_vertical_sigma_m;
            let reached = scan.pass.vertical_sigma_m.max(requirement.bias_floor_m);
            check(
                "vertical_accuracy",
                match sigma {
                    None => CheckStatus::Unknown,
                    Some(max) if reached > max => CheckStatus::Failed,
                    Some(max) if reached > CLOSE * max => CheckStatus::Warning,
                    Some(_) => CheckStatus::Ok,
                },
                match sigma {
                    None => "not asked".into(),
                    Some(max) => format!("{reached:.2} m a pass, {max:.2} m asked"),
                },
            );
            check(
                "image_detail",
                match (requirement.max_image_gsd_m, input.finest_image_gsd_m) {
                    (None, _) => CheckStatus::Unknown,
                    (Some(_), None) => CheckStatus::Failed,
                    (Some(asked), Some(finest)) if finest > asked => CheckStatus::Failed,
                    _ => match (requirement.max_image_gsd_m, scan.pass.image_gsd_m) {
                        (Some(asked), Some(gsd)) if gsd > asked => CheckStatus::Warning,
                        _ => CheckStatus::Ok,
                    },
                },
                match (requirement.max_image_gsd_m, scan.pass.image_gsd_m) {
                    (None, _) => "not asked".into(),
                    (Some(asked), None) => format!("{asked:.3} m asked, no camera modelled"),
                    (Some(asked), Some(gsd)) => {
                        format!("{gsd:.3} m a pixel straight down, {asked:.3} m asked")
                    }
                },
            );
            check(
                "density",
                if scan.passes > 1.0 {
                    CheckStatus::Warning
                } else {
                    CheckStatus::Ok
                },
                format!(
                    "{:.3} returns a square metre a pass: {:.1} passes",
                    scan.pass.density_per_m2, scan.passes
                ),
            );
            let dwell: f64 = points.iter().map(|p| p.dwell_s).sum();
            let transit: f64 = points
                .iter()
                .filter_map(|p| p.out_m)
                .map(|m| 2.0 * m / input.redirect_speed_mps.max(1e-9))
                .sum();
            let least = scan.least_s + (dwell + transit) / drones.len().max(1) as f64;
            check(
                "time",
                if least > input.duration_s {
                    CheckStatus::Failed
                } else if least > CLOSE * input.duration_s {
                    CheckStatus::Warning
                } else {
                    CheckStatus::Ok
                },
                format!(
                    "at least {least:.0} s of flying, {:.0} s in the run",
                    input.duration_s
                ),
            );
            let room = drones
                .iter()
                .map(|d| d.energy_wh - d.reserve_wh)
                .fold(f64::INFINITY, f64::min);
            check(
                "scan_energy",
                if scan.least_wh_per_drone > room {
                    CheckStatus::Failed
                } else if scan.least_wh_per_drone > CLOSE * room {
                    CheckStatus::Warning
                } else {
                    CheckStatus::Ok
                },
                format!(
                    "at least {:.0} Wh a drone, {room:.0} Wh over its reserve",
                    scan.least_wh_per_drone
                ),
            );
        }
        _ => check(
            "survey",
            CheckStatus::Unknown,
            "no requirement and declared sensors to survey with".into(),
        ),
    }
    let rank = |status: CheckStatus| match status {
        CheckStatus::Ok | CheckStatus::Unknown => 0,
        CheckStatus::Warning => 1,
        CheckStatus::Failed => 2,
    };
    let status = checks
        .iter()
        .map(|c| c.status)
        .filter(|s| *s != CheckStatus::Unknown)
        .max_by_key(|s| rank(*s))
        .unwrap_or(CheckStatus::Unknown);
    MissionPlan {
        format: PLAN_FORMAT.into(),
        version: PLAN_VERSION,
        scenario_name: input.scenario_name.into(),
        seed: input.seed,
        run_profile: input.run_profile,
        duration_s: input.duration_s,
        bounds_m: input.bounds_m,
        requirement: input.requirement,
        drones,
        points,
        scan,
        checks,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement() -> QualityRequirement {
        QualityRequirement {
            min_ground_density_per_m2: Some(0.03),
            max_vertical_sigma_m: Some(0.5),
            max_age_s: None,
            max_image_gsd_m: Some(0.04),
            max_horizontal_sigma_m: None,
            max_incidence_deg: 40.0,
            bias_floor_m: 0.05,
        }
    }

    fn pass() -> PassPrediction {
        PassPrediction {
            agl_m: 60.0,
            speed_mps: 20.0,
            swath_m: 100.0,
            density_per_m2: 0.2,
            vertical_sigma_m: 0.07,
            image_gsd_m: Some(0.036),
        }
    }

    fn plan(change: impl FnOnce(&mut PlanInput<'_>)) -> MissionPlan {
        let battery = BatteryModel {
            capacity_wh: 500.0,
            ..BatteryModel::default()
        };
        let mut input = PlanInput {
            scenario_name: "s",
            seed: 7,
            run_profile: Some("field_v3".into()),
            duration_s: 600.0,
            bounds_m: [-500.0, -500.0, 500.0, 500.0],
            battery: &battery,
            drones: vec![
                ("a".into(), [0.0, -400.0, 0.0], 500.0),
                ("b".into(), [0.0, 400.0, 0.0], 500.0),
            ],
            points: vec![("p".into(), [100.0, 300.0, 40.0], 20.0, 1)],
            redirect_speed_mps: 20.0,
            requirement: Some(requirement()),
            gate_fraction: 0.9,
            pass: Some(pass()),
            finest_image_gsd_m: Some(0.01),
        };
        change(&mut input);
        build(input)
    }

    fn status(plan: &MissionPlan, name: &str) -> CheckStatus {
        plan.checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("{name} in {:?}", plan.checks))
            .status
    }

    #[test]
    fn a_mission_within_its_limits_checks_out() {
        let plan = plan(|_| {});
        assert_eq!(plan.status, CheckStatus::Ok, "{:?}", plan.checks);
        let scan = plan.scan.as_ref().unwrap();
        // 1 km², nine tenths, 100 m swath, two drones at 20 m/s.
        assert!((scan.track_m - 9000.0).abs() < 1e-6);
        assert!((scan.least_s - 225.0).abs() < 1e-6);
        let point = &plan.points[0];
        assert_eq!(point.nearest_drone.as_deref(), Some("b"));
        assert!(point.feasible && point.margin_wh.unwrap() > 300.0);
        let text = serde_json::to_string(&plan).unwrap();
        let read: MissionPlan = serde_json::from_str(&text).unwrap();
        assert_eq!(read, plan);
        assert!(read.check_format().is_ok());
    }

    #[test]
    fn each_limit_is_the_one_named() {
        let failed = |name: &str, change: fn(&mut PlanInput<'_>)| {
            let plan = plan(change);
            assert_eq!(status(&plan, name), CheckStatus::Failed, "{name}");
            assert_eq!(plan.status, CheckStatus::Failed);
        };
        failed("drones", |input| input.drones.clear());
        failed("points_on_the_map", |input| {
            input.points[0].1 = [600.0, 0.0, 40.0];
        });
        failed("points_within_reach", |input| {
            input.points[0].2 = 10_000.0;
        });
        failed("vertical_accuracy", |input| {
            input.requirement.as_mut().unwrap().bias_floor_m = 0.6;
        });
        failed("image_detail", |input| {
            input.finest_image_gsd_m = Some(0.1);
        });
        failed("time", |input| input.duration_s = 200.0);
        failed("scan_energy", |input| {
            for drone in &mut input.drones {
                drone.2 = 110.0;
            }
        });
    }

    #[test]
    fn close_to_a_limit_is_a_warning() {
        let close = plan(|input| input.duration_s = 280.0);
        assert_eq!(status(&close, "time"), CheckStatus::Warning);
        assert_eq!(close.status, CheckStatus::Warning);
        let twice = plan(|input| {
            input.pass.as_mut().unwrap().density_per_m2 = 0.015;
        });
        assert_eq!(status(&twice, "density"), CheckStatus::Warning);
        assert!((twice.scan.unwrap().passes - 2.0).abs() < 1e-9);
    }

    #[test]
    fn what_is_not_declared_is_unknown_and_not_passed() {
        let plan = plan(|input| {
            input.pass = None;
            input.requirement = None;
            input.points.clear();
        });
        assert_eq!(status(&plan, "survey"), CheckStatus::Unknown);
        assert_eq!(status(&plan, "points_within_reach"), CheckStatus::Unknown);
        assert!(plan.scan.is_none());
        let mut other = plan.clone();
        other.version = 2;
        assert!(other.check_format().is_err());
    }
}
