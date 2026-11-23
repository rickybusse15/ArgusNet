//! The Plan workspace: a mission plan `argusnet-rs plan` wrote, read only.
//!
//! The viewer works nothing out. What it shows is the backend's document, its
//! checks and their limits; with none loaded it says how to make one. Authoring a
//! plan here waits for the canonical contracts (S1).

use std::path::Path;
use std::sync::OnceLock;

use argusnet_sim::plan_report::{CheckStatus, MissionPlan};
use bevy_egui::egui;

use crate::theme::Status;

/// The plan the viewer was given.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PlanState {
    #[default]
    None,
    /// What was given is not one, and why.
    Invalid(String),
    Loaded(Box<MissionPlan>),
}

static GIVEN: OnceLock<PlanState> = OnceLock::new();

/// The plan to show, from the command line. Before the app starts.
pub fn give_plan(path: &Path) {
    let _ = GIVEN.set(read_plan(path));
}

/// The plan given, if any.
pub fn given_plan() -> PlanState {
    GIVEN.get().cloned().unwrap_or_default()
}

/// A plan document, or why it is not one.
pub fn read_plan(path: &Path) -> PlanState {
    let read = std::fs::read_to_string(path)
        .map_err(|error| error.to_string())
        .and_then(|text| {
            serde_json::from_str::<MissionPlan>(&text).map_err(|error| error.to_string())
        })
        .and_then(|plan| plan.check_format().map(|()| plan));
    match read {
        Ok(plan) => PlanState::Loaded(Box::new(plan)),
        Err(error) => PlanState::Invalid(format!("{}: {error}", path.display())),
    }
}

fn status_of(status: CheckStatus) -> Status {
    match status {
        CheckStatus::Ok => Status::Nominal,
        CheckStatus::Warning => Status::Degraded,
        CheckStatus::Failed => Status::Critical,
        CheckStatus::Unknown => Status::Unavailable,
    }
}

fn word(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Ok => "ok",
        CheckStatus::Warning => "close to a limit",
        CheckStatus::Failed => "cannot be flown as declared",
        CheckStatus::Unknown => "unknown",
    }
}

/// The lines under a heading, each with its status.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PlanSection {
    pub title: &'static str,
    pub lines: Vec<(Status, String)>,
}

/// What the rail shows of a plan.
pub(super) fn sections(plan: &MissionPlan) -> Vec<PlanSection> {
    let mut sections = vec![PlanSection {
        title: "Checks",
        lines: plan
            .checks
            .iter()
            .map(|check| {
                (
                    status_of(check.status),
                    format!("{}: {}", check.name.replace('_', " "), check.detail),
                )
            })
            .collect(),
    }];
    if let Some(scan) = &plan.scan {
        sections.push(PlanSection {
            title: "Scan",
            lines: vec![
                (
                    Status::Nominal,
                    format!(
                        "{:.2} km2, to {:.0}% of it",
                        scan.area_m2 / 1.0e6,
                        scan.gate_fraction * 100.0
                    ),
                ),
                (
                    Status::Nominal,
                    format!(
                        "{:.0} m up at {:.1} m/s, {:.0} m swath",
                        scan.pass.agl_m, scan.pass.speed_mps, scan.pass.swath_m
                    ),
                ),
                (
                    Status::Nominal,
                    format!(
                        "at least {:.0} s and {:.0} Wh a drone",
                        scan.least_s, scan.least_wh_per_drone
                    ),
                ),
            ],
        });
    }
    sections.push(PlanSection {
        title: "Points",
        lines: plan
            .points
            .iter()
            .map(|point| {
                let status = if point.feasible {
                    Status::Nominal
                } else {
                    Status::Critical
                };
                let reach = match (&point.nearest_drone, point.out_m, point.margin_wh) {
                    (Some(drone), Some(out), Some(margin)) => {
                        format!("{drone}, {out:.0} m, {margin:.0} Wh over its reserve")
                    }
                    _ => "no drone".into(),
                };
                (
                    status,
                    format!("{} ({:.0} s): {reach}", point.poi_id, point.dwell_s),
                )
            })
            .collect(),
    });
    sections.push(PlanSection {
        title: "Drones",
        lines: plan
            .drones
            .iter()
            .map(|drone| {
                (
                    Status::Nominal,
                    format!(
                        "{}: {:.0} Wh, {:.0} Wh reserve",
                        drone.drone_id, drone.energy_wh, drone.reserve_wh
                    ),
                )
            })
            .collect(),
    });
    sections
}

/// The Plan rail: the plan given, or why there is none.
pub(super) fn plan_rail(ui: &mut egui::Ui, state: &PlanState) {
    ui.strong("Plan");
    let plan = match state {
        PlanState::None => {
            ui.label(Status::Unavailable.rich_text());
            ui.label("No mission plan is loaded.");
            ui.small(
                "argusnet-rs plan --config SCENARIO --profile field_v3 --output plan.json, \
                 then open the viewer with --run SCENARIO --plan plan.json. Writing a plan here waits \
                 for the \
                 canonical contracts (S1).",
            );
            return;
        }
        PlanState::Invalid(why) => {
            ui.label(Status::Critical.rich_text());
            ui.label(format!("The mission plan could not be read: {why}"));
            return;
        }
        PlanState::Loaded(plan) => plan,
    };
    let status = status_of(plan.status);
    ui.colored_label(
        status.color(),
        egui::RichText::new(word(plan.status)).strong(),
    );
    ui.small(format!(
        "{} · seed {} · {} · {:.0} s",
        plan.scenario_name,
        plan.seed,
        plan.run_profile.as_deref().unwrap_or("no profile"),
        plan.duration_s
    ));
    ui.small("From what is declared, with nothing of the world: one that checks out may fail.");
    for section in sections(plan) {
        egui::CollapsingHeader::new(format!("{} ({})", section.title, section.lines.len()))
            .default_open(section.title != "Drones")
            .show(ui, |ui| {
                if section.lines.is_empty() {
                    ui.small("None.");
                }
                for (status, line) in &section.lines {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(status.color(), status.glyph().to_string());
                        ui.small(line);
                    });
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> serde_json::Value {
        serde_json::json!({
            "format": "argusnet.mission-plan",
            "version": 1,
            "scenario_name": "town",
            "seed": 7,
            "run_profile": "field_v3",
            "duration_s": 600.0,
            "bounds_m": [-500.0, -500.0, 500.0, 500.0],
            "requirement": null,
            "drones": [{
                "drone_id": "a", "home_m": [0.0, 0.0, 0.0],
                "energy_wh": 500.0, "reserve_wh": 100.0
            }],
            "points": [
                {
                    "poi_id": "near", "position_m": [1.0, 2.0, 3.0], "dwell_s": 10.0,
                    "priority": 1, "nearest_drone": "a", "out_m": 50.0, "energy_wh": 5.0,
                    "margin_wh": 395.0, "feasible": true
                },
                {
                    "poi_id": "far", "position_m": [1.0, 2.0, 3.0], "dwell_s": 9000.0,
                    "priority": 1, "nearest_drone": "a", "out_m": 50.0, "energy_wh": 900.0,
                    "margin_wh": -500.0, "feasible": false
                }
            ],
            "scan": null,
            "checks": [
                {"name": "points_within_reach", "status": "failed", "detail": "far"},
                {"name": "survey", "status": "unknown", "detail": "not declared"}
            ],
            "status": "failed"
        })
    }

    fn written(name: &str, text: &str) -> PlanState {
        let path =
            std::env::temp_dir().join(format!("argusnet-plan-{}-{name}", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let state = read_plan(&path);
        std::fs::remove_file(&path).unwrap();
        state
    }

    #[test]
    fn a_plan_is_shown_as_the_backend_wrote_it() {
        let PlanState::Loaded(plan) = written("good", &document().to_string()) else {
            panic!("a plan")
        };
        let sections = sections(&plan);
        let titles: Vec<_> = sections.iter().map(|s| s.title).collect();
        assert_eq!(titles, ["Checks", "Points", "Drones"], "no scan declared");
        assert_eq!(
            sections[0].lines[0],
            (Status::Critical, "points within reach: far".to_string())
        );
        // What is not declared is unavailable, not passed.
        assert_eq!(sections[0].lines[1].0, Status::Unavailable);
        assert_eq!(sections[1].lines[0].0, Status::Nominal);
        assert_eq!(sections[1].lines[1].0, Status::Critical);
        assert!(sections[1].lines[0].1.contains("395 Wh over its reserve"));
    }

    #[test]
    fn what_is_not_a_plan_says_so() {
        for (name, text) in [
            ("empty", String::new()),
            ("other", "{\"format\": \"argusnet.terrain\"}".into()),
            ("later", {
                let mut later = document();
                later["version"] = 2.into();
                later.to_string()
            }),
        ] {
            assert!(
                matches!(written(name, &text), PlanState::Invalid(_)),
                "{name}"
            );
        }
        assert!(matches!(
            read_plan(Path::new("/no/such/plan.json")),
            PlanState::Invalid(_)
        ));
        assert_eq!(given_plan(), PlanState::None);
    }
}
