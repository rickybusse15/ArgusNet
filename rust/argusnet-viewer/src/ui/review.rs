//! The Review workspace (S1.2): the run's assurance report, read from the
//! replay a client package annotated (`summary.mission_assurance`,
//! `argusnet.mission-assurance/v1`).
//!
//! What it shows comes from one pure [`review_view`]:
//! - the recorded status and the F1 outcome;
//! - the outcome's reasons as findings, each jumping to the first frame it
//!   rests on;
//! - the objectives, each jumping to its frame;
//! - completion and the minimum recorded energy;
//! - the recorded events;
//! - every field the report states it cannot support.
//!
//! A replay without a report, or with one that fails its checks, says so and
//! how to make one. It never shows mock data, and never a missing value as a
//! success.

use super::*;
use crate::replay::{assurance_frame_index, MissionAssurance, ReplayDocument};
use crate::theme::Status;

/// What Review has to show for the loaded replay.
#[derive(Debug, Clone, PartialEq)]
pub enum ReviewState {
    NoReplay,
    /// The replay carries no assurance report.
    NoReport,
    /// It carries one that failed its checks: why.
    Invalid(String),
    Report(ReviewReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReviewReport {
    pub status: String,
    pub outcome: Option<String>,
    pub findings: Vec<ReviewItem>,
    pub objectives: Vec<ReviewItem>,
    pub completion: Option<f64>,
    pub energy_reserve_min: Option<f64>,
    /// The first [`EVENTS_SHOWN`] recorded events: only what is drawn is
    /// built, however many the report holds.
    pub events: Vec<ReviewItem>,
    /// How many events the report holds in all.
    pub events_total: usize,
    pub unavailable_fields: Vec<String>,
}

/// One line of the report, and the frame it jumps to when it has one.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewItem {
    pub title: String,
    pub detail: String,
    pub frame: Option<usize>,
}

/// How to make a report, for a replay that has none.
pub const MAKE_A_REPORT: &str =
    "Make one with `argusnet client-package --replay <replay> --output <dir>`, \
     then open the package's replay.json (or a scene built from it).";

/// Review's view of `document`.
pub fn review_view(document: Option<&ReplayDocument>) -> ReviewState {
    let Some(document) = document else {
        return ReviewState::NoReplay;
    };
    match &document.mission_assurance {
        None => ReviewState::NoReport,
        Some(Err(why)) => ReviewState::Invalid(why.clone()),
        Some(Ok(report)) => ReviewState::Report(review_report(report, document.frames.len())),
    }
}

fn review_report(report: &MissionAssurance, frame_count: usize) -> ReviewReport {
    let findings = report
        .outcome
        .iter()
        .flat_map(|outcome| outcome.reasons.iter())
        .map(|reason| ReviewItem {
            title: reason.code.replace('_', " "),
            detail: format!("{} ({} references)", reason.detail, reason.evidence_count),
            frame: reason.first_frame(frame_count),
        })
        .collect();
    let objectives = report
        .objectives
        .iter()
        .map(|objective| ReviewItem {
            title: objective.objective_id.clone(),
            detail: format!(
                "{} · evidence sufficiency {}",
                objective.status,
                objective
                    .evidence_sufficiency
                    .as_deref()
                    .unwrap_or("unavailable")
            ),
            frame: assurance_frame_index(&objective.source_pointer, frame_count),
        })
        .collect();
    let events = report
        .events
        .iter()
        .take(EVENTS_SHOWN)
        .map(|event| ReviewItem {
            title: event.kind.replace('_', " "),
            detail: format!(
                "{} · {} · {}",
                event
                    .timestamp_s
                    .map_or("time unavailable".to_string(), |t| format!("{t:.1} s")),
                event.subject_id.as_deref().unwrap_or("no subject"),
                event.reason.as_deref().unwrap_or("reason unavailable"),
            ),
            frame: assurance_frame_index(&event.source_pointer, frame_count),
        })
        .collect();
    ReviewReport {
        status: report.status.clone(),
        outcome: report.outcome.as_ref().map(|o| o.outcome.replace('_', " ")),
        findings,
        objectives,
        completion: report.completion_rate,
        energy_reserve_min: report.energy_reserve_min,
        events,
        events_total: report.events.len(),
        unavailable_fields: report.unavailable_fields.clone(),
    }
}

/// How many recorded events Review lists before it says how many more.
const EVENTS_SHOWN: usize = 50;

/// The Review rail: the report, or why there is none.
pub(super) fn review_rail(ui: &mut egui::Ui, replay_state: &mut ReplayState) {
    let state = review_view(replay_state.document.as_ref());
    ui.strong("Review");
    let report = match state {
        ReviewState::NoReplay => {
            ui.small("No replay loaded.");
            return;
        }
        ReviewState::NoReport => {
            ui.label(Status::Unavailable.rich_text());
            ui.label("This replay carries no assurance report.");
            ui.small(MAKE_A_REPORT);
            return;
        }
        ReviewState::Invalid(why) => {
            ui.label(Status::Critical.rich_text());
            ui.label(format!("The assurance report failed its checks: {why}"));
            ui.small(MAKE_A_REPORT);
            return;
        }
        ReviewState::Report(report) => report,
    };
    let status = match report.status.as_str() {
        "completed" => Status::Nominal,
        "aborted" => Status::Critical,
        "incomplete" => Status::Degraded,
        _ => Status::Unavailable,
    };
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(status.color(), egui::RichText::new(&report.status).strong());
        if let Some(outcome) = &report.outcome {
            ui.small(format!("F1 outcome: {outcome}"));
        }
    });
    let percent = |value: Option<f64>| {
        value.map_or("unavailable".to_string(), |v| format!("{:.0}%", v * 100.0))
    };
    ui.small(format!(
        "Completion {} · lowest energy {}",
        percent(report.completion),
        percent(report.energy_reserve_min)
    ));
    let mut jump = None;
    let mut section = |ui: &mut egui::Ui, title: &str, items: &[ReviewItem], total: usize| {
        egui::CollapsingHeader::new(format!("{title} ({total})"))
            .default_open(title == "Findings")
            .show(ui, |ui| {
                if items.is_empty() {
                    ui.small("None recorded.");
                }
                for item in items {
                    let response = ui
                        .add_enabled(item.frame.is_some(), egui::Button::new(&item.title).small())
                        .on_hover_text(&item.detail);
                    if item.frame.is_none() {
                        response.on_disabled_hover_text("No frame of this replay to jump to");
                    } else if response.clicked() {
                        jump = item.frame;
                    }
                    ui.small(&item.detail);
                }
                if total > items.len() {
                    ui.small(format!(
                        "… and {} more in assurance.json",
                        total - items.len()
                    ));
                }
            });
    };
    section(ui, "Findings", &report.findings, report.findings.len());
    section(
        ui,
        "Objectives",
        &report.objectives,
        report.objectives.len(),
    );
    section(ui, "Events", &report.events, report.events_total);
    egui::CollapsingHeader::new(format!("Unavailable ({})", report.unavailable_fields.len())).show(
        ui,
        |ui| {
            ui.small("What this report cannot claim from the replay:");
            for field in &report.unavailable_fields {
                ui.small(format!("· {field}"));
            }
        },
    );
    if let Some(frame) = jump {
        replay_state.playing = false;
        replay_state.step_to(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(summary: serde_json::Value) -> ReplayDocument {
        let mut value = serde_json::json!({
            "summary": summary,
            "frames": [{"timestamp_s": 0.0}, {"timestamp_s": 1.0}],
        });
        // Bound to these frames, as a package binds its replay; loaded the
        // way the viewer loads a scene's replay, which checks the binding.
        let binding = crate::replay::source_sha256(&value);
        if let Some(report) = value["summary"].get_mut("mission_assurance") {
            if report.get("source_binding").is_some() {
                report["source_binding"]["sha256"] = serde_json::json!(binding);
            }
        }
        ReplayDocument::try_from(value).unwrap()
    }

    /// The shared Python fixture: a client package's own report.
    fn report() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../tests/fixtures/mission_assurance_fixture.json"
        ))
        .unwrap()
    }

    #[test]
    fn a_packaged_replay_shows_its_report() {
        let doc = document(serde_json::json!({ "mission_assurance": report() }));
        let ReviewState::Report(view) = review_view(Some(&doc)) else {
            panic!("a report");
        };
        assert_eq!(view.status, "completed");
        assert_eq!(view.outcome.as_deref(), Some("completed"));
        // The F1 reasons are the findings, each jumping to its frame.
        let met = view
            .findings
            .iter()
            .find(|f| f.title == "objectives met")
            .unwrap();
        assert_eq!(met.frame, Some(0));
        assert_eq!(view.objectives[0].title, "asset");
        assert_eq!(view.objectives[0].frame, Some(0));
        assert!(view.objectives[0]
            .detail
            .contains("evidence sufficiency unavailable"));
        assert!(view
            .unavailable_fields
            .iter()
            .any(|f| f == "landing_outcome"));
        assert_eq!(view.completion, Some(1.0));
    }

    #[test]
    fn no_report_and_a_bad_report_say_so() {
        assert_eq!(review_view(None), ReviewState::NoReplay);
        let plain = document(serde_json::json!({}));
        assert_eq!(review_view(Some(&plain)), ReviewState::NoReport);
        let mut bad = report();
        bad["schema_version"] = serde_json::json!("argusnet.mission-assurance/v9");
        let doc = document(serde_json::json!({ "mission_assurance": bad }));
        assert!(
            matches!(review_view(Some(&doc)), ReviewState::Invalid(why) if why.contains("version"))
        );
    }

    /// A report whose replay changed after it was made is not shown as that
    /// replay's: Review says it failed its checks.
    #[test]
    fn a_report_for_other_frames_is_invalid() {
        let mut value = serde_json::json!({
            "summary": {"mission_assurance": report()},
            "frames": [{"timestamp_s": 0.0}, {"timestamp_s": 1.0}],
        });
        value["summary"]["mission_assurance"]["source_binding"]["sha256"] =
            serde_json::json!(crate::replay::source_sha256(&value));
        value["frames"][1]["timestamp_s"] = serde_json::json!(2.0);
        let doc = ReplayDocument::try_from(value).unwrap();
        assert!(matches!(
            review_view(Some(&doc)),
            ReviewState::Invalid(why) if why.contains("does not describe this replay")
        ));
    }

    /// A report with thousands of events builds only what Review draws.
    #[test]
    fn only_the_shown_events_are_built() {
        let mut many = report();
        let event = many["events"][0].clone();
        many["events"] = serde_json::json!(vec![event; 5000]);
        let doc = document(serde_json::json!({ "mission_assurance": many }));
        let ReviewState::Report(view) = review_view(Some(&doc)) else {
            panic!("a report");
        };
        assert_eq!(view.events.len(), EVENTS_SHOWN);
        assert_eq!(view.events_total, 5000);
    }

    /// A pointer past the replay's frames is not a jump.
    #[test]
    fn links_only_reach_frames_the_replay_has() {
        let mut short = report();
        short["objectives"][0]["source_pointer"] =
            serde_json::json!("/frames/9/scan_mission_state/poi_statuses/0");
        let doc = document(serde_json::json!({ "mission_assurance": short }));
        let ReviewState::Report(view) = review_view(Some(&doc)) else {
            panic!("a report");
        };
        assert_eq!(view.objectives[0].frame, None);
    }
}
