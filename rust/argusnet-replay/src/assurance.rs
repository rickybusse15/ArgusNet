//! Additive S1 evaluator-owned report carried under summary.mission_assurance.

use serde::Deserialize;

pub const ASSURANCE_VERSION: &str = "argusnet.mission-assurance/v1";

#[derive(Debug, Clone, Deserialize)]
pub struct MissionAssurance {
    pub schema_version: String,
    pub source_replay_sha256: String,
    pub status: String,
    /// The F1 outcome record (`argusnet.mission-outcome/1`), when the
    /// producer carries it: how the run ended and why.
    #[serde(default)]
    pub outcome: Option<AssuranceOutcome>,
    pub terminal_phase: Option<String>,
    pub completion_rate: Option<f64>,
    pub energy_reserve_min: Option<f64>,
    pub landing_outcome: Option<String>,
    pub localization_integrity: Option<String>,
    pub objectives: Vec<AssuranceObjective>,
    pub events: Vec<AssuranceEvent>,
    pub unavailable_fields: Vec<String>,
    pub localization_timeout_sources: Vec<String>,
}

/// How a run ended and every reason why (F1), each with how many frames or
/// events support it.
#[derive(Debug, Clone, Deserialize)]
pub struct AssuranceOutcome {
    pub outcome: String,
    #[serde(default)]
    pub reasons: Vec<OutcomeReason>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutcomeReason {
    pub code: String,
    pub detail: String,
    #[serde(default)]
    pub evidence_count: usize,
    /// What the reason rests on: frames (`{"kind": "frame", "tick"}`),
    /// safety events (with their `tick`) and other references.
    #[serde(default)]
    pub evidence: Vec<serde_json::Value>,
}

impl OutcomeReason {
    /// The first frame the reason's evidence points at, when one is within
    /// a replay of `frame_count` frames.
    pub fn first_frame(&self, frame_count: usize) -> Option<usize> {
        self.evidence
            .iter()
            .filter_map(|item| item.get("tick").and_then(serde_json::Value::as_u64))
            .map(|tick| tick as usize)
            .find(|tick| *tick < frame_count)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssuranceObjective {
    pub objective_id: String,
    pub status: String,
    pub source_pointer: String,
    pub evidence_sufficiency: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssuranceEvent {
    pub event_id: String,
    pub kind: String,
    pub timestamp_s: Option<f64>,
    pub subject_id: Option<String>,
    pub reason: Option<String>,
    pub source_pointer: String,
}

impl MissionAssurance {
    /// Refuse a report whose version or numeric result cannot be presented.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != ASSURANCE_VERSION {
            return Err("Unsupported mission assurance version");
        }
        if !["unavailable", "incomplete", "completed", "aborted"].contains(&self.status.as_str()) {
            return Err("Invalid mission assurance status");
        }
        if self.source_replay_sha256.len() != 64
            || !self
                .source_replay_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("Invalid source replay hash");
        }
        for value in [self.completion_rate, self.energy_reserve_min]
            .into_iter()
            .flatten()
        {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err("Invalid mission assurance fraction");
            }
        }
        Ok(())
    }
}

/// Resolve a recorded frame reference, never an external path or new command.
pub fn assurance_frame_index(pointer: &str, frame_count: usize) -> Option<usize> {
    let mut parts = pointer.strip_prefix("/frames/")?.split('/');
    let index: usize = parts.next()?.parse().ok()?;
    parts.next()?;
    (index < frame_count).then_some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> MissionAssurance {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/mission_assurance_fixture.json"
        ))
        .expect("Python client package fixture")
    }

    #[test]
    fn python_assurance_keeps_missing_results_and_reads_recorded_energy() {
        let report = fixture();
        report.validate().unwrap();
        assert_eq!(report.status, "completed");
        assert_eq!(report.completion_rate, Some(1.0));
        assert_eq!(report.landing_outcome, None);
        assert_eq!(report.localization_integrity, None);
        assert_eq!(report.energy_reserve_min, Some(0.4));
        assert_eq!(report.objectives[0].objective_id, "asset");
        assert_eq!(report.objectives[0].evidence_sufficiency, None);
        assert_eq!(report.events[0].reason.as_deref(), Some("clearance"));
        assert_eq!(report.events[0].source_pointer, "/frames/0/safety_events/0");
        assert_eq!(report.localization_timeout_sources.len(), 1);
        assert!(report
            .unavailable_fields
            .iter()
            .any(|field| field == "qualification"));
    }

    /// The report carries the F1 outcome with its reasons; it agrees with
    /// the report's own status.
    #[test]
    fn the_report_carries_the_f1_outcome() {
        let report = fixture();
        let outcome = report.outcome.as_ref().expect("an F1 outcome");
        assert_eq!(outcome.outcome, report.status);
        assert!(outcome.reasons.iter().any(|r| r.code == "objectives_met"));
        assert!(outcome.reasons.iter().all(|r| !r.detail.is_empty()));
        // Each reason points at a frame of the replay it describes.
        let met = outcome
            .reasons
            .iter()
            .find(|r| r.code == "objectives_met")
            .unwrap();
        assert_eq!(met.first_frame(1), Some(0));
        assert_eq!(met.first_frame(0), None);
    }

    #[test]
    fn incompatible_or_invalid_reports_cannot_look_like_success() {
        let mut report = fixture();
        report.schema_version = "future/v99".into();
        assert!(report.validate().is_err());
        report = fixture();
        report.completion_rate = Some(1.1);
        assert!(report.validate().is_err());
        report = fixture();
        report.energy_reserve_min = Some(f64::NAN);
        assert!(report.validate().is_err());
        report = fixture();
        report.source_replay_sha256 = "unverified".into();
        assert!(report.validate().is_err());
    }

    #[test]
    fn an_invalid_assurance_extension_does_not_break_replay_playback() {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/replay_contract_fixture.json"
        ))
        .unwrap();
        let mut report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/mission_assurance_fixture.json"
        ))
        .unwrap();
        report["schema_version"] = "unsupported/v99".into();
        value["summary"]["mission_assurance"] = report;
        let document: crate::ReplayDocument = serde_json::from_value(value).unwrap();
        assert_eq!(document.frames.len(), 3);
        assert!(document.mission_assurance.unwrap().is_err());
    }

    #[test]
    fn explanation_links_only_resolve_available_recorded_frames() {
        assert_eq!(
            assurance_frame_index("/frames/2/safety_events/0", 3),
            Some(2)
        );
        assert_eq!(assurance_frame_index("/frames/3/safety_events/0", 3), None);
        assert_eq!(assurance_frame_index("/frames/0", 3), None);
        assert_eq!(assurance_frame_index("/frames/-1/safety_events/0", 3), None);
        assert_eq!(
            assurance_frame_index("https://example.com/frames/0", 3),
            None
        );
    }
}
