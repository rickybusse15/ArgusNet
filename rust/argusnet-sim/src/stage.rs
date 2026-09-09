//! The eleven tick stages, and the trace that records them.
//!
//! [`ARCHITECTURE_AND_PROTOCOLS.md`](../../../docs/migration/ARCHITECTURE_AND_PROTOCOLS.md)
//! §4 fixes the order and says plainly why: *"The first Rust port MUST preserve
//! the Python causal order, including decisions that take effect on the next
//! tick. It MUST NOT silently 'improve' that order."*
//!
//! So the order lives in one place — [`Stage::PIPELINE`] — rather than being
//! implied by the sequence of calls in some function body, where reordering two
//! lines during a refactor would be invisible in review.
//!
//! §4 also asks for a machine-readable stage trace *against the Python
//! reference, before extracting the orchestrator*. [`StageTrace`] is that
//! harness. It records what ran, in what order, at which tick — which is what
//! makes "did the port preserve the order?" a question with an answer, rather
//! than a matter of reading two implementations side by side.

use std::fmt;

use serde::{Deserialize, Serialize};

/// One stage of the tick transaction, in causal order.
///
/// The discriminants are the order numbers from §4's table and are part of the
/// trace format, so they are written out rather than left implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// 0 — apply validated commands scheduled for this boundary.
    ApplyCommands,
    /// 1 — sample nominal controllers and account battery **using the prior
    /// positions**. The ordering relative to overrides is a compatibility
    /// constraint, not an accident.
    SampleControllersAndBattery,
    /// 2 — frontier, then POI, then egress overrides, in that precedence.
    ApplyMissionOverrides,
    /// 3 — cooperative deconfliction, with stable platform ordering.
    Deconflict,
    /// 4 — opt-in hard safety monitor and clamp. Runs *after* deconfliction.
    SafetyMonitor,
    /// 5 — truth/synthetic IMU features, VIO update, observation pose select.
    UpdateVioAndPose,
    /// 6 — generate the observation batch, preserving rejection stages.
    GenerateObservations,
    /// 7 — fusion update, in-process rather than over gRPC.
    UpdateFusion,
    /// 8 — coverage/belief and localization.
    UpdateMappingAndLocalization,
    /// 9 — mission/frontier/POI/egress transitions and events.
    EvaluateMissionTransitions,
    /// 10 — freeze snapshot, metrics, journal position, checkpoint eligibility.
    CommitSnapshot,
}

impl Stage {
    /// The causal order. Iterating this is the only sanctioned way to run a
    /// tick, so the order cannot drift by editing a call sequence.
    pub const PIPELINE: [Stage; 11] = [
        Stage::ApplyCommands,
        Stage::SampleControllersAndBattery,
        Stage::ApplyMissionOverrides,
        Stage::Deconflict,
        Stage::SafetyMonitor,
        Stage::UpdateVioAndPose,
        Stage::GenerateObservations,
        Stage::UpdateFusion,
        Stage::UpdateMappingAndLocalization,
        Stage::EvaluateMissionTransitions,
        Stage::CommitSnapshot,
    ];

    /// The stage's number in §4's table.
    pub fn order(self) -> u8 {
        Self::PIPELINE
            .iter()
            .position(|stage| *stage == self)
            .expect("PIPELINE contains every variant") as u8
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Stage::ApplyCommands => "apply_commands",
            Stage::SampleControllersAndBattery => "sample_controllers_and_battery",
            Stage::ApplyMissionOverrides => "apply_mission_overrides",
            Stage::Deconflict => "deconflict",
            Stage::SafetyMonitor => "safety_monitor",
            Stage::UpdateVioAndPose => "update_vio_and_pose",
            Stage::GenerateObservations => "generate_observations",
            Stage::UpdateFusion => "update_fusion",
            Stage::UpdateMappingAndLocalization => "update_mapping_and_localization",
            Stage::EvaluateMissionTransitions => "evaluate_mission_transitions",
            Stage::CommitSnapshot => "commit_snapshot",
        };
        formatter.write_str(name)
    }
}

/// Why a stage failed.
///
/// A stage failure is not recoverable within the tick: §4 requires that no
/// partial snapshot be published and the run be marked faulted. The variants
/// exist so a diagnostic can say which kind of failure it was, not so a caller
/// can choose to continue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// The stage found state it should not have been able to reach.
    Invariant(String),
    /// A dependency this stage needs is not wired up yet. Expected while
    /// stages are still being ported; a run that hits one is not a physics
    /// failure and should not be reported as one.
    NotImplemented(Stage),
}

impl fmt::Display for StageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageError::Invariant(detail) => write!(formatter, "invariant violated: {detail}"),
            StageError::NotImplemented(stage) => {
                write!(formatter, "stage {stage} is not implemented yet")
            }
        }
    }
}

impl std::error::Error for StageError {}

/// What a stage did, as recorded in the trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum StageOutcome {
    /// Ran and mutated state.
    Ran,
    /// Ran and had nothing to do. Distinguished from `Ran` because "the
    /// deconfliction stage did nothing" and "the deconfliction stage never ran"
    /// are different bugs and a trace that conflated them would hide one.
    NoWork,
    /// Deliberately disabled for this run — the opt-in safety monitor, say.
    Skipped { reason: String },
    /// Failed. The tick is abandoned; nothing after this stage ran.
    Failed { detail: String },
}

/// One line of the trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRecord {
    pub tick: u64,
    pub order: u8,
    pub stage: Stage,
    #[serde(flatten)]
    pub outcome: StageOutcome,
    /// Free-form detail a stage chose to record: counts, decisions that take
    /// effect on a later tick, the reason a branch was taken. §4 specifically
    /// asks that stage 9 record *which future tick* a controller change lands
    /// on, and this is where that goes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// An ordered, serializable record of what the tick pipeline did.
///
/// Bounded on purpose: a long interactive run would otherwise grow this without
/// limit, and the trace is a diagnostic, not a journal. Durable mission and
/// safety events use the separate ordered journal §5 describes — losing the
/// oldest trace lines is acceptable in a way that losing an event is not.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTrace {
    records: Vec<StageRecord>,
    #[serde(default)]
    capacity: Option<usize>,
    /// How many records were dropped to stay within `capacity`. Reported
    /// rather than hidden, so a reader can tell a truncated trace from a
    /// complete one.
    #[serde(default)]
    dropped: u64,
}

impl StageTrace {
    /// A trace that keeps at most `capacity` records, dropping oldest first.
    pub fn bounded(capacity: usize) -> Self {
        Self {
            records: Vec::new(),
            capacity: Some(capacity),
            dropped: 0,
        }
    }

    /// A trace that keeps everything. For tests and short batch runs.
    pub fn unbounded() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: StageRecord) {
        self.records.push(record);
        if let Some(capacity) = self.capacity {
            while self.records.len() > capacity {
                self.records.remove(0);
                self.dropped += 1;
            }
        }
    }

    pub fn records(&self) -> &[StageRecord] {
        &self.records
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn clear(&mut self) {
        self.records.clear();
        self.dropped = 0;
    }

    /// Every record for one tick, in the order the stages ran.
    pub fn tick(&self, tick: u64) -> Vec<&StageRecord> {
        self.records
            .iter()
            .filter(|record| record.tick == tick)
            .collect()
    }

    /// The stages that ran on one tick, in order.
    ///
    /// This is the comparison surface for "did the port preserve the causal
    /// order?" — a `Vec<Stage>` diffs readably against the Python reference
    /// trace in a way that two implementations do not.
    pub fn stage_order(&self, tick: u64) -> Vec<Stage> {
        self.tick(tick)
            .into_iter()
            .map(|record| record.stage)
            .collect()
    }

    /// Serialize for comparison against a reference trace.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pipeline_is_the_order_from_the_architecture_table() {
        // Spelled out rather than derived from PIPELINE, so reordering
        // PIPELINE fails here instead of silently redefining "correct".
        let expected = [
            "apply_commands",
            "sample_controllers_and_battery",
            "apply_mission_overrides",
            "deconflict",
            "safety_monitor",
            "update_vio_and_pose",
            "generate_observations",
            "update_fusion",
            "update_mapping_and_localization",
            "evaluate_mission_transitions",
            "commit_snapshot",
        ];
        let actual: Vec<String> = Stage::PIPELINE.iter().map(|s| s.to_string()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn safety_runs_after_deconfliction() {
        // Called out separately because §4 states it as a constraint in prose,
        // and it is the one pair a reader is most likely to think is arbitrary.
        assert!(Stage::SafetyMonitor.order() > Stage::Deconflict.order());
    }

    #[test]
    fn controllers_are_sampled_before_overrides_are_applied() {
        // Stage 1 uses the *prior* positions; stage 2 overrides them. Swapping
        // these changes battery accounting, which §4 flags as a behaviour
        // change that must be separate from the port.
        assert!(Stage::SampleControllersAndBattery.order() < Stage::ApplyMissionOverrides.order());
    }

    #[test]
    fn order_numbers_are_contiguous_from_zero() {
        for (index, stage) in Stage::PIPELINE.iter().enumerate() {
            assert_eq!(stage.order() as usize, index);
        }
    }

    #[test]
    fn a_bounded_trace_drops_oldest_and_says_how_many() {
        let mut trace = StageTrace::bounded(3);
        for tick in 0..6u64 {
            trace.record(StageRecord {
                tick,
                order: 0,
                stage: Stage::ApplyCommands,
                outcome: StageOutcome::Ran,
                notes: Vec::new(),
            });
        }
        assert_eq!(trace.records().len(), 3);
        assert_eq!(trace.dropped(), 3);
        assert_eq!(
            trace.records().iter().map(|r| r.tick).collect::<Vec<_>>(),
            vec![3, 4, 5],
            "the newest records are the ones kept"
        );
    }

    #[test]
    fn stage_order_reports_what_ran_on_one_tick() {
        let mut trace = StageTrace::unbounded();
        for tick in 0..2u64 {
            for stage in [
                Stage::ApplyCommands,
                Stage::Deconflict,
                Stage::CommitSnapshot,
            ] {
                trace.record(StageRecord {
                    tick,
                    order: stage.order(),
                    stage,
                    outcome: StageOutcome::Ran,
                    notes: Vec::new(),
                });
            }
        }
        assert_eq!(
            trace.stage_order(1),
            vec![
                Stage::ApplyCommands,
                Stage::Deconflict,
                Stage::CommitSnapshot
            ]
        );
        assert_eq!(trace.stage_order(7), Vec::<Stage>::new());
    }

    #[test]
    fn the_trace_round_trips_through_json() {
        let mut trace = StageTrace::unbounded();
        trace.record(StageRecord {
            tick: 4,
            order: Stage::EvaluateMissionTransitions.order(),
            stage: Stage::EvaluateMissionTransitions,
            outcome: StageOutcome::Skipped {
                reason: "no mission configured".to_string(),
            },
            notes: vec!["controller change effective at tick 5".to_string()],
        });
        let json = trace.to_json().expect("serializable");
        let parsed: Vec<StageRecord> = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(parsed, trace.records());
        // The note is the machine-readable half of §4's "record which future
        // tick controller changes take effect".
        assert!(json.contains("effective at tick 5"));
    }

    #[test]
    fn no_work_is_distinguishable_from_never_ran() {
        let mut trace = StageTrace::unbounded();
        trace.record(StageRecord {
            tick: 0,
            order: Stage::Deconflict.order(),
            stage: Stage::Deconflict,
            outcome: StageOutcome::NoWork,
            notes: Vec::new(),
        });
        assert_eq!(trace.stage_order(0), vec![Stage::Deconflict]);
        assert!(matches!(trace.tick(0)[0].outcome, StageOutcome::NoWork));
    }
}
