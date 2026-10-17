//! Snapshots and checkpoints — two contracts that look alike and are not.
//!
//! [`ARCHITECTURE_AND_PROTOCOLS.md`](../../../docs/migration/ARCHITECTURE_AND_PROTOCOLS.md)
//! §6 is emphatic that a replay is not a checkpoint: *"A fused-state frame
//! alone omits IMM filters, controller integrators, random-stream state, safety
//! escalation and planner caches that affect future decisions."* The same
//! applies to a snapshot. Modelling them as one type is the mistake the section
//! exists to prevent, so they are separate types here even though P4.1's
//! versions are mostly headers.
//!
//! | type | answers |
//! |---|---|
//! | [`RuntimeSnapshot`] | "what should the operator see right now?" |
//! | [`SimulationCheckpoint`] | "what does resuming this run need?" |
//!
//! A snapshot is safe to publish and lossy by design — truth is filtered out of
//! it. A checkpoint is causally complete and is *not* safe to publish, because
//! it necessarily contains the truth state the snapshot filtered.

use std::sync::Arc;

use argusnet_types::{NodeState, TrackState};
use serde::{Deserialize, Serialize};

/// Identifies one committed revision of the run.
///
/// Every field is something a consumer needs in order to tell whether two
/// snapshots are comparable. `epoch` in particular: §6 says a seek that changes
/// a command *creates a new epoch/branch*, and §5 says never interpolate across
/// epochs. A renderer that has only a tick number cannot honour that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotHeader {
    pub run_id: String,
    /// Bumped whenever history branches. Two snapshots from different epochs
    /// describe different worlds, however close their ticks.
    pub epoch: u64,
    pub tick: u64,
    /// Monotonic across the run, including across epochs, so a consumer can
    /// order snapshots it received out of order.
    pub revision: u64,
    /// Derived seconds — see [`crate::clock`]. Present so interchange consumers
    /// need not know the timestep.
    pub timestamp_s: f64,
    pub world_revision: u64,
    pub belief_revision: u64,
    /// Where the durable event journal stood when this was frozen. §5 keeps
    /// events off the lossy presentation path; this is the cursor that lets a
    /// consumer pick them up from the journal instead.
    pub event_cursor: u64,
    /// Whether truth state was withheld. §3 makes truth a capability, and a
    /// snapshot that had truth filtered must say so — a consumer cannot infer
    /// "no truth available" from "no truth present" without it.
    pub truth_filtered: bool,
}

/// The immutable in-process presentation contract.
///
/// Handed out behind an `Arc`: §5 asks for immutable snapshots shared with the
/// renderer, and cloning platform and track lists per consumer per frame is the
/// cost that motivates it.
///
/// Deliberately **not** serializable. §6 calls this an *in-process* contract,
/// and the persisted artifacts are the checkpoint and the replay document,
/// which have their own types and their own compatibility rules. Deriving
/// `Serialize` here would invite a third persistence format that nothing
/// versions.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeSnapshot {
    pub header: SnapshotHeader,
    pub platforms: Vec<NodeState>,
    pub tracks: Vec<TrackState>,
}

impl RuntimeSnapshot {
    /// An empty snapshot at a given point in the run.
    pub fn empty(header: SnapshotHeader) -> Arc<Self> {
        Arc::new(Self {
            header,
            platforms: Vec::new(),
            tracks: Vec::new(),
        })
    }
}

/// Versions that make a checkpoint restorable, or refuse it.
///
/// §6: *"Restore contract tied to runtime algorithm/config/world versions."*
/// Restoring RNG state into a runtime whose stage order or sampler changed
/// would resume a run that never existed, and it would do so silently, which is
/// the worst available outcome — hence a header that is checked before any
/// state is touched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Layout of this struct. Bumped when fields move.
    pub format_version: u32,
    /// Bumped when stage order or any stage's arithmetic changes. Two
    /// checkpoints with the same format and different algorithm versions are
    /// both well-formed and mutually incompatible.
    pub runtime_algorithm_version: u32,
    /// Bumped when seed derivation or the sampler changes. Separate from the
    /// algorithm version because ADR-006 lets the stream be re-baselined
    /// without the physics moving.
    pub seed_version: u32,
    pub run_id: String,
    pub epoch: u64,
    pub tick: u64,
    pub config_hash: u64,
    /// Content hash of the resolved scenario the run was assembled from:
    /// world, fleet, targets, mission and tracker (format 4). A checkpoint
    /// restores only into a run assembled from the same scenario. `0` for a
    /// bare run, which has no scenario.
    pub scenario_hash: u64,
    /// The world's revision counter, not a content hash despite the name;
    /// [`Self::scenario_hash`] is what binds a checkpoint to its world.
    pub world_content_hash: u64,
    pub belief_revision: u64,
    pub journal_cursor: u64,
}

/// The current versions this build writes and accepts.
impl CheckpointHeader {
    /// Format 3 added per-stage handler state (`stage_states`); format 4 adds
    /// `scenario_hash`. A format-3 checkpoint is refused with a versioned
    /// error rather than restored without a scenario check.
    pub const FORMAT_VERSION: u32 = 4;
    /// Format 4 plus `stage_lag` and `lane_rates` (B2.4): written only by a
    /// run with a lane at its own rate, so a reader that predates the fields
    /// refuses it instead of resuming without them. Every other checkpoint
    /// stays format 4.
    pub const FORMAT_VERSION_STAGE_LAG: u32 = 5;
    /// Version 2 changes command-boundary recovery and single-step semantics.
    /// Old checkpoints must not silently resume under a different control contract.
    pub const RUNTIME_ALGORITHM_VERSION: u32 = 2;
    /// Version 2 with POIs allocated under a planning policy other than the
    /// reference (P1): written only by such a run, so a reader that predates
    /// the policy refuses its checkpoints instead of resuming them under the
    /// reference planner. The mission driver's own state then checks the
    /// policy itself. Every other checkpoint stays version 2. Bumped each
    /// time `efficient_v1` changes what it decides (4: teams on active
    /// POIs; 5: safe handoff; 6: return home at the reserve; 7: routes
    /// around known obstacles; 8: routes whose ends sit inside the
    /// clearance; 9: deconfliction kept out of structures; 10: the reserve
    /// home priced along the route home), so a checkpoint never resumes
    /// under a different allocator.
    pub const RUNTIME_ALGORITHM_VERSION_POLICY: u32 = 10;
    /// Still 1: the ChaCha8 stream and the Box–Muller transform in
    /// [`crate::deterministic`]. Bump alongside any re-baseline.
    pub const SEED_VERSION: u32 = 1;
}

/// Why a checkpoint cannot be restored.
///
/// Each variant names one specific mismatch, because "incompatible checkpoint"
/// tells an operator nothing about what to change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreError {
    InvalidState {
        detail: String,
    },
    FormatVersion {
        found: u32,
        expected: u32,
    },
    AlgorithmVersion {
        found: u32,
        expected: u32,
    },
    SeedVersion {
        found: u32,
        expected: u32,
    },
    ConfigHash {
        found: u64,
        expected: u64,
    },
    /// The checkpoint was taken in a run assembled from a different scenario.
    ScenarioHash {
        found: u64,
        expected: u64,
    },
    RunId {
        found: String,
        expected: String,
    },
    /// The checkpoint carries state for a stage this run has no handler for.
    UnknownStageState {
        stage: String,
    },
    /// A handler refused the state recorded for it.
    StageState {
        stage: String,
        detail: String,
    },
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestoreError::InvalidState { detail } => write!(formatter, "invalid checkpoint state: {detail}"),
            RestoreError::FormatVersion { found, expected } => write!(
                formatter,
                "checkpoint format version {found} cannot be read by this build (expects {expected})"
            ),
            RestoreError::AlgorithmVersion { found, expected } => write!(
                formatter,
                "checkpoint was written by runtime algorithm version {found}; this build is \
                 {expected}, so resuming would continue a run with different arithmetic"
            ),
            RestoreError::SeedVersion { found, expected } => write!(
                formatter,
                "checkpoint carries seed version {found}; this build is {expected}, so the \
                 random stream would not continue where it left off"
            ),
            RestoreError::ConfigHash { found, expected } => write!(
                formatter,
                "checkpoint was taken under config hash {found:#x}; this run is {expected:#x}"
            ),
            RestoreError::ScenarioHash { found, expected } => write!(
                formatter,
                "checkpoint was taken in a run of scenario {found:#x}; this run was assembled \
                 from scenario {expected:#x}"
            ),
            RestoreError::UnknownStageState { stage } => write!(
                formatter,
                "checkpoint carries state for stage {stage}, which has no handler in this run"
            ),
            RestoreError::StageState { stage, detail } => {
                write!(formatter, "stage {stage} refused its checkpointed state: {detail}")
            }
            RestoreError::RunId { found, expected } => {
                write!(formatter, "checkpoint belongs to run {found:?}, not {expected:?}")
            }
        }
    }
}

impl std::error::Error for RestoreError {}

/// Everything needed to resume a run exactly where it stopped.
///
/// Format 2 carries the clock, control state, RNG contract version, pending
/// commands and bounded retry history. It grows a field per subsystem as stages land — and the rule
/// from §6 is that *anything affecting a future decision* belongs here, while
/// render entities, wall-clock timers and rebuildable caches do not. A cache
/// may only be excluded once its rebuild is shown to be behaviourally neutral.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationCheckpoint {
    pub header: CheckpointHeader,
    /// Tick to resume from. Redundant with `header.tick` on purpose: the header
    /// is the compatibility contract and this is causal state, and conflating
    /// them is how a header field ends up silently load-bearing.
    pub tick: u64,
    /// Base seed. Per-stream generators are re-derived from it and the tick, so
    /// no generator's internal position needs storing — which is a direct
    /// benefit of ADR-006's per-key scheme and worth stating, because a shared
    /// generator would have forced a much larger checkpoint.
    pub seed: u32,
    /// Commands accepted but not yet applied. §6: *"Persist all causal state,
    /// including pending commands."*
    pub pending_commands: Vec<crate::command::CommandEnvelope>,
    pub command_history: crate::command::CommandHistory,
    pub revision: u64,
    pub paused: bool,
    pub speed: f64,
    /// Each stateful stage handler's causal state, keyed by stage name.
    ///
    /// Format 3. A handler whose state is absent here is reset on restore
    /// rather than left holding whatever the restoring object had.
    pub stage_states: std::collections::BTreeMap<String, serde_json::Value>,
    /// Boundaries each stage has been not due for since it last ran, keyed
    /// by stage name (B2.4: a lane at its own rate). The stage's next run is
    /// given the time those boundaries spanned. Empty, and so absent, for a
    /// run where every stage runs every boundary.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub stage_lag: std::collections::BTreeMap<String, u64>,
    /// Each stage whose lane runs at its own rate, and that rate (`NUM/DEN`
    /// Hz), keyed by stage name (B2.4). A run restoring this checkpoint must
    /// have declared exactly these rates. Empty, and so absent, when every
    /// lane runs every boundary.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub lane_rates: std::collections::BTreeMap<String, String>,
}

impl SimulationCheckpoint {
    /// Check every compatibility field before any state is touched.
    pub fn verify(
        &self,
        run_id: &str,
        config_hash: u64,
        scenario_hash: u64,
    ) -> Result<(), RestoreError> {
        let expected = if self.stage_lag.is_empty() && self.lane_rates.is_empty() {
            CheckpointHeader::FORMAT_VERSION
        } else {
            CheckpointHeader::FORMAT_VERSION_STAGE_LAG
        };
        let readable = self.header.format_version == expected
            || (self.header.format_version == CheckpointHeader::FORMAT_VERSION_STAGE_LAG
                && self.stage_lag.is_empty()
                && self.lane_rates.is_empty());
        if !readable {
            return Err(RestoreError::FormatVersion {
                found: self.header.format_version,
                expected,
            });
        }
        if self.header.runtime_algorithm_version != CheckpointHeader::RUNTIME_ALGORITHM_VERSION
            && self.header.runtime_algorithm_version
                != CheckpointHeader::RUNTIME_ALGORITHM_VERSION_POLICY
        {
            return Err(RestoreError::AlgorithmVersion {
                found: self.header.runtime_algorithm_version,
                expected: CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
            });
        }
        if self.header.seed_version != CheckpointHeader::SEED_VERSION {
            return Err(RestoreError::SeedVersion {
                found: self.header.seed_version,
                expected: CheckpointHeader::SEED_VERSION,
            });
        }
        if self.header.run_id != run_id {
            return Err(RestoreError::RunId {
                found: self.header.run_id.clone(),
                expected: run_id.to_string(),
            });
        }
        if self.header.config_hash != config_hash {
            return Err(RestoreError::ConfigHash {
                found: self.header.config_hash,
                expected: config_hash,
            });
        }
        if self.header.scenario_hash != scenario_hash {
            return Err(RestoreError::ScenarioHash {
                found: self.header.scenario_hash,
                expected: scenario_hash,
            });
        }
        let invalid = |detail: &str| RestoreError::InvalidState {
            detail: detail.to_string(),
        };
        if self.tick != self.header.tick {
            return Err(invalid("tick disagrees with header"));
        }
        if self.tick == u64::MAX {
            return Err(invalid("tick cannot advance without overflow"));
        }
        if self.pending_commands.len() > crate::command::CommandQueue::DEFAULT_CAPACITY {
            return Err(invalid("pending command queue exceeds capacity"));
        }
        if !self.speed.is_finite() || self.speed <= 0.0 {
            return Err(invalid("invalid playback speed"));
        }
        if self.revision == u64::MAX {
            return Err(invalid("revision cannot advance without overflow"));
        }
        if self.command_history.receipts.len()
            > crate::command::CommandQueue::DEFAULT_DEDUP_CAPACITY
        {
            return Err(invalid("command history exceeds capacity"));
        }
        let mut receipt_ids = std::collections::BTreeSet::new();
        for receipt in &self.command_history.receipts {
            if !receipt_ids.insert(receipt.ack.command_id) || receipt.ack.duplicate {
                return Err(invalid("duplicate receipt or stored retry flag"));
            }
            if receipt.ack.epoch > self.header.epoch || receipt.ack.revision > self.revision {
                return Err(invalid("receipt refers to a future epoch or revision"));
            }
            match receipt.request.as_ref().map(|request| &request.payload) {
                Some(crate::command::CommandKind::SetSpeed { multiplier }) => {
                    if multiplier.to_bits() != 0.0f64.to_bits() || receipt.speed_bits.is_none() {
                        return Err(invalid("invalid encoded request speed"));
                    }
                }
                _ if receipt.speed_bits.is_some() => {
                    return Err(invalid("unexpected request speed bits"))
                }
                _ => {}
            }
            if let Some(request) = &receipt.request {
                if request.command_id != receipt.ack.command_id
                    || request.run_id != self.header.run_id
                {
                    return Err(invalid("receipt request identity mismatch"));
                }
            }
            let pending = self
                .pending_commands
                .iter()
                .find(|command| command.command_id == receipt.ack.command_id);
            if matches!(receipt.ack.status, crate::command::AckStatus::Accepted) {
                let Some(pending) = pending else {
                    return Err(invalid("accepted receipt has no pending command"));
                };
                if receipt
                    .request()
                    .is_none_or(|request| !request.same_request(pending))
                {
                    return Err(invalid("pending command disagrees with receipt"));
                }
            } else if pending.is_some() {
                return Err(invalid("terminal receipt still has a pending command"));
            }
        }
        let mut ids = std::collections::BTreeSet::new();
        for command in &self.pending_commands {
            if !receipt_ids.contains(&command.command_id)
                && !self
                    .command_history
                    .retired_through
                    .is_some_and(|id| command.command_id <= id)
            {
                return Err(invalid(
                    "pending command has no receipt or retirement history",
                ));
            }
            if !ids.insert(command.command_id) {
                return Err(invalid("duplicate pending command ID"));
            }
            if command.run_id != self.header.run_id
                || command.expected_epoch != self.header.epoch
                || command.protocol_version != crate::command::PROTOCOL_VERSION
            {
                return Err(invalid("pending command identity or protocol mismatch"));
            }
            if command
                .requested_effective_tick
                .is_some_and(|tick| tick < self.tick)
            {
                return Err(invalid("pending command is scheduled before checkpoint"));
            }
            match &command.payload {
                crate::command::CommandKind::SetSpeed { multiplier }
                    if !multiplier.is_finite() || *multiplier <= 0.0 =>
                {
                    return Err(invalid("pending command has invalid speed"));
                }
                crate::command::CommandKind::Seek { .. }
                | crate::command::CommandKind::Unsupported { .. } => {
                    return Err(invalid("pending command is unsupported"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> CheckpointHeader {
        CheckpointHeader {
            format_version: CheckpointHeader::FORMAT_VERSION,
            runtime_algorithm_version: CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
            seed_version: CheckpointHeader::SEED_VERSION,
            run_id: "scan".to_string(),
            epoch: 0,
            tick: 12,
            config_hash: 0xabc,
            scenario_hash: 0x5ce,
            world_content_hash: 0xdef,
            belief_revision: 3,
            journal_cursor: 9,
        }
    }

    fn checkpoint() -> SimulationCheckpoint {
        SimulationCheckpoint {
            header: header(),
            tick: 12,
            seed: 7,
            pending_commands: Vec::new(),
            command_history: crate::command::CommandHistory::default(),
            revision: 12,
            paused: false,
            speed: 1.0,
            stage_states: Default::default(),
            stage_lag: Default::default(),
            lane_rates: Default::default(),
        }
    }

    #[test]
    fn a_matching_checkpoint_verifies() {
        assert_eq!(checkpoint().verify("scan", 0xabc, 0x5ce), Ok(()));
    }

    #[test]
    fn every_compatibility_field_is_actually_checked() {
        // One mutation per field: a `verify` that forgot a field would pass
        // for that mutation, and a single happy-path test would never notice.
        let cases: Vec<(SimulationCheckpoint, &str, u64)> = vec![
            (
                SimulationCheckpoint {
                    header: CheckpointHeader {
                        format_version: 99,
                        ..header()
                    },
                    ..checkpoint()
                },
                "scan",
                0xabc,
            ),
            (
                SimulationCheckpoint {
                    header: CheckpointHeader {
                        runtime_algorithm_version: 99,
                        ..header()
                    },
                    ..checkpoint()
                },
                "scan",
                0xabc,
            ),
            (
                SimulationCheckpoint {
                    header: CheckpointHeader {
                        seed_version: 99,
                        ..header()
                    },
                    ..checkpoint()
                },
                "scan",
                0xabc,
            ),
            (checkpoint(), "other-run", 0xabc),
            (checkpoint(), "scan", 0x999),
            (
                SimulationCheckpoint {
                    header: CheckpointHeader {
                        scenario_hash: 0x999,
                        ..header()
                    },
                    ..checkpoint()
                },
                "scan",
                0xabc,
            ),
        ];
        for (candidate, run_id, config_hash) in cases {
            assert!(
                candidate.verify(run_id, config_hash, 0x5ce).is_err(),
                "verify accepted an incompatible checkpoint: {:?}",
                candidate.header
            );
        }
    }

    #[test]
    fn restore_errors_name_what_to_change() {
        let mismatched = SimulationCheckpoint {
            header: CheckpointHeader {
                seed_version: 42,
                ..header()
            },
            ..checkpoint()
        };
        let message = mismatched
            .verify("scan", 0xabc, 0x5ce)
            .expect_err("mismatch")
            .to_string();
        assert!(message.contains("42"), "{message}");
        assert!(message.contains("random stream"), "{message}");
    }

    #[test]
    fn a_snapshot_says_whether_truth_was_filtered() {
        // §3 makes truth a capability. "No truth present" and "truth withheld"
        // must be distinguishable, or a consumer will read an empty list as
        // ground truth about the world.
        let snapshot = RuntimeSnapshot::empty(SnapshotHeader {
            run_id: "scan".to_string(),
            epoch: 0,
            tick: 0,
            revision: 0,
            timestamp_s: 0.0,
            world_revision: 0,
            belief_revision: 0,
            event_cursor: 0,
            truth_filtered: true,
        });
        assert!(snapshot.header.truth_filtered);
    }

    #[test]
    fn a_checkpoint_round_trips_through_json() {
        let json = serde_json::to_string(&checkpoint()).expect("serializable");
        let parsed: SimulationCheckpoint = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(parsed, checkpoint());
    }
}
