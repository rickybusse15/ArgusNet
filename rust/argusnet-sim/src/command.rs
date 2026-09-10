//! Operator commands: envelopes, acknowledgements and the bounded queue.
//!
//! [`ARCHITECTURE_AND_PROTOCOLS.md`](../../../docs/migration/ARCHITECTURE_AND_PROTOCOLS.md)
//! §5 sets the rules this module implements, and one of them is unusually
//! strong: *"Queue-full returns Busy to the UI; commands MUST NOT be silently
//! dropped."* A dropped command is worse than a rejected one, because the UI
//! shows a control that appears to have worked.
//!
//! The retry story is the other half. A UI that does not hear back resends, so
//! the same `command_id` arrives twice and must not be applied twice —
//! §5 requires the *original acknowledgement* to be returned within a bounded
//! dedup window, and requires an id that has aged out of that window to force
//! resynchronisation rather than a blind replay.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Wire/IPC version of the command protocol itself.
///
/// Separate from every other version in the system: a UI can be older than the
/// runtime without the physics having changed.
pub const PROTOCOL_VERSION: u32 = 1;

/// What the operator asked for.
///
/// P4.1 carries the subset §5 names as initial commands and that the shell can
/// actually honour. Anything else is [`CommandKind::Unsupported`], which is an
/// explicit outcome rather than a silent no-op — §5 lists "unsupported command"
/// among the states that must be distinguishable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CommandKind {
    Pause,
    Resume,
    SingleStep,
    /// Playback pacing only. §5: *"Playback speed scales desired wall-clock
    /// pacing, never physics dt."* Nothing here touches the timestep.
    SetSpeed {
        multiplier: f64,
    },
    Reset,
    /// Named so the reason for refusing it is honest: rewind needs checkpoints
    /// this shell does not retain yet.
    Seek {
        tick: u64,
    },
    Unsupported {
        name: String,
    },
}

/// A command plus everything needed to decide whether it still applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandEnvelope {
    pub protocol_version: u32,
    pub command_id: u64,
    pub run_id: String,
    /// The epoch the sender believed was current. A command composed before a
    /// seek branched history is stale, and applying it would act on a world the
    /// operator was not looking at.
    pub expected_epoch: u64,
    /// Optional finer check for callers that track revisions.
    pub expected_revision: Option<u64>,
    pub payload: CommandKind,
    /// The tick boundary at which this should take effect. `None` means the
    /// next boundary.
    pub requested_effective_tick: Option<u64>,
}

impl CommandEnvelope {
    /// A command for the current epoch, effective at the next boundary.
    pub fn new(
        command_id: u64,
        run_id: impl Into<String>,
        epoch: u64,
        payload: CommandKind,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            command_id,
            run_id: run_id.into(),
            expected_epoch: epoch,
            expected_revision: None,
            payload,
            requested_effective_tick: None,
        }
    }
}

/// Why a command was refused. §5 requires each of these to be distinguishable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum RejectReason {
    ProtocolVersion {
        found: u32,
        expected: u32,
    },
    /// Addressed to a different run.
    StaleRun {
        found: String,
        expected: String,
    },
    /// Composed against an epoch that has since branched.
    StaleEpoch {
        found: u64,
        expected: u64,
    },
    StaleRevision {
        found: u64,
        expected: u64,
    },
    /// The queue is full. The UI should back off and retry; the command was
    /// **not** accepted and **not** dropped in silence.
    Busy {
        capacity: usize,
    },
    /// Well-formed, but this build cannot honour it.
    Unsupported {
        name: String,
    },
    /// Rewind was asked for and no checkpoint covers the requested tick.
    RewindUnavailable {
        tick: u64,
    },
    /// The run is faulted; only reset or restore is accepted.
    Faulted,
    /// The id aged out of the dedup window, so whether it was already applied
    /// is unknown. §5: resynchronise rather than replay blindly.
    DedupWindowExpired {
        command_id: u64,
    },
}

/// The answer the caller gets, always.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandAck {
    pub command_id: u64,
    pub status: AckStatus,
    /// Which tick the command takes effect on, when accepted.
    pub effective_tick: Option<u64>,
    pub epoch: u64,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum AckStatus {
    Accepted,
    Rejected(RejectReason),
    /// This id was already answered; the original answer is repeated verbatim.
    /// Distinguished from a fresh `Accepted` so a caller can tell a retry that
    /// landed from a command that was applied twice.
    Duplicate,
}

impl CommandAck {
    pub fn is_accepted(&self) -> bool {
        matches!(self.status, AckStatus::Accepted)
    }
}

/// Bounded FIFO with a bounded dedup memory.
///
/// Both bounds are the configurable starting budgets from §5 — 256 queued, 4096
/// dedup entries — and §5 is explicit that they are *"resource limits, not
/// performance claims"*, to be measured before raising.
#[derive(Debug, Clone)]
pub struct CommandQueue {
    pending: VecDeque<CommandEnvelope>,
    capacity: usize,
    /// Answered ids in arrival order, alongside the answer to repeat.
    answered: VecDeque<(u64, CommandAck)>,
    dedup_capacity: usize,
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY, Self::DEFAULT_DEDUP_CAPACITY)
    }
}

impl CommandQueue {
    pub const DEFAULT_CAPACITY: usize = 256;
    pub const DEFAULT_DEDUP_CAPACITY: usize = 4096;

    pub fn new(capacity: usize, dedup_capacity: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            capacity,
            answered: VecDeque::new(),
            dedup_capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// The answer already given for this id, if it is still remembered.
    pub fn previous_ack(&self, command_id: u64) -> Option<&CommandAck> {
        self.answered
            .iter()
            .find(|(id, _)| *id == command_id)
            .map(|(_, ack)| ack)
    }

    /// Whether this id was answered and has since been forgotten.
    ///
    /// Cannot be known exactly without unbounded memory, so it is approximated
    /// the way §5 permits: ids below the oldest remembered one are treated as
    /// expired. Monotonic ids from a single UI make that exact; a caller that
    /// reuses ids out of order gets a conservative answer, which errs toward
    /// resynchronisation rather than toward replaying a command twice.
    pub fn is_expired(&self, command_id: u64) -> bool {
        match self.answered.front() {
            Some((oldest, _)) => command_id < *oldest && self.forgot_any(),
            None => false,
        }
    }

    fn forgot_any(&self) -> bool {
        self.answered.len() >= self.dedup_capacity
    }

    /// Push an accepted command. Returns `false` when the queue is full, which
    /// the caller must surface as [`RejectReason::Busy`].
    pub fn push(&mut self, envelope: CommandEnvelope) -> bool {
        if self.pending.len() >= self.capacity {
            return false;
        }
        self.pending.push_back(envelope);
        true
    }

    /// Remember an answer so a retry can be given the same one.
    pub fn remember(&mut self, ack: CommandAck) {
        self.answered.push_back((ack.command_id, ack));
        while self.answered.len() > self.dedup_capacity {
            self.answered.pop_front();
        }
    }

    /// Take everything due at or before `tick`, in arrival order.
    ///
    /// Commands scheduled for a later tick stay queued — that is what makes
    /// `requested_effective_tick` mean anything.
    pub fn drain_due(&mut self, tick: u64) -> Vec<CommandEnvelope> {
        let mut due = Vec::new();
        let mut deferred = VecDeque::new();
        while let Some(envelope) = self.pending.pop_front() {
            match envelope.requested_effective_tick {
                Some(effective) if effective > tick => deferred.push_back(envelope),
                _ => due.push(envelope),
            }
        }
        self.pending = deferred;
        due
    }

    /// Drop pending commands, keeping the dedup memory.
    ///
    /// Reset clears work that has not happened; it does not make the runtime
    /// forget which ids it already answered, because the UI's retries do not
    /// stop just because the run did.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    pub fn pending(&self) -> &VecDeque<CommandEnvelope> {
        &self.pending
    }

    /// Restore pending commands from a checkpoint.
    pub fn restore_pending(&mut self, commands: Vec<CommandEnvelope>) {
        self.pending = commands.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(id: u64) -> CommandEnvelope {
        CommandEnvelope::new(id, "run", 0, CommandKind::Pause)
    }

    #[test]
    fn a_full_queue_refuses_rather_than_dropping() {
        let mut queue = CommandQueue::new(2, 16);
        assert!(queue.push(envelope(1)));
        assert!(queue.push(envelope(2)));
        assert!(
            !queue.push(envelope(3)),
            "the third push must fail so the caller can answer Busy"
        );
        assert_eq!(queue.len(), 2, "the refused command did not displace one");
    }

    #[test]
    fn commands_come_out_in_arrival_order() {
        let mut queue = CommandQueue::default();
        for id in 1..=4 {
            assert!(queue.push(envelope(id)));
        }
        let ids: Vec<u64> = queue.drain_due(0).iter().map(|e| e.command_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_command_scheduled_later_stays_queued() {
        let mut queue = CommandQueue::default();
        let mut later = envelope(1);
        later.requested_effective_tick = Some(5);
        queue.push(later);
        queue.push(envelope(2));

        let due: Vec<u64> = queue.drain_due(0).iter().map(|e| e.command_id).collect();
        assert_eq!(due, vec![2], "only the unscheduled command was due");
        assert_eq!(queue.len(), 1);

        let due: Vec<u64> = queue.drain_due(5).iter().map(|e| e.command_id).collect();
        assert_eq!(due, vec![1], "and the scheduled one comes out at its tick");
    }

    #[test]
    fn deferred_commands_keep_their_relative_order() {
        let mut queue = CommandQueue::default();
        for id in 1..=3u64 {
            let mut later = envelope(id);
            later.requested_effective_tick = Some(9);
            queue.push(later);
        }
        queue.drain_due(0);
        let ids: Vec<u64> = queue.drain_due(9).iter().map(|e| e.command_id).collect();
        assert_eq!(ids, vec![1, 2, 3], "requeueing must not reverse them");
    }

    #[test]
    fn a_remembered_answer_is_repeated_for_a_retry() {
        let mut queue = CommandQueue::default();
        let ack = CommandAck {
            command_id: 7,
            status: AckStatus::Accepted,
            effective_tick: Some(3),
            epoch: 0,
            revision: 11,
        };
        queue.remember(ack.clone());
        assert_eq!(queue.previous_ack(7), Some(&ack));
        assert_eq!(queue.previous_ack(8), None);
    }

    #[test]
    fn the_dedup_window_is_bounded_and_expiry_is_detectable() {
        let mut queue = CommandQueue::new(4, 2);
        for id in 1..=4u64 {
            queue.remember(CommandAck {
                command_id: id,
                status: AckStatus::Accepted,
                effective_tick: None,
                epoch: 0,
                revision: id,
            });
        }
        // Only the newest two are remembered.
        assert!(queue.previous_ack(4).is_some());
        assert!(queue.previous_ack(3).is_some());
        assert!(queue.previous_ack(1).is_none());
        // And an id that fell out is reported as expired rather than as new,
        // which is what forces resynchronisation instead of a blind replay.
        assert!(queue.is_expired(1));
        assert!(!queue.is_expired(9), "an id from the future is not expired");
    }

    #[test]
    fn nothing_is_expired_before_anything_is_forgotten() {
        let mut queue = CommandQueue::new(4, 100);
        queue.remember(CommandAck {
            command_id: 50,
            status: AckStatus::Accepted,
            effective_tick: None,
            epoch: 0,
            revision: 1,
        });
        assert!(
            !queue.is_expired(10),
            "with room to spare, an unseen id is simply unseen"
        );
    }

    #[test]
    fn reset_clears_pending_work_but_not_the_dedup_memory() {
        let mut queue = CommandQueue::default();
        queue.push(envelope(1));
        queue.remember(CommandAck {
            command_id: 1,
            status: AckStatus::Accepted,
            effective_tick: None,
            epoch: 0,
            revision: 1,
        });
        queue.clear_pending();
        assert!(queue.is_empty());
        assert!(
            queue.previous_ack(1).is_some(),
            "a retry after reset must still get its original answer"
        );
    }

    #[test]
    fn an_envelope_round_trips_through_json() {
        let envelope = CommandEnvelope::new(3, "run", 2, CommandKind::SetSpeed { multiplier: 0.5 });
        let json = serde_json::to_string(&envelope).expect("serializable");
        let parsed: CommandEnvelope = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(parsed, envelope);
    }
}
