//! The tick transaction.
//!
//! One owner, one mutable world, one committed snapshot per tick. The rules
//! implemented here come from
//! [`ARCHITECTURE_AND_PROTOCOLS.md`](../../../docs/migration/ARCHITECTURE_AND_PROTOCOLS.md)
//! §4, and the sharp one is what happens when a stage fails:
//!
//! > *"On an unexpected mid-tick failure, publish no partial snapshot, mark the
//! > run faulted and require reset or restore from the last complete
//! > checkpoint. Do not continue a half-updated run or promise automatic
//! > rollback without retaining sufficient state."*
//!
//! So there is no error path that returns a snapshot, and no `step` that
//! quietly resumes after a failure. A faulted run refuses to step until someone
//! decides what to do about it.
//!
//! # What is and is not here yet
//!
//! P4.1 builds the substrate, not the physics. Stages 1–9 are delegated to
//! [`StageHandler`]s and the default handlers do nothing, so a run of this
//! shell advances the clock, commits empty snapshots and produces a trace. That
//! is deliberate — P4.1 says *"Keep it runnable with fixture sources; do not
//! present it as the production replacement yet."* Stages 0 and 10 are
//! implemented here because they are the transaction boundary rather than
//! domain work: the first admits commands, the last freezes the revision.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::clock::Clock;
use crate::command::{
    AckStatus, CommandAck, CommandEnvelope, CommandKind, CommandQueue, RejectReason,
    PROTOCOL_VERSION,
};
use crate::config::{ConfigError, SimulationConfig, ValidatedConfig};
use crate::snapshot::{
    CheckpointHeader, RestoreError, RuntimeSnapshot, SimulationCheckpoint, SnapshotHeader,
};
use crate::stage::{Stage, StageError, StageOutcome, StageRecord, StageTrace};
use crate::world::TickWorld;

use argusnet_types::{NodeState, TrackState, TruthState};

/// What a stage handler is given, and what it may record.
///
/// Read-only over the clock on purpose: a stage that could advance time would
/// break the "derived, never accumulated" rule from [`crate::clock`] from the
/// inside, where no test of the clock would see it.
#[derive(Debug)]
pub struct TickContext<'a> {
    pub tick: u64,
    pub timestamp_s: f64,
    /// Base seed. Stages derive their own per-key streams from it; see
    /// [`crate::deterministic`].
    pub seed: u32,
    /// The millisecond key `build_observations` uses.
    pub seed_time_key: i64,
    /// The tick's mutable platform and track state.
    ///
    /// One list, edited in place. P4.8 asks to "eliminate repeated
    /// dataclass-style list rebuilds; freeze once at the snapshot boundary",
    /// and this is the list that is not rebuilt -- see [`crate::world`].
    pub world: &'a mut TickWorld,
    /// The true world, for the stages allowed to see it.
    ///
    /// `None` for every stage [`Stage::may_read_truth`] does not admit, which
    /// is most of them. §3 makes truth a capability; this field is that
    /// capability, and a handler handed `None` has no other way to reach
    /// truth, so the restriction is structural rather than advisory.
    pub truth: Option<&'a [TruthState]>,
    notes: &'a mut Vec<String>,
}

impl TickContext<'_> {
    /// Record something the trace should carry.
    ///
    /// §4 asks specifically that stage 9 record *which future tick* a
    /// controller change takes effect on. This is where that goes, and putting
    /// it in the trace rather than a log means it can be compared mechanically
    /// against the Python reference.
    pub fn note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

/// A pluggable stage body.
///
/// Stages arrive one PR at a time through P4.2–P4.8. Registering them rather
/// than hard-coding a call sequence means adding one cannot accidentally
/// reorder the others — the order lives in [`Stage::PIPELINE`] and nowhere
/// else.
pub trait StageHandler: fmt::Debug + Send {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError>;
}

/// The default body for a stage that has not been ported yet.
///
/// Reports [`StageOutcome::NoWork`] rather than succeeding silently, so a trace
/// from the shell is visibly a trace of a shell.
#[derive(Debug, Default)]
pub struct UnportedStage;

impl StageHandler for UnportedStage {
    fn run(&mut self, _context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        Ok(StageOutcome::NoWork)
    }
}

/// Why the run is not stepping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickError {
    /// A stage failed on this tick. Nothing was published.
    Stage {
        tick: u64,
        stage: Stage,
        error: StageError,
    },
    /// The run faulted earlier and has not been reset or restored.
    Faulted { at_tick: u64, stage: Stage },
    /// The configured step count is exhausted. Not a failure.
    Completed { tick: u64 },
    /// The run is paused. Not a failure either — `step` is how a caller finds
    /// out, since a paused run that silently advanced would be worse.
    Paused,
}

impl fmt::Display for TickError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TickError::Stage { tick, stage, error } => {
                write!(formatter, "tick {tick} failed in stage {stage}: {error}")
            }
            TickError::Faulted { at_tick, stage } => write!(
                formatter,
                "run is faulted (stage {stage} failed at tick {at_tick}); reset or restore \
                 from a checkpoint before stepping again"
            ),
            TickError::Completed { tick } => {
                write!(formatter, "run completed after {tick} ticks")
            }
            TickError::Paused => write!(formatter, "run is paused"),
        }
    }
}

impl std::error::Error for TickError {}

/// Whether the run may step.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RunState {
    Ready,
    Faulted { at_tick: u64, stage: Stage },
}

/// The simulation owner.
pub struct Simulation {
    config: ValidatedConfig,
    clock: Clock,
    epoch: u64,
    revision: u64,
    state: RunState,
    queue: CommandQueue,
    trace: StageTrace,
    handlers: BTreeMap<Stage, Box<dyn StageHandler>>,
    last_snapshot: Option<Arc<RuntimeSnapshot>>,
    paused: bool,
    /// Wall-clock pacing only. §5: *"Playback speed scales desired wall-clock
    /// pacing, never physics dt."* Nothing in `step` reads it.
    speed: f64,
    world_revision: u64,
    /// The tick's mutable state. Stages edit it; stage 10 freezes it once.
    world: TickWorld,
    /// The true world. Reaches a stage only through
    /// [`Stage::may_read_truth`]; never reaches a snapshot at all.
    truth: Vec<TruthState>,
    belief_revision: u64,
    journal_cursor: u64,
}

impl fmt::Debug for Simulation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Simulation")
            .field("run_id", &self.config.run_id())
            .field("tick", &self.clock.tick())
            .field("epoch", &self.epoch)
            .field("revision", &self.revision)
            .field("state", &self.state)
            .field("paused", &self.paused)
            .finish_non_exhaustive()
    }
}

impl Simulation {
    /// Build a run, validating the configuration before anything is allocated.
    pub fn new(config: SimulationConfig) -> Result<Self, ConfigError> {
        let config = ValidatedConfig::new(config)?;
        let clock = Clock::new(config.timestep());
        Ok(Self {
            config,
            clock,
            epoch: 0,
            revision: 0,
            state: RunState::Ready,
            queue: CommandQueue::default(),
            trace: StageTrace::bounded(Self::DEFAULT_TRACE_CAPACITY),
            handlers: BTreeMap::new(),
            last_snapshot: None,
            paused: false,
            speed: 1.0,
            world_revision: 0,
            belief_revision: 0,
            journal_cursor: 0,
            world: TickWorld::default(),
            truth: Vec::new(),
        })
    }

    /// Trace records retained by default.
    ///
    /// Eleven stages a tick, so this is roughly the last 400 ticks. A
    /// diagnostic window, not a journal — durable events use the ordered
    /// journal §5 describes.
    pub const DEFAULT_TRACE_CAPACITY: usize = 4_400;

    /// Install the body for one stage.
    ///
    /// Stages 0 and 10 are the transaction boundary and are owned by the
    /// simulation; registering a handler for either is refused rather than
    /// silently ignored, because a handler that never runs is a bug that hides.
    pub fn set_stage_handler(
        &mut self,
        stage: Stage,
        handler: Box<dyn StageHandler>,
    ) -> Result<(), StageError> {
        if matches!(stage, Stage::ApplyCommands | Stage::CommitSnapshot) {
            return Err(StageError::Invariant(format!(
                "stage {stage} is the tick transaction boundary and is owned by Simulation"
            )));
        }
        self.handlers.insert(stage, handler);
        Ok(())
    }

    pub fn tick(&self) -> u64 {
        self.clock.tick()
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn is_faulted(&self) -> bool {
        matches!(self.state, RunState::Faulted { .. })
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn trace(&self) -> &StageTrace {
        &self.trace
    }

    /// The last committed snapshot, if any tick has committed.
    /// Load the fleet and any existing tracks.
    ///
    /// The run driver's entry point: `TickWorld` is what the stages mutate, and
    /// nothing inside the pipeline creates platforms.
    pub fn load_world(&mut self, platforms: Vec<NodeState>, tracks: Vec<TrackState>) {
        self.world = TickWorld::new(platforms, tracks);
    }

    /// Supply the true world for the ticks that follow.
    ///
    /// Written from outside the pipeline because truth is prescribed by the
    /// scenario rather than integrated by a stage. It reaches a stage only
    /// through [`Stage::may_read_truth`] and never reaches a snapshot.
    pub fn set_truth(&mut self, truth: Vec<TruthState>) {
        self.truth = truth;
    }

    /// Read-only view of the tick's mutable state.
    pub fn world(&self) -> &TickWorld {
        &self.world
    }

    pub fn latest_snapshot(&self) -> Option<Arc<RuntimeSnapshot>> {
        self.last_snapshot.clone()
    }

    /// Validate a command and queue it. Always answers.
    pub fn apply_command(&mut self, envelope: CommandEnvelope) -> CommandAck {
        // A retry must get the original answer, never a second application.
        if let Some(previous) = self.queue.previous_ack(envelope.command_id) {
            let mut repeated = previous.clone();
            repeated.status = AckStatus::Duplicate;
            return repeated;
        }
        if self.queue.is_expired(envelope.command_id) {
            return self.reject(
                envelope.command_id,
                RejectReason::DedupWindowExpired {
                    command_id: envelope.command_id,
                },
            );
        }

        if envelope.protocol_version != PROTOCOL_VERSION {
            return self.reject(
                envelope.command_id,
                RejectReason::ProtocolVersion {
                    found: envelope.protocol_version,
                    expected: PROTOCOL_VERSION,
                },
            );
        }
        if envelope.run_id != self.config.run_id() {
            return self.reject(
                envelope.command_id,
                RejectReason::StaleRun {
                    found: envelope.run_id.clone(),
                    expected: self.config.run_id().to_string(),
                },
            );
        }
        if envelope.expected_epoch != self.epoch {
            return self.reject(
                envelope.command_id,
                RejectReason::StaleEpoch {
                    found: envelope.expected_epoch,
                    expected: self.epoch,
                },
            );
        }
        if let Some(expected) = envelope.expected_revision {
            if expected != self.revision {
                return self.reject(
                    envelope.command_id,
                    RejectReason::StaleRevision {
                        found: expected,
                        expected: self.revision,
                    },
                );
            }
        }
        // A faulted run accepts only the commands that can clear the fault.
        if self.is_faulted() && !matches!(envelope.payload, CommandKind::Reset) {
            return self.reject(envelope.command_id, RejectReason::Faulted);
        }
        if let CommandKind::Unsupported { name } = &envelope.payload {
            return self.reject(
                envelope.command_id,
                RejectReason::Unsupported { name: name.clone() },
            );
        }
        if let CommandKind::Seek { tick } = envelope.payload {
            // Honest refusal: this shell retains no checkpoint history, so
            // there is nothing to seek to. Answering "accepted" and then not
            // moving would be the silent failure §5 forbids.
            return self.reject(
                envelope.command_id,
                RejectReason::RewindUnavailable { tick },
            );
        }

        let effective_tick = envelope
            .requested_effective_tick
            .unwrap_or_else(|| self.clock.tick());
        let command_id = envelope.command_id;
        if !self.queue.push(envelope) {
            return self.reject(
                command_id,
                RejectReason::Busy {
                    capacity: self.queue.capacity(),
                },
            );
        }

        let ack = CommandAck {
            command_id,
            status: AckStatus::Accepted,
            effective_tick: Some(effective_tick),
            epoch: self.epoch,
            revision: self.revision,
        };
        self.queue.remember(ack.clone());
        ack
    }

    fn reject(&mut self, command_id: u64, reason: RejectReason) -> CommandAck {
        let ack = CommandAck {
            command_id,
            status: AckStatus::Rejected(reason),
            effective_tick: None,
            epoch: self.epoch,
            revision: self.revision,
        };
        self.queue.remember(ack.clone());
        ack
    }

    /// Run one tick, or explain why not.
    ///
    /// On success exactly one snapshot is committed. On failure nothing is
    /// published and the run is faulted.
    pub fn step(&mut self) -> Result<Arc<RuntimeSnapshot>, TickError> {
        if let RunState::Faulted { at_tick, stage } = &self.state {
            return Err(TickError::Faulted {
                at_tick: *at_tick,
                stage: *stage,
            });
        }
        if let Some(steps) = self.config.steps() {
            if self.clock.tick() >= steps {
                return Err(TickError::Completed {
                    tick: self.clock.tick(),
                });
            }
        }

        let tick = self.clock.tick();
        self.world.begin_tick();

        // Stage 0 runs before the pause check, and that ordering is load-bearing.
        // §5 says pause is "checked between ticks", and commands are admitted at
        // the same boundary. Checking pause first instead would trap the run: a
        // queued Resume is applied in stage 0, stage 0 only runs inside `step`,
        // and `step` would have already refused -- so nothing could ever
        // un-pause it.
        {
            let mut notes = Vec::new();
            match self.run_apply_commands(tick, &mut notes) {
                Ok(outcome) => self.trace.record(StageRecord {
                    tick,
                    order: Stage::ApplyCommands.order(),
                    stage: Stage::ApplyCommands,
                    outcome,
                    notes,
                }),
                Err(error) => {
                    self.trace.record(StageRecord {
                        tick,
                        order: Stage::ApplyCommands.order(),
                        stage: Stage::ApplyCommands,
                        outcome: StageOutcome::Failed {
                            detail: error.to_string(),
                        },
                        notes,
                    });
                    self.state = RunState::Faulted {
                        at_tick: tick,
                        stage: Stage::ApplyCommands,
                    };
                    return Err(TickError::Stage {
                        tick,
                        stage: Stage::ApplyCommands,
                        error,
                    });
                }
            }
        }

        // A Reset applied above returns the run to tick 0 on a new epoch, so
        // the rest of this tick would be operating on a clock that no longer
        // matches `tick`. Stop here and let the caller step the fresh run.
        if self.clock.tick() != tick {
            return Ok(self.commit_current());
        }

        if self.paused {
            return Err(TickError::Paused);
        }

        for stage in Stage::PIPELINE.into_iter().skip(1) {
            let mut notes = Vec::new();
            let outcome = match stage {
                Stage::ApplyCommands => unreachable!("stage 0 ran before the pause check"),
                Stage::CommitSnapshot => Ok(StageOutcome::Ran),
                _ => {
                    let mut context = TickContext {
                        tick,
                        timestamp_s: self.clock.timestamp_s(),
                        seed: self.config.seed(),
                        seed_time_key: self.clock.seed_time_key(),
                        world: &mut self.world,
                        // The capability, handed out by stage and nowhere else.
                        truth: stage.may_read_truth().then_some(self.truth.as_slice()),
                        notes: &mut notes,
                    };
                    match self.handlers.get_mut(&stage) {
                        Some(handler) => handler.run(&mut context),
                        None => UnportedStage.run(&mut context),
                    }
                }
            };

            match outcome {
                Ok(outcome) => self.trace.record(StageRecord {
                    tick,
                    order: stage.order(),
                    stage,
                    outcome,
                    notes,
                }),
                Err(error) => {
                    // Record the failure, then stop. Nothing after this stage
                    // ran, and the trace shows exactly that.
                    self.trace.record(StageRecord {
                        tick,
                        order: stage.order(),
                        stage,
                        outcome: StageOutcome::Failed {
                            detail: error.to_string(),
                        },
                        notes,
                    });
                    self.state = RunState::Faulted {
                        at_tick: tick,
                        stage,
                    };
                    return Err(TickError::Stage { tick, stage, error });
                }
            }
        }

        // Commit. Only now does the tick advance and a revision become visible.
        self.clock.advance();
        Ok(self.commit_current())
    }

    /// Freeze a snapshot at the current clock position and publish it.
    fn commit_current(&mut self) -> Arc<RuntimeSnapshot> {
        self.revision += 1;
        self.world_revision = self.world.revision();
        // The one clone of the tick. Everything before this mutated in place.
        let (platforms, tracks) = self.world.freeze();
        let snapshot = Arc::new(RuntimeSnapshot {
            header: SnapshotHeader {
                run_id: self.config.run_id().to_string(),
                epoch: self.epoch,
                tick: self.clock.tick(),
                revision: self.revision,
                timestamp_s: self.clock.timestamp_s(),
                world_revision: self.world_revision,
                belief_revision: self.belief_revision,
                event_cursor: self.journal_cursor,
                // Not a filtering step someone has to remember: `TickWorld`
                // never held truth, so this is a statement about the type.
                truth_filtered: true,
            },
            platforms,
            tracks,
        });
        self.last_snapshot = Some(Arc::clone(&snapshot));
        snapshot
    }

    fn run_apply_commands(
        &mut self,
        tick: u64,
        notes: &mut Vec<String>,
    ) -> Result<StageOutcome, StageError> {
        let due = self.queue.drain_due(tick);
        if due.is_empty() {
            return Ok(StageOutcome::NoWork);
        }
        for envelope in due {
            // §4: "Log accepted/rejected command and effective tick."
            notes.push(format!(
                "applied command {} ({:?}) effective at tick {tick}",
                envelope.command_id, envelope.payload
            ));
            match envelope.payload {
                CommandKind::Pause => self.paused = true,
                CommandKind::Resume => self.paused = false,
                CommandKind::SingleStep => self.paused = false,
                CommandKind::SetSpeed { multiplier } => self.speed = multiplier,
                CommandKind::Reset => self.reset(),
                // Refused at admission; unreachable here, and an invariant
                // rather than a silent skip so a future edit to `apply_command`
                // that lets one through fails loudly.
                CommandKind::Seek { .. } | CommandKind::Unsupported { .. } => {
                    return Err(StageError::Invariant(format!(
                        "command {} should have been rejected at admission",
                        envelope.command_id
                    )))
                }
            }
        }
        Ok(StageOutcome::Ran)
    }

    /// Return to tick zero on a new epoch, clearing any fault.
    ///
    /// A new epoch because §6 treats a branch in history as a new epoch, and a
    /// consumer holding a pre-reset snapshot must be able to tell that it no
    /// longer describes this run — §5 forbids interpolating across epochs, and
    /// it can only honour that if the epoch actually changes.
    ///
    /// The revision keeps counting up rather than restarting: it orders
    /// snapshots a consumer may receive out of order, and restarting it would
    /// make a stale snapshot look newer than a current one.
    pub fn reset(&mut self) {
        self.clock.reset();
        self.epoch += 1;
        self.state = RunState::Ready;
        self.queue.clear_pending();
        self.trace.clear();
        self.last_snapshot = None;
        self.paused = false;
        self.speed = 1.0;
        self.belief_revision = 0;
        self.journal_cursor = 0;
        // A reset is a new epoch, so the previous epoch's platforms are not
        // this run's. Truth goes with them: the caller reloads both, and
        // leaving stale truth behind would let a fresh run sense the old
        // world for one tick.
        self.world.clear();
        self.truth.clear();
    }

    /// Freeze everything needed to resume.
    ///
    /// Refused while faulted: §4 requires restoring from *the last complete
    /// checkpoint*, and a checkpoint taken from half-mutated state would be
    /// exactly the thing that rule exists to prevent.
    pub fn checkpoint(&self) -> Result<SimulationCheckpoint, TickError> {
        if let RunState::Faulted { at_tick, stage } = &self.state {
            return Err(TickError::Faulted {
                at_tick: *at_tick,
                stage: *stage,
            });
        }
        Ok(SimulationCheckpoint {
            header: CheckpointHeader {
                format_version: CheckpointHeader::FORMAT_VERSION,
                runtime_algorithm_version: CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
                seed_version: CheckpointHeader::SEED_VERSION,
                run_id: self.config.run_id().to_string(),
                epoch: self.epoch,
                tick: self.clock.tick(),
                config_hash: self.config.content_hash(),
                world_content_hash: self.world_revision,
                belief_revision: self.belief_revision,
                journal_cursor: self.journal_cursor,
            },
            tick: self.clock.tick(),
            seed: self.config.seed(),
            pending_commands: self.queue.pending().iter().cloned().collect(),
        })
    }

    /// Resume from a checkpoint, or refuse with a reason.
    ///
    /// Every compatibility field is checked *before* any state is touched, so a
    /// refused restore leaves the run exactly as it was — including a faulted
    /// run, which stays faulted rather than being half-recovered.
    pub fn restore(&mut self, checkpoint: &SimulationCheckpoint) -> Result<(), RestoreError> {
        checkpoint.verify(self.config.run_id(), self.config.content_hash())?;

        self.clock.reset();
        for _ in 0..checkpoint.tick {
            self.clock.advance();
        }
        self.epoch = checkpoint.header.epoch;
        self.belief_revision = checkpoint.header.belief_revision;
        self.journal_cursor = checkpoint.header.journal_cursor;
        self.world_revision = checkpoint.header.world_content_hash;
        self.state = RunState::Ready;
        self.queue.clear_pending();
        self.queue
            .restore_pending(checkpoint.pending_commands.clone());
        self.trace.clear();
        self.last_snapshot = None;
        self.paused = false;
        Ok(())
    }
}
