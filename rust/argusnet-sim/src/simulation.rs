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
use std::time::Instant;

use crate::clock::Clock;
use crate::command::{
    AckStatus, CommandAck, CommandEnvelope, CommandKind, CommandQueue, RejectReason,
    PROTOCOL_VERSION,
};
use crate::config::{ConfigError, SimulationConfig, ValidatedConfig};
use crate::controller::ControllerStage;
use crate::fusion::FusionStage;
use crate::mission_driver::MissionDriverStage;
use crate::observe::ObserveStage;
use crate::overrides::MissionOverrideStage;
use crate::safety::SafetyStage;
use crate::snapshot::{
    CheckpointHeader, RestoreError, RuntimeSnapshot, SimulationCheckpoint, SnapshotHeader,
};
use crate::stage::{Stage, StageError, StageOutcome, StageRecord, StageTrace};
use crate::timing::StageTimings;
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
    /// The tick interval. Derived from the config, never accumulated, and
    /// carried here because a stage that reached for its own copy would be one
    /// more place the two could disagree.
    pub dt_s: f64,
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
    /// Read-only views of the other installed stages, for same-tick reads
    /// (SESSION_OWNERSHIP.md §2a).
    pub stages: StageViews<'a>,
    notes: &'a mut Vec<String>,
}

/// The installed stages, lent read-only to the stage that is running.
///
/// The running stage's own handler is not among them: it is taken out of the
/// map for its run, which is what makes lending the rest sound. Stages run one
/// at a time, so nothing can write what a view reads while it reads it.
///
/// Nor, for a stage [`Stage::may_read_truth`] does not admit, are the stages it
/// does admit: their state is derived from truth (stage 6 keeps each pair's
/// true range), so lending it would hand truth to a stage the capability
/// withholds it from (A3.4).
#[derive(Clone, Copy)]
pub struct StageViews<'a> {
    handlers: &'a BTreeMap<Stage, Box<dyn StageHandler>>,
    truth_visible: bool,
}

impl<'a> StageViews<'a> {
    /// The installed handler for `stage`, if it is a `T` and is not the stage
    /// asking.
    pub fn get<T: StageHandler>(self, stage: Stage) -> Option<&'a T> {
        if stage.may_read_truth() && !self.truth_visible {
            return None;
        }
        let handler: &dyn std::any::Any = self.handlers.get(&stage)?.as_ref();
        handler.downcast_ref::<T>()
    }
}

impl fmt::Debug for StageViews<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_set().entries(self.handlers.keys()).finish()
    }
}

/// Why a [`ProductionBuilder`] would not build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A stage the run requires has no handler: it would run as
    /// [`UnportedStage`] and silently do nothing.
    MissingStage { stage: Stage },
    /// Installing a handler or connecting a handoff was refused.
    Stage(StageError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::MissingStage { stage } => write!(
                formatter,
                "production run is missing required stage {stage}: it would run unported"
            ),
            BuildError::Stage(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<StageError> for BuildError {
    fn from(error: StageError) -> Self {
        BuildError::Stage(error)
    }
}

/// A production [`Simulation`] under construction (A1.3).
///
/// Starts out requiring [`Stage::PRODUCTION_CORE`]; a run that needs more
/// (the mission stages, the safety floor) says so with [`Self::require`].
/// [`Self::build`] refuses any required stage left unported, naming it.
#[derive(Debug)]
pub struct ProductionBuilder {
    sim: Simulation,
    required: std::collections::BTreeSet<Stage>,
}

impl ProductionBuilder {
    /// Require `stage` as well as the core.
    pub fn require(&mut self, stage: Stage) -> &mut Self {
        self.required.insert(stage);
        self
    }

    /// The stages [`Self::build`] will insist on.
    pub fn required(&self) -> impl Iterator<Item = Stage> + '_ {
        self.required.iter().copied()
    }

    /// [`Simulation::set_stage_handler`].
    pub fn install(
        &mut self,
        stage: Stage,
        handler: Box<dyn StageHandler>,
    ) -> Result<&mut Self, BuildError> {
        self.sim.set_stage_handler(stage, handler)?;
        Ok(self)
    }

    /// [`Simulation::connect`].
    pub fn connect(&mut self, handoff: Handoff) -> Result<&mut Self, BuildError> {
        self.sim.connect(handoff)?;
        Ok(self)
    }

    /// Bind the run to the scenario it is assembled from: checkpoints carry
    /// this hash, and restore refuses one taken from another scenario before
    /// changing anything (A2.2).
    pub fn scenario_hash(&mut self, hash: u64) -> &mut Self {
        self.sim.scenario_hash = hash;
        self
    }

    /// [`Simulation::load_world`].
    pub fn load_world(&mut self, platforms: Vec<NodeState>, tracks: Vec<TrackState>) {
        self.sim.load_world(platforms, tracks);
    }

    /// The run, if every required stage is installed with a real handler;
    /// otherwise the first missing one in pipeline order. An explicitly
    /// installed [`UnportedStage`] counts as missing.
    pub fn build(self) -> Result<Simulation, BuildError> {
        let ported = |stage: &Stage| {
            self.sim.handlers.get(stage).is_some_and(|handler| {
                let handler: &dyn std::any::Any = handler.as_ref();
                !handler.is::<UnportedStage>()
            })
        };
        if let Some(stage) = self.required.iter().find(|stage| !ported(stage)) {
            return Err(BuildError::MissingStage { stage: *stage });
        }
        Ok(self.sim)
    }
}

/// A typed write from one stage into another, applied by [`Simulation`]
/// between stages (SESSION_OWNERSHIP.md §2a).
///
/// The edge is named here and applied in [`Simulation::step`], in a fixed
/// order, instead of travelling through a handle the two stages share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Handoff {
    /// 6→1: stage 6's accepted-observation counts engage stage 1's
    /// search-then-follow controllers. Engagement is visible as soon as stage 6
    /// has run; the motion it causes starts with the next tick's stage 1.
    EngagementToControllers,
    /// 6→7: stage 6's observation batch is what stage 7 fuses this tick.
    ObservationsToFusion,
    /// 4→7: stage 4's validations are stage 7's `safety_events` this tick.
    SafetyEventsToFusion,
    /// 9→2: the POI and egress writes stage 9 queued, applied to stage 2's
    /// tables in the order queued. Stage 2 flies them from the next tick.
    /// Applied after every run, whatever it reported, because stage 9 queues
    /// writes on runs that report `NoWork` too.
    OverridesFromMission,
    /// 6→8: stage 6's range scans are what stage 8 maps this tick, in a run
    /// that builds its own map. Handed on whatever stage 6 reported: a stage
    /// with no bearing sensors still scans.
    RangeScansToMapping,
}

impl Handoff {
    /// The stage whose successful run the handoff follows.
    pub fn after(self) -> Stage {
        match self {
            Handoff::EngagementToControllers
            | Handoff::ObservationsToFusion
            | Handoff::RangeScansToMapping => Stage::GenerateObservations,
            Handoff::SafetyEventsToFusion => Stage::SafetyMonitor,
            Handoff::OverridesFromMission => Stage::EvaluateMissionTransitions,
        }
    }

    /// Whether a run that reported `outcome` hands anything on.
    fn follows(self, outcome: &StageOutcome) -> bool {
        match self {
            Handoff::EngagementToControllers
            | Handoff::ObservationsToFusion
            | Handoff::SafetyEventsToFusion => *outcome == StageOutcome::Ran,
            Handoff::OverridesFromMission | Handoff::RangeScansToMapping => true,
        }
    }
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
pub trait StageHandler: std::any::Any + fmt::Debug + Send {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError>;

    /// Return to the state a freshly constructed handler would have.
    ///
    /// [`Simulation::reset`] calls this on every installed handler. Without
    /// it, `reset` returned the clock to zero and left every handler's
    /// accumulated state in place, so a "fresh" run inherited the previous
    /// one's battery ledger and controller memory.
    ///
    /// The default is **a declaration, not a freebie**: implementing nothing
    /// asserts that this handler holds no state that survives a tick.
    fn reset(&mut self) {}

    /// This boundary the stage is not due (B2.4: its lane runs at its own
    /// rate) and does not run. Clear whatever the last run left as *this
    /// tick's* output -- events, decisions, commands for a later stage -- so
    /// a boundary it did not run reports none.
    ///
    /// The default refuses: only a handler that has said what its per-tick
    /// outputs are may be skipped. One that never ran is fine to skip.
    fn not_due(&mut self) -> Result<(), StageError> {
        Err(StageError::Invariant(
            "this stage has not declared how to sit out a boundary".to_string(),
        ))
    }

    /// Causal state this handler needs back in order to resume identically.
    ///
    /// `None` means "I hold nothing that affects a future decision". §6: what
    /// belongs here is *anything affecting a future decision*; render state,
    /// wall-clock timers and rebuildable caches do not.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        None
    }

    /// Check that `state` is something [`StageHandler::restore_state`] would
    /// accept, **without** taking it.
    ///
    /// Separate from `restore_state` because a restore must be refused before
    /// anything is mutated: with several stateful handlers, taking each one's
    /// state as it validated would leave the first ones restored and the rest
    /// not when a later one refused.
    fn check_state(&self, state: &serde_json::Value) -> Result<(), StageStateError> {
        Err(StageStateError::Unexpected {
            found: state.to_string(),
        })
    }

    /// Take back what [`StageHandler::checkpoint_state`] produced.
    ///
    /// Only called after [`StageHandler::check_state`] accepted the same
    /// value, so an error here is an invariant failure rather than a refusal.
    fn restore_state(&mut self, state: &serde_json::Value) -> Result<(), StageStateError> {
        Err(StageStateError::Unexpected {
            found: state.to_string(),
        })
    }
}

/// Why a handler could not take its state back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageStateError {
    /// The handler does not checkpoint anything, but was handed something --
    /// a checkpoint written by a build that knew more than this one.
    Unexpected { found: String },
    /// The value was not the shape this handler writes.
    Malformed { detail: String },
}

impl fmt::Display for StageStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageStateError::Unexpected { found } => write!(
                formatter,
                "this handler holds no checkpointable state, but the checkpoint carries {found}"
            ),
            StageStateError::Malformed { detail } => {
                write!(formatter, "handler state could not be read: {detail}")
            }
        }
    }
}

impl std::error::Error for StageStateError {}

/// Decode a handler's checkpointed state, mapping a shape error to
/// [`StageStateError::Malformed`]. The helper every stateful handler's
/// `check_state`/`restore_state` pair wants.
pub fn decode_stage_state<T: serde::de::DeserializeOwned>(
    state: &serde_json::Value,
) -> Result<T, StageStateError> {
    serde_json::from_value(state.clone()).map_err(|error| StageStateError::Malformed {
        detail: error.to_string(),
    })
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

/// What [`Simulation::begin_tick`] left.
#[derive(Debug)]
pub enum TickStart {
    /// The tick is open: run its stages, then commit it.
    Open(OpenTick),
    /// Stage 0 applied a Reset and committed the fresh run; there is nothing
    /// more to run this tick.
    Committed(Arc<RuntimeSnapshot>),
}

/// A tick in progress: how far its stages have run.
#[derive(Debug)]
pub struct OpenTick {
    tick: u64,
    epoch: u64,
    /// The next stage's position in [`Stage::PIPELINE`].
    next: usize,
}

impl OpenTick {
    pub fn tick(&self) -> u64 {
        self.tick
    }
}

impl crate::scheduler::PhasedTick for Simulation {
    type Open = OpenTick;
    type Output = Arc<RuntimeSnapshot>;
    type Error = TickError;

    fn current_tick(&self) -> u64 {
        self.tick()
    }
    fn begin(&mut self) -> Result<Result<OpenTick, Arc<RuntimeSnapshot>>, TickError> {
        Ok(match self.begin_tick()? {
            TickStart::Open(open) => Ok(open),
            TickStart::Committed(snapshot) => Err(snapshot),
        })
    }
    fn run_through(&mut self, open: &mut OpenTick, last: Stage) -> Result<(), TickError> {
        Simulation::run_through(self, open, last)
    }
    fn skip_through(
        &mut self,
        open: &mut OpenTick,
        last: Stage,
        reason: &str,
    ) -> Result<(), TickError> {
        Simulation::skip_through(self, open, last, reason)
    }
    fn commit(&mut self, open: OpenTick) -> Result<Arc<RuntimeSnapshot>, TickError> {
        self.commit_tick(open)
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
    /// A tick run in parts was driven out of order (B2.3): a stage asked for
    /// twice or skipped, or a commit before every stage ran.
    OutOfOrder { detail: String },
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
            TickError::OutOfOrder { detail } => {
                write!(formatter, "tick run out of order: {detail}")
            }
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
    single_step: bool,
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
    /// Per-stage wall time, when a caller asked for it. Never causal: it is
    /// not in the trace, the replay or a checkpoint.
    stage_timings: Option<StageTimings>,
    /// The connected handoffs, in the order they apply.
    handoffs: Vec<Handoff>,
    /// What [`ProductionBuilder::scenario_hash`] recorded; `0` for a bare run.
    scenario_hash: u64,
    /// Boundaries each stage has been not due for since it last ran (B2.4).
    /// Empty unless a lane runs at its own rate.
    stage_lag: BTreeMap<Stage, u64>,
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
    /// A production run: a [`ProductionBuilder`] that refuses to build until
    /// every stage the run requires is installed (A1.3). The CLI, and later
    /// the viewer worker, construct runs only this way.
    pub fn production(config: SimulationConfig) -> Result<ProductionBuilder, ConfigError> {
        Ok(ProductionBuilder {
            sim: Self::bare(config)?,
            required: Stage::PRODUCTION_CORE.into_iter().collect(),
        })
    }

    /// A run with every stage unported, validating the configuration before
    /// anything is allocated.
    ///
    /// **For tests and benches.** Nothing checks that the stages a real run
    /// needs are installed, so a production path must use
    /// [`Self::production`] instead.
    pub fn bare(config: SimulationConfig) -> Result<Self, ConfigError> {
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
            single_step: false,
            speed: 1.0,
            world_revision: 0,
            belief_revision: 0,
            journal_cursor: 0,
            world: TickWorld::default(),
            truth: Vec::new(),
            stage_timings: None,
            handoffs: Vec::new(),
            scenario_hash: 0,
            stage_lag: BTreeMap::new(),
        })
    }

    /// Start recording each stage's wall time (P4.10-pre). Off by default:
    /// it keeps one sample per stage per tick, which suits a batch run and
    /// not an unbounded interactive one. Reset and restore keep the samples;
    /// they describe the machine, not the run.
    pub fn enable_stage_timing(&mut self) {
        self.stage_timings.get_or_insert_with(StageTimings::default);
    }

    /// The recorded wall times, if [`Self::enable_stage_timing`] was called.
    pub fn stage_timings(&self) -> Option<&StageTimings> {
        self.stage_timings.as_ref()
    }

    fn time_stage(&mut self, stage: Stage, started: Option<Instant>) {
        if let (Some(timings), Some(started)) = (self.stage_timings.as_mut(), started) {
            timings.record(stage, started.elapsed());
        }
    }

    fn timer(&self) -> Option<Instant> {
        self.stage_timings.is_some().then(Instant::now)
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

    /// Connect a [`Handoff`]. Both of its stages must already be installed,
    /// as the types it moves data between. Connecting one twice is a no-op.
    pub fn connect(&mut self, handoff: Handoff) -> Result<(), StageError> {
        let observe = self
            .stage::<ObserveStage>(Stage::GenerateObservations)
            .is_some();
        let fusion = self.stage::<FusionStage>(Stage::UpdateFusion).is_some();
        let installed = match handoff {
            Handoff::EngagementToControllers => {
                observe
                    && self
                        .stage::<ControllerStage>(Stage::SampleControllersAndBattery)
                        .is_some()
            }
            Handoff::ObservationsToFusion => observe && fusion,
            Handoff::RangeScansToMapping => {
                observe
                    && self
                        .stage::<crate::mapping::MappingStage>(Stage::UpdateMappingAndLocalization)
                        .is_some()
            }
            Handoff::SafetyEventsToFusion => {
                fusion && self.stage::<SafetyStage>(Stage::SafetyMonitor).is_some()
            }
            Handoff::OverridesFromMission => {
                self.stage::<MissionDriverStage>(Stage::EvaluateMissionTransitions)
                    .is_some()
                    && self
                        .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
                        .is_some()
            }
        };
        if !installed {
            return Err(StageError::Invariant(format!(
                "{handoff:?} needs both of its stages installed first"
            )));
        }
        if !self.handoffs.contains(&handoff) {
            self.handoffs.push(handoff);
        }
        Ok(())
    }

    /// Apply the handoffs that follow `stage`, which has just run and
    /// reported `outcome`.
    fn apply_handoffs(&mut self, stage: Stage, outcome: &StageOutcome, notes: &mut Vec<String>) {
        for index in 0..self.handoffs.len() {
            let handoff = self.handoffs[index];
            if handoff.after() != stage || !handoff.follows(outcome) {
                continue;
            }
            match handoff {
                Handoff::EngagementToControllers => {
                    let Some(accepted) = self
                        .stage::<ObserveStage>(Stage::GenerateObservations)
                        .map(|observe| observe.accepted_by_node_target().clone())
                    else {
                        continue;
                    };
                    if let Some(controllers) =
                        self.stage_mut::<ControllerStage>(Stage::SampleControllersAndBattery)
                    {
                        for node_id in controllers.observe(&accepted) {
                            notes.push(format!(
                                "controller {node_id} engaged; follows from next tick"
                            ));
                        }
                    }
                }
                Handoff::ObservationsToFusion => {
                    let Some(observations) = self
                        .stage::<ObserveStage>(Stage::GenerateObservations)
                        .map(ObserveStage::observations)
                    else {
                        continue;
                    };
                    if let Some(fusion) = self.stage_mut::<FusionStage>(Stage::UpdateFusion) {
                        fusion.submit_observations(observations);
                    }
                }
                Handoff::RangeScansToMapping => {
                    let Some(scans) = self
                        .stage::<ObserveStage>(Stage::GenerateObservations)
                        .map(|observe| observe.range_scans().to_vec())
                    else {
                        continue;
                    };
                    if let Some(mapping) = self.stage_mut::<crate::mapping::MappingStage>(
                        Stage::UpdateMappingAndLocalization,
                    ) {
                        mapping.submit_range_scans(scans);
                    }
                }
                Handoff::SafetyEventsToFusion => {
                    let Some(events) = self
                        .stage::<SafetyStage>(Stage::SafetyMonitor)
                        .map(SafetyStage::safety_events)
                    else {
                        continue;
                    };
                    if let Some(fusion) = self.stage_mut::<FusionStage>(Stage::UpdateFusion) {
                        fusion.submit_safety_events(events);
                    }
                }
                Handoff::OverridesFromMission => {
                    let Some(commands) = self
                        .stage_mut::<MissionDriverStage>(Stage::EvaluateMissionTransitions)
                        .map(MissionDriverStage::take_override_commands)
                    else {
                        continue;
                    };
                    if let Some(overrides) =
                        self.stage_mut::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
                    {
                        for command in commands {
                            overrides.apply(command);
                        }
                    }
                }
            }
        }
    }

    pub fn tick(&self) -> u64 {
        self.clock.tick()
    }

    /// Where the durable event journal stands: set by the recorder after each
    /// durable append (A4), carried by every snapshot and checkpoint, so a
    /// recovery knows how much of the journal a checkpoint already covers.
    pub fn set_journal_cursor(&mut self, cursor: u64) {
        self.journal_cursor = cursor;
    }

    pub fn journal_cursor(&self) -> u64 {
        self.journal_cursor
    }

    /// The run id every command must carry.
    pub fn run_id(&self) -> &str {
        self.config.run_id()
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

    /// The installed handler for `stage`, if it is a `T`.
    ///
    /// How a caller outside the tick reads a stage's state: through the
    /// session that owns it, rather than through a handle cloned out of the
    /// stage (SESSION_OWNERSHIP.md §2).
    pub fn stage<T: StageHandler>(&self, stage: Stage) -> Option<&T> {
        let handler: &dyn std::any::Any = self.handlers.get(&stage)?.as_ref();
        handler.downcast_ref::<T>()
    }

    /// [`Self::stage`], mutably: for configuration between ticks and for
    /// draining what a stage reported.
    pub fn stage_mut<T: StageHandler>(&mut self, stage: Stage) -> Option<&mut T> {
        let handler: &mut dyn std::any::Any = self.handlers.get_mut(&stage)?.as_mut();
        handler.downcast_mut::<T>()
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

    /// Mutable access for the run driver, **between** ticks.
    ///
    /// Platform positions are prescribed by the scenario rather than
    /// integrated by a stage -- the reference evaluates
    /// `scenario.node_states(timestamp_s)` at the top of each loop iteration --
    /// so something outside the pipeline has to write them. Stages reach the
    /// same state through `TickContext::world`.
    pub fn world_mut(&mut self) -> &mut TickWorld {
        &mut self.world
    }

    pub fn latest_snapshot(&self) -> Option<Arc<RuntimeSnapshot>> {
        self.last_snapshot.clone()
    }

    /// Validate a command and queue it. Always answers.
    pub fn apply_command(&mut self, envelope: CommandEnvelope) -> CommandAck {
        // Foreign callers must not read or poison this run's retry history.
        if envelope.run_id != self.config.run_id() {
            return self.rejection(
                envelope.command_id,
                RejectReason::StaleRun {
                    found: envelope.run_id.clone(),
                    expected: self.config.run_id().to_string(),
                },
            );
        }
        if let Some(previous) = self.queue.previous_ack(envelope.command_id) {
            if self
                .queue
                .previous_request(envelope.command_id)
                .is_some_and(|request| !request.same_request(&envelope))
            {
                return self.rejection(envelope.command_id, RejectReason::CommandIdConflict);
            }
            let mut repeated = previous.clone();
            repeated.duplicate = true;
            return repeated;
        }
        if self.queue.is_expired(envelope.command_id) {
            return self.rejection(
                envelope.command_id,
                RejectReason::DedupWindowExpired {
                    command_id: envelope.command_id,
                },
            );
        }
        let ack = self.admit_command(envelope.clone());
        self.queue.remember_request(envelope);
        ack
    }

    /// Latest bounded command outcome, including application or cancellation.
    pub fn command_ack(&self, command_id: u64) -> Option<&CommandAck> {
        self.queue.previous_ack(command_id)
    }

    fn admit_command(&mut self, envelope: CommandEnvelope) -> CommandAck {
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
        if self
            .config
            .steps()
            .is_some_and(|steps| self.tick() >= steps)
            && !matches!(envelope.payload, CommandKind::Reset)
        {
            return self.reject(envelope.command_id, RejectReason::Completed);
        }
        if let CommandKind::SetSpeed { multiplier } = envelope.payload {
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return self.reject(envelope.command_id, RejectReason::InvalidSpeed);
            }
        }
        if let Some(requested) = envelope.requested_effective_tick {
            if requested < self.tick() {
                return self.reject(
                    envelope.command_id,
                    RejectReason::PastEffectiveTick {
                        requested,
                        current: self.tick(),
                    },
                );
            }
            if matches!(envelope.payload, CommandKind::Reset)
                && requested > self.tick()
                && (self.is_faulted() || self.config.steps().is_some_and(|n| self.tick() >= n))
            {
                return self.reject(envelope.command_id, RejectReason::RecoveryMustBeImmediate);
            }
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
            duplicate: false,
            status: AckStatus::Accepted,
            effective_tick: Some(effective_tick),
            epoch: self.epoch,
            revision: self.revision,
        };
        self.queue.remember(ack.clone());
        ack
    }

    fn rejection(&self, command_id: u64, reason: RejectReason) -> CommandAck {
        CommandAck {
            command_id,
            status: AckStatus::Rejected(reason),
            effective_tick: None,
            epoch: self.epoch,
            revision: self.revision,
            duplicate: false,
        }
    }

    fn reject(&mut self, command_id: u64, reason: RejectReason) -> CommandAck {
        let ack = self.rejection(command_id, reason);
        self.queue.remember(ack.clone());
        ack
    }

    fn cancel(&mut self, command_id: u64, reason: RejectReason) {
        let mut ack = self.rejection(command_id, reason);
        ack.status = match ack.status {
            AckStatus::Rejected(reason) => AckStatus::Cancelled(reason),
            _ => unreachable!(),
        };
        self.queue.remember(ack);
    }

    /// Run one tick, or explain why not.
    ///
    /// On success exactly one snapshot is committed. On failure nothing is
    /// published and the run is faulted. The same as [`Self::begin_tick`],
    /// [`Self::run_through`] stage 9 and [`Self::commit_tick`], which a
    /// scheduler uses to run the tick in parts (B2.3).
    pub fn step(&mut self) -> Result<Arc<RuntimeSnapshot>, TickError> {
        let mut open = match self.begin_tick()? {
            TickStart::Open(open) => open,
            TickStart::Committed(snapshot) => return Ok(snapshot),
        };
        self.run_through(&mut open, Stage::EvaluateMissionTransitions)?;
        self.commit_tick(open)
    }

    /// Stage 0 and the boundary checks: the first part of [`Self::step`].
    ///
    /// A Reset applied by stage 0 commits the fresh run at once
    /// ([`TickStart::Committed`]); otherwise the tick is open for
    /// [`Self::run_through`].
    pub fn begin_tick(&mut self) -> Result<TickStart, TickError> {
        let tick = self.clock.tick();
        let epoch = self.epoch;

        // Stage 0 runs before the pause check, and that ordering is load-bearing.
        // §5 says pause is "checked between ticks", and commands are admitted at
        // the same boundary. Checking pause first instead would trap the run: a
        // queued Resume is applied in stage 0, stage 0 only runs inside `step`,
        // and `step` would have already refused -- so nothing could ever
        // un-pause it.
        {
            let mut notes = Vec::new();
            let started = self.timer();
            let applied = self.run_apply_commands(tick, &mut notes);
            self.time_stage(Stage::ApplyCommands, started);
            match applied {
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
        if self.epoch != epoch {
            return Ok(TickStart::Committed(self.commit_timed()));
        }

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

        if self.paused && !self.single_step {
            return Err(TickError::Paused);
        }
        if self.single_step {
            self.single_step = false;
            self.paused = true;
        }
        self.world.begin_tick();
        Ok(TickStart::Open(OpenTick {
            tick,
            epoch,
            next: 1,
        }))
    }

    /// Run the open tick's stages up to and including `last`, in pipeline
    /// order, from wherever the tick has got to. Stages 0 and 10 are
    /// [`Self::begin_tick`]'s and [`Self::commit_tick`]'s.
    pub fn run_through(&mut self, open: &mut OpenTick, last: Stage) -> Result<(), TickError> {
        if matches!(last, Stage::ApplyCommands | Stage::CommitSnapshot) {
            return Err(TickError::OutOfOrder {
                detail: format!("stage {last} is the tick's boundary, not a stage to run"),
            });
        }
        if open.epoch != self.epoch || open.tick != self.clock.tick() {
            return Err(TickError::OutOfOrder {
                detail: "the open tick is not this run's current tick".into(),
            });
        }
        if (last.order() as usize) < open.next {
            return Err(TickError::OutOfOrder {
                detail: format!("stage {last} has already run this tick"),
            });
        }
        let tick = open.tick;
        while open.next <= last.order() as usize {
            let stage = Stage::PIPELINE[open.next];
            let mut notes = Vec::new();
            let started = self.timer();
            // Taken out for its run, so the others can be lent.
            let mut handler = self.handlers.remove(&stage);
            // A stage that sat out boundaries (its lane at its own rate) is
            // given the time since it last ran. With no lag this is the
            // timestep itself, bit for bit.
            let dt_s = match self.stage_lag.remove(&stage) {
                None => self.config.timestep().seconds(),
                Some(lag) => self.config.timestep().seconds() * (lag + 1) as f64,
            };
            let mut context = TickContext {
                tick,
                timestamp_s: self.clock.timestamp_s(),
                dt_s,
                seed: self.config.seed(),
                seed_time_key: self.clock.seed_time_key(),
                world: &mut self.world,
                // The capability, handed out by stage and nowhere else.
                truth: stage.may_read_truth().then_some(self.truth.as_slice()),
                stages: StageViews {
                    handlers: &self.handlers,
                    truth_visible: stage.may_read_truth(),
                },
                notes: &mut notes,
            };
            let outcome = match handler.as_mut() {
                Some(handler) => handler.run(&mut context),
                None => UnportedStage.run(&mut context),
            };
            if let Some(handler) = handler {
                self.handlers.insert(stage, handler);
            }
            if let Ok(ran) = &outcome {
                self.apply_handoffs(stage, ran, &mut notes);
            }
            self.time_stage(stage, started);
            open.next += 1;

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
        Ok(())
    }

    /// Record the open tick's stages up to and including `last` as not due
    /// this boundary (B2.4: a lane at its own rate), without running them.
    /// Each handler clears its per-tick output ([`StageHandler::not_due`]);
    /// one that cannot faults the run, as a failed stage does. The trace
    /// shows every one as skipped, with `reason`: a stage is never left out
    /// silently.
    pub fn skip_through(
        &mut self,
        open: &mut OpenTick,
        last: Stage,
        reason: &str,
    ) -> Result<(), TickError> {
        if matches!(last, Stage::ApplyCommands | Stage::CommitSnapshot) {
            return Err(TickError::OutOfOrder {
                detail: format!("stage {last} is the tick's boundary, not a stage to skip"),
            });
        }
        if open.epoch != self.epoch || open.tick != self.clock.tick() {
            return Err(TickError::OutOfOrder {
                detail: "the open tick is not this run's current tick".into(),
            });
        }
        if (last.order() as usize) < open.next {
            return Err(TickError::OutOfOrder {
                detail: format!("stage {last} has already run this tick"),
            });
        }
        let tick = open.tick;
        while open.next <= last.order() as usize {
            let stage = Stage::PIPELINE[open.next];
            open.next += 1;
            let declined = match self.handlers.get_mut(&stage) {
                Some(handler) => handler.not_due().err(),
                None => None,
            };
            if let Some(error) = declined {
                self.trace.record(StageRecord {
                    tick,
                    order: stage.order(),
                    stage,
                    outcome: StageOutcome::Failed {
                        detail: error.to_string(),
                    },
                    notes: Vec::new(),
                });
                self.state = RunState::Faulted {
                    at_tick: tick,
                    stage,
                };
                return Err(TickError::Stage { tick, stage, error });
            }
            *self.stage_lag.entry(stage).or_insert(0) += 1;
            self.trace.record(StageRecord {
                tick,
                order: stage.order(),
                stage,
                outcome: StageOutcome::Skipped {
                    reason: reason.to_string(),
                },
                notes: Vec::new(),
            });
        }
        Ok(())
    }

    /// Stage 10: commit the open tick. Every stage must have run.
    pub fn commit_tick(&mut self, open: OpenTick) -> Result<Arc<RuntimeSnapshot>, TickError> {
        let commit = Stage::CommitSnapshot.order() as usize;
        if open.next != commit || open.epoch != self.epoch || open.tick != self.clock.tick() {
            return Err(TickError::OutOfOrder {
                detail: format!("commit with the tick at stage {} of {commit}", open.next),
            });
        }
        self.trace.record(StageRecord {
            tick: open.tick,
            order: Stage::CommitSnapshot.order(),
            stage: Stage::CommitSnapshot,
            outcome: StageOutcome::Ran,
            notes: Vec::new(),
        });
        // Commit. Only now does the tick advance and a revision become visible.
        self.clock.advance();
        Ok(self.commit_timed())
    }

    /// [`Self::commit_current`], timed as stage 10: the freeze is its work.
    fn commit_timed(&mut self) -> Arc<RuntimeSnapshot> {
        let started = self.timer();
        let snapshot = self.commit_current();
        self.time_stage(Stage::CommitSnapshot, started);
        snapshot
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
            let reason = if envelope.expected_epoch != self.epoch {
                Some(RejectReason::StaleEpoch {
                    found: envelope.expected_epoch,
                    expected: self.epoch,
                })
            } else if envelope
                .expected_revision
                .is_some_and(|revision| revision != self.revision)
            {
                Some(RejectReason::StaleRevision {
                    found: envelope.expected_revision.expect("checked"),
                    expected: self.revision,
                })
            } else if self.is_faulted() && !matches!(envelope.payload, CommandKind::Reset) {
                Some(RejectReason::Faulted)
            } else if self.config.steps().is_some_and(|n| self.tick() >= n)
                && !matches!(envelope.payload, CommandKind::Reset)
            {
                Some(RejectReason::Completed)
            } else {
                None
            };
            if let Some(reason) = reason {
                notes.push(format!(
                    "cancelled command {}: {reason:?}",
                    envelope.command_id
                ));
                self.cancel(envelope.command_id, reason);
                continue;
            }
            // §4: "Log accepted/rejected command and effective tick."
            notes.push(format!(
                "applied command {} ({:?}) effective at tick {tick}",
                envelope.command_id, envelope.payload
            ));
            match envelope.payload {
                CommandKind::Pause => {
                    self.paused = true;
                    self.single_step = false;
                }
                CommandKind::Resume => {
                    self.paused = false;
                    self.single_step = false;
                }
                CommandKind::SingleStep => self.single_step = true,
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
            self.queue.remember(CommandAck {
                command_id: envelope.command_id,
                status: AckStatus::Applied,
                effective_tick: Some(tick),
                epoch: self.epoch,
                revision: self.revision,
                duplicate: false,
            });
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
        self.stage_lag.clear();
        for pending in self.queue.drain_due(u64::MAX) {
            self.cancel(
                pending.command_id,
                RejectReason::StaleEpoch {
                    found: pending.expected_epoch,
                    expected: self.epoch,
                },
            );
        }
        self.trace.clear();
        self.last_snapshot = None;
        self.paused = false;
        self.single_step = false;
        self.speed = 1.0;
        self.belief_revision = 0;
        self.journal_cursor = 0;
        // A reset is a new epoch, so the previous epoch's platforms are not
        // this run's. Truth goes with them: the caller reloads both, and
        // leaving stale truth behind would let a fresh run sense the old
        // world for one tick.
        self.world.clear();
        self.truth.clear();
        // Handlers are part of the run, not part of the harness around it.
        for handler in self.handlers.values_mut() {
            handler.reset();
        }
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
                format_version: if self.stage_lag.is_empty() {
                    CheckpointHeader::FORMAT_VERSION
                } else {
                    CheckpointHeader::FORMAT_VERSION_STAGE_LAG
                },
                runtime_algorithm_version: CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
                seed_version: CheckpointHeader::SEED_VERSION,
                run_id: self.config.run_id().to_string(),
                epoch: self.epoch,
                tick: self.clock.tick(),
                config_hash: self.config.content_hash(),
                scenario_hash: self.scenario_hash,
                world_content_hash: self.world_revision,
                belief_revision: self.belief_revision,
                journal_cursor: self.journal_cursor,
            },
            tick: self.clock.tick(),
            seed: self.config.seed(),
            pending_commands: self.queue.pending().iter().cloned().collect(),
            command_history: self.queue.history(),
            revision: self.revision,
            paused: self.paused,
            speed: self.speed,
            stage_states: self
                .handlers
                .iter()
                .filter_map(|(stage, handler)| {
                    handler
                        .checkpoint_state()
                        .map(|state| (stage.to_string(), state))
                })
                .collect(),
            stage_lag: self
                .stage_lag
                .iter()
                .map(|(stage, lag)| (stage.to_string(), *lag))
                .collect(),
        })
    }

    /// Resume from a checkpoint, or refuse with a reason.
    ///
    /// Every compatibility field is checked *before* any state is touched, so a
    /// refused restore leaves the run exactly as it was — including a faulted
    /// run, which stays faulted rather than being half-recovered.
    pub fn restore(&mut self, checkpoint: &SimulationCheckpoint) -> Result<(), RestoreError> {
        checkpoint.verify(
            self.config.run_id(),
            self.config.content_hash(),
            self.scenario_hash,
        )?;

        if checkpoint.seed != self.config.seed() {
            return Err(RestoreError::InvalidState {
                detail: "seed disagrees with configured seed".into(),
            });
        }
        if self
            .config
            .steps()
            .is_some_and(|steps| checkpoint.tick > steps)
        {
            return Err(RestoreError::InvalidState {
                detail: "tick exceeds configured run length".into(),
            });
        }
        // Every stage named in the checkpoint must have a handler here, and
        // every handler must accept its own state, BEFORE anything is
        // mutated. A restore that failed halfway would leave a run that is
        // neither where it was nor where it was going.
        for (name, state) in &checkpoint.stage_states {
            let handler = self
                .handlers
                .iter()
                .find(|(stage, _)| stage.to_string() == *name)
                .map(|(_, handler)| handler)
                .ok_or_else(|| RestoreError::UnknownStageState {
                    stage: name.clone(),
                })?;
            handler
                .check_state(state)
                .map_err(|error| RestoreError::StageState {
                    stage: name.clone(),
                    detail: error.to_string(),
                })?;
        }
        let mut stage_lag = BTreeMap::new();
        for (name, lag) in &checkpoint.stage_lag {
            let stage = Stage::PIPELINE
                .into_iter()
                .find(|stage| stage.to_string() == *name)
                .ok_or_else(|| RestoreError::InvalidState {
                    detail: format!("lag recorded for an unknown stage {name}"),
                })?;
            // A stage cannot have sat out more boundaries than the run has had.
            if *lag > checkpoint.tick {
                return Err(RestoreError::InvalidState {
                    detail: format!(
                        "stage {name} is recorded as {lag} boundaries behind at tick {}",
                        checkpoint.tick
                    ),
                });
            }
            stage_lag.insert(stage, *lag);
        }
        let restored_queue = CommandQueue::from_checkpoint(
            &checkpoint.pending_commands,
            &checkpoint.command_history,
        );
        self.clock.restore_tick(checkpoint.tick);
        self.epoch = checkpoint.header.epoch;
        self.belief_revision = checkpoint.header.belief_revision;
        self.journal_cursor = checkpoint.header.journal_cursor;
        self.world_revision = checkpoint.header.world_content_hash;
        self.state = RunState::Ready;
        self.queue = restored_queue;
        self.revision = checkpoint.revision;
        self.trace.clear();
        self.last_snapshot = None;
        self.paused = checkpoint.paused;
        self.speed = checkpoint.speed;
        self.single_step = false;
        self.stage_lag = stage_lag;
        for (stage, handler) in self.handlers.iter_mut() {
            match checkpoint.stage_states.get(&stage.to_string()) {
                // `check_state` accepted this value above, so a refusal now
                // is the handler contradicting itself, not a bad checkpoint.
                Some(state) => {
                    handler
                        .restore_state(state)
                        .map_err(|error| RestoreError::InvalidState {
                            detail: format!(
                                "stage {stage} accepted its state and then refused it: {error}"
                            ),
                        })?
                }
                // A handler with no state in the checkpoint returns to fresh
                // rather than keeping this object's: the checkpoint is the
                // authority on what the run knows.
                None => handler.reset(),
            }
        }
        Ok(())
    }
}
