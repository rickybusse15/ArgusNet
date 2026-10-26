//! The simulation worker (D1.1, P5.0): one thread owns an assembled run.
//!
//! A viewer (or any other front end) never runs a tick itself. It talks to a
//! [`SimWorker`] over three channels with different contracts:
//!
//! - **Commands in, bounded.** [`SimWorker::submit`] carries the existing
//!   [`CommandEnvelope`]. A full channel answers at once with
//!   [`RejectReason::Busy`]: never a silent drop (ARCHITECTURE_AND_PROTOCOLS
//!   §5).
//! - **Acknowledgements out, lossless and ordered.** Every submitted command
//!   gets its admission answer, then, if admitted, the answer for when it took
//!   effect ([`AckStatus::Applied`]) or was cancelled. A front end shows a
//!   control as pending until that second answer, and never infers success
//!   from motion.
//! - **Frames out, lossless and ordered.** Every committed tick's replay
//!   frame, so a slow reader falls behind but never loses a tick. Next to it,
//!   the latest snapshot is published **latest-wins**, for a reader that only
//!   wants to draw the present.
//!
//! The worker starts **held**: the run is paused through the simulation's own
//! Pause command before the first tick, so Run is an ordinary Resume and every
//! control goes through the same admission, dedup and acknowledgement path.
//!
//! Reset is the one command the worker applies itself. The simulation core's
//! Reset starts a new epoch but cannot reload the fleet or clear the replay,
//! which belong to the [`AssembledRun`] around it; [`AssembledRun::reset`] does
//! all of it. After a reset the run is held again, so a reset never sets a run
//! going on its own.
//!
//! Seek (P5.2) is the other. The worker keeps a bounded history of
//! checkpoints, one every [`CHECKPOINT_EVERY_TICKS`] ticks, at most
//! [`RETAINED_CHECKPOINTS`] of them. A seek to an earlier tick restores the
//! nearest checkpoint at or before it, which branches the epoch, and runs
//! forward to the tick without publishing those frames: a restore equals the
//! uninterrupted run (A2), so they are the frames the front end already has.
//! The run is then held at the tick, and the status says where the new epoch
//! branched ([`WorkerStatus::branched_at`]). A seek forward, or to a tick
//! older than the history keeps, is refused with its reason.
//!
//! A run with lanes at their own rates (B2.4, B2.5, `scheduled_v1`) is
//! stepped through a [`PhasedLaneExecutor`] at those rates
//! ([`SimWorker::spawn_with_rates`]); every other run is stepped as the CLI's
//! legacy run is, one tick at a time.
//!
//! World preparation runs on the worker thread too. Closing the worker while it
//! loads returns at once; the half-built world is discarded when the build
//! finishes, and never published.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::command::{AckStatus, CommandAck, CommandEnvelope, CommandKind, RejectReason};
use crate::replay::ReplayFrame;
use crate::scenario::{AssembledRun, RunError};
use crate::scheduler::{ExecutorError, LaneRates, PhasedLaneExecutor};
use crate::simulation::TickError;
use crate::snapshot::RuntimeSnapshot;

/// Commands the worker holds before answering Busy.
pub const DEFAULT_COMMAND_CAPACITY: usize = 64;

/// A checkpoint is kept every this many ticks, for seeking back (P5.2).
pub const CHECKPOINT_EVERY_TICKS: u64 = 25;
/// At most this many are kept; the oldest goes first. With a 1 s tick the
/// history reaches back 1000 s.
pub const RETAINED_CHECKPOINTS: usize = 40;

/// How the worker paces ticks against the wall clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pacing {
    /// One tick per `dt / speed` of wall time: the operator's view.
    RealTime,
    /// As fast as the run computes: batch runs and tests.
    Unpaced,
}

/// Where the worker is.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerPhase {
    /// Preparing the world.
    Loading,
    /// Paused, waiting for a command.
    Paused,
    Running,
    /// Every tick has run; only Reset is accepted.
    Completed,
    /// A tick failed; only Reset is accepted.
    Faulted(String),
    /// The world could not be prepared. Nothing was published.
    Failed(String),
    /// Closed.
    Stopped,
}

/// What a front end reads to draw its controls.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerStatus {
    pub phase: WorkerPhase,
    /// Commands must carry this; `None` until the world is ready.
    pub run_id: Option<String>,
    pub epoch: u64,
    /// The next tick to run.
    pub tick: u64,
    pub speed: f64,
    pub dt_s: f64,
    /// The oldest tick a seek can reach, when the history holds any.
    pub retained_from: Option<u64>,
    /// When this epoch began with a seek: the tick it branched at. Every
    /// frame before it is the previous epoch's, unchanged.
    pub branched_at: Option<u64>,
}

/// One committed tick, in order.
#[derive(Debug, Clone)]
pub struct WorkerFrame {
    pub epoch: u64,
    pub tick: u64,
    pub frame: ReplayFrame,
    /// The mission's decisions this tick (F3), as the journal records them:
    /// sent with the frame on the lossless channel, so a front end that
    /// skips drawing a frame still has them.
    pub decisions: Vec<crate::mission_driver::Decision>,
}

enum Control {
    Command(CommandEnvelope),
    Stop,
}

/// The lanes executor that steps `run` with its lanes at `rates` (B2.4,
/// B2.5), from the run's current tick. Refused when the run's timestep is not
/// a whole number of nanoseconds, or a rate is invalid: a front end calls
/// this before it opens, so such a run never starts.
pub fn scheduled_executor(
    run: &AssembledRun,
    rates: &LaneRates,
) -> Result<PhasedLaneExecutor, String> {
    PhasedLaneExecutor::new::<RunError>(run.dt_s(), run.sim.tick())
        .and_then(|executor| rates.apply(executor).map_err(ExecutorError::Time))
        .map_err(|error| error.to_string())
}

struct Shared {
    status: Mutex<WorkerStatus>,
    latest: Mutex<Option<Arc<RuntimeSnapshot>>>,
}

/// The handle a front end holds. Dropping it stops the worker.
pub struct SimWorker {
    controls: SyncSender<Control>,
    capacity: usize,
    // Behind locks so the handle is `Sync`, as a front end's shared
    // resources must be; only the handle's owner ever drains them.
    acks: Mutex<Receiver<CommandAck>>,
    frames: Mutex<Receiver<WorkerFrame>>,
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl SimWorker {
    /// Start a worker that builds its run with `build`, on the worker thread.
    pub fn spawn<F>(build: F, pacing: Pacing) -> Self
    where
        F: FnOnce() -> Result<AssembledRun, String> + Send + 'static,
    {
        Self::spawn_with_capacity(build, pacing, DEFAULT_COMMAND_CAPACITY)
    }

    pub fn spawn_with_capacity<F>(build: F, pacing: Pacing, capacity: usize) -> Self
    where
        F: FnOnce() -> Result<AssembledRun, String> + Send + 'static,
    {
        Self::spawn_inner(build, pacing, capacity, LaneRates::default())
    }

    /// [`Self::spawn`], with lanes at their own rates (B2.4, B2.5): the run
    /// is stepped through a [`PhasedLaneExecutor`] at those rates. `build`
    /// must declare the same rates on the run, as
    /// [`crate::run_options::RunOptions::apply`] does. Empty rates are
    /// [`Self::spawn`].
    pub fn spawn_with_rates<F>(build: F, pacing: Pacing, rates: LaneRates) -> Self
    where
        F: FnOnce() -> Result<AssembledRun, String> + Send + 'static,
    {
        Self::spawn_inner(build, pacing, DEFAULT_COMMAND_CAPACITY, rates)
    }

    fn spawn_inner<F>(build: F, pacing: Pacing, capacity: usize, rates: LaneRates) -> Self
    where
        F: FnOnce() -> Result<AssembledRun, String> + Send + 'static,
    {
        let capacity = capacity.max(1);
        let (controls, control_rx) = mpsc::sync_channel(capacity);
        let (ack_tx, acks) = mpsc::channel();
        let (frame_tx, frames) = mpsc::channel();
        let shared = Arc::new(Shared {
            status: Mutex::new(WorkerStatus {
                phase: WorkerPhase::Loading,
                run_id: None,
                epoch: 0,
                tick: 0,
                speed: 1.0,
                dt_s: 0.0,
                retained_from: None,
                branched_at: None,
            }),
            latest: Mutex::new(None),
        });
        let thread_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("argusnet-sim-worker".into())
            .spawn(move || {
                let built = build().and_then(|run| {
                    let executor = (!rates.is_empty())
                        .then(|| scheduled_executor(&run, &rates))
                        .transpose()?;
                    Ok((run, executor))
                });
                let (run, executor) = match built {
                    Ok(built) => built,
                    Err(error) => {
                        thread_shared.status.lock().expect("status").phase =
                            WorkerPhase::Failed(error);
                        return;
                    }
                };
                // Closed while loading: discard the world unpublished.
                if matches!(control_rx.try_recv(), Ok(Control::Stop))
                    || thread_shared.status.lock().expect("status").phase == WorkerPhase::Stopped
                {
                    return;
                }
                Loop {
                    run,
                    executor,
                    rates,
                    history: std::collections::VecDeque::new(),
                    branched_at: None,
                    seek_acks: BTreeMap::new(),
                    pacing,
                    controls: control_rx,
                    acks: ack_tx,
                    frames: frame_tx,
                    shared: thread_shared,
                    pending: Vec::new(),
                    reset_acks: BTreeMap::new(),
                    internal_id: u64::MAX,
                    next_due: Instant::now(),
                }
                .run();
            })
            .expect("spawn the simulation worker");
        Self {
            controls,
            capacity,
            acks: Mutex::new(acks),
            frames: Mutex::new(frames),
            shared,
            handle: Some(handle),
        }
    }

    /// Hand a command to the worker. `Err` is the answer when it could not
    /// even be queued (Busy, or the worker has stopped); otherwise the answers
    /// arrive through [`Self::acks`].
    pub fn submit(&self, envelope: CommandEnvelope) -> Result<(), CommandAck> {
        let command_id = envelope.command_id;
        let refuse = |reason| {
            let status = self.status();
            CommandAck {
                command_id,
                status: AckStatus::Rejected(reason),
                effective_tick: None,
                epoch: status.epoch,
                revision: 0,
                duplicate: false,
            }
        };
        match self.controls.try_send(Control::Command(envelope)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(refuse(RejectReason::Busy {
                capacity: self.capacity,
            })),
            Err(TrySendError::Disconnected(_)) => Err(refuse(RejectReason::Unsupported {
                name: "worker stopped".to_string(),
            })),
        }
    }

    /// Every acknowledgement since the last call, in order. Lossless.
    pub fn acks(&self) -> Vec<CommandAck> {
        self.acks.lock().expect("acks").try_iter().collect()
    }

    /// Every committed tick's frame since the last call, in order. Lossless.
    pub fn frames(&self) -> Vec<WorkerFrame> {
        self.frames.lock().expect("frames").try_iter().collect()
    }

    /// The newest snapshot: latest-wins.
    pub fn latest_snapshot(&self) -> Option<Arc<RuntimeSnapshot>> {
        self.shared.latest.lock().expect("latest").clone()
    }

    pub fn status(&self) -> WorkerStatus {
        self.shared.status.lock().expect("status").clone()
    }

    /// Stop the worker. Returns at once while the world is still loading; the
    /// thread finishes the build, discards it, and exits on its own.
    pub fn stop(&mut self) {
        let loading = {
            let mut status = self.shared.status.lock().expect("status");
            let loading = status.phase == WorkerPhase::Loading;
            status.phase = WorkerPhase::Stopped;
            loading
        };
        // The channel may be full; the status flag above is the fallback the
        // loop checks every turn.
        let _ = self.controls.try_send(Control::Stop);
        if let Some(handle) = self.handle.take() {
            if !loading {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for SimWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Loop {
    run: AssembledRun,
    /// The lanes executor of a run with a lane at its own rate; `None` steps
    /// the run one legacy tick at a time.
    executor: Option<PhasedLaneExecutor>,
    /// The lane rates the executor was built with, to rebuild it after a
    /// seek restores an earlier tick.
    rates: LaneRates,
    /// Checkpoints for seeking back, oldest first, each with the next tick
    /// it would run (P5.2).
    history: std::collections::VecDeque<(u64, crate::snapshot::SimulationCheckpoint)>,
    /// Where the current epoch branched, if a seek began it.
    branched_at: Option<u64>,
    /// Seeks the worker applied, for answering a retry with the original.
    seek_acks: BTreeMap<u64, CommandAck>,
    pacing: Pacing,
    controls: Receiver<Control>,
    acks: Sender<CommandAck>,
    frames: Sender<WorkerFrame>,
    shared: Arc<Shared>,
    /// Admitted commands still waiting to take effect.
    pending: Vec<u64>,
    /// Resets the worker applied, for answering a retry with the original.
    reset_acks: BTreeMap<u64, CommandAck>,
    /// Ids for the worker's own Pause commands, counted down from the top so
    /// they cannot collide with a front end's.
    internal_id: u64,
    next_due: Instant,
}

impl Loop {
    fn run(mut self) {
        self.hold();
        // The run as it starts: a seek can always go back to tick 0 while
        // the history reaches it.
        self.retain_checkpoint();
        loop {
            if self.shared.status.lock().expect("status").phase == WorkerPhase::Stopped {
                return;
            }
            let timeout = match (self.phase(), self.pacing) {
                (WorkerPhase::Running, Pacing::Unpaced) => Duration::ZERO,
                (WorkerPhase::Running, Pacing::RealTime) => {
                    self.next_due.saturating_duration_since(Instant::now())
                }
                // Idle: wake now and then to notice a stop flagged while the
                // control channel was full.
                _ => Duration::from_millis(100),
            };
            let mut handled = false;
            match self.controls.recv_timeout(timeout) {
                Ok(Control::Stop) | Err(RecvTimeoutError::Disconnected) => return,
                Ok(Control::Command(envelope)) => {
                    self.command(envelope);
                    handled = true;
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            // Everything else already queued, before the next boundary.
            while let Ok(control) = self.controls.try_recv() {
                match control {
                    Control::Stop => return,
                    Control::Command(envelope) => {
                        self.command(envelope);
                        handled = true;
                    }
                }
            }
            match self.phase() {
                // A paused run applies commands only at a boundary, so step
                // it: stage 0 applies them and, unless one resumed or
                // single-stepped, the tick stops there.
                WorkerPhase::Paused if handled => self.advance(),
                WorkerPhase::Running
                    if self.pacing == Pacing::Unpaced || Instant::now() >= self.next_due =>
                {
                    self.advance()
                }
                _ => {}
            }
        }
    }

    /// One boundary of the run: through the lanes executor when the run has
    /// one, so a lane at its own rate sits out the boundaries it is not due.
    fn step(&mut self) -> Result<Arc<RuntimeSnapshot>, RunError> {
        match self.executor.as_mut() {
            None => self.run.step(),
            Some(executor) => executor.step(&mut self.run).map_err(|error| match error {
                ExecutorError::Tick(error) => error,
                other => RunError::Invariant(other.to_string()),
            }),
        }
    }

    fn phase(&self) -> WorkerPhase {
        self.shared.status.lock().expect("status").phase.clone()
    }

    fn set_phase(&self, phase: WorkerPhase) {
        let mut status = self.shared.status.lock().expect("status");
        if status.phase == WorkerPhase::Stopped {
            return;
        }
        status.phase = phase;
        status.run_id = Some(self.run.run_id().to_string());
        status.epoch = self.run.sim.epoch();
        status.tick = self.run.sim.tick();
        status.speed = self.run.sim.speed();
        status.dt_s = self.run.dt_s();
        status.retained_from = self.history.front().map(|(tick, _)| *tick);
        status.branched_at = self.branched_at;
    }

    /// Keep a checkpoint of the run as it stands, when its next tick is on
    /// the history's interval, dropping the oldest beyond the bound.
    fn retain_checkpoint(&mut self) {
        let tick = self.run.sim.tick();
        if !tick.is_multiple_of(CHECKPOINT_EVERY_TICKS)
            || self.history.back().is_some_and(|(last, _)| *last == tick)
        {
            return;
        }
        if let Ok(checkpoint) = self.run.sim.checkpoint() {
            self.history.push_back((tick, checkpoint));
            while self.history.len() > RETAINED_CHECKPOINTS {
                self.history.pop_front();
            }
        }
    }

    /// Whether `envelope` is addressed to this run and its current epoch.
    fn stale(&self, envelope: &CommandEnvelope) -> Option<RejectReason> {
        let epoch = self.run.sim.epoch();
        if envelope.run_id != self.run.run_id() {
            Some(RejectReason::StaleRun {
                found: envelope.run_id.clone(),
                expected: self.run.run_id().to_string(),
            })
        } else if envelope.expected_epoch != epoch {
            Some(RejectReason::StaleEpoch {
                found: envelope.expected_epoch,
                expected: epoch,
            })
        } else {
            None
        }
    }

    /// Seek back to `target`, the next tick to run (P5.2). See the module
    /// docs.
    fn seek(&mut self, envelope: CommandEnvelope, target: u64) -> CommandAck {
        if let Some(previous) = self.seek_acks.get(&envelope.command_id) {
            let mut repeated = previous.clone();
            repeated.duplicate = true;
            return repeated;
        }
        let refuse = |this: &Self, reason| CommandAck {
            command_id: envelope.command_id,
            status: AckStatus::Rejected(reason),
            effective_tick: None,
            epoch: this.run.sim.epoch(),
            revision: this.run.sim.revision(),
            duplicate: false,
        };
        let from = self
            .history
            .iter()
            .rev()
            .find(|(tick, _)| *tick <= target)
            .map(|(tick, checkpoint)| (*tick, checkpoint.clone()));
        let ack = if let Some(reason) = self.stale(&envelope) {
            refuse(self, reason)
        } else if target > self.run.sim.tick() {
            refuse(
                self,
                RejectReason::Unsupported {
                    name: "seek forward: run or step the run instead".to_string(),
                },
            )
        } else if let Some((from_tick, checkpoint)) = from {
            match self.branch(from_tick, &checkpoint, target) {
                Ok(()) => CommandAck {
                    command_id: envelope.command_id,
                    status: AckStatus::Applied,
                    effective_tick: Some(target),
                    epoch: self.run.sim.epoch(),
                    revision: self.run.sim.revision(),
                    duplicate: false,
                },
                Err(error) => {
                    self.set_phase(WorkerPhase::Faulted(error));
                    refuse(self, RejectReason::Faulted)
                }
            }
        } else {
            refuse(self, RejectReason::RewindUnavailable { tick: target })
        };
        self.seek_acks.insert(envelope.command_id, ack.clone());
        ack
    }

    /// Restore the checkpoint taken at `from_tick`, run forward to `target`
    /// unpublished, and hold there.
    fn branch(
        &mut self,
        from_tick: u64,
        checkpoint: &crate::snapshot::SimulationCheckpoint,
        target: u64,
    ) -> Result<(), String> {
        let (left_epoch, left_revision) = (self.run.sim.epoch(), self.run.sim.revision());
        self.run
            .restore(checkpoint)
            .map_err(|error| error.to_string())?;
        // A branch of the history it left, not a resume of it.
        self.run.sim.branch_from(left_epoch, left_revision);
        if !self.rates.is_empty() {
            self.executor = Some(scheduled_executor(&self.run, &self.rates)?);
        }
        // Commands still pending on the epoch left are cancelled with it:
        // the restore replaced the history their answers would come from,
        // so they are answered here, not left pending forever.
        let epoch = self.run.sim.epoch();
        for command_id in std::mem::take(&mut self.pending) {
            let _ = self.acks.send(CommandAck {
                command_id,
                status: AckStatus::Cancelled(RejectReason::StaleEpoch {
                    found: left_epoch,
                    expected: epoch,
                }),
                effective_tick: None,
                epoch,
                revision: self.run.sim.revision(),
                duplicate: false,
            });
        }
        // The checkpoint may have been taken paused (the first one is):
        // run forward regardless.
        if self.run.sim.is_paused() {
            let id = self.internal_id;
            self.internal_id -= 1;
            let resume = CommandEnvelope::new(
                id,
                self.run.run_id(),
                self.run.sim.epoch(),
                CommandKind::Resume,
            );
            let _ = self.run.sim.apply_command(resume);
        }
        let mut tick = from_tick;
        while tick < target {
            match self.step() {
                Ok(_) => tick = self.run.sim.tick(),
                // The resume applies at the first boundary.
                Err(RunError::Tick(TickError::Paused)) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        // Later history belonged to the abandoned branch.
        self.history.retain(|(kept, _)| *kept <= target);
        self.branched_at = Some(target);
        *self.shared.latest.lock().expect("latest") = None;
        self.hold();
        Ok(())
    }

    /// Pause the run through its own command path, before any tick runs.
    fn hold(&mut self) {
        let id = self.internal_id;
        self.internal_id -= 1;
        let pause = CommandEnvelope::new(
            id,
            self.run.run_id(),
            self.run.sim.epoch(),
            CommandKind::Pause,
        );
        let _ = self.run.sim.apply_command(pause);
        // Stage 0 applies it; the tick then stops at the pause.
        match self.step() {
            Err(RunError::Tick(TickError::Paused)) | Ok(_) => {}
            Err(error) => {
                self.set_phase(WorkerPhase::Faulted(error.to_string()));
                return;
            }
        }
        self.set_phase(WorkerPhase::Paused);
    }

    fn command(&mut self, envelope: CommandEnvelope) {
        if matches!(envelope.payload, CommandKind::Reset) {
            let ack = self.reset(envelope);
            let _ = self.acks.send(ack);
            return;
        }
        if let CommandKind::Seek { tick } = envelope.payload {
            let ack = self.seek(envelope, tick);
            let _ = self.acks.send(ack);
            return;
        }
        let ack = self.run.sim.apply_command(envelope);
        if ack.is_accepted() && !ack.duplicate {
            self.pending.push(ack.command_id);
        }
        let _ = self.acks.send(ack);
    }

    fn reset(&mut self, envelope: CommandEnvelope) -> CommandAck {
        if let Some(previous) = self.reset_acks.get(&envelope.command_id) {
            let mut repeated = previous.clone();
            repeated.duplicate = true;
            return repeated;
        }
        let epoch = self.run.sim.epoch();
        let ack = match self.stale(&envelope) {
            Some(reason) => CommandAck {
                command_id: envelope.command_id,
                status: AckStatus::Rejected(reason),
                effective_tick: None,
                epoch,
                revision: self.run.sim.revision(),
                duplicate: false,
            },
            None => {
                // Commands waiting on the old epoch are cancelled by the reset.
                self.run.reset();
                if let Some(executor) = self.executor.as_mut() {
                    executor.reset();
                }
                self.history.clear();
                self.branched_at = None;
                self.flush_pending();
                *self.shared.latest.lock().expect("latest") = None;
                self.hold();
                self.retain_checkpoint();
                CommandAck {
                    command_id: envelope.command_id,
                    status: AckStatus::Applied,
                    effective_tick: Some(0),
                    epoch: self.run.sim.epoch(),
                    revision: self.run.sim.revision(),
                    duplicate: false,
                }
            }
        };
        self.reset_acks.insert(envelope.command_id, ack.clone());
        ack
    }

    /// One boundary: apply what is queued, and run a tick if the run is not
    /// paused.
    fn advance(&mut self) {
        let epoch = self.run.sim.epoch();
        match self.step() {
            Ok(snapshot) => {
                if self.run.sim.epoch() == epoch {
                    if let Some(frame) = self.run.last_frame() {
                        let _ = self.frames.send(WorkerFrame {
                            epoch,
                            tick: self.run.sim.tick().saturating_sub(1),
                            frame: frame.clone(),
                            decisions: self
                                .run
                                .driver()
                                .map(|driver| driver.decisions().to_vec())
                                .unwrap_or_default(),
                        });
                    }
                }
                *self.shared.latest.lock().expect("latest") = Some(snapshot);
                self.retain_checkpoint();
                let step = Duration::from_secs_f64(self.run.dt_s() / self.run.sim.speed());
                self.next_due += step;
                // Far behind (a slow tick, a long pause): pace from now
                // rather than racing to catch up.
                let now = Instant::now();
                if self.next_due + Duration::from_millis(500) < now {
                    self.next_due = now + step;
                }
                let phase = if self.run.sim.is_paused() {
                    WorkerPhase::Paused
                } else {
                    WorkerPhase::Running
                };
                self.flush_pending();
                self.set_phase(phase);
            }
            Err(RunError::Tick(TickError::Paused)) => {
                self.flush_pending();
                if !self.run.sim.is_paused() {
                    self.next_due = Instant::now();
                }
                let phase = if self.run.sim.is_paused() {
                    WorkerPhase::Paused
                } else {
                    WorkerPhase::Running
                };
                self.set_phase(phase);
            }
            Err(RunError::Tick(TickError::Completed { .. })) => {
                self.flush_pending();
                self.set_phase(WorkerPhase::Completed);
            }
            Err(error) => {
                self.flush_pending();
                self.set_phase(WorkerPhase::Faulted(error.to_string()));
            }
        }
        // A Resume applied at this boundary starts the pacing clock now.
        if self.phase() == WorkerPhase::Running
            && self.next_due + Duration::from_secs(1) < Instant::now()
        {
            self.next_due = Instant::now();
        }
    }

    /// Send the outcome of every admitted command that has now taken effect
    /// or been cancelled.
    fn flush_pending(&mut self) {
        let sim = &self.run.sim;
        let acks = &self.acks;
        self.pending.retain(|id| match sim.command_ack(*id) {
            Some(ack) if !ack.is_accepted() => {
                let _ = acks.send(ack.clone());
                false
            }
            _ => true,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{assemble, ScenarioBundle};
    use std::sync::atomic::{AtomicBool, Ordering};

    fn bundle() -> ScenarioBundle {
        let path = format!(
            "{}/../../tests/fixtures/scenario_bundle_tt.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).expect("fixture");
        ScenarioBundle::from_json(&text).expect("bundle parses")
    }

    fn worker(pacing: Pacing) -> (SimWorker, ScenarioBundle) {
        let bundle = bundle();
        let build = bundle.clone();
        (
            SimWorker::spawn(move || assemble(&build).map_err(|e| e.to_string()), pacing),
            bundle,
        )
    }

    fn wait_for(worker: &SimWorker, what: impl Fn(&WorkerStatus) -> bool) -> WorkerStatus {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let status = worker.status();
            if what(&status) {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "timed out; last status {status:?}"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn command(worker: &SimWorker, id: u64, payload: CommandKind) {
        let status = wait_for(worker, |s| s.run_id.is_some());
        worker
            .submit(CommandEnvelope::new(
                id,
                status.run_id.expect("ready"),
                status.epoch,
                payload,
            ))
            .expect("queued");
    }

    fn acks_until(worker: &SimWorker, id: u64, final_only: bool) -> Vec<CommandAck> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut seen = Vec::new();
        loop {
            seen.extend(worker.acks());
            if seen
                .iter()
                .any(|a| a.command_id == id && (!final_only || !a.is_accepted()))
            {
                return seen;
            }
            assert!(Instant::now() < deadline, "no answer for {id}: {seen:?}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn it_starts_held_and_runs_nothing_until_told() {
        let (worker, _) = worker(Pacing::Unpaced);
        let status = wait_for(&worker, |s| s.phase == WorkerPhase::Paused);
        assert_eq!(status.tick, 0);
        std::thread::sleep(Duration::from_millis(50));
        assert!(worker.frames().is_empty());
        assert!(worker.latest_snapshot().is_none());
    }

    /// The frames a reader receives are the straight run's, however slowly it
    /// reads: no tick is dropped and render pace changes nothing.
    #[test]
    fn a_slow_reader_receives_every_tick_of_the_straight_run() {
        let (worker, bundle) = worker(Pacing::Unpaced);
        command(&worker, 1, CommandKind::Resume);
        let mut frames = Vec::new();
        let mut naps = [3u64, 0, 7, 1, 0, 0, 5].iter().cycle();
        loop {
            // Completed is published after the last frame is sent, so one
            // more drain after seeing it collects everything.
            let completed = worker.status().phase == WorkerPhase::Completed;
            frames.extend(worker.frames());
            if completed {
                break;
            }
            std::thread::sleep(Duration::from_millis(*naps.next().unwrap()));
        }
        let straight = assemble(&bundle)
            .expect("assembles")
            .run_to_replay()
            .expect("runs");
        assert_eq!(frames.len(), straight.frames.len());
        for (index, (got, want)) in frames.iter().zip(&straight.frames).enumerate() {
            assert_eq!(got.tick, index as u64);
            assert!(
                serde_json::to_string(&got.frame).unwrap() == serde_json::to_string(want).unwrap(),
                "frame {index} differs from the straight run"
            );
        }
        let acks = acks_until(&worker, 1, true);
        assert!(acks
            .iter()
            .any(|a| a.command_id == 1 && a.status == AckStatus::Applied));
    }

    /// P5.2: after the run finishes, a seek back restores the nearest
    /// checkpoint, branches the epoch and holds at the tick without
    /// publishing the ticks it re-ran; running on gives exactly the straight
    /// run's frames from there.
    #[test]
    fn a_seek_back_branches_and_runs_on_as_the_straight_run() {
        let (worker, _) = worker(Pacing::Unpaced);
        command(&worker, 1, CommandKind::Resume);
        let first = run_to_completion(&worker);
        assert_eq!(first.len(), 80);
        let before = worker.status();
        assert_eq!(before.retained_from, Some(0));
        assert_eq!(before.branched_at, None);

        command(&worker, 2, CommandKind::Seek { tick: 30 });
        let acks = acks_until(&worker, 2, true);
        let ack = acks.iter().find(|a| a.command_id == 2).unwrap();
        assert_eq!(ack.status, AckStatus::Applied, "{ack:?}");
        assert_eq!(ack.effective_tick, Some(30));
        let status = wait_for(&worker, |s| {
            s.phase == WorkerPhase::Paused && s.epoch != before.epoch
        });
        assert_eq!(status.tick, 30);
        assert_eq!(status.branched_at, Some(30));
        assert!(
            worker.frames().is_empty(),
            "the re-run ticks are not resent"
        );

        command(&worker, 3, CommandKind::Resume);
        let rest = run_to_completion(&worker);
        assert_eq!(rest.len(), 50);
        assert!(
            rest == first[30..],
            "the branch reproduces the straight run"
        );
    }

    /// A seek in a run with lanes at their own rates rebuilds the lanes
    /// executor at the restored tick, and the branch still runs on as the
    /// straight run did, buffered lane state included.
    #[test]
    fn a_seek_with_lanes_at_their_own_rates_runs_on_as_the_straight_run() {
        use crate::run_options::RunOptions;
        use crate::scheduler::parse_rate;
        let path = format!(
            "{}/../../tests/fixtures/scenario_bundle_scan.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let bundle =
            ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture")).unwrap();
        let rates = LaneRates {
            fusion: Some(parse_rate("1/2").expect("rate")),
            mapping: Some(parse_rate("1/3").expect("rate")),
            mission: Some(parse_rate("1/7").expect("rate")),
        };
        let options = RunOptions {
            lane_rates: rates,
            ..RunOptions::default()
        };
        let worker = SimWorker::spawn_with_rates(
            move || {
                let mut run = assemble(&bundle).map_err(|e| e.to_string())?;
                options.apply(&mut run, &bundle)?;
                Ok(run)
            },
            Pacing::Unpaced,
            rates,
        );
        command(&worker, 1, CommandKind::Resume);
        let first = run_to_completion(&worker);
        // Mid-interval: the nearest checkpoint is at 50, so 13 ticks re-run.
        command(&worker, 2, CommandKind::Seek { tick: 63 });
        let acks = acks_until(&worker, 2, true);
        let ack = acks.iter().find(|a| a.command_id == 2).unwrap();
        assert_eq!(ack.status, AckStatus::Applied, "{ack:?}");
        wait_for(&worker, |s| s.phase == WorkerPhase::Paused && s.tick == 63);
        command(&worker, 3, CommandKind::Resume);
        let rest = run_to_completion(&worker);
        assert!(
            rest == first[63..],
            "the branch reproduces the straight run"
        );
    }

    /// A branch from after a scheduled source restart back to before it
    /// applies the restart again when it gets there: the restart guard is
    /// the restored timeline's, not the one left (review finding).
    #[test]
    fn a_branch_back_past_a_source_restart_applies_it_again() {
        use crate::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
        use crate::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};
        let build_bundle = bundle();
        let worker = SimWorker::spawn(
            move || {
                let mut run = assemble(&build_bundle).map_err(|e| e.to_string())?;
                run.set_delivery(DeliveryProfile::from_seconds(
                    build_bundle.dt_s,
                    1.5,
                    1.0,
                    &BTreeMap::new(),
                    0.0,
                    DEFAULT_QUEUE_CAPACITY,
                    11,
                )?)?;
                run.set_faults(FaultManifest {
                    format: MANIFEST_FORMAT.into(),
                    version: MANIFEST_VERSION,
                    faults: vec![Fault::SourceRestart {
                        node_id: "drone-east".into(),
                        at_s: 15.0,
                    }],
                })?;
                Ok(run)
            },
            Pacing::Unpaced,
        );
        command(&worker, 1, CommandKind::Resume);
        let first = run_to_completion(&worker);
        // The restart is at tick 30 and catches evidence in flight; branch
        // back to 28, from the tick-25 checkpoint.
        let stale = |frames: &[String]| {
            frames
                .iter()
                .filter(|f| f.contains("delivery_stale_epoch"))
                .count()
        };
        assert!(
            stale(&first) > 0,
            "the case needs evidence caught in flight"
        );
        command(&worker, 2, CommandKind::Seek { tick: 28 });
        let acks = acks_until(&worker, 2, true);
        assert_eq!(
            acks.iter().find(|a| a.command_id == 2).unwrap().status,
            AckStatus::Applied
        );
        wait_for(&worker, |s| s.phase == WorkerPhase::Paused && s.tick == 28);
        command(&worker, 3, CommandKind::Resume);
        let rest = run_to_completion(&worker);
        assert_eq!(stale(&rest), stale(&first[28..]));
        assert!(
            rest == first[28..],
            "the restart is applied again on the branch"
        );
    }

    /// A command still pending when a seek branches the run is answered as
    /// cancelled, not left pending forever (review finding).
    #[test]
    fn a_seek_cancels_commands_pending_on_the_epoch_it_leaves() {
        let (worker, _) = worker(Pacing::Unpaced);
        let status = wait_for(&worker, |s| s.phase == WorkerPhase::Paused);
        let mut deferred = CommandEnvelope::new(
            1,
            status.run_id.clone().unwrap(),
            status.epoch,
            CommandKind::SetSpeed { multiplier: 2.0 },
        );
        deferred.requested_effective_tick = Some(1000);
        worker.submit(deferred).unwrap();
        let acks = acks_until(&worker, 1, false);
        assert!(acks
            .iter()
            .any(|a| a.command_id == 1 && a.status == AckStatus::Accepted));
        command(&worker, 2, CommandKind::Seek { tick: 0 });
        let acks = acks_until(&worker, 1, true);
        assert!(matches!(
            &acks.iter().rfind(|a| a.command_id == 1).unwrap().status,
            AckStatus::Cancelled(RejectReason::StaleEpoch { .. })
        ));
    }

    /// A seek forward, or one composed against an old epoch, is refused
    /// with its reason and changes nothing.
    #[test]
    fn a_seek_forward_or_stale_is_refused() {
        let (worker, _) = worker(Pacing::Unpaced);
        let status = wait_for(&worker, |s| s.phase == WorkerPhase::Paused);
        command(&worker, 1, CommandKind::Seek { tick: 5 });
        let acks = acks_until(&worker, 1, true);
        assert!(matches!(
            &acks.iter().find(|a| a.command_id == 1).unwrap().status,
            AckStatus::Rejected(RejectReason::Unsupported { .. })
        ));
        worker
            .submit(CommandEnvelope::new(
                2,
                status.run_id.clone().unwrap(),
                status.epoch + 7,
                CommandKind::Seek { tick: 0 },
            ))
            .unwrap();
        let acks = acks_until(&worker, 2, true);
        assert!(matches!(
            &acks.iter().find(|a| a.command_id == 2).unwrap().status,
            AckStatus::Rejected(RejectReason::StaleEpoch { .. })
        ));
        let after = worker.status();
        assert_eq!((after.epoch, after.tick), (status.epoch, status.tick));
        assert_eq!(after.branched_at, None);
    }

    /// Every frame of a finished worker run, collected after it completes.
    fn run_to_completion(worker: &SimWorker) -> Vec<String> {
        let mut frames = Vec::new();
        loop {
            let completed = worker.status().phase == WorkerPhase::Completed;
            frames.extend(
                worker
                    .frames()
                    .into_iter()
                    .map(|frame| serde_json::to_string(&frame.frame).unwrap()),
            );
            if completed {
                return frames;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// B2.4 and B2.5 in the worker: a run with lanes at their own rates is
    /// stepped through the lanes executor at those rates, so its frames are
    /// the CLI's (`--executor lanes --mapping-rate --mission-rate`), and a
    /// reset starts the lanes' deadlines over with the run.
    #[test]
    fn a_lane_at_its_own_rate_runs_as_the_clis_scheduled_run() {
        use crate::run_options::RunOptions;
        use crate::scheduler::parse_rate;

        let path = format!(
            "{}/../../tests/fixtures/scenario_bundle_scan.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let bundle =
            ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture")).unwrap();
        let rates = LaneRates {
            fusion: Some(parse_rate("1/2").expect("rate")),
            mapping: Some(parse_rate("1/3").expect("rate")),
            mission: Some(parse_rate("1/7").expect("rate")),
        };
        let options = RunOptions {
            lane_rates: rates,
            ..RunOptions::default()
        };
        let build = |bundle: &ScenarioBundle, options: &RunOptions| {
            let mut run = assemble(bundle).map_err(|e| e.to_string())?;
            options.apply(&mut run, bundle)?;
            Ok::<_, String>(run)
        };

        // What `argusnet-rs sim --executor lanes --mapping-rate 1/3
        // --mission-rate 1/7` steps.
        let mut direct = build(&bundle, &options).unwrap();
        let mut executor = rates
            .apply(PhasedLaneExecutor::new::<RunError>(bundle.dt_s, 0).unwrap())
            .unwrap();
        loop {
            match executor.step(&mut direct) {
                Ok(_) => {}
                Err(ExecutorError::Tick(RunError::Tick(TickError::Completed { .. }))) => break,
                Err(error) => panic!("{error}"),
            }
        }
        let want: Vec<String> = direct
            .run_to_replay()
            .unwrap()
            .frames
            .iter()
            .map(|frame| serde_json::to_string(frame).unwrap())
            .collect();
        let legacy: Vec<String> = assemble(&bundle)
            .unwrap()
            .run_to_replay()
            .unwrap()
            .frames
            .iter()
            .map(|frame| serde_json::to_string(frame).unwrap())
            .collect();
        assert_ne!(want, legacy, "the rate changes the run");

        let (worker_bundle, worker_options) = (bundle.clone(), options.clone());
        let worker = SimWorker::spawn_with_rates(
            move || build(&worker_bundle, &worker_options),
            Pacing::Unpaced,
            rates,
        );
        command(&worker, 1, CommandKind::Resume);
        assert_eq!(run_to_completion(&worker), want);

        // A reset runs it again from the lane's first deadline.
        command(&worker, 2, CommandKind::Reset);
        wait_for(&worker, |s| s.epoch == 1 && s.phase == WorkerPhase::Paused);
        command(&worker, 3, CommandKind::Resume);
        assert_eq!(run_to_completion(&worker), want);
    }

    /// A mission rate the run's timestep cannot be scheduled at is refused
    /// by the check a front end makes before it opens, and the worker built
    /// from it fails with the same reason rather than running unscheduled.
    #[test]
    fn a_rate_on_an_inexact_timestep_is_refused_before_the_run_starts() {
        let mut inexact = bundle();
        inexact.dt_s = 1.0 / 3.0;
        let run = assemble(&inexact).expect("an inexact timestep still assembles");
        let rates = LaneRates {
            mission: Some(crate::scheduler::parse_rate("1/1").unwrap()),
            ..LaneRates::default()
        };
        let refused = scheduled_executor(&run, &rates).expect_err("refused");
        assert!(refused.contains("whole number of nanoseconds"), "{refused}");

        let worker = SimWorker::spawn_with_rates(
            move || assemble(&inexact).map_err(|e| e.to_string()),
            Pacing::Unpaced,
            rates,
        );
        let status = wait_for(&worker, |s| matches!(s.phase, WorkerPhase::Failed(_)));
        assert_eq!(status.phase, WorkerPhase::Failed(refused));
    }

    #[test]
    fn every_admitted_command_is_answered_twice_and_pause_holds() {
        let (worker, _) = worker(Pacing::RealTime);
        command(&worker, 1, CommandKind::SetSpeed { multiplier: 50.0 });
        command(&worker, 2, CommandKind::Resume);
        wait_for(&worker, |s| s.tick >= 5);
        command(&worker, 3, CommandKind::Pause);
        let acks = acks_until(&worker, 3, true);
        for id in [1, 2, 3] {
            let answers: Vec<_> = acks.iter().filter(|a| a.command_id == id).collect();
            assert_eq!(answers.len(), 2, "command {id}: {answers:?}");
            assert!(answers[0].is_accepted());
            assert_eq!(answers[1].status, AckStatus::Applied);
        }
        let paused = wait_for(&worker, |s| s.phase == WorkerPhase::Paused);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            worker.status().tick,
            paused.tick,
            "a paused run does not tick"
        );
        assert_eq!(worker.status().speed, 50.0);
    }

    /// Real-time pacing runs one tick per `dt / speed` of wall time. The bounds
    /// are wide: this checks the pacing law, not the scheduler's jitter.
    #[test]
    fn real_time_pacing_follows_dt_over_speed() {
        let (worker, _) = worker(Pacing::RealTime);
        // The fixture's dt is 0.5 s; at 20x, one tick every 25 ms, 40 a second,
        // so 20 in the 0.5 s below. A late wake-up on a loaded machine only
        // loses ticks, so the lower bound is half that: it still separates
        // real-time pacing from a stalled run, and the upper bound from an
        // unpaced one (thousands).
        command(&worker, 1, CommandKind::SetSpeed { multiplier: 20.0 });
        command(&worker, 2, CommandKind::Resume);
        let start = wait_for(&worker, |s| s.tick >= 1);
        std::thread::sleep(Duration::from_millis(500));
        let ticks = worker.status().tick - start.tick;
        assert!(
            (10..=60).contains(&ticks),
            "{ticks} ticks in 0.5 s at 40 per second (20 expected)"
        );
    }

    #[test]
    fn single_step_runs_exactly_one_tick() {
        let (worker, _) = worker(Pacing::Unpaced);
        command(&worker, 1, CommandKind::SingleStep);
        acks_until(&worker, 1, true);
        let status = wait_for(&worker, |s| s.tick == 1 && s.phase == WorkerPhase::Paused);
        assert_eq!(status.tick, 1);
        std::thread::sleep(Duration::from_millis(50));
        let frames = worker.frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].tick, 0);
    }

    #[test]
    fn a_retry_keeps_the_original_answer_and_says_so() {
        let (worker, _) = worker(Pacing::Unpaced);
        command(&worker, 7, CommandKind::SetSpeed { multiplier: 2.0 });
        acks_until(&worker, 7, true);
        command(&worker, 7, CommandKind::SetSpeed { multiplier: 2.0 });
        let deadline = Instant::now() + Duration::from_secs(10);
        let retry = loop {
            if let Some(ack) = worker.acks().into_iter().find(|a| a.command_id == 7) {
                break ack;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        };
        assert!(retry.duplicate);
        assert_eq!(retry.status, AckStatus::Applied);
    }

    #[test]
    fn reset_starts_a_new_epoch_held_at_tick_zero() {
        let (worker, _) = worker(Pacing::Unpaced);
        command(&worker, 1, CommandKind::SingleStep);
        wait_for(&worker, |s| s.tick == 1 && s.phase == WorkerPhase::Paused);
        let before = worker.status().epoch;
        command(&worker, 2, CommandKind::Reset);
        let acks = acks_until(&worker, 2, true);
        let reset = acks.iter().find(|a| a.command_id == 2).unwrap();
        assert_eq!(reset.status, AckStatus::Applied);
        assert_eq!(reset.epoch, before + 1);
        let status = wait_for(&worker, |s| s.epoch == before + 1);
        assert_eq!((status.tick, status.phase), (0, WorkerPhase::Paused));
        assert!(worker.latest_snapshot().is_none());
        // A command composed against the old epoch is refused, not applied.
        worker
            .submit(CommandEnvelope::new(
                3,
                status.run_id.clone().unwrap(),
                before,
                CommandKind::Resume,
            ))
            .expect("queued");
        let acks = acks_until(&worker, 3, true);
        let stale = acks.iter().rev().find(|a| a.command_id == 3).unwrap();
        assert!(matches!(
            stale.status,
            AckStatus::Cancelled(RejectReason::StaleEpoch { .. })
                | AckStatus::Rejected(RejectReason::StaleEpoch { .. })
        ));
    }

    /// Queue saturation answers Busy, explicitly, while the world loads.
    #[test]
    fn a_full_queue_answers_busy() {
        let release = Arc::new(AtomicBool::new(false));
        let gate = Arc::clone(&release);
        let bundle = bundle();
        let worker = SimWorker::spawn_with_capacity(
            move || {
                while !gate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                assemble(&bundle).map_err(|e| e.to_string())
            },
            Pacing::Unpaced,
            4,
        );
        let mut busy = None;
        for id in 1..=10 {
            if let Err(ack) = worker.submit(CommandEnvelope::new(id, "x", 0, CommandKind::Pause)) {
                busy = Some((id, ack));
                break;
            }
        }
        let (id, ack) = busy.expect("the queue filled");
        assert_eq!(id, 5);
        assert_eq!(
            ack.status,
            AckStatus::Rejected(RejectReason::Busy { capacity: 4 })
        );
        release.store(true, Ordering::SeqCst);
    }

    /// Closing while the world loads returns at once, and the world built
    /// afterwards is never published.
    #[test]
    fn closing_during_load_returns_at_once() {
        let release = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let (gate, done) = (Arc::clone(&release), Arc::clone(&finished));
        let bundle = bundle();
        let mut worker = SimWorker::spawn(
            move || {
                while !gate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                let run = assemble(&bundle).map_err(|e| e.to_string());
                done.store(true, Ordering::SeqCst);
                run
            },
            Pacing::Unpaced,
        );
        let started = Instant::now();
        worker.stop();
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(worker.status().phase, WorkerPhase::Stopped);
        release.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(60);
        while !finished.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(worker.status().phase, WorkerPhase::Stopped);
        assert!(worker.latest_snapshot().is_none());
    }

    /// Run and Stop in quick succession, many times: no hang, no panic.
    #[test]
    fn rapid_run_and_stop() {
        let bundle = bundle();
        for round in 0..20 {
            let build = bundle.clone();
            let mut worker = SimWorker::spawn(
                move || assemble(&build).map_err(|e| e.to_string()),
                Pacing::Unpaced,
            );
            if round % 2 == 0 {
                command(&worker, 1, CommandKind::Resume);
            }
            worker.stop();
            assert_eq!(worker.status().phase, WorkerPhase::Stopped);
        }
    }

    #[test]
    fn a_world_that_fails_to_build_says_why() {
        let worker = SimWorker::spawn(|| Err("terrain tile missing".to_string()), Pacing::Unpaced);
        let status = wait_for(&worker, |s| matches!(s.phase, WorkerPhase::Failed(_)));
        assert_eq!(
            status.phase,
            WorkerPhase::Failed("terrain tile missing".into())
        );
        assert!(status.run_id.is_none());
    }
}
