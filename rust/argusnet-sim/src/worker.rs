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
use crate::simulation::TickError;
use crate::snapshot::RuntimeSnapshot;

/// Commands the worker holds before answering Busy.
pub const DEFAULT_COMMAND_CAPACITY: usize = 64;

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
}

/// One committed tick, in order.
#[derive(Debug, Clone)]
pub struct WorkerFrame {
    pub epoch: u64,
    pub tick: u64,
    pub frame: ReplayFrame,
}

enum Control {
    Command(CommandEnvelope),
    Stop,
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
            }),
            latest: Mutex::new(None),
        });
        let thread_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("argusnet-sim-worker".into())
            .spawn(move || {
                let run = match build() {
                    Ok(run) => run,
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
        match self.run.step() {
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
        let reason = if envelope.run_id != self.run.run_id() {
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
        };
        let ack = match reason {
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
                self.flush_pending();
                *self.shared.latest.lock().expect("latest") = None;
                self.hold();
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
        match self.run.step() {
            Ok(snapshot) => {
                if self.run.sim.epoch() == epoch {
                    if let Some(frame) = self.run.last_frame() {
                        let _ = self.frames.send(WorkerFrame {
                            epoch,
                            tick: self.run.sim.tick().saturating_sub(1),
                            frame: frame.clone(),
                        });
                    }
                }
                *self.shared.latest.lock().expect("latest") = Some(snapshot);
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
        // The fixture's dt is 0.5 s; at 20x, one tick every 25 ms, 40 a second.
        command(&worker, 1, CommandKind::SetSpeed { multiplier: 20.0 });
        command(&worker, 2, CommandKind::Resume);
        let start = wait_for(&worker, |s| s.tick >= 1);
        std::thread::sleep(Duration::from_millis(500));
        let ticks = worker.status().tick - start.tick;
        assert!(
            (20..=60).contains(&ticks),
            "{ticks} ticks in 0.5 s at 80 per second"
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
