//! B2.3: the first lanes out of the compound task (SCHEDULER_LEGACY_MAPPING.md).
//!
//! Fusion, mapping, mission and publication leave the compound lane as their
//! own tasks, each in its own [`Phase`] at the legacy boundary's deadline.
//! Their legacy order (7 → 8 → 9 → 10) is the phases' order (estimators →
//! mapping → mission/planning → publication), so the extraction keeps every
//! stage where it ran and the run stays bit-for-bit the legacy one. Stages
//! 0–6 stay one compound task in [`Phase::Propagate`]: moving any of them is a
//! reorder (§1 of the mapping note), and so a declared behaviour change, not
//! this slice.
//!
//! Every lane still reads the legacy float time (`tick * dt_s`): none has left
//! the legacy derivation yet, so none mixes the two.

use std::num::NonZeroU64;

use super::executor::{exact_nanos, ExecutorError, LEGACY_LANE};
use super::queue::{EventKey, EventQueue, Phase, SourceId};
use super::time::{Schedule, SimTime};
use crate::stage::Stage;

/// A tick that can be run in parts: opened, its stages run up to a point, and
/// committed. [`crate::simulation::Simulation`] and
/// [`crate::scenario::AssembledRun`] implement it.
pub trait PhasedTick {
    /// A tick in progress.
    type Open;
    /// What a committed tick yields.
    type Output;
    type Error;

    /// The legacy tick the next [`Self::begin`] opens.
    fn current_tick(&self) -> u64;
    /// Stage 0 and the boundary checks. `Err(output)` inside the `Ok` is a tick
    /// that stage 0 already committed (a reset): nothing more runs.
    fn begin(&mut self) -> Result<Result<Self::Open, Self::Output>, Self::Error>;
    /// Run the open tick's stages up to and including `last`.
    fn run_through(&mut self, open: &mut Self::Open, last: Stage) -> Result<(), Self::Error>;
    /// Stage 10.
    fn commit(&mut self, open: Self::Open) -> Result<Self::Output, Self::Error>;
}

/// One task of a legacy boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePart {
    /// Open legacy tick `n` and run its stages through `last` (the compound
    /// lane: stages 0–6).
    Open { n: u64, last: Stage },
    /// Run legacy tick `n`'s stages through `last`.
    Run { n: u64, last: Stage },
    /// Commit legacy tick `n`.
    Commit { n: u64 },
}

/// A lane: its stable source, its phase, and the last stage it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    pub source: SourceId,
    pub phase: Phase,
    /// `None` for publication, which commits.
    pub last: Option<Stage>,
}

/// `lanes_v1`: the compound lane, then fusion, mapping, mission and
/// publication, in phase order.
pub const LANES_V1: [Lane; 5] = [
    Lane {
        source: LEGACY_LANE,
        phase: Phase::Propagate,
        last: Some(Stage::GenerateObservations),
    },
    Lane {
        source: SourceId(1),
        phase: Phase::Estimators,
        last: Some(Stage::UpdateFusion),
    },
    Lane {
        source: SourceId(2),
        phase: Phase::Mapping,
        last: Some(Stage::UpdateMappingAndLocalization),
    },
    Lane {
        source: SourceId(3),
        phase: Phase::MissionPlanning,
        last: Some(Stage::EvaluateMissionTransitions),
    },
    Lane {
        source: SourceId(4),
        phase: Phase::Publication,
        last: None,
    },
];

/// `legacy_stages_v1` driven as [`LANES_V1`] from the scheduler's queue.
#[derive(Debug)]
pub struct PhasedLaneExecutor {
    queue: EventQueue<LanePart>,
    boundaries: Schedule,
    epoch: SimTime,
    /// The next legacy tick to schedule.
    next: u64,
    /// The keys of the tasks the last boundary ran, in the order they ran.
    ran: Vec<EventKey>,
}

impl PhasedLaneExecutor {
    /// An executor for a run whose legacy timestep is `dt_s`, starting at
    /// legacy tick `first`.
    pub fn new<E>(dt_s: f64, first: u64) -> Result<Self, ExecutorError<E>> {
        let dt_ns: NonZeroU64 = exact_nanos(dt_s).ok_or(ExecutorError::InexactTimestep { dt_s })?;
        Ok(Self {
            queue: EventQueue::new(SimTime::EPOCH, dt_ns),
            boundaries: Schedule::Period(dt_ns),
            epoch: SimTime::EPOCH,
            next: first,
            ran: Vec::new(),
        })
    }

    /// Run the next legacy boundary's lanes, in queue order, and return what
    /// its commit yields.
    pub fn step<P: PhasedTick>(
        &mut self,
        run: &mut P,
    ) -> Result<P::Output, ExecutorError<P::Error>> {
        let n = self.next;
        let ran = run.current_tick();
        if ran != n {
            return Err(ExecutorError::OutOfStep { scheduled: n, ran });
        }
        let deadline = self
            .boundaries
            .nth_deadline(self.epoch, n)
            .map_err(ExecutorError::Time)?;
        for lane in LANES_V1 {
            let part = match lane.last {
                Some(last) if lane.source == LEGACY_LANE => LanePart::Open { n, last },
                Some(last) => LanePart::Run { n, last },
                None => LanePart::Commit { n },
            };
            self.queue
                .push(deadline, lane.phase, lane.source, part)
                .map_err(ExecutorError::Time)?;
        }
        self.ran.clear();
        let outcome = self.drain(run, n);
        if outcome.is_err() {
            // The tick failed: whatever of it is still queued never runs.
            self.queue.reset(self.epoch);
        }
        outcome
    }

    fn drain<P: PhasedTick>(
        &mut self,
        run: &mut P,
        n: u64,
    ) -> Result<P::Output, ExecutorError<P::Error>> {
        let mut open = None;
        while let Some((key, part)) = self.queue.pop() {
            self.ran.push(key);
            match part {
                LanePart::Open { last, .. } => match run.begin().map_err(ExecutorError::Tick)? {
                    Ok(mut opened) => {
                        run.run_through(&mut opened, last)
                            .map_err(ExecutorError::Tick)?;
                        open = Some(opened);
                    }
                    Err(output) => {
                        // Stage 0 reset the run and committed it: the rest
                        // of this boundary belongs to the old epoch.
                        self.queue.reset(self.epoch);
                        self.next = run.current_tick();
                        return Ok(output);
                    }
                },
                LanePart::Run { last, .. } => {
                    let opened = open.as_mut().ok_or(ExecutorError::OutOfStep {
                        scheduled: n,
                        ran: run.current_tick(),
                    })?;
                    run.run_through(opened, last).map_err(ExecutorError::Tick)?;
                }
                LanePart::Commit { .. } => {
                    let opened = open.take().ok_or(ExecutorError::OutOfStep {
                        scheduled: n,
                        ran: run.current_tick(),
                    })?;
                    let output = run.commit(opened).map_err(ExecutorError::Tick)?;
                    self.next = n + 1;
                    return Ok(output);
                }
            }
        }
        Err(ExecutorError::OutOfStep {
            scheduled: n,
            ran: run.current_tick(),
        })
    }

    /// The keys of the tasks the last boundary ran, in the order they ran.
    pub fn ran(&self) -> &[EventKey] {
        &self.ran
    }

    /// A new epoch: anything queued is discarded (the count is returned), and
    /// scheduling restarts at legacy tick 0.
    pub fn reset(&mut self) -> usize {
        self.next = 0;
        self.ran.clear();
        self.queue.reset(self.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tick that records the parts it was driven through.
    #[derive(Default)]
    struct Recorder {
        tick: u64,
        log: Vec<String>,
        fail_at: Option<Stage>,
    }

    impl PhasedTick for Recorder {
        type Open = u64;
        type Output = u64;
        type Error = String;

        fn current_tick(&self) -> u64 {
            self.tick
        }
        fn begin(&mut self) -> Result<Result<u64, u64>, String> {
            self.log.push(format!("begin {}", self.tick));
            Ok(Ok(self.tick))
        }
        fn run_through(&mut self, open: &mut u64, last: Stage) -> Result<(), String> {
            self.log
                .push(format!("run {open} through {}", last.order()));
            if self.fail_at == Some(last) {
                return Err(format!("failed at {}", last.order()));
            }
            Ok(())
        }
        fn commit(&mut self, open: u64) -> Result<u64, String> {
            self.log.push(format!("commit {open}"));
            self.tick += 1;
            Ok(open)
        }
    }

    #[test]
    fn a_boundary_runs_its_lanes_in_phase_order() {
        let mut executor = PhasedLaneExecutor::new::<String>(0.1, 0).unwrap();
        let mut run = Recorder::default();
        for n in 0..3u64 {
            run.log.clear();
            assert_eq!(executor.step(&mut run).unwrap(), n);
            assert_eq!(
                run.log,
                [
                    format!("begin {n}"),
                    format!("run {n} through 6"),
                    format!("run {n} through 7"),
                    format!("run {n} through 8"),
                    format!("run {n} through 9"),
                    format!("commit {n}"),
                ]
            );
            let phases: Vec<Phase> = executor.ran().iter().map(|key| key.phase).collect();
            assert_eq!(
                phases,
                [
                    Phase::Propagate,
                    Phase::Estimators,
                    Phase::Mapping,
                    Phase::MissionPlanning,
                    Phase::Publication,
                ]
            );
            assert!(executor
                .ran()
                .iter()
                .all(|key| key.time.as_nanos() == n as i64 * 100_000_000));
        }
    }

    #[test]
    fn a_failed_lane_stops_the_boundary_and_discards_the_rest() {
        let mut executor = PhasedLaneExecutor::new::<String>(1.0, 0).unwrap();
        let mut run = Recorder {
            fail_at: Some(Stage::UpdateMappingAndLocalization),
            ..Recorder::default()
        };
        assert_eq!(
            executor.step(&mut run),
            Err(ExecutorError::Tick("failed at 8".into()))
        );
        assert_eq!(run.log.last().map(String::as_str), Some("run 0 through 8"));
        assert_eq!(executor.ran().len(), 3);
        // The same boundary is scheduled again, from a clean queue.
        run.fail_at = None;
        run.log.clear();
        assert_eq!(executor.step(&mut run), Ok(0));
        assert_eq!(run.log.len(), 6);
    }

    #[test]
    fn a_run_out_of_step_is_refused() {
        let mut executor = PhasedLaneExecutor::new::<String>(1.0, 2).unwrap();
        let mut run = Recorder::default();
        assert_eq!(
            executor.step(&mut run),
            Err(ExecutorError::OutOfStep {
                scheduled: 2,
                ran: 0
            })
        );
        assert!(run.log.is_empty());
    }
}
