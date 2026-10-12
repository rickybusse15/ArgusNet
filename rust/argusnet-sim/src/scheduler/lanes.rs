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
use super::time::{Schedule, SimTime, TimeError};
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
    /// Record the open tick's stages up to and including `last` as not due
    /// this boundary, without running them (B2.4).
    fn skip_through(
        &mut self,
        open: &mut Self::Open,
        last: Stage,
        reason: &str,
    ) -> Result<(), Self::Error>;
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
    /// Legacy tick `n`'s stages through `last` are not due this boundary:
    /// their lane runs at its own rate (B2.4). Recorded, not run.
    NotDue { n: u64, last: Stage },
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

/// The mission lane's own rate (B2.4) and its next occurrence.
#[derive(Debug, Clone, Copy)]
struct LaneRate {
    schedule: Schedule,
    /// The next occurrence index not yet served by a boundary.
    next: u64,
}

/// `legacy_stages_v1` driven as [`LANES_V1`] from the scheduler's queue.
///
/// Every lane runs every legacy boundary unless it is given its own rate
/// ([`Self::with_mission_rate`]). A lane at its own rate runs at the first
/// boundary at or after each of its deadlines -- a deadline between two
/// boundaries is served by the later one -- and sits out the rest, recorded
/// as not due. That is `scheduled_v1`: a declared behaviour change, never the
/// default.
#[derive(Debug)]
pub struct PhasedLaneExecutor {
    queue: EventQueue<LanePart>,
    boundaries: Schedule,
    epoch: SimTime,
    /// The next legacy tick to schedule.
    next: u64,
    /// The keys of the tasks the last boundary ran, in the order they ran.
    ran: Vec<EventKey>,
    /// The mission lane's own rate; `None` runs it every boundary.
    mission: Option<LaneRate>,
    /// The occurrence the mission lane moves on to once the boundary in
    /// flight commits. A boundary that does not commit (paused, failed,
    /// reset) consumes nothing.
    mission_after_commit: Option<u64>,
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
            mission: None,
            mission_after_commit: None,
        })
    }

    /// The mission lane (stage 9) at its own rate, from the boundary this
    /// executor starts at. Mission is the first lane allowed off the legacy
    /// rate: its inputs are cumulative (the map, the fleet), so a boundary it
    /// sits out loses no evidence, and its one use of elapsed time (inspect
    /// dwell) is given the time since it last ran. The occurrences before the
    /// start boundary count as served, so a resumed run picks up where an
    /// uninterrupted one would be.
    pub fn with_mission_rate(mut self, schedule: Schedule) -> Result<Self, TimeError> {
        schedule.validate()?;
        let next = self.occurrences_before(schedule, self.next)?;
        self.mission = Some(LaneRate { schedule, next });
        Ok(self)
    }

    /// The mission lane's own rate, if it has one.
    pub fn mission_rate(&self) -> Option<Schedule> {
        self.mission.map(|rate| rate.schedule)
    }

    /// How many of `schedule`'s occurrences boundaries before `n` served:
    /// the first occurrence after boundary `n - 1`.
    fn occurrences_before(&self, schedule: Schedule, n: u64) -> Result<u64, TimeError> {
        if n == 0 {
            return Ok(0);
        }
        let previous = self.boundaries.nth_deadline(self.epoch, n - 1)?;
        schedule.first_at_or_after(self.epoch, previous.checked_add_nanos(1)?)
    }

    /// Whether the mission lane runs at the boundary whose deadline is
    /// `deadline`, and if so the occurrence it moves on to once that boundary
    /// commits (every occurrence the boundary serves). Nothing is consumed
    /// here: a boundary that never commits must not use up a deadline.
    fn mission_due(&self, deadline: SimTime) -> Result<(bool, Option<u64>), TimeError> {
        let Some(rate) = self.mission.as_ref() else {
            return Ok((true, None));
        };
        if rate.schedule.nth_deadline(self.epoch, rate.next)? > deadline {
            return Ok((false, None));
        }
        let after = rate
            .schedule
            .first_at_or_after(self.epoch, deadline.checked_add_nanos(1)?)?;
        Ok((true, Some(after)))
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
        let (mission_due, mission_after) =
            self.mission_due(deadline).map_err(ExecutorError::Time)?;
        self.mission_after_commit = mission_after;
        for lane in LANES_V1 {
            let part = match lane.last {
                Some(last) if lane.source == LEGACY_LANE => LanePart::Open { n, last },
                Some(last) if lane.phase == Phase::MissionPlanning && !mission_due => {
                    LanePart::NotDue { n, last }
                }
                Some(last) => LanePart::Run { n, last },
                None => LanePart::Commit { n },
            };
            self.queue
                .push(deadline, lane.phase, lane.source, part)
                .map_err(ExecutorError::Time)?;
        }
        self.ran.clear();
        let outcome = self.drain(run, n);
        self.mission_after_commit = None;
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
                        if let Some(rate) = self.mission {
                            let next = self
                                .occurrences_before(rate.schedule, self.next)
                                .map_err(ExecutorError::Time)?;
                            self.mission = Some(LaneRate { next, ..rate });
                        }
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
                LanePart::NotDue { last, .. } => {
                    let opened = open.as_mut().ok_or(ExecutorError::OutOfStep {
                        scheduled: n,
                        ran: run.current_tick(),
                    })?;
                    let reason = match self.mission.map(|rate| rate.schedule) {
                        Some(schedule) => {
                            format!("not due: lane at {} (scheduled_v1)", describe(schedule))
                        }
                        None => "not due (scheduled_v1)".to_string(),
                    };
                    run.skip_through(opened, last, &reason)
                        .map_err(ExecutorError::Tick)?;
                }
                LanePart::Commit { .. } => {
                    let opened = open.take().ok_or(ExecutorError::OutOfStep {
                        scheduled: n,
                        ran: run.current_tick(),
                    })?;
                    let output = run.commit(opened).map_err(ExecutorError::Tick)?;
                    self.next = n + 1;
                    if let (Some(rate), Some(after)) =
                        (self.mission.as_mut(), self.mission_after_commit.take())
                    {
                        rate.next = after;
                    }
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
        if let Some(rate) = self.mission.as_mut() {
            rate.next = 0;
        }
        self.queue.reset(self.epoch)
    }
}

/// A schedule as a reader would write it: `3/1 Hz`, `every 250000000 ns`.
/// A rate as `NUM/DEN` (what `--mission-rate` takes, and what a run directory
/// and a checkpoint record); other schedules as [`describe`] says them.
pub fn rate_text(schedule: Schedule) -> String {
    match schedule {
        Schedule::Rational { hz_num, hz_den } => format!("{hz_num}/{hz_den}"),
        other => describe(other),
    }
}

/// `NUM/DEN` Hz (or `NUM`, for `NUM/1`), validated as a schedule: the text
/// `--mission-rate` takes and [`rate_text`] writes.
pub fn parse_rate(text: &str) -> Result<Schedule, String> {
    let (num, den) = text.split_once('/').unwrap_or((text, "1"));
    let parse = |part: &str| {
        part.trim()
            .parse::<std::num::NonZeroU64>()
            .map_err(|_| format!("{text:?} is not NUM/DEN Hz with positive whole numbers"))
    };
    Schedule::rational(parse(num)?, parse(den)?).map_err(|error| error.to_string())
}

pub fn describe(schedule: Schedule) -> String {
    match schedule {
        Schedule::Rational { hz_num, hz_den } => format!("{hz_num}/{hz_den} Hz"),
        Schedule::Period(period) => format!("every {period} ns"),
        Schedule::Once(at) => format!("once at {} ns", at.as_nanos()),
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
        /// The next `begin` refuses, as a paused run does: no tick opens.
        paused: bool,
    }

    impl PhasedTick for Recorder {
        type Open = u64;
        type Output = u64;
        type Error = String;

        fn current_tick(&self) -> u64 {
            self.tick
        }
        fn begin(&mut self) -> Result<Result<u64, u64>, String> {
            if std::mem::take(&mut self.paused) {
                return Err("paused".to_string());
            }
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
        fn skip_through(
            &mut self,
            open: &mut u64,
            last: Stage,
            reason: &str,
        ) -> Result<(), String> {
            self.log
                .push(format!("skip {open} through {} ({reason})", last.order()));
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

    fn hz(num: u64, den: u64) -> Schedule {
        Schedule::rational(NonZeroU64::new(num).unwrap(), NonZeroU64::new(den).unwrap()).unwrap()
    }

    /// Which of the first `count` boundaries the mission lane runs at.
    fn due_boundaries(executor: &mut PhasedLaneExecutor, first: u64, count: u64) -> Vec<u64> {
        (first..first + count)
            .filter(|&n| {
                let deadline = executor.boundaries.nth_deadline(executor.epoch, n).unwrap();
                let (due, after) = executor.mission_due(deadline).unwrap();
                if let (Some(rate), Some(after)) = (executor.mission.as_mut(), after) {
                    rate.next = after;
                }
                due
            })
            .collect()
    }

    #[test]
    fn a_lane_rate_never_drifts_against_the_boundaries() {
        // 3 Hz against 10 Hz boundaries, a million boundaries (100 000 s):
        // exactly one run per deadline, every one on the first boundary at
        // or after it.
        let mut executor = PhasedLaneExecutor::new::<String>(0.1, 0)
            .unwrap()
            .with_mission_rate(hz(3, 1))
            .unwrap();
        let schedule = hz(3, 1);
        let runs = due_boundaries(&mut executor, 0, 1_000_000);
        assert_eq!(runs.len(), 300_000);
        for (k, n) in runs.iter().enumerate() {
            let deadline = schedule.nth_deadline(SimTime::EPOCH, k as u64).unwrap();
            let boundary = executor
                .boundaries
                .nth_deadline(SimTime::EPOCH, *n)
                .unwrap();
            let before = executor
                .boundaries
                .nth_deadline(SimTime::EPOCH, n.saturating_sub(1))
                .unwrap();
            assert!(
                boundary >= deadline && (*n == 0 || before < deadline),
                "deadline {k}"
            );
        }
    }

    #[test]
    fn an_executor_started_mid_run_serves_the_same_boundaries() {
        let mut straight = PhasedLaneExecutor::new::<String>(0.25, 0)
            .unwrap()
            .with_mission_rate(hz(2, 5))
            .unwrap();
        let all = due_boundaries(&mut straight, 0, 200);
        for start in [1, 9, 10, 11, 57] {
            let mut resumed = PhasedLaneExecutor::new::<String>(0.25, start)
                .unwrap()
                .with_mission_rate(hz(2, 5))
                .unwrap();
            let later: Vec<u64> = all.iter().copied().filter(|n| *n >= start).collect();
            assert_eq!(
                due_boundaries(&mut resumed, start, 200 - start),
                later,
                "from {start}"
            );
        }
        // 2/5 Hz at 0.25 s boundaries: every tenth.
        assert_eq!(&all[..4], [0, 10, 20, 30]);
    }

    #[test]
    fn a_lane_not_due_is_recorded_not_run() {
        let mut run = Recorder::default();
        let mut executor = PhasedLaneExecutor::new::<String>(0.5, 0)
            .unwrap()
            .with_mission_rate(hz(1, 1))
            .unwrap();
        executor.step(&mut run).unwrap();
        run.log.clear();
        executor.step(&mut run).unwrap();
        assert!(
            run.log
                .iter()
                .any(|line| line == "skip 1 through 9 (not due: lane at 1/1 Hz (scheduled_v1))"),
            "{:?}",
            run.log
        );
        assert!(!run
            .log
            .iter()
            .any(|line| line.starts_with("run 1 through 9")));
    }

    #[test]
    fn a_boundary_that_never_opens_uses_up_no_deadline() {
        // 1 Hz at 0.5 s boundaries: due at boundaries 0 and 2. A pause that
        // stops boundary 2 from opening must leave it due when it does.
        let mut run = Recorder::default();
        let mut executor = PhasedLaneExecutor::new::<String>(0.5, 0)
            .unwrap()
            .with_mission_rate(hz(1, 1))
            .unwrap();
        executor.step(&mut run).unwrap();
        executor.step(&mut run).unwrap();
        run.paused = true;
        assert!(executor.step(&mut run).is_err());
        run.log.clear();
        executor.step(&mut run).unwrap();
        assert!(
            run.log.iter().any(|line| line == "run 2 through 9"),
            "{:?}",
            run.log
        );
    }
}
