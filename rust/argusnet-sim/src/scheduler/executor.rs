//! The serial executor, B2.2: `legacy_stages_v1` as one compound lane
//! (SCHEDULER_LEGACY_MAPPING.md).
//!
//! Each legacy boundary is one [`LaneTask::LegacyTick`] in the
//! [`Phase::Propagate`] phase, at `Schedule::Period(dt)`'s deadline. The task
//! runs the eleven stages in their legacy order through the caller's closure,
//! with the legacy float time, so the run is bit-for-bit the legacy one; what
//! this proves is the plumbing -- integer deadlines, one total order, epochs
//! and reset through the queue -- that later slices extract lanes onto.

use std::fmt;
use std::num::NonZeroU64;

use super::queue::{EventKey, EventQueue, Phase, SourceId};
use super::time::{Schedule, SimTime, TimeError, NANOS_PER_SECOND};

/// The legacy pipeline's stable source identity.
pub const LEGACY_LANE: SourceId = SourceId(0);

/// What the executor runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneTask {
    /// Legacy tick `n`: all eleven stages, in order.
    LegacyTick { n: u64 },
}

/// Why the executor would not run.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorError<E> {
    /// `dt_s` is not a whole number of nanoseconds, so the integer boundaries
    /// would not be the legacy ones. Refused at configuration.
    InexactTimestep {
        dt_s: f64,
    },
    Time(TimeError),
    /// The legacy tick the closure reports is not the one scheduled.
    OutOfStep {
        scheduled: u64,
        ran: u64,
    },
    /// The tick itself failed.
    Tick(E),
}

impl<E: fmt::Display> fmt::Display for ExecutorError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::InexactTimestep { dt_s } => write!(
                formatter,
                "timestep {dt_s} s is not a whole number of nanoseconds"
            ),
            ExecutorError::Time(error) => write!(formatter, "{error}"),
            ExecutorError::OutOfStep { scheduled, ran } => {
                write!(
                    formatter,
                    "scheduled legacy tick {scheduled}, but tick {ran} ran"
                )
            }
            ExecutorError::Tick(error) => write!(formatter, "{error}"),
        }
    }
}

/// `legacy_stages_v1` driven from the scheduler's queue.
#[derive(Debug)]
pub struct LegacyLaneExecutor {
    queue: EventQueue<LaneTask>,
    boundaries: Schedule,
    epoch: SimTime,
    /// The next legacy tick to schedule.
    next: u64,
    /// The key of the last task run, for the trace.
    last: Option<EventKey>,
}

impl LegacyLaneExecutor {
    /// An executor for a run whose legacy timestep is `dt_s`, starting at
    /// legacy tick `first` (0 for a fresh run; the restored tick for a
    /// resumed one).
    pub fn new<E>(dt_s: f64, first: u64) -> Result<Self, ExecutorError<E>> {
        let dt_ns = exact_nanos(dt_s).ok_or(ExecutorError::InexactTimestep { dt_s })?;
        Ok(Self {
            queue: EventQueue::new(SimTime::EPOCH, dt_ns),
            boundaries: Schedule::Period(dt_ns),
            epoch: SimTime::EPOCH,
            next: first,
            last: None,
        })
    }

    /// Run the next due task. `tick` runs one legacy tick and returns the tick
    /// number it ran and its output.
    pub fn step<T, E>(
        &mut self,
        mut tick: impl FnMut() -> Result<(u64, T), E>,
    ) -> Result<T, ExecutorError<E>> {
        if self.queue.is_empty() {
            let deadline = self
                .boundaries
                .nth_deadline(self.epoch, self.next)
                .map_err(ExecutorError::Time)?;
            self.queue
                .push(
                    deadline,
                    Phase::Propagate,
                    LEGACY_LANE,
                    LaneTask::LegacyTick { n: self.next },
                )
                .map_err(ExecutorError::Time)?;
        }
        let (key, task) = self.queue.pop().expect("a task was just queued");
        self.last = Some(key);
        match task {
            LaneTask::LegacyTick { n } => {
                let (ran, output) = tick().map_err(ExecutorError::Tick)?;
                if ran != n {
                    return Err(ExecutorError::OutOfStep { scheduled: n, ran });
                }
                self.next = n + 1;
                Ok(output)
            }
        }
    }

    /// The deadline and phase of the last task run.
    pub fn last(&self) -> Option<EventKey> {
        self.last
    }

    /// A new epoch: the old epoch's queued tasks are discarded (the count is
    /// returned), and scheduling restarts at legacy tick 0.
    pub fn reset(&mut self) -> usize {
        self.next = 0;
        self.last = None;
        self.queue.reset(self.epoch)
    }
}

/// `dt_s` in whole nanoseconds, if it is exactly that: the integer boundary
/// converts back to the same `f64`.
pub(super) fn exact_nanos(dt_s: f64) -> Option<NonZeroU64> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return None;
    }
    let nanos = (dt_s * NANOS_PER_SECOND as f64).round();
    if nanos < 1.0 || nanos > u64::MAX as f64 {
        return None;
    }
    let nanos = nanos as u64;
    (nanos as f64 / NANOS_PER_SECOND as f64 == dt_s)
        .then_some(nanos)
        .and_then(NonZeroU64::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_legacy_timesteps_are_whole_nanoseconds() {
        for dt_s in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0] {
            assert!(exact_nanos(dt_s).is_some(), "{dt_s}");
        }
        assert_eq!(exact_nanos(0.1).map(NonZeroU64::get), Some(100_000_000));
        for dt_s in [1e-10, 1.0 / 3.0, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(exact_nanos(dt_s).is_none(), "{dt_s}");
        }
    }

    #[test]
    fn ticks_run_in_order_on_integer_deadlines() {
        let mut executor = LegacyLaneExecutor::new::<()>(0.1, 0).unwrap();
        let mut tick = 0u64;
        for expected in 0..5u64 {
            let ran = executor
                .step(|| -> Result<(u64, u64), ()> {
                    let ran = tick;
                    tick += 1;
                    Ok((ran, ran))
                })
                .unwrap();
            assert_eq!(ran, expected);
            let key = executor.last().unwrap();
            assert_eq!(key.time.as_nanos(), expected as i64 * 100_000_000);
            assert_eq!(key.phase, Phase::Propagate);
        }
    }

    #[test]
    fn a_tick_out_of_step_is_refused() {
        let mut executor = LegacyLaneExecutor::new::<()>(1.0, 3).unwrap();
        assert_eq!(
            executor.step(|| -> Result<(u64, ()), ()> { Ok((4, ())) }),
            Err(ExecutorError::OutOfStep {
                scheduled: 3,
                ran: 4
            })
        );
    }

    #[test]
    fn an_inexact_timestep_is_refused_at_configuration() {
        assert_eq!(
            LegacyLaneExecutor::new::<()>(1.0 / 3.0, 0).map(|_| ()),
            Err(ExecutorError::InexactTimestep { dt_s: 1.0 / 3.0 })
        );
    }
}
