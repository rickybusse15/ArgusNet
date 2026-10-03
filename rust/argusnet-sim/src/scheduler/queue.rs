//! The event queue: one total order over everything due.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use super::time::{Schedule, SimTime, TimeError};

/// The phases of one timestamp, in execution order
/// (RUST_RUNTIME_ARCHITECTURE.md §3, ADR-009): propagate physics to the time,
/// then boundary commands, due captures, due deliveries, estimator updates,
/// due mapping, mission and planning, control and safety, publication.
///
/// The derived order **is** the execution order; the variants must not be
/// reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Propagate,
    Commands,
    Captures,
    Deliveries,
    Estimators,
    Mapping,
    MissionPlanning,
    ControlSafety,
    Publication,
}

impl Phase {
    pub const ORDER: [Phase; 9] = [
        Phase::Propagate,
        Phase::Commands,
        Phase::Captures,
        Phase::Deliveries,
        Phase::Estimators,
        Phase::Mapping,
        Phase::MissionPlanning,
        Phase::ControlSafety,
        Phase::Publication,
    ];
}

/// A stable identity for whatever raised an event: a sensor, a subsystem.
/// Assigned by configuration, never by arrival order, so ties break the same
/// way on every run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub u32);

/// The total order: time, then phase, then source, then sequence.
///
/// Field order is the comparison order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventKey {
    pub time: SimTime,
    pub phase: Phase,
    pub source: SourceId,
    /// Per-queue creation counter: the last tie-break, between events one
    /// source raised for the same time and phase.
    pub seq: u64,
}

/// Where an event landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scheduled {
    pub key: EventKey,
    /// The time it asked for, when that phase had already closed at that
    /// time and it was deferred to `key.time`. §3: *"record its effective
    /// time"*.
    pub requested: Option<SimTime>,
}

/// Events in [`EventKey`] order, popped one at a time.
///
/// Popping an event closes every earlier `(time, phase)`: an event pushed for
/// a closed slot -- its phase has already run at its time, or its time has
/// passed -- is deferred to the same phase at the next physics boundary after
/// the current time, never run out of order.
#[derive(Debug)]
pub struct EventQueue<T> {
    events: BTreeMap<EventKey, T>,
    boundaries: Schedule,
    epoch: SimTime,
    seq: u64,
    /// The `(time, phase)` of the last event popped.
    cursor: Option<(SimTime, Phase)>,
}

impl<T> EventQueue<T> {
    /// A queue whose deferred events go to the next of `physics_step_ns`'s
    /// boundaries from `epoch`.
    pub fn new(epoch: SimTime, physics_step_ns: NonZeroU64) -> Self {
        Self {
            events: BTreeMap::new(),
            boundaries: Schedule::Period(physics_step_ns),
            epoch,
            seq: 0,
            cursor: None,
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The `(time, phase)` the queue has executed up to, if anything.
    pub fn cursor(&self) -> Option<(SimTime, Phase)> {
        self.cursor
    }

    /// Queue `payload` for `phase` at `time`, from `source`.
    pub fn push(
        &mut self,
        time: SimTime,
        phase: Phase,
        source: SourceId,
        payload: T,
    ) -> Result<Scheduled, TimeError> {
        let (effective, requested) = match self.cursor {
            Some(closed) if (time, phase) <= closed => {
                let n = self.boundaries.first_at_or_after(self.epoch, closed.0)?;
                let mut next = self.boundaries.nth_deadline(self.epoch, n)?;
                if next <= closed.0 {
                    next = self.boundaries.nth_deadline(self.epoch, n + 1)?;
                }
                (next, Some(time))
            }
            _ => (time, None),
        };
        let key = EventKey {
            time: effective,
            phase,
            source,
            seq: self.seq,
        };
        self.seq += 1;
        self.events.insert(key, payload);
        Ok(Scheduled { key, requested })
    }

    /// The next event in order, closing everything before it.
    pub fn pop(&mut self) -> Option<(EventKey, T)> {
        let (key, payload) = self.events.pop_first()?;
        self.cursor = Some((key.time, key.phase));
        Some((key, payload))
    }

    /// The next event's key, without taking it.
    pub fn peek(&self) -> Option<&EventKey> {
        self.events.keys().next()
    }

    /// A new epoch: every remaining event is discarded (§3, *"at reset,
    /// discard the old epoch's remaining events"*), and the count returned so
    /// the caller can report it.
    pub fn reset(&mut self, epoch: SimTime) -> usize {
        let discarded = self.events.len();
        self.events.clear();
        self.epoch = epoch;
        self.seq = 0;
        self.cursor = None;
        discarded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("non-zero")
    }

    fn queue() -> EventQueue<&'static str> {
        EventQueue::new(SimTime::EPOCH, nz(20_000_000))
    }

    fn t(nanos: i64) -> SimTime {
        SimTime::from_nanos(nanos)
    }

    fn drain<T>(queue: &mut EventQueue<T>) -> Vec<T> {
        std::iter::from_fn(|| queue.pop().map(|(_, payload)| payload)).collect()
    }

    /// Equal times break by phase, then source, then creation order --
    /// whatever order they were pushed in.
    #[test]
    fn equal_times_order_by_phase_source_then_sequence() {
        let mut q = queue();
        q.push(t(5), Phase::Publication, SourceId(0), "publish")
            .unwrap();
        q.push(t(5), Phase::Captures, SourceId(2), "camera-b")
            .unwrap();
        q.push(t(5), Phase::Captures, SourceId(1), "camera-a first")
            .unwrap();
        q.push(t(5), Phase::Captures, SourceId(1), "camera-a second")
            .unwrap();
        q.push(t(5), Phase::Propagate, SourceId(9), "physics")
            .unwrap();
        q.push(t(4), Phase::Publication, SourceId(0), "earlier")
            .unwrap();
        assert_eq!(
            drain(&mut q),
            vec![
                "earlier",
                "physics",
                "camera-a first",
                "camera-a second",
                "camera-b",
                "publish"
            ]
        );
    }

    #[test]
    fn the_phase_order_is_the_architectures() {
        let mut sorted = Phase::ORDER;
        sorted.sort();
        assert_eq!(sorted, Phase::ORDER);
    }

    /// An event raised for a phase that has already run at its time goes to
    /// that phase at the next physics boundary, and says so.
    #[test]
    fn an_event_for_a_closed_phase_is_deferred_to_the_next_boundary() {
        let mut q = queue();
        q.push(t(40_000_000), Phase::Estimators, SourceId(0), "estimator")
            .unwrap();
        q.pop().unwrap();
        // While estimators run at 40 ms, something raises a capture for 40 ms:
        // captures have closed.
        let late = q
            .push(t(40_000_000), Phase::Captures, SourceId(3), "late capture")
            .unwrap();
        assert_eq!(late.key.time, t(60_000_000));
        assert_eq!(late.key.phase, Phase::Captures);
        assert_eq!(late.requested, Some(t(40_000_000)));
        // A later phase at the same time is still open.
        let open = q
            .push(t(40_000_000), Phase::Mapping, SourceId(3), "mapping")
            .unwrap();
        assert_eq!((open.key.time, open.requested), (t(40_000_000), None));
        // And a time already in the past is deferred too, never run late.
        let past = q
            .push(t(1), Phase::Publication, SourceId(0), "past")
            .unwrap();
        assert_eq!((past.key.time, past.requested), (t(60_000_000), Some(t(1))));
        assert_eq!(drain(&mut q), vec!["mapping", "late capture", "past"]);
    }

    #[test]
    fn reset_discards_the_old_epochs_events() {
        let mut q = queue();
        q.push(t(10), Phase::Commands, SourceId(0), "a").unwrap();
        q.push(t(20), Phase::Commands, SourceId(0), "b").unwrap();
        q.pop().unwrap();
        assert_eq!(q.reset(SimTime::EPOCH), 1);
        assert!(q.is_empty());
        assert_eq!(q.cursor(), None);
        let fresh = q.push(t(0), Phase::Commands, SourceId(0), "c").unwrap();
        assert_eq!(fresh.key.seq, 0);
    }

    /// VERIFICATION_AND_ROLLOUT.md's mixed-rate fixture: 200, 30 and 5 Hz
    /// sources over one virtual second, merged into one order. Every
    /// occurrence appears once, in time order, with the slower source never
    /// pre-empting a faster one at a shared instant (source order breaks it).
    #[test]
    fn mixed_200_30_5_hz_sources_merge_in_one_stable_order() {
        let sources = [
            (SourceId(0), Schedule::rational(nz(200), nz(1)).unwrap()),
            (SourceId(1), Schedule::rational(nz(30), nz(1)).unwrap()),
            (SourceId(2), Schedule::rational(nz(5), nz(1)).unwrap()),
        ];
        let end = SimTime::from_nanos(1_000_000_000);
        let mut q: EventQueue<(SourceId, u64)> = EventQueue::new(SimTime::EPOCH, nz(5_000_000));
        // Pushed slowest first, to show push order does not matter.
        for (source, schedule) in sources.iter().rev() {
            let mut n = 0;
            while let Ok(time) = schedule.nth_deadline(SimTime::EPOCH, n) {
                if time >= end {
                    break;
                }
                q.push(time, Phase::Captures, *source, (*source, n))
                    .unwrap();
                n += 1;
            }
        }
        let mut counts = [0usize; 3];
        let mut last: Option<EventKey> = None;
        while let Some((key, (source, _))) = q.pop() {
            counts[source.0 as usize] += 1;
            if let Some(previous) = last {
                assert!(previous < key);
            }
            last = Some(key);
        }
        assert_eq!(counts, [200, 30, 5]);
    }
}
