//! Work that takes longer than a tick (P4.10d): submitted on one, applied on a
//! later one declared when it is submitted, on the session's own workers.
//!
//! What a run does must not depend on how fast a machine is. So a job's result
//! is taken at its declared tick and no other: one finished early waits, and one
//! not finished is waited for. Results due together are taken in the order their
//! jobs were submitted, whatever order the workers finished in. A job is a pure
//! function of what it captured when submitted, and it names the revision of its
//! inputs: whoever takes the result judges whether that is still good enough, and
//! what is safe now is checked now ([`Done::input_revision`]).
//!
//! The board is bounded. A submission past its capacity is refused, not queued:
//! the caller keeps what it had.
//!
//! A checkpoint holds the pending jobs' *results* (P4.10e). They are finished
//! first, which changes nothing: their inputs are fixed and their ticks are
//! still to come. So a resumed run needs none of the inputs, and applies the same
//! results at the same ticks.

use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Why a job was not taken on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The board holds as many as it may.
    Full,
    /// One for this key is already pending.
    Pending,
    /// A job cannot be applied on the tick it is submitted, or before.
    Due,
}

/// A finished job, taken at its tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Done<T> {
    pub key: String,
    /// In order of submission over the run.
    pub sequence: u64,
    pub submitted_tick: u64,
    pub due_tick: u64,
    /// The revision of what it was worked out from.
    pub input_revision: u64,
    pub value: T,
}

enum Value<T> {
    Ready(T),
    Running(Receiver<T>),
}

struct Pending<T> {
    sequence: u64,
    submitted_tick: u64,
    due_tick: u64,
    input_revision: u64,
    value: Value<T>,
}

/// What a board checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobState<T> {
    pub next_sequence: u64,
    pub pending: Vec<Done<T>>,
}

impl<T> JobState<T> {
    /// A board that has had no job: nothing to write.
    pub fn is_new(&self) -> bool {
        self.next_sequence == 0 && self.pending.is_empty()
    }
}

/// The pending jobs of one kind, by key.
pub struct JobBoard<T> {
    capacity: usize,
    workers: Option<Arc<rayon::ThreadPool>>,
    next_sequence: u64,
    pending: BTreeMap<String, Pending<T>>,
}

impl<T> std::fmt::Debug for JobBoard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobBoard")
            .field("capacity", &self.capacity)
            .field("pending", &self.pending.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl<T: Send + Clone + 'static> JobBoard<T> {
    /// A board of at most `capacity` jobs, worked out on the calling thread.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            workers: None,
            next_sequence: 0,
            pending: BTreeMap::new(),
        }
    }

    /// With `workers` threads of its own; one is the calling thread.
    pub fn with_workers(mut self, workers: usize) -> Result<Self, String> {
        self.workers = if workers > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .thread_name(|index| format!("argusnet-job-{index}"))
                    .build()
                    .map_err(|error| format!("job pool: {error}"))?,
            ))
        } else {
            None
        };
        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn is_pending(&self, key: &str) -> bool {
        self.pending.contains_key(key)
    }

    /// Take on `work` for `key`, to be applied at `due_tick`.
    pub fn submit(
        &mut self,
        key: &str,
        tick: u64,
        due_tick: u64,
        input_revision: u64,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<u64, Refused> {
        if due_tick <= tick {
            return Err(Refused::Due);
        }
        if self.pending.contains_key(key) {
            return Err(Refused::Pending);
        }
        if self.pending.len() >= self.capacity {
            return Err(Refused::Full);
        }
        let value = match &self.workers {
            None => Value::Ready(work()),
            Some(pool) => {
                let (send, receive) = std::sync::mpsc::sync_channel(1);
                pool.spawn(move || {
                    // A cancelled job's result has nobody to take it.
                    let _ = send.send(work());
                });
                Value::Running(receive)
            }
        };
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.pending.insert(
            key.to_string(),
            Pending {
                sequence,
                submitted_tick: tick,
                due_tick,
                input_revision,
                value,
            },
        );
        Ok(sequence)
    }

    /// Drop `key`'s job, finished or not. True when there was one.
    pub fn cancel(&mut self, key: &str) -> bool {
        self.pending.remove(key).is_some()
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.next_sequence = 0;
    }

    /// The jobs due by `tick`, in the order they were submitted. Waits for any
    /// not finished.
    pub fn take_due(&mut self, tick: u64) -> Vec<Done<T>> {
        let keys: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, job)| job.due_tick <= tick)
            .map(|(key, _)| key.clone())
            .collect();
        let mut done: Vec<Done<T>> = keys
            .into_iter()
            .map(|key| {
                let job = self.pending.remove(&key).expect("just listed");
                finished(key, job)
            })
            .collect();
        done.sort_by_key(|job| job.sequence);
        done
    }

    /// How many are finished, without waiting. Says nothing a run may act on.
    pub fn finished_now(&mut self) -> usize {
        let mut count = 0;
        for job in self.pending.values_mut() {
            if let Value::Running(receive) = &job.value {
                match receive.try_recv() {
                    Ok(value) => job.value = Value::Ready(value),
                    Err(TryRecvError::Empty) => continue,
                    Err(TryRecvError::Disconnected) => panic!("a job's worker died"),
                }
            }
            count += 1;
        }
        count
    }

    /// The board as it stands, every pending job finished.
    pub fn state(&mut self) -> JobState<T> {
        let mut pending: Vec<Done<T>> = Vec::new();
        for (key, job) in &mut self.pending {
            let value = match &job.value {
                Value::Ready(value) => value.clone(),
                Value::Running(receive) => receive.recv().expect("a job's worker died"),
            };
            job.value = Value::Ready(value.clone());
            pending.push(Done {
                key: key.clone(),
                sequence: job.sequence,
                submitted_tick: job.submitted_tick,
                due_tick: job.due_tick,
                input_revision: job.input_revision,
                value,
            });
        }
        pending.sort_by_key(|job| job.sequence);
        JobState {
            next_sequence: self.next_sequence,
            pending,
        }
    }

    /// Refuse a state this board could not have left.
    pub fn check(&self, state: &JobState<T>) -> Result<(), String> {
        if state.pending.len() > self.capacity {
            return Err(format!(
                "{} pending jobs on a board of {}",
                state.pending.len(),
                self.capacity
            ));
        }
        let mut keys = std::collections::BTreeSet::new();
        for job in &state.pending {
            if job.sequence >= state.next_sequence
                || job.due_tick <= job.submitted_tick
                || !keys.insert(&job.key)
            {
                return Err(format!("pending job {:?} is malformed", job.key));
            }
        }
        Ok(())
    }

    pub fn restore(&mut self, state: JobState<T>) {
        self.next_sequence = state.next_sequence;
        self.pending = state
            .pending
            .into_iter()
            .map(|job| {
                (
                    job.key,
                    Pending {
                        sequence: job.sequence,
                        submitted_tick: job.submitted_tick,
                        due_tick: job.due_tick,
                        input_revision: job.input_revision,
                        value: Value::Ready(job.value),
                    },
                )
            })
            .collect();
    }
}

fn finished<T>(key: String, job: Pending<T>) -> Done<T> {
    let value = match job.value {
        Value::Ready(value) => value,
        Value::Running(receive) => receive.recv().expect("a job's worker died"),
    };
    Done {
        key,
        sequence: job.sequence,
        submitted_tick: job.submitted_tick,
        due_tick: job.due_tick,
        input_revision: job.input_revision,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn slow(value: u32, ms: u64) -> impl FnOnce() -> u32 + Send + 'static {
        move || {
            std::thread::sleep(Duration::from_millis(ms));
            value
        }
    }

    fn values(done: Vec<Done<u32>>) -> Vec<u32> {
        done.into_iter().map(|job| job.value).collect()
    }

    /// The first submitted is the last finished, and still the first taken,
    /// on its tick and not before, whatever the workers.
    #[test]
    fn results_are_taken_at_their_tick_in_the_order_submitted() {
        for workers in [1, 2, 4] {
            let mut board = JobBoard::new(8).with_workers(workers).unwrap();
            board.submit("z", 10, 12, 1, slow(1, 60)).unwrap();
            board.submit("a", 10, 12, 1, slow(2, 1)).unwrap();
            board.submit("m", 10, 14, 1, slow(3, 1)).unwrap();
            assert!(board.take_due(11).is_empty(), "{workers} workers");
            assert_eq!(values(board.take_due(12)), [1, 2], "{workers} workers");
            assert!(board.take_due(13).is_empty());
            assert_eq!(values(board.take_due(20)), [3]);
            assert!(board.is_empty());
        }
    }

    #[test]
    fn the_board_is_bounded_and_one_key_has_one_job() {
        let mut board = JobBoard::new(2);
        assert_eq!(board.submit("a", 0, 0, 0, || 1), Err(Refused::Due));
        board.submit("a", 0, 5, 0, || 1).unwrap();
        assert_eq!(board.submit("a", 1, 6, 0, || 2), Err(Refused::Pending));
        board.submit("b", 0, 5, 0, || 2).unwrap();
        assert_eq!(board.submit("c", 0, 5, 0, || 3), Err(Refused::Full));
        assert!(board.cancel("a"));
        assert!(!board.cancel("a"));
        board.submit("c", 0, 5, 0, || 3).unwrap();
        assert_eq!(values(board.take_due(5)), [2, 3]);
    }

    #[test]
    fn a_cancelled_job_s_result_is_never_taken() {
        let mut board = JobBoard::new(4).with_workers(2).unwrap();
        board.submit("a", 0, 2, 0, slow(1, 20)).unwrap();
        board.cancel("a");
        board.submit("a", 0, 2, 0, slow(2, 1)).unwrap();
        assert_eq!(values(board.take_due(2)), [2]);
    }

    /// A board restored from a checkpoint taken while its jobs ran applies the
    /// same results at the same ticks, and goes on numbering as it would have.
    #[test]
    fn a_board_resumes_with_its_jobs_pending() {
        let make = || JobBoard::new(4).with_workers(2).unwrap();
        let fill = |board: &mut JobBoard<u32>| {
            board.submit("z", 3, 9, 7, slow(1, 30)).unwrap();
            board.submit("a", 4, 6, 8, slow(2, 30)).unwrap();
        };
        let mut straight = make();
        fill(&mut straight);
        let mut stopped = make();
        fill(&mut stopped);
        let state = stopped.state();
        let text = serde_json::to_string(&state).unwrap();
        let state: JobState<u32> = serde_json::from_str(&text).unwrap();
        let mut resumed = make();
        resumed.check(&state).unwrap();
        resumed.restore(state);
        for tick in 5..10 {
            let (a, b) = (straight.take_due(tick), resumed.take_due(tick));
            assert_eq!(a, b, "tick {tick}");
        }
        assert_eq!(
            straight.submit("n", 10, 11, 0, || 0),
            resumed.submit("n", 10, 11, 0, || 0)
        );
        // The one stopped went on as if it had not been.
        assert_eq!(values(stopped.take_due(9)), [1, 2]);
    }

    #[test]
    fn a_malformed_state_is_refused() {
        let board: JobBoard<u32> = JobBoard::new(1);
        let job = |key: &str, sequence, due_tick| Done {
            key: key.into(),
            sequence,
            submitted_tick: 5,
            due_tick,
            input_revision: 0,
            value: 0,
        };
        let state = |pending| JobState {
            next_sequence: 2,
            pending,
        };
        assert!(board.check(&state(vec![job("a", 0, 6)])).is_ok());
        assert!(board.check(&state(vec![job("a", 2, 6)])).is_err());
        assert!(board.check(&state(vec![job("a", 0, 5)])).is_err());
        let two = state(vec![job("a", 0, 6), job("b", 1, 6)]);
        assert!(board.check(&two).is_err(), "more than it holds");
        let roomy: JobBoard<u32> = JobBoard::new(4);
        let twice = state(vec![job("a", 0, 6), job("a", 1, 6)]);
        assert!(roomy.check(&twice).is_err());
    }
}
