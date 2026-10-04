//! Fork-join inside a phase (B3, P4.10a.2; PARALLEL_TICK_PLAN.md).
//!
//! A [`ComputePool`] is owned by the session that runs the simulation: a
//! dedicated, bounded set of workers, never a process-global pool, so two
//! sessions in one process cannot steal each other's threads and a session's
//! worker count is part of its configuration. `workers = 1` runs the very same
//! code serially on the calling thread.
//!
//! The one primitive is [`ComputePool::map_ordered`]: results come back in
//! **input order**, whatever order the workers finish in. Anything that
//! combines them (pushes into a replay, sums floating-point values, steps a
//! stateful stream) does so afterwards, serially, in that order. There is no
//! unordered reduction here, because floating-point addition is not
//! associative and an unordered sum would make a run's output depend on
//! thread timing.
//!
//! Below [`PARALLEL_THRESHOLD`] items the serial path is taken even with
//! workers: handing a small fleet's handful of pairs to threads costs more
//! than the work.

use rayon::prelude::*;

/// Fewest items worth splitting across workers.
pub const PARALLEL_THRESHOLD: usize = 8;

/// The session's workers.
pub struct ComputePool {
    workers: usize,
    pool: Option<rayon::ThreadPool>,
}

impl std::fmt::Debug for ComputePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePool")
            .field("workers", &self.workers)
            .finish()
    }
}

impl Default for ComputePool {
    fn default() -> Self {
        Self::serial()
    }
}

impl ComputePool {
    /// One worker: the calling thread, no pool.
    pub fn serial() -> Self {
        Self {
            workers: 1,
            pool: None,
        }
    }

    /// `workers` threads of its own (at least one; one means serial).
    pub fn new(workers: usize) -> Result<Self, String> {
        let workers = workers.max(1);
        if workers == 1 {
            return Ok(Self::serial());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("argusnet-compute-{index}"))
            .build()
            .map_err(|error| format!("compute pool: {error}"))?;
        Ok(Self {
            workers,
            pool: Some(pool),
        })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// `f` applied to every item, results in input order.
    pub fn map_ordered<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send,
    {
        self.map_ordered_from(PARALLEL_THRESHOLD, items, f)
    }

    /// [`map_ordered`](Self::map_ordered) for heavier items: forked from
    /// `min_items` up rather than from [`PARALLEL_THRESHOLD`]. For tasks
    /// that each cost tens of microseconds (a platform's particle scoring),
    /// two items are already worth a thread; for sub-microsecond ones they
    /// are not.
    pub fn map_ordered_from<T, R, F>(&self, min_items: usize, items: &[T], f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send,
    {
        match &self.pool {
            Some(pool) if items.len() >= min_items.max(2) => {
                // `collect` on an indexed parallel iterator places each result
                // at its input's index, whatever order the tasks finish in.
                pool.install(|| items.par_iter().map(&f).collect())
            }
            _ => items.iter().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn results_come_back_in_input_order_whatever_finishes_first() {
        let pool = ComputePool::new(4).unwrap();
        let items: Vec<u64> = (0..32).collect();
        // Earlier items sleep longer, so they finish last.
        let out = pool.map_ordered(&items, |i| {
            std::thread::sleep(Duration::from_micros((32 - i) * 200));
            i * 10
        });
        assert_eq!(out, items.iter().map(|i| i * 10).collect::<Vec<_>>());
    }

    #[test]
    fn the_work_really_runs_on_the_pool_threads() {
        let pool = ComputePool::new(3).unwrap();
        let items: Vec<u32> = (0..64).collect();
        let names = pool.map_ordered(&items, |_| {
            std::thread::current().name().unwrap_or("").to_string()
        });
        assert!(names
            .iter()
            .all(|name| name.starts_with("argusnet-compute-")));
        // Below the threshold the caller's thread does it.
        let small = pool.map_ordered(&items[..3], |_| {
            std::thread::current().name().unwrap_or("").to_string()
        });
        assert!(small
            .iter()
            .all(|name| !name.starts_with("argusnet-compute-")));
    }

    #[test]
    fn one_worker_is_the_serial_path() {
        let pool = ComputePool::new(1).unwrap();
        assert_eq!(pool.workers(), 1);
        let items: Vec<i32> = (0..100).collect();
        assert_eq!(
            pool.map_ordered(&items, |i| i + 1),
            (1..101).collect::<Vec<_>>()
        );
    }

    /// Why combining happens afterwards, in order: summed in another order,
    /// these values give a different total. Ordered results summed in input
    /// order give the same total at every worker count.
    #[test]
    fn combining_in_input_order_is_independent_of_the_worker_count() {
        // 1e16 + 1 rounds back to 1e16, so where the small terms land
        // relative to the large ones decides the total.
        let values: Vec<f64> = [1e16, 1.0, -1e16, 1.0].repeat(50);
        let forward: f64 = values.iter().sum();
        let reversed: f64 = values.iter().rev().sum();
        assert_ne!(
            forward.to_bits(),
            reversed.to_bits(),
            "order matters for these values"
        );
        for workers in [1, 2, 4] {
            let pool = ComputePool::new(workers).unwrap();
            let mapped = pool.map_ordered(&values, |v| *v);
            let total: f64 = mapped.iter().sum();
            assert_eq!(total.to_bits(), forward.to_bits(), "{workers} workers");
        }
    }
}
