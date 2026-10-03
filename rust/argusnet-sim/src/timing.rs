//! Per-stage wall time (P4.10-pre).
//!
//! The scheduler and parallel work in P4.10 are to be chosen from a measured
//! Rust profile, not from the Python-era one. This is that measurement: how
//! long each stage of each tick took on the machine that ran it.
//!
//! Wall time is not causal state. It lives outside [`StageTrace`], the replay
//! and the checkpoint, so turning it on cannot change a deterministic result,
//! and it is off unless a caller asks for it, so a long interactive run does not
//! grow it without limit.
//!
//! [`StageTrace`]: crate::stage::StageTrace

use std::time::Duration;

use serde::Serialize;

use crate::stage::Stage;

/// Every stage's wall time, one sample per tick it ran.
#[derive(Debug, Clone)]
pub struct StageTimings {
    /// Nanoseconds, indexed by [`Stage::order`].
    samples: [Vec<u64>; Stage::PIPELINE.len()],
}

impl Default for StageTimings {
    fn default() -> Self {
        Self {
            samples: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// Distribution of one stage's samples, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TimingSummary {
    pub count: usize,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl StageTimings {
    pub fn record(&mut self, stage: Stage, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.samples[usize::from(stage.order())].push(nanos);
    }

    /// One stage's samples, in tick order.
    pub fn samples_ns(&self, stage: Stage) -> &[u64] {
        &self.samples[usize::from(stage.order())]
    }

    /// One stage's distribution, or `None` if it never ran.
    pub fn summary(&self, stage: Stage) -> Option<TimingSummary> {
        let samples = self.samples_ns(stage);
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let total: u64 = sorted.iter().sum();
        // Nearest rank, as `argusnet-rs`'s tick summary computes it.
        let at =
            |fraction: f64| ms(sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]);
        Some(TimingSummary {
            count: sorted.len(),
            total_ms: ms(total),
            mean_ms: ms(total) / sorted.len() as f64,
            p50_ms: at(0.5),
            p95_ms: at(0.95),
            p99_ms: at(0.99),
            max_ms: ms(*sorted.last().expect("non-empty")),
        })
    }

    /// Every stage that ran, in pipeline order.
    pub fn summaries(&self) -> Vec<(Stage, TimingSummary)> {
        Stage::PIPELINE
            .into_iter()
            .filter_map(|stage| self.summary(stage).map(|summary| (stage, summary)))
            .collect()
    }

    pub fn clear(&mut self) {
        self.samples.iter_mut().for_each(Vec::clear);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_that_never_ran_has_no_summary() {
        let timings = StageTimings::default();
        assert!(timings.summary(Stage::Deconflict).is_none());
        assert!(timings.summaries().is_empty());
    }

    #[test]
    fn summaries_are_per_stage_and_in_pipeline_order() {
        let mut timings = StageTimings::default();
        for micros in [400, 100, 300, 200] {
            timings.record(Stage::UpdateFusion, Duration::from_micros(micros));
        }
        timings.record(Stage::ApplyCommands, Duration::from_micros(10));

        let fusion = timings.summary(Stage::UpdateFusion).expect("ran");
        assert_eq!(fusion.count, 4);
        assert!((fusion.total_ms - 1.0).abs() < 1e-12);
        assert!((fusion.mean_ms - 0.25).abs() < 1e-12);
        assert!((fusion.max_ms - 0.4).abs() < 1e-12);
        // Nearest rank over [0.1, 0.2, 0.3, 0.4]: index round(3 * 0.5) = 2.
        assert!((fusion.p50_ms - 0.3).abs() < 1e-12);
        assert_eq!(timings.samples_ns(Stage::UpdateFusion)[0], 400_000);

        let order: Vec<Stage> = timings.summaries().into_iter().map(|(s, _)| s).collect();
        assert_eq!(order, vec![Stage::ApplyCommands, Stage::UpdateFusion]);

        timings.clear();
        assert!(timings.summaries().is_empty());
    }
}
