//! Mutable coverage state with O(1) statistics.
//!
//! P4.6 asks to "replace repeated full-grid stats with revision-aware
//! aggregates" and to "prove incremental totals against full reductions,
//! including reset and removal".
//!
//! # Why this is worth doing
//!
//! `CoverageStats` in Python recomputes two whole-grid reductions on every
//! access -- `count > 0` summed, and `count` summed. The simulation reads it
//! from inside a doubly nested loop over mission zones and mobile platforms:
//!
//! ```text
//! for zone in scenario.mission_zones:
//!     for ns in mobile_states:
//!         if inside:
//!             zmap.mark_circular(...)
//!         zcov = zmap.stats.coverage_fraction     # <- outside the `if`
//! ```
//!
//! The read sits *outside* the `if inside:` that guards the write, so most of
//! those reductions recompute a value that cannot have changed. Measured on
//! this repository: 14.3 us per `stats` on the default 86x86 grid, 33.4 us at
//! 172x172, and 180.9 us at 430x430 -- multiplied by zones x platforms, every
//! tick.
//!
//! Maintaining the two totals as cells are marked makes the read O(1). The
//! aggregates are exact, not approximate: every mutation updates them, and
//! `full_reduction_stats` recomputes from the grid so tests can assert the two
//! agree. That equality is the whole safety argument, so it is asserted across
//! randomised operation sequences rather than at a few hand-picked points.

use crate::footprint::{rasterize_circle, GridBounds, Sampling};

/// Coverage totals. Identical in meaning to Python's `CoverageStats`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageStats {
    pub total_cells: u64,
    pub covered_cells: u64,
    /// Fraction of cells observed at least once, in [0, 1].
    pub coverage_fraction: f64,
    /// Mean observations per *covered* cell -- not per cell. Zero when nothing
    /// is covered, matching the reference rather than producing a NaN.
    pub mean_revisits: f64,
}

/// A grid of per-cell observation counts.
///
/// Counts are `u32` as Python's are `int32`, but saturating rather than
/// wrapping: a cell observed 2^32 times is implausible, and silently wrapping
/// to zero would turn a heavily covered cell into an uncovered one.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoverageGrid {
    bounds: GridBounds,
    nx: usize,
    ny: usize,
    counts: Vec<u32>,
    covered_cells: u64,
    total_observations: u64,
    revision: u64,
}

impl CoverageGrid {
    pub fn new(bounds: GridBounds) -> Result<Self, &'static str> {
        let (nx, ny) = bounds.dimensions()?;
        let cells = nx.checked_mul(ny).ok_or("grid too large")?;
        Ok(Self {
            bounds,
            nx,
            ny,
            counts: vec![0; cells],
            covered_cells: 0,
            total_observations: 0,
            revision: 0,
        })
    }

    pub fn bounds(&self) -> GridBounds {
        self.bounds
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Bumped on every mutation that changes the grid.
    ///
    /// A caller that reads coverage repeatedly between writes -- the zone loop
    /// above -- can compare this instead of recomputing, and a cache keyed on
    /// it cannot go stale silently.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// O(1). Reads the maintained aggregates; touches no cell.
    pub fn stats(&self) -> CoverageStats {
        let total_cells = self.counts.len() as u64;
        CoverageStats {
            total_cells,
            covered_cells: self.covered_cells,
            coverage_fraction: if total_cells > 0 {
                self.covered_cells as f64 / total_cells as f64
            } else {
                0.0
            },
            mean_revisits: if self.covered_cells > 0 {
                self.total_observations as f64 / self.covered_cells as f64
            } else {
                0.0
            },
        }
    }

    /// Recompute the same statistics by scanning every cell.
    ///
    /// This is what the maintained aggregates must always equal. It exists so
    /// the equality is testable; nothing on the hot path should call it.
    pub fn full_reduction_stats(&self) -> CoverageStats {
        let covered = self.counts.iter().filter(|&&c| c > 0).count() as u64;
        let observations: u64 = self.counts.iter().map(|&c| u64::from(c)).sum();
        let total_cells = self.counts.len() as u64;
        CoverageStats {
            total_cells,
            covered_cells: covered,
            coverage_fraction: if total_cells > 0 {
                covered as f64 / total_cells as f64
            } else {
                0.0
            },
            mean_revisits: if covered > 0 {
                observations as f64 / covered as f64
            } else {
                0.0
            },
        }
    }

    pub fn count_at(&self, i: usize, j: usize) -> u32 {
        self.counts[i * self.ny + j]
    }

    /// Increment one cell, maintaining both aggregates.
    ///
    /// `covered_cells` rises only on the 0 -> 1 transition, which is the whole
    /// trick: counting it on every hit would make it a hit counter rather than
    /// a cell counter, and the two agree only while nothing is ever revisited.
    pub fn mark_cell(&mut self, i: usize, j: usize) {
        let count = &mut self.counts[i * self.ny + j];
        if *count == 0 {
            self.covered_cells += 1;
        }
        if *count < u32::MAX {
            *count += 1;
            self.total_observations += 1;
        }
        self.revision += 1;
    }

    /// Increment every cell in a circular footprint.
    ///
    /// Hits arrive with multiplicity from the rasterizer, and each is applied,
    /// because a cell the lattice folds onto twice is incremented twice by the
    /// reference's `np.add.at`.
    pub fn mark_circular(
        &mut self,
        center_xy: [f64; 2],
        radius_m: f64,
        sampling: Sampling,
    ) -> Result<(), &'static str> {
        for (i, j) in rasterize_circle(&self.bounds, center_xy, radius_m, sampling)? {
            self.mark_cell(i, j);
        }
        Ok(())
    }

    /// Clear one cell, maintaining both aggregates.
    ///
    /// P4.6 names removal explicitly. Nothing in the reference removes
    /// coverage today, which is exactly why the aggregates need to handle it
    /// before something does: an aggregate that is only ever incremented is
    /// trivially correct and proves nothing.
    pub fn clear_cell(&mut self, i: usize, j: usize) {
        let count = &mut self.counts[i * self.ny + j];
        if *count > 0 {
            self.total_observations -= u64::from(*count);
            self.covered_cells -= 1;
            *count = 0;
            self.revision += 1;
        }
    }

    /// Drop all coverage, aggregates included.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.covered_cells = 0;
        self.total_observations = 0;
        self.revision += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> GridBounds {
        GridBounds {
            x_min_m: 0.0,
            x_max_m: 1000.0,
            y_min_m: 0.0,
            y_max_m: 1000.0,
            resolution_m: 50.0,
        }
    }

    /// The safety argument, stated as a property: after any sequence of marks,
    /// clears and resets, the maintained aggregates equal a full reduction.
    ///
    /// Randomised rather than hand-picked, because the interesting failures are
    /// in the transitions -- a revisit that double-counts a covered cell, a
    /// clear that forgets its observations -- and those need revisits and
    /// clears of already-covered cells to occur, which a tidy scripted sequence
    /// tends to avoid.
    #[test]
    fn maintained_aggregates_always_equal_a_full_reduction() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        let (nx, ny) = grid.dimensions();
        // A small deterministic LCG: no dev-dependency, and a failure is
        // reproducible from the seed alone.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for step in 0..4000 {
            match next() % 10 {
                0..=6 => {
                    // Marks concentrated in a small area, so revisits are
                    // common rather than rare.
                    let i = (next() as usize) % nx.min(6);
                    let j = (next() as usize) % ny.min(6);
                    grid.mark_cell(i, j);
                }
                7 | 8 => {
                    let i = (next() as usize) % nx.min(6);
                    let j = (next() as usize) % ny.min(6);
                    grid.clear_cell(i, j);
                }
                _ => {
                    let cx = (next() % 1000) as f64;
                    let cy = (next() % 1000) as f64;
                    grid.mark_circular([cx, cy], 120.0, Sampling::WorldAnchored)
                        .unwrap();
                }
            }
            if step % 37 == 0 {
                assert_eq!(
                    grid.stats(),
                    grid.full_reduction_stats(),
                    "aggregates diverged at step {step}"
                );
            }
        }
        assert_eq!(grid.stats(), grid.full_reduction_stats());

        grid.reset();
        assert_eq!(grid.stats(), grid.full_reduction_stats());
        assert_eq!(grid.stats().covered_cells, 0);
        assert_eq!(grid.stats().mean_revisits, 0.0);

        // Keep going AFTER the reset. Checking only the empty state hides a
        // stale observation total: with no covered cells `mean_revisits`
        // short-circuits to zero whatever the total holds, so the corruption
        // is invisible until the next mark. A reset that cleared
        // `covered_cells` but not `total_observations` survived this test
        // until these lines existed.
        grid.mark_cell(2, 3);
        assert_eq!(
            grid.stats(),
            grid.full_reduction_stats(),
            "a stale total from before the reset leaked into the first mark after it"
        );
        assert_eq!(grid.stats().mean_revisits, 1.0);
        for step in 0..500 {
            let i = (next() as usize) % nx.min(6);
            let j = (next() as usize) % ny.min(6);
            if step % 5 == 0 {
                grid.clear_cell(i, j);
            } else {
                grid.mark_cell(i, j);
            }
        }
        assert_eq!(grid.stats(), grid.full_reduction_stats());
    }

    /// A revisited cell raises the observation total but not the covered count.
    /// Conflating the two is the obvious way to write this wrong, and it stays
    /// invisible until something is observed twice.
    #[test]
    fn a_revisit_raises_revisits_but_not_coverage() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        grid.mark_cell(3, 4);
        let once = grid.stats();
        assert_eq!(once.covered_cells, 1);
        assert_eq!(once.mean_revisits, 1.0);

        grid.mark_cell(3, 4);
        let twice = grid.stats();
        assert_eq!(twice.covered_cells, 1, "the same cell is still one cell");
        assert_eq!(twice.mean_revisits, 2.0);
        assert_eq!(twice, grid.full_reduction_stats());
    }

    /// Clearing a cell must give back every observation it held, not one.
    #[test]
    fn clearing_a_cell_returns_all_of_its_observations() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        for _ in 0..5 {
            grid.mark_cell(1, 1);
        }
        grid.mark_cell(2, 2);
        assert_eq!(grid.stats().mean_revisits, 3.0); // 6 observations over 2 cells

        grid.clear_cell(1, 1);
        let after = grid.stats();
        assert_eq!(after.covered_cells, 1);
        assert_eq!(
            after.mean_revisits, 1.0,
            "all five must be returned, not one"
        );
        assert_eq!(after, grid.full_reduction_stats());
    }

    /// Clearing a cell that was never covered changes nothing -- including the
    /// revision, so a cache keyed on it is not invalidated for no reason.
    #[test]
    fn clearing_an_uncovered_cell_is_a_no_op() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        grid.mark_cell(0, 0);
        let before = grid.stats();
        let revision = grid.revision();
        grid.clear_cell(7, 7);
        assert_eq!(grid.stats(), before);
        assert_eq!(
            grid.revision(),
            revision,
            "a no-op must not bump the revision"
        );
    }

    /// The revision is what lets the zone loop skip recomputing between writes.
    #[test]
    fn the_revision_moves_only_on_real_mutations() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        let start = grid.revision();
        let _ = grid.stats();
        let _ = grid.count_at(0, 0);
        assert_eq!(grid.revision(), start, "reads must not bump the revision");

        grid.mark_cell(0, 0);
        assert!(grid.revision() > start);
        let after_mark = grid.revision();
        grid.reset();
        assert!(grid.revision() > after_mark, "reset is a mutation");
    }

    /// An empty grid reports zeroes rather than dividing by zero.
    #[test]
    fn an_empty_grid_reports_zeroes_not_nan() {
        let grid = CoverageGrid::new(bounds()).unwrap();
        let stats = grid.stats();
        assert_eq!(stats.covered_cells, 0);
        assert_eq!(stats.coverage_fraction, 0.0);
        assert_eq!(stats.mean_revisits, 0.0);
        assert!(stats.mean_revisits.is_finite());
        assert_eq!(stats, grid.full_reduction_stats());
    }

    /// Counts saturate rather than wrapping. A wrap would take a heavily
    /// observed cell to zero, turning it uncovered.
    #[test]
    fn counts_saturate_rather_than_wrapping() {
        let mut grid = CoverageGrid::new(bounds()).unwrap();
        grid.mark_cell(0, 0);
        // Drive the cell to the ceiling directly; marking 2^32 times is not a
        // test anyone should wait for.
        let index = 0;
        grid.counts[index] = u32::MAX;
        grid.total_observations = u64::from(u32::MAX);
        grid.mark_cell(0, 0);
        assert_eq!(grid.count_at(0, 0), u32::MAX, "must not wrap to zero");
        assert_eq!(grid.stats().covered_cells, 1);
        assert_eq!(grid.stats(), grid.full_reduction_stats());
    }

    /// Marking a footprint applies every hit the rasterizer reports, including
    /// a cell the lattice folds onto twice at the far edge.
    #[test]
    fn a_footprint_applies_hits_with_multiplicity() {
        let regional = GridBounds {
            x_min_m: 0.0,
            x_max_m: 4300.0,
            y_min_m: 0.0,
            y_max_m: 4100.0,
            resolution_m: 50.0,
        };
        let mut grid = CoverageGrid::new(regional).unwrap();
        // The far-edge case from the rasterizer fixture, which carries folds.
        grid.mark_circular([4000.0, 3800.0], 350.0, Sampling::WorldAnchored)
            .unwrap();
        let stats = grid.stats();
        assert_eq!(stats, grid.full_reduction_stats());
        assert!(
            stats.mean_revisits > 1.0,
            "the folded cells must push mean revisits above one, got {}",
            stats.mean_revisits
        );
    }
}
