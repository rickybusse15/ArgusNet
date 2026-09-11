//! Frontier-based exploration: which uncovered cell a platform flies toward,
//! and which uncovered regions are enclosed holes rather than open frontier.
//!
//! Takes the coverage grid as a plain view rather than depending on the
//! mapping crate. The selection is a pure function over counts, bounds and
//! claims, so keeping it that way makes the tie-breaking testable and avoids
//! a dependency on work that is still in flight.
//!
//! # Two behaviours that decide the answer
//!
//! **Ties go to the lowest `(i, j)`.** The reference scores every candidate
//! and takes `np.argmax`, which returns the FIRST maximum, over candidates
//! from `np.where(grid == 0)` — row-major, so `i` ascending then `j`. At low
//! coverage the gradient term is off and equidistant cells score identically,
//! which makes ties the normal case rather than an edge case. A port that
//! kept the last maximum, or that walked the grid column-major, would send
//! platforms somewhere else entirely.
//!
//! **The neighbour count is an integer here and is not in the reference.**
//! `uniform_filter(mask, size=3) * 9` is a 3x3 mean computed as two separable
//! `1/3` passes, so it lands near the count rather than on it: the recorded
//! fixture contains values like `0.9999999999999991` and `-8.3e-16` where the
//! true counts are 1 and 0. This computes the count directly. The difference
//! can only matter if it flips a tie, so the fixture records how far ahead the
//! winner was in every case and the tests check each margin against the noise
//! — the smallest is 9.2e-2, thirteen orders of magnitude clear.

use std::collections::{HashMap, HashSet, VecDeque};

/// Where a grid sits in the world. Mirrors `mapping.occupancy.GridBounds`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridBounds {
    pub x_min_m: f64,
    pub y_min_m: f64,
    pub resolution_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl GridBounds {
    /// Metres to cell indices, clamped into the grid.
    ///
    /// The reference truncates with `int()` rather than flooring. The two
    /// differ only on `(-1, 0)`, which the clamp sends to 0 either way, but
    /// truncation is what is written so truncation is what is reproduced.
    pub fn xy_to_ij(&self, x_m: f64, y_m: f64) -> (usize, usize) {
        let i = ((x_m - self.x_min_m) / self.resolution_m).trunc() as i64;
        let j = ((y_m - self.y_min_m) / self.resolution_m).trunc() as i64;
        (
            i.clamp(0, self.nx as i64 - 1) as usize,
            j.clamp(0, self.ny as i64 - 1) as usize,
        )
    }
}

/// A read-only view of an observation-count grid, indexed `[i * ny + j]` to
/// match the reference's row-major `(nx, ny)` array.
#[derive(Debug, Clone, Copy)]
pub struct CoverageView<'a> {
    pub counts: &'a [u32],
    pub bounds: GridBounds,
}

impl CoverageView<'_> {
    pub fn count_at(&self, i: usize, j: usize) -> u32 {
        self.counts[i * self.bounds.ny + j]
    }

    pub fn is_covered(&self, i: usize, j: usize) -> bool {
        self.count_at(i, j) > 0
    }

    pub fn coverage_fraction(&self) -> f64 {
        let total = self.counts.len();
        if total == 0 {
            return 0.0;
        }
        let covered = self.counts.iter().filter(|c| **c > 0).count();
        covered as f64 / total as f64
    }
}

/// Which platform is heading to which cell.
#[derive(Debug, Default, Clone)]
pub struct ClaimedCells {
    assignment: HashMap<String, (usize, usize)>,
}

impl ClaimedCells {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn claim(&mut self, drone_id: &str, cell: (usize, usize)) {
        self.assignment.insert(drone_id.to_string(), cell);
    }

    pub fn release(&mut self, drone_id: &str) {
        self.assignment.remove(drone_id);
    }

    /// Every other platform's claim. A platform's own claim never excludes it.
    pub fn others(&self, drone_id: &str) -> HashSet<(usize, usize)> {
        self.assignment
            .iter()
            .filter(|(id, _)| id.as_str() != drone_id)
            .map(|(_, cell)| *cell)
            .collect()
    }
}

/// Scoring parameters. Defaults match the reference's `FrontierConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierConfig {
    pub distance_weight: f64,
    pub coverage_gradient_weight: f64,
    pub exclusion_radius_cells: i64,
    pub gap_fill_min_fraction: f64,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            distance_weight: 1.0,
            coverage_gradient_weight: 2.0,
            exclusion_radius_cells: 3,
            gap_fill_min_fraction: 0.02,
        }
    }
}

/// Covered cells in the 3x3 window centred on `(i, j)`, counting
/// out-of-bounds as uncovered — which is what the reference's
/// `mode="constant"` zero padding does.
fn neighbour_count(view: &CoverageView<'_>, i: usize, j: usize) -> u32 {
    let mut count = 0;
    for di in -1i64..=1 {
        for dj in -1i64..=1 {
            let ni = i as i64 + di;
            let nj = j as i64 + dj;
            if ni < 0 || nj < 0 || ni >= view.bounds.nx as i64 || nj >= view.bounds.ny as i64 {
                continue;
            }
            if view.is_covered(ni as usize, nj as usize) {
                count += 1;
            }
        }
    }
    count
}

/// How strongly the coverage gradient counts, given how much of the grid is
/// already covered.
///
/// Ramps from zero to `coverage_gradient_weight` between 25% and 75%
/// coverage: early on, platforms spread out by distance alone, because the
/// gradient signal is noise when almost nothing is covered; later they prefer
/// cells beside already-covered area, which fills holes faster.
///
/// A separate function so the parity test can exercise the ramp itself rather
/// than only its effect on which cell wins. A wrongly-scaled weight often
/// picks the same cell anyway — widening the ramp or dropping its upper clamp
/// changed no selection in the fixture, and went unnoticed until the test
/// called this directly instead of recomputing the formula alongside it.
pub fn gradient_weight(config: &FrontierConfig, coverage_fraction: f64) -> f64 {
    config.coverage_gradient_weight * ((coverage_fraction - 0.25) / 0.50).clamp(0.0, 1.0)
}

/// The neighbour count, exposed for the parity test that compares it against
/// what the reference's `uniform_filter` produced.
#[doc(hidden)]
pub fn neighbour_count_for_test(view: &CoverageView<'_>, i: usize, j: usize) -> f64 {
    neighbour_count(view, i, j) as f64
}

/// The best uncovered cell for `drone_id` to head for, or `None` when nothing
/// is uncovered.
///
/// Score is `-distance_weight * normalised_distance + gradient_weight *
/// neighbour_count`, with the gradient weight ramping from zero to full
/// between 25% and 75% coverage so platforms spread out by distance alone
/// early on and fill holes later.
pub fn select_frontier_cell(
    view: &CoverageView<'_>,
    config: &FrontierConfig,
    drone_xy: [f64; 2],
    claimed: &ClaimedCells,
    drone_id: &str,
) -> Option<(usize, usize)> {
    let bounds = view.bounds;
    // Row-major, matching `np.where` on an `(nx, ny)` array. The order is the
    // tie-break, so it is not an implementation detail.
    let uncovered: Vec<(usize, usize)> = (0..bounds.nx)
        .flat_map(|i| (0..bounds.ny).map(move |j| (i, j)))
        .filter(|(i, j)| !view.is_covered(*i, *j))
        .collect();
    if uncovered.is_empty() {
        return None;
    }

    let others = claimed.others(drone_id);
    let candidates: Vec<(usize, usize)> = if others.is_empty() {
        uncovered.clone()
    } else {
        let radius = config.exclusion_radius_cells;
        let kept: Vec<(usize, usize)> = uncovered
            .iter()
            .copied()
            .filter(|(i, j)| {
                !others.iter().any(|(ei, ej)| {
                    (*ei as i64 - *i as i64).abs() <= radius
                        && (*ej as i64 - *j as i64).abs() <= radius
                })
            })
            .collect();
        // Every candidate excluded means every uncovered cell is spoken for.
        // The reference then ignores exclusion entirely rather than returning
        // nothing, so a platform still gets somewhere to go.
        if kept.is_empty() {
            uncovered.clone()
        } else {
            kept
        }
    };

    let (gi, gj) = bounds.xy_to_ij(drone_xy[0], drone_xy[1]);
    let max_distance = (bounds.nx as f64).hypot(bounds.ny as f64).max(1.0);
    let fraction = view.coverage_fraction();
    let gradient_weight = gradient_weight(config, fraction);

    let mut best: Option<(usize, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (i, j) in candidates {
        let distance = (i as f64 - gi as f64).hypot(j as f64 - gj as f64) / max_distance;
        let gradient = if gradient_weight > 0.0 {
            neighbour_count(view, i, j) as f64
        } else {
            0.0
        };
        let score = -config.distance_weight * distance + gradient_weight * gradient;
        // Strictly greater: the FIRST maximum wins, as `np.argmax` does.
        if score > best_score {
            best_score = score;
            best = Some((i, j));
        }
    }
    best
}

/// Uncovered cells that are fully enclosed by covered ones.
///
/// An uncovered region touching the border is open frontier, not a hole, and
/// is excluded — which is what `scipy.ndimage.binary_fill_holes` does, since
/// it fills only background that cannot reach the edge. Implemented as a
/// flood fill inward from the border with 4-connectivity, matching that
/// function's default structure.
///
/// The reference takes a `threshold` parameter and never reads it; it is not
/// carried across. Its only caller uses the result's LENGTH, comparing it
/// against `gap_fill_min_fraction` of the grid, so the returned order is not
/// load-bearing — but it is row-major regardless, matching `np.where`.
pub fn find_gap_cells(view: &CoverageView<'_>) -> Vec<(usize, usize)> {
    let GridBounds { nx, ny, .. } = view.bounds;
    let mut reachable = vec![false; nx * ny];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    let push = |i: usize, j: usize, reachable: &mut Vec<bool>, queue: &mut VecDeque<_>| {
        if !view.is_covered(i, j) && !reachable[i * ny + j] {
            reachable[i * ny + j] = true;
            queue.push_back((i, j));
        }
    };
    for i in 0..nx {
        push(i, 0, &mut reachable, &mut queue);
        push(i, ny - 1, &mut reachable, &mut queue);
    }
    for j in 0..ny {
        push(0, j, &mut reachable, &mut queue);
        push(nx - 1, j, &mut reachable, &mut queue);
    }
    while let Some((i, j)) = queue.pop_front() {
        for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let ni = i as i64 + di;
            let nj = j as i64 + dj;
            if ni < 0 || nj < 0 || ni >= nx as i64 || nj >= ny as i64 {
                continue;
            }
            push(ni as usize, nj as usize, &mut reachable, &mut queue);
        }
    }

    (0..nx)
        .flat_map(|i| (0..ny).map(move |j| (i, j)))
        .filter(|(i, j)| !view.is_covered(*i, *j) && !reachable[i * ny + j])
        .collect()
}
