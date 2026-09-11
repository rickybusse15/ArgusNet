//! Planning-facing belief queries over the coverage / occupancy / height grids.
//!
//! This is the read-only facade `WorldBeliefQuery` provides in the reference:
//! planners ask about coverage, confidence, obstacles and geofence membership
//! without touching raw arrays or simulation truth.
//!
//! # Two coordinate systems that do not agree, deliberately
//!
//! `GridBounds` sizes the grid as `ceil(span / resolution)`, so when the span
//! is not a whole number of cells **the grid extends past the bounds**. The
//! reference's predicates split on that gap rather than papering over it:
//!
//! * `coverage_at`, and everything built on it, first tests the raw inclusive
//!   bounds and returns `0` outside them — so a point inside the last cell but
//!   past `x_max_m` reads as never observed.
//! * `height_estimate_at` does **not** test the bounds. It clamps to the grid
//!   and reports the last cell's height.
//!
//! The consequence is sharp enough to look like a bug and is reproduced:
//! `cell_ij` on the **last** cell of a non-aligned grid takes its centre from
//! `ij_to_xy`, which can land past `x_max_m`, so the cell reports itself
//! `OutsideGeofence` with zero coverage even though the grid holds counts for
//! it. `geofence_excludes_the_last_cell_of_a_ragged_grid` pins it.
//!
//! # Sample positions along a corridor follow numpy's `linspace`
//!
//! `safe_corridor_between` walks `np.linspace(0, 1, n)`, which is
//! `k * (1 / (n - 1))` with the final value forced to exactly `1.0` — **not**
//! `k / (n - 1)`. The two disagree for n = 7 and n = 101 and agree at the
//! default n = 16, so a port checked only at the default would look correct.

use crate::footprint::GridBounds;

/// Where a cell sits in the belief world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefCellStatus {
    OutsideGeofence,
    Unknown,
    KnownObstacle,
    Uncertain,
    KnownSafe,
}

impl BeliefCellStatus {
    /// The string the reference's enum carries, which is what reaches replay.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OutsideGeofence => "outside_geofence",
            Self::Unknown => "unknown",
            Self::KnownObstacle => "known_obstacle",
            Self::Uncertain => "uncertain",
            Self::KnownSafe => "known_safe",
        }
    }
}

/// One cell of the belief world.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefCell {
    pub cell_id: String,
    pub ij: (usize, usize),
    pub center_xy_m: (f64, f64),
    pub height_estimate_m: Option<f64>,
    pub height_uncertainty_m: Option<f64>,
    pub obstacle_probability: f64,
    pub terrain_confidence: f64,
    pub coverage_count: u32,
    pub last_observed_s: Option<f64>,
    pub inside_geofence: bool,
    pub status: BeliefCellStatus,
    pub source_ids: Vec<String>,
}

/// A scored frontier cell.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierCandidate {
    pub cell: BeliefCell,
    pub score: f64,
    pub expected_coverage_gain: f64,
    pub uncertainty_reduction: f64,
    pub travel_cost_m: f64,
    pub risk_cost: f64,
}

/// The grids the query reads. All are `nx * ny`, **i-major** (j contiguous),
/// matching the reference's numpy layout.
pub struct BeliefGrids<'a> {
    pub coverage: &'a [u32],
    /// `WorldMap.mean_height_grid`. Absent when the query has no world map, in
    /// which case `height_estimate_at` is always `None`.
    pub mean_height: Option<&'a [f64]>,
    /// `OccupancyGrid` probabilities. Absent when no occupancy grid is wired
    /// in, which switches `obstacle_probability_at` to the observed/unobserved
    /// fallback rather than to a default probability.
    pub occupancy: Option<&'a [f64]>,
    pub last_observed_s: Option<&'a [f64]>,
}

type Geofence<'a> = &'a dyn Fn(f64, f64) -> bool;
type SemanticLabel<'a> = &'a dyn Fn(f64, f64) -> Option<String>;

/// Read-only query facade over coverage, height, occupancy and geofence belief.
pub struct WorldBeliefQuery<'a> {
    bounds: GridBounds,
    nx: usize,
    ny: usize,
    grids: BeliefGrids<'a>,
    geofence: Option<Geofence<'a>>,
    semantic_label: Option<SemanticLabel<'a>>,
}

impl<'a> WorldBeliefQuery<'a> {
    pub fn new(bounds: GridBounds, grids: BeliefGrids<'a>) -> Result<Self, &'static str> {
        let (nx, ny) = bounds.dimensions()?;
        let expected = nx.checked_mul(ny).ok_or("grid too large")?;
        for grid in [
            Some(grids.coverage.len()),
            grids.mean_height.map(<[f64]>::len),
            grids.occupancy.map(<[f64]>::len),
            grids.last_observed_s.map(<[f64]>::len),
        ]
        .into_iter()
        .flatten()
        {
            if grid != expected {
                return Err("grid length does not match bounds");
            }
        }
        Ok(Self {
            bounds,
            nx,
            ny,
            grids,
            geofence: None,
            semantic_label: None,
        })
    }

    pub fn with_geofence(mut self, geofence: Geofence<'a>) -> Self {
        self.geofence = Some(geofence);
        self
    }

    pub fn with_semantic_label(mut self, semantic_label: SemanticLabel<'a>) -> Self {
        self.semantic_label = Some(semantic_label);
        self
    }

    pub fn bounds(&self) -> GridBounds {
        self.bounds
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    // ------------------------------------------------------------------
    // Coordinates
    // ------------------------------------------------------------------

    /// `GridBounds.xy_to_ij`: truncate toward zero, then clamp into the grid.
    pub fn xy_to_ij(&self, x_m: f64, y_m: f64) -> (usize, usize) {
        let axis = |value: f64, min: f64, n: usize| -> usize {
            // Python `int()` truncates toward zero before `np.clip`, so a point
            // just below the origin lands on -0 and clamps to 0 either way.
            let index = ((value - min) / self.bounds.resolution_m) as i64;
            index.clamp(0, n as i64 - 1) as usize
        };
        (
            axis(x_m, self.bounds.x_min_m, self.nx),
            axis(y_m, self.bounds.y_min_m, self.ny),
        )
    }

    /// `GridBounds.ij_to_xy`: the cell centre, which for the last cell of a
    /// ragged grid can lie past the bounds.
    pub fn ij_to_xy(&self, i: usize, j: usize) -> (f64, f64) {
        (
            self.bounds.x_min_m + (i as f64 + 0.5) * self.bounds.resolution_m,
            self.bounds.y_min_m + (j as f64 + 0.5) * self.bounds.resolution_m,
        )
    }

    /// The raw inclusive bounds test, which is **not** the same as being inside
    /// the grid.
    fn in_bounds(&self, x_m: f64, y_m: f64) -> bool {
        self.bounds.x_min_m <= x_m
            && x_m <= self.bounds.x_max_m
            && self.bounds.y_min_m <= y_m
            && y_m <= self.bounds.y_max_m
    }

    fn at(&self, grid: &[f64], i: usize, j: usize) -> f64 {
        grid[i * self.ny + j]
    }

    // ------------------------------------------------------------------
    // Scalar queries
    // ------------------------------------------------------------------

    /// The mean height at a point, or `None` with no world map or a non-finite
    /// cell. Deliberately does not test the bounds — it clamps.
    pub fn height_estimate_at(&self, x_m: f64, y_m: f64) -> Option<f64> {
        let grid = self.grids.mean_height?;
        let (i, j) = self.xy_to_ij(x_m, y_m);
        let value = self.at(grid, i, j);
        value.is_finite().then_some(value)
    }

    /// A conservative proxy: `resolution / sqrt(visits)`, or `None` when the
    /// cell has never been visited.
    pub fn height_uncertainty_at(&self, x_m: f64, y_m: f64) -> Option<f64> {
        let visits = self.coverage_at(x_m, y_m);
        if visits == 0 {
            return None;
        }
        Some(self.bounds.resolution_m / (visits as f64).sqrt())
    }

    /// With no occupancy grid this is `0.5` for an unobserved point and `0.0`
    /// for an observed one — an unknown-obstacle prior, not a default
    /// probability, so wiring in an all-`0.5` grid is a different thing.
    pub fn obstacle_probability_at(&self, x_m: f64, y_m: f64) -> f64 {
        match self.grids.occupancy {
            Some(grid) => {
                let (i, j) = self.xy_to_ij(x_m, y_m);
                self.at(grid, i, j)
            }
            None => {
                if self.is_observed(x_m, y_m) {
                    0.0
                } else {
                    0.5
                }
            }
        }
    }

    /// Visit count, or `0` outside the raw bounds.
    pub fn coverage_at(&self, x_m: f64, y_m: f64) -> u32 {
        if !self.in_bounds(x_m, y_m) {
            return 0;
        }
        let (i, j) = self.xy_to_ij(x_m, y_m);
        self.grids.coverage[i * self.ny + j]
    }

    /// `min(1, visits / 5) * (1 - obstacle_probability)`, floored at zero and
    /// zeroed outside the geofence.
    pub fn confidence_at(&self, x_m: f64, y_m: f64) -> f64 {
        let visits = self.coverage_at(x_m, y_m);
        if visits == 0 || !self.is_inside_geofence(x_m, y_m) {
            return 0.0;
        }
        let coverage_confidence = (visits as f64 / 5.0).min(1.0);
        let obstacle_risk = self.obstacle_probability_at(x_m, y_m);
        (coverage_confidence * (1.0 - obstacle_risk)).max(0.0)
    }

    pub fn is_observed(&self, x_m: f64, y_m: f64) -> bool {
        self.coverage_at(x_m, y_m) > 0
    }

    /// Outside the raw bounds is outside the geofence, whether or not a
    /// geofence predicate is supplied.
    pub fn is_inside_geofence(&self, x_m: f64, y_m: f64) -> bool {
        if !self.in_bounds(x_m, y_m) {
            return false;
        }
        match self.geofence {
            Some(predicate) => predicate(x_m, y_m),
            None => true,
        }
    }

    /// Note the threshold is strict `< 0.4`, while `cell_ij` calls `>= 0.4`
    /// `Uncertain`. They agree, but only because both are spelled at 0.4.
    pub fn is_known_safe(&self, x_m: f64, y_m: f64) -> bool {
        self.is_inside_geofence(x_m, y_m)
            && self.is_observed(x_m, y_m)
            && self.obstacle_probability_at(x_m, y_m) < 0.4
    }

    // ------------------------------------------------------------------
    // Cells
    // ------------------------------------------------------------------

    /// The cell containing a point, snapped to that cell's centre.
    pub fn cell_at(&self, x_m: f64, y_m: f64) -> BeliefCell {
        let (i, j) = self.xy_to_ij(x_m, y_m);
        self.cell_ij(i, j)
    }

    /// One cell, evaluated at its own centre.
    pub fn cell_ij(&self, i: usize, j: usize) -> BeliefCell {
        let (x_m, y_m) = self.ij_to_xy(i, j);
        let inside = self.is_inside_geofence(x_m, y_m);
        let observed = self.is_observed(x_m, y_m);
        let obstacle_probability = self.obstacle_probability_at(x_m, y_m);

        let status = if !inside {
            BeliefCellStatus::OutsideGeofence
        } else if !observed {
            BeliefCellStatus::Unknown
        } else if obstacle_probability >= 0.6 {
            BeliefCellStatus::KnownObstacle
        } else if obstacle_probability >= 0.4 {
            BeliefCellStatus::Uncertain
        } else {
            BeliefCellStatus::KnownSafe
        };

        // Read by index rather than by position: this is the one place the
        // reference indexes the grid directly, and it guards with IndexError
        // rather than clamping.
        let last_observed_s = self.grids.last_observed_s.and_then(|grid| {
            if i >= self.nx || j >= self.ny {
                return None;
            }
            let raw = self.at(grid, i, j);
            raw.is_finite().then_some(raw)
        });

        let source_ids = match self.semantic_label {
            Some(label) => match label(x_m, y_m) {
                // An empty label is falsy in the reference and contributes
                // nothing, so it is not the same as a label of `""`.
                Some(name) if !name.is_empty() => vec![format!("semantic:{name}")],
                _ => Vec::new(),
            },
            None => Vec::new(),
        };

        BeliefCell {
            cell_id: format!("cell-{i}-{j}"),
            ij: (i, j),
            center_xy_m: (x_m, y_m),
            height_estimate_m: self.height_estimate_at(x_m, y_m),
            height_uncertainty_m: self.height_uncertainty_at(x_m, y_m),
            obstacle_probability,
            terrain_confidence: self.confidence_at(x_m, y_m),
            coverage_count: self.coverage_at(x_m, y_m),
            last_observed_s,
            inside_geofence: inside,
            status,
            source_ids,
        }
    }

    /// Every cell, i-major.
    pub fn cells(&self) -> impl Iterator<Item = BeliefCell> + '_ {
        (0..self.nx).flat_map(move |i| (0..self.ny).map(move |j| self.cell_ij(i, j)))
    }

    // ------------------------------------------------------------------
    // Frontiers
    // ------------------------------------------------------------------

    /// The 3x3 patch around `(i, j)`, clamped at both edges, as `(i0, i1, j0, j1)`
    /// half-open ranges.
    ///
    /// The reference clamps the low edge with `max(0, i - 1)` and lets numpy
    /// clamp the high edge, so an edge patch is genuinely smaller — 4 cells at
    /// a corner, 6 along a side, 9 inside. That size is the denominator of
    /// `expected_coverage_gain`, so the shrinking is load-bearing, not incidental.
    fn patch(&self, i: usize, j: usize) -> (usize, usize, usize, usize) {
        (
            i.saturating_sub(1),
            (i + 2).min(self.nx),
            j.saturating_sub(1),
            (j + 2).min(self.ny),
        )
    }

    /// Uncovered cells inside the geofence that touch a covered cell, in
    /// i-major order — the order `np.where` yields.
    pub fn frontier_cells(&self) -> Vec<BeliefCell> {
        let mut result = Vec::new();
        for i in 0..self.nx {
            for j in 0..self.ny {
                if self.grids.coverage[i * self.ny + j] > 0 {
                    continue;
                }
                let (x_m, y_m) = self.ij_to_xy(i, j);
                if !self.is_inside_geofence(x_m, y_m) {
                    continue;
                }
                let (i0, i1, j0, j1) = self.patch(i, j);
                let touches_covered = (i0..i1)
                    .any(|pi| (j0..j1).any(|pj| self.grids.coverage[pi * self.ny + pj] > 0));
                if touches_covered {
                    result.push(self.cell_ij(i, j));
                }
            }
        }
        result
    }

    /// The fraction of the 3x3 patch that is still uncovered.
    pub fn expected_coverage_gain(&self, i: usize, j: usize) -> f64 {
        let (i0, i1, j0, j1) = self.patch(i, j);
        let mut zeros = 0usize;
        let mut size = 0usize;
        for pi in i0..i1 {
            for pj in j0..j1 {
                size += 1;
                if self.grids.coverage[pi * self.ny + pj] == 0 {
                    zeros += 1;
                }
            }
        }
        zeros as f64 / size.max(1) as f64
    }

    /// Frontier cells scored and ranked, best first.
    ///
    /// The reference sorts on `(-score, cell.ij)`, a total key, so ties break
    /// by `(i, j)` ascending rather than by the order the frontiers were found.
    pub fn score_frontiers(
        &self,
        origin_xy_m: (f64, f64),
        max_candidates: usize,
    ) -> Vec<FrontierCandidate> {
        let (ox, oy) = origin_xy_m;
        let mut candidates: Vec<FrontierCandidate> = self
            .frontier_cells()
            .into_iter()
            .map(|cell| {
                let (x_m, y_m) = cell.center_xy_m;
                let travel_cost_m = (x_m - ox).hypot(y_m - oy);
                let expected_coverage_gain = self.expected_coverage_gain(cell.ij.0, cell.ij.1);
                let uncertainty_reduction = if cell.status == BeliefCellStatus::Unknown {
                    1.0
                } else {
                    0.25
                };
                // The out-of-fence penalty is dead: `frontier_cells` already
                // skips any cell outside the geofence, so this arm never fires.
                // Kept because it is dead in the reference too, and removing
                // it here would make the two read differently for no gain.
                let risk_cost =
                    cell.obstacle_probability + if cell.inside_geofence { 0.0 } else { 1.0 };
                let score = expected_coverage_gain + uncertainty_reduction
                    - 0.002 * travel_cost_m
                    - risk_cost;
                FrontierCandidate {
                    cell,
                    score,
                    expected_coverage_gain,
                    uncertainty_reduction,
                    travel_cost_m,
                    risk_cost,
                }
            })
            .collect();

        // The `ij` tie-break is redundant in practice — `frontier_cells`
        // emits cells ij-ascending and both sorts are stable, so ties already
        // keep that order — but it is what the reference's total sort key
        // says, and it is the only thing keeping the ranking well-defined if
        // `frontier_cells` ever stops being ordered.
        candidates.sort_by(|a, b| {
            (-a.score)
                .total_cmp(&(-b.score))
                .then(a.cell.ij.cmp(&b.cell.ij))
        });
        candidates.truncate(max_candidates);
        candidates
    }

    // ------------------------------------------------------------------
    // Corridors
    // ------------------------------------------------------------------

    /// Whether a straight line between two points is safe at every sample.
    ///
    /// `allow_unknown` skips unobserved samples rather than accepting them, so
    /// the geofence is still enforced at every sample either way — the order of
    /// the two tests is what makes that true.
    pub fn safe_corridor_between(
        &self,
        start_xy_m: (f64, f64),
        end_xy_m: (f64, f64),
        sample_count: usize,
        allow_unknown: bool,
    ) -> bool {
        let (sx, sy) = start_xy_m;
        let (ex, ey) = end_xy_m;
        let n = sample_count.max(2);
        // numpy's linspace: `k * (1 / (n - 1))`, with the last value forced to
        // exactly 1.0. `k / (n - 1)` is a different number for n = 7 and 101.
        let step = 1.0 / (n - 1) as f64;
        for k in 0..n {
            let alpha = if k == n - 1 { 1.0 } else { k as f64 * step };
            let x_m = sx + (ex - sx) * alpha;
            let y_m = sy + (ey - sy) * alpha;
            if !self.is_inside_geofence(x_m, y_m) {
                return false;
            }
            if allow_unknown && !self.is_observed(x_m, y_m) {
                continue;
            }
            if !self.is_known_safe(x_m, y_m) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(x_max: f64, y_max: f64, resolution: f64) -> GridBounds {
        GridBounds {
            x_min_m: 0.0,
            x_max_m: x_max,
            y_min_m: 0.0,
            y_max_m: y_max,
            resolution_m: resolution,
        }
    }

    /// `(i, j)` into the i-major grid, so the tests read the way the module does.
    fn idx(i: usize, j: usize, ny: usize) -> usize {
        i * ny + j
    }

    fn grids(coverage: &[u32]) -> BeliefGrids<'_> {
        BeliefGrids {
            coverage,
            mean_height: None,
            occupancy: None,
            last_observed_s: None,
        }
    }

    #[test]
    fn grid_length_must_match_the_bounds() {
        let coverage = [0u32; 5];
        assert!(WorldBeliefQuery::new(bounds(10.0, 10.0, 1.0), grids(&coverage)).is_err());
    }

    #[test]
    fn coverage_outside_the_raw_bounds_is_zero() {
        let coverage = [7u32; 4];
        let query = WorldBeliefQuery::new(bounds(2.0, 2.0, 1.0), grids(&coverage)).unwrap();
        assert_eq!(query.coverage_at(0.5, 0.5), 7);
        // Inside the grid's last cell by index, but past x_max.
        assert_eq!(query.coverage_at(2.5, 0.5), 0);
    }

    #[test]
    fn height_ignores_the_bounds_where_coverage_does_not() {
        let coverage = [1u32; 4];
        let heights = [11.0, 12.0, 13.0, 14.0];
        let query = WorldBeliefQuery::new(
            bounds(2.0, 2.0, 1.0),
            BeliefGrids {
                coverage: &coverage,
                mean_height: Some(&heights),
                occupancy: None,
                last_observed_s: None,
            },
        )
        .unwrap();
        // The same point: no coverage, but a clamped height.
        assert_eq!(query.coverage_at(50.0, 50.0), 0);
        assert_eq!(query.height_estimate_at(50.0, 50.0), Some(14.0));
    }

    #[test]
    fn a_non_finite_height_reads_as_absent() {
        let coverage = [1u32; 1];
        let heights = [f64::NAN];
        let query = WorldBeliefQuery::new(
            bounds(1.0, 1.0, 1.0),
            BeliefGrids {
                coverage: &coverage,
                mean_height: Some(&heights),
                occupancy: None,
                last_observed_s: None,
            },
        )
        .unwrap();
        assert_eq!(query.height_estimate_at(0.5, 0.5), None);
    }

    #[test]
    fn geofence_excludes_the_last_cell_of_a_ragged_grid() {
        // span 10, resolution 3 -> 4 cells, centres at 1.5, 4.5, 7.5, 10.5.
        // The last centre is past x_max, so the cell reports itself outside.
        let coverage = [5u32; 16];
        let query = WorldBeliefQuery::new(bounds(10.0, 10.0, 3.0), grids(&coverage)).unwrap();
        assert_eq!(query.dimensions(), (4, 4));
        let (x, _) = query.ij_to_xy(3, 0);
        assert!(x > 10.0, "the last centre should fall past x_max, got {x}");

        let cell = query.cell_ij(3, 0);
        assert!(!cell.inside_geofence);
        assert_eq!(cell.status, BeliefCellStatus::OutsideGeofence);
        assert_eq!(cell.coverage_count, 0, "despite the grid holding 5");
    }

    #[test]
    fn no_occupancy_grid_means_an_unknown_prior_not_a_default() {
        let coverage = [0u32, 3, 0, 0];
        let query = WorldBeliefQuery::new(bounds(2.0, 2.0, 1.0), grids(&coverage)).unwrap();
        assert_eq!(query.obstacle_probability_at(0.5, 1.5), 0.0); // observed
        assert_eq!(query.obstacle_probability_at(0.5, 0.5), 0.5); // unobserved
    }

    #[test]
    fn confidence_saturates_at_five_visits() {
        let coverage = [4u32, 5, 6, 0];
        let query = WorldBeliefQuery::new(bounds(2.0, 2.0, 1.0), grids(&coverage)).unwrap();
        assert!((query.confidence_at(0.5, 0.5) - 0.8).abs() < 1e-12);
        assert!((query.confidence_at(0.5, 1.5) - 1.0).abs() < 1e-12);
        assert!((query.confidence_at(1.5, 0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn status_thresholds_are_inclusive_at_both_steps() {
        let coverage = [1u32; 3];
        let occupancy = [0.6, 0.4, 0.39999999999999997];
        let query = WorldBeliefQuery::new(
            bounds(3.0, 1.0, 1.0),
            BeliefGrids {
                coverage: &coverage,
                mean_height: None,
                occupancy: Some(&occupancy),
                last_observed_s: None,
            },
        )
        .unwrap();
        assert_eq!(query.cell_ij(0, 0).status, BeliefCellStatus::KnownObstacle);
        assert_eq!(query.cell_ij(1, 0).status, BeliefCellStatus::Uncertain);
        assert_eq!(query.cell_ij(2, 0).status, BeliefCellStatus::KnownSafe);
    }

    #[test]
    fn frontier_cells_are_uncovered_neighbours_of_covered_ones() {
        // Only (1,1) is covered, on a 3x3 grid.
        let mut coverage = [0u32; 9];
        coverage[idx(1, 1, 3)] = 2;
        let query = WorldBeliefQuery::new(bounds(3.0, 3.0, 1.0), grids(&coverage)).unwrap();
        let ids: Vec<(usize, usize)> = query.frontier_cells().iter().map(|c| c.ij).collect();
        // Every cell but the covered one touches it, in i-major order.
        assert_eq!(
            ids,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2)
            ]
        );
    }

    #[test]
    fn expected_gain_denominator_shrinks_at_the_edges() {
        let coverage = [0u32; 9];
        let query = WorldBeliefQuery::new(bounds(3.0, 3.0, 1.0), grids(&coverage)).unwrap();
        // All zero, so the ratio is 1.0 everywhere; what differs is the size,
        // which only shows once some cells are covered.
        let mut covered = [0u32; 9];
        covered[0] = 1;
        let query2 = WorldBeliefQuery::new(bounds(3.0, 3.0, 1.0), grids(&covered)).unwrap();
        // Corner patch is 2x2 = 4 cells, one of them covered.
        assert!((query2.expected_coverage_gain(0, 0) - 0.75).abs() < 1e-12);
        // Centre patch is 3x3 = 9 cells, one covered.
        assert!((query2.expected_coverage_gain(1, 1) - 8.0 / 9.0).abs() < 1e-12);
        assert!((query.expected_coverage_gain(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn frontier_ties_break_by_ij_ascending() {
        // A symmetric layout: several frontiers at identical distance.
        let mut coverage = [0u32; 9];
        coverage[idx(1, 1, 3)] = 1;
        let query = WorldBeliefQuery::new(bounds(3.0, 3.0, 1.0), grids(&coverage)).unwrap();
        let ranked = query.score_frontiers((1.5, 1.5), 4);
        let ids: Vec<(usize, usize)> = ranked.iter().map(|c| c.cell.ij).collect();
        // The four edge-adjacent cells tie on score; (0,1) sorts before (1,0).
        assert_eq!(ids[0], (0, 1));
        assert_eq!(ids[1], (1, 0));
    }

    #[test]
    fn max_candidates_truncates_after_sorting() {
        let mut coverage = [0u32; 9];
        coverage[idx(1, 1, 3)] = 1;
        let query = WorldBeliefQuery::new(bounds(3.0, 3.0, 1.0), grids(&coverage)).unwrap();
        assert_eq!(query.score_frontiers((1.5, 1.5), 2).len(), 2);
        assert_eq!(query.score_frontiers((1.5, 1.5), 100).len(), 8);
    }

    #[test]
    fn a_corridor_over_covered_safe_ground_passes() {
        let coverage = [3u32; 16];
        let query = WorldBeliefQuery::new(bounds(4.0, 4.0, 1.0), grids(&coverage)).unwrap();
        assert!(query.safe_corridor_between((0.5, 0.5), (3.5, 3.5), 16, false));
    }

    #[test]
    fn an_unobserved_sample_fails_unless_unknown_is_allowed() {
        let mut coverage = [3u32; 16];
        coverage[idx(2, 2, 4)] = 0;
        let query = WorldBeliefQuery::new(bounds(4.0, 4.0, 1.0), grids(&coverage)).unwrap();
        assert!(!query.safe_corridor_between((0.5, 0.5), (3.5, 3.5), 16, false));
        assert!(query.safe_corridor_between((0.5, 0.5), (3.5, 3.5), 16, true));
    }

    #[test]
    fn the_geofence_is_enforced_even_when_unknown_is_allowed() {
        let coverage = [0u32; 16];
        let fence = |x: f64, _y: f64| x < 2.0;
        let query = WorldBeliefQuery::new(bounds(4.0, 4.0, 1.0), grids(&coverage))
            .unwrap()
            .with_geofence(&fence);
        assert!(!query.safe_corridor_between((0.5, 0.5), (3.5, 0.5), 16, true));
    }

    #[test]
    fn sample_count_has_a_floor_of_two() {
        let coverage = [3u32; 16];
        let query = WorldBeliefQuery::new(bounds(4.0, 4.0, 1.0), grids(&coverage)).unwrap();
        // 0 and 1 both become 2, so the endpoints are still both sampled.
        for n in [0, 1, 2] {
            assert!(query.safe_corridor_between((0.5, 0.5), (3.5, 3.5), n, false));
        }
    }

    #[test]
    fn confidence_is_floored_at_zero_for_an_out_of_range_occupancy() {
        // The reference cannot reach this: its occupancy is a sigmoid, so
        // `1 - risk` is always positive and the `max(0.0)` never fires. The
        // Rust API takes the probabilities as data, so a caller can hand it
        // something outside [0, 1] and the floor becomes load-bearing. Pinned
        // as a unit test rather than in the fixture, which is captured from
        // the real OccupancyGrid and could not produce this.
        let coverage = [5u32; 2];
        let occupancy = [1.5, -0.5];
        let query = WorldBeliefQuery::new(
            bounds(2.0, 1.0, 1.0),
            BeliefGrids {
                coverage: &coverage,
                mean_height: None,
                occupancy: Some(&occupancy),
                last_observed_s: None,
            },
        )
        .unwrap();
        assert_eq!(query.confidence_at(0.5, 0.5), 0.0, "risk above 1 floors");
        assert_eq!(
            query.confidence_at(1.5, 0.5),
            1.5,
            "risk below 0 is not capped"
        );
    }

    #[test]
    fn an_empty_semantic_label_contributes_nothing() {
        let coverage = [1u32; 1];
        let empty = |_x: f64, _y: f64| Some(String::new());
        let named = |_x: f64, _y: f64| Some("scrub".to_string());
        let b = bounds(1.0, 1.0, 1.0);
        let q1 = WorldBeliefQuery::new(b, grids(&coverage))
            .unwrap()
            .with_semantic_label(&empty);
        assert!(q1.cell_ij(0, 0).source_ids.is_empty());
        let q2 = WorldBeliefQuery::new(b, grids(&coverage))
            .unwrap()
            .with_semantic_label(&named);
        assert_eq!(q2.cell_ij(0, 0).source_ids, vec!["semantic:scrub"]);
    }
}
