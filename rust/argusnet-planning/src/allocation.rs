//! Fleet-wide task allocation (PLANNING_EFFICIENCY_PLAN P1).
//!
//! The reference assigns POIs one drone at a time, in fleet order, so the
//! first drones take the best POIs and nothing is optimised across the fleet.
//! [`PlanningPolicy::EfficientV1`] solves every free drone against every
//! pending POI together, as a minimum-cost assignment ([`min_cost_assignment`]).
//! The reference policy is kept byte for byte; the new one is opt-in.

/// Which planner decides the mission's allocations. Recorded in replay `meta`
/// (additively: absent for the reference) and in the driver's checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningPolicy {
    /// The Python reference's greedy, fleet-order assignment, quirks included.
    #[default]
    Reference,
    /// Every free drone against every pending POI at once, on a cost that
    /// includes the hover for the dwell and refuses a POI a drone could not
    /// afford to finish and fly home from.
    EfficientV1,
}

impl PlanningPolicy {
    pub fn is_reference(&self) -> bool {
        *self == PlanningPolicy::Reference
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PlanningPolicy::Reference => "reference",
            PlanningPolicy::EfficientV1 => "efficient_v1",
        }
    }
}

/// The minimum-cost assignment of rows to columns.
///
/// `cost[r][c]` is `None` when row `r` may not take column `c`. The result
/// gives each row its column, or `None`. As many rows as possible are
/// assigned first (an infeasible pair never is), and among assignments of
/// that size the total cost is least. Rows and columns may differ in number.
///
/// Deterministic: the Hungarian method in its potentials form, scanning
/// indices in order with strict comparisons, so among equal-cost solutions
/// the lower indices win. Callers order rows and columns by id.
pub fn min_cost_assignment(cost: &[Vec<Option<f64>>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    let cols = cost.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }
    assert!(
        cost.iter().all(|row| row.len() == cols),
        "a rectangular cost matrix"
    );
    // An infeasible or padding pair costs more than any set of feasible pairs
    // can, so a larger feasible assignment always wins.
    let spread: f64 = cost
        .iter()
        .flatten()
        .flatten()
        .map(|c| c.abs())
        .sum::<f64>();
    let big = 2.0 * spread + 1.0;
    let n = rows.max(cols);
    let at = |r: usize, c: usize| -> f64 {
        if r < rows && c < cols {
            cost[r][c].unwrap_or(big)
        } else {
            big
        }
    };

    // Hungarian method, 1-based with a sentinel column 0.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut matched = vec![0usize; n + 1]; // column -> row
    let mut way = vec![0usize; n + 1];
    for row in 1..=n {
        matched[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[col0] = true;
            let r0 = matched[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;
            for col in 1..=n {
                if used[col] {
                    continue;
                }
                let reduced = at(r0 - 1, col - 1) - u[r0] - v[col];
                if reduced < minv[col] {
                    minv[col] = reduced;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }
            for col in 0..=n {
                if used[col] {
                    u[matched[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;
            if matched[col0] == 0 {
                break;
            }
        }
        loop {
            let col1 = way[col0];
            matched[col0] = matched[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }
    let mut result = vec![None; rows];
    for col in 1..=n {
        let row = matched[col];
        if row >= 1 && row <= rows && col <= cols && cost[row - 1][col - 1].is_some() {
            result[row - 1] = Some(col - 1);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &[Vec<Option<f64>>], result: &[Option<usize>]) -> f64 {
        result
            .iter()
            .enumerate()
            .filter_map(|(r, c)| c.map(|c| cost[r][c].unwrap()))
            .sum()
    }

    #[test]
    fn it_beats_the_greedy_order() {
        // Greedy in row order gives row 0 column 0 (1) and row 1 column 1
        // (10): 11. Crossing over costs 2 + 3 = 5.
        let cost = vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(10.0)]];
        let result = min_cost_assignment(&cost);
        assert_eq!(result, vec![Some(1), Some(0)]);
        assert_eq!(total(&cost, &result), 5.0);
    }

    #[test]
    fn infeasible_pairs_are_never_assigned_and_coverage_comes_first() {
        // Row 1 can only take column 0; giving row 0 the cheap column 0 would
        // leave row 1 with nothing.
        let cost = vec![vec![Some(1.0), Some(50.0)], vec![Some(5.0), None]];
        assert_eq!(min_cost_assignment(&cost), vec![Some(1), Some(0)]);
        let none = vec![vec![None, None]];
        assert_eq!(min_cost_assignment(&none), vec![None]);
    }

    #[test]
    fn rectangular_problems_and_negative_costs() {
        // Three drones, two POIs: the two cheapest distinct pairs.
        let cost = vec![
            vec![Some(-5.0), Some(4.0)],
            vec![Some(1.0), Some(1.0)],
            vec![Some(0.0), Some(-2.0)],
        ];
        let result = min_cost_assignment(&cost);
        assert_eq!(result, vec![Some(0), None, Some(1)]);
        // One drone, three POIs.
        let one = vec![vec![Some(3.0), Some(1.0), Some(2.0)]];
        assert_eq!(min_cost_assignment(&one), vec![Some(1)]);
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        let cost = vec![vec![Some(1.0), Some(1.0)], vec![Some(1.0), Some(1.0)]];
        assert_eq!(min_cost_assignment(&cost), vec![Some(0), Some(1)]);
        let rows = vec![vec![Some(1.0)], vec![Some(1.0)]];
        assert_eq!(min_cost_assignment(&rows), vec![Some(0), None]);
    }

    #[test]
    fn it_matches_brute_force_on_small_random_problems() {
        // A fixed LCG, so the cases are the same every run.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..300 {
            let rows = 1 + (next() * 4.0) as usize;
            let cols = 1 + (next() * 4.0) as usize;
            let cost: Vec<Vec<Option<f64>>> = (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (next() > 0.25).then(|| (next() * 20.0 - 5.0).round()))
                        .collect()
                })
                .collect();
            let result = min_cost_assignment(&cost);
            let (best_count, best_cost) = brute(&cost, 0, &mut vec![false; cols]);
            let count = result.iter().flatten().count();
            assert_eq!(count, best_count, "{cost:?}");
            assert!((total(&cost, &result) - best_cost).abs() < 1e-9, "{cost:?}");
            let mut columns: Vec<usize> = result.iter().flatten().copied().collect();
            columns.sort_unstable();
            columns.dedup();
            assert_eq!(columns.len(), count, "a column assigned twice");
        }
    }

    /// The largest feasible assignment, then the least cost among those.
    fn brute(cost: &[Vec<Option<f64>>], row: usize, used: &mut Vec<bool>) -> (usize, f64) {
        if row == cost.len() {
            return (0, 0.0);
        }
        let mut best = brute(cost, row + 1, used);
        for col in 0..used.len() {
            if used[col] {
                continue;
            }
            if let Some(c) = cost[row][col] {
                used[col] = true;
                let (count, sub) = brute(cost, row + 1, used);
                used[col] = false;
                let candidate = (count + 1, sub + c);
                if candidate.0 > best.0 || (candidate.0 == best.0 && candidate.1 < best.1) {
                    best = candidate;
                }
            }
        }
        best
    }
}
