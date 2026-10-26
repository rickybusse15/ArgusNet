//! Mapping/localization kernels shared by the transitional Python loop and future Rust sim.

/// Score particle footprints on the coverage grid (x-major, y-contiguous).
/// Preserves the legacy clamped-index and exclusive-upper-edge contract,
/// including its -1 sentinel for empty patches. Counts must be nonnegative.
/// One summed-area table makes each particle query constant time.
pub fn score_footprints(
    grid: &[i32],
    shape: [usize; 2],
    origin: [f64; 2],
    resolution: f64,
    radius: f64,
    positions: &[[f64; 2]],
) -> Result<Vec<f64>, &'static str> {
    let [nx, ny] = shape;
    if nx == 0
        || ny == 0
        || nx.checked_mul(ny) != Some(grid.len())
        || grid.iter().any(|&n| n < 0)
        || !resolution.is_finite()
        || resolution <= 0.0
        || !radius.is_finite()
        || radius < 0.0
        || origin.iter().any(|v| !v.is_finite())
        || positions.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("invalid coverage grid or footprint");
    }
    let stride = ny.checked_add(1).ok_or("grid too large")?;
    let size = nx
        .checked_add(1)
        .and_then(|n| n.checked_mul(stride))
        .ok_or("grid too large")?;
    // Checked integer accumulation avoids floating-point cancellation in tiny patches.
    let mut sums = vec![0u64; size];
    for i in 0..nx {
        let mut row = 0u64;
        for j in 0..ny {
            row = row
                .checked_add(grid[i * ny + j] as u64)
                .ok_or("count overflow")?;
            sums[(i + 1) * stride + j + 1] = sums[i * stride + j + 1]
                .checked_add(row)
                .ok_or("count overflow")?;
        }
    }
    let r = ((radius / resolution) as usize).max(1);
    Ok(positions
        .iter()
        .map(|p| {
            // Float-to-integer cast truncates, matching Python int before clipping.
            let i = (((p[0] - origin[0]) / resolution) as usize).min(nx - 1);
            let j = (((p[1] - origin[1]) / resolution) as usize).min(ny - 1);
            let i0 = i.saturating_sub(r);
            let i1 = i.saturating_add(r).min(nx - 1);
            let j0 = j.saturating_sub(r);
            let j1 = j.saturating_add(r).min(ny - 1);
            if i0 >= i1 || j0 >= j1 {
                return -1.0;
            }
            let total = (sums[i1 * stride + j1] - sums[i0 * stride + j1])
                - (sums[i1 * stride + j0] - sums[i0 * stride + j0]);
            (total as f64 / ((i1 - i0) * (j1 - j0)) as f64).ln_1p()
        })
        .collect())
}

pub mod belief;
pub mod coverage;
pub mod footprint;
pub mod quality;
pub mod revision;

pub use belief::{BeliefCell, BeliefCellStatus, BeliefGrids, WorldBeliefQuery};
pub use coverage::{CoverageGrid, CoverageStats};
pub use footprint::{rasterize_circle, GridBounds, Sampling};
pub use revision::{
    BeliefRevision, BeliefStore, BeliefTilePatch, CellObservation, Observation, OccupancyEvidence,
    Revision, TileIndex,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_grid_edges_and_outside_match_reference() {
        let grid: Vec<i32> = (0..35).collect();
        let positions = [[-500.0, 2.0], [1.0, 2.0], [3.9, 5.1], [99.0, 99.0]];
        let actual = score_footprints(&grid, [5, 7], [0.0, 0.0], 1.0, 2.0, &positions).unwrap();
        for (p, score) in positions.iter().zip(actual) {
            let i = (p[0] as usize).min(4);
            let j = (p[1] as usize).min(6);
            let mut values = vec![];
            for x in i.saturating_sub(2)..(i + 2).min(4) {
                for y in j.saturating_sub(2)..(j + 2).min(6) {
                    values.push(grid[x * 7 + y]);
                }
            }
            let expected = (values.iter().sum::<i32>() as f64 / values.len() as f64).ln_1p();
            assert!((score - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn degenerate_and_invalid_inputs() {
        assert_eq!(
            score_footprints(&[0; 5], [1, 5], [0.0; 2], 1.0, 0.0, &[[0.0; 2]]).unwrap(),
            [-1.0]
        );
        assert!(score_footprints(&[-1], [1, 1], [0.0; 2], 1.0, 1.0, &[]).is_err());
        assert!(score_footprints(&[], [0, 1], [0.0; 2], 1.0, 1.0, &[]).is_err());
        assert!(score_footprints(&[0], [1, 1], [0.0; 2], 0.0, 1.0, &[]).is_err());
    }
}
