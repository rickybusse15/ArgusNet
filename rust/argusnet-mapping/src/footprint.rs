//! One circular-footprint rasterizer for coverage and belief updates.
//!
//! P4.6 asks for "one footprint rasterizer whose indices/weights feed both
//! coverage and belief updates". Python has two, and they do not agree:
//!
//! - [`CoverageMap.mark_circular`] samples a lattice in **metres**, anchored at
//!   `centre - radius` and stepped by the grid resolution, testing membership
//!   with the true radius.
//! - [`WorldMap.add_scan_observation`] samples **cell indices** offset from the
//!   centre cell, testing membership against `trunc(radius / resolution)` cells.
//!
//! Truncating the radius to whole cells makes the second disc strictly smaller
//! whenever `radius / resolution` is not an integer. The simulation's own
//! footprint derivation keeps that ratio integral in most configurations, which
//! is why the divergence has gone unnoticed -- but it does not when the 50 m
//! footprint floor binds, and there the height disc covers as little as 59% of
//! the area the coverage disc does. See `docs/migration/FOOTPRINT_RASTERIZATION.md`
//! for the measurements.
//!
//! Rather than pick one silently and re-baseline the goldens, both samplings are
//! named and implemented here as [`Sampling`] modes over a single rasterizer.
//! Callers migrate mode-for-mode with behaviour preserved; choosing between them
//! is then a separate, visible decision with a fixture to measure it against.
//!
//! [`CoverageMap.mark_circular`]: https://docs.rs/  (src/argusnet/mapping/coverage.py)
//! [`WorldMap.add_scan_observation`]: https://docs.rs/  (src/argusnet/mapping/world_map.py)

/// Axis-aligned grid extent, mirroring `argusnet.mapping.occupancy.GridBounds`.
///
/// Cell `(i, j)` spans `[x_min + i*res, x_min + (i+1)*res)` in x and likewise in
/// y; `i` is the x index and `j` the y index, matching the `(nx, ny)` array
/// layout both Python maps use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridBounds {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
    pub resolution_m: f64,
}

impl GridBounds {
    /// Cell counts along x and y.
    ///
    /// `max(1, ceil(span / resolution))` -- the `max` keeps a degenerate or
    /// sub-cell span from producing a zero-sized grid, as Python's does.
    pub fn dimensions(&self) -> Result<(usize, usize), &'static str> {
        if !self.resolution_m.is_finite() || self.resolution_m <= 0.0 {
            return Err("resolution must be finite and positive");
        }
        if !self.x_min_m.is_finite()
            || !self.x_max_m.is_finite()
            || !self.y_min_m.is_finite()
            || !self.y_max_m.is_finite()
        {
            return Err("bounds must be finite");
        }
        let dim = |span: f64| -> Result<usize, &'static str> {
            let cells = (span / self.resolution_m).ceil();
            if !cells.is_finite() || cells > usize::MAX as f64 {
                return Err("grid too large");
            }
            // Python: max(1, int(ceil(...))) -- int() truncates toward zero, so a
            // negative span floors to 0 before the max, not to a negative count.
            Ok((cells as i64).max(1) as usize)
        };
        Ok((
            dim(self.x_max_m - self.x_min_m)?,
            dim(self.y_max_m - self.y_min_m)?,
        ))
    }
}

/// Which of Python's two circle rasterizations to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    /// `CoverageMap.mark_circular`: a metre-space lattice anchored at
    /// `centre - radius`, membership tested with the true radius.
    ///
    /// Because the lattice pitch equals the cell size, distinct samples land in
    /// distinct cells -- except at the upper bound, where clamping can fold two
    /// samples onto the last cell. That fold is a genuine double-increment under
    /// `np.add.at`, so hits are returned with multiplicity.
    WorldAnchored,
    /// `WorldMap.add_scan_observation`: an index-space disc of radius
    /// `max(1, trunc(radius / resolution))` cells around the centre cell.
    CellDisc,
}

/// Rasterize a circular footprint, returning `(i, j)` cell hits **with
/// multiplicity** in the order Python accumulates them.
///
/// Multiplicity is part of the contract: coverage counts revisits, so a cell hit
/// twice by one call must be incremented twice, exactly as `np.add.at` does.
pub fn rasterize_circle(
    bounds: &GridBounds,
    center_xy: [f64; 2],
    radius_m: f64,
    sampling: Sampling,
) -> Result<Vec<(usize, usize)>, &'static str> {
    let (nx, ny) = bounds.dimensions()?;
    if !radius_m.is_finite() || radius_m < 0.0 {
        return Err("radius must be finite and nonnegative");
    }
    if center_xy.iter().any(|v| !v.is_finite()) {
        return Err("centre must be finite");
    }
    match sampling {
        Sampling::WorldAnchored => Ok(world_anchored(bounds, center_xy, radius_m, nx, ny)),
        Sampling::CellDisc => Ok(cell_disc(bounds, center_xy, radius_m, nx, ny)),
    }
}

/// `np.arange(start, stop, step)` for finite positive `step`.
///
/// Reproducing this exactly takes more care than it looks. numpy does not
/// accumulate, and it does not use `step` directly either: its `DOUBLE_fill`
/// writes the first two entries as `start` and `start + step`, derives
/// `delta = (start + step) - start`, and computes the rest as `start + i*delta`.
/// Rounding makes that `delta` differ from `step` whenever `start` is large
/// relative to `step`, so `arange(-89.7, 97.3, 7.3)[4]` is `-60.500000000000014`
/// and not the `-60.5` a plain `start + i*step` yields.
///
/// Two ULP sounds ignorable, and for the sampled coordinate it usually is -- but
/// the coordinate is then truncated to a cell index, where an ULP either side of
/// a boundary is a different cell. `tests/footprint_parity.rs` pins this against
/// numpy bit for bit rather than trusting that it rounds out.
///
/// Length is `ceil((stop - start) / step)`, which is why the lattice can
/// overshoot `stop` on the high side; the circle test rejects the overshoot.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let count = ((stop - start) / step).ceil();
    // `is_finite` also rejects NaN, which a zero-length or degenerate span can
    // produce and which would otherwise fall through the comparison below.
    if !count.is_finite() || count < 1.0 {
        return Vec::new();
    }
    let count = count as usize;
    if count == 1 {
        return vec![start];
    }
    let second = start + step;
    let delta = second - start;
    let mut values = Vec::with_capacity(count);
    values.push(start);
    values.push(second);
    values.extend((2..count).map(|i| start + i as f64 * delta));
    values
}

/// Exposed for the parity test in `tests/footprint_parity.rs`, which pins this
/// against numpy bit for bit. Not part of the crate's intended surface.
#[doc(hidden)]
pub fn arange_for_test(start: f64, stop: f64, step: f64) -> Vec<f64> {
    arange(start, stop, step)
}

fn world_anchored(
    bounds: &GridBounds,
    center_xy: [f64; 2],
    radius_m: f64,
    nx: usize,
    ny: usize,
) -> Vec<(usize, usize)> {
    let [cx, cy] = center_xy;
    let res = bounds.resolution_m;
    let xs = arange(cx - radius_m, cx + radius_m + res, res);
    let ys = arange(cy - radius_m, cy + radius_m + res, res);
    let r_sq = radius_m * radius_m;
    let mut hits = Vec::new();
    // `indexing="ij"` iterates x-major, and the boolean mask preserves that
    // order, so the accumulation order below matches Python's.
    for &x in &xs {
        let dx = x - cx;
        for &y in &ys {
            let dy = y - cy;
            if dx * dx + dy * dy > r_sq {
                continue;
            }
            // Bounds test is inclusive on both edges, then indices are truncated
            // and clamped -- so a sample exactly on the far edge folds onto the
            // last cell rather than being dropped.
            if x < bounds.x_min_m || x > bounds.x_max_m || y < bounds.y_min_m || y > bounds.y_max_m
            {
                continue;
            }
            let i = trunc_index((x - bounds.x_min_m) / res, nx);
            let j = trunc_index((y - bounds.y_min_m) / res, ny);
            hits.push((i, j));
        }
    }
    hits
}

fn cell_disc(
    bounds: &GridBounds,
    center_xy: [f64; 2],
    radius_m: f64,
    nx: usize,
    ny: usize,
) -> Vec<(usize, usize)> {
    let [cx, cy] = center_xy;
    let res = bounds.resolution_m;
    // The truncation that shrinks this disc relative to WorldAnchored.
    let r_cells = ((radius_m / res) as i64).max(1);
    // Python takes int() of the offset before building the range, so a centre
    // left of the grid gives a negative centre index rather than a clamped one.
    let ci = ((cx - bounds.x_min_m) / res) as i64;
    let cj = ((cy - bounds.y_min_m) / res) as i64;
    let i_lo = (ci - r_cells).max(0);
    let i_hi = (ci + r_cells + 1).min(nx as i64);
    let j_lo = (cj - r_cells).max(0);
    let j_hi = (cj + r_cells + 1).min(ny as i64);
    if i_lo >= i_hi || j_lo >= j_hi {
        return Vec::new();
    }
    let r_sq = r_cells * r_cells;
    let mut hits = Vec::new();
    for i in i_lo..i_hi {
        let di = i - ci;
        for j in j_lo..j_hi {
            let dj = j - cj;
            if di * di + dj * dj <= r_sq {
                hits.push((i as usize, j as usize));
            }
        }
    }
    hits
}

/// Python's `int(v)` then `np.clip(_, 0, n-1)`: truncate toward zero, then clamp.
fn trunc_index(value: f64, n: usize) -> usize {
    let truncated = value as i64;
    truncated.clamp(0, n as i64 - 1) as usize
}
