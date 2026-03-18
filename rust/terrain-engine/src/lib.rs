//! # terrain-engine
//!
//! 3D terrain query library for the Smart Trajectory Tracker.
//!
//! All coordinates are in **meters** using a local ENU (East-North-Up) frame.
//! Horizontal axes: X = East, Y = North. Vertical axis: Z = Up (altitude above datum).
//!
//! ## Core abstractions
//!
//! - [`TerrainBounds`] – axis-aligned bounding box of the terrain region.
//! - [`TerrainHit`] – result of a ray–terrain intersection.
//! - [`TerrainQuery`] – trait implemented by all terrain backends.
//! - [`GridTerrain`] – bilinearly-interpolated heightmap on a regular grid.
//! - [`FlatTerrain`] – constant-height plane, useful for unit testing and flat-world simulations.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TerrainBounds
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box of the terrain region, in meters.
///
/// The box spans `[x_min_m, x_max_m] × [y_min_m, y_max_m] × [z_min_m, z_max_m]`.
/// Elevation bounds (`z_*`) represent the minimum and maximum **terrain surface** altitude
/// found within the horizontal extent of this terrain, not agent altitudes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TerrainBounds {
    /// Minimum X (East) coordinate, metres.
    pub x_min_m: f64,
    /// Maximum X (East) coordinate, metres.
    pub x_max_m: f64,
    /// Minimum Y (North) coordinate, metres.
    pub y_min_m: f64,
    /// Maximum Y (North) coordinate, metres.
    pub y_max_m: f64,
    /// Minimum terrain-surface elevation (Z), metres above datum.
    pub z_min_m: f64,
    /// Maximum terrain-surface elevation (Z), metres above datum.
    pub z_max_m: f64,
}

impl TerrainBounds {
    /// Construct a new [`TerrainBounds`].
    pub fn new(
        x_min_m: f64,
        x_max_m: f64,
        y_min_m: f64,
        y_max_m: f64,
        z_min_m: f64,
        z_max_m: f64,
    ) -> Self {
        Self {
            x_min_m,
            x_max_m,
            y_min_m,
            y_max_m,
            z_min_m,
            z_max_m,
        }
    }

    /// Returns `true` if the horizontal point `(x_m, y_m)` lies within the bounds.
    #[inline]
    pub fn contains_xy(&self, x_m: f64, y_m: f64) -> bool {
        x_m >= self.x_min_m
            && x_m <= self.x_max_m
            && y_m >= self.y_min_m
            && y_m <= self.y_max_m
    }

    /// Width of the domain in the X direction, metres.
    #[inline]
    pub fn width_m(&self) -> f64 {
        self.x_max_m - self.x_min_m
    }

    /// Depth of the domain in the Y direction, metres.
    #[inline]
    pub fn depth_m(&self) -> f64 {
        self.y_max_m - self.y_min_m
    }
}

// ---------------------------------------------------------------------------
// TerrainHit
// ---------------------------------------------------------------------------

/// Record of a ray–terrain intersection.
///
/// Produced by [`TerrainQuery::los_raycast`] when the line-of-sight is blocked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainHit {
    /// Ray parameter `t ∈ [0, 1]` at the point of intersection,
    /// where `t = 0` is the ray origin and `t = 1` is the ray target.
    pub t: f64,
    /// World-space intersection point `[x_m, y_m, z_m]`.
    pub point_m: [f64; 3],
    /// Terrain surface elevation (Z) directly below the intersection point, metres.
    pub terrain_z_m: f64,
}

// ---------------------------------------------------------------------------
// TerrainQuery trait
// ---------------------------------------------------------------------------

/// Common interface for all terrain backends.
///
/// Implementations are required to be `Send + Sync` so they can be shared
/// across worker threads.  All inputs and outputs use **metres** and **radians**.
pub trait TerrainQuery: Send + Sync {
    // ------------------------------------------------------------------
    // Core height queries
    // ------------------------------------------------------------------

    /// Return the terrain-surface elevation at `(x_m, y_m)`, in metres.
    ///
    /// For [`GridTerrain`] this is the bilinearly-interpolated grid height.
    /// Out-of-bounds queries return the `ground_plane_m` value.
    fn height_at(&self, x_m: f64, y_m: f64) -> f64;

    /// Return the *analytic* (smooth, formula-based) terrain height at `(x_m, y_m)`.
    ///
    /// For grid-backed terrains this is typically the same as [`height_at`](TerrainQuery::height_at);
    /// for procedurally generated terrains it may bypass the discrete grid.
    fn analytic_height_at(&self, x_m: f64, y_m: f64) -> f64;

    // ------------------------------------------------------------------
    // Differential geometry
    // ------------------------------------------------------------------

    /// Return the outward unit normal of the terrain surface at `(x_m, y_m)`.
    ///
    /// Computed from the gradient: `n = normalize([-dz/dx, -dz/dy, 1])`.
    fn normal_at(&self, x_m: f64, y_m: f64) -> [f64; 3];

    /// Return the terrain height gradient `[dz/dx, dz/dy]` at `(x_m, y_m)`
    /// using a central finite-difference stencil with step `delta_m`.
    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> [f64; 2];

    /// Return the mean curvature of the terrain surface at `(x_m, y_m)`.
    ///
    /// Approximated as `(d²z/dx² + d²z/dy²) / 2` via second-order finite differences
    /// with step `delta_m` (in metres).
    fn curvature_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> f64;

    /// Return the terrain slope magnitude at `(x_m, y_m)`, in **radians**.
    ///
    /// Slope = `atan(|∇h|)` where `∇h` is the gradient vector.
    fn slope_rad_at(&self, x_m: f64, y_m: f64) -> f64;

    // ------------------------------------------------------------------
    // Altitude helpers
    // ------------------------------------------------------------------

    /// Clamp altitude `z_m` so the entity remains at least `min_agl_m` metres
    /// above ground level (AGL) at `(x_m, y_m)`.
    ///
    /// Returns `max(z_m, height_at(x_m, y_m) + min_agl_m)`.
    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64;

    // ------------------------------------------------------------------
    // Line-of-sight / RF propagation
    // ------------------------------------------------------------------

    /// Cast a ray from `origin` to `target` (both `[x_m, y_m, z_m]`) checking
    /// whether the line of sight is blocked by terrain within a `clearance_m`
    /// envelope below the ray.
    ///
    /// Returns `Some(TerrainHit)` at the **first** point where
    /// `ray_z(t) - terrain_z < clearance_m`, or `None` if LOS is clear.
    fn los_raycast(
        &self,
        origin: [f64; 3],
        target: [f64; 3],
        clearance_m: f64,
    ) -> Option<TerrainHit>;

    /// Return `true` if `rx_xyz` is in the **communications shadow** of the terrain
    /// as seen from `tx_xyz`.
    ///
    /// Uses a Fresnel-zone approximation: internally calls
    /// `los_raycast(tx, rx, 5.0)` — a 5 m clearance corresponds roughly to
    /// the first Fresnel zone radius at 2.4 GHz over a 1 km link.
    fn comms_shadow(&self, tx_xyz: [f64; 3], rx_xyz: [f64; 3]) -> bool;

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    /// Return the axis-aligned bounding box of this terrain region.
    fn bounds(&self) -> TerrainBounds;

    /// Return the datum (base) elevation in metres.
    ///
    /// Points below this altitude are guaranteed to be underground.
    fn ground_plane_m(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Shared helpers (free functions used by both implementations)
// ---------------------------------------------------------------------------

/// Default finite-difference step size, in metres.
const DEFAULT_DELTA_M: f64 = 1.0;

/// Number of samples used when marching a ray for LOS checks.
const LOS_SAMPLES: usize = 256;

/// Compute gradient `[dz/dx, dz/dy]` via central differences.
fn compute_gradient<T: TerrainQuery + ?Sized>(
    terrain: &T,
    x_m: f64,
    y_m: f64,
    delta_m: f64,
) -> [f64; 2] {
    let dzdx = (terrain.height_at(x_m + delta_m, y_m)
        - terrain.height_at(x_m - delta_m, y_m))
        / (2.0 * delta_m);
    let dzdy = (terrain.height_at(x_m, y_m + delta_m)
        - terrain.height_at(x_m, y_m - delta_m))
        / (2.0 * delta_m);
    [dzdx, dzdy]
}

/// Compute outward unit normal from gradient.
fn gradient_to_normal(grad: [f64; 2]) -> [f64; 3] {
    let nx = -grad[0];
    let ny = -grad[1];
    let nz = 1.0_f64;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    [nx / len, ny / len, nz / len]
}

/// March a ray from `origin` to `target` checking terrain clearance.
fn march_los<T: TerrainQuery + ?Sized>(
    terrain: &T,
    origin: [f64; 3],
    target: [f64; 3],
    clearance_m: f64,
) -> Option<TerrainHit> {
    let dx = target[0] - origin[0];
    let dy = target[1] - origin[1];
    let dz = target[2] - origin[2];

    for i in 1..=LOS_SAMPLES {
        let t = i as f64 / LOS_SAMPLES as f64;
        let px = origin[0] + t * dx;
        let py = origin[1] + t * dy;
        let pz = origin[2] + t * dz;
        let terrain_z = terrain.height_at(px, py);
        if pz - terrain_z < clearance_m {
            return Some(TerrainHit {
                t,
                point_m: [px, py, pz],
                terrain_z_m: terrain_z,
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------
// FlatTerrain
// ---------------------------------------------------------------------------

/// Constant-height terrain, useful for unit tests and flat-world simulations.
///
/// Every query returns the same `height_m` value, and LOS is always clear
/// (provided entities are above the ground plane).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatTerrain {
    /// Constant terrain height, metres above datum.
    pub height_m: f64,
    /// Spatial extent of this terrain.
    pub terrain_bounds: TerrainBounds,
}

impl FlatTerrain {
    /// Construct a new [`FlatTerrain`].
    ///
    /// # Arguments
    /// * `height_m` – constant surface elevation, metres.
    /// * `bounds`   – horizontal and vertical extent of the terrain.
    pub fn new(height_m: f64, bounds: TerrainBounds) -> Self {
        Self {
            height_m,
            terrain_bounds: bounds,
        }
    }
}

impl TerrainQuery for FlatTerrain {
    #[inline]
    fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
        self.height_m
    }

    #[inline]
    fn analytic_height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
        self.height_m
    }

    fn normal_at(&self, _x_m: f64, _y_m: f64) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }

    fn gradient_at(&self, _x_m: f64, _y_m: f64, _delta_m: f64) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn curvature_at(&self, _x_m: f64, _y_m: f64, _delta_m: f64) -> f64 {
        0.0
    }

    fn slope_rad_at(&self, _x_m: f64, _y_m: f64) -> f64 {
        0.0
    }

    fn clamp_altitude(&self, _x_m: f64, _y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        let floor = self.height_m + min_agl_m;
        if z_m < floor {
            floor
        } else {
            z_m
        }
    }

    fn los_raycast(
        &self,
        origin: [f64; 3],
        target: [f64; 3],
        clearance_m: f64,
    ) -> Option<TerrainHit> {
        march_los(self, origin, target, clearance_m)
    }

    fn comms_shadow(&self, tx_xyz: [f64; 3], rx_xyz: [f64; 3]) -> bool {
        self.los_raycast(tx_xyz, rx_xyz, 5.0).is_some()
    }

    fn bounds(&self) -> TerrainBounds {
        self.terrain_bounds
    }

    fn ground_plane_m(&self) -> f64 {
        self.height_m
    }
}

// ---------------------------------------------------------------------------
// GridTerrain
// ---------------------------------------------------------------------------

/// Bilinearly-interpolated heightmap on a regular grid.
///
/// The grid origin is the **lower-left corner** (minimum X, minimum Y).
/// Cell centres are at `(origin_x_m + (col + 0.5) * cell_size_m, ...)`.
/// Heights are stored in **row-major order**: index `row * cols + col`,
/// where `row = 0` corresponds to the minimum Y row.
///
/// Out-of-bounds queries fall back to `ground_plane_m`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridTerrain {
    /// Flat heightmap buffer, length = `rows * cols`.
    heights: Vec<f64>,
    /// Number of columns (X direction).
    cols: usize,
    /// Number of rows (Y direction).
    rows: usize,
    /// Grid spacing, metres (uniform in X and Y).
    cell_size_m: f64,
    /// X coordinate of the grid origin (lower-left corner), metres.
    origin_x_m: f64,
    /// Y coordinate of the grid origin (lower-left corner), metres.
    origin_y_m: f64,
    /// Datum elevation returned for out-of-bounds queries, metres.
    floor_m: f64,
}

impl GridTerrain {
    /// Construct a [`GridTerrain`] from an explicit heightmap.
    ///
    /// # Arguments
    /// * `heights`        – row-major height buffer of length `rows * cols`.
    /// * `cols`           – number of grid columns (X axis).
    /// * `rows`           – number of grid rows (Y axis).
    /// * `cell_size_m`    – uniform grid spacing, metres.
    /// * `origin_x_m`     – X of the lower-left corner, metres.
    /// * `origin_y_m`     – Y of the lower-left corner, metres.
    /// * `ground_plane_m` – fallback elevation for out-of-bounds queries, metres.
    ///
    /// # Panics
    /// Panics if `heights.len() != rows * cols`.
    pub fn new(
        heights: Vec<f64>,
        cols: usize,
        rows: usize,
        cell_size_m: f64,
        origin_x_m: f64,
        origin_y_m: f64,
        ground_plane_m: f64,
    ) -> Self {
        assert_eq!(
            heights.len(),
            rows * cols,
            "heights buffer length ({}) must equal rows * cols ({})",
            heights.len(),
            rows * cols
        );
        assert!(cell_size_m > 0.0, "cell_size_m must be positive");
        Self {
            heights,
            cols,
            rows,
            cell_size_m,
            origin_x_m,
            origin_y_m,
            floor_m: ground_plane_m,
        }
    }

    /// Construct a flat (constant-height) [`GridTerrain`] covering `bounds`.
    ///
    /// Useful when you need a grid backend (e.g. to test gradient code) but
    /// want a featureless surface.
    ///
    /// # Arguments
    /// * `ground_plane_m` – constant surface height, metres.
    /// * `bounds`         – spatial extent.
    /// * `cell_size_m`    – grid spacing, metres.
    pub fn from_flat(ground_plane_m: f64, bounds: TerrainBounds, cell_size_m: f64) -> Self {
        assert!(cell_size_m > 0.0, "cell_size_m must be positive");
        let cols = ((bounds.width_m() / cell_size_m).ceil() as usize).max(2);
        let rows = ((bounds.depth_m() / cell_size_m).ceil() as usize).max(2);
        let heights = vec![ground_plane_m; rows * cols];
        Self {
            heights,
            cols,
            rows,
            cell_size_m,
            origin_x_m: bounds.x_min_m,
            origin_y_m: bounds.y_min_m,
            floor_m: ground_plane_m,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Convert world coordinates to fractional grid coordinates `(fx, fy)`.
    #[inline]
    fn world_to_frac(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let fx = (x_m - self.origin_x_m) / self.cell_size_m;
        let fy = (y_m - self.origin_y_m) / self.cell_size_m;
        (fx, fy)
    }

    /// Return `true` if fractional grid coords are within valid interpolation range.
    #[inline]
    fn frac_in_bounds(&self, fx: f64, fy: f64) -> bool {
        fx >= 0.0
            && fy >= 0.0
            && fx <= (self.cols as f64 - 1.0)
            && fy <= (self.rows as f64 - 1.0)
    }

    /// Bilinear interpolation of the heightmap at fractional grid coordinates.
    ///
    /// Returns `floor_m` for out-of-bounds queries (conservative fallback).
    fn bilinear(&self, fx: f64, fy: f64) -> f64 {
        if !self.frac_in_bounds(fx, fy) {
            return self.floor_m;
        }
        let col0 = fx.floor() as usize;
        let row0 = fy.floor() as usize;
        // Clamp upper indices to grid edge.
        let col1 = (col0 + 1).min(self.cols - 1);
        let row1 = (row0 + 1).min(self.rows - 1);

        let tx = fx - col0 as f64;
        let ty = fy - row0 as f64;

        let h00 = self.heights[row0 * self.cols + col0];
        let h10 = self.heights[row0 * self.cols + col1];
        let h01 = self.heights[row1 * self.cols + col0];
        let h11 = self.heights[row1 * self.cols + col1];

        // Standard bilinear interpolation formula.
        h00 * (1.0 - tx) * (1.0 - ty)
            + h10 * tx * (1.0 - ty)
            + h01 * (1.0 - tx) * ty
            + h11 * tx * ty
    }

    /// Compute [`TerrainBounds`] from the grid extent and height range.
    fn compute_bounds(&self) -> TerrainBounds {
        let x_max = self.origin_x_m + self.cols as f64 * self.cell_size_m;
        let y_max = self.origin_y_m + self.rows as f64 * self.cell_size_m;
        let (z_min, z_max) = self
            .heights
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        TerrainBounds {
            x_min_m: self.origin_x_m,
            x_max_m: x_max,
            y_min_m: self.origin_y_m,
            y_max_m: y_max,
            z_min_m: z_min,
            z_max_m: z_max,
        }
    }
}

impl TerrainQuery for GridTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        let (fx, fy) = self.world_to_frac(x_m, y_m);
        self.bilinear(fx, fy)
    }

    fn analytic_height_at(&self, x_m: f64, y_m: f64) -> f64 {
        // Grid terrain has no closed-form analytic formula; use the same
        // bilinearly-interpolated value.
        self.height_at(x_m, y_m)
    }

    fn normal_at(&self, x_m: f64, y_m: f64) -> [f64; 3] {
        let grad = compute_gradient(self, x_m, y_m, DEFAULT_DELTA_M);
        gradient_to_normal(grad)
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> [f64; 2] {
        compute_gradient(self, x_m, y_m, delta_m)
    }

    fn curvature_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> f64 {
        let h = self.height_at(x_m, y_m);
        let hpx = self.height_at(x_m + delta_m, y_m);
        let hmx = self.height_at(x_m - delta_m, y_m);
        let hpy = self.height_at(x_m, y_m + delta_m);
        let hmy = self.height_at(x_m, y_m - delta_m);
        let d2zdx2 = (hpx - 2.0 * h + hmx) / (delta_m * delta_m);
        let d2zdy2 = (hpy - 2.0 * h + hmy) / (delta_m * delta_m);
        (d2zdx2 + d2zdy2) / 2.0
    }

    fn slope_rad_at(&self, x_m: f64, y_m: f64) -> f64 {
        let [gx, gy] = compute_gradient(self, x_m, y_m, DEFAULT_DELTA_M);
        let mag = (gx * gx + gy * gy).sqrt();
        mag.atan()
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        let floor = self.height_at(x_m, y_m) + min_agl_m;
        if z_m < floor {
            floor
        } else {
            z_m
        }
    }

    fn los_raycast(
        &self,
        origin: [f64; 3],
        target: [f64; 3],
        clearance_m: f64,
    ) -> Option<TerrainHit> {
        march_los(self, origin, target, clearance_m)
    }

    fn comms_shadow(&self, tx_xyz: [f64; 3], rx_xyz: [f64; 3]) -> bool {
        // 5 m Fresnel-zone clearance approximation for 2.4 GHz, ~1 km link.
        self.los_raycast(tx_xyz, rx_xyz, 5.0).is_some()
    }

    fn bounds(&self) -> TerrainBounds {
        self.compute_bounds()
    }

    fn ground_plane_m(&self) -> f64 {
        self.floor_m
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience builder: 100×100 m flat terrain at Z = 0.
    fn unit_flat() -> FlatTerrain {
        FlatTerrain::new(
            0.0,
            TerrainBounds::new(0.0, 100.0, 0.0, 100.0, 0.0, 0.0),
        )
    }

    /// Build an 11×11 node grid (10×10 m domain, 1 m spacing) with a 20 m hill
    /// at the centre node (5, 5).
    fn hill_grid() -> GridTerrain {
        let cols = 11_usize;
        let rows = 11_usize;
        let mut heights = vec![0.0_f64; rows * cols];
        heights[5 * cols + 5] = 20.0;
        GridTerrain::new(heights, cols, rows, 1.0, 0.0, 0.0, 0.0)
    }

    // -----------------------------------------------------------------------
    // FlatTerrain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flat_terrain_height() {
        let t = unit_flat();
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert_eq!(t.height_at(50.0, 50.0), 0.0);
        assert_eq!(t.height_at(100.0, 100.0), 0.0);

        // Non-zero flat terrain.
        let t2 = FlatTerrain::new(
            42.5,
            TerrainBounds::new(-500.0, 500.0, -500.0, 500.0, 42.5, 42.5),
        );
        assert!((t2.height_at(123.4, -99.9) - 42.5).abs() < 1e-12);
    }

    #[test]
    fn test_flat_terrain_los() {
        let t = unit_flat();
        // LOS well above surface (Z = 50 m) — should be clear.
        let hit = t.los_raycast([10.0, 10.0, 50.0], [90.0, 90.0, 50.0], 0.0);
        assert!(
            hit.is_none(),
            "expected clear LOS above flat terrain, got {:?}",
            hit
        );

        // LOS below surface (Z = -1 m) — should be blocked.
        let hit2 = t.los_raycast([10.0, 10.0, -1.0], [90.0, 90.0, -1.0], 0.0);
        assert!(
            hit2.is_some(),
            "expected blocked LOS below flat terrain surface"
        );
    }

    // -----------------------------------------------------------------------
    // GridTerrain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_terrain_height() {
        // 3×3 nodes, 10 m spacing:
        //   row 0: 0  0  0
        //   row 1: 0  4  0
        //   row 2: 0  0  0
        let cols = 3_usize;
        let rows = 3_usize;
        let mut heights = vec![0.0_f64; rows * cols];
        heights[1 * cols + 1] = 4.0;
        let t = GridTerrain::new(heights, cols, rows, 10.0, 0.0, 0.0, 0.0);

        // Exact node values.
        assert!((t.height_at(0.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((t.height_at(10.0, 10.0) - 4.0).abs() < 1e-10);

        // Bilinear at (5, 5): fractional (0.5, 0.5) in the lower-left cell.
        // h = 0*(0.5)(0.5) + 0*(0.5)(0.5) + 0*(0.5)(0.5) + 4*(0.5)(0.5) = 1.0
        let h_mid = t.height_at(5.0, 5.0);
        assert!(
            (h_mid - 1.0).abs() < 1e-10,
            "bilinear mid expected 1.0, got {h_mid}"
        );
    }

    #[test]
    fn test_grid_terrain_los_blocked() {
        let t = hill_grid();

        // Ray at Z = 2 m through (5, 5) — blocked by 20 m hill.
        let hit = t.los_raycast([5.0, 0.0, 2.0], [5.0, 10.0, 2.0], 0.0);
        assert!(
            hit.is_some(),
            "expected LOS to be blocked by the hill, got None"
        );
        let h = hit.unwrap();
        assert!(
            h.terrain_z_m > 0.0,
            "expected hit terrain_z_m > 0, got {}",
            h.terrain_z_m
        );

        // Ray at Z = 30 m — clears the hill.
        let clear = t.los_raycast([5.0, 0.0, 30.0], [5.0, 10.0, 30.0], 0.0);
        assert!(
            clear.is_none(),
            "expected clear LOS above the hill at Z = 30 m, got {:?}",
            clear
        );
    }

    #[test]
    fn test_clamp_altitude() {
        let t = unit_flat(); // surface at Z = 0
        // Already above floor.
        assert!((t.clamp_altitude(50.0, 50.0, 10.0, 5.0) - 10.0).abs() < 1e-12);
        // Below floor: clamped to 0 + 5 = 5.
        assert!((t.clamp_altitude(50.0, 50.0, 3.0, 5.0) - 5.0).abs() < 1e-12);
        // Exactly at floor.
        assert!((t.clamp_altitude(50.0, 50.0, 5.0, 5.0) - 5.0).abs() < 1e-12);

        // Grid terrain with hill: at (5, 5) height ≈ 20 m; min_agl = 2 → floor = 22 m.
        let g = hill_grid();
        let z_clamped = g.clamp_altitude(5.0, 5.0, 15.0, 2.0);
        let floor = g.height_at(5.0, 5.0) + 2.0;
        assert!(
            (z_clamped - floor).abs() < 1e-9,
            "expected z_clamped ≈ {floor}, got {z_clamped}"
        );
    }

    #[test]
    fn test_gradient() {
        // Tilted plane: h(x, y) = 2*x + 3*y → gradient = [2, 3].
        let cols = 21_usize;
        let rows = 21_usize;
        let cell = 1.0_f64;
        let mut heights = vec![0.0_f64; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let x = col as f64 * cell;
                let y = row as f64 * cell;
                heights[row * cols + col] = 2.0 * x + 3.0 * y;
            }
        }
        let t = GridTerrain::new(heights, cols, rows, cell, 0.0, 0.0, 0.0);
        let [gx, gy] = t.gradient_at(10.0, 10.0, 1.0);
        assert!((gx - 2.0).abs() < 1e-6, "expected gx ≈ 2, got {gx}");
        assert!((gy - 3.0).abs() < 1e-6, "expected gy ≈ 3, got {gy}");
    }

    #[test]
    fn test_curvature() {
        // Paraboloid h(x, y) = x² + y²:
        // d²z/dx² = 2, d²z/dy² = 2, mean curvature = (2 + 2) / 2 = 2.
        let cols = 41_usize;
        let rows = 41_usize;
        let cell = 1.0_f64;
        let origin = -20.0_f64;
        let mut heights = vec![0.0_f64; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let x = origin + col as f64 * cell;
                let y = origin + row as f64 * cell;
                heights[row * cols + col] = x * x + y * y;
            }
        }
        let t = GridTerrain::new(heights, cols, rows, cell, origin, origin, 0.0);
        let curv = t.curvature_at(0.0, 0.0, 1.0);
        assert!(
            (curv - 2.0).abs() < 1e-6,
            "expected curvature ≈ 2.0 for paraboloid, got {curv}"
        );
    }

    #[test]
    fn test_out_of_bounds() {
        let t = GridTerrain::new(
            vec![5.0; 4],
            2,
            2,
            10.0,
            0.0,
            0.0,
            -1.0, // ground_plane_m (OOB fallback)
        );
        // Within bounds.
        assert!((t.height_at(5.0, 5.0) - 5.0).abs() < 1e-10);
        // Outside bounds: returns floor_m = -1.0 (conservative).
        let h_oob = t.height_at(999.0, 999.0);
        assert!(
            (h_oob - (-1.0)).abs() < 1e-10,
            "out-of-bounds query should return ground_plane_m, got {h_oob}"
        );
        let h_neg = t.height_at(-1.0, -1.0);
        assert!(
            (h_neg - (-1.0)).abs() < 1e-10,
            "negative OOB query should return ground_plane_m, got {h_neg}"
        );
    }
}
