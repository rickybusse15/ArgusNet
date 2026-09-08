//! Tiled terrain, shared interpolation and conservative min/max mip pyramids.
use crate::{obstacles::Bounds2d, TerrainBounds, TerrainHit, TerrainQuery};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub fn validate_grid(grid: &[Vec<f64>]) -> Result<(), String> {
    if grid.len() < 2
        || grid[0].len() < 2
        || grid
            .iter()
            .any(|r| r.len() != grid[0].len() || r.iter().any(|v| !v.is_finite()))
    {
        return Err("height grid must be finite, rectangular and at least 2 by 2".into());
    }
    Ok(())
}
/// Bilinear interpolation in grid coordinates; callers decide coverage policy.
pub fn bilinear(grid: &[Vec<f64>], x: f64, y: f64) -> f64 {
    let col = (x.floor().max(0.0) as usize).min(grid[0].len() - 2);
    let row = (y.floor().max(0.0) as usize).min(grid.len() - 2);
    let tx = (x - col as f64).clamp(0.0, 1.0);
    let ty = (y - row as f64).clamp(0.0, 1.0);
    bilinear_cell(
        [
            grid[row][col],
            grid[row][col + 1],
            grid[row + 1][col],
            grid[row + 1][col + 1],
        ],
        tx,
        ty,
    )
}
pub fn pyramid(grid: &[Vec<f64>], maximum: bool) -> Result<Vec<Vec<Vec<f64>>>, String> {
    validate_grid(grid)?;
    let mut levels = vec![grid.to_vec()];
    while levels.last().unwrap().len() > 2 || levels.last().unwrap()[0].len() > 2 {
        let g = levels.last().unwrap();
        let rows = (g.len() - 1).div_ceil(2) + 1;
        let cols = (g[0].len() - 1).div_ceil(2) + 1;
        let mut next = vec![vec![0.0; cols]; rows];
        for (y, row) in next.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                let y0 = (y * 2).min(g.len() - 2);
                let x0 = (x * 2).min(g[0].len() - 2);
                *v = if maximum {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                for r in &g[y0..(y0 + 3).min(g.len())] {
                    for h in &r[x0..(x0 + 3).min(r.len())] {
                        *v = if maximum { v.max(*h) } else { v.min(*h) };
                    }
                }
            }
        }
        levels.push(next);
    }
    Ok(levels)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    pub tx: i64,
    pub ty: i64,
    pub lod: usize,
    pub x_min_m: f64,
    pub y_min_m: f64,
    pub cell_size_m: f64,
    pub heights_m: Vec<Vec<f64>>,
    #[serde(default)]
    pub min_pyramid: Vec<Vec<Vec<f64>>>,
    #[serde(default)]
    pub max_pyramid: Vec<Vec<Vec<f64>>>,
}
impl Tile {
    pub fn prepare(&mut self) -> Result<(), String> {
        validate_grid(&self.heights_m)?;
        if !self.cell_size_m.is_finite() || self.cell_size_m <= 0.0 {
            return Err("invalid tile resolution".into());
        }
        self.min_pyramid = pyramid(&self.heights_m, false)?;
        self.max_pyramid = pyramid(&self.heights_m, true)?;
        Ok(())
    }
    pub fn bounds(&self) -> Bounds2d {
        Bounds2d::new(
            self.x_min_m,
            self.x_min_m + (self.heights_m[0].len() - 1) as f64 * self.cell_size_m,
            self.y_min_m,
            self.y_min_m + (self.heights_m.len() - 1) as f64 * self.cell_size_m,
        )
    }
    pub fn max_height(&self) -> f64 {
        self.max_pyramid
            .last()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}
#[derive(Debug, Clone)]
pub struct TiledTerrain {
    pub bounds: Bounds2d,
    pub tile_size_cells: usize,
    pub base_resolution_m: f64,
    pub floor_m: f64,
    pub tiles: BTreeMap<(usize, i64, i64), Tile>,
}
impl TiledTerrain {
    pub fn new(
        bounds: Bounds2d,
        tile_size_cells: usize,
        base_resolution_m: f64,
        floor_m: f64,
        tiles: Vec<Tile>,
    ) -> Result<Self, String> {
        if tile_size_cells == 0
            || !base_resolution_m.is_finite()
            || base_resolution_m <= 0.0
            || !floor_m.is_finite()
            || bounds.x_max_m <= bounds.x_min_m
            || bounds.y_max_m <= bounds.y_min_m
        {
            return Err("invalid terrain geometry".into());
        }
        let mut result = Self {
            bounds,
            tile_size_cells,
            base_resolution_m,
            floor_m,
            tiles: BTreeMap::new(),
        };
        for mut t in tiles {
            t.prepare()?;
            let key = (t.lod, t.tx, t.ty);
            if result.tiles.insert(key, t).is_some() {
                return Err("duplicate terrain tile".into());
            }
        }
        Ok(result)
    }
    pub fn tile(&self, x: f64, y: f64, lod: usize) -> Option<&Tile> {
        if !self.bounds.contains_xy(x, y) {
            return None;
        }
        let span = self.tile_size_cells as f64 * self.base_resolution_m;
        let tx = (((x - self.bounds.x_min_m) / span).floor() as i64)
            .min(((self.bounds.x_max_m - self.bounds.x_min_m) / span).ceil() as i64 - 1);
        let ty = (((y - self.bounds.y_min_m) / span).floor() as i64)
            .min(((self.bounds.y_max_m - self.bounds.y_min_m) / span).ceil() as i64 - 1);
        self.tiles.get(&(lod, tx, ty))
    }
    pub fn height_lod(&self, x: f64, y: f64, lod: usize) -> f64 {
        self.tile(x, y, lod).map_or(self.floor_m, |t| {
            bilinear(
                &t.heights_m,
                (x - t.x_min_m) / t.cell_size_m,
                (y - t.y_min_m) / t.cell_size_m,
            )
            .max(self.floor_m)
        })
    }
}
impl TerrainQuery for TiledTerrain {
    fn height_at(&self, x: f64, y: f64) -> f64 {
        self.height_lod(x, y, 0)
    }
    fn analytic_height_at(&self, x: f64, y: f64) -> f64 {
        self.height_at(x, y)
    }
    fn gradient_at(&self, x: f64, y: f64, d: f64) -> [f64; 2] {
        gradient(|x, y| self.height_at(x, y), x, y, d.max(0.25))
    }
    fn normal_at(&self, x: f64, y: f64) -> [f64; 3] {
        normal(self.gradient_at(x, y, self.base_resolution_m * 0.5))
    }
    fn curvature_at(&self, x: f64, y: f64, d: f64) -> f64 {
        curvature(|x, y| self.height_at(x, y), x, y, d)
    }
    fn slope_rad_at(&self, x: f64, y: f64) -> f64 {
        let g = self.gradient_at(x, y, 1.0);
        g[0].hypot(g[1]).atan()
    }
    fn clamp_altitude(&self, x: f64, y: f64, z: f64, agl: f64) -> f64 {
        z.max(self.height_at(x, y) + agl.max(0.0))
    }
    fn los_raycast(&self, a: [f64; 3], b: [f64; 3], c: f64) -> Option<TerrainHit> {
        crate::visibility::terrain_intersection(self, a, b, c).map(|(_, t, p)| TerrainHit {
            t: t.unwrap_or(0.0),
            point_m: p,
            terrain_z_m: p[2] - c,
        })
    }
    fn comms_shadow(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        self.los_raycast(a, b, 5.0).is_some()
    }
    fn bounds(&self) -> TerrainBounds {
        let lo = self
            .tiles
            .values()
            .flat_map(|t| t.heights_m.iter().flatten())
            .copied()
            .fold(self.floor_m, f64::min);
        let hi = self
            .tiles
            .values()
            .map(Tile::max_height)
            .fold(self.floor_m, f64::max);
        TerrainBounds::new(
            self.bounds.x_min_m,
            self.bounds.x_max_m,
            self.bounds.y_min_m,
            self.bounds.y_max_m,
            lo,
            hi,
        )
    }
    fn ground_plane_m(&self) -> f64 {
        self.floor_m
    }
}
pub fn gradient(h: impl Fn(f64, f64) -> f64, x: f64, y: f64, d: f64) -> [f64; 2] {
    [
        (h(x + d, y) - h(x - d, y)) / (2.0 * d),
        (h(x, y + d) - h(x, y - d)) / (2.0 * d),
    ]
}
pub fn curvature(h: impl Fn(f64, f64) -> f64, x: f64, y: f64, d: f64) -> f64 {
    let d = d.max(0.1);
    (h(x + d, y) + h(x - d, y) + h(x, y + d) + h(x, y - d) - 4.0 * h(x, y)) / (d * d)
}
pub fn normal(g: [f64; 2]) -> [f64; 3] {
    let len = (g[0] * g[0] + g[1] * g[1] + 1.0).sqrt();
    [-g[0] / len, -g[1] / len, 1.0 / len]
}

/// Shared interpolation kernel for grids, tiles and replay rasters.
pub fn bilinear_cell(h: [f64; 4], tx: f64, ty: f64) -> f64 {
    let a = h[0] + (h[1] - h[0]) * tx;
    let b = h[2] + (h[3] - h[2]) * tx;
    a + (b - a) * ty
}

impl TiledTerrain {
    /// Split a source raster into padded tiles and materialize requested LODs.
    pub fn from_grid(
        bounds: Bounds2d,
        grid: &[Vec<f64>],
        resolution: f64,
        tile_cells: usize,
        floor: f64,
        lod_resolutions: &[f64],
    ) -> Result<Self, String> {
        validate_grid(grid)?;
        if tile_cells == 0 || !resolution.is_finite() || resolution <= 0.0 {
            return Err("invalid grid resolution".into());
        }
        let resolutions = if lod_resolutions.is_empty() {
            vec![resolution]
        } else {
            lod_resolutions.to_vec()
        };
        if resolutions[0] != resolution
            || resolutions
                .iter()
                .any(|r| !r.is_finite() || *r < resolution)
        {
            return Err("LODs must start at the base resolution and be no finer".into());
        }
        let span = tile_cells as f64 * resolution;
        let mut tiles = vec![];
        for (lod, cell) in resolutions.iter().copied().enumerate() {
            let n = (span / cell).ceil() as usize + 1;
            for ty in 0..(grid.len() - 1).div_ceil(tile_cells) {
                for tx in 0..(grid[0].len() - 1).div_ceil(tile_cells) {
                    let mut heights = vec![vec![0.0; n]; n];
                    for (r, row) in heights.iter_mut().enumerate() {
                        for (c, v) in row.iter_mut().enumerate() {
                            *v = bilinear(
                                grid,
                                tx as f64 * tile_cells as f64 + c as f64 * cell / resolution,
                                ty as f64 * tile_cells as f64 + r as f64 * cell / resolution,
                            )
                            .max(floor);
                        }
                    }
                    tiles.push(Tile {
                        tx: tx as i64,
                        ty: ty as i64,
                        lod,
                        x_min_m: bounds.x_min_m + tx as f64 * span,
                        y_min_m: bounds.y_min_m + ty as f64 * span,
                        cell_size_m: cell,
                        heights_m: heights,
                        min_pyramid: vec![],
                        max_pyramid: vec![],
                    });
                }
            }
        }
        Self::new(bounds, tile_cells, resolution, floor, tiles)
    }
}
