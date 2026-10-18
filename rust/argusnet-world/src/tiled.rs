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
/// The ground along part of a segment: from fraction `t0` to `t1` of it,
/// the height is `c0 + c1·s + c2·s²` with `s` running 0 to 1 across the
/// piece (see [`TiledTerrain::height_along`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightPiece {
    pub t0: f64,
    pub t1: f64,
    pub coefficients: [f64; 3],
}

impl HeightPiece {
    pub fn height_at(&self, s: f64) -> f64 {
        let [c0, c1, c2] = self.coefficients;
        c0 + (c1 + c2 * s) * s
    }
}

/// The fractions strictly inside (0, 1) at which the coordinate going from
/// `a` to `b` crosses `origin + k·step` for some whole `k` (`step` infinite:
/// `origin` alone).
fn crossings(a: f64, b: f64, origin: f64, step: f64, out: &mut Vec<f64>) {
    let d = b - a;
    if d.abs() < 1e-12 {
        return;
    }
    let mut push = |line: f64| {
        let t = (line - a) / d;
        if t > 0.0 && t < 1.0 {
            out.push(t);
        }
    };
    if step.is_infinite() {
        push(origin);
        return;
    }
    let (lo, hi) = (a.min(b), a.max(b));
    let (k0, k1) = (
        ((lo - origin) / step).ceil(),
        ((hi - origin) / step).floor(),
    );
    let mut k = k0;
    while k <= k1 {
        push(origin + k * step);
        k += 1.0;
    }
}

fn sort_breaks(breaks: &mut Vec<f64>) {
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
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
    /// The lod-0 height under the straight segment from `from` to `to`,
    /// exactly: the segment is cut where it crosses a tile's edge, a cell's
    /// edge or the terrain's bounds, and along each piece a bilinear cell
    /// (or the floor, off the tiles) is a quadratic in the distance along
    /// it. The height at a point of a piece is that quadratic raised to
    /// [`Self::floor_m`], as [`Self::height_lod`] gives it; a check against
    /// the ground takes both. Pieces in order, covering the whole segment.
    pub fn height_along(&self, from: [f64; 2], to: [f64; 2]) -> Vec<HeightPiece> {
        let span = self.tile_size_cells as f64 * self.base_resolution_m;
        let mut coarse = vec![0.0, 1.0];
        for axis in 0..2 {
            let (origin, end) = if axis == 0 {
                (self.bounds.x_min_m, self.bounds.x_max_m)
            } else {
                (self.bounds.y_min_m, self.bounds.y_max_m)
            };
            crossings(from[axis], to[axis], origin, span, &mut coarse);
            crossings(from[axis], to[axis], end, f64::INFINITY, &mut coarse);
        }
        sort_breaks(&mut coarse);
        let at = |t: f64| {
            [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            ]
        };
        let mut breaks = Vec::with_capacity(coarse.len());
        for pair in coarse.windows(2) {
            breaks.push(pair[0]);
            let mid = at(0.5 * (pair[0] + pair[1]));
            if let Some(tile) = self.tile(mid[0], mid[1], 0) {
                let mut cells = Vec::new();
                for axis in 0..2 {
                    let origin = if axis == 0 {
                        tile.x_min_m
                    } else {
                        tile.y_min_m
                    };
                    crossings(from[axis], to[axis], origin, tile.cell_size_m, &mut cells);
                }
                breaks.extend(cells.into_iter().filter(|t| *t > pair[0] && *t < pair[1]));
            }
        }
        breaks.push(1.0);
        sort_breaks(&mut breaks);
        breaks
            .windows(2)
            .filter(|pair| pair[1] > pair[0])
            .map(|pair| {
                let (t0, t1) = (pair[0], pair[1]);
                // Inside the piece, away from its ends, so a step in the
                // height at an edge (onto the floor, off the tiles) is not
                // read as part of it.
                let sample = |s: f64| {
                    let p = at(t0 + (t1 - t0) * s);
                    let mid = at(0.5 * (t0 + t1));
                    self.tile(mid[0], mid[1], 0).map_or(self.floor_m, |t| {
                        bilinear(
                            &t.heights_m,
                            (p[0] - t.x_min_m) / t.cell_size_m,
                            (p[1] - t.y_min_m) / t.cell_size_m,
                        )
                    })
                };
                let (y1, y2, y3) = (sample(0.25), sample(0.5), sample(0.75));
                let c2 = 8.0 * (y1 - 2.0 * y2 + y3);
                let c1 = 2.0 * (y3 - y1) - c2;
                let c0 = y2 - 0.5 * c1 - 0.25 * c2;
                HeightPiece {
                    t0,
                    t1,
                    coefficients: [c0, c1, c2],
                }
            })
            .collect()
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

#[cfg(test)]
mod height_along_tests {
    use super::*;

    /// 2 x 2 tiles of 4 x 4 cells of 5 m over 0..40 m, with uneven heights
    /// and one tile missing (the floor there).
    fn terrain() -> TiledTerrain {
        let tile = |tx: i64, ty: i64| {
            let heights_m = (0..5)
                .map(|row| {
                    (0..5)
                        .map(|col| {
                            let k = (tx * 7 + ty * 13 + row * 5 + col) as f64;
                            10.0 + 9.0 * (k * 1.7).sin()
                                + if (row, col) == (2, 3) { 40.0 } else { 0.0 }
                        })
                        .collect()
                })
                .collect();
            Tile {
                tx,
                ty,
                lod: 0,
                x_min_m: tx as f64 * 20.0,
                y_min_m: ty as f64 * 20.0,
                cell_size_m: 5.0,
                heights_m,
                min_pyramid: Vec::new(),
                max_pyramid: Vec::new(),
            }
        };
        TiledTerrain::new(
            Bounds2d::new(0.0, 40.0, 0.0, 40.0),
            4,
            5.0,
            2.0,
            vec![tile(0, 0), tile(1, 0), tile(0, 1)],
        )
        .unwrap()
    }

    /// The pieces are the height: at thousands of points along segments
    /// that cross cells, tiles, the missing tile and the bounds, the piece
    /// raised to the floor is the height the terrain reports.
    #[test]
    fn the_pieces_are_the_height_along_the_segment() {
        let terrain = terrain();
        for (from, to) in [
            ([-3.0, 1.3], [43.0, 38.9]),
            ([7.1, 2.2], [7.1, 37.0]),
            ([39.0, 3.0], [1.0, 22.7]),
            ([12.4, 11.9], [13.0, 12.1]),
        ] {
            let pieces = terrain.height_along(from, to);
            assert_eq!(pieces[0].t0, 0.0);
            assert_eq!(pieces.last().unwrap().t1, 1.0);
            assert!(pieces.windows(2).all(|w| w[0].t1 == w[1].t0));
            for k in 0..=4000 {
                let t = k as f64 / 4000.0;
                let x = from[0] + (to[0] - from[0]) * t;
                let y = from[1] + (to[1] - from[1]) * t;
                let truth = terrain.height_at(x, y);
                // A point on an edge belongs to either piece; one of them
                // gives the height.
                let ok = pieces.iter().filter(|p| p.t0 <= t && t <= p.t1).any(|p| {
                    let s = (t - p.t0) / (p.t1 - p.t0);
                    (p.height_at(s).max(terrain.floor_m) - truth).abs() < 1e-9
                });
                assert!(ok, "{from:?} -> {to:?} at t = {t}: {truth}");
            }
        }
    }
}
