//! Categorical land-cover sampling and vegetation density.
use crate::obstacles::Bounds2d;
use serde::Deserialize;
use std::collections::BTreeMap;
#[derive(Debug, Clone, Deserialize)]
pub struct LandCoverTile {
    pub tx: i64,
    pub ty: i64,
    pub x_min_m: f64,
    pub y_min_m: f64,
    pub cell_size_m: f64,
    pub classes: Vec<Vec<u8>>,
    pub density: Option<Vec<Vec<u8>>>,
}
#[derive(Debug, Clone)]
pub struct LandCoverLayer {
    bounds: Bounds2d,
    span: f64,
    tiles: BTreeMap<(i64, i64), LandCoverTile>,
}
impl LandCoverLayer {
    pub fn new(
        bounds: Bounds2d,
        tile_cells: usize,
        resolution: f64,
        tiles: Vec<LandCoverTile>,
    ) -> Result<Self, String> {
        if tile_cells == 0 || !resolution.is_finite() || resolution <= 0.0 {
            return Err("invalid land-cover resolution".into());
        }
        let mut out = Self {
            bounds,
            span: tile_cells as f64 * resolution,
            tiles: BTreeMap::new(),
        };
        for t in tiles {
            if t.classes.is_empty()
                || t.classes[0].is_empty()
                || t.classes.iter().any(|r| r.len() != t.classes[0].len())
                || !t.cell_size_m.is_finite()
                || t.cell_size_m <= 0.0
            {
                return Err("invalid land-cover tile".into());
            }
            if t.density.as_ref().is_some_and(|d| {
                d.len() != t.classes.len() || d.iter().any(|r| r.len() != t.classes[0].len())
            }) {
                return Err("density shape differs from land cover".into());
            }
            out.tiles.insert((t.tx, t.ty), t);
        }
        Ok(out)
    }
    pub fn sample(&self, x: f64, y: f64) -> (u8, f64) {
        if !self.bounds.contains_xy(x, y) {
            return (0, 0.0);
        }
        let tx = (((x - self.bounds.x_min_m) / self.span).floor() as i64)
            .min(((self.bounds.x_max_m - self.bounds.x_min_m) / self.span).ceil() as i64 - 1);
        let ty = (((y - self.bounds.y_min_m) / self.span).floor() as i64)
            .min(((self.bounds.y_max_m - self.bounds.y_min_m) / self.span).ceil() as i64 - 1);
        let Some(t) = self.tiles.get(&(tx, ty)) else {
            return (0, 0.0);
        };
        let c = (((x - t.x_min_m) / t.cell_size_m).floor().max(0.0) as usize)
            .min(t.classes[0].len() - 1);
        let r =
            (((y - t.y_min_m) / t.cell_size_m).floor().max(0.0) as usize).min(t.classes.len() - 1);
        (
            t.classes[r][c],
            t.density.as_ref().map_or(0.0, |d| d[r][c] as f64 / 255.0),
        )
    }
}
