//! Replay raster adapter. The wire layout remains unchanged; sampling belongs here.
use serde::Deserialize;
#[derive(Debug, Clone, Deserialize)]
pub struct TerrainRaster {
    pub x_min_m: f32,
    pub x_max_m: f32,
    pub y_min_m: f32,
    pub y_max_m: f32,
    pub rows: usize,
    pub cols: usize,
    pub heights_m: Vec<Vec<f32>>,
}
impl TerrainRaster {
    /// Missing coverage or malformed raster data is not a zero-metre surface.
    pub fn height_at(&self, x: f32, y: f32) -> Option<f32> {
        if !x.is_finite()
            || !y.is_finite()
            || self.rows < 2
            || self.cols < 2
            || self.heights_m.len() != self.rows
            || self.x_max_m <= self.x_min_m
            || self.y_max_m <= self.y_min_m
            || x < self.x_min_m
            || x > self.x_max_m
            || y < self.y_min_m
            || y > self.y_max_m
        {
            return None;
        }
        let tx = (x as f64 - self.x_min_m as f64) / (self.x_max_m as f64 - self.x_min_m as f64)
            * (self.cols - 1) as f64;
        let ty = (y as f64 - self.y_min_m as f64) / (self.y_max_m as f64 - self.y_min_m as f64)
            * (self.rows - 1) as f64;
        let c = (tx.floor() as usize).min(self.cols - 2);
        let r = (ty.floor() as usize).min(self.rows - 2);
        let a = self.heights_m.get(r)?;
        let b = self.heights_m.get(r + 1)?;
        if a.len() != self.cols || b.len() != self.cols {
            return None;
        }
        let h = [a[c] as f64, a[c + 1] as f64, b[c] as f64, b[c + 1] as f64];
        if h.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(crate::tiled::bilinear_cell(h, tx - c as f64, ty - r as f64) as f32)
    }
}
