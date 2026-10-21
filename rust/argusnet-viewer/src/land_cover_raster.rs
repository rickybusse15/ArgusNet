//! Land-cover classes as the terrain material tints by them (P2.1).
//!
//! A scene built from an environment carries `metadata/land_cover.json`: the
//! bundle's own `land_cover` payload, plus the legend and each class's style
//! colour. [`LandCoverRaster`] rebuilds the runtime's
//! [`argusnet_world::land_cover::LandCoverLayer`] from it and samples that,
//! one class per cell, into an `R8Uint` texture read with `textureLoad`. So
//! the class is always the runtime's, by the runtime's rule, never
//! interpolated between two classes.
//!
//! Each cell is sampled at its centre, clamped into the bounds: the last cell
//! of a ragged extent has its centre past the edge, where the layer answers
//! "no class", though the part of the cell inside the bounds has one. The layer
//! is constant across a cell, so any point of it inside the bounds gives that
//! class.
//!
//! A scene without the file gets no tint, and says so; nothing is inferred
//! from terrain colour or height.

use std::collections::BTreeMap;

use argusnet_world::land_cover::{LandCoverLayer, LandCoverTile};
use argusnet_world::obstacles::Bounds2d;
use bevy::math::Vec4;
use serde::Deserialize;

/// The document format the scene builder writes.
pub const LAND_COVER_FORMAT: &str = "argusnet.land-cover";
pub const LAND_COVER_VERSION: u32 = 1;

/// Largest land-cover raster side, in cells, and its memory cap (one byte a
/// cell).
pub const MAX_LAND_COVER_DIM: usize = 4096;
pub const MAX_LAND_COVER_BYTES: usize = 16 << 20;

/// How many class codes the palette carries.
pub const PALETTE_SIZE: usize = 16;

/// `metadata/land_cover.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct LandCoverDocument {
    pub format: String,
    pub version: u32,
    pub bounds: Bounds2d,
    pub tile_size_cells: usize,
    pub resolution: f64,
    pub tiles: Vec<LandCoverTile>,
    /// Class name to code.
    pub legend: BTreeMap<String, u8>,
    /// Class name to its style colour, RGBA.
    pub palette: BTreeMap<String, [f32; 4]>,
}

/// One class per cell, row by row (`y` rows, `x` contiguous).
#[derive(Debug, Clone)]
pub struct LandCoverRaster {
    pub x_min_m: f64,
    pub y_min_m: f64,
    pub resolution_m: f64,
    pub cols: usize,
    pub rows: usize,
    pub classes: Vec<u8>,
    /// Each code's colour; alpha 0 for a code with none, which draws no tint.
    pub palette: [Vec4; PALETTE_SIZE],
    /// The code the legend names `water`, if any: drawn as a smooth,
    /// reflective surface (R4).
    pub water_code: Option<u8>,
    layer: LandCoverLayer,
}

impl LandCoverRaster {
    pub fn from_document(document: LandCoverDocument) -> Result<Self, String> {
        if document.format != LAND_COVER_FORMAT || document.version != LAND_COVER_VERSION {
            return Err(format!(
                "land cover is {} v{}, not {LAND_COVER_FORMAT} v{LAND_COVER_VERSION}",
                document.format, document.version
            ));
        }
        let bounds = document.bounds;
        let resolution = document.resolution;
        if !(resolution.is_finite() && resolution > 0.0) {
            return Err("land-cover resolution must be positive".into());
        }
        let cells = |span: f64| (span / resolution).ceil().max(1.0);
        let (cols, rows) = (
            cells(bounds.x_max_m - bounds.x_min_m),
            cells(bounds.y_max_m - bounds.y_min_m),
        );
        if cols > MAX_LAND_COVER_DIM as f64
            || rows > MAX_LAND_COVER_DIM as f64
            || cols * rows > MAX_LAND_COVER_BYTES as f64
        {
            return Err(format!(
                "a {cols} x {rows} land-cover raster exceeds the {MAX_LAND_COVER_DIM}-cell, {} MiB cap",
                MAX_LAND_COVER_BYTES >> 20
            ));
        }
        let (cols, rows) = (cols as usize, rows as usize);
        let layer =
            LandCoverLayer::new(bounds, document.tile_size_cells, resolution, document.tiles)?;

        let mut classes = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                let x = (bounds.x_min_m + (c as f64 + 0.5) * resolution).min(bounds.x_max_m);
                let y = (bounds.y_min_m + (r as f64 + 0.5) * resolution).min(bounds.y_max_m);
                classes.push(layer.sample(x, y).0);
            }
        }

        let mut palette = [Vec4::ZERO; PALETTE_SIZE];
        for (name, code) in &document.legend {
            let Some(rgba) = document.palette.get(name) else {
                continue;
            };
            if let Some(slot) = palette.get_mut(*code as usize) {
                *slot = Vec4::from_array(*rgba);
            }
        }

        let water_code = document
            .legend
            .get("water")
            .copied()
            .filter(|code| (*code as usize) < PALETTE_SIZE);
        Ok(Self {
            water_code,
            x_min_m: bounds.x_min_m,
            y_min_m: bounds.y_min_m,
            resolution_m: resolution,
            cols,
            rows,
            classes,
            palette,
            layer,
        })
    }

    /// The class the shader reads at a world point: the containing cell's,
    /// or `None` off the raster.
    pub fn class_at(&self, x_m: f64, y_m: f64) -> Option<u8> {
        let c = ((x_m - self.x_min_m) / self.resolution_m).floor();
        let r = ((y_m - self.y_min_m) / self.resolution_m).floor();
        if c < 0.0 || r < 0.0 || c >= self.cols as f64 || r >= self.rows as f64 {
            return None;
        }
        Some(self.classes[r as usize * self.cols + c as usize])
    }

    /// The runtime layer the raster was sampled from.
    pub fn layer(&self) -> &LandCoverLayer {
        &self.layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(x_max: f64) -> LandCoverDocument {
        // A 2 x 2 arrangement of 4-cell tiles over a ragged extent: 72 m is
        // 7.2 cells, so the eighth column's centre (75 m) lies past the edge.
        let tile = |tx: i64, ty: i64, class: u8| LandCoverTile {
            tx,
            ty,
            x_min_m: tx as f64 * 40.0,
            y_min_m: ty as f64 * 40.0,
            cell_size_m: 10.0,
            classes: (0..4)
                .map(|r| {
                    (0..4)
                        .map(|c| if (r + c) % 3 == 0 { class } else { 0 })
                        .collect()
                })
                .collect(),
            density: None,
        };
        LandCoverDocument {
            format: LAND_COVER_FORMAT.into(),
            version: 1,
            bounds: Bounds2d::new(0.0, x_max, 0.0, 72.0),
            tile_size_cells: 4,
            resolution: 10.0,
            tiles: vec![tile(0, 0, 2), tile(1, 0, 3), tile(0, 1, 4), tile(1, 1, 8)],
            legend: [
                ("open", 0),
                ("forest", 2),
                ("water", 3),
                ("scrub", 4),
                ("road", 8),
            ]
            .into_iter()
            .map(|(n, c)| (n.to_string(), c))
            .collect(),
            palette: [
                ("forest", [0.1, 0.6, 0.2, 0.7]),
                ("water", [0.1, 0.3, 0.9, 0.7]),
                ("road", [0.4, 0.4, 0.4, 0.5]),
            ]
            .into_iter()
            .map(|(n, c)| (n.to_string(), c))
            .collect(),
        }
    }

    /// Wherever the raster answers, it answers the runtime layer's class, on
    /// a dense lattice of points including the ragged edge cells.
    #[test]
    fn the_raster_is_the_runtime_layers_classes() {
        let raster = LandCoverRaster::from_document(document(72.0)).unwrap();
        assert_eq!((raster.cols, raster.rows), (8, 8));
        let mut compared = 0;
        let mut classes = std::collections::BTreeSet::new();
        for xi in 0..=144 {
            for yi in 0..=144 {
                let (x, y) = (xi as f64 * 0.5, yi as f64 * 0.5);
                let expected = raster.layer().sample(x, y).0;
                assert_eq!(raster.class_at(x, y), Some(expected), "({x}, {y})");
                classes.insert(expected);
                compared += 1;
            }
        }
        assert!(compared > 20_000);
        assert!(classes.len() >= 5, "{classes:?}");
    }

    #[test]
    fn off_the_raster_is_no_class_not_the_edge() {
        let raster = LandCoverRaster::from_document(document(72.0)).unwrap();
        assert_eq!(raster.class_at(-0.1, 10.0), None);
        assert_eq!(raster.class_at(10.0, 80.1), None);
    }

    #[test]
    fn the_palette_follows_the_legend_and_unnamed_codes_draw_nothing() {
        let raster = LandCoverRaster::from_document(document(72.0)).unwrap();
        assert_eq!(raster.palette[2], Vec4::new(0.1, 0.6, 0.2, 0.7));
        assert_eq!(raster.water_code, Some(3), "water, from the legend");
        assert_eq!(raster.palette[3].z, 0.9);
        assert_eq!(
            raster.palette[0],
            Vec4::ZERO,
            "open has no colour in this palette"
        );
        assert_eq!(
            raster.palette[8],
            Vec4::new(0.4, 0.4, 0.4, 0.5),
            "placed by code, not by order"
        );
        assert_eq!(
            raster.palette[4],
            Vec4::ZERO,
            "scrub is in the legend, not the palette"
        );
    }

    #[test]
    fn documents_it_cannot_honour_are_refused_with_the_reason() {
        let mut wrong = document(75.0);
        wrong.format = "something-else".into();
        assert!(LandCoverRaster::from_document(wrong)
            .unwrap_err()
            .contains("argusnet.land-cover"));
        let mut newer = document(75.0);
        newer.version += 1;
        assert!(
            LandCoverRaster::from_document(newer).is_err(),
            "a newer version"
        );
        let huge = document(1.0e6);
        assert!(LandCoverRaster::from_document(huge)
            .unwrap_err()
            .contains("exceeds"));
    }
}
