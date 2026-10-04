//! Belief, as the terrain material draws it (P2.1).
//!
//! [`BeliefRenderSnapshot`] is the one place belief is sampled and packed for
//! the GPU. It takes `argusnet-mapping`'s [`BeliefTilePatch`]es, the
//! publication P4.6 defines, and nothing else: no terrain, no truth, no drone
//! positions. So the shader cannot infer coverage from terrain colour or from
//! distance to a drone, because it is never given either.
//!
//! # Channels
//!
//! Discrete and numeric channels live in separate textures, so neither can be
//! blended into the other:
//!
//! * **class**, `Rg8Uint`: `r` is the [`BeliefClass`] code, `g` the observed
//!   mask. Read with `textureLoad`, exact, never filtered: a class halfway
//!   between *safe* and *obstacle* is not a class.
//! * **numeric**, `Rg32Float`: `r` is confidence, `g` the last-observed time
//!   in seconds (`NaN` where unobserved). Also read with `textureLoad`.
//!   Confidence is interpolated in the shader, bilinearly between the four
//!   nearest cell centres **among observed cells only**, renormalised, so
//!   unobserved cells never bleed a zero into their neighbours. Time is never
//!   interpolated. [`BeliefRenderSnapshot::sample`] is the same arithmetic on
//!   the CPU, and the pixel probes hold the two together.
//!
//! # Class and confidence are the query's, not new rules
//!
//! Both follow `WorldBeliefQuery::cell_ij` exactly: each cell is judged at its
//! own centre, so the last cell of a ragged grid, whose centre lies past the
//! bounds, is *outside the geofence*; *unknown* is zero coverage; then the
//! obstacle-probability thresholds 0.6 and 0.4; confidence is
//! `min(1, visits / 5) * (1 - p)`. The snapshot carries no geofence predicate
//! beyond the bounds, as the patch carries none. A test compares every cell
//! against the query.
//!
//! # Tiles, seams and edges
//!
//! Tiles are sub-rectangles of one texture per channel, not separate
//! textures. There are no seams to halo: the interpolation reads across a tile
//! boundary exactly as it reads across any other. At the grid's edge,
//! sampling looks at the cells that exist; a point outside the grid is
//! [`BeliefClass::NoData`], not the nearest edge cell. `GridBounds.xy_to_ij`
//! clamps, which is right for queries and wrong for pixels.
//!
//! # Unknown and unavailable are drawn, never faked
//!
//! A replay with no belief data gets [`BeliefLayer::Unavailable`], with the
//! reason, and the material then draws an explicit unavailable state rather
//! than a plausible guess. Inside a snapshot, an unobserved cell is
//! [`BeliefClass::Unknown`], and nothing numeric is meaningful there.

use std::collections::BTreeSet;

use argusnet_mapping::{
    BeliefRevision, BeliefStore, BeliefTilePatch, GridBounds, Observation, OccupancyEvidence,
    Revision, Sampling, TileIndex,
};
use bevy::prelude::Resource;

/// Largest belief grid side the viewer will upload, in cells.
pub const MAX_BELIEF_DIM: usize = 2048;
/// Largest belief texture memory, both channels, in bytes (48 MiB).
pub const MAX_BELIEF_BYTES: usize = 48 << 20;
/// Bytes per cell: class `Rg8Uint` (2) and numeric `Rg32Float` (8).
pub const BYTES_PER_CELL: usize = 2 + 8;

/// Confidence at which coverage stops adding: `min(1, visits / 5)`.
const FULL_COVERAGE_VISITS: f64 = 5.0;

/// What a cell is, as the query classifies it. The discriminant is the code
/// the class texture carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BeliefClass {
    /// Outside the grid altogether. Never stored in a texture.
    NoData = 0,
    OutsideGeofence = 1,
    Unknown = 2,
    KnownObstacle = 3,
    Uncertain = 4,
    KnownSafe = 5,
}

impl BeliefClass {
    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::OutsideGeofence,
            2 => Self::Unknown,
            3 => Self::KnownObstacle,
            4 => Self::Uncertain,
            5 => Self::KnownSafe,
            _ => Self::NoData,
        }
    }
}

/// Why no belief can be drawn.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefUnavailable {
    /// The source carries no belief at all: every replay the repository ships
    /// today, since nothing serialises belief into one.
    NoBeliefData,
    /// The grid is larger than the viewer will upload.
    TooLarge { nx: usize, ny: usize },
    /// The grid's bounds are not a grid.
    InvalidBounds(&'static str),
}

impl std::fmt::Display for BeliefUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBeliefData => write!(formatter, "this source carries no belief data"),
            Self::TooLarge { nx, ny } => write!(
                formatter,
                "a {nx} x {ny} grid exceeds the {MAX_BELIEF_DIM}-cell, {} MiB belief texture cap",
                MAX_BELIEF_BYTES >> 20
            ),
            Self::InvalidBounds(why) => write!(formatter, "invalid belief grid: {why}"),
        }
    }
}

/// The belief the viewer has to draw, or why it has none.
#[derive(Debug, Resource)]
pub enum BeliefLayer {
    Unavailable(BeliefUnavailable),
    Ready(Box<BeliefRenderSnapshot>),
}

impl Default for BeliefLayer {
    fn default() -> Self {
        Self::Unavailable(BeliefUnavailable::NoBeliefData)
    }
}

/// One point's belief, as the shader samples it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeliefSample {
    pub class: BeliefClass,
    /// Interpolated among observed cells; `0` where none is observed.
    pub confidence: f32,
    /// The containing cell's; `None` where it is unobserved.
    pub last_observed_s: Option<f32>,
}

/// A rectangle of texels, `x` along cells' `i`, `y` along `j`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// One tile's texels, packed for upload.
#[derive(Debug, Clone, PartialEq)]
pub struct TileTexels {
    pub rect: TexelRect,
    /// `Rg8Uint`, row by row.
    pub class: Vec<u8>,
    /// `Rg32Float`, row by row, native endian.
    pub numeric: Vec<u8>,
}

/// Belief sampled and packed for the terrain material. See the module docs.
#[derive(Debug, Clone)]
pub struct BeliefRenderSnapshot {
    bounds: GridBounds,
    nx: usize,
    ny: usize,
    tile_size_cells: usize,
    tiles_j: usize,
    revision: Revision,
    tile_revisions: Vec<Revision>,
    /// Texel-major (`j * nx + i`): the class code.
    class: Vec<u8>,
    observed: Vec<u8>,
    confidence: Vec<f32>,
    last_observed_s: Vec<f32>,
    dirty: BTreeSet<TileIndex>,
    rejected_patches: u64,
    stale_patches: u64,
}

impl BeliefRenderSnapshot {
    /// An all-unknown snapshot for `bounds`, tiled as the store tiles it.
    pub fn new(bounds: GridBounds, tile_size_cells: usize) -> Result<Self, BeliefUnavailable> {
        let (nx, ny) = bounds
            .dimensions()
            .map_err(BeliefUnavailable::InvalidBounds)?;
        if tile_size_cells == 0 {
            return Err(BeliefUnavailable::InvalidBounds(
                "tile size must be positive",
            ));
        }
        if nx > MAX_BELIEF_DIM
            || ny > MAX_BELIEF_DIM
            || nx.saturating_mul(ny).saturating_mul(BYTES_PER_CELL) > MAX_BELIEF_BYTES
        {
            return Err(BeliefUnavailable::TooLarge { nx, ny });
        }
        let cells = nx * ny;
        let tiles_i = nx.div_ceil(tile_size_cells);
        let tiles_j = ny.div_ceil(tile_size_cells);
        let mut snapshot = Self {
            bounds,
            nx,
            ny,
            tile_size_cells,
            tiles_j,
            revision: Revision::INITIAL,
            tile_revisions: vec![Revision::INITIAL; tiles_i * tiles_j],
            class: vec![BeliefClass::Unknown as u8; cells],
            observed: vec![0; cells],
            confidence: vec![0.0; cells],
            last_observed_s: vec![f32::NAN; cells],
            dirty: BTreeSet::new(),
            rejected_patches: 0,
            stale_patches: 0,
        };
        // Cells whose centre lies past the bounds are outside even unobserved.
        for i in 0..nx {
            for j in 0..ny {
                if !snapshot.centre_in_bounds(i, j) {
                    snapshot.class[j * nx + i] = BeliefClass::OutsideGeofence as u8;
                }
            }
        }
        Ok(snapshot)
    }

    /// Everything a store has published, from scratch.
    pub fn from_store(store: &BeliefStore) -> Result<Self, BeliefUnavailable> {
        let mut snapshot = Self::new(store.bounds(), store.tile_size_cells())?;
        for patch in store.all_patches() {
            snapshot.apply_patch(&patch);
        }
        Ok(snapshot)
    }

    pub fn bounds(&self) -> GridBounds {
        self.bounds
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// The newest belief revision applied.
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// Patches refused for not fitting this grid.
    pub fn rejected_patches(&self) -> u64 {
        self.rejected_patches
    }

    /// Patches ignored for being older than what a tile already holds.
    pub fn stale_patches(&self) -> u64 {
        self.stale_patches
    }

    /// Tiles changed since the last [`Self::take_dirty`].
    pub fn dirty_len(&self) -> usize {
        self.dirty.len()
    }

    /// Apply every patch of a revision.
    pub fn apply_revision(&mut self, revision: &BeliefRevision) {
        for patch in &revision.patches {
            self.apply_patch(patch);
        }
    }

    /// Apply one tile patch. A patch that does not fit the grid is refused
    /// and counted; one older than the tile's current revision is ignored and
    /// counted.
    pub fn apply_patch(&mut self, patch: &BeliefTilePatch) {
        let (i0, j0) = patch.origin_ij;
        let (di, dj) = patch.dims;
        let fits = patch.tile.ti * self.tile_size_cells == i0
            && patch.tile.tj * self.tile_size_cells == j0
            && di <= self.tile_size_cells
            && dj <= self.tile_size_cells
            && i0 + di <= self.nx
            && j0 + dj <= self.ny
            && [
                patch.observed.len(),
                patch.coverage_count.len(),
                patch.obstacle_probability.len(),
                patch.last_observed_s.len(),
            ]
            .iter()
            .all(|&len| len == di * dj);
        if !fits {
            self.rejected_patches += 1;
            return;
        }
        let slot = patch.tile.ti * self.tiles_j + patch.tile.tj;
        if patch.revision < self.tile_revisions[slot] {
            self.stale_patches += 1;
            return;
        }
        self.tile_revisions[slot] = patch.revision;
        self.revision = self.revision.max(patch.revision);

        for li in 0..di {
            for lj in 0..dj {
                let local = li * dj + lj;
                let (i, j) = (i0 + li, j0 + lj);
                let texel = j * self.nx + i;
                let inside = self.centre_in_bounds(i, j);
                let visits = patch.coverage_count[local];
                let observed = patch.observed[local] && visits > 0;
                let p = patch.obstacle_probability[local];
                // `WorldBeliefQuery::cell_ij`'s order, at the cell's centre.
                let class = if !inside {
                    BeliefClass::OutsideGeofence
                } else if !observed {
                    BeliefClass::Unknown
                } else if p >= 0.6 {
                    BeliefClass::KnownObstacle
                } else if p >= 0.4 {
                    BeliefClass::Uncertain
                } else {
                    BeliefClass::KnownSafe
                };
                // `confidence_at`.
                let confidence = if !observed || !inside {
                    0.0
                } else {
                    ((f64::from(visits) / FULL_COVERAGE_VISITS).min(1.0) * (1.0 - p)).max(0.0)
                };
                self.class[texel] = class as u8;
                self.observed[texel] = u8::from(observed);
                self.confidence[texel] = confidence as f32;
                self.last_observed_s[texel] = if observed {
                    patch.last_observed_s[local] as f32
                } else {
                    f32::NAN
                };
            }
        }
        self.dirty.insert(patch.tile);
    }

    /// `GridBounds.ij_to_xy`'s centre, tested against the raw bounds.
    fn centre_in_bounds(&self, i: usize, j: usize) -> bool {
        let b = self.bounds;
        let x = b.x_min_m + (i as f64 + 0.5) * b.resolution_m;
        let y = b.y_min_m + (j as f64 + 0.5) * b.resolution_m;
        b.x_min_m <= x && x <= b.x_max_m && b.y_min_m <= y && y <= b.y_max_m
    }

    /// The cell's class, by index.
    pub fn class_ij(&self, i: usize, j: usize) -> BeliefClass {
        BeliefClass::from_code(self.class[j * self.nx + i])
    }

    /// The cell containing a world point, as `sample` finds it; `None`
    /// outside the grid.
    pub fn cell_at(&self, x_m: f64, y_m: f64) -> Option<(usize, usize)> {
        let b = self.bounds;
        let (ci, cj) = (
            ((x_m - b.x_min_m) / b.resolution_m).floor(),
            ((y_m - b.y_min_m) / b.resolution_m).floor(),
        );
        (ci >= 0.0 && cj >= 0.0 && ci < self.nx as f64 && cj < self.ny as f64)
            .then_some((ci as usize, cj as usize))
    }

    /// Whether the cell has ever been observed, by index.
    pub fn observed_ij(&self, i: usize, j: usize) -> bool {
        self.observed[j * self.nx + i] != 0
    }

    /// The cell's confidence, by index, uninterpolated.
    pub fn confidence_ij(&self, i: usize, j: usize) -> f32 {
        self.confidence[j * self.nx + i]
    }

    /// Sample a world point exactly as the shader does.
    pub fn sample(&self, x_m: f64, y_m: f64) -> BeliefSample {
        let b = self.bounds;
        // Cell-space coordinates, cell centres at half-integers. `floor`, not
        // truncation: a point below the origin is outside, not in cell 0.
        let u = (x_m - b.x_min_m) / b.resolution_m;
        let v = (y_m - b.y_min_m) / b.resolution_m;
        let (ci, cj) = (u.floor(), v.floor());
        if ci < 0.0 || cj < 0.0 || ci >= self.nx as f64 || cj >= self.ny as f64 {
            return BeliefSample {
                class: BeliefClass::NoData,
                confidence: 0.0,
                last_observed_s: None,
            };
        }
        let texel = cj as usize * self.nx + ci as usize;
        let last = self.last_observed_s[texel];

        // Bilinear among observed neighbours, renormalised.
        let (fu, fv) = (u - 0.5, v - 0.5);
        let (i0, j0) = (fu.floor(), fv.floor());
        let (tu, tv) = ((fu - i0) as f32, (fv - j0) as f32);
        let mut weighted = 0.0_f32;
        let mut weight = 0.0_f32;
        for (di, dj, w) in [
            (0.0, 0.0, (1.0 - tu) * (1.0 - tv)),
            (1.0, 0.0, tu * (1.0 - tv)),
            (0.0, 1.0, (1.0 - tu) * tv),
            (1.0, 1.0, tu * tv),
        ] {
            let (i, j) = (i0 + di, j0 + dj);
            if i < 0.0 || j < 0.0 || i >= self.nx as f64 || j >= self.ny as f64 {
                continue;
            }
            let neighbour = j as usize * self.nx + i as usize;
            if self.observed[neighbour] == 0 {
                continue;
            }
            weighted += w * self.confidence[neighbour];
            weight += w;
        }
        BeliefSample {
            class: BeliefClass::from_code(self.class[texel]),
            confidence: if weight > 0.0 { weighted / weight } else { 0.0 },
            last_observed_s: last.is_finite().then_some(last),
        }
    }

    /// The whole grid, packed: `(class Rg8Uint, numeric Rg32Float)`.
    pub fn texels(&self) -> TileTexels {
        self.pack(TexelRect {
            x: 0,
            y: 0,
            width: self.nx as u32,
            height: self.ny as u32,
        })
    }

    /// Each changed tile's texels, clearing the changed set.
    pub fn take_dirty(&mut self) -> Vec<TileTexels> {
        let tiles: Vec<TileIndex> = std::mem::take(&mut self.dirty).into_iter().collect();
        tiles
            .into_iter()
            .map(|tile| {
                let i0 = tile.ti * self.tile_size_cells;
                let j0 = tile.tj * self.tile_size_cells;
                self.pack(TexelRect {
                    x: i0 as u32,
                    y: j0 as u32,
                    width: ((i0 + self.tile_size_cells).min(self.nx) - i0) as u32,
                    height: ((j0 + self.tile_size_cells).min(self.ny) - j0) as u32,
                })
            })
            .collect()
    }

    fn pack(&self, rect: TexelRect) -> TileTexels {
        let cells = rect.width as usize * rect.height as usize;
        let mut class = Vec::with_capacity(cells * 2);
        let mut numeric = Vec::with_capacity(cells * 8);
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let texel = y as usize * self.nx + x as usize;
                class.push(self.class[texel]);
                class.push(self.observed[texel]);
                numeric.extend_from_slice(&self.confidence[texel].to_ne_bytes());
                numeric.extend_from_slice(&self.last_observed_s[texel].to_ne_bytes());
            }
        }
        TileTexels {
            rect,
            class,
            numeric,
        }
    }
}

/// A small, deterministic belief store for fixtures and probes: a 37 x 29
/// grid of 10 m cells over a ragged extent (the last column and row are
/// outside by the centre rule), tiled 8, with safe, uncertain and obstacle
/// cells, unknown gaps and a range of ages.
pub fn synthetic_belief_store() -> BeliefStore {
    let bounds = GridBounds {
        x_min_m: -180.0,
        // 364 m over 10 m cells is 37 cells; the 37th's centre, 185, lies
        // past 184, and likewise the 29th row's.
        x_max_m: 184.0,
        y_min_m: -140.0,
        y_max_m: 144.0,
        resolution_m: 10.0,
    };
    let mut store = BeliefStore::with_tile_size(bounds, 8).expect("fixture grid");
    // Four sweeps, at increasing times, over overlapping discs. Each reports
    // its cells free, twice: with no occupancy evidence a cell sits at the 0.5
    // prior, which the query calls *uncertain*; two free marks take it to
    // p = 0.31, *known safe*.
    for (k, (x, y, r)) in [
        (-100.0, -60.0, 45.0),
        (-60.0, -40.0, 35.0),
        (40.0, 30.0, 60.0),
        (120.0, 90.0, 30.0),
    ]
    .into_iter()
    .enumerate()
    {
        let observation = Observation::from_footprint(
            &bounds,
            format!("drone-{k}"),
            10.0 * (k as f64 + 1.0),
            [x, y],
            r,
            Some(0.0),
            Sampling::CellDisc,
        )
        .expect("fixture footprint");
        let mut observation = observation;
        for cell in &mut observation.cells {
            cell.occupancy = Some(OccupancyEvidence::Free);
        }
        store.apply(observation.clone());
        store.apply(observation);
    }
    store.commit();
    // Occupancy evidence, inside the third sweep's free area: one cluster
    // driven to obstacle (-0.8 + 3 x 0.85, p = 0.85), one cell pulled back to
    // uncertain (-0.8 + 0.85, p = 0.51).
    let mut occupied = Vec::new();
    for (i, j) in [(22, 17), (23, 17), (22, 18), (23, 18)] {
        for _ in 0..3 {
            occupied.push(argusnet_mapping::CellObservation {
                ij: (i, j),
                height_m: None,
                occupancy: Some(OccupancyEvidence::Occupied { height_m: 12.0 }),
            });
        }
    }
    occupied.push(argusnet_mapping::CellObservation {
        ij: (20, 16),
        height_m: None,
        occupancy: Some(OccupancyEvidence::Occupied { height_m: 3.0 }),
    });
    store.apply(Observation {
        source_id: "lidar".into(),
        timestamp_s: 55.0,
        cells: occupied,
    });
    store.commit();
    store
}

/// What a pixel probe expects to see at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeExpectation {
    Class(BeliefClass),
    /// Known safe, but observed long before the render's simulation time.
    StaleSafe,
}

/// A point to probe and what should be drawn there.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeTarget {
    pub label: &'static str,
    pub xy: [f64; 2],
    pub expect: ProbeExpectation,
}

/// A belief store laid out in large uniform blocks over `extent`, for pixel
/// probes, and the block centres to probe.
///
/// The grid is 36 x 30 cells, with one extra ragged column whose centres lie
/// past the bounds (outside), and all 37 columns lie within `extent`, so each
/// is drawn over terrain. Bottom half, left to right:
/// known safe, known obstacle, uncertain. Top half: unknown (never
/// observed), then known safe observed at `t = -10000 s`, which is stale at
/// any render time.
pub fn probe_belief_layout(extent: GridBounds) -> (BeliefStore, Vec<ProbeTarget>) {
    let resolution = (extent.x_max_m - extent.x_min_m) / 38.0;
    // A little past 36 cells: a 37th column whose centre (36.5) is outside.
    let bounds = GridBounds {
        x_min_m: extent.x_min_m,
        x_max_m: extent.x_min_m + 36.4 * resolution,
        y_min_m: extent.y_min_m,
        y_max_m: extent.y_min_m + 30.0 * resolution,
        resolution_m: resolution,
    };
    let mut store = BeliefStore::with_tile_size(bounds, 8).expect("probe grid");
    let block = |i0: usize, i1: usize, j0: usize, j1: usize| {
        (i0..i1).flat_map(move |i| (j0..j1).map(move |j| (i, j)))
    };
    let free = |cells: &mut Vec<argusnet_mapping::CellObservation>, (i, j)| {
        cells.push(argusnet_mapping::CellObservation {
            ij: (i, j),
            height_m: None,
            occupancy: Some(OccupancyEvidence::Free),
        });
    };
    let occupied = |cells: &mut Vec<argusnet_mapping::CellObservation>, (i, j)| {
        cells.push(argusnet_mapping::CellObservation {
            ij: (i, j),
            height_m: None,
            occupancy: Some(OccupancyEvidence::Occupied { height_m: 10.0 }),
        });
    };
    let mut recent = Vec::new();
    // Two free marks: p = 0.31, safe. Plus three occupied: 0.85, obstacle.
    // Plus one: 0.51, uncertain.
    for ij in block(0, 12, 0, 15) {
        free(&mut recent, ij);
        free(&mut recent, ij);
    }
    for ij in block(12, 24, 0, 15) {
        free(&mut recent, ij);
        free(&mut recent, ij);
        for _ in 0..3 {
            occupied(&mut recent, ij);
        }
    }
    for ij in block(24, 37, 0, 15) {
        free(&mut recent, ij);
        free(&mut recent, ij);
        occupied(&mut recent, ij);
    }
    store.apply(Observation {
        source_id: "probe-recent".into(),
        timestamp_s: 0.0,
        cells: recent,
    });
    let mut old = Vec::new();
    for ij in block(18, 37, 15, 30) {
        free(&mut old, ij);
        free(&mut old, ij);
    }
    store.apply(Observation {
        source_id: "probe-old".into(),
        timestamp_s: -10_000.0,
        cells: old,
    });
    store.commit();

    let centre = |i: f64, j: f64| {
        [
            bounds.x_min_m + i * resolution,
            bounds.y_min_m + j * resolution,
        ]
    };
    let targets = vec![
        ProbeTarget {
            label: "safe",
            xy: centre(6.0, 7.5),
            expect: ProbeExpectation::Class(BeliefClass::KnownSafe),
        },
        ProbeTarget {
            label: "obstacle",
            xy: centre(18.0, 7.5),
            expect: ProbeExpectation::Class(BeliefClass::KnownObstacle),
        },
        ProbeTarget {
            label: "uncertain",
            xy: centre(30.0, 7.5),
            expect: ProbeExpectation::Class(BeliefClass::Uncertain),
        },
        ProbeTarget {
            label: "unknown",
            xy: centre(8.0, 22.5),
            expect: ProbeExpectation::Class(BeliefClass::Unknown),
        },
        ProbeTarget {
            label: "stale",
            xy: centre(27.0, 22.5),
            expect: ProbeExpectation::StaleSafe,
        },
        ProbeTarget {
            label: "outside",
            xy: centre(36.5, 7.5),
            expect: ProbeExpectation::Class(BeliefClass::OutsideGeofence),
        },
    ];
    (store, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_mapping::{BeliefCellStatus, BeliefGrids, WorldBeliefQuery};

    fn class_of(status: BeliefCellStatus) -> BeliefClass {
        match status {
            BeliefCellStatus::OutsideGeofence => BeliefClass::OutsideGeofence,
            BeliefCellStatus::Unknown => BeliefClass::Unknown,
            BeliefCellStatus::KnownObstacle => BeliefClass::KnownObstacle,
            BeliefCellStatus::Uncertain => BeliefClass::Uncertain,
            BeliefCellStatus::KnownSafe => BeliefClass::KnownSafe,
        }
    }

    /// Every cell's class and confidence are the query's, the ragged edge
    /// included, and the fixture reaches every class.
    #[test]
    fn every_cell_agrees_with_the_belief_query() {
        let store = synthetic_belief_store();
        let snapshot = BeliefRenderSnapshot::from_store(&store).unwrap();
        let heights = store.mean_height_grid();
        let occupancy = store.obstacle_probability_grid();
        let query = WorldBeliefQuery::new(
            store.bounds(),
            BeliefGrids {
                coverage: store.coverage_counts(),
                mean_height: Some(&heights),
                occupancy: Some(&occupancy),
                last_observed_s: Some(store.last_observed_grid()),
            },
        )
        .unwrap();
        let (nx, ny) = snapshot.dimensions();
        let mut seen = BTreeSet::new();
        for i in 0..nx {
            for j in 0..ny {
                let cell = query.cell_ij(i, j);
                assert_eq!(
                    snapshot.class_ij(i, j),
                    class_of(cell.status),
                    "cell {i},{j}"
                );
                assert_eq!(
                    snapshot.confidence_ij(i, j),
                    cell.terrain_confidence as f32,
                    "confidence {i},{j}"
                );
                seen.insert(snapshot.class_ij(i, j));
            }
        }
        assert_eq!(
            seen.len(),
            5,
            "the fixture reaches every stored class: {seen:?}"
        );
        assert_eq!(
            snapshot.class_ij(nx - 1, 3),
            BeliefClass::OutsideGeofence,
            "ragged edge"
        );
    }

    /// Unobserved cells carry nothing numeric; observed ones their time.
    #[test]
    fn unobserved_cells_carry_no_time_and_no_confidence() {
        let snapshot = BeliefRenderSnapshot::from_store(&synthetic_belief_store()).unwrap();
        let far = snapshot.sample(-175.0, 135.0);
        assert_eq!(far.class, BeliefClass::Unknown);
        assert_eq!(far.confidence, 0.0);
        assert_eq!(far.last_observed_s, None);
        let seen = snapshot.sample(-100.0, -60.0);
        assert_eq!(seen.class, BeliefClass::KnownSafe);
        assert_eq!(seen.last_observed_s, Some(10.0));
    }

    /// Outside the grid is no data, not the nearest edge cell.
    #[test]
    fn points_off_the_grid_are_no_data() {
        let snapshot = BeliefRenderSnapshot::from_store(&synthetic_belief_store()).unwrap();
        for (x, y) in [(-180.1, 0.0), (0.0, -140.5), (190.1, 0.0), (0.0, 151.0)] {
            assert_eq!(
                snapshot.sample(x, y).class,
                BeliefClass::NoData,
                "({x}, {y})"
            );
        }
        // Exactly the low corner is cell 0.
        assert_ne!(snapshot.sample(-180.0, -140.0).class, BeliefClass::NoData);
    }

    /// At a cell centre the interpolation is the cell's own value; between an
    /// observed and an unobserved cell it is the observed one's, not a blend
    /// toward zero.
    #[test]
    fn confidence_interpolates_among_observed_cells_only() {
        let snapshot = BeliefRenderSnapshot::from_store(&synthetic_belief_store()).unwrap();
        let (nx, ny) = snapshot.dimensions();
        let b = snapshot.bounds();
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                let x = b.x_min_m + (i as f64 + 0.5) * b.resolution_m;
                let y = b.y_min_m + (j as f64 + 0.5) * b.resolution_m;
                let at_centre = snapshot.sample(x, y).confidence;
                if snapshot.observed[j * nx + i] == 1 {
                    assert!((at_centre - snapshot.confidence_ij(i, j)).abs() < 1e-6);
                }
                // Halfway to the next cell along x.
                let a = snapshot.observed[j * nx + i] == 1;
                let c = snapshot.observed[j * nx + i + 1] == 1;
                let mid = snapshot.sample(x + 5.0, y).confidence;
                if a && !c {
                    assert!((mid - snapshot.confidence_ij(i, j)).abs() < 1e-6);
                }
                if a && c {
                    let expect =
                        0.5 * (snapshot.confidence_ij(i, j) + snapshot.confidence_ij(i + 1, j));
                    assert!((mid - expect).abs() < 1e-5);
                }
            }
        }
    }

    /// Tiles are sub-rectangles of one texture: packing each dirty tile and
    /// laying it down reproduces the whole-grid packing exactly.
    #[test]
    fn dirty_tiles_reassemble_the_whole_texture() {
        let mut snapshot = BeliefRenderSnapshot::from_store(&synthetic_belief_store()).unwrap();
        let whole = snapshot.texels();
        let (nx, _) = snapshot.dimensions();
        let tiles = snapshot.take_dirty();
        assert!(!tiles.is_empty());
        assert_eq!(snapshot.dirty_len(), 0);
        let mut class = vec![0u8; whole.class.len()];
        let mut numeric = vec![0u8; whole.numeric.len()];
        for tile in &tiles {
            let r = tile.rect;
            for row in 0..r.height as usize {
                for col in 0..r.width as usize {
                    let texel = (r.y as usize + row) * nx + r.x as usize + col;
                    let local = row * r.width as usize + col;
                    class[texel * 2..texel * 2 + 2]
                        .copy_from_slice(&tile.class[local * 2..local * 2 + 2]);
                    numeric[texel * 8..texel * 8 + 8]
                        .copy_from_slice(&tile.numeric[local * 8..local * 8 + 8]);
                }
            }
        }
        // Tiles never touched are all-unknown, which is what the whole packing
        // holds there; compare the touched ones.
        for tile in &tiles {
            let r = tile.rect;
            for row in 0..r.height as usize {
                let start = ((r.y as usize + row) * nx + r.x as usize) * 2;
                let end = start + r.width as usize * 2;
                assert_eq!(class[start..end], whole.class[start..end]);
                assert_eq!(
                    numeric[start * 4..end * 4],
                    whole.numeric[start * 4..end * 4]
                );
            }
        }
    }

    /// A later commit dirties only what it touched, and older patches never
    /// overwrite newer ones.
    #[test]
    fn revisions_dirty_only_their_tiles_and_stale_patches_are_ignored() {
        let mut store = synthetic_belief_store();
        let mut snapshot = BeliefRenderSnapshot::from_store(&store).unwrap();
        snapshot.take_dirty();
        let old_patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        store.apply(
            Observation::from_footprint(
                &store.bounds(),
                "drone-9",
                99.0,
                [-170.0, -130.0],
                5.0,
                None,
                Sampling::CellDisc,
            )
            .unwrap(),
        );
        let revision = store.commit();
        snapshot.apply_revision(&revision);
        assert_eq!(snapshot.dirty_len(), revision.patches.len());
        assert!(snapshot.dirty_len() >= 1 && snapshot.dirty_len() < 4);
        assert_eq!(snapshot.revision(), revision.revision);
        snapshot.apply_patch(&old_patch);
        assert_eq!(snapshot.stale_patches(), 1);
        assert_eq!(snapshot.sample(-170.0, -130.0).last_observed_s, Some(99.0));
    }

    #[test]
    fn a_patch_that_does_not_fit_is_refused_and_counted() {
        let store = synthetic_belief_store();
        let mut snapshot = BeliefRenderSnapshot::from_store(&store).unwrap();
        let mut patch = store.tile_patch(TileIndex { ti: 1, tj: 1 });
        patch.origin_ij = (0, 0);
        snapshot.apply_patch(&patch);
        let mut short = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        short.observed.pop();
        snapshot.apply_patch(&short);
        assert_eq!(snapshot.rejected_patches(), 2);
    }

    #[test]
    fn a_grid_past_the_texture_cap_is_unavailable_with_its_reason() {
        let bounds = GridBounds {
            x_min_m: 0.0,
            x_max_m: 30_000.0,
            y_min_m: 0.0,
            y_max_m: 100.0,
            resolution_m: 10.0,
        };
        let error = BeliefRenderSnapshot::new(bounds, 32).unwrap_err();
        assert_eq!(error, BeliefUnavailable::TooLarge { nx: 3000, ny: 10 });
        assert!(error.to_string().contains("exceeds"));
        assert!(matches!(
            BeliefLayer::default(),
            BeliefLayer::Unavailable(BeliefUnavailable::NoBeliefData)
        ));
    }

    /// Every probe target is, on the CPU, what the probe expects.
    #[test]
    fn the_probe_layout_is_what_the_probes_expect() {
        let extent = GridBounds {
            x_min_m: -2160.0,
            x_max_m: 2160.0,
            y_min_m: -1980.0,
            y_max_m: 2160.0,
            resolution_m: 1.0,
        };
        let (store, targets) = probe_belief_layout(extent);
        let snapshot = BeliefRenderSnapshot::from_store(&store).unwrap();
        for target in &targets {
            let sample = snapshot.sample(target.xy[0], target.xy[1]);
            match target.expect {
                ProbeExpectation::Class(class) => {
                    assert_eq!(sample.class, class, "{}", target.label)
                }
                ProbeExpectation::StaleSafe => {
                    assert_eq!(sample.class, BeliefClass::KnownSafe);
                    assert_eq!(sample.last_observed_s, Some(-10_000.0));
                }
            }
        }
    }
}
