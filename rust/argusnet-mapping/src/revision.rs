//! Immutable belief revisions and dirty tile patches.
//!
//! P4.6's second paragraph asks for the *publication* side of mapping:
//! "Publish immutable belief revisions and dirty tile patches for planners and
//! P2.1. Carry observed/unknown, uncertainty, timestamp and evidence
//! provenance. Define out-of-bounds/missing-tile behavior and deterministic
//! update order when multiple sensors touch a cell."
//!
//! Unlike every other P4.6 slice, this one has **no Python reference**. Nothing
//! in `argusnet.mapping` publishes anything: `WorldMap` and `OccupancyGrid`
//! expose live mutable arrays, `belief_cells` is `[]` in every replay the repo
//! ships, and `mapping_state`'s belief aggregates are all `null`. So this is
//! design, and the contract below is stated rather than transcribed.
//!
//! What *is* ported is the fusion arithmetic, cell for cell, from
//! [`WorldMap.add_scan_observation`] and [`OccupancyGrid`]. That matters,
//! because one of those two does not commute.
//!
//! # Which updates commute, measured rather than assumed
//!
//! P4.6 says to "test commutative updates where promised, deterministic ordered
//! updates where not". The reference splits cleanly, and not the way the code
//! reads:
//!
//! | quantity | fusion | commutes? |
//! |---|---|---|
//! | `coverage_count` | saturating `+= 1` | **yes**, exactly |
//! | `last_observed_s` | `max` | **yes**, exactly |
//! | `height_sum` / `height_count` | `+=` | yes in exact arithmetic; **4.1e-13 m** apart over 4,000 observations in f64 |
//! | `log_odds` | `clip(lo + delta, -4, 4)` | **no** |
//!
//! The occupancy update clamps **on every observation**, and a clamp that binds
//! discards information that a later observation in the other order would have
//! consumed. Six observations — five `mark_occupied` and one `mark_free` —
//! give an obstacle probability of **0.9734** with the free one last and
//! **0.9792** with it first. Same evidence, different answer, and the
//! difference is large enough to move a cell across a planner threshold if the
//! thresholds were tighter.
//!
//! That is the whole reason [`BeliefStore::apply`] **stages** rather than
//! applies. Observations accumulate through a tick and are ordered at
//! [`BeliefStore::commit`] by `(source_id, arrival index within that source)`.
//! A single sensor's own frames keep their arrival order, because they are
//! genuinely sequential; *between* sensors the order is lexicographic by
//! source id, so it does not depend on which drone's gRPC response landed
//! first. Applying at call time would have been deterministic too — and
//! deterministic on something the mapping layer does not control.
//!
//! # Unknown is a value, not an absence
//!
//! Every patch carries an explicit `observed` mask, and P2.1b's constraint —
//! *"a replay without belief data renders an explicit unknown state rather than
//! fabricated confidence"* — is why nothing else in the patch is meaningful
//! where that mask is `false`. Unobserved cells carry `NaN` heights and
//! timestamps rather than `0.0`, so a consumer that ignores the mask produces
//! visible garbage instead of a plausible lie about a flat, freshly-scanned
//! world at the datum.
//!
//! For the same reason [`BeliefStore::tile_patch`] answers for a tile that has
//! never been touched: it returns a patch that is entirely unknown, rather than
//! `None`. A missing tile and an unobserved tile are the same thing to a
//! renderer, and only one of them can be drawn.
//!
//! # Out of bounds is dropped and counted, never clamped
//!
//! `OccupancyGrid.mark_free` calls `bounds.xy_to_ij`, which clamps. Clamping an
//! observation from outside the grid piles that evidence onto an edge cell —
//! the cell then reports confident belief about ground nobody looked at. Here
//! an out-of-range index is discarded, counted on the store, and reported on
//! the revision that dropped it.
//!
//! # No truth access
//!
//! Structural: [`BeliefStore`] holds grids and a source table and nothing else.
//! It has no terrain, no obstacle layer and no entity list, and every path into
//! it is an [`Observation`]. There is no method that could consult truth, which
//! is a stronger guarantee than a test that one currently does not.
//!
//! [`WorldMap.add_scan_observation`]: https://docs.rs/ (src/argusnet/mapping/world_map.py)
//! [`OccupancyGrid`]: https://docs.rs/ (src/argusnet/mapping/occupancy.py)

use std::collections::BTreeMap;

use crate::footprint::{rasterize_circle, GridBounds, Sampling};

/// `OccupancyGrid.LOG_ODDS_FREE`.
pub const LOG_ODDS_FREE: f64 = -0.4;
/// `OccupancyGrid.LOG_ODDS_OCC`.
pub const LOG_ODDS_OCC: f64 = 0.85;
/// `OccupancyGrid.LOG_ODDS_MIN`.
pub const LOG_ODDS_MIN: f64 = -4.0;
/// `OccupancyGrid.LOG_ODDS_MAX`.
pub const LOG_ODDS_MAX: f64 = 4.0;

/// Tile edge in cells when the caller does not choose one.
pub const DEFAULT_TILE_SIZE_CELLS: usize = 32;

/// How many distinct sources a cell can carry provenance bits for.
///
/// Provenance is a bitmask over an interned source table, which bounds the
/// per-cell cost at 8 bytes regardless of how long a run is. A 65th distinct
/// source still has its *evidence* applied in full; only its provenance bit is
/// unavailable, and [`BeliefStore::source_overflow_count`] reports that rather
/// than letting it pass silently.
pub const MAX_TRACKED_SOURCES: usize = 64;

/// A publication counter. Monotonic, and never reused within a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Revision(pub u64);

impl Revision {
    /// The revision of a store that has published nothing.
    pub const INITIAL: Revision = Revision(0);

    fn next(self) -> Revision {
        Revision(self.0 + 1)
    }
}

/// What one sensor concluded about one cell's occupancy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OccupancyEvidence {
    /// The cell was traversed without a return: `mark_free`.
    Free,
    /// Something was detected in the cell: `mark_occupied`.
    Occupied {
        /// Obstacle height above terrain, metres. Fused with `max`.
        height_m: f64,
    },
}

impl OccupancyEvidence {
    fn log_odds_delta(&self) -> f64 {
        match self {
            Self::Free => LOG_ODDS_FREE,
            Self::Occupied { .. } => LOG_ODDS_OCC,
        }
    }
}

/// One sensor's report about one cell.
///
/// Coverage is incremented for every cell in an observation, whether or not it
/// carries a height or occupancy conclusion — that is what coverage means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellObservation {
    pub ij: (usize, usize),
    /// Terrain height at the cell, metres above datum. `None` leaves the
    /// height accumulators untouched.
    pub height_m: Option<f64>,
    /// `None` leaves occupancy untouched, which is *not* the same as
    /// [`OccupancyEvidence::Free`].
    pub occupancy: Option<OccupancyEvidence>,
}

/// One sensor's report about a set of cells, at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Orders non-commutative fusion between sources. Lexicographic.
    pub source_id: String,
    pub timestamp_s: f64,
    /// Cells with multiplicity: a cell listed twice is covered twice, matching
    /// the rasterizer's `np.add.at` contract.
    pub cells: Vec<CellObservation>,
}

impl Observation {
    /// Build an observation from a circular sensor footprint.
    ///
    /// This is the join P4.6 asks for — *"one footprint rasterizer whose
    /// indices/weights feed both coverage and belief updates"*. The reference
    /// rasterizes the same disc twice, once per map, with two different
    /// samplings; see `footprint.rs` for the measurement. Coming through one
    /// call here means coverage and height cover **the same cells**, which in
    /// the reference they do not whenever `radius / resolution` is not an
    /// integer.
    ///
    /// Choosing between the two samplings stays the caller's visible decision.
    pub fn from_footprint(
        bounds: &GridBounds,
        source_id: impl Into<String>,
        timestamp_s: f64,
        center_xy: [f64; 2],
        radius_m: f64,
        terrain_height_m: Option<f64>,
        sampling: Sampling,
    ) -> Result<Self, &'static str> {
        let hits = rasterize_circle(bounds, center_xy, radius_m, sampling)?;
        Ok(Self {
            source_id: source_id.into(),
            timestamp_s,
            cells: hits
                .into_iter()
                .map(|ij| CellObservation {
                    ij,
                    height_m: terrain_height_m,
                    occupancy: None,
                })
                .collect(),
        })
    }
}

/// Which tile, in tile coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileIndex {
    pub ti: usize,
    pub tj: usize,
}

/// An immutable snapshot of one tile's belief at one revision.
///
/// Every `Vec` is `dims.0 * dims.1` long, i-major (j contiguous), matching the
/// grid layout the rest of the crate uses. `observed` gates all of them.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefTilePatch {
    /// The store revision this tile was last modified at. `Revision::INITIAL`
    /// for a tile nothing has ever touched.
    pub revision: Revision,
    pub tile: TileIndex,
    /// Cell index of the tile's low corner in the full grid.
    pub origin_ij: (usize, usize),
    /// Cells along i and j. Smaller than the tile size at the high edge, where
    /// the grid does not divide evenly.
    pub dims: (usize, usize),
    /// `false` means nothing below is meaningful for that cell.
    pub observed: Vec<bool>,
    pub coverage_count: Vec<u32>,
    /// `NaN` where unobserved, matching `WorldMap.mean_height_grid`.
    pub mean_height_m: Vec<f64>,
    /// `resolution / sqrt(coverage_count)`, the geometric proxy
    /// `WorldBeliefQuery::height_uncertainty_at` reports. `NaN` where
    /// unobserved. It never looks at the heights — see `height_dispersion_m`.
    pub height_uncertainty_m: Vec<f64>,
    /// Sample standard deviation of the heights actually observed in the cell.
    /// `NaN` with fewer than two height observations. **New here**: the
    /// reference has no dispersion measure at all, and the field above is not
    /// one.
    pub height_dispersion_m: Vec<f64>,
    /// `1 / (1 + exp(-log_odds))`. `0.5` where no occupancy evidence has
    /// arrived, which is the unknown prior rather than a measured value.
    pub obstacle_probability: Vec<f64>,
    /// Greatest obstacle height reported in the cell, metres. `0.0` where none.
    pub max_obstacle_height_m: Vec<f64>,
    /// `NaN` where unobserved.
    pub last_observed_s: Vec<f64>,
    /// Bitmask over `source_names`. Bit `k` means `source_names[k]`
    /// contributed to that cell.
    pub source_mask: Vec<u64>,
    /// The store's interned source table, by bit index.
    pub source_names: Vec<String>,
}

impl BeliefTilePatch {
    /// Cells in the patch.
    pub fn len(&self) -> usize {
        self.dims.0 * self.dims.1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether any cell in the tile has been observed.
    ///
    /// A patch for a tile nothing has touched is *not* empty — it is a full
    /// tile of explicit unknown.
    pub fn any_observed(&self) -> bool {
        self.observed.iter().any(|&o| o)
    }

    /// Source names contributing to one cell, in table order.
    pub fn sources_at(&self, local_i: usize, local_j: usize) -> Vec<&str> {
        let mask = self.source_mask[local_i * self.dims.1 + local_j];
        self.source_names
            .iter()
            .enumerate()
            .filter(|(k, _)| mask & (1u64 << k) != 0)
            .map(|(_, name)| name.as_str())
            .collect()
    }
}

/// Everything one [`BeliefStore::commit`] produced.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevision {
    pub revision: Revision,
    /// Greatest observation timestamp applied in this commit. `NaN` when the
    /// commit applied nothing.
    pub timestamp_s: f64,
    /// One patch per tile this commit dirtied, in `(ti, tj)` order.
    pub patches: Vec<BeliefTilePatch>,
    /// Source ids applied, sorted — the order fusion actually used.
    pub applied_sources: Vec<String>,
    /// Cell observations discarded for naming a cell outside the grid.
    pub dropped_out_of_bounds: u64,
}

impl BeliefRevision {
    /// Whether this commit changed anything.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

#[derive(Debug, Clone)]
struct Staged {
    observation: Observation,
    /// Arrival index within this source, which keeps one sensor's own frames
    /// in order under a stable sort.
    seq: usize,
}

/// Accumulates observations and publishes immutable revisions.
///
/// Holds no truth: every path in is an [`Observation`].
#[derive(Debug, Clone)]
pub struct BeliefStore {
    bounds: GridBounds,
    nx: usize,
    ny: usize,
    tile_size_cells: usize,
    tiles_i: usize,
    tiles_j: usize,

    coverage_count: Vec<u32>,
    height_sum: Vec<f64>,
    height_count: Vec<u32>,
    /// Welford's sum of squared deviations, for `height_dispersion_m`.
    height_m2: Vec<f64>,
    log_odds: Vec<f64>,
    has_occupancy: Vec<bool>,
    max_obstacle_height_m: Vec<f64>,
    last_observed_s: Vec<f64>,
    source_mask: Vec<u64>,

    /// Per-tile revision of last modification.
    tile_revision: Vec<Revision>,

    source_bits: BTreeMap<String, usize>,
    source_names: Vec<String>,
    source_overflow_count: u64,

    staged: Vec<Staged>,
    staged_seq: BTreeMap<String, usize>,
    revision: Revision,
    dropped_out_of_bounds: u64,
}

impl BeliefStore {
    /// A store over `bounds`, tiled at [`DEFAULT_TILE_SIZE_CELLS`].
    pub fn new(bounds: GridBounds) -> Result<Self, &'static str> {
        Self::with_tile_size(bounds, DEFAULT_TILE_SIZE_CELLS)
    }

    pub fn with_tile_size(
        bounds: GridBounds,
        tile_size_cells: usize,
    ) -> Result<Self, &'static str> {
        if tile_size_cells == 0 {
            return Err("tile size must be at least one cell");
        }
        let (nx, ny) = bounds.dimensions()?;
        let cells = nx.checked_mul(ny).ok_or("grid too large")?;
        // Ceiling division: the high edge tile is partial when the grid does
        // not divide evenly, and it is a tile, not a remainder to drop.
        let tiles_i = nx.div_ceil(tile_size_cells);
        let tiles_j = ny.div_ceil(tile_size_cells);
        let tiles = tiles_i.checked_mul(tiles_j).ok_or("grid too large")?;
        Ok(Self {
            bounds,
            nx,
            ny,
            tile_size_cells,
            tiles_i,
            tiles_j,
            coverage_count: vec![0; cells],
            height_sum: vec![0.0; cells],
            height_count: vec![0; cells],
            height_m2: vec![0.0; cells],
            log_odds: vec![0.0; cells],
            has_occupancy: vec![false; cells],
            max_obstacle_height_m: vec![0.0; cells],
            last_observed_s: vec![f64::NAN; cells],
            source_mask: vec![0; cells],
            tile_revision: vec![Revision::INITIAL; tiles],
            source_bits: BTreeMap::new(),
            source_names: Vec::new(),
            source_overflow_count: 0,
            staged: Vec::new(),
            staged_seq: BTreeMap::new(),
            revision: Revision::INITIAL,
            dropped_out_of_bounds: 0,
        })
    }

    pub fn bounds(&self) -> GridBounds {
        self.bounds
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn tile_dimensions(&self) -> (usize, usize) {
        (self.tiles_i, self.tiles_j)
    }

    pub fn tile_size_cells(&self) -> usize {
        self.tile_size_cells
    }

    /// The last published revision. Staged observations do not move it.
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// Cell observations discarded for being outside the grid, since
    /// construction.
    pub fn dropped_out_of_bounds(&self) -> u64 {
        self.dropped_out_of_bounds
    }

    /// Distinct sources beyond [`MAX_TRACKED_SOURCES`] whose evidence was
    /// applied without a provenance bit.
    pub fn source_overflow_count(&self) -> u64 {
        self.source_overflow_count
    }

    /// Observations staged but not yet committed.
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Stage an observation. Nothing is fused until [`Self::commit`].
    pub fn apply(&mut self, observation: Observation) {
        let seq = self
            .staged_seq
            .entry(observation.source_id.clone())
            .or_insert(0);
        let this_seq = *seq;
        *seq += 1;
        self.staged.push(Staged {
            observation,
            seq: this_seq,
        });
    }

    /// Fuse everything staged and publish a revision.
    ///
    /// Sources are applied in lexicographic order; within a source, arrival
    /// order. Committing nothing returns an empty revision and does **not**
    /// bump the counter, so a consumer polling `patches_since` sees no churn on
    /// an idle tick.
    pub fn commit(&mut self) -> BeliefRevision {
        if self.staged.is_empty() {
            return BeliefRevision {
                revision: self.revision,
                timestamp_s: f64::NAN,
                patches: Vec::new(),
                applied_sources: Vec::new(),
                dropped_out_of_bounds: 0,
            };
        }

        let mut staged = std::mem::take(&mut self.staged);
        self.staged_seq.clear();
        // Stable by construction: the key is total, so ties cannot occur.
        staged.sort_by(|a, b| {
            a.observation
                .source_id
                .cmp(&b.observation.source_id)
                .then(a.seq.cmp(&b.seq))
        });

        let revision = self.revision.next();
        let mut dropped = 0u64;
        let mut latest = f64::NEG_INFINITY;
        let mut applied_sources: Vec<String> = Vec::new();
        let mut dirty: Vec<TileIndex> = Vec::new();

        for item in &staged {
            let obs = &item.observation;
            let bit = self.intern_source(&obs.source_id);
            if applied_sources.last() != Some(&obs.source_id) {
                applied_sources.push(obs.source_id.clone());
            }
            for cell in &obs.cells {
                let (i, j) = cell.ij;
                if i >= self.nx || j >= self.ny {
                    dropped += 1;
                    continue;
                }
                self.fuse_cell(i, j, cell, obs.timestamp_s, bit);
                if obs.timestamp_s > latest {
                    latest = obs.timestamp_s;
                }
                let tile = self.tile_of(i, j);
                let slot = &mut self.tile_revision[tile.ti * self.tiles_j + tile.tj];
                if *slot != revision {
                    *slot = revision;
                    dirty.push(tile);
                }
            }
        }

        self.dropped_out_of_bounds += dropped;

        if dirty.is_empty() {
            // Every staged cell was out of bounds. Nothing changed, so nothing
            // is published and the counter stays put -- a revision that dirties
            // no tile would make `patches_since` advance past nothing.
            return BeliefRevision {
                revision: self.revision,
                timestamp_s: f64::NAN,
                patches: Vec::new(),
                applied_sources,
                dropped_out_of_bounds: dropped,
            };
        }

        self.revision = revision;
        dirty.sort_unstable();
        let patches = dirty.into_iter().map(|t| self.tile_patch(t)).collect();

        BeliefRevision {
            revision,
            timestamp_s: latest,
            patches,
            applied_sources,
            dropped_out_of_bounds: dropped,
        }
    }

    /// Patches for every tile modified strictly after `since`.
    ///
    /// Generated from current state rather than replayed from a log, so a
    /// consumer that falls behind by many revisions still receives each tile
    /// once, at its latest value. Pass [`Revision::INITIAL`] for a full
    /// resync of everything ever touched.
    pub fn patches_since(&self, since: Revision) -> Vec<BeliefTilePatch> {
        let mut out = Vec::new();
        for ti in 0..self.tiles_i {
            for tj in 0..self.tiles_j {
                if self.tile_revision[ti * self.tiles_j + tj] > since {
                    out.push(self.tile_patch(TileIndex { ti, tj }));
                }
            }
        }
        out
    }

    /// Every tile, touched or not — a cold consumer's first frame.
    pub fn all_patches(&self) -> Vec<BeliefTilePatch> {
        (0..self.tiles_i)
            .flat_map(|ti| (0..self.tiles_j).map(move |tj| TileIndex { ti, tj }))
            .map(|t| self.tile_patch(t))
            .collect()
    }

    /// Which tile a cell belongs to.
    ///
    /// # Panics
    ///
    /// If the cell is outside the grid.
    pub fn tile_of(&self, i: usize, j: usize) -> TileIndex {
        assert!(i < self.nx && j < self.ny, "cell outside grid");
        TileIndex {
            ti: i / self.tile_size_cells,
            tj: j / self.tile_size_cells,
        }
    }

    /// An immutable snapshot of one tile.
    ///
    /// A tile nothing has observed yields a full patch of explicit unknown,
    /// never `None`: a renderer has to draw that tile either way, and the only
    /// honest thing to draw is "no data".
    ///
    /// # Panics
    ///
    /// If the tile index is outside the tiling.
    pub fn tile_patch(&self, tile: TileIndex) -> BeliefTilePatch {
        assert!(
            tile.ti < self.tiles_i && tile.tj < self.tiles_j,
            "tile outside grid"
        );
        let i0 = tile.ti * self.tile_size_cells;
        let j0 = tile.tj * self.tile_size_cells;
        let di = (i0 + self.tile_size_cells).min(self.nx) - i0;
        let dj = (j0 + self.tile_size_cells).min(self.ny) - j0;
        let n = di * dj;

        let mut patch = BeliefTilePatch {
            revision: self.tile_revision[tile.ti * self.tiles_j + tile.tj],
            tile,
            origin_ij: (i0, j0),
            dims: (di, dj),
            observed: Vec::with_capacity(n),
            coverage_count: Vec::with_capacity(n),
            mean_height_m: Vec::with_capacity(n),
            height_uncertainty_m: Vec::with_capacity(n),
            height_dispersion_m: Vec::with_capacity(n),
            obstacle_probability: Vec::with_capacity(n),
            max_obstacle_height_m: Vec::with_capacity(n),
            last_observed_s: Vec::with_capacity(n),
            source_mask: Vec::with_capacity(n),
            source_names: self.source_names.clone(),
        };

        for li in 0..di {
            for lj in 0..dj {
                let idx = (i0 + li) * self.ny + (j0 + lj);
                let visits = self.coverage_count[idx];
                let observed = visits > 0;
                patch.observed.push(observed);
                patch.coverage_count.push(visits);
                patch.mean_height_m.push(if self.height_count[idx] > 0 {
                    self.height_sum[idx] / f64::from(self.height_count[idx])
                } else {
                    f64::NAN
                });
                patch.height_uncertainty_m.push(if observed {
                    self.bounds.resolution_m / f64::from(visits).sqrt()
                } else {
                    f64::NAN
                });
                patch
                    .height_dispersion_m
                    .push(if self.height_count[idx] > 1 {
                        (self.height_m2[idx] / f64::from(self.height_count[idx] - 1)).sqrt()
                    } else {
                        f64::NAN
                    });
                patch.obstacle_probability.push(if self.has_occupancy[idx] {
                    1.0 / (1.0 + (-self.log_odds[idx]).exp())
                } else {
                    0.5
                });
                patch
                    .max_obstacle_height_m
                    .push(self.max_obstacle_height_m[idx]);
                patch.last_observed_s.push(self.last_observed_s[idx]);
                patch.source_mask.push(self.source_mask[idx]);
            }
        }
        patch
    }

    /// Mean observed height per cell, `NaN` where unobserved — the array
    /// `WorldMap.mean_height_grid` returns, ready for [`crate::BeliefGrids`].
    pub fn mean_height_grid(&self) -> Vec<f64> {
        (0..self.height_count.len())
            .map(|k| {
                if self.height_count[k] > 0 {
                    self.height_sum[k] / f64::from(self.height_count[k])
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Obstacle probability per cell; `0.5` where no occupancy evidence has
    /// arrived.
    pub fn obstacle_probability_grid(&self) -> Vec<f64> {
        (0..self.log_odds.len())
            .map(|k| {
                if self.has_occupancy[k] {
                    1.0 / (1.0 + (-self.log_odds[k]).exp())
                } else {
                    0.5
                }
            })
            .collect()
    }

    pub fn coverage_counts(&self) -> &[u32] {
        &self.coverage_count
    }

    pub fn last_observed_grid(&self) -> &[f64] {
        &self.last_observed_s
    }

    /// The interned source table, by provenance bit index.
    pub fn source_names(&self) -> &[String] {
        &self.source_names
    }

    fn intern_source(&mut self, source_id: &str) -> Option<usize> {
        if let Some(&bit) = self.source_bits.get(source_id) {
            return Some(bit);
        }
        if self.source_names.len() >= MAX_TRACKED_SOURCES {
            self.source_overflow_count += 1;
            return None;
        }
        let bit = self.source_names.len();
        self.source_names.push(source_id.to_string());
        self.source_bits.insert(source_id.to_string(), bit);
        Some(bit)
    }

    fn fuse_cell(
        &mut self,
        i: usize,
        j: usize,
        cell: &CellObservation,
        timestamp_s: f64,
        source_bit: Option<usize>,
    ) {
        let idx = i * self.ny + j;

        // Saturating, as CoverageGrid is: wrapping a heavily covered cell to
        // zero would turn it back into an unobserved one.
        self.coverage_count[idx] = self.coverage_count[idx].saturating_add(1);

        // `max`, so it commutes exactly. A non-finite timestamp never wins,
        // which keeps one bad frame from pinning a cell's staleness for good.
        if timestamp_s.is_finite() {
            let previous = self.last_observed_s[idx];
            if previous.is_nan() || timestamp_s > previous {
                self.last_observed_s[idx] = timestamp_s;
            }
        }

        if let Some(bit) = source_bit {
            self.source_mask[idx] |= 1u64 << bit;
        }

        if let Some(height) = cell.height_m {
            if height.is_finite() {
                // Welford, but with the mean taken from the ported sum/count so
                // `mean_height_m` stays bit-identical to
                // `WorldMap.mean_height_grid`. A separate running mean would
                // drift from it in the last place and look like a port bug.
                let prev_count = self.height_count[idx];
                let prev_mean = if prev_count > 0 {
                    self.height_sum[idx] / f64::from(prev_count)
                } else {
                    0.0
                };
                self.height_sum[idx] += height;
                self.height_count[idx] += 1;
                let new_mean = self.height_sum[idx] / f64::from(self.height_count[idx]);
                if prev_count > 0 {
                    self.height_m2[idx] += (height - prev_mean) * (height - new_mean);
                }
            }
        }

        if let Some(evidence) = cell.occupancy {
            self.has_occupancy[idx] = true;
            // Clamped per observation, exactly as `OccupancyGrid` does -- which
            // is what makes this the one non-commutative update in the store.
            self.log_odds[idx] =
                (self.log_odds[idx] + evidence.log_odds_delta()).clamp(LOG_ODDS_MIN, LOG_ODDS_MAX);
            if let OccupancyEvidence::Occupied { height_m } = evidence {
                if height_m > self.max_obstacle_height_m[idx] {
                    self.max_obstacle_height_m[idx] = height_m;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(res: f64, span: f64) -> GridBounds {
        GridBounds {
            x_min_m: 0.0,
            x_max_m: span,
            y_min_m: 0.0,
            y_max_m: span,
            resolution_m: res,
        }
    }

    fn occupied_at(ij: (usize, usize), height_m: f64) -> CellObservation {
        CellObservation {
            ij,
            height_m: None,
            occupancy: Some(OccupancyEvidence::Occupied { height_m }),
        }
    }

    fn free_at(ij: (usize, usize)) -> CellObservation {
        CellObservation {
            ij,
            height_m: None,
            occupancy: Some(OccupancyEvidence::Free),
        }
    }

    fn obs(source: &str, t: f64, cells: Vec<CellObservation>) -> Observation {
        Observation {
            source_id: source.to_string(),
            timestamp_s: t,
            cells,
        }
    }

    fn probability_at(store: &BeliefStore, i: usize, j: usize) -> f64 {
        let (_, ny) = store.dimensions();
        store.obstacle_probability_grid()[i * ny + j]
    }

    /// Flat index into a grid of `ny` columns.
    fn flat(i: usize, ny: usize, j: usize) -> usize {
        i * ny + j
    }

    /// Flat index into a patch, from a *global* cell index.
    fn patch_slot(patch: &BeliefTilePatch, i: usize, j: usize) -> usize {
        flat(i - patch.origin_ij.0, patch.dims.1, j - patch.origin_ij.1)
    }

    /// Deterministic, dependency-free, and only ever used to build orderings.
    struct Lcg(u64);
    impl Lcg {
        fn next_usize(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as usize) % n
        }
        fn shuffle<T>(&mut self, items: &mut [T]) {
            for k in (1..items.len()).rev() {
                let swap = self.next_usize(k + 1);
                items.swap(k, swap);
            }
        }
    }

    // ---------------------------------------------------------------
    // Commutativity: which updates commute, and which demonstrably do not
    // ---------------------------------------------------------------

    /// The finding this module is built around. Same six observations, two
    /// orders, two answers -- because `OccupancyGrid` clamps on every update
    /// and a clamp that binds throws away evidence the other order still had
    /// room for.
    #[test]
    fn the_occupancy_log_odds_update_does_not_commute() {
        let build = |free_first: bool| {
            let mut store = BeliefStore::new(bounds(10.0, 100.0)).unwrap();
            let mut cells: Vec<CellObservation> =
                (0..5).map(|_| occupied_at((0, 0), 3.0)).collect();
            if free_first {
                cells.insert(0, free_at((0, 0)));
            } else {
                cells.push(free_at((0, 0)));
            }
            // One observation, so the store's own ordering plays no part: this
            // is the arithmetic, not the scheduling.
            store.apply(obs("sensor", 1.0, cells));
            store.commit();
            probability_at(&store, 0, 0)
        };
        let free_last = build(false);
        let free_first = build(true);
        assert!(
            (free_last - 0.973_403_006_423_134).abs() < 1e-12,
            "{free_last}"
        );
        assert!(
            (free_first - 0.979_163_655_481_319_6).abs() < 1e-12,
            "{free_first}"
        );
        assert_ne!(free_last, free_first);
    }

    /// Coverage and staleness are pure `+` and `max`, so they are immune to the
    /// ordering the clamp above is sensitive to.
    #[test]
    fn coverage_and_staleness_commute_exactly() {
        let mut forward = BeliefStore::new(bounds(10.0, 100.0)).unwrap();
        let mut backward = BeliefStore::new(bounds(10.0, 100.0)).unwrap();
        let batch = [
            obs("a", 3.0, vec![free_at((1, 1)), free_at((1, 2))]),
            obs("b", 9.0, vec![free_at((1, 1))]),
            obs("c", 5.0, vec![free_at((1, 2)), free_at((1, 1))]),
        ];
        for o in batch.iter() {
            forward.apply(o.clone());
        }
        for o in batch.iter().rev() {
            backward.apply(o.clone());
        }
        forward.commit();
        backward.commit();
        assert_eq!(forward.coverage_counts(), backward.coverage_counts());
        // Compared by bits: unobserved cells hold NaN, which is never equal to
        // itself, so a plain slice comparison would fail on identical grids.
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(forward.last_observed_grid()),
            bits(backward.last_observed_grid())
        );
        // and the values are the ones `+` and `max` predict
        let (_, ny) = forward.dimensions();
        assert_eq!(forward.coverage_counts()[flat(1, ny, 1)], 3);
        assert_eq!(forward.last_observed_grid()[flat(1, ny, 1)], 9.0);
    }

    /// The property the staging buys: the *store* fixes the order, so the
    /// non-commutative update stops depending on which sensor reported first.
    #[test]
    fn commit_order_does_not_depend_on_arrival_order() {
        let run = |alpha_first: bool| {
            let mut store = BeliefStore::new(bounds(10.0, 100.0)).unwrap();
            let alpha: Vec<Observation> = (0..5)
                .map(|k| obs("alpha", k as f64, vec![occupied_at((0, 0), 1.0)]))
                .collect();
            let beta = obs("beta", 9.0, vec![free_at((0, 0))]);
            if alpha_first {
                for o in alpha {
                    store.apply(o);
                }
                store.apply(beta);
            } else {
                store.apply(beta);
                for o in alpha {
                    store.apply(o);
                }
            }
            let revision = store.commit();
            (probability_at(&store, 0, 0), revision.applied_sources)
        };
        let (p_alpha_first, sources_a) = run(true);
        let (p_beta_first, sources_b) = run(false);
        assert_eq!(p_alpha_first, p_beta_first);
        // Lexicographic, both ways -- and "alpha" sorting first is what makes
        // the answer the free-last one from the test above.
        assert_eq!(sources_a, vec!["alpha".to_string(), "beta".to_string()]);
        assert_eq!(sources_b, sources_a);
        assert!((p_alpha_first - 0.973_403_006_423_134).abs() < 1e-12);
    }

    /// Sorting between sources must not sort *within* one. A sensor's frames
    /// are genuinely sequential and reordering them would be a lie about time.
    #[test]
    fn one_sources_own_frames_keep_their_arrival_order() {
        let run = |free_first: bool| {
            let mut store = BeliefStore::new(bounds(10.0, 100.0)).unwrap();
            let mut frames: Vec<Observation> = (0..5)
                .map(|k| obs("solo", k as f64, vec![occupied_at((0, 0), 1.0)]))
                .collect();
            let free = obs("solo", 9.0, vec![free_at((0, 0))]);
            if free_first {
                frames.insert(0, free);
            } else {
                frames.push(free);
            }
            for f in frames {
                store.apply(f);
            }
            store.commit();
            probability_at(&store, 0, 0)
        };
        // Named values, not just a difference: asserting only `!=` passes just
        // as happily when the frames are reversed, which is the bug.
        assert!(
            (run(false) - 0.973_403_006_423_134).abs() < 1e-12,
            "free frame arriving last must be applied last"
        );
        assert!(
            (run(true) - 0.979_163_655_481_319_6).abs() < 1e-12,
            "free frame arriving first must be applied first"
        );
    }

    /// The strong form: many cells, many sources, many shuffles of arrival
    /// order, one answer. A few hand-picked orders would not catch a sort key
    /// that happens to be right for them.
    #[test]
    fn a_shuffled_tick_always_commits_to_the_same_belief() {
        let mut rng = Lcg(0x5eed);
        let mut batch: Vec<Observation> = Vec::new();
        for s in 0..6usize {
            for k in 0..7usize {
                let ij = (rng.next_usize(9), rng.next_usize(9));
                let cells = vec![CellObservation {
                    ij,
                    height_m: Some(100.0 + (k as f64) * 0.25),
                    occupancy: Some(if (s + k) % 3 == 0 {
                        OccupancyEvidence::Free
                    } else {
                        OccupancyEvidence::Occupied {
                            height_m: (k as f64) * 1.5,
                        }
                    }),
                }];
                batch.push(obs(&format!("src-{s}"), (k as f64) * 0.1, cells));
            }
        }

        let commit_shuffled = |order: &[Observation]| {
            let mut store = BeliefStore::with_tile_size(bounds(10.0, 90.0), 4).unwrap();
            for o in order {
                store.apply(o.clone());
            }
            let revision = store.commit();
            (
                store.obstacle_probability_grid(),
                store.mean_height_grid(),
                store.coverage_counts().to_vec(),
                revision.patches.len(),
            )
        };

        let reference = commit_shuffled(&batch);
        for _ in 0..16 {
            let mut shuffled = batch.clone();
            rng.shuffle(&mut shuffled);
            let got = commit_shuffled(&shuffled);
            assert_eq!(got.0, reference.0, "obstacle probability moved");
            // NaN != NaN, so compare the mean-height grids by bits.
            let bits = |v: &Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got.1), bits(&reference.1), "mean height moved");
            assert_eq!(got.2, reference.2, "coverage moved");
            assert_eq!(got.3, reference.3, "dirty tile set moved");
        }
    }

    // ---------------------------------------------------------------
    // Out of bounds
    // ---------------------------------------------------------------

    #[test]
    fn an_out_of_bounds_cell_is_dropped_and_counted_never_clamped() {
        let mut store = BeliefStore::new(bounds(10.0, 50.0)).unwrap();
        let (nx, ny) = store.dimensions();
        assert_eq!((nx, ny), (5, 5));
        store.apply(obs(
            "rogue",
            1.0,
            vec![free_at((99, 2)), free_at((2, 99)), free_at((2, 2))],
        ));
        let revision = store.commit();
        assert_eq!(revision.dropped_out_of_bounds, 2);
        assert_eq!(store.dropped_out_of_bounds(), 2);
        // The failure mode being avoided: `xy_to_ij` clamps, which would have
        // piled both drops onto the far corner cell.
        assert_eq!(store.coverage_counts()[4 * ny + 2], 0);
        assert_eq!(store.coverage_counts()[2 * ny + 4], 0);
        assert_eq!(store.coverage_counts()[2 * ny + 2], 1);
    }

    #[test]
    fn a_commit_of_nothing_but_out_of_bounds_cells_publishes_nothing() {
        let mut store = BeliefStore::new(bounds(10.0, 50.0)).unwrap();
        store.apply(obs("rogue", 1.0, vec![free_at((99, 99))]));
        let revision = store.commit();
        assert_eq!(revision.revision, Revision::INITIAL);
        assert_eq!(store.revision(), Revision::INITIAL);
        assert!(revision.is_empty());
        assert_eq!(revision.dropped_out_of_bounds, 1);
        // The source was still seen, so the caller can tell the difference
        // between "nothing arrived" and "everything arrived off-grid".
        assert_eq!(revision.applied_sources, vec!["rogue".to_string()]);
    }

    #[test]
    fn committing_nothing_does_not_move_the_revision() {
        let mut store = BeliefStore::new(bounds(10.0, 50.0)).unwrap();
        let empty = store.commit();
        assert!(empty.is_empty());
        assert_eq!(store.revision(), Revision::INITIAL);
        store.apply(obs("a", 1.0, vec![free_at((0, 0))]));
        assert_eq!(store.staged_len(), 1);
        assert_eq!(
            store.revision(),
            Revision::INITIAL,
            "staging is not publishing"
        );
        store.commit();
        assert_eq!(store.revision(), Revision(1));
        assert!(store.commit().is_empty());
        assert_eq!(store.revision(), Revision(1));
    }

    // ---------------------------------------------------------------
    // Unknown, and the missing tile
    // ---------------------------------------------------------------

    #[test]
    fn an_untouched_tile_patches_as_explicit_unknown_not_as_absence() {
        let store = BeliefStore::with_tile_size(bounds(10.0, 80.0), 4).unwrap();
        let patch = store.tile_patch(TileIndex { ti: 1, tj: 1 });
        assert_eq!(patch.dims, (4, 4));
        assert_eq!(patch.len(), 16);
        assert!(
            !patch.is_empty(),
            "a tile of unknown is a tile, not nothing"
        );
        assert!(!patch.any_observed());
        assert_eq!(patch.revision, Revision::INITIAL);
        assert!(patch.observed.iter().all(|&o| !o));
        // P2.1b's constraint: a consumer that ignores the mask must produce
        // visible garbage, not a plausible flat freshly-scanned world.
        assert!(patch.mean_height_m.iter().all(|h| h.is_nan()));
        assert!(patch.last_observed_s.iter().all(|t| t.is_nan()));
        assert!(patch.height_uncertainty_m.iter().all(|u| u.is_nan()));
        assert!(patch.height_dispersion_m.iter().all(|u| u.is_nan()));
        // 0.5 is the unknown prior, and it is paired with observed == false so
        // it cannot be mistaken for a measured coin flip.
        assert!(patch.obstacle_probability.iter().all(|&p| p == 0.5));
        assert!(patch.source_mask.iter().all(|&m| m == 0));
    }

    #[test]
    fn an_empty_store_publishes_a_full_grid_of_unknown() {
        let store = BeliefStore::with_tile_size(bounds(10.0, 80.0), 4).unwrap();
        let patches = store.all_patches();
        assert_eq!(patches.len(), 4);
        assert!(patches.iter().all(|p| !p.any_observed()));
        // patches_since, by contrast, reports nothing changed.
        assert!(store.patches_since(Revision::INITIAL).is_empty());
    }

    // ---------------------------------------------------------------
    // Tiles and seams
    // ---------------------------------------------------------------

    /// A grid that does not divide evenly: 9 cells at tile size 4 is 2 full
    /// tiles and a 1-cell strip. The strip is a tile.
    #[test]
    fn tiles_partition_a_ragged_grid_exactly_once() {
        let store = BeliefStore::with_tile_size(bounds(10.0, 90.0), 4).unwrap();
        let (nx, ny) = store.dimensions();
        assert_eq!((nx, ny), (9, 9));
        assert_eq!(store.tile_dimensions(), (3, 3));

        let mut seen = vec![0usize; nx * ny];
        for patch in store.all_patches() {
            assert_eq!(patch.len(), patch.observed.len());
            for li in 0..patch.dims.0 {
                for lj in 0..patch.dims.1 {
                    let i = patch.origin_ij.0 + li;
                    let j = patch.origin_ij.1 + lj;
                    assert!(i < nx && j < ny, "tile ran past the grid");
                    seen[i * ny + j] += 1;
                    assert_eq!(store.tile_of(i, j), patch.tile);
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "cells covered {seen:?} times");
        // The high-edge tiles are the 1-cell strip, not dropped and not padded.
        let corner = store.tile_patch(TileIndex { ti: 2, tj: 2 });
        assert_eq!(corner.dims, (1, 1));
        assert_eq!(corner.origin_ij, (8, 8));
    }

    #[test]
    fn a_footprint_across_a_seam_dirties_both_tiles() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 90.0), 4).unwrap();
        // (3, 4) straddles the ti = 0 / tj = 1 seam together with (4, 4).
        store.apply(obs("a", 1.0, vec![free_at((3, 4)), free_at((4, 4))]));
        let revision = store.commit();
        let tiles: Vec<TileIndex> = revision.patches.iter().map(|p| p.tile).collect();
        assert_eq!(
            tiles,
            vec![TileIndex { ti: 0, tj: 1 }, TileIndex { ti: 1, tj: 1 }]
        );
        // Each patch carries only its own side of the seam.
        let left = &revision.patches[0];
        assert_eq!(left.origin_ij, (0, 4));
        assert!(left.observed[patch_slot(left, 3, 4)]);
        let right = &revision.patches[1];
        assert_eq!(right.origin_ij, (4, 4));
        assert!(right.observed[patch_slot(right, 4, 4)]);
    }

    #[test]
    fn patches_since_returns_each_changed_tile_once_at_its_latest_value() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 80.0), 4).unwrap();
        store.apply(obs("a", 1.0, vec![free_at((0, 0))]));
        let first = store.commit();
        assert_eq!(first.revision, Revision(1));

        store.apply(obs("a", 2.0, vec![free_at((0, 0))]));
        store.apply(obs("a", 2.0, vec![free_at((5, 5))]));
        let second = store.commit();
        assert_eq!(second.revision, Revision(2));
        assert_eq!(second.patches.len(), 2);

        // A consumer that fell behind at revision 0 gets both tiles once, each
        // at its current value -- not a replay of three separate edits.
        let catchup = store.patches_since(Revision::INITIAL);
        assert_eq!(catchup.len(), 2);
        assert_eq!(catchup[0].coverage_count[0], 2);

        // A consumer already at revision 1 gets only what moved since.
        let incremental = store.patches_since(Revision(1));
        assert_eq!(incremental.len(), 2, "tile (0,0) changed again at rev 2");
        assert!(store.patches_since(Revision(2)).is_empty());
    }

    // ---------------------------------------------------------------
    // Height: the ported mean, and the uncertainty that is not a dispersion
    // ---------------------------------------------------------------

    #[test]
    fn mean_height_is_the_references_sum_over_count() {
        let mut store = BeliefStore::new(bounds(10.0, 50.0)).unwrap();
        let heights = [101.5, 99.25, 100.75, 98.5];
        for (k, h) in heights.iter().enumerate() {
            store.apply(obs(
                "a",
                k as f64,
                vec![CellObservation {
                    ij: (1, 1),
                    height_m: Some(*h),
                    occupancy: None,
                }],
            ));
        }
        store.commit();
        let expected = heights.iter().sum::<f64>() / heights.len() as f64;
        let (_, ny) = store.dimensions();
        assert_eq!(store.mean_height_grid()[flat(1, ny, 1)], expected);
    }

    /// The two uncertainty fields mean different things and must not be
    /// conflated. `height_uncertainty_m` is `resolution / sqrt(visits)` -- the
    /// reference's geometric proxy, which never reads a height at all --
    /// while `height_dispersion_m` is the spread of the heights observed.
    #[test]
    fn height_uncertainty_is_geometric_and_dispersion_is_measured() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        let heights = [100.0, 102.0, 104.0, 106.0];
        for (k, h) in heights.iter().enumerate() {
            store.apply(obs(
                "a",
                k as f64,
                vec![CellObservation {
                    ij: (0, 0),
                    height_m: Some(*h),
                    occupancy: None,
                }],
            ));
        }
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        // 10 m resolution over 4 visits.
        assert!((patch.height_uncertainty_m[0] - 5.0).abs() < 1e-12);
        // Sample standard deviation of 100, 102, 104, 106 is sqrt(20/3).
        let expected = (20.0f64 / 3.0).sqrt();
        assert!(
            (patch.height_dispersion_m[0] - expected).abs() < 1e-12,
            "{}",
            patch.height_dispersion_m[0]
        );
        assert_ne!(patch.height_uncertainty_m[0], patch.height_dispersion_m[0]);
    }

    #[test]
    fn a_single_height_observation_has_no_dispersion() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs(
            "a",
            0.0,
            vec![CellObservation {
                ij: (0, 0),
                height_m: Some(100.0),
                occupancy: None,
            }],
        ));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert_eq!(patch.mean_height_m[0], 100.0);
        assert!(patch.height_dispersion_m[0].is_nan(), "n = 1 has no spread");
        assert!(patch.height_uncertainty_m[0].is_finite());
    }

    /// Coverage counts a cell even when the sensor concluded nothing about it,
    /// so a cell can be observed with no height. `mean_height_m` is NaN there
    /// while `observed` is true -- the two masks are not the same mask.
    #[test]
    fn an_observed_cell_may_still_have_no_height() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs("a", 1.0, vec![free_at((0, 0))]));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert!(patch.observed[0]);
        assert_eq!(patch.coverage_count[0], 1);
        assert!(patch.mean_height_m[0].is_nan());
    }

    #[test]
    fn a_non_finite_height_or_timestamp_is_ignored_rather_than_stored() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs(
            "a",
            f64::NAN,
            vec![CellObservation {
                ij: (0, 0),
                height_m: Some(f64::INFINITY),
                occupancy: None,
            }],
        ));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        // The cell was still looked at, which is what coverage records.
        assert!(patch.observed[0]);
        assert!(patch.mean_height_m[0].is_nan());
        assert!(patch.last_observed_s[0].is_nan());
    }

    // ---------------------------------------------------------------
    // Provenance
    // ---------------------------------------------------------------

    #[test]
    fn provenance_names_every_source_that_touched_a_cell() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs("drone-b", 1.0, vec![free_at((0, 0)), free_at((1, 1))]));
        store.apply(obs("drone-a", 1.0, vec![free_at((0, 0))]));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert_eq!(patch.sources_at(0, 0), vec!["drone-a", "drone-b"]);
        assert_eq!(patch.sources_at(1, 1), vec!["drone-b"]);
        assert!(patch.sources_at(2, 2).is_empty());
        // Bit indices follow first-application order, which is the sorted
        // order, so the table is stable across runs.
        assert_eq!(patch.source_names, vec!["drone-a", "drone-b"]);
    }

    #[test]
    fn a_source_past_the_provenance_limit_still_fuses_and_is_counted() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        for k in 0..=MAX_TRACKED_SOURCES {
            store.apply(obs(&format!("src-{k:03}"), 1.0, vec![free_at((0, 0))]));
        }
        store.commit();
        assert_eq!(store.source_names().len(), MAX_TRACKED_SOURCES);
        assert_eq!(store.source_overflow_count(), 1);
        // Evidence is not dropped -- only the provenance bit is unavailable.
        assert_eq!(
            store.coverage_counts()[0],
            MAX_TRACKED_SOURCES as u32 + 1,
            "the 65th source's observation must still count"
        );
    }

    // ---------------------------------------------------------------
    // The footprint join, and no truth access
    // ---------------------------------------------------------------

    /// P4.6's "one footprint rasterizer whose indices/weights feed both
    /// coverage and belief updates". The reference calls two rasterizers on the
    /// same disc and they disagree; coming through one call, coverage and
    /// height necessarily cover the same cells.
    #[test]
    fn a_footprint_gives_coverage_and_height_the_same_cells() {
        let b = bounds(10.0, 200.0);
        let mut store = BeliefStore::with_tile_size(b, 8).unwrap();
        let footprint = Observation::from_footprint(
            &b,
            "drone-1",
            4.0,
            [100.0, 100.0],
            35.0,
            Some(120.0),
            Sampling::WorldAnchored,
        )
        .unwrap();
        store.apply(footprint);
        store.commit();
        let covered = store.coverage_counts().iter().filter(|&&c| c > 0).count();
        let with_height = store
            .mean_height_grid()
            .iter()
            .filter(|h| h.is_finite())
            .count();
        assert!(covered > 0);
        assert_eq!(covered, with_height);

        // And the two samplings the reference mixes really do differ here:
        // 35 / 10 is not a whole number of cells.
        let anchored = rasterize_circle(&b, [100.0, 100.0], 35.0, Sampling::WorldAnchored).unwrap();
        let disc = rasterize_circle(&b, [100.0, 100.0], 35.0, Sampling::CellDisc).unwrap();
        assert_ne!(anchored.len(), disc.len());
    }

    /// No-truth-access is structural: the store holds grids and a source table,
    /// and every path in is an `Observation`. This pins the consequence -- a
    /// cell nobody observed reads unknown no matter what is really there,
    /// because the store has no way to find out.
    #[test]
    fn the_store_can_only_know_what_was_observed() {
        let b = bounds(10.0, 40.0);
        let mut store = BeliefStore::with_tile_size(b, 4).unwrap();
        store.apply(obs("drone-1", 1.0, vec![occupied_at((0, 0), 12.0)]));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert!(patch.observed[0]);
        assert_eq!(patch.max_obstacle_height_m[0], 12.0);
        // The cell next door holds an obstacle in any caller's world model. The
        // store was never told, so it reports the unknown prior, not free
        // ground -- and its neighbour's evidence does not leak sideways.
        assert!(!patch.observed[1]);
        assert_eq!(patch.obstacle_probability[1], 0.5);
        assert_eq!(patch.max_obstacle_height_m[1], 0.0);
    }

    #[test]
    fn the_max_obstacle_height_takes_the_greatest_report() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs("a", 1.0, vec![occupied_at((0, 0), 4.0)]));
        store.apply(obs("b", 2.0, vec![occupied_at((0, 0), 11.0)]));
        store.apply(obs("c", 3.0, vec![occupied_at((0, 0), 7.0)]));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert_eq!(patch.max_obstacle_height_m[0], 11.0);
    }

    #[test]
    fn a_cell_with_no_occupancy_evidence_keeps_the_unknown_prior() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        store.apply(obs(
            "a",
            1.0,
            vec![CellObservation {
                ij: (0, 0),
                height_m: Some(50.0),
                occupancy: None,
            }],
        ));
        store.commit();
        let patch = store.tile_patch(TileIndex { ti: 0, tj: 0 });
        assert!(patch.observed[0], "a height observation is an observation");
        // Seeing the ground is not evidence about obstacles either way.
        assert_eq!(patch.obstacle_probability[0], 0.5);
    }

    /// Coverage means "a sensor looked here", not "a sensor concluded
    /// something here". A footprint cell with neither a height nor an
    /// occupancy verdict still counts, which is what makes coverage a
    /// different quantity from the belief it carries.
    #[test]
    fn a_cell_is_covered_even_when_the_sensor_concluded_nothing() {
        let b = bounds(10.0, 40.0);
        let mut store = BeliefStore::with_tile_size(b, 4).unwrap();
        let sweep = Observation::from_footprint(
            &b,
            "drone-1",
            2.0,
            [20.0, 20.0],
            12.0,
            None,
            Sampling::CellDisc,
        )
        .unwrap();
        assert!(!sweep.cells.is_empty());
        assert!(sweep
            .cells
            .iter()
            .all(|c| c.height_m.is_none() && c.occupancy.is_none()));
        store.apply(sweep);
        store.commit();

        let covered = store.coverage_counts().iter().filter(|&&c| c > 0).count();
        assert!(covered > 0, "a look with no verdict is still coverage");
        assert!(
            store.mean_height_grid().iter().all(|h| h.is_nan()),
            "and it must not invent a height"
        );
        assert!(store.obstacle_probability_grid().iter().all(|&p| p == 0.5));
    }

    /// `mean_height_m` must stay bit-identical to
    /// `WorldMap.mean_height_grid`, which is `sum / count`. Welford's own
    /// running mean -- the obvious thing to use once M2 is being maintained --
    /// is a different number in the last place, and a port that silently used
    /// it would read as a floating-point bug in P4.6b rather than a design
    /// choice here.
    #[test]
    fn mean_height_is_bit_identical_to_sum_over_count() {
        let mut store = BeliefStore::with_tile_size(bounds(10.0, 40.0), 4).unwrap();
        // These four disagree between the two formulas by one ULP.
        let heights = [287.78, 285.65, 267.49, 188.44];
        for (k, h) in heights.iter().enumerate() {
            store.apply(obs(
                "a",
                k as f64,
                vec![CellObservation {
                    ij: (0, 0),
                    height_m: Some(*h),
                    occupancy: None,
                }],
            ));
        }
        store.commit();

        let mut sum = 0.0f64;
        for h in heights {
            sum += h;
        }
        let reference = sum / heights.len() as f64;

        let mut running = 0.0f64;
        for (k, h) in heights.iter().enumerate() {
            running += (h - running) / (k as f64 + 1.0);
        }
        assert_ne!(
            reference.to_bits(),
            running.to_bits(),
            "the two formulas must actually differ for this test to bind"
        );
        assert_eq!(store.mean_height_grid()[0].to_bits(), reference.to_bits());
    }

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    #[test]
    fn a_zero_tile_size_is_rejected() {
        assert!(BeliefStore::with_tile_size(bounds(10.0, 40.0), 0).is_err());
        assert!(BeliefStore::new(bounds(0.0, 40.0)).is_err());
    }

    #[test]
    fn a_tile_larger_than_the_grid_is_one_tile() {
        let store = BeliefStore::with_tile_size(bounds(10.0, 30.0), 1024).unwrap();
        assert_eq!(store.tile_dimensions(), (1, 1));
        assert_eq!(store.tile_patch(TileIndex { ti: 0, tj: 0 }).dims, (3, 3));
    }

    #[test]
    fn the_default_tile_size_is_used_by_new() {
        let store = BeliefStore::new(bounds(10.0, 1000.0)).unwrap();
        assert_eq!(store.tile_size_cells(), DEFAULT_TILE_SIZE_CELLS);
        assert_eq!(store.tile_dimensions(), (4, 4));
    }
}
