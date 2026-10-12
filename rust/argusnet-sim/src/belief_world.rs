//! Starting from no world knowledge: the map platforms build for themselves,
//! and what they plan against while they build it.
//!
//! In a `prior` run (the default, and every run before this one) the
//! controllers, deconfliction and safety read the true terrain and obstacle
//! layer directly. In a `none` run they never do. Instead:
//!
//! 1. **Sensing.** Each mobile platform carries a range scanner
//!    ([`crate::range_scan`]). Stage 6 scans the true world with it, as it
//!    does with the bearing sensors.
//! 2. **Mapping.** Stage 8 fuses the returns into a [`BeliefGrid`]: a ground
//!    height per cell (the mean of its ground returns), structure evidence
//!    (log-odds occupancy with the highest return seen), and when each cell
//!    was last seen. The grid starts empty. Nothing is known until a ray
//!    lands.
//! 3. **Publication.** After each tick stage 8 publishes an immutable
//!    [`BeliefSnapshot`] to a [`BeliefHandle`]. The next tick's stages 1, 3
//!    and 4 read it through [`BeliefTerrain`], [`BeliefObstacles`] and
//!    [`BeliefVisibility`], which implement the same sampler traits the true
//!    world does.
//! 4. **Unknown is not free.** A query over a cell nobody has seen does not
//!    answer zero height or open air (ARCHITECTURE_AND_PROTOCOLS §3). Ground
//!    height there is the highest ground seen within a search radius, plus a
//!    margin. With nothing seen nearby it is the launch floor plus the
//!    margin. So platforms climb out over unseen ground and settle to their
//!    working height as the map fills under them.
//! 5. **Physics stays true.** A platform cannot be inside the ground or a
//!    building, whatever its map says. The [`PhysicalGuard`] holds the true
//!    world for that one purpose. It puts a platform back above the ground or
//!    outside the structure and records a [`WorldContact`]. A contact means
//!    the map led a platform somewhere it could not be. The count is the
//!    measure of how safe flying on the belief was.
//!
//! The grid is stage 8's state and is checkpointed with it, so a resumed run
//! publishes the same snapshot and plans identically. The handle and the
//! samplers are derived: they hold no state of their own.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, RwLock};

use argusnet_types::Bounds2d;
use serde::{Deserialize, Serialize};

use crate::follow_path::VisibilitySampler;
use crate::motion::{ObstacleSampler, TerrainSampler};
use crate::observe::TerrainHeight;
use crate::range_scan::{RangeScan, RangeScanConfig, ReturnClass};
use crate::runtime_world::{SharedObstacles, SharedTerrain};

/// What a run's platforms know of the world before they sense it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldKnowledge {
    /// The true terrain and obstacle layer, as every run before this had.
    #[default]
    Prior,
    /// Nothing: the platforms map the world themselves.
    #[serde(rename = "none")]
    NoPrior,
}

impl WorldKnowledge {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prior => "prior",
            Self::NoPrior => "none",
        }
    }
}

/// How a no-prior run senses and maps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Belief grid cell size, metres.
    pub resolution_m: f64,
    /// Added to the highest ground seen nearby to stand in for unseen ground.
    pub unknown_margin_m: f64,
    /// How far to look for seen ground when standing in for unseen ground.
    pub unknown_search_radius_m: f64,
    /// Structure evidence above which a cell is treated as occupied.
    pub occupied_probability: f64,
    /// How close above true ground (or how far outside a structure) physics
    /// keeps a platform.
    pub contact_clearance_m: f64,
    pub scan: RangeScanConfig,
    /// Where follow controllers learn a target's position: the run's own
    /// fused tracks (the default), or the legacy truth oracle, kept to
    /// measure what the truth was worth.
    #[serde(default)]
    pub pursuit: PursuitSource,
}

/// The follow controllers' source of target positions in a no-prior run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PursuitSource {
    /// The run's own fused tracks ([`crate::pursuit::TrackPursuit`]).
    #[default]
    Tracks,
    /// The targets' true trajectories, as in a prior run.
    Truth,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            resolution_m: 10.0,
            unknown_margin_m: 30.0,
            unknown_search_radius_m: 200.0,
            // One structure return reaches 0.70; two, 0.85.
            occupied_probability: 0.75,
            contact_clearance_m: 1.0,
            scan: RangeScanConfig::default(),
            pursuit: PursuitSource::default(),
        }
    }
}

impl ExplorationConfig {
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be positive, got {value}"))
            }
        };
        positive("resolution_m", self.resolution_m)?;
        positive("unknown_search_radius_m", self.unknown_search_radius_m)?;
        if !(self.unknown_margin_m.is_finite() && self.unknown_margin_m >= 0.0) {
            return Err(format!(
                "unknown_margin_m must not be negative, got {}",
                self.unknown_margin_m
            ));
        }
        if !(0.5..1.0).contains(&self.occupied_probability) {
            return Err(format!(
                "occupied_probability must be in [0.5, 1), got {}",
                self.occupied_probability
            ));
        }
        self.scan.validate()
    }
}

/// Log-odds added per structure return, and per ground return in the cell.
const LOG_ODDS_STRUCTURE: f64 = 0.85;
const LOG_ODDS_GROUND: f64 = -0.4;
const LOG_ODDS_LIMIT: f64 = 6.0;
/// How far below a structure's highest return a platform may pass under it
/// before it counts as inside: none. A platform at or below the top plus this
/// margin collides.
const STRUCTURE_MARGIN_M: f64 = 2.0;

/// Stage 8's map, built only from range returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefGrid {
    x_min_m: f64,
    y_min_m: f64,
    resolution_m: f64,
    nx: usize,
    ny: usize,
    ground_sum: Vec<f64>,
    ground_count: Vec<u32>,
    structure_top_m: Vec<f64>,
    structure_count: Vec<u32>,
    log_odds: Vec<f64>,
    /// Tick each cell last had a return, `-1` for never.
    last_seen_tick: Vec<i64>,
    /// The launch pads' highest ground: the one thing known before sensing.
    launch_floor_m: f64,
    /// Each cell as the replay last showed it, so a row is written only
    /// when what it shows changes. Checkpointed: a resumed run must write
    /// the rows the uninterrupted run would.
    shown: Vec<ShownCell>,
    /// Cells touched since the replay last took them.
    #[serde(skip)]
    changed: BTreeSet<usize>,
    /// What snapshots share, kept current as returns land. Derived: rebuilt
    /// from the grid after a restore or a reset.
    #[serde(skip)]
    views: Option<Views>,
}

/// Cells per side of the blocks the unseen-ground stand-in is computed over.
const BLOCK_CELLS: usize = 5;

/// The grid as the planners read it, kept current incrementally.
#[derive(Debug, Clone, PartialEq)]
struct Views {
    occupied_probability: f64,
    /// Mean ground per cell, NaN unseen.
    ground: Arc<Vec<f64>>,
    /// Structure top per cell, NaN unless believed occupied.
    structure: Arc<Vec<f64>>,
    /// Highest seen ground per block, NaN where none.
    block_max: Vec<f64>,
    dirty_blocks: BTreeSet<usize>,
}

impl BeliefGrid {
    pub fn new(bounds: Bounds2d, resolution_m: f64, launch_floor_m: f64) -> Result<Self, String> {
        if !(resolution_m.is_finite() && resolution_m > 0.0) {
            return Err(format!("resolution {resolution_m} m is not positive"));
        }
        let span_x = bounds.x_max_m - bounds.x_min_m;
        let span_y = bounds.y_max_m - bounds.y_min_m;
        if !(span_x > 0.0 && span_y > 0.0) {
            return Err("the map bounds are empty".into());
        }
        let nx = ((span_x / resolution_m).ceil() as usize).max(1);
        let ny = ((span_y / resolution_m).ceil() as usize).max(1);
        let cells = nx * ny;
        Ok(Self {
            x_min_m: bounds.x_min_m,
            y_min_m: bounds.y_min_m,
            resolution_m,
            nx,
            ny,
            ground_sum: vec![0.0; cells],
            ground_count: vec![0; cells],
            structure_top_m: vec![0.0; cells],
            structure_count: vec![0; cells],
            log_odds: vec![0.0; cells],
            last_seen_tick: vec![-1; cells],
            launch_floor_m,
            shown: vec![ShownCell::default(); cells],
            changed: BTreeSet::new(),
            views: None,
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn resolution_m(&self) -> f64 {
        self.resolution_m
    }

    pub fn origin_m(&self) -> [f64; 2] {
        [self.x_min_m, self.y_min_m]
    }

    /// Forget everything: the grid a run starts with.
    pub fn clear(&mut self) {
        self.ground_sum.fill(0.0);
        self.ground_count.fill(0);
        self.structure_top_m.fill(0.0);
        self.structure_count.fill(0);
        self.log_odds.fill(0.0);
        self.last_seen_tick.fill(-1);
        self.shown.fill(ShownCell::default());
        self.changed.clear();
        self.views = None;
    }

    /// Whether `other` covers the same cells (for a checkpoint's check).
    pub fn same_layout(&self, other: &Self) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && self.x_min_m == other.x_min_m
            && self.y_min_m == other.y_min_m
            && self.resolution_m == other.resolution_m
    }

    fn cell_of(&self, x_m: f64, y_m: f64) -> Option<usize> {
        let i = ((x_m - self.x_min_m) / self.resolution_m).floor();
        let j = ((y_m - self.y_min_m) / self.resolution_m).floor();
        if i < 0.0 || j < 0.0 || i >= self.nx as f64 || j >= self.ny as f64 {
            return None;
        }
        Some(j as usize * self.nx + i as usize)
    }

    fn touch(&mut self, index: usize, tick: u64) {
        self.last_seen_tick[index] = tick as i64;
        self.changed.insert(index);
        let ground = self.ground_value(index);
        let block = self.block_of(index);
        if let Some(views) = self.views.as_mut() {
            let structure = if self.structure_count[index] > 0
                && probability(self.log_odds[index]) >= views.occupied_probability
            {
                self.structure_top_m[index]
            } else {
                f64::NAN
            };
            Arc::make_mut(&mut views.ground)[index] = ground;
            Arc::make_mut(&mut views.structure)[index] = structure;
            views.dirty_blocks.insert(block);
        }
    }

    fn ground_value(&self, index: usize) -> f64 {
        if self.ground_count[index] > 0 {
            self.ground_sum[index] / self.ground_count[index] as f64
        } else {
            f64::NAN
        }
    }

    fn blocks(&self) -> (usize, usize) {
        (self.nx.div_ceil(BLOCK_CELLS), self.ny.div_ceil(BLOCK_CELLS))
    }

    fn block_of(&self, index: usize) -> usize {
        let (i, j) = (index % self.nx, index / self.nx);
        (j / BLOCK_CELLS) * self.blocks().0 + i / BLOCK_CELLS
    }

    fn block_max_of(&self, ground: &[f64], block: usize) -> f64 {
        let bnx = self.blocks().0;
        let (bi, bj) = (block % bnx, block / bnx);
        let mut best = f64::NAN;
        for j in bj * BLOCK_CELLS..((bj + 1) * BLOCK_CELLS).min(self.ny) {
            for i in bi * BLOCK_CELLS..((bi + 1) * BLOCK_CELLS).min(self.nx) {
                let h = ground[j * self.nx + i];
                if h.is_finite() && (best.is_nan() || h > best) {
                    best = h;
                }
            }
        }
        best
    }

    fn build_views(&self, occupied_probability: f64) -> Views {
        let cells = self.nx * self.ny;
        let ground: Vec<f64> = (0..cells).map(|k| self.ground_value(k)).collect();
        let structure: Vec<f64> = (0..cells)
            .map(|k| {
                if self.structure_count[k] > 0
                    && probability(self.log_odds[k]) >= occupied_probability
                {
                    self.structure_top_m[k]
                } else {
                    f64::NAN
                }
            })
            .collect();
        let (bnx, bny) = self.blocks();
        let block_max = (0..bnx * bny)
            .map(|b| self.block_max_of(&ground, b))
            .collect();
        Views {
            occupied_probability,
            ground: Arc::new(ground),
            structure: Arc::new(structure),
            block_max,
            dirty_blocks: BTreeSet::new(),
        }
    }

    /// Fuse one platform's scan, in return order.
    pub fn integrate(&mut self, scan: &RangeScan, tick: u64) {
        for hit in &scan.returns {
            let Some(index) = self.cell_of(hit.point_m[0], hit.point_m[1]) else {
                continue;
            };
            match hit.class {
                ReturnClass::Ground => {
                    self.ground_sum[index] += hit.point_m[2];
                    self.ground_count[index] += 1;
                    self.log_odds[index] =
                        (self.log_odds[index] + LOG_ODDS_GROUND).max(-LOG_ODDS_LIMIT);
                }
                ReturnClass::Structure => {
                    let top = &mut self.structure_top_m[index];
                    *top = if self.structure_count[index] == 0 {
                        hit.point_m[2]
                    } else {
                        top.max(hit.point_m[2])
                    };
                    self.structure_count[index] += 1;
                    self.log_odds[index] =
                        (self.log_odds[index] + LOG_ODDS_STRUCTURE).min(LOG_ODDS_LIMIT);
                }
            }
            self.touch(index, tick);
        }
    }

    /// Cells with any return.
    pub fn known_cells(&self) -> usize {
        self.last_seen_tick.iter().filter(|t| **t >= 0).count()
    }

    /// Cells believed occupied by a structure, at `threshold`.
    /// Whether cell `(i, j)` is believed occupied at `threshold`.
    pub fn is_structure_ij(&self, i: usize, j: usize, threshold: f64) -> bool {
        let k = j * self.nx + i;
        self.structure_count[k] > 0 && probability(self.log_odds[k]) >= threshold
    }

    pub fn structure_cells_at(&self, threshold: f64) -> usize {
        (0..self.cell_count())
            .filter(|&k| self.structure_count[k] > 0 && probability(self.log_odds[k]) >= threshold)
            .count()
    }

    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    /// Mean ground height of cell `(i, j)`, if any ground return landed there.
    pub fn ground_ij(&self, i: usize, j: usize) -> Option<f64> {
        let index = j * self.nx + i;
        (self.ground_count[index] > 0)
            .then(|| self.ground_sum[index] / self.ground_count[index] as f64)
    }

    /// Centre of cell `(i, j)`, metres.
    pub fn centre_ij(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.x_min_m + (i as f64 + 0.5) * self.resolution_m,
            self.y_min_m + (j as f64 + 0.5) * self.resolution_m,
        ]
    }

    /// The cells whose shown state changed since the last call, as replay
    /// rows in cell order. A touched cell is written when it is first seen,
    /// its ground moves by [`SHOWN_GROUND_STEP_CM`] or more, its occupancy by
    /// [`SHOWN_OCCUPANCY_STEP_PCT`] or more, its structure appears, goes or
    /// moves, or its confidence level steps. Smaller changes wait until they
    /// add up, so the replay's map stays within those steps of the belief.
    pub fn take_changed_cells(&mut self, occupied_probability: f64) -> Vec<BeliefCellUpdate> {
        let mut rows = Vec::new();
        for index in std::mem::take(&mut self.changed) {
            let (i, j) = (index % self.nx, index / self.nx);
            let p = probability(self.log_odds[index]);
            let now = ShownCell {
                ground_cm: self
                    .ground_ij(i, j)
                    .map_or(ShownCell::NONE, |h| (h * 100.0).round() as i32),
                structure_cm: if self.structure_count[index] > 0 && p >= occupied_probability {
                    (self.structure_top_m[index] * 100.0).round() as i32
                } else {
                    ShownCell::NONE
                },
                // Only structure evidence makes occupancy worth showing: a
                // cell with ground returns alone shows as free.
                occupancy_pct: if self.structure_count[index] > 0 {
                    (p * 100.0).round() as u8
                } else {
                    0
                },
                level: confidence_level(self.ground_count[index] + self.structure_count[index]),
            };
            let was = self.shown[index];
            let moved = |a: i32, b: i32, step: i32| {
                (a == ShownCell::NONE) != (b == ShownCell::NONE)
                    || (a != ShownCell::NONE && (a - b).abs() >= step)
            };
            if moved(now.ground_cm, was.ground_cm, SHOWN_GROUND_STEP_CM)
                || moved(now.structure_cm, was.structure_cm, SHOWN_GROUND_STEP_CM)
                || now.occupancy_pct.abs_diff(was.occupancy_pct) >= SHOWN_OCCUPANCY_STEP_PCT
                || now.level != was.level
            {
                self.shown[index] = now;
                let some = |v: i32| (v != ShownCell::NONE).then_some(v);
                rows.push(BeliefCellUpdate(
                    i as u32,
                    j as u32,
                    some(now.ground_cm),
                    some(now.structure_cm),
                    now.occupancy_pct,
                    now.level,
                ));
            }
        }
        rows
    }

    /// The immutable view the next tick plans against. Incremental: only the
    /// blocks returns landed in since the last snapshot are recomputed, and
    /// the cell arrays are shared with the snapshot, not copied.
    pub fn snapshot(&mut self, config: &ExplorationConfig) -> BeliefSnapshot {
        let stale = self
            .views
            .as_ref()
            .is_none_or(|views| views.occupied_probability != config.occupied_probability);
        if stale {
            self.views = Some(self.build_views(config.occupied_probability));
        }
        let mut views = self.views.take().expect("views were just built");
        for block in std::mem::take(&mut views.dirty_blocks) {
            views.block_max[block] = self.block_max_of(&views.ground, block);
        }
        let (bnx, bny) = self.blocks();
        let block_m = self.resolution_m * BLOCK_CELLS as f64;
        let radius = (config.unknown_search_radius_m / block_m).ceil() as usize;
        let stand_in = windowed_max(&views.block_max, bnx, bny, radius)
            .into_iter()
            .map(|h| {
                if h.is_finite() {
                    h.max(self.launch_floor_m) + config.unknown_margin_m
                } else {
                    self.launch_floor_m + config.unknown_margin_m
                }
            })
            .collect();
        let snapshot = BeliefSnapshot {
            x_min_m: self.x_min_m,
            y_min_m: self.y_min_m,
            resolution_m: self.resolution_m,
            nx: self.nx,
            ny: self.ny,
            bnx,
            ground: Arc::clone(&views.ground),
            stand_in,
            structure_top: Arc::clone(&views.structure),
        };
        self.views = Some(views);
        snapshot
    }
}

fn probability(log_odds: f64) -> f64 {
    1.0 / (1.0 + (-log_odds).exp())
}

/// For each cell, the greatest finite value within `radius` cells (a square
/// window), NaN where there is none. Two passes of a monotone deque, so the
/// cost does not grow with the radius.
fn windowed_max(values: &[f64], nx: usize, ny: usize, radius: usize) -> Vec<f64> {
    fn pass(line: &[f64], radius: usize, out: &mut [f64]) {
        let n = line.len();
        let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut next = 0usize;
        for (centre, slot) in out.iter_mut().enumerate() {
            let hi = (centre + radius).min(n - 1);
            while next <= hi {
                if line[next].is_finite() {
                    while deque.back().is_some_and(|&b| line[b] <= line[next]) {
                        deque.pop_back();
                    }
                    deque.push_back(next);
                }
                next += 1;
            }
            let lo = centre.saturating_sub(radius);
            while deque.front().is_some_and(|&f| f < lo) {
                deque.pop_front();
            }
            *slot = deque.front().map_or(f64::NAN, |&f| line[f]);
        }
    }
    let mut rows = vec![f64::NAN; nx * ny];
    for j in 0..ny {
        pass(
            &values[j * nx..(j + 1) * nx],
            radius,
            &mut rows[j * nx..(j + 1) * nx],
        );
    }
    let mut out = vec![f64::NAN; nx * ny];
    let mut column = vec![0.0; ny];
    let mut result = vec![0.0; ny];
    for i in 0..nx {
        for j in 0..ny {
            column[j] = rows[j * nx + i];
        }
        pass(&column, radius, &mut result);
        for j in 0..ny {
            out[j * nx + i] = result[j];
        }
    }
    out
}

/// One belief cell as the replay shows it: `[i, j, ground_cm,
/// structure_top_cm, occupancy_pct, confidence]`. Ground is `null` until a
/// ground return lands; structure is `null` unless the cell is believed
/// occupied; occupancy is 0 for a cell with no structure return; confidence
/// is 1 + floor(log2(returns)), capped at 4 (8 returns or more). Applying a
/// run's rows in frame order rebuilds its map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefCellUpdate(
    pub u32,
    pub u32,
    pub Option<i32>,
    pub Option<i32>,
    pub u8,
    pub u8,
);

/// The smallest ground or structure move, cm, worth a replay row.
pub const SHOWN_GROUND_STEP_CM: i32 = 10;
/// The smallest occupancy move, percentage points, worth a replay row.
pub const SHOWN_OCCUPANCY_STEP_PCT: u8 = 5;

/// A cell as the replay last showed it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct ShownCell {
    ground_cm: i32,
    structure_cm: i32,
    occupancy_pct: u8,
    level: u8,
}

impl ShownCell {
    const NONE: i32 = i32::MIN;
}

impl Default for ShownCell {
    fn default() -> Self {
        Self {
            ground_cm: Self::NONE,
            structure_cm: Self::NONE,
            occupancy_pct: 50,
            level: 0,
        }
    }
}

fn confidence_level(returns: u32) -> u8 {
    if returns == 0 {
        0
    } else {
        (32 - returns.leading_zeros()).min(4) as u8
    }
}

/// The belief as one tick published it. Immutable; shared by `Arc`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefSnapshot {
    x_min_m: f64,
    y_min_m: f64,
    resolution_m: f64,
    nx: usize,
    ny: usize,
    /// Blocks along x, for the stand-in.
    bnx: usize,
    /// NaN where no ground return has landed.
    ground: Arc<Vec<f64>>,
    /// What an unseen cell's ground is taken to be, per block of
    /// [`BLOCK_CELLS`] cells: the highest ground seen within the search
    /// radius of the block (measured in blocks, so up to a block further),
    /// plus the margin. See the module docs.
    stand_in: Vec<f64>,
    /// NaN where the cell is not believed occupied.
    structure_top: Arc<Vec<f64>>,
}

impl BeliefSnapshot {
    fn ij(&self, x_m: f64, y_m: f64) -> (usize, usize) {
        let i = ((x_m - self.x_min_m) / self.resolution_m).floor();
        let j = ((y_m - self.y_min_m) / self.resolution_m).floor();
        (
            i.clamp(0.0, (self.nx - 1) as f64) as usize,
            j.clamp(0.0, (self.ny - 1) as f64) as usize,
        )
    }

    fn stand_in_ij(&self, i: usize, j: usize) -> f64 {
        self.stand_in[(j / BLOCK_CELLS) * self.bnx + i / BLOCK_CELLS]
    }

    fn cell_ground(&self, i: usize, j: usize) -> f64 {
        let ground = self.ground[j * self.nx + i];
        if ground.is_finite() {
            ground
        } else {
            self.stand_in_ij(i, j)
        }
    }

    /// Ground height to plan against at `(x, y)`: the belief where seen,
    /// bilinear between cell centres, and the stand-in where not.
    pub fn ground_at(&self, x_m: f64, y_m: f64) -> f64 {
        let fx = (x_m - self.x_min_m) / self.resolution_m - 0.5;
        let fy = (y_m - self.y_min_m) / self.resolution_m - 0.5;
        let i0 = fx.floor().clamp(0.0, (self.nx - 1) as f64) as usize;
        let j0 = fy.floor().clamp(0.0, (self.ny - 1) as f64) as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let tx = (fx - i0 as f64).clamp(0.0, 1.0);
        let ty = (fy - j0 as f64).clamp(0.0, 1.0);
        let h00 = self.cell_ground(i0, j0);
        let h10 = self.cell_ground(i1, j0);
        let h01 = self.cell_ground(i0, j1);
        let h11 = self.cell_ground(i1, j1);
        let bilinear = h00 * (1.0 - tx) * (1.0 - ty)
            + h10 * tx * (1.0 - ty)
            + h01 * (1.0 - tx) * ty
            + h11 * tx * ty;
        // Never below the cell's own value when that is a stand-in: unseen
        // ground blended toward seen ground must stay conservative.
        let (i, j) = self.ij(x_m, y_m);
        if self.ground[j * self.nx + i].is_finite() {
            bilinear
        } else {
            bilinear.max(self.stand_in_ij(i, j))
        }
    }

    pub fn is_seen(&self, x_m: f64, y_m: f64) -> bool {
        let (i, j) = self.ij(x_m, y_m);
        self.ground[j * self.nx + i].is_finite()
    }

    /// The highest structure return in the cell at `(x, y)`, if it is
    /// believed occupied.
    pub fn structure_top_at(&self, x_m: f64, y_m: f64) -> Option<f64> {
        let (i, j) = self.ij(x_m, y_m);
        let top = self.structure_top[j * self.nx + i];
        top.is_finite().then_some(top)
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        self.structure_top_at(point_m[0], point_m[1])
            .is_some_and(|top| point_m[2] <= top + STRUCTURE_MARGIN_M)
    }
}

/// The published belief: written by stage 8 after each tick, read by the
/// next tick's planning stages.
#[derive(Debug, Clone)]
pub struct BeliefHandle(Arc<RwLock<Arc<BeliefSnapshot>>>);

impl BeliefHandle {
    pub fn new(initial: BeliefSnapshot) -> Self {
        Self(Arc::new(RwLock::new(Arc::new(initial))))
    }

    pub fn publish(&self, snapshot: BeliefSnapshot) {
        *self.0.write().expect("belief lock") = Arc::new(snapshot);
    }

    pub fn current(&self) -> Arc<BeliefSnapshot> {
        Arc::clone(&self.0.read().expect("belief lock"))
    }
}

/// Terrain as the belief has it. Implements every terrain read the planning
/// stages make, so it stands in for the true terrain without their knowing.
#[derive(Debug, Clone)]
pub struct BeliefTerrain {
    handle: BeliefHandle,
    gradient_step_m: f64,
}

impl BeliefTerrain {
    pub fn new(handle: BeliefHandle, resolution_m: f64) -> Self {
        Self {
            handle,
            gradient_step_m: resolution_m * 0.5,
        }
    }
}

impl TerrainSampler for BeliefTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        self.handle.current().ground_at(x_m, y_m)
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2] {
        let belief = self.handle.current();
        let d = delta_m.unwrap_or(self.gradient_step_m).max(1e-3);
        [
            (belief.ground_at(x_m + d, y_m) - belief.ground_at(x_m - d, y_m)) / (2.0 * d),
            (belief.ground_at(x_m, y_m + d) - belief.ground_at(x_m, y_m - d)) / (2.0 * d),
        ]
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        z_m.max(self.handle.current().ground_at(x_m, y_m) + min_agl_m.max(0.0))
    }
}

impl TerrainHeight for BeliefTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        TerrainSampler::height_at(self, x_m, y_m)
    }
}

/// Structures as the belief has them: occupied cells, up to their highest
/// return.
#[derive(Debug, Clone)]
pub struct BeliefObstacles {
    handle: BeliefHandle,
}

impl BeliefObstacles {
    pub fn new(handle: BeliefHandle) -> Self {
        Self { handle }
    }
}

impl ObstacleSampler for BeliefObstacles {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        let belief = self.handle.current();
        if !belief.collides(point_m) {
            return None;
        }
        // The nearest clear cell centre, searched ring by ring outward, then
        // `margin` further along the same direction. Ties go to the first
        // cell in ring order, so the push is deterministic.
        let (ci, cj) = belief.ij(point_m[0], point_m[1]);
        let max_ring = belief.nx.max(belief.ny);
        for ring in 1..=max_ring {
            let mut best: Option<([f64; 2], f64)> = None;
            let r = ring as i64;
            for dj in -r..=r {
                for di in -r..=r {
                    if di.abs() != r && dj.abs() != r {
                        continue;
                    }
                    let (i, j) = (ci as i64 + di, cj as i64 + dj);
                    if i < 0 || j < 0 || i >= belief.nx as i64 || j >= belief.ny as i64 {
                        continue;
                    }
                    let centre = [
                        belief.x_min_m + (i as f64 + 0.5) * belief.resolution_m,
                        belief.y_min_m + (j as f64 + 0.5) * belief.resolution_m,
                    ];
                    if belief.collides([centre[0], centre[1], point_m[2]]) {
                        continue;
                    }
                    let d = (centre[0] - point_m[0]).hypot(centre[1] - point_m[1]);
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((centre, d));
                    }
                }
            }
            if let Some((centre, d)) = best {
                let (ux, uy) = if d > 1e-9 {
                    ((centre[0] - point_m[0]) / d, (centre[1] - point_m[1]) / d)
                } else {
                    (1.0, 0.0)
                };
                return Some([centre[0] + ux * margin_m, centre[1] + uy * margin_m]);
            }
        }
        None
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        self.handle.current().collides(point_m)
    }
}

/// Line of sight over the belief: blocked by seen ground and believed
/// structures. Unseen cells do not block. This answers "can I expect to see
/// the target from there", a planning question, not a safety one.
#[derive(Debug, Clone)]
pub struct BeliefVisibility {
    handle: BeliefHandle,
    clearance_m: f64,
}

impl BeliefVisibility {
    pub fn new(handle: BeliefHandle, clearance_m: f64) -> Self {
        Self {
            handle,
            clearance_m,
        }
    }
}

impl VisibilitySampler for BeliefVisibility {
    fn is_visible(&self, observer_m: [f64; 3], target_m: [f64; 3]) -> bool {
        let belief = self.handle.current();
        let length = (0..3)
            .map(|k| (target_m[k] - observer_m[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let steps = ((length / (belief.resolution_m * 0.5)).ceil() as usize).max(1);
        // The end points themselves are not tested: the observer and target
        // stand where they stand.
        for s in 1..steps {
            let t = s as f64 / steps as f64;
            let p: [f64; 3] =
                std::array::from_fn(|k| observer_m[k] + (target_m[k] - observer_m[k]) * t);
            let (i, j) = belief.ij(p[0], p[1]);
            let index = j * belief.nx + i;
            let ground = belief.ground[index];
            if ground.is_finite() && p[2] <= ground + self.clearance_m {
                return false;
            }
            if belief.structure_top[index].is_finite() && p[2] <= belief.structure_top[index] {
                return false;
            }
        }
        true
    }
}

/// A platform physics put back: the map had it somewhere it could not be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldContact {
    pub node_id: String,
    pub timestamp_s: f64,
    /// `terrain` or `structure`.
    pub kind: String,
    /// How far physics moved the platform, metres.
    pub correction_m: f64,
}

/// The true world, held only to keep platforms out of it.
#[derive(Clone)]
pub struct PhysicalGuard {
    terrain: SharedTerrain,
    obstacles: SharedObstacles,
    clearance_m: f64,
    contacts: Arc<Mutex<Vec<WorldContact>>>,
}

impl std::fmt::Debug for PhysicalGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PhysicalGuard")
            .field("clearance_m", &self.clearance_m)
            .finish()
    }
}

impl PhysicalGuard {
    pub fn new(terrain: SharedTerrain, obstacles: SharedObstacles, clearance_m: f64) -> Self {
        Self {
            terrain,
            obstacles,
            clearance_m,
            contacts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// `position` moved out of the ground and out of any structure, with a
    /// contact recorded for each move.
    pub fn enforce(&self, node_id: &str, timestamp_s: f64, position: [f64; 3]) -> [f64; 3] {
        let mut p = position;
        if let Some(xy) = self.obstacles.push_clear(p, self.clearance_m) {
            let correction = (xy[0] - p[0]).hypot(xy[1] - p[1]);
            p[0] = xy[0];
            p[1] = xy[1];
            self.record(node_id, timestamp_s, "structure", correction);
        }
        let floor = TerrainSampler::height_at(&self.terrain, p[0], p[1]) + self.clearance_m;
        if p[2] < floor {
            self.record(node_id, timestamp_s, "terrain", floor - p[2]);
            p[2] = floor;
        }
        p
    }

    fn record(&self, node_id: &str, timestamp_s: f64, kind: &str, correction_m: f64) {
        self.contacts
            .lock()
            .expect("contact log")
            .push(WorldContact {
                node_id: node_id.to_string(),
                timestamp_s,
                kind: kind.to_string(),
                correction_m,
            });
    }

    /// The contacts since the last call.
    pub fn take_contacts(&self) -> Vec<WorldContact> {
        std::mem::take(&mut *self.contacts.lock().expect("contact log"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_scan::RangeReturn;

    fn grid() -> BeliefGrid {
        BeliefGrid::new(Bounds2d::new(0.0, 200.0, 0.0, 200.0), 10.0, 50.0).unwrap()
    }

    fn scan(returns: Vec<RangeReturn>) -> RangeScan {
        RangeScan {
            node_id: "d".into(),
            timestamp_s: 0.0,
            origin_m: [0.0, 0.0, 100.0],
            returns,
            misses: 0,
        }
    }

    fn ground(x: f64, y: f64, z: f64) -> RangeReturn {
        RangeReturn {
            point_m: [x, y, z],
            class: ReturnClass::Ground,
        }
    }

    #[test]
    fn an_empty_belief_stands_in_conservatively_everywhere() {
        let config = ExplorationConfig::default();
        let snapshot = grid().snapshot(&config);
        assert_eq!(
            snapshot.ground_at(155.0, 155.0),
            50.0 + config.unknown_margin_m
        );
        assert_eq!(
            snapshot.ground_at(55.0, 55.0),
            50.0 + config.unknown_margin_m
        );
        assert!(!snapshot.is_seen(55.0, 55.0));
    }

    #[test]
    fn seen_ground_is_its_mean_and_unseen_ground_stays_above_it() {
        let config = ExplorationConfig {
            unknown_margin_m: 10.0,
            unknown_search_radius_m: 30.0,
            ..ExplorationConfig::default()
        };
        let mut belief = grid();
        belief.integrate(
            &scan(vec![ground(15.0, 15.0, 60.0), ground(15.0, 15.0, 62.0)]),
            3,
        );
        assert_eq!(belief.known_cells(), 1);
        let snapshot = belief.snapshot(&config);
        assert!((snapshot.ground_at(15.0, 15.0) - 61.0).abs() < 1e-9);
        // A neighbour within the radius stands in at the highest seen + margin.
        assert!((snapshot.ground_at(35.0, 15.0) - 71.0).abs() < 1e-9);
        // Beyond it (the radius is counted in 50 m blocks), the launch floor
        // + margin.
        assert!((snapshot.ground_at(195.0, 195.0) - 60.0).abs() < 1e-9);
        // Incremental snapshots agree with one built from scratch.
        belief.integrate(&scan(vec![ground(185.0, 15.0, 90.0)]), 4);
        let incremental = belief.snapshot(&config);
        let mut fresh: BeliefGrid =
            serde_json::from_str(&serde_json::to_string(&belief).unwrap()).unwrap();
        // Debug renderings, since NaN (unseen) never equals itself.
        assert_eq!(
            format!("{:?}", fresh.snapshot(&config)),
            format!("{incremental:?}")
        );
        let rows = belief.take_changed_cells(config.occupied_probability);
        assert_eq!(
            rows,
            vec![
                BeliefCellUpdate(1, 1, Some(6100), None, 0, 2),
                BeliefCellUpdate(18, 1, Some(9000), None, 0, 1),
            ]
        );
        // A return that moves nothing shown writes no row.
        belief.integrate(&scan(vec![ground(15.0, 15.0, 61.0)]), 4);
        assert!(belief
            .take_changed_cells(config.occupied_probability)
            .is_empty());
        assert!(belief.take_changed_cells(0.65).is_empty());
    }

    #[test]
    fn structure_returns_make_an_obstacle_to_its_top() {
        let config = ExplorationConfig::default();
        let mut belief = grid();
        let wall = RangeReturn {
            point_m: [45.0, 45.0, 80.0],
            class: ReturnClass::Structure,
        };
        belief.integrate(&scan(vec![wall]), 0);
        let handle = BeliefHandle::new(belief.snapshot(&config));
        let obstacles = BeliefObstacles::new(handle.clone());
        assert!(
            !obstacles.collides([45.0, 45.0, 79.0]),
            "one return is not proof"
        );
        belief.integrate(&scan(vec![wall, wall]), 1);
        handle.publish(belief.snapshot(&config));
        assert!(obstacles.collides([45.0, 45.0, 79.0]));
        assert!(
            !obstacles.collides([45.0, 45.0, 90.0]),
            "above the top is clear"
        );
        let pushed = obstacles
            .push_clear([45.0, 45.0, 79.0], 1.0)
            .expect("pushed");
        assert!(!obstacles.collides([pushed[0], pushed[1], 79.0]));
        assert!(
            BeliefVisibility::new(handle, 0.0).is_visible([5.0, 45.0, 85.0], [95.0, 45.0, 85.0])
        );
    }

    #[test]
    fn the_windowed_max_matches_brute_force() {
        let (nx, ny, r) = (7usize, 5usize, 2usize);
        let values: Vec<f64> = (0..nx * ny)
            .map(|k| {
                if k % 3 == 0 {
                    f64::NAN
                } else {
                    (k * 37 % 11) as f64
                }
            })
            .collect();
        let fast = windowed_max(&values, nx, ny, r);
        for j in 0..ny {
            for i in 0..nx {
                let mut best = f64::NAN;
                for jj in j.saturating_sub(r)..=(j + r).min(ny - 1) {
                    for ii in i.saturating_sub(r)..=(i + r).min(nx - 1) {
                        let v = values[jj * nx + ii];
                        if v.is_finite() && (best.is_nan() || v > best) {
                            best = v;
                        }
                    }
                }
                let got = fast[j * nx + i];
                assert!(got == best || (got.is_nan() && best.is_nan()), "({i},{j})");
            }
        }
    }

    #[test]
    fn the_grid_round_trips_through_a_checkpoint() {
        let mut belief = grid();
        belief.integrate(&scan(vec![ground(25.0, 35.0, 70.0)]), 2);
        let text = serde_json::to_string(&belief).unwrap();
        let back: BeliefGrid = serde_json::from_str(&text).unwrap();
        assert!(back.same_layout(&belief));
        assert_eq!(back.ground_ij(2, 3), Some(70.0));
    }
}
