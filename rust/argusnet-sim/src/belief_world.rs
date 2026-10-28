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
//! 5. **Physics stays true.** A platform cannot be inside the ground, a
//!    building or a power line's wire, whatever its map says. The
//!    [`PhysicalGuard`] holds the true world for that one purpose. It puts a
//!    platform back above the ground or outside the structure or the wire and
//!    records a [`WorldContact`]. A contact means
//!    the map led a platform somewhere it could not be. The count is the
//!    measure of how safe flying on the belief was.
//!
//! The grid is stage 8's state and is checkpointed with it, so a resumed run
//! publishes the same snapshot and plans identically. The handle and the
//! samplers are derived: they hold no state of their own.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, RwLock};

use argusnet_mapping::quality::{
    assess, CellEvidence, DeficitCounts, DeficitReason, QualityRequirement,
};
use argusnet_types::Bounds2d;
use argusnet_world::powerline::{ConductorEnvelope, PowerLine};
use serde::{Deserialize, Serialize};

use crate::follow_path::VisibilitySampler;
use crate::motion::{ObstacleSampler, TerrainSampler};
use crate::nav_camera::NavCameraConfig;
use crate::observe::TerrainHeight;
use crate::range_scan::{RangeScan, RangeScanConfig, ReturnClass, ScanSensor};
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
    /// Whether search drones explore the frontier of the map
    /// ([`crate::frontier::FrontierExplorer`]) instead of flying the lanes
    /// planned before the run. The scan mission's progress is the map's known
    /// fraction either way.
    #[serde(default = "default_explore")]
    pub explore: bool,
    /// Where follow controllers learn a target's position: the run's own
    /// fused tracks (the default), or the legacy truth oracle, kept to
    /// measure what the truth was worth.
    #[serde(default)]
    pub pursuit: PursuitSource,
    /// The wire-hazard prior (R3): every power line's spans are known from
    /// the start, as a utility's map of its corridors would give them, so
    /// the run plans round them before any scan has seen one. Off (the
    /// default) learns each span only from the scanners.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub wire_prior: bool,
    /// The forward navigation camera, the primary obstacle-avoidance sensor
    /// ([`crate::nav_camera`]). Its detections are safety evidence only, and
    /// every controller is capped to the speed its envelope allows. Absent
    /// (the default) is every run before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nav_camera: Option<NavCameraConfig>,
    /// The declared rig (SR1): with it, the pod's LiDAR on a 3-axis gimbal
    /// maps in place of the legacy scanner, and the rig's navigation camera
    /// is the avoidance sensor. Absent is every run before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<crate::rig::RigProfile>,
    /// What the planner asks the pod to hold while mapping (SR2). Only with
    /// a rig; nadir when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointing: Option<crate::pod::PointingPolicy>,
    /// View planning (SR5d): explorers go to the deficit worth most per
    /// watt-hour, and the pod rolls toward the deficits its footprint can
    /// fix. Needs a rig and a quality requirement; cruise only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_planning: Option<argusnet_planning::views::ViewPlanningConfig>,
    /// What each cell of the map must reach (SR5a). With it, the map keeps
    /// each cell's accepted evidence and publishes why each falls short;
    /// explorers go where flying can still fix a deficit (SR5b) and the scan
    /// phase measures progress as the share of reachable cells that meet
    /// it. Absent (the default) is every run before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualityRequirement>,
}

impl ExplorationConfig {
    /// The declared one-sigma range noise of the sensor that maps: the
    /// rig's LiDAR, or the legacy scanner.
    pub fn mapping_range_noise_m(&self) -> f64 {
        self.rig
            .as_ref()
            .and_then(|rig| rig.mount(crate::rig::SensorRole::Lidar))
            .and_then(|mount| mount.lidar.as_ref())
            .map_or(self.scan.range_noise_m, |lidar| lidar.range_noise_m)
    }

    /// The navigation camera the run flies: the rig's, or the stand-alone
    /// one.
    pub fn effective_nav_camera(&self) -> Option<NavCameraConfig> {
        match &self.rig {
            Some(rig) => rig
                .mount(crate::rig::SensorRole::NavCamera)
                .and_then(|mount| mount.nav_camera.clone()),
            None => self.nav_camera.clone(),
        }
    }
}

fn default_explore() -> bool {
    true
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
            explore: true,
            pursuit: PursuitSource::default(),
            wire_prior: false,
            nav_camera: None,
            rig: None,
            pointing: None,
            view_planning: None,
            quality: None,
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
        if let Some(camera) = &self.nav_camera {
            camera.validate()?;
        }
        if let Some(rig) = &self.rig {
            if self.nav_camera.is_some() {
                return Err(
                    "the rig carries the navigation camera; do not declare a second one".into(),
                );
            }
            rig.validate()?;
        }
        match (&self.pointing, &self.rig) {
            (Some(_), None) => return Err("pointing needs a rig with a pod".into()),
            (Some(policy), Some(_)) => policy.validate()?,
            _ => {}
        }
        if let Some(quality) = &self.quality {
            quality.validate()?;
        }
        if let Some(planning) = &self.view_planning {
            if self.rig.is_none() || self.quality.is_none() {
                return Err(
                    "view planning needs a rig to point and a quality requirement to plan on"
                        .into(),
                );
            }
            if self.pointing.is_some() {
                return Err("view planning chooses the pointing; do not set one as well".into());
            }
            planning.validate()?;
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
    /// The power-line spans the scanners have put points on (R3), by the
    /// fitted span's id (`line#k`). Absent when none, so a checkpoint without
    /// lines is unchanged.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    wires: BTreeMap<String, PowerLine>,
    /// The spans known before sensing: the opt-in wire-hazard prior. A reset
    /// goes back to them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    prior_wires: Vec<PowerLine>,
    /// [`Self::wires`] as snapshots share them. Derived.
    #[serde(skip)]
    wires_view: Option<Arc<Vec<PowerLine>>>,
    /// Cells touched since the replay last took them.
    #[serde(skip)]
    changed: BTreeSet<usize>,
    /// Canopy evidence (R2): per cell, the highest canopy return (meaningful
    /// where any landed; never NaN, so a checkpoint is plain JSON) and how
    /// many landed. Soft structure: a crown is something to
    /// keep clear of, not a wall, so it never makes a cell occupied. Empty
    /// until the first canopy return lands, so a run in a world without
    /// trees checkpoints exactly as it did before them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    canopy_top_m: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    canopy_count: Vec<u32>,
    /// Each cell's canopy top as the replay last showed it, cm
    /// ([`ShownCell::NONE`] for none). Checkpointed like `shown`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    canopy_shown_cm: Vec<i32>,
    /// Cells with canopy returns since the replay last took them.
    #[serde(skip)]
    canopy_changed: BTreeSet<usize>,
    /// The navigation camera's obstacle evidence (§8.4 of the sensing
    /// guide): per cell, the highest detection (meaningful where any landed)
    /// and its log-odds. Safety evidence only: it never makes a cell
    /// measured, so it neither moves the frontier nor counts as mapped
    /// structure, and only ever adds obstacles (a camera that saw nothing
    /// clears nothing). Empty until the first detection lands, so a run
    /// without the camera checkpoints exactly as before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    safety_top_m: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    safety_log_odds: Vec<f64>,
    /// Pod captures refused because the gimbal had not settled (SR2).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    unsettled_scans: u64,
    /// Pod captures refused because the pod was stowed: take-off, landing
    /// or a hold below cruise, when it is not surveying (SR2).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    stowed_scans: u64,
    /// Accepted-evidence tracking (SR5a): the steepest-from-nadir ray, in
    /// degrees, whose ground return a cell accepts. `None` keeps no such
    /// evidence, and the grid checkpoints as it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accept_incidence_deg: Option<f64>,
    /// Per cell, the accepted ground returns: count, sum and sum of squares
    /// of their heights, and the tick of the newest (`-1` for none). Empty
    /// without tracking.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_count: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_sum_m: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_sq_m2: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_tick: Vec<i64>,
    /// The tick the quality was last published at, so a restored grid
    /// publishes the same ages. `None` without tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quality_tick: Option<u64>,
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
    /// The camera's obstacle top per cell, NaN unless believed occupied;
    /// empty without camera evidence.
    safety: Arc<Vec<f64>>,
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
            wires: BTreeMap::new(),
            prior_wires: Vec::new(),
            wires_view: None,
            changed: BTreeSet::new(),
            canopy_top_m: Vec::new(),
            canopy_count: Vec::new(),
            canopy_shown_cm: Vec::new(),
            canopy_changed: BTreeSet::new(),
            safety_top_m: Vec::new(),
            safety_log_odds: Vec::new(),
            unsettled_scans: 0,
            stowed_scans: 0,
            accept_incidence_deg: None,
            accepted_count: Vec::new(),
            accepted_sum_m: Vec::new(),
            accepted_sq_m2: Vec::new(),
            accepted_tick: Vec::new(),
            quality_tick: None,
            views: None,
        })
    }

    /// Know `spans` before sensing (the wire-hazard prior, R3): mapped from
    /// the start, and again after a reset. Each is one span of a line
    /// ([`PowerLine::span_line`]).
    pub fn know_wires(&mut self, spans: Vec<PowerLine>) {
        self.prior_wires = spans;
        self.reset_wires();
    }

    fn reset_wires(&mut self) {
        self.wires = self
            .prior_wires
            .iter()
            .map(|span| (span.id.clone(), span.clone()))
            .collect();
        self.wires_view = None;
    }

    /// The spans mapped so far.
    pub fn wires(&self) -> impl Iterator<Item = &PowerLine> {
        self.wires.values()
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
        self.canopy_top_m = Vec::new();
        self.canopy_count = Vec::new();
        self.canopy_shown_cm = Vec::new();
        self.canopy_changed.clear();
        self.safety_top_m = Vec::new();
        self.safety_log_odds = Vec::new();
        self.unsettled_scans = 0;
        self.stowed_scans = 0;
        if self.accept_incidence_deg.is_some() {
            self.accepted_count.fill(0);
            self.accepted_sum_m.fill(0.0);
            self.accepted_sq_m2.fill(0.0);
            self.accepted_tick.fill(-1);
            self.quality_tick = Some(0);
        }
        self.views = None;
        self.reset_wires();
    }

    /// Whether `other` keeps accepted evidence under the same incidence
    /// limit (SR5a), or neither keeps any: a checkpoint's evidence is only
    /// meaningful under the admission policy that gathered it.
    pub fn same_quality_tracking(&self, other: &Self) -> bool {
        self.accept_incidence_deg == other.accept_incidence_deg
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
        let safety: Vec<f64> = (0..self.safety_log_odds.len())
            .map(|k| self.safety_value(k, occupied_probability))
            .collect();
        let (bnx, bny) = self.blocks();
        let block_max = (0..bnx * bny)
            .map(|b| self.block_max_of(&ground, b))
            .collect();
        Views {
            occupied_probability,
            ground: Arc::new(ground),
            structure: Arc::new(structure),
            safety: Arc::new(safety),
            block_max,
            dirty_blocks: BTreeSet::new(),
        }
    }

    /// The camera's obstacle top in cell `index`, NaN unless occupied. A
    /// cell needs positive camera evidence: an untouched cell's zero
    /// log-odds passes a threshold of 0.5 but has no obstacle in it.
    fn safety_value(&self, index: usize, occupied_probability: f64) -> f64 {
        match self.safety_log_odds.get(index) {
            Some(&log_odds) if log_odds > 0.0 && probability(log_odds) >= occupied_probability => {
                self.safety_top_m[index]
            }
            _ => f64::NAN,
        }
    }

    /// Fuse one navigation camera frame: structure detections raise the
    /// cell's safety evidence. Crowns stay soft, as the scanner's do: never
    /// occupied. Nothing here touches the mapped cells.
    fn integrate_camera(&mut self, scan: &RangeScan) {
        for hit in &scan.returns {
            if hit.class != ReturnClass::Structure {
                continue;
            }
            let Some(index) = self.cell_of(hit.point_m[0], hit.point_m[1]) else {
                continue;
            };
            if self.safety_log_odds.is_empty() {
                let cells = self.cell_count();
                self.safety_top_m = vec![0.0; cells];
                self.safety_log_odds = vec![0.0; cells];
            }
            let top = &mut self.safety_top_m[index];
            *top = if self.safety_log_odds[index] == 0.0 {
                hit.point_m[2]
            } else {
                top.max(hit.point_m[2])
            };
            self.safety_log_odds[index] =
                (self.safety_log_odds[index] + LOG_ODDS_STRUCTURE).min(LOG_ODDS_LIMIT);
            let cells = self.cell_count();
            if let Some(occupied) = self.views.as_ref().map(|v| v.occupied_probability) {
                let value = self.safety_value(index, occupied);
                let views = self.views.as_mut().expect("checked above");
                let safety = Arc::make_mut(&mut views.safety);
                if safety.is_empty() {
                    *safety = vec![f64::NAN; cells];
                }
                safety[index] = value;
            }
        }
    }

    /// Pod captures refused unsettled.
    pub fn unsettled_scans(&self) -> u64 {
        self.unsettled_scans
    }

    /// Pod captures refused stowed.
    pub fn stowed_scans(&self) -> u64 {
        self.stowed_scans
    }

    /// Cells the navigation camera believes occupied.
    pub fn safety_cells_at(&self, threshold: f64) -> usize {
        self.safety_log_odds
            .iter()
            .filter(|&&log_odds| log_odds > 0.0 && probability(log_odds) >= threshold)
            .count()
    }

    /// Keep each cell's accepted evidence: ground returns from rays within
    /// `max_incidence_deg` of nadir (SR5a). Before any scan.
    pub fn track_quality(&mut self, max_incidence_deg: f64) {
        let cells = self.cell_count();
        self.accept_incidence_deg = Some(max_incidence_deg);
        self.accepted_count = vec![0; cells];
        self.accepted_sum_m = vec![0.0; cells];
        self.accepted_sq_m2 = vec![0.0; cells];
        self.accepted_tick = vec![-1; cells];
        self.quality_tick = Some(0);
    }

    /// Record that the quality is published as of `tick`; a grid without
    /// tracking keeps none.
    pub fn mark_quality_tick(&mut self, tick: u64) {
        if self.quality_tick.is_some() {
            self.quality_tick = Some(tick);
        }
    }

    /// The tick the quality was last published at (SR5a).
    pub fn quality_tick(&self) -> Option<u64> {
        self.quality_tick
    }

    /// Cell `index`'s accepted evidence at tick `now`, ages in seconds at
    /// `dt_s` a tick.
    fn evidence(&self, index: usize, now: u64, dt_s: f64) -> CellEvidence {
        let accepted_returns = self.accepted_count.get(index).copied().unwrap_or(0);
        CellEvidence {
            ground_returns: self.ground_count[index],
            accepted_returns,
            accepted_height_sum_m: self.accepted_sum_m.get(index).copied().unwrap_or(0.0),
            accepted_height_sq_sum_m2: self.accepted_sq_m2.get(index).copied().unwrap_or(0.0),
            accepted_age_s: self
                .accepted_tick
                .get(index)
                .filter(|&&t| t >= 0 && accepted_returns > 0)
                .map(|&t| (now as i64 - t).max(0) as f64 * dt_s),
        }
    }

    /// Every cell's quality against `requirement` at tick `now` (SR5a).
    /// `per_return_sigma_m` is the scanner's declared range noise.
    pub fn quality(
        &self,
        requirement: &QualityRequirement,
        per_return_sigma_m: f64,
        now: u64,
        dt_s: f64,
    ) -> QualityLayer {
        let area = self.resolution_m * self.resolution_m;
        let reasons: Vec<DeficitReason> = (0..self.cell_count())
            .map(|k| {
                assess(
                    &self.evidence(k, now, dt_s),
                    area,
                    per_return_sigma_m,
                    requirement,
                )
                .reason
            })
            .collect();
        let mut counts = DeficitCounts::default();
        for reason in &reasons {
            counts.add(*reason);
        }
        QualityLayer { reasons, counts }
    }

    /// Fuse one platform's scan, in return order. A wire span is mapped the
    /// first time a scan puts points on it, whichever sensor saw it. A
    /// navigation camera frame is safety evidence only
    /// ([`Self::integrate_camera`]).
    pub fn integrate(&mut self, scan: &RangeScan, tick: u64) {
        // A pod capture is survey evidence only while the pod is surveying:
        // stowed (take-off, landing, a hold below cruise) its rays graze the
        // ground far ahead, and still moving its pose is not the one the
        // returns were georeferenced with to within the settle tolerance.
        // Refused whole, wires included.
        if let Some(pod) = &scan.pod {
            if pod.source == crate::pod::PointingSource::Stow {
                self.stowed_scans += 1;
                return;
            }
            if !pod.settled {
                self.unsettled_scans += 1;
                return;
            }
        }
        for wire in &scan.wires {
            if !self.wires.contains_key(&wire.span.id) {
                self.wires.insert(wire.span.id.clone(), wire.span.clone());
                self.wires_view = None;
            }
        }
        if scan.sensor == ScanSensor::NavCamera {
            self.integrate_camera(scan);
            return;
        }
        for hit in &scan.returns {
            let Some(index) = self.cell_of(hit.point_m[0], hit.point_m[1]) else {
                continue;
            };
            match hit.class {
                ReturnClass::Ground => {
                    self.ground_sum[index] += hit.point_m[2];
                    self.ground_count[index] += 1;
                    if let Some(limit_deg) = self.accept_incidence_deg {
                        let horizontal = (hit.point_m[0] - scan.origin_m[0])
                            .hypot(hit.point_m[1] - scan.origin_m[1]);
                        let drop = scan.origin_m[2] - hit.point_m[2];
                        // From nadir: 0 straight down, 90 level.
                        let incidence_deg = horizontal.atan2(drop).to_degrees();
                        if incidence_deg <= limit_deg {
                            let z = hit.point_m[2];
                            self.accepted_count[index] += 1;
                            self.accepted_sum_m[index] += z;
                            self.accepted_sq_m2[index] += z * z;
                            self.accepted_tick[index] = tick as i64;
                        }
                    }
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
                ReturnClass::Canopy => {
                    if self.canopy_count.is_empty() {
                        let cells = self.cell_count();
                        self.canopy_top_m = vec![0.0; cells];
                        self.canopy_count = vec![0; cells];
                        self.canopy_shown_cm = vec![ShownCell::NONE; cells];
                    }
                    let top = &mut self.canopy_top_m[index];
                    *top = if self.canopy_count[index] == 0 {
                        hit.point_m[2]
                    } else {
                        top.max(hit.point_m[2])
                    };
                    self.canopy_count[index] += 1;
                    self.canopy_changed.insert(index);
                }
            }
            self.touch(index, tick);
        }
    }

    /// Cells with any return.
    pub fn known_cells(&self) -> usize {
        self.last_seen_tick.iter().filter(|t| **t >= 0).count()
    }

    /// The highest canopy return in cell `(i, j)`, if any landed there.
    pub fn canopy_top_ij(&self, i: usize, j: usize) -> Option<f64> {
        let index = j * self.nx + i;
        (self.canopy_count.get(index).copied().unwrap_or(0) > 0).then(|| self.canopy_top_m[index])
    }

    /// Cells with a canopy return.
    pub fn canopy_cells(&self) -> usize {
        self.canopy_count.iter().filter(|&&n| n > 0).count()
    }

    /// The cells whose shown canopy top changed since the last call, as
    /// replay rows in cell order: when first seen, or moved by
    /// [`SHOWN_GROUND_STEP_CM`] or more.
    pub fn take_changed_canopy(&mut self) -> Vec<CanopyCellUpdate> {
        let mut rows = Vec::new();
        for index in std::mem::take(&mut self.canopy_changed) {
            let now = (self.canopy_top_m[index] * 100.0).round() as i32;
            let was = self.canopy_shown_cm[index];
            if was == ShownCell::NONE || (now - was).abs() >= SHOWN_GROUND_STEP_CM {
                self.canopy_shown_cm[index] = now;
                rows.push(CanopyCellUpdate(
                    (index % self.nx) as u32,
                    (index / self.nx) as u32,
                    now,
                ));
            }
        }
        rows
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
            safety_top: Arc::clone(&views.safety),
            wires: Arc::clone(
                self.wires_view
                    .get_or_insert_with(|| Arc::new(self.wires.values().cloned().collect())),
            ),
            quality: None,
        };
        self.views = Some(views);
        snapshot
    }
}

/// One cell's worth under `weights`.
fn deficit_weight(
    reason: DeficitReason,
    weights: &argusnet_planning::views::DeficitWeights,
) -> f64 {
    match reason {
        DeficitReason::Unseen => weights.unseen,
        DeficitReason::TooOblique => weights.too_oblique,
        DeficitReason::Sparse => weights.sparse,
        DeficitReason::Stale => weights.stale,
        DeficitReason::Met | DeficitReason::BiasLimited => 0.0,
    }
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
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

/// One cell's canopy top as the replay shows it (R2): `(i, j, top_cm)`.
/// Each row replaces the cell's last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanopyCellUpdate(pub u32, pub u32, pub i32);

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
    /// The navigation camera's obstacles: NaN where the camera does not
    /// believe the cell occupied; empty without camera evidence.
    safety_top: Arc<Vec<f64>>,
    /// The power-line spans mapped so far (R3), in id order.
    wires: Arc<Vec<PowerLine>>,
    /// Each cell's deficit against the run's quality requirement (SR5a);
    /// `None` without one.
    quality: Option<Arc<QualityLayer>>,
}

/// Every cell's deficit reason, and how many give each.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityLayer {
    pub reasons: Vec<DeficitReason>,
    pub counts: DeficitCounts,
}

impl QualityLayer {
    /// The scan's progress: the share of cells with nothing left for flying
    /// to do, because they meet the requirement or no flying can make them
    /// (bias-limited, unreachable). An unreachable cell never holds the scan
    /// open, and is never counted as met: [`DeficitCounts::met_fraction`]
    /// and the unreachable count are reported beside this.
    pub fn resolved_fraction(&self) -> f64 {
        (self.counts.met + self.counts.unreachable()) as f64 / self.counts.total().max(1) as f64
    }
}

impl BeliefSnapshot {
    /// This snapshot with `layer` as its quality.
    pub fn with_quality(mut self, layer: QualityLayer) -> Self {
        self.quality = Some(Arc::new(layer));
        self
    }

    pub fn quality(&self) -> Option<&QualityLayer> {
        self.quality.as_deref()
    }

    /// The deficit reason of the cell at `(x, y)`, with a requirement.
    /// `None` outside the grid: nothing there can be mapped, so nothing
    /// there is worth looking at.
    pub fn deficit_at(&self, x_m: f64, y_m: f64) -> Option<DeficitReason> {
        let fx = (x_m - self.x_min_m) / self.resolution_m;
        let fy = (y_m - self.y_min_m) / self.resolution_m;
        if !(fx >= 0.0 && fy >= 0.0 && fx < self.nx as f64 && fy < self.ny as f64) {
            return None;
        }
        let (i, j) = self.ij(x_m, y_m);
        self.quality
            .as_ref()
            .map(|layer| layer.reasons[j * self.nx + i])
    }

    /// The scan's progress: the resolved share with a quality requirement
    /// ([`QualityLayer::resolved_fraction`]), the known fraction without
    /// one.
    pub fn scan_progress(&self) -> f64 {
        match &self.quality {
            Some(layer) => layer.resolved_fraction(),
            None => self.known_fraction(),
        }
    }

    /// Whether the block holding `(x, y)` still needs flying, at least
    /// `share` of it: unmeasured without a requirement; improvable (unseen,
    /// too oblique, sparse or stale) with one. An explorer keeps its goal
    /// while it does.
    pub fn needs_work(&self, x_m: f64, y_m: f64, share: f64) -> bool {
        let (i, j) = self.ij(x_m, y_m);
        self.block_work_share(i / BLOCK_CELLS, j / BLOCK_CELLS) >= share
    }

    /// The centre of every block that needs flying ([`Self::needs_work`]),
    /// in block order: where an explorer can go to improve the map.
    pub fn work_centres(&self, share: f64) -> Vec<[f64; 2]> {
        if self.quality.is_none() {
            return self.unexplored_centres(share);
        }
        self.centres_where(|bi, bj| self.block_work_share(bi, bj) >= share)
    }

    fn block_work_share(&self, bi: usize, bj: usize) -> f64 {
        let Some(layer) = &self.quality else {
            return self.block_unknown_share(bi, bj);
        };
        let (i0, j0) = (bi * BLOCK_CELLS, bj * BLOCK_CELLS);
        let (i1, j1) = (
            (i0 + BLOCK_CELLS).min(self.nx),
            (j0 + BLOCK_CELLS).min(self.ny),
        );
        let total = (i1 - i0) * (j1 - j0);
        let improvable = (j0..j1)
            .flat_map(|j| (i0..i1).map(move |i| (i, j)))
            .filter(|&(i, j)| layer.reasons[j * self.nx + i].is_improvable())
            .count();
        improvable as f64 / total.max(1) as f64
    }

    /// How much fixing the cell at `(x, y)` is worth under `weights`: its
    /// deficit's weight, 0 when met, unreachable, or without a requirement.
    pub fn deficit_weight_at(
        &self,
        x_m: f64,
        y_m: f64,
        weights: &argusnet_planning::views::DeficitWeights,
    ) -> f64 {
        self.deficit_at(x_m, y_m)
            .map_or(0.0, |reason| deficit_weight(reason, weights))
    }

    /// Every block that needs work ([`Self::work_centres`]), with the
    /// weighted deficit flying over it would fix (SR5d). Empty without a
    /// requirement.
    pub fn work_blocks(
        &self,
        share: f64,
        weights: &argusnet_planning::views::DeficitWeights,
    ) -> Vec<argusnet_planning::views::WorkBlock> {
        let Some(layer) = &self.quality else {
            return Vec::new();
        };
        let bny = self.ny.div_ceil(BLOCK_CELLS);
        let mut blocks = Vec::new();
        for bj in 0..bny {
            for bi in 0..self.bnx {
                if self.block_work_share(bi, bj) < share {
                    continue;
                }
                let (i0, j0) = (bi * BLOCK_CELLS, bj * BLOCK_CELLS);
                let (i1, j1) = (
                    (i0 + BLOCK_CELLS).min(self.nx),
                    (j0 + BLOCK_CELLS).min(self.ny),
                );
                let gain: f64 = (j0..j1)
                    .flat_map(|j| (i0..i1).map(move |i| (i, j)))
                    .map(|(i, j)| deficit_weight(layer.reasons[j * self.nx + i], weights))
                    .sum();
                blocks.push(argusnet_planning::views::WorkBlock {
                    centre_m: self.block_centre(bi, bj),
                    gain,
                });
            }
        }
        blocks
    }

    /// The power-line spans mapped when this snapshot was taken (R3): the
    /// wires a run without a prior map plans round.
    pub fn wires(&self) -> &Arc<Vec<PowerLine>> {
        &self.wires
    }

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

    pub fn resolution_m(&self) -> f64 {
        self.resolution_m
    }

    /// Whether cell `(i, j)` has been measured: ground or structure.
    fn known(&self, i: usize, j: usize) -> bool {
        let c = j * self.nx + i;
        self.ground[c].is_finite() || self.structure_top[c].is_finite()
    }

    /// The share of the block holding cell `(bi, bj)` that is unmeasured.
    fn block_unknown_share(&self, bi: usize, bj: usize) -> f64 {
        let (i0, j0) = (bi * BLOCK_CELLS, bj * BLOCK_CELLS);
        let (i1, j1) = (
            (i0 + BLOCK_CELLS).min(self.nx),
            (j0 + BLOCK_CELLS).min(self.ny),
        );
        let total = (i1 - i0) * (j1 - j0);
        let unknown = (j0..j1)
            .flat_map(|j| (i0..i1).map(move |i| (i, j)))
            .filter(|&(i, j)| !self.known(i, j))
            .count();
        unknown as f64 / total.max(1) as f64
    }

    /// Whether the block holding `(x, y)` is still mostly unmeasured (at
    /// least `share` of it). An explorer keeps its goal while it is.
    pub fn is_unexplored(&self, x_m: f64, y_m: f64, share: f64) -> bool {
        let (i, j) = self.ij(x_m, y_m);
        self.block_unknown_share(i / BLOCK_CELLS, j / BLOCK_CELLS) >= share
    }

    /// The centre of every block at least `share` unmeasured, in block order:
    /// where an explorer can go to map something new.
    pub fn unexplored_centres(&self, share: f64) -> Vec<[f64; 2]> {
        self.centres_where(|bi, bj| self.block_unknown_share(bi, bj) >= share)
    }

    /// The centre of every block `pick` accepts, in block order.
    fn centres_where(&self, pick: impl Fn(usize, usize) -> bool) -> Vec<[f64; 2]> {
        let bny = self.ny.div_ceil(BLOCK_CELLS);
        let mut out = Vec::new();
        for bj in 0..bny {
            for bi in 0..self.bnx {
                if pick(bi, bj) {
                    out.push(self.block_centre(bi, bj));
                }
            }
        }
        out
    }

    /// Block `(bi, bj)`'s centre, held inside the grid's last cell.
    fn block_centre(&self, bi: usize, bj: usize) -> [f64; 2] {
        let block_m = BLOCK_CELLS as f64 * self.resolution_m;
        let cx = self.x_min_m + (bi as f64 + 0.5) * block_m;
        let cy = self.y_min_m + (bj as f64 + 0.5) * block_m;
        let x_max = self.x_min_m + self.nx as f64 * self.resolution_m;
        let y_max = self.y_min_m + self.ny as f64 * self.resolution_m;
        [
            cx.min(x_max - 0.5 * self.resolution_m),
            cy.min(y_max - 0.5 * self.resolution_m),
        ]
    }

    /// The share of the grid's cells that have been measured.
    pub fn known_fraction(&self) -> f64 {
        let known = (0..self.ny)
            .flat_map(|j| (0..self.nx).map(move |i| (i, j)))
            .filter(|&(i, j)| self.known(i, j))
            .count();
        known as f64 / (self.nx * self.ny).max(1) as f64
    }

    /// The share of the grid in *unexplored blocks that are enclosed*: blocks
    /// at least `share` unmeasured that are not connected (4-neighbour)
    /// through such blocks to the grid's edge. These are the holes inside the
    /// map that an explorer would go and fill, the belief's equivalent of the
    /// footprint grid's enclosed gaps. Single unmeasured cells, which the
    /// scanner's ring gaps leave scattered through measured ground, are not
    /// holes: counted cell by cell they hold a scan open forever.
    pub fn enclosed_unexplored_fraction(&self, share: f64) -> f64 {
        let (bnx, bny) = (self.bnx, self.ny.div_ceil(BLOCK_CELLS));
        let unexplored: Vec<bool> = (0..bny)
            .flat_map(|bj| (0..bnx).map(move |bi| (bi, bj)))
            .map(|(bi, bj)| self.block_unknown_share(bi, bj) >= share)
            .collect();
        let mut open = vec![false; bnx * bny];
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for bi in 0..bnx {
            stack.push((bi, 0));
            stack.push((bi, bny - 1));
        }
        for bj in 0..bny {
            stack.push((0, bj));
            stack.push((bnx - 1, bj));
        }
        while let Some((bi, bj)) = stack.pop() {
            let b = bj * bnx + bi;
            if open[b] || !unexplored[b] {
                continue;
            }
            open[b] = true;
            if bi > 0 {
                stack.push((bi - 1, bj));
            }
            if bi + 1 < bnx {
                stack.push((bi + 1, bj));
            }
            if bj > 0 {
                stack.push((bi, bj - 1));
            }
            if bj + 1 < bny {
                stack.push((bi, bj + 1));
            }
        }
        let enclosed = (0..bnx * bny)
            .filter(|&b| unexplored[b] && !open[b])
            .count();
        (enclosed * BLOCK_CELLS * BLOCK_CELLS) as f64 / (self.nx * self.ny).max(1) as f64
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

    /// The camera's obstacle top in the cell at `(x, y)`, if the
    /// navigation camera believes it occupied.
    pub fn safety_top_at(&self, x_m: f64, y_m: f64) -> Option<f64> {
        let (i, j) = self.ij(x_m, y_m);
        let top = *self.safety_top.get(j * self.nx + i)?;
        top.is_finite().then_some(top)
    }

    /// Whether `point_m` is inside a believed obstacle: a mapped structure,
    /// or one the navigation camera detected.
    fn collides(&self, point_m: [f64; 3]) -> bool {
        let below = |top: f64| point_m[2] <= top + STRUCTURE_MARGIN_M;
        self.structure_top_at(point_m[0], point_m[1])
            .is_some_and(below)
            || self
                .safety_top_at(point_m[0], point_m[1])
                .is_some_and(below)
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

/// How far short of a wire's envelope [`PhysicalGuard::enforce_from`] stops
/// a flight that would enter it, metres.
const WIRE_STOP_SHORT_M: f64 = 0.01;

/// Every `WorldContact::kind` the guard writes: out of the ground, out of a
/// structure, and still inside one after the push limit. The replay schema
/// lists the same (`tests/replay_vocabulary.rs`).
pub const WORLD_CONTACT_KINDS: [&str; 3] = ["terrain", "structure", "unresolved"];

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
    /// Each power line's conductors, out to the clearance from their
    /// surface (R3). Empty in a world without lines.
    wires: Arc<Vec<ConductorEnvelope>>,
    contacts: Arc<Mutex<Vec<WorldContact>>>,
}

impl std::fmt::Debug for PhysicalGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PhysicalGuard")
            .field("clearance_m", &self.clearance_m)
            .field("wires", &self.wires.len())
            .finish()
    }
}

impl PhysicalGuard {
    pub fn new(terrain: SharedTerrain, obstacles: SharedObstacles, clearance_m: f64) -> Self {
        Self {
            terrain,
            obstacles,
            clearance_m,
            wires: Arc::new(Vec::new()),
            contacts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The guard, also keeping platforms out of `lines`' conductors: the
    /// guard's clearance from each wire's surface, the wire's own radius
    /// added (R3).
    pub fn with_power_lines(mut self, lines: &[PowerLine]) -> Self {
        self.wires = Arc::new(
            lines
                .iter()
                .map(|line| line.envelope(self.clearance_m))
                .collect(),
        );
        self
    }

    /// How many power lines' wires the guard keeps platforms off.
    pub fn power_line_count(&self) -> usize {
        self.wires.len()
    }

    /// `position` moved out of the ground, out of any structure and away
    /// from any wire, with a contact recorded for each move.
    pub fn enforce(&self, node_id: &str, timestamp_s: f64, position: [f64; 3]) -> [f64; 3] {
        let mut p = position;
        // A push out of one structure can land in the next: push again, up
        // to eight times, as the reference's `push_outside_obstacles` does.
        // The height is clamped to the terrain after every push, and the
        // structures checked again at that height (a lift, after a push
        // uphill or on its own, can put it into a building whose base was
        // above it).
        for _ in 0..8 {
            let pushed = self.obstacles.push_clear(p, self.clearance_m);
            if let Some(xy) = pushed {
                let correction = (xy[0] - p[0]).hypot(xy[1] - p[1]);
                p[0] = xy[0];
                p[1] = xy[1];
                self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[1], correction);
            }
            let wired = self.out_of_wire(p);
            if let Some(out) = wired {
                let correction =
                    ((out[0] - p[0]).powi(2) + (out[1] - p[1]).powi(2) + (out[2] - p[2]).powi(2))
                        .sqrt();
                p = out;
                self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[1], correction);
            }
            let floor = self.floor_at(p);
            let raised = p[2] < floor;
            if raised {
                self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[0], floor - p[2]);
                p[2] = floor;
            }
            // Settled only when nothing moved it: a lift alone can put it
            // into a structure whose base was above it, or into a wire.
            if pushed.is_none() && wired.is_none() && !raised {
                break;
            }
        }
        // Overlapping structures can bounce it between them past the limit;
        // that is recorded, not passed off as a clear position.
        if self.obstacles.collides(p) || self.wires.iter().any(|wire| wire.contains(p)) {
            self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[2], 0.0);
        }
        p
    }

    /// [`Self::enforce`] for a platform that flew from `from`: the flight is
    /// swept against the wires first, since a wire is thin enough to pass
    /// between two positions each clear of it. A flight that comes within
    /// the guard's clearance of one is stopped just short of where it first
    /// does (solved per piece, [`ConductorEnvelope::first_entry`]), and the
    /// stop recorded; then the stop is enforced. One that starts inside an
    /// envelope is left to `enforce` to push out. With no wires this is
    /// `enforce`.
    pub fn enforce_from(
        &self,
        node_id: &str,
        timestamp_s: f64,
        from: [f64; 3],
        position: [f64; 3],
    ) -> [f64; 3] {
        let length = ((position[0] - from[0]).powi(2)
            + (position[1] - from[1]).powi(2)
            + (position[2] - from[2]).powi(2))
        .sqrt();
        let entry = self
            .wires
            .iter()
            .filter_map(|wire| wire.first_entry(from, position))
            .min_by(f64::total_cmp);
        let stop = match entry {
            Some(t) if t > 0.0 && length > 0.0 => {
                let f = ((t * length - WIRE_STOP_SHORT_M) / length).max(0.0);
                let stop = [
                    from[0] + (position[0] - from[0]) * f,
                    from[1] + (position[1] - from[1]) * f,
                    from[2] + (position[2] - from[2]) * f,
                ];
                let correction = ((position[0] - stop[0]).powi(2)
                    + (position[1] - stop[1]).powi(2)
                    + (position[2] - stop[2]).powi(2))
                .sqrt();
                self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[1], correction);
                stop
            }
            _ => position,
        };
        self.enforce(node_id, timestamp_s, stop)
    }

    /// The lowest a platform may be at `p`'s place: the guard's clearance
    /// over the ground.
    /// Whether `p` is somewhere a platform cannot be: below the true ground,
    /// inside a structure or inside a wire. Within the clearance above the
    /// ground is not: a platform lands there.
    pub fn violates(&self, p: [f64; 3]) -> bool {
        p[2] < TerrainSampler::height_at(&self.terrain, p[0], p[1])
            || self.obstacles.collides(p)
            || self.wires.iter().any(|wire| wire.contains(p))
    }

    fn floor_at(&self, p: [f64; 3]) -> f64 {
        TerrainSampler::height_at(&self.terrain, p[0], p[1]) + self.clearance_m
    }

    /// `p` out of the first wire's envelope it is in: pushed straight away
    /// from the wire, or, when that would put it under the floor, lifted
    /// straight up over the wire instead. Physics never pushes a platform
    /// into the ground. `None` when it is in no wire's envelope. A push can
    /// land in the next piece where two meet at an angle, or in another
    /// line; the caller pushes again.
    fn out_of_wire(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        let wire = self.wires.iter().find(|wire| wire.contains(p))?;
        let pushed = wire.push_out(p)?;
        if pushed[2] >= self.floor_at(pushed) {
            Some(pushed)
        } else {
            wire.lift_out(p)
        }
    }

    /// Where the segment from `from` to `to` first meets a building or a
    /// wall, or comes within the guard's clearance of a wire, as a fraction
    /// of its length: intersected with each structure near it, so one
    /// thinner than any sampling step is still met, and solved against each
    /// wire piece's capsule ([`ConductorEnvelope::first_entry`]), counting
    /// how far the wire can sag below the piece. `None` when it meets none.
    pub fn first_structure_hit(&self, from: [f64; 3], to: [f64; 3]) -> Option<f64> {
        let layer = self.obstacles.layer();
        let around = argusnet_types::Bounds2d::new(
            from[0].min(to[0]),
            from[0].max(to[0]),
            from[1].min(to[1]),
            from[1].max(to[1]),
        )
        .expanded(1e-6);
        layer
            .query(around)
            .into_iter()
            .filter(|obstacle| obstacle.blocker_type.is_hard())
            .filter_map(|obstacle| obstacle.surface_hit_t(from, to))
            .chain(
                self.wires
                    .iter()
                    .filter_map(|wire| wire.first_entry(from, to)),
            )
            .min_by(f64::total_cmp)
    }

    /// Where the flight from `from` to `to` first comes within the guard's
    /// clearance of the ground, as a fraction of it: solved on the terrain's
    /// cells ([`crate::ground::first_within`]), not sampled. `None` when it
    /// never does.
    pub fn first_ground_contact(&self, from: [f64; 3], to: [f64; 3]) -> Option<f64> {
        crate::ground::first_within(&self.terrain, from, to, self.clearance_m)
    }

    /// Record that the ground stopped `node_id` `correction_m` short of
    /// where it was going, as [`Self::enforce`] records a lift.
    pub fn record_ground_stop(&self, node_id: &str, timestamp_s: f64, correction_m: f64) {
        self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[0], correction_m);
    }

    /// Record that a structure stopped `node_id` `correction_m` short of
    /// where it was going, as [`Self::enforce`] records a push out of one.
    pub fn record_structure_stop(&self, node_id: &str, timestamp_s: f64, correction_m: f64) {
        self.record(node_id, timestamp_s, WORLD_CONTACT_KINDS[1], correction_m);
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
            wires: Vec::new(),
            sensor: Default::default(),
            pod: None,
        }
    }

    fn ground(x: f64, y: f64, z: f64) -> RangeReturn {
        RangeReturn {
            point_m: [x, y, z],
            class: ReturnClass::Ground,
        }
    }

    #[test]
    fn unexplored_blocks_known_share_and_enclosed_holes() {
        let config = ExplorationConfig::default();
        let mut belief = grid();
        // Measure the 20 x 20 grid except a 3 x 3 hole at cells (8..11, 8..11)
        // and the whole top-right 5 x 5 block (cells 15..20, 15..20).
        let mut returns = Vec::new();
        for j in 0..20 {
            for i in 0..20 {
                let hole = (8..11).contains(&i) && (8..11).contains(&j);
                let corner = i >= 15 && j >= 15;
                if !hole && !corner {
                    returns.push(ground(i as f64 * 10.0 + 5.0, j as f64 * 10.0 + 5.0, 50.0));
                }
            }
        }
        belief.integrate(&scan(returns), 1);
        let snapshot = belief.snapshot(&config);
        assert!((snapshot.known_fraction() - (400.0 - 9.0 - 25.0) / 400.0).abs() < 1e-12);
        // The 3 x 3 hole spans four blocks and fills none of them, so it is
        // speckle, not a hole; the corner block touches the edge.
        assert_eq!(snapshot.enclosed_unexplored_fraction(0.5), 0.0);
        // Only the corner block is mostly unmeasured (the hole spans blocks).
        let centres = snapshot.unexplored_centres(0.5);
        assert_eq!(centres, vec![[175.0, 175.0]]);
        assert!(snapshot.is_unexplored(171.0, 199.0, 0.5));
        assert!(!snapshot.is_unexplored(95.0, 95.0, 0.5));
    }

    #[test]
    fn a_whole_unmeasured_block_inside_the_map_is_a_hole() {
        let mut belief = grid();
        // Everything but block (1, 1): cells 5..10 x 5..10.
        let returns = (0..20)
            .flat_map(|j| (0..20).map(move |i| (i, j)))
            .filter(|&(i, j)| !((5..10).contains(&i) && (5..10).contains(&j)))
            .map(|(i, j)| ground(i as f64 * 10.0 + 5.0, j as f64 * 10.0 + 5.0, 50.0))
            .collect();
        belief.integrate(&scan(returns), 1);
        let snapshot = belief.snapshot(&ExplorationConfig::default());
        assert!((snapshot.enclosed_unexplored_fraction(0.5) - 25.0 / 400.0).abs() < 1e-12);
        assert_eq!(snapshot.unexplored_centres(0.5), vec![[75.0, 75.0]]);
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

    /// A navigation camera frame is safety evidence: two detections make
    /// the cell an obstacle the planners avoid, but the cell is not
    /// measured, so the frontier, the known fraction and the mapped
    /// structures are untouched; the evidence survives a checkpoint.
    #[test]
    fn camera_detections_are_obstacles_but_not_map() {
        let config = ExplorationConfig::default();
        let mut belief = grid();
        let snapshot = belief.snapshot(&config);
        let handle = BeliefHandle::new(snapshot);
        let mut frame = scan(vec![RangeReturn {
            point_m: [45.0, 45.0, 80.0],
            class: ReturnClass::Structure,
        }]);
        frame.sensor = ScanSensor::NavCamera;
        belief.integrate(&frame, 1);
        handle.publish(belief.snapshot(&config));
        let obstacles = BeliefObstacles::new(handle.clone());
        assert!(
            !obstacles.collides([45.0, 45.0, 79.0]),
            "one detection is not yet an obstacle"
        );
        belief.integrate(&frame, 2);
        handle.publish(belief.snapshot(&config));
        assert!(obstacles.collides([45.0, 45.0, 79.0]));
        assert!(!obstacles.collides([45.0, 45.0, 90.0]), "above its top");
        assert_eq!(belief.safety_cells_at(config.occupied_probability), 1);
        assert_eq!(belief.known_cells(), 0, "never measured");
        assert_eq!(belief.structure_cells_at(config.occupied_probability), 0);
        assert!(belief
            .take_changed_cells(config.occupied_probability)
            .is_empty());
        assert_eq!(handle.current().known_fraction(), 0.0);
        let text = serde_json::to_string(&belief).unwrap();
        let restored: BeliefGrid = serde_json::from_str(&text).unwrap();
        assert_eq!(restored.safety_cells_at(config.occupied_probability), 1);
        // A grid without camera evidence serialises as it always did.
        assert!(!serde_json::to_string(&grid()).unwrap().contains("safety"));

        // At a 0.5 threshold an untouched cell's zero log-odds passes the
        // probability test, but it has no evidence: it never becomes an
        // obstacle, before a checkpoint or after.
        let even = ExplorationConfig {
            occupied_probability: 0.5,
            ..ExplorationConfig::default()
        };
        let mut belief = grid();
        belief.integrate(&frame, 1);
        let before = belief.snapshot(&even);
        assert!(before.safety_top_at(45.0, 45.0).is_some());
        assert_eq!(before.safety_top_at(155.0, 155.0), None);
        let mut restored: BeliefGrid =
            serde_json::from_str(&serde_json::to_string(&belief).unwrap()).unwrap();
        let after = restored.snapshot(&even);
        assert_eq!(after.safety_top_at(155.0, 155.0), None);
        assert!(!after.collides([155.0, 155.0, 0.5]));
        assert_eq!(restored.safety_cells_at(0.5), 1);
    }

    /// SR5b: a block measured only by shallow rays is known, so it is no
    /// longer unexplored, but every cell is too oblique, so it still needs
    /// work and stays a goal. Once steep returns meet it, it drops out.
    #[test]
    fn a_known_block_that_falls_short_is_still_a_goal() {
        use argusnet_mapping::quality::{DeficitReason, QualityRequirement};
        let requirement = QualityRequirement {
            min_ground_density_per_m2: Some(0.02),
            max_vertical_sigma_m: None,
            max_age_s: None,
            max_incidence_deg: 30.0,
            bias_floor_m: 0.0,
        };
        let config = ExplorationConfig::default();
        let mut belief = grid();
        belief.track_quality(requirement.max_incidence_deg);
        // Every cell of the first block, from a platform far off to the
        // side: 60+ degrees from nadir.
        let shallow: Vec<RangeReturn> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i, j)))
            .map(|(i, j)| ground(5.0 + 10.0 * i as f64, 5.0 + 10.0 * j as f64, 0.0))
            .collect();
        let mut far = scan(shallow.clone());
        far.origin_m = [400.0, 25.0, 100.0];
        belief.integrate(&far, 1);
        let snapshot =
            belief
                .snapshot(&config)
                .with_quality(belief.quality(&requirement, 0.3, 1, 1.0));
        assert!(!snapshot.is_unexplored(25.0, 25.0, 0.5), "measured");
        assert!(snapshot.needs_work(25.0, 25.0, 0.5), "but not good enough");
        assert_eq!(
            snapshot.deficit_at(25.0, 25.0),
            Some(DeficitReason::TooOblique)
        );
        assert!(snapshot.work_centres(0.5).contains(&[25.0, 25.0]));
        assert!(!snapshot.unexplored_centres(0.5).contains(&[25.0, 25.0]));
        // Straight down, twice: two accepted returns per 100 m² cell.
        let mut over = scan(shallow);
        over.origin_m = [25.0, 25.0, 100.0];
        belief.integrate(&over, 2);
        belief.integrate(&over, 3);
        let snapshot =
            belief
                .snapshot(&config)
                .with_quality(belief.quality(&requirement, 0.3, 3, 1.0));
        assert_eq!(snapshot.deficit_at(25.0, 25.0), Some(DeficitReason::Met));
        // Outside the grid nothing can be mapped: no deficit, no worth.
        assert_eq!(snapshot.deficit_at(-5.0, 25.0), None);
        assert_eq!(snapshot.deficit_at(25.0, 200.0), None);
        let weights = argusnet_planning::views::DeficitWeights::default();
        assert_eq!(snapshot.deficit_weight_at(250.0, 25.0, &weights), 0.0);
        assert!(!snapshot.needs_work(25.0, 25.0, 0.5));
        assert!(!snapshot.work_centres(0.5).contains(&[25.0, 25.0]));
        // Without a requirement, work is the unexplored share.
        let plain = belief.snapshot(&config);
        assert_eq!(plain.work_centres(0.5), plain.unexplored_centres(0.5));
    }

    /// Canopy returns are soft structure: the cell is seen and keeps its
    /// highest crown, shown to the replay when first seen or moved 10 cm,
    /// but it is never occupied, so nothing collides with it. A grid that
    /// never saw a crown checkpoints without a canopy key.
    #[test]
    fn canopy_returns_are_seen_but_never_occupied() {
        let config = ExplorationConfig::default();
        let mut belief = grid();
        let plain = serde_json::to_string(&belief).unwrap();
        assert!(!plain.contains("canopy"), "{plain}");
        let crown = |z: f64| RangeReturn {
            point_m: [45.0, 45.0, z],
            class: ReturnClass::Canopy,
        };
        belief.integrate(&scan(vec![crown(70.0), crown(72.0), crown(71.0)]), 0);
        assert_eq!(belief.canopy_cells(), 1);
        assert_eq!(belief.canopy_top_ij(4, 4), Some(72.0));
        assert_eq!(belief.canopy_top_ij(5, 4), None);
        assert_eq!(belief.known_cells(), 1);
        assert!(!belief.is_structure_ij(4, 4, config.occupied_probability));
        assert_eq!(
            belief.take_changed_canopy(),
            vec![CanopyCellUpdate(4, 4, 7200)]
        );
        assert!(belief.take_changed_canopy().is_empty());
        // The belief row says only that the cell is seen and not occupied:
        // no ground, no structure.
        assert_eq!(
            belief.take_changed_cells(config.occupied_probability),
            vec![BeliefCellUpdate(4, 4, None, None, 0, 0)]
        );
        // A lower return, or one 5 cm higher, changes nothing shown; 10 cm does.
        belief.integrate(&scan(vec![crown(60.0), crown(72.05)]), 1);
        assert!(belief.take_changed_canopy().is_empty());
        belief.integrate(&scan(vec![crown(72.15)]), 2);
        assert_eq!(
            belief.take_changed_canopy(),
            vec![CanopyCellUpdate(4, 4, 7215)]
        );
        let handle = BeliefHandle::new(belief.snapshot(&config));
        assert!(!BeliefObstacles::new(handle).collides([45.0, 45.0, 65.0]));
        // Through a checkpoint, and gone with a clear.
        let back: BeliefGrid =
            serde_json::from_str(&serde_json::to_string(&belief).unwrap()).unwrap();
        assert_eq!(back.canopy_top_ij(4, 4), Some(72.15));
        belief.clear();
        assert_eq!(belief.canopy_cells(), 0);
        assert_eq!(serde_json::to_string(&belief).unwrap(), plain);
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
    /// R3: a span is mapped the first time a scan puts points on it, and
    /// published with the snapshot; a reset forgets it, but not the spans
    /// known beforehand; a grid without wires checkpoints as it always did.
    #[test]
    fn wires_are_mapped_as_scans_see_them() {
        let config = ExplorationConfig::default();
        let span = |k: usize| {
            let mut line = wire(20.0);
            line.supports.push(argusnet_world::powerline::Support {
                xy_m: [200.0, 0.0],
                base_z_m: 0.0,
                height_m: 20.0,
            });
            line.span_line(k)
        };
        let mut belief = grid();
        let plain = serde_json::to_value(&belief).unwrap();
        assert!(plain.get("wires").is_none() && plain.get("prior_wires").is_none());
        assert!(belief.snapshot(&config).wires().is_empty());

        let mut seen = scan(Vec::new());
        seen.wires = vec![crate::range_scan::WireReturn {
            span: span(1),
            point_m: [150.0, 0.0, 20.0],
        }];
        belief.integrate(&seen, 3);
        belief.integrate(&seen, 4);
        let first = belief.snapshot(&config);
        assert_eq!(first.wires().as_slice(), [span(1)]);
        // Unchanged, the snapshots share one list.
        assert!(Arc::ptr_eq(first.wires(), belief.snapshot(&config).wires()));
        // Round trip with the wire.
        let back: BeliefGrid =
            serde_json::from_str(&serde_json::to_string(&belief).unwrap()).unwrap();
        assert_eq!(back.wires().cloned().collect::<Vec<_>>(), [span(1)]);

        belief.clear();
        assert!(belief.snapshot(&config).wires().is_empty());
        belief.know_wires(vec![span(0)]);
        belief.integrate(&seen, 5);
        assert_eq!(
            belief.snapshot(&config).wires().as_slice(),
            [span(0), span(1)]
        );
        belief.clear();
        assert_eq!(belief.snapshot(&config).wires().as_slice(), [span(0)]);
    }

    /// One wire along x from 0 to 100, `height_m` up on flat ground at 0,
    /// with no sag: 2 cm radius, 10 m clearance.
    fn wire(height_m: f64) -> PowerLine {
        use argusnet_world::powerline::Support;
        let support = |x: f64| Support {
            xy_m: [x, 0.0],
            base_z_m: 0.0,
            height_m,
        };
        PowerLine {
            id: "wire".into(),
            supports: vec![support(0.0), support(100.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.0,
            conductor_radius_m: 0.02,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap()
    }

    /// A guard with a 1 m clearance over flat ground at 0, with `building`
    /// if any, and `lines`.
    fn guard_over(
        building: Option<argusnet_world::obstacles::Obstacle>,
        lines: &[PowerLine],
    ) -> PhysicalGuard {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let bounds = Bounds2d::new(-200.0, 200.0, -200.0, 200.0);
        let layer = ObstacleLayer::new(bounds, 400.0, building.into_iter().collect()).unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer)),
            1.0,
        )
        .with_power_lines(lines)
    }

    /// A refused pod capture is refused whole: its wires do not reach the
    /// map either, while a settled mission capture's do.
    #[test]
    fn a_refused_pod_capture_maps_no_wires() {
        use crate::pod::{PodCapture, PointingSource};
        let capture = |source, settled| {
            let mut s = scan(Vec::new());
            s.wires = vec![crate::range_scan::WireReturn {
                span: wire(20.0).span_line(0),
                point_m: [50.0, 0.0, 20.0],
            }];
            s.pod = Some(PodCapture {
                source,
                settled,
                command_error_rad: 0.0,
                boresight_world: [0.0, 0.0, -1.0],
                occluded: 0,
            });
            s
        };
        let mut belief = grid();
        belief.integrate(&capture(PointingSource::Stow, true), 1);
        belief.integrate(&capture(PointingSource::Mission, false), 2);
        assert_eq!(belief.wires().count(), 0);
        assert_eq!((belief.stowed_scans(), belief.unsettled_scans()), (1, 1));
        belief.integrate(&capture(PointingSource::Mission, true), 3);
        assert_eq!(belief.wires().count(), 1);
    }

    /// Somewhere a platform cannot be: below the true ground, inside a
    /// structure or a wire. Within the clearance above the ground is not: a
    /// platform lands there.
    #[test]
    fn the_guard_says_where_a_platform_cannot_be() {
        let guard = guard_over(None, &[wire(20.0)]);
        assert!(guard.violates([10.0, 10.0, -0.1]), "below the ground");
        assert!(
            !guard.violates([10.0, 10.0, 0.5]),
            "landed, within the clearance"
        );
        assert!(guard.violates([50.0, 0.0, 20.0]), "on the wire");
        assert!(!guard.violates([50.0, 0.0, 30.0]));
    }

    /// A platform within the guard's clearance of a wire is pushed straight
    /// away from it, to the wire's radius plus the clearance, and the push
    /// is a recorded contact. Clear of it, nothing moves.
    #[test]
    fn the_guard_pushes_a_platform_off_a_wire() {
        let guard = guard_over(None, &[wire(20.0)]);
        let reach = 0.02 + 1.0;
        let settled = guard.enforce("d", 0.0, [50.0, 0.3, 19.6]);
        let gap = argusnet_world::powerline::segment_distance(
            settled,
            settled,
            [0.0, 0.0, 20.0],
            [100.0, 0.0, 20.0],
        );
        assert!((gap - reach).abs() < 1e-5, "{settled:?}: {gap}");
        // Straight away: the same direction from the wire as before.
        assert!(((settled[1]) / (20.0 - settled[2]) - 0.3 / 0.4).abs() < 1e-9);
        let contacts = guard.take_contacts();
        assert_eq!(contacts.len(), 1, "{contacts:?}");
        assert_eq!(contacts[0].kind, "structure");
        assert!((contacts[0].correction_m - (reach - 0.5)).abs() < 1e-5);
        // Just outside the clearance: left alone.
        let clear = [50.0, 0.0, 20.0 - reach - 1e-3];
        assert_eq!(guard.enforce("d", 0.0, clear), clear);
        assert!(guard.take_contacts().is_empty());
    }

    /// Where a sagging wire's pieces meet at an angle, a push out of one can
    /// land in the next; the guard pushes again until it is clear of both,
    /// and records nothing unresolved.
    #[test]
    fn the_guard_settles_a_platform_where_wire_pieces_meet() {
        let mut sagging = wire(20.0);
        sagging.sag_ratio = 0.05;
        let guard = guard_over(None, std::slice::from_ref(&sagging));
        let envelope = sagging.envelope(1.0);
        for piece in sagging.pieces().iter().step_by(4) {
            let joint = piece.end_m;
            for dz in [-0.9, -0.3, 0.3, 0.9] {
                let settled = guard.enforce("d", 0.0, [joint[0], 0.0, joint[2] + dz]);
                assert!(!envelope.contains(settled), "{joint:?} {dz}: {settled:?}");
            }
        }
        assert!(guard.take_contacts().iter().all(|c| c.kind != "unresolved"));
    }

    /// Two wires closer together than the guard's reach either side leave
    /// no clear place between them: pushed back and forth until the limit,
    /// it is recorded unresolved, not passed off as clear.
    #[test]
    fn the_guard_records_a_platform_it_cannot_free_from_two_wires() {
        let mut beside = wire(20.0);
        beside.id = "beside".into();
        for support in &mut beside.supports {
            support.xy_m[1] = 1.5;
        }
        let guard = guard_over(None, &[wire(20.0), beside]);
        guard.enforce("d", 0.0, [50.0, 0.75, 20.0]);
        let contacts = guard.take_contacts();
        assert!(
            contacts.iter().any(|c| c.kind == "unresolved"),
            "{contacts:?}"
        );
    }

    /// A flight across a wire between two positions each clear of it is
    /// stopped just short of where it first comes within the clearance, and
    /// the stop recorded; one that does not cross is `enforce` exactly.
    #[test]
    fn a_flight_through_a_wire_is_stopped_short_of_it() {
        let guard = guard_over(None, &[wire(20.0)]);
        let (from, to) = ([50.0, -10.0, 20.0], [50.0, 10.0, 20.0]);
        let stopped = guard.enforce_from("d", 0.0, from, to);
        assert!((stopped[1] - (-1.02 - 0.01)).abs() < 1e-9, "{stopped:?}");
        assert_eq!([stopped[0], stopped[2]], [50.0, 20.0]);
        let contacts = guard.take_contacts();
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].correction_m - (10.0 + 1.03)).abs() < 1e-9);
        // Over it: as `enforce`.
        let over = [50.0, 10.0, 25.0];
        assert_eq!(
            guard.enforce_from("d", 0.0, [50.0, -10.0, 25.0], over),
            over
        );
        assert!(guard.take_contacts().is_empty());
        // From inside: `enforce` pushes it out.
        let pushed = guard.enforce_from("d", 0.0, [50.0, 0.3, 20.0], [50.0, 0.3, 19.6]);
        assert_eq!(pushed, guard.enforce("e", 0.0, [50.0, 0.3, 19.6]));
        // Without wires, `enforce` whatever the flight.
        let bare = guard_over(None, &[]);
        assert_eq!(bare.enforce_from("d", 0.0, from, to), to);
    }

    /// Physics never pushes a platform into the ground: under a wire 1.5 m
    /// up, a push down would end under the 1 m floor, so it is lifted
    /// straight over the wire instead.
    #[test]
    fn the_guard_lifts_a_platform_over_a_low_wire_rather_than_into_the_ground() {
        let guard = guard_over(None, &[wire(1.5)]);
        let settled = guard.enforce("d", 0.0, [50.0, 0.0, 1.2]);
        assert_eq!([settled[0], settled[1]], [50.0, 0.0]);
        assert!((settled[2] - (1.5 + 1.02)).abs() < 1e-5, "{settled:?}");
        assert!(guard.take_contacts().iter().all(|c| c.kind == "structure"));
        // Off to the side it is pushed out sideways (and down, where that
        // keeps it over the floor); wherever it starts, it ends over the
        // floor and clear of the wire.
        let envelope = wire(1.5).envelope(1.0);
        for z in [1.0, 1.3, 1.49, 1.5, 1.6, 2.4] {
            let settled = guard.enforce("d", 0.0, [50.0, 0.4, z]);
            assert!(settled[2] >= 1.0, "{z}: {settled:?}");
            assert!(!envelope.contains(settled), "{z}: {settled:?}");
        }
    }

    /// A step toward a wire stops where it first comes within the guard's
    /// clearance of it, solved: a 20 m step across the wire is stopped
    /// 1.02 m short of it, though neither end is near it. The nearer of a
    /// building and a wire stops it.
    #[test]
    fn a_step_stops_at_a_wires_envelope_exactly() {
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        let guard = guard_over(None, &[wire(20.0)]);
        let (from, to) = ([50.0, -10.0, 20.0], [50.0, 10.0, 20.0]);
        let t = guard.first_structure_hit(from, to).expect("met");
        assert!((t - (10.0 - 1.02) / 20.0).abs() < 1e-12, "{t}");
        // Over it with room to spare: not met.
        assert_eq!(
            guard.first_structure_hit([50.0, -10.0, 21.1], [50.0, 10.0, 21.1]),
            None
        );
        let wall = Obstacle::polygon_prism(
            "wall",
            BlockerType::Building,
            vec![[49.0, -6.0], [51.0, -6.0], [51.0, -5.0], [49.0, -5.0]],
            0.0,
            40.0,
        )
        .unwrap();
        let guarded = guard_over(Some(wall), &[wire(20.0)]);
        let t = guarded.first_structure_hit(from, to).expect("met");
        assert!((t - 4.0 / 20.0).abs() < 1e-12, "{t}");
        // A guard without lines meets no wire.
        assert_eq!(guard_over(None, &[]).first_structure_hit(from, to), None);
    }
    /// A push out of one structure can land in the next; the guard pushes
    /// again until the point is clear (the reference's fixture).
    #[test]
    fn the_guard_pushes_out_of_a_neighbour_too() {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let building = |id: &str, x0: f64, x1: f64, y0: f64, y1: f64| {
            Obstacle::polygon_prism(
                id,
                BlockerType::Building,
                vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                0.0,
                100.0,
            )
            .unwrap()
        };
        let bounds = Bounds2d::new(-50.0, 50.0, -50.0, 50.0);
        let layer = ObstacleLayer::new(
            bounds,
            100.0,
            vec![
                building("a", 0.0, 10.0, 0.0, 10.0),
                building("b", 10.5, 30.0, 4.8, 5.2),
            ],
        )
        .unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer.clone())),
            1.0,
        );
        let settled = guard.enforce("d", 0.0, [9.8, 5.0, 10.0]);
        assert!(layer.point_collides(settled).is_none(), "{settled:?}");
        assert!(guard.take_contacts().len() >= 2, "pushed more than once");
    }
    /// The terrain clamp can lift a pushed platform into a neighbour whose
    /// base was above it; the guard checks again at the clamped height.
    #[test]
    fn the_guard_rechecks_structures_after_the_terrain_clamp() {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let building = |id: &str, x0: f64, x1: f64, y0: f64, y1: f64, base: f64| {
            Obstacle::polygon_prism(
                id,
                BlockerType::Building,
                vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                base,
                100.0,
            )
            .unwrap()
        };
        let bounds = Bounds2d::new(-200.0, 200.0, -200.0, 200.0);
        // `b` starts 20 m up: the platform at 10 m clears it, but the floor
        // (ground + 25 m clearance) puts it inside.
        let layer = ObstacleLayer::new(
            bounds,
            400.0,
            vec![
                building("a", 0.0, 10.0, 0.0, 10.0, 0.0),
                building("b", 10.5, 80.0, 4.8, 5.2, 20.0),
            ],
        )
        .unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer.clone())),
            25.0,
        );
        let settled = guard.enforce("d", 0.0, [9.8, 5.0, 10.0]);
        assert!(settled[2] >= 25.0, "{settled:?}");
        assert!(layer.point_collides(settled).is_none(), "{settled:?}");
    }
    /// A lift by the terrain clamp alone can put the platform into a
    /// structure whose base was above it; the guard checks again.
    #[test]
    fn the_guard_rechecks_structures_after_a_lift_alone() {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let bounds = Bounds2d::new(-200.0, 200.0, -200.0, 200.0);
        let raised = Obstacle::polygon_prism(
            "raised",
            BlockerType::Building,
            vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]],
            20.0,
            100.0,
        )
        .unwrap();
        let layer = ObstacleLayer::new(bounds, 400.0, vec![raised]).unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer.clone())),
            25.0,
        );
        // At 10 m it is under the structure; the 25 m floor lifts it in.
        let settled = guard.enforce("d", 0.0, [0.0, 0.0, 10.0]);
        assert!(settled[2] >= 25.0, "{settled:?}");
        assert!(layer.point_collides(settled).is_none(), "{settled:?}");
    }
    /// Overlapping structures can bounce a platform between them until the
    /// push limit runs out; the guard records that rather than hiding it.
    #[test]
    fn the_guard_records_a_collision_it_cannot_resolve() {
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let block = |id: &str, x0: f64, x1: f64| {
            Obstacle::polygon_prism(
                id,
                BlockerType::Building,
                vec![[x0, -50.0], [x1, -50.0], [x1, 50.0], [x0, 50.0]],
                0.0,
                100.0,
            )
            .unwrap()
        };
        let bounds = Bounds2d::new(-200.0, 200.0, -200.0, 200.0);
        let layer = ObstacleLayer::new(
            bounds,
            400.0,
            vec![block("a", 0.0, 10.0), block("b", 9.0, 19.0)],
        )
        .unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer.clone())),
            1.0,
        );
        let settled = guard.enforce("d", 0.0, [9.5, 0.0, 10.0]);
        let contacts = guard.take_contacts();
        // The fixture bounces it until the limit runs out, still inside.
        assert!(layer.point_collides(settled).is_some(), "{settled:?}");
        assert!(
            contacts.iter().any(|c| c.kind == "unresolved"),
            "{contacts:?}"
        );
    }
}
