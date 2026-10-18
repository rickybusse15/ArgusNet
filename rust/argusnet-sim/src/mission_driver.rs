//! Stage 9 in full: the mission machine, and the work its phases do.
//!
//! P4.8a separated *deciding* the phase ([`MissionState::advance`]) from the
//! side effects the reference performs inline -- rescoring POIs, assigning
//! them, redirecting drones, routing everyone home. The machine emits events
//! and leaves the work to "the caller". This is that caller.
//!
//! [`MissionState::advance`]: crate::mission::MissionState::advance
//!
//! # One tick, in the reference's order
//!
//! `run_simulation`'s `elif` chain tests the phase as it stood at the **top**
//! of the tick. In the inspecting branch it rescores (once per run), assigns,
//! redirects, hands off, credits dwell, team-assigns and completes, and only
//! *then* asks `all_complete` to decide on egress. So this stage:
//!
//! 1. reads the phase the machine settled on last tick;
//! 2. if that is `Inspecting`, does the whole inspecting body;
//! 3. builds the machine's inputs from **this** tick's stage 8 results and
//!    the POI manager as step 2 left it;
//! 4. advances the machine, and turns its egress events into overrides.
//!
//! The overrides are flown from the next tick's stage 2, which is where the
//! reference's `_poi_trajectories` / `_egress_trajectories` take effect.
//!
//! # The rescore is the driver's, not the event's
//!
//! The machine emits `RescorePois` on the tick it *enters* `Inspecting`. The
//! reference rescores at the top of the first tick *spent* inspecting --
//! after that tick's coverage update, which can change the density a POI
//! reads. So the driver keeps its own latch and rescores at step 2, and the
//! event is informational.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use argusnet_planning::allocation::{min_cost_assignment, PlanningPolicy};
use argusnet_planning::frontier::{find_gap_cells, CoverageView};
use argusnet_planning::planner::{PlannerConfig, PlannerRoute, RoutePlanner, Scene};
use argusnet_planning::poi::{
    InspectionPoi, PoiAssignmentContext, PoiManager, PoiState, PoiStatus,
};
use argusnet_planning::BatteryModel;
use argusnet_types::Bounds2d;
use argusnet_world::index::ObstacleLayer;

use crate::battery::{stage_one_battery, BatteryStage};
use crate::mapping::MappingStage;
use crate::mission::{MissionConfig, MissionEvent, MissionInputs, MissionPhase, MissionStage};
use crate::overrides::{GotoHover, OverrideCommand};
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{Stage, StageError, StageOutcome};

/// Everything the driver takes from the scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionDriverConfig {
    pub mission: MissionConfig,
    /// `_make_redirect_goto(..., 28.0)` for POI and egress redirects.
    pub redirect_speed_mps: f64,
    /// A drone below this battery fraction hands its POI off.
    pub handoff_low_fraction: f64,
    /// ...to a free drone above this one.
    pub handoff_fresh_fraction: f64,
    /// `coverage_in_region(poi, radius_m=50.0)` in `rescore_from_map`.
    pub rescore_radius_m: f64,
}

impl Default for MissionDriverConfig {
    fn default() -> Self {
        Self {
            mission: MissionConfig::default(),
            redirect_speed_mps: 28.0,
            handoff_low_fraction: 0.30,
            handoff_fresh_fraction: 0.50,
            rescore_radius_m: 50.0,
        }
    }
}

/// `WorldMap.coverage_in_region`: the fraction of grid cells within a disc
/// that have been visited at least once.
///
/// The disc is in **cell** units -- `r_cells = max(1, int(radius /
/// resolution))`, cells kept where `di^2 + dj^2 <= r_cells^2` -- centred on the
/// cell the point truncates into. `int()` truncates toward zero, as `as i64`
/// does. An empty window reports zero coverage.
pub fn coverage_in_region(mapping: &MappingStage, center_xy: [f64; 2], radius_m: f64) -> f64 {
    coverage_in_grid_region(
        mapping.bounds(),
        |i, j| mapping.coverage_count_at(i, j),
        center_xy,
        radius_m,
    )
}

/// [`coverage_in_region`] over any count grid, indexed `(i, j)`.
pub fn coverage_in_grid_region(
    bounds: argusnet_mapping::GridBounds,
    count_at: impl Fn(usize, usize) -> u32,
    center_xy: [f64; 2],
    radius_m: f64,
) -> f64 {
    let Ok((nx, ny)) = bounds.dimensions() else {
        return 0.0;
    };
    let ci = ((center_xy[0] - bounds.x_min_m) / bounds.resolution_m) as i64;
    let cj = ((center_xy[1] - bounds.y_min_m) / bounds.resolution_m) as i64;
    let r_cells = ((radius_m / bounds.resolution_m) as i64).max(1);
    let (i_lo, i_hi) = ((ci - r_cells).max(0), (ci + r_cells + 1).min(nx as i64));
    let (j_lo, j_hi) = ((cj - r_cells).max(0), (cj + r_cells + 1).min(ny as i64));
    if i_lo >= i_hi || j_lo >= j_hi {
        return 0.0;
    }
    let (mut total, mut covered) = (0u64, 0u64);
    for i in i_lo..i_hi {
        for j in j_lo..j_hi {
            if (i - ci).pow(2) + (j - cj).pow(2) <= r_cells.pow(2) {
                total += 1;
                if count_at(i as usize, j as usize) > 0 {
                    covered += 1;
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        covered as f64 / total as f64
    }
}

/// The fraction of cells that are enclosed interior holes -- the gap-fill
/// gate's input.
fn gap_cell_fraction(mapping: &MappingStage) -> f64 {
    let (cells, total) = gap_cells(mapping);
    cells.len() as f64 / total.max(1) as f64
}

/// The enclosed interior holes, and the grid's cell count.
fn gap_cells(mapping: &MappingStage) -> (Vec<(usize, usize)>, usize) {
    let bounds = mapping.bounds();
    let Ok((nx, ny)) = bounds.dimensions() else {
        return (Vec::new(), 0);
    };
    let counts = mapping.coverage_counts();
    debug_assert_eq!(counts.len(), nx * ny, "coverage grid and bounds disagree");
    let view = CoverageView {
        counts: &counts,
        bounds: argusnet_planning::frontier::GridBounds {
            x_min_m: bounds.x_min_m,
            y_min_m: bounds.y_min_m,
            resolution_m: bounds.resolution_m,
            nx,
            ny,
        },
    };
    (find_gap_cells(&view), nx * ny)
}

/// One tick of recorded localization: what the machine would have read from
/// stage 8.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedLocalization {
    pub confidences: Vec<f64>,
    pub timed_out: bool,
}

/// Recorded localization, replayed into the mission machine.
///
/// Localization is the one input to the mission that no parity test can match
/// bit for bit: the Rust localizer draws from ADR-006 streams, not NumPy's. A
/// run handed the reference's recorded confidences instead is left with no
/// stochastic input at all, so everything the mission *does* -- transitions,
/// assignments, redirects, dwell, egress -- can be compared exactly for the
/// whole run. A replay seam, not a production path: an empty replay (the
/// default) reads stage 8 as normal.
#[derive(Debug, Clone, Default)]
struct LocalizationReplay {
    ticks: Vec<RecordedLocalization>,
}

impl LocalizationReplay {
    fn at(&self, tick: u64) -> Option<RecordedLocalization> {
        self.ticks.get(tick as usize).cloned()
    }

    fn is_loaded(&self) -> bool {
        !self.ticks.is_empty()
    }
}

/// Whether a decision took effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    /// Applied to stage 9's own state this run.
    Applied,
    /// A command queued for stage 2, which flies it from the next tick
    /// (`Handoff::OverridesFromMission`).
    QueuedForNextTick,
    /// Attempted and refused.
    Rejected,
}

/// A typed record of one mission decision (F3): what was chosen, from what,
/// why, on which inputs, and what it commands. Written to the durable journal
/// (A4), independent of display frames; never part of a replay or a
/// checkpoint, and rebuilt every run.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Decision {
    pub kind: &'static str,
    pub status: DecisionStatus,
    /// The platform the decision is about, if one.
    pub subject: Option<String>,
    pub selected: Option<String>,
    /// What was chosen among, in the order considered.
    pub candidates: Vec<String>,
    /// A stable reason code.
    pub reason: &'static str,
    pub detail: String,
    /// The inputs the decision was made on.
    pub inputs: serde_json::Value,
    /// The command it produced, if any.
    pub command: Option<&'static str>,
}

/// What stage 9 checkpoints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DriverCheckpoint {
    machine: crate::mission::MissionState,
    pois: PoiManager,
    redirected: BTreeMap<String, String>,
    rescored: bool,
    #[serde(default)]
    coordinator: Option<String>,
    /// Absent for the reference policy, so its checkpoints are unchanged.
    #[serde(default, skip_serializing_if = "PlanningPolicy::is_reference")]
    policy: PlanningPolicy,
    /// Drones sent home on their own reserve (P2). Absent when none, so
    /// every other checkpoint is unchanged.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    homeward: BTreeSet<String>,
    /// Drones sweeping holes (P4). Absent when none.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    gap_fill: BTreeMap<String, GapFill>,
}

/// A drone sweeping one hole in the coverage map (P4).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct GapFill {
    /// The hole's cells, as `[i, j]`.
    cells: Vec<[usize; 2]>,
    /// When its sweep ends: a hole the sweep left open is let go then.
    arrive_s: f64,
}

/// What stage 9 reports for a replay: POI statuses with their positions, and
/// the elected coordinator. Refreshed at the end of every run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriverReport {
    /// Each POI's status beside its position, in construction order --
    /// `replace(s, position=_poi_manager._pois[s.poi_id].position)`.
    pub pois: Vec<(PoiStatus, [f64; 3])>,
    pub completed_count: usize,
    pub total_count: usize,
    /// `_shared_state.coordinator_id`.
    pub coordinator_drone_id: Option<String>,
}

/// `CoordinationManager.elect_coordinator`: the highest battery fraction,
/// the first in fleet order on a tie, and a drone off the ledger counted as
/// full.
pub fn elect_coordinator(
    drone_ids: &[String],
    battery: &BatteryStage,
    capacity_wh: f64,
) -> Option<String> {
    let mut best: Option<(&String, f64)> = None;
    for id in drone_ids {
        let fraction = battery
            .remaining_wh(id)
            .map_or(1.0, |remaining| remaining / capacity_wh.max(1.0));
        // `best_frac = -1.0` and a strict `>`.
        if fraction > best.map_or(-1.0, |(_, f)| f) {
            best = Some((id, fraction));
        }
    }
    // `best_id = drone_ids[0]`, kept if no fraction ever beat -1.
    best.map(|(id, _)| id.clone())
        .or_else(|| drone_ids.first().cloned())
}

/// Stage 9: the mission machine and the work its phases do.
pub struct MissionDriverStage {
    machine: MissionStage,
    /// This run's decisions (F3).
    decisions: Vec<Decision>,
    /// Writes into stage 2's tables, queued this run and drained by
    /// [`Handoff::OverridesFromMission`] straight after it.
    ///
    /// [`Handoff::OverridesFromMission`]: crate::simulation::Handoff::OverridesFromMission
    override_commands: Vec<OverrideCommand>,
    pois: PoiManager,
    initial_pois: Vec<InspectionPoi>,
    /// `_drone_home_positions`: each launched drone's station.
    homes: BTreeMap<String, [f64; 3]>,
    /// `_drone_poi_assignments`: which POI each drone was last redirected to.
    redirected: BTreeMap<String, String>,
    /// `_poi_rescored`.
    rescored: bool,
    /// `_shared_state.coordinator_id`, elected once on the first scanning
    /// tick with a fleet (`_coordinator_elected`).
    coordinator: Option<String>,
    config: MissionDriverConfig,
    /// Who allocates POIs (P1). Scenario, like `config`: set before the
    /// first tick, kept by reset, and checked against a checkpoint's.
    policy: PlanningPolicy,
    /// Drones sent home on their own reserve under efficient_v1 (P2): they
    /// hold no POI and take none again.
    homeward: BTreeSet<String>,
    /// Drones sweeping a hole in the coverage map under efficient_v1 (P4),
    /// with the cells they sweep.
    gap_fill: BTreeMap<String, GapFill>,
    replay: LocalizationReplay,
    /// Refreshed at the end of every run, and on reset and restore.
    report: DriverReport,
    /// The known hard obstacles that POI and egress flights route around,
    /// and that the allocation costs, under efficient_v1 (P3). `None` flies
    /// straight lines, as the reference always does. Scenario, like
    /// `config`: set at assembly and kept by reset.
    obstacles: Option<KnownObstacles>,
    /// The prior map's terrain, so a route round obstacles that would carry a
    /// drone lower than the ground under its corners is not flown (P3).
    terrain: Option<Arc<dyn crate::observe::TerrainHeight>>,
}

/// The obstacle layer the driver plans routes over (P3), with its planner.
///
/// The planner works in the plane, but a platform only has to go around what
/// reaches its height: one flying over a low building collides with nothing
/// (stage 2's collision is 3D). So a trip is planned only against the hard
/// obstacles whose top, plus the planner's clearance, is above the lowest
/// height the flight passes through. That height is rounded down to a
/// [`ROUTE_HEIGHT_BAND_M`] band, which can only add obstacles, and the
/// filtered layer is kept per band.
///
/// The planner's own route cache is not used: it is keyed on endpoints
/// snapped to 10 m, so it would hand back a route from wherever an earlier
/// request started, and a resumed run, whose cache starts empty, would fly
/// differently. Instead each run keeps the routes it has planned by their
/// exact endpoints, so the cost matrices and the redirects that follow them
/// plan each trip once, and starts the next run empty.
struct KnownObstacles {
    layer: Arc<ObstacleLayer>,
    planner: RefCell<RoutePlanner>,
    /// The obstacles that reach above each height band, by band; `None`
    /// when nothing does.
    reaching: RefCell<BTreeMap<i64, Option<Arc<ObstacleLayer>>>>,
    this_run: RefCell<BTreeMap<[u64; 6], Option<PlannerRoute>>>,
    /// The corners of routes between points that do not move (a POI and a
    /// station), checked against the terrain, kept across runs. They depend
    /// on the two ends, the layer and the terrain alone, so keeping them only
    /// saves work; their number is bounded by the mission's POIs and
    /// stations. Cleared when the terrain changes.
    kept: RefCell<BTreeMap<[u64; 6], Option<Corners>>>,
}

/// How many times [`MissionDriverStage::sweep_height`] raises a sweep's end
/// height to clear the lowest points it finds before giving the sweep up.
const SWEEP_HEIGHT_ROUNDS: usize = 64;

/// The height bands a route's obstacles are filtered in, in metres.
pub const ROUTE_HEIGHT_BAND_M: f64 = 10.0;

impl KnownObstacles {
    /// The obstacles a flight never lower than `floor_m` must go around, or
    /// `None` when none reaches that high.
    fn reaching(&self, floor_m: f64, clearance_m: f64) -> Option<Arc<ObstacleLayer>> {
        let band = (floor_m / ROUTE_HEIGHT_BAND_M).floor() as i64;
        if let Some(known) = self.reaching.borrow().get(&band) {
            return known.clone();
        }
        let band_floor_m = band as f64 * ROUTE_HEIGHT_BAND_M;
        let primitives: Vec<_> = self
            .layer
            .primitives
            .iter()
            .filter(|obstacle| obstacle.top_z_m + clearance_m > band_floor_m)
            .cloned()
            .collect();
        let layer = if primitives.is_empty() {
            None
        } else {
            ObstacleLayer::new(self.layer.bounds, self.layer.tile_size_m, primitives)
                .ok()
                .map(Arc::new)
        };
        self.reaching.borrow_mut().insert(band, layer.clone());
        layer
    }

    /// A route around the hard obstacles from `from` to `to`, or `None` when
    /// nothing reaches the flight's height or the planner finds no route (an
    /// end inside an obstacle or out of bounds, or no way through). An end
    /// inside the clearance, such as a POI beside a wall, is joined to the
    /// route by a short straight leg ([`RoutePlanner::plan_route_between`]).
    fn route(&self, from: [f64; 3], to: [f64; 3]) -> Option<PlannerRoute> {
        let key = route_key(from, to);
        if let Some(known) = self.this_run.borrow().get(&key) {
            return known.clone();
        }
        let mut planner = self.planner.borrow_mut();
        let clearance_m = planner.config().drone_clearance_m;
        let route = self
            .reaching(from[2].min(to[2]), clearance_m)
            .and_then(|layer| {
                // Each band's layer is fixed for the run; the band is its
                // revision, so the planner never mixes their footprints.
                let band = (from[2].min(to[2]) / ROUTE_HEIGHT_BAND_M).floor() as i64;
                let scene = Scene {
                    layer: &layer,
                    revision: band as u64,
                };
                planner.plan_route_between(scene, [from[0], from[1]], [to[0], to[1]], clearance_m)
            });
        self.this_run.borrow_mut().insert(key, route.clone());
        route
    }

    /// Whether a flight at `z_m` must go around anything at all.
    fn blocks_at(&self, z_m: f64) -> bool {
        let clearance_m = self.planner.borrow().config().drone_clearance_m;
        self.reaching(z_m, clearance_m).is_some()
    }

    /// Whether `xy` lies in the footprint of a building or wall a flight at
    /// `z_m` must go around: the planner has no way to it.
    fn stands_in(&self, xy: [f64; 2], z_m: f64) -> bool {
        let clearance_m = self.planner.borrow().config().drone_clearance_m;
        let Some(layer) = self.reaching(z_m, clearance_m) else {
            return false;
        };
        let at = Bounds2d::new(xy[0], xy[0], xy[1], xy[1]).expanded(1e-6);
        layer.query(at).into_iter().any(|o| {
            o.blocker_type.is_hard() && o.point_inside(xy[0], xy[1], 0.5 * (o.base_z_m + o.top_z_m))
        })
    }
}

/// A free drone's flight to sweep one hole: the cells in the order flown,
/// the corners on the way (around obstacles, to the first cell and between
/// cells), the height flown, and the lengths of the way there and of the
/// sweep.
#[derive(Debug, Clone)]
struct GapPlan {
    points: Vec<[f64; 2]>,
    via: Vec<[f64; 2]>,
    z: f64,
    approach: f64,
    sweep_m: f64,
}

/// The holes' 4-connected clusters, each in row-major order, in the order
/// of their first cell.
fn hole_clusters(cells: &[(usize, usize)]) -> Vec<Vec<[usize; 2]>> {
    let mut left: BTreeSet<[usize; 2]> = cells.iter().map(|&(i, j)| [i, j]).collect();
    let mut clusters = Vec::new();
    while let Some(&seed) = left.iter().next() {
        left.remove(&seed);
        let mut cluster = vec![seed];
        let mut queue = std::collections::VecDeque::from([seed]);
        while let Some([i, j]) = queue.pop_front() {
            let neighbours = [
                [i.wrapping_sub(1), j],
                [i + 1, j],
                [i, j.wrapping_sub(1)],
                [i, j + 1],
            ];
            for next in neighbours {
                if left.remove(&next) {
                    cluster.push(next);
                    queue.push_back(next);
                }
            }
        }
        cluster.sort_unstable();
        clusters.push(cluster);
    }
    clusters
}

/// A cluster's cells in sweep order: column by column (`i`), alternating
/// direction along each, so the sweep never doubles back across the hole.
fn sweep_order(cluster: &[[usize; 2]]) -> Vec<[usize; 2]> {
    let mut columns: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &[i, j] in cluster {
        columns.entry(i).or_default().push(j);
    }
    let mut order = Vec::with_capacity(cluster.len());
    for (k, (i, mut js)) in columns.into_iter().enumerate() {
        js.sort_unstable();
        if k % 2 == 1 {
            js.reverse();
        }
        order.extend(js.into_iter().map(|j| [i, j]));
    }
    order
}

/// A route's corners between its ends, and its length.
type Corners = (Vec<[f64; 2]>, f64);

/// A trip's exact ends, as a cache key.
fn route_key(from: [f64; 3], to: [f64; 3]) -> [u64; 6] {
    [
        from[0].to_bits(),
        from[1].to_bits(),
        from[2].to_bits(),
        to[0].to_bits(),
        to[1].to_bits(),
        to[2].to_bits(),
    ]
}

/// A flight the driver commanded, and the route it follows if it turns.
struct Flight {
    trajectory: GotoHover,
    /// `(length, corners)` of a route around obstacles; `None` for a straight
    /// flight.
    route: Option<(f64, usize)>,
}

impl Flight {
    /// Add the route to a decision's inputs, when there is one. Additive: a
    /// straight flight's inputs are unchanged.
    fn record(&self, inputs: &mut serde_json::Value) {
        if let (Some((length_m, corners)), Some(fields)) = (self.route, inputs.as_object_mut()) {
            fields.insert("route_length_m".into(), serde_json::json!(length_m));
            fields.insert("route_waypoints".into(), serde_json::json!(corners));
        }
    }
}

/// Under [`PlanningPolicy::EfficientV1`], what one step of rescored POI
/// priority is worth against energy. At the reference battery's 350 W and
/// 28 m/s, 5 Wh is about 1.4 km of flight: a higher-priority POI is preferred
/// unless it is that much further. Tuned on the qualification seeds only.
pub const PRIORITY_WORTH_WH: f64 = 5.0;

/// The longest route home, as a multiple of the straight line, that a
/// drone's reserve check prices exactly under efficient_v1 (P3). A drone with
/// the energy to fly this far home, against the worst the wind can do, is
/// not planned a route home at all: routing from every drone's position on
/// every run would cost more than the run. A route home longer than this
/// (behind a very long wall) is priced only once the battery falls to this
/// bound, so the drone turns home later than its exact reserve says.
pub const MAX_PRICED_DETOUR: f64 = 2.0;

/// An upper bound on the reserve home from `at` along any route at most
/// [`MAX_PRICED_DETOUR`] times the straight line: every leg at the slowest
/// ground speed the wind allows (`return_home_cost_wh`'s floor of a quarter
/// of cruise), then the battery's floor and margin. It is never below the
/// straight-line reserve, so a drone above it is above that too.
fn detour_bound_wh(
    model: &BatteryModel,
    at: [f64; 3],
    home: [f64; 3],
    wind: Option<[f64; 3]>,
) -> f64 {
    let (dx, dy) = (home[0] - at[0], home[1] - at[1]);
    let straight_m = (dx * dx + dy * dy).sqrt();
    let wind_mps = wind.map_or(0.0, |w| (w[0] * w[0] + w[1] * w[1]).sqrt());
    let cruise = model.cruise_speed_m_per_s.max(0.01);
    let slowest = (cruise - wind_mps).max(0.25 * cruise);
    let travel_wh = model.travel_power_w * (MAX_PRICED_DETOUR * straight_m / slowest) / 3600.0;
    let cost_wh = travel_wh + model.climb_cost_wh(home[2] - at[2]);
    (cost_wh * BatteryModel::DEFAULT_SAFETY_MARGIN).max(model.capacity_wh * model.reserve_fraction)
}

/// How close a drone must be to a POI, horizontally, to credit its dwell:
/// the `<= 30.0` in [`PoiManager::accumulate_dwell`].
const POI_DWELL_RANGE_M: f64 = 30.0;

impl std::fmt::Debug for MissionDriverStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MissionDriverStage")
            .field("phase", &self.machine.phase())
            .field("pois", &self.pois.total_count())
            .field("rescored", &self.rescored)
            .finish()
    }
}

/// A drone the fleet allocator added to a POI being inspected (P1 teams).
#[derive(Debug, Clone)]
struct TeamJoin {
    drone_id: String,
    poi_id: String,
    /// Drones holding the POI before this one joined.
    members: usize,
    remaining_dwell_s: f64,
    candidates: Vec<String>,
}

impl MissionDriverStage {
    pub fn new(
        config: MissionDriverConfig,
        pois: Vec<InspectionPoi>,
        homes: BTreeMap<String, [f64; 3]>,
    ) -> Self {
        let machine = MissionStage::new(config.mission);
        Self {
            machine,
            override_commands: Vec::new(),
            decisions: Vec::new(),
            pois: PoiManager::new(pois.clone()),
            initial_pois: pois,
            homes,
            redirected: BTreeMap::new(),
            homeward: BTreeSet::new(),
            gap_fill: BTreeMap::new(),
            rescored: false,
            coordinator: None,
            config,
            policy: PlanningPolicy::Reference,
            replay: LocalizationReplay::default(),
            report: DriverReport::default(),
            obstacles: None,
            terrain: None,
        }
    }

    /// The hard obstacles POI and egress flights route around under
    /// efficient_v1 (P3), planned within `bounds`. `None` (the default) flies
    /// straight lines. The reference never reads it. A run whose platforms
    /// do not know the world in advance must not be given the true layer.
    pub fn set_known_obstacles(&mut self, layer: Option<Arc<ObstacleLayer>>, bounds: Bounds2d) {
        self.obstacles = layer.map(|layer| KnownObstacles {
            layer,
            planner: RefCell::new(RoutePlanner::new(bounds, PlannerConfig::default())),
            reaching: RefCell::new(BTreeMap::new()),
            this_run: RefCell::new(BTreeMap::new()),
            kept: RefCell::new(BTreeMap::new()),
        });
    }

    /// The terrain a routed flight must stay above (P3). Without it a route
    /// is flown as planned.
    pub fn set_known_terrain(&mut self, terrain: Option<Arc<dyn crate::observe::TerrainHeight>>) {
        self.terrain = terrain;
        if let Some(obstacles) = &self.obstacles {
            obstacles.kept.borrow_mut().clear();
        }
    }

    /// Whether flights route around known obstacles when the policy does.
    pub fn knows_obstacles(&self) -> bool {
        self.obstacles.is_some()
    }

    /// The route from `from` to `to` around known obstacles, when the policy
    /// routes, a layer is known and the planner finds one. A route that does
    /// not turn is returned too, as two points.
    fn route(&self, from: [f64; 3], to: [f64; 3]) -> Option<PlannerRoute> {
        if self.policy != PlanningPolicy::EfficientV1 {
            return None;
        }
        self.obstacles.as_ref()?.route(from, to)
    }

    /// The route's corners, if it turns at all.
    fn corners(&self, from: [f64; 3], to: [f64; 3]) -> Option<(Vec<[f64; 2]>, f64)> {
        self.corners_between(from, to, false)
    }

    /// [`Self::corners`]; `fixed_ends` when neither end moves, so the
    /// outcome is kept for later runs.
    fn corners_between(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        fixed_ends: bool,
    ) -> Option<(Vec<[f64; 2]>, f64)> {
        let kept = match (fixed_ends, &self.obstacles) {
            (true, Some(obstacles)) if self.policy == PlanningPolicy::EfficientV1 => {
                Some(&obstacles.kept)
            }
            _ => None,
        };
        let key = route_key(from, to);
        if let Some(known) = kept.and_then(|kept| kept.borrow().get(&key).cloned()) {
            return known;
        }
        let corners = self.route(from, to).and_then(|route| {
            let points = &route.points_xy_m;
            (points.len() > 2 && self.clears_terrain(points, from[2], to[2], route.length_m))
                .then(|| (points[1..points.len() - 1].to_vec(), route.length_m))
        });
        if let Some(kept) = kept {
            kept.borrow_mut().insert(key, corners.clone());
        }
        corners
    }

    /// The free drones (by position) matched to the sweeps, each pair the
    /// drone with its flight: least total way there, over only the pairs
    /// whose whole flight is safe, so a nearer drone that cannot fly a sweep
    /// never takes it from a farther one that can.
    fn assign_gap_sweeps(
        &self,
        positions: &[[f64; 3]],
        sweeps: &[Vec<[f64; 2]>],
    ) -> Vec<(usize, usize, GapPlan)> {
        let mut plans: Vec<Vec<Option<GapPlan>>> = positions
            .iter()
            .map(|position| {
                sweeps
                    .iter()
                    .map(|sweep| self.gap_plan(*position, sweep))
                    .collect()
            })
            .collect();
        let cost: Vec<Vec<Option<f64>>> = plans
            .iter()
            .map(|row| {
                row.iter()
                    .map(|plan| plan.as_ref().map(|plan| plan.approach))
                    .collect()
            })
            .collect();
        min_cost_assignment(&cost)
            .into_iter()
            .enumerate()
            .filter_map(|(row, column)| {
                let column = column?;
                let plan = plans[row][column]
                    .take()
                    .expect("only a plan has a cost to be assigned on");
                Some((row, column, plan))
            })
            .collect()
    }

    /// The flight of a drone at `position` to sweep `sweep`, or `None` when
    /// it cannot be made safe. It sweeps from the nearer end, at its own
    /// height above the ground, raised to clear the highest ground the sweep
    /// crosses.
    fn gap_plan(&self, position: [f64; 3], sweep: &[[f64; 2]]) -> Option<GapPlan> {
        let ends = [sweep[0], sweep[sweep.len() - 1]];
        let reach = |end: [f64; 2]| {
            let (dx, dy) = (end[0] - position[0], end[1] - position[1]);
            (dx * dx + dy * dy).sqrt()
        };
        let mut points = sweep.to_vec();
        if reach(ends[1]) < reach(ends[0]) {
            points.reverse();
        }
        let z = match &self.terrain {
            Some(terrain) => {
                let above = position[2] - terrain.height_at(position[0], position[1]);
                points
                    .iter()
                    .map(|p| terrain.height_at(p[0], p[1]))
                    .fold(f64::NEG_INFINITY, f64::max)
                    + above
            }
            None => position[2],
        };
        // A cell centre inside a building it cannot clear is left out:
        // the sweep passes beside it.
        if let Some(obstacles) = &self.obstacles {
            points.retain(|p| !obstacles.stands_in(*p, z));
        }
        if points.is_empty() {
            return None;
        }
        let first = [points[0][0], points[0][1], z];
        let approach = self.trip_length_m(position, first);
        let sweep_m: f64 = points
            .windows(2)
            .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
            .sum();
        // The whole flight, the way there and every leg of the sweep,
        // routed around the known obstacles, at the lowest height that
        // keeps it above the known ground everywhere. The way there is
        // routed at the lower of the two heights it flies between, so it
        // goes around all it climbs past. A flight that cannot be made
        // safe is no plan at all, so the assignment never picks it.
        let mut via =
            self.routed_sweep(&[[position[0], position[1]], points[0]], z.min(position[2]))?;
        via.remove(0);
        via.pop();
        via.extend(self.routed_sweep(&points, z)?);
        let z = self.sweep_height(position, &via, z)?;
        via.pop();
        Some(GapPlan {
            points,
            via,
            z,
            approach,
            sweep_m,
        })
    }

    /// The sweep through `points` at `z_m`, each leg routed around the
    /// known obstacles a flight at that height must avoid: the corners of
    /// every leg that turns, then its end. `None` when a leg has no route.
    fn routed_sweep(&self, points: &[[f64; 2]], z_m: f64) -> Option<Vec<[f64; 2]>> {
        let mut path = vec![points[0]];
        for leg in points.windows(2) {
            let (from, to) = ([leg[0][0], leg[0][1], z_m], [leg[1][0], leg[1][1], z_m]);
            match self.route(from, to) {
                Some(route) => {
                    let turns = &route.points_xy_m;
                    if turns.len() > 2 {
                        path.extend_from_slice(&turns[1..turns.len() - 1]);
                    }
                }
                None if self.obstacles.as_ref().is_some_and(|o| o.blocks_at(z_m)) => {
                    return None;
                }
                None => {}
            }
            path.push(leg[1]);
        }
        Some(path)
    }

    /// The height a flight from `start` through `path` should end at, at
    /// least `base_z`, so that climbing (or descending) linearly along the
    /// whole of it, as [`GotoHover::routed`] flies, it keeps the planner's
    /// clearance above the known terrain everywhere, solved on the
    /// terrain's cells ([`crate::ground`]). Each round raises the end height
    /// just enough to clear every leg's lowest point; `None` when no height
    /// will do (the start itself is below the clearance) or none is found
    /// in [`SWEEP_HEIGHT_ROUNDS`] rounds.
    fn sweep_height(&self, start: [f64; 3], path: &[[f64; 2]], base_z: f64) -> Option<f64> {
        let Some(terrain) = &self.terrain else {
            return Some(base_z);
        };
        let clearance_m = PlannerConfig::default().drone_clearance_m;
        let mut points = Vec::with_capacity(path.len() + 1);
        points.push([start[0], start[1]]);
        points.extend(path.iter().copied());
        let leg_m = |pair: &[[f64; 2]]| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
        let length_m: f64 = points.windows(2).map(leg_m).sum();
        if start[2] < terrain.height_at(start[0], start[1]) + clearance_m {
            return None;
        }
        if length_m <= 1e-9 {
            return Some(base_z);
        }
        let mut z = base_z;
        for _ in 0..SWEEP_HEIGHT_ROUNDS {
            // At distance s along, the height is start + s / L * (z - start);
            // a point of ground h there needs z at least
            // start + (h + clearance - start) * L / s.
            let mut needed = z;
            let mut flown_m = 0.0;
            for pair in points.windows(2) {
                let leg = leg_m(pair);
                let at = |s: f64| start[2] + (z - start[2]) * s / length_m;
                let (from, to) = (
                    [pair[0][0], pair[0][1], at(flown_m)],
                    [pair[1][0], pair[1][1], at(flown_m + leg)],
                );
                let (margin, t) = crate::ground::lowest_margin(terrain.as_ref(), from, to);
                let s = flown_m + t * leg;
                if margin < clearance_m && s > 1e-9 {
                    // A micrometre over, so rounding never fails the check.
                    let ground = at(s) - margin;
                    needed = needed
                        .max(start[2] + (ground + clearance_m + 1e-6 - start[2]) * length_m / s);
                }
                flown_m += leg;
            }
            if needed <= z {
                return self
                    .clears_terrain(&points, start[2], z, length_m)
                    .then_some(z);
            }
            z = needed;
        }
        None
    }

    /// Whether a routed flight, its height changing linearly along the route
    /// as [`GotoHover::routed`] flies it, keeps the planner's clearance above
    /// the terrain everywhere along it, solved on the terrain's cells
    /// ([`crate::ground`]). A detour can cross higher ground than the
    /// straight line between its ends; one that would go below it is not
    /// flown.
    fn clears_terrain(&self, points: &[[f64; 2]], from_z: f64, to_z: f64, length_m: f64) -> bool {
        let Some(terrain) = &self.terrain else {
            return true;
        };
        let clearance_m = PlannerConfig::default().drone_clearance_m;
        let z_at = |s: f64| {
            let frac = if length_m > 0.0 {
                (s / length_m).clamp(0.0, 1.0)
            } else {
                0.0
            };
            from_z + frac * (to_z - from_z)
        };
        let mut flown_m = 0.0;
        for pair in points.windows(2) {
            let leg_m = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            let from = [pair[0][0], pair[0][1], z_at(flown_m)];
            let to = [pair[1][0], pair[1][1], z_at(flown_m + leg_m)];
            if crate::ground::lowest_margin(terrain.as_ref(), from, to).0 < clearance_m {
                return false;
            }
            flown_m += leg_m;
        }
        true
    }

    /// The horizontal distance a trip flies: the route's length when it has
    /// to turn, else the straight line.
    fn trip_length_m(&self, from: [f64; 3], to: [f64; 3]) -> f64 {
        match self.corners(from, to) {
            Some((_, length_m)) => length_m,
            None => {
                let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
                (dx * dx + dy * dy).sqrt()
            }
        }
    }

    /// The energy a trip costs: the route's length when it has to turn, else
    /// exactly the straight segment's cost the allocator has always used.
    fn trip_cost_wh(&self, model: &BatteryModel, from: [f64; 3], to: [f64; 3]) -> f64 {
        match self.corners(from, to) {
            Some((_, length_m)) => {
                model.travel_cost_wh(length_m) + model.climb_cost_wh(to[2] - from[2])
            }
            None => model.segment_cost_wh(from, to),
        }
    }

    /// Gap filling under efficient_v1 (P4). Each drone loops its own scan
    /// sector, so cells between sectors that no lane reaches stay uncovered,
    /// and once coverage passes the threshold those enclosed holes hold the
    /// scan open (the gap gate) for the rest of the run. While they do, each
    /// hole (4-connected) gets the free drone that reaches it soonest, which
    /// sweeps its cells from the nearer end, around known obstacles on the
    /// way there, and goes back to its lane once the hole is covered or its
    /// sweep ends. The flight is a frontier override, so it is flown only
    /// while scanning or localizing.
    fn fill_gaps(
        &mut self,
        holes: Option<(Vec<(usize, usize)>, usize)>,
        gap_cell_fraction: Option<f64>,
        mobile: &[(String, [f64; 3])],
        mapping: &MappingStage,
        timestamp_s: f64,
    ) {
        // Let go of drones whose hole is covered, whose sweep has ended, that
        // have left the fleet or turned home, and of every one once the scan has
        // ended: stage 2 still flies a frontier override while localizing,
        // so each is cleared, not just forgotten.
        let scanning = self.machine.phase() == MissionPhase::Scanning;
        let finished: Vec<(String, &'static str)> = self
            .gap_fill
            .iter()
            .filter_map(|(id, fill)| {
                if !scanning {
                    Some((id.clone(), "scan_ended"))
                } else if !mobile.iter().any(|(m, _)| m == id) {
                    Some((id.clone(), "left_fleet"))
                } else if self.homeward.contains(id) {
                    // Going home on its reserve: its hole is free for
                    // another drone at once, not when its sweep would have
                    // ended.
                    Some((id.clone(), "homeward"))
                } else if fill
                    .cells
                    .iter()
                    .all(|[i, j]| mapping.coverage_count_at(*i, *j) > 0)
                {
                    Some((id.clone(), "gap_covered"))
                } else if timestamp_s >= fill.arrive_s {
                    Some((id.clone(), "sweep_ended"))
                } else {
                    None
                }
            })
            .collect();
        for (id, reason) in finished {
            let fill = self.gap_fill.remove(&id).expect("listed above");
            self.override_commands
                .push(OverrideCommand::ClearFrontier(id.clone()));
            self.decisions.push(Decision {
                kind: "gap_fill_release",
                status: DecisionStatus::QueuedForNextTick,
                subject: Some(id.clone()),
                selected: None,
                candidates: Vec::new(),
                reason,
                detail: format!(
                    "{id} goes back to its lane ({} cells swept)",
                    fill.cells.len()
                ),
                inputs: serde_json::json!({ "cells": fill.cells.len() }),
                command: Some("clear_frontier"),
            });
        }
        if !scanning {
            return;
        }
        let (Some((cells, _)), Some(fraction)) = (holes, gap_cell_fraction) else {
            return;
        };
        if fraction < self.machine.config().gap_fill_min_fraction {
            return;
        }
        let taken: BTreeSet<[usize; 2]> = self
            .gap_fill
            .values()
            .flat_map(|fill| fill.cells.iter().copied())
            .collect();
        let clusters: Vec<Vec<[usize; 2]>> = hole_clusters(&cells)
            .into_iter()
            .filter(|cluster| cluster.iter().all(|cell| !taken.contains(cell)))
            .collect();
        let free: Vec<&(String, [f64; 3])> = mobile
            .iter()
            .filter(|(id, _)| {
                !self.gap_fill.contains_key(id)
                    && !self.homeward.contains(id)
                    && self.pois.assignment_for(id).is_none()
            })
            .collect();
        if clusters.is_empty() || free.is_empty() {
            return;
        }
        let bounds = mapping.bounds();
        let centre = |[i, j]: [usize; 2]| {
            [
                bounds.x_min_m + (i as f64 + 0.5) * bounds.resolution_m,
                bounds.y_min_m + (j as f64 + 0.5) * bounds.resolution_m,
            ]
        };
        let sweeps: Vec<Vec<[f64; 2]>> = clusters
            .iter()
            .map(|cluster| sweep_order(cluster).into_iter().map(centre).collect())
            .collect();
        let positions: Vec<[f64; 3]> = free.iter().map(|(_, position)| *position).collect();
        let assigned = self.assign_gap_sweeps(&positions, &sweeps);
        let free: Vec<(String, [f64; 3])> = free.into_iter().cloned().collect();
        let candidates: Vec<String> = free.iter().map(|(id, _)| id.clone()).collect();
        let speed = self.config.redirect_speed_mps.max(f64::EPSILON);
        for (row, column, plan) in assigned {
            let (id, position) = &free[row];
            let GapPlan {
                points,
                via,
                z,
                approach,
                sweep_m,
            } = plan;
            let target = points[points.len() - 1];
            let trajectory = GotoHover::routed(*position, via, target, z, timestamp_s, speed);
            let arrive_s = trajectory.t_arrive();
            self.override_commands
                .push(OverrideCommand::SetFrontier(id.clone(), trajectory));
            let cluster = clusters[column].clone();
            self.decisions.push(Decision {
                kind: "gap_fill",
                status: DecisionStatus::QueuedForNextTick,
                subject: Some(id.clone()),
                selected: Some(format!("gap:{}:{}", cluster[0][0], cluster[0][1])),
                candidates: candidates.clone(),
                reason: "gap_gate_blocked",
                detail: format!(
                    "{id} sweeps {} uncovered cells enclosed by the scan ({sweep_m:.0} m, \
                     {approach:.0} m away)",
                    cluster.len()
                ),
                inputs: serde_json::json!({
                    "gap_cell_fraction": fraction,
                    "cells": cluster.len(),
                    "approach_m": approach,
                    "sweep_m": sweep_m,
                    "position_m": position,
                }),
                command: Some("set_frontier"),
            });
            self.gap_fill.insert(
                id.clone(),
                GapFill {
                    cells: cluster,
                    arrive_s,
                },
            );
        }
    }

    /// The reserve a drone needs to get home from `at`: the battery's own
    /// rule (`dynamic_reserve_wh`, its floor and safety margin), priced
    /// along the route it would fly home when that route turns (P3), each leg
    /// with the wind along that leg. A trip home that does not turn gets
    /// exactly the battery's straight-line reserve.
    fn reserve_home_wh(
        &self,
        model: &BatteryModel,
        at: [f64; 3],
        home: [f64; 3],
        wind: Option<[f64; 3]>,
        fixed_ends: bool,
    ) -> f64 {
        let margin = BatteryModel::DEFAULT_SAFETY_MARGIN;
        let Some((via, _)) = self.corners_between(at, home, fixed_ends) else {
            return model.dynamic_reserve_wh(at, home, wind, margin);
        };
        let mut points = Vec::with_capacity(via.len() + 2);
        points.push([at[0], at[1]]);
        points.extend(via);
        points.push([home[0], home[1]]);
        let travel_wh: f64 = points
            .windows(2)
            .map(|leg| {
                model.return_home_cost_wh(
                    [leg[0][0], leg[0][1], 0.0],
                    [leg[1][0], leg[1][1], 0.0],
                    wind,
                )
            })
            .sum();
        let cost_wh = travel_wh + model.climb_cost_wh(home[2] - at[2]);
        (cost_wh * margin).max(model.capacity_wh * model.reserve_fraction)
    }

    /// [`Self::reserve_home_wh`] for drone `id` from a POI at `at`; a drone
    /// with no station holds the battery's floor.
    fn reserve_for(
        &self,
        model: &BatteryModel,
        id: &str,
        at: [f64; 3],
        wind: Option<[f64; 3]>,
    ) -> f64 {
        match self.homes.get(id) {
            Some(home) => self.reserve_home_wh(model, at, *home, wind, true),
            None => model.capacity_wh * model.reserve_fraction,
        }
    }

    /// The flight from `start` to `target`: around known obstacles when the
    /// route turns, else the straight redirect, exactly as before.
    fn flight(&self, start: [f64; 3], target: [f64; 3], t_start: f64) -> Flight {
        let speed = self.config.redirect_speed_mps;
        match self.corners(start, target) {
            Some((via, length_m)) => Flight {
                route: Some((length_m, via.len())),
                trajectory: GotoHover::routed(
                    start,
                    via,
                    [target[0], target[1]],
                    target[2],
                    t_start,
                    speed,
                ),
            },
            None => Flight {
                route: None,
                trajectory: GotoHover::redirect(
                    start,
                    [target[0], target[1]],
                    target[2],
                    t_start,
                    speed,
                ),
            },
        }
    }

    /// Allocate POIs under `policy` (P1). Before the first tick only.
    /// Allocate POIs under `policy`. A policy other than the reference
    /// refuses a mission that repeats a POI id: it assigns, redirects and
    /// credits POIs by id, which a repeated id makes ambiguous. (The
    /// reference's own scenarios never repeat one, and the reference keeps
    /// accepting whatever it accepted.)
    pub fn set_policy(&mut self, policy: PlanningPolicy) -> Result<(), String> {
        if !policy.is_reference() {
            let mut seen = std::collections::BTreeSet::new();
            if let Some(repeated) = self
                .initial_pois
                .iter()
                .find(|poi| !seen.insert(poi.poi_id.as_str()))
            {
                return Err(format!(
                    "--planning-policy {} needs unique POI ids; {} is repeated",
                    policy.as_str(),
                    repeated.poi_id
                ));
            }
        }
        self.policy = policy;
        Ok(())
    }

    pub fn policy(&self) -> PlanningPolicy {
        self.policy
    }

    /// Per-drone return home under efficient_v1 (P2). A drone whose battery
    /// is at or below its reserve home, from where it is now (the battery's
    /// own rule: `dynamic_reserve_wh` to its station), is routed home on its
    /// own, whatever the mission is doing, and gives up any POI it holds
    /// (`hand_off_safely`, run by the allocator). Once homeward it takes
    /// nothing again. A drone with no station has nowhere to go and stays.
    fn send_low_drones_home(
        &mut self,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
        timestamp_s: f64,
    ) {
        let model = battery.model();
        let wind = battery.wind_mps();
        for (id, position) in mobile {
            if self.homeward.contains(id) {
                continue;
            }
            let Some(home) = self.homes.get(id).copied() else {
                continue;
            };
            let Some(remaining_wh) = battery.remaining_wh(id) else {
                continue;
            };
            // Planning a route home from wherever each drone is, every run,
            // is expensive; it is skipped while the drone could fly even a
            // detour of MAX_PRICED_DETOUR times the straight line home.
            if remaining_wh > detour_bound_wh(&model, *position, home, wind) {
                continue;
            }
            let reserve_wh = self.reserve_home_wh(&model, *position, home, wind, false);
            if remaining_wh > reserve_wh {
                continue;
            }
            self.homeward.insert(id.clone());
            let flight = self.flight(*position, home, timestamp_s);
            let mut inputs = serde_json::json!({
                "remaining_wh": remaining_wh,
                "reserve_wh": reserve_wh,
                "position_m": position,
                "poi": self.pois.assignment_for(id),
            });
            flight.record(&mut inputs);
            self.override_commands
                .push(OverrideCommand::SetEgress(id.clone(), flight.trajectory));
            self.decisions.push(Decision {
                kind: "return_home",
                status: DecisionStatus::QueuedForNextTick,
                subject: Some(id.clone()),
                selected: None,
                candidates: Vec::new(),
                reason: "reserve_reached",
                detail: format!(
                    "{id} has {remaining_wh:.1} Wh, at or below the {reserve_wh:.1} Wh it needs \
                     to get home from here: routed home on its own"
                ),
                inputs,
                command: Some("set_egress"),
            });
        }
    }

    /// Safe handoff under efficient_v1 (P1). Every drone below the handoff
    /// threshold gives up the POI it holds. A lead's POI goes to a receiver
    /// that can afford the dwell still to do and the trip home above its
    /// reserve: a teammate already on it (no trip), or a free drone above
    /// the threshold. The solver picks receivers across all low leads on
    /// trip-plus-hover energy. A POI no drone can afford goes back to the
    /// pool (`poi_release`), and a low teammate simply leaves its team. A
    /// drone that gives a POI up stops flying to it.
    fn hand_off_safely(&mut self, mobile: &[(String, [f64; 3])], battery: &BatteryStage) {
        let model = battery.model();
        let capacity_wh = model.capacity_wh;
        let fraction_of = |id: &str| {
            battery
                .remaining_wh(id)
                .map_or(1.0, |remaining| remaining / capacity_wh)
        };
        let threshold = self.config.handoff_low_fraction;
        // A drone at its reserve is already homeward; say so, rather than
        // that its battery is low, whatever its fraction.
        let homeward = self.homeward.clone();
        let giving_up = |id: &str| fraction_of(id) < threshold || homeward.contains(id);
        let why = |id: &str| -> (bool, String) {
            if homeward.contains(id) {
                (true, format!("{id} at its reserve and heading home"))
            } else {
                (false, format!("{id} below {threshold} of capacity"))
            }
        };
        let mut low: Vec<&(String, [f64; 3])> = mobile
            .iter()
            .filter(|(id, _)| giving_up(id) && self.pois.assignment_for(id).is_some())
            .collect();
        if low.is_empty() {
            return;
        }
        low.sort_by(|a, b| a.0.cmp(&b.0));
        // Low teammates leave first, so a POI's members are the drones that
        // stay when its handoff is costed; low leads need a receiver.
        let (leads, teammates): (Vec<_>, Vec<_>) =
            low.into_iter().partition(|(id, _)| self.pois.leads(id));
        for (teammate, _) in teammates {
            let poi = self.pois.assignment_for(teammate).map(str::to_string);
            let (at_reserve, cause) = why(teammate);
            self.pois.release(teammate);
            self.decisions.push(Decision {
                kind: "poi_release",
                status: DecisionStatus::Applied,
                subject: Some(teammate.clone()),
                selected: poi.clone(),
                candidates: Vec::new(),
                reason: if at_reserve {
                    "reserve_teammate"
                } else {
                    "low_battery_teammate"
                },
                detail: format!(
                    "{cause} leaves {}'s team",
                    poi.as_deref().unwrap_or("its POI")
                ),
                inputs: serde_json::json!({
                    "from_fraction": fraction_of(teammate),
                    "at_reserve": at_reserve,
                    "poi": poi,
                }),
                command: None,
            });
            self.stop_flying_to_poi(teammate);
        }
        let active = self.pois.active_pois();
        let poi_of = |id: &str| -> Option<&argusnet_planning::poi::ActivePoi> {
            let held = self.pois.assignment_for(id)?;
            active.iter().find(|poi| poi.poi_id == held)
        };
        let mut receivers: Vec<&(String, [f64; 3])> = mobile
            .iter()
            .filter(|(id, _)| !giving_up(id))
            .filter(|(id, _)| match self.pois.assignment_for(id) {
                None => true,
                Some(_) => !self.pois.leads(id),
            })
            .collect();
        receivers.sort_by(|a, b| a.0.cmp(&b.0));
        let wind = battery.wind_mps();
        let cost: Vec<Vec<Option<f64>>> = leads
            .iter()
            .map(|(lead, _)| {
                let Some(poi) = poi_of(lead) else {
                    return vec![None; receivers.len()];
                };
                receivers
                    .iter()
                    .map(|(id, position)| {
                        let teammate = self.pois.assignment_for(id) == Some(poi.poi_id.as_str());
                        if !teammate && self.pois.assignment_for(id).is_some() {
                            return None;
                        }
                        // It shares the dwell only with the holders that
                        // stay and are already in range: one still en route
                        // would leave it hovering alone until it arrives.
                        let sharing = mobile
                            .iter()
                            .filter(|(other, other_position)| {
                                other != lead
                                    && other != id
                                    && self.pois.assignment_for(other) == Some(poi.poi_id.as_str())
                                    && {
                                        let (dx, dy) = (
                                            poi.position[0] - other_position[0],
                                            poi.position[1] - other_position[1],
                                        );
                                        (dx * dx + dy * dy).sqrt() <= POI_DWELL_RANGE_M
                                    }
                            })
                            .count();
                        let hover_s = poi.remaining_dwell_s / (sharing + 1) as f64;
                        // From where it is: a teammate still en route has
                        // the rest of its trip to fly as well.
                        let trip_wh = self.trip_cost_wh(&model, *position, poi.position);
                        let spent_wh = trip_wh + model.hover_power_w * hover_s / 3600.0;
                        let reserve_wh = match self.homes.get(id.as_str()) {
                            Some(home) => {
                                self.reserve_home_wh(&model, poi.position, *home, wind, true)
                            }
                            None => model.capacity_wh * model.reserve_fraction,
                        };
                        let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                        (remaining_wh - spent_wh >= reserve_wh).then_some(spent_wh)
                    })
                    .collect()
            })
            .collect();
        let chosen = if leads.is_empty() || receivers.is_empty() {
            vec![None; leads.len()]
        } else {
            min_cost_assignment(&cost)
        };
        let candidates: Vec<String> = receivers.iter().map(|(id, _)| id.clone()).collect();
        for ((lead, _), receiver) in leads.iter().zip(chosen) {
            let poi = self.pois.assignment_for(lead).map(str::to_string);
            let fraction = fraction_of(lead);
            let (at_reserve, cause) = why(lead);
            match receiver {
                Some(column) => {
                    let to = &receivers[column].0;
                    let handed = self.pois.trigger_handoff(lead, to);
                    debug_assert!(handed, "a lead hands its active POI on");
                    self.decisions.push(Decision {
                        kind: "poi_handoff",
                        status: DecisionStatus::Applied,
                        subject: Some(lead.clone()),
                        selected: Some(to.clone()),
                        candidates: candidates.clone(),
                        reason: if at_reserve {
                            "reserve_handoff"
                        } else {
                            "safe_handoff"
                        },
                        detail: format!(
                            "{cause} hands {} to {to}, the cheapest drone that can afford \
                             the dwell left and the trip home",
                            poi.as_deref().unwrap_or("its POI")
                        ),
                        inputs: serde_json::json!({
                            "from_fraction": fraction,
                            "at_reserve": at_reserve,
                            "to_fraction": fraction_of(to),
                            "poi": poi,
                        }),
                        command: None,
                    });
                }
                None => {
                    // The whole team lets go: a teammate left holding it is
                    // one the solve found could not afford it either.
                    let released = self.pois.release_poi(lead);
                    for drone in &released {
                        self.stop_flying_to_poi(drone);
                    }
                    self.decisions.push(Decision {
                        kind: "poi_release",
                        status: DecisionStatus::Applied,
                        subject: Some(lead.clone()),
                        selected: poi.clone(),
                        candidates: candidates.clone(),
                        reason: "no_affordable_receiver",
                        detail: format!(
                            "{cause} gives up {}: no drone can \
                             afford the dwell left and the trip home, so it goes back to the pool",
                            poi.as_deref().unwrap_or("its POI")
                        ),
                        inputs: serde_json::json!({
                            "from_fraction": fraction,
                            "at_reserve": at_reserve,
                            "poi": poi,
                            "released": released,
                        }),
                        command: None,
                    });
                }
            }
            self.stop_flying_to_poi(lead);
        }
    }

    /// Release `drone_id`'s POI redirect, so it stops flying to a POI it no
    /// longer holds.
    fn stop_flying_to_poi(&mut self, drone_id: &str) {
        if self.redirected.remove(drone_id).is_some() {
            self.override_commands
                .push(OverrideCommand::ClearPoi(drone_id.to_string()));
        }
    }

    /// [`PlanningPolicy::EfficientV1`]: every free drone against every
    /// pending POI at once. A drone's cost for a POI is the energy to fly
    /// there and hover through its dwell, less [`PRIORITY_WORTH_WH`] per step
    /// of the POI's rescored priority. A POI is infeasible for a drone that
    /// would then hold less than its dynamic reserve to get home from it (the
    /// static reserve without a home). Drones already on a POI keep it.
    fn allocate_fleet_wide(
        &mut self,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
    ) -> Vec<TeamJoin> {
        // By id, so the solver's lower-index tie-break is the lower id
        // whatever order the bundle declares drones and POIs in.
        let mut pending = self.pois.pending_pois();
        pending.sort_by(|a, b| a.poi_id.cmp(&b.poi_id).then(a.index.cmp(&b.index)));
        let capacity_wh = battery.model().capacity_wh;
        let fraction_of = |id: &str| {
            battery
                .remaining_wh(id)
                .map_or(1.0, |remaining| remaining / capacity_wh)
        };
        self.hand_off_safely(mobile, battery);
        let threshold = self.config.handoff_low_fraction;
        let low = |id: &str| fraction_of(id) < threshold;
        // A drone below the handoff threshold takes nothing new: it has just
        // given its POI up, or would be giving it up.
        let mut free: Vec<&(String, [f64; 3])> = mobile
            .iter()
            .filter(|(id, _)| {
                self.pois.assignment_for(id).is_none()
                    && !low(id)
                    && !self.homeward.contains(id.as_str())
            })
            .collect();
        free.sort_by(|a, b| a.0.cmp(&b.0));
        if free.is_empty() {
            return Vec::new();
        }
        let model = battery.model();
        let wind = battery.wind_mps();
        let cost: Vec<Vec<Option<f64>>> = free
            .iter()
            .map(|(id, position)| {
                let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                pending
                    .iter()
                    .map(|poi| {
                        let poi_position = &poi.position;
                        let spent_wh = self.trip_cost_wh(&model, *position, *poi_position)
                            + model.hover_power_w * poi.required_dwell_s.max(0.0) / 3600.0;
                        let reserve_wh = self.reserve_for(&model, id, *poi_position, wind);
                        (remaining_wh - spent_wh >= reserve_wh)
                            .then_some(spent_wh - PRIORITY_WORTH_WH * poi.priority as f64)
                    })
                    .collect()
            })
            .collect();
        if !pending.is_empty() {
            for (row, column) in min_cost_assignment(&cost).into_iter().enumerate() {
                if let Some(column) = column {
                    let assigned = self.pois.assign_at(&free[row].0, pending[column].index);
                    debug_assert!(assigned, "a free drone takes a pending POI");
                }
            }
        }

        // Teams: a drone still free joins a POI being inspected, so both
        // credit its dwell, when that helps and it can afford it. It helps if
        // the drone arrives before the POI's holders would finish it alone
        // (they credit the remaining dwell together). Only a POI whose
        // holders are all already in dwell range takes a teammate: one still
        // en route would leave the newcomer hovering alone for time the
        // energy estimate does not charge. It can
        // afford it if, after the trip and hovering until the POI is done
        // with it, it still has its reserve home. One new teammate per POI
        // per run, chosen by the same solver on trip-plus-hover energy.
        let free: Vec<&(String, [f64; 3])> = free
            .into_iter()
            .filter(|(id, _)| self.pois.assignment_for(id).is_none())
            .collect();
        let in_range = |poi: &argusnet_planning::poi::ActivePoi| {
            let holders: Vec<&(String, [f64; 3])> = mobile
                .iter()
                .filter(|(id, _)| self.pois.assignment_for(id) == Some(poi.poi_id.as_str()))
                .collect();
            holders.len() == poi.members
                && holders.iter().all(|(id, position)| {
                    // A holder about to hand the POI on leaves it this run,
                    // and the newcomer would hover alone until its receiver
                    // arrived.
                    fraction_of(id) >= self.config.handoff_low_fraction && {
                        let (dx, dy) =
                            (poi.position[0] - position[0], poi.position[1] - position[1]);
                        (dx * dx + dy * dy).sqrt() <= POI_DWELL_RANGE_M
                    }
                })
        };
        let mut active: Vec<_> = self
            .pois
            .active_pois()
            .into_iter()
            .filter(in_range)
            .collect();
        active.sort_by(|a, b| a.poi_id.cmp(&b.poi_id).then(a.index.cmp(&b.index)));
        if free.is_empty() || active.is_empty() {
            return Vec::new();
        }
        let speed = self.config.redirect_speed_mps.max(f64::EPSILON);
        let cost: Vec<Vec<Option<f64>>> = free
            .iter()
            .map(|(id, position)| {
                let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                active
                    .iter()
                    .map(|poi| {
                        let alone_s = poi.remaining_dwell_s / poi.members.max(1) as f64;
                        let arrive_s = self.trip_length_m(*position, poi.position) / speed;
                        if arrive_s >= alone_s {
                            return None;
                        }
                        // The holders keep crediting while it travels; it
                        // shares only what is left when it arrives.
                        let left_s = poi.remaining_dwell_s - poi.members as f64 * arrive_s;
                        let together_s = left_s.max(0.0) / (poi.members + 1) as f64;
                        let spent_wh = self.trip_cost_wh(&model, *position, poi.position)
                            + model.hover_power_w * together_s / 3600.0;
                        (remaining_wh - spent_wh
                            >= self.reserve_for(&model, id, poi.position, wind))
                        .then_some(spent_wh)
                    })
                    .collect()
            })
            .collect();
        let candidates: Vec<String> = active.iter().map(|poi| poi.poi_id.clone()).collect();
        let mut joins = Vec::new();
        for (row, column) in min_cost_assignment(&cost).into_iter().enumerate() {
            if let Some(column) = column {
                let poi = &active[column];
                let joined = self.pois.team_assign_at(&free[row].0, poi.index);
                debug_assert!(joined, "a free drone joins an active POI");
                joins.push(TeamJoin {
                    drone_id: free[row].0.clone(),
                    poi_id: poi.poi_id.clone(),
                    members: poi.members,
                    remaining_dwell_s: poi.remaining_dwell_s,
                    candidates: candidates.clone(),
                });
            }
        }
        joins
    }

    /// The POI `drone_id` was last redirected to, if it still holds one.
    pub fn redirected_poi(&self, drone_id: &str) -> Option<&str> {
        self.redirected.get(drone_id).map(String::as_str)
    }

    /// The last run's decisions, in the order made.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Take the override writes the last run queued, in order.
    pub fn take_override_commands(&mut self) -> Vec<OverrideCommand> {
        std::mem::take(&mut self.override_commands)
    }

    /// What the last run reported, for the replay writer.
    pub fn report(&self) -> &DriverReport {
        &self.report
    }

    fn publish_report(&mut self) {
        let position_of = |poi_id: &str| {
            self.initial_pois
                .iter()
                .find(|poi| poi.poi_id == poi_id)
                .map(|poi| poi.position)
                .expect("a status is for one of the manager's POIs")
        };
        self.report = DriverReport {
            pois: self
                .pois
                .statuses()
                .into_iter()
                .map(|status| {
                    let position = position_of(&status.poi_id);
                    (status, position)
                })
                .collect(),
            completed_count: self.pois.completed_count(),
            total_count: self.pois.total_count(),
            coordinator_drone_id: self.coordinator.clone(),
        };
    }

    /// Replace stage 8's localization with `ticks`, indexed by tick number.
    /// See [`LocalizationReplay`].
    pub fn load_localization_replay(&mut self, ticks: Vec<RecordedLocalization>) {
        self.replay = LocalizationReplay { ticks };
    }

    pub fn config(&self) -> &MissionDriverConfig {
        &self.config
    }

    /// The POIs as constructed, before any rescore.
    pub fn pois(&self) -> &[InspectionPoi] {
        &self.initial_pois
    }

    /// The mission machine this stage drives. Stages 2 and 8 read its phase
    /// through the views lent to them ([`crate::mission::stage_nine_phase`]).
    pub fn machine(&self) -> &MissionStage {
        &self.machine
    }

    pub fn machine_mut(&mut self) -> &mut MissionStage {
        &mut self.machine
    }

    fn pending_poi_ids(&self) -> Vec<String> {
        self.pois
            .statuses()
            .into_iter()
            .filter(|s| s.state == PoiState::Pending)
            .map(|s| s.poi_id)
            .collect()
    }

    fn inspect(
        &mut self,
        timestamp_s: f64,
        dt_s: f64,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
        mapping: &MappingStage,
    ) {
        if !self.rescored {
            let radius = self.config.rescore_radius_m;
            self.pois
                .rescore_from_map(|xy| Some(coverage_in_region(mapping, xy, radius)));
            self.rescored = true;
            self.decisions.push(Decision {
                kind: "poi_rescore",
                status: DecisionStatus::Applied,
                subject: None,
                selected: None,
                candidates: self.pending_poi_ids(),
                reason: "coverage_density",
                detail: format!(
                    "pending POIs reprioritised by mapped coverage within {radius} m: sparse \
                     areas rank higher"
                ),
                inputs: serde_json::json!({ "rescore_radius_m": radius }),
                command: None,
            });
        }

        let capacity_wh = battery.model().capacity_wh;
        let fraction_of = |id: &str| {
            battery
                .remaining_wh(id)
                .map_or(1.0, |remaining| remaining / capacity_wh)
        };
        let mut unassigned: Vec<String> = Vec::new();
        let efficient = self.policy == PlanningPolicy::EfficientV1;
        let mut pending_before = self.pending_poi_ids();
        let mut joins = Vec::new();
        if efficient {
            // In the order the allocator considers them, by id.
            pending_before.sort();
            joins = self.allocate_fleet_wide(mobile, battery);
        }
        for join in &joins {
            self.decisions.push(Decision {
                kind: "team_assignment",
                status: DecisionStatus::Applied,
                subject: Some(join.drone_id.clone()),
                selected: Some(join.poi_id.clone()),
                candidates: join.candidates.clone(),
                reason: "fleet_team_join",
                detail: format!(
                    "{} joins {}: free after allocation, it arrives before the {} drone(s) \
                     holding it would finish and can afford the dwell and the trip home",
                    join.drone_id, join.poi_id, join.members
                ),
                inputs: serde_json::json!({
                    "members": join.members,
                    "remaining_dwell_s": join.remaining_dwell_s,
                }),
                command: None,
            });
        }
        for (id, position) in mobile {
            let pending = if efficient {
                pending_before.clone()
            } else {
                self.pending_poi_ids()
            };
            let remaining_wh = battery.remaining_wh(id).unwrap_or(capacity_wh);
            let assigned = if efficient {
                self.pois.assignment_for(id).is_some()
            } else {
                let context = PoiAssignmentContext::new(id, *position, remaining_wh, capacity_wh);
                self.pois.assign_energy_aware(&context).is_some()
            };
            if !assigned && self.pois.assignment_for(id).is_none() {
                unassigned.push(id.clone());
            }
            // Redirect on a NEW assignment only.
            if let Some(active) = self.pois.assignment_for(id).map(str::to_string) {
                if self.redirected.get(id) != Some(&active) {
                    self.redirected.insert(id.clone(), active.clone());
                    let poi_position = self
                        .initial_pois
                        .iter()
                        .find(|p| p.poi_id == active)
                        .expect("an assigned POI is one of the manager's")
                        .position;
                    let flight = self.flight(*position, poi_position, timestamp_s);
                    let mut inputs = serde_json::json!({
                        "position_m": position,
                        "remaining_wh": remaining_wh,
                        "capacity_wh": capacity_wh,
                    });
                    flight.record(&mut inputs);
                    self.override_commands
                        .push(OverrideCommand::SetPoi(id.clone(), flight.trajectory));
                    self.decisions.push(Decision {
                        kind: "poi_assignment",
                        status: DecisionStatus::QueuedForNextTick,
                        subject: Some(id.clone()),
                        selected: Some(active.clone()),
                        candidates: joins
                            .iter()
                            .find(|join| &join.drone_id == id)
                            .map_or(pending, |join| join.candidates.clone()),
                        reason: if joins.iter().any(|join| &join.drone_id == id) {
                            "fleet_team_join"
                        } else if efficient {
                            "fleet_min_cost"
                        } else {
                            "energy_aware_priority"
                        },
                        detail: if joins.iter().any(|join| &join.drone_id == id) {
                            format!("{id} joins {active}'s team: both credit its dwell")
                        } else if efficient {
                            format!(
                                "{id} takes {active}: the fleet's least-cost assignment, \
                                 trip and dwell energy less priority, within the reserve \
                                 home"
                            )
                        } else {
                            format!(
                                "{id} takes {active}: the highest-priority pending POI it can \
                                 reach within its usable energy"
                            )
                        },
                        inputs,
                        command: Some("set_poi"),
                    });
                }
            }
            // Handoff: a low drone with a POI passes it to the first fresh,
            // free drone in fleet order.
            // Under efficient_v1 handoffs were made safely before allocation
            // (`hand_off_safely`); this is the reference's.
            if !efficient
                && fraction_of(id) < self.config.handoff_low_fraction
                && self.pois.assignment_for(id).is_some()
            {
                let considered: Vec<String> = mobile
                    .iter()
                    .map(|(other, _)| other.clone())
                    .filter(|other| other != id)
                    .collect();
                for (other, _) in mobile {
                    if other != id
                        && fraction_of(other) > self.config.handoff_fresh_fraction
                        && self.pois.assignment_for(other).is_none()
                    {
                        let poi = self.pois.assignment_for(id).map(str::to_string);
                        let handed = self.pois.trigger_handoff(id, other);
                        self.decisions.push(Decision {
                            kind: "poi_handoff",
                            status: if handed {
                                DecisionStatus::Applied
                            } else {
                                DecisionStatus::Rejected
                            },
                            subject: Some(id.clone()),
                            selected: Some(other.clone()),
                            candidates: considered,
                            reason: "low_battery",
                            detail: format!(
                                "{id} below {} of capacity hands {} to {other}, the first \
                                 free drone above {}",
                                self.config.handoff_low_fraction,
                                poi.as_deref().unwrap_or("its POI"),
                                self.config.handoff_fresh_fraction
                            ),
                            inputs: serde_json::json!({
                                "from_fraction": fraction_of(id),
                                "to_fraction": fraction_of(other),
                                "poi": poi,
                            }),
                            command: None,
                        });
                        break;
                    }
                }
            }
            self.pois.accumulate_dwell(id, *position, dt_s);
        }
        // Team assignment: one POI left and more than one free drone.
        let pending: Vec<String> = self
            .pois
            .statuses()
            .into_iter()
            .filter(|s| s.state == PoiState::Pending)
            .map(|s| s.poi_id)
            .collect();
        if !efficient && pending.len() == 1 && unassigned.len() > 1 {
            let teamed = self.pois.request_team_assign(&pending[0], &unassigned[1]);
            self.decisions.push(Decision {
                kind: "team_assignment",
                status: if teamed {
                    DecisionStatus::Applied
                } else {
                    DecisionStatus::Rejected
                },
                subject: Some(unassigned[1].clone()),
                selected: Some(pending[0].clone()),
                candidates: unassigned.clone(),
                reason: "one_pending_poi_several_free_drones",
                detail: "the reference asks to team on a pending POI, which it only grants \
                         for an active one"
                    .to_string(),
                inputs: serde_json::json!({ "free_drones": unassigned.len() }),
                command: None,
            });
        }
        for done in self.pois.check_completions(timestamp_s) {
            let finished: Vec<String> = self
                .redirected
                .iter()
                .filter(|(_, poi)| **poi == done)
                .map(|(drone, _)| drone.clone())
                .collect();
            for drone in finished {
                self.redirected.remove(&drone);
                self.override_commands
                    .push(OverrideCommand::ClearPoi(drone.clone()));
                self.decisions.push(Decision {
                    kind: "poi_complete",
                    status: DecisionStatus::QueuedForNextTick,
                    subject: Some(drone.clone()),
                    selected: Some(done.clone()),
                    candidates: Vec::new(),
                    reason: "dwell_satisfied",
                    detail: format!("{done} inspected; {drone}'s redirect is released"),
                    inputs: serde_json::json!({ "timestamp_s": timestamp_s }),
                    command: Some("clear_poi"),
                });
            }
        }
    }
}

/// `mobile_states`: flown positions, fleet order.
fn mobile_positions(context: &TickContext<'_>) -> Vec<(String, [f64; 3])> {
    context
        .world
        .mobile_platforms()
        .map(|p| {
            (
                p.node_id.clone(),
                [p.position.x, p.position.y, p.position.z],
            )
        })
        .collect()
}

impl StageHandler for MissionDriverStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let timestamp_s = context.timestamp_s;
        // Stage 1's ledger, as it left it this tick, through the lent views
        // (SESSION_OWNERSHIP.md §2a). A run with none reads as an empty one:
        // every drone off the ledger, counted as full.
        // Whatever a run that no handoff drained left behind is dropped, so a
        // driver installed without stage 2 does not grow without bound.
        self.override_commands.clear();
        self.decisions.clear();
        if let Some(obstacles) = &self.obstacles {
            obstacles.this_run.borrow_mut().clear();
        }
        let empty = BatteryStage::new();
        let battery = stage_one_battery(context.stages).unwrap_or(&empty);
        // Stage 8's map as it left it this tick, through the lent views.
        let mapping = context
            .stages
            .get::<MappingStage>(Stage::UpdateMappingAndLocalization)
            .ok_or_else(|| {
                StageError::Invariant("stage 9 needs stage 8's map installed".to_string())
            })?;
        let phase = self.machine.phase();
        let mobile = mobile_positions(context);

        // Coordinator election, once, on the first scanning tick that has a
        // fleet. The reference reads the batteries as stage 1 left them.
        if self.coordinator.is_none() && phase == MissionPhase::Scanning && !mobile.is_empty() {
            let ids: Vec<String> = mobile.iter().map(|(id, _)| id.clone()).collect();
            let capacity_wh = battery.model().capacity_wh;
            self.coordinator = elect_coordinator(&ids, battery, capacity_wh);
            let fractions = serde_json::Value::Object(
                ids.iter()
                    .map(|id| {
                        (
                            id.clone(),
                            serde_json::json!(battery
                                .remaining_wh(id)
                                .map(|wh| wh / capacity_wh.max(1.0))),
                        )
                    })
                    .collect(),
            );
            self.decisions.push(Decision {
                kind: "coordinator_election",
                status: DecisionStatus::Applied,
                subject: None,
                selected: self.coordinator.clone(),
                candidates: ids,
                reason: "highest_battery_fraction",
                detail: "the fullest battery, first in fleet order on a tie; off the ledger \
                         counts as full"
                    .to_string(),
                inputs: serde_json::json!({ "battery_fractions": fractions }),
                command: None,
            });
        }

        if self.policy == PlanningPolicy::EfficientV1 {
            self.send_low_drones_home(&mobile, battery, timestamp_s);
        }
        if phase == MissionPhase::Inspecting {
            // Dwell is credited per boundary: the boundaries this lane sat
            // out were credited in `not_due`, each at the position then.
            self.inspect(
                timestamp_s,
                context.boundary_dt_s,
                &mobile,
                battery,
                mapping,
            );
        }

        let home_distances_m = mobile
            .iter()
            .map(|(id, position)| {
                self.homes.get(id).map(|home| {
                    let (dx, dy) = (home[0] - position[0], home[1] - position[1]);
                    (dx * dx + dy * dy).sqrt()
                })
            })
            .collect();
        let (localization_confidences, localizer_timed_out) = if self.replay.is_loaded() {
            let recorded = self.replay.at(context.tick).ok_or_else(|| {
                StageError::Invariant(format!(
                    "localization replay has no entry for tick {}",
                    context.tick
                ))
            })?;
            (recorded.confidences, recorded.timed_out)
        } else {
            (mapping.confidences(), mapping.localizer_timed_out())
        };
        // The gap search scans the whole grid. The reference runs it only on
        // the scanning gate -- while scanning, once coverage has reached the
        // threshold (`sim.py`) -- and that is the only place the machine
        // reads it, so it is computed there and nowhere else (P4.10c: it was
        // ~45 % of a scan scenario's tick when run every tick).
        // In a run with no prior map, progress is the platforms' own map: the
        // share of it measured, and the share unmeasured and enclosed (holes
        // inside the map), not the footprint bookkeeping.
        let belief = mapping.belief_snapshot();
        let coverage_fraction = match &belief {
            Some(belief) => belief.known_fraction(),
            None => mapping.coverage_fraction(),
        };
        let gate_open = phase == MissionPhase::Scanning
            && coverage_fraction >= self.machine.config().scan_coverage_threshold;
        // The holes themselves, for the gap filler (P4), in a run with a
        // prior map; their share is the gate's input.
        let holes = (gate_open && belief.is_none() && self.policy == PlanningPolicy::EfficientV1)
            .then(|| gap_cells(mapping));
        let gap_cell_fraction = gate_open.then(|| match (&belief, &holes) {
            (Some(belief), _) => {
                belief.enclosed_unexplored_fraction(crate::frontier::UNEXPLORED_SHARE)
            }
            (None, Some((cells, total))) => cells.len() as f64 / (*total).max(1) as f64,
            (None, None) => gap_cell_fraction(mapping),
        });
        let all_pois_complete = self.pois.all_complete();
        let gate_inputs = serde_json::json!({
            "coverage_fraction": coverage_fraction,
            "gap_cell_fraction": gap_cell_fraction,
            "mobile_count": mobile.len(),
            "localization_confidences": &localization_confidences,
            "localizer_timed_out": localizer_timed_out,
            "all_pois_complete": all_pois_complete,
        });
        self.machine.set_inputs(MissionInputs {
            timestamp_s,
            coverage_fraction,
            gap_cell_fraction,
            mobile_count: mobile.len(),
            localization_confidences,
            localizer_timed_out,
            all_pois_complete,
            home_distances_m,
        });
        let outcome = self.machine.run(context)?;

        for event in self.machine.take_events() {
            match event {
                MissionEvent::EgressRoutesRequested { at_s } => {
                    for (id, _) in &mobile {
                        self.override_commands
                            .push(OverrideCommand::ClearPoi(id.clone()));
                    }
                    self.redirected.clear();
                    // Routes around obstacles, by drone, for the journal;
                    // only drones whose way home turns.
                    let mut routes = BTreeMap::new();
                    for (id, position) in &mobile {
                        if let Some(home) = self.homes.get(id).copied() {
                            let flight = self.flight(*position, home, at_s);
                            if let Some((length_m, corners)) = flight.route {
                                routes.insert(
                                    id.clone(),
                                    serde_json::json!({
                                        "route_length_m": length_m,
                                        "route_waypoints": corners,
                                    }),
                                );
                            }
                            self.override_commands
                                .push(OverrideCommand::SetEgress(id.clone(), flight.trajectory));
                        }
                    }
                    let mut inputs = gate_inputs.clone();
                    if let (false, Some(fields)) = (routes.is_empty(), inputs.as_object_mut()) {
                        fields.insert("routes".into(), serde_json::json!(routes));
                    }
                    context.note("egress routes set");
                    self.decisions.push(Decision {
                        kind: "egress",
                        status: DecisionStatus::QueuedForNextTick,
                        subject: None,
                        selected: None,
                        candidates: mobile
                            .iter()
                            .filter(|(id, _)| self.homes.contains_key(id))
                            .map(|(id, _)| id.clone())
                            .collect(),
                        reason: "mission_requested_egress",
                        detail: "every POI redirect released; each drone with a station is \
                                 routed home"
                            .to_string(),
                        inputs,
                        command: Some("set_egress"),
                    });
                }
                MissionEvent::EgressRoutesCleared { .. } => {
                    self.override_commands.push(OverrideCommand::ClearAllEgress);
                    self.decisions.push(Decision {
                        kind: "egress_cleared",
                        status: DecisionStatus::QueuedForNextTick,
                        subject: None,
                        selected: None,
                        candidates: Vec::new(),
                        reason: "egress_complete",
                        detail: "every recall dropped at once".to_string(),
                        inputs: gate_inputs.clone(),
                        command: Some("clear_all_egress"),
                    });
                }
                MissionEvent::PhaseChanged { from, to, .. } => {
                    self.decisions.push(Decision {
                        kind: "phase_change",
                        status: DecisionStatus::Applied,
                        subject: None,
                        selected: Some(to.to_string()),
                        candidates: Vec::new(),
                        reason: "mission_gate",
                        detail: format!("{from} -> {to}"),
                        inputs: gate_inputs.clone(),
                        command: None,
                    });
                }
                MissionEvent::LocalizationTimedOut { .. } => {
                    self.decisions.push(Decision {
                        kind: "localization_timeout",
                        status: DecisionStatus::Applied,
                        subject: None,
                        selected: None,
                        candidates: Vec::new(),
                        reason: "localizer_timed_out",
                        detail: "the localizer gave up; the mission continues without it"
                            .to_string(),
                        inputs: gate_inputs.clone(),
                        command: None,
                    });
                }
                MissionEvent::RescorePois { .. } => {}
            }
        }
        if self.policy == PlanningPolicy::EfficientV1 {
            self.fill_gaps(holes, gap_cell_fraction, &mobile, mapping, timestamp_s);
        }
        self.publish_report();
        Ok(outcome)
    }

    /// The machine, the POI manager, which drone was last redirected where,
    /// and the rescore latch. The POIs themselves and the homes are scenario,
    /// rebuilt by assembly -- and checked against, below.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let state = DriverCheckpoint {
            machine: self.machine.state(),
            pois: self.pois.clone(),
            redirected: self.redirected.clone(),
            rescored: self.rescored,
            coordinator: self.coordinator.clone(),
            policy: self.policy,
            homeward: self.homeward.clone(),
            gap_fill: self.gap_fill.clone(),
        };
        Some(serde_json::to_value(state).expect("mission driver state serialises"))
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: DriverCheckpoint = crate::simulation::decode_stage_state(state)?;
        let ids = |statuses: Vec<argusnet_planning::poi::PoiStatus>| {
            statuses.into_iter().map(|s| s.poi_id).collect::<Vec<_>>()
        };
        if ids(decoded.pois.statuses()) != ids(self.pois.statuses()) {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: "the checkpoint's POIs are not this run's".into(),
            });
        }
        if decoded.policy != self.policy {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: format!(
                    "the checkpoint allocated POIs under {}, this run under {}",
                    decoded.policy.as_str(),
                    self.policy.as_str()
                ),
            });
        }
        Ok(())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: DriverCheckpoint = crate::simulation::decode_stage_state(state)?;
        self.machine.restore_machine(decoded.machine);
        self.pois = decoded.pois;
        self.redirected = decoded.redirected;
        self.rescored = decoded.rescored;
        self.coordinator = decoded.coordinator;
        self.homeward = decoded.homeward;
        self.gap_fill = decoded.gap_fill;
        self.override_commands.clear();
        self.decisions.clear();
        self.publish_report();
        Ok(())
    }

    fn not_due(&mut self, context: &mut TickContext<'_>) -> Result<(), StageError> {
        // This tick's outputs: the decisions it records and the override
        // commands stage 2 takes next tick. A boundary it sat out made none,
        // but for a return home under `efficient_v1` (below).
        // The machine's events are taken where they are made, in `run`.
        self.override_commands.clear();
        self.decisions.clear();
        // Dwell is residence, so it is credited at every boundary, where each
        // drone is then, to the POI it holds then. Credited in bulk at the
        // next run it would count a drone that arrived or left in between for
        // the whole interval, and a POI assigned at that run for time before
        // it was assigned. Assignment and completion wait for the lane's run.
        let mobile = mobile_positions(context);
        if self.machine.phase() == MissionPhase::Inspecting {
            for (id, position) in &mobile {
                self.pois
                    .accumulate_dwell(id, *position, context.boundary_dt_s);
            }
        }
        // A drone that reaches its reserve between the lane's runs turns for
        // home at once, not at the next run: what it would fly meanwhile is
        // the energy that gets it home. Its POI goes at the next run, which
        // treats a homeward drone as giving it up.
        if self.policy == PlanningPolicy::EfficientV1 {
            let empty = BatteryStage::new();
            let battery = stage_one_battery(context.stages).unwrap_or(&empty);
            self.send_low_drones_home(&mobile, battery, context.timestamp_s);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.machine.reset_machine();
        self.pois = PoiManager::new(self.initial_pois.clone());
        self.redirected.clear();
        self.rescored = false;
        self.coordinator = None;
        self.homeward.clear();
        self.gap_fill.clear();
        self.override_commands.clear();
        self.decisions.clear();
        self.publish_report();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_mapping::GridBounds;

    /// A one-drone run with stage 8 and this driver installed, and
    /// localization replayed so only the gate under test decides the phase.
    fn gate_rig(replay: Vec<RecordedLocalization>) -> crate::simulation::Simulation {
        gate_rig_with(replay, Vec::new())
    }

    /// [`gate_rig`], with inspection POIs.
    fn gate_rig_with(
        replay: Vec<RecordedLocalization>,
        pois: Vec<InspectionPoi>,
    ) -> crate::simulation::Simulation {
        use crate::config::SimulationConfig;
        use crate::stage::Stage;
        use argusnet_types::{NodeState, Vector3};

        let mut sim = crate::simulation::Simulation::bare(SimulationConfig {
            seed: 3,
            dt_s: 1.0,
            steps: Some(500),
            run_id: "gate".to_string(),
        })
        .expect("config");
        sim.load_world(
            vec![NodeState {
                node_id: "drone-1".into(),
                position: Vector3::new(25.0, 25.0, 80.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                is_mobile: true,
                timestamp_s: 0.0,
                health: 1.0,
                ..Default::default()
            }],
            Vec::new(),
        );
        // 10 x 10 cells of 50 m.
        let bounds = GridBounds {
            x_min_m: 0.0,
            x_max_m: 500.0,
            y_min_m: 0.0,
            y_max_m: 500.0,
            resolution_m: 50.0,
        };
        let mut mapping = crate::mapping::MappingStage::new(bounds).expect("bounds");
        mapping.set_footprint_radius_m(40.0);
        let mut stage = MissionDriverStage::new(
            MissionDriverConfig {
                mission: MissionConfig {
                    scan_coverage_threshold: 0.3,
                    ..MissionConfig::default()
                },
                ..MissionDriverConfig::default()
            },
            pois,
            BTreeMap::new(),
        );
        stage.load_localization_replay(replay);
        mapping.follow_mission_phase();
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
            .expect("stage 8");
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
            .expect("stage 9");
        sim
    }

    fn installed_driver(sim: &crate::simulation::Simulation) -> &MissionDriverStage {
        sim.stage(crate::stage::Stage::EvaluateMissionTransitions)
            .expect("stage 9 is a driver")
    }

    /// Stage 9's machine, reached through the session that owns it.
    fn machine(sim: &crate::simulation::Simulation) -> &MissionStage {
        MissionStage::of(sim).expect("stage 9 drives a mission")
    }

    fn converged(ticks: usize) -> Vec<RecordedLocalization> {
        vec![
            RecordedLocalization {
                confidences: vec![1.0],
                timed_out: false,
            };
            ticks
        ]
    }

    /// The gap-fill gate: coverage above the threshold is not enough while an
    /// enclosed hole of at least 2% of the grid remains. The 12 cells around a
    /// 2 x 2 hole (4%) are flown FIRST, so the hole is enclosed long before
    /// coverage reaches the 30% threshold -- an unenclosed hole is connected
    /// to the uncovered border and is legitimately not a gap. Then the rest.
    /// The mission must stay in `Scanning` until the hole is flown too.
    #[test]
    fn an_enclosed_hole_holds_the_mission_in_scanning() {
        let in_hole = |i: i32, j: i32| (4..=5).contains(&i) && (4..=5).contains(&j);
        let around_hole = |i: i32, j: i32| (3..=6).contains(&i) && (3..=6).contains(&j);
        let cells: Vec<(i32, i32)> = (0..10).flat_map(|i| (0..10).map(move |j| (i, j))).collect();
        let ring: Vec<(f64, f64)> = cells
            .iter()
            .filter(|(i, j)| around_hole(*i, *j) && !in_hole(*i, *j))
            .chain(cells.iter().filter(|(i, j)| !around_hole(*i, *j)))
            .map(|(i, j)| (25.0 + 50.0 * *i as f64, 25.0 + 50.0 * *j as f64))
            .collect();
        assert_eq!(ring.len(), 96);
        let hole = [
            (225.0, 225.0),
            (275.0, 225.0),
            (225.0, 275.0),
            (275.0, 275.0),
        ];
        let mut sim = gate_rig(converged(ring.len() + hole.len()));
        for (x, y) in &ring {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = *x;
            drone.position.y = *y;
            sim.step().expect("tick");
            assert_eq!(
                machine(&sim).phase(),
                MissionPhase::Scanning,
                "left Scanning with the hole still open, at ({x}, {y})"
            );
        }
        // Fill the hole a cell at a time: 3%, 2%, then 1% remains. The gate
        // holds while the gap is >= 2% -- including exactly 2% -- and releases
        // on the third cell. (With no POIs, the fourth tick would go straight
        // on to egress, so the check stops at the release.)
        let expected = [
            MissionPhase::Scanning,
            MissionPhase::Scanning,
            MissionPhase::Inspecting,
        ];
        for ((x, y), phase) in hole.iter().zip(expected) {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = *x;
            drone.position.y = *y;
            sim.step().expect("tick");
            assert_eq!(
                machine(&sim).phase(),
                phase,
                "after flying hole cell ({x}, {y})"
            );
        }
    }

    /// The ring of [`an_enclosed_hole_holds_the_mission_in_scanning`], flown
    /// under `policy`: every decision recorded on the way, by tick.
    fn fly_the_ring(policy: PlanningPolicy) -> (crate::simulation::Simulation, Vec<Decision>) {
        let in_hole = |i: i32, j: i32| (4..=5).contains(&i) && (4..=5).contains(&j);
        let around_hole = |i: i32, j: i32| (3..=6).contains(&i) && (3..=6).contains(&j);
        let cells: Vec<(i32, i32)> = (0..10).flat_map(|i| (0..10).map(move |j| (i, j))).collect();
        let ring: Vec<(f64, f64)> = cells
            .iter()
            .filter(|(i, j)| around_hole(*i, *j) && !in_hole(*i, *j))
            .chain(cells.iter().filter(|(i, j)| !around_hole(*i, *j)))
            .map(|(i, j)| (25.0 + 50.0 * *i as f64, 25.0 + 50.0 * *j as f64))
            .collect();
        let mut sim = gate_rig(converged(ring.len() + 8));
        sim.stage_mut::<MissionDriverStage>(crate::stage::Stage::EvaluateMissionTransitions)
            .expect("stage 9")
            .set_policy(policy)
            .expect("no POIs to repeat");
        let mut decisions = Vec::new();
        for (x, y) in &ring {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = *x;
            drone.position.y = *y;
            sim.step().expect("tick");
            decisions.extend(installed_driver(&sim).decisions().iter().cloned());
        }
        (sim, decisions)
    }

    /// P4: under efficient_v1 a hole that holds the scan open is swept by a
    /// free drone, from its nearer end; the reference only waits.
    #[test]
    fn efficient_v1_sends_a_drone_to_sweep_an_enclosed_hole() {
        let (_, reference) = fly_the_ring(PlanningPolicy::Reference);
        assert!(reference.iter().all(|d| d.kind != "gap_fill"));

        let (sim, efficient) = fly_the_ring(PlanningPolicy::EfficientV1);
        let fill = efficient
            .iter()
            .find(|d| d.kind == "gap_fill")
            .expect("the hole is assigned");
        assert_eq!(fill.subject.as_deref(), Some("drone-1"));
        assert_eq!(fill.selected.as_deref(), Some("gap:4:4"));
        assert_eq!(fill.inputs["cells"], 4);
        assert_eq!(fill.command, Some("set_frontier"));
        assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);
        // Its sweep is what it holds, and what a checkpoint carries.
        let state = installed_driver(&sim).checkpoint_state().unwrap();
        let held = installed_driver(&sim)
            .gap_fill
            .get("drone-1")
            .map(|f| f.cells.len());
        assert!(held.is_none() || state.get("gap_fill").is_some());
    }

    /// A sweeper that turns home on its reserve lets go of its hole at once
    /// (cleared, so its frontier flight ends), freeing it for another drone
    /// instead of holding it until its sweep would have ended.
    #[test]
    fn a_sweep_is_let_go_when_its_drone_turns_home() {
        let (mut sim, _) = fly_the_ring(PlanningPolicy::EfficientV1);
        assert!(installed_driver(&sim).gap_fill.contains_key("drone-1"));
        sim.stage_mut::<MissionDriverStage>(crate::stage::Stage::EvaluateMissionTransitions)
            .expect("stage 9")
            .homeward
            .insert("drone-1".to_string());
        sim.step().expect("tick");
        assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);
        assert!(installed_driver(&sim).gap_fill.is_empty());
        let release = installed_driver(&sim)
            .decisions()
            .iter()
            .find(|d| d.kind == "gap_fill_release")
            .cloned()
            .expect("released");
        assert_eq!(release.reason, "homeward");
        assert_eq!(release.command, Some("clear_frontier"));
        assert!(
            installed_driver(&sim)
                .decisions()
                .iter()
                .all(|d| d.kind != "gap_fill"),
            "not sent again"
        );
    }

    /// Once the hole is flown the gate opens, and no sweep is held past the
    /// scan.
    ///
    /// The gate opens with one hole cell still uncovered, so the sweep is
    /// still held then: it is cleared (stage 2 would otherwise go on flying
    /// it while localizing), not just forgotten.
    #[test]
    fn a_sweep_is_let_go_once_the_scan_ends() {
        let (mut sim, _) = fly_the_ring(PlanningPolicy::EfficientV1);
        let mut held_before_the_end = false;
        for (x, y) in [
            (225.0, 225.0),
            (275.0, 225.0),
            (225.0, 275.0),
            (275.0, 275.0),
        ] {
            held_before_the_end = installed_driver(&sim).gap_fill.contains_key("drone-1");
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = x;
            drone.position.y = y;
            sim.step().expect("tick");
            if machine(&sim).phase() != MissionPhase::Scanning {
                break;
            }
        }
        assert_ne!(machine(&sim).phase(), MissionPhase::Scanning);
        assert!(installed_driver(&sim).gap_fill.is_empty());
        assert!(held_before_the_end, "a sweep was held when the scan ended");
        let release = installed_driver(&sim)
            .decisions()
            .iter()
            .find(|d| d.kind == "gap_fill_release")
            .cloned()
            .expect("the held sweep is released");
        assert_eq!(release.subject.as_deref(), Some("drone-1"));
        assert_eq!(release.reason, "scan_ended");
        assert_eq!(release.command, Some("clear_frontier"));
    }

    /// Ground 100 m high in a narrow band at x = 520, flat elsewhere.
    #[derive(Debug)]
    struct Ridge;

    impl crate::observe::TerrainHeight for Ridge {
        fn height_at(&self, x_m: f64, _y_m: f64) -> f64 {
            if (x_m - 520.0).abs() < 1.0 {
                100.0
            } else {
                0.0
            }
        }
    }

    /// On a grid terrain the sweep's height is solved, not sampled: a ridge
    /// 1 m wide and 150 m high whose crest (x = 500.5) no 2 m sample from
    /// x = 400 lands on still raises the flight over it, with the
    /// clearance, and the flight's lowest point over the ground is exactly
    /// that clearance.
    #[test]
    fn a_gap_sweep_clears_a_ridge_between_samples() {
        use crate::runtime_world::SharedTerrain;
        use argusnet_types::Bounds2d;
        use argusnet_world::tiled::{Tile, TiledTerrain};
        let heights_m = (0..9)
            .map(|_| {
                (0..9)
                    .map(|col| if col == 1 { 150.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let tile = Tile {
            tx: 125,
            ty: 0,
            lod: 0,
            x_min_m: 500.0,
            y_min_m: 0.0,
            cell_size_m: 0.5,
            heights_m,
            min_pyramid: Vec::new(),
            max_pyramid: Vec::new(),
        };
        let terrain = Arc::new(SharedTerrain::new(
            Arc::new(
                TiledTerrain::new(
                    Bounds2d::new(0.0, 1000.0, 0.0, 1000.0),
                    8,
                    0.5,
                    0.0,
                    vec![tile],
                )
                .unwrap(),
            ),
            0.5,
        ));
        let mut stage = efficient_driver(Vec::new());
        stage.set_known_terrain(Some(terrain.clone()));
        let start = [400.0, 1.0, 40.0];
        let z = stage
            .sweep_height(start, &[[600.0, 1.0]], 40.0)
            .expect("it can climb over");
        let clearance = PlannerConfig::default().drone_clearance_m;
        let (low, _) = crate::ground::lowest_margin(terrain.as_ref(), start, [600.0, 1.0, z]);
        assert!(low >= clearance - 1e-6, "{low}");
        assert!(low < clearance + 1e-3, "no higher than it needs: {low}");
    }

    /// A sweep whose way crosses a ridge between its sample points is flown
    /// high enough to clear it all the way; one that starts below the ground's
    /// clearance is not flown at all.
    #[test]
    fn a_gap_sweep_clears_the_ground_along_its_whole_flight() {
        let mut stage = efficient_driver(Vec::new());
        stage.set_known_terrain(Some(Arc::new(Ridge)));
        let start = [0.0, 0.0, 40.0];
        let path = [[500.0, 0.0], [550.0, 0.0], [1000.0, 0.0]];
        let z = stage
            .sweep_height(start, &path, 40.0)
            .expect("can clear it");
        let clearance = PlannerConfig::default().drone_clearance_m;
        // At the ridge (520 m of 1000 m flown) the linear climb is above it.
        let at_ridge = start[2] + 0.52 * (z - start[2]);
        assert!(at_ridge >= 100.0 + clearance, "{at_ridge}");
        let mut points = vec![[start[0], start[1]]];
        points.extend(path);
        assert!(stage.clears_terrain(&points, start[2], z, 1000.0));
        // Flat ground needs nothing above the base height.
        let mut flat = efficient_driver(Vec::new());
        flat.set_known_terrain(Some(Arc::new(Ridges)));
        assert_eq!(flat.sweep_height(start, &[[100.0, 0.0]], 40.0), Some(40.0));
        // A start already inside the clearance cannot be made safe.
        assert_eq!(stage.sweep_height([520.0, 0.0, 50.0], &path, 40.0), None);
    }

    /// A drone nearer a hole but with no safe flight to it (here it starts
    /// inside the ground's clearance, on the ridge) is not matched to it:
    /// the farther drone that can fly it is, instead of nobody.
    #[test]
    fn a_hole_goes_to_the_nearest_drone_that_can_safely_sweep_it() {
        let mut stage = efficient_driver(Vec::new());
        stage.set_known_terrain(Some(Arc::new(Ridge)));
        let near_but_unsafe = [520.0, 0.0, 50.0];
        let far_and_safe = [0.0, 0.0, 40.0];
        let sweep = vec![[600.0, 0.0], [650.0, 0.0]];
        assert!(stage.gap_plan(near_but_unsafe, &sweep).is_none());
        let assigned = stage.assign_gap_sweeps(
            &[near_but_unsafe, far_and_safe],
            std::slice::from_ref(&sweep),
        );
        assert_eq!(assigned.len(), 1);
        let (row, column, plan) = &assigned[0];
        assert_eq!((*row, *column), (1, 0));
        assert_eq!(plan.points, sweep);
        // Its flight climbs over the ridge on the way.
        assert!(plan.z > 40.0);
    }

    #[test]
    fn holes_cluster_four_connected_and_sweep_column_by_column() {
        let cells = [(1, 1), (1, 2), (2, 2), (5, 5), (2, 1)];
        let clusters = hole_clusters(&cells);
        assert_eq!(
            clusters,
            vec![vec![[1, 1], [1, 2], [2, 1], [2, 2]], vec![[5, 5]]]
        );
        // Up the first column, back down the second.
        assert_eq!(
            sweep_order(&clusters[0]),
            vec![[1, 1], [1, 2], [2, 2], [2, 1]]
        );
        // Diagonal neighbours are separate holes.
        assert_eq!(hole_clusters(&[(0, 0), (1, 1)]).len(), 2);
    }

    /// B2.4: a mission lane below the base rate credits dwell boundary by
    /// boundary, where the drone was at each, not the whole interval at the
    /// drone's position when the lane next runs.
    #[test]
    fn a_lane_that_sits_out_boundaries_credits_dwell_where_the_drone_was() {
        use crate::stage::Stage;

        // Every cell flown, hole last, as in the gate test: Inspecting after
        // the third hole cell. One POI at (475, 475), far from all of them.
        let cells: Vec<(f64, f64)> = (0..10)
            .flat_map(|i| (0..10).map(move |j| (25.0 + 50.0 * i as f64, 25.0 + 50.0 * j as f64)))
            .collect();
        let mut sim = gate_rig_with(
            converged(cells.len() + 8),
            vec![InspectionPoi {
                poi_id: "poi-00".into(),
                name: "poi-00".into(),
                position: [475.0, 475.0, 40.0],
                priority: 1,
                required_dwell_s: 3.0,
            }],
        );
        let fly = |sim: &mut crate::simulation::Simulation, (x, y): (f64, f64)| {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = x;
            drone.position.y = y;
        };
        // Lower-left first, so the last cells flown are well away from the POI.
        for cell in cells.iter().rev() {
            fly(&mut sim, *cell);
            sim.step().expect("tick");
        }
        // One more run, away from the POI: Inspecting, so it is assigned.
        fly(&mut sim, (25.0, 25.0));
        sim.step().expect("tick");
        assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);
        let status =
            |sim: &crate::simulation::Simulation| installed_driver(sim).pois.statuses()[0].clone();
        assert_eq!(
            installed_driver(&sim).pois.assignment_for("drone-1"),
            Some("poi-00")
        );
        assert_eq!(status(&sim).state, PoiState::Active);

        // Stage 9 sits out three boundaries -- away, away, at the POI -- then
        // runs at the POI. 1 s boundaries: 2 s of dwell, short of the 3 s
        // required, not the 4 s since it last ran.
        sim.declare_lane_rate(Stage::EvaluateMissionTransitions, "1/4");
        let boundary = |sim: &mut crate::simulation::Simulation, at: (f64, f64), due: bool| {
            fly(sim, at);
            let mut open = match sim.begin_tick().expect("begin") {
                crate::simulation::TickStart::Open(open) => open,
                crate::simulation::TickStart::Committed(_) => panic!("reset"),
            };
            sim.run_through(&mut open, Stage::UpdateMappingAndLocalization)
                .expect("stages 1-8");
            if due {
                sim.run_through(&mut open, Stage::EvaluateMissionTransitions)
                    .expect("stage 9");
            } else {
                sim.skip_through(&mut open, Stage::EvaluateMissionTransitions, "not due")
                    .expect("stage 9 sits out");
            }
            sim.commit_tick(open).expect("commit");
        };
        boundary(&mut sim, (25.0, 25.0), false);
        boundary(&mut sim, (25.0, 75.0), false);
        boundary(&mut sim, (475.0, 475.0), false);
        boundary(&mut sim, (475.0, 475.0), true);
        assert_eq!(
            status(&sim).state,
            PoiState::Active,
            "the boundaries away from the POI were credited"
        );
        // The third second at the POI completes it, with exactly that dwell.
        boundary(&mut sim, (475.0, 475.0), true);
        let done = status(&sim);
        assert_eq!(done.state, PoiState::Complete);
        assert_eq!(done.dwell_accumulated_s, 3.0);
    }

    /// A timed-out localizer latches in the machine, from the replay too.
    #[test]
    fn a_replayed_localization_timeout_latches() {
        let mut replay = converged(3);
        replay[1].timed_out = true;
        let mut sim = gate_rig(replay);
        sim.step().expect("tick 0");
        assert!(!machine(&sim).state().localization_timed_out());
        sim.step().expect("tick 1");
        assert!(
            machine(&sim).state().localization_timed_out(),
            "the timeout did not latch"
        );
    }

    fn driver(pois: Vec<InspectionPoi>) -> MissionDriverStage {
        MissionDriverStage::new(MissionDriverConfig::default(), pois, BTreeMap::new())
    }

    fn empty_map() -> crate::mapping::MappingStage {
        crate::mapping::MappingStage::new(GridBounds {
            x_min_m: 0.0,
            x_max_m: 1000.0,
            y_min_m: 0.0,
            y_max_m: 1000.0,
            resolution_m: 50.0,
        })
        .expect("bounds")
    }

    fn poi(id: &str, x: f64, priority: i64) -> InspectionPoi {
        InspectionPoi {
            poi_id: id.to_string(),
            name: id.to_string(),
            position: [x, 0.0, 40.0],
            priority,
            required_dwell_s: 15.0,
        }
    }

    /// The team branch is dead in the reference, and stays dead here.
    ///
    /// `run_simulation` calls `request_team_assign(pending_pois[0], ...)` --
    /// on a POI it has just filtered as **pending** -- and `request_team_assign`
    /// refuses anything that is not already **active**. So when one POI is
    /// left pending and two drones are free (here because both are too low to
    /// afford it), nothing is assigned. The driver reproduces the call; this
    /// pins that it changes nothing, so a "fix" that assigned the POI would be
    /// a behaviour change and fail here.
    #[test]
    fn the_reference_team_assignment_branch_assigns_nothing() {
        let mut battery = BatteryStage::new();
        battery.set_model(argusnet_planning::battery::BatteryModel {
            capacity_wh: 500.0,
            ..Default::default()
        });
        for id in ["drone-a", "drone-b", "drone-c"] {
            battery.register(id, None);
        }
        let mut stage = driver(vec![poi("poi-00", 100.0, 2), poi("poi-01", 900.0, 1)]);
        // Below the 20% reserve: no usable energy, so no assignment.
        for id in ["drone-b", "drone-c"] {
            assert!(battery.set_remaining_wh(id, 50.0));
        }
        let mobile = vec![
            ("drone-a".to_string(), [100.0, 0.0, 40.0]),
            ("drone-b".to_string(), [500.0, 0.0, 40.0]),
            ("drone-c".to_string(), [600.0, 0.0, 40.0]),
        ];
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        let pending = stage
            .pois
            .statuses()
            .into_iter()
            .filter(|s| s.state == PoiState::Pending)
            .count();
        assert_eq!(pending, 1, "the setup no longer reaches the team branch");
        assert_eq!(stage.pois.assignment_for("drone-b"), None);
        assert_eq!(stage.pois.assignment_for("drone-c"), None);
    }

    /// `elect_coordinator`: the highest battery fraction; the first in fleet
    /// order on a tie; a drone off the ledger counts as full.
    #[test]
    fn the_coordinator_is_the_fullest_battery_first_on_a_tie() {
        let mut battery = BatteryStage::new();
        battery.register("a", None);
        battery.register("b", None);
        let capacity_wh = battery.model().capacity_wh;
        let ids = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            elect_coordinator(&ids(&["a", "b"]), &battery, capacity_wh),
            Some("a".into())
        );
        battery.set_remaining_wh("a", 0.2 * capacity_wh);
        assert_eq!(
            elect_coordinator(&ids(&["a", "b"]), &battery, capacity_wh),
            Some("b".into())
        );
        battery.set_remaining_wh("b", 0.1 * capacity_wh);
        assert_eq!(
            elect_coordinator(&ids(&["a", "b", "ghost"]), &battery, capacity_wh),
            Some("ghost".into()),
            "off the ledger is full"
        );
        assert_eq!(elect_coordinator(&[], &battery, capacity_wh), None);
    }

    /// Elected once (`_coordinator_elected`): a later scanning tick on which
    /// another drone's battery is fuller does not re-elect.
    #[test]
    fn the_coordinator_is_elected_once() {
        use crate::config::SimulationConfig;
        use crate::stage::Stage;
        use argusnet_types::{NodeState, Vector3};

        let mut sim = crate::simulation::Simulation::bare(SimulationConfig {
            seed: 3,
            dt_s: 1.0,
            steps: Some(10),
            run_id: "coordinator".to_string(),
        })
        .expect("config");
        let drone = |id: &str, x: f64| NodeState {
            node_id: id.into(),
            position: Vector3::new(x, 25.0, 80.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: true,
            timestamp_s: 0.0,
            health: 1.0,
            ..Default::default()
        };
        sim.load_world(
            vec![drone("drone-1", 25.0), drone("drone-2", 475.0)],
            Vec::new(),
        );
        let bounds = GridBounds {
            x_min_m: 0.0,
            x_max_m: 500.0,
            y_min_m: 0.0,
            y_max_m: 500.0,
            resolution_m: 50.0,
        };
        let mapping = crate::mapping::MappingStage::new(bounds).expect("bounds");
        let mut battery = BatteryStage::new();
        battery.register("drone-1", None);
        battery.register("drone-2", None);
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(battery))
            .expect("stage 1");
        let stage =
            MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), BTreeMap::new());
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
            .expect("stage 8");
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
            .expect("stage 9");

        sim.step().expect("tick 0");
        assert_eq!(
            installed_driver(&sim)
                .report()
                .coordinator_drone_id
                .as_deref(),
            Some("drone-1")
        );
        assert!(BatteryStage::of_mut(&mut sim)
            .expect("stage 1 is a battery")
            .set_remaining_wh("drone-1", 10.0));
        sim.step().expect("tick 1");
        assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);
        assert_eq!(
            installed_driver(&sim)
                .report()
                .coordinator_drone_id
                .as_deref(),
            Some("drone-1"),
            "re-elected"
        );
    }

    fn efficient_driver(pois: Vec<InspectionPoi>) -> MissionDriverStage {
        let mut stage = driver(pois);
        stage
            .set_policy(PlanningPolicy::EfficientV1)
            .expect("unique ids");
        stage
    }

    fn fleet(ids: &[(&str, f64)]) -> (BatteryStage, Vec<(String, [f64; 3])>) {
        let mut battery = BatteryStage::new();
        for (id, _) in ids {
            battery.register(*id, None);
        }
        let mobile = ids
            .iter()
            .map(|(id, x)| (id.to_string(), [*x, 0.0, 40.0]))
            .collect();
        (battery, mobile)
    }

    /// P1: the reference hands the first drone in fleet order the best POI,
    /// however far; the fleet-wide planner sends each drone to the near one
    /// when every POI is served either way.
    #[test]
    fn efficient_v1_beats_the_fleet_order_greedy() {
        let pois = || vec![poi("poi-00", 900.0, 2), poi("poi-01", 100.0, 1)];
        let (battery, mobile) = fleet(&[("drone-a", 100.0), ("drone-b", 900.0)]);

        let mut reference = driver(pois());
        reference.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(reference.pois.assignment_for("drone-a"), Some("poi-00"));
        assert_eq!(reference.pois.assignment_for("drone-b"), Some("poi-01"));

        let mut efficient = efficient_driver(pois());
        efficient.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(efficient.pois.assignment_for("drone-a"), Some("poi-01"));
        assert_eq!(efficient.pois.assignment_for("drone-b"), Some("poi-00"));
        let reasons: Vec<&str> = efficient
            .decisions()
            .iter()
            .filter(|d| d.kind == "poi_assignment")
            .map(|d| d.reason)
            .collect();
        assert_eq!(reasons, vec!["fleet_min_cost", "fleet_min_cost"]);
        // Declared poi-00 first here; the record lists them by id, as solved.
        let first = efficient
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .unwrap();
        assert_eq!(first.candidates, vec!["poi-00", "poi-01"]);
        let commands = efficient.take_override_commands();
        assert_eq!(commands.len(), 2, "one redirect per new assignment");
    }

    /// Equal costs go to the lower drone id and the lower POI id, not to
    /// whichever the bundle declared first.
    #[test]
    fn efficient_v1_breaks_ties_by_id() {
        // Equidistant drones, declared z first: the POI goes to drone-a.
        let (battery, mobile) = fleet(&[("drone-z", 600.0), ("drone-a", 400.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        assert!(stage.pois.leads("drone-a"), "the POI is drone-a's");
        // drone-z, left free, waits while drone-a is still en route, then
        // joins it as a teammate once drone-a is in dwell range (P1 teams).
        assert_eq!(stage.pois.assignment_for("drone-z"), None);
        let arrived = vec![
            mobile[0].clone(),
            ("drone-a".to_string(), [500.0, 0.0, 40.0]),
        ];
        stage.inspect(1.0, 1.0, &arrived, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-z"), Some("poi-00"));
        assert!(!stage.pois.leads("drone-z"));
        // Equal POIs, declared poi-01 first: the one drone takes poi-00.
        let (battery, mobile) = fleet(&[("drone-a", 500.0)]);
        let mut stage = efficient_driver(vec![poi("poi-01", 400.0, 1), poi("poi-00", 600.0, 1)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        // The decision records the candidates in the order solved: by id.
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .unwrap();
        assert_eq!(decision.candidates, vec!["poi-00", "poi-01"]);
    }

    /// A drone already below the handoff threshold is given no POI, so the
    /// team stage cannot take the receiver its handoff would need.
    #[test]
    fn efficient_v1_gives_a_low_drone_nothing_new() {
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 520.0)]);
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), None);
        assert!(stage.pois.leads("drone-b"));
    }

    /// No drone joins a POI whose holder is about to hand it on: it would
    /// hover alone until the receiver arrived.
    #[test]
    fn efficient_v1_never_teams_onto_a_poi_being_handed_on() {
        // drone-b is kept free to receive it; drone-c, close by, is free too.
        let (mut battery, mobile) =
            fleet(&[("drone-a", 500.0), ("drone-b", 3_000.0), ("drone-c", 520.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage
            .decisions()
            .iter()
            .all(|d| d.kind != "team_assignment"));
    }

    /// P1 teams: a drone left free after allocation joins a POI being
    /// inspected when it arrives before the holder would finish alone. Both
    /// credit the dwell, the POI completes sooner than with its holder
    /// alone, and completion releases both. The join is an applied decision.
    #[test]
    fn efficient_v1_teams_a_free_drone_onto_an_active_poi() {
        let (battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 520.0)]);
        // One POI, 40 s of dwell, both drones already on it.
        let one = || {
            vec![InspectionPoi {
                required_dwell_s: 40.0,
                ..poi("poi-00", 500.0, 1)
            }]
        };
        let ticks_to_complete = |mut stage: MissionDriverStage| {
            for tick in 0..100 {
                stage.inspect(tick as f64, 1.0, &mobile, &battery, &empty_map());
                if stage.pois.all_complete() {
                    return (tick, stage);
                }
            }
            panic!("never completed");
        };
        let (reference_ticks, _) = ticks_to_complete(driver(one()));

        let mut stage = efficient_driver(one());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.pois.leads("drone-a"));
        assert_eq!(stage.pois.assignment_for("drone-b"), Some("poi-00"));
        let join = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "team_assignment")
            .expect("the join is recorded");
        assert_eq!(join.status, DecisionStatus::Applied);
        assert_eq!(join.reason, "fleet_team_join");
        assert_eq!(join.subject.as_deref(), Some("drone-b"));
        assert_eq!(stage.take_override_commands().len(), 2, "both fly to it");

        let (efficient_ticks, stage) = ticks_to_complete(efficient_driver(one()));
        assert!(
            efficient_ticks * 2 <= reference_ticks + 2,
            "two drones halve the dwell: {efficient_ticks} vs {reference_ticks} ticks"
        );
        assert_eq!(stage.pois.assignment_for("drone-a"), None, "released");
        assert_eq!(stage.pois.assignment_for("drone-b"), None, "released");
    }

    /// No join that cannot help or cannot be afforded: a drone that would
    /// arrive after the holder finishes stays free, and so does one whose
    /// battery could not cover the trip, its share of the dwell and the trip
    /// home.
    #[test]
    fn efficient_v1_teams_only_when_it_helps_and_is_affordable() {
        // 10 s of dwell left, the second drone 5 km away: it would arrive
        // long after.
        let (battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 5_500.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 10.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.pois.leads("drone-a"));
        assert_eq!(stage.pois.assignment_for("drone-b"), None);

        // Close enough to help, but nearly flat.
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 520.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 40.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        stage.inspect(0.0, 1.0, &mobile[..1], &battery, &empty_map());
        assert!(stage.pois.leads("drone-a"));
        assert!(battery.set_remaining_wh("drone-b", 1.0));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-b"), None);
        assert!(stage
            .decisions()
            .iter()
            .all(|d| d.kind != "team_assignment"));
    }

    /// The reference keeps its dead team branch: no drone ever joins an
    /// active POI there.
    #[test]
    fn the_reference_still_never_teams() {
        let (battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 520.0)]);
        let mut stage = driver(vec![InspectionPoi {
            required_dwell_s: 40.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        for tick in 0..5 {
            stage.inspect(tick as f64, 1.0, &mobile, &battery, &empty_map());
            assert_eq!(stage.pois.assignment_for("drone-b"), None);
        }
    }

    /// A mission that repeats a POI id is refused under efficient_v1 (the
    /// driver resolves POIs by id), and accepted under the reference as
    /// before.
    #[test]
    fn efficient_v1_refuses_repeated_poi_ids() {
        let pois = || vec![poi("poi-00", 100.0, 1), poi("poi-00", 900.0, 1)];
        let error = driver(pois())
            .set_policy(PlanningPolicy::EfficientV1)
            .unwrap_err();
        assert!(error.contains("poi-00 is repeated"), "{error}");
        assert!(driver(pois()).set_policy(PlanningPolicy::Reference).is_ok());
    }

    /// A low drone holding a POI keeps its receiver: the fresh free drone the
    /// handoff will choose is not spent on a pending POI first.
    #[test]
    fn efficient_v1_keeps_the_handoff_receiver_free() {
        let (mut battery, mobile) = fleet(&[("drone-low", 100.0), ("drone-fresh", 900.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 100.0, 1), poi("poi-01", 900.0, 1)]);
        // drone-low takes poi-00 while it is still full.
        let only_low = vec![mobile[0].clone()];
        stage.inspect(0.0, 1.0, &only_low, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-low"), Some("poi-00"));
        // Then it runs low, and drone-fresh appears with poi-01 pending.
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-low", 0.25 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let handoff = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_handoff")
            .expect("the low drone hands off");
        assert_eq!(handoff.status, DecisionStatus::Applied);
        assert_eq!(handoff.selected.as_deref(), Some("drone-fresh"));
    }

    /// Safe handoff: with no drone able to afford the POI, the low drone
    /// gives it back to the pool instead of handing it to a drone that would
    /// run short, stops flying to it, and takes nothing new.
    #[test]
    fn efficient_v1_returns_a_poi_no_one_can_afford_to_the_pool() {
        let (mut battery, mobile) = fleet(&[("drone-low", 100.0), ("drone-flat", 20_000.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 100.0, 1)]);
        stage.inspect(0.0, 1.0, &mobile[..1], &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-low"), Some("poi-00"));
        stage.take_override_commands();
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-low", 0.25 * capacity));
        // Above the handoff threshold, but 20 km away: the trip alone would
        // take it below its reserve.
        assert!(battery.set_remaining_wh("drone-flat", 0.31 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let release = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_release")
            .expect("the POI is released");
        assert_eq!(release.reason, "no_affordable_receiver");
        assert!(stage.decisions().iter().all(|d| d.kind != "poi_handoff"));
        assert_eq!(stage.pois.assignment_for("drone-low"), None);
        assert_eq!(stage.pois.assignment_for("drone-flat"), None);
        assert_eq!(stage.pois.pending_pois().len(), 1, "back in the pool");
        assert!(stage
            .take_override_commands()
            .iter()
            .any(|c| matches!(c, OverrideCommand::ClearPoi(id) if id == "drone-low")));
    }

    /// A teammate already on the POI costs no trip, so it takes the lead
    /// from a low lead before a free drone farther away does. A low teammate
    /// just leaves its team.
    #[test]
    fn efficient_v1_hands_a_teams_poi_to_the_teammate() {
        let (mut battery, mobile) =
            fleet(&[("drone-a", 500.0), ("drone-b", 510.0), ("drone-c", 2_000.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        stage.inspect(0.0, 1.0, &mobile[..2], &battery, &empty_map());
        assert!(stage.pois.leads("drone-a"));
        assert_eq!(stage.pois.assignment_for("drone-b"), Some("poi-00"));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let handoff = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_handoff")
            .expect("handed on");
        assert_eq!(handoff.reason, "safe_handoff");
        assert_eq!(handoff.selected.as_deref(), Some("drone-b"));
        assert!(stage.pois.leads("drone-b"));
        assert_eq!(stage.pois.assignment_for("drone-a"), None);

        // Now drone-c has joined; drone-b's teammate runs low and leaves.
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 510.0)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert!(battery.set_remaining_wh("drone-b", 0.25 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let left = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_release")
            .expect("the teammate leaves");
        assert_eq!(left.reason, "low_battery_teammate");
        assert!(stage.pois.leads("drone-a"), "the lead keeps it");
        assert_eq!(stage.pois.assignment_for("drone-b"), None);
    }

    /// A wall across the x axis at x = 300, 400 m long and 10 m thick, in a
    /// 2 km square: whatever is west of it flies around one end to reach a
    /// POI east of it.
    fn walled(stage: &mut MissionDriverStage) {
        let wall = argusnet_world::obstacles::Obstacle::wall_segment(
            "wall",
            argusnet_world::obstacles::BlockerType::Wall,
            [300.0, -200.0],
            [300.0, 200.0],
            10.0,
            0.0,
            120.0,
        )
        .expect("valid wall");
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let layer = ObstacleLayer::new(bounds, 250.0, vec![wall]).expect("valid layer");
        stage.set_known_obstacles(Some(Arc::new(layer)), bounds);
    }

    fn poi_flight(commands: &[OverrideCommand], drone: &str) -> GotoHover {
        commands
            .iter()
            .find_map(|command| match command {
                OverrideCommand::SetPoi(id, flight) if id == drone => Some(flight.clone()),
                _ => None,
            })
            .expect("a POI redirect")
    }

    /// P3: under efficient_v1 a drone flies around the wall between it and
    /// its POI, not through it, and the journal records the route.
    #[test]
    fn efficient_v1_routes_a_poi_redirect_around_a_wall() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut stage);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");

        assert!(!flight.via_xy().is_empty(), "the route turns");
        assert!(
            flight.path_length_m() > 400.0,
            "longer than the straight line"
        );
        assert_eq!(flight.target(), [500.0, 0.0, 40.0]);
        let steps = 400;
        for step in 0..=steps {
            let t = flight.t_arrive() * step as f64 / steps as f64;
            let ([x, y, _], _) = flight.sample(t);
            assert!(
                !((295.0..=305.0).contains(&x) && (-200.0..=200.0).contains(&y)),
                "inside the wall at t = {t}: ({x}, {y})"
            );
        }
        let (arrived, stopped) = flight.sample(flight.t_arrive());
        assert_eq!(arrived, [500.0, 0.0, 40.0]);
        assert_eq!(stopped, [0.0, 0.0, 0.0]);

        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.reason, "fleet_min_cost", "reason codes unchanged");
        let length = decision.inputs["route_length_m"]
            .as_f64()
            .expect("recorded");
        assert_eq!(length, flight.path_length_m());
        assert_eq!(decision.inputs["route_waypoints"], flight.via_xy().len());
    }

    /// Ground 100 m high beyond |y| = 150, flat inside: a ridge under both
    /// ends of the wall.
    #[derive(Debug)]
    struct Ridges;

    impl crate::observe::TerrainHeight for Ridges {
        fn height_at(&self, _x_m: f64, y_m: f64) -> f64 {
            if y_m.abs() > 150.0 {
                100.0
            } else {
                0.0
            }
        }
    }

    /// P3: a detour that would carry the drone below the ground under it is
    /// not flown; the drone keeps the straight redirect.
    #[test]
    fn efficient_v1_never_routes_a_drone_below_the_terrain() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut stage);
        stage.set_known_terrain(Some(Arc::new(Ridges)));
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(flight.via_xy().is_empty(), "straight, not over the ridge");
    }

    /// A ridge 2 m wide, narrower than the old 10 m check step, still stops
    /// a detour that would cross it too low.
    #[derive(Debug)]
    struct NarrowRidge;

    impl crate::observe::TerrainHeight for NarrowRidge {
        fn height_at(&self, _x_m: f64, y_m: f64) -> f64 {
            if (y_m.abs() - 180.0).abs() < 1.0 {
                100.0
            } else {
                0.0
            }
        }
    }

    #[test]
    fn efficient_v1_sees_a_narrow_ridge_on_a_detour() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut stage);
        stage.set_known_terrain(Some(Arc::new(NarrowRidge)));
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(flight.via_xy().is_empty(), "straight, not over the ridge");
    }

    /// P3: a POI 3 m beyond the wall, inside the planner's clearance, is
    /// still reached round the wall, not straight through it: the route ends
    /// with a short leg from the nearest clear point to the POI itself.
    #[test]
    fn efficient_v1_routes_to_a_poi_beside_a_wall() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 308.0, 1)]);
        walled(&mut stage);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(!flight.via_xy().is_empty(), "round the wall");
        assert_eq!(flight.target(), [308.0, 0.0, 40.0]);
        let steps = 400;
        for step in 0..=steps {
            let t = flight.t_arrive() * step as f64 / steps as f64;
            let ([x, y, _], _) = flight.sample(t);
            assert!(
                !((295.0..=305.0).contains(&x) && (-200.0..=200.0).contains(&y)),
                "inside the wall at t = {t}: ({x}, {y})"
            );
        }
    }

    /// P3: the allocation costs the route, not the straight line. drone-a is
    /// 250 m from the POI in a straight line and drone-b 300 m, but drone-a
    /// has the wall in front of it and must fly some 500 m around it.
    #[test]
    fn efficient_v1_allocates_on_route_length() {
        let (battery, mobile) = fleet(&[("drone-a", 250.0), ("drone-b", 800.0)]);

        let mut straight = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        straight.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(straight.pois.assignment_for("drone-a"), Some("poi-00"));

        let mut routed = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut routed);
        assert!(routed.trip_length_m(mobile[0].1, [500.0, 0.0, 40.0]) > 450.0);
        routed.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(routed.pois.assignment_for("drone-b"), Some("poi-00"));
        assert_eq!(routed.pois.assignment_for("drone-a"), None);
        // drone-b's way is clear, so it flies straight and records no route.
        let flight = poi_flight(&routed.take_override_commands(), "drone-b");
        assert!(flight.via_xy().is_empty());
        let decision = routed
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert!(decision.inputs.get("route_length_m").is_none());
    }

    /// A drone flying over the wall (its top is 120 m; the flight never goes
    /// below 200 m) has nothing to go around and flies straight.
    #[test]
    fn efficient_v1_flies_straight_over_an_obstacle_below_it() {
        let (battery, _) = fleet(&[("drone-a", 100.0)]);
        let mobile = vec![("drone-a".to_string(), [100.0, 0.0, 200.0])];
        let mut stage = efficient_driver(vec![InspectionPoi {
            position: [500.0, 0.0, 200.0],
            ..poi("poi-00", 500.0, 1)
        }]);
        walled(&mut stage);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert_eq!(
            flight,
            GotoHover::redirect([100.0, 0.0, 200.0], [500.0, 0.0], 200.0, 0.0, 28.0)
        );
    }

    /// With no obstacle layer, or under the reference, a redirect is the
    /// straight one exactly, wall or no wall.
    #[test]
    fn without_a_layer_or_under_the_reference_redirects_fly_straight() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let straight = GotoHover::redirect([100.0, 0.0, 40.0], [500.0, 0.0], 40.0, 0.0, 28.0);

        let mut unlayered = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        assert!(!unlayered.knows_obstacles());
        unlayered.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(
            poi_flight(&unlayered.take_override_commands(), "drone-a"),
            straight
        );
        assert!(unlayered.decisions()[1..]
            .iter()
            .all(|d| d.inputs.get("route_length_m").is_none()));

        let mut reference = driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut reference);
        reference.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(
            poi_flight(&reference.take_override_commands(), "drone-a"),
            straight
        );

        // A layer that is withdrawn (a run with no prior map) flies straight.
        let mut withdrawn = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        walled(&mut withdrawn);
        withdrawn.set_known_obstacles(None, Bounds2d::new(0.0, 1.0, 0.0, 1.0));
        withdrawn.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(
            poi_flight(&withdrawn.take_override_commands(), "drone-a"),
            straight
        );
    }

    /// A wall at x = 0 from y = -900 to 900 in a 2 km square: a drone east
    /// of it whose home is west of it flies round one end, much further than
    /// the straight line.
    fn long_walled(stage: &mut MissionDriverStage) {
        let wall = argusnet_world::obstacles::Obstacle::wall_segment(
            "wall",
            argusnet_world::obstacles::BlockerType::Wall,
            [0.0, -900.0],
            [0.0, 900.0],
            10.0,
            0.0,
            120.0,
        )
        .expect("valid wall");
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let layer = ObstacleLayer::new(bounds, 250.0, vec![wall]).expect("valid layer");
        stage.set_known_obstacles(Some(Arc::new(layer)), bounds);
    }

    /// A gap sweep's legs go round the known obstacles in their way, a
    /// sweep cell inside one is left out, and a flight above them all is
    /// flown straight.
    #[test]
    fn a_gap_sweep_routes_every_leg_around_known_obstacles() {
        let mut stage = efficient_driver(Vec::new());
        long_walled(&mut stage);
        let obstacles = stage.obstacles.as_ref().expect("known");
        // The leg from (-100, 0) to (100, 0) crosses the wall: it turns
        // round an end of it, beyond y = +-900.
        let swept = stage
            .routed_sweep(&[[-100.0, 0.0], [100.0, 0.0]], 40.0)
            .expect("routed");
        assert_eq!(swept.first(), Some(&[-100.0, 0.0]));
        assert_eq!(swept.last(), Some(&[100.0, 0.0]));
        assert!(swept.len() > 2, "{swept:?}");
        assert!(swept.iter().any(|p| p[1].abs() > 900.0), "{swept:?}");
        // A leg on one side stays straight.
        assert_eq!(
            stage.routed_sweep(&[[100.0, 0.0], [100.0, 50.0]], 40.0),
            Some(vec![[100.0, 0.0], [100.0, 50.0]])
        );
        // The wall's footprint is no place to fly to below its top, and
        // nothing blocks a flight above it.
        assert!(obstacles.stands_in([0.0, 0.0], 40.0));
        assert!(!obstacles.stands_in([100.0, 0.0], 40.0));
        assert!(!obstacles.blocks_at(400.0));
        assert_eq!(
            stage.routed_sweep(&[[-100.0, 0.0], [100.0, 0.0]], 400.0),
            Some(vec![[-100.0, 0.0], [100.0, 0.0]])
        );
    }

    /// P3: the reserve home is priced along the route home when it turns.
    /// A drone with more than the straight-line reserve but less than the
    /// routed one is sent home when the wall is known, and not otherwise.
    #[test]
    fn efficient_v1_prices_the_reserve_along_the_route_home() {
        let home = [-900.0, 0.0, 40.0];
        let homes: BTreeMap<String, [f64; 3]> =
            [("drone-a".to_string(), home)].into_iter().collect();
        let (mut battery, mobile) = fleet(&[("drone-a", 900.0)]);
        let model = battery.model();
        let wind = battery.wind_mps();
        let at = mobile[0].1;

        let mut open =
            MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), homes.clone());
        open.set_policy(PlanningPolicy::EfficientV1).unwrap();
        let straight = model.dynamic_reserve_wh(
            at,
            home,
            wind,
            argusnet_planning::BatteryModel::DEFAULT_SAFETY_MARGIN,
        );
        assert_eq!(
            open.reserve_home_wh(&model, at, home, wind, false),
            straight
        );
        assert!(
            straight > model.capacity_wh * model.reserve_fraction,
            "above the floor, so the distance shows"
        );

        let mut walled_off =
            MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), homes);
        walled_off.set_policy(PlanningPolicy::EfficientV1).unwrap();
        long_walled(&mut walled_off);
        let routed = walled_off.reserve_home_wh(&model, at, home, wind, false);
        assert!(routed > 1.2 * straight, "{routed} vs {straight}");
        assert!(
            routed <= detour_bound_wh(&model, at, home, wind),
            "the skip bound covers a detour this long"
        );
        // The same reserve from fixed ends, kept for later runs.
        assert_eq!(
            walled_off.reserve_home_wh(&model, at, home, wind, true),
            routed
        );
        let kept = walled_off
            .obstacles
            .as_ref()
            .expect("walled")
            .kept
            .borrow()
            .len();
        assert_eq!(kept, 1);

        let between = 0.5 * (straight + routed);
        assert!(battery.set_remaining_wh("drone-a", between));
        open.send_low_drones_home(&mobile, &battery, 1.0);
        assert!(open.decisions().iter().all(|d| d.kind != "return_home"));
        walled_off.send_low_drones_home(&mobile, &battery, 1.0);
        let sent = walled_off
            .decisions()
            .iter()
            .find(|d| d.kind == "return_home")
            .expect("sent home");
        assert_eq!(sent.inputs["reserve_wh"].as_f64(), Some(routed));
        assert!(sent.inputs["route_length_m"].as_f64().expect("routed") > 2000.0);
    }

    /// The reference never reads the layer: its reserve is the battery's.
    #[test]
    fn the_reference_reserve_ignores_known_obstacles() {
        let home = [-900.0, 0.0, 40.0];
        let (battery, mobile) = fleet(&[("drone-a", 900.0)]);
        let model = battery.model();
        let wind = battery.wind_mps();
        let mut reference = driver(Vec::new());
        long_walled(&mut reference);
        assert_eq!(
            reference.reserve_home_wh(&model, mobile[0].1, home, wind, false),
            model.dynamic_reserve_wh(
                mobile[0].1,
                home,
                wind,
                argusnet_planning::BatteryModel::DEFAULT_SAFETY_MARGIN,
            )
        );
    }

    /// P2: a drone at its reserve home is routed home on its own, gives its
    /// POI to a drone that can afford it, and takes nothing again. The
    /// reference never does.
    #[test]
    fn efficient_v1_sends_a_drone_at_its_reserve_home() {
        let homes: BTreeMap<String, [f64; 3]> = [
            ("drone-a".to_string(), [0.0, 0.0, 40.0]),
            ("drone-b".to_string(), [0.0, 0.0, 40.0]),
        ]
        .into_iter()
        .collect();
        let pois = || {
            vec![InspectionPoi {
                required_dwell_s: 60.0,
                ..poi("poi-00", 900.0, 1)
            }]
        };
        let (mut battery, mobile) = fleet(&[("drone-a", 900.0), ("drone-b", 300.0)]);
        let mut stage =
            MissionDriverStage::new(MissionDriverConfig::default(), pois(), homes.clone());
        stage.set_policy(PlanningPolicy::EfficientV1).unwrap();
        stage.inspect(0.0, 1.0, &mobile[..1], &battery, &empty_map());
        assert!(stage.pois.leads("drone-a"));
        stage.take_override_commands();

        // drone-a down to its reserve: above the handoff threshold, but no
        // more than it needs to fly the 900 m home.
        let model = battery.model();
        let reserve = model.dynamic_reserve_wh(
            mobile[0].1,
            homes["drone-a"],
            battery.wind_mps(),
            argusnet_planning::BatteryModel::DEFAULT_SAFETY_MARGIN,
        );
        assert!(battery.set_remaining_wh("drone-a", reserve));
        stage.send_low_drones_home(&mobile, &battery, 1.0);
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let home = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "return_home")
            .expect("sent home");
        assert_eq!(home.reason, "reserve_reached");
        assert_eq!(home.subject.as_deref(), Some("drone-a"));
        let commands = stage.take_override_commands();
        assert!(commands
            .iter()
            .any(|c| matches!(c, OverrideCommand::SetEgress(id, _) if id == "drone-a")));
        assert_eq!(stage.pois.assignment_for("drone-a"), None);
        assert!(stage.pois.leads("drone-b"), "the POI went to drone-b");
        let handoff = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_handoff")
            .expect("handed on");
        assert_eq!(handoff.reason, "reserve_handoff", "not a low battery");
        assert!(
            handoff.detail.contains("at its reserve"),
            "{}",
            handoff.detail
        );
        assert_eq!(handoff.inputs["at_reserve"], true);

        // Never given anything again, and not sent home twice. (`run` clears
        // a run's decisions; the steps are called directly here.)
        stage.decisions.clear();
        stage.send_low_drones_home(&mobile, &battery, 2.0);
        stage.inspect(2.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.decisions().iter().all(|d| d.kind != "return_home"));
        assert_eq!(stage.pois.assignment_for("drone-a"), None);

        // Survives a checkpoint.
        let state = stage.checkpoint_state().unwrap();
        assert_eq!(state["homeward"], serde_json::json!(["drone-a"]));
        let mut restored = MissionDriverStage::new(MissionDriverConfig::default(), pois(), homes);
        restored.set_policy(PlanningPolicy::EfficientV1).unwrap();
        restored.restore_state(&state).unwrap();
        assert!(restored.homeward.contains("drone-a"));
        let empty = efficient_driver(pois()).checkpoint_state().unwrap();
        assert!(
            empty.get("homeward").is_none(),
            "absent when no drone is homeward"
        );
    }

    /// A low lead whose teammate cannot afford the dwell left either: the
    /// whole team lets go and the POI goes back to the pool, rather than to
    /// the teammate the solve just ruled out.
    #[test]
    fn efficient_v1_returns_a_team_no_one_can_afford_to_the_pool() {
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 510.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 36_000.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        assert!(stage.pois.team_assign_at("drone-b", 0));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let released = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_release")
            .expect("given up");
        assert_eq!(released.reason, "no_affordable_receiver");
        assert_eq!(
            released.inputs["released"],
            serde_json::json!(["drone-a", "drone-b"])
        );
        assert_eq!(stage.pois.assignment_for("drone-a"), None);
        assert_eq!(stage.pois.assignment_for("drone-b"), None);
        assert_eq!(stage.pois.pending_pois().len(), 1);
    }

    /// A teammate still en route is charged the rest of its trip: one that
    /// could afford the dwell from the POI but not the flight there first is
    /// no receiver, and the POI goes back to the pool.
    #[test]
    fn efficient_v1_charges_an_en_route_teammate_its_trip() {
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 20_500.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        assert!(stage.pois.team_assign_at("drone-b", 0));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        assert!(battery.set_remaining_wh("drone-b", 0.45 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let released = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_release")
            .expect("given up, not handed to drone-b");
        assert_eq!(released.reason, "no_affordable_receiver");
        assert_eq!(stage.pois.assignment_for("drone-b"), None);

        // The same teammate at the POI takes it.
        let (mut battery, mut mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 500.0)]);
        mobile[1].1[1] = 10.0;
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        assert!(stage.pois.team_assign_at("drone-b", 0));
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        assert!(battery.set_remaining_wh("drone-b", 0.45 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.pois.leads("drone-b"));
    }

    /// A teammate still en route shares nothing yet: a receiver already at
    /// the POI is charged the dwell alone, and one that can afford only half
    /// of it is passed over for one that can afford its share.
    #[test]
    fn efficient_v1_shares_a_handoff_only_with_teammates_in_range() {
        let (mut battery, mobile) =
            fleet(&[("drone-a", 500.0), ("drone-b", 510.0), ("drone-c", 2_000.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 360.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        assert!(stage.pois.team_assign_at("drone-b", 0));
        assert!(stage.pois.team_assign_at("drone-c", 0));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        // 40 Wh: 360 s of hover alone (30 Wh) would leave it under its
        // 20 Wh reserve; half of it (15 Wh) would not.
        assert!(battery.set_remaining_wh("drone-b", 0.40 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let handoff = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_handoff")
            .expect("handed on");
        assert_eq!(handoff.selected.as_deref(), Some("drone-c"));
    }

    /// A low teammate leaves before the lead's handoff is costed, so the
    /// receiver is charged for the dwell shared among the drones that stay.
    #[test]
    fn efficient_v1_costs_a_handoff_without_departing_teammates() {
        let (mut battery, mobile) =
            fleet(&[("drone-a", 500.0), ("drone-b", 510.0), ("drone-c", 505.0)]);
        let mut stage = efficient_driver(vec![InspectionPoi {
            required_dwell_s: 60.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        assert!(stage.pois.assign_at("drone-a", 0));
        assert!(stage.pois.team_assign_at("drone-b", 0));
        assert!(stage.pois.team_assign_at("drone-c", 0));
        let capacity = battery.model().capacity_wh;
        assert!(battery.set_remaining_wh("drone-a", 0.25 * capacity));
        assert!(battery.set_remaining_wh("drone-c", 0.25 * capacity));
        stage.inspect(1.0, 1.0, &mobile, &battery, &empty_map());
        let kinds: Vec<(&str, &str)> = stage
            .decisions()
            .iter()
            .filter(|d| d.kind != "poi_rescore")
            .map(|d| (d.kind, d.reason))
            .collect();
        assert_eq!(
            &kinds[..2],
            &[
                ("poi_release", "low_battery_teammate"),
                ("poi_handoff", "safe_handoff")
            ]
        );
        assert!(stage.pois.leads("drone-b"));
        assert_eq!(stage.pois.active_pois()[0].members, 1);
    }

    /// A POI the drone could not reach, dwell at and keep its reserve after
    /// is never assigned to it, however good the POI.
    #[test]
    fn efficient_v1_never_assigns_past_the_reserve() {
        let (mut battery, mobile) = fleet(&[("drone-a", 100.0), ("drone-b", 900.0)]);
        let model = battery.model();
        // Exactly the static reserve: any trip at all would dip into it.
        assert!(battery.set_remaining_wh("drone-a", model.capacity_wh * model.reserve_fraction));
        let mut stage = efficient_driver(vec![poi("poi-00", 100.0, 3)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), None);
        assert_eq!(stage.pois.assignment_for("drone-b"), Some("poi-00"));
    }

    /// No preemption in this slice: a drone keeps the POI it holds when a
    /// later solve would rather it went elsewhere.
    #[test]
    fn efficient_v1_never_takes_a_held_poi_away() {
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 100.0, 1), poi("poi-01", 900.0, 1)]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        let moved = vec![("drone-a".to_string(), [900.0, 0.0, 40.0])];
        stage.inspect(1.0, 1.0, &moved, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        // `inspect` appends; `run` clears. One assignment, one decision.
        let assignments = stage
            .decisions()
            .iter()
            .filter(|d| d.kind == "poi_assignment")
            .count();
        assert_eq!(assignments, 1, "no new assignment, so no new decision");
    }

    /// The checkpoint records the policy (absent for the reference, so a
    /// reference checkpoint is unchanged), and a resume under the other
    /// policy is refused.
    #[test]
    fn a_checkpoint_resumes_only_under_its_own_policy() {
        let pois = || vec![poi("poi-00", 100.0, 1)];
        let reference = driver(pois());
        let efficient = efficient_driver(pois());
        let reference_state = reference.checkpoint_state().expect("state");
        let efficient_state = efficient.checkpoint_state().expect("state");
        assert!(reference_state.get("policy").is_none());
        assert_eq!(efficient_state["policy"], "efficient_v1");
        assert!(reference.check_state(&reference_state).is_ok());
        assert!(efficient.check_state(&efficient_state).is_ok());
        let refused = efficient.check_state(&reference_state).unwrap_err();
        assert!(
            format!("{refused:?}").contains("under reference, this run under efficient_v1"),
            "{refused:?}"
        );
        assert!(reference.check_state(&efficient_state).is_err());
    }
}
