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
use argusnet_world::obstacles::{BlockerType, Obstacle};
use argusnet_world::powerline::PowerLine;
use argusnet_world::trees::TreeLayer;

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
    /// How many power-line spans the self-built map had at the last run
    /// (R3). Absent when none.
    #[serde(default, skip_serializing_if = "is_zero")]
    mapped_wires: usize,
    /// Pending POIs no drone can afford under contingency_v1 (F5). Absent
    /// when none.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    unassigned: BTreeSet<String>,
}

fn is_zero(count: &usize) -> bool {
    *count == 0
}

/// Why a drone gives up the POI it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GiveUp {
    /// At its reserve home, and heading there.
    Reserve,
    /// Below the fixed handoff fraction (efficient_v1).
    Low,
    /// It can no longer afford the dwell left and the trip home (F5).
    CannotFinish,
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
    /// Pending POIs no drone can afford to finish and fly home from, under
    /// contingency_v1 (F5): reported unassigned, not merely pending.
    pub unassigned: BTreeSet<String>,
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
    /// The run's battery model, read off stage 1 at the start of every run:
    /// whether a flight climbs over a power line or goes round its ends is
    /// priced on it (R3).
    route_model: BatteryModel,
    /// In a run without a prior map under efficient_v1, the self-built map,
    /// whose mapped wires flights keep their clearance from (R3). Scenario.
    mapped: Option<crate::belief_world::BeliefHandle>,
    /// How many mapped spans the routes were last given. Derived.
    mapped_installed: usize,
    /// Whether [`Self::obstacles`] was installed, empty, only to carry the
    /// mapped wires. Derived.
    mapped_layer: bool,
    /// How many spans were mapped at the last run, so a run that maps more
    /// replans the POI flights a new wire lies across. Checkpointed.
    mapped_wires: usize,
    /// The map's bounds, which an empty layer carrying mapped wires covers.
    mapped_bounds: Option<Bounds2d>,
    /// Whether this run mapped more wires than the last. Transient.
    replan_for_wires: bool,
    /// Pending POIs no drone can afford under contingency_v1 (F5).
    /// Checkpointed.
    unassigned: BTreeSet<String>,
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
    /// The power lines a flight keeps its clearance from (R3), and their
    /// spans' corridors. Empty when the world has none.
    lines: Vec<PowerLine>,
    corridors: Vec<Corridor>,
    /// [`Self::reaching`] with the corridors as obstacles too, by band, for
    /// routes round a line's ends.
    reaching_lines: RefCell<BTreeMap<i64, Option<Arc<ObstacleLayer>>>>,
    /// [`Self::this_run`] for routes round a line's ends.
    around_this_run: RefCell<BTreeMap<[u64; 6], Option<PlannerRoute>>>,
    /// The trees whose crowns a flight keeps above (R2). `None` when the
    /// world has none.
    trees: Option<Arc<TreeLayer>>,
}

/// How far past a crown's radius its keep-out reaches, metres (R2).
const CROWN_MARGIN_M: f64 = 1.0;
/// How far over a crown's top a flight across its keep-out keeps, metres
/// (R2): leaves are soft, so a small margin, not a wire's clearance.
const CROWN_CLEARANCE_M: f64 = 3.0;
/// How far under a keep-out's height a flight may pass and still count as
/// over it, metres: a route raised to the height exactly at the keep-out's
/// edge is split there, and the edge found again on the new leg can fall a
/// rounding error inside.
const CROWN_TOLERANCE_M: f64 = 1e-6;

/// How far past a power line's own clearance a planned crossing keeps,
/// metres (R3): room for the straight pieces' sag error and for rounding,
/// so a route laid out on the corridors passes the swept test.
const LINE_MARGIN_M: f64 = 1.0;

/// The steepest a climb over a power line, or the descent after it, is
/// made when there is room (R3): 1 in 8, 3.5 m/s at the 28 m/s redirect
/// speed, inside every role's climb and descent rates. An end nearer the
/// line than that allows climbs as steeply as it must.
const LINE_RAMP_GRADIENT: f64 = 0.125;

/// One span of a power line from above, with its clearance (R3): a
/// rectangle along the span holding every conductor of it and the clearance
/// round them (plus [`LINE_MARGIN_M`]), and the height a flight over it
/// keeps. Off the rectangle a flight is farther from the span's wires than
/// their clearance at any height. Over it, a flight at `over_z_m` or higher
/// is too: no wire hangs above the higher of its supports' attachments,
/// and `over_z_m` is the clearance above that.
#[derive(Debug, Clone)]
struct Corridor {
    /// The rectangle's centre line, carried past each support by the
    /// half-width: a conductor's attachment on a turned cross-arm sits up
    /// to its offset along the span.
    from: [f64; 2],
    to: [f64; 2],
    half_width_m: f64,
    over_z_m: f64,
    /// The lower support's ground, the base of the corridor as an obstacle.
    ground_z_m: f64,
}

impl Corridor {
    /// Each span's corridor.
    fn of_line(line: &PowerLine) -> Vec<Corridor> {
        let reach = line
            .conductor_offsets_m
            .iter()
            .fold(0.0_f64, |widest, offset| widest.max(offset.abs()))
            + line.conductor_radius_m
            + line.clearance_m
            + LINE_MARGIN_M;
        line.supports
            .windows(2)
            .map(|pair| {
                let (a, b) = (pair[0].xy_m, pair[1].xy_m);
                let length = (b[0] - a[0]).hypot(b[1] - a[1]);
                let along = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
                Corridor {
                    from: [a[0] - along[0] * reach, a[1] - along[1] * reach],
                    to: [b[0] + along[0] * reach, b[1] + along[1] * reach],
                    half_width_m: reach,
                    over_z_m: pair[0].top_z_m().max(pair[1].top_z_m())
                        + line.conductor_radius_m
                        + line.clearance_m
                        + LINE_MARGIN_M,
                    ground_z_m: pair[0].base_z_m.min(pair[1].base_z_m),
                }
            })
            .collect()
    }

    /// The part of the leg from `p` to `q` over the rectangle, as fractions
    /// of the leg, or `None` when the leg misses it (Liang–Barsky, in the
    /// span's own frame).
    fn clip(&self, p: [f64; 2], q: [f64; 2]) -> Option<(f64, f64)> {
        let length = (self.to[0] - self.from[0]).hypot(self.to[1] - self.from[1]);
        let along = [
            (self.to[0] - self.from[0]) / length,
            (self.to[1] - self.from[1]) / length,
        ];
        let local = |x: [f64; 2]| {
            let d = [x[0] - self.from[0], x[1] - self.from[1]];
            [
                d[0] * along[0] + d[1] * along[1],
                d[1] * along[0] - d[0] * along[1],
            ]
        };
        let (a, b) = (local(p), local(q));
        let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
        for (start, delta, low, high) in [
            (a[0], b[0] - a[0], 0.0, length),
            (a[1], b[1] - a[1], -self.half_width_m, self.half_width_m),
        ] {
            if delta.abs() < 1e-12 {
                if start < low || start > high {
                    return None;
                }
                continue;
            }
            let (e0, e1) = ((low - start) / delta, (high - start) / delta);
            t0 = t0.max(e0.min(e1));
            t1 = t1.min(e0.max(e1));
            if t0 > t1 {
                return None;
            }
        }
        Some((t0, t1))
    }

    /// The corridor as a hard obstacle up to its crossing height, for the
    /// planner to route round.
    fn obstacle(&self, id: String) -> Option<Obstacle> {
        Obstacle::wall_segment(
            id,
            BlockerType::Wall,
            self.from,
            self.to,
            2.0 * self.half_width_m,
            self.ground_z_m,
            self.over_z_m,
        )
        .ok()
    }
}

/// The legs flown from `start` through `path` (the corners, then the end),
/// each with its ends' heights: `heights` at the corners when given, else
/// changing linearly with the distance flown, from `start`'s height to
/// `end_z`, as [`GotoHover::routed`] flies it.
fn flown_legs(
    start: [f64; 3],
    path: &[[f64; 2]],
    heights: &[f64],
    end_z: f64,
) -> Vec<([f64; 3], [f64; 3])> {
    let mut points = vec![[start[0], start[1]]];
    points.extend_from_slice(path);
    let leg_m = |pair: &[[f64; 2]]| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
    let length_m: f64 = points.windows(2).map(leg_m).sum();
    let mut flown_m = 0.0;
    let mut z_at = |index: usize, leg: f64| {
        if !heights.is_empty() {
            return match index {
                0 => start[2],
                i if i > heights.len() => end_z,
                i => heights[i - 1],
            };
        }
        flown_m += leg;
        if length_m > 0.0 {
            start[2] + (end_z - start[2]) * (flown_m / length_m).clamp(0.0, 1.0)
        } else {
            start[2]
        }
    };
    let mut from_z = z_at(0, 0.0);
    points
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            let to_z = z_at(index + 1, leg_m(pair));
            let leg = (
                [pair[0][0], pair[0][1], from_z],
                [pair[1][0], pair[1][1], to_z],
            );
            from_z = to_z;
            leg
        })
        .collect()
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

    /// Whether the segment from `from` to `to` enters a building or wall.
    fn crosses(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        let around = Bounds2d::new(
            from[0].min(to[0]),
            from[0].max(to[0]),
            from[1].min(to[1]),
            from[1].max(to[1]),
        )
        .expanded(1e-6);
        self.layer
            .query(around)
            .into_iter()
            .any(|o| o.blocker_type.is_hard() && o.surface_hit_t(from, to).is_some())
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

    /// Whether any of `legs` comes closer to a power line than the line's
    /// clearance: the swept test on each leg ([`PowerLine::violated_by`]).
    fn violates_lines(&self, legs: &[([f64; 3], [f64; 3])]) -> bool {
        legs.iter().any(|(from, to)| {
            self.lines.iter().any(|line| {
                let reach = line.bounds_xy_m();
                let misses = from[0].max(to[0]) < reach.x_min_m
                    || from[0].min(to[0]) > reach.x_max_m
                    || from[1].max(to[1]) < reach.y_min_m
                    || from[1].min(to[1]) > reach.y_max_m;
                !misses && line.violated_by(*from, *to)
            })
        })
    }

    /// Whether a flight has anything beyond the structures to keep clear
    /// of: a power line or a crown.
    fn guards_anything(&self) -> bool {
        !self.lines.is_empty() || self.trees.is_some()
    }

    /// The crowns' keep-outs the leg from `p` to `q` crosses from above (R2),
    /// each as the fractions of the leg over it and the height a flight
    /// keeps there, in id order. A keep-out holding `start` or `end` (the
    /// whole flight's ends) is left out: an end beside a crown cannot keep
    /// above it, and touching leaves there is a contact, not a failure.
    fn crown_stretches(
        &self,
        p: [f64; 2],
        q: [f64; 2],
        start: [f64; 2],
        end: [f64; 2],
    ) -> Vec<(f64, f64, f64)> {
        let Some(trees) = &self.trees else {
            return Vec::new();
        };
        let around = Bounds2d::new(
            p[0].min(q[0]),
            p[0].max(q[0]),
            p[1].min(q[1]),
            p[1].max(q[1]),
        )
        .expanded(CROWN_MARGIN_M);
        let d = [q[0] - p[0], q[1] - p[1]];
        let qa = d[0] * d[0] + d[1] * d[1];
        trees
            .near(around)
            .into_iter()
            .filter_map(|tree| {
                let radius = tree.crown_radius_m + CROWN_MARGIN_M;
                let holds =
                    |x: [f64; 2]| (x[0] - tree.xy_m[0]).hypot(x[1] - tree.xy_m[1]) <= radius;
                if holds(start) || holds(end) || qa <= 1e-18 {
                    return None;
                }
                let r = [p[0] - tree.xy_m[0], p[1] - tree.xy_m[1]];
                let qb = 2.0 * (r[0] * d[0] + r[1] * d[1]);
                let qc = r[0] * r[0] + r[1] * r[1] - radius * radius;
                let disc = qb * qb - 4.0 * qa * qc;
                if disc <= 0.0 {
                    return None;
                }
                let root = disc.sqrt();
                let t0 = ((-qb - root) / (2.0 * qa)).max(0.0);
                let t1 = ((-qb + root) / (2.0 * qa)).min(1.0);
                (t0 < t1).then_some((t0, t1, tree.top_z_m() + CROWN_CLEARANCE_M))
            })
            .collect()
    }

    /// Whether any of `legs` (a flight from `start` to `end`) passes over a
    /// crown's keep-out lower than its height (R2), by more than
    /// [`CROWN_TOLERANCE_M`]. The height changes linearly along a leg, so
    /// its lowest over a stretch is at an end of the stretch.
    fn violates_crowns(
        &self,
        legs: &[([f64; 3], [f64; 3])],
        start: [f64; 2],
        end: [f64; 2],
    ) -> bool {
        legs.iter().any(|(from, to)| {
            self.crown_stretches([from[0], from[1]], [to[0], to[1]], start, end)
                .into_iter()
                .any(|(t0, t1, z)| {
                    let at = |t: f64| from[2] + (to[2] - from[2]) * t;
                    at(t0).min(at(t1)) < z - CROWN_TOLERANCE_M
                })
        })
    }

    /// [`Self::reaching`], with the corridors of the power lines too.
    fn reaching_lines(&self, floor_m: f64, clearance_m: f64) -> Option<Arc<ObstacleLayer>> {
        let band = (floor_m / ROUTE_HEIGHT_BAND_M).floor() as i64;
        if let Some(known) = self.reaching_lines.borrow().get(&band) {
            return known.clone();
        }
        let band_floor_m = band as f64 * ROUTE_HEIGHT_BAND_M;
        let primitives: Vec<_> = self
            .layer
            .primitives
            .iter()
            .cloned()
            .chain(
                self.corridors
                    .iter()
                    .enumerate()
                    .filter_map(|(k, corridor)| corridor.obstacle(format!("power-line-span-{k}"))),
            )
            .filter(|obstacle| obstacle.top_z_m + clearance_m > band_floor_m)
            .collect();
        let layer = if primitives.is_empty() {
            None
        } else {
            ObstacleLayer::new(self.layer.bounds, self.layer.tile_size_m, primitives)
                .ok()
                .map(Arc::new)
        };
        self.reaching_lines.borrow_mut().insert(band, layer.clone());
        layer
    }

    /// [`Self::route`] round the power lines' corridors as well as the
    /// structures: past a line's ends rather than over it.
    fn route_around_lines(&self, from: [f64; 3], to: [f64; 3]) -> Option<PlannerRoute> {
        let key = route_key(from, to);
        if let Some(known) = self.around_this_run.borrow().get(&key) {
            return known.clone();
        }
        let mut planner = self.planner.borrow_mut();
        let clearance_m = planner.config().drone_clearance_m;
        let floor_m = from[2].min(to[2]);
        let route = self.reaching_lines(floor_m, clearance_m).and_then(|layer| {
            // Its own revisions, apart from the structures-only layers'.
            let band = (floor_m / ROUTE_HEIGHT_BAND_M).floor() as i64;
            let scene = Scene {
                layer: &layer,
                revision: (band as u64) ^ (1 << 63),
            };
            planner.plan_route_between(scene, [from[0], from[1]], [to[0], to[1]], clearance_m)
        });
        self.around_this_run.borrow_mut().insert(key, route.clone());
        route
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
    /// The height at each of `via` when the flight climbs over a power line
    /// (R3); empty when it is flown at one height.
    heights: Vec<f64>,
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
#[derive(Debug, Clone)]
struct Corners {
    via: Vec<[f64; 2]>,
    length_m: f64,
    /// The height at each corner when the route climbs over a power line
    /// (R3); empty when its height changes linearly from end to end.
    heights: Vec<f64>,
    /// How the route keeps a power line's clearance that the route round
    /// the structures alone would not: `"over"` or `"around"`. `None` when
    /// that route keeps it.
    lines: Option<&'static str>,
    /// Whether the route climbs over crowns (R2) the route round the
    /// structures would cross too low.
    crowns: bool,
}

impl Corners {
    /// A route round the structures, its height changing linearly.
    fn turning(via: Vec<[f64; 2]>, length_m: f64) -> Self {
        Self {
            via,
            length_m,
            heights: Vec::new(),
            lines: None,
            crowns: false,
        }
    }

    /// The energy the route's climbs and descents cost from `from_z` to
    /// `to_z`: leg by leg over its corners' heights, or the one change of
    /// height the allocator has always priced when it has none.
    fn climb_cost_wh(&self, model: &BatteryModel, from_z: f64, to_z: f64) -> f64 {
        climb_along_wh(model, from_z, &self.heights, to_z)
    }
}

/// The energy of the climbs and descents from `from_z` through `heights` to
/// `to_z`, leg by leg; exactly `climb_cost_wh(to_z - from_z)` with no
/// heights.
fn climb_along_wh(model: &BatteryModel, from_z: f64, heights: &[f64], to_z: f64) -> f64 {
    if heights.is_empty() {
        return model.climb_cost_wh(to_z - from_z);
    }
    let mut cost_wh = 0.0;
    let mut z = from_z;
    for &next in heights.iter().chain(std::iter::once(&to_z)) {
        cost_wh += model.climb_cost_wh(next - z);
        z = next;
    }
    cost_wh
}

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
    /// How the route keeps a power line's clearance, when it had to change
    /// to (R3).
    lines: Option<&'static str>,
    /// Whether the route climbs over crowns (R2).
    crowns: bool,
}

impl Flight {
    /// Add the route to a decision's inputs, when there is one. Additive: a
    /// straight flight's inputs are unchanged.
    fn record(&self, inputs: &mut serde_json::Value) {
        if let (Some((length_m, corners)), Some(fields)) = (self.route, inputs.as_object_mut()) {
            fields.insert("route_length_m".into(), serde_json::json!(length_m));
            fields.insert("route_waypoints".into(), serde_json::json!(corners));
            if let Some(lines) = self.lines {
                fields.insert("route_power_lines".into(), serde_json::json!(lines));
            }
            if self.crowns {
                fields.insert("route_trees".into(), serde_json::json!("over"));
            }
        }
    }
}

/// Under [`PlanningPolicy::EfficientV1`], what one step of rescored POI
/// priority is worth against energy. At the reference battery's 350 W and
/// 28 m/s, 5 Wh is about 1.4 km of flight: a higher-priority POI is preferred
/// unless it is that much further. Tuned on the qualification seeds only.
pub const PRIORITY_WORTH_WH: f64 = 5.0;

/// Under [`PlanningPolicy::ContingencyV1`] (F5), the margin above its reserve
/// home, as a fraction of capacity, that a drone must keep after new work to
/// take it. It gives work up only when finishing would leave less than the
/// reserve itself, so the band between is hysteresis: a drone that spends
/// energy as its plan said never drops a POI it has just taken; only a
/// change the plan did not price (a headwind, a longer way home) moves it.
pub const CONTINGENCY_TAKE_MARGIN_FRACTION: f64 = 0.05;

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
            route_model: BatteryModel::default(),
            mapped: None,
            mapped_installed: 0,
            mapped_layer: false,
            mapped_wires: 0,
            mapped_bounds: None,
            replan_for_wires: false,
            unassigned: BTreeSet::new(),
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
            lines: Vec::new(),
            corridors: Vec::new(),
            reaching_lines: RefCell::new(BTreeMap::new()),
            around_this_run: RefCell::new(BTreeMap::new()),
            trees: None,
        });
    }

    /// The trees whose crowns POI, egress and gap-fill flights keep above
    /// under efficient_v1 (R2): a crown is a soft keep-out, its radius plus
    /// [`CROWN_MARGIN_M`] from above, that a flight crosses at least
    /// [`CROWN_CLEARANCE_M`] over its top. Part of the prior map, like the
    /// power lines: set after [`Self::set_known_obstacles`], which forgets
    /// them, and ignored without it. The reference never reads them.
    pub fn set_known_trees(&mut self, trees: Arc<TreeLayer>) {
        if let Some(obstacles) = &mut self.obstacles {
            obstacles.trees = (!trees.is_empty()).then_some(trees);
            obstacles.kept.borrow_mut().clear();
        }
    }

    /// The power lines POI, egress and gap-fill flights keep their clearance
    /// from under efficient_v1 (R3), passing over the wires or round a
    /// line's ends. Part of the prior map: set after
    /// [`Self::set_known_obstacles`], which forgets them, and ignored
    /// without it. The reference never reads them.
    pub fn set_known_power_lines(&mut self, lines: Vec<PowerLine>) {
        if let Some(obstacles) = &mut self.obstacles {
            obstacles.corridors = lines.iter().flat_map(Corridor::of_line).collect();
            obstacles.lines = lines;
            obstacles.reaching_lines.borrow_mut().clear();
            obstacles.around_this_run.borrow_mut().clear();
            obstacles.kept.borrow_mut().clear();
        }
    }

    /// Plan round the power lines `map` has mapped (R3), in a run without a
    /// prior map: from each run on, POI, egress and gap-fill flights keep
    /// the clearance of every span mapped so far, as a prior run's keep the
    /// whole line's, and a POI flight a newly mapped wire lies across is
    /// replanned from where the drone is (`poi_reroute`). Under efficient_v1
    /// only; `None` (the default) maps nothing.
    pub fn set_mapped_power_lines(
        &mut self,
        map: Option<crate::belief_world::BeliefHandle>,
        bounds: Bounds2d,
    ) {
        self.mapped = map;
        self.mapped_bounds = Some(bounds);
    }

    /// Whether flights keep clear of the wires a self-built map holds.
    pub fn follows_mapped_power_lines(&self) -> bool {
        self.mapped.is_some()
    }

    /// The spans the self-built map holds now, given to the routes when
    /// they changed; whether more are mapped than at the last run.
    fn follow_mapped_wires(&mut self) -> bool {
        let (Some(map), Some(bounds)) = (&self.mapped, self.mapped_bounds) else {
            return false;
        };
        let wires = Arc::clone(map.current().wires());
        if wires.len() != self.mapped_installed {
            if self.obstacles.is_none() {
                // The routes plan round nothing else: no structure is known.
                let empty = ObstacleLayer::new(bounds, 1000.0, Vec::new())
                    .expect("an empty layer over the map's bounds");
                self.set_known_obstacles(Some(Arc::new(empty)), bounds);
                self.mapped_layer = true;
            }
            self.set_known_power_lines(wires.to_vec());
            self.mapped_installed = wires.len();
        }
        let more = wires.len() > self.mapped_wires;
        self.mapped_wires = wires.len();
        more
    }

    /// The terrain a routed flight must stay above (P3). Without it a route
    /// is flown as planned.
    pub fn set_known_terrain(&mut self, terrain: Option<Arc<dyn crate::observe::TerrainHeight>>) {
        self.terrain = terrain;
        if let Some(obstacles) = &self.obstacles {
            obstacles.kept.borrow_mut().clear();
        }
    }

    /// Price routes on `model` (R3): the run's battery, as stage 1 holds
    /// it. A route kept from before was chosen on another model's prices,
    /// so a change forgets them.
    fn price_routes_on(&mut self, model: BatteryModel) {
        if model != self.route_model {
            self.route_model = model;
            if let Some(obstacles) = &self.obstacles {
                obstacles.kept.borrow_mut().clear();
            }
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
        if !self.policy.is_efficient() {
            return None;
        }
        self.obstacles.as_ref()?.route(from, to)
    }

    /// The route's corners, if it turns (or climbs over a power line) at
    /// all.
    fn corners(&self, from: [f64; 3], to: [f64; 3]) -> Option<Corners> {
        self.corners_between(from, to, false)
    }

    /// [`Self::corners`]; `fixed_ends` when neither end moves, so the
    /// outcome is kept for later runs.
    fn corners_between(&self, from: [f64; 3], to: [f64; 3], fixed_ends: bool) -> Option<Corners> {
        let kept = match (fixed_ends, &self.obstacles) {
            (true, Some(obstacles)) if self.policy.is_efficient() => Some(&obstacles.kept),
            _ => None,
        };
        let key = route_key(from, to);
        if let Some(known) = kept.and_then(|kept| kept.borrow().get(&key).cloned()) {
            return known;
        }
        let corners = self.route(from, to).and_then(|route| {
            let points = &route.points_xy_m;
            (points.len() > 2 && self.clears_terrain(points, from[2], to[2], route.length_m))
                .then(|| Corners::turning(points[1..points.len() - 1].to_vec(), route.length_m))
        });
        let corners = self.keep_line_clearance(from, to, corners);
        if let Some(kept) = kept {
            kept.borrow_mut().insert(key, corners.clone());
        }
        corners
    }

    /// `corners`, the route from `from` to `to` round the structures (`None`
    /// for the straight line), made to keep every power line's clearance
    /// under efficient_v1 (R3). A route whose every leg already keeps it,
    /// by the swept test, is returned as it is. Otherwise it is flown over
    /// the wires ([`Self::over_lines`]) or round the line's ends
    /// ([`Self::around_lines`]), whichever costs less energy on the run's
    /// battery model; a route that can do neither is returned as it is.
    fn keep_line_clearance(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        corners: Option<Corners>,
    ) -> Option<Corners> {
        if !self.policy.is_efficient() {
            return corners;
        }
        let Some(obstacles) = self.obstacles.as_ref().filter(|o| o.guards_anything()) else {
            return corners;
        };
        let mut path = corners.as_ref().map_or_else(Vec::new, |c| c.via.clone());
        path.push([to[0], to[1]]);
        let legs = flown_legs(from, &path, &[], to[2]);
        let (ends_from, ends_to) = ([from[0], from[1]], [to[0], to[1]]);
        let lines = obstacles.violates_lines(&legs);
        let crowns = obstacles.violates_crowns(&legs, ends_from, ends_to);
        if !lines && !crowns {
            return corners;
        }
        let length_m = corners.as_ref().map_or_else(
            || {
                let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
                (dx * dx + dy * dy).sqrt()
            },
            |c| c.length_m,
        );
        let over = self
            .over_lines(from, &path, to[2])
            .map(|(mut path, heights)| {
                path.pop();
                Corners {
                    via: path,
                    length_m,
                    heights,
                    lines: lines.then_some("over"),
                    crowns,
                }
            });
        let model = self.route_model;
        let cost = |c: &Corners| {
            model.travel_cost_wh(c.length_m) + c.climb_cost_wh(&model, from[2], to[2])
        };
        // Round a line's ends only when it is a line to keep clear of, and
        // the way round keeps above the crowns too.
        let around = if lines {
            self.around_lines(from, to).filter(|around| {
                let mut path = around.via.clone();
                path.push(ends_to);
                !obstacles.violates_crowns(&flown_legs(from, &path, &[], to[2]), ends_from, ends_to)
            })
        } else {
            None
        };
        match (over, around) {
            (Some(over), Some(around)) if cost(&around) < cost(&over) => Some(around),
            (Some(over), _) => Some(over),
            (None, Some(around)) => Some(around),
            (None, None) => corners,
        }
    }

    /// The flight from `start` through `path` (its corners, then its end at
    /// `end_z`), raised over every power line it would pass too close to:
    /// the path, with a corner added where each climb starts and ends and
    /// where it crosses onto and off each corridor, and the height at every
    /// corner. `None` when the flight still comes too close to a line
    /// (checked leg by leg, by the swept test), enters a building or wall,
    /// or would have to climb at an end: one standing on a corridor below
    /// its crossing height.
    ///
    /// Over each stretch the path crosses a corridor, the flight is at
    /// least the corridor's crossing height. Before it, the flight climbs to
    /// it at [`LINE_RAMP_GRADIENT`] (steeper when the start is nearer), and
    /// after it comes down the same way; elsewhere it is the linear flight.
    /// The heights are that profile's upper envelope at the corners; it is
    /// convex between them (its only concave kinks are the corridor edges,
    /// which are corners), so the legs between the corners never pass below
    /// it. The flight is never lower than the linear one anywhere, so never
    /// nearer the ground.
    fn over_lines(
        &self,
        start: [f64; 3],
        path: &[[f64; 2]],
        end_z: f64,
    ) -> Option<(Vec<[f64; 2]>, Vec<f64>)> {
        let obstacles = self.obstacles.as_ref()?;
        let mut points = vec![[start[0], start[1]]];
        points.extend_from_slice(path);
        let mut along = vec![0.0];
        for pair in points.windows(2) {
            let leg_m = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            along.push(along[along.len() - 1] + leg_m);
        }
        let length_m = along[along.len() - 1];
        if length_m <= 1e-9 {
            return None;
        }
        let (z0, z1) = (start[2], end_z);
        let slope = (z1 - z0) / length_m;
        let linear = |s: f64| z0 + slope * s;
        // Each stretch over a corridor the linear flight is below, as
        // (onto, off, height, climb gradient, descent gradient).
        let mut stretches = Vec::new();
        let ends = ([start[0], start[1]], points[points.len() - 1]);
        for (k, pair) in points.windows(2).enumerate() {
            let leg_m = along[k + 1] - along[k];
            // Crowns (R2): their keep-outs never hold an end, so each has
            // room to climb.
            for (t0, t1, z) in obstacles.crown_stretches(pair[0], pair[1], ends.0, ends.1) {
                let (a, b) = (along[k] + t0 * leg_m, along[k] + t1 * leg_m);
                if z <= linear(a).min(linear(b)) {
                    continue;
                }
                let up = if a > 1e-9 {
                    LINE_RAMP_GRADIENT.max((z - z0) / a)
                } else {
                    LINE_RAMP_GRADIENT
                };
                let down = if b < length_m - 1e-9 {
                    LINE_RAMP_GRADIENT.max((z - z1) / (length_m - b))
                } else {
                    LINE_RAMP_GRADIENT
                };
                stretches.push((a, b, z, up, down));
            }
            for corridor in &obstacles.corridors {
                let Some((t0, t1)) = corridor.clip(pair[0], pair[1]) else {
                    continue;
                };
                let (a, b, z) = (
                    along[k] + t0 * leg_m,
                    along[k] + t1 * leg_m,
                    corridor.over_z_m,
                );
                if z <= linear(a).min(linear(b)) {
                    continue;
                }
                // An end on the corridor below its height cannot climb.
                if (a <= 1e-9 && z > z0) || (b >= length_m - 1e-9 && z > z1) {
                    return None;
                }
                let up = if a > 1e-9 {
                    LINE_RAMP_GRADIENT.max((z - z0) / a)
                } else {
                    LINE_RAMP_GRADIENT
                };
                let down = if b < length_m - 1e-9 {
                    LINE_RAMP_GRADIENT.max((z - z1) / (length_m - b))
                } else {
                    LINE_RAMP_GRADIENT
                };
                stretches.push((a, b, z, up, down));
            }
        }
        if stretches.is_empty() {
            return None;
        }
        let envelope = |s: f64| {
            stretches
                .iter()
                .map(|&(a, b, z, up, down)| {
                    if s < a {
                        z - up * (a - s)
                    } else if s > b {
                        z - down * (s - b)
                    } else {
                        z
                    }
                })
                .fold(linear(s), f64::max)
        };
        // The corners: the path's own, each corridor edge, and where each
        // climb leaves the linear flight and each descent rejoins it.
        let mut knots: Vec<(f64, bool)> = (1..points.len() - 1).map(|k| (along[k], true)).collect();
        for &(a, b, z, up, down) in &stretches {
            knots.push((a, false));
            knots.push((b, false));
            if z > linear(a) && up > slope {
                knots.push((a - (z - linear(a)) / (up - slope), false));
            }
            if z > linear(b) && down + slope > 0.0 {
                knots.push((b + (z - linear(b)) / (down + slope), false));
            }
        }
        knots.retain(|&(s, _)| s > 1e-6 && s < length_m - 1e-6);
        // Path corners first among equals, so a knot on one is the corner.
        knots.sort_by(|x, y| x.0.total_cmp(&y.0).then(y.1.cmp(&x.1)));
        // Knots closer than a micrometre are one, never two corners of the
        // path, and a corner of the path where there is one: no leg of no
        // length, and the path's own shape kept.
        let mut merged: Vec<(f64, bool)> = Vec::with_capacity(knots.len());
        for knot in knots {
            match merged.last_mut() {
                Some(last) if knot.0 - last.0 < 1e-6 && !(knot.1 && last.1) => {
                    if knot.1 {
                        *last = knot;
                    }
                }
                _ => merged.push(knot),
            }
        }
        let mut corners = Vec::with_capacity(merged.len() + 1);
        let mut heights = Vec::with_capacity(merged.len());
        for (s, _) in merged {
            let k = along
                .partition_point(|&d| d <= s)
                .clamp(1, points.len() - 1)
                - 1;
            let leg_m = along[k + 1] - along[k];
            let t = if leg_m > 0.0 {
                (s - along[k]) / leg_m
            } else {
                0.0
            };
            let (p, q) = (points[k], points[k + 1]);
            corners.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            heights.push(envelope(s));
        }
        corners.push(points[points.len() - 1]);
        let legs = flown_legs(start, &corners, &heights, end_z);
        if obstacles.violates_lines(&legs)
            || obstacles.violates_crowns(&legs, ends.0, ends.1)
            || legs.iter().any(|(from, to)| obstacles.crosses(*from, *to))
        {
            return None;
        }
        Some((corners, heights))
    }

    /// The route from `from` to `to` round the power lines' corridors as
    /// well as the structures, past a line's ends, when the planner finds
    /// one that keeps above the terrain and clear of every line by the
    /// swept test.
    fn around_lines(&self, from: [f64; 3], to: [f64; 3]) -> Option<Corners> {
        let obstacles = self.obstacles.as_ref()?;
        let route = obstacles.route_around_lines(from, to)?;
        let points = &route.points_xy_m;
        if points.len() <= 2 || !self.clears_terrain(points, from[2], to[2], route.length_m) {
            return None;
        }
        let via = points[1..points.len() - 1].to_vec();
        let mut path = via.clone();
        path.push([to[0], to[1]]);
        let legs = flown_legs(from, &path, &[], to[2]);
        if obstacles.violates_lines(&legs)
            || legs
                .iter()
                .any(|(start, end)| obstacles.crosses(*start, *end))
        {
            return None;
        }
        Some(Corners {
            lines: Some("around"),
            ..Corners::turning(via, route.length_m)
        })
    }

    /// The free drones matched to the sweeps, each pair the drone with its
    /// flight: least total way there, over only the pairs whose whole flight
    /// is safe and leaves the drone at least its reserve home from where the
    /// sweep ends, as the POI allocator prices it. A nearer drone that cannot
    /// fly a sweep, or cannot afford it, never takes it from a farther one
    /// that can.
    fn assign_gap_sweeps(
        &self,
        free: &[(String, [f64; 3])],
        sweeps: &[Vec<[f64; 2]>],
        battery: &BatteryStage,
    ) -> Vec<(usize, usize, GapPlan)> {
        let model = battery.model();
        let wind = battery.wind_mps();
        let mut plans: Vec<Vec<Option<GapPlan>>> = free
            .iter()
            .map(|(id, position)| {
                let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                sweeps
                    .iter()
                    .map(|sweep| {
                        // Either end may be the better start once routed:
                        // the affordable one with the shorter way there.
                        self.gap_plans(*position, sweep)
                            .into_iter()
                            .filter(|plan| {
                                let end = plan.points[plan.points.len() - 1];
                                let mut route = vec![[position[0], position[1]]];
                                route.extend(plan.via.iter().copied());
                                route.push(end);
                                let length_m: f64 = route
                                    .windows(2)
                                    .map(|leg| (leg[1][0] - leg[0][0]).hypot(leg[1][1] - leg[0][1]))
                                    .sum();
                                let spent_wh = model.travel_cost_wh(length_m)
                                    + climb_along_wh(&model, position[2], &plan.heights, plan.z);
                                let reserve_wh =
                                    self.reserve_for(&model, id, [end[0], end[1], plan.z], wind);
                                remaining_wh - spent_wh >= reserve_wh
                            })
                            .min_by(|a, b| a.approach.total_cmp(&b.approach))
                    })
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

    /// The flights of a drone at `position` to sweep `sweep`, one from each
    /// end that can be made safe (nearer end first): which is better is only
    /// known once each is routed and priced.
    fn gap_plans(&self, position: [f64; 3], sweep: &[[f64; 2]]) -> Vec<GapPlan> {
        let reach = |end: [f64; 2]| (end[0] - position[0]).hypot(end[1] - position[1]);
        let mut reversed = sweep.to_vec();
        reversed.reverse();
        let mut orders = vec![sweep.to_vec()];
        if sweep.len() > 1 {
            orders.push(reversed);
        }
        orders.sort_by(|a, b| reach(a[0]).total_cmp(&reach(b[0])));
        orders
            .iter()
            .filter_map(|points| self.gap_plan_from(position, points))
            .collect()
    }

    /// The safe flight with the shorter way there of [`Self::gap_plans`].
    #[cfg(test)]
    fn gap_plan(&self, position: [f64; 3], sweep: &[[f64; 2]]) -> Option<GapPlan> {
        self.gap_plans(position, sweep)
            .into_iter()
            .min_by(|a, b| a.approach.total_cmp(&b.approach))
    }

    /// The flight of a drone at `position` to sweep `points` in that order,
    /// or `None` when it cannot be made safe: at its own height above the
    /// ground, raised to clear the highest ground the sweep crosses.
    fn gap_plan_from(&self, position: [f64; 3], points: &[[f64; 2]]) -> Option<GapPlan> {
        let mut points = points.to_vec();
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
            points.retain(|p| !obstacles.stands_in(*p, z.min(position[2])));
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
        // routed around the known obstacles, then raised to the lowest end
        // height that keeps it above the known ground everywhere. It climbs
        // linearly from where it starts to that height, never below `z`, so
        // it never flies lower than the lower of `z` and its start: every
        // leg is routed at that height, going around all it could fly
        // beside. Once the height is known, each leg is checked against the
        // structures at the heights it is actually flown. A flight that
        // cannot be made safe is no plan at all, so the assignment never
        // picks it.
        let lowest_z = z.min(position[2]);
        let mut via = self.routed_sweep(&[[position[0], position[1]], points[0]], lowest_z)?;
        via.remove(0);
        via.pop();
        via.extend(self.routed_sweep(&points, lowest_z)?);
        let z = self.sweep_height(position, &via, z)?;
        if !self.misses_structures(position, &via, z) {
            return None;
        }
        // Clear of every power line by its clearance (R3), by the swept test
        // on each leg: over a line the flight climbs, and one that cannot is
        // no plan.
        let mut heights = Vec::new();
        if let Some(obstacles) = self.obstacles.as_ref().filter(|o| o.guards_anything()) {
            let legs = flown_legs(position, &via, &[], z);
            let end = via[via.len() - 1];
            if obstacles.violates_lines(&legs)
                || obstacles.violates_crowns(&legs, [position[0], position[1]], end)
            {
                (via, heights) = self.over_lines(position, &via, z)?;
            }
        }
        via.pop();
        Some(GapPlan {
            points,
            via,
            heights,
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

    /// Whether the flight from `start` through `path`, its height changing
    /// linearly from `start`'s to `end_z` as [`GotoHover::routed`] flies it,
    /// stays out of every known building and wall: each leg intersected
    /// with them at the heights it is flown, not sampled.
    fn misses_structures(&self, start: [f64; 3], path: &[[f64; 2]], end_z: f64) -> bool {
        let Some(obstacles) = &self.obstacles else {
            return true;
        };
        let mut points = vec![[start[0], start[1]]];
        points.extend(path.iter().copied());
        let leg_m = |pair: &[[f64; 2]]| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
        let length_m: f64 = points.windows(2).map(leg_m).sum();
        let z_at = |s: f64| {
            if length_m > 0.0 {
                start[2] + (end_z - start[2]) * (s / length_m).clamp(0.0, 1.0)
            } else {
                start[2]
            }
        };
        let mut flown_m = 0.0;
        for pair in points.windows(2) {
            let leg = leg_m(pair);
            let from = [pair[0][0], pair[0][1], z_at(flown_m)];
            let to = [pair[1][0], pair[1][1], z_at(flown_m + leg)];
            if obstacles.crosses(from, to) {
                return false;
            }
            flown_m += leg;
        }
        true
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
            Some(corners) => corners.length_m,
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
            Some(corners) => {
                model.travel_cost_wh(corners.length_m)
                    + corners.climb_cost_wh(model, from[2], to[2])
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
        battery: &BatteryStage,
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
        let free: Vec<(String, [f64; 3])> = free.into_iter().cloned().collect();
        let assigned = self.assign_gap_sweeps(&free, &sweeps, battery);
        let candidates: Vec<String> = free.iter().map(|(id, _)| id.clone()).collect();
        let speed = self.config.redirect_speed_mps.max(f64::EPSILON);
        for (row, column, plan) in assigned {
            let (id, position) = &free[row];
            let GapPlan {
                points,
                via,
                heights,
                z,
                approach,
                sweep_m,
            } = plan;
            let target = points[points.len() - 1];
            let over_lines = !heights.is_empty();
            let trajectory =
                GotoHover::routed_over(*position, via, heights, target, z, timestamp_s, speed);
            let arrive_s = trajectory.t_arrive();
            self.override_commands
                .push(OverrideCommand::SetFrontier(id.clone(), trajectory));
            let cluster = clusters[column].clone();
            let mut inputs = serde_json::json!({
                "gap_cell_fraction": fraction,
                "cells": cluster.len(),
                "approach_m": approach,
                "sweep_m": sweep_m,
                "position_m": position,
            });
            if over_lines {
                inputs["route_power_lines"] = serde_json::json!("over");
            }
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
                inputs,
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
        let Some(corners) = self.corners_between(at, home, fixed_ends) else {
            return model.dynamic_reserve_wh(at, home, wind, margin);
        };
        let mut points = Vec::with_capacity(corners.via.len() + 2);
        points.push([at[0], at[1]]);
        points.extend(corners.via.iter().copied());
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
        let cost_wh = travel_wh + corners.climb_cost_wh(model, at[2], home[2]);
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
            Some(corners) => Flight {
                route: Some((corners.length_m, corners.via.len())),
                lines: corners.lines,
                crowns: corners.crowns,
                trajectory: GotoHover::routed_over(
                    start,
                    corners.via,
                    corners.heights,
                    [target[0], target[1]],
                    target[2],
                    t_start,
                    speed,
                ),
            },
            None => Flight {
                route: None,
                lines: None,
                crowns: false,
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

    /// A wire mapped since the last run may lie across a flight home (R3):
    /// every drone flying home, on its own reserve or in the mission's
    /// egress, flies home again from where it is, round what is mapped now,
    /// when its way home must change for a line (`egress_reroute`). One whose
    /// way home is clear keeps its flight.
    fn replan_egress_for_wires(
        &mut self,
        mobile: &[(String, [f64; 3])],
        phase: MissionPhase,
        timestamp_s: f64,
    ) {
        let egress = matches!(phase, MissionPhase::Egress | MissionPhase::Complete);
        for (id, position) in mobile {
            if !(egress || self.homeward.contains(id)) {
                continue;
            }
            let Some(home) = self.homes.get(id).copied() else {
                continue;
            };
            let flight = self.flight(*position, home, timestamp_s);
            if flight.lines.is_none() {
                continue;
            }
            let mut inputs = serde_json::json!({ "position_m": position });
            flight.record(&mut inputs);
            self.override_commands
                .push(OverrideCommand::SetEgress(id.clone(), flight.trajectory));
            self.decisions.push(Decision {
                kind: "egress_reroute",
                status: DecisionStatus::QueuedForNextTick,
                subject: Some(id.clone()),
                selected: None,
                candidates: Vec::new(),
                reason: "power_line_mapped",
                detail: format!(
                    "{id} flies home again from where it is: a power line mapped since the \
                     last run lies across its way"
                ),
                inputs,
                command: Some("set_egress"),
            });
        }
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

    /// Fly every drone sent home on its own reserve home again, from where
    /// it is, after the mission's egress drops every recall at once.
    fn keep_reserve_returns(&mut self, mobile: &[(String, [f64; 3])], at_s: f64) {
        for (id, position) in mobile {
            if !self.homeward.contains(id) {
                continue;
            }
            let Some(home) = self.homes.get(id).copied() else {
                continue;
            };
            let flight = self.flight(*position, home, at_s);
            let mut inputs = serde_json::json!({ "position_m": position });
            flight.record(&mut inputs);
            self.override_commands
                .push(OverrideCommand::SetEgress(id.clone(), flight.trajectory));
            self.decisions.push(Decision {
                kind: "return_home",
                status: DecisionStatus::QueuedForNextTick,
                subject: Some(id.clone()),
                selected: None,
                candidates: Vec::new(),
                reason: "reserve_return_kept",
                detail: format!(
                    "the mission's recall ended, but {id} went home on its own reserve: it \
                     stays on its way home"
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
        // Under contingency_v1 (F5) a holder gives its POI up when it can no
        // longer afford what is left of it and the trip home, whatever its
        // fraction; otherwise below the fixed threshold.
        let contingency = self.policy.plans_contingencies();
        let cannot_finish = if contingency {
            self.holders_that_cannot_finish(mobile, battery)
        } else {
            BTreeSet::new()
        };
        // A drone at its reserve is already homeward; say so, rather than
        // that its battery is low, whatever its fraction.
        let homeward = self.homeward.clone();
        let giving_up = |id: &str| {
            homeward.contains(id)
                || if contingency {
                    cannot_finish.contains(id)
                } else {
                    fraction_of(id) < threshold
                }
        };
        let why = |id: &str| -> (GiveUp, String) {
            if homeward.contains(id) {
                (
                    GiveUp::Reserve,
                    format!("{id} at its reserve and heading home"),
                )
            } else if contingency {
                (
                    GiveUp::CannotFinish,
                    format!("{id} can no longer afford the dwell left and the trip home"),
                )
            } else {
                (GiveUp::Low, format!("{id} below {threshold} of capacity"))
            }
        };
        // New work needs this much above the reserve (F5's hysteresis).
        let take_margin_wh = self.take_margin_wh(&battery.model());
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
            let (given_up, cause) = why(teammate);
            let at_reserve = given_up == GiveUp::Reserve;
            self.pois.release(teammate);
            self.decisions.push(Decision {
                kind: "poi_release",
                status: DecisionStatus::Applied,
                subject: Some(teammate.clone()),
                selected: poi.clone(),
                candidates: Vec::new(),
                reason: match given_up {
                    GiveUp::Reserve => "reserve_teammate",
                    GiveUp::Low => "low_battery_teammate",
                    GiveUp::CannotFinish => "cannot_finish_teammate",
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
                        (remaining_wh - spent_wh >= reserve_wh + take_margin_wh).then_some(spent_wh)
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
            let (given_up, cause) = why(lead);
            let at_reserve = given_up == GiveUp::Reserve;
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
                        reason: match given_up {
                            GiveUp::Reserve => "reserve_handoff",
                            GiveUp::Low => "safe_handoff",
                            GiveUp::CannotFinish => "cannot_finish_handoff",
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

    /// The margin above its reserve a drone keeps to take new work: none but
    /// under contingency_v1 (F5, [`CONTINGENCY_TAKE_MARGIN_FRACTION`]).
    fn take_margin_wh(&self, model: &BatteryModel) -> f64 {
        if self.policy.plans_contingencies() {
            model.capacity_wh * CONTINGENCY_TAKE_MARGIN_FRACTION
        } else {
            0.0
        }
    }

    /// The drones holding a POI that can no longer afford to finish it and
    /// fly home (F5): the trip there from where they are (none once in
    /// range), their share of the dwell left with the holders already in
    /// range, and the reserve home from the POI. Checked every run, so a
    /// headwind or a longer way home moves the work as soon as it costs too
    /// much, not when the battery falls past a fixed fraction.
    fn holders_that_cannot_finish(
        &self,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
    ) -> BTreeSet<String> {
        let model = battery.model();
        let wind = battery.wind_mps();
        let active = self.pois.active_pois();
        let in_range = |position: &[f64; 3], poi: [f64; 3]| {
            let (dx, dy) = (poi[0] - position[0], poi[1] - position[1]);
            (dx * dx + dy * dy).sqrt() <= POI_DWELL_RANGE_M
        };
        // To a fixed point: a teammate that gives the POI up (homeward, or
        // unable to finish itself) shares none of the dwell, which can leave
        // a drone that relied on it unable to finish too. The set only
        // grows, so this ends.
        let mut cannot = BTreeSet::new();
        loop {
            let leaving = |other: &String| cannot.contains(other) || self.homeward.contains(other);
            let mut newly = Vec::new();
            for (id, position) in mobile {
                if cannot.contains(id) {
                    continue;
                }
                let Some(held) = self.pois.assignment_for(id) else {
                    continue;
                };
                let Some(poi) = active.iter().find(|poi| poi.poi_id == held) else {
                    continue;
                };
                let sharing = mobile
                    .iter()
                    .filter(|(other, other_position)| {
                        other != id
                            && !leaving(other)
                            && self.pois.assignment_for(other) == Some(held)
                            && in_range(other_position, poi.position)
                    })
                    .count();
                let hover_s = poi.remaining_dwell_s.max(0.0) / (sharing + 1) as f64;
                let trip_wh = if in_range(position, poi.position) {
                    0.0
                } else {
                    self.trip_cost_wh(&model, *position, poi.position)
                };
                let spent_wh = trip_wh + model.hover_power_w * hover_s / 3600.0;
                let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                if remaining_wh - spent_wh < self.reserve_for(&model, id, poi.position, wind) {
                    newly.push(id.clone());
                }
            }
            // Added after the pass, so one pass judges a whole team on the
            // same teammates.
            if newly.is_empty() {
                return cannot;
            }
            cannot.extend(newly);
        }
    }

    /// Under contingency_v1 (F5), mark the pending POIs that no drone still
    /// working (not homeward) could afford, from where it is now, with the
    /// margin new work needs; and clear the mark from one that some drone
    /// could afford again. Each change is a decision. A marked POI is
    /// reported unassigned and no longer holds the mission open.
    fn mark_unaffordable(
        &mut self,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
        timestamp_s: f64,
    ) {
        let model = battery.model();
        let wind = battery.wind_mps();
        let margin_wh = self.take_margin_wh(&model);
        let mut pending = self.pois.pending_pois();
        pending.sort_by(|a, b| a.poi_id.cmp(&b.poi_id).then(a.index.cmp(&b.index)));
        let workers: Vec<&(String, [f64; 3])> = mobile
            .iter()
            .filter(|(id, _)| !self.homeward.contains(id.as_str()))
            .collect();
        let mut now = BTreeSet::new();
        for poi in &pending {
            let affordable = workers.iter().any(|(id, position)| {
                let spent_wh = self.trip_cost_wh(&model, *position, poi.position)
                    + model.hover_power_w * poi.required_dwell_s.max(0.0) / 3600.0;
                let remaining_wh = battery.remaining_wh(id).unwrap_or(model.capacity_wh);
                remaining_wh - spent_wh
                    >= self.reserve_for(&model, id, poi.position, wind) + margin_wh
            });
            if !affordable {
                now.insert(poi.poi_id.clone());
            }
        }
        let candidates: Vec<String> = workers.iter().map(|(id, _)| id.clone()).collect();
        for poi in now.difference(&self.unassigned) {
            self.decisions.push(Decision {
                kind: "poi_unassigned",
                status: DecisionStatus::Applied,
                subject: None,
                selected: Some(poi.clone()),
                candidates: candidates.clone(),
                reason: "no_affordable_drone",
                detail: format!(
                    "no drone still working can afford {poi}'s trip and dwell and the trip \
                     home with the margin new work needs: it is reported unassigned and \
                     no longer holds the mission open"
                ),
                inputs: serde_json::json!({
                    "timestamp_s": timestamp_s,
                    "take_margin_wh": margin_wh,
                    "homeward": self.homeward,
                }),
                command: None,
            });
        }
        for poi in self.unassigned.difference(&now) {
            self.decisions.push(Decision {
                kind: "poi_unassigned",
                status: DecisionStatus::Applied,
                subject: None,
                selected: Some(poi.clone()),
                candidates: candidates.clone(),
                reason: "affordable_again",
                detail: format!("a drone can afford {poi} again: it is pending once more"),
                inputs: serde_json::json!({ "timestamp_s": timestamp_s }),
                command: None,
            });
        }
        self.unassigned = now;
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
        // Under contingency_v1 affordability alone decides (F5).
        let contingency = self.policy.plans_contingencies();
        let low = |id: &str| !contingency && fraction_of(id) < threshold;
        let take_margin_wh = self.take_margin_wh(&battery.model());
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
                        (remaining_wh - spent_wh >= reserve_wh + take_margin_wh)
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
                    (contingency || fraction_of(id) >= self.config.handoff_low_fraction) && {
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
                            >= self.reserve_for(&model, id, poi.position, wind) + take_margin_wh)
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
            unassigned: self.unassigned.clone(),
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
        self.price_routes_on(battery.model());
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
        let efficient = self.policy.is_efficient();
        let mut pending_before = self.pending_poi_ids();
        let mut joins = Vec::new();
        if efficient {
            // In the order the allocator considers them, by id.
            pending_before.sort();
            joins = self.allocate_fleet_wide(mobile, battery);
            if self.policy.plans_contingencies() {
                self.mark_unaffordable(mobile, battery, timestamp_s);
            }
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
                } else if self.replan_for_wires {
                    // A wire mapped since the last run may lie across the
                    // flight there: fly it again from here, round what is
                    // mapped now.
                    let poi_position = self
                        .initial_pois
                        .iter()
                        .find(|p| p.poi_id == active)
                        .expect("an assigned POI is one of the manager's")
                        .position;
                    let flight = self.flight(*position, poi_position, timestamp_s);
                    if flight.lines.is_some() {
                        let mut inputs = serde_json::json!({ "position_m": position });
                        flight.record(&mut inputs);
                        self.override_commands
                            .push(OverrideCommand::SetPoi(id.clone(), flight.trajectory));
                        self.decisions.push(Decision {
                            kind: "poi_reroute",
                            status: DecisionStatus::QueuedForNextTick,
                            subject: Some(id.clone()),
                            selected: Some(active.clone()),
                            candidates: Vec::new(),
                            reason: "power_line_mapped",
                            detail: format!(
                                "{id} flies to {active} again from where it is: a power line \
                                 mapped since the last run lies across its way"
                            ),
                            inputs,
                            command: Some("set_poi"),
                        });
                    }
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
            obstacles.around_this_run.borrow_mut().clear();
        }
        let empty = BatteryStage::new();
        let battery = stage_one_battery(context.stages).unwrap_or(&empty);
        self.price_routes_on(battery.model());
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

        // The wires the self-built map holds now (R3), before any flight is
        // planned this run.
        self.replan_for_wires = self.policy.is_efficient() && self.follow_mapped_wires();
        if self.replan_for_wires {
            self.replan_egress_for_wires(&mobile, phase, timestamp_s);
        }
        if self.policy.is_efficient() {
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
        let holes = (gate_open && belief.is_none() && self.policy.is_efficient())
            .then(|| gap_cells(mapping));
        let gap_cell_fraction = gate_open.then(|| match (&belief, &holes) {
            (Some(belief), _) => {
                belief.enclosed_unexplored_fraction(crate::frontier::UNEXPLORED_SHARE)
            }
            (None, Some((cells, total))) => cells.len() as f64 / (*total).max(1) as f64,
            (None, None) => gap_cell_fraction(mapping),
        });
        // Under contingency_v1 a POI no drone can afford is closed for the
        // mission (F5): reported unassigned, it does not hold it open.
        let all_pois_complete = if self.policy.plans_contingencies() {
            self.pois
                .statuses()
                .iter()
                .all(|s| s.state == PoiState::Complete || self.unassigned.contains(&s.poi_id))
        } else {
            self.pois.all_complete()
        };
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
                MissionEvent::EgressRoutesCleared { at_s } => {
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
                    // The mission's recall is over, but a drone sent home on
                    // its own reserve still has to stay there: dropped with
                    // the rest, it would fly its lanes again until its
                    // battery ran out, since a homeward drone is never sent
                    // home twice.
                    if self.policy.is_efficient() {
                        self.keep_reserve_returns(&mobile, at_s);
                    }
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
        if self.policy.is_efficient() {
            self.fill_gaps(
                holes,
                gap_cell_fraction,
                &mobile,
                mapping,
                battery,
                timestamp_s,
            );
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
            mapped_wires: self.mapped_wires,
            unassigned: self.unassigned.clone(),
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
        self.mapped_wires = decoded.mapped_wires;
        self.unassigned = decoded.unassigned;
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
        if self.policy.is_efficient() {
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
        if self.mapped_layer {
            self.obstacles = None;
            self.mapped_layer = false;
        }
        self.mapped_installed = 0;
        self.mapped_wires = 0;
        self.unassigned.clear();
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
            &[
                ("near".to_string(), near_but_unsafe),
                ("far".to_string(), far_and_safe),
            ],
            std::slice::from_ref(&sweep),
            &BatteryStage::new(),
        );
        assert_eq!(assigned.len(), 1);
        let (row, column, plan) = &assigned[0];
        assert_eq!((*row, *column), (1, 0));
        assert_eq!(plan.points, sweep);
        // Its flight climbs over the ridge on the way.
        assert!(plan.z > 40.0);
    }

    /// A hole goes only to a drone that can fly the sweep and still get
    /// home on its reserve, priced as the POI allocator prices a trip: the
    /// near drone holding just its reserve is passed over for a far one
    /// with a full battery.
    #[test]
    fn a_hole_goes_only_to_a_drone_that_can_afford_the_sweep() {
        let stage = efficient_driver(Vec::new());
        let (mut battery, _) = fleet(&[("near", 0.0), ("far", 0.0)]);
        let model = battery.model();
        assert!(battery.set_remaining_wh("near", model.capacity_wh * model.reserve_fraction));
        let sweep = vec![[600.0, 0.0], [650.0, 0.0]];
        let free = [
            ("near".to_string(), [550.0, 0.0, 40.0]),
            ("far".to_string(), [0.0, 0.0, 40.0]),
        ];
        let assigned = stage.assign_gap_sweeps(&free, std::slice::from_ref(&sweep), &battery);
        assert_eq!(assigned.len(), 1);
        assert_eq!((assigned[0].0, assigned[0].1), (1, 0));
        // With a full battery the near one takes it.
        assert!(battery.set_remaining_wh("near", model.capacity_wh));
        let assigned = stage.assign_gap_sweeps(&free, std::slice::from_ref(&sweep), &battery);
        assert_eq!((assigned[0].0, assigned[0].1), (0, 0));
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

    /// F5: a teammate that cannot finish shares none of the dwell, so a
    /// drone that could afford the POI only with it cannot either: the
    /// test runs to a fixed point. 600 s of dwell at 300 W is 50 Wh alone,
    /// 25 Wh shared; the reserve floor is 20 Wh.
    #[test]
    fn contingency_v1_drops_a_holder_that_relied_on_a_leaving_teammate() {
        let (mut battery, mobile) = fleet(&[("drone-a", 500.0), ("drone-b", 510.0)]);
        let mut stage = driver(vec![InspectionPoi {
            required_dwell_s: 600.0,
            ..poi("poi-00", 500.0, 1)
        }]);
        stage.set_policy(PlanningPolicy::ContingencyV1).unwrap();
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        assert_eq!(stage.pois.assignment_for("drone-b"), Some("poi-00"));
        // a: 30 Wh, short even sharing (25 + 20). b: 46 Wh, enough sharing
        // (25 + 20) but not alone (50 + 20).
        assert!(battery.set_remaining_wh("drone-a", 30.0));
        assert!(battery.set_remaining_wh("drone-b", 46.0));
        let cannot = stage.holders_that_cannot_finish(&mobile, &battery);
        assert_eq!(
            cannot.into_iter().collect::<Vec<_>>(),
            ["drone-a", "drone-b"]
        );
        // With a teammate that can finish, b keeps it.
        assert!(battery.set_remaining_wh("drone-a", 60.0));
        assert!(stage.holders_that_cannot_finish(&mobile, &battery).is_empty());
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

    /// The end nearer in a straight line is not always the better start:
    /// here it is 100 m away across the wall (a detour round its end), the
    /// other 250 m away on the drone's own side. Both are planned, and the
    /// hole is swept from the end with the shorter way there.
    #[test]
    fn a_sweep_starts_from_the_end_with_the_shorter_routed_way_there() {
        let mut stage = efficient_driver(Vec::new());
        long_walled(&mut stage);
        let position = [-50.0, 0.0, 40.0];
        let sweep = vec![[50.0, 0.0], [-300.0, 0.0]];
        let plans = stage.gap_plans(position, &sweep);
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].points[0], [50.0, 0.0], "nearer end planned first");
        assert!(plans[0].approach > 1000.0, "{}", plans[0].approach);
        let (battery, _) = fleet(&[("d", 0.0)]);
        let assigned = stage.assign_gap_sweeps(
            &[("d".to_string(), position)],
            std::slice::from_ref(&sweep),
            &battery,
        );
        let plan = &assigned[0].2;
        assert_eq!(plan.points[0], [-300.0, 0.0]);
        assert!((plan.approach - 250.0).abs() < 1e-6, "{}", plan.approach);
    }

    /// Ground 200 m up east of x = 200.
    #[derive(Debug)]
    struct Shelf;

    impl crate::observe::TerrainHeight for Shelf {
        fn height_at(&self, x_m: f64, _y_m: f64) -> f64 {
            if x_m >= 200.0 {
                200.0
            } else {
                0.0
            }
        }
    }

    /// The sweep ends 240 m up (40 m over the shelf), but climbs there from
    /// 40 m, so it crosses the 120 m wall at x = 0 well below that height:
    /// its legs are routed at the lowest height it flies, round the wall's
    /// end, and the flight, checked at the heights it is flown, enters
    /// nothing.
    #[test]
    fn a_gap_sweep_is_routed_at_the_heights_it_climbs_through() {
        let mut stage = efficient_driver(Vec::new());
        long_walled(&mut stage);
        stage.set_known_terrain(Some(Arc::new(Shelf)));
        let start = [-150.0, 0.0, 40.0];
        let plan = stage
            .gap_plan(start, &[[-100.0, 0.0], [300.0, 0.0]])
            .expect("a safe sweep");
        assert!(plan.z >= 240.0, "{}", plan.z);
        assert!(
            plan.via.iter().any(|p| p[1].abs() > 900.0),
            "round the wall: {:?}",
            plan.via
        );
        let mut path = plan.via.clone();
        path.push(*plan.points.last().unwrap());
        assert!(stage.misses_structures(start, &path, plan.z));
        // Straight across, ending at 240 m, it would cross the wall at
        // 40 + 200 / 3 m, below its top.
        assert!(!stage.misses_structures(start, &[[-100.0, 0.0], [300.0, 0.0]], 240.0));
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

    /// A power line across the x axis at x = 300 (R3): supports every
    /// 400 m from y = -600 to 600, `height_m` tall on flat ground, with
    /// `offsets` conductors sagging 3 % of the span and a 10 m clearance, in
    /// a 2 km square with no structures.
    fn power_line(supports_y: &[f64], height_m: f64, offsets: &[f64]) -> PowerLine {
        PowerLine {
            id: "line".to_string(),
            supports: supports_y
                .iter()
                .map(|&y| argusnet_world::powerline::Support {
                    xy_m: [300.0, y],
                    base_z_m: 0.0,
                    height_m,
                })
                .collect(),
            conductor_offsets_m: offsets.to_vec(),
            sag_ratio: 0.03,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .expect("a valid line")
    }

    fn wired(stage: &mut MissionDriverStage, line: PowerLine) {
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let layer = ObstacleLayer::new(bounds, 250.0, Vec::new()).expect("valid layer");
        stage.set_known_obstacles(Some(Arc::new(layer)), bounds);
        stage.set_known_power_lines(vec![line]);
    }

    /// Every leg a flight flies, with the heights it flies them at.
    fn legs_of(flight: &GotoHover, start: [f64; 3]) -> Vec<([f64; 3], [f64; 3])> {
        let mut path = flight.via_xy().to_vec();
        let target = flight.target();
        path.push([target[0], target[1]]);
        flown_legs(start, &path, flight.via_z(), target[2])
    }

    /// R3, a run without a prior map: flights know no line until one is
    /// mapped. The run that maps one hands the routes the span and flies
    /// again the POI flight it lies across, from where the drone is, over
    /// the wires; later runs that map nothing new leave it be, and a reset
    /// forgets the routes' knowledge again.
    #[test]
    fn a_mapped_wire_replans_the_poi_flight_it_lies_across() {
        use crate::belief_world::{BeliefGrid, BeliefHandle, ExplorationConfig};
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let config = ExplorationConfig::default();
        let mut grid = BeliefGrid::new(bounds, 10.0, 0.0).unwrap();
        let map = BeliefHandle::new(grid.snapshot(&config));
        let (battery, mut mobile) = fleet(&[("drone-a", 100.0)]);
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        stage.set_mapped_power_lines(Some(map.clone()), bounds);

        assert!(!stage.follow_mapped_wires(), "nothing mapped yet");
        assert!(!stage.knows_obstacles());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let first = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(first.via_xy().is_empty(), "straight: no line is known");

        // A scan near the line maps the span it crosses.
        let mut scan = crate::range_scan::RangeScan {
            node_id: "drone-a".into(),
            timestamp_s: 1.0,
            origin_m: [280.0, 0.0, 60.0],
            returns: Vec::new(),
            misses: 0,
            wires: vec![crate::range_scan::WireReturn {
                span: line.span_line(1),
                point_m: [300.0, 0.0, 39.0],
            }],
        };
        grid.integrate(&scan, 4);
        map.publish(grid.snapshot(&config));
        mobile[0].1 = [150.0, 0.0, mobile[0].1[2]];
        stage.replan_for_wires = stage.follow_mapped_wires();
        assert!(stage.replan_for_wires);
        stage.inspect(5.0, 1.0, &mobile, &battery, &empty_map());
        let again = poi_flight(&stage.take_override_commands(), "drone-a");
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_reroute")
            .expect("replanned");
        assert_eq!(decision.reason, "power_line_mapped");
        assert_eq!(decision.inputs["route_power_lines"], "over");
        let span = line.span_line(1);
        for (from, to) in legs_of(&again, mobile[0].1) {
            assert!(!span.violated_by(from, to), "{from:?} -> {to:?}");
        }

        // Nothing new: no replan, even with the scan seen again.
        grid.integrate(&scan, 5);
        map.publish(grid.snapshot(&config));
        stage.replan_for_wires = stage.follow_mapped_wires();
        assert!(!stage.replan_for_wires);
        stage.inspect(6.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.take_override_commands().is_empty());

        // A span mapped where the flight no longer crosses: no replan.
        scan.wires[0].span = line.span_line(0);
        grid.integrate(&scan, 6);
        map.publish(grid.snapshot(&config));
        mobile[0].1 = [450.0, 0.0, 40.0];
        stage.replan_for_wires = stage.follow_mapped_wires();
        assert!(stage.replan_for_wires);
        stage.inspect(7.0, 1.0, &mobile, &battery, &empty_map());
        assert!(stage.take_override_commands().is_empty());

        // How many were mapped is checkpointed: a resumed run that maps no
        // more replans nothing.
        let state = StageHandler::checkpoint_state(&stage).unwrap();
        assert_eq!(state["mapped_wires"], 2);
        let mut resumed = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        resumed.set_mapped_power_lines(Some(map.clone()), bounds);
        StageHandler::restore_state(&mut resumed, &state).unwrap();
        assert!(!resumed.follow_mapped_wires());
        assert!(
            resumed.knows_obstacles(),
            "the spans are given to the routes"
        );

        StageHandler::reset(&mut stage);
        assert!(!stage.knows_obstacles(), "a reset forgets the mapped wires");
    }
    /// R3: a drone already flying home (on its own reserve, or in the
    /// mission's egress) flies home again round a wire mapped across its way;
    /// one whose way home is clear keeps its flight, and a drone still at
    /// work is left to the POI replan.
    #[test]
    fn a_mapped_wire_replans_the_flights_home_it_lies_across() {
        use crate::belief_world::{BeliefGrid, BeliefHandle, ExplorationConfig};
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let config = ExplorationConfig::default();
        let mut grid = BeliefGrid::new(bounds, 10.0, 0.0).unwrap();
        grid.know_wires(vec![line.span_line(1)]);
        let homes: BTreeMap<String, [f64; 3]> = [
            ("drone-a".to_string(), [0.0, 0.0, 40.0]),
            ("drone-b".to_string(), [0.0, 0.0, 40.0]),
            ("drone-c".to_string(), [0.0, 0.0, 40.0]),
        ]
        .into_iter()
        .collect();
        let mut stage = MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), homes);
        stage.set_policy(PlanningPolicy::EfficientV1).unwrap();
        stage.set_mapped_power_lines(Some(BeliefHandle::new(grid.snapshot(&config))), bounds);
        assert!(stage.follow_mapped_wires());
        // a: homeward beyond the line; b: homeward on the near side; c: at
        // work beyond the line.
        stage.homeward.insert("drone-a".into());
        stage.homeward.insert("drone-b".into());
        let mobile = vec![
            ("drone-a".to_string(), [500.0, 0.0, 40.0]),
            ("drone-b".to_string(), [100.0, 0.0, 40.0]),
            ("drone-c".to_string(), [500.0, 0.0, 40.0]),
        ];
        stage.replan_egress_for_wires(&mobile, MissionPhase::Inspecting, 3.0);
        let commands = stage.take_override_commands();
        assert_eq!(commands.len(), 1, "{commands:?}");
        let OverrideCommand::SetEgress(id, flight) = &commands[0] else {
            panic!("{commands:?}");
        };
        assert_eq!(id, "drone-a");
        let span = line.span_line(1);
        for (from, to) in legs_of(flight, mobile[0].1) {
            assert!(!span.violated_by(from, to), "{from:?} -> {to:?}");
        }
        let decision = &stage.decisions()[0];
        assert_eq!(
            (decision.kind, decision.reason),
            ("egress_reroute", "power_line_mapped")
        );
        // In the mission's egress every drone flying home is replanned.
        stage.decisions.clear();
        stage.replan_egress_for_wires(&mobile, MissionPhase::Egress, 4.0);
        let ids: Vec<String> = stage
            .take_override_commands()
            .into_iter()
            .map(|c| match c {
                OverrideCommand::SetEgress(id, _) => id,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(ids, ["drone-a", "drone-c"]);
    }
    /// The run itself replans: one tick of a run whose map holds a span
    /// across a homeward drone's way flies it home again.
    #[test]
    fn a_run_that_maps_a_wire_replans_the_flight_home() {
        use crate::belief_world::{BeliefGrid, BeliefHandle, ExplorationConfig};
        use crate::stage::Stage;
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let mut grid = BeliefGrid::new(bounds, 10.0, 0.0).unwrap();
        grid.know_wires(vec![line.span_line(1)]);
        let map = BeliefHandle::new(grid.snapshot(&ExplorationConfig::default()));
        let mut sim = gate_rig_with(Vec::new(), Vec::new());
        let homes = [("drone-1".to_string(), [480.0, 25.0, 10.0])]
            .into_iter()
            .collect();
        let mut stage = MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), homes);
        stage.set_policy(PlanningPolicy::EfficientV1).unwrap();
        stage.set_mapped_power_lines(Some(map), bounds);
        stage.homeward.insert("drone-1".into());
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
            .expect("stage 9");
        sim.step().expect("tick runs");
        assert!(
            installed_driver(&sim)
                .decisions()
                .iter()
                .any(|d| d.kind == "egress_reroute"),
            "{:?}",
            installed_driver(&sim).decisions()
        );
    }
    /// R3: under efficient_v1 a POI redirect whose straight line passes 7 m
    /// over a three-wire line climbs over it with the clearance and comes
    /// down to the POI, every leg clear of every wire by the swept test, and
    /// the journal says so. It does not go round: the line's ends are 600 m
    /// off.
    #[test]
    fn efficient_v1_climbs_a_poi_redirect_over_a_power_line() {
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let start = mobile[0].1;
        let target = [500.0, 0.0, 40.0];
        assert!(
            line.violated_by(start, target),
            "the straight line crosses low"
        );
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        wired(&mut stage, line.clone());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");

        assert!(!flight.via_z().is_empty(), "the flight climbs");
        assert_eq!(flight.via_z().len(), flight.via_xy().len());
        assert!(
            flight.via_xy().iter().all(|p| p[1].abs() < 1e-9),
            "straight on, not round: {:?}",
            flight.via_xy()
        );
        assert!((flight.path_length_m() - 400.0).abs() < 1e-9);
        for (from, to) in legs_of(&flight, start) {
            assert!(
                line.clearance_along(from, to) >= line.clearance_m,
                "{from:?} -> {to:?}: {} m",
                line.clearance_along(from, to)
            );
        }
        // Higher than the top wire by the clearance over the line.
        let highest = flight
            .via_z()
            .iter()
            .fold(f64::NEG_INFINITY, |m, z| m.max(*z));
        assert!(highest >= 45.0 + 10.0, "{highest}");
        let (arrived, stopped) = flight.sample(flight.t_arrive());
        assert_eq!(arrived, target);
        assert_eq!(stopped, [0.0, 0.0, 0.0]);

        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.inputs["route_power_lines"], "over");
        assert_eq!(decision.inputs["route_length_m"], 400.0);
    }

    /// R3: a short line, a single wire on two supports 40 m apart, is
    /// cheaper to fly round than to climb 11 m over and come down again, so
    /// the redirect goes past its end, clear of the wire on every leg and
    /// at its own height.
    #[test]
    fn efficient_v1_routes_a_poi_redirect_round_a_short_line() {
        let line = power_line(&[-20.0, 20.0], 40.0, &[0.0]);
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let start = mobile[0].1;
        assert!(line.violated_by(start, [500.0, 0.0, 40.0]));
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        wired(&mut stage, line.clone());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");

        assert!(flight.via_z().is_empty(), "level, not over");
        assert!(
            flight.via_xy().iter().any(|p| p[1].abs() > 20.0),
            "past an end: {:?}",
            flight.via_xy()
        );
        assert!(flight.path_length_m() > 400.0);
        for (from, to) in legs_of(&flight, start) {
            assert!(!line.violated_by(from, to), "{from:?} -> {to:?}");
        }
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.inputs["route_power_lines"], "around");
    }

    /// R3: over or round is priced on the run's own battery. On a platform
    /// whose level flight costs ten times the default's and whose climbs
    /// cost a tenth, the same short line is climbed over instead.
    #[test]
    fn over_or_round_is_priced_on_the_runs_battery() {
        let line = power_line(&[-20.0, 20.0], 40.0, &[0.0]);
        let base = BatteryModel::default();
        let mut battery = BatteryStage::new();
        battery.set_model(BatteryModel {
            travel_power_w: 10.0 * base.travel_power_w,
            climb_power_w: 0.1 * base.climb_power_w,
            hover_power_w: 0.1 * base.hover_power_w,
            capacity_wh: 100.0 * base.capacity_wh,
            ..base
        });
        battery.register("drone-a", None);
        let mobile = vec![("drone-a".to_string(), [100.0, 0.0, 40.0])];
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        wired(&mut stage, line.clone());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(!flight.via_z().is_empty(), "over, not round");
        for (from, to) in legs_of(&flight, mobile[0].1) {
            assert!(!line.violated_by(from, to), "{from:?} -> {to:?}");
        }
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.inputs["route_power_lines"], "over");
    }

    /// R3: the reference never reads the lines, and a flight high over one
    /// is flown as it was.
    #[test]
    fn only_a_flight_too_close_to_a_line_changes() {
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let straight = GotoHover::redirect(mobile[0].1, [500.0, 0.0], 40.0, 0.0, 28.0);

        let mut reference = driver(vec![poi("poi-00", 500.0, 1)]);
        wired(&mut reference, line.clone());
        reference.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(
            poi_flight(&reference.take_override_commands(), "drone-a"),
            straight
        );

        let high = vec![("drone-a".to_string(), [100.0, 0.0, 200.0])];
        let mut stage = efficient_driver(vec![InspectionPoi {
            position: [500.0, 0.0, 200.0],
            ..poi("poi-00", 500.0, 1)
        }]);
        wired(&mut stage, line);
        stage.inspect(0.0, 1.0, &high, &battery, &empty_map());
        assert_eq!(
            poi_flight(&stage.take_override_commands(), "drone-a"),
            GotoHover::redirect([100.0, 0.0, 200.0], [500.0, 0.0], 200.0, 0.0, 28.0)
        );
    }

    /// R3: a gap sweep that would cross a line low climbs over it, every
    /// leg clear by the swept test, and is priced with the climb.
    #[test]
    fn a_gap_sweep_climbs_over_a_power_line() {
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let mut stage = efficient_driver(Vec::new());
        wired(&mut stage, line.clone());
        let start = [100.0, 0.0, 40.0];
        let plan = stage
            .gap_plan(start, &[[150.0, 0.0], [450.0, 0.0]])
            .expect("a safe sweep");
        assert_eq!(plan.heights.len(), plan.via.len());
        assert!(!plan.heights.is_empty(), "it climbs");
        let mut path = plan.via.clone();
        path.push(*plan.points.last().unwrap());
        for (from, to) in flown_legs(start, &path, &plan.heights, plan.z) {
            assert!(!line.violated_by(from, to), "{from:?} -> {to:?}");
        }
        assert!(line.violated_by([150.0, 0.0, 40.0], [450.0, 0.0, 40.0]));
    }

    /// R3: the profile over a corridor is the envelope of the linear
    /// flight and the climb: a flight already high enough at the crossing
    /// is not raised, and a start on a corridor below its height cannot
    /// climb, so there is no flight over.
    #[test]
    fn a_flight_over_a_line_is_raised_only_where_it_must_be() {
        let line = power_line(&[-600.0, -200.0, 200.0, 600.0], 45.0, &[-4.0, 0.0, 4.0]);
        let mut stage = efficient_driver(Vec::new());
        wired(&mut stage, line);
        // Descending from 300 m to 40 m over 400 m: over the line, 100 m
        // in, it is still 235 m up, so there is nothing to raise.
        assert!(stage
            .over_lines([200.0, 0.0, 300.0], &[[600.0, 0.0]], 40.0)
            .is_none());
        // Standing on the corridor, below the wires' clearance.
        assert!(stage
            .over_lines([300.0, 0.0, 40.0], &[[600.0, 0.0]], 40.0)
            .is_none());
        // From 50 m out the climb is steeper than 1 in 8, but it still
        // clears.
        let (path, heights) = stage
            .over_lines([250.0, 0.0, 40.0], &[[350.0, 0.0]], 40.0)
            .expect("over");
        assert_eq!(path.last(), Some(&[350.0, 0.0]));
        let over = heights.iter().fold(f64::NEG_INFINITY, |m, z| m.max(*z));
        assert!(over >= 45.0 + 10.0);
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

    /// A broadleaf on flat ground at `(x, y)`, `height_m` tall with a crown
    /// of radius `radius_m` from half its height.
    fn tree(id: &str, x: f64, y: f64, height_m: f64, radius_m: f64) -> argusnet_world::trees::Tree {
        argusnet_world::trees::Tree {
            id: id.to_string(),
            xy_m: [x, y],
            base_z_m: 0.0,
            height_m,
            crown_base_m: 0.5 * height_m,
            crown_radius_m: radius_m,
            trunk_radius_m: 0.4,
            species: argusnet_world::trees::Species::Broadleaf,
            leaf_density: 0.9,
        }
    }

    fn treed(stage: &mut MissionDriverStage, trees: Vec<argusnet_world::trees::Tree>) {
        let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
        let layer = ObstacleLayer::new(bounds, 250.0, Vec::new()).expect("valid layer");
        stage.set_known_obstacles(Some(Arc::new(layer)), bounds);
        stage.set_known_trees(Arc::new(
            argusnet_world::trees::TreeLayer::new(trees).expect("valid trees"),
        ));
    }

    /// The lowest a flight's legs pass over the keep-out of `tree` (its
    /// crown radius plus the margin, from above), `None` when no leg
    /// crosses it.
    fn lowest_over(
        legs: &[([f64; 3], [f64; 3])],
        tree: &argusnet_world::trees::Tree,
    ) -> Option<f64> {
        let radius = tree.crown_radius_m + CROWN_MARGIN_M;
        let mut lowest: Option<f64> = None;
        for (from, to) in legs {
            for k in 0..=1000 {
                let t = k as f64 / 1000.0;
                let p: [f64; 3] = std::array::from_fn(|i| from[i] + (to[i] - from[i]) * t);
                if (p[0] - tree.xy_m[0]).hypot(p[1] - tree.xy_m[1]) <= radius {
                    lowest = Some(lowest.map_or(p[2], |l: f64| l.min(p[2])));
                }
            }
        }
        lowest
    }

    /// R2: under efficient_v1 a POI redirect whose straight line would pass
    /// through a 45 m tree's crown at 40 m climbs over it, 3 m above its
    /// top all across its keep-out, and comes down to the POI; the journal
    /// says so. A tree whose keep-out holds the POI itself is not climbed
    /// over: an end beside a crown cannot be above it.
    #[test]
    fn efficient_v1_climbs_a_poi_redirect_over_a_crown() {
        let tall = tree("tall", 300.0, 0.0, 45.0, 6.0);
        let beside_poi = tree("beside", 503.0, 0.0, 30.0, 5.0);
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let start = mobile[0].1;
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        treed(&mut stage, vec![tall.clone(), beside_poi.clone()]);
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        assert!(!flight.via_z().is_empty(), "the flight climbs");
        let legs = legs_of(&flight, start);
        let over = lowest_over(&legs, &tall).expect("crosses the tall crown");
        assert!(over >= 45.0 + CROWN_CLEARANCE_M - 1e-9, "{over}");
        // Never lower than the straight flight.
        assert!(flight.via_z().iter().all(|z| *z >= 40.0 - 1e-9));
        let (arrived, _) = flight.sample(flight.t_arrive());
        assert_eq!(arrived, [500.0, 0.0, 40.0]);
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.inputs["route_trees"], "over");
        assert!(decision.inputs.get("route_power_lines").is_none());
    }

    /// A leg that climbs across a crown's keep-out is judged at the lower
    /// end of its stretch over it: entering at 40 m and leaving at 50 m
    /// over a crown that wants 48 m violates it, though it leaves above.
    #[test]
    fn a_climbing_leg_is_judged_at_its_lowest_over_the_crown() {
        let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
        treed(&mut stage, vec![tree("tall", 300.0, 0.0, 45.0, 6.0)]);
        let known = stage.obstacles.as_ref().expect("known obstacles");
        let (start, end) = ([0.0, 0.0], [500.0, 0.0]);
        // The keep-out spans x = 293..307; this leg is at 40.5 m and 51 m
        // there.
        let climbing = [([270.0, 0.0, 30.0], [320.0, 0.0, 55.0])];
        assert!(known.violates_crowns(&climbing, start, end));
        // The same leg descending is judged at its other end, as low.
        let descending = [([320.0, 0.0, 55.0], [270.0, 0.0, 30.0])];
        assert!(known.violates_crowns(&descending, start, end));
        // Over it all the way, by the clearance: clear.
        let over = [([270.0, 0.0, 48.0], [320.0, 0.0, 60.0])];
        assert!(!known.violates_crowns(&over, start, end));
        // Wide of it: clear at any height.
        let wide = [([270.0, 20.0, 10.0], [320.0, 20.0, 10.0])];
        assert!(!known.violates_crowns(&wide, start, end));
    }

    /// R2: the reference never reads the trees; a flight already over the
    /// crowns, or wide of them, is flown as it was.
    #[test]
    fn only_a_flight_too_low_over_a_crown_changes() {
        let tall = tree("tall", 300.0, 0.0, 45.0, 6.0);
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let straight = GotoHover::redirect(mobile[0].1, [500.0, 0.0], 40.0, 0.0, 28.0);
        let mut reference = driver(vec![poi("poi-00", 500.0, 1)]);
        treed(&mut reference, vec![tall.clone()]);
        reference.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(
            poi_flight(&reference.take_override_commands(), "drone-a"),
            straight
        );
        // Wide of the crown (8 m off its keep-out) and over a short one.
        for trees in [
            vec![tree("wide", 300.0, 15.0, 45.0, 6.0)],
            vec![tree("short", 300.0, 0.0, 30.0, 6.0)],
        ] {
            let mut stage = efficient_driver(vec![poi("poi-00", 500.0, 1)]);
            treed(&mut stage, trees);
            stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
            assert_eq!(
                poi_flight(&stage.take_override_commands(), "drone-a"),
                straight
            );
        }
    }

    /// R2: a gap sweep across a crown too low climbs over it.
    #[test]
    fn crown_keep_outs_hold_no_end_and_stretch_over_the_crossing() {
        let mut stage = efficient_driver(Vec::new());
        treed(
            &mut stage,
            vec![
                tree("a", 300.0, 0.0, 45.0, 6.0),
                tree("b", 100.0, 0.0, 20.0, 5.0),
            ],
        );
        let obstacles = stage.obstacles.as_ref().expect("known");
        // From (0, 0) to (600, 0): the keep-out of "a" is 7 m round x = 300.
        let stretches =
            obstacles.crown_stretches([0.0, 0.0], [600.0, 0.0], [0.0, 0.0], [600.0, 0.0]);
        assert_eq!(stretches.len(), 2);
        let a = stretches
            .iter()
            .find(|s| (s.2 - 48.0).abs() < 1e-9)
            .expect("a");
        assert!((a.0 - 293.0 / 600.0).abs() < 1e-9 && (a.1 - 307.0 / 600.0).abs() < 1e-9);
        // An end inside a keep-out drops that crown.
        let from_b =
            obstacles.crown_stretches([100.0, 0.0], [600.0, 0.0], [100.0, 0.0], [600.0, 0.0]);
        assert_eq!(from_b.len(), 1);
        // Violation is judged on the flown height over the stretch.
        let low = [([0.0, 0.0, 40.0], [600.0, 0.0, 40.0])];
        let high = [([0.0, 0.0, 49.0], [600.0, 0.0, 49.0])];
        assert!(obstacles.violates_crowns(&low, [0.0, 0.0], [600.0, 0.0]));
        assert!(!obstacles.violates_crowns(&high, [0.0, 0.0], [600.0, 0.0]));
    }

    /// R2: through a stand of crowns, uneven in height and overlapping, a
    /// POI redirect from a low start to a higher POI climbs over every one
    /// it crosses, 3 m over its top, however the route is split at their
    /// edges.
    #[test]
    fn a_redirect_climbs_over_a_whole_stand() {
        let mut rng = argusnet_world::numpy_rng::NumpyRng::new(5);
        let stand: Vec<_> = (0..40)
            .map(|k| {
                tree(
                    &format!("t{k:02}"),
                    rng.uniform(160.0, 440.0),
                    rng.uniform(-6.0, 6.0),
                    rng.uniform(42.0, 55.0),
                    rng.uniform(3.0, 7.0),
                )
            })
            .collect();
        let (battery, mobile) = fleet(&[("drone-a", 100.0)]);
        let target = InspectionPoi {
            position: [500.0, 0.0, 47.3],
            ..poi("poi-00", 500.0, 1)
        };
        let mut stage = efficient_driver(vec![target]);
        treed(&mut stage, stand.clone());
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        let flight = poi_flight(&stage.take_override_commands(), "drone-a");
        let legs = legs_of(&flight, mobile[0].1);
        let mut crossed = 0;
        for tree in &stand {
            if let Some(over) = lowest_over(&legs, tree) {
                crossed += 1;
                assert!(
                    over >= tree.top_z_m() + CROWN_CLEARANCE_M - 1e-6,
                    "{}: {over}",
                    tree.id
                );
            }
        }
        assert!(crossed > 10, "{crossed}");
        let decision = stage
            .decisions()
            .iter()
            .find(|d| d.kind == "poi_assignment")
            .expect("assigned");
        assert_eq!(decision.inputs["route_trees"], "over");
    }
}
