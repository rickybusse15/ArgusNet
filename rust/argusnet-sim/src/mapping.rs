//! Stage 8's coverage and belief accumulation.
//!
//! §4 names stage 8 *"coverage/belief and localization"*, and it is one stage
//! doing two things. Both halves live here, because the second reads the grid
//! the first just filled.
//!
//! # Localization runs in two phases only, and reads the phase from *before*
//! this tick
//!
//! ```python
//! if _scan_phase in ("scanning", "localizing"):
//!     for ns in mobile_states:
//!         est = _grid_localizer.update(...)
//! ```
//!
//! Stage 8 runs **before** stage 9, so the phase it sees is the one committed
//! at the end of the previous tick — not the one this tick is about to move
//! to. That is §4's causal order made concrete, and it is why
//! [`MappingStage::follow_mission_phase`] reads stage 9 rather than the stage
//! being told a phase: the view shows whatever the machine had settled on, at
//! the moment stage 8 asks.
//!
//! # Heading is zero below 0.1 m/s, not the last heading
//!
//! ```python
//! heading = atan2(vy, vx) if spd > 0.1 else 0.0
//! ```
//!
//! A platform that slows to a hover reports heading **0.0**, pointing east —
//! not the direction it was last travelling. Carrying the previous heading
//! forward would be the kinder behaviour and it is not the reference's.
//!
//! # The reference keeps two identical coverage maps
//!
//! `run_simulation` builds a top-level `CoverageMap` **and** a `WorldMap`,
//! which builds a second `CoverageMap` of its own. Both are then marked from
//! the same drone positions, with the same radius, the same visibility
//! predicate and the same sample cap:
//!
//! ```python
//! _coverage_map.mark_circular(center_xy=(cx, cy), radius_m=_FOOTPRINT_RADIUS_M,
//!                             visibility_predicate=..., los_max_samples=...)
//! # ... and, inside WorldMap.add_scan_observation:
//! self._coverage.mark_circular(center_xy=(cx, cy), radius_m=footprint_radius_m,
//!                              visibility_predicate=..., los_max_samples=...)
//! ```
//!
//! Identical arguments to the same method on two instances, so
//! `MappingState.coverage_fraction` (read from the first) and the mission
//! machine's `scan_frac` (read from the second) are **the same number computed
//! twice** — every tick, for every drone.
//!
//! The asymmetry worth keeping: the second map is only fed under
//! `mission_mode == "scan_map_inspect"`. In any other mode `WorldMap` stays
//! empty while the top-level map fills, so the two agree only in scan mode.
//! [`MappingStage::set_scan_mode`] reproduces that, and the coverage *fraction*
//! is read from one grid either way — which is the point of P4.6's *"one
//! footprint rasterizer whose indices/weights feed both coverage and belief
//! updates"*.
//!
//! # One rasterization, and the second consumer that is not here yet
//!
//! [`rasterize_circle`] is called **once** per platform per tick, and the hits
//! feed [`CoverageGrid`]'s revision-aware aggregates from P4.6b.
//!
//! P4.6d's `BeliefStore` is the second consumer those same hits are meant to
//! feed, and it is deliberately **not** wired here: it lands with its own PR,
//! and stacking this slice on top of it to get one accessor would make a
//! three-deep chain across two crates for no reviewer's benefit. Nothing
//! consumes belief yet — P2.1b is still deferred — whereas the coverage
//! fraction has a consumer today, which is the mission machine of P4.8a.
//! [`MappingStage::coverage_fraction`] is what closes that loop.
//!
//! # The footprint radius is derived, and floored
//!
//! ```python
//! _target_cells = max(5, int(0.05 * nx * ny / _n_mobile))
//! _r_cells = max(1, int(ceil(sqrt(_target_cells / pi))))
//! _FOOTPRINT_RADIUS_M = max(50.0, float(_r_cells) * resolution_m)
//! ```
//!
//! Each drone is scaled to cover about 5% of the map per pass, and then the
//! result is floored at 50 m. [`footprint_radius_m`] reproduces it, floor and
//! all — and the floor is what makes the two rasterizations of P4.6a diverge,
//! since it breaks the integral `radius / resolution` the rest of the
//! configuration maintains.
//!
//! # Zone inspection rides on the same footprint
//!
//! ```python
//! for zone in scenario.mission_zones:
//!     for ns in mobile_states:
//!         inside = float(np.linalg.norm(node_xy - zcenter)) <= zone.radius_m
//!         if inside:
//!             zmap.mark_circular(center_xy=..., radius_m=_FOOTPRINT_RADIUS_M)
//!         zcov = zmap.stats.coverage_fraction
//!         # entered / violation / exited / coverage_updated
//! ```
//!
//! Every mission zone has a coverage map of its own, over the whole map's
//! bounds, marked only while a drone is inside the zone; and which drones are
//! inside is remembered, so entry and exit are edges. Both are state a
//! resumed run needs, so they are checkpointed with the rest of stage 8
//! (P4.9a.2). The fraction an event reports is read **after that drone's
//! mark and before the next drone's**, which is why the zone loop is the
//! outer one.

use std::collections::BTreeSet;

use argusnet_localization::{GridLocalizer, LocalizationConfig, LocalizationEstimate, StepInput};
use argusnet_mapping::{
    rasterize_circle, score_footprints, CoverageGrid, CoverageStats, GridBounds, Sampling,
};

use crate::mission::{stage_nine_phase, MissionPhase};
use crate::particles::particle_offsets;
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// The inspection event types a replay carries, as the reference names them.
pub const INSPECTION_ENTERED: &str = "entered";
pub const INSPECTION_COVERAGE_UPDATED: &str = "coverage_updated";
pub const INSPECTION_EXITED: &str = "exited";
pub const INSPECTION_VIOLATION: &str = "violation";
/// Every one of them, for the replay vocabulary check.
pub const INSPECTION_EVENT_TYPES: [&str; 4] = [
    INSPECTION_ENTERED,
    INSPECTION_COVERAGE_UPDATED,
    INSPECTION_EXITED,
    INSPECTION_VIOLATION,
];

/// `getattr(scenario.options, "coverage_resolution_m", 50.0)`.
pub const DEFAULT_COVERAGE_RESOLUTION_M: f64 = 50.0;
/// The `max(50.0, ...)` floor on the derived footprint.
pub const MIN_FOOTPRINT_RADIUS_M: f64 = 50.0;
/// `0.05` — each drone covers about 5% of the map per pass.
pub const TARGET_MAP_FRACTION_PER_PASS: f64 = 0.05;
/// The `max(5, ...)` floor on the target cell count.
pub const MIN_TARGET_CELLS: i64 = 5;
/// `spd > 0.1` — below this the reference reports a heading of exactly zero.
pub const MIN_SPEED_FOR_HEADING_MPS: f64 = 0.1;

/// The reference's auto-scaled footprint radius.
///
/// Reproduced with its floors: `max(5, ...)` on the cell target, `max(1, ...)`
/// on the cell radius, and `max(50.0, ...)` on the metres. The last is the one
/// that matters downstream — it breaks the integral `radius / resolution` the
/// rest of the configuration maintains, which is exactly when P4.6a's two
/// rasterizations stop agreeing.
pub fn footprint_radius_m(nx: usize, ny: usize, mobile_count: usize, resolution_m: f64) -> f64 {
    // `or 1`: a fleet of zero still divides.
    let mobile = mobile_count.max(1) as f64;
    let cells = TARGET_MAP_FRACTION_PER_PASS * (nx as f64) * (ny as f64) / mobile;
    // `int()` truncates toward zero, then `max(5, ...)`.
    let target_cells = (cells as i64).max(MIN_TARGET_CELLS);
    let r_cells = ((target_cells as f64 / std::f64::consts::PI).sqrt().ceil() as i64).max(1);
    (r_cells as f64 * resolution_m).max(MIN_FOOTPRINT_RADIUS_M)
}

/// `ZONE_TYPE_EXCLUSION`: entering one is a violation, not an entry.
pub const EXCLUSION_ZONE_TYPE: &str = "exclusion";

/// A mission zone as stage 8's inspection tracking reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectionZone {
    pub zone_id: String,
    pub zone_type: String,
    pub center_xy: [f64; 2],
    pub radius_m: f64,
}

/// One zone event, as the reference's `InspectionEvent` records it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InspectionEvent {
    pub zone_id: String,
    pub node_id: String,
    /// `"entered"`, `"violation"`, `"coverage_updated"` or `"exited"`.
    pub event_type: String,
    pub timestamp_s: f64,
    pub zone_coverage_fraction: f64,
}

#[derive(Debug, Clone)]
struct ZoneTrack {
    zone: InspectionZone,
    coverage: CoverageGrid,
    /// `_node_zone_membership`, inverted: the drones inside this zone.
    inside: BTreeSet<String>,
}

/// `TeamLocalization`: the fleet's estimates fused, under cooperative search.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TeamLocalization {
    pub position: [f64; 3],
    pub position_std_m: f64,
    pub confidence: f64,
    /// The drones tight enough to have been fused, in estimate order.
    pub contributing_node_ids: Vec<String>,
}

/// One zone's share of a stage 8 checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ZoneCheckpoint {
    zone_id: String,
    coverage: CoverageGrid,
    inside: BTreeSet<String>,
}

#[derive(Debug)]
struct State {
    bounds: GridBounds,
    footprint_radius_m: f64,
    sampling: Sampling,
    /// `mission_mode == "scan_map_inspect"`. The reference only feeds its
    /// `WorldMap` in that mode.
    scan_mode: bool,
    coverage: CoverageGrid,
    /// `None` until localization is enabled, matching a run that does not
    /// localize at all rather than one that localizes with nothing to say.
    localizer: Option<GridLocalizer>,
    /// Gate localization on stage 9's phase, read at stage 8 time, so it is
    /// the phase committed at the end of the previous tick.
    follows_mission: bool,
    estimates: Vec<LocalizationEstimate>,
    /// `_loc_estimates`: the estimates of the last tick that localized. The
    /// reference keeps reporting them after localization stops; `estimates`
    /// does not, because the mission machine must not read stale ones.
    last_estimates: Vec<LocalizationEstimate>,
    localizer_timed_out: bool,
    /// `cooperative_search`: fuse the fleet's estimates after each update.
    cooperative: bool,
    /// `_team_loc_current`: the last fusion that produced an estimate.
    team: Option<TeamLocalization>,
    zones: Vec<ZoneTrack>,
    /// This tick's zone events. Output, not state.
    inspection_events: Vec<InspectionEvent>,
    /// This tick's cells that went from unobserved to observed, `(i, j)`
    /// ascending. Output, not state.
    newly_covered: Vec<(usize, usize)>,
}

impl MappingStage {
    pub fn bounds(&self) -> GridBounds {
        self.state.bounds
    }

    pub fn footprint_radius_m(&self) -> f64 {
        self.state.footprint_radius_m
    }

    pub fn set_footprint_radius_m(&mut self, radius_m: f64) {
        self.state.footprint_radius_m = radius_m;
    }

    /// Which of P4.6a's two rasterizations to use. The reference's coverage
    /// map uses the metre-anchored lattice.
    pub fn set_sampling(&mut self, sampling: Sampling) {
        self.state.sampling = sampling;
    }

    pub fn set_scan_mode(&mut self, scan_mode: bool) {
        self.state.scan_mode = scan_mode;
    }

    /// `MappingState`'s four fields, from the maintained aggregates.
    pub fn stats(&self) -> CoverageStats {
        self.state.coverage.stats()
    }

    /// What the mission machine reads as `scan_frac`.
    ///
    /// One grid, so this cannot drift from [`Self::stats`] the way the
    /// reference's two maps could.
    pub fn coverage_fraction(&self) -> f64 {
        self.state.coverage.stats().coverage_fraction
    }

    /// Per-cell observation counts, for a consumer that needs more than the
    /// aggregates.
    pub fn coverage_count_at(&self, i: usize, j: usize) -> u32 {
        self.state.coverage.count_at(i, j)
    }

    /// Every cell's count, `i`-major (`counts[i * ny + j]`).
    pub fn coverage_counts(&self) -> Vec<u32> {
        let shared = &self.state;
        let (nx, ny) = shared.coverage.dimensions();
        let mut counts = Vec::with_capacity(nx * ny);
        for i in 0..nx {
            for j in 0..ny {
                counts.push(shared.coverage.count_at(i, j));
            }
        }
        counts
    }

    /// Track inspection of these zones, replacing any tracked before.
    ///
    /// Each zone's coverage map covers the stage's whole grid, as the
    /// reference's does.
    pub fn set_zones(&mut self, zones: Vec<InspectionZone>) -> Result<(), &'static str> {
        let shared = &mut self.state;
        let bounds = shared.bounds;
        let tracks = zones
            .into_iter()
            .map(|zone| {
                Ok(ZoneTrack {
                    zone,
                    coverage: CoverageGrid::new(bounds)?,
                    inside: BTreeSet::new(),
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        shared.zones = tracks;
        Ok(())
    }

    /// This tick's zone events, zone by zone, drone by drone.
    pub fn inspection_events(&self) -> Vec<InspectionEvent> {
        self.state.inspection_events.clone()
    }

    /// The centres of this tick's newly observed cells, in `np.where` order.
    pub fn newly_covered_cells(&self) -> Vec<[f64; 2]> {
        let shared = &self.state;
        let bounds = shared.bounds;
        shared
            .newly_covered
            .iter()
            .map(|&(i, j)| {
                // `GridBounds.ij_to_xy`.
                [
                    bounds.x_min_m + (i as f64 + 0.5) * bounds.resolution_m,
                    bounds.y_min_m + (j as f64 + 0.5) * bounds.resolution_m,
                ]
            })
            .collect()
    }

    /// Fuse the fleet's estimates after every localizing tick, as the
    /// reference's cooperative search does.
    pub fn set_cooperative_search(&mut self, cooperative: bool) {
        self.state.cooperative = cooperative;
    }

    /// The last team fusion, if cooperative search has produced one.
    pub fn team_localization(&self) -> Option<TeamLocalization> {
        self.state.team.clone()
    }

    /// The estimates of the last tick that localized; see `last_estimates`.
    pub fn last_estimates(&self) -> Vec<LocalizationEstimate> {
        self.state.last_estimates.clone()
    }

    /// Turn on the localization half of stage 8.
    pub fn enable_localization(&mut self, config: LocalizationConfig) {
        self.state.localizer = Some(GridLocalizer::new(config));
    }

    /// Follow stage 9's mission phase, so localization runs only in the
    /// phases the reference runs it in. The phase is read through the views
    /// lent to stage 8 ([`stage_nine_phase`]).
    ///
    /// Without this the localizer runs every tick, which is what a run with
    /// no mission machine does; so does a run whose stage 9 has none.
    pub fn follow_mission_phase(&mut self) {
        self.state.follows_mission = true;
    }

    /// This tick's estimates, in platform order. Empty when localization is
    /// off or the phase does not run it.
    pub fn estimates(&self) -> Vec<LocalizationEstimate> {
        self.state.estimates.clone()
    }

    /// What the mission machine reads as `localization_confidences`.
    pub fn confidences(&self) -> Vec<f64> {
        self.state
            .estimates
            .iter()
            .map(|estimate| estimate.confidence)
            .collect()
    }

    /// `GridLocalizer.any_timed_out` as of the last run.
    pub fn localizer_timed_out(&self) -> bool {
        self.state.localizer_timed_out
    }

    /// Drop all accumulated coverage, and any localization state.
    pub fn clear(&mut self) {
        let shared = &mut self.state;
        shared.coverage.reset();
        if let Some(localizer) = shared.localizer.as_mut() {
            localizer.reset(None);
        }
        shared.estimates.clear();
        shared.last_estimates.clear();
        shared.team = None;
        shared.localizer_timed_out = false;
        for track in &mut shared.zones {
            track.coverage.reset();
            track.inside.clear();
        }
        shared.inspection_events.clear();
        shared.newly_covered.clear();
    }
}

/// What stage 8 checkpoints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MappingCheckpoint {
    coverage: CoverageGrid,
    localizer: Option<GridLocalizer>,
    estimates: Vec<LocalizationEstimate>,
    localizer_timed_out: bool,
    #[serde(default)]
    last_estimates: Vec<LocalizationEstimate>,
    #[serde(default)]
    zones: Vec<ZoneCheckpoint>,
    #[serde(default)]
    team: Option<TeamLocalization>,
    /// The self-built map, in a run without a prior one. Absent otherwise,
    /// so a prior run's checkpoint is the one it always was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    belief: Option<crate::belief_world::BeliefGrid>,
    /// The boundaries the lane sat out since it last ran (B2.5), still to be
    /// integrated. Absent when empty, so every checkpoint taken at the legacy
    /// rate is the one it always was.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pending: Vec<PendingBoundary>,
}

/// One boundary stage 8 sat out (its lane at its own rate, B2.5): what it
/// would have integrated then. Integrated, in order, when the lane next runs,
/// so coverage, zone inspection and the self-built map are what they would
/// have been had it run every boundary.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct PendingBoundary {
    tick: u64,
    timestamp_s: f64,
    /// The mobile platforms' positions then, in platform order.
    positions: Vec<(String, [f64; 2])>,
    /// The range scans stage 6 handed over then (a run without a prior map).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    scans: Vec<crate::range_scan::RangeScan>,
}

/// Stage 8's self-built map (`world_knowledge: none`): the grid, the handle
/// the next tick's planners read, and this tick's scans.
#[derive(Debug)]
struct BeliefMapping {
    grid: crate::belief_world::BeliefGrid,
    config: crate::belief_world::ExplorationConfig,
    handle: crate::belief_world::BeliefHandle,
    inbox: Vec<crate::range_scan::RangeScan>,
    /// The scans fused this tick, kept for the replay.
    fused: Vec<crate::range_scan::RangeScan>,
}

impl BeliefMapping {
    fn publish(&mut self) {
        self.handle.publish(self.grid.snapshot(&self.config));
    }
}

/// Fewest platforms whose particle work is forked across the session's
/// workers (B3). Each platform costs 10-40 us; below this many, waking the
/// workers costs more than they save. Measured on 4 cores: 4 and 8 drones
/// were slower forked, 16 and 32 were 1.4x and 1.3x faster at 4 workers
/// (docs/performance/b3-mapping-fork-join-2026-10-04.md).
pub const MIN_PLATFORMS_TO_FORK: usize = 12;

/// One platform's particle offsets and their footprint scores.
type PreparedParticles = (Vec<[f64; 2]>, Vec<f64>);

/// One platform's localizer inputs this tick.
struct LocalizeInput<'a> {
    node_id: &'a str,
    heading_rad: f64,
    nominal_position: [f64; 3],
}

/// [`Stage::UpdateMappingAndLocalization`]'s coverage and belief half.
///
/// [`Stage::UpdateMappingAndLocalization`]: crate::stage::Stage::UpdateMappingAndLocalization
#[derive(Debug)]
pub struct MappingStage {
    state: State,
    /// The session's workers for the per-platform particle draws and
    /// scoring (B3); serial unless the session gives it more.
    compute: std::sync::Arc<crate::compute::ComputePool>,
    belief: Option<BeliefMapping>,
    /// The boundaries sat out since the lane last ran (B2.5).
    pending: Vec<PendingBoundary>,
}

impl MappingStage {
    /// Build a map from range scans into `grid` (empty), and publish it to
    /// the returned handle after every tick. The handle starts with the empty
    /// grid's snapshot, so the first tick plans against nothing known.
    pub fn set_belief(
        &mut self,
        mut grid: crate::belief_world::BeliefGrid,
        config: crate::belief_world::ExplorationConfig,
    ) -> crate::belief_world::BeliefHandle {
        let handle = crate::belief_world::BeliefHandle::new(grid.snapshot(&config));
        self.belief = Some(BeliefMapping {
            grid,
            config,
            handle: handle.clone(),
            inbox: Vec::new(),
            fused: Vec::new(),
        });
        handle
    }

    /// The scans stage 6 made this tick (`Handoff::RangeScansToMapping`).
    pub fn submit_range_scans(&mut self, scans: Vec<crate::range_scan::RangeScan>) {
        if let Some(belief) = self.belief.as_mut() {
            belief.inbox = scans;
        }
    }

    /// The self-built map, in a run without a prior one.
    /// The platforms' own map as published this tick, in a no-prior run: what
    /// the scan mission measures its progress on. `None` in a prior run.
    pub fn belief_snapshot(&self) -> Option<std::sync::Arc<crate::belief_world::BeliefSnapshot>> {
        self.belief.as_ref().map(|belief| belief.handle.current())
    }

    pub fn belief_grid(&self) -> Option<&crate::belief_world::BeliefGrid> {
        self.belief.as_ref().map(|belief| &belief.grid)
    }

    /// The occupancy threshold the belief uses.
    pub fn exploration_config(&self) -> Option<&crate::belief_world::ExplorationConfig> {
        self.belief.as_ref().map(|belief| &belief.config)
    }

    /// The belief cells changed since the last call, as replay rows.
    pub fn take_belief_updates(&mut self) -> Vec<crate::belief_world::BeliefCellUpdate> {
        self.belief.as_mut().map_or_else(Vec::new, |belief| {
            let threshold = belief.config.occupied_probability;
            belief.grid.take_changed_cells(threshold)
        })
    }

    /// The canopy cells changed since the last call, as replay rows (R2).
    pub fn take_canopy_updates(&mut self) -> Vec<crate::belief_world::CanopyCellUpdate> {
        self.belief
            .as_mut()
            .map_or_else(Vec::new, |belief| belief.grid.take_changed_canopy())
    }

    /// The scans fused this tick.
    pub fn fused_scans(&self) -> &[crate::range_scan::RangeScan] {
        self.belief.as_ref().map_or(&[], |belief| &belief.fused)
    }

    /// Fuse this tick's scans, after those of the boundaries the lane sat
    /// out (each at its own tick), and publish the result.
    fn update_belief(&mut self, pending: &mut [PendingBoundary], tick: u64) {
        let Some(belief) = self.belief.as_mut() else {
            return;
        };
        let mut fused = Vec::new();
        for boundary in pending {
            let scans = std::mem::take(&mut boundary.scans);
            for scan in &scans {
                belief.grid.integrate(scan, boundary.tick);
            }
            fused.extend(scans);
        }
        let scans = std::mem::take(&mut belief.inbox);
        for scan in &scans {
            belief.grid.integrate(scan, tick);
        }
        fused.extend(scans);
        belief.fused = fused;
        belief.publish();
    }

    /// A stage over `bounds`, with the footprint radius still to be set.
    pub fn new(bounds: GridBounds) -> Result<Self, &'static str> {
        let coverage = CoverageGrid::new(bounds)?;
        Ok(Self {
            state: State {
                bounds,
                footprint_radius_m: MIN_FOOTPRINT_RADIUS_M,
                // The reference's coverage map uses the metre-anchored
                // lattice; see P4.6a for why the two differ.
                sampling: Sampling::WorldAnchored,
                scan_mode: true,
                coverage,
                localizer: None,
                follows_mission: false,
                estimates: Vec::new(),
                last_estimates: Vec::new(),
                localizer_timed_out: false,
                cooperative: false,
                team: None,
                zones: Vec::new(),
                inspection_events: Vec::new(),
                newly_covered: Vec::new(),
            },
            compute: std::sync::Arc::new(crate::compute::ComputePool::serial()),
            belief: None,
            pending: Vec::new(),
        })
    }

    /// The session's workers for the per-platform particle work (B3). The
    /// output is the same at every worker count.
    pub fn set_compute(&mut self, compute: std::sync::Arc<crate::compute::ComputePool>) {
        self.compute = compute;
    }

    /// A stage whose footprint is derived the way the reference derives it.
    pub fn with_derived_footprint(
        bounds: GridBounds,
        mobile_count: usize,
    ) -> Result<Self, &'static str> {
        let mut stage = Self::new(bounds)?;
        let (nx, ny) = bounds.dimensions()?;
        stage.set_footprint_radius_m(footprint_radius_m(
            nx,
            ny,
            mobile_count,
            bounds.resolution_m,
        ));
        Ok(stage)
    }
}

/// The mobile platforms' positions, in platform order: what stage 8 marks.
fn mobile_positions(context: &TickContext<'_>) -> Vec<(String, [f64; 2])> {
    context
        .world
        .platforms()
        .iter()
        .filter(|platform| platform.is_mobile)
        .map(|platform| {
            (
                platform.node_id.clone(),
                [platform.position.x, platform.position.y],
            )
        })
        .collect()
}

/// One boundary's coverage and zone inspection: every footprint marked, the
/// newly covered cells and the zone events recorded. Returns the cell hits.
fn mark_boundary(
    shared: &mut State,
    positions: &[(String, [f64; 2])],
    timestamp_s: f64,
) -> Result<usize, StageError> {
    let State {
        bounds,
        footprint_radius_m,
        sampling,
        ..
    } = *shared;
    let mut marked = 0usize;
    for (node_id, centre) in positions {
        // ONE rasterization, feeding both consumers -- P4.6's ask.
        let hits = rasterize_circle(&bounds, *centre, footprint_radius_m, sampling)
            .map_err(|e| StageError::Invariant(format!("footprint for {node_id}: {e}")))?;
        if hits.is_empty() {
            continue;
        }
        marked += hits.len();

        // Hits arrive with multiplicity and every one is applied, because
        // a cell the lattice folds onto twice is incremented twice by the
        // reference's `np.add.at`.
        for (i, j) in &hits {
            if shared.coverage.count_at(*i, *j) == 0 {
                shared.newly_covered.push((*i, *j));
            }
            shared.coverage.mark_cell(*i, *j);
        }
    }
    // `np.where` over the (nx, ny) grid: ascending i, then j. Each cell is
    // listed once already: it is recorded only while its count is zero,
    // and its first mark ends that, whichever footprint (or repeated
    // lattice hit) reaches it next.
    shared.newly_covered.sort_unstable();

    // --- zone inspection ------------------------------------------------
    let State {
        zones,
        inspection_events,
        ..
    } = &mut *shared;
    for track in zones.iter_mut() {
        let [zx, zy] = track.zone.center_xy;
        for (node_id, centre) in positions {
            // `np.linalg.norm` of a 2-vector: no `hypot`.
            let (dx, dy) = (centre[0] - zx, centre[1] - zy);
            let inside = (dx * dx + dy * dy).sqrt() <= track.zone.radius_m;
            let was_inside = track.inside.contains(node_id);
            if inside {
                let hits = rasterize_circle(&bounds, *centre, footprint_radius_m, sampling)
                    .map_err(|e| {
                        StageError::Invariant(format!(
                            "zone {} footprint for {node_id}: {e}",
                            track.zone.zone_id
                        ))
                    })?;
                for (i, j) in hits {
                    track.coverage.mark_cell(i, j);
                }
            }
            let zone_coverage_fraction = track.coverage.stats().coverage_fraction;
            let event_type = if inside && !was_inside {
                track.inside.insert(node_id.clone());
                if track.zone.zone_type == EXCLUSION_ZONE_TYPE {
                    INSPECTION_VIOLATION
                } else {
                    INSPECTION_ENTERED
                }
            } else if !inside && was_inside {
                track.inside.remove(node_id);
                INSPECTION_EXITED
            } else if inside {
                INSPECTION_COVERAGE_UPDATED
            } else {
                continue;
            };
            inspection_events.push(InspectionEvent {
                zone_id: track.zone.zone_id.clone(),
                node_id: node_id.clone(),
                event_type: event_type.to_string(),
                timestamp_s,
                zone_coverage_fraction,
            });
        }
    }
    Ok(marked)
}

impl StageHandler for MappingStage {
    fn reset(&mut self) {
        self.clear();
        if let Some(belief) = self.belief.as_mut() {
            belief.grid.clear();
            belief.inbox.clear();
            belief.fused.clear();
            belief.publish();
        }
        self.pending.clear();
    }

    /// Coverage counts (and their running aggregates), the localizer, its
    /// last estimates and the timeout latch.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        let state = MappingCheckpoint {
            coverage: shared.coverage.clone(),
            localizer: shared.localizer.clone(),
            estimates: shared.estimates.clone(),
            localizer_timed_out: shared.localizer_timed_out,
            last_estimates: shared.last_estimates.clone(),
            zones: shared
                .zones
                .iter()
                .map(|track| ZoneCheckpoint {
                    zone_id: track.zone.zone_id.clone(),
                    coverage: track.coverage.clone(),
                    inside: track.inside.clone(),
                })
                .collect(),
            team: shared.team.clone(),
            belief: self.belief.as_ref().map(|belief| belief.grid.clone()),
            pending: self.pending.clone(),
        };
        Some(serde_json::to_value(state).expect("mapping state serialises"))
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: MappingCheckpoint = crate::simulation::decode_stage_state(state)?;
        let shared = &self.state;
        let malformed = |detail: &str| crate::simulation::StageStateError::Malformed {
            detail: detail.to_string(),
        };
        if decoded.coverage.bounds() != shared.coverage.bounds() {
            return Err(malformed(
                "coverage grid covers a different area or resolution",
            ));
        }
        if decoded.localizer.is_some() != shared.localizer.is_some() {
            return Err(malformed(
                "localization is enabled in one run and not the other",
            ));
        }
        let ids = |ids: &mut dyn Iterator<Item = &String>| ids.cloned().collect::<Vec<_>>();
        if ids(&mut decoded.zones.iter().map(|zone| &zone.zone_id))
            != ids(&mut shared.zones.iter().map(|track| &track.zone.zone_id))
        {
            return Err(malformed("the inspected zones differ"));
        }
        if decoded
            .zones
            .iter()
            .any(|zone| zone.coverage.bounds() != shared.coverage.bounds())
        {
            return Err(malformed("a zone's coverage grid differs"));
        }
        match (&self.belief, &decoded.belief) {
            (None, None) => {}
            (Some(belief), Some(saved)) if belief.grid.same_layout(saved) => {}
            (Some(_), Some(_)) => return Err(malformed("the belief grid's layout differs")),
            (Some(_), None) | (None, Some(_)) => {
                return Err(malformed(
                    "one run builds its own map and the other has a prior one",
                ))
            }
        }
        Ok(())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: MappingCheckpoint = crate::simulation::decode_stage_state(state)?;
        let shared = &mut self.state;
        shared.coverage = decoded.coverage;
        shared.localizer = decoded.localizer;
        shared.estimates = decoded.estimates;
        shared.localizer_timed_out = decoded.localizer_timed_out;
        shared.last_estimates = decoded.last_estimates;
        shared.team = decoded.team;
        for (track, zone) in shared.zones.iter_mut().zip(decoded.zones) {
            track.coverage = zone.coverage;
            track.inside = zone.inside;
        }
        shared.inspection_events.clear();
        shared.newly_covered.clear();
        if let (Some(belief), Some(saved)) = (self.belief.as_mut(), decoded.belief) {
            belief.grid = saved;
            belief.inbox.clear();
            belief.fused.clear();
            belief.publish();
        }
        self.pending = decoded.pending;
        Ok(())
    }

    /// The kept boundaries must be exactly the `lag` before `tick`, in order:
    /// a queue with one missing or repeated would lose or replay evidence.
    fn check_lag(
        &self,
        state: &serde_json::Value,
        lag: u64,
        tick: u64,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: MappingCheckpoint = crate::simulation::decode_stage_state(state)?;
        let kept: Vec<u64> = decoded
            .pending
            .iter()
            .map(|boundary| boundary.tick)
            .collect();
        let expected: Vec<u64> = (tick.saturating_sub(lag)..tick).collect();
        if kept != expected {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: format!(
                    "the kept boundaries {kept:?} are not the {lag} sat out before tick {tick}"
                ),
            });
        }
        Ok(())
    }

    /// The lane sits this boundary out (B2.5): keep what it would have
    /// integrated -- the platforms' positions and the scans stage 6 handed
    /// over -- for its next run, and clear this tick's outputs (coverage
    /// deltas, inspection events, fused scans) so a boundary it did not run
    /// reports none. The localization estimates stay: they are the latest the
    /// lane has, and the mission machine reads them every tick.
    fn not_due(&mut self, context: &mut TickContext<'_>) -> Result<(), StageError> {
        let scans = self
            .belief
            .as_mut()
            .map(|belief| {
                belief.fused.clear();
                std::mem::take(&mut belief.inbox)
            })
            .unwrap_or_default();
        self.pending.push(PendingBoundary {
            tick: context.tick,
            timestamp_s: context.timestamp_s,
            positions: mobile_positions(context),
            scans,
        });
        self.state.inspection_events.clear();
        self.state.newly_covered.clear();
        Ok(())
    }

    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        // The boundaries the lane sat out first, each as it was then (B2.5);
        // at the legacy rate there are none.
        let mut pending = std::mem::take(&mut self.pending);
        self.update_belief(&mut pending, context.tick);
        let compute = &*self.compute;
        let shared = &mut self.state;
        shared.inspection_events.clear();
        shared.newly_covered.clear();
        let mut marked = 0usize;
        for boundary in &pending {
            marked += mark_boundary(shared, &boundary.positions, boundary.timestamp_s)?;
        }

        let positions = mobile_positions(context);
        if positions.is_empty() {
            return Ok(StageOutcome::NoWork);
        }
        marked += mark_boundary(shared, &positions, context.timestamp_s)?;
        let State {
            bounds,
            footprint_radius_m,
            ..
        } = *shared;
        let stats = shared.coverage.stats();

        // --- localization -------------------------------------------------
        // Reads the phase committed at the END of the previous tick, because
        // stage 8 runs before stage 9. See the module docs.
        let phase = shared
            .follows_mission
            .then(|| stage_nine_phase(context.stages))
            .flatten()
            // No mission machine means no phase gate, which is what a run
            // without one does.
            .unwrap_or(MissionPhase::Scanning);
        let localizing_phase = matches!(phase, MissionPhase::Scanning | MissionPhase::Localizing);

        shared.estimates.clear();
        if let Some(localizer) = shared.localizer.as_mut().filter(|_| localizing_phase) {
            let coverage_fraction = stats.coverage_fraction;
            let (grid_nx, grid_ny) = shared.coverage.dimensions();
            // `score_footprints` takes the reference's `int32` layout, and
            // saturates rather than wrapping: a cell observed 2^31 times is
            // implausible, and wrapping it negative would make the scorer
            // reject the grid outright.
            let mut counts: Vec<i32> = Vec::with_capacity(grid_nx * grid_ny);
            for i in 0..grid_nx {
                for j in 0..grid_ny {
                    counts.push(i32::try_from(shared.coverage.count_at(i, j)).unwrap_or(i32::MAX));
                }
            }
            let origin = [bounds.x_min_m, bounds.y_min_m];
            let seed = context.seed;

            // Each platform's inputs, in platform order.
            let inputs: Vec<LocalizeInput<'_>> = positions
                .iter()
                .map(|(node_id, _)| {
                    let platform = context
                        .world
                        .platform(node_id)
                        .expect("the id came from this world");
                    // `atan2(vy, vx)` above 0.1 m/s, otherwise exactly zero.
                    let speed_xy = platform.velocity.x.hypot(platform.velocity.y);
                    let heading_rad = if speed_xy > MIN_SPEED_FOR_HEADING_MPS {
                        platform.velocity.y.atan2(platform.velocity.x)
                    } else {
                        0.0
                    };
                    LocalizeInput {
                        node_id,
                        heading_rad,
                        nominal_position: [
                            platform.position.x,
                            platform.position.y,
                            platform.position.z,
                        ],
                    }
                })
                .collect();

            // Each platform's particle draws and footprint scores (B3). They
            // read only the platform's own `(platform, step)` stream, the
            // localizer's configuration and the coverage counts, so they may
            // run on the session's workers in any order; the results come back
            // in platform order.
            let prepared: Vec<Result<PreparedParticles, StageError>> = {
                let localizer: &GridLocalizer = localizer;
                let counts = &counts;
                compute.map_ordered_from(MIN_PLATFORMS_TO_FORK, &inputs, |input| {
                    // The step the draw is keyed on is the one this update
                    // will become, so it is the stored count plus one.
                    let step = localizer.step_count(input.node_id) + 1;
                    let scale = localizer.particle_scale(step, coverage_fraction);
                    let config = localizer.config();
                    let offsets =
                        particle_offsets(seed, input.node_id, step, config.particle_count, scale);

                    let candidates: Vec<[f64; 2]> = offsets
                        .iter()
                        .map(|offset| {
                            [
                                input.nominal_position[0] + offset[0],
                                input.nominal_position[1] + offset[1],
                            ]
                        })
                        .collect();
                    let scores = score_footprints(
                        counts,
                        [grid_nx, grid_ny],
                        origin,
                        bounds.resolution_m,
                        footprint_radius_m,
                        &candidates,
                    )
                    .map_err(|e| {
                        StageError::Invariant(format!("scoring {}: {e}", input.node_id))
                    })?;
                    Ok((offsets, scores))
                })
            };

            // The updates are the localizer's state, so they stay serial, in
            // platform order: a failure stops at the same platform, with the
            // same platforms before it updated, as the single loop it replaced.
            for (input, prepared) in inputs.iter().zip(prepared) {
                let (offsets, scores) = prepared?;
                let estimate = localizer
                    .update(
                        StepInput {
                            platform_id: input.node_id,
                            nominal_position: input.nominal_position,
                            heading_rad: input.heading_rad,
                            timestamp_s: context.timestamp_s,
                            coverage_fraction,
                        },
                        &offsets,
                        &scores,
                    )
                    .map_err(|e| {
                        StageError::Invariant(format!("localizing {}: {e}", input.node_id))
                    })?;
                shared.estimates.push(estimate);
            }

            shared.last_estimates = shared.estimates.clone();

            // `if _coop_search_active and len(_loc_estimates) > 1`: fuse, and
            // keep the previous team estimate when the fusion has nothing.
            if shared.cooperative && shared.estimates.len() > 1 {
                let ids: Vec<&str> = shared
                    .estimates
                    .iter()
                    .map(|estimate| estimate.platform_id.as_str())
                    .collect();
                if let Some(team) = localizer.fuse_estimates(&ids) {
                    let half_radius_m = localizer.config().search_radius_m * 0.5;
                    let contributing_node_ids = shared
                        .estimates
                        .iter()
                        .filter(|estimate| estimate.position_std_m < half_radius_m)
                        .map(|estimate| estimate.platform_id.clone())
                        .collect();
                    shared.team = Some(TeamLocalization {
                        position: team.position_estimate,
                        position_std_m: team.position_std_m,
                        confidence: team.confidence,
                        contributing_node_ids,
                    });
                }
            }

            // Latched by the mission machine, but read here because this is
            // where the localizer lives.
            if shared
                .localizer
                .as_ref()
                .is_some_and(GridLocalizer::any_timed_out)
            {
                shared.localizer_timed_out = true;
            }
        }
        let localized = shared.estimates.len();

        if marked == 0 {
            // Every platform's footprint fell outside the grid.
            return Ok(StageOutcome::NoWork);
        }
        context.note(format!(
            "mapping marked {marked} cell hit(s); coverage {:.4} over {} cells; \
             {localized} estimate(s)",
            stats.coverage_fraction, stats.total_cells
        ));
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_types::{NodeState, Vector3};

    fn bounds() -> GridBounds {
        GridBounds {
            x_min_m: 0.0,
            x_max_m: 1_000.0,
            y_min_m: 0.0,
            y_max_m: 1_000.0,
            resolution_m: DEFAULT_COVERAGE_RESOLUTION_M,
        }
    }

    /// The seed the rig runs with. Named because the particle stream is keyed
    /// on it, and a test that reconstructs the stream has to agree.
    const SEED: u32 = 11;

    fn simulation() -> Simulation {
        Simulation::bare(SimulationConfig {
            seed: SEED,
            dt_s: 1.0,
            steps: Some(4_000),
            run_id: "mapping".to_string(),
        })
        .expect("config is valid")
    }

    fn node(id: &str, mobile: bool, x: f64, y: f64) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, y, 80.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: mobile,
            timestamp_s: 0.0,
            health: 1.0,
            ..Default::default()
        }
    }

    fn rig(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        let stage = MappingStage::new(bounds()).expect("bounds are valid");
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(stage))
            .expect("stage 8 is not a transaction boundary");
        sim
    }

    /// The installed stage, reached through the session that owns it.
    fn map(sim: &mut Simulation) -> &mut MappingStage {
        sim.stage_mut(Stage::UpdateMappingAndLocalization)
            .expect("stage 8 is installed")
    }

    /// Stage 9's machine, reached through the session that owns it.
    fn machine(sim: &Simulation) -> &crate::mission::MissionStage {
        crate::mission::MissionStage::of(sim).expect("stage 9 is a mission machine")
    }

    fn machine_mut(sim: &mut Simulation) -> &mut crate::mission::MissionStage {
        crate::mission::MissionStage::of_mut(sim).expect("stage 9 is a mission machine")
    }

    fn mapped(sim: &Simulation) -> &MappingStage {
        sim.stage(Stage::UpdateMappingAndLocalization)
            .expect("stage 8 is installed")
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::UpdateMappingAndLocalization)
            .expect("stage 8 ran")
            .outcome
            .clone()
    }

    // -----------------------------------------------------------------
    // The derived footprint
    // -----------------------------------------------------------------

    /// Each drone is scaled to cover ~5% of the map per pass, then floored at
    /// 50 m. The floor is the part that matters downstream.
    #[test]
    fn the_footprint_radius_reproduces_the_references_derivation() {
        // 20x20 cells at 50 m, one drone: 0.05 * 400 = 20 target cells,
        // ceil(sqrt(20/pi)) = ceil(2.52) = 3 cells, 3 * 50 = 150 m.
        assert_eq!(footprint_radius_m(20, 20, 1, 50.0), 150.0);
        // Four drones share it: 0.05 * 400 / 4 = 5 cells, ceil(sqrt(5/pi)) =
        // ceil(1.26) = 2 cells, 2 * 50 = 100 m.
        assert_eq!(footprint_radius_m(20, 20, 4, 50.0), 100.0);
    }

    #[test]
    fn the_footprint_radius_is_floored_at_fifty_metres() {
        // A tiny map would derive 1 cell * 5 m = 5 m; the floor lifts it.
        assert_eq!(footprint_radius_m(4, 4, 1, 5.0), MIN_FOOTPRINT_RADIUS_M);
    }

    /// `max(5, ...)` on the target cells and `max(1, ...)` on the cell radius
    /// both bind on a small map, and an empty fleet still divides.
    #[test]
    fn the_derivation_floors_hold_for_degenerate_inputs() {
        // 0.05 * 4 = 0.2 -> int() -> 0 -> max(5) -> 5 cells -> 2 cell radius.
        assert_eq!(footprint_radius_m(2, 2, 1, 200.0), 400.0);
        // `or 1`: zero mobile platforms must not divide by zero.
        assert!(footprint_radius_m(20, 20, 0, 50.0).is_finite());
        assert_eq!(
            footprint_radius_m(20, 20, 0, 50.0),
            footprint_radius_m(20, 20, 1, 50.0),
            "a fleet of zero is treated as one"
        );
    }

    #[test]
    fn the_derived_footprint_reaches_the_stage() {
        let stage = MappingStage::with_derived_footprint(bounds(), 4).unwrap();
        let (nx, ny) = bounds().dimensions().unwrap();
        assert_eq!(
            stage.footprint_radius_m(),
            footprint_radius_m(nx, ny, 4, bounds().resolution_m)
        );
    }

    // -----------------------------------------------------------------
    // Marking
    // -----------------------------------------------------------------

    #[test]
    fn a_drone_marks_a_disc_and_coverage_rises() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        assert_eq!(mapped(&sim).stats().covered_cells, 0);

        sim.step().unwrap();

        let stats = mapped(&sim).stats();
        assert!(stats.covered_cells > 0);
        assert_eq!(stats.total_cells, 20 * 20);
        assert!(stats.coverage_fraction > 0.0 && stats.coverage_fraction < 1.0);
        assert_eq!(outcome(&sim), StageOutcome::Ran);
        // The cell under the drone is marked.
        assert!(mapped(&sim).coverage_count_at(10, 10) > 0);
    }

    /// Revisiting the same ground raises revisits without raising coverage:
    /// `covered_cells` counts cells, not hits.
    #[test]
    fn revisiting_raises_revisits_not_coverage() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        sim.step().unwrap();
        let first = mapped(&sim).stats();

        for _ in 0..4 {
            sim.step().unwrap();
        }
        let later = mapped(&sim).stats();

        assert_eq!(later.covered_cells, first.covered_cells);
        assert_eq!(later.coverage_fraction, first.coverage_fraction);
        assert!(later.mean_revisits > first.mean_revisits);
    }

    #[test]
    fn a_fixed_node_never_marks_anything() {
        let mut sim = rig(vec![node("tower", false, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        sim.step().unwrap();
        assert_eq!(mapped(&sim).stats().covered_cells, 0);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    #[test]
    fn a_world_with_no_mobile_platforms_reports_no_work() {
        let mut sim = rig(Vec::new());
        sim.step().unwrap();
        assert_eq!(mapped(&sim).stats().covered_cells, 0);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// Two drones cover more between them than either does alone.
    #[test]
    fn two_drones_cover_more_than_one() {
        let one = {
            let mut sim = rig(vec![node("drone-1", true, 250.0, 500.0)]);
            map(&mut sim).set_footprint_radius_m(150.0);
            sim.step().unwrap();
            mapped(&sim).stats().covered_cells
        };
        let two = {
            let mut sim = rig(vec![
                node("drone-1", true, 250.0, 500.0),
                node("drone-2", true, 750.0, 500.0),
            ]);
            map(&mut sim).set_footprint_radius_m(150.0);
            sim.step().unwrap();
            mapped(&sim).stats().covered_cells
        };
        assert!(two > one, "{two} should exceed {one}");
    }

    /// A footprint entirely outside the grid marks nothing, and the stage says
    /// it had nothing to do rather than claiming it worked.
    #[test]
    fn a_footprint_off_the_grid_marks_nothing() {
        let mut sim = rig(vec![node("drone-1", true, 50_000.0, 50_000.0)]);
        map(&mut sim).set_footprint_radius_m(10.0);
        sim.step().unwrap();
        assert_eq!(mapped(&sim).stats().covered_cells, 0);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    // -----------------------------------------------------------------
    // The aggregates
    // -----------------------------------------------------------------

    /// The maintained aggregates must equal a full reduction over the grid.
    /// That equality is the whole safety argument for P4.6b's incremental
    /// totals, so it is asserted after real marking rather than at
    /// construction.
    #[test]
    fn the_maintained_aggregates_equal_a_full_reduction() {
        let mut sim = rig(vec![
            node("drone-1", true, 200.0, 200.0),
            node("drone-2", true, 800.0, 300.0),
        ]);
        map(&mut sim).set_footprint_radius_m(120.0);
        for k in 0..6 {
            // Move them around so cells are hit different numbers of times.
            let shift = f64::from(k) * 60.0;
            sim.world_mut().platform_mut("drone-1").unwrap().position.x = 200.0 + shift;
            sim.world_mut().platform_mut("drone-2").unwrap().position.y = 300.0 + shift;
            sim.step().unwrap();
        }
        // Reach the grid through the stage's own accessor, then compare.
        let stats = mapped(&sim).stats();
        assert!(stats.covered_cells > 0);
        assert!(stats.mean_revisits >= 1.0);

        let mut covered = 0u64;
        let mut observations = 0u64;
        for i in 0..20 {
            for j in 0..20 {
                let count = u64::from(mapped(&sim).coverage_count_at(i, j));
                if count > 0 {
                    covered += 1;
                }
                observations += count;
            }
        }
        assert_eq!(stats.covered_cells, covered);
        assert_eq!(stats.total_cells, 400);
        assert!((stats.coverage_fraction - covered as f64 / 400.0).abs() < 1e-12);
        assert!((stats.mean_revisits - observations as f64 / covered as f64).abs() < 1e-12);
    }

    /// `coverage_fraction` is read from the same grid the aggregates come
    /// from, so it cannot drift from them the way the reference's two maps
    /// could.
    #[test]
    fn the_coverage_fraction_matches_the_stats() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(200.0);
        sim.step().unwrap();
        assert_eq!(
            mapped(&sim).coverage_fraction(),
            mapped(&sim).stats().coverage_fraction
        );
    }

    #[test]
    fn clearing_drops_all_coverage() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        sim.step().unwrap();
        assert!(mapped(&sim).stats().covered_cells > 0);

        map(&mut sim).clear();
        let stats = mapped(&sim).stats();
        assert_eq!(stats.covered_cells, 0);
        assert_eq!(stats.coverage_fraction, 0.0);
        assert_eq!(stats.mean_revisits, 0.0, "not a NaN with nothing covered");
    }

    /// The two samplings of P4.6a disagree once the 50 m floor breaks the
    /// integral `radius / resolution`, which is exactly the configuration the
    /// reference's derivation produces.
    #[test]
    fn the_two_samplings_disagree_at_a_non_integral_radius() {
        let covered_with = |sampling: Sampling| {
            let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
            map(&mut sim).set_sampling(sampling);
            // 125 / 50 is not a whole number of cells.
            map(&mut sim).set_footprint_radius_m(125.0);
            sim.step().unwrap();
            mapped(&sim).stats().covered_cells
        };
        assert_ne!(
            covered_with(Sampling::WorldAnchored),
            covered_with(Sampling::CellDisc),
            "the choice of rasterization is a visible decision, not a detail"
        );
    }

    /// P4.6a's multiplicity contract, reaching the grid.
    ///
    /// The metre-anchored lattice steps by the cell size, so distinct samples
    /// normally land in distinct cells. The exception is narrow and worth
    /// stating exactly, because the obvious guess is wrong: a sample *past*
    /// the bounds is **skipped**, not clamped. Only a sample landing exactly
    /// on the inclusive far edge folds — `x == x_max` truncates to index
    /// `nx`, which clamps back onto `nx - 1`, the same cell `x_max - res`
    /// already hit.
    ///
    /// That fold is a genuine double-increment under the reference's
    /// `np.add.at`, so hits must be applied with multiplicity rather than
    /// deduplicated. A single tick is the whole test: revisits across ticks
    /// would raise the count either way.
    #[test]
    fn a_sample_on_the_far_edge_folds_onto_the_last_cell_and_counts_twice() {
        // x_max is 1000 and the resolution 50, so a centre at 950 puts lattice
        // samples on both 950 (index 19) and exactly 1000 (index 20 -> 19).
        let mut sim = rig(vec![node("drone-1", true, 950.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);

        sim.step().unwrap();

        assert_eq!(
            mapped(&sim).coverage_count_at(19, 10),
            2,
            "the edge sample and its neighbour both land on the last cell"
        );
        // One column in, nothing folds, so the same footprint hits once.
        assert_eq!(
            mapped(&sim).coverage_count_at(18, 10),
            1,
            "an unfolded cell is hit once by one footprint"
        );
    }

    // -----------------------------------------------------------------
    // Localization
    // -----------------------------------------------------------------

    fn localizing_rig(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = rig(platforms);
        map(&mut sim).set_footprint_radius_m(150.0);
        map(&mut sim).enable_localization(LocalizationConfig::default());
        sim
    }

    /// With localization off — the default — stage 8 is coverage only, and
    /// says so by producing no estimates rather than empty ones.
    #[test]
    fn localization_is_off_until_it_is_enabled() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        for _ in 0..5 {
            sim.step().unwrap();
        }
        assert!(mapped(&sim).estimates().is_empty());
        assert!(mapped(&sim).confidences().is_empty());
        assert!(!mapped(&sim).localizer_timed_out());
    }

    #[test]
    fn an_enabled_localizer_produces_one_estimate_per_mobile_platform() {
        let mut sim = localizing_rig(vec![
            node("drone-a", true, 300.0, 300.0),
            node("tower", false, 500.0, 500.0),
            node("drone-b", true, 700.0, 700.0),
        ]);
        sim.step().unwrap();

        let estimates = mapped(&sim).estimates();
        assert_eq!(estimates.len(), 2, "one per mobile platform, not per node");
        let ids: Vec<&str> = estimates.iter().map(|e| e.platform_id.as_str()).collect();
        assert_eq!(ids, vec!["drone-a", "drone-b"]);
        assert_eq!(mapped(&sim).confidences().len(), 2);
    }

    /// Below the coverage floor the estimator reports no confidence at all,
    /// and it still records an estimate rather than skipping the platform.
    #[test]
    fn below_the_coverage_floor_confidence_is_zero() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        sim.step().unwrap();

        let coverage = mapped(&sim).coverage_fraction();
        assert!(
            coverage < LocalizationConfig::default().min_coverage_to_localize,
            "one footprint is {coverage}, which must be under the floor"
        );
        let estimates = mapped(&sim).estimates();
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0].confidence, 0.0);
    }

    /// Confidence rises as the map fills: that is the whole premise of
    /// matching a footprint against an accumulated coverage map.
    #[test]
    fn confidence_rises_as_the_map_fills() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);

        let mut x = 100.0;
        let mut y = 100.0;
        let mut best = 0.0f64;
        for _ in 0..200 {
            {
                let platform = sim.world_mut().platform_mut("drone-1").unwrap();
                platform.position.x = x;
                platform.position.y = y;
                platform.velocity.x = 20.0;
            }
            sim.step().unwrap();
            if let Some(estimate) = mapped(&sim).estimates().first() {
                best = best.max(estimate.confidence);
            }
            x += 100.0;
            if x > 950.0 {
                x = 100.0;
                y += 100.0;
            }
            if y > 950.0 {
                y = 100.0;
            }
        }
        assert!(
            mapped(&sim).coverage_fraction()
                > LocalizationConfig::default().min_coverage_to_localize
        );
        assert!(best > 0.0, "confidence never rose above zero");
    }

    // -----------------------------------------------------------------
    // The phase gate, and the heading rule
    // -----------------------------------------------------------------

    /// The reference localizes only while `scanning` or `localizing`. Stage 8
    /// runs before stage 9, so the phase it reads is the one committed at the
    /// END of the previous tick.
    #[test]
    fn localization_stops_outside_the_scanning_and_localizing_phases() {
        use crate::mission::{MissionConfig, MissionInputs, MissionStage};

        let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        let mission = MissionStage::new(MissionConfig::default());
        map(&mut sim).follow_mission_phase();
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
            .unwrap();

        // Tick 0: still Scanning, so it localizes.
        let mut inputs = MissionInputs::at(0.0);
        inputs.mobile_count = 1;
        inputs.localization_confidences = vec![0.99];
        machine_mut(&mut sim).set_inputs(inputs);
        sim.step().unwrap();
        assert_eq!(mapped(&sim).estimates().len(), 1, "Scanning localizes");

        // Push it out of Scanning.
        let mut done = MissionInputs::at(1.0);
        done.mobile_count = 1;
        done.localization_confidences = vec![0.99];
        done.coverage_fraction = 0.99;
        done.gap_cell_fraction = Some(0.0);
        machine_mut(&mut sim).set_inputs(done);
        sim.step().unwrap();
        assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);
        assert_eq!(
            mapped(&sim).estimates().len(),
            1,
            "that tick still saw the PREVIOUS phase, which was Scanning"
        );
        let last_localized = mapped(&sim).estimates();

        // The next tick reads Inspecting and stops.
        let mut after = MissionInputs::at(2.0);
        after.mobile_count = 1;
        after.localization_confidences = vec![0.99];
        machine_mut(&mut sim).set_inputs(after);
        sim.step().unwrap();
        assert!(
            mapped(&sim).estimates().is_empty(),
            "Inspecting must not localize"
        );
        // The replay keeps reporting the last estimates, as `_loc_estimates`
        // does; the mission machine's own read above stays empty.
        assert_eq!(mapped(&sim).last_estimates(), last_localized);
        assert!(!last_localized.is_empty());
    }

    /// `atan2(vy, vx)` above 0.1 m/s, and exactly **zero** below it — not the
    /// last heading. A platform that slows to a hover points east.
    #[test]
    fn heading_is_zero_below_the_speed_floor_not_the_last_heading() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);

        // Moving north: atan2(20, 0) = pi/2.
        {
            let platform = sim.world_mut().platform_mut("drone-1").unwrap();
            platform.velocity.x = 0.0;
            platform.velocity.y = 20.0;
        }
        sim.step().unwrap();
        let moving = mapped(&sim).estimates()[0].heading_rad;
        assert!(
            (moving - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "moving north should read pi/2, got {moving}"
        );

        // Now hover. The heading must snap to 0, not stay at pi/2.
        {
            let platform = sim.world_mut().platform_mut("drone-1").unwrap();
            platform.velocity.x = 0.0;
            platform.velocity.y = 0.05;
        }
        sim.step().unwrap();
        assert_eq!(
            mapped(&sim).estimates()[0].heading_rad,
            0.0,
            "below 0.1 m/s the reference reports exactly zero"
        );
    }

    /// The floor is strict: exactly 0.1 m/s is still "too slow".
    #[test]
    fn the_heading_speed_floor_is_strict() {
        let heading_at = |speed: f64| {
            let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
            {
                let platform = sim.world_mut().platform_mut("drone-1").unwrap();
                platform.velocity.x = 0.0;
                platform.velocity.y = speed;
            }
            sim.step().unwrap();
            mapped(&sim).estimates()[0].heading_rad
        };
        assert_eq!(heading_at(MIN_SPEED_FOR_HEADING_MPS), 0.0, "`>`, not `>=`");
        assert!(heading_at(MIN_SPEED_FOR_HEADING_MPS + 1e-9) > 0.0);
    }

    // -----------------------------------------------------------------
    // Determinism and lifecycle
    // -----------------------------------------------------------------

    /// The particle draw is keyed on `(seed, platform, step)`, so the same run
    /// twice gives the same estimates. ADR-006 makes this the contract in
    /// place of bit-parity with numpy.
    #[test]
    fn a_run_is_reproducible_for_a_fixed_seed() {
        let run = || {
            let mut sim = localizing_rig(vec![node("drone-1", true, 200.0, 200.0)]);
            for k in 0..12 {
                {
                    let platform = sim.world_mut().platform_mut("drone-1").unwrap();
                    platform.position.x = 200.0 + f64::from(k) * 60.0;
                    platform.velocity.x = 15.0;
                }
                sim.step().unwrap();
            }
            mapped(&sim).estimates()
        };
        let first = run();
        let second = run();
        assert_eq!(first.len(), 1);
        assert_eq!(first, second, "a fixed seed and config must reproduce");
    }

    #[test]
    fn clearing_resets_the_localizer_too() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        sim.step().unwrap();
        assert!(!mapped(&sim).estimates().is_empty());

        map(&mut sim).clear();
        assert!(mapped(&sim).estimates().is_empty());
        assert!(!mapped(&sim).localizer_timed_out());
        assert_eq!(mapped(&sim).stats().covered_cells, 0);
    }

    /// Fly a lawnmower until coverage clears the localizer's floor, so the
    /// estimator is actually estimating rather than short-circuiting on
    /// `MapUnavailable`. Returns the drone to a fixed pose afterwards.
    fn cover_the_map(sim: &mut Simulation, node_id: &str) {
        let mut x = 100.0;
        let mut y = 100.0;
        for _ in 0..80 {
            {
                let platform = sim.world_mut().platform_mut(node_id).unwrap();
                platform.position.x = x;
                platform.position.y = y;
                platform.velocity.x = 20.0;
            }
            sim.step().unwrap();
            if mapped(sim).coverage_fraction()
                > LocalizationConfig::default().min_coverage_to_localize
            {
                break;
            }
            x += 100.0;
            if x > 950.0 {
                x = 100.0;
                y += 100.0;
            }
            if y > 950.0 {
                y = 100.0;
            }
        }
        assert!(
            mapped(sim).coverage_fraction()
                > LocalizationConfig::default().min_coverage_to_localize,
            "the map never filled enough to localize"
        );
        let platform = sim.world_mut().platform_mut(node_id).unwrap();
        platform.position.x = 500.0;
        platform.position.y = 500.0;
        platform.velocity.x = 20.0;
        platform.velocity.y = 0.0;
    }

    /// The particle stream is keyed on the step, so holding a platform
    /// perfectly still still draws a **different** cloud each tick. Keying on
    /// the previous step would repeat the draw and freeze the estimate.
    #[test]
    fn consecutive_ticks_draw_different_particles() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");

        sim.step().unwrap();
        let first = mapped(&sim).estimates()[0].clone();
        sim.step().unwrap();
        let second = mapped(&sim).estimates()[0].clone();

        assert_ne!(
            first.position_estimate, second.position_estimate,
            "a fresh particle cloud each step must move the estimate"
        );
    }

    /// Each platform has its own stream, keyed on its id. Two platforms in the
    /// same place over the same map must still draw different clouds -- a
    /// shared stream would make them agree exactly, which would look like
    /// agreement and be an artefact.
    #[test]
    fn two_platforms_in_the_same_place_draw_different_particles() {
        let mut sim = localizing_rig(vec![
            node("drone-a", true, 100.0, 100.0),
            node("drone-b", true, 100.0, 100.0),
        ]);
        cover_the_map(&mut sim, "drone-a");
        {
            let platform = sim.world_mut().platform_mut("drone-b").unwrap();
            platform.position.x = 500.0;
            platform.position.y = 500.0;
            platform.velocity.x = 20.0;
        }
        sim.step().unwrap();

        let estimates = mapped(&sim).estimates();
        assert_eq!(estimates.len(), 2);
        assert_eq!(
            estimates[0].position_estimate[0],
            estimates[0].position_estimate[0]
        );
        assert_ne!(
            estimates[0].position_estimate, estimates[1].position_estimate,
            "identical poses must not produce identical estimates: the streams \
             are keyed per platform"
        );
    }

    /// The cloud is scaled by `particle_scale`, which starts near the search
    /// radius and anneals down. A fixed small scale would collapse the spread
    /// and report a confident estimate it has not earned.
    #[test]
    fn the_particle_cloud_is_scaled_by_the_search_radius() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");
        sim.step().unwrap();

        let estimate = &mapped(&sim).estimates()[0];
        assert!(
            estimate.position_std_m > 5.0,
            "an early cloud spans tens of metres, not {}",
            estimate.position_std_m
        );
    }

    /// Particles are offsets **around the platform**, not absolute positions.
    /// Scoring the raw offsets would search near the origin instead of near
    /// the drone.
    #[test]
    fn particles_are_offsets_around_the_platform() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");
        sim.step().unwrap();

        let estimate = &mapped(&sim).estimates()[0];
        let dx = estimate.position_estimate[0] - 500.0;
        let dy = estimate.position_estimate[1] - 500.0;
        let offset = dx.hypot(dy);
        assert!(
            offset < LocalizationConfig::default().search_radius_m * 3.0,
            "the estimate is {offset:.1} m from the platform, so the cloud was \
             not centred on it"
        );
    }

    /// The timeout latch is raised from here and never cleared, which is what
    /// the mission machine reads.
    #[test]
    fn the_localizer_timeout_latches() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");

        // A fresh localizer with a short budget, over the map already built.
        map(&mut sim).enable_localization(LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        });
        assert!(!mapped(&sim).localizer_timed_out());

        for _ in 0..5 {
            sim.step().unwrap();
        }
        assert!(
            mapped(&sim).localizer_timed_out(),
            "running past the step budget must latch the timeout"
        );

        // The latch is checkpoint state: nothing re-derives it after the
        // localizing phase, so a resumed run must be handed it.
        let checkpoint = sim.checkpoint().expect("not faulted");
        let mut fresh = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        map(&mut fresh).enable_localization(LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        });
        assert!(!mapped(&fresh).localizer_timed_out());
        fresh.restore(&checkpoint).expect("restore accepted");
        assert!(
            mapped(&fresh).localizer_timed_out(),
            "a restore dropped the localization-timeout latch"
        );
    }

    /// The budget counts **localization steps, not ticks**.
    ///
    /// `update` returns early with `MapUnavailable` while coverage is below
    /// the floor, and the timeout is assigned well after that return — so a
    /// platform that never clears the floor never times out, however long it
    /// flies. That is not obvious from the configuration, whose field is named
    /// `localization_timeout_steps`, and it is what a run that never builds a
    /// map actually does.
    #[test]
    fn a_platform_below_the_coverage_floor_never_times_out() {
        let mut sim = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        map(&mut sim).set_footprint_radius_m(150.0);
        map(&mut sim).enable_localization(LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        });

        // Stationary, so coverage never clears the floor.
        for _ in 0..40 {
            sim.step().unwrap();
        }
        assert!(
            mapped(&sim).coverage_fraction()
                < LocalizationConfig::default().min_coverage_to_localize
        );
        assert!(
            !mapped(&sim).localizer_timed_out(),
            "forty ticks with no map is zero localization steps"
        );
    }

    /// `clear` resets the localizer's own step counts, not just the published
    /// estimates -- otherwise a fresh epoch would inherit the previous one's
    /// annealing and start with a cloud far tighter than it has earned.
    #[test]
    fn clearing_resets_the_localizers_annealing() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");
        for _ in 0..20 {
            sim.step().unwrap();
        }
        let annealed = mapped(&sim).estimates()[0].position_std_m;

        map(&mut sim).clear();
        cover_the_map(&mut sim, "drone-1");
        sim.step().unwrap();
        let fresh = mapped(&sim).estimates()[0].position_std_m;

        assert!(
            fresh > annealed,
            "a cleared localizer must start wide again: fresh {fresh} vs \
             annealed {annealed}"
        );
    }

    /// The gate admits `Localizing` as well as `Scanning`. A gate that only
    /// admitted `Scanning` would stop estimating exactly when the mission is
    /// waiting on the estimate to converge -- a deadlock that looks like slow
    /// convergence.
    #[test]
    fn the_localizing_phase_still_localizes() {
        use crate::mission::{MissionConfig, MissionInputs, MissionStage};

        let mut sim = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        let mission = MissionStage::new(MissionConfig::default());
        map(&mut sim).follow_mission_phase();
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
            .unwrap();

        // Coverage clears the gate but confidence has not, so the mission goes
        // Scanning -> Localizing rather than straight to Inspecting.
        let mut inputs = MissionInputs::at(0.0);
        inputs.mobile_count = 1;
        inputs.localization_confidences = vec![0.10];
        inputs.coverage_fraction = 0.99;
        inputs.gap_cell_fraction = Some(0.0);
        machine_mut(&mut sim).set_inputs(inputs);
        sim.step().unwrap();
        assert_eq!(machine(&sim).phase(), MissionPhase::Localizing);

        let mut next = MissionInputs::at(1.0);
        next.mobile_count = 1;
        next.localization_confidences = vec![0.10];
        machine_mut(&mut sim).set_inputs(next);
        sim.step().unwrap();

        assert_eq!(machine(&sim).phase(), MissionPhase::Localizing);
        assert_eq!(
            mapped(&sim).estimates().len(),
            1,
            "Localizing is exactly when the estimate matters most"
        );
    }

    /// The exact inputs the stage hands the estimator, pinned by
    /// reconstructing them independently and requiring the same answer.
    ///
    /// Two things this catches that nothing else can, because both leave the
    /// estimate plausibly near the platform either way:
    ///
    /// * **Which step the stream is keyed on.** `update` increments the count
    ///   internally, so during this call the step is `step_count() + 1`. The
    ///   same number scales the cloud, so keying on `step_count()` anneals a
    ///   step behind and draws a different cloud — deterministically, and
    ///   wrongly.
    /// * **That candidates are `nominal + offset`.** The estimate is a
    ///   weighted mean of the offsets around the nominal position, so scoring
    ///   the raw offsets instead still returns something near the platform;
    ///   only the weights change.
    ///
    /// It also pins a third thing, which writing it turned up: the estimator
    /// sees the coverage map **including this tick's own footprint**, because
    /// stage 8 marks and then localizes. Reconstructing from the previous
    /// tick's map gives a different scale and a different answer.
    #[test]
    fn the_estimator_receives_the_step_scale_and_candidates_the_contract_specifies() {
        let mut sim = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, "drone-1");

        // The pose the tick will run at. Nothing else is installed, so the
        // platform does not move during it.
        let platform = sim.world().platform("drone-1").unwrap();
        let nominal = [
            platform.position.x,
            platform.position.y,
            platform.position.z,
        ];
        let speed_xy = platform.velocity.x.hypot(platform.velocity.y);
        let heading_rad = if speed_xy > MIN_SPEED_FOR_HEADING_MPS {
            platform.velocity.y.atan2(platform.velocity.x)
        } else {
            0.0
        };
        let timestamp_s = sim.tick() as f64;

        // A fresh localizer on both sides, so both are at step 1 over the same
        // coverage.
        let config = LocalizationConfig::default();
        let mut reference = GridLocalizer::new(config);
        map(&mut sim).enable_localization(config);
        sim.step().unwrap();
        let from_stage = mapped(&sim).estimates()[0].clone();

        // Read the map AFTER the tick: stage 8 marks coverage and then
        // localizes, so the estimator sees this tick's own footprint. Reading
        // the previous tick's map would be a different -- and staler --
        // estimate.
        let coverage_fraction = mapped(&sim).coverage_fraction();
        let (grid_nx, grid_ny) = (20usize, 20usize);
        let mut counts: Vec<i32> = Vec::with_capacity(grid_nx * grid_ny);
        for i in 0..grid_nx {
            for j in 0..grid_ny {
                counts.push(i32::try_from(mapped(&sim).coverage_count_at(i, j)).unwrap());
            }
        }

        // The step this update IS, not the count before it.
        let step = 1u32;
        let scale = reference.particle_scale(step, coverage_fraction);
        // The run's own seed; the stream is keyed on (seed, platform, step).
        let offsets = particle_offsets(SEED, "drone-1", step, config.particle_count, scale);
        let candidates: Vec<[f64; 2]> = offsets
            .iter()
            .map(|offset| [nominal[0] + offset[0], nominal[1] + offset[1]])
            .collect();
        let scores = score_footprints(
            &counts,
            [grid_nx, grid_ny],
            [0.0, 0.0],
            50.0,
            150.0,
            &candidates,
        )
        .unwrap();
        let expected = reference
            .update(
                StepInput {
                    platform_id: "drone-1",
                    nominal_position: nominal,
                    heading_rad,
                    timestamp_s,
                    coverage_fraction,
                },
                &offsets,
                &scores,
            )
            .unwrap();

        assert_eq!(
            from_stage, expected,
            "the stage must key the stream on step_count() + 1, scale the \
             cloud by that same step, and score `nominal + offset`"
        );
    }

    // -----------------------------------------------------------------
    // Zone inspection and the team readout (P4.9a.2)
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // B2.5: the mapping lane at its own rate
    // -----------------------------------------------------------------

    /// A rig with a zone and a self-built map, for the lane-rate tests.
    fn buffered_rig() -> Simulation {
        let mut sim = rig(vec![node("drone-1", true, 100.0, 500.0)]);
        map(&mut sim)
            .set_zones(vec![zone("z", "survey", 300.0, 500.0, 120.0)])
            .expect("zones");
        let grid = crate::belief_world::BeliefGrid::new(
            argusnet_types::Bounds2d::new(0.0, 1_000.0, 0.0, 1_000.0),
            10.0,
            0.0,
        )
        .expect("grid");
        map(&mut sim).set_belief(grid, crate::belief_world::ExplorationConfig::default());
        sim
    }

    /// Boundary `k`'s inputs: the drone moves 40 m east a tick, and its scan
    /// sees one structure ahead of it.
    fn feed(sim: &mut Simulation, k: u64) {
        let x = 100.0 + 40.0 * k as f64;
        place(sim, "drone-1", x, 500.0);
        let scan = crate::range_scan::RangeScan {
            node_id: "drone-1".into(),
            timestamp_s: k as f64,
            origin_m: [x, 500.0, 80.0],
            returns: vec![crate::range_scan::RangeReturn {
                point_m: [x + 60.0, 500.0 + 10.0 * k as f64, 30.0],
                class: crate::range_scan::ReturnClass::Structure,
            }],
            misses: 0,
        };
        map(sim).submit_range_scans(vec![scan]);
    }

    fn mapping_state(sim: &Simulation) -> serde_json::Value {
        mapped(sim).checkpoint_state().expect("stage 8 checkpoints")
    }

    fn executor(rate: Option<&str>, first: u64) -> crate::scheduler::PhasedLaneExecutor {
        let executor =
            crate::scheduler::PhasedLaneExecutor::new::<crate::simulation::TickError>(1.0, first)
                .unwrap();
        match rate {
            Some(rate) => executor
                .with_mapping_rate(crate::scheduler::parse_rate(rate).unwrap())
                .unwrap(),
            None => executor,
        }
    }

    /// A mapping lane at a third of the boundary rate keeps every boundary
    /// it sits out and integrates it when it runs: at each boundary it runs,
    /// coverage, zone inspection and the self-built map are exactly the
    /// every-boundary run's. In between, the rest of the tick reads the map
    /// as it last left it.
    #[test]
    fn a_mapping_lane_at_its_own_rate_catches_up_exactly() {
        let mut every = buffered_rig();
        let mut rated = buffered_rig();
        rated.declare_lane_rate(Stage::UpdateMappingAndLocalization, "1/3");
        let (mut every_lanes, mut rated_lanes) = (executor(None, 0), executor(Some("1/3"), 0));
        for k in 0..12 {
            feed(&mut every, k);
            feed(&mut rated, k);
            every_lanes.step(&mut every).expect("tick");
            rated_lanes.step(&mut rated).expect("tick");
            if k % 3 == 0 {
                assert_eq!(mapping_state(&rated), mapping_state(&every), "boundary {k}");
            } else {
                assert_ne!(mapping_state(&rated), mapping_state(&every), "boundary {k}");
                assert_eq!(
                    mapping_state(&rated)["pending"].as_array().unwrap().len() as u64,
                    k % 3
                );
            }
        }
        // The legacy rate checkpoints no pending boundaries at all.
        assert!(mapping_state(&every).get("pending").is_none());
    }

    /// A checkpoint taken while boundaries are pending (ticks 4 and 5, before
    /// tick 6 runs) carries them, so the resumed run integrates them as the
    /// uninterrupted one does.
    #[test]
    fn pending_boundaries_survive_a_checkpoint() {
        let mut straight = buffered_rig();
        straight.declare_lane_rate(Stage::UpdateMappingAndLocalization, "1/3");
        let mut lanes = executor(Some("1/3"), 0);
        let mut checkpoint = None;
        for k in 0..9 {
            if k == 6 {
                checkpoint = Some(straight.checkpoint().expect("checkpoint"));
            }
            feed(&mut straight, k);
            lanes.step(&mut straight).expect("tick");
        }
        let checkpoint = checkpoint.unwrap();
        assert!(checkpoint.stage_states.values().any(|state| state
            .get("pending")
            .is_some_and(|p| p.as_array().unwrap().len() == 2)));

        let mut resumed = buffered_rig();
        resumed.declare_lane_rate(Stage::UpdateMappingAndLocalization, "1/3");
        resumed.restore(&checkpoint).expect("restores");
        let mut lanes = executor(Some("1/3"), 6);
        for k in 6..9 {
            feed(&mut resumed, k);
            lanes.step(&mut resumed).expect("tick");
        }
        assert_eq!(mapping_state(&resumed), mapping_state(&straight));
    }

    /// A checkpoint whose kept boundaries disagree with the recorded lag --
    /// one dropped, one repeated, or the mapping state missing altogether --
    /// is refused, and the run is untouched.
    #[test]
    fn kept_boundaries_must_match_the_recorded_lag() {
        let mut sim = buffered_rig();
        sim.declare_lane_rate(Stage::UpdateMappingAndLocalization, "1/3");
        let mut lanes = executor(Some("1/3"), 0);
        for k in 0..6 {
            feed(&mut sim, k);
            lanes.step(&mut sim).expect("tick");
        }
        let checkpoint = sim.checkpoint().expect("checkpoint");
        let name = Stage::UpdateMappingAndLocalization.to_string();
        assert_eq!(checkpoint.stage_lag.get(&name), Some(&2));

        let damaged = |edit: &dyn Fn(&mut Vec<serde_json::Value>)| {
            let mut copy = checkpoint.clone();
            let pending = copy
                .stage_states
                .get_mut(&name)
                .and_then(|state| state.get_mut("pending"))
                .and_then(serde_json::Value::as_array_mut)
                .expect("two kept boundaries");
            edit(pending);
            copy
        };
        let mut fresh = buffered_rig();
        fresh.declare_lane_rate(Stage::UpdateMappingAndLocalization, "1/3");
        let dropped = damaged(&|pending| {
            pending.remove(0);
        });
        assert!(
            fresh.restore(&dropped).is_err(),
            "a dropped boundary was accepted"
        );
        let repeated = damaged(&|pending| {
            let first = pending[0].clone();
            pending.insert(0, first);
        });
        assert!(
            fresh.restore(&repeated).is_err(),
            "a repeated boundary was accepted"
        );
        let mut missing = checkpoint.clone();
        missing.stage_states.remove(&name);
        assert!(
            fresh.restore(&missing).is_err(),
            "a missing mapping state was accepted"
        );
        assert_eq!(fresh.tick(), 0, "a refused restore changed the run");
        fresh
            .restore(&checkpoint)
            .expect("the undamaged checkpoint restores");
    }

    fn zone(id: &str, zone_type: &str, x: f64, y: f64, radius_m: f64) -> InspectionZone {
        InspectionZone {
            zone_id: id.to_string(),
            zone_type: zone_type.to_string(),
            center_xy: [x, y],
            radius_m,
        }
    }

    fn place(sim: &mut Simulation, node_id: &str, x: f64, y: f64) {
        let platform = sim.world_mut().platform_mut(node_id).unwrap();
        platform.position.x = x;
        platform.position.y = y;
    }

    fn events(sim: &Simulation) -> Vec<(String, String, String)> {
        mapped(sim)
            .inspection_events()
            .into_iter()
            .map(|e| (e.zone_id, e.node_id, e.event_type))
            .collect()
    }

    fn event(zone: &str, node: &str, kind: &str) -> (String, String, String) {
        (zone.to_string(), node.to_string(), kind.to_string())
    }

    /// Entry and exit are edges; staying inside reports coverage; entering an
    /// exclusion zone is a violation; the radius is inclusive.
    #[test]
    fn zone_events_are_edges_and_an_exclusion_entry_is_a_violation() {
        let mut sim = rig(vec![
            node("drone-1", true, 900.0, 900.0),
            node("drone-2", true, 900.0, 100.0),
        ]);
        map(&mut sim)
            .set_zones(vec![
                zone("patrol", "patrol", 500.0, 500.0, 100.0),
                zone("keep-out", EXCLUSION_ZONE_TYPE, 200.0, 200.0, 50.0),
            ])
            .unwrap();

        sim.step().unwrap();
        assert!(events(&sim).is_empty(), "nobody is inside anything");

        // Exactly on the radius: inside.
        place(&mut sim, "drone-1", 600.0, 500.0);
        sim.step().unwrap();
        assert_eq!(events(&sim), vec![event("patrol", "drone-1", "entered")]);

        place(&mut sim, "drone-2", 200.0, 200.0);
        sim.step().unwrap();
        assert_eq!(
            events(&sim),
            vec![
                event("patrol", "drone-1", "coverage_updated"),
                event("keep-out", "drone-2", "violation"),
            ]
        );

        place(&mut sim, "drone-1", 900.0, 900.0);
        place(&mut sim, "drone-2", 900.0, 100.0);
        sim.step().unwrap();
        assert_eq!(
            events(&sim),
            vec![
                event("patrol", "drone-1", "exited"),
                event("keep-out", "drone-2", "exited"),
            ]
        );
    }

    /// An event reports its zone's coverage after THAT drone's footprint and
    /// before the next drone's -- the zone loop is the outer one.
    #[test]
    fn a_zone_event_reads_coverage_between_drones() {
        let mut sim = rig(vec![
            node("drone-1", true, 450.0, 500.0),
            node("drone-2", true, 550.0, 500.0),
        ]);
        map(&mut sim)
            .set_zones(vec![zone("patrol", "patrol", 500.0, 500.0, 100.0)])
            .unwrap();
        sim.step().unwrap();
        let fractions: Vec<f64> = mapped(&sim)
            .inspection_events()
            .iter()
            .map(|e| e.zone_coverage_fraction)
            .collect();
        let mut alone = CoverageGrid::new(bounds()).unwrap();
        alone
            .mark_circular(
                [450.0, 500.0],
                MIN_FOOTPRINT_RADIUS_M,
                Sampling::WorldAnchored,
            )
            .unwrap();
        assert_eq!(fractions.len(), 2);
        assert_eq!(fractions[0], alone.stats().coverage_fraction);
        assert!(
            fractions[1] > fractions[0],
            "the second drone's mark is counted"
        );
        // The zone map is separate from the stage's own grid only in what
        // it marks: here both drones were inside, so they agree.
        assert_eq!(fractions[1], mapped(&sim).coverage_fraction());
    }

    /// Cooperative search reports the fleet's fused estimate from the drones
    /// tight enough to fuse, keeps it once localization stops, and reports
    /// nothing without cooperative search.
    #[test]
    fn cooperative_search_reports_a_team_estimate_and_keeps_it() {
        for cooperative in [false, true] {
            let mut sim = localizing_rig(vec![
                node("drone-1", true, 100.0, 100.0),
                node("drone-2", true, 900.0, 900.0),
            ]);
            map(&mut sim).set_cooperative_search(cooperative);
            let half_radius_m = LocalizationConfig::default().search_radius_m * 0.5;
            let mut x = 100.0;
            let mut y = 100.0;
            let mut mixed_ticks = 0;
            for _ in 0..200 {
                place(&mut sim, "drone-1", x, y);
                place(&mut sim, "drone-2", 1_000.0 - x, 1_000.0 - y);
                sim.step().unwrap();
                // Every tick that fuses, fuses exactly the tight drones.
                let estimates = mapped(&sim).estimates();
                let tight: Vec<String> = estimates
                    .iter()
                    .filter(|e| e.position_std_m < half_radius_m)
                    .map(|e| e.platform_id.clone())
                    .collect();
                if cooperative && !tight.is_empty() {
                    let team = mapped(&sim)
                        .team_localization()
                        .expect("a tight drone fuses");
                    assert_eq!(team.contributing_node_ids, tight);
                    if tight.len() < estimates.len() {
                        mixed_ticks += 1;
                    }
                }
                x += 100.0;
                if x > 950.0 {
                    x = 100.0;
                    y += 100.0;
                }
                if y > 950.0 {
                    y = 100.0;
                }
            }
            let team = mapped(&sim).team_localization();
            if !cooperative {
                assert_eq!(team, None);
                continue;
            }
            assert!(mixed_ticks > 0, "no tick fused some drones and not others");
            let team = team.expect("two converged drones fuse");
            let tight: Vec<String> = mapped(&sim)
                .estimates()
                .iter()
                .filter(|e| e.position_std_m < half_radius_m)
                .map(|e| e.platform_id.clone())
                .collect();
            assert!(!tight.is_empty());
            assert_eq!(team.contributing_node_ids, tight);
            // Inverse-variance fusion is at least as tight as its best input.
            let best = mapped(&sim)
                .estimates()
                .iter()
                .map(|e| e.position_std_m)
                .fold(f64::INFINITY, f64::min);
            assert!(team.position_std_m <= best);

            // Survives a checkpoint, as the reference's `_team_loc_current`
            // survives the rest of the run.
            let checkpoint = sim.checkpoint().unwrap();
            map(&mut sim).clear();
            assert_eq!(mapped(&sim).team_localization(), None);
            sim.restore(&checkpoint).unwrap();
            assert_eq!(mapped(&sim).team_localization(), Some(team));
        }
    }

    /// `np.where` order -- ascending `i`, then `j` -- across drones, each
    /// cell once even where two footprints first reach it together; the
    /// later drone here covers the lower `i`.
    #[test]
    fn newly_covered_cells_are_in_np_where_order_and_unique() {
        let mut sim = rig(vec![
            node("drone-1", true, 600.0, 500.0),
            node("drone-2", true, 450.0, 500.0),
        ]);
        map(&mut sim).set_footprint_radius_m(150.0);
        sim.step().unwrap();
        let cells = mapped(&sim).newly_covered_cells();
        let mut expected = cells.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expected.dedup();
        assert_eq!(cells, expected);
        assert_eq!(cells.len() as u64, mapped(&sim).stats().covered_cells);
        sim.step().unwrap();
        assert!(
            mapped(&sim).newly_covered_cells().is_empty(),
            "nothing new the second time"
        );
    }
}
