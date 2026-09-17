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
//! [`MappingLink::watch_mission`] takes a handle rather than the stage being
//! told a phase: the handle reads whatever the machine had settled on, at the
//! moment stage 8 asks.
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
//! [`MappingLink::set_scan_mode`] reproduces that, and the coverage *fraction*
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
//! [`MappingLink::coverage_fraction`] is what closes that loop.
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

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_localization::{GridLocalizer, LocalizationConfig, LocalizationEstimate, StepInput};
use argusnet_mapping::{
    rasterize_circle, score_footprints, CoverageGrid, CoverageStats, GridBounds, Sampling,
};

use crate::mission::{MissionLink, MissionPhase};
use crate::particles::particle_offsets;
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

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

/// A handle onto the mapping stage.
#[derive(Debug, Clone)]
pub struct MappingLink {
    inner: Arc<Mutex<Shared>>,
}

#[derive(Debug)]
struct Shared {
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
    /// Read at stage 8 time, so it carries the phase committed at the end of
    /// the previous tick.
    mission: Option<MissionLink>,
    estimates: Vec<LocalizationEstimate>,
    localizer_timed_out: bool,
}

impl MappingLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn bounds(&self) -> GridBounds {
        self.lock().bounds
    }

    pub fn footprint_radius_m(&self) -> f64 {
        self.lock().footprint_radius_m
    }

    pub fn set_footprint_radius_m(&self, radius_m: f64) {
        self.lock().footprint_radius_m = radius_m;
    }

    /// Which of P4.6a's two rasterizations to use. The reference's coverage
    /// map uses the metre-anchored lattice.
    pub fn set_sampling(&self, sampling: Sampling) {
        self.lock().sampling = sampling;
    }

    pub fn set_scan_mode(&self, scan_mode: bool) {
        self.lock().scan_mode = scan_mode;
    }

    /// `MappingState`'s four fields, from the maintained aggregates.
    pub fn stats(&self) -> CoverageStats {
        self.lock().coverage.stats()
    }

    /// What the mission machine reads as `scan_frac`.
    ///
    /// One grid, so this cannot drift from [`Self::stats`] the way the
    /// reference's two maps could.
    pub fn coverage_fraction(&self) -> f64 {
        self.lock().coverage.stats().coverage_fraction
    }

    /// Per-cell observation counts, for a consumer that needs more than the
    /// aggregates.
    pub fn coverage_count_at(&self, i: usize, j: usize) -> u32 {
        self.lock().coverage.count_at(i, j)
    }

    /// Turn on the localization half of stage 8.
    pub fn enable_localization(&self, config: LocalizationConfig) {
        self.lock().localizer = Some(GridLocalizer::new(config));
    }

    /// Watch a mission machine, so localization runs only in the phases the
    /// reference runs it in.
    ///
    /// Without a handle the localizer runs every tick, which is what a run
    /// with no mission machine does.
    pub fn watch_mission(&self, mission: MissionLink) {
        self.lock().mission = Some(mission);
    }

    /// This tick's estimates, in platform order. Empty when localization is
    /// off or the phase does not run it.
    pub fn estimates(&self) -> Vec<LocalizationEstimate> {
        self.lock().estimates.clone()
    }

    /// What the mission machine reads as `localization_confidences`.
    pub fn confidences(&self) -> Vec<f64> {
        self.lock()
            .estimates
            .iter()
            .map(|estimate| estimate.confidence)
            .collect()
    }

    /// `GridLocalizer.any_timed_out` as of the last run.
    pub fn localizer_timed_out(&self) -> bool {
        self.lock().localizer_timed_out
    }

    /// Drop all accumulated coverage, and any localization state.
    pub fn clear(&self) {
        let mut shared = self.lock();
        shared.coverage.reset();
        if let Some(localizer) = shared.localizer.as_mut() {
            localizer.reset(None);
        }
        shared.estimates.clear();
        shared.localizer_timed_out = false;
    }
}

/// [`Stage::UpdateMappingAndLocalization`]'s coverage and belief half.
///
/// [`Stage::UpdateMappingAndLocalization`]: crate::stage::Stage::UpdateMappingAndLocalization
#[derive(Debug)]
pub struct MappingStage {
    link: MappingLink,
}

impl MappingStage {
    /// A stage over `bounds`, with the footprint radius still to be set.
    pub fn new(bounds: GridBounds) -> Result<Self, &'static str> {
        let coverage = CoverageGrid::new(bounds)?;
        Ok(Self {
            link: MappingLink {
                inner: Arc::new(Mutex::new(Shared {
                    bounds,
                    footprint_radius_m: MIN_FOOTPRINT_RADIUS_M,
                    // The reference's coverage map uses the metre-anchored
                    // lattice; see P4.6a for why the two differ.
                    sampling: Sampling::WorldAnchored,
                    scan_mode: true,
                    coverage,
                    localizer: None,
                    mission: None,
                    estimates: Vec::new(),
                    localizer_timed_out: false,
                })),
            },
        })
    }

    /// A stage whose footprint is derived the way the reference derives it.
    pub fn with_derived_footprint(
        bounds: GridBounds,
        mobile_count: usize,
    ) -> Result<Self, &'static str> {
        let stage = Self::new(bounds)?;
        let (nx, ny) = bounds.dimensions()?;
        stage.link.set_footprint_radius_m(footprint_radius_m(
            nx,
            ny,
            mobile_count,
            bounds.resolution_m,
        ));
        Ok(stage)
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> MappingLink {
        self.link.clone()
    }
}

impl StageHandler for MappingStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();

        let positions: Vec<(String, [f64; 2])> = context
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
            .collect();
        if positions.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let Shared {
            bounds,
            footprint_radius_m,
            sampling,
            ..
        } = *shared;

        let mut marked = 0usize;
        for (node_id, centre) in &positions {
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
                shared.coverage.mark_cell(*i, *j);
            }
        }

        let stats = shared.coverage.stats();

        // --- localization -------------------------------------------------
        // Reads the phase committed at the END of the previous tick, because
        // stage 8 runs before stage 9. See the module docs.
        let phase = shared
            .mission
            .as_ref()
            .map(|mission| mission.phase())
            // No mission machine means no phase gate, which is what a run
            // without one does.
            .unwrap_or(MissionPhase::Scanning);
        let localizing_phase = matches!(phase, MissionPhase::Scanning | MissionPhase::Localizing);

        shared.estimates.clear();
        if shared.localizer.is_some() && localizing_phase {
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

            for (node_id, _) in &positions {
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
                let nominal_position = [
                    platform.position.x,
                    platform.position.y,
                    platform.position.z,
                ];

                let localizer = shared.localizer.as_mut().expect("checked above");
                // The step the draw is keyed on is the one this update will
                // become, so it is the stored count plus one.
                let step = localizer.step_count(node_id) + 1;
                let scale = localizer.particle_scale(step, coverage_fraction);
                let config = localizer.config();
                let offsets = particle_offsets(seed, node_id, step, config.particle_count, scale);

                let candidates: Vec<[f64; 2]> = offsets
                    .iter()
                    .map(|offset| {
                        [
                            nominal_position[0] + offset[0],
                            nominal_position[1] + offset[1],
                        ]
                    })
                    .collect();
                let scores = score_footprints(
                    &counts,
                    [grid_nx, grid_ny],
                    origin,
                    bounds.resolution_m,
                    footprint_radius_m,
                    &candidates,
                )
                .map_err(|e| StageError::Invariant(format!("scoring {node_id}: {e}")))?;

                let estimate = localizer
                    .update(
                        StepInput {
                            platform_id: node_id,
                            nominal_position,
                            heading_rad,
                            timestamp_s: context.timestamp_s,
                            coverage_fraction,
                        },
                        &offsets,
                        &scores,
                    )
                    .map_err(|e| StageError::Invariant(format!("localizing {node_id}: {e}")))?;
                shared.estimates.push(estimate);
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
        drop(shared);

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
        Simulation::new(SimulationConfig {
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
        }
    }

    fn rig(platforms: Vec<NodeState>) -> (Simulation, MappingLink) {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        let stage = MappingStage::new(bounds()).expect("bounds are valid");
        let link = stage.link();
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(stage))
            .expect("stage 8 is not a transaction boundary");
        (sim, link)
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
            stage.link().footprint_radius_m(),
            footprint_radius_m(nx, ny, 4, bounds().resolution_m)
        );
    }

    // -----------------------------------------------------------------
    // Marking
    // -----------------------------------------------------------------

    #[test]
    fn a_drone_marks_a_disc_and_coverage_rises() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        assert_eq!(link.stats().covered_cells, 0);

        sim.step().unwrap();

        let stats = link.stats();
        assert!(stats.covered_cells > 0);
        assert_eq!(stats.total_cells, 20 * 20);
        assert!(stats.coverage_fraction > 0.0 && stats.coverage_fraction < 1.0);
        assert_eq!(outcome(&sim), StageOutcome::Ran);
        // The cell under the drone is marked.
        assert!(link.coverage_count_at(10, 10) > 0);
    }

    /// Revisiting the same ground raises revisits without raising coverage:
    /// `covered_cells` counts cells, not hits.
    #[test]
    fn revisiting_raises_revisits_not_coverage() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        sim.step().unwrap();
        let first = link.stats();

        for _ in 0..4 {
            sim.step().unwrap();
        }
        let later = link.stats();

        assert_eq!(later.covered_cells, first.covered_cells);
        assert_eq!(later.coverage_fraction, first.coverage_fraction);
        assert!(later.mean_revisits > first.mean_revisits);
    }

    #[test]
    fn a_fixed_node_never_marks_anything() {
        let (mut sim, link) = rig(vec![node("tower", false, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        sim.step().unwrap();
        assert_eq!(link.stats().covered_cells, 0);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    #[test]
    fn a_world_with_no_mobile_platforms_reports_no_work() {
        let (mut sim, link) = rig(Vec::new());
        sim.step().unwrap();
        assert_eq!(link.stats().covered_cells, 0);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// Two drones cover more between them than either does alone.
    #[test]
    fn two_drones_cover_more_than_one() {
        let one = {
            let (mut sim, link) = rig(vec![node("drone-1", true, 250.0, 500.0)]);
            link.set_footprint_radius_m(150.0);
            sim.step().unwrap();
            link.stats().covered_cells
        };
        let two = {
            let (mut sim, link) = rig(vec![
                node("drone-1", true, 250.0, 500.0),
                node("drone-2", true, 750.0, 500.0),
            ]);
            link.set_footprint_radius_m(150.0);
            sim.step().unwrap();
            link.stats().covered_cells
        };
        assert!(two > one, "{two} should exceed {one}");
    }

    /// A footprint entirely outside the grid marks nothing, and the stage says
    /// it had nothing to do rather than claiming it worked.
    #[test]
    fn a_footprint_off_the_grid_marks_nothing() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 50_000.0, 50_000.0)]);
        link.set_footprint_radius_m(10.0);
        sim.step().unwrap();
        assert_eq!(link.stats().covered_cells, 0);
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
        let (mut sim, link) = rig(vec![
            node("drone-1", true, 200.0, 200.0),
            node("drone-2", true, 800.0, 300.0),
        ]);
        link.set_footprint_radius_m(120.0);
        for k in 0..6 {
            // Move them around so cells are hit different numbers of times.
            let shift = f64::from(k) * 60.0;
            sim.world_mut().platform_mut("drone-1").unwrap().position.x = 200.0 + shift;
            sim.world_mut().platform_mut("drone-2").unwrap().position.y = 300.0 + shift;
            sim.step().unwrap();
        }
        // Reach the grid through the link's own accessor, then compare.
        let stats = link.stats();
        assert!(stats.covered_cells > 0);
        assert!(stats.mean_revisits >= 1.0);

        let mut covered = 0u64;
        let mut observations = 0u64;
        for i in 0..20 {
            for j in 0..20 {
                let count = u64::from(link.coverage_count_at(i, j));
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
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(200.0);
        sim.step().unwrap();
        assert_eq!(link.coverage_fraction(), link.stats().coverage_fraction);
    }

    #[test]
    fn clearing_drops_all_coverage() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        sim.step().unwrap();
        assert!(link.stats().covered_cells > 0);

        link.clear();
        let stats = link.stats();
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
            let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
            link.set_sampling(sampling);
            // 125 / 50 is not a whole number of cells.
            link.set_footprint_radius_m(125.0);
            sim.step().unwrap();
            link.stats().covered_cells
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
        let (mut sim, link) = rig(vec![node("drone-1", true, 950.0, 500.0)]);
        link.set_footprint_radius_m(150.0);

        sim.step().unwrap();

        assert_eq!(
            link.coverage_count_at(19, 10),
            2,
            "the edge sample and its neighbour both land on the last cell"
        );
        // One column in, nothing folds, so the same footprint hits once.
        assert_eq!(
            link.coverage_count_at(18, 10),
            1,
            "an unfolded cell is hit once by one footprint"
        );
    }

    // -----------------------------------------------------------------
    // Localization
    // -----------------------------------------------------------------

    fn localizing_rig(platforms: Vec<NodeState>) -> (Simulation, MappingLink) {
        let (sim, link) = rig(platforms);
        link.set_footprint_radius_m(150.0);
        link.enable_localization(LocalizationConfig::default());
        (sim, link)
    }

    /// With localization off — the default — stage 8 is coverage only, and
    /// says so by producing no estimates rather than empty ones.
    #[test]
    fn localization_is_off_until_it_is_enabled() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        for _ in 0..5 {
            sim.step().unwrap();
        }
        assert!(link.estimates().is_empty());
        assert!(link.confidences().is_empty());
        assert!(!link.localizer_timed_out());
    }

    #[test]
    fn an_enabled_localizer_produces_one_estimate_per_mobile_platform() {
        let (mut sim, link) = localizing_rig(vec![
            node("drone-a", true, 300.0, 300.0),
            node("tower", false, 500.0, 500.0),
            node("drone-b", true, 700.0, 700.0),
        ]);
        sim.step().unwrap();

        let estimates = link.estimates();
        assert_eq!(estimates.len(), 2, "one per mobile platform, not per node");
        let ids: Vec<&str> = estimates.iter().map(|e| e.platform_id.as_str()).collect();
        assert_eq!(ids, vec!["drone-a", "drone-b"]);
        assert_eq!(link.confidences().len(), 2);
    }

    /// Below the coverage floor the estimator reports no confidence at all,
    /// and it still records an estimate rather than skipping the platform.
    #[test]
    fn below_the_coverage_floor_confidence_is_zero() {
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        sim.step().unwrap();

        let coverage = link.coverage_fraction();
        assert!(
            coverage < LocalizationConfig::default().min_coverage_to_localize,
            "one footprint is {coverage}, which must be under the floor"
        );
        let estimates = link.estimates();
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0].confidence, 0.0);
    }

    /// Confidence rises as the map fills: that is the whole premise of
    /// matching a footprint against an accumulated coverage map.
    #[test]
    fn confidence_rises_as_the_map_fills() {
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);

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
            if let Some(estimate) = link.estimates().first() {
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
        assert!(link.coverage_fraction() > LocalizationConfig::default().min_coverage_to_localize);
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

        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        let mission = MissionStage::new(MissionConfig::default());
        let mission_link = mission.link();
        link.watch_mission(mission_link.clone());
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
            .unwrap();

        // Tick 0: still Scanning, so it localizes.
        let mut inputs = MissionInputs::at(0.0);
        inputs.mobile_count = 1;
        inputs.localization_confidences = vec![0.99];
        mission_link.set_inputs(inputs);
        sim.step().unwrap();
        assert_eq!(link.estimates().len(), 1, "Scanning localizes");

        // Push it out of Scanning.
        let mut done = MissionInputs::at(1.0);
        done.mobile_count = 1;
        done.localization_confidences = vec![0.99];
        done.coverage_fraction = 0.99;
        done.gap_cell_fraction = 0.0;
        mission_link.set_inputs(done);
        sim.step().unwrap();
        assert_eq!(mission_link.phase(), MissionPhase::Inspecting);
        assert_eq!(
            link.estimates().len(),
            1,
            "that tick still saw the PREVIOUS phase, which was Scanning"
        );

        // The next tick reads Inspecting and stops.
        let mut after = MissionInputs::at(2.0);
        after.mobile_count = 1;
        after.localization_confidences = vec![0.99];
        mission_link.set_inputs(after);
        sim.step().unwrap();
        assert!(link.estimates().is_empty(), "Inspecting must not localize");
    }

    /// `atan2(vy, vx)` above 0.1 m/s, and exactly **zero** below it — not the
    /// last heading. A platform that slows to a hover points east.
    #[test]
    fn heading_is_zero_below_the_speed_floor_not_the_last_heading() {
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);

        // Moving north: atan2(20, 0) = pi/2.
        {
            let platform = sim.world_mut().platform_mut("drone-1").unwrap();
            platform.velocity.x = 0.0;
            platform.velocity.y = 20.0;
        }
        sim.step().unwrap();
        let moving = link.estimates()[0].heading_rad;
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
            link.estimates()[0].heading_rad,
            0.0,
            "below 0.1 m/s the reference reports exactly zero"
        );
    }

    /// The floor is strict: exactly 0.1 m/s is still "too slow".
    #[test]
    fn the_heading_speed_floor_is_strict() {
        let heading_at = |speed: f64| {
            let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
            {
                let platform = sim.world_mut().platform_mut("drone-1").unwrap();
                platform.velocity.x = 0.0;
                platform.velocity.y = speed;
            }
            sim.step().unwrap();
            link.estimates()[0].heading_rad
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
            let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 200.0, 200.0)]);
            for k in 0..12 {
                {
                    let platform = sim.world_mut().platform_mut("drone-1").unwrap();
                    platform.position.x = 200.0 + f64::from(k) * 60.0;
                    platform.velocity.x = 15.0;
                }
                sim.step().unwrap();
            }
            link.estimates()
        };
        let first = run();
        let second = run();
        assert_eq!(first.len(), 1);
        assert_eq!(first, second, "a fixed seed and config must reproduce");
    }

    #[test]
    fn clearing_resets_the_localizer_too() {
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        sim.step().unwrap();
        assert!(!link.estimates().is_empty());

        link.clear();
        assert!(link.estimates().is_empty());
        assert!(!link.localizer_timed_out());
        assert_eq!(link.stats().covered_cells, 0);
    }

    /// Fly a lawnmower until coverage clears the localizer's floor, so the
    /// estimator is actually estimating rather than short-circuiting on
    /// `MapUnavailable`. Returns the drone to a fixed pose afterwards.
    fn cover_the_map(sim: &mut Simulation, link: &MappingLink, node_id: &str) {
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
            if link.coverage_fraction() > LocalizationConfig::default().min_coverage_to_localize {
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
            link.coverage_fraction() > LocalizationConfig::default().min_coverage_to_localize,
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
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");

        sim.step().unwrap();
        let first = link.estimates()[0].clone();
        sim.step().unwrap();
        let second = link.estimates()[0].clone();

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
        let (mut sim, link) = localizing_rig(vec![
            node("drone-a", true, 100.0, 100.0),
            node("drone-b", true, 100.0, 100.0),
        ]);
        cover_the_map(&mut sim, &link, "drone-a");
        {
            let platform = sim.world_mut().platform_mut("drone-b").unwrap();
            platform.position.x = 500.0;
            platform.position.y = 500.0;
            platform.velocity.x = 20.0;
        }
        sim.step().unwrap();

        let estimates = link.estimates();
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
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");
        sim.step().unwrap();

        let estimate = &link.estimates()[0];
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
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");
        sim.step().unwrap();

        let estimate = &link.estimates()[0];
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
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");

        // A fresh localizer with a short budget, over the map already built.
        link.enable_localization(LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        });
        assert!(!link.localizer_timed_out());

        for _ in 0..5 {
            sim.step().unwrap();
        }
        assert!(
            link.localizer_timed_out(),
            "running past the step budget must latch the timeout"
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
        let (mut sim, link) = rig(vec![node("drone-1", true, 500.0, 500.0)]);
        link.set_footprint_radius_m(150.0);
        link.enable_localization(LocalizationConfig {
            localization_timeout_steps: 3,
            ..LocalizationConfig::default()
        });

        // Stationary, so coverage never clears the floor.
        for _ in 0..40 {
            sim.step().unwrap();
        }
        assert!(link.coverage_fraction() < LocalizationConfig::default().min_coverage_to_localize);
        assert!(
            !link.localizer_timed_out(),
            "forty ticks with no map is zero localization steps"
        );
    }

    /// `clear` resets the localizer's own step counts, not just the published
    /// estimates -- otherwise a fresh epoch would inherit the previous one's
    /// annealing and start with a cloud far tighter than it has earned.
    #[test]
    fn clearing_resets_the_localizers_annealing() {
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");
        for _ in 0..20 {
            sim.step().unwrap();
        }
        let annealed = link.estimates()[0].position_std_m;

        link.clear();
        cover_the_map(&mut sim, &link, "drone-1");
        sim.step().unwrap();
        let fresh = link.estimates()[0].position_std_m;

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

        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 500.0, 500.0)]);
        let mission = MissionStage::new(MissionConfig::default());
        let mission_link = mission.link();
        link.watch_mission(mission_link.clone());
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
            .unwrap();

        // Coverage clears the gate but confidence has not, so the mission goes
        // Scanning -> Localizing rather than straight to Inspecting.
        let mut inputs = MissionInputs::at(0.0);
        inputs.mobile_count = 1;
        inputs.localization_confidences = vec![0.10];
        inputs.coverage_fraction = 0.99;
        inputs.gap_cell_fraction = 0.0;
        mission_link.set_inputs(inputs);
        sim.step().unwrap();
        assert_eq!(mission_link.phase(), MissionPhase::Localizing);

        let mut next = MissionInputs::at(1.0);
        next.mobile_count = 1;
        next.localization_confidences = vec![0.10];
        mission_link.set_inputs(next);
        sim.step().unwrap();

        assert_eq!(mission_link.phase(), MissionPhase::Localizing);
        assert_eq!(
            link.estimates().len(),
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
        let (mut sim, link) = localizing_rig(vec![node("drone-1", true, 100.0, 100.0)]);
        cover_the_map(&mut sim, &link, "drone-1");

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
        link.enable_localization(config);
        sim.step().unwrap();
        let from_stage = link.estimates()[0].clone();

        // Read the map AFTER the tick: stage 8 marks coverage and then
        // localizes, so the estimator sees this tick's own footprint. Reading
        // the previous tick's map would be a different -- and staler --
        // estimate.
        let coverage_fraction = link.coverage_fraction();
        let (grid_nx, grid_ny) = (20usize, 20usize);
        let mut counts: Vec<i32> = Vec::with_capacity(grid_nx * grid_ny);
        for i in 0..grid_nx {
            for j in 0..grid_ny {
                counts.push(i32::try_from(link.coverage_count_at(i, j)).unwrap());
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
}
