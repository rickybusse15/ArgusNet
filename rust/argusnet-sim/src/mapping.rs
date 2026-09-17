//! Stage 8's coverage and belief accumulation.
//!
//! §4 names stage 8 *"coverage/belief and localization"*. This is the
//! coverage-and-belief half; localization is a separate slice.
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

use argusnet_mapping::{rasterize_circle, CoverageGrid, CoverageStats, GridBounds, Sampling};

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

    /// Drop all accumulated coverage.
    pub fn clear(&self) {
        self.lock().coverage.reset();
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
        drop(shared);

        if marked == 0 {
            // Every platform's footprint fell outside the grid.
            return Ok(StageOutcome::NoWork);
        }
        context.note(format!(
            "mapping marked {marked} cell hit(s); coverage {:.4} over {} cells",
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

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 11,
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
}
