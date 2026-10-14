//! Stage 3's cooperative deconfliction.
//!
//! §4 names stage 3 *"cooperative deconfliction, with stable platform
//! ordering"*, and places it **after** the mission overrides of stage 2 and
//! **before** the safety monitor of stage 4. The reference says why in a
//! comment at the call site: the resolution *"runs after all controller and
//! trajectory overrides but before observation generation, so the tracker and
//! replay see the resolved positions."*
//!
//! The resolution itself is [`argusnet_planning::deconfliction::resolve_step`],
//! ported in P4.7g. This is the loop around it, and the loop has three
//! behaviours the reference states by construction rather than by comment.
//!
//! # Fewer than two mobile platforms is not a call at all
//!
//! ```python
//! if len(_mobile_proposed) >= 2:
//!     _resolved, _step_deconfliction_events = _deconfliction.resolve_step(...)
//! ```
//!
//! `resolve_step` also returns early below two proposals, so this reads like a
//! redundant guard. It is not redundant in one respect that matters to a
//! **trace**: with one drone the reference never enters the block, so there is
//! no call and nothing to record. Reporting `NoWork` here rather than `Ran`
//! keeps "nothing to deconflict" distinguishable from "deconfliction ran and
//! changed nothing", which §4's trace exists to separate.
//!
//! # Positions are written back only when an event was produced
//!
//! ```python
//! if _step_deconfliction_events:
//!     node_states = [ ... _resolved[ns.node_id] ... ]
//! ```
//!
//! The write-back is **guarded on the event list, not on the proposals having
//! changed**. Today the two agree — `resolve_step` only moves a platform when
//! it also emits an event — so the guard is invisible. It stops being
//! invisible the moment a resolution is added that adjusts without reporting,
//! and at that point the reference silently discards the adjustment.
//!
//! Preserved rather than simplified to "write back whatever came out": a port
//! that dropped the guard would apply an adjustment the reference does not,
//! and no existing test would notice. `an_adjustment_without_an_event_is_discarded`
//! pins it with a stub resolver, which is the only way to reach the case.
//!
//! # Only mobile platforms are proposed, and only they are written back
//!
//! A fixed node is never in the proposal map, so it can neither yield nor be
//! yielded to — and the write-back leaves anything absent from the resolved
//! map untouched.

use argusnet_planning::deconfliction::{
    resolve_step, DeconflictionConfig, DeconflictionEvent, ExclusionZone, FlightCorridor, Proposal,
};

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// Resolves a step's proposals.
///
/// Exists so a test can reach the "adjusted but did not report" case that
/// [`resolve_step`] never produces; production installs
/// [`PlanningResolver`], which is `resolve_step` itself.
pub trait ConflictResolver: std::fmt::Debug + Send {
    fn resolve(
        &mut self,
        config: &DeconflictionConfig,
        proposals: &[Proposal],
        timestamp_s: f64,
        zones: &[ExclusionZone],
        corridors: &[FlightCorridor],
        roles: &RoleTable,
    ) -> (Vec<Proposal>, Vec<DeconflictionEvent>);
}

/// Platform roles, and the terrain floor they are clamped against.
///
/// Both reach `resolve_step` as closures; carrying them as data keeps the
/// handler `Send` and keeps the terrain out of the stage.
#[derive(Clone, Default)]
pub struct RoleTable {
    roles: std::collections::BTreeMap<String, String>,
    /// Lowest altitude any platform may hold, or `None` for no clamp.
    floor_m: Option<f64>,
    /// The real terrain, when the run has one. Takes precedence over
    /// `floor_m`: the reference's `_clamp_terrain` is
    /// `terrain.clamp_altitude(xy, z, min_agl)`, a clamp against the ground
    /// under the platform, not against a constant.
    terrain: Option<std::sync::Arc<dyn crate::motion::TerrainSampler + Send + Sync>>,
}

impl std::fmt::Debug for RoleTable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RoleTable")
            .field("roles", &self.roles)
            .field("floor_m", &self.floor_m)
            .field("has_terrain", &self.terrain.is_some())
            .finish()
    }
}

impl RoleTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_role(mut self, drone_id: impl Into<String>, role: impl Into<String>) -> Self {
        self.roles.insert(drone_id.into(), role.into());
        self
    }

    /// A flat terrain floor. The reference clamps against real terrain; a
    /// constant is what this stage can honestly offer until the world is
    /// wired in, and `None` reproduces "no terrain to clamp against".
    pub fn with_floor_m(mut self, floor_m: Option<f64>) -> Self {
        self.floor_m = floor_m;
        self
    }

    /// Clamp against real terrain, as the reference does whenever its
    /// deconfliction layer is given one (every production scenario).
    pub fn with_terrain(
        mut self,
        terrain: std::sync::Arc<dyn crate::motion::TerrainSampler + Send + Sync>,
    ) -> Self {
        self.terrain = Some(terrain);
        self
    }

    /// The altitude a platform at `xy_m`, `z_m` is clamped to, or `None`.
    pub fn clamp(&self, xy_m: [f64; 2], z_m: f64, min_agl_m: f64) -> Option<f64> {
        match &self.terrain {
            Some(terrain) => Some(terrain.clamp_altitude(xy_m[0], xy_m[1], z_m, min_agl_m)),
            None => self.floor_m,
        }
    }

    pub fn role_of(&self, drone_id: &str) -> String {
        self.roles.get(drone_id).cloned().unwrap_or_default()
    }

    pub fn floor_m(&self) -> Option<f64> {
        self.floor_m
    }
}

/// The production resolver: `argusnet-planning`'s `resolve_step`.
#[derive(Debug, Default)]
pub struct PlanningResolver;

impl ConflictResolver for PlanningResolver {
    fn resolve(
        &mut self,
        config: &DeconflictionConfig,
        proposals: &[Proposal],
        timestamp_s: f64,
        zones: &[ExclusionZone],
        corridors: &[FlightCorridor],
        roles: &RoleTable,
    ) -> (Vec<Proposal>, Vec<DeconflictionEvent>) {
        resolve_step(
            config,
            proposals,
            timestamp_s,
            |id| roles.role_of(id),
            |xy, z, min_agl_m| roles.clamp(xy, z, min_agl_m),
            zones,
            corridors,
        )
    }
}

/// What the deconfliction stage is configured with, and what its last run
/// reported. Owned by the stage: callers outside the tick reach it through
/// [`Simulation::stage`](crate::simulation::Simulation::stage).
#[derive(Debug, Default)]
struct State {
    config: DeconflictionConfig,
    roles: RoleTable,
    zones: Vec<ExclusionZone>,
    corridors: Vec<FlightCorridor>,
    events: Vec<DeconflictionEvent>,
}

/// [`Stage::Deconflict`] as a [`StageHandler`].
///
/// [`Stage::Deconflict`]: crate::stage::Stage::Deconflict
#[derive(Debug)]
pub struct DeconflictStage {
    state: State,
    resolver: Box<dyn ConflictResolver>,
    /// Physics, in a run whose planners read a belief rather than the true
    /// world: a resolved position inside the ground or a structure is put
    /// back and recorded. `None` (every prior run) changes nothing.
    guard: Option<crate::belief_world::PhysicalGuard>,
}

impl Default for DeconflictStage {
    fn default() -> Self {
        Self::new()
    }
}

impl DeconflictStage {
    pub fn new() -> Self {
        Self::with_resolver(Box::new(PlanningResolver))
    }

    pub fn with_resolver(resolver: Box<dyn ConflictResolver>) -> Self {
        Self {
            state: State::default(),
            resolver,
            guard: None,
        }
    }

    /// Keep resolved positions out of the true ground and structures.
    pub fn set_physical_guard(&mut self, guard: crate::belief_world::PhysicalGuard) {
        self.guard = Some(guard);
    }

    /// Whether adjusted positions are kept out of the ground and structures.
    pub fn is_guarded(&self) -> bool {
        self.guard.is_some()
    }

    pub fn with_config(mut self, config: DeconflictionConfig) -> Self {
        self.state.config = config;
        self
    }

    pub fn with_roles(mut self, roles: RoleTable) -> Self {
        self.state.roles = roles;
        self
    }

    pub fn with_zones(mut self, zones: Vec<ExclusionZone>) -> Self {
        self.state.zones = zones;
        self
    }

    pub fn with_corridors(mut self, corridors: Vec<FlightCorridor>) -> Self {
        self.state.corridors = corridors;
        self
    }

    pub fn set_config(&mut self, config: DeconflictionConfig) {
        self.state.config = config;
    }

    pub fn set_roles(&mut self, roles: RoleTable) {
        self.state.roles = roles;
    }

    pub fn config(&self) -> DeconflictionConfig {
        self.state.config
    }

    /// The roles and terrain floor installed now.
    pub fn roles(&self) -> &RoleTable {
        &self.state.roles
    }

    pub fn has_terrain(&self) -> bool {
        self.state.roles.terrain.is_some()
    }

    /// Take the events the last run produced, leaving none behind.
    pub fn take_events(&mut self) -> Vec<DeconflictionEvent> {
        std::mem::take(&mut self.state.events)
    }
}

impl StageHandler for DeconflictStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let shared = &mut self.state;
        shared.events.clear();

        if !shared.config.enabled {
            return Ok(StageOutcome::Skipped {
                reason: "deconfliction disabled for this run".to_string(),
            });
        }

        // Declaration order, which §4 calls "stable platform ordering": the
        // corridor pass short-circuits later pairs, so the proposed order
        // decides who ends up holding.
        let proposals: Vec<Proposal> = context
            .world
            .platforms()
            .iter()
            .filter(|platform| platform.is_mobile)
            .map(|platform| Proposal {
                drone_id: platform.node_id.clone(),
                position: [
                    platform.position.x,
                    platform.position.y,
                    platform.position.z,
                ],
                velocity: [
                    platform.velocity.x,
                    platform.velocity.y,
                    platform.velocity.z,
                ],
            })
            .collect();

        // `if len(_mobile_proposed) >= 2` -- with one drone the reference never
        // enters the block, so there is no call to record.
        if proposals.len() < 2 {
            return Ok(StageOutcome::NoWork);
        }

        let (resolved, events) = {
            let State {
                config,
                roles,
                zones,
                corridors,
                ..
            } = &*shared;
            self.resolver.resolve(
                config,
                &proposals,
                context.timestamp_s,
                zones,
                corridors,
                roles,
            )
        };

        // Guarded on the EVENT LIST, not on the proposals having changed. See
        // the module docs: a resolution that adjusted without reporting has
        // its adjustment discarded, and that is the reference's behaviour.
        if events.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        for proposal in &resolved {
            let Some(platform) = context.world.platform_mut(&proposal.drone_id) else {
                // The resolver invented an id. A silent skip would let a
                // resolution go missing with nothing to show for it.
                return Err(StageError::Invariant(format!(
                    "deconfliction resolved unknown platform {}",
                    proposal.drone_id
                )));
            };
            let position = match &self.guard {
                Some(guard) => {
                    guard.enforce(&proposal.drone_id, context.timestamp_s, proposal.position)
                }
                None => proposal.position,
            };
            platform.position.x = position[0];
            platform.position.y = position[1];
            platform.position.z = position[2];
            platform.velocity.x = proposal.velocity[0];
            platform.velocity.y = proposal.velocity[1];
            platform.velocity.z = proposal.velocity[2];
        }

        for event in &events {
            context.note(format!(
                "deconflict {} yields to {} at {:.2} m ({})",
                event.yielding_drone_id,
                event.conflicting_drone_id,
                event.predicted_separation_m,
                event.resolution.as_str()
            ));
        }
        shared.events = events;
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_planning::deconfliction::Resolution;
    use argusnet_types::{NodeState, Vector3};
    use std::sync::{Arc, Mutex};

    fn simulation() -> Simulation {
        Simulation::bare(SimulationConfig {
            seed: 5,
            dt_s: 1.0,
            steps: Some(64),
            run_id: "deconflict".to_string(),
        })
        .expect("config is valid")
    }

    fn node(id: &str, mobile: bool, position: [f64; 3], velocity: [f64; 3]) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(position[0], position[1], position[2]),
            velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
            is_mobile: mobile,
            timestamp_s: 0.0,
            health: 1.0,
            ..Default::default()
        }
    }

    fn rig(platforms: Vec<NodeState>) -> Simulation {
        rig_with(platforms, DeconflictStage::new())
    }

    fn rig_with(platforms: Vec<NodeState>, stage: DeconflictStage) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        sim.set_stage_handler(Stage::Deconflict, Box::new(stage))
            .expect("stage 3 is not a transaction boundary");
        sim
    }

    /// The installed stage, reached through the session that owns it.
    fn decon(sim: &mut Simulation) -> &mut DeconflictStage {
        sim.stage_mut::<DeconflictStage>(Stage::Deconflict)
            .expect("stage 3 is a DeconflictStage")
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::Deconflict)
            .expect("stage 3 ran")
            .outcome
            .clone()
    }

    /// A resolver that moves a platform but reports nothing, to reach the case
    /// `resolve_step` never produces.
    #[derive(Debug)]
    struct SilentNudger;

    impl ConflictResolver for SilentNudger {
        fn resolve(
            &mut self,
            _config: &DeconflictionConfig,
            proposals: &[Proposal],
            _timestamp_s: f64,
            _zones: &[ExclusionZone],
            _corridors: &[FlightCorridor],
            _roles: &RoleTable,
        ) -> (Vec<Proposal>, Vec<DeconflictionEvent>) {
            let mut moved = proposals.to_vec();
            for proposal in &mut moved {
                proposal.position[1] += 1_000.0;
            }
            (moved, Vec::new())
        }
    }

    /// A resolver that reports an event and moves the first platform.
    #[derive(Debug)]
    struct AlwaysYield;

    impl ConflictResolver for AlwaysYield {
        fn resolve(
            &mut self,
            _config: &DeconflictionConfig,
            proposals: &[Proposal],
            timestamp_s: f64,
            _zones: &[ExclusionZone],
            _corridors: &[FlightCorridor],
            _roles: &RoleTable,
        ) -> (Vec<Proposal>, Vec<DeconflictionEvent>) {
            let mut moved = proposals.to_vec();
            moved[0].position[1] += 7.0;
            // Move every component, so a write-back that drops one is visible.
            moved[0].position[2] += 3.0;
            moved[0].velocity[0] += 11.0;
            moved[0].velocity[1] += 13.0;
            moved[0].velocity[2] += 17.0;
            let event = DeconflictionEvent {
                yielding_drone_id: moved[0].drone_id.clone(),
                conflicting_drone_id: moved[1].drone_id.clone(),
                predicted_separation_m: 3.0,
                resolution: Resolution::LateralOffset,
                timestamp_s,
            };
            (moved, vec![event])
        }
    }

    // -----------------------------------------------------------------
    // Two converging drones are separated
    // -----------------------------------------------------------------

    /// Two platforms closing head-on inside the minimum separation must be
    /// resolved, and the resolved positions must land in the world -- which is
    /// what "the tracker and replay see the resolved positions" requires.
    #[test]
    fn two_converging_drones_are_moved_apart_in_the_world() {
        let mut sim = rig(vec![
            node("drone-a", true, [0.0, 0.0, 60.0], [5.0, 0.0, 0.0]),
            node("drone-b", true, [10.0, 0.0, 60.0], [-5.0, 0.0, 0.0]),
        ]);
        let before: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();

        sim.step().expect("no stage fails");

        let events = decon(&mut sim).take_events();
        assert!(!events.is_empty(), "10 m apart and closing must conflict");
        assert_eq!(outcome(&sim), StageOutcome::Ran);

        let after: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();
        assert_ne!(before, after, "the resolution must reach the world");
    }

    /// Two platforms far apart are left alone, and the stage says so.
    #[test]
    fn well_separated_drones_are_left_alone() {
        let mut sim = rig(vec![
            node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            node("drone-b", true, [5_000.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
        ]);
        let before: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();

        sim.step().unwrap();

        assert!(decon(&mut sim).take_events().is_empty());
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
        let after: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();
        assert_eq!(before, after);
    }

    // -----------------------------------------------------------------
    // Who is proposed
    // -----------------------------------------------------------------

    /// Fewer than two mobile platforms is not a call at all.
    #[test]
    fn a_single_drone_is_never_deconflicted() {
        let mut sim = rig(vec![
            node("drone-a", true, [0.0, 0.0, 60.0], [1.0, 0.0, 0.0]),
            // Right on top of the drone, and fixed -- so it cannot conflict.
            node("tower", false, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
        ]);
        sim.step().unwrap();
        assert!(decon(&mut sim).take_events().is_empty());
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// A fixed node in the middle of a conflict is neither moved nor consulted.
    #[test]
    fn a_fixed_node_is_never_moved_by_a_resolution() {
        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [5.0, 0.0, 0.0]),
                node("tower", false, [1.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [2.0, 0.0, 60.0], [-5.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(AlwaysYield)),
        );
        let tower_before = sim.world().platform("tower").unwrap().position;

        sim.step().unwrap();

        assert!(!decon(&mut sim).take_events().is_empty());
        assert_eq!(
            sim.world().platform("tower").unwrap().position,
            tower_before,
            "a fixed node is not in the proposal map, so nothing can move it"
        );
        // The first mobile platform in declaration order is the one that moved,
        // and EVERY component of the resolution reached it: a write-back that
        // dropped velocity, or dropped z, would leave the rest looking right.
        let moved = sim.world().platform("drone-a").unwrap();
        assert_eq!(moved.position.y, 7.0);
        assert_eq!(moved.position.z, 63.0);
        assert_eq!(moved.velocity.x, 16.0);
        assert_eq!(moved.velocity.y, 13.0);
        assert_eq!(moved.velocity.z, 17.0);
    }

    // -----------------------------------------------------------------
    // The event-guarded write-back
    // -----------------------------------------------------------------

    /// The reference guards the write-back on the **event list**, not on the
    /// proposals having changed. A resolution that adjusts without reporting
    /// has its adjustment discarded, and this is the only way to reach it.
    #[test]
    fn an_adjustment_without_an_event_is_discarded() {
        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [5_000.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(SilentNudger)),
        );
        let before: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();

        sim.step().unwrap();

        assert!(decon(&mut sim).take_events().is_empty());
        let after: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();
        assert_eq!(
            before, after,
            "a 1 km nudge reported as nothing must not reach the world"
        );
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    // -----------------------------------------------------------------
    // Configuration and failure
    // -----------------------------------------------------------------

    #[test]
    fn a_disabled_layer_is_skipped_not_quiet() {
        let mut sim = rig(vec![
            node("drone-a", true, [0.0, 0.0, 60.0], [5.0, 0.0, 0.0]),
            node("drone-b", true, [10.0, 0.0, 60.0], [-5.0, 0.0, 0.0]),
        ]);
        decon(&mut sim).set_config(DeconflictionConfig {
            enabled: false,
            ..DeconflictionConfig::default()
        });
        let before: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();

        sim.step().unwrap();

        assert!(matches!(outcome(&sim), StageOutcome::Skipped { .. }));
        assert!(decon(&mut sim).take_events().is_empty());
        let after: Vec<_> = sim.world().platforms().iter().map(|p| p.position).collect();
        assert_eq!(before, after);
    }

    /// A wider minimum separation catches a pair a narrower one does not.
    #[test]
    fn the_minimum_separation_decides_what_conflicts() {
        let conflicts_at = |min_separation_m: f64| {
            let mut sim = rig(vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [40.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ]);
            decon(&mut sim).set_config(DeconflictionConfig {
                min_separation_m,
                ..DeconflictionConfig::default()
            });
            sim.step().unwrap();
            !decon(&mut sim).take_events().is_empty()
        };
        assert!(!conflicts_at(16.0), "40 m apart clears a 16 m minimum");
        assert!(conflicts_at(100.0), "40 m apart breaches a 100 m minimum");
    }

    /// A resolver naming a platform that is not in the fleet is a bug, and it
    /// faults the run rather than losing the resolution silently.
    #[test]
    fn a_resolution_for_an_unknown_platform_faults_the_run() {
        #[derive(Debug)]
        struct Inventor;
        impl ConflictResolver for Inventor {
            fn resolve(
                &mut self,
                _config: &DeconflictionConfig,
                proposals: &[Proposal],
                timestamp_s: f64,
                _zones: &[ExclusionZone],
                _corridors: &[FlightCorridor],
                _roles: &RoleTable,
            ) -> (Vec<Proposal>, Vec<DeconflictionEvent>) {
                let mut moved = proposals.to_vec();
                moved[0].drone_id = "ghost".to_string();
                let event = DeconflictionEvent {
                    yielding_drone_id: "ghost".to_string(),
                    conflicting_drone_id: proposals[1].drone_id.clone(),
                    predicted_separation_m: 1.0,
                    resolution: Resolution::LateralOffset,
                    timestamp_s,
                };
                (moved, vec![event])
            }
        }

        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [1.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(Inventor)),
        );

        assert!(sim.step().is_err());
        assert!(sim.is_faulted());
        assert!(sim.latest_snapshot().is_none(), "no partial snapshot");
    }

    #[test]
    fn events_are_taken_once_and_cleared_between_ticks() {
        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [1.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(AlwaysYield)),
        );
        sim.step().unwrap();
        assert_eq!(decon(&mut sim).take_events().len(), 1);
        assert!(
            decon(&mut sim).take_events().is_empty(),
            "a second take returns none"
        );

        // An untaken event does not survive into the next tick.
        sim.step().unwrap();
        sim.step().unwrap();
        assert_eq!(
            decon(&mut sim).take_events().len(),
            1,
            "each tick's events replace the last tick's rather than accruing"
        );
    }

    #[test]
    fn a_role_table_reports_the_default_role_for_an_unknown_platform() {
        let roles = RoleTable::new().with_role("drone-a", "interceptor");
        assert_eq!(roles.role_of("drone-a"), "interceptor");
        assert_eq!(roles.role_of("drone-z"), "");
        assert_eq!(roles.floor_m(), None);
        assert_eq!(
            RoleTable::new().with_floor_m(Some(12.0)).floor_m(),
            Some(12.0)
        );
    }

    /// The reference never enters the deconfliction block with fewer than two
    /// mobile platforms, so the resolver is not merely a no-op there -- it is
    /// not called. A counting resolver is the only way to see the difference.
    #[test]
    fn the_resolver_is_not_called_with_fewer_than_two_proposals() {
        #[derive(Debug, Default)]
        struct Counter(Arc<Mutex<Vec<usize>>>);
        impl ConflictResolver for Counter {
            fn resolve(
                &mut self,
                _config: &DeconflictionConfig,
                proposals: &[Proposal],
                _timestamp_s: f64,
                _zones: &[ExclusionZone],
                _corridors: &[FlightCorridor],
                _roles: &RoleTable,
            ) -> (Vec<Proposal>, Vec<DeconflictionEvent>) {
                self.0.lock().unwrap().push(proposals.len());
                (proposals.to_vec(), Vec::new())
            }
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [1.0, 0.0, 0.0]),
                node("tower", false, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(Counter(Arc::clone(&calls)))),
        );
        sim.step().unwrap();
        sim.step().unwrap();
        assert!(
            calls.lock().unwrap().is_empty(),
            "one mobile platform must not reach the resolver at all, got {:?}",
            calls.lock().unwrap()
        );
    }

    /// Role priority decides who gives way: the lower-priority platform
    /// yields. Without the roles reaching the resolver both would be
    /// `DEFAULT_ROLE_PRIORITY` and the tie-break -- the LARGER id yields --
    /// would pick the other one.
    #[test]
    fn role_priority_decides_who_yields() {
        let yielded_by = |roles: RoleTable| {
            let mut sim = rig(vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [5.0, 0.0, 0.0]),
                node("drone-b", true, [10.0, 0.0, 60.0], [-5.0, 0.0, 0.0]),
            ]);
            decon(&mut sim).set_roles(roles);
            sim.step().unwrap();
            let events = decon(&mut sim).take_events();
            assert!(!events.is_empty(), "the pair must conflict");
            events[0].yielding_drone_id.clone()
        };

        // No roles: both default, so the tie-break applies and the LARGER id
        // ("drone-b") gives way.
        assert_eq!(yielded_by(RoleTable::new()), "drone-b");

        // Give drone-b the higher priority and drone-a must yield instead.
        assert_eq!(
            yielded_by(
                RoleTable::new()
                    .with_role("drone-a", "reserve")
                    .with_role("drone-b", "primary_observer")
            ),
            "drone-a",
            "the lower-priority platform gives way, so the roles must reach \
             the resolver"
        );
    }

    /// The stage clears its events at the top of each run, so a tick that
    /// returns early cannot hand back the previous tick's events.
    ///
    /// The `shared.events = events` assignment on the success path masks this:
    /// only an early return -- disabled, or fewer than two proposals -- can
    /// leave a stale list behind.
    #[test]
    fn a_tick_that_returns_early_does_not_replay_the_last_ticks_events() {
        let mut sim = rig_with(
            vec![
                node("drone-a", true, [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
                node("drone-b", true, [1.0, 0.0, 60.0], [0.0, 0.0, 0.0]),
            ],
            DeconflictStage::with_resolver(Box::new(AlwaysYield)),
        );
        sim.step().unwrap();
        // Deliberately do NOT take them.

        // Now make the next tick return early, before the resolver runs.
        decon(&mut sim).set_config(DeconflictionConfig {
            enabled: false,
            ..DeconflictionConfig::default()
        });
        sim.step().unwrap();

        assert!(
            decon(&mut sim).take_events().is_empty(),
            "an early return must leave no events, not the previous tick's"
        );
    }
}
