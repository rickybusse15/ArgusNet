//! Stage 4's hard safety monitor.
//!
//! §4 names stage 4 *"opt-in hard safety monitor and clamp. Runs **after**
//! deconfliction."* The reference is explicit about why it is opt-in:
//!
//! > *"Hard safety floor runs after cooperative deconfliction. It is opt-in so
//! > legacy seeded simulations remain bit-identical."*
//!
//! So a run without `safety_blocking` must not merely produce the same numbers
//! — the monitor must not run at all. [`StageOutcome::Skipped`] says that,
//! where `NoWork` would claim it ran and found nothing.
//!
//! The policy is [`safety_engine::simulation::SimulationSafetyMonitor`]. This
//! is the loop around it.
//!
//! # A monitor per platform, with limits chosen by a substring
//!
//! ```python
//! role = scenario.drone_roles.get(ns.node_id, "tracker")
//! monitor = SafetyMonitor(
//!     DronePhysicalLimits.interceptor_default()
//!     if "interceptor" in role
//!     else DronePhysicalLimits.tracker_default()
//! )
//! ```
//!
//! `"interceptor" in role` is a **substring** test, not an equality. A role of
//! `"interceptor-lead"` or `"backup_interceptor"` gets the interceptor limits;
//! anything else, including an unknown role, gets the tracker's. The limits
//! differ — 42 m/s against 35 m/s top speed, and a higher minimum AGL — so
//! reading it as equality silently gives half the fleet the wrong envelope.
//!
//! The monitor is created **once per platform** and kept, because it carries
//! the consecutive-violation count that decides `warning` from `abort`.
//! Rebuilding it each tick would mean a platform could never escalate.
//!
//! # `previous_velocity` is the clamped velocity, not the proposed one
//!
//! ```python
//! _safety_previous_velocities[ns.node_id] = decision.velocity.copy()
//! ```
//!
//! The monitor derives acceleration from the previous velocity, and what it
//! stores is what it **allowed**, not what was asked for. A platform commanded
//! far past its limit therefore shows a plausible acceleration next tick rather
//! than an enormous one, because the clamp already absorbed the difference.
//!
//! # Peers are every *other* mobile platform, post-deconfliction
//!
//! Separation is checked against the positions as they stand **after** stage 3
//! has already moved people apart, which is what makes stage 4 a floor under
//! deconfliction rather than a competitor to it.
//!
//! # The energy sentinel reaches the monitor
//!
//! The reference passes `ns.battery_fraction` straight through, and that field
//! is **-1.0** for a platform with no battery in the ledger. `check` flags
//! `min_energy_reserve` for any fraction below the limit, so a *mobile*
//! platform with no battery would violate its energy reserve on every tick,
//! for ever.
//!
//! It does not happen today only because the reference registers a battery for
//! every mobile node at start-up. Preserved rather than papered over with a
//! `None`: the sentinel is what the reference sends, and
//! `a_mobile_platform_with_no_battery_violates_its_energy_reserve` records what
//! that costs.
//!
//! # What P4.9d corrected before assembling it
//!
//! Stage 4 was ported and tested in P4.8c but never assembled from a bundle,
//! and three of its arguments were not the reference's:
//!
//! * **`dt_s`** was the tick's *timestamp*. The reference passes
//!   `simulation_config.dt_s`, and the monitor derives acceleration from it.
//! * **The caution fraction** was 0.85. The reference constructs
//!   `SafetyMonitor(limits)`, whose default is **0.9**.
//! * **Terrain** was one constant height. The reference samples
//!   `scenario.terrain.height_at(x, y)` under each platform, after
//!   deconfliction has moved it.
//!
//! The monitors' escalation records and the stored previous velocities are
//! causal state, so the stage is on the checkpoint contract (P4.8e.4). And the
//! tick's validations reach stage 7 as `safety_events`, as
//! `_ingest_stream.ingest_frame(..., safety_events=_step_safety_events)` sends
//! them, so the fused frame, and the replay, carry them.

use std::collections::BTreeMap;

use safety_engine::limits::DronePhysicalLimits;
use safety_engine::simulation::{SimulationSafetyMonitor, State, Violation};
use serde::{Deserialize, Serialize};

use crate::battery::{stage_one_battery, NOT_APPLICABLE_FRACTION};
use crate::observe::TerrainHeight;
use crate::simulation::{decode_stage_state, StageHandler, StageStateError, TickContext};
use crate::stage::{StageError, StageOutcome};

/// `SimulationSafetyMonitor::new`'s reference arguments.
pub const DEFAULT_ABORT_FRAMES: u32 = 3;
/// `SafetyMonitor.__init__`'s default, which the reference does not override.
pub const DEFAULT_CAUTION_FRACTION: f64 = 0.9;
/// `scenario.drone_roles.get(ns.node_id, "tracker")`.
pub const DEFAULT_ROLE: &str = "tracker";

/// The substring that selects the interceptor envelope.
pub const INTERCEPTOR_ROLE_MARKER: &str = "interceptor";

/// One platform's safety outcome for a tick, mirroring
/// `SafetyValidationResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyValidation {
    pub validation_id: String,
    pub subject_id: String,
    /// `not decision.blocked`, and `blocked` is exactly `state is ABORT`.
    pub accepted: bool,
    pub timestamp_s: f64,
    pub validator: String,
    pub violations: Vec<String>,
    /// Whether the clamp actually moved the platform.
    pub clamped: bool,
    pub reason: String,
}

impl SafetyValidation {
    /// `_safety_event_to_proto`: the form stage 7 receives and echoes.
    pub fn to_event(&self) -> argusnet_types::SafetyEvent {
        argusnet_types::SafetyEvent {
            event_id: self.validation_id.clone(),
            subject_id: self.subject_id.clone(),
            timestamp_s: self.timestamp_s,
            // `value.reason or ("nominal" if value.accepted else "abort")`;
            // the monitor always gives a reason, its state.
            state: if self.reason.is_empty() {
                if self.accepted { "nominal" } else { "abort" }.to_string()
            } else {
                self.reason.clone()
            },
            violations: self.violations.clone(),
            command_clamped: self.clamped,
            blocked: !self.accepted,
            reason: self.reason.clone(),
        }
    }
}

/// `validator="blocking_monitor"`, hardcoded in the reference.
pub const VALIDATOR_NAME: &str = "blocking_monitor";

/// Which physical envelope a role selects.
///
/// A free function because the substring rule is the part worth naming: it is
/// not an equality, and reading it as one gives half the fleet the wrong
/// limits without any error.
pub fn limits_for_role(role: &str) -> DronePhysicalLimits {
    if role.contains(INTERCEPTOR_ROLE_MARKER) {
        DronePhysicalLimits::interceptor_default()
    } else {
        DronePhysicalLimits::tracker_default()
    }
}

/// The monitors, their configuration and the last run's validations, owned
/// by the stage.
#[derive(Debug)]
struct SafetyState {
    enabled: bool,
    abort_frames: u32,
    caution_fraction: f64,
    roles: BTreeMap<String, String>,
    /// Sampled under each platform, post-deconfliction.
    terrain: Box<dyn TerrainHeight>,
    monitors: BTreeMap<String, SimulationSafetyMonitor>,
    previous_velocities: BTreeMap<String, [f64; 3]>,
    reads_battery: bool,
    validations: Vec<SafetyValidation>,
}

/// One height everywhere, for rigs that have no terrain model.
#[derive(Debug, Clone, Copy)]
struct Level(f64);

impl TerrainHeight for Level {
    fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
        self.0
    }
}

/// What survives a tick: each platform's escalation record and the velocity
/// the monitor last allowed it.
#[derive(Debug, Serialize, Deserialize)]
struct SafetyCheckpoint {
    /// `node_id -> (consecutive violating frames, state)`.
    records: BTreeMap<String, (u32, String)>,
    previous_velocities: BTreeMap<String, [f64; 3]>,
}

/// The states `SimulationSafetyMonitor::process` assigns.
const MONITOR_STATES: [&str; 4] = ["nominal", "caution", "warning", "abort"];

impl Default for SafetyState {
    fn default() -> Self {
        Self {
            // Opt-in, exactly as `scenario.options.safety_blocking` is.
            enabled: false,
            abort_frames: DEFAULT_ABORT_FRAMES,
            caution_fraction: DEFAULT_CAUTION_FRACTION,
            roles: BTreeMap::new(),
            terrain: Box::new(Level(0.0)),
            monitors: BTreeMap::new(),
            previous_velocities: BTreeMap::new(),
            reads_battery: false,
            validations: Vec::new(),
        }
    }
}

/// [`Stage::SafetyMonitor`] as a [`StageHandler`].
///
/// [`Stage::SafetyMonitor`]: crate::stage::Stage::SafetyMonitor
#[derive(Debug, Default)]
pub struct SafetyStage {
    state: SafetyState,
}

impl SafetyStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// `scenario.options.safety_blocking`. Off by default, so a run that does
    /// not ask for the hard floor does not get one.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.state.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.state.enabled
    }

    pub fn set_role(&mut self, drone_id: impl Into<String>, role: impl Into<String>) {
        self.state.roles.insert(drone_id.into(), role.into());
    }

    /// A level floor at `terrain_height_m`, for rigs with no terrain model.
    pub fn set_terrain_height_m(&mut self, terrain_height_m: f64) {
        self.state.terrain = Box::new(Level(terrain_height_m));
    }

    /// The terrain the reference samples under each platform.
    pub fn set_terrain(&mut self, terrain: Box<dyn TerrainHeight>) {
        self.state.terrain = terrain;
    }

    /// Read `energy_fraction` from stage 1's ledger.
    ///
    /// Without this the monitor is handed `None` and the energy checks do not
    /// fire at all — which is *not* what the reference does, so a production
    /// wiring should always supply it.
    ///
    /// Read through the views lent to stage 4 (SESSION_OWNERSHIP.md §2a):
    /// stage 1 has already run this tick. A stage 1 with no ledger reads as
    /// an empty one, [`NOT_APPLICABLE_FRACTION`] for everyone.
    pub fn read_stage_one_battery(&mut self) {
        self.state.reads_battery = true;
    }

    /// The safety state the monitor last assigned, or `"nominal"`.
    pub fn state_of(&self, drone_id: &str) -> String {
        self.state
            .monitors
            .get(drone_id)
            .map(|monitor| monitor.state(drone_id).to_string())
            .unwrap_or_else(|| "nominal".to_string())
    }

    /// The last run's validations as stage 7's `safety_events`, which
    /// [`Handoff::SafetyEventsToFusion`] hands it.
    ///
    /// [`Handoff::SafetyEventsToFusion`]: crate::simulation::Handoff::SafetyEventsToFusion
    pub fn safety_events(&self) -> Vec<argusnet_types::SafetyEvent> {
        self.state
            .validations
            .iter()
            .map(SafetyValidation::to_event)
            .collect()
    }

    /// Take the validations the last run produced, leaving none behind.
    pub fn take_validations(&mut self) -> Vec<SafetyValidation> {
        std::mem::take(&mut self.state.validations)
    }

    /// Drop every monitor, so a reset does not carry the previous epoch's
    /// escalation counts into a fresh run.
    pub fn clear(&mut self) {
        let shared = &mut self.state;
        shared.monitors.clear();
        shared.previous_velocities.clear();
        shared.validations.clear();
    }
}

impl StageHandler for SafetyStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let shared = &mut self.state;
        shared.validations.clear();

        if !shared.enabled {
            return Ok(StageOutcome::Skipped {
                reason: "safety_blocking is off, so legacy seeded runs stay bit-identical"
                    .to_string(),
            });
        }

        // Post-deconfliction positions: stage 4 is a floor UNDER stage 3, not a
        // competitor to it.
        let mobile: Vec<(String, [f64; 3], [f64; 3])> = context
            .world
            .platforms()
            .iter()
            .filter(|platform| platform.is_mobile)
            .map(|platform| {
                (
                    platform.node_id.clone(),
                    [
                        platform.position.x,
                        platform.position.y,
                        platform.position.z,
                    ],
                    [
                        platform.velocity.x,
                        platform.velocity.y,
                        platform.velocity.z,
                    ],
                )
            })
            .collect();
        if mobile.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let abort_frames = shared.abort_frames;
        let caution_fraction = shared.caution_fraction;
        // `simulation_config.dt_s`: the tick interval, not its timestamp.
        let dt_s = context.dt_s;
        let tick = context.tick;
        let mut clamped_any = false;
        let mut notes: Vec<String> = Vec::new();
        let battery = stage_one_battery(context.stages);

        for (node_id, position, velocity) in &mobile {
            let peers: Vec<[f64; 3]> = mobile
                .iter()
                .filter(|(peer_id, _, _)| peer_id != node_id)
                .map(|(_, peer_position, _)| *peer_position)
                .collect();

            // The sentinel goes through untouched -- see the module docs.
            let energy_fraction = shared.reads_battery.then(|| {
                battery.map_or(NOT_APPLICABLE_FRACTION, |battery| battery.fraction(node_id))
            });

            if !shared.monitors.contains_key(node_id) {
                let role = shared
                    .roles
                    .get(node_id)
                    .cloned()
                    .unwrap_or_else(|| DEFAULT_ROLE.to_string());
                let monitor = SimulationSafetyMonitor::new(
                    limits_for_role(&role),
                    abort_frames,
                    caution_fraction,
                )
                .map_err(StageError::Invariant)?;
                shared.monitors.insert(node_id.clone(), monitor);
            }

            let previous_velocity = shared.previous_velocities.get(node_id).copied();
            let terrain_height_m = shared.terrain.height_at(position[0], position[1]);
            let state = State {
                position: *position,
                velocity: *velocity,
                terrain_height_m,
                acceleration: None,
                previous_velocity,
                dt_s: Some(dt_s),
                other_drone_positions: peers,
                energy_fraction,
            };

            let monitor = shared
                .monitors
                .get_mut(node_id)
                .expect("monitor was just inserted");
            let decision = monitor.process(node_id, &state);

            // What is stored is what was ALLOWED, not what was asked for.
            shared
                .previous_velocities
                .insert(node_id.clone(), decision.velocity);

            let clamped = decision.position != *position || decision.velocity != *velocity;
            clamped_any |= clamped;

            if !decision.violations.is_empty() {
                shared.validations.push(SafetyValidation {
                    validation_id: format!("safety-{tick}-{node_id}"),
                    subject_id: node_id.clone(),
                    // `not decision.blocked`, and `blocked` is exactly
                    // `state is ABORT`.
                    accepted: decision.state != "abort",
                    timestamp_s: context.timestamp_s,
                    validator: VALIDATOR_NAME.to_string(),
                    violations: decision
                        .violations
                        .iter()
                        .map(|violation: &Violation| violation.constraint.clone())
                        .collect(),
                    clamped,
                    reason: decision.state.clone(),
                });
                notes.push(format!(
                    "safety {node_id} {} ({} violation(s){})",
                    decision.state,
                    decision.violations.len(),
                    if clamped { ", clamped" } else { "" }
                ));
            }

            // The decision is written back whether or not it violated
            // anything: the clamp is unconditional.
            let Some(platform) = context.world.platform_mut(node_id) else {
                return Err(StageError::Invariant(format!(
                    "safety monitor lost platform {node_id} mid-tick"
                )));
            };
            platform.position.x = decision.position[0];
            platform.position.y = decision.position[1];
            platform.position.z = decision.position[2];
            platform.velocity.x = decision.velocity[0];
            platform.velocity.y = decision.velocity[1];
            platform.velocity.z = decision.velocity[2];
        }

        let reported = !shared.validations.is_empty();
        for note in notes {
            context.note(note);
        }

        Ok(if reported || clamped_any {
            StageOutcome::Ran
        } else {
            StageOutcome::NoWork
        })
    }

    /// A fresh run of the same scenario: no escalation, no previous velocity.
    fn reset(&mut self) {
        self.clear();
    }

    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        let state = SafetyCheckpoint {
            records: shared
                .monitors
                .iter()
                .filter_map(|(id, monitor)| {
                    monitor
                        .record(id)
                        .map(|(count, state)| (id.clone(), (count, state.to_string())))
                })
                .collect(),
            previous_velocities: shared.previous_velocities.clone(),
        };
        Some(serde_json::to_value(state).expect("safety checkpoint serialises"))
    }

    fn check_state(&self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: SafetyCheckpoint = decode_stage_state(state)?;
        if let Some((id, (_, state))) = decoded
            .records
            .iter()
            .find(|(_, (_, state))| !MONITOR_STATES.contains(&state.as_str()))
        {
            return Err(StageStateError::Malformed {
                detail: format!(
                    "safety monitor for {id} is in no state the monitor assigns: {state:?}"
                ),
            });
        }
        if let Some((id, _)) = decoded
            .previous_velocities
            .iter()
            .find(|(_, velocity)| !velocity.iter().all(|v| v.is_finite()))
        {
            return Err(StageStateError::Malformed {
                detail: format!("safety previous velocity for {id} is not finite"),
            });
        }
        Ok(())
    }

    fn restore_state(&mut self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: SafetyCheckpoint = decode_stage_state(state)?;
        let shared = &mut self.state;
        let mut monitors = BTreeMap::new();
        for (id, (count, state)) in decoded.records {
            let role = shared
                .roles
                .get(&id)
                .cloned()
                .unwrap_or_else(|| DEFAULT_ROLE.to_string());
            let mut monitor = SimulationSafetyMonitor::new(
                limits_for_role(&role),
                shared.abort_frames,
                shared.caution_fraction,
            )
            .map_err(|detail| StageStateError::Malformed { detail })?;
            monitor
                .restore_record(&id, count, &state)
                .map_err(|detail| StageStateError::Malformed { detail })?;
            monitors.insert(id, monitor);
        }
        shared.monitors = monitors;
        shared.previous_velocities = decoded.previous_velocities;
        shared.validations.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_types::{NodeState, Vector3};

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 9,
            dt_s: 1.0,
            steps: Some(256),
            run_id: "safety".to_string(),
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

    /// Enabled by default in the rig, since almost every test wants the floor.
    fn rig(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        let mut stage = SafetyStage::new();
        stage.set_enabled(true);
        sim.set_stage_handler(Stage::SafetyMonitor, Box::new(stage))
            .expect("stage 4 is not a transaction boundary");
        sim
    }

    /// The installed stage, reached through the session that owns it.
    fn safety(sim: &mut Simulation) -> &mut SafetyStage {
        sim.stage_mut::<SafetyStage>(Stage::SafetyMonitor)
            .expect("stage 4 is a SafetyStage")
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::SafetyMonitor)
            .expect("stage 4 ran")
            .outcome
            .clone()
    }

    // -----------------------------------------------------------------
    // Opt-in
    // -----------------------------------------------------------------

    /// A run without `safety_blocking` must not merely produce the same
    /// numbers -- the monitor must not run at all, which is what keeps legacy
    /// seeded runs bit-identical.
    #[test]
    fn a_run_without_safety_blocking_is_skipped_not_quiet() {
        let mut sim = simulation();
        // Deep underground and far too fast: it would be clamped hard.
        sim.load_world(
            vec![node("drone-1", true, [0.0, 0.0, -500.0], [900.0, 0.0, 0.0])],
            Vec::new(),
        );
        let stage = SafetyStage::new();
        assert!(!stage.is_enabled(), "off unless asked for");
        sim.set_stage_handler(Stage::SafetyMonitor, Box::new(stage))
            .unwrap();

        sim.step().unwrap();

        assert!(matches!(outcome(&sim), StageOutcome::Skipped { .. }));
        assert!(safety(&mut sim).take_validations().is_empty());
        let platform = sim.world().platform("drone-1").unwrap();
        assert_eq!(platform.position.z, -500.0, "nothing was clamped");
        assert_eq!(platform.velocity.x, 900.0);
    }

    // -----------------------------------------------------------------
    // The clamp
    // -----------------------------------------------------------------

    #[test]
    fn a_platform_below_the_terrain_floor_is_lifted_to_the_minimum_agl() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [1.0, 0.0, 0.0],
        )]);
        safety(&mut sim).set_terrain_height_m(0.0);

        sim.step().unwrap();

        let limits = limits_for_role(DEFAULT_ROLE);
        let platform = sim.world().platform("drone-1").unwrap();
        assert_eq!(
            platform.position.z, limits.min_agl_m,
            "clamped up to terrain + min_agl"
        );
        let validations = safety(&mut sim).take_validations();
        assert_eq!(validations.len(), 1);
        assert!(validations[0].clamped);
        assert!(validations[0].violations.iter().any(|v| v == "min_agl"));
    }

    /// A `min_agl` violation aborts immediately rather than warning first, and
    /// an abort zeroes the velocity.
    #[test]
    fn a_minimum_agl_violation_aborts_on_the_first_frame() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [20.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();

        let validations = safety(&mut sim).take_validations();
        assert_eq!(validations[0].reason, "abort");
        assert!(
            !validations[0].accepted,
            "blocked is exactly state == abort"
        );
        let platform = sim.world().platform("drone-1").unwrap();
        assert_eq!(platform.velocity.x, 0.0);
        assert_eq!(platform.velocity.y, 0.0);
        assert_eq!(platform.velocity.z, 0.0);
    }

    #[test]
    fn an_overspeed_platform_is_scaled_back_to_its_limit() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();

        let limits = limits_for_role(DEFAULT_ROLE);
        let platform = sim.world().platform("drone-1").unwrap();
        assert!(
            (platform.velocity.x - limits.max_speed_mps).abs() < 1e-9,
            "scaled to {} not {}",
            platform.velocity.x,
            limits.max_speed_mps
        );
        assert!(safety(&mut sim).take_validations()[0].clamped);
    }

    /// A platform well inside every limit is untouched, and the stage says it
    /// had nothing to do.
    #[test]
    fn a_compliant_platform_is_untouched() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [5.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();

        let platform = sim.world().platform("drone-1").unwrap();
        assert_eq!(platform.position.z, 100.0);
        assert_eq!(platform.velocity.x, 5.0);
        assert!(safety(&mut sim).take_validations().is_empty());
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    #[test]
    fn a_fixed_node_is_never_clamped() {
        let mut sim = rig(vec![node(
            "tower",
            false,
            [0.0, 0.0, -900.0],
            [800.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();

        let tower = sim.world().platform("tower").unwrap();
        assert_eq!(tower.position.z, -900.0);
        assert_eq!(tower.velocity.x, 800.0);
        assert!(safety(&mut sim).take_validations().is_empty());
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    // -----------------------------------------------------------------
    // Roles
    // -----------------------------------------------------------------

    /// `"interceptor" in role` is a substring test. A role of
    /// `"interceptor-lead"` gets the interceptor envelope; reading it as an
    /// equality would silently give it the tracker's.
    #[test]
    fn the_interceptor_envelope_is_selected_by_substring_not_equality() {
        let tracker = DronePhysicalLimits::tracker_default();
        let interceptor = DronePhysicalLimits::interceptor_default();
        assert_ne!(tracker.max_speed_mps, interceptor.max_speed_mps);

        assert_eq!(
            limits_for_role("interceptor").max_speed_mps,
            interceptor.max_speed_mps
        );
        assert_eq!(
            limits_for_role("interceptor-lead").max_speed_mps,
            interceptor.max_speed_mps,
            "a substring match, not an equality"
        );
        assert_eq!(
            limits_for_role("backup_interceptor").max_speed_mps,
            interceptor.max_speed_mps
        );
        assert_eq!(
            limits_for_role("tracker").max_speed_mps,
            tracker.max_speed_mps
        );
        assert_eq!(
            limits_for_role("something-unknown").max_speed_mps,
            tracker.max_speed_mps,
            "an unknown role takes the tracker envelope"
        );
    }

    /// And the role actually reaches the monitor: the same speed is legal for
    /// an interceptor and clamped for a tracker.
    #[test]
    fn the_role_changes_what_is_clamped() {
        let tracker_limit = DronePhysicalLimits::tracker_default().max_speed_mps;
        let interceptor_limit = DronePhysicalLimits::interceptor_default().max_speed_mps;
        // Between the two envelopes.
        let speed = (tracker_limit + interceptor_limit) / 2.0;

        let speed_after = |role: &str| {
            let mut sim = rig(vec![node(
                "drone-1",
                true,
                [0.0, 0.0, 100.0],
                [speed, 0.0, 0.0],
            )]);
            safety(&mut sim).set_role("drone-1", role);
            sim.step().unwrap();
            sim.world().platform("drone-1").unwrap().velocity.x
        };

        assert!(
            (speed_after("interceptor-lead") - speed).abs() < 1e-9,
            "legal for an interceptor"
        );
        assert!(
            (speed_after("tracker") - tracker_limit).abs() < 1e-9,
            "clamped for a tracker"
        );
    }

    // -----------------------------------------------------------------
    // State that survives the tick
    // -----------------------------------------------------------------

    /// The monitor is created once and kept, because it carries the
    /// consecutive-violation count that escalates `warning` to `abort`.
    /// Rebuilding it each tick would mean a platform could never escalate.
    #[test]
    fn repeated_violations_escalate_from_warning_to_abort() {
        // Overspeed violates without tripping the immediate-abort min_agl path.
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 0.0],
        )]);

        let mut states = Vec::new();
        for _ in 0..DEFAULT_ABORT_FRAMES + 1 {
            // Re-command the illegal speed each tick; the clamp undid it.
            sim.world_mut().platform_mut("drone-1").unwrap().velocity.x = 500.0;
            sim.step().unwrap();
            states.extend(
                safety(&mut sim)
                    .take_validations()
                    .into_iter()
                    .map(|v| v.reason),
            );
        }

        assert_eq!(states.first().map(String::as_str), Some("warning"));
        assert!(
            states.iter().any(|s| s == "abort"),
            "the count must survive the tick: {states:?}"
        );
    }

    /// What is stored as `previous_velocity` is what the monitor **allowed**,
    /// not what was proposed. Storing the proposal instead would show an
    /// enormous acceleration next tick, because the clamp already absorbed the
    /// difference.
    #[test]
    fn the_previous_velocity_is_the_clamped_one() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        let clamped = sim.world().platform("drone-1").unwrap().velocity.x;
        assert!(clamped < 500.0);

        // Next tick, hold exactly the clamped speed. Against the stored
        // *clamped* velocity that is zero acceleration and no new violation;
        // against the stored *proposal* it would look like a huge deceleration.
        sim.world_mut().platform_mut("drone-1").unwrap().velocity.x = clamped;
        sim.step().unwrap();
        let violations: Vec<String> = safety(&mut sim)
            .take_validations()
            .into_iter()
            .flat_map(|v| v.violations)
            .collect();
        assert!(
            !violations.iter().any(|v| v.contains("accel")),
            "holding the allowed speed is not an acceleration: {violations:?}"
        );
    }

    /// The reference constructs `SafetyMonitor(limits)`, whose caution
    /// fraction defaults to 0.9. Caution never reaches an event or a clamp,
    /// only the monitor's state, so it is pinned there: a tracker at 31 m/s
    /// is nominal at 0.9 (the threshold is 31.5) but would be caution at 0.85
    /// (29.75).
    #[test]
    fn the_caution_fraction_is_the_references() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [31.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        assert_eq!(safety(&mut sim).state_of("drone-1"), "nominal");

        sim.world_mut().platform_mut("drone-1").unwrap().velocity.x = 32.0;
        sim.step().unwrap();
        assert_eq!(safety(&mut sim).state_of("drone-1"), "caution");
    }

    /// A checkpoint names only states the monitor assigns, and finite
    /// velocities.
    #[test]
    fn a_checkpoint_with_an_impossible_record_is_refused() {
        let stage = SafetyStage::new();
        let good = serde_json::json!({
            "records": {"drone-1": [2, "warning"]},
            "previous_velocities": {"drone-1": [1.0, 0.0, 0.0]},
        });
        stage.check_state(&good).expect("a real record");
        let unknown_state = serde_json::json!({
            "records": {"drone-1": [2, "panicking"]},
            "previous_velocities": {},
        });
        assert!(stage.check_state(&unknown_state).is_err());
        let infinite = serde_json::json!({
            "records": {},
            "previous_velocities": {"drone-1": [f64::MAX, 0.0, 0.0]},
        });
        stage.check_state(&infinite).expect("large but finite");
        let mut shape = serde_json::to_value(SafetyCheckpoint {
            records: BTreeMap::new(),
            previous_velocities: BTreeMap::from([("drone-1".to_string(), [f64::NAN, 0.0, 0.0])]),
        })
        .unwrap();
        // serde_json writes NaN as null, which is not a velocity at all.
        assert!(stage.check_state(&shape).is_err());
        shape["previous_velocities"]["drone-1"] = serde_json::json!([1.0, 2.0]);
        assert!(stage.check_state(&shape).is_err());
    }

    /// Restored escalation carries on: two warnings before the checkpoint,
    /// and the third violation after the restore aborts.
    #[test]
    fn a_restored_stage_escalates_from_where_it_was() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 0.0],
        )]);
        for _ in 0..2 {
            sim.world_mut().platform_mut("drone-1").unwrap().velocity.x = 500.0;
            sim.step().unwrap();
        }
        assert_eq!(safety(&mut sim).state_of("drone-1"), "warning");
        let saved = safety(&mut sim).checkpoint_state().unwrap();
        assert_eq!(
            saved["records"]["drone-1"],
            serde_json::json!([2, "warning"])
        );

        let mut fresh = SafetyStage::new();
        fresh.check_state(&saved).unwrap();
        fresh.restore_state(&saved).unwrap();
        assert_eq!(fresh.state_of("drone-1"), "warning");
        assert_eq!(
            fresh.checkpoint_state().unwrap(),
            saved,
            "the restore took everything back"
        );
    }

    #[test]
    fn clearing_drops_the_escalation_counts() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 0.0],
        )]);
        for _ in 0..DEFAULT_ABORT_FRAMES + 1 {
            sim.world_mut().platform_mut("drone-1").unwrap().velocity.x = 500.0;
            sim.step().unwrap();
        }
        let _ = safety(&mut sim).take_validations();
        assert_eq!(safety(&mut sim).state_of("drone-1"), "abort");

        safety(&mut sim).clear();
        assert_eq!(
            safety(&mut sim).state_of("drone-1"),
            "nominal",
            "a fresh epoch must not inherit the previous one's escalation"
        );
    }

    // -----------------------------------------------------------------
    // Peers, and the energy sentinel
    // -----------------------------------------------------------------

    /// Separation is checked against every *other* mobile platform. A platform
    /// is never its own peer, or every drone would violate separation with
    /// itself at zero distance.
    #[test]
    fn a_lone_platform_does_not_conflict_with_itself() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [5.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        assert!(
            safety(&mut sim).take_validations().is_empty(),
            "a single drone has no peers to be too close to"
        );
    }

    #[test]
    fn two_platforms_on_top_of_each_other_violate_separation() {
        let mut sim = rig(vec![
            node("drone-a", true, [0.0, 0.0, 100.0], [1.0, 0.0, 0.0]),
            node("drone-b", true, [0.5, 0.0, 100.0], [1.0, 0.0, 0.0]),
        ]);
        sim.step().unwrap();

        let validations = safety(&mut sim).take_validations();
        assert_eq!(validations.len(), 2, "both report it");
        assert!(validations
            .iter()
            .all(|v| v.violations.iter().any(|c| c == "min_drone_separation")));
    }

    /// The reference passes `battery_fraction` straight through, and that is
    /// **-1.0** for a platform with no battery. A mobile platform in that state
    /// violates its energy reserve every tick, for ever.
    ///
    /// It does not bite today only because the reference registers a battery
    /// for every mobile node at start-up. Recorded rather than papered over.
    #[test]
    fn a_mobile_platform_with_no_battery_violates_its_energy_reserve() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [5.0, 0.0, 0.0],
        )]);
        // A ledger with nothing registered: `fraction` returns the sentinel.
        let battery = crate::battery::BatteryStage::new();
        assert_eq!(
            battery.fraction("drone-1"),
            crate::battery::NOT_APPLICABLE_FRACTION
        );
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(battery))
            .expect("stage 1 is not a transaction boundary");
        safety(&mut sim).read_stage_one_battery();

        sim.step().unwrap();

        let validations = safety(&mut sim).take_validations();
        assert_eq!(validations.len(), 1);
        assert!(
            validations[0]
                .violations
                .iter()
                .any(|v| v == "min_energy_reserve"),
            "the -1.0 sentinel reads as an empty battery: {:?}",
            validations[0].violations
        );
    }

    /// With a real battery registered, the same platform is clean.
    #[test]
    fn a_registered_full_battery_raises_no_energy_violation() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, 100.0],
            [5.0, 0.0, 0.0],
        )]);
        let mut battery = crate::battery::BatteryStage::new();
        battery.register("drone-1", None);
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(battery))
            .expect("stage 1 is not a transaction boundary");
        safety(&mut sim).read_stage_one_battery();

        sim.step().unwrap();
        assert!(safety(&mut sim).take_validations().is_empty());
    }

    #[test]
    fn validations_are_taken_once() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [1.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        assert_eq!(safety(&mut sim).take_validations().len(), 1);
        assert!(safety(&mut sim).take_validations().is_empty());
    }

    /// Validations are **pushed**, and nothing reassigns the list, so the
    /// clear at the top of each run is the only thing stopping them piling up
    /// across ticks. A caller that polls every few ticks must get that tick's
    /// validations, not a growing backlog dated to earlier ones.
    #[test]
    fn validations_do_not_accumulate_across_ticks() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [1.0, 0.0, 0.0],
        )]);

        // Two consecutive violating ticks, taking nothing in between.
        sim.step().unwrap();
        sim.world_mut().platform_mut("drone-1").unwrap().position.z = -50.0;
        sim.step().unwrap();

        assert_eq!(
            safety(&mut sim).take_validations().len(),
            1,
            "each tick replaces the last tick's validations rather than adding to them"
        );
    }

    /// And a tick that returns early leaves none behind either.
    #[test]
    fn a_disabled_tick_does_not_replay_the_last_ticks_validations() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [1.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        // Deliberately do NOT take them.
        safety(&mut sim).set_enabled(false);
        sim.step().unwrap();
        assert!(safety(&mut sim).take_validations().is_empty());
    }

    #[test]
    fn the_validation_id_carries_the_tick_and_the_subject() {
        let mut sim = rig(vec![node(
            "drone-1",
            true,
            [0.0, 0.0, -50.0],
            [1.0, 0.0, 0.0],
        )]);
        sim.step().unwrap();
        let validations = safety(&mut sim).take_validations();
        assert_eq!(validations[0].validation_id, "safety-0-drone-1");
        assert_eq!(validations[0].subject_id, "drone-1");
        assert_eq!(validations[0].validator, VALIDATOR_NAME);
    }
}
