//! Stage 1: sample every platform's controller, then account battery.
//!
//! Until this module, [`Stage::SampleControllersAndBattery`] ran only
//! [`BatteryStage`], and platform positions were written into the world by
//! whoever drove the run -- the Rust tick did not move anything. This is the
//! stage that moves the fleet: it owns each platform's controller, samples
//! them in declaration order at the tick's timestamp, writes position,
//! velocity and timestamp into [`TickWorld`], and then runs the battery
//! accounting over those new positions, exactly the order of the reference's
//! `node_states = scenario.node_states(t)` followed by its battery loop.
//!
//! [`Stage::SampleControllersAndBattery`]: crate::stage::Stage::SampleControllersAndBattery
//! [`TickWorld`]: crate::world::TickWorld
//!
//! # The controllers the default scenario composes
//!
//! A search-then-follow drone is `Launch(ObservationTriggered(search,
//! follow))`; an inspect/follow drone is `Launch(Follow)`; a ground station is
//! a fixed point. [`PlatformController`] and [`Operational`] are exactly those
//! shapes, as closed enums rather than trait objects, so the checkpoint can
//! name every piece of state they carry and refuse one that does not match.
//!
//! # Pursuit reads the true target, and that is now explicit
//!
//! The reference's `FollowPathController` is handed the *target's trajectory
//! function* -- it steers off ground truth, not off a track. §3 makes truth a
//! capability and stage 1 is not admitted to it ([`Stage::may_read_truth`]),
//! so this stage cannot quietly do what the reference does. It takes a
//! [`PursuitOracle`] instead, installed deliberately by the run driver, named
//! for what it is, and counted in the trace on every tick it is consulted.
//! Preserving it keeps parity; naming it keeps it from becoming invisible.
//! Replacing it with fused-track pursuit is a behaviour change with its own
//! validation, not something to slip into a port.
//!
//! [`Stage::may_read_truth`]: crate::stage::Stage::may_read_truth
//!
//! # Reset means fresh construction, which is not the reference's `reset_state`
//!
//! `FollowPathController.reset_state` clears seven keys and leaves the
//! acceleration limiter's last velocity and the smoothed terrain height
//! behind. The reference only ever reaches that path when one scenario object
//! is run twice, which its CLI never does, so a second run there inherits a
//! velocity from the first. [`StageHandler::reset`] promises the state of a
//! freshly constructed handler, so this stage resets controllers to their
//! constructed state rather than calling the reference's partial reset --
//! and the differential test pins that a reset run equals a fresh one.

use std::cell::Cell;
use std::fmt;

use argusnet_types::Vector3;
use serde::{Deserialize, Serialize};

use crate::battery::BatteryStage;
use crate::follow_path::{
    FollowPathContext, FollowPathController, FollowPathState, MissionZone, RoutePlanner,
    VisibilitySampler,
};
use crate::frontier::{ExplorerState, FrontierExplorer};
use crate::motion::{BoundsXy, ObstacleSampler, TerrainSampler};
use crate::observation_follow::{
    AcceptedByNodeTarget, EngagementState, ObservationTriggeredFollow,
};
use crate::simulation::{decode_stage_state, StageHandler, StageStateError, TickContext};
use crate::stage::{StageError, StageOutcome};
use crate::trajectory::{LaunchController, WeatherContext, WeatherSampler};
use crate::waypoint_path::LoopedWaypointPath;

/// Where a target truly is -- the legacy pursuit capability. See the module
/// comment for why this exists as a named trait rather than a truth read.
pub trait PursuitOracle: Send {
    /// Position and velocity of `target_id` at `timestamp_s`, or `None` for a
    /// target the oracle does not know.
    fn target_state(&self, target_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])>;

    /// A committed tick's fused tracks, handed over by the run driver after
    /// the tick (fusion runs after stage 1), so the next tick pursues them.
    /// Given between ticks rather than read from the world inside stage 1,
    /// so the oracle's state at a checkpoint already holds them. The legacy
    /// truth oracle ignores them.
    fn observe_tracks(&mut self, _tracks: &[argusnet_types::TrackState]) {}

    /// Whether answers are true target states. The stage note counts truth
    /// reads as such; an oracle that does not read truth writes its own note.
    fn reads_truth(&self) -> bool {
        true
    }

    /// The oracle's note for this tick, if it has one; clears its counts.
    fn take_note(&mut self) -> Option<String> {
        None
    }

    /// Back to the state at construction.
    fn reset(&mut self) {}

    /// Oracle state for stage 1's checkpoint; `None` keeps the checkpoint
    /// exactly as it was before oracles had state.
    fn checkpoint(&self) -> Option<serde_json::Value> {
        None
    }

    fn restore(&mut self, _state: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }
}

/// The world the controllers are lent each tick.
///
/// Owned capabilities rather than borrowed ones: the stage outlives any one
/// tick, and it is `Send` because the simulation is.
pub struct ControllerEnvironment {
    pub terrain: Box<dyn TerrainSampler + Send>,
    /// `None` means no obstacle layer at all, which is a different read
    /// pattern (no probe) from an empty one.
    pub obstacles: Option<Box<dyn ObstacleSampler + Send>>,
    pub visibility: Box<dyn VisibilitySampler + Send>,
    pub planner: Option<Box<dyn RoutePlanner + Send>>,
    pub zones: Vec<MissionZone>,
    /// `None` makes the launch path's weather adjustment a passthrough.
    pub weather: Option<Box<dyn WeatherSampler + Send>>,
    pub map_bounds_m: Option<BoundsXy>,
    pub pursuit: Option<Box<dyn PursuitOracle>>,
    /// The platforms' own map, for explorers. `None` outside a no-prior run.
    pub belief: Option<crate::belief_world::BeliefHandle>,
}

/// What a platform flies once it is airborne.
// One value per platform, built once at scenario construction and never moved
// per tick, so the size spread between variants costs nothing that boxing
// would buy back -- it would only add an indirection to every constructor.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Operational {
    /// A search lane, flown forever. Stateless.
    Search(LoopedWaypointPath),
    /// Orbit a target from launch.
    Follow {
        target_id: String,
        controller: FollowPathController,
    },
    /// Search until this node's sensor accepts its target, then follow.
    SearchThenFollow(ObservationTriggeredFollow<LoopedWaypointPath, FollowPathController>),
    /// A search lane turned explorer: fly to the frontier of the platforms'
    /// own map (a no-prior run; see [`crate::frontier`]).
    Explore(FrontierExplorer),
}

impl Operational {
    /// What the platform is doing, for the assignment history
    /// ([`crate::assignments`]): `search`, `follow:<target>` or `explore`. A
    /// search-then-follow platform searches until its sensor engages.
    pub fn task(&self) -> String {
        match self {
            Operational::Search(_) => "search".to_string(),
            Operational::Follow { target_id, .. } => format!("follow:{target_id}"),
            Operational::SearchThenFollow(search) if search.is_engaged() => {
                format!("follow:{}", search.preferred_target_id())
            }
            Operational::SearchThenFollow(_) => "search".to_string(),
            Operational::Explore(_) => "explore".to_string(),
        }
    }
}

/// The launch path's two tuning values; the rest of its weather context is
/// the environment's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaunchProfile {
    pub min_operational_agl_m: f64,
    pub wind_scale: f64,
    /// The reference's `run_simulation` launches every launchable drone at
    /// `t = 0` against its assigned target, *after* resetting runtime state.
    /// Carried here so a reset run relaunches the way a fresh one does.
    pub auto_launch_at_s: Option<f64>,
}

/// One platform's controller.
#[derive(Debug, Clone)]
pub enum PlatformController {
    /// A ground station: fixed position, zero velocity.
    Static { position_m: [f64; 3] },
    /// Airborne from the start, no launch phase.
    Free(Operational),
    /// Parked at a station until launched, then climbing, then operational.
    Launched {
        controller: LaunchController<Operational>,
        profile: LaunchProfile,
        /// The target id passed to the automatic launch.
        launch_target_id: String,
    },
}

impl PlatformController {
    /// Apply the construction-time launch, if any. Used at install and reset.
    fn arm(&mut self) {
        if let PlatformController::Launched {
            controller,
            profile,
            launch_target_id,
        } = self
        {
            if let Some(at_s) = profile.auto_launch_at_s {
                controller.trigger_launch(at_s, launch_target_id.clone());
            }
        }
    }

    fn reset_to_constructed(&mut self) {
        match self {
            PlatformController::Static { .. } => {}
            PlatformController::Free(operational) => operational.reset_to_constructed(),
            PlatformController::Launched { controller, .. } => {
                controller.reset_launch();
                controller.operational_mut().reset_to_constructed();
            }
        }
        self.arm();
    }
}

impl Operational {
    fn reset_to_constructed(&mut self) {
        match self {
            Operational::Search(_) => {}
            Operational::Explore(explorer) => explorer.restore(ExplorerState::default()),
            Operational::Follow { controller, .. } => {
                controller.restore_state(FollowPathState::default())
            }
            Operational::SearchThenFollow(wrapper) => {
                wrapper.restore_engagement(EngagementState::default());
                wrapper
                    .follow_mut()
                    .restore_state(FollowPathState::default());
            }
        }
    }

    fn state(&self) -> OperationalState {
        match self {
            Operational::Search(_) => OperationalState::Search,
            Operational::Explore(explorer) => OperationalState::Explore(explorer.state().clone()),
            Operational::Follow { controller, .. } => {
                OperationalState::Follow(controller.state().clone())
            }
            Operational::SearchThenFollow(wrapper) => OperationalState::SearchThenFollow {
                engagement: wrapper.engagement().clone(),
                follow: wrapper.follow().state().clone(),
            },
        }
    }

    fn accepts(&self, state: &OperationalState) -> bool {
        matches!(
            (self, state),
            (Operational::Search(_), OperationalState::Search)
                | (Operational::Explore(_), OperationalState::Explore(_))
                | (Operational::Follow { .. }, OperationalState::Follow(_))
                | (
                    Operational::SearchThenFollow(_),
                    OperationalState::SearchThenFollow { .. }
                )
        )
    }

    fn restore(&mut self, state: OperationalState) {
        match (self, state) {
            (Operational::Explore(explorer), OperationalState::Explore(saved)) => {
                explorer.restore(saved)
            }
            (Operational::Follow { controller, .. }, OperationalState::Follow(follow)) => {
                controller.restore_state(follow)
            }
            (
                Operational::SearchThenFollow(wrapper),
                OperationalState::SearchThenFollow { engagement, follow },
            ) => {
                wrapper.restore_engagement(engagement);
                wrapper.follow_mut().restore_state(follow);
            }
            _ => {}
        }
    }
}

/// The checkpointed half of an [`Operational`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum OperationalState {
    Search,
    Explore(ExplorerState),
    Follow(FollowPathState),
    SearchThenFollow {
        engagement: EngagementState,
        follow: FollowPathState,
    },
}

/// The checkpointed half of a [`PlatformController`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum ControllerState {
    Static,
    Free(OperationalState),
    Launched {
        launched_at_s: Option<f64>,
        launch_target_id: Option<String>,
        operational: OperationalState,
    },
}

/// Everything stage 1 checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StageOneState {
    /// Declaration order, which is also sampling order.
    controllers: Vec<(String, ControllerState)>,
    /// The battery stage's own checkpoint, when one is chained.
    battery: Option<serde_json::Value>,
    /// The pursuit oracle's state, when it has any (fused-track pursuit).
    /// Absent from every legacy checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pursuit: Option<serde_json::Value>,
}

fn controller_state(controller: &PlatformController) -> ControllerState {
    match controller {
        PlatformController::Static { .. } => ControllerState::Static,
        PlatformController::Free(operational) => ControllerState::Free(operational.state()),
        PlatformController::Launched { controller, .. } => ControllerState::Launched {
            launched_at_s: controller.launched_at_s(),
            launch_target_id: controller.launch_target_id().map(str::to_string),
            operational: controller.operational().state(),
        },
    }
}

fn controller_accepts(controller: &PlatformController, state: &ControllerState) -> bool {
    match (controller, state) {
        (PlatformController::Static { .. }, ControllerState::Static) => true,
        (PlatformController::Free(operational), ControllerState::Free(saved)) => {
            operational.accepts(saved)
        }
        (
            PlatformController::Launched { controller, .. },
            ControllerState::Launched {
                launched_at_s,
                operational,
                ..
            },
        ) => {
            launched_at_s.is_none_or(f64::is_finite)
                && controller.operational().accepts(operational)
        }
        _ => false,
    }
}

#[derive(Default)]
struct State {
    controllers: Vec<(String, PlatformController)>,
    environment: Option<ControllerEnvironment>,
    /// This tick's controller output per platform, keyed by the sample
    /// time's bits: the *nominal* pose, before overrides and deconfliction
    /// move the platform. Transient -- rebuilt every tick by the stage that
    /// reads it first -- so it is not checkpoint state.
    nominal: std::collections::BTreeMap<String, (u64, [f64; 3], [f64; 3])>,
}

/// Which capabilities a [`ControllerEnvironment`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvironmentSummary {
    pub has_obstacles: bool,
    pub has_planner: bool,
    pub has_weather: bool,
    pub has_pursuit: bool,
    pub zone_count: usize,
    pub has_map_bounds: bool,
}

impl ControllerStage {
    /// Give `node_id` a controller. Installation order is sampling order.
    /// Re-installing an id replaces it in place, keeping its position.
    pub fn install(&mut self, node_id: impl Into<String>, mut controller: PlatformController) {
        controller.arm();
        let node_id = node_id.into();
        let shared = &mut self.state;
        match shared.controllers.iter_mut().find(|(id, _)| *id == node_id) {
            Some((_, slot)) => *slot = controller,
            None => shared.controllers.push((node_id, controller)),
        }
    }

    pub fn set_environment(&mut self, environment: ControllerEnvironment) {
        self.state.environment = Some(environment);
    }

    /// The installed environment, to change what the controllers read.
    pub fn environment_mut(&mut self) -> Option<&mut ControllerEnvironment> {
        self.state.environment.as_mut()
    }

    /// Hand a committed tick's fused tracks to the pursuit oracle. See
    /// [`PursuitOracle::observe_tracks`].
    pub fn observe_tracks(&mut self, tracks: &[argusnet_types::TrackState]) {
        if let Some(pursuit) = self.pursuit_mut() {
            pursuit.observe_tracks(tracks);
        }
    }

    fn pursuit_mut(&mut self) -> Option<&mut Box<dyn PursuitOracle>> {
        self.state
            .environment
            .as_mut()
            .and_then(|environment| environment.pursuit.as_mut())
    }

    /// Keep sampled positions out of the true ground and structures.
    pub fn set_physical_guard(&mut self, guard: crate::belief_world::PhysicalGuard) {
        self.guard = Some(guard);
    }

    /// Launch a parked platform. Ignored for one already launched, as in the
    /// reference.
    pub fn trigger_launch(&mut self, node_id: &str, timestamp_s: f64, target_id: &str) -> bool {
        let shared = &mut self.state;
        match shared.controllers.iter_mut().find(|(id, _)| id == node_id) {
            Some((_, PlatformController::Launched { controller, .. })) => {
                controller.trigger_launch(timestamp_s, target_id);
                true
            }
            _ => false,
        }
    }

    /// Feed one tick's accepted-observation counts to every search-then-follow
    /// controller. Returns the ids that engaged on this call.
    pub fn observe(&mut self, accepted: &AcceptedByNodeTarget) -> Vec<String> {
        let shared = &mut self.state;
        let mut engaged = Vec::new();
        for (node_id, controller) in shared.controllers.iter_mut() {
            let operational = match controller {
                PlatformController::Free(operational) => operational,
                PlatformController::Launched { controller, .. } => controller.operational_mut(),
                PlatformController::Static { .. } => continue,
            };
            if let Operational::SearchThenFollow(wrapper) = operational {
                if wrapper.update_from_observations(accepted) {
                    engaged.push(node_id.clone());
                }
            }
        }
        engaged
    }

    /// Turn every pure search lane into a frontier explorer that starts from
    /// it. Search-then-follow controllers keep their lanes. Returns how many
    /// were turned.
    pub fn explore_search_lanes(&mut self) -> usize {
        let mut turned = 0;
        for (_, controller) in self.state.controllers.iter_mut() {
            let operational = match controller {
                PlatformController::Free(operational) => operational,
                PlatformController::Launched { controller, .. } => controller.operational_mut(),
                PlatformController::Static { .. } => continue,
            };
            if let Operational::Search(lane) = operational {
                *operational = Operational::Explore(FrontierExplorer::new(lane.clone()));
                turned += 1;
            }
        }
        turned
    }

    /// How many controllers are frontier explorers.
    pub fn explorer_count(&self) -> usize {
        self.state
            .controllers
            .iter()
            .filter(|(_, controller)| match controller {
                PlatformController::Free(operational) => {
                    matches!(operational, Operational::Explore(_))
                }
                PlatformController::Launched { controller, .. } => {
                    matches!(controller.operational(), Operational::Explore(_))
                }
                PlatformController::Static { .. } => false,
            })
            .count()
    }

    /// What the controllers have been lent, for diagnostics and wiring tests.
    /// `None` before an environment is installed.
    pub fn environment_summary(&self) -> Option<EnvironmentSummary> {
        let shared = &self.state;
        shared.environment.as_ref().map(|env| EnvironmentSummary {
            has_obstacles: env.obstacles.is_some(),
            has_planner: env.planner.is_some(),
            has_weather: env.weather.is_some(),
            has_pursuit: env.pursuit.is_some(),
            zone_count: env.zones.len(),
            has_map_bounds: env.map_bounds_m.is_some(),
        })
    }

    /// Whether `node_id`'s search-then-follow controller is following.
    pub fn is_engaged(&self, node_id: &str) -> Option<bool> {
        let shared = &self.state;
        let (_, controller) = shared.controllers.iter().find(|(id, _)| id == node_id)?;
        let operational = match controller {
            PlatformController::Free(operational) => operational,
            PlatformController::Launched { controller, .. } => controller.operational(),
            PlatformController::Static { .. } => return None,
        };
        match operational {
            Operational::SearchThenFollow(wrapper) => Some(wrapper.is_engaged()),
            _ => None,
        }
    }
}

/// An obstacle layer that contains nothing, for the follow controller, which
/// always probes. Identical results to "no layer" on its path.
struct NoObstacles;

impl ObstacleSampler for NoObstacles {
    fn push_clear(&self, _point_m: [f64; 3], _margin_m: f64) -> Option<[f64; 2]> {
        None
    }

    fn collides(&self, _point_m: [f64; 3]) -> bool {
        false
    }
}

/// Per-tick bookkeeping the sampling closures cannot return through their
/// fixed signatures: pursuit reads, and the first failure.
#[derive(Default)]
struct SampleLog {
    pursuit_reads: Cell<u64>,
    failure: std::cell::RefCell<Option<String>>,
    /// This tick's map, for explorers: one snapshot for the whole stage.
    belief: Option<std::sync::Arc<crate::belief_world::BeliefSnapshot>>,
    /// The explorers' goals: gathered before sampling, updated as each
    /// explorer re-goals, so later explorers avoid earlier ones' choices.
    claims: std::cell::RefCell<Vec<[f64; 2]>>,
    /// The snapshot's frontier cells, listed once a tick when first needed.
    frontier: std::cell::OnceCell<Vec<[f64; 2]>>,
}

impl SampleLog {
    fn fail(&self, detail: String) {
        self.failure.borrow_mut().get_or_insert(detail);
    }
}

fn sample_follow(
    controller: &mut FollowPathController,
    target_id: &str,
    timestamp_s: f64,
    environment: &ControllerEnvironment,
    log: &SampleLog,
) -> ([f64; 3], [f64; 3]) {
    let Some(pursuit) = environment.pursuit.as_deref() else {
        log.fail(format!(
            "follow controller for {target_id} has no pursuit oracle installed"
        ));
        return ([0.0; 3], [0.0; 3]);
    };
    let mut target = |t: f64| {
        log.pursuit_reads.set(log.pursuit_reads.get() + 1);
        pursuit.target_state(target_id, t).unwrap_or_else(|| {
            log.fail(format!("pursuit oracle does not know target {target_id}"));
            ([0.0; 3], [0.0; 3])
        })
    };
    let obstacles: &dyn ObstacleSampler = match environment.obstacles.as_deref() {
        Some(obstacles) => obstacles,
        None => &NoObstacles,
    };
    let context = FollowPathContext {
        terrain: environment.terrain.as_ref(),
        obstacles,
        visibility: environment.visibility.as_ref(),
        planner: environment
            .planner
            .as_deref()
            .map(|planner| planner as &dyn RoutePlanner),
        zones: &environment.zones,
    };
    controller.sample(timestamp_s, &mut target, &context)
}

fn sample_operational(
    operational: &mut Operational,
    timestamp_s: f64,
    environment: &ControllerEnvironment,
    log: &SampleLog,
) -> ([f64; 3], [f64; 3]) {
    let terrain = environment.terrain.as_ref();
    let obstacles = environment
        .obstacles
        .as_deref()
        .map(|obstacles| obstacles as &dyn ObstacleSampler);
    match operational {
        Operational::Search(path) => path.sample(timestamp_s, terrain, obstacles),
        Operational::Explore(explorer) => explorer.sample(
            timestamp_s,
            terrain,
            obstacles,
            log.belief.as_deref(),
            &log.frontier,
            &mut log.claims.borrow_mut(),
        ),
        Operational::Follow {
            target_id,
            controller,
        } => sample_follow(controller, target_id, timestamp_s, environment, log),
        Operational::SearchThenFollow(wrapper) => {
            let target_id = wrapper.preferred_target_id().to_string();
            wrapper.sample(
                timestamp_s,
                |search, t| search.sample(t, terrain, obstacles),
                |follow, t| sample_follow(follow, &target_id, t, environment, log),
            )
        }
    }
}

fn sample_platform(
    controller: &mut PlatformController,
    timestamp_s: f64,
    environment: &ControllerEnvironment,
    log: &SampleLog,
) -> ([f64; 3], [f64; 3]) {
    match controller {
        PlatformController::Static { position_m } => (*position_m, [0.0; 3]),
        PlatformController::Free(operational) => {
            sample_operational(operational, timestamp_s, environment, log)
        }
        PlatformController::Launched {
            controller,
            profile,
            ..
        } => {
            let weather = WeatherContext {
                weather: environment
                    .weather
                    .as_deref()
                    .map(|weather| weather as &dyn WeatherSampler),
                terrain: Some(environment.terrain.as_ref()),
                bounds: environment.map_bounds_m,
                min_agl_m: profile.min_operational_agl_m,
                wind_scale: profile.wind_scale,
                // The reference's launch path passes no dt, so its drift is
                // always integrated over one second. Preserved; see
                // `LaunchController::sample`.
                dt_s: WeatherContext::DEFAULT_DT_S,
            };
            controller.sample_with(timestamp_s, &weather, |operational, t| {
                sample_operational(operational, t, environment, log)
            })
        }
    }
}

fn guard_nominal(
    nominal: &mut std::collections::BTreeMap<String, (u64, [f64; 3], [f64; 3])>,
    node_id: &str,
    timestamp_s: f64,
    position: [f64; 3],
    velocity: [f64; 3],
) {
    nominal.insert(
        node_id.to_string(),
        (timestamp_s.to_bits(), position, velocity),
    );
}

impl ControllerStage {
    /// This tick's nominal pose for `node_id`, if it was sampled at exactly
    /// `timestamp_s`. A request for any other time is refused rather than
    /// answered with a stale pose.
    pub fn nominal_pose(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        let (bits, position, velocity) = self.state.nominal.get(node_id)?;
        (*bits == timestamp_s.to_bits()).then_some((*position, *velocity))
    }
}

/// [`Stage::SampleControllersAndBattery`] as a [`StageHandler`].
///
/// [`Stage::SampleControllersAndBattery`]: crate::stage::Stage::SampleControllersAndBattery
pub struct ControllerStage {
    state: State,
    battery: Option<BatteryStage>,
    /// Physics, in a run whose controllers plan against a belief rather than
    /// the true world (`world_knowledge: none`): a sampled position inside
    /// the ground or a structure is put back and recorded. `None` in every
    /// prior run, which then samples exactly as before.
    guard: Option<crate::belief_world::PhysicalGuard>,
}

impl ControllerStage {
    /// Each mobile platform's own task, in fleet order: what it flies when no
    /// recall overrides it. Ground stations have none.
    pub fn tasks(&self) -> Vec<(String, String)> {
        self.state
            .controllers
            .iter()
            .filter_map(|(id, controller)| {
                let operational = match controller {
                    PlatformController::Static { .. } => return None,
                    PlatformController::Free(operational) => operational,
                    PlatformController::Launched { controller, .. } => controller.operational(),
                };
                Some((id.clone(), operational.task()))
            })
            .collect()
    }
}

impl fmt::Debug for ControllerStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControllerStage")
            .field(
                "controllers",
                &self
                    .state
                    .controllers
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>(),
            )
            .field("has_environment", &self.state.environment.is_some())
            .field("battery", &self.battery)
            .finish()
    }
}

impl Default for ControllerStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ControllerStage {
    pub fn new() -> Self {
        Self {
            state: State::default(),
            battery: None,
            guard: None,
        }
    }

    /// Run `battery` after the controllers, in the same stage, as the
    /// reference does. Stage 1 has one handler slot; this is how both fit.
    pub fn with_battery(mut self, battery: BatteryStage) -> Self {
        self.battery = Some(battery);
        self
    }

    /// The battery stage chained after the controllers, if any.
    pub fn battery(&self) -> Option<&BatteryStage> {
        self.battery.as_ref()
    }

    pub fn battery_mut(&mut self) -> Option<&mut BatteryStage> {
        self.battery.as_mut()
    }

    fn sample_all(&mut self, context: &mut TickContext<'_>) -> Result<bool, StageError> {
        let guard = self.guard.as_ref();
        let State {
            controllers,
            environment: environment_slot,
            nominal,
        } = &mut self.state;
        nominal.clear();
        if controllers.is_empty() {
            return Ok(false);
        }
        let environment = environment_slot.as_ref().ok_or_else(|| {
            StageError::Invariant("controllers are installed but no environment is".into())
        })?;
        let claims = controllers
            .iter()
            .filter_map(|(_, controller)| match controller {
                PlatformController::Free(Operational::Explore(explorer)) => explorer.goal(),
                PlatformController::Launched { controller, .. } => match controller.operational() {
                    Operational::Explore(explorer) => explorer.goal(),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let log = SampleLog {
            belief: environment.belief.as_ref().map(|handle| handle.current()),
            claims: std::cell::RefCell::new(claims),
            ..SampleLog::default()
        };
        let timestamp_s = context.timestamp_s;
        for (node_id, controller) in controllers.iter_mut() {
            let (position, velocity) = sample_platform(controller, timestamp_s, environment, &log);
            if let Some(detail) = log.failure.borrow_mut().take() {
                return Err(StageError::Invariant(format!("{node_id}: {detail}")));
            }
            let position = match guard {
                Some(guard) => guard.enforce(node_id, timestamp_s, position),
                None => position,
            };
            let platform = context.world.platform_mut(node_id).ok_or_else(|| {
                StageError::Invariant(format!("controller {node_id} has no platform in the world"))
            })?;
            guard_nominal(nominal, node_id, timestamp_s, position, velocity);
            platform.position = Vector3::new(position[0], position[1], position[2]);
            platform.velocity = Vector3::new(velocity[0], velocity[1], velocity[2]);
            platform.timestamp_s = timestamp_s;
        }
        let reads = log.pursuit_reads.get();
        let pursuit = environment_slot
            .as_mut()
            .and_then(|environment| environment.pursuit.as_mut());
        match pursuit {
            Some(pursuit) if !pursuit.reads_truth() => {
                if let Some(note) = pursuit.take_note() {
                    context.note(note);
                }
            }
            _ if reads > 0 => {
                context.note(format!(
                    "legacy pursuit read {reads} true target states (truth-fed follow controllers)"
                ));
            }
            _ => {}
        }
        Ok(true)
    }
}

impl StageHandler for ControllerStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let moved = self.sample_all(context)?;
        let battery = match self.battery.as_mut() {
            Some(battery) => battery.run(context)?,
            None => StageOutcome::NoWork,
        };
        Ok(if moved || battery == StageOutcome::Ran {
            StageOutcome::Ran
        } else {
            StageOutcome::NoWork
        })
    }

    fn reset(&mut self) {
        let shared = &mut self.state;
        shared.nominal.clear();
        for (_, controller) in shared.controllers.iter_mut() {
            controller.reset_to_constructed();
        }
        if let Some(battery) = self.battery.as_mut() {
            battery.reset();
        }
        if let Some(pursuit) = self.pursuit_mut() {
            pursuit.reset();
        }
    }

    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let state = StageOneState {
            controllers: self
                .state
                .controllers
                .iter()
                .map(|(id, controller)| (id.clone(), controller_state(controller)))
                .collect(),
            battery: self
                .battery
                .as_ref()
                .and_then(StageHandler::checkpoint_state),
            pursuit: self
                .state
                .environment
                .as_ref()
                .and_then(|environment| environment.pursuit.as_ref())
                .and_then(|pursuit| pursuit.checkpoint()),
        };
        Some(serde_json::to_value(state).expect("stage 1 state serialises"))
    }

    fn check_state(&self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: StageOneState = decode_stage_state(state)?;
        {
            let shared = &self.state;
            if decoded.controllers.len() != shared.controllers.len() {
                return Err(StageStateError::Malformed {
                    detail: format!(
                        "checkpoint has {} controllers, this run has {}",
                        decoded.controllers.len(),
                        shared.controllers.len()
                    ),
                });
            }
            for ((saved_id, saved), (id, controller)) in
                decoded.controllers.iter().zip(shared.controllers.iter())
            {
                if saved_id != id {
                    return Err(StageStateError::Malformed {
                        detail: format!("controller order differs: {saved_id} where {id} is"),
                    });
                }
                if !controller_accepts(controller, saved) {
                    return Err(StageStateError::Malformed {
                        detail: format!("controller {id} is not the kind the checkpoint describes"),
                    });
                }
            }
        }
        let has_pursuit_state = self
            .state
            .environment
            .as_ref()
            .and_then(|environment| environment.pursuit.as_ref())
            .is_some_and(|pursuit| pursuit.checkpoint().is_some());
        if has_pursuit_state != decoded.pursuit.is_some() {
            return Err(StageStateError::Malformed {
                detail: "the checkpoint and this run disagree on how follow controllers pursue \
                         (truth or fused tracks)"
                    .into(),
            });
        }
        match (&self.battery, &decoded.battery) {
            (Some(battery), Some(saved)) => battery.check_state(saved),
            (None, None) => Ok(()),
            (Some(_), None) | (None, Some(_)) => Err(StageStateError::Malformed {
                detail: "battery chaining differs between checkpoint and run".into(),
            }),
        }
    }

    fn restore_state(&mut self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: StageOneState = decode_stage_state(state)?;
        {
            let shared = &mut self.state;
            for ((_, saved), (_, controller)) in decoded
                .controllers
                .into_iter()
                .zip(shared.controllers.iter_mut())
            {
                match (controller, saved) {
                    (PlatformController::Free(operational), ControllerState::Free(saved)) => {
                        operational.restore(saved)
                    }
                    (
                        PlatformController::Launched { controller, .. },
                        ControllerState::Launched {
                            launched_at_s,
                            launch_target_id,
                            operational,
                        },
                    ) => {
                        controller.restore_launch(launched_at_s, launch_target_id);
                        controller.operational_mut().restore(operational);
                    }
                    _ => {}
                }
            }
        }
        if let (Some(battery), Some(saved)) = (self.battery.as_mut(), decoded.battery.as_ref()) {
            battery.restore_state(saved)?;
        }
        if let (Some(pursuit), Some(saved)) = (self.pursuit_mut(), decoded.pursuit.as_ref()) {
            pursuit
                .restore(saved)
                .map_err(|detail| StageStateError::Malformed { detail })?;
        }
        Ok(())
    }
}
