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
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_types::Vector3;
use serde::{Deserialize, Serialize};

use crate::battery::BatteryStage;
use crate::follow_path::{
    FollowPathContext, FollowPathController, FollowPathState, MissionZone, RoutePlanner,
    VisibilitySampler,
};
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
                | (Operational::Follow { .. }, OperationalState::Follow(_))
                | (
                    Operational::SearchThenFollow(_),
                    OperationalState::SearchThenFollow { .. }
                )
        )
    }

    fn restore(&mut self, state: OperationalState) {
        match (self, state) {
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
struct Shared {
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

/// A handle onto stage 1, kept by the run driver and by stage 6.
#[derive(Clone, Default)]
pub struct ControllerLink {
    inner: Arc<Mutex<Shared>>,
}

impl fmt::Debug for ControllerLink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shared = self.lock();
        formatter
            .debug_struct("ControllerLink")
            .field(
                "controllers",
                &shared
                    .controllers
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>(),
            )
            .field("has_environment", &shared.environment.is_some())
            .finish()
    }
}

impl ControllerLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Give `node_id` a controller. Installation order is sampling order.
    /// Re-installing an id replaces it in place, keeping its position.
    pub fn install(&self, node_id: impl Into<String>, mut controller: PlatformController) {
        controller.arm();
        let node_id = node_id.into();
        let mut shared = self.lock();
        match shared.controllers.iter_mut().find(|(id, _)| *id == node_id) {
            Some((_, slot)) => *slot = controller,
            None => shared.controllers.push((node_id, controller)),
        }
    }

    pub fn set_environment(&self, environment: ControllerEnvironment) {
        self.lock().environment = Some(environment);
    }

    /// Launch a parked platform. Ignored for one already launched, as in the
    /// reference.
    pub fn trigger_launch(&self, node_id: &str, timestamp_s: f64, target_id: &str) -> bool {
        let mut shared = self.lock();
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
    pub fn observe(&self, accepted: &AcceptedByNodeTarget) -> Vec<String> {
        let mut shared = self.lock();
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

    /// What the controllers have been lent, for diagnostics and wiring tests.
    /// `None` before an environment is installed.
    pub fn environment_summary(&self) -> Option<EnvironmentSummary> {
        let shared = self.lock();
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
        let shared = self.lock();
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

impl ControllerLink {
    /// This tick's nominal pose for `node_id`, if it was sampled at exactly
    /// `timestamp_s`. A request for any other time is refused rather than
    /// answered with a stale pose.
    pub fn nominal_pose(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        let shared = self.lock();
        let (bits, position, velocity) = shared.nominal.get(node_id)?;
        (*bits == timestamp_s.to_bits()).then_some((*position, *velocity))
    }
}

impl crate::observe::SensorPoseSource for ControllerLink {
    fn pose(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        self.nominal_pose(node_id, timestamp_s)
    }
}

impl crate::vio::NominalTrajectory for ControllerLink {
    fn sample(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        self.nominal_pose(node_id, timestamp_s)
    }
}

/// [`Stage::SampleControllersAndBattery`] as a [`StageHandler`].
///
/// [`Stage::SampleControllersAndBattery`]: crate::stage::Stage::SampleControllersAndBattery
pub struct ControllerStage {
    link: ControllerLink,
    battery: Option<BatteryStage>,
}

impl fmt::Debug for ControllerStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControllerStage")
            .field("link", &self.link)
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
            link: ControllerLink::default(),
            battery: None,
        }
    }

    /// Run `battery` after the controllers, in the same stage, as the
    /// reference does. Stage 1 has one handler slot; this is how both fit.
    pub fn with_battery(mut self, battery: BatteryStage) -> Self {
        self.battery = Some(battery);
        self
    }

    pub fn link(&self) -> ControllerLink {
        self.link.clone()
    }

    fn sample_all(&mut self, context: &mut TickContext<'_>) -> Result<bool, StageError> {
        let mut guard = self.link.lock();
        let Shared {
            controllers,
            environment,
            nominal,
        } = &mut *guard;
        nominal.clear();
        if controllers.is_empty() {
            return Ok(false);
        }
        let environment = environment.as_ref().ok_or_else(|| {
            StageError::Invariant("controllers are installed but no environment is".into())
        })?;
        let log = SampleLog::default();
        let timestamp_s = context.timestamp_s;
        for (node_id, controller) in controllers.iter_mut() {
            let (position, velocity) = sample_platform(controller, timestamp_s, environment, &log);
            if let Some(detail) = log.failure.borrow_mut().take() {
                return Err(StageError::Invariant(format!("{node_id}: {detail}")));
            }
            let platform = context.world.platform_mut(node_id).ok_or_else(|| {
                StageError::Invariant(format!("controller {node_id} has no platform in the world"))
            })?;
            guard_nominal(nominal, node_id, timestamp_s, position, velocity);
            platform.position = Vector3::new(position[0], position[1], position[2]);
            platform.velocity = Vector3::new(velocity[0], velocity[1], velocity[2]);
            platform.timestamp_s = timestamp_s;
        }
        let reads = log.pursuit_reads.get();
        drop(guard);
        if reads > 0 {
            context.note(format!(
                "legacy pursuit read {reads} true target states (truth-fed follow controllers)"
            ));
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
        let mut shared = self.link.lock();
        shared.nominal.clear();
        for (_, controller) in shared.controllers.iter_mut() {
            controller.reset_to_constructed();
        }
        drop(shared);
        if let Some(battery) = self.battery.as_mut() {
            battery.reset();
        }
    }

    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let state = StageOneState {
            controllers: self
                .link
                .lock()
                .controllers
                .iter()
                .map(|(id, controller)| (id.clone(), controller_state(controller)))
                .collect(),
            battery: self
                .battery
                .as_ref()
                .and_then(StageHandler::checkpoint_state),
        };
        Some(serde_json::to_value(state).expect("stage 1 state serialises"))
    }

    fn check_state(&self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: StageOneState = decode_stage_state(state)?;
        {
            let shared = self.link.lock();
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
            let mut shared = self.link.lock();
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
        Ok(())
    }
}
