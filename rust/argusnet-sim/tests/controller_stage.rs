//! Stage 1 end to end: controllers move the fleet, and a resumed run is the
//! same run.
//!
//! The load-bearing tests here are **differentials**. P4.1's exit criterion is
//! not "the tick counter comes back" but that uninterrupted execution equals
//! checkpoint → JSON → fresh-process resume for actual outputs. So one mission
//! is flown straight through, and then flown again with a checkpoint taken at
//! each of four interesting moments -- mid-climb, mid-search, the tick an
//! observation engaged the follow controller but before it was flown, and
//! deep in the follow -- serialised to text, and restored into a *different*
//! `Simulation` that was first sent down a wrong history. Every platform's
//! position, velocity and battery fraction must then agree bit for bit on
//! every remaining tick.
//!
//! Engagement comes through the real pipeline: stage 6's sensor accepts the
//! target, the session hands the counts to stage 1, and the next tick's stage 1 follows. No test
//! shortcut calls `observe` directly.

use argusnet_sensing::SensorErrorConfig;
use argusnet_sim::battery::BatteryStage;
use argusnet_sim::config::SimulationConfig;
use argusnet_sim::controller::{
    ControllerEnvironment, ControllerStage, LaunchProfile, Operational, PlatformController,
    PursuitOracle,
};
use argusnet_sim::follow_path::{FollowPathConfig, FollowPathController, VisibilitySampler};
use argusnet_sim::motion::{BoundsXy, TerrainSampler};
use argusnet_sim::observation_follow::ObservationTriggeredFollow;
use argusnet_sim::observe::{ObserveStage, SensorSpec};
use argusnet_sim::simulation::{Handoff, Simulation};
use argusnet_sim::snapshot::{RestoreError, SimulationCheckpoint};
use argusnet_sim::stage::Stage;
use argusnet_sim::trajectory::{LaunchController, WeatherSampler};
use argusnet_sim::waypoint_path::{LoopedWaypointPath, WaypointPathConfig};
use argusnet_types::{NodeState, TruthState, Vector3};

// Half-second ticks, not one: the launch path integrates wind drift over a
// hard-coded 1.0 s whatever the timestep (the reference's quirk), and at
// dt = 1.0 a port that used the real dt would be indistinguishable.
const STEPS: u64 = 200;
const DT_S: f64 = 0.5;

fn ground(x_m: f64, y_m: f64) -> f64 {
    60.0 + 0.05 * x_m + 0.03 * y_m
}

struct Plane;

impl TerrainSampler for Plane {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        ground(x_m, y_m)
    }

    fn gradient_at(&self, _x_m: f64, _y_m: f64, _delta_m: Option<f64>) -> [f64; 2] {
        [0.05, 0.03]
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        z_m.max(ground(x_m, y_m) + min_agl_m.max(0.0))
    }
}

struct ClearSky;

impl VisibilitySampler for ClearSky {
    fn is_visible(&self, _observer_m: [f64; 3], _target_m: [f64; 3]) -> bool {
        true
    }
}

/// Gusty enough that the launch path's drift is not a constant.
struct Breeze;

impl WeatherSampler for Breeze {
    fn flight_speed_penalty(&self, _altitude_m: f64) -> f64 {
        0.93
    }

    fn wind_at(&self, altitude_m: f64, time_s: f64) -> [f64; 3] {
        [
            1.5 + 0.4 * (0.3 * time_s).sin() + altitude_m * 0.001,
            -0.8,
            0.0,
        ]
    }
}

/// The target's true path. The same function feeds the pursuit oracle in
/// stage 1 and the truth list in stage 6, as one scenario trajectory does in
/// the reference.
fn target_state(timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
    let x = 450.0 - 1.5 * timestamp_s;
    let y = 150.0 + 0.5 * timestamp_s;
    ([x, y, ground(x, y) + 5.0], [-1.5, 0.5, -0.06])
}

struct AnalyticPursuit;

impl PursuitOracle for AnalyticPursuit {
    fn target_state(&self, target_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        (target_id == "target-a").then(|| target_state(timestamp_s))
    }
}

fn lane() -> LoopedWaypointPath {
    LoopedWaypointPath::new(WaypointPathConfig {
        waypoints_xy_m: vec![
            [-300.0, -200.0],
            [300.0, -200.0],
            [300.0, -100.0],
            [-300.0, -100.0],
        ],
        speed_mps: 12.0,
        base_agl_m: 80.0,
        vertical_amplitude_m: 6.0,
        vertical_frequency_rad_s: 0.2,
        phase: 0.0,
        min_agl_m: 18.0,
        avoid_obstacles: false,
    })
    .expect("valid lane")
}

fn follow(phase: f64) -> FollowPathController {
    FollowPathController::new(FollowPathConfig {
        map_bounds_m: BoundsXy {
            x_min_m: -2000.0,
            x_max_m: 2000.0,
            y_min_m: -2000.0,
            y_max_m: 2000.0,
        },
        base_agl_m: 80.0,
        standoff_radius_m: 150.0,
        max_speed_mps: 18.0,
        phase,
        // The limiter carries a velocity across ticks: state the checkpoint
        // must hold, or the first resumed tick is limited against nothing.
        max_accel_mps2: Some(3.0),
        ..FollowPathConfig::default()
    })
}

fn launched(operational: Operational, station: [f64; 3]) -> PlatformController {
    PlatformController::Launched {
        controller: LaunchController::new(station, operational, 8.0, 80.0),
        profile: LaunchProfile {
            min_operational_agl_m: 18.0,
            wind_scale: 0.35,
            auto_launch_at_s: Some(0.0),
        },
        launch_target_id: "target-a".to_string(),
    }
}

/// A search drone with no automatic launch; the driver launches it.
fn late_drone() -> PlatformController {
    let PlatformController::Launched {
        controller,
        launch_target_id,
        ..
    } = launched(Operational::Search(lane()), station(0.0, -400.0))
    else {
        unreachable!()
    };
    PlatformController::Launched {
        controller,
        profile: LaunchProfile {
            min_operational_agl_m: 18.0,
            wind_scale: 0.35,
            auto_launch_at_s: None,
        },
        launch_target_id,
    }
}

fn node(id: &str, mobile: bool) -> NodeState {
    NodeState {
        node_id: id.to_string(),
        position: Vector3::new(0.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        is_mobile: mobile,
        timestamp_s: 0.0,
        health: 1.0,
        ..Default::default()
    }
}

fn station(x: f64, y: f64) -> [f64; 3] {
    [x, y, ground(x, y) + 2.0]
}

fn fleet() -> Vec<NodeState> {
    vec![
        node("gs-0", false),
        node("gs-1", false),
        node("drone-search", true),
        node("drone-follow", true),
        node("drone-late", true),
    ]
}

/// Knobs a test turns to make one `Rig` differ from the reference.
#[derive(Clone, Copy)]
struct Options {
    /// The tick on which the driver launches `drone-late` by hand. The other
    /// drones auto-launch at t = 0, so without a hand launch every object
    /// holds the same launch state and a restore that dropped it would pass.
    late_launch_tick: u64,
    /// Install the two ground stations in the other order. Two controllers
    /// of the *same* kind, so only the id check can tell -- swapping two of
    /// different kinds is refused by the kind check first, which is how a
    /// first version of this test let an order check that did nothing pass.
    swap_order: bool,
}

const REFERENCE: Options = Options {
    late_launch_tick: 30,
    swap_order: false,
};

/// A wrong history for the resumed object: the late drone launched at once.
const WRONG_HISTORY: Options = Options {
    late_launch_tick: 2,
    swap_order: false,
};

struct Rig {
    options: Options,
    sim: Simulation,
}

impl Rig {
    /// Build the whole mission. Called once per `Simulation`, so every
    /// resumed run is a genuinely different object with fresh handlers.
    fn new() -> Self {
        Self::build(REFERENCE)
    }

    fn build(options: Options) -> Self {
        let mut sim = Simulation::new(SimulationConfig {
            seed: 21,
            dt_s: DT_S,
            steps: Some(STEPS),
            run_id: "controller-stage".to_string(),
        })
        .expect("config is valid");
        sim.load_world(fleet(), Vec::new());

        let mut battery = BatteryStage::new();
        battery.register("drone-search", Some(station(0.0, -400.0)));
        battery.register("drone-follow", Some(station(0.0, -400.0)));

        let mut stage = ControllerStage::new().with_battery(battery);
        stage.set_environment(ControllerEnvironment {
            terrain: Box::new(Plane),
            obstacles: None,
            visibility: Box::new(ClearSky),
            planner: None,
            zones: Vec::new(),
            weather: Some(Box::new(Breeze)),
            map_bounds_m: Some(BoundsXy {
                x_min_m: -2000.0,
                x_max_m: 2000.0,
                y_min_m: -2000.0,
                y_max_m: 2000.0,
            }),
            pursuit: Some(Box::new(AnalyticPursuit)),
        });
        let stations = [
            ("gs-0", station(0.0, -400.0)),
            ("gs-1", station(250.0, -420.0)),
        ];
        let station_order: [usize; 2] = if options.swap_order { [1, 0] } else { [0, 1] };
        for index in station_order {
            let (id, position_m) = stations[index];
            stage.install(id, PlatformController::Static { position_m });
        }
        let search = launched(
            Operational::SearchThenFollow(ObservationTriggeredFollow::new(
                "drone-search",
                "target-a",
                lane(),
                follow(1.1),
            )),
            station(0.0, -400.0),
        );
        let follower = launched(
            Operational::Follow {
                target_id: "target-a".to_string(),
                controller: follow(2.6),
            },
            station(0.0, -400.0),
        );
        stage.install("drone-search", search);
        stage.install("drone-follow", follower);
        stage.install("drone-late", late_drone());
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(stage))
            .expect("stage 1 is not a transaction boundary");

        let mut observe = ObserveStage::new();
        observe.install_sensor(
            "drone-search",
            SensorSpec {
                max_range_m: 400.0,
                error: SensorErrorConfig {
                    detection_range_knee_m: 100_000.0,
                    atmospheric_attenuation_coeff: 0.0,
                    ..SensorErrorConfig::default()
                },
                ..SensorSpec::default()
            },
        );
        sim.set_stage_handler(Stage::GenerateObservations, Box::new(observe))
            .expect("stage 6 is not a transaction boundary");
        sim.connect(Handoff::EngagementToControllers)
            .expect("stages 1 and 6 are installed");

        Self { options, sim }
    }

    fn controllers(&self) -> &ControllerStage {
        self.sim
            .stage(Stage::SampleControllersAndBattery)
            .expect("stage 1 is installed")
    }

    fn battery(&self) -> &BatteryStage {
        BatteryStage::of(&self.sim).expect("stage 1 chains a battery")
    }

    fn controllers_mut(&mut self) -> &mut ControllerStage {
        self.sim
            .stage_mut(Stage::SampleControllersAndBattery)
            .expect("stage 1 is installed")
    }

    /// One tick, with truth prescribed for it first, as the scenario does.
    fn step(&mut self) -> Frame {
        if self.sim.tick() == self.options.late_launch_tick {
            let tick = self.sim.tick();
            self.controllers_mut()
                .trigger_launch("drone-late", tick as f64 * DT_S, "target-a");
        }
        let (position, velocity) = target_state(self.sim.tick() as f64 * DT_S);
        self.sim.set_truth(vec![TruthState {
            target_id: "target-a".to_string(),
            position: Vector3::new(position[0], position[1], position[2]),
            velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
            timestamp_s: self.sim.tick() as f64 * DT_S,
        }]);
        let snapshot = self.sim.step().expect("tick runs");
        Frame {
            tick: snapshot.header.tick,
            platforms: snapshot
                .platforms
                .iter()
                .map(|p| {
                    (
                        p.node_id.clone(),
                        [
                            p.position.x.to_bits(),
                            p.position.y.to_bits(),
                            p.position.z.to_bits(),
                            p.velocity.x.to_bits(),
                            p.velocity.y.to_bits(),
                            p.velocity.z.to_bits(),
                            p.timestamp_s.to_bits(),
                        ],
                        p.battery_fraction.map(f64::to_bits),
                    )
                })
                .collect(),
        }
    }
}

/// One tick's output, as bits, so equality is exact.
#[derive(Debug, Clone, PartialEq)]
struct Frame {
    tick: u64,
    platforms: Vec<(String, [u64; 7], Option<u64>)>,
}

/// The uninterrupted run, plus the tick on which the search drone engaged.
fn uninterrupted() -> (Vec<Frame>, u64) {
    let mut rig = Rig::new();
    let mut frames = Vec::new();
    let mut engaged_at = None;
    for _ in 0..STEPS {
        frames.push(rig.step());
        if engaged_at.is_none() && rig.controllers().is_engaged("drone-search") == Some(true) {
            engaged_at = Some(rig.sim.tick());
        }
    }
    let engaged_at = engaged_at.expect("the search drone never saw its target");
    (frames, engaged_at)
}

fn position(frame: &Frame, id: &str) -> [f64; 3] {
    let (_, bits, _) = frame
        .platforms
        .iter()
        .find(|(node, _, _)| node == id)
        .expect("platform in frame");
    [
        f64::from_bits(bits[0]),
        f64::from_bits(bits[1]),
        f64::from_bits(bits[2]),
    ]
}

#[test]
fn the_tick_moves_the_fleet() {
    let (frames, engaged_at) = uninterrupted();

    // The station never moves; both drones do, every tick after the climb.
    let first = &frames[0];
    let last = frames.last().unwrap();
    assert_eq!(position(first, "gs-0"), position(last, "gs-0"));
    for drone in ["drone-search", "drone-follow"] {
        assert_ne!(
            position(first, drone),
            position(last, drone),
            "{drone} never moved"
        );
        for window in frames[20..].windows(2) {
            assert_ne!(
                position(&window[0], drone),
                position(&window[1], drone),
                "{drone} stood still at tick {}",
                window[1].tick
            );
        }
    }

    // Every controlled platform carries the time it was sampled at, which is
    // the tick's start: the snapshot's tick is already one past it.
    for frame in &frames {
        for (id, bits, _) in &frame.platforms {
            assert_eq!(
                f64::from_bits(bits[6]),
                (frame.tick - 1) as f64 * DT_S,
                "{id} timestamp at tick {}",
                frame.tick
            );
        }
    }

    // Tick 0 is launch: both drones sit exactly on the station.
    assert_eq!(position(first, "drone-follow"), station(0.0, -400.0));

    // Nothing is ever below terrain -- CLAUDE.md's hard invariant.
    for frame in &frames {
        for (id, _, _) in &frame.platforms {
            let [x, y, z] = position(frame, id);
            assert!(z >= ground(x, y), "{id} underground at tick {}", frame.tick);
        }
    }

    // Engagement arrived mid-run, through stage 6, not at launch.
    assert!(
        (40..STEPS - 40).contains(&engaged_at),
        "engaged at tick {engaged_at}; the scenario no longer exercises the handover"
    );

    // Battery accounting ran over the moved positions.
    let drawn = frames
        .last()
        .unwrap()
        .platforms
        .iter()
        .find(|(id, _, _)| id == "drone-search")
        .and_then(|(_, _, fraction)| fraction.map(f64::from_bits))
        .expect("the drone has a battery");
    assert!(drawn < 1.0, "no energy drawn: {drawn}");
}

#[test]
fn the_trace_names_every_truth_fed_pursuit_read() {
    let mut rig = Rig::new();
    for _ in 0..24 {
        rig.step();
    }
    let notes: Vec<String> = rig
        .sim
        .trace()
        .records()
        .iter()
        .filter(|r| r.stage == Stage::SampleControllersAndBattery && r.tick == 23)
        .flat_map(|r| r.notes.clone())
        .collect();
    assert!(
        notes.iter().any(|n| n.contains("legacy pursuit read")),
        "a truth-fed follow flew without the trace saying so: {notes:?}"
    );
}

/// The differential. Four checkpoint ticks, each resumed into a stranger.
#[test]
fn a_resumed_run_is_the_same_run() {
    let (reference, engaged_at) = uninterrupted();
    for checkpoint_tick in [5, 60, engaged_at, STEPS - 25] {
        let mut original = Rig::new();
        for _ in 0..checkpoint_tick {
            original.step();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().expect("not faulted"))
            .expect("checkpoint serialises");
        drop(original);

        // A different object, sent down a wrong history first, so a restore
        // that only overwrote some of the state would leave evidence.
        let mut resumed = Rig::build(WRONG_HISTORY);
        for _ in 0..(checkpoint_tick / 2 + 7) {
            resumed.step();
        }
        let checkpoint: SimulationCheckpoint =
            serde_json::from_str(&text).expect("checkpoint parses");
        resumed.sim.restore(&checkpoint).expect("restore accepted");
        // The driver's launch schedule is scenario, not state: the resumed
        // run keeps the reference's from here on.
        resumed.options = REFERENCE;
        assert_eq!(resumed.sim.tick(), checkpoint_tick);

        for expected in &reference[checkpoint_tick as usize..] {
            let actual = resumed.step();
            assert_eq!(
                &actual, expected,
                "resumed from tick {checkpoint_tick} (engagement at {engaged_at}), diverged at \
                 tick {}",
                expected.tick
            );
        }
    }
}

#[test]
fn a_reset_run_is_a_fresh_run() {
    let (reference, _) = uninterrupted();
    let mut rig = Rig::new();
    for _ in 0..STEPS {
        rig.step();
    }
    // `reset` clears the world and truth; the caller reloads the world, as
    // the reference's driver rebuilds node states.
    rig.sim.reset();
    rig.sim.load_world(fleet(), Vec::new());
    for expected in &reference {
        let actual = rig.step();
        assert_eq!(
            actual.platforms, expected.platforms,
            "reset run diverged from a fresh one at tick {}",
            expected.tick
        );
    }
    assert_eq!(
        rig.battery().has_warned("drone-search"),
        Rig::new().battery().has_warned("drone-search")
    );
}

#[test]
fn a_checkpoint_for_a_different_fleet_is_refused_without_mutation() {
    let mut source = Rig::new();
    for _ in 0..60 {
        source.step();
    }
    let checkpoint = source.sim.checkpoint().expect("not faulted");

    // Same ids, but the follow drone is a search drone here: the checkpoint's
    // follow state has nowhere to go.
    let mut other = Rig::new();
    other.controllers_mut().install(
        "drone-follow",
        launched(Operational::Search(lane()), station(0.0, -400.0)),
    );
    for _ in 0..3 {
        other.step();
    }
    let before = other.sim.checkpoint().expect("not faulted");
    let refusal = other
        .sim
        .restore(&checkpoint)
        .expect_err("mismatched fleet");
    assert!(
        matches!(&refusal, RestoreError::StageState { stage, .. } if stage == "sample_controllers_and_battery"),
        "{refusal}"
    );
    assert_eq!(
        other.sim.checkpoint().expect("not faulted"),
        before,
        "a refused restore changed the run"
    );
}

#[test]
fn a_checkpoint_naming_an_absent_stage_is_refused() {
    let mut source = Rig::new();
    source.step();
    let checkpoint = source.sim.checkpoint().expect("not faulted");

    let mut bare = Simulation::new(SimulationConfig {
        seed: 21,
        dt_s: DT_S,
        steps: Some(STEPS),
        run_id: "controller-stage".to_string(),
    })
    .expect("config is valid");
    assert!(matches!(
        bare.restore(&checkpoint),
        Err(RestoreError::UnknownStageState { .. })
    ));
}

#[test]
fn a_checkpoint_for_a_reordered_fleet_is_refused() {
    let mut source = Rig::new();
    for _ in 0..10 {
        source.step();
    }
    let checkpoint = source.sim.checkpoint().expect("not faulted");
    let mut reordered = Rig::build(Options {
        swap_order: true,
        ..REFERENCE
    });
    // Sampling order is declaration order, and battery and deconfliction tie
    // on it; the same controllers in another order are another scenario.
    assert!(matches!(
        reordered.sim.restore(&checkpoint),
        Err(RestoreError::StageState { .. })
    ));
}

/// A checkpoint that carries no stage-1 state -- written by a run with no
/// stage-1 handler -- returns this run's controllers and battery to fresh,
/// rather than leaving them wherever this object had flown them.
#[test]
fn a_checkpoint_without_stage_state_resets_the_stage() {
    let (reference, _) = uninterrupted();
    let bare = Simulation::new(SimulationConfig {
        seed: 21,
        dt_s: DT_S,
        steps: Some(STEPS),
        run_id: "controller-stage".to_string(),
    })
    .expect("config is valid");
    let checkpoint = bare.checkpoint().expect("not faulted");
    assert!(checkpoint.stage_states.is_empty());

    let mut rig = Rig::new();
    for _ in 0..70 {
        rig.step();
    }
    rig.sim.restore(&checkpoint).expect("restore accepted");
    for expected in &reference {
        assert_eq!(
            rig.step().platforms,
            expected.platforms,
            "stage state survived a restore that did not carry it (tick {})",
            expected.tick
        );
    }
}

/// The stage hands the launch path the reference's one-second drift step,
/// not the run's timestep. Compared against a `LaunchController` driven
/// directly with the reference's weather context -- the trajectory parity
/// fixture pins that controller; this pins what the stage gives it.
#[test]
fn launch_drift_is_integrated_over_the_reference_one_second() {
    let (frames, _) = uninterrupted();
    let mut direct = LaunchController::new(station(0.0, -400.0), lane(), 8.0, 80.0);
    direct.trigger_launch(REFERENCE.late_launch_tick as f64 * DT_S, "target-a");
    let bounds = BoundsXy {
        x_min_m: -2000.0,
        x_max_m: 2000.0,
        y_min_m: -2000.0,
        y_max_m: 2000.0,
    };
    let weather = argusnet_sim::trajectory::WeatherContext {
        weather: Some(&Breeze),
        terrain: Some(&Plane),
        bounds: Some(bounds),
        min_agl_m: 18.0,
        wind_scale: 0.35,
        dt_s: 1.0,
    };
    let mut operational_ticks = 0;
    for frame in &frames {
        let t = (frame.tick - 1) as f64 * DT_S;
        let (expected, _) = direct.sample_with(t, &weather, |lane, t| lane.sample(t, &Plane, None));
        assert_eq!(
            position(frame, "drone-late"),
            expected,
            "tick {}",
            frame.tick
        );
        if t >= REFERENCE.late_launch_tick as f64 * DT_S + 8.0 {
            operational_ticks += 1;
        }
    }
    assert!(
        operational_ticks > 50,
        "the late drone barely flew its lane"
    );
}

/// Stage 1's nominal poses answer only for the tick they were sampled on: a
/// later stage asking about any other time is refused rather than handed a
/// stale pose.
#[test]
fn nominal_poses_answer_only_for_their_own_tick() {
    let mut rig = Rig::new();
    for _ in 0..12 {
        rig.step();
    }
    let sampled_at = 11.0 * DT_S;
    let (position, _) = rig
        .controllers()
        .nominal_pose("drone-follow", sampled_at)
        .expect("the last tick's pose");
    assert!(position.iter().all(|v| v.is_finite()));
    assert_eq!(
        rig.controllers()
            .nominal_pose("drone-follow", sampled_at + DT_S),
        None
    );
    assert_eq!(
        rig.controllers()
            .nominal_pose("drone-follow", sampled_at - DT_S),
        None
    );
    assert_eq!(rig.controllers().nominal_pose("nobody", sampled_at), None);
}
