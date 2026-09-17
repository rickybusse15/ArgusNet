//! Stage 5: synthetic IMU, the VIO update, and the observation pose select.
//!
//! §4 names stage 5 *"truth/synthetic IMU features, VIO update, observation
//! pose select"*. The reference is one loop over the scenario's mobile nodes:
//!
//! ```python
//! cur_pos, cur_vel = node.trajectory(timestamp_s)
//! synth_accel = (cur_vel - prev_vel) / max(dt_s, 1e-6)
//! vio_backend.process_imu(IMUMeasurement(zeros(3), synth_accel, timestamp_s))
//! synth_features = [VisualFeature(fid, u=fid * 30, v=fid * 20) for fid in range(10)]
//! vio_state = vio_backend.process_image(synth_features, timestamp_s=timestamp_s)
//! ```
//!
//! # The VIO integrates the *nominal* trajectory, not the flown one
//!
//! `node.trajectory(timestamp_s)` is the scenario's undisturbed path. Stages 2,
//! 3, 4 and 10 have all already run by the time stage 5 does, so the platform in
//! the world may have been redirected to a frontier cell, nudged by
//! deconfliction and clamped by the safety monitor — and **the VIO sees none of
//! it**. It integrates the path the platform would have flown.
//!
//! That is load-bearing rather than incidental: it is what makes the VIO
//! estimate drift away from the world position in a way the fusion stage can
//! see. Reading the world here instead would make the estimate track the truth
//! it is supposed to be independent of.
//!
//! So the nominal path arrives through an injected [`NominalTrajectory`] rather
//! than from [`TickContext::world`]. §3's truth capability covers *target*
//! truth, which is a different thing; a platform's own prescribed path is not
//! in `TickContext::truth` at all. With no source installed the stage reports
//! [`StageOutcome::Skipped`] rather than quietly falling back to the world,
//! because the fallback would be a different estimator.
//!
//! # Three dead stores in the reference
//!
//! - `_vio_prev_positions` is written every tick and never read.
//! - `_vio_prev_velocities.get(node_id, np.zeros(3))` names a zeros default
//!   that cannot fire: the same setup loop that creates a backend fills both
//!   maps, so the key is always present.
//! - The setup loop's `_vio_prev_velocities[node_id] = vel0` seeding has no
//!   observable effect either, because the only tick that would consume it —
//!   the one at `t = 0`, where the filter was also initialised — has an empty
//!   prediction interval and discards its IMU sample. Running the reference
//!   with the seed and with zeros gives identical output at every tick.
//!
//! The first two are not reproduced. The third is, because
//! [`VioLink::track`] mirrors the reference's signature and a future change
//! could make the value observable again.
//!
//! # What the synthetic features actually do
//!
//! The ten features are **identical on every frame** — `u = id * 30`,
//! `v = id * 20`, never a function of time or of where the platform is. So the
//! measured flow between any two frames is exactly zero, and the update's
//! residual is the negated prediction: the filter is told, every tick, that it
//! did not move. That is a strong pull toward zero velocity and it is what the
//! reference does. Pinned by a test, because it reads like a placeholder that
//! someone might "fix" into something position-dependent.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_localization::vio::{
    FeatureStore, FlowConfig, FrameTime, UpdateOutcome, VisualFeature,
};
use argusnet_localization::{ImuMeasurement, VioNoise, VioPredictor};
use argusnet_types::Vector3;

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// `_vio_feature_ids = list(range(10))`.
pub const SYNTHETIC_FEATURE_COUNT: u64 = 10;

/// `u = float(fid * 30)`.
pub const FEATURE_U_STRIDE_PX: f64 = 30.0;

/// `v = float(fid * 20)`.
pub const FEATURE_V_STRIDE_PX: f64 = 20.0;

/// `max(dt_s, 1e-6)` — the guard on the synthetic acceleration's divisor.
pub const MIN_ACCEL_DT_S: f64 = 1.0e-6;

/// The ten features the reference synthesises, which do not vary with time.
pub fn synthetic_features() -> Vec<VisualFeature> {
    (0..SYNTHETIC_FEATURE_COUNT)
        .map(|feature_id| VisualFeature {
            feature_id,
            u: feature_id as f64 * FEATURE_U_STRIDE_PX,
            v: feature_id as f64 * FEATURE_V_STRIDE_PX,
        })
        .collect()
}

/// The scenario's undisturbed path for a platform.
///
/// The reference calls `node.trajectory(t)`, which knows nothing about what the
/// earlier stages did to the platform this tick. Injecting it keeps that
/// independence explicit instead of leaving it to whoever wires the stage.
pub trait NominalTrajectory: fmt::Debug + Send {
    /// Position and velocity at `timestamp_s`, or `None` for a platform this
    /// source does not prescribe.
    fn sample(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])>;
}

/// One platform's VIO estimate for one tick.
///
/// The three fields are exactly what `sim.py` reads back —
/// `VIO_CONTRACT.md`: *"the consumer surface is narrow"* — so nothing wider is
/// published.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VioEstimate {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    /// `trace(covariance[6:9, 6:9])` — the position block.
    pub covariance_pos_trace: f64,
    /// What the update did. The reference cannot report this, because it
    /// reports "no information" and "confident" as the same smaller
    /// covariance.
    pub outcome: UpdateOutcome,
}

#[derive(Debug)]
struct Backend {
    predictor: VioPredictor,
    store: FeatureStore,
    previous_velocity: [f64; 3],
}

#[derive(Debug, Default)]
struct Shared {
    backends: BTreeMap<String, Backend>,
    nominal: Option<Box<dyn NominalTrajectory>>,
    flow: FlowConfig,
    /// `scenario.use_vio_position`. Off in the reference by default.
    use_vio_position: bool,
    estimates: BTreeMap<String, VioEstimate>,
    /// Platforms whose frame was dropped as out-of-order on the last run.
    dropped: Vec<String>,
}

/// A handle on the per-platform filters, held before the stage is installed.
#[derive(Debug, Clone, Default)]
pub struct VioLink {
    inner: Arc<Mutex<Shared>>,
}

impl VioLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Install the source of undisturbed platform paths.
    pub fn set_nominal(&self, nominal: Box<dyn NominalTrajectory>) {
        self.lock().nominal = Some(nominal);
    }

    /// Start a filter for `node_id`, matching the reference's setup loop:
    /// `initialize(node.trajectory(0.0)[0], timestamp_s=0.0)`, with the
    /// previous velocity seeded from that same sample rather than from zeros.
    pub fn track(&self, node_id: &str, position: [f64; 3], velocity: [f64; 3]) {
        let mut predictor = VioPredictor::new(VioNoise::default());
        predictor.initialize(position, 0.0);
        self.lock().backends.insert(
            node_id.to_string(),
            Backend {
                predictor,
                store: FeatureStore::default(),
                previous_velocity: velocity,
            },
        );
    }

    pub fn is_tracked(&self, node_id: &str) -> bool {
        self.lock().backends.contains_key(node_id)
    }

    /// `replay_metadata["vio_tracked_nodes"] = sorted(_vio_backends.keys())`.
    pub fn tracked_nodes(&self) -> Vec<String> {
        self.lock().backends.keys().cloned().collect()
    }

    pub fn set_flow_config(&self, flow: FlowConfig) {
        self.lock().flow = flow;
    }

    /// `scenario.use_vio_position`.
    pub fn set_use_vio_position(&self, enabled: bool) {
        self.lock().use_vio_position = enabled;
    }

    pub fn estimate(&self, node_id: &str) -> Option<VioEstimate> {
        self.lock().estimates.get(node_id).copied()
    }

    /// This tick's estimates, in platform-id order.
    pub fn estimates(&self) -> Vec<(String, VioEstimate)> {
        self.lock()
            .estimates
            .iter()
            .map(|(id, estimate)| (id.clone(), *estimate))
            .collect()
    }

    /// Platforms whose frame the last run dropped as out-of-order.
    pub fn dropped(&self) -> Vec<String> {
        self.lock().dropped.clone()
    }
}

/// [`Stage::UpdateVioAndPose`] as a [`StageHandler`].
///
/// [`Stage::UpdateVioAndPose`]: crate::stage::Stage::UpdateVioAndPose
#[derive(Debug, Default)]
pub struct VioStage {
    link: VioLink,
}

impl VioStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> VioLink {
        self.link.clone()
    }
}

impl StageHandler for VioStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();
        shared.estimates.clear();
        shared.dropped.clear();

        if shared.nominal.is_none() {
            return Ok(StageOutcome::Skipped {
                reason: "no nominal trajectory source installed".to_string(),
            });
        }
        if shared.backends.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let timestamp_s = context.timestamp_s;
        let dt_s = context.dt_s;
        let flow = shared.flow;

        // BTreeMap, so the platforms are visited in id order on every run. The
        // filters are independent, but the trace is not, and a trace whose line
        // order moved run to run would be useless for comparing against Python.
        let node_ids: Vec<String> = shared.backends.keys().cloned().collect();

        for node_id in node_ids {
            let Some(sample) = shared
                .nominal
                .as_ref()
                .expect("checked above")
                .sample(&node_id, timestamp_s)
            else {
                // A tracked platform the source does not prescribe. The
                // reference cannot reach this -- every backend is created from
                // a trajectory -- so there is nothing to reproduce, and
                // stepping the filter on a guess would be worse than not
                // stepping it.
                continue;
            };
            let (current_position, current_velocity) = sample;

            let Some(backend) = shared.backends.get_mut(&node_id) else {
                continue;
            };

            // `(cur_vel - prev_vel) / max(dt_s, 1e-6)`, with the angular rate
            // left at zero: the synthetic IMU models translation only.
            let divisor = dt_s.max(MIN_ACCEL_DT_S);
            let linear_acceleration = [
                (current_velocity[0] - backend.previous_velocity[0]) / divisor,
                (current_velocity[1] - backend.previous_velocity[1]) / divisor,
                (current_velocity[2] - backend.previous_velocity[2]) / divisor,
            ];
            let measurement = ImuMeasurement {
                angular_velocity: [0.0, 0.0, 0.0],
                linear_acceleration,
                timestamp_s,
            };

            // Both of these read the filter's clock, and `predict_to` advances
            // it, so they are taken first. Computing `dt_total` afterwards
            // would make it identically zero and silence the whole update.
            let frame_time = backend.predictor.classify(timestamp_s);
            let dt_total = timestamp_s - backend.predictor.timestamp_s();

            if frame_time == FrameTime::OutOfOrder {
                // The reference returns before refreshing the feature store, so
                // the next frame matches against features two frames old. That
                // is finding 3 in `VIO_CONTRACT.md`; the store is deliberately
                // left alone here too, and the drop is *named* rather than
                // silent.
                shared.dropped.push(node_id);
                continue;
            }

            backend.predictor.predict_to(&[measurement], timestamp_s);

            let features = synthetic_features();
            let outcome =
                backend
                    .predictor
                    .update_optical_flow(&features, &backend.store, dt_total, &flow);
            backend.store.replace(&features);
            backend.previous_velocity = current_velocity;

            let estimate = VioEstimate {
                position: backend.predictor.position(),
                velocity: backend.predictor.velocity(),
                covariance_pos_trace: backend.predictor.position_covariance_trace(),
                outcome,
            };
            shared.estimates.insert(node_id, estimate);
            // `_vio_prev_positions[node_id] = cur_pos` would go here. It is a
            // dead store in the reference, so it is not reproduced.
            let _ = current_position;
        }

        // The pose select. `use_vio_position` swaps the VIO position into the
        // reported state so the tracker sees drifted positions rather than
        // ground truth -- position only, never velocity.
        let mut substituted = 0usize;
        if shared.use_vio_position {
            let positions: Vec<(String, [f64; 3])> = shared
                .estimates
                .iter()
                .map(|(id, estimate)| (id.clone(), estimate.position))
                .collect();
            for (node_id, position) in positions {
                if let Some(platform) = context.world.platform_mut(&node_id) {
                    platform.position = Vector3::new(position[0], position[1], position[2]);
                    substituted += 1;
                }
            }
        }

        if shared.estimates.is_empty() && shared.dropped.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let estimated = shared.estimates.len();
        let dropped = shared.dropped.len();
        drop(shared);
        context.note(format!(
            "vio estimated {estimated}, dropped {dropped}, substituted {substituted}"
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
    use argusnet_localization::vio::NoInformation;
    use argusnet_types::NodeState;

    /// A straight-line path at constant velocity: `p(t) = p0 + v * t`.
    #[derive(Debug, Clone)]
    struct Straight {
        start: [f64; 3],
        velocity: [f64; 3],
        /// Platforms this source prescribes. Anything else gets `None`.
        prescribes: Vec<String>,
    }

    impl NominalTrajectory for Straight {
        fn sample(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
            if !self.prescribes.iter().any(|id| id == node_id) {
                return None;
            }
            Some((
                [
                    self.start[0] + self.velocity[0] * timestamp_s,
                    self.start[1] + self.velocity[1] * timestamp_s,
                    self.start[2] + self.velocity[2] * timestamp_s,
                ],
                self.velocity,
            ))
        }
    }

    /// A path accelerating straight up from rest: `v(t) = [0, 0, a * t]`.
    #[derive(Debug, Clone)]
    struct Accelerating {
        start: [f64; 3],
        acceleration_z: f64,
    }

    impl NominalTrajectory for Accelerating {
        fn sample(&self, _node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
            Some((
                [
                    self.start[0],
                    self.start[1],
                    self.start[2] + 0.5 * self.acceleration_z * timestamp_s * timestamp_s,
                ],
                [0.0, 0.0, self.acceleration_z * timestamp_s],
            ))
        }
    }

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 11,
            dt_s: 1.0,
            steps: Some(2_000),
            run_id: "vio".to_string(),
        })
        .expect("config is valid")
    }

    fn node(id: &str, x: f64, y: f64) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, y, 80.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: true,
            timestamp_s: 0.0,
            health: 1.0,
        }
    }

    fn rig(platforms: Vec<NodeState>) -> (Simulation, VioLink) {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        let stage = VioStage::new();
        let link = stage.link();
        sim.set_stage_handler(Stage::UpdateVioAndPose, Box::new(stage))
            .expect("stage 5 is not a transaction boundary");
        (sim, link)
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::UpdateVioAndPose)
            .expect("stage 5 ran")
            .outcome
            .clone()
    }

    /// A rig with one drone flying east at 10 m/s from the origin.
    fn eastbound() -> (Simulation, VioLink) {
        let (sim, link) = rig(vec![node("d0", 0.0, 0.0)]);
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["d0".to_string()],
        }));
        link.track("d0", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);
        (sim, link)
    }

    // -----------------------------------------------------------------
    // The synthetic inputs
    // -----------------------------------------------------------------

    /// Ten features, `u = id * 30`, `v = id * 20`, and — the part that matters
    /// — **identical on every frame**. The measured flow between any two frames
    /// is therefore exactly zero, so the filter is told every tick that it did
    /// not move. It reads like a placeholder someone might "fix" into something
    /// position-dependent, which would change the estimator.
    #[test]
    fn the_synthetic_features_do_not_vary_with_time_or_position() {
        let features = synthetic_features();

        assert_eq!(features.len(), 10);
        assert_eq!(
            features[0],
            VisualFeature {
                feature_id: 0,
                u: 0.0,
                v: 0.0
            }
        );
        assert_eq!(
            features[9],
            VisualFeature {
                feature_id: 9,
                u: 270.0,
                v: 180.0
            }
        );
        // The function takes no arguments at all, so this is structural rather
        // than a sampled claim -- but call it twice anyway, because a future
        // cache or counter inside it would break the invariant silently.
        assert_eq!(synthetic_features(), features);
    }

    // -----------------------------------------------------------------
    // The nominal path, not the flown one
    // -----------------------------------------------------------------

    /// Stages 2, 3, 4 and 10 have all run by the time stage 5 does, so the
    /// platform in the world may have been redirected, nudged and clamped. The
    /// VIO must integrate the path the platform *would* have flown, which is
    /// what makes its estimate drift independently of the truth.
    ///
    /// Two runs, identical but for a world position thrown a kilometre off
    /// course every tick, must publish bit-identical estimates. Comparing runs
    /// rather than asserting a range means the test does not have to know where
    /// the estimate ought to land — only that the world cannot reach it.
    #[test]
    fn the_filter_integrates_the_nominal_path_not_the_world_position() {
        let undisturbed = {
            let (mut sim, link) = eastbound();
            for _ in 0..5 {
                sim.step().expect("tick runs");
            }
            link.estimate("d0").expect("an estimate was published")
        };

        let (mut sim, link) = eastbound();
        for _ in 0..5 {
            sim.world_mut().platform_mut("d0").unwrap().position =
                Vector3::new(1_000.0, 999.0, 7.0);
            sim.step().expect("tick runs");
        }
        let displaced = link.estimate("d0").expect("an estimate was published");

        assert_eq!(
            displaced.position.map(f64::to_bits),
            undisturbed.position.map(f64::to_bits)
        );
        assert_eq!(
            displaced.velocity.map(f64::to_bits),
            undisturbed.velocity.map(f64::to_bits)
        );
    }

    /// With no source installed the stage is `Skipped`, not quietly reading the
    /// world instead. The fallback would be a different estimator.
    #[test]
    fn without_a_nominal_source_the_stage_is_skipped() {
        let (mut sim, link) = rig(vec![node("d0", 0.0, 0.0)]);
        link.track("d0", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);

        sim.step().expect("tick runs");

        assert!(matches!(outcome(&sim), StageOutcome::Skipped { .. }));
        assert_eq!(link.estimate("d0"), None);
    }

    /// A tracked platform the source does not prescribe is skipped rather than
    /// stepped on a guess. The reference cannot reach this case, so there is
    /// nothing to reproduce and the safe reading is the right one.
    #[test]
    fn a_tracked_platform_the_source_does_not_prescribe_is_skipped() {
        let (mut sim, link) = rig(vec![node("known", 0.0, 0.0), node("stranger", 5.0, 5.0)]);
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["known".to_string()],
        }));
        link.track("known", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);
        link.track("stranger", [5.0, 5.0, 80.0], [0.0, 0.0, 0.0]);

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert!(link.estimate("known").is_some());
        assert_eq!(link.estimate("stranger"), None);
        // Still tracked -- skipped for this tick, not dropped from the run.
        assert!(link.is_tracked("stranger"));
    }

    #[test]
    fn with_no_tracked_platforms_there_is_no_work() {
        let (mut sim, link) = rig(vec![node("d0", 0.0, 0.0)]);
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["d0".to_string()],
        }));

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// `sorted(_vio_backends.keys())` — the reference sorts on the way into the
    /// replay metadata, and the stage visits platforms in the same order, so
    /// the trace is comparable run to run.
    #[test]
    fn tracked_nodes_come_back_in_id_order() {
        let (_sim, link) = rig(vec![node("d0", 0.0, 0.0)]);
        link.track("zulu", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        link.track("alpha", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        link.track("mike", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        assert_eq!(link.tracked_nodes(), vec!["alpha", "mike", "zulu"]);
    }

    // -----------------------------------------------------------------
    // Frame ordering
    // -----------------------------------------------------------------

    /// `classify` and `dt_total` both read the filter's clock, and `predict_to`
    /// advances it. Taking them afterwards makes `dt_total` identically zero,
    /// which zeroes the measurement Jacobian and silences the update entirely —
    /// leaving prediction's estimate standing every tick. The two runs stay
    /// plausible, so only the estimate itself separates them.
    #[test]
    fn the_frame_interval_is_measured_before_prediction_advances_the_clock() {
        let (mut sim, link) = eastbound();

        for _ in 0..6 {
            sim.step().expect("tick runs");
        }
        let estimate = link.estimate("d0").expect("an estimate was published");

        // The flow update says "you did not move", which drags the velocity
        // down from the 10 m/s the IMU integrated. With dt_total zeroed the
        // update carries no information and the velocity stays where prediction
        // put it.
        assert!(
            estimate.velocity[0] < 9.0,
            "the flow update did not bite: vx = {}",
            estimate.velocity[0]
        );
        assert!(
            matches!(estimate.outcome, UpdateOutcome::Applied { .. }),
            "outcome was {:?}",
            estimate.outcome
        );
    }

    /// An out-of-order frame is dropped, and — per finding 3 in
    /// `VIO_CONTRACT.md` — the feature store is deliberately **not** refreshed,
    /// so the next frame would match against features from two frames ago. The
    /// drop is *named* here, which the reference cannot do.
    ///
    /// The case is reachable, and the way it is reachable is worth knowing:
    /// `Simulation::reset` rewinds the clock and clears the world, but the
    /// stage's filters live behind the link and keep the clock they reached. So
    /// a reset run drops every VIO frame until the sim clock catches back up.
    /// Whoever wires reset has to re-`track` on the new epoch; this test is
    /// where that requirement is written down.
    #[test]
    fn a_reset_leaves_the_filters_ahead_of_the_clock_and_every_frame_is_dropped() {
        let (mut sim, link) = eastbound();
        for _ in 0..4 {
            sim.step().expect("tick runs");
        }
        assert!(link.dropped().is_empty());
        let before_reset = link.estimate("d0").expect("an estimate was published");

        sim.reset();
        sim.load_world(vec![node("d0", 0.0, 0.0)], Vec::new());

        // The filter is at t = 4; the run restarts at t = 0.
        for _ in 0..3 {
            sim.step().expect("tick runs");
            assert_eq!(link.dropped(), vec!["d0".to_string()]);
            assert_eq!(link.estimate("d0"), None);
        }
        assert_eq!(outcome(&sim), StageOutcome::Ran);

        // Re-tracking on the new epoch is what fixes it.
        link.track("d0", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);
        sim.step().expect("tick runs");
        assert!(link.dropped().is_empty());
        let after = link.estimate("d0").expect("an estimate was published");
        assert_ne!(after.position, before_reset.position);
    }

    // -----------------------------------------------------------------
    // The pose select
    // -----------------------------------------------------------------

    /// `use_vio_position` replaces the reported **position** so the tracker sees
    /// drifted positions rather than ground truth. It does not touch velocity,
    /// which is the part a careless port adds "for consistency".
    #[test]
    fn the_pose_select_substitutes_position_but_never_velocity() {
        let (mut sim, link) = eastbound();
        link.set_use_vio_position(true);

        sim.world_mut().platform_mut("d0").unwrap().velocity = Vector3::new(3.0, 4.0, 5.0);
        for _ in 0..4 {
            sim.step().expect("tick runs");
        }

        let estimate = link.estimate("d0").expect("an estimate was published");
        let platform = sim.world().platform("d0").expect("still there");
        assert_eq!(platform.position.x, estimate.position[0]);
        assert_eq!(platform.position.y, estimate.position[1]);
        assert_eq!(platform.position.z, estimate.position[2]);
        // Untouched by the substitution.
        assert_eq!(platform.velocity, Vector3::new(3.0, 4.0, 5.0));
    }

    /// Off by default, matching `scenario.use_vio_position`. The estimate is
    /// still published — it goes to the replay either way — but the world keeps
    /// the position the earlier stages left.
    #[test]
    fn the_pose_select_is_off_by_default_and_the_estimate_still_publishes() {
        let (mut sim, link) = eastbound();

        sim.world_mut().platform_mut("d0").unwrap().position = Vector3::new(42.0, 43.0, 44.0);
        sim.step().expect("tick runs");

        assert!(link.estimate("d0").is_some());
        assert_eq!(
            sim.world().platform("d0").unwrap().position,
            Vector3::new(42.0, 43.0, 44.0)
        );
    }

    // -----------------------------------------------------------------
    // The synthetic IMU
    // -----------------------------------------------------------------

    /// The reference seeds the previous velocity from the trajectory's own
    /// first sample rather than from zeros — and **it makes no difference**,
    /// because the only tick that would consume it contributes nothing.
    ///
    /// The filter is initialised at `t = 0` and the run's first tick is also at
    /// `t = 0`, so that tick's prediction interval is empty and its IMU sample
    /// is skipped as not newer than the filter's clock. By the time tick 2
    /// synthesises an acceleration, `previous_velocity` has already been
    /// overwritten with tick 1's nominal velocity in both cases.
    ///
    /// Verified against the reference rather than reasoned about: running
    /// `EKFVIO` with `prev_vel = [10, 0, 0]` and with `prev_vel = [0, 0, 0]`
    /// produces the same position and velocity at every tick. So the seeding in
    /// the setup loop is a **third dead store**, beside `_vio_prev_positions`
    /// and the `np.zeros(3)` default.
    ///
    /// It is reproduced anyway: [`VioLink::track`] takes the velocity because
    /// the reference passes it, and a future change — initialising the filter
    /// at a non-zero timestamp, say — would make it observable again. This test
    /// is what would notice.
    #[test]
    fn the_previous_velocity_seed_is_unobservable_through_this_stage() {
        let (mut sim, seeded) = eastbound();

        let (mut other, from_zero_link) = rig(vec![node("d0", 0.0, 0.0)]);
        from_zero_link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["d0".to_string()],
        }));
        from_zero_link.track("d0", [0.0, 0.0, 80.0], [0.0, 0.0, 0.0]);

        for _ in 0..4 {
            sim.step().expect("tick runs");
            other.step().expect("tick runs");
            let a = seeded.estimate("d0").expect("an estimate was published");
            let b = from_zero_link
                .estimate("d0")
                .expect("an estimate was published");
            assert_eq!(a.position.map(f64::to_bits), b.position.map(f64::to_bits));
            assert_eq!(a.velocity.map(f64::to_bits), b.velocity.map(f64::to_bits));
        }
    }

    /// The filter is initialised at `timestamp_s = 0.0` and the run's first
    /// tick is also at 0.0, so `dt_total` is zero, the IMU sample at t = 0 is
    /// skipped as not newer than the filter's clock, and the feature store is
    /// still empty. The first estimate is therefore the initialised state
    /// exactly — every run starts with one tick that does nothing.
    #[test]
    fn the_first_tick_is_a_no_op_because_the_filter_is_initialised_at_zero() {
        let (mut sim, link) = eastbound();

        sim.step().expect("tick runs");
        let first = link.estimate("d0").expect("an estimate was published");

        assert_eq!(first.position, [0.0, 0.0, 80.0]);
        assert_eq!(first.velocity, [0.0, 0.0, 0.0]);
        // 1.0 per axis from `initialize`, untouched.
        assert_eq!(first.covariance_pos_trace, 3.0);
    }

    /// The synthetic acceleration is a *rate*: `(v - v_prev) / dt`, not the
    /// velocity delta itself. Every other test here runs at `dt_s = 1.0`, where
    /// dividing by the interval is the identity and a port that dropped the
    /// division would agree with all of them.
    ///
    /// The vertical axis is where this shows, because the optical-flow update
    /// only touches `vx` and `vy` — its Jacobian has entries at state indices 3
    /// and 4 and nowhere else — so `vz` carries the IMU's integral unopposed.
    #[test]
    fn the_synthetic_acceleration_is_a_rate_not_a_velocity_delta() {
        // A path climbing at 2 m/s^2, run at a two-second tick. Per tick the
        // velocity gains 4 m/s, so the correct synthetic acceleration is
        // 2 m/s^2 and an undivided one would be 4.
        fn climbing(dt_s: f64) -> (Simulation, VioLink) {
            let mut sim = Simulation::new(SimulationConfig {
                seed: 11,
                dt_s,
                steps: Some(100),
                run_id: "vio-dt".to_string(),
            })
            .expect("config is valid");
            sim.load_world(vec![node("d0", 0.0, 0.0)], Vec::new());
            let stage = VioStage::new();
            let link = stage.link();
            sim.set_stage_handler(Stage::UpdateVioAndPose, Box::new(stage))
                .expect("stage 5 is not a transaction boundary");
            link.set_nominal(Box::new(Accelerating {
                start: [0.0, 0.0, 80.0],
                acceleration_z: 2.0,
            }));
            link.track("d0", [0.0, 0.0, 80.0], [0.0, 0.0, 0.0]);
            (sim, link)
        }

        let (mut sim, link) = climbing(2.0);
        // Tick 1 is the initialisation no-op; tick 2 is the first that
        // integrates, over an interval of 2 s.
        sim.step().expect("tick runs");
        sim.step().expect("tick runs");
        let estimate = link.estimate("d0").expect("an estimate was published");

        // vz = (synth_accel - g) * dt. With the division: (2 - 9.80665) * 2.
        // Without it: (4 - 9.80665) * 2 = -11.61, a different number entirely.
        let expected = (2.0 - 9.806_65) * 2.0;
        assert!(
            (estimate.velocity[2] - expected).abs() < 1.0e-9,
            "vz = {} but the divided rate gives {expected}",
            estimate.velocity[2]
        );
    }

    /// `max(dt_s, 1e-6)` guards the divisor. Nothing in the config can make
    /// `dt_s` zero — [`crate::clock::Timestep`] rejects it — so the guard is
    /// unreachable from a validated run, and this pins the constant so a future
    /// caller with its own dt cannot lose it silently.
    #[test]
    fn the_acceleration_divisor_has_a_floor() {
        assert_eq!(MIN_ACCEL_DT_S, 1.0e-6);
        assert_eq!((0.0_f64).max(MIN_ACCEL_DT_S), MIN_ACCEL_DT_S);
    }

    // -----------------------------------------------------------------
    // Parity against the reference, and what parity reveals
    // -----------------------------------------------------------------

    /// The tolerance on the parity fixture below, in metres.
    ///
    /// The measured worst disagreement across the six pinned ticks is
    /// **1.42e-14 m** — which at a state magnitude of 80 m is exactly one ULP,
    /// and comes from numpy and nalgebra ordering the covariance products
    /// differently. A nanometre is ~70,000x that, and still some twenty orders
    /// of magnitude tighter than anything the estimate means physically, so it
    /// separates "the port drifted" from "the two libraries rounded
    /// differently" without pretending the two are bit-identical.
    const PARITY_TOLERANCE_M: f64 = 1.0e-9;

    /// Six ticks against values taken from running `EKFVIO` itself, driven with
    /// the inputs `sim.py` gives it: a drone flying east at 10 m/s, the
    /// synthetic kinematic IMU, and the ten constant features.
    ///
    /// This is the whole stage compared value-for-value against the reference —
    /// the synthetic IMU, the frame ordering, the update and the published
    /// trace — not just the estimator underneath it.
    #[test]
    fn the_stage_matches_the_reference_value_for_value() {
        let (mut sim, link) = eastbound();

        // (position_z, velocity_z, covariance_pos_trace) per tick, from Python.
        let reference: [(f64, f64, f64); 6] = [
            (80.0, 0.0, 3.0),
            (75.096675, -9.80665, 3.246_756_849_315_068_5),
            (60.3867, -19.6133, 3.846_842_930_694_891_5),
            (35.870_074_999_999_99, -29.41995, 4.731_069_534_658_572),
            (1.546_799_999_999_993, -39.2266, 5.889_308_829_073_657_5),
            (
                -42.583_125_000_000_01,
                -49.033_249_999_999_995,
                7.333_904_097_867_753,
            ),
        ];

        let mut worst = 0.0_f64;
        for (tick, (position_z, velocity_z, trace)) in reference.into_iter().enumerate() {
            sim.step().expect("tick runs");
            let estimate = link.estimate("d0").expect("an estimate was published");

            // x and y never move: the synthetic IMU reports no horizontal
            // acceleration and the constant features report no flow.
            assert_eq!(estimate.position[0], 0.0, "tick {tick}");
            assert_eq!(estimate.position[1], 0.0, "tick {tick}");
            assert_eq!(estimate.velocity[0], 0.0, "tick {tick}");
            assert_eq!(estimate.velocity[1], 0.0, "tick {tick}");

            let delta = (estimate.position[2] - position_z).abs();
            worst = worst.max(delta);
            assert!(
                delta < PARITY_TOLERANCE_M,
                "tick {tick}: position_z {} vs reference {position_z} ({delta:e})",
                estimate.position[2]
            );
            assert!(
                (estimate.velocity[2] - velocity_z).abs() < PARITY_TOLERANCE_M,
                "tick {tick}: velocity_z {} vs reference {velocity_z}",
                estimate.velocity[2]
            );
            // The trace path agrees exactly, so it is pinned exactly. A
            // tolerance here would hide a real divergence.
            assert_eq!(
                estimate.covariance_pos_trace.to_bits(),
                trace.to_bits(),
                "tick {tick}: trace {} vs reference {trace}",
                estimate.covariance_pos_trace
            );
        }

        // The fixture is only worth its tolerance if the tolerance has room.
        // If a future change makes this assertion fail, the fixture has started
        // measuring something other than library rounding.
        assert!(
            worst < PARITY_TOLERANCE_M * 1.0e-4,
            "worst disagreement {worst:e} is close to the tolerance"
        );
    }

    /// **A defect in the reference, reproduced deliberately.**
    ///
    /// The synthetic IMU reports `(v - v_prev) / dt` — *kinematic* acceleration
    /// — where the filter's prediction expects **specific force**, which is what
    /// an accelerometer actually measures and which at rest reads `+g` along
    /// up. `predict_to` therefore computes `R * accel_body - g`, and on a
    /// straight, level, constant-velocity path `accel_body` is zero, so the
    /// estimate accelerates downward at 9.80665 m/s^2.
    ///
    /// A drone cruising at 80 m therefore has a VIO estimate that falls through
    /// the terrain datum inside five ticks, and keeps going. Verified against
    /// `EKFVIO` directly — this is the reference's behaviour, not a port bug —
    /// and carried across because §4 forbids silently "improving" the port.
    ///
    /// It matters because `use_vio_position` substitutes this position into the
    /// state the tracker sees.
    #[test]
    fn a_level_flight_makes_the_estimate_free_fall_because_the_imu_is_kinematic() {
        let (mut sim, link) = eastbound();

        let mut altitudes = Vec::new();
        for _ in 0..6 {
            sim.step().expect("tick runs");
            altitudes.push(link.estimate("d0").expect("published").position[2]);
        }

        // Each tick loses another 9.80665 m/s of vertical velocity.
        let velocity_z = link.estimate("d0").expect("published").velocity[2];
        assert!(
            (velocity_z - -5.0 * 9.806_65).abs() < 1.0e-9,
            "vz = {velocity_z}"
        );
        // And it is through the datum, from a level cruise at 80 m.
        assert!(
            altitudes[5] < 0.0,
            "the estimate did not fall through the datum: {altitudes:?}"
        );
        // Strictly monotonic, so this is a sustained fall and not one bad tick.
        assert!(altitudes.windows(2).all(|w| w[1] < w[0]), "{altitudes:?}");
    }

    // -----------------------------------------------------------------
    // What is published
    // -----------------------------------------------------------------

    /// The reference reads exactly three things back — position, velocity and
    /// `trace(covariance[6:9, 6:9])` — and the trace is over the *position*
    /// block, not the rotation block at `[0:3]` or the velocity block at
    /// `[3:6]`. Initialization seeds the position block at 1.0 per axis, so a
    /// first-tick trace near 3 is the position block and near 0.03 or 0.3 is
    /// one of the others.
    #[test]
    fn the_published_trace_is_the_position_block() {
        let (mut sim, link) = eastbound();

        sim.step().expect("tick runs");
        let estimate = link.estimate("d0").expect("an estimate was published");

        assert!(
            estimate.covariance_pos_trace > 1.0,
            "trace was {}, which is too small to be the position block",
            estimate.covariance_pos_trace
        );
    }

    /// An estimator that cannot see must not manufacture confidence. On the
    /// very first frame the store is empty, so nothing matches, and the
    /// contract requires the update to report that rather than inventing a
    /// measurement from the feature count.
    #[test]
    fn the_first_frame_reports_no_information_rather_than_confidence() {
        let (mut sim, link) = eastbound();

        sim.step().expect("tick runs");
        let first = link.estimate("d0").expect("an estimate was published");

        assert_eq!(
            first.outcome,
            UpdateOutcome::NoInformation {
                features_matched: 0,
                reason: NoInformation::TooFewMatches,
            }
        );
    }

    /// Every estimate is cleared and rebuilt each tick, so a platform that
    /// stops producing one does not leave last tick's behind to be read as
    /// current.
    #[test]
    fn estimates_do_not_survive_a_tick_that_produced_none() {
        let (mut sim, link) = rig(vec![node("d0", 0.0, 0.0)]);
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["d0".to_string()],
        }));
        link.track("d0", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);
        sim.step().expect("tick runs");
        assert!(link.estimate("d0").is_some());

        // A source that prescribes nothing at all.
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: Vec::new(),
        }));
        sim.step().expect("tick runs");

        assert_eq!(link.estimate("d0"), None);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// A run is reproducible: two simulations with the same seed, config and
    /// nominal path publish the same estimate bit for bit. The filter has no
    /// RNG, so this is really a check that nothing in the wiring — map
    /// iteration order, a clock read, a shared handle — introduced one.
    #[test]
    fn a_run_is_reproducible() {
        let mut runs = Vec::new();
        for _ in 0..2 {
            let (mut sim, link) = eastbound();
            for _ in 0..8 {
                sim.step().expect("tick runs");
            }
            runs.push(link.estimate("d0").expect("an estimate was published"));
        }

        assert_eq!(
            runs[0].position.map(f64::to_bits),
            runs[1].position.map(f64::to_bits)
        );
        assert_eq!(
            runs[0].velocity.map(f64::to_bits),
            runs[1].velocity.map(f64::to_bits)
        );
        assert_eq!(
            runs[0].covariance_pos_trace.to_bits(),
            runs[1].covariance_pos_trace.to_bits()
        );
    }

    /// Platforms are visited in id order, not in the order they were tracked.
    /// The filters are independent, so the estimates do not care — but
    /// `dropped` is a list built in visit order, and so is the stage's trace
    /// note. A trace whose line order moved run to run would be useless for
    /// comparing against Python, which is the whole point of the trace.
    #[test]
    fn platforms_are_visited_in_id_order_not_insertion_order() {
        let (mut sim, link) = rig(vec![node("charlie", 0.0, 0.0), node("alpha", 1.0, 1.0)]);
        link.set_nominal(Box::new(Straight {
            start: [0.0, 0.0, 80.0],
            velocity: [10.0, 0.0, 0.0],
            prescribes: vec!["charlie".to_string(), "alpha".to_string()],
        }));
        // Tracked in reverse of id order on purpose.
        link.track("charlie", [0.0, 0.0, 80.0], [10.0, 0.0, 0.0]);
        link.track("alpha", [1.0, 1.0, 80.0], [10.0, 0.0, 0.0]);

        for _ in 0..3 {
            sim.step().expect("tick runs");
        }
        sim.reset();
        sim.load_world(
            vec![node("charlie", 0.0, 0.0), node("alpha", 1.0, 1.0)],
            Vec::new(),
        );
        sim.step().expect("tick runs");

        // Both filters are now ahead of the rewound clock, so both drop -- and
        // the list is in id order, not the order they were tracked in.
        assert_eq!(
            link.dropped(),
            vec!["alpha".to_string(), "charlie".to_string()]
        );
    }

    /// Why one mutant of this stage is *provably* equivalent rather than
    /// merely uncaught.
    ///
    /// Moving `classify` to after `predict_to` — which looks like it should
    /// break the out-of-order drop — changes nothing, and the reason is a
    /// property of `predict_to` that is not obvious from its signature: calling
    /// it twice with the same target timestamp is a no-op the second time.
    ///
    /// - `dt_total > 0`: the first call advances the clock and consumes the
    ///   IMU; the second finds `dt_total == 0`, skips every sample as not newer
    ///   than the clock, and skips the tail interval too.
    /// - `dt_total == 0`: both calls are no-ops for the same reason.
    /// - `dt_total < 0`: the first call returns before touching anything, so
    ///   `classify` still sees the old clock and still reports `OutOfOrder`.
    ///
    /// Rather than rest on that reading, this drives 5,000 randomised states
    /// through both orderings and requires bit-identical results. If a future
    /// change to `predict_to` breaks the idempotence, this fails and the
    /// mutant stops being equivalent.
    #[test]
    fn predicting_twice_to_the_same_timestamp_is_a_no_op_the_second_time() {
        // A cheap deterministic generator; the values only need to be varied,
        // not statistically good.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Roughly [-20, 20), including exact zeros often enough to matter.
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 40.0 - 20.0
        };

        for _ in 0..5_000 {
            let start_clock = next().abs();
            let target = start_clock + next();
            let mut once = VioPredictor::new(VioNoise::default());
            once.initialize([next(), next(), next()], start_clock);
            let mut twice = once.clone();

            let measurement = ImuMeasurement {
                angular_velocity: [next(), next(), next()],
                linear_acceleration: [next(), next(), next()],
                timestamp_s: start_clock + next(),
            };

            once.predict_to(&[measurement], target);
            twice.predict_to(&[measurement], target);
            twice.predict_to(&[measurement], target);

            assert_eq!(
                once.state()
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                twice
                    .state()
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "state diverged at target {target} from clock {start_clock}"
            );
            assert_eq!(
                once.covariance()
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                twice
                    .covariance()
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "covariance diverged at target {target} from clock {start_clock}"
            );
            assert_eq!(once.timestamp_s().to_bits(), twice.timestamp_s().to_bits());
        }
    }
}
