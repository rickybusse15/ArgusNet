//! Stage 6: generating the observation batch.
//!
//! §4 names stage 6 *"generate the observation batch, preserving rejection
//! stages"*. The cascade itself landed in P4.5b and the error model in P4.5a;
//! this is the loop around them — enumerating `(node, target)` pairs, building
//! each [`PairInput`] from the world and the truth, opening the per-pair random
//! stream, and turning an acceptance into a [`BearingObservation`].
//!
//! # Sensor parameters are injected, not read off `NodeState`
//!
//! The reference reads `node.max_range_m`, `node.fov_half_angle_deg`,
//! `node.dropout_probability` and `node.sensor_type` off its scenario node.
//! `argusnet_types::NodeState` carries none of those, and widening it means a
//! proto change that has to move together with the Python bindings,
//! `argusnet-proto`, `argusnet-core` and `argusnet-server`.
//!
//! That widening is still wanted — the viewer cannot draw a sensor cone it
//! cannot see — but it is a *replay and transport* concern, not a prerequisite
//! for generating observations. [`PairInput`] already takes `max_range_m` and
//! the FOV as plain values, so this stage takes a [`SensorSpec`] table the same
//! way stage 5 takes its nominal trajectory. Stage 6 is therefore not blocked on
//! the proto, and a first reading that said it was is corrected here.
//!
//! # Line of sight is injected too
//!
//! Visibility is a [`VisibilityOracle`], defaulting to [`AlwaysVisible`]. The
//! two-level DDA lives in `argusnet-world` and wiring it in pulls terrain,
//! obstacles and land cover into this crate's dependency graph for one call.
//! The cascade already models visibility as an input rather than something it
//! computes, so honouring that boundary costs nothing and keeps the slice
//! reviewable.
//!
//! **`AlwaysVisible` is not a stand-in for the real thing.** A run using it
//! generates observations through terrain, and
//! [`ObserveLink::visibility_is_modelled`] reports whether an oracle was
//! installed so a caller cannot mistake one for the other.
//!
//! # The per-pair stream is opened *after* the draw-free gates
//!
//! ```python
//! # Per-pair RNG streams are seeded independently, so deferring
//! # construction past the draw-free rejects above is draw-order safe.
//! ```
//!
//! Range and FOV consume no draws, so where the stream is opened cannot change
//! any value — the reference says so itself. It is reproduced anyway, because
//! "this is safe *because* the streams are independent" is exactly the kind of
//! reasoning that stops being true when someone later threads one generator
//! through the loop, and the shape is the reminder.
//!
//! # What this slice defers
//!
//! Three quality modifiers are computed by the reference and are **explicitly
//! 1.0 here**, named rather than silently absent:
//!
//! - FOV-edge degradation (noise up, confidence down near the boundary),
//! - altitude falloff above 200 m AGL, which needs a terrain sampler, and
//! - the bearing bias random walk from P4.5a.
//!
//! The thermal duplicate observation a `thermal`/`lwir`/`mwir`/`ir` node emits
//! is also not here. All four are stage 6b.
//!
//! Every deferred factor is neutral for an omnidirectional, low-altitude,
//! drift-free sensor, so a run configured that way is already exact — and
//! `the_deferred_quality_modifiers_are_all_neutral` pins that they are 1.0
//! rather than forgotten.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_sensing::cascade::{
    evaluate_pair, DominantLoss, FovInput, GenerationTally, Outcome, PairInput, Visibility,
};
use argusnet_sensing::{atmospheric_attenuation, detection_probability, SensorErrorConfig};
use argusnet_types::{BearingObservation, Vector3};

use crate::deterministic::{DeterministicRng, SeedPart};
use crate::fusion::FusionLink;
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// `confidence_floor` — the floor on a published observation's confidence.
pub const CONFIDENCE_FLOOR: f64 = 0.15;

/// `confidence_range_slope` — how fast confidence falls across the sensor's
/// range.
pub const CONFIDENCE_RANGE_SLOPE: f64 = 0.7;

/// Below this speed a mobile sensor has no resolvable look direction, so the
/// FOV gate is skipped rather than applied against a degenerate heading.
pub const MIN_SPEED_FOR_LOOK_DIRECTION_MPS: f64 = 1.0e-6;

/// What this stage needs to know about one sensor.
///
/// The reference reads these off its scenario node; see the module docs for why
/// they are injected instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSpec {
    pub max_range_m: f64,
    /// `None` for an omnidirectional sensor. The cascade skips the FOV gate.
    pub fov_half_angle_deg: Option<f64>,
    pub required_elevation_deg: f64,
    pub dropout_probability: f64,
    /// The sensor's bearing noise at the reference range, before the
    /// range-dependent scaling.
    pub base_bearing_std_rad: f64,
    pub error: SensorErrorConfig,
}

impl Default for SensorSpec {
    fn default() -> Self {
        Self {
            max_range_m: 400.0,
            fov_half_angle_deg: None,
            required_elevation_deg: -90.0,
            dropout_probability: 0.0,
            base_bearing_std_rad: 0.015,
            error: SensorErrorConfig::default(),
        }
    }
}

/// Whether the path from a sensor to a target is clear.
pub trait VisibilityOracle: fmt::Debug + Send {
    fn visibility(&self, from: [f64; 3], to: [f64; 3]) -> Visibility;
}

/// The default oracle: everything is visible.
///
/// A run using this generates observations **through terrain**. It is the right
/// default for a slice that does not own line of sight, and
/// [`ObserveLink::visibility_is_modelled`] is how a caller tells the two apart.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysVisible;

impl VisibilityOracle for AlwaysVisible {
    fn visibility(&self, _from: [f64; 3], _to: [f64; 3]) -> Visibility {
        Visibility::Visible
    }
}

/// One pair's outcome, kept for the trace and for tests.
#[derive(Debug, Clone, PartialEq)]
pub struct PairRecord {
    pub node_id: String,
    pub target_id: String,
    pub range_m: f64,
    pub outcome: Outcome,
}

#[derive(Debug)]
struct Shared {
    sensors: BTreeMap<String, SensorSpec>,
    oracle: Box<dyn VisibilityOracle>,
    oracle_installed: bool,
    observations: Vec<BearingObservation>,
    tally: GenerationTally,
    records: Vec<PairRecord>,
    fusion: Option<FusionLink>,
}

/// A handle on the sensor table and the last batch.
#[derive(Debug, Clone)]
pub struct ObserveLink {
    inner: Arc<Mutex<Shared>>,
}

impl ObserveLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Give `node_id` a sensor. A platform with no spec generates nothing.
    pub fn install_sensor(&self, node_id: &str, spec: SensorSpec) {
        self.lock().sensors.insert(node_id.to_string(), spec);
    }

    pub fn remove_sensor(&self, node_id: &str) {
        self.lock().sensors.remove(node_id);
    }

    pub fn sensor(&self, node_id: &str) -> Option<SensorSpec> {
        self.lock().sensors.get(node_id).copied()
    }

    pub fn set_visibility_oracle(&self, oracle: Box<dyn VisibilityOracle>) {
        let mut shared = self.lock();
        shared.oracle = oracle;
        shared.oracle_installed = true;
    }

    /// Whether line of sight is actually being modelled, or whether the stage
    /// is running on [`AlwaysVisible`] and generating observations through
    /// terrain.
    pub fn visibility_is_modelled(&self) -> bool {
        self.lock().oracle_installed
    }

    /// Hand each tick's batch straight to stage 7.
    pub fn feed_fusion(&self, fusion: FusionLink) {
        self.lock().fusion = Some(fusion);
    }

    pub fn observations(&self) -> Vec<BearingObservation> {
        self.lock().observations.clone()
    }

    pub fn tally(&self) -> GenerationTally {
        self.lock().tally.clone()
    }

    /// Every pair considered on the last run, in `(node, target)` order.
    pub fn records(&self) -> Vec<PairRecord> {
        self.lock().records.clone()
    }
}

/// [`Stage::GenerateObservations`] as a [`StageHandler`].
///
/// [`Stage::GenerateObservations`]: crate::stage::Stage::GenerateObservations
#[derive(Debug)]
pub struct ObserveStage {
    link: ObserveLink,
}

impl Default for ObserveStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ObserveStage {
    pub fn new() -> Self {
        Self {
            link: ObserveLink {
                inner: Arc::new(Mutex::new(Shared {
                    sensors: BTreeMap::new(),
                    oracle: Box::new(AlwaysVisible),
                    oracle_installed: false,
                    observations: Vec::new(),
                    tally: GenerationTally::default(),
                    records: Vec::new(),
                    fusion: None,
                })),
            },
        }
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> ObserveLink {
        self.link.clone()
    }
}

/// `direction + perturbation`, renormalised — the arithmetic half of
/// `noisy_unit_vector`.
///
/// The perturbation is an argument rather than drawn inside, for the reason
/// `argusnet_sensing::BearingBiasDrift` gives for its own increment: it makes
/// this a pure function and therefore checkable value-for-value. It also makes
/// the degenerate branch reachable from a test at all — a perturbation that
/// almost exactly cancels the unit direction is not something a random stream
/// will produce on demand.
///
/// That branch returns the **unperturbed** direction rather than a zero vector,
/// so a pathological draw cannot publish an observation with no direction.
fn perturb_unit_vector(direction: Vector3<f64>, perturbation: Vector3<f64>) -> Vector3<f64> {
    let unit = direction.normalize();
    let candidate = unit + perturbation;
    let norm = candidate.norm();
    if norm < 1e-12 {
        return unit;
    }
    candidate / norm
}

/// Three independent normal draws, in x, y, z order — the order numpy fills
/// `size=3`, and therefore the order the stream must be consumed in.
fn bearing_perturbation(noise_std_rad: f64, rng: &mut DeterministicRng) -> Vector3<f64> {
    let x = rng.normal(0.0, noise_std_rad);
    let y = rng.normal(0.0, noise_std_rad);
    let z = rng.normal(0.0, noise_std_rad);
    Vector3::new(x, y, z)
}

impl StageHandler for ObserveStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();
        shared.observations.clear();
        shared.records.clear();
        shared.tally = GenerationTally::default();

        // Stage 6 is one of the two stages §3 admits to truth, so `None` here
        // means the capability was withheld -- a wiring fault, not an empty
        // world, and reporting NoWork would hide it.
        let Some(truths) = context.truth else {
            return Err(StageError::Invariant(
                "stage 6 was not handed the truth capability".to_string(),
            ));
        };
        if shared.sensors.is_empty() || truths.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let seed = context.seed;
        let seed_time_key = context.seed_time_key;
        let timestamp_s = context.timestamp_s;
        // Truth is copied out so the platform borrow below is free to be
        // mutable-free; the vector is one entry per target.
        let targets: Vec<(String, Vector3<f64>)> = truths
            .iter()
            .map(|truth| (truth.target_id.clone(), truth.position))
            .collect();

        // Sensors in id order, targets in the order truth arrived. The pair
        // order decides nothing -- each pair has its own stream -- but it
        // decides the trace, and a trace whose order moved run to run would be
        // useless for comparing against Python.
        let specs: Vec<(String, SensorSpec)> = shared
            .sensors
            .iter()
            .map(|(id, spec)| (id.clone(), *spec))
            .collect();

        for (node_id, spec) in specs {
            let Some(platform) = context.world.platform(&node_id) else {
                // A sensor installed for a platform that is not in the world.
                // The reference iterates its scenario nodes, so this cannot
                // arise there; skipping is the reading that cannot invent a
                // sensor position.
                continue;
            };
            let origin = platform.position;
            let velocity = platform.velocity;

            for (target_id, target_position) in &targets {
                let line_of_sight = target_position - origin;
                let range_m = line_of_sight.norm();

                // Elevation above the horizontal, in degrees. Guarded against a
                // zero-length sight line, which the range gate rejects anyway
                // but which would produce a NaN before it got there.
                let elevation_deg = if range_m > 0.0 {
                    (line_of_sight.z / range_m).asin().to_degrees()
                } else {
                    0.0
                };

                // The FOV gate needs a look direction. A sensor that is not
                // moving has none, and the cascade's contract is to skip the
                // gate rather than treat that as a rejection.
                let fov = spec.fov_half_angle_deg.and_then(|half_angle_deg| {
                    let speed = velocity.norm();
                    if speed <= MIN_SPEED_FOR_LOOK_DIRECTION_MPS || range_m <= 0.0 {
                        return None;
                    }
                    let cosine =
                        (velocity.dot(&line_of_sight) / (speed * range_m)).clamp(-1.0, 1.0);
                    Some(FovInput {
                        angle_deg: cosine.acos().to_degrees(),
                        half_angle_deg,
                    })
                });

                let transmittance = atmospheric_attenuation(range_m, &spec.error);
                let combined_pd = detection_probability(range_m, &spec.error, transmittance);
                // Which factor did the most damage. The reference names
                // vegetation when transmittance dominates and reports a plain
                // dropout otherwise.
                let dominant_loss = if transmittance < 1.0 && transmittance <= combined_pd {
                    DominantLoss::Transmittance
                } else {
                    DominantLoss::Other
                };

                let visibility = shared.oracle.visibility(
                    [origin.x, origin.y, origin.z],
                    [target_position.x, target_position.y, target_position.z],
                );

                let input = PairInput {
                    range_m,
                    max_range_m: spec.max_range_m,
                    fov,
                    dropout_probability: spec.dropout_probability,
                    elevation_deg,
                    required_elevation_deg: spec.required_elevation_deg,
                    visibility,
                    combined_pd,
                    dominant_loss,
                };

                // Opened here, after the draw-free gates, exactly as the
                // reference defers it. See the module docs.
                let mut pair_rng = DeterministicRng::keyed(
                    seed,
                    &[
                        SeedPart::Str(&node_id),
                        SeedPart::Str(target_id),
                        SeedPart::Int(seed_time_key),
                    ],
                );
                let outcome = {
                    let mut uniform = || pair_rng.uniform();
                    evaluate_pair(&input, &mut uniform)
                };
                shared.tally.record(&node_id, target_id, outcome);
                shared.records.push(PairRecord {
                    node_id: node_id.clone(),
                    target_id: target_id.clone(),
                    range_m,
                    outcome,
                });

                let Outcome::Accepted { fov_fraction: _ } = outcome else {
                    continue;
                };

                // Deferred to stage 6b, and neutral here. Named as constants
                // rather than left out so 6b has a landing spot and a reader
                // can see what is missing.
                let fov_noise_multiplier = 1.0;
                let altitude_noise_multiplier = 1.0;
                let fov_confidence_factor = 1.0;
                let altitude_confidence_factor = 1.0;

                let effective_bearing_std = argusnet_sensing::range_dependent_bearing_noise(
                    spec.base_bearing_std_rad,
                    range_m,
                    &spec.error,
                ) * fov_noise_multiplier
                    * altitude_noise_multiplier;

                let perturbation = bearing_perturbation(effective_bearing_std, &mut pair_rng);
                let direction = perturb_unit_vector(line_of_sight, perturbation);

                let confidence = CONFIDENCE_FLOOR.max(
                    (1.0 - (range_m / spec.max_range_m) * CONFIDENCE_RANGE_SLOPE)
                        * combined_pd
                        * fov_confidence_factor
                        * altitude_confidence_factor,
                );

                shared.observations.push(BearingObservation {
                    node_id: node_id.clone(),
                    target_id: target_id.clone(),
                    origin,
                    direction,
                    bearing_std_rad: effective_bearing_std,
                    timestamp_s,
                    confidence,
                });
            }
        }

        let accepted = shared.observations.len();
        let attempted = shared.tally.attempted;
        if let Some(fusion) = shared.fusion.clone() {
            fusion.submit_observations(shared.observations.clone());
        }
        drop(shared);

        context.note(format!("generated {accepted} of {attempted} attempted"));
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::fusion::FusionStage;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_sensing::cascade::{Blocker, RejectReason};

    use argusnet_types::{NodeState, TruthState};

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 11,
            dt_s: 1.0,
            steps: Some(2_000),
            run_id: "observe".to_string(),
        })
        .expect("config is valid")
    }

    fn sensor_node(id: &str, x: f64, y: f64, mobile: bool) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, y, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: mobile,
            timestamp_s: 0.0,
            health: 1.0,
        }
    }

    fn target(id: &str, x: f64, y: f64, z: f64) -> TruthState {
        TruthState {
            target_id: id.to_string(),
            position: Vector3::new(x, y, z),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            timestamp_s: 0.0,
        }
    }

    /// A sensor that always sees, at close range, with no dropout — so the only
    /// gate that can bite is the one a test sets out to exercise.
    fn permissive() -> SensorSpec {
        SensorSpec {
            max_range_m: 10_000.0,
            error: SensorErrorConfig {
                // Detection is flat at 1.0 out to the knee; put the knee past
                // every range these tests use.
                detection_range_knee_m: 100_000.0,
                atmospheric_attenuation_coeff: 0.0,
                ..SensorErrorConfig::default()
            },
            ..SensorSpec::default()
        }
    }

    fn rig(platforms: Vec<NodeState>, truths: Vec<TruthState>) -> (Simulation, ObserveLink) {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        sim.set_truth(truths);
        let stage = ObserveStage::new();
        let link = stage.link();
        sim.set_stage_handler(Stage::GenerateObservations, Box::new(stage))
            .expect("stage 6 is not a transaction boundary");
        (sim, link)
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::GenerateObservations)
            .expect("stage 6 ran")
            .outcome
            .clone()
    }

    fn reason(link: &ObserveLink, node_id: &str, target_id: &str) -> Option<RejectReason> {
        link.records()
            .into_iter()
            .find(|r| r.node_id == node_id && r.target_id == target_id)
            .and_then(|r| match r.outcome {
                Outcome::Rejected(reason) => Some(reason),
                Outcome::Accepted { .. } => None,
            })
    }

    // -----------------------------------------------------------------
    // The loop
    // -----------------------------------------------------------------

    /// Every sensor is paired with every target, and an accepted pair becomes
    /// an observation whose origin is the sensor's own position.
    #[test]
    fn every_sensor_is_paired_with_every_target() {
        let (mut sim, link) = rig(
            vec![
                sensor_node("s0", 0.0, 0.0, false),
                sensor_node("s1", 100.0, 0.0, false),
            ],
            vec![
                target("t0", 50.0, 50.0, 30.0),
                target("t1", 60.0, 10.0, 20.0),
            ],
        );
        link.install_sensor("s0", permissive());
        link.install_sensor("s1", permissive());

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert_eq!(link.records().len(), 4);
        assert_eq!(link.tally().attempted, 4);
        let observations = link.observations();
        assert_eq!(observations.len(), 4);
        for observation in &observations {
            let expected = if observation.node_id == "s0" {
                0.0
            } else {
                100.0
            };
            assert_eq!(observation.origin.x, expected);
        }
    }

    /// A platform with no sensor installed generates nothing — it is a drone,
    /// not an observer.
    #[test]
    fn a_platform_without_a_sensor_generates_nothing() {
        let (mut sim, link) = rig(
            vec![
                sensor_node("eyes", 0.0, 0.0, false),
                sensor_node("blind", 10.0, 0.0, false),
            ],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        link.install_sensor("eyes", permissive());

        sim.step().expect("tick runs");

        assert_eq!(link.records().len(), 1);
        assert_eq!(link.observations().len(), 1);
        assert_eq!(link.observations()[0].node_id, "eyes");
    }

    /// A sensor installed for a platform that is not in the world is skipped
    /// rather than invented a position for.
    #[test]
    fn a_sensor_for_an_absent_platform_is_skipped() {
        let (mut sim, link) = rig(
            vec![sensor_node("present", 0.0, 0.0, false)],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        link.install_sensor("present", permissive());
        link.install_sensor("ghost", permissive());

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert_eq!(link.records().len(), 1);
        assert_eq!(link.tally().attempted, 1);
    }

    #[test]
    fn with_no_sensors_or_no_targets_there_is_no_work() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        sim.step().expect("tick runs");
        assert_eq!(outcome(&sim), StageOutcome::NoWork);

        let (mut empty, empty_link) = rig(vec![sensor_node("s0", 0.0, 0.0, false)], Vec::new());
        empty_link.install_sensor("s0", permissive());
        empty.step().expect("tick runs");
        assert_eq!(outcome(&empty), StageOutcome::NoWork);
        let _ = link;
    }

    // -----------------------------------------------------------------
    // The gates, through the real stage
    // -----------------------------------------------------------------

    /// The range gate, reached through the stage rather than by calling the
    /// cascade directly — so the geometry this stage computes is what is being
    /// tested, not the cascade's arithmetic, which P4.5b already pinned.
    #[test]
    fn a_target_past_the_sensors_range_is_rejected() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("near", 100.0, 0.0, 0.0),
                target("far", 5_000.0, 0.0, 0.0),
            ],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                max_range_m: 1_000.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(&link, "s0", "far"), Some(RejectReason::OutOfRange));
        assert_eq!(reason(&link, "s0", "near"), None);
    }

    /// Elevation is `asin(dz / slant_range)` in degrees — the angle above the
    /// horizontal — and the two plausible wrong answers are *the raw height
    /// difference* and *`asin(dz / horizontal_range)`*.
    ///
    /// Picking a threshold where all three disagree took some care, because at
    /// most gate settings they agree by accident:
    ///
    /// | target | true angle | raw `dz` | `dz`/horizontal |
    /// |---|---|---|---|
    /// | `(100, 0, 100)` | 45° | 100 | 90° |
    /// | `(10, 0, 20)` | 63.4° | 20 | `asin(2)` = NaN |
    ///
    /// At a 60° gate the true answer **rejects** the first and **accepts** the
    /// second, and each wrong answer gets at least one of them backwards — the
    /// NaN case because `NaN < 60.0` is false, so it accepts by default.
    #[test]
    fn elevation_is_the_angle_above_horizontal_not_the_height_or_the_horizontal_ratio() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("shallow", 100.0, 0.0, 100.0),
                target("steep", 10.0, 0.0, 20.0),
            ],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                required_elevation_deg: 60.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        // 45 degrees, below the gate.
        assert_eq!(
            reason(&link, "s0", "shallow"),
            Some(RejectReason::LowElevation)
        );
        // 63.4 degrees, above it.
        assert_eq!(reason(&link, "s0", "steep"), None);
    }

    /// A directional sensor gates on the angle between its look direction — its
    /// velocity — and the sight line.
    #[test]
    fn a_directional_sensor_gates_on_the_angle_to_its_heading() {
        let mut platform = sensor_node("s0", 0.0, 0.0, true);
        platform.velocity = Vector3::new(10.0, 0.0, 0.0);
        let (mut sim, link) = rig(
            vec![platform],
            vec![
                target("ahead", 100.0, 0.0, 0.0),
                target("behind", -100.0, 0.0, 0.0),
            ],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: Some(45.0),
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(&link, "s0", "ahead"), None);
        assert_eq!(
            reason(&link, "s0", "behind"),
            Some(RejectReason::OutsideFov)
        );
    }

    /// A directional sensor that is not moving has no resolvable look
    /// direction, and the cascade's contract is to **skip** the FOV gate rather
    /// than treat that as a rejection. Without the skip, a hovering drone would
    /// see nothing at all.
    #[test]
    fn a_stationary_directional_sensor_skips_the_fov_gate_rather_than_rejecting() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, true)],
            vec![target("behind", -100.0, 0.0, 0.0)],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: Some(1.0),
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(&link, "s0", "behind"), None);
        assert_eq!(link.observations().len(), 1);
    }

    /// Dropout is a probability, and the draw comes from the pair's own stream.
    #[test]
    fn a_certain_dropout_rejects_every_pair() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                dropout_probability: 1.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(&link, "s0", "t0"), Some(RejectReason::Dropout));
        assert!(link.observations().is_empty());
        assert_eq!(link.tally().rejected(), 1);
    }

    // -----------------------------------------------------------------
    // Visibility
    // -----------------------------------------------------------------

    #[derive(Debug)]
    struct NeverVisible;

    impl VisibilityOracle for NeverVisible {
        fn visibility(&self, _from: [f64; 3], _to: [f64; 3]) -> Visibility {
            Visibility::Blocked(Blocker::Terrain)
        }
    }

    /// The oracle's verdict reaches the cascade.
    #[test]
    fn an_installed_oracle_can_block_every_pair() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        link.install_sensor("s0", permissive());
        link.set_visibility_oracle(Box::new(NeverVisible));

        sim.step().expect("tick runs");

        // `Blocker::reason` is private, so this names the reason the terrain
        // blocker maps to rather than asking the blocker for it.
        assert_eq!(
            reason(&link, "s0", "t0"),
            Some(RejectReason::TerrainOcclusion)
        );
        assert!(link.observations().is_empty());
    }

    /// The default oracle sees through terrain, and the link says so. A caller
    /// that mistook a default run for a line-of-sight-modelled one would be
    /// reading acceptance rates that mean something else entirely.
    #[test]
    fn the_default_oracle_is_reported_as_not_modelling_visibility() {
        let (_sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );

        assert!(!link.visibility_is_modelled());

        link.set_visibility_oracle(Box::new(NeverVisible));
        assert!(link.visibility_is_modelled());
    }

    // -----------------------------------------------------------------
    // Determinism
    // -----------------------------------------------------------------

    /// Each pair has its own stream, keyed on `(node_id, target_id,
    /// round(t * 1000))`. That is what makes the batch independent of the order
    /// pairs are visited in — ADR-006's central property — so two sensors
    /// looking at the same target from the same place still draw differently.
    #[test]
    fn two_sensors_in_the_same_place_draw_different_noise() {
        let (mut sim, link) = rig(
            vec![
                sensor_node("a", 0.0, 0.0, false),
                sensor_node("b", 0.0, 0.0, false),
            ],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        link.install_sensor("a", permissive());
        link.install_sensor("b", permissive());

        sim.step().expect("tick runs");

        let observations = link.observations();
        assert_eq!(observations.len(), 2);
        assert_ne!(observations[0].direction, observations[1].direction);
    }

    /// And the same pair draws differently on successive ticks, because the
    /// millisecond key is part of the seed.
    #[test]
    fn the_same_pair_draws_differently_on_successive_ticks() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        link.install_sensor("s0", permissive());

        sim.step().expect("tick runs");
        let first = link.observations()[0].direction;
        sim.step().expect("tick runs");
        let second = link.observations()[0].direction;

        assert_ne!(first, second);
    }

    /// Two runs with the same seed and world produce the same batch bit for
    /// bit.
    #[test]
    fn a_run_is_reproducible() {
        let mut runs = Vec::new();
        for _ in 0..2 {
            let (mut sim, link) = rig(
                vec![sensor_node("s0", 0.0, 0.0, false)],
                vec![target("t0", 100.0, 50.0, 25.0)],
            );
            link.install_sensor("s0", permissive());
            for _ in 0..4 {
                sim.step().expect("tick runs");
            }
            runs.push(link.observations());
        }

        assert_eq!(runs[0].len(), 1);
        assert_eq!(
            runs[0][0]
                .direction
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            runs[1][0]
                .direction
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            runs[0][0].bearing_std_rad.to_bits(),
            runs[1][0].bearing_std_rad.to_bits()
        );
    }

    // -----------------------------------------------------------------
    // The observation itself
    // -----------------------------------------------------------------

    /// The published direction is the true sight line *perturbed*, not the true
    /// sight line. An observation that carried the exact geometry would make
    /// fusion look perfect while measuring nothing.
    #[test]
    fn the_published_direction_is_perturbed_not_the_exact_sight_line() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        link.install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let observation = &link.observations()[0];
        let exact = Vector3::new(1.0, 0.0, 0.0);
        assert_ne!(observation.direction, exact);
        // …but close to it, and a unit vector.
        assert!((observation.direction.norm() - 1.0).abs() < 1.0e-12);
        assert!(observation.direction.dot(&exact) > 0.9);
    }

    /// Noise grows with range and is clamped at both ends — the lower clamp is
    /// the surprising half, so a very close target is not modelled as
    /// arbitrarily precise.
    #[test]
    fn the_published_noise_grows_with_range_and_holds_its_floor() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("close", 1.0, 0.0, 0.0),
                target("mid", 2_000.0, 0.0, 0.0),
                target("far", 8_000.0, 0.0, 0.0),
            ],
        );
        link.install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let by_target = |id: &str| {
            link.observations()
                .into_iter()
                .find(|o| o.target_id == id)
                .expect("accepted")
                .bearing_std_rad
        };
        // The floor: never below the base value, even at 1 m.
        assert_eq!(by_target("close"), 0.015);
        assert!(by_target("mid") > by_target("close"));
        // The cap.
        assert_eq!(by_target("far"), 0.15);
    }

    /// Confidence falls across the sensor's range and is floored.
    #[test]
    fn confidence_falls_with_range_and_holds_its_floor() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("close", 10.0, 0.0, 0.0),
                target("edge", 990.0, 0.0, 0.0),
            ],
        );
        link.install_sensor(
            "s0",
            SensorSpec {
                max_range_m: 1_000.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        let by_target = |id: &str| {
            link.observations()
                .into_iter()
                .find(|o| o.target_id == id)
                .expect("accepted")
                .confidence
        };
        assert!(by_target("close") > by_target("edge"));
        assert!(by_target("edge") >= CONFIDENCE_FLOOR);
    }

    /// **The deferral, pinned.** Three quality modifiers are stage 6b's, and
    /// they are 1.0 here rather than absent. This test states the arithmetic
    /// that makes a 6a run exact for an omnidirectional, low-altitude,
    /// drift-free sensor — and gives 6b a test to change rather than a gap to
    /// discover.
    #[test]
    fn the_deferred_quality_modifiers_are_all_neutral() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 500.0, 0.0, 0.0)],
        );
        let spec = SensorSpec {
            max_range_m: 1_000.0,
            ..permissive()
        };
        link.install_sensor("s0", spec);

        sim.step().expect("tick runs");

        let observation = &link.observations()[0];
        let range_m = 500.0;
        // No FOV multiplier, no altitude multiplier: the published noise is
        // exactly the range-dependent value.
        assert_eq!(
            observation.bearing_std_rad,
            argusnet_sensing::range_dependent_bearing_noise(
                spec.base_bearing_std_rad,
                range_m,
                &spec.error
            )
        );
        // And the confidence carries no FOV or altitude factor either. `pd` is
        // 1.0 because this spec's detection knee is past every test range.
        let expected = CONFIDENCE_FLOOR
            .max((1.0 - (range_m / spec.max_range_m) * CONFIDENCE_RANGE_SLOPE) * 1.0);
        assert_eq!(observation.confidence, expected);
    }

    // -----------------------------------------------------------------
    // Handing off to stage 7
    // -----------------------------------------------------------------

    /// Stage 6 feeds stage 7 directly, which is the point of doing fusion in
    /// process: the batch never leaves the tick.
    #[test]
    fn the_batch_is_handed_to_the_fusion_stage() {
        let mut sim = simulation();
        sim.load_world(
            vec![
                sensor_node("s0", 0.0, 0.0, false),
                sensor_node("s1", 400.0, 0.0, false),
            ],
            Vec::new(),
        );
        sim.set_truth(vec![target("t0", 200.0, 300.0, 100.0)]);

        let observe = ObserveStage::new();
        let observe_link = observe.link();
        let fusion = FusionStage::new(argusnet_core::TrackerConfig::default()).expect("valid");
        let fusion_link = fusion.link();
        observe_link.feed_fusion(fusion_link.clone());
        observe_link.install_sensor("s0", permissive());
        observe_link.install_sensor("s1", permissive());

        sim.set_stage_handler(Stage::GenerateObservations, Box::new(observe))
            .expect("stage 6 is not a transaction boundary");
        sim.set_stage_handler(Stage::UpdateFusion, Box::new(fusion))
            .expect("stage 7 is not a transaction boundary");

        for _ in 0..5 {
            sim.step().expect("tick runs");
        }

        let frame = fusion_link.frame().expect("a frame was produced");
        assert!(
            frame.observations.len() + frame.rejected_observations.len() > 0,
            "stage 7 saw nothing"
        );
        assert_eq!(
            sim.world().tracks().len(),
            1,
            "tracks: {:?}",
            sim.world().tracks()
        );
    }

    /// An omnidirectional sensor has **no** FOV gate, even while moving. Every
    /// other omnidirectional test here uses a stationary sensor, where the gate
    /// is skipped for a different reason (no look direction) — so a stage that
    /// treated `None` as a default half-angle would pass all of them.
    #[test]
    fn a_moving_omnidirectional_sensor_still_has_no_fov_gate() {
        let mut platform = sensor_node("s0", 0.0, 0.0, true);
        platform.velocity = Vector3::new(10.0, 0.0, 0.0);
        let (mut sim, link) = rig(vec![platform], vec![target("behind", -100.0, 0.0, 0.0)]);
        link.install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: None,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(&link, "s0", "behind"), None);
        assert_eq!(link.observations().len(), 1);
    }

    /// Two targets in the *same place*, seen by one sensor. The geometry is
    /// identical, so the only thing that can separate the two observations is
    /// the target id in the stream key — which is what ADR-006's per-pair
    /// streams are for.
    #[test]
    fn two_targets_in_the_same_place_draw_different_noise() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0), target("t1", 100.0, 0.0, 0.0)],
        );
        link.install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let observations = link.observations();
        assert_eq!(observations.len(), 2);
        assert_ne!(observations[0].direction, observations[1].direction);
    }

    /// A spec whose detection knee is *inside* the test's range, so `pd` is
    /// genuinely below 1 and the two things that multiply into confidence can
    /// be told apart.
    fn short_sighted() -> SensorSpec {
        SensorSpec {
            max_range_m: 2_000.0,
            error: SensorErrorConfig {
                detection_range_knee_m: 100.0,
                detection_range_falloff_m: 200.0,
                atmospheric_attenuation_coeff: 0.0,
                min_detection_probability: 0.0,
                ..SensorErrorConfig::default()
            },
            ..SensorSpec::default()
        }
    }

    /// Confidence carries the detection probability as a factor. Every other
    /// test here uses a spec whose knee is past the target, where `pd` is
    /// exactly 1.0 and dropping the factor changes nothing.
    #[test]
    fn confidence_carries_the_detection_probability() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 300.0, 0.0, 0.0)],
        );
        let spec = short_sighted();
        link.install_sensor("s0", spec);

        sim.step().expect("tick runs");

        let observed = link.observations();
        // The pair may be rejected by the detection draw; run until one is
        // accepted so the assertion is about confidence, not about luck.
        let mut confidence = observed.first().map(|o| o.confidence);
        for _ in 0..50 {
            if confidence.is_some() {
                break;
            }
            sim.step().expect("tick runs");
            confidence = link.observations().first().map(|o| o.confidence);
        }
        let confidence = confidence.expect("some tick accepted the pair");

        let range_m = 300.0;
        let pd = argusnet_sensing::detection_probability(range_m, &spec.error, 1.0);
        assert!(
            pd < 1.0,
            "the spec did not put the target past the knee: pd = {pd}"
        );
        let expected = CONFIDENCE_FLOOR
            .max((1.0 - (range_m / spec.max_range_m) * CONFIDENCE_RANGE_SLOPE) * pd);
        assert_eq!(confidence, expected);
    }

    /// The floor actually binds somewhere. The range slope alone can only drag
    /// confidence to 0.3, so it takes a low `pd` as well — which is why no
    /// other test in this file reaches the floor at all.
    #[test]
    fn the_confidence_floor_binds_for_a_distant_hard_to_detect_target() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 1_900.0, 0.0, 0.0)],
        );
        let spec = short_sighted();
        link.install_sensor("s0", spec);

        let range_m = 1_900.0;
        let pd = argusnet_sensing::detection_probability(range_m, &spec.error, 1.0);
        let unfloored = (1.0 - (range_m / spec.max_range_m) * CONFIDENCE_RANGE_SLOPE) * pd;
        assert!(
            unfloored < CONFIDENCE_FLOOR,
            "the raw confidence {unfloored} is above the floor, so this test proves nothing"
        );

        let mut confidence = None;
        for _ in 0..400 {
            sim.step().expect("tick runs");
            if let Some(observation) = link.observations().first() {
                confidence = Some(observation.confidence);
                break;
            }
        }
        let confidence = confidence.expect("some tick accepted the pair");

        assert_eq!(confidence, CONFIDENCE_FLOOR);
    }

    /// The tally is per tick, not cumulative. A tally that accumulated would
    /// make the acceptance rate drift upward over a run for no reason.
    #[test]
    fn the_tally_counts_this_tick_only() {
        let (mut sim, link) = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0), target("t1", 120.0, 0.0, 0.0)],
        );
        link.install_sensor("s0", permissive());

        sim.step().expect("tick runs");
        assert_eq!(link.tally().attempted, 2);
        assert_eq!(link.records().len(), 2);

        sim.step().expect("tick runs");
        assert_eq!(link.tally().attempted, 2);
        assert_eq!(link.records().len(), 2);
        assert_eq!(link.observations().len(), 2);
    }

    // -----------------------------------------------------------------
    // The perturbation, as a pure function
    // -----------------------------------------------------------------

    /// Three draws, in x, y, z order. A port that took two and left z at zero
    /// would still produce a plausible unit vector — just one confined to a
    /// plane — and nothing about the observation's shape would give it away.
    #[test]
    fn the_perturbation_takes_three_draws_in_xyz_order() {
        let mut rng = DeterministicRng::from_seed(7);
        let expected = Vector3::new(
            rng.normal(0.0, 0.1),
            rng.normal(0.0, 0.1),
            rng.normal(0.0, 0.1),
        );

        let mut same = DeterministicRng::from_seed(7);
        let actual = bearing_perturbation(0.1, &mut same);

        assert_eq!(actual, expected);
        assert_ne!(actual.z, 0.0, "the z draw was not taken");
    }

    /// A perturbation that almost exactly cancels the direction returns the
    /// **unperturbed** direction, not a zero vector. An observation with no
    /// direction would be published as a valid bearing and fused as one.
    ///
    /// This branch is why the perturbation is an argument: a random stream will
    /// not produce the cancelling value on demand, so with the draw inside the
    /// function the branch would be untestable and would sit there unverified.
    #[test]
    fn a_cancelling_perturbation_returns_the_direction_rather_than_zero() {
        let direction = Vector3::new(3.0, 0.0, 4.0);
        let unit = direction.normalize();

        let exact = perturb_unit_vector(direction, -unit);

        assert_eq!(exact, unit);
        assert!((exact.norm() - 1.0).abs() < 1.0e-12);
    }

    /// The ordinary path renormalises, so the published direction is always a
    /// unit vector however large the perturbation was.
    #[test]
    fn the_perturbed_direction_is_always_renormalised() {
        for perturbation in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.3, -0.2, 0.1),
            Vector3::new(50.0, 50.0, 50.0),
        ] {
            let result = perturb_unit_vector(Vector3::new(1.0, 2.0, 3.0), perturbation);
            assert!(
                (result.norm() - 1.0).abs() < 1.0e-12,
                "norm was {} for {perturbation:?}",
                result.norm()
            );
        }
    }

    /// Stage 6 is one of the two stages §3 admits to truth. The `Invariant` it
    /// raises for a missing capability is therefore unreachable through the
    /// pipeline — which is the point: the capability model guarantees the
    /// truth is there rather than this stage hoping for it.
    ///
    /// A mutant that turns that branch into `NoWork` is equivalent for exactly
    /// that reason, and this is what would notice if the list ever changed.
    #[test]
    fn stage_six_is_admitted_to_truth() {
        assert!(Stage::GenerateObservations.may_read_truth());
    }
}
