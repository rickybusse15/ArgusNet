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
//! [`ObserveStage::visibility_is_modelled`] reports whether an oracle was
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

use argusnet_sensing::cascade::{
    evaluate_pair_traced, Blocker, DominantLoss, FovInput, Gate, GenerationTally, Outcome,
    PairInput, RejectReason, Visibility,
};
use argusnet_sensing::{
    atmospheric_attenuation, detection_probability, BearingBiasDrift, SensorErrorConfig,
};
use argusnet_types::{BearingObservation, ObservationRejection, Vector3};

use crate::controller::ControllerStage;
use crate::deterministic::{DeterministicRng, SeedPart};
use crate::observation_follow::AcceptedByNodeTarget;
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{Stage, StageError, StageOutcome};

/// `confidence_floor` — the floor on a published observation's confidence.
pub const CONFIDENCE_FLOOR: f64 = 0.15;

/// `confidence_range_slope` — how fast confidence falls across the sensor's
/// range.
pub const CONFIDENCE_RANGE_SLOPE: f64 = 0.7;

// The look rule is shared with the viewer, which draws sensor volumes along
// the same axis (P2.2); it lives in `argusnet-sensing`.
pub use argusnet_sensing::look::{
    fov_angle_deg, sensor_look_direction, DEFAULT_LOOK_DOWN_ANGLE_DEG, MIN_LOOK_SPEED_XY_MPS,
};

/// Where a sensor is, and how it is moving, at a tick -- when that is not
/// where the world says the platform is.
///
/// The reference's `build_observations` is handed the scenario's nodes and
/// calls `node.state(t)`: observations originate at the **nominal**
/// controller position, before mission overrides and deconfliction move the
/// platform, and a directional sensor looks along the nominal velocity. A run
/// that installs a source reproduces that; one that does not senses from
/// `TickWorld`, as P4.8c did.
pub trait SensorPoseSource: fmt::Debug + Send {
    /// `(position, velocity)` at `timestamp_s`, or `None` for a platform this
    /// source does not prescribe -- which then senses from the world.
    fn pose(&self, node_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])>;
}

/// `fov_noise_multiplier_slope` — how fast noise grows toward the FOV edge.
pub const FOV_NOISE_MULTIPLIER_SLOPE: f64 = 1.5;

/// `fov_confidence_slope` — how fast confidence falls toward the FOV edge.
pub const FOV_CONFIDENCE_SLOPE: f64 = 0.4;

/// `altitude_quality_threshold_m` — AGL above which quality starts to degrade.
pub const ALTITUDE_QUALITY_THRESHOLD_M: f64 = 200.0;

/// `altitude_quality_range_m` — the AGL band over which the degradation ramps
/// to full.
pub const ALTITUDE_QUALITY_RANGE_M: f64 = 400.0;

/// `altitude_noise_multiplier_max` — the extra noise at the top of the ramp.
pub const ALTITUDE_NOISE_MULTIPLIER_MAX: f64 = 1.2;

/// `altitude_confidence_slope`.
pub const ALTITUDE_CONFIDENCE_SLOPE: f64 = 0.5;

/// `altitude_confidence_floor`.
pub const ALTITUDE_CONFIDENCE_FLOOR: f64 = 0.3;

/// Clutter observations carry `target_id = "clutter"`.
pub const CLUTTER_TARGET_ID: &str = "clutter";

/// The caller caps a clutter return's confidence at `confidence_floor * 0.9`,
/// so no false alarm is ever more trusted than the least-trusted real one.
pub const CLUTTER_CONFIDENCE_CAP: f64 = CONFIDENCE_FLOOR * 0.9;

/// The cap really is below the floor a real observation can reach. A compile-
/// time assertion rather than a test, because it is a property of the two
/// constants and nothing at runtime can change it.
const _: () = assert!(CLUTTER_CONFIDENCE_CAP < CONFIDENCE_FLOOR);

/// A clutter return's confidence is drawn uniformly from this band, then
/// capped.
pub const CLUTTER_CONFIDENCE_LOW: f64 = 0.05;
pub const CLUTTER_CONFIDENCE_HIGH: f64 = 0.4;

/// A thermal sensor also emits a second, noisier observation of the same
/// target: `1.3x` the bearing noise.
pub const THERMAL_NOISE_FACTOR: f64 = 1.3;

/// …and `0.9x` the confidence.
pub const THERMAL_CONFIDENCE_FACTOR: f64 = 0.9;

/// Which band a sensor observes in.
///
/// The reference tests `sensor_type in ("thermal", "lwir", "mwir", "ir")`, so
/// the four names are one case rather than four.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SensorBand {
    #[default]
    Optical,
    /// `thermal`, `lwir`, `mwir` or `ir`.
    Thermal,
}

impl SensorBand {
    /// Classify a reference `sensor_type` string.
    pub fn from_sensor_type(sensor_type: &str) -> Self {
        match sensor_type {
            "thermal" | "lwir" | "mwir" | "ir" => SensorBand::Thermal,
            _ => SensorBand::Optical,
        }
    }
}

/// Terrain height under a point, for the AGL the altitude ramp needs.
///
/// Injected for the same reason the visibility oracle is: the sampler lives in
/// `argusnet-world`, and the reference only consults it for one number.
pub trait TerrainHeight: fmt::Debug + Send + Sync {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64;
}

/// Terrain at the datum. The AGL ramp then reads altitude above zero, which is
/// what a run with no terrain model gets in the reference too.
#[derive(Debug, Default, Clone, Copy)]
pub struct FlatDatum;

impl TerrainHeight for FlatDatum {
    fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
        0.0
    }
}

/// The FOV-edge quality pair: `(noise_multiplier, confidence_factor)`.
///
/// Noise grows and confidence falls linearly with how far toward the boundary
/// the target sat. Both are exactly neutral at the centre.
pub fn fov_quality(fov_fraction: f64) -> (f64, f64) {
    (
        1.0 + fov_fraction * FOV_NOISE_MULTIPLIER_SLOPE,
        1.0 - fov_fraction * FOV_CONFIDENCE_SLOPE,
    )
}

/// The altitude quality pair for a sensor `agl_m` above the ground.
///
/// Flat below the threshold, then ramping over `ALTITUDE_QUALITY_RANGE_M` and
/// saturating: `normalized_excess` is clamped at 1, so a sensor at 10 km is
/// treated the same as one at 600 m. The confidence has its own floor beneath
/// the ramp.
pub fn altitude_quality(agl_m: f64) -> (f64, f64) {
    if agl_m <= ALTITUDE_QUALITY_THRESHOLD_M {
        return (1.0, 1.0);
    }
    let excess = (agl_m - ALTITUDE_QUALITY_THRESHOLD_M) / ALTITUDE_QUALITY_RANGE_M.max(1.0);
    let normalized_excess = excess.min(1.0);
    (
        1.0 + normalized_excess * ALTITUDE_NOISE_MULTIPLIER_MAX,
        ALTITUDE_CONFIDENCE_FLOOR.max(1.0 - normalized_excess * ALTITUDE_CONFIDENCE_SLOPE),
    )
}

/// What this stage needs to know about one sensor.
///
/// The reference reads these off its scenario node; see the module docs for why
/// they are injected instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSpec {
    pub max_range_m: f64,
    /// `None` for an omnidirectional sensor. The cascade skips the FOV gate.
    pub fov_half_angle_deg: Option<f64>,
    /// How far below its heading a directional sensor looks. Ignored when
    /// `fov_half_angle_deg` is `None`.
    pub look_down_angle_deg: f64,
    pub required_elevation_deg: f64,
    pub dropout_probability: f64,
    /// The sensor's bearing noise at the reference range, before the
    /// range-dependent scaling.
    pub base_bearing_std_rad: f64,
    /// A thermal sensor emits a second, noisier observation of every target it
    /// accepts.
    pub band: SensorBand,
    /// `false_alarm_rate_per_scan`. Zero disables clutter, which is the
    /// reference's default.
    ///
    /// On the spec rather than on `SensorErrorConfig`: P4.5a scoped that struct
    /// to the *error* model, and clutter is a second observation source rather
    /// than an error on an existing one. The reference keeps them together, so
    /// this is a deliberate divergence in where the field lives, not in what it
    /// does.
    pub false_alarm_rate_per_scan: f64,
    /// `clutter_bearing_std_rad`.
    pub clutter_bearing_std_rad: f64,
    pub error: SensorErrorConfig,
}

impl Default for SensorSpec {
    fn default() -> Self {
        Self {
            max_range_m: 400.0,
            fov_half_angle_deg: None,
            look_down_angle_deg: DEFAULT_LOOK_DOWN_ANGLE_DEG,
            required_elevation_deg: -90.0,
            dropout_probability: 0.0,
            base_bearing_std_rad: 0.015,
            band: SensorBand::Optical,
            false_alarm_rate_per_scan: 0.0,
            clutter_bearing_std_rad: 0.3,
            error: SensorErrorConfig::default(),
        }
    }
}

/// What a sight-line query answers.
///
/// The reference's visibility result carries an `effective_noise_multiplier`
/// beside the verdict — partial vegetation widens the bearing noise without
/// blocking the observation. Stage 6a omitted that factor and did not name it
/// among its deferrals; this is where that is put right, and the multiplier
/// lives on the oracle's result because that is where the reference computes
/// it.
///
/// It also carries what the reference's `ObservationRejection` reports about
/// the path -- the blocker's own name, where the ray first hit, and the
/// closest point reached -- which stage 6 copies into a generation rejection.
/// The blocker name is the world's string rather than [`Blocker`]'s, because
/// a scene may declare a blocker type the cascade folds into `Other`, and the
/// reference reports the declared name.
#[derive(Debug, Clone, PartialEq)]
pub struct SightLine {
    pub visibility: Visibility,
    /// `>= 1.0`. Exactly 1.0 for a clear path.
    pub noise_multiplier: f64,
    /// `visibility.blocker_type`: `"none"` for a clear path.
    pub blocker_type: String,
    pub first_hit_range_m: Option<f64>,
    pub closest_point: Option<[f64; 3]>,
}

impl Detection {
    pub fn transmittance_dominant(&self) -> bool {
        self.dominant_loss_factor == "transmittance"
    }
}

impl SightLine {
    pub fn clear() -> Self {
        Self {
            visibility: Visibility::Visible,
            noise_multiplier: 1.0,
            blocker_type: "none".to_string(),
            first_hit_range_m: None,
            closest_point: None,
        }
    }

    /// A path blocked by `blocker`, with nothing else known about it.
    pub fn blocked(blocker: Blocker) -> Self {
        Self {
            visibility: Visibility::Blocked(blocker),
            blocker_type: blocker_name(blocker).to_string(),
            ..Self::clear()
        }
    }
}

/// The world's name for a cascade blocker, for an oracle that only has the
/// cascade's type to hand.
fn blocker_name(blocker: Blocker) -> &'static str {
    match blocker {
        Blocker::Terrain => "terrain",
        Blocker::Building => "building",
        Blocker::Wall => "wall",
        Blocker::Vegetation => "vegetation",
        Blocker::OutOfCoverage => "out_of_coverage",
        Blocker::Other => "",
    }
}

/// Whether the path from a sensor to a target is clear.
pub trait VisibilityOracle: fmt::Debug + Send + Sync {
    fn sight_line(&self, from: [f64; 3], to: [f64; 3]) -> SightLine;

    /// The sight line, and the world's detection model along it.
    ///
    /// The reference's stage 6 calls `compute_detection_probability`, which
    /// runs line of sight once and derives from it a detection probability
    /// (path loss against the sensor's range, transmittance, weather
    /// visibility, the land cover's detection multiplier), an effective noise
    /// multiplier and the dominant loss. An oracle that models a world
    /// returns those; the default returns `None`, and stage 6 then falls back
    /// to the sensor model alone -- which is what P4.8c did for every oracle,
    /// and what `AlwaysVisible` and the unit-test oracles still mean.
    fn sight_and_detection(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        _max_range_m: f64,
    ) -> (SightLine, Option<Detection>) {
        (self.sight_line(from, to), None)
    }
}

/// The world's contribution to one pair's detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// `detection.p_d`: path loss × transmittance × weather × land cover.
    pub p_d: f64,
    /// `detection.effective_noise_multiplier`: the land cover's noise
    /// multiplier grown with range and scaled by weather.
    pub effective_noise_multiplier: f64,
    /// `detection.dominant_loss_factor`: `"range"`, `"transmittance"`,
    /// `"weather"` or `"blocked"`. A transmittance-dominated miss is reported
    /// as a vegetation occlusion rather than a dropout.
    pub dominant_loss_factor: &'static str,
    /// Stage 6's cloud-layer factor for this sight line (1 without clouds).
    pub cloud_factor: f64,
}

/// The default oracle: everything is visible.
///
/// A run using this generates observations **through terrain**. It is the right
/// default for a slice that does not own line of sight, and
/// [`ObserveStage::visibility_is_modelled`] is how a caller tells the two apart.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysVisible;

impl VisibilityOracle for AlwaysVisible {
    fn sight_line(&self, _from: [f64; 3], _to: [f64; 3]) -> SightLine {
        SightLine::clear()
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

/// One sensor's bias drift and its own random stream.
///
/// The drift is **not** driven from the pair stream. The reference gives
/// `BearingBiasDrift` its own generator, so the bias walk is per sensor and
/// continuous across ticks while the pair streams are opened fresh for every
/// `(node, target, tick)`. Mixing the two would make the bias depend on how
/// many targets happened to be in view.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SensorState {
    drift: BearingBiasDrift,
    rng: DeterministicRng,
}

/// The sensor table, its feeds and the last batch, owned by the stage.
#[derive(Debug)]
struct ObserveState {
    sensors: BTreeMap<String, SensorSpec>,
    drifts: BTreeMap<String, SensorState>,
    terrain: Box<dyn TerrainHeight>,
    oracle: Box<dyn VisibilityOracle>,
    oracle_installed: bool,
    pose_source: PoseSource,
    observations: Vec<BearingObservation>,
    tally: GenerationTally,
    records: Vec<PairRecord>,
    /// The last run's generation rejections, as the reference writes them.
    rejections: Vec<ObservationRejection>,
}

/// Where stage 6 takes each sensor's pose from.
enum PoseSource {
    /// `TickWorld`: where the platform is now.
    World,
    /// An injected source.
    Custom(Box<dyn SensorPoseSource>),
    /// Stage 1's nominal poses for this tick, read through the lent view.
    Controllers,
}

impl fmt::Debug for PoseSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoseSource::World => formatter.write_str("World"),
            PoseSource::Custom(source) => formatter.debug_tuple("Custom").field(source).finish(),
            PoseSource::Controllers => formatter.write_str("Controllers"),
        }
    }
}

impl SensorState {
    /// A sensor's walk as installed: zero bias, and a stream seeded from the
    /// node id alone -- the walk is continuous across ticks, so it must not be
    /// re-keyed per tick.
    fn fresh(node_id: &str, spec: &SensorSpec) -> Self {
        Self {
            drift: BearingBiasDrift::from_config(&spec.error),
            rng: DeterministicRng::keyed(0, &[SeedPart::Str(node_id), SeedPart::Str("bias")]),
        }
    }
}

/// [`Stage::GenerateObservations`] as a [`StageHandler`].
///
/// [`Stage::GenerateObservations`]: crate::stage::Stage::GenerateObservations
#[derive(Debug)]
pub struct ObserveStage {
    state: ObserveState,
    /// The session's workers for the pair evaluations (B3); serial unless
    /// the session gives it more.
    compute: std::sync::Arc<crate::compute::ComputePool>,
    /// The mobile platforms' range scanner, in a run that maps the world for
    /// itself (`world_knowledge: none`). `None` otherwise: no scans, and the
    /// stage is the one it always was.
    range: Option<crate::range_scan::RangeScanner>,
    /// This tick's scans, handed to stage 8.
    scans: Vec<crate::range_scan::RangeScan>,
    /// Sensors that capture at their own rate (P4.10b); `None` is every
    /// sensor every tick.
    capture: Option<CaptureRates>,
}

/// Sensors capturing at their own rates rather than every tick.
///
/// A sensor captures at the first tick at or after each of its deadlines, as
/// a lane at its own rate runs: positions exist only at ticks, so a capture
/// is stamped at the tick that takes it, never interpolated between them.
/// Whether it captures at a tick is a function of the tick alone, so nothing
/// here changes as a run goes on.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureRates {
    rates: BTreeMap<String, crate::scheduler::Schedule>,
    tick_ns: std::num::NonZeroU64,
}

impl CaptureRates {
    /// Rates for the given sensors, in a run whose tick is `dt_s`. Refused if
    /// the tick is not a whole number of nanoseconds, or a rate is faster
    /// than the tick: a sensor cannot capture between ticks.
    pub fn new(
        rates: BTreeMap<String, crate::scheduler::Schedule>,
        dt_s: f64,
    ) -> Result<Self, String> {
        let tick_ns = crate::scheduler::exact_tick_nanos(dt_s)
            .ok_or_else(|| format!("the tick ({dt_s} s) is not a whole number of nanoseconds"))?;
        for (node_id, schedule) in &rates {
            schedule
                .validate()
                .map_err(|error| format!("{node_id}: {error}"))?;
            let faster = match *schedule {
                crate::scheduler::Schedule::Rational { hz_num, hz_den } => {
                    u128::from(hz_num.get()) * u128::from(tick_ns.get())
                        > u128::from(hz_den.get()) * 1_000_000_000
                }
                crate::scheduler::Schedule::Period(period) => period < tick_ns,
                crate::scheduler::Schedule::Once(_) => {
                    return Err(format!("{node_id}: a sensor rate must recur"));
                }
            };
            if faster {
                return Err(format!(
                    "{node_id}: {} is faster than the tick ({dt_s} s); a sensor captures at most \
                     once a tick",
                    crate::scheduler::describe(*schedule)
                ));
            }
        }
        Ok(Self { rates, tick_ns })
    }

    /// Each rated sensor's rate.
    pub fn rates(&self) -> &BTreeMap<String, crate::scheduler::Schedule> {
        &self.rates
    }

    /// Whether `node_id` captures at `tick`: one of its deadlines falls in
    /// the tick's interval `(previous tick, this tick]`, or this is tick 0.
    fn captures(&self, node_id: &str, tick: u64) -> bool {
        let Some(schedule) = self.rates.get(node_id) else {
            return true;
        };
        let epoch = crate::scheduler::SimTime::EPOCH;
        let served_through = |tick: u64| -> u64 {
            // Occurrences with a deadline at or before this tick's time.
            let at = u64::from(self.tick_ns)
                .saturating_mul(tick)
                .saturating_add(1)
                .min(i64::MAX as u64) as i64;
            schedule
                .first_at_or_after(epoch, crate::scheduler::SimTime::from_nanos(at))
                .unwrap_or(u64::MAX)
        };
        tick == 0 || served_through(tick) > served_through(tick - 1)
    }

    /// As recorded: each sensor's rate as `NUM/DEN`.
    fn recorded(&self) -> BTreeMap<String, String> {
        self.rates
            .iter()
            .map(|(id, rate)| (id.clone(), crate::scheduler::rate_text(*rate)))
            .collect()
    }
}

/// The stage's checkpoint when sensors capture at their own rates: the bias
/// walks and the rates they were taken under. Without rates the checkpoint is
/// the bias walks alone, as it always was.
#[derive(serde::Serialize, serde::Deserialize)]
struct RatedState {
    drifts: BTreeMap<String, SensorState>,
    capture_rates: BTreeMap<String, String>,
}

impl Default for ObserveStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ObserveStage {
    pub fn new() -> Self {
        Self {
            state: ObserveState {
                sensors: BTreeMap::new(),
                drifts: BTreeMap::new(),
                terrain: Box::new(FlatDatum),
                oracle: Box::new(AlwaysVisible),
                oracle_installed: false,
                pose_source: PoseSource::World,
                observations: Vec::new(),
                tally: GenerationTally::default(),
                records: Vec::new(),
                rejections: Vec::new(),
            },
            compute: std::sync::Arc::new(crate::compute::ComputePool::serial()),
            range: None,
            scans: Vec::new(),
            capture: None,
        }
    }

    /// Sensors that capture at their own rates (P4.10b). Every rated sensor
    /// must be installed. Configuration, like the sensor table: set before
    /// any tick or restore, kept across reset.
    pub fn set_capture_rates(&mut self, capture: CaptureRates) -> Result<(), String> {
        if let Some(missing) = capture
            .rates
            .keys()
            .find(|id| !self.state.sensors.contains_key(*id))
        {
            return Err(format!("{missing} has no sensor to set a capture rate for"));
        }
        self.capture = (!capture.rates.is_empty()).then_some(capture);
        Ok(())
    }

    /// The sensors capturing at their own rates; `None` when every sensor
    /// captures every tick.
    pub fn capture_rates(&self) -> Option<&CaptureRates> {
        self.capture.as_ref()
    }

    /// Whether `node_id` captures at `tick`.
    pub fn captures_at(&self, node_id: &str, tick: u64) -> bool {
        self.capture
            .as_ref()
            .is_none_or(|capture| capture.captures(node_id, tick))
    }

    /// The checkpointed bias walks, refusing a checkpoint taken under other
    /// capture rates. The form is this run's: the wrapper when it has rates,
    /// the plain map of bias walks when it has none. It is never guessed from
    /// the keys, since a node may be called anything. A checkpoint of the
    /// other form does not decode as this one, and is refused.
    fn decode_drifts(
        &self,
        state: &serde_json::Value,
    ) -> Result<BTreeMap<String, SensorState>, crate::simulation::StageStateError> {
        let mismatch = |recorded: Option<&BTreeMap<String, String>>| {
            crate::simulation::StageStateError::Malformed {
                detail: format!(
                    "the checkpoint's sensor capture rates ({}) are not this run's ({})",
                    describe_rates(recorded),
                    describe_rates(self.capture.as_ref().map(CaptureRates::recorded).as_ref())
                ),
            }
        };
        match &self.capture {
            None => crate::simulation::decode_stage_state(state).map_err(|error| {
                // Said plainly when it is the rated form that failed to decode.
                match serde_json::from_value::<RatedState>(state.clone()) {
                    Ok(rated) => mismatch(Some(&rated.capture_rates)),
                    Err(_) => error,
                }
            }),
            Some(capture) => {
                let rated: RatedState =
                    crate::simulation::decode_stage_state(state).map_err(|_| mismatch(None))?;
                if rated.capture_rates != capture.recorded() {
                    return Err(mismatch(Some(&rated.capture_rates)));
                }
                Ok(rated.drifts)
            }
        }
    }

    /// Scan with `scanner` from every mobile platform, every tick.
    pub fn set_range_scanner(&mut self, scanner: crate::range_scan::RangeScanner) {
        self.range = Some(scanner);
    }

    /// This tick's range scans, in platform order. Empty without a scanner.
    pub fn range_scans(&self) -> &[crate::range_scan::RangeScan] {
        &self.scans
    }

    /// Scan from where each mobile platform is after this tick's motion.
    /// Each platform's scan is independent and keyed on its own node and
    /// tick, so they fork across the session's workers (B3) and come back in
    /// platform order: the result is the same at every worker count.
    fn scan_ranges(&mut self, context: &TickContext<'_>) {
        self.scans.clear();
        let Some(scanner) = &self.range else {
            return;
        };
        let poses: Vec<&argusnet_types::NodeState> = context
            .world
            .platforms()
            .iter()
            .filter(|p| p.is_mobile)
            .collect();
        let (tick, timestamp_s) = (context.tick, context.timestamp_s);
        self.scans = self.compute.map_ordered(&poses, |platform| {
            scanner.scan(
                &platform.node_id,
                tick,
                timestamp_s,
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
        });
    }

    /// Give `node_id` a sensor. A platform with no spec generates nothing.
    ///
    /// Installing also opens the sensor's own bias stream, keyed on the node id
    /// so two sensors never share a walk. Re-installing resets the walk, which
    /// matches `SensorModel::reset`.
    pub fn install_sensor(&mut self, node_id: &str, spec: SensorSpec) {
        let shared = &mut self.state;
        shared.sensors.insert(node_id.to_string(), spec);
        shared
            .drifts
            .insert(node_id.to_string(), SensorState::fresh(node_id, &spec));
    }

    /// Every installed sensor, by node.
    pub fn sensors(&self) -> &BTreeMap<String, SensorSpec> {
        &self.state.sensors
    }

    pub fn remove_sensor(&mut self, node_id: &str) {
        let shared = &mut self.state;
        shared.sensors.remove(node_id);
        shared.drifts.remove(node_id);
    }

    /// Install the terrain sampler the altitude ramp needs. Without one, AGL is
    /// measured from the datum.
    pub fn set_terrain(&mut self, terrain: Box<dyn TerrainHeight>) {
        self.state.terrain = terrain;
    }

    /// The accumulated bearing bias for `node_id`, radians.
    pub fn bias_rad(&self, node_id: &str) -> Option<f64> {
        self.state
            .drifts
            .get(node_id)
            .map(|state| state.drift.current_bias_rad())
    }

    pub fn sensor(&self, node_id: &str) -> Option<SensorSpec> {
        self.state.sensors.get(node_id).copied()
    }

    /// The session's workers for the pair evaluations (B3). The output is
    /// the same at every worker count.
    pub fn set_compute(&mut self, compute: std::sync::Arc<crate::compute::ComputePool>) {
        self.compute = compute;
    }

    pub fn set_visibility_oracle(&mut self, oracle: Box<dyn VisibilityOracle>) {
        let shared = &mut self.state;
        shared.oracle = oracle;
        shared.oracle_installed = true;
    }

    /// Whether line of sight is actually being modelled, or whether the stage
    /// is running on [`AlwaysVisible`] and generating observations through
    /// terrain.
    pub fn visibility_is_modelled(&self) -> bool {
        self.state.oracle_installed
    }

    /// Ask the installed oracle about one sight line, exactly as stage 6
    /// would. A diagnostic, and how a caller confirms what an assembled run
    /// is actually sensing through.
    pub fn probe(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        max_range_m: f64,
    ) -> (SightLine, Option<Detection>) {
        self.state.oracle.sight_and_detection(from, to, max_range_m)
    }

    /// Sense from `source`'s poses rather than from `TickWorld`. See
    /// [`SensorPoseSource`].
    pub fn set_pose_source(&mut self, source: Box<dyn SensorPoseSource>) {
        self.state.pose_source = PoseSource::Custom(source);
    }

    /// Sense from stage 1's nominal controller poses for this tick, as the
    /// reference does: before overrides and deconfliction move the platform
    /// (P4.8e.2). A platform stage 1 did not sample senses from the world.
    pub fn sense_from_controllers(&mut self) {
        self.state.pose_source = PoseSource::Controllers;
    }

    pub fn observations(&self) -> Vec<BearingObservation> {
        self.state.observations.clone()
    }

    pub fn tally(&self) -> GenerationTally {
        self.state.tally.clone()
    }

    /// The last run's accepted-observation counts, which
    /// [`Handoff::EngagementToControllers`] hands to stage 1.
    ///
    /// That is the reference's `controller.update_from_frame(frame, batch)`
    /// loop. It runs after fusion there and straight after stage 6 here; the
    /// difference is unobservable, because the only reader of the motion the
    /// engagement causes is the *next* tick's stage 1.
    ///
    /// [`Handoff::EngagementToControllers`]: crate::simulation::Handoff::EngagementToControllers
    pub fn accepted_by_node_target(&self) -> &AcceptedByNodeTarget {
        &self.state.tally.accepted_by_node_target
    }

    /// Every pair considered on the last run, in `(node, target)` order.
    pub fn records(&self) -> Vec<PairRecord> {
        self.state.records.clone()
    }

    /// The last run's generation rejections: `ObservationBatch.
    /// generation_rejections`, in `(node, target)` order.
    pub fn generation_rejections(&self) -> Vec<ObservationRejection> {
        self.state.rejections.clone()
    }
}

/// A sensor as the pairs see it this tick.
struct NodeView {
    node_id: String,
    spec: SensorSpec,
    origin: Vector3<f64>,
    is_mobile: bool,
    look: Option<[f64; 3]>,
}

/// What every pair reads: immutable world state and the tick's keys.
struct PairContext<'a> {
    oracle: &'a dyn VisibilityOracle,
    terrain: &'a dyn TerrainHeight,
    seed: u32,
    seed_time_key: i64,
    timestamp_s: f64,
}

/// One pair's evaluation, before anything is combined.
struct PairResult {
    range_m: f64,
    outcome: Outcome,
    rejection: Option<ObservationRejection>,
    accepted: Option<AcceptedPair>,
}

/// An accepted pair's observation before the sensor's bias walk is applied,
/// and its thermal duplicate, if the band has one.
struct AcceptedPair {
    direction: Vector3<f64>,
    bearing_std_rad: f64,
    confidence: f64,
    thermal: Option<ThermalDuplicate>,
}

struct ThermalDuplicate {
    direction: Vector3<f64>,
    bearing_std_rad: f64,
    confidence: f64,
}

/// One `(sensor, target)` pair: the gates, and for an accepted pair its
/// noise draws, all from the pair's own stream. Pure in its inputs, so pairs
/// may run in any order on any thread.
fn evaluate_pair(
    context: &PairContext<'_>,
    node: &NodeView,
    target_id: &str,
    target_position: Vector3<f64>,
) -> PairResult {
    let NodeView {
        node_id,
        spec,
        origin,
        is_mobile,
        look,
    } = node;
    let (origin, is_mobile, look) = (*origin, *is_mobile, *look);
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
        let look = look?;
        if range_m <= 0.0 {
            return None;
        }
        Some(FovInput {
            angle_deg: fov_angle_deg(
                [line_of_sight.x, line_of_sight.y, line_of_sight.z],
                range_m,
                look,
            ),
            half_angle_deg,
        })
    });

    let transmittance = atmospheric_attenuation(range_m, &spec.error);
    let sensor_pd = detection_probability(range_m, &spec.error, transmittance);
    let (sight_line, detection) = context.oracle.sight_and_detection(
        [origin.x, origin.y, origin.z],
        [target_position.x, target_position.y, target_position.z],
        spec.max_range_m,
    );
    let visibility = sight_line.visibility;
    let dominant_loss_factor = detection.map(|detection| detection.dominant_loss_factor);
    let (combined_pd, dominant_loss, noise_multiplier) = match detection {
        // The reference: `max(0, min(1, p_d * sensor_pd *
        // cloud_factor))`, multiplied left to right.
        Some(detection) => (
            (detection.p_d * sensor_pd * detection.cloud_factor).clamp(0.0, 1.0),
            if detection.transmittance_dominant() {
                DominantLoss::Transmittance
            } else {
                DominantLoss::Other
            },
            detection.effective_noise_multiplier,
        ),
        // No world detection model: the sensor alone, as P4.8c.
        None => (
            sensor_pd,
            if transmittance < 1.0 && transmittance <= sensor_pd {
                DominantLoss::Transmittance
            } else {
                DominantLoss::Other
            },
            sight_line.noise_multiplier,
        ),
    };

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
        context.seed,
        &[
            SeedPart::Str(node_id),
            SeedPart::Str(target_id),
            SeedPart::Int(context.seed_time_key),
        ],
    );
    let (outcome, gate) = {
        let mut uniform = || pair_rng.uniform();
        evaluate_pair_traced(&input, &mut uniform)
    };
    let rejection = if let Outcome::Rejected(reason) = outcome {
        let rejection =
            generation_rejection(&input, gate, reason, &sight_line, dominant_loss_factor);
        Some(ObservationRejection {
            node_id: node_id.clone(),
            target_id: target_id.to_string(),
            timestamp_s: context.timestamp_s,
            reason: reason.as_str().to_string(),
            detail: rejection.detail,
            origin: Some(origin),
            attempted_point: Some(target_position),
            closest_point: rejection.closest_point.map(Vector3::from),
            blocker_type: rejection.blocker_type,
            first_hit_range_m: rejection.first_hit_range_m,
        })
    } else {
        None
    };

    let Outcome::Accepted { fov_fraction } = outcome else {
        return PairResult {
            range_m,
            outcome,
            rejection,
            accepted: None,
        };
    };

    let (fov_noise_multiplier, fov_confidence_factor) = fov_quality(fov_fraction);
    // The reference ramps on altitude only for a mobile sensor: a
    // mast-mounted camera at the same height is not treated as
    // airborne.
    let (altitude_noise_multiplier, altitude_confidence_factor) = if is_mobile {
        let agl_m = (origin.z - context.terrain.height_at(origin.x, origin.y)).max(0.0);
        altitude_quality(agl_m)
    } else {
        (1.0, 1.0)
    };

    let effective_bearing_std = argusnet_sensing::range_dependent_bearing_noise(
        spec.base_bearing_std_rad,
        range_m,
        &spec.error,
    ) * noise_multiplier
        * fov_noise_multiplier
        * altitude_noise_multiplier;

    let perturbation = bearing_perturbation(effective_bearing_std, &mut pair_rng);
    let direction = perturb_unit_vector(line_of_sight, perturbation);

    let confidence = CONFIDENCE_FLOOR.max(
        (1.0 - (range_m / spec.max_range_m) * CONFIDENCE_RANGE_SLOPE)
            * combined_pd
            * fov_confidence_factor
            * altitude_confidence_factor,
    );

    // A thermal sensor publishes a second observation of the same
    // target: noisier, less confident, and drawn from the SAME pair
    // stream immediately after the first -- so its three draws
    // follow the primary's three, and adding or removing the
    // thermal path shifts nothing else.
    let thermal = (spec.band == SensorBand::Thermal).then(|| {
        let thermal_std = effective_bearing_std * THERMAL_NOISE_FACTOR;
        let thermal_perturbation = bearing_perturbation(thermal_std, &mut pair_rng);
        ThermalDuplicate {
            direction: perturb_unit_vector(line_of_sight, thermal_perturbation),
            bearing_std_rad: thermal_std,
            confidence: confidence * THERMAL_CONFIDENCE_FACTOR,
        }
    });

    PairResult {
        range_m,
        outcome,
        rejection,
        accepted: Some(AcceptedPair {
            direction,
            bearing_std_rad: effective_bearing_std,
            confidence,
            thermal,
        }),
    }
}

/// What a generation rejection says about its pair beyond the reason.
struct RejectionDetail {
    detail: String,
    closest_point: Option<[f64; 3]>,
    blocker_type: String,
    first_hit_range_m: Option<f64>,
}

/// `build_observations`' rejection texts, gate by gate.
///
/// Python's `f"{x:.1f}"` and Rust's `{:.1}` both round the exact binary value
/// half to even, so the texts agree digit for digit. Only the line-of-sight
/// and detection gates carry the sight line's points; the earlier gates
/// reject before the reference queries it.
fn generation_rejection(
    input: &PairInput,
    gate: Gate,
    reason: RejectReason,
    sight_line: &SightLine,
    dominant_loss_factor: Option<&'static str>,
) -> RejectionDetail {
    let bare = |detail: String| RejectionDetail {
        detail,
        closest_point: None,
        blocker_type: String::new(),
        first_hit_range_m: None,
    };
    let with_sight_line = |detail: String| RejectionDetail {
        detail,
        closest_point: sight_line.closest_point,
        blocker_type: sight_line.blocker_type.clone(),
        first_hit_range_m: sight_line.first_hit_range_m,
    };
    match gate {
        Gate::Range if input.range_m < 1e-6 => {
            bare("Target is coincident with the node origin.".to_string())
        }
        Gate::Range => bare(format!(
            "Range {:.1} m exceeds max sensor range {:.1} m.",
            input.range_m, input.max_range_m
        )),
        Gate::FieldOfView => {
            let fov = input.fov.expect("the FOV gate needs an FOV");
            bare(format!(
                "Target angle {:.1} deg exceeds FOV half-angle {:.1} deg.",
                fov.angle_deg, fov.half_angle_deg
            ))
        }
        Gate::Dropout => bare(format!(
            "Observation dropped with p={:.3}.",
            input.dropout_probability
        )),
        Gate::Elevation => bare(format!(
            "Elevation angle {:.1} deg is below required {:.1} deg.",
            input.elevation_deg, input.required_elevation_deg
        )),
        Gate::LineOfSight => {
            // `blocker_type or 'obstacle'`.
            let named = if sight_line.blocker_type.is_empty() {
                "obstacle"
            } else {
                sight_line.blocker_type.as_str()
            };
            with_sight_line(match sight_line.first_hit_range_m {
                None => format!("LOS blocked by {named}."),
                Some(range_m) => format!("LOS blocked by {named} at {range_m:.1} m."),
            })
        }
        Gate::Detection => {
            // Without a world detection model the cascade's own verdict on
            // the loss is all there is to name.
            let dominant = dominant_loss_factor.unwrap_or(match input.dominant_loss {
                DominantLoss::Transmittance => "transmittance",
                DominantLoss::Other => "range",
            });
            with_sight_line(format!(
                "Probabilistic detection rejected the observation with P_d={:.3} ({dominant}).",
                input.combined_pd
            ))
        }
        Gate::Accepted => unreachable!("{reason:?} from a pair that passed every gate"),
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

/// Rotate a direction by a bearing bias, about an axis perpendicular to it.
///
/// **The axis is not random**, although `apply_bias_to_direction`'s docstring
/// says it is *"a rotation about a random axis perpendicular to the direction
/// vector"* and the function takes an `rng` it never uses. The axis is the
/// cross product with world X, falling back to world Y when the direction is
/// within about 25 degrees of X. So the bias always tilts a given direction the
/// same way. The code is the contract; the docstring is wrong, and the unused
/// generator is not reproduced.
fn apply_bias_to_direction(direction: Vector3<f64>, bias_rad: f64) -> Vector3<f64> {
    if bias_rad.abs() < 1e-12 {
        return direction;
    }
    let length = direction.norm();
    if length < 1e-12 {
        return direction;
    }
    let unit = direction / length;

    let arbitrary = if unit.dot(&Vector3::new(1.0, 0.0, 0.0)).abs() > 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let perpendicular = unit.cross(&arbitrary);
    let perpendicular_norm = perpendicular.norm();
    if perpendicular_norm < 1e-12 {
        return direction;
    }
    let perpendicular = perpendicular / perpendicular_norm;

    // Rodrigues. Note the third term uses `dot(perp, unit)`, which is zero by
    // construction here -- the reference writes the general form and it
    // contributes nothing. Reproduced as written rather than simplified away,
    // because "this term is always zero" is only true while the axis stays
    // perpendicular.
    let cos_b = bias_rad.cos();
    let sin_b = bias_rad.sin();
    let rotated = unit * cos_b
        + perpendicular.cross(&unit) * sin_b
        + perpendicular * perpendicular.dot(&unit) * (1.0 - cos_b);
    let rotated_norm = rotated.norm();
    if rotated_norm < 1e-12 {
        return direction;
    }
    rotated / rotated_norm * length
}

/// Three independent normal draws, in x, y, z order — the order numpy fills
/// `size=3`, and therefore the order the stream must be consumed in.
fn bearing_perturbation(noise_std_rad: f64, rng: &mut DeterministicRng) -> Vector3<f64> {
    let x = rng.normal(0.0, noise_std_rad);
    let y = rng.normal(0.0, noise_std_rad);
    let z = rng.normal(0.0, noise_std_rad);
    Vector3::new(x, y, z)
}

/// Capture rates as an error message names them.
fn describe_rates(rates: Option<&BTreeMap<String, String>>) -> String {
    match rates {
        None => "every sensor every tick".to_string(),
        Some(rates) => rates
            .iter()
            .map(|(id, rate)| format!("{id} at {rate} Hz"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

impl StageHandler for ObserveStage {
    /// Every sensor's walk back to zero, streams re-seeded: installation state.
    fn reset(&mut self) {
        let shared = &mut self.state;
        let fresh: BTreeMap<String, SensorState> = shared
            .sensors
            .iter()
            .map(|(id, spec)| (id.clone(), SensorState::fresh(id, spec)))
            .collect();
        shared.drifts = fresh;
        self.scans.clear();
        shared.observations.clear();
        shared.records.clear();
        shared.rejections.clear();
        shared.tally = GenerationTally::default();
    }

    /// The bias walks, and where each walk's stream has got to. The sensor
    /// table, oracle and feeds are configuration, rebuilt by assembly.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        Some(match &self.capture {
            None => serde_json::to_value(&shared.drifts).expect("drift state serialises"),
            Some(capture) => serde_json::json!({
                "drifts": shared.drifts,
                "capture_rates": capture.recorded(),
            }),
        })
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let drifts = self.decode_drifts(state)?;
        let shared = &self.state;
        if !drifts.keys().eq(shared.sensors.keys()) {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: "bias walks name a different sensor table from this run's".into(),
            });
        }
        Ok(())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let drifts = self.decode_drifts(state)?;
        let shared = &mut self.state;
        shared.drifts = drifts;
        shared.observations.clear();
        shared.records.clear();
        shared.rejections.clear();
        shared.tally = GenerationTally::default();
        Ok(())
    }

    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        self.scan_ranges(context);
        let shared = &mut self.state;
        shared.observations.clear();
        shared.records.clear();
        shared.rejections.clear();
        shared.tally = GenerationTally::default();

        // Stage 6 is one of the two stages §3 admits to truth, so `None` here
        // means the capability was withheld -- a wiring fault, not an empty
        // world, and reporting NoWork would hide it.
        let Some(truths) = context.truth else {
            return Err(StageError::Invariant(
                "stage 6 was not handed the truth capability".to_string(),
            ));
        };
        // Not `|| truths.is_empty()`: the reference's clutter loop sits inside
        // the per-node loop *after* the per-target one, so a sensor with a
        // false-alarm rate still reports clutter in an empty world. Returning
        // early on no targets would silently drop it.
        if shared.sensors.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let seed = context.seed;
        let stages = context.stages;
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
        // A sensor at its own rate that is not due this tick takes nothing:
        // no detections, no clutter, no step of its bias walk (P4.10b).
        let capture = self.capture.as_ref();
        let tick = context.tick;
        let specs: Vec<(String, SensorSpec)> = shared
            .sensors
            .iter()
            .filter(|(id, _)| capture.is_none_or(|capture| capture.captures(id, tick)))
            .map(|(id, spec)| (id.clone(), *spec))
            .collect();

        // Each sensor's pose and look, in id order. Serial: the pose can come
        // from stage 1 through the lent view.
        let mut nodes: Vec<NodeView> = Vec::with_capacity(specs.len());
        for (node_id, spec) in specs {
            let Some(platform) = context.world.platform(&node_id) else {
                // A sensor installed for a platform that is not in the world.
                // The reference iterates its scenario nodes, so this cannot
                // arise there; skipping is the reading that cannot invent a
                // sensor position.
                continue;
            };
            let is_mobile = platform.is_mobile;
            let pose = match &shared.pose_source {
                PoseSource::World => None,
                PoseSource::Custom(source) => source.pose(&node_id, timestamp_s),
                PoseSource::Controllers => stages
                    .get::<ControllerStage>(Stage::SampleControllersAndBattery)
                    .and_then(|controllers| controllers.nominal_pose(&node_id, timestamp_s)),
            };
            let (origin, velocity) = match pose {
                Some((p, v)) => (
                    Vector3::new(p[0], p[1], p[2]),
                    Vector3::new(v[0], v[1], v[2]),
                ),
                None => (platform.position, platform.velocity),
            };
            // Resolved once per node, as the reference does: the look
            // direction depends on the node and the tick, not the target.
            let look = spec.fov_half_angle_deg.and(sensor_look_direction(
                [velocity.x, velocity.y, velocity.z],
                spec.look_down_angle_deg,
            ));
            nodes.push(NodeView {
                node_id,
                spec,
                origin,
                is_mobile,
                look,
            });
        }

        // Every pair evaluated, possibly on the session's workers (B3). Each
        // pair reads only immutable world state and draws only from its own
        // `(node, target, tick)` stream, so where and in what order the pairs
        // run cannot change any value; results come back in pair order.
        let pairs: Vec<(usize, usize)> = (0..nodes.len())
            .flat_map(|n| (0..targets.len()).map(move |t| (n, t)))
            .collect();
        let evaluated = {
            let oracle = &*shared.oracle;
            let terrain = &*shared.terrain;
            let pair_context = PairContext {
                oracle,
                terrain,
                seed,
                seed_time_key,
                timestamp_s,
            };
            self.compute.map_ordered(&pairs, |&(n, t)| {
                let (target_id, target_position) = &targets[t];
                evaluate_pair(&pair_context, &nodes[n], target_id, *target_position)
            })
        };

        // Combined serially, in pair order -- sensors in id order, each
        // sensor's targets in truth order, then its clutter -- exactly the
        // order of the single loop this replaced. The bias walk is per-sensor
        // state stepped once per accepted observation, so it is stepped here.
        let mut evaluated = evaluated.into_iter();
        for node in &nodes {
            let NodeView {
                node_id,
                spec,
                origin,
                ..
            } = node;
            let origin = *origin;
            for (target_id, _) in &targets {
                let pair = evaluated.next().expect("one result per pair");
                if let Some(rejection) = pair.rejection {
                    shared.rejections.push(rejection);
                }
                shared.tally.record(node_id, target_id, pair.outcome);
                shared.records.push(PairRecord {
                    node_id: node_id.clone(),
                    target_id: target_id.clone(),
                    range_m: pair.range_m,
                    outcome: pair.outcome,
                });
                let Some(accepted) = pair.accepted else {
                    continue;
                };

                // The bias walk steps once per observation, from the sensor's
                // own stream -- see `SensorState`. It is applied after the
                // noise, so the two compose as rotate-then-tilt.
                let direction = match shared.drifts.get_mut(node_id) {
                    Some(state) => {
                        let bias_rad = match state.drift.increment_std(timestamp_s) {
                            Some(increment_std) => {
                                let increment = state.rng.normal(0.0, increment_std);
                                state.drift.step(timestamp_s, increment)
                            }
                            None => state.drift.step(timestamp_s, 0.0),
                        };
                        apply_bias_to_direction(accepted.direction, bias_rad)
                    }
                    None => accepted.direction,
                };

                shared.observations.push(BearingObservation {
                    node_id: node_id.clone(),
                    target_id: target_id.clone(),
                    origin,
                    direction,
                    bearing_std_rad: accepted.bearing_std_rad,
                    timestamp_s,
                    confidence: accepted.confidence,
                });

                if let Some(thermal) = accepted.thermal {
                    shared.observations.push(BearingObservation {
                        node_id: node_id.clone(),
                        target_id: target_id.clone(),
                        origin,
                        direction: thermal.direction,
                        bearing_std_rad: thermal.bearing_std_rad,
                        timestamp_s,
                        confidence: thermal.confidence,
                    });
                }
            }

            // Clutter: false alarms this sensor reports, independent of any
            // real target. Once per sensor per tick, not once per pair.
            // Its own stream, keyed `(node_id, "clutter", tick)`, so the number
            // of real targets in view cannot shift the false-alarm draws.
            if spec.false_alarm_rate_per_scan > 0.0 {
                let mut clutter_rng = DeterministicRng::keyed(
                    seed,
                    &[
                        SeedPart::Str(node_id),
                        SeedPart::Str(CLUTTER_TARGET_ID),
                        SeedPart::Int(seed_time_key),
                    ],
                );
                let count = clutter_rng.poisson(spec.false_alarm_rate_per_scan);
                for _ in 0..count {
                    // A uniform direction on the sphere, then noise on top of
                    // it -- which is what the reference does, even though
                    // perturbing an already-uniform direction changes nothing
                    // about the distribution. The draws are reproduced because
                    // the stream position is the contract.
                    let raw = bearing_perturbation(1.0, &mut clutter_rng);
                    let norm = raw.norm();
                    if norm < 1e-12 {
                        // The reference `continue`s, so the confidence draw for
                        // this alarm is never taken -- the skip shifts the
                        // stream for every later alarm in the same scan.
                        continue;
                    }
                    let direction = raw / norm;
                    let noise =
                        bearing_perturbation(spec.clutter_bearing_std_rad, &mut clutter_rng);
                    let noisy = direction + noise;
                    let noisy_norm = noisy.norm();
                    if noisy_norm < 1e-12 {
                        continue;
                    }
                    let confidence = clutter_rng
                        .uniform_range(CLUTTER_CONFIDENCE_LOW, CLUTTER_CONFIDENCE_HIGH)
                        .min(CLUTTER_CONFIDENCE_CAP);
                    shared.observations.push(BearingObservation {
                        node_id: node_id.clone(),
                        target_id: CLUTTER_TARGET_ID.to_string(),
                        origin,
                        direction: noisy / noisy_norm,
                        bearing_std_rad: spec.clutter_bearing_std_rad,
                        timestamp_s,
                        confidence,
                    });
                }
            }
        }

        if shared.records.is_empty() && shared.observations.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let accepted = shared.observations.len();
        let attempted = shared.tally.attempted;
        context.note(format!("generated {accepted} of {attempted} attempted"));
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;

    fn capture(rates: &[(&str, &str)], dt_s: f64) -> Result<CaptureRates, String> {
        CaptureRates::new(
            rates
                .iter()
                .map(|(id, rate)| (id.to_string(), crate::scheduler::parse_rate(rate).unwrap()))
                .collect(),
            dt_s,
        )
    }

    fn ticks_captured(capture: &CaptureRates, node_id: &str, ticks: u64) -> Vec<u64> {
        (0..ticks)
            .filter(|&tick| capture.captures(node_id, tick))
            .collect()
    }

    /// A sensor captures at the first tick at or after each of its deadlines:
    /// every second tick at 1 Hz on 0.5 s ticks, and on the 0.5 s grid
    /// nearest after each 4/3 s deadline at 3/4 Hz. A sensor without a rate
    /// captures every tick.
    #[test]
    fn a_sensor_captures_at_the_first_tick_at_or_after_each_deadline() {
        let capture = capture(&[("slow", "1/1"), ("odd", "3/4")], 0.5).unwrap();
        assert_eq!(ticks_captured(&capture, "slow", 9), [0, 2, 4, 6, 8]);
        // Deadlines 0, 1.333, 2.667, 4.0 s: ticks at 0, 1.5, 3.0 and 4.0 s.
        assert_eq!(ticks_captured(&capture, "odd", 9), [0, 3, 6, 8]);
        assert_eq!(ticks_captured(&capture, "other", 3), [0, 1, 2]);
        // At the tick's own rate, every tick.
        let every = super::tests::capture(&[("s", "2/1")], 0.5).unwrap();
        assert_eq!(ticks_captured(&every, "s", 4), [0, 1, 2, 3]);
    }

    #[test]
    fn a_rate_faster_than_the_tick_is_refused() {
        let error = capture(&[("fast", "3/1")], 0.5).unwrap_err();
        assert!(error.contains("faster than the tick"), "{error}");
        assert!(
            capture(&[("s", "1/1")], 0.1 + 0.2).is_err(),
            "an inexact tick"
        );
    }

    /// The rates are configuration a checkpoint must agree with: a stage at
    /// other rates refuses it, in both directions, and a stage without rates
    /// still writes the bias walks alone.
    #[test]
    fn a_checkpoint_resumes_only_under_its_capture_rates() {
        let spec = SensorSpec::default();
        let plain = || {
            let mut stage = ObserveStage::new();
            stage.install_sensor("a", spec);
            stage
        };
        let rated = |rate: &str| {
            let mut stage = plain();
            stage
                .set_capture_rates(capture(&[("a", rate)], 1.0).unwrap())
                .unwrap();
            stage
        };
        let legacy_state = plain().checkpoint_state().unwrap();
        assert!(legacy_state.get("capture_rates").is_none());
        let rated_state = rated("1/2").checkpoint_state().unwrap();
        assert_eq!(rated_state["capture_rates"]["a"], "1/2");

        assert!(rated("1/2").restore_state(&rated_state).is_ok());
        assert!(plain().restore_state(&legacy_state).is_ok());
        for (mut stage, state) in [
            (plain(), &rated_state),
            (rated("1/2"), &legacy_state),
            (rated("1/3"), &rated_state),
        ] {
            let error = stage.restore_state(state).unwrap_err().to_string();
            assert!(error.contains("capture rates"), "{error}");
        }

        let mut stage = plain();
        let error = stage
            .set_capture_rates(capture(&[("b", "1/2")], 1.0).unwrap())
            .unwrap_err();
        assert!(error.contains("no sensor"), "{error}");
    }

    /// The checkpoint's form follows the run's configuration, not its keys:
    /// a sensor may be called `capture_rates` or `drifts`.
    #[test]
    fn a_sensor_may_be_called_anything() {
        let spec = SensorSpec::default();
        let stage = |rated: bool| {
            let mut stage = ObserveStage::new();
            stage.install_sensor("capture_rates", spec);
            stage.install_sensor("drifts", spec);
            if rated {
                stage
                    .set_capture_rates(capture(&[("drifts", "1/2")], 1.0).unwrap())
                    .unwrap();
            }
            stage
        };
        for rated in [false, true] {
            let state = stage(rated).checkpoint_state().unwrap();
            stage(rated).check_state(&state).unwrap();
            stage(rated).restore_state(&state).unwrap();
            let error = stage(!rated).restore_state(&state).unwrap_err().to_string();
            assert!(error.contains("capture rates"), "{error}");
        }
    }
    use crate::fusion::FusionStage;
    use crate::simulation::Simulation;
    use argusnet_sensing::cascade::{Blocker, RejectReason};

    use argusnet_types::{NodeState, TruthState};

    fn simulation() -> Simulation {
        Simulation::bare(SimulationConfig {
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
            ..Default::default()
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

    fn rig(platforms: Vec<NodeState>, truths: Vec<TruthState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        sim.set_truth(truths);
        sim.set_stage_handler(Stage::GenerateObservations, Box::new(ObserveStage::new()))
            .expect("stage 6 is not a transaction boundary");
        sim
    }

    /// The installed stage, read-only.
    fn observed(sim: &Simulation) -> &ObserveStage {
        sim.stage::<ObserveStage>(Stage::GenerateObservations)
            .expect("stage 6 is an ObserveStage")
    }

    /// The installed stage, reached through the session that owns it.
    fn obs(sim: &mut Simulation) -> &mut ObserveStage {
        sim.stage_mut::<ObserveStage>(Stage::GenerateObservations)
            .expect("stage 6 is an ObserveStage")
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

    fn reason(stage: &ObserveStage, node_id: &str, target_id: &str) -> Option<RejectReason> {
        stage
            .records()
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
        let mut sim = rig(
            vec![
                sensor_node("s0", 0.0, 0.0, false),
                sensor_node("s1", 100.0, 0.0, false),
            ],
            vec![
                target("t0", 50.0, 50.0, 30.0),
                target("t1", 60.0, 10.0, 20.0),
            ],
        );
        obs(&mut sim).install_sensor("s0", permissive());
        obs(&mut sim).install_sensor("s1", permissive());

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert_eq!(obs(&mut sim).records().len(), 4);
        assert_eq!(obs(&mut sim).tally().attempted, 4);
        let observations = obs(&mut sim).observations();
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
        let mut sim = rig(
            vec![
                sensor_node("eyes", 0.0, 0.0, false),
                sensor_node("blind", 10.0, 0.0, false),
            ],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        obs(&mut sim).install_sensor("eyes", permissive());

        sim.step().expect("tick runs");

        assert_eq!(obs(&mut sim).records().len(), 1);
        assert_eq!(obs(&mut sim).observations().len(), 1);
        assert_eq!(obs(&mut sim).observations()[0].node_id, "eyes");
    }

    /// A sensor installed for a platform that is not in the world is skipped
    /// rather than invented a position for.
    #[test]
    fn a_sensor_for_an_absent_platform_is_skipped() {
        let mut sim = rig(
            vec![sensor_node("present", 0.0, 0.0, false)],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        obs(&mut sim).install_sensor("present", permissive());
        obs(&mut sim).install_sensor("ghost", permissive());

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert_eq!(obs(&mut sim).records().len(), 1);
        assert_eq!(obs(&mut sim).tally().attempted, 1);
    }

    #[test]
    fn with_no_sensors_or_no_targets_there_is_no_work() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 50.0, 50.0, 30.0)],
        );
        sim.step().expect("tick runs");
        assert_eq!(outcome(&sim), StageOutcome::NoWork);

        let mut empty = rig(vec![sensor_node("s0", 0.0, 0.0, false)], Vec::new());
        obs(&mut empty).install_sensor("s0", permissive());
        empty.step().expect("tick runs");
        assert_eq!(outcome(&empty), StageOutcome::NoWork);
    }

    // -----------------------------------------------------------------
    // The gates, through the real stage
    // -----------------------------------------------------------------

    /// The range gate, reached through the stage rather than by calling the
    /// cascade directly — so the geometry this stage computes is what is being
    /// tested, not the cascade's arithmetic, which P4.5b already pinned.
    #[test]
    fn a_target_past_the_sensors_range_is_rejected() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("near", 100.0, 0.0, 0.0),
                target("far", 5_000.0, 0.0, 0.0),
            ],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                max_range_m: 1_000.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(
            reason(obs(&mut sim), "s0", "far"),
            Some(RejectReason::OutOfRange)
        );
        assert_eq!(reason(obs(&mut sim), "s0", "near"), None);
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("shallow", 100.0, 0.0, 100.0),
                target("steep", 10.0, 0.0, 20.0),
            ],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                required_elevation_deg: 60.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        // 45 degrees, below the gate.
        assert_eq!(
            reason(obs(&mut sim), "s0", "shallow"),
            Some(RejectReason::LowElevation)
        );
        // 63.4 degrees, above it.
        assert_eq!(reason(obs(&mut sim), "s0", "steep"), None);
    }

    /// A directional sensor gates on the angle between its look direction — its
    /// velocity — and the sight line.
    #[test]
    fn a_directional_sensor_gates_on_the_angle_to_its_heading() {
        let mut platform = sensor_node("s0", 0.0, 0.0, true);
        platform.velocity = Vector3::new(10.0, 0.0, 0.0);
        let mut sim = rig(
            vec![platform],
            vec![
                target("ahead", 100.0, 0.0, 0.0),
                target("behind", -100.0, 0.0, 0.0),
            ],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: Some(45.0),
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(obs(&mut sim), "s0", "ahead"), None);
        assert_eq!(
            reason(obs(&mut sim), "s0", "behind"),
            Some(RejectReason::OutsideFov)
        );
    }

    /// A directional sensor that is not moving has no resolvable look
    /// direction, and the cascade's contract is to **skip** the FOV gate rather
    /// than treat that as a rejection. Without the skip, a hovering drone would
    /// see nothing at all.
    #[test]
    fn a_stationary_directional_sensor_skips_the_fov_gate_rather_than_rejecting() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, true)],
            vec![target("behind", -100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: Some(1.0),
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(obs(&mut sim), "s0", "behind"), None);
        assert_eq!(obs(&mut sim).observations().len(), 1);
    }

    /// An oracle that reports a fixed detection result, so what stage 6 does
    /// with each field is observable on its own.
    #[derive(Debug)]
    struct FixedDetection(Detection);

    impl VisibilityOracle for FixedDetection {
        fn sight_line(&self, _from: [f64; 3], _to: [f64; 3]) -> SightLine {
            SightLine::clear()
        }

        fn sight_and_detection(
            &self,
            from: [f64; 3],
            to: [f64; 3],
            _max_range_m: f64,
        ) -> (SightLine, Option<Detection>) {
            (self.sight_line(from, to), Some(self.0))
        }
    }

    fn detection(p_d: f64, noise: f64, transmittance_dominant: bool, cloud: f64) -> Detection {
        Detection {
            p_d,
            effective_noise_multiplier: noise,
            dominant_loss_factor: if transmittance_dominant {
                "transmittance"
            } else {
                "range"
            },
            cloud_factor: cloud,
        }
    }

    fn one_pair_with(oracle: Detection) -> Simulation {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());
        obs(&mut sim).set_visibility_oracle(Box::new(FixedDetection(oracle)));
        sim.step().expect("tick runs");
        sim
    }

    /// The world's `p_d` multiplies into the draw: zero can never be accepted.
    /// A dominant transmittance loss is reported as a vegetation occlusion,
    /// anything else as a dropout -- the reference's naming.
    #[test]
    fn the_world_detection_probability_gates_the_draw_and_names_the_loss() {
        let mut sim = one_pair_with(detection(0.0, 1.0, false, 1.0));
        assert_eq!(
            reason(obs(&mut sim), "s0", "t0"),
            Some(RejectReason::Dropout)
        );
        let mut sim = one_pair_with(detection(0.0, 1.0, true, 1.0));
        assert_eq!(
            reason(obs(&mut sim), "s0", "t0"),
            Some(RejectReason::VegetationOcclusion)
        );
        // A full cloud block is a zero factor, whatever p_d says.
        let mut sim = one_pair_with(detection(1.0, 1.0, false, 0.0));
        assert_eq!(
            reason(obs(&mut sim), "s0", "t0"),
            Some(RejectReason::Dropout)
        );
        // Certain detection is accepted.
        let mut sim = one_pair_with(detection(1.0, 1.0, false, 1.0));
        assert_eq!(reason(obs(&mut sim), "s0", "t0"), None);
    }

    /// The bearing noise scales by the world's effective noise multiplier, not
    /// by the sight line's raw one (which the fixed oracle reports as 1).
    #[test]
    fn the_effective_noise_multiplier_scales_the_bearing_noise() {
        let plain = obs(&mut one_pair_with(detection(1.0, 1.0, false, 1.0))).observations();
        let noisy = obs(&mut one_pair_with(detection(1.0, 2.5, false, 1.0))).observations();
        let (plain, noisy) = (&plain[0], &noisy[0]);
        assert_eq!(noisy.bearing_std_rad, plain.bearing_std_rad * 2.5);
    }

    /// Dropout is a probability, and the draw comes from the pair's own stream.
    #[test]
    fn a_certain_dropout_rejects_every_pair() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                dropout_probability: 1.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(
            reason(obs(&mut sim), "s0", "t0"),
            Some(RejectReason::Dropout)
        );
        assert!(obs(&mut sim).observations().is_empty());
        assert_eq!(obs(&mut sim).tally().rejected(), 1);
    }

    // -----------------------------------------------------------------
    // Visibility
    // -----------------------------------------------------------------

    #[derive(Debug)]
    struct NeverVisible;

    impl VisibilityOracle for NeverVisible {
        fn sight_line(&self, _from: [f64; 3], _to: [f64; 3]) -> SightLine {
            SightLine::blocked(Blocker::Terrain)
        }
    }

    /// The oracle's verdict reaches the cascade.
    #[test]
    fn an_installed_oracle_can_block_every_pair() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());
        obs(&mut sim).set_visibility_oracle(Box::new(NeverVisible));

        sim.step().expect("tick runs");

        // `Blocker::reason` is private, so this names the reason the terrain
        // blocker maps to rather than asking the blocker for it.
        assert_eq!(
            reason(obs(&mut sim), "s0", "t0"),
            Some(RejectReason::TerrainOcclusion)
        );
        assert!(obs(&mut sim).observations().is_empty());
    }

    /// The default oracle sees through terrain, and the stage says so. A caller
    /// that mistook a default run for a line-of-sight-modelled one would be
    /// reading acceptance rates that mean something else entirely.
    #[test]
    fn the_default_oracle_is_reported_as_not_modelling_visibility() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );

        assert!(!obs(&mut sim).visibility_is_modelled());

        obs(&mut sim).set_visibility_oracle(Box::new(NeverVisible));
        assert!(obs(&mut sim).visibility_is_modelled());
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
        let mut sim = rig(
            vec![
                sensor_node("a", 0.0, 0.0, false),
                sensor_node("b", 0.0, 0.0, false),
            ],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("a", permissive());
        obs(&mut sim).install_sensor("b", permissive());

        sim.step().expect("tick runs");

        let observations = obs(&mut sim).observations();
        assert_eq!(observations.len(), 2);
        assert_ne!(observations[0].direction, observations[1].direction);
    }

    /// And the same pair draws differently on successive ticks, because the
    /// millisecond key is part of the seed.
    #[test]
    fn the_same_pair_draws_differently_on_successive_ticks() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        sim.step().expect("tick runs");
        let first = obs(&mut sim).observations()[0].direction;
        sim.step().expect("tick runs");
        let second = obs(&mut sim).observations()[0].direction;

        assert_ne!(first, second);
    }

    /// Two runs with the same seed and world produce the same batch bit for
    /// bit.
    #[test]
    fn a_run_is_reproducible() {
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut sim = rig(
                vec![sensor_node("s0", 0.0, 0.0, false)],
                vec![target("t0", 100.0, 50.0, 25.0)],
            );
            obs(&mut sim).install_sensor("s0", permissive());
            for _ in 0..4 {
                sim.step().expect("tick runs");
            }
            runs.push(obs(&mut sim).observations());
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let observation = &obs(&mut sim).observations()[0];
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("close", 1.0, 0.0, 0.0),
                target("mid", 2_000.0, 0.0, 0.0),
                target("far", 8_000.0, 0.0, 0.0),
            ],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let by_target = |id: &str| {
            observed(&sim)
                .observations()
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![
                target("close", 10.0, 0.0, 0.0),
                target("edge", 990.0, 0.0, 0.0),
            ],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                max_range_m: 1_000.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        let by_target = |id: &str| {
            observed(&sim)
                .observations()
                .into_iter()
                .find(|o| o.target_id == id)
                .expect("accepted")
                .confidence
        };
        assert!(by_target("close") > by_target("edge"));
        assert!(by_target("edge") >= CONFIDENCE_FLOOR);
    }

    /// Every quality modifier is neutral for an omnidirectional, ground-fixed,
    /// clear-path, drift-free sensor — so for that configuration the published
    /// noise is exactly the range-dependent value and the confidence carries
    /// only the range slope and `pd`.
    ///
    /// This started life as `the_deferred_quality_modifiers_are_all_neutral`,
    /// pinning that 6a had hardcoded them to 1.0. They are computed now, and
    /// the test still holds — which is the point: it says the *baseline* case
    /// is unchanged by adding the ramps.
    #[test]
    fn every_quality_modifier_is_neutral_for_a_plain_ground_sensor() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 500.0, 0.0, 0.0)],
        );
        let spec = SensorSpec {
            max_range_m: 1_000.0,
            ..permissive()
        };
        obs(&mut sim).install_sensor("s0", spec);

        sim.step().expect("tick runs");

        let observation = &obs(&mut sim).observations()[0];
        let range_m = 500.0;
        assert_eq!(
            observation.bearing_std_rad,
            argusnet_sensing::range_dependent_bearing_noise(
                spec.base_bearing_std_rad,
                range_m,
                &spec.error
            )
        );
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

        let mut observe = ObserveStage::new();
        let fusion = FusionStage::new(argusnet_core::TrackerConfig::default()).expect("valid");
        observe.install_sensor("s0", permissive());
        observe.install_sensor("s1", permissive());

        sim.set_stage_handler(Stage::GenerateObservations, Box::new(observe))
            .expect("stage 6 is not a transaction boundary");
        sim.set_stage_handler(Stage::UpdateFusion, Box::new(fusion))
            .expect("stage 7 is not a transaction boundary");
        sim.connect(crate::simulation::Handoff::ObservationsToFusion)
            .expect("stages 6 and 7 are installed");

        for _ in 0..5 {
            sim.step().expect("tick runs");
        }

        let frame = sim
            .stage::<FusionStage>(Stage::UpdateFusion)
            .and_then(FusionStage::frame)
            .expect("a frame was produced");
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
        let mut sim = rig(vec![platform], vec![target("behind", -100.0, 0.0, 0.0)]);
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: None,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        assert_eq!(reason(obs(&mut sim), "s0", "behind"), None);
        assert_eq!(obs(&mut sim).observations().len(), 1);
    }

    /// Two targets in the *same place*, seen by one sensor. The geometry is
    /// identical, so the only thing that can separate the two observations is
    /// the target id in the stream key — which is what ADR-006's per-pair
    /// streams are for.
    #[test]
    fn two_targets_in_the_same_place_draw_different_noise() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0), target("t1", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        sim.step().expect("tick runs");

        let observations = obs(&mut sim).observations();
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 300.0, 0.0, 0.0)],
        );
        let spec = short_sighted();
        obs(&mut sim).install_sensor("s0", spec);

        sim.step().expect("tick runs");

        let observed = obs(&mut sim).observations();
        // The pair may be rejected by the detection draw; run until one is
        // accepted so the assertion is about confidence, not about luck.
        let mut confidence = observed.first().map(|o| o.confidence);
        for _ in 0..50 {
            if confidence.is_some() {
                break;
            }
            sim.step().expect("tick runs");
            confidence = obs(&mut sim).observations().first().map(|o| o.confidence);
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 1_900.0, 0.0, 0.0)],
        );
        let spec = short_sighted();
        obs(&mut sim).install_sensor("s0", spec);

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
            if let Some(observation) = obs(&mut sim).observations().first() {
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
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0), target("t1", 120.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        sim.step().expect("tick runs");
        assert_eq!(obs(&mut sim).tally().attempted, 2);
        assert_eq!(obs(&mut sim).records().len(), 2);

        sim.step().expect("tick runs");
        assert_eq!(obs(&mut sim).tally().attempted, 2);
        assert_eq!(obs(&mut sim).records().len(), 2);
        assert_eq!(obs(&mut sim).observations().len(), 2);
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

    // -----------------------------------------------------------------
    // Quality modifiers (6b)
    // -----------------------------------------------------------------

    /// Noise grows and confidence falls linearly toward the FOV boundary, and
    /// both are exactly neutral at the centre.
    #[test]
    fn the_fov_quality_ramp_is_neutral_at_the_centre_and_linear_to_the_edge() {
        assert_eq!(fov_quality(0.0), (1.0, 1.0));
        assert_eq!(
            fov_quality(1.0),
            (1.0 + FOV_NOISE_MULTIPLIER_SLOPE, 1.0 - FOV_CONFIDENCE_SLOPE)
        );
        let (noise, confidence) = fov_quality(0.5);
        assert_eq!(noise, 1.75);
        assert_eq!(confidence, 0.8);
    }

    /// A target near the FOV edge is noisier and less trusted than the same
    /// target dead ahead, through the real stage rather than by calling the
    /// ramp.
    #[test]
    fn a_target_near_the_fov_edge_is_noisier_and_less_confident() {
        let mut platform = sensor_node("s0", 0.0, 0.0, true);
        platform.velocity = Vector3::new(10.0, 0.0, 0.0);
        let mut sim = rig(
            vec![platform],
            // Dead ahead, and 40 degrees off-axis inside a 45 degree cone --
            // at the SAME range, so range cannot explain the difference.
            // An earlier version put the edge target further away, and the
            // range slope alone accounted for the confidence drop.
            vec![
                target("centre", 100.0, 0.0, 0.0),
                target(
                    "edge",
                    100.0 * 0.766_044_443_118_978,
                    100.0 * 0.642_787_609_686_539,
                    0.0,
                ),
            ],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                fov_half_angle_deg: Some(45.0),
                // A level look, so "40 degrees off-axis in the horizontal
                // plane" is 40 degrees from the look direction. With the
                // reference's 30 degree tilt the same target sits outside
                // the cone; the tilt has its own fixture.
                look_down_angle_deg: 0.0,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        let by_target = |id: &str| {
            observed(&sim)
                .observations()
                .into_iter()
                .find(|o| o.target_id == id)
                .unwrap_or_else(|| panic!("{id} was rejected"))
        };
        let centre = by_target("centre");
        let edge = by_target("edge");
        // Identical ranges to within rounding, so the difference is the ramp
        // and nothing else.
        let centre_range = (centre.origin - Vector3::new(100.0, 0.0, 0.0)).norm();
        let edge_range = 100.0;
        assert!((centre_range - edge_range).abs() < 1.0e-9);
        assert!(
            edge.bearing_std_rad > centre.bearing_std_rad,
            "edge {} vs centre {}",
            edge.bearing_std_rad,
            centre.bearing_std_rad
        );
        assert!(edge.confidence < centre.confidence);
    }

    /// The altitude ramp is flat below 200 m AGL, linear over the next 400 m,
    /// and then **saturates** — a sensor at 10 km is treated exactly like one
    /// at 600 m. The confidence has its own floor beneath the ramp.
    #[test]
    fn the_altitude_ramp_is_flat_then_linear_then_saturates() {
        assert_eq!(altitude_quality(0.0), (1.0, 1.0));
        assert_eq!(altitude_quality(ALTITUDE_QUALITY_THRESHOLD_M), (1.0, 1.0));

        // Halfway up the ramp: 400 m AGL is 200 m of excess over a 400 m band.
        let (noise, confidence) = altitude_quality(400.0);
        assert_eq!(noise, 1.0 + 0.5 * ALTITUDE_NOISE_MULTIPLIER_MAX);
        assert_eq!(confidence, 1.0 - 0.5 * ALTITUDE_CONFIDENCE_SLOPE);

        // Saturated, and the same at any greater height.
        let top = altitude_quality(600.0);
        assert_eq!(top, (1.0 + ALTITUDE_NOISE_MULTIPLIER_MAX, 0.5));
        assert_eq!(altitude_quality(10_000.0), top);
        // The confidence floor is below the ramp's own minimum, so it never
        // binds for these constants -- worth stating, since a future slope
        // change would make it live.
        assert!(top.1 > ALTITUDE_CONFIDENCE_FLOOR);
    }

    /// AGL is measured against terrain, not against the datum. A drone at 300 m
    /// over a 250 m hill is 50 m up and below the ramp; the same drone over
    /// flat ground is 300 m up and inside it.
    #[test]
    fn altitude_is_measured_above_ground_not_above_the_datum() {
        #[derive(Debug)]
        struct Plateau(f64);
        impl TerrainHeight for Plateau {
            fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
                self.0
            }
        }

        let noise_at = |ground_m: f64| {
            let mut platform = sensor_node("s0", 0.0, 0.0, true);
            platform.position = Vector3::new(0.0, 0.0, 300.0);
            let mut sim = rig(vec![platform], vec![target("t0", 500.0, 0.0, 300.0)]);
            obs(&mut sim).install_sensor("s0", permissive());
            obs(&mut sim).set_terrain(Box::new(Plateau(ground_m)));
            sim.step().expect("tick runs");
            obs(&mut sim).observations()[0].bearing_std_rad
        };

        let over_flat = noise_at(0.0);
        let over_hill = noise_at(250.0);
        assert!(
            over_flat > over_hill,
            "flat {over_flat} should be noisier than hill {over_hill}"
        );
    }

    /// The reference ramps on altitude only for a **mobile** sensor. A
    /// mast-mounted camera at the same height is not treated as airborne.
    #[test]
    fn a_fixed_sensor_is_never_subject_to_the_altitude_ramp() {
        let high = |mobile: bool| {
            let mut platform = sensor_node("s0", 0.0, 0.0, mobile);
            platform.position = Vector3::new(0.0, 0.0, 600.0);
            let mut sim = rig(vec![platform], vec![target("t0", 500.0, 0.0, 600.0)]);
            obs(&mut sim).install_sensor("s0", permissive());
            sim.step().expect("tick runs");
            obs(&mut sim).observations()[0].bearing_std_rad
        };

        assert!(high(true) > high(false));
        // The fixed one carries no ramp at all.
        assert_eq!(
            high(false),
            argusnet_sensing::range_dependent_bearing_noise(0.015, 500.0, &permissive().error)
        );
    }

    /// Partial occlusion widens the noise without blocking the observation —
    /// the factor stage 6a omitted *and failed to name* among its deferrals.
    #[test]
    fn a_partly_occluded_sight_line_widens_the_noise_without_blocking() {
        #[derive(Debug)]
        struct Hazy(f64);
        impl VisibilityOracle for Hazy {
            fn sight_line(&self, _from: [f64; 3], _to: [f64; 3]) -> SightLine {
                SightLine {
                    noise_multiplier: self.0,
                    ..SightLine::clear()
                }
            }
        }

        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 500.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());
        obs(&mut sim).set_visibility_oracle(Box::new(Hazy(2.5)));

        sim.step().expect("tick runs");

        let observation = &obs(&mut sim).observations()[0];
        assert_eq!(
            obs(&mut sim).observations().len(),
            1,
            "the pair was blocked, not hazed"
        );
        assert_eq!(
            observation.bearing_std_rad,
            argusnet_sensing::range_dependent_bearing_noise(0.015, 500.0, &permissive().error)
                * 2.5
        );
    }

    // -----------------------------------------------------------------
    // Bias drift
    // -----------------------------------------------------------------

    /// The bias walk runs on the **sensor's own** stream, not the pair's, so it
    /// is continuous across ticks and does not depend on how many targets
    /// happened to be in view. Mixing the two would make a drone's bias a
    /// function of the scenario's target count.
    #[test]
    fn the_bias_walk_accumulates_across_ticks() {
        let drifting = SensorSpec {
            error: SensorErrorConfig {
                bias_drift_rate_rad_per_s: 0.01,
                bias_drift_max_rad: 1.0,
                ..permissive().error
            },
            ..permissive()
        };
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", drifting);

        // The first step records the clock without drifting -- there is no
        // previous time to measure an interval against.
        sim.step().expect("tick runs");
        assert_eq!(obs(&mut sim).bias_rad("s0"), Some(0.0));

        sim.step().expect("tick runs");
        let after_two = obs(&mut sim).bias_rad("s0").expect("tracked");
        assert_ne!(after_two, 0.0);

        sim.step().expect("tick runs");
        let after_three = obs(&mut sim).bias_rad("s0").expect("tracked");
        assert_ne!(after_three, after_two);
    }

    /// A sensor with drift disabled never accumulates one, and its direction is
    /// whatever the noise produced.
    #[test]
    fn a_drift_free_sensor_stays_at_zero_bias() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());

        for _ in 0..5 {
            sim.step().expect("tick runs");
        }

        assert_eq!(obs(&mut sim).bias_rad("s0"), Some(0.0));
    }

    /// Two sensors drift independently: the walk is keyed on the node id, so a
    /// fleet does not share one bias.
    #[test]
    fn two_sensors_drift_independently() {
        let drifting = SensorSpec {
            error: SensorErrorConfig {
                bias_drift_rate_rad_per_s: 0.01,
                bias_drift_max_rad: 1.0,
                ..permissive().error
            },
            ..permissive()
        };
        let mut sim = rig(
            vec![
                sensor_node("a", 0.0, 0.0, false),
                sensor_node("b", 10.0, 0.0, false),
            ],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("a", drifting);
        obs(&mut sim).install_sensor("b", drifting);

        for _ in 0..4 {
            sim.step().expect("tick runs");
        }

        assert_ne!(obs(&mut sim).bias_rad("a"), obs(&mut sim).bias_rad("b"));
    }

    /// The bias rotation's axis is **deterministic**, not random — despite the
    /// reference's docstring calling it "a rotation about a random axis" and
    /// the function taking an `rng` it never reads. The same direction and bias
    /// always produce the same result.
    #[test]
    fn the_bias_rotation_axis_is_deterministic_despite_the_docstring() {
        let direction = Vector3::new(0.0, 1.0, 0.0);

        let once = apply_bias_to_direction(direction, 0.2);
        let twice = apply_bias_to_direction(direction, 0.2);

        assert_eq!(once, twice);
        // It rotated, and kept its length.
        assert_ne!(once, direction);
        assert!((once.norm() - direction.norm()).abs() < 1.0e-12);
    }

    /// A zero bias is the identity, and the axis fallback triggers for a
    /// direction close to world X — where the cross product with X would
    /// otherwise be degenerate.
    #[test]
    fn the_bias_rotation_handles_zero_and_the_degenerate_axis() {
        let along_x = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(apply_bias_to_direction(along_x, 0.0), along_x);

        // Falls back to world Y for the axis, so this still rotates rather than
        // returning the input or a NaN.
        let rotated = apply_bias_to_direction(along_x, 0.3);
        assert_ne!(rotated, along_x);
        assert!(rotated.iter().all(|v| v.is_finite()));
        assert!((rotated.norm() - 1.0).abs() < 1.0e-12);
    }

    // -----------------------------------------------------------------
    // The thermal duplicate
    // -----------------------------------------------------------------

    /// A thermal sensor publishes a **second** observation of every target it
    /// accepts: 1.3x the noise, 0.9x the confidence, and a separately drawn
    /// direction.
    #[test]
    fn a_thermal_sensor_publishes_a_second_noisier_observation() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 500.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor(
            "s0",
            SensorSpec {
                band: SensorBand::Thermal,
                ..permissive()
            },
        );

        sim.step().expect("tick runs");

        let observations = obs(&mut sim).observations();
        assert_eq!(observations.len(), 2);
        let primary = &observations[0];
        let thermal = &observations[1];
        assert_eq!(thermal.target_id, primary.target_id);
        // Literals, not the constants. Asserting against `THERMAL_NOISE_FACTOR`
        // would move with the constant, so a mutant that set it to 1.0 moved
        // the expectation too and survived -- which is exactly what the
        // mutation run caught.
        assert_eq!(thermal.bearing_std_rad, primary.bearing_std_rad * 1.3);
        assert_eq!(thermal.confidence, primary.confidence * 0.9);
        assert_eq!(THERMAL_NOISE_FACTOR, 1.3);
        assert_eq!(THERMAL_CONFIDENCE_FACTOR, 0.9);
        // Separately drawn, not a copy.
        assert_ne!(thermal.direction, primary.direction);
        // One accepted *pair*, two observations -- the tally counts pairs.
        assert_eq!(obs(&mut sim).tally().attempted, 1);
    }

    /// The four names the reference treats as thermal, and the fact that
    /// anything else is not.
    #[test]
    fn the_thermal_band_covers_four_sensor_type_names() {
        for name in ["thermal", "lwir", "mwir", "ir"] {
            assert_eq!(
                SensorBand::from_sensor_type(name),
                SensorBand::Thermal,
                "{name}"
            );
        }
        for name in ["eo", "optical", "radar", "", "THERMAL"] {
            assert_eq!(
                SensorBand::from_sensor_type(name),
                SensorBand::Optical,
                "{name}"
            );
        }
    }

    /// An optical sensor publishes exactly one observation per accepted pair,
    /// and the thermal path consumes no draws when it does not run — so adding
    /// or removing a thermal sensor cannot shift another sensor's stream.
    #[test]
    fn an_optical_sensor_publishes_one_observation_and_consumes_the_same_draws() {
        let direction_for = |band: SensorBand| {
            let mut sim = rig(
                vec![sensor_node("s0", 0.0, 0.0, false)],
                vec![target("t0", 500.0, 0.0, 0.0)],
            );
            obs(&mut sim).install_sensor(
                "s0",
                SensorSpec {
                    band,
                    ..permissive()
                },
            );
            sim.step().expect("tick runs");
            (
                obs(&mut sim).observations().len(),
                obs(&mut sim).observations()[0].direction,
            )
        };

        let (optical_count, optical_direction) = direction_for(SensorBand::Optical);
        let (thermal_count, thermal_primary) = direction_for(SensorBand::Thermal);

        assert_eq!(optical_count, 1);
        assert_eq!(thermal_count, 2);
        // The primary observation is identical either way: the thermal draws
        // come after it.
        assert_eq!(optical_direction, thermal_primary);
    }

    /// Altitude costs **confidence** as well as noise. The noise half was
    /// tested; the confidence half was not, and a mutant that dropped the
    /// factor survived because of it.
    #[test]
    fn altitude_costs_confidence_as_well_as_noise() {
        let confidence_at = |altitude_m: f64| {
            let mut platform = sensor_node("s0", 0.0, 0.0, true);
            platform.position = Vector3::new(0.0, 0.0, altitude_m);
            let mut sim = rig(vec![platform], vec![target("t0", 500.0, 0.0, altitude_m)]);
            obs(&mut sim).install_sensor("s0", permissive());
            sim.step().expect("tick runs");
            obs(&mut sim).observations()[0].confidence
        };

        // Same slant range at both altitudes, so only the ramp differs.
        let low = confidence_at(100.0);
        let high = confidence_at(600.0);
        assert!(
            high < low,
            "high {high} should be less confident than low {low}"
        );
        assert_eq!(high, low * 0.5);
    }

    /// The bias actually reaches the published direction. Every other bias test
    /// here reads `bias_rad()`, which a stage that computed the walk and then
    /// threw it away would still satisfy.
    ///
    /// Both runs draw from the same pair stream, so the perturbation is
    /// identical and the only possible difference is the rotation.
    #[test]
    fn the_accumulated_bias_reaches_the_published_direction() {
        let direction_after = |drift_rate: f64| {
            let mut sim = rig(
                vec![sensor_node("s0", 0.0, 0.0, false)],
                vec![target("t0", 100.0, 0.0, 0.0)],
            );
            obs(&mut sim).install_sensor(
                "s0",
                SensorSpec {
                    error: SensorErrorConfig {
                        bias_drift_rate_rad_per_s: drift_rate,
                        bias_drift_max_rad: 1.0,
                        ..permissive().error
                    },
                    ..permissive()
                },
            );
            for _ in 0..3 {
                sim.step().expect("tick runs");
            }
            (
                obs(&mut sim).observations()[0].direction,
                obs(&mut sim).bias_rad("s0").unwrap(),
            )
        };

        let (undrifted, zero_bias) = direction_after(0.0);
        let (drifted, bias) = direction_after(0.05);

        assert_eq!(zero_bias, 0.0);
        assert_ne!(bias, 0.0, "the drifting run did not accumulate a bias");
        assert_ne!(
            drifted, undrifted,
            "the bias was computed but never applied to the direction"
        );
        // And it is exactly the rotation, not some other perturbation.
        assert_eq!(drifted, apply_bias_to_direction(undrifted, bias));
    }

    /// `apply_bias_to_direction` preserves the input's length, which only shows
    /// on a non-unit input — and inside the stage the direction is always unit,
    /// so nothing there would ever notice. The function is general; this pins
    /// that it stays so.
    #[test]
    fn the_bias_rotation_preserves_a_non_unit_length() {
        let direction = Vector3::new(0.0, 3.0, 4.0);
        assert_eq!(direction.norm(), 5.0);

        let rotated = apply_bias_to_direction(direction, 0.25);

        assert!(
            (rotated.norm() - 5.0).abs() < 1.0e-12,
            "norm {}",
            rotated.norm()
        );
        assert_ne!(rotated, direction);
    }

    /// A zero bias returns the input **exactly**, from the early return rather
    /// than by rotating through an identity. Rodrigues with `cos = 1, sin = 0`
    /// lands on the same place mathematically, but going through the
    /// renormalisation is not bit-exact for every input.
    #[test]
    fn a_zero_bias_returns_the_input_bit_for_bit() {
        for direction in [
            Vector3::new(3.0, 0.0, 4.0),
            Vector3::new(0.1, -0.7, 0.3),
            Vector3::new(1.0, 0.0, 0.0),
        ] {
            let result = apply_bias_to_direction(direction, 0.0);
            assert_eq!(
                result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                direction.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{direction:?} did not round-trip"
            );
        }
    }

    // -----------------------------------------------------------------
    // Three mutants that are equivalent, and why
    // -----------------------------------------------------------------

    /// The altitude ramp's threshold comparison can be `<=` or `<` with no
    /// observable difference: at exactly the threshold the excess is zero, so
    /// the ramp evaluates to `(1.0, 1.0)` — the same pair the early return
    /// gives. The early return is kept because it says the intent.
    #[test]
    fn the_altitude_threshold_boundary_is_neutral_either_way() {
        let excess = 0.0_f64;
        let through_the_ramp = (
            1.0 + excess * ALTITUDE_NOISE_MULTIPLIER_MAX,
            ALTITUDE_CONFIDENCE_FLOOR.max(1.0 - excess * ALTITUDE_CONFIDENCE_SLOPE),
        );
        assert_eq!(through_the_ramp, (1.0, 1.0));
        assert_eq!(altitude_quality(ALTITUDE_QUALITY_THRESHOLD_M), (1.0, 1.0));
    }

    /// `ALTITUDE_CONFIDENCE_FLOOR` is unreachable with the current slope: the
    /// ramp saturates at `1 - 0.5 = 0.5`, well above `0.3`. A mutant that drops
    /// the floor is therefore equivalent **today**, and this is what would
    /// notice if the slope ever grew past the point where the floor bites.
    #[test]
    fn the_altitude_confidence_floor_is_currently_unreachable() {
        let saturated = 1.0 - ALTITUDE_CONFIDENCE_SLOPE;
        assert!(
            saturated > ALTITUDE_CONFIDENCE_FLOOR,
            "the floor now binds ({saturated} <= {ALTITUDE_CONFIDENCE_FLOOR});              a mutant dropping it is no longer equivalent and needs its own test"
        );
    }

    /// Flooring AGL at zero is likewise unobservable: a sensor below terrain
    /// yields a negative AGL, and the ramp treats anything at or below the
    /// threshold as neutral — negatives included. The floor is kept because a
    /// negative "height above ground" is a nonsense to propagate.
    #[test]
    fn a_negative_agl_is_already_neutral_without_the_floor() {
        assert_eq!(altitude_quality(-500.0), (1.0, 1.0));
        assert_eq!(altitude_quality(-500.0), altitude_quality(0.0));
    }

    // -----------------------------------------------------------------
    // Clutter (6c)
    // -----------------------------------------------------------------

    fn cluttered(rate: f64) -> SensorSpec {
        SensorSpec {
            false_alarm_rate_per_scan: rate,
            ..permissive()
        }
    }

    /// Clutter is off by default — `false_alarm_rate_per_scan = 0.0` — so a
    /// scenario that never asks for it never pays for it, and every other test
    /// in this file is unaffected.
    #[test]
    fn clutter_is_off_by_default() {
        assert_eq!(SensorSpec::default().false_alarm_rate_per_scan, 0.0);

        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", permissive());
        for _ in 0..20 {
            sim.step().expect("tick runs");
            assert!(obs(&mut sim)
                .observations()
                .iter()
                .all(|o| o.target_id != CLUTTER_TARGET_ID));
        }
    }

    /// A high false-alarm rate produces clutter returns, tagged `"clutter"`,
    /// originating at the sensor and carrying the clutter noise figure.
    #[test]
    fn a_high_false_alarm_rate_produces_tagged_clutter_returns() {
        let mut sim = rig(
            vec![sensor_node("s0", 7.0, 8.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", cluttered(5.0));

        let mut seen = 0;
        for _ in 0..10 {
            sim.step().expect("tick runs");
            for alarm in obs(&mut sim)
                .observations()
                .iter()
                .filter(|o| o.target_id == CLUTTER_TARGET_ID)
            {
                seen += 1;
                assert_eq!(alarm.node_id, "s0");
                assert_eq!(alarm.origin, Vector3::new(7.0, 8.0, 0.0));
                assert_eq!(alarm.bearing_std_rad, 0.3);
                assert!((alarm.direction.norm() - 1.0).abs() < 1.0e-12);
            }
        }
        assert!(seen > 0, "a rate of 5.0 produced no clutter in 10 ticks");
    }

    /// Clutter never outranks the least-trusted real observation: the caller
    /// caps it at `confidence_floor * 0.9`. The draw itself runs up to 0.4, so
    /// without the cap most clutter would be *more* confident than a
    /// long-range real return.
    #[test]
    fn clutter_confidence_is_capped_below_the_real_floor() {
        let mut sim = rig(
            vec![sensor_node("s0", 0.0, 0.0, false)],
            vec![target("t0", 100.0, 0.0, 0.0)],
        );
        obs(&mut sim).install_sensor("s0", cluttered(5.0));

        let mut seen = 0;
        for _ in 0..10 {
            sim.step().expect("tick runs");
            for alarm in obs(&mut sim)
                .observations()
                .iter()
                .filter(|o| o.target_id == CLUTTER_TARGET_ID)
            {
                seen += 1;
                assert!(
                    alarm.confidence <= CLUTTER_CONFIDENCE_CAP,
                    "clutter confidence {} exceeded the cap {CLUTTER_CONFIDENCE_CAP}",
                    alarm.confidence
                );
                assert!(alarm.confidence >= CLUTTER_CONFIDENCE_LOW);
            }
        }
        assert!(seen > 0);
    }

    /// Clutter is emitted even with **no targets at all**. The reference's
    /// clutter loop sits inside the per-node loop after the per-target one, so
    /// an empty world still produces false alarms — and an early return on "no
    /// targets" would silently drop them, which is what an earlier draft did.
    #[test]
    fn clutter_is_emitted_in_a_world_with_no_targets() {
        let mut sim = rig(vec![sensor_node("s0", 0.0, 0.0, false)], Vec::new());
        obs(&mut sim).install_sensor("s0", cluttered(5.0));

        let mut seen = 0;
        for _ in 0..10 {
            sim.step().expect("tick runs");
            seen += obs(&mut sim).observations().len();
        }

        assert!(seen > 0, "no clutter without targets");
        assert_eq!(
            obs(&mut sim).tally().attempted,
            0,
            "there were no pairs to attempt"
        );
    }

    /// Clutter has its **own** stream, keyed `(node_id, "clutter", tick)`, so
    /// the number of real targets in view cannot shift the false-alarm draws.
    /// Two runs differing only in target count must produce identical clutter.
    #[test]
    fn the_clutter_stream_is_independent_of_the_targets_in_view() {
        let clutter_from = |targets: Vec<TruthState>| {
            let mut sim = rig(vec![sensor_node("s0", 0.0, 0.0, false)], targets);
            obs(&mut sim).install_sensor("s0", cluttered(3.0));
            let mut all = Vec::new();
            for _ in 0..6 {
                sim.step().expect("tick runs");
                all.extend(
                    obs(&mut sim)
                        .observations()
                        .into_iter()
                        .filter(|o| o.target_id == CLUTTER_TARGET_ID)
                        .map(|o| o.direction.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                );
            }
            all
        };

        let alone = clutter_from(Vec::new());
        let crowded = clutter_from(vec![
            target("t0", 100.0, 0.0, 0.0),
            target("t1", 120.0, 30.0, 10.0),
            target("t2", 90.0, -40.0, 5.0),
        ]);

        assert!(!alone.is_empty(), "no clutter was produced");
        assert_eq!(alone, crowded);
    }

    /// Two sensors draw their clutter independently.
    #[test]
    fn two_sensors_draw_clutter_independently() {
        let mut sim = rig(
            vec![
                sensor_node("a", 0.0, 0.0, false),
                sensor_node("b", 0.0, 0.0, false),
            ],
            Vec::new(),
        );
        obs(&mut sim).install_sensor("a", cluttered(4.0));
        obs(&mut sim).install_sensor("b", cluttered(4.0));

        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..8 {
            sim.step().expect("tick runs");
            for alarm in obs(&mut sim).observations() {
                if alarm.node_id == "a" {
                    a.push(alarm.direction);
                } else {
                    b.push(alarm.direction);
                }
            }
        }

        assert!(!a.is_empty() && !b.is_empty());
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------
    // The Poisson draw
    // -----------------------------------------------------------------

    /// A non-positive or non-finite rate yields nothing, rather than looping or
    /// producing a NaN count.
    #[test]
    fn the_poisson_draw_rejects_a_non_positive_rate() {
        let mut rng = DeterministicRng::from_seed(3);
        assert_eq!(rng.poisson(0.0), 0);
        assert_eq!(rng.poisson(-1.0), 0);
        assert_eq!(rng.poisson(f64::NAN), 0);
    }

    /// `exp(-lambda)` underflows to zero past about 745, at which point Knuth's
    /// loop can never terminate. The guard returns zero instead of hanging.
    #[test]
    fn the_poisson_draw_does_not_hang_on_an_absurd_rate() {
        let mut rng = DeterministicRng::from_seed(3);
        assert_eq!(rng.poisson(1.0e6), 0);
        assert_eq!(rng.poisson(f64::INFINITY), 0);
    }

    /// The draw's mean tracks its parameter. Not a distribution test — it is a
    /// check that the loop counts what it claims to, which a `k` vs `k - 1`
    /// slip would break.
    #[test]
    fn the_poisson_draw_has_roughly_the_right_mean() {
        for lambda in [0.5_f64, 2.0, 8.0] {
            let mut rng = DeterministicRng::from_seed(17);
            let samples = 20_000;
            let total: u64 = (0..samples).map(|_| rng.poisson(lambda)).sum();
            let mean = total as f64 / samples as f64;
            // Three standard errors for a Poisson mean is 3*sqrt(lambda/n),
            // which is well under 0.1 at these rates; 0.15 leaves room.
            assert!(
                (mean - lambda).abs() < 0.15,
                "lambda {lambda} gave mean {mean}"
            );
        }
    }

    /// **The contract test for clutter.** Two mutants outlast every other
    /// assertion here: dropping the noise perturbation entirely, and drawing it
    /// at the sensor's own bearing std instead of the clutter figure. Neither
    /// changes the *published* `bearing_std_rad` (which is copied from the spec,
    /// not derived from the draw) and neither stops the direction being a
    /// plausible unit vector — so nothing about the observation's shape
    /// separates them.
    ///
    /// What separates them is the stream. This reconstructs the clutter stream
    /// independently — same key, same draw order — and requires exact
    /// agreement on the count, every direction and every confidence. It
    /// therefore pins the draw order, which std each step uses, and the
    /// normalise-perturb-normalise composition all at once.
    #[test]
    fn the_clutter_stream_draws_exactly_what_the_contract_specifies() {
        const SEED: u32 = 11;
        let spec = cluttered(3.0);
        let mut sim = rig(vec![sensor_node("s0", 0.0, 0.0, false)], Vec::new());
        obs(&mut sim).install_sensor("s0", spec);

        sim.step().expect("tick runs");

        // Tick 0 runs at t = 0.0, so the millisecond key is 0.
        let mut expected_rng = DeterministicRng::keyed(
            SEED,
            &[
                SeedPart::Str("s0"),
                SeedPart::Str(CLUTTER_TARGET_ID),
                SeedPart::Int(0),
            ],
        );
        let count = expected_rng.poisson(spec.false_alarm_rate_per_scan);

        let mut expected = Vec::new();
        for _ in 0..count {
            let raw = bearing_perturbation(1.0, &mut expected_rng);
            let norm = raw.norm();
            if norm < 1e-12 {
                continue;
            }
            let direction = raw / norm;
            let noise = bearing_perturbation(spec.clutter_bearing_std_rad, &mut expected_rng);
            let noisy = direction + noise;
            let noisy_norm = noisy.norm();
            if noisy_norm < 1e-12 {
                continue;
            }
            let confidence = expected_rng
                .uniform_range(CLUTTER_CONFIDENCE_LOW, CLUTTER_CONFIDENCE_HIGH)
                .min(CLUTTER_CONFIDENCE_CAP);
            expected.push((noisy / noisy_norm, confidence));
        }

        let published: Vec<_> = obs(&mut sim)
            .observations()
            .into_iter()
            .filter(|o| o.target_id == CLUTTER_TARGET_ID)
            .map(|o| (o.direction, o.confidence))
            .collect();

        assert!(
            !expected.is_empty(),
            "the fixture drew no clutter, so it proves nothing"
        );
        assert_eq!(published.len(), expected.len());
        for (index, ((got_dir, got_conf), (want_dir, want_conf))) in
            published.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(
                got_dir.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want_dir.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "alarm {index}: direction"
            );
            assert_eq!(
                got_conf.to_bits(),
                want_conf.to_bits(),
                "alarm {index}: confidence"
            );
        }
    }
}
