//! Scenario bundle v1, and the assembly that turns one into a running tick.
//!
//! Every stage body exists (P4.8c, P4.8d.3). What did not exist is the thing
//! that builds a configured [`Simulation`] from a scenario: installs each
//! stage, lends each the world, gives each node its sensor and controller,
//! drives the targets' truth and records replay frames. That is this module.
//!
//! The input is `argusnet.scenario-bundle/v1`, written by
//! `argusnet.simulation.scenario_bundle.export_bundle`. It is resolved data:
//! terrain tiles and obstacle metadata exactly as the native ABI already
//! decodes them, so the world here is built by the same code every Python
//! run exercises; controllers as parameters; sensors as the reference's own
//! error configs; targets as truth sampled at every time the tick asks.
//! `docs/migration/SCENARIO_BUNDLE.md` records the format and its decisions.
//!
//! # What v1 assembles
//!
//! Target tracking: stage 1 (controllers, then battery), stage 3
//! (deconfliction over the real terrain), stage 6 (observations, with line of
//! sight over the real world), stage 7 (fusion). The mission stages (2, 8, 9)
//! came with P4.8e.3. Stage 4, the opt-in hard safety monitor, is assembled
//! when the bundle says `safety_blocking` (P4.9d); stage 5, VIO substitution,
//! is still refused by the exporter. [`assemble`] refuses a bundle whose
//! `mission_mode` it does not know rather than running half a mission.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use argusnet_core::TrackerConfig;
use argusnet_planning::battery::BatteryModel;
use argusnet_planning::deconfliction::ExclusionZone;
use argusnet_sensing::SensorErrorConfig;
use argusnet_types::{Bounds2d, NodeState, TruthState, Vector3};
use argusnet_world::index::ObstacleLayer;
use argusnet_world::land_cover::{LandCoverLayer, LandCoverTile};
use argusnet_world::obstacles::ObstacleMetadata;
use argusnet_world::tiled::{Tile, TiledTerrain};
use argusnet_world::visibility::VisibilityProfile;
use serde::Deserialize;

use crate::battery::{BatteryLink, BatteryStage};
use crate::config::SimulationConfig;
use crate::controller::{
    ControllerEnvironment, ControllerStage, LaunchProfile, Operational, PlatformController,
    PursuitOracle,
};
use crate::deconflict::{DeconflictStage, RoleTable};
use crate::follow_path::{FollowPathConfig, FollowPathController, MissionZone};
use crate::fusion::{FusionLink, FusionStage};
use crate::mapping::{InspectionZone, MappingLink, MappingStage};
use crate::mission::{MissionConfig, MissionLink};
use crate::mission_driver::{DriverReportLink, MissionDriverConfig, MissionDriverStage};
use crate::motion::BoundsXy;
use crate::observation_follow::ObservationTriggeredFollow;
use crate::observe::{ObserveStage, SensorBand, SensorSpec};
use crate::overrides::MissionOverrideStage;
use crate::replay::{
    ReplayDeconflictionEvent, ReplayDocument, ReplayEgressProgress, ReplayFrame,
    ReplayInspectionEvent, ReplayLocalizationEstimate, ReplayLocalizationState, ReplayMappingState,
    ReplayPoiStatus, ReplayScanMissionState, ReplayTargetMetadata, ReplayTeamLocalization,
    ReplayWriter,
};
use crate::runtime_world::{SharedObstacles, SharedTerrain, SharedVisibility};
use crate::safety::SafetyStage;
use crate::simulation::{Handoff, Simulation, TickError};
use crate::snapshot::RuntimeSnapshot;
use crate::stage::Stage;
use crate::trajectory::LaunchController;
use crate::waypoint_path::{LoopedWaypointPath, WaypointPathConfig};
use crate::weather::MotionWeather;
use argusnet_planning::poi::InspectionPoi;

pub const BUNDLE_FORMAT: &str = "argusnet.scenario-bundle";
pub const BUNDLE_VERSION: u32 = 1;

/// The reference's `run_simulation` battery model: 500 Wh, and cruise/climb
/// speeds matched to the drones' real search speed.
pub fn reference_battery_model() -> BatteryModel {
    BatteryModel {
        capacity_wh: 500.0,
        cruise_speed_m_per_s: 28.0,
        climb_speed_m_per_s: 5.0,
        ..BatteryModel::default()
    }
}

/// `EnvironmentQuery.los`'s default `terrain_clearance_m`, which the follow
/// controller's visibility read uses (it passes none). Distinct from the
/// sensing clearance the bundle carries, even where the two are equal.
pub const FOLLOW_LOS_CLEARANCE_M: f64 = 1.0;

/// Directional sensors are those the reference gates: `fov_half_angle_deg <
/// 179.0`. Anything wider is omnidirectional.
pub const OMNIDIRECTIONAL_FOV_DEG: f64 = 179.0;

// ---------------------------------------------------------------------------
// The document
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioBundle {
    pub format: String,
    pub version: u32,
    pub scenario_name: String,
    pub mission_mode: String,
    pub seed: u32,
    pub dt_s: f64,
    pub steps: u64,
    pub map_bounds_m: Bounds2d,
    pub world: WorldDocument,
    pub weather: Option<MotionWeather>,
    #[serde(deserialize_with = "python_tracker_config")]
    pub tracker: TrackerConfig,
    pub drone_roles: BTreeMap<String, String>,
    pub mission_zones: Vec<ZoneDocument>,
    /// The scan/map/localize/inspect/egress block's configuration; `None` for
    /// target tracking, which never enters it.
    pub mission: Option<MissionDocument>,
    pub nodes: Vec<NodeDocument>,
    pub targets: Vec<TargetDocument>,
    /// `_coverage_bounds`: the coverage map and the per-zone inspection maps
    /// every mission mode keeps. Absent from bundles written before P4.9a.2,
    /// whose target-tracking runs then have no stage 8.
    #[serde(default)]
    pub coverage_bounds: Option<CoverageBoundsDocument>,
    /// `options.cooperative_search`: fuse the fleet's localization estimates.
    #[serde(default)]
    pub cooperative_search: bool,
    /// `options.safety_blocking`: assemble stage 4's hard safety monitor.
    /// Written only when set, so bundles from before P4.9d still read as off.
    #[serde(default)]
    pub safety_blocking: bool,
    /// `SimulationConfig.requested_duration_s`, for the replay's `meta`.
    #[serde(default)]
    pub requested_duration_s: Option<f64>,
    /// The scenario half of the replay's `meta` (`scenario_replay_metadata`),
    /// copied into every replay this bundle produces. Presentation only:
    /// nothing in the run reads it.
    #[serde(default)]
    pub presentation: Option<serde_json::Map<String, serde_json::Value>>,
}

/// What `run_simulation`'s scan-map-inspect block reads.
#[derive(Debug, Clone, Deserialize)]
pub struct MissionDocument {
    pub coverage_bounds: CoverageBoundsDocument,
    pub scan_coverage_threshold: f64,
    pub localization_confidence_threshold: f64,
    pub egress_arrival_radius_m: f64,
    pub redirect_speed_mps: f64,
    pub gap_fill_min_fraction: f64,
    pub localization: argusnet_localization::LocalizationConfig,
    pub pois: Vec<PoiDocument>,
}

/// The coverage grid's extent and cell size.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CoverageBoundsDocument {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
    pub resolution_m: f64,
}

impl CoverageBoundsDocument {
    fn grid(self) -> argusnet_mapping::GridBounds {
        argusnet_mapping::GridBounds {
            x_min_m: self.x_min_m,
            x_max_m: self.x_max_m,
            y_min_m: self.y_min_m,
            y_max_m: self.y_max_m,
            resolution_m: self.resolution_m,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PoiDocument {
    pub poi_id: String,
    pub name: String,
    pub position: [f64; 3],
    pub priority: i64,
    pub required_dwell_s: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldDocument {
    pub terrain: TerrainDocument,
    pub obstacles: ObstaclesDocument,
    pub land_cover: Option<LandCoverDocument>,
    pub visibility_profile: VisibilityProfile,
    pub los_terrain_clearance_m: f64,
}

/// The `terrain_new` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct TerrainDocument {
    pub bounds: Bounds2d,
    pub tile_size_cells: usize,
    pub base_resolution_m: f64,
    pub floor_m: f64,
    pub tiles: Vec<Tile>,
}

/// The `obstacles_new` payload, with each primitive's metadata inline.
#[derive(Debug, Clone, Deserialize)]
pub struct ObstaclesDocument {
    pub bounds: Bounds2d,
    pub tile_size_m: f64,
    pub primitives: Vec<ObstacleMetadata>,
}

/// The `land_cover_new` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct LandCoverDocument {
    pub bounds: Bounds2d,
    pub tile_size_cells: usize,
    pub resolution: f64,
    pub tiles: Vec<LandCoverTile>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ZoneDocument {
    pub zone_id: String,
    pub zone_type: String,
    pub center_m: [f64; 3],
    pub radius_m: f64,
    #[serde(default)]
    pub priority: i64,
    #[serde(default)]
    pub label: String,
}

impl ZoneDocument {
    fn mission_zone(&self) -> MissionZone {
        MissionZone {
            zone_id: self.zone_id.clone(),
            zone_type: self.zone_type.clone(),
            center_m: self.center_m,
            radius_m: self.radius_m,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeDocument {
    pub node_id: String,
    pub is_mobile: bool,
    pub health: f64,
    pub sensor: SensorDocument,
    pub controller: ControllerDocument,
}

/// A node's sensor, as `_sensor_model_for_node` configures it.
#[derive(Debug, Clone, Deserialize)]
pub struct SensorDocument {
    pub sensor_type: String,
    pub max_range_m: f64,
    pub fov_half_angle_deg: f64,
    pub required_elevation_deg: f64,
    pub dropout_probability: f64,
    pub bearing_std_rad: f64,
    /// `drone_look_down_angle_deg`, applied to every directional sensor.
    pub look_down_angle_deg: f64,
    /// The reference's `SensorErrorConfig`. It carries clutter fields the
    /// Rust config keeps on [`SensorSpec`] instead, so both are read from it.
    pub error: SensorErrorDocument,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SensorErrorDocument {
    #[serde(flatten)]
    pub config: SensorErrorConfig,
    pub false_alarm_rate_per_scan: f64,
    pub clutter_bearing_std_rad: f64,
}

impl SensorDocument {
    pub fn spec(&self) -> SensorSpec {
        SensorSpec {
            max_range_m: self.max_range_m,
            fov_half_angle_deg: (self.fov_half_angle_deg < OMNIDIRECTIONAL_FOV_DEG)
                .then_some(self.fov_half_angle_deg),
            look_down_angle_deg: self.look_down_angle_deg,
            required_elevation_deg: self.required_elevation_deg,
            dropout_probability: self.dropout_probability,
            base_bearing_std_rad: self.bearing_std_rad,
            band: SensorBand::from_sensor_type(&self.sensor_type),
            false_alarm_rate_per_scan: self.error.false_alarm_rate_per_scan,
            clutter_bearing_std_rad: self.error.clutter_bearing_std_rad,
            error: self.error.config,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControllerDocument {
    Static {
        position_m: [f64; 3],
    },
    Launched {
        station_position_m: [f64; 3],
        climb_duration_s: f64,
        operational_altitude_agl_m: f64,
        min_operational_agl_m: f64,
        wind_scale: f64,
        auto_launch_at_s: Option<f64>,
        launch_target_id: String,
        /// Boxed: a follow document is an order of magnitude larger than a
        /// station's three floats, and every node carries one of these.
        operational: Box<OperationalDocument>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperationalDocument {
    Search {
        search: WaypointPathConfig,
    },
    Follow {
        target_id: String,
        follow: FollowDocument,
    },
    SearchThenFollow {
        preferred_target_id: String,
        search: WaypointPathConfig,
        follow: FollowDocument,
        follow_target_id: String,
    },
}

/// `FollowPathController`'s parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct FollowDocument {
    pub map_bounds_m: Bounds2d,
    pub base_agl_m: f64,
    pub standoff_radius_m: f64,
    pub map_scale: f64,
    pub max_speed_mps: f64,
    pub phase: f64,
    pub lead_s: f64,
    pub candidate_count: usize,
    pub rotation_rate_rad_s: f64,
    pub min_agl_m: f64,
    pub vertical_amplitude_m: f64,
    pub vertical_frequency_rad_s: f64,
    pub target_altitude_offset_m: Option<f64>,
    pub max_accel_mps2: Option<f64>,
    pub terrain_following: bool,
    pub terrain_following_agl_m: f64,
    pub terrain_following_smoothing_s: f64,
    pub slot_index: usize,
    pub slot_count: usize,
    pub mission_zones: Vec<ZoneDocument>,
}

impl FollowDocument {
    fn config(&self) -> FollowPathConfig {
        FollowPathConfig {
            map_bounds_m: bounds_xy(self.map_bounds_m),
            base_agl_m: self.base_agl_m,
            standoff_radius_m: self.standoff_radius_m,
            map_scale: self.map_scale,
            max_speed_mps: self.max_speed_mps,
            phase: self.phase,
            lead_s: self.lead_s,
            candidate_count: self.candidate_count,
            rotation_rate_rad_s: self.rotation_rate_rad_s,
            min_agl_m: self.min_agl_m,
            vertical_amplitude_m: self.vertical_amplitude_m,
            vertical_frequency_rad_s: self.vertical_frequency_rad_s,
            target_altitude_offset_m: self.target_altitude_offset_m,
            max_accel_mps2: self.max_accel_mps2,
            terrain_following: self.terrain_following,
            terrain_following_agl_m: self.terrain_following_agl_m,
            terrain_following_smoothing_s: self.terrain_following_smoothing_s,
            slot_index: self.slot_index,
            slot_count: self.slot_count,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetDocument {
    pub target_id: String,
    /// `target_motion_assignments[target_id]`, for the replay's target
    /// metadata. `"unknown"` when absent, as the reference defaults it.
    #[serde(default = "unknown_behavior")]
    pub behavior: String,
    pub samples: Vec<TruthSample>,
}

fn unknown_behavior() -> String {
    "unknown".to_string()
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct TruthSample {
    pub t: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

/// Python's `TrackerConfig`, whose association mode is spelled `labeled` /
/// `gnn` / `jpda`, where `argusnet-core`'s serde spelling is the Rust variant
/// name. Mapped here, at the boundary, as the gRPC adapter maps it, rather
/// than by changing the core's serialized form under its other consumers.
fn python_tracker_config<'de, D>(deserializer: D) -> Result<TrackerConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = serde_json::Value::deserialize(deserializer)?;
    if let Some(mode) = value.get_mut("data_association_mode") {
        let mapped = match mode.as_str() {
            Some("labeled") => "Labeled",
            Some("gnn") => "GNN",
            Some("jpda") => "JPDA",
            Some(other) => {
                return Err(serde::de::Error::custom(format!(
                    "unknown data_association_mode {other:?}"
                )))
            }
            None => {
                return Err(serde::de::Error::custom(
                    "data_association_mode is not a string",
                ))
            }
        };
        *mode = serde_json::Value::String(mapped.to_string());
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

fn bounds_xy(bounds: Bounds2d) -> BoundsXy {
    BoundsXy {
        x_min_m: bounds.x_min_m,
        x_max_m: bounds.x_max_m,
        y_min_m: bounds.y_min_m,
        y_max_m: bounds.y_max_m,
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a bundle cannot be run.
#[derive(Debug, Clone, PartialEq)]
pub enum BundleError {
    Parse(String),
    Format {
        found: String,
    },
    Version {
        found: u32,
    },
    /// Valid, but uses something this build does not assemble.
    Unsupported(String),
    /// Internally inconsistent: names a target it does not carry, etc.
    Invalid(String),
    World(String),
}

impl fmt::Display for BundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BundleError::Parse(detail) => write!(formatter, "bundle does not parse: {detail}"),
            BundleError::Format { found } => {
                write!(formatter, "not a scenario bundle (format {found:?})")
            }
            BundleError::Version { found } => write!(
                formatter,
                "bundle version {found} cannot be read by this build (reads {BUNDLE_VERSION})"
            ),
            BundleError::Unsupported(what) => write!(formatter, "unsupported: {what}"),
            BundleError::Invalid(what) => write!(formatter, "invalid bundle: {what}"),
            BundleError::World(what) => write!(formatter, "world does not build: {what}"),
        }
    }
}

impl std::error::Error for BundleError {}

impl ScenarioBundle {
    /// Parse and version-check. Refuses before anything is built.
    pub fn from_json(text: &str) -> Result<Self, BundleError> {
        // Check the envelope first, so a v2 bundle gets a version error
        // rather than whatever field v2 renamed.
        #[derive(Deserialize)]
        struct Envelope {
            format: String,
            version: u32,
        }
        let envelope: Envelope =
            serde_json::from_str(text).map_err(|error| BundleError::Parse(error.to_string()))?;
        if envelope.format != BUNDLE_FORMAT {
            return Err(BundleError::Format {
                found: envelope.format,
            });
        }
        if envelope.version != BUNDLE_VERSION {
            return Err(BundleError::Version {
                found: envelope.version,
            });
        }
        serde_json::from_str(text).map_err(|error| BundleError::Parse(error.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Truth
// ---------------------------------------------------------------------------

/// The targets' true trajectories, as sampled by the exporter.
///
/// Keyed on the timestamp's **bits**: the exporter computed each key with the
/// same single addition the tick performs, so a lookup either hits exactly or
/// is a time the reference never asked about -- which is a failure, not a
/// value to interpolate.
#[derive(Debug, Default)]
pub struct TruthTable {
    order: Vec<String>,
    samples: HashMap<String, HashMap<u64, TruthPair>>,
}

/// Position and velocity, metres and metres per second.
type TruthPair = ([f64; 3], [f64; 3]);

impl TruthTable {
    pub fn new(targets: &[TargetDocument]) -> Result<Self, BundleError> {
        let mut table = Self::default();
        for target in targets {
            if table.samples.contains_key(&target.target_id) {
                return Err(BundleError::Invalid(format!(
                    "target {} appears twice",
                    target.target_id
                )));
            }
            table.order.push(target.target_id.clone());
            table.samples.insert(
                target.target_id.clone(),
                target
                    .samples
                    .iter()
                    .map(|s| (s.t.to_bits(), (s.position, s.velocity)))
                    .collect(),
            );
        }
        Ok(table)
    }

    pub fn contains(&self, target_id: &str) -> bool {
        self.samples.contains_key(target_id)
    }

    pub fn state(&self, target_id: &str, t: f64) -> Option<([f64; 3], [f64; 3])> {
        self.samples.get(target_id)?.get(&t.to_bits()).copied()
    }

    /// Every target at `t`, in bundle order, or the first one missing.
    pub fn truths_at(&self, t: f64) -> Result<Vec<TruthState>, String> {
        self.order
            .iter()
            .map(|id| {
                let (p, v) = self
                    .state(id, t)
                    .ok_or_else(|| format!("no truth for {id} at t={t}"))?;
                Ok(TruthState {
                    target_id: id.clone(),
                    position: Vector3::new(p[0], p[1], p[2]),
                    velocity: Vector3::new(v[0], v[1], v[2]),
                    timestamp_s: t,
                })
            })
            .collect()
    }
}

/// The follow controllers' pursuit capability, answered from the table.
#[derive(Debug)]
struct TablePursuit(Arc<TruthTable>);

impl PursuitOracle for TablePursuit {
    fn target_state(&self, target_id: &str, timestamp_s: f64) -> Option<([f64; 3], [f64; 3])> {
        self.0.state(target_id, timestamp_s)
    }
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// A built world, shared by every stage that reads it.
pub struct BuiltWorld {
    pub terrain: SharedTerrain,
    pub obstacles: SharedObstacles,
    pub visibility: SharedVisibility,
}

pub fn build_world(world: &WorldDocument) -> Result<BuiltWorld, BundleError> {
    let terrain = TiledTerrain::new(
        world.terrain.bounds,
        world.terrain.tile_size_cells,
        world.terrain.base_resolution_m,
        world.terrain.floor_m,
        world.terrain.tiles.clone(),
    )
    .map_err(|error| BundleError::World(format!("terrain: {error}")))?;
    let primitives = world
        .obstacles
        .primitives
        .iter()
        .cloned()
        .map(|metadata| {
            metadata
                .into_obstacle()
                .map_err(|error| BundleError::World(format!("obstacle: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let obstacles = ObstacleLayer::new(
        world.obstacles.bounds,
        world.obstacles.tile_size_m,
        primitives,
    )
    .map_err(|error| BundleError::World(format!("obstacle layer: {error}")))?;
    let land_cover = world
        .land_cover
        .as_ref()
        .map(|document| {
            LandCoverLayer::new(
                document.bounds,
                document.tile_size_cells,
                document.resolution,
                document.tiles.clone(),
            )
            .map(Arc::new)
            .map_err(|error| BundleError::World(format!("land cover: {error}")))
        })
        .transpose()?;

    let terrain = Arc::new(terrain);
    let obstacles = Arc::new(obstacles);
    Ok(BuiltWorld {
        terrain: SharedTerrain::new(Arc::clone(&terrain), world.terrain.base_resolution_m),
        obstacles: SharedObstacles::new(Arc::clone(&obstacles)),
        visibility: SharedVisibility::new(
            terrain,
            obstacles,
            land_cover,
            Arc::new(world.visibility_profile.clone()),
            world.los_terrain_clearance_m,
        ),
    })
}

/// A bundle, assembled into a running simulation.
pub struct AssembledRun {
    pub sim: Simulation,
    pub battery: BatteryLink,
    pub fusion: FusionLink,
    /// Scan-map-inspect runs only.
    pub mission: Option<MissionLink>,
    pub mapping: Option<MappingLink>,
    /// The mission driver's configuration and POIs, as constructed.
    pub mission_setup: Option<(MissionDriverConfig, Vec<InspectionPoi>)>,
    /// The mission driver's replay seam; see `LocalizationReplay`.
    pub localization_replay: Option<crate::mission_driver::LocalizationReplay>,
    truth: Arc<TruthTable>,
    replay: ReplayWriter,
    dt_s: f64,
    /// Stage 9's POI and coordinator report, scan-map-inspect runs only.
    driver_report: Option<DriverReportLink>,
    report: FrameReport,
}

/// What the replay frames need beyond the stages' own handles.
#[derive(Debug, Clone)]
struct FrameReport {
    scan_coverage_threshold: Option<f64>,
    /// `_drone_home_positions`, for the egress readout.
    homes: BTreeMap<String, [f64; 3]>,
    /// Each target's `target_motion_assignments` entry, in truth order.
    behaviors: Vec<(String, String)>,
    totals: GenerationTotals,
    terrain: Option<SharedTerrain>,
}

/// Stage 6's tallies summed over the frames this replay holds.
#[derive(Debug, Clone, Default)]
struct GenerationTotals {
    attempted: u64,
    rejection_counts: BTreeMap<String, u64>,
    accepted_by_target: BTreeMap<String, u64>,
    rejected_by_target: BTreeMap<String, u64>,
}

impl GenerationTotals {
    fn add(&mut self, tally: &argusnet_sensing::GenerationTally) {
        self.attempted += tally.attempted;
        for (reason, count) in &tally.rejection_counts {
            *self
                .rejection_counts
                .entry((*reason).to_string())
                .or_default() += count;
        }
        for (target, count) in &tally.accepted_by_target {
            *self.accepted_by_target.entry(target.clone()).or_default() += count;
        }
        for (target, count) in &tally.rejected_by_target {
            *self.rejected_by_target.entry(target.clone()).or_default() += count;
        }
    }

    /// `meta.observation_generation`, as `_build_simulation_summary` counts
    /// it: accepted and rejected are the by-target and by-reason sums.
    fn document(&self) -> serde_json::Value {
        let accepted: u64 = self.accepted_by_target.values().sum();
        let rejected: u64 = self.rejection_counts.values().sum();
        serde_json::json!({
            "attempted_count": self.attempted,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "acceptance_rate": if self.attempted > 0 {
                accepted as f64 / self.attempted as f64
            } else {
                0.0
            },
            "rejection_counts": self.rejection_counts,
            "accepted_by_target": self.accepted_by_target,
            "rejected_by_target": self.rejected_by_target,
        })
    }
}

impl fmt::Debug for AssembledRun {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AssembledRun")
            .field("tick", &self.sim.tick())
            .field("frames", &self.replay.frame_count())
            .finish()
    }
}

fn operational(
    document: &OperationalDocument,
    truth: &TruthTable,
) -> Result<(Operational, Vec<ZoneDocument>), BundleError> {
    let known = |id: &str| {
        if truth.contains(id) {
            Ok(())
        } else {
            Err(BundleError::Invalid(format!(
                "follow target {id} is not in the bundle"
            )))
        }
    };
    let lane = |config: &WaypointPathConfig| {
        LoopedWaypointPath::new(config.clone())
            .map_err(|error| BundleError::Invalid(format!("search lane: {error}")))
    };
    Ok(match document {
        OperationalDocument::Search { search } => (Operational::Search(lane(search)?), Vec::new()),
        OperationalDocument::Follow { target_id, follow } => {
            known(target_id)?;
            (
                Operational::Follow {
                    target_id: target_id.clone(),
                    controller: FollowPathController::new(follow.config()),
                },
                follow.mission_zones.clone(),
            )
        }
        OperationalDocument::SearchThenFollow {
            preferred_target_id,
            search,
            follow,
            follow_target_id,
        } => {
            // The reference builds the follow half against
            // `target_by_id[target_id]` with the same id it passes as
            // `preferred_target_id`; the port pursues the preferred one, so
            // the two must agree or the bundle describes something else.
            if follow_target_id != preferred_target_id {
                return Err(BundleError::Unsupported(format!(
                    "a search-then-follow drone that watches {preferred_target_id} but follows \
                     {follow_target_id}"
                )));
            }
            known(preferred_target_id)?;
            (
                Operational::SearchThenFollow(ObservationTriggeredFollow::new(
                    String::new(),
                    preferred_target_id.clone(),
                    lane(search)?,
                    FollowPathController::new(follow.config()),
                )),
                follow.mission_zones.clone(),
            )
        }
    })
}

/// Build a configured [`Simulation`] from a bundle.
pub fn assemble(bundle: &ScenarioBundle) -> Result<AssembledRun, BundleError> {
    match (bundle.mission_mode.as_str(), &bundle.mission) {
        ("target_tracking", None) | ("scan_map_inspect", Some(_)) => {}
        ("scan_map_inspect", None) => {
            return Err(BundleError::Invalid(
                "scan_map_inspect bundle carries no mission section".into(),
            ))
        }
        (mode, _) => {
            return Err(BundleError::Unsupported(format!("mission_mode {mode:?}")));
        }
    }
    let world = build_world(&bundle.world)?;
    let truth = Arc::new(TruthTable::new(&bundle.targets)?);

    let mut sim = Simulation::new(SimulationConfig {
        seed: bundle.seed,
        dt_s: bundle.dt_s,
        steps: Some(bundle.steps),
        run_id: bundle.scenario_name.clone(),
    })
    .map_err(|error| BundleError::Invalid(format!("config: {error:?}")))?;

    sim.load_world(
        bundle
            .nodes
            .iter()
            .map(|node| NodeState {
                node_id: node.node_id.clone(),
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                is_mobile: node.is_mobile,
                timestamp_s: 0.0,
                health: node.health,
                sensor_type: node.sensor.sensor_type.clone(),
                fov_half_angle_deg: Some(node.sensor.fov_half_angle_deg),
                max_range_m: Some(node.sensor.max_range_m),
                ..Default::default()
            })
            .collect(),
        Vec::new(),
    );

    // Stage 1: controllers, then battery.
    let battery = BatteryStage::new();
    let battery_link = battery.link();
    battery_link.set_model(reference_battery_model());
    let mut stage_one = ControllerStage::new().with_battery(battery);
    let mut follow_zones: Option<Vec<ZoneDocument>> = None;
    for node in &bundle.nodes {
        let controller = match &node.controller {
            ControllerDocument::Static { position_m } => PlatformController::Static {
                position_m: *position_m,
            },
            ControllerDocument::Launched {
                station_position_m,
                climb_duration_s,
                operational_altitude_agl_m,
                min_operational_agl_m,
                wind_scale,
                auto_launch_at_s,
                launch_target_id,
                operational: document,
            } => {
                let (mut operational, zones) = operational(document, &truth)?;
                if let Operational::SearchThenFollow(wrapper) = &mut operational {
                    // The wrapper keys engagement on its node id.
                    *wrapper = ObservationTriggeredFollow::new(
                        node.node_id.clone(),
                        wrapper.preferred_target_id().to_string(),
                        wrapper.search().clone(),
                        wrapper.follow().clone(),
                    );
                }
                if matches!(
                    operational,
                    Operational::Follow { .. } | Operational::SearchThenFollow(_)
                ) {
                    match &follow_zones {
                        None => follow_zones = Some(zones),
                        Some(existing) if *existing == zones => {}
                        Some(_) => {
                            return Err(BundleError::Unsupported(
                                "follow controllers with different mission-zone lists".into(),
                            ))
                        }
                    }
                }
                if node.is_mobile {
                    battery_link.register(node.node_id.clone(), Some(*station_position_m));
                }
                PlatformController::Launched {
                    controller: LaunchController::new(
                        *station_position_m,
                        operational,
                        *climb_duration_s,
                        *operational_altitude_agl_m,
                    ),
                    profile: LaunchProfile {
                        min_operational_agl_m: *min_operational_agl_m,
                        wind_scale: *wind_scale,
                        auto_launch_at_s: *auto_launch_at_s,
                    },
                    launch_target_id: launch_target_id.clone(),
                }
            }
        };
        stage_one.install(node.node_id.clone(), controller);
    }
    // Every mobile node is on the ledger, launched or not -- the reference
    // registers `if node.is_mobile`, with no home unless it has a station.
    for node in bundle.nodes.iter().filter(|node| node.is_mobile) {
        if battery_link.remaining_wh(&node.node_id).is_none() {
            battery_link.register(node.node_id.clone(), None);
        }
    }
    stage_one.set_environment(ControllerEnvironment {
        terrain: Box::new(world.terrain.clone()),
        obstacles: Some(Box::new(world.obstacles.clone())),
        visibility: Box::new(world.visibility.with_clearance(FOLLOW_LOS_CLEARANCE_M)),
        planner: None,
        zones: follow_zones
            .unwrap_or_default()
            .iter()
            .map(ZoneDocument::mission_zone)
            .collect(),
        weather: bundle
            .weather
            .clone()
            .map(|weather| Box::new(weather) as Box<dyn crate::trajectory::WeatherSampler + Send>),
        map_bounds_m: Some(bounds_xy(bundle.map_bounds_m)),
        pursuit: Some(Box::new(TablePursuit(Arc::clone(&truth)))),
    });
    sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(stage_one))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;

    // Stage 3: deconfliction over the real terrain, exclusion zones only.
    let mut roles = RoleTable::new().with_terrain(Arc::new(world.terrain.clone()));
    for (drone_id, role) in &bundle.drone_roles {
        roles = roles.with_role(drone_id.clone(), role.clone());
    }
    let deconflict = DeconflictStage::new().with_roles(roles).with_zones(
        bundle
            .mission_zones
            .iter()
            .filter(|zone| zone.zone_type == "exclusion")
            .map(|zone| ExclusionZone {
                zone_id: zone.zone_id.clone(),
                center: zone.center_m,
                radius_m: zone.radius_m,
            })
            .collect(),
    );
    sim.set_stage_handler(Stage::Deconflict, Box::new(deconflict))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;

    // Stages 6 and 7: observations over the real world, fused in process.
    let fusion = FusionStage::new(bundle.tracker.clone()).map_err(BundleError::Invalid)?;
    let fusion_link = fusion.link();

    // Stage 4: the hard floor under deconfliction, only when asked for. Not
    // installed otherwise, so a run without it is the run it always was.
    if bundle.safety_blocking {
        let mut safety = SafetyStage::new();
        safety.set_enabled(true);
        for (drone_id, role) in &bundle.drone_roles {
            safety.set_role(drone_id.clone(), role.clone());
        }
        safety.set_terrain(Box::new(world.terrain.clone()));
        safety.use_battery(battery_link.clone());
        safety.feed_fusion(fusion_link.clone());
        sim.set_stage_handler(Stage::SafetyMonitor, Box::new(safety))
            .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
    }
    let mut observe = ObserveStage::new();
    for node in &bundle.nodes {
        observe.install_sensor(&node.node_id, node.sensor.spec());
    }
    observe.set_visibility_oracle(Box::new(
        world
            .visibility
            .with_weather(bundle.weather.clone().map(Arc::new)),
    ));
    observe.set_terrain(Box::new(world.terrain.clone()));
    observe.feed_fusion(fusion_link.clone());
    // The reference senses from each node's nominal controller pose, not
    // from where overrides and deconfliction moved it (P4.8e.2).
    observe.sense_from_controllers();
    sim.set_stage_handler(Stage::GenerateObservations, Box::new(observe))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
    sim.connect(Handoff::EngagementToControllers)
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
    sim.set_stage_handler(Stage::UpdateFusion, Box::new(fusion))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;

    // Stage 8: coverage and zone inspection, which every mission mode keeps
    // (P4.9a.2). A pre-P4.9a.2 bundle names the grid only in its mission
    // section, so a target-tracking run from one has no stage 8.
    let coverage_bounds = match (&bundle.coverage_bounds, &bundle.mission) {
        (Some(top), Some(mission)) if top.grid() != mission.coverage_bounds.grid() => {
            return Err(BundleError::Invalid(
                "coverage_bounds and mission.coverage_bounds disagree".into(),
            ))
        }
        (Some(top), _) => Some(*top),
        (None, Some(mission)) => Some(mission.coverage_bounds),
        (None, None) => None,
    };
    let mut mapping_stage = None;
    if let Some(bounds) = coverage_bounds {
        let mobile_count = bundle.nodes.iter().filter(|n| n.is_mobile).count();
        let mapping = MappingStage::with_derived_footprint(bounds.grid(), mobile_count)
            .map_err(|error| BundleError::Invalid(format!("coverage bounds: {error}")))?;
        let link = mapping.link();
        link.set_scan_mode(bundle.mission.is_some());
        link.set_cooperative_search(bundle.cooperative_search);
        link.set_zones(
            bundle
                .mission_zones
                .iter()
                .map(|zone| InspectionZone {
                    zone_id: zone.zone_id.clone(),
                    zone_type: zone.zone_type.clone(),
                    center_xy: [zone.center_m[0], zone.center_m[1]],
                    radius_m: zone.radius_m,
                })
                .collect(),
        )
        .map_err(|error| BundleError::Invalid(format!("zone coverage: {error}")))?;
        mapping_stage = Some(mapping);
    }
    let mapping_link = mapping_stage.as_ref().map(MappingStage::link);

    // Stages 2 and 9: the scan/map/localize/inspect/egress mission.
    let mut mission_link = None;
    let mut localization_replay = None;
    let mut mission_setup = None;
    let mut driver_report = None;
    let mut homes_for_egress = BTreeMap::new();
    if let Some(mission) = &bundle.mission {
        let mapping_handle = mapping_link
            .clone()
            .expect("a mission names its coverage grid, so stage 8 exists");
        mapping_handle.enable_localization(mission.localization);

        let overrides = MissionOverrideStage::new();
        let override_link = overrides.link();

        // `_drone_home_positions`: every launched drone's station.
        let homes: BTreeMap<String, [f64; 3]> = bundle
            .nodes
            .iter()
            .filter_map(|node| match &node.controller {
                ControllerDocument::Launched {
                    station_position_m, ..
                } => Some((node.node_id.clone(), *station_position_m)),
                ControllerDocument::Static { .. } => None,
            })
            .collect();
        homes_for_egress = homes.clone();
        let driver = MissionDriverStage::new(
            MissionDriverConfig {
                mission: MissionConfig {
                    scan_coverage_threshold: mission.scan_coverage_threshold,
                    localization_confidence_threshold: mission.localization_confidence_threshold,
                    gap_fill_min_fraction: mission.gap_fill_min_fraction,
                    egress_arrival_radius_m: mission.egress_arrival_radius_m,
                },
                redirect_speed_mps: mission.redirect_speed_mps,
                ..MissionDriverConfig::default()
            },
            mission
                .pois
                .iter()
                .map(|poi| InspectionPoi {
                    poi_id: poi.poi_id.clone(),
                    name: poi.name.clone(),
                    position: poi.position,
                    priority: poi.priority,
                    required_dwell_s: poi.required_dwell_s,
                })
                .collect(),
            homes,
            mapping_handle.clone(),
            override_link.clone(),
            battery_link.clone(),
        );
        let link = driver.mission_link();
        localization_replay = Some(driver.localization_replay());
        driver_report = Some(driver.report_link());
        // Read back off the constructed driver, so a caller (and the wiring
        // test) sees what the stage was actually given.
        mission_setup = Some((driver.config().clone(), driver.pois().to_vec()));
        mapping_handle.watch_mission(link.clone());
        override_link.watch_mission(link.clone());
        sim.set_stage_handler(Stage::ApplyMissionOverrides, Box::new(overrides))
            .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(driver))
            .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
        mission_link = Some(link);
    }
    if let Some(mapping) = mapping_stage {
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
            .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
    }

    let mut replay = ReplayWriter::new(
        bundle.scenario_name.clone(),
        bundle.dt_s,
        u64::from(bundle.seed),
    );
    replay.extend_meta(run_meta(bundle));

    Ok(AssembledRun {
        localization_replay,
        mission_setup,
        mission: mission_link,
        mapping: mapping_link,
        sim,
        battery: battery_link,
        fusion: fusion_link,
        truth,
        replay,
        dt_s: bundle.dt_s,
        driver_report,
        report: FrameReport {
            terrain: Some(world.terrain.clone()),
            scan_coverage_threshold: bundle
                .mission
                .as_ref()
                .map(|mission| mission.scan_coverage_threshold),
            homes: homes_for_egress,
            behaviors: bundle
                .targets
                .iter()
                .map(|target| (target.target_id.clone(), target.behavior.clone()))
                .collect(),
            totals: GenerationTotals::default(),
        },
    })
}

/// The replay `meta` a bundle states before anything runs: the scenario half
/// the bundle carries, and the tracker and duration keys `_build_replay_
/// metadata` adds. The generation totals are added when the replay is
/// finished; `vio_tracked_nodes` and `performance` are not written, because
/// stage 5 is not assembled and a replay's bytes must not depend on the clock.
fn run_meta(bundle: &ScenarioBundle) -> Vec<(String, serde_json::Value)> {
    let mut meta: Vec<(String, serde_json::Value)> = bundle
        .presentation
        .iter()
        .flat_map(|presentation| presentation.clone())
        .collect();
    // `SimulationConfig.actual_duration_s`: `max(0.0, (steps - 1) * dt_s)`.
    let actual_duration_s = (bundle.steps.saturating_sub(1) as f64 * bundle.dt_s).max(0.0);
    meta.push((
        "platform".into(),
        serde_json::json!({
            "min_observations": bundle.tracker.min_observations,
            "max_stale_steps": bundle.tracker.max_stale_steps,
        }),
    ));
    meta.push((
        "requested_duration_s".into(),
        serde_json::json!(bundle.requested_duration_s.unwrap_or(actual_duration_s)),
    ));
    meta.push((
        "actual_duration_s".into(),
        serde_json::json!(actual_duration_s),
    ));
    meta
}

/// `np.mean` over a Python list: numpy's pairwise summation, then one
/// division. Below eight values that is a plain left-to-right sum from 0.0;
/// above, eight running partial sums combined in a fixed tree, and above 128
/// a recursive split -- `pairwise_sum` in numpy's `loops_utils.h`.
pub fn numpy_mean(values: &[f64]) -> f64 {
    fn pairwise(values: &[f64]) -> f64 {
        let n = values.len();
        if n < 8 {
            let mut sum = 0.0;
            for value in values {
                sum += value;
            }
            sum
        } else if n <= 128 {
            let mut r = [0.0f64; 8];
            r.copy_from_slice(&values[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for (k, partial) in r.iter_mut().enumerate() {
                    *partial += values[i + k];
                }
                i += 8;
            }
            let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for value in &values[i..] {
                sum += value;
            }
            sum
        } else {
            let mut half = n / 2;
            half -= half % 8;
            pairwise(&values[..half]) + pairwise(&values[half..])
        }
    }
    pairwise(values) / values.len() as f64
}

/// `LocalizationState` from a frame's tracks and the tick's generated
/// observations; `None` on a frame with no tracks, as the reference leaves it.
fn localization_state(
    tracks: &[crate::replay::ReplayTrack],
    confidences: &[f64],
) -> Option<ReplayLocalizationState> {
    if tracks.is_empty() {
        return None;
    }
    // `np.mean(np.sqrt(flat[[0, 7, 14]]))` for each 6x6 covariance.
    let position_stds: Vec<f64> = tracks
        .iter()
        .filter(|track| track.covariance.len() == 36)
        .map(|track| {
            numpy_mean(&[
                track.covariance[0].sqrt(),
                track.covariance[7].sqrt(),
                track.covariance[14].sqrt(),
            ])
        })
        .collect();
    Some(ReplayLocalizationState {
        active_localizations: tracks.len() as u32,
        mean_position_std_m: if position_stds.is_empty() {
            0.0
        } else {
            numpy_mean(&position_stds)
        },
        mean_observation_confidence: if confidences.is_empty() {
            0.0
        } else {
            numpy_mean(confidences)
        },
        status_counts: BTreeMap::new(),
        frame_id: "map".to_string(),
    })
}

/// `3 if index == 0 else 2 if index < 3 else 1`: the first target in truth
/// order leads, the next two follow.
pub fn target_priority(index: usize) -> u32 {
    match index {
        0 => 3,
        1 | 2 => 2,
        _ => 1,
    }
}

/// Python's `str.title()`: a cased character is upper-cased after an uncased
/// one and lower-cased after a cased one, so a letter after a digit is
/// capitalised too (`"target 1a"` -> `"Target 1A"`).
pub fn python_title(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_cased = false;
    for character in text.chars() {
        let cased = character.is_lowercase() || character.is_uppercase();
        if cased && previous_cased {
            out.extend(character.to_lowercase());
        } else if cased {
            out.extend(character.to_uppercase());
        } else {
            out.push(character);
        }
        previous_cased = cased;
    }
    out
}

/// Python's `round(x, 1)`. Rust's `{:.1}` rounds the exact binary value half
/// to even, as CPython's correctly rounded `round` does, and the string reads
/// back as the float `round` returns: checked on 300,000 values, half of them
/// exact ties, before this was relied on.
pub fn python_round1(value: f64) -> f64 {
    format!("{value:.1}")
        .parse()
        .expect("a formatted float parses")
}

/// Why a step of an assembled run failed.
#[derive(Debug, Clone, PartialEq)]
pub enum RunError {
    Tick(TickError),
    Truth(String),
    Invariant(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Tick(error) => write!(formatter, "{error}"),
            RunError::Truth(detail) | RunError::Invariant(detail) => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for RunError {}

impl AssembledRun {
    /// Stage 1, as assembled. Every assembled run installs it.
    pub fn controllers(&self) -> &ControllerStage {
        self.sim
            .stage::<ControllerStage>(Stage::SampleControllersAndBattery)
            .expect("assembly installs a ControllerStage")
    }

    /// Stage 4, when the bundle asked for `safety_blocking`.
    pub fn safety(&self) -> Option<&SafetyStage> {
        self.sim.stage::<SafetyStage>(Stage::SafetyMonitor)
    }

    /// Stage 6, as assembled. Every assembled run installs it.
    pub fn observe(&self) -> &ObserveStage {
        self.sim
            .stage::<ObserveStage>(Stage::GenerateObservations)
            .expect("assembly installs an ObserveStage")
    }

    /// Stage 3, as assembled. Every assembled run installs it.
    pub fn deconflict(&self) -> &DeconflictStage {
        self.sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .expect("assembly installs a DeconflictStage")
    }

    /// The timestamp the next tick will run at.
    pub fn next_timestamp_s(&self) -> f64 {
        self.sim.tick() as f64 * self.dt_s
    }

    /// One tick: prescribe truth for it, step, record a replay frame.
    pub fn step(&mut self) -> Result<Arc<RuntimeSnapshot>, RunError> {
        let timestamp_s = self.next_timestamp_s();
        let truths = self.truth.truths_at(timestamp_s).map_err(RunError::Truth)?;
        self.sim.set_truth(truths.clone());
        let snapshot = self.sim.step().map_err(RunError::Tick)?;
        // Stage 7 runs on every tick, barren ones included (the track
        // lifecycle ages on ticks passing), so a missing frame is a broken
        // invariant, not an empty tick.
        let frame = self.fusion.frame().ok_or_else(|| {
            RunError::Invariant(format!("stage 7 produced no frame at t={timestamp_s}"))
        })?;
        let mut replay_frame = ReplayWriter::frame(
            timestamp_s,
            &snapshot.platforms,
            &frame.observations,
            &frame.rejected_observations,
            &frame.tracks,
            &truths,
            &frame.metrics,
            &self.observe().generation_rejections(),
        );
        replay_frame.safety_events = frame
            .safety_events
            .iter()
            .map(crate::replay::ReplaySafetyEvent::from)
            .collect();
        self.enrich(&mut replay_frame, timestamp_s, &snapshot.platforms, &truths);
        self.replay.push_frame(replay_frame);
        Ok(snapshot)
    }

    /// The frame's optional payloads, each as `run_simulation` builds its
    /// `PlatformFrame` (P4.9a.2).
    fn enrich(
        &mut self,
        frame: &mut ReplayFrame,
        timestamp_s: f64,
        platforms: &[argusnet_types::NodeState],
        truths: &[argusnet_types::TruthState],
    ) {
        let tally = self.observe().tally();
        self.report.totals.add(&tally);

        frame.deconfliction_events = self
            .sim
            .stage_mut::<DeconflictStage>(Stage::Deconflict)
            .map(DeconflictStage::take_events)
            .unwrap_or_default()
            .into_iter()
            .map(|event| ReplayDeconflictionEvent {
                yielding_drone_id: event.yielding_drone_id,
                conflicting_drone_id: event.conflicting_drone_id,
                predicted_separation_m: event.predicted_separation_m,
                resolution: event.resolution.as_str().to_string(),
                timestamp_s: event.timestamp_s,
            })
            .collect();

        frame.localization_state = localization_state(
            &frame.tracks,
            &self
                .observe()
                .observations()
                .iter()
                .map(|observation| observation.confidence)
                .collect::<Vec<_>>(),
        );

        frame.target_metadata = truths
            .iter()
            .enumerate()
            .map(|(index, truth)| {
                let accepted = tally
                    .accepted_by_target
                    .get(&truth.target_id)
                    .copied()
                    .unwrap_or(0);
                ReplayTargetMetadata {
                    target_id: truth.target_id.clone(),
                    operator_label: python_title(&truth.target_id.replace('-', " ")),
                    target_type: "simulated_mobile".to_string(),
                    priority: target_priority(index),
                    behavior: self
                        .report
                        .behaviors
                        .iter()
                        .find(|(id, _)| *id == truth.target_id)
                        .map_or_else(|| "unknown".to_string(), |(_, b)| b.clone()),
                    status: if accepted > 0 {
                        "observed"
                    } else {
                        "unobserved"
                    }
                    .to_string(),
                    first_seen_s: 0.0,
                    last_seen_s: timestamp_s,
                    confidence: (accepted as f64 / 3.0).min(1.0),
                    notes: String::new(),
                }
            })
            .collect();

        let Some(mapping) = &self.mapping else {
            return;
        };
        let stats = mapping.stats();
        frame.mapping_state = Some(ReplayMappingState {
            coverage_fraction: stats.coverage_fraction,
            covered_cells: stats.covered_cells,
            total_cells: stats.total_cells,
            mean_revisits: stats.mean_revisits,
        });
        frame.inspection_events = mapping
            .inspection_events()
            .into_iter()
            .map(|event| ReplayInspectionEvent {
                zone_id: event.zone_id,
                node_id: event.node_id,
                event_type: event.event_type,
                timestamp_s: event.timestamp_s,
                zone_coverage_fraction: event.zone_coverage_fraction,
            })
            .collect();

        let (Some(mission), Some(driver)) = (&self.mission, &self.driver_report) else {
            return;
        };
        let state = mission.state();
        let report = driver.report();
        let terrain = self.report.terrain.as_ref();
        let newly_scanned_cells = mapping
            .newly_covered_cells()
            .into_iter()
            .flat_map(|[x, y]| {
                // `with contextlib.suppress(Exception)`: 0.0 when unknown.
                let height = terrain.map_or(0.0, |terrain| {
                    crate::observe::TerrainHeight::height_at(terrain, x, y)
                });
                [python_round1(x), python_round1(y), python_round1(height)]
            })
            .collect();
        let egress_progress = if state.phase() == crate::mission::MissionPhase::Egress {
            platforms
                .iter()
                .filter(|platform| platform.is_mobile)
                .filter_map(|platform| {
                    let home = self.report.homes.get(&platform.node_id)?;
                    let (dx, dy) = (home[0] - platform.position.x, home[1] - platform.position.y);
                    Some(ReplayEgressProgress {
                        drone_id: platform.node_id.clone(),
                        distance_to_home_m: (dx * dx + dy * dy).sqrt(),
                        home_position: *home,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        frame.scan_mission_state = Some(ReplayScanMissionState {
            phase: state.phase().as_str().to_string(),
            scan_coverage_fraction: stats.coverage_fraction,
            scan_coverage_threshold: self.report.scan_coverage_threshold.unwrap_or(0.0),
            localization_estimates: mapping
                .last_estimates()
                .into_iter()
                .map(|estimate| ReplayLocalizationEstimate {
                    drone_id: estimate.platform_id,
                    timestamp_s: estimate.timestamp_s,
                    position_estimate: estimate.position_estimate,
                    heading_rad: estimate.heading_rad,
                    position_std_m: estimate.position_std_m,
                    confidence: estimate.confidence,
                })
                .collect(),
            poi_statuses: report
                .pois
                .iter()
                .map(|(status, position)| ReplayPoiStatus {
                    poi_id: status.poi_id.clone(),
                    status: status.state.as_str().to_string(),
                    assigned_drone_id: status.assigned_drone_id.clone(),
                    arrival_time_s: status.arrival_time_s,
                    completion_time_s: status.completion_time_s,
                    dwell_accumulated_s: status.dwell_accumulated_s,
                    position: Some(*position),
                })
                .collect(),
            completed_poi_count: report.completed_count,
            total_poi_count: report.total_count,
            phase_started_at_s: state.phase_started_at_s(),
            newly_scanned_cells,
            localization_timed_out: state.localization_timed_out(),
            coordinator_drone_id: report.coordinator_drone_id,
            egress_progress,
            team_localization: mapping
                .team_localization()
                .map(|team| ReplayTeamLocalization {
                    position: team.position,
                    position_std_m: team.position_std_m,
                    confidence: team.confidence,
                    contributing_node_ids: team.contributing_node_ids,
                }),
        });
    }

    /// Add keys to the replay's `meta` -- provenance a caller knows and the
    /// run does not, such as the config it was built from.
    pub fn extend_replay_meta(
        &mut self,
        entries: impl IntoIterator<Item = (String, serde_json::Value)>,
    ) {
        self.replay.extend_meta(entries);
    }

    /// Run every remaining tick and return the replay.
    pub fn run_to_replay(mut self) -> Result<ReplayDocument, RunError> {
        loop {
            match self.step() {
                Ok(_) => {}
                Err(RunError::Tick(TickError::Completed { .. })) => break,
                Err(error) => return Err(error),
            }
        }
        let totals = self.report.totals.document();
        self.replay
            .extend_meta([("observation_generation".to_string(), totals)]);
        Ok(self.replay.finish())
    }
}
