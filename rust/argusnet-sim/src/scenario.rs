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
//! and the opt-in ones (4 safety, 5 VIO substitution) are refused by the
//! exporter until their assembly lands, and [`assemble`] refuses a bundle
//! whose `mission_mode` it does not know rather than running half a mission.

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
    ControllerEnvironment, ControllerLink, ControllerStage, LaunchProfile, Operational,
    PlatformController, PursuitOracle,
};
use crate::deconflict::{DeconflictLink, DeconflictStage, RoleTable};
use crate::follow_path::{FollowPathConfig, FollowPathController, MissionZone};
use crate::fusion::{FusionLink, FusionStage};
use crate::motion::BoundsXy;
use crate::observation_follow::ObservationTriggeredFollow;
use crate::observe::{ObserveLink, ObserveStage, SensorBand, SensorSpec};
use crate::replay::{ReplayDocument, ReplayWriter};
use crate::runtime_world::{SharedObstacles, SharedTerrain, SharedVisibility};
use crate::simulation::{Simulation, TickError};
use crate::snapshot::RuntimeSnapshot;
use crate::stage::Stage;
use crate::trajectory::LaunchController;
use crate::waypoint_path::{LoopedWaypointPath, WaypointPathConfig};
use crate::weather::MotionWeather;

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
    pub nodes: Vec<NodeDocument>,
    pub targets: Vec<TargetDocument>,
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
    pub samples: Vec<TruthSample>,
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
    pub controllers: ControllerLink,
    pub battery: BatteryLink,
    pub observe: ObserveLink,
    pub fusion: FusionLink,
    pub deconflict: DeconflictLink,
    truth: Arc<TruthTable>,
    replay: ReplayWriter,
    dt_s: f64,
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
    if bundle.mission_mode != "target_tracking" {
        return Err(BundleError::Unsupported(format!(
            "mission_mode {:?} (v1 assembles target_tracking)",
            bundle.mission_mode
        )));
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
    let stage_one = ControllerStage::new().with_battery(battery);
    let controllers = stage_one.link();
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
        controllers.install(node.node_id.clone(), controller);
    }
    // Every mobile node is on the ledger, launched or not -- the reference
    // registers `if node.is_mobile`, with no home unless it has a station.
    for node in bundle.nodes.iter().filter(|node| node.is_mobile) {
        if battery_link.remaining_wh(&node.node_id).is_none() {
            battery_link.register(node.node_id.clone(), None);
        }
    }
    controllers.set_environment(ControllerEnvironment {
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
    let deconflict = DeconflictStage::new();
    let deconflict_link = deconflict.link();
    let mut roles = RoleTable::new().with_terrain(Arc::new(world.terrain.clone()));
    for (drone_id, role) in &bundle.drone_roles {
        roles = roles.with_role(drone_id.clone(), role.clone());
    }
    deconflict_link.set_roles(roles);
    deconflict_link.set_zones(
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
    let observe = ObserveStage::new();
    let observe_link = observe.link();
    for node in &bundle.nodes {
        observe_link.install_sensor(&node.node_id, node.sensor.spec());
    }
    observe_link.set_visibility_oracle(Box::new(
        world
            .visibility
            .with_weather(bundle.weather.clone().map(Arc::new)),
    ));
    observe_link.set_terrain(Box::new(world.terrain.clone()));
    observe_link.feed_fusion(fusion_link.clone());
    observe_link.feed_controllers(controllers.clone());
    // The reference senses from each node's nominal controller pose, not
    // from where overrides and deconfliction moved it (P4.8e.2).
    observe_link.set_pose_source(Box::new(controllers.clone()));
    sim.set_stage_handler(Stage::GenerateObservations, Box::new(observe))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;
    sim.set_stage_handler(Stage::UpdateFusion, Box::new(fusion))
        .map_err(|error| BundleError::Invalid(format!("{error:?}")))?;

    Ok(AssembledRun {
        sim,
        controllers,
        battery: battery_link,
        observe: observe_link,
        fusion: fusion_link,
        deconflict: deconflict_link,
        truth,
        replay: ReplayWriter::new(
            bundle.scenario_name.clone(),
            bundle.dt_s,
            u64::from(bundle.seed),
        ),
        dt_s: bundle.dt_s,
    })
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
        self.replay.push_frame(ReplayWriter::frame(
            timestamp_s,
            &snapshot.platforms,
            &frame.observations,
            &frame.rejected_observations,
            &frame.tracks,
            &truths,
            &frame.metrics,
            &[],
        ));
        Ok(snapshot)
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
        Ok(self.replay.finish())
    }
}
