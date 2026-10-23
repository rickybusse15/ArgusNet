//! The Rust-native scenario builder (DELIVERY_PLAN P4.9, P4.9b).
//!
//! P4.9 makes versioned config files the canonical input and requires the
//! Rust CLI to have no Python runtime dependency. Until now a scenario bundle
//! could only come from Python's `build_default_scenario` + `export_bundle`.
//! This module builds one from an `argusnet.scenario-config` document instead.
//!
//! P4.9b.1 (this slice) builds the **world half**: map bounds, the procedural
//! terrain (P4.3's port), the environment's obstacles and patches
//! ([`environment`]), the land cover, and from them the bundle's `world`
//! section and the scenario half of its `presentation` (terrain metadata with
//! the viewer mesh, terrain summary, land cover, bounds, CRS). The platforms,
//! targets, zones and POIs are P4.9b.2.
//!
//! # Parity is not bit-identity, and why
//!
//! The parity target is the bundle Python exports for the same options. It
//! cannot be met bit for bit wherever a value derives from terrain height:
//! P4.3 measured numpy's `exp` differing from the platform's by one ulp on
//! ~5% of inputs, which leaves ~5% of terrain cells up to 32 ulp (5.7e-14 m)
//! apart, and tiling spreads that to about half the tile samples. numpy's
//! `exp` is also platform-dependent, so the "Python value" is not one value
//! across machines. Everything not derived from a height -- structure,
//! identifiers, footprints, counts, masks, classes -- is compared exactly;
//! heights and what is computed from them are compared within that bound.

pub mod behaviors;
pub mod buildings;
pub mod environment;
pub mod features;
pub mod power_lines;
pub mod scenario;
pub mod terrain_import;
pub mod trees;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use argusnet_world::land_cover_build::CoverGrid;
use argusnet_world::landscape::{
    build_data_land_cover, build_land_cover_layer, build_terrain_layer, season_from_month,
    BuiltTerrain, LandCoverRequest, TerrainBuildConfig, TerrainSource,
};
use argusnet_world::noise::Bounds2D;
use argusnet_world::obstacles::{Obstacle, ObstacleMetadata};
use argusnet_world::terrain_grid::mask_coverage_fraction;
use argusnet_world::visibility::VisibilityProfile;

use crate::deterministic::{stable_seed, SeedPart};
use environment::{build_environment, MapBounds};

pub const CONFIG_FORMAT: &str = "argusnet.scenario-config";
pub const CONFIG_VERSION: u32 = 1;

/// Largest terrain, buildings, or features document a scenario may name.
/// A shared scenario is untrusted input; this matches the replay cap.
const MAX_IMPORT_BYTES: u64 = 256 * 1024 * 1024;

pub(super) fn read_import_limited(path: &str) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if len > MAX_IMPORT_BYTES {
        return Err(format!(
            "{len} bytes, over the {MAX_IMPORT_BYTES} byte limit"
        ));
    }
    std::fs::read(path).map_err(|error| error.to_string())
}

/// `SimulationConstants.dynamics`' map extent.
pub const MAP_BOUNDS_X_EXTENT_M: f64 = 360.0;
pub const MAP_BOUNDS_Y_MIN_M: f64 = -330.0;
pub const MAP_BOUNDS_Y_MAX_M: f64 = 360.0;
/// `SimulationConstants.sensor.los_terrain_clearance_m`.
pub const LOS_TERRAIN_CLEARANCE_M: f64 = 1.0;
/// `TerrainLayer.viewer_mesh`'s default `max_dimension`.
pub const VIEWER_MESH_MAX_DIMENSION: usize = 128;
const TILE_SIZE_CELLS: usize = 256;

fn default_map_preset() -> String {
    "regional".into()
}
fn mixed() -> String {
    "mixed".into()
}
fn procedural() -> String {
    "procedural".into()
}
fn alpine() -> String {
    "alpine".into()
}
fn one() -> f64 {
    1.0
}
fn july() -> u32 {
    7
}
fn clear() -> String {
    "clear".into()
}
fn baseline() -> String {
    "baseline".into()
}
fn seven() -> u32 {
    7
}
fn two() -> u32 {
    2
}
fn scan_map_inspect() -> String {
    "scan_map_inspect".into()
}
fn threshold() -> f64 {
    0.7
}
fn three() -> u32 {
    3
}
fn fifty() -> f64 {
    50.0
}

/// `ScenarioOptions`, field for field, with the reference's defaults.
///
/// Unknown fields are refused, so a typo is an error rather than a silently
/// defaulted option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioOptions {
    #[serde(default = "default_map_preset")]
    pub map_preset: String,
    #[serde(default = "mixed")]
    pub target_motion_preset: String,
    #[serde(default = "mixed")]
    pub drone_mode_preset: String,
    #[serde(default = "procedural")]
    pub terrain_source: String,
    #[serde(default = "alpine")]
    pub terrain_preset: String,
    #[serde(default)]
    pub terrain_seed: Option<i64>,
    #[serde(default)]
    pub dem_path: Option<String>,
    #[serde(default)]
    pub dem_crs: Option<String>,
    #[serde(default = "one")]
    pub detail_strength: f64,
    #[serde(default)]
    pub terrain_resolution_m: Option<f64>,
    #[serde(default = "july")]
    pub season_month: u32,
    #[serde(default = "clear")]
    pub weather_preset: String,
    #[serde(default)]
    pub clean_terrain: bool,
    #[serde(default = "baseline")]
    pub platform_preset: String,
    #[serde(default = "seven")]
    pub ground_station_count: u32,
    #[serde(default)]
    pub target_count: u32,
    #[serde(default = "two")]
    pub drone_count: u32,
    #[serde(default = "scan_map_inspect")]
    pub mission_mode: String,
    #[serde(default = "threshold")]
    pub scan_coverage_threshold: f64,
    #[serde(default = "three")]
    pub poi_count: u32,
    #[serde(default = "fifty")]
    pub coverage_resolution_m: f64,
    #[serde(default)]
    pub cooperative_search: bool,
    #[serde(default)]
    pub search_origin_m: Option<[f64; 2]>,
    #[serde(default)]
    pub adaptive_search: bool,
    #[serde(default)]
    pub enforce_flight_envelope: bool,
    #[serde(default)]
    pub safety_blocking: bool,
    #[serde(default)]
    pub frontier_exploration: bool,
    #[serde(default)]
    pub occlusion_aware_mapping: bool,
    #[serde(default)]
    pub mission_profile_id: Option<String>,
    #[serde(default)]
    pub mission_domain: Option<String>,
    #[serde(default)]
    pub mission_workflow: Option<String>,
    #[serde(default)]
    pub mission_tags: Vec<String>,
    /// R3: string a transmission line across the map. Off by default, so
    /// every existing scenario's world is unchanged; written only when on,
    /// so the replay's provenance is too.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub power_lines: bool,
    /// R3: inspect the `power_lines` line from low down. The mission's POIs
    /// stand beside the line just over the wires, alternately on either
    /// side, so a drone flying from one to the next crosses the wires low.
    /// Needs `power_lines` and a `scan_map_inspect` mission without a
    /// profile. Off by default; written only when on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub power_line_inspection: bool,
    /// R2: scatter trees (`tree-v1`) over the map's forest and scrub. Off
    /// by default, so every existing scenario's world is unchanged; written
    /// only when on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub trees: bool,
    /// R2: inspect among the `trees` from inside the canopy. The mission's
    /// POIs stand beside forest trees, below the crown tops, so a drone
    /// flying from one to the next passes through crowns. Needs `trees` and
    /// a `scan_map_inspect` mission without a profile. Off by default;
    /// written only when on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tree_inspection: bool,
    /// R1: real buildings, an `argusnet.buildings` document
    /// (`scripts/import_buildings.py`), in place of the procedural
    /// buildings. Absent by default; written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buildings_path: Option<String>,
    /// R5: real land cover, an `argusnet.features` document
    /// (`scripts/import_features.py`): water, land use and roads, painted
    /// in place of the procedural land cover. Absent by default; written
    /// only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features_path: Option<String>,
    /// R5: real terrain, an `argusnet.terrain` document
    /// (`scripts/import_terrain.py`) on this map preset's grid, in place of
    /// the procedural terrain. Absent by default; written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terrain_path: Option<String>,
}

impl Default for ScenarioOptions {
    fn default() -> Self {
        serde_json::from_value(json!({})).expect("every option has a default")
    }
}

fn quarter() -> f64 {
    0.25
}

/// An `argusnet.scenario-config` v1 document: the options, the seed and the
/// run length -- what `argusnet sim`'s flags describe, as data.
///
/// The run length is `steps`, or `duration_s` (default 180 s, as
/// `SimulationConfig.from_duration` gives `ceil(duration / dt) + 1` ticks),
/// at `dt_s` (default 0.25 s).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioConfig {
    pub format: String,
    pub version: u32,
    #[serde(default = "seven")]
    pub seed: u32,
    #[serde(default = "quarter")]
    pub dt_s: f64,
    #[serde(default)]
    pub steps: Option<u64>,
    #[serde(default)]
    pub duration_s: Option<f64>,
    #[serde(default)]
    pub options: ScenarioOptions,
}

/// Build the scenario bundle a config describes, as `argusnet sim
/// --export-bundle` would write it: the tracker is the one that command runs
/// with (`DEFAULT_SIMULATION_TRACKER_CONFIG`).
pub fn build_bundle(config: &ScenarioConfig) -> Result<Value, BuildError> {
    let world = build_world(&config.options)?;
    scenario::build_bundle(
        scenario::Assembly {
            config,
            tracker: scenario::TABLES["tracker_simulation"].clone(),
        },
        &world,
    )
}

/// Why a config could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// Not a config this build reads.
    Invalid(String),
    /// A valid config asking for something the native builder does not do.
    Unsupported(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Invalid(detail) => write!(formatter, "invalid scenario config: {detail}"),
            BuildError::Unsupported(detail) => {
                write!(formatter, "unsupported by the native builder: {detail}")
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl ScenarioConfig {
    pub fn from_json(text: &str) -> Result<Self, BuildError> {
        let config: Self =
            serde_json::from_str(text).map_err(|error| BuildError::Invalid(error.to_string()))?;
        if config.format != CONFIG_FORMAT {
            return Err(BuildError::Invalid(format!(
                "format {:?}, expected {CONFIG_FORMAT:?}",
                config.format
            )));
        }
        if config.version != CONFIG_VERSION {
            return Err(BuildError::Invalid(format!(
                "version {}, this build reads {CONFIG_VERSION}",
                config.version
            )));
        }
        Ok(config)
    }
}

/// `MAP_PRESET_SCALES`.
pub fn map_scale_for_preset(map_preset: &str) -> Result<f64, BuildError> {
    Ok(match map_preset {
        "small" => 1.0,
        "medium" => 1.75,
        "large" => 2.5,
        "xlarge" => 4.0,
        "regional" => 6.0,
        "theater" => 10.0,
        "operational" => 15.0,
        other => return Err(BuildError::Invalid(format!("unknown map preset {other:?}"))),
    })
}

/// `map_bounds_for_scale`.
pub fn map_bounds_for_scale(scale: f64) -> MapBounds {
    MapBounds {
        x_min_m: -MAP_BOUNDS_X_EXTENT_M * scale,
        x_max_m: MAP_BOUNDS_X_EXTENT_M * scale,
        y_min_m: MAP_BOUNDS_Y_MIN_M * scale,
        y_max_m: MAP_BOUNDS_Y_MAX_M * scale,
    }
}

/// `terrain_resolution_for_scale`.
pub fn terrain_resolution_for_scale(scale: f64) -> f64 {
    if scale <= 2.5 {
        5.0
    } else if scale <= 4.0 {
        10.0
    } else {
        15.0
    }
}

/// Python's `repr(float)`: the shortest string that reads back as the same
/// float, fixed-point for exponents in `-4..16` and scientific (with a signed,
/// at-least-two-digit exponent) outside it. `_stable_seed` hashes `str(part)`,
/// so a float part must be rendered exactly as Python renders it.
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    // Rust's `{:e}` is the shortest round-trip digit string, in scientific.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("{:e} has an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{whole}.{fraction}")
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// The world a config describes, built.
pub struct BuiltWorld {
    pub scale: f64,
    pub bounds: MapBounds,
    pub terrain: BuiltTerrain,
    pub terrain_config: TerrainBuildConfig,
    pub obstacles: Vec<environment::BuiltObstacle>,
    pub land_cover: CoverGrid,
    pub land_cover_resolution_m: f64,
    /// Empty unless the `power_lines` option asks for them.
    pub power_lines: Vec<argusnet_world::powerline::PowerLine>,
    /// Empty unless the `trees` option asks for them.
    pub trees: Vec<argusnet_world::trees::Tree>,
    /// With `buildings_path`: the georeference, the document's hash and the
    /// counts. `None` otherwise.
    pub buildings: Option<Value>,
    /// With `terrain_path`: the georeference and the document's hashes.
    /// `None` otherwise.
    pub terrain_import: Option<Value>,
    /// With `features_path`: the georeference, the document's hashes and
    /// the polygons by class. `None` otherwise.
    pub features: Option<Value>,
}

/// `build_terrain_layer` + `build_default_environment`.
pub fn build_world(options: &ScenarioOptions) -> Result<BuiltWorld, BuildError> {
    if options.terrain_source != "procedural" || options.dem_path.is_some() {
        return Err(BuildError::Unsupported(format!(
            "terrain_source {:?}: DEM and hybrid terrain are imported, not built (P4.3)",
            options.terrain_source
        )));
    }
    let scale = map_scale_for_preset(&options.map_preset)?;
    let bounds = map_bounds_for_scale(scale);
    let terrain_config = TerrainBuildConfig {
        terrain_preset: options.terrain_preset.clone(),
        terrain_seed: options.terrain_seed,
        detail_strength: options.detail_strength,
        terrain_resolution_m: Some(
            options
                .terrain_resolution_m
                .unwrap_or_else(|| terrain_resolution_for_scale(scale)),
        ),
        season_month: options.season_month,
        tile_size_cells: TILE_SIZE_CELLS,
        ground_plane_m: 0.0,
    };
    // Real terrain in place of the procedural one (R5), else the preset's.
    let imported_terrain = match &options.terrain_path {
        None => None,
        Some(path) => Some((path, terrain_import::load(path)?)),
    };
    let terrain = match &imported_terrain {
        Some((path, (document, _))) => {
            terrain_import::terrain(path, document, &options.map_preset, bounds, &terrain_config)?
        }
        None => build_terrain_layer(
            &terrain_config,
            Bounds2D {
                x_min_m: bounds.x_min_m,
                x_max_m: bounds.x_max_m,
                y_min_m: bounds.y_min_m,
                y_max_m: bounds.y_max_m,
            },
            TerrainSource::Procedural,
        )
        .map_err(BuildError::Invalid)?,
    };

    // Every real layer must stand in one frame: each origin is checked
    // against the first layer's.
    let mut origins: Vec<(&str, f64, f64)> = Vec::new();
    if let Some((_, (document, _))) = &imported_terrain {
        origins.push((
            "terrain_path",
            document.georeference.origin_lat_deg,
            document.georeference.origin_lon_deg,
        ));
    }
    let mut environment = build_environment(
        &terrain,
        scale,
        bounds,
        &options.map_preset,
        &options.terrain_preset,
        options.clean_terrain,
    );
    // Real buildings in place of the procedural ones (R1).
    let buildings = match &options.buildings_path {
        None => None,
        Some(path) => {
            let (document, hash) = buildings::load(path)?;
            origins.push((
                "buildings_path",
                document.georeference.origin_lat_deg,
                document.georeference.origin_lon_deg,
            ));
            let placed = buildings::place(&document, hash, &terrain, bounds)?;
            environment
                .obstacles
                .retain(|obstacle| obstacle.blocker_type != "building");
            environment.obstacles.extend(placed.obstacles);
            Some(placed.record)
        }
    };
    // Real land cover (R5): the data's, with none of the procedural
    // environment's invented walls or forest stands (its patches are never
    // read: the data's cover is painted from the document alone).
    let features = match &options.features_path {
        None => None,
        Some(path) => {
            let (document, hash) = features::load(path)?;
            origins.push((
                "features_path",
                document.georeference.origin_lat_deg,
                document.georeference.origin_lon_deg,
            ));
            environment
                .obstacles
                .retain(|o| o.blocker_type != "wall" && o.blocker_type != "vegetation");
            Some((document, hash))
        }
    };
    if let Some(first) = origins.first() {
        for other in &origins[1..] {
            features::check_same_origin(*first, *other)?;
        }
    }
    let obstacles: Vec<Obstacle> = environment
        .obstacles
        .iter()
        .map(|built| {
            serde_json::from_value::<ObstacleMetadata>(built.metadata.clone())
                .map_err(|error| BuildError::Invalid(error.to_string()))?
                .into_obstacle()
                .map_err(|error| BuildError::Invalid(error.to_string()))
        })
        .collect::<Result<_, _>>()?;
    // `max(10.0, terrain.base_resolution_m * 2.0)`, and the seed
    // `_stable_seed(0, "land-cover", terrain_preset, map_preset, scale)`,
    // whose float part hashes as Python prints it.
    let land_cover_resolution_m = (terrain.resolution_m * 2.0).max(10.0);
    let scale_text = python_float_repr(scale);
    let land_cover_seed = stable_seed(
        0,
        &[
            SeedPart::Str("land-cover"),
            SeedPart::Str(&options.terrain_preset),
            SeedPart::Str(&options.map_preset),
            SeedPart::Str(&scale_text),
        ],
    );
    let season = season_from_month(options.season_month).map_err(BuildError::Invalid)?;
    let cover_for = |season| {
        if let Some((document, _)) = &features {
            return build_data_land_cover(
                &terrain,
                &obstacles,
                &features::painted(document, season, land_cover_resolution_m),
                land_cover_resolution_m,
            )
            .map_err(BuildError::Invalid);
        }
        build_land_cover_layer(LandCoverRequest {
            terrain: &terrain,
            obstacles: &obstacles,
            patches: &environment.patches,
            resolution_m: land_cover_resolution_m,
            season,
            terrain_preset: &options.terrain_preset,
            seed: i64::from(land_cover_seed),
            suppress_vegetation: options.clean_terrain,
        })
        .map_err(BuildError::Invalid)
    };
    let land_cover = cover_for(season)?;
    // The data's trees and lines (R5.2b) stand where they were mapped; with
    // a features file the procedural line is not built.
    let placed = features
        .as_ref()
        .map(|(document, _)| features::place(document, &terrain, bounds, &obstacles, season));
    let power_lines = match &placed {
        Some(placed) => placed.power_lines.clone(),
        None if options.power_lines => power_lines::build_power_lines(&terrain, bounds, &obstacles),
        None => Vec::new(),
    };
    let mapped_trees: &[argusnet_world::trees::Tree] = placed
        .as_ref()
        .map_or(&[], |placed| placed.trees.as_slice());
    // The world's, not the run's: seeded like the land cover, from the
    // presets and the terrain's effective seed (the one it was built from,
    // given or derived), so every run seed sees one forest and an equal
    // terrain configuration the same forest.
    let trees = if options.trees {
        let seed = stable_seed(
            0,
            &[
                SeedPart::Str("trees"),
                SeedPart::Int(terrain_config.effective_seed()),
                SeedPart::Str(&options.terrain_preset),
                SeedPart::Str(&options.map_preset),
                SeedPart::Str(&scale_text),
            ],
        );
        // Snow paints over forest in winter, but the trees stay: they are
        // placed on the cover as it is without snow.
        let snowless;
        let placement_cover = if season.snow_cover {
            snowless = cover_for(argusnet_world::land_cover_build::SeasonalVariation {
                snow_cover: false,
                ..season
            })?;
            &snowless
        } else {
            &land_cover
        };
        trees::build_trees(
            &trees::TreeGround {
                terrain: &terrain,
                bounds,
                land_cover: placement_cover,
                land_cover_resolution_m,
                obstacles: &obstacles,
                season,
                terrain_preset: &options.terrain_preset,
                existing: mapped_trees,
            },
            seed,
        )
    } else {
        Vec::new()
    };
    let trees: Vec<argusnet_world::trees::Tree> =
        mapped_trees.iter().cloned().chain(trees).collect();

    Ok(BuiltWorld {
        scale,
        bounds,
        terrain,
        terrain_config,
        obstacles: environment.obstacles,
        land_cover,
        land_cover_resolution_m,
        power_lines,
        trees,
        buildings,
        terrain_import: imported_terrain
            .map(|(_, (document, hash))| terrain_import::record(&document, hash)),
        features: features
            .zip(placed)
            .map(|((document, hash), placed)| features::record(&document, hash, &placed)),
    })
}

fn bounds_document(bounds: MapBounds) -> Value {
    json!({
        "x_min_m": bounds.x_min_m,
        "x_max_m": bounds.x_max_m,
        "y_min_m": bounds.y_min_m,
        "y_max_m": bounds.y_max_m,
    })
}

impl BuiltWorld {
    /// The bundle's `world` section: `terrain_new`, `obstacles_new` and
    /// `land_cover_new` payloads, the optical visibility profile and the
    /// sensing clearance.
    pub fn world_document(&self) -> Value {
        let terrain = &self.terrain.tiled;
        let tiles: Vec<Value> = terrain
            .tiles
            .values()
            .map(|tile| {
                json!({
                    "tx": tile.tx,
                    "ty": tile.ty,
                    "lod": tile.lod,
                    "x_min_m": tile.x_min_m,
                    "y_min_m": tile.y_min_m,
                    "cell_size_m": tile.cell_size_m,
                    "heights_m": tile.heights_m,
                })
            })
            .collect();
        let mut world = json!({
            "terrain": {
                "bounds": bounds_document(self.bounds),
                "tile_size_cells": terrain.tile_size_cells,
                "base_resolution_m": terrain.base_resolution_m,
                "floor_m": terrain.floor_m,
                "tiles": tiles,
            },
            "obstacles": {
                "bounds": bounds_document(self.bounds),
                "tile_size_m": terrain.tile_size_cells as f64 * terrain.base_resolution_m,
                "primitives": self.obstacles.iter().map(|o| o.metadata.clone()).collect::<Vec<_>>(),
            },
            "land_cover": self.land_cover_document(),
            "visibility_profile": VisibilityProfile::default(),
            "los_terrain_clearance_m": LOS_TERRAIN_CLEARANCE_M,
        });
        // Written only when there are some, so a world without lines is
        // the bundle it was before lines existed.
        if !self.power_lines.is_empty() {
            world["power_lines"] = json!(self.power_lines);
        }
        if !self.trees.is_empty() {
            world["trees"] = json!(self.trees);
        }
        if let Some(buildings) = &self.buildings {
            world["buildings"] = buildings.clone();
        }
        if let Some(terrain_import) = &self.terrain_import {
            world["terrain_import"] = terrain_import.clone();
        }
        if let Some(features) = &self.features {
            world["features"] = features.clone();
        }
        world
    }

    /// `LandCoverLayer.from_rasters`: 256-cell tiles, zero-padded past the
    /// raster's edge.
    fn land_cover_document(&self) -> Value {
        let grid = &self.land_cover;
        let size = TILE_SIZE_CELLS;
        let (rows, cols) = (grid.rows, grid.cols);
        let resolution = self.land_cover_resolution_m;
        let mut tiles = Vec::new();
        for ty in 0..rows.div_ceil(size) {
            for tx in 0..cols.div_ceil(size) {
                let tile = |values: &[u8]| -> Vec<Vec<u8>> {
                    (0..size)
                        .map(|r| {
                            (0..size)
                                .map(|c| {
                                    let (row, col) = (ty * size + r, tx * size + c);
                                    if row < rows && col < cols {
                                        values[row * cols + col]
                                    } else {
                                        0
                                    }
                                })
                                .collect()
                        })
                        .collect()
                };
                tiles.push(json!({
                    "tx": tx,
                    "ty": ty,
                    "x_min_m": self.bounds.x_min_m + (tx * size) as f64 * resolution,
                    "y_min_m": self.bounds.y_min_m + (ty * size) as f64 * resolution,
                    "cell_size_m": resolution,
                    "classes": tile(&grid.classes),
                    "density": tile(&grid.density),
                }));
            }
        }
        json!({
            "bounds": bounds_document(self.bounds),
            "tile_size_cells": size,
            "resolution": resolution,
            "tiles": tiles,
        })
    }

    /// `TerrainLayer.terrain_summary`.
    pub fn terrain_summary(&self) -> Value {
        let terrain = &self.terrain.tiled;
        let heights = || {
            terrain
                .tiles
                .values()
                .flat_map(|t| t.heights_m.iter().flatten())
        };
        let min_height = heights().copied().fold(f64::INFINITY, f64::min);
        let max_height = heights().copied().fold(f64::NEG_INFINITY, f64::max);
        let masks = &self.terrain.masks;
        let named = [
            ("lowland", &masks.lowland),
            ("ridge", &masks.ridge),
            ("road", &masks.road),
            ("steep_slope", &masks.steep_slope),
            ("valley", &masks.valley),
            ("water", &masks.water),
        ];
        let names: Vec<&str> = named.iter().map(|(name, _)| *name).collect();
        let coverage: Map<String, Value> = named
            .iter()
            .map(|(name, mask)| ((*name).to_string(), json!(mask_coverage_fraction(mask))))
            .collect();
        let profile = self.terrain.profile;
        let config = &self.terrain_config;
        let mut summary = json!({
            "kind": "tiled-heightmap-v1",
            "base_resolution_m": terrain.base_resolution_m,
            "tile_size_cells": terrain.tile_size_cells,
            "lod_resolutions_m": [terrain.base_resolution_m],
            "ground_plane_m": terrain.floor_m,
            "min_height_m": min_height,
            "max_height_m": max_height,
            "semantic_masks": names,
            "mask_coverage_fraction": coverage,
            "source": {
                "source": "procedural",
                "preset": config.terrain_preset,
                "seed": config.effective_seed(),
                "detail_strength": config.detail_strength,
                "season_month": config.season_month,
                "generator_version": "procedural-landscape-v1",
                "profile": {
                    "name": profile.name,
                    "base_elevation_m": profile.base_elevation_m,
                    "relief_m": profile.relief_m,
                    "roughness": profile.roughness,
                    "moisture": profile.moisture,
                    "water_bias": profile.water_bias,
                    "snow_bias": profile.snow_bias,
                    "vegetation_bias": profile.vegetation_bias,
                    "urban_bias": profile.urban_bias,
                    "hydrology_strength": profile.hydrology_strength,
                    "smoothing_passes": profile.smoothing_passes,
                },
                "semantic_masks": names,
                "mask_coverage_fraction": coverage,
            },
        });
        // An imported terrain says so, and where it came from, instead of
        // the generator it did not use.
        if let Some(terrain_import) = &self.terrain_import {
            summary["source"] = json!({
                "source": "dem-import",
                "preset": config.terrain_preset,
                "import": terrain_import,
                "semantic_masks": names,
                "mask_coverage_fraction": summary["mask_coverage_fraction"].clone(),
            });
        }
        summary
    }

    /// `TerrainLayer.viewer_mesh`: the terrain sampled on a `linspace` grid of
    /// at most 128 x 128, rows of heights, nested as the viewer reads them.
    pub fn viewer_mesh(&self) -> Value {
        let b = self.bounds;
        let resolution = self.terrain.tiled.base_resolution_m;
        let count = |extent: f64| {
            ((extent / resolution).ceil() as usize).clamp(2, VIEWER_MESH_MAX_DIMENSION.max(2))
        };
        let cols = count(b.x_max_m - b.x_min_m);
        let rows = count(b.y_max_m - b.y_min_m);
        let heights: Vec<Vec<f64>> = linspace(b.y_min_m, b.y_max_m, rows)
            .into_iter()
            .map(|y| {
                linspace(b.x_min_m, b.x_max_m, cols)
                    .into_iter()
                    .map(|x| self.terrain.height_at(x, y))
                    .collect()
            })
            .collect();
        json!({
            "x_min_m": b.x_min_m,
            "x_max_m": b.x_max_m,
            "y_min_m": b.y_min_m,
            "y_max_m": b.y_max_m,
            "cols": cols,
            "rows": rows,
            "heights_m": heights,
        })
    }

    /// The world half of `scenario_replay_metadata`: every key that the
    /// terrain, obstacles and land cover decide.
    pub fn presentation_world(&self) -> Map<String, Value> {
        let mut terrain = self.terrain_summary();
        terrain["xy_bounds_m"] = bounds_document(self.bounds);
        terrain["viewer_mesh"] = self.viewer_mesh();
        let legend = json!({
            "open": 0, "urban": 1, "forest": 2, "water": 3, "scrub": 4,
            "wetland": 5, "rocky": 6, "snow": 7, "road": 8,
        });
        let mut map = Map::new();
        map.insert("terrain".into(), terrain);
        map.insert(
            "occluding_objects".into(),
            json!(self
                .obstacles
                .iter()
                .map(|o| o.metadata.clone())
                .collect::<Vec<_>>()),
        );
        // `f"procedural-{scale:.2f}"`.
        map.insert(
            "environment_id".into(),
            json!(format!("procedural-{:.2}", self.scale)),
        );
        if !self.power_lines.is_empty() {
            map.insert("power_lines".into(), json!(self.power_lines));
        }
        if !self.trees.is_empty() {
            map.insert("trees".into(), json!(self.trees));
        }
        if let Some(buildings) = &self.buildings {
            map.insert("buildings".into(), buildings.clone());
        }
        if let Some(terrain_import) = &self.terrain_import {
            map.insert("terrain_import".into(), terrain_import.clone());
        }
        if let Some(features) = &self.features {
            map.insert("features".into(), features.clone());
        }
        map.insert("crs_id".into(), json!("local-enu"));
        map.insert("environment_bounds_m".into(), bounds_document(self.bounds));
        map.insert("terrain_summary".into(), self.terrain_summary());
        map.insert(
            "land_cover".into(),
            json!({
                "kind": "tiled-landcover-v1",
                "base_resolution_m": self.land_cover_resolution_m,
                "tile_size_cells": TILE_SIZE_CELLS,
                "xy_bounds_m": bounds_document(self.bounds),
                "legend": legend,
            }),
        );
        map.insert("land_cover_legend".into(), legend);
        map.insert(
            "enu_origin".into(),
            // Real terrain, buildings or features bring their own origin:
            // the run stands where they do. The terrain's first, whose
            // altitude is the heights' offset (the layers' origins are
            // checked equal). Otherwise the reference's fixed one.
            match self
                .terrain_import
                .as_ref()
                .or(self.buildings.as_ref())
                .or(self.features.as_ref())
                .and_then(|record| record.get("georeference"))
            {
                Some(georeference) => json!({
                    "latitude_deg": georeference["origin_lat_deg"],
                    "longitude_deg": georeference["origin_lon_deg"],
                    "altitude_m": georeference["origin_h_m"],
                }),
                None => {
                    json!({"latitude_deg": 37.7749, "longitude_deg": -122.4194, "altitude_m": 0.0})
                }
            },
        );
        map
    }
}

/// `np.linspace(start, stop, num)` with its endpoint: `i * step + start`, and
/// the last value set to `stop` exactly.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut values: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    values[num - 1] = stop;
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_repr_matches_cpython() {
        for (value, expected) in [
            (1.0, "1.0"),
            (1.75, "1.75"),
            (6.0, "6.0"),
            (15.0, "15.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (123.456, "123.456"),
            (0.0, "0.0"),
            (1.5e300, "1.5e+300"),
        ] {
            assert_eq!(python_float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn linspace_ends_on_its_stop() {
        let values = linspace(-330.0, 360.0, 128);
        assert_eq!(values.len(), 128);
        assert_eq!(values[0], -330.0);
        assert_eq!(values[127], 360.0);
        // Where `i * step + start` overshoots, numpy sets the last value to
        // `stop` itself: 10 * 0.42 - 1.3 is 2.9000000000000004.
        let values = linspace(-1.3, 2.9, 11);
        assert_eq!(values[10], 2.9);
        assert_eq!(
            values[9].to_bits(),
            (9.0 * ((2.9 - -1.3) / 10.0) + -1.3f64).to_bits()
        );
    }

    #[test]
    fn a_config_defaults_every_option_and_refuses_unknown_ones() {
        let config =
            ScenarioConfig::from_json(r#"{"format":"argusnet.scenario-config","version":1}"#)
                .unwrap();
        assert_eq!(config.seed, 7);
        assert_eq!(config.options.map_preset, "regional");
        assert_eq!(config.options.ground_station_count, 7);
        assert!(matches!(
            ScenarioConfig::from_json(
                r#"{"format":"argusnet.scenario-config","version":1,"options":{"map_presett":"small"}}"#
            ),
            Err(BuildError::Invalid(_))
        ));
    }

    #[test]
    fn a_config_of_another_format_or_version_is_refused_by_name() {
        let refused = |text: &str| match ScenarioConfig::from_json(text) {
            Err(BuildError::Invalid(reason)) => reason,
            other => panic!("accepted {text}: {}", other.is_ok()),
        };
        let reason = refused(r#"{"format":"argusnet.scenario-bundle","version":1}"#);
        assert!(reason.contains("argusnet.scenario-bundle"), "{reason}");
        let reason = refused(r#"{"format":"argusnet.scenario-config","version":2}"#);
        assert!(reason.contains("version 2"), "{reason}");
    }
}
