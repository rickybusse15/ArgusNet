//! Real terrain in a built world (R5): the `terrain_path` option.
//!
//! The file is an `argusnet.terrain` v1 document, written by the offline
//! importer (`scripts/import_terrain.py`) from a DEM GeoTIFF. It is already
//! the run's height grid: the map preset's bounds at its terrain
//! resolution, in the local east-north-up frame around a geodetic origin,
//! heights above `origin_h_m`. The builder checks it is the grid this run
//! would build (same preset, bounds, resolution and shape), tiles it in
//! place of the procedural terrain, and builds everything else on it as
//! usual: buildings stand on it, trees and lines follow it.
//!
//! With real buildings too, both files must share an origin, so the layers
//! line up; a mismatch fails the build by name. The georeference and the
//! document's hash go into the bundle and the replay. Without the option
//! nothing changes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use argusnet_world::grid_ops::Grid;
use argusnet_world::landscape::{terrain_from_heights, BuiltTerrain, TerrainBuildConfig};
use argusnet_world::noise::Bounds2D;

use super::environment::MapBounds;
use super::BuildError;

pub const TERRAIN_FORMAT: &str = "argusnet.terrain";
pub const TERRAIN_VERSION: u32 = 1;

/// Where the document's local frame is on Earth, and its vertical offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Georeference {
    /// The DEM's CRS, e.g. `EPSG:32611`.
    pub source_crs_id: String,
    /// Always `local-enu`.
    pub runtime_crs_id: String,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    /// Heights are the DEM's elevations less this, metres.
    pub origin_h_m: f64,
    /// The heights' vertical datum: `dem`, the DEM's own.
    pub z_datum: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainDocument {
    pub format: String,
    pub version: u32,
    pub georeference: Georeference,
    pub map_preset: String,
    pub bounds_xy_m: Bounds,
    pub resolution_m: f64,
    pub rows: usize,
    pub cols: usize,
    /// Row-major, rows walking y from `y_min_m`, columns x from `x_min_m`.
    pub heights_m: Vec<f64>,
    #[serde(default)]
    pub source_sha256: String,
    #[serde(default)]
    pub summary: Value,
}

/// Read and check the document at `path`.
pub fn load(path: &str) -> Result<(TerrainDocument, u64), BuildError> {
    let invalid = |message: String| BuildError::Invalid(format!("terrain_path {path}: {message}"));
    let bytes = std::fs::read(path).map_err(|error| invalid(error.to_string()))?;
    let document: TerrainDocument =
        serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    if document.format != TERRAIN_FORMAT || document.version != TERRAIN_VERSION {
        return Err(invalid(format!(
            "expected {TERRAIN_FORMAT} v{TERRAIN_VERSION}, got {} v{}",
            document.format, document.version
        )));
    }
    if document.georeference.runtime_crs_id != "local-enu" {
        return Err(invalid(format!(
            "runtime_crs_id must be local-enu, got {}",
            document.georeference.runtime_crs_id
        )));
    }
    Ok((document, crate::config::fnv1a(&bytes)))
}

/// The run's terrain from `document`, which must be this run's grid:
/// `map_preset`'s `bounds` at `config`'s resolution.
pub fn terrain(
    path: &str,
    document: &TerrainDocument,
    map_preset: &str,
    bounds: MapBounds,
    config: &TerrainBuildConfig,
) -> Result<BuiltTerrain, BuildError> {
    let invalid = |message: String| BuildError::Invalid(format!("terrain_path {path}: {message}"));
    if document.map_preset != map_preset {
        return Err(invalid(format!(
            "imported for map_preset {:?}, the run's is {map_preset:?}",
            document.map_preset
        )));
    }
    let b = &document.bounds_xy_m;
    if (b.x_min_m, b.x_max_m, b.y_min_m, b.y_max_m)
        != (
            bounds.x_min_m,
            bounds.x_max_m,
            bounds.y_min_m,
            bounds.y_max_m,
        )
    {
        return Err(invalid(format!(
            "bounds {b:?} are not the {map_preset} map's {bounds:?}"
        )));
    }
    if document.resolution_m != config.resolution_m() {
        return Err(invalid(format!(
            "resolution {} m is not the run's terrain resolution {} m",
            document.resolution_m,
            config.resolution_m()
        )));
    }
    let heights = Grid::new(document.rows, document.cols, document.heights_m.clone())
        .map_err(|error| invalid(error.to_string()))?;
    terrain_from_heights(
        config,
        Bounds2D {
            x_min_m: bounds.x_min_m,
            x_max_m: bounds.x_max_m,
            y_min_m: bounds.y_min_m,
            y_max_m: bounds.y_max_m,
        },
        heights,
    )
    .map_err(invalid)
}

/// The bundle's and the replay's record of the terrain a world stands on.
pub fn record(document: &TerrainDocument, hash: u64) -> Value {
    json!({
        "format": TERRAIN_FORMAT,
        "version": TERRAIN_VERSION,
        "georeference": document.georeference,
        "content_fnv1a": format!("{hash:016x}"),
        "source_sha256": document.source_sha256,
        "map_preset": document.map_preset,
        "rows": document.rows,
        "cols": document.cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{build_bundle, build_world, ScenarioConfig, ScenarioOptions};
    use argusnet_world::noise::grid_values;

    const TOWN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/buildings_town.buildings.json"
    );

    /// A ground that is not any preset's: a tilted, rolling surface.
    fn ground(x: f64, y: f64) -> f64 {
        100.0 + 0.02 * x - 0.01 * y + 6.0 * (x / 90.0).sin() * (y / 70.0).cos()
    }

    /// The document the importer would write for `map_preset` around the
    /// synthetic town's origin.
    fn document(map_preset: &str) -> Value {
        let scale = crate::builder::map_scale_for_preset(map_preset).unwrap();
        let bounds = crate::builder::map_bounds_for_scale(scale);
        let resolution = crate::builder::terrain_resolution_for_scale(scale);
        let (xs, ys) = grid_values(
            Bounds2D {
                x_min_m: bounds.x_min_m,
                x_max_m: bounds.x_max_m,
                y_min_m: bounds.y_min_m,
                y_max_m: bounds.y_max_m,
            },
            resolution,
        );
        let heights: Vec<f64> = ys
            .iter()
            .flat_map(|y| xs.iter().map(move |x| ground(*x, *y)))
            .collect();
        json!({
            "format": TERRAIN_FORMAT,
            "version": TERRAIN_VERSION,
            "georeference": {
                "source_crs_id": "EPSG:32632",
                "runtime_crs_id": "local-enu",
                "origin_lat_deg": 45.999892,
                "origin_lon_deg": 7.0,
                "origin_h_m": 1180.0,
                "z_datum": "dem",
            },
            "map_preset": map_preset,
            "bounds_xy_m": {
                "x_min_m": bounds.x_min_m, "x_max_m": bounds.x_max_m,
                "y_min_m": bounds.y_min_m, "y_max_m": bounds.y_max_m,
            },
            "resolution_m": resolution,
            "rows": ys.len(),
            "cols": xs.len(),
            "heights_m": heights,
            "source_sha256": "ab",
        })
    }

    fn write(name: &str, document: &Value) -> String {
        let dir = std::env::temp_dir().join("argusnet-terrain-import-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}-{}.json", std::process::id()));
        std::fs::write(&path, document.to_string()).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn options(extra: Value) -> ScenarioOptions {
        let mut options = json!({"terrain_preset": "urban_flat", "map_preset": "small"});
        for (key, value) in extra.as_object().unwrap() {
            options[key] = value.clone();
        }
        serde_json::from_value(options).unwrap()
    }

    fn error(options: ScenarioOptions) -> String {
        match build_world(&options) {
            Err(BuildError::Invalid(message)) => message,
            Err(other) => panic!("unexpected {other:?}"),
            Ok(_) => panic!("built"),
        }
    }

    /// The run's ground is the imported one, at every grid node, and
    /// between them as the procedural terrain is (the same tiling).
    #[test]
    fn the_run_stands_on_the_imported_ground() {
        let path = write("ground", &document("small"));
        let world = build_world(&options(json!({"terrain_path": path}))).expect("builds");
        for (x, y) in [(0.0, 0.0), (-360.0, -330.0), (355.0, 360.0), (125.0, -40.0)] {
            let height = world.terrain.height_at(x, y);
            assert!(
                (height - ground(x, y)).abs() < 1e-9,
                "at {x}, {y}: {height} against {}",
                ground(x, y)
            );
        }
        // Between nodes: bilinear, within what the surface bends.
        assert!((world.terrain.height_at(2.5, 2.5) - ground(2.5, 2.5)).abs() < 0.05);
        // No mask is invented from the ground's shape.
        let masks = &world.terrain.masks;
        assert!(masks.water.values.iter().all(|v| !v));
        assert!(masks.road.values.iter().all(|v| !v));
        // Not the procedural terrain it replaces.
        let procedural = build_world(&options(json!({}))).unwrap();
        assert!((procedural.terrain.height_at(125.0, -40.0) - ground(125.0, -40.0)).abs() > 1.0);
    }

    /// The bundle and the replay say which terrain and where on Earth; the
    /// run's origin is the terrain's, its altitude the heights' offset.
    #[test]
    fn the_bundle_records_the_imported_terrain() {
        let path = write("record", &document("small"));
        let config = |options: Value| -> ScenarioConfig {
            serde_json::from_value(json!({
                "format": "argusnet.scenario-config",
                "version": 1,
                "duration_s": 5.0,
                "options": options,
            }))
            .unwrap()
        };
        let base = json!({"terrain_preset": "urban_flat", "map_preset": "small"});
        let mut with = base.clone();
        with["terrain_path"] = json!(path);
        let bundle = build_bundle(&config(with)).expect("bundle");
        let record = &bundle["world"]["terrain_import"];
        assert_eq!(record["format"], json!(TERRAIN_FORMAT));
        assert_eq!(record["georeference"]["origin_h_m"], json!(1180.0));
        assert_eq!(bundle["presentation"]["terrain_import"], *record);
        let origin = &bundle["presentation"]["enu_origin"];
        assert_eq!(origin["latitude_deg"], json!(45.999892));
        assert_eq!(origin["altitude_m"], json!(1180.0));
        assert_eq!(
            bundle["presentation"]["terrain_summary"]["source"]["source"],
            json!("dem-import")
        );
        assert_eq!(
            bundle["presentation"]["scenario_options"]["terrain_path"],
            json!(path)
        );
        // Without the option, none of it.
        let plain = build_bundle(&config(base)).unwrap();
        assert!(plain["world"].get("terrain_import").is_none());
        assert!(plain["presentation"].get("terrain_import").is_none());
        assert!(plain["presentation"]["scenario_options"]
            .get("terrain_path")
            .is_none());
        assert_eq!(
            plain["presentation"]["terrain_summary"]["source"]["source"],
            json!("procedural")
        );
    }

    /// Real buildings stand on real ground when both share an origin, and
    /// fail the build by name when they do not.
    #[test]
    fn buildings_and_terrain_must_share_an_origin() {
        let path = write("town", &document("regional"));
        let world = build_world(&options(json!({
            "map_preset": "regional", "terrain_path": path, "buildings_path": TOWN,
        })))
        .expect("builds");
        assert_eq!(world.buildings.as_ref().unwrap()["buildings"], json!(10));
        // The run's origin is the terrain's, whose altitude is the heights'
        // offset (the buildings' is 0).
        let origin = &world.presentation_world()["enu_origin"];
        assert_eq!(origin["altitude_m"], json!(1180.0));
        let mut moved = document("regional");
        moved["georeference"]["origin_lon_deg"] = json!(7.001);
        let moved = write("moved", &moved);
        let message = error(options(json!({
            "map_preset": "regional", "terrain_path": moved, "buildings_path": TOWN,
        })));
        assert!(message.contains("different origins"), "{message}");
    }

    /// A document for another grid is refused by name, never stretched.
    #[test]
    fn a_terrain_for_another_grid_is_refused() {
        let small = write("small", &document("small"));
        let message = error(options(
            json!({"map_preset": "medium", "terrain_path": small}),
        ));
        assert!(message.contains("map_preset"), "{message}");

        let mut shifted = document("small");
        shifted["bounds_xy_m"]["x_min_m"] = json!(-355.0);
        let message = error(options(json!({"terrain_path": write("shifted", &shifted)})));
        assert!(message.contains("bounds"), "{message}");

        let mut coarse = document("small");
        coarse["resolution_m"] = json!(10.0);
        let message = error(options(json!({"terrain_path": write("coarse", &coarse)})));
        assert!(message.contains("resolution"), "{message}");

        let mut short = document("small");
        short["heights_m"].as_array_mut().unwrap().pop();
        let message = error(options(json!({"terrain_path": write("short", &short)})));
        assert!(message.contains("length"), "{message}");

        let mut wrong_shape = document("small");
        let (rows, cols) = (wrong_shape["rows"].clone(), wrong_shape["cols"].clone());
        wrong_shape["rows"] = cols;
        wrong_shape["cols"] = rows;
        let message = error(options(
            json!({"terrain_path": write("transposed", &wrong_shape)}),
        ));
        assert!(message.contains("grid"), "{message}");
    }

    /// Heights below the ground plane, or not numbers, are refused: nothing
    /// may stand below the terrain's floor.
    #[test]
    fn heights_below_the_ground_plane_are_refused() {
        let mut low = document("small");
        low["heights_m"][7] = json!(-0.5);
        let message = error(options(json!({"terrain_path": write("low", &low)})));
        assert!(message.contains("ground plane"), "{message}");

        let mut other = document("small");
        other["format"] = json!("argusnet.buildings");
        let message = error(options(json!({"terrain_path": write("other", &other)})));
        assert!(message.contains(TERRAIN_FORMAT), "{message}");

        let mut frame = document("small");
        frame["georeference"]["runtime_crs_id"] = json!("EPSG:32632");
        let message = error(options(json!({"terrain_path": write("frame", &frame)})));
        assert!(message.contains("local-enu"), "{message}");
    }
}
