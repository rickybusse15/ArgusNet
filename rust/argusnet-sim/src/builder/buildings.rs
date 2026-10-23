//! Real buildings in a built world (R1): the `buildings_path` option.
//!
//! The file is an `argusnet.buildings` v1 document, written by the offline
//! importer (`scripts/import_buildings.py`) from GeoJSON or OSM footprints.
//! It is already in the run's local metres, and says how it got there: the
//! source CRS and the geodetic origin of the local east-north-up frame. The
//! builder checks every building against the model (`building-v2`), stands
//! it on the run's own terrain (its base at the lowest ground under its
//! footprint) and replaces the procedural buildings with them. Walls,
//! vegetation and everything else stay. A building not wholly inside the
//! map is left out and counted, never clipped into a different shape.
//!
//! The georeference and the document's hash go into the bundle and the
//! replay, so a run says which buildings it flew among and where on Earth
//! they stand. Without the option nothing changes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use argusnet_world::landscape::BuiltTerrain;
use argusnet_world::obstacles::RoofShape;

use super::environment::{imported_building, BuiltObstacle, MapBounds};
use super::BuildError;

pub const BUILDINGS_FORMAT: &str = "argusnet.buildings";
pub const BUILDINGS_VERSION: u32 = 1;

/// Where the document's local frame is on Earth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Georeference {
    /// The CRS the footprints came in, e.g. `EPSG:4326`.
    pub source_crs_id: String,
    /// Always `local-enu`: metres east, north and up from the origin.
    pub runtime_crs_id: String,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    #[serde(default)]
    pub origin_h_m: f64,
}

/// One building, in local metres, heights above its own ground.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedBuilding {
    pub id: String,
    pub footprint_xy_m: Vec<[f64; 2]>,
    /// Ground to ridge (the whole building), metres.
    pub height_m: f64,
    /// Ground to eave, metres: where the walls end and the roof begins.
    pub eave_height_m: f64,
    pub roof_shape: String,
    #[serde(default)]
    pub ridge_azimuth_rad: Option<f64>,
    /// `measured`, `levels` or `default`: how the height was known.
    #[serde(default)]
    pub height_source: String,
    /// `tagged`, `default` or `unsupported:<shape>`: how the roof was known.
    #[serde(default)]
    pub roof_source: String,
    /// The facade's material class (`brick`, `glass`, ...), `generic` when
    /// the data does not say. Drawn by the viewer; the run ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facade_material: Option<String>,
    /// The roof's material class, `generic` when the data does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roof_material: Option<String>,
    /// Storeys, when the data gives them: the viewer draws window rows only
    /// then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub levels: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildingsDocument {
    pub format: String,
    pub version: u32,
    pub georeference: Georeference,
    pub buildings: Vec<ImportedBuilding>,
}

/// The buildings a world was built with, and what was left out.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedBuildings {
    pub obstacles: Vec<BuiltObstacle>,
    /// The bundle's and the replay's record: georeference, source hash and
    /// counts.
    pub record: Value,
}

/// Read and check the document at `path`.
pub fn load(path: &str) -> Result<(BuildingsDocument, u64), BuildError> {
    let bytes = super::read_import_limited(path)
        .map_err(|error| BuildError::Invalid(format!("buildings_path {path}: {error}")))?;
    let document: BuildingsDocument = serde_json::from_slice(&bytes)
        .map_err(|error| BuildError::Invalid(format!("buildings_path {path}: {error}")))?;
    if document.format != BUILDINGS_FORMAT || document.version != BUILDINGS_VERSION {
        return Err(BuildError::Invalid(format!(
            "buildings_path {path}: expected {BUILDINGS_FORMAT} v{BUILDINGS_VERSION}, got {} v{}",
            document.format, document.version
        )));
    }
    if document.georeference.runtime_crs_id != "local-enu" {
        return Err(BuildError::Invalid(format!(
            "buildings_path {path}: runtime_crs_id must be local-enu, got {}",
            document.georeference.runtime_crs_id
        )));
    }
    Ok((document, crate::config::fnv1a(&bytes)))
}

/// Stand each building on `terrain` inside `bounds`. A building outside the
/// map is left out and counted; one the model refuses fails the build by
/// name (`build_world` reads each through the model), never silently
/// dropped.
pub fn place(
    document: &BuildingsDocument,
    hash: u64,
    terrain: &BuiltTerrain,
    bounds: MapBounds,
) -> Result<PlacedBuildings, BuildError> {
    let inside = |p: &[f64; 2]| {
        p[0] >= bounds.x_min_m
            && p[0] <= bounds.x_max_m
            && p[1] >= bounds.y_min_m
            && p[1] <= bounds.y_max_m
    };
    let mut obstacles = Vec::new();
    let mut outside = 0usize;
    let mut estimated_heights = 0usize;
    for building in &document.buildings {
        if !building.footprint_xy_m.iter().all(inside) {
            outside += 1;
            continue;
        }
        let shape = RoofShape::parse(&building.roof_shape).ok_or_else(|| {
            BuildError::Invalid(format!(
                "building {}: unknown roof_shape {:?}",
                building.id, building.roof_shape
            ))
        })?;
        let base = building
            .footprint_xy_m
            .iter()
            .map(|p| terrain.height_at(p[0], p[1]))
            .fold(f64::INFINITY, f64::min);
        // `build_world` reads every obstacle through the model as soon as
        // these are placed, so one it refuses fails the build by name.
        let (eave, top) = (base + building.eave_height_m, base + building.height_m);
        if building.height_source != "measured" {
            estimated_heights += 1;
        }
        let mut metadata = json!({
            "object_id": building.id,
            "kind": "building-v2",
            "blocker_type": "building",
            "footprint_xy_m": building.footprint_xy_m,
            "base_elevation_m": base,
            "eave_elevation_m": eave,
            "top_elevation_m": top,
            "roof_shape": shape.as_str(),
            "height_source": building.height_source,
            "roof_source": building.roof_source,
        });
        if let Some(azimuth) = building.ridge_azimuth_rad {
            metadata["ridge_azimuth_rad"] = json!(azimuth);
        }
        // How it looks, for the scene; the run reads none of these.
        for (key, value) in [
            (
                "facade_material",
                building.facade_material.as_ref().map(|v| json!(v)),
            ),
            (
                "roof_material",
                building.roof_material.as_ref().map(|v| json!(v)),
            ),
            ("levels", building.levels.map(|v| json!(v))),
        ] {
            if let Some(value) = value {
                metadata[key] = value;
            }
        }
        obstacles.push(imported_building(metadata, building.footprint_xy_m.clone()));
    }
    let record = json!({
        "format": BUILDINGS_FORMAT,
        "version": BUILDINGS_VERSION,
        "georeference": document.georeference,
        "content_fnv1a": format!("{hash:016x}"),
        "buildings": obstacles.len(),
        "left_out_outside_map": outside,
        "estimated_heights": estimated_heights,
    });
    Ok(PlacedBuildings { obstacles, record })
}

#[cfg(test)]
mod tests {
    use crate::builder::{build_bundle, ScenarioConfig};
    use crate::scenario::{build_world, ScenarioBundle};
    use argusnet_world::obstacles::RoofShape;
    use serde_json::{json, Value};

    /// The committed import of the synthetic town (tests/fixtures), which
    /// `tests/test_buildings_import.py` keeps equal to the importer's output.
    const TOWN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/buildings_town.buildings.json"
    );

    fn bundle(buildings_path: Option<&str>) -> Value {
        let mut options = json!({"terrain_preset": "urban_flat"});
        if let Some(path) = buildings_path {
            options["buildings_path"] = json!(path);
        }
        let config: ScenarioConfig = serde_json::from_value(json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "duration_s": 5.0,
            "options": options,
        }))
        .expect("config");
        build_bundle(&config).expect("bundle builds")
    }

    fn kinds(world: &Value) -> Vec<(String, String)> {
        world["obstacles"]["primitives"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                (
                    o["kind"].as_str().unwrap().to_string(),
                    o["blocker_type"].as_str().unwrap_or("building").to_string(),
                )
            })
            .collect()
    }

    /// The imported town replaces the procedural buildings; the run reads
    /// every one with its roof, standing on its own ground; the bundle and
    /// the replay say where on Earth it is.
    #[test]
    fn imported_buildings_replace_the_procedural_ones_in_the_run() {
        let plain = bundle(None);
        let town = bundle(Some(TOWN));
        let before = kinds(&plain["world"]);
        let after = kinds(&town["world"]);
        assert!(before.iter().any(|(_, b)| b == "building"));
        let imported: Vec<_> = after.iter().filter(|(_, b)| b == "building").collect();
        assert_eq!(imported.len(), 10);
        assert!(imported.iter().all(|(kind, _)| kind == "building-v2"));
        // Everything that is not a building is untouched.
        let others = |list: &[(String, String)]| -> Vec<(String, String)> {
            list.iter()
                .filter(|(_, b)| b != "building")
                .cloned()
                .collect()
        };
        assert_eq!(others(&before), others(&after));

        // Recorded: the georeference, the document's hash, the counts.
        let record = &town["world"]["buildings"];
        assert_eq!(record["buildings"], json!(10));
        assert_eq!(record["estimated_heights"], json!(4));
        assert_eq!(record["georeference"]["source_crs_id"], json!("EPSG:4326"));
        assert_eq!(town["presentation"]["buildings"], *record);
        assert_eq!(
            town["presentation"]["enu_origin"]["latitude_deg"],
            record["georeference"]["origin_lat_deg"]
        );
        assert_eq!(
            town["presentation"]["scenario_options"]["buildings_path"],
            json!(TOWN)
        );
        // Without the option: no record, the reference origin.
        assert!(plain["world"].get("buildings").is_none());
        assert!(plain["presentation"].get("buildings").is_none());
        assert_eq!(
            plain["presentation"]["enu_origin"]["latitude_deg"],
            json!(37.7749)
        );

        // The run reads them as the model's roofed solids, on the ground.
        let parsed = ScenarioBundle::from_json(&town.to_string()).expect("parses");
        let world = build_world(&parsed.world).expect("builds");
        let layer = world.obstacles.layer();
        let everything = argusnet_types::Bounds2d::new(-1e6, 1e6, -1e6, 1e6);
        let roofed: Vec<_> = layer
            .query(everything)
            .into_iter()
            .filter(|o| o.roof.is_some())
            .collect();
        assert_eq!(roofed.len(), 10);
        let church = roofed.iter().find(|o| o.id == "way/3").unwrap();
        let roof = church.roof.as_ref().unwrap();
        assert_eq!(roof.shape, RoofShape::Pyramidal);
        assert!((church.top_z_m - church.base_z_m - 20.0).abs() < 1e-9);
        assert!((roof.eave_z_m - church.base_z_m - 12.0).abs() < 1e-9);
        for o in &roofed {
            let argusnet_world::obstacles::ObstacleShape::Polygon { footprint_xy_m } = &o.shape
            else {
                panic!("a footprint");
            };
            let lowest = footprint_xy_m
                .iter()
                .map(|p| crate::observe::TerrainHeight::height_at(&world.terrain, p[0], p[1]))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (o.base_z_m - lowest).abs() < 1e-9,
                "{} stands on its ground",
                o.id
            );
        }
    }

    #[test]
    fn a_bad_document_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("argusnet-buildings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut document: Value =
            serde_json::from_str(&std::fs::read_to_string(TOWN).unwrap()).unwrap();
        document["buildings"][0]["roof_shape"] = json!("mansard");
        let path = dir.join("bad.json");
        std::fs::write(&path, document.to_string()).unwrap();
        let config: ScenarioConfig = serde_json::from_value(json!({
            "format": "argusnet.scenario-config", "version": 1, "duration_s": 5.0,
            "options": {"buildings_path": path.to_str().unwrap()},
        }))
        .unwrap();
        let error = build_bundle(&config).unwrap_err().to_string();
        assert!(
            error.contains("way/1") && error.contains("mansard"),
            "{error}"
        );
        // A building only the model refuses (a hip over an L, which the
        // importer would have written flat) is refused here, by name, not
        // when the run reads the bundle.
        document["buildings"][0]["roof_shape"] = json!("gabled");
        document["buildings"][1]["footprint_xy_m"] = json!([
            [-60.0, -30.0],
            [-46.0, -30.0],
            [-46.0, -25.0],
            [-53.0, -25.0],
            [-53.0, -20.0],
            [-60.0, -20.0]
        ]);
        std::fs::write(&path, document.to_string()).unwrap();
        let error = build_bundle(&config).unwrap_err().to_string();
        assert!(
            error.contains("way/2") && error.contains("convex"),
            "{error}"
        );
        // A building outside the map is left out and counted, not clipped.
        let mut outside: Value =
            serde_json::from_str(&std::fs::read_to_string(TOWN).unwrap()).unwrap();
        outside["buildings"][4]["footprint_xy_m"] =
            json!([[1.0e5, 0.0], [1.0e5 + 10.0, 0.0], [1.0e5 + 10.0, 10.0]]);
        std::fs::write(&path, outside.to_string()).unwrap();
        let built = build_bundle(&config).unwrap();
        assert_eq!(built["world"]["buildings"]["buildings"], json!(9));
        assert_eq!(
            built["world"]["buildings"]["left_out_outside_map"],
            json!(1)
        );
        document["format"] = json!("something-else");
        std::fs::write(&path, document.to_string()).unwrap();
        let error = build_bundle(&config).unwrap_err().to_string();
        assert!(error.contains("argusnet.buildings"), "{error}");
    }
}
