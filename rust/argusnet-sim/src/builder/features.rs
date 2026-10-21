//! Real land cover in a built world (R5): the `features_path` option.
//!
//! The file is an `argusnet.features` v1 document, written by the offline
//! importer (`scripts/import_features.py`) from OSM or other GeoJSON: water,
//! land use, natural cover and roads as polygons in the run's local
//! metres, in the order they are painted. With it, the land cover is the
//! data's (`landscape::build_data_land_cover`), not the procedural
//! classifier's. The ground is open where the data says nothing and rock
//! where it is steep. The procedural environment's invented patches, walls
//! and forest stands are dropped. Its buildings stay unless `buildings_path`
//! replaces them. Trees, when asked for, are scattered over the data's
//! forests.
//!
//! Every layer of a run must share one origin: a features file whose origin
//! differs from the terrain's or the buildings' fails the build by name.
//! The georeference, the document's hash and the counts go into the bundle
//! and the replay. Without the option nothing changes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use argusnet_world::land_cover_build::{CoverPolygon, LandCoverClass, SeasonalVariation};

use super::BuildError;

pub const FEATURES_FORMAT: &str = "argusnet.features";
pub const FEATURES_VERSION: u32 = 1;
/// The most two origins may differ and still be one frame, degrees (about
/// a millimetre).
pub const ORIGIN_TOLERANCE_DEG: f64 = 1e-8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Georeference {
    pub source_crs_id: String,
    pub runtime_crs_id: String,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    #[serde(default)]
    pub origin_h_m: f64,
}

/// One painted polygon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandCoverFeature {
    pub id: String,
    pub class: String,
    pub polygon_xy_m: Vec<[f64; 2]>,
    /// The line's width when the polygon is a segment of one, metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaturesDocument {
    pub format: String,
    pub version: u32,
    pub georeference: Georeference,
    pub land_cover: Vec<LandCoverFeature>,
    #[serde(default)]
    pub source_sha256: String,
    #[serde(default)]
    pub summary: Value,
}

/// A class name as the importer writes it.
pub fn cover_class(name: &str) -> Option<LandCoverClass> {
    Some(match name {
        "open" => LandCoverClass::Open,
        "urban" => LandCoverClass::Urban,
        "forest" => LandCoverClass::Forest,
        "water" => LandCoverClass::Water,
        "scrub" => LandCoverClass::Scrub,
        "wetland" => LandCoverClass::Wetland,
        "rocky" => LandCoverClass::Rocky,
        "snow" => LandCoverClass::Snow,
        "road" => LandCoverClass::Road,
        _ => return None,
    })
}

/// A class's density in `season`: the classifier's own values, foliage
/// scaled by the season for forest and scrub, urban as the procedural
/// districts are drawn, and none on open ground, water and roads.
pub fn cover_density(class: LandCoverClass, season: SeasonalVariation) -> u8 {
    let foliage = |full: f64| (full * season.foliage_density_factor).clamp(0.0, 255.0) as u8;
    match class {
        LandCoverClass::Forest => foliage(190.0),
        LandCoverClass::Scrub => foliage(105.0),
        LandCoverClass::Wetland => 120,
        LandCoverClass::Rocky => 60,
        LandCoverClass::Snow => 40,
        LandCoverClass::Urban => 180,
        LandCoverClass::Open | LandCoverClass::Water | LandCoverClass::Road => 0,
    }
}

/// Read and check the document at `path`: its format, its frame, and every
/// feature's class and polygon.
pub fn load(path: &str) -> Result<(FeaturesDocument, u64), BuildError> {
    let invalid = |message: String| BuildError::Invalid(format!("features_path {path}: {message}"));
    let bytes = std::fs::read(path).map_err(|error| invalid(error.to_string()))?;
    let document: FeaturesDocument =
        serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    if document.format != FEATURES_FORMAT || document.version != FEATURES_VERSION {
        return Err(invalid(format!(
            "expected {FEATURES_FORMAT} v{FEATURES_VERSION}, got {} v{}",
            document.format, document.version
        )));
    }
    if document.georeference.runtime_crs_id != "local-enu" {
        return Err(invalid(format!(
            "runtime_crs_id must be local-enu, got {}",
            document.georeference.runtime_crs_id
        )));
    }
    for feature in &document.land_cover {
        if cover_class(&feature.class).is_none() {
            return Err(invalid(format!(
                "feature {}: unknown class {:?}",
                feature.id, feature.class
            )));
        }
        if feature.polygon_xy_m.len() < 3
            || feature
                .polygon_xy_m
                .iter()
                .any(|[x, y]| !x.is_finite() || !y.is_finite())
        {
            return Err(invalid(format!(
                "feature {}: a polygon needs at least 3 finite points",
                feature.id
            )));
        }
    }
    Ok((document, crate::config::fnv1a(&bytes)))
}

/// The document's polygons as the land cover paints them in `season`, in
/// the document's order.
pub fn painted(document: &FeaturesDocument, season: SeasonalVariation) -> Vec<CoverPolygon> {
    document
        .land_cover
        .iter()
        .map(|feature| {
            let class = cover_class(&feature.class).expect("checked on load");
            CoverPolygon {
                polygon_xy_m: feature.polygon_xy_m.clone(),
                cover_class: class,
                cover_density: cover_density(class, season),
            }
        })
        .collect()
}

/// Two layers' origins are one frame, or the build fails naming both.
pub fn check_same_origin(
    (name, lat, lon): (&str, f64, f64),
    (other, other_lat, other_lon): (&str, f64, f64),
) -> Result<(), BuildError> {
    if (lat - other_lat).abs().max((lon - other_lon).abs()) > ORIGIN_TOLERANCE_DEG {
        return Err(BuildError::Invalid(format!(
            "{name} and {other} have different origins ({lat}, {lon} and {other_lat}, \
             {other_lon}): import both with the same --origin"
        )));
    }
    Ok(())
}

/// The bundle's and the replay's record.
pub fn record(document: &FeaturesDocument, hash: u64) -> Value {
    let mut by_class = serde_json::Map::new();
    for feature in &document.land_cover {
        let count = by_class
            .entry(feature.class.clone())
            .or_insert(json!(0))
            .as_u64()
            .unwrap_or(0);
        by_class.insert(feature.class.clone(), json!(count + 1));
    }
    json!({
        "format": FEATURES_FORMAT,
        "version": FEATURES_VERSION,
        "georeference": document.georeference,
        "content_fnv1a": format!("{hash:016x}"),
        "source_sha256": document.source_sha256,
        "land_cover_polygons": by_class,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{build_bundle, build_world, BuiltWorld, ScenarioConfig, ScenarioOptions};

    const TOWN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/buildings_town.buildings.json"
    );

    fn square(cx: f64, cy: f64, half: f64) -> Value {
        json!([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half]
        ])
    }

    /// Around the synthetic town's origin: a forest with a road through
    /// it, a lake, a meadow. Painted in this order.
    fn document() -> Value {
        json!({
            "format": FEATURES_FORMAT,
            "version": FEATURES_VERSION,
            "georeference": {
                "source_crs_id": "EPSG:4326", "runtime_crs_id": "local-enu",
                "origin_lat_deg": 45.999892, "origin_lon_deg": 7.0, "origin_h_m": 0.0,
            },
            "land_cover": [
                {"id": "way/1", "class": "open", "polygon_xy_m": square(-200.0, 150.0, 80.0)},
                {"id": "way/2", "class": "forest", "polygon_xy_m": square(150.0, -150.0, 120.0)},
                {"id": "way/3", "class": "water", "polygon_xy_m": square(-200.0, 150.0, 40.0)},
                {"id": "way/4/0", "class": "road", "width_m": 10.0,
                 "polygon_xy_m": [[0.0, -155.0], [320.0, -155.0], [320.0, -145.0], [0.0, -145.0]]},
            ],
            "source_sha256": "cd",
        })
    }

    fn write(name: &str, document: &Value) -> String {
        let dir = std::env::temp_dir().join("argusnet-features-tests");
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

    fn class_at(world: &BuiltWorld, x: f64, y: f64) -> (LandCoverClass, u8) {
        let res = world.land_cover_resolution_m;
        let col = ((x - world.bounds.x_min_m) / res).floor() as usize;
        let row = ((y - world.bounds.y_min_m) / res).floor() as usize;
        let k = row * world.land_cover.cols + col;
        (
            LandCoverClass::from_u8(world.land_cover.classes[k]).unwrap(),
            world.land_cover.density[k],
        )
    }

    fn error(options: ScenarioOptions) -> String {
        match build_world(&options) {
            Err(BuildError::Invalid(message)) => message,
            Err(other) => panic!("unexpected {other:?}"),
            Ok(_) => panic!("built"),
        }
    }

    /// The land cover is the data's: forest, a road through it that clears
    /// the foliage, a lake in a meadow, and open ground where the data says
    /// nothing. July foliage is full.
    #[test]
    fn the_land_cover_is_painted_from_the_data() {
        let path = write("cover", &document());
        let world = build_world(&options(json!({"features_path": path}))).expect("builds");
        assert_eq!(
            class_at(&world, 100.0, -100.0),
            (LandCoverClass::Forest, 190)
        );
        assert_eq!(class_at(&world, 100.0, -150.0), (LandCoverClass::Road, 0));
        assert_eq!(class_at(&world, -200.0, 150.0), (LandCoverClass::Water, 0));
        assert_eq!(class_at(&world, -260.0, 150.0), (LandCoverClass::Open, 0));
        // Nothing mapped here, and the ground is flat: open, not the
        // classifier's invented cover.
        assert_eq!(class_at(&world, 300.0, 300.0).0, LandCoverClass::Open);
        let classes: std::collections::BTreeSet<u8> =
            world.land_cover.classes.iter().copied().collect();
        let invented = [
            LandCoverClass::Scrub,
            LandCoverClass::Wetland,
            LandCoverClass::Snow,
        ];
        assert!(invented.iter().all(|c| !classes.contains(&(*c as u8))));
        // In winter the forest stands bare by the season's foliage, and the
        // data's cover invents no snow over it.
        let winter = build_world(&options(json!({"features_path": path, "season_month": 1})))
            .expect("builds");
        let (class, density) = class_at(&winter, 100.0, -100.0);
        assert_eq!(class, LandCoverClass::Forest);
        assert!(density < 190, "winter foliage {density}");
    }

    /// The procedural environment's invented walls, forest stands and
    /// patches go; its buildings stay unless real ones replace them, and
    /// they stay urban over a land-use polygon.
    #[test]
    fn invented_cover_goes_and_buildings_stay_on_top() {
        let kinds = |world: &BuiltWorld, kind: &str| {
            world
                .obstacles
                .iter()
                .filter(|o| o.blocker_type == kind)
                .count()
        };
        let mut meadow = document();
        meadow["land_cover"].as_array_mut().unwrap().insert(
            0,
            json!({"id": "all", "class": "open", "polygon_xy_m": square(0.0, 0.0, 400.0)}),
        );
        let path = write("meadow", &meadow);
        // Walls on one preset, forest stands on another.
        for (preset, invented) in [("military_compound", "wall"), ("alpine", "vegetation")] {
            let plain = build_world(&options(json!({"terrain_preset": preset}))).unwrap();
            assert!(kinds(&plain, invented) > 0, "{preset} has no {invented}");
            let world = build_world(&options(
                json!({"terrain_preset": preset, "features_path": path}),
            ))
            .unwrap();
            assert_eq!(kinds(&world, invented), 0, "{preset}: {invented} kept");
            assert_eq!(kinds(&world, "building"), kinds(&plain, "building"));
            // A building's footprint is urban although the meadow covers it.
            let building = world
                .obstacles
                .iter()
                .find(|o| o.blocker_type == "building")
                .unwrap();
            let footprint = building.metadata["footprint_xy_m"].as_array().unwrap();
            let (sx, sy) = footprint.iter().fold((0.0, 0.0), |(sx, sy), p| {
                (sx + p[0].as_f64().unwrap(), sy + p[1].as_f64().unwrap())
            });
            let n = footprint.len() as f64;
            assert_eq!(class_at(&world, sx / n, sy / n).0, LandCoverClass::Urban);
        }
    }

    /// Asked-for trees grow only in the data's forest.
    #[test]
    fn trees_grow_in_the_mapped_forest() {
        let path = write("trees", &document());
        let world = build_world(&options(json!({"features_path": path, "trees": true}))).unwrap();
        assert!(!world.trees.is_empty());
        for tree in &world.trees {
            let [x, y] = [tree.xy_m[0], tree.xy_m[1]];
            assert!(
                (30.0..=270.0).contains(&x) && (-270.0..=-30.0).contains(&y),
                "a tree at {x}, {y} outside the forest"
            );
            assert!((y + 150.0).abs() > 5.0, "a tree on the road at {x}, {y}");
        }
    }

    /// The bundle and the replay say which features and where; every layer
    /// shares one origin or the build fails naming them.
    #[test]
    fn the_record_and_the_shared_origin() {
        let path = write("record", &document());
        let config: ScenarioConfig = serde_json::from_value(json!({
            "format": "argusnet.scenario-config", "version": 1, "duration_s": 5.0,
            "options": {"terrain_preset": "urban_flat", "map_preset": "small", "features_path": path},
        }))
        .unwrap();
        let bundle = build_bundle(&config).unwrap();
        let record = &bundle["world"]["features"];
        assert_eq!(record["land_cover_polygons"]["forest"], json!(1));
        assert_eq!(record["land_cover_polygons"]["road"], json!(1));
        assert_eq!(bundle["presentation"]["features"], *record);
        assert_eq!(
            bundle["presentation"]["enu_origin"]["latitude_deg"],
            json!(45.999892)
        );
        assert_eq!(
            bundle["presentation"]["scenario_options"]["features_path"],
            json!(path)
        );
        assert!(build_world(&options(json!({"map_preset": "regional",
            "features_path": path, "buildings_path": TOWN})))
        .is_ok());
        let mut moved = document();
        moved["georeference"]["origin_lat_deg"] = json!(46.1);
        let message = error(options(json!({"map_preset": "regional",
            "features_path": write("moved", &moved), "buildings_path": TOWN})));
        assert!(
            message.contains("buildings_path and features_path"),
            "{message}"
        );
        let plain = build_world(&options(json!({}))).unwrap();
        assert!(plain.features.is_none());
    }

    #[test]
    fn a_feature_the_runtime_cannot_paint_is_refused() {
        let mut unknown = document();
        unknown["land_cover"][1]["class"] = json!("orchard");
        let message = error(options(
            json!({"features_path": write("unknown", &unknown)}),
        ));
        assert!(message.contains("unknown class"), "{message}");
        let mut thin = document();
        thin["land_cover"][1]["polygon_xy_m"] = json!([[0.0, 0.0], [1.0, 1.0]]);
        let message = error(options(json!({"features_path": write("thin", &thin)})));
        assert!(message.contains("at least 3"), "{message}");
        let mut other = document();
        other["format"] = json!("argusnet.terrain");
        let message = error(options(json!({"features_path": write("other", &other)})));
        assert!(message.contains(FEATURES_FORMAT), "{message}");
    }
}
