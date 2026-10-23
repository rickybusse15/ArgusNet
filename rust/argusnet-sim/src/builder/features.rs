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
//! Trees and power lines (R5.2b) are the data's too. Each tree stands
//! where it was mapped, sized by its tags or by its species' middle values.
//! Asked-for scattered trees keep clear of it. Each power line's supports
//! stand on the run's ground, and the procedural line is not built. A tree
//! or line outside the map, inside a structure, or that the model refuses
//! is left out and counted, never moved.
//!
//! Every layer of a run must share one origin: a features file whose origin
//! differs from the terrain's or the buildings' fails the build by name.
//! The georeference, the document's hash and the counts go into the bundle
//! and the replay. Without the option nothing changes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use argusnet_world::land_cover_build::{CoverPolygon, LandCoverClass, SeasonalVariation};
use argusnet_world::landscape::BuiltTerrain;
use argusnet_world::obstacles::Obstacle;
use argusnet_world::powerline::{PowerLine, Support};
use argusnet_world::trees::{Species, Tree};

use super::environment::MapBounds;
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
    /// An area's outer ring. Either this or a line segment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polygon_xy_m: Option<Vec<[f64; 2]>>,
    /// A line's segment, start and end, painted [`width_m`] wide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_xy_m: Option<[[f64; 2]; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width_m: Option<f64>,
}

/// A line segment as the land cover paints it: a rectangle `width_m`
/// across, but never narrower than `resolution_m` x sqrt(2), with its ends
/// extended by half that width. Then every cell the segment's centre line
/// crosses has its centre inside (a cell's centre is at most half its
/// diagonal from any point of the cell), so a road narrower than a cell is
/// never lost between cell centres.
pub fn segment_polygon(line: [[f64; 2]; 2], width_m: f64, resolution_m: f64) -> Vec<[f64; 2]> {
    let [[x0, y0], [x1, y1]] = line;
    let half = 0.5 * width_m.max(resolution_m * std::f64::consts::SQRT_2);
    let length = (x1 - x0).hypot(y1 - y0);
    let (dx, dy) = ((x1 - x0) / length * half, (y1 - y0) / length * half);
    let (nx, ny) = (-dy, dx);
    vec![
        [x0 - dx - nx, y0 - dy - ny],
        [x1 + dx - nx, y1 + dy - ny],
        [x1 + dx + nx, y1 + dy + ny],
        [x0 - dx + nx, y0 - dy + ny],
    ]
}

/// One mapped tree, its sizes when tagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedTree {
    pub id: String,
    pub xy_m: [f64; 2],
    pub species: String,
    #[serde(default)]
    pub height_m: Option<f64>,
    #[serde(default)]
    pub crown_radius_m: Option<f64>,
    #[serde(default)]
    pub trunk_radius_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedSupport {
    pub xy_m: [f64; 2],
    /// Its top cross-arm above the ground, when a tower or pole is tagged.
    #[serde(default)]
    pub height_m: Option<f64>,
}

/// One mapped power line: `line` (transmission) or `minor_line`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedPowerLine {
    pub id: String,
    pub kind: String,
    pub supports: Vec<ImportedSupport>,
    #[serde(default)]
    pub cables: Option<u32>,
    #[serde(default)]
    pub voltage_v: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaturesDocument {
    pub format: String,
    pub version: u32,
    pub georeference: Georeference,
    pub land_cover: Vec<LandCoverFeature>,
    #[serde(default)]
    pub trees: Vec<ImportedTree>,
    #[serde(default)]
    pub power_lines: Vec<ImportedPowerLine>,
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
    let bytes = super::read_import_limited(path).map_err(invalid)?;
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
        let finite = |points: &[[f64; 2]]| points.iter().flatten().all(|v| v.is_finite());
        match (&feature.polygon_xy_m, &feature.line_xy_m, feature.width_m) {
            (Some(polygon), None, _) if polygon.len() >= 3 && finite(polygon) => {}
            (Some(_), None, _) => {
                return Err(invalid(format!(
                    "feature {}: a polygon needs at least 3 finite points",
                    feature.id
                )))
            }
            (None, Some(line), Some(width))
                if finite(line) && line[0] != line[1] && width.is_finite() && width > 0.0 => {}
            (None, Some(_), _) => {
                return Err(invalid(format!(
                    "feature {}: a line needs two distinct finite points and a positive width",
                    feature.id
                )))
            }
            _ => {
                return Err(invalid(format!(
                    "feature {}: needs either polygon_xy_m or line_xy_m",
                    feature.id
                )))
            }
        }
    }
    for tree in &document.trees {
        if tree_species(&tree.species).is_none() {
            return Err(invalid(format!(
                "tree {}: unknown species {:?}",
                tree.id, tree.species
            )));
        }
    }
    for line in &document.power_lines {
        if line_kind(&line.kind).is_none() {
            return Err(invalid(format!(
                "power line {}: unknown kind {:?}",
                line.id, line.kind
            )));
        }
    }
    Ok((document, crate::config::fnv1a(&bytes)))
}

pub fn tree_species(name: &str) -> Option<Species> {
    Some(match name {
        "conifer" => Species::Conifer,
        "broadleaf" => Species::Broadleaf,
        "palm" => Species::Palm,
        "shrub" => Species::Shrub,
        _ => return None,
    })
}

/// What a mapped line's kind gives where the tags do not: support height,
/// cross-arm width, sag ratio, conductor radius and clearance (metres but
/// the ratio).
pub struct LineKind {
    pub support_height_m: f64,
    pub arm_m: f64,
    pub sag_ratio: f64,
    pub conductor_radius_m: f64,
    pub clearance_m: f64,
}

pub fn line_kind(name: &str) -> Option<LineKind> {
    match name {
        // As the procedural transmission line.
        "line" => Some(LineKind {
            support_height_m: 30.0,
            arm_m: 8.0,
            sag_ratio: 0.03,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }),
        "minor_line" => Some(LineKind {
            support_height_m: 10.0,
            arm_m: 1.6,
            sag_ratio: 0.02,
            conductor_radius_m: 0.006,
            clearance_m: 4.0,
        }),
        _ => None,
    }
}

/// How far a platform keeps from a line of `voltage_v`, metres: 10 at
/// 110 kV and over, 6 at 33 kV and over, 4 below; the kind's when the
/// voltage is not tagged.
pub fn clearance_m(kind: &LineKind, voltage_v: Option<f64>) -> f64 {
    match voltage_v {
        Some(v) if v >= 110_000.0 => 10.0,
        Some(v) if v >= 33_000.0 => 6.0,
        Some(_) => 4.0,
        None => kind.clearance_m,
    }
}

/// `count` conductors (1 to 12) spread evenly across an arm `arm_m` wide,
/// to the millimetre; one hangs on the line.
pub fn conductor_offsets(count: u32, arm_m: f64) -> Vec<f64> {
    let count = count.clamp(1, 12);
    if count == 1 {
        return vec![0.0];
    }
    (0..count)
        .map(|k| {
            let offset = -0.5 * arm_m + arm_m * f64::from(k) / f64::from(count - 1);
            // To the millimetre, as the bundle carries it.
            (offset * 1000.0).round() / 1000.0
        })
        .collect()
}

/// What the data's trees and lines became, and what was left out.
#[derive(Debug, Default)]
pub struct Placed {
    pub trees: Vec<Tree>,
    pub power_lines: Vec<PowerLine>,
    pub trees_outside_map: usize,
    pub trees_in_structure: usize,
    pub trees_refused: usize,
    /// Lines with no run of two supports inside the map and clear of
    /// structures.
    pub lines_without_a_clear_run: usize,
    pub lines_refused: usize,
    /// Supports inside a building or wall: each splits its line there.
    pub supports_in_structure: usize,
}

fn inside(bounds: MapBounds, xy: [f64; 2]) -> bool {
    xy[0] >= bounds.x_min_m
        && xy[0] <= bounds.x_max_m
        && xy[1] >= bounds.y_min_m
        && xy[1] <= bounds.y_max_m
}

fn in_structure(obstacles: &[Obstacle], xy: [f64; 2]) -> bool {
    obstacles.iter().any(|obstacle| {
        obstacle.blocker_type.is_hard()
            && obstacle.point_inside(xy[0], xy[1], 0.5 * (obstacle.base_z_m + obstacle.top_z_m))
    })
}

/// Stand the data's trees and lines on `terrain` inside `bounds`. A line
/// keeps each run of two or more supports that are inside the map and
/// clear of buildings and walls (`id`, then `id#1`, ...): a support off
/// the map or inside a structure splits it, so no wire runs into either.
pub fn place(
    document: &FeaturesDocument,
    terrain: &BuiltTerrain,
    bounds: MapBounds,
    obstacles: &[Obstacle],
    season: SeasonalVariation,
) -> Placed {
    let mut placed = Placed::default();
    for tree in &document.trees {
        if !inside(bounds, tree.xy_m) {
            placed.trees_outside_map += 1;
            continue;
        }
        if in_structure(obstacles, tree.xy_m) {
            placed.trees_in_structure += 1;
            continue;
        }
        let species = tree_species(&tree.species).expect("checked on load");
        let built = super::trees::imported_tree(
            &tree.id,
            tree.xy_m,
            species,
            (tree.height_m, tree.crown_radius_m, tree.trunk_radius_m),
            terrain,
            season,
        );
        match built.validated() {
            Ok(tree) => placed.trees.push(tree),
            Err(_) => placed.trees_refused += 1,
        }
    }
    for line in &document.power_lines {
        let kind = line_kind(&line.kind).expect("checked on load");
        let mut runs: Vec<Vec<Support>> = vec![Vec::new()];
        for support in &line.supports {
            let on_map = inside(bounds, support.xy_m);
            let blocked = on_map && in_structure(obstacles, support.xy_m);
            placed.supports_in_structure += usize::from(blocked);
            if on_map && !blocked {
                runs.last_mut().expect("one run").push(Support {
                    xy_m: support.xy_m,
                    base_z_m: terrain.height_at(support.xy_m[0], support.xy_m[1]),
                    height_m: support.height_m.unwrap_or(kind.support_height_m),
                });
            } else if !runs.last().expect("one run").is_empty() {
                runs.push(Vec::new());
            }
        }
        let runs: Vec<Vec<Support>> = runs.into_iter().filter(|run| run.len() >= 2).collect();
        if runs.is_empty() {
            placed.lines_without_a_clear_run += 1;
            continue;
        }
        for (k, supports) in runs.into_iter().enumerate() {
            let built = PowerLine {
                id: if k == 0 {
                    line.id.clone()
                } else {
                    format!("{}#{k}", line.id)
                },
                supports,
                conductor_offsets_m: conductor_offsets(line.cables.unwrap_or(3), kind.arm_m),
                sag_ratio: kind.sag_ratio,
                conductor_radius_m: kind.conductor_radius_m,
                clearance_m: clearance_m(&kind, line.voltage_v),
            };
            match built.validated() {
                Ok(line) => placed.power_lines.push(line),
                Err(_) => placed.lines_refused += 1,
            }
        }
    }
    placed
}

/// The document's polygons as the land cover paints them in `season` on
/// cells `resolution_m` across, in the document's order.
pub fn painted(
    document: &FeaturesDocument,
    season: SeasonalVariation,
    resolution_m: f64,
) -> Vec<CoverPolygon> {
    document
        .land_cover
        .iter()
        .map(|feature| {
            let class = cover_class(&feature.class).expect("checked on load");
            let polygon_xy_m = match (&feature.polygon_xy_m, feature.line_xy_m) {
                (Some(polygon), _) => polygon.clone(),
                (None, Some(line)) => segment_polygon(
                    line,
                    feature.width_m.expect("checked on load"),
                    resolution_m,
                ),
                (None, None) => unreachable!("checked on load"),
            };
            CoverPolygon {
                polygon_xy_m,
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

/// The bundle's and the replay's record: the source, the polygons by
/// class, and what became of the data's trees and lines.
pub fn record(document: &FeaturesDocument, hash: u64, placed: &Placed) -> Value {
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
        "trees": placed.trees.len(),
        "trees_left_out": {
            "outside_map": placed.trees_outside_map,
            "in_structure": placed.trees_in_structure,
            "refused": placed.trees_refused,
        },
        "power_lines": placed.power_lines.len(),
        "power_lines_left_out": {
            "without_a_clear_run": placed.lines_without_a_clear_run,
            "refused": placed.lines_refused,
        },
        "power_line_supports_in_structure": placed.supports_in_structure,
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
                 "line_xy_m": [[5.0, -150.0], [315.0, -150.0]]},
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

    /// A line narrower than a cell still marks every cell it crosses: a
    /// 3 m track along y = 0 runs between the cell centres at y = -5 and
    /// 5, and paints both rows. A line wider than that keeps its own width.
    #[test]
    fn narrow_lines_mark_every_cell_they_cross() {
        let mut document = document();
        document["land_cover"] = json!([
            {"id": "track", "class": "road", "width_m": 3.0,
             "line_xy_m": [[-200.0, 0.0], [200.0, 0.0]]},
            {"id": "river", "class": "water", "width_m": 30.0,
             "line_xy_m": [[-200.0, 200.0], [200.0, 200.0]]},
        ]);
        let path = write("narrow", &document);
        let world = build_world(&options(json!({"features_path": path}))).unwrap();
        assert_eq!(world.land_cover_resolution_m, 10.0);
        assert_eq!(class_at(&world, 100.0, -2.0).0, LandCoverClass::Road);
        assert_eq!(class_at(&world, 100.0, 2.0).0, LandCoverClass::Road);
        assert_eq!(class_at(&world, 100.0, -12.0).0, LandCoverClass::Open);
        assert_eq!(class_at(&world, 100.0, 212.0).0, LandCoverClass::Water);
        assert_eq!(class_at(&world, 100.0, 188.0).0, LandCoverClass::Water);
        assert_eq!(class_at(&world, 100.0, 222.0).0, LandCoverClass::Open);
        let wide = segment_polygon([[0.0, 0.0], [10.0, 0.0]], 30.0, 10.0);
        assert_eq!(
            wide,
            [[-15.0, -15.0], [25.0, -15.0], [25.0, 15.0], [-15.0, 15.0]]
        );
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

    /// Mapped trees and lines (R5.2b) around the forest of [`document`].
    fn with_trees_and_lines(building_xy: [f64; 2]) -> Value {
        let mut document = document();
        document["trees"] = json!([
            {"id": "node/1", "xy_m": [100.0, -100.0], "species": "broadleaf",
             "height_m": 15.0, "crown_radius_m": 3.0, "trunk_radius_m": 0.3},
            {"id": "node/2", "xy_m": [200.0, -200.0], "species": "conifer"},
            {"id": "node/3", "xy_m": [5000.0, 0.0], "species": "broadleaf"},
            {"id": "node/4", "xy_m": building_xy, "species": "broadleaf"},
            {"id": "node/5", "xy_m": [-100.0, -250.0], "species": "palm", "height_m": 500.0},
        ]);
        document["power_lines"] = json!([
            {"id": "way/30", "kind": "line", "cables": 6, "voltage_v": 66000.0, "supports": [
                {"xy_m": [-300.0, 300.0]},
                {"xy_m": [-100.0, 300.0], "height_m": 40.0},
                {"xy_m": [100.0, 300.0]},
            ]},
            {"id": "way/31", "kind": "minor_line", "supports": [
                {"xy_m": [-300.0, -300.0]}, {"xy_m": [-200.0, -300.0]},
                {"xy_m": [-200.0, -900.0]},
                {"xy_m": [0.0, -300.0]}, {"xy_m": [100.0, -300.0]},
            ]},
            {"id": "way/32", "kind": "minor_line", "supports": [
                {"xy_m": [900.0, 0.0]}, {"xy_m": [950.0, 0.0]},
            ]},
            // Through a building: split there, its two halves too short.
            {"id": "way/33", "kind": "minor_line", "supports": [
                {"xy_m": [building_xy[0] - 40.0, building_xy[1]]},
                {"xy_m": building_xy},
                {"xy_m": [building_xy[0] + 40.0, building_xy[1]]},
            ]},
        ]);
        document
    }

    fn a_building_centre(world: &BuiltWorld) -> [f64; 2] {
        let building = world
            .obstacles
            .iter()
            .find(|o| o.blocker_type == "building")
            .unwrap();
        let footprint = building.metadata["footprint_xy_m"].as_array().unwrap();
        let n = footprint.len() as f64;
        let (sx, sy) = footprint.iter().fold((0.0, 0.0), |(sx, sy), p| {
            (sx + p[0].as_f64().unwrap(), sy + p[1].as_f64().unwrap())
        });
        [sx / n, sy / n]
    }

    /// Each mapped tree stands where it was mapped, on the ground, sized by
    /// its tags or its species' middle values; those outside the map, in a
    /// building or refused by the model are counted, never moved.
    #[test]
    fn mapped_trees_stand_where_they_were_mapped() {
        let centre = a_building_centre(&build_world(&options(json!({}))).unwrap());
        let path = write("mapped-trees", &with_trees_and_lines(centre));
        let world = build_world(&options(json!({"features_path": path}))).unwrap();
        let ids: Vec<&str> = world.trees.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["node/1", "node/2"]);
        let tagged = &world.trees[0];
        assert_eq!(tagged.xy_m, [100.0, -100.0]);
        assert_eq!(
            (
                tagged.height_m,
                tagged.crown_radius_m,
                tagged.trunk_radius_m
            ),
            (15.0, 3.0, 0.3)
        );
        assert!((tagged.base_z_m - world.terrain.height_at(100.0, -100.0)).abs() < 1e-3);
        // The conifer's middle: 21 m tall, crown 0.185 of it, trunk 0.012
        // of it, evergreen in the middle of its density.
        let conifer = &world.trees[1];
        assert_eq!(conifer.height_m, 21.0);
        assert_eq!(conifer.crown_radius_m, 3.89);
        assert_eq!(conifer.trunk_radius_m, 0.25);
        assert_eq!(conifer.crown_base_m, 5.25);
        assert_eq!(conifer.leaf_density, 0.83);
        // A mapped broadleaf thins in winter, by the season's foliage.
        assert_eq!(tagged.leaf_density, 0.88);
        let winter =
            build_world(&options(json!({"features_path": path, "season_month": 1}))).unwrap();
        assert!(winter.trees[0].leaf_density < 0.88);
        assert_eq!(winter.trees[1].leaf_density, 0.83);
        let record = features_record(&world);
        assert_eq!(record["trees"], json!(2));
        assert_eq!(
            record["trees_left_out"],
            json!({"outside_map": 1, "in_structure": 1, "refused": 1})
        );
    }

    fn features_record(world: &BuiltWorld) -> Value {
        world.features.clone().expect("a features record")
    }

    /// Asked-for scattered trees keep the spacing from a mapped one.
    #[test]
    fn the_scatter_keeps_clear_of_mapped_trees() {
        let path = write("scatter", &with_trees_and_lines([0.0, 0.0]));
        let world = build_world(&options(json!({"features_path": path, "trees": true}))).unwrap();
        assert!(world.trees.len() > 10);
        let mapped = &world.trees[0];
        assert_eq!(mapped.id, "node/1");
        for tree in world.trees.iter().filter(|t| t.id.starts_with("tree-")) {
            let apart = (tree.xy_m[0] - mapped.xy_m[0]).hypot(tree.xy_m[1] - mapped.xy_m[1]);
            assert!(
                apart >= 8.0,
                "{} is {apart} m from the mapped tree",
                tree.id
            );
        }
    }

    /// Mapped lines stand on the ground with their tagged heights and the
    /// kind's defaults; a line that leaves the map keeps its runs inside;
    /// the procedural line is not built even when asked for.
    #[test]
    fn mapped_lines_replace_the_procedural_one() {
        let centre = a_building_centre(&build_world(&options(json!({}))).unwrap());
        let path = write("lines", &with_trees_and_lines(centre));
        let world = build_world(&options(
            json!({"features_path": path, "power_lines": true}),
        ))
        .unwrap();
        let ids: Vec<&str> = world.power_lines.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, ["way/30", "way/31", "way/31#1"]);
        let main = &world.power_lines[0];
        let heights: Vec<f64> = main.supports.iter().map(|s| s.height_m).collect();
        assert_eq!(heights, [30.0, 40.0, 30.0]);
        for support in &main.supports {
            let ground = world.terrain.height_at(support.xy_m[0], support.xy_m[1]);
            assert!((support.base_z_m - ground).abs() < 1e-9);
        }
        assert_eq!(main.conductor_offsets_m, [-4.0, -2.4, -0.8, 0.8, 2.4, 4.0]);
        // 66 kV: 6 m, not the transmission default of 10.
        assert_eq!(main.clearance_m, 6.0);
        let minor = &world.power_lines[1];
        assert_eq!(minor.supports.len(), 2);
        assert_eq!(minor.conductor_offsets_m, [-0.8, 0.0, 0.8]);
        assert_eq!(minor.clearance_m, 4.0);
        assert_eq!(world.power_lines[2].supports[0].xy_m, [0.0, -300.0]);
        let record = features_record(&world);
        assert_eq!(record["power_lines"], json!(3));
        assert_eq!(
            record["power_lines_left_out"],
            json!({"without_a_clear_run": 2, "refused": 0})
        );
        assert_eq!(record["power_line_supports_in_structure"], json!(1));
        // Without features the same option builds the procedural line.
        let plain = build_world(&options(json!({"power_lines": true}))).unwrap();
        assert_eq!(plain.power_lines[0].id, "powerline-transmission");
    }

    #[test]
    fn conductors_spread_across_the_arm_and_clearance_follows_voltage() {
        assert_eq!(conductor_offsets(1, 8.0), [0.0]);
        assert_eq!(conductor_offsets(3, 8.0), [-4.0, 0.0, 4.0]);
        assert_eq!(conductor_offsets(0, 8.0), [0.0]);
        assert_eq!(conductor_offsets(40, 8.0).len(), 12);
        let line = line_kind("line").unwrap();
        let minor = line_kind("minor_line").unwrap();
        assert_eq!(clearance_m(&minor, Some(110_000.0)), 10.0);
        assert_eq!(clearance_m(&line, Some(33_000.0)), 6.0);
        assert_eq!(clearance_m(&line, Some(32_999.0)), 4.0);
        assert_eq!(clearance_m(&line, None), 10.0);
        assert_eq!(clearance_m(&minor, None), 4.0);
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
        let mut species = document();
        species["trees"] = json!([{"id": "n", "xy_m": [0.0, 0.0], "species": "fern"}]);
        let message = error(options(
            json!({"features_path": write("species", &species)}),
        ));
        assert!(message.contains("unknown species"), "{message}");
        let mut kind = document();
        kind["power_lines"] = json!([{"id": "w", "kind": "cable", "supports": []}]);
        let message = error(options(json!({"features_path": write("kind", &kind)})));
        assert!(message.contains("unknown kind"), "{message}");
        let mut line = document();
        line["land_cover"][3]["width_m"] = json!(0.0);
        let message = error(options(json!({"features_path": write("zero", &line)})));
        assert!(message.contains("positive width"), "{message}");
        let mut neither = document();
        neither["land_cover"][3] = json!({"id": "x", "class": "road"});
        let message = error(options(
            json!({"features_path": write("neither", &neither)}),
        ));
        assert!(
            message.contains("either polygon_xy_m or line_xy_m"),
            "{message}"
        );
        let mut other = document();
        other["format"] = json!("argusnet.terrain");
        let message = error(options(json!({"features_path": write("other", &other)})));
        assert!(message.contains(FEATURES_FORMAT), "{message}");
    }
}
