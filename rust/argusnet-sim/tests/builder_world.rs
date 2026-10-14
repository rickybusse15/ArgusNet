//! The native builder's world against the reference's (P4.9b.1).
//!
//! Two sources. `scenario_world.json` (`scripts/capture_scenario_world_fixture.py`)
//! records `build_default_environment` over a preset matrix that reaches every
//! obstacle kind, cluster, corridor and land-cover patch. The committed
//! scenario bundles give whole `world` and `presentation` sections -- terrain
//! tiles, land-cover tiles, the viewer mesh -- for the small alpine map.
//!
//! Exact everywhere a value does not derive from a terrain height: structure,
//! identifiers, kinds, footprints, counts, masks and their coverage, every
//! land-cover class and density. Heights, and what is computed from them
//! (obstacle elevations, min/max height, the viewer mesh), within
//! [`HEIGHT_TOLERANCE_M`]: numpy's `exp` differs from the platform's on ~5% of
//! inputs (P4.3), which no port can close short of reimplementing numpy.

mod common;

use argusnet_sim::builder::environment::{box_footprint, wall_footprint};
use argusnet_sim::builder::{build_world, ScenarioOptions};
use common::parity::parity_of;
use serde_json::Value;

/// P4.3's own ceiling for the `exp` divergence, which it measured at 5.7e-14 m.
const HEIGHT_TOLERANCE_M: f64 = 1.0e-12;

fn read(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// Keys whose numbers derive from terrain heights.
fn height_derived(path: &str) -> bool {
    [
        "base_elevation_m",
        "top_elevation_m",
        "min_height_m",
        "max_height_m",
        "heights_m",
    ]
    .iter()
    .any(|key| path.contains(key))
}

#[derive(Default)]
struct Tally {
    exact: usize,
    within_tolerance: usize,
    worst_m: f64,
}

/// `ours` equals `theirs` exactly, except height-derived numbers, which are
/// within tolerance. Panics with the path of the first disagreement.
fn assert_same(ours: &Value, theirs: &Value, path: &str, tally: &mut Tally) {
    match (ours, theirs) {
        (Value::Object(a), Value::Object(b)) => {
            let keys = |m: &serde_json::Map<String, Value>| {
                m.keys().cloned().collect::<std::collections::BTreeSet<_>>()
            };
            assert_eq!(keys(a), keys(b), "{path}: keys");
            for (key, value) in a {
                assert_same(value, &b[key], &format!("{path}.{key}"), tally);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: length");
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                assert_same(x, y, &format!("{path}[{index}]"), tally);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            assert_eq!(a.is_f64(), b.is_f64(), "{path}: {a} and {b} differ in kind");
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            if x == y {
                tally.exact += 1;
            } else if height_derived(path) && (x - y).abs() <= HEIGHT_TOLERANCE_M {
                tally.within_tolerance += 1;
                tally.worst_m = tally.worst_m.max((x - y).abs());
            } else {
                panic!("{path}: {x} against the reference's {y}");
            }
        }
        _ => assert_eq!(ours, theirs, "{path}"),
    }
}

fn options_of(case: &Value) -> ScenarioOptions {
    let o = &case["options"];
    ScenarioOptions {
        map_preset: o["map_preset"].as_str().unwrap().into(),
        terrain_preset: o["terrain_preset"].as_str().unwrap().into(),
        clean_terrain: o["clean_terrain"].as_bool().unwrap(),
        season_month: o["season_month"].as_u64().unwrap() as u32,
        terrain_seed: o["terrain_seed"].as_i64(),
        detail_strength: o["detail_strength"].as_f64().unwrap(),
        terrain_resolution_m: o["terrain_resolution_m"].as_f64(),
        ..ScenarioOptions::default()
    }
}

/// A land-cover raster in the fixture's encoding: two hex digits a cell, row
/// by row.
fn hex(values: &[u8]) -> String {
    values.iter().map(|v| format!("{v:02x}")).collect()
}

#[test]
fn every_preset_builds_the_references_world() {
    let fixture: Value = serde_json::from_str(&read("scenario_world.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 29, "case count");
    let mut tally = Tally::default();
    let mut kinds = std::collections::BTreeSet::new();
    for case in cases {
        let options = options_of(case);
        let name = format!(
            "{}/{}{}",
            options.map_preset,
            options.terrain_preset,
            if options.clean_terrain { " clean" } else { "" }
        );
        let world = build_world(&options).unwrap_or_else(|e| panic!("{name}: {e}"));
        let document = world.world_document();
        let presentation = world.presentation_world();

        assert_eq!(
            presentation["environment_id"], case["environment_id"],
            "{name}: environment id"
        );
        assert_same(
            &document["obstacles"],
            &case["obstacles"],
            &format!("{name}.obstacles"),
            &mut tally,
        );
        for primitive in case["obstacles"]["primitives"].as_array().unwrap() {
            kinds.insert(primitive["kind"].as_str().unwrap().to_string());
        }

        let land_cover = &case["land_cover"];
        assert_eq!(
            world.land_cover_resolution_m,
            land_cover["resolution"].as_f64().unwrap(),
            "{name}: land-cover resolution"
        );
        assert!(
            hex(&world.land_cover.classes) == land_cover["classes"].as_str().unwrap(),
            "{name}: land-cover classes differ"
        );
        assert!(
            hex(&world.land_cover.density) == land_cover["density"].as_str().unwrap(),
            "{name}: land-cover density differs"
        );
        assert_same(
            &presentation["land_cover"],
            &land_cover["metadata"],
            &format!("{name}.land_cover"),
            &mut tally,
        );
        assert_same(
            &presentation["terrain_summary"],
            &case["terrain_summary"],
            &format!("{name}.terrain_summary"),
            &mut tally,
        );

        let mesh = world.viewer_mesh();
        let reference_mesh = &case["viewer_mesh"];
        assert_eq!(mesh["rows"], reference_mesh["rows"], "{name}: mesh rows");
        assert_eq!(mesh["cols"], reference_mesh["cols"], "{name}: mesh cols");
        let stride = reference_mesh["stride"].as_u64().unwrap() as usize;
        let flat: Vec<Value> = mesh["heights_m"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().clone())
            .step_by(stride)
            .collect();
        assert_same(
            &Value::Array(flat),
            &reference_mesh["sample"],
            &format!("{name}.viewer_mesh.heights_m"),
            &mut tally,
        );
    }
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        vec![
            "box-v1",
            "cylinder-v1",
            "forest-stand-v1",
            "polygon-prism-v1",
            "wall-v1"
        ],
        "every obstacle kind is compared"
    );
    println!(
        "{} numbers exact, {} within {HEIGHT_TOLERANCE_M:e} m (worst {:e} m)",
        tally.exact, tally.within_tolerance, tally.worst_m
    );
    // Every number either matched exactly or was a height within tolerance
    // (enforced value by value above); the divergence must exist, or the
    // tolerance is untested.
    assert!(tally.within_tolerance > 0 && tally.exact > 0);
}

/// Whole `world` and `presentation` sections, tile for tile, against the
/// committed bundles: every key the world decides.
#[test]
fn the_small_alpine_bundles_world_and_presentation_are_rebuilt() {
    for case in ["scan", "tt", "tt_follow"] {
        let bundle: Value =
            serde_json::from_str(&read(&format!("scenario_bundle_{case}.json"))).unwrap();
        let options: ScenarioOptions =
            serde_json::from_value(bundle["presentation"]["scenario_options"].clone())
                .expect("the bundle's scenario options are a config's options");
        let world = build_world(&options).unwrap();
        let mut tally = Tally::default();
        assert_same(&world.world_document(), &bundle["world"], case, &mut tally);
        for (key, value) in world.presentation_world() {
            assert_same(
                &value,
                &bundle["presentation"][&key],
                &format!("{case}.presentation.{key}"),
                &mut tally,
            );
        }
        assert!(
            tally.within_tolerance > 0,
            "{case}: the exp divergence is real"
        );
    }
}

/// Footprints over 300 random boxes and 300 random walls, from the
/// reference's own `OrientedBox` and `WallSegment`: numpy's fused matmul,
/// `np.linalg.norm` and the offset's association, bit for bit, far beyond the
/// handful of hand-placed preset obstacles. Bit for bit on the capture
/// platform; off it the trig and `norm` may round differently.
#[test]
fn box_and_wall_footprints_match_the_references_arithmetic() {
    let fixture: Value = serde_json::from_str(&read("scenario_world.json")).unwrap();
    let parity = parity_of(&fixture);
    let geometry = &fixture["geometry"];
    let points = |value: &Value| -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
            .collect()
    };
    let flat = |footprint: Vec<[f64; 2]>| -> Vec<f64> { footprint.concat() };
    let boxes = geometry["boxes"].as_array().unwrap();
    let walls = geometry["walls"].as_array().unwrap();
    assert_eq!((boxes.len(), walls.len()), (300, 300));
    for (index, case) in boxes.iter().enumerate() {
        let a: Vec<f64> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got = flat(box_footprint(a[0], a[1], a[2], a[3], a[4]));
        let want = points(&case["footprint"]);
        assert!(
            parity.same_all(&got, &want),
            "box {index}: {got:?} against {want:?}"
        );
    }
    for (index, case) in walls.iter().enumerate() {
        let a: Vec<f64> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got = flat(wall_footprint([a[0], a[1]], [a[2], a[3]], a[4]));
        let want = points(&case["footprint"]);
        assert!(
            parity.same_all(&got, &want),
            "wall {index}: {got:?} against {want:?}"
        );
    }
}
