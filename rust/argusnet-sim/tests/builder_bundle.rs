//! Whole bundles from the native builder against `export_bundle` (P4.9b.2b).
//!
//! Two sources: the three committed scenario bundles (compared whole), and
//! `native_bundles.json` (`scripts/capture_native_bundle_fixture.py`), a
//! matrix over every target motion, drone mode, weather and platform preset,
//! station overflow, building-anchored zones, larger maps and scan runs --
//! compared without the world sections, which `builder_world.rs` covers.
//!
//! Every key, string, boolean, integer and array length must match exactly.
//! Floats must match to within [`TOLERANCE_M`]; the ones that differ at all
//! are those downstream of a terrain height (numpy's `exp`, P4.3) or of a
//! `uniform` draw that this platform's numpy fuses into an FMA (P4.9b.2a).

use argusnet_sim::builder::scenario::{build_bundle, Assembly, TABLES};
use argusnet_sim::builder::{build_world, ScenarioConfig};
use serde_json::{json, Value};

const TOLERANCE_M: f64 = 1.0e-12;

fn read(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

#[derive(Default, Debug)]
struct Tally {
    exact: usize,
    close: usize,
    worst: f64,
    worst_path: String,
}

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
            if x.to_bits() == y.to_bits() {
                tally.exact += 1;
            } else {
                let d = (x - y).abs();
                assert!(
                    a.is_f64() && d <= TOLERANCE_M,
                    "{path}: {x} against the reference's {y}"
                );
                tally.close += 1;
                if d > tally.worst {
                    tally.worst = d;
                    tally.worst_path = path.to_string();
                }
            }
        }
        _ => assert_eq!(ours, theirs, "{path}"),
    }
}

fn config(seed: u64, steps: u64, dt_s: f64, options: &Value) -> ScenarioConfig {
    serde_json::from_value(json!({
        "format": "argusnet.scenario-config",
        "version": 1,
        "seed": seed,
        "steps": steps,
        "dt_s": dt_s,
        "options": options,
    }))
    .expect("a config")
}

fn build(config: &ScenarioConfig) -> Value {
    let world = build_world(&config.options).unwrap();
    build_bundle(
        Assembly {
            config,
            tracker: TABLES["tracker_default"].clone(),
        },
        &world,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn the_committed_bundles_are_rebuilt_whole() {
    let cases = [
        (
            "tt",
            11,
            80,
            0.5,
            json!({"map_preset":"small","mission_mode":"target_tracking","target_count":2,"drone_count":3,"ground_station_count":3}),
        ),
        (
            "tt_follow",
            11,
            80,
            0.5,
            json!({"map_preset":"small","mission_mode":"target_tracking","drone_mode_preset":"follow","target_count":1,"drone_count":4,"ground_station_count":3}),
        ),
        (
            "scan",
            5,
            400,
            1.0,
            json!({"map_preset":"small","drone_count":2,"ground_station_count":3,"poi_count":3,"scan_coverage_threshold":0.3}),
        ),
        (
            "tt_safety",
            11,
            80,
            0.5,
            json!({"map_preset":"small","mission_mode":"target_tracking","drone_mode_preset":"follow","target_count":1,"drone_count":4,"ground_station_count":3,"safety_blocking":true}),
        ),
        (
            "scan_safety",
            5,
            400,
            1.0,
            json!({"map_preset":"small","drone_count":2,"ground_station_count":3,"poi_count":3,"scan_coverage_threshold":0.3,"safety_blocking":true}),
        ),
    ];
    for (case, seed, steps, dt, options) in cases {
        let reference: Value =
            serde_json::from_str(&read(&format!("scenario_bundle_{case}.json"))).unwrap();
        let mut tally = Tally::default();
        assert_same(
            &build(&config(seed, steps, dt, &options)),
            &reference,
            case,
            &mut tally,
        );
        println!("{case}: {tally:?}");
    }
}

#[test]
fn a_matrix_of_scenarios_builds_the_references_bundles() {
    let fixture: Value = serde_json::from_str(&read("native_bundles.json")).unwrap();
    let steps = fixture["steps"].as_u64().unwrap();
    let dt = fixture["dt_s"].as_f64().unwrap();
    let world_keys = [
        "terrain",
        "occluding_objects",
        "environment_id",
        "crs_id",
        "environment_bounds_m",
        "terrain_summary",
        "land_cover",
        "land_cover_legend",
        "enu_origin",
    ];
    let mut tally = Tally::default();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 15);
    for case in cases {
        let seed = case["seed"].as_u64().unwrap();
        let mut ours = build(&config(seed, steps, dt, &case["options"]));
        ours.as_object_mut().unwrap().remove("world");
        let presentation = ours["presentation"].as_object_mut().unwrap();
        for key in world_keys {
            presentation.remove(key);
        }
        assert_same(&ours, &case["bundle"], &format!("seed {seed}"), &mut tally);
    }
    println!("matrix: {tally:?}");
}

/// A profile is checked as `ScenarioOptions.__post_init__` checks it: a
/// known id, and the domain, workflow and mission mode it declares. Anything
/// else is an invalid config, not an unsupported one (P4.9d).
#[test]
fn a_mission_profile_is_checked_as_the_reference_checks_it() {
    use argusnet_sim::builder::BuildError;

    let sar = json!({
        "map_preset": "small",
        "mission_mode": "target_tracking",
        "target_count": 1,
        "drone_count": 2,
        "ground_station_count": 3,
        "mission_profile_id": "sar_person_search",
        "mission_domain": "search_and_rescue",
        "mission_workflow": "person_search_and_localization",
        "mission_tags": ["type:search"],
    });
    let attempt = |options: &Value| {
        let config = config(3, 4, 0.5, options);
        let world = build_world(&config.options).unwrap();
        build_bundle(
            Assembly {
                config: &config,
                tracker: TABLES["tracker_default"].clone(),
            },
            &world,
        )
    };
    let built = attempt(&sar).expect("a consistent profile builds");
    assert_eq!(built["scenario_name"], "sar_person_search");
    assert_eq!(
        built["presentation"]["mission_profile"],
        json!({"profile_id": "sar_person_search", "domain": "search_and_rescue",
               "workflow": "person_search_and_localization", "tags": ["type:search"]})
    );
    assert_eq!(built["mission_zones"].as_array().unwrap().len(), 1);
    assert!(
        built.get("safety_blocking").is_none(),
        "the flag is the option's, not the profile's"
    );

    for (key, value) in [
        ("mission_profile_id", json!("no_such_profile")),
        ("mission_domain", json!("industrial_survey")),
        ("mission_domain", Value::Null),
        (
            "mission_workflow",
            json!("fixed_asset_mapping_and_inspection"),
        ),
        ("mission_mode", json!("scan_map_inspect")),
    ] {
        let mut options = sar.clone();
        options[key] = value.clone();
        match attempt(&options) {
            Err(BuildError::Invalid(_)) => {}
            other => panic!("{key} = {value}: expected Invalid, got {other:?}"),
        }
    }
}
