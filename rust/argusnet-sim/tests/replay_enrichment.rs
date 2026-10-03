//! The Rust replay's optional payloads, against the reference's own (P4.9a.2).
//!
//! Fixtures from `scripts/capture_scenario_bundle_fixture.py`, whose frames
//! carry, under `replay`, `run_simulation`'s serialisation of generation
//! rejections, mapping and localization state, inspection and deconfliction
//! events, the scan mission state and target metadata.
//!
//! What is exact follows from what is exact in the runs themselves (see
//! `scenario_bundle.rs`): in the all-follow case every platform matches on
//! every tick, and with the reference's localization replayed so does the
//! whole default mission. So every payload computed from positions, the
//! coverage grids, the mission machine and the POI manager must match bit
//! for bit. Payloads fed by stage 6's draws -- which observations were
//! accepted, what fusion made of them, what the localizer concluded -- differ
//! by design (ADR-006) and are checked for shape and for the parts no draw
//! reaches.

use std::collections::BTreeMap;

use argusnet_sim::mission_driver::RecordedLocalization;
use argusnet_sim::replay::ReplayDocument;
use argusnet_sim::scenario::{
    assemble, numpy_mean, python_round1, python_title, target_priority, ScenarioBundle,
};
use serde_json::Value;

fn read(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_scenario_bundle_fixture.py")
    })
}

fn bundle_of(case: &str) -> ScenarioBundle {
    ScenarioBundle::from_json(&read(&format!("scenario_bundle_{case}.json")))
        .expect("bundle parses")
}

fn reference_of(case: &str) -> Value {
    serde_json::from_str(&read(&format!("scenario_bundle_{case}_reference.json")))
        .expect("reference parses")
}

fn num(value: &Value) -> f64 {
    value
        .as_str()
        .expect("stringified float")
        .parse()
        .expect("parses")
}

/// Run a case to its replay, with the reference's localization replayed in
/// when the case has a mission.
fn run(case: &str) -> (ScenarioBundle, Value, ReplayDocument) {
    let bundle = bundle_of(case);
    let reference = reference_of(case);
    let mut run = assemble(&bundle).expect("bundle assembles");
    if run.driver().is_some() {
        run.load_localization_replay(
            reference["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| RecordedLocalization {
                    confidences: f["mission"]["confidences"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(num)
                        .collect(),
                    timed_out: f["mission"]["localization_timed_out"].as_bool().unwrap(),
                })
                .collect(),
        );
    }
    let document = run.run_to_replay().expect("runs to completion");
    (bundle, reference, document)
}

/// The Rust frame's payload `key`, as JSON, or null when the writer omitted
/// it -- and an omitted list reads as an empty one, as the viewer reads it.
fn payload(frame: &argusnet_sim::replay::ReplayFrame, key: &str) -> Value {
    let value = serde_json::to_value(frame).expect("frame serialises");
    match value.get(key) {
        Some(found) => found.clone(),
        None if [
            "generation_rejections",
            "inspection_events",
            "deconfliction_events",
            "target_metadata",
        ]
        .contains(&key) =>
        {
            Value::Array(Vec::new())
        }
        None => Value::Null,
    }
}

fn pick(value: &Value, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .map(|key| ((*key).to_string(), value[*key].clone()))
            .collect(),
    )
}

const MAPPING_KEYS: [&str; 4] = [
    "coverage_fraction",
    "covered_cells",
    "total_cells",
    "mean_revisits",
];

/// Target metadata no draw reaches. `status` and `confidence` follow the
/// tick's accepted observations, which are draw-dependent.
const TARGET_KEYS: [&str; 8] = [
    "target_id",
    "operator_label",
    "target_type",
    "priority",
    "behavior",
    "first_seen_s",
    "last_seen_s",
    "notes",
];

/// The gate a generation rejection's detail text names.
fn gate_of(detail: &str) -> &str {
    detail.split_whitespace().next().unwrap_or("")
}

/// The all-follow case matches on every tick, so every payload computed from
/// positions and grids does too; generation rejections match wherever both
/// runs rejected the same pair at the same gate.
#[test]
fn a_truth_fed_run_reports_every_deterministic_payload_exactly() {
    let (_, reference, document) = run("tt_follow");
    let frames = reference["frames"].as_array().unwrap();
    assert_eq!(document.frames.len(), frames.len());

    let mut inspection_events = 0usize;
    let mut deconfliction_events = 0usize;
    let mut compared_rejections: BTreeMap<String, usize> = BTreeMap::new();
    let mut reference_gates: std::collections::BTreeSet<String> = Default::default();
    for (tick, (ours, theirs)) in document.frames.iter().zip(frames).enumerate() {
        let theirs = &theirs["replay"];

        assert_eq!(
            pick(&payload(ours, "mapping_state"), &MAPPING_KEYS),
            pick(&theirs["mapping_state"], &MAPPING_KEYS),
            "mapping state at tick {tick}"
        );
        assert_eq!(
            payload(ours, "inspection_events"),
            theirs["inspection_events"],
            "inspection events at tick {tick}"
        );
        assert_eq!(
            payload(ours, "deconfliction_events"),
            theirs["deconfliction_events"],
            "deconfliction events at tick {tick}"
        );
        inspection_events += theirs["inspection_events"].as_array().unwrap().len();
        deconfliction_events += theirs["deconfliction_events"].as_array().unwrap().len();

        let targets = |value: Value| -> Vec<Value> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|t| pick(t, &TARGET_KEYS))
                .collect()
        };
        assert_eq!(
            targets(payload(ours, "target_metadata")),
            targets(theirs["target_metadata"].clone()),
            "target metadata at tick {tick}"
        );

        // Generation rejections: the draw-free gates reject the same pairs;
        // every pair both runs rejected at the same gate is described
        // identically, points and text alike.
        // The Rust writer omits an absent point where Python writes null.
        let by_pair = |value: Value| -> BTreeMap<(String, String), Value> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    let mut r = r.clone();
                    for key in [
                        "origin",
                        "attempted_point",
                        "closest_point",
                        "first_hit_range_m",
                    ] {
                        r.as_object_mut().unwrap().entry(key).or_insert(Value::Null);
                    }
                    (
                        (
                            r["node_id"].as_str().unwrap().to_string(),
                            r["target_id"].as_str().unwrap().to_string(),
                        ),
                        r,
                    )
                })
                .collect()
        };
        let ours_by_pair = by_pair(payload(ours, "generation_rejections"));
        let theirs_by_pair = by_pair(theirs["generation_rejections"].clone());
        let draw_free = |map: &BTreeMap<(String, String), Value>| -> Vec<(String, String)> {
            map.iter()
                .filter(|(_, r)| {
                    matches!(r["reason"].as_str(), Some("out_of_range" | "outside_fov"))
                })
                .map(|(pair, _)| pair.clone())
                .collect()
        };
        assert_eq!(
            draw_free(&ours_by_pair),
            draw_free(&theirs_by_pair),
            "draw-free rejections at tick {tick}"
        );
        for (pair, theirs) in &theirs_by_pair {
            reference_gates.insert(gate_of(theirs["detail"].as_str().unwrap()).to_string());
            let Some(ours) = ours_by_pair.get(pair) else {
                continue;
            };
            let gate = gate_of(theirs["detail"].as_str().unwrap());
            if ours["reason"] != theirs["reason"]
                || gate_of(ours["detail"].as_str().unwrap()) != gate
            {
                continue;
            }
            assert_eq!(ours, theirs, "{pair:?} rejected at tick {tick}");
            *compared_rejections.entry(gate.to_string()).or_default() += 1;
        }

        // Fed by fusion: present exactly when the frame has tracks.
        let localization = payload(ours, "localization_state");
        assert_eq!(
            localization.is_null(),
            ours.tracks.is_empty(),
            "localization state presence at tick {tick}"
        );
        if !localization.is_null() {
            let confidence = localization["mean_observation_confidence"]
                .as_f64()
                .unwrap();
            assert!(
                (0.0..=1.0).contains(&confidence),
                "tick {tick}: {confidence}"
            );
            assert_eq!(localization["frame_id"], "map");
        }
        assert!(payload(ours, "scan_mission_state").is_null());
    }

    assert!(inspection_events > 0, "no zone was ever entered");
    assert!(deconfliction_events > 0, "no deconfliction to compare");
    // Every gate the reference rejected at was compared at least once, and
    // the ones that carry sight-line points (LOS, detection) among them.
    for gate in &reference_gates {
        assert!(
            compared_rejections.get(gate).copied().unwrap_or(0) > 0,
            "no {gate} rejection compared: {compared_rejections:?}"
        );
    }
    for gate in ["LOS", "Probabilistic"] {
        assert!(
            reference_gates.contains(gate),
            "the scenario never reached {gate}"
        );
    }
}

/// With localization replayed, the default mission matches through
/// `complete`, so its whole state does: phase, coverage, POIs, coordinator,
/// the newly scanned cells, egress progress. Only the localizer's own
/// estimates differ, and those are compared for who and when.
#[test]
fn a_replayed_scan_mission_reports_its_state_exactly() {
    let (_, reference, document) = run("scan");
    let frames = reference["frames"].as_array().unwrap();
    assert_eq!(document.frames.len(), frames.len());

    let mut phases_with_egress = 0usize;
    let mut newly_scanned = 0usize;
    let mut estimate_ticks = 0usize;
    let mut inspection_events = 0usize;
    for (tick, (ours, theirs)) in document.frames.iter().zip(frames).enumerate() {
        let theirs = &theirs["replay"];
        assert_eq!(
            pick(&payload(ours, "mapping_state"), &MAPPING_KEYS),
            pick(&theirs["mapping_state"], &MAPPING_KEYS),
            "mapping state at tick {tick}"
        );
        assert_eq!(
            payload(ours, "inspection_events"),
            theirs["inspection_events"],
            "inspection events at tick {tick}"
        );
        inspection_events += theirs["inspection_events"].as_array().unwrap().len();

        let mut ours_state = payload(ours, "scan_mission_state");
        let mut theirs_state = theirs["scan_mission_state"].clone();
        assert!(!ours_state.is_null(), "tick {tick} has no mission state");

        // Who localized, and when; not what they concluded.
        let who = |state: &mut Value| -> Vec<Value> {
            state
                .as_object_mut()
                .unwrap()
                .remove("localization_estimates")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|e| pick(e, &["drone_id", "timestamp_s"]))
                .collect()
        };
        let ours_who = who(&mut ours_state);
        assert_eq!(ours_who, who(&mut theirs_state), "estimates at tick {tick}");
        if !ours_who.is_empty() {
            estimate_ticks += 1;
        }
        if !theirs_state["egress_progress"]
            .as_array()
            .unwrap()
            .is_empty()
        {
            phases_with_egress += 1;
        }
        newly_scanned += theirs_state["newly_scanned_cells"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(ours_state, theirs_state, "mission state at tick {tick}");
    }
    assert!(
        phases_with_egress > 0,
        "the egress readout was never compared"
    );
    assert!(newly_scanned > 0, "no newly scanned cell was compared");
    assert!(estimate_ticks > 0, "no localization estimate was compared");
    assert!(inspection_events > 0, "no zone was ever entered");
}

/// `meta`: the scenario half is the bundle's presentation, verbatim; the run
/// half is computed, and matches wherever no draw reaches.
#[test]
fn the_replay_meta_carries_the_scenario_and_the_run() {
    for case in ["tt_follow", "scan"] {
        let (bundle, reference, document) = run(case);
        let meta = serde_json::to_value(&document.meta).unwrap();
        let presentation = bundle
            .presentation
            .as_ref()
            .expect("bundle has presentation");
        assert!(presentation.contains_key("terrain"), "{case}: no terrain");
        for (key, value) in presentation {
            assert_eq!(&meta[key], value, "{case}: meta.{key}");
        }
        let run_keys = &reference["meta_run"];
        for key in ["platform", "requested_duration_s", "actual_duration_s"] {
            assert_eq!(meta[key], run_keys[key], "{case}: meta.{key}");
        }
        // Every (node, target, tick) is attempted whatever the draws.
        assert_eq!(
            meta["observation_generation"]["attempted_count"],
            run_keys["observation_generation"]["attempted_count"],
            "{case}: attempted"
        );
        assert!(
            meta.get("performance").is_none(),
            "timing makes bytes unstable"
        );
    }
    // No targets: every generation total is exactly the reference's.
    let (_, reference, document) = run("scan");
    let meta = serde_json::to_value(&document.meta).unwrap();
    assert_eq!(
        meta["observation_generation"],
        reference["meta_run"]["observation_generation"]
    );
}

#[test]
fn python_title_and_round_match_the_reference_on_their_edges() {
    assert_eq!(python_title("asset lynx"), "Asset Lynx");
    assert_eq!(python_title("target 1a"), "Target 1A");
    assert_eq!(python_title("ALPHA bravo"), "Alpha Bravo");
    assert_eq!(python_round1(0.25), 0.2);
    assert_eq!(python_round1(0.35), 0.3);
    assert_eq!(python_round1(-2.75), -2.8);
}

/// `np.mean`'s pairwise summation, on lengths either side of its 8- and
/// 128-element thresholds. Expected bits recorded from numpy 2.x; at 8, 13,
/// 128 and 129 values a left-to-right sum rounds differently.
#[test]
fn numpy_mean_is_numpys_pairwise_mean() {
    let expected: [(usize, u64); 7] = [
        (5, 4601320018827931699),
        (8, 4603094240495417656),
        (13, 4605289679894147344),
        (128, 4622820238720934300),
        (129, 4622896229481417714),
        (300, 4631504127023021949),
        (1000, 4645441720070992218),
    ];
    for (n, bits) in expected {
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let i = i as f64;
                0.1 * i + 1e-3 * i * i + 1.0 / (i + 3.0)
            })
            .collect();
        assert_eq!(numpy_mean(&values).to_bits(), bits, "n = {n}");
    }
}

#[test]
fn target_priority_is_the_references_ladder() {
    let ladder: Vec<u32> = (0..5).map(target_priority).collect();
    assert_eq!(ladder, vec![3, 2, 2, 1, 1]);
}
