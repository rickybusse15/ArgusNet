//! `field_v2`: `field_v1` plus what it adds, as one named set of options.
//!
//! The gates: the profile is recorded with what it adds; `field_v1` is
//! untouched by it; the composed run is deterministic, resumes exactly and
//! still touches nothing and ends at home.

use argusnet_sim::run_options::{RunOptions, RunProfile};
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run_with(bundle: &ScenarioBundle, options: &RunOptions) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    options.apply(&mut run, bundle).expect("options apply");
    run
}

fn replay(run: AssembledRun) -> (serde_json::Value, Vec<serde_json::Value>) {
    let document = run.run_to_replay().expect("runs");
    (
        serde_json::to_value(&document.meta).unwrap(),
        document
            .frames
            .iter()
            .map(|frame| serde_json::to_value(frame).unwrap())
            .collect(),
    )
}

/// The profile's sensing and flight, with a scan gate the 400-tick fixture
/// reaches: the whole mission, scan to home.
fn whole_mission() -> RunOptions {
    RunOptions {
        scan_coverage_threshold: Some(0.3),
        profile: None,
        ..RunOptions::field_v2()
    }
}

fn phase(frame: &serde_json::Value) -> &str {
    frame["scan_mission_state"]["phase"].as_str().unwrap()
}

#[test]
fn the_profile_is_field_v1_and_what_it_adds() {
    assert_eq!(RunProfile::parse("field_v2"), Ok(RunProfile::FieldV2));
    let v1 = RunProfile::FieldV1.options();
    let v2 = RunProfile::FieldV2.options();
    // Everything but the additions and the name is field_v1's.
    let mut stripped = v2.clone();
    stripped.profile = v1.profile;
    stripped.exploration.as_mut().unwrap().view_planning = None;
    assert_eq!(stripped, v1);

    let bundle = bundle();
    let (meta, _) = replay(run_with(&bundle, &v2));
    assert_eq!(meta["run_profile"]["id"], "field_v2");
    let exploration = &meta["world_knowledge"]["exploration"];
    assert!(
        exploration
            .get("view_planning")
            .is_some_and(|v| !v.is_null()),
        "the profile records its view planning: {exploration}"
    );
    // field_v1 records none: its replays are what they were.
    let (meta, _) = replay(run_with(&bundle, &v1));
    assert!(meta["world_knowledge"]["exploration"]
        .get("view_planning")
        .is_none());
}

#[test]
fn the_composed_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let options = whole_mission();
    let (_, straight) = replay(run_with(&bundle, &options));
    let (_, again) = replay(run_with(&bundle, &options));
    assert_eq!(straight, again);

    let first = |name: &str| {
        straight
            .iter()
            .position(|frame| phase(frame) == name)
            .unwrap_or_else(|| panic!("the mission reaches {name}"))
    };
    for at in [15, 60, first("inspecting") + 3, first("egress") + 3] {
        let mut original = run_with(&bundle, &options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&bundle, &options);
        resumed.restore(&checkpoint).expect("restore accepted");
        let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
            .map(|_| {
                resumed.step().unwrap();
                serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
            })
            .collect();
        assert_eq!(rest, straight[at..], "resumed at tick {at}");
    }
}

#[test]
fn the_whole_mission_touches_nothing_and_completes() {
    let bundle = bundle();
    let (_, frames) = replay(run_with(&bundle, &whole_mission()));
    let last = frames.last().unwrap();
    assert_eq!(phase(last), "complete", "the mission completes");
    let contacts: usize = frames
        .iter()
        .map(|frame| frame["world_contacts"].as_array().map_or(0, Vec::len))
        .sum();
    assert_eq!(contacts, 0, "physics never had to stop a platform");
    let pois = last["scan_mission_state"]["poi_statuses"]
        .as_array()
        .unwrap();
    assert!(
        pois.iter().all(|poi| poi["status"] == "complete"),
        "{pois:?}"
    );
}
