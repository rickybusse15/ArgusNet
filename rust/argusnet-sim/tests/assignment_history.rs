//! P0: the replay records each drone's task over the run, and every change.
//!
//! The history is read from what each drone flies (a recall from stage 2, or
//! its own controller), so a POI assignment shows up when stage 2 first flies
//! it, and a reset run records exactly the history a fresh run does.

use argusnet_sim::scenario::{assemble, ScenarioBundle};

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn history(case: &str) -> serde_json::Value {
    let replay = assemble(&bundle(case))
        .expect("assembles")
        .run_to_replay()
        .expect("runs");
    let meta = serde_json::to_value(&replay.meta).expect("meta serialises");
    meta["assignment_history"].clone()
}

#[test]
fn every_drone_starts_with_a_task_and_scan_drones_take_pois() {
    let history = history("scan");
    let drones = history.as_object().expect("a history keyed by drone");
    assert!(!drones.is_empty());
    let mut poi_assignments = 0;
    for (drone, events) in drones {
        let events = events.as_array().expect("events");
        assert_eq!(events[0]["kind"], "assign", "{drone}");
        assert_eq!(events[0]["reason"], "initial", "{drone}");
        let times: Vec<f64> = events
            .iter()
            .map(|e| e["timestamp_s"].as_f64().unwrap())
            .collect();
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "{drone}: out of order"
        );
        for event in events.iter().skip(1) {
            if event["kind"] == "reassign" {
                assert_ne!(event["task"], event["previous"], "{drone}: a no-op change");
            }
            if event["reason"] == "poi_assigned" {
                assert!(event["task"].as_str().unwrap().starts_with("poi:"));
                assert_ne!(
                    event["task"], "poi:unknown",
                    "{drone}: a POI without its id"
                );
                poi_assignments += 1;
            }
        }
    }
    assert!(
        poi_assignments > 0,
        "the scan mission assigns POIs: {history}"
    );
}

#[test]
fn follow_drones_start_on_their_target() {
    let history = history("tt_follow");
    let tasks: Vec<&str> = history
        .as_object()
        .expect("a history")
        .values()
        .map(|events| events[0]["task"].as_str().unwrap())
        .collect();
    assert!(
        tasks.iter().any(|task| task.starts_with("follow:")),
        "{tasks:?}"
    );
}

#[test]
fn a_reset_run_records_the_history_a_fresh_run_does() {
    let fresh = history("scan");
    let mut run = assemble(&bundle("scan")).expect("assembles");
    for _ in 0..40 {
        run.step().expect("steps");
    }
    run.reset();
    let replay = run.run_to_replay().expect("runs");
    let meta = serde_json::to_value(&replay.meta).expect("meta");
    assert_eq!(meta["assignment_history"], fresh);
}
