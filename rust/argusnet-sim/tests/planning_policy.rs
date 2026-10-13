//! P1: `efficient_v1` allocates the scan mission's POIs fleet-wide. Opt-in:
//! the reference run is untouched, and the replay records the policy only
//! when it is not the reference.

use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, AssembledRun, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run(case: &str, policy: PlanningPolicy) -> AssembledRun {
    let bundle = bundle(case);
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        planning_policy: policy,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    run
}

/// Every POI assignment decision the run makes, as `(drone, poi, reason)`.
fn assignments(mut run: AssembledRun) -> Vec<(String, String, String)> {
    let mut made = Vec::new();
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let body = run.journal_body();
        for decision in body["decisions"].as_array().into_iter().flatten() {
            if decision["kind"] == "poi_assignment" {
                made.push((
                    decision["subject"].as_str().unwrap().to_string(),
                    decision["selected"].as_str().unwrap().to_string(),
                    decision["reason"].as_str().unwrap().to_string(),
                ));
            }
        }
    }
    made
}

#[test]
fn the_replay_records_a_non_reference_policy_only() {
    let meta = |policy| {
        let replay = run("scan", policy).run_to_replay().expect("runs");
        serde_json::to_value(&replay.meta).expect("meta")
    };
    assert!(meta(PlanningPolicy::Reference)
        .get("planning_policy")
        .is_none());
    assert_eq!(
        meta(PlanningPolicy::EfficientV1)["planning_policy"],
        "efficient_v1"
    );
    assert!(!RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        ..RunOptions::default()
    }
    .is_legacy());
}

#[test]
fn efficient_v1_assigns_the_scan_missions_pois_fleet_wide() {
    let made = assignments(run("scan", PlanningPolicy::EfficientV1));
    assert!(!made.is_empty(), "the scan mission reaches its POIs");
    assert!(made.iter().all(|(_, _, reason)| reason == "fleet_min_cost"));
    let reference = assignments(run("scan", PlanningPolicy::Reference));
    assert!(reference
        .iter()
        .all(|(_, _, reason)| reason == "energy_aware_priority"));
}

#[test]
fn a_reset_efficient_run_is_a_fresh_one() {
    let fresh = serde_json::to_string(
        &run("scan", PlanningPolicy::EfficientV1)
            .run_to_replay()
            .expect("runs"),
    )
    .unwrap();
    let mut reset = run("scan", PlanningPolicy::EfficientV1);
    for _ in 0..60 {
        reset.step().expect("steps");
    }
    reset.reset();
    let again = serde_json::to_string(&reset.run_to_replay().expect("runs")).unwrap();
    assert!(fresh == again, "a reset keeps the policy and starts over");
}
