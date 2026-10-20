//! R2: an assembled run hands its trees to the planner. On the opt-in
//! `tree_inspection` scenario (POIs among the crowns), efficient_v1 flies
//! some POI flight over a crown (`route_trees: "over"` in its decision);
//! the reference never reads the trees.

use argusnet_sim::builder::{build_bundle, ScenarioConfig};
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;

fn bundle() -> ScenarioBundle {
    let config: ScenarioConfig = serde_json::from_value(serde_json::json!({
        "format": "argusnet.scenario-config",
        "version": 1,
        "seed": 7,
        "duration_s": 240.0,
        "options": {
            "trees": true,
            "tree_inspection": true,
            "drone_count": 3,
            "poi_count": 8,
            "scan_coverage_threshold": 0.3,
        },
    }))
    .expect("config");
    let document = build_bundle(&config).expect("bundle builds");
    ScenarioBundle::from_json(&document.to_string()).expect("bundle parses")
}

/// How many decisions the whole run journals with `route_trees: "over"`.
fn flights_over_crowns(policy: PlanningPolicy) -> usize {
    let bundle = bundle();
    let mut run = assemble(&bundle).expect("assembles");
    RunOptions {
        planning_policy: policy,
        ..RunOptions::default()
    }
    .apply(&mut run, &bundle)
    .expect("applies");
    let mut over = 0;
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let body = run.journal_body();
        over += body["decisions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["inputs"]["route_trees"] == "over")
            .count();
    }
    over
}

#[test]
fn a_run_flies_over_the_crowns_it_knows_only_under_efficient_v1() {
    assert!(flights_over_crowns(PlanningPolicy::EfficientV1) > 0);
    assert_eq!(flights_over_crowns(PlanningPolicy::Reference), 0);
}
