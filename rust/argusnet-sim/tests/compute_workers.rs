//! B3 gate: a run's output is identical at 1, 2 and N workers. The pair
//! evaluations of observation generation fork across the session's pool;
//! everything that combines them runs serially in pair order, so the replay
//! must not move by a bit whatever the worker count or scheduling.

use std::sync::Arc;

use argusnet_sim::builder::{build_bundle, ScenarioConfig};
use argusnet_sim::compute::{ComputePool, PARALLEL_THRESHOLD};
use argusnet_sim::mapping::MIN_PLATFORMS_TO_FORK;
use argusnet_sim::scenario::{assemble_with, ScenarioBundle};

fn bundle(name: &str) -> ScenarioBundle {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    ScenarioBundle::from_json(&text).expect("bundle parses")
}

fn replay_at(bundle: &ScenarioBundle, workers: usize) -> String {
    let pool = Arc::new(ComputePool::new(workers).expect("pool"));
    let replay = assemble_with(bundle, pool)
        .expect("assembles")
        .run_to_replay()
        .expect("runs");
    serde_json::to_string(&replay).expect("serialises")
}

#[test]
fn the_replay_is_identical_at_every_worker_count() {
    let bundle = bundle("scenario_bundle_tt.json");
    let pairs = bundle.nodes.len() * bundle.targets.len();
    assert!(
        pairs >= PARALLEL_THRESHOLD,
        "{pairs} pairs: the fixture must be large enough to take the parallel path"
    );
    let serial = replay_at(&bundle, 1);
    for workers in [2, 4, 8] {
        assert!(
            replay_at(&bundle, workers) == serial,
            "the replay differs at {workers} workers"
        );
    }
}

/// A scan mission large enough that stage 8 forks each platform's particle
/// draws and scoring: the localizer's estimates, and the whole replay, must
/// not move at any worker count.
#[test]
fn a_forked_localizer_matches_the_serial_one() {
    let path = format!(
        "{}/../../tests/fixtures/scenario_config_scan_16_drones.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    // Twenty seconds keeps the debug-build test short; the localizer runs
    // from the first tick of the scanning phase.
    let mut config: serde_json::Value = serde_json::from_str(&text).expect("config is JSON");
    config["duration_s"] = serde_json::json!(20.0);
    let config = ScenarioConfig::from_json(&config.to_string()).expect("config parses");
    let bundle = build_bundle(&config).expect("bundle builds");
    let bundle = ScenarioBundle::from_json(&bundle.to_string()).expect("bundle parses");

    let run_at = |workers: usize| {
        let pool = Arc::new(ComputePool::new(workers).expect("pool"));
        let mut run = assemble_with(&bundle, pool).expect("assembles");
        let mut estimates = Vec::new();
        let mut most = 0;
        for _ in 0..bundle.steps {
            run.step().expect("tick runs");
            let tick = run.mapping().expect("stage 8 is installed").estimates();
            most = most.max(tick.len());
            estimates.push(serde_json::to_string(&tick).expect("serialises"));
        }
        (estimates, most)
    };

    let (serial, most) = run_at(1);
    assert!(
        most >= MIN_PLATFORMS_TO_FORK,
        "{most} platforms localized at once: too few to take the forked path"
    );
    for workers in [2, 4] {
        let (forked, _) = run_at(workers);
        assert!(
            forked == serial,
            "localizer estimates differ at {workers} workers"
        );
    }
}
