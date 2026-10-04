//! B3 gate: a run's output is identical at 1, 2 and N workers. The pair
//! evaluations of observation generation fork across the session's pool;
//! everything that combines them runs serially in pair order, so the replay
//! must not move by a bit whatever the worker count or scheduling.

use std::sync::Arc;

use argusnet_sim::compute::{ComputePool, PARALLEL_THRESHOLD};
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
