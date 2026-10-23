//! R3: in a run that maps the world for itself, the replay records the
//! power-line spans the platforms' own map gains (`mapped_wires`), so the
//! viewer can draw what the drones know rather than the true lines.

use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::mapping::MappingStage;
use argusnet_sim::scenario::{assemble, AssembledRun, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;
use argusnet_sim::stage::Stage;

/// The scan fixture with one two-span line across the middle, strung at the
/// height the drones fly their lanes, so their scanners pass within range.
fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).unwrap();
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "line",
        "supports": [
            {"xy_m": [-300.0, 0.0], "base_z_m": 230.0, "height_m": 40.0},
            {"xy_m": [0.0, 0.0], "base_z_m": 230.0, "height_m": 40.0},
            {"xy_m": [300.0, 0.0], "base_z_m": 230.0, "height_m": 40.0}
        ],
        "conductor_offsets_m": [0.0],
        "sag_ratio": 0.03,
        "conductor_radius_m": 0.02,
        "clearance_m": 10.0
    }]);
    ScenarioBundle::from_json(&document.to_string()).expect("bundle parses")
}

fn self_built(wire_prior: bool) -> AssembledRun {
    let mut run = assemble(&bundle()).expect("assembles");
    run.set_world_knowledge(
        WorldKnowledge::NoPrior,
        ExplorationConfig {
            wire_prior,
            ..ExplorationConfig::default()
        },
    )
    .unwrap();
    run
}

fn mapped(run: &AssembledRun) -> Vec<String> {
    run.sim
        .stage::<MappingStage>(Stage::UpdateMappingAndLocalization)
        .unwrap()
        .belief_wires()
}

/// Every frame's new span ids, with the map's spans after that tick.
fn step_all(mut run: AssembledRun) -> Vec<(Vec<String>, Vec<String>)> {
    let mut ticks = Vec::new();
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let frame = run.last_frame().expect("a frame");
        let new = frame
            .mapped_wires
            .iter()
            .map(|span| span.id.clone())
            .collect();
        ticks.push((new, mapped(&run)));
    }
    ticks
}

/// Each span appears once, on the tick the map gained it, as the whole span
/// the map holds; applying the frames in order gives the map's spans.
#[test]
fn each_mapped_span_is_recorded_once_when_the_map_gains_it() {
    let ticks = step_all(self_built(false));
    let mut seen: Vec<String> = Vec::new();
    let mut before: Vec<String> = Vec::new();
    for (tick, (new, after)) in ticks.iter().enumerate() {
        let gained: Vec<&String> = after.iter().filter(|id| !before.contains(id)).collect();
        assert_eq!(new.iter().collect::<Vec<_>>(), gained, "tick {tick}");
        seen.extend(new.iter().cloned());
        before = after.clone();
    }
    assert!(!seen.is_empty(), "the lanes pass a span within range");
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        seen.len(),
        "a span is recorded once: {seen:?}"
    );
    assert!(unique.iter().all(|id| id == "line#0" || id == "line#1"));
    let first = ticks.iter().position(|(new, _)| !new.is_empty()).unwrap();
    assert!(first > 0, "mapped by sensing, not known from the start");
}

/// With the wire-hazard prior the first frame carries the prior's spans,
/// and nothing after it does. A reset run records them again, as a fresh
/// run would.
#[test]
fn the_prior_spans_arrive_on_the_first_frame_and_again_after_a_reset() {
    let ticks = step_all(self_built(true));
    assert_eq!(ticks[0].0, ["line#0", "line#1"]);
    assert!(ticks[1..].iter().all(|(new, _)| new.is_empty()));

    let fresh = serde_json::to_string(&self_built(true).run_to_replay().unwrap()).unwrap();
    let mut reset = self_built(true);
    for _ in 0..40 {
        reset.step().unwrap();
    }
    reset.reset();
    let again = serde_json::to_string(&reset.run_to_replay().unwrap()).unwrap();
    assert!(fresh == again, "a reset run is a fresh one");
}

/// A run with a prior map records no mapped wires: it knows the lines.
#[test]
fn a_run_with_a_prior_map_records_none() {
    let run = assemble(&bundle()).expect("assembles");
    let replay = serde_json::to_value(run.run_to_replay().unwrap()).unwrap();
    assert!(replay["frames"]
        .as_array()
        .unwrap()
        .iter()
        .all(|frame| frame.get("mapped_wires").is_none()));
}
