//! W3: poles measured among the returns, and corridors read from them.
//!
//! The gates: pole detection needs the tracker and is refused otherwise;
//! poles found lie on the true supports, further out than wires are seen; no
//! track runs through a pole; corridors are mapped under fitted names; the
//! run is deterministic and resumes exactly; and without detection nothing
//! is recorded.
//!
//! Truth appears only here, in the tests.

use argusnet_mapping::poles::PoleConfig;
use argusnet_mapping::wires::WireTrackerConfig;
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::rig::RigProfile;
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, build_world, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::wire_sensing::WireSensorModel;

/// The scan fixture with a double conductor line across the map on five
/// poles, its top 260 m above the datum, near the height drones fly.
fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    let support =
        |y: f64| serde_json::json!({ "xy_m": [0.0, y], "base_z_m": 100.0, "height_m": 160.0 });
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "across",
        "supports": [support(-300.0), support(-150.0), support(0.0), support(150.0), support(300.0)],
        "conductor_offsets_m": [-2.0, 2.0],
        "sag_ratio": 0.02,
        "conductor_radius_m": 0.015,
        "clearance_m": 10.0
    }]);
    ScenarioBundle::from_json(&document.to_string()).unwrap()
}

fn config(poles: bool) -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(RigProfile::provisional()),
        measured_wires: Some(WireSensorModel::default()),
        wire_tracker: Some(WireTrackerConfig::default()),
        poles: poles.then(PoleConfig::default),
        ..ExplorationConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        world_knowledge: WorldKnowledge::NoPrior,
        exploration: Some(config),
        ..Default::default()
    }
    .apply(&mut run, bundle)
    .expect("assembles");
    run
}

fn frames(run: AssembledRun) -> Vec<serde_json::Value> {
    run.run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

#[test]
fn detection_needs_the_tracker() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                wire_tracker: None,
                ..config(true)
            },
        )
        .unwrap_err();
    assert!(error.contains("tracker"), "{error}");
}

#[test]
fn poles_lie_on_the_true_supports() {
    let bundle = bundle();
    let supports = build_world(&bundle.world).unwrap().power_lines[0]
        .supports
        .clone();
    let mut run = run_with(&bundle, config(true));
    for _ in 0..bundle.steps {
        run.step().unwrap();
    }
    let grid = run.mapping().unwrap().belief_grid().unwrap();
    let poles = grid.pole_detector().expect("detecting").poles();
    assert!(poles.len() >= 3, "{} poles", poles.len());
    for pole in poles {
        let [x, y] = pole.xy_m();
        let off = supports
            .iter()
            .map(|s| (s.xy_m[0] - x).hypot(s.xy_m[1] - y))
            .fold(f64::INFINITY, f64::min);
        assert!(off < 1.0, "pole {} is {off:.2} m off a support", pole.id);
        assert!(pole.high_z_m <= 261.0, "top {}", pole.high_z_m);
    }
}

/// A true conductor has a corner at each pole. No fitted track that knows
/// its pole reaches past it.
#[test]
fn no_track_runs_through_a_pole() {
    let bundle = bundle();
    let mut run = run_with(&bundle, config(true));
    for _ in 0..bundle.steps {
        run.step().unwrap();
    }
    let grid = run.mapping().unwrap().belief_grid().unwrap();
    let poles = grid.pole_detector().unwrap().poles();
    for track in grid.wire_tracker().unwrap().estimates() {
        if !track.confirmed {
            continue;
        }
        let along = |p: [f64; 3]| p[1];
        let (low, high) = (
            along(track.from_m).min(along(track.to_m)),
            along(track.from_m).max(along(track.to_m)),
        );
        for pole in poles {
            let [x, y] = pole.xy_m();
            let on_line = (x - track.from_m[0]).abs() < 8.0;
            assert!(
                !(on_line && y > low + 10.0 && y < high - 10.0),
                "track {} runs through pole {} ({low:.0} to {high:.0}, pole at {y:.0})",
                track.id,
                pole.id
            );
        }
    }
}

#[test]
fn corridors_are_mapped_and_recorded() {
    let bundle = bundle();
    let frames = frames(run_with(&bundle, config(true)));
    let ids: Vec<&str> = frames
        .iter()
        .flat_map(|frame| frame["mapped_wires"].as_array().into_iter().flatten())
        .map(|span| span["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.iter().any(|id| id.starts_with("wire-corridor-")),
        "{ids:?}"
    );
    let last = &frames.last().unwrap()["mapping_state"]["belief"];
    assert!(last["wire_poles"].as_u64().unwrap() >= 3);
    assert!(last["wire_corridors"].as_u64().unwrap() >= 1);
    assert!(frames.iter().any(|f| f["mapped_poles"]
        .as_array()
        .is_some_and(|poles| !poles.is_empty())));
}

#[test]
fn a_detected_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let straight = frames(run_with(&bundle, config(true)));
    assert_eq!(straight, frames(run_with(&bundle, config(true))));
    let at = 150usize;
    let mut original = run_with(&bundle, config(true));
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&bundle, config(true));
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);
}

/// A short tick puts "far from the aircraft" within a few metres. An explorer
/// rejoining must not swap places with its aircraft tick after tick: at any
/// tick length, no platform moves further in a tick than it can fly.
#[test]
fn explorers_do_not_swap_places_with_their_aircraft() {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    document["dt_s"] = serde_json::json!(0.25);
    document["steps"] = serde_json::json!(400);
    let bundle = ScenarioBundle::from_json(&document.to_string()).unwrap();
    let mut run = run_with(&bundle, config(true));
    let mut last = std::collections::BTreeMap::new();
    let mut reversals = 0;
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let p: [f64; 2] = std::array::from_fn(|k| node["position"][k].as_f64().unwrap());
            let id = node["node_id"].as_str().unwrap().to_string();
            let previous: Option<[[f64; 2]; 2]> = last.get(&id).copied();
            if let Some([a, b]) = previous {
                let (hop, back) = (
                    (p[0] - b[0]).hypot(p[1] - b[1]),
                    (p[0] - a[0]).hypot(p[1] - a[1]),
                );
                if hop > 100.0 && back < hop / 4.0 {
                    reversals += 1;
                }
                last.insert(id, [b, p]);
            } else {
                last.insert(id, [p, p]);
            }
        }
    }
    assert_eq!(reversals, 0);
}

#[test]
fn without_detection_nothing_is_recorded() {
    let bundle = bundle();
    let frames = frames(run_with(&bundle, config(false)));
    assert!(frames.iter().all(|frame| {
        frame.get("mapped_poles").is_none()
            && frame["mapping_state"]["belief"].get("wire_poles").is_none()
    }));
}
