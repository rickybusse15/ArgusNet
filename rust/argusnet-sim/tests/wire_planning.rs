//! W4 and W5: flying what is believed about wires, and a prior that may be
//! wrong.
//!
//! Truth appears only here, in the tests.

use argusnet_mapping::poles::PoleConfig;
use argusnet_mapping::wire_prior::{PriorDegradation, PriorSpan, PriorStatus};
use argusnet_mapping::wires::WireTrackerConfig;
use argusnet_sim::belief_world::{
    ExplorationConfig, UncertainPrior, WirePlanningConfig, WorldKnowledge,
};
use argusnet_sim::rig::RigProfile;
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, build_world, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::wire_sensing::WireSensorModel;

fn bundle(x_m: f64, top_m: f64) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    let support = |y: f64| serde_json::json!({ "xy_m": [x_m, y], "base_z_m": 100.0, "height_m": top_m - 100.0 });
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

fn tracked() -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(RigProfile::provisional()),
        measured_wires: Some(WireSensorModel::default()),
        wire_tracker: Some(WireTrackerConfig::default()),
        ..ExplorationConfig::default()
    }
}

fn planned() -> ExplorationConfig {
    ExplorationConfig {
        poles: Some(PoleConfig::default()),
        wire_planning: Some(WirePlanningConfig::default()),
        ..tracked()
    }
}

fn with_prior() -> ExplorationConfig {
    ExplorationConfig {
        uncertain_prior: Some(UncertainPrior {
            config: Default::default(),
            spans: Vec::new(),
            degraded_truth: Some(PriorDegradation::default()),
        }),
        ..planned()
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

fn ticks_near(bundle: &ScenarioBundle, config: ExplorationConfig) -> usize {
    let lines = build_world(&bundle.world).unwrap().power_lines.to_vec();
    let mut run = run_with(bundle, config);
    let mut last = std::collections::BTreeMap::new();
    let mut near = 0;
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|k| node["position"][k].as_f64().unwrap());
            let id = node["node_id"].as_str().unwrap().to_string();
            let from = last.insert(id, p).unwrap_or(p);
            if lines
                .iter()
                .any(|line| line.clearance_along(from, p) < 10.0)
            {
                near += 1;
            }
        }
    }
    near
}

fn frames(run: AssembledRun) -> Vec<serde_json::Value> {
    run.run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

fn resumes_exactly(bundle: &ScenarioBundle, config: ExplorationConfig) {
    let straight = frames(run_with(bundle, config.clone()));
    assert_eq!(straight, frames(run_with(bundle, config.clone())));
    let at = 150usize;
    let mut original = run_with(bundle, config.clone());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(bundle, config);
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);
}

#[test]
fn planning_needs_poles_and_a_prior_needs_the_tracker() {
    let bundle = bundle(0.0, 260.0);
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                poles: None,
                ..planned()
            },
        )
        .unwrap_err();
    assert!(error.contains("pole"), "{error}");
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                wire_tracker: None,
                poles: None,
                wire_planning: None,
                ..with_prior()
            },
        )
        .unwrap_err();
    assert!(error.contains("tracker"), "{error}");
}

/// Over the placements where W2 came nearest: at least a quarter fewer
/// platform-ticks within 10 m in total (measured 2026-11-06: 38 against 59).
/// A single placement can go either way.
#[test]
fn planning_comes_near_a_conductor_less_often() {
    let placements = [(-120.0, 260.0), (0.0, 300.0), (60.0, 300.0), (0.0, 340.0)];
    let total = |config: fn() -> ExplorationConfig| -> usize {
        placements
            .iter()
            .map(|&(x, top)| ticks_near(&bundle(x, top), config()))
            .sum()
    };
    let (w2, w4) = (total(tracked), total(planned));
    assert!(4 * w4 < 3 * w2, "planned {w4} against tracked {w2}");
}

#[test]
fn a_planned_run_is_deterministic_and_resumes_exactly() {
    resumes_exactly(&bundle(0.0, 260.0), planned());
}

#[test]
fn a_run_with_a_prior_is_deterministic_and_resumes_exactly() {
    resumes_exactly(&bundle(0.0, 260.0), with_prior());
}

/// The map's spans are hazards from the first frame, before any sensor sees
/// a wire, and carry their declared error in their clearance.
#[test]
fn a_prior_is_mapped_from_the_start_with_its_error_in_the_clearance() {
    let bundle = bundle(0.0, 260.0);
    let frames = frames(run_with(&bundle, with_prior()));
    let first = frames[0]["mapped_wires"]
        .as_array()
        .expect("mapped at once");
    let prior: Vec<&serde_json::Value> = first
        .iter()
        .filter(|span| span["id"].as_str().unwrap().starts_with("wire-prior-"))
        .collect();
    assert!(!prior.is_empty());
    for span in prior {
        assert_eq!(span["clearance_m"].as_f64().unwrap(), 34.0);
    }
}

/// A span with a wire under it is confirmed where it is flown, and moved
/// onto what was measured.
#[test]
fn flight_confirms_a_true_span_and_corrects_it() {
    let bundle = bundle(0.0, 260.0);
    let mut run = run_with(&bundle, with_prior());
    for _ in 0..bundle.steps {
        run.step().unwrap();
    }
    let grid = run.mapping().unwrap().belief_grid().unwrap();
    let prior = grid.wire_prior_map().unwrap();
    assert!(prior.count(PriorStatus::Confirmed) >= 1);
    for state in prior.spans() {
        if state.status == PriorStatus::Confirmed {
            assert!(
                state.span.from_m[0].abs() < 3.5,
                "{} still {:.1} m off the line",
                state.span.id,
                state.span.from_m[0]
            );
        }
    }
}

/// A span with nothing under it, flown near, is dropped from the map.
#[test]
fn flight_refutes_a_span_with_no_wire() {
    let bundle = bundle(0.0, 260.0);
    let phantom = PriorSpan {
        id: "gone".into(),
        from_m: [-250.0, -60.0, 200.0],
        to_m: [-250.0, 60.0, 200.0],
        sigma_lateral_m: 3.0,
        sigma_vertical_m: 3.0,
    };
    let config = ExplorationConfig {
        uncertain_prior: Some(UncertainPrior {
            config: Default::default(),
            spans: vec![phantom],
            degraded_truth: None,
        }),
        ..planned()
    };
    let frames = frames(run_with(&bundle, config));
    assert!(frames.iter().any(|frame| frame["retired_wires"]
        .as_array()
        .is_some_and(|ids| ids.iter().any(|id| id == "wire-prior-gone"))));
    let last = &frames.last().unwrap()["mapping_state"]["belief"];
    assert_eq!(last["wire_prior_refuted"].as_u64(), Some(1));
}

/// A span missing from the map is found as an unknown wire is.
#[test]
fn a_line_the_prior_lacks_is_still_tracked() {
    let bundle = bundle(0.0, 260.0);
    let config = ExplorationConfig {
        uncertain_prior: Some(UncertainPrior {
            config: Default::default(),
            spans: Vec::new(),
            degraded_truth: Some(PriorDegradation {
                drop_share: 1.0,
                phantom_spans: 0,
                ..PriorDegradation::default()
            }),
        }),
        ..planned()
    };
    let mut run = run_with(&bundle, config);
    for _ in 0..bundle.steps {
        run.step().unwrap();
    }
    let grid = run.mapping().unwrap().belief_grid().unwrap();
    assert!(grid.wire_prior_map().unwrap().spans().is_empty());
    assert!(grid.wires().count() >= 2);
}

/// A drone under a wire whose survey height is above the wire used to be
/// lifted back through it by the launch wrapper's floor, however its escape
/// turned. Measured 2026-11-07 on this placement: 7 ticks within 10 m, and 35
/// without the escape's floor.
#[test]
fn a_drone_under_a_wire_gets_away_from_it() {
    let mut seeded = bundle(-120.0, 260.0);
    seeded.seed = 7;
    let near = ticks_near(&seeded, tracked());
    assert!(near < 20, "{near} ticks within 10 m");
}

/// After a mission a drone's explorer had flown on unseen, and the jump to
/// it was stopped at a ridge: the drone sat on its pad recording contacts
/// for the rest of the run. Measured 2026-11-08 on this placement: 2,382
/// contacts before, 22 after.
#[test]
fn a_released_drone_does_not_sit_pinned_on_its_pad() {
    let mut seeded = bundle(120.0, 260.0);
    seeded.seed = 7;
    let frames = frames(run_with(&seeded, tracked()));
    let contacts: usize = frames
        .iter()
        .map(|frame| frame["world_contacts"].as_array().map_or(0, Vec::len))
        .sum();
    assert!(contacts < 200, "{contacts} contacts");
}
