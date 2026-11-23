//! F2: a lost sensor sends its drone home.
//!
//! Expectations, declared before the runs: a navigation camera or a mapping
//! payload out for longer than declared ends that drone's mission and no
//! other's. The drone goes home over what the map believes, touching nothing
//! and jumping nowhere, never faster than the blind speed with no camera; the
//! rest of the fleet finishes. A sensor back sooner changes nothing. A run
//! checkpointed on the way home finishes as the uninterrupted one.

use argusnet_sim::belief_world::FaultReturnConfig;
use argusnet_sim::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};
use argusnet_sim::run_options::RunOptions;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use serde_json::Value;

const DRONE: &str = "drone-east";

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn camera(start_s: f64, end_s: f64) -> Fault {
    Fault::NavCameraOutage {
        node_id: DRONE.into(),
        start_s,
        end_s,
    }
}

fn payload(start_s: f64, end_s: f64) -> Fault {
    Fault::MappingOutage {
        node_id: DRONE.into(),
        start_s,
        end_s,
    }
}

fn options(faults: Vec<Fault>, returns: bool) -> RunOptions {
    let mut options = RunOptions {
        scan_coverage_threshold: Some(0.3),
        profile: None,
        faults: Some(FaultManifest {
            format: MANIFEST_FORMAT.into(),
            version: MANIFEST_VERSION,
            faults,
        }),
        ..RunOptions::field_v1()
    };
    options.exploration.as_mut().unwrap().fault_return = returns.then(FaultReturnConfig::default);
    options
}

fn run_with(options: &RunOptions) -> AssembledRun {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    options.apply(&mut run, &bundle).expect("options apply");
    run
}

fn frames(options: &RunOptions) -> Vec<Value> {
    run_with(options)
        .run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

fn phase(frame: &Value) -> &str {
    frame["scan_mission_state"]["phase"].as_str().unwrap()
}

fn position(frame: &Value, node_id: &str) -> [f64; 3] {
    let node = frame["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["node_id"] == node_id)
        .expect("node in frame");
    std::array::from_fn(|k| node["position"][k].as_f64().unwrap())
}

fn away(frames: &[Value], k: usize) -> f64 {
    let (home, at) = (position(&frames[0], DRONE), position(&frames[k], DRONE));
    (at[0] - home[0]).hypot(at[1] - home[1])
}

fn plan_step(frames: &[Value], k: usize) -> f64 {
    let (a, b) = (position(&frames[k - 1], DRONE), position(&frames[k], DRONE));
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn contacts(frames: &[Value]) -> usize {
    frames
        .iter()
        .map(|frame| frame["world_contacts"].as_array().map_or(0, Vec::len))
        .sum()
}

fn finished(frames: &[Value]) {
    let last = frames.last().unwrap();
    assert_eq!(phase(last), "complete", "the mission completes");
    let pois = last["scan_mission_state"]["poi_statuses"]
        .as_array()
        .unwrap();
    assert!(
        pois.iter().all(|poi| poi["status"] == "complete"),
        "{pois:?}"
    );
    let radius_m = bundle().mission.unwrap().egress_arrival_radius_m;
    assert!(away(frames, frames.len() - 1) <= radius_m, "home");
}

/// What a camera that never comes back did before: the drone holds where it
/// stopped for the rest of the run, and the mission never ends.
#[test]
fn without_a_fault_return_a_lost_camera_holds_the_mission_open() {
    let frames = frames(&options(vec![camera(20.0, 1.0e6)], false));
    assert_ne!(phase(frames.last().unwrap()), "complete");
    assert!(away(&frames, frames.len() - 1) > 100.0, "held away");
}

#[test]
fn a_camera_lost_for_good_brings_its_drone_home_slowly_and_the_rest_finish() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let config = FaultReturnConfig::default();
    let frames = frames(&options(vec![camera(20.0, 1.0e6)], true));
    finished(&frames);
    assert_eq!(contacts(&frames), 0);
    let lost = (20.0 / dt) as usize;
    let cruise = plan_step(&frames, lost);
    // It brakes as it always did, then flies home no faster than declared.
    let mut previous = cruise;
    let blind = config.blind_speed_mps * dt;
    for k in lost + 1..frames.len() {
        let step = plan_step(&frames, k);
        assert!(
            step <= (previous - 3.0 * dt * dt).max(blind) + 1e-9,
            "tick {k}: {step} m after {previous} m"
        );
        previous = step;
    }
    // It is on its way back once it has held for as long as declared.
    let held = lost + (config.camera_hold_s / dt) as usize;
    let moving = (held..held + 10).any(|k| plan_step(&frames, k) > 0.0);
    assert!(moving, "it left the hold");
    let arrived = (held..frames.len())
        .find(|&k| away(&frames, k) <= bundle.mission.as_ref().unwrap().egress_arrival_radius_m)
        .expect("it gets home");
    assert!(
        (arrived + 1..frames.len()).all(|k| away(&frames, k) <= away(&frames, arrived) + 1.0),
        "and stays there"
    );
}

#[test]
fn a_mapping_payload_lost_for_good_brings_its_drone_home() {
    let config = FaultReturnConfig::default();
    let dt = bundle().dt_s;
    let frames = frames(&options(vec![payload(20.0, 1.0e6)], true));
    finished(&frames);
    assert_eq!(contacts(&frames), 0);
    let sent = ((20.0 + config.payload_out_s) / dt) as usize;
    // Its camera works: it flies home at its own speed, and takes no POI.
    for frame in &frames[sent + 2..] {
        let pois = frame["scan_mission_state"]["poi_statuses"]
            .as_array()
            .unwrap();
        assert!(
            pois.iter().all(|poi| poi["assigned_drone_id"] != DRONE),
            "{pois:?}"
        );
    }
}

/// A sensor back sooner than declared changes nothing.
#[test]
fn a_short_outage_is_the_run_it_was() {
    for fault in [camera(20.0, 28.0), payload(20.0, 70.0)] {
        assert_eq!(
            frames(&options(vec![fault.clone()], true)),
            frames(&options(vec![fault], false))
        );
    }
}

#[test]
fn a_run_resumed_on_the_way_home_finishes_as_the_uninterrupted_one() {
    let options = options(vec![camera(20.0, 1.0e6)], true);
    let straight = frames(&options);
    for at in [26usize, 34, 60] {
        let mut original = run_with(&options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&options);
        resumed.restore(&checkpoint).expect("restore accepted");
        for (k, frame) in straight.iter().enumerate().skip(at) {
            resumed.step().unwrap();
            assert_eq!(
                &serde_json::to_value(resumed.last_frame().unwrap()).unwrap(),
                frame,
                "resumed at {at}, tick {k}"
            );
        }
    }
    let mut run = run_with(&options);
    for _ in 0..60 {
        run.step().unwrap();
    }
    run.reset();
    let again: Vec<Value> = run
        .run_to_replay()
        .unwrap()
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect();
    assert_eq!(again, straight, "a reset run is a fresh one");
}

#[test]
fn a_fault_return_is_refused_where_nothing_could_fly_it() {
    let bundle = bundle();
    let refused = |options: &RunOptions| {
        let mut run = assemble(&bundle).unwrap();
        options.apply(&mut run, &bundle).unwrap_err()
    };
    let mut ungoverned = options(vec![], true);
    ungoverned.exploration.as_mut().unwrap().mission_flights = None;
    assert!(refused(&ungoverned).contains("mission flights"));
    let mut reference = options(vec![], true);
    reference.planning_policy = argusnet_sim::run_options::PlanningPolicy::Reference;
    assert!(refused(&reference).contains("policy"));
    let mut still = options(vec![], true);
    still.exploration.as_mut().unwrap().fault_return = Some(FaultReturnConfig {
        blind_speed_mps: 0.0,
        ..Default::default()
    });
    assert!(refused(&still).contains("blind_speed_mps"));
}
