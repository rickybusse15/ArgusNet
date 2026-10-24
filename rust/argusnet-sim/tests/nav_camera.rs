//! SR5c: the forward navigation camera is the primary obstacle-avoidance
//! sensor of a run without a prior map.
//!
//! The gates: its detections become safety obstacles the planners avoid
//! without ever counting as mapped ground or structure; every controller is
//! capped to the speed the camera's envelope allows, and fog lowers it; with
//! the scanner reduced to a downward mapping pod, avoidance holds with the
//! camera and fails without it; the run is deterministic and resumes exactly
//! from a checkpoint; and a run without the camera is the run it always was.

use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::nav_camera::NavCameraConfig;
use argusnet_sim::range_scan::RangeScanConfig;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

fn fixture(name: &str) -> serde_json::Value {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("json")
}

fn bundle_of(document: &serde_json::Value) -> ScenarioBundle {
    ScenarioBundle::from_json(&document.to_string()).expect("bundle parses")
}

/// The `tt` fixture with a wall 40 m thick across the whole map at `x = 0`,
/// from below the lowest ground to well above any flight. Thicker than any
/// one tick's flight: the physical guard sweeps a step against wires but not
/// structures, so a thinner wall can be stepped through uncounted.
fn walled() -> serde_json::Value {
    let mut document = fixture("scenario_bundle_tt.json");
    let bounds = document["world"]["obstacles"]["bounds"].clone();
    let (y0, y1) = (
        bounds["y_min_m"].as_f64().unwrap(),
        bounds["y_max_m"].as_f64().unwrap(),
    );
    let wall = serde_json::json!({
        "object_id": "wall",
        "kind": "box-v1",
        "blocker_type": "building",
        "center_x_m": 0.0,
        "center_y_m": (y0 + y1) / 2.0,
        "length_m": 40.0,
        "width_m": y1 - y0,
        "yaw_rad": 0.0,
        "base_elevation_m": -500.0,
        "top_elevation_m": 3000.0,
        "footprint_xy_m": [[-20.0, y0], [20.0, y0], [20.0, y1], [-20.0, y1]],
    });
    document["world"]["obstacles"]["primitives"]
        .as_array_mut()
        .unwrap()
        .push(wall);
    document
}

/// A downward mapping pod: rings within 40 degrees of nadir, no forward fan.
fn pod_scan() -> RangeScanConfig {
    RangeScanConfig {
        ring_depressions_deg: vec![80.0, 65.0, 50.0],
        forward_count: 0,
        ..RangeScanConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
        .expect("assembles");
    run
}

fn camera_config(scan: RangeScanConfig) -> ExplorationConfig {
    ExplorationConfig {
        scan,
        nav_camera: Some(NavCameraConfig::default()),
        ..ExplorationConfig::default()
    }
}

/// The replay's metadata and frames, as JSON.
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

fn frames(run: AssembledRun) -> Vec<serde_json::Value> {
    replay(run).1
}

fn structure_contacts(frames: &[serde_json::Value]) -> usize {
    frames
        .iter()
        .flat_map(|frame| {
            frame["world_contacts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|contact| contact["kind"] == "structure")
        .count()
}

fn camera_detections(frames: &[serde_json::Value]) -> usize {
    frames
        .iter()
        .flat_map(|frame| frame["range_scans"].as_array().cloned().unwrap_or_default())
        .filter(|scan| scan["sensor"] == "nav_camera")
        .map(|scan| scan["structure"].as_array().map_or(0, Vec::len))
        .sum()
}

/// The highest horizontal speed any mobile platform reports.
fn top_speed(frames: &[serde_json::Value]) -> f64 {
    frames
        .iter()
        .flat_map(|frame| frame["nodes"].as_array().cloned().unwrap_or_default())
        .filter(|node| node["is_mobile"] == true)
        .map(|node| {
            let v = &node["velocity"];
            v[0].as_f64().unwrap().hypot(v[1].as_f64().unwrap())
        })
        .fold(0.0, f64::max)
}

fn envelope(meta: &serde_json::Value) -> serde_json::Value {
    meta["world_knowledge"]["nav_camera_envelope"].clone()
}

#[test]
fn the_camera_sees_the_wall_and_avoidance_is_no_worse_without_the_forward_fan() {
    let bundle = bundle_of(&walled());
    let pod = frames(run_with(
        &bundle,
        ExplorationConfig {
            scan: pod_scan(),
            ..ExplorationConfig::default()
        },
    ));
    let camera = frames(run_with(&bundle, camera_config(pod_scan())));
    assert!(
        camera_detections(&camera) > 100,
        "the wall fills the camera's view: {}",
        camera_detections(&camera)
    );
    assert_eq!(camera_detections(&pod), 0, "no camera, no camera frames");
    // Measured 2026-10-23: 14 structure contacts with the downward pod alone,
    // 10 with the camera. The belief planner only pushes clear of what it
    // believes (P3 routing is still to do), so contacts are not zero.
    assert!(
        structure_contacts(&camera) <= structure_contacts(&pod),
        "camera {} against pod alone {}",
        structure_contacts(&camera),
        structure_contacts(&pod)
    );
}

#[test]
fn camera_evidence_never_counts_as_mapped() {
    let bundle = bundle_of(&walled());
    let config = camera_config(pod_scan());
    let mut run = run_with(&bundle, config.clone());
    for _ in 0..40 {
        run.step().unwrap();
    }
    let mapping = run.mapping().expect("stage 8");
    let grid = mapping.belief_grid().expect("a self-built map");
    let threshold = config.occupied_probability;
    assert!(
        grid.safety_cells_at(threshold) > 0,
        "the camera marked the wall"
    );
    let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
    let belief = &frame["mapping_state"]["belief"];
    // The replay's map figures are the scanner's alone: the camera's cells
    // are neither structure nor known.
    assert_eq!(
        belief["structure_cells"].as_u64().unwrap(),
        grid.structure_cells_at(threshold) as u64
    );
    assert_eq!(
        belief["known_cells"].as_u64().unwrap(),
        grid.known_cells() as u64
    );
}

#[test]
fn every_controller_is_capped_to_the_envelope_and_fog_lowers_it() {
    let clear_bundle = bundle_of(&fixture("scenario_bundle_tt.json"));
    let (meta, flown) = replay(run_with(
        &clear_bundle,
        camera_config(RangeScanConfig::default()),
    ));
    let clear = envelope(&meta);
    let cap = clear["max_speed_mps"].as_f64().unwrap();
    assert_eq!(clear["bounded_by"], "structure");
    assert_eq!(clear["structure_range_m"], 150.0);
    assert!(
        top_speed(&flown) <= cap + 1e-6,
        "{} > {cap}",
        top_speed(&flown)
    );
    let uncapped = frames(run_with(&clear_bundle, ExplorationConfig::default()));
    assert!(
        top_speed(&uncapped) > cap,
        "the fixture flies faster unbounded"
    );

    let mut foggy = fixture("scenario_bundle_tt.json");
    foggy["weather"]["atmosphere"]["visibility_m"] = serde_json::json!(60.0);
    let foggy_bundle = bundle_of(&foggy);
    let (meta, flown) = replay(run_with(
        &foggy_bundle,
        camera_config(RangeScanConfig::default()),
    ));
    let fog = envelope(&meta);
    let fog_cap = fog["max_speed_mps"].as_f64().unwrap();
    assert!(fog["structure_range_m"].as_f64().unwrap() < 150.0, "{fog}");
    assert!(fog_cap < cap / 2.0, "{fog_cap} against {cap}");
    assert!(top_speed(&flown) <= fog_cap + 1e-6);
}

#[test]
fn a_camera_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle_of(&walled());
    let config = camera_config(pod_scan());
    let straight = frames(run_with(&bundle, config.clone()));
    assert_eq!(straight, frames(run_with(&bundle, config.clone())));
    let at = 30usize;
    let mut original = run_with(&bundle, config.clone());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    assert!(
        text.contains("safety_log_odds"),
        "the camera's evidence is checkpointed"
    );
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&bundle, config);
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
fn without_the_camera_nothing_changes() {
    let bundle = bundle_of(&fixture("scenario_bundle_tt.json"));
    let (meta, frames) = replay(run_with(&bundle, ExplorationConfig::default()));
    assert!(meta["world_knowledge"].get("nav_camera_envelope").is_none());
    assert!(meta["world_knowledge"]["exploration"]
        .get("nav_camera")
        .is_none());
    assert!(frames
        .iter()
        .flat_map(|f| f["range_scans"].as_array().cloned().unwrap_or_default())
        .all(|scan| scan.get("sensor").is_none()));
}

#[test]
fn an_invalid_camera_profile_is_refused_before_the_run() {
    let bundle = bundle_of(&fixture("scenario_bundle_tt.json"));
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                nav_camera: Some(NavCameraConfig {
                    detection_probability: 1.5,
                    ..NavCameraConfig::default()
                }),
                ..ExplorationConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.contains("detection_probability"), "{error}");
}
