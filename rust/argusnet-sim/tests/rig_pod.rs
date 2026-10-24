//! SR1/SR2: a run that flies a declared rig maps with the pod's LiDAR on a
//! 3-axis gimbal.
//!
//! The gates: the rig is recorded with its identity, its mass and what it
//! simulates; every controller flies low enough for the LiDAR's furthest
//! ray, and the map is good; stowed and unsettled captures are refused; an
//! injected boresight bias shows in the map; no platform is ever below the
//! true ground; the run is deterministic and resumes exactly, and a
//! checkpoint does not cross between a rig and no rig; an invalid rig is
//! refused before the run.

use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::motion::TerrainSampler;
use argusnet_sim::pod::PointingPolicy;
use argusnet_sim::rig::RigProfile;
use argusnet_sim::scenario::{assemble, build_world, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn rigged(rig: RigProfile, pointing: Option<PointingPolicy>) -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(rig),
        pointing,
        ..ExplorationConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
        .expect("assembles");
    run
}

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

fn belief(frame: &serde_json::Value) -> &serde_json::Value {
    &frame["mapping_state"]["belief"]
}

#[test]
fn a_rig_run_records_its_rig_and_maps_from_the_pod() {
    let bundle = bundle();
    let (meta, frames) = replay(run_with(&bundle, rigged(RigProfile::provisional(), None)));
    let knowledge = &meta["world_knowledge"];
    assert_eq!(
        knowledge["rig_identity"],
        RigProfile::provisional().identity()
    );
    assert_eq!(
        knowledge["rig_sensors"]["declared_not_simulated"],
        serde_json::json!(["visible-left", "visible-right", "thermal", "mapping-imu"])
    );
    assert!((knowledge["rig_mass_kg"]["scanning_kg"].as_f64().unwrap() - 2.65).abs() < 1e-9);
    assert_eq!(knowledge["pointing"]["mode"], "nadir");
    // The rig's navigation camera is the avoidance sensor.
    assert!(knowledge["nav_camera_envelope"]["max_speed_mps"].is_number());
    // The legacy fixture's lanes fly 230 m up; the 250 m LiDAR's corner ray
    // reaches the ground only from ~148 m.
    let cap = knowledge["mapping_agl_cap_m"].as_f64().unwrap();
    assert!(cap > 120.0 && cap < 160.0, "{cap}");

    let last = frames.last().unwrap();
    let b = belief(last);
    assert!(b["known_fraction"].as_f64().unwrap() > 0.95, "{b}");
    assert!(b["ground_rmse_m"].as_f64().unwrap() < 1.5, "{b}");
    assert_eq!(b["structure_recall"], 1.0);
    // Every scan of the map is the pod's.
    let pod_scans = frames
        .iter()
        .flat_map(|f| f["range_scans"].as_array().cloned().unwrap_or_default())
        .filter(|s| s.get("sensor").is_none())
        .collect::<Vec<_>>();
    assert!(!pod_scans.is_empty());
    assert!(pod_scans.iter().all(|s| s["pod"].is_object()));
}

#[test]
fn stowed_and_unsettled_captures_never_reach_the_map() {
    let bundle = bundle();
    let (_, frames) = replay(run_with(&bundle, rigged(RigProfile::provisional(), None)));
    let last = belief(frames.last().unwrap());
    assert!(
        last["stowed_scans"].as_u64().unwrap() > 0,
        "the drones park and launch: {last}"
    );
    for frame in &frames {
        for scan in frame["range_scans"].as_array().into_iter().flatten() {
            let Some(pod) = scan.get("pod") else { continue };
            let node = frame["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|n| n["node_id"] == scan["node_id"])
                .unwrap();
            let v = &node["velocity"];
            let speed = v[0].as_f64().unwrap().hypot(v[1].as_f64().unwrap());
            // Stowed exactly when below cruise.
            assert_eq!(
                pod["source"] == "stow",
                speed <= argusnet_sim::pod::CRUISE_THRESHOLD_MPS,
                "{pod} at {speed} m/s"
            );
        }
    }
}

#[test]
fn a_boresight_bias_shows_in_the_map() {
    let bundle = bundle();
    let rmse_with = |pitch_bias_rad: f64| {
        let mut rig = RigProfile::provisional();
        rig.calibration.lidar_boresight_bias_rad = [0.0, pitch_bias_rad, 0.0];
        let mut run = run_with(&bundle, rigged(rig, None));
        for _ in 0..120 {
            run.step().unwrap();
        }
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        belief(&frame)["ground_rmse_m"].as_f64().unwrap()
    };
    let nominal = rmse_with(0.0);
    let biased = rmse_with(0.02);
    // Measured 2026-10-23: 1.41 m nominal, 2.08 m at 20 mrad. Near nadir a
    // pitch bias mostly moves a return along the track rather than up or
    // down, so the height error grows on slopes rather than everywhere.
    assert!(
        biased > nominal * 1.3,
        "biased {biased} against nominal {nominal}"
    );
}

#[test]
fn no_platform_is_ever_below_the_true_ground() {
    let bundle = bundle();
    let world = build_world(&bundle.world).unwrap();
    for pointing in [
        None,
        Some(PointingPolicy::CrossTrackSweep {
            amplitude_deg: 25.0,
            period_s: 8.0,
        }),
    ] {
        let (_, frames) = replay(run_with(
            &bundle,
            rigged(RigProfile::provisional(), pointing),
        ));
        for (tick, frame) in frames.iter().enumerate() {
            for node in frame["nodes"].as_array().unwrap() {
                if node["is_mobile"] != true {
                    continue;
                }
                let p = &node["position"];
                let (x, y, z) = (
                    p[0].as_f64().unwrap(),
                    p[1].as_f64().unwrap(),
                    p[2].as_f64().unwrap(),
                );
                let ground = world.terrain.height_at(x, y);
                assert!(
                    z >= ground - 1e-6,
                    "{} at tick {tick} is {:.1} m below the ground",
                    node["node_id"],
                    ground - z
                );
            }
        }
    }
}

#[test]
fn a_rig_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let config = rigged(
        RigProfile::provisional(),
        Some(PointingPolicy::CrossTrackSweep {
            amplitude_deg: 20.0,
            period_s: 7.0,
        }),
    );
    let (_, straight) = replay(run_with(&bundle, config.clone()));
    assert_eq!(straight, replay(run_with(&bundle, config.clone())).1);
    let at = 45usize;
    let mut original = run_with(&bundle, config.clone());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    assert!(
        text.contains("on_target_since_s"),
        "the gimbals are checkpointed"
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
    // Neither way across a rig and no rig.
    let mut plain = run_with(&bundle, ExplorationConfig::default());
    assert!(plain.restore(&checkpoint).is_err());
    let mut plain_run = run_with(&bundle, ExplorationConfig::default());
    for _ in 0..5 {
        plain_run.step().unwrap();
    }
    let without = plain_run.sim.checkpoint().unwrap();
    let mut rig_run = run_with(&bundle, rigged(RigProfile::provisional(), None));
    assert!(rig_run.restore(&without).is_err());
}

#[test]
fn an_invalid_rig_or_a_second_camera_is_refused_before_the_run() {
    let bundle = bundle();
    let mut heavy = RigProfile::provisional();
    heavy.mass[4].mass_kg = 4.0;
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(WorldKnowledge::NoPrior, rigged(heavy, None))
        .unwrap_err();
    assert!(error.contains("10 lb"), "{error}");
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                nav_camera: Some(Default::default()),
                ..rigged(RigProfile::provisional(), None)
            },
        )
        .unwrap_err();
    assert!(error.contains("second one"), "{error}");
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                pointing: Some(PointingPolicy::Nadir),
                ..ExplorationConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.contains("needs a rig"), "{error}");
}
