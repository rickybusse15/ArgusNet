//! SR5d: view planning. Explorers cruise to the deficit worth most per
//! watt-hour; the pod's roll can be planned per leg from the deficit map.
//!
//! The gates: view planning needs a rig and a quality requirement and is
//! refused otherwise; aircraft keep cruising while they map (no hover to
//! look); a planned pointing rolls off nadir but stays settled; the run is
//! deterministic and resumes exactly; and, in release, it maps more of the
//! regional scenario than the nearest-goal explorer for no more energy
//! (`--ignored`).

use argusnet_mapping::quality::QualityRequirement;
use argusnet_planning::views::ViewPlanningConfig;
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::pod::PointingPolicy;
use argusnet_sim::rig::RigProfile;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn requirement() -> QualityRequirement {
    QualityRequirement {
        min_ground_density_per_m2: Some(0.03),
        max_vertical_sigma_m: Some(0.5),
        max_age_s: None,
        max_incidence_deg: 40.0,
        bias_floor_m: 0.05,
    }
}

fn planned(view_planning: Option<ViewPlanningConfig>) -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(RigProfile::provisional()),
        quality: Some(requirement()),
        view_planning,
        ..ExplorationConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
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

fn pod_captures(frames: &[serde_json::Value]) -> Vec<serde_json::Value> {
    frames
        .iter()
        .flat_map(|f| f["range_scans"].as_array().cloned().unwrap_or_default())
        .filter_map(|scan| scan.get("pod").cloned())
        .collect()
}

#[test]
fn view_planning_needs_a_rig_and_a_requirement() {
    let bundle = bundle();
    let refused = |config: ExplorationConfig| {
        let mut run = assemble(&bundle).unwrap();
        run.set_world_knowledge(WorldKnowledge::NoPrior, config)
            .unwrap_err()
    };
    let error = refused(ExplorationConfig {
        quality: None,
        ..planned(Some(ViewPlanningConfig::default()))
    });
    assert!(error.contains("needs a rig"), "{error}");
    let error = refused(ExplorationConfig {
        rig: None,
        ..planned(Some(ViewPlanningConfig::default()))
    });
    assert!(error.contains("needs a rig"), "{error}");
    let error = refused(ExplorationConfig {
        pointing: Some(PointingPolicy::Nadir),
        ..planned(Some(ViewPlanningConfig::default()))
    });
    assert!(error.contains("chooses the pointing"), "{error}");
    let error = refused(planned(Some(ViewPlanningConfig {
        claimed_worth: 2.0,
        ..ViewPlanningConfig::default()
    })));
    assert!(error.contains("claimed_worth"), "{error}");
}

/// Cruise only: once airborne and scanning, no aircraft stows its pod to
/// hold and look. Every pod capture is the mission's until the scan ends.
#[test]
fn aircraft_keep_cruising_while_they_map() {
    let bundle = bundle();
    let frames = frames(run_with(
        &bundle,
        planned(Some(ViewPlanningConfig::default())),
    ));
    let launched_s = 10.0;
    for frame in &frames {
        if frame["scan_mission_state"]["phase"] != "scanning"
            || frame["timestamp_s"].as_f64().unwrap() < launched_s
        {
            continue;
        }
        for scan in frame["range_scans"].as_array().into_iter().flatten() {
            if let Some(pod) = scan.get("pod") {
                assert_eq!(
                    pod["source"], "mission",
                    "{} held to look at {}",
                    scan["node_id"], frame["timestamp_s"]
                );
            }
        }
    }
}

#[test]
fn planned_pointing_rolls_off_nadir_and_stays_settled() {
    let bundle = bundle();
    let pointing = ViewPlanningConfig {
        plan_pointing: true,
        ..ViewPlanningConfig::default()
    };
    // At a 40 degree limit the provisional LiDAR's 70 degree swath already
    // fills what the requirement accepts, so rolling only throws rays away
    // and the planner holds nadir. At 60 there is room to roll into.
    let config = ExplorationConfig {
        quality: Some(QualityRequirement {
            max_incidence_deg: 60.0,
            ..requirement()
        }),
        ..planned(Some(pointing.clone()))
    };
    let at_40 = pod_captures(&frames(run_with(&bundle, planned(Some(pointing)))));
    assert!(
        at_40
            .iter()
            .filter(|pod| pod["source"] == "mission" && pod["settled"] == true)
            .all(|pod| pod["boresight_world"][2].as_f64().unwrap() < -0.999),
        "no roll pays at 40 degrees"
    );
    let frames = frames(run_with(&bundle, config));
    let captures = pod_captures(&frames);
    let mission: Vec<_> = captures
        .iter()
        .filter(|pod| pod["source"] == "mission")
        .collect();
    let off_nadir = mission
        .iter()
        .filter(|pod| pod["boresight_world"][2].as_f64().unwrap() > -0.999)
        .count();
    assert!(off_nadir > 0, "the planner rolls somewhere");
    // A roll is chosen per leg, not per capture, so few captures are lost
    // to slewing.
    let unsettled = mission.iter().filter(|pod| pod["settled"] == false).count();
    assert!(
        unsettled * 20 < mission.len(),
        "{unsettled} of {} unsettled",
        mission.len()
    );
    // By default the pod is not rolled.
    let nadir = pod_captures(&frames_of_default(&bundle));
    assert!(nadir
        .iter()
        .filter(|pod| pod["source"] == "mission" && pod["settled"] == true)
        .all(|pod| pod["boresight_world"][2].as_f64().unwrap() < -0.999));
}

fn frames_of_default(bundle: &ScenarioBundle) -> Vec<serde_json::Value> {
    frames(run_with(
        bundle,
        planned(Some(ViewPlanningConfig::default())),
    ))
}

#[test]
fn a_planned_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let config = planned(Some(ViewPlanningConfig {
        plan_pointing: true,
        ..ViewPlanningConfig::default()
    }));
    let straight = frames(run_with(&bundle, config.clone()));
    assert_eq!(straight, frames(run_with(&bundle, config.clone())));
    let at = 70usize;
    let mut original = run_with(&bundle, config.clone());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
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

/// The regional scenario (16 drones, ~179,000 cells, 120 s) on three seeds:
/// view planning meets more cells than the nearest-goal explorer on every
/// one, for no more energy. Measured 2026-10-27: +26%, +9%, +10%. Slow in
/// debug; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn view_planning_maps_more_of_the_regional_scenario_per_watt_hour() {
    use argusnet_sim::builder::{build_bundle, ScenarioConfig};
    for seed in [7, 11, 23] {
        let config: ScenarioConfig = serde_json::from_value(serde_json::json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "seed": seed,
            "dt_s": 0.25,
            "duration_s": 120.0,
            "options": {
                "map_preset": "regional",
                "drone_count": 16,
                "mission_mode": "scan_map_inspect",
            },
        }))
        .unwrap();
        let document = build_bundle(&config).unwrap();
        let bundle = ScenarioBundle::from_json(&document.to_string()).unwrap();
        let outcome = |view_planning| {
            let frames = frames(run_with(&bundle, planned(view_planning)));
            let last = frames.last().unwrap();
            let met = last["mapping_state"]["belief"]["quality"]["met_cells"]
                .as_u64()
                .unwrap();
            let used: f64 = last["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|n| n["is_mobile"] == true)
                .map(|n| 1.0 - n["battery_fraction"].as_f64().unwrap())
                .sum();
            (met, used)
        };
        let (nearest_met, nearest_used) = outcome(None);
        let (planned_met, planned_used) = outcome(Some(ViewPlanningConfig::default()));
        assert!(
            planned_met as f64 > nearest_met as f64 * 1.05,
            "seed {seed}: planned {planned_met} against nearest {nearest_met}"
        );
        assert!(
            planned_used <= nearest_used * 1.02,
            "seed {seed}: planned used {planned_used} against {nearest_used}"
        );
    }
}
