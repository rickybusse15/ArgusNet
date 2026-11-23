//! The estimator in the loop: scans placed by an estimated pose, and a map
//! whose quality knows how well it was known.

use argusnet_mapping::quality::QualityRequirement;
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};
use argusnet_sim::navigation::NavigationConfig;
use argusnet_sim::rig::RigProfile;
use argusnet_sim::run_options::{RunOptions, FIELD_V1_QUALITY};
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use serde_json::Value;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn options(navigation: Option<NavigationConfig>) -> RunOptions {
    RunOptions {
        world_knowledge: WorldKnowledge::NoPrior,
        exploration: Some(ExplorationConfig {
            rig: Some(RigProfile::provisional()),
            quality: Some(FIELD_V1_QUALITY),
            map_report: true,
            navigation,
            ..ExplorationConfig::default()
        }),
        ..RunOptions::default()
    }
}

fn run_with(options: &RunOptions) -> AssembledRun {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    options.apply(&mut run, &bundle).expect("options apply");
    run
}

fn run(options: &RunOptions) -> (Value, Vec<Value>) {
    let replay = run_with(options).run_to_replay().expect("runs");
    let frames = replay
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect();
    (serde_json::to_value(&replay.meta).unwrap(), frames)
}

/// Nothing in the run has GNSS over the interval.
fn lost_everywhere(start_s: f64, end_s: f64) -> FaultManifest {
    let run = run_with(&options(None));
    FaultManifest {
        format: MANIFEST_FORMAT.into(),
        version: MANIFEST_VERSION,
        faults: run
            .observe()
            .sensors()
            .keys()
            .map(|node_id| Fault::GnssOutage {
                node_id: node_id.clone(),
                start_s,
                end_s,
            })
            .collect(),
    }
}

fn met(frame: &Value) -> f64 {
    frame["mapping_state"]["belief"]["quality"]["met_fraction"]
        .as_f64()
        .unwrap()
}

/// A run without it carries no trace of it.
#[test]
fn without_navigation_nothing_is_written() {
    let (meta, frames) = run(&options(None));
    let text = serde_json::to_string(&meta).unwrap();
    assert!(!text.contains("navigation\":") && !text.contains("pose_"));
    for frame in &frames {
        assert!(frame.get("navigation").is_none());
    }
}

/// Every frame says where each aircraft thinks it is, and a survey-grade
/// receiver keeps that within its own claim.
#[test]
fn every_frame_says_where_navigation_has_each_aircraft() {
    let (meta, frames) = run(&options(Some(NavigationConfig::rtk())));
    assert_eq!(
        meta["world_knowledge"]["exploration"]["navigation"]["id"],
        "rtk"
    );
    let (mut within, mut total) = (0u32, 0u32);
    for frame in &frames {
        let estimates = frame["navigation"].as_array().expect("estimates");
        assert!(!estimates.is_empty());
        for estimate in estimates {
            let node = frame["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|node| node["node_id"] == estimate["node_id"])
                .unwrap();
            let error =
                estimate["position_m"][2].as_f64().unwrap() - node["position"][2].as_f64().unwrap();
            total += 1;
            within +=
                u32::from(error.abs() <= 3.0 * estimate["vertical_sigma_m"].as_f64().unwrap());
        }
    }
    assert!(
        f64::from(within) > 0.95 * f64::from(total),
        "{within} of {total}"
    );
}

/// The defect this closes: a map placed by a poor pose used to claim what a
/// map placed by a good one did. Now it does not, and what it claims is no
/// less than its true error.
#[test]
fn the_map_claims_no_more_than_its_poses_allow() {
    let report = |navigation| {
        let (meta, frames) = run(&options(navigation));
        (meta["map_report"].clone(), met(frames.last().unwrap()))
    };
    let (_, truth_met) = report(None);
    let (rtk, rtk_met) = report(Some(NavigationConfig::rtk()));
    let (alone, alone_met) = report(Some(NavigationConfig::standalone()));
    assert!(truth_met > 0.5, "{truth_met}");
    assert!(rtk_met > truth_met - 0.1, "{rtk_met} against {truth_met}");
    assert_eq!(
        alone_met, 0.0,
        "metres of error meet no half-metre requirement"
    );
    let calibration = |report: &Value, key: &str| report["calibration"][key].as_f64().unwrap();
    assert!(calibration(&alone, "rms_claimed_sigma_m") > 1.0);
    assert!(
        calibration(&alone, "normalized_rms") < 1.5,
        "{}",
        alone["calibration"]
    );
    assert!(calibration(&rtk, "rms_claimed_sigma_m") < 0.5);
    assert!(
        rtk["achieved"]["horizontal_sigma_m"]["p50"]
            .as_f64()
            .unwrap()
            < 0.2
    );
}

/// F2's `gnss_outage`: what is mapped during it is known to be worse.
#[test]
fn a_gnss_outage_is_known_about_in_the_map() {
    let clear = options(Some(NavigationConfig::rtk()));
    let lost = RunOptions {
        faults: Some(lost_everywhere(10.0, 1.0e6)),
        ..clear.clone()
    };
    let (_, clear_frames) = run(&clear);
    let (_, lost_frames) = run(&lost);
    let sigma = |frames: &[Value]| {
        frames.last().unwrap()["navigation"][0]["horizontal_sigma_m"]
            .as_f64()
            .unwrap()
    };
    assert!(sigma(&clear_frames) < 0.1);
    assert!(sigma(&lost_frames) > 1.0);
    assert!(met(lost_frames.last().unwrap()) < met(clear_frames.last().unwrap()));
}

#[test]
fn what_needs_navigation_is_refused_without_it() {
    let bundle = bundle();
    let refused = |options: &RunOptions| {
        let mut run = assemble(&bundle).unwrap();
        options.apply(&mut run, &bundle).unwrap_err()
    };
    let outage = RunOptions {
        faults: Some(lost_everywhere(10.0, 20.0)),
        ..options(None)
    };
    assert!(refused(&outage).contains("gnss_outage"));
    let mut sideways = options(None);
    sideways.exploration.as_mut().unwrap().quality = Some(QualityRequirement {
        max_horizontal_sigma_m: Some(0.5),
        ..FIELD_V1_QUALITY
    });
    assert!(refused(&sideways).contains("horizontal"));
}

/// Checkpointed inside an outage, a run finishes as it would have; and its
/// checkpoint is not one for a run without navigation.
#[test]
fn a_run_resumes_exactly_inside_an_outage() {
    let options_lost = RunOptions {
        faults: Some(lost_everywhere(5.0, 15.0)),
        ..options(Some(NavigationConfig::standalone()))
    };
    let (_, straight) = run(&options_lost);
    let mut original = run_with(&options_lost);
    let at = (10.0 / original.dt_s()).round() as usize;
    for _ in 0..at {
        original.step().expect("steps");
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&options_lost);
    resumed.restore(&checkpoint).expect("restore accepted");
    for frame in &straight[at..] {
        resumed.step().expect("steps");
        assert_eq!(
            &serde_json::to_value(resumed.last_frame().unwrap()).unwrap(),
            frame
        );
    }
    let mut without = run_with(&options(None));
    assert!(without.restore(&checkpoint).is_err());
}

/// Lost in height by more than it clears the ground, an aircraft flies into
/// what it does not know is there; knowing its height, it does not. The
/// replay's nodes are where it was, and no step jumps.
#[test]
fn a_lost_aircraft_meets_what_it_flies_into() {
    let with = |correction: bool, vertical_m: f64| {
        let mut config = NavigationConfig::rtk();
        config.sensors.gnss_correlated_sigma_m = [0.5, vertical_m];
        config.sensors.gnss_correlation_s = 1.0e4;
        if !correction {
            config.correction_mps = None;
        }
        run(&options(Some(config))).1
    };
    let contacts = |frames: &[Value]| -> usize {
        frames
            .iter()
            .filter_map(|frame| frame["world_contacts"].as_array())
            .map(Vec::len)
            .sum()
    };
    let known = contacts(&with(true, 0.02));
    let lost = with(true, 120.0);
    let not_flown = contacts(&with(false, 120.0));
    assert!(
        contacts(&lost) > known,
        "{} against {known}",
        contacts(&lost)
    );
    assert!(contacts(&lost) > not_flown, "flying truth: {not_flown}");
    // Where the replay has it is not where it thought it was.
    let off = |frame: &Value| -> f64 {
        let estimate = &frame["navigation"][0];
        let node = frame["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["node_id"] == estimate["node_id"])
            .unwrap();
        (estimate["position_m"][2].as_f64().unwrap() - node["position"][2].as_f64().unwrap()).abs()
    };
    let worst = lost.iter().map(off).fold(0.0f64, f64::max);
    assert!(worst > 20.0, "{worst}");
}
