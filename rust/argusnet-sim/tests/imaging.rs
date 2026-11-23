//! SR4: what the pod's images are worth, and an image requirement.

use argusnet_mapping::quality::QualityRequirement;
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::imaging::ImagingConfig;
use argusnet_sim::pod::PointingPolicy;
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

fn options(imaging: bool, gsd_m: Option<f64>, pointing: Option<PointingPolicy>) -> RunOptions {
    RunOptions {
        world_knowledge: WorldKnowledge::NoPrior,
        exploration: Some(ExplorationConfig {
            rig: Some(RigProfile::provisional()),
            pointing,
            imaging: imaging.then(ImagingConfig::default),
            quality: Some(QualityRequirement {
                max_image_gsd_m: gsd_m,
                ..FIELD_V1_QUALITY
            }),
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

fn frames(options: &RunOptions) -> Vec<Value> {
    run_with(options)
        .run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

/// Every pod capture's images, over a run.
fn images(frames: &[Value]) -> Vec<Value> {
    frames
        .iter()
        .flat_map(|frame| frame["range_scans"].as_array().cloned().unwrap_or_default())
        .filter_map(|scan| scan["pod"].get("images").cloned())
        .collect()
}

fn share(images: &[Value], camera: &str, accepted: impl Fn(&Value) -> bool) -> f64 {
    let looking: Vec<&Value> = images
        .iter()
        .map(|image| &image[camera])
        .filter(|image| image["refusal"] != "stowed" && image["refusal"] != "unsettled")
        .collect();
    assert!(looking.len() > 50, "{} images", looking.len());
    looking.iter().filter(|image| accepted(image)).count() as f64 / looking.len() as f64
}

/// With no image requirement the cameras only watch: the run is the run it
/// was, but for what they record.
#[test]
fn watching_changes_nothing() {
    let without = frames(&options(false, None, None));
    let mut with = frames(&options(true, None, None));
    assert!(images(&without).is_empty());
    let seen = images(&with);
    assert!(!seen.is_empty());
    for frame in &mut with {
        if let Some(scans) = frame["range_scans"].as_array_mut() {
            for scan in scans {
                if let Some(pod) = scan.get_mut("pod").and_then(Value::as_object_mut) {
                    pod.remove("images");
                }
            }
        }
    }
    assert_eq!(with, without);
    // A replay shows what images are worth, not the pose the map used.
    assert!(seen.iter().all(|image| image.get("visible_pose").is_none()));
    let refused = |image: &&Value| image["visible"]["refusal"] == "stowed";
    assert!(seen.iter().any(|image| refused(&image)));
    let usable = seen
        .iter()
        .find(|image| image["visible"].get("refusal").is_none())
        .expect("an image that counts");
    assert!(usable["visible"]["gsd_m"].as_f64().unwrap() > 0.0);
    assert!(usable["stereo_depth_sigma_m"].as_f64().unwrap() > 0.0);
    assert!(usable["thermal"]["blur_px"].as_f64().unwrap() > 0.0);
}

/// A sweeping pod is settled on its command and still turning: its images
/// blur where a nadir pod's do not.
#[test]
fn a_sweep_blurs_what_nadir_keeps_sharp() {
    let sharp = |pointing| {
        let images = images(&frames(&options(true, None, pointing)));
        share(&images, "visible", |image| image.get("refusal").is_none())
    };
    let nadir = sharp(None);
    let sweep = sharp(Some(PointingPolicy::CrossTrackSweep {
        amplitude_deg: 30.0,
        period_s: 8.0,
    }));
    assert!(nadir > 0.5, "nadir keeps {nadir}");
    assert!(sweep < 0.5 * nadir, "sweep keeps {sweep} against {nadir}");
}

/// A requirement lowers the flight to where its images are fine enough, and
/// the map reports what still lacks one.
#[test]
fn an_image_requirement_is_flown_for() {
    let count = |frames: &[Value], key: &str| {
        frames.last().unwrap()["mapping_state"]["belief"]["quality"][key]
            .as_u64()
            .unwrap_or(0)
    };
    let free = frames(&options(true, None, None));
    let held = frames(&options(true, Some(0.02), None));
    let fine = |frames: &[Value]| {
        share(&images(frames), "visible", |image| {
            image["gsd_m"].as_f64().is_some_and(|gsd| gsd <= 0.02)
        })
    };
    assert!(fine(&free) < 0.1, "{}", fine(&free));
    assert!(fine(&held) > 0.5, "{}", fine(&held));
    assert!(count(&held, "met_cells") > 500, "cells meet it");
    assert!(
        held.iter().any(|frame| {
            frame["mapping_state"]["belief"]["quality"]["image_coarse_cells"]
                .as_u64()
                .unwrap_or(0)
                > 0
        }),
        "cells lacking only an image are named"
    );
    // Finer than the floor allows is a limit, never met and never waited on.
    let beyond = frames(&options(true, Some(0.005), None));
    assert_eq!(count(&beyond, "met_cells"), 0);
    assert!(count(&beyond, "unreachable_cells") > 500);
    assert!(fine(&beyond) < 0.1, "nor flown low for: {}", fine(&beyond));
}

/// A flight already lower than the requirement asks is slowed for the height
/// it flies: at the speed the requirement's own height allows, every image
/// would blur.
#[test]
fn a_lower_flight_is_slowed_for_its_own_height() {
    let long_exposure = ImagingConfig {
        visible_exposure_s: 4.0e-3,
        ..ImagingConfig::default()
    };
    let kept = |gsd_m| {
        let mut options = options(true, gsd_m, None);
        options.exploration.as_mut().unwrap().imaging = Some(long_exposure.clone());
        share(&images(&frames(&options)), "visible", |image| {
            image.get("refusal").is_none()
        })
    };
    let (free, held) = (kept(None), kept(Some(0.2)));
    // 0.2 m asks for no lower than 600 m, so only the speed can help.
    assert!(free < 0.5, "unheld, {free} of images count");
    assert!(held > 0.8, "held, {held} of images count");
}

#[test]
fn a_run_with_an_image_requirement_resumes_exactly() {
    let options = options(true, Some(0.02), None);
    let straight = frames(&options);
    for at in [40usize, 180] {
        let mut original = run_with(&options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&options);
        resumed.restore(&checkpoint).expect("restore accepted");
        for frame in &straight[at..] {
            resumed.step().unwrap();
            assert_eq!(
                &serde_json::to_value(resumed.last_frame().unwrap()).unwrap(),
                frame
            );
        }
    }
    // A reset forgets the images with the map.
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
    assert_eq!(again, straight);
}

#[test]
fn a_requirement_without_the_model_or_a_model_without_a_rig_is_refused() {
    let bundle = bundle();
    let refused = |options: &RunOptions| {
        let mut run = assemble(&bundle).unwrap();
        options.apply(&mut run, &bundle).unwrap_err()
    };
    assert!(refused(&options(false, Some(0.02), None)).contains("imaging model"));
    let mut rigless = options(true, None, None);
    rigless.exploration.as_mut().unwrap().rig = None;
    assert!(refused(&rigless).contains("rig"));
    let mut blind = options(true, None, None);
    let rig = blind.exploration.as_mut().unwrap().rig.as_mut().unwrap();
    for sensor in &mut rig.sensors {
        if sensor.role == argusnet_sim::rig::SensorRole::VisibleCamera {
            sensor.camera = None;
        }
    }
    assert!(refused(&blind).contains("camera"), "{}", refused(&blind));
}
