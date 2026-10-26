//! SR5a/SR5b: a run without a prior map measured against a declared quality
//! requirement, and planned against what still falls short.
//!
//! The gates: without a requirement nothing changes; with one each cell says
//! why it falls short; noisier sensing meets less; a bias above the
//! requirement makes cells unreachable, which end the scan without being
//! counted as met, and the scan decision names them; old evidence goes
//! stale; the run is deterministic and resumes exactly from a checkpoint.

use argusnet_mapping::quality::QualityRequirement;
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::range_scan::RangeScanConfig;
use argusnet_sim::scenario::{assemble, AssembledRun, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;
use argusnet_sim::snapshot::SimulationCheckpoint;

fn bundle(name: &str) -> ScenarioBundle {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

fn with_quality(quality: QualityRequirement) -> ExplorationConfig {
    ExplorationConfig {
        quality: Some(quality),
        ..ExplorationConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
        .expect("assembles");
    run
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

fn quality(frame: &serde_json::Value) -> serde_json::Value {
    frame["mapping_state"]["belief"]["quality"].clone()
}

fn count(frame: &serde_json::Value, key: &str) -> u64 {
    quality(frame)[key].as_u64().unwrap()
}

/// The tick the scan phase first leaves `scanning`, if it does.
fn scan_ends(frames: &[serde_json::Value]) -> Option<usize> {
    frames
        .iter()
        .position(|frame| frame["scan_mission_state"]["phase"] != "scanning")
}

#[test]
fn without_a_requirement_nothing_changes() {
    let bundle = bundle("scenario_bundle_scan.json");
    let (meta, frames) = replay(run_with(&bundle, ExplorationConfig::default()));
    assert!(meta["world_knowledge"]["exploration"]
        .get("quality")
        .is_none());
    assert!(frames
        .iter()
        .all(|frame| frame["mapping_state"]["belief"].get("quality").is_none()));
    // The scan gate still reads the known fraction.
    let last = frames.last().unwrap();
    assert_eq!(
        last["scan_mission_state"]["scan_coverage_fraction"],
        last["mapping_state"]["belief"]["known_fraction"]
    );
}

#[test]
fn every_cell_says_why_it_falls_short_and_seen_is_not_met() {
    let bundle = bundle("scenario_bundle_scan.json");
    let (meta, frames) = replay(run_with(&bundle, with_quality(requirement())));
    assert_eq!(
        meta["world_knowledge"]["exploration"]["quality"]["max_incidence_deg"],
        40.0
    );
    let first = &frames[0];
    let last = frames.last().unwrap();
    let total = last["mapping_state"]["belief"]["total_cells"]
        .as_u64()
        .unwrap();
    let sum = |f: &serde_json::Value| {
        [
            "met_cells",
            "unseen_cells",
            "too_oblique_cells",
            "sparse_cells",
            "stale_cells",
            "unreachable_cells",
        ]
        .iter()
        .map(|k| count(f, k))
        .sum::<u64>()
    };
    assert_eq!(sum(first), total, "every cell has a reason");
    assert_eq!(sum(last), total);
    assert!(count(first, "unseen_cells") > total / 2);
    assert!(count(last, "met_cells") > count(first, "met_cells"));
    assert!(count(last, "sparse_cells") > 0, "seen is not good enough");
    assert!(
        count(last, "too_oblique_cells") > 0,
        "shallow rays are not accepted"
    );
    // Seen is not met: the map is far more known than it is good.
    let known = last["mapping_state"]["belief"]["known_fraction"]
        .as_f64()
        .unwrap();
    let met = quality(last)["met_fraction"].as_f64().unwrap();
    assert!(met < known - 0.2, "met {met} against known {known}");
    // The scan runs longer than on the known fraction, which it passes
    // almost at once.
    let (_, legacy) = replay(run_with(&bundle, ExplorationConfig::default()));
    assert!(scan_ends(&frames).unwrap() > 3 * scan_ends(&legacy).unwrap());
}

#[test]
fn noisier_sensing_meets_less() {
    let bundle = bundle("scenario_bundle_scan.json");
    let noisy = ExplorationConfig {
        scan: RangeScanConfig {
            range_noise_m: 1.5,
            ..RangeScanConfig::default()
        },
        ..with_quality(requirement())
    };
    let met_at = |config: ExplorationConfig| {
        let mut run = run_with(&bundle, config);
        for _ in 0..150 {
            run.step().unwrap();
        }
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        count(&frame, "met_cells")
    };
    let clean = met_at(with_quality(requirement()));
    let degraded = met_at(noisy);
    assert!(
        degraded < clean / 2,
        "noisy {degraded} against clean {clean}"
    );
}

#[test]
fn a_bias_above_the_requirement_is_unreachable_and_never_met() {
    let bundle = bundle("scenario_bundle_scan.json");
    let biased = QualityRequirement {
        bias_floor_m: 0.6,
        ..requirement()
    };
    let mut run = run_with(&bundle, with_quality(biased));
    let mut unreachable_named = false;
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let body = run.journal_body();
        unreachable_named |= body["decisions"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|d| d["inputs"]["quality_unreachable_cells"].as_u64() > Some(0));
    }
    assert!(
        unreachable_named,
        "the scan decision names the unreachable cells"
    );
    let (_, frames) = replay(run_with(
        &bundle,
        with_quality(QualityRequirement {
            bias_floor_m: 0.6,
            ..requirement()
        }),
    ));
    for frame in &frames {
        assert_eq!(count(frame, "met_cells"), 0, "no cell is ever met");
        assert_eq!(count(frame, "sparse_cells"), 0, "more returns cannot help");
    }
    let last = frames.last().unwrap();
    assert!(count(last, "unreachable_cells") > 0);
    assert_eq!(quality(last)["met_fraction"], 0.0);
    assert!(
        scan_ends(&frames).is_some(),
        "unreachable cells never hold the scan open"
    );
}

#[test]
fn old_evidence_goes_stale() {
    let bundle = bundle("scenario_bundle_scan.json");
    let fresh = QualityRequirement {
        max_age_s: Some(20.0),
        ..requirement()
    };
    let (_, frames) = replay(run_with(&bundle, with_quality(fresh)));
    let stale = frames
        .iter()
        .map(|f| count(f, "stale_cells"))
        .max()
        .unwrap();
    assert!(stale > 0, "cells met once and not revisited go stale");
}

#[test]
fn a_quality_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle("scenario_bundle_scan.json");
    let config = with_quality(QualityRequirement {
        max_age_s: Some(60.0),
        ..requirement()
    });
    let (_, straight) = replay(run_with(&bundle, config.clone()));
    assert_eq!(straight, replay(run_with(&bundle, config.clone())).1);
    let at = 90usize;
    let mut original = run_with(&bundle, config.clone());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    assert!(
        text.contains("accepted_count"),
        "the evidence is checkpointed"
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
fn an_invalid_requirement_is_refused_before_the_run() {
    let bundle = bundle("scenario_bundle_scan.json");
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            with_quality(QualityRequirement {
                min_ground_density_per_m2: None,
                max_vertical_sigma_m: None,
                max_age_s: None,
                ..requirement()
            }),
        )
        .unwrap_err();
    assert!(error.contains("must name"), "{error}");
}

#[test]
fn a_checkpoint_from_other_quality_tracking_is_refused() {
    let bundle = bundle("scenario_bundle_scan.json");
    let checkpoint_of = |config: ExplorationConfig| {
        let mut run = run_with(&bundle, config);
        for _ in 0..10 {
            run.step().unwrap();
        }
        run.sim.checkpoint().unwrap()
    };
    let without = checkpoint_of(ExplorationConfig::default());
    let at_40 = checkpoint_of(with_quality(requirement()));
    let at_20 = checkpoint_of(with_quality(QualityRequirement {
        max_incidence_deg: 20.0,
        ..requirement()
    }));
    let mut tracked = run_with(&bundle, with_quality(requirement()));
    assert!(tracked.restore(&without).is_err(), "no evidence to resume");
    assert!(tracked.restore(&at_20).is_err(), "another incidence limit");
    assert!(tracked.restore(&at_40).is_ok());
    let mut plain = run_with(&bundle, ExplorationConfig::default());
    assert!(
        plain.restore(&at_40).is_err(),
        "evidence the run cannot use"
    );
}
