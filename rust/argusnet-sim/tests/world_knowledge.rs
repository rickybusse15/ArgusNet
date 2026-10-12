//! Starting from no world knowledge (`world_knowledge: none`).
//!
//! The gates: the run maps the world from its own range returns and the map
//! improves as it flies; it is deterministic, resumes exactly from a
//! checkpoint and resets to a fresh run; physics keeps every platform out of
//! the true ground; the replay's rows rebuild the map; and a run that cannot
//! host the map is refused before it starts.

use std::collections::BTreeMap;

use argusnet_sim::belief_world::{
    BeliefCellUpdate, ExplorationConfig, PursuitSource, WorldKnowledge,
};
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::stage::Stage;

fn bundle(name: &str) -> ScenarioBundle {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn no_prior(bundle: &ScenarioBundle) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, ExplorationConfig::default())
        .expect("a run with stage 8 can map for itself");
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
fn the_map_starts_empty_and_fills_from_the_platforms_own_returns() {
    let bundle = bundle("scenario_bundle_tt.json");
    let frames = frames(no_prior(&bundle));
    let belief = |frame: &serde_json::Value| frame["mapping_state"]["belief"].clone();
    let first = belief(&frames[0]);
    let last = belief(frames.last().unwrap());
    let known = |b: &serde_json::Value| b["known_fraction"].as_f64().unwrap();
    assert!(
        known(&first) < 0.05,
        "almost nothing is known after one scan: {first}"
    );
    assert!(
        known(&last) > 4.0 * known(&first),
        "the map fills as the platforms fly: {first} -> {last}"
    );
    let mut previous = 0.0;
    for frame in &frames {
        let now = known(&belief(frame));
        assert!(now >= previous, "known cells are never forgotten");
        previous = now;
    }
    let rmse = last["ground_rmse_m"].as_f64().unwrap();
    assert!(
        rmse < 3.0,
        "the self-built ground is close to the truth: {last}"
    );
    assert!(frames[0]["range_scans"]
        .as_array()
        .is_some_and(|s| !s.is_empty()));
}

#[test]
fn the_prior_option_is_the_run_it_always_was() {
    let bundle = bundle("scenario_bundle_tt.json");
    let mut run = assemble(&bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::Prior, ExplorationConfig::default())
        .unwrap();
    assert_eq!(frames(run), frames(assemble(&bundle).unwrap()));
}

#[test]
fn a_no_prior_run_is_deterministic_and_differs_from_the_prior_one() {
    let bundle = bundle("scenario_bundle_tt.json");
    let once = frames(no_prior(&bundle));
    assert_eq!(once, frames(no_prior(&bundle)));
    let prior = frames(assemble(&bundle).unwrap());
    assert!(prior[0].get("belief_updates").is_none());
    assert!(once[0].get("belief_updates").is_some());
}

/// Physics keeps every platform at least the clearance above the true
/// ground, whatever the belief said; each correction is recorded.
#[test]
fn no_platform_is_ever_below_the_true_ground() {
    let bundle = bundle("scenario_bundle_scan_safety.json");
    let run = no_prior(&bundle);
    let replay = run.run_to_replay().unwrap();
    let mut contacts = 0;
    for frame in &replay.frames {
        contacts += frame.world_contacts.len();
        for contact in &frame.world_contacts {
            assert!(contact.correction_m > 0.0);
        }
    }
    // The contact count is the measure; this scenario needs none.
    assert_eq!(
        contacts, 0,
        "the belief never led a platform into the ground"
    );
}

#[test]
fn a_checkpoint_resumes_the_map_exactly() {
    let bundle = bundle("scenario_bundle_tt.json");
    let straight = frames(no_prior(&bundle));
    let at = 30usize;
    let mut original = no_prior(&bundle);
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = no_prior(&bundle);
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);

    // A run with a prior map refuses a checkpoint with a self-built one.
    let mut prior = assemble(&bundle).unwrap();
    assert!(prior.restore(&checkpoint).is_err());
}

#[test]
fn a_reset_run_starts_from_nothing_again() {
    let bundle = bundle("scenario_bundle_tt.json");
    let straight = frames(no_prior(&bundle));
    let mut run = no_prior(&bundle);
    for _ in 0..25 {
        run.step().unwrap();
    }
    run.reset();
    assert_eq!(frames(run), straight);
}

/// Applying every frame's rows in order gives each cell its last row.
#[test]
fn the_replay_rows_rebuild_the_map() {
    let bundle = bundle("scenario_bundle_tt.json");
    let replay = no_prior(&bundle).run_to_replay().unwrap();
    let mut map: BTreeMap<(u32, u32), BeliefCellUpdate> = BTreeMap::new();
    for frame in &replay.frames {
        for row in &frame.belief_updates {
            map.insert((row.0, row.1), row.clone());
        }
    }
    let last = replay.frames.last().unwrap();
    let stats = last
        .mapping_state
        .as_ref()
        .and_then(|m| m.belief.clone())
        .unwrap();
    assert_eq!(
        map.len() as u64,
        stats.known_cells,
        "one row per known cell"
    );
    let structures = map.values().filter(|row| row.3.is_some()).count() as u64;
    assert_eq!(structures, stats.structure_cells);
}

#[test]
fn the_built_map_saves_as_a_scene_package_with_the_run_that_built_it() {
    let bundle = bundle("scenario_bundle_tt.json");
    let replay = no_prior(&bundle).run_to_replay().unwrap();
    let dir = std::env::temp_dir().join(format!("wk-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let export = argusnet_sim::scene_export::export_replay_scene(
        &dir,
        "tt-self-built",
        &replay,
        serde_json::json!({"test": true}),
    )
    .expect("a no-prior run exports");
    let stats = replay
        .frames
        .last()
        .and_then(|frame| frame.mapping_state.as_ref())
        .and_then(|m| m.belief.clone())
        .unwrap();
    assert_eq!(export.known_cells as u64, stats.known_cells);
    assert_eq!(export.structure_cells as u64, stats.structure_cells);
    assert!(export.ground_triangles > 0);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("scene_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["replay"]["path"], "replay/replay.json");
    let packaged = std::fs::read_to_string(dir.join("replay/replay.json")).unwrap();
    assert_eq!(packaged, argusnet_sim::replay::to_json(&replay).unwrap());
    let _ = std::fs::remove_dir_all(&dir);

    // A prior run built no map, so there is nothing to save.
    let prior = assemble(&bundle).unwrap().run_to_replay().unwrap();
    let error =
        argusnet_sim::scene_export::export_replay_scene(&dir, "x", &prior, serde_json::json!({}))
            .unwrap_err();
    assert!(error.contains("built no map"), "{error}");
    assert!(!dir.exists());
}

fn with_pursuit(bundle: &ScenarioBundle, pursuit: PursuitSource) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    let config = ExplorationConfig {
        pursuit,
        ..Default::default()
    };
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
        .unwrap();
    run
}

fn stage_one_notes(run: &AssembledRun) -> Vec<String> {
    run.sim
        .trace()
        .records()
        .iter()
        .filter(|record| record.stage == Stage::SampleControllersAndBattery)
        .flat_map(|record| record.notes.clone())
        .collect()
}

#[test]
fn follow_drones_pursue_the_fused_tracks_and_never_read_truth() {
    let bundle = bundle("scenario_bundle_tt.json");
    let mut run = no_prior(&bundle);
    let mut notes = Vec::new();
    for _ in 0..bundle.steps {
        run.step().unwrap();
        notes.extend(stage_one_notes(&run));
    }
    assert!(
        notes.iter().all(|note| !note.contains("legacy pursuit")),
        "stage 1 read truth: {notes:?}"
    );
    let track_reads: u64 = notes
        .iter()
        .filter_map(|note| note.strip_prefix("pursuit read "))
        .filter_map(|rest| rest.split(' ').next()?.parse::<u64>().ok())
        .sum();
    assert!(track_reads > 0, "no pursuit read a fused track: {notes:?}");

    // The legacy oracle is still there when asked for, and it differs.
    let mut truth = with_pursuit(&bundle, PursuitSource::Truth);
    let mut truth_notes = Vec::new();
    for _ in 0..bundle.steps {
        truth.step().unwrap();
        truth_notes.extend(stage_one_notes(&truth));
    }
    assert!(truth_notes
        .iter()
        .any(|note| note.contains("legacy pursuit")));
    assert_ne!(
        frames(no_prior(&bundle)),
        frames(with_pursuit(&bundle, PursuitSource::Truth))
    );
}

#[test]
fn a_checkpoint_does_not_cross_between_pursuit_sources() {
    let bundle = bundle("scenario_bundle_tt.json");
    let mut tracks = no_prior(&bundle);
    for _ in 0..30 {
        tracks.step().unwrap();
    }
    let checkpoint = tracks.sim.checkpoint().unwrap();
    let mut truth = with_pursuit(&bundle, PursuitSource::Truth);
    assert!(truth.restore(&checkpoint).is_err());
}

#[test]
fn a_run_without_stage_eight_or_with_a_bad_config_is_refused() {
    let mut bundle = bundle("scenario_bundle_tt.json");
    let config = ExplorationConfig {
        resolution_m: 0.0,
        ..ExplorationConfig::default()
    };
    let mut run = assemble(&bundle).unwrap();
    assert!(run
        .set_world_knowledge(WorldKnowledge::NoPrior, config)
        .is_err());

    bundle.coverage_bounds = None;
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(WorldKnowledge::NoPrior, ExplorationConfig::default())
        .unwrap_err();
    assert!(error.contains("stage 8"), "{error}");

    let mut started = assemble(&self::bundle("scenario_bundle_tt.json")).unwrap();
    started.step().unwrap();
    assert!(started
        .set_world_knowledge(WorldKnowledge::NoPrior, ExplorationConfig::default())
        .is_err());
}
