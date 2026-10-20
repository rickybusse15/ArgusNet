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
fn search_lanes_become_explorers_and_resume_exactly() {
    let bundle = bundle("scenario_bundle_scan_safety.json");
    let explore = no_prior(&bundle);
    assert!(explore.controllers().explorer_count() > 0);
    let mut lanes = assemble(&bundle).unwrap();
    lanes
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                explore: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(lanes.controllers().explorer_count(), 0);
    let explored = frames(no_prior(&bundle));
    assert_ne!(explored, frames(lanes));
    for frame in &explored {
        let contacts = frame.get("world_contacts").and_then(|c| c.as_array());
        assert!(
            contacts.is_none_or(|c| c.is_empty()),
            "an explorer hit the world"
        );
    }

    // A checkpoint taken mid-flight resumes exactly: goals and positions are
    // checkpointed, claims are rebuilt.
    let at = 40usize;
    let mut original = no_prior(&bundle);
    for _ in 0..at {
        original.step().unwrap();
    }
    let checkpoint = original.sim.checkpoint().unwrap();
    let mut resumed = no_prior(&bundle);
    resumed.restore(&checkpoint).unwrap();
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, explored[at..]);
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

/// R2: in a world with trees the scanner's canopy returns reach the
/// belief and the replay (canopy rows, range-scan canopy points, the count
/// of canopy cells), trunk returns count as right structure, and the run
/// still resumes exactly from a checkpoint.
#[test]
fn a_forest_maps_as_canopy_and_resumes_exactly() {
    let config: argusnet_sim::builder::ScenarioConfig = serde_json::from_value(serde_json::json!({
        "format": "argusnet.scenario-config",
        "version": 1,
        "duration_s": 40.0,
        "options": {"map_preset": "small", "terrain_preset": "jungle_canopy", "trees": true},
    }))
    .unwrap();
    let built = argusnet_sim::builder::build_bundle(&config).unwrap();
    let bundle = ScenarioBundle::from_json(&built.to_string()).unwrap();
    let frames = frames(no_prior(&bundle));
    let rows: usize = frames
        .iter()
        .map(|f| f["canopy_updates"].as_array().map_or(0, Vec::len))
        .sum();
    let points: usize = frames
        .iter()
        .flat_map(|f| f["range_scans"].as_array().cloned().unwrap_or_default())
        .map(|s| s["canopy"].as_array().map_or(0, Vec::len))
        .sum();
    let last = &frames.last().unwrap()["mapping_state"]["belief"];
    assert!(rows > 0 && points > 0, "{rows} rows, {points} points");
    assert!(last["canopy_cells"].as_u64().unwrap() > 0, "{last}");
    // No building or wall stands on the jungle map: every believed
    // structure is a trunk, and right.
    assert!(last["structure_cells"].as_u64().unwrap() > 0, "{last}");
    assert_eq!(last["structure_precision"].as_f64(), Some(1.0), "{last}");

    let straight = frames;
    let at = straight.len() / 2;
    let mut original = no_prior(&bundle);
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    assert!(text.contains("canopy_top_m"), "the canopy is checkpointed");
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
}

/// R3: a wire is thin enough to fly through between two ticks, each
/// position clear of it. Strung square across a drone's own path at its
/// height, at the middle of a tick's flight, it stops the drone short of
/// it (a recorded contact) instead of being flown through: no flight
/// between two published positions passes through the wire.
#[test]
fn a_platform_never_flies_through_a_wire() {
    let fixture = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&fixture).expect("fixture");
    let path_of = |frames: &[serde_json::Value], id: &str| -> Vec<[f64; 3]> {
        frames
            .iter()
            .map(|frame| {
                let node = frame["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|n| n["node_id"] == id)
                    .unwrap();
                let p = &node["position"];
                [
                    p[0].as_f64().unwrap(),
                    p[1].as_f64().unwrap(),
                    p[2].as_f64().unwrap(),
                ]
            })
            .collect()
    };
    let without = frames(no_prior(&ScenarioBundle::from_json(&text).unwrap()));
    let path = path_of(&without, "drone-east");
    // The first tick it flies 10 m or more nearly level, after launch.
    let k = (5..path.len() - 1)
        .find(|&k| {
            let (a, b) = (path[k], path[k + 1]);
            (b[0] - a[0]).hypot(b[1] - a[1]) >= 10.0 && (b[2] - a[2]).abs() < 2.0
        })
        .expect("a level stretch");
    let (a, b) = (path[k], path[k + 1]);
    let mid = [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ];
    let along = [(b[0] - a[0]), (b[1] - a[1])];
    let norm = along[0].hypot(along[1]);
    let across = [-along[1] / norm, along[0] / norm];
    let support = |s: f64| {
        serde_json::json!({
            "xy_m": [mid[0] + across[0] * s, mid[1] + across[1] * s],
            "base_z_m": mid[2] - 20.0,
            "height_m": 20.0,
        })
    };
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "across",
        "supports": [support(-100.0), support(100.0)],
        "conductor_offsets_m": [0.0],
        "sag_ratio": 0.0,
        "conductor_radius_m": 0.02,
        "clearance_m": 10.0
    }]);
    let bundle = ScenarioBundle::from_json(&document.to_string()).unwrap();
    let line = bundle_line(&document);
    let replay = no_prior(&bundle).run_to_replay().unwrap();
    let with: Vec<serde_json::Value> = replay
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect();
    assert!(
        replay
            .frames
            .iter()
            .flat_map(|frame| &frame.world_contacts)
            .any(|contact| contact.node_id == "drone-east" && contact.kind == "structure"),
        "the wire stopped it"
    );
    for id in ["drone-east", "drone-north"] {
        let path = path_of(&with, id);
        for (tick, pair) in path.windows(2).enumerate() {
            let kept = line.clearance_along(pair[0], pair[1]);
            assert!(kept >= 0.0, "{id} through the wire at tick {tick}: {kept}");
        }
    }
}

fn bundle_line(document: &serde_json::Value) -> argusnet_world::powerline::PowerLine {
    serde_json::from_value(document["world"]["power_lines"][0].clone()).unwrap()
}
