//! The tick's shared world, and the two properties P4.8 asks of it.
//!
//! > *"Eliminate repeated dataclass-style list rebuilds; freeze once at the
//! > snapshot boundary."*
//!
//! and, from `ARCHITECTURE_AND_PROTOCOLS.md` §3, truth as a capability rather
//! than a convention. Both are asserted here against the real pipeline, because
//! both are properties of how the stages are *driven*, not of any one stage.

use std::sync::{Arc, Mutex};

use argusnet_sim::config::SimulationConfig;
use argusnet_sim::simulation::{Simulation, StageHandler, TickContext};
use argusnet_sim::stage::{Stage, StageError, StageOutcome};
use argusnet_types::{NodeState, TruthState, Vector3};

fn simulation() -> Simulation {
    Simulation::new(SimulationConfig {
        seed: 3,
        dt_s: 0.25,
        steps: Some(32),
        run_id: "world".to_string(),
    })
    .expect("config is valid")
}

fn node(id: &str, mobile: bool) -> NodeState {
    NodeState {
        node_id: id.to_string(),
        position: Vector3::new(0.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        is_mobile: mobile,
        timestamp_s: 0.0,
        health: 1.0,
        ..Default::default()
    }
}

fn target(id: &str, x: f64) -> TruthState {
    TruthState {
        target_id: id.to_string(),
        position: Vector3::new(x, 0.0, 0.0),
        velocity: Vector3::new(1.0, 0.0, 0.0),
        timestamp_s: 0.0,
    }
}

/// Records whether it was handed truth, and nudges a platform so the world
/// visibly moves.
#[derive(Debug)]
struct Probe {
    saw_truth: Arc<Mutex<Vec<(Stage, bool, usize)>>>,
    stage: Stage,
    nudge: Option<&'static str>,
}

impl StageHandler for Probe {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        self.saw_truth.lock().unwrap().push((
            self.stage,
            context.truth.is_some(),
            context.truth.map(<[TruthState]>::len).unwrap_or(0),
        ));
        if let Some(id) = self.nudge {
            let platform = context
                .world
                .platform_mut(id)
                .ok_or_else(|| StageError::Invariant(format!("no platform {id}")))?;
            platform.position.x += 1.0;
        }
        Ok(StageOutcome::Ran)
    }
}

fn install_probes(sim: &mut Simulation) -> Arc<Mutex<Vec<(Stage, bool, usize)>>> {
    let log = Arc::new(Mutex::new(Vec::new()));
    for stage in Stage::PIPELINE {
        if matches!(stage, Stage::ApplyCommands | Stage::CommitSnapshot) {
            continue;
        }
        sim.set_stage_handler(
            stage,
            Box::new(Probe {
                saw_truth: Arc::clone(&log),
                stage,
                nudge: None,
            }),
        )
        .expect("not a transaction boundary");
    }
    log
}

// ---------------------------------------------------------------------
// Truth is a capability
// ---------------------------------------------------------------------

/// Exactly two stages are handed truth, and they are the two that exist to
/// sense the true world. Every other handler sees `None` — not an empty slice,
/// which would be indistinguishable from "there are no targets".
#[test]
fn only_the_sensing_stages_are_handed_truth() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    sim.set_truth(vec![target("t-1", 10.0), target("t-2", 20.0)]);
    let log = install_probes(&mut sim);

    sim.step().expect("no stage fails");

    let saw: Vec<(Stage, bool, usize)> = log.lock().unwrap().clone();
    let with_truth: Vec<Stage> = saw
        .iter()
        .filter(|(_, had, _)| *had)
        .map(|(stage, _, _)| *stage)
        .collect();
    assert_eq!(
        with_truth,
        vec![Stage::UpdateVioAndPose, Stage::GenerateObservations]
    );
    // And they got the real targets, not an empty view.
    for (stage, had, count) in &saw {
        if *had {
            assert_eq!(*count, 2, "{stage} was handed an empty truth view");
        }
    }
    // Every stage in the pipeline ran and reported.
    assert_eq!(saw.len(), Stage::PIPELINE.len() - 2);
}

/// The capability is per stage, not per run: a truth-blind stage sees `None`
/// even when truth is loaded and other stages are reading it.
#[test]
fn a_truth_blind_stage_sees_none_even_while_truth_exists() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    sim.set_truth(vec![target("t-1", 5.0)]);
    let log = install_probes(&mut sim);

    sim.step().unwrap();

    let saw = log.lock().unwrap().clone();
    for (stage, had, _) in saw {
        assert_eq!(
            had,
            stage.may_read_truth(),
            "{stage} disagreed with its own capability"
        );
    }
}

/// Truth never reaches a snapshot, and the header says so rather than leaving
/// a consumer to infer "no truth available" from "no truth present".
#[test]
fn truth_never_reaches_a_snapshot() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    sim.set_truth(vec![target("t-1", 5.0), target("t-2", 6.0)]);

    let snapshot = sim.step().unwrap();
    assert!(snapshot.header.truth_filtered);
    assert_eq!(snapshot.platforms.len(), 1);
    assert!(snapshot.tracks.is_empty());
}

// ---------------------------------------------------------------------
// One list, frozen once
// ---------------------------------------------------------------------

/// The property that makes this worth doing: every stage in the pipeline edits
/// the same platform and the list is cloned exactly **once**, at stage 10. The
/// reference rebuilds it up to seven times per tick.
#[test]
fn nine_stages_edit_one_list_and_it_is_frozen_once() {
    let mut sim = simulation();
    sim.load_world(
        vec![node("drone-1", true), node("tower", false)],
        Vec::new(),
    );
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut editing = 0usize;
    for stage in Stage::PIPELINE {
        if matches!(stage, Stage::ApplyCommands | Stage::CommitSnapshot) {
            continue;
        }
        editing += 1;
        sim.set_stage_handler(
            stage,
            Box::new(Probe {
                saw_truth: Arc::clone(&log),
                stage,
                nudge: Some("drone-1"),
            }),
        )
        .unwrap();
    }
    assert_eq!(editing, 9, "stages 1-9 all edit");

    let snapshot = sim.step().unwrap();

    assert_eq!(
        sim.world().freezes_this_tick(),
        1,
        "the tick's state must be cloned exactly once, at the snapshot boundary"
    );
    // Nine edits of one metre, on one list, with the other platform untouched.
    assert_eq!(snapshot.platforms[0].position.x, 9.0);
    assert_eq!(snapshot.platforms[1].position.x, 0.0);
    assert_eq!(snapshot.platforms.len(), 2);
}

/// The counter resets per tick, so "once" means once *per tick* rather than
/// once per run.
#[test]
fn the_freeze_count_is_per_tick() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    for _ in 0..5 {
        sim.step().unwrap();
        assert_eq!(sim.world().freezes_this_tick(), 1);
    }
}

/// A published snapshot is immutable: later ticks move the world and must not
/// move a snapshot someone is already holding.
#[test]
fn a_published_snapshot_does_not_move_when_later_ticks_do() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    let log = Arc::new(Mutex::new(Vec::new()));
    sim.set_stage_handler(
        Stage::SampleControllersAndBattery,
        Box::new(Probe {
            saw_truth: log,
            stage: Stage::SampleControllersAndBattery,
            nudge: Some("drone-1"),
        }),
    )
    .unwrap();

    let first = sim.step().unwrap();
    assert_eq!(first.platforms[0].position.x, 1.0);
    for _ in 0..3 {
        sim.step().unwrap();
    }
    assert_eq!(first.platforms[0].position.x, 1.0, "still the tick-0 value");
    assert_eq!(sim.world().platforms()[0].position.x, 4.0);
}

/// `world_revision` advances with the edits, so a consumer can tell two
/// snapshots apart without diffing their contents.
#[test]
fn the_world_revision_tracks_mutation() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    let quiet = sim.step().unwrap();
    assert_eq!(quiet.header.world_revision, 0, "no stage edited anything");

    let log = Arc::new(Mutex::new(Vec::new()));
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(Probe {
            saw_truth: log,
            stage: Stage::Deconflict,
            nudge: Some("drone-1"),
        }),
    )
    .unwrap();
    let moved = sim.step().unwrap();
    assert!(moved.header.world_revision > quiet.header.world_revision);
}

// ---------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------

/// A reset is a new epoch, so the previous epoch's platforms are not this
/// run's. Leaving stale truth behind would let a fresh run sense the old world
/// for one tick.
#[test]
fn a_reset_clears_the_world_and_the_truth() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    sim.set_truth(vec![target("t-1", 1.0)]);
    let log = install_probes(&mut sim);
    sim.step().unwrap();
    assert!(!sim.world().platforms().is_empty());

    sim.reset();
    assert!(sim.world().platforms().is_empty());

    log.lock().unwrap().clear();
    sim.step().unwrap();
    let saw = log.lock().unwrap().clone();
    for (stage, had, count) in saw {
        assert_eq!(had, stage.may_read_truth());
        assert_eq!(count, 0, "{stage} saw truth from the previous epoch");
    }
}

/// A stage naming a platform that is not in the fleet is a bug, and it faults
/// the run rather than silently creating one.
#[test]
fn a_stage_naming_an_unknown_platform_faults_the_run() {
    let mut sim = simulation();
    sim.load_world(vec![node("drone-1", true)], Vec::new());
    let log = Arc::new(Mutex::new(Vec::new()));
    sim.set_stage_handler(
        Stage::SafetyMonitor,
        Box::new(Probe {
            saw_truth: log,
            stage: Stage::SafetyMonitor,
            nudge: Some("ghost"),
        }),
    )
    .unwrap();

    assert!(sim.step().is_err());
    assert!(sim.is_faulted());
    assert!(sim.latest_snapshot().is_none(), "no partial snapshot");
}
