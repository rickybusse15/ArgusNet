//! A resumed run is the same run -- for a whole assembled scenario.
//!
//! P4.8d.3 proved checkpoint/resume for stage 1 alone. P4.8e.4 puts every
//! stateful stage on the contract (stage 6's bias walks, stage 7's tracker,
//! stage 8's coverage and localizer, stage 2's redirects, stage 9's mission)
//! and this is the test that says it is complete: a scenario bundle is flown
//! straight through; then flown again, checkpointed at an interesting tick,
//! serialised to JSON text, and restored into a *freshly assembled* run that
//! was first sent down a wrong history. Every later tick must match bit for
//! bit -- node kinematics, health and battery; every fused track; every
//! observation's bearing (which carries the bias walk); and in scan mode the
//! mission phase and coverage.
//!
//! The Rust run is fully deterministic (ADR-006 keyed streams), so unlike the
//! Python comparisons nothing here is statistical: any difference at all is
//! state the checkpoint failed to carry.

use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::{RestoreError, SimulationCheckpoint};

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

/// One tick's observable output, as bits.
#[derive(Debug, Clone, PartialEq)]
struct Frame {
    tick: u64,
    platforms: Vec<(String, Vec<u64>)>,
    tracks: Vec<(String, Vec<u64>)>,
    observations: Vec<(String, String, Vec<u64>)>,
    phase: Option<String>,
    coverage: Option<u64>,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn step(run: &mut AssembledRun) -> Frame {
    let snapshot = run.step().expect("tick runs");
    let frame = run.fusion.frame().expect("stage 7 ran");
    let mut tracks: Vec<(String, Vec<u64>)> = frame
        .tracks
        .iter()
        .map(|t| {
            (
                t.track_id.clone(),
                bits(&[
                    t.position.x,
                    t.position.y,
                    t.position.z,
                    t.velocity.x,
                    t.velocity.y,
                    t.velocity.z,
                ]),
            )
        })
        .collect();
    tracks.sort();
    Frame {
        tick: snapshot.header.tick,
        platforms: snapshot
            .platforms
            .iter()
            .map(|p| {
                (
                    p.node_id.clone(),
                    bits(&[
                        p.position.x,
                        p.position.y,
                        p.position.z,
                        p.velocity.x,
                        p.velocity.y,
                        p.velocity.z,
                        p.health,
                        p.battery_fraction.unwrap_or(-1.0),
                    ]),
                )
            })
            .collect(),
        tracks,
        observations: frame
            .observations
            .iter()
            .map(|o| {
                (
                    o.node_id.clone(),
                    o.target_id.clone(),
                    bits(&[
                        o.direction.x,
                        o.direction.y,
                        o.direction.z,
                        o.bearing_std_rad,
                    ]),
                )
            })
            .collect(),
        phase: run.mission.as_ref().map(|m| m.phase().to_string()),
        coverage: run
            .mapping
            .as_ref()
            .map(|m| m.coverage_fraction().to_bits()),
    }
}

fn straight(case: &str, ticks: u64) -> Vec<Frame> {
    let mut run = assemble(&bundle(case)).expect("assembles");
    (0..ticks).map(|_| step(&mut run)).collect()
}

/// Checkpoint at `at`, restore into a stranger, fly on, compare.
fn resume_matches(case: &str, reference: &[Frame], at: u64) {
    let mut original = assemble(&bundle(case)).expect("assembles");
    for _ in 0..at {
        step(&mut original);
    }
    let text = serde_json::to_string(&original.sim.checkpoint().expect("not faulted"))
        .expect("checkpoint serialises");
    drop(original);

    let mut resumed = assemble(&bundle(case)).expect("assembles");
    for _ in 0..(at / 2 + 5) {
        step(&mut resumed);
    }
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).expect("parses");
    resumed.sim.restore(&checkpoint).expect("restore accepted");
    // Every stage took ALL of its state back: checkpointing straight away
    // reproduces what was restored, field for field. This sees a dropped
    // field even when this scenario never makes it decisive.
    let again = resumed.sim.checkpoint().expect("not faulted");
    for (stage, state) in &checkpoint.stage_states {
        assert_eq!(
            again.stage_states.get(stage),
            Some(state),
            "{case}: stage {stage} did not take all of its state back at tick {at}"
        );
    }
    for expected in &reference[at as usize..] {
        let actual = step(&mut resumed);
        assert_eq!(
            &actual, expected,
            "{case}: resumed from tick {at}, diverged at tick {}",
            expected.tick
        );
    }
}

fn phase_changes(frames: &[Frame]) -> Vec<(u64, String)> {
    let mut changes = Vec::new();
    let mut last: Option<&String> = None;
    for frame in frames {
        if let Some(phase) = &frame.phase {
            if last != Some(phase) {
                changes.push((frame.tick, phase.clone()));
                last = Some(phase);
            }
        }
    }
    changes
}

/// Target tracking: engagement, fusion tracks, bias walks, deconfliction.
#[test]
fn a_target_tracking_run_resumes_exactly() {
    let reference = straight("tt", 80);
    assert!(
        reference.iter().any(|f| !f.tracks.is_empty()),
        "no tracks were ever fused; the tracker's state is untested"
    );
    for at in [3, 12, 40, 70] {
        resume_matches("tt", &reference, at);
    }
}

/// The default mission, checkpointed in every phase it passes through --
/// including mid-inspection, with POIs assigned and redirects in flight.
#[test]
fn a_scan_map_inspect_run_resumes_exactly_in_every_phase() {
    let reference = straight("scan", 400);
    let changes = phase_changes(&reference);
    let phases: Vec<&str> = changes.iter().map(|(_, p)| p.as_str()).collect();
    for phase in ["scanning", "inspecting", "egress", "complete"] {
        assert!(
            phases.contains(&phase),
            "never reached {phase}: {changes:?}"
        );
    }
    // A few ticks into each phase, so each checkpoint sits inside one.
    let mut points: Vec<u64> = changes
        .iter()
        .map(|(tick, _)| tick + 3)
        .filter(|tick| *tick < 395)
        .collect();
    points.insert(0, 10);
    for at in points {
        resume_matches("scan", &reference, at);
    }
}

/// A reset run is a fresh run, for a whole assembled scenario.
#[test]
fn a_reset_assembled_run_is_a_fresh_run() {
    let reference = straight("tt", 50);
    let mut run = assemble(&bundle("tt")).expect("assembles");
    for _ in 0..30 {
        step(&mut run);
    }
    run.sim.reset();
    // `reset` clears the world; the driver reloads it, as assembly did.
    let fresh = assemble(&bundle("tt")).expect("assembles");
    run.sim
        .load_world(fresh.sim.world().platforms().to_vec(), Vec::new());
    for expected in &reference {
        let actual = step(&mut run);
        assert_eq!(
            (&actual.platforms, &actual.tracks, &actual.observations),
            (
                &expected.platforms,
                &expected.tracks,
                &expected.observations
            ),
            "a reset run diverged from a fresh one at tick {}",
            expected.tick
        );
    }
}

/// A checkpoint that disagrees with the run it is restored into is refused --
/// before anything is mutated -- for each stage's own consistency check. The
/// checkpoint is a real one, tampered with in one field at a time.
#[test]
fn a_checkpoint_for_a_different_scenario_is_refused_stage_by_stage() {
    let case = "scan";
    let mut source = assemble(&bundle(case)).expect("assembles");
    // Deep enough that POIs, overrides and coverage all hold something.
    for _ in 0..120 {
        step(&mut source);
    }
    let original = serde_json::to_value(source.sim.checkpoint().expect("not faulted")).unwrap();

    type Tamper = fn(&mut serde_json::Value);
    let tampers: [(&str, &str, Tamper); 4] = [
        (
            "generate_observations",
            "a sensor the run does not have",
            |state| {
                let map = state.as_object_mut().unwrap();
                let first = map.keys().next().unwrap().clone();
                let walk = map.remove(&first).unwrap();
                map.insert("ghost-sensor".into(), walk);
            },
        ),
        (
            "update_fusion",
            "a differently configured tracker",
            |state| {
                state["config"]["max_stale_steps"] = serde_json::json!(999);
            },
        ),
        (
            "update_mapping_and_localization",
            "a different coverage grid",
            |state| {
                state["coverage"]["bounds"]["resolution_m"] = serde_json::json!(25.0);
            },
        ),
        ("evaluate_mission_transitions", "different POIs", |state| {
            state["pois"]["pois"][0]["poi_id"] = serde_json::json!("poi-99");
        }),
    ];
    for (stage, what, tamper) in tampers {
        let mut tampered = original.clone();
        tamper(
            tampered["stage_states"]
                .get_mut(stage)
                .unwrap_or_else(|| panic!("no state for {stage}")),
        );
        let checkpoint: SimulationCheckpoint = serde_json::from_value(tampered).expect("shape");

        let mut target = assemble(&bundle(case)).expect("assembles");
        for _ in 0..7 {
            step(&mut target);
        }
        let before = target.sim.checkpoint().expect("not faulted");
        let refusal = target.sim.restore(&checkpoint).expect_err(what);
        assert!(
            matches!(&refusal, RestoreError::StageState { stage: s, .. } if s == stage),
            "{what}: refused for the wrong reason: {refusal}"
        );
        assert_eq!(
            target.sim.checkpoint().expect("not faulted"),
            before,
            "{what}: a refused restore changed the run"
        );
    }
}
