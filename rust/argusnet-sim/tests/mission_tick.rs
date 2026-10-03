//! The mission machine driven through the real tick transaction.
//!
//! `mission.rs`'s own tests exercise the state machine directly, which is
//! where the transition rules belong. This file checks the other half: that
//! stage 9 is wired into the pipeline P4.1 built, that a mission failure
//! faults the run like any other stage failure, and that the trace carries
//! enough to tell what the mission did on which tick.

use argusnet_sim::config::SimulationConfig;
use argusnet_sim::mission::{
    MissionConfig, MissionEvent, MissionInputs, MissionPhase, MissionStage,
};
use argusnet_sim::simulation::{Simulation, TickError};
use argusnet_sim::stage::{Stage, StageOutcome};

fn simulation() -> Simulation {
    Simulation::bare(SimulationConfig {
        seed: 7,
        dt_s: 0.5,
        steps: Some(64),
        run_id: "mission".to_string(),
    })
    .expect("config is valid")
}

/// Inputs for a fleet of `n`, at `t`, that has finished scanning.
fn scan_done(t: f64, n: usize, confidence: f64) -> MissionInputs {
    MissionInputs {
        timestamp_s: t,
        coverage_fraction: 0.9,
        gap_cell_fraction: Some(0.0),
        mobile_count: n,
        localization_confidences: vec![confidence; n],
        localizer_timed_out: false,
        all_pois_complete: false,
        home_distances_m: vec![Some(900.0); n],
    }
}

/// Stage 9's machine, reached through the session that owns it.
fn machine(sim: &Simulation) -> &MissionStage {
    MissionStage::of(sim).expect("stage 9 is a mission machine")
}

fn machine_mut(sim: &mut Simulation) -> &mut MissionStage {
    MissionStage::of_mut(sim).expect("stage 9 is a mission machine")
}

#[test]
fn a_mission_runs_to_complete_through_the_tick_pipeline() {
    let mut sim = simulation();
    let stage = MissionStage::new(MissionConfig::default());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .expect("stage 9 is not a transaction boundary");

    assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);

    let mut phases = vec![machine(&sim).phase()];
    let mut all_events: Vec<MissionEvent> = Vec::new();
    for _ in 0..6 {
        // Stages run at the clock's *current* tick and only then advance, so
        // the first step runs at t = 0 -- matching `sim.py`'s `step * dt_s`
        // with `step` starting at zero.
        let t = sim.tick() as f64 * 0.5;
        let mut inputs = scan_done(t, 2, 0.99);
        inputs.all_pois_complete = true;
        inputs.home_distances_m = vec![Some(0.0), Some(1.0)];
        machine_mut(&mut sim).set_inputs(inputs);

        sim.step().expect("no stage should fail");
        all_events.extend(machine_mut(&mut sim).take_events());
        if machine(&sim).phase() != *phases.last().unwrap() {
            phases.push(machine(&sim).phase());
        }
    }

    // One transition per tick, even with every gate open from the first.
    assert_eq!(
        phases,
        vec![
            MissionPhase::Scanning,
            MissionPhase::Inspecting,
            MissionPhase::Egress,
            MissionPhase::Complete,
        ]
    );
    assert!(all_events.contains(&MissionEvent::RescorePois { at_s: 0.0 }));
    assert!(all_events.contains(&MissionEvent::EgressRoutesRequested { at_s: 0.5 }));
    assert!(all_events.contains(&MissionEvent::EgressRoutesCleared { at_s: 1.0 }));
}

/// The trace is what makes "did the port preserve the order?" answerable, so
/// stage 9's own record has to distinguish a tick that transitioned from one
/// that did not.
#[test]
fn the_trace_distinguishes_a_transition_from_a_quiet_tick() {
    let mut sim = simulation();
    let stage = MissionStage::new(MissionConfig::default());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .unwrap();

    // Tick 0 (t = 0): nothing has been scanned, so the machine holds.
    machine_mut(&mut sim).set_inputs(MissionInputs::at(0.0));
    sim.step().unwrap();
    assert!(machine_mut(&mut sim).take_events().is_empty());

    // Tick 1 (t = 0.5): scanning completes.
    machine_mut(&mut sim).set_inputs(scan_done(0.5, 1, 0.99));
    sim.step().unwrap();
    assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);

    let records: Vec<_> = sim
        .trace()
        .records()
        .iter()
        .filter(|r| r.stage == Stage::EvaluateMissionTransitions)
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].outcome, StageOutcome::NoWork);
    assert_eq!(records[0].tick, 0);
    assert_eq!(records[1].outcome, StageOutcome::Ran);
    assert_eq!(records[1].tick, 1);
    assert!(
        records[1]
            .notes
            .iter()
            .any(|n| n.contains("mission phase scanning -> inspecting at 0.5")),
        "notes were {:?}",
        records[1].notes
    );
}

/// A disabled machine is `Skipped`, which is a different trace entry from
/// `NoWork` — "this run has no mission" and "the mission had nothing to do"
/// are different facts.
#[test]
fn a_disabled_mission_is_skipped_not_quiet() {
    let mut sim = simulation();
    let stage = MissionStage::disabled();
    assert!(!stage.is_enabled());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .unwrap();

    // No inputs are supplied at all, which an enabled stage would refuse.
    sim.step().unwrap();
    sim.step().unwrap();
    assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);

    let records: Vec<_> = sim
        .trace()
        .records()
        .iter()
        .filter(|r| r.stage == Stage::EvaluateMissionTransitions)
        .collect();
    assert_eq!(records.len(), 2);
    assert!(matches!(records[0].outcome, StageOutcome::Skipped { .. }));
}

/// An enabled stage with no inputs is a wiring bug, and §4 says a mid-tick
/// failure publishes nothing and faults the run.
#[test]
fn an_enabled_mission_with_no_inputs_faults_the_run() {
    let mut sim = simulation();
    let stage = MissionStage::new(MissionConfig::default());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .unwrap();

    let error = sim.step().expect_err("no inputs were supplied");
    assert!(matches!(
        error,
        TickError::Stage {
            stage: Stage::EvaluateMissionTransitions,
            ..
        }
    ));
    assert!(sim.is_faulted());
    assert!(sim.latest_snapshot().is_none(), "no partial snapshot");
    // A faulted run refuses to step until someone decides what to do.
    assert!(matches!(sim.step(), Err(TickError::Faulted { .. })));
}

/// `phase_started_at_s` is taken from the inputs, so inputs stamped for a
/// different tick would date a transition wrongly. That is refused rather
/// than silently accepted.
#[test]
fn inputs_stamped_for_another_tick_are_refused() {
    let mut sim = simulation();
    let stage = MissionStage::new(MissionConfig::default());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .unwrap();

    // Tick 0 runs at t = 0; these are stamped for t = 3.0.
    machine_mut(&mut sim).set_inputs(scan_done(3.0, 1, 0.99));
    let error = sim.step().expect_err("stale inputs");
    assert!(matches!(
        error,
        TickError::Stage {
            stage: Stage::EvaluateMissionTransitions,
            ..
        }
    ));
    assert_eq!(
        machine(&sim).phase(),
        MissionPhase::Scanning,
        "a refused tick must not have advanced the machine"
    );
}

/// Events are consumed once. A caller that takes them twice gets them once,
/// which is the guarantee P4.8 asks for.
#[test]
fn events_are_taken_once() {
    let mut sim = simulation();
    let stage = MissionStage::new(MissionConfig::default());
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
        .unwrap();

    machine_mut(&mut sim).set_inputs(scan_done(0.0, 1, 0.99));
    sim.step().unwrap();
    let first = machine_mut(&mut sim).take_events();
    assert!(!first.is_empty());
    assert!(machine_mut(&mut sim).take_events().is_empty());

    // And the next tick's run clears what the previous one left, so a quiet
    // tick cannot hand back stale events.
    machine_mut(&mut sim).set_inputs(MissionInputs::at(0.5));
    sim.step().unwrap();
    assert!(machine_mut(&mut sim).take_events().is_empty());
}
