//! B2.4: a lane at its own rate (`scheduled_v1`).
//!
//! The mission lane (stage 9) runs at the first legacy boundary at or after
//! each of its deadlines, sits out the rest -- recorded as skipped, never
//! left out -- and is given the time since it last ran. A checkpoint carries
//! how long it has been sitting out, so a resumed run is the uninterrupted one.

use std::num::NonZeroU64;

use argusnet_sim::config::SimulationConfig;
use argusnet_sim::scheduler::{ExecutorError, PhasedLaneExecutor, Schedule};
use argusnet_sim::simulation::{Simulation, StageHandler, TickContext, TickError};
use argusnet_sim::stage::{Stage, StageError, StageOutcome};

fn simulation() -> Simulation {
    Simulation::bare(SimulationConfig {
        seed: 42,
        dt_s: 0.1,
        steps: Some(100),
        run_id: "rates".to_string(),
    })
    .expect("config is valid")
}

/// Stage 9 that records the ticks it ran with the time it was given, and
/// declares it has no per-tick output to clear when not due.
#[derive(Debug, Default)]
struct ClockStage {
    ran: Vec<(u64, f64)>,
}

impl StageHandler for ClockStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        self.ran.push((context.tick, context.dt_s));
        Ok(StageOutcome::Ran)
    }
    fn not_due(&mut self) -> Result<(), StageError> {
        Ok(())
    }
}

fn hz(num: u64, den: u64) -> Schedule {
    Schedule::rational(NonZeroU64::new(num).unwrap(), NonZeroU64::new(den).unwrap()).unwrap()
}

fn clocked(sim: &mut Simulation) {
    sim.set_stage_handler(
        Stage::EvaluateMissionTransitions,
        Box::new(ClockStage::default()),
    )
    .expect("stage 9 is delegable");
}

fn ran(sim: &Simulation) -> Vec<(u64, f64)> {
    sim.stage::<ClockStage>(Stage::EvaluateMissionTransitions)
        .expect("installed")
        .ran
        .clone()
}

#[test]
fn a_lane_at_its_own_rate_runs_at_its_deadlines_with_the_time_since_it_last_ran() {
    let mut sim = simulation();
    clocked(&mut sim);
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0)
        .unwrap()
        .with_mission_rate(hz(3, 1))
        .unwrap();
    for _ in 0..20 {
        executor.step(&mut sim).expect("tick");
    }
    // 3 Hz against 10 Hz boundaries: deadlines at 0, 1/3, 2/3, 1, 4/3, 5/3 s
    // are served by the first boundary at or after each.
    let ticks: Vec<u64> = ran(&sim).iter().map(|(tick, _)| *tick).collect();
    assert_eq!(ticks, [0, 4, 7, 10, 14, 17]);
    let mut previous = None;
    for (tick, dt_s) in ran(&sim) {
        let boundaries = previous.map_or(1, |last| tick - last);
        assert_eq!(dt_s, 0.1 * boundaries as f64, "tick {tick}");
        previous = Some(tick);
    }
    // Every boundary it sat out is in the trace, with the reason.
    let trace = sim.trace().to_json().unwrap();
    let skipped = trace
        .matches("not due: lane at 3/1 Hz (scheduled_v1)")
        .count();
    assert_eq!(skipped, 20 - 6);
}

#[test]
fn a_resumed_run_sits_out_and_runs_the_same_boundaries_as_an_uninterrupted_one() {
    let mut straight = simulation();
    clocked(&mut straight);
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0)
        .unwrap()
        .with_mission_rate(hz(3, 1))
        .unwrap();
    for _ in 0..20 {
        executor.step(&mut straight).expect("tick");
    }

    // Stopped at tick 9: stage 9 last ran at tick 7 and sat out tick 8, so
    // the checkpoint has to say so.
    let mut first = simulation();
    clocked(&mut first);
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0)
        .unwrap()
        .with_mission_rate(hz(3, 1))
        .unwrap();
    for _ in 0..9 {
        executor.step(&mut first).expect("tick");
    }
    let checkpoint = first.checkpoint().expect("not faulted");
    assert_eq!(
        checkpoint.stage_lag.get("evaluate_mission_transitions"),
        Some(&1)
    );

    let mut resumed = simulation();
    clocked(&mut resumed);
    resumed.restore(&checkpoint).expect("compatible");
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, resumed.tick())
        .unwrap()
        .with_mission_rate(hz(3, 1))
        .unwrap();
    for _ in 9..20 {
        executor.step(&mut resumed).expect("tick");
    }
    let after: Vec<(u64, u64)> = ran(&straight)
        .into_iter()
        .filter(|(tick, _)| *tick >= 9)
        .map(|(tick, dt)| (tick, dt.to_bits()))
        .collect();
    let resumed: Vec<(u64, u64)> = ran(&resumed)
        .into_iter()
        .map(|(tick, dt)| (tick, dt.to_bits()))
        .collect();
    assert_eq!(resumed, after);
}

#[test]
fn every_boundary_runs_without_a_rate_and_a_checkpoint_carries_no_lag() {
    let mut sim = simulation();
    clocked(&mut sim);
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0).unwrap();
    for _ in 0..5 {
        executor.step(&mut sim).expect("tick");
    }
    assert_eq!(ran(&sim).len(), 5);
    assert!(ran(&sim)
        .iter()
        .all(|(_, dt)| dt.to_bits() == 0.1f64.to_bits()));
    let checkpoint = sim.checkpoint().unwrap();
    assert!(checkpoint.stage_lag.is_empty());
    assert!(!serde_json::to_string(&checkpoint)
        .unwrap()
        .contains("stage_lag"));
}

/// A stage that has not said how to sit out a boundary cannot be skipped.
#[derive(Debug)]
struct Undeclared;

impl StageHandler for Undeclared {
    fn run(&mut self, _: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        Ok(StageOutcome::Ran)
    }
}

#[test]
fn a_stage_that_has_not_declared_how_to_sit_out_a_boundary_faults_the_run() {
    let mut sim = simulation();
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(Undeclared))
        .unwrap();
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0)
        .unwrap()
        .with_mission_rate(hz(1, 1))
        .unwrap();
    executor.step(&mut sim).expect("tick 0 is due");
    assert!(matches!(
        executor.step(&mut sim),
        Err(ExecutorError::Tick(TickError::Stage {
            tick: 1,
            stage: Stage::EvaluateMissionTransitions,
            ..
        }))
    ));
}

#[test]
fn a_checkpoint_with_lag_is_format_5_and_a_damaged_lag_is_refused() {
    let mut sim = simulation();
    clocked(&mut sim);
    let mut executor = PhasedLaneExecutor::new::<TickError>(0.1, 0)
        .unwrap()
        .with_mission_rate(hz(3, 1))
        .unwrap();
    for _ in 0..9 {
        executor.step(&mut sim).expect("tick");
    }
    let checkpoint = sim.checkpoint().unwrap();
    assert_eq!(
        checkpoint.header.format_version,
        argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION_STAGE_LAG
    );

    // A reader that predates the field would read format 4: a format-4
    // header carrying lag is refused, not half-read.
    let mut as_four = checkpoint.clone();
    as_four.header.format_version = argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION;
    let mut fresh = simulation();
    clocked(&mut fresh);
    assert!(fresh.restore(&as_four).is_err());

    // More boundaries behind than the run has had is damage.
    let mut damaged = checkpoint.clone();
    damaged
        .stage_lag
        .insert("evaluate_mission_transitions".to_string(), u64::MAX);
    assert!(fresh.restore(&damaged).is_err());
    assert!(fresh.restore(&checkpoint).is_ok());
}
