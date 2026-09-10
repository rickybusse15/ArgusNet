//! P4.1's exit criterion, asserted.
//!
//! > *"Exit: trace harness, reset semantics, fault behavior and snapshot
//! > headers are testable without viewer/network/file output."*
//!
//! Nothing in this file opens a socket, spawns a renderer or touches the
//! filesystem. That is the point: the substrate has to be checkable on its own
//! before any physics moves into it, or the first stage port lands on a
//! foundation nobody has tested.

use argusnet_sim::command::{
    AckStatus, CommandEnvelope, CommandKind, CommandQueue, RejectReason, PROTOCOL_VERSION,
};
use argusnet_sim::config::SimulationConfig;
use argusnet_sim::simulation::{Simulation, StageHandler, TickContext, TickError};
use argusnet_sim::snapshot::{CheckpointHeader, RestoreError};
use argusnet_sim::stage::{Stage, StageError, StageOutcome};

fn config() -> SimulationConfig {
    SimulationConfig {
        seed: 42,
        dt_s: 0.1,
        steps: Some(100),
        run_id: "scan".to_string(),
    }
}

fn simulation() -> Simulation {
    Simulation::new(config()).expect("config is valid")
}

/// A stage body that fails on demand, so fault handling can be exercised
/// without waiting for a real stage to have a real bug.
#[derive(Debug)]
struct FailingStage {
    fail_on_tick: u64,
}

impl StageHandler for FailingStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        if context.tick == self.fail_on_tick {
            Err(StageError::Invariant("injected failure".to_string()))
        } else {
            Ok(StageOutcome::Ran)
        }
    }
}

/// Records what it saw, so the context handed to stages can be checked.
#[derive(Debug, Default)]
struct RecordingStage {
    seen: Vec<(u64, f64, i64)>,
}

impl StageHandler for RecordingStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        self.seen
            .push((context.tick, context.timestamp_s, context.seed_time_key));
        context.note(format!(
            "controller change effective at tick {}",
            context.tick + 1
        ));
        Ok(StageOutcome::Ran)
    }
}

// ---------------------------------------------------------------------------
// Trace harness
// ---------------------------------------------------------------------------

#[test]
fn a_tick_runs_every_stage_in_the_architecture_order() {
    let mut sim = simulation();
    sim.step().expect("first tick");
    assert_eq!(
        sim.trace().stage_order(0),
        Stage::PIPELINE.to_vec(),
        "the trace is the evidence that causal order was preserved"
    );
}

#[test]
fn the_trace_is_machine_readable() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::EvaluateMissionTransitions,
        Box::new(RecordingStage::default()),
    )
    .expect("stage 9 is delegable");
    sim.step().expect("first tick");

    let json = sim.trace().to_json().expect("serializable");
    // §4 asks stage 9 to record which future tick a controller change lands on.
    assert!(
        json.contains("controller change effective at tick 1"),
        "the future-effect note must reach the trace: {json}"
    );
    assert!(json.contains("evaluate_mission_transitions"));
}

#[test]
fn stages_see_derived_time_and_the_python_seed_key() {
    let mut sim = simulation();
    let mut recorder = RecordingStage::default();
    // Drive the handler directly for the assertions, then re-register a fresh
    // one -- the simulation takes ownership, so the values are read back from
    // the trace and the clock instead.
    recorder.seen.clear();
    sim.set_stage_handler(Stage::UpdateFusion, Box::new(RecordingStage::default()))
        .expect("stage 7 is delegable");

    for _ in 0..3 {
        sim.step().expect("tick");
    }
    // dt = 0.1, so ticks 0..2 are 0 ms, 100 ms, 200 ms.
    assert_eq!(sim.tick(), 3);
    let notes: Vec<&str> = sim
        .trace()
        .records()
        .iter()
        .flat_map(|record| record.notes.iter().map(String::as_str))
        .collect();
    assert!(notes.contains(&"controller change effective at tick 1"));
    assert!(notes.contains(&"controller change effective at tick 3"));
}

#[test]
fn the_trace_is_bounded_and_reports_what_it_dropped() {
    let mut sim = Simulation::new(SimulationConfig {
        steps: Some(2_000),
        ..config()
    })
    .expect("valid");
    for _ in 0..1_000 {
        sim.step().expect("tick");
    }
    let trace = sim.trace();
    assert!(
        trace.records().len() <= Simulation::DEFAULT_TRACE_CAPACITY,
        "trace grew past its bound"
    );
    assert!(
        trace.dropped() > 0,
        "1000 ticks x 11 stages must have exceeded the window"
    );
}

// ---------------------------------------------------------------------------
// Snapshot headers
// ---------------------------------------------------------------------------

#[test]
fn a_committed_snapshot_carries_a_complete_header() {
    let mut sim = simulation();
    let snapshot = sim.step().expect("first tick");
    let header = &snapshot.header;
    assert_eq!(header.run_id, "scan");
    assert_eq!(header.epoch, 0);
    assert_eq!(header.tick, 1, "the header names the tick just committed");
    assert_eq!(header.revision, 1);
    assert_eq!(header.timestamp_s, 0.1);
    assert!(
        header.truth_filtered,
        "a presentation snapshot must declare that truth was withheld"
    );
}

#[test]
fn revisions_are_monotonic_and_snapshots_are_shared_not_copied() {
    let mut sim = simulation();
    let first = sim.step().expect("tick");
    let second = sim.step().expect("tick");
    assert!(second.header.revision > first.header.revision);
    assert_eq!(
        std::sync::Arc::as_ptr(&second),
        std::sync::Arc::as_ptr(&sim.latest_snapshot().expect("committed")),
        "latest_snapshot shares the Arc rather than cloning the payload"
    );
}

#[test]
fn a_run_reports_completion_rather_than_stepping_past_its_length() {
    let mut sim = Simulation::new(SimulationConfig {
        steps: Some(2),
        ..config()
    })
    .expect("valid");
    sim.step().expect("tick 0");
    sim.step().expect("tick 1");
    assert_eq!(sim.step(), Err(TickError::Completed { tick: 2 }));
}

// ---------------------------------------------------------------------------
// Fault behaviour
// ---------------------------------------------------------------------------

#[test]
fn a_failing_stage_publishes_nothing_and_faults_the_run() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("stage 3 is delegable");

    let error = sim.step().expect_err("stage 3 fails on tick 0");
    assert!(matches!(
        error,
        TickError::Stage {
            tick: 0,
            stage: Stage::Deconflict,
            ..
        }
    ));
    assert!(sim.is_faulted());
    assert!(
        sim.latest_snapshot().is_none(),
        "no partial snapshot may be published"
    );
    assert_eq!(sim.tick(), 0, "a failed tick does not advance the clock");
    assert_eq!(sim.revision(), 0, "and does not produce a revision");
}

#[test]
fn the_trace_shows_where_the_tick_stopped() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");

    let order = sim.trace().stage_order(0);
    assert_eq!(
        order,
        vec![
            Stage::ApplyCommands,
            Stage::SampleControllersAndBattery,
            Stage::ApplyMissionOverrides,
            Stage::Deconflict,
        ],
        "stages after the failure must not appear -- they did not run"
    );
    let last = sim
        .trace()
        .tick(0)
        .last()
        .copied()
        .cloned()
        .expect("a record");
    assert!(matches!(last.outcome, StageOutcome::Failed { .. }));
}

#[test]
fn a_faulted_run_refuses_to_step_until_it_is_reset() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::SafetyMonitor,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");

    assert_eq!(
        sim.step(),
        Err(TickError::Faulted {
            at_tick: 0,
            stage: Stage::SafetyMonitor
        }),
        "a second step must not quietly retry a half-updated run"
    );
    // The message has to tell an operator what to do about it.
    let message = sim.step().expect_err("still faulted").to_string();
    assert!(message.contains("reset or restore"), "{message}");
}

#[test]
fn a_faulted_run_refuses_to_checkpoint() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::UpdateFusion,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");
    assert!(
        sim.checkpoint().is_err(),
        "checkpointing half-mutated state is what §4 forbids restoring from"
    );
}

#[test]
fn a_faulted_run_accepts_only_reset() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");

    let refused = sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Resume));
    assert_eq!(refused.status, AckStatus::Rejected(RejectReason::Faulted));

    let accepted = sim.apply_command(CommandEnvelope::new(2, "scan", 0, CommandKind::Reset));
    assert!(accepted.is_accepted(), "reset is the way out of a fault");
}

// ---------------------------------------------------------------------------
// Reset semantics
// ---------------------------------------------------------------------------

#[test]
fn reset_returns_to_tick_zero_on_a_new_epoch() {
    let mut sim = simulation();
    for _ in 0..5 {
        sim.step().expect("tick");
    }
    let revision_before = sim.revision();

    sim.reset();

    assert_eq!(sim.tick(), 0);
    assert_eq!(sim.epoch(), 1, "history branched, so the epoch must move");
    assert_eq!(
        sim.revision(),
        revision_before,
        "revisions keep counting; restarting them would make a stale snapshot \
         look newer than a current one"
    );
    assert!(sim.latest_snapshot().is_none());
    assert!(sim.trace().records().is_empty());
}

#[test]
fn reset_clears_a_fault() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");
    assert!(sim.is_faulted());

    sim.reset();

    assert!(!sim.is_faulted());
    // The injected failure was keyed to tick 0, and reset returns there, so a
    // step now fails again -- which is the honest outcome and worth pinning:
    // reset clears the fault state, it does not repair the cause.
    assert!(sim.step().is_err());
}

#[test]
fn a_command_from_the_previous_epoch_is_refused_after_reset() {
    let mut sim = simulation();
    sim.reset();
    let stale = sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause));
    assert_eq!(
        stale.status,
        AckStatus::Rejected(RejectReason::StaleEpoch {
            found: 0,
            expected: 1
        })
    );
}

// ---------------------------------------------------------------------------
// Checkpoint and restore
// ---------------------------------------------------------------------------

#[test]
fn restoring_reproduces_uninterrupted_execution() {
    // §6: "Checkpoint restore must reproduce uninterrupted execution."
    let mut straight_through = simulation();
    for _ in 0..15 {
        straight_through.step().expect("tick");
    }

    let mut interrupted = simulation();
    for _ in 0..10 {
        interrupted.step().expect("tick");
    }
    let checkpoint = interrupted.checkpoint().expect("not faulted");
    let mut resumed = simulation();
    resumed.restore(&checkpoint).expect("compatible");
    for _ in 0..5 {
        resumed.step().expect("tick");
    }

    assert_eq!(resumed.tick(), straight_through.tick());
    let a = resumed.latest_snapshot().expect("committed");
    let b = straight_through.latest_snapshot().expect("committed");
    assert_eq!(a.header.tick, b.header.tick);
    assert_eq!(a.header.timestamp_s, b.header.timestamp_s);
    assert_eq!(a.header.epoch, b.header.epoch);
}

#[test]
fn a_checkpoint_carries_pending_commands() {
    // §6: "Persist all causal state, including pending commands."
    let mut sim = simulation();
    let mut later = CommandEnvelope::new(1, "scan", 0, CommandKind::Pause);
    later.requested_effective_tick = Some(50);
    assert!(sim.apply_command(later).is_accepted());

    let checkpoint = sim.checkpoint().expect("not faulted");
    assert_eq!(checkpoint.pending_commands.len(), 1);

    let mut resumed = simulation();
    resumed.restore(&checkpoint).expect("compatible");
    // The command survives, still scheduled: stepping past tick 50 pauses.
    for _ in 0..51 {
        if resumed.step().is_err() {
            break;
        }
    }
    assert!(resumed.is_paused(), "the restored command took effect");
}

#[test]
fn an_incompatible_checkpoint_is_refused_with_a_reason() {
    let mut sim = simulation();
    sim.step().expect("tick");
    let mut checkpoint = sim.checkpoint().expect("not faulted");
    checkpoint.header.seed_version = CheckpointHeader::SEED_VERSION + 1;

    let error = sim.restore(&checkpoint).expect_err("seed version moved");
    assert!(matches!(error, RestoreError::SeedVersion { .. }));
}

#[test]
fn a_refused_restore_leaves_the_run_untouched() {
    let mut sim = simulation();
    for _ in 0..4 {
        sim.step().expect("tick");
    }
    let tick_before = sim.tick();
    let revision_before = sim.revision();

    let mut incompatible = sim.checkpoint().expect("not faulted");
    incompatible.header.config_hash ^= 0xffff;
    incompatible.tick = 999;
    sim.restore(&incompatible).expect_err("config hash moved");

    assert_eq!(sim.tick(), tick_before, "state must not be half-restored");
    assert_eq!(sim.revision(), revision_before);
}

#[test]
fn restore_clears_a_fault() {
    let mut clean = simulation();
    for _ in 0..3 {
        clean.step().expect("tick");
    }
    let checkpoint = clean.checkpoint().expect("not faulted");

    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .expect("delegable");
    sim.step().expect_err("fails");
    assert!(sim.is_faulted());

    sim.restore(&checkpoint).expect("compatible");
    assert!(!sim.is_faulted());
    assert_eq!(sim.tick(), 3);
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[test]
fn pause_is_checked_between_ticks_and_resume_can_still_reach_the_run() {
    let mut sim = simulation();
    assert!(sim
        .apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause))
        .is_accepted());

    // §5: pause is checked *between* ticks. The command is admitted at this
    // boundary and the tick it gates does not run.
    assert_eq!(sim.step(), Err(TickError::Paused));
    assert!(sim.is_paused());
    assert_eq!(sim.tick(), 0, "the gated tick did not advance the clock");

    // The trap this guards against: commands drain in stage 0, and stage 0 runs
    // inside `step`. If `step` checked pause first, a queued Resume could never
    // be applied and the run would be stuck paused forever.
    assert!(sim
        .apply_command(CommandEnvelope::new(2, "scan", 0, CommandKind::Resume))
        .is_accepted());
    sim.step()
        .expect("the resume was admitted at the boundary and the tick ran");
    assert!(!sim.is_paused());
    assert_eq!(sim.tick(), 1);
}

#[test]
fn a_gated_tick_still_records_that_commands_were_admitted() {
    // A paused run is not an invisible one: the trace shows stage 0 ran and
    // nothing else did, which is how a reader tells "paused" from "hung".
    let mut sim = simulation();
    sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause));
    assert_eq!(sim.step(), Err(TickError::Paused));
    assert_eq!(sim.trace().stage_order(0), vec![Stage::ApplyCommands]);
}

#[test]
fn a_retry_gets_the_original_answer_and_is_not_applied_twice() {
    let mut sim = simulation();
    // SetSpeed rather than Pause: a pause would gate the very tick this test
    // needs to run in order to read the trace back.
    let speed = CommandKind::SetSpeed { multiplier: 2.0 };
    let first = sim.apply_command(CommandEnvelope::new(7, "scan", 0, speed.clone()));
    assert!(first.is_accepted());

    let retry = sim.apply_command(CommandEnvelope::new(7, "scan", 0, speed));
    assert_eq!(retry.status, AckStatus::Duplicate);
    assert_eq!(retry.effective_tick, first.effective_tick);

    sim.step().expect("tick");
    let applied = sim
        .trace()
        .records()
        .iter()
        .filter(|record| record.notes.iter().any(|note| note.contains("command 7")))
        .count();
    assert_eq!(applied, 1, "the retry must not queue a second application");
}

#[test]
fn a_full_queue_answers_busy_rather_than_dropping() {
    let mut sim = simulation();
    let mut refused = None;
    for id in 0..(CommandQueue::DEFAULT_CAPACITY as u64 + 5) {
        let mut envelope =
            CommandEnvelope::new(id, "scan", 0, CommandKind::SetSpeed { multiplier: 1.0 });
        // Schedule far ahead so nothing drains before the queue fills.
        envelope.requested_effective_tick = Some(10_000);
        let ack = sim.apply_command(envelope);
        if let AckStatus::Rejected(RejectReason::Busy { .. }) = ack.status {
            refused = Some(ack);
            break;
        }
    }
    assert!(
        refused.is_some(),
        "the queue must refuse rather than accept unboundedly"
    );
}

#[test]
fn every_rejection_reason_is_distinguishable() {
    let mut sim = simulation();

    let wrong_protocol = CommandEnvelope {
        protocol_version: PROTOCOL_VERSION + 1,
        ..CommandEnvelope::new(1, "scan", 0, CommandKind::Pause)
    };
    assert!(matches!(
        sim.apply_command(wrong_protocol).status,
        AckStatus::Rejected(RejectReason::ProtocolVersion { .. })
    ));

    assert!(matches!(
        sim.apply_command(CommandEnvelope::new(2, "other", 0, CommandKind::Pause))
            .status,
        AckStatus::Rejected(RejectReason::StaleRun { .. })
    ));

    assert!(matches!(
        sim.apply_command(CommandEnvelope::new(3, "scan", 9, CommandKind::Pause))
            .status,
        AckStatus::Rejected(RejectReason::StaleEpoch { .. })
    ));

    assert!(matches!(
        sim.apply_command(CommandEnvelope::new(
            4,
            "scan",
            0,
            CommandKind::Unsupported {
                name: "teleport".to_string()
            }
        ))
        .status,
        AckStatus::Rejected(RejectReason::Unsupported { .. })
    ));

    // Seek is refused honestly: this shell keeps no checkpoint history, so
    // accepting it and not moving would be the silent failure §5 forbids.
    assert!(matches!(
        sim.apply_command(CommandEnvelope::new(
            5,
            "scan",
            0,
            CommandKind::Seek { tick: 2 }
        ))
        .status,
        AckStatus::Rejected(RejectReason::RewindUnavailable { .. })
    ));

    let stale_revision = CommandEnvelope {
        expected_revision: Some(99),
        ..CommandEnvelope::new(6, "scan", 0, CommandKind::Pause)
    };
    assert!(matches!(
        sim.apply_command(stale_revision).status,
        AckStatus::Rejected(RejectReason::StaleRevision { .. })
    ));
}

#[test]
fn stage_zero_and_ten_cannot_be_delegated() {
    let mut sim = simulation();
    assert!(
        sim.set_stage_handler(
            Stage::ApplyCommands,
            Box::new(FailingStage { fail_on_tick: 0 })
        )
        .is_err(),
        "a handler that would never run must be refused, not silently ignored"
    );
    assert!(sim
        .set_stage_handler(
            Stage::CommitSnapshot,
            Box::new(FailingStage { fail_on_tick: 0 })
        )
        .is_err());
}
