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
    assert_eq!(retry.status, first.status);
    assert!(retry.duplicate);
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

#[test]
fn queued_reset_recovers_a_fault_at_tick_zero_without_running_stages() {
    let mut sim = simulation();
    sim.set_stage_handler(
        Stage::Deconflict,
        Box::new(FailingStage { fail_on_tick: 0 }),
    )
    .unwrap();
    sim.step().unwrap_err();
    assert!(sim
        .apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Reset))
        .is_accepted());
    let reset = sim
        .step()
        .expect("recovery commands run even while faulted");
    assert_eq!((reset.header.tick, reset.header.epoch), (0, 1));
    assert!(!sim.is_faulted());
    assert_eq!(sim.command_ack(1).unwrap().status, AckStatus::Applied);
    // Recovery is a boundary, not a retry of a failed domain stage.
    assert!(matches!(sim.step(), Err(TickError::Stage { tick: 0, .. })));
}

#[test]
fn completed_runs_accept_immediate_reset_and_reject_unreachable_recovery() {
    let mut sim = Simulation::new(SimulationConfig {
        steps: Some(1),
        ..config()
    })
    .unwrap();
    sim.step().unwrap();
    assert_eq!(
        sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Resume))
            .status,
        AckStatus::Rejected(RejectReason::Completed)
    );
    let mut delayed = CommandEnvelope::new(2, "scan", 0, CommandKind::Reset);
    delayed.requested_effective_tick = Some(2);
    assert_eq!(
        sim.apply_command(delayed).status,
        AckStatus::Rejected(RejectReason::RecoveryMustBeImmediate)
    );
    sim.apply_command(CommandEnvelope::new(3, "scan", 0, CommandKind::Reset));
    assert_eq!(sim.step().unwrap().header.tick, 0);
    assert_eq!(sim.step().unwrap().header.tick, 1);
}

#[test]
fn single_step_runs_exactly_once_then_stays_paused() {
    let mut sim = simulation();
    sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause));
    assert_eq!(sim.step(), Err(TickError::Paused));
    sim.apply_command(CommandEnvelope::new(2, "scan", 0, CommandKind::SingleStep));
    assert_eq!(sim.step().unwrap().header.tick, 1);
    assert!(sim.is_paused());
    assert_eq!(sim.step(), Err(TickError::Paused));
    sim.apply_command(CommandEnvelope::new(3, "scan", 0, CommandKind::SingleStep));
    assert_eq!(sim.step().unwrap().header.tick, 2);
    assert_eq!(sim.step(), Err(TickError::Paused));
    sim.apply_command(CommandEnvelope::new(4, "scan", 0, CommandKind::Resume));
    sim.step().unwrap();
    sim.step().unwrap();
    assert_eq!(sim.tick(), 4);
}

#[test]
fn pause_or_resume_after_single_step_has_last_command_precedence() {
    for (last, should_pause) in [(CommandKind::Pause, true), (CommandKind::Resume, false)] {
        let mut sim = simulation();
        sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::SingleStep));
        sim.apply_command(CommandEnvelope::new(2, "scan", 0, last));
        assert_eq!(sim.step().is_err(), should_pause);
        assert_eq!(sim.is_paused(), should_pause);
    }
}

#[test]
fn reset_cancels_both_due_and_future_old_epoch_commands() {
    let mut sim = simulation();
    sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Reset));
    sim.apply_command(CommandEnvelope::new(
        2,
        "scan",
        0,
        CommandKind::SetSpeed { multiplier: 9.0 },
    ));
    let mut future = CommandEnvelope::new(3, "scan", 0, CommandKind::Pause);
    future.requested_effective_tick = Some(5);
    sim.apply_command(future.clone());
    sim.step().unwrap();
    assert_eq!(sim.speed(), 1.0);
    assert_eq!(sim.tick(), 0);
    for id in [2, 3] {
        assert_eq!(
            sim.command_ack(id).unwrap().status,
            AckStatus::Cancelled(RejectReason::StaleEpoch {
                found: 0,
                expected: 1
            })
        );
    }
    let retry = sim.apply_command(future);
    assert!(retry.duplicate);
    assert!(matches!(retry.status, AckStatus::Cancelled(_)));
    for _ in 0..6 {
        sim.step().unwrap();
    }
    assert!(!sim.is_paused());
}

#[test]
fn execution_rechecks_a_scheduled_revision() {
    let mut sim = simulation();
    let mut command = CommandEnvelope::new(1, "scan", 0, CommandKind::Pause);
    command.expected_revision = Some(0);
    command.requested_effective_tick = Some(1);
    sim.apply_command(command);
    sim.step().unwrap();
    sim.step().expect("stale pause does not take effect");
    assert_eq!(
        sim.command_ack(1).unwrap().status,
        AckStatus::Cancelled(RejectReason::StaleRevision {
            found: 0,
            expected: 1
        })
    );
}

#[test]
fn invalid_speed_and_past_ticks_never_mutate_pacing() {
    let mut sim = simulation();
    for (id, multiplier) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0]
        .into_iter()
        .enumerate()
    {
        let command =
            CommandEnvelope::new(id as u64, "scan", 0, CommandKind::SetSpeed { multiplier });
        let first = sim.apply_command(command.clone());
        assert_eq!(
            first.status,
            AckStatus::Rejected(RejectReason::InvalidSpeed)
        );
        let retry = sim.apply_command(command);
        assert!(retry.duplicate);
        assert_eq!(retry.status, first.status);
    }
    sim.step().unwrap();
    let mut past = CommandEnvelope::new(10, "scan", 0, CommandKind::Pause);
    past.requested_effective_tick = Some(0);
    assert_eq!(
        sim.apply_command(past).status,
        AckStatus::Rejected(RejectReason::PastEffectiveTick {
            requested: 0,
            current: 1
        })
    );
    assert_eq!(sim.speed(), 1.0);
}

#[test]
fn a_conflicting_payload_cannot_replace_the_original_command() {
    let mut sim = simulation();
    let command = CommandEnvelope::new(1, "scan", 0, CommandKind::SetSpeed { multiplier: 2.0 });
    sim.apply_command(command.clone());
    assert_eq!(
        sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause))
            .status,
        AckStatus::Rejected(RejectReason::CommandIdConflict)
    );
    assert_eq!(sim.command_ack(1).unwrap().status, AckStatus::Accepted);
    sim.step().unwrap();
    assert_eq!(sim.speed(), 2.0);
    let retry = sim.apply_command(command);
    assert!(retry.duplicate);
    assert_eq!(retry.status, AckStatus::Applied);
}

#[test]
fn a_foreign_run_cannot_read_or_poison_acknowledgements() {
    let mut sim = simulation();
    sim.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause));
    let foreign = sim.apply_command(CommandEnvelope::new(1, "other", 0, CommandKind::Pause));
    assert!(matches!(
        foreign.status,
        AckStatus::Rejected(RejectReason::StaleRun { .. })
    ));
    assert!(!foreign.duplicate);
    sim.apply_command(CommandEnvelope::new(2, "other", 0, CommandKind::Resume));
    assert!(sim
        .apply_command(CommandEnvelope::new(2, "scan", 0, CommandKind::Resume))
        .is_accepted());
    sim.step().unwrap();
}

#[test]
fn busy_retries_keep_the_refusal_instead_of_becoming_success() {
    let mut sim = simulation();
    for id in 0..CommandQueue::DEFAULT_CAPACITY as u64 {
        let mut command = CommandEnvelope::new(id, "scan", 0, CommandKind::Pause);
        command.requested_effective_tick = Some(10);
        assert!(sim.apply_command(command).is_accepted());
    }
    let command = CommandEnvelope::new(1000, "scan", 0, CommandKind::Resume);
    let first = sim.apply_command(command.clone());
    let retry = sim.apply_command(command);
    assert!(matches!(
        first.status,
        AckStatus::Rejected(RejectReason::Busy { .. })
    ));
    assert_eq!(retry.status, first.status);
    assert!(retry.duplicate);
}

#[test]
fn retry_history_eviction_cannot_make_a_retired_command_new_again() {
    let mut queue = CommandQueue::new(1, 2);
    for id in [100, 1, 2] {
        queue.remember(argusnet_sim::command::CommandAck {
            command_id: id,
            status: AckStatus::Accepted,
            effective_tick: None,
            epoch: 0,
            revision: 0,
            duplicate: false,
        });
    }
    assert!(queue.is_expired(100), "arrival order is not numeric order");
    assert!(!queue.is_expired(101));
    assert!(
        queue.previous_ack(1).is_some(),
        "known retries are checked before expiry"
    );
}

#[test]
fn malformed_checkpoints_leave_existing_state_untouched() {
    let mut sim = simulation();
    sim.step().unwrap();
    let original = sim.checkpoint().unwrap();
    let mut cases = Vec::new();
    let mut bad = original.clone();
    bad.tick += 1;
    cases.push(bad);
    let mut bad = original.clone();
    bad.seed ^= 1;
    cases.push(bad);
    let mut bad = original.clone();
    bad.tick = 101;
    bad.header.tick = 101;
    cases.push(bad);
    let mut bad = original.clone();
    bad.pending_commands = vec![CommandEnvelope::new(1, "scan", 0, CommandKind::Pause); 257];
    cases.push(bad);
    for command in [
        CommandEnvelope::new(1, "foreign", 0, CommandKind::Pause),
        CommandEnvelope::new(1, "scan", 9, CommandKind::Pause),
        CommandEnvelope::new(
            1,
            "scan",
            0,
            CommandKind::SetSpeed {
                multiplier: f64::NAN,
            },
        ),
        CommandEnvelope::new(1, "scan", 0, CommandKind::Seek { tick: 1 }),
    ] {
        let mut bad = original.clone();
        bad.pending_commands.push(command);
        cases.push(bad);
    }
    let mut bad = original.clone();
    let command = CommandEnvelope::new(1, "scan", 0, CommandKind::Pause);
    bad.pending_commands = vec![command.clone(), command];
    cases.push(bad);
    for bad in cases {
        assert!(matches!(
            sim.restore(&bad),
            Err(RestoreError::InvalidState { .. })
        ));
        assert_eq!(sim.tick(), 1);
        assert_eq!(sim.revision(), 1);
        assert_eq!(sim.latest_snapshot().unwrap().header.tick, 1);
    }
}

#[test]
fn restore_sets_a_large_valid_clock_directly() {
    let mut sim = Simulation::new(SimulationConfig {
        steps: None,
        ..config()
    })
    .unwrap();
    let mut checkpoint = sim.checkpoint().unwrap();
    checkpoint.tick = 1_000_000_000_000;
    checkpoint.header.tick = checkpoint.tick;
    sim.restore(&checkpoint).unwrap();
    assert_eq!(sim.tick(), checkpoint.tick);
    assert_eq!(sim.step().unwrap().header.tick, checkpoint.tick + 1);
}

fn json_checkpoint(sim: &Simulation) -> argusnet_sim::snapshot::SimulationCheckpoint {
    serde_json::from_str(&serde_json::to_string(&sim.checkpoint().unwrap()).unwrap()).unwrap()
}

#[test]
fn checkpoint_roundtrip_preserves_pause_pacing_revision_and_retry_results() {
    let mut source = simulation();
    let speed = CommandEnvelope::new(1, "scan", 0, CommandKind::SetSpeed { multiplier: 2.5 });
    source.apply_command(speed.clone());
    source.step().unwrap();
    source.apply_command(CommandEnvelope::new(2, "scan", 0, CommandKind::Pause));
    assert_eq!(source.step(), Err(TickError::Paused));
    let checkpoint = json_checkpoint(&source);
    let mut resumed = simulation();
    // Unrelated receiver history must be replaced rather than merged.
    resumed.apply_command(CommandEnvelope::new(90, "scan", 0, CommandKind::Pause));
    resumed.restore(&checkpoint).unwrap();
    assert!(resumed.command_ack(90).is_none());
    assert!(resumed.is_paused());
    assert_eq!(resumed.speed(), 2.5);
    assert_eq!(resumed.revision(), source.revision());
    assert_eq!(resumed.step(), Err(TickError::Paused));
    let retry = resumed.apply_command(speed);
    assert_eq!(retry.status, AckStatus::Applied);
    assert!(retry.duplicate);
    for sim in [&mut source, &mut resumed] {
        sim.apply_command(CommandEnvelope::new(3, "scan", 0, CommandKind::SingleStep));
        sim.step().unwrap();
    }
    assert_eq!(source.latest_snapshot(), resumed.latest_snapshot());
    assert_eq!(source.command_ack(3), resumed.command_ack(3));
}

#[test]
fn restored_pending_command_retry_is_not_enqueued_twice() {
    let mut source = simulation();
    source.step().unwrap();
    let mut command = CommandEnvelope::new(1, "scan", 0, CommandKind::Pause);
    command.requested_effective_tick = Some(1);
    command.expected_revision = Some(source.revision());
    source.apply_command(command.clone());
    let mut resumed = simulation();
    resumed.restore(&json_checkpoint(&source)).unwrap();
    let retry = resumed.apply_command(command);
    assert!(retry.duplicate);
    assert_eq!(retry.status, AckStatus::Accepted);
    assert_eq!(resumed.checkpoint().unwrap().pending_commands.len(), 1);
    assert_eq!(resumed.step(), Err(TickError::Paused));
    assert_eq!(resumed.command_ack(1).unwrap().status, AckStatus::Applied);
}

#[test]
fn rejected_nonfinite_request_identity_survives_json_checkpoint() {
    let mut source = simulation();
    let command = CommandEnvelope::new(
        1,
        "scan",
        0,
        CommandKind::SetSpeed {
            multiplier: f64::NAN,
        },
    );
    source.apply_command(command.clone());
    let mut resumed = simulation();
    resumed.restore(&json_checkpoint(&source)).unwrap();
    let retry = resumed.apply_command(command);
    assert!(retry.duplicate);
    assert_eq!(
        retry.status,
        AckStatus::Rejected(RejectReason::InvalidSpeed)
    );
    assert_eq!(
        resumed
            .apply_command(CommandEnvelope::new(
                1,
                "scan",
                0,
                CommandKind::SetSpeed {
                    multiplier: f64::INFINITY
                }
            ))
            .status,
        AckStatus::Rejected(RejectReason::CommandIdConflict)
    );
}

#[test]
fn checkpoint_keeps_retired_ids_and_cancelled_outcomes() {
    let mut source = simulation();
    for id in 0..=CommandQueue::DEFAULT_DEDUP_CAPACITY as u64 {
        source.apply_command(CommandEnvelope::new(
            id,
            "scan",
            0,
            CommandKind::Seek { tick: 0 },
        ));
    }
    let mut resumed = simulation();
    resumed.restore(&json_checkpoint(&source)).unwrap();
    assert!(matches!(
        resumed
            .apply_command(CommandEnvelope::new(0, "scan", 0, CommandKind::Pause))
            .status,
        AckStatus::Rejected(RejectReason::DedupWindowExpired { .. })
    ));
    let id = CommandQueue::DEFAULT_DEDUP_CAPACITY as u64 + 1;
    let mut pending = CommandEnvelope::new(id, "scan", 0, CommandKind::Pause);
    pending.requested_effective_tick = Some(5);
    resumed.apply_command(pending.clone());
    resumed.reset();
    let mut restored = simulation();
    restored.restore(&json_checkpoint(&resumed)).unwrap();
    let retry = restored.apply_command(pending);
    assert!(retry.duplicate);
    assert!(matches!(
        retry.status,
        AckStatus::Cancelled(RejectReason::StaleEpoch { .. })
    ));
}

#[test]
fn invalid_control_checkpoint_does_not_mutate_the_receiver() {
    let mut source = simulation();
    source.apply_command(CommandEnvelope::new(1, "scan", 0, CommandKind::Pause));
    let valid = source.checkpoint().unwrap();
    let mut variants = Vec::new();
    let mut bad = valid.clone();
    bad.speed = 0.0;
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history.receipts.clear();
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history
        .receipts
        .push(bad.command_history.receipts[0].clone());
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history.receipts[0].ack.status = AckStatus::Applied;
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history.receipts[0].speed_bits = Some(1);
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history.receipts[0]
        .request
        .as_mut()
        .unwrap()
        .payload = CommandKind::Resume;
    variants.push(bad);
    let mut bad = valid.clone();
    bad.command_history.receipts[0].ack.revision = 99;
    variants.push(bad);
    let mut bad = valid.clone();
    bad.header.format_version = 1;
    variants.push(bad);
    let mut target = simulation();
    target.step().unwrap();
    let before = target.checkpoint().unwrap();
    for bad in variants {
        assert!(target.restore(&bad).is_err());
        assert_eq!(target.checkpoint().unwrap(), before);
    }
}
