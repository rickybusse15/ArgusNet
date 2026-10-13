//! P4.10b: each track's evidence time in the replay.
//!
//! A run whose tracks can be older than their frame -- delivery modelled, or
//! fusion at its own rate -- records when every track last took a
//! measurement. A legacy run records nothing, so its replay is unchanged.

use std::collections::BTreeMap;

use argusnet_sim::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
use argusnet_sim::replay::ReplayFrame;
use argusnet_sim::run_options::RunOptions;
use argusnet_sim::scenario::{assemble, RunError, ScenarioBundle};
use argusnet_sim::scheduler::{parse_rate, LaneRates, PhasedLaneExecutor};
use argusnet_sim::simulation::TickError;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_tt.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn tracks_seen(frames: &[ReplayFrame]) -> usize {
    frames.iter().map(|frame| frame.tracks.len()).sum()
}

#[test]
fn a_legacy_run_records_no_evidence_time() {
    let run = assemble(&bundle()).unwrap();
    assert!(!run.tracks_lag_their_frames());
    let replay = run.run_to_replay().unwrap();
    assert!(tracks_seen(&replay.frames) > 0);
    assert!(replay
        .frames
        .iter()
        .flat_map(|frame| &frame.tracks)
        .all(|track| track.evidence_time_s.is_none()));
}

/// Fusion two ticks behind with a tick of delay and jitter: every track
/// records its last measurement, no later than its own time (the fusion
/// horizon), which is no later than the frame. A track updated at the horizon
/// was measured then; a coasting one was measured earlier.
#[test]
fn a_delayed_run_records_when_each_track_was_last_measured() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let profile = DeliveryProfile::from_seconds(
        bundle.dt_s,
        2.0 * bundle.dt_s,
        bundle.dt_s,
        &BTreeMap::new(),
        bundle.dt_s,
        DEFAULT_QUEUE_CAPACITY,
        11,
    )
    .expect("a supported profile");
    run.set_delivery(profile).unwrap();
    assert!(run.tracks_lag_their_frames());
    let replay = run.run_to_replay().unwrap();
    assert!(tracks_seen(&replay.frames) > 0);
    let mut coasting = 0;
    for frame in &replay.frames {
        for track in &frame.tracks {
            let evidence = track.evidence_time_s.expect("a timed run records it");
            assert!(track.timestamp_s <= frame.timestamp_s);
            if track.stale_steps == 0 {
                assert_eq!(evidence, track.timestamp_s, "{}", track.track_id);
            } else {
                assert!(evidence < track.timestamp_s, "{}", track.track_id);
                coasting += 1;
            }
        }
    }
    assert!(coasting > 0, "the case exercises coasting tracks");
}

/// Fusion at a third of the boundaries: between its runs a frame republishes
/// the last tracks, and their evidence times say how old they are.
#[test]
fn a_rated_fusion_run_records_evidence_time_on_every_frame() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let dt_ms = (bundle.dt_s * 1000.0).round() as u64;
    let rates = LaneRates {
        fusion: Some(parse_rate(&format!("1000/{}", 3 * dt_ms)).unwrap()),
        ..LaneRates::default()
    };
    RunOptions {
        lane_rates: rates,
        ..RunOptions::default()
    }
    .apply(&mut run, &bundle)
    .unwrap();
    assert!(run.tracks_lag_their_frames());
    let mut lanes = rates
        .apply(PhasedLaneExecutor::new::<RunError>(bundle.dt_s, 0).expect("exact timestep"))
        .expect("rates");
    let mut ages = Vec::new();
    loop {
        match lanes.step(&mut run) {
            Ok(_) => {
                let frame = run.last_frame().expect("a frame per boundary");
                for track in &frame.tracks {
                    let evidence = track.evidence_time_s.expect("a timed run records it");
                    ages.push(frame.timestamp_s - evidence);
                }
            }
            Err(argusnet_sim::scheduler::ExecutorError::Tick(RunError::Tick(
                TickError::Completed { .. },
            ))) => break,
            Err(error) => panic!("{error:?}"),
        }
    }
    assert!(!ages.is_empty());
    assert!(ages.iter().all(|age| *age >= 0.0));
    // Republished between fusion runs, some tracks are a boundary or two old.
    assert!(ages.iter().any(|age| *age >= bundle.dt_s));
}
