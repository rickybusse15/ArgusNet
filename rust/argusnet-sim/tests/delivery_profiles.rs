//! B4: timestamped delivery on an assembled run.
//!
//! The gates: a zero profile through the delivery network reproduces the
//! legacy run exactly; a delayed profile fuses each tick's evidence at its
//! acquisition time, deterministically, with nothing lost; a checkpoint taken
//! with evidence in transit resumes exactly; an unsupported profile is refused
//! before the run; and tracks age through intervals with no evidence.

use std::collections::BTreeMap;

use argusnet_sim::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
use argusnet_sim::observe::ObserveStage;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::stage::Stage;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_tt.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn profile(lag_s: f64, delay_s: f64, jitter_s: f64, dt_s: f64) -> DeliveryProfile {
    DeliveryProfile::from_seconds(
        dt_s,
        lag_s,
        delay_s,
        &BTreeMap::new(),
        jitter_s,
        DEFAULT_QUEUE_CAPACITY,
        11,
    )
    .expect("a supported profile")
}

fn frames(run: AssembledRun) -> Vec<String> {
    run.run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|f| serde_json::to_string(f).unwrap())
        .collect()
}

#[test]
fn a_zero_profile_through_the_network_is_the_legacy_run() {
    let bundle = bundle();
    let legacy = frames(assemble(&bundle).unwrap());
    let mut run = assemble(&bundle).unwrap();
    run.set_delivery(profile(0.0, 0.0, 0.0, bundle.dt_s))
        .unwrap();
    // Through the network, every track also records its evidence time
    // (P4.10b); apart from that label the frames are the legacy run's.
    let delivered = frames(run);
    assert!(delivered.iter().any(|f| f.contains("\"evidence_time_s\"")));
    let parsed = |frames: &[String]| {
        frames
            .iter()
            .map(|f| without_evidence_time(f))
            .collect::<Vec<_>>()
    };
    assert_eq!(parsed(&delivered), parsed(&legacy));
}

/// A serialised frame, parsed, with every track's `evidence_time_s` taken out.
fn without_evidence_time(frame: &str) -> serde_json::Value {
    let mut value: serde_json::Value = serde_json::from_str(frame).unwrap();
    for track in value["tracks"].as_array_mut().into_iter().flatten() {
        track.as_object_mut().unwrap().remove("evidence_time_s");
    }
    value
}

/// Lag 2 ticks, delay 1, jitter 1: every fused observation at tick t was
/// acquired at t - 2, every captured one is fused or accounted for, and the
/// run is the same twice.
#[test]
fn a_delayed_profile_fuses_each_tick_at_its_acquisition_time() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let run_once = || {
        let mut run = assemble(&bundle).unwrap();
        run.set_delivery(profile(2.0 * dt, dt, dt, dt)).unwrap();
        let mut fused_at = Vec::new();
        let mut delivery_drops = 0usize;
        for tick in 0..bundle.steps {
            run.step().expect("tick runs");
            let frame = run.fusion().frame().expect("stage 7 ran");
            delivery_drops += frame
                .rejected_observations
                .iter()
                .filter(|r| r.reason.starts_with("delivery_"))
                .count();
            for observation in &frame.observations {
                fused_at.push((tick, observation.timestamp_s));
            }
        }
        let totals = run.fusion().delivery().expect("modelled").totals().clone();
        (
            fused_at,
            delivery_drops,
            totals,
            run.fusion().frame().unwrap().tracks.len(),
        )
    };
    let (fused_at, drops, totals, tracks) = run_once();
    assert!(!fused_at.is_empty());
    for (tick, acquired_s) in &fused_at {
        let expected = (*tick as f64 - 2.0) * dt;
        assert!(
            (acquired_s - expected).abs() < 1e-9,
            "fused at tick {tick} but acquired at {acquired_s} s, not {expected} s"
        );
    }
    assert_eq!(drops, 0, "a validated profile loses nothing");
    let pending = totals.captured - totals.fused;
    assert!(
        pending <= 3 * 64,
        "only the last lag's evidence is unfused: {totals:?}"
    );
    assert!(tracks > 0, "tracks form from delayed evidence");
    let again = run_once();
    assert_eq!(
        again.0, fused_at,
        "deterministic for a fixed seed and profile"
    );
}

#[test]
fn a_checkpoint_with_evidence_in_transit_resumes_exactly() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let make = || {
        let mut run = assemble(&bundle).unwrap();
        run.set_delivery(profile(3.0 * dt, dt, 2.0 * dt, dt))
            .unwrap();
        run
    };
    let straight = frames(make());
    let at = 30usize;
    let mut original = make();
    for _ in 0..at {
        original.step().unwrap();
    }
    assert!(
        original.fusion().delivery().unwrap().pending() > 0,
        "evidence in transit"
    );
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = make();
    resumed.sim.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<String> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_string(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);

    // A run without delivery refuses a checkpoint that has it, and back.
    let mut legacy = assemble(&bundle).unwrap();
    assert!(legacy.sim.restore(&checkpoint).is_err());
}

#[test]
fn an_unsupported_profile_is_refused_before_the_run() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let slow = DeliveryProfile {
        fusion_lag_ticks: 1,
        default_delay_ticks: 2,
        per_source_delay_ticks: BTreeMap::new(),
        jitter_ticks: 0,
        queue_capacity: 8,
        seed: 1,
    };
    let mut run = assemble(&bundle).unwrap();
    let error = run.set_delivery(slow).unwrap_err();
    assert!(error.contains("unsupported delivery profile"), "{error}");
    assert!(run.fusion().delivery().is_none(), "nothing was installed");
    assert!(DeliveryProfile::from_seconds(dt, dt, 2.0 * dt, &BTreeMap::new(), 0.0, 8, 1).is_err());
}

/// Tracks age through ticks with no evidence: once the sensors go quiet,
/// a delayed run loses its tracks the same number of ticks after the last
/// fused evidence as the legacy run does.
#[test]
fn tracks_age_through_empty_intervals() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let quiet_from = 40u64;
    let lifetime = |delivery: Option<DeliveryProfile>| {
        let mut run = assemble(&bundle).unwrap();
        if let Some(p) = delivery {
            run.set_delivery(p).unwrap();
        }
        let mut last_track_tick = None;
        for tick in 0..bundle.steps {
            if tick == quiet_from {
                // Blind every sensor: nothing more is captured.
                let observe = run
                    .sim
                    .stage_mut::<ObserveStage>(Stage::GenerateObservations)
                    .expect("installed");
                for node in &bundle.nodes {
                    observe.remove_sensor(&node.node_id);
                }
            }
            run.step().unwrap();
            if !run.fusion().frame().unwrap().tracks.is_empty() {
                last_track_tick = Some(tick);
            }
        }
        last_track_tick
    };
    let legacy = lifetime(None).expect("tracks existed");
    let delayed = lifetime(Some(profile(2.0 * dt, dt, 0.0, dt))).expect("tracks existed");
    assert!(
        legacy < bundle.steps - 1,
        "tracks expire after the sensors go quiet"
    );
    assert_eq!(
        delayed,
        legacy + 2,
        "the same expiry, two ticks of lag later"
    );
}

/// The replay accounts for every captured envelope: fused, dropped with a
/// recorded reason, or still in transit when the run ended.
#[test]
fn the_replay_accounts_for_every_envelope() {
    let bundle = bundle();
    let dt = bundle.dt_s;
    let mut run = assemble(&bundle).unwrap();
    let mut p = profile(3.0 * dt, 2.0 * dt, dt, dt);
    p.queue_capacity = 2;
    run.set_delivery(p).unwrap();
    let replay = run.run_to_replay().unwrap();
    let meta = serde_json::to_value(&replay.meta).unwrap();
    let totals = &meta["delivery_totals"];
    let captured = totals["captured"].as_u64().unwrap();
    let fused = totals["fused"].as_u64().unwrap();
    let pending = totals["in_transit_at_end"].as_u64().unwrap();
    let dropped: u64 = totals["dropped"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert!(dropped > 0, "a two-envelope queue overflows");
    assert_eq!(captured, fused + dropped + pending, "{totals}");
    let recorded = replay
        .frames
        .iter()
        .flat_map(|f| &f.rejected_observations)
        .filter(|r| r.reason.starts_with("delivery_"))
        .count();
    assert_eq!(
        recorded as u64, dropped,
        "every drop is in a frame with its reason"
    );
}
