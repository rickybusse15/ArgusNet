//! F2: sensor and link failure scenarios, each with expectations declared
//! before it ran.
//!
//! The deadlines come from the fixture's tracker configuration, not from
//! measuring the runs: dt is 0.5 s; labeled association (a track per target);
//! a track is dropped after `max_stale_steps` = 8 stale ticks (4 s); a track is
//! confirmed after `confirmation_m` = 3 updates in `confirmation_n` = 5 ticks
//! (at most 2.5 s once evidence returns).

use std::collections::BTreeMap;

use argusnet_sim::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
use argusnet_sim::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};
use argusnet_sim::replay::ReplayFrame;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

const DT: f64 = 0.5;
const DROP_AFTER_S: f64 = 8.0 * DT;
const CONFIRM_WITHIN_S: f64 = 5.0 * DT;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_tt.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let bundle = ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses");
    assert_eq!(
        bundle.dt_s, DT,
        "the declared deadlines assume this timestep"
    );
    bundle
}

fn manifest(faults: Vec<Fault>) -> FaultManifest {
    FaultManifest {
        format: MANIFEST_FORMAT.into(),
        version: MANIFEST_VERSION,
        faults,
    }
}

fn outage(node: &str, start_s: f64, end_s: f64) -> Fault {
    Fault::SensorOutage {
        node_id: node.into(),
        start_s,
        end_s,
    }
}

fn all_outage(bundle: &ScenarioBundle, start_s: f64, end_s: f64) -> Vec<Fault> {
    bundle
        .nodes
        .iter()
        .map(|n| outage(&n.node_id, start_s, end_s))
        .collect()
}

fn delivery(lag_s: f64, delay_s: f64) -> DeliveryProfile {
    DeliveryProfile::from_seconds(
        DT,
        lag_s,
        delay_s,
        &BTreeMap::new(),
        0.0,
        DEFAULT_QUEUE_CAPACITY,
        11,
    )
    .expect("supported")
}

/// Every frame of the run, as committed.
fn run_frames(mut run: AssembledRun, steps: u64) -> Vec<ReplayFrame> {
    (0..steps)
        .map(|_| {
            run.step().expect("tick runs");
            run.last_frame().expect("a frame").clone()
        })
        .collect()
}

fn frames_in(frames: &[ReplayFrame], from_s: f64, to_s: f64) -> Vec<&ReplayFrame> {
    frames
        .iter()
        .filter(|f| f.timestamp_s >= from_s - 1e-9 && f.timestamp_s < to_s - 1e-9)
        .collect()
}

/// Case A, a total sensor outage over [10 s, 20 s).
/// Declared: detection within one tick (every track stale at the first
/// outage frame); containment (no track is updated during the outage, and
/// none is still held once the drop window has passed: tracks are dropped,
/// not extrapolated forever); terminal state recovered (by the end of the
/// outage plus the confirmation window, at least as many fresh tracks as
/// before it). Restated after its first run: see the note at the end.
#[test]
fn a_total_sensor_outage_is_detected_contained_and_recovered() {
    let bundle = bundle();
    let (start, end) = (10.0, 20.0);
    let mut run = assemble(&bundle).unwrap();
    run.set_faults(manifest(all_outage(&bundle, start, end)))
        .unwrap();
    let frames = run_frames(run, bundle.steps);

    let before = frames_in(&frames, start - DT, start)[0];
    let tracked_before = before.tracks.iter().filter(|t| t.stale_steps == 0).count();
    assert!(
        tracked_before > 0,
        "the case needs live tracks before the outage"
    );

    let first = frames_in(&frames, start, start + DT)[0];
    assert!(
        first.tracks.iter().all(|t| t.stale_steps >= 1),
        "detection: every track stale at the first outage frame"
    );
    assert!(first.observations.is_empty());

    let mut last_counts: BTreeMap<String, u32> = before
        .tracks
        .iter()
        .map(|t| (t.track_id.clone(), t.update_count))
        .collect();
    for frame in frames_in(&frames, start, end) {
        for track in &frame.tracks {
            assert!(
                track.stale_steps >= 1,
                "containment: {} updated at {}",
                track.track_id,
                frame.timestamp_s
            );
            let previous = last_counts.insert(track.track_id.clone(), track.update_count);
            assert!(
                previous.is_none_or(|count| count >= track.update_count),
                "containment: {} gained updates during the outage",
                track.track_id
            );
        }
        if frame.timestamp_s >= start + DROP_AFTER_S + DT {
            assert!(
                frame.tracks.is_empty(),
                "containment: tracks are dropped after the stale window, not held (t={})",
                frame.timestamp_s
            );
        }
    }

    // Terminal state, restated after the first run (see the case notes in
    // DELIVERY_TIMING.md and F2's STATUS row): the first declaration assumed
    // the targets would be in view the moment the outage ended, and this
    // fixture has a natural coverage gap at 17-21 s in which the unfaulted
    // run loses its tracks too. Recovery is therefore declared against
    // evidence and against the unfaulted run, not against the clock:
    // (1) every target is tracked within the confirmation window of its first
    // accepted bearing after the outage; (2) the run ends with the same tracks
    // as the unfaulted run of the same seed -- the fault's effect is transient.
    let after = frames_in(&frames, end, f64::INFINITY);
    let mut targets: Vec<&str> = after
        .iter()
        .flat_map(|f| f.observations.iter().map(|o| o.target_id.as_str()))
        .collect();
    targets.sort_unstable();
    targets.dedup();
    assert!(!targets.is_empty(), "evidence returns after the outage");
    for target in targets {
        let first = after
            .iter()
            .find(|f| f.observations.iter().any(|o| o.target_id == target))
            .expect("listed above")
            .timestamp_s;
        assert!(
            frames_in(&frames, first, first + CONFIRM_WITHIN_S + DT)
                .iter()
                .any(|f| f.tracks.iter().any(|t| t.track_id == target)),
            "reacquisition: {target} first seen again at {first} s, untracked {CONFIRM_WITHIN_S} s later"
        );
    }
    let baseline = run_frames(assemble(&bundle).unwrap(), bundle.steps);
    let ids = |frame: &ReplayFrame| -> Vec<String> {
        let mut ids: Vec<String> = frame.tracks.iter().map(|t| t.track_id.clone()).collect();
        ids.sort();
        ids
    };
    assert_eq!(
        ids(frames.last().unwrap()),
        ids(baseline.last().unwrap()),
        "terminal state recovered: the same tracks as the unfaulted run"
    );
}

/// Case B, one drone's sensor out over [10 s, 30 s).
/// Declared: containment (the drone contributes to no track update during
/// the outage; fusion of the others continues); terminal state recovered
/// (it contributes again within the confirmation window after the outage).
#[test]
fn one_sensor_out_stops_its_contributions_and_they_resume() {
    let bundle = bundle();
    let node = "drone-east";
    let (start, end) = (10.0, 30.0);
    let mut run = assemble(&bundle).unwrap();
    run.set_faults(manifest(vec![outage(node, start, end)]))
        .unwrap();
    let frames = run_frames(run, bundle.steps);

    let contributes = |frame: &ReplayFrame| frame.observations.iter().any(|o| o.node_id == node);
    assert!(
        frames_in(&frames, 0.0, start)
            .iter()
            .any(|f| contributes(f)),
        "the case needs the drone to contribute before the outage"
    );
    for frame in frames_in(&frames, start, end) {
        assert!(
            !contributes(frame),
            "{node} contributed at {}",
            frame.timestamp_s
        );
    }
    assert!(
        frames_in(&frames, start, end)
            .iter()
            .any(|f| !f.observations.is_empty()),
        "fusion of the other sensors continued"
    );
    assert!(
        frames_in(&frames, end, end + CONFIRM_WITHIN_S)
            .iter()
            .any(|f| contributes(f)),
        "terminal state recovered: {node} contributes again within {CONFIRM_WITHIN_S} s"
    );
}

/// Case C, a link delay spike within the fusion lag.
/// Declared: no evidence lost (no delivery drops); the drone's evidence
/// is still fused, at its acquisition time.
#[test]
fn a_delay_spike_within_the_lag_loses_nothing() {
    let bundle = bundle();
    let node = "drone-east";
    let mut run = assemble(&bundle).unwrap();
    run.set_delivery(delivery(1.0, 0.0)).unwrap();
    run.set_faults(manifest(vec![Fault::LinkDelay {
        node_id: node.into(),
        start_s: 10.0,
        end_s: 20.0,
        delay_s: 1.0,
    }]))
    .unwrap();
    let frames = run_frames(run, bundle.steps);
    let drops = frames
        .iter()
        .flat_map(|f| &f.rejected_observations)
        .filter(|r| r.reason.starts_with("delivery_"))
        .count();
    assert_eq!(drops, 0);
    for frame in &frames {
        for o in &frame.observations {
            assert!((o.timestamp_s - (frame.timestamp_s - 1.0)).abs() < 1e-9);
        }
    }
    assert!(frames_in(&frames, 11.0, 21.0)
        .iter()
        .any(|f| f.observations.iter().any(|o| o.node_id == node)));
}

/// Case D, a transport restart with evidence in flight.
/// Declared: the drone's in-flight evidence is dropped as stale, and only
/// it; the drone's evidence is fused again after the restart.
#[test]
fn a_restart_drops_only_the_in_flight_evidence_of_that_source() {
    let bundle = bundle();
    let node = "drone-east";
    let at = 15.0;
    let mut run = assemble(&bundle).unwrap();
    run.set_delivery(delivery(1.5, 1.0)).unwrap();
    run.set_faults(manifest(vec![Fault::SourceRestart {
        node_id: node.into(),
        at_s: at,
    }]))
    .unwrap();
    let frames = run_frames(run, bundle.steps);
    let drops: Vec<_> = frames
        .iter()
        .flat_map(|f| &f.rejected_observations)
        .filter(|r| r.reason.starts_with("delivery_"))
        .collect();
    assert!(!drops.is_empty(), "the restart caught evidence in flight");
    for drop in &drops {
        assert_eq!(drop.reason, "delivery_stale_epoch");
        assert_eq!(drop.node_id, node);
    }
    assert!(frames_in(&frames, at + 1.5, at + 1.5 + CONFIRM_WITHIN_S)
        .iter()
        .any(|f| f.observations.iter().any(|o| o.node_id == node)));
}

/// Case E: a delay the fusion lag cannot absorb is refused before the run.
#[test]
fn a_delay_beyond_the_lag_is_refused_before_the_run() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    run.set_delivery(delivery(1.0, 0.0)).unwrap();
    let error = run
        .set_faults(manifest(vec![Fault::LinkDelay {
            node_id: "drone-east".into(),
            start_s: 5.0,
            end_s: 6.0,
            delay_s: 2.0,
        }]))
        .unwrap_err();
    assert!(error.contains("exceeds the fusion lag"), "{error}");
    let mut plain = assemble(&bundle).unwrap();
    assert!(plain
        .set_faults(manifest(vec![Fault::SourceRestart {
            node_id: "drone-east".into(),
            at_s: 1.0
        }]))
        .is_err());
}

/// Deterministic for a fixed manifest, and a checkpoint taken mid-outage
/// resumes exactly.
#[test]
fn a_faulted_run_is_deterministic_and_resumes_mid_outage() {
    let bundle = bundle();
    let make = || {
        let mut run = assemble(&bundle).unwrap();
        run.set_delivery(delivery(1.5, 0.5)).unwrap();
        let mut faults = all_outage(&bundle, 10.0, 15.0);
        faults.push(Fault::SourceRestart {
            node_id: "drone-north".into(),
            at_s: 22.0,
        });
        run.set_faults(manifest(faults)).unwrap();
        run
    };
    let json = |frames: &[ReplayFrame]| -> Vec<String> {
        frames
            .iter()
            .map(|f| serde_json::to_string(f).unwrap())
            .collect()
    };
    let straight = json(&run_frames(make(), bundle.steps));
    assert_eq!(json(&run_frames(make(), bundle.steps)), straight);

    let at = 24usize; // t = 12 s, inside the outage
    let mut original = make();
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = make();
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest = json(&run_frames(resumed, bundle.steps - at as u64));
    assert_eq!(rest, straight[at..]);
}

fn drift(node: &str, start_s: f64, end_s: f64, rate_rad_per_s: f64) -> Fault {
    Fault::CalibrationDrift {
        node_id: node.into(),
        start_s,
        end_s,
        rate_rad_per_s,
    }
}

/// Case E, a calibration drift on one sensor: 0.01 rad/s over [10 s, 20 s).
/// Declared: that sensor's bearings turn by exactly the drift so far
/// (0.1 rad by 20 s) and stay turned, since nothing recalibrates it; every
/// other sensor's observations are unchanged; the frames name the fault
/// while it drifts and carry the offset from its start on.
#[test]
fn a_calibration_drift_turns_one_sensors_bearings_and_stays() {
    let bundle = bundle();
    let node = "drone-east";
    let (start, end, rate) = (10.0, 20.0, 0.01);
    let baseline = run_frames(assemble(&bundle).unwrap(), bundle.steps);
    let mut run = assemble(&bundle).unwrap();
    run.set_faults(manifest(vec![drift(node, start, end, rate)]))
        .unwrap();
    let drifted = run_frames(run, bundle.steps);

    let offset_at = |t: f64| rate * (t.clamp(start, end) - start);
    let mut turned = 0;
    for (plain, frame) in baseline.iter().zip(&drifted) {
        let t = frame.timestamp_s;
        assert_eq!(plain.observations.len(), frame.observations.len(), "t={t}");
        for (a, b) in plain.observations.iter().zip(&frame.observations) {
            assert_eq!(a.node_id, b.node_id);
            if a.node_id != node || t < start - 1e-9 {
                assert_eq!(a.direction, b.direction, "{} at {t}", a.node_id);
                continue;
            }
            let cos = a.direction[0] * b.direction[0]
                + a.direction[1] * b.direction[1]
                + a.direction[2] * b.direction[2];
            let angle = cos.clamp(-1.0, 1.0).acos();
            assert!(
                (angle - offset_at(t)).abs() < 1e-6,
                "{node} turned {angle} at {t}, expected {}",
                offset_at(t)
            );
            turned += 1;
        }
        let active: Vec<_> = frame.active_faults.iter().collect();
        if (start..end).contains(&t) {
            assert_eq!(active, [&drift(node, start, end, rate)], "t={t}");
        } else {
            assert!(active.is_empty(), "t={t}");
        }
        let recorded = frame.sensor_bias_offsets_rad.get(node).copied();
        if t <= start + 1e-9 {
            assert_eq!(recorded, None, "t={t}");
        } else {
            assert!((recorded.unwrap() - offset_at(t)).abs() < 1e-12, "t={t}");
        }
    }
    assert!(
        turned > 10,
        "the case needs the drone to observe after the drift starts"
    );
    assert!(baseline
        .iter()
        .all(|f| f.active_faults.is_empty() && f.sensor_bias_offsets_rad.is_empty()));
}

/// A drift rate that is zero or not finite is refused before the run.
#[test]
fn a_drift_with_no_rate_is_refused() {
    let bundle = bundle();
    for rate in [0.0, f64::NAN] {
        let mut run = assemble(&bundle).unwrap();
        let error = run
            .set_faults(manifest(vec![drift("drone-east", 1.0, 2.0, rate)]))
            .unwrap_err();
        assert!(error.contains("calibration_drift"), "{error}");
    }
}

/// The drift is a pure function of the tick: a run resumed during the drift,
/// or after it, carries on exactly as the straight run.
#[test]
fn a_drifting_run_resumes_mid_drift_and_after() {
    let bundle = bundle();
    let make = || {
        let mut run = assemble(&bundle).unwrap();
        run.set_faults(manifest(vec![
            drift("drone-east", 10.0, 20.0, 0.01),
            drift("drone-north", 5.0, 30.0, -0.004),
        ]))
        .unwrap();
        run
    };
    let json = |frames: &[ReplayFrame]| -> Vec<String> {
        frames
            .iter()
            .map(|f| serde_json::to_string(f).unwrap())
            .collect()
    };
    let straight = json(&run_frames(make(), bundle.steps));
    for at in [30usize, 50] {
        let mut original = make();
        for _ in 0..at {
            original.step().unwrap();
        }
        let checkpoint = original.sim.checkpoint().unwrap();
        let mut resumed = make();
        resumed.restore(&checkpoint).expect("restore accepted");
        let rest = json(&run_frames(resumed, bundle.steps - at as u64));
        assert_eq!(rest, straight[at..], "resumed at tick {at}");
    }
}

/// A drift overlapping an outage of the same sensor goes on accumulating,
/// and the frames carry its offset throughout the outage, as before and
/// after it.
#[test]
fn a_drift_is_recorded_through_an_outage_of_its_sensor() {
    let bundle = bundle();
    let node = "drone-east";
    let mut run = assemble(&bundle).unwrap();
    run.set_faults(manifest(vec![
        drift(node, 10.0, 20.0, 0.01),
        outage(node, 12.0, 25.0),
    ]))
    .unwrap();
    let frames = run_frames(run, bundle.steps);
    let during = frames_in(&frames, 12.0, 25.0);
    assert!(!during.is_empty());
    for frame in during {
        assert!(
            frame.observations.iter().all(|o| o.node_id != node),
            "out at {}",
            frame.timestamp_s
        );
        let expected = 0.01 * (frame.timestamp_s.min(20.0) - 10.0);
        let recorded = frame.sensor_bias_offsets_rad.get(node).copied();
        assert!(
            recorded.is_some_and(|r| (r - expected).abs() < 1e-12),
            "{recorded:?} at {}",
            frame.timestamp_s
        );
    }
}
