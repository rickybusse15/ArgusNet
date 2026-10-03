//! A3.4: hold the measurements fixed, change the hidden truth, and nothing a
//! truth-blind stage produces may change.
//!
//! The scan scenario's fleet searches, maps and inspects; it has no
//! controller fed from truth (the legacy follow controllers' pursuit oracle is
//! the one documented truth path outside the capability, and the scan fleet
//! has none). Its sensors are blinded -- zero range -- so every target pair is
//! rejected at the range gate and the measurements are fixed: clutter only,
//! which does not depend on truth. Two runs then differ only in where the
//! targets truly are. Planning, mapping, mission, safety, fusion and the
//! fleet's motion must agree bit for bit.

use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle, TargetDocument, TruthSample};

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

/// The scan scenario with two targets circling inside its map -- the second
/// truth circles the other way about a different centre -- and every sensor
/// blind.
fn blind_scan_with_targets(variant: f64) -> ScenarioBundle {
    let mut scan = bundle("scan");
    let ticks = scan.steps as usize + 1;
    let (centre, turn) = if variant == 0.0 {
        ([0.0, 0.0], 1.0)
    } else {
        ([variant, -variant], -1.0)
    };
    scan.targets = (0..2)
        .map(|index| {
            let radius = 120.0 + 60.0 * index as f64;
            let rate = turn * 0.02;
            TargetDocument {
                target_id: format!("target-{index}"),
                behavior: "unknown".to_string(),
                samples: (0..ticks)
                    .map(|tick| {
                        let t = tick as f64 * scan.dt_s;
                        let angle = rate * t + index as f64;
                        TruthSample {
                            t,
                            position: [
                                centre[0] + radius * angle.cos(),
                                centre[1] + radius * angle.sin(),
                                200.0,
                            ],
                            velocity: [
                                -radius * rate * angle.sin(),
                                radius * rate * angle.cos(),
                                0.0,
                            ],
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    for node in &mut scan.nodes {
        node.sensor.max_range_m = 0.0;
    }
    scan
}

/// Everything a truth-blind stage produces on one tick, as bits.
#[derive(Debug, PartialEq)]
struct Blind {
    tick: u64,
    platforms: Vec<(String, Vec<u64>, Option<u64>)>,
    tracks: usize,
    observations: Vec<Vec<u64>>,
    phase: Option<String>,
    coverage: Option<u64>,
    safety_events: usize,
}

fn step(run: &mut AssembledRun) -> Blind {
    let snapshot = run.step().expect("tick runs");
    let frame = run.fusion().frame().expect("stage 7 ran");
    Blind {
        tick: snapshot.header.tick,
        platforms: snapshot
            .platforms
            .iter()
            .map(|p| {
                (
                    p.node_id.clone(),
                    [
                        p.position.x,
                        p.position.y,
                        p.position.z,
                        p.velocity.x,
                        p.velocity.y,
                        p.velocity.z,
                    ]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect(),
                    p.battery_fraction.map(f64::to_bits),
                )
            })
            .collect(),
        tracks: frame.tracks.len(),
        observations: frame
            .observations
            .iter()
            .map(|o| {
                [
                    o.origin.x,
                    o.origin.y,
                    o.origin.z,
                    o.direction.x,
                    o.direction.y,
                    o.direction.z,
                ]
                .iter()
                .map(|v| v.to_bits())
                .collect()
            })
            .collect(),
        phase: run.mission().map(|m| m.phase().to_string()),
        coverage: run.mapping().map(|m| m.coverage_fraction().to_bits()),
        safety_events: frame.safety_events.len(),
    }
}

#[test]
fn changing_hidden_truth_with_measurements_fixed_changes_nothing_downstream() {
    let ticks = 400;
    let mut here = assemble(&blind_scan_with_targets(0.0)).expect("assembles");
    let mut there = assemble(&blind_scan_with_targets(80.0)).expect("assembles");
    let mut moved = false;
    for _ in 0..ticks {
        let a = step(&mut here);
        let b = step(&mut there);
        assert_eq!(
            a, b,
            "truth leaked into a truth-blind stage at tick {}",
            a.tick
        );
        moved |= a
            .platforms
            .iter()
            .any(|(_, bits, _)| bits[3] != 0f64.to_bits());
    }
    // Guard against a vacuous pass: the fleet flew, and the two runs really
    // did hold different truth.
    assert!(moved, "the fleet never moved");
    assert_ne!(
        blind_scan_with_targets(0.0).targets[0].samples[0].position,
        blind_scan_with_targets(80.0).targets[0].samples[0].position
    );
}

/// The control: with the sensors seeing, the same two truths do produce
/// different measurements -- so the equality above is the measurements being
/// held fixed, not the targets being out of reach.
#[test]
fn with_sensors_seeing_the_two_truths_are_measured_differently() {
    let seeing = |shift_m: f64| {
        let mut sighted = blind_scan_with_targets(shift_m);
        for (node, original) in sighted.nodes.iter_mut().zip(bundle("scan").nodes) {
            node.sensor.max_range_m = original.sensor.max_range_m;
        }
        sighted
    };
    let mut here = assemble(&seeing(0.0)).expect("assembles");
    let mut there = assemble(&seeing(80.0)).expect("assembles");
    let differs = (0..400).any(|_| step(&mut here).observations != step(&mut there).observations);
    assert!(differs, "the targets were never in reach of a sensor");
}
