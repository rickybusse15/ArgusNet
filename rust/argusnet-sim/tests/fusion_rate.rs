//! B2.6: the fusion lane at its own rate in an assembled run.
//!
//! Stage 7 keeps every boundary it sits out and fuses it, in order, when it
//! next runs, so the tracks at each boundary it runs are the every-boundary
//! run's -- with timestamped delivery modelled too. In a run with a prior
//! world nothing else reads the tracks between its runs, so the runs agree
//! exactly at those boundaries.

use argusnet_sim::run_options::{DeliverySeconds, RunOptions};
use argusnet_sim::scenario::{assemble, AssembledRun, RunError, ScenarioBundle};
use argusnet_sim::scheduler::{parse_rate, LaneRates, PhasedLaneExecutor};
use argusnet_sim::simulation::TickError;

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

/// Every boundary's tracks (their `Debug` form, which prints every float in
/// full), stepping through the lanes executor.
fn tracks_per_boundary(
    case: &str,
    rates: LaneRates,
    delivery: Option<DeliverySeconds>,
) -> Vec<String> {
    let bundle = bundle(case);
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        lane_rates: rates,
        delivery,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    let mut lanes = rates
        .apply(PhasedLaneExecutor::new::<RunError>(bundle.dt_s, 0).expect("exact timestep"))
        .expect("rates");
    let mut tracks = Vec::new();
    loop {
        match lanes.step(&mut run) {
            Ok(_) => tracks.push(format!("{:?}", run.sim.world().tracks())),
            Err(argusnet_sim::scheduler::ExecutorError::Tick(RunError::Tick(
                TickError::Completed { .. },
            ))) => break,
            Err(error) => panic!("{error:?}"),
        }
    }
    let _: &AssembledRun = &run;
    tracks
}

/// Fusion at one run every third boundary: `1000 / (3 * dt_ms)` Hz.
fn fusion_every_third_boundary(case: &str) -> LaneRates {
    let dt_ms = (bundle(case).dt_s * 1000.0).round() as u64;
    LaneRates {
        fusion: Some(parse_rate(&format!("1000/{}", 3 * dt_ms)).unwrap()),
        ..LaneRates::default()
    }
}

fn check(case: &str, delivery: Option<DeliverySeconds>) {
    let every = tracks_per_boundary(case, LaneRates::default(), delivery.clone());
    let rated = tracks_per_boundary(case, fusion_every_third_boundary(case), delivery);
    assert_eq!(every.len(), rated.len());
    assert!(
        every.iter().any(|t| t != "[]"),
        "no tracks, so nothing compared"
    );
    let mut differed = false;
    for (n, (a, b)) in every.iter().zip(&rated).enumerate() {
        if n % 3 == 0 {
            assert_eq!(a, b, "{case}: boundary {n}, where fusion ran");
        } else {
            differed |= a != b;
        }
    }
    assert!(
        differed,
        "{case}: the boundaries fusion sat out never showed stale tracks"
    );
}

#[test]
fn tracks_catch_up_exactly_when_the_fusion_lane_runs() {
    check("tt", None);
}

#[test]
fn tracks_catch_up_exactly_with_delivery_modelled() {
    let mut delivery = DeliverySeconds::with_lag(2.0);
    delivery.delay_s = 1.0;
    check("tt", Some(delivery));
}
