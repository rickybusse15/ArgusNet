//! F5: `flown_v1` charges energy on where each platform flew -- after stage
//! 2's overrides, deconfliction and safety -- one tick late, instead of on
//! its controller's nominal position before them. A drone held at home by an
//! override then draws hover power, not the lanes it is not flying.

use argusnet_planning::battery::EnergyProfile;
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{
    assemble, reference_battery_model, AssembledRun, RunError, ScenarioBundle,
};
use argusnet_sim::simulation::TickError;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run(profile: EnergyProfile) -> AssembledRun {
    let bundle = bundle();
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        energy_profile: profile,
        planning_policy: PlanningPolicy::Reference,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    run
}

fn to_end(mut run: AssembledRun) -> serde_json::Value {
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
    }
    serde_json::to_value(run.run_to_replay().expect("replay")).expect("serialises")
}

/// `(position, battery fraction)` at the end of every tick, for one drone.
type Track = Vec<([f64; 3], f64)>;

/// Per drone, its [`Track`].
fn flown(replay: &serde_json::Value) -> Vec<(String, Track)> {
    let frames = replay["frames"].as_array().unwrap();
    let mut drones: Vec<(String, Track)> = Vec::new();
    for frame in frames {
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let id = node["node_id"].as_str().unwrap().to_string();
            let p = node["position"].as_array().unwrap();
            let sample = (
                [
                    p[0].as_f64().unwrap(),
                    p[1].as_f64().unwrap(),
                    p[2].as_f64().unwrap(),
                ],
                node["battery_fraction"].as_f64().unwrap(),
            );
            match drones.iter_mut().find(|(drone, _)| *drone == id) {
                Some((_, samples)) => samples.push(sample),
                None => drones.push((id, vec![sample])),
            }
        }
    }
    drones
}

/// The ticks at which the ledger's draw differs from the per-tick cost of
/// the flown motion, and how many ticks the drones sat still. Tick k draws
/// the motion from the end of tick k-1 to the end of tick k, the last tick
/// included; tick 0, with nothing flown before it, draws hover alone.
fn mismatches(replay: &serde_json::Value) -> (usize, usize) {
    let model = reference_battery_model();
    let dt = replay["meta"]["dt_s"].as_f64().unwrap_or(1.0);
    let (mut wrong, mut still) = (0, 0);
    for (_, samples) in flown(replay) {
        let first = (1.0 - samples[0].1) * model.capacity_wh;
        if (first - model.tick_cost_wh(None, samples[0].0, dt)).abs() > 1e-9 {
            wrong += 1;
        }
        for k in 1..samples.len() {
            let drawn = (samples[k - 1].1 - samples[k].1) * model.capacity_wh;
            let (from, to) = (samples[k - 1].0, samples[k].0);
            let expected = model.tick_cost_wh(Some(from), to, dt);
            if from == to {
                still += 1;
            }
            if samples[k].1 > 0.0 && (drawn - expected).abs() > 1e-9 {
                wrong += 1;
            }
        }
    }
    (wrong, still)
}

/// Under `flown_v1` every tick's draw is the cost of what the replay shows
/// was flown, so a drone sitting still draws exactly hover power. Under
/// `physical_v1` the ledger and the flown path disagree wherever an override
/// moves a drone off its nominal lane.
#[test]
fn the_ledger_charges_exactly_what_was_flown() {
    let replay = to_end(run(EnergyProfile::FlownV1));
    assert_eq!(replay["meta"]["energy_profile"], "flown_v1");
    let (wrong, still) = mismatches(&replay);
    assert_eq!(wrong, 0);
    assert!(
        still > 0,
        "the run holds a drone still, so hover is exercised"
    );

    let (wrong, _) = mismatches(&to_end(run(EnergyProfile::PhysicalV1)));
    assert!(wrong > 0, "physical_v1 charges the nominal lanes");
}

/// The profile is part of the battery's checkpoint: a `flown_v1` run resumes
/// frame for frame, and a `physical_v1` run refuses its checkpoint.
#[test]
fn a_flown_checkpoint_resumes_exactly_and_only_as_flown() {
    let whole = to_end(run(EnergyProfile::FlownV1));
    let mut source = run(EnergyProfile::FlownV1);
    for _ in 0..90 {
        source.step().expect("steps");
    }
    let checkpoint = source.sim.checkpoint().expect("not faulted");
    let mut resumed = run(EnergyProfile::FlownV1);
    resumed.restore(&checkpoint).expect("restores");
    let resumed = to_end(resumed);
    let whole = whole["frames"].as_array().unwrap();
    let resumed = resumed["frames"].as_array().unwrap();
    assert!(!resumed.is_empty() && resumed.len() < whole.len());
    assert!(whole[whole.len() - resumed.len()..] == resumed[..]);
    let refused = run(EnergyProfile::PhysicalV1)
        .restore(&checkpoint)
        .unwrap_err();
    assert!(format!("{refused:?}").contains("flown_v1"), "{refused:?}");
}
