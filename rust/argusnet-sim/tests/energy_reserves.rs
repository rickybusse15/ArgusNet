//! F5: under a physical energy profile the replay records, per platform and
//! tick, its remaining energy and the reserve it needs to get home, as the
//! battery's own reserve check computed them. Legacy replays are unchanged.

use argusnet_planning::battery::EnergyProfile;
use argusnet_sim::run_options::RunOptions;
use argusnet_sim::scenario::{assemble, reference_battery_model, ScenarioBundle};

fn replay(profile: EnergyProfile) -> serde_json::Value {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let bundle = ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses");
    let mut run = assemble(&bundle).expect("assembles");
    RunOptions {
        energy_profile: profile,
        ..RunOptions::default()
    }
    .apply(&mut run, &bundle)
    .expect("applies");
    serde_json::to_value(run.run_to_replay().expect("runs")).expect("serialises")
}

#[test]
fn a_physical_replay_records_each_drones_reserve_and_legacy_none() {
    let model = reference_battery_model();
    let floor = model.capacity_wh * model.reserve_fraction;
    let physical = replay(EnergyProfile::PhysicalV1);
    let mut seen = 0;
    for frame in physical["frames"].as_array().unwrap() {
        let reserves = frame["energy_reserves"].as_object().expect("every tick");
        for node in frame["nodes"].as_array().unwrap() {
            let id = node["node_id"].as_str().unwrap();
            let Some(reserve) = reserves.get(id) else {
                assert_ne!(node["is_mobile"], true, "{id} has no reserve");
                continue;
            };
            let remaining = reserve["remaining_wh"].as_f64().unwrap();
            let needed = reserve["reserve_wh"].as_f64().unwrap();
            let fraction = node["battery_fraction"].as_f64().unwrap();
            assert!(
                (remaining - fraction * model.capacity_wh).abs() < 1e-9,
                "{id}"
            );
            assert!(needed >= floor - 1e-9, "{id}: {needed} below the floor");
            seen += 1;
        }
    }
    assert!(seen > 100, "both drones, most ticks: {seen}");

    let legacy = replay(EnergyProfile::Legacy);
    assert!(legacy["frames"]
        .as_array()
        .unwrap()
        .iter()
        .all(|frame| frame.get("energy_reserves").is_none()));
}
