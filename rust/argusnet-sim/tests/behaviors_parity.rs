//! Target behaviours against `build_target_trajectory` (P4.9b.2).
//!
//! `scripts/capture_behaviors_fixture.py` samples every preset over three
//! bounds, four seeds each, at times that cross every internal boundary. The
//! capture computes `uniform` unfused (see its docstring), so the draws are
//! platform-independent. The trajectories still call libm `sin`/`cos` and
//! numpy's `norm`, so values are bit for bit on the capture platform and
//! within a few ULPs elsewhere (`common::parity`).

mod common;

use argusnet_sim::builder::behaviors::{build_target_behavior, Bounds};
use common::parity::parity_of;
use serde_json::Value;

#[test]
fn every_preset_matches_the_reference() {
    let path = format!(
        "{}/../../tests/fixtures/behaviors.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let parity = parity_of(&fixture);
    let terrain = |x: f64, y: f64| 40.0 + 0.05 * x - 0.03 * y;
    let mut compared = 0usize;
    let mut presets = std::collections::BTreeSet::new();
    for case in fixture["cases"].as_array().unwrap() {
        let preset = case["preset"].as_str().unwrap();
        presets.insert(preset.to_string());
        let b = &case["bounds"];
        let bounds = Bounds {
            x_min_m: b["x_min_m"].as_f64().unwrap(),
            x_max_m: b["x_max_m"].as_f64().unwrap(),
            y_min_m: b["y_min_m"].as_f64().unwrap(),
            y_max_m: b["y_max_m"].as_f64().unwrap(),
        };
        let seed = case["seed"].as_u64().unwrap() as u32;
        let behavior = build_target_behavior(
            preset,
            bounds,
            case["altitude_m"].as_f64().unwrap(),
            case["speed_mps"].as_f64().unwrap(),
            seed,
        )
        .unwrap();
        for sample in case["samples"].as_array().unwrap() {
            let v: Vec<f64> = sample
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let (position, velocity) = behavior.sample(v[0], &terrain);
            let got = [position, velocity].concat();
            for (k, (g, w)) in got.iter().zip(&v[1..]).enumerate() {
                assert!(
                    parity.same(*g, *w),
                    "{preset} seed {seed} bounds {b} t={}: component {k} is {g}, reference {w}",
                    v[0]
                );
            }
            compared += 1;
        }
    }
    assert_eq!(presets.len(), 8, "every preset: {presets:?}");
    assert!(compared > 10_000, "{compared}");
}
