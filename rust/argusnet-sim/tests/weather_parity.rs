//! `weather.rs` against the reference's `WindModel` and `flight_speed_penalty`.
//!
//! Fixture from `scripts/capture_weather_fixture.py`: every preset, plus the
//! storm, no-gust and floored-speed cases, with the parameters written by the
//! bundle exporter itself -- so this also proves the exporter's weather
//! document deserializes into the model it describes.

use argusnet_sim::trajectory::WeatherSampler;
use argusnet_sim::weather::MotionWeather;
use serde_json::Value;

fn num(value: &Value) -> f64 {
    value
        .as_str()
        .expect("stringified float")
        .parse()
        .expect("parses")
}

#[test]
fn every_preset_matches_the_reference_bit_for_bit() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/weather.json"
    );
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(cases.len() >= 10, "the fixture lost cases");
    let mut checked = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let weather: MotionWeather =
            serde_json::from_value(case["parameters"].clone()).expect("exporter document parses");
        weather
            .precipitation
            .validate()
            .expect("known precipitation");
        for sample in case["winds"].as_array().unwrap() {
            let (altitude, time) = (num(&sample["altitude_m"]), num(&sample["time_s"]));
            let wind = weather.wind_at(altitude, time);
            let expected: Vec<f64> = sample["wind"].as_array().unwrap().iter().map(num).collect();
            assert_eq!(
                wind.map(f64::to_bits).to_vec(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}: wind_at({altitude}, {time}) = {wind:?}, reference {expected:?}"
            );
            checked += 1;
        }
        for sample in case["penalties"].as_array().unwrap() {
            let altitude = num(&sample["altitude_m"]);
            let penalty = weather.flight_speed_penalty(altitude);
            let expected = num(&sample["penalty"]);
            assert_eq!(
                penalty.to_bits(),
                expected.to_bits(),
                "{name}: penalty({altitude}) = {penalty}, reference {expected}"
            );
            checked += 1;
        }
    }
    assert!(checked > 500, "only {checked} samples");
}

#[test]
fn an_unknown_precipitation_type_is_refused() {
    let weather: MotionWeather = serde_json::from_value(serde_json::json!({
        "wind": {"base_speed_mps": 1.0, "base_heading_rad": 0.0, "gust_amplitude_mps": 0.0,
                 "gust_period_s": 10.0, "altitude_scaling": 0.0},
        "precipitation": {"precip_type": "hail", "rate_mmph": 1.0}
    }))
    .expect("shape is fine");
    assert!(weather.precipitation.validate().is_err());
}
