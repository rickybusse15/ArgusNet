//! The mission driver's reference primitives, from
//! `scripts/capture_mission_driver_fixture.py`: `coverage_in_region` against a
//! real `WorldMap`, and the two distances whose last bit decides a boundary
//! (the 30 m dwell radius and the redirect's flight time) against the
//! reference's `np.linalg.norm`.

use argusnet_mapping::GridBounds;
use argusnet_planning::poi::{InspectionPoi, PoiAssignmentContext, PoiManager};
use argusnet_sim::mission_driver::coverage_in_grid_region;
use argusnet_sim::overrides::GotoHover;
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mission_driver.json"
    );
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .expect("regenerate with scripts/capture_mission_driver_fixture.py"),
    )
    .expect("json")
}

fn num(value: &Value) -> f64 {
    value.as_str().expect("stringified").parse().expect("float")
}

#[test]
fn coverage_in_region_matches_the_reference_world_map() {
    let fixture = fixture();
    let region = &fixture["coverage_in_region"];
    let b = &region["bounds"];
    let bounds = GridBounds {
        x_min_m: num(&b["x_min_m"]),
        x_max_m: num(&b["x_max_m"]),
        y_min_m: num(&b["y_min_m"]),
        y_max_m: num(&b["y_max_m"]),
        resolution_m: num(&b["resolution_m"]),
    };
    let counts: Vec<Vec<u32>> = serde_json::from_value(region["counts"].clone()).expect("counts");
    let queries = region["queries"].as_array().expect("queries");
    assert!(queries.len() >= 150);
    for q in queries {
        let center = q["center"].as_array().unwrap();
        let fraction = coverage_in_grid_region(
            bounds,
            |i, j| counts[i][j],
            [num(&center[0]), num(&center[1])],
            num(&q["radius_m"]),
        );
        assert_eq!(fraction.to_bits(), num(&q["fraction"]).to_bits(), "{q}");
    }
}

/// `sqrt(x^2 + y^2)` is exactly 30.0 for this offset and `hypot` is one ULP
/// above it. The reference credits the dwell, so the port must too.
#[test]
fn the_dwell_boundary_is_decided_by_numpy_norm() {
    let fixture = fixture();
    let d = &fixture["dwell_boundary"];
    let position = |v: &Value| {
        let a = v.as_array().unwrap();
        [num(&a[0]), num(&a[1]), num(&a[2])]
    };
    let mut manager = PoiManager::new(vec![InspectionPoi {
        poi_id: "poi-00".into(),
        name: "Site A".into(),
        position: position(&d["poi_position"]),
        priority: 1,
        required_dwell_s: 15.0,
    }]);
    let drone = position(&d["drone_position"]);
    manager.assign_energy_aware(&PoiAssignmentContext::new("drone-1", drone, 500.0, 500.0));
    manager.accumulate_dwell("drone-1", drone, num(&d["dwell_s"]));
    let completed = manager.check_completions(num(&d["dwell_s"]));
    let expected: Vec<String> = serde_json::from_value(d["completed"].clone()).unwrap();
    assert_eq!(completed, expected);
    assert!(
        !expected.is_empty(),
        "the fixture no longer sits on the boundary"
    );
}

/// The redirect's flight time is `norm / speed`, with the reference's norm.
#[test]
fn the_redirect_flight_time_uses_numpy_norm() {
    let fixture = fixture();
    let offset = fixture["norm"]["offset"].as_array().unwrap();
    let (dx, dy) = (num(&offset[0]), num(&offset[1]));
    let norm = num(&fixture["norm"]["norm"]);
    // Starting at t = 0 at 1 m/s makes the flight time the distance itself,
    // so no addition or division can round the last bit away.
    let hover = GotoHover::redirect([0.0, 0.0, 50.0], [dx, dy], 60.0, 0.0, 1.0);
    assert_eq!(
        hover.t_arrive().to_bits(),
        norm.to_bits(),
        "flight distance {} is not the reference's norm {norm}",
        hover.t_arrive()
    );
}
