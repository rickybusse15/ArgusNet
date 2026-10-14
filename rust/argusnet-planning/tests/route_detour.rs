//! P3: a detour that leaves the start-to-goal corridor is checked against
//! every hard obstacle, not only the ones the corridor query found.

use argusnet_planning::planner::{PlannerConfig, RoutePlanner, Scene};
use argusnet_planning::route::segment_is_free;
use argusnet_types::Bounds2d;
use argusnet_world::index::ObstacleLayer;
use argusnet_world::obstacles::{BlockerType, Obstacle};

fn building(id: &str, x: f64, y: f64) -> Obstacle {
    Obstacle::oriented_box(id, BlockerType::Building, x, y, 40.0, 40.0, 0.0, 0.0, 120.0)
        .expect("valid building")
}

/// A 400 m wall across the direct path at x = 300, and a building on each
/// diagonal leg of the detour round either end of it. The buildings sit
/// outside the corridor's padded window (|y| <= 80) and its 50 m query
/// cells, so the corridor query never sees them, and the detour it plans
/// cuts through them.
#[test]
fn an_uncached_route_avoids_obstacles_outside_the_corridor() {
    let wall = Obstacle::wall_segment(
        "wall",
        BlockerType::Wall,
        [300.0, -200.0],
        [300.0, 200.0],
        10.0,
        0.0,
        120.0,
    )
    .expect("valid wall");
    let buildings = [
        building("sw", 240.0, -156.0),
        building("se", 360.0, -156.0),
        building("nw", 240.0, 156.0),
        building("ne", 360.0, 156.0),
    ];
    let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
    let mut obstacles = vec![wall];
    obstacles.extend(buildings.iter().cloned());
    let layer = ObstacleLayer::new(bounds, 50.0, obstacles).expect("valid layer");
    let scene = Scene {
        layer: &layer,
        revision: 0,
    };
    let mut planner = RoutePlanner::new(bounds, PlannerConfig::default());
    let route = planner
        .plan_route_uncached(scene, [100.0, 0.0], [500.0, 0.0], 8.0)
        .expect("a route round everything");
    // The buildings' own footprints, with no clearance added.
    let footprints: Vec<_> = buildings
        .iter()
        .map(|b| planner.expanded_polygon(scene, b, 0.0))
        .collect();
    for pair in route.points_xy_m.windows(2) {
        assert!(
            segment_is_free(pair[0], pair[1], &footprints),
            "{pair:?} crosses a building: {:?}",
            route.points_xy_m
        );
    }
}
