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

/// A wall reaching the map's edge leaves no way round inside the world: no
/// route, rather than one through the expanded corners past the edge.
#[test]
fn an_uncached_route_never_leaves_the_bounds() {
    let wall = Obstacle::wall_segment(
        "wall",
        BlockerType::Wall,
        [300.0, -1000.0],
        [300.0, 1000.0],
        10.0,
        0.0,
        120.0,
    )
    .expect("valid wall");
    let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
    let layer = ObstacleLayer::new(bounds, 50.0, vec![wall]).expect("valid layer");
    let scene = Scene {
        layer: &layer,
        revision: 0,
    };
    let mut planner = RoutePlanner::new(bounds, PlannerConfig::default());
    assert!(planner
        .plan_route_uncached(scene, [100.0, 0.0], [500.0, 0.0], 8.0)
        .is_none());
}

/// A POI 3 m from the wall's far face, inside the 8 m clearance:
/// `plan_route_uncached` refuses it, `plan_route_between` routes round the
/// wall to the nearest clear point and finishes with a short straight leg,
/// clear of the wall's own footprint.
#[test]
fn a_route_reaches_an_end_inside_the_clearance() {
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
    let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
    let layer = ObstacleLayer::new(bounds, 50.0, vec![wall]).expect("valid layer");
    let scene = Scene {
        layer: &layer,
        revision: 0,
    };
    let mut planner = RoutePlanner::new(bounds, PlannerConfig::default());
    let (start, goal) = ([100.0, 0.0], [308.0, 0.0]);
    assert!(planner
        .plan_route_uncached(scene, start, goal, 8.0)
        .is_none());
    let route = planner
        .plan_route_between(scene, start, goal, 8.0)
        .expect("a route to the POI by the wall");
    assert_eq!(route.points_xy_m.first(), Some(&start));
    assert_eq!(route.points_xy_m.last(), Some(&goal));
    assert!(route.points_xy_m.len() > 3, "round the wall");
    let footprint = planner.expanded_polygon(scene, &layer.primitives[0], 0.0);
    for pair in route.points_xy_m.windows(2) {
        assert!(
            segment_is_free(pair[0], pair[1], std::slice::from_ref(&footprint)),
            "{pair:?} crosses the wall: {:?}",
            route.points_xy_m
        );
    }

    // Nothing in the way: the straight line, not a detour.
    let straight = planner
        .plan_route_between(scene, [100.0, 300.0], [500.0, 300.0], 8.0)
        .expect("open ground");
    assert_eq!(straight.points_xy_m.len(), 2);

    // Both ends 3 m from the wall's face, 100 m apart along it: the route
    // keeps the clearance between the clear points, with only the two short
    // legs inside it, not a straight line hugging the wall.
    let along = planner
        .plan_route_between(scene, [308.0, 0.0], [308.0, 100.0], 8.0)
        .expect("along the wall");
    let inner = &along.points_xy_m[1..along.points_xy_m.len() - 1];
    assert!(inner.len() >= 2, "{:?}", along.points_xy_m);
    for point in inner {
        assert!(
            point[0] >= 313.0 - 1e-9,
            "inside the clearance: {:?}",
            along.points_xy_m
        );
    }

    // An end on the wall's face is inside the wall: no route.
    assert!(planner
        .plan_route_between(scene, start, [305.0, 0.0], 8.0)
        .is_none());
}

/// A round building is checked as its circle, not as the polygon the
/// planner inscribes in it: an end at radius 9.9 m and 15° round a 10 m
/// tower lies between that polygon's chord and the curve, and is refused.
#[test]
fn an_end_inside_a_round_building_is_refused() {
    let tower = Obstacle::cylinder("tower", BlockerType::Building, 300.0, 0.0, 10.0, 0.0, 120.0)
        .expect("valid tower");
    let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
    let layer = ObstacleLayer::new(bounds, 50.0, vec![tower]).expect("valid layer");
    let scene = Scene {
        layer: &layer,
        revision: 0,
    };
    let mut planner = RoutePlanner::new(bounds, PlannerConfig::default());
    let angle = 15f64.to_radians();
    let inside = [300.0 + 9.9 * angle.cos(), 9.9 * angle.sin()];
    assert!(layer.primitives[0].point_inside(inside[0], inside[1], 10.0));
    assert!(planner
        .plan_route_between(scene, [100.0, 0.0], inside, 8.0)
        .is_none());
    // Just outside the circle, at the same bearing, is reached.
    let outside = [300.0 + 10.5 * angle.cos(), 10.5 * angle.sin()];
    assert!(planner
        .plan_route_between(scene, [100.0, 0.0], outside, 8.0)
        .is_some());
}

/// A tower large enough that its inscribed routing polygon plus the clearance
/// still falls short of the curve: an end there, which needs no moving by the
/// planner's own test, is refused all the same.
#[test]
fn an_end_inside_a_large_round_building_is_refused() {
    let tower = Obstacle::cylinder("tower", BlockerType::Building, 0.0, 0.0, 300.0, 0.0, 120.0)
        .expect("valid tower");
    let bounds = Bounds2d::new(-1000.0, 1000.0, -1000.0, 1000.0);
    let layer = ObstacleLayer::new(bounds, 50.0, vec![tower]).expect("valid layer");
    let scene = Scene {
        layer: &layer,
        revision: 0,
    };
    let mut planner = RoutePlanner::new(bounds, PlannerConfig::default());
    let angle = 15f64.to_radians();
    let inside = [299.0 * angle.cos(), 299.0 * angle.sin()];
    assert!(layer.primitives[0].point_inside(inside[0], inside[1], 10.0));
    assert!(planner
        .plan_route_between(scene, [-800.0, 0.0], inside, 8.0)
        .is_none());
}
