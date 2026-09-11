//! Parity for `PathPlanner2D`'s public routing surface.
//!
//! Fixture from `scripts/capture_plan_route_fixture.py`, which drives the
//! reference and reads its results back. Scenes arrive as the metadata
//! `to_metadata()` emits and are rebuilt here through
//! `ObstacleMetadata::into_obstacle` — the same conversion the native bridge
//! uses — so an oriented box's or a wall's footprint construction is exercised
//! rather than assumed.
//!
//! Geometry is compared at `1e-9` relative: the reference measures distance
//! with `np.linalg.norm`, whose summation need not match `sqrt(dx*dx+dy*dy)`.
//! Vertex counts, skip counts and cache flags are integers and booleans and
//! are compared exactly.
//!
//! **One case set is tagged `divergence`.** The route cache key carries a
//! scene revision here and does not in the reference; see `plan_route`'s
//! module docs. The reference's cache behaviour for a FIXED scene is recorded
//! as agreement and reproduced; what the port adds has nothing to compare
//! against, so it is asserted directly.

use argusnet_planning::planner::{PlannerConfig, PlannerRoute, RoutePlanner, Scene};
use argusnet_planning::route::Point2;
use argusnet_types::Bounds2d;
use argusnet_world::index::ObstacleLayer;
use argusnet_world::obstacles::ObstacleMetadata;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    config: ConfigSpec,
    scenes: Vec<SceneSpec>,
    scenes_wide: Vec<SceneSpec>,
    nearest_free_point: Vec<NearestCase>,
    plan_route: Vec<RouteCase>,
    route_waypoints: Vec<WaypointCase>,
    cache_sequence: CacheSequence,
    cache_eviction: EvictionSequence,
    cache_quantization: QuantizationSequence,
    snap_floor: SnapFloor,
    cylinder_vertex_floor: CylinderFloor,
}

#[derive(Deserialize)]
struct SnapFloor {
    scene: String,
    snap_m: String,
    calls: Vec<HitCall>,
}

#[derive(Deserialize)]
struct CylinderFloor {
    scene: String,
    cylinder_vertex_count: usize,
    clearance_m: String,
    expected: Vec<[String; 2]>,
}

#[derive(Deserialize)]
struct EvictionSequence {
    max_cache_entries: usize,
    clearance_m: String,
    scene: String,
    calls: Vec<HitCall>,
}

#[derive(Deserialize)]
struct QuantizationSequence {
    scene: String,
    calls: Vec<QuantCall>,
}

#[derive(Deserialize)]
struct HitCall {
    start: [String; 2],
    goal: [String; 2],
    note: String,
    cache_hit: bool,
}

#[derive(Deserialize)]
struct QuantCall {
    start: [String; 2],
    goal: [String; 2],
    clearance_m: String,
    note: String,
    cache_hit: bool,
}

#[derive(Deserialize)]
struct ConfigSpec {
    drone_clearance_m: String,
    target_clearance_m: String,
    snap_m: String,
    cylinder_vertex_count: usize,
    turn_penalty_m: String,
    max_cache_entries: usize,
}

#[derive(Deserialize)]
struct BoundsSpec {
    x_min_m: String,
    x_max_m: String,
    y_min_m: String,
    y_max_m: String,
}

#[derive(Deserialize)]
struct SceneSpec {
    name: String,
    clearance_m: String,
    obstacles: Vec<ObstacleMetadata>,
    tile_size_m: String,
    bounds: BoundsSpec,
    polygons: Vec<PolygonSpec>,
}

#[derive(Deserialize)]
struct PolygonSpec {
    primitive_id: String,
    blocker_type: String,
    vertices: Vec<[String; 2]>,
}

#[derive(Deserialize)]
struct NearestCase {
    scene: String,
    note: String,
    point: [String; 2],
    clearance_m: String,
    expected: [String; 2],
    moved: bool,
}

#[derive(Deserialize)]
struct RouteExpectation {
    points: Vec<[String; 2]>,
    length_m: String,
    vertex_count: usize,
    cache_hit: bool,
    skipped_segments: usize,
}

#[derive(Deserialize)]
struct RouteCase {
    scene: String,
    note: String,
    start: [String; 2],
    goal: [String; 2],
    clearance_m: String,
    expected: Option<RouteExpectation>,
}

#[derive(Deserialize)]
struct WaypointCase {
    scene: String,
    note: String,
    anchors: Vec<[String; 2]>,
    clearance_m: String,
    closed: bool,
    expected: Option<RouteExpectation>,
}

#[derive(Deserialize)]
struct CacheSequence {
    tag: String,
    calls: Vec<CacheCall>,
}

#[derive(Deserialize)]
struct CacheCall {
    start: [String; 2],
    goal: [String; 2],
    clearance_m: String,
    note: String,
    expected: RouteExpectation,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn point(values: &[String; 2]) -> Point2 {
    [f(&values[0]), f(&values[1])]
}

fn points(values: &[[String; 2]]) -> Vec<Point2> {
    values.iter().map(point).collect()
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/plan_route.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1.0e-9 * scale,
        "{what}: {actual:?} != {expected:?}"
    );
}

fn assert_points_close(actual: &[Point2], expected: &[Point2], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: point count");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_close(a[0], e[0], &format!("{what}[{index}].x"));
        assert_close(a[1], e[1], &format!("{what}[{index}].y"));
    }
}

fn assert_route(actual: &PlannerRoute, expected: &RouteExpectation, what: &str) {
    assert_points_close(&actual.points_xy_m, &points(&expected.points), what);
    assert_close(
        actual.length_m,
        f(&expected.length_m),
        &format!("{what}.length"),
    );
    assert_eq!(
        actual.vertex_count, expected.vertex_count,
        "{what}.vertex_count"
    );
    assert_eq!(actual.cache_hit, expected.cache_hit, "{what}.cache_hit");
    assert_eq!(
        actual.skipped_segments, expected.skipped_segments,
        "{what}.skipped_segments"
    );
}

fn config_of(spec: &ConfigSpec) -> PlannerConfig {
    PlannerConfig {
        drone_clearance_m: f(&spec.drone_clearance_m),
        target_clearance_m: f(&spec.target_clearance_m),
        snap_m: f(&spec.snap_m),
        cylinder_vertex_count: spec.cylinder_vertex_count,
        turn_penalty_m: f(&spec.turn_penalty_m),
        max_cache_entries: spec.max_cache_entries,
    }
}

fn bounds_of(spec: &BoundsSpec) -> Bounds2d {
    Bounds2d::new(
        f(&spec.x_min_m),
        f(&spec.x_max_m),
        f(&spec.y_min_m),
        f(&spec.y_max_m),
    )
}

fn layer_of(spec: &SceneSpec) -> ObstacleLayer {
    let obstacles = spec
        .obstacles
        .iter()
        .map(|m| {
            m.clone()
                .into_obstacle()
                .unwrap_or_else(|e| panic!("scene {}: {e:?}", spec.name))
        })
        .collect();
    ObstacleLayer::new(bounds_of(&spec.bounds), f(&spec.tile_size_m), obstacles)
        .unwrap_or_else(|e| panic!("scene {}: {e}", spec.name))
}

fn scene_named<'a>(fixture: &'a Fixture, name: &str) -> &'a SceneSpec {
    fixture
        .scenes
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("scene {name} missing"))
}

/// Every scene is built at revision 1; the revision only matters where a test
/// deliberately changes it.
const REVISION: u64 = 1;

#[test]
fn expanded_footprints_match_python() {
    let fixture = load();
    for spec in fixture.scenes.iter().chain(fixture.scenes_wide.iter()) {
        let layer = layer_of(spec);
        let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
        let scene = Scene {
            layer: &layer,
            revision: REVISION,
        };
        assert_eq!(
            layer.primitives.len(),
            spec.polygons.len(),
            "scene {}: obstacle count",
            spec.name
        );
        for expected in &spec.polygons {
            let obstacle = layer
                .primitives
                .iter()
                .find(|o| o.id == expected.primitive_id)
                .unwrap_or_else(|| panic!("{} missing from {}", expected.primitive_id, spec.name))
                .clone();
            assert_eq!(
                obstacle.blocker_type.as_str(),
                expected.blocker_type,
                "{}: blocker type",
                expected.primitive_id
            );
            let actual = planner.expanded_polygon(scene, &obstacle, f(&spec.clearance_m));
            assert_points_close(
                &actual,
                &points(&expected.vertices),
                &format!(
                    "{}/{} @ {}",
                    spec.name, expected.primitive_id, spec.clearance_m
                ),
            );
        }
    }
}

#[test]
fn nearest_free_point_matches_python() {
    let fixture = load();
    for case in &fixture.nearest_free_point {
        let spec = scene_named(&fixture, &case.scene);
        let layer = layer_of(spec);
        let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
        let scene = Scene {
            layer: &layer,
            revision: REVISION,
        };
        let actual = planner.nearest_free_point(scene, point(&case.point), f(&case.clearance_m));
        assert_points_close(
            &[actual],
            &[point(&case.expected)],
            &format!("nearest_free_point[{}]", case.note),
        );
        let moved = actual != point(&case.point);
        assert_eq!(
            moved, case.moved,
            "nearest_free_point[{}]: moved",
            case.note
        );
    }
}

#[test]
fn plan_route_matches_python() {
    let fixture = load();
    for case in &fixture.plan_route {
        let spec = scene_named(&fixture, &case.scene);
        let layer = layer_of(spec);
        let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
        let scene = Scene {
            layer: &layer,
            revision: REVISION,
        };
        let actual = planner.plan_route(
            scene,
            point(&case.start),
            point(&case.goal),
            f(&case.clearance_m),
        );
        match (&actual, &case.expected) {
            (None, None) => {}
            (Some(_), None) => panic!("plan_route[{}]: expected None, got a route", case.note),
            (None, Some(_)) => panic!("plan_route[{}]: expected a route, got None", case.note),
            (Some(route), Some(expected)) => {
                assert_route(route, expected, &format!("plan_route[{}]", case.note))
            }
        }
    }
}

#[test]
fn route_waypoints_matches_python() {
    let fixture = load();
    for case in &fixture.route_waypoints {
        let spec = scene_named(&fixture, &case.scene);
        let layer = layer_of(spec);
        let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
        let scene = Scene {
            layer: &layer,
            revision: REVISION,
        };
        let actual = planner.route_waypoints(
            scene,
            &points(&case.anchors),
            f(&case.clearance_m),
            case.closed,
        );
        match (&actual, &case.expected) {
            (None, None) => {}
            (Some(_), None) => panic!("route_waypoints[{}]: expected None", case.note),
            (None, Some(_)) => panic!("route_waypoints[{}]: expected a route", case.note),
            (Some(route), Some(expected)) => {
                assert_route(route, expected, &format!("route_waypoints[{}]", case.note))
            }
        }
    }
}

/// The cache, for a FIXED scene — where the port and the reference agree.
///
/// The sequence matters, not just the values: the third call is 3 m from the
/// first and must be served the FIRST route, points and all, because
/// `snap_m` quantizes them to the same key.
#[test]
fn the_cache_matches_python_for_a_fixed_scene() {
    let fixture = load();
    assert_eq!(fixture.cache_sequence.tag, "agreement");
    let spec = scene_named(&fixture, "one_building");
    let layer = layer_of(spec);
    let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
    let scene = Scene {
        layer: &layer,
        revision: REVISION,
    };
    for (index, call) in fixture.cache_sequence.calls.iter().enumerate() {
        let route = planner
            .plan_route(
                scene,
                point(&call.start),
                point(&call.goal),
                f(&call.clearance_m),
            )
            .unwrap_or_else(|| panic!("cache[{index}] {}: no route", call.note));
        assert_route(
            &route,
            &call.expected,
            &format!("cache[{index}] {}", call.note),
        );
    }
}

// ---------------------------------------------------------------------------
// Divergence: the cache key carries a scene revision. Nothing to compare
// against, so these assert the new contract directly.
// ---------------------------------------------------------------------------

/// A route planned before the obstacles changed must not be served afterwards.
///
/// The reference cannot express this: its key is quantized start, goal and
/// clearance, and `clear_cache()` has no caller anywhere. A platform would be
/// handed a route around a no-fly volume that has since moved.
#[test]
fn a_scene_revision_bump_invalidates_the_cache() {
    let fixture = load();
    let empty = scene_named(&fixture, "empty");
    let blocked = scene_named(&fixture, "one_building");
    let empty_layer = layer_of(empty);
    let blocked_layer = layer_of(blocked);
    let mut planner = RoutePlanner::new(bounds_of(&empty.bounds), config_of(&fixture.config));

    let start = [-200.0, 0.0];
    let goal = [200.0, 0.0];
    let clearance = f(&fixture.config.drone_clearance_m);

    let before = planner
        .plan_route(
            Scene {
                layer: &empty_layer,
                revision: 1,
            },
            start,
            goal,
            clearance,
        )
        .expect("an empty scene routes");
    assert!(!before.cache_hit);
    assert_eq!(before.vertex_count, 2, "an empty scene routes straight");

    // Same start, same goal, same clearance -- the reference's whole key. A
    // building now sits on the straight line.
    let after = planner
        .plan_route(
            Scene {
                layer: &blocked_layer,
                revision: 2,
            },
            start,
            goal,
            clearance,
        )
        .expect("the blocked scene still routes, around the building");
    assert!(
        !after.cache_hit,
        "a new revision must miss the cache; serving the old route here is \
         exactly the defect this key exists to prevent"
    );
    assert!(
        after.vertex_count > 2,
        "the route must go around the building, not through it"
    );

    // And the old revision is still cached, so the revision partitions the
    // cache rather than merely clearing it.
    let again = planner
        .plan_route(
            Scene {
                layer: &empty_layer,
                revision: 1,
            },
            start,
            goal,
            clearance,
        )
        .expect("routes");
    assert!(again.cache_hit, "revision 1's entry must survive");
}

/// The expanded-footprint cache is keyed on the revision for the same reason.
#[test]
fn a_scene_revision_bump_invalidates_the_footprint_cache() {
    let fixture = load();
    let small = scene_named(&fixture, "one_building");
    let large = scene_named(&fixture, "cylinder");
    let small_layer = layer_of(small);
    let large_layer = layer_of(large);
    let mut planner = RoutePlanner::new(bounds_of(&small.bounds), config_of(&fixture.config));

    // Both scenes name their obstacle differently, so build one that collides
    // by id: reuse `one_building`'s id against a different shape.
    let first = small_layer.primitives[0].clone();
    let mut renamed = large_layer.primitives[0].clone();
    renamed.id = first.id.clone();

    let clearance = 8.0;
    let a = planner.expanded_polygon(
        Scene {
            layer: &small_layer,
            revision: 1,
        },
        &first,
        clearance,
    );
    let b = planner.expanded_polygon(
        Scene {
            layer: &large_layer,
            revision: 2,
        },
        &renamed,
        clearance,
    );
    assert_ne!(
        a.len(),
        b.len(),
        "the two shapes must differ, or this proves nothing"
    );
    // Re-asking at revision 1 must still give the square, not the circle.
    let a_again = planner.expanded_polygon(
        Scene {
            layer: &small_layer,
            revision: 1,
        },
        &first,
        clearance,
    );
    assert_eq!(a, a_again, "revision 1's footprint must be unchanged");
}

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

/// Every guard in `plan_route` must have a case that reaches it, and each must
/// be individually observable: all four rejections return `None`, so a fixture
/// that only counted `None`s could not tell which guard was dropped.
#[test]
fn the_fixture_reaches_every_plan_route_guard() {
    let fixture = load();
    let rejected = fixture.plan_route.iter().filter(|c| c.expected.is_none());
    let notes: Vec<&str> = rejected.map(|c| c.note.as_str()).collect();
    for expected in [
        "start inside an obstacle",
        "goal inside an obstacle",
        "start inside the clearance margin",
        "goal inside the clearance margin",
        "unroutable cluster",
    ] {
        assert!(
            notes.contains(&expected),
            "no rejected case for {expected:?}; that guard is untested"
        );
    }
    let routed = fixture
        .plan_route
        .iter()
        .filter(|c| c.expected.is_some())
        .count();
    assert!(routed >= 4, "only {routed} successful routes");
}

/// `closed` must change the answer somewhere. With every segment free the
/// point list is identical either way and smoothing collapses a four-corner
/// loop to two points regardless, so the first version of this fixture could
/// not tell a port that ignored the flag from one that honoured it.
#[test]
fn the_fixture_makes_the_closed_flag_observable() {
    let fixture = load();
    let pair: Vec<&WaypointCase> = fixture
        .route_waypoints
        .iter()
        .filter(|c| c.scene == "edge_blocker")
        .collect();
    assert_eq!(pair.len(), 2, "the closed/open pair must exist");
    assert_ne!(
        pair[0].closed, pair[1].closed,
        "the pair must differ in `closed`"
    );
    assert_eq!(
        points(&pair[0].anchors),
        points(&pair[1].anchors),
        "the pair must share anchors, or it compares two different questions"
    );
    let vertices = |c: &WaypointCase| c.expected.as_ref().map(|e| e.vertex_count);
    assert_ne!(
        vertices(pair[0]),
        vertices(pair[1]),
        "closed and open produced the same route; the flag is unobservable"
    );
}

/// The port computes the candidate obstacle set ONCE for the whole start-goal
/// window, where the reference re-queries per segment. That is equivalent
/// because an obstacle the per-segment query would exclude cannot intersect
/// the segment — a superset only adds blocking, and a non-intersecting
/// polygon adds none. Asserted rather than argued: adding a far-away obstacle
/// to the set must not change any route.
#[test]
fn a_superset_of_candidate_obstacles_changes_nothing() {
    let fixture = load();
    let spec = scene_named(&fixture, "one_building");
    let near = layer_of(spec);
    let mut with_far = near.primitives.clone();
    // Far outside the corridor, but inside the bounds.
    let mut distant = near.primitives[0].clone();
    distant.id = "far".into();
    distant.shape = argusnet_world::obstacles::ObstacleShape::Polygon {
        footprint_xy_m: vec![
            [400.0, 400.0],
            [460.0, 400.0],
            [460.0, 460.0],
            [400.0, 460.0],
        ],
    };
    with_far.push(distant);
    let far_layer = ObstacleLayer::new(bounds_of(&spec.bounds), f(&spec.tile_size_m), with_far)
        .expect("layer builds");

    let config = config_of(&fixture.config);
    let clearance = config.drone_clearance_m;
    let mut a = RoutePlanner::new(bounds_of(&spec.bounds), config);
    let mut b = RoutePlanner::new(bounds_of(&spec.bounds), config);
    let route_a = a
        .plan_route(
            Scene {
                layer: &near,
                revision: 1,
            },
            [-200.0, 0.0],
            [200.0, 0.0],
            clearance,
        )
        .expect("routes");
    let route_b = b
        .plan_route(
            Scene {
                layer: &far_layer,
                revision: 1,
            },
            [-200.0, 0.0],
            [200.0, 0.0],
            clearance,
        )
        .expect("routes");
    assert_points_close(&route_a.points_xy_m, &route_b.points_xy_m, "superset route");
}

/// Eviction order, which is invisible at the default capacity of 512.
///
/// At a capacity of two the reference's `OrderedDict` discipline shows: a hit
/// calls `move_to_end`, and an overflow calls `popitem(last=False)` — so
/// touching the oldest entry saves it and the middle one is evicted instead.
/// A cache that evicted from the back, or that skipped the touch, gets the
/// opposite answer on the last two calls.
#[test]
fn cache_eviction_order_matches_python() {
    let fixture = load();
    let spec = scene_named(&fixture, &fixture.cache_eviction.scene);
    let layer = layer_of(spec);
    let config = PlannerConfig {
        max_cache_entries: fixture.cache_eviction.max_cache_entries,
        ..config_of(&fixture.config)
    };
    let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config);
    let scene = Scene {
        layer: &layer,
        revision: REVISION,
    };
    let clearance = f(&fixture.cache_eviction.clearance_m);
    for (index, call) in fixture.cache_eviction.calls.iter().enumerate() {
        let route = planner
            .plan_route(scene, point(&call.start), point(&call.goal), clearance)
            .unwrap_or_else(|| panic!("eviction[{index}] {}: no route", call.note));
        assert_eq!(
            route.cache_hit, call.cache_hit,
            "eviction[{index}] {}",
            call.note
        );
    }
}

/// Cache-key quantization, and the rounding rule underneath it.
///
/// Python's `round()` is BANKER'S rounding, so +5 and -5 at a snap of 10 both
/// key to 0 and the second is a hit. Rust's `f64::round()` rounds half away
/// from zero and would key them +1 and -1 — two entries for what the
/// reference treats as one request. `round_ties_even` is what makes this pass,
/// and this case is the only thing in the suite that says so.
///
/// +6 then separates rounding from truncation: it keys to 1 under either
/// rounding rule and to 0 — +5's key — under `floor`.
#[test]
fn cache_key_quantization_matches_python() {
    let fixture = load();
    let spec = scene_named(&fixture, &fixture.cache_quantization.scene);
    let layer = layer_of(spec);
    let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config_of(&fixture.config));
    let scene = Scene {
        layer: &layer,
        revision: REVISION,
    };
    for (index, call) in fixture.cache_quantization.calls.iter().enumerate() {
        let route = planner
            .plan_route(
                scene,
                point(&call.start),
                point(&call.goal),
                f(&call.clearance_m),
            )
            .unwrap_or_else(|| panic!("quantization[{index}] {}: no route", call.note));
        assert_eq!(
            route.cache_hit, call.cache_hit,
            "quantization[{index}] {}",
            call.note
        );
    }
}

/// Vegetation is not a hard blocker: a route flies through a forest stand.
/// Without a scene containing some, a port that treated every obstacle as
/// solid would pass the whole suite.
#[test]
fn the_fixture_covers_a_non_blocking_obstacle() {
    let fixture = load();
    let forest = fixture
        .plan_route
        .iter()
        .find(|c| c.scene == "forest")
        .expect("a vegetation case must exist");
    let expected = forest
        .expected
        .as_ref()
        .expect("vegetation must not block the route");
    assert_eq!(
        expected.vertex_count, 2,
        "a route through vegetation is a straight line, not a detour"
    );
}

/// The endpoint guards must be reachable in a way that DISTINGUISHES them
/// from the graph simply failing.
///
/// A start strictly inside an obstacle yields no route either way: every edge
/// leaving it is rejected, so A* finds nothing. Only a start exactly ON the
/// expanded boundary separates the two — `blocking_primitive` uses
/// point-in-polygon, which counts a boundary point as inside, while
/// `segment_is_free` exempts boundary points so the graph would route.
#[test]
fn the_fixture_separates_the_endpoint_guards_from_a_failed_search() {
    let fixture = load();
    for note in [
        "start on the expanded boundary",
        "goal on the expanded boundary",
    ] {
        let case = fixture
            .plan_route
            .iter()
            .find(|c| c.note == note)
            .unwrap_or_else(|| panic!("missing case {note:?}"));
        assert!(
            case.expected.is_none(),
            "{note}: must be rejected by the guard"
        );
    }
}

/// `snap_m` is floored at 1.0, which the default of 10 hides entirely.
///
/// Configured at 0.1, two starts 0.4 m apart are four keys apart — unless the
/// floor lifts the snap to 1.0, which puts them in the same bucket. The
/// second call is a hit only if the floor is applied.
#[test]
fn the_snap_floor_matches_python() {
    let fixture = load();
    let spec = scene_named(&fixture, &fixture.snap_floor.scene);
    let layer = layer_of(spec);
    let config = PlannerConfig {
        snap_m: f(&fixture.snap_floor.snap_m),
        ..config_of(&fixture.config)
    };
    let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config);
    let scene = Scene {
        layer: &layer,
        revision: REVISION,
    };
    for (index, call) in fixture.snap_floor.calls.iter().enumerate() {
        let route = planner
            .plan_route(scene, point(&call.start), point(&call.goal), 8.0)
            .unwrap_or_else(|| panic!("snap_floor[{index}]: no route"));
        assert_eq!(
            route.cache_hit, call.cache_hit,
            "snap_floor[{index}] {}",
            call.note
        );
    }
}

/// `cylinder_vertex_count` is floored at 6, which the default of 12 hides.
/// Configured at 3, the reference still produces a hexagon.
#[test]
fn the_cylinder_vertex_floor_matches_python() {
    let fixture = load();
    let spec = scene_named(&fixture, &fixture.cylinder_vertex_floor.scene);
    let layer = layer_of(spec);
    let config = PlannerConfig {
        cylinder_vertex_count: fixture.cylinder_vertex_floor.cylinder_vertex_count,
        ..config_of(&fixture.config)
    };
    let mut planner = RoutePlanner::new(bounds_of(&spec.bounds), config);
    let obstacle = layer.primitives[0].clone();
    let actual = planner.expanded_polygon(
        Scene {
            layer: &layer,
            revision: REVISION,
        },
        &obstacle,
        f(&fixture.cylinder_vertex_floor.clearance_m),
    );
    assert_eq!(
        actual.len(),
        6,
        "a count of 3 must be lifted to the floor of 6"
    );
    assert_points_close(
        &actual,
        &points(&fixture.cylinder_vertex_floor.expected),
        "cylinder_vertex_floor",
    );
}

/// The threshold case must skip EXACTLY `max(1, segments / 2)`, or it cannot
/// tell `>` from `>=` — the difference between giving up and returning a
/// route that visits half of what it was asked to.
#[test]
fn the_fixture_sits_exactly_on_the_skip_threshold() {
    let fixture = load();
    let case = fixture
        .route_waypoints
        .iter()
        .find(|c| c.note == "skips exactly at the threshold")
        .expect("the threshold case must exist");
    let expected = case
        .expected
        .as_ref()
        .expect("at the threshold the reference still returns a route");
    let segments = case.anchors.len(); // closed, so one segment per anchor
    assert!(case.closed);
    assert_eq!(
        expected.skipped_segments,
        (segments / 2).max(1),
        "the case must sit exactly on the threshold, not below it"
    );
}

/// **Four equivalent mutants, recorded rather than tested around.**
///
/// 1. `candidate_primitives` falls back to every hard primitive when the
///    padded query returns none. Dropping the fallback changes nothing, and
///    provably so: the query pads by `max(clearance * 4, 80)`, and an
///    obstacle's expanded footprint grows by only `clearance`. Since
///    `max(4c, 80) >= c` for every non-negative `c`, an empty query means no
///    obstacle can reach the segment — so the straight line is free, the
///    early return fires, and no graph is ever built from the fallback's
///    extra polygons.
/// 3. The `max(80.0)` floor on the candidate query window, for the same
///    reason: `clearance * 4` already exceeds `clearance`.
/// 4. `nearest_free_point`'s `margin.max(0.5)` at the call site.
///    `push_outside_polygon` floors its own `margin_m` at 0.5 as well, so the
///    outer floor can never change the step. The reference double-floors in
///    exactly the same way — `nearest_free_point` passes
///    `margin_m=max(margin_m, 0.5)` into a function whose first act is
///    `max(margin_m, 0.5)` — so this is faithful redundancy, not a port
///    mistake, and is left in place rather than tidied.
/// 2. Not de-duplicating the join point between consecutive segments. The
///    duplicate is coincident with the previous point, so it is trivially
///    visible from it and `smooth_path` skips straight past it.
///
/// Both are asserted here rather than closed with a contrived test.
#[test]
fn four_recorded_equivalent_mutants_in_the_planner() {
    let fixture = load();
    let config = config_of(&fixture.config);

    // 1 and 3. Both rest on one inequality: the query window is always at
    // least as wide as the distance a footprint expands, so an obstacle the
    // query excludes cannot reach the segment. That makes the empty-query
    // fallback unnecessary AND makes the 80 m floor on the window
    // unnecessary — the `clearance * 4` term alone already dominates.
    let _ = &config;
    for clearance in [0.0_f64, 0.2, 8.0, 25.0, 200.0, 10_000.0] {
        let with_floor = (clearance * 4.0).max(80.0);
        let without_floor = clearance * 4.0;
        assert!(
            without_floor >= clearance,
            "at a clearance of {clearance} the unfloored window {without_floor} \
             is narrower than the expansion itself"
        );
        assert!(with_floor >= without_floor);
    }

    // 4. The inner floor makes the outer one unobservable: pushing with a
    //    margin below 0.5 gives the same point as pushing with 0.5.
    let square = vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
    for margin in [0.0_f64, 0.2, 0.4999] {
        assert_eq!(
            argusnet_planning::route::push_outside_polygon([0.0, 0.0], &square, margin),
            argusnet_planning::route::push_outside_polygon([0.0, 0.0], &square, 0.5),
            "a margin of {margin} must be floored to 0.5 inside push_outside_polygon"
        );
    }

    // 2. A coincident duplicate does not survive smoothing.
    let with_duplicate = [[0.0, 0.0], [50.0, 0.0], [50.0, 0.0], [100.0, 0.0]];
    let without = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]];
    assert_eq!(
        argusnet_planning::route::smooth_path(&with_duplicate, &[]),
        argusnet_planning::route::smooth_path(&without, &[]),
        "a duplicated join point must not survive smoothing"
    );
}
