//! `PathPlanner2D`'s public routing surface: `nearest_free_point`,
//! `plan_route`, `route_waypoints`, and the route cache.
//!
//! The graph, search and smoothing live in [`crate::route`]; this is the layer
//! that decides which obstacles to consider, guards the endpoints, and caches.
//!
//! # The cache carries a scene revision, and the reference's does not
//!
//! This is a deliberate divergence under ADR-006, recorded in
//! `docs/migration/PLANNING_INVENTORY.md`. The reference's key is quantized
//! start, quantized goal and clearance — no obstacle or scene identity of any
//! kind — and `PathPlanner2D.clear_cache()` has no caller anywhere in `src/`
//! or `tests/`. Nothing invalidates it. `Environment.add_obstacle()` already
//! bumps `scene_version`, and `EnvironmentQuery.los` already keys its own
//! cache on that counter, so the planner is the one cache in the
//! world/planning stack that ignores it.
//!
//! Reproducing that would mean a platform routed around a no-fly volume that
//! has since moved, silently, because the start and goal happened to quantize
//! the same way. [`Scene`] therefore carries a revision, and it is part of
//! both cache keys. Latent in the reference rather than live — nothing calls
//! `add_obstacle` today — but there is no reason to carry the omission into a
//! new implementation.

use crate::route::{
    astar_path, build_visibility_graph, expand_polygon, path_length, segment_is_free, smooth_path,
    Point2, RouteConfig,
};
use argusnet_types::Bounds2d;
use argusnet_world::index::ObstacleLayer;
use argusnet_world::obstacles::{BlockerType, Obstacle, ObstacleShape};
use std::collections::HashMap;

/// The obstacle set a route is planned against, plus the revision that
/// identifies it. Bump `revision` whenever `layer` changes, exactly as
/// `Environment.bump_scene_version()` does.
#[derive(Debug, Clone, Copy)]
pub struct Scene<'a> {
    pub layer: &'a ObstacleLayer,
    pub revision: u64,
}

/// Tuning constants. Defaults match the reference's `PlannerConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    pub drone_clearance_m: f64,
    pub target_clearance_m: f64,
    pub snap_m: f64,
    pub cylinder_vertex_count: usize,
    pub turn_penalty_m: f64,
    pub max_cache_entries: usize,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            drone_clearance_m: 8.0,
            target_clearance_m: 4.0,
            snap_m: 10.0,
            cylinder_vertex_count: 12,
            turn_penalty_m: 3.0,
            max_cache_entries: 512,
        }
    }
}

/// A planned route.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerRoute {
    pub points_xy_m: Vec<Point2>,
    pub length_m: f64,
    pub vertex_count: usize,
    /// True when this route came from the cache. Not stored — the cached copy
    /// always holds `false` and the flag is set on the way out, as the
    /// reference does with `dataclasses.replace`.
    pub cache_hit: bool,
    pub skipped_segments: usize,
}

/// Only these block a route. Vegetation is seen through, not flown around.
fn is_hard(blocker: &BlockerType) -> bool {
    matches!(blocker, BlockerType::Building | BlockerType::Wall)
}

fn padded(bounds: Bounds2d, padding_m: f64) -> Bounds2d {
    Bounds2d::new(
        bounds.x_min_m - padding_m,
        bounds.x_max_m + padding_m,
        bounds.y_min_m - padding_m,
        bounds.y_max_m + padding_m,
    )
}

fn segment_bounds(start: Point2, end: Point2) -> Bounds2d {
    let epsilon = 1.0e-6;
    Bounds2d::new(
        start[0].min(end[0]) - epsilon,
        start[0].max(end[0]) + epsilon,
        start[1].min(end[1]) - epsilon,
        start[1].max(end[1]) + epsilon,
    )
}

/// The route cache key.
///
/// `revision` is the addition; everything after it matches the reference.
/// Clearance is rounded to three decimals and then compared by bit pattern,
/// so the key is hashable without a float-equality wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RouteKey {
    revision: u64,
    start_x: i64,
    start_y: i64,
    goal_x: i64,
    goal_y: i64,
    clearance_bits: u64,
}

/// The reference's `OrderedDict` cache, including its eviction order:
/// `move_to_end` on a hit, `popitem(last=False)` when full. Order is
/// observable through `cache_hit`, so it is reproduced rather than swapped
/// for a plain map. `max_cache_entries` is 512, so the linear `order` scan is
/// not worth a fancier structure.
#[derive(Debug, Default)]
struct RouteCache {
    entries: HashMap<RouteKey, PlannerRoute>,
    order: Vec<RouteKey>,
}

impl RouteCache {
    fn get(&mut self, key: &RouteKey) -> Option<PlannerRoute> {
        let route = self.entries.get(key)?.clone();
        self.touch(key);
        Some(route)
    }

    fn touch(&mut self, key: &RouteKey) {
        if let Some(index) = self.order.iter().position(|k| k == key) {
            let existing = self.order.remove(index);
            self.order.push(existing);
        }
    }

    fn insert(&mut self, key: RouteKey, route: PlannerRoute, capacity: usize) {
        self.entries.insert(key, route);
        self.touch(&key);
        if !self.order.contains(&key) {
            self.order.push(key);
        }
        while self.order.len() > capacity.max(1) {
            let evicted = self.order.remove(0);
            self.entries.remove(&evicted);
        }
    }
}

/// Plans routes around an obstacle layer.
#[derive(Debug)]
pub struct RoutePlanner {
    bounds_xy_m: Bounds2d,
    config: PlannerConfig,
    route_cache: RouteCache,
    /// Keyed on the scene revision too, for the same reason as the routes: an
    /// expanded footprint is only valid for the obstacle that produced it.
    polygon_cache: HashMap<(u64, String, u64), Vec<Point2>>,
}

impl RoutePlanner {
    pub fn new(bounds_xy_m: Bounds2d, config: PlannerConfig) -> Self {
        Self {
            bounds_xy_m,
            config,
            route_cache: RouteCache::default(),
            polygon_cache: HashMap::new(),
        }
    }

    pub fn config(&self) -> &PlannerConfig {
        &self.config
    }

    fn route_config(&self) -> RouteConfig {
        RouteConfig {
            turn_penalty_m: self.config.turn_penalty_m,
        }
    }

    /// The obstacle's footprint grown by `clearance_m`.
    ///
    /// A cylinder is the one shape with no stored footprint: the reference
    /// samples `cylinder_vertex_count` points around a circle of
    /// `radius + clearance`, starting at angle zero and going anticlockwise,
    /// with the endpoint excluded. Everything else — polygon prisms,
    /// buildings, oriented boxes and walls — reaches the same
    /// `_expand_polygon` call in the reference, and arrives here already
    /// converted to a polygon by `argusnet-world`.
    pub fn expanded_polygon(
        &mut self,
        scene: Scene<'_>,
        obstacle: &Obstacle,
        clearance_m: f64,
    ) -> Vec<Point2> {
        let rounded = (clearance_m * 1000.0).round() / 1000.0;
        let key = (scene.revision, obstacle.id.clone(), rounded.to_bits());
        if let Some(cached) = self.polygon_cache.get(&key) {
            return cached.clone();
        }
        let polygon = match &obstacle.shape {
            ObstacleShape::Cylinder {
                center_xy_m,
                radius_m,
            } => {
                let radius = radius_m + clearance_m.max(0.0);
                let count = self.config.cylinder_vertex_count.max(6);
                (0..count)
                    .map(|index| {
                        let angle = std::f64::consts::TAU * (index as f64) / (count as f64);
                        [
                            center_xy_m[0] + radius * angle.cos(),
                            center_xy_m[1] + radius * angle.sin(),
                        ]
                    })
                    .collect()
            }
            ObstacleShape::Polygon { footprint_xy_m } => {
                expand_polygon(footprint_xy_m, clearance_m)
            }
        };
        self.polygon_cache.insert(key, polygon.clone());
        polygon
    }

    fn hard_primitives<'a>(&self, scene: Scene<'a>) -> Vec<&'a Obstacle> {
        scene
            .layer
            .primitives
            .iter()
            .filter(|o| is_hard(&o.blocker_type))
            .collect()
    }

    /// The obstacles worth considering for a route between these points.
    ///
    /// Falls back to every hard primitive when the padded query finds none,
    /// exactly as the reference does — which looks redundant and is not: it is
    /// what makes a route past an obstacle just outside the padded window
    /// still avoid it.
    fn candidate_primitives<'a>(
        &self,
        scene: Scene<'a>,
        start: Point2,
        goal: Point2,
    ) -> Vec<&'a Obstacle> {
        let window = padded(
            segment_bounds(start, goal),
            (self.config.drone_clearance_m * 4.0).max(80.0),
        );
        let hard: Vec<&Obstacle> = scene
            .layer
            .query(window)
            .into_iter()
            .filter(|o| is_hard(&o.blocker_type))
            .collect();
        if hard.is_empty() {
            self.hard_primitives(scene)
        } else {
            hard
        }
    }

    fn point_within_bounds(&self, point: Point2, clearance_m: f64) -> bool {
        let margin = clearance_m.max(0.0);
        self.bounds_xy_m.x_min_m + margin <= point[0]
            && point[0] <= self.bounds_xy_m.x_max_m - margin
            && self.bounds_xy_m.y_min_m + margin <= point[1]
            && point[1] <= self.bounds_xy_m.y_max_m - margin
    }

    /// The first hard obstacle whose expanded footprint contains the point.
    fn blocking_primitive(
        &mut self,
        scene: Scene<'_>,
        point: Point2,
        clearance_m: f64,
    ) -> Option<String> {
        let ids: Vec<String> = self
            .hard_primitives(scene)
            .into_iter()
            .map(|o| o.id.clone())
            .collect();
        for id in ids {
            let obstacle = scene.layer.primitives.iter().find(|o| o.id == id)?.clone();
            let polygon = self.expanded_polygon(scene, &obstacle, clearance_m);
            if argusnet_world::obstacles::point_in_polygon(point, &polygon) {
                return Some(obstacle.id);
            }
        }
        None
    }

    /// Move a point to somewhere a platform can actually be.
    ///
    /// Clamped into bounds first, then pushed out of whatever contains it, at
    /// most ten times. The cap is the reference's, and it means a point deep
    /// inside a concave cluster can still be returned inside one — recorded
    /// rather than fixed, since callers already treat the result as advisory.
    pub fn nearest_free_point(
        &mut self,
        scene: Scene<'_>,
        point: Point2,
        clearance_m: f64,
    ) -> Point2 {
        let margin = clearance_m.max(0.0);
        let clamp = |p: Point2, bounds: Bounds2d| -> Point2 {
            [
                p[0].clamp(bounds.x_min_m + margin, bounds.x_max_m - margin),
                p[1].clamp(bounds.y_min_m + margin, bounds.y_max_m - margin),
            ]
        };
        let mut candidate = clamp(point, self.bounds_xy_m);
        for _ in 0..10 {
            let Some(id) = self.blocking_primitive(scene, candidate, clearance_m) else {
                return candidate;
            };
            let Some(obstacle) = scene.layer.primitives.iter().find(|o| o.id == id).cloned() else {
                return candidate;
            };
            let polygon = self.expanded_polygon(scene, &obstacle, clearance_m);
            candidate = crate::route::push_outside_polygon(candidate, &polygon, margin.max(0.5));
            candidate = clamp(candidate, self.bounds_xy_m);
        }
        candidate
    }

    /// The route cache key.
    ///
    /// `round_ties_even`, not `round`. Python's built-in `round()` is BANKER'S
    /// rounding, so a coordinate landing exactly on a half-snap goes to even:
    /// `round(0.5) == 0` and `round(-0.5) == 0`, and +5 and -5 at a snap of
    /// 10 therefore share a key. Rust's `f64::round()` rounds half away from
    /// zero and would split them into +1 and -1 -- two entries for requests
    /// the reference serves from one. Caught by the fixture, which refuses to
    /// emit a quantization case that cannot tell the two rules apart.
    fn route_cache_key(
        &self,
        scene: Scene<'_>,
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> RouteKey {
        let snap = self.config.snap_m.max(1.0);
        RouteKey {
            revision: scene.revision,
            start_x: (start[0] / snap).round_ties_even() as i64,
            start_y: (start[1] / snap).round_ties_even() as i64,
            goal_x: (goal[0] / snap).round_ties_even() as i64,
            goal_y: (goal[1] / snap).round_ties_even() as i64,
            // `round(x, 3)` in Python is banker's too, for the same reason.
            clearance_bits: ((clearance_m * 1000.0).round_ties_even() / 1000.0).to_bits(),
        }
    }

    /// Plan a route from `start` to `goal`.
    ///
    /// Returns `None` when either endpoint is outside the bounds by less than
    /// the clearance, when either is inside an obstacle, or when the goal is
    /// unreachable through the visibility graph — which a cluster of
    /// obstacles with overlapping expanded footprints makes possible even
    /// when a route plainly exists around it. See `VISIBILITY_ROUTING.md`.
    pub fn plan_route(
        &mut self,
        scene: Scene<'_>,
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> Option<PlannerRoute> {
        if !self.point_within_bounds(start, clearance_m)
            || !self.point_within_bounds(goal, clearance_m)
        {
            return None;
        }
        if self.blocking_primitive(scene, start, clearance_m).is_some() {
            return None;
        }
        if self.blocking_primitive(scene, goal, clearance_m).is_some() {
            return None;
        }

        let key = self.route_cache_key(scene, start, goal, clearance_m);
        if let Some(mut cached) = self.route_cache.get(&key) {
            cached.cache_hit = true;
            return Some(cached);
        }
        let route = self.search(scene, start, goal, clearance_m)?;
        self.route_cache
            .insert(key, route.clone(), self.config.max_cache_entries);
        Some(route)
    }

    /// [`Self::plan_route`] without the route cache: the same guards and the
    /// same search, always from exactly `start` to exactly `goal`.
    ///
    /// The cache quantizes its key to `snap_m`, so a hit hands back a route
    /// that starts where an *earlier* request started, up to half a snap
    /// away. That is harmless at build time, where the planner is asked once.
    /// A runtime caller that plans from wherever a platform happens to be
    /// would make its output depend on what it asked before, and a run
    /// resumed from a checkpoint (with an empty cache) would then fly
    /// differently from the run it resumed. This entry point is for that
    /// caller. The expanded-footprint cache is still used: it is keyed
    /// exactly, so it can only save work, never change an answer.
    pub fn plan_route_uncached(
        &mut self,
        scene: Scene<'_>,
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> Option<PlannerRoute> {
        if !self.point_within_bounds(start, clearance_m)
            || !self.point_within_bounds(goal, clearance_m)
        {
            return None;
        }
        if self.blocking_primitive(scene, start, clearance_m).is_some() {
            return None;
        }
        if self.blocking_primitive(scene, goal, clearance_m).is_some() {
            return None;
        }
        // The corridor's candidates can miss an obstacle on a detour that
        // leaves the corridor, so the route is checked against the obstacles
        // the corridor left out, and searched again among every hard
        // obstacle if it crosses one of them.
        let corridor: Vec<Obstacle> = self
            .candidate_primitives(scene, start, goal)
            .into_iter()
            .cloned()
            .collect();
        let route = self.search_among(scene, &corridor, start, goal, clearance_m);
        let omitted: Vec<Obstacle> = self
            .hard_primitives(scene)
            .into_iter()
            .filter(|o| !corridor.iter().any(|c| c.id == o.id))
            .cloned()
            .collect();
        let omitted_polygons: Vec<Vec<Point2>> = omitted
            .iter()
            .map(|o| self.expanded_polygon(scene, o, clearance_m))
            .collect();
        let crosses_omitted = |route: &PlannerRoute| {
            route
                .points_xy_m
                .windows(2)
                .any(|pair| !segment_is_free(pair[0], pair[1], &omitted_polygons))
        };
        // No route among fewer obstacles means none among more.
        let route = match route? {
            route if !crosses_omitted(&route) => route,
            _ => {
                let mut all = corridor;
                all.extend(omitted);
                self.search_among(scene, &all, start, goal, clearance_m)?
            }
        };
        // An obstacle at the map's edge has corners past it; a route round
        // one would leave the world, so there is none.
        route
            .points_xy_m
            .iter()
            .all(|point| self.point_within_bounds(*point, clearance_m))
            .then_some(route)
    }

    /// [`Self::plan_route_uncached`] for ends that may sit inside the
    /// clearance: a POI beside a wall, or a scan lane's end at the map's
    /// edge. Each end in the clearance is moved to the nearest point a
    /// platform can be ([`Self::nearest_free_point`]), the route is planned
    /// between those, and the true ends are joined to it by straight legs.
    /// A leg is accepted only if it crosses no hard obstacle's own footprint
    /// (it is inside the clearance by construction), and an end must itself
    /// be in bounds and outside every footprint, its faces included;
    /// otherwise there is no route. The legs are the only part of the route
    /// inside the clearance: even when the clear points see each other, the
    /// route still goes by them rather than straight between the true ends,
    /// which could run along a wall inside the clearance all the way.
    pub fn plan_route_between(
        &mut self,
        scene: Scene<'_>,
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> Option<PlannerRoute> {
        let free_start = self.nearest_free_point(scene, start, clearance_m);
        let free_goal = self.nearest_free_point(scene, goal, clearance_m);
        let middle = self.plan_route_uncached(scene, free_start, free_goal, clearance_m)?;
        if free_start == start && free_goal == goal {
            return Some(middle);
        }
        if !self.point_within_bounds(start, 0.0) || !self.point_within_bounds(goal, 0.0) {
            return None;
        }
        let footprints: Vec<Vec<Point2>> = self
            .hard_primitives(scene)
            .into_iter()
            .cloned()
            .collect::<Vec<Obstacle>>()
            .iter()
            .map(|o| self.expanded_polygon(scene, o, 0.0))
            .collect();
        // `segment_is_free` lets a segment end on a polygon's boundary, so
        // an end on a face (which `Obstacle::point_inside` counts as inside)
        // is refused here first.
        let solid = |point: Point2| {
            footprints
                .iter()
                .any(|footprint| argusnet_world::obstacles::point_in_polygon(point, footprint))
        };
        if solid(start) || solid(goal) {
            return None;
        }
        let leg_clear =
            |from: Point2, to: Point2| from == to || segment_is_free(from, to, &footprints);
        if !leg_clear(start, free_start) || !leg_clear(free_goal, goal) {
            return None;
        }
        let mut points = Vec::with_capacity(middle.points_xy_m.len() + 2);
        if free_start != start {
            points.push(start);
        }
        points.extend(middle.points_xy_m.iter().copied());
        if free_goal != goal {
            points.push(goal);
        }
        Some(PlannerRoute {
            length_m: path_length(&points),
            vertex_count: points.len(),
            points_xy_m: points,
            cache_hit: false,
            skipped_segments: 0,
        })
    }

    /// The straight segment if it is free, else the smoothed A* route over
    /// the visibility graph. The endpoints are already known to be in bounds
    /// and outside every obstacle.
    fn search(
        &mut self,
        scene: Scene<'_>,
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> Option<PlannerRoute> {
        // The candidate set is computed once for the whole start-goal window
        // and reused for every pair, where the reference re-queries the layer
        // per segment. That is equivalent, not a shortcut: an obstacle the
        // per-segment query would exclude has bounds that do not overlap the
        // segment, so it can neither contain an endpoint nor cross it, and a
        // superset only ever adds blocking. Asserted in the tests.
        let candidates: Vec<Obstacle> = self
            .candidate_primitives(scene, start, goal)
            .into_iter()
            .cloned()
            .collect();
        self.search_among(scene, &candidates, start, goal, clearance_m)
    }

    /// [`Self::search`] over a given set of obstacles.
    fn search_among(
        &mut self,
        scene: Scene<'_>,
        candidates: &[Obstacle],
        start: Point2,
        goal: Point2,
        clearance_m: f64,
    ) -> Option<PlannerRoute> {
        let polygons: Vec<Vec<Point2>> = candidates
            .iter()
            .map(|o| self.expanded_polygon(scene, o, clearance_m))
            .collect();

        if segment_is_free(start, goal, &polygons) {
            return Some(PlannerRoute {
                points_xy_m: vec![start, goal],
                length_m: path_length(&[start, goal]),
                vertex_count: 2,
                cache_hit: false,
                skipped_segments: 0,
            });
        }

        let mut nodes: Vec<Point2> = vec![start, goal];
        let mut polygon_node_indices: Vec<Vec<usize>> = Vec::new();
        for polygon in &polygons {
            let mut indices = Vec::with_capacity(polygon.len());
            for vertex in polygon {
                nodes.push(*vertex);
                indices.push(nodes.len() - 1);
            }
            polygon_node_indices.push(indices);
        }
        let adjacency = build_visibility_graph(&nodes, &polygon_node_indices, &polygons);
        let indices = astar_path(&nodes, &adjacency, &self.route_config())?;
        let raw: Vec<Point2> = indices.iter().map(|&i| nodes[i]).collect();
        let smoothed = smooth_path(&raw, &polygons);
        Some(PlannerRoute {
            length_m: path_length(&smoothed),
            vertex_count: smoothed.len(),
            points_xy_m: smoothed,
            cache_hit: false,
            skipped_segments: 0,
        })
    }

    /// Route through a sequence of anchors, optionally closing the loop.
    ///
    /// Every anchor is first moved to a reachable point, then consecutive
    /// pairs are routed and concatenated. A pair that cannot be routed is
    /// skipped and counted; more than `max(1, segments / 2)` skipped gives up
    /// entirely, so a route that misses most of what it was asked to visit is
    /// reported as no route rather than as a shorter one.
    pub fn route_waypoints(
        &mut self,
        scene: Scene<'_>,
        anchors: &[Point2],
        clearance_m: f64,
        closed: bool,
    ) -> Option<PlannerRoute> {
        assert!(
            anchors.len() >= 2,
            "route_waypoints needs at least 2 anchors"
        );
        let adjusted: Vec<Point2> = anchors
            .iter()
            .map(|a| self.nearest_free_point(scene, *a, clearance_m))
            .collect();

        let segment_count = if closed {
            adjusted.len()
        } else {
            adjusted.len() - 1
        };
        let mut segments: Vec<Vec<Point2>> = Vec::new();
        let mut skipped = 0usize;
        for index in 0..segment_count {
            let start = adjusted[index];
            let end = adjusted[(index + 1) % adjusted.len()];
            let Some(route) = self.plan_route(scene, start, end, clearance_m) else {
                skipped += 1;
                continue;
            };
            if segments.is_empty() {
                segments.push(route.points_xy_m);
            } else {
                segments.push(route.points_xy_m[1..].to_vec());
            }
        }
        if segments.is_empty() {
            return None;
        }
        if skipped > (segment_count / 2).max(1) {
            return None;
        }

        let mut points: Vec<Point2> = segments.concat();
        if closed && points.len() >= 2 {
            let first = points[0];
            let last = points[points.len() - 1];
            // `np.allclose`'s defaults: atol 1e-8, rtol 1e-5 against the
            // SECOND argument. Spelled out rather than approximated, since a
            // plain equality here would leave the closing duplicate in place.
            let close = |a: f64, b: f64| (a - b).abs() <= 1.0e-8 + 1.0e-5 * b.abs();
            if close(first[0], last[0]) && close(first[1], last[1]) {
                points.pop();
            }
        }
        if points.len() < 2 {
            return None;
        }
        let polygons: Vec<Vec<Point2>> = {
            let candidates: Vec<Obstacle> = self
                .candidate_primitives(scene, points[0], points[points.len() - 1])
                .into_iter()
                .cloned()
                .collect();
            candidates
                .iter()
                .map(|o| self.expanded_polygon(scene, o, clearance_m))
                .collect()
        };
        let smoothed = smooth_path(&points, &polygons);
        if smoothed.len() < 2 || path_length(&smoothed) <= 1.0e-6 {
            return None;
        }
        Some(PlannerRoute {
            length_m: path_length(&smoothed),
            vertex_count: smoothed.len(),
            points_xy_m: smoothed,
            cache_hit: false,
            skipped_segments: skipped,
        })
    }
}
