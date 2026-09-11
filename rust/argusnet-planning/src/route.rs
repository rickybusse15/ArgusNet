//! Visibility-graph routing: the ordering-sensitive core of `PathPlanner2D`.
//!
//! Scope is polygon expansion, segment freedom, the visibility graph, the A*
//! search and the path smoothing. `plan_route`'s cache and `route_waypoints`
//! are P4.7d; the revision-keyed cache P4.7 asks for lands with them, since
//! `docs/migration/PLANNING_INVENTORY.md` establishes that the reference's key
//! carries no scene revision at all.
//!
//! Geometry comes from `argusnet-world`, which Phase 3 already ported and
//! tests separately -- `point_in_polygon`, `segment_polygon_intervals`,
//! `nearest_point_on_segment`, `signed_polygon_area`, `edge_outward_normal`
//! and `point_on_segment` are all reused rather than written again here.
//!
//! # The ordering hazard
//!
//! A* iterates a node's neighbours in the order they were appended to its
//! adjacency list, and rejects a neighbour whose cost is not *strictly* better
//! (`next_cost + 1e-9 >= best`). Between two routes of equal cost, the one
//! reached first therefore wins. On an obstacle symmetric about the
//! start-goal axis -- a square building straddling the line -- the two detours
//! cost the same to the last bit, and only the ordering decides which is
//! returned.
//!
//! So adjacency is a `Vec`, never a set, and it is built in exactly the
//! reference's order: ascending `right_index` for the visible pairs, then
//! polygon edges appended afterwards. Storing it in a `HashSet`, or sorting
//! it, would return the mirror-image route while every individual edge still
//! agreed with Python.

use argusnet_world::obstacles::{
    edge_outward_normal, nearest_point_on_segment, point_in_polygon, point_on_segment,
    segment_polygon_intervals, signed_polygon_area,
};
use std::collections::{BinaryHeap, HashMap};

/// A point in the local XY plane, in metres.
pub type Point2 = [f64; 2];

/// Tuning constants. Defaults match the reference's `PlannerConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteConfig {
    /// Metres of extra cost per radian of heading change at a graph vertex.
    pub turn_penalty_m: f64,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            turn_penalty_m: 3.0,
        }
    }
}

/// Grow a polygon outward by `clearance_m`, mitred at the vertices.
///
/// Two details are load-bearing and easy to lose:
///
/// * The winding is measured, not assumed. `signed_polygon_area() >= 0` marks
///   counter-clockwise, and the outward normal flips with it -- so a polygon
///   given clockwise expands outward, not inward.
/// * `scale` is floored at `0.2`, a **miter limit**. At a sharp vertex the
///   two edge normals nearly oppose, their sum is short, and the offset
///   `clearance / scale` would run away to a spike; the floor caps it at five
///   times the clearance. The floor only binds on acute vertices, so a fixture
///   of rectangles would never reach it.
///
/// A clearance at or below `1e-9` returns the polygon unchanged, rather than
/// running the mitre with a zero offset.
pub fn expand_polygon(polygon: &[Point2], clearance_m: f64) -> Vec<Point2> {
    let clearance = clearance_m.max(0.0);
    if clearance <= 1.0e-9 {
        return polygon.to_vec();
    }
    let is_ccw = signed_polygon_area(polygon) >= 0.0;
    let n = polygon.len();
    (0..n)
        .map(|index| {
            let previous = polygon[(index + n - 1) % n];
            let current = polygon[index];
            let next = polygon[(index + 1) % n];
            let incoming = edge_outward_normal(previous, current, is_ccw);
            let outgoing = edge_outward_normal(current, next, is_ccw);
            let sum = [incoming[0] + outgoing[0], incoming[1] + outgoing[1]];
            let norm = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
            let direction = if norm <= 1.0e-9 {
                incoming
            } else {
                [sum[0] / norm, sum[1] / norm]
            };
            let scale = (direction[0] * incoming[0] + direction[1] * incoming[1]).max(0.2);
            let offset = clearance / scale;
            [
                current[0] + direction[0] * offset,
                current[1] + direction[1] * offset,
            ]
        })
        .collect()
}

/// Move a point out of a polygon, if it is inside one.
///
/// Every edge is tried and the nearest resulting candidate wins -- not the
/// nearest edge, which is not the same thing once the margin is applied. The
/// inner loop pushes at most eight times, so a point deep inside a concave
/// polygon can still be returned inside it; the reference accepts that and so
/// does this.
pub fn push_outside_polygon(point: Point2, polygon: &[Point2], margin_m: f64) -> Point2 {
    if !point_in_polygon(point, polygon) {
        return point;
    }
    let is_ccw = signed_polygon_area(polygon) >= 0.0;
    let step = margin_m.max(0.5);
    let n = polygon.len();
    let mut best = point;
    let mut best_distance = f64::INFINITY;
    for index in 0..n {
        let edge_start = polygon[index];
        let edge_end = polygon[(index + 1) % n];
        let boundary = nearest_point_on_segment(point, edge_start, edge_end);
        let normal = edge_outward_normal(edge_start, edge_end, is_ccw);
        let mut candidate = [
            boundary[0] + normal[0] * step,
            boundary[1] + normal[1] * step,
        ];
        for _ in 0..8 {
            if !point_in_polygon(candidate, polygon) {
                break;
            }
            candidate = [
                candidate[0] + normal[0] * step,
                candidate[1] + normal[1] * step,
            ];
        }
        let dx = candidate[0] - point[0];
        let dy = candidate[1] - point[1];
        let distance = (dx * dx + dy * dy).sqrt();
        if distance < best_distance {
            best_distance = distance;
            best = candidate;
        }
    }
    best
}

/// Whether a point lies on any edge of the polygon.
pub fn point_on_polygon_boundary(point: Point2, polygon: &[Point2]) -> bool {
    let n = polygon.len();
    (0..n).any(|index| point_on_segment(point, polygon[index], polygon[(index + 1) % n]))
}

/// Whether the straight segment can be flown given these expanded polygons.
///
/// Being *on* a polygon's boundary is not being inside it: the endpoint checks
/// deliberately exempt boundary points, because an expanded corner is itself a
/// graph node and every edge leaving it starts on a boundary. Without the
/// exemption the graph would have no usable vertices at all.
///
/// A crossing counts only if it spans more than `1e-6` of the segment, so
/// grazing a corner is not an intersection.
pub fn segment_is_free(start: Point2, end: Point2, polygons: &[Vec<Point2>]) -> bool {
    for polygon in polygons {
        if point_in_polygon(start, polygon) && !point_on_polygon_boundary(start, polygon) {
            return false;
        }
        if point_in_polygon(end, polygon) && !point_on_polygon_boundary(end, polygon) {
            return false;
        }
        if segment_polygon_intervals(start, end, polygon)
            .iter()
            .any(|(entry, exit)| (exit - entry) > 1.0e-6)
        {
            return false;
        }
    }
    true
}

/// Adjacency lists in insertion order. See the module docstring: the order is
/// the contract, not an implementation detail.
pub fn build_visibility_graph(
    nodes: &[Point2],
    polygon_node_indices: &[Vec<usize>],
    polygons: &[Vec<Point2>],
) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for left in 0..nodes.len() {
        for right in (left + 1)..nodes.len() {
            if !segment_is_free(nodes[left], nodes[right], polygons) {
                continue;
            }
            adjacency[left].push(right);
            adjacency[right].push(left);
        }
    }
    // Polygon edges are added afterwards, and only when absent -- so a pair
    // already linked as visible keeps its earlier position in the list.
    for indices in polygon_node_indices {
        if indices.len() < 2 {
            continue;
        }
        for position in 0..indices.len() {
            let current = indices[position];
            let next = indices[(position + 1) % indices.len()];
            if !adjacency[current].contains(&next) {
                adjacency[current].push(next);
            }
            if !adjacency[next].contains(&current) {
                adjacency[next].push(current);
            }
        }
    }
    adjacency
}

/// One entry in the A* frontier.
///
/// `BinaryHeap` is a max-heap ordered by `Ord`, and Python's `heapq` is a
/// min-heap over the tuple `(f, g, node, previous)`. `Ord` is therefore
/// implemented as the REVERSE of that tuple comparison, field by field in the
/// same order, so the two pop the same entry every time. The float fields use
/// `total_cmp`: the costs here are finite, and a partial comparison would need
/// an unwrap that could only ever be a latent panic.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Frontier {
    estimate: f64,
    cost: f64,
    node: usize,
    previous: i64,
}

impl Eq for Frontier {}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .estimate
            .total_cmp(&self.estimate)
            .then(other.cost.total_cmp(&self.cost))
            .then(other.node.cmp(&self.node))
            .then(other.previous.cmp(&self.previous))
    }
}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn distance(a: Point2, b: Point2) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// A* from node 0 to node 1, with a turn penalty.
///
/// The search state is `(node, previous)`, not `node` alone. It has to be: the
/// turn penalty makes the cost of leaving a node depend on how it was entered,
/// so two arrivals at the same node by different predecessors are genuinely
/// different states and collapsing them would mis-price the turn.
///
/// Returns the node indices of the path, or `None` if the goal is unreachable.
/// Unreachable is a real outcome here rather than a degenerate one: a cluster
/// of obstacles whose expanded footprints overlap swallows its own corner
/// nodes, and the reference returns no route even though one exists around the
/// cluster. See `docs/migration/VISIBILITY_ROUTING.md`.
pub fn astar_path(
    nodes: &[Point2],
    adjacency: &[Vec<usize>],
    config: &RouteConfig,
) -> Option<Vec<usize>> {
    const GOAL: usize = 1;
    if nodes.len() < 2 {
        return None;
    }
    let start_state = (0usize, -1i64);
    let mut best_cost: HashMap<(usize, i64), f64> = HashMap::new();
    let mut parents: HashMap<(usize, i64), Option<(usize, i64)>> = HashMap::new();
    best_cost.insert(start_state, 0.0);
    parents.insert(start_state, None);

    let mut queue = BinaryHeap::new();
    queue.push(Frontier {
        estimate: distance(nodes[0], nodes[GOAL]),
        cost: 0.0,
        node: 0,
        previous: -1,
    });

    let mut goal_state: Option<(usize, i64)> = None;
    while let Some(current) = queue.pop() {
        let state = (current.node, current.previous);
        if current.cost > best_cost.get(&state).copied().unwrap_or(f64::INFINITY) + 1.0e-9 {
            continue;
        }
        if current.node == GOAL {
            goal_state = Some(state);
            break;
        }
        for &neighbor in adjacency.get(current.node).into_iter().flatten() {
            let edge_length = distance(nodes[current.node], nodes[neighbor]);
            let mut turn_penalty = 0.0;
            if current.previous >= 0 {
                let previous = nodes[current.previous as usize];
                let here = nodes[current.node];
                let there = nodes[neighbor];
                let incoming = [here[0] - previous[0], here[1] - previous[1]];
                let outgoing = [there[0] - here[0], there[1] - here[1]];
                let incoming_norm = (incoming[0] * incoming[0] + incoming[1] * incoming[1]).sqrt();
                let outgoing_norm = (outgoing[0] * outgoing[0] + outgoing[1] * outgoing[1]).sqrt();
                let denominator = (incoming_norm * outgoing_norm).max(1.0e-9);
                let dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1];
                let cosine = (dot / denominator).clamp(-1.0, 1.0);
                turn_penalty = config.turn_penalty_m * cosine.acos();
            }
            let next_cost = current.cost + edge_length + turn_penalty;
            let next_state = (neighbor, current.node as i64);
            // Strictly better, by more than the epsilon. This is what makes
            // adjacency ORDER decide between two equal-cost routes.
            if next_cost + 1.0e-9 >= best_cost.get(&next_state).copied().unwrap_or(f64::INFINITY) {
                continue;
            }
            best_cost.insert(next_state, next_cost);
            parents.insert(next_state, Some(state));
            queue.push(Frontier {
                estimate: next_cost + distance(nodes[neighbor], nodes[GOAL]),
                cost: next_cost,
                node: neighbor,
                previous: current.node as i64,
            });
        }
    }

    let mut indices = Vec::new();
    let mut cursor = goal_state?;
    loop {
        indices.push(cursor.0);
        match parents.get(&cursor).copied().flatten() {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    indices.reverse();
    Some(indices)
}

/// Drop intermediate points the route can see past.
///
/// Greedy and forward-only: from each kept point it takes the FARTHEST later
/// point still directly reachable, walking the candidate backwards from the
/// end. That is not the same as dropping any point whose neighbours can see
/// each other, and it is what makes the result depend on where smoothing
/// starts.
pub fn smooth_path(points: &[Point2], polygons: &[Vec<Point2>]) -> Vec<Point2> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut smoothed = vec![points[0]];
    let mut index = 0;
    while index < points.len() - 1 {
        let mut next = points.len() - 1;
        while next > index + 1 {
            if segment_is_free(points[index], points[next], polygons) {
                break;
            }
            next -= 1;
        }
        smoothed.push(points[next]);
        index = next;
    }
    smoothed
}

/// Total length of a polyline.
pub fn path_length(points: &[Point2]) -> f64 {
    points
        .windows(2)
        .map(|pair| distance(pair[0], pair[1]))
        .sum()
}
