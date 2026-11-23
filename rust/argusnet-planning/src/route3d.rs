//! Routes in three dimensions over what a map believes.
//!
//! The visibility-graph planner ([`crate::planner`]) goes round obstacles in
//! plan, and a height governor climbs over what a flight meets. Neither
//! weighs the one against the other, and a governor starts to climb only when
//! what is ahead comes into its reach. This is A* over plan position and
//! height together, costed with the aircraft's own energy model: round, over
//! and under (a wire has an under) compete.
//!
//! The space is asked for a column at a time ([`Airspace`]): the lowest height
//! flown there, and the band a wire closes. Columns are asked for only as the
//! search reaches them. Everything is deterministic: ties go to the earlier
//! node.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use serde::{Deserialize, Serialize};

use crate::battery::BatteryModel;

/// Where flight is allowed over one place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Column {
    /// The lowest height flown, metres.
    pub floor_m: f64,
    /// Heights between these are closed.
    pub closed_m: Option<[f64; 2]>,
}

impl Column {
    pub fn allows(&self, z_m: f64) -> bool {
        z_m >= self.floor_m - 1e-9
            && !self
                .closed_m
                .is_some_and(|[low, high]| z_m > low + 1e-9 && z_m < high - 1e-9)
    }

    /// Whether going from one height to the other here passes through what is
    /// closed, whole.
    fn crosses(&self, a_m: f64, b_m: f64) -> bool {
        self.closed_m
            .is_some_and(|[low, high]| a_m.min(b_m) <= low + 1e-9 && a_m.max(b_m) >= high - 1e-9)
    }
}

/// What is known of the air over a place.
pub trait Airspace {
    /// Over the square of `half_m` about `xy_m`; `None` off the map.
    fn column(&self, xy_m: [f64; 2], half_m: f64) -> Option<Column>;
}

/// How a route is searched for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route3dConfig {
    /// The search grid's spacing in plan, metres.
    pub cell_m: f64,
    /// And in height.
    pub layer_m: f64,
    /// The steepest a leg climbs or descends, rise over run. A step in plan
    /// changes height by one layer at most.
    pub max_gradient: f64,
    /// Within this of either end, what stands on the ground is the end's own
    /// business (a pad, a point to inspect): only wires close the air there.
    pub free_radius_m: f64,
    /// The most nodes searched before giving up.
    pub max_nodes: usize,
    /// How far above the ends a route may climb, metres.
    pub max_climb_m: f64,
    /// How much further than what they are flown clear of by a route keeps
    /// from wires: a map's wire is a fit.
    pub wire_margin_m: f64,
    /// The share of the straight flight's energy a route must save: the
    /// straight one is shorter, and a height governor flies it over what it
    /// meets. One whose straight flight meets a wire is flown if it costs no
    /// more: there a governor has only what it sees coming.
    pub min_saving: f64,
    /// Plan a flight again when the map closes its way. Absent plans it once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replan: Option<ReplanConfig>,
}

/// When a flight under way is planned again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplanConfig {
    /// Seconds between looks at the way a flight has left.
    pub check_s: f64,
    /// How long planning is declared to take, seconds: what it finds is flown
    /// that much later, on any machine.
    pub latency_s: f64,
    /// The most flights being planned at once.
    pub capacity: usize,
}

impl Default for ReplanConfig {
    fn default() -> Self {
        Self {
            check_s: 1.0,
            latency_s: 0.5,
            capacity: 16,
        }
    }
}

impl Default for Route3dConfig {
    fn default() -> Self {
        Self {
            cell_m: 20.0,
            layer_m: 10.0,
            max_gradient: 0.75,
            free_radius_m: 40.0,
            max_nodes: 200_000,
            max_climb_m: 200.0,
            wire_margin_m: 10.0,
            min_saving: 0.05,
            replan: None,
        }
    }
}

impl Route3dConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("cell_m", self.cell_m),
            ("layer_m", self.layer_m),
            ("max_gradient", self.max_gradient),
            ("max_climb_m", self.max_climb_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("route {name} cannot be {value}"));
            }
        }
        for (name, value) in [
            ("free_radius_m", self.free_radius_m),
            ("wire_margin_m", self.wire_margin_m),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("route {name} cannot be {value}"));
            }
        }
        if !(self.min_saving.is_finite() && (0.0..1.0).contains(&self.min_saving)) {
            return Err(format!("route min_saving cannot be {}", self.min_saving));
        }
        if let Some(replan) = &self.replan {
            for (name, value) in [("check_s", replan.check_s), ("latency_s", replan.latency_s)] {
                if !(value.is_finite() && value > 0.0) {
                    return Err(format!("route replan {name} cannot be {value}"));
                }
            }
            if replan.capacity == 0 {
                return Err("a route replan of no flights replans none".into());
            }
        }
        if self.max_nodes == 0 {
            return Err("a route search of no nodes finds nothing".into());
        }
        if self.layer_m > self.max_gradient * self.cell_m + 1e-9 {
            return Err(format!(
                "a layer of {} m over a cell of {} m is steeper than {}",
                self.layer_m, self.cell_m, self.max_gradient
            ));
        }
        Ok(())
    }
}

/// A route found: the corners between its ends, each with its height.
#[derive(Debug, Clone, PartialEq)]
pub struct Route3d {
    pub corners_m: Vec<[f64; 3]>,
    /// Its length in plan, metres.
    pub length_m: f64,
    pub cost_wh: f64,
    /// What the straight flight would cost, climbing over what it meets.
    pub direct_wh: f64,
    /// Whether the straight flight meets a wire.
    pub meets_wire: bool,
    /// Nodes expanded to find it.
    pub expanded: usize,
}

/// What the straight flight and the search say of a trip.
#[derive(Debug, Clone, PartialEq)]
pub enum Plan3d {
    /// The straight line is open: nothing to plan.
    Straight,
    /// Nothing found saves enough over the straight flight climbed over what
    /// is there.
    NotWorth,
    Route(Route3d),
    /// Nothing was found.
    None,
}

struct Search<'a, A: Airspace> {
    air: &'a A,
    config: &'a Route3dConfig,
    start: [f64; 3],
    goal: [f64; 3],
    columns: BTreeMap<(i32, i32), Option<Column>>,
}

impl<A: Airspace> Search<'_, A> {
    fn xy(&self, i: i32, j: i32) -> [f64; 2] {
        [
            self.start[0] + f64::from(i) * self.config.cell_m,
            self.start[1] + f64::from(j) * self.config.cell_m,
        ]
    }

    fn free(&self, xy: [f64; 2]) -> bool {
        [self.start, self.goal]
            .iter()
            .any(|end| (xy[0] - end[0]).hypot(xy[1] - end[1]) <= self.config.free_radius_m)
    }

    /// The column at grid place `(i, j)`, asked for once.
    fn column(&mut self, i: i32, j: i32) -> Option<Column> {
        if let Some(known) = self.columns.get(&(i, j)) {
            return *known;
        }
        let xy = self.xy(i, j);
        let mut column = self.air.column(xy, 0.5 * self.config.cell_m);
        if self.free(xy) {
            if let Some(column) = &mut column {
                column.floor_m = f64::NEG_INFINITY;
            }
        }
        self.columns.insert((i, j), column);
        column
    }

    /// Whether a straight leg is open all along: in plan every quarter cell, in
    /// height every half layer, and across no closed band between.
    fn open(&mut self, from: [f64; 3], to: [f64; 3]) -> bool {
        let steps = self.samples(from, to, 0.25 * self.config.cell_m);
        let mut last: Option<(f64, Column)> = None;
        (0..=steps).all(|k| {
            let f = k as f64 / steps as f64;
            let at: [f64; 3] = std::array::from_fn(|a| from[a] + f * (to[a] - from[a]));
            let i = ((at[0] - self.start[0]) / self.config.cell_m).round() as i32;
            let j = ((at[1] - self.start[1]) / self.config.cell_m).round() as i32;
            let Some(column) = self.column(i, j) else {
                return false;
            };
            let open = column.allows(at[2])
                && last.is_none_or(|(z_m, before)| {
                    !column.crosses(z_m, at[2]) && !before.crosses(z_m, at[2])
                });
            last = Some((at[2], column));
            open
        })
    }

    /// How many samples a leg takes, no further apart than `plan_m` in plan or
    /// half a layer in height.
    fn samples(&self, from: [f64; 3], to: [f64; 3], plan_m: f64) -> usize {
        let length = (to[0] - from[0]).hypot(to[1] - from[1]);
        let rise = (to[2] - from[2]).abs();
        (length / plan_m)
            .max(rise / (0.5 * self.config.layer_m))
            .ceil()
            .max(1.0) as usize
    }

    /// Whether the straight flight is steep enough to pass from one side of a
    /// wire to the other over one place.
    fn drops_through_a_wire(&self) -> bool {
        let (from, to) = (self.start, self.goal);
        let steps = self.samples(from, to, 0.5 * self.config.cell_m);
        let raised = |k: usize| {
            let f = k as f64 / steps as f64;
            let at: [f64; 3] = std::array::from_fn(|a| from[a] + f * (to[a] - from[a]));
            let column = self.air.column([at[0], at[1]], 0.0);
            let z_m = match column {
                Some(column) if !column.allows(at[2]) => {
                    column.closed_m.map_or(at[2], |[_, high]| at[2].max(high))
                }
                _ => at[2],
            };
            (z_m, column)
        };
        (1..=steps).any(|k| {
            let ((a_m, before), (b_m, here)) = (raised(k - 1), raised(k));
            [before, here]
                .iter()
                .all(|column| column.is_some_and(|column| column.crosses(a_m, b_m)))
        })
    }

    /// The energy of the straight flight, climbed over what stands under it
    /// as a height governor would.
    /// Whether it meets a wire on the way.
    fn direct_wh(&self, battery: &BatteryModel) -> (f64, bool) {
        let (from, to) = (self.start, self.goal);
        let length = (to[0] - from[0]).hypot(to[1] - from[1]);
        // Down through a wire, over the one place, there is no climbing over it.
        if self.drops_through_a_wire() {
            return (f64::INFINITY, true);
        }
        let step_m = 0.5 * self.config.cell_m;
        let steps = (length / step_m).ceil().max(1.0) as usize;
        let mut wire = false;
        let mut points: Vec<[f64; 3]> = (0..=steps)
            .map(|k| {
                let f = k as f64 / steps as f64;
                let mut at: [f64; 3] = std::array::from_fn(|a| from[a] + f * (to[a] - from[a]));
                let xy = [at[0], at[1]];
                if let Some(column) = self.air.column(xy, 0.0) {
                    if !self.free(xy) {
                        at[2] = at[2].max(column.floor_m);
                    }
                    if let (false, Some([_, high])) = (column.allows(at[2]), column.closed_m) {
                        at[2] = at[2].max(high);
                        wire = true;
                    }
                }
                at
            })
            .collect();
        // Its heights reached at the gradient it can climb at, ahead and
        // after.
        let rise = self.config.max_gradient * length / steps as f64;
        for k in (0..steps).rev() {
            points[k][2] = points[k][2].max(points[k + 1][2] - rise);
        }
        for k in 1..=steps {
            points[k][2] = points[k][2].max(points[k - 1][2] - rise);
        }
        (cost_wh(battery, &points), wire)
    }

    fn gentle(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        let length = (to[0] - from[0]).hypot(to[1] - from[1]);
        (to[2] - from[2]).abs() <= self.config.max_gradient * length + 1e-9
    }
}

/// A place in the search: steps east, north and up from the start.
type Node = (i32, i32, i32);

/// A node waiting to be expanded, cheapest first, then earliest found.
struct Waiting {
    total_wh: f64,
    order: usize,
    node: Node,
}

impl PartialEq for Waiting {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Waiting {}
impl PartialOrd for Waiting {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Waiting {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .total_wh
            .total_cmp(&self.total_wh)
            .then(other.order.cmp(&self.order))
    }
}

/// The energy of flying `points` in order.
fn cost_wh(battery: &BatteryModel, points: &[[f64; 3]]) -> f64 {
    points
        .windows(2)
        .map(|leg| battery.segment_cost_wh(leg[0], leg[1]))
        .sum()
}

/// Whether `way_m`, its end last, is open all along from `start`.
pub fn way_is_open(
    air: &impl Airspace,
    start: [f64; 3],
    way_m: &[[f64; 3]],
    config: &Route3dConfig,
) -> bool {
    let Some(goal) = way_m.last().copied() else {
        return true;
    };
    let mut search = Search {
        air,
        config,
        start,
        goal,
        columns: BTreeMap::new(),
    };
    let mut from = start;
    way_m.iter().all(|&to| {
        let open = search.open(from, to);
        from = to;
        open
    })
}

/// The cheapest way found from `start` to `goal` through `air`.
pub fn plan(
    air: &impl Airspace,
    battery: &BatteryModel,
    start: [f64; 3],
    goal: [f64; 3],
    config: &Route3dConfig,
) -> Plan3d {
    let mut search = Search {
        air,
        config,
        start,
        goal,
        columns: BTreeMap::new(),
    };
    if search.open(start, goal) {
        return Plan3d::Straight;
    }
    // The grid is laid from the start; the goal is joined from any node within
    // a cell and a half of it, however steeply: an aircraft that hovers comes
    // down over its end, and does not circle down to it.
    let z = |k: i32| start[2] + f64::from(k) * config.layer_m;
    let top_m = start[2].max(goal[2]) + config.max_climb_m;
    let low_m = start[2].min(goal[2]) - config.max_climb_m;
    let point = |search: &Search<'_, _>, (i, j, k): Node| {
        let xy = search.xy(i, j);
        [xy[0], xy[1], z(k)]
    };
    // Straight there is the least it can cost: climbing costs no less than
    // the change of height there is to make.
    let least = |at: [f64; 3]| battery.segment_cost_wh(at, goal);
    let root = (0, 0, 0);
    let mut best: BTreeMap<Node, (f64, Option<Node>)> = BTreeMap::new();
    let mut waiting = BinaryHeap::new();
    let mut order = 0;
    best.insert(root, (0.0, None));
    waiting.push(Waiting {
        total_wh: least(start),
        order,
        node: root,
    });
    let mut closed = std::collections::BTreeSet::new();
    let mut found = None;
    while let Some(Waiting { node, .. }) = waiting.pop() {
        if !closed.insert(node) {
            continue;
        }
        if closed.len() > config.max_nodes {
            break;
        }
        let here = point(&search, node);
        let near = (goal[0] - here[0]).hypot(goal[1] - here[1]) <= 1.5 * config.cell_m;
        if near && search.open(here, goal) {
            found = Some(node);
            break;
        }
        let so_far = best[&node].0;
        for di in -1..=1 {
            for dj in -1..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let (i, j) = (node.0 + di, node.1 + dj);
                let Some(column) = search.column(i, j) else {
                    continue;
                };
                for dk in -1..=1 {
                    let next = (i, j, node.2 + dk);
                    let there = point(&search, next);
                    if there[2] > top_m
                        || there[2] < low_m
                        || !column.allows(there[2])
                        || closed.contains(&next)
                        || !search.open(here, there)
                    {
                        continue;
                    }
                    let cost = so_far + battery.segment_cost_wh(here, there);
                    if best.get(&next).is_some_and(|(known, _)| *known <= cost) {
                        continue;
                    }
                    best.insert(next, (cost, Some(node)));
                    order += 1;
                    waiting.push(Waiting {
                        total_wh: cost + least(there),
                        order,
                        node: next,
                    });
                }
            }
        }
    }
    let Some(mut node) = found else {
        return Plan3d::None;
    };
    let mut points = vec![goal];
    loop {
        points.push(point(&search, node));
        match best[&node].1 {
            Some(previous) => node = previous,
            None => break,
        }
    }
    points.reverse();
    points[0] = start;
    // Pulled straight: from each corner, the furthest reached by an open leg
    // that costs no more than the way through what it skips.
    let mut pulled = vec![points[0]];
    let mut at = 0;
    while at + 1 < points.len() {
        let mut to = at + 1;
        for far in (at + 2..points.len()).rev() {
            let (a, b) = (points[at], points[far]);
            if search.gentle(a, b)
                && search.open(a, b)
                && battery.segment_cost_wh(a, b) <= cost_wh(battery, &points[at..=far]) + 1e-9
            {
                to = far;
                break;
            }
        }
        pulled.push(points[to]);
        at = to;
    }
    let length_m = pulled
        .windows(2)
        .map(|leg| (leg[1][0] - leg[0][0]).hypot(leg[1][1] - leg[0][1]))
        .sum();
    let cost = cost_wh(battery, &pulled);
    pulled.pop();
    pulled.remove(0);
    let (direct_wh, wire) = search.direct_wh(battery);
    let saving = if wire { 0.0 } else { config.min_saving };
    if cost > (1.0 - saving.min(config.min_saving)) * direct_wh {
        return Plan3d::NotWorth;
    }
    Plan3d::Route(Route3d {
        corners_m: pulled,
        length_m,
        cost_wh: cost,
        direct_wh,
        meets_wire: wire,
        expanded: closed.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat ground flown 10 m over, with boxes on it and wires across it.
    #[derive(Default)]
    struct Boxes {
        /// `[x0, x1, y0, y1, top]`.
        boxes: Vec<[f64; 5]>,
        /// `[x0, x1, y0, y1, low, high]` closed.
        wires: Vec<[f64; 6]>,
        half_m: f64,
    }

    impl Airspace for Boxes {
        fn column(&self, xy: [f64; 2], half: f64) -> Option<Column> {
            if xy[0].abs() > self.half_m || xy[1].abs() > self.half_m {
                return None;
            }
            let over = |b: &[f64]| {
                xy[0] + half > b[0]
                    && xy[0] - half < b[1]
                    && xy[1] + half > b[2]
                    && xy[1] - half < b[3]
            };
            let mut column = Column {
                floor_m: 10.0,
                closed_m: None,
            };
            for b in self.boxes.iter().filter(|b| over(&b[..])) {
                column.floor_m = column.floor_m.max(b[4] + 10.0);
            }
            for w in self.wires.iter().filter(|w| over(&w[..])) {
                column.closed_m = Some([w[4], w[5]]);
            }
            Some(column)
        }
    }

    fn world(boxes: Vec<[f64; 5]>, wires: Vec<[f64; 6]>) -> Boxes {
        Boxes {
            boxes,
            wires,
            half_m: 1000.0,
        }
    }

    fn battery() -> BatteryModel {
        BatteryModel::default()
    }

    const START: [f64; 3] = [-300.0, 0.0, 40.0];
    const GOAL: [f64; 3] = [300.0, 0.0, 40.0];

    /// The way found, whatever it saves.
    fn any() -> Route3dConfig {
        Route3dConfig {
            min_saving: -10.0,
            ..Route3dConfig::default()
        }
    }

    fn route(air: &Boxes) -> Route3d {
        match plan(air, &battery(), START, GOAL, &any()) {
            Plan3d::Route(route) => route,
            other => panic!("{other:?}"),
        }
    }

    /// Every point of every leg open in the space itself.
    fn flies_clear(air: &Boxes, route: &Route3d) {
        let mut points = vec![START];
        points.extend(&route.corners_m);
        points.push(GOAL);
        for leg in points.windows(2) {
            for k in 0..=200 {
                let f = f64::from(k) / 200.0;
                let at: [f64; 3] = std::array::from_fn(|a| leg[0][a] + f * (leg[1][a] - leg[0][a]));
                let column = air.column([at[0], at[1]], 0.0).unwrap();
                assert!(column.allows(at[2]), "{at:?} in {column:?}");
            }
            let run = (leg[1][0] - leg[0][0]).hypot(leg[1][1] - leg[0][1]);
            if leg[1] == GOAL {
                continue;
            }
            let gradient = (leg[1][2] - leg[0][2]).abs() / run;
            assert!(
                gradient <= Route3dConfig::default().max_gradient + 1e-9,
                "{gradient}"
            );
        }
    }

    #[test]
    fn an_open_line_is_left_alone() {
        let low = world(vec![[-20.0, 20.0, -20.0, 20.0, 25.0]], vec![]);
        let plan = plan(&low, &battery(), START, GOAL, &Route3dConfig::default());
        assert_eq!(plan, Plan3d::Straight);
    }

    #[test]
    fn a_narrow_tower_is_gone_round_and_a_long_ridge_over() {
        let tower = world(vec![[-15.0, 15.0, -15.0, 15.0, 180.0]], vec![]);
        let round = route(&tower);
        flies_clear(&tower, &round);
        assert!(!round.meets_wire);
        assert!(
            round.corners_m.iter().all(|c| c[2] < 60.0),
            "round, not over: {round:?}"
        );
        assert!(round.length_m < 640.0, "{round:?}");

        let ridge = world(vec![[-15.0, 15.0, -1000.0, 1000.0, 60.0]], vec![]);
        let over = route(&ridge);
        flies_clear(&ridge, &over);
        assert!(over.corners_m.iter().any(|c| c[2] >= 70.0), "{over:?}");
        assert!(over.length_m < 610.0, "straight over: {over:?}");
    }

    /// Straight over a ridge is what a governor flies anyway, and is left to
    /// it. Round a tower it would climb is worth a route.
    #[test]
    fn a_route_that_saves_nothing_is_not_flown() {
        let config = Route3dConfig::default();
        let ridge = world(vec![[-15.0, 15.0, -1000.0, 1000.0, 60.0]], vec![]);
        assert_eq!(
            plan(&ridge, &battery(), START, GOAL, &config),
            Plan3d::NotWorth
        );
        let tower = world(vec![[-15.0, 15.0, -15.0, 15.0, 180.0]], vec![]);
        let Plan3d::Route(round) = plan(&tower, &battery(), START, GOAL, &config) else {
            panic!("a route")
        };
        assert!(round.cost_wh <= 0.95 * round.direct_wh, "{round:?}");
        // What the straight flight costs: up 150 m and down again, at least.
        let climbed = battery().segment_cost_wh(START, GOAL)
            + battery().climb_cost_wh(150.0)
            + battery().climb_cost_wh(-150.0);
        assert!(
            (round.direct_wh - climbed).abs() < 0.05 * climbed,
            "{round:?}"
        );
    }

    #[test]
    fn a_wire_across_the_way_is_passed_under_where_there_is_room() {
        // Closed from 35 m, so the straight line at 40 m meets it.
        let wire = |low: f64| world(vec![], vec![[-15.0, 15.0, -1000.0, 1000.0, low, 120.0]]);
        let under = wire(35.0);
        let found = route(&under);
        assert!(found.meets_wire);
        flies_clear(&under, &found);
        assert!(found.corners_m.iter().all(|c| c[2] <= 40.0), "{found:?}");
        // No room between the ground and the wire: over it.
        let over = wire(5.0);
        let found = route(&over);
        flies_clear(&over, &found);
        assert!(found.corners_m.iter().any(|c| c[2] >= 120.0), "{found:?}");
    }

    /// Straight down, or steeply, through a wire: every sample is clear of it
    /// and the way is not. Round it, to the far side and under.
    #[test]
    fn a_wire_is_not_stepped_over_by_a_steep_leg() {
        let wire = world(vec![], vec![[-30.0, 30.0, -1000.0, 1000.0, 35.0, 65.0]]);
        let config = Route3dConfig::default();
        for (start, goal) in [
            ([0.0, 0.0, 100.0], [0.0, 0.0, 10.0]),
            ([0.0, 0.0, 10.0], [0.0, 0.0, 100.0]),
            ([-2.0, 0.0, 100.0], [2.0, 1.0, 12.0]),
            ([20.0, 0.0, 70.0], [0.0, 0.0, 30.0]),
        ] {
            match plan(&wire, &battery(), start, goal, &config) {
                Plan3d::Route(route) => {
                    assert!(route.meets_wire);
                    let mut points = vec![start];
                    points.extend(&route.corners_m);
                    points.push(goal);
                    for leg in points.windows(2) {
                        for k in 0..=400 {
                            let f = f64::from(k) / 400.0;
                            let at: [f64; 3] =
                                std::array::from_fn(|a| leg[0][a] + f * (leg[1][a] - leg[0][a]));
                            let column = wire.column([at[0], at[1]], 0.0).unwrap();
                            assert!(column.allows(at[2]), "{at:?} of {route:?}");
                        }
                    }
                }
                other => panic!("{start:?} to {goal:?}: {other:?}"),
            }
        }
        // Clear of it, straight down is straight down.
        let clear = plan(
            &wire,
            &battery(),
            [200.0, 0.0, 100.0],
            [200.0, 0.0, 10.0],
            &config,
        );
        assert_eq!(clear, Plan3d::Straight);
    }

    #[test]
    fn it_costs_what_it_says_and_no_more_than_the_ways_there_are() {
        let tower = world(vec![[-15.0, 15.0, -15.0, 15.0, 60.0]], vec![]);
        let found = route(&tower);
        let mut points = vec![START];
        points.extend(&found.corners_m);
        points.push(GOAL);
        assert!((cost_wh(&battery(), &points) - found.cost_wh).abs() < 1e-9);
        let over = [START, [-40.0, 0.0, 70.0], [40.0, 0.0, 70.0], GOAL];
        let round = [START, [0.0, 45.0, 40.0], GOAL];
        for way in [&over[..], &round[..]] {
            let way = cost_wh(&battery(), way);
            assert!(
                found.cost_wh <= way * 1.03,
                "{} against {way}",
                found.cost_wh
            );
        }
    }

    #[test]
    fn a_way_that_is_not_there_is_not_found_and_the_search_is_bounded() {
        let walled = world(vec![[-15.0, 15.0, -2000.0, 2000.0, 900.0]], vec![]);
        let config = Route3dConfig::default();
        assert_eq!(
            plan(&walled, &battery(), START, GOAL, &config),
            Plan3d::None
        );
        let short = Route3dConfig {
            max_nodes: 5,
            ..config
        };
        let tower = world(vec![[-15.0, 15.0, -15.0, 15.0, 180.0]], vec![]);
        assert_eq!(plan(&tower, &battery(), START, GOAL, &short), Plan3d::None);
    }

    #[test]
    fn a_pad_under_the_floor_is_left_and_a_low_point_come_down_to() {
        let air = world(vec![[-15.0, 15.0, -1000.0, 1000.0, 60.0]], vec![]);
        let (start, goal) = ([-300.0, 0.0, 0.5], [300.0, 0.0, 3.0]);
        let Plan3d::Route(route) = plan(&air, &battery(), start, goal, &any()) else {
            panic!("a route")
        };
        assert!(route.corners_m.iter().any(|c| c[2] >= 70.0), "{route:?}");
    }

    /// High over its end, it comes down over it: no circling to lose height.
    #[test]
    fn it_comes_down_over_its_end() {
        let air = world(vec![[-15.0, 15.0, -1000.0, 1000.0, 120.0]], vec![]);
        let (start, goal) = ([-300.0, 0.0, 130.0], [60.0, 0.0, 2.0]);
        let Plan3d::Route(route) = plan(&air, &battery(), start, goal, &any()) else {
            panic!("a route")
        };
        assert!(route.length_m < 365.0, "{route:?}");
    }

    #[test]
    fn the_same_trip_is_planned_the_same() {
        let air = world(
            vec![
                [-15.0, 15.0, -60.0, 60.0, 180.0],
                [-120.0, -90.0, 20.0, 200.0, 90.0],
            ],
            vec![[60.0, 80.0, -1000.0, 1000.0, 35.0, 70.0]],
        );
        let config = Route3dConfig::default();
        let first = plan(&air, &battery(), START, GOAL, &config);
        assert!(matches!(first, Plan3d::Route(_)));
        for _ in 0..3 {
            assert_eq!(plan(&air, &battery(), START, GOAL, &config), first);
        }
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        let good = Route3dConfig::default();
        assert!(good.validate().is_ok());
        for bad in [
            Route3dConfig {
                cell_m: 0.0,
                ..good.clone()
            },
            Route3dConfig {
                layer_m: 30.0,
                ..good.clone()
            },
            Route3dConfig {
                max_nodes: 0,
                ..good.clone()
            },
            Route3dConfig {
                free_radius_m: f64::NAN,
                ..good
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
