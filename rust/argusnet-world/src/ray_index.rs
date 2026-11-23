//! Which solid a ray meets first, without testing every solid near it.
//!
//! [`ObstacleLayer::query`](crate::index::ObstacleLayer::query) answers with
//! everything in a ray's bounding box. A long diagonal ray's box holds most
//! of a town, and every footprint in it was then intersected exactly. This is
//! a bounding-volume tree over the hard obstacles' 3D boxes, walked nearest
//! box first and cut off at the nearest hit so far, so the exact test runs
//! only on the few solids whose box the ray enters.
//!
//! The boxes only choose what to test. The answer is
//! [`Obstacle::surface_hit_t`]'s, and ties go to the lowest obstacle index, so
//! it does not depend on the shape of the tree or the order it is walked in.
//!
//! One case is refused: a hit that is not on the solid's own box. A ray that
//! misses a corner by less than the exact test's edge tolerance, with its
//! midpoint at that corner, crosses no edge and yet counts as inside for its
//! whole length, so the test reports a solid somewhere it does not reach.

use crate::obstacles::Obstacle;

/// How far a box stands outside its solid, metres: more than the tolerances
/// of the exact test, so a box never rejects a hit on its solid.
const BOX_MARGIN_M: f64 = 1.0e-3;

/// How far past the nearest hit a box may start and still be tested, as a
/// fraction of the segment: more than the exact test's own rounding, so a
/// tie is always seen.
const PRUNE_SLACK: f64 = 1.0e-6;

const LEAF_SIZE: usize = 4;

/// Deeper than any tree [`RayIndex::new`] builds: it halves at each level.
const MAX_STACK: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Box3 {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Box3 {
    fn of(obstacle: &Obstacle) -> Self {
        let b = obstacle.bounds_xy_m();
        let m = BOX_MARGIN_M;
        Self {
            lo: [b.x_min_m - m, b.y_min_m - m, obstacle.base_z_m - m],
            hi: [b.x_max_m + m, b.y_max_m + m, obstacle.top_z_m + m],
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            lo: std::array::from_fn(|k| self.lo[k].min(other.lo[k])),
            hi: std::array::from_fn(|k| self.hi[k].max(other.hi[k])),
        }
    }

    fn center(&self, axis: usize) -> f64 {
        0.5 * (self.lo[axis] + self.hi[axis])
    }
}

/// A segment, ready to be tested against boxes.
struct Ray {
    from: [f64; 3],
    inverse: [f64; 3],
}

impl Ray {
    fn new(from: [f64; 3], to: [f64; 3]) -> Self {
        Self {
            from,
            inverse: std::array::from_fn(|k| 1.0 / (to[k] - from[k])),
        }
    }

    /// Where the ray enters `b`, if it does no later than `limit`.
    fn enters(&self, b: &Box3, limit: f64) -> Option<f64> {
        let (mut enter, mut leave) = (0.0_f64, limit);
        for k in 0..3 {
            if self.inverse[k].is_finite() {
                let one = (b.lo[k] - self.from[k]) * self.inverse[k];
                let other = (b.hi[k] - self.from[k]) * self.inverse[k];
                enter = enter.max(one.min(other));
                leave = leave.min(one.max(other));
            } else if self.from[k] < b.lo[k] || self.from[k] > b.hi[k] {
                // No travel along this axis, and outside the box on it.
                return None;
            }
        }
        (enter <= leave).then_some(enter)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Children {
    /// `order[start..end]`.
    Leaf {
        start: usize,
        end: usize,
    },
    Split {
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    bounds: Box3,
    children: Children,
}

/// The tree over a layer's hard obstacles.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RayIndex {
    nodes: Vec<Node>,
    /// Obstacle index and its box, grouped by leaf.
    order: Vec<(usize, Box3)>,
}

impl RayIndex {
    pub(crate) fn new(primitives: &[Obstacle]) -> Self {
        let mut index = Self {
            nodes: Vec::new(),
            order: primitives
                .iter()
                .enumerate()
                .filter(|(_, o)| o.blocker_type.is_hard())
                .map(|(i, o)| (i, Box3::of(o)))
                .collect(),
        };
        if !index.order.is_empty() {
            index.build(0, index.order.len());
        }
        index
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let items = &mut self.order[start..end];
        let bounds = items
            .iter()
            .map(|(_, b)| *b)
            .reduce(Box3::union)
            .expect("a node holds at least one obstacle");
        let node = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            children: Children::Leaf { start, end },
        });
        if items.len() > LEAF_SIZE {
            let extent = |k: usize| bounds.hi[k] - bounds.lo[k];
            let axis = if extent(0) >= extent(1) { 0 } else { 1 };
            items
                .sort_by(|(i, a), (j, b)| a.center(axis).total_cmp(&b.center(axis)).then(i.cmp(j)));
            let middle = start + (end - start) / 2;
            let left = self.build(start, middle);
            let right = self.build(middle, end);
            self.nodes[node].children = Children::Split { left, right };
        }
        node
    }

    /// The first hard obstacle the segment meets: its index and where along
    /// the segment. `tested` counts the exact tests that took.
    pub(crate) fn first_hit(
        &self,
        primitives: &[Obstacle],
        from: [f64; 3],
        to: [f64; 3],
        tested: &mut usize,
    ) -> Option<(usize, f64)> {
        let ray = Ray::new(from, to);
        let mut best: Option<(usize, f64)> = None;
        let limit = |best: Option<(usize, f64)>| best.map_or(1.0, |(_, t)| t) + PRUNE_SLACK;
        let mut stack = [(0_usize, 0.0_f64); MAX_STACK];
        let mut depth = 0;
        if let Some(enter) = self
            .nodes
            .first()
            .and_then(|root| ray.enters(&root.bounds, limit(best)))
        {
            stack[0] = (0, enter);
            depth = 1;
        }
        while depth > 0 {
            depth -= 1;
            let (node, enter) = stack[depth];
            if enter > limit(best) {
                continue;
            }
            match self.nodes[node].children {
                Children::Leaf { start, end } => {
                    for (i, b) in &self.order[start..end] {
                        let Some(enter) = ray.enters(b, limit(best)) else {
                            continue;
                        };
                        *tested += 1;
                        let Some(t) = primitives[*i].surface_hit_t(from, to) else {
                            continue;
                        };
                        if t < enter - PRUNE_SLACK {
                            continue; // Short of the solid: see the module docs.
                        }
                        if best.is_none_or(|(j, s)| t.total_cmp(&s).then(i.cmp(&j)).is_lt()) {
                            best = Some((*i, t));
                        }
                    }
                }
                Children::Split { left, right } => {
                    let mut next = [left, right].map(|child| {
                        let enter = ray.enters(&self.nodes[child].bounds, limit(best));
                        (child, enter)
                    });
                    // The nearer child is pushed last, so it is walked first.
                    if let (Some(a), Some(b)) = (next[0].1, next[1].1) {
                        if a < b {
                            next.swap(0, 1);
                        }
                    }
                    for (child, enter) in next {
                        if let Some(enter) = enter {
                            stack[depth] = (child, enter);
                            depth += 1;
                        }
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacles::{BlockerType, ObstacleShape, Roof, RoofShape};

    /// A small deterministic stream (SplitMix64).
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            lo + (hi - lo) * (z >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    fn solid(id: usize, shape: ObstacleShape, base_z_m: f64, top_z_m: f64) -> Obstacle {
        Obstacle {
            id: format!("o{id:04}"),
            blocker_type: BlockerType::Building,
            shape,
            base_z_m,
            top_z_m,
            density: None,
            kind: "building".into(),
            roof: None,
        }
    }

    fn rectangle(x: f64, y: f64, w: f64, d: f64) -> ObstacleShape {
        ObstacleShape::Polygon {
            footprint_xy_m: vec![[x, y], [x + w, y], [x + w, y + d], [x, y + d]],
        }
    }

    /// A town 1 km square: concave and convex footprints, cylinders and
    /// pitched roofs, overlapping, with vegetation the index must leave out.
    fn town(rng: &mut Rng, count: usize) -> Vec<Obstacle> {
        (0..count)
            .map(|i| {
                let (x, y) = (rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0));
                let (w, d) = (rng.uniform(5.0, 40.0), rng.uniform(5.0, 40.0));
                let base = rng.uniform(-5.0, 20.0);
                let top = base + rng.uniform(3.0, 60.0);
                let shape = match i % 4 {
                    0 => ObstacleShape::Cylinder {
                        center_xy_m: [x, y],
                        radius_m: w * 0.5,
                    },
                    1 => ObstacleShape::Polygon {
                        // An L.
                        footprint_xy_m: vec![
                            [x, y],
                            [x + w, y],
                            [x + w, y + d * 0.5],
                            [x + w * 0.5, y + d * 0.5],
                            [x + w * 0.5, y + d],
                            [x, y + d],
                        ],
                    },
                    _ => rectangle(x, y, w, d),
                };
                let mut o = solid(i, shape, base, top);
                if i % 4 == 3 {
                    // A shed roof, falling from its ridge along x.
                    let fall = (top - base) * 0.5 / w;
                    o.roof = Some(Roof {
                        shape: RoofShape::Gabled,
                        eave_z_m: base + (top - base) * 0.5,
                        planes: vec![[-fall, 0.0, top + fall * x]],
                    });
                }
                if i % 11 == 5 {
                    o.blocker_type = BlockerType::Vegetation;
                    o.density = Some(0.5);
                }
                o
            })
            .collect()
    }

    /// Every hard obstacle tested, with no index: what the index must answer.
    fn brute(primitives: &[Obstacle], from: [f64; 3], to: [f64; 3]) -> Option<(usize, f64)> {
        primitives
            .iter()
            .enumerate()
            .filter(|(_, o)| o.blocker_type.is_hard())
            .filter_map(|(i, o)| o.surface_hit_t(from, to).map(|t| (i, t)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    #[test]
    fn it_answers_what_testing_every_obstacle_answers() {
        let mut rng = Rng(7);
        let mut hits = 0;
        for count in [0, 1, 3, 4, 5, 60, 600] {
            let primitives = town(&mut rng, count);
            let index = RayIndex::new(&primitives);
            for k in 0..1500 {
                let from: [f64; 3] = [
                    rng.uniform(-50.0, 1050.0),
                    rng.uniform(-50.0, 1050.0),
                    rng.uniform(-10.0, 100.0),
                ];
                let reach = [1.0, 20.0, 250.0, 1500.0][k / 8 % 4];
                let mut to: [f64; 3] =
                    std::array::from_fn(|a| from[a] + rng.uniform(-reach, reach));
                // Half of them at a solid, on past its middle.
                if k % 2 == 0 && count > 0 {
                    let b = Box3::of(&primitives[k % count]);
                    let past = rng.uniform(0.5, 2.0);
                    to = std::array::from_fn(|a| from[a] + (b.center(a) - from[a]) * past);
                }
                // Every way a ray can lie along the axes, down to a point.
                let fixed = k / 32 % 8;
                for axis in 0..3 {
                    if fixed >> axis & 1 == 1 {
                        to[axis] = from[axis];
                    }
                }
                let got = index.first_hit(&primitives, from, to, &mut 0);
                let want = brute(&primitives, from, to);
                assert_eq!(
                    got.map(|(i, t)| (i, t.to_bits())),
                    want.map(|(i, t)| (i, t.to_bits())),
                    "{count} obstacles, {from:?} -> {to:?}"
                );
                hits += usize::from(got.is_some());
            }
        }
        assert!(hits > 1000, "only {hits} rays met anything");
    }

    /// [`brute`], less the hits that are not on the solid they name (see the
    /// module docs).
    fn brute_on_solid(
        primitives: &[Obstacle],
        from: [f64; 3],
        to: [f64; 3],
    ) -> Option<(usize, f64)> {
        let on_box = |o: &Obstacle, t: f64| {
            let b = Box3::of(o);
            (0..3).all(|k| {
                let p = from[k] + (to[k] - from[k]) * t;
                p >= b.lo[k] - BOX_MARGIN_M && p <= b.hi[k] + BOX_MARGIN_M
            })
        };
        primitives
            .iter()
            .enumerate()
            .filter(|(_, o)| o.blocker_type.is_hard())
            .filter_map(|(i, o)| o.surface_hit_t(from, to).map(|t| (i, t)))
            .filter(|(i, t)| on_box(&primitives[*i], *t))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The exact test puts a solid 50 nm off the ray's path at (37, 37, 10),
    /// seven metres past where it ends. The index does not.
    #[test]
    fn a_hit_off_its_own_solid_is_refused() {
        let primitives = [solid(0, rectangle(20.0, 30.0, 10.0, 10.0), 0.0, 10.0)];
        let index = RayIndex::new(&primitives);
        for (from, to) in [
            ([-19.999_999_9, -20.0, 30.0], [80.0, 80.0, -5.0]),
            ([-19.999_999_9, -20.0, 5.0], [80.0, 80.0, 5.0]),
        ] {
            assert!(primitives[0].surface_hit_t(from, to).is_some());
            assert_eq!(index.first_hit(&primitives, from, to, &mut 0), None);
        }
    }

    /// Rays along walls and through corners, where the exact test's own
    /// tolerances decide.
    #[test]
    fn it_agrees_along_walls_and_through_corners() {
        let primitives: Vec<Obstacle> = (0..36)
            .map(|i| {
                let (x, y) = ((i % 6) as f64 * 10.0, (i / 6) as f64 * 10.0);
                solid(i, rectangle(x, y, 10.0, 10.0), 0.0, 10.0)
            })
            .collect();
        let index = RayIndex::new(&primitives);
        let offsets = [0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-7, -1e-7, 1e-5, -1e-5];
        for line in 0..7 {
            let at = line as f64 * 10.0;
            for dx in offsets {
                for dz in offsets {
                    for (from, to) in [
                        ([at + dx, -20.0, 10.0 + dz], [at + dx, 80.0, 10.0 + dz]),
                        ([-20.0, at + dx, dz], [80.0, at + dx, dz]),
                        ([at + dx, at, 60.0], [at + dx, at, -60.0]),
                        ([-20.0 + dx, -20.0, 30.0 + dz], [80.0, 80.0, -5.0]),
                        ([-20.0, at, 5.0], [dx, at, 5.0 + dz]),
                    ] {
                        let got = index.first_hit(&primitives, from, to, &mut 0);
                        assert_eq!(
                            got,
                            brute_on_solid(&primitives, from, to),
                            "{from:?} -> {to:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn obstacles_met_at_the_same_place_give_the_first() {
        let primitives: Vec<Obstacle> = (0..40)
            .map(|i| solid(i, rectangle(10.0, 10.0, 5.0, 5.0), 0.0, 10.0))
            .collect();
        let index = RayIndex::new(&primitives);
        let hit = index.first_hit(&primitives, [0.0, 12.0, 5.0], [30.0, 12.0, 5.0], &mut 0);
        assert_eq!(hit.map(|h| h.0), Some(0));
    }

    #[test]
    fn vegetation_is_not_a_hit() {
        let mut primitives = vec![solid(0, rectangle(10.0, 10.0, 5.0, 5.0), 0.0, 10.0)];
        primitives[0].blocker_type = BlockerType::Vegetation;
        let index = RayIndex::new(&primitives);
        let hit = index.first_hit(&primitives, [0.0, 12.0, 5.0], [30.0, 12.0, 5.0], &mut 0);
        assert_eq!(hit, None);
    }

    /// The point of the index, held by a count rather than a clock: a
    /// long ray down into a town tests a few solids, not the hundreds in its
    /// bounding box.
    #[test]
    fn a_long_ray_tests_few_obstacles() {
        let mut rng = Rng(42);
        let primitives = town(&mut rng, 600);
        let index = RayIndex::new(&primitives);
        let (rays, mut tested) = (1000, 0);
        for _ in 0..rays {
            let from = [rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0), 150.0];
            let to = [
                from[0] + rng.uniform(-200.0, 200.0),
                from[1] + rng.uniform(-200.0, 200.0),
                -10.0,
            ];
            index.first_hit(&primitives, from, to, &mut tested);
        }
        assert!(tested < 5 * rays, "{tested} exact tests for {rays} rays");
        // Straight down too, where two axes have no travel to cut on.
        let mut tested = 0;
        for _ in 0..rays {
            let from = [rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0), 150.0];
            index.first_hit(&primitives, from, [from[0], from[1], -10.0], &mut tested);
        }
        assert!(tested < 5 * rays, "{tested} exact tests for {rays} rays");
    }
}
