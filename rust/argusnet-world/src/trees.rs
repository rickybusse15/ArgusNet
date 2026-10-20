//! Trees (`tree-v1`, R2 in `docs/WORLD_REALISM_PLAN.md`).
//!
//! A tree is a trunk and a crown. The trunk is a vertical cylinder from the
//! ground to the middle of the crown: solid, something a platform hits and a
//! ray stops at. The crown is an ellipsoid of leaves: soft, something a ray
//! is dimmed by and a platform can push into without being stopped. That is
//! the split the rest of the world keeps too: [`crate::obstacles::BlockerType::is_hard`]
//! stays buildings and walls, and trees answer for themselves here, part by
//! part.
//!
//! Light through a crown falls off as Beer–Lambert says it does through a
//! leaf canopy: `exp(-k · d · L)` over a chord of length `L`, with the
//! tree's leaf density `d` (0 bare to 1 in full leaf) and
//! [`CROWN_EXTINCTION_PER_M`] for `k`. Forest stands
//! ([`crate::obstacles::Obstacle::forest_stand`]) stay the cheap aggregate
//! for large forests; trees are the individual case.
//!
//! [`TreeLayer`] indexes trees on a grid and walks a ray through it cell by
//! cell, skipping cells the ray passes above, so a scanner's long rays over
//! a forest cost what the trees near them cost. [`poisson_disk`] is the
//! seeded, fixed-order sampler procedural placement scatters trees with.
//!
//! This module is the model. A run's bundle carries trees in its world
//! (`argusnet-sim`'s builder scatters them with the `trees` option);
//! sensing, flight and drawing come in later slices.

use serde::{Deserialize, Serialize};

use crate::numpy_rng::NumpyRng;
use crate::obstacles::Bounds2d;
use crate::visibility::dda;

/// The kind tag a tree carries in a world bundle.
pub const TREE_KIND: &str = "tree-v1";

/// How fast light dies in a crown in full leaf, per metre of chord:
/// Beer–Lambert's `G · LAD` for a spherical leaf-angle distribution
/// (`G = 0.5`) and a leaf area density of 0.7 m²/m³, a middling value for a
/// temperate crown. A 6 m chord through a crown of density 0.9 lets through
/// 15 %; two such crowns, 2 %.
pub const CROWN_EXTINCTION_PER_M: f64 = 0.35;

/// The side of a [`TreeLayer`]'s grid cell, metres: about the width of a
/// large crown, so a tree is listed in a few cells.
pub const TREE_CELL_M: f64 = 16.0;

/// The most cells a [`TreeLayer`] or a [`poisson_disk`] grid may hold: at
/// 16 m cells, a square 65 km on a side. Beyond it the trees (or the
/// bounds) are spread too far to index, and are refused rather than
/// allocated.
pub const MAX_GRID_CELLS: usize = 1 << 24;

/// The farthest a tree may stand from the origin, or its ground from the
/// datum, metres: 10,000 km, any local frame on Earth.
pub const MAX_COORDINATE_M: f64 = 1.0e7;
/// The tallest a tree may be, metres (the tallest known is 116 m).
pub const MAX_TREE_HEIGHT_M: f64 = 200.0;
/// The widest a crown's radius may be, metres.
pub const MAX_CROWN_RADIUS_M: f64 = 100.0;
/// The widest a trunk's radius may be, metres.
pub const MAX_TRUNK_RADIUS_M: f64 = 20.0;

/// What kind of tree, which sets its shape when placed procedurally and how
/// the viewer draws it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Species {
    Conifer,
    Broadleaf,
    Palm,
    Shrub,
}

impl Species {
    pub const ALL: [Species; 4] = [
        Species::Conifer,
        Species::Broadleaf,
        Species::Palm,
        Species::Shrub,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conifer => "conifer",
            Self::Broadleaf => "broadleaf",
            Self::Palm => "palm",
            Self::Shrub => "shrub",
        }
    }
}

/// One tree. Read only through [`Tree::validated`], so a tree in a bundle
/// that the model refuses is an error, never a tree that silently blocks
/// nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedTree")]
pub struct Tree {
    pub id: String,
    /// Where the trunk stands.
    pub xy_m: [f64; 2],
    /// Ground level at the trunk, metres above datum.
    pub base_z_m: f64,
    /// Height of the top of the crown above `base_z_m`.
    pub height_m: f64,
    /// Height of the bottom of the crown above `base_z_m`; below `height_m`.
    pub crown_base_m: f64,
    /// The crown's horizontal radius. Its vertical half-axis is half of
    /// `height_m - crown_base_m`.
    pub crown_radius_m: f64,
    /// The trunk's radius. Zero is a crown on no trunk worth hitting.
    pub trunk_radius_m: f64,
    pub species: Species,
    /// How much the crown dims what passes through it, 0 (bare) to 1 (in
    /// full leaf): a deciduous tree's density falls with the season.
    pub leaf_density: f64,
}

/// A [`Tree`] as written, before it is checked.
#[derive(Deserialize)]
struct UncheckedTree {
    id: String,
    xy_m: [f64; 2],
    base_z_m: f64,
    height_m: f64,
    crown_base_m: f64,
    crown_radius_m: f64,
    trunk_radius_m: f64,
    species: Species,
    leaf_density: f64,
}

impl TryFrom<UncheckedTree> for Tree {
    type Error = TreeError;

    fn try_from(tree: UncheckedTree) -> Result<Self, Self::Error> {
        Tree {
            id: tree.id,
            xy_m: tree.xy_m,
            base_z_m: tree.base_z_m,
            height_m: tree.height_m,
            crown_base_m: tree.crown_base_m,
            crown_radius_m: tree.crown_radius_m,
            trunk_radius_m: tree.trunk_radius_m,
            species: tree.species,
            leaf_density: tree.leaf_density,
        }
        .validated()
    }
}

/// Why a [`Tree`] or a [`TreeLayer`] cannot be used.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeError {
    /// A value out of its range: not finite, or (for the named field)
    /// outside what a tree can be.
    BadValue { id: String, field: &'static str },
    /// Two trees with one id.
    DuplicateId { id: String },
    /// The trees are spread over more ground than a grid of
    /// [`MAX_GRID_CELLS`] covers.
    TooSpread { cells: f64 },
}

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadValue { id, field } => write!(f, "{id}: {field} is out of range"),
            Self::DuplicateId { id } => write!(f, "{id}: two trees have this id"),
            Self::TooSpread { cells } => write!(
                f,
                "the trees span {cells:.0} grid cells, more than {MAX_GRID_CELLS}"
            ),
        }
    }
}

impl std::error::Error for TreeError {}

/// Which part of a tree a ray met. A trunk sorts before a crown, for ties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TreePart {
    Trunk,
    Crown,
}

impl Tree {
    /// The tree, checked: every value finite and within what a tree can be
    /// ([`MAX_COORDINATE_M`], [`MAX_TREE_HEIGHT_M`], [`MAX_CROWN_RADIUS_M`],
    /// [`MAX_TRUNK_RADIUS_M`]), the height positive, the crown base at or
    /// above the ground and below the top, the crown radius positive, the
    /// trunk radius not negative and the leaf density in `[0, 1]`.
    pub fn validated(self) -> Result<Self, TreeError> {
        let bad = |field| TreeError::BadValue {
            id: self.id.clone(),
            field,
        };
        // Within what a tree can be, so no value derived from them (a top,
        // a footprint, a square in an intersection) can overflow.
        let within = |v: f64, limit: f64| v.is_finite() && v.abs() <= limit;
        if !(self.xy_m.iter().all(|v| within(*v, MAX_COORDINATE_M))
            && within(self.base_z_m, MAX_COORDINATE_M))
        {
            return Err(bad("position"));
        }
        if !(within(self.height_m, MAX_TREE_HEIGHT_M) && self.height_m > 0.0) {
            return Err(bad("height_m"));
        }
        if !(self.crown_base_m.is_finite()
            && self.crown_base_m >= 0.0
            && self.crown_base_m < self.height_m)
        {
            return Err(bad("crown_base_m"));
        }
        if !(within(self.crown_radius_m, MAX_CROWN_RADIUS_M) && self.crown_radius_m > 0.0) {
            return Err(bad("crown_radius_m"));
        }
        if !(within(self.trunk_radius_m, MAX_TRUNK_RADIUS_M) && self.trunk_radius_m >= 0.0) {
            return Err(bad("trunk_radius_m"));
        }
        if !(self.leaf_density.is_finite() && (0.0..=1.0).contains(&self.leaf_density)) {
            return Err(bad("leaf_density"));
        }
        Ok(self)
    }

    /// The top of the crown, metres above datum.
    pub fn top_z_m(&self) -> f64 {
        self.base_z_m + self.height_m
    }

    /// The crown's centre: over the trunk, halfway up the crown.
    pub fn crown_centre_m(&self) -> [f64; 3] {
        [
            self.xy_m[0],
            self.xy_m[1],
            // The crown base plus its half-height, not the mean of base and
            // top: each step stays between the ground and the top, so no
            // intermediate overflows when those are finite.
            self.base_z_m + self.crown_base_m + self.crown_half_height_m(),
        ]
    }

    /// The crown's vertical half-axis, metres.
    pub fn crown_half_height_m(&self) -> f64 {
        0.5 * (self.height_m - self.crown_base_m)
    }

    /// Where the trunk ends, metres above datum: the middle of the crown.
    /// Above that the crown is branches and leaves, not one solid stem.
    pub fn trunk_top_z_m(&self) -> f64 {
        self.crown_centre_m()[2]
    }

    /// The ground the tree covers, from above: its crown, or its trunk if
    /// that is wider.
    pub fn bounds_xy_m(&self) -> Bounds2d {
        let reach = self.crown_radius_m.max(self.trunk_radius_m);
        Bounds2d::new(
            self.xy_m[0] - reach,
            self.xy_m[0] + reach,
            self.xy_m[1] - reach,
            self.xy_m[1] + reach,
        )
    }

    /// Whether `point` is inside the crown (on its surface counts).
    pub fn crown_contains(&self, point: [f64; 3]) -> bool {
        let c = self.crown_centre_m();
        let (r, h) = (self.crown_radius_m, self.crown_half_height_m());
        let q = [
            (point[0] - c[0]) / r,
            (point[1] - c[1]) / r,
            (point[2] - c[2]) / h,
        ];
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= 1.0
    }

    /// Whether `point` is inside the trunk (on its surface counts).
    pub fn trunk_contains(&self, point: [f64; 3]) -> bool {
        self.trunk_radius_m > 0.0
            && point[2] >= self.base_z_m
            && point[2] <= self.trunk_top_z_m()
            && (point[0] - self.xy_m[0]).hypot(point[1] - self.xy_m[1]) <= self.trunk_radius_m
    }

    /// The part of the segment from `a` to `b` inside the crown, as fractions
    /// of the segment `(t0, t1)` with `t0 < t1`, or `None` when it misses (or
    /// only grazes the surface).
    pub fn crown_interval(&self, a: [f64; 3], b: [f64; 3]) -> Option<(f64, f64)> {
        let c = self.crown_centre_m();
        let (r, h) = (self.crown_radius_m, self.crown_half_height_m());
        // In the crown's own units it is the unit sphere.
        let p = [(a[0] - c[0]) / r, (a[1] - c[1]) / r, (a[2] - c[2]) / h];
        let d = [(b[0] - a[0]) / r, (b[1] - a[1]) / r, (b[2] - a[2]) / h];
        let qa = dot(d, d);
        let qb = 2.0 * dot(p, d);
        let qc = dot(p, p) - 1.0;
        if qa <= 1e-18 {
            // A point: inside for all of it or none.
            return (qc < 0.0).then_some((0.0, 1.0));
        }
        let disc = qb * qb - 4.0 * qa * qc;
        if disc <= 0.0 {
            return None;
        }
        let root = disc.sqrt();
        let t0 = ((-qb - root) / (2.0 * qa)).max(0.0);
        let t1 = ((-qb + root) / (2.0 * qa)).min(1.0);
        (t0 < t1).then_some((t0, t1))
    }

    /// The length of the segment from `a` to `b` inside the crown, metres.
    pub fn crown_chord_m(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        self.crown_interval(a, b)
            .map_or(0.0, |(t0, t1)| (t1 - t0) * length(sub(b, a)))
    }

    /// The crown's optical depth along the segment: `k · d · L`, so the light
    /// through is `exp(-depth)`.
    pub fn crown_optical_depth(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        CROWN_EXTINCTION_PER_M * self.leaf_density * self.crown_chord_m(a, b)
    }

    /// The fraction of the segment from `a` to `b` at which it first touches
    /// the trunk, or `None` when it misses. A segment that starts inside the
    /// trunk touches it at 0.
    pub fn trunk_hit_t(&self, a: [f64; 3], b: [f64; 3]) -> Option<f64> {
        if self.trunk_radius_m <= 0.0 {
            return None;
        }
        let (z0, z1) = (self.base_z_m, self.trunk_top_z_m());
        // The z slab.
        let dz = b[2] - a[2];
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        if dz.abs() < 1e-12 {
            if a[2] < z0 || a[2] > z1 {
                return None;
            }
        } else {
            let (ta, tb) = ((z0 - a[2]) / dz, (z1 - a[2]) / dz);
            lo = lo.max(ta.min(tb));
            hi = hi.min(ta.max(tb));
        }
        // The circle.
        let p = [a[0] - self.xy_m[0], a[1] - self.xy_m[1]];
        let d = [b[0] - a[0], b[1] - a[1]];
        let qa = d[0] * d[0] + d[1] * d[1];
        let qc = p[0] * p[0] + p[1] * p[1] - self.trunk_radius_m * self.trunk_radius_m;
        if qa <= 1e-18 {
            if qc > 0.0 {
                return None;
            }
        } else {
            let qb = 2.0 * (p[0] * d[0] + p[1] * d[1]);
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                return None;
            }
            let root = disc.sqrt();
            lo = lo.max((-qb - root) / (2.0 * qa));
            hi = hi.min((-qb + root) / (2.0 * qa));
        }
        (lo <= hi).then_some(lo)
    }
}

/// What a ray first met among the trees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeHit<'a> {
    /// Fraction of the ray at which it met the tree.
    pub t: f64,
    pub part: TreePart,
    pub tree: &'a Tree,
}

/// Trees, indexed on a grid for the questions a run asks of them: what is
/// at a point, what a ray meets first, and how much light gets along it.
#[derive(Debug, Clone)]
pub struct TreeLayer {
    /// In id order, so every answer is the same whatever order the trees
    /// were given in.
    trees: Vec<Tree>,
    /// The grid's extent: every tree's bounds, rounded out to whole cells.
    bounds: Bounds2d,
    columns: usize,
    rows: usize,
    /// Per cell, row-major: the trees whose bounds touch it, in index
    /// order.
    cells: Vec<Vec<u32>>,
    /// Per cell: the highest tree top listed in it, so a ray above it skips
    /// the cell.
    cell_top_z_m: Vec<f64>,
}

impl TreeLayer {
    /// Index `trees`, each checked again. Two trees with one id are refused.
    pub fn new(trees: Vec<Tree>) -> Result<Self, TreeError> {
        let mut trees = trees
            .into_iter()
            .map(Tree::validated)
            .collect::<Result<Vec<_>, _>>()?;
        trees.sort_by(|a, b| a.id.cmp(&b.id));
        if let Some(pair) = trees.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(TreeError::DuplicateId {
                id: pair[0].id.clone(),
            });
        }
        let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for tree in &trees {
            let b = tree.bounds_xy_m();
            x_min = x_min.min(b.x_min_m);
            x_max = x_max.max(b.x_max_m);
            y_min = y_min.min(b.y_min_m);
            y_max = y_max.max(b.y_max_m);
        }
        if trees.is_empty() {
            (x_min, x_max, y_min, y_max) = (0.0, TREE_CELL_M, 0.0, TREE_CELL_M);
        }
        // Counted in floats first: trees each finite can still be spread
        // too far apart to grid.
        let (columns, rows) = (
            ((x_max - x_min) / TREE_CELL_M).floor() + 1.0,
            ((y_max - y_min) / TREE_CELL_M).floor() + 1.0,
        );
        if columns * rows > MAX_GRID_CELLS as f64 {
            return Err(TreeError::TooSpread {
                cells: columns * rows,
            });
        }
        let (columns, rows) = (columns as usize, rows as usize);
        let bounds = Bounds2d::new(
            x_min,
            x_min + columns as f64 * TREE_CELL_M,
            y_min,
            y_min + rows as f64 * TREE_CELL_M,
        );
        let mut layer = Self {
            trees,
            bounds,
            columns,
            rows,
            cells: vec![Vec::new(); columns * rows],
            cell_top_z_m: vec![f64::NEG_INFINITY; columns * rows],
        };
        for index in 0..layer.trees.len() {
            let tree = &layer.trees[index];
            let (b, top) = (tree.bounds_xy_m(), tree.top_z_m());
            let (c0, r0) = layer.cell_of(b.x_min_m, b.y_min_m);
            let (c1, r1) = layer.cell_of(b.x_max_m, b.y_max_m);
            for row in r0..=r1 {
                for column in c0..=c1 {
                    let cell = row * layer.columns + column;
                    layer.cells[cell].push(index as u32);
                    layer.cell_top_z_m[cell] = layer.cell_top_z_m[cell].max(top);
                }
            }
        }
        Ok(layer)
    }

    /// The trees, in id order.
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// The cell holding `(x, y)`, clamped to the grid.
    fn cell_of(&self, x_m: f64, y_m: f64) -> (usize, usize) {
        let column = ((x_m - self.bounds.x_min_m) / TREE_CELL_M).floor();
        let row = ((y_m - self.bounds.y_min_m) / TREE_CELL_M).floor();
        (
            (column.max(0.0) as usize).min(self.columns - 1),
            (row.max(0.0) as usize).min(self.rows - 1),
        )
    }

    /// The trees whose bounds touch `bounds`, in id order.
    pub fn near(&self, bounds: Bounds2d) -> Vec<&Tree> {
        if self.trees.is_empty()
            || bounds.x_max_m < self.bounds.x_min_m
            || bounds.x_min_m > self.bounds.x_max_m
            || bounds.y_max_m < self.bounds.y_min_m
            || bounds.y_min_m > self.bounds.y_max_m
        {
            return Vec::new();
        }
        let (c0, r0) = self.cell_of(bounds.x_min_m, bounds.y_min_m);
        let (c1, r1) = self.cell_of(bounds.x_max_m, bounds.y_max_m);
        let mut indices: Vec<u32> = (r0..=r1)
            .flat_map(|row| (c0..=c1).map(move |column| row * self.columns + column))
            .flat_map(|cell| self.cells[cell].iter().copied())
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
            .into_iter()
            .map(|i| &self.trees[i as usize])
            .filter(|tree| {
                let b = tree.bounds_xy_m();
                b.x_max_m >= bounds.x_min_m
                    && b.x_min_m <= bounds.x_max_m
                    && b.y_max_m >= bounds.y_min_m
                    && b.y_min_m <= bounds.y_max_m
            })
            .collect()
    }

    /// The tree whose trunk holds `point`, if any (the first in id order).
    pub fn trunk_at(&self, point: [f64; 3]) -> Option<&Tree> {
        self.near(Bounds2d::new(point[0], point[0], point[1], point[1]))
            .into_iter()
            .find(|tree| tree.trunk_contains(point))
    }

    /// The tree whose crown holds `point`, if any (the first in id order).
    pub fn crown_at(&self, point: [f64; 3]) -> Option<&Tree> {
        self.near(Bounds2d::new(point[0], point[0], point[1], point[1]))
            .into_iter()
            .find(|tree| tree.crown_contains(point))
    }

    /// The cells the segment from `a` to `b` passes through low enough to
    /// meet a tree in, in order along it, each with the fractions of the
    /// segment inside it.
    fn cells_along(&self, a: [f64; 3], b: [f64; 3]) -> Vec<(usize, f64, f64)> {
        if self.trees.is_empty() {
            return Vec::new();
        }
        dda([a[0], a[1]], [b[0], b[1]], self.bounds, TREE_CELL_M)
            .into_iter()
            .filter_map(|(column, row, t0, t1)| {
                // `dda` counts cells from the bounds' width, which rounding
                // can make one more than the grid's.
                let (column, row) = (usize::try_from(column).ok()?, usize::try_from(row).ok()?);
                if column >= self.columns || row >= self.rows {
                    return None;
                }
                let cell = row * self.columns + column;
                let low = (a[2] + (b[2] - a[2]) * t0).min(a[2] + (b[2] - a[2]) * t1);
                (!self.cells[cell].is_empty() && low <= self.cell_top_z_m[cell])
                    .then_some((cell, t0, t1))
            })
            .collect()
    }

    /// The trees listed in the cells the segment passes low through, in
    /// index (id) order.
    fn candidates(&self, a: [f64; 3], b: [f64; 3]) -> Vec<u32> {
        let mut indices: Vec<u32> = self
            .cells_along(a, b)
            .into_iter()
            .flat_map(|(cell, _, _)| self.cells[cell].iter().copied())
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// The crowns' total optical depth along the segment from `a` to `b`,
    /// summed in id order.
    pub fn optical_depth_along(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        self.candidates(a, b)
            .into_iter()
            .map(|i| self.trees[i as usize].crown_optical_depth(a, b))
            .sum()
    }

    /// The fraction of light that gets from `a` to `b` through the crowns:
    /// 1 when the segment crosses none. Trunks are not counted here; they
    /// stop a ray outright ([`TreeLayer::first_hit`]).
    pub fn transmittance_along(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        (-self.optical_depth_along(a, b)).exp()
    }

    /// The first trunk or crown the segment from `a` to `b` meets, nearest
    /// first (a trunk before a crown at the same point, then id order).
    /// Walks the grid in order along the segment and stops at the first cell
    /// past which nothing nearer can lie.
    pub fn first_hit(&self, a: [f64; 3], b: [f64; 3]) -> Option<TreeHit<'_>> {
        self.first_hit_where(a, b, |_| true)
    }

    /// The first trunk the segment from `a` to `b` meets, crowns passed
    /// through.
    pub fn first_trunk_hit(&self, a: [f64; 3], b: [f64; 3]) -> Option<TreeHit<'_>> {
        self.first_hit_where(a, b, |part| part == TreePart::Trunk)
    }

    fn first_hit_where(
        &self,
        a: [f64; 3],
        b: [f64; 3],
        wanted: impl Fn(TreePart) -> bool,
    ) -> Option<TreeHit<'_>> {
        let mut best: Option<TreeHit<'_>> = None;
        let mut seen: Vec<u32> = Vec::new();
        for (cell, _, t1) in self.cells_along(a, b) {
            for &index in &self.cells[cell] {
                if seen.contains(&index) {
                    continue;
                }
                seen.push(index);
                let tree = &self.trees[index as usize];
                let hits = [
                    (TreePart::Trunk, tree.trunk_hit_t(a, b)),
                    (TreePart::Crown, tree.crown_interval(a, b).map(|(t0, _)| t0)),
                ];
                for (part, t) in hits {
                    let Some(t) = t.filter(|_| wanted(part)) else {
                        continue;
                    };
                    let nearer = best.as_ref().is_none_or(|hit| {
                        t < hit.t || (t == hit.t && (part, &tree.id) < (hit.part, &hit.tree.id))
                    });
                    if nearer {
                        best = Some(TreeHit { t, part, tree });
                    }
                }
            }
            // Every tree a hit before `t1` could belong to is listed in a
            // cell already walked: the hit point is in the tree's bounds.
            // A hit exactly at `t1` is not settled yet: a tree whose bounds
            // begin on that boundary is listed only in the next cell, and
            // may tie it.
            if best.as_ref().is_some_and(|hit| hit.t < t1) {
                break;
            }
        }
        best
    }
}

/// Points over `bounds` no two closer than `spacing_m`, seeded: a
/// Poisson-disk sample, drawn in a fixed order. The plane is cut into cells
/// of side `spacing_m / √2` (so a cell holds at most one point), visited row
/// by row from the low corner; each cell draws `attempts` candidates inside
/// itself and keeps the first that is far enough from every point already
/// kept. Every cell draws all its candidates whether or not one is kept, so
/// the stream of draws is fixed by the seed and the grid alone. The sample
/// does not depend on what the ground is; a caller that wants trees only
/// over forest thins it afterwards, which moves no point it keeps.
pub fn poisson_disk(bounds: Bounds2d, spacing_m: f64, attempts: usize, seed: u32) -> Vec<[f64; 2]> {
    let width = bounds.x_max_m - bounds.x_min_m;
    let height = bounds.y_max_m - bounds.y_min_m;
    if !(spacing_m.is_finite()
        && spacing_m > 0.0
        && width.is_finite()
        && height.is_finite()
        && width >= 0.0
        && height >= 0.0)
    {
        return Vec::new();
    }
    let cell = spacing_m / std::f64::consts::SQRT_2;
    let (columns, rows) = (
        (width / cell).ceil().max(1.0),
        (height / cell).ceil().max(1.0),
    );
    if columns * rows > MAX_GRID_CELLS as f64 {
        return Vec::new();
    }
    let (columns, rows) = (columns as usize, rows as usize);
    let mut grid: Vec<Option<[f64; 2]>> = vec![None; columns * rows];
    let mut rng = NumpyRng::new(seed);
    let mut points = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let x0 = bounds.x_min_m + column as f64 * cell;
            let y0 = bounds.y_min_m + row as f64 * cell;
            let mut kept: Option<[f64; 2]> = None;
            for _ in 0..attempts {
                // The last row and column may be cut short by the bounds:
                // draw over the part inside, not over the whole cell and
                // clamped (which would line points up on the edge).
                let candidate = [
                    x0 + rng.random() * cell.min(bounds.x_max_m - x0),
                    y0 + rng.random() * cell.min(bounds.y_max_m - y0),
                ];
                if kept.is_some() {
                    continue;
                }
                let clear = (row.saturating_sub(2)..(row + 3).min(rows)).all(|r| {
                    (column.saturating_sub(2)..(column + 3).min(columns)).all(|c| {
                        grid[r * columns + c].is_none_or(|p| {
                            (p[0] - candidate[0]).hypot(p[1] - candidate[1]) >= spacing_m
                        })
                    })
                });
                if clear {
                    kept = Some(candidate);
                }
            }
            if let Some(point) = kept {
                grid[row * columns + column] = Some(point);
                points.push(point);
            }
        }
    }
    points
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 20 m broadleaf at the origin on flat ground at 100 m: crown from
    /// 8 m to 20 m (centre 114 m above datum, half-height 6 m), radius 5 m,
    /// a 0.3 m trunk, leaf density 0.8.
    fn tree() -> Tree {
        Tree {
            id: "tree-a".into(),
            xy_m: [0.0, 0.0],
            base_z_m: 100.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 5.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: 0.8,
        }
        .validated()
        .expect("valid")
    }

    fn at(id: &str, x: f64, y: f64) -> Tree {
        Tree {
            id: id.into(),
            xy_m: [x, y],
            ..tree()
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn the_crown_is_an_ellipsoid_halfway_up_its_span() {
        let tree = tree();
        assert_eq!(tree.crown_centre_m(), [0.0, 0.0, 114.0]);
        assert!(close(tree.crown_half_height_m(), 6.0));
        assert!(close(tree.top_z_m(), 120.0));
        assert!(close(tree.trunk_top_z_m(), 114.0));
        // On the surface counts; just past it does not.
        assert!(tree.crown_contains([5.0, 0.0, 114.0]));
        assert!(!tree.crown_contains([5.01, 0.0, 114.0]));
        assert!(tree.crown_contains([0.0, 0.0, 120.0]));
        assert!(!tree.crown_contains([0.0, 0.0, 120.01]));
        assert!(tree.crown_contains([0.0, 0.0, 108.0]));
        assert!(!tree.crown_contains([0.0, 0.0, 107.99]));
        // Off-axis: x/5 = 0.6, z/6 = 0.8 is on the surface.
        assert!(tree.crown_contains([3.0, 0.0, 114.0 + 4.8]));
        assert!(!tree.crown_contains([3.0, 0.0, 114.0 + 4.9]));
    }

    #[test]
    fn the_trunk_runs_from_the_ground_to_the_crowns_middle() {
        let tree = tree();
        assert!(tree.trunk_contains([0.0, 0.0, 100.0]));
        assert!(tree.trunk_contains([0.3, 0.0, 114.0]));
        assert!(!tree.trunk_contains([0.31, 0.0, 110.0]));
        assert!(!tree.trunk_contains([0.0, 0.0, 99.99]));
        assert!(!tree.trunk_contains([0.0, 0.0, 114.01]));
        let bare = Tree {
            trunk_radius_m: 0.0,
            ..tree
        };
        assert!(!bare.trunk_contains([0.0, 0.0, 105.0]));
        assert_eq!(
            bare.trunk_hit_t([-5.0, 0.0, 105.0], [5.0, 0.0, 105.0]),
            None
        );
    }

    #[test]
    fn a_ray_through_the_crown_reports_its_chord() {
        let tree = tree();
        // Level through the centre: the full 10 m diameter.
        let (a, b) = ([-10.0, 0.0, 114.0], [10.0, 0.0, 114.0]);
        let (t0, t1) = tree.crown_interval(a, b).expect("crosses");
        assert!(close(t0, 0.25) && close(t1, 0.75), "{t0} {t1}");
        assert!(close(tree.crown_chord_m(a, b), 10.0));
        // Straight down the axis: the 12 m height.
        assert!(close(
            tree.crown_chord_m([0.0, 0.0, 200.0], [0.0, 0.0, 100.0]),
            12.0
        ));
        // A chord at x = 3 m, level at the centre: 2·√(25 − 9) = 8 m.
        assert!(close(
            tree.crown_chord_m([3.0, -10.0, 114.0], [3.0, 10.0, 114.0]),
            8.0
        ));
        // A segment that stops inside counts only its inside part.
        assert!(close(
            tree.crown_chord_m([-10.0, 0.0, 114.0], [0.0, 0.0, 114.0]),
            5.0
        ));
        // Missing above, or tangent, is nothing.
        assert_eq!(
            tree.crown_interval([-10.0, 0.0, 121.0], [10.0, 0.0, 121.0]),
            None
        );
        assert_eq!(
            tree.crown_interval([-10.0, 5.0, 114.0], [10.0, 5.0, 114.0]),
            None
        );
        assert!(close(
            tree.crown_chord_m([-10.0, 6.0, 114.0], [10.0, 6.0, 114.0]),
            0.0
        ));
    }

    #[test]
    fn light_through_a_crown_falls_off_by_beer_lambert() {
        let tree = tree();
        let (a, b) = ([-10.0, 0.0, 114.0], [10.0, 0.0, 114.0]);
        assert!(close(
            tree.crown_optical_depth(a, b),
            CROWN_EXTINCTION_PER_M * 0.8 * 10.0
        ));
        let bare = Tree {
            leaf_density: 0.0,
            ..tree.clone()
        };
        assert_eq!(bare.crown_optical_depth(a, b), 0.0);

        // Two crowns in a row dim twice as much; one off the ray, not at all.
        let layer = TreeLayer::new(vec![
            at("a", 0.0, 0.0),
            at("b", 20.0, 0.0),
            at("c", 0.0, 30.0),
        ])
        .expect("layer");
        let depth = CROWN_EXTINCTION_PER_M * 0.8 * 10.0;
        let ray = ([-10.0, 0.0, 114.0], [40.0, 0.0, 114.0]);
        assert!(close(layer.optical_depth_along(ray.0, ray.1), 2.0 * depth));
        assert!(close(
            layer.transmittance_along(ray.0, ray.1),
            (-2.0 * depth).exp()
        ));
        // Over every crown: all the light.
        assert_eq!(
            layer.transmittance_along([-10.0, 0.0, 125.0], [40.0, 0.0, 125.0]),
            1.0
        );
    }

    #[test]
    fn a_ray_meets_the_trunk_at_its_surface() {
        let tree = tree();
        let t = tree
            .trunk_hit_t([-10.0, 0.0, 105.0], [10.0, 0.0, 105.0])
            .expect("hits");
        assert!(close(t, (10.0 - 0.3) / 20.0), "{t}");
        // Above the trunk's top, or beside it, misses.
        assert_eq!(
            tree.trunk_hit_t([-10.0, 0.0, 115.0], [10.0, 0.0, 115.0]),
            None
        );
        assert_eq!(
            tree.trunk_hit_t([-10.0, 0.5, 105.0], [10.0, 0.5, 105.0]),
            None
        );
        // Coming down onto the trunk's top from above lands on the top.
        let t = tree
            .trunk_hit_t([0.0, 0.0, 124.0], [0.0, 0.0, 104.0])
            .expect("hits");
        assert!(close(t, 0.5), "{t}");
        // Starting inside touches at once.
        assert_eq!(
            tree.trunk_hit_t([0.0, 0.0, 105.0], [10.0, 0.0, 105.0]),
            Some(0.0)
        );
        // Slanting down past the trunk's foot: the slab cuts it off.
        assert_eq!(
            tree.trunk_hit_t([-10.0, 0.0, 99.0], [10.0, 0.0, 98.0]),
            None
        );
    }

    #[test]
    fn the_first_hit_is_the_nearest_part_of_the_nearest_tree() {
        let layer =
            TreeLayer::new(vec![at("far", 40.0, 0.0), at("near", 10.0, 0.0)]).expect("layer");
        // Level at trunk height under the crowns: the near trunk.
        let hit = layer
            .first_hit([-10.0, 0.0, 105.0], [60.0, 0.0, 105.0])
            .expect("hits");
        assert_eq!((hit.tree.id.as_str(), hit.part), ("near", TreePart::Trunk));
        assert!(close(hit.t, (20.0 - 0.3) / 70.0));
        // Through the crowns: the near crown's surface.
        let hit = layer
            .first_hit([-10.0, 0.0, 114.0], [60.0, 0.0, 114.0])
            .expect("hits");
        assert_eq!((hit.tree.id.as_str(), hit.part), ("near", TreePart::Crown));
        assert!(close(hit.t, 15.0 / 70.0));
        // The same ray backwards meets the far tree first.
        let hit = layer
            .first_hit([60.0, 0.0, 114.0], [-10.0, 0.0, 114.0])
            .expect("hits");
        assert_eq!(hit.tree.id, "far");
        // Trunks only: the crowns are passed through, and at crown height
        // the trunk is inside the crown, at the centre.
        let hit = layer
            .first_trunk_hit([-10.0, 0.0, 114.0], [60.0, 0.0, 114.0])
            .expect("hits");
        assert_eq!((hit.tree.id.as_str(), hit.part), ("near", TreePart::Trunk));
        // Nothing above them.
        assert!(layer
            .first_hit([-10.0, 0.0, 130.0], [60.0, 0.0, 130.0])
            .is_none());
    }

    /// The walk stops early, so check it against a brute-force scan on rays
    /// in every direction over a small forest.
    #[test]
    fn the_grid_walk_agrees_with_testing_every_tree() {
        let mut trees = Vec::new();
        let mut rng = NumpyRng::new(11);
        for k in 0..60 {
            trees.push(Tree {
                id: format!("t{k:02}"),
                xy_m: [rng.uniform(-80.0, 80.0), rng.uniform(-80.0, 80.0)],
                base_z_m: rng.uniform(95.0, 105.0),
                height_m: rng.uniform(4.0, 25.0),
                crown_base_m: 2.0,
                crown_radius_m: rng.uniform(1.0, 7.0),
                trunk_radius_m: rng.uniform(0.0, 0.6),
                species: Species::Conifer,
                leaf_density: rng.uniform(0.0, 1.0),
            });
        }
        let layer = TreeLayer::new(trees.clone()).expect("layer");
        for _ in 0..400 {
            let a = [
                rng.uniform(-120.0, 120.0),
                rng.uniform(-120.0, 120.0),
                rng.uniform(95.0, 160.0),
            ];
            let b = [
                rng.uniform(-120.0, 120.0),
                rng.uniform(-120.0, 120.0),
                rng.uniform(90.0, 130.0),
            ];
            let brute = trees
                .iter()
                .flat_map(|tree| {
                    [
                        tree.trunk_hit_t(a, b).map(|t| (t, 0, tree.id.clone())),
                        tree.crown_interval(a, b)
                            .map(|(t, _)| (t, 1, tree.id.clone())),
                    ]
                })
                .flatten()
                .min_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
            let walked = layer.first_hit(a, b).map(|hit| {
                (
                    hit.t,
                    if hit.part == TreePart::Trunk { 0 } else { 1 },
                    hit.tree.id.clone(),
                )
            });
            assert_eq!(walked, brute, "{a:?} -> {b:?}");
            let depth: f64 = trees
                .iter()
                .map(|tree| tree.crown_optical_depth(a, b))
                .sum();
            assert!((layer.optical_depth_along(a, b) - depth).abs() < 1e-9);
        }
        // Straight down onto every tree's axis finds that tree.
        for tree in &trees {
            let (x, y) = (tree.xy_m[0], tree.xy_m[1]);
            let hit = layer
                .first_hit([x, y, 200.0], [x, y, 50.0])
                .expect("a nadir ray down a tree's axis meets a tree");
            assert!(hit.t <= (200.0 - tree.top_z_m()) / 150.0 + 1e-12);
        }
    }

    #[test]
    fn points_find_the_trunk_and_crown_they_are_in() {
        let layer = TreeLayer::new(vec![at("a", 0.0, 0.0), at("b", 30.0, 0.0)]).expect("layer");
        assert_eq!(
            layer.trunk_at([30.0, 0.1, 103.0]).map(|t| t.id.as_str()),
            Some("b")
        );
        assert_eq!(layer.trunk_at([30.0, 1.0, 103.0]), None);
        assert_eq!(
            layer.crown_at([2.0, 2.0, 115.0]).map(|t| t.id.as_str()),
            Some("a")
        );
        assert_eq!(layer.crown_at([2.0, 2.0, 125.0]), None);
        assert_eq!(layer.crown_at([500.0, 0.0, 114.0]), None);
        assert_eq!(layer.near(Bounds2d::new(24.0, 26.0, -1.0, 1.0)).len(), 1);
        assert!(layer
            .near(Bounds2d::new(-900.0, -800.0, 0.0, 1.0))
            .is_empty());
    }

    #[test]
    fn an_empty_layer_answers_nothing() {
        let layer = TreeLayer::new(Vec::new()).expect("layer");
        assert!(layer.is_empty());
        assert_eq!(layer.transmittance_along([0.0; 3], [100.0, 0.0, 0.0]), 1.0);
        assert!(layer.first_hit([0.0; 3], [100.0, 0.0, 0.0]).is_none());
        assert!(layer.crown_at([0.0; 3]).is_none());
    }

    #[test]
    fn the_layer_is_the_same_whatever_order_the_trees_came_in() {
        let trees = vec![at("c", 40.0, 0.0), at("a", 0.0, 0.0), at("b", 20.0, 0.0)];
        let mut reversed = trees.clone();
        reversed.reverse();
        let (one, two) = (
            TreeLayer::new(trees).expect("layer"),
            TreeLayer::new(reversed).expect("layer"),
        );
        assert_eq!(one.trees(), two.trees());
        assert_eq!(one.trees()[0].id, "a");
        // Each tree fine, but too far apart to grid: refused, not allocated.
        let spread = TreeLayer::new(vec![at("a", -1e7, 0.0), at("b", 1e7, 1e7)]);
        assert!(
            matches!(spread, Err(TreeError::TooSpread { .. })),
            "{spread:?}"
        );
        let duplicate = TreeLayer::new(vec![at("a", 0.0, 0.0), at("a", 9.0, 0.0)]);
        assert_eq!(
            duplicate.unwrap_err(),
            TreeError::DuplicateId { id: "a".into() }
        );
    }

    #[test]
    fn a_bad_tree_is_refused_with_its_reason() {
        let field = |tree: Tree| match tree.validated() {
            Err(TreeError::BadValue { field, .. }) => field,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            field(Tree {
                xy_m: [f64::NAN, 0.0],
                ..tree()
            }),
            "position"
        );
        assert_eq!(
            field(Tree {
                base_z_m: f64::INFINITY,
                ..tree()
            }),
            "position"
        );
        assert_eq!(
            field(Tree {
                height_m: 0.0,
                ..tree()
            }),
            "height_m"
        );
        assert_eq!(
            field(Tree {
                crown_base_m: 20.0,
                ..tree()
            }),
            "crown_base_m"
        );
        assert_eq!(
            field(Tree {
                crown_base_m: -1.0,
                ..tree()
            }),
            "crown_base_m"
        );
        assert_eq!(
            field(Tree {
                crown_radius_m: 0.0,
                ..tree()
            }),
            "crown_radius_m"
        );
        assert_eq!(
            field(Tree {
                trunk_radius_m: -0.1,
                ..tree()
            }),
            "trunk_radius_m"
        );
        assert_eq!(
            field(Tree {
                leaf_density: 1.01,
                ..tree()
            }),
            "leaf_density"
        );
        assert_eq!(
            field(Tree {
                leaf_density: -0.01,
                ..tree()
            }),
            "leaf_density"
        );
        // Finite but beyond what a tree can be: refused, so nothing derived
        // from them (a top, a footprint, a square) can overflow.
        assert_eq!(
            field(Tree {
                xy_m: [1.0e7 + 1.0, 0.0],
                ..tree()
            }),
            "position"
        );
        assert_eq!(
            field(Tree {
                base_z_m: -9e307,
                ..tree()
            }),
            "position"
        );
        assert_eq!(
            field(Tree {
                height_m: 201.0,
                ..tree()
            }),
            "height_m"
        );
        assert_eq!(
            field(Tree {
                crown_radius_m: 101.0,
                ..tree()
            }),
            "crown_radius_m"
        );
        assert_eq!(
            field(Tree {
                trunk_radius_m: 1e200,
                ..tree()
            }),
            "trunk_radius_m"
        );
        // At every limit at once, the geometry stays finite and right.
        let largest = Tree {
            xy_m: [1.0e7, -1.0e7],
            base_z_m: -1.0e7,
            height_m: 200.0,
            crown_base_m: 199.0,
            crown_radius_m: 100.0,
            trunk_radius_m: 20.0,
            ..tree()
        }
        .validated()
        .expect("at the limits");
        assert!(largest.crown_centre_m()[2].is_finite());
        let outside = [1.0e7 + 25.0, -1.0e7, -1.0e7 + 50.0];
        assert!(!largest.trunk_contains(outside));
        assert_eq!(
            largest.trunk_hit_t(outside, [outside[0], outside[1], -1.0e7 + 60.0]),
            None
        );
        // The edges of each range are allowed.
        assert!(Tree {
            crown_base_m: 0.0,
            leaf_density: 1.0,
            trunk_radius_m: 0.0,
            ..tree()
        }
        .validated()
        .is_ok());
    }

    #[test]
    fn a_tree_is_checked_as_it_is_read_and_round_trips() {
        let text = serde_json::to_string(&tree()).expect("json");
        assert!(text.contains("\"species\":\"broadleaf\""), "{text}");
        let back: Tree = serde_json::from_str(&text).expect("reads");
        assert_eq!(back, tree());
        let bad = text.replace("\"leaf_density\":0.8", "\"leaf_density\":2.0");
        let error = serde_json::from_str::<Tree>(&bad).unwrap_err().to_string();
        assert!(error.contains("leaf_density"), "{error}");
        let unknown = text.replace("broadleaf", "cactus");
        assert!(serde_json::from_str::<Tree>(&unknown).is_err());
    }

    #[test]
    fn the_sampler_keeps_its_spacing_and_fills_the_plane() {
        let bounds = Bounds2d::new(0.0, 200.0, 0.0, 150.0);
        let points = poisson_disk(bounds, 8.0, 12, 7);
        for (i, p) in points.iter().enumerate() {
            assert!(bounds.contains_xy(p[0], p[1]));
            for q in &points[i + 1..] {
                assert!((p[0] - q[0]).hypot(p[1] - q[1]) >= 8.0, "{p:?} {q:?}");
            }
        }
        // Every cell tries until one candidate is clear, so the sample is
        // close to maximal: about 0.75 points per r² here.
        let per_r2 = points.len() as f64 * 64.0 / (200.0 * 150.0);
        assert!(
            (0.7..0.8).contains(&per_r2),
            "{} points, {per_r2}",
            points.len()
        );
        // A cut-short last row and column are drawn over, not clamped to
        // the edge: no point lies on it.
        assert!(points
            .iter()
            .all(|p| p[0] < bounds.x_max_m && p[1] < bounds.y_max_m));
        // Seeded: the same seed, the same points; another seed, others.
        assert_eq!(points, poisson_disk(bounds, 8.0, 12, 7));
        assert_ne!(points, poisson_disk(bounds, 8.0, 12, 8));
    }

    #[test]
    fn the_sampler_refuses_nonsense_quietly() {
        let bounds = Bounds2d::new(0.0, 10.0, 0.0, 10.0);
        assert!(poisson_disk(bounds, 0.0, 10, 1).is_empty());
        assert!(poisson_disk(bounds, f64::NAN, 10, 1).is_empty());
        assert!(poisson_disk(Bounds2d::new(10.0, 0.0, 0.0, 10.0), 2.0, 10, 1).is_empty());
        assert!(poisson_disk(bounds, 2.0, 0, 1).is_empty());
        // Unbounded, or too large to grid: nothing, not an allocation.
        assert!(poisson_disk(Bounds2d::new(0.0, f64::INFINITY, 0.0, 10.0), 2.0, 10, 1).is_empty());
        assert!(poisson_disk(Bounds2d::new(f64::NAN, 10.0, 0.0, 10.0), 2.0, 10, 1).is_empty());
        assert!(poisson_disk(Bounds2d::new(0.0, 1e9, 0.0, 1e9), 2.0, 10, 1).is_empty());
        assert!(poisson_disk(Bounds2d::new(-f64::MAX, f64::MAX, 0.0, 1.0), 2.0, 10, 1).is_empty());
    }

    #[test]
    fn a_point_is_inside_the_crown_for_all_of_itself_or_none() {
        let tree = tree();
        let inside = [1.0, 1.0, 114.0];
        assert_eq!(tree.crown_interval(inside, inside), Some((0.0, 1.0)));
        let outside = [9.0, 0.0, 114.0];
        assert_eq!(tree.crown_interval(outside, outside), None);
    }

    #[test]
    fn a_trunk_wider_than_its_crown_sets_the_bounds() {
        let stump = Tree {
            crown_radius_m: 0.5,
            trunk_radius_m: 2.0,
            ..tree()
        };
        assert_eq!(stump.bounds_xy_m(), Bounds2d::new(-2.0, 2.0, -2.0, 2.0));
        let layer = TreeLayer::new(vec![stump]).expect("layer");
        assert!(layer.trunk_at([1.9, 0.0, 105.0]).is_some());
    }

    /// The walk may meet a far tree's crown in an early cell (the crown is
    /// wide and reaches back over the ray's start) before a nearer tree
    /// listed only in a later cell: it must not stop at the first hit it
    /// finds, only at one inside the cell it has walked.
    #[test]
    fn a_wide_crown_seen_early_does_not_hide_a_nearer_tree() {
        let layer = TreeLayer::new(vec![
            // Stretches the grid west, so the cells start at x = -51.
            Tree {
                crown_radius_m: 1.0,
                ..at("anchor", -50.0, 0.0)
            },
            // Listed from x = 11.5; the ray at y = 0 enters it at x ≈ 22.2.
            Tree {
                crown_radius_m: 14.5,
                ..at("wide", 26.0, 14.0)
            },
            // Listed only in the cell from x = 13; met at x = 19.
            Tree {
                crown_radius_m: 1.0,
                ..at("near", 20.0, 0.0)
            },
        ])
        .expect("layer");
        let hit = layer
            .first_hit([0.0, 0.0, 114.0], [100.0, 0.0, 114.0])
            .expect("hits");
        assert_eq!((hit.tree.id.as_str(), hit.part), ("near", TreePart::Crown));
        assert!(close(hit.t, 0.19), "{}", hit.t);
    }

    #[test]
    fn two_trees_met_at_once_resolve_by_id() {
        let layer = TreeLayer::new(vec![at("b", 0.0, 0.0), at("a", 0.0, 0.0)]).expect("layer");
        for (from, to) in [
            ([-10.0, 0.0, 114.0], [10.0, 0.0, 114.0]),
            ([10.0, 0.0, 105.0], [-10.0, 0.0, 105.0]),
        ] {
            assert_eq!(layer.first_hit(from, to).expect("hits").tree.id, "a");
        }
    }

    #[test]
    fn near_lists_each_tree_once() {
        let layer = TreeLayer::new(vec![at("a", 0.0, 0.0), at("b", 30.0, 0.0)]).expect("layer");
        let ids: Vec<_> = layer
            .near(Bounds2d::new(-100.0, 100.0, -100.0, 100.0))
            .into_iter()
            .map(|tree| tree.id.as_str())
            .collect();
        assert_eq!(ids, ["a", "b"]);
    }

    /// The first cell's first candidate is the first point: nothing is near
    /// it yet, and a cell keeps the first candidate that is clear.
    #[test]
    fn a_cell_keeps_its_first_clear_candidate() {
        let bounds = Bounds2d::new(0.0, 100.0, 0.0, 100.0);
        let points = poisson_disk(bounds, 10.0, 8, 5);
        let mut rng = NumpyRng::new(5);
        let cell = 10.0 / std::f64::consts::SQRT_2;
        let first = [rng.random() * cell, rng.random() * cell];
        assert_eq!(points[0], first);
    }
}
