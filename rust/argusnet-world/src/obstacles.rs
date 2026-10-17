//! Obstacle primitives: the geometry of things that block sight and movement.
//!
//! Ported from `src/argusnet/world/obstacles.py` as the first of the three
//! layers this crate absorbs on the way to being the single world authority
//! (terrain, obstacles, line-of-sight). Python calls this implementation through
//! the native world bindings; the viewer consumes the same world crate.
//!
//! # Shape model
//!
//! Python models five primitives through an inheritance chain, but three of
//! them — `BuildingPrism`, `OrientedBox`, `WallSegment` — reduce to a polygon
//! prism, and do so explicitly: `OrientedBox._as_prism()` and
//! `WallSegment._as_prism()` forward every query to a `PolygonPrism` built from
//! their derived footprint. There are really only two shapes. This module says
//! so directly: constructors derive the footprint once, and everything after
//! that is a polygon or a cylinder. The originating `kind` is retained so
//! metadata can still be attributed to the primitive it came from.
//!
//! # A defect this port surfaced, since fixed on both sides
//!
//! `point_in_polygon` here divides by the true edge `dy`. When this was
//! written the Python original divided by `max(b.y - a.y, 1e-12)`, which
//! clamps the denominator to a tiny *positive* value on any downward-going
//! edge and makes the crossing test degenerate into "toggle iff the edge moves
//! left", independent of the query point. On the repo's own `scene.smartscene`
//! that misclassified 14–16% of probes around each rotated building.
//!
//! Axis-aligned footprints were unaffected, which is why it went unnoticed for
//! so long: every footprint in `tests/test_collision.py` was axis-aligned, and
//! that is exactly the case the degenerate branch happens to get right. Worth
//! remembering when adding fixtures — geometry tests that only use
//! axis-aligned shapes cannot see a whole class of error.
//!
//! Python was corrected to divide by the signed `dy`, so the two sides now
//! agree. `polygon_tests::rotated_footprints_classify_correctly` keeps the
//! cases that used to differ, as a guard against reintroducing it here.

use serde::{Deserialize, Serialize};

/// Tolerance for "on the boundary" comparisons, metres.
const EPSILON_M: f64 = 1.0e-9;

/// Tolerance for the segment-on-segment tests, metres.
const SEGMENT_EPSILON_M: f64 = 1.0e-6;

/// Minimum push distance applied by [`Obstacle::push_outside_xy`], metres.
///
/// Python takes `max(margin_m, 1.0)` at every use site; the floor lives here
/// instead of being repeated.
const MIN_PUSH_MARGIN_M: f64 = 1.0;

/// How many times `push_outside_xy` will step further out when the first
/// candidate is still inside a concave footprint. Matches Python's `range(6)`.
const MAX_PUSH_ITERATIONS: usize = 6;

// ---------------------------------------------------------------------------
// Bounds2d
// ---------------------------------------------------------------------------

pub use argusnet_types::Bounds2d;

// ---------------------------------------------------------------------------
// Blocker classification
// ---------------------------------------------------------------------------

/// What kind of thing is blocking, which decides how it attenuates.
///
/// Solid blockers stop a ray outright; vegetation attenuates it, which is why
/// [`Obstacle::density`] is meaningful only for [`BlockerType::Vegetation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockerType {
    Building,
    Wall,
    Vegetation,
    /// Anything the scene declared that this crate does not model specially.
    #[serde(untagged)]
    Other(String),
}

impl BlockerType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Building => "building",
            Self::Wall => "wall",
            Self::Vegetation => "vegetation",
            Self::Other(value) => value,
        }
    }

    /// Whether the blocker stops a ray rather than attenuating it.
    pub fn is_solid(&self) -> bool {
        !matches!(self, Self::Vegetation)
    }

    /// Whether a platform collides with it and a route must go around it: a
    /// building or a wall. Vegetation is seen through and flown through, and
    /// a kind this crate does not model is not a structure.
    pub fn is_hard(&self) -> bool {
        matches!(self, Self::Building | Self::Wall)
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The horizontal cross-section of an obstacle. Vertical extent lives on
/// [`Obstacle`], since every primitive is a prism of constant cross-section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObstacleShape {
    /// A closed polygon, vertices in order. Winding may be either way.
    Polygon { footprint_xy_m: Vec<[f64; 2]> },
    Cylinder {
        center_xy_m: [f64; 2],
        radius_m: f64,
    },
}

// ---------------------------------------------------------------------------
// Obstacle
// ---------------------------------------------------------------------------

/// One occluding object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Obstacle {
    pub id: String,
    pub blocker_type: BlockerType,
    pub shape: ObstacleShape,
    pub base_z_m: f64,
    pub top_z_m: f64,
    /// Canopy density, for vegetation only. `None` for solid blockers.
    pub density: Option<f64>,
    /// The `kind` tag the scene metadata used, retained so an obstacle can be
    /// attributed back to the primitive that produced it.
    pub kind: String,
}

/// Rejected construction input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObstacleError {
    /// Fewer than three vertices cannot enclose an area.
    DegenerateFootprint { vertices: usize },
    /// `top_z_m` must be strictly above `base_z_m`.
    NonPositiveHeight,
    /// A radius, length, width or thickness was zero or negative.
    NonPositiveExtent { field: &'static str },
}

impl std::fmt::Display for ObstacleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateFootprint { vertices } => write!(
                f,
                "a footprint needs at least three vertices, got {vertices}"
            ),
            Self::NonPositiveHeight => write!(f, "top_z_m must be greater than base_z_m"),
            Self::NonPositiveExtent { field } => write!(f, "{field} must be greater than zero"),
        }
    }
}

impl std::error::Error for ObstacleError {}

impl Obstacle {
    /// A polygonal prism — the `polygon-prism-v1` and `building-v1` kinds.
    pub fn polygon_prism(
        id: impl Into<String>,
        blocker_type: BlockerType,
        footprint_xy_m: Vec<[f64; 2]>,
        base_z_m: f64,
        top_z_m: f64,
    ) -> Result<Self, ObstacleError> {
        if footprint_xy_m.len() < 3 {
            return Err(ObstacleError::DegenerateFootprint {
                vertices: footprint_xy_m.len(),
            });
        }
        if top_z_m <= base_z_m {
            return Err(ObstacleError::NonPositiveHeight);
        }
        Ok(Self {
            id: id.into(),
            blocker_type,
            shape: ObstacleShape::Polygon { footprint_xy_m },
            base_z_m,
            top_z_m,
            density: None,
            kind: "polygon-prism-v1".to_string(),
        })
    }

    /// A stand of trees: a polygonal prism that attenuates rather than blocks.
    pub fn forest_stand(
        id: impl Into<String>,
        footprint_xy_m: Vec<[f64; 2]>,
        canopy_base_z_m: f64,
        canopy_top_z_m: f64,
        density: f64,
    ) -> Result<Self, ObstacleError> {
        let mut obstacle = Self::polygon_prism(
            id,
            BlockerType::Vegetation,
            footprint_xy_m,
            canopy_base_z_m,
            canopy_top_z_m,
        )?;
        // Python clamps with `max(density, 0.0)` rather than rejecting.
        obstacle.density = Some(density.max(0.0));
        obstacle.kind = "forest-stand-v1".to_string();
        Ok(obstacle)
    }

    /// A rotated rectangular box — the `box-v1` kind.
    ///
    /// The footprint is derived here, once, mirroring Python's
    /// `OrientedBox.footprint_xy_m()` followed by `_as_prism()`.
    #[allow(clippy::too_many_arguments)]
    pub fn oriented_box(
        id: impl Into<String>,
        blocker_type: BlockerType,
        center_x_m: f64,
        center_y_m: f64,
        length_m: f64,
        width_m: f64,
        yaw_rad: f64,
        base_z_m: f64,
        top_z_m: f64,
    ) -> Result<Self, ObstacleError> {
        if length_m <= 0.0 {
            return Err(ObstacleError::NonPositiveExtent { field: "length_m" });
        }
        if width_m <= 0.0 {
            return Err(ObstacleError::NonPositiveExtent { field: "width_m" });
        }
        let (half_length, half_width) = (length_m * 0.5, width_m * 0.5);
        let (sin_yaw, cos_yaw) = yaw_rad.sin_cos();
        let corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ];
        let footprint = corners
            .iter()
            .map(|[x, y]| {
                [
                    center_x_m + (x * cos_yaw) - (y * sin_yaw),
                    center_y_m + (x * sin_yaw) + (y * cos_yaw),
                ]
            })
            .collect();

        let mut obstacle = Self::polygon_prism(id, blocker_type, footprint, base_z_m, top_z_m)?;
        obstacle.kind = "box-v1".to_string();
        Ok(obstacle)
    }

    /// A wall of finite thickness between two points — the `wall-v1` kind.
    #[allow(clippy::too_many_arguments)]
    pub fn wall_segment(
        id: impl Into<String>,
        blocker_type: BlockerType,
        start_xy_m: [f64; 2],
        end_xy_m: [f64; 2],
        thickness_m: f64,
        base_z_m: f64,
        top_z_m: f64,
    ) -> Result<Self, ObstacleError> {
        if thickness_m <= 0.0 {
            return Err(ObstacleError::NonPositiveExtent {
                field: "thickness_m",
            });
        }
        let segment = [end_xy_m[0] - start_xy_m[0], end_xy_m[1] - start_xy_m[1]];
        let length = segment[0].hypot(segment[1]).max(EPSILON_M);
        let normal = [-segment[1] / length, segment[0] / length];
        let offset = [normal[0] * thickness_m * 0.5, normal[1] * thickness_m * 0.5];
        let footprint = vec![
            [start_xy_m[0] - offset[0], start_xy_m[1] - offset[1]],
            [end_xy_m[0] - offset[0], end_xy_m[1] - offset[1]],
            [end_xy_m[0] + offset[0], end_xy_m[1] + offset[1]],
            [start_xy_m[0] + offset[0], start_xy_m[1] + offset[1]],
        ];

        let mut obstacle = Self::polygon_prism(id, blocker_type, footprint, base_z_m, top_z_m)?;
        obstacle.kind = "wall-v1".to_string();
        Ok(obstacle)
    }

    /// A vertical cylinder — the `cylinder-v1` kind.
    pub fn cylinder(
        id: impl Into<String>,
        blocker_type: BlockerType,
        center_x_m: f64,
        center_y_m: f64,
        radius_m: f64,
        base_z_m: f64,
        top_z_m: f64,
    ) -> Result<Self, ObstacleError> {
        if radius_m <= 0.0 {
            return Err(ObstacleError::NonPositiveExtent { field: "radius_m" });
        }
        if top_z_m <= base_z_m {
            return Err(ObstacleError::NonPositiveHeight);
        }
        Ok(Self {
            id: id.into(),
            blocker_type,
            shape: ObstacleShape::Cylinder {
                center_xy_m: [center_x_m, center_y_m],
                radius_m,
            },
            base_z_m,
            top_z_m,
            density: None,
            kind: "cylinder-v1".to_string(),
        })
    }

    // -----------------------------------------------------------------
    // Queries — the five-method interface the Python primitives expose
    // -----------------------------------------------------------------

    /// Horizontal extent of the footprint.
    pub fn bounds_xy_m(&self) -> Bounds2d {
        match &self.shape {
            ObstacleShape::Polygon { footprint_xy_m } => {
                let mut bounds = Bounds2d::new(f64::MAX, f64::MIN, f64::MAX, f64::MIN);
                for [x, y] in footprint_xy_m {
                    bounds.x_min_m = bounds.x_min_m.min(*x);
                    bounds.x_max_m = bounds.x_max_m.max(*x);
                    bounds.y_min_m = bounds.y_min_m.min(*y);
                    bounds.y_max_m = bounds.y_max_m.max(*y);
                }
                bounds
            }
            ObstacleShape::Cylinder {
                center_xy_m,
                radius_m,
            } => Bounds2d::new(
                center_xy_m[0] - radius_m,
                center_xy_m[0] + radius_m,
                center_xy_m[1] - radius_m,
                center_xy_m[1] + radius_m,
            ),
        }
    }

    /// Whether `(x_m, y_m, z_m)` is within the solid.
    pub fn point_inside(&self, x_m: f64, y_m: f64, z_m: f64) -> bool {
        if z_m < self.base_z_m - EPSILON_M || z_m > self.top_z_m + EPSILON_M {
            return false;
        }
        match &self.shape {
            ObstacleShape::Polygon { footprint_xy_m } => {
                point_in_polygon([x_m, y_m], footprint_xy_m)
            }
            ObstacleShape::Cylinder {
                center_xy_m,
                radius_m,
            } => {
                let dx = x_m - center_xy_m[0];
                let dy = y_m - center_xy_m[1];
                (dx * dx) + (dy * dy) <= (radius_m * radius_m) + EPSILON_M
            }
        }
    }

    /// Length of the `origin -> target` segment that lies inside the solid, metres.
    pub fn path_length_inside(&self, origin: [f64; 3], target: [f64; 3]) -> f64 {
        let distance_m = distance_3d(origin, target);
        self.inside_intervals(origin, target)
            .iter()
            .map(|(start, end)| (end - start) * distance_m)
            .sum()
    }

    /// Parameter along `origin -> target` at which the solid is first entered.
    pub fn first_hit_t(&self, origin: [f64; 3], target: [f64; 3]) -> Option<f64> {
        self.inside_intervals(origin, target)
            .first()
            .map(|(start, _)| *start)
    }

    /// Parameter along `origin -> target` at which the ray first meets the
    /// solid's surface in 3D: its wall, its roof or its floor.
    ///
    /// [`Self::first_hit_t`] answers where the ray enters the footprint and
    /// keeps it if the segment overlaps the solid's height range anywhere,
    /// which is what the blocking test needs. A ray coming down onto a roof
    /// enters the footprint above it, so that is not where it lands; here
    /// each footprint interval is cut to the part between the base and the
    /// top first.
    pub fn surface_hit_t(&self, origin: [f64; 3], target: [f64; 3]) -> Option<f64> {
        let dz = target[2] - origin[2];
        let mut intervals = self.inside_intervals(origin, target);
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        intervals.into_iter().find_map(|(start, end)| {
            let (lo, hi) = if dz.abs() <= f64::EPSILON {
                if origin[2] < self.base_z_m || origin[2] > self.top_z_m {
                    return None;
                }
                (start, end)
            } else {
                let at_base = (self.base_z_m - origin[2]) / dz;
                let at_top = (self.top_z_m - origin[2]) / dz;
                (start.max(at_base.min(at_top)), end.min(at_base.max(at_top)))
            };
            (lo <= hi).then_some(lo)
        })
    }

    /// Nearest point outside the footprint, at least `margin_m` clear of it.
    ///
    /// Returns the input unchanged when it is already outside.
    pub fn push_outside_xy(&self, x_m: f64, y_m: f64, margin_m: f64) -> [f64; 2] {
        let push = margin_m.max(MIN_PUSH_MARGIN_M);
        match &self.shape {
            ObstacleShape::Cylinder {
                center_xy_m,
                radius_m,
            } => {
                let delta = [x_m - center_xy_m[0], y_m - center_xy_m[1]];
                let radius = delta[0].hypot(delta[1]);
                // A point exactly at the axis has no meaningful outward
                // direction; Python picks +X, and so does this.
                let direction = if radius <= EPSILON_M {
                    [1.0, 0.0]
                } else {
                    [delta[0] / radius, delta[1] / radius]
                };
                [
                    center_xy_m[0] + direction[0] * (radius_m + push),
                    center_xy_m[1] + direction[1] * (radius_m + push),
                ]
            }
            ObstacleShape::Polygon { footprint_xy_m } => {
                let point = [x_m, y_m];
                if !point_in_polygon(point, footprint_xy_m) {
                    return point;
                }
                let is_ccw = signed_polygon_area(footprint_xy_m) >= 0.0;
                let mut best = point;
                let mut best_distance = f64::INFINITY;
                for index in 0..footprint_xy_m.len() {
                    let edge_start = footprint_xy_m[index];
                    let edge_end = footprint_xy_m[(index + 1) % footprint_xy_m.len()];
                    let boundary = nearest_point_on_segment(point, edge_start, edge_end);
                    let normal = edge_outward_normal(edge_start, edge_end, is_ccw);
                    let mut candidate = [
                        boundary[0] + normal[0] * push,
                        boundary[1] + normal[1] * push,
                    ];
                    // A concave footprint can put the first step back inside.
                    for _ in 0..MAX_PUSH_ITERATIONS {
                        if !point_in_polygon(candidate, footprint_xy_m) {
                            break;
                        }
                        candidate = [
                            candidate[0] + normal[0] * push,
                            candidate[1] + normal[1] * push,
                        ];
                    }
                    let distance = (candidate[0] - point[0]).hypot(candidate[1] - point[1]);
                    if distance < best_distance {
                        best_distance = distance;
                        best = candidate;
                    }
                }
                best
            }
        }
    }

    /// Sub-intervals of `origin -> target`, in `t`, that lie inside the solid.
    fn inside_intervals(&self, origin: [f64; 3], target: [f64; 3]) -> Vec<(f64, f64)> {
        match &self.shape {
            ObstacleShape::Polygon { footprint_xy_m } => segment_polygon_intervals(
                [origin[0], origin[1]],
                [target[0], target[1]],
                footprint_xy_m,
            )
            .into_iter()
            .filter(|(start, end)| {
                let z_start = origin[2] + (target[2] - origin[2]) * start;
                let z_end = origin[2] + (target[2] - origin[2]) * end;
                z_start.max(z_end) >= self.base_z_m && z_start.min(z_end) <= self.top_z_m
            })
            .collect(),
            ObstacleShape::Cylinder {
                center_xy_m,
                radius_m,
            } => self
                .cylinder_interval(origin, target, *center_xy_m, *radius_m)
                .into_iter()
                .collect(),
        }
    }

    fn cylinder_interval(
        &self,
        origin: [f64; 3],
        target: [f64; 3],
        center_xy_m: [f64; 2],
        radius_m: f64,
    ) -> Option<(f64, f64)> {
        let segment = [target[0] - origin[0], target[1] - origin[1]];
        let relative = [origin[0] - center_xy_m[0], origin[1] - center_xy_m[1]];
        let a = dot2(segment, segment);
        let spans_height = |start: f64, end: f64| {
            let z_start = origin[2] + (target[2] - origin[2]) * start;
            let z_end = origin[2] + (target[2] - origin[2]) * end;
            !(z_start.max(z_end) < self.base_z_m || z_start.min(z_end) > self.top_z_m)
        };

        // A vertical ray has no horizontal extent to intersect: it is either
        // inside the disc for its whole length or not at all.
        if a <= EPSILON_M {
            if dot2(relative, relative) > radius_m * radius_m {
                return None;
            }
            return spans_height(0.0, 1.0).then_some((0.0, 1.0));
        }

        let b = 2.0 * dot2(relative, segment);
        let c = dot2(relative, relative) - (radius_m * radius_m);
        let discriminant = (b * b) - (4.0 * a * c);
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_discriminant = discriminant.sqrt();
        let t0 = (-b - sqrt_discriminant) / (2.0 * a);
        let t1 = (-b + sqrt_discriminant) / (2.0 * a);
        let start = t0.min(t1).max(0.0);
        let end = t0.max(t1).min(1.0);
        if start > end {
            return None;
        }
        spans_height(start, end).then_some((start, end))
    }
}

// ---------------------------------------------------------------------------
// Planar geometry
// ---------------------------------------------------------------------------

#[inline]
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] * b[0]) + (a[1] * b[1])
}

#[inline]
fn distance_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    ((dx * dx) + (dy * dy) + (dz * dz)).sqrt()
}

/// Whether `point` lies on the closed segment `start -> end`.
pub fn point_on_segment(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> bool {
    let segment = [end[0] - start[0], end[1] - start[1]];
    let relative = [point[0] - start[0], point[1] - start[1]];
    let cross = ((segment[0] * relative[1]) - (segment[1] * relative[0])).abs();
    if cross > SEGMENT_EPSILON_M {
        return false;
    }
    let projection = dot2(relative, segment);
    if projection < -SEGMENT_EPSILON_M {
        return false;
    }
    projection <= dot2(segment, segment) + SEGMENT_EPSILON_M
}

/// Even-odd ray crossing test, boundary counted as inside.
///
/// See the module docs: this divides by the true edge `dy`, where Python
/// clamps the denominator and gets rotated footprints wrong.
pub fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        if point_on_segment(point, a, b) {
            return true;
        }
        // Half-open in y, so a vertex shared by two edges is counted once.
        if (a[1] > point[1]) != (b[1] > point[1]) {
            let crossing_x = ((b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1])) + a[0];
            if point[0] < crossing_x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Intersection parameters of two segments, if they meet within both spans.
pub fn segment_intersection_parameters(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Option<(f64, f64)> {
    let da = [a1[0] - a0[0], a1[1] - a0[1]];
    let db = [b1[0] - b0[0], b1[1] - b0[1]];
    let denominator = (da[0] * db[1]) - (da[1] * db[0]);
    if denominator.abs() <= EPSILON_M {
        return None;
    }
    let delta = [b0[0] - a0[0], b0[1] - a0[1]];
    let ta = ((delta[0] * db[1]) - (delta[1] * db[0])) / denominator;
    let tb = ((delta[0] * da[1]) - (delta[1] * da[0])) / denominator;
    let in_span = |t: f64| (-EPSILON_M..=1.0 + EPSILON_M).contains(&t);
    (in_span(ta) && in_span(tb)).then_some((ta, tb))
}

/// Sub-intervals of `origin -> target`, in `t`, that lie inside `polygon`.
///
/// Splits the segment at every edge crossing, then classifies each span by its
/// midpoint. That handles concave footprints, where a single segment can enter
/// and leave more than once.
pub fn segment_polygon_intervals(
    origin: [f64; 2],
    target: [f64; 2],
    polygon: &[[f64; 2]],
) -> Vec<(f64, f64)> {
    let mut crossings = vec![0.0, 1.0];
    for index in 0..polygon.len() {
        let edge_start = polygon[index];
        let edge_end = polygon[(index + 1) % polygon.len()];
        if let Some((t, _)) = segment_intersection_parameters(origin, target, edge_start, edge_end)
        {
            crossings.push(t);
        }
    }
    crossings.retain(|t| (-EPSILON_M..=1.0 + EPSILON_M).contains(t));
    crossings.sort_by(|a, b| a.partial_cmp(b).expect("crossing parameters are finite"));
    crossings.dedup_by(|a, b| (*a - *b).abs() <= EPSILON_M);

    let mut intervals = Vec::new();
    for window in crossings.windows(2) {
        let (start, end) = (window[0], window[1]);
        if end - start <= EPSILON_M {
            continue;
        }
        let midpoint = (start + end) * 0.5;
        let sample = [
            origin[0] + (target[0] - origin[0]) * midpoint,
            origin[1] + (target[1] - origin[1]) * midpoint,
        ];
        if point_in_polygon(sample, polygon) {
            intervals.push((start, end));
        }
    }
    intervals
}

pub fn nearest_point_on_segment(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> [f64; 2] {
    let segment = [end[0] - start[0], end[1] - start[1]];
    let length_squared = dot2(segment, segment);
    if length_squared <= 1.0e-12 {
        return start;
    }
    let relative = [point[0] - start[0], point[1] - start[1]];
    let t = (dot2(relative, segment) / length_squared).clamp(0.0, 1.0);
    [start[0] + segment[0] * t, start[1] + segment[1] * t]
}

/// Shoelace area. Positive for counter-clockwise winding.
pub fn signed_polygon_area(polygon: &[[f64; 2]]) -> f64 {
    let mut total = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        total += (a[0] * b[1]) - (a[1] * b[0]);
    }
    total * 0.5
}

/// Unit normal pointing away from the interior, given the winding direction.
pub fn edge_outward_normal(start: [f64; 2], end: [f64; 2], is_ccw: bool) -> [f64; 2] {
    let edge = [end[0] - start[0], end[1] - start[1]];
    let length = edge[0].hypot(edge[1]).max(EPSILON_M);
    if is_ccw {
        [edge[1] / length, -edge[0] / length]
    } else {
        [-edge[1] / length, edge[0] / length]
    }
}

// ---------------------------------------------------------------------------
// Scene metadata
// ---------------------------------------------------------------------------

/// One entry of a scene package's `occluding_objects` array.
///
/// Mirrors what `ObstaclePrimitive.to_metadata()` emits on the Python side.
/// Fields are optional because which ones are present depends on `kind`;
/// [`ObstacleMetadata::into_obstacle`] is where the per-kind requirements are
/// enforced.
#[derive(Debug, Clone, Deserialize)]
pub struct ObstacleMetadata {
    pub object_id: String,
    pub kind: String,
    #[serde(default)]
    pub blocker_type: Option<String>,
    #[serde(default)]
    pub footprint_xy_m: Option<FootprintCoords>,
    pub base_elevation_m: f64,
    pub top_elevation_m: f64,
    #[serde(default)]
    pub center_x_m: Option<f64>,
    #[serde(default)]
    pub center_y_m: Option<f64>,
    #[serde(default)]
    pub radius_m: Option<f64>,
    #[serde(default)]
    pub density: Option<f64>,
}

/// How a scene stored a footprint.
///
/// Both forms occur. `ObstaclePrimitive.to_metadata()` documents and emits
/// `[[x, y], ...]`, but every packaged scene currently on disk holds a flat run
/// of numbers, because the writer sends obstacle metadata through `to_jsonable`,
/// whose rule for flattening covariance matrices also catches coordinate lists.
/// A reader that accepted only one of them could not read both the documented
/// format and the shipped files, so this accepts either and normalises.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum FootprintCoords {
    Pairs(Vec<[f64; 2]>),
    Flat(Vec<f64>),
}

/// Why a metadata entry could not become an [`Obstacle`].
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataError {
    UnknownKind {
        object_id: String,
        kind: String,
    },
    MissingField {
        object_id: String,
        field: &'static str,
    },
    /// A footprint whose coordinate count is not a multiple of two.
    RaggedFootprint {
        object_id: String,
        values: usize,
    },
    Invalid {
        object_id: String,
        source: ObstacleError,
    },
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownKind { object_id, kind } => {
                write!(f, "{object_id}: unknown obstacle kind {kind:?}")
            }
            Self::MissingField { object_id, field } => {
                write!(f, "{object_id}: this obstacle kind requires {field}")
            }
            Self::RaggedFootprint { object_id, values } => write!(
                f,
                "{object_id}: footprint_xy_m has {values} values, which is not a whole number of xy pairs"
            ),
            Self::Invalid { object_id, source } => write!(f, "{object_id}: {source}"),
        }
    }
}

impl std::error::Error for MetadataError {}

impl ObstacleMetadata {
    fn require(&self, value: Option<f64>, field: &'static str) -> Result<f64, MetadataError> {
        value.ok_or(MetadataError::MissingField {
            object_id: self.object_id.clone(),
            field,
        })
    }

    /// Coordinate pairs from the stored footprint, in either encoding.
    fn footprint_pairs(&self) -> Result<Vec<[f64; 2]>, MetadataError> {
        let stored = self
            .footprint_xy_m
            .as_ref()
            .ok_or(MetadataError::MissingField {
                object_id: self.object_id.clone(),
                field: "footprint_xy_m",
            })?;
        match stored {
            FootprintCoords::Pairs(pairs) => Ok(pairs.clone()),
            FootprintCoords::Flat(values) => {
                if values.len() % 2 != 0 {
                    return Err(MetadataError::RaggedFootprint {
                        object_id: self.object_id.clone(),
                        values: values.len(),
                    });
                }
                // The length check above guarantees no remainder, so the
                // slice `as_chunks` discards is always empty.
                Ok(values.as_chunks::<2>().0.to_vec())
            }
        }
    }

    fn blocker(&self) -> BlockerType {
        match self.blocker_type.as_deref() {
            Some("building") | None => BlockerType::Building,
            Some("wall") => BlockerType::Wall,
            Some("vegetation") => BlockerType::Vegetation,
            Some(other) => BlockerType::Other(other.to_string()),
        }
    }

    /// Builds the primitive this entry describes.
    ///
    /// `box-v1` and `wall-v1` are reconstructed from their stored footprint
    /// rather than re-derived from centre/yaw or endpoints: the footprint is
    /// what the writer committed to, so honouring it keeps Rust and the baked
    /// GLBs describing the same polygon even if the derivation ever drifts.
    pub fn into_obstacle(self) -> Result<Obstacle, MetadataError> {
        let blocker = self.blocker();
        let invalid = |source| MetadataError::Invalid {
            object_id: self.object_id.clone(),
            source,
        };

        match self.kind.as_str() {
            "cylinder-v1" => {
                let center_x = self.require(self.center_x_m, "center_x_m")?;
                let center_y = self.require(self.center_y_m, "center_y_m")?;
                let radius = self.require(self.radius_m, "radius_m")?;
                Obstacle::cylinder(
                    self.object_id.clone(),
                    blocker,
                    center_x,
                    center_y,
                    radius,
                    self.base_elevation_m,
                    self.top_elevation_m,
                )
                .map_err(invalid)
            }
            "forest-stand-v1" => {
                let footprint = self.footprint_pairs()?;
                Obstacle::forest_stand(
                    self.object_id.clone(),
                    footprint,
                    self.base_elevation_m,
                    self.top_elevation_m,
                    self.density.unwrap_or(0.35),
                )
                .map_err(invalid)
            }
            kind @ ("polygon-prism-v1" | "building-v1" | "box-v1" | "wall-v1") => {
                let footprint = self.footprint_pairs()?;
                let mut obstacle = Obstacle::polygon_prism(
                    self.object_id.clone(),
                    blocker,
                    footprint,
                    self.base_elevation_m,
                    self.top_elevation_m,
                )
                .map_err(invalid)?;
                obstacle.kind = kind.to_string();
                Ok(obstacle)
            }
            _ => Err(MetadataError::UnknownKind {
                object_id: self.object_id.clone(),
                kind: self.kind.clone(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod polygon_tests {
    use super::*;

    #[test]
    fn a_ray_coming_down_lands_on_the_roof_not_where_it_crosses_the_footprint() {
        let solid = Obstacle::polygon_prism(
            "b",
            BlockerType::Building,
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            0.0,
            20.0,
        )
        .unwrap();
        // From (-10, 5, 60) down to (20, 5, -30): it crosses x = 0 at z = 30,
        // above the roof, and meets the roof (z = 20) at x = 3.33.
        let (from, to) = ([-10.0, 5.0, 60.0], [20.0, 5.0, -30.0]);
        let legacy = solid.first_hit_t(from, to).unwrap();
        assert!((legacy - 1.0 / 3.0).abs() < 1e-9);
        let t = solid.surface_hit_t(from, to).unwrap();
        let z = from[2] + (to[2] - from[2]) * t;
        assert!((z - 20.0).abs() < 1e-9, "{z}");
        // Level with the wall, the wall is hit where the footprint starts.
        let t = solid
            .surface_hit_t([-10.0, 5.0, 5.0], [20.0, 5.0, 5.0])
            .unwrap();
        assert!((t - 1.0 / 3.0).abs() < 1e-9);
        // Passing over the top misses.
        assert!(solid
            .surface_hit_t([-10.0, 5.0, 30.0], [20.0, 5.0, 25.0])
            .is_none());
    }

    fn square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    }

    fn triangle() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]]
    }

    fn diamond() -> Vec<[f64; 2]> {
        vec![[5.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]]
    }

    #[test]
    fn axis_aligned_footprints_classify_correctly() {
        assert!(point_in_polygon([5.0, 5.0], &square()));
        assert!(!point_in_polygon([15.0, 5.0], &square()));
        assert!(!point_in_polygon([5.0, -1.0], &square()));
    }

    /// The cases a clamped crossing denominator gets wrong.
    ///
    /// Both implementations answered these incorrectly until the denominator
    /// was corrected to keep its sign on downward-going edges — see the module
    /// docs. Kept as a regression guard: they are the smallest shapes that
    /// distinguish a correct crossing test from that degenerate one, and no
    /// axis-aligned fixture can stand in for them.
    #[test]
    fn rotated_footprints_classify_correctly() {
        assert!(
            point_in_polygon([5.0, 5.0], &triangle()),
            "the centroid of a triangle is inside it"
        );
        assert!(
            point_in_polygon([5.0, 5.0], &diamond()),
            "the centre of a diamond is inside it"
        );
        assert!(
            !point_in_polygon([9.0, 9.0], &diamond()),
            "a corner of the diamond's bounding box is outside the diamond"
        );
    }

    /// Cross-check against an independent property rather than hand-picked
    /// answers: a convex polygon contains exactly the points that lie on the
    /// interior side of every edge.
    #[test]
    fn agrees_with_a_half_plane_test_over_a_grid() {
        for polygon in [square(), triangle(), diamond()] {
            let is_ccw = signed_polygon_area(&polygon) >= 0.0;
            for step_x in 0..40 {
                for step_y in 0..40 {
                    let point = [-2.0 + step_x as f64 * 0.35, -2.0 + step_y as f64 * 0.35];
                    let half_plane = (0..polygon.len()).all(|index| {
                        let a = polygon[index];
                        let b = polygon[(index + 1) % polygon.len()];
                        let cross = ((b[0] - a[0]) * (point[1] - a[1]))
                            - ((b[1] - a[1]) * (point[0] - a[0]));
                        if is_ccw {
                            cross >= 0.0
                        } else {
                            cross <= 0.0
                        }
                    });
                    assert_eq!(
                        point_in_polygon(point, &polygon),
                        half_plane,
                        "disagreement at {point:?} for {polygon:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn winding_direction_does_not_change_the_answer() {
        let mut reversed = diamond();
        reversed.reverse();
        for step in 0..40 {
            let point = [step as f64 * 0.3, 5.0];
            assert_eq!(
                point_in_polygon(point, &diamond()),
                point_in_polygon(point, &reversed),
                "winding changed the verdict at {point:?}"
            );
        }
    }
}

#[cfg(test)]
mod primitive_tests {
    use super::*;

    /// The shapes from `tests/test_collision.py::ObstacleCollisionTest.setUp`,
    /// so the Rust side is exercised on the same geometry as the Python side.
    fn building() -> Obstacle {
        Obstacle::polygon_prism(
            "building-a",
            BlockerType::Building,
            vec![[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
            0.0,
            20.0,
        )
        .expect("valid building")
    }

    fn box_obstacle() -> Obstacle {
        Obstacle::oriented_box(
            "box-a",
            BlockerType::Building,
            60.0,
            20.0,
            20.0,
            10.0,
            0.0,
            0.0,
            25.0,
        )
        .expect("valid box")
    }

    fn wall() -> Obstacle {
        Obstacle::wall_segment(
            "wall-a",
            BlockerType::Wall,
            [80.0, 10.0],
            [80.0, 30.0],
            4.0,
            0.0,
            18.0,
        )
        .expect("valid wall")
    }

    fn tower() -> Obstacle {
        Obstacle::cylinder(
            "tower-a",
            BlockerType::Building,
            110.0,
            20.0,
            6.0,
            0.0,
            30.0,
        )
        .expect("valid cylinder")
    }

    fn forest() -> Obstacle {
        Obstacle::forest_stand(
            "forest-a",
            vec![[130.0, 10.0], [150.0, 10.0], [150.0, 30.0], [130.0, 30.0]],
            2.0,
            22.0,
            0.5,
        )
        .expect("valid forest")
    }

    /// Mirrors `test_point_inside_covers_all_primitive_types`.
    #[test]
    fn point_inside_covers_all_primitive_types() {
        assert!(building().point_inside(20.0, 20.0, 10.0));
        assert!(!building().point_inside(20.0, 20.0, 25.0), "above the roof");
        assert!(box_obstacle().point_inside(60.0, 20.0, 10.0));
        assert!(wall().point_inside(80.0, 20.0, 10.0));
        assert!(tower().point_inside(110.0, 20.0, 10.0));
        assert!(forest().point_inside(140.0, 20.0, 10.0));
        assert!(
            !forest().point_inside(140.0, 20.0, 1.0),
            "below the canopy base"
        );
    }

    /// Mirrors `test_push_outside_xy_moves_point_outside`.
    #[test]
    fn push_outside_xy_moves_the_point_clear() {
        for obstacle in [building(), box_obstacle(), wall(), tower(), forest()] {
            let bounds = obstacle.bounds_xy_m();
            let inside_x = (bounds.x_min_m + bounds.x_max_m) * 0.5;
            let inside_y = (bounds.y_min_m + bounds.y_max_m) * 0.5;
            let mid_z = (obstacle.base_z_m + obstacle.top_z_m) * 0.5;
            assert!(
                obstacle.point_inside(inside_x, inside_y, mid_z),
                "{} should contain its own centre",
                obstacle.id
            );

            let [pushed_x, pushed_y] = obstacle.push_outside_xy(inside_x, inside_y, 1.0);
            assert!(
                !obstacle.point_inside(pushed_x, pushed_y, mid_z),
                "{} left the pushed point inside at ({pushed_x}, {pushed_y})",
                obstacle.id
            );
        }
    }

    #[test]
    fn push_outside_leaves_an_outside_point_alone() {
        let outside = building().push_outside_xy(100.0, 100.0, 1.0);
        assert_eq!(outside, [100.0, 100.0]);
    }

    #[test]
    fn a_ray_through_a_box_reports_its_chord() {
        let obstacle = box_obstacle();
        // Straight through the middle along X: the box is 20 m long.
        let length = obstacle.path_length_inside([0.0, 20.0, 10.0], [200.0, 20.0, 10.0]);
        assert!(
            (length - 20.0).abs() < 1.0e-6,
            "expected a 20 m chord, got {length}"
        );
    }

    #[test]
    fn a_ray_that_misses_reports_nothing() {
        let obstacle = box_obstacle();
        let origin = [0.0, 200.0, 10.0];
        let target = [200.0, 200.0, 10.0];
        assert_eq!(obstacle.path_length_inside(origin, target), 0.0);
        assert_eq!(obstacle.first_hit_t(origin, target), None);
    }

    /// A ray passing over the roof must not register, even though it crosses
    /// the footprint in plan view.
    #[test]
    fn a_ray_above_the_roof_does_not_register() {
        let obstacle = building();
        let over = [0.0, 20.0, 100.0];
        let target = [200.0, 20.0, 100.0];
        assert_eq!(obstacle.path_length_inside(over, target), 0.0);
        assert_eq!(obstacle.first_hit_t(over, target), None);
    }

    #[test]
    fn first_hit_is_the_near_face() {
        let obstacle = tower();
        // 6 m radius at x = 110, approached from the origin along X.
        let hit = obstacle
            .first_hit_t([0.0, 20.0, 10.0], [220.0, 20.0, 10.0])
            .expect("the ray passes through the tower");
        let entry_x = hit * 220.0;
        assert!(
            (entry_x - 104.0).abs() < 1.0e-6,
            "expected entry at x = 104, got {entry_x}"
        );
    }

    #[test]
    fn a_vertical_ray_inside_a_cylinder_is_fully_enclosed() {
        let obstacle = tower();
        let length = obstacle.path_length_inside([110.0, 20.0, 5.0], [110.0, 20.0, 25.0]);
        assert!(
            (length - 20.0).abs() < 1.0e-6,
            "expected the full 20 m, got {length}"
        );
    }

    #[test]
    fn bounds_cover_the_footprint() {
        let bounds = box_obstacle().bounds_xy_m();
        assert!((bounds.x_min_m - 50.0).abs() < 1.0e-9);
        assert!((bounds.x_max_m - 70.0).abs() < 1.0e-9);
        assert!((bounds.y_min_m - 15.0).abs() < 1.0e-9);
        assert!((bounds.y_max_m - 25.0).abs() < 1.0e-9);

        let tower_bounds = tower().bounds_xy_m();
        assert!((tower_bounds.x_min_m - 104.0).abs() < 1.0e-9);
        assert!((tower_bounds.x_max_m - 116.0).abs() < 1.0e-9);
    }

    /// A rotated box must enclose its own centre and have the area its
    /// dimensions imply — the property the Python point-in-polygon defect
    /// breaks for exactly this shape.
    #[test]
    fn a_rotated_box_still_encloses_its_centre() {
        for yaw_rad in [0.0, 0.12, 0.18, 0.7, 1.4, -0.4] {
            let obstacle = Obstacle::oriented_box(
                "rotated",
                BlockerType::Building,
                120.0,
                -210.0,
                432.0,
                252.0,
                yaw_rad,
                0.0,
                50.0,
            )
            .expect("valid box");
            assert!(
                obstacle.point_inside(120.0, -210.0, 25.0),
                "a box at yaw {yaw_rad} did not contain its own centre"
            );
            let ObstacleShape::Polygon { footprint_xy_m } = &obstacle.shape else {
                panic!("an oriented box is stored as a polygon");
            };
            let area = signed_polygon_area(footprint_xy_m).abs();
            assert!(
                (area - 432.0 * 252.0).abs() < 1.0e-6,
                "rotation changed the area at yaw {yaw_rad}: {area}"
            );
        }
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        assert_eq!(
            Obstacle::polygon_prism(
                "x",
                BlockerType::Building,
                vec![[0.0, 0.0], [1.0, 1.0]],
                0.0,
                1.0
            ),
            Err(ObstacleError::DegenerateFootprint { vertices: 2 })
        );
        assert_eq!(
            Obstacle::cylinder("x", BlockerType::Building, 0.0, 0.0, 0.0, 0.0, 1.0),
            Err(ObstacleError::NonPositiveExtent { field: "radius_m" })
        );
        assert_eq!(
            Obstacle::cylinder("x", BlockerType::Building, 0.0, 0.0, 1.0, 5.0, 5.0),
            Err(ObstacleError::NonPositiveHeight)
        );
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    fn parse(json: &str) -> Result<Obstacle, MetadataError> {
        serde_json::from_str::<ObstacleMetadata>(json)
            .expect("well-formed metadata")
            .into_obstacle()
    }

    /// Both encodings must land on the same geometry: the documented nested
    /// form, and the flat form every packaged scene currently holds.
    #[test]
    fn flat_and_nested_footprints_agree() {
        let flat = parse(
            r#"{"object_id":"a","kind":"polygon-prism-v1","blocker_type":"building",
                "footprint_xy_m":[0.0,0.0,10.0,0.0,10.0,10.0,0.0,10.0],
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        )
        .expect("flat footprint parses");
        let nested = parse(
            r#"{"object_id":"a","kind":"polygon-prism-v1","blocker_type":"building",
                "footprint_xy_m":[[0.0,0.0],[10.0,0.0],[10.0,10.0],[0.0,10.0]],
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        )
        .expect("nested footprint parses");
        assert_eq!(flat, nested);
    }

    #[test]
    fn an_odd_footprint_length_is_rejected_not_truncated() {
        let result = parse(
            r#"{"object_id":"ragged","kind":"polygon-prism-v1",
                "footprint_xy_m":[0.0,0.0,10.0,0.0,10.0],
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        );
        assert_eq!(
            result,
            Err(MetadataError::RaggedFootprint {
                object_id: "ragged".to_string(),
                values: 5
            })
        );
    }

    #[test]
    fn a_cylinder_needs_its_centre_and_radius() {
        let result = parse(
            r#"{"object_id":"t","kind":"cylinder-v1","center_x_m":1.0,
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        );
        assert_eq!(
            result,
            Err(MetadataError::MissingField {
                object_id: "t".to_string(),
                field: "center_y_m"
            })
        );
    }

    #[test]
    fn an_unknown_kind_is_reported_rather_than_guessed() {
        let result = parse(
            r#"{"object_id":"z","kind":"sphere-v9",
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        );
        assert!(matches!(result, Err(MetadataError::UnknownKind { .. })));
    }

    #[test]
    fn vegetation_carries_a_density_and_solids_do_not() {
        let forest = parse(
            r#"{"object_id":"f","kind":"forest-stand-v1","blocker_type":"vegetation",
                "footprint_xy_m":[0.0,0.0,10.0,0.0,10.0,10.0,0.0,10.0],
                "base_elevation_m":2.0,"top_elevation_m":22.0,"density":0.5}"#,
        )
        .expect("forest parses");
        assert_eq!(forest.density, Some(0.5));
        assert!(!forest.blocker_type.is_solid());

        let building = parse(
            r#"{"object_id":"b","kind":"polygon-prism-v1","blocker_type":"building",
                "footprint_xy_m":[0.0,0.0,10.0,0.0,10.0,10.0,0.0,10.0],
                "base_elevation_m":0.0,"top_elevation_m":5.0}"#,
        )
        .expect("building parses");
        assert_eq!(building.density, None);
        assert!(building.blocker_type.is_solid());
    }

    /// Only buildings and walls are hard: what a platform collides with and a
    /// route goes around. A kind read from the scene that this crate does not
    /// model (here a capitalised "Building") is not one, as before, when the
    /// rule was a string match on `as_str()`.
    #[test]
    fn only_buildings_and_walls_are_hard() {
        let hard = |text: &str| {
            serde_json::from_str::<BlockerType>(text)
                .expect("parses")
                .is_hard()
        };
        assert!(hard(r#""building""#));
        assert!(hard(r#""wall""#));
        assert!(!hard(r#""vegetation""#));
        assert!(!hard(r#""Building""#));
        assert!(!hard(r#""tower""#));
    }
}
