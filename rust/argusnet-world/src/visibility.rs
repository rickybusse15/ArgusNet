//! Two-level DDA visibility and probabilistic detection, in metres and ENU.
use crate::{index::ObstacleLayer, obstacles::Bounds2d, tiled::TiledTerrain, TerrainQuery};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
fn point(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
fn xy(p: [f64; 3]) -> [f64; 2] {
    [p[0], p[1]]
}
pub fn clip(a: [f64; 2], b: [f64; 2], bounds: Bounds2d) -> Option<(f64, f64)> {
    let mut lo: f64 = 0.0;
    let mut hi: f64 = 1.0;
    for (p, q) in [
        (-(b[0] - a[0]), a[0] - bounds.x_min_m),
        (b[0] - a[0], bounds.x_max_m - a[0]),
        (-(b[1] - a[1]), a[1] - bounds.y_min_m),
        (b[1] - a[1], bounds.y_max_m - a[1]),
    ] {
        if p.abs() < 1e-12 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                lo = lo.max(t);
            } else {
                hi = hi.min(t);
            }
            if lo > hi {
                return None;
            }
        }
    }
    Some((lo, hi))
}
/// Returns visited cells and parameters in the original segment's [0,1] domain.
pub fn dda(a: [f64; 2], b: [f64; 2], bounds: Bounds2d, cell: f64) -> Vec<(i64, i64, f64, f64)> {
    if !cell.is_finite() || cell <= 0.0 {
        return vec![];
    }
    let Some((start, end)) = clip(a, b, bounds) else {
        return vec![];
    };
    let d = [b[0] - a[0], b[1] - a[1]];
    let mins = [bounds.x_min_m, bounds.y_min_m];
    let counts = [
        ((bounds.x_max_m - bounds.x_min_m) / cell).ceil() as i64,
        ((bounds.y_max_m - bounds.y_min_m) / cell).ceil() as i64,
    ];
    let mut ids = [0; 2];
    let mut step = [0; 2];
    let mut next = [f64::INFINITY; 2];
    let mut delta = [f64::INFINITY; 2];
    for i in 0..2 {
        ids[i] = (((a[i] + d[i] * start - mins[i]) / cell).floor() as i64)
            .clamp(0, (counts[i] - 1).max(0));
        if d[i].abs() > 1e-12 {
            step[i] = if d[i] > 0.0 { 1 } else { -1 };
            let boundary = mins[i] + (ids[i] + i64::from(step[i] > 0)) as f64 * cell;
            next[i] = (boundary - a[i]) / d[i];
            delta[i] = (cell / d[i]).abs();
        }
    }
    let mut t = start;
    let mut out = vec![];
    // At most one crossing per row/column; bound also protects degenerate input.
    for _ in 0..(counts[0] + counts[1] + 3).max(1) {
        let t1 = end.min(next[0]).min(next[1]).max(t);
        out.push((ids[0], ids[1], t, t1));
        if t1 >= end - 1e-9 {
            break;
        }
        let axis = usize::from(next[0] > next[1]);
        ids[axis] += step[axis];
        next[axis] += delta[axis];
        t = t1;
        if ids[axis] < 0 || ids[axis] >= counts[axis] {
            break;
        }
    }
    out
}
fn coverage_point(a: [f64; 3], b: [f64; 3], bounds: Bounds2d) -> [f64; 3] {
    let mut p = clip(xy(a), xy(b), bounds).map_or(b, |(lo, hi)| {
        point(
            a,
            b,
            if bounds.contains_xy(a[0], a[1]) {
                hi
            } else {
                lo
            },
        )
    });
    p[0] = p[0].clamp(bounds.x_min_m, bounds.x_max_m);
    p[1] = p[1].clamp(bounds.y_min_m, bounds.y_max_m);
    p
}
pub fn terrain_intersection(
    t: &TiledTerrain,
    a: [f64; 3],
    b: [f64; 3],
    clearance: f64,
) -> Option<(String, Option<f64>, [f64; 3])> {
    let intervals = dda(
        xy(a),
        xy(b),
        t.bounds,
        t.tile_size_cells as f64 * t.base_resolution_m,
    );
    if intervals.is_empty() {
        return Some((
            "out_of_coverage".into(),
            None,
            coverage_point(a, b, t.bounds),
        ));
    }
    for (x, y, t0, t1) in intervals {
        let Some(tile) = t.tiles.get(&(0, x, y)) else {
            return Some((
                "out_of_coverage".into(),
                None,
                coverage_point(a, b, t.bounds),
            ));
        };
        let p0 = point(a, b, t0);
        let p1 = point(a, b, t1);
        if tile.max_height() + clearance < p0[2].min(p1[2]) {
            continue;
        }
        for (_, _, c0, c1) in dda(xy(p0), xy(p1), tile.bounds(), tile.cell_size_m) {
            let mid = t0 + (c0 + c1) * 0.5 * (t1 - t0);
            let mut p = point(a, b, mid);
            let h = t.height_at(p[0], p[1]) + clearance;
            if p[2] <= h {
                p[2] = h;
                return Some(("terrain".into(), Some(mid), p));
            }
        }
    }
    None
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibilityProfile {
    pub hard_blockers: BTreeSet<String>,
    pub attenuation_by_land_cover: BTreeMap<i32, f64>,
    pub noise_multiplier_by_land_cover: BTreeMap<i32, f64>,
    pub vegetation_hard_block_threshold: f64,
}
impl Default for VisibilityProfile {
    fn default() -> Self {
        Self {
            hard_blockers: ["terrain", "building", "wall", "out_of_coverage"]
                .map(str::to_owned)
                .into(),
            attenuation_by_land_cover: [1.0, 0.92, 0.82, 0.9, 0.95, 0.88, 0.98, 0.97, 1.0]
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i as i32, v))
                .collect(),
            noise_multiplier_by_land_cover: [1.0, 1.12, 1.45, 1.18, 1.15, 1.25, 1.05, 1.08, 1.0]
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i as i32, v))
                .collect(),
            vegetation_hard_block_threshold: 0.12,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibilityResult {
    pub visible: bool,
    pub blocker_type: String,
    pub first_hit_range_m: Option<f64>,
    pub closest_point: Option<[f64; 3]>,
    pub transmittance: f64,
    pub detection_multiplier: f64,
    pub noise_multiplier: f64,
}
impl Default for VisibilityResult {
    fn default() -> Self {
        Self {
            visible: true,
            blocker_type: "none".into(),
            first_hit_range_m: None,
            closest_point: None,
            transmittance: 1.0,
            detection_multiplier: 1.0,
            noise_multiplier: 1.0,
        }
    }
}
pub fn los(
    terrain: &TiledTerrain,
    obstacles: &ObstacleLayer,
    a: [f64; 3],
    b: [f64; 3],
    profile: &VisibilityProfile,
    land_cover: i32,
    clearance: f64,
) -> VisibilityResult {
    let distance = (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<f64>().sqrt();
    if distance <= 1e-9 {
        return VisibilityResult::default();
    }
    let blocked = |kind: String, t: Option<f64>, p: [f64; 3], noise: f64| VisibilityResult {
        visible: false,
        blocker_type: kind,
        first_hit_range_m: t.map(|t| t * distance),
        closest_point: Some(p),
        transmittance: 0.0,
        detection_multiplier: 0.0,
        noise_multiplier: noise,
    };
    if !terrain.bounds.contains_xy(a[0], a[1]) || !terrain.bounds.contains_xy(b[0], b[1]) {
        return blocked(
            "out_of_coverage".into(),
            Some(0.0),
            coverage_point(a, b, terrain.bounds),
            profile
                .noise_multiplier_by_land_cover
                .values()
                .copied()
                .fold(1.0, f64::max),
        );
    }
    if let Some((kind, t, p)) = terrain_intersection(terrain, a, b, clearance) {
        return blocked(kind, t, p, 2.0);
    }
    let bounds = Bounds2d::new(
        a[0].min(b[0]),
        a[0].max(b[0]) + 1e-6,
        a[1].min(b[1]),
        a[1].max(b[1]) + 1e-6,
    );
    let mut nearest: Option<(f64, String)> = None;
    let mut vegetation: Option<f64> = None;
    let mut transmittance = 1.0;
    for o in obstacles.query(bounds) {
        if let Some(t) = o.first_hit_t(a, b) {
            if o.blocker_type.as_str() == "vegetation" {
                vegetation = Some(vegetation.map_or(t, |v| v.min(t)));
                if let Some(d) = o.density {
                    transmittance *= (-d * o.path_length_inside(a, b) / 45.0).exp();
                }
            } else if profile.hard_blockers.contains(o.blocker_type.as_str())
                && nearest.as_ref().is_none_or(|(v, _)| t < *v)
            {
                nearest = Some((t, o.blocker_type.as_str().into()));
            }
        }
    }
    if let Some((t, kind)) = nearest {
        return blocked(kind, Some(t), point(a, b, t), 2.0);
    }
    let visible = transmittance >= profile.vegetation_hard_block_threshold;
    VisibilityResult {
        visible,
        blocker_type: if visible { "none" } else { "vegetation" }.into(),
        first_hit_range_m: vegetation.map(|v| v * distance),
        closest_point: vegetation.map(|v| point(a, b, v)),
        transmittance,
        detection_multiplier: if visible {
            transmittance
                * profile
                    .attenuation_by_land_cover
                    .get(&land_cover)
                    .copied()
                    .unwrap_or(1.0)
        } else {
            0.0
        },
        noise_multiplier: (profile
            .noise_multiplier_by_land_cover
            .get(&land_cover)
            .copied()
            .unwrap_or(1.0)
            / transmittance.max(0.05))
        .max(1.0),
    }
}
pub fn path_loss(range: f64, max_range: f64) -> f64 {
    if range <= 0.0 {
        1.0
    } else {
        (-std::f64::consts::LN_2 * (range / (max_range * 0.5).max(1.0)).powi(2)).exp()
    }
}
pub fn effective_noise(range: f64, land_noise: f64, weather_noise: f64, max_range: f64) -> f64 {
    (land_noise * (1.0 + range / max_range.max(1.0) * 1.5) * weather_noise).max(1.0)
}
pub fn dominant(path: f64, trans: f64, weather: f64, blocked: bool) -> &'static str {
    if blocked {
        "blocked"
    } else if path <= trans && path <= weather {
        "range"
    } else if trans <= weather {
        "transmittance"
    } else {
        "weather"
    }
}
#[derive(Debug, Serialize)]
pub struct DetectionResult {
    pub p_d: f64,
    pub effective_noise_multiplier: f64,
    pub dominant_loss_factor: String,
}
pub fn detection(
    vis: &VisibilityResult,
    range: f64,
    max_range: f64,
    weather: f64,
    weather_noise: f64,
) -> DetectionResult {
    let path = path_loss(range, max_range);
    DetectionResult {
        p_d: (path * vis.transmittance * weather.max(0.0) * vis.detection_multiplier)
            .clamp(0.0, 1.0),
        effective_noise_multiplier: effective_noise(
            range,
            vis.noise_multiplier,
            weather_noise,
            max_range,
        ),
        dominant_loss_factor: dominant(path, vis.transmittance, weather, !vis.visible).into(),
    }
}
