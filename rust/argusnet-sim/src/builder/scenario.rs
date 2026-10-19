//! The scenario half of `build_default_scenario`, and the bundle it exports
//! (P4.9b.2b).
//!
//! Targets, ground stations, drones and their controllers, sensors, mission
//! zones, inspection POIs and the mission section, ported line for line from
//! `sim.py` and assembled with [`super::BuiltWorld`] into the document
//! `export_bundle` writes. Constant tables come from `reference_tables.json`
//! (`scripts/export_builder_tables.py`), generated from the reference and
//! checked for drift.
//!
//! One generator threads through the platforms, as in the reference:
//! `default_rng(seed)` supplies every drone's follow and search phase in
//! drone order, then the mission zones. Targets, POIs and sensors derive
//! their own seeds.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::sync::{Arc, LazyLock};

use serde_json::{json, Map, Value};

use argusnet_planning::planner::{PlannerConfig, RoutePlanner, Scene};
use argusnet_world::index::ObstacleLayer;
use argusnet_world::numpy_rng::NumpyRng;
use argusnet_world::obstacles::{Obstacle, ObstacleMetadata};
use argusnet_world::powerline::PowerLine;

use super::behaviors::{build_target_behavior, Bounds};
use super::environment::MapBounds;
use super::{BuildError, BuiltWorld, ScenarioConfig, ScenarioOptions};
use crate::deterministic::{stable_seed, SeedPart};
use crate::motion::{BoundsXy, TerrainSampler};
use crate::runtime_world::SharedTerrain;
use crate::trajectory::{apply_weather_adjustments, WeatherContext};
use crate::weather::MotionWeather;

/// The reference's constant tables, as `scripts/export_builder_tables.py`
/// writes them.
pub static TABLES: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("reference_tables.json"))
        .expect("reference_tables.json is valid JSON")
});

fn table(path: &[&str]) -> &'static Value {
    let mut value: &Value = &TABLES;
    for key in path {
        value = &value[*key];
    }
    assert!(!value.is_null(), "reference table {path:?} is missing");
    value
}

fn number(path: &[&str]) -> f64 {
    table(path).as_f64().expect("a numeric table entry")
}

fn dynamics(key: &str) -> f64 {
    number(&["dynamics", key])
}

fn sensor(key: &str) -> f64 {
    number(&["sensor", key])
}

fn sensor_list(key: &str) -> Vec<f64> {
    table(&["sensor", key])
        .as_array()
        .expect("a list")
        .iter()
        .map(|v| v.as_f64().expect("numeric"))
        .collect()
}

/// `np.arange(start, stop, step)` for floats: `ceil((stop - start) / step)`
/// values, filled as `start + i * (a[1] - a[0])` -- the rounded difference of
/// the first two, not `step`.
pub fn np_arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let length = ((stop - start) / step).ceil();
    // Also empty for a NaN length, as numpy's `arange` is.
    if length.is_nan() || length <= 0.0 {
        return Vec::new();
    }
    let length = length as usize;
    let mut values = Vec::with_capacity(length);
    values.push(start);
    if length > 1 {
        let second = start + step;
        let delta = second - start;
        values.push(second);
        for i in 2..length {
            values.push(start + i as f64 * delta);
        }
    }
    values
}

/// `clamp_to_bounds`, including its collapse to the midpoint when the margin
/// swallows the span.
pub fn clamp_to_bounds(xy: [f64; 2], bounds: MapBounds, margin_m: f64) -> [f64; 2] {
    let (mut min_x, mut max_x) = (bounds.x_min_m + margin_m, bounds.x_max_m - margin_m);
    let (mut min_y, mut max_y) = (bounds.y_min_m + margin_m, bounds.y_max_m - margin_m);
    if min_x > max_x {
        min_x = (bounds.x_min_m + bounds.x_max_m) * 0.5;
        max_x = min_x;
    }
    if min_y > max_y {
        min_y = (bounds.y_min_m + bounds.y_max_m) * 0.5;
        max_y = min_y;
    }
    // `np.clip`: `min(max(v, lo), hi)`.
    [xy[0].max(min_x).min(max_x), xy[1].max(min_y).min(max_y)]
}

/// `build_lawnmower_waypoints`.
pub fn lawnmower_waypoints(bounds: MapBounds, lane_spacing_m: f64, inset: f64) -> Vec<[f64; 2]> {
    let span_x = bounds.x_max_m - bounds.x_min_m;
    let span_y = bounds.y_max_m - bounds.y_min_m;
    let x_min = bounds.x_min_m + span_x * inset;
    let x_max = bounds.x_max_m - span_x * inset;
    let y_min = bounds.y_min_m + span_y * inset;
    let y_max = bounds.y_max_m - span_y * inset;
    if x_max <= x_min || y_max <= y_min {
        let cx = (bounds.x_min_m + bounds.x_max_m) * 0.5;
        let cy = (bounds.y_min_m + bounds.y_max_m) * 0.5;
        return vec![[cx - span_x * 0.15, cy], [cx + span_x * 0.15, cy]];
    }
    let lane = lane_spacing_m.max(20.0);
    let mut ys = np_arange(y_min, y_max + lane * 0.5, lane);
    if ys.is_empty() {
        ys = vec![y_min, y_max];
    } else if *ys.last().expect("non-empty") < y_max {
        ys.push(y_max);
    }
    let mut waypoints = Vec::new();
    let mut sweep_to_max = true;
    for y in ys {
        if sweep_to_max {
            waypoints.push([x_min, y]);
            waypoints.push([x_max, y]);
        } else {
            waypoints.push([x_max, y]);
            waypoints.push([x_min, y]);
        }
        sweep_to_max = !sweep_to_max;
    }
    waypoints
}

/// `perimeter_waypoints`.
fn perimeter_waypoints(bounds: MapBounds, inset: f64) -> Vec<[f64; 2]> {
    let ix = (bounds.x_max_m - bounds.x_min_m) * inset;
    let iy = (bounds.y_max_m - bounds.y_min_m) * inset;
    vec![
        [bounds.x_min_m + ix, bounds.y_min_m + iy],
        [bounds.x_max_m - ix, bounds.y_min_m + iy],
        [bounds.x_max_m - ix, bounds.y_max_m - iy],
        [bounds.x_min_m + ix, bounds.y_max_m - iy],
    ]
}

/// `split_bounds_along_x`.
fn split_bounds_along_x(bounds: MapBounds, count: usize) -> Vec<MapBounds> {
    if count <= 1 {
        return vec![bounds];
    }
    let width = (bounds.x_max_m - bounds.x_min_m) / count as f64;
    (0..count)
        .map(|i| MapBounds {
            x_min_m: bounds.x_min_m + i as f64 * width,
            x_max_m: bounds.x_min_m + (i + 1) as f64 * width,
            y_min_m: bounds.y_min_m,
            y_max_m: bounds.y_max_m,
        })
        .collect()
}

/// `drone_mode_sequence`.
fn drone_mode_sequence(preset: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|i| match preset {
            "mixed" => if i % 2 == 0 { "follow" } else { "search" }.to_string(),
            "inspect" => "follow".to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// `ground_station_name`.
fn ground_station_name(index: usize) -> String {
    const NAMES: [&str; 12] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima",
    ];
    NAMES.get(index).map_or_else(
        || format!("ground-{:02}", index + 1),
        |n| format!("ground-{n}"),
    )
}

/// The world's pieces the platforms read.
struct Ground<'a> {
    world: &'a BuiltWorld,
    terrain: SharedTerrain,
    layer: ObstacleLayer,
}

impl Ground<'_> {
    fn height_at(&self, x: f64, y: f64) -> f64 {
        TerrainSampler::height_at(&self.terrain, x, y)
    }

    fn push_outside(&self, xy: [f64; 2], agl_m: f64) -> [f64; 2] {
        push_outside_obstacles(xy, agl_m, &|x, y| self.height_at(x, y), &self.layer)
    }
}

/// `push_waypoint_outside_obstacles`: up to eight pushes out of whichever
/// solid obstacle the point is inside at `agl_m`, each with a 1 m margin --
/// a push can land in a neighbour, which the next round resolves.
pub fn push_outside_obstacles(
    xy: [f64; 2],
    agl_m: f64,
    height_at: &dyn Fn(f64, f64) -> f64,
    layer: &ObstacleLayer,
) -> [f64; 2] {
    let mut candidate = xy;
    for _ in 0..8 {
        let z = height_at(candidate[0], candidate[1]) + agl_m;
        match layer.point_collides([candidate[0], candidate[1], z]) {
            None => return candidate,
            Some(obstacle) => {
                candidate = obstacle.push_outside_xy(candidate[0], candidate[1], 1.0);
            }
        }
    }
    candidate
}

/// `generate_mission_zones`' candidate lists: the highest samples, and the
/// flattest (ties to the higher), each the first `limit`. Python's `sorted`
/// is stable, and so are these, so equal keys keep sample order.
fn candidate_orders(
    samples: &[([f64; 2], f64, f64)],
    limit: usize,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut by_height: Vec<&([f64; 2], f64, f64)> = samples.iter().collect();
    by_height.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite"));
    let mut by_flatness: Vec<&([f64; 2], f64, f64)> = samples.iter().collect();
    by_flatness.sort_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .expect("finite")
            .then((-a.1).partial_cmp(&(-b.1)).expect("finite"))
    });
    (
        by_height.iter().take(limit).map(|s| s.0).collect(),
        by_flatness.iter().take(limit).map(|s| s.0).collect(),
    )
}

struct StationSpec {
    node_id: String,
    xy: [f64; 2],
    mast_agl_m: f64,
    bearing_std_rad: f64,
    dropout_probability: f64,
    max_range_m: f64,
}

/// `build_ground_station_specs`.
fn ground_station_specs(ground: &Ground<'_>, bounds: MapBounds, count: usize) -> Vec<StationSpec> {
    let layout = table(&["ground_station_layout"]);
    let offsets: Vec<[f64; 2]> = layout["normalized_offsets"]
        .as_array()
        .expect("offsets")
        .iter()
        .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
        .collect();
    let overflow_angle = layout["overflow_angle_offset_rad"].as_f64().unwrap();
    let overflow_radius = layout["overflow_radius_fraction"].as_f64().unwrap();
    let margin_fraction = layout["bounds_margin_fraction"].as_f64().unwrap();
    let masts = sensor_list("ground_station_mast_agls_m");
    let stds = sensor_list("ground_station_bearing_stds_rad");
    let dropouts = sensor_list("ground_station_dropout_probabilities");
    let ranges = sensor_list("ground_station_max_range_factors_m");
    let span_x = bounds.x_max_m - bounds.x_min_m;
    let span_y = bounds.y_max_m - bounds.y_min_m;
    let centre = [
        (bounds.x_min_m + bounds.x_max_m) * 0.5,
        (bounds.y_min_m + bounds.y_max_m) * 0.5,
    ];
    (0..count)
        .map(|index| {
            let nominal = if let Some([ox, oy]) = offsets.get(index) {
                [
                    centre[0] + ox * (span_x * 0.5),
                    centre[1] + oy * (span_y * 0.5),
                ]
            } else {
                let n = offsets.len();
                let angle = (index - n) as f64 * (TAU / (count - n).max(4) as f64) + overflow_angle;
                [
                    centre[0] + angle.cos() * span_x * overflow_radius,
                    centre[1] + angle.sin() * span_y * overflow_radius,
                ]
            };
            let mast = masts[index % masts.len()];
            let clamped = clamp_to_bounds(nominal, bounds, span_x.min(span_y) * margin_fraction);
            StationSpec {
                node_id: ground_station_name(index),
                xy: ground.push_outside(clamped, mast),
                mast_agl_m: mast,
                bearing_std_rad: stds[index % stds.len()],
                dropout_probability: dropouts[index % dropouts.len()],
                max_range_m: ranges[index % ranges.len()],
            }
        })
        .collect()
}

/// `_sensor_model_for_node`'s `SensorErrorConfig`, over the defaults.
fn sensor_error(is_mobile: bool, sensor_type: &str, bearing_std: f64, max_range: f64) -> Value {
    let mut error = table(&["sensor_error_defaults"]).clone();
    let clutter = if sensor_type == "radar" { 0.01 } else { 0.004 };
    let clutter = if is_mobile { clutter * 0.5 } else { clutter };
    let values = [
        ("base_bearing_std_rad", bearing_std),
        ("range_noise_exponent", 1.5),
        ("range_noise_reference_m", (max_range * 0.45).max(50.0)),
        (
            "max_bearing_std_rad",
            (bearing_std * 6.0).max(bearing_std + 0.05),
        ),
        (
            "min_detection_probability",
            if is_mobile { 0.08 } else { 0.12 },
        ),
        ("detection_range_knee_m", (max_range * 0.55).max(50.0)),
        ("detection_range_falloff_m", (max_range * 0.20).max(25.0)),
        ("false_alarm_rate_per_scan", clutter),
        (
            "clutter_bearing_std_rad",
            (bearing_std * 8.0).clamp(0.08, 0.25),
        ),
        (
            "bias_drift_rate_rad_per_s",
            if is_mobile { 2.0e-5 } else { 1.0e-5 },
        ),
        ("bias_drift_max_rad", (bearing_std * 2.0).max(0.006)),
    ];
    for (key, value) in values {
        error[key] = json!(value);
    }
    error
}

/// `_sensor_document`.
#[allow(clippy::too_many_arguments)]
fn sensor_document(
    is_mobile: bool,
    sensor_type: &str,
    max_range: f64,
    fov: f64,
    dropout: f64,
    bearing_std: f64,
) -> Value {
    json!({
        "sensor_type": sensor_type,
        "max_range_m": max_range,
        "fov_half_angle_deg": fov,
        "required_elevation_deg": if is_mobile { sensor("mobile_min_elevation_deg") } else { 0.5 },
        "dropout_probability": dropout,
        "bearing_std_rad": bearing_std,
        "look_down_angle_deg": sensor("drone_look_down_angle_deg"),
        "error": sensor_error(is_mobile, sensor_type, bearing_std, max_range),
    })
}

fn bounds_json(bounds: MapBounds) -> Value {
    json!({
        "x_min_m": bounds.x_min_m,
        "x_max_m": bounds.x_max_m,
        "y_min_m": bounds.y_min_m,
        "y_max_m": bounds.y_max_m,
    })
}

/// A zone as `generate_mission_zones` places it.
#[derive(Debug, Clone)]
struct Zone {
    zone_id: String,
    zone_type: &'static str,
    center: [f64; 3],
    radius_m: f64,
    priority: i64,
    label: String,
}

/// One axis of `np.gradient(f, coords, edge_order=1)`: numpy's uniform
/// formula when every spacing is bit-identical, its non-uniform one otherwise.
pub fn np_gradient_axis(values: &[f64], coords: &[f64]) -> Vec<f64> {
    let n = values.len();
    let diffs: Vec<f64> = coords.windows(2).map(|w| w[1] - w[0]).collect();
    let uniform = diffs.iter().all(|d| *d == diffs[0]);
    let mut out = vec![0.0; n];
    if uniform {
        let dx = diffs[0];
        for i in 1..n - 1 {
            out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dx);
        }
        out[0] = (values[1] - values[0]) / dx;
        out[n - 1] = (values[n - 1] - values[n - 2]) / dx;
    } else {
        for i in 1..n - 1 {
            let (dx1, dx2) = (diffs[i - 1], diffs[i]);
            let a = -dx2 / (dx1 * (dx1 + dx2));
            let b = (dx2 - dx1) / (dx1 * dx2);
            let c = dx1 / (dx2 * (dx1 + dx2));
            out[i] = a * values[i - 1] + b * values[i] + c * values[i + 1];
        }
        out[0] = (values[1] - values[0]) / diffs[0];
        out[n - 1] = (values[n - 1] - values[n - 2]) / diffs[n - 2];
    }
    out
}

/// `generate_mission_zones`.
fn generate_mission_zones(
    ground: &Ground<'_>,
    bounds: MapBounds,
    scale: f64,
    rng: &mut NumpyRng,
) -> Vec<Zone> {
    let (x_min, x_max, y_min, y_max) = (
        bounds.x_min_m,
        bounds.x_max_m,
        bounds.y_min_m,
        bounds.y_max_m,
    );
    let (x_span, y_span) = (x_max - x_min, y_max - y_min);
    let min_span = x_span.min(y_span);
    let edge_buffer = 0.05 * min_span;
    let count = 3usize.max((scale * 1.5) as usize);
    const TYPES: [&str; 4] = ["surveillance", "objective", "patrol", "exclusion"];
    let base_radius = min_span * 0.06;
    struct Spec {
        index: usize,
        zone_type: &'static str,
        radius_m: f64,
        priority: i64,
        label: String,
    }
    let specs: Vec<Spec> = (0..count)
        .map(|i| {
            let zone_type = TYPES[i % 4];
            Spec {
                index: i,
                zone_type,
                radius_m: rng.uniform(base_radius * 0.6, base_radius * 1.4),
                priority: match zone_type {
                    "objective" => 1,
                    "surveillance" => 2,
                    _ => 3,
                },
                label: format!("{zone_type}-{i}"),
            }
        })
        .collect();

    // Buildings (and untyped obstacles) as anchors.
    let mut buildings: Vec<([f64; 2], f64)> = Vec::new();
    for obstacle in &ground.world.obstacles {
        let m = &obstacle.metadata;
        let blocker = m["blocker_type"].as_str().unwrap_or("");
        if !blocker.is_empty() && blocker != "building" {
            continue;
        }
        let footprint: Option<Vec<[f64; 2]>> = m["footprint_xy_m"].as_array().map(|points| {
            points
                .iter()
                .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
                .collect()
        });
        let centre =
            if let (Some(x), Some(y)) = (m["center_x_m"].as_f64(), m["center_y_m"].as_f64()) {
                [x, y]
            } else if let Some(points) = &footprint {
                let mut sum = points[0];
                for p in &points[1..] {
                    sum[0] += p[0];
                    sum[1] += p[1];
                }
                [sum[0] / points.len() as f64, sum[1] / points.len() as f64]
            } else {
                continue;
            };
        let extent = if let Some(radius) = m["radius_m"].as_f64() {
            radius
        } else if let (Some(l), Some(w)) = (m["length_m"].as_f64(), m["width_m"].as_f64()) {
            0.5 * l.max(w)
        } else if let Some(points) = &footprint {
            points
                .iter()
                .map(|p| {
                    let (dx, dy) = (p[0] - centre[0], p[1] - centre[1]);
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            base_radius
        };
        buildings.push((centre, extent.max(1.0)));
    }

    // A 16 x 16 terrain sample and its gradient magnitude.
    let sx = super::linspace(x_min + edge_buffer, x_max - edge_buffer, 16);
    let sy = super::linspace(y_min + edge_buffer, y_max - edge_buffer, 16);
    let heights: Vec<Vec<f64>> = sy
        .iter()
        .map(|y| sx.iter().map(|x| ground.height_at(*x, *y)).collect())
        .collect();
    let mut gx = vec![vec![0.0; 16]; 16];
    let mut gy = vec![vec![0.0; 16]; 16];
    for row in 0..16 {
        let g = np_gradient_axis(&heights[row], &sx);
        gx[row] = g;
    }
    for col in 0..16 {
        let column: Vec<f64> = (0..16).map(|row| heights[row][col]).collect();
        let g = np_gradient_axis(&column, &sy);
        for row in 0..16 {
            gy[row][col] = g[row];
        }
    }
    let samples: Vec<([f64; 2], f64, f64)> = (0..16)
        .flat_map(|row| (0..16).map(move |col| (row, col)))
        .map(|(row, col)| {
            (
                [sx[col], sy[row]],
                heights[row][col],
                (gx[row][col] * gx[row][col] + gy[row][col] * gy[row][col]).sqrt(),
            )
        })
        .collect();
    let limit = 8usize.max(count * 4);
    let (peaks, flats) = candidate_orders(&samples, limit);

    let norm2 = |a: [f64; 2], b: [f64; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        (dx * dx + dy * dy).sqrt()
    };
    let clamped = |xy: [f64; 2], radius: f64| clamp_to_bounds(xy, bounds, radius + edge_buffer);
    let mut placed: Vec<Zone> = Vec::new();
    let mut objective_centres: Vec<[f64; 2]> = Vec::new();
    let mut by_index: BTreeMap<usize, Zone> = BTreeMap::new();

    for pass in [["objective", "exclusion"], ["surveillance", "patrol"]] {
        for spec in &specs {
            if !pass.contains(&spec.zone_type) {
                continue;
            }
            let radius = spec.radius_m;
            let separated = |xy: [f64; 2], placed: &[Zone]| {
                placed
                    .iter()
                    .all(|z| norm2(xy, [z.center[0], z.center[1]]) >= 0.3 * (radius + z.radius_m))
            };
            let relative = |candidates: &[[f64; 2]], objectives: &[[f64; 2]]| -> Vec<[f64; 2]> {
                if objectives.is_empty() {
                    return candidates.to_vec();
                }
                let related: Vec<[f64; 2]> = candidates
                    .iter()
                    .copied()
                    .filter(|c| {
                        let nearest = objectives
                            .iter()
                            .map(|o| norm2(*c, *o))
                            .fold(f64::INFINITY, f64::min);
                        2.0 * radius <= nearest && nearest <= 3.0 * radius
                    })
                    .collect();
                if related.is_empty() {
                    candidates.to_vec()
                } else {
                    related
                }
            };
            let sampler = |rng: &mut NumpyRng| -> Option<[f64; 2]> {
                match spec.zone_type {
                    "objective" | "exclusion" => {
                        if buildings.is_empty() {
                            return None;
                        }
                        let index = rng.integers(0, buildings.len() as i64) as usize;
                        let (anchor, extent) = buildings[index];
                        let angle = rng.uniform(0.0, TAU);
                        let offset = if spec.zone_type == "objective" {
                            1.5 * extent + radius + rng.uniform(0.0, radius.max(0.35 * extent))
                        } else {
                            extent + radius + rng.uniform(0.0, radius)
                        };
                        let direction = [angle.cos(), angle.sin()];
                        Some([
                            anchor[0] + direction[0] * offset,
                            anchor[1] + direction[1] * offset,
                        ])
                    }
                    kind => {
                        let (pool, band) = if kind == "surveillance" {
                            (relative(&peaks, &objective_centres), 6)
                        } else {
                            (relative(&flats, &objective_centres), 8)
                        };
                        if pool.is_empty() {
                            return None;
                        }
                        let top = pool.len().min(band);
                        let anchor = pool[rng.integers(0, top as i64) as usize];
                        let spread = if kind == "surveillance" { 0.25 } else { 0.35 };
                        let draw = [rng.uniform(-spread, spread), rng.uniform(-spread, spread)];
                        Some([anchor[0] + draw[0] * radius, anchor[1] + draw[1] * radius])
                    }
                }
            };
            let random_centre = |rng: &mut NumpyRng| {
                let cx = rng.uniform(x_min + 0.20 * x_span, x_max - 0.20 * x_span);
                let cy = rng.uniform(y_min + 0.20 * y_span, y_max - 0.20 * y_span);
                clamped([cx, cy], radius)
            };
            let mut centre = None;
            for _ in 0..20 {
                let Some(candidate) = sampler(rng) else {
                    break;
                };
                let candidate = clamped(candidate, radius);
                if separated(candidate, &placed) {
                    centre = Some(candidate);
                    break;
                }
            }
            if centre.is_none() {
                for _ in 0..20 {
                    let candidate = random_centre(rng);
                    if separated(candidate, &placed) {
                        centre = Some(candidate);
                        break;
                    }
                }
            }
            let centre = centre.unwrap_or_else(|| random_centre(rng));
            let [cx, cy] = clamped(centre, radius);
            let zone = Zone {
                zone_id: format!("zone-{}", spec.index),
                zone_type: spec.zone_type,
                center: [cx, cy, ground.height_at(cx, cy)],
                radius_m: radius,
                priority: spec.priority,
                label: spec.label.clone(),
            };
            if zone.zone_type == "objective" {
                objective_centres.push([cx, cy]);
            }
            placed.push(zone.clone());
            by_index.insert(spec.index, zone);
        }
    }
    (0..count)
        .map(|i| by_index.remove(&i).expect("every zone placed"))
        .collect()
}

/// A drone before its controller is chosen.
struct DroneSpec {
    node_id: String,
    bearing_std_rad: f64,
    dropout_probability: f64,
    max_range_m: f64,
    base_agl_m: f64,
    vertical_amplitude_m: f64,
    vertical_frequency_rad_s: f64,
    search_speed_mps: f64,
}

fn looped_path_document(
    waypoints: &[[f64; 2]],
    speed: f64,
    drone: &DroneSpec,
    phase: f64,
    min_agl_m: f64,
) -> Value {
    json!({
        "waypoints_xy_m": waypoints.iter().map(|p| vec![p[0], p[1]]).collect::<Vec<_>>(),
        "speed_mps": speed,
        "base_agl_m": drone.base_agl_m,
        "vertical_amplitude_m": drone.vertical_amplitude_m,
        "vertical_frequency_rad_s": drone.vertical_frequency_rad_s,
        "phase": phase,
        "min_agl_m": min_agl_m,
        "avoid_obstacles": true,
    })
}

/// The inputs a bundle needs that are not in the config.
pub struct Assembly<'a> {
    pub config: &'a ScenarioConfig,
    /// The tracker configuration, as `TrackerConfig`'s fields.
    pub tracker: Value,
}

/// `build_default_scenario` + `export_bundle`: the whole bundle.
pub fn build_bundle(assembly: Assembly<'_>, world: &BuiltWorld) -> Result<Value, BuildError> {
    let config = assembly.config;
    let options = &config.options;
    refuse_unsupported(options)?;
    let mission_profile = mission_profile(options)?;
    let seed = config.seed;
    let scale = world.scale;
    let bounds = world.bounds;
    let platform = table(&["platform_presets", &options.platform_preset]);
    if platform.is_null() {
        return Err(BuildError::Invalid(format!(
            "unknown platform preset {:?}",
            options.platform_preset
        )));
    }
    let profile = |key: &str| platform[key].as_f64().expect("platform table");
    let weather_doc = table(&["weather_presets"])
        .get(&options.weather_preset)
        .cloned()
        .ok_or_else(|| {
            BuildError::Invalid(format!(
                "unknown weather preset {:?}",
                options.weather_preset
            ))
        })?;
    let weather: MotionWeather = serde_json::from_value(weather_doc.clone())
        .map_err(|e| BuildError::Invalid(format!("weather table: {e}")))?;

    let obstacles: Vec<Obstacle> = world
        .obstacles
        .iter()
        .map(|o| {
            serde_json::from_value::<ObstacleMetadata>(o.metadata.clone())
                .map_err(|e| BuildError::Invalid(e.to_string()))?
                .into_obstacle()
                .map_err(|e| BuildError::Invalid(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let terrain = &world.terrain.tiled;
    let layer = ObstacleLayer::new(
        terrain.bounds,
        terrain.tile_size_cells as f64 * terrain.base_resolution_m,
        obstacles,
    )
    .map_err(BuildError::Invalid)?;
    let ground = Ground {
        world,
        terrain: SharedTerrain::new(Arc::new(terrain.clone()), terrain.base_resolution_m),
        layer,
    };
    let mut rng = NumpyRng::new(seed);

    // --- targets ---------------------------------------------------------
    const TARGET_NAMES: [&str; 16] = [
        "lynx", "orca", "fox", "hawk", "viper", "cobra", "eagle", "raven", "wolf", "bear", "puma",
        "falcon", "shark", "stag", "crane", "pike",
    ];
    let target_count = options.target_count as usize;
    let target_ids: Vec<String> = (0..target_count)
        .map(|i| {
            if i < TARGET_NAMES.len() {
                format!("asset-{}", TARGET_NAMES[i % TARGET_NAMES.len()])
            } else {
                format!("asset-{i}")
            }
        })
        .collect();
    let centre = [
        0.5 * (bounds.x_min_m + bounds.x_max_m),
        0.5 * (bounds.y_min_m + bounds.y_max_m),
    ];
    let centre_height = ground.height_at(centre[0], centre[1]);
    let speed_of = |motion: &str| match motion {
        "sinusoid" => 7.0,
        "racetrack" => 7.2,
        "waypoint_patrol" => 6.8,
        "loiter" => 18.0,
        "transit" => 24.0,
        "evasive" => 22.0,
        "search_pattern" => 16.0,
        "mixed" => 20.0,
        _ => 18.0,
    };
    let behavior_bounds = Bounds {
        x_min_m: bounds.x_min_m,
        x_max_m: bounds.x_max_m,
        y_min_m: bounds.y_min_m,
        y_max_m: bounds.y_max_m,
    };
    let mut targets = Vec::new();
    for (index, target_id) in target_ids.iter().enumerate() {
        let motion = options.target_motion_preset.clone();
        let altitude =
            centre_height + (dynamics("aerial_target_min_agl_m") + 20.0 * (index % 4) as f64);
        let speed = speed_of(&motion) * profile("target_speed_scale");
        let target_seed = stable_seed(
            seed,
            &[
                SeedPart::Str("target"),
                SeedPart::Str(target_id),
                SeedPart::Str(&motion),
                SeedPart::Int(index as i64),
            ],
        );
        let behavior =
            build_target_behavior(&motion, behavior_bounds, altitude, speed, target_seed)
                .map_err(BuildError::Invalid)?;
        targets.push((target_id.clone(), motion, behavior));
    }

    // --- ground stations -----------------------------------------------
    let station_specs =
        ground_station_specs(&ground, bounds, options.ground_station_count as usize);
    if station_specs.is_empty() {
        return Err(BuildError::Unsupported(
            "a scenario with no ground stations".into(),
        ));
    }
    let station_position = |spec: &StationSpec| {
        [
            spec.xy[0],
            spec.xy[1],
            ground.height_at(spec.xy[0], spec.xy[1]) + spec.mast_agl_m,
        ]
    };
    let mut nodes: Vec<Value> = station_specs
        .iter()
        .map(|spec| {
            let bearing = spec.bearing_std_rad * sensor("ground_station_bearing_std_rad_scale");
            let dropout =
                spec.dropout_probability * sensor("ground_station_dropout_probability_scale");
            let max_range = spec.max_range_m * profile("ground_station_range_scale");
            json!({
                "node_id": spec.node_id,
                "is_mobile": false,
                "health": 1.0 - dropout * 0.5,
                "sensor": sensor_document(
                    false,
                    "radar",
                    max_range,
                    sensor("ground_station_fov_half_angle_deg"),
                    dropout,
                    bearing,
                ),
                "controller": {"kind": "static", "position_m": station_position(spec)},
            })
        })
        .collect();

    // --- drones ------------------------------------------------------------
    const COMPASS: [&str; 8] = ["east", "north", "west", "south", "ne", "nw", "se", "sw"];
    let drone_count = options.drone_count as usize;
    let drones: Vec<DroneSpec> = (0..drone_count)
        .map(|di| {
            let name = if di < COMPASS.len() {
                COMPASS[di % COMPASS.len()].to_string()
            } else {
                di.to_string()
            };
            DroneSpec {
                node_id: format!("drone-{name}"),
                bearing_std_rad: sensor("drone_base_bearing_std_rad")
                    + sensor("drone_bearing_std_rad_increment") * (di % 4) as f64,
                dropout_probability: sensor("drone_base_dropout_probability")
                    + sensor("drone_dropout_probability_increment") * (di % 3) as f64,
                max_range_m: sensor("drone_base_max_range_m") * profile("drone_max_range_scale"),
                base_agl_m: dynamics("drone_base_agl_m")
                    + dynamics("drone_agl_increment_m") * (di % 4) as f64,
                vertical_amplitude_m: dynamics("drone_vertical_amplitude_base_m")
                    + dynamics("drone_vertical_amplitude_increment_m") * (di % 3) as f64,
                vertical_frequency_rad_s: dynamics("drone_vertical_frequency_base_rad_s")
                    - dynamics("drone_vertical_frequency_decrement_rad_s") * (di % 3) as f64,
                search_speed_mps: (dynamics("drone_search_speed_base_mps")
                    - dynamics("drone_search_speed_decrement_mps") * (di % 3) as f64)
                    * profile("drone_search_speed_scale"),
            }
        })
        .collect();
    let modes = drone_mode_sequence(&options.drone_mode_preset, drones.len());
    let lane_spacing = dynamics("drone_search_lane_spacing_scale") * scale;
    let search_min_agl = dynamics("interceptor_search_min_agl_m");
    let min_operational_agl = search_min_agl.min(dynamics("interceptor_follow_min_agl_m"));
    let mut planner = RoutePlanner::new(terrain.bounds, PlannerConfig::default());
    let mut planner_modes = Map::new();
    let mut drone_targets = Map::new();
    let mut drone_roles = Map::new();
    let mut leads: Vec<f64> = Vec::new();

    let launched = |station: [f64; 3], drone: &DroneSpec, target: &str, operational: Value| {
        json!({
            "kind": "launched",
            "station_position_m": station,
            "climb_duration_s": dynamics("launch_climb_duration_s"),
            "operational_altitude_agl_m": drone.base_agl_m,
            "min_operational_agl_m": min_operational_agl,
            "wind_scale": dynamics("wind_drift_scale"),
            "auto_launch_at_s": 0.0,
            "launch_target_id": target,
            "operational": operational,
        })
    };
    let drone_node = |drone: &DroneSpec, controller: Value| {
        json!({
            "node_id": drone.node_id,
            "is_mobile": true,
            "health": 1.0 - drone.dropout_probability * 0.5,
            "sensor": sensor_document(
                true,
                "optical",
                drone.max_range_m,
                sensor("drone_fov_half_angle_deg"),
                drone.dropout_probability,
                drone.bearing_std_rad,
            ),
            "controller": controller,
        })
    };

    if options.mission_mode == "target_tracking" {
        let search_indices: Vec<usize> = (0..drones.len())
            .filter(|i| modes[*i] == "search")
            .collect();
        let sectors = split_bounds_along_x(bounds, search_indices.len());
        let sector_of: BTreeMap<usize, MapBounds> = search_indices
            .iter()
            .enumerate()
            .map(|(k, i)| (*i, sectors[k]))
            .collect();
        let roles: Vec<&str> = modes
            .iter()
            .map(|m| {
                if m == "inspect" || m == "follow" {
                    "interceptor"
                } else {
                    "tracker"
                }
            })
            .collect();
        let mut totals: BTreeMap<(&str, String), usize> = BTreeMap::new();
        if !target_ids.is_empty() {
            for i in 0..drones.len() {
                *totals
                    .entry((roles[i], target_ids[i % target_ids.len()].clone()))
                    .or_default() += 1;
            }
        }
        let mut slots: BTreeMap<(&str, String), usize> = BTreeMap::new();
        for (index, drone) in drones.iter().enumerate() {
            let mode = modes[index].as_str();
            let target_id = if target_ids.is_empty() {
                String::new()
            } else {
                target_ids[index % target_ids.len()].clone()
            };
            if target_ids.is_empty() {
                return Err(BuildError::Unsupported(
                    "target tracking with no targets (the reference indexes an empty target map)"
                        .into(),
                ));
            }
            let role = roles[index];
            planner_modes.insert(
                drone.node_id.clone(),
                json!(if mode == "inspect" || mode == "follow" {
                    "follow"
                } else {
                    mode
                }),
            );
            drone_targets.insert(drone.node_id.clone(), json!(target_id));
            drone_roles.insert(drone.node_id.clone(), json!(role));
            let key = (role, target_id.clone());
            let slot_index = *slots.get(&key).unwrap_or(&0);
            slots.insert(key.clone(), slot_index + 1);
            let slot_count = *totals.get(&key).unwrap_or(&1);
            let (standoff, alt_offset, min_agl, rotation, lead, candidates) = if role == "tracker" {
                (
                    dynamics("tracker_standoff_radius_m"),
                    dynamics("tracker_altitude_offset_m"),
                    dynamics("tracker_follow_min_agl_m"),
                    dynamics("tracker_rotation_rate_rad_s"),
                    dynamics("tracker_lead_s"),
                    dynamics("tracker_candidate_count"),
                )
            } else {
                (
                    dynamics("interceptor_follow_radius_m"),
                    dynamics("interceptor_follow_altitude_offset_m"),
                    dynamics("interceptor_follow_min_agl_m"),
                    dynamics("interceptor_follow_rotation_rate_rad_s"),
                    dynamics("interceptor_follow_lead_s"),
                    dynamics("interceptor_follow_candidate_count"),
                )
            };
            let station = station_position(&station_specs[index % station_specs.len()]);
            let follow = |phase: f64| {
                json!({
                    "map_bounds_m": bounds_json(bounds),
                    "base_agl_m": drone.base_agl_m,
                    "standoff_radius_m": standoff,
                    "map_scale": scale,
                    "max_speed_mps": profile("follow_speed_cap_mps"),
                    "phase": phase,
                    "lead_s": lead,
                    "candidate_count": candidates as i64,
                    "rotation_rate_rad_s": rotation,
                    "min_agl_m": min_agl,
                    "vertical_amplitude_m": drone.vertical_amplitude_m,
                    "vertical_frequency_rad_s": drone.vertical_frequency_rad_s,
                    "target_altitude_offset_m": alt_offset,
                    "max_accel_mps2": dynamics("drone_max_accel_mps2"),
                    "terrain_following": false,
                    "terrain_following_agl_m": 30.0,
                    "terrain_following_smoothing_s": 1.5,
                    "slot_index": slot_index,
                    "slot_count": slot_count,
                    "mission_zones": [],
                })
            };
            leads.push(lead);
            let operational = if mode == "inspect" || mode == "follow" {
                let phase = rng.uniform(0.0, TAU);
                json!({"kind": "follow", "target_id": target_id, "follow": follow(phase)})
            } else {
                let sector = sector_of[&index];
                let lawnmower = lawnmower_waypoints(sector, lane_spacing, 0.2);
                let scene = Scene {
                    layer: &ground.layer,
                    revision: 0,
                };
                let clearance = planner.config().drone_clearance_m;
                let route = planner
                    .route_waypoints(scene, &lawnmower, clearance, true)
                    .or_else(|| {
                        planner.route_waypoints(
                            Scene {
                                layer: &ground.layer,
                                revision: 0,
                            },
                            &perimeter_waypoints(sector, 0.18),
                            clearance,
                            true,
                        )
                    });
                let waypoints = match route {
                    Some(route) => route.points_xy_m,
                    None => lawnmower
                        .iter()
                        .map(|p| ground.push_outside(*p, drone.base_agl_m))
                        .collect(),
                };
                let search_phase = rng.uniform(0.0, TAU);
                let follow_phase = rng.uniform(0.0, TAU);
                json!({
                    "kind": "search_then_follow",
                    "preferred_target_id": target_id,
                    "search": looped_path_document(
                        &waypoints,
                        drone.search_speed_mps,
                        drone,
                        search_phase,
                        search_min_agl,
                    ),
                    "follow": follow(follow_phase),
                    "follow_target_id": target_id,
                })
            };
            nodes.push(drone_node(
                drone,
                launched(station, drone, &target_id, operational),
            ));
        }
    } else {
        if options.cooperative_search {
            return Err(BuildError::Unsupported(
                "cooperative search in the native builder (P4.9b.2c)".into(),
            ));
        }
        let sectors = split_bounds_along_x(bounds, drones.len());
        let mut by_station_x: Vec<(usize, f64)> = (0..drones.len())
            .map(|i| {
                (
                    i,
                    station_position(&station_specs[i % station_specs.len()])[0],
                )
            })
            .collect();
        by_station_x.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
        let sector_of: BTreeMap<usize, usize> = by_station_x
            .iter()
            .enumerate()
            .map(|(sector, (drone, _))| (*drone, sector))
            .collect();
        for (index, drone) in drones.iter().enumerate() {
            planner_modes.insert(drone.node_id.clone(), json!("search"));
            drone_targets.insert(drone.node_id.clone(), json!(""));
            drone_roles.insert(drone.node_id.clone(), json!("tracker"));
            let station = station_position(&station_specs[index % station_specs.len()]);
            let sector = sectors[sector_of[&index]];
            let waypoints: Vec<[f64; 2]> = lawnmower_waypoints(sector, lane_spacing, 0.0)
                .into_iter()
                .map(|p| ground.push_outside(p, drone.base_agl_m))
                .collect();
            let phase = rng.uniform(0.0, TAU);
            let operational = json!({
                "kind": "search",
                "search": looped_path_document(&waypoints, drone.search_speed_mps, drone, phase, search_min_agl),
            });
            nodes.push(drone_node(drone, launched(station, drone, "", operational)));
        }
    }

    // --- zones, then everything that reads them --------------------------
    // A profile's zones replace the generated ones, which then draw nothing:
    // they are the last thing the reference's generator draws for, so
    // skipping them moves no other value.
    let zones = match mission_profile {
        Some(profile) => profile_zones(&ground, profile, map_extent_m(bounds)),
        None => generate_mission_zones(&ground, bounds, scale, &mut rng),
    };
    let zone_documents: Vec<Value> = zones
        .iter()
        .map(|z| {
            json!({
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "center_m": z.center,
                "radius_m": z.radius_m,
                "priority": z.priority,
                "label": z.label,
            })
        })
        .collect();

    // --- run length ------------------------------------------------------
    let (steps, dt_s, requested_duration_s) = run_length(config)?;

    // --- target truth, sampled where the tick will ask -------------------
    let sample_times = target_sample_times(steps, dt_s, &leads);
    let terrain_fn = |x: f64, y: f64| ground.height_at(x, y);
    let context = WeatherContext {
        weather: Some(&weather),
        terrain: Some(&ground.terrain),
        bounds: Some(BoundsXy {
            x_min_m: bounds.x_min_m,
            x_max_m: bounds.x_max_m,
            y_min_m: bounds.y_min_m,
            y_max_m: bounds.y_max_m,
        }),
        min_agl_m: dynamics("aerial_target_min_agl_m"),
        wind_scale: dynamics("wind_drift_scale"),
        dt_s: 1.0,
    };
    let target_documents: Vec<Value> = targets
        .iter()
        .map(|(target_id, motion, behavior)| {
            let samples: Vec<Value> = sample_times
                .iter()
                .map(|t| {
                    let (p, v) = behavior.sample(*t, &terrain_fn);
                    let (p, v) = apply_weather_adjustments(p, v, *t, &context);
                    json!({"t": t, "position": p, "velocity": v})
                })
                .collect();
            json!({"target_id": target_id, "behavior": motion, "samples": samples})
        })
        .collect();

    let mission = if options.mission_mode == "scan_map_inspect" {
        Some(mission_document(
            &ground,
            bounds,
            options,
            seed,
            mission_profile,
        )?)
    } else if options.power_line_inspection {
        return Err(BuildError::Invalid(
            "power_line_inspection needs mission_mode scan_map_inspect".into(),
        ));
    } else {
        None
    };

    // --- presentation ----------------------------------------------------
    let mut presentation = world.presentation_world();
    presentation.insert(
        "scenario_options".into(),
        scenario_options_document(options),
    );
    presentation.insert(
        "mission_profile".into(),
        match &options.mission_profile_id {
            Some(id) if mission_profile.is_some() => json!({
                "profile_id": id,
                "domain": options.mission_domain,
                "workflow": options.mission_workflow,
                "tags": options.mission_tags,
            }),
            _ => Value::Null,
        },
    );
    presentation.insert(
        "zones".into(),
        json!(zones
            .iter()
            .map(|z| json!({
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "center": z.center,
                "radius_m": z.radius_m,
                "priority": z.priority,
                "label": z.label,
            }))
            .collect::<Vec<_>>()),
    );
    let motions: Map<String, Value> = targets
        .iter()
        .map(|(id, motion, _)| (id.clone(), json!(motion)))
        .collect();
    presentation.insert("target_motion_assignments".into(), Value::Object(motions));
    presentation.insert("drone_planner_modes".into(), Value::Object(planner_modes));
    presentation.insert(
        "drone_target_assignments".into(),
        Value::Object(drone_targets),
    );
    presentation.insert("drone_roles".into(), Value::Object(drone_roles.clone()));

    let scenario_name = match (&options.mission_profile_id, mission_profile) {
        (Some(id), Some(_)) => id.clone(),
        _ => format!(
            "civilian-observation-{}-{}-{}",
            options.map_preset, options.target_motion_preset, options.drone_mode_preset
        ),
    };
    let mut bundle = json!({
        "format": crate::scenario::BUNDLE_FORMAT,
        "version": crate::scenario::BUNDLE_VERSION,
        "scenario_name": scenario_name,
        "mission_mode": options.mission_mode,
        "seed": seed,
        "dt_s": dt_s,
        "steps": steps,
        "map_bounds_m": bounds_json(bounds),
        "coverage_bounds": {
            "x_min_m": bounds.x_min_m,
            "x_max_m": bounds.x_max_m,
            "y_min_m": bounds.y_min_m,
            "y_max_m": bounds.y_max_m,
            "resolution_m": options.coverage_resolution_m,
        },
        "cooperative_search": options.cooperative_search,
        "requested_duration_s": requested_duration_s,
        "world": world.world_document(),
        "weather": weather_doc,
        "tracker": assembly.tracker,
        "drone_roles": drone_roles,
        "mission_zones": zone_documents,
        "mission": mission,
        "nodes": nodes,
        "targets": target_documents,
        "presentation": presentation,
    });
    // Written only when set, as `export_bundle` writes it.
    if options.safety_blocking {
        bundle["safety_blocking"] = json!(true);
    }
    Ok(bundle)
}

/// The mission profile `options` names, checked as `ScenarioOptions` checks
/// it on construction: a known id, and the domain, workflow and mission mode
/// that profile declares. `None` when no profile is named.
fn mission_profile(options: &ScenarioOptions) -> Result<Option<&'static Value>, BuildError> {
    let Some(id) = options
        .mission_profile_id
        .as_deref()
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let profile = &TABLES["mission_profiles"][id];
    if profile.is_null() {
        return Err(BuildError::Invalid(format!(
            "unknown mission profile {id:?}"
        )));
    }
    let declared = |key: &str| profile[key].as_str();
    if options.mission_domain.as_deref() != declared("domain") {
        return Err(BuildError::Invalid(
            "mission_domain does not match mission_profile_id".into(),
        ));
    }
    if options.mission_workflow.as_deref() != declared("workflow") {
        return Err(BuildError::Invalid(
            "mission_workflow does not match mission_profile_id".into(),
        ));
    }
    if Some(options.mission_mode.as_str()) != declared("mission_mode") {
        return Err(BuildError::Invalid(
            "mission_mode does not match mission_profile_id".into(),
        ));
    }
    Ok(Some(profile))
}

/// `min(x span, y span)`, the extent a profile's zones and POIs scale by.
fn map_extent_m(bounds: MapBounds) -> f64 {
    (bounds.x_max_m - bounds.x_min_m).min(bounds.y_max_m - bounds.y_min_m)
}

/// `build_profile_mission_zones`, lifted onto the terrain as the reference
/// lifts them.
fn profile_zones(ground: &Ground<'_>, profile: &Value, extent: f64) -> Vec<Zone> {
    profile["zones"]
        .as_array()
        .expect("profile zones")
        .iter()
        .map(|zone| {
            let x = extent * zone["center"][0].as_f64().expect("x");
            let y = extent * zone["center"][1].as_f64().expect("y");
            Zone {
                zone_id: zone["zone_id"].as_str().expect("id").to_string(),
                zone_type: match zone["zone_type"].as_str().expect("type") {
                    "objective" => "objective",
                    "exclusion" => "exclusion",
                    "surveillance" => "surveillance",
                    "patrol" => "patrol",
                    other => panic!("profile zone type {other:?} the builder does not know"),
                },
                center: [x, y, ground.height_at(x, y)],
                radius_m: extent * zone["radius_m"].as_f64().expect("radius"),
                priority: zone["priority"].as_i64().expect("priority"),
                label: zone["label"].as_str().expect("label").to_string(),
            }
        })
        .collect()
}

/// `export_bundle`'s refusals, and the native builder's own.
fn refuse_unsupported(options: &ScenarioOptions) -> Result<(), BuildError> {
    let refusals = [
        (options.enforce_flight_envelope, "enforce_flight_envelope"),
        (options.frontier_exploration, "frontier_exploration"),
        (options.occlusion_aware_mapping, "occlusion_aware_mapping"),
        (options.adaptive_search, "adaptive_search"),
    ];
    for (used, name) in refusals {
        if used {
            return Err(BuildError::Unsupported(name.into()));
        }
    }
    if !matches!(
        options.mission_mode.as_str(),
        "target_tracking" | "scan_map_inspect"
    ) {
        return Err(BuildError::Unsupported(format!(
            "mission_mode {:?}",
            options.mission_mode
        )));
    }
    Ok(())
}

/// `SimulationConfig`: `steps` as given, or `ceil(duration / dt) + 1`.
fn run_length(config: &ScenarioConfig) -> Result<(u64, f64, f64), BuildError> {
    let dt = config.dt_s;
    if !(dt.is_finite() && dt > 0.0) {
        return Err(BuildError::Invalid(
            "dt_s must be finite and positive".into(),
        ));
    }
    match (config.steps, config.duration_s) {
        (Some(_), Some(_)) => Err(BuildError::Invalid(
            "give steps or duration_s, not both".into(),
        )),
        (Some(steps), None) => {
            // `requested_duration_s` falls back to the actual duration.
            Ok((steps, dt, ((steps.saturating_sub(1)) as f64 * dt).max(0.0)))
        }
        (None, duration) => {
            let duration = duration.unwrap_or_else(|| dynamics("default_duration_s"));
            if !(duration.is_finite() && duration > 0.0) {
                return Err(BuildError::Invalid(
                    "duration_s must be finite and positive".into(),
                ));
            }
            Ok(((duration / dt).ceil() as u64 + 1, dt, duration))
        }
    }
}

/// `_target_sample_times`: every tick's timestamp, and each follow lead past
/// it, by the same single addition, sorted and deduplicated.
fn target_sample_times(steps: u64, dt_s: f64, leads: &[f64]) -> Vec<f64> {
    let mut times: Vec<f64> = Vec::new();
    for step in 0..steps {
        let t = step as f64 * dt_s;
        times.push(t);
        for lead in leads {
            times.push(t + lead);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    times
}

/// `_mission_document`, with `build_run_inspection_pois`' POIs.
fn mission_document(
    ground: &Ground<'_>,
    bounds: MapBounds,
    options: &ScenarioOptions,
    seed: u32,
    mission_profile: Option<&Value>,
) -> Result<Value, BuildError> {
    let count = options.poi_count as usize;
    let hover = number(&["poi_hover_agl_m"]);
    let pois = match mission_profile {
        _ if options.power_line_inspection => {
            if mission_profile.is_some() {
                return Err(BuildError::Invalid(
                    "power_line_inspection places its own POIs; it takes no mission profile"
                        .into(),
                ));
            }
            let Some(line) = ground.world.power_lines.first() else {
                return Err(BuildError::Invalid(
                    "power_line_inspection needs the power_lines option".into(),
                ));
            };
            line_inspection_pois(ground, line, count)
        }
        Some(profile) => profile_pois(ground, profile, map_extent_m(bounds), count, hover),
        None => generated_pois(ground, bounds, count, hover, seed),
    };
    let localization = table(&["localization"]);
    let (x_min, x_max, y_min, y_max) = (
        bounds.x_min_m,
        bounds.x_max_m,
        bounds.y_min_m,
        bounds.y_max_m,
    );
    Ok(json!({
        "coverage_bounds": {
            "x_min_m": x_min,
            "x_max_m": x_max,
            "y_min_m": y_min,
            "y_max_m": y_max,
            "resolution_m": options.coverage_resolution_m,
        },
        "scan_coverage_threshold": options.scan_coverage_threshold,
        "localization_confidence_threshold": 0.65,
        "egress_arrival_radius_m": 15.0,
        "redirect_speed_mps": 28.0,
        "gap_fill_min_fraction": number(&["gap_fill_min_fraction"]),
        "localization": {
            "search_radius_m": localization["search_radius_m"],
            "particle_count": localization["particle_count"],
            "convergence_threshold_m": localization["convergence_threshold_m"],
            "min_coverage_to_localize": localization["min_coverage_to_localize"],
            "confidence_decay": localization["confidence_decay"],
            "annealing_rate": localization["annealing_rate"],
            "localization_timeout_steps": localization["localization_timeout_steps"],
        },
        "pois": pois,
    }))
}

/// `build_profile_inspection_pois`, the first `poi_count` of them, at the
/// hover height over the terrain.
fn profile_pois(
    ground: &Ground<'_>,
    profile: &Value,
    extent: f64,
    count: usize,
    hover: f64,
) -> Vec<Value> {
    profile["pois"]
        .as_array()
        .expect("profile pois")
        .iter()
        .take(count)
        .map(|poi| {
            let x = extent * poi["position"][0].as_f64().expect("x");
            let y = extent * poi["position"][1].as_f64().expect("y");
            json!({
                "poi_id": poi["poi_id"],
                "name": poi["name"],
                "position": [x, y, ground.height_at(x, y) + hover],
                "priority": poi["priority"],
                "required_dwell_s": poi["required_dwell_s"],
            })
        })
        .collect()
}

/// `make_poi_positions` and the POIs `run_simulation` builds on them.
fn generated_pois(
    ground: &Ground<'_>,
    bounds: MapBounds,
    count: usize,
    hover: f64,
    seed: u32,
) -> Vec<Value> {
    let mut rng = NumpyRng::new(seed + 999);
    let (x_min, x_max, y_min, y_max) = (
        bounds.x_min_m,
        bounds.x_max_m,
        bounds.y_min_m,
        bounds.y_max_m,
    );
    let margin_x = (x_max - x_min) * 0.15;
    let margin_y = (y_max - y_min) * 0.15;
    let mut pois = Vec::new();
    for i in 0..count {
        let mut chosen = None;
        let mut last = [0.0; 3];
        for _ in 0..200 {
            let x = rng.uniform(x_min + margin_x, x_max - margin_x);
            let y = rng.uniform(y_min + margin_y, y_max - margin_y);
            let z = ground.height_at(x, y) + hover;
            last = [x, y, z];
            if ground.layer.point_collides([x, y, z]).is_some() {
                continue;
            }
            chosen = Some(last);
            break;
        }
        let position = chosen.unwrap_or(last);
        pois.push(json!({
            "poi_id": format!("poi-{i:02}"),
            "name": format!("Site {}", char::from(b'A' + (i % 26) as u8)),
            "position": position,
            "priority": (count - i) as i64,
            "required_dwell_s": 15.0,
        }));
    }
    pois
}

/// How far from the line's centre a `power_line_inspection` POI stands,
/// metres: clear of the outer conductors' 10 m clearance, close enough to
/// see the hardware.
const LINE_INSPECTION_OFFSET_M: f64 = 35.0;
/// How high over the ground a `power_line_inspection` POI hovers, metres:
/// a few metres over the conductors, which hang 24–30 m up on the builder's
/// line, so a flight between POIs at this height is well inside their
/// 10 m clearance. Like the builder's other POIs (40 m), it is below a
/// tracker's 50 m safety floor (`min_agl_m`).
const LINE_INSPECTION_HOVER_AGL_M: f64 = 32.0;

/// `power_line_inspection`'s POIs: `count` of them spaced evenly along the
/// line from its first support to its last, each beside it just over the
/// wires, the first to its left and then alternately right and left, so
/// neighbouring POIs are on opposite sides of the wires. Laid out from the
/// line alone, so the same for every seed. A POI that would stand in a
/// building is pushed out of it.
fn line_inspection_pois(ground: &Ground<'_>, line: &PowerLine, count: usize) -> Vec<Value> {
    let (first, last) = (
        line.supports[0].xy_m,
        line.supports[line.supports.len() - 1].xy_m,
    );
    let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
    let length = dx.hypot(dy);
    let left = [-dy / length, dx / length];
    (0..count)
        .map(|i| {
            let t = (i as f64 + 0.5) / count as f64;
            let side = if i % 2 == 0 { 1.0 } else { -1.0 };
            let beside = [
                first[0] + dx * t + side * LINE_INSPECTION_OFFSET_M * left[0],
                first[1] + dy * t + side * LINE_INSPECTION_OFFSET_M * left[1],
            ];
            let [x, y] = ground.push_outside(beside, LINE_INSPECTION_HOVER_AGL_M);
            json!({
                "poi_id": format!("poi-{i:02}"),
                "name": format!("Span {}", i + 1),
                "position": [x, y, ground.height_at(x, y) + LINE_INSPECTION_HOVER_AGL_M],
                "priority": (count - i) as i64,
                "required_dwell_s": 15.0,
            })
        })
        .collect()
}

/// `scenario_replay_metadata`'s `scenario_options`.
fn scenario_options_document(o: &ScenarioOptions) -> Value {
    let mut document = json!({
        "map_preset": o.map_preset,
        "target_motion_preset": o.target_motion_preset,
        "drone_mode_preset": o.drone_mode_preset,
        "terrain_source": o.terrain_source,
        "terrain_preset": o.terrain_preset,
        "terrain_seed": o.terrain_seed,
        "dem_path": o.dem_path,
        "dem_crs": o.dem_crs,
        "detail_strength": o.detail_strength,
        "terrain_resolution_m": o.terrain_resolution_m,
        "coverage_resolution_m": o.coverage_resolution_m,
        "cooperative_search": o.cooperative_search,
        "adaptive_search": o.adaptive_search,
        "occlusion_aware_mapping": o.occlusion_aware_mapping,
        "search_origin_m": o.search_origin_m,
        "season_month": o.season_month,
        "weather_preset": o.weather_preset,
        "clean_terrain": o.clean_terrain,
        "platform_preset": o.platform_preset,
        "ground_station_count": o.ground_station_count,
        "target_count": o.target_count,
        "drone_count": o.drone_count,
        "mission_profile_id": o.mission_profile_id,
        "mission_domain": o.mission_domain,
        "mission_workflow": o.mission_workflow,
        "mission_tags": o.mission_tags,
    });
    // Only when set, so every other run's replay meta is unchanged.
    if o.power_lines {
        document["power_lines"] = json!(true);
    }
    if o.power_line_inspection {
        document["power_line_inspection"] = json!(true);
    }
    document
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.arange` fills with the rounded difference of its first two values,
    /// not `step`. Inputs chosen (from numpy 2.5) where that difference is not
    /// `step`, a naive `start + i * step` differs, and fused and unfused
    /// arithmetic agree -- so the expectation holds on every platform.
    #[test]
    fn np_arange_fills_from_the_rounded_first_difference() {
        let cases: [(f64, f64, f64, &[u64]); 2] = [
            (
                2501.9093320933393,
                448.60792834677807,
                6315.076723040953,
                &[
                    4657720171878278193,
                    4658706671145337763,
                    4659693170412397333,
                    4660679669679456903,
                    4661445891637489916,
                    4661939141271019702,
                    4662432390904549486,
                    4662925640538079272,
                    4663418890171609056,
                ],
            ),
            (
                -5495.856200188163,
                150.0901407927636,
                -4069.9998626569086,
                &[
                    13886136809352560381,
                    13885971783497544200,
                    13885806757642528019,
                    13885641731787511838,
                    13885476705932495657,
                    13885311680077479476,
                    13885146654222463295,
                    13884981628367447114,
                    13884816602512430933,
                    13884651576657414752,
                ],
            ),
        ];
        for (start, step, stop, expected) in cases {
            let values: Vec<u64> = np_arange(start, stop, step)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(values, expected, "arange({start}, {stop}, {step})");
        }
        assert!(np_arange(1.0, 1.0, 0.5).is_empty());
    }

    /// `SimulationConfig`: `steps` as given, or `ceil(duration / dt) + 1`,
    /// with the reference's 180 s default; `requested_duration_s` is the
    /// duration asked for, or the actual one.
    #[test]
    fn the_run_length_is_the_references() {
        let config = |extra: Value| -> ScenarioConfig {
            let mut document = json!({"format": "argusnet.scenario-config", "version": 1});
            for (k, v) in extra.as_object().unwrap() {
                document[k] = v.clone();
            }
            serde_json::from_value(document).unwrap()
        };
        assert_eq!(
            run_length(&config(json!({"duration_s": 60.0}))).unwrap(),
            (241, 0.25, 60.0)
        );
        assert_eq!(run_length(&config(json!({}))).unwrap(), (721, 0.25, 180.0));
        assert_eq!(
            run_length(&config(json!({"steps": 80, "dt_s": 0.5}))).unwrap(),
            (80, 0.5, 39.5)
        );
        assert!(run_length(&config(json!({"steps": 8, "duration_s": 2.0}))).is_err());
        assert!(run_length(&config(json!({"dt_s": 0.0}))).is_err());
    }

    /// `np.gradient(f, coords, edge_order=1)`, both of numpy's branches:
    /// bit-identical spacing takes the central difference, anything else the
    /// three-point non-uniform formula. Expected bits from numpy 2.5.
    #[test]
    fn np_gradient_takes_both_of_numpys_branches() {
        let f = [3.0, 7.5, 2.25, 9.125, 4.0, 11.0, 0.5, 6.75];
        let uniform = super::super::linspace(0.0, 35.0, 8);
        let nonuniform = super::super::linspace(-311.5, 342.1, 8);
        let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(np_gradient_axis(&f, &uniform)),
            vec![
                4606281698874543309,
                13813440777070785331,
                4595022699806117069,
                4595473059768854118,
                4595923419731591168,
                13823348696251000422,
                13824699776139211571,
                4608308318706860032,
            ]
        );
        assert_eq!(
            bits(np_gradient_axis(&f, &nonuniform)),
            vec![
                4587106396960502908,
                13794652442958264380,
                4576169871060009560,
                4576555736389160019,
                4576941601718310504,
                13804431372871306319,
                13805588968858757718,
                4589487737277545817,
            ]
        );
    }

    /// `clamp_to_bounds` collapses to the midpoint when the margin swallows
    /// the span, rather than producing an inverted range.
    #[test]
    fn clamping_collapses_to_the_midpoint_when_the_margin_swallows_the_span() {
        let bounds = MapBounds {
            x_min_m: 0.0,
            x_max_m: 10.0,
            y_min_m: -4.0,
            y_max_m: 100.0,
        };
        assert_eq!(clamp_to_bounds([1.0, 1.0], bounds, 6.0), [5.0, 2.0]);
        assert_eq!(clamp_to_bounds([7.0, 99.5], bounds, 1.0), [7.0, 99.0]);
        assert_eq!(clamp_to_bounds([-9.0, 500.0], bounds, 0.0), [0.0, 100.0]);
    }

    /// Stable, like Python's `sorted`: equal heights keep sample order among
    /// the peaks; equal gradients go to the higher sample among the flats,
    /// and equal both keep sample order.
    #[test]
    fn candidate_orders_are_stable_with_the_references_tiebreak() {
        let samples = [
            ([0.0, 0.0], 5.0, 0.0),
            ([1.0, 0.0], 9.0, 0.0),
            ([2.0, 0.0], 9.0, 0.3),
            ([3.0, 0.0], 5.0, 0.0),
            ([4.0, 0.0], 1.0, 0.1),
        ];
        let (peaks, flats) = candidate_orders(&samples, 5);
        let xs = |v: Vec<[f64; 2]>| v.iter().map(|p| p[0]).collect::<Vec<_>>();
        assert_eq!(xs(peaks), vec![1.0, 2.0, 0.0, 3.0, 4.0]);
        assert_eq!(xs(flats), vec![1.0, 0.0, 3.0, 4.0, 2.0]);
    }

    /// A push out of one building can land in the next; the reference keeps
    /// pushing, up to eight times, until the point is clear.
    #[test]
    fn a_push_that_lands_in_a_neighbour_is_pushed_again() {
        use argusnet_types::Bounds2d;
        use argusnet_world::obstacles::BlockerType;
        let building = |id: &str, x0: f64, x1: f64, y0: f64, y1: f64| {
            Obstacle::polygon_prism(
                id,
                BlockerType::Building,
                vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                0.0,
                100.0,
            )
            .unwrap()
        };
        // Pushing out of `a` through its near edge (to x = 11) lands inside
        // the thin `b`, whose nearest edge from there is its long side, so a
        // second push clears both.
        let layer = ObstacleLayer::new(
            Bounds2d::new(-50.0, 50.0, -50.0, 50.0),
            100.0,
            vec![
                building("a", 0.0, 10.0, 0.0, 10.0),
                building("b", 10.5, 30.0, 4.8, 5.2),
            ],
        )
        .unwrap();
        let flat = |_: f64, _: f64| 0.0;
        let start = [9.8, 5.0];
        let first = layer
            .point_collides([start[0], start[1], 10.0])
            .unwrap()
            .push_outside_xy(start[0], start[1], 1.0);
        assert!(
            layer.point_collides([first[0], first[1], 10.0]).is_some(),
            "the fixture must make one push insufficient: {first:?}"
        );
        let settled = push_outside_obstacles(start, 10.0, &flat, &layer);
        assert!(
            layer
                .point_collides([settled[0], settled[1], 10.0])
                .is_none(),
            "{settled:?}"
        );
    }
}
