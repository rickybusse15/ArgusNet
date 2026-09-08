//! Small, versioned C ABI for Python. Objects stay in Rust behind integer handles.
//! JSON carries commands and metadata, never a per-query copy of the world.
use argusnet_world::{
    analytic::{AnalyticTerrain, Feature},
    index::ObstacleLayer,
    obstacles::{Bounds2d, Obstacle, ObstacleMetadata},
    tiled::{self, Tile, TiledTerrain},
    visibility, TerrainQuery,
};
use safety_engine::{
    limits::DronePhysicalLimits,
    simulation::{self, SimulationSafetyMonitor},
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr, CString},
    sync::{Mutex, OnceLock},
};
#[derive(Debug, Clone)]
enum Object {
    LandCover(argusnet_world::land_cover::LandCoverLayer),
    Analytic(AnalyticTerrain),
    Terrain(TiledTerrain),
    Obstacle(Obstacle),
    Obstacles(ObstacleLayer),
    Safety(SimulationSafetyMonitor),
}
#[derive(Default)]
struct Store {
    next: u64,
    objects: BTreeMap<u64, Object>,
}
static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
fn read<T: DeserializeOwned>(v: &Value, key: &str) -> Result<T, String> {
    serde_json::from_value(
        v.get(key)
            .cloned()
            .ok_or_else(|| format!("missing {key}"))?,
    )
    .map_err(|e| e.to_string())
}
fn object<'a>(s: &'a Store, v: &Value, key: &str) -> Result<&'a Object, String> {
    s.objects
        .get(&read::<u64>(v, key)?)
        .ok_or_else(|| "expired native handle".into())
}
fn encode(v: impl serde::Serialize) -> Result<Value, String> {
    serde_json::to_value(v).map_err(|e| e.to_string())
}
fn limits(v: &Value) -> Result<DronePhysicalLimits, String> {
    let mut base = serde_json::to_value(DronePhysicalLimits::tracker_default()).unwrap();
    if let Some(overrides) = v.as_object() {
        for (k, val) in overrides {
            if base.get(k).is_none() {
                return Err(format!("unknown physical limit {k}"));
            }
            base[k] = val.clone();
        }
    }
    serde_json::from_value(base).map_err(|e| e.to_string())
}
fn dispatch(v: Value, s: &mut Store) -> Result<Value, String> {
    let op = v["op"].as_str().ok_or("missing operation")?;
    let created = match op {
        "land_cover_new" => Some(Object::LandCover(
            argusnet_world::land_cover::LandCoverLayer::new(
                read(&v, "bounds")?,
                read(&v, "tile_size_cells")?,
                read(&v, "resolution")?,
                read(&v, "tiles")?,
            )?,
        )),
        "clone" => Some(object(s, &v, "handle")?.clone()),
        "analytic_new" => Some(Object::Analytic(read(&v, "parameters")?)),
        "terrain_new" => Some(Object::Terrain(TiledTerrain::new(
            read(&v, "bounds")?,
            read(&v, "tile_size_cells")?,
            read(&v, "base_resolution_m")?,
            read(&v, "floor_m")?,
            read::<Vec<Tile>>(&v, "tiles")?,
        )?)),
        "obstacle_new" => Some(Object::Obstacle(
            read::<ObstacleMetadata>(&v, "metadata")?
                .into_obstacle()
                .map_err(|e| e.to_string())?,
        )),
        "obstacles_new" => {
            let ids: Vec<u64> = read(&v, "handles")?;
            let mut obstacles = vec![];
            for id in ids {
                match s.objects.get(&id) {
                    Some(Object::Obstacle(o)) => obstacles.push(o.clone()),
                    _ => return Err("expected obstacle handle".into()),
                }
            }
            Some(Object::Obstacles(ObstacleLayer::new(
                read(&v, "bounds")?,
                read(&v, "tile_size_m")?,
                obstacles,
            )?))
        }
        "safety_new" => Some(Object::Safety(SimulationSafetyMonitor::new(
            limits(&v["limits"])?,
            read(&v, "abort_frames")?,
            read(&v, "caution_fraction")?,
        )?)),
        _ => None,
    };
    if let Some(o) = created {
        s.next = s.next.checked_add(1).ok_or("native handle overflow")?;
        s.objects.insert(s.next, o);
        return Ok(json!(s.next));
    }
    match op {
        "land_cover" => {
            let Object::LandCover(l) = object(s, &v, "handle")? else {
                return Err("expected land cover".into());
            };
            let p: [f64; 2] = read(&v, "point")?;
            encode(l.sample(p[0], p[1]))
        }
        "terrain_build" => {
            let t = TiledTerrain::from_grid(
                read(&v, "bounds")?,
                &read::<Vec<Vec<f64>>>(&v, "grid")?,
                read(&v, "resolution")?,
                read(&v, "tile_size_cells")?,
                read(&v, "floor")?,
                &read::<Vec<f64>>(&v, "lods")?,
            )?;
            encode(t.tiles.into_values().collect::<Vec<_>>())
        }
        "segment_hit" => encode(argusnet_world::obstacles::segment_intersection_parameters(
            read(&v, "a0")?,
            read(&v, "a1")?,
            read(&v, "b0")?,
            read(&v, "b1")?,
        )),
        "polygon_intervals" => encode(argusnet_world::obstacles::segment_polygon_intervals(
            read(&v, "origin")?,
            read(&v, "target")?,
            &read::<Vec<[f64; 2]>>(&v, "polygon")?,
        )),
        "nearest_point" => encode(argusnet_world::obstacles::nearest_point_on_segment(
            read(&v, "point")?,
            read(&v, "start")?,
            read(&v, "end")?,
        )),
        "polygon_area" => encode(argusnet_world::obstacles::signed_polygon_area(&read::<
            Vec<[f64; 2]>,
        >(
            &v, "polygon",
        )?)),
        "outward_normal" => encode(argusnet_world::obstacles::edge_outward_normal(
            read(&v, "start")?,
            read(&v, "end")?,
            read(&v, "ccw")?,
        )),
        "visibility_profile" => encode(visibility::VisibilityProfile::default()),
        "version" => Ok(json!(1)),
        "drop" => {
            s.objects.remove(&read::<u64>(&v, "handle")?);
            Ok(Value::Null)
        }
        "pyramid" => encode(tiled::pyramid(
            &read::<Vec<Vec<f64>>>(&v, "grid")?,
            read(&v, "maximum")?,
        )?),
        "feature" => encode(read::<Feature>(&v, "feature")?.height(read(&v, "point")?)),
        "noise_hash" => encode(argusnet_world::analytic::noise_hash(
            read(&v, "ix")?,
            read(&v, "iy")?,
            read(&v, "seed")?,
        )),
        "value_noise" => encode(argusnet_world::analytic::value_noise(
            read(&v, "x")?,
            read(&v, "y")?,
            read(&v, "seed")?,
        )),
        "point_in_polygon" => encode(argusnet_world::obstacles::point_in_polygon(
            read(&v, "point")?,
            &read::<Vec<[f64; 2]>>(&v, "polygon")?,
        )),
        "heights" | "gradients" | "curvature" => {
            let obj = object(s, &v, "handle")?;
            let raw = v["analytic"].as_bool().unwrap_or(false);
            let height = |x, y| match obj {
                Object::Analytic(t) => {
                    if raw {
                        t.analytic_height([x, y])
                    } else {
                        t.height([x, y])
                    }
                }
                Object::Terrain(t) => t.height_at(x, y),
                _ => f64::NAN,
            };
            if !matches!(obj, Object::Analytic(_) | Object::Terrain(_)) {
                return Err("expected terrain handle".into());
            }
            let points: Vec<[f64; 2]> = read(&v, "points")?;
            let delta = v["delta"].as_f64().unwrap_or(0.5).max(0.05);
            if op == "heights" {
                encode(
                    points
                        .iter()
                        .map(|p| height(p[0], p[1]))
                        .collect::<Vec<_>>(),
                )
            } else if op == "gradients" {
                encode(
                    points
                        .iter()
                        .map(|p| tiled::gradient(height, p[0], p[1], delta))
                        .collect::<Vec<_>>(),
                )
            } else {
                encode(
                    points
                        .iter()
                        .map(|p| tiled::curvature(height, p[0], p[1], delta))
                        .collect::<Vec<_>>(),
                )
            }
        }
        "obstacle" => {
            let Object::Obstacle(o) = object(s, &v, "handle")? else {
                return Err("expected obstacle".into());
            };
            match v["method"].as_str().unwrap_or("") {
                "bounds" => encode(o.bounds_xy_m()),
                "inside" => {
                    let p: [f64; 3] = read(&v, "point")?;
                    encode(o.point_inside(p[0], p[1], p[2]))
                }
                "push" => {
                    let p: [f64; 2] = read(&v, "point")?;
                    encode(o.push_outside_xy(p[0], p[1], read(&v, "margin")?))
                }
                "hit" => encode(o.first_hit_t(read(&v, "origin")?, read(&v, "target")?)),
                "length" => encode(o.path_length_inside(read(&v, "origin")?, read(&v, "target")?)),
                _ => Err("unknown obstacle method".into()),
            }
        }
        "obstacles_query" | "collision" => {
            let Object::Obstacles(o) = object(s, &v, "handle")? else {
                return Err("expected obstacle layer".into());
            };
            if op == "collision" {
                encode(o.point_collides(read(&v, "point")?).map(|o| &o.id))
            } else {
                encode(
                    o.query(read(&v, "bounds")?)
                        .iter()
                        .map(|o| &o.id)
                        .collect::<Vec<_>>(),
                )
            }
        }
        "los" | "terrain_intersection" => {
            let Object::Terrain(t) = object(s, &v, "terrain")? else {
                return Err("expected tiled terrain".into());
            };
            let a = read(&v, "origin")?;
            let b = read(&v, "target")?;
            let c = read(&v, "clearance")?;
            if op == "terrain_intersection" {
                encode(visibility::terrain_intersection(t, a, b, c))
            } else {
                let Object::Obstacles(o) = object(s, &v, "obstacles")? else {
                    return Err("expected obstacle layer".into());
                };
                encode(visibility::los(
                    t,
                    o,
                    a,
                    b,
                    &read(&v, "profile")?,
                    read(&v, "land_cover")?,
                    c,
                ))
            }
        }
        "dda" => encode(visibility::dda(
            read(&v, "origin")?,
            read(&v, "target")?,
            read::<Bounds2d>(&v, "bounds")?,
            read(&v, "cell")?,
        )),
        "clip" => encode(visibility::clip(
            read(&v, "origin")?,
            read(&v, "target")?,
            read(&v, "bounds")?,
        )),
        "detection" => encode(visibility::detection(
            &read(&v, "visibility")?,
            read(&v, "range")?,
            read(&v, "max_range")?,
            read(&v, "weather")?,
            read(&v, "weather_noise")?,
        )),
        "path_loss" => encode(visibility::path_loss(
            read(&v, "range")?,
            read(&v, "max_range")?,
        )),
        "noise" => encode(visibility::effective_noise(
            read(&v, "range")?,
            read(&v, "land_noise")?,
            read(&v, "weather_noise")?,
            read(&v, "max_range")?,
        )),
        "dominant" => encode(visibility::dominant(
            read(&v, "path")?,
            read(&v, "transmittance")?,
            read(&v, "weather")?,
            read(&v, "blocked")?,
        )),
        "limits" => encode(if v["profile"] == "interceptor" {
            DronePhysicalLimits::interceptor_default()
        } else {
            DronePhysicalLimits::tracker_default()
        }),
        "turn_radius" => encode(limits(&v["limits"])?.min_turn_radius_at(read(&v, "speed")?)),
        "safety_check" => encode(simulation::check(
            &limits(&v["limits"])?,
            &read(&v, "state")?,
        )),
        "safety_process" | "safety_state" | "safety_clear" => {
            let id = read::<u64>(&v, "handle")?;
            let Some(Object::Safety(m)) = s.objects.get_mut(&id) else {
                return Err("expected safety monitor".into());
            };
            let name = read::<String>(&v, "id")?;
            match op {
                "safety_process" => encode(m.process(&name, &read(&v, "state")?)),
                "safety_state" => encode(m.state(&name)),
                _ => {
                    m.clear(&name);
                    Ok(Value::Null)
                }
            }
        }
        _ => Err(format!("unknown native operation {op}")),
    }
}
/// Execute a UTF-8 JSON request. Returns an owned JSON envelope, freed by `argusnet_free`.
/// # Safety
/// `input` must point to a NUL-terminated readable string for this call's duration.
#[no_mangle]
pub unsafe extern "C" fn argusnet_call(input: *const c_char) -> *mut c_char {
    let outcome = std::panic::catch_unwind(|| -> Result<Value, String> {
        if input.is_null() {
            return Err("null request".into());
        }
        // SAFETY: caller contract requires a valid NUL-terminated string.
        let bytes = unsafe { CStr::from_ptr(input) }.to_bytes();
        let v = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let mut store = STORE
            .get_or_init(|| Mutex::new(Store::default()))
            .lock()
            .map_err(|_| "native registry poisoned")?;
        dispatch(v, &mut store)
    });
    let envelope = match outcome {
        Ok(Ok(v)) => json!({"ok":v}),
        Ok(Err(e)) => json!({"error":e}),
        Err(_) => json!({"error":"native operation panicked"}),
    };
    CString::new(envelope.to_string()).unwrap().into_raw()
}
/// # Safety
/// `output` must be a live pointer returned by `argusnet_call`, freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn argusnet_free(output: *mut c_char) {
    if !output.is_null() {
        // SAFETY: ownership is returned by the caller exactly once.
        drop(unsafe { CString::from_raw(output) });
    }
}

/// Batch height/differential queries over contiguous f64 buffers, without JSON.
/// Return codes: 0 success, 1 invalid arguments/handle, 2 poisoned registry.
/// # Safety
/// `points` must reference 2*count readable f64s and `output` must reference
/// count writable f64s (2*count for mode 2). Buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn argusnet_sample(
    handle: u64,
    points: *const f64,
    count: usize,
    mode: u32,
    delta: f64,
    output: *mut f64,
) -> i32 {
    if points.is_null()
        || output.is_null()
        || count > isize::MAX as usize / 16
        || mode > 3
        || !delta.is_finite()
    {
        return 1;
    }
    let Ok(s) = STORE.get_or_init(|| Mutex::new(Store::default())).lock() else {
        return 2;
    };
    let Some(obj) = s.objects.get(&handle) else {
        return 1;
    };
    if !matches!(obj, Object::Analytic(_) | Object::Terrain(_)) {
        return 1;
    }
    // SAFETY: buffers and their extents are guaranteed by the caller contract.
    let input = unsafe { std::slice::from_raw_parts(points, count * 2) };
    if input.iter().any(|v| !v.is_finite()) {
        return 1;
    }
    // SAFETY: output length is determined by mode, as specified in the contract.
    let output =
        unsafe { std::slice::from_raw_parts_mut(output, count * if mode == 2 { 2 } else { 1 }) };
    let h = |x, y| match obj {
        Object::Analytic(t) => {
            if mode == 1 {
                t.analytic_height([x, y])
            } else {
                t.height([x, y])
            }
        }
        Object::Terrain(t) => t.height_at(x, y),
        _ => unreachable!(),
    };
    let delta = delta.max(0.05);
    for (i, p) in input.as_chunks::<2>().0.iter().enumerate() {
        match mode {
            2 => {
                let g = tiled::gradient(h, p[0], p[1], delta);
                output[i * 2..i * 2 + 2].copy_from_slice(&g);
            }
            3 => output[i] = tiled::curvature(h, p[0], p[1], delta),
            _ => output[i] = h(p[0], p[1]),
        }
    }
    0
}

/// Score a batch of footprints without serializing the coverage raster.
/// Returns 0 on success and 1 for invalid arguments.
/// # Safety
/// grid points to nx*ny i32 values; positions to count*2 f64 values;
/// origin to two f64 values; output to count writable f64 values.
/// All buffers must be valid, aligned, and output must not overlap inputs.
#[no_mangle]
pub unsafe extern "C" fn argusnet_score_footprints(
    grid: *const i32,
    nx: usize,
    ny: usize,
    origin: *const f64,
    resolution: f64,
    radius: f64,
    positions: *const f64,
    count: usize,
    output: *mut f64,
) -> i32 {
    let Some(cells) = nx.checked_mul(ny) else {
        return 1;
    };
    if grid.is_null()
        || origin.is_null()
        || positions.is_null()
        || output.is_null()
        || cells > isize::MAX as usize / 4
        || count > isize::MAX as usize / 16
    {
        return 1;
    }
    // SAFETY: the caller provides valid buffers with the extents documented above.
    let (grid, origin, positions) = unsafe {
        (
            std::slice::from_raw_parts(grid, cells),
            [*origin, *origin.add(1)],
            std::slice::from_raw_parts(positions.cast::<[f64; 2]>(), count),
        )
    };
    let Ok(scores) =
        argusnet_mapping::score_footprints(grid, [nx, ny], origin, resolution, radius, positions)
    else {
        return 1;
    };
    // SAFETY: writable output has count elements and does not overlap inputs.
    unsafe {
        std::ptr::copy_nonoverlapping(scores.as_ptr(), output, count);
    }
    0
}
