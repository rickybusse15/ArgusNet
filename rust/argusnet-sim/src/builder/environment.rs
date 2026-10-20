//! `build_default_environment`: the obstacles and land-cover patches a map
//! and terrain preset put on the terrain (P4.9b.1).
//!
//! Ported line for line from `sim.py` -- `build_default_environment`,
//! `build_obstacle_cluster`, `_preset_environment_profile`,
//! `_filter_obstacles_for_landscape` and the elevation helpers. Obstacles are
//! produced as their **metadata documents** (`ObstaclePrimitive.to_metadata`),
//! which is what a bundle carries; the runtime and the land-cover pass build
//! their `Obstacle`s from those, exactly as they do from a Python bundle.
//!
//! # Arithmetic that must match
//!
//! * A box footprint is `corners @ rotation.T + centre`, which numpy computes
//!   with a fused multiply-add ([`box_footprint`]). The world crate's own
//!   `Obstacle::oriented_box` associates `(cx + x cos) - y sin`, which rounds
//!   differently; it is not used here.
//! * A wall's length is `np.linalg.norm` (`sqrt(x*x + y*y)`, not `hypot`),
//!   and its offset is `normal * (thickness * 0.5)`.
//! * Obstacle elevations come from terrain heights, so they inherit the
//!   terrain's numpy-`exp` divergence (P4.3): within ~1e-13 m of Python's,
//!   not bit-equal.

use serde_json::{json, Value};

use argusnet_world::land_cover_build::{LandCoverClass, LandCoverPatch};
use argusnet_world::landscape::BuiltTerrain;

/// `GROUND_CONTACT_TOP_PAD_M`: `SimulationConstants.dynamics.ground_contact_top_pad_m`.
pub const GROUND_CONTACT_TOP_PAD_M: f64 = 3.0;

/// One obstacle as the reference describes it, and the footprint used for
/// sampling and filtering (`None` for a cylinder, which has none).
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltObstacle {
    pub metadata: Value,
    pub blocker_type: &'static str,
    footprint: Option<Vec<[f64; 2]>>,
    centre: Option<[f64; 2]>,
}

/// `_terrain_height_span_at_points`: min and max, in the order Python's
/// `min()`/`max()` scan them.
fn span_at_points(terrain: &BuiltTerrain, points: &[[f64; 2]]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for [x, y] in points {
        let h = terrain.height_at(*x, *y);
        // Python's `min`/`max` keep the first of equal values; for floats
        // that is only observable with signed zeros, which heights never are.
        if h < min {
            min = h;
        }
        if h > max {
            max = h;
        }
    }
    (min, max)
}

/// `_terrain_height_span_for_cylinder`: eight rim samples from
/// `np.linspace(0, tau, 8, endpoint=False)`, plus the centre.
fn span_for_cylinder(terrain: &BuiltTerrain, cx: f64, cy: f64, radius: f64) -> (f64, f64) {
    let step = std::f64::consts::TAU / 8.0;
    let mut points: Vec<[f64; 2]> = (0..8)
        .map(|i| {
            let angle = i as f64 * step;
            [cx + radius * angle.cos(), cy + radius * angle.sin()]
        })
        .collect();
    points.push([cx, cy]);
    span_at_points(terrain, &points)
}

/// `_solid_obstacle_elevations`.
fn solid_elevations(min_z: f64, max_z: f64, nominal_height_m: f64) -> (f64, f64) {
    (
        min_z,
        max_z + nominal_height_m.max(0.0) + GROUND_CONTACT_TOP_PAD_M,
    )
}

/// `OrientedBox.footprint_xy_m`: `corners @ rotation.T + centre`.
///
/// numpy's matmul accumulates each output with a fused multiply-add:
/// `fma(y, r, x * c)`, then the centre is added. Measured on the reference
/// platform against both association orders and both FMA placements; only
/// this one reproduces every corner. Like `np.exp`, that is the platform's
/// kernel speaking, so on a machine whose matmul does not fuse the reference
/// itself would differ in the last place.
pub fn box_footprint(cx: f64, cy: f64, length: f64, width: f64, yaw: f64) -> Vec<[f64; 2]> {
    let (hl, hw) = (length * 0.5, width * 0.5);
    let (cos, sin) = (yaw.cos(), yaw.sin());
    [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]]
        .iter()
        .map(|[x, y]| [y.mul_add(-sin, x * cos) + cx, y.mul_add(cos, x * sin) + cy])
        .collect()
}

/// `WallSegment.footprint_xy_m`.
pub fn wall_footprint(start: [f64; 2], end: [f64; 2], thickness: f64) -> Vec<[f64; 2]> {
    let segment = [end[0] - start[0], end[1] - start[1]];
    let length = (segment[0] * segment[0] + segment[1] * segment[1])
        .sqrt()
        .max(1.0e-9);
    let normal = [-segment[1] / length, segment[0] / length];
    let half = thickness * 0.5;
    let offset = [normal[0] * half, normal[1] * half];
    vec![
        [start[0] - offset[0], start[1] - offset[1]],
        [end[0] - offset[0], end[1] - offset[1]],
        [end[0] + offset[0], end[1] + offset[1]],
        [start[0] + offset[0], start[1] + offset[1]],
    ]
}

fn pairs(points: &[[f64; 2]]) -> Value {
    json!(points.iter().map(|p| vec![p[0], p[1]]).collect::<Vec<_>>())
}

#[allow(clippy::too_many_arguments)]
fn oriented_box(
    id: &str,
    blocker: &'static str,
    cx: f64,
    cy: f64,
    length: f64,
    width: f64,
    yaw: f64,
    base_z: f64,
    top_z: f64,
) -> BuiltObstacle {
    let footprint = box_footprint(cx, cy, length, width, yaw);
    BuiltObstacle {
        metadata: json!({
            "object_id": id,
            "kind": "box-v1",
            "blocker_type": blocker,
            "center_x_m": cx,
            "center_y_m": cy,
            "length_m": length,
            "width_m": width,
            "yaw_rad": yaw,
            "base_elevation_m": base_z,
            "top_elevation_m": top_z,
            "footprint_xy_m": pairs(&footprint),
        }),
        blocker_type: blocker,
        footprint: Some(footprint),
        centre: Some([cx, cy]),
    }
}

fn cylinder(id: &str, cx: f64, cy: f64, radius: f64, base_z: f64, top_z: f64) -> BuiltObstacle {
    BuiltObstacle {
        metadata: json!({
            "object_id": id,
            "kind": "cylinder-v1",
            "blocker_type": "building",
            "center_x_m": cx,
            "center_y_m": cy,
            "radius_m": radius,
            "base_elevation_m": base_z,
            "top_elevation_m": top_z,
        }),
        blocker_type: "building",
        footprint: None,
        centre: Some([cx, cy]),
    }
}

/// An imported building (R1, `buildings_path`): its metadata as written
/// (`building-v2`), and its footprint for sampling.
pub fn imported_building(metadata: Value, footprint: Vec<[f64; 2]>) -> BuiltObstacle {
    BuiltObstacle {
        metadata,
        blocker_type: "building",
        footprint: Some(footprint),
        centre: None,
    }
}

/// `BuildingPrism`: a polygon prism whose blocker is always `building`.
fn building_prism(id: &str, footprint: Vec<[f64; 2]>, base_z: f64, top_z: f64) -> BuiltObstacle {
    BuiltObstacle {
        metadata: json!({
            "object_id": id,
            "kind": "polygon-prism-v1",
            "blocker_type": "building",
            "footprint_xy_m": pairs(&footprint),
            "base_elevation_m": base_z,
            "top_elevation_m": top_z,
        }),
        blocker_type: "building",
        footprint: Some(footprint),
        centre: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn wall(
    id: &str,
    start: [f64; 2],
    end: [f64; 2],
    thickness: f64,
    base_z: f64,
    top_z: f64,
) -> BuiltObstacle {
    let footprint = wall_footprint(start, end, thickness);
    BuiltObstacle {
        metadata: json!({
            "object_id": id,
            "kind": "wall-v1",
            "blocker_type": "wall",
            "start_xy_m": [start[0], start[1]],
            "end_xy_m": [end[0], end[1]],
            "thickness_m": thickness,
            "base_elevation_m": base_z,
            "top_elevation_m": top_z,
            "footprint_xy_m": pairs(&footprint),
        }),
        blocker_type: "wall",
        footprint: Some(footprint),
        centre: None,
    }
}

/// `_forest_stand_for_terrain`.
fn forest_stand(
    terrain: &BuiltTerrain,
    id: &str,
    footprint: Vec<[f64; 2]>,
    reference_xy: [f64; 2],
    canopy_top_agl_m: f64,
    density: f64,
) -> BuiltObstacle {
    let (min_z, max_z) = span_at_points(terrain, &footprint);
    let reference_z = terrain.height_at(reference_xy[0], reference_xy[1]);
    let top_z = (reference_z + canopy_top_agl_m).max(max_z + GROUND_CONTACT_TOP_PAD_M);
    BuiltObstacle {
        metadata: json!({
            "object_id": id,
            "kind": "forest-stand-v1",
            "blocker_type": "vegetation",
            "footprint_xy_m": pairs(&footprint),
            "base_elevation_m": min_z,
            "top_elevation_m": top_z,
            "density": density,
        }),
        blocker_type: "vegetation",
        footprint: Some(footprint),
        centre: None,
    }
}

/// A box whose elevations come from the terrain under its footprint.
#[allow(clippy::too_many_arguments)]
fn seated_box(
    terrain: &BuiltTerrain,
    id: &str,
    cx: f64,
    cy: f64,
    length: f64,
    width: f64,
    yaw: f64,
    nominal_height_m: f64,
) -> BuiltObstacle {
    let footprint = box_footprint(cx, cy, length, width, yaw);
    let (min_z, max_z) = span_at_points(terrain, &footprint);
    let (base, top) = solid_elevations(min_z, max_z, nominal_height_m);
    oriented_box(id, "building", cx, cy, length, width, yaw, base, top)
}

fn seated_wall(
    terrain: &BuiltTerrain,
    id: &str,
    start: [f64; 2],
    end: [f64; 2],
    thickness: f64,
    nominal_height_m: f64,
) -> BuiltObstacle {
    let (min_z, max_z) = span_at_points(terrain, &wall_footprint(start, end, thickness));
    let (base, top) = solid_elevations(min_z, max_z, nominal_height_m);
    wall(id, start, end, thickness, base, top)
}

fn seated_cylinder(
    terrain: &BuiltTerrain,
    id: &str,
    cx: f64,
    cy: f64,
    radius: f64,
    nominal_height_m: f64,
) -> BuiltObstacle {
    let (min_z, max_z) = span_for_cylinder(terrain, cx, cy, radius);
    let (base, top) = solid_elevations(min_z, max_z, nominal_height_m);
    cylinder(id, cx, cy, radius, base, top)
}

fn seated_prism(
    terrain: &BuiltTerrain,
    id: &str,
    footprint: Vec<[f64; 2]>,
    nominal_height_m: f64,
) -> BuiltObstacle {
    let (min_z, max_z) = span_at_points(terrain, &footprint);
    let (base, top) = solid_elevations(min_z, max_z, nominal_height_m);
    building_prism(id, footprint, base, top)
}

/// Block centres of `urban_dense`'s grid along x, before scaling, in metres.
const URBAN_DENSE_COLUMNS_X: [f64; 6] = [-225.0, -135.0, -45.0, 45.0, 135.0, 225.0];
/// Block centres of `urban_dense`'s grid along y, before scaling, in metres.
/// Around the map's middle, which sits at y = 15 before scaling.
const URBAN_DENSE_ROWS_Y: [f64; 5] = [-170.0, -75.0, 20.0, 115.0, 210.0];
/// A block's size along x and y, before scaling. With the 90 m and 95 m
/// pitches above, the streets between blocks are 42 m and 45 m wide on the
/// small map and wider on larger ones, so a POI or a ground station pushed
/// out of a block lands in open street.
const URBAN_DENSE_BLOCK_LENGTH_M: f64 = 48.0;
const URBAN_DENSE_BLOCK_WIDTH_M: f64 = 50.0;

/// `_urban_dense_blocks`: the 30 blocks of `urban_dense`, row by row.
///
/// The preset exists to make a straight redirect cross buildings (P3). A
/// drone inspects a POI hovering 40 m above the ground and flies from one
/// POI to the next at that height, so every block stands 70 m to 110 m tall:
/// even the lowest top is more than 30 m above a POI-to-POI flight. All of
/// them stay well below the 230 m operating height, so scanning flies over
/// the grid. The heights follow a fixed pattern rather than a random draw, so
/// the layout is the same for every seed.
fn urban_dense_blocks(terrain: &BuiltTerrain, scale: f64) -> Vec<BuiltObstacle> {
    let mut blocks = Vec::new();
    for (row, y) in URBAN_DENSE_ROWS_Y.iter().enumerate() {
        for (column, x) in URBAN_DENSE_COLUMNS_X.iter().enumerate() {
            let nominal_height_m = 70.0 + 10.0 * ((2 * row + 3 * column) % 5) as f64;
            blocks.push(seated_box(
                terrain,
                &format!("urban-block-{row}-{column}"),
                x * scale,
                y * scale,
                URBAN_DENSE_BLOCK_LENGTH_M * scale,
                URBAN_DENSE_BLOCK_WIDTH_M * scale,
                0.0,
                nominal_height_m,
            ));
        }
    }
    blocks
}

/// `build_obstacle_cluster`: a hangar, a warehouse and a tower (hard), and a
/// forest stand (soft), around `centre`.
fn obstacle_cluster(
    terrain: &BuiltTerrain,
    cluster_id: &str,
    centre: [f64; 2],
    scale: f64,
) -> (Vec<BuiltObstacle>, Vec<BuiltObstacle>) {
    let [cx, cy] = centre;
    let hangar = [cx - 28.0 * scale, cy - 18.0 * scale];
    let warehouse = vec![
        [cx + 12.0 * scale, cy - 12.0 * scale],
        [cx + 62.0 * scale, cy - 12.0 * scale],
        [cx + 62.0 * scale, cy + 28.0 * scale],
        [cx + 12.0 * scale, cy + 28.0 * scale],
    ];
    let tower = [cx - 6.0 * scale, cy + 38.0 * scale];
    let hard = vec![
        seated_box(
            terrain,
            &format!("{cluster_id}-hangar"),
            hangar[0],
            hangar[1],
            56.0 * scale,
            32.0 * scale,
            0.18,
            32.0,
        ),
        seated_prism(terrain, &format!("{cluster_id}-warehouse"), warehouse, 28.0),
        seated_cylinder(
            terrain,
            &format!("{cluster_id}-tower"),
            tower[0],
            tower[1],
            10.0 * scale,
            54.0,
        ),
    ];
    let soft = vec![forest_stand(
        terrain,
        &format!("{cluster_id}-forest"),
        vec![
            [cx - 84.0 * scale, cy + 26.0 * scale],
            [cx - 24.0 * scale, cy + 18.0 * scale],
            [cx + 8.0 * scale, cy + 74.0 * scale],
            [cx - 62.0 * scale, cy + 92.0 * scale],
            [cx - 102.0 * scale, cy + 58.0 * scale],
        ],
        [cx - 48.0 * scale, cy + 48.0 * scale],
        26.0,
        0.52,
    )];
    (hard, soft)
}

fn rectangle_patch(
    centre: [f64; 2],
    length: f64,
    width: f64,
    class: LandCoverClass,
    density: i64,
) -> LandCoverPatch {
    let (hl, hw) = (length * 0.5, width * 0.5);
    LandCoverPatch {
        polygon_xy_m: vec![
            [centre[0] - hl, centre[1] - hw],
            [centre[0] + hl, centre[1] - hw],
            [centre[0] + hl, centre[1] + hw],
            [centre[0] - hl, centre[1] + hw],
        ],
        land_cover_class: class,
        density,
    }
}

/// `_ellipse_patch` with its default 20 segments. `np.cos`/`np.sin` over the
/// angle array: if numpy's vector trigonometry differs from libm's, this is
/// where it would show.
fn ellipse_patch(centre: [f64; 2], rx: f64, ry: f64, class: LandCoverClass) -> LandCoverPatch {
    let count = 20usize;
    let step = std::f64::consts::TAU / count as f64;
    LandCoverPatch {
        polygon_xy_m: (0..count)
            .map(|i| {
                let angle = i as f64 * step;
                [centre[0] + angle.cos() * rx, centre[1] + angle.sin() * ry]
            })
            .collect(),
        land_cover_class: class,
        density: 0,
    }
}

fn river_corridor_patch(scale: f64) -> LandCoverPatch {
    let points: [[f64; 2]; 10] = [
        [-520.0, -42.0],
        [-260.0, -28.0],
        [-20.0, -38.0],
        [220.0, -18.0],
        [520.0, 2.0],
        [520.0, 68.0],
        [220.0, 48.0],
        [-20.0, 72.0],
        [-260.0, 60.0],
        [-520.0, 36.0],
    ];
    LandCoverPatch {
        polygon_xy_m: points.iter().map(|[x, y]| [x * scale, y * scale]).collect(),
        land_cover_class: LandCoverClass::Water,
        density: 0,
    }
}

/// The map bounds, as `map_bounds_m`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapBounds {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

/// What `_preset_environment_profile` returns: the obstacles, the patches
/// with obstacles, and the patches for a clean-terrain run.
struct PresetProfile {
    obstacles: Vec<BuiltObstacle>,
    patches: Vec<LandCoverPatch>,
    terrain_only_patches: Vec<LandCoverPatch>,
}

fn preset_profile(
    terrain_preset: &str,
    scale: f64,
    bounds: MapBounds,
    terrain: &BuiltTerrain,
    hard: Vec<BuiltObstacle>,
    soft: Vec<BuiltObstacle>,
) -> PresetProfile {
    let centre = [
        (bounds.x_min_m + bounds.x_max_m) * 0.5,
        (bounds.y_min_m + bounds.y_max_m) * 0.5,
    ];
    let river = river_corridor_patch(scale);
    let coastal = rectangle_patch(
        [centre[0], bounds.y_min_m + 95.0 * scale],
        (bounds.x_max_m - bounds.x_min_m) * 0.92,
        (bounds.y_max_m - bounds.y_min_m) * 0.28,
        LandCoverClass::Water,
        0,
    );
    let lakes = vec![
        ellipse_patch(
            [-220.0 * scale, 165.0 * scale],
            115.0 * scale,
            78.0 * scale,
            LandCoverClass::Water,
        ),
        ellipse_patch(
            [245.0 * scale, -175.0 * scale],
            95.0 * scale,
            62.0 * scale,
            LandCoverClass::Water,
        ),
    ];
    let urban = vec![
        rectangle_patch(
            centre,
            280.0 * scale,
            220.0 * scale,
            LandCoverClass::Urban,
            190,
        ),
        rectangle_patch(
            [310.0 * scale, 205.0 * scale],
            150.0 * scale,
            120.0 * scale,
            LandCoverClass::Urban,
            180,
        ),
        rectangle_patch(
            [-260.0 * scale, -180.0 * scale],
            160.0 * scale,
            120.0 * scale,
            LandCoverClass::Urban,
            180,
        ),
    ];
    let profile = |obstacles, patches, terrain_only_patches| PresetProfile {
        obstacles,
        patches,
        terrain_only_patches,
    };
    match terrain_preset {
        "jungle_canopy" => {
            let mut obstacles = soft;
            obstacles.push(forest_stand(
                terrain,
                "jungle-canopy-central",
                vec![
                    [-360.0 * scale, -120.0 * scale],
                    [240.0 * scale, -180.0 * scale],
                    [340.0 * scale, 160.0 * scale],
                    [-280.0 * scale, 220.0 * scale],
                ],
                [0.0, 0.0],
                30.0,
                0.82,
            ));
            obstacles.push(forest_stand(
                terrain,
                "jungle-canopy-east",
                vec![
                    [160.0 * scale, -250.0 * scale],
                    [460.0 * scale, -180.0 * scale],
                    [420.0 * scale, 120.0 * scale],
                    [180.0 * scale, 80.0 * scale],
                ],
                [300.0 * scale, -40.0 * scale],
                32.0,
                0.78,
            ));
            profile(obstacles, Vec::new(), Vec::new())
        }
        "arctic_tundra" => profile(
            hard.into_iter()
                .filter(|o| matches!(o.blocker_type, "building" | "wall"))
                .take(2)
                .collect(),
            Vec::new(),
            Vec::new(),
        ),
        "military_compound" => profile(hard, urban, Vec::new()),
        "river_valley" => profile(
            vec![
                forest_stand(
                    terrain,
                    "riparian-west",
                    vec![
                        [-420.0 * scale, 60.0 * scale],
                        [-90.0 * scale, 84.0 * scale],
                        [-70.0 * scale, 132.0 * scale],
                        [-380.0 * scale, 112.0 * scale],
                    ],
                    [-220.0 * scale, 96.0 * scale],
                    24.0,
                    0.55,
                ),
                forest_stand(
                    terrain,
                    "riparian-east",
                    vec![
                        [40.0 * scale, -82.0 * scale],
                        [360.0 * scale, -58.0 * scale],
                        [380.0 * scale, -8.0 * scale],
                        [60.0 * scale, -24.0 * scale],
                    ],
                    [220.0 * scale, -42.0 * scale],
                    24.0,
                    0.52,
                ),
            ],
            vec![river.clone()],
            vec![river],
        ),
        "urban_flat" => profile(
            hard.into_iter().take(3).collect(),
            urban.into_iter().take(2).collect(),
            Vec::new(),
        ),
        // The block grid alone: the default hangar, tower and warehouse would
        // stand inside it. One urban patch covers the whole grid.
        "urban_dense" => profile(
            urban_dense_blocks(terrain, scale),
            vec![rectangle_patch(
                [0.0, 20.0 * scale],
                520.0 * scale,
                460.0 * scale,
                LandCoverClass::Urban,
                200,
            )],
            Vec::new(),
        ),
        "coastal" => profile(
            soft.into_iter().take(1).collect(),
            vec![coastal.clone()],
            vec![coastal],
        ),
        "lake_district" => profile(soft.into_iter().take(2).collect(), lakes.clone(), lakes),
        "rolling_highlands" => profile(soft, Vec::new(), Vec::new()),
        "desert_canyon" => profile(hard.into_iter().take(1).collect(), Vec::new(), Vec::new()),
        "alpine" => {
            let mut obstacles: Vec<BuiltObstacle> = soft.into_iter().take(1).collect();
            obstacles.extend(hard.into_iter().take(2));
            profile(obstacles, Vec::new(), Vec::new())
        }
        _ => profile(Vec::new(), Vec::new(), Vec::new()),
    }
}

/// `_mask_value_at_terrain_xy`: the nearest mask cell by Python's `round`,
/// which rounds half to even.
fn mask_at(terrain: &BuiltTerrain, mask: &argusnet_world::grid_ops::Mask, x: f64, y: f64) -> bool {
    let b = terrain.bounds_xy_m;
    if !(b.x_min_m <= x && x <= b.x_max_m && b.y_min_m <= y && y <= b.y_max_m) {
        return false;
    }
    // Axis 0 is y: `nx` rows, `ny` columns.
    let (rows, cols) = (mask.nx, mask.ny);
    if rows == 0 || cols == 0 {
        return false;
    }
    let col = mask_index(x, b.x_min_m, b.x_max_m - b.x_min_m, cols);
    let row = mask_index(y, b.y_min_m, b.y_max_m - b.y_min_m, rows);
    mask.values[row * cols + col]
}

/// One axis of `_mask_value_at_terrain_xy`:
/// `int(np.clip(round((v - min) / max(extent, 1e-9) * max(count - 1, 1)), 0, count - 1))`,
/// where Python's `round` goes half to even.
pub fn mask_index(value: f64, min: f64, extent: f64, count: usize) -> usize {
    let raw = ((value - min) / extent.max(1.0e-9) * (count.saturating_sub(1).max(1)) as f64)
        .round_ties_even();
    raw.clamp(0.0, count.saturating_sub(1) as f64) as usize
}

/// `_obstacle_sample_points`: the footprint and its centroid, or the centre.
fn sample_points(obstacle: &BuiltObstacle) -> Vec<[f64; 2]> {
    if let Some(footprint) = &obstacle.footprint {
        // `np.mean(points, axis=0)`: a column sum (pairwise for eight or more
        // points, which no footprint here reaches) over the count.
        let n = footprint.len() as f64;
        let mut sum = [0.0, 0.0];
        for [x, y] in footprint {
            sum[0] += x;
            sum[1] += y;
        }
        let mut points = footprint.clone();
        points.push([sum[0] / n, sum[1] / n]);
        return points;
    }
    obstacle.centre.map(|c| vec![c]).unwrap_or_default()
}

/// `_filter_obstacles_for_landscape`: drop vegetation standing mostly in
/// water, and buildings or walls touching water or mostly on steep slopes --
/// unless that would drop everything.
fn filter_for_landscape(
    terrain: &BuiltTerrain,
    obstacles: Vec<BuiltObstacle>,
) -> Vec<BuiltObstacle> {
    let masks = &terrain.masks;
    let mut kept = Vec::new();
    for obstacle in &obstacles {
        let points = sample_points(obstacle);
        if points.is_empty() {
            kept.push(obstacle.clone());
            continue;
        }
        let n = points.len() as f64;
        let water = points
            .iter()
            .filter(|[x, y]| mask_at(terrain, &masks.water, *x, *y))
            .count() as f64
            / n;
        let steep = points
            .iter()
            .filter(|[x, y]| mask_at(terrain, &masks.steep_slope, *x, *y))
            .count() as f64
            / n;
        if obstacle.blocker_type == "vegetation" && water >= 0.5 {
            continue;
        }
        if matches!(obstacle.blocker_type, "building" | "wall") && (water > 0.0 || steep > 0.5) {
            continue;
        }
        kept.push(obstacle.clone());
    }
    if kept.is_empty() {
        obstacles
    } else {
        kept
    }
}

/// The environment's obstacles and the patches its land cover applies.
pub struct Environment {
    pub obstacles: Vec<BuiltObstacle>,
    pub patches: Vec<LandCoverPatch>,
}

/// `build_default_environment`, up to (not including) the land-cover build.
pub fn build_environment(
    terrain: &BuiltTerrain,
    scale: f64,
    bounds: MapBounds,
    map_preset: &str,
    terrain_preset: &str,
    clean_terrain: bool,
) -> Environment {
    let s = scale;
    let mut hard = vec![
        seated_box(
            terrain,
            "hangar-south",
            20.0 * s,
            -35.0 * s,
            72.0 * s,
            42.0 * s,
            0.12,
            42.0,
        ),
        seated_cylinder(terrain, "tower-east", 145.0 * s, 20.0 * s, 14.0 * s, 68.0),
        seated_prism(
            terrain,
            "warehouse-north",
            vec![
                [-145.0 * s, 135.0 * s],
                [-80.0 * s, 135.0 * s],
                [-80.0 * s, 190.0 * s],
                [-145.0 * s, 190.0 * s],
            ],
            34.0,
        ),
        seated_wall(
            terrain,
            "east-fence",
            [175.0 * s, -120.0 * s],
            [230.0 * s, 110.0 * s],
            6.0,
            18.0,
        ),
    ];
    let mut soft = vec![forest_stand(
        terrain,
        "pine-stand-west",
        vec![
            [-250.0 * s, 35.0 * s],
            [-150.0 * s, 30.0 * s],
            [-120.0 * s, 135.0 * s],
            [-205.0 * s, 185.0 * s],
            [-275.0 * s, 125.0 * s],
        ],
        [-200.0 * s, 105.0 * s],
        30.0,
        0.58,
    )];
    let clusters: &[[f64; 2]] = match map_preset {
        "large" => &[[-230.0, -175.0]],
        "xlarge" => &[[-320.0, -210.0], [285.0, -230.0], [-260.0, 235.0]],
        "regional" => &[
            [-360.0, -240.0],
            [320.0, -260.0],
            [-340.0, 260.0],
            [310.0, 245.0],
            [10.0, 20.0],
        ],
        _ => &[],
    };
    for (index, [x, y]) in clusters.iter().enumerate() {
        let (extra_hard, extra_soft) = obstacle_cluster(
            terrain,
            &format!("{map_preset}-cluster-{index}"),
            [x * s, y * s],
            s,
        );
        hard.extend(extra_hard);
        soft.extend(extra_soft);
    }
    if map_preset == "regional" {
        hard.push(seated_wall(
            terrain,
            "regional-corridor-east",
            [420.0 * s, -320.0 * s],
            [470.0 * s, 320.0 * s],
            8.0,
            20.0,
        ));
        hard.push(seated_wall(
            terrain,
            "regional-corridor-west",
            [-470.0 * s, -290.0 * s],
            [-430.0 * s, 300.0 * s],
            8.0,
            20.0,
        ));
    }
    let profile = preset_profile(terrain_preset, scale, bounds, terrain, hard, soft);
    let obstacles = if clean_terrain {
        Vec::new()
    } else {
        filter_for_landscape(terrain, profile.obstacles)
    };
    Environment {
        obstacles,
        patches: if clean_terrain {
            profile.terrain_only_patches
        } else {
            profile.patches
        },
    }
}

#[cfg(test)]
mod tests {
    use super::mask_index;
    use crate::builder::{build_world, ScenarioOptions};
    use serde_json::Value;

    /// `urban_dense` is there to put buildings across POI-to-POI flights,
    /// which hover 40 m above the ground (`poi_hover_agl_m`). Every block
    /// must survive the landscape filter, stand clear of that height by
    /// more than 30 m, and leave streets of at least 40 m between blocks so
    /// POIs and ground stations have room. Checked on the small map (scale
    /// one, the narrowest streets) and the medium map the scenario uses.
    #[test]
    fn urban_dense_stands_thirty_tall_blocks_with_wide_streets() {
        const POI_HOVER_AGL_M: f64 = 40.0;
        for map_preset in ["small", "medium"] {
            let world = build_world(&ScenarioOptions {
                map_preset: map_preset.into(),
                terrain_preset: "urban_dense".into(),
                ..ScenarioOptions::default()
            })
            .unwrap();
            let blocks: Vec<&Value> = world
                .obstacles
                .iter()
                .filter(|o| o.blocker_type == "building")
                .map(|o| &o.metadata)
                .collect();
            assert_eq!(blocks.len(), 30, "{map_preset}: block count");
            assert_eq!(world.obstacles.len(), 30, "{map_preset}: only blocks");
            let number = |block: &Value, key: &str| block[key].as_f64().unwrap();
            for block in &blocks {
                let ground_m = world
                    .terrain
                    .height_at(number(block, "center_x_m"), number(block, "center_y_m"));
                let above_poi_flight_m =
                    number(block, "top_elevation_m") - (ground_m + POI_HOVER_AGL_M);
                assert!(
                    above_poi_flight_m > 30.0,
                    "{map_preset}: {} tops out {above_poi_flight_m} m above a POI flight",
                    block["object_id"]
                );
            }
            // Axis-aligned blocks: the street between two is the larger of
            // the gaps along x and along y.
            let mut narrowest_m = f64::INFINITY;
            for (index, a) in blocks.iter().enumerate() {
                for b in &blocks[index + 1..] {
                    let gap = |centre: &str, size: &str| {
                        (number(a, centre) - number(b, centre)).abs()
                            - (number(a, size) + number(b, size)) * 0.5
                    };
                    let street_m = gap("center_x_m", "length_m").max(gap("center_y_m", "width_m"));
                    narrowest_m = narrowest_m.min(street_m);
                }
            }
            assert!(
                narrowest_m >= 40.0,
                "{map_preset}: narrowest street {narrowest_m} m"
            );
        }
    }

    /// Python's `round` is half to even: 2.5 -> 2 and 3.5 -> 4, where Rust's
    /// `round` would give 3 and 4. Clamped into the grid either side.
    #[test]
    fn the_mask_lookup_rounds_half_to_even_and_clamps() {
        assert_eq!(mask_index(2.5, 0.0, 10.0, 11), 2);
        assert_eq!(mask_index(3.5, 0.0, 10.0, 11), 4);
        assert_eq!(mask_index(2.4, 0.0, 10.0, 11), 2);
        assert_eq!(mask_index(-5.0, 0.0, 10.0, 11), 0);
        assert_eq!(mask_index(50.0, 0.0, 10.0, 11), 10);
    }
}
