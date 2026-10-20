//! A scene package (`smartscene-v1`) built from a run's replay metadata, in
//! Rust: what `argusnet build-scene` (`world/scene_loader.py`,
//! `build_scene_from_replay`) writes, so the viewer can build a scene without
//! a Python interpreter (D2).
//!
//! Every input comes from the replay's `meta`, which a Rust run writes in the
//! same form as the Python one (`ScenarioBundle.presentation`):
//!
//! - `terrain.viewer_mesh`: a height grid, `heights_m[rows][cols]`, on
//!   evenly spaced x (cols) and y (rows). It becomes `terrain/terrain-base.glb`
//!   with the same normals (central differences) and the same vertex shading
//!   (height gradient, value noise, slope-to-rock, contour lines, lighting).
//! - `occluding_objects`: footprints (nested `[[x, y], ...]` or flat) and
//!   `cylinder-v1` columns. They become `overlays/{buildings,walls,vegetation}.glb`,
//!   extruded with the same ground-contact rule: the base goes down to the
//!   lowest terrain under the footprint and the top at least 3 m above its
//!   highest. Roofs are ear-clipped.
//! - `metadata/style.json`, `metadata/environment.json` and
//!   `scene_manifest.json` carry the same fields, with `source_kind:
//!   "synthetic"` so the viewer treats the scene as one it can rebuild.
//!
//! Every triangle is wound counter-clockwise seen from its outside, so it
//! faces the way its normals point: the terrain faces up. The Python writer
//! winds the same way, triangle for triangle (until October 2026 both wound
//! the terrain facing down, and the viewer's exposure was set to make up for
//! the unlit ground that gave; see `docs/KNOWN_GAPS.md`).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::belief_mesh::MeshData;
use crate::scene_export::write_glb_parts;

/// The top of an obstacle is at least this far above the highest terrain
/// under it (`GROUND_CONTACT_TOP_PAD_M`).
pub const GROUND_CONTACT_TOP_PAD_M: f64 = 3.0;

/// What [`build_scene`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneBuild {
    pub root: PathBuf,
    /// Layer ids, in manifest order.
    pub layers: Vec<String>,
    pub triangles: usize,
}

/// A replay's terrain height grid, sampled as the Python builder samples it.
#[derive(Debug, Clone)]
pub struct HeightGrid {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Row-major: `heights[row][col]`, rows along y.
    pub heights: Vec<Vec<f64>>,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![a],
        _ => (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

impl HeightGrid {
    pub fn from_viewer_mesh(mesh: &Value) -> Result<Self, String> {
        let number = |key: &str| {
            mesh.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("terrain.viewer_mesh.{key} is missing"))
        };
        let rows = number("rows")? as usize;
        let cols = number("cols")? as usize;
        if rows < 2 || cols < 2 {
            return Err(format!(
                "terrain.viewer_mesh is {rows} x {cols}; need 2 x 2"
            ));
        }
        let heights: Vec<Vec<f64>> = mesh
            .get("heights_m")
            .and_then(Value::as_array)
            .ok_or("terrain.viewer_mesh.heights_m is missing")?
            .iter()
            .map(|row| {
                row.as_array()
                    .map(|row| row.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
                    .unwrap_or_default()
            })
            .collect();
        if heights.len() != rows || heights.iter().any(|row| row.len() != cols) {
            return Err(format!(
                "terrain.viewer_mesh.heights_m is not {rows} rows of {cols}"
            ));
        }
        Ok(Self {
            x: linspace(number("x_min_m")?, number("x_max_m")?, cols),
            y: linspace(number("y_min_m")?, number("y_max_m")?, rows),
            heights,
        })
    }

    fn rows(&self) -> usize {
        self.y.len()
    }

    fn cols(&self) -> usize {
        self.x.len()
    }

    /// Bilinear height at `(x, y)`, clamped to the grid
    /// (`height_function_from_mesh`).
    pub fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        let fractional = |values: &[f64], v: f64| -> f64 {
            // `np.interp(v, values, arange)`, values increasing and evenly spaced.
            let n = values.len();
            if v <= values[0] {
                return 0.0;
            }
            if v >= values[n - 1] {
                return (n - 1) as f64;
            }
            let step = (values[n - 1] - values[0]) / (n - 1) as f64;
            ((v - values[0]) / step).clamp(0.0, (n - 1) as f64)
        };
        let tx = fractional(&self.x, x_m);
        let ty = fractional(&self.y, y_m);
        let col = (tx.floor() as usize).min(self.cols() - 2);
        let row = (ty.floor() as usize).min(self.rows() - 2);
        let (ax, ay) = (tx - col as f64, ty - row as f64);
        let h = &self.heights;
        let z0 = h[row][col] + (h[row][col + 1] - h[row][col]) * ax;
        let z1 = h[row + 1][col] + (h[row + 1][col + 1] - h[row + 1][col]) * ax;
        z0 + (z1 - z0) * ay
    }

    /// Per-vertex normals by central differences (`grid_normals`).
    fn normals(&self) -> Vec<[f32; 3]> {
        let (rows, cols) = (self.rows(), self.cols());
        let mut out = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let (r0, r1) = (row.saturating_sub(1), (row + 1).min(rows - 1));
            for col in 0..cols {
                let (c0, c1) = (col.saturating_sub(1), (col + 1).min(cols - 1));
                let dzdx = (self.heights[row][c1] - self.heights[row][c0])
                    / (self.x[c1] - self.x[c0]).max(1.0e-6);
                let dzdy = (self.heights[r1][col] - self.heights[r0][col])
                    / (self.y[r1] - self.y[r0]).max(1.0e-6);
                out.push(normalize([-dzdx, -dzdy, 1.0]));
            }
        }
        out
    }
}

fn normalize(v: [f64; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length <= 1.0e-9 {
        [0.0; 3]
    } else {
        [
            (v[0] / length) as f32,
            (v[1] / length) as f32,
            (v[2] / length) as f32,
        ]
    }
}

/// `_hash_2d`: a deterministic hash in [-1, 1].
fn hash_2d(ix: i64, iy: i64) -> f64 {
    let h = (ix
        .wrapping_mul(374_761_393)
        .wrapping_add(iy.wrapping_mul(668_265_263))) as u64
        & 0xFFFF_FFFF;
    let h = ((h ^ (h >> 13)).wrapping_mul(1_274_126_177)) & 0xFFFF_FFFF;
    (h & 0xFFFF) as f64 / 32768.0 - 1.0
}

/// `_value_noise_2d`: smooth value noise in [-1, 1].
fn value_noise_2d(x: f64, y: f64) -> f64 {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - ix as f64, y - iy as f64);
    let fx = fx * fx * (3.0 - 2.0 * fx);
    let fy = fy * fy * (3.0 - 2.0 * fy);
    let (c00, c10) = (hash_2d(ix, iy), hash_2d(ix + 1, iy));
    let (c01, c11) = (hash_2d(ix, iy + 1), hash_2d(ix + 1, iy + 1));
    let a = c00 + (c10 - c00) * fx;
    let b = c01 + (c11 - c01) * fx;
    a + (b - a) * fy
}

/// `shade_terrain_colors` without land cover: the colours the Python builder
/// bakes into the terrain, written as it writes them.
fn shade_terrain(grid: &HeightGrid, normals: &[[f32; 3]]) -> Vec<[f32; 4]> {
    let light = normalize([0.45, 0.25, 0.86]);
    let fill = normalize([-0.3, 0.4, 0.55]);
    let dot = |a: [f32; 3], b: [f32; 3]| (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) as f64;
    let (low, mid, high, rock) = (
        [0.45, 0.65, 0.32],
        [0.72, 0.68, 0.45],
        [0.55, 0.50, 0.44],
        [0.52, 0.48, 0.42],
    );
    let all = grid.heights.iter().flatten();
    let min_h = all.clone().copied().fold(f64::INFINITY, f64::min);
    let max_h = all.copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max_h - min_h).max(1.0);
    let contour = (span / 6.0).max(2.0);
    let mut out = Vec::with_capacity(normals.len());
    for (row, &y) in grid.y.iter().enumerate() {
        for (col, &x) in grid.x.iter().enumerate() {
            let h = grid.heights[row][col];
            let normal = normals[row * grid.cols() + col];
            let t = (h - min_h) / span;
            let mut c: [f64; 3] = if t < 0.5 {
                std::array::from_fn(|k| low[k] + (mid[k] - low[k]) * t * 2.0)
            } else {
                std::array::from_fn(|k| mid[k] + (high[k] - mid[k]) * (t - 0.5) * 2.0)
            };
            let noise = value_noise_2d(x / 41.7, y / 41.7) * 0.08
                + value_noise_2d(x / 12.3, y / 12.3) * 0.04
                + value_noise_2d(x / 24.0 + 7.3, y / 24.0 + 3.1) * 0.05;
            for v in &mut c {
                *v += noise;
            }
            let slope = 1.0 - normal[2] as f64;
            let blend = (slope * 3.0).min(1.0) * 0.55;
            for k in 0..3 {
                c[k] = c[k] * (1.0 - blend) + rock[k] * blend;
            }
            let distance = ((h - min_h).rem_euclid(contour) - contour * 0.5).abs();
            let factor = 1.0 - 0.06 * (1.0 - distance / (contour * 0.08)).max(0.0);
            let shade =
                0.65 + 0.25 * dot(normal, light).max(0.0) + dot(normal, fill).max(0.0) * 0.15;
            out.push([
                (c[0] * factor * shade).clamp(0.0, 1.0) as f32,
                (c[1] * factor * shade).clamp(0.0, 1.0) as f32,
                (c[2] * factor * shade).clamp(0.0, 1.0) as f32,
                1.0,
            ]);
        }
    }
    out
}

/// The terrain as one mesh, every triangle facing up (counter-clockwise seen
/// from above), as the Python builder winds it.
pub fn terrain_mesh(grid: &HeightGrid) -> MeshData {
    let (rows, cols) = (grid.rows(), grid.cols());
    let normals = grid.normals();
    let colours = shade_terrain(grid, &normals);
    let positions = grid
        .y
        .iter()
        .enumerate()
        .flat_map(|(row, &y)| {
            grid.x
                .iter()
                .enumerate()
                .map(move |(col, &x)| [x as f32, y as f32, grid.heights[row][col] as f32])
        })
        .collect();
    let mut indices = Vec::with_capacity((rows - 1) * (cols - 1) * 6);
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let a = (row * cols + col) as u32;
            let b = a + 1;
            let c = a + cols as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    MeshData {
        positions,
        normals,
        colours,
        indices,
        ..MeshData::default()
    }
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    let n = points.len();
    0.5 * (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum::<f64>()
}

fn point_in_triangle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let sub = |u: [f64; 2], v: [f64; 2]| [u[0] - v[0], u[1] - v[1]];
    let dot = |u: [f64; 2], v: [f64; 2]| u[0] * v[0] + u[1] * v[1];
    let (v0, v1, v2) = (sub(c, a), sub(b, a), sub(p, a));
    let (d00, d01, d02, d11, d12) = (
        dot(v0, v0),
        dot(v0, v1),
        dot(v0, v2),
        dot(v1, v1),
        dot(v1, v2),
    );
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1.0e-12 {
        return false;
    }
    let u = (d11 * d02 - d01 * d12) / denom;
    let v = (d00 * d12 - d01 * d02) / denom;
    u >= -1.0e-6 && v >= -1.0e-6 && u + v <= 1.0 + 1.0e-6
}

/// Ear clipping (`triangulate_polygon`): counter-clockwise triangles over a
/// simple polygon, whichever way it is wound.
pub fn triangulate_polygon(points: &[[f64; 2]]) -> Vec<u32> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let clockwise = polygon_area(points) < 0.0;
    let mut vertices: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::new();
    let mut guard = 0;
    while vertices.len() > 2 && guard < n * n {
        let m = vertices.len();
        let ear = (0..m).find(|&index| {
            let (p, c, x) = (
                vertices[(index + m - 1) % m],
                vertices[index],
                vertices[(index + 1) % m],
            );
            let (a, b, cc) = (points[p], points[c], points[x]);
            let cross = (b[0] - a[0]) * (cc[1] - b[1]) - (b[1] - a[1]) * (cc[0] - b[0]);
            let convex = if clockwise {
                cross < -1.0e-9
            } else {
                cross > 1.0e-9
            };
            convex
                && !vertices
                    .iter()
                    .filter(|&&o| o != p && o != c && o != x)
                    .any(|&o| point_in_triangle(points[o], a, b, cc))
        });
        let Some(index) = ear else { break };
        let (p, c, x) = (
            vertices[(index + m - 1) % m],
            vertices[index],
            vertices[(index + 1) % m],
        );
        if clockwise {
            triangles.extend_from_slice(&[p as u32, x as u32, c as u32]);
        } else {
            triangles.extend_from_slice(&[p as u32, c as u32, x as u32]);
        }
        vertices.remove(index);
        guard += 1;
    }
    triangles
}

/// Per-vertex normals accumulated from triangle faces (`triangle_normals`).
fn triangle_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![[0.0f64; 3]; positions.len()];
    for tri in indices.chunks(3) {
        let p = |i: u32| positions[i as usize].map(f64::from);
        let (a, b, c) = (p(tri[0]), p(tri[1]), p(tri[2]));
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if length > 1.0e-9 {
            n = n.map(|x| x / length);
        }
        for &i in tri {
            for k in 0..3 {
                sums[i as usize][k] += n[k];
            }
        }
    }
    sums.into_iter().map(normalize).collect()
}

/// A footprint extruded from `base_z_m` to `top_z_m` (`mesh_from_extrusion`):
/// an ear-clipped roof facing up and one quad per edge facing out.
pub fn extrusion_mesh(
    footprint: &[[f64; 2]],
    base_z_m: f64,
    top_z_m: f64,
    colour: [f32; 4],
) -> MeshData {
    let mut mesh = MeshData::default();
    if footprint.len() < 3 || top_z_m <= base_z_m {
        return mesh;
    }
    let roof = triangulate_polygon(footprint);
    if !roof.is_empty() {
        for p in footprint {
            mesh.positions
                .push([p[0] as f32, p[1] as f32, top_z_m as f32]);
            mesh.normals.push([0.0, 0.0, 1.0]);
            mesh.colours.push(colour);
        }
        mesh.indices.extend(roof);
    }
    // `[sb, eb, et]` per edge, as the Python builder writes it: outward for
    // a counter-clockwise footprint, so a clockwise one is walked backwards.
    // Outward walls are what lets a translucent layer cull back faces.
    let ccw: Vec<[f64; 2]> = if polygon_area(footprint) < 0.0 {
        footprint.iter().rev().copied().collect()
    } else {
        footprint.to_vec()
    };
    let wall_start = mesh.positions.len();
    let mut wall_positions = Vec::new();
    let mut wall_indices = Vec::new();
    for i in 0..ccw.len() {
        let (s, e) = (ccw[i], ccw[(i + 1) % ccw.len()]);
        let base = wall_positions.len() as u32;
        wall_positions.extend_from_slice(&[
            [s[0] as f32, s[1] as f32, base_z_m as f32],
            [e[0] as f32, e[1] as f32, base_z_m as f32],
            [e[0] as f32, e[1] as f32, top_z_m as f32],
            [s[0] as f32, s[1] as f32, top_z_m as f32],
        ]);
        wall_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let wall_normals = triangle_normals(&wall_positions, &wall_indices);
    mesh.colours
        .extend(std::iter::repeat_n(colour, wall_positions.len()));
    mesh.positions.extend(wall_positions);
    mesh.normals.extend(wall_normals);
    mesh.indices
        .extend(wall_indices.into_iter().map(|i| i + wall_start as u32));
    mesh
}

/// `polygon` cut to where `f(x, y) = a x + b y + c <= 0` (Sutherland-Hodgman
/// against one half-plane).
fn clip_to_half_plane(polygon: &[[f64; 2]], [a, b, c]: [f64; 3]) -> Vec<[f64; 2]> {
    let f = |p: [f64; 2]| a * p[0] + b * p[1] + c;
    let mut out = Vec::with_capacity(polygon.len() + 2);
    for i in 0..polygon.len() {
        let (p, q) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        let (fp, fq) = (f(p), f(q));
        if fp <= 0.0 {
            out.push(p);
        }
        if (fp < 0.0 && fq > 0.0) || (fp > 0.0 && fq < 0.0) {
            let t = fp / (fp - fq);
            out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
        }
    }
    out.dedup_by(|x, y| (x[0] - y[0]).hypot(x[1] - y[1]) < 1.0e-9);
    out
}

/// How a roofed building is drawn (R1): its facade and roof colours, by
/// material class, and, when the data gives its storeys, what the viewer
/// needs for window rows: its own ground and its level count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingLook {
    pub facade: [f32; 4],
    pub roof: [f32; 4],
    /// The building's own ground, metres: levels count up from here.
    pub base_z_m: f64,
    /// Storeys between the ground and the eave, when the data gives them.
    pub levels: Option<f64>,
}

impl BuildingLook {
    /// One colour, no window rows.
    pub fn plain(colour: [f32; 4], base_z_m: f64) -> Self {
        Self {
            facade: colour,
            roof: colour,
            base_z_m,
            levels: None,
        }
    }
}

/// A facade material class's colour (sRGB, as the style's): `None` for
/// `generic` or a class this does not know, which keeps the layer's own.
pub fn facade_colour(class: &str) -> Option<[f32; 3]> {
    Some(match class {
        "concrete" => [0.70, 0.70, 0.68],
        "brick" => [0.62, 0.36, 0.28],
        "glass" => [0.55, 0.68, 0.75],
        "metal" => [0.60, 0.63, 0.66],
        "wood" => [0.55, 0.42, 0.30],
        "stone" => [0.72, 0.68, 0.60],
        "plaster" => [0.86, 0.83, 0.76],
        _ => return None,
    })
}

/// A roof material class's colour, as [`facade_colour`].
pub fn roof_colour(class: &str) -> Option<[f32; 3]> {
    Some(match class {
        "tiles" => [0.60, 0.30, 0.22],
        "slate" => [0.30, 0.32, 0.36],
        "metal" => [0.55, 0.58, 0.60],
        "concrete" => [0.62, 0.62, 0.60],
        "membrane" => [0.25, 0.25, 0.27],
        "glass" => [0.55, 0.68, 0.75],
        "green" => [0.35, 0.52, 0.30],
        _ => return None,
    })
}

/// A `building-v2`: walls from `wall_base_z_m` up to the roof along every
/// edge (a gable end rises to its peak), and one facet per roof plane,
/// where that plane is the lowest: the solid the simulation intersects,
/// drawn. `footprint` and `roof` come from the model
/// ([`argusnet_world::obstacles::Obstacle::building_v2`]).
///
/// Every vertex carries texture coordinates for the viewer's window rows:
/// on a wall, its height above the building's ground in storeys and its
/// distance along the wall in metres (`TEXCOORD_0`), and the level count
/// (`TEXCOORD_1`). A roof, or a building whose storeys the data does not
/// give, carries -1 there: no windows are drawn that the data did not
/// imply.
pub fn roofed_mesh(
    footprint: &[[f64; 2]],
    wall_base_z_m: f64,
    roof: &argusnet_world::obstacles::Roof,
    look: &BuildingLook,
) -> MeshData {
    let mut mesh = MeshData::default();
    if footprint.len() < 3 || roof.planes.is_empty() {
        return mesh;
    }
    let ccw: Vec<[f64; 2]> = if polygon_area(footprint) < 0.0 {
        footprint.iter().rev().copied().collect()
    } else {
        footprint.to_vec()
    };
    let plane_at = |[a, b, c]: [f64; 3], p: [f64; 2]| a * p[0] + b * p[1] + c;
    // Roof facets.
    for (k, &plane) in roof.planes.iter().enumerate() {
        let mut facet = ccw.clone();
        for (j, &other) in roof.planes.iter().enumerate() {
            if j != k && !facet.is_empty() {
                facet = clip_to_half_plane(
                    &facet,
                    [
                        plane[0] - other[0],
                        plane[1] - other[1],
                        plane[2] - other[2],
                    ],
                );
            }
        }
        if facet.len() < 3 || polygon_area(&facet).abs() < 1.0e-6 {
            continue;
        }
        let start = mesh.positions.len() as u32;
        let normal = normalize([-plane[0], -plane[1], 1.0]);
        for p in &facet {
            mesh.positions
                .push([p[0] as f32, p[1] as f32, plane_at(plane, *p) as f32]);
            mesh.normals.push(normal);
            mesh.colours.push(look.roof);
            mesh.uvs.push([-1.0, 0.0]);
            mesh.uvs_b.push([-1.0, 0.0]);
        }
        mesh.indices
            .extend(triangulate_polygon(&facet).into_iter().map(|i| i + start));
    }
    // Walls, split wherever the roof line kinks along an edge.
    let wall_start = mesh.positions.len();
    let mut wall_positions = Vec::new();
    let mut wall_indices = Vec::new();
    let level_height = look
        .levels
        .filter(|levels| *levels > 0.0)
        .map(|levels| (roof.eave_z_m - look.base_z_m) / levels);
    let storey = |z: f64| level_height.map_or(-1.0, |h| ((z - look.base_z_m) / h) as f32);
    let levels = look
        .levels
        .filter(|_| level_height.is_some())
        .unwrap_or(-1.0) as f32;
    for i in 0..ccw.len() {
        let (s, e) = (ccw[i], ccw[(i + 1) % ccw.len()]);
        let at = |t: f64| [s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t];
        let mut cuts = vec![0.0, 1.0];
        for (j, &pj) in roof.planes.iter().enumerate() {
            for &pk in &roof.planes[j + 1..] {
                let (d0, d1) = (
                    plane_at(pj, s) - plane_at(pk, s),
                    plane_at(pj, e) - plane_at(pk, e),
                );
                if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
                    // Only where the two planes are the roof there: other
                    // planes crossing above it leave the roof line straight.
                    let t = d0 / (d0 - d1);
                    let p = at(t);
                    if (plane_at(pj, p) - roof.height_at(p[0], p[1])).abs() < 1.0e-9 {
                        cuts.push(t);
                    }
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|x, y| (*x - *y).abs() < 1.0e-9);
        let edge_m = (e[0] - s[0]).hypot(e[1] - s[1]);
        for pair in cuts.windows(2) {
            let (along_p, along_q) = ((pair[0] * edge_m) as f32, (pair[1] * edge_m) as f32);
            let (p, q) = (at(pair[0]), at(pair[1]));
            let (zp, zq) = (roof.height_at(p[0], p[1]), roof.height_at(q[0], q[1]));
            let base = wall_positions.len() as u32;
            wall_positions.extend_from_slice(&[
                [p[0] as f32, p[1] as f32, wall_base_z_m as f32],
                [q[0] as f32, q[1] as f32, wall_base_z_m as f32],
                [q[0] as f32, q[1] as f32, zq as f32],
                [p[0] as f32, p[1] as f32, zp as f32],
            ]);
            wall_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            mesh.uvs.extend_from_slice(&[
                [storey(wall_base_z_m), along_p],
                [storey(wall_base_z_m), along_q],
                [storey(zq), along_q],
                [storey(zp), along_p],
            ]);
            mesh.uvs_b.extend(std::iter::repeat_n([levels, 0.0], 4));
        }
    }
    let wall_normals = triangle_normals(&wall_positions, &wall_indices);
    mesh.colours
        .extend(std::iter::repeat_n(look.facade, wall_positions.len()));
    mesh.positions.extend(wall_positions);
    mesh.normals.extend(wall_normals);
    mesh.indices
        .extend(wall_indices.into_iter().map(|i| i + wall_start as u32));
    mesh
}

/// A style entry from `_scene_style.default_style`, for the ids a built scene
/// uses.
fn style(id: &str) -> Option<Value> {
    let (semantic_kind, hex, alpha, elevation_mode, draw_order) = match id {
        "terrain-base" => ("terrain", "A7B58B", 1.0, "terrain-draped", 0),
        "buildings" => ("buildings", "F5EFE4", 0.45, "extruded", 30),
        "vegetation" => ("landcover", "B7D48C", 0.6, "extruded", 31),
        "walls" => ("buildings", "D2D8DF", 0.50, "extruded", 32),
        "tracks" => ("tracks", "FF9A4D", 1.0, "runtime", 100),
        "truths" => ("truths", "4DD5E7", 1.0, "runtime", 101),
        "nodes" => ("nodes", "F3F0A5", 1.0, "runtime", 102),
        _ => return None,
    };
    let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap() as f64 / 255.0;
    Some(json!({
        "id": id,
        "semantic_kind": semantic_kind,
        "color_rgba": [channel(0), channel(2), channel(4), alpha],
        "opacity": alpha,
        "elevation_mode": elevation_mode,
        "draw_order": draw_order,
        "default_visibility": true,
    }))
}

fn style_colour(id: &str) -> [f32; 4] {
    let doc = style(id).expect("a known style id");
    let c = doc["color_rgba"].as_array().expect("an array");
    std::array::from_fn(|k| c[k].as_f64().unwrap_or(1.0) as f32)
}

/// A footprint as `(x, y)` pairs, from the nested or the flat form.
fn footprint_pairs(value: &Value) -> Result<Vec<[f64; 2]>, String> {
    let items = value.as_array().ok_or("footprint_xy_m is not a list")?;
    if items.first().is_some_and(Value::is_array) {
        items
            .iter()
            .map(|p| {
                let p = p.as_array().ok_or("a footprint point is not a pair")?;
                match (
                    p.first().and_then(Value::as_f64),
                    p.get(1).and_then(Value::as_f64),
                ) {
                    (Some(x), Some(y)) => Ok([x, y]),
                    _ => Err("a footprint point is not two numbers".to_string()),
                }
            })
            .collect()
    } else {
        let flat: Vec<f64> = items.iter().filter_map(Value::as_f64).collect();
        if !flat.len().is_multiple_of(2) || flat.len() != items.len() {
            return Err(format!(
                "a flat footprint must hold an even number of coordinates, got {}",
                items.len()
            ));
        }
        Ok(flat.chunks(2).map(|c| [c[0], c[1]]).collect())
    }
}

/// `_solid_mesh_elevations`.
fn solid_elevations(object: &Value, min_terrain: f64, max_terrain: f64) -> (f64, f64) {
    let get = |key: &str| object.get(key).and_then(Value::as_f64);
    let base = get("base_elevation_m").unwrap_or(min_terrain);
    let top = get("top_elevation_m").unwrap_or_else(|| {
        base + get("height_agl_m")
            .unwrap_or(12.0)
            .max(GROUND_CONTACT_TOP_PAD_M)
    });
    (
        base.min(min_terrain),
        top.max(max_terrain + GROUND_CONTACT_TOP_PAD_M),
    )
}

/// The obstacle layers, in the order their first obstacle appears.
pub fn obstacle_layers(
    objects: &[Value],
    grid: &HeightGrid,
) -> Result<Vec<(String, Vec<MeshData>)>, String> {
    let mut layers: Vec<(String, Vec<MeshData>)> = Vec::new();
    for object in objects {
        let blocker = object
            .get("blocker_type")
            .and_then(Value::as_str)
            .unwrap_or("building");
        let layer = match blocker {
            "wall" => "walls",
            "vegetation" => "vegetation",
            _ => "buildings",
        };
        let colour = style_colour(layer);
        let roofed = (object.get("kind").and_then(Value::as_str) == Some("building-v2"))
            .then(|| {
                serde_json::from_value::<argusnet_world::obstacles::ObstacleMetadata>(
                    object.clone(),
                )
                .map_err(|error| format!("building-v2: {error}"))
                .and_then(|metadata| metadata.into_obstacle().map_err(|e| e.to_string()))
            })
            .transpose()?;
        let mesh = if let Some(building) = roofed {
            // The model's own roof, walls reaching down to the lowest ground
            // under the footprint as every other solid's do.
            let argusnet_world::obstacles::ObstacleShape::Polygon { footprint_xy_m } =
                &building.shape
            else {
                return Err(format!("{}: building-v2 without a footprint", building.id));
            };
            let lowest = footprint_xy_m
                .iter()
                .map(|p| grid.height_at(p[0], p[1]))
                .fold(f64::INFINITY, f64::min);
            let roof = building.roof.as_ref().expect("building-v2 has a roof");
            let classed = |key: &str, palette: fn(&str) -> Option<[f32; 3]>| {
                object
                    .get(key)
                    .and_then(Value::as_str)
                    .and_then(palette)
                    .map_or(colour, |[r, g, b]| [r, g, b, colour[3]])
            };
            let look = BuildingLook {
                facade: classed("facade_material", facade_colour),
                roof: classed("roof_material", roof_colour),
                base_z_m: building.base_z_m,
                levels: object.get("levels").and_then(Value::as_f64),
            };
            roofed_mesh(footprint_xy_m, building.base_z_m.min(lowest), roof, &look)
        } else if let Some(footprint) = object.get("footprint_xy_m").filter(|v| v.is_array()) {
            let footprint = footprint_pairs(footprint)?;
            let heights: Vec<f64> = footprint
                .iter()
                .map(|p| grid.height_at(p[0], p[1]))
                .collect();
            let (lo, hi) = (
                heights.iter().copied().fold(f64::INFINITY, f64::min),
                heights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            let (base, top) = solid_elevations(object, lo, hi);
            extrusion_mesh(&footprint, base, top, colour)
        } else if object.get("kind").and_then(Value::as_str) == Some("cylinder-v1") {
            let get = |key: &str| {
                object
                    .get(key)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("cylinder-v1 has no {key}"))
            };
            let (r, cx, cy) = (get("radius_m")?, get("center_x_m")?, get("center_y_m")?);
            let mut heights: Vec<f64> = (0..8)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / 8.0;
                    grid.height_at(cx + r * a.cos(), cy + r * a.sin())
                })
                .collect();
            heights.push(grid.height_at(cx, cy));
            let (lo, hi) = (
                heights.iter().copied().fold(f64::INFINITY, f64::min),
                heights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            let (base, top) = solid_elevations(object, lo, hi);
            let ring: Vec<[f64; 2]> = (0..18)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / 18.0;
                    [cx + a.cos() * r, cy + a.sin() * r]
                })
                .collect();
            extrusion_mesh(&ring, base, top, colour)
        } else {
            continue;
        };
        match layers.iter_mut().find(|(id, _)| id == layer) {
            Some((_, meshes)) => meshes.push(mesh),
            None => layers.push((layer.to_string(), vec![mesh])),
        }
    }
    // One part per obstacle, kept apart: each is written as its own glTF
    // node, so the viewer depth-sorts translucent buildings one by one.
    for (_, meshes) in &mut layers {
        meshes.retain(|mesh| !mesh.is_empty());
    }
    Ok(layers)
}

/// `_normalize_scene_id`.
pub fn normalize_scene_id(candidate: &str) -> String {
    let cleaned: String = candidate
        .trim()
        .to_lowercase()
        .replace([' ', '_'], "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect();
    if cleaned.is_empty() {
        "scene".into()
    } else {
        cleaned
    }
}

/// Write the scene package for a replay's `meta` into `dir`, which must be
/// new or empty. `replay_json`, when given, is packaged as
/// `replay/replay.json` and named in the manifest. `provenance` is merged into
/// the manifest's `build`.
/// The replay major version this reader accepts.
pub const SUPPORTED_REPLAY_SCHEMA_MAJOR: &str = "2";

/// The structural checks Python's `validate_replay_document` makes before a
/// replay is packaged: a non-empty `frames` list, a `meta` object, a
/// supported `meta.schema_version` major, a finite positive `meta.dt_s`, and
/// a `meta.frame_count` (when given) equal to the number of frames. The
/// JSON-schema pass Python also runs is not repeated here.
pub fn validate_replay_document(document: &Value) -> Result<(), String> {
    let mut errors: Vec<String> = Vec::new();
    let frames = document.get("frames").and_then(Value::as_array);
    if frames.is_none_or(|frames| frames.is_empty()) {
        errors.push("Replay document must contain a non-empty frames list.".into());
    }
    match document.get("meta").and_then(Value::as_object) {
        None => errors.push("Replay document must contain a meta object.".into()),
        Some(meta) => {
            if let Some(declared) = meta.get("schema_version").filter(|v| !v.is_null()) {
                let supported = declared
                    .as_str()
                    .is_some_and(|v| v.split('.').next() == Some(SUPPORTED_REPLAY_SCHEMA_MAJOR));
                if !supported {
                    errors.push(format!(
                        "Replay document meta.schema_version {declared} is not supported; \
                         this reader reads {SUPPORTED_REPLAY_SCHEMA_MAJOR}.x."
                    ));
                }
            }
            let dt_ok = meta
                .get("dt_s")
                .and_then(Value::as_f64)
                .is_some_and(|dt| dt.is_finite() && dt > 0.0);
            if !dt_ok {
                errors.push(
                    "Replay document meta.dt_s must be a finite value greater than 0.".into(),
                );
            }
            // Present means a non-negative integer (1.0 counts, as the
            // schema's `integer` does); null, text and fractions are refused.
            if let Some(count) = meta.get("frame_count") {
                let declared = count.as_u64().or_else(|| {
                    count
                        .as_f64()
                        .filter(|c| *c >= 0.0 && c.fract() == 0.0)
                        .map(|c| c as u64)
                });
                match (declared, frames) {
                    (None, _) => errors.push(format!(
                        "Replay document meta.frame_count {count} is not a non-negative integer."
                    )),
                    (Some(declared), Some(frames)) if declared as usize != frames.len() => errors
                        .push(
                            "Replay document meta.frame_count must match the number of frames."
                                .into(),
                        ),
                    _ => {}
                }
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Replay document validation failed with {} error(s): {}",
            errors.len(),
            errors.join("; ")
        ))
    }
}

pub fn build_scene(
    dir: &Path,
    meta: &Value,
    replay_json: Option<&str>,
    provenance: Value,
) -> Result<SceneBuild, String> {
    if dir.exists()
        && std::fs::read_dir(dir)
            .map_err(|error| format!("{}: {error}", dir.display()))?
            .next()
            .is_some()
    {
        return Err(format!(
            "{} is not empty; build into a new directory",
            dir.display()
        ));
    }
    if let Some(text) = replay_json {
        let document: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
        validate_replay_document(&document)?;
    }
    let terrain = meta
        .get("terrain")
        .filter(|t| t.get("viewer_mesh").is_some_and(Value::is_object))
        .ok_or("replay metadata does not include terrain.viewer_mesh")?;
    let grid = HeightGrid::from_viewer_mesh(&terrain["viewer_mesh"])?;
    let objects: Vec<Value> = meta
        .get("occluding_objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let scene_id = normalize_scene_id(
        meta.get("scenario_name")
            .and_then(Value::as_str)
            .unwrap_or("synthetic-scene"),
    );
    let bounds = meta
        .get("environment_bounds_m")
        .or_else(|| terrain.get("xy_bounds_m"))
        .cloned()
        .ok_or("replay metadata has no environment_bounds_m or terrain.xy_bounds_m")?;
    let crs_id = meta.get("crs_id").and_then(Value::as_str);
    let runtime_crs = json!({
        "source_crs_id": crs_id.unwrap_or("local-synthetic"),
        "runtime_crs_id": crs_id.unwrap_or("local-enu"),
        "origin_geodetic": {"lat_deg": null, "lon_deg": null, "h_m": null},
        "xy_units": "meters",
        "z_datum": "local",
    });

    let mut layers = vec![("terrain-base".to_string(), vec![terrain_mesh(&grid)])];
    layers.extend(obstacle_layers(&objects, &grid)?);

    let mkdir = |path: PathBuf| {
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))
    };
    mkdir(dir.join("metadata"))?;
    let mut manifest_layers = Vec::new();
    let mut written = Vec::new();
    let mut triangles = 0;
    for (id, parts) in &layers {
        let parts: Vec<&MeshData> = parts.iter().filter(|mesh| !mesh.is_empty()).collect();
        if parts.is_empty() {
            continue;
        }
        let terrain_layer = id == "terrain-base";
        let relative = format!(
            "{}/{id}.glb",
            if terrain_layer { "terrain" } else { "overlays" }
        );
        mkdir(dir.join(if terrain_layer { "terrain" } else { "overlays" }))?;
        write_glb_parts(&dir.join(&relative), &parts, id)?;
        let semantic_kind = style(id).expect("a known style id")["semantic_kind"].clone();
        manifest_layers.push(json!({
            "id": id,
            "kind": if terrain_layer { "terrain" } else { "overlay" },
            "semantic_kind": semantic_kind,
            "asset_path": relative,
            "style_id": id,
        }));
        written.push(id.clone());
        triangles += parts
            .iter()
            .map(|mesh| mesh.indices.len() / 3)
            .sum::<usize>();
    }

    let replay_path = match replay_json {
        Some(text) => {
            mkdir(dir.join("replay"))?;
            std::fs::write(dir.join("replay/replay.json"), text)
                .map_err(|error| format!("replay/replay.json: {error}"))?;
            Some("replay/replay.json")
        }
        None => None,
    };
    let style_ids = written
        .iter()
        .map(String::as_str)
        .chain(["tracks", "truths", "nodes"]);
    let styles: Vec<Value> = style_ids.filter_map(style).collect();
    let terrain_summary = json!({
        "kind": terrain.get("kind"),
        "base_resolution_m": terrain.get("base_resolution_m"),
        "min_height_m": terrain.get("min_height_m"),
        "max_height_m": terrain.get("max_height_m"),
    });
    let mut build = json!({
        "generated_at_utc": chrono::Utc::now().to_rfc3339(),
        "source_kind": "synthetic",
        "tool": "argusnet_sim::scene_build",
    });
    if let (Some(build), Some(extra)) = (build.as_object_mut(), provenance.as_object()) {
        build.extend(extra.clone());
    }
    let documents = [
        (
            "metadata/style.json",
            json!({"style_version": "smartstyle-v1", "layers": styles}),
        ),
        (
            "metadata/environment.json",
            json!({
                "source_kind": "synthetic",
                "scene_id": scene_id,
                "source_crs_id": runtime_crs["source_crs_id"],
                "runtime_crs": runtime_crs,
                "bounds_xy_m": bounds,
                "terrain_summary": terrain_summary,
                "occluding_objects": objects
                    .iter()
                    .map(|object| {
                        let mut object = object.clone();
                        if let Some(pairs) = object
                            .get("footprint_xy_m")
                            .and_then(|f| footprint_pairs(f).ok())
                        {
                            object["footprint_xy_m"] = json!(pairs);
                        }
                        object
                    })
                    .collect::<Vec<_>>(),
                "replay_meta": meta,
            }),
        ),
        (
            "scene_manifest.json",
            json!({
                "format_version": "smartscene-v1",
                "scene_id": scene_id,
                "bounds_xy_m": bounds,
                "runtime_crs": runtime_crs,
                "source_crs_id": runtime_crs["source_crs_id"],
                "layers": manifest_layers,
                "replay": replay_path.map(|path| json!({"path": path})),
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json",
                },
                "build": build,
            }),
        ),
    ];
    for (path, value) in documents {
        let text = serde_json::to_vec_pretty(&value).map_err(|error| error.to_string())?;
        std::fs::write(dir.join(path), text)
            .map_err(|error| format!("{}: {error}", dir.join(path).display()))?;
    }
    Ok(SceneBuild {
        root: dir.to_path_buf(),
        layers: written,
        triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> Value {
        // A 3 x 3 grid rising 10 m per column, one square building and one
        // cylinder of vegetation.
        json!({
            "scenario_name": "Test Scene_1",
            "crs_id": "local-enu",
            "environment_bounds_m": {"x_min_m": 0.0, "x_max_m": 200.0, "y_min_m": 0.0, "y_max_m": 200.0},
            "terrain": {
                "kind": "tiled-heightmap-v1",
                "base_resolution_m": 5.0,
                "min_height_m": 0.0,
                "max_height_m": 20.0,
                "viewer_mesh": {
                    "x_min_m": 0.0, "x_max_m": 200.0, "y_min_m": 0.0, "y_max_m": 200.0,
                    "rows": 3, "cols": 3,
                    "heights_m": [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]
                }
            },
            "occluding_objects": [
                {"blocker_type": "building", "footprint_xy_m": [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]],
                 "base_elevation_m": 5.0, "top_elevation_m": 30.0},
                {"blocker_type": "vegetation", "kind": "cylinder-v1", "center_x_m": 100.0, "center_y_m": 100.0,
                 "radius_m": 10.0, "height_agl_m": 8.0}
            ]
        })
    }

    #[test]
    fn the_terrain_samples_like_the_python_builder_and_faces_up() {
        let grid = HeightGrid::from_viewer_mesh(&meta()["terrain"]["viewer_mesh"]).unwrap();
        assert_eq!(grid.height_at(50.0, 10.0), 5.0);
        assert_eq!(grid.height_at(-50.0, 10.0), 0.0);
        let mesh = terrain_mesh(&grid);
        assert_eq!(mesh.positions.len(), 9);
        assert_eq!(mesh.indices.len(), 2 * 2 * 6);
        // `[a, b, c, b, d, c]` per quad, c being the next row.
        assert_eq!(&mesh.indices[..6], &[0, 1, 3, 1, 4, 3]);
        assert!(mesh.normals.iter().all(|n| n[2] > 0.0));
        // Every triangle is counter-clockwise seen from above: it faces up,
        // the way its normals point.
        for triangle in mesh.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[triangle[i] as usize]);
            let cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(cross_z > 0.0, "{triangle:?} faces down");
        }
    }

    #[test]
    fn noise_and_hash_match_the_python_reference() {
        // Values from `_hash_2d` and `_value_noise_2d` in `_scene_geometry.py`.
        assert_eq!(hash_2d(0, 0), -1.0);
        assert!((hash_2d(1, 2) - (python_hash(1, 2))).abs() < 1e-12);
        assert!((hash_2d(-3, 7) - (python_hash(-3, 7))).abs() < 1e-12);
    }

    /// The Python hash, transcribed with Python's unbounded integers.
    fn python_hash(ix: i64, iy: i64) -> f64 {
        let h = (ix as i128 * 374_761_393 + iy as i128 * 668_265_263).rem_euclid(1 << 32) as u64;
        let h = (((h ^ (h >> 13)) as u128 * 1_274_126_177) % (1u128 << 32)) as u64;
        (h & 0xFFFF) as f64 / 32768.0 - 1.0
    }

    #[test]
    fn a_clockwise_footprint_still_gets_outward_walls() {
        // Translucent layers cull back faces, so a wall facing in would vanish
        // from outside.
        let cw = [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]];
        let mesh = extrusion_mesh(&cw, 0.0, 5.0, [1.0, 1.0, 1.0, 0.5]);
        let mut walls = 0;
        for triangle in mesh.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[triangle[i] as usize]);
            let (u, v) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let face = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            if face[2].abs() > 1e-6 {
                continue;
            }
            walls += 1;
            let centre = [
                (a[0] + b[0] + c[0]) / 3.0 - 5.0,
                (a[1] + b[1] + c[1]) / 3.0 - 5.0,
            ];
            assert!(
                face[0] * centre[0] + face[1] * centre[1] > 0.0,
                "{triangle:?} faces in"
            );
        }
        assert_eq!(walls, 8);
    }

    /// Projected area of a mesh's upward-facing triangles.
    fn roof_area(mesh: &MeshData) -> f64 {
        mesh.indices
            .chunks(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                0.5 * (((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) as f64)
            })
            .filter(|area| *area > 0.0)
            .sum()
    }

    /// A building-v2 is drawn as the model's solid: its roof facets tile the
    /// footprint and lie on the model's roof, and a gable end's wall rises
    /// to the ridge (split where the roof line turns).
    #[test]
    fn a_roofed_building_is_drawn_as_the_model_solid() {
        use argusnet_world::obstacles::{Obstacle, ObstacleShape, RoofShape};
        let footprint = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        for (shape, walls) in [
            (RoofShape::Gabled, 6),
            (RoofShape::Hipped, 4),
            (RoofShape::Pyramidal, 4),
            (RoofShape::Flat, 4),
        ] {
            let height = if shape == RoofShape::Flat { 0.0 } else { 5.0 };
            let b = Obstacle::building_v2("b", footprint.clone(), 2.0, 10.0, shape, height, None)
                .unwrap();
            let roof = b.roof.as_ref().unwrap();
            let ObstacleShape::Polygon { footprint_xy_m } = &b.shape else {
                unreachable!()
            };
            let mesh = roofed_mesh(
                footprint_xy_m,
                0.0,
                roof,
                &BuildingLook::plain([1.0; 4], 2.0),
            );
            assert!(
                (roof_area(&mesh) - 200.0).abs() < 1e-3,
                "{shape:?}: {}",
                roof_area(&mesh)
            );
            // Every roof vertex on the model's roof; the walls from the base.
            let mut wall_quads = 0;
            for chunk in mesh.positions.chunks(1) {
                let p = chunk[0];
                if p[2] == 0.0 {
                    wall_quads += 1;
                    continue;
                }
                let z = roof.height_at(p[0] as f64, p[1] as f64);
                assert!((p[2] as f64 - z).abs() < 1e-3, "{shape:?}: {p:?} vs {z}");
            }
            assert_eq!(wall_quads / 2, walls, "{shape:?}: wall pieces");
            let top = mesh.positions.iter().map(|p| p[2]).fold(f32::MIN, f32::max);
            assert!(
                (top as f64 - b.top_z_m).abs() < 1e-3,
                "{shape:?}: top {top}"
            );
        }
    }

    /// Classes colour the walls and the roof; storeys reach the level count
    /// at the eave and only on walls; a building without levels, or a roof,
    /// carries -1 (no windows); and only such a part writes texture
    /// coordinates into the GLB.
    #[test]
    fn a_roofed_building_carries_its_look() {
        let meta = meta();
        let grid = HeightGrid::from_viewer_mesh(&meta["terrain"]["viewer_mesh"]).unwrap();
        let object = |levels: Option<f64>| {
            let mut o = json!({
                "object_id": "b", "kind": "building-v2", "blocker_type": "building",
                "footprint_xy_m": [[90.0, 90.0], [110.0, 90.0], [110.0, 100.0], [90.0, 100.0]],
                "base_elevation_m": 20.0, "eave_elevation_m": 29.0, "top_elevation_m": 33.0,
                "roof_shape": "gabled", "facade_material": "brick", "roof_material": "slate",
            });
            if let Some(levels) = levels {
                o["levels"] = json!(levels);
            }
            o
        };
        let layers = obstacle_layers(&[object(Some(3.0))], &grid).unwrap();
        let mesh = &layers[0].1[0];
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert_eq!(mesh.uvs_b.len(), mesh.positions.len());
        let brick = facade_colour("brick").unwrap();
        let slate = roof_colour("slate").unwrap();
        for ((p, n), (c, (uv, uv_b))) in mesh
            .positions
            .iter()
            .zip(&mesh.normals)
            .zip(mesh.colours.iter().zip(mesh.uvs.iter().zip(&mesh.uvs_b)))
        {
            if n[2].abs() > 0.5 {
                // A roof facet.
                assert_eq!(&c[..3], &slate[..]);
                assert_eq!((uv[0], uv_b[0]), (-1.0, -1.0));
            } else {
                assert_eq!(&c[..3], &brick[..]);
                assert_eq!(uv_b[0], 3.0);
                // Three storeys between 20 and 29 m: a storey every 3 m.
                assert!((uv[0] - (p[2] - 20.0) / 3.0).abs() < 1e-4, "{p:?} {uv:?}");
            }
        }
        // At the eave, the third storey ends.
        assert!(mesh
            .uvs
            .iter()
            .zip(&mesh.positions)
            .any(|(uv, p)| p[2] == 29.0 && (uv[0] - 3.0).abs() < 1e-4));
        let unknown = obstacle_layers(&[object(None)], &grid).unwrap();
        assert!(unknown[0].1[0].uvs.iter().all(|uv| uv[0] == -1.0));

        // In the file: texture coordinates on this part only.
        let dir = std::env::temp_dir().join(format!("argusnet-look-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("parts.glb");
        let plain = MeshData {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            colours: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
            ..MeshData::default()
        };
        crate::scene_export::write_glb_parts(&path, &[&plain, mesh], "parts").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let gltf: Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
        let attributes = |i: usize| gltf["meshes"][i]["primitives"][0]["attributes"].clone();
        assert!(attributes(0).get("TEXCOORD_0").is_none());
        assert!(attributes(1).get("TEXCOORD_0").is_some());
        assert!(attributes(1).get("TEXCOORD_1").is_some());
    }

    #[test]
    fn building_v2_metadata_is_drawn_with_its_roof() {
        let meta = meta();
        let grid = HeightGrid::from_viewer_mesh(&meta["terrain"]["viewer_mesh"]).unwrap();
        let object = json!({
            "object_id": "roofed", "kind": "building-v2", "blocker_type": "building",
            "footprint_xy_m": [[90.0, 90.0], [110.0, 90.0], [110.0, 100.0], [90.0, 100.0]],
            "base_elevation_m": 20.0, "eave_elevation_m": 28.0, "top_elevation_m": 33.0,
            "roof_shape": "gabled",
        });
        let layers = obstacle_layers(std::slice::from_ref(&object), &grid).unwrap();
        let mesh = &layers[0].1[0];
        let top = mesh.positions.iter().map(|p| p[2]).fold(f32::MIN, f32::max);
        let low = mesh.positions.iter().map(|p| p[2]).fold(f32::MAX, f32::min);
        assert_eq!(top, 33.0, "the ridge, not a padded prism top");
        assert!(low <= 20.0, "walls reach the ground");
        let mut bad = object;
        bad["roof_shape"] = json!("mansard");
        assert!(obstacle_layers(&[bad], &grid)
            .unwrap_err()
            .contains("mansard"));
    }

    #[test]
    fn obstacles_reach_the_ground_and_clear_it_and_ccw_walls_face_out() {
        let meta = meta();
        let grid = HeightGrid::from_viewer_mesh(&meta["terrain"]["viewer_mesh"]).unwrap();
        let layers = obstacle_layers(meta["occluding_objects"].as_array().unwrap(), &grid).unwrap();
        let ids: Vec<&str> = layers.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["buildings", "vegetation"]);
        let building = &layers[0].1[0];
        let zs: Vec<f32> = building.positions.iter().map(|p| p[2]).collect();
        let (lo, hi) = (
            zs.iter().copied().fold(f32::MAX, f32::min),
            zs.iter().copied().fold(f32::MIN, f32::max),
        );
        // Terrain under the footprint runs 5..15 m: base min(5, 5), top max(30, 18).
        assert_eq!((lo, hi), (5.0, 30.0));
        // The footprint is counter-clockwise, so the walls face out.
        let wall_normals = &building.normals[4..];
        for (p, n) in building.positions[4..].iter().zip(wall_normals) {
            let out = [p[0] - 100.0, p[1] - 100.0];
            assert!(
                out[0] * n[0] + out[1] * n[1] > 0.0,
                "a wall faces in at {p:?}"
            );
        }
        let vegetation = &layers[1].1[0];
        let top = vegetation
            .positions
            .iter()
            .map(|p| p[2])
            .fold(f32::MIN, f32::max);
        // Ground under the ring runs 9..11 m, so the base is 9 and the top
        // max(9 + 8, 11 + 3) = 17.
        assert!((top - 17.0).abs() < 1e-4, "{top}");
    }

    #[test]
    fn the_package_has_what_the_viewer_and_the_python_tests_read() {
        let dir = std::env::temp_dir().join(format!("scene-build-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let built = build_scene(
            &dir,
            &meta(),
            Some(r#"{"meta":{"schema_version":"2.0","dt_s":0.5,"frame_count":1},"frames":[{}]}"#),
            json!({"replay_source": "test"}),
        )
        .unwrap();
        assert_eq!(built.layers, ["terrain-base", "buildings", "vegetation"]);
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("scene_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["format_version"], "smartscene-v1");
        assert_eq!(manifest["scene_id"], "test-scene-1");
        assert_eq!(manifest["layers"][0]["style_id"], "terrain-base");
        assert_eq!(
            manifest["layers"][1]["asset_path"],
            "overlays/buildings.glb"
        );
        assert_eq!(manifest["replay"]["path"], "replay/replay.json");
        assert_eq!(manifest["build"]["source_kind"], "synthetic");
        assert_eq!(manifest["build"]["replay_source"], "test");
        let style: Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata/style.json")).unwrap())
                .unwrap();
        let ids: Vec<&str> = style["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            [
                "terrain-base",
                "buildings",
                "vegetation",
                "tracks",
                "truths",
                "nodes"
            ]
        );
        let env: Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata/environment.json")).unwrap())
                .unwrap();
        assert_eq!(env["source_kind"], "synthetic");
        assert_eq!(
            env["occluding_objects"][0]["footprint_xy_m"][0],
            json!([50.0, 50.0])
        );
        assert!(dir.join("terrain/terrain-base.glb").is_file());
        // Not into a directory that is not empty.
        assert!(build_scene(&dir, &meta(), None, json!({})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        // A replay without a terrain mesh is refused.
        assert!(build_scene(&dir, &json!({}), None, json!({})).is_err());
        // So is one that breaks the replay contract, before anything is written.
        let error = build_scene(
            &dir,
            &meta(),
            Some(r#"{"meta":{"dt_s":0.5},"frames":[]}"#),
            json!({}),
        )
        .unwrap_err();
        assert!(error.contains("non-empty frames"), "{error}");
        assert!(!dir.exists());
    }

    #[test]
    fn replays_are_held_to_the_python_readers_contract() {
        let ok = json!({"meta": {"schema_version": "2.3", "dt_s": 0.25, "frame_count": 2}, "frames": [{}, {}]});
        assert_eq!(validate_replay_document(&ok), Ok(()));
        // No version, no frame count: both optional.
        assert_eq!(
            validate_replay_document(&json!({"meta": {"dt_s": 1}, "frames": [{}]})),
            Ok(())
        );
        // An integral float is an integer count, as the schema reads it.
        assert_eq!(
            validate_replay_document(
                &json!({"meta": {"dt_s": 1, "frame_count": 1.0}, "frames": [{}]})
            ),
            Ok(())
        );
        for (document, expected) in [
            (
                json!({"meta": {"dt_s": 1.0}, "frames": []}),
                "non-empty frames",
            ),
            (json!({"frames": [{}]}), "meta object"),
            (
                json!({"meta": {"schema_version": "3.0", "dt_s": 1.0}, "frames": [{}]}),
                "not supported",
            ),
            (
                json!({"meta": {"schema_version": 2, "dt_s": 1.0}, "frames": [{}]}),
                "not supported",
            ),
            (json!({"meta": {"dt_s": 0.0}, "frames": [{}]}), "dt_s"),
            (
                json!({"meta": {"dt_s": 1.0, "frame_count": 3}, "frames": [{}]}),
                "frame_count",
            ),
            (
                json!({"meta": {"dt_s": 1.0, "frame_count": "1"}, "frames": [{}]}),
                "non-negative integer",
            ),
            (
                json!({"meta": {"dt_s": 1.0, "frame_count": 1.5}, "frames": [{}]}),
                "non-negative integer",
            ),
            (
                json!({"meta": {"dt_s": 1.0, "frame_count": null}, "frames": [{}]}),
                "non-negative integer",
            ),
            (
                json!({"meta": {"dt_s": 1.0, "frame_count": -1}, "frames": [{}]}),
                "non-negative integer",
            ),
        ] {
            let error = validate_replay_document(&document).unwrap_err();
            assert!(error.contains(expected), "{document}: {error}");
        }
    }
}
