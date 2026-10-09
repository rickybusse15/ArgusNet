//! The map a run builds for itself (`world_knowledge: none`), drawn as it
//! grows.
//!
//! The replay's `belief_updates` rows are applied in frame order up to the
//! cursor, so the map on screen is the map the drones had at that moment.
//! Scrubbing back rebuilds it from the start. What is drawn:
//!
//! - **Known ground**: a surface over every cell with a ground return. Its
//!   corners are the mean of the neighbouring known cells, so it reads as
//!   terrain rather than tiles. The colour runs from low to high ground,
//!   washed out where a cell has few returns and solid where it has many.
//! - **New knowledge**: cells updated in the last few seconds glow, so the
//!   frontier of the map is visible as it moves.
//! - **Structures**: cells believed occupied, extruded to their highest
//!   return.
//! - **Unknown**: nothing, over a faint outline of the map's bounds. The
//!   scene's own layers (the true terrain and buildings) are hidden when such
//!   a run opens. They stay in the Layers panel, as truth, for comparison.
//! - **Sensing**: each drone's sampled range returns this frame, as rays.
//! - **Contacts**: a red ring where physics had to put a drone back.
//!
//! A readout gives the map's progress and its error against the truth, which
//! is evaluation the run itself never sees.

use std::collections::{BTreeSet, HashMap};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{egui, EguiContexts};

use crate::replay::{BeliefCellRow, BeliefGridLayout, BeliefStats, ReplayState};
use crate::schema::ScenePackage;
use crate::state::LayerVisibilityState;

/// Cells per side of a mesh chunk: a change rebuilds only its chunk.
const CHUNK: u32 = 32;
/// How long, in source seconds, a newly updated cell glows.
const FRESH_S: f32 = 2.5;

#[derive(Component)]
pub struct BeliefChunk;

#[derive(Debug, Clone, Copy)]
struct Cell {
    row: BeliefCellRow,
    updated_s: f32,
}

/// The map as of the cursor, and the entities that draw it.
#[derive(Resource, Default)]
pub struct BeliefMapView {
    layout: Option<BeliefGridLayout>,
    epoch: u64,
    cells: HashMap<(u32, u32), Cell>,
    /// Frames are applied through this source time.
    applied_through_s: Option<f32>,
    dirty: BTreeSet<(u32, u32)>,
    /// Chunks with glowing cells, and when the last glow ends.
    fresh_until: HashMap<(u32, u32), f32>,
    chunks: HashMap<(u32, u32), Entity>,
    /// The believed structures, as one mesh.
    buildings: Option<Entity>,
    buildings_dirty: bool,
    material: Option<Handle<StandardMaterial>>,
    truth_hidden: bool,
    now_s: f32,
    stats: Option<BeliefStats>,
    contacts: usize,
}

impl BeliefMapView {
    /// Whether a self-built map is being shown.
    pub fn active(&self) -> bool {
        self.layout.is_some()
    }

    fn forget(&mut self, commands: &mut Commands) {
        for (_, entity) in self.chunks.drain() {
            commands.entity(entity).despawn();
        }
        if let Some(entity) = self.buildings.take() {
            commands.entity(entity).despawn();
        }
        self.cells.clear();
        self.applied_through_s = None;
        self.dirty.clear();
        self.fresh_until.clear();
        self.stats = None;
        self.contacts = 0;
    }

    /// Apply a frame's rows. A cell glows from when it first became known,
    /// which is the map's leading edge; later refinements do not relight it.
    fn apply(&mut self, rows: &[BeliefCellRow], at_s: f32) {
        for row in rows {
            let (i, j) = row.ij();
            let previous = self.cells.get(&(i, j)).copied();
            let first_known_s = previous.map_or(at_s, |cell| cell.updated_s);
            if previous.map(|cell| cell.row.structure_top_m()) != Some(row.structure_top_m())
                && (row.structure_top_m().is_some()
                    || previous.is_some_and(|cell| cell.row.structure_top_m().is_some()))
            {
                self.buildings_dirty = true;
            }
            self.cells.insert(
                (i, j),
                Cell {
                    row: *row,
                    updated_s: first_known_s,
                },
            );
            // A cell's corners are shared with its neighbours, which may sit
            // in the next chunk over: every chunk touching it is redrawn.
            for (di, dj) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni >= 0 && nj >= 0 {
                    self.dirty.insert((ni as u32 / CHUNK, nj as u32 / CHUNK));
                }
            }
            if previous.is_none() {
                let chunk = (i / CHUNK, j / CHUNK);
                let until = self.fresh_until.entry(chunk).or_insert(0.0);
                *until = until.max(at_s + FRESH_S);
            }
        }
    }

    fn ground(&self, i: i64, j: i64) -> Option<f32> {
        if i < 0 || j < 0 {
            return None;
        }
        self.cells
            .get(&(i as u32, j as u32))
            .and_then(|cell| cell.row.ground_m())
    }

    /// A cell corner's height: the mean of the known cells that share it.
    fn corner(&self, ci: i64, cj: i64) -> Option<f32> {
        let mut sum = 0.0;
        let mut count = 0;
        for (di, dj) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            if let Some(h) = self.ground(ci + di, cj + dj) {
                sum += h;
                count += 1;
            }
        }
        (count > 0).then(|| sum / count as f32)
    }
}

/// Apply the frames up to the cursor, and hide the scene's true layers the
/// first time a self-built map appears.
pub fn sync_belief_map_system(
    mut commands: Commands,
    replay: Res<ReplayState>,
    mut view: ResMut<BeliefMapView>,
    mut layers: ResMut<LayerVisibilityState>,
    scene: Option<Res<ScenePackage>>,
) {
    let layout = replay.document.as_ref().and_then(|doc| doc.belief_grid());
    if layout != view.layout || replay.epoch() != view.epoch {
        view.forget(&mut commands);
        view.layout = layout;
        view.epoch = replay.epoch();
    }
    let (Some(_), Some(document)) = (layout, replay.document.as_ref()) else {
        return;
    };
    if !view.truth_hidden {
        if let Some(scene) = scene {
            for layer in &scene.manifest.layers {
                layers.base_layers.insert(layer.id.clone(), false);
            }
        }
        view.truth_hidden = true;
    }
    let Some(current) = replay.current_frame() else {
        return;
    };
    let now = current.timestamp_s;
    if view.applied_through_s.is_some_and(|applied| now < applied) {
        // Scrubbed back: the map at an earlier moment is rebuilt from the
        // start, since rows replace and cannot be undone.
        view.forget(&mut commands);
    }
    let from = view.applied_through_s;
    for frame in &document.frames {
        if from.is_some_and(|applied| frame.timestamp_s <= applied) {
            continue;
        }
        if frame.timestamp_s > now {
            break;
        }
        view.apply(&frame.belief_updates, frame.timestamp_s);
        view.contacts += frame.world_contacts.len();
    }
    view.applied_through_s = Some(now);
    view.stats = current
        .mapping_state
        .as_ref()
        .and_then(|mapping| mapping.belief.clone());
    view.now_s = now;
    // Chunks whose glow has run out are redrawn once more, unlit.
    let expired: Vec<(u32, u32)> = view
        .fresh_until
        .iter()
        .filter(|(_, until)| **until <= now)
        .map(|(chunk, _)| *chunk)
        .collect();
    for chunk in expired {
        view.fresh_until.remove(&chunk);
        view.dirty.insert(chunk);
    }
}

/// Rebuild the meshes of the chunks that changed.
pub fn render_belief_chunks_system(
    mut commands: Commands,
    mut view: ResMut<BeliefMapView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(layout) = view.layout else {
        return;
    };
    if view.dirty.is_empty() && !view.buildings_dirty {
        return;
    }
    // Lit like the scene's own terrain: on the main layer, where the sun is,
    // as well as the reconstruction layer the split view's right pane draws.
    let material = view
        .material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.95,
                metallic: 0.0,
                reflectance: 0.2,
                cull_mode: None,
                double_sided: true,
                ..default()
            })
        })
        .clone();
    let layers = RenderLayers::from_layers(&[0, crate::app::RECON_LAYER]);
    if std::mem::take(&mut view.buildings_dirty) {
        match (buildings_mesh(&view, layout), view.buildings) {
            (Some(mesh), Some(entity)) => {
                commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
            }
            (Some(mesh), None) => {
                let entity = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.clone()),
                        layers.clone(),
                        Visibility::Visible,
                        BeliefChunk,
                        Name::new("belief-structures"),
                    ))
                    .id();
                view.buildings = Some(entity);
            }
            (None, Some(entity)) => {
                commands.entity(entity).despawn();
                view.buildings = None;
            }
            (None, None) => {}
        }
    }
    let (low, high) = height_range(&view);
    let dirty = std::mem::take(&mut view.dirty);
    for chunk in dirty {
        let mesh = chunk_mesh(&view, layout, chunk, low, high);
        match (mesh, view.chunks.get(&chunk).copied()) {
            (Some(mesh), Some(entity)) => {
                commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
            }
            (Some(mesh), None) => {
                let entity = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.clone()),
                        layers.clone(),
                        Visibility::Visible,
                        BeliefChunk,
                        Name::new(format!("belief-chunk-{}-{}", chunk.0, chunk.1)),
                    ))
                    .id();
                view.chunks.insert(chunk, entity);
            }
            (None, Some(entity)) => {
                commands.entity(entity).despawn();
                view.chunks.remove(&chunk);
            }
            (None, None) => {}
        }
    }
}

fn height_range(view: &BeliefMapView) -> (f32, f32) {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for cell in view.cells.values() {
        if let Some(h) = cell.row.ground_m() {
            low = low.min(h);
            high = high.max(h);
        }
    }
    if low.is_finite() {
        (low, high.max(low + 1.0))
    } else {
        (0.0, 1.0)
    }
}

/// Ground colour: low ground dark green-grey, high ground pale stone,
/// washed toward slate where a cell has few returns, and toward cyan while it
/// is new. Lighting comes from the renderer; this is albedo.
fn ground_colour(height: f32, low: f32, high: f32, level: u8, glow: f32) -> [f32; 3] {
    let t = ((height - low) / (high - low)).clamp(0.0, 1.0);
    let base = [0.20 + 0.40 * t, 0.29 + 0.32 * t, 0.23 + 0.25 * t];
    let confidence = (level.clamp(1, 4) as f32 - 1.0) / 3.0;
    let washed = [0.40, 0.46, 0.52];
    let fresh = [0.40, 0.90, 1.0];
    std::array::from_fn(|k| {
        let settled = washed[k] + (base[k] - washed[k]) * (0.35 + 0.65 * confidence);
        settled * (1.0 - glow) + fresh[k] * glow
    })
}

/// A cell side: the neighbour across it, its two corners (cell-corner
/// indices for the ground, metres for structures) and its outward normal.
type CornerSide = ((i64, i64), [(i64, i64); 2], [f32; 3]);
type WallSide = ((i64, i64), [[f32; 2]; 2], [f32; 3]);

/// How far the cut edge of the known map drops below its surface, metres.
const SKIRT_M: f32 = 6.0;

/// One chunk of the known ground as a continuous surface: vertices at cell
/// corners, shared by every known cell that touches them, so the surface
/// has no seams; normals from the corner heights, so it takes the scene's
/// light like terrain; and a darker skirt dropping from every edge where the
/// known map meets the unknown, so the map reads as a cut-out of the world.
fn chunk_mesh(
    view: &BeliefMapView,
    layout: BeliefGridLayout,
    chunk: (u32, u32),
    low: f32,
    high: f32,
) -> Option<Mesh> {
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    let (i_lo, j_lo) = (chunk.0 * CHUNK, chunk.1 * CHUNK);
    let i_hi = ((chunk.0 + 1) * CHUNK).min(layout.dimensions[0]);
    let j_hi = ((chunk.1 + 1) * CHUNK).min(layout.dimensions[1]);
    let known = |i: i64, j: i64| view.ground(i, j).is_some();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colours: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut corner_index: HashMap<(i64, i64), u32> = HashMap::new();
    let corner_height = |ci: i64, cj: i64| view.corner(ci, cj);
    // A corner's colour is the mean of the known cells around it; its glow,
    // the brightest of them.
    let corner_colour = |ci: i64, cj: i64| -> [f32; 4] {
        let mut sum = [0.0f32; 3];
        let mut glow = 0.0f32;
        let mut n = 0.0;
        for (di, dj) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            let (i, j) = (ci + di, cj + dj);
            if i < 0 || j < 0 {
                continue;
            }
            if let Some(cell) = view.cells.get(&(i as u32, j as u32)) {
                if let Some(h) = cell.row.ground_m() {
                    let g =
                        ((cell.updated_s + FRESH_S - view.now_s) / FRESH_S).clamp(0.0, 1.0) * 0.8;
                    let c = ground_colour(h, low, high, cell.row.confidence_level(), 0.0);
                    for k in 0..3 {
                        sum[k] += c[k];
                    }
                    glow = glow.max(g);
                    n += 1.0;
                }
            }
        }
        let fresh = [0.40, 0.90, 1.0];
        let c: [f32; 3] = std::array::from_fn(|k| {
            let base = if n > 0.0 { sum[k] / n } else { 0.3 };
            base * (1.0 - glow) + fresh[k] * glow
        });
        [c[0], c[1], c[2], 1.0]
    };
    let mut vertex = |ci: i64,
                      cj: i64,
                      positions: &mut Vec<[f32; 3]>,
                      normals: &mut Vec<[f32; 3]>,
                      colours: &mut Vec<[f32; 4]>|
     -> u32 {
        *corner_index.entry((ci, cj)).or_insert_with(|| {
            let h = corner_height(ci, cj).unwrap_or(low);
            let around = |a: i64, b: i64| corner_height(a, b).unwrap_or(h);
            let dzdx = (around(ci + 1, cj) - around(ci - 1, cj)) / (2.0 * res);
            let dzdy = (around(ci, cj + 1) - around(ci, cj - 1)) / (2.0 * res);
            let n = Vec3::new(-dzdx, -dzdy, 1.0).normalize();
            positions.push([x0 + ci as f32 * res, y0 + cj as f32 * res, h]);
            normals.push(n.to_array());
            colours.push(corner_colour(ci, cj));
            (positions.len() - 1) as u32
        })
    };
    for j in j_lo..j_hi {
        for i in i_lo..i_hi {
            let (ii, jj) = (i as i64, j as i64);
            if !known(ii, jj) {
                continue;
            }
            let a = vertex(ii, jj, &mut positions, &mut normals, &mut colours);
            let b = vertex(ii + 1, jj, &mut positions, &mut normals, &mut colours);
            let c = vertex(ii + 1, jj + 1, &mut positions, &mut normals, &mut colours);
            let d = vertex(ii, jj + 1, &mut positions, &mut normals, &mut colours);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Skirts along the boundary of the known map, after the surface so they
    // do not share its smoothed vertices.
    for j in j_lo..j_hi {
        for i in i_lo..i_hi {
            let (ii, jj) = (i as i64, j as i64);
            if !known(ii, jj) {
                continue;
            }
            // (neighbour offset, edge corners, outward normal)
            let edges: [CornerSide; 4] = [
                ((0, -1), [(ii + 1, jj), (ii, jj)], [0.0, -1.0, 0.0]),
                ((1, 0), [(ii + 1, jj + 1), (ii + 1, jj)], [1.0, 0.0, 0.0]),
                ((0, 1), [(ii, jj + 1), (ii + 1, jj + 1)], [0.0, 1.0, 0.0]),
                ((-1, 0), [(ii, jj), (ii, jj + 1)], [-1.0, 0.0, 0.0]),
            ];
            for ((di, dj), corners, normal) in edges {
                if known(ii + di, jj + dj) {
                    continue;
                }
                let base = positions.len() as u32;
                for &(ci, cj) in &corners {
                    let h = corner_height(ci, cj).unwrap_or(low);
                    let p = [x0 + ci as f32 * res, y0 + cj as f32 * res];
                    positions.push([p[0], p[1], h]);
                    positions.push([p[0], p[1], h - SKIRT_M]);
                    for _ in 0..2 {
                        normals.push(normal);
                    }
                    colours.push([0.20, 0.22, 0.24, 1.0]);
                    colours.push([0.07, 0.08, 0.09, 1.0]);
                }
                indices.extend_from_slice(&[base, base + 1, base + 3, base, base + 3, base + 2]);
            }
        }
    }
    finish_mesh(positions, normals, colours, indices)
}

fn finish_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colours: Vec<[f32; 4]>,
    indices: Vec<u32>,
) -> Option<Mesh> {
    if indices.is_empty() {
        return None;
    }
    let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[1]]).collect();
    // The palette is written in sRGB; vertex colours are linear.
    let colours: Vec<[f32; 4]> = colours
        .into_iter()
        .map(|c| {
            let l = LinearRgba::from(Color::srgba(c[0], c[1], c[2], c[3]));
            [l.red, l.green, l.blue, l.alpha]
        })
        .collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

/// Every believed structure as one solid: a roof over each occupied cell at
/// its own top, and a wall on each side where the neighbour is lower or not
/// a structure. Neighbouring cells merge into one outline, so a building
/// reads as a building, not a stack of boxes.
fn buildings_mesh(view: &BeliefMapView, layout: BeliefGridLayout) -> Option<Mesh> {
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    let top = |i: i64, j: i64| -> Option<f32> {
        if i < 0 || j < 0 {
            return None;
        }
        view.cells
            .get(&(i as u32, j as u32))
            .and_then(|cell| cell.row.structure_top_m())
    };
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colours = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut quad = |corners: [[f32; 3]; 4], normal: [f32; 3], colour: [f32; 4]| {
        let base = positions.len() as u32;
        for c in corners {
            positions.push(c);
            normals.push(normal);
            colours.push(colour);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    let mut cells: Vec<(u32, u32)> = view
        .cells
        .iter()
        .filter(|(_, cell)| cell.row.structure_top_m().is_some())
        .map(|(ij, _)| *ij)
        .collect();
    cells.sort_unstable();
    for (i, j) in cells {
        let (ii, jj) = (i as i64, j as i64);
        let cell = view.cells[&(i, j)];
        let roof = cell.row.structure_top_m().unwrap_or_default();
        // The ground the walls stand on: this cell's, a neighbour's, or a
        // storey below the roof when nothing under it has been seen.
        let footing = view
            .ground(ii, jj)
            .or_else(|| {
                [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter_map(|(di, dj)| view.ground(ii + di, jj + dj))
                    .reduce(f32::min)
            })
            .unwrap_or(roof - 12.0)
            .min(roof - 1.0);
        let tint = 0.62 + 0.38 * cell.row.occupancy();
        let wall = [0.80 * tint, 0.66 * tint, 0.78 * tint, 1.0];
        let roof_colour = [0.90 * tint, 0.74 * tint, 0.86 * tint, 1.0];
        let (ax, ay) = (x0 + i as f32 * res, y0 + j as f32 * res);
        let (bx, by) = (ax + res, ay + res);
        quad(
            [
                [ax, ay, roof],
                [bx, ay, roof],
                [bx, by, roof],
                [ax, by, roof],
            ],
            [0.0, 0.0, 1.0],
            roof_colour,
        );
        let sides: [WallSide; 4] = [
            ((0, -1), [[bx, ay], [ax, ay]], [0.0, -1.0, 0.0]),
            ((1, 0), [[bx, by], [bx, ay]], [1.0, 0.0, 0.0]),
            ((0, 1), [[ax, by], [bx, by]], [0.0, 1.0, 0.0]),
            ((-1, 0), [[ax, ay], [ax, by]], [-1.0, 0.0, 0.0]),
        ];
        for ((di, dj), [p, q], normal) in sides {
            let foot = top(ii + di, jj + dj).unwrap_or(footing).max(footing);
            if foot >= roof {
                continue;
            }
            quad(
                [
                    [p[0], p[1], foot],
                    [q[0], q[1], foot],
                    [q[0], q[1], roof],
                    [p[0], p[1], roof],
                ],
                normal,
                wall,
            );
        }
    }
    finish_mesh(positions, normals, colours, indices)
}

/// The map's bounds, this frame's range returns, and any contacts.
pub fn belief_overlays_system(
    view: Res<BeliefMapView>,
    replay: Res<ReplayState>,
    mut gizmos: Gizmos,
) {
    let Some(layout) = view.layout else {
        return;
    };
    let Some(frame) = replay.current_frame() else {
        return;
    };
    let z = height_range(&view).0 - 2.0;
    let [x0, y0] = layout.origin_m;
    let x1 = x0 + layout.dimensions[0] as f32 * layout.resolution_m;
    let y1 = y0 + layout.dimensions[1] as f32 * layout.resolution_m;
    let edge = Color::srgba(0.55, 0.62, 0.66, 0.55);
    let corners = [
        Vec3::new(x0, y0, z),
        Vec3::new(x1, y0, z),
        Vec3::new(x1, y1, z),
        Vec3::new(x0, y1, z),
    ];
    for k in 0..4 {
        gizmos.line(corners[k], corners[(k + 1) % 4], edge);
    }
    let ground_ray = Color::srgba(1.0, 0.78, 0.35, 0.30);
    let ground_hit = Color::srgba(1.0, 0.82, 0.40, 0.95);
    let structure = Color::srgba(1.0, 0.45, 0.85, 0.95);
    for scan in &frame.range_scans {
        let origin = Vec3::from_array(scan.origin_m);
        for point in &scan.ground {
            let p = Vec3::from_array(*point);
            gizmos.line(origin, p, ground_ray);
            gizmos.sphere(Isometry3d::from_translation(p), 1.2, ground_hit);
        }
        for point in &scan.structure {
            let p = Vec3::from_array(*point);
            gizmos.line(origin, p, structure.with_alpha(0.45));
            gizmos.sphere(Isometry3d::from_translation(p), 1.6, structure);
        }
    }
    for contact in &frame.world_contacts {
        if let Some(node) = frame.nodes.iter().find(|n| n.node_id == contact.node_id) {
            gizmos.circle(
                Isometry3d::from_translation(Vec3::from_array(node.position)),
                12.0,
                Color::srgb(0.91, 0.42, 0.36),
            );
        }
    }
}

/// The map's progress, bottom left of the canvas.
pub fn belief_readout_ui_system(
    mut contexts: EguiContexts,
    view: Res<BeliefMapView>,
    ui_state: Res<crate::state::ViewerUiState>,
) {
    if !view.active() {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    if !crate::theme::fonts_ready(ctx) {
        return;
    }
    let origin = ui_state.canvas_origin();
    egui::Area::new(egui::Id::new("argusnet_belief_readout"))
        .fixed_pos(egui::pos2(origin[0] + 12.0, origin[1] + 104.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(
                    crate::theme::TypeRole::Utility
                        .text("SELF-BUILT MAP · NO PRIOR WORLD")
                        .color(crate::theme::ACCENT)
                        .strong(),
                );
                match &view.stats {
                    Some(stats) => {
                        ui.monospace(format!(
                            "known {:>5.1}%  {} / {} cells",
                            stats.known_fraction * 100.0,
                            stats.known_cells,
                            stats.total_cells
                        ));
                        ui.monospace(format!(
                            "ground {}",
                            stats
                                .ground_rmse_m
                                .map_or("—".to_string(), |e| format!("±{e:.1} m RMSE")),
                        ));
                        let percent = |v: Option<f32>| {
                            v.map_or("—".to_string(), |v| format!("{:.0}%", v * 100.0))
                        };
                        ui.monospace(format!(
                            "structures {} cells  found {}  right {}",
                            stats.structure_cells,
                            percent(stats.structure_recall),
                            percent(stats.structure_precision),
                        ));
                    }
                    None => {
                        ui.monospace("nothing sensed yet");
                    }
                }
                let contacts = if view.contacts == 0 {
                    "no contacts".to_string()
                } else {
                    format!("{} contacts", view.contacts)
                };
                ui.monospace(contacts);
                ui.label(
                    crate::theme::TypeRole::Utility
                        .text("error is against truth, for evaluation · truth layers hidden")
                        .color(crate::theme::TEXT_MUTED),
                );
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(i: u32, j: u32, ground_cm: Option<i32>) -> BeliefCellRow {
        BeliefCellRow(i, j, ground_cm, None, 0, 1)
    }

    #[test]
    fn corners_average_the_known_cells_around_them() {
        let mut view = BeliefMapView::default();
        view.apply(&[row(0, 0, Some(1000)), row(1, 0, Some(2000))], 0.0);
        assert_eq!(view.corner(1, 0), Some(15.0));
        assert_eq!(view.corner(0, 0), Some(10.0));
        assert_eq!(view.corner(5, 5), None);
    }

    #[test]
    fn a_later_row_replaces_the_cell_and_marks_its_chunk() {
        let mut view = BeliefMapView::default();
        view.apply(&[row(40, 3, Some(500))], 1.0);
        view.apply(&[row(40, 3, Some(700))], 2.0);
        assert_eq!(view.cells.len(), 1);
        assert_eq!(view.ground(40, 3), Some(7.0));
        assert!(view.dirty.contains(&(1, 0)));
        assert_eq!(
            view.fresh_until[&(1, 0)],
            1.0 + FRESH_S,
            "the glow runs from first sight"
        );
    }

    fn layout() -> BeliefGridLayout {
        BeliefGridLayout {
            origin_m: [0.0, 0.0],
            resolution_m: 10.0,
            dimensions: [64, 64],
        }
    }

    #[test]
    fn known_ground_is_one_surface_with_a_skirt_round_its_edge() {
        let mut view = BeliefMapView::default();
        view.apply(&[row(1, 1, Some(1200)), row(2, 1, Some(1400))], 0.0);
        let mesh = chunk_mesh(&view, layout(), (0, 0), 12.0, 14.0).expect("meshes");
        // Two cells share an edge: 6 surface corners, not 8. The outline has
        // 6 edges, each a skirt quad of 4 vertices.
        assert_eq!(mesh.count_vertices(), 6 + 6 * 4);
        assert!(chunk_mesh(&view, layout(), (1, 1), 12.0, 14.0).is_none());
    }

    #[test]
    fn a_cell_on_a_chunk_edge_redraws_its_neighbouring_chunk() {
        let mut view = BeliefMapView::default();
        view.apply(&[row(CHUNK, 5, Some(1000))], 0.0);
        assert!(view.dirty.contains(&(1, 0)));
        assert!(
            view.dirty.contains(&(0, 0)),
            "its corners are shared with chunk (0, 0)"
        );
    }

    #[test]
    fn neighbouring_structure_cells_merge_into_one_outline() {
        let mut view = BeliefMapView::default();
        let building = |i: u32| BeliefCellRow(i, 4, Some(1000), Some(4000), 90, 3);
        view.apply(&[building(3), building(4)], 0.0);
        assert!(view.buildings_dirty);
        let mesh = buildings_mesh(&view, layout()).expect("a building");
        // Two roofs and the six outer walls; no wall between the two cells.
        assert_eq!(mesh.count_vertices(), 4 * (2 + 6));
        // A lower neighbour gets a wall down to its own roof.
        view.apply(&[BeliefCellRow(5, 4, Some(1000), Some(2500), 90, 3)], 1.0);
        let mesh = buildings_mesh(&view, layout()).expect("a building");
        assert_eq!(mesh.count_vertices(), 4 * (3 + 6 + 3));
    }

    #[test]
    fn a_cell_glows_from_when_it_first_became_known() {
        let mut view = BeliefMapView::default();
        view.apply(&[row(0, 0, Some(1000))], 1.0);
        view.apply(&[row(0, 0, Some(1050))], 9.0);
        assert_eq!(view.cells[&(0, 0)].updated_s, 1.0);
    }
}
