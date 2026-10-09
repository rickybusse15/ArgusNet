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
        self.cells.clear();
        self.applied_through_s = None;
        self.dirty.clear();
        self.fresh_until.clear();
        self.stats = None;
        self.contacts = 0;
    }

    fn apply(&mut self, rows: &[BeliefCellRow], at_s: f32) {
        for row in rows {
            let (i, j) = row.ij();
            self.cells.insert(
                (i, j),
                Cell {
                    row: *row,
                    updated_s: at_s,
                },
            );
            let chunk = (i / CHUNK, j / CHUNK);
            self.dirty.insert(chunk);
            let until = self.fresh_until.entry(chunk).or_insert(0.0);
            *until = until.max(at_s + FRESH_S);
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
    if view.dirty.is_empty() {
        return;
    }
    let material = view
        .material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                cull_mode: None,
                double_sided: true,
                ..default()
            })
        })
        .clone();
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
                        RenderLayers::layer(crate::app::RECON_LAYER),
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
/// is new.
fn ground_colour(height: f32, low: f32, high: f32, level: u8, glow: f32, shade: f32) -> [f32; 4] {
    let t = ((height - low) / (high - low)).clamp(0.0, 1.0);
    let base = [0.18 + 0.42 * t, 0.26 + 0.34 * t, 0.22 + 0.26 * t];
    let confidence = (level.clamp(1, 4) as f32 - 1.0) / 3.0;
    let washed = [0.34, 0.40, 0.46];
    let fresh = [0.40, 0.90, 1.0];
    let mut rgb = [0.0; 3];
    for k in 0..3 {
        let settled = washed[k] + (base[k] - washed[k]) * (0.35 + 0.65 * confidence);
        rgb[k] = (settled * shade) * (1.0 - glow) + fresh[k] * glow;
    }
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn chunk_mesh(
    view: &BeliefMapView,
    layout: BeliefGridLayout,
    chunk: (u32, u32),
    low: f32,
    high: f32,
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colours: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    let light = Vec3::new(-0.45, -0.35, 0.82).normalize();
    for j in chunk.1 * CHUNK..((chunk.1 + 1) * CHUNK).min(layout.dimensions[1]) {
        for i in chunk.0 * CHUNK..((chunk.0 + 1) * CHUNK).min(layout.dimensions[0]) {
            let Some(cell) = view.cells.get(&(i, j)) else {
                continue;
            };
            let glow = ((cell.updated_s + FRESH_S - view.now_s) / FRESH_S).clamp(0.0, 1.0) * 0.8;
            if let Some(h) = cell.row.ground_m() {
                let (ii, jj) = (i as i64, j as i64);
                let corners = [(ii, jj), (ii + 1, jj), (ii + 1, jj + 1), (ii, jj + 1)];
                let z: Vec<f32> = corners
                    .iter()
                    .map(|&(ci, cj)| view.corner(ci, cj).unwrap_or(h))
                    .collect();
                // Baked hillshade from the quad's slope; the material is unlit.
                let dzdx = ((z[1] + z[2]) - (z[0] + z[3])) / (2.0 * res);
                let dzdy = ((z[2] + z[3]) - (z[0] + z[1])) / (2.0 * res);
                let normal = Vec3::new(-dzdx, -dzdy, 1.0).normalize();
                let shade = 0.55 + 0.6 * normal.dot(light).max(0.0);
                let colour = ground_colour(h, low, high, cell.row.confidence_level(), glow, shade);
                let base = positions.len() as u32;
                for (k, &(ci, cj)) in corners.iter().enumerate() {
                    positions.push([x0 + ci as f32 * res, y0 + cj as f32 * res, z[k] + 0.3]);
                    colours.push(colour);
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
            if let Some(top) = cell.row.structure_top_m() {
                let floor = cell.row.ground_m().unwrap_or(top - 8.0).min(top - 1.0);
                let tint = 0.55 + 0.45 * cell.row.occupancy();
                let colour = |s: f32| {
                    let c = [0.86 * tint * s, 0.62 * tint * s, 0.78 * tint * s];
                    [
                        c[0] * (1.0 - glow) + 0.40 * glow,
                        c[1] * (1.0 - glow) + 0.90 * glow,
                        c[2] * (1.0 - glow) + glow,
                        1.0,
                    ]
                };
                let inset = res * 0.08;
                let (ax, ay) = (x0 + i as f32 * res + inset, y0 + j as f32 * res + inset);
                let (bx, by) = (ax + res - 2.0 * inset, ay + res - 2.0 * inset);
                let quads: [([[f32; 3]; 4], f32); 5] = [
                    (
                        [[ax, ay, top], [bx, ay, top], [bx, by, top], [ax, by, top]],
                        1.0,
                    ),
                    (
                        [
                            [ax, ay, floor],
                            [bx, ay, floor],
                            [bx, ay, top],
                            [ax, ay, top],
                        ],
                        0.7,
                    ),
                    (
                        [
                            [bx, ay, floor],
                            [bx, by, floor],
                            [bx, by, top],
                            [bx, ay, top],
                        ],
                        0.85,
                    ),
                    (
                        [
                            [bx, by, floor],
                            [ax, by, floor],
                            [ax, by, top],
                            [bx, by, top],
                        ],
                        0.6,
                    ),
                    (
                        [
                            [ax, by, floor],
                            [ax, ay, floor],
                            [ax, ay, top],
                            [ax, by, top],
                        ],
                        0.75,
                    ),
                ];
                for (quad, s) in quads {
                    let base = positions.len() as u32;
                    for p in quad {
                        positions.push(p);
                        colours.push(colour(s));
                    }
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
    }
    if positions.is_empty() {
        return None;
    }
    let count = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 0.0, 1.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
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
        assert_eq!(view.fresh_until[&(1, 0)], 2.0 + FRESH_S);
    }

    #[test]
    fn a_chunk_of_known_ground_meshes_and_an_empty_one_does_not() {
        let mut view = BeliefMapView::default();
        let layout = BeliefGridLayout {
            origin_m: [0.0, 0.0],
            resolution_m: 10.0,
            dimensions: [64, 64],
        };
        view.apply(
            &[
                row(1, 1, Some(1200)),
                BeliefCellRow(2, 1, Some(1200), Some(4000), 90, 3),
            ],
            0.0,
        );
        let mesh = chunk_mesh(&view, layout, (0, 0), 12.0, 13.0).expect("meshes");
        // Two ground quads and one structure's five faces.
        assert_eq!(mesh.count_vertices(), 4 * 2 + 4 * 5);
        assert!(chunk_mesh(&view, layout, (1, 1), 12.0, 13.0).is_none());
    }
}
