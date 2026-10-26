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

use argusnet_sim::belief_mesh::{
    ground_mesh, structures_mesh, BeliefCells, GridLayout, GroundColours, MeshData,
};
use argusnet_sim::belief_world::BeliefCellUpdate;
use argusnet_sim::scene_export::{
    export_belief_scene, BeliefRows, SKIRT_M, STRUCTURE_ROOF_SRGB, STRUCTURE_WALL_SRGB,
};

use crate::belief_render::{BeliefLayer, BeliefRenderSnapshot, BeliefUnavailable};
use crate::replay::{BeliefCellRow, BeliefGridLayout, BeliefStats, ReplayState};
use crate::schema::ScenePackage;
use crate::state::{LayerVisibilityState, LoadedMissionZones, ViewMode};
use crate::terrain_material::{scene_settings, TerrainExtension, TerrainMaterial};
use crate::terrain_rasters::TerrainRasters;

/// The drones' map in the Layers panel, beside the scene's own layers.
pub const LAYER_ID: &str = "drones-map";
/// How often, in source seconds, the viewer's terrain-height lookups are
/// refreshed from the map. They feed marker footprints, zone walls and the
/// camera's ground, none of which needs every tick.
const HEIGHTS_EVERY_S: f32 = 1.0;

/// Cells per side of a mesh chunk: a change rebuilds only its chunk.
const CHUNK: u32 = 32;
/// How long, in source seconds, a newly updated cell glows.
const FRESH_S: f32 = 2.5;

#[derive(Component)]
pub struct BeliefChunk;

#[derive(Debug, Clone, Copy)]
struct Cell {
    row: BeliefCellRow,
    /// When the cell first became known: what the glow runs from.
    updated_s: f32,
    /// When its last row arrived: what the belief overlay ages from.
    last_s: f32,
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
    ground_material: Option<Handle<TerrainMaterial>>,
    /// Belief tiles (one per chunk) to republish to the terrain material's
    /// overlay, and whether every tile must be (the map was rebuilt).
    overlay_dirty: BTreeSet<(u32, u32)>,
    overlay_all: bool,
    overlay_revision: u64,
    /// Whether this view put the [`BeliefLayer`] there.
    owns_belief_layer: bool,
    heights_dirty: bool,
    heights_at_s: Option<f32>,
    truth_hidden: bool,
    now_s: f32,
    stats: Option<BeliefStats>,
    contacts: usize,
    /// What the last "Save map as scene" wrote, or why it could not.
    saved: Option<Result<String, String>>,
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
        self.overlay_all = true;
        self.heights_dirty = true;
        self.heights_at_s = None;
        self.applied_through_s = None;
        self.dirty.clear();
        self.fresh_until.clear();
        self.stats = None;
        self.contacts = 0;
    }

    /// Save the map at the cursor as a scene package, the one
    /// `argusnet-rs sim --export-scene` writes, in a new directory under
    /// `root`. `argusnet-viewer --scene DIR` reopens it.
    pub fn save_scene(&self, root: &std::path::Path) -> Result<std::path::PathBuf, String> {
        let layout = self.layout.ok_or("no self-built map is shown")?;
        let mut rows = BeliefRows::default();
        rows.0.extend(self.cells.iter().map(|(ij, cell)| {
            let r = cell.row;
            (*ij, BeliefCellUpdate(r.0, r.1, r.2, r.3, r.4, r.5))
        }));
        let at_s = self.applied_through_s.unwrap_or(self.now_s).max(0.0);
        let stem = format!("self-built-t{at_s:.0}s");
        let dir = (0..)
            .map(|n| match n {
                0 => root.join(&stem),
                n => root.join(format!("{stem}-{n}")),
            })
            .find(|dir| !dir.exists())
            .expect("an unused name");
        let provenance = serde_json::json!({
            "producer": "argusnet-viewer",
            "map_at_s": at_s,
        });
        export_belief_scene(&dir, &stem, grid_layout(layout), &rows, provenance)
            .map(|export| export.root)
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
                    last_s: at_s,
                },
            );
            self.overlay_dirty.insert((i / CHUNK, j / CHUNK));
            self.heights_dirty = true;
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
}

/// Apply the frames up to the cursor, and hide the true world the first
/// time a self-built map appears: the scene's layers, and the run's true
/// power lines (with their clearance corridors) and trees, which the replay
/// carries for the simulation, not as anything the drones measured. Each
/// can be turned back on to compare.
pub fn sync_belief_map_system(
    mut commands: Commands,
    replay: Res<ReplayState>,
    mut view: ResMut<BeliefMapView>,
    mut layers: ResMut<LayerVisibilityState>,
    mut overlays: ResMut<crate::state::RuntimeOverlayVisibility>,
    scene: Option<Res<ScenePackage>>,
) {
    let layout = replay.document.as_ref().and_then(|doc| doc.belief_grid());
    if layout != view.layout || replay.epoch() != view.epoch {
        // Another run: its truth is hidden afresh. A seek within the run
        // keeps the epoch, and with it whatever the viewer turned back on.
        if replay.epoch() != view.epoch {
            view.truth_hidden = false;
        }
        view.forget(&mut commands);
        view.layout = layout;
        view.epoch = replay.epoch();
    }
    let (Some(_), Some(document)) = (layout, replay.document.as_ref()) else {
        return;
    };
    if !view.truth_hidden {
        // The primary switch's own rule (V3): the drones' map on, every
        // truth-bearing layer off together.
        let scene_layers: Vec<String> = scene
            .map(|scene| scene.manifest.layers.iter().map(|l| l.id.clone()).collect())
            .unwrap_or_default();
        crate::world_view::apply(
            crate::world_view::WorldView::DronesMap,
            true,
            &scene_layers,
            LAYER_ID,
            &mut overlays,
            &mut layers.base_layers,
        );
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
#[allow(clippy::too_many_arguments)]
pub fn render_belief_chunks_system(
    mut commands: Commands,
    pick_generation: Option<ResMut<crate::picking::PickGeneration>>,
    mut view: ResMut<BeliefMapView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    rasters: Res<TerrainRasters>,
    scene: Res<ScenePackage>,
) {
    let Some(layout) = view.layout else {
        return;
    };
    if view.dirty.is_empty() && !view.buildings_dirty {
        return;
    }
    // The geometry under the pointer is about to change: a cached hover is
    // recomputed against it.
    if let Some(mut generation) = pick_generation {
        generation.0 += 1;
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
    // The ground takes the terrain material every scene's terrain takes: the
    // same elevation tint, contours and slope shading, and the same belief
    // overlay. It is created with the rasters' current settings and textures;
    // the raster system keeps every terrain material current after that.
    let ground_material = view
        .ground_material
        .get_or_insert_with(|| {
            terrain_materials.add(TerrainMaterial {
                base: StandardMaterial {
                    // What a scene's terrain layer is baked with (its style's
                    // `terrain-base` colour); the extension's elevation tint
                    // multiplies it.
                    base_color: Color::srgb(0.655, 0.710, 0.545),
                    perceptual_roughness: 0.95,
                    reflectance: 0.2,
                    cull_mode: None,
                    double_sided: true,
                    ..default()
                },
                extension: TerrainExtension {
                    settings: rasters.apply_to(scene_settings(&scene)),
                    belief_class: rasters.belief_class.clone(),
                    belief_numeric: rasters.belief_numeric.clone(),
                    land_cover: rasters.land_cover.clone(),
                },
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
                        // The drones' terrain and buildings hide what is
                        // behind them from picking, as a scene's own do.
                        crate::picking::PickOccluder,
                        bevy::picking::mesh_picking::ray_cast::RayCastBackfaces,
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
    let (low, _) = height_range(&view);
    let dirty = std::mem::take(&mut view.dirty);
    for chunk in dirty {
        let mesh = chunk_mesh(&view, layout, chunk, low);
        match (mesh, view.chunks.get(&chunk).copied()) {
            (Some(mesh), Some(entity)) => {
                commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
            }
            (Some(mesh), None) => {
                let entity = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(ground_material.clone()),
                        layers.clone(),
                        Visibility::Visible,
                        BeliefChunk,
                        // The drones' terrain and buildings hide what is
                        // behind them from picking, as a scene's own do.
                        crate::picking::PickOccluder,
                        bevy::picking::mesh_picking::ray_cast::RayCastBackfaces,
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

/// What a cell multiplies the terrain material's colour by: white once well
/// measured, so it looks like any scene's terrain; washed toward slate while
/// it has few returns; toward cyan while it is new.
fn ground_tint(level: u8, glow: f32) -> [f32; 3] {
    let confidence = (level.clamp(1, 4) as f32 - 1.0) / 3.0;
    let washed = [0.62, 0.70, 0.82];
    let fresh = [0.45, 1.0, 1.0];
    std::array::from_fn(|k| {
        let settled = washed[k] + (1.0 - washed[k]) * confidence;
        settled * (1.0 - glow) + fresh[k] * glow
    })
}

impl BeliefCells for BeliefMapView {
    fn ground(&self, i: i64, j: i64) -> Option<f32> {
        BeliefMapView::ground(self, i, j)
    }

    fn structure_top(&self, i: i64, j: i64) -> Option<f32> {
        if i < 0 || j < 0 {
            return None;
        }
        self.cells
            .get(&(i as u32, j as u32))
            .and_then(|cell| cell.row.structure_top_m())
    }

    fn structure_cells(&self) -> Vec<(u32, u32)> {
        let mut cells: Vec<(u32, u32)> = self
            .cells
            .iter()
            .filter(|(_, cell)| cell.row.structure_top_m().is_some())
            .map(|(ij, _)| *ij)
            .collect();
        cells.sort_unstable();
        cells
    }
}

fn grid_layout(layout: BeliefGridLayout) -> GridLayout {
    GridLayout {
        origin_m: layout.origin_m,
        resolution_m: layout.resolution_m,
        dimensions: layout.dimensions,
    }
}

/// One chunk of the known ground, built by the same code that writes a saved
/// map's terrain ([`argusnet_sim::belief_mesh::ground_mesh`]): one surface on
/// shared cell corners with a skirt round its edge. The viewer adds its own
/// colours: a corner is the mean tint of the known cells around it, and glows
/// with the brightest of them.
fn chunk_mesh(
    view: &BeliefMapView,
    layout: BeliefGridLayout,
    chunk: (u32, u32),
    low: f32,
) -> Option<Mesh> {
    let i_lo = chunk.0 * CHUNK;
    let j_lo = chunk.1 * CHUNK;
    let i_hi = ((chunk.0 + 1) * CHUNK).min(layout.dimensions[0]);
    let j_hi = ((chunk.1 + 1) * CHUNK).min(layout.dimensions[1]);
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
                if cell.row.ground_m().is_some() {
                    let g =
                        ((cell.updated_s + FRESH_S - view.now_s) / FRESH_S).clamp(0.0, 1.0) * 0.8;
                    let c = ground_tint(cell.row.confidence_level(), 0.0);
                    for k in 0..3 {
                        sum[k] += c[k];
                    }
                    glow = glow.max(g);
                    n += 1.0;
                }
            }
        }
        let fresh = [0.45, 1.0, 1.0];
        let c: [f32; 3] = std::array::from_fn(|k| {
            let base = if n > 0.0 { sum[k] / n } else { 1.0 };
            base * (1.0 - glow) + fresh[k] * glow
        });
        [c[0], c[1], c[2], 1.0]
    };
    let colours = GroundColours {
        corner: &corner_colour,
        skirt_top: [0.20, 0.22, 0.24, 1.0],
        skirt_bottom: [0.07, 0.08, 0.09, 1.0],
    };
    let mesh = ground_mesh(
        view,
        grid_layout(layout),
        i_lo..i_hi,
        j_lo..j_hi,
        low,
        SKIRT_M,
        &colours,
    );
    finish_mesh(mesh)
}

fn finish_mesh(mesh: MeshData) -> Option<Mesh> {
    if mesh.is_empty() {
        return None;
    }
    let MeshData {
        positions,
        normals,
        colours,
        indices,
        ..
    } = mesh;
    let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[1]]).collect();
    // The palette is written in sRGB; vertex colours are linear.
    let colours: Vec<[f32; 4]> = colours
        .into_iter()
        .map(|c| {
            let l = LinearRgba::from(Color::srgba(c[0], c[1], c[2], c[3]));
            [l.red, l.green, l.blue, l.alpha]
        })
        .collect();
    let mut out = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    out.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    out.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    out.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    out.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    out.insert_indices(Indices::U32(indices));
    Some(out)
}

/// Every believed structure as one solid, as a saved map's buildings are
/// ([`argusnet_sim::belief_mesh::structures_mesh`]), shaded by occupancy.
fn buildings_mesh(view: &BeliefMapView, layout: BeliefGridLayout) -> Option<Mesh> {
    let colour = |i: i64, j: i64| -> ([f32; 4], [f32; 4]) {
        let occupancy = view
            .cells
            .get(&(i as u32, j as u32))
            .map_or(1.0, |cell| cell.row.occupancy());
        let tint = 0.62 + 0.38 * occupancy;
        let [r, g, b] = STRUCTURE_ROOF_SRGB;
        let [wr, wg, wb] = STRUCTURE_WALL_SRGB;
        (
            [r * tint, g * tint, b * tint, 1.0],
            [wr * tint, wg * tint, wb * tint, 1.0],
        )
    };
    finish_mesh(structures_mesh(view, grid_layout(layout), &colour))
}

/// The map as the rest of the viewer reads it:
///
/// - **The belief overlay.** Changed tiles are republished to the
///   [`BeliefLayer`], which the terrain material draws as classes, confidence
///   and staleness when the belief layer is on (the Reconstruction preset).
///   This is the same publication the belief pipeline was built for.
/// - **Terrain heights.** Marker footprints, zone walls and the camera's
///   ground read `LoadedMissionZones::terrain_mesh`. It becomes the map's
///   measured ground, refreshed every [`HEIGHTS_EVERY_S`]; unknown cells are
///   NaN, which the lookup reads as no data rather than as zero.
/// - **Layers and views.** The map shows when its Layers entry is on. In the
///   split view it moves to the right pane only, beside whatever the left
///   pane shows (the scene's truth layers, if switched on).
#[allow(clippy::too_many_arguments)]
pub fn sync_belief_viewer_state_system(
    mut view: ResMut<BeliefMapView>,
    mut belief: ResMut<BeliefLayer>,
    mut zones: ResMut<LoadedMissionZones>,
    layers: Res<LayerVisibilityState>,
    view_mode: Res<ViewMode>,
    mut entities: Query<(&mut Visibility, &mut RenderLayers), With<BeliefChunk>>,
) {
    let Some(layout) = view.layout else {
        if view.owns_belief_layer {
            *belief = BeliefLayer::Unavailable(BeliefUnavailable::NoBeliefData);
            view.owns_belief_layer = false;
        }
        return;
    };
    let view = &mut *view;

    // The belief overlay.
    let fresh_snapshot = !view.owns_belief_layer || view.overlay_all;
    if fresh_snapshot {
        let bounds = argusnet_mapping::GridBounds {
            x_min_m: layout.origin_m[0] as f64,
            x_max_m: (layout.origin_m[0] + layout.dimensions[0] as f32 * layout.resolution_m)
                as f64,
            y_min_m: layout.origin_m[1] as f64,
            y_max_m: (layout.origin_m[1] + layout.dimensions[1] as f32 * layout.resolution_m)
                as f64,
            resolution_m: layout.resolution_m as f64,
        };
        *belief = match BeliefRenderSnapshot::new(bounds, CHUNK as usize) {
            Ok(snapshot) => BeliefLayer::Ready(Box::new(snapshot)),
            Err(reason) => BeliefLayer::Unavailable(reason),
        };
        view.owns_belief_layer = true;
        view.overlay_all = false;
        // Every tile, so tiles that went back to unknown are republished too.
        let (ti, tj) = (
            layout.dimensions[0].div_ceil(CHUNK),
            layout.dimensions[1].div_ceil(CHUNK),
        );
        for a in 0..ti {
            for b in 0..tj {
                view.overlay_dirty.insert((a, b));
            }
        }
    }
    if let BeliefLayer::Ready(snapshot) = &mut *belief {
        for tile in std::mem::take(&mut view.overlay_dirty) {
            view.overlay_revision += 1;
            snapshot.apply_patch(&tile_patch(view, layout, tile, view.overlay_revision));
        }
    } else {
        view.overlay_dirty.clear();
    }

    // Terrain heights for the rest of the viewer.
    let due = view
        .heights_at_s
        .is_none_or(|at| (view.now_s - at).abs() >= HEIGHTS_EVERY_S);
    if view.heights_dirty && due {
        zones.terrain_mesh = Some(height_raster(view, layout));
        view.heights_dirty = false;
        view.heights_at_s = Some(view.now_s);
    }

    // Layers and views.
    let shown = layers.base_layers.get(LAYER_ID).copied().unwrap_or(true);
    let wanted_layers = if *view_mode == ViewMode::Split {
        RenderLayers::layer(crate::app::RECON_LAYER)
    } else {
        RenderLayers::from_layers(&[0, crate::app::RECON_LAYER])
    };
    let wanted_visibility = if shown {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for (mut visibility, mut render_layers) in &mut entities {
        if *visibility != wanted_visibility {
            *visibility = wanted_visibility;
        }
        if *render_layers != wanted_layers {
            *render_layers = wanted_layers.clone();
        }
    }
}

/// One belief tile in the store's publication format, from the replay's
/// rows: what [`BeliefRenderSnapshot::apply_patch`] takes. Cells are i-major,
/// j contiguous. Coverage is the confidence level's return count
/// (1, 2, 4, 8); obstacle probability is the row's occupancy, which is 0 for
/// a cell with ground returns only.
fn tile_patch(
    view: &BeliefMapView,
    layout: BeliefGridLayout,
    tile: (u32, u32),
    revision: u64,
) -> argusnet_mapping::BeliefTilePatch {
    let (i0, j0) = (tile.0 * CHUNK, tile.1 * CHUNK);
    let di = CHUNK.min(layout.dimensions[0] - i0) as usize;
    let dj = CHUNK.min(layout.dimensions[1] - j0) as usize;
    let n = di * dj;
    let mut patch = argusnet_mapping::BeliefTilePatch {
        revision: argusnet_mapping::Revision(revision),
        tile: argusnet_mapping::TileIndex {
            ti: tile.0 as usize,
            tj: tile.1 as usize,
        },
        origin_ij: (i0 as usize, j0 as usize),
        dims: (di, dj),
        observed: vec![false; n],
        coverage_count: vec![0; n],
        mean_height_m: vec![f64::NAN; n],
        height_uncertainty_m: vec![f64::NAN; n],
        height_dispersion_m: vec![f64::NAN; n],
        obstacle_probability: vec![0.5; n],
        max_obstacle_height_m: vec![0.0; n],
        last_observed_s: vec![f64::NAN; n],
        source_mask: vec![0; n],
        source_names: Vec::new(),
    };
    for a in 0..di {
        for b in 0..dj {
            let Some(cell) = view.cells.get(&(i0 + a as u32, j0 + b as u32)) else {
                continue;
            };
            let k = a * dj + b;
            let returns = 1u32 << cell.row.confidence_level().clamp(1, 4).saturating_sub(1);
            patch.observed[k] = true;
            patch.coverage_count[k] = returns;
            patch.obstacle_probability[k] = cell.row.occupancy() as f64;
            patch.last_observed_s[k] = cell.last_s as f64;
            if let Some(h) = cell.row.ground_m() {
                patch.mean_height_m[k] = h as f64;
                patch.height_uncertainty_m[k] =
                    layout.resolution_m as f64 / (returns as f64).sqrt();
            }
            if let (Some(top), Some(ground)) = (cell.row.structure_top_m(), cell.row.ground_m()) {
                patch.max_obstacle_height_m[k] = (top - ground).max(0.0) as f64;
            }
        }
    }
    patch
}

/// The measured ground as the viewer's terrain-height lookup: one sample per
/// cell centre, NaN where the drones have no ground return.
fn height_raster(
    view: &BeliefMapView,
    layout: BeliefGridLayout,
) -> crate::replay::TerrainViewerMesh {
    let [nx, ny] = layout.dimensions;
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    crate::replay::TerrainViewerMesh {
        x_min_m: x0 + 0.5 * res,
        x_max_m: x0 + (nx as f32 - 0.5) * res,
        y_min_m: y0 + 0.5 * res,
        y_max_m: y0 + (ny as f32 - 0.5) * res,
        rows: ny as usize,
        cols: nx as usize,
        heights_m: (0..ny)
            .map(|j| {
                (0..nx)
                    .map(|i| view.ground(i as i64, j as i64).unwrap_or(f32::NAN))
                    .collect()
            })
            .collect(),
    }
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
    mut view: ResMut<BeliefMapView>,
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
                        .text("error is against truth, for evaluation · scene layers hidden")
                        .color(crate::theme::TEXT_MUTED),
                );
                if ui
                    .button("Save map as scene")
                    .on_hover_text(
                        "Write the map at the cursor as a scene package under ./argusnet-maps; \
                         open it with argusnet-viewer --scene DIR",
                    )
                    .clicked()
                {
                    let saved = view
                        .save_scene(std::path::Path::new(SAVED_MAPS_DIR))
                        .map(|dir| dir.display().to_string());
                    view.saved = Some(saved);
                }
                match &view.saved {
                    Some(Ok(path)) => {
                        ui.label(
                            crate::theme::TypeRole::Utility
                                .text(format!("saved → {path}"))
                                .color(crate::theme::TEXT_MUTED),
                        );
                    }
                    Some(Err(error)) => {
                        ui.colored_label(egui::Color32::from_rgb(230, 110, 100), error);
                    }
                    None => {}
                }
            });
        });
}

/// Where "Save map as scene" writes, relative to the working directory.
const SAVED_MAPS_DIR: &str = "argusnet-maps";

#[cfg(test)]
mod tests {
    use super::*;

    /// The run's true lines and trees are hidden with the scene's layers
    /// when a self-built map appears, once (so a viewer can turn them back
    /// on to compare), and left alone in a run with a prior world.
    #[test]
    fn a_self_built_map_hides_the_true_lines_and_trees() {
        let run = |knowledge: &str| {
            let document: crate::replay::ReplayDocument = serde_json::from_value(serde_json::json!({
                "meta": {
                    "world_knowledge": {
                        "knowledge": knowledge,
                        "belief_grid": {"origin_m": [0.0, 0.0], "resolution_m": 5.0, "dimensions": [4, 4]},
                    },
                },
                "frames": [{"timestamp_s": 0.0}, {"timestamp_s": 1.0}],
            }))
            .unwrap();
            let mut app = App::new();
            app.insert_resource(ReplayState::new(Some(document)))
                .init_resource::<BeliefMapView>()
                .init_resource::<LayerVisibilityState>()
                // Everything on, the corridors included, as a viewer may
                // have left them before loading this run.
                .insert_resource(crate::state::RuntimeOverlayVisibility {
                    power_lines: true,
                    power_line_clearance: true,
                    trees: true,
                    ..Default::default()
                })
                .add_systems(Update, sync_belief_map_system);
            app.update();
            app
        };
        let overlays = |app: &App| {
            let o = app
                .world()
                .resource::<crate::state::RuntimeOverlayVisibility>();
            (o.power_lines, o.power_line_clearance, o.trees)
        };
        assert_eq!(overlays(&run("prior")), (true, true, true));
        let mut app = run("none");
        assert_eq!(overlays(&app), (false, false, false));
        // Turned back on to compare, they stay on.
        {
            let mut o = app
                .world_mut()
                .resource_mut::<crate::state::RuntimeOverlayVisibility>();
            o.power_lines = true;
            o.trees = true;
        }
        app.update();
        assert_eq!(overlays(&app), (true, false, true));
        // A seek within the run keeps them.
        app.world_mut().resource_mut::<ReplayState>().step_to(1);
        app.update();
        assert_eq!(overlays(&app), (true, false, true));
        // Another self-built run hides its truth afresh.
        let next = app.world().resource::<ReplayState>().document.clone();
        app.world_mut()
            .resource_mut::<ReplayState>()
            .begin_epoch(next);
        app.update();
        assert_eq!(overlays(&app), (false, false, false));
    }

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
        let mesh = chunk_mesh(&view, layout(), (0, 0), 12.0).expect("meshes");
        // Two cells share an edge: 6 surface corners, not 8. The outline has
        // 6 edges, each a skirt quad of 4 vertices.
        assert_eq!(mesh.count_vertices(), 6 + 6 * 4);
        assert!(chunk_mesh(&view, layout(), (1, 1), 12.0).is_none());
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

    #[test]
    fn rows_become_store_patches_the_belief_overlay_classifies() {
        let mut view = BeliefMapView::default();
        let layout = layout();
        view.apply(
            &[
                BeliefCellRow(1, 2, Some(1000), None, 0, 4),
                BeliefCellRow(3, 1, Some(1000), Some(4000), 90, 2),
            ],
            5.0,
        );
        let patch = tile_patch(&view, layout, (0, 0), 7);
        assert_eq!(patch.dims, (32, 32));
        // i-major, j contiguous.
        let k = 32 + 2;
        assert!(patch.observed[k]);
        assert_eq!(patch.coverage_count[k], 8);
        assert_eq!(patch.obstacle_probability[k], 0.0);
        assert_eq!(patch.last_observed_s[k], 5.0);
        assert_eq!(patch.max_obstacle_height_m[3 * 32 + 1], 30.0);
        let bounds = argusnet_mapping::GridBounds {
            x_min_m: 0.0,
            x_max_m: 640.0,
            y_min_m: 0.0,
            y_max_m: 640.0,
            resolution_m: 10.0,
        };
        let mut snapshot = BeliefRenderSnapshot::new(bounds, CHUNK as usize).unwrap();
        snapshot.apply_patch(&patch);
        assert_eq!(snapshot.rejected_patches(), 0);
        assert_eq!(
            snapshot.class_ij(1, 2),
            crate::belief_render::BeliefClass::KnownSafe
        );
        assert_eq!(
            snapshot.class_ij(3, 1),
            crate::belief_render::BeliefClass::KnownObstacle
        );
        assert_eq!(
            snapshot.class_ij(9, 9),
            crate::belief_render::BeliefClass::Unknown
        );
    }

    #[test]
    fn the_map_at_the_cursor_saves_as_a_scene_package() {
        let mut view = BeliefMapView {
            layout: Some(layout()),
            ..default()
        };
        assert!(view
            .save_scene(std::path::Path::new("/nonexistent"))
            .is_err());
        view.apply(
            &[
                row(1, 1, Some(500)),
                row(2, 1, Some(520)),
                BeliefCellRow(3, 1, Some(510), Some(2_500), 90, 4),
            ],
            3.0,
        );
        view.applied_through_s = Some(3.0);
        let root = std::env::temp_dir().join(format!("belief-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let first = view.save_scene(&root).unwrap();
        let second = view.save_scene(&root).unwrap();
        assert_eq!(first, root.join("self-built-t3s"));
        assert_eq!(second, root.join("self-built-t3s-1"));
        for file in [
            "scene_manifest.json",
            "terrain/terrain-base.glb",
            "overlays/buildings.glb",
            "metadata/belief_map.json",
        ] {
            assert!(first.join(file).is_file(), "{file}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_height_lookup_answers_only_where_ground_was_measured() {
        let mut view = BeliefMapView::default();
        let layout = layout();
        view.apply(
            &[
                row(0, 0, Some(1000)),
                row(1, 0, Some(1200)),
                row(0, 1, Some(1000)),
                row(1, 1, Some(1200)),
            ],
            0.0,
        );
        let raster = height_raster(&view, layout);
        let h = raster
            .height_at(10.0, 10.0)
            .expect("inside the measured cells");
        assert!((h - 11.0).abs() < 1e-4);
        assert_eq!(
            raster.height_at(300.0, 300.0),
            None,
            "unknown is no data, not zero"
        );
    }
}
