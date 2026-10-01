//! The terrain material's rasters: land cover and belief (P2.1).
//!
//! [`TerrainRasters`] owns three images: the land-cover class raster and the
//! two belief channels. Each is created whole when its source's geometry
//! appears or changes. After that, belief changes reach the GPU **only as
//! changed tiles**. A dirty tile is written straight into the existing GPU
//! texture by a render-world system (`RenderQueue::write_texture`) rather
//! than by editing the `Image` asset, which would re-upload the whole texture.
//! The images are `RENDER_WORLD`-only for the same reason.
//!
//! Uploads that cannot be made are exposed, not dropped quietly.
//! [`UploadStatus`] counts tiles written, tiles waiting for their texture to
//! exist, and tiles refused (a rectangle outside the texture, or a byte count
//! that does not match). A tile that waits too long is refused.
//!
//! The material's settings are rewritten only when something they carry
//! changed: the overlay toggles, a raster's geometry, or the simulation time
//! the belief overlay measures staleness against.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDimension, TextureFormat,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::belief_render::{BeliefLayer, TileTexels};
use crate::replay::ReplayState;
use crate::schema::{LandCoverState, ScenePackage};
use crate::state::RuntimeOverlayVisibility;
use crate::terrain_material::{scene_settings, BeliefMode, TerrainMaterial, TerrainSettings};

/// How strongly land cover tints the terrain when shown.
pub const LAND_COVER_STRENGTH: f32 = 0.45;
/// How strongly the belief overlay colours the terrain when shown.
pub const BELIEF_STRENGTH: f32 = 0.75;
/// Age, in simulation seconds, past which an observed cell reads stale.
pub const BELIEF_STALE_AFTER_S: f32 = 60.0;
/// Render frames a tile may wait for its texture before it is refused.
pub const MAX_UPLOAD_WAIT_FRAMES: u32 = 600;

/// Counters shared with the render world.
#[derive(Debug, Default)]
pub struct UploadStatus {
    pub uploaded_tiles: AtomicU64,
    pub waiting_tiles: AtomicU64,
    pub refused_tiles: AtomicU64,
}

/// One changed tile, bound for the textures it belongs to.
#[derive(Debug)]
struct PendingUpload {
    class: Handle<Image>,
    numeric: Handle<Image>,
    tile: TileTexels,
    waited: u32,
}

type UploadQueue = Arc<Mutex<Vec<PendingUpload>>>;

#[derive(Resource, Clone)]
struct RenderUploads {
    queue: UploadQueue,
    status: Arc<UploadStatus>,
}

/// The rasters and the belief/land-cover half of the terrain settings.
#[derive(Resource)]
pub struct TerrainRasters {
    pub belief_class: Handle<Image>,
    pub belief_numeric: Handle<Image>,
    pub land_cover: Handle<Image>,
    placeholders: (Handle<Image>, Handle<Image>, Handle<Image>),
    land_cover_grid: Option<(Vec4, UVec4, [Vec4; crate::land_cover_raster::PALETTE_SIZE])>,
    land_cover_source: Option<usize>,
    belief_geometry: Option<(Vec4, UVec4)>,
    belief_mode: BeliefMode,
    land_cover_shown: bool,
    now_s: f32,
    applied: Option<(TerrainSettings, [AssetId<Image>; 3])>,
    queue: UploadQueue,
    status: Arc<UploadStatus>,
}

impl FromWorld for TerrainRasters {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let class = images.add(raster_image(1, 1, TextureFormat::Rg8Uint, vec![0; 2]));
        let numeric = images.add(raster_image(1, 1, TextureFormat::Rg32Float, vec![0; 8]));
        let land_cover = images.add(raster_image(1, 1, TextureFormat::R8Uint, vec![0]));
        Self {
            belief_class: class.clone(),
            belief_numeric: numeric.clone(),
            land_cover: land_cover.clone(),
            placeholders: (class, numeric, land_cover),
            land_cover_grid: None,
            land_cover_source: None,
            belief_geometry: None,
            belief_mode: BeliefMode::Off,
            land_cover_shown: true,
            now_s: 0.0,
            applied: None,
            queue: Arc::default(),
            status: Arc::default(),
        }
    }
}

/// What the rasters can report to an operator.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterReport {
    pub belief_mode: BeliefMode,
    pub land_cover_bound: bool,
    pub uploaded_tiles: u64,
    pub waiting_tiles: u64,
    pub refused_tiles: u64,
}

impl TerrainRasters {
    /// `settings` with the rasters' half filled in.
    pub fn apply_to(&self, mut settings: TerrainSettings) -> TerrainSettings {
        if let Some((grid, dims, palette)) = self.land_cover_grid {
            let strength = if self.land_cover_shown {
                LAND_COVER_STRENGTH
            } else {
                0.0
            };
            settings.land_cover_grid = grid.truncate().extend(strength);
            settings.land_cover_dims = dims;
            settings.land_cover_palette = palette;
        }
        if let Some((grid, dims)) = self.belief_geometry {
            settings.belief_grid = grid.truncate().extend(BELIEF_STRENGTH);
            settings.belief_dims = UVec4::new(dims.x, dims.y, 0, 0);
        }
        // The mode is the shader's `belief_dims.z`, set last so no geometry
        // write can land it in the wrong lane.
        settings.belief_dims.z = self.belief_mode as u32;
        if self.belief_geometry.is_none() {
            settings.belief_grid.w = BELIEF_STRENGTH;
        }
        settings.belief_time = Vec4::new(self.now_s, BELIEF_STALE_AFTER_S, 0.0, 0.0);
        settings
    }

    pub fn report(&self) -> RasterReport {
        RasterReport {
            belief_mode: self.belief_mode,
            land_cover_bound: self.land_cover_grid.is_some(),
            uploaded_tiles: self.status.uploaded_tiles.load(Ordering::Relaxed),
            waiting_tiles: self.status.waiting_tiles.load(Ordering::Relaxed),
            refused_tiles: self.status.refused_tiles.load(Ordering::Relaxed),
        }
    }

    fn handles(&self) -> [AssetId<Image>; 3] {
        [
            self.belief_class.id(),
            self.belief_numeric.id(),
            self.land_cover.id(),
        ]
    }

    /// Tiles queued and not yet written.
    pub fn pending_uploads(&self) -> usize {
        self.queue.lock().map(|queue| queue.len()).unwrap_or(0)
    }
}

pub fn raster_image(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    )
}

pub struct TerrainRastersPlugin;

impl Plugin for TerrainRastersPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BeliefLayer>()
            .init_resource::<TerrainRasters>();
        let rasters = app.world().resource::<TerrainRasters>();
        let uploads = RenderUploads {
            queue: Arc::clone(&rasters.queue),
            status: Arc::clone(&rasters.status),
        };
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(uploads).add_systems(
                Render,
                write_belief_tiles.in_set(RenderSystems::PrepareResources),
            );
        }
    }
}

/// Keeps the rasters, and the terrain materials' settings, current.
#[allow(clippy::too_many_arguments)]
pub fn sync_terrain_rasters_system(
    scene: Res<ScenePackage>,
    mut belief: ResMut<BeliefLayer>,
    visibility: Res<RuntimeOverlayVisibility>,
    replay: Res<ReplayState>,
    mut rasters: ResMut<TerrainRasters>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    let rasters = rasters.bypass_change_detection();

    // Land cover: rebuilt only when the scene's raster is a different one.
    let source = match &scene.land_cover {
        LandCoverState::Ready(raster) => Some(Arc::as_ptr(raster) as usize),
        _ => None,
    };
    if source != rasters.land_cover_source {
        rasters.land_cover_source = source;
        match &scene.land_cover {
            LandCoverState::Ready(raster) => {
                rasters.land_cover = images.add(raster_image(
                    raster.cols as u32,
                    raster.rows as u32,
                    TextureFormat::R8Uint,
                    raster.classes.clone(),
                ));
                rasters.land_cover_grid = Some((
                    Vec4::new(
                        raster.x_min_m as f32,
                        raster.y_min_m as f32,
                        raster.resolution_m as f32,
                        0.0,
                    ),
                    UVec4::new(raster.cols as u32, raster.rows as u32, 1, 0),
                    raster.palette,
                ));
            }
            _ => {
                rasters.land_cover = rasters.placeholders.2.clone();
                rasters.land_cover_grid = None;
            }
        }
    }
    rasters.land_cover_shown = visibility.land_cover_tint;

    // Belief: whole images when the geometry changes, tiles after that.
    match belief.bypass_change_detection() {
        BeliefLayer::Unavailable(_) => {
            if rasters.belief_geometry.take().is_some() {
                rasters.belief_class = rasters.placeholders.0.clone();
                rasters.belief_numeric = rasters.placeholders.1.clone();
                if let Ok(mut queue) = rasters.queue.lock() {
                    queue.clear();
                }
            }
        }
        BeliefLayer::Ready(snapshot) => {
            let b = snapshot.bounds();
            let (nx, ny) = snapshot.dimensions();
            let geometry = (
                Vec4::new(
                    b.x_min_m as f32,
                    b.y_min_m as f32,
                    b.resolution_m as f32,
                    0.0,
                ),
                UVec4::new(nx as u32, ny as u32, 0, 0),
            );
            if rasters.belief_geometry != Some(geometry) {
                let whole = snapshot.texels();
                // The whole image carries every tile; none is left to upload.
                snapshot.take_dirty();
                rasters.belief_class = images.add(raster_image(
                    nx as u32,
                    ny as u32,
                    TextureFormat::Rg8Uint,
                    whole.class,
                ));
                rasters.belief_numeric = images.add(raster_image(
                    nx as u32,
                    ny as u32,
                    TextureFormat::Rg32Float,
                    whole.numeric,
                ));
                rasters.belief_geometry = Some(geometry);
                if let Ok(mut queue) = rasters.queue.lock() {
                    queue.clear();
                }
            } else if snapshot.dirty_len() > 0 {
                let tiles = snapshot.take_dirty();
                if let Ok(mut queue) = rasters.queue.lock() {
                    queue.extend(tiles.into_iter().map(|tile| PendingUpload {
                        class: rasters.belief_class.clone(),
                        numeric: rasters.belief_numeric.clone(),
                        tile,
                        waited: 0,
                    }));
                }
            }
        }
    }
    rasters.belief_mode = if !visibility.belief {
        BeliefMode::Off
    } else if rasters.belief_geometry.is_some() {
        BeliefMode::Ready
    } else {
        BeliefMode::Unavailable
    };
    // Staleness is measured against simulation time, which only matters when
    // there is something to be stale.
    if rasters.belief_mode == BeliefMode::Ready {
        rasters.now_s = replay.current_timestamp_s();
    }

    let settings = rasters.apply_to(scene_settings(&scene));
    let handles = rasters.handles();
    if rasters.applied.as_ref() == Some(&(settings.clone(), handles)) {
        return;
    }
    for (_, material) in materials.iter_mut() {
        material.extension.settings = settings.clone();
        material.extension.belief_class = rasters.belief_class.clone();
        material.extension.belief_numeric = rasters.belief_numeric.clone();
        material.extension.land_cover = rasters.land_cover.clone();
    }
    rasters.applied = Some((settings, handles));
}

/// Write queued belief tiles into their GPU textures. Render world.
fn write_belief_tiles(
    uploads: Res<RenderUploads>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_queue: Res<RenderQueue>,
) {
    let Ok(mut queue) = uploads.queue.lock() else {
        return;
    };
    let status = &uploads.status;
    let mut waiting = 0u64;
    queue.retain_mut(|upload| {
        let (Some(class), Some(numeric)) = (
            gpu_images.get(&upload.class),
            gpu_images.get(&upload.numeric),
        ) else {
            upload.waited += 1;
            if upload.waited > MAX_UPLOAD_WAIT_FRAMES {
                status.refused_tiles.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            waiting += 1;
            return true;
        };
        let rect = upload.tile.rect;
        let fits = |image: &GpuImage, bytes: usize, data: &[u8]| {
            let size = image.texture_descriptor.size;
            rect.x + rect.width <= size.width
                && rect.y + rect.height <= size.height
                && data.len() == rect.width as usize * rect.height as usize * bytes
        };
        if !fits(class, 2, &upload.tile.class) || !fits(numeric, 8, &upload.tile.numeric) {
            status.refused_tiles.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        for (image, bytes, data) in [
            (class, 2u32, &upload.tile.class),
            (numeric, 8u32, &upload.tile.numeric),
        ] {
            render_queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &image.texture,
                    mip_level: 0,
                    origin: Origin3d {
                        x: rect.x,
                        y: rect.y,
                        z: 0,
                    },
                    aspect: TextureAspect::All,
                },
                data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(rect.width * bytes),
                    rows_per_image: Some(rect.height),
                },
                Extent3d {
                    width: rect.width,
                    height: rect.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        status.uploaded_tiles.fetch_add(1, Ordering::Relaxed);
        false
    });
    status.waiting_tiles.store(waiting, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mode is the shader's `belief_dims.z`, with or without geometry. It
    /// once landed in `w`, where the shader never looked, and the overlay
    /// silently drew nothing; the pixel probes caught it.
    #[test]
    fn the_belief_mode_lands_in_the_lane_the_shader_reads() {
        let mut world = World::new();
        world.init_resource::<Assets<Image>>();
        let mut rasters = TerrainRasters::from_world(&mut world);
        let base = TerrainSettings::for_elevation_range(0.0, 10.0);
        rasters.belief_mode = BeliefMode::Unavailable;
        assert_eq!(rasters.apply_to(base.clone()).belief_dims.z, 1);
        rasters.belief_geometry = Some((Vec4::new(1.0, 2.0, 3.0, 0.0), UVec4::new(37, 30, 0, 0)));
        rasters.belief_mode = BeliefMode::Ready;
        let settings = rasters.apply_to(base);
        assert_eq!(settings.belief_dims, UVec4::new(37, 30, 2, 0));
        assert_eq!(
            settings.belief_grid,
            Vec4::new(1.0, 2.0, 3.0, BELIEF_STRENGTH)
        );
    }
}
