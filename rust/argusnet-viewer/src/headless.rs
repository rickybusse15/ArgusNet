//! Headless rendering: the viewer's own pipeline, no window.
//!
//! This used to be a hand-written CPU rasterizer — ~700 lines drawing a flat 2-D
//! schematic that shared no code with the 3-D view, so every visual change had
//! to be made twice and the CI stills looked nothing like the product. It now
//! drives the real Bevy app into an offscreen image, so a still is what an
//! operator would see.
//!
//! The capture is a render-graph readback, adapted from Bevy's own
//! `examples/app/headless_renderer.rs`: a node copies the render target texture
//! into a mapped buffer, and the bytes cross into the main world over a channel.
//! The simpler `Screenshot::image()` API does not work here — it captures
//! nothing for a camera rendering to an offscreen image, yielding a uniformly
//! black PNG.
//!
//! Rendering needs a GPU adapter. wgpu has no software fallback of its own; on a
//! machine without one install a software Vulkan driver (`mesa-vulkan-drivers`
//! provides lavapipe), which is what CI does.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::image::TextureFormatPixelInfo;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode, PollType,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Extract, Render, RenderApp, RenderSystems};
use crossbeam_channel::{Receiver, Sender};

use bevy::world_serialization::WorldAssetRoot;

use crate::app::{run_headless, MainCamera};
use crate::replay::ReplayState;
use crate::state::{ReconstructionCamera, ViewMode};

/// Camera framing for a headless render.
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum CameraPreset {
    TopDown,
    Isometric,
    FollowTarget,
}

#[derive(Clone, Debug)]
pub struct HeadlessRenderOptions {
    pub output: Option<PathBuf>,
    pub record_dir: Option<PathBuf>,
    pub camera: CameraPreset,
    pub target_id: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// RealWorld draws the world alone; ScanMap/Split additionally show the
    /// accumulated scan-map reconstruction, so the believed terrain and team
    /// co-localization are captured in the still / sequence.
    pub view_mode: ViewMode,
}

/// What a headless run should produce, after validation.
#[derive(Clone, Debug, Resource)]
pub struct HeadlessRequest {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output: Option<PathBuf>,
    pub record_dir: Option<PathBuf>,
    pub camera: CameraPreset,
    pub target_id: Option<String>,
}

impl HeadlessRenderOptions {
    /// Splits out the render request, rejecting a run that would produce
    /// nothing. Kept separate from [`render_headless`] so the validation is
    /// testable without a GPU.
    fn into_request(self) -> Result<(ViewMode, HeadlessRequest)> {
        if self.output.is_none() && self.record_dir.is_none() {
            bail!("headless rendering requires --output or --record-dir");
        }
        Ok((
            self.view_mode,
            HeadlessRequest {
                width: self.width.max(16),
                height: self.height.max(16),
                fps: self.fps.max(1),
                output: self.output,
                record_dir: self.record_dir,
                camera: self.camera,
                target_id: self.target_id,
            },
        ))
    }
}

pub fn render_headless(scene_path: impl AsRef<Path>, options: HeadlessRenderOptions) -> Result<()> {
    let (view_mode, request) = options.into_request()?;
    run_headless(scene_path.as_ref(), view_mode, request)
}

// ---------------------------------------------------------------------------
// Render-world plumbing
// ---------------------------------------------------------------------------

/// Receives finished frames in the main world.
#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

/// Sends finished frames out of the render world.
#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

/// The offscreen image the cameras render into, plus its extent. Systems that
/// need the render size prefer this over the (absent) window.
#[derive(Resource, Clone)]
pub struct HeadlessTarget {
    pub image: Handle<Image>,
    pub width: u32,
    pub height: u32,
}

/// Copies one render target texture into a CPU-mappable buffer each frame.
#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    enabled: Arc<AtomicBool>,
    src_image: Handle<Image>,
}

impl ImageCopier {
    fn new(src_image: Handle<Image>, size: Extent3d, render_device: &RenderDevice) -> Self {
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(size.width as usize * 4);
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("argusnet-headless-readback"),
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            src_image,
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ImageCopiers(Vec<ImageCopier>);

/// Everything a headless run adds to the app: the render-world readback, the
/// offscreen target, and the capture state machine. Bundled as one plugin so
/// the channel and buffer types stay private to this module.
pub struct HeadlessRenderPlugin(pub HeadlessRequest);

impl Plugin for HeadlessRenderPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        app.insert_resource(MainWorldReceiver(receiver))
            .insert_resource(HeadlessCapture::new(&self.0))
            .insert_resource(self.0.clone())
            .add_systems(PostStartup, setup_headless_target_system)
            .add_systems(Last, headless_capture_system);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(RenderWorldSender(sender))
            .add_systems(ExtractSchedule, image_copy_extract)
            // Both run after the render graph has finished for the frame.
            //
            // Bevy's headless example puts the copy *inside* the RenderGraph
            // schedule, where it is an unordered sibling of the graph's own
            // nodes and can execute before the camera has drawn — which yielded
            // a texture that was still untouched (all zero) whenever the scene
            // contained geometry. Running it in `Render` after
            // `RenderSystems::Render` guarantees the whole graph, final output
            // write included, has already happened.
            .add_systems(
                Render,
                (image_copy_driver, receive_image_from_buffer)
                    .chain()
                    .after(RenderSystems::Render),
            );
    }
}

fn image_copy_extract(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(image_copiers.iter().cloned().collect()));
}

fn image_copy_driver(
    render_device: Res<RenderDevice>,
    image_copiers: Res<ImageCopiers>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
) {
    for image_copier in image_copiers.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
            continue;
        };

        let mut encoder =
            render_device.create_command_encoder(&CommandEncoderDescriptor::default());

        let block_dimensions = src_image.texture_descriptor.format.block_dimensions();
        let Some(block_size) = src_image.texture_descriptor.format.block_copy_size(None) else {
            continue;
        };

        // `copy_texture_to_buffer` only copies rows aligned to
        // COPY_BYTES_PER_ROW_ALIGNMENT, so the buffer can be wider than the
        // image. `unpad_rows` trims that back on the way out.
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.texture_descriptor.size.width as usize / block_dimensions.0 as usize)
                * block_size as usize,
        );
        let Some(bytes_per_row) = std::num::NonZero::<u32>::new(padded_bytes_per_row as u32) else {
            continue;
        };

        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &image_copier.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row.into()),
                    rows_per_image: None,
                },
            },
            src_image.texture_descriptor.size,
        );

        render_queue.submit(std::iter::once(encoder.finish()));
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for image_copier in image_copiers.0.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let buffer_slice = image_copier.buffer.slice(..);

        // Mapping is asynchronous and `get_mapped_range` panics if called early,
        // so hand the completion through a channel and block on it.
        let (sync_sender, sync_receiver) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = sync_sender.send(result);
        });

        if render_device.poll(PollType::wait_indefinitely()).is_err() {
            continue;
        }
        match sync_receiver.recv() {
            Ok(Ok(())) => {}
            // A failed map leaves nothing to read; the next frame retries.
            _ => {
                image_copier.buffer.unmap();
                continue;
            }
        }

        // Fails only if the main world has already torn down the receiver.
        let _ = sender.send(buffer_slice.get_mapped_range().to_vec());
        image_copier.buffer.unmap();
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Where a headless capture has got to.
///
/// The readback runs every frame and the channel always carries the most recent
/// frame, so this walks a state machine rather than capturing at a fixed frame
/// number: wait for every layer asset to finish loading, let the scene settle,
/// then save. A fixed frame count would race asset loading and silently write an
/// empty world — Bevy's own example warns the first frames are transparent, then
/// black, before the scene appears.
#[derive(Resource)]
struct HeadlessCapture {
    output: Option<PathBuf>,
    record_dir: Option<PathBuf>,
    fps: u32,
    phase: CapturePhase,
    next_index: usize,
    next_time_s: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CapturePhase {
    WaitingForAssets,
    /// Rendering until the output stops changing.
    Settling {
        seen: u32,
        stable: u32,
        last_hash: Option<u64>,
        min_frames: u32,
    },
    Capture,
    Done,
}

impl CapturePhase {
    fn settling(min_frames: u32) -> Self {
        Self::Settling {
            seen: 0,
            stable: 0,
            last_hash: None,
            min_frames,
        }
    }
}

/// Consecutive identical frames that count as "the scene has finished drawing".
const STABLE_FRAMES_REQUIRED: u32 = 10;

/// Frames to render before stability is even considered, on the first capture.
///
/// Stability alone is not a sufficient signal. Bevy specializes render pipelines
/// lazily on first use, and until that finishes the camera clears the target but
/// draws nothing — so the output sits *unchanged* at the clear colour for a long
/// stretch that looks exactly like a settled frame. On the software rasterizer
/// CI uses this takes several seconds; capturing on stability alone reliably
/// saves an empty image. This floor covers the compile, and the stability check
/// then catches whatever is slower still.
const WARMUP_FRAMES: u32 = 300;

/// Frames before stability counts on subsequent captures in a `--record-dir`
/// run. Pipelines are warm by then, so only the frame's own contents need to
/// catch up.
const MIN_SETTLE_FRAMES: u32 = 8;

/// Upper bound, so a scene that never settles (an animated overlay, say) still
/// produces an image instead of hanging.
const MAX_SETTLE_FRAMES: u32 = 2000;

impl HeadlessCapture {
    fn new(request: &HeadlessRequest) -> Self {
        Self {
            output: request.output.clone(),
            record_dir: request.record_dir.clone(),
            fps: request.fps,
            phase: CapturePhase::WaitingForAssets,
            next_index: 0,
            next_time_s: 0.0,
        }
    }
}

/// The main camera, excluding the reconstruction camera that also carries an
/// `OrbitCamera`.
type MainCameraQuery<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static mut crate::orbit_camera::OrbitCamera),
    (With<MainCamera>, Without<ReconstructionCamera>),
>;

/// Builds the offscreen target and points both cameras at it.
///
/// Runs in `PostStartup` so `setup_world` has already spawned the cameras: the
/// interactive setup stays untouched and headless only retargets what it made.
fn setup_headless_target_system(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    request: Res<HeadlessRequest>,
    replay_state: Res<ReplayState>,
    mut main_cam: MainCameraQuery,
    recon_cam: Query<Entity, With<ReconstructionCamera>>,
) {
    let size = Extent3d {
        width: request.width,
        height: request.height,
        ..default()
    };

    let mut render_target_image =
        Image::new_target_texture(size.width, size.height, TextureFormat::Rgba8UnormSrgb, None);
    // `new_target_texture` sets TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT;
    // reading the frame back out needs COPY_SRC on top of those.
    render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(render_target_image);

    commands.spawn(ImageCopier::new(handle.clone(), size, &render_device));

    if let Ok((entity, mut orbit)) = main_cam.single_mut() {
        commands
            .entity(entity)
            .insert(RenderTarget::Image(handle.clone().into()));
        apply_camera_preset(&mut orbit, &request, replay_state.as_ref());
    }
    // The reconstruction camera shares the image; Split mode gives the two of
    // them viewports over it, exactly as they share a window.
    if let Ok(entity) = recon_cam.single() {
        commands
            .entity(entity)
            .insert(RenderTarget::Image(handle.clone().into()));
    }

    commands.insert_resource(HeadlessTarget {
        image: handle,
        width: size.width,
        height: size.height,
    });
}

/// Frames the orbit camera per `--camera`, leaving `from_bounds`' scene-sized
/// radius intact so a preset changes the angle, not the framing distance.
///
/// Pitch is positive-is-above (`eye_position` puts the eye at
/// `focus.z + radius * sin(pitch)`) and the interactive camera clamps it to
/// [`MIN_PITCH_RAD`, `MAX_PITCH_RAD`], stopping short of straight down where
/// `look_at` against the Z-up reference vector degenerates. Presets stay inside
/// that range rather than inventing their own — a negative pitch would put the
/// camera underground, looking up at the underside of the terrain.
fn apply_camera_preset(
    orbit: &mut crate::orbit_camera::OrbitCamera,
    request: &HeadlessRequest,
    replay_state: &ReplayState,
) {
    use crate::orbit_camera::MAX_PITCH_RAD;

    /// Classic isometric elevation, inside the interactive pitch limits.
    const ISOMETRIC_PITCH_RAD: f32 = 0.6;

    match request.camera {
        CameraPreset::TopDown => {
            orbit.yaw = 0.0;
            orbit.pitch = MAX_PITCH_RAD;
        }
        CameraPreset::Isometric => {
            orbit.yaw = std::f32::consts::FRAC_PI_4;
            orbit.pitch = ISOMETRIC_PITCH_RAD;
        }
        CameraPreset::FollowTarget => {
            orbit.yaw = std::f32::consts::FRAC_PI_4;
            orbit.pitch = ISOMETRIC_PITCH_RAD;
            if let Some(target_id) = request.target_id.as_deref() {
                if let Some(marker) = replay_state
                    .current_markers()
                    .into_iter()
                    .find(|marker| marker.label == target_id)
                {
                    orbit.focus = marker.position;
                    orbit.radius = (orbit.radius * 0.25).max(orbit.min_radius);
                }
            }
        }
    }
}

/// Drives the capture state machine described on [`HeadlessCapture`].
fn headless_capture_system(
    mut capture: ResMut<HeadlessCapture>,
    mut replay_state: ResMut<ReplayState>,
    target: Option<Res<HeadlessTarget>>,
    receiver: Res<MainWorldReceiver>,
    asset_server: Res<AssetServer>,
    layer_roots: Query<&WorldAssetRoot>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(target) = target else {
        return;
    };

    // The render world produces a frame every tick; only the newest is useful.
    let mut latest = None;
    while let Ok(data) = receiver.try_recv() {
        latest = Some(data);
    }

    match capture.phase {
        CapturePhase::WaitingForAssets => {
            let pending = layer_roots
                .iter()
                .any(|root| !asset_server.is_loaded_with_dependencies(&root.0));
            if pending {
                return;
            }
            // A single still shows the completed mission, so jump to the final
            // frame rather than capturing an empty opening one.
            if capture.record_dir.is_none() {
                let last = replay_state.frame_count().saturating_sub(1);
                replay_state.step_to(last);
            } else {
                replay_state.step_to(0);
            }
            capture.phase = CapturePhase::settling(WARMUP_FRAMES);
        }
        CapturePhase::Settling {
            seen,
            stable,
            last_hash,
            min_frames,
        } => {
            let Some(frame) = latest.as_ref() else {
                return;
            };
            let hash = frame_hash(frame);
            let stable = if last_hash == Some(hash) {
                stable + 1
            } else {
                0
            };
            let seen = seen + 1;

            capture.phase = if seen >= MAX_SETTLE_FRAMES {
                warn!("headless render did not settle after {seen} frames; capturing anyway");
                CapturePhase::Capture
            } else if seen >= min_frames && stable >= STABLE_FRAMES_REQUIRED {
                CapturePhase::Capture
            } else {
                CapturePhase::Settling {
                    seen,
                    stable,
                    last_hash: Some(hash),
                    min_frames,
                }
            };
        }
        CapturePhase::Capture => {
            let Some(frame) = latest else {
                // Nothing has arrived yet; try again next tick.
                return;
            };

            if let Some(output) = capture.output.take() {
                if let Err(error) = save_frame(&output, &frame, target.width, target.height) {
                    error!("failed to write {}: {error:#}", output.display());
                }
                if capture.record_dir.is_none() {
                    capture.phase = CapturePhase::Done;
                    return;
                }
            }

            let Some(record_dir) = capture.record_dir.clone() else {
                capture.phase = CapturePhase::Done;
                return;
            };
            let index = capture.next_index;
            let path = record_dir.join(format!("frame_{index:05}.png"));
            if let Err(error) = save_frame(&path, &frame, target.width, target.height) {
                error!("failed to write {}: {error:#}", path.display());
            }
            capture.next_index += 1;
            capture.next_time_s += 1.0 / capture.fps as f32;

            match frame_at_or_after(replay_state.as_ref(), capture.next_time_s) {
                Some(index) => {
                    replay_state.step_to(index);
                    capture.phase = CapturePhase::settling(MIN_SETTLE_FRAMES);
                }
                None => capture.phase = CapturePhase::Done,
            }
        }
        CapturePhase::Done => {
            exit.write(AppExit::Success);
        }
    }
}

/// Cheap identity for a rendered frame, used to detect that the output has
/// stopped changing.
fn frame_hash(frame: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    frame.hash(&mut hasher);
    hasher.finish()
}

/// First frame index at or after `time_s`, or `None` once the replay is
/// exhausted — which is what ends a `--record-dir` run.
fn frame_at_or_after(replay_state: &ReplayState, time_s: f32) -> Option<usize> {
    let document = replay_state.document.as_ref()?;
    document
        .frames
        .iter()
        .position(|frame| frame.timestamp_s >= time_s)
}

/// Trims the row padding wgpu's copy alignment adds, so the bytes match the
/// image's real width.
fn unpad_rows(padded: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row_bytes = width as usize * TextureFormat::Rgba8UnormSrgb.pixel_size().unwrap_or(4);
    let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
    if row_bytes == aligned_row_bytes {
        return padded.to_vec();
    }
    padded
        .chunks(aligned_row_bytes)
        .take(height as usize)
        .flat_map(|row| &row[..row_bytes.min(row.len())])
        .copied()
        .collect()
}

fn save_frame(path: &Path, padded: &[u8], width: u32, height: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let pixels = unpad_rows(padded, width, height);
    let image = image::RgbaImage::from_raw(width, height, pixels)
        .context("readback buffer does not match the render target size")?;
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> HeadlessRenderOptions {
        HeadlessRenderOptions {
            output: None,
            record_dir: None,
            camera: CameraPreset::TopDown,
            target_id: None,
            width: 640,
            height: 360,
            fps: 30,
            view_mode: ViewMode::RealWorld,
        }
    }

    #[test]
    fn rejects_a_run_that_would_write_nothing() {
        let err = options().into_request().unwrap_err();
        assert!(err.to_string().contains("--output or --record-dir"));
    }

    #[test]
    fn carries_the_cli_surface_into_the_request() {
        let opts = HeadlessRenderOptions {
            output: Some(PathBuf::from("still.png")),
            record_dir: Some(PathBuf::from("frames")),
            camera: CameraPreset::Isometric,
            target_id: Some("truth-1".into()),
            width: 1280,
            height: 720,
            fps: 12,
            view_mode: ViewMode::Split,
        };
        let (view_mode, request) = opts.into_request().unwrap();
        assert_eq!(view_mode, ViewMode::Split);
        assert_eq!(request.output, Some(PathBuf::from("still.png")));
        assert_eq!(request.record_dir, Some(PathBuf::from("frames")));
        assert_eq!(request.camera, CameraPreset::Isometric);
        assert_eq!(request.target_id.as_deref(), Some("truth-1"));
        assert_eq!(
            (request.width, request.height, request.fps),
            (1280, 720, 12)
        );
    }

    #[test]
    fn either_output_alone_is_enough() {
        let mut still = options();
        still.output = Some(PathBuf::from("a.png"));
        assert!(still.into_request().is_ok());

        let mut sequence = options();
        sequence.record_dir = Some(PathBuf::from("frames"));
        assert!(sequence.into_request().is_ok());
    }

    #[test]
    fn degenerate_extents_are_clamped_rather_than_producing_an_empty_image() {
        let mut opts = options();
        opts.output = Some(PathBuf::from("a.png"));
        opts.width = 0;
        opts.height = 0;
        opts.fps = 0;
        let (_, request) = opts.into_request().unwrap();
        assert_eq!((request.width, request.height), (16, 16));
        // fps divides the record interval, so zero would be a division by zero.
        assert_eq!(request.fps, 1);
    }

    #[test]
    fn unpadded_rows_are_returned_unchanged() {
        // 64 px * 4 bytes = 256, already the alignment, so no padding to trim.
        let width = 64;
        let height = 2;
        let data: Vec<u8> = (0..width * 4 * height).map(|i| i as u8).collect();
        assert_eq!(unpad_rows(&data, width as u32, height as u32), data);
    }

    #[test]
    fn padding_is_trimmed_back_to_the_image_width() {
        // 3 px * 4 bytes = 12, which wgpu pads out to 256 per row.
        let width = 3u32;
        let height = 2u32;
        let row_bytes = width as usize * 4;
        let aligned = RenderDevice::align_copy_bytes_per_row(row_bytes);
        assert!(aligned > row_bytes, "expected this width to need padding");

        let mut padded = vec![0u8; aligned * height as usize];
        for row in 0..height as usize {
            for byte in 0..row_bytes {
                padded[row * aligned + byte] = (row * row_bytes + byte) as u8;
            }
        }

        let trimmed = unpad_rows(&padded, width, height);
        assert_eq!(trimmed.len(), row_bytes * height as usize);
        let expected: Vec<u8> = (0..row_bytes * height as usize).map(|i| i as u8).collect();
        assert_eq!(trimmed, expected);
    }
}
