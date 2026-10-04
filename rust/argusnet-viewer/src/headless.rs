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
use std::time::Instant;

use anyhow::{bail, Context, Result};
use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::image::TextureFormatPixelInfo;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::PipelineCache;
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
    /// Isometric, a fifth of the scene-sized distance from its centre (G2's
    /// close view).
    Close,
    /// Grazing: just above the horizon, where depth complexity and fog are
    /// greatest (G2's low pitch).
    LowPitch,
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
    /// Pixel probes (P2.1): not on the command line; the probe tests use it.
    pub probe: Option<ProbeRequest>,
    /// The rendering quality tier (C6).
    pub quality: crate::quality::QualityTier,
    /// After the still, time this many unpaced frames and write the
    /// statistics beside it (G2, E2). Zero measures nothing.
    pub frame_stats: u32,
    /// After the still, run this many lifecycle rotations (seek, toggle,
    /// reload, tier, projection) and write the world's counts after each
    /// (C8 resource growth). Zero runs none.
    pub lifecycle: u32,
}

/// A pointer position for a probe: a world point's pixel through the live
/// main camera, plus an offset in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProbePointer {
    pub world: [f64; 3],
    pub offset_px: [f32; 2],
}

/// Probe the captured still at world points, projected through the live
/// camera, and write what was drawn there.
#[derive(Clone, Debug, Default)]
pub struct ProbeRequest {
    /// Belief to draw.
    pub belief: Option<argusnet_mapping::BeliefStore>,
    /// Turn the belief overlay on. With no `belief`, that draws the explicit
    /// unavailable state.
    pub belief_overlay: bool,
    /// Tile patches applied once the scene has loaded, after the belief
    /// textures exist: they reach the GPU as tile writes, not a new upload.
    pub late_patches: Vec<argusnet_mapping::BeliefTilePatch>,
    /// Overlay toggles for the render; the defaults when `None`. The belief
    /// toggle is `belief_overlay`'s either way.
    pub overlays: Option<crate::state::RuntimeOverlayVisibility>,
    /// World points, `(x, y)`; `z` is the terrain's there.
    pub points: Vec<[f64; 2]>,
    /// World points at a given height, probed after `points`.
    pub elevated: Vec<[f64; 3]>,
    /// Select a marker directly, by kind (`"node"`, `"track"` or `"truth"`)
    /// and id, as the keyboard and list paths do (P2.3).
    pub select: Option<(String, String)>,
    /// Clicks, in order, through the real picker; each one's outcome is
    /// reported (P2.3).
    pub clicks: Vec<ProbePointer>,
    /// Where the pointer rests for the capture, for the hover ring (P2.3).
    pub hover: Option<ProbePointer>,
    /// Where to write the probe results, as JSON.
    pub output: PathBuf,
}

/// Half-width of the window of pixels recorded around each probe.
pub const PROBE_WINDOW_RADIUS: i64 = 4;

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
    pub probe: Option<ProbeRequest>,
    pub quality: crate::quality::QualityTier,
    pub frame_stats: u32,
    pub lifecycle: u32,
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
                probe: self.probe,
                quality: self.quality,
                frame_stats: self.frame_stats,
                lifecycle: self.lifecycle,
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
struct MainWorldReceiver(Receiver<RenderedFrame>);

/// Sends finished frames out of the render world.
#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<RenderedFrame>);

/// One frame's pixels, plus whether the renderer had finished compiling.
///
/// `pipelines_pending` is the piece the main world cannot see for itself:
/// `PipelineCache` lives in the render world, and it is the only authority on
/// whether a shader is still being specialized. Without it the capture has to
/// guess from pixels, and guessing is not good enough — a frame showing only
/// the sky is non-uniform and can hold still for a dozen frames while the
/// terrain pipeline is mid-compile, which captures a near-empty image roughly
/// one run in three.
struct RenderedFrame {
    pixels: Vec<u8>,
    pipelines_pending: bool,
    /// Pipelines that failed for good: a shader that does not compile. Such a
    /// frame still draws (whatever used the pipeline is simply missing), so
    /// only this says it is wrong.
    pipeline_errors: Vec<String>,
}

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
            .add_systems(
                Last,
                (
                    headless_capture_system,
                    lifecycle_system,
                    frame_stats_system,
                )
                    .chain(),
            );

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
    pipeline_cache: Res<PipelineCache>,
) {
    let pipelines_pending = pipeline_cache.waiting_pipelines().next().is_some();
    // Retryable states (a shader or import still loading) are not failures.
    let pipeline_errors: Vec<String> = pipeline_cache
        .pipelines()
        .filter_map(|pipeline| match &pipeline.state {
            bevy::render::render_resource::CachedPipelineState::Err(error)
                if !matches!(
                    error,
                    bevy::shader::ShaderCacheError::ShaderNotLoaded(_)
                        | bevy::shader::ShaderCacheError::ShaderImportNotYetAvailable
                ) =>
            {
                Some(error.to_string())
            }
            _ => None,
        })
        .collect();
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
        let _ = sender.send(RenderedFrame {
            pixels: buffer_slice.get_mapped_range().to_vec(),
            pipelines_pending,
            pipeline_errors: pipeline_errors.clone(),
        });
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
    /// Set when the run must exit with an error rather than success.
    failed: bool,
    output: Option<PathBuf>,
    record_dir: Option<PathBuf>,
    fps: u32,
    phase: CapturePhase,
    next_index: usize,
    next_time_s: f32,
    timing: CaptureTiming,
    /// The probe's interactions have run; what each click selected.
    interacted: bool,
    clicks: Vec<serde_json::Value>,
    /// Frames to time after the still, and where to write what was found.
    frame_stats: u32,
    stats_path: Option<PathBuf>,
    last_frame: Option<Instant>,
    samples_ms: Vec<f32>,
    /// Lifecycle rotations to run, the step reached, and the counts taken.
    lifecycle: u32,
    lifecycle_step: u32,
    lifecycle_counts: Vec<serde_json::Value>,
}

/// Where a headless run spent its time, and on which frame each milestone fell.
///
/// The render step is the slowest thing in CI and has grown with every visual
/// change, but wall-clock alone cannot say whether the *drawing* got more
/// expensive or the release condition simply waits too long. Splitting the two
/// is what `docs/migration/DELIVERY_PLAN.md`'s G2 guardrail asks for before more
/// expensive effects go in — and it is what turned the blind frame floor below
/// into a measured condition.
#[derive(Debug)]
struct CaptureTiming {
    started: Instant,
    assets_ready: Option<Instant>,
    frames_waiting_assets: u32,
    /// First frame whose output was not a single flat colour — the first frame
    /// on which something was actually drawn.
    drawn_at_frame: Option<u32>,
    /// First frame that had been stable for `STABLE_FRAMES_REQUIRED`, whether
    /// or not anything had been drawn yet.
    ///
    /// Recorded because it is the trap: a cleared target is perfectly stable,
    /// so this lands far earlier than the scene is ready. On the bundled scene
    /// it is frame ~20, where the image still holds exactly one colour. Keeping
    /// it in the log makes that concrete for anyone tempted to release on
    /// stability alone.
    stable_at_frame: Option<u32>,
    /// Frames spent settling in total: the frame the capture fired on.
    settle_frames: u32,
}

impl CaptureTiming {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            assets_ready: None,
            frames_waiting_assets: 0,
            drawn_at_frame: None,
            stable_at_frame: None,
            settle_frames: 0,
        }
    }

    /// Emits the breakdown. Called once per run, on the first still.
    fn report(&self) {
        let assets_s = self
            .assets_ready
            .map(|ready| ready.duration_since(self.started).as_secs_f32())
            .unwrap_or_default();
        info!(
            "headless timing: total {:.1}s (assets {:.1}s over {} frames); drawn at frame {}, \
             stable at frame {}, captured at frame {}",
            self.started.elapsed().as_secs_f32(),
            assets_s,
            self.frames_waiting_assets,
            OptionalFrame(self.drawn_at_frame),
            OptionalFrame(self.stable_at_frame),
            self.settle_frames,
        );
    }
}

/// Renders a milestone a run may never have reached.
struct OptionalFrame(Option<u32>);

impl std::fmt::Display for OptionalFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(frame) => write!(f, "{frame}"),
            None => write!(f, "never"),
        }
    }
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
    /// Driving the probe's selection, clicks and hover through the picker,
    /// one step at a time: `step` is the next to issue, `waiting` a click the
    /// picker has not handled yet.
    Interacting {
        step: usize,
        waiting: Option<u64>,
    },
    Capture,
    /// Cycling the presentation through seek, toggle, reload, tier and
    /// projection changes, counting what the world holds after each rotation.
    Cycling,
    /// Timing frames after the still: the loop runs unpaced, so each sample
    /// is the frame's own cost, including the readback headless performs.
    Measuring,
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

/// Frames rendered before the release condition is even consulted.
///
/// This was a blind 300, and replacing it took three attempts because the two
/// obvious signals are both wrong. Recording them here so the next person does
/// not repeat the sequence:
///
/// 1. **Stability alone.** Bevy specializes pipelines lazily; until that
///    finishes the camera clears the target and draws nothing, so the output
///    sits *unchanged* at the clear colour. On the bundled scene it is stable by
///    frame 11 and the saved image holds exactly one colour.
/// 2. **"Wait for the output to change."** The first change is the target going
///    transparent, then black, then the clear colour. None of that is drawing.
/// 3. **"Wait until the frame is more than one colour" ([`is_uniform`]).**
///    Closer, and it captured correctly most of the time — but the sky draws
///    before the terrain, and a sky-only frame is both non-uniform and perfectly
///    still while the terrain pipeline compiles. Roughly one run in three saved
///    an 8-colour image.
///
/// The signal that actually works is the one the render world already owns:
/// `PipelineCache::waiting_pipelines`. The release condition is all three of
/// drawn, no pipeline still compiling, and stable — see [`RenderedFrame`].
/// Measured: capture moved from frame 300 to frame ~48, the render step from
/// ~35s to ~4.6s, and eight consecutive captures are byte-identical to what the
/// 300-frame floor produced.
///
/// What remains here is only a small lower bound before any of it is consulted.
const WARMUP_FRAMES: u32 = MIN_SETTLE_FRAMES;

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
            failed: false,
            output: request.output.clone(),
            record_dir: request.record_dir.clone(),
            fps: request.fps,
            phase: CapturePhase::WaitingForAssets,
            next_index: 0,
            next_time_s: 0.0,
            timing: CaptureTiming::new(),
            interacted: false,
            clicks: Vec::new(),
            frame_stats: request.frame_stats,
            stats_path: request
                .output
                .as_ref()
                .map(|output| output.with_extension("frames.json")),
            last_frame: None,
            samples_ms: Vec::new(),
            lifecycle: request.lifecycle,
            lifecycle_step: 0,
            lifecycle_counts: Vec::new(),
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
        CameraPreset::Close => {
            orbit.yaw = std::f32::consts::FRAC_PI_4;
            orbit.pitch = ISOMETRIC_PITCH_RAD;
            orbit.radius = (orbit.radius * 0.2).max(orbit.min_radius);
        }
        CameraPreset::LowPitch => {
            orbit.yaw = std::f32::consts::FRAC_PI_4;
            orbit.pitch = crate::orbit_camera::MIN_PITCH_RAD;
        }
        CameraPreset::FollowTarget => {
            orbit.yaw = std::f32::consts::FRAC_PI_4;
            orbit.pitch = ISOMETRIC_PITCH_RAD;
            if let Some(target_id) = request.target_id.as_deref() {
                // The recorded pose, looked up once: tracks first, then
                // truths, then nodes, as the markers were ordered.
                let position = replay_state.current_frame().and_then(|frame| {
                    frame
                        .tracks
                        .iter()
                        .find(|t| t.track_id == target_id)
                        .map(|t| t.position)
                        .or_else(|| {
                            frame
                                .truths
                                .iter()
                                .find(|t| t.target_id == target_id)
                                .map(|t| t.position)
                        })
                        .or_else(|| {
                            frame
                                .nodes
                                .iter()
                                .find(|n| n.node_id == target_id)
                                .map(|n| n.position)
                        })
                });
                if let Some(position) = position {
                    orbit.focus = Vec3::from_array(position);
                    orbit.radius = (orbit.radius * 0.25).max(orbit.min_radius);
                }
            }
        }
    }
}

/// Whether every pixel of `frame` is the same colour.
///
/// A cleared-but-undrawn target is uniform; a rendered scene is not. This is the
/// same property CI asserts on the saved PNG, checked here so the capture waits
/// for it rather than trusting a frame count to have covered it.
///
/// `frame` is the raw readback buffer, so it must be walked through [`pixel_rows`]
/// rather than read straight through. The padding bytes wgpu adds to each row are
/// never written by the render, so at any width whose row stride is not already
/// 256-byte aligned they stay zero — and comparing them as if they were pixels
/// makes *every* frame look non-uniform, including the cleared one this exists to
/// reject.
fn is_uniform(frame: &[u8], width: u32, height: u32) -> bool {
    // A trailing partial pixel cannot make a frame non-uniform, so the
    // remainder of each row is dropped rather than compared.
    let mut pixels = pixel_rows(frame, width, height).flat_map(|row| row.as_chunks::<4>().0.iter());
    let Some(first) = pixels.next() else {
        return true;
    };
    pixels.all(|pixel| pixel == first)
}

/// The main camera's projection, for probes.
type ProbeCamera<'w, 's> = Query<
    'w,
    's,
    (&'static Camera, &'static GlobalTransform),
    (With<MainCamera>, Without<ReconstructionCamera>),
>;

/// What the probe interactions drive: the picker's input and its results.
#[derive(bevy::ecs::system::SystemParam)]
pub struct ProbeInteraction<'w> {
    pointer: ResMut<'w, crate::picking::PointerInput>,
    log: Res<'w, crate::picking::PickLog>,
    hover: Res<'w, crate::picking::HoverState>,
    selection: ResMut<'w, crate::state::SelectionState>,
    markers: Res<'w, crate::presentation::CurrentRuntimeMarkers>,
}

fn marker_kind(name: &str) -> Option<crate::replay::MarkerKind> {
    use crate::replay::MarkerKind;
    match name {
        "node" => Some(MarkerKind::Node),
        "track" => Some(MarkerKind::Track),
        "truth" => Some(MarkerKind::Truth),
        _ => None,
    }
}

fn kind_name(kind: crate::replay::MarkerKind) -> &'static str {
    use crate::replay::MarkerKind;
    match kind {
        MarkerKind::Node => "node",
        MarkerKind::Track => "track",
        MarkerKind::Truth => "truth",
    }
}

/// A key as `{"kind", "id"}`, or `null`.
fn key_json(
    markers: &crate::presentation::CurrentRuntimeMarkers,
    key: Option<crate::presentation::EntityKey>,
) -> serde_json::Value {
    key.and_then(|key| markers.get(key))
        .map(|marker| serde_json::json!({"kind": kind_name(marker.kind), "id": &*marker.label}))
        .unwrap_or(serde_json::Value::Null)
}

/// The window pixel of a probe pointer, through the main camera.
fn pointer_pixel(context: &ProbeContext, pointer: &ProbePointer) -> Option<Vec2> {
    let (camera, transform) = context.camera.single().ok()?;
    let [x, y, z] = pointer.world;
    let at = camera
        .world_to_viewport(transform, Vec3::new(x as f32, y as f32, z as f32))
        .ok()?;
    Some(at + Vec2::from(pointer.offset_px))
}

/// What the probes need from the world at capture time.
#[derive(bevy::ecs::system::SystemParam)]
pub struct ProbeContext<'w, 's> {
    request: Res<'w, HeadlessRequest>,
    camera: ProbeCamera<'w, 's>,
    zones: Res<'w, crate::state::LoadedMissionZones>,
    scene: Res<'w, crate::schema::ScenePackage>,
    belief: ResMut<'w, crate::belief_render::BeliefLayer>,
    rasters: Res<'w, crate::terrain_rasters::TerrainRasters>,
}

fn write_probes(
    request: &ProbeRequest,
    context: &ProbeContext,
    interaction: &ProbeInteraction,
    clicks: &[serde_json::Value],
    padded: &[u8],
    target: &HeadlessTarget,
) -> Result<()> {
    let (camera, transform) = context
        .camera
        .single()
        .map_err(|error| anyhow::anyhow!("no main camera to project probes: {error}"))?;
    let rows: Vec<&[u8]> = pixel_rows(padded, target.width, target.height).collect();
    let pixel = |x: i64, y: i64| -> Option<[u8; 4]> {
        if x < 0 || y < 0 || x >= i64::from(target.width) || y >= i64::from(target.height) {
            return None;
        }
        let row = rows.get(y as usize)?;
        let at = x as usize * 4;
        Some([row[at], row[at + 1], row[at + 2], row[at + 3]])
    };
    let ground = context
        .scene
        .environment
        .terrain_summary
        .as_ref()
        .and_then(|summary| summary.min_height_m)
        .unwrap_or(0.0);
    let mut points = Vec::with_capacity(request.points.len() + request.elevated.len());
    let located = request
        .points
        .iter()
        .map(|xy| {
            let z = context
                .zones
                .terrain_mesh
                .as_ref()
                .and_then(|mesh| mesh.height_at(xy[0] as f32, xy[1] as f32))
                .unwrap_or(ground);
            [xy[0], xy[1], f64::from(z)]
        })
        .chain(request.elevated.iter().copied());
    for [x, y, z] in located {
        let (xy, z) = ([x, y], z as f32);
        let world = Vec3::new(xy[0] as f32, xy[1] as f32, z);
        let projected = camera.world_to_viewport(transform, world).ok();
        let (centre, window) = match projected {
            Some(p) => {
                let (px, py) = (p.x.floor() as i64, p.y.floor() as i64);
                let window: Vec<[u8; 4]> = (-PROBE_WINDOW_RADIUS..=PROBE_WINDOW_RADIUS)
                    .flat_map(|dy| {
                        (-PROBE_WINDOW_RADIUS..=PROBE_WINDOW_RADIUS).map(move |dx| (dx, dy))
                    })
                    .filter_map(|(dx, dy)| pixel(px + dx, py + dy))
                    .collect();
                (Some([px, py]), window)
            }
            None => (None, Vec::new()),
        };
        points.push(serde_json::json!({
            "xy": xy,
            "z": z,
            "pixel": centre,
            "rgba": centre.and_then(|[x, y]| pixel(x, y)),
            "window": window,
        }));
    }
    if let Some(parent) = request.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let report = context.rasters.report();
    std::fs::write(
        &request.output,
        serde_json::to_string(&serde_json::json!({
            "points": points,
            "clicks": clicks,
            "selected": key_json(
                &interaction.markers,
                interaction.selection.selected.as_ref().map(|s| s.key),
            ),
            "hovered": key_json(&interaction.markers, interaction.hover.hovered()),
            "rasters": {
                "belief_mode": report.belief_mode as u32,
                "land_cover_bound": report.land_cover_bound,
                "uploaded_tiles": report.uploaded_tiles,
                "waiting_tiles": report.waiting_tiles,
                "refused_tiles": report.refused_tiles,
            },
        }))?,
    )?;
    Ok(())
}

/// Drives the capture state machine described on [`HeadlessCapture`].
#[allow(clippy::too_many_arguments)]
fn headless_capture_system(
    mut capture: ResMut<HeadlessCapture>,
    mut replay_state: ResMut<ReplayState>,
    target: Option<Res<HeadlessTarget>>,
    receiver: Res<MainWorldReceiver>,
    asset_server: Res<AssetServer>,
    layer_roots: Query<&WorldAssetRoot>,
    mut exit: MessageWriter<AppExit>,
    mut probe: ProbeContext,
    mut interaction: ProbeInteraction,
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
                capture.timing.frames_waiting_assets += 1;
                return;
            }
            capture.timing.assets_ready = Some(Instant::now());
            // Late belief patches land now, after the textures were created.
            let late = probe
                .request
                .probe
                .as_ref()
                .map(|request| request.late_patches.clone())
                .unwrap_or_default();
            if let crate::belief_render::BeliefLayer::Ready(snapshot) = &mut *probe.belief {
                for patch in &late {
                    snapshot.apply_patch(patch);
                }
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
            let hash = frame_hash(&frame.pixels);
            let stable = if last_hash == Some(hash) {
                stable + 1
            } else {
                0
            };
            let seen = seen + 1;
            // Three conditions, and each rules out a failure the others miss:
            // `drawn` proves rendering started (so pipelines have been queued
            // at all), `!pipelines_pending` proves none is still compiling, and
            // stability proves the contents have settled.
            let drawn = !is_uniform(&frame.pixels, target.width, target.height);
            let ready = drawn && !frame.pipelines_pending;

            if drawn && capture.timing.drawn_at_frame.is_none() {
                capture.timing.drawn_at_frame = Some(seen);
            }
            if stable >= STABLE_FRAMES_REQUIRED && capture.timing.stable_at_frame.is_none() {
                capture.timing.stable_at_frame = Some(seen);
            }
            capture.timing.settle_frames = seen;

            capture.phase = if seen >= MAX_SETTLE_FRAMES {
                warn!(
                    "headless render did not settle after {seen} frames \
                     (drawn: {drawn}, pipelines pending: {}); capturing anyway",
                    frame.pipelines_pending
                );
                CapturePhase::Capture
            } else if seen >= min_frames && ready && stable >= STABLE_FRAMES_REQUIRED {
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
        CapturePhase::Interacting { step, waiting } => {
            let Some(request) = probe.request.probe.clone() else {
                capture.phase = CapturePhase::Capture;
                return;
            };
            // A click issued last frame: wait for the picker to handle it.
            if let Some(expected) = waiting {
                if interaction.log.clicks < expected {
                    return;
                }
                let pixel = interaction.pointer.position.map(|p| [p.x, p.y]);
                let selected = key_json(&interaction.markers, interaction.log.last);
                let ignored = interaction.log.last_ignored;
                capture.clicks.push(serde_json::json!({
                    "pixel": pixel,
                    "selected": selected,
                    "ignored": ignored,
                }));
                capture.phase = CapturePhase::Interacting {
                    step,
                    waiting: None,
                };
                return;
            }
            // Step 0 is the direct selection; then the clicks; then the hover.
            let clicks = request.clicks.len();
            if step == 0 {
                if let Some((kind, id)) = request.select.as_ref() {
                    let key =
                        marker_kind(kind).and_then(|kind| interaction.markers.key_of(kind, id));
                    if key.is_none() {
                        error!("probe select: no {kind} {id:?} in the frame");
                        capture.failed = true;
                    }
                    let markers = interaction.markers.as_ref();
                    crate::picking::select_key(interaction.selection.as_mut(), markers, key);
                }
                capture.phase = CapturePhase::Interacting {
                    step: 1,
                    waiting: None,
                };
            } else if step <= clicks {
                let pointer = &request.clicks[step - 1];
                let Some(pixel) = pointer_pixel(&probe, pointer) else {
                    error!("probe click {step}: {:?} is not in view", pointer.world);
                    capture.failed = true;
                    capture.phase = CapturePhase::Done;
                    return;
                };
                *interaction.pointer = crate::picking::PointerInput {
                    position: Some(pixel),
                    over_ui: false,
                    pressed: true,
                };
                capture.phase = CapturePhase::Interacting {
                    step: step + 1,
                    waiting: Some(interaction.log.clicks + 1),
                };
            } else {
                *interaction.pointer = crate::picking::PointerInput {
                    position: request
                        .hover
                        .as_ref()
                        .and_then(|p| pointer_pixel(&probe, p)),
                    over_ui: false,
                    pressed: false,
                };
                capture.interacted = true;
                capture.phase = CapturePhase::settling(MIN_SETTLE_FRAMES);
            }
        }
        CapturePhase::Capture => {
            // The probe's selection, clicks and hover run once, before the
            // still is taken.
            let interacts = probe.request.probe.as_ref().is_some_and(|request| {
                request.select.is_some() || !request.clicks.is_empty() || request.hover.is_some()
            });
            if interacts && !capture.interacted {
                capture.phase = CapturePhase::Interacting {
                    step: 0,
                    waiting: None,
                };
                return;
            }
            let Some(frame) = latest else {
                // Nothing has arrived yet; try again next tick.
                return;
            };
            // A shader that failed to compile leaves a plausible picture with
            // something missing from it. Refuse it rather than save it.
            if !frame.pipeline_errors.is_empty() {
                for error in &frame.pipeline_errors {
                    error!("headless render refused: a pipeline failed: {error}");
                }
                capture.failed = true;
                capture.phase = CapturePhase::Done;
                return;
            }

            // Once per run: a --record-dir sequence re-enters Capture for every
            // frame and only the first is the cold one.
            if capture.next_index == 0 {
                capture.timing.report();
            }

            if let Some(output) = capture.output.take() {
                if let Err(error) = save_frame(&output, &frame.pixels, target.width, target.height)
                {
                    error!("failed to write {}: {error:#}", output.display());
                    capture.failed = true;
                }
                if let Some(request) = probe.request.probe.as_ref() {
                    let clicks = std::mem::take(&mut capture.clicks);
                    if let Err(error) = write_probes(
                        request,
                        &probe,
                        &interaction,
                        &clicks,
                        &frame.pixels,
                        &target,
                    ) {
                        error!("failed to write probes: {error:#}");
                        capture.failed = true;
                    }
                }
                if capture.record_dir.is_none() {
                    capture.phase = if capture.lifecycle > 0 {
                        CapturePhase::Cycling
                    } else {
                        capture.after_lifecycle()
                    };
                    return;
                }
            }

            let Some(record_dir) = capture.record_dir.clone() else {
                capture.phase = CapturePhase::Done;
                return;
            };
            let index = capture.next_index;
            let path = record_dir.join(format!("frame_{index:05}.png"));
            if let Err(error) = save_frame(&path, &frame.pixels, target.width, target.height) {
                error!("failed to write {}: {error:#}", path.display());
                capture.failed = true;
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
        // `lifecycle_system` and `frame_stats_system` own these phases.
        CapturePhase::Cycling | CapturePhase::Measuring => {}
        CapturePhase::Done => {
            exit.write(if capture.failed {
                AppExit::error()
            } else {
                AppExit::Success
            });
        }
    }
}

impl HeadlessCapture {
    /// What follows the lifecycle (or the still, without one).
    fn after_lifecycle(&self) -> CapturePhase {
        if self.frame_stats > 0 {
            CapturePhase::Measuring
        } else {
            CapturePhase::Done
        }
    }
}

/// Steps in one lifecycle rotation.
const LIFECYCLE_STEPS: u32 = 8;

/// What the world holds: the counts a lifecycle must keep flat.
#[derive(bevy::ecs::system::SystemParam)]
struct WorldCounts<'w, 's> {
    entities: Query<'w, 's, Entity>,
    meshes: Res<'w, Assets<Mesh>>,
    standard: Res<'w, Assets<StandardMaterial>>,
    hologram: Option<Res<'w, Assets<crate::marker_meshes::HologramMaterial>>>,
    terrain: Option<Res<'w, Assets<crate::terrain_material::TerrainMaterial>>>,
    images: Res<'w, Assets<Image>>,
    ribbons: Option<Res<'w, crate::trail_ribbons::TrailRibbons>>,
}

impl WorldCounts<'_, '_> {
    fn snapshot(&self, rotation: u32) -> serde_json::Value {
        serde_json::json!({
            "rotation": rotation,
            "entities": self.entities.iter().count(),
            "meshes": self.meshes.len(),
            "standard_materials": self.standard.len(),
            "hologram_materials": self.hologram.as_ref().map(|a| a.len()),
            "terrain_materials": self.terrain.as_ref().map(|a| a.len()),
            "images": self.images.len(),
            "trail_ribbons": self.ribbons.as_ref().map(|r| r.ribbon_count()),
            "trail_samples": self.ribbons.as_ref().map(|r| r.store.total_samples()),
        })
    }
}

/// One frame of a lifecycle rotation (reload, seek to the start, toggle off
/// and on, tier step, Section and back, seek to the end): the step's change,
/// and after each rotation the world's counts (VERIFICATION_AND_ROLLOUT.md §5: seek, reload,
/// source switch and toggle repeatedly; counts must stay bounded).
fn lifecycle_system(
    mut capture: ResMut<HeadlessCapture>,
    mut replay: ResMut<ReplayState>,
    mut visibility: ResMut<crate::state::RuntimeOverlayVisibility>,
    mut ui_state: ResMut<crate::state::ViewerUiState>,
    mut tier: ResMut<crate::quality::QualityTier>,
    counts: WorldCounts,
) {
    if capture.phase != CapturePhase::Cycling {
        return;
    }
    let step = capture.lifecycle_step;
    // Counts are taken at the start of a rotation, once the last one's final
    // change has been presented.
    if step > 0 && step.is_multiple_of(LIFECYCLE_STEPS) {
        let rotation = step / LIFECYCLE_STEPS;
        capture.lifecycle_counts.push(counts.snapshot(rotation));
        if rotation >= capture.lifecycle {
            write_lifecycle(&mut capture);
            capture.phase = capture.after_lifecycle();
            return;
        }
    }
    // Ends on the last frame, so the counts see full trails.
    match step % LIFECYCLE_STEPS {
        0 => {
            // A reload: the same run again, as a new epoch.
            let document = replay.document.clone();
            replay.begin_epoch(document);
        }
        1 => replay.step_to(0),
        2 | 3 => {
            visibility.tracks = !visibility.tracks;
            visibility.nodes = !visibility.nodes;
            visibility.trails = !visibility.trails;
            visibility.zones = !visibility.zones;
            visibility.radar_rings = !visibility.radar_rings;
        }
        4 => {
            let all = crate::quality::QualityTier::ALL;
            let next = all
                .iter()
                .position(|t| *t == *tier)
                .map_or(0, |i| (i + 1) % all.len());
            *tier = all[next];
        }
        5 => ui_state.projection = crate::state::Projection::Section,
        6 => ui_state.projection = crate::state::Projection::Perspective,
        _ => {
            let last = replay.frame_count().saturating_sub(1);
            replay.step_to(last);
        }
    }
    capture.lifecycle_step += 1;
}

fn write_lifecycle(capture: &mut HeadlessCapture) {
    let Some(path) = capture
        .stats_path
        .as_ref()
        .map(|stats| stats.with_extension("").with_extension("lifecycle.json"))
    else {
        return;
    };
    let report = serde_json::json!({
        "kind": "argusnet.lifecycle-counts",
        "version": 1,
        "steps_per_rotation": LIFECYCLE_STEPS,
        "rotations": capture.lifecycle_counts,
    });
    let written = serde_json::to_string_pretty(&report)
        .map_err(anyhow::Error::from)
        .and_then(|text| std::fs::write(&path, text).map_err(anyhow::Error::from));
    if let Err(error) = written {
        error!("failed to write {}: {error:#}", path.display());
        capture.failed = true;
    }
}

/// A frame-time distribution: nearest-rank percentiles, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct FrameSummary {
    pub frames: usize,
    pub mean_ms: f32,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub max_ms: f32,
    /// Frames over 100 ms: VIEWER_UI's "no unexplained stall".
    pub stalls_over_100ms: usize,
}

/// Summarise frame times; `None` for no samples.
pub fn frame_summary(samples_ms: &[f32]) -> Option<FrameSummary> {
    if samples_ms.is_empty() {
        return None;
    }
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(f32::total_cmp);
    let rank = |p: f32| {
        let index = ((p * sorted.len() as f32).ceil() as usize).clamp(1, sorted.len()) - 1;
        sorted[index]
    };
    Some(FrameSummary {
        frames: sorted.len(),
        mean_ms: sorted.iter().sum::<f32>() / sorted.len() as f32,
        p50_ms: rank(0.50),
        p95_ms: rank(0.95),
        p99_ms: rank(0.99),
        max_ms: sorted[sorted.len() - 1],
        stalls_over_100ms: sorted.iter().filter(|ms| **ms > 100.0).count(),
    })
}

/// Times frames after the still (G2), then writes the statistics, with the
/// warm-up reported apart from them rather than discarded, and exits.
fn frame_stats_system(
    mut capture: ResMut<HeadlessCapture>,
    request: Res<HeadlessRequest>,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
) {
    if capture.phase != CapturePhase::Measuring {
        return;
    }
    let now = Instant::now();
    if let Some(last) = capture.last_frame {
        let ms = now.duration_since(last).as_secs_f32() * 1000.0;
        capture.samples_ms.push(ms);
    }
    capture.last_frame = Some(now);
    if capture.samples_ms.len() < capture.frame_stats as usize {
        return;
    }
    let adapter = adapter.map(|info| {
        serde_json::json!({
            "name": info.name,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
            "driver": info.driver,
            "driver_info": info.driver_info,
        })
    });
    let software = adapter
        .as_ref()
        .is_some_and(|info| info["device_type"] == "Cpu");
    let warmup_s = capture
        .timing
        .assets_ready
        .map(|ready| ready.duration_since(capture.timing.started).as_secs_f32());
    let report = serde_json::json!({
        "kind": "argusnet.g2-frame-stats",
        "version": 1,
        "viewport": [request.width, request.height],
        "camera": format!("{:?}", request.camera),
        "quality": request.quality.name(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "adapter": adapter,
        "renderer_class": if software { "software" } else { "hardware" },
        "timing_scope": "unpaced headless loop: main and render world, including the \
                         per-frame image readback headless performs",
        "gpu_time_ms": null,
        "gpu_time_unavailable": "GPU timestamp queries are not recorded yet",
        "warmup": {
            "assets_s": warmup_s,
            "frames_waiting_assets": capture.timing.frames_waiting_assets,
            "drawn_at_frame": capture.timing.drawn_at_frame,
            "settle_frames": capture.timing.settle_frames,
        },
        "summary": frame_summary(&capture.samples_ms),
        "samples_ms": capture.samples_ms,
    });
    if let Some(path) = capture.stats_path.clone() {
        let written = serde_json::to_string_pretty(&report)
            .map_err(anyhow::Error::from)
            .and_then(|text| std::fs::write(&path, text).map_err(anyhow::Error::from));
        if let Err(error) = written {
            error!("failed to write {}: {error:#}", path.display());
            capture.failed = true;
        }
    }
    capture.phase = CapturePhase::Done;
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

/// The real pixel bytes of a readback buffer, one row at a time.
///
/// wgpu pads every copied row out to a 256-byte boundary, so a buffer is only
/// `width * 4` bytes of image per `align_copy_bytes_per_row(width * 4)` bytes of
/// buffer. Anything that reads the readback as pixels has to go through here;
/// reading it flat silently mixes padding into the image.
fn pixel_rows(padded: &[u8], width: u32, height: u32) -> impl Iterator<Item = &[u8]> {
    let row_bytes = width as usize * TextureFormat::Rgba8UnormSrgb.pixel_size().unwrap_or(4);
    // A zero-width target would give a zero stride, and `chunks(0)` panics.
    // `into_request` clamps the extent, so this only guards the unreachable case.
    let stride = RenderDevice::align_copy_bytes_per_row(row_bytes).max(1);
    padded
        .chunks(stride)
        .take(height as usize)
        .map(move |row| &row[..row_bytes.min(row.len())])
}

/// Trims the row padding wgpu's copy alignment adds, so the bytes match the
/// image's real width.
fn unpad_rows(padded: &[u8], width: u32, height: u32) -> Vec<u8> {
    pixel_rows(padded, width, height)
        .flatten()
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
mod uniformity_tests {
    use super::{is_uniform, RenderDevice};

    const CLEAR: [u8; 4] = [12, 14, 20, 255];

    /// 64 px * 4 bytes = 256, exactly the copy alignment, so the readback for
    /// this width carries no padding.
    const ALIGNED_W: u32 = 64;

    /// Builds a readback buffer the way wgpu delivers one: `width * 4` bytes of
    /// image per row, then zero padding out to the 256-byte copy alignment.
    fn readback(width: u32, height: u32, pixel: [u8; 4]) -> Vec<u8> {
        let row_bytes = width as usize * 4;
        let stride = RenderDevice::align_copy_bytes_per_row(row_bytes);
        let mut buffer = vec![0u8; stride * height as usize];
        for row in 0..height as usize {
            for x in 0..width as usize {
                buffer[row * stride + x * 4..row * stride + x * 4 + 4].copy_from_slice(&pixel);
            }
        }
        buffer
    }

    /// A cleared-but-undrawn target: every pixel identical. This is the state
    /// the capture must not mistake for a finished render, and the reason a
    /// stability check alone is not enough.
    #[test]
    fn a_cleared_target_is_uniform() {
        assert!(is_uniform(&readback(ALIGNED_W, 4, CLEAR), ALIGNED_W, 4));
    }

    /// The defect this signature exists for. At a width whose row stride is not
    /// already 256-byte aligned, wgpu pads each row with bytes the render never
    /// writes. Read flat, those zeros differ from the clear colour, so a
    /// perfectly empty target reports as drawn — and since a cleared target is
    /// also stable, the capture releases on an empty frame.
    #[test]
    fn row_padding_does_not_make_a_cleared_target_look_drawn() {
        let width = 650;
        let height = 3;
        let row_bytes = width as usize * 4;
        assert!(
            RenderDevice::align_copy_bytes_per_row(row_bytes) > row_bytes,
            "this width must need padding for the test to mean anything"
        );

        let cleared = readback(width, height, CLEAR);
        assert!(
            cleared.contains(&0),
            "the padding should be present and unwritten"
        );
        assert!(is_uniform(&cleared, width, height));
    }

    /// One pixel differing anywhere is enough to say something drew — including
    /// the last pixel of a padded row, which is the one nearest the bytes that
    /// have to be ignored.
    #[test]
    fn a_single_differing_pixel_is_not_uniform() {
        for width in [ALIGNED_W, 650] {
            let row_bytes = width as usize * 4;
            let stride = RenderDevice::align_copy_bytes_per_row(row_bytes);

            let mut first = readback(width, 3, CLEAR);
            first[6] = 21;
            assert!(!is_uniform(&first, width, 3), "width {width}, first pixel");

            let mut last_in_row = readback(width, 3, CLEAR);
            last_in_row[row_bytes - 4] = 13;
            assert!(
                !is_uniform(&last_in_row, width, 3),
                "width {width}, row end"
            );

            let mut last_row = readback(width, 3, CLEAR);
            last_row[2 * stride] = 13;
            assert!(!is_uniform(&last_row, width, 3), "width {width}, last row");
        }
    }

    /// An empty or truncated buffer must not be reported as drawn: treating a
    /// missing frame as ready would capture nothing at all.
    #[test]
    fn a_frame_too_short_to_hold_a_pixel_is_uniform() {
        assert!(is_uniform(&[], ALIGNED_W, 4));
        assert!(is_uniform(&[1, 2, 3], 1, 1));
    }

    /// Alpha alone distinguishes two pixels, since the readback is RGBA.
    #[test]
    fn alpha_counts_as_a_difference() {
        let mut frame = readback(ALIGNED_W, 2, CLEAR);
        frame[7] = 254;
        assert!(!is_uniform(&frame, ALIGNED_W, 2));
    }
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
            probe: None,
            quality: Default::default(),
            frame_stats: 0,
            lifecycle: 0,
        }
    }

    #[test]
    fn frame_summaries_use_nearest_rank_and_count_stalls() {
        assert_eq!(frame_summary(&[]), None);
        let mut samples: Vec<f32> = (1..=100).map(|ms| ms as f32).collect();
        samples.push(250.0);
        let summary = frame_summary(&samples).unwrap();
        assert_eq!(summary.frames, 101);
        assert_eq!(summary.p50_ms, 51.0);
        assert_eq!(summary.p95_ms, 96.0);
        assert_eq!(summary.p99_ms, 100.0);
        assert_eq!(summary.max_ms, 250.0);
        assert_eq!(summary.stalls_over_100ms, 1);
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
            probe: None,
            quality: crate::quality::QualityTier::Low,
            frame_stats: 0,
            lifecycle: 0,
        };
        let (view_mode, request) = opts.into_request().unwrap();
        assert_eq!(view_mode, ViewMode::Split);
        assert_eq!(request.output, Some(PathBuf::from("still.png")));
        assert_eq!(request.record_dir, Some(PathBuf::from("frames")));
        assert_eq!(request.camera, CameraPreset::Isometric);
        assert_eq!(request.target_id.as_deref(), Some("truth-1"));
        assert_eq!(request.quality, crate::quality::QualityTier::Low);
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
