use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
// Bevy 0.20's prelude brings its own `ContextExt`, whose `context` and
// `with_context` turn a result into a `BevyError`. Naming anyhow's trait
// `ContextExt` here shadows the prelude's, so these calls stay anyhow's.
use anyhow::Context as ContextExt;
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::Hdr;
use bevy::camera::{ClearColorConfig, Exposure, Viewport};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::world_serialization::{WorldAsset, WorldAssetRoot};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

/// Render layer holding only the scan-map reconstruction mesh. The reconstruction
/// camera renders this layer alone, so the Split-mode right pane shows the believed
/// terrain on an otherwise empty background (no real terrain, drones, or gizmos).
pub(crate) const RECON_LAYER: usize = 1;

use crate::glyphs::{GlyphProjection, NODE_GLYPH_PX, TRACK_GLYPH_PX, TRUTH_GLYPH_PX};
use crate::headless::{HeadlessRequest, HeadlessTarget};
use crate::mission_zones::{self, build_projected_badges_system, ProjectedZoneBadges};
use crate::orbit_camera::{OrbitCamera, OrbitCameraPlugin};
use crate::presentation::PresentationChange;
use crate::replay::{
    MarkerKind, RejectedObservation, ReplayDocument, ReplayState, RuntimeMarker, TerrainViewerMesh,
};
use crate::schema::ScenePackage;
use crate::sky::{spawn_sky, GradientSkyPlugin, SkyMaterial, HORIZON_COLOR};
use crate::state::{
    CurrentFrameMetrics, CurrentRuntimeMarkers, LayerEntityIndex, LayerVisibilityState,
    LoadedMissionZones, MissionOverlaySettings, NodeDetails, ReconstructionCamera,
    ReconstructionCloud, RuntimeOverlayVisibility, Selection, SelectionDetails, SelectionState,
    SimPhase, SimulationRunner, TrackDetails, TruthDetails, ViewMode, ViewerUiState,
    WorkingSceneRoot, ZoneFocus,
};
use crate::terrain_material::{is_terrain_layer, TerrainLayerRoot, TerrainMaterialPlugin};
use crate::ui::viewer_ui_system;

/// Both 3-D cameras' exposure: EV100 12, Bevy's `Exposure::OVERCAST`, which
/// matches the 12 000 lux key light. The terrain faces up and is lit by that
/// sun (`docs/KNOWN_GAPS.md`); until it did, the ground was lit by the ambient
/// term alone and the camera was opened three stops (EV100 9) to make up for
/// it, which washed every sunlit surface out to white.
const SCENE_EXPOSURE: Exposure = Exposure { ev100: 12.0 };

/// Marker component on the persistent GPU line-list mesh used to render the
/// LiDAR reconstruction point cloud.  Rebuilt only when the cloud is dirty.
#[derive(Component)]
struct ReconstructionMeshMarker;

/// A Tools → Scenario run in progress -- inserted as a resource while it
/// runs. The Rust simulation runs on its own thread (D5): no Python child.
#[derive(Resource)]
struct SimulationProcess {
    worker: Option<JoinHandle<Result<(), String>>>,
    progress: Arc<RunProgress>,
    phase: SimPhase,
    replay_path: PathBuf,
    next_scene_root: PathBuf,
}

#[derive(Component)]
pub struct MainCamera;

struct ReloadedSceneBundle {
    scene_package: ScenePackage,
    replay_state: ReplayState,
    mission_zones: LoadedMissionZones,
    layer_visibility: LayerVisibilityState,
    should_reframe_camera: bool,
}

pub fn run(
    scene_path: impl AsRef<Path>,
    initial_view_mode: ViewMode,
    autoplay: bool,
    quality: Option<crate::quality::QualityTier>,
) -> Result<()> {
    run_inner(
        scene_path.as_ref(),
        None,
        initial_view_mode,
        autoplay,
        quality,
        None,
        None,
    )
}

#[cfg(feature = "live-stream")]
pub fn run_live(
    scene_path: impl AsRef<Path>,
    endpoint: String,
    tls: crate::live::LiveTlsConfig,
    initial_view_mode: ViewMode,
    autoplay: bool,
    quality: Option<crate::quality::QualityTier>,
) -> Result<()> {
    run_inner(
        scene_path.as_ref(),
        Some((endpoint, tls)),
        initial_view_mode,
        autoplay,
        quality,
        None,
        None,
    )
}

/// Open `scene_path` and drive an in-process simulation of `run_path` (a
/// scenario config or bundle) in it (D1): the run ticks on its own thread and
/// the viewer presents its frames as they commit. Without a scene, a run with
/// a prior world gets one built in process from its own metadata
/// (`argusnet_sim::scene_build`); a run with no prior world opens on the
/// map's bounds with nothing drawn, and what appears is what the run reveals.
pub fn run_sim(
    scene_path: Option<std::path::PathBuf>,
    run_path: std::path::PathBuf,
    options: argusnet_sim::run_options::RunOptions,
    initial_view_mode: ViewMode,
    quality: Option<crate::quality::QualityTier>,
) -> Result<()> {
    // Refuse a run that cannot start before opening a window: an unreadable
    // scenario, an unsupported delivery profile, an impossible fault
    // manifest, a mission rate the run's timestep cannot be scheduled at.
    // The worker would only report the same refusal in the strip.
    let probe = crate::sim_session::load_run(&run_path, &options)
        .map_err(|error| anyhow::anyhow!(error))?;
    if !options.lane_rates.is_empty() {
        argusnet_sim::worker::scheduled_executor(&probe, &options.lane_rates)
            .map_err(|error| anyhow::anyhow!(error))?;
    }
    let run_meta = probe.replay_extra_meta().clone();
    let stamp = || {
        format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos())
        )
    };
    let no_prior = run_meta
        .get("world_knowledge")
        .and_then(|knowledge| knowledge.get("knowledge"))
        .and_then(|knowledge| knowledge.as_str())
        == Some("none");
    // No scene for a run with a prior world: build one in process from the
    // run's own metadata (terrain mesh, obstacles), as `argusnet build-scene`
    // would from its replay, with no Python (D2). A run with no prior world
    // gets a package with the map's bounds and nothing in it: the world is
    // only what the run reveals.
    let built = match (&scene_path, no_prior) {
        (None, false) => {
            let mut meta: serde_json::Map<String, serde_json::Value> =
                run_meta.clone().into_iter().collect();
            meta.entry("scenario_name")
                .or_insert_with(|| serde_json::json!(probe.run_id()));
            let dir = std::env::temp_dir().join(format!("argusnet-scene-{}", stamp()));
            match argusnet_sim::scene_build::build_scene(
                &dir,
                &serde_json::Value::Object(meta),
                None,
                serde_json::json!({ "replay_source": run_path.display().to_string() }),
            ) {
                Ok(built) => Some(built.root),
                Err(error) => {
                    warn!("could not build a scene from the run ({error}); opening without one");
                    None
                }
            }
        }
        _ => None,
    };
    let scene_path = match scene_path.or(built) {
        Some(path) => path,
        None => {
            let bounds = probe.map_bounds_m();
            let floor = probe.launch_floor_m();
            let dir = std::env::temp_dir().join(format!("argusnet-unmapped-{}", stamp()));
            crate::schema::write_unmapped_scene(
                &dir,
                probe.run_id(),
                [
                    bounds.x_min_m,
                    bounds.x_max_m,
                    bounds.y_min_m,
                    bounds.y_max_m,
                ],
                (floor - 100.0, floor + 50.0),
            )?
        }
    };
    drop(probe);
    run_inner(
        scene_path.as_path(),
        None,
        initial_view_mode,
        false,
        quality,
        None,
        Some((run_path, options, run_meta)),
    )
}

#[cfg(feature = "live-stream")]
type LiveEndpoint = Option<(String, crate::live::LiveTlsConfig)>;
#[cfg(not(feature = "live-stream"))]
type LiveEndpoint = Option<String>;

/// Renders the scene through the real pipeline with no window, saving PNGs.
///
/// This is the same App as the interactive viewer — same systems, same
/// materials, same camera — driven by `ScheduleRunnerPlugin` into an offscreen
/// image instead of a surface, so a CI still shows what an operator sees.
pub fn run_headless(
    scene_path: impl AsRef<Path>,
    initial_view_mode: ViewMode,
    request: HeadlessRequest,
) -> Result<()> {
    let quality = Some(request.quality);
    run_inner(
        scene_path.as_ref(),
        None,
        initial_view_mode,
        false,
        quality,
        Some(request),
        None,
    )
}

fn run_inner(
    scene_path: &Path,
    live_endpoint: LiveEndpoint,
    initial_view_mode: ViewMode,
    autoplay: bool,
    quality: Option<crate::quality::QualityTier>,
    headless: Option<HeadlessRequest>,
    sim_run: Option<(
        std::path::PathBuf,
        argusnet_sim::run_options::RunOptions,
        std::collections::BTreeMap<String, serde_json::Value>,
    )>,
) -> Result<()> {
    let (scene_package, working_scene_root) = initialize_scene_package(scene_path)?;
    let replay_document = scene_package
        .replay
        .clone()
        .map(ReplayDocument::try_from)
        .transpose()
        .context("failed to parse packaged replay.json")?;

    let mission_zones = replay_document
        .as_ref()
        .map(|doc| doc.zones())
        .unwrap_or_default();
    let terrain_mesh = replay_document
        .as_ref()
        .and_then(|doc| doc.terrain_viewer_mesh());
    if terrain_mesh.is_none() && replay_document.is_some() {
        warn!("Replay has no terrain mesh — 3-D elevation will fall back to entity altitude");
    }
    let mut zone_overlap_model =
        mission_zones::build_zone_overlap_model(&mission_zones, ZoneFocus::None);
    zone_overlap_model.generation = 1;

    let layer_visibility = merged_layer_visibility(&scene_package, None);

    let asset_root = working_scene_root.asset_root.clone();

    // Pixel probes (P2.1) bring their own belief, with the overlay on.
    let mut runtime_visibility = RuntimeOverlayVisibility::default();
    let mut belief_layer = crate::belief_render::BeliefLayer::default();
    if let Some(probe) = headless.as_ref().and_then(|request| request.probe.as_ref()) {
        if let Some(overlays) = probe.overlays.clone() {
            runtime_visibility = overlays;
        }
        if let Some(store) = probe.belief.as_ref() {
            belief_layer = match crate::belief_render::BeliefRenderSnapshot::from_store(store) {
                Ok(snapshot) => crate::belief_render::BeliefLayer::Ready(Box::new(snapshot)),
                Err(reason) => crate::belief_render::BeliefLayer::Unavailable(reason),
            };
        }
        runtime_visibility.belief = probe.belief_overlay;
    }

    let mut replay_state = ReplayState::new(replay_document);
    // Autoplay starts the timeline immediately so the scan-map reconstruction
    // accumulates without the operator pressing Space (has no effect on a
    // single-frame replay).
    if autoplay {
        replay_state.playing = true;
    }

    let mut app = App::new();
    // The sky dome covers every direction the main camera can look, so this is
    // now only what shows for the one frame before the dome spawns, and behind
    // any camera that opts out of layer 0.
    // Presentation preferences (P2.4b): the operator's camera bookmarks and
    // key remaps. Headless reads none, so a render does not depend on who
    // ran it.
    let mut prefs = if headless.is_some() {
        crate::prefs::ViewerPrefs::default()
    } else {
        crate::prefs::ViewerPrefs::load(
            crate::prefs::default_path(),
            &scene_package.manifest.scene_id,
        )
    };
    let (bindings, binding_problems) =
        crate::bindings::Bindings::with_overrides(&prefs.file.bindings);
    prefs.problems.extend(binding_problems);
    // The command line wins over the preferences file; headless runs read no
    // preferences, so a render's tier is the one it asked for.
    let quality = match quality {
        Some(tier) => tier,
        None => prefs.quality_tier(),
    };
    for problem in &prefs.problems {
        warn!("viewer preferences: {problem}");
    }

    app.insert_resource(ClearColor(HORIZON_COLOR))
        .insert_resource(prefs)
        .insert_resource(quality)
        .insert_resource(bindings)
        .init_resource::<crate::bindings::KeyboardGate>()
        .init_resource::<crate::camera_control::EscapeUsed>()
        .init_resource::<crate::projection::SavedPerspective>()
        .init_resource::<crate::projection::SectionSlab>()
        .init_resource::<crate::projection::SectionCut>()
        .init_resource::<crate::ui::conditions::ActiveConditions>()
        .init_resource::<crate::ui::preferences::KeyCapture>()
        .insert_resource(scene_package)
        .insert_resource(working_scene_root)
        .insert_resource(replay_state)
        .insert_resource(layer_visibility)
        .insert_resource(runtime_visibility)
        .insert_resource(belief_layer)
        .insert_resource(CurrentRuntimeMarkers::default())
        .insert_resource(CurrentFrameMetrics::default())
        .insert_resource(SelectionState::default())
        .insert_resource(zone_overlap_model)
        .insert_resource(LoadedMissionZones {
            zones: mission_zones,
            terrain_mesh,
        })
        .insert_resource(LayerEntityIndex::default())
        .init_resource::<crate::sensor_volumes::SensorVolumes>()
        .init_resource::<crate::sensor_volumes::RangeRings>()
        .init_resource::<crate::trail_ribbons::TrailRibbons>()
        .init_resource::<crate::zone_walls::ZoneWalls>()
        .init_resource::<crate::picking::PointerInput>()
        .init_resource::<crate::picking::PickGeneration>()
        .init_resource::<crate::picking::HoverState>()
        .init_resource::<crate::picking::PickLog>()
        .init_resource::<crate::selection_outline::SelectionRings>()
        .insert_resource(SimulationRunner::default())
        .insert_resource(ProjectedZoneBadges::default())
        .insert_resource(MissionOverlaySettings::default())
        .insert_resource(ReconstructionCloud::default())
        .init_resource::<crate::belief_map::BeliefMapView>()
        .insert_resource(ViewerUiState::default())
        .insert_resource(initial_view_mode);

    let asset_plugin = AssetPlugin {
        file_path: asset_root.to_string_lossy().into_owned(),
        ..default()
    };
    if let Some(request) = headless {
        // Paced at 60 Hz for stills; unpaced when frames are being timed, so
        // a sample is the frame's own cost rather than the pacing's.
        let pacing = if request.frame_stats > 0 {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_secs_f64(1.0 / 60.0)
        };
        // No window at all: winit panics without a display server, and
        // ScheduleRunnerPlugin drives the loop in its place.
        //
        // egui is left out entirely — there is no surface for it to draw on, and
        // its context pass panics on unapplied texture deltas when nothing
        // renders them. The two systems that consult `EguiContexts` to see
        // whether the UI owns the pointer or keyboard take it as an `Option`, so
        // they degrade to "the UI is not consuming this" rather than failing
        // parameter validation when the plugin is absent.
        app.add_plugins(
            DefaultPlugins
                .set(asset_plugin)
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>(),
        )
        .add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(pacing))
        .add_plugins(crate::headless::HeadlessRenderPlugin(request));
    } else {
        app.add_plugins(DefaultPlugins.set(asset_plugin).set(WindowPlugin {
            primary_window: Some(Window {
                title: "ArgusNet Viewer".into(),
                resolution: (1440_u32, 920_u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        // The shell's context goes on the main camera, chosen, not on "the
        // first camera created": with the reconstruction camera (inactive
        // until Split) also spawned, the automatic choice could land there,
        // and then the UI was drawn every frame and shown nowhere -- seen in
        // the real window when an unrelated change reordered the schedule.
        .insert_resource(bevy_egui::EguiGlobalSettings {
            auto_create_primary_context: false,
            ..default()
        })
        .add_systems(Startup, spawn_ui_camera.after(setup_world))
        // bevy_egui 0.42 replaced `EguiSet::BeginPass` inside `Update` with a
        // dedicated schedule that runs between egui's begin/end pass, so UI
        // systems must be registered there rather than ordered within `Update`.
        .add_systems(
            EguiPrimaryContextPass,
            (
                crate::theme::install_theme_system,
                crate::ui::preferences::apply_accessibility_system,
                viewer_ui_system,
                crate::ui::conditions::conditions_ui_system,
                crate::ui::preferences::preferences_ui_system,
                crate::belief_map::belief_readout_ui_system,
            )
                .chain(),
        );
    }

    app.add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugins(OrbitCameraPlugin)
        .add_plugins((
            GradientSkyPlugin,
            crate::marker_meshes::MarkerMeshPlugin,
            crate::power_lines::PowerLinePlugin,
            crate::trees::TreePlugin,
            crate::building_material::BuildingMaterialPlugin,
            TerrainMaterialPlugin,
            crate::terrain_rasters::TerrainRastersPlugin,
        ))
        .add_systems(Startup, (setup_world, setup_reconstruction_mesh));
    configure_viewer_sets(&mut app);
    // After both writers of the main camera's transform (C5.1).
    app.add_systems(Update, crate::ui::conditions::detect_conditions_system);
    app.add_systems(
        Update,
        (
            crate::quality::sync_quality_choice_system,
            crate::quality::apply_quality_tier_system,
        )
            .chain()
            .before(sync_fog_to_camera_system),
    );
    app.add_systems(
        Update,
        (
            crate::projection::apply_projection_system
                .after(crate::orbit_camera::update_orbit_camera)
                .after(crate::camera_control::drive_main_camera_system),
            crate::projection::draw_section_cut_system
                .after(crate::projection::apply_projection_system),
        ),
    );
    app.add_systems(
        Update,
        (
            (
                crate::bindings::keyboard_gate_system,
                advance_playback_system,
                keyboard_playback_system,
                crate::ui::shell_keyboard_system,
                sync_split_viewports_system,
                crate::picking::pointer_from_window_system,
            )
                .chain()
                .in_set(ViewerSet::Input),
            // Pull in whatever the simulation produced, then present it.
            (
                spawn_simulation_system,
                poll_simulation_system,
                sync_current_markers_system,
            )
                .chain()
                .in_set(ViewerSet::SnapshotApply),
            (
                mission_zones::refresh_zone_overlap_model_system,
                crate::terrain_rasters::sync_terrain_rasters_system,
                crate::marker_meshes::sync_marker_meshes,
                // Persistent sensor, range and trail geometry (P2.2).
                crate::sensor_volumes::sync_sensor_volumes,
                crate::sensor_volumes::sync_range_rings,
                crate::trail_ribbons::sync_trail_ribbons,
                crate::zone_walls::sync_zone_walls,
                // The run's towers and conductors (R3).
                crate::power_lines::sync_power_lines,
                // The spans the drones' own map holds (R3).
                crate::power_lines::sync_mapped_wires,
                // The run's trees (R2).
                crate::trees::sync_trees,
                sync_base_layer_visibility_system,
                sync_fog_to_camera_system,
                // Reconstruction owns real state, so it runs regardless of its
                // overlay toggle -- see the note on the run conditions below.
                update_reconstruction_system,
                maintain_reconstruction_mesh_system,
                // A self-built map (world_knowledge: none), as of the cursor.
                crate::belief_map::sync_belief_map_system,
                crate::belief_map::render_belief_chunks_system,
                crate::belief_map::sync_belief_viewer_state_system,
            )
                .chain()
                .in_set(ViewerSet::Presentation),
            // Ray-cast picking, hover and keyboard selection (P2.3).
            (
                crate::picking::tag_pick_occluders_system,
                crate::picking::pick_world_system,
                // The camera before the selection keys, so a camera Escape
                // (leaving free-fly, follow or chase) is not also a clear.
                crate::camera_control::drive_main_camera_system,
                crate::ui::close_event_on_escape_system,
                crate::picking::keyboard_selection_system,
            )
                .chain()
                .in_set(ViewerSet::Picking),
            // Immediate-mode overlays, each gated on its own toggle.
            (
                crate::selection_outline::sync_selection_rings,
                draw_runtime_overlays_system,
                mission_zones::draw_mission_zones_system,
                build_projected_badges_system,
                draw_scan_grid_system.run_if(scan_grid_enabled),
                draw_poi_markers_system.run_if(poi_markers_enabled),
                draw_team_localization_system.run_if(loc_ellipses_enabled),
                draw_coord_frame_system.run_if(coord_frame_enabled),
                draw_egress_paths_system.run_if(egress_paths_enabled),
                crate::belief_map::belief_overlays_system,
            )
                .chain()
                .in_set(ViewerSet::Overlays),
        ),
    );

    #[cfg(feature = "live-stream")]
    if let Some((endpoint, tls)) = live_endpoint {
        // Refuses plaintext off loopback, so this fails before the window opens
        // rather than silently streaming in the clear.
        app.insert_resource(crate::live::connect(endpoint, tls)?);
        app.add_systems(
            Update,
            crate::live::ingest_live_frames_system
                .in_set(ViewerSet::Input)
                .before(advance_playback_system),
        );
        app.add_systems(
            EguiPrimaryContextPass,
            crate::live::live_status_ui_system.after(viewer_ui_system),
        );
    }
    #[cfg(not(feature = "live-stream"))]
    let _ = live_endpoint;

    if let Some((run_path, options, run_meta)) = sim_run {
        app.insert_resource(
            crate::sim_session::SimSession::start(
                run_path,
                options,
                argusnet_sim::worker::Pacing::RealTime,
            )
            .with_run_meta(run_meta),
        );
        app.init_resource::<crate::sim_session::SimSourceState>();
        app.add_systems(
            Update,
            crate::sim_session::ingest_sim_frames_system
                .in_set(ViewerSet::Input)
                .before(advance_playback_system),
        );
        app.add_systems(
            EguiPrimaryContextPass,
            crate::sim_session::sim_controls_ui_system.after(viewer_ui_system),
        );
    }

    // A headless capture that refused its frame exits with an error; so
    // must the process.
    if let AppExit::Error(code) = app.run() {
        anyhow::bail!("the viewer exited with error code {code}");
    }

    Ok(())
}

/// The viewer's frame, in order (P2.0).
///
/// * `Input`: advance time, take operator input, ingest live frames.
/// * `SnapshotApply`: take in new runs, then bring the presentation up to
///   date with the replay cursor (the snapshot adapter).
/// * `Presentation`: persistent entities, visibility, fog, reconstruction.
/// * `Picking`: what the pointer selects, against this frame's presentation.
/// * `Overlays`: immediate-mode gizmos.
///
/// The operator UI runs in egui's own pass (`EguiPrimaryContextPass`), after
/// `Update`, so it reads a finished frame.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewerSet {
    Input,
    SnapshotApply,
    Presentation,
    Picking,
    Overlays,
}

/// Order [`ViewerSet`] in `Update`.
pub fn configure_viewer_sets(app: &mut App) {
    app.configure_sets(
        Update,
        (
            ViewerSet::Input,
            ViewerSet::SnapshotApply,
            ViewerSet::Presentation,
            ViewerSet::Picking,
            ViewerSet::Overlays,
        )
            .chain(),
    );
}

/// Keeps fog distances tracking the orbit camera.
///
/// `DistanceFog` measures from the camera, and this camera sits outside the
/// scene looking in: at the default framing it is roughly 1.5 scene-widths away,
/// and zooming changes that by an order of magnitude. Fixed distances therefore
/// cannot work — tuned for the default view they black out a zoomed-in one, and
/// tuned for close range they never engage at all. Anchoring the band to the
/// camera's own distance keeps the near edge of the terrain clear and lets the
/// far edge recede, at every zoom level.
fn sync_fog_to_camera_system(
    mut fog_q: Query<(&mut DistanceFog, &OrbitCamera), With<MainCamera>>,
    scene_package: Res<ScenePackage>,
) {
    let Ok((mut fog, orbit)) = fog_q.single_mut() else {
        return;
    };
    let bounds = &scene_package.environment.bounds_xy_m;
    let span = (bounds.x_max_m - bounds.x_min_m)
        .max(bounds.y_max_m - bounds.y_min_m)
        .max(200.0);

    let (start, end) = fog_band(orbit.radius, span);
    fog.falloff = FogFalloff::Linear { start, end };
}

/// Near and far fog distances for a camera `radius` from the centre of a scene
/// `span` metres across.
///
/// The terrain occupies roughly `radius - span/2 ..= radius + span/2` in view
/// depth. The haze starts a little past the near edge so the foreground stays
/// readable, and finishes beyond the far edge so the horizon fades out instead
/// of ending on a line.
fn fog_band(radius: f32, span: f32) -> (f32, f32) {
    let near_edge = (radius - span * 0.5).max(1.0);
    let far_edge = radius + span * 0.5;
    (near_edge + span * 0.15, far_edge + span * 0.35)
}

// ---------------------------------------------------------------------------
// Overlay run conditions
// ---------------------------------------------------------------------------
//
// The overlay systems below draw only through `Gizmos`, which is immediate
// mode: skipping a frame draws nothing and leaves no state behind, so gating
// them on their own toggle is safe and saves the whole iteration when an
// operator has the overlay switched off.
//
// `maintain_reconstruction_mesh_system` is deliberately *not* gated this way.
// It owns real state — it hides the reconstruction mesh and clears the cloud's
// dirty flag — so it has to keep running when its overlay is off, or the mesh
// would stay on screen and the dirty flag would never clear.

fn scan_grid_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_scan_grid
}

fn poi_markers_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_poi_markers || overlay.show_loc_ellipses
}

fn loc_ellipses_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_loc_ellipses
}

fn coord_frame_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_coord_frame
}

fn egress_paths_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_egress_paths
}

// Eight system parameters: the scene resources it reads, plus the two asset
// collections the sky dome is built from.
#[allow(clippy::too_many_arguments)]
fn setup_world(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    working_scene_root: Res<WorkingSceneRoot>,
    scene_package: Res<ScenePackage>,
    layer_visibility: Res<LayerVisibilityState>,
    mut layer_entities: ResMut<LayerEntityIndex>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut sky_materials: ResMut<Assets<SkyMaterial>>,
) {
    let terrain_summary = scene_package.environment.terrain_summary.as_ref();
    let orbit = OrbitCamera::from_bounds(
        &scene_package.environment.bounds_xy_m,
        terrain_summary.and_then(|summary| summary.min_height_m),
        terrain_summary.and_then(|summary| summary.max_height_m),
    );
    let eye = orbit.eye_position();
    let focus = orbit.focus;
    let orbit_radius = orbit.radius;
    // Home for R, and the rest of the main camera's behaviour (P2.4b).
    commands.insert_resource(crate::camera_control::CameraControl::new(
        crate::camera_control::OrbitPose::of(&orbit),
    ));

    // 0.19 split ambient light in two: `AmbientLight` is now a per-camera
    // component override, and `GlobalAmbientLight` is the scene-wide resource
    // this previously was.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.95, 0.96, 0.98),
        brightness: 350.0,
        ..default()
    });

    let bounds = &scene_package.environment.bounds_xy_m;
    let span = (bounds.x_max_m - bounds.x_min_m)
        .max(bounds.y_max_m - bounds.y_min_m)
        .max(200.0);

    // Cascades must be sized to the scene. The builder's defaults are tuned for
    // a character-scale world (Unity-like: first bound 10 m, maximum 150 m), so
    // on a multi-kilometre site every shadow past ~150 m of the camera simply
    // vanished. Spanning the scene keeps distant terrain shadowed; the first
    // bound stays short so near geometry still gets usable texel density.
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 1.0,
            maximum_distance: span,
            first_cascade_far_bound: (span / 20.0).max(50.0),
            overlap_proportion: 0.2,
        }
        .build(),
        Transform::from_xyz(2000.0, 1200.0, 2600.0).looking_at(Vec3::ZERO, Vec3::Z),
        // The sun lights the reconstruction layer too: a self-built map drawn
        // only in the split view's right pane is still terrain in daylight.
        RenderLayers::from_layers(&[0, RECON_LAYER]),
    ));

    // The main camera must set its projection explicitly. Without it Bevy applies
    // `PerspectiveProjection::default()` (far = 1000 m), which is shorter than the
    // distance this camera is framed at on every preset larger than `small`, so the
    // terrain is clipped away entirely. See `OrbitCamera::far_plane`.
    let main_far = orbit.far_plane(span);
    commands.spawn((
        MainCamera,
        orbit,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 45.0_f32.to_radians(),
            near: 1.0,
            far: main_far,
            ..default()
        }),
        Transform::from_translation(eye).looking_at(focus, Vec3::Z),
        // Filmic tonemapping, mapping the HDR range down for display. AcesFitted
        // is analytic; the LUT-based tonemappers (AgX, TonyMcMapface,
        // BlenderFilmic) would need Bevy's `tonemapping_luts` feature, and
        // without it they silently fall back to a placeholder LUT. Keeping to an
        // analytic curve holds the trimmed feature set from the dependency pass.
        Tonemapping::AcesFitted,
        // An HDR intermediate target, so anything emissive reads as bright
        // rather than merely pale. Bloom, which used to put it there, is now
        // the high quality tier's alone (§15: off by default).
        Hdr,
        SCENE_EXPOSURE,
        // Depth cueing (`DistanceFog`) and bloom are the quality tier's: see
        // `crate::quality::apply_quality_tier_system`, which adds them to this
        // camera, and `sync_fog_to_camera_system`, which sizes the fog.
        // Sees the world (layer 0) plus the reconstruction (RECON_LAYER) so the
        // scan-map overlay still shows in RealWorld/ScanMap and the Split left pane.
        RenderLayers::from_layers(&[0, RECON_LAYER]),
    ));

    // Angled 3-D perspective camera for the Split-mode reconstruction viewport.
    // Renders only RECON_LAYER, so the right pane is an empty area that fills with
    // a 3-D reconstruction of the believed terrain as the mission scans it.
    // Inactive by default; sync_split_viewports_system enables it in ViewMode::Split.
    //
    // Independent orbit controller for the reconstruction pane, framed at a lower
    // pitch than the main camera so the believed terrain reads as 3-D relief. It is
    // driven only when the cursor is over the right (Split) viewport, so the right
    // pane can be rotated, panned, and zoomed without moving the main view.
    let mut recon_orbit = OrbitCamera::from_bounds(
        &scene_package.environment.bounds_xy_m,
        terrain_summary.and_then(|summary| summary.min_height_m),
        terrain_summary.and_then(|summary| summary.max_height_m),
    );
    recon_orbit.pitch = 0.6;
    let recon_eye = recon_orbit.eye_position();
    let recon_focus = recon_orbit.focus;
    let recon_far = recon_orbit.far_plane(span);
    commands.spawn((
        ReconstructionCamera,
        recon_orbit,
        Camera3d::default(),
        Camera {
            order: 1,
            is_active: false,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.04, 0.05, 0.07)),
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: 45.0_f32.to_radians(),
            near: 1.0,
            far: recon_far,
            ..default()
        }),
        Transform::from_translation(recon_eye).looking_at(recon_focus, Vec3::Z),
        // Exposed and tonemapped as the main camera is: the believed terrain
        // in this pane is the same ground under the same sun.
        Tonemapping::AcesFitted,
        SCENE_EXPOSURE,
        RenderLayers::layer(RECON_LAYER),
    ));

    spawn_sky(
        &mut commands,
        &mut meshes,
        &mut sky_materials,
        asset_server.as_ref(),
        orbit_radius + span,
    );

    spawn_base_layers(
        &mut commands,
        asset_server.as_ref(),
        &working_scene_root,
        &scene_package,
        &layer_visibility,
        &mut layer_entities,
    );
}

// Keep synthetic scene loads under a stable asset root so rebuilt scene packages can
// swap subdirectories without changing Bevy's global AssetPlugin base path.
fn initialize_scene_package(scene_path: &Path) -> Result<(ScenePackage, WorkingSceneRoot)> {
    let original_scene_package = ScenePackage::load(scene_path)?;
    if original_scene_package.is_synthetic_source() {
        let asset_root = create_working_asset_root()?;
        let working_scene_root = asset_root.join("scene-0000");
        copy_directory_recursive(&original_scene_package.root, &working_scene_root)?;
        let working_scene_package = ScenePackage::load(&working_scene_root)?;
        return Ok((
            working_scene_package,
            WorkingSceneRoot {
                original_scene_root: original_scene_package.root.clone(),
                asset_root,
                current_scene_root: working_scene_root,
                revision: 0,
                is_synthetic: true,
            },
        ));
    }

    let root = original_scene_package.root.clone();
    Ok((
        original_scene_package,
        WorkingSceneRoot {
            original_scene_root: root.clone(),
            asset_root: root.clone(),
            current_scene_root: root,
            revision: 0,
            is_synthetic: false,
        },
    ))
}

fn create_working_asset_root() -> Result<PathBuf> {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("argusnet-viewer-{}-{suffix}", std::process::id()));
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create working scene root {}", root.display()))?;
    // Canonicalize so that strip_prefix in asset_path_for matches the
    // canonicalized paths that ScenePackage::load produces (macOS resolves
    // /var → /private/var during canonicalize).
    root.canonicalize()
        .with_context(|| format!("failed to canonicalize working root {}", root.display()))
}

fn copy_directory_recursive(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_directory_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} -> {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn spawn_base_layers(
    commands: &mut Commands,
    asset_server: &AssetServer,
    working_scene_root: &WorkingSceneRoot,
    scene_package: &ScenePackage,
    layer_visibility: &LayerVisibilityState,
    layer_entities: &mut LayerEntityIndex,
) {
    layer_entities.entities_by_layer.clear();
    for layer in &scene_package.manifest.layers {
        let Some(asset_path) =
            working_scene_root.asset_path_for(&scene_package.root, &layer.asset_path)
        else {
            warn!(
                "Refusing to load layer {:?}: asset_path {:?} escapes the scene root",
                layer.id, layer.asset_path
            );
            continue;
        };
        let handle: Handle<WorldAsset> = asset_server.load(format!("{asset_path}#Scene0"));
        let visibility = if *layer_visibility.base_layers.get(&layer.id).unwrap_or(&true) {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let entity = commands
            .spawn((
                Name::new(layer.id.clone()),
                WorldAssetRoot(handle),
                visibility,
            ))
            .id();
        // Attaches by the style's semantic kind rather than by layer id, so a
        // scene that names its ground something else still gets the material.
        if is_terrain_layer(scene_package, &layer.style_id) {
            commands.entity(entity).insert(TerrainLayerRoot);
        }
        if crate::building_material::is_building_layer(scene_package, &layer.style_id) {
            commands
                .entity(entity)
                .insert(crate::building_material::BuildingLayerRoot);
        }
        layer_entities
            .entities_by_layer
            .insert(layer.id.clone(), entity);
    }
}

fn merged_layer_visibility(
    scene_package: &ScenePackage,
    previous: Option<&LayerVisibilityState>,
) -> LayerVisibilityState {
    let mut layer_visibility = LayerVisibilityState::default();
    for layer in &scene_package.manifest.layers {
        let visible = previous
            .and_then(|previous| previous.base_layers.get(&layer.id).copied())
            .unwrap_or_else(|| {
                scene_package
                    .style_for(&layer.style_id)
                    .map(|style| style.default_visibility)
                    .unwrap_or(true)
            });
        layer_visibility
            .base_layers
            .insert(layer.id.clone(), visible);
    }
    layer_visibility
}

/// The camera the operator shell is drawn by (windowed runs only).
#[derive(Component)]
pub struct UiCamera;

/// The render layer nothing is on: the UI camera draws egui alone.
pub const UI_LAYER: usize = 31;

/// Give the egui shell a camera of its own: the whole window, drawn after
/// every world camera, rendering no world layer. On the main camera the
/// shell took that camera's viewport for the window, so Split, which halves
/// the main camera's viewport, squeezed the whole shell into the left half.
fn spawn_ui_camera(mut commands: Commands) {
    commands.spawn((
        UiCamera,
        Camera2d,
        Camera {
            order: 100,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        Msaa::Off,
        RenderLayers::layer(UI_LAYER),
        bevy_egui::PrimaryEguiContext,
    ));
}

fn advance_playback_system(time: Res<Time>, mut replay_state: ResMut<ReplayState>) {
    replay_state.advance(time.delta_secs());
}

fn keyboard_playback_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut replay_state: ResMut<ReplayState>,
    gate: Res<crate::bindings::KeyboardGate>,
    bindings: Res<crate::bindings::Bindings>,
    mut view_mode: ResMut<ViewMode>,
) {
    use crate::bindings::Action;
    // Every key through the remappable bindings, none while a text field has
    // the keyboard (P2.4b). The camera's keys, reset included, are
    // `camera_control`'s.
    let pressed = |action| bindings.just_pressed(&gate, &keys, action);

    // View mode works with or without a replay; before P2.4b it sat after
    // the early return below and did nothing in a scene without one.
    if pressed(Action::CycleView) {
        *view_mode = match *view_mode {
            ViewMode::RealWorld => ViewMode::ScanMap,
            ViewMode::ScanMap => ViewMode::Split,
            ViewMode::Split => ViewMode::RealWorld,
        };
    }

    // The same commands the timeline's buttons issue (C4).
    use crate::ui::PlaybackCommand;
    for (action, command) in [
        (Action::PlayPause, PlaybackCommand::PlayPause),
        (Action::StepBack, PlaybackCommand::StepBack),
        (Action::StepForward, PlaybackCommand::StepForward),
        (Action::First, PlaybackCommand::First),
        (Action::Last, PlaybackCommand::Last),
        (Action::PreviousEvent, PlaybackCommand::PreviousEvent),
        (Action::NextEvent, PlaybackCommand::NextEvent),
    ] {
        if pressed(action) {
            crate::ui::apply_playback(&mut replay_state, command);
        }
    }
}

/// Adjusts camera viewports when ViewMode::Split is active.
/// Left half → main perspective camera; right half → top-down reconstruction camera.
/// The main and reconstruction cameras' viewports in Split: the left and
/// right halves of the shell's canvas (`[x0, y0, x1, y1]`, logical
/// points, at `scale` physical pixels per point), or of the whole `w` by
/// `h` target when there is no shell (headless). Clamped to the target.
pub fn split_viewports(
    canvas: Option<[f32; 4]>,
    scale: f32,
    w: u32,
    h: u32,
) -> (Viewport, Viewport) {
    let [x0, y0, x1, y1] = canvas
        .map(|c| c.map(|v| (v * scale).round().max(0.0) as u32))
        .unwrap_or([0, 0, w, h]);
    let (x0, x1) = (x0.min(w), x1.clamp(x0.min(w), w));
    let (y0, y1) = (y0.min(h), y1.clamp(y0.min(h), h));
    let half = ((x1 - x0) / 2).max(1);
    let height = (y1 - y0).max(1);
    (
        Viewport {
            physical_position: UVec2::new(x0, y0),
            physical_size: UVec2::new(half, height),
            ..default()
        },
        Viewport {
            physical_position: UVec2::new(x0 + half, y0),
            physical_size: UVec2::new(half, height),
            ..default()
        },
    )
}

fn sync_split_viewports_system(
    view_mode: Res<ViewMode>,
    ui_state: Res<crate::state::ViewerUiState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    headless_target: Option<Res<HeadlessTarget>>,
    mut main_cam: Query<&mut Camera, (With<MainCamera>, Without<ReconstructionCamera>)>,
    mut recon_cam: Query<&mut Camera, With<ReconstructionCamera>>,
) {
    // Headless has no window (and no shell), so the offscreen image supplies
    // the extent the two cameras split between them; a window splits the
    // canvas its shell leaves.
    let (w, h, scale, canvas) = match headless_target.as_ref() {
        Some(target) => (target.width, target.height, 1.0, None),
        None => {
            let Ok(window) = windows.single() else {
                return;
            };
            (
                window.physical_width(),
                window.physical_height(),
                window.scale_factor(),
                ui_state.canvas_rect,
            )
        }
    };
    let Ok(mut main) = main_cam.single_mut() else {
        return;
    };
    let Ok(mut recon) = recon_cam.single_mut() else {
        return;
    };

    if *view_mode == ViewMode::Split {
        let (left, right) = split_viewports(canvas, scale, w, h);
        main.viewport = Some(left);
        main.is_active = true;
        recon.viewport = Some(right);
        recon.is_active = true;
    } else {
        main.viewport = None;
        main.is_active = true;
        recon.viewport = None;
        recon.is_active = false;
    }
}

/// The snapshot-apply stage: bring the presentation up to date with the
/// replay cursor, and re-resolve the selection against it.
///
/// Membership and poses are updated in place by the adapter, and only when
/// their inputs changed; a paused frame does nothing here.
pub fn sync_current_markers_system(
    replay_state: Res<ReplayState>,
    mut markers: ResMut<CurrentRuntimeMarkers>,
    mut frame_metrics: ResMut<CurrentFrameMetrics>,
    mut selection: ResMut<SelectionState>,
) {
    // Checked through `bypass_change_detection` so an unchanged frame does not
    // mark the resource changed for every reader downstream.
    let change = markers.bypass_change_detection().apply(&replay_state);
    // A new selection is resolved at once, paused or not: before P2.3 a
    // click on a paused frame showed empty details until the frame moved.
    let fresh_selection = selection.is_changed();
    if change == PresentationChange::default() && !fresh_selection {
        return;
    }
    if change != PresentationChange::default() {
        markers.set_changed();
    }

    // Every frame has its own metrics, whether or not its entities changed.
    if change.frame {
        frame_metrics.metrics = replay_state
            .current_frame()
            .map(|frame| frame.metrics.clone());
    }

    // Re-resolve the selection by key. Each kind builds its own detail
    // variant, so there is no other kind's data to null.
    let Some(selected) = selection.selected.as_ref() else {
        return;
    };
    let (key, epoch) = (selected.key, selected.epoch);
    let marker = match markers.get(key) {
        Some(marker) if epoch == markers.epoch() => marker,
        // Gone from this frame, or from another run altogether.
        _ => {
            selection.clear();
            return;
        }
    };
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let id = &*marker.label;

    let details = match marker.kind {
        MarkerKind::Track => frame.tracks.iter().find(|t| t.track_id == id).map(|track| {
            SelectionDetails::Track(TrackDetails {
                measurement_std_m: Some(track.measurement_std_m),
                update_count: Some(track.update_count),
                stale_steps: Some(track.stale_steps),
                track_error_m: frame_metrics
                    .metrics
                    .as_ref()
                    .and_then(|m| m.track_errors_m.get(id).copied()),
                covariance_diag: track
                    .covariance
                    .as_ref()
                    .map(|cov| crate::ui::covariance_diagonal(cov)),
                mode_probability_cv: track.mode_probability_cv,
                contributing_nodes: track.contributing_node_ids.clone(),
            })
        }),
        MarkerKind::Node => frame.nodes.iter().find(|n| n.node_id == id).map(|node| {
            SelectionDetails::Node(NodeDetails {
                health: Some(node.health),
                sensor_type: Some(node.sensor_type.clone()),
                fov_half_angle_deg: node.fov_half_angle_deg,
                max_range_m: node.max_range_m,
                is_mobile: Some(node.is_mobile),
            })
        }),
        MarkerKind::Truth => frame
            .truths
            .iter()
            .find(|t| t.target_id == id)
            .map(|truth| {
                let truth_pos = Vec3::from_array(truth.position);
                let nearest = frame
                    .tracks
                    .iter()
                    .map(|t| truth_pos.distance(Vec3::from_array(t.position)))
                    .fold(f32::INFINITY, f32::min);
                SelectionDetails::Truth(TruthDetails {
                    nearest_track_dist_m: nearest.is_finite().then_some(nearest),
                })
            }),
    };

    if let Some(details) = details {
        let (label, position, velocity) =
            (marker.label.to_string(), marker.position, marker.velocity);
        let resolved = Selection {
            key,
            epoch,
            label,
            position,
            velocity,
            details,
        };
        // Written only when it differs, so resolving does not itself read as
        // a fresh selection next frame.
        if selection.selected.as_ref() != Some(&resolved) {
            selection.selected = Some(resolved);
        }
    }
}

fn sync_base_layer_visibility_system(
    layer_entities: Res<LayerEntityIndex>,
    layer_visibility: Res<LayerVisibilityState>,
    mut query: Query<&mut Visibility>,
) {
    if !layer_visibility.is_changed() {
        return;
    }

    for (layer_id, visible) in &layer_visibility.base_layers {
        let Some(entity) = layer_entities.entities_by_layer.get(layer_id) else {
            continue;
        };
        let Ok(mut entity_visibility) = query.get_mut(*entity) else {
            continue;
        };
        *entity_visibility = if *visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn draw_runtime_overlays_system(
    mut gizmos: Gizmos,
    scene_package: Res<ScenePackage>,
    replay_state: Res<ReplayState>,
    markers: Res<CurrentRuntimeMarkers>,
    runtime_visibility: Res<RuntimeOverlayVisibility>,
    mission_zones: Res<LoadedMissionZones>,
    camera_query: Query<(&GlobalTransform, &Projection, &Camera), With<MainCamera>>,
) {
    // Glyphs are symbology, not scenery: they hold a constant size on screen and
    // face the viewer. See `crate::glyphs` for what this replaces.
    let glyphs = camera_query
        .single()
        .map(|(transform, projection, camera)| {
            GlyphProjection::from_camera(
                transform,
                projection,
                camera.logical_viewport_size().map(|size| size.y),
            )
        })
        .unwrap_or_default();

    for marker in &markers.markers {
        if !is_marker_visible(marker, &runtime_visibility) {
            continue;
        }

        let (draw_color, radius_scale) = marker_appearance(marker, &scene_package);
        if runtime_visibility.diagnostic_glyphs {
            draw_marker_scaled(&mut gizmos, &glyphs, marker, draw_color, radius_scale);
        } else {
            let pixels = match marker.kind {
                MarkerKind::Track => TRACK_GLYPH_PX,
                MarkerKind::Truth => TRUTH_GLYPH_PX,
                MarkerKind::Node => NODE_GLYPH_PX,
            };
            draw_velocity_leader(
                &mut gizmos,
                marker,
                draw_color,
                glyphs.radius(marker.position, pixels) * radius_scale,
            );
        }

        // Trails are persistent ribbons now: `crate::trail_ribbons`.
    }

    if let Some(frame) = replay_state.current_frame() {
        if runtime_visibility.observations {
            let obs_color = Color::srgba(0.2, 0.85, 0.3, 0.6);
            let ray_length = 200.0_f32;
            for obs in &frame.observations {
                let origin = Vec3::from_array(obs.origin);
                let dir = Vec3::from_array(obs.direction);
                if dir.length_squared() > 0.0 {
                    let endpoint = origin + dir.normalize() * ray_length;
                    gizmos.line(origin, endpoint, obs_color);
                }
            }
        }

        if runtime_visibility.rejection_markers {
            for rej in &frame.generation_rejections {
                let Some(pos) = generation_rejection_position(rej) else {
                    continue;
                };
                draw_rejection_marker(&mut gizmos, &glyphs, pos, rejection_marker_style(rej, true));
            }
            for rej in &frame.rejected_observations {
                let pos = tracker_rejection_position(frame, rej);
                draw_rejection_marker(
                    &mut gizmos,
                    &glyphs,
                    pos,
                    rejection_marker_style(rej, false),
                );
            }
        }

        // Item 6: Inspection event spatial rendering — draw zone circles colored by
        // coverage fraction and brightened when the zone has an active inspection event.
        if runtime_visibility.inspection_events && !mission_zones.zones.is_empty() {
            // Build a map of zone_id -> latest coverage_fraction for this frame.
            let mut zone_coverage: std::collections::HashMap<&str, f32> =
                std::collections::HashMap::new();
            let mut active_zone_ids: std::collections::HashSet<&str> =
                std::collections::HashSet::new();
            for ev in &frame.inspection_events {
                zone_coverage
                    .entry(ev.zone_id.as_str())
                    .and_modify(|f| *f = f.max(ev.zone_coverage_fraction))
                    .or_insert(ev.zone_coverage_fraction);
                active_zone_ids.insert(ev.zone_id.as_str());
            }

            let zone_ring_segs: u32 = 32;
            for zone in &mission_zones.zones {
                let coverage = zone_coverage
                    .get(zone.zone_id.as_str())
                    .copied()
                    .unwrap_or(0.0);
                let is_active = active_zone_ids.contains(zone.zone_id.as_str());
                // Coverage-based hue: red (0°) = 0% → green (120°) = 100%
                let alpha = if is_active { 1.0_f32 } else { 0.4_f32 };
                let hue = coverage.clamp(0.0, 1.0) * 120.0;
                let zone_color = Color::hsla(hue, 1.0, 0.5, alpha);

                let cx = zone.center[0];
                let cy = zone.center[1];
                let cz = zone.center[2];
                let r = zone.radius_m;
                for i in 0..zone_ring_segs {
                    let a0 = std::f32::consts::TAU * i as f32 / zone_ring_segs as f32;
                    let a1 = std::f32::consts::TAU * (i + 1) as f32 / zone_ring_segs as f32;
                    let p0 = Vec3::new(cx + a0.cos() * r, cy + a0.sin() * r, cz);
                    let p1 = Vec3::new(cx + a1.cos() * r, cy + a1.sin() * r, cz);
                    gizmos.line(p0, p1, zone_color);
                }
            }
        }
    }

    // Covariance ellipsoids — draw 3 axis-aligned circles per track
    if runtime_visibility.show_covariance_ellipsoids {
        if let Some(frame) = replay_state.current_frame() {
            let ell_color = Color::srgba(0.9, 0.65, 0.1, 0.55);
            let k = 2.45_f64; // 95% confidence factor
            let segments = 20;
            for track in &frame.tracks {
                if let Some(cov) = &track.covariance {
                    let diag = crate::ui::covariance_diagonal(cov);
                    let rx = ((diag[0].max(0.0)).sqrt() * k) as f32;
                    let ry = ((diag[1].max(0.0)).sqrt() * k) as f32;
                    let rz = ((diag[2].max(0.0)).sqrt() * k) as f32;
                    let center = Vec3::from_array(track.position);
                    // XY-plane ring
                    for i in 0..segments {
                        let a0 = std::f32::consts::TAU * i as f32 / segments as f32;
                        let a1 = std::f32::consts::TAU * (i + 1) as f32 / segments as f32;
                        let p0 = center + Vec3::new(a0.cos() * rx, a0.sin() * ry, 0.0);
                        let p1 = center + Vec3::new(a1.cos() * rx, a1.sin() * ry, 0.0);
                        gizmos.line(p0, p1, ell_color);
                    }
                    // XZ-plane ring
                    for i in 0..segments {
                        let a0 = std::f32::consts::TAU * i as f32 / segments as f32;
                        let a1 = std::f32::consts::TAU * (i + 1) as f32 / segments as f32;
                        let p0 = center + Vec3::new(a0.cos() * rx, 0.0, a0.sin() * rz);
                        let p1 = center + Vec3::new(a1.cos() * rx, 0.0, a1.sin() * rz);
                        gizmos.line(p0, p1, ell_color);
                    }
                    // YZ-plane ring
                    for i in 0..segments {
                        let a0 = std::f32::consts::TAU * i as f32 / segments as f32;
                        let a1 = std::f32::consts::TAU * (i + 1) as f32 / segments as f32;
                        let p0 = center + Vec3::new(0.0, a0.cos() * ry, a0.sin() * rz);
                        let p1 = center + Vec3::new(0.0, a1.cos() * ry, a1.sin() * rz);
                        gizmos.line(p0, p1, ell_color);
                    }
                }
            }
        }
    }
}

fn find_node_position_opt(frame: &crate::replay::ReplayFrame, node_id: &str) -> Option<Vec3> {
    frame
        .nodes
        .iter()
        .find(|n| n.node_id == node_id)
        .map(|n| Vec3::from_array(n.position))
}

#[derive(Clone, Copy)]
enum RejectionGlyph {
    Cross,
    Diamond,
    Square,
    Triangle,
    Ring,
}

#[derive(Clone, Copy)]
struct RejectionMarkerStyle {
    color: Color,
    glyph: RejectionGlyph,
    /// On-screen radius in logical pixels — see `crate::glyphs`.
    size_px: f32,
}

fn generation_rejection_position(rej: &RejectedObservation) -> Option<Vec3> {
    rej.closest_point
        .or(rej.attempted_point)
        .or(rej.origin)
        .map(Vec3::from_array)
}

fn tracker_rejection_position(
    frame: &crate::replay::ReplayFrame,
    rej: &RejectedObservation,
) -> Vec3 {
    rej.origin
        .map(Vec3::from_array)
        .or_else(|| find_node_position_opt(frame, &rej.node_id))
        .unwrap_or(Vec3::ZERO)
}

fn rejection_marker_style(rej: &RejectedObservation, is_generation: bool) -> RejectionMarkerStyle {
    if !is_generation {
        return RejectionMarkerStyle {
            color: Color::srgba(0.94, 0.2, 0.25, 0.9),
            glyph: RejectionGlyph::Cross,
            size_px: 7.0,
        };
    }
    match rej.reason.as_str() {
        "terrain_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.76, 0.42, 0.16, 0.92),
            glyph: RejectionGlyph::Diamond,
            size_px: 7.5,
        },
        "building_occlusion" | "wall_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.92, 0.62, 0.18, 0.92),
            glyph: RejectionGlyph::Square,
            size_px: 7.0,
        },
        "vegetation_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.2, 0.72, 0.32, 0.92),
            glyph: RejectionGlyph::Cross,
            size_px: 7.0,
        },
        "outside_fov" | "low_elevation" => RejectionMarkerStyle {
            color: Color::srgba(0.95, 0.78, 0.2, 0.92),
            glyph: RejectionGlyph::Triangle,
            size_px: 8.0,
        },
        "out_of_range" | "out_of_coverage" => RejectionMarkerStyle {
            color: Color::srgba(0.24, 0.66, 0.95, 0.92),
            glyph: RejectionGlyph::Ring,
            size_px: 7.5,
        },
        "dropout" => RejectionMarkerStyle {
            color: Color::srgba(0.82, 0.84, 0.86, 0.88),
            glyph: RejectionGlyph::Ring,
            size_px: 6.0,
        },
        _ => RejectionMarkerStyle {
            color: Color::srgba(0.96, 0.34, 0.34, 0.9),
            glyph: RejectionGlyph::Cross,
            size_px: 7.0,
        },
    }
}

/// Draws one observation-rejection glyph, sized in pixels and camera-facing.
///
/// These are flat symbols with no altitude component, so unlike the marker
/// glyphs they have no world-Z spine — the whole shape lives in the screen plane.
fn draw_rejection_marker(
    gizmos: &mut Gizmos,
    glyphs: &GlyphProjection,
    position: Vec3,
    style: RejectionMarkerStyle,
) {
    let r = glyphs.radius(position, style.size_px);
    let right = glyphs.right * r;
    let up = glyphs.up * r;

    match style.glyph {
        RejectionGlyph::Cross => {
            gizmos.line(position - right - up, position + right + up, style.color);
            gizmos.line(position + right - up, position - right + up, style.color);
        }
        RejectionGlyph::Diamond => {
            gizmos.line(position + up, position + right, style.color);
            gizmos.line(position + right, position - up, style.color);
            gizmos.line(position - up, position - right, style.color);
            gizmos.line(position - right, position + up, style.color);
        }
        RejectionGlyph::Square => {
            let a = position - right - up;
            let b = position + right - up;
            let c = position + right + up;
            let d = position - right + up;
            gizmos.line(a, b, style.color);
            gizmos.line(b, c, style.color);
            gizmos.line(c, d, style.color);
            gizmos.line(d, a, style.color);
        }
        RejectionGlyph::Triangle => {
            let top = position + up;
            let right_foot = position + right * 0.86 - up * 0.6;
            let left_foot = position - right * 0.86 - up * 0.6;
            gizmos.line(top, right_foot, style.color);
            gizmos.line(right_foot, left_foot, style.color);
            gizmos.line(left_foot, top, style.color);
        }
        RejectionGlyph::Ring => {
            let segments = 14;
            for index in 0..segments {
                let a0 = std::f32::consts::TAU * index as f32 / segments as f32;
                let a1 = std::f32::consts::TAU * (index + 1) as f32 / segments as f32;
                let p0 = position + right * a0.cos() + up * a0.sin();
                let p1 = position + right * a1.cos() + up * a1.sin();
                gizmos.line(p0, p1, style.color);
            }
        }
    }
}

/// Draws one marker glyph, sized in pixels and oriented to the camera.
///
/// `scale` is the track-confidence multiplier, applied on top of the target
/// pixel size so an uncertain track still reads as a larger symbol.
fn draw_marker_scaled(
    gizmos: &mut Gizmos,
    glyphs: &GlyphProjection,
    marker: &RuntimeMarker,
    color: Color,
    scale: f32,
) {
    let pixels = match marker.kind {
        MarkerKind::Track => TRACK_GLYPH_PX,
        MarkerKind::Truth => TRUTH_GLYPH_PX,
        MarkerKind::Node => NODE_GLYPH_PX,
    };
    let pos = marker.position;
    let r = glyphs.radius(pos, pixels) * scale;

    // The glyph plane faces the camera. The vertical spines below stay on world
    // Z on purpose: they mean altitude, so they belong to the world rather than
    // to the screen, and collapsing to a point from directly overhead is the
    // correct reading of a vertical line.
    let right = glyphs.right * r;
    let up = glyphs.up * r;

    match marker.kind {
        MarkerKind::Track => {
            // Cross (+) glyph
            gizmos.line(pos - right, pos + right, color);
            gizmos.line(pos - up, pos + up, color);
            gizmos.line(pos - Vec3::Z * (r * 0.6), pos + Vec3::Z * (r * 0.6), color);
        }
        MarkerKind::Truth => {
            // Circle ring (12 segments) — ground-truth target
            let segs = 12u32;
            for i in 0..segs {
                let a0 = std::f32::consts::TAU * i as f32 / segs as f32;
                let a1 = std::f32::consts::TAU * (i + 1) as f32 / segs as f32;
                let p0 = pos + right * a0.cos() + up * a0.sin();
                let p1 = pos + right * a1.cos() + up * a1.sin();
                gizmos.line(p0, p1, color);
            }
            // Vertical tick
            gizmos.line(pos - Vec3::Z * (r * 0.5), pos + Vec3::Z * (r * 0.5), color);
        }
        MarkerKind::Node => {
            if marker.is_mobile == Some(true) {
                // Diamond glyph for mobile drones
                gizmos.line(pos + up, pos + right, color);
                gizmos.line(pos + right, pos - up, color);
                gizmos.line(pos - up, pos - right, color);
                gizmos.line(pos - right, pos + up, color);
                // Vertical spine for altitude context
                gizmos.line(pos - Vec3::Z * (r * 0.5), pos + Vec3::Z * r, color);
            } else {
                // Square bracket glyph for static ground sensors
                let a = pos - right - up;
                let b = pos + right - up;
                let c = pos + right + up;
                let d = pos - right + up;
                gizmos.line(a, b, color);
                gizmos.line(b, c, color);
                gizmos.line(c, d, color);
                gizmos.line(d, a, color);
            }
        }
    }

    draw_velocity_leader(gizmos, marker, color, r);
}

fn draw_velocity_leader(gizmos: &mut Gizmos, marker: &RuntimeMarker, color: Color, r: f32) {
    let pos = marker.position;
    // Velocity leader. The direction is the real world-space heading, but the
    // length is expressed in glyph radii rather than metres: at default framing
    // the old 15 m maximum was under three pixels, so speed was unreadable
    // exactly where the overview needs it.
    if marker.velocity.length_squared() > 0.0 {
        let speed_fraction =
            marker.velocity.clamp_length_max(MAX_LEADER_SPEED_MPS) / MAX_LEADER_SPEED_MPS;
        gizmos.line(
            pos,
            pos + speed_fraction * r * LEADER_RADII_AT_MAX_SPEED,
            color,
        );
    }
}

/// Speed at which a velocity leader reaches its full length. Above this the
/// leader stops growing, so one fast outlier cannot swamp the display.
const MAX_LEADER_SPEED_MPS: f32 = 60.0;

/// Length of a full-speed velocity leader, in glyph radii.
const LEADER_RADII_AT_MAX_SPEED: f32 = 4.0;

pub(crate) fn marker_appearance(
    marker: &RuntimeMarker,
    scene_package: &ScenePackage,
) -> (Color, f32) {
    let base_color = marker_color(marker.kind, scene_package);
    let (draw_color, radius_scale) = if marker.kind == crate::replay::MarkerKind::Track {
        // Carried on the marker by the adapter: no search of the frame.
        let std_m = marker.measurement_std_m.unwrap_or(0.0);
        let conf_color = if std_m < 5.0 {
            Color::srgba(0.2, 0.9, 0.2, 1.0) // green
        } else if std_m < 20.0 {
            Color::srgba(0.95, 0.85, 0.1, 1.0) // yellow
        } else if std_m < 50.0 {
            Color::srgba(0.95, 0.5, 0.1, 1.0) // orange
        } else {
            Color::srgba(0.95, 0.15, 0.15, 1.0) // red
        };
        let scale = 1.0 + std_m.min(50.0) / 50.0;
        (conf_color, scale)
    } else {
        (base_color, 1.0)
    };
    (draw_color, radius_scale)
}

pub(crate) fn is_marker_visible(
    marker: &RuntimeMarker,
    runtime_visibility: &RuntimeOverlayVisibility,
) -> bool {
    match marker.kind {
        MarkerKind::Track => runtime_visibility.tracks,
        MarkerKind::Truth => runtime_visibility.truths,
        MarkerKind::Node => runtime_visibility.nodes,
    }
}

fn marker_color(kind: MarkerKind, scene_package: &ScenePackage) -> Color {
    let style_id = match kind {
        MarkerKind::Track => "tracks",
        MarkerKind::Truth => "truths",
        MarkerKind::Node => "nodes",
    };
    let rgba = scene_package
        .style_for(style_id)
        .map(|style| style.color_rgba)
        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
    Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3])
}

fn draw_scan_grid_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    overlay: Res<MissionOverlaySettings>,
    scene_package: Res<ScenePackage>,
    mission_zones: Res<LoadedMissionZones>,
) {
    if !overlay.show_scan_grid {
        return;
    }
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    // Coverage fraction drives a grid-edge outline hinting at map bounds.
    if let Some(ms) = &frame.scan_mission_state {
        let frac = ms.scan_coverage_fraction.clamp(0.0, 1.0);
        // Outer bound rect — fades from red (0%) to green (100%).
        // Z-up: ground plane is XY; Quat::IDENTITY gives a flat horizontal rect.
        let hue = 120.0 * frac;
        let border_color = Color::hsla(hue, 0.9, 0.5, 0.5);
        let b = &scene_package.environment.bounds_xy_m;
        let cx = (b.x_min_m + b.x_max_m) * 0.5;
        let cy = (b.y_min_m + b.y_max_m) * 0.5;
        let width = b.x_max_m - b.x_min_m;
        let height = b.y_max_m - b.y_min_m;
        let ground_z = scene_package
            .environment
            .terrain_summary
            .as_ref()
            .and_then(|s| s.min_height_m)
            .unwrap_or(0.0);
        let iso = bevy::math::Isometry3d::new(Vec3::new(cx, cy, ground_z + 2.0), Quat::IDENTITY);
        gizmos.rect(iso, Vec2::new(width, height), border_color);

        // During scanning, draw sensor-footprint circles under each mobile drone.
        // Anchor each circle to the local terrain height under the drone so it
        // doesn't sink below the surface on non-flat terrain. Falls back to the
        // global ground_z only when terrain isn't sampleable at the drone's XY.
        if ms.phase == "scanning" {
            let footprint_r = 75.0_f32;
            let footprint_color = Color::hsla(52.0, 1.0, 0.88, 0.65);
            const LIFT_M: f32 = 1.5;
            // The terrain mesh is parsed once at load and kept in `LoadedMissionZones`
            // (see `setup_world` and `reload_rebuilt_scene`). Re-deriving it from the
            // replay JSON here deserialized the whole heightmap every frame.
            let terrain_mesh = mission_zones.terrain_mesh.as_ref();
            for node in &frame.nodes {
                if node.is_mobile {
                    let local_ground = footprint_ground_z_m(
                        terrain_mesh,
                        ground_z,
                        [node.position[0], node.position[1]],
                    );
                    let iso_c = bevy::math::Isometry3d::new(
                        Vec3::new(node.position[0], node.position[1], local_ground + LIFT_M),
                        Quat::IDENTITY,
                    );
                    gizmos.circle(iso_c, footprint_r, footprint_color);
                }
            }
        }
    }
}

fn footprint_ground_z_m(
    terrain_mesh: Option<&TerrainViewerMesh>,
    fallback_ground_z: f32,
    xy_m: [f32; 2],
) -> f32 {
    terrain_mesh
        .and_then(|mesh| mesh.height_at(xy_m[0], xy_m[1]))
        .unwrap_or(fallback_ground_z)
}

fn draw_poi_markers_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    overlay: Res<MissionOverlaySettings>,
) {
    if !overlay.show_poi_markers {
        return;
    }
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(mission) = &frame.scan_mission_state else {
        return;
    };

    for poi_status in &mission.poi_statuses {
        let color = match poi_status.status.as_str() {
            "complete" => Color::srgb(0.0, 1.0, 0.2),
            "active" => Color::srgb(1.0, 0.8, 0.0),
            _ => Color::srgb(0.8, 0.8, 0.8),
        };
        // Skip POIs with no known position (old replays without position field).
        let Some(pos) = poi_status.position else {
            continue;
        };
        let marker_pos = Vec3::new(pos[0], pos[1], pos[2] + 5.0);
        let iso_top = bevy::math::Isometry3d::new(marker_pos + Vec3::Z * 8.0, Quat::IDENTITY);
        let iso_bot = bevy::math::Isometry3d::new(marker_pos - Vec3::Z * 3.0, Quat::IDENTITY);
        gizmos.sphere(iso_top, 5.0, color);
        gizmos.sphere(iso_bot, 3.0, color);
        gizmos.line(
            marker_pos - Vec3::Z * 3.0,
            marker_pos + Vec3::Z * 8.0,
            color,
        );
    }

    if !overlay.show_loc_ellipses {
        return;
    }
    for est in &mission.localization_estimates {
        // Z-up world, matching every other overlay (x, y, z).
        let pos = Vec3::new(
            est.position_estimate[0],
            est.position_estimate[1],
            est.position_estimate[2],
        );
        let radius = est.position_std_m.max(5.0);
        let alpha = 0.5 * est.confidence;
        let color = Color::hsla(200.0, 0.8, 0.6, alpha);
        let isometry = bevy::math::Isometry3d::new(pos, Quat::IDENTITY);
        gizmos.circle(isometry, radius, color);
    }
}

/// Draws the fused team position from cooperative co-localization: a beacon at the
/// team estimate, its uncertainty ring, and spokes to each drone whose estimate was
/// fused in — making "the drones localize to each other's search data" visible.
fn draw_team_localization_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    overlay: Res<MissionOverlaySettings>,
) {
    if !overlay.show_loc_ellipses {
        return;
    }
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(mission) = &frame.scan_mission_state else {
        return;
    };
    let Some(team) = &mission.team_localization else {
        return;
    };

    let team_pos = Vec3::new(team.position[0], team.position[1], team.position[2]);
    // Cyan beacon, brightening as the fused estimate gains confidence.
    let beacon = Color::hsla(
        185.0,
        0.9,
        0.55,
        0.4 + 0.55 * team.confidence.clamp(0.0, 1.0),
    );

    // Spokes from the team estimate to each contributing drone's own estimate.
    for est in &mission.localization_estimates {
        if !team
            .contributing_node_ids
            .iter()
            .any(|id| id == &est.drone_id)
        {
            continue;
        }
        let drone_pos = Vec3::new(
            est.position_estimate[0],
            est.position_estimate[1],
            est.position_estimate[2],
        );
        gizmos.line(team_pos, drone_pos, Color::hsla(185.0, 0.8, 0.6, 0.35));
    }

    // Uncertainty ring (true 1-sigma, floored so a tight fusion is still visible).
    let ring = bevy::math::Isometry3d::new(team_pos, Quat::IDENTITY);
    gizmos.circle(ring, team.position_std_m.max(8.0), beacon);

    // Vertical beacon stalk + top sphere so the datum is findable from any angle.
    gizmos.line(team_pos, team_pos + Vec3::Z * 90.0, beacon);
    let top = bevy::math::Isometry3d::new(team_pos + Vec3::Z * 90.0, Quat::IDENTITY);
    gizmos.sphere(top, 10.0, beacon);
}

/// Accumulates newly-scanned cell points from the replay into the persistent
/// ReconstructionCloud.  Resets when the playhead rewinds.
fn update_reconstruction_system(
    replay_state: Res<ReplayState>,
    mut reconstruction: ResMut<ReconstructionCloud>,
) {
    let frame_idx = replay_state.frame_index;

    // Reset if the playhead was rewound.
    if frame_idx < reconstruction.last_frame_index {
        reconstruction.reset();
    }
    reconstruction.last_frame_index = frame_idx;

    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref mission) = frame.scan_mission_state else {
        return;
    };

    // Python emits a flat [x0,y0,h0, x1,y1,h1, ...] array; chunk into triples.
    // Viewer is Z-up: Python (x=east, y=north, h=terrain_height) maps directly.
    let prev_len = reconstruction.points.len();
    for triple in mission.newly_scanned_cells.as_chunks::<3>().0 {
        let z = triple[2];
        reconstruction.points.push([triple[0], triple[1], z]);
        if z > reconstruction.max_z {
            reconstruction.max_z = z;
        }
        if z < reconstruction.min_z {
            reconstruction.min_z = z;
        }
    }
    if reconstruction.points.len() > prev_len {
        reconstruction.dirty = true;
    }
}

/// Infers the coverage grid cell size (meters) from the reconstruction points as the
/// smallest positive gap between distinct x-coordinates. Falls back to 50 m (the default
/// coverage resolution) when there are too few points to establish spacing.
fn infer_cell_size(points: &[[f32; 3]]) -> f32 {
    let mut xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = f32::INFINITY;
    for w in xs.windows(2) {
        let d = w[1] - w[0];
        if d > 0.5 && d < best {
            best = d;
        }
    }
    if best.is_finite() {
        best
    } else {
        50.0
    }
}

/// Spawns the persistent GPU line-list mesh used for the reconstruction cloud.
/// Called once at startup; the mesh geometry is empty until `maintain_reconstruction_mesh_system`
/// fills it.
fn setup_reconstruction_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use bevy::render::mesh::PrimitiveTopology;

    // Initialise with a degenerate triangle so the GPU allocator never encounters
    // a zero-vertex buffer (Bevy's MeshAllocator panics on divide-by-zero).
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; 3]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 0.0, 1.0]; 3]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0_f32; 4]; 3]);

    // Bright-yellow scan-coverage overlay — maximally distinct from terrain greens/cyans.
    let mat = StandardMaterial {
        base_color: bevy::color::Color::srgba(0.90, 0.80, 0.15, 0.45),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        cull_mode: None,
        unlit: true,
        ..default()
    };
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(mat)),
        Visibility::Hidden,
        ReconstructionMeshMarker,
        // Reconstruction-only layer: the main camera sees it (layers 0+RECON) but
        // the reconstruction camera sees ONLY this, keeping its pane empty of terrain.
        RenderLayers::layer(RECON_LAYER),
    ));
}

/// Maintains the GPU triangle-mesh for the scan-coverage overlay.
/// Each scanned 50 m grid cell is drawn as a flat quad (2 triangles) sitting
/// 1 m above the terrain surface so it is never z-fought by the ground mesh.
/// Updates visibility every frame (cheap), rebuilds geometry only when `dirty`.
fn maintain_reconstruction_mesh_system(
    mut reconstruction: ResMut<ReconstructionCloud>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_q: Query<(&Mesh3d, &mut Visibility), With<ReconstructionMeshMarker>>,
    overlay: Res<MissionOverlaySettings>,
    view_mode: Res<ViewMode>,
) {
    let should_show = overlay.show_reconstruction && *view_mode != ViewMode::RealWorld;

    // Collect handles and update visibility in one pass.
    let mut handles: Vec<Handle<Mesh>> = Vec::new();
    for (mesh_3d, mut vis) in &mut mesh_q {
        *vis = if should_show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if reconstruction.dirty && should_show {
            handles.push(mesh_3d.0.clone());
        }
    }

    if !reconstruction.dirty {
        return;
    }
    // Always clear dirty to prevent unbounded re-queuing; skip GPU upload when hidden.
    reconstruction.dirty = false;
    if !should_show {
        return;
    }

    // Tile half-extent, inferred from the actual cell spacing so the reconstruction
    // renders as a near-continuous believed-terrain surface at any coverage
    // resolution. A 4% gap per edge keeps the grid crisp without visible seams.
    let cell = infer_cell_size(&reconstruction.points);
    let half = cell * 0.48;
    // Sit just above terrain to avoid z-fighting without floating awkwardly.
    const Z_LIFT: f32 = 1.0;
    // Up normal for all vertices in Z-up space.
    const NORMAL: [f32; 3] = [0.0, 0.0, 1.0];

    let z_range = (reconstruction.max_z - reconstruction.min_z).max(1.0);
    let min_z = reconstruction.min_z;

    let n = reconstruction.points.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    for pt in &reconstruction.points {
        // Height-mapped hue: blue (180°) at low elevation → yellow (52°) at high.
        // This gives the reconstruction a height-map appearance that matches terrain
        // colour intuitions while staying distinct from the green vegetation layer.
        let t = ((pt[2] - min_z) / z_range).clamp(0.0, 1.0);
        let hue = 180.0 - t * 128.0; // 180° (blue-cyan) → 52° (yellow)
        let scan_lin = LinearRgba::from(Color::hsla(hue, 0.90, 0.58, 0.80));
        let rgba = [scan_lin.red, scan_lin.green, scan_lin.blue, scan_lin.alpha];

        let base = positions.len() as u32;
        let z = pt[2] + Z_LIFT;
        // Quad corners (CCW from below → correct winding for Z-up top face).
        positions.push([pt[0] - half, pt[1] - half, z]);
        positions.push([pt[0] + half, pt[1] - half, z]);
        positions.push([pt[0] + half, pt[1] + half, z]);
        positions.push([pt[0] - half, pt[1] + half, z]);
        for _ in 0..4 {
            normals.push(NORMAL);
            colors.push(rgba);
        }
        // Two CCW triangles.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    // Degenerate fallback so MeshAllocator never sees an empty buffer.
    if positions.is_empty() {
        positions = vec![[0.0; 3]; 3];
        normals = vec![[0.0, 0.0, 1.0]; 3];
        colors = vec![[0.0; 4]; 3];
        indices = vec![0, 1, 2];
    }

    for handle in handles {
        if let Some(mut mesh) = meshes.get_mut(&handle) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors.clone());
            mesh.insert_indices(bevy::render::mesh::Indices::U32(indices.clone()));
        }
    }
}

/// Draws a 3D coordinate frame (XYZ axes) at the mean localization position
/// once localization or inspection has started.
fn draw_coord_frame_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    overlay: Res<MissionOverlaySettings>,
) {
    if !overlay.show_coord_frame {
        return;
    }
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref mission) = frame.scan_mission_state else {
        return;
    };

    // Only draw after localisation has begun.
    let phase = mission.phase.as_str();
    if phase == "scanning" || mission.localization_estimates.is_empty() {
        return;
    }

    // Mean localization position (average of all drone estimates).
    // Z-up: position_estimate = [east, north, altitude]; map directly.
    let n = mission.localization_estimates.len() as f32;
    let mean_pos = mission
        .localization_estimates
        .iter()
        .fold(Vec3::ZERO, |acc, est| {
            acc + Vec3::new(
                est.position_estimate[0],
                est.position_estimate[1],
                est.position_estimate[2],
            )
        })
        / n;

    // Confidence drives axis length (small when uncertain, 80 m when confident).
    let mean_conf = mission
        .localization_estimates
        .iter()
        .map(|e| e.confidence)
        .sum::<f32>()
        / n;
    let axis_len = 20.0 + mean_conf * 80.0;
    let alpha = 0.4 + mean_conf * 0.6;

    // Draw axes — X=red (east), Y=green (north), Z=blue (up in Z-up scene)
    let origin = mean_pos + Vec3::Z * 2.0;
    gizmos.arrow(
        origin,
        origin + Vec3::X * axis_len,
        Color::srgba(1.0, 0.1, 0.1, alpha),
    );
    gizmos.arrow(
        origin,
        origin + Vec3::Y * axis_len,
        Color::srgba(0.1, 1.0, 0.1, alpha),
    );
    gizmos.arrow(
        origin,
        origin + Vec3::Z * axis_len,
        Color::srgba(0.1, 0.3, 1.0, alpha),
    );

    // Label sphere at origin
    let iso = bevy::math::Isometry3d::new(origin, Quat::IDENTITY);
    gizmos.sphere(iso, 3.0, Color::srgba(1.0, 1.0, 0.0, alpha));

    // Heading ring around the origin — rotates around Z (up) in Z-up system.
    let mean_heading = mission
        .localization_estimates
        .iter()
        .map(|e| e.heading_rad)
        .sum::<f32>()
        / n;
    let ring_iso = bevy::math::Isometry3d::new(origin, Quat::from_rotation_z(mean_heading));
    gizmos.circle(
        ring_iso,
        axis_len * 0.25,
        Color::srgba(1.0, 1.0, 0.0, alpha * 0.5),
    );
}

/// Draws return-to-home gizmo lines during the egress phase.
/// For each drone with known egress progress, a line is drawn from the drone's
/// current position to its home station, colored red→green as it closes the gap.
fn draw_egress_paths_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    overlay: Res<MissionOverlaySettings>,
) {
    if !overlay.show_egress_paths {
        return;
    }
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref mission) = frame.scan_mission_state else {
        return;
    };
    if mission.phase != "egress" {
        return;
    }

    // Max distance for color normalization (fall back to 2000 m).
    let max_dist = mission
        .egress_progress
        .iter()
        .map(|e| e.distance_to_home_m)
        .fold(1.0_f32, f32::max)
        .max(200.0);

    for ep in &mission.egress_progress {
        // Find drone's current position from node list.
        let Some(node) = frame.nodes.iter().find(|n| n.node_id == ep.drone_id) else {
            continue;
        };
        let drone_pos = Vec3::new(node.position[0], node.position[1], node.position[2]);
        let home_pos = Vec3::new(
            ep.home_position[0],
            ep.home_position[1],
            ep.home_position[2],
        );

        // Color: red (far) → green (arrived).
        let frac = (1.0 - ep.distance_to_home_m / max_dist).clamp(0.0, 1.0);
        let line_color = Color::hsla(frac * 120.0, 0.95, 0.55, 0.85);
        let pad_color = Color::hsla(frac * 120.0, 0.80, 0.70, 0.60);

        gizmos.line(drone_pos, home_pos, line_color);
        let iso = bevy::math::Isometry3d::new(home_pos + Vec3::Z * 1.0, Quat::IDENTITY);
        gizmos.circle(iso, 8.0, pad_color);
    }
}

/// Starts the Rust simulation on its own thread when the user clicks "Run
/// Simulation" (D5).
// Re-simulation is a three-stage pipeline: simulate replay, rebuild a synthetic scene
// package, then hot-swap the viewer resources/entities to the rebuilt package.
fn spawn_simulation_system(
    mut commands: Commands,
    mut sim_runner: ResMut<SimulationRunner>,
    working_scene_root: Res<WorkingSceneRoot>,
    existing: Option<Res<SimulationProcess>>,
) {
    if sim_runner.phase != SimPhase::Simulating || existing.is_some() {
        return;
    }
    if !working_scene_root.supports_hot_swap() {
        sim_runner.fail(
            "In-app re-simulation is only supported for synthetic/replay-built scene packages.",
        );
        return;
    }

    let scratch_root = working_scene_root.asset_root.join(".viewer-runtime");
    if let Err(err) = fs::create_dir_all(&scratch_root) {
        sim_runner.fail(format!("Failed to prepare viewer temp directory: {err}"));
        return;
    }

    let replay_path = scratch_root.join(format!(
        "replay-{:04}.json",
        working_scene_root.revision.saturating_add(1)
    ));
    let next_scene_root = working_scene_root.next_scene_root();
    let config = match scenario_config(&sim_runner) {
        Ok(config) => config,
        Err(err) => {
            sim_runner.fail(format!("Cannot run this scenario: {err}"));
            return;
        }
    };
    let progress = Arc::new(RunProgress::default());
    let worker = {
        let progress = Arc::clone(&progress);
        let replay_path = replay_path.clone();
        std::thread::Builder::new()
            .name("argusnet-scenario-run".into())
            .spawn(move || simulate_to_replay(&config, &replay_path, &progress))
    };
    match worker {
        Ok(worker) => {
            commands.insert_resource(SimulationProcess {
                worker: Some(worker),
                progress,
                phase: SimPhase::Simulating,
                replay_path,
                next_scene_root,
            });
        }
        Err(err) => {
            sim_runner.fail(format!("Failed to start simulation: {err}"));
        }
    }
}

/// How far a Tools → Scenario run has got: ticks run of the ticks it has.
/// The total is 0 until the scenario is built.
#[derive(Default)]
struct RunProgress {
    ticks: AtomicU64,
    total: AtomicU64,
}

impl RunProgress {
    fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.ticks.load(Ordering::Relaxed) as f64 / total as f64).clamp(0.0, 1.0) as f32
    }
}

/// The `argusnet.scenario-config` the Tools → Scenario panel describes: the
/// options `python -m argusnet sim` was given, flag for flag, so the run is
/// the one that command made. Values the panel holds as `f32` go through
/// their decimal text, as they did on the command line.
fn scenario_config(
    runner: &SimulationRunner,
) -> Result<argusnet_sim::builder::ScenarioConfig, String> {
    use argusnet_sim::builder::{ScenarioConfig, CONFIG_FORMAT, CONFIG_VERSION};
    let decimal = |text: String| text.parse::<f64>().map_err(|error| error.to_string());
    let seed = u32::try_from(runner.seed)
        .map_err(|_| format!("seed {} is outside 0..={}", runner.seed, u32::MAX))?;
    let mut options = serde_json::json!({
        "map_preset": runner.map_preset,
        "terrain_preset": runner.terrain_preset,
        "platform_preset": runner.platform_preset,
        "mission_mode": runner.mission_mode,
        "drone_count": runner.drone_count,
        "ground_station_count": runner.ground_stations,
        "clean_terrain": runner.clean_terrain,
    });
    if runner.mission_mode == "scan_map_inspect" {
        options["scan_coverage_threshold"] =
            decimal(format!("{:.2}", runner.scan_coverage_threshold))?.into();
        options["poi_count"] = runner.poi_count.into();
    } else {
        options["target_motion_preset"] = runner.target_motion.clone().into();
        options["drone_mode_preset"] = runner.drone_mode.clone().into();
        options["target_count"] = runner.target_count.into();
    }
    let config = serde_json::json!({
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "seed": seed,
        "duration_s": decimal(runner.duration_s.to_string())?,
        "options": options,
    });
    ScenarioConfig::from_json(&config.to_string()).map_err(|error| error.to_string())
}

/// Build `config`, run it to completion and write its replay to
/// `replay_path`: what `argusnet-rs sim --config <config> --replay <path>`
/// does, on this thread.
fn simulate_to_replay(
    config: &argusnet_sim::builder::ScenarioConfig,
    replay_path: &Path,
    progress: &RunProgress,
) -> Result<(), String> {
    use argusnet_sim::scenario::{assemble, RunError, ScenarioBundle};
    use argusnet_sim::simulation::TickError;
    let built = argusnet_sim::builder::build_bundle(config).map_err(|error| error.to_string())?;
    let bundle =
        ScenarioBundle::from_json(&built.to_string()).map_err(|error| error.to_string())?;
    let mut run = assemble(&bundle).map_err(|error| error.to_string())?;
    // The CLI's defaults: a prior world, no delivery lag, no faults.
    argusnet_sim::run_options::RunOptions::default().apply(&mut run, &bundle)?;
    run.extend_replay_meta([(
        "provenance".to_string(),
        serde_json::json!({
            "runtime": "argusnet-viewer",
            "version": env!("CARGO_PKG_VERSION"),
            "bundle_format": bundle.format,
            "bundle_version": bundle.version,
            "input": "config",
            "config": serde_json::to_value(config).ok(),
        }),
    )]);
    progress.total.store(bundle.steps, Ordering::Relaxed);
    loop {
        match run.step() {
            Ok(_) => {
                progress.ticks.fetch_add(1, Ordering::Relaxed);
            }
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error @ RunError::Tick(_)) => return Err(error.to_string()),
            Err(error) => return Err(format!("tick {}: {error}", run.sim.tick())),
        }
    }
    let replay = run.run_to_replay().map_err(|error| error.to_string())?;
    let json = argusnet_sim::replay::to_json(&replay).map_err(|error| error.to_string())?;
    fs::write(replay_path, json).map_err(|error| format!("{}: {error}", replay_path.display()))
}

/// Build the scene package for `replay_path` into `output_root`, replacing
/// whatever is there: what `python -m argusnet build-scene` did, in Rust.
fn build_scene_from_replay_file(replay_path: &Path, output_root: &Path) -> Result<()> {
    let text = fs::read_to_string(replay_path)
        .with_context(|| format!("reading {}", replay_path.display()))?;
    let replay: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", replay_path.display()))?;
    if output_root.exists() {
        fs::remove_dir_all(output_root)
            .with_context(|| format!("clearing {}", output_root.display()))?;
    }
    let meta = replay.get("meta").cloned().unwrap_or_default();
    argusnet_sim::scene_build::build_scene(
        output_root,
        &meta,
        Some(&text),
        serde_json::json!({ "replay_source": replay_path.display().to_string() }),
    )
    .map_err(|error| anyhow::anyhow!(error))?;
    Ok(())
}

/// Polls the running simulation and rebuilds/reloads the scene when it is done.
#[allow(clippy::too_many_arguments)]
fn poll_simulation_system(
    mut commands: Commands,
    mut sim_process: Option<ResMut<SimulationProcess>>,
    mut sim_runner: ResMut<SimulationRunner>,
    mut working_scene_root: ResMut<WorkingSceneRoot>,
    mut scene_package: ResMut<ScenePackage>,
    mut replay_state: ResMut<ReplayState>,
    mut mission_zones: ResMut<LoadedMissionZones>,
    mut layer_visibility: ResMut<LayerVisibilityState>,
    mut layer_entities: ResMut<LayerEntityIndex>,
    mut selection: ResMut<SelectionState>,
    mut frame_metrics: ResMut<CurrentFrameMetrics>,
    mut reconstruction: ResMut<ReconstructionCloud>,
    asset_server: Res<AssetServer>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), With<MainCamera>>,
) {
    let Some(process) = sim_process.as_mut() else {
        return;
    };

    if !process.worker.as_ref().is_some_and(JoinHandle::is_finished) {
        if process.phase == SimPhase::Simulating {
            sim_runner.set_phase(SimPhase::Simulating, 0.48 * process.progress.fraction());
        }
        return;
    }
    let outcome = match process.worker.take().map(JoinHandle::join) {
        Some(Ok(outcome)) => outcome,
        Some(Err(_)) => Err("the simulation thread panicked".to_string()),
        None => Err("the simulation is not running".to_string()),
    };
    if let Err(err) = outcome {
        sim_runner.fail(format!("Simulation failed: {err}"));
        commands.remove_resource::<SimulationProcess>();
        return;
    }
    // The scene is built in process from the replay the simulation wrote
    // (`argusnet_sim::scene_build`, D2).
    sim_runner.set_phase(SimPhase::BuildingScene, 0.55);
    if let Err(err) = build_scene_from_replay_file(&process.replay_path, &process.next_scene_root) {
        sim_runner.fail(format!("Failed to build rebuilt scene: {err}"));
        commands.remove_resource::<SimulationProcess>();
        return;
    }
    sim_runner.set_phase(SimPhase::ReloadingScene, 0.92);
    let reload_result = reload_rebuilt_scene(
        &mut commands,
        asset_server.as_ref(),
        &mut camera_query,
        &mut working_scene_root,
        &mut scene_package,
        &mut replay_state,
        &mut mission_zones,
        &mut layer_visibility,
        &mut layer_entities,
        &mut selection,
        &mut frame_metrics,
        &process.next_scene_root,
    );
    match reload_result {
        Ok(()) => {
            reconstruction.reset();
            sim_runner.finish();
        }
        Err(err) => sim_runner.fail(err.to_string()),
    }
    commands.remove_resource::<SimulationProcess>();
}

#[allow(clippy::too_many_arguments)]
fn reload_rebuilt_scene(
    commands: &mut Commands,
    asset_server: &AssetServer,
    camera_query: &mut Query<(&mut OrbitCamera, &mut Transform), With<MainCamera>>,
    working_scene_root: &mut WorkingSceneRoot,
    scene_package: &mut ScenePackage,
    replay_state: &mut ReplayState,
    mission_zones: &mut LoadedMissionZones,
    layer_visibility: &mut LayerVisibilityState,
    layer_entities: &mut LayerEntityIndex,
    selection: &mut SelectionState,
    frame_metrics: &mut CurrentFrameMetrics,
    next_scene_root: &Path,
) -> Result<()> {
    let next_scene_package = ScenePackage::load(next_scene_root)?;
    let next_replay_document = next_scene_package
        .replay
        .clone()
        .map(ReplayDocument::try_from)
        .transpose()
        .context("failed to parse rebuilt replay.json")?;

    let bundle = prepare_reloaded_scene_bundle(
        scene_package,
        replay_state,
        layer_visibility,
        next_scene_package,
        next_replay_document,
    );

    let old_entities: Vec<_> = layer_entities.entities_by_layer.values().copied().collect();
    for entity in old_entities {
        commands.entity(entity).despawn();
    }

    *scene_package = bundle.scene_package;
    *replay_state = bundle.replay_state;
    *mission_zones = bundle.mission_zones;
    *layer_visibility = bundle.layer_visibility;
    working_scene_root.advance_to(scene_package.root.clone());

    // The new run is a new epoch (see `preserved_replay_state`): the adapter
    // drops every entity keyed to the old one on its next apply.
    selection.clear();
    frame_metrics.metrics = None;

    spawn_base_layers(
        commands,
        asset_server,
        working_scene_root,
        scene_package,
        layer_visibility,
        layer_entities,
    );

    if bundle.should_reframe_camera {
        reframe_camera_for_scene(scene_package, camera_query);
    }

    Ok(())
}

// Build the next viewer state first, then apply it as a single bundle so
// replay/zones/scene metadata stay in sync across the hot-swap boundary.
fn prepare_reloaded_scene_bundle(
    current_scene_package: &ScenePackage,
    current_replay_state: &ReplayState,
    current_layer_visibility: &LayerVisibilityState,
    next_scene_package: ScenePackage,
    next_replay_document: Option<ReplayDocument>,
) -> ReloadedSceneBundle {
    let terrain_mesh = next_replay_document
        .as_ref()
        .and_then(|document| document.terrain_viewer_mesh());
    if terrain_mesh.is_none() && next_replay_document.is_some() {
        warn!(
            "Reloaded replay has no terrain mesh — 3-D elevation will fall back to entity altitude"
        );
    }
    let mission_zones = LoadedMissionZones {
        zones: next_replay_document
            .as_ref()
            .map(|document| document.zones())
            .unwrap_or_default(),
        terrain_mesh,
    };

    ReloadedSceneBundle {
        replay_state: preserved_replay_state(current_replay_state, next_replay_document),
        layer_visibility: merged_layer_visibility(
            &next_scene_package,
            Some(current_layer_visibility),
        ),
        should_reframe_camera: scene_extents_changed(current_scene_package, &next_scene_package),
        scene_package: next_scene_package,
        mission_zones,
    }
}

fn preserved_replay_state(previous: &ReplayState, document: Option<ReplayDocument>) -> ReplayState {
    let progress = if previous.frame_count() > 1 {
        previous.frame_index as f32 / previous.frame_count().saturating_sub(1) as f32
    } else {
        0.0
    };
    let mut next = ReplayState::new(document);
    next.succeed(previous);
    next.playback_speed = previous.playback_speed;
    if next.frame_count() > 1 {
        let max_index = next.frame_count().saturating_sub(1);
        next.frame_index = ((max_index as f32) * progress).round() as usize;
        next.playing = previous.playing;
    } else {
        next.frame_index = 0;
        next.playing = false;
    }
    next
}

fn scene_extents_changed(current: &ScenePackage, next: &ScenePackage) -> bool {
    let current_bounds = &current.environment.bounds_xy_m;
    let next_bounds = &next.environment.bounds_xy_m;
    if (current_bounds.x_min_m - next_bounds.x_min_m).abs() > 1.0e-3
        || (current_bounds.x_max_m - next_bounds.x_max_m).abs() > 1.0e-3
        || (current_bounds.y_min_m - next_bounds.y_min_m).abs() > 1.0e-3
        || (current_bounds.y_max_m - next_bounds.y_max_m).abs() > 1.0e-3
    {
        return true;
    }

    let current_heights = current.environment.terrain_summary.as_ref();
    let next_heights = next.environment.terrain_summary.as_ref();
    current_heights.and_then(|summary| summary.min_height_m)
        != next_heights.and_then(|summary| summary.min_height_m)
        || current_heights.and_then(|summary| summary.max_height_m)
            != next_heights.and_then(|summary| summary.max_height_m)
}

fn reframe_camera_for_scene(
    scene_package: &ScenePackage,
    camera_query: &mut Query<(&mut OrbitCamera, &mut Transform), With<MainCamera>>,
) {
    let Ok((mut orbit, mut transform)) = camera_query.single_mut() else {
        return;
    };

    let terrain_summary = scene_package.environment.terrain_summary.as_ref();
    *orbit = OrbitCamera::from_bounds(
        &scene_package.environment.bounds_xy_m,
        terrain_summary.and_then(|summary| summary.min_height_m),
        terrain_summary.and_then(|summary| summary.max_height_m),
    );
    let eye = orbit.eye_position();
    let focus = orbit.focus;
    *transform = Transform::from_translation(eye).looking_at(focus, Vec3::Z);
}

#[cfg(test)]
mod tests {
    /// View mode cycles with `M` in a scene with no replay: before P2.4b the
    /// key sat after the "no replay" return and did nothing there.
    /// The shell's context is on a camera of its own, covering the whole
    /// window and drawn last, never on a world camera: Split halves the
    /// world cameras' viewports and must not halve the shell.
    #[test]
    fn the_ui_context_is_on_its_own_full_window_camera() {
        use bevy::prelude::*;
        let mut app = App::new();
        let recon = app
            .world_mut()
            .spawn((Camera::default(), crate::state::ReconstructionCamera))
            .id();
        let main = app
            .world_mut()
            .spawn((Camera::default(), super::MainCamera))
            .id();
        app.add_systems(Update, super::spawn_ui_camera);
        app.update();
        let world = app.world_mut();
        assert!(world.get::<bevy_egui::PrimaryEguiContext>(main).is_none());
        assert!(world.get::<bevy_egui::PrimaryEguiContext>(recon).is_none());
        let mut ui = world.query_filtered::<&Camera, With<super::UiCamera>>();
        let camera = ui.single(world).unwrap();
        assert!(camera.viewport.is_none(), "the whole window");
        assert!(camera.order > 1, "after the world cameras");
    }

    /// Split halves the canvas the shell leaves, not the window: the world
    /// on the left of it, the reconstruction on the right, in physical
    /// pixels.
    #[test]
    fn split_halves_the_canvas() {
        use bevy::math::UVec2;
        // A canvas from (194, 92) to (1188, 800) logical, at scale 2.
        let (main, recon) =
            super::split_viewports(Some([194.0, 92.0, 1188.0, 800.0]), 2.0, 2880, 1800);
        assert_eq!(main.physical_position, UVec2::new(388, 184));
        assert_eq!(main.physical_size, UVec2::new(994, 1416));
        assert_eq!(recon.physical_position, UVec2::new(388 + 994, 184));
        assert_eq!(recon.physical_size, UVec2::new(994, 1416));
        // No shell (headless): the halves of the target.
        let (main, recon) = super::split_viewports(None, 1.0, 650, 480);
        assert_eq!(
            (main.physical_position, main.physical_size),
            (UVec2::ZERO, UVec2::new(325, 480))
        );
        assert_eq!(recon.physical_position, UVec2::new(325, 0));
    }

    #[test]
    fn cycle_view_works_without_a_replay() {
        use bevy::prelude::*;
        let mut app = App::new();
        app.insert_resource(crate::replay::ReplayState::default())
            .insert_resource(crate::state::ViewMode::RealWorld)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<crate::bindings::KeyboardGate>()
            .init_resource::<crate::bindings::Bindings>()
            .add_systems(Update, super::keyboard_playback_system);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyM);
        app.update();
        assert_eq!(
            *app.world().resource::<crate::state::ViewMode>(),
            crate::state::ViewMode::ScanMap
        );
    }

    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{
        footprint_ground_z_m, generation_rejection_position, prepare_reloaded_scene_bundle,
        preserved_replay_state, tracker_rejection_position,
    };
    use crate::mission_zones::{
        build_group_contour_segments, build_zone_overlap_model, compute_zone_overlap_components,
        sample_terrain_height,
    };
    use crate::replay::{
        FrameMetrics, MissionZone, NodeState, RejectedObservation, ReplayDocument, ReplayFrame,
        ReplayState, TerrainViewerMesh,
    };
    use crate::schema::ScenePackage;
    use crate::state::{LayerVisibilityState, ZoneFocus};

    #[test]
    fn generation_rejection_prefers_closest_point_then_attempted_point() {
        let rejection = RejectedObservation {
            node_id: "node-1".into(),
            target_id: "asset-a".into(),
            reason: "terrain_occlusion".into(),
            detail: String::new(),
            timestamp_s: 0.0,
            origin: Some([1.0, 2.0, 3.0]),
            attempted_point: Some([4.0, 5.0, 6.0]),
            closest_point: Some([7.0, 8.0, 9.0]),
            blocker_type: "terrain".into(),
            first_hit_range_m: Some(25.0),
        };
        let position = generation_rejection_position(&rejection).expect("position");
        assert_eq!(position.to_array(), [7.0, 8.0, 9.0]);

        let fallback = generation_rejection_position(&RejectedObservation {
            closest_point: None,
            ..rejection
        })
        .expect("fallback position");
        assert_eq!(fallback.to_array(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn tracker_rejection_prefers_origin_then_node_position() {
        let frame = ReplayFrame {
            timestamp_s: 0.0,
            tracks: Vec::new(),
            truths: Vec::new(),
            nodes: vec![NodeState {
                node_id: "node-1".into(),
                position: [10.0, 20.0, 30.0],
                velocity: [0.0, 0.0, 0.0],
                is_mobile: false,
                health: 1.0,
                sensor_type: "optical".into(),
                fov_half_angle_deg: None,
                max_range_m: None,
                battery_fraction: None,
            }],
            observations: Vec::new(),
            rejected_observations: Vec::new(),
            generation_rejections: Vec::new(),
            metrics: FrameMetrics::default(),
            mapping_state: None,
            localization_state: None,
            inspection_events: Vec::new(),
            deconfliction_events: Vec::new(),
            scan_mission_state: None,
            target_metadata: Vec::new(),
            safety_events: Vec::new(),
            belief_updates: Vec::new(),
            range_scans: Vec::new(),
            world_contacts: Vec::new(),
            mapped_wires: Vec::new(),
        };

        let rejection = RejectedObservation {
            node_id: "node-1".into(),
            target_id: "asset-a".into(),
            reason: "low_confidence".into(),
            detail: String::new(),
            timestamp_s: 0.0,
            origin: None,
            attempted_point: None,
            closest_point: None,
            blocker_type: String::new(),
            first_hit_range_m: None,
        };
        assert_eq!(
            tracker_rejection_position(&frame, &rejection).to_array(),
            [10.0, 20.0, 30.0]
        );

        let with_origin = RejectedObservation {
            origin: Some([1.0, 2.0, 3.0]),
            ..rejection
        };
        assert_eq!(
            tracker_rejection_position(&frame, &with_origin).to_array(),
            [1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn zone_height_sampling_uses_mesh_and_falls_back_to_zone_center() {
        let mesh = TerrainViewerMesh {
            x_min_m: -10.0,
            x_max_m: 10.0,
            y_min_m: -10.0,
            y_max_m: 10.0,
            rows: 2,
            cols: 2,
            heights_m: vec![vec![10.0, 14.0], vec![18.0, 22.0]],
        };

        let sampled = sample_terrain_height(&Some(mesh), 42.0, 0.0, 0.0);
        assert!((sampled - 16.0).abs() < 1.0e-6);
        assert_eq!(sample_terrain_height(&None, 42.0, 0.0, 0.0), 42.0);
    }

    #[test]
    fn overlap_grouping_splits_isolated_and_merges_pairwise_and_chain_connected_zones() {
        let isolated = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 80.0, 0.0, 20.0),
        ];
        let isolated_groups = compute_zone_overlap_components(&isolated);
        assert_eq!(isolated_groups.len(), 2);

        let pair = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 32.0, 0.0, 20.0),
        ];
        let pair_groups = compute_zone_overlap_components(&pair);
        assert_eq!(pair_groups, vec![vec![0, 1]]);

        let chain = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 35.0, 0.0, 20.0),
            zone("c", "patrol", 70.0, 0.0, 20.0),
        ];
        let chain_groups = compute_zone_overlap_components(&chain);
        assert_eq!(chain_groups, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn merged_contour_generation_returns_geometry_for_each_overlap_component() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 35.0, 0.0, 20.0),
            zone("c", "objective", 70.0, 0.0, 20.0),
        ];
        let groups = compute_zone_overlap_components(&zones);
        assert_eq!(groups.len(), 1);

        let contour = build_group_contour_segments(&zones, &groups[0]);
        assert!(contour.len() > 8);
    }

    #[test]
    fn focused_zone_indices_only_expand_the_selected_zone_or_group() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 32.0, 0.0, 20.0),
            zone("c", "objective", 120.0, 0.0, 20.0),
        ];

        let zone_focus = build_zone_overlap_model(&zones, ZoneFocus::Zone(1));
        assert_eq!(zone_focus.focused_zone_indices(), vec![1]);

        let group_focus = build_zone_overlap_model(&zones, ZoneFocus::Group(0));
        assert_eq!(group_focus.focused_zone_indices(), vec![0, 1]);
    }

    #[test]
    fn preserved_replay_state_keeps_relative_progress_and_playback_speed() {
        let previous_document = replay_document_with_zone("old-scene", 18.0, 4);
        let mut previous_state = ReplayState::new(Some(previous_document));
        previous_state.playback_speed = 2.0;
        previous_state.playing = true;
        previous_state.frame_index = 2;

        let next_document = replay_document_with_zone("new-scene", 24.0, 8);
        let next_state = preserved_replay_state(&previous_state, Some(next_document));

        assert_eq!(next_state.playback_speed, 2.0);
        assert!(next_state.playing);
        assert_eq!(next_state.frame_index, 5);
    }

    #[test]
    fn footprint_ground_z_samples_terrain_per_drone() {
        let terrain_mesh = TerrainViewerMesh {
            x_min_m: 0.0,
            x_max_m: 10.0,
            y_min_m: 0.0,
            y_max_m: 10.0,
            rows: 2,
            cols: 2,
            heights_m: vec![vec![4.0, 8.0], vec![12.0, 16.0]],
        };

        assert_eq!(
            footprint_ground_z_m(Some(&terrain_mesh), -5.0, [10.0, 10.0]),
            16.0
        );
        assert_eq!(
            footprint_ground_z_m(Some(&terrain_mesh), -5.0, [20.0, 20.0]),
            -5.0
        );
    }

    #[test]
    fn reload_bundle_replaces_scene_replay_and_zones_together() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!("argusnet-reload-bundle-{suffix}"));
        let current_root = temp_root.join("current");
        let next_root = temp_root.join("next");

        write_scene_package(
            &current_root,
            "current-scene",
            [-50.0, 50.0, -50.0, 50.0],
            32.0,
            &[("terrain-base", true)],
            replay_document_with_zone("current-scene", 18.0, 4),
        );
        write_scene_package(
            &next_root,
            "next-scene",
            [-120.0, 120.0, -90.0, 90.0],
            64.0,
            &[("terrain-base", true), ("buildings", true)],
            replay_document_with_zone("next-scene", 26.0, 6),
        );

        let current_scene = ScenePackage::load(&current_root).unwrap();
        let next_scene = ScenePackage::load(&next_root).unwrap();

        let mut current_replay =
            ReplayState::new(Some(replay_document_with_zone("current-scene", 18.0, 4)));
        current_replay.playback_speed = 1.5;
        current_replay.frame_index = 2;
        current_replay.playing = true;

        let mut visibility = LayerVisibilityState::default();
        visibility.base_layers.insert("terrain-base".into(), false);

        let next_document = next_scene
            .replay
            .clone()
            .map(ReplayDocument::try_from)
            .transpose()
            .unwrap();
        let bundle = prepare_reloaded_scene_bundle(
            &current_scene,
            &current_replay,
            &visibility,
            next_scene,
            next_document,
        );

        assert_eq!(bundle.scene_package.manifest.scene_id, "next-scene");
        assert_eq!(bundle.mission_zones.zones.len(), 1);
        assert_eq!(bundle.mission_zones.zones[0].radius_m, 26.0);
        assert_eq!(bundle.replay_state.playback_speed, 1.5);
        assert!(bundle.should_reframe_camera);
        assert_eq!(
            bundle.layer_visibility.base_layers.get("terrain-base"),
            Some(&false)
        );
        assert_eq!(
            bundle.layer_visibility.base_layers.get("buildings"),
            Some(&true)
        );

        let _ = fs::remove_dir_all(temp_root);
    }

    fn zone(zone_id: &str, zone_type: &str, x: f32, y: f32, radius_m: f32) -> MissionZone {
        MissionZone {
            zone_id: zone_id.into(),
            zone_type: zone_type.into(),
            center: [x, y, 12.0],
            radius_m,
            priority: 1,
            label: zone_id.into(),
        }
    }

    fn replay_document_with_zone(
        scene_id: &str,
        radius_m: f32,
        frame_count: usize,
    ) -> ReplayDocument {
        serde_json::from_value(replay_document_value(scene_id, radius_m, frame_count)).unwrap()
    }

    fn replay_document_value(
        scene_id: &str,
        radius_m: f32,
        frame_count: usize,
    ) -> serde_json::Value {
        json!({
            "meta": {
                "scene_id": scene_id,
                "dt_s": 1.0,
                "zones": [{
                    "zone_id": format!("{scene_id}-zone"),
                    "zone_type": "surveillance",
                    "center": [0.0, 0.0, 12.0],
                    "radius_m": radius_m,
                    "priority": 1,
                    "label": format!("{scene_id}-zone")
                }],
                "terrain": {
                    "viewer_mesh": {
                        "x_min_m": -10.0,
                        "x_max_m": 10.0,
                        "y_min_m": -10.0,
                        "y_max_m": 10.0,
                        "rows": 2,
                        "cols": 2,
                        "heights_m": [[8.0, 12.0], [16.0, 20.0]]
                    }
                }
            },
            "frames": (0..frame_count).map(|index| json!({
                "timestamp_s": index as f32,
                "tracks": [],
                "truths": [],
                "nodes": []
            })).collect::<Vec<_>>()
        })
    }

    fn write_scene_package(
        root: &std::path::Path,
        scene_id: &str,
        bounds: [f32; 4],
        max_height_m: f32,
        layers: &[(&str, bool)],
        replay_document: ReplayDocument,
    ) {
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::create_dir_all(root.join("replay")).unwrap();

        fs::write(
            root.join("scene_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "format_version": "smartscene-v1",
                "scene_id": scene_id,
                "bounds_xy_m": {
                    "x_min_m": bounds[0],
                    "x_max_m": bounds[1],
                    "y_min_m": bounds[2],
                    "y_max_m": bounds[3]
                },
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "source_crs_id": "local-enu",
                "layers": layers.iter().map(|(layer_id, _)| json!({
                    "id": layer_id,
                    "kind": "mesh",
                    "semantic_kind": layer_id,
                    "asset_path": format!("terrain/{layer_id}.glb"),
                    "style_id": layer_id
                })).collect::<Vec<_>>(),
                "replay": {"path": "replay/replay.json"},
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json"
                },
                "build": {"source_kind": "synthetic"}
            }))
            .unwrap(),
        )
        .unwrap();

        fs::write(
            root.join("metadata/style.json"),
            serde_json::to_string_pretty(&json!({
                "style_version": "smartstyle-v1",
                "layers": layers.iter().map(|(layer_id, default_visibility)| json!({
                    "id": layer_id,
                    "semantic_kind": layer_id,
                    "color_rgba": [1.0, 1.0, 1.0, 1.0],
                    "opacity": 1.0,
                    "elevation_mode": "terrain-draped",
                    "draw_order": 0,
                    "default_visibility": default_visibility
                })).chain([
                    json!({
                        "id": "tracks",
                        "semantic_kind": "tracks",
                        "color_rgba": [1.0, 1.0, 1.0, 1.0],
                        "opacity": 1.0,
                        "elevation_mode": "absolute",
                        "draw_order": 10,
                        "default_visibility": true
                    }),
                    json!({
                        "id": "truths",
                        "semantic_kind": "truths",
                        "color_rgba": [1.0, 1.0, 1.0, 1.0],
                        "opacity": 1.0,
                        "elevation_mode": "absolute",
                        "draw_order": 10,
                        "default_visibility": true
                    }),
                    json!({
                        "id": "nodes",
                        "semantic_kind": "nodes",
                        "color_rgba": [1.0, 1.0, 1.0, 1.0],
                        "opacity": 1.0,
                        "elevation_mode": "absolute",
                        "draw_order": 10,
                        "default_visibility": true
                    })
                ]).collect::<Vec<_>>()
            }))
            .unwrap(),
        )
        .unwrap();

        fs::write(
            root.join("metadata/environment.json"),
            serde_json::to_string_pretty(&json!({
                "source_kind": "synthetic",
                "scene_id": scene_id,
                "source_crs_id": "local-enu",
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "bounds_xy_m": {
                    "x_min_m": bounds[0],
                    "x_max_m": bounds[1],
                    "y_min_m": bounds[2],
                    "y_max_m": bounds[3]
                },
                "terrain_summary": {
                    "min_height_m": 8.0,
                    "max_height_m": max_height_m
                }
            }))
            .unwrap(),
        )
        .unwrap();

        fs::write(
            root.join("replay/replay.json"),
            serde_json::to_string_pretty(&replay_document_value(
                scene_id,
                replay_document.zones()[0].radius_m,
                replay_document.frames.len(),
            ))
            .unwrap(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod overlay_gating_tests {
    use super::*;

    /// The run conditions read resources, so exercise them through a World the
    /// same way the schedule does.
    fn run_condition<M>(system: impl IntoSystem<(), bool, M>, world: &mut World) -> bool {
        let mut system = IntoSystem::into_system(system);
        system.initialize(world);
        system
            .run((), world)
            .expect("run condition should not fail")
    }

    #[test]
    fn gizmo_overlays_are_skipped_when_their_toggle_is_off() {
        let mut world = World::new();
        world.insert_resource(MissionOverlaySettings {
            show_scan_grid: false,
            show_poi_markers: false,
            show_loc_ellipses: false,
            show_reconstruction: false,
            show_coord_frame: false,
            show_egress_paths: false,
        });
        assert!(!run_condition(scan_grid_enabled, &mut world));
        assert!(!run_condition(poi_markers_enabled, &mut world));
        assert!(!run_condition(loc_ellipses_enabled, &mut world));
        assert!(!run_condition(coord_frame_enabled, &mut world));
        assert!(!run_condition(egress_paths_enabled, &mut world));
    }

    #[test]
    fn poi_markers_run_for_either_of_the_two_overlays_they_draw() {
        let mut world = World::new();
        // That system draws POI markers *and* localization ellipses, so it must
        // run when either is on — gating it on `show_poi_markers` alone would
        // silently drop the ellipses.
        world.insert_resource(MissionOverlaySettings {
            show_poi_markers: false,
            show_loc_ellipses: true,
            ..Default::default()
        });
        assert!(run_condition(poi_markers_enabled, &mut world));

        world.insert_resource(MissionOverlaySettings {
            show_poi_markers: true,
            show_loc_ellipses: false,
            ..Default::default()
        });
        assert!(run_condition(poi_markers_enabled, &mut world));
    }
}

#[cfg(test)]
mod scenario_run_tests {
    use super::*;
    use argusnet_sim::builder::ScenarioConfig;

    fn config(json: serde_json::Value) -> ScenarioConfig {
        ScenarioConfig::from_json(&json.to_string()).unwrap()
    }

    #[test]
    fn the_panel_describes_the_run_python_was_asked_for() {
        let runner = SimulationRunner {
            scan_coverage_threshold: 0.7,
            duration_s: 90.5,
            ..Default::default()
        };
        // `python -m argusnet sim` was given the scan flags only; the target
        // options keep their defaults.
        assert_eq!(
            scenario_config(&runner).unwrap(),
            config(serde_json::json!({
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 7,
                "duration_s": 90.5,
                "options": {
                    "map_preset": "regional",
                    "terrain_preset": "military_compound",
                    "platform_preset": "baseline",
                    "mission_mode": "scan_map_inspect",
                    "drone_count": 4,
                    "ground_station_count": 7,
                    "clean_terrain": false,
                    "scan_coverage_threshold": 0.7,
                    "poi_count": 3
                }
            }))
        );
        // The threshold goes through the `{:.2}` text the flag carried.
        assert_eq!(
            scenario_config(&SimulationRunner {
                scan_coverage_threshold: 0.678,
                ..Default::default()
            })
            .unwrap()
            .options
            .scan_coverage_threshold,
            0.68
        );
    }

    #[test]
    fn a_tracking_run_carries_the_target_options_and_not_the_scan_ones() {
        let runner = SimulationRunner {
            mission_mode: "target_tracking".into(),
            target_motion: "racetrack".into(),
            drone_mode: "search".into(),
            target_count: 3,
            scan_coverage_threshold: 0.4,
            poi_count: 9,
            ..Default::default()
        };
        let options = scenario_config(&runner).unwrap().options;
        assert_eq!(options.target_motion_preset, "racetrack");
        assert_eq!(options.drone_mode_preset, "search");
        assert_eq!(options.target_count, 3);
        assert_eq!(options.scan_coverage_threshold, 0.7);
        assert_eq!(options.poi_count, 3);
    }

    #[test]
    fn a_seed_the_simulation_cannot_take_is_refused() {
        let runner = SimulationRunner {
            seed: -1,
            ..Default::default()
        };
        assert!(scenario_config(&runner).unwrap_err().contains("seed -1"));
    }

    #[test]
    fn the_run_writes_a_replay_of_every_tick_and_reports_its_progress() {
        let runner = SimulationRunner {
            map_preset: "small".into(),
            terrain_preset: "default".into(),
            drone_count: 1,
            ground_stations: 2,
            duration_s: 2.0,
            ..Default::default()
        };
        let config = scenario_config(&runner).unwrap();
        let path =
            std::env::temp_dir().join(format!("argusnet-scenario-run-{}.json", std::process::id()));
        let progress = RunProgress::default();
        assert_eq!(progress.fraction(), 0.0);
        simulate_to_replay(&config, &path, &progress).unwrap();
        let replay: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let _ = fs::remove_file(&path);
        let ticks = progress.ticks.load(Ordering::Relaxed);
        assert_eq!(ticks, progress.total.load(Ordering::Relaxed));
        assert_eq!(progress.fraction(), 1.0);
        assert_eq!(replay["frames"].as_array().unwrap().len() as u64, ticks);
        assert_eq!(replay["meta"]["provenance"]["runtime"], "argusnet-viewer");
        // A replay the scene builder will take.
        argusnet_sim::scene_build::validate_replay_document(&replay).unwrap();
    }

    /// D5: the viewer runs and builds everything in process. No source file
    /// starts a process (Python, Cargo or anything else).
    #[test]
    fn the_viewer_starts_no_processes() {
        let needle = ["process::", "Command"].concat();
        let mut pending = vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")];
        let mut offenders = Vec::new();
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|ext| ext == "rs")
                    && fs::read_to_string(&path).unwrap().contains(&needle)
                {
                    offenders.push(path);
                }
            }
        }
        assert!(offenders.is_empty(), "{offenders:?} start processes");
    }
}

#[cfg(test)]
mod fog_tests {
    use super::fog_band;

    const SPAN: f32 = 4320.0;

    #[test]
    fn the_band_straddles_the_terrain_rather_than_starting_at_the_camera() {
        // Default framing: the camera orbits outside the scene, so fog must not
        // begin until well into it. Deriving the band from the span alone — the
        // first attempt — put the start at 1512 m and blacked out the whole site.
        let radius = SPAN * 1.5;
        let (start, end) = fog_band(radius, SPAN);
        let near_edge = radius - SPAN * 0.5;
        let far_edge = radius + SPAN * 0.5;
        assert!(
            start > near_edge,
            "fog must clear the near edge of the terrain"
        );
        assert!(start < far_edge, "fog must engage before the far edge");
        assert!(end > far_edge, "fog must extend past the far edge");
    }

    #[test]
    fn the_band_tracks_zoom() {
        // Fog distances are measured from the camera, so zooming in has to pull
        // the band in with it or the effect either vanishes or swallows the view.
        let (near_start, near_end) = fog_band(SPAN * 0.6, SPAN);
        let (far_start, far_end) = fog_band(SPAN * 3.0, SPAN);
        assert!(far_start > near_start);
        assert!(far_end > near_end);
    }

    #[test]
    fn a_camera_inside_the_scene_still_gets_a_valid_band() {
        // Zoomed close enough that `radius - span/2` goes negative; the band must
        // stay positive and ordered rather than inverting.
        let (start, end) = fog_band(1.0, SPAN);
        assert!(start > 0.0);
        assert!(end > start);
    }
}
