use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin};

use crate::mission_zones::{self, build_projected_badges_system, ProjectedZoneBadges};
use crate::orbit_camera::{OrbitCamera, OrbitCameraPlugin};
use crate::replay::{
    MarkerKind, RejectedObservation, ReplayDocument, ReplayState, RuntimeMarker,
};
use crate::schema::ScenePackage;
use crate::state::{
    CurrentFrameMetrics, CurrentRuntimeMarkers, LayerEntityIndex, LayerVisibilityState,
    LoadedMissionZones, RuntimeOverlayVisibility, SelectionState, SimPhase, SimulationRunner,
    WorkingSceneRoot, ZoneFocus,
};
use crate::ui::viewer_ui_system;

/// Maximum number of historical positions kept in a trail.
const TRAIL_MAX_POINTS: usize = 32;

/// Active simulation subprocess — inserted as a resource while running.
#[derive(Resource)]
struct SimulationProcess {
    child: Child,
    phase: SimPhase,
    replay_path: PathBuf,
    next_scene_root: PathBuf,
}

#[derive(Component)]
struct MainCamera;

struct ReloadedSceneBundle {
    scene_package: ScenePackage,
    replay_state: ReplayState,
    mission_zones: LoadedMissionZones,
    layer_visibility: LayerVisibilityState,
    should_reframe_camera: bool,
}


pub fn run(scene_path: impl AsRef<Path>) -> Result<()> {
    let (scene_package, working_scene_root) = initialize_scene_package(scene_path.as_ref())?;
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
    let mut zone_overlap_model =
        mission_zones::build_zone_overlap_model(&mission_zones, ZoneFocus::None);
    zone_overlap_model.generation = 1;

    let layer_visibility = merged_layer_visibility(&scene_package, None);

    let asset_root = working_scene_root.asset_root.clone();

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.18, 0.20, 0.22)))
        .insert_resource(scene_package)
        .insert_resource(working_scene_root)
        .insert_resource(ReplayState::new(replay_document))
        .insert_resource(layer_visibility)
        .insert_resource(RuntimeOverlayVisibility::default())
        .insert_resource(CurrentRuntimeMarkers::default())
        .insert_resource(CurrentFrameMetrics::default())
        .insert_resource(SelectionState::default())
        .insert_resource(zone_overlap_model)
        .insert_resource(LoadedMissionZones {
            zones: mission_zones,
            terrain_mesh,
        })
        .insert_resource(LayerEntityIndex::default())
        .insert_resource(SimulationRunner::default())
        .insert_resource(ProjectedZoneBadges::default())
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Smart Tracker Viewer".into(),
                        resolution: (1440.0, 920.0).into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin)
        .add_plugins(OrbitCameraPlugin)
        .add_systems(Startup, setup_world)
        .add_systems(
            Update,
            (
                advance_playback_system,
                keyboard_playback_system,
                spawn_simulation_system,
                poll_simulation_system,
                mission_zones::refresh_zone_overlap_model_system,
                sync_current_markers_system,
                sync_base_layer_visibility_system,
                pick_runtime_marker_system,
                draw_runtime_overlays_system,
                draw_sensor_overlays_system,
                mission_zones::draw_mission_zones_system,
                build_projected_badges_system,
            )
                .chain(),
        )
        .add_systems(Update, viewer_ui_system.in_set(bevy_egui::EguiSet::BeginPass))
        .run();

    Ok(())
}

fn setup_world(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    working_scene_root: Res<WorkingSceneRoot>,
    scene_package: Res<ScenePackage>,
    layer_visibility: Res<LayerVisibilityState>,
    mut layer_entities: ResMut<LayerEntityIndex>,
) {
    let terrain_summary = scene_package.environment.terrain_summary.as_ref();
    let orbit = OrbitCamera::from_bounds(
        &scene_package.environment.bounds_xy_m,
        terrain_summary.and_then(|summary| summary.min_height_m),
        terrain_summary.and_then(|summary| summary.max_height_m),
    );
    let eye = orbit.eye_position();
    let focus = orbit.focus;

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.95, 0.96, 0.98),
        brightness: 350.0,
        ..default()
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(2000.0, 1200.0, 2600.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));

    commands.spawn((
        MainCamera,
        orbit,
        Camera3d::default(),
        Transform::from_translation(eye).looking_at(focus, Vec3::Z),
    ));
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
    let root = std::env::temp_dir().join(format!(
        "smart-tracker-viewer-{}-{suffix}",
        std::process::id()
    ));
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
        let asset_path = working_scene_root.asset_path_for(&scene_package.root, &layer.asset_path);
        let handle: Handle<Scene> = asset_server.load(format!("{asset_path}#Scene0"));
        let visibility = if *layer_visibility.base_layers.get(&layer.id).unwrap_or(&true) {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let entity = commands
            .spawn((Name::new(layer.id.clone()), SceneRoot(handle), visibility))
            .id();
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

fn advance_playback_system(time: Res<Time>, mut replay_state: ResMut<ReplayState>) {
    replay_state.advance(time.delta_secs());
}

fn keyboard_playback_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut replay_state: ResMut<ReplayState>,
    mut contexts: EguiContexts,
) {
    // Don't capture keys when egui wants keyboard input (e.g. text fields).
    if contexts.ctx_mut().wants_keyboard_input() {
        return;
    }
    if replay_state.frame_count() == 0 {
        return;
    }

    if keys.just_pressed(KeyCode::Space) {
        replay_state.playing = !replay_state.playing;
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        replay_state.playing = false;
        let prev = replay_state.frame_index.saturating_sub(1);
        replay_state.step_to(prev);
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        replay_state.playing = false;
        let next = replay_state.frame_index + 1;
        replay_state.step_to(next);
    }
    if keys.just_pressed(KeyCode::Home) {
        replay_state.playing = false;
        replay_state.step_to(0);
    }
    if keys.just_pressed(KeyCode::End) {
        replay_state.playing = false;
        let last = replay_state.frame_count().saturating_sub(1);
        replay_state.step_to(last);
    }
}

fn sync_current_markers_system(
    replay_state: Res<ReplayState>,
    mut markers: ResMut<CurrentRuntimeMarkers>,
    mut frame_metrics: ResMut<CurrentFrameMetrics>,
    mut selection: ResMut<SelectionState>,
) {
    markers.markers = replay_state.current_markers();

    if let Some(frame) = replay_state.current_frame() {
        frame_metrics.metrics = Some(frame.metrics.clone());
    } else {
        frame_metrics.metrics = None;
    }

    let sel_label = selection.selected_label.clone();
    let sel_kind = selection.selected_kind.clone();
    if let (Some(selected_label), Some(selected_kind)) = (sel_label, sel_kind) {
        if let Some(marker) = markers.markers.iter().find(|m| m.label == selected_label) {
            selection.selected_position = Some(marker.position);
            selection.selected_velocity = Some(marker.velocity);
        } else {
            selection.clear();
            return;
        }
        if let Some(frame) = replay_state.current_frame() {
            match selected_kind.as_str() {
                "Track" => {
                    if let Some(track) = frame.tracks.iter().find(|t| t.track_id == selected_label)
                    {
                        selection.selected_measurement_std_m = Some(track.measurement_std_m);
                        selection.selected_update_count = Some(track.update_count);
                        selection.selected_stale_steps = Some(track.stale_steps);
                        selection.selected_track_error_m = frame_metrics
                            .metrics
                            .as_ref()
                            .and_then(|m| m.track_errors_m.get(&selected_label).copied());
                        selection.selected_covariance_diag =
                            track.covariance.as_ref().map(|cov| {
                                crate::ui::covariance_diagonal(cov)
                            });
                        selection.selected_health = None;
                        selection.selected_sensor_type = None;
                        selection.selected_fov_half_angle_deg = None;
                        selection.selected_max_range_m = None;
                        selection.selected_is_mobile = None;
                        selection.selected_nearest_track_dist_m = None;
                    }
                }
                "Node" => {
                    if let Some(node) = frame.nodes.iter().find(|n| n.node_id == selected_label) {
                        selection.selected_health = Some(node.health);
                        selection.selected_sensor_type = Some(node.sensor_type.clone());
                        selection.selected_fov_half_angle_deg = node.fov_half_angle_deg;
                        selection.selected_max_range_m = node.max_range_m;
                        selection.selected_is_mobile = Some(node.is_mobile);
                    }
                    selection.selected_measurement_std_m = None;
                    selection.selected_update_count = None;
                    selection.selected_stale_steps = None;
                    selection.selected_track_error_m = None;
                    selection.selected_covariance_diag = None;
                    selection.selected_nearest_track_dist_m = None;
                }
                "Truth" => {
                    // Find nearest track distance for truth inspection
                    if let Some(truth) = frame.truths.iter().find(|t| t.target_id == selected_label) {
                        let truth_pos = Vec3::from_array(truth.position);
                        let nearest_dist = frame.tracks.iter().map(|t| {
                            truth_pos.distance(Vec3::from_array(t.position))
                        }).fold(f32::INFINITY, f32::min);
                        if nearest_dist < f32::INFINITY {
                            selection.selected_nearest_track_dist_m = Some(nearest_dist);
                        } else {
                            selection.selected_nearest_track_dist_m = None;
                        }
                    }
                    selection.selected_health = None;
                    selection.selected_measurement_std_m = None;
                    selection.selected_update_count = None;
                    selection.selected_stale_steps = None;
                    selection.selected_track_error_m = None;
                    selection.selected_covariance_diag = None;
                    selection.selected_sensor_type = None;
                    selection.selected_fov_half_angle_deg = None;
                    selection.selected_max_range_m = None;
                    selection.selected_is_mobile = None;
                }
                _ => {
                    selection.selected_health = None;
                    selection.selected_measurement_std_m = None;
                    selection.selected_update_count = None;
                    selection.selected_stale_steps = None;
                    selection.selected_track_error_m = None;
                    selection.selected_covariance_diag = None;
                    selection.selected_sensor_type = None;
                    selection.selected_fov_half_angle_deg = None;
                    selection.selected_max_range_m = None;
                    selection.selected_is_mobile = None;
                    selection.selected_nearest_track_dist_m = None;
                }
            }
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
) {
    // Compute altitude range across all current markers for color normalization.
    let alt_min = markers.markers.iter().map(|m| m.position.z).fold(f32::INFINITY, f32::min);
    let alt_max = markers.markers.iter().map(|m| m.position.z).fold(f32::NEG_INFINITY, f32::max);
    let alt_span = (alt_max - alt_min).max(1.0);

    // Retrieve measurement_std_m for each track from the current frame.
    let frame_tracks: Vec<(String, f32)> = replay_state
        .current_frame()
        .map(|f| f.tracks.iter().map(|t| (t.track_id.clone(), t.measurement_std_m)).collect())
        .unwrap_or_default();

    for marker in &markers.markers {
        if !is_marker_visible(marker, &runtime_visibility) {
            continue;
        }

        // Item 5: pick color based on measurement_std_m for track markers.
        let base_color = marker_color(marker.kind, &scene_package);
        let (draw_color, radius_scale) = if marker.kind == crate::replay::MarkerKind::Track {
            let std_m = frame_tracks
                .iter()
                .find(|(id, _)| id == &marker.label)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
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
        draw_marker_scaled(&mut gizmos, marker, draw_color, radius_scale);

        // Item 15: trail with alpha fading and altitude-based hue coloring.
        let trail_points = replay_state.trail_points(marker, TRAIL_MAX_POINTS);
        let n = trail_points.len();
        for (seg_idx, segment) in trail_points.windows(2).enumerate() {
            // t goes from 0 (oldest segment) to 1 (newest segment).
            let t = if n > 2 { seg_idx as f32 / (n - 2) as f32 } else { 1.0 };
            let alpha = 0.1 + 0.9 * t;
            // Use midpoint altitude for hue
            let mid_z = (segment[0].z + segment[1].z) * 0.5;
            let norm_alt = ((mid_z - alt_min) / alt_span).clamp(0.0, 1.0);
            // blue (hue=240) at low alt → red (hue=0) at high alt
            let hue = 240.0 * (1.0 - norm_alt);
            let trail_color = Color::hsla(hue, 1.0, 0.55, alpha);
            gizmos.line(segment[0], segment[1], trail_color);
        }
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
                draw_rejection_marker(&mut gizmos, pos, rejection_marker_style(rej, true));
            }
            for rej in &frame.rejected_observations {
                let pos = tracker_rejection_position(frame, rej);
                draw_rejection_marker(&mut gizmos, pos, rejection_marker_style(rej, false));
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
    size: f32,
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
            size: 7.0,
        };
    }
    match rej.reason.as_str() {
        "terrain_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.76, 0.42, 0.16, 0.92),
            glyph: RejectionGlyph::Diamond,
            size: 7.5,
        },
        "building_occlusion" | "wall_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.92, 0.62, 0.18, 0.92),
            glyph: RejectionGlyph::Square,
            size: 7.0,
        },
        "vegetation_occlusion" => RejectionMarkerStyle {
            color: Color::srgba(0.2, 0.72, 0.32, 0.92),
            glyph: RejectionGlyph::Cross,
            size: 7.0,
        },
        "outside_fov" | "low_elevation" => RejectionMarkerStyle {
            color: Color::srgba(0.95, 0.78, 0.2, 0.92),
            glyph: RejectionGlyph::Triangle,
            size: 8.0,
        },
        "out_of_range" | "out_of_coverage" => RejectionMarkerStyle {
            color: Color::srgba(0.24, 0.66, 0.95, 0.92),
            glyph: RejectionGlyph::Ring,
            size: 7.5,
        },
        "dropout" => RejectionMarkerStyle {
            color: Color::srgba(0.82, 0.84, 0.86, 0.88),
            glyph: RejectionGlyph::Ring,
            size: 6.0,
        },
        _ => RejectionMarkerStyle {
            color: Color::srgba(0.96, 0.34, 0.34, 0.9),
            glyph: RejectionGlyph::Cross,
            size: 7.0,
        },
    }
}

fn draw_rejection_marker(gizmos: &mut Gizmos, position: Vec3, style: RejectionMarkerStyle) {
    match style.glyph {
        RejectionGlyph::Cross => {
            gizmos.line(
                position - Vec3::X * style.size - Vec3::Y * style.size,
                position + Vec3::X * style.size + Vec3::Y * style.size,
                style.color,
            );
            gizmos.line(
                position + Vec3::X * style.size - Vec3::Y * style.size,
                position - Vec3::X * style.size + Vec3::Y * style.size,
                style.color,
            );
        }
        RejectionGlyph::Diamond => {
            let top = position + Vec3::Y * style.size;
            let right = position + Vec3::X * style.size;
            let bottom = position - Vec3::Y * style.size;
            let left = position - Vec3::X * style.size;
            gizmos.line(top, right, style.color);
            gizmos.line(right, bottom, style.color);
            gizmos.line(bottom, left, style.color);
            gizmos.line(left, top, style.color);
        }
        RejectionGlyph::Square => {
            let half = style.size;
            let a = position + Vec3::new(-half, -half, 0.0);
            let b = position + Vec3::new(half, -half, 0.0);
            let c = position + Vec3::new(half, half, 0.0);
            let d = position + Vec3::new(-half, half, 0.0);
            gizmos.line(a, b, style.color);
            gizmos.line(b, c, style.color);
            gizmos.line(c, d, style.color);
            gizmos.line(d, a, style.color);
        }
        RejectionGlyph::Triangle => {
            let top = position + Vec3::Y * style.size;
            let right = position + Vec3::new(style.size * 0.86, -style.size * 0.6, 0.0);
            let left = position + Vec3::new(-style.size * 0.86, -style.size * 0.6, 0.0);
            gizmos.line(top, right, style.color);
            gizmos.line(right, left, style.color);
            gizmos.line(left, top, style.color);
        }
        RejectionGlyph::Ring => {
            let segments = 14;
            for index in 0..segments {
                let a0 = std::f32::consts::TAU * index as f32 / segments as f32;
                let a1 = std::f32::consts::TAU * (index + 1) as f32 / segments as f32;
                let p0 = position + Vec3::new(a0.cos() * style.size, a0.sin() * style.size, 0.0);
                let p1 = position + Vec3::new(a1.cos() * style.size, a1.sin() * style.size, 0.0);
                gizmos.line(p0, p1, style.color);
            }
        }
    }
}

fn pick_runtime_marker_system(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    markers: Res<CurrentRuntimeMarkers>,
    runtime_visibility: Res<RuntimeOverlayVisibility>,
    mut selection: ResMut<SelectionState>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(cursor_position) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera_query.get_single() else {
        return;
    };

    let mut best_match: Option<(&RuntimeMarker, f32)> = None;
    for marker in &markers.markers {
        if !is_marker_visible(marker, &runtime_visibility) {
            continue;
        }
        let Ok(viewport_position) = camera.world_to_viewport(camera_transform, marker.position)
        else {
            continue;
        };
        let distance = cursor_position.distance(viewport_position);
        if distance > 18.0 {
            continue;
        }
        match best_match {
            Some((_, best_distance)) if best_distance <= distance => {}
            _ => best_match = Some((marker, distance)),
        }
    }

    if let Some((marker, _)) = best_match {
        selection.selected_kind = Some(match marker.kind {
            MarkerKind::Track => "Track".to_string(),
            MarkerKind::Truth => "Truth".to_string(),
            MarkerKind::Node => "Node".to_string(),
        });
        selection.selected_label = Some(marker.label.clone());
        selection.selected_position = Some(marker.position);
    } else {
        selection.clear();
    }
}

fn draw_marker_scaled(gizmos: &mut Gizmos, marker: &RuntimeMarker, color: Color, scale: f32) {
    let base_radius = match marker.kind {
        MarkerKind::Track => 12.0,
        MarkerKind::Truth => 10.0,
        MarkerKind::Node => 8.0,
    };
    let radius = base_radius * scale;
    gizmos.line(
        marker.position - Vec3::X * radius,
        marker.position + Vec3::X * radius,
        color,
    );
    gizmos.line(
        marker.position - Vec3::Y * radius,
        marker.position + Vec3::Y * radius,
        color,
    );
    gizmos.line(
        marker.position - Vec3::Z * (radius * 0.6),
        marker.position + Vec3::Z * (radius * 0.6),
        color,
    );

    if marker.velocity.length_squared() > 0.0 {
        gizmos.line(
            marker.position,
            marker.position + marker.velocity.clamp_length_max(60.0) * 0.25,
            color,
        );
    }
}

fn is_marker_visible(
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

/// Draws FOV cones for optical nodes, radar range rings for radar nodes,
/// and vertical launch lines for drones in their climb phase.
fn draw_sensor_overlays_system(
    mut gizmos: Gizmos,
    replay_state: Res<ReplayState>,
    runtime_visibility: Res<RuntimeOverlayVisibility>,
    scene_package: Res<crate::schema::ScenePackage>,
) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    for node in &frame.nodes {
        let pos = Vec3::from_array(node.position);
        let vel = Vec3::from_array(node.velocity);

        // Radar range rings for ground stations
        if runtime_visibility.radar_rings && node.sensor_type == "radar" {
            if let Some(max_range) = node.max_range_m {
                let ring_color = Color::srgba(0.3, 0.7, 0.95, 0.35);
                let bounds = &scene_package.environment.bounds_xy_m;
                let seg_count = 48;
                for i in 0..seg_count {
                    let a0 = std::f32::consts::TAU * i as f32 / seg_count as f32;
                    let a1 = std::f32::consts::TAU * (i + 1) as f32 / seg_count as f32;
                    let raw_p0 = [pos.x + a0.cos() * max_range, pos.y + a0.sin() * max_range];
                    let raw_p1 = [pos.x + a1.cos() * max_range, pos.y + a1.sin() * max_range];
                    let Some([c0, c1]) = mission_zones::clip_segment_to_bounds(raw_p0, raw_p1, bounds) else {
                        continue;
                    };
                    let p0 = Vec3::new(c0[0], c0[1], pos.z);
                    let p1 = Vec3::new(c1[0], c1[1], pos.z);
                    gizmos.line(p0, p1, ring_color);
                }
            }
        }

        // FOV cones for optical drones
        if runtime_visibility.fov_cones && node.sensor_type == "optical" {
            if let Some(fov_half) = node.fov_half_angle_deg {
                let fov_color = Color::srgba(0.95, 0.75, 0.2, 0.4);
                let cone_length = node.max_range_m.unwrap_or(400.0).min(1500.0);
                let half_rad = fov_half.to_radians();

                // Omnidirectional sensor: draw a detection dome in multiple azimuth planes
                if half_rad >= std::f32::consts::PI - 0.05 {
                    let dome_radius = cone_length;
                    let ring_segs = 24;
                    // Horizontal ring at node altitude
                    for i in 0..ring_segs {
                        let a0 = std::f32::consts::TAU * i as f32 / ring_segs as f32;
                        let a1 = std::f32::consts::TAU * (i + 1) as f32 / ring_segs as f32;
                        let p0 = pos + Vec3::new(a0.cos() * dome_radius, a0.sin() * dome_radius, 0.0);
                        let p1 = pos + Vec3::new(a1.cos() * dome_radius, a1.sin() * dome_radius, 0.0);
                        gizmos.line(p0, p1, fov_color);
                    }
                    // Four vertical great-circle arcs
                    let vert_segs = 16;
                    for plane in 0..4 {
                        let az = std::f32::consts::TAU * plane as f32 / 4.0;
                        let axis = Vec3::new(az.cos(), az.sin(), 0.0);
                        let perp = Vec3::new(-az.sin(), az.cos(), 0.0);
                        for i in 0..vert_segs {
                            let el0 = std::f32::consts::PI * i as f32 / vert_segs as f32 - std::f32::consts::FRAC_PI_2;
                            let el1 = std::f32::consts::PI * (i + 1) as f32 / vert_segs as f32 - std::f32::consts::FRAC_PI_2;
                            let _ = perp; // used indirectly via axis below
                            let p0 = pos + (axis * el0.cos() + Vec3::Z * el0.sin()) * dome_radius;
                            let p1 = pos + (axis * el1.cos() + Vec3::Z * el1.sin()) * dome_radius;
                            gizmos.line(p0, p1, fov_color);
                        }
                    }
                } else if vel.length_squared() > 0.25 {
                    let radius_at_tip = cone_length * half_rad.tan();
                    let look = vel.normalize();
                    // Build a local frame from the look direction
                    let right = if look.cross(Vec3::Z).length_squared() > 1e-6 {
                        look.cross(Vec3::Z).normalize()
                    } else {
                        Vec3::X
                    };
                    let up = right.cross(look).normalize_or_zero();
                    let tip_center = pos + look * cone_length;
                    // Smooth cone rim: 24 segments
                    let rim_segs = 24;
                    for i in 0..rim_segs {
                        let a0 = std::f32::consts::TAU * i as f32 / rim_segs as f32;
                        let a1 = std::f32::consts::TAU * (i + 1) as f32 / rim_segs as f32;
                        let p0 = tip_center + (right * a0.cos() + up * a0.sin()) * radius_at_tip;
                        let p1 = tip_center + (right * a1.cos() + up * a1.sin()) * radius_at_tip;
                        gizmos.line(p0, p1, fov_color);
                    }
                    // 12 edge lines from origin to cone boundary (was 4)
                    for i in 0..12 {
                        let a = std::f32::consts::TAU * i as f32 / 12.0;
                        let edge = tip_center + (right * a.cos() + up * a.sin()) * radius_at_tip;
                        gizmos.line(pos, edge, fov_color);
                    }
                }
            }
        }
    }

}

/// Spawns a Python simulation subprocess when the user clicks "Run Simulation".
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
    let python = simulation_python();
    let child = spawn_simulation_child(&python, &sim_runner, &replay_path);

    match child {
        Ok(child) => {
            commands.insert_resource(SimulationProcess {
                child,
                phase: SimPhase::Simulating,
                replay_path,
                next_scene_root,
            });
        }
        Err(err) => {
            sim_runner.fail(format!("Failed to spawn simulation: {err}"));
        }
    }
}

fn simulation_python() -> String {
    if let Ok(p) = std::env::var("SMART_TRACKER_PYTHON") {
        return p;
    }
    // Probe common conda/miniforge installations before falling back to system python3.
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{}/miniforge3/bin/python3", home),
        format!("{}/mambaforge/bin/python3", home),
        format!("{}/miniconda3/bin/python3", home),
        format!("{}/anaconda3/bin/python3", home),
        format!("{}/opt/miniconda3/bin/python3", home),
    ];
    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return candidate.clone();
        }
    }
    "python3".to_string()
}

fn spawn_simulation_child(
    python: &str,
    sim_runner: &SimulationRunner,
    replay_path: &Path,
) -> Result<Child> {
    let mut args = vec![
        "-m".to_string(),
        "smart_tracker".to_string(),
        "sim".to_string(),
        "--map-preset".to_string(),
        sim_runner.map_preset.clone(),
        "--terrain-preset".to_string(),
        sim_runner.terrain_preset.clone(),
        "--platform-preset".to_string(),
        sim_runner.platform_preset.clone(),
        "--target-motion".to_string(),
        sim_runner.target_motion.clone(),
        "--drone-mode".to_string(),
        sim_runner.drone_mode.clone(),
        "--target-count".to_string(),
        sim_runner.target_count.to_string(),
        "--drone-count".to_string(),
        sim_runner.drone_count.to_string(),
        "--ground-stations".to_string(),
        sim_runner.ground_stations.to_string(),
        "--duration-s".to_string(),
        sim_runner.duration_s.to_string(),
        "--seed".to_string(),
        sim_runner.seed.to_string(),
        "--replay".to_string(),
        replay_path.to_string_lossy().into_owned(),
    ];
    if sim_runner.clean_terrain {
        args.push("--clean-terrain".to_string());
    }

    std::process::Command::new(python)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn smart_tracker sim")
}

fn spawn_build_scene_child(python: &str, replay_path: &Path, output_root: &Path) -> Result<Child> {
    if output_root.exists() {
        fs::remove_dir_all(output_root)
            .with_context(|| format!("failed to clear {}", output_root.display()))?;
    }
    if let Some(parent) = output_root.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    std::process::Command::new(python)
        .args([
            "-m",
            "smart_tracker",
            "build-scene",
            "--replay",
            &replay_path.to_string_lossy(),
            "--output",
            &output_root.to_string_lossy(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn smart_tracker build-scene")
}

fn read_process_error(child: &mut Child, fallback: &str) -> String {
    let stderr_msg = child
        .stderr
        .take()
        .and_then(|mut stderr| {
            use std::io::Read;
            let mut buf = String::new();
            stderr.read_to_string(&mut buf).ok()?;
            Some(buf)
        })
        .unwrap_or_else(|| fallback.to_string());
    if stderr_msg.len() > 280 {
        format!("...{}", &stderr_msg[stderr_msg.len() - 280..])
    } else {
        stderr_msg
    }
}

/// Polls the running simulation subprocess and rebuilds/reloads the scene when done.
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
    mut current_markers: ResMut<CurrentRuntimeMarkers>,
    mut frame_metrics: ResMut<CurrentFrameMetrics>,
    asset_server: Res<AssetServer>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), With<MainCamera>>,
) {
    let Some(process) = sim_process.as_mut() else {
        return;
    };

    match process.child.try_wait() {
        Ok(Some(status)) => {
            if status.success() {
                match process.phase {
                    SimPhase::Simulating => {
                        let python = simulation_python();
                        match spawn_build_scene_child(
                            &python,
                            &process.replay_path,
                            &process.next_scene_root,
                        ) {
                            Ok(child) => {
                                process.child = child;
                                process.phase = SimPhase::BuildingScene;
                                sim_runner.set_phase(SimPhase::BuildingScene, 0.55);
                            }
                            Err(err) => {
                                sim_runner.fail(format!("Failed to build rebuilt scene: {err}"));
                                commands.remove_resource::<SimulationProcess>();
                            }
                        }
                    }
                    SimPhase::BuildingScene => {
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
                            &mut current_markers,
                            &mut frame_metrics,
                            &process.next_scene_root,
                        );
                        match reload_result {
                            Ok(()) => sim_runner.finish(),
                            Err(err) => sim_runner.fail(err.to_string()),
                        }
                        commands.remove_resource::<SimulationProcess>();
                    }
                    _ => {
                        sim_runner.fail("Simulation entered an unexpected process phase.");
                        commands.remove_resource::<SimulationProcess>();
                    }
                }
            } else {
                let message = match process.phase {
                    SimPhase::Simulating => "Simulation failed",
                    SimPhase::BuildingScene => "Scene build failed",
                    _ => "Process failed",
                };
                sim_runner.fail(read_process_error(&mut process.child, message));
                commands.remove_resource::<SimulationProcess>();
            }
        }
        Ok(None) => match process.phase {
            SimPhase::Simulating => {
                let next_progress = (sim_runner.progress + 0.01).clamp(0.0, 0.48);
                sim_runner.set_phase(SimPhase::Simulating, next_progress);
            }
            SimPhase::BuildingScene => {
                let next_progress = (sim_runner.progress + 0.01).clamp(0.55, 0.88);
                sim_runner.set_phase(SimPhase::BuildingScene, next_progress);
            }
            _ => {}
        },
        Err(err) => {
            sim_runner.fail(format!("Process error: {err}"));
            commands.remove_resource::<SimulationProcess>();
        }
    }
}

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
    current_markers: &mut CurrentRuntimeMarkers,
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
        commands.entity(entity).despawn_recursive();
    }

    *scene_package = bundle.scene_package;
    *replay_state = bundle.replay_state;
    *mission_zones = bundle.mission_zones;
    *layer_visibility = bundle.layer_visibility;
    working_scene_root.advance_to(scene_package.root.clone());

    selection.clear();
    current_markers.markers.clear();
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
    let mission_zones = LoadedMissionZones {
        zones: next_replay_document
            .as_ref()
            .map(|document| document.zones())
            .unwrap_or_default(),
        terrain_mesh: next_replay_document
            .as_ref()
            .and_then(|document| document.terrain_viewer_mesh()),
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
    let Ok((mut orbit, mut transform)) = camera_query.get_single_mut() else {
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
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{
        generation_rejection_position, prepare_reloaded_scene_bundle, preserved_replay_state,
        tracker_rejection_position,
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
            }],
            observations: Vec::new(),
            rejected_observations: Vec::new(),
            generation_rejections: Vec::new(),
            metrics: FrameMetrics::default(),
            mapping_state: None,
            localization_state: None,
            inspection_events: Vec::new(),
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
    fn reload_bundle_replaces_scene_replay_and_zones_together() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!("tracker-viewer-reload-bundle-{suffix}"));
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
