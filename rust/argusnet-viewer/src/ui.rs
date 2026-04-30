use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_extras::{Column, TableBuilder};
use egui_plot::{HLine, Line, Plot, PlotPoints, Points, Polygon, VLine};

use crate::mission_zones::{zone_color_rgba, ProjectedZoneBadges};
use crate::replay::{EgressProgress, MarkerKind, ReplayState, RuntimeMarker};
use crate::schema::ScenePackage;
use crate::state::{
    CurrentFrameMetrics, LayerVisibilityState, LoadedMissionZones, MapLayerPreset,
    MissionOverlaySettings, ReconstructionCloud, RuntimeOverlayVisibility, SelectionState,
    SimulationRunner, ViewMode, ViewerMode, ViewerUiState, ZoneFocus, ZoneOverlapModel,
};

const MISSION_MODES: &[&str] = &["scan_map_inspect", "target_tracking"];

/// Distance (m) below which an egress drone is counted as "returned to home".
///
/// Must match the simulator's egress→complete transition threshold in
/// `src/argusnet/simulation/sim.py` so the HUD doesn't claim full RTH while
/// the mission is still in egress.
pub const EGRESS_ARRIVAL_THRESHOLD_M: f32 = 15.0;
const MAP_PRESETS: &[&str] = &[
    "small",
    "medium",
    "large",
    "xlarge",
    "regional",
    "theater",
    "operational",
];
const TERRAIN_PRESETS: &[&str] = &[
    "default",
    "alpine",
    "coastal",
    "urban_flat",
    "desert_canyon",
    "rolling_highlands",
    "lake_district",
    "jungle_canopy",
    "arctic_tundra",
    "military_compound",
    "river_valley",
    "mountain_pass",
];
const PLATFORM_PRESETS: &[&str] = &["baseline", "wide_area"];
const MOTION_PRESETS: &[&str] = &["sinusoid", "racetrack", "waypoint_patrol", "mixed"];
const DRONE_MODES: &[&str] = &["inspect", "search", "mixed"];

/// Stale step threshold: nodes/tracks exceeding this many missed updates are
/// flagged in the alert panel. Mirrors `TrackerConfig.max_stale_steps` from
/// `rust/argusnet-core/src/lib.rs`; keep in sync if that default changes.
const STALE_THRESHOLD: u32 = 5;
/// Node health fraction below which a node is flagged in the alert panel.
/// Health is computed in `argusnet_grpc.py` as `1.0 - dropout_probability * 0.5`.
const HEALTH_ALERT_THRESHOLD: f32 = 0.3;

const TOP_STATUS_HEIGHT: f32 = 52.0;
const BOTTOM_TIMELINE_HEIGHT: f32 = 126.0;
const DRAWER_WIDTH: f32 = 390.0;

#[allow(clippy::too_many_arguments)]
pub fn viewer_ui_system(
    mut contexts: EguiContexts,
    scene_package: Res<ScenePackage>,
    mut replay_state: ResMut<ReplayState>,
    mut layer_visibility: ResMut<LayerVisibilityState>,
    mut runtime_visibility: ResMut<RuntimeOverlayVisibility>,
    mut mission_overlay: ResMut<MissionOverlaySettings>,
    mut view_mode: ResMut<ViewMode>,
    selection: Res<SelectionState>,
    frame_metrics: Res<CurrentFrameMetrics>,
    mission_zones: Res<LoadedMissionZones>,
    mut zone_overlap_model: ResMut<ZoneOverlapModel>,
    mut sim_runner: ResMut<SimulationRunner>,
    projected_badges: Res<ProjectedZoneBadges>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut ui_state: ResMut<ViewerUiState>,
    reconstruction: Res<ReconstructionCloud>,
) {
    let context = contexts.ctx_mut();
    let supports_hot_swap = scene_package.is_synthetic_source();
    ui_state.focused_entity_label = selection.selected_label.clone();
    configure_operator_style(context);

    egui::TopBottomPanel::top("operator_status_bar")
        .exact_height(TOP_STATUS_HEIGHT)
        .show(context, |ui| {
            top_status_bar(
                ui,
                &scene_package,
                &mut replay_state,
                &mut view_mode,
                &mut ui_state,
            );
        });

    egui::TopBottomPanel::bottom("operator_timeline")
        .exact_height(BOTTOM_TIMELINE_HEIGHT)
        .show(context, |ui| {
            section_operator_timeline(ui, &mut replay_state, &mut ui_state);
        });

    if ui_state.drawer_open {
        egui::SidePanel::right("operator_drawer")
            .resizable(true)
            .default_width(DRAWER_WIDTH)
            .width_range(320.0..=540.0)
            .show(context, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| match ui_state.mode {
                        ViewerMode::Mission => drawer_mission_content(
                            ui,
                            &scene_package,
                            &replay_state,
                            &mission_zones,
                            &reconstruction,
                            &mut view_mode,
                            &mut runtime_visibility,
                            &mut mission_overlay,
                            &selection,
                            &mut layer_visibility,
                            &mut zone_overlap_model,
                            &mut ui_state,
                        ),
                        ViewerMode::Replay => {
                            drawer_replay_content(ui, &mut replay_state, &mut ui_state)
                        }
                        ViewerMode::Diagnostics => drawer_diagnostics_content(
                            ui,
                            &replay_state,
                            &frame_metrics,
                            &mission_zones,
                            &diagnostics,
                        ),
                        ViewerMode::Scenario => drawer_scenario_content(
                            ui,
                            &scene_package,
                            &replay_state,
                            &mission_zones,
                            &mut zone_overlap_model,
                            supports_hot_swap,
                            &mut sim_runner,
                        ),
                    });
            });
    }

    draw_zone_badges(context, &projected_badges);
    draw_mission_phase_hud(context, &replay_state);
}

fn configure_operator_style(context: &egui::Context) {
    let mut style = (*context.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(9.0, 4.0);
    style.visuals.widgets.active.rounding = 4.0.into();
    style.visuals.widgets.hovered.rounding = 4.0.into();
    style.visuals.widgets.inactive.rounding = 4.0.into();
    context.set_style(style);
}

fn top_status_bar(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &mut ReplayState,
    view_mode: &mut ViewMode,
    ui_state: &mut ViewerUiState,
) {
    ui.horizontal(|ui| {
        ui.strong("ArgusNet");
        ui.label(&scene_package.manifest.scene_id);
        ui.separator();

        if let Some((phase, color, detail)) = mission_status_summary(replay_state) {
            status_chip(ui, phase, color);
            ui.label(detail);
        } else {
            status_chip(ui, "NO REPLAY", egui::Color32::GRAY);
        }

        ui.separator();
        playback_buttons(ui, replay_state);

        if replay_state.frame_count() > 0 {
            let cur_s = replay_state.current_timestamp_s();
            let total_s = replay_state
                .document
                .as_ref()
                .and_then(|doc| doc.frames.last())
                .map(|frame| frame.timestamp_s)
                .unwrap_or(0.0);
            ui.monospace(format_time_range(cur_s, total_s));
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let drawer_label = if ui_state.drawer_open {
                "Hide Panel"
            } else {
                "Show Panel"
            };
            if ui.button(drawer_label).clicked() {
                ui_state.drawer_open = !ui_state.drawer_open;
            }
            mode_button(ui, ui_state, ViewerMode::Scenario, "Scenario");
            mode_button(ui, ui_state, ViewerMode::Diagnostics, "Diagnostics");
            mode_button(ui, ui_state, ViewerMode::Replay, "Replay");
            mode_button(ui, ui_state, ViewerMode::Mission, "Mission");
            ui.separator();
            if ui
                .selectable_label(*view_mode == ViewMode::Split, "Split")
                .clicked()
            {
                *view_mode = ViewMode::Split;
            }
            if ui
                .selectable_label(*view_mode == ViewMode::ScanMap, "Scan")
                .clicked()
            {
                *view_mode = ViewMode::ScanMap;
            }
            if ui
                .selectable_label(*view_mode == ViewMode::RealWorld, "Real")
                .clicked()
            {
                *view_mode = ViewMode::RealWorld;
            }
        });
    });
}

fn mode_button(ui: &mut egui::Ui, ui_state: &mut ViewerUiState, mode: ViewerMode, label: &str) {
    if ui
        .selectable_label(ui_state.mode == mode, label)
        .on_hover_text(format!("{label} mode"))
        .clicked()
    {
        ui_state.mode = mode;
        ui_state.drawer_open = true;
    }
}

fn playback_buttons(ui: &mut egui::Ui, replay_state: &mut ReplayState) {
    let has_replay = replay_state.frame_count() > 0;
    ui.add_enabled_ui(has_replay, |ui| {
        if ui.small_button("|<").on_hover_text("First frame").clicked() {
            replay_state.playing = false;
            replay_state.step_to(0);
        }
        if ui
            .small_button("<")
            .on_hover_text("Previous frame")
            .clicked()
        {
            replay_state.playing = false;
            replay_state.step_to(replay_state.frame_index.saturating_sub(1));
        }
        let play_label = if replay_state.playing {
            "Pause"
        } else {
            "Play"
        };
        if ui.small_button(play_label).clicked() {
            replay_state.playing = !replay_state.playing;
        }
        if ui.small_button(">").on_hover_text("Next frame").clicked() {
            replay_state.playing = false;
            replay_state.step_to(replay_state.frame_index + 1);
        }
        if ui.small_button(">|").on_hover_text("Last frame").clicked() {
            replay_state.playing = false;
            replay_state.step_to(replay_state.frame_count().saturating_sub(1));
        }
    });
}

fn status_chip(ui: &mut egui::Ui, label: &str, color: egui::Color32) {
    ui.colored_label(color, egui::RichText::new(label).strong());
}

fn mission_status_summary(
    replay_state: &ReplayState,
) -> Option<(&'static str, egui::Color32, String)> {
    let frame = replay_state.current_frame()?;
    let mission = frame.scan_mission_state.as_ref()?;
    let (label, color) = phase_color_pair(mission.phase.as_str());
    let detail = match mission.phase.as_str() {
        "scanning" => format!(
            "Coverage {:.0}% / {:.0}%",
            mission.scan_coverage_fraction * 100.0,
            mission.scan_coverage_threshold * 100.0
        ),
        "inspecting" => format!(
            "POI {}/{} complete",
            mission.completed_poi_count,
            mission.total_poi_count.max(1)
        ),
        "egress" => format!(
            "RTH {}/{}",
            egress_arrived_count(&mission.egress_progress),
            mission.egress_progress.len().max(1)
        ),
        _ => format!("Coverage {:.0}%", mission.scan_coverage_fraction * 100.0),
    };
    Some((label, color, detail))
}

fn format_time_range(current_s: f32, total_s: f32) -> String {
    format!(
        "{:.0}:{:04.1} / {:.0}:{:04.1}",
        (current_s / 60.0).floor(),
        current_s % 60.0,
        (total_s / 60.0).floor(),
        total_s % 60.0
    )
}

#[allow(clippy::too_many_arguments)]
fn drawer_mission_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    reconstruction: &ReconstructionCloud,
    view_mode: &mut ViewMode,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
    selection: &SelectionState,
    layer_visibility: &mut LayerVisibilityState,
    zone_overlap_model: &mut ZoneOverlapModel,
    ui_state: &mut ViewerUiState,
) {
    ui.heading("Mission");
    ui.label("Operator view");
    ui.separator();

    mission_layer_presets(ui, ui_state, runtime_visibility, mission_overlay);
    section_mission_progress(ui, replay_state, view_mode);
    ui.separator();
    tab_mission_content(
        ui,
        scene_package,
        replay_state,
        mission_zones,
        reconstruction,
    );
    ui.separator();
    section_safety_alerts(ui, replay_state, mission_zones);
    section_selection(ui, selection);
    ui.separator();
    section_runtime_overlays(ui, runtime_visibility, mission_overlay);
    ui.separator();
    section_layers(ui, scene_package, layer_visibility);
    ui.separator();
    section_mission_zones(ui, mission_zones, zone_overlap_model);
}

fn mission_layer_presets(
    ui: &mut egui::Ui,
    ui_state: &mut ViewerUiState,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
) {
    ui.horizontal(|ui| {
        ui.label("Layer preset:");
        for (preset, label) in &[
            (MapLayerPreset::Operator, "Operator"),
            (MapLayerPreset::Reconstruction, "Reconstruction"),
            (MapLayerPreset::Safety, "Safety"),
        ] {
            if ui
                .selectable_label(ui_state.map_layer_preset == *preset, *label)
                .clicked()
            {
                ui_state.map_layer_preset = *preset;
                apply_layer_preset(*preset, runtime_visibility, mission_overlay);
            }
        }
    });
}

fn apply_layer_preset(
    preset: MapLayerPreset,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
) {
    match preset {
        MapLayerPreset::Operator => {
            runtime_visibility.nodes = true;
            runtime_visibility.tracks = true;
            runtime_visibility.truths = true;
            runtime_visibility.zones = true;
            runtime_visibility.radar_rings = true;
            runtime_visibility.observations = false;
            runtime_visibility.rejection_markers = false;
            mission_overlay.show_scan_grid = true;
            mission_overlay.show_poi_markers = true;
            mission_overlay.show_reconstruction = true;
            mission_overlay.show_coord_frame = true;
        }
        MapLayerPreset::Reconstruction => {
            runtime_visibility.nodes = true;
            runtime_visibility.tracks = false;
            runtime_visibility.truths = false;
            runtime_visibility.zones = true;
            runtime_visibility.radar_rings = false;
            runtime_visibility.observations = false;
            runtime_visibility.rejection_markers = false;
            mission_overlay.show_scan_grid = true;
            mission_overlay.show_poi_markers = true;
            mission_overlay.show_reconstruction = true;
            mission_overlay.show_coord_frame = true;
        }
        MapLayerPreset::Safety => {
            runtime_visibility.nodes = true;
            runtime_visibility.tracks = true;
            runtime_visibility.truths = true;
            runtime_visibility.zones = true;
            runtime_visibility.radar_rings = true;
            runtime_visibility.observations = true;
            runtime_visibility.rejection_markers = true;
            mission_overlay.show_scan_grid = false;
            mission_overlay.show_poi_markers = true;
            mission_overlay.show_reconstruction = false;
            mission_overlay.show_coord_frame = false;
        }
    }
}

fn drawer_replay_content(
    ui: &mut egui::Ui,
    replay_state: &mut ReplayState,
    ui_state: &mut ViewerUiState,
) {
    ui.heading("Replay");
    ui.label("Timeline and frame events");
    ui.separator();
    section_operator_timeline(ui, replay_state, ui_state);
    ui.separator();
    jump_to_event_controls(ui, replay_state, ui_state);
    section_inspection_events(ui, replay_state);
    section_frame_events(ui, replay_state);
}

fn drawer_diagnostics_content(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    frame_metrics: &CurrentFrameMetrics,
    mission_zones: &LoadedMissionZones,
    diagnostics: &bevy::diagnostic::DiagnosticsStore,
) {
    ui.heading("Diagnostics");
    ui.label("Current-frame health, tracking, and event tables");
    ui.separator();
    section_performance(ui, replay_state, diagnostics);
    ui.separator();
    section_tracking_metrics(ui, frame_metrics);
    section_mapping_status(ui, replay_state);
    section_localization_status(ui, replay_state);
    diagnostic_node_table(ui, replay_state);
    diagnostic_track_table(ui, replay_state, frame_metrics);
    diagnostic_event_table(ui, replay_state);
    section_safety_alerts(ui, replay_state, mission_zones);
}

fn drawer_scenario_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    zone_overlap_model: &mut ZoneOverlapModel,
    supports_hot_swap: bool,
    sim_runner: &mut SimulationRunner,
) {
    ui.heading("Scenario");
    ui.label("Synthetic mission setup and scene reload");
    ui.separator();
    section_scenario(ui, supports_hot_swap, sim_runner);
    ui.separator();
    section_scene_header(ui, scene_package, mission_zones, replay_state);
    ui.separator();
    section_mission_zones(ui, mission_zones, zone_overlap_model);
    section_keyboard_shortcuts(ui);
}

fn section_operator_timeline(
    ui: &mut egui::Ui,
    replay_state: &mut ReplayState,
    ui_state: &mut ViewerUiState,
) {
    let has_replay = replay_state.frame_count() > 0;
    if !has_replay {
        ui.label("No replay packaged with this scene.");
        return;
    }

    let total_frames = replay_state.frame_count();
    let current_frame = replay_state.frame_index;
    ui.horizontal(|ui| {
        ui.strong("Timeline");
        ui.label(format!("frame {} / {}", current_frame + 1, total_frames));
        ui.separator();
        ui.label(format!("speed {:.2}x", replay_state.playback_speed));
        ui.add(
            egui::Slider::new(&mut replay_state.playback_speed, 0.25..=16.0)
                .show_value(false)
                .step_by(0.25)
                .clamping(egui::SliderClamping::Always),
        );
    });

    let mut frame_value = current_frame as u32;
    let max_frame = total_frames.saturating_sub(1) as u32;
    if ui
        .add(
            egui::Slider::new(&mut frame_value, 0..=max_frame)
                .text("Frame")
                .clamping(egui::SliderClamping::Always),
        )
        .changed()
    {
        replay_state.playing = false;
        replay_state.step_to(frame_value as usize);
        ui_state.selected_timeline_frame = Some(frame_value as usize);
    }

    if let Some(document) = &replay_state.document {
        let event_points: PlotPoints = document
            .frames
            .iter()
            .enumerate()
            .map(|(i, frame)| [i as f64, frame_event_count(frame) as f64])
            .collect();
        let obs_points: PlotPoints = document
            .frames
            .iter()
            .enumerate()
            .map(|(i, frame)| [i as f64, frame.metrics.observation_count as f64])
            .collect();
        let phase_transitions = phase_transitions(document);
        Plot::new("operator_timeline_plot")
            .height(56.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, true])
            .label_formatter(|_, _| String::new())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(obs_points)
                        .color(egui::Color32::from_rgb(95, 150, 220))
                        .width(1.5),
                );
                plot_ui.line(
                    Line::new(event_points)
                        .color(egui::Color32::from_rgb(245, 180, 70))
                        .width(1.5),
                );
                for (frame_idx, phase) in &phase_transitions {
                    plot_ui.vline(
                        VLine::new(*frame_idx as f64)
                            .color(phase_vline_color(phase.as_str()))
                            .width(2.0),
                    );
                }
                plot_ui.vline(
                    VLine::new(replay_state.frame_index as f64)
                        .color(egui::Color32::WHITE)
                        .width(2.0),
                );
            });
    }
}

fn jump_to_event_controls(
    ui: &mut egui::Ui,
    replay_state: &mut ReplayState,
    ui_state: &mut ViewerUiState,
) {
    let event_frames = event_frame_indices(replay_state);
    if event_frames.is_empty() {
        ui.label("No frame events in this replay.");
        return;
    }

    ui.horizontal(|ui| {
        if ui.button("Previous Event").clicked() {
            if let Some(frame_idx) = event_frames
                .iter()
                .rev()
                .copied()
                .find(|idx| *idx < replay_state.frame_index)
                .or_else(|| event_frames.last().copied())
            {
                replay_state.playing = false;
                replay_state.step_to(frame_idx);
                ui_state.selected_timeline_frame = Some(frame_idx);
            }
        }
        if ui.button("Next Event").clicked() {
            if let Some(frame_idx) = event_frames
                .iter()
                .copied()
                .find(|idx| *idx > replay_state.frame_index)
                .or_else(|| event_frames.first().copied())
            {
                replay_state.playing = false;
                replay_state.step_to(frame_idx);
                ui_state.selected_timeline_frame = Some(frame_idx);
            }
        }
        ui.label(format!("{} event frames", event_frames.len()));
    });
}

fn event_frame_indices(replay_state: &ReplayState) -> Vec<usize> {
    replay_state
        .document
        .as_ref()
        .map(|document| {
            document
                .frames
                .iter()
                .enumerate()
                .filter_map(|(idx, frame)| (frame_event_count(frame) > 0).then_some(idx))
                .collect()
        })
        .unwrap_or_default()
}

fn phase_transitions(document: &crate::replay::ReplayDocument) -> Vec<(usize, String)> {
    let mut transitions = Vec::new();
    let mut last_phase = String::new();
    for (idx, frame) in document.frames.iter().enumerate() {
        if let Some(mission) = &frame.scan_mission_state {
            if mission.phase != last_phase {
                if idx > 0 {
                    transitions.push((idx, mission.phase.clone()));
                }
                last_phase = mission.phase.clone();
            }
        }
    }
    transitions
}

fn frame_event_count(frame: &crate::replay::ReplayFrame) -> usize {
    frame.generation_rejections.len()
        + frame.rejected_observations.len()
        + frame.deconfliction_events.len()
        + frame.inspection_events.len()
}

fn diagnostic_node_table(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    ui.heading(format!("Nodes ({})", frame.nodes.len()));
    if frame.nodes.is_empty() {
        ui.label("No nodes in this frame.");
        return;
    }
    TableBuilder::new(ui)
        .striped(true)
        .column(Column::auto())
        .column(Column::remainder())
        .column(Column::auto())
        .column(Column::auto())
        .column(Column::auto())
        .header(18.0, |mut header| {
            header.col(|ui| {
                ui.strong("ID");
            });
            header.col(|ui| {
                ui.strong("Sensor");
            });
            header.col(|ui| {
                ui.strong("Health");
            });
            header.col(|ui| {
                ui.strong("Alt");
            });
            header.col(|ui| {
                ui.strong("Mobile");
            });
        })
        .body(|mut body| {
            for node in &frame.nodes {
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        ui.label(&node.node_id);
                    });
                    row.col(|ui| {
                        ui.label(&node.sensor_type);
                    });
                    row.col(|ui| {
                        ui.colored_label(
                            health_color_egui(node.health),
                            format!("{:.0}%", node.health * 100.0),
                        );
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.0}m", node.position[2]));
                    });
                    row.col(|ui| {
                        ui.label(if node.is_mobile { "Yes" } else { "No" });
                    });
                });
            }
        });
}

fn diagnostic_track_table(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    frame_metrics: &CurrentFrameMetrics,
) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    ui.heading(format!("Tracks ({})", frame.tracks.len()));
    if frame.tracks.is_empty() {
        ui.label("No active tracks in this frame.");
        return;
    }
    TableBuilder::new(ui)
        .striped(true)
        .column(Column::remainder())
        .column(Column::auto())
        .column(Column::auto())
        .column(Column::auto())
        .header(18.0, |mut header| {
            header.col(|ui| {
                ui.strong("ID");
            });
            header.col(|ui| {
                ui.strong("Error");
            });
            header.col(|ui| {
                ui.strong("Updates");
            });
            header.col(|ui| {
                ui.strong("Stale");
            });
        })
        .body(|mut body| {
            for track in &frame.tracks {
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        ui.label(&track.track_id);
                    });
                    row.col(|ui| {
                        let error = frame_metrics
                            .metrics
                            .as_ref()
                            .and_then(|metrics| metrics.track_errors_m.get(&track.track_id))
                            .copied();
                        if let Some(error) = error {
                            ui.label(format!("{error:.1}m"));
                        } else {
                            ui.label("-");
                        }
                    });
                    row.col(|ui| {
                        ui.label(track.update_count.to_string());
                    });
                    row.col(|ui| {
                        let color = if track.stale_steps > STALE_THRESHOLD {
                            egui::Color32::RED
                        } else if track.stale_steps > 0 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::GREEN
                        };
                        ui.colored_label(color, track.stale_steps.to_string());
                    });
                });
            }
        });
}

fn diagnostic_event_table(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let rows = [
        ("Generation rejections", frame.generation_rejections.len()),
        ("Tracker rejections", frame.rejected_observations.len()),
        ("Deconfliction yields", frame.deconfliction_events.len()),
        ("Inspection events", frame.inspection_events.len()),
        ("Observations", frame.observations.len()),
    ];
    ui.heading("Frame Events");
    TableBuilder::new(ui)
        .striped(true)
        .column(Column::remainder())
        .column(Column::auto())
        .body(|mut body| {
            for (label, count) in rows {
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        ui.label(label);
                    });
                    row.col(|ui| {
                        ui.strong(count.to_string());
                    });
                });
            }
        });
}

// ---------------------------------------------------------------------------
// Tab: Mission — 2D top-down map + per-drone cards
// ---------------------------------------------------------------------------

fn tab_mission_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    reconstruction: &ReconstructionCloud,
) {
    // Scene header (compact)
    ui.horizontal(|ui| {
        ui.strong(&scene_package.manifest.scene_id);
        if let Some(frame) = replay_state.current_frame() {
            if let Some(ref ms) = frame.scan_mission_state {
                let pct = (ms.scan_coverage_fraction * 100.0).min(100.0);
                let (phase_label, phase_color) = phase_color_pair(ms.phase.as_str());
                ui.colored_label(phase_color, format!("\u{25cf} {} {:.0}%", phase_label, pct));
            }
        }
    });
    ui.separator();

    // ── 2D Top-down map ──────────────────────────────────────────────────────
    let bounds = &scene_package.environment.bounds_xy_m;
    let bx_min = bounds.x_min_m as f64;
    let bx_max = bounds.x_max_m as f64;
    let by_min = bounds.y_min_m as f64;
    let by_max = bounds.y_max_m as f64;
    let bx_mid = (bx_min + bx_max) * 0.5;
    let by_mid = (by_min + by_max) * 0.5;

    let map_height = (ui.available_height() * 0.55).max(200.0);

    Plot::new("map2d")
        .data_aspect(1.0)
        .allow_zoom([true, true])
        .allow_drag(true)
        .allow_scroll(true)
        .show_axes([false, false])
        .show_grid(false)
        .include_x(bx_min)
        .include_x(bx_max)
        .include_y(by_min)
        .include_y(by_max)
        .center_x_axis(false)
        .center_y_axis(false)
        .height(map_height)
        .show(ui, |plot_ui| {
            // Bounds outline
            let border = PlotPoints::new(vec![
                [bx_min, by_min],
                [bx_max, by_min],
                [bx_max, by_max],
                [bx_min, by_max],
                [bx_min, by_min],
            ]);
            plot_ui.line(
                Line::new(border)
                    .color(egui::Color32::from_rgba_unmultiplied(100, 120, 160, 180))
                    .width(1.5),
            );

            // Mission zones
            for zone in &mission_zones.zones {
                let rgba = zone_color_rgba(&zone.zone_type, 0.7);
                let color = egui::Color32::from_rgba_unmultiplied(
                    (rgba[0] * 255.0) as u8,
                    (rgba[1] * 255.0) as u8,
                    (rgba[2] * 255.0) as u8,
                    (rgba[3] * 200.0) as u8,
                );
                let cx = zone.center[0] as f64;
                let cy = zone.center[1] as f64;
                let r = zone.radius_m as f64;
                let circle: Vec<[f64; 2]> = (0..=48)
                    .map(|i| {
                        let a = i as f64 * std::f64::consts::TAU / 48.0;
                        [cx + r * a.cos(), cy + r * a.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new(PlotPoints::new(circle)).color(color).width(1.5));
                // Zone label dot at center
                plot_ui.points(
                    Points::new(PlotPoints::new(vec![[cx, cy]]))
                        .color(color)
                        .radius(3.0),
                );
            }

            // Accumulated scan cells (subsampled for performance)
            if !reconstruction.points.is_empty() {
                let scan_pts: Vec<[f64; 2]> = reconstruction
                    .points
                    .iter()
                    .step_by(8)
                    .map(|p| [p[0] as f64, p[1] as f64])
                    .collect();
                plot_ui.points(
                    Points::new(PlotPoints::new(scan_pts))
                        .color(egui::Color32::from_rgba_unmultiplied(220, 200, 40, 120))
                        .radius(1.5),
                );
            }

            // Drone + node positions
            if let Some(frame) = replay_state.current_frame() {
                for node in &frame.nodes {
                    let nx = node.position[0] as f64;
                    let ny = node.position[1] as f64;
                    let color = if node.is_mobile {
                        health_color_egui(node.health)
                    } else {
                        egui::Color32::from_rgb(160, 160, 220)
                    };

                    if node.is_mobile {
                        // Arrow triangle pointing in velocity direction
                        let vel = node.velocity;
                        let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
                        let heading = if speed > 0.1 {
                            (vel[1] as f64).atan2(vel[0] as f64)
                        } else {
                            std::f64::consts::FRAC_PI_2
                        };
                        let tip_len = (bx_max - bx_min) * 0.018;
                        let base_w = tip_len * 0.5;
                        let tip = [nx + tip_len * heading.cos(), ny + tip_len * heading.sin()];
                        let left = [
                            nx + base_w * (heading + std::f64::consts::FRAC_PI_2 + 0.5).cos(),
                            ny + base_w * (heading + std::f64::consts::FRAC_PI_2 + 0.5).sin(),
                        ];
                        let right = [
                            nx + base_w * (heading - std::f64::consts::FRAC_PI_2 - 0.5).cos(),
                            ny + base_w * (heading - std::f64::consts::FRAC_PI_2 - 0.5).sin(),
                        ];
                        plot_ui.polygon(
                            Polygon::new(PlotPoints::new(vec![tip, left, right])).fill_color(
                                egui::Color32::from_rgba_unmultiplied(
                                    color.r(),
                                    color.g(),
                                    color.b(),
                                    200,
                                ),
                            ),
                        );
                    } else {
                        // Static sensor: small square
                        let s = (bx_max - bx_min) * 0.008;
                        plot_ui.polygon(
                            Polygon::new(PlotPoints::new(vec![
                                [nx - s, ny - s],
                                [nx + s, ny - s],
                                [nx + s, ny + s],
                                [nx - s, ny + s],
                            ]))
                            .fill_color(
                                egui::Color32::from_rgba_unmultiplied(
                                    color.r(),
                                    color.g(),
                                    color.b(),
                                    160,
                                ),
                            ),
                        );
                    }
                }

                // POI markers
                if let Some(ref ms) = frame.scan_mission_state {
                    for poi in &ms.poi_statuses {
                        let Some(pos) = poi.position else { continue };
                        let px = pos[0] as f64;
                        let py = pos[1] as f64;
                        let r = (bx_max - bx_min) * 0.012;
                        let color = match poi.status.as_str() {
                            "complete" => egui::Color32::from_rgb(60, 210, 80),
                            "active" => egui::Color32::from_rgb(255, 200, 40),
                            _ => egui::Color32::from_rgb(160, 160, 160),
                        };
                        let circle: Vec<[f64; 2]> = (0..=24)
                            .map(|i| {
                                let a = i as f64 * std::f64::consts::TAU / 24.0;
                                [px + r * a.cos(), py + r * a.sin()]
                            })
                            .collect();
                        plot_ui.line(Line::new(PlotPoints::new(circle)).color(color).width(2.0));
                    }
                }

                // Trails: last 16 positions per mobile node
                for node in &frame.nodes {
                    if !node.is_mobile {
                        continue;
                    }
                    let marker = RuntimeMarker {
                        kind: MarkerKind::Node,
                        label: node.node_id.clone(),
                        position: Vec3::ZERO,
                        velocity: Vec3::ZERO,
                        is_mobile: Some(true),
                    };
                    let trail_3d = replay_state.trail_points(&marker, 16);
                    if trail_3d.len() >= 2 {
                        let trail_xy: Vec<[f64; 2]> =
                            trail_3d.iter().map(|p| [p.x as f64, p.y as f64]).collect();
                        plot_ui.line(
                            Line::new(PlotPoints::new(trail_xy))
                                .color(egui::Color32::from_rgba_unmultiplied(180, 180, 80, 80))
                                .width(1.0),
                        );
                    }
                }
            }

            // Crosshair at scene center (faint)
            plot_ui.vline(
                VLine::new(bx_mid)
                    .color(egui::Color32::from_rgba_unmultiplied(120, 120, 120, 60)),
            );
            plot_ui.hline(
                HLine::new(by_mid)
                    .color(egui::Color32::from_rgba_unmultiplied(120, 120, 120, 60)),
            );
        });

    ui.separator();

    // ── Per-drone status cards ────────────────────────────────────────────────
    if let Some(frame) = replay_state.current_frame() {
        let mobile_nodes: Vec<_> = frame.nodes.iter().filter(|n| n.is_mobile).collect();
        if !mobile_nodes.is_empty() {
            ui.label(egui::RichText::new("Drones").strong());
            ui.add_space(2.0);

            let mission = frame.scan_mission_state.as_ref();
            let coordinator_id = mission.and_then(|ms| ms.coordinator_drone_id.as_deref());
            let phase_str = mission.map(|ms| ms.phase.as_str()).unwrap_or("");
            let global_coverage = mission.map(|ms| ms.scan_coverage_fraction).unwrap_or(0.0);

            let card_cols = if mobile_nodes.len() <= 2 {
                mobile_nodes.len()
            } else {
                2
            };
            egui::Grid::new("drone_cards")
                .num_columns(card_cols)
                .spacing([6.0, 6.0])
                .show(ui, |ui| {
                    for (i, node) in mobile_nodes.iter().enumerate() {
                        egui::Frame::group(ui.style())
                            .inner_margin(egui::Margin::same(6.0))
                            .show(ui, |ui| {
                                ui.set_min_width(140.0);

                                // Header: ID + coordinator badge
                                ui.horizontal(|ui| {
                                    ui.strong(&node.node_id);
                                    if Some(node.node_id.as_str()) == coordinator_id {
                                        ui.label(
                                            egui::RichText::new("[COORD]")
                                                .color(egui::Color32::from_rgb(180, 220, 255))
                                                .small(),
                                        );
                                    }
                                });

                                // Phase dot + label
                                let (phase_label, phase_color) = phase_color_pair(phase_str);
                                ui.horizontal(|ui| {
                                    ui.colored_label(phase_color, "\u{25cf}");
                                    ui.colored_label(
                                        phase_color,
                                        egui::RichText::new(phase_label).small(),
                                    );
                                });

                                // Health bar
                                let health_pct = node.health.clamp(0.0, 1.0);
                                let hcolor = health_color_egui(node.health);
                                ui.add(
                                    egui::ProgressBar::new(health_pct)
                                        .text(format!("{:.0}%", health_pct * 100.0))
                                        .fill(hcolor)
                                        .desired_height(10.0),
                                );

                                // Alt + coverage row
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "Alt {:.0}m",
                                            node.position[2]
                                        ))
                                        .small(),
                                    );
                                    if global_coverage > 0.0 {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "  Cov {:.0}%",
                                                global_coverage * 100.0
                                            ))
                                            .small(),
                                        );
                                    }
                                });

                                // Egress distance if applicable
                                if let Some(ms) = mission {
                                    if let Some(ep) = ms
                                        .egress_progress
                                        .iter()
                                        .find(|ep| ep.drone_id == node.node_id)
                                    {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "\u{21a9} {:.0}m to home",
                                                ep.distance_to_home_m
                                            ))
                                            .small()
                                            .color(egui::Color32::from_rgb(255, 160, 30)),
                                        );
                                    }
                                }
                            });
                        if (i + 1) % card_cols == 0 {
                            ui.end_row();
                        }
                    }
                });
        }
    }
}

// ---------------------------------------------------------------------------
// Tab: 3D Scene — overlays, layers, selection, alerts
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn tab_scene_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    view_mode: &mut ViewMode,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
    selection: &SelectionState,
    layer_visibility: &mut LayerVisibilityState,
    zone_overlap_model: &mut ZoneOverlapModel,
    supports_hot_swap: bool,
    sim_runner: &mut SimulationRunner,
    diagnostics: &bevy::diagnostic::DiagnosticsStore,
) {
    section_scene_header(ui, scene_package, mission_zones, replay_state);
    ui.separator();

    section_mission_progress(ui, replay_state, view_mode);
    ui.separator();

    section_scenario(ui, supports_hot_swap, sim_runner);
    ui.separator();

    section_playback_timeline(ui, replay_state);
    ui.separator();

    section_performance(ui, replay_state, diagnostics);
    ui.separator();

    section_safety_alerts(ui, replay_state, mission_zones);

    section_selection(ui, selection);
    ui.separator();

    section_layers(ui, scene_package, layer_visibility);
    ui.separator();

    section_runtime_overlays(ui, runtime_visibility, mission_overlay);
    ui.separator();

    section_mission_zones(ui, mission_zones, zone_overlap_model);

    section_keyboard_shortcuts(ui);
}

// ---------------------------------------------------------------------------
// Tab: Tracks — metrics, node table, track table
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn tab_tracks_content(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    frame_metrics: &CurrentFrameMetrics,
) {
    section_tracking_metrics(ui, frame_metrics);
    ui.separator();

    section_mapping_status(ui, replay_state);
    section_localization_status(ui, replay_state);

    section_node_summary(ui, replay_state);
    section_track_summary(ui, replay_state, frame_metrics);
}

// ---------------------------------------------------------------------------
// Tab: Events — frame events, inspection events, deconfliction
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn tab_events_content(ui: &mut egui::Ui, replay_state: &ReplayState) {
    section_inspection_events(ui, replay_state);
    section_frame_events(ui, replay_state);
}

// ---------------------------------------------------------------------------
// Section: Scene Header + Terrain Info
// ---------------------------------------------------------------------------

fn section_scene_header(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    mission_zones: &LoadedMissionZones,
    replay_state: &ReplayState,
) {
    ui.heading("ArgusNet Viewer");
    ui.label(format!("Scene: {}", scene_package.manifest.scene_id));
    ui.label(format!("CRS: {}", scene_package.manifest.source_crs_id));

    // Terrain info
    let bounds = &scene_package.environment.bounds_xy_m;
    ui.label(format!(
        "Bounds: X [{:.0}, {:.0}] m  Y [{:.0}, {:.0}] m",
        bounds.x_min_m, bounds.x_max_m, bounds.y_min_m, bounds.y_max_m
    ));
    if let Some(terrain) = &scene_package.environment.terrain_summary {
        let min_h = terrain.min_height_m.unwrap_or(0.0);
        let max_h = terrain.max_height_m.unwrap_or(0.0);
        ui.label(format!("Elevation: {:.1} m to {:.1} m", min_h, max_h));
    }
    if let Some(mesh) = &mission_zones.terrain_mesh {
        ui.label(format!("Terrain mesh: {}x{} grid", mesh.cols, mesh.rows));
    }

    // Inline mission phase + coverage summary
    if let Some(frame) = replay_state.current_frame() {
        if let Some(ref ms) = frame.scan_mission_state {
            let pct = (ms.scan_coverage_fraction * 100.0).min(100.0);
            let (phase_label, phase_color) = phase_color_pair(ms.phase.as_str());
            ui.horizontal(|ui| {
                ui.colored_label(phase_color, format!("\u{25cf} {}", phase_label));
                ui.label(format!("  {:.0}% covered", pct));
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Section: Scenario Controls
// ---------------------------------------------------------------------------

fn section_scenario(ui: &mut egui::Ui, supports_hot_swap: bool, sim_runner: &mut SimulationRunner) {
    ui.collapsing("Scenario", |ui| {
        if !supports_hot_swap {
            ui.label("Re-simulation only for synthetic scenes.");
        }

        ui.add_enabled_ui(supports_hot_swap && !sim_runner.is_busy(), |ui| {
            combo_box(
                ui,
                "Mission Mode",
                &sim_runner.mission_mode.clone(),
                MISSION_MODES,
                |v| {
                    sim_runner.mission_mode = v.to_string();
                },
            );
            ui.separator();
            combo_box(
                ui,
                "Map",
                &sim_runner.map_preset.clone(),
                MAP_PRESETS,
                |v| {
                    sim_runner.map_preset = v.to_string();
                },
            );
            combo_box(
                ui,
                "Terrain",
                &sim_runner.terrain_preset.clone(),
                TERRAIN_PRESETS,
                |v| {
                    sim_runner.terrain_preset = v.to_string();
                },
            );
            ui.checkbox(
                &mut sim_runner.clean_terrain,
                "Clean Terrain (terrain-only blockers off)",
            );
            combo_box(
                ui,
                "Platform",
                &sim_runner.platform_preset.clone(),
                PLATFORM_PRESETS,
                |v| {
                    sim_runner.platform_preset = v.to_string();
                },
            );
            ui.add(egui::Slider::new(&mut sim_runner.drone_count, 1..=8).text("Drones"));
            ui.add(
                egui::Slider::new(&mut sim_runner.ground_stations, 1..=12).text("Ground Stations"),
            );
            ui.add(
                egui::Slider::new(&mut sim_runner.duration_s, 30.0..=600.0)
                    .text("Duration (s)")
                    .step_by(10.0),
            );
            ui.add(egui::Slider::new(&mut sim_runner.seed, 0..=999).text("Seed"));

            // Mission-mode-specific parameters
            if sim_runner.mission_mode == "scan_map_inspect" {
                ui.separator();
                ui.label("Scan / Map / Inspect");
                ui.add(
                    egui::Slider::new(&mut sim_runner.scan_coverage_threshold, 0.3..=0.95)
                        .text("Scan threshold")
                        .step_by(0.05),
                );
                ui.add(egui::Slider::new(&mut sim_runner.poi_count, 1..=10).text("POI count"));
            } else {
                // target_tracking mode — show target-centric controls
                ui.separator();
                ui.label("Target Tracking");
                combo_box(
                    ui,
                    "Target Motion",
                    &sim_runner.target_motion.clone(),
                    MOTION_PRESETS,
                    |v| {
                        sim_runner.target_motion = v.to_string();
                    },
                );
                combo_box(
                    ui,
                    "Drone Mode",
                    &sim_runner.drone_mode.clone(),
                    DRONE_MODES,
                    |v| {
                        sim_runner.drone_mode = v.to_string();
                    },
                );
                ui.add(egui::Slider::new(&mut sim_runner.target_count, 1..=8).text("Targets"));
            }
        });

        if sim_runner.is_busy() {
            ui.add(
                egui::ProgressBar::new(sim_runner.progress)
                    .text(sim_runner.status_label())
                    .animate(true),
            );
        } else if supports_hot_swap && ui.button("\u{25b6} Run Simulation").clicked() {
            sim_runner.start();
        } else if !supports_hot_swap {
            ui.add_enabled(false, egui::Button::new("\u{25b6} Run Simulation"));
        }

        if let Some(error) = &sim_runner.error {
            ui.colored_label(egui::Color32::RED, error);
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Mission Progress
// ---------------------------------------------------------------------------

fn section_mission_progress(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    view_mode: &mut ViewMode,
) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref ms) = frame.scan_mission_state else {
        return;
    };

    egui::CollapsingHeader::new("Mission Progress")
        .default_open(true)
        .show(ui, |ui| {
            // View mode toggle
            ui.horizontal(|ui| {
                ui.label("View:");
                if ui
                    .selectable_label(*view_mode == ViewMode::RealWorld, "Real")
                    .clicked()
                {
                    *view_mode = ViewMode::RealWorld;
                }
                if ui
                    .selectable_label(*view_mode == ViewMode::ScanMap, "Scan Map")
                    .clicked()
                {
                    *view_mode = ViewMode::ScanMap;
                }
                if ui
                    .selectable_label(*view_mode == ViewMode::Split, "Split")
                    .clicked()
                {
                    *view_mode = ViewMode::Split;
                }
            });
            ui.separator();

            // Map reconstruction progress
            let recon_frac = ms.scan_coverage_fraction.clamp(0.0, 1.0);
            if recon_frac > 0.0 {
                ui.add(
                    egui::ProgressBar::new(recon_frac)
                        .text(format!("Map built: {:.0}%", recon_frac * 100.0))
                        .fill(egui::Color32::from_rgb(20, 120, 180)),
                );
            }

            // Phase label
            let phase = ms.phase.as_str();
            let (phase_label, phase_color) = phase_color_pair(phase);

            if ms.phase == "complete" {
                ui.colored_label(phase_color, format!("Phase: {}", phase_label));
            } else {
                ui.horizontal(|ui| {
                    ui.label("Phase:");
                    ui.colored_label(phase_color, phase_label);
                });
            }

            if let Some(ref coord_id) = ms.coordinator_drone_id {
                ui.horizontal(|ui| {
                    ui.label("Coordinator:");
                    ui.colored_label(egui::Color32::from_rgb(180, 220, 255), coord_id.as_str());
                });
            }

            if ms.phase == "scanning" {
                let frac = ms.scan_coverage_fraction.clamp(0.0, 1.0);
                let threshold = ms.scan_coverage_threshold.clamp(0.0, 1.0);
                ui.add(
                    egui::ProgressBar::new(frac)
                        .text(format!(
                            "{:.0}% \u{2192} {:.0}%",
                            frac * 100.0,
                            threshold * 100.0
                        ))
                        .fill(egui::Color32::from_rgb(60, 100, 200)),
                );
            }

            if ms.phase == "localizing" {
                if ms.localization_estimates.is_empty() {
                    ui.label("No localization estimates yet.");
                } else {
                    for est in &ms.localization_estimates {
                        ui.label(format!(
                            "{}: {:.0}% conf \u{00b1}{:.0} m",
                            est.drone_id,
                            est.confidence * 100.0,
                            est.position_std_m
                        ));
                    }
                }
                if ms.localization_timed_out {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 180, 60),
                        "Timeout: forced convergence",
                    );
                }
            }

            if phase == "localizing" || phase == "inspecting" {
                if let Some(est) = ms.localization_estimates.first() {
                    let mean_conf = ms
                        .localization_estimates
                        .iter()
                        .map(|e| e.confidence)
                        .sum::<f32>()
                        / ms.localization_estimates.len().max(1) as f32;
                    let frame_alpha = (mean_conf * 100.0) as u8;
                    ui.horizontal(|ui| {
                        ui.label("Coord frame:");
                        ui.colored_label(
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                220,
                                30,
                                frame_alpha.max(60),
                            ),
                            format!(
                                "XYZ @ ({:.0},{:.0}) {:.0}% conf",
                                est.position_estimate[0],
                                est.position_estimate[1],
                                mean_conf * 100.0
                            ),
                        );
                    });
                }
            }

            if ms.phase == "inspecting" {
                let total = ms.total_poi_count.max(1);
                let done = ms.completed_poi_count;
                let poi_frac = done as f32 / total as f32;
                ui.add(
                    egui::ProgressBar::new(poi_frac)
                        .text(format!("{} / {} complete", done, total))
                        .fill(egui::Color32::from_rgb(200, 100, 30)),
                );

                for poi in &ms.poi_statuses {
                    let (icon, color) = match poi.status.as_str() {
                        "complete" => ("DONE", egui::Color32::from_rgb(60, 200, 80)),
                        "active" => ("ACTIVE", egui::Color32::from_rgb(255, 200, 60)),
                        _ => ("OPEN", egui::Color32::GRAY),
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(color, icon);
                        let drone_suffix = poi
                            .assigned_drone_id
                            .as_deref()
                            .map(|d| format!("  ({})", d))
                            .unwrap_or_default();
                        ui.label(format!("[{}]{}", poi.poi_id, drone_suffix));
                    });
                }
            }

            if ms.phase == "egress" {
                let total = ms.egress_progress.len().max(1);
                let arrived = egress_arrived_count(&ms.egress_progress);
                ui.label(format!("{}/{} drones RTH", arrived, total));
                for ep in &ms.egress_progress {
                    let max_dist = ms
                        .egress_progress
                        .iter()
                        .map(|e| e.distance_to_home_m)
                        .fold(1.0_f32, f32::max)
                        .max(200.0);
                    let frac = (1.0 - ep.distance_to_home_m / max_dist).clamp(0.0, 1.0);
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(format!("{}: {:.0} m", ep.drone_id, ep.distance_to_home_m))
                            .fill(egui::Color32::from_rgb(255, 160, 30)),
                    );
                }
            }

            if ms.phase != "scanning" && ms.scan_coverage_fraction > 0.0 {
                let frac = ms.scan_coverage_fraction.clamp(0.0, 1.0);
                kv_row(ui, "Scan coverage", &format!("{:.1}%", frac * 100.0));
            }
        });
}

// ---------------------------------------------------------------------------
// Section: Playback (timeline-only, controls moved to tab bar)
// ---------------------------------------------------------------------------

fn section_playback_timeline(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let has_replay = replay_state.frame_count() > 0;
    if !has_replay {
        ui.label("No replay packaged with this scene.");
        return;
    }

    ui.heading("Timeline");
    let total_frames = replay_state.frame_count();
    let duration_s = replay_state
        .document
        .as_ref()
        .and_then(|d| d.frames.last())
        .map(|f| f.timestamp_s)
        .unwrap_or(0.0);
    ui.label(format!("{} frames · {:.1}s", total_frames, duration_s));

    // Frame scrub slider
    let mut frame_value = replay_state.frame_index as u32;
    let max_frame = total_frames.saturating_sub(1) as u32;
    if ui
        .add(egui::Slider::new(&mut frame_value, 0..=max_frame).text("Frame"))
        .changed()
    {
        // can't write through immutable ref here; slider drag handled by tab bar buttons
        // (this is read-only display when called from tab_scene_content)
    }

    // Timeline sparkline with phase transitions
    if let Some(ref document) = replay_state.document {
        let points: PlotPoints = document
            .frames
            .iter()
            .enumerate()
            .map(|(i, f)| [i as f64, f.metrics.observation_count as f64])
            .collect();
        let line = Line::new(points);

        let mut phase_transitions: Vec<(usize, String)> = Vec::new();
        let mut last_phase = String::new();
        for (i, frame) in document.frames.iter().enumerate() {
            if let Some(ref ms) = frame.scan_mission_state {
                if ms.phase != last_phase {
                    if i > 0 {
                        phase_transitions.push((i, ms.phase.clone()));
                    }
                    last_phase = ms.phase.clone();
                }
            }
        }

        let current_frame = replay_state.frame_index as f64;

        Plot::new("obs_sparkline")
            .height(60.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, true])
            .label_formatter(|_, _| String::new())
            .show(ui, |plot_ui| {
                plot_ui.line(line);
                for (frame_idx, phase) in &phase_transitions {
                    let color = phase_vline_color(phase.as_str());
                    plot_ui.vline(VLine::new(*frame_idx as f64).color(color).width(2.0));
                }
                plot_ui.vline(
                    VLine::new(current_frame)
                        .color(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180))
                        .width(1.5),
                );
            });
        if !phase_transitions.is_empty() {
            ui.horizontal(|ui| {
                ui.label("Phases:");
                for (_, phase) in &phase_transitions {
                    let color = phase_label_color(phase.as_str());
                    ui.colored_label(color, format!("▎{}", phase));
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Section: Performance
// ---------------------------------------------------------------------------

fn section_performance(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    diagnostics: &bevy::diagnostic::DiagnosticsStore,
) {
    ui.collapsing("Performance", |ui| {
        use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
        if let Some(fps) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
        {
            ui.label(format!("FPS: {:.1}", fps));
        } else {
            ui.label("FPS: --");
        }
        if let Some(frame_time) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
            .and_then(|d| d.smoothed())
        {
            ui.label(format!("Frame time: {:.2} ms", frame_time));
        }
        let frame_count = replay_state.frame_count();
        ui.label(format!("Replay frames: {}", frame_count));
        if frame_count > 0 {
            let current = replay_state.frame_index;
            let pct = current as f32 / frame_count as f32 * 100.0;
            ui.label(format!(
                "Frame: {}/{} ({:.0}%)",
                current + 1,
                frame_count,
                pct
            ));
            let ts = replay_state.current_timestamp_s();
            let speed = replay_state.playback_speed;
            ui.label(format!(
                "Timestamp: {:.1} s  Speed: {:.1}\u{00d7}",
                ts, speed
            ));
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Tracking Metrics
// ---------------------------------------------------------------------------

fn section_tracking_metrics(ui: &mut egui::Ui, frame_metrics: &CurrentFrameMetrics) {
    ui.collapsing("Tracking Metrics", |ui| {
        let Some(metrics) = &frame_metrics.metrics else {
            ui.label("No metrics available.");
            return;
        };
        kv_row(
            ui,
            "Active tracks",
            &format!("{}", metrics.active_track_count),
        );
        if let Some(mean_err) = metrics.mean_error_m {
            kv_row(ui, "Mean error", &format!("{:.2} m", mean_err));
        }
        if let Some(max_err) = metrics.max_error_m {
            kv_row(ui, "Max error", &format!("{:.2} m", max_err));
        }
        kv_row(
            ui,
            "Observations",
            &format!("{}", metrics.observation_count),
        );
        let total_obs = metrics.accepted_observation_count + metrics.rejected_observation_count;
        if total_obs > 0 {
            let accept_rate = metrics.accepted_observation_count as f32 / total_obs as f32;
            kv_row(
                ui,
                "Accepted",
                &format!(
                    "{}/{} ({:.0}%)",
                    metrics.accepted_observation_count,
                    total_obs,
                    accept_rate * 100.0
                ),
            );
            ui.add(egui::ProgressBar::new(accept_rate).text("acceptance rate"));
        }
        if !metrics.rejection_counts.is_empty() {
            ui.collapsing("Rejection Breakdown", |ui| {
                let mut reasons: Vec<_> = metrics.rejection_counts.iter().collect();
                reasons.sort_by(|a, b| b.1.cmp(a.1));
                for (reason, count) in reasons {
                    ui.label(format!("  {reason}: {count}"));
                }
            });
        }
        if let Some(std_m) = metrics.mean_measurement_std_m {
            kv_row(ui, "Mean meas. std", &format!("{:.2} m", std_m));
        }
        if !metrics.track_errors_m.is_empty() {
            ui.collapsing("Per-Track Errors", |ui| {
                let mut errors: Vec<_> = metrics.track_errors_m.iter().collect();
                errors.sort_by(|a, b| a.1.total_cmp(b.1));
                for (track_id, error) in errors {
                    ui.label(format!("  {track_id}: {error:.2} m"));
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Node Summary
// ---------------------------------------------------------------------------

fn section_node_summary(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    if frame.nodes.is_empty() {
        return;
    }

    ui.collapsing(format!("Nodes ({})", frame.nodes.len()), |ui| {
        egui::Grid::new("node_summary_grid")
            .striped(true)
            .min_col_width(40.0)
            .show(ui, |ui| {
                ui.strong("ID");
                ui.strong("Type");
                ui.strong("Health");
                ui.strong("Mobile");
                ui.strong("Alt (m)");
                ui.end_row();

                for node in &frame.nodes {
                    ui.label(&node.node_id);
                    ui.label(&node.sensor_type);
                    let health_pct = node.health * 100.0;
                    let health_color = health_color_egui(node.health);
                    ui.colored_label(health_color, format!("{:.0}%", health_pct));
                    ui.label(if node.is_mobile { "Yes" } else { "No" });
                    ui.label(format!("{:.1}", node.position[2]));
                    ui.end_row();
                }
            });

        ui.collapsing("Sensor Details", |ui| {
            for node in &frame.nodes {
                ui.horizontal(|ui| {
                    ui.label(format!("{}: {}", node.node_id, node.sensor_type));
                    if let Some(fov) = node.fov_half_angle_deg {
                        ui.label(format!("FOV: {:.0}\u{00b0}", fov * 2.0));
                    }
                    if let Some(range) = node.max_range_m {
                        ui.label(format!("Range: {:.0} m", range));
                    }
                });
            }
        });
    });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Track Summary
// ---------------------------------------------------------------------------

fn section_track_summary(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    frame_metrics: &CurrentFrameMetrics,
) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    if frame.tracks.is_empty() {
        return;
    }

    ui.collapsing(format!("Tracks ({})", frame.tracks.len()), |ui| {
        egui::Grid::new("track_summary_grid")
            .striped(true)
            .min_col_width(40.0)
            .show(ui, |ui| {
                ui.strong("ID");
                ui.strong("Error (m)");
                ui.strong("Updates");
                ui.strong("Stale");
                ui.strong("Std (m)");
                ui.end_row();

                for track in &frame.tracks {
                    ui.label(&track.track_id);
                    let error = frame_metrics
                        .metrics
                        .as_ref()
                        .and_then(|m| m.track_errors_m.get(&track.track_id).copied());
                    if let Some(err) = error {
                        let err_color = if err > 10.0 {
                            egui::Color32::RED
                        } else if err > 5.0 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::GREEN
                        };
                        ui.colored_label(err_color, format!("{:.2}", err));
                    } else {
                        ui.label("-");
                    }
                    ui.label(format!("{}", track.update_count));
                    let stale_color = if track.stale_steps > STALE_THRESHOLD {
                        egui::Color32::RED
                    } else if track.stale_steps > 0 {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::GREEN
                    };
                    ui.colored_label(stale_color, format!("{}", track.stale_steps));
                    ui.label(format!("{:.2}", track.measurement_std_m));
                    ui.end_row();
                }
            });

        let has_covariance = frame.tracks.iter().any(|t| t.covariance.is_some());
        if has_covariance {
            ui.collapsing("Covariance Diagonals", |ui| {
                for track in &frame.tracks {
                    if let Some(cov) = &track.covariance {
                        let diag = covariance_diagonal(cov);
                        ui.label(format!(
                            "{}: \u{03c3}x={:.2} \u{03c3}y={:.2} \u{03c3}z={:.2} m",
                            track.track_id,
                            diag[0].sqrt(),
                            diag[1].sqrt(),
                            diag[2].sqrt()
                        ));
                    }
                }
            });
        }
    });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Selection Inspector
// ---------------------------------------------------------------------------

fn section_selection(ui: &mut egui::Ui, selection: &SelectionState) {
    ui.heading("Selection");
    let (Some(kind), Some(label), Some(position)) = (
        &selection.selected_kind,
        &selection.selected_label,
        &selection.selected_position,
    ) else {
        ui.label("Click a marker to inspect it.");
        return;
    };

    ui.strong(format!("{kind}: {label}"));
    ui.label(format!(
        "Position: x={:.1} y={:.1} z={:.1} m",
        position.x, position.y, position.z
    ));
    if let Some(vel) = &selection.selected_velocity {
        let speed = vel.length();
        ui.label(format!(
            "Velocity: [{:.1}, {:.1}, {:.1}] m/s (speed {:.1} m/s)",
            vel.x, vel.y, vel.z, speed
        ));
        if speed > 0.1 {
            let heading_deg = vel.y.atan2(vel.x).to_degrees();
            ui.label(format!("Heading: {:.1}\u{00b0}", heading_deg));
        }
    }

    match kind.as_str() {
        "Track" => {
            if let Some(std_m) = selection.selected_measurement_std_m {
                kv_row(ui, "Meas. std", &format!("{:.2} m", std_m));
            }
            if let Some(updates) = selection.selected_update_count {
                kv_row(ui, "Update count", &format!("{}", updates));
            }
            if let Some(stale) = selection.selected_stale_steps {
                let color = if stale > STALE_THRESHOLD {
                    egui::Color32::RED
                } else if stale > 0 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::GREEN
                };
                ui.horizontal(|ui| {
                    ui.label("Stale steps:");
                    ui.colored_label(color, format!("{}", stale));
                });
            }
            if let Some(err) = selection.selected_track_error_m {
                kv_row(ui, "Track error", &format!("{:.2} m", err));
            }
            if let Some(diag) = &selection.selected_covariance_diag {
                ui.label(format!(
                    "Cov diag: \u{03c3}x={:.2} \u{03c3}y={:.2} \u{03c3}z={:.2} m",
                    diag[0].sqrt(),
                    diag[1].sqrt(),
                    diag[2].sqrt()
                ));
            }
            if let Some(p_cv) = selection.selected_mode_probability_cv {
                let p_ct = 1.0 - p_cv;
                kv_row(
                    ui,
                    "IMM blend",
                    &format!("CV {:.0}% / CT {:.0}%", p_cv * 100.0, p_ct * 100.0),
                );
            }
            if !selection.selected_contributing_nodes.is_empty() {
                kv_row(
                    ui,
                    "Updated by",
                    &selection.selected_contributing_nodes.join(", "),
                );
            }
        }
        "Node" => {
            if let Some(health) = selection.selected_health {
                ui.horizontal(|ui| {
                    ui.label("Health:");
                    ui.colored_label(health_color_egui(health), format!("{:.0}%", health * 100.0));
                });
            }
            if let Some(sensor) = &selection.selected_sensor_type {
                kv_row(ui, "Sensor", sensor);
            }
            if let Some(mobile) = selection.selected_is_mobile {
                kv_row(ui, "Mobile", if mobile { "Yes" } else { "No" });
            }
            if let Some(fov) = selection.selected_fov_half_angle_deg {
                kv_row(
                    ui,
                    "FOV",
                    &format!("{:.0}\u{00b0} (half {:.0}\u{00b0})", fov * 2.0, fov),
                );
            }
            if let Some(range) = selection.selected_max_range_m {
                kv_row(ui, "Max range", &format!("{:.0} m", range));
            }
        }
        "Truth" => {
            if let Some(dist) = selection.selected_nearest_track_dist_m {
                kv_row(ui, "Nearest track", &format!("{:.2} m", dist));
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Section: Safety & Alerts
// ---------------------------------------------------------------------------

fn section_safety_alerts(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    let mut alerts: Vec<(AlertLevel, String)> = Vec::new();

    if mission_zones.terrain_mesh.is_none() {
        alerts.push((
            AlertLevel::Warning,
            "Terrain data unavailable \u{2014} elevation falls back to entity altitude".to_string(),
        ));
    }

    for track in &frame.tracks {
        if track.stale_steps > STALE_THRESHOLD {
            alerts.push((
                AlertLevel::Warning,
                format!(
                    "Track {} stale for {} steps",
                    track.track_id, track.stale_steps
                ),
            ));
        }
    }

    for node in &frame.nodes {
        if node.health < HEALTH_ALERT_THRESHOLD {
            alerts.push((
                AlertLevel::Error,
                format!(
                    "Node {} health critical: {:.0}%",
                    node.node_id,
                    node.health * 100.0
                ),
            ));
        }
    }

    for truth in &frame.truths {
        let truth_pos = Vec3::from_array(truth.position);
        let has_nearby_track = frame.tracks.iter().any(|t| {
            let track_pos = Vec3::from_array(t.position);
            truth_pos.distance(track_pos) < 50.0
        });
        if !has_nearby_track {
            alerts.push((
                AlertLevel::Warning,
                format!("Truth {} has no nearby track", truth.target_id),
            ));
        }
    }

    let total_obs =
        frame.metrics.accepted_observation_count + frame.metrics.rejected_observation_count;
    if total_obs > 0 {
        let reject_rate = frame.metrics.rejected_observation_count as f32 / total_obs as f32;
        if reject_rate > 0.5 {
            alerts.push((
                AlertLevel::Warning,
                format!(
                    "High rejection rate: {:.0}% ({}/{})",
                    reject_rate * 100.0,
                    frame.metrics.rejected_observation_count,
                    total_obs
                ),
            ));
        }
    }

    let alert_count = alerts.len();
    let header_text = if alert_count > 0 {
        format!("Alerts ({alert_count})")
    } else {
        "Alerts".to_string()
    };

    ui.collapsing(header_text, |ui| {
        if alerts.is_empty() {
            ui.colored_label(egui::Color32::GREEN, "No active alerts.");
        } else {
            for (level, message) in &alerts {
                let (icon, color) = match level {
                    AlertLevel::Error => ("ERROR", egui::Color32::RED),
                    AlertLevel::Warning => ("WARN", egui::Color32::YELLOW),
                };
                ui.horizontal(|ui| {
                    ui.colored_label(color, icon);
                    ui.label(message);
                });
            }
        }
    });
    ui.separator();
}

#[derive(Clone, Copy)]
enum AlertLevel {
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// Section: Mapping Status
// ---------------------------------------------------------------------------

fn section_mapping_status(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref ms) = frame.mapping_state else {
        return;
    };
    egui::CollapsingHeader::new("Mapping")
        .default_open(true)
        .show(ui, |ui| {
            kv_row(
                ui,
                "Coverage",
                &format!("{:.1}%", ms.coverage_fraction * 100.0),
            );
            kv_row(ui, "Covered cells", &ms.covered_cells.to_string());
            kv_row(ui, "Total cells", &ms.total_cells.to_string());
            kv_row(ui, "Mean revisits", &format!("{:.2}", ms.mean_revisits));
        });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Localization Status
// ---------------------------------------------------------------------------

fn section_localization_status(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref ls) = frame.localization_state else {
        return;
    };
    egui::CollapsingHeader::new("Localization")
        .default_open(true)
        .show(ui, |ui| {
            kv_row(ui, "Active tracks", &ls.active_localizations.to_string());
            kv_row(
                ui,
                "Mean pos. std.",
                &format!("{:.1} m", ls.mean_position_std_m),
            );
            kv_row(
                ui,
                "Mean confidence",
                &format!("{:.2}", ls.mean_observation_confidence),
            );
        });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Inspection Events
// ---------------------------------------------------------------------------

fn section_inspection_events(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    if frame.inspection_events.is_empty() {
        return;
    }
    let count = frame.inspection_events.len();
    egui::CollapsingHeader::new(format!("Inspection Events ({count})"))
        .default_open(true)
        .show(ui, |ui| {
            let limit = 20;
            for (i, ev) in frame.inspection_events.iter().enumerate() {
                if i >= limit {
                    ui.label(format!("  ... and {} more", count - limit));
                    break;
                }
                let icon = match ev.event_type.as_str() {
                    "entered" => "\u{25b6}",
                    "exited" => "\u{25c0}",
                    _ => "\u{25cf}",
                };
                ui.label(format!(
                    "  {} {} in {} ({:.0}%) t={:.1}s",
                    icon,
                    ev.node_id,
                    ev.zone_id,
                    ev.zone_coverage_fraction * 100.0,
                    ev.timestamp_s,
                ));
            }
        });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Frame Events
// ---------------------------------------------------------------------------

fn section_frame_events(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    let gen_rej_count = frame.generation_rejections.len();
    let tracker_rej_count = frame.rejected_observations.len();
    let deconflict_count = frame.deconfliction_events.len();
    let obs_count = frame.observations.len();
    let total_events = gen_rej_count + tracker_rej_count + deconflict_count;

    if total_events == 0 && obs_count == 0 {
        return;
    }

    let header = format!("Frame Events ({total_events})");
    ui.collapsing(header, |ui| {
        if !frame.generation_rejections.is_empty() {
            ui.collapsing(format!("Generation Rejections ({gen_rej_count})"), |ui| {
                let display_limit = 30;
                for (i, rej) in frame.generation_rejections.iter().enumerate() {
                    if i >= display_limit {
                        ui.label(format!("  ... and {} more", gen_rej_count - display_limit));
                        break;
                    }
                    ui.label(format!(
                        "  {} {}->{}: {}",
                        rej.reason, rej.node_id, rej.target_id, rej.blocker_type
                    ));
                }
            });
        }

        if !frame.rejected_observations.is_empty() {
            ui.collapsing(format!("Tracker Rejections ({tracker_rej_count})"), |ui| {
                let display_limit = 30;
                for (i, rej) in frame.rejected_observations.iter().enumerate() {
                    if i >= display_limit {
                        ui.label(format!(
                            "  ... and {} more",
                            tracker_rej_count - display_limit
                        ));
                        break;
                    }
                    ui.label(format!(
                        "  {} {}->{}: {}",
                        rej.reason, rej.node_id, rej.target_id, rej.detail
                    ));
                }
            });
        }

        if !frame.deconfliction_events.is_empty() {
            ui.collapsing(format!("Deconfliction Yields ({deconflict_count})"), |ui| {
                let display_limit = 30;
                for (i, ev) in frame.deconfliction_events.iter().enumerate() {
                    if i >= display_limit {
                        ui.label(format!(
                            "  ... and {} more",
                            deconflict_count - display_limit
                        ));
                        break;
                    }
                    ui.label(format!(
                        "  {} yields to {} ({:.1} m, {})",
                        ev.yielding_drone_id,
                        ev.conflicting_drone_id,
                        ev.predicted_separation_m,
                        ev.resolution,
                    ));
                }
            });
        }

        if obs_count > 0 {
            ui.label(format!(
                "Active observations: {obs_count} (toggle overlay to view)"
            ));
        }
    });
    ui.separator();
}

// ---------------------------------------------------------------------------
// Section: Layers
// ---------------------------------------------------------------------------

fn section_layers(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    layer_visibility: &mut LayerVisibilityState,
) {
    ui.collapsing("Layers", |ui| {
        let mut base_layers = scene_package.manifest.layers.clone();
        base_layers.sort_by_key(|layer| {
            scene_package
                .style_for(&layer.style_id)
                .map(|style| style.draw_order)
                .unwrap_or_default()
        });
        for layer in base_layers {
            let style_name = scene_package
                .style_for(&layer.style_id)
                .map(|style| format!("{} ({})", layer.id, style.semantic_kind))
                .unwrap_or_else(|| layer.id.clone());
            let entry = layer_visibility
                .base_layers
                .entry(layer.id.clone())
                .or_insert(true);
            ui.checkbox(entry, style_name);
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Runtime Overlays
// ---------------------------------------------------------------------------

fn section_runtime_overlays(
    ui: &mut egui::Ui,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
) {
    ui.collapsing("Runtime Overlays", |ui| {
        ui.checkbox(&mut runtime_visibility.tracks, "Tracks");
        ui.checkbox(&mut runtime_visibility.truths, "Truths");
        ui.checkbox(&mut runtime_visibility.nodes, "Nodes");
        ui.checkbox(&mut runtime_visibility.observations, "Observation Rays");
        ui.checkbox(
            &mut runtime_visibility.rejection_markers,
            "Rejection Markers",
        );
        ui.checkbox(&mut runtime_visibility.zones, "Mission Zones");
        ui.checkbox(&mut runtime_visibility.fov_cones, "FOV Cones");
        ui.checkbox(&mut runtime_visibility.radar_rings, "Radar Rings");
        ui.checkbox(&mut runtime_visibility.coverage_overlay, "Coverage Overlay");
        ui.checkbox(
            &mut runtime_visibility.inspection_events,
            "Inspection Events",
        );
        ui.checkbox(&mut runtime_visibility.launch_lines, "Launch Lines");
        ui.checkbox(
            &mut runtime_visibility.show_covariance_ellipsoids,
            "Covariance Ellipsoids",
        );
        ui.separator();
        ui.label("Mission overlays:");
        ui.checkbox(&mut mission_overlay.show_scan_grid, "Coverage grid");
        ui.checkbox(&mut mission_overlay.show_poi_markers, "POI markers");
        ui.checkbox(
            &mut mission_overlay.show_loc_ellipses,
            "Localization ellipses",
        );
        ui.checkbox(&mut mission_overlay.show_egress_paths, "Egress paths");
    });
}

// ---------------------------------------------------------------------------
// Section: Mission Zones
// ---------------------------------------------------------------------------

fn section_mission_zones(
    ui: &mut egui::Ui,
    mission_zones: &LoadedMissionZones,
    zone_overlap_model: &mut ZoneOverlapModel,
) {
    if mission_zones.zones.is_empty() {
        return;
    }

    ui.horizontal(|ui| {
        ui.label("Zones:");
        for (label, zone_type) in &[
            ("S", "surveillance"),
            ("X", "exclusion"),
            ("P", "patrol"),
            ("O", "objective"),
        ] {
            let rgba = zone_color_rgba(zone_type, 1.0);
            let color = egui::Color32::from_rgba_unmultiplied(
                (rgba[0] * 255.0) as u8,
                (rgba[1] * 255.0) as u8,
                (rgba[2] * 255.0) as u8,
                200,
            );
            ui.colored_label(color, *label);
        }
    });

    ui.collapsing(
        format!("Mission Zones ({})", mission_zones.zones.len()),
        |ui| {
            if !matches!(zone_overlap_model.focus, ZoneFocus::None)
                && ui.small_button("Clear Focus").clicked()
            {
                zone_overlap_model.focus = ZoneFocus::None;
            }

            let groups = zone_overlap_model.groups.clone();
            for group in groups {
                let selected_group = matches!(
                    zone_overlap_model.focus,
                    ZoneFocus::Group(group_index) if group_index == group.group_id
                );
                let is_expanded = selected_group;

                let chip_summary = format_group_chips(&group);
                let summary = format!(
                    "Group {} \u{00b7} {} zone{} \u{00b7} {}",
                    group.group_id + 1,
                    group.zone_indices.len(),
                    if group.zone_indices.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    chip_summary
                );
                if ui.selectable_label(selected_group, summary).clicked() {
                    zone_overlap_model.focus = if selected_group {
                        ZoneFocus::None
                    } else {
                        ZoneFocus::Group(group.group_id)
                    };
                }

                if is_expanded {
                    ui.indent(format!("zone-group-{}", group.group_id), |ui| {
                        for &zone_index in &group.zone_indices {
                            let Some(zone) = mission_zones.zones.get(zone_index) else {
                                continue;
                            };
                            let selected_zone = matches!(
                                zone_overlap_model.focus,
                                ZoneFocus::Zone(active) if active == zone_index
                            );
                            let row = format!(
                                "{} {} \u{00b7} r={:.0}m \u{00b7} pri={}",
                                zone_type_icon(&zone.zone_type),
                                zone.label,
                                zone.radius_m,
                                zone.priority
                            );
                            if ui.selectable_label(selected_zone, row).clicked() {
                                zone_overlap_model.focus = if selected_zone {
                                    ZoneFocus::None
                                } else {
                                    ZoneFocus::Zone(zone_index)
                                };
                            }
                        }
                    });
                }
            }
        },
    );
}

// ---------------------------------------------------------------------------
// Section: Keyboard Shortcuts
// ---------------------------------------------------------------------------

fn section_keyboard_shortcuts(ui: &mut egui::Ui) {
    ui.separator();
    ui.collapsing("Keyboard Shortcuts", |ui| {
        ui.label("Space       Play / Pause");
        ui.label("Left        Previous frame");
        ui.label("Right       Next frame");
        ui.label("Home        First frame");
        ui.label("End         Last frame");
        ui.label("M           Cycle view mode");
        ui.label("R           Reset camera");
        ui.label("Right-drag  Orbit camera");
        ui.label("Mid-drag    Pan camera");
        ui.label("Scroll      Zoom");
        ui.label("Left-click  Select marker");
    });
}

// ---------------------------------------------------------------------------
// Screen-space Zone Badges
// ---------------------------------------------------------------------------

fn draw_zone_badges(context: &egui::Context, projected_badges: &ProjectedZoneBadges) {
    if projected_badges.badges.is_empty() {
        return;
    }

    egui::Area::new(egui::Id::new("zone_badges_overlay"))
        .fixed_pos(egui::pos2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            for badge in &projected_badges.badges {
                let pill_rect = egui::Rect::from_min_size(
                    egui::pos2(badge.screen_x, badge.screen_y),
                    egui::vec2(badge_pill_width(badge), 20.0),
                );
                let painter = ui.painter();
                painter.rect_filled(
                    pill_rect,
                    4.0,
                    egui::Color32::from_rgba_unmultiplied(30, 30, 40, 200),
                );
                let mut cx = pill_rect.left() + 4.0;
                for chip in &badge.chips {
                    let chip_text = format!("{}{}", zone_type_letter(&chip.zone_type), chip.count);
                    let chip_color = egui::Color32::from_rgba_unmultiplied(
                        (chip.color[0] * 255.0) as u8,
                        (chip.color[1] * 255.0) as u8,
                        (chip.color[2] * 255.0) as u8,
                        (chip.color[3] * 255.0) as u8,
                    );
                    painter.text(
                        egui::pos2(cx, pill_rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        &chip_text,
                        egui::FontId::proportional(11.0),
                        chip_color,
                    );
                    cx += chip_text.len() as f32 * 7.0 + 4.0;
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Mission Phase HUD
// ---------------------------------------------------------------------------

fn draw_mission_phase_hud(context: &egui::Context, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let Some(ref ms) = frame.scan_mission_state else {
        return;
    };

    let (badge_text, badge_color) = match ms.phase.as_str() {
        "scanning" => {
            let pct = (ms.scan_coverage_fraction * 100.0) as u32;
            (
                format!("\u{25a6} SCANNING  {}%", pct),
                egui::Color32::from_rgb(80, 140, 255),
            )
        }
        "localizing" => (
            "\u{25ce} LOCALIZING".to_string(),
            egui::Color32::from_rgb(255, 210, 60),
        ),
        "inspecting" => {
            let done = ms.completed_poi_count;
            let total = ms.total_poi_count.max(1);
            (
                format!("INSPECTING  {}/{}", done, total),
                egui::Color32::from_rgb(255, 140, 40),
            )
        }
        "egress" => {
            let arrived = egress_arrived_count(&ms.egress_progress);
            let total = ms.egress_progress.len().max(1);
            (
                format!("EGRESS  {}/{}", arrived, total),
                egui::Color32::from_rgb(255, 160, 30),
            )
        }
        "complete" => ("COMPLETE".to_string(), egui::Color32::from_rgb(60, 200, 80)),
        other => (other.to_ascii_uppercase(), egui::Color32::GRAY),
    };

    // Position below the operator status strip.
    egui::Area::new(egui::Id::new("phase_hud"))
        .fixed_pos(egui::pos2(12.0, TOP_STATUS_HEIGHT + 8.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            let galley = ui.painter().layout_no_wrap(
                badge_text.clone(),
                egui::FontId::proportional(14.0),
                badge_color,
            );
            let text_size = galley.size();
            let pad = egui::vec2(10.0, 5.0);
            let rect = egui::Rect::from_min_size(ui.next_widget_position(), text_size + pad * 2.0);
            ui.allocate_rect(rect, egui::Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(
                rect,
                5.0,
                egui::Color32::from_rgba_unmultiplied(10, 12, 18, 200),
            );
            painter.text(
                rect.left_center() + egui::vec2(pad.x, 0.0),
                egui::Align2::LEFT_CENTER,
                &badge_text,
                egui::FontId::proportional(14.0),
                badge_color,
            );
        });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn egress_arrived_count(egress_progress: &[EgressProgress]) -> usize {
    egress_progress
        .iter()
        .filter(|progress| progress.distance_to_home_m < EGRESS_ARRIVAL_THRESHOLD_M)
        .count()
}

fn kv_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.strong(value);
    });
}

/// Color for a health fraction (0.0 = red, 0.3 = yellow, 0.7+ = green).
pub fn health_color(health: f32) -> egui::Color32 {
    health_color_egui(health)
}

fn health_color_egui(health: f32) -> egui::Color32 {
    if health > 0.7 {
        egui::Color32::GREEN
    } else if health > 0.3 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::RED
    }
}

/// Extract diagonal elements [0,0], [1,1], [2,2] from a flattened covariance matrix.
pub fn covariance_diagonal(flat: &[f64]) -> [f64; 3] {
    match flat.len() {
        16 => [flat[0], flat[5], flat[10]],
        9 => [flat[0], flat[4], flat[8]],
        n if n >= 4 => [flat[0], flat.get(n / 2 + 1).copied().unwrap_or(0.0), 0.0],
        _ => [0.0, 0.0, 0.0],
    }
}

/// Returns (label, color) pair for a mission phase string.
fn phase_color_pair(phase: &str) -> (&'static str, egui::Color32) {
    match phase {
        "scanning" => ("SCANNING", egui::Color32::from_rgb(80, 140, 255)),
        "localizing" => ("LOCALIZING", egui::Color32::from_rgb(255, 210, 60)),
        "inspecting" => ("INSPECTING", egui::Color32::from_rgb(255, 140, 40)),
        "egress" => ("EGRESS", egui::Color32::from_rgb(255, 160, 30)),
        "complete" => ("COMPLETE", egui::Color32::from_rgb(60, 200, 80)),
        _ => ("UNKNOWN", egui::Color32::GRAY),
    }
}

fn phase_vline_color(phase: &str) -> egui::Color32 {
    match phase {
        "localizing" => egui::Color32::from_rgba_unmultiplied(255, 210, 60, 200),
        "inspecting" => egui::Color32::from_rgba_unmultiplied(255, 140, 40, 200),
        "egress" => egui::Color32::from_rgba_unmultiplied(255, 160, 30, 200),
        "complete" => egui::Color32::from_rgba_unmultiplied(60, 200, 80, 200),
        _ => egui::Color32::from_rgba_unmultiplied(120, 120, 120, 150),
    }
}

fn phase_label_color(phase: &str) -> egui::Color32 {
    match phase {
        "localizing" => egui::Color32::from_rgb(255, 210, 60),
        "inspecting" => egui::Color32::from_rgb(255, 140, 40),
        "egress" => egui::Color32::from_rgb(255, 160, 30),
        "complete" => egui::Color32::from_rgb(60, 200, 80),
        _ => egui::Color32::GRAY,
    }
}

fn format_group_chips(group: &crate::state::ZoneOverlapGroup) -> String {
    let mut counts: Vec<_> = group.type_counts.iter().collect();
    counts.sort_by(|a, b| a.0.cmp(b.0));
    counts
        .into_iter()
        .map(|(zone_type, count)| format!("{}{}", zone_type_letter(zone_type), count))
        .collect::<Vec<_>>()
        .join("  ")
}

fn zone_type_icon(zone_type: &str) -> &'static str {
    match zone_type {
        "surveillance" => "[S]",
        "exclusion" => "[X]",
        "patrol" => "[P]",
        "objective" => "[O]",
        _ => "[?]",
    }
}

fn zone_type_letter(zone_type: &str) -> &'static str {
    match zone_type {
        "surveillance" => "S",
        "exclusion" => "X",
        "patrol" => "P",
        "objective" => "O",
        _ => "?",
    }
}

fn badge_pill_width(badge: &crate::mission_zones::ProjectedZoneBadge) -> f32 {
    let chip_widths: f32 = badge
        .chips
        .iter()
        .map(|chip| {
            let text = format!("{}{}", zone_type_letter(&chip.zone_type), chip.count);
            text.len() as f32 * 7.0 + 4.0
        })
        .sum();
    chip_widths + 8.0
}

fn combo_box(
    ui: &mut egui::Ui,
    label: &str,
    current: &str,
    options: &[&str],
    mut on_change: impl FnMut(&str),
) {
    egui::ComboBox::from_label(label)
        .selected_text(current)
        .show_ui(ui, |ui| {
            for &opt in options {
                if ui.selectable_label(current == opt, opt).clicked() {
                    on_change(opt);
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_diagonal_extracts_4x4() {
        let flat_4x4: Vec<f64> = vec![
            1.0, 0.1, 0.0, 0.0, 0.1, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 16.0,
        ];
        let diag = covariance_diagonal(&flat_4x4);
        assert!((diag[0] - 1.0).abs() < 1e-9);
        assert!((diag[1] - 4.0).abs() < 1e-9);
        assert!((diag[2] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_diagonal_extracts_3x3() {
        let flat_3x3: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 8.0];
        let diag = covariance_diagonal(&flat_3x3);
        assert!((diag[0] - 2.0).abs() < 1e-9);
        assert!((diag[1] - 5.0).abs() < 1e-9);
        assert!((diag[2] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_diagonal_handles_empty() {
        let diag = covariance_diagonal(&[]);
        assert_eq!(diag, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn health_color_returns_expected_colors() {
        assert_eq!(health_color(0.9), egui::Color32::GREEN);
        assert_eq!(health_color(0.5), egui::Color32::YELLOW);
        assert_eq!(health_color(0.1), egui::Color32::RED);
    }

    #[test]
    fn egress_arrived_count_uses_shared_threshold() {
        let progress = vec![
            EgressProgress {
                drone_id: "arrived".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M - 0.1,
                home_position: [0.0, 0.0, 0.0],
            },
            EgressProgress {
                drone_id: "at-threshold".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M,
                home_position: [0.0, 0.0, 0.0],
            },
            EgressProgress {
                drone_id: "en-route".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M + 5.0,
                home_position: [0.0, 0.0, 0.0],
            },
        ];

        assert_eq!(egress_arrived_count(&progress), 1);
    }
}
