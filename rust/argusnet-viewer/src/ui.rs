use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_plot::{Line, Plot, PlotPoints};

use crate::mission_zones::{zone_color_rgba, ProjectedZoneBadges};
use crate::replay::ReplayState;
use crate::schema::ScenePackage;
use crate::state::{
    CurrentFrameMetrics, LayerVisibilityState, LoadedMissionZones, RuntimeOverlayVisibility,
    SelectionState, SimulationRunner, ZoneFocus, ZoneOverlapModel,
};

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

/// Stale step threshold: nodes/tracks exceeding this are flagged in alerts.
const STALE_THRESHOLD: u32 = 5;
/// Health threshold: nodes below this fraction are flagged in alerts.
const HEALTH_ALERT_THRESHOLD: f32 = 0.3;

pub fn viewer_ui_system(
    mut contexts: EguiContexts,
    scene_package: Res<ScenePackage>,
    mut replay_state: ResMut<ReplayState>,
    mut layer_visibility: ResMut<LayerVisibilityState>,
    mut runtime_visibility: ResMut<RuntimeOverlayVisibility>,
    selection: Res<SelectionState>,
    frame_metrics: Res<CurrentFrameMetrics>,
    mission_zones: Res<LoadedMissionZones>,
    mut zone_overlap_model: ResMut<ZoneOverlapModel>,
    mut sim_runner: ResMut<SimulationRunner>,
    projected_badges: Res<ProjectedZoneBadges>,
) {
    let context = contexts.ctx_mut();
    let supports_hot_swap = scene_package.is_synthetic_source();

    egui::SidePanel::left("scene_panel")
        .resizable(true)
        .default_width(320.0)
        .show(context, |ui| {
            // Wrap the entire panel content in a vertical scroll area so that
            // overflow is always reachable via scrolling.
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    section_scene_header(ui, &scene_package, &mission_zones);
                    ui.separator();

                    section_scenario(ui, supports_hot_swap, &mut sim_runner);
                    ui.separator();

                    section_playback(ui, &mut replay_state);
                    ui.separator();

                    section_tracking_metrics(ui, &frame_metrics);
                    ui.separator();

                    section_mapping_status(ui, &replay_state);

                    section_localization_status(ui, &replay_state);

                    section_inspection_events(ui, &replay_state);

                    section_node_summary(ui, &replay_state);

                    section_track_summary(ui, &replay_state, &frame_metrics);

                    section_selection(ui, &selection);
                    ui.separator();

                    section_safety_alerts(ui, &replay_state);

                    section_frame_events(ui, &replay_state);

                    section_layers(ui, &scene_package, &mut layer_visibility);
                    ui.separator();

                    section_runtime_overlays(ui, &mut runtime_visibility);
                    ui.separator();

                    section_mission_zones(
                        ui,
                        &mission_zones,
                        &mut zone_overlap_model,
                    );

                    section_keyboard_shortcuts(ui);
                });
        });

    // --- Screen-space zone badges (viewport overlay) ---
    draw_zone_badges(context, &projected_badges);
}

// ---------------------------------------------------------------------------
// Section: Scene Header + Terrain Info
// ---------------------------------------------------------------------------

fn section_scene_header(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    mission_zones: &LoadedMissionZones,
) {
    ui.heading("Smart Tracker Viewer");
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
        ui.label(format!(
            "Terrain mesh: {}x{} grid",
            mesh.cols, mesh.rows
        ));
    }
}

// ---------------------------------------------------------------------------
// Section: Scenario Controls
// ---------------------------------------------------------------------------

fn section_scenario(
    ui: &mut egui::Ui,
    supports_hot_swap: bool,
    sim_runner: &mut SimulationRunner,
) {
    ui.collapsing("Scenario", |ui| {
        if !supports_hot_swap {
            ui.label("Re-simulation only for synthetic scenes.");
        }

        ui.add_enabled_ui(supports_hot_swap && !sim_runner.is_busy(), |ui| {
            combo_box(ui, "Map", &sim_runner.map_preset.clone(), MAP_PRESETS, |v| {
                sim_runner.map_preset = v.to_string();
            });
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

            ui.add(
                egui::Slider::new(&mut sim_runner.target_count, 1..=8).text("Targets"),
            );
            ui.add(
                egui::Slider::new(&mut sim_runner.drone_count, 1..=8).text("Drones"),
            );
            ui.add(
                egui::Slider::new(&mut sim_runner.ground_stations, 1..=12)
                    .text("Ground Stations"),
            );
            ui.add(
                egui::Slider::new(&mut sim_runner.duration_s, 30.0..=600.0)
                    .text("Duration (s)")
                    .step_by(10.0),
            );
            ui.add(egui::Slider::new(&mut sim_runner.seed, 0..=999).text("Seed"));
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
// Section: Playback
// ---------------------------------------------------------------------------

fn section_playback(ui: &mut egui::Ui, replay_state: &mut ReplayState) {
    ui.heading("Playback");
    let has_replay = replay_state.frame_count() > 0;
    if !has_replay {
        ui.label("No replay packaged with this scene.");
        return;
    }

    // Replay summary
    let total_frames = replay_state.frame_count();
    let duration_s = replay_state
        .document
        .as_ref()
        .and_then(|d| d.frames.last())
        .map(|f| f.timestamp_s)
        .unwrap_or(0.0);
    ui.label(format!(
        "Replay: {} frames, {:.1}s duration",
        total_frames, duration_s
    ));

    let status = if replay_state.playing {
        "Playing"
    } else {
        "Paused"
    };
    ui.label(format!(
        "Status: {} | t = {:.2}s | frame {}/{}",
        status,
        replay_state.current_timestamp_s(),
        replay_state.frame_index + 1,
        total_frames
    ));

    ui.horizontal(|ui| {
        let button_label = if replay_state.playing {
            "Pause"
        } else {
            "Play"
        };
        if ui.button(button_label).clicked() {
            replay_state.playing = !replay_state.playing;
        }
        if ui.button("|<").clicked() {
            replay_state.playing = false;
            replay_state.step_to(0);
        }
        if ui.button("<").clicked() {
            let previous_index = replay_state.frame_index.saturating_sub(1);
            replay_state.playing = false;
            replay_state.step_to(previous_index);
        }
        if ui.button(">").clicked() {
            let next_index = replay_state.frame_index + 1;
            replay_state.playing = false;
            replay_state.step_to(next_index);
        }
        if ui.button(">|").clicked() {
            replay_state.playing = false;
            let last = replay_state.frame_count().saturating_sub(1);
            replay_state.step_to(last);
        }
    });

    ui.add(
        egui::Slider::new(&mut replay_state.playback_speed, 0.25..=4.0)
            .text("Speed")
            .step_by(0.25),
    );

    let mut frame_value = replay_state.frame_index as u32;
    let max_frame = replay_state.frame_count().saturating_sub(1) as u32;
    if ui
        .add(egui::Slider::new(&mut frame_value, 0..=max_frame).text("Frame"))
        .changed()
    {
        replay_state.playing = false;
        replay_state.step_to(frame_value as usize);
    }

    // Item 13: Timeline sparkline — observations/frame across all frames.
    if let Some(ref document) = replay_state.document {
        let points: PlotPoints = document
            .frames
            .iter()
            .enumerate()
            .map(|(i, f)| [i as f64, f.metrics.observation_count as f64])
            .collect();
        let line = Line::new(points);
        Plot::new("obs_sparkline")
            .height(60.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, true])
            .label_formatter(|_, _| String::new())
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
        ui.label("Observations/frame");
    }
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
        kv_row(ui, "Active tracks", &format!("{}", metrics.active_track_count));
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
        let total_obs =
            metrics.accepted_observation_count + metrics.rejected_observation_count;
        if total_obs > 0 {
            let accept_rate =
                metrics.accepted_observation_count as f32 / total_obs as f32;
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
            ui.add(
                egui::ProgressBar::new(accept_rate).text("acceptance rate"),
            );
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
        // Per-track error breakdown
        if !metrics.track_errors_m.is_empty() {
            ui.collapsing("Per-Track Errors", |ui| {
                let mut errors: Vec<_> = metrics.track_errors_m.iter().collect();
                errors.sort_by(|a, b| {
                    a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (track_id, error) in errors {
                    ui.label(format!("  {track_id}: {error:.2} m"));
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Node (Drone/Station) Summary Table
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
                // Header
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
                    let health_color = health_color(node.health);
                    ui.colored_label(health_color, format!("{:.0}%", health_pct));
                    ui.label(if node.is_mobile { "Yes" } else { "No" });
                    ui.label(format!("{:.1}", node.position[2]));
                    ui.end_row();
                }
            });

        // Sensor details sub-section
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
// Section: Track Summary Table
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
                // Header
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

        // Covariance details
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
// Section: Selection Inspector (enhanced)
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
        }
        "Node" => {
            if let Some(health) = selection.selected_health {
                ui.horizontal(|ui| {
                    ui.label("Health:");
                    ui.colored_label(
                        health_color(health),
                        format!("{:.0}%", health * 100.0),
                    );
                });
            }
            if let Some(sensor) = &selection.selected_sensor_type {
                kv_row(ui, "Sensor", sensor);
            }
            if let Some(mobile) = selection.selected_is_mobile {
                kv_row(ui, "Mobile", if mobile { "Yes" } else { "No" });
            }
            if let Some(fov) = selection.selected_fov_half_angle_deg {
                kv_row(ui, "FOV", &format!("{:.0}\u{00b0} (half {:.0}\u{00b0})", fov * 2.0, fov));
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

fn section_safety_alerts(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    // Collect alerts from current frame state
    let mut alerts: Vec<(AlertLevel, String)> = Vec::new();

    // Check for stale tracks
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

    // Check for low health nodes
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

    // Check for untracked truths (truth with no matching track nearby)
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

    // High rejection rate
    let total_obs = frame.metrics.accepted_observation_count
        + frame.metrics.rejected_observation_count;
    if total_obs > 0 {
        let reject_rate =
            frame.metrics.rejected_observation_count as f32 / total_obs as f32;
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
                    AlertLevel::Error => ("\u{2716}", egui::Color32::RED),
                    AlertLevel::Warning => ("\u{26a0}", egui::Color32::YELLOW),
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
    ui.collapsing("Mapping", |ui| {
        kv_row(ui, "Coverage", &format!("{:.1}%", ms.coverage_fraction * 100.0));
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
    ui.collapsing("Localization", |ui| {
        kv_row(ui, "Active tracks", &ls.active_localizations.to_string());
        kv_row(ui, "Mean pos. std.", &format!("{:.1} m", ls.mean_position_std_m));
        kv_row(ui, "Mean confidence", &format!("{:.2}", ls.mean_observation_confidence));
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
    ui.collapsing(format!("Inspection Events ({count})"), |ui| {
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
// Section: Current Frame Events
// ---------------------------------------------------------------------------

fn section_frame_events(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };

    let gen_rej_count = frame.generation_rejections.len();
    let tracker_rej_count = frame.rejected_observations.len();
    let obs_count = frame.observations.len();
    let total_events = gen_rej_count + tracker_rej_count;

    if total_events == 0 && obs_count == 0 {
        return;
    }

    let header = format!("Frame Events ({total_events})");
    ui.collapsing(header, |ui| {
        // Generation rejections
        if !frame.generation_rejections.is_empty() {
            ui.collapsing(
                format!("Generation Rejections ({gen_rej_count})"),
                |ui| {
                    let display_limit = 30;
                    for (i, rej) in frame.generation_rejections.iter().enumerate() {
                        if i >= display_limit {
                            ui.label(format!(
                                "  ... and {} more",
                                gen_rej_count - display_limit
                            ));
                            break;
                        }
                        ui.label(format!(
                            "  {} {}->{}: {}",
                            rej.reason, rej.node_id, rej.target_id, rej.blocker_type
                        ));
                    }
                },
            );
        }

        // Tracker rejections
        if !frame.rejected_observations.is_empty() {
            ui.collapsing(
                format!("Tracker Rejections ({tracker_rej_count})"),
                |ui| {
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
                },
            );
        }

        // Observation summary (not individual obs to save space)
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
        ui.checkbox(&mut runtime_visibility.inspection_events, "Inspection Events");
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

    // Compact legend row: type-color mapping.
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
                    let chip_text = format!(
                        "{}{}",
                        zone_type_letter(&chip.zone_type),
                        chip.count
                    );
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
// Helpers
// ---------------------------------------------------------------------------

/// Key-value row: label on left, strong value on right.
fn kv_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.strong(value);
    });
}

/// Color for a health fraction (0.0 = red, 0.3 = yellow, 0.7+ = green).
pub fn health_color(health: f32) -> egui::Color32 {
    if health > 0.7 {
        egui::Color32::GREEN
    } else if health > 0.3 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::RED
    }
}

/// Extract diagonal elements [0,0], [1,1], [2,2] from a flattened covariance
/// matrix. Assumes either a 4x4 (16 elements) or 3x3 (9 elements) layout.
pub fn covariance_diagonal(flat: &[f64]) -> [f64; 3] {
    match flat.len() {
        // 4x4 state covariance (x, y, z in first 3 diagonal positions)
        16 => [flat[0], flat[5], flat[10]],
        // 3x3 position covariance
        9 => [flat[0], flat[4], flat[8]],
        // 2x2 or other — return what we can
        n if n >= 4 => [flat[0], flat.get(n / 2 + 1).copied().unwrap_or(0.0), 0.0],
        _ => [0.0, 0.0, 0.0],
    }
}

/// Chip format for group summary: "S2  X1  P3".
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
            1.0, 0.1, 0.0, 0.0, 0.1, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0,
            0.0, 16.0,
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
}
