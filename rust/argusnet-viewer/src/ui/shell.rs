//! The shell: header, status line, mission rail, selection inspector, drawers,
//! tabs and HUD overlays (VIEWER_UI §4.3, §11.3).

use super::*;

/// The application header (VIEWER_UI §4.3): identity and source, the
/// workspace / projection / preset controls (C2.3), the world view, and the
/// tool panel (secondary destinations, §4.2). Time controls live in the
/// timeline. Returns a shell command
/// for the caller to apply.
pub(super) fn header_bar(
    ui: &mut egui::Ui,
    source: SourceContext,
    scene_package: &ScenePackage,
    view_mode: &mut ViewMode,
    ui_state: &mut ViewerUiState,
) -> Option<ShellCommand> {
    let mut command = None;
    // One wrapping row: at the reference width it fits on a line, and a
    // narrower window wraps it within the header rather than overlapping.
    ui.horizontal_wrapped(|ui| {
        ui.strong("ArgusNet");
        // The source indicator (§3): first after identity, in the one region
        // no window size or drawer state removes.
        ui.label(source.rich_text())
            .on_hover_text(source.explanation());
        ui.label(&scene_package.manifest.scene_id);
        ui.separator();
        command = shell_controls(ui, ui_state).or(command);
        ui.separator();
        for (mode, label) in [
            (ViewMode::RealWorld, "Real"),
            (ViewMode::ScanMap, "Scan"),
            (ViewMode::Split, "Split"),
        ] {
            if ui
                .selectable_label(*view_mode == mode, label)
                .on_hover_text("World view (M cycles)")
                .clicked()
            {
                *view_mode = mode;
            }
        }
        ui.separator();
        let tools_label = if ui_state.drawer_open {
            format!("Tools: {}", mode_label(ui_state.mode))
        } else {
            "Tools".to_string()
        };
        ui.menu_button(tools_label, |ui| {
            for (mode, label) in [
                (ViewerMode::Mission, "Mission"),
                (ViewerMode::Replay, "Replay"),
                (ViewerMode::Diagnostics, "Diagnostics"),
                (ViewerMode::Scenario, "Scenario"),
            ] {
                mode_button(ui, ui_state, mode, label);
            }
            ui.separator();
            let toggle = if ui_state.drawer_open {
                "Close tool panel (T)"
            } else {
                "Open tool panel (T)"
            };
            if ui.button(toggle).clicked() {
                command = Some(ShellCommand::ToggleToolPanel);
            }
        });
    });
    command
}

fn mode_label(mode: ViewerMode) -> &'static str {
    match mode {
        ViewerMode::Mission => "Mission",
        ViewerMode::Replay => "Replay",
        ViewerMode::Diagnostics => "Diagnostics",
        ViewerMode::Scenario => "Scenario",
    }
}

/// The status and freshness line: mission phase, time, alert counts, the
/// camera, and anything the layout had to give up. Alert counts live here so
/// that no window size hides a critical alert (§4.4 rule 5).
pub(super) fn status_line(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    alerts: Option<&[(AlertLevel, String)]>,
    camera: &crate::camera_control::CameraControl,
    prefs: &crate::prefs::ViewerPrefs,
    layout: &ShellLayout,
) {
    ui.horizontal_centered(|ui| {
        if let Some((phase, color, detail)) = mission_status_summary(replay_state) {
            status_chip(ui, phase, color);
            ui.label(detail);
        } else if replay_state.frame_count() > 0 {
            status_chip(ui, "NO MISSION STATE", egui::Color32::GRAY);
        } else {
            status_chip(ui, "NO REPLAY", egui::Color32::GRAY);
        }

        if replay_state.frame_count() > 0 {
            let cur_s = replay_state.current_timestamp_s();
            let total_s = replay_state
                .document
                .as_ref()
                .and_then(|doc| doc.frames.last())
                .map(|frame| frame.timestamp_s)
                .unwrap_or(0.0);
            ui.separator();
            ui.monospace(format_time_range(cur_s, total_s));
        }

        if let Some(alerts) = alerts {
            let errors = alerts
                .iter()
                .filter(|(level, _)| *level == AlertLevel::Error)
                .count();
            let warnings = alerts.len() - errors;
            ui.separator();
            if errors > 0 {
                status_chip(ui, &format!("{errors} CRITICAL"), egui::Color32::RED);
            }
            if warnings > 0 {
                status_chip(ui, &format!("{warnings} WARN"), egui::Color32::YELLOW);
            }
            if errors == 0 && warnings == 0 {
                ui.label("No alerts");
            }
        }

        // The camera's mode, and what it last had to say (P2.4b): a lost
        // follow is reported here, not silently dropped.
        ui.separator();
        ui.label(format!("Camera: {}", camera.mode_label()));
        if let Some(notice) = camera.notice() {
            ui.colored_label(egui::Color32::from_rgb(230, 190, 110), notice);
        }
        if !prefs.problems.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(230, 120, 100),
                "Preferences problem",
            )
            .on_hover_text(prefs.problems.join("\n"));
        }
        if layout.compact {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(230, 190, 110), "COMPACT")
                .on_hover_text(
                    "The window is below the supported operational size: monitoring only.",
                );
        } else if layout.drawer_suppressed {
            ui.separator();
            ui.label("Panel hidden: window too narrow");
        }
    });
}

/// The mission rail (§4.3): task phase, fleet index, alerts. Compact and
/// index-like.
/// The platforms, as typed keys, in the order the markers give them: the
/// identity world picking and Tab resolve to, never a label.
pub(super) fn fleet_index(
    markers: &CurrentRuntimeMarkers,
) -> Vec<(crate::presentation::EntityKey, std::sync::Arc<str>)> {
    markers
        .markers
        .iter()
        .filter(|marker| marker.kind == MarkerKind::Node)
        .map(|marker| (marker.key, marker.label.clone()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mission_rail(
    ui: &mut egui::Ui,
    workspace: crate::state::Workspace,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    markers: &CurrentRuntimeMarkers,
    selection: &SelectionState,
    ui_state: &mut ViewerUiState,
) {
    // Plan and Review say what they wait for rather than show mock data.
    if workspace_unavailable(ui, workspace) {
        return;
    }
    ui.strong("Mission");
    if let Some((phase, color, detail)) = mission_status_summary(replay_state) {
        status_chip(ui, phase, color);
        ui.small(detail);
    } else if replay_state.frame_count() > 0 {
        ui.small("This replay carries no mission state.");
    } else {
        ui.small("No replay loaded.");
    }
    ui.separator();
    ui.strong("Fleet");
    let fleet = fleet_index(markers);
    if fleet.is_empty() {
        ui.small("No platforms in this frame.");
    }
    let selected = selection.selected.as_ref().map(|selected| selected.key);
    let frame = replay_state.current_frame();
    egui::Grid::new("mission_rail_fleet")
        .num_columns(2)
        .show(ui, |ui| {
            for (index, (key, label)) in fleet.iter().enumerate() {
                // The indexed list is picking's keyboard and pointer twin:
                // the same typed key, through the same request.
                if ui
                    .selectable_label(selected == Some(*key), format!("{} {label}", index + 1))
                    .on_hover_text("Select (Tab steps through every shown entity)")
                    .clicked()
                {
                    ui_state.select_request = Some(crate::picking::SelectRequest::Select(*key));
                }
                match frame.and_then(|frame| {
                    frame
                        .nodes
                        .iter()
                        .find(|node| node.node_id.as_str() == &**label)
                }) {
                    Some(node) => ui.colored_label(
                        health_color_egui(node.health),
                        format!("{:.0}%", node.health * 100.0),
                    ),
                    None => ui.label(crate::theme::Status::Unavailable.rich_text()),
                };
                ui.end_row();
            }
        });
    ui.separator();
    section_safety_alerts(ui, replay_state, mission_zones);
}

/// §7.2's platform sections; with `essential`, only §4.4 rule 3's.
pub(super) fn platform_inspector(
    ui: &mut egui::Ui,
    label: &str,
    sections: &[Section],
    essential: bool,
) {
    ui.label(crate::theme::TypeRole::ObjectIdentity.text(label));
    for section in sections
        .iter()
        .filter(|section| section.essential || !essential)
    {
        ui.add_space(4.0);
        ui.label(crate::theme::TypeRole::SectionTitle.text(section.title));
        // Label then value, wrapping inside the inspector's width: a long
        // unavailable reason flows onto the next line instead of clipping.
        for row in &section.rows {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if row.legacy {
                    ui.label(format!("{} (legacy)", row.label))
                        .on_hover_text("A legacy estimator output: not physical integrity.");
                } else {
                    ui.label(row.label);
                }
                let text = row.value.display(String::clone);
                ui.label(egui::RichText::new(text).color(row.value.status().color()));
            });
        }
    }
}

/// The selection inspector (§4.3). With `essential`, only the selection
/// itself: the entity list is the first thing a narrow window gives up.
pub(super) fn selection_inspector(
    ui: &mut egui::Ui,
    replay_state: &ReplayState,
    selection: &SelectionState,
    markers: &CurrentRuntimeMarkers,
    runtime_visibility: &RuntimeOverlayVisibility,
    ui_state: &mut ViewerUiState,
    essential: bool,
) {
    let platform = selection.selected.as_ref().and_then(|selected| {
        matches!(selected.key, crate::presentation::EntityKey::Platform(_))
            .then(|| replay_state.current_frame())
            .flatten()
            .and_then(|frame| inspect_platform(frame, &selected.label))
            .map(|sections| (selected.label.clone(), sections))
    });
    match platform {
        Some((label, sections)) => platform_inspector(ui, &label, &sections, essential),
        None => section_selection(ui, selection, replay_state.max_stale_steps()),
    }
    if !essential {
        ui.separator();
        section_entity_list(ui, markers, runtime_visibility, selection, ui_state);
    }
}

pub(super) fn mode_button(
    ui: &mut egui::Ui,
    ui_state: &mut ViewerUiState,
    mode: ViewerMode,
    label: &str,
) {
    if ui
        .selectable_label(ui_state.mode == mode, label)
        .on_hover_text(format!("{label} mode"))
        .clicked()
    {
        ui_state.mode = mode;
        ui_state.drawer_open = true;
    }
}

pub(super) fn playback_buttons(ui: &mut egui::Ui, replay_state: &mut ReplayState) {
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

pub(super) fn status_chip(ui: &mut egui::Ui, label: &str, color: egui::Color32) {
    ui.colored_label(color, egui::RichText::new(label).strong());
}

pub(super) fn mission_status_summary(
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

pub(super) fn format_time_range(current_s: f32, total_s: f32) -> String {
    format!(
        "{:.0}:{:04.1} / {:.0}:{:04.1}",
        (current_s / 60.0).floor(),
        current_s % 60.0,
        (total_s / 60.0).floor(),
        total_s % 60.0
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn drawer_mission_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    mission_zones: &LoadedMissionZones,
    reconstruction: &ReconstructionCloud,
    view_mode: &mut ViewMode,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
    markers: &CurrentRuntimeMarkers,
    raster_status: &str,
    layer_visibility: &mut LayerVisibilityState,
    zone_overlap_model: &mut ZoneOverlapModel,
) {
    ui.heading("Mission");
    ui.label("Operator view");
    ui.separator();

    section_mission_progress(ui, replay_state, view_mode);
    ui.separator();
    tab_mission_content(
        ui,
        scene_package,
        replay_state,
        markers,
        mission_zones,
        reconstruction,
    );
    ui.separator();
    section_targets(ui, replay_state);
    ui.separator();
    section_runtime_overlays(ui, runtime_visibility, mission_overlay, Some(raster_status));
    ui.separator();
    section_layers(ui, scene_package, layer_visibility);
    ui.separator();
    section_mission_zones(ui, mission_zones, zone_overlap_model);
}

pub(super) fn apply_layer_preset(
    preset: MapLayerPreset,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
) {
    match preset {
        MapLayerPreset::Operator => {
            runtime_visibility.nodes = true;
            runtime_visibility.tracks = true;
            runtime_visibility.truths = false;
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
            runtime_visibility.truths = false;
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

pub(super) fn drawer_replay_content(
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

pub(super) fn drawer_diagnostics_content(
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

pub(super) fn drawer_scenario_content(
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

pub(super) fn tab_mission_content(
    ui: &mut egui::Ui,
    scene_package: &ScenePackage,
    replay_state: &ReplayState,
    markers: &CurrentRuntimeMarkers,
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
                Line::new("", border)
                    .color(egui::Color32::from_rgba_unmultiplied(100, 120, 160, 180))
                    .width(1.5_f32),
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
                plot_ui.line(
                    Line::new("", PlotPoints::new(circle))
                        .color(color)
                        .width(1.5_f32),
                );
                // Zone label dot at center
                plot_ui.points(
                    Points::new("", PlotPoints::new(vec![[cx, cy]]))
                        .color(color)
                        .radius(3.0_f32),
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
                    Points::new("", PlotPoints::new(scan_pts))
                        .color(egui::Color32::from_rgba_unmultiplied(220, 200, 40, 120))
                        .radius(1.5_f32),
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
                            Polygon::new("", PlotPoints::new(vec![tip, left, right])).fill_color(
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
                            Polygon::new(
                                "",
                                PlotPoints::new(vec![
                                    [nx - s, ny - s],
                                    [nx + s, ny - s],
                                    [nx + s, ny + s],
                                    [nx - s, ny + s],
                                ]),
                            )
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
                        plot_ui.line(
                            Line::new("", PlotPoints::new(circle))
                                .color(color)
                                .width(2.0_f32),
                        );
                    }
                }

                // Trails: the last 16 frames' positions per mobile node, as the
                // adapter keeps them.
                for (index, marker) in markers.markers.iter().enumerate() {
                    if marker.kind != MarkerKind::Node || marker.is_mobile != Some(true) {
                        continue;
                    }
                    let trail_xy: Vec<[f64; 2]> = markers
                        .trail(index, 16)
                        .map(|p| [p.x as f64, p.y as f64])
                        .collect();
                    if trail_xy.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(trail_xy))
                                .color(egui::Color32::from_rgba_unmultiplied(180, 180, 80, 80))
                                .width(1.0_f32),
                        );
                    }
                }
            }

            // Crosshair at scene center (faint)
            plot_ui.vline(
                VLine::new("", bx_mid)
                    .color(egui::Color32::from_rgba_unmultiplied(120, 120, 120, 60)),
            );
            plot_ui.hline(
                HLine::new("", by_mid)
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
                            .inner_margin(egui::Margin::same(6))
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
pub(super) fn tab_scene_content(
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

    section_selection(ui, selection, replay_state.max_stale_steps());
    ui.separator();

    section_layers(ui, scene_package, layer_visibility);
    ui.separator();

    section_runtime_overlays(ui, runtime_visibility, mission_overlay, None);
    ui.separator();

    section_mission_zones(ui, mission_zones, zone_overlap_model);

    section_keyboard_shortcuts(ui);
}

// ---------------------------------------------------------------------------
// Tab: Tracks — metrics, node table, track table
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(super) fn tab_tracks_content(
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
pub(super) fn tab_events_content(ui: &mut egui::Ui, replay_state: &ReplayState) {
    section_inspection_events(ui, replay_state);
    section_frame_events(ui, replay_state);
}

// ---------------------------------------------------------------------------
// Section: Scene Header + Terrain Info
// ---------------------------------------------------------------------------

pub(super) fn section_keyboard_shortcuts(ui: &mut egui::Ui) {
    ui.separator();
    ui.collapsing("Keyboard Shortcuts", |ui| {
        ui.label("Space       Play / Pause");
        ui.label("Left        Previous frame");
        ui.label("Right       Next frame");
        ui.label("Home        First frame");
        ui.label("End         Last frame");
        ui.label("[ / ]       Previous / next event");
        ui.label("M           Cycle view mode");
        ui.label("O           Cycle workspace");
        ui.label("V           Cycle projection");
        ui.label("P           Cycle layer preset");
        ui.label("T           Open / close the tool panel");
        ui.label("R           Reset camera");
        ui.label("F           Focus the selection");
        ui.label("L / C       Follow / chase the selection");
        ui.label("G           Free-fly (WASD, E/Q, Shift)");
        ui.label("1-9         Bookmark (Ctrl/Cmd: save)");
        ui.label("Esc         Leave free-fly/follow/chase");
        ui.label("Right-drag  Orbit camera");
        ui.label("Mid-drag    Pan camera");
        ui.label("Scroll      Zoom");
        ui.label("Left-click  Select marker");
        ui.label("Tab         Next marker (Shift: previous)");
        ui.label("Esc         Clear selection");
    });
}

// ---------------------------------------------------------------------------
// Screen-space Zone Badges
// ---------------------------------------------------------------------------

pub(super) fn draw_zone_badges(context: &egui::Context, projected_badges: &ProjectedZoneBadges) {
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

pub(super) fn draw_mission_phase_hud(
    context: &egui::Context,
    replay_state: &ReplayState,
    canvas_origin: egui::Pos2,
) {
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

    // On the canvas, clear of the shell's regions.
    egui::Area::new(egui::Id::new("phase_hud"))
        .fixed_pos(canvas_origin + egui::vec2(12.0, 8.0))
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
