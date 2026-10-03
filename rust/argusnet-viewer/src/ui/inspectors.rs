//! Inspectors: the read-only sections and diagnostic tables that describe what is selected or loaded.

use super::*;

pub(super) fn diagnostic_node_table(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn diagnostic_track_table(
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
                        let color = if track.stale_steps > replay_state.max_stale_steps() {
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

pub(super) fn diagnostic_event_table(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn section_scene_header(
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

pub(super) fn section_mission_progress(
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

pub(super) fn section_performance(
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

pub(super) fn section_tracking_metrics(ui: &mut egui::Ui, frame_metrics: &CurrentFrameMetrics) {
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

pub(super) fn section_node_summary(ui: &mut egui::Ui, replay_state: &ReplayState) {
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
                ui.strong("Battery");
                ui.end_row();

                for node in &frame.nodes {
                    ui.label(&node.node_id);
                    ui.label(&node.sensor_type);
                    let health_pct = node.health * 100.0;
                    let health_color = health_color_egui(node.health);
                    ui.colored_label(health_color, format!("{:.0}%", health_pct));
                    ui.label(if node.is_mobile { "Yes" } else { "No" });
                    ui.label(format!("{:.1}", node.position[2]));
                    // Replay-only: `pb::NodeState` carries no battery field yet.
                    match node.battery_fraction {
                        Some(fraction) => ui.colored_label(
                            health_color_egui(fraction),
                            format!("{:.0}%", fraction * 100.0),
                        ),
                        None => ui.label("-"),
                    };
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

pub(super) fn section_track_summary(
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
                    let stale_color = if track.stale_steps > replay_state.max_stale_steps() {
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
// Section: Entity list (P2.3)
// ---------------------------------------------------------------------------

/// Every shown marker, by kind: the list path to a selection, equivalent to
/// clicking one. Rows are keyed, so a track and a truth that share an id are
/// two rows under two headings. The click becomes a request the picking
/// stage applies; this panel only reads the selection.
pub(super) fn section_entity_list(
    ui: &mut egui::Ui,
    markers: &CurrentRuntimeMarkers,
    runtime_visibility: &RuntimeOverlayVisibility,
    selection: &SelectionState,
    ui_state: &mut ViewerUiState,
) {
    let selected = selection.selected.as_ref().map(|s| s.key);
    let shown: Vec<_> = markers
        .markers
        .iter()
        .filter(|marker| crate::app::is_marker_visible(marker, runtime_visibility))
        .collect();
    egui::CollapsingHeader::new(format!("Entities ({} shown)", shown.len()))
        .id_salt("entity_list")
        .show(ui, |ui| {
            if shown.is_empty() {
                ui.label("Nothing shown: turn on tracks, truths or nodes.");
                return;
            }
            ui.small("Tab / Shift+Tab step through these; Esc clears.");
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .id_salt("entity_list_scroll")
                .show(ui, |ui| {
                    for (kind, heading) in [
                        (MarkerKind::Node, "Platforms"),
                        (MarkerKind::Track, "Tracks"),
                        (MarkerKind::Truth, "Truth (evaluation)"),
                    ] {
                        let rows: Vec<_> = shown.iter().filter(|m| m.kind == kind).collect();
                        if rows.is_empty() {
                            continue;
                        }
                        ui.label(egui::RichText::new(heading).strong());
                        for marker in rows {
                            let is_selected = selected == Some(marker.key);
                            if ui.selectable_label(is_selected, &*marker.label).clicked() {
                                ui_state.select_request =
                                    Some(crate::picking::SelectRequest::Select(marker.key));
                            }
                        }
                    }
                });
            if selected.is_some() && ui.button("Clear selection").clicked() {
                ui_state.select_request = Some(crate::picking::SelectRequest::Clear);
            }
        });
}

// ---------------------------------------------------------------------------
// Section: Selection Inspector
// ---------------------------------------------------------------------------

pub(super) fn section_selection(
    ui: &mut egui::Ui,
    selection: &SelectionState,
    max_stale_steps: u32,
) {
    ui.heading("Selection");
    let Some(selected) = selection.selected.as_ref() else {
        ui.label("Click a marker, press Tab, or pick one from the list.");
        return;
    };

    let position = selected.position;
    ui.strong(format!(
        "{}: {}",
        selected.details.kind_label(),
        selected.label
    ));
    ui.label(format!(
        "Position: x={:.1} y={:.1} z={:.1} m",
        position.x, position.y, position.z
    ));
    let vel = selected.velocity;
    let speed = vel.length();
    ui.label(format!(
        "Velocity: [{:.1}, {:.1}, {:.1}] m/s (speed {:.1} m/s)",
        vel.x, vel.y, vel.z, speed
    ));
    if speed > 0.1 {
        let heading_deg = vel.y.atan2(vel.x).to_degrees();
        ui.label(format!("Heading: {:.1}\u{00b0}", heading_deg));
    }

    match &selected.details {
        SelectionDetails::Track(track) => {
            if let Some(std_m) = track.measurement_std_m {
                kv_row(ui, "Meas. std", &format!("{:.2} m", std_m));
            }
            if let Some(updates) = track.update_count {
                kv_row(ui, "Update count", &format!("{}", updates));
            }
            if let Some(stale) = track.stale_steps {
                let color = if stale > max_stale_steps {
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
            if let Some(err) = track.track_error_m {
                kv_row(ui, "Track error", &format!("{:.2} m", err));
            }
            if let Some(diag) = &track.covariance_diag {
                ui.label(format!(
                    "Cov diag: \u{03c3}x={:.2} \u{03c3}y={:.2} \u{03c3}z={:.2} m",
                    diag[0].sqrt(),
                    diag[1].sqrt(),
                    diag[2].sqrt()
                ));
            }
            if let Some(p_cv) = track.mode_probability_cv {
                let p_ct = 1.0 - p_cv;
                kv_row(
                    ui,
                    "IMM blend",
                    &format!("CV {:.0}% / CT {:.0}%", p_cv * 100.0, p_ct * 100.0),
                );
            }
            if !track.contributing_nodes.is_empty() {
                kv_row(ui, "Updated by", &track.contributing_nodes.join(", "));
            }
        }
        SelectionDetails::Node(node) => {
            if let Some(health) = node.health {
                ui.horizontal(|ui| {
                    ui.label("Health:");
                    ui.colored_label(health_color_egui(health), format!("{:.0}%", health * 100.0));
                });
            }
            if let Some(sensor) = &node.sensor_type {
                kv_row(ui, "Sensor", sensor);
            }
            if let Some(mobile) = node.is_mobile {
                kv_row(ui, "Mobile", if mobile { "Yes" } else { "No" });
            }
            if let Some(fov) = node.fov_half_angle_deg {
                kv_row(
                    ui,
                    "FOV",
                    &format!("{:.0}\u{00b0} (half {:.0}\u{00b0})", fov * 2.0, fov),
                );
            }
            if let Some(range) = node.max_range_m {
                kv_row(ui, "Max range", &format!("{:.0} m", range));
            }
        }
        SelectionDetails::Truth(truth) => {
            if let Some(dist) = truth.nearest_track_dist_m {
                kv_row(ui, "Nearest track", &format!("{:.2} m", dist));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section: Safety & Alerts
// ---------------------------------------------------------------------------

pub(super) fn section_safety_alerts(
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
        if track.stale_steps > replay_state.max_stale_steps() {
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

    for event in &frame.safety_events {
        let state = if event.state.is_empty() {
            event.reason.as_str()
        } else {
            event.state.as_str()
        };
        alerts.push((
            if event.blocked {
                AlertLevel::Error
            } else {
                AlertLevel::Warning
            },
            format!(
                "Safety {}: {} [{}]",
                event.subject_id,
                state,
                event.violations.join(", ")
            ),
        ));
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

pub(super) fn section_targets(ui: &mut egui::Ui, replay_state: &ReplayState) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    ui.collapsing(format!("Targets ({})", frame.target_metadata.len()), |ui| {
        if frame.target_metadata.is_empty() {
            ui.label("No operational target metadata in this frame.");
            return;
        }
        egui::Grid::new("operational_target_table")
            .striped(true)
            .num_columns(5)
            .show(ui, |ui| {
                ui.strong("Priority");
                ui.strong("Target");
                ui.strong("Type");
                ui.strong("Status");
                ui.strong("Confidence");
                ui.end_row();
                let mut targets = frame.target_metadata.iter().collect::<Vec<_>>();
                targets.sort_by_key(|target| std::cmp::Reverse(target.priority));
                for target in targets {
                    ui.label(target.priority.to_string());
                    ui.label(if target.operator_label.is_empty() {
                        &target.target_id
                    } else {
                        &target.operator_label
                    });
                    ui.label(&target.target_type);
                    ui.label(&target.status);
                    ui.label(format!("{:.0}%", target.confidence * 100.0));
                    ui.end_row();
                }
            });
    });
}

#[derive(Clone, Copy)]
pub(super) enum AlertLevel {
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// Section: Mapping Status
// ---------------------------------------------------------------------------

pub(super) fn section_mapping_status(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn section_localization_status(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn section_inspection_events(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn section_frame_events(ui: &mut egui::Ui, replay_state: &ReplayState) {
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

pub(super) fn section_layers(
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

/// What the terrain rasters are doing, in words an operator can act on: why
/// belief is unavailable, and whether any tile failed to upload.
pub fn terrain_raster_status(
    report: &crate::terrain_rasters::RasterReport,
    belief: &crate::belief_render::BeliefLayer,
    land_cover: &crate::schema::LandCoverState,
) -> String {
    let belief_text = match belief {
        crate::belief_render::BeliefLayer::Unavailable(reason) => {
            format!("Belief unavailable: {reason}.")
        }
        crate::belief_render::BeliefLayer::Ready(snapshot) => {
            let (nx, ny) = snapshot.dimensions();
            format!(
                "Belief r{}, {nx} x {ny} cells; {} tile(s) uploaded, {} waiting, {} refused.",
                snapshot.revision().0,
                report.uploaded_tiles,
                report.waiting_tiles,
                report.refused_tiles
            )
        }
    };
    let land_cover_text = match land_cover {
        crate::schema::LandCoverState::Absent => "No land cover in this scene.".to_string(),
        crate::schema::LandCoverState::Ready(raster) => {
            format!("Land cover {} x {} cells.", raster.cols, raster.rows)
        }
        crate::schema::LandCoverState::Unavailable(reason) => {
            format!("Land cover unavailable: {reason}.")
        }
    };
    format!("{belief_text} {land_cover_text}")
}

// ---------------------------------------------------------------------------
// Section: Runtime Overlays
// ---------------------------------------------------------------------------

pub(super) fn section_runtime_overlays(
    ui: &mut egui::Ui,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
    raster_status: Option<&str>,
) {
    ui.collapsing("Runtime Overlays", |ui| {
        ui.checkbox(
            &mut runtime_visibility.diagnostic_glyphs,
            "Diagnostic Wire Glyphs",
        );
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
        ui.checkbox(&mut runtime_visibility.belief, "Belief (terrain)");
        ui.checkbox(&mut runtime_visibility.land_cover_tint, "Land-cover tint");
        if let Some(status) = raster_status {
            ui.small(status);
        }
        ui.checkbox(
            &mut runtime_visibility.inspection_events,
            "Inspection Events",
        );
        ui.checkbox(&mut runtime_visibility.launch_lines, "Launch Lines");
        ui.checkbox(&mut runtime_visibility.trails, "Trails");
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

pub(super) fn section_mission_zones(
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
