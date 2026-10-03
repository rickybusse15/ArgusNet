//! The timeline: playback and operator timelines, phase markers and event jumps.

use super::*;

/// The single source-time control surface (P2.4a). Scrubbing is over source
/// time, not frame number, so an irregular replay scrubs as recorded.
pub(super) fn section_operator_timeline(
    ui: &mut egui::Ui,
    replay_state: &mut ReplayState,
    ui_state: &mut ViewerUiState,
) {
    let Some((first_s, last_s)) = replay_state.time_range() else {
        ui.label("No replay packaged with this scene.");
        return;
    };

    let total_frames = replay_state.frame_count();
    let now_s = replay_state.time_s();
    let gaps = replay_state.gaps();
    // Row 1: the time controls (the timeline is the one place they live),
    // where the cursor is, and what cannot be seen.
    ui.horizontal(|ui| {
        playback_buttons(ui, replay_state);
        ui.separator();
        ui.strong(if replay_state.live {
            "Live history (read-only)"
        } else {
            "Replay"
        });
        ui.monospace(format!(
            "t {:.2} s of {:.2}–{:.2} s · frame {} / {}",
            now_s,
            first_s,
            last_s,
            replay_state.frame_index + 1,
            total_frames
        ));
        ui.separator();
        ui.label(format!("speed {:.2}x", replay_state.playback_speed));
        ui.add(
            egui::Slider::new(&mut replay_state.playback_speed, 0.25..=16.0)
                .show_value(false)
                .step_by(0.25)
                .clamping(egui::SliderClamping::Always),
        );
        if !replay_state.live {
            ui.checkbox(&mut replay_state.looping, "Loop");
        }
        // What cannot be seen, said as such: trimmed live history and
        // recording gaps.
        if let Some(trimmed) = replay_state.trimmed_before_s() {
            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(210, 160, 90),
                format!("History before {trimmed:.1} s was trimmed"),
            );
        }
        if !gaps.is_empty() {
            let missing: f32 = gaps.iter().map(|(a, b)| b - a).sum();
            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(210, 160, 90),
                format!(
                    "{} recording gap(s), {missing:.1} s unavailable",
                    gaps.len()
                ),
            );
        }
    });

    // Row 2: the scrubber, full width. Seeking reviews recorded frames; it
    // never rewinds a runtime.
    let mut time_value = now_s;
    let scrub = ui.scope(|ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(80.0);
        ui.add_enabled(
            last_s > first_s,
            egui::Slider::new(&mut time_value, first_s..=last_s)
                .text("s")
                .clamping(egui::SliderClamping::Always),
        )
    });
    let scrub = scrub.inner.on_hover_text(if replay_state.live {
        "Reviewing live history never pauses or rewinds the runtime."
    } else {
        "Seeking a replay is local; it changes no runtime."
    });
    if scrub.changed() {
        replay_state.playing = false;
        replay_state.seek_time(time_value);
        ui_state.selected_timeline_frame = Some(replay_state.frame_index);
    }

    if let Some(document) = &replay_state.document {
        let event_points: PlotPoints = document
            .frames
            .iter()
            .map(|frame| {
                [
                    f64::from(frame.timestamp_s),
                    crate::replay::frame_event_count(frame) as f64,
                ]
            })
            .collect();
        let obs_points: PlotPoints = document
            .frames
            .iter()
            .map(|frame| {
                [
                    f64::from(frame.timestamp_s),
                    frame.metrics.observation_count as f64,
                ]
            })
            .collect();
        let peak = document
            .frames
            .iter()
            .map(|frame| frame.metrics.observation_count as f64)
            .fold(1.0, f64::max);
        let phase_transitions = phase_transitions(document);
        // Row 3: what happened over time, in whatever height is left.
        Plot::new("operator_timeline_plot")
            .height(ui.available_height().max(16.0))
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, true])
            .label_formatter(|_| None)
            .show(ui, |plot_ui| {
                for (from, to) in &gaps {
                    let (from, to) = (f64::from(*from), f64::from(*to));
                    plot_ui.polygon(
                        Polygon::new(
                            "",
                            PlotPoints::from(vec![
                                [from, 0.0],
                                [to, 0.0],
                                [to, peak],
                                [from, peak],
                            ]),
                        )
                        .fill_color(egui::Color32::from_rgba_unmultiplied(210, 160, 90, 40)),
                    );
                }
                plot_ui.line(
                    Line::new("", obs_points)
                        .color(egui::Color32::from_rgb(95, 150, 220))
                        .width(1.5_f32),
                );
                plot_ui.line(
                    Line::new("", event_points)
                        .color(egui::Color32::from_rgb(245, 180, 70))
                        .width(1.5_f32),
                );
                for (time_s, phase) in &phase_transitions {
                    plot_ui.vline(
                        VLine::new("", f64::from(*time_s))
                            .color(phase_vline_color(phase.as_str()))
                            .width(2.0_f32),
                    );
                }
                plot_ui.vline(
                    VLine::new("", f64::from(now_s))
                        .color(egui::Color32::WHITE)
                        .width(2.0_f32),
                );
            });
    }
}

pub(super) fn jump_to_event_controls(
    ui: &mut egui::Ui,
    replay_state: &mut ReplayState,
    ui_state: &mut ViewerUiState,
) {
    let events = replay_state.event_times();
    if events.is_empty() {
        ui.label("No frame events in this replay.");
        return;
    }

    ui.horizontal(|ui| {
        if ui.button("Previous Event").on_hover_text("[").clicked()
            && replay_state.jump_to_event(false)
        {
            ui_state.selected_timeline_frame = Some(replay_state.frame_index);
        }
        if ui.button("Next Event").on_hover_text("]").clicked() && replay_state.jump_to_event(true)
        {
            ui_state.selected_timeline_frame = Some(replay_state.frame_index);
        }
        ui.label(format!("{} event frames", events.len()));
    });
}

/// Where the scan mission's phase changes, by source time.
pub(super) fn phase_transitions(document: &crate::replay::ReplayDocument) -> Vec<(f32, String)> {
    let mut transitions = Vec::new();
    let mut last_phase = String::new();
    for (idx, frame) in document.frames.iter().enumerate() {
        if let Some(mission) = &frame.scan_mission_state {
            if mission.phase != last_phase {
                if idx > 0 {
                    transitions.push((frame.timestamp_s, mission.phase.clone()));
                }
                last_phase = mission.phase.clone();
            }
        }
    }
    transitions
}

pub(super) fn section_playback_timeline(ui: &mut egui::Ui, replay_state: &ReplayState) {
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
        let line = Line::new("", points);

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
            .label_formatter(|_| None)
            .show(ui, |plot_ui| {
                plot_ui.line(line);
                for (frame_idx, phase) in &phase_transitions {
                    let color = phase_vline_color(phase.as_str());
                    plot_ui.vline(
                        VLine::new("", *frame_idx as f64)
                            .color(color)
                            .width(2.0_f32),
                    );
                }
                plot_ui.vline(
                    VLine::new("", current_frame)
                        .color(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180))
                        .width(1.5_f32),
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

/// Returns (label, color) pair for a mission phase string.
pub(super) fn phase_color_pair(phase: &str) -> (&'static str, egui::Color32) {
    match phase {
        "scanning" => ("SCANNING", egui::Color32::from_rgb(80, 140, 255)),
        "localizing" => ("LOCALIZING", egui::Color32::from_rgb(255, 210, 60)),
        "inspecting" => ("INSPECTING", egui::Color32::from_rgb(255, 140, 40)),
        "egress" => ("EGRESS", egui::Color32::from_rgb(255, 160, 30)),
        "complete" => ("COMPLETE", egui::Color32::from_rgb(60, 200, 80)),
        _ => ("UNKNOWN", egui::Color32::GRAY),
    }
}

pub(super) fn phase_vline_color(phase: &str) -> egui::Color32 {
    match phase {
        "localizing" => egui::Color32::from_rgba_unmultiplied(255, 210, 60, 200),
        "inspecting" => egui::Color32::from_rgba_unmultiplied(255, 140, 40, 200),
        "egress" => egui::Color32::from_rgba_unmultiplied(255, 160, 30, 200),
        "complete" => egui::Color32::from_rgba_unmultiplied(60, 200, 80, 200),
        _ => egui::Color32::from_rgba_unmultiplied(120, 120, 120, 150),
    }
}

pub(super) fn phase_label_color(phase: &str) -> egui::Color32 {
    match phase {
        "localizing" => egui::Color32::from_rgb(255, 210, 60),
        "inspecting" => egui::Color32::from_rgb(255, 140, 40),
        "egress" => egui::Color32::from_rgb(255, 160, 30),
        "complete" => egui::Color32::from_rgb(60, 200, 80),
        _ => egui::Color32::GRAY,
    }
}
