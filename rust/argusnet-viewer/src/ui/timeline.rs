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

    let now_s = replay_state.time_s();
    let gaps = replay_state.gaps();
    // Row 1: the transport (the timeline is the one place time is
    // controlled, C4), where the cursor is, and what cannot be seen.
    ui.horizontal(|ui| {
        playback_buttons(ui, replay_state);
        ui.separator();
        let (source, recording) = if replay_state.live {
            (
                "Live history",
                "Read-only review of retained live history. The viewer does not record (D3).",
            )
        } else {
            (
                "Replay",
                "A recorded replay. The viewer does not record (D3).",
            )
        };
        ui.strong(source).on_hover_text(recording);
        if let Some(readout) = time_readout(replay_state) {
            ui.monospace(readout);
        }
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

    let marks = replay_state
        .document
        .as_ref()
        .map(timeline_marks)
        .unwrap_or_default();

    // Row 2: the scrubber, then what the current frame holds (the selected
    // event summary). Seeking reviews recorded frames; it never rewinds a
    // runtime.
    let mut time_value = now_s;
    ui.horizontal(|ui| {
        let scrub = ui.scope(|ui| {
            ui.spacing_mut().slider_width = (ui.available_width() - 420.0).max(80.0);
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
        let here = marks_in_frame(&marks, replay_state.frame_index);
        let summary = match here.as_slice() {
            [] => "No events in this frame".to_string(),
            [only] => format!("{}: {}", only.group.label(), only.label),
            [first, rest @ ..] => {
                format!(
                    "{}: {} (+{} more)",
                    first.group.label(),
                    first.label,
                    rest.len()
                )
            }
        };
        let all = here
            .iter()
            .map(|mark| format!("{}: {}", mark.group.label(), mark.label))
            .collect::<Vec<_>>()
            .join("\n");
        ui.label(summary).on_hover_text(if all.is_empty() {
            "No events in this frame.".to_string()
        } else {
            all
        });
    });

    // Row 3: the event marks, one lane per kind (mission at the top), in
    // whatever height is left. Gaps are shaded; the cursor is the white line.
    let lane = |group: MarkGroup| {
        (MarkGroup::ALL.len() - 1 - MarkGroup::ALL.iter().position(|g| *g == group).unwrap_or(0))
            as f64
    };
    // The lanes' legend, with counts: on hover, since 100 px holds no
    // captions for five lanes.
    let legend = MarkGroup::ALL
        .iter()
        .map(|group| {
            let count = marks.iter().filter(|mark| mark.group == *group).count();
            match (count, group.unavailable_in_replays()) {
                (0, Some(why)) => format!("{}: none ({why})", group.label()),
                _ => format!("{}: {count}", group.label()),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let plot = Plot::new("operator_timeline_plot")
        .height(ui.available_height().max(16.0))
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .show_axes([false, false])
        .show_grid(false)
        .include_x(f64::from(first_s))
        .include_x(f64::from(last_s))
        .include_y(-0.6)
        .include_y(MarkGroup::ALL.len() as f64 - 0.4)
        .label_formatter(|_| None)
        .show(ui, |plot_ui| {
            let top = MarkGroup::ALL.len() as f64 - 0.4;
            for (from, to) in &gaps {
                let (from, to) = (f64::from(*from), f64::from(*to));
                plot_ui.polygon(
                    Polygon::new(
                        "",
                        PlotPoints::from(vec![[from, -0.6], [to, -0.6], [to, top], [from, top]]),
                    )
                    .fill_color(egui::Color32::from_rgba_unmultiplied(210, 160, 90, 40)),
                );
            }
            for group in MarkGroup::ALL {
                let y = lane(group);
                let points: Vec<[f64; 2]> = marks
                    .iter()
                    .filter(|mark| mark.group == group)
                    .map(|mark| [f64::from(mark.time_s), y])
                    .collect();
                if !points.is_empty() {
                    plot_ui.points(Points::new("", points).radius(3.0).color(mark_color(group)));
                }
            }
            plot_ui.vline(
                VLine::new("", f64::from(now_s))
                    .color(egui::Color32::WHITE)
                    .width(1.5_f32),
            );
        });
    plot.response
        .on_hover_text(format!("Event lanes, top to bottom:\n{legend}"));
}

/// One colour per lane; safety and fault are the alarming ones.
fn mark_color(group: MarkGroup) -> egui::Color32 {
    match group {
        MarkGroup::Mission => egui::Color32::from_rgb(95, 150, 220),
        MarkGroup::Safety => egui::Color32::from_rgb(232, 106, 92),
        MarkGroup::Command => egui::Color32::from_rgb(165, 170, 172),
        MarkGroup::Evidence => egui::Color32::from_rgb(120, 200, 140),
        MarkGroup::Fault => egui::Color32::from_rgb(210, 173, 98),
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
