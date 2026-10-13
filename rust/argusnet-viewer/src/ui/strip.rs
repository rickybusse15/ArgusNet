//! The status strip (VIEWER_FEATURE_INTEGRATION §7.1, V1 of #225): the run's
//! headline figures for the current frame, in the status line in every
//! workspace. A figure the frame has no source for reads "unavailable" with
//! the reason, never zero.

use argusnet_replay::ReplayFrame;
use bevy_egui::egui;

/// One figure: its label, its value or why there is none, and what it means.
#[derive(Debug, Clone, PartialEq)]
pub struct StripCell {
    pub label: &'static str,
    pub value: Result<String, &'static str>,
    pub hint: &'static str,
}

impl StripCell {
    fn new(label: &'static str, value: Result<String, &'static str>, hint: &'static str) -> Self {
        Self { label, value, hint }
    }
}

/// The strip's figures for `frame`, in display order.
/// `live` frames come off the tracker wire, which carries no world
/// contacts, so their count is unavailable rather than zero.
pub fn strip_cells(frame: &ReplayFrame, live: bool) -> Vec<StripCell> {
    let metrics = &frame.metrics;
    let judged = metrics.accepted_observation_count + metrics.rejected_observation_count;
    let evidence_ages: Vec<f64> = frame
        .tracks
        .iter()
        .filter_map(|track| track.evidence_time_s)
        .map(|evidence| f64::from(frame.timestamp_s) - evidence)
        .collect();
    vec![
        StripCell::new(
            "Tracks",
            Ok(frame.tracks.len().to_string()),
            "Fused tracks published this frame",
        ),
        StripCell::new(
            "Error now",
            metrics
                .mean_error_m
                .map(|error| format!("{error:.1} m"))
                .ok_or("no track paired with a target this frame"),
            "Mean distance from each fused track to its target, this frame",
        ),
        StripCell::new(
            "Bearings",
            Ok(metrics.accepted_observation_count.to_string()),
            "Bearings fusion accepted this frame",
        ),
        StripCell::new(
            "Rejected",
            if judged == 0 {
                Err("no bearings reached fusion this frame")
            } else {
                Ok(format!(
                    "{:.0}%",
                    100.0 * f64::from(metrics.rejected_observation_count) / f64::from(judged)
                ))
            },
            "Share of the bearings that reached fusion and were rejected",
        ),
        StripCell::new(
            "Map known",
            frame
                .mapping_state
                .as_ref()
                .filter(|mapping| mapping.total_cells > 0)
                .map(|mapping| format!("{:.0}%", 100.0 * mapping.coverage_fraction))
                .ok_or("the frame carries no mapping state"),
            "Share of the map's cells the platforms have covered",
        ),
        StripCell::new(
            "Evidence age",
            if evidence_ages.is_empty() {
                Err("tracks record no evidence time (a run without delivery or rates)")
            } else {
                Ok(format!(
                    "{:.1} s",
                    evidence_ages.iter().sum::<f64>() / evidence_ages.len() as f64
                ))
            },
            "Mean time since the published tracks last took a measurement",
        ),
        StripCell::new(
            "Contacts",
            if live {
                Err("live frames do not carry world contacts")
            } else {
                Ok(frame.world_contacts.len().to_string())
            },
            "Platforms physics put back out of the ground or a structure this frame \
             (a replay written before contacts were recorded shows 0)",
        ),
    ]
}

/// Why bearings were not generated (§7.1): the frame's generation
/// rejections counted by reason, most first, ties by reason name.
pub fn rejection_breakdown(
    rejections: &[argusnet_replay::RejectedObservation],
) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for rejection in rejections {
        let reason = if rejection.reason.is_empty() {
            "unspecified"
        } else {
            rejection.reason.as_str()
        };
        *counts.entry(reason).or_default() += 1;
    }
    let mut rows: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(reason, count)| (reason.to_string(), count))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

/// Draw [`rejection_breakdown`] as labelled bars against the largest.
pub(super) fn draw_rejection_breakdown(
    ui: &mut egui::Ui,
    rejections: &[argusnet_replay::RejectedObservation],
) {
    let rows = rejection_breakdown(rejections);
    let Some(most) = rows.first().map(|(_, count)| *count) else {
        ui.label(egui::RichText::new("Every attempted bearing was generated").weak());
        return;
    };
    egui::Grid::new("rejection_breakdown")
        .num_columns(3)
        .show(ui, |ui| {
            for (reason, count) in &rows {
                ui.label(reason.replace('_', " "));
                ui.add(
                    egui::ProgressBar::new(*count as f32 / most as f32)
                        .desired_width(120.0)
                        .desired_height(8.0),
                );
                ui.monospace(count.to_string());
                ui.end_row();
            }
        });
}

/// A curve as runs of `[time, value]` points, broken where a frame has no
/// value or the recording has a gap.
pub type Curve = Vec<Vec<[f64; 2]>>;

/// The timeline's curves (§7.1): each frame's mean track error as a share
/// of the run's worst, and the share of the map known, both `0..=1` against
/// source time. A frame with no paired track or no mapping state leaves a
/// gap in that curve rather than a zero, and so does each recording gap in
/// `gaps` (the timeline's, as `(from, to)` source times). Also returns the
/// worst error, for the legend.
pub fn metric_curves(frames: &[ReplayFrame], gaps: &[(f32, f32)]) -> (Curve, Curve, Option<f32>) {
    let worst = frames
        .iter()
        .filter_map(|frame| frame.metrics.mean_error_m)
        .fold(None, |most: Option<f32>, error| {
            Some(most.map_or(error, |m| m.max(error)))
        });
    let runs = |value: &dyn Fn(&ReplayFrame) -> Option<f64>| {
        let mut runs: Curve = Vec::new();
        let mut current: Vec<[f64; 2]> = Vec::new();
        // Both in time order, so one pass over each.
        let mut gap_ends = gaps.iter().map(|(_, to)| *to).peekable();
        for frame in frames {
            while gap_ends.next_if(|to| *to < frame.timestamp_s).is_some() {}
            if gap_ends.next_if(|to| *to == frame.timestamp_s).is_some() && !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            match value(frame) {
                Some(y) => current.push([f64::from(frame.timestamp_s), y]),
                None if !current.is_empty() => runs.push(std::mem::take(&mut current)),
                None => {}
            }
        }
        if !current.is_empty() {
            runs.push(current);
        }
        runs
    };
    let error = runs(&|frame| {
        let worst = f64::from(worst?);
        let error = f64::from(frame.metrics.mean_error_m?);
        Some(if worst > 0.0 { error / worst } else { 0.0 })
    });
    let known = runs(&|frame| {
        frame
            .mapping_state
            .as_ref()
            .filter(|mapping| mapping.total_cells > 0)
            .map(|mapping| f64::from(mapping.coverage_fraction))
    });
    (error, known, worst)
}

/// Draw the strip into the status line.
pub(super) fn draw(ui: &mut egui::Ui, frame: &ReplayFrame, live: bool) {
    for cell in strip_cells(frame, live) {
        ui.separator();
        match &cell.value {
            Ok(value) => {
                ui.label(egui::RichText::new(cell.label).weak())
                    .on_hover_text(cell.hint);
                ui.monospace(value).on_hover_text(cell.hint);
            }
            Err(reason) => {
                ui.label(egui::RichText::new(cell.label).weak())
                    .on_hover_text(cell.hint);
                ui.label(egui::RichText::new("unavailable").weak().italics())
                    .on_hover_text(*reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(json: serde_json::Value) -> ReplayFrame {
        serde_json::from_value(json).expect("a frame")
    }

    fn value<'a>(cells: &'a [StripCell], label: &str) -> &'a Result<String, &'static str> {
        &cells.iter().find(|cell| cell.label == label).unwrap().value
    }

    /// A frame with nothing to say reads unavailable, with a reason, never
    /// zero; counts that are genuinely zero read zero.
    #[test]
    fn a_figure_without_a_source_is_unavailable_not_zero() {
        let cells = strip_cells(&frame(serde_json::json!({ "timestamp_s": 1.0 })), false);
        assert_eq!(value(&cells, "Tracks"), &Ok("0".to_string()));
        for label in ["Error now", "Rejected", "Map known", "Evidence age"] {
            assert!(value(&cells, label).is_err(), "{label}");
        }
    }

    #[test]
    fn rejections_are_counted_by_reason_most_first() {
        let rejection = |reason: &str| {
            serde_json::from_value::<argusnet_replay::RejectedObservation>(serde_json::json!({
                "node_id": "n", "target_id": "t", "reason": reason
            }))
            .unwrap()
        };
        let rows = rejection_breakdown(&[
            rejection("terrain_occlusion"),
            rejection("dropout"),
            rejection("terrain_occlusion"),
            rejection("noise"),
            rejection(""),
        ]);
        assert_eq!(
            rows,
            vec![
                ("terrain_occlusion".to_string(), 2),
                ("dropout".to_string(), 1),
                ("noise".to_string(), 1),
                ("unspecified".to_string(), 1),
            ]
        );
        assert!(rejection_breakdown(&[]).is_empty());
    }

    /// Curves are scaled to `0..=1` and broken where a frame has no value.
    #[test]
    fn curves_are_scaled_and_broken_at_missing_values() {
        let frames: Vec<ReplayFrame> = [
            serde_json::json!({ "timestamp_s": 0.0, "metrics": { "mean_error_m": 2.0 } }),
            serde_json::json!({ "timestamp_s": 1.0 }),
            serde_json::json!({ "timestamp_s": 2.0, "metrics": { "mean_error_m": 4.0 },
                "mapping_state": { "coverage_fraction": 0.5, "covered_cells": 1, "total_cells": 2 } }),
        ]
        .into_iter()
        .map(frame)
        .collect();
        let (error, known, worst) = metric_curves(&frames, &[]);
        assert_eq!(worst, Some(4.0));
        assert_eq!(error, vec![vec![[0.0, 0.5]], vec![[2.0, 1.0]]]);
        assert_eq!(known, vec![vec![[2.0, 0.5]]]);
    }

    /// A recording gap breaks a curve even where both sides have a value,
    /// so missing history never draws as measured.
    #[test]
    fn curves_break_at_recording_gaps() {
        let frames: Vec<ReplayFrame> = [0.0, 1.0, 9.0, 10.0]
            .into_iter()
            .map(|t| {
                frame(serde_json::json!({ "timestamp_s": t, "metrics": { "mean_error_m": 1.0 } }))
            })
            .collect();
        let (error, _, _) = metric_curves(&frames, &[(1.0, 9.0)]);
        assert_eq!(
            error,
            vec![vec![[0.0, 1.0], [1.0, 1.0]], vec![[9.0, 1.0], [10.0, 1.0]]]
        );
    }

    #[test]
    fn the_figures_come_from_the_frame() {
        let frame = frame(serde_json::json!({
            "timestamp_s": 10.0,
            "tracks": [
                { "track_id": "t1", "position": [0, 0, 0], "velocity": [0, 0, 0], "evidence_time_s": 9.0 },
                { "track_id": "t2", "position": [0, 0, 0], "velocity": [0, 0, 0], "evidence_time_s": 7.0 }
            ],
            "metrics": {
                "mean_error_m": 4.25,
                "accepted_observation_count": 6,
                "rejected_observation_count": 2
            },
            "mapping_state": { "coverage_fraction": 0.375, "covered_cells": 3, "total_cells": 8 }
        }));
        let live = strip_cells(&frame, true);
        let cells = strip_cells(&frame, false);
        assert_eq!(value(&cells, "Tracks"), &Ok("2".to_string()));
        assert_eq!(value(&cells, "Error now"), &Ok("4.2 m".to_string()));
        assert_eq!(value(&cells, "Bearings"), &Ok("6".to_string()));
        assert_eq!(value(&cells, "Rejected"), &Ok("25%".to_string()));
        assert_eq!(value(&cells, "Map known"), &Ok("38%".to_string()));
        assert_eq!(value(&cells, "Evidence age"), &Ok("2.0 s".to_string()));
        assert_eq!(value(&cells, "Contacts"), &Ok("0".to_string()));
        // The tracker wire carries no contacts: unavailable, not zero.
        assert!(value(&live, "Contacts").is_err());
    }
}
