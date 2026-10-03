//! Small shared widgets and formatting helpers.

use super::*;

pub(super) fn egress_arrived_count(egress_progress: &[EgressProgress]) -> usize {
    egress_progress
        .iter()
        .filter(|progress| progress.distance_to_home_m < EGRESS_ARRIVAL_THRESHOLD_M)
        .count()
}

pub(super) fn kv_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.strong(value);
    });
}

/// Color for a health fraction (0.0 = red, 0.3 = yellow, 0.7+ = green).
pub fn health_color(health: f32) -> egui::Color32 {
    health_color_egui(health)
}

pub(super) fn health_color_egui(health: f32) -> egui::Color32 {
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

pub(super) fn format_group_chips(group: &crate::state::ZoneOverlapGroup) -> String {
    let mut counts: Vec<_> = group.type_counts.iter().collect();
    counts.sort_by(|a, b| a.0.cmp(b.0));
    counts
        .into_iter()
        .map(|(zone_type, count)| format!("{}{}", zone_type_letter(zone_type), count))
        .collect::<Vec<_>>()
        .join("  ")
}

pub(super) fn zone_type_icon(zone_type: &str) -> &'static str {
    match zone_type {
        "surveillance" => "[S]",
        "exclusion" => "[X]",
        "patrol" => "[P]",
        "objective" => "[O]",
        _ => "[?]",
    }
}

pub(super) fn zone_type_letter(zone_type: &str) -> &'static str {
    match zone_type {
        "surveillance" => "S",
        "exclusion" => "X",
        "patrol" => "P",
        "objective" => "O",
        _ => "?",
    }
}

pub(super) fn badge_pill_width(badge: &crate::mission_zones::ProjectedZoneBadge) -> f32 {
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

pub(super) fn combo_box(
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
