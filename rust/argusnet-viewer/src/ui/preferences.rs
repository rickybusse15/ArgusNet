//! Preferences (C7.2, VIEWER_UI §13): UI scale independent of the world's,
//! reduced motion, and the key reference with conflict-checked remapping.
//! The key table is the live bindings, not a copy, so it cannot drift from
//! what the keys do. Every change is saved to the preferences file.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::bindings::{key_name, Action, Bindings};
use crate::camera_control::CameraControl;
use crate::prefs::{ViewerPrefs, UI_SCALES};
use crate::state::ViewerUiState;

/// What the window is doing with the keyboard.
#[derive(Resource, Debug, Default, Clone, PartialEq)]
pub struct KeyCapture {
    /// The action waiting for a key.
    pub action: Option<Action>,
    /// The last rebinding's outcome, for the window.
    pub message: Option<String>,
}

/// The next offered UI scale up (`steps` > 0) or down from `current`.
pub fn step_ui_scale(current: f32, steps: i32) -> f32 {
    let index = UI_SCALES
        .iter()
        .position(|scale| (scale - current).abs() < 1e-3)
        .unwrap_or(2) as i32;
    UI_SCALES[(index + steps).clamp(0, UI_SCALES.len() as i32 - 1) as usize]
}

/// Bind `key` to the action being captured; Escape cancels. Returns whether
/// the bindings changed; the outcome is left in `capture.message`.
pub fn capture_key(bindings: &mut Bindings, capture: &mut KeyCapture, key: KeyCode) -> bool {
    let Some(action) = capture.action else {
        return false;
    };
    capture.action = None;
    if key == KeyCode::Escape {
        capture.message = Some("Cancelled".into());
        return false;
    }
    match bindings.rebind(action, key) {
        Ok(()) => {
            capture.message = Some(format!("{} is now {}", action.name(), key_name(key)));
            true
        }
        Err(problem) => {
            capture.message = Some(format!("Not changed: {problem}"));
            false
        }
    }
}

/// The preferences window, when open.
#[allow(clippy::too_many_arguments)]
pub fn preferences_ui_system(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<ViewerUiState>,
    mut bindings: ResMut<Bindings>,
    mut prefs: ResMut<ViewerPrefs>,
    mut capture: ResMut<KeyCapture>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    // A captured key: the gate held every other use of it this frame.
    if capture.action.is_some() {
        if let Some(key) = keys.get_just_pressed().next().copied() {
            if capture_key(&mut bindings, &mut capture, key) {
                if let Err(problem) = prefs.set_bindings(bindings.overrides()) {
                    capture.message = Some(format!("Bound, but not saved: {problem}"));
                }
            }
        }
    }
    let capturing = capture.action.is_some();
    if ui_state.capturing_key != capturing {
        ui_state.capturing_key = capturing;
    }
    if !ui_state.show_preferences {
        return;
    }
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    if !crate::theme::fonts_ready(context) {
        return;
    }
    let mut open = true;
    let mut scale = ui_state.ui_scale;
    let mut reduced = ui_state.reduced_motion;
    let mut reset = false;
    egui::Window::new("Preferences")
        .open(&mut open)
        .default_width(380.0)
        .show(context, |ui| {
            ui.strong("Display");
            ui.horizontal(|ui| {
                ui.label("UI scale");
                if ui.small_button("−").clicked() {
                    scale = step_ui_scale(scale, -1);
                }
                ui.label(format!("{:.0}%", scale * 100.0));
                if ui.small_button("+").clicked() {
                    scale = step_ui_scale(scale, 1);
                }
            });
            ui.label(egui::RichText::new("The interface only; the world keeps its scale.").small());
            ui.checkbox(&mut reduced, "Reduce motion")
                .on_hover_text("Camera moves arrive at once; interface animations are off.");
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Keys");
                if ui.small_button("Reset to defaults").clicked() {
                    reset = true;
                }
            });
            if let Some(message) = &capture.message {
                ui.label(egui::RichText::new(message).small());
            }
            egui::ScrollArea::vertical()
                .max_height(320.0)
                .show(ui, |ui| {
                    egui::Grid::new("key_table").striped(true).show(ui, |ui| {
                        for action in Action::ALL {
                            ui.label(action.name());
                            if capture.action == Some(action) {
                                ui.label("press a key… (Esc cancels)");
                            } else {
                                ui.monospace(key_name(bindings.key(action)));
                            }
                            if ui.small_button("Rebind").clicked() {
                                capture.action = Some(action);
                                capture.message = None;
                            }
                            ui.end_row();
                        }
                    });
                });
        });
    if reset {
        *bindings = Bindings::default();
        capture.message = match prefs.set_bindings(bindings.overrides()) {
            Ok(()) => Some("Keys reset to defaults".into()),
            Err(problem) => Some(format!("Reset, but not saved: {problem}")),
        };
    }
    if !open {
        ui_state.show_preferences = false;
        capture.action = None;
    }
    if scale != ui_state.ui_scale || reduced != ui_state.reduced_motion {
        ui_state.ui_scale = scale;
        ui_state.reduced_motion = reduced;
        if let Err(problem) = prefs.set_accessibility(scale, reduced) {
            warn!("could not save preferences: {problem}");
        }
    }
}

/// Apply UI scale and reduced motion to egui and the camera; on the first
/// run, seed them from the preferences file.
pub fn apply_accessibility_system(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<ViewerUiState>,
    prefs: Res<ViewerPrefs>,
    mut control: ResMut<CameraControl>,
    mut seeded: Local<bool>,
    mut animations_reduced: Local<Option<bool>>,
) {
    if !*seeded {
        *seeded = true;
        ui_state.ui_scale = prefs.ui_scale();
        ui_state.reduced_motion = prefs.file.reduced_motion.unwrap_or(false);
    }
    if control.reduced_motion != ui_state.reduced_motion {
        control.reduced_motion = ui_state.reduced_motion;
    }
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    if (context.zoom_factor() - ui_state.ui_scale).abs() > 1e-4 {
        context.set_zoom_factor(ui_state.ui_scale);
    }
    if *animations_reduced != Some(ui_state.reduced_motion) {
        *animations_reduced = Some(ui_state.reduced_motion);
        let animation_time = if ui_state.reduced_motion {
            0.0
        } else {
            egui::Style::default().animation_time
        };
        context.all_styles_mut(|style| style.animation_time = animation_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rebinding_that_clashes_is_refused_and_says_why() {
        let mut bindings = Bindings::default();
        let mut capture = KeyCapture {
            action: Some(Action::Focus),
            message: None,
        };
        // Space is play/pause.
        assert!(!capture_key(&mut bindings, &mut capture, KeyCode::Space));
        assert_eq!(bindings.key(Action::Focus), KeyCode::KeyF);
        let message = capture.message.clone().unwrap();
        assert!(message.contains("play_pause"), "{message}");
        assert_eq!(capture.action, None, "the capture ends either way");
    }

    #[test]
    fn a_free_key_binds_and_is_saved_as_an_override() {
        let mut bindings = Bindings::default();
        let mut capture = KeyCapture {
            action: Some(Action::Focus),
            message: None,
        };
        assert!(capture_key(&mut bindings, &mut capture, KeyCode::KeyZ));
        assert_eq!(bindings.key(Action::Focus), KeyCode::KeyZ);
        let overrides = bindings.overrides();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides["focus"], "KeyZ");
        // What is saved loads back as the same bindings.
        let (loaded, problems) = Bindings::with_overrides(&overrides);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(loaded, bindings);
    }

    #[test]
    fn escape_cancels_a_capture() {
        let mut bindings = Bindings::default();
        let mut capture = KeyCapture {
            action: Some(Action::Focus),
            message: None,
        };
        assert!(!capture_key(&mut bindings, &mut capture, KeyCode::Escape));
        assert_eq!(bindings, Bindings::default());
    }

    #[test]
    fn every_default_key_has_a_name_the_preferences_file_reads() {
        let bindings = Bindings::default();
        for action in Action::ALL {
            let key = bindings.key(action);
            assert_eq!(
                crate::bindings::key_from_name(&key_name(key)),
                Some(key),
                "{action:?}"
            );
        }
    }

    #[test]
    fn alerts_are_ordered_by_severity_stably() {
        use crate::ui::inspectors::{order_by_severity, AlertLevel};
        let mut alerts = vec![
            (AlertLevel::Warning, "w1".to_string()),
            (AlertLevel::Error, "e1".to_string()),
            (AlertLevel::Warning, "w2".to_string()),
            (AlertLevel::Error, "e2".to_string()),
        ];
        order_by_severity(&mut alerts);
        let order: Vec<&str> = alerts.iter().map(|(_, text)| text.as_str()).collect();
        assert_eq!(order, ["e1", "e2", "w1", "w2"]);
    }

    #[test]
    fn ui_scale_steps_stay_within_the_offered_range() {
        assert_eq!(step_ui_scale(1.0, 1), 1.125);
        assert_eq!(step_ui_scale(1.0, -1), 0.875);
        assert_eq!(step_ui_scale(2.0, 1), 2.0);
        assert_eq!(step_ui_scale(0.75, -3), 0.75);
    }

    #[test]
    fn the_canvas_test_converts_window_pixels_to_ui_points() {
        let mut ui = ViewerUiState {
            canvas_rect: Some([100.0, 50.0, 500.0, 400.0]),
            ..Default::default()
        };
        assert!(ui.on_canvas([450.0, 350.0]));
        ui.ui_scale = 1.5;
        // 450 px is 300 points: inside; 900 px is 600 points: outside.
        assert!(ui.on_canvas([450.0, 350.0]));
        assert!(!ui.on_canvas([900.0, 350.0]));
        assert!(
            !ui.on_canvas([120.0, 350.0]),
            "80 points is left of the canvas"
        );
    }

    #[test]
    fn reduced_motion_survives_the_preferences_file() {
        let mut prefs = ViewerPrefs::default();
        prefs.set_accessibility(1.25, true).unwrap();
        let text = serde_json::to_string(&prefs.file).unwrap();
        let file: crate::prefs::PrefsFile = serde_json::from_str(&text).unwrap();
        assert_eq!(file.ui_scale, Some(1.25));
        assert_eq!(file.reduced_motion, Some(true));
        let loaded = ViewerPrefs {
            file,
            ..Default::default()
        };
        assert_eq!(loaded.ui_scale(), 1.25);
    }
}
