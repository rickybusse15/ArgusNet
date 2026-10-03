//! Workspace, projection and layer preset: three distinct controls (C2.3,
//! VIEWER_UI §4.1). Pointer and keyboard both go through
//! [`apply_shell_command`], so every pointer action has a keyboard path and
//! neither can reach the selection: it is not among the arguments.

use super::*;
use crate::state::{Projection, Workspace};

/// A change to the shell's own presentation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellCommand {
    Workspace(Workspace),
    Projection(Projection),
    Preset(MapLayerPreset),
    CycleWorkspace,
    CycleProjection,
    CyclePreset,
    ToggleToolPanel,
}

/// Apply `command`. An unavailable projection is refused, so the shell never
/// claims a projection it cannot draw. Returns whether anything changed.
pub fn apply_shell_command(
    command: ShellCommand,
    ui_state: &mut ViewerUiState,
    runtime_visibility: &mut RuntimeOverlayVisibility,
    mission_overlay: &mut MissionOverlaySettings,
) -> bool {
    let before = (
        ui_state.workspace,
        ui_state.projection,
        ui_state.map_layer_preset,
        ui_state.drawer_open,
    );
    match command {
        ShellCommand::Workspace(workspace) => ui_state.workspace = workspace,
        ShellCommand::CycleWorkspace => {
            ui_state.workspace = next_of(&Workspace::ALL, ui_state.workspace, |_| true);
        }
        ShellCommand::Projection(projection) => {
            if projection.unavailable_until().is_none() {
                ui_state.projection = projection;
            }
        }
        ShellCommand::CycleProjection => {
            ui_state.projection = next_of(&Projection::ALL, ui_state.projection, |p| {
                p.unavailable_until().is_none()
            });
        }
        ShellCommand::Preset(preset) => {
            ui_state.map_layer_preset = preset;
            apply_layer_preset(preset, runtime_visibility, mission_overlay);
        }
        ShellCommand::CyclePreset => {
            let preset = next_of(&MapLayerPreset::ALL, ui_state.map_layer_preset, |_| true);
            ui_state.map_layer_preset = preset;
            apply_layer_preset(preset, runtime_visibility, mission_overlay);
        }
        ShellCommand::ToggleToolPanel => ui_state.drawer_open = !ui_state.drawer_open,
    }
    before
        != (
            ui_state.workspace,
            ui_state.projection,
            ui_state.map_layer_preset,
            ui_state.drawer_open,
        )
}

/// The next of `all` after `current` that `allowed` accepts, wrapping;
/// `current` if none is.
fn next_of<T: Copy + PartialEq>(all: &[T], current: T, allowed: impl Fn(T) -> bool) -> T {
    let start = all.iter().position(|item| *item == current).unwrap_or(0);
    (1..=all.len())
        .map(|step| all[(start + step) % all.len()])
        .find(|item| allowed(*item))
        .unwrap_or(current)
}

/// The three controls, side by side in the header.
pub(super) fn shell_controls(ui: &mut egui::Ui, ui_state: &ViewerUiState) -> Option<ShellCommand> {
    let mut command = None;
    for workspace in Workspace::ALL {
        if ui
            .selectable_label(ui_state.workspace == workspace, workspace.label())
            .on_hover_text("Workspace (O cycles)")
            .clicked()
        {
            command = Some(ShellCommand::Workspace(workspace));
        }
    }
    ui.separator();
    for projection in Projection::ALL {
        let available = projection.unavailable_until().is_none();
        let response = ui.add_enabled(
            available,
            egui::Button::selectable(ui_state.projection == projection, projection.label()),
        );
        let response = match projection.unavailable_until() {
            Some(why) => response.on_disabled_hover_text(why),
            None => response.on_hover_text("Projection (V cycles)"),
        };
        if response.clicked() {
            command = Some(ShellCommand::Projection(projection));
        }
    }
    ui.separator();
    for preset in MapLayerPreset::ALL {
        if ui
            .selectable_label(ui_state.map_layer_preset == preset, preset.label())
            .on_hover_text("Layer preset (P cycles)")
            .clicked()
        {
            command = Some(ShellCommand::Preset(preset));
        }
    }
    command
}

/// What a workspace without its contract shows instead of mock data.
pub(super) fn workspace_unavailable(ui: &mut egui::Ui, workspace: Workspace) -> bool {
    let Some(why) = workspace.unavailable_until() else {
        return false;
    };
    ui.strong(workspace.label());
    ui.label(crate::theme::Status::Unavailable.rich_text());
    ui.label(why);
    true
}

/// The shell's keys, through the remappable bindings and only while no text
/// field has the keyboard.
pub fn shell_keyboard_system(
    keys: Res<ButtonInput<KeyCode>>,
    gate: Res<crate::bindings::KeyboardGate>,
    bindings: Res<crate::bindings::Bindings>,
    mut ui_state: ResMut<ViewerUiState>,
    mut runtime_visibility: ResMut<RuntimeOverlayVisibility>,
    mut mission_overlay: ResMut<MissionOverlaySettings>,
) {
    use crate::bindings::Action;
    for (action, command) in [
        (Action::CycleWorkspace, ShellCommand::CycleWorkspace),
        (Action::CycleProjection, ShellCommand::CycleProjection),
        (Action::CyclePreset, ShellCommand::CyclePreset),
        (Action::ToggleToolPanel, ShellCommand::ToggleToolPanel),
    ] {
        if bindings.just_pressed(&gate, &keys, action) {
            apply_shell_command(
                command,
                &mut ui_state,
                &mut runtime_visibility,
                &mut mission_overlay,
            );
        }
    }
}

/// Escape closes an open event inspector, unless the camera already used it.
/// Runs between the camera and the selection keys: one Escape leaves one
/// transient mode -- the camera's, then the inspector -- before it ever
/// clears the selection (VIEWER_UI §12).
pub fn close_event_on_escape_system(
    keys: Res<ButtonInput<KeyCode>>,
    gate: Res<crate::bindings::KeyboardGate>,
    bindings: Res<crate::bindings::Bindings>,
    mut escape: ResMut<crate::camera_control::EscapeUsed>,
    mut ui_state: ResMut<ViewerUiState>,
) {
    if escape.0 || ui_state.inspected_event.is_none() {
        return;
    }
    if bindings.just_pressed(&gate, &keys, crate::bindings::Action::Escape) {
        ui_state.inspected_event = None;
        escape.0 = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> (
        ViewerUiState,
        RuntimeOverlayVisibility,
        MissionOverlaySettings,
    ) {
        (
            ViewerUiState::default(),
            RuntimeOverlayVisibility::default(),
            MissionOverlaySettings::default(),
        )
    }

    #[test]
    fn the_three_controls_are_independent() {
        let (mut ui, mut vis, mut overlay) = state();
        apply_shell_command(
            ShellCommand::Workspace(Workspace::Review),
            &mut ui,
            &mut vis,
            &mut overlay,
        );
        apply_shell_command(
            ShellCommand::Preset(MapLayerPreset::Safety),
            &mut ui,
            &mut vis,
            &mut overlay,
        );
        assert_eq!(ui.workspace, Workspace::Review);
        assert_eq!(ui.map_layer_preset, MapLayerPreset::Safety);
        assert_eq!(ui.projection, Projection::Perspective);
        apply_shell_command(
            ShellCommand::Workspace(Workspace::Operate),
            &mut ui,
            &mut vis,
            &mut overlay,
        );
        assert_eq!(
            ui.map_layer_preset,
            MapLayerPreset::Safety,
            "the preset survives"
        );
    }

    #[test]
    fn no_control_touches_the_selection() {
        let (mut ui, mut vis, mut overlay) = state();
        ui.focused_entity_label = Some("drone-1".into());
        for command in [
            ShellCommand::CycleWorkspace,
            ShellCommand::CycleWorkspace,
            ShellCommand::CycleProjection,
            ShellCommand::CyclePreset,
            ShellCommand::CyclePreset,
            ShellCommand::ToggleToolPanel,
            ShellCommand::Workspace(Workspace::Plan),
            ShellCommand::Preset(MapLayerPreset::Reconstruction),
        ] {
            apply_shell_command(command, &mut ui, &mut vis, &mut overlay);
            assert!(
                ui.select_request.is_none(),
                "{command:?} asked to change the selection"
            );
            assert_eq!(ui.focused_entity_label.as_deref(), Some("drone-1"));
        }
    }

    #[test]
    fn cycling_visits_every_workspace_and_preset() {
        let (mut ui, mut vis, mut overlay) = state();
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(ui.workspace);
            apply_shell_command(
                ShellCommand::CycleWorkspace,
                &mut ui,
                &mut vis,
                &mut overlay,
            );
        }
        assert_eq!(seen, Workspace::ALL);
        assert_eq!(ui.workspace, Workspace::Operate, "and wraps");
        let mut presets = Vec::new();
        for _ in 0..3 {
            presets.push(ui.map_layer_preset);
            apply_shell_command(ShellCommand::CyclePreset, &mut ui, &mut vis, &mut overlay);
        }
        assert_eq!(presets, MapLayerPreset::ALL);
    }

    #[test]
    fn a_preset_by_key_applies_its_layers_as_by_pointer() {
        let (mut by_key, mut key_vis, mut key_overlay) = state();
        let (mut by_click, mut click_vis, mut click_overlay) = state();
        apply_shell_command(
            ShellCommand::CyclePreset,
            &mut by_key,
            &mut key_vis,
            &mut key_overlay,
        );
        apply_shell_command(
            ShellCommand::Preset(MapLayerPreset::Reconstruction),
            &mut by_click,
            &mut click_vis,
            &mut click_overlay,
        );
        assert_eq!(by_key.map_layer_preset, by_click.map_layer_preset);
        assert_eq!(key_vis.tracks, click_vis.tracks);
        assert!(!key_vis.tracks, "Reconstruction hides tracks");
    }

    #[test]
    fn unavailable_projections_are_refused_not_claimed() {
        let (mut ui, mut vis, mut overlay) = state();
        assert!(!apply_shell_command(
            ShellCommand::Projection(Projection::Section),
            &mut ui,
            &mut vis,
            &mut overlay
        ));
        assert!(!apply_shell_command(
            ShellCommand::CycleProjection,
            &mut ui,
            &mut vis,
            &mut overlay
        ));
        assert_eq!(ui.projection, Projection::Perspective);
        for projection in [Projection::TopDown, Projection::Section] {
            assert!(projection.unavailable_until().is_some());
        }
    }

    #[test]
    fn plan_and_review_say_what_they_wait_for() {
        assert!(Workspace::Operate.unavailable_until().is_none());
        for workspace in [Workspace::Plan, Workspace::Review] {
            let why = workspace.unavailable_until().expect("not built yet");
            assert!(
                why.contains("unavailable until") && why.contains("S1"),
                "{why}"
            );
        }
    }

    /// One Escape, one transient mode: the event inspector closes and the
    /// selection keys see the Escape as used; an Escape the camera already
    /// used leaves the inspector open.
    #[test]
    fn escape_closes_the_event_inspector_before_the_selection() {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<crate::bindings::KeyboardGate>()
            .insert_resource(crate::bindings::Bindings::default())
            .init_resource::<crate::camera_control::EscapeUsed>()
            .init_resource::<ViewerUiState>()
            .add_systems(Update, close_event_on_escape_system);
        let open = || {
            Some(crate::ui::EventRef {
                epoch: 0,
                frame_index: 1,
                event_id: "se-1".into(),
            })
        };
        let press_escape = |app: &mut App, camera_used: bool| {
            app.world_mut()
                .resource_mut::<crate::camera_control::EscapeUsed>()
                .0 = camera_used;
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.reset_all();
            keys.press(KeyCode::Escape);
            app.update();
        };
        app.world_mut()
            .resource_mut::<ViewerUiState>()
            .inspected_event = open();
        press_escape(&mut app, true);
        assert!(
            app.world()
                .resource::<ViewerUiState>()
                .inspected_event
                .is_some(),
            "the camera's Escape is not also the inspector's"
        );
        press_escape(&mut app, false);
        assert!(app
            .world()
            .resource::<ViewerUiState>()
            .inspected_event
            .is_none());
        assert!(
            app.world()
                .resource::<crate::camera_control::EscapeUsed>()
                .0,
            "and the selection keeps this Escape's selection"
        );
    }

    /// Off the canvas is the UI's: the picker must not see a click on the
    /// rail or the inspector (seen in the real window: it cleared the
    /// selection).
    #[test]
    fn only_the_canvas_belongs_to_the_world() {
        let mut ui = ViewerUiState::default();
        assert!(ui.on_canvas([5.0, 5.0]), "no shell laid out: all world");
        ui.canvas_rect = Some([194.0, 92.0, 1188.0, 800.0]);
        assert!(ui.on_canvas([600.0, 400.0]));
        for off in [
            [40.0, 359.0],
            [1260.0, 412.0],
            [600.0, 30.0],
            [600.0, 850.0],
        ] {
            assert!(!ui.on_canvas(off), "{off:?} is the shell's");
        }
        assert_eq!(ui.canvas_origin(), [194.0, 92.0]);
    }

    #[test]
    fn the_tool_panel_toggles_by_key() {
        let (mut ui, mut vis, mut overlay) = state();
        assert!(!ui.drawer_open);
        assert!(apply_shell_command(
            ShellCommand::ToggleToolPanel,
            &mut ui,
            &mut vis,
            &mut overlay
        ));
        assert!(ui.drawer_open);
    }
}
