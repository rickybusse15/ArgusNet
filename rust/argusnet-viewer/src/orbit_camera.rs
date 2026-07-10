use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;

use crate::schema::Bounds2d;
use crate::state::{ReconstructionCamera, ViewMode};

const MIN_PITCH_RAD: f32 = 0.1;
const MAX_PITCH_RAD: f32 = 1.4;

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_orbit_camera);
    }
}

#[derive(Component, Debug, Clone)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub min_radius: f32,
    pub max_radius: f32,
}

impl OrbitCamera {
    pub fn from_bounds(
        bounds: &Bounds2d,
        min_height_m: Option<f32>,
        max_height_m: Option<f32>,
    ) -> Self {
        let center = bounds.center_xy();
        let span = bounds.span_xy();
        let diagonal = span[0].hypot(span[1]).max(250.0);
        let ground = min_height_m.unwrap_or(0.0);
        let peak = max_height_m.unwrap_or(ground + 80.0);
        Self {
            focus: Vec3::new(center[0], center[1], (ground + peak) * 0.5),
            radius: (diagonal * 1.05).clamp(40.0, diagonal * 5.0),
            yaw: 0.85,
            pitch: 1.30,
            min_radius: 40.0,
            max_radius: diagonal * 5.0,
        }
    }

    pub fn eye_position(&self) -> Vec3 {
        let horizontal = self.radius * self.pitch.cos();
        Vec3::new(
            self.focus.x + horizontal * self.yaw.cos(),
            self.focus.y + horizontal * self.yaw.sin(),
            self.focus.z + self.radius * self.pitch.sin(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn update_orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut contexts: EguiContexts,
    windows: Query<&Window, With<PrimaryWindow>>,
    view_mode: Res<ViewMode>,
    mut query: Query<(&mut OrbitCamera, &mut Transform, Option<&ReconstructionCamera>)>,
) {
    // Accumulate input deltas. Always drain the readers so events never pile up.
    let mut orbit_delta = Vec2::ZERO;
    let mut pan_delta = Vec2::ZERO;
    for event in mouse_motion.read() {
        if buttons.pressed(MouseButton::Right)
            && !keys.pressed(KeyCode::ShiftLeft)
            && !keys.pressed(KeyCode::ShiftRight)
        {
            orbit_delta += event.delta;
        } else if buttons.pressed(MouseButton::Middle)
            || (buttons.pressed(MouseButton::Right)
                && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight)))
        {
            pan_delta += event.delta;
        }
    }

    let mut scroll_delta = 0.0;
    for event in mouse_wheel.read() {
        scroll_delta += event.y;
    }

    // When the pointer is over an egui panel (the drawer / tabs), let egui consume
    // the scroll so the panel scrolls on its own — the cameras stay put.
    let over_ui = contexts.ctx_mut().is_pointer_over_area();

    // Route input to the pane under the cursor. In Split mode the left half drives
    // the main camera and the right half drives the reconstruction camera; in every
    // other mode only the main camera is active and takes the whole window.
    let window = windows.get_single().ok();
    let cursor = window.and_then(|w| w.cursor_position());
    let half_width = window.map(|w| w.width() * 0.5).unwrap_or(f32::MAX);
    let cursor_over_recon = matches!(*view_mode, ViewMode::Split)
        && cursor.map(|c| c.x >= half_width).unwrap_or(false);

    for (mut camera, mut transform, is_recon) in &mut query {
        let is_target = !over_ui
            && if cursor_over_recon {
                is_recon.is_some()
            } else {
                is_recon.is_none()
            };

        if is_target {
            if orbit_delta.length_squared() > 0.0 {
                camera.yaw -= orbit_delta.x * 0.005;
                camera.pitch =
                    (camera.pitch + orbit_delta.y * 0.004).clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
            }

            if pan_delta.length_squared() > 0.0 {
                let forward = (camera.focus - camera.eye_position()).normalize_or_zero();
                let right = forward.cross(Vec3::Z).normalize_or_zero();
                let up = right.cross(forward).normalize_or_zero();
                let pan_scale = camera.radius * 0.0025;
                camera.focus += ((-pan_delta.x * right) + (pan_delta.y * up)) * pan_scale;
            }

            if scroll_delta.abs() > f32::EPSILON {
                let zoom_factor = (1.0 - (scroll_delta * 0.1)).clamp(0.2, 4.0);
                camera.radius =
                    (camera.radius * zoom_factor).clamp(camera.min_radius, camera.max_radius);
            }
        }

        // Keep every camera's transform in sync with its orbit parameters so both
        // panes render correctly even on frames where they receive no input.
        transform.translation = camera.eye_position();
        transform.look_at(camera.focus, Vec3::Z);
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::Vec3;

    use super::OrbitCamera;
    use crate::schema::Bounds2d;

    #[test]
    fn default_orbit_camera_starts_above_scene_focus() {
        let camera = OrbitCamera::from_bounds(
            &Bounds2d {
                x_min_m: -500.0,
                x_max_m: 500.0,
                y_min_m: -300.0,
                y_max_m: 300.0,
            },
            Some(90.0),
            Some(180.0),
        );

        let eye = camera.eye_position();
        assert!(eye.z > camera.focus.z);
    }

    #[test]
    fn positive_pitch_keeps_eye_above_ground_plane() {
        let camera = OrbitCamera {
            focus: Vec3::new(0.0, 0.0, 100.0),
            radius: 250.0,
            yaw: 0.0,
            pitch: super::MIN_PITCH_RAD,
            min_radius: 40.0,
            max_radius: 1200.0,
        };

        assert!(camera.eye_position().z > 100.0);
    }
}
