//! The main view's projection (C5.1, VIEWER_UI §6): Perspective, or a true
//! orthographic Top-down over the same orbit focus.
//!
//! The camera control drives the orbit as always; this runs after it and,
//! in Top-down, replaces the view with an orthographic one looking straight
//! down at the orbit's focus, north up, sized so the ground at the focus is
//! the scale the perspective view showed. Pan and zoom keep working through
//! the orbit's focus and radius; leaving Top-down restores the perspective
//! projection exactly as it was.

use bevy::camera::ScalingMode;
use bevy::prelude::*;

use crate::app::MainCamera;
use crate::orbit_camera::OrbitCamera;
use crate::state::{Projection as ViewProjection, ViewerUiState};

/// The perspective projection set aside while Top-down is shown.
#[derive(Resource, Debug, Default)]
pub struct SavedPerspective(pub Option<PerspectiveProjection>);

/// The height of ground an orthographic view must show to match a
/// perspective one at `distance` metres.
pub fn matching_ortho_height(fov: f32, distance: f32) -> f32 {
    2.0 * distance.max(1.0) * (fov * 0.5).tan()
}

/// Apply the selected projection to the main camera.
pub fn apply_projection_system(
    ui_state: Res<ViewerUiState>,
    mut saved: ResMut<SavedPerspective>,
    mut camera: Query<(&OrbitCamera, &mut Transform, &mut Projection), With<MainCamera>>,
) {
    let Ok((orbit, mut transform, mut projection)) = camera.single_mut() else {
        return;
    };
    match ui_state.projection {
        ViewProjection::TopDown => {
            if let Projection::Perspective(perspective) = &*projection {
                saved.0 = Some(perspective.clone());
            }
            let Some(perspective) = saved.0.as_ref() else {
                return;
            };
            let height = matching_ortho_height(perspective.fov, orbit.radius);
            // Terrain can stand above the focus: the near plane is behind the
            // eye rather than in front of it, so nothing is clipped away.
            let depth = perspective.far;
            *projection = Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical {
                    viewport_height: height,
                },
                near: -depth,
                far: depth,
                ..OrthographicProjection::default_3d()
            });
            *transform = Transform::from_translation(orbit.focus + Vec3::Z * orbit.radius.max(1.0))
                .looking_at(orbit.focus, Vec3::Y);
        }
        _ => {
            if let Some(perspective) = saved.0.take() {
                *projection = Projection::Perspective(perspective);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(projection: ViewProjection) -> App {
        let mut app = App::new();
        app.insert_resource(ViewerUiState {
            projection,
            ..ViewerUiState::default()
        })
        .init_resource::<SavedPerspective>()
        .add_systems(Update, apply_projection_system);
        let mut orbit = OrbitCamera::from_bounds(
            &crate::schema::Bounds2d {
                x_min_m: 0.0,
                x_max_m: 1000.0,
                y_min_m: 0.0,
                y_max_m: 1000.0,
            },
            None,
            None,
        );
        orbit.radius = 800.0;
        app.world_mut().spawn((
            MainCamera,
            orbit,
            Projection::Perspective(PerspectiveProjection {
                fov: 45.0_f32.to_radians(),
                near: 1.0,
                far: 9000.0,
                ..default()
            }),
            Transform::from_xyz(100.0, -900.0, 600.0),
        ));
        app
    }

    fn camera(app: &mut App) -> (Transform, Projection, Vec3) {
        let mut query = app
            .world_mut()
            .query_filtered::<(&Transform, &Projection, &OrbitCamera), With<MainCamera>>();
        let (transform, projection, orbit) = query.single(app.world()).unwrap();
        (*transform, projection.clone(), orbit.focus)
    }

    #[test]
    fn top_down_looks_straight_down_north_up_at_the_focus() {
        let mut app = app(ViewProjection::TopDown);
        app.update();
        let (transform, projection, focus) = camera(&mut app);
        let Projection::Orthographic(ortho) = projection else {
            panic!("top-down is orthographic");
        };
        assert!(matches!(
            ortho.scaling_mode,
            ScalingMode::FixedVertical { .. }
        ));
        assert!(ortho.near < 0.0, "terrain above the focus is not clipped");
        let forward = transform.forward();
        assert!(forward.dot(-Vec3::Z) > 0.9999, "{forward:?}");
        assert!(transform.up().dot(Vec3::Y) > 0.9999, "north is up");
        assert!((transform.translation.truncate() - focus.truncate()).length() < 1e-3);
    }

    #[test]
    fn top_down_matches_the_perspective_scale_at_the_focus() {
        let height = matching_ortho_height(45.0_f32.to_radians(), 800.0);
        assert!((height - 2.0 * 800.0 * (22.5_f32.to_radians()).tan()).abs() < 1e-3);
        let mut app = app(ViewProjection::TopDown);
        app.update();
        let (_, projection, _) = camera(&mut app);
        let Projection::Orthographic(ortho) = projection else {
            unreachable!()
        };
        let ScalingMode::FixedVertical { viewport_height } = ortho.scaling_mode else {
            unreachable!()
        };
        assert!((viewport_height - height).abs() < 1e-3);
    }

    #[test]
    fn leaving_top_down_restores_the_perspective_exactly() {
        let mut app = app(ViewProjection::TopDown);
        app.update();
        app.world_mut().resource_mut::<ViewerUiState>().projection = ViewProjection::Perspective;
        app.update();
        let (_, projection, _) = camera(&mut app);
        let Projection::Perspective(perspective) = projection else {
            panic!("perspective again");
        };
        assert_eq!(perspective.fov, 45.0_f32.to_radians());
        assert_eq!(perspective.far, 9000.0);
    }

    #[test]
    fn perspective_is_left_alone() {
        let mut app = app(ViewProjection::Perspective);
        app.update();
        let (transform, projection, _) = camera(&mut app);
        assert!(matches!(projection, Projection::Perspective(_)));
        assert_eq!(transform.translation, Vec3::new(100.0, -900.0, 600.0));
    }
}
