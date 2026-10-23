//! The main view's projection (C5, VIEWER_UI §6): Perspective, a true
//! orthographic Top-down over the same orbit focus (C5.1), or Section (C5.2):
//! an orthographic view along a vertical slab's normal whose near and far
//! planes are the slab's faces, so the GPU clips everything outside it.
//!
//! Seen exactly edge-on, a surface collapses to a line the GPU barely
//! rasterises, so Section also draws where the slab's centre plane cuts the
//! base layers' meshes: the terrain's profile and the buildings' outlines,
//! an overhang and the passage under it as separate lines.
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
use crate::state::{Projection as ViewProjection, SectionPlane, ViewerUiState};

/// The perspective projection set aside while Top-down is shown.
#[derive(Resource, Debug, Default)]
pub struct SavedPerspective(pub Option<PerspectiveProjection>);

/// The slab Section is showing, for picking to honour: a click finds
/// nothing outside it. `None` in the other projections.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct SectionSlab(pub Option<(Vec3, SectionPlane)>);

impl SectionSlab {
    /// Whether a world point can be picked: always outside Section, only
    /// inside the slab in it.
    pub fn admits(&self, point: Vec3) -> bool {
        self.0
            .is_none_or(|(focus, plane)| plane.contains(focus, point))
    }
}

/// The Section camera for `plane` around `focus`: looking along the plane's
/// normal, up is up, near and far on the slab's faces.
pub fn section_view(
    focus: Vec3,
    plane: SectionPlane,
    viewport_height: f32,
) -> (Transform, OrthographicProjection) {
    let normal = plane.axis.normal();
    let centre = focus + normal * plane.offset_m;
    let half = plane.thickness_m.max(SectionPlane::MIN_THICKNESS_M) * 0.5;
    // The eye stands on the slab's near face. Any distance works for an
    // orthographic view, but distance fog does not: from further back the
    // slab's own meshes fade into the fog colour.
    let eye = centre - normal * half;
    (
        Transform::from_translation(eye).looking_at(centre, Vec3::Z),
        OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical { viewport_height },
            near: 0.0,
            far: 2.0 * half,
            ..OrthographicProjection::default_3d()
        },
    )
}

/// Where `plane` around `focus` cuts the triangles of a mesh: one segment
/// per triangle the centre plane crosses. `positions` are in world space.
/// A triangle lying in the plane draws nothing; its neighbours' cuts outline
/// it.
pub fn plane_cut(
    positions: &[Vec3],
    triangles: impl Iterator<Item = [usize; 3]>,
    centre: Vec3,
    normal: Vec3,
    segments: &mut Vec<[Vec3; 2]>,
) {
    for triangle in triangles {
        let Some(corners) = triangle
            .iter()
            .map(|&i| positions.get(i).copied())
            .collect::<Option<Vec<Vec3>>>()
        else {
            continue;
        };
        let side = corners
            .iter()
            .map(|p| (*p - centre).dot(normal))
            .collect::<Vec<f32>>();
        let mut crossings = Vec::with_capacity(2);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (da, db) = (side[a], side[b]);
            // A strict sign change, or one end on the plane counted once.
            if (da < 0.0 && db >= 0.0) || (da >= 0.0 && db < 0.0) {
                let t = da / (da - db);
                crossings.push(corners[a].lerp(corners[b], t));
            }
        }
        if let [start, end] = crossings[..] {
            if start != end {
                segments.push([start, end]);
            }
        }
    }
}

/// The section's cut through the base layers (terrain, buildings), redone
/// only when the slab or the occluders change.
#[derive(Resource, Debug, Default)]
pub struct SectionCut {
    key: Option<(Vec3, SectionPlane, u64)>,
    pub segments: Vec<[Vec3; 2]>,
}

/// The cut's colour: amber, apart from every layer's colour, so the cut
/// reads as the cut.
const CUT_COLOR: Color = Color::srgb(1.0, 0.72, 0.25);

/// Cut the visible base-layer meshes with the slab's centre plane, and draw
/// the cut, while Section is shown.
pub fn draw_section_cut_system(
    mut gizmos: Gizmos,
    slab: Res<SectionSlab>,
    generation: Res<crate::picking::PickGeneration>,
    meshes: Res<Assets<Mesh>>,
    occluders: Query<
        (&Mesh3d, &GlobalTransform, &InheritedVisibility),
        With<crate::picking::PickOccluder>,
    >,
    mut cut: ResMut<SectionCut>,
) {
    let Some((focus, plane)) = slab.0 else {
        if cut.key.is_some() {
            *cut = SectionCut::default();
        }
        return;
    };
    let key = Some((focus, plane, generation.0));
    if cut.key != key {
        let normal = plane.axis.normal();
        let centre = focus + normal * plane.offset_m;
        let mut segments = Vec::new();
        for (mesh, transform, visibility) in &occluders {
            if !visibility.get() {
                continue;
            }
            let Some(mesh) = meshes.get(&mesh.0) else {
                continue;
            };
            let Some(bevy::mesh::VertexAttributeValues::Float32x3(local)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                continue;
            };
            let affine = transform.affine();
            let world: Vec<Vec3> = local
                .iter()
                .map(|p| affine.transform_point3(Vec3::from(*p)))
                .collect();
            match mesh.indices() {
                Some(indices) => {
                    let indices: Vec<usize> = indices.iter().collect();
                    let triangles = indices.as_chunks::<3>().0.iter().copied();
                    plane_cut(&world, triangles, centre, normal, &mut segments);
                }
                None => {
                    let triangles = (0..world.len() / 3).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]);
                    plane_cut(&world, triangles, centre, normal, &mut segments);
                }
            }
        }
        *cut = SectionCut { key, segments };
    }
    for [start, end] in &cut.segments {
        gizmos.line(*start, *end, CUT_COLOR);
    }
}

/// The height of ground an orthographic view must show to match a
/// perspective one at `distance` metres.
pub fn matching_ortho_height(fov: f32, distance: f32) -> f32 {
    2.0 * distance.max(1.0) * (fov * 0.5).tan()
}

/// Metres per screen pixel at the camera's focus (V3's scale bar): the
/// height the view spans there over the canvas's height in pixels. Both
/// projections share it, since top-down and Section match the perspective
/// scale at the focus ([`matching_ortho_height`]). `None` before a camera
/// and its canvas exist, and in free-fly, which has no focus to measure at:
/// the orbit's radius there is stale, and a bar from it would lie.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct CanvasScale {
    pub metres_per_px: Option<f32>,
}

/// Keep [`CanvasScale`] current, after the projection is applied.
pub fn update_canvas_scale_system(
    mut scale: ResMut<CanvasScale>,
    saved: Res<SavedPerspective>,
    control: Option<Res<crate::camera_control::CameraControl>>,
    camera: Query<(&Camera, &OrbitCamera, &Projection), With<MainCamera>>,
) {
    let free_fly =
        control.is_some_and(|control| control.mode == crate::camera_control::CameraMode::FreeFly);
    let measured = camera
        .single()
        .ok()
        .and_then(|(camera, orbit, projection)| {
            let fov = match projection {
                Projection::Perspective(perspective) => Some(perspective.fov),
                _ => saved.0.as_ref().map(|perspective| perspective.fov),
            }?;
            canvas_metres_per_px(
                free_fly,
                fov,
                orbit.radius,
                camera.logical_viewport_size()?.y,
            )
        });
    scale.set_if_neq(CanvasScale {
        metres_per_px: measured,
    });
}

/// Metres per logical pixel at the focus of an orbit at `radius` with
/// vertical field of view `fov`, on a canvas `height_px` tall; `None` in
/// free-fly, where the orbit's radius is stale and there is no focus.
pub fn canvas_metres_per_px(free_fly: bool, fov: f32, radius: f32, height_px: f32) -> Option<f32> {
    (!free_fly && height_px > 0.0).then(|| matching_ortho_height(fov, radius) / height_px)
}

/// The scale bar in egui points, which the UI-scale preference zooms: a
/// point is `zoom` logical pixels, so it spans `zoom` times the ground.
pub fn scale_bar_in_points(metres_per_px: f32, zoom: f32, max_points: f32) -> Option<(f32, f32)> {
    scale_bar(metres_per_px * zoom, max_points)
}

/// The scale bar for `metres_per_px`: the longest 1, 2 or 5 × 10ⁿ metres
/// that fits in `max_px`, and its length in pixels. `None` for a scale with
/// nothing sensible to show.
pub fn scale_bar(metres_per_px: f32, max_px: f32) -> Option<(f32, f32)> {
    if !(metres_per_px.is_finite() && metres_per_px > 0.0 && max_px > 0.0) {
        return None;
    }
    let span = metres_per_px * max_px;
    let decade = 10_f32.powf(span.log10().floor());
    let metres = [5.0, 2.0, 1.0]
        .into_iter()
        .map(|step| step * decade)
        .find(|metres| *metres <= span)?;
    Some((metres, metres / metres_per_px))
}

/// A scale bar's length as an operator reads it.
pub fn scale_label(metres: f32) -> String {
    if metres >= 1000.0 {
        format!("{} km", metres / 1000.0)
    } else if metres >= 1.0 {
        format!("{} m", metres)
    } else {
        format!("{} cm", (metres * 100.0).round())
    }
}

/// Apply the selected projection to the main camera.
pub fn apply_projection_system(
    ui_state: Res<ViewerUiState>,
    mut saved: ResMut<SavedPerspective>,
    mut slab: ResMut<SectionSlab>,
    mut camera: Query<(&OrbitCamera, &mut Transform, &mut Projection), With<MainCamera>>,
) {
    let Ok((orbit, mut transform, mut projection)) = camera.single_mut() else {
        return;
    };
    let section =
        (ui_state.projection == ViewProjection::Section).then_some((orbit.focus, ui_state.section));
    slab.set_if_neq(SectionSlab(section));
    if ui_state.projection != ViewProjection::Perspective {
        if let Projection::Perspective(perspective) = &*projection {
            saved.0 = Some(perspective.clone());
        }
    }
    match ui_state.projection {
        ViewProjection::Section => {
            let Some(perspective) = saved.0.as_ref() else {
                return;
            };
            let height = matching_ortho_height(perspective.fov, orbit.radius);
            let (view, ortho) = section_view(orbit.focus, ui_state.section, height);
            *transform = view;
            *projection = Projection::Orthographic(ortho);
        }
        ViewProjection::TopDown => {
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
    use super::{canvas_metres_per_px, scale_bar, scale_bar_in_points, scale_label};

    /// At 200 % UI scale a point is two pixels, so it covers twice the
    /// ground: the bar's metres per drawn point is the canvas scale times
    /// the zoom, whatever the zoom.
    #[test]
    fn the_scale_bar_holds_its_ground_length_at_any_ui_scale() {
        assert_eq!(scale_bar_in_points(1.0, 1.0, 120.0), Some((100.0, 100.0)));
        assert_eq!(scale_bar_in_points(1.0, 2.0, 120.0), Some((200.0, 100.0)));
        for zoom in [0.75_f32, 1.25, 2.0, 3.0] {
            let (metres, points) = scale_bar_in_points(1.0, zoom, 120.0).unwrap();
            assert!((metres / points - zoom).abs() < 1e-5, "zoom {zoom}");
        }
    }

    /// Free-fly has no focus: no scale, rather than the stale orbit's.
    #[test]
    fn free_fly_has_no_scale() {
        assert!(canvas_metres_per_px(true, 0.8, 500.0, 700.0).is_none());
        assert!(canvas_metres_per_px(false, 0.8, 500.0, 0.0).is_none());
        let m = canvas_metres_per_px(false, 0.8, 500.0, 700.0).unwrap();
        assert!((m - super::matching_ortho_height(0.8, 500.0) / 700.0).abs() < 1e-6);
    }

    /// The bar is the longest 1, 2 or 5 x 10^n metres that fits, at its
    /// length in pixels; nothing for a scale with nothing to show.
    #[test]
    fn the_scale_bar_is_a_round_length_that_fits() {
        let (metres, px) = scale_bar(1.0, 120.0).unwrap();
        assert_eq!((metres, px), (100.0, 100.0));
        let (metres, px) = scale_bar(3.0, 120.0).unwrap();
        assert_eq!(metres, 200.0);
        assert!((px - 200.0 / 3.0).abs() < 1e-4);
        assert_eq!(scale_bar(0.02, 120.0).unwrap().0, 2.0);
        assert_eq!(scale_bar(40.0, 120.0).unwrap().0, 2000.0);
        assert!(scale_bar(0.0, 120.0).is_none());
        assert!(scale_bar(f32::NAN, 120.0).is_none());
        assert_eq!(scale_label(2000.0), "2 km");
        assert_eq!(scale_label(200.0), "200 m");
        assert_eq!(scale_label(0.5), "50 cm");
    }

    use super::*;

    fn app(projection: ViewProjection) -> App {
        let mut app = App::new();
        app.insert_resource(ViewerUiState {
            projection,
            ..ViewerUiState::default()
        })
        .init_resource::<SavedPerspective>()
        .init_resource::<SectionSlab>()
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
    fn section_looks_along_the_normal_with_the_slab_as_near_and_far() {
        let mut app = app(ViewProjection::Section);
        app.update();
        let (transform, projection, focus) = camera(&mut app);
        let Projection::Orthographic(ortho) = projection else {
            panic!("section is orthographic");
        };
        assert!(transform.forward().dot(Vec3::Y) > 0.9999, "looking north");
        assert!(transform.up().dot(Vec3::Z) > 0.9999, "up is up");
        assert!((ortho.far - ortho.near - 50.0).abs() < 1e-3, "the slab");
        // The slab's centre plane passes through the focus.
        let centre_depth = (focus - transform.translation).dot(Vec3::Y);
        assert!((centre_depth - (ortho.near + ortho.far) * 0.5).abs() < 1e-2);
        assert_eq!(
            *app.world().resource::<SectionSlab>(),
            SectionSlab(Some((focus, SectionPlane::default())))
        );
    }

    #[test]
    fn an_overhang_and_the_passage_under_it_stay_apart() {
        // The point of a section: two surfaces at the same XY, one above the
        // other, which a top-down view draws on top of each other.
        let focus = Vec3::new(500.0, 500.0, 0.0);
        let (view, _) = section_view(focus, SectionPlane::default(), 400.0);
        let to_view = view.compute_affine().inverse();
        let roof = to_view.transform_point3(Vec3::new(520.0, 500.0, 40.0));
        let floor = to_view.transform_point3(Vec3::new(520.0, 500.0, 5.0));
        assert!((roof.y - floor.y - 35.0).abs() < 1e-2, "{roof:?} {floor:?}");
        assert!((roof.x - floor.x).abs() < 1e-3);
        let slab = SectionSlab(Some((focus, SectionPlane::default())));
        assert!(slab.admits(Vec3::new(520.0, 510.0, 40.0)));
        assert!(
            !slab.admits(Vec3::new(520.0, 530.0, 40.0)),
            "behind the slab"
        );
        assert!(SectionSlab(None).admits(Vec3::splat(1e6)));
    }

    #[test]
    fn leaving_section_clears_the_slab_and_restores_the_perspective() {
        let mut app = app(ViewProjection::Section);
        app.update();
        app.world_mut().resource_mut::<ViewerUiState>().projection = ViewProjection::Perspective;
        app.update();
        let (_, projection, _) = camera(&mut app);
        assert!(matches!(projection, Projection::Perspective(_)));
        assert_eq!(*app.world().resource::<SectionSlab>(), SectionSlab(None));
    }

    #[test]
    fn the_cut_traces_where_the_plane_crosses_each_triangle() {
        // A ground quad at z = 0 spanning y in [-10, 10], and an overhang
        // quad above it at z = 30: the plane y = 0 cuts each along x.
        let positions = [
            Vec3::new(0.0, -10.0, 0.0),
            Vec3::new(100.0, -10.0, 0.0),
            Vec3::new(100.0, 10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(20.0, -10.0, 30.0),
            Vec3::new(60.0, -10.0, 30.0),
            Vec3::new(60.0, 10.0, 30.0),
            Vec3::new(20.0, 10.0, 30.0),
        ];
        let triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
        let mut segments = Vec::new();
        plane_cut(
            &positions,
            triangles.into_iter(),
            Vec3::ZERO,
            Vec3::Y,
            &mut segments,
        );
        assert_eq!(segments.len(), 4, "{segments:?}");
        assert!(segments.iter().flatten().all(|p| p.y.abs() < 1e-4));
        let mut heights: Vec<f32> = segments.iter().map(|[a, _]| a.z).collect();
        heights.sort_by(f32::total_cmp);
        assert_eq!(heights, [0.0, 0.0, 30.0, 30.0], "ground and overhang apart");
        // A plane that misses the meshes cuts nothing.
        segments.clear();
        plane_cut(
            &positions,
            triangles.into_iter(),
            Vec3::new(0.0, 50.0, 0.0),
            Vec3::Y,
            &mut segments,
        );
        assert!(segments.is_empty());
        // Out-of-range indices are skipped, not a panic.
        plane_cut(
            &positions,
            [[0, 1, 99]].into_iter(),
            Vec3::ZERO,
            Vec3::Y,
            &mut segments,
        );
        assert!(segments.is_empty());
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
