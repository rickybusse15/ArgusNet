//! Hover and selection rings (P2.3).
//!
//! The selected marker, and the one under the pointer, each get a ring: a
//! camera-facing annulus just outside the glyph, a few pixels wide whatever
//! the zoom. It is its own pass, not a change to the glyph's material: marker
//! materials are shared by colour across every marker, and the hologram is
//! additive, so an inverted-hull outline behind it would show through as a
//! filled disc rather than an edge.
//!
//! The ring is unlit and opaque, and depth-tested like any mesh, so terrain
//! and buildings hide it where they hide the glyph: there is no
//! through-terrain outline. It carries
//! [`crate::sensor_volumes::PresentationOnly`], so it never takes a pick, and
//! casts no shadow. The hover ring is thinner and dimmer, and is not drawn on
//! the selected marker. A ring whose marker is hidden, or gone, is hidden;
//! selection itself is cleared by the snapshot adapter, not here.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::app::{marker_appearance, MainCamera};
use crate::glyphs::{GlyphProjection, NODE_GLYPH_PX, TRACK_GLYPH_PX, TRUTH_GLYPH_PX};
use crate::marker_meshes::MarkerEntityIndex;
use crate::picking::HoverState;
use crate::presentation::{CurrentRuntimeMarkers, EntityKey};
use crate::replay::MarkerKind;
use crate::schema::ScenePackage;
use crate::sensor_volumes::PresentationOnly;
use crate::state::SelectionState;

/// How far a glyph reaches, in glyph radii: a drone's arms and a station's
/// corners sit a little past one.
const GLYPH_REACH: f32 = 1.15;
/// Clear space between the glyph and its ring, in pixels.
pub const RING_GAP_PX: f32 = 3.0;
const RING_SEGMENTS: usize = 48;

/// Which ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RingRole {
    Selected,
    Hover,
}

impl RingRole {
    pub fn width_px(self) -> f32 {
        match self {
            Self::Selected => 2.5,
            Self::Hover => 1.5,
        }
    }

    pub fn color(self) -> Color {
        match self {
            Self::Selected => Color::srgb(1.0, 0.86, 0.25),
            Self::Hover => Color::srgb(0.55, 0.8, 1.0),
        }
    }
}

/// The ring's inner radius, in pixels, around a glyph drawn `glyph_px` wide
/// at `confidence` scale.
pub fn ring_radius_px(glyph_px: f32, confidence: f32) -> f32 {
    glyph_px * confidence * GLYPH_REACH + RING_GAP_PX
}

/// An annulus in the XY plane, facing +Z, from `inner` to `inner + width`.
pub fn ring_mesh(inner: f32, width: f32) -> Mesh {
    let outer = inner + width;
    let mut positions = Vec::with_capacity(RING_SEGMENTS * 2);
    for i in 0..RING_SEGMENTS {
        let a = std::f32::consts::TAU * i as f32 / RING_SEGMENTS as f32;
        let (s, c) = a.sin_cos();
        positions.push([c * inner, s * inner, 0.0]);
        positions.push([c * outer, s * outer, 0.0]);
    }
    let mut indices = Vec::with_capacity(RING_SEGMENTS * 6);
    for i in 0..RING_SEGMENTS as u32 {
        let (a, b) = (2 * i, 2 * ((i + 1) % RING_SEGMENTS as u32));
        indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
    }
    let normals = vec![[0.0, 0.0, 1.0]; positions.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Where a ring goes: at `position`, facing the camera, scaled so one mesh
/// unit is one pixel there.
pub fn ring_transform(projection: &GlyphProjection, position: Vec3) -> Transform {
    let facing = projection.right.cross(projection.up).normalize_or(Vec3::Z);
    let basis = Mat3::from_cols(projection.right, projection.up, facing);
    Transform::from_translation(position)
        .with_rotation(Quat::from_mat3(&basis))
        .with_scale(Vec3::splat(projection.radius(position, 1.0)))
}

#[derive(Component, Debug, Clone, Copy)]
pub struct SelectionRing {
    pub role: RingRole,
}

/// The two ring entities, and ring meshes by role and radius.
#[derive(Resource, Default)]
pub struct SelectionRings {
    entities: HashMap<RingRole, Entity>,
    /// `(role, radius in half pixels) -> mesh`.
    meshes: HashMap<(RingRole, u32), Handle<Mesh>>,
    /// What each ring was last placed on, so an idle frame writes nothing.
    placed: HashMap<RingRole, Option<(EntityKey, Transform)>>,
}

impl SelectionRings {
    /// The key each ring is on, if it is shown.
    pub fn target(&self, role: RingRole) -> Option<EntityKey> {
        self.placed
            .get(&role)
            .copied()
            .flatten()
            .map(|(key, _)| key)
    }
}

/// Keeps the rings on their markers. See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn sync_selection_rings(
    mut commands: Commands,
    selection: Res<SelectionState>,
    hover: Res<HoverState>,
    markers: Res<CurrentRuntimeMarkers>,
    index: Res<MarkerEntityIndex>,
    package: Res<ScenePackage>,
    camera: Query<(&GlobalTransform, &Projection, &Camera), With<MainCamera>>,
    marker_visibility: Query<&Visibility, (With<EntityKey>, Without<SelectionRing>)>,
    mut rings: ResMut<SelectionRings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut placed: Query<(&mut Transform, &mut Visibility, &mut Mesh3d), With<SelectionRing>>,
) {
    let projection = camera
        .single()
        .map(|(t, p, c)| GlyphProjection::from_camera(t, p, c.logical_viewport_size().map(|s| s.y)))
        .unwrap_or_default();
    let selected = selection
        .selected
        .as_ref()
        .filter(|s| s.epoch == markers.epoch())
        .map(|s| s.key);
    let hovered = hover.hovered().filter(|key| Some(*key) != selected);

    for (role, target) in [(RingRole::Selected, selected), (RingRole::Hover, hovered)] {
        // Shown only on a marker that is presented and visible.
        let shown = target.and_then(|key| {
            let marker = markers.get(key)?;
            let entity = index.entity(key)?;
            let visible = marker_visibility.get(entity).ok()? != Visibility::Hidden;
            visible.then_some((key, marker))
        });
        let placement = shown.map(|(key, marker)| {
            let glyph_px = match marker.kind {
                MarkerKind::Track => TRACK_GLYPH_PX,
                MarkerKind::Truth => TRUTH_GLYPH_PX,
                MarkerKind::Node => NODE_GLYPH_PX,
            };
            let (_, confidence) = marker_appearance(marker, &package);
            let radius = ring_radius_px(glyph_px, confidence);
            (key, ring_transform(&projection, marker.position), radius)
        });
        let now = placement.map(|(key, transform, _)| (key, transform));
        if rings.placed.get(&role) == Some(&now) {
            continue;
        }
        rings.placed.insert(role, now);

        let entity = match rings.entities.get(&role) {
            Some(entity) => *entity,
            None => {
                let material = materials.add(StandardMaterial {
                    base_color: role.color(),
                    unlit: true,
                    cull_mode: None,
                    ..default()
                });
                let entity = commands
                    .spawn((
                        SelectionRing { role },
                        PresentationOnly,
                        NotShadowCaster,
                        Mesh3d::default(),
                        MeshMaterial3d(material),
                        Transform::default(),
                        Visibility::Hidden,
                    ))
                    .id();
                rings.entities.insert(role, entity);
                // Its components land when commands apply; place it next frame.
                rings.placed.insert(role, None);
                continue;
            }
        };
        let Ok((mut transform, mut visibility, mut mesh)) = placed.get_mut(entity) else {
            continue;
        };
        match placement {
            Some((_, at, radius)) => {
                let half_px = (radius * 2.0).round() as u32;
                let handle = rings
                    .meshes
                    .entry((role, half_px))
                    .or_insert_with(|| meshes.add(ring_mesh(half_px as f32 * 0.5, role.width_px())))
                    .clone();
                if mesh.0 != handle {
                    mesh.0 = handle;
                }
                transform.set_if_neq(at);
                visibility.set_if_neq(Visibility::Inherited);
            }
            None => {
                visibility.set_if_neq(Visibility::Hidden);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_is_an_annulus_of_the_given_radii() {
        let mesh = ring_mesh(10.0, 2.5);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let radii: Vec<f32> = positions
            .iter()
            .map(|p| Vec2::new(p[0], p[1]).length())
            .collect();
        assert!(radii
            .iter()
            .all(|r| (r - 10.0).abs() < 1e-4 || (r - 12.5).abs() < 1e-4));
        assert_eq!(
            radii.iter().filter(|r| (**r - 10.0).abs() < 1e-4).count(),
            RING_SEGMENTS
        );
        assert!(positions.iter().all(|p| p[2] == 0.0));
        assert_eq!(mesh.indices().unwrap().len(), RING_SEGMENTS * 6);
    }

    /// The ring clears the glyph's reach by the gap, at every confidence.
    #[test]
    fn the_ring_sits_just_outside_the_glyph() {
        for (px, confidence) in [
            (NODE_GLYPH_PX, 1.0),
            (TRACK_GLYPH_PX, 1.6),
            (TRUTH_GLYPH_PX, 0.8),
        ] {
            let radius = ring_radius_px(px, confidence);
            let clearance = radius - px * confidence * GLYPH_REACH;
            assert!(
                (2.0..=4.0).contains(&clearance),
                "{clearance} px between glyph and ring"
            );
            assert!(radius > px * confidence, "inside the glyph");
            assert!(
                radius < px * confidence * 2.0 + RING_GAP_PX,
                "far from the glyph"
            );
        }
    }

    /// Facing the camera: the ring's normal is the camera's view direction
    /// reversed, and one unit is one pixel at the ring's depth.
    #[test]
    fn the_ring_faces_the_camera_at_pixel_scale() {
        let camera = Transform::from_xyz(100.0, -200.0, 300.0).looking_at(Vec3::ZERO, Vec3::Z);
        let projection = GlyphProjection::from_camera(
            &GlobalTransform::from(camera),
            &Projection::Perspective(PerspectiveProjection::default()),
            Some(600.0),
        );
        let at = Vec3::new(5.0, 5.0, 10.0);
        let t = ring_transform(&projection, at);
        let normal = t.rotation * Vec3::Z;
        let to_camera = (camera.translation - at).normalize();
        assert!(
            normal.dot(camera.forward().as_vec3()) < -0.99,
            "{normal} vs forward"
        );
        assert!(normal.dot(to_camera) > 0.9);
        assert!((t.scale.x - projection.radius(at, 1.0)).abs() < 1e-6);
        assert_eq!(t.translation, at);
    }
}
