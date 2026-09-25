//! Sensor fields of view as persistent, world-sized meshes (P2.2).
//!
//! # The drawn volume is the gated one
//!
//! Stage 6 accepts a target when it is within the sensor's range and, **only
//! when the sensor has both a half-angle and a look direction**, within that
//! half-angle of the look. The look is `argusnet_sensing::look`'s rule, the
//! runtime's own: along the velocity, tilted down, and none below 0.5 m/s of
//! horizontal speed. So the accepted region is a spherical sector (a cone
//! capped by a sphere), or a full sphere when there is no gate, and that is
//! what is drawn, at any half-angle including past 90 degrees.
//!
//! * **An unset range draws nothing.** The adapter reads `max_range_m: 0` as
//!   unset, and an unset range bounds nothing; the old gizmo invented 400 m.
//! * **The pose is the recorded one**, as presented (interpolated). The
//!   runtime senses from the nominal controller pose, before deconfliction
//!   and the safety clamp move a platform; the replay does not carry it. So a
//!   drone the safety monitor has zeroed reads as having no look here, and
//!   draws a sphere, while the runtime still gated along its nominal heading.
//! * **The tilt is the runtime default.** The replay carries no per-sensor
//!   tilt; every writer today uses `DEFAULT_LOOK_DOWN_ANGLE_DEG`.
//!
//! # Topology cached by shape
//!
//! One unit mesh per shape (sphere, or sector at a half-angle quantised to
//! half a degree), scaled by range and turned onto the look axis by the
//! entity's transform. Entities are keyed by platform and spawned or
//! despawned only when a platform's volume appears, disappears or changes
//! shape; transforms are written only where they changed.
//!
//! # Depth, transparency, picking
//!
//! The hologram material is additive and writes no depth, so a volume never
//! hides what is behind it and ordering between volumes does not matter.
//! Back faces are culled. A volume that **encloses the camera** is hidden
//! outright, so no field of view can wash out the view from inside it.
//! Nothing is drawn below the terrain's lowest point ([`volume_floor_z`]), so
//! a range's buried part does not show under the map's edge.
//! Volumes carry [`PresentationOnly`], which picking excludes.

use std::collections::HashMap;

use argusnet_sensing::look::{sensor_look_direction, DEFAULT_LOOK_DOWN_ANGLE_DEG};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::app::MainCamera;
use crate::marker_meshes::HologramMaterial;
use crate::presentation::{CurrentRuntimeMarkers, EntityKey};
use crate::replay::{MarkerKind, ReplayState};
use crate::state::RuntimeOverlayVisibility;

/// Marks presentation geometry that is not a domain entity: never picked.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct PresentationOnly;

/// A half-angle at or past this is a full sphere.
pub const FULL_SPHERE_HALF_ANGLE_DEG: f32 = 179.9;
/// Half-angles are quantised to this, in degrees, for the mesh cache.
pub const HALF_ANGLE_STEP_DEG: f32 = 0.5;

const AZIMUTH_SEGMENTS: usize = 32;
const POLAR_STEP_DEG: f32 = 7.5;

/// The shape of a sensor's accepted region, as a mesh cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VolumeShape {
    Sphere,
    /// Half-angle in [`HALF_ANGLE_STEP_DEG`] steps.
    Sector {
        half_steps: u16,
    },
}

impl VolumeShape {
    pub fn half_angle_deg(self) -> f32 {
        match self {
            Self::Sphere => 180.0,
            Self::Sector { half_steps } => f32::from(half_steps) * HALF_ANGLE_STEP_DEG,
        }
    }
}

/// Where and how one sensor's volume is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeSpec {
    pub shape: VolumeShape,
    pub origin: Vec3,
    /// Unit axis the sector opens around; +Z for a sphere.
    pub axis: Vec3,
    pub range_m: f32,
}

impl VolumeSpec {
    /// The transform that places the unit mesh.
    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.origin)
            .with_rotation(Quat::from_rotation_arc(Vec3::Z, self.axis))
            .with_scale(Vec3::splat(self.range_m))
    }

    /// Whether `point` lies inside the volume: the gate's own test.
    pub fn contains(&self, point: Vec3) -> bool {
        let offset = point - self.origin;
        let distance = offset.length();
        if distance > self.range_m {
            return false;
        }
        match self.shape {
            VolumeShape::Sphere => true,
            VolumeShape::Sector { .. } => {
                distance < 1e-6
                    || offset.angle_between(self.axis).to_degrees() <= self.shape.half_angle_deg()
            }
        }
    }
}

/// The volume a sensor's recorded state gates, or `None` when there is
/// nothing to draw: an unset range.
pub fn volume_for(
    position: Vec3,
    velocity: Vec3,
    fov_half_angle_deg: Option<f32>,
    max_range_m: Option<f32>,
) -> Option<VolumeSpec> {
    let range_m = max_range_m.filter(|range| range.is_finite() && *range > 0.0)?;
    let look = sensor_look_direction(
        [velocity.x as f64, velocity.y as f64, velocity.z as f64],
        DEFAULT_LOOK_DOWN_ANGLE_DEG,
    );
    let gated = fov_half_angle_deg
        .filter(|half| half.is_finite() && *half < FULL_SPHERE_HALF_ANGLE_DEG)
        .zip(look);
    Some(match gated {
        Some((half, look)) => VolumeSpec {
            shape: VolumeShape::Sector {
                half_steps: (half.max(0.0) / HALF_ANGLE_STEP_DEG).round() as u16,
            },
            origin: position,
            axis: Vec3::new(look[0] as f32, look[1] as f32, look[2] as f32),
            range_m,
        },
        None => VolumeSpec {
            shape: VolumeShape::Sphere,
            origin: position,
            axis: Vec3::Z,
            range_m,
        },
    })
}

/// The height below which no volume is drawn: the terrain's lowest point.
/// A sensor's range reaches below the ground, where nothing can be, and a
/// volume drawn there shows from under the terrain's edge at low pitch.
/// Between that floor and the local terrain the volume is hidden by the
/// terrain itself, except in that sliver.
pub fn volume_floor_z(scene: &crate::schema::ScenePackage) -> f32 {
    scene
        .environment
        .terrain_summary
        .as_ref()
        .and_then(|summary| summary.min_height_m)
        .unwrap_or(HologramMaterial::NO_FLOOR)
}

/// Whether a volume is drawn: its toggle is on and it does not enclose the
/// camera eye. For a convex volume culling alone would draw nothing from
/// inside; a sector past 90 degrees is concave and would not.
pub fn volume_shown(spec: &VolumeSpec, eye: Vec3, enabled: bool) -> bool {
    enabled && !spec.contains(eye)
}

/// A unit spherical sector around +Z, apex at the origin: the cap of the unit
/// sphere out to `half_angle_deg` from +Z, closed by a cone back to the apex.
/// At 180 degrees it is the whole sphere. Normals point outward.
pub fn sector_mesh(half_angle_deg: f32) -> Mesh {
    let half = half_angle_deg.clamp(0.5, 180.0).to_radians();
    let polar_segments = ((half.to_degrees() / POLAR_STEP_DEG).ceil() as usize).max(2);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let ring = AZIMUTH_SEGMENTS + 1;

    // Cap: polar rings from the pole to the rim.
    for p in 0..=polar_segments {
        let polar = half * p as f32 / polar_segments as f32;
        for a in 0..ring {
            let azimuth = std::f32::consts::TAU * a as f32 / AZIMUTH_SEGMENTS as f32;
            let v = Vec3::new(
                polar.sin() * azimuth.cos(),
                polar.sin() * azimuth.sin(),
                polar.cos(),
            );
            positions.push(v.to_array());
            normals.push(v.to_array());
        }
    }
    for p in 0..polar_segments {
        for a in 0..AZIMUTH_SEGMENTS {
            let i0 = (p * ring + a) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + ring as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // Cone: from the apex to the rim, unless the cap already closed.
    if half < std::f32::consts::PI - 1e-3 {
        let (rim_sin, rim_cos) = half.sin_cos();
        let base = positions.len() as u32;
        for a in 0..ring {
            let azimuth = std::f32::consts::TAU * a as f32 / AZIMUTH_SEGMENTS as f32;
            let rim = Vec3::new(rim_sin * azimuth.cos(), rim_sin * azimuth.sin(), rim_cos);
            // The cone's outward normal: perpendicular to the slant, pointing
            // away from the axis.
            let normal = Vec3::new(rim_cos * azimuth.cos(), rim_cos * azimuth.sin(), -rim_sin);
            positions.push([0.0, 0.0, 0.0]);
            normals.push(normal.to_array());
            positions.push(rim.to_array());
            normals.push(normal.to_array());
        }
        for a in 0..AZIMUTH_SEGMENTS as u32 {
            let apex = base + 2 * a;
            let rim = apex + 1;
            let next_apex = apex + 2;
            let next_rim = apex + 3;
            indices.extend_from_slice(&[apex, next_rim, rim, apex, next_apex, next_rim]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Linear colour of a field-of-view volume: dim, since the hologram adds up
/// to four times this at the silhouette.
pub const VOLUME_COLOR: Vec4 = Vec4::new(0.16, 0.11, 0.02, 0.9);

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct SensorVolume {
    pub key: EntityKey,
    pub shape: VolumeShape,
}

/// Which entity draws which platform's volume, and the cached topology.
#[derive(Resource, Default)]
pub struct SensorVolumes {
    epoch: Option<u64>,
    entities: HashMap<EntityKey, (Entity, VolumeShape)>,
    meshes: HashMap<VolumeShape, Handle<Mesh>>,
    material: Option<Handle<HologramMaterial>>,
    inputs: Option<(u64, u64, Vec3, bool)>,
    recomputes: u64,
}

impl SensorVolumes {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn shape_of(&self, key: EntityKey) -> Option<VolumeShape> {
        self.entities.get(&key).map(|(_, shape)| *shape)
    }

    pub fn entity(&self, key: EntityKey) -> Option<Entity> {
        self.entities.get(&key).map(|(entity, _)| *entity)
    }

    /// Distinct meshes built: one per shape, however many sensors share it.
    pub fn cached_shapes(&self) -> usize {
        self.meshes.len()
    }

    pub fn recomputes(&self) -> u64 {
        self.recomputes
    }
}

/// Keeps one volume entity per sensing platform. See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn sync_sensor_volumes(
    mut commands: Commands,
    markers: Res<CurrentRuntimeMarkers>,
    replay: Res<ReplayState>,
    visibility: Res<RuntimeOverlayVisibility>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    scene: Res<crate::schema::ScenePackage>,
    mut volumes: ResMut<SensorVolumes>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HologramMaterial>>,
    mut placed: Query<(&mut Transform, &mut Visibility), With<SensorVolume>>,
) {
    let eye = camera
        .single()
        .map(GlobalTransform::translation)
        .unwrap_or(Vec3::splat(f32::MAX));
    let inputs = (
        markers.epoch(),
        markers.pose_revision(),
        eye,
        visibility.fov_cones,
    );
    if volumes.inputs == Some(inputs) {
        return;
    }
    volumes.inputs = Some(inputs);
    volumes.recomputes += 1;

    if volumes.epoch != Some(markers.epoch()) {
        for (_, (entity, _)) in volumes.entities.drain() {
            commands.entity(entity).despawn();
        }
        volumes.epoch = Some(markers.epoch());
    }
    let material = volumes
        .material
        .get_or_insert_with(|| {
            materials.add(HologramMaterial::with_floor(
                VOLUME_COLOR,
                volume_floor_z(&scene),
            ))
        })
        .clone();

    let mut wanted: HashMap<EntityKey, VolumeSpec> = HashMap::new();
    if let Some(frame) = replay.current_frame() {
        for node in &frame.nodes {
            let Some(key) = markers.key_of(MarkerKind::Node, &node.node_id) else {
                continue;
            };
            let Some(marker) = markers.get(key) else {
                continue;
            };
            if let Some(spec) = volume_for(
                marker.position,
                marker.velocity,
                node.fov_half_angle_deg,
                node.max_range_m,
            ) {
                wanted.insert(key, spec);
            }
        }
    }

    // Gone, or changed shape: despawn.
    let stale: Vec<EntityKey> = volumes
        .entities
        .iter()
        .filter(|(key, (_, shape))| wanted.get(key).map(|spec| spec.shape) != Some(*shape))
        .map(|(key, _)| *key)
        .collect();
    for key in stale {
        if let Some((entity, _)) = volumes.entities.remove(&key) {
            commands.entity(entity).despawn();
        }
    }

    for (key, spec) in &wanted {
        let shown = if volume_shown(spec, eye, visibility.fov_cones) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        match volumes.entities.get(key) {
            Some((entity, _)) => {
                if let Ok((mut transform, mut visible)) = placed.get_mut(*entity) {
                    transform.set_if_neq(spec.transform());
                    visible.set_if_neq(shown);
                }
            }
            None => {
                let mesh = volumes
                    .meshes
                    .entry(spec.shape)
                    .or_insert_with(|| meshes.add(sector_mesh(spec.shape.half_angle_deg())))
                    .clone();
                let entity = commands
                    .spawn((
                        SensorVolume {
                            key: *key,
                            shape: spec.shape,
                        },
                        PresentationOnly,
                        Mesh3d(mesh),
                        MeshMaterial3d(material.clone()),
                        spec.transform(),
                        shown,
                    ))
                    .id();
                volumes.entities.insert(*key, (entity, spec.shape));
            }
        }
    }
}

/// How far above the terrain a range ring is draped, metres.
pub const RING_DRAPE_M: f32 = 1.5;
const RING_SEGMENTS: usize = 64;

/// A radar's range where it meets the ground: a ring of `range_m` around
/// `centre`, clipped to the scene, draped on the terrain. Line-list vertices.
pub fn range_ring_vertices(
    centre: Vec3,
    range_m: f32,
    bounds: &crate::schema::Bounds2d,
    ground: &dyn Fn(f32, f32) -> Option<f32>,
) -> Vec<[f32; 3]> {
    let mut vertices = Vec::with_capacity(RING_SEGMENTS * 2);
    for i in 0..RING_SEGMENTS {
        let a0 = std::f32::consts::TAU * i as f32 / RING_SEGMENTS as f32;
        let a1 = std::f32::consts::TAU * (i + 1) as f32 / RING_SEGMENTS as f32;
        let p0 = [centre.x + a0.cos() * range_m, centre.y + a0.sin() * range_m];
        let p1 = [centre.x + a1.cos() * range_m, centre.y + a1.sin() * range_m];
        let Some([c0, c1]) = crate::mission_zones::clip_segment_to_bounds(p0, p1, bounds) else {
            continue;
        };
        for c in [c0, c1] {
            let z = ground(c[0], c[1]).unwrap_or(centre.z) + RING_DRAPE_M;
            vertices.push([c[0], c[1], z]);
        }
    }
    vertices
}

/// A ring's entity, its mesh, and the `(centre, range)` it was built for.
type BuiltRing = (Entity, Handle<Mesh>, (Vec3, f32));

#[derive(Component, Debug, Clone, Copy)]
pub struct RangeRing {
    pub key: EntityKey,
}

/// One draped ring per radar, rebuilt only when the radar moves or its range
/// changes: stations do not, so in practice each is built once.
#[derive(Resource, Default)]
pub struct RangeRings {
    epoch: Option<u64>,
    entities: HashMap<EntityKey, BuiltRing>,
    material: Option<Handle<StandardMaterial>>,
    builds: u64,
}

impl RangeRings {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Ring meshes built or rebuilt so far.
    pub fn builds(&self) -> u64 {
        self.builds
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sync_range_rings(
    mut commands: Commands,
    markers: Res<CurrentRuntimeMarkers>,
    replay: Res<ReplayState>,
    visibility: Res<RuntimeOverlayVisibility>,
    scene: Res<crate::schema::ScenePackage>,
    zones: Res<crate::state::LoadedMissionZones>,
    mut rings: ResMut<RangeRings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shown: Query<&mut Visibility, With<RangeRing>>,
) {
    if rings.epoch != Some(markers.epoch()) {
        for (_, (entity, _, _)) in rings.entities.drain() {
            commands.entity(entity).despawn();
        }
        rings.epoch = Some(markers.epoch());
    }
    let material = rings
        .material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::srgba(0.3, 0.7, 0.95, 0.35),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        })
        .clone();
    let visible = if visibility.radar_rings {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };

    let mut present: Vec<EntityKey> = Vec::new();
    if let Some(frame) = replay.current_frame() {
        for node in frame.nodes.iter().filter(|n| n.sensor_type == "radar") {
            let (Some(key), Some(range_m)) = (
                markers.key_of(MarkerKind::Node, &node.node_id),
                node.max_range_m.filter(|r| *r > 0.0),
            ) else {
                continue;
            };
            let Some(marker) = markers.get(key) else {
                continue;
            };
            present.push(key);
            let built_for = (marker.position, range_m);
            let current = rings
                .entities
                .get(&key)
                .map(|(e, m, b)| (*e, m.clone(), *b));
            match current {
                Some((entity, _, built)) if built == built_for => {
                    if let Ok(mut v) = shown.get_mut(entity) {
                        v.set_if_neq(visible);
                    }
                }
                other => {
                    let vertices = range_ring_vertices(
                        marker.position,
                        range_m,
                        &scene.environment.bounds_xy_m,
                        &|x, y| zones.terrain_mesh.as_ref().and_then(|m| m.height_at(x, y)),
                    );
                    let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
                    rings.builds += 1;
                    match other {
                        Some((entity, handle, _)) => {
                            // Same entity and asset, new vertices.
                            if let Some(mut existing) = meshes.get_mut(&handle) {
                                *existing = mesh;
                            }
                            if let Ok(mut v) = shown.get_mut(entity) {
                                v.set_if_neq(visible);
                            }
                            rings.entities.insert(key, (entity, handle, built_for));
                        }
                        None => {
                            let handle = meshes.add(mesh);
                            let entity = commands
                                .spawn((
                                    RangeRing { key },
                                    PresentationOnly,
                                    Mesh3d(handle.clone()),
                                    MeshMaterial3d(material.clone()),
                                    Transform::IDENTITY,
                                    visible,
                                ))
                                .id();
                            rings.entities.insert(key, (entity, handle, built_for));
                        }
                    }
                }
            }
        }
    }
    rings.entities.retain(|key, (entity, _, _)| {
        let keep = present.contains(key);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Along the look, tilted down: the runtime's rule, not the velocity.
    #[test]
    fn a_moving_directional_sensor_opens_along_the_runtimes_look() {
        let spec = volume_for(
            Vec3::new(1.0, 2.0, 100.0),
            Vec3::new(10.0, 0.0, 0.0),
            Some(55.0),
            Some(650.0),
        )
        .unwrap();
        assert_eq!(spec.shape, VolumeShape::Sector { half_steps: 110 });
        let expected =
            sensor_look_direction([10.0, 0.0, 0.0], DEFAULT_LOOK_DOWN_ANGLE_DEG).unwrap();
        assert!(
            (spec.axis - Vec3::new(expected[0] as f32, expected[1] as f32, expected[2] as f32))
                .length()
                < 1e-6
        );
        assert!(spec.axis.z < 0.0, "tilted down");
        assert_eq!(spec.range_m, 650.0);
    }

    /// No look (too slow), or no half-angle, or a half-angle past a
    /// hemisphere's full sweep: the gate is skipped, so the region is the
    /// range sphere.
    #[test]
    fn without_a_gate_the_region_is_the_range_sphere() {
        let slow = volume_for(
            Vec3::ZERO,
            Vec3::new(0.3, 0.3, 5.0),
            Some(55.0),
            Some(100.0),
        )
        .unwrap();
        assert_eq!(slow.shape, VolumeShape::Sphere);
        let omni = volume_for(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), None, Some(100.0)).unwrap();
        assert_eq!(omni.shape, VolumeShape::Sphere);
        let full = volume_for(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Some(180.0),
            Some(100.0),
        )
        .unwrap();
        assert_eq!(full.shape, VolumeShape::Sphere);
    }

    /// An unset range bounds nothing, so nothing is drawn; the adapter's `0`
    /// already reads as unset.
    #[test]
    fn an_unset_range_draws_nothing() {
        assert_eq!(
            volume_for(Vec3::ZERO, Vec3::X * 10.0, Some(55.0), None),
            None
        );
        assert_eq!(
            volume_for(Vec3::ZERO, Vec3::X * 10.0, Some(55.0), Some(0.0)),
            None
        );
    }

    /// `contains` is the gate: range, and the half-angle around the look.
    #[test]
    fn the_volume_contains_exactly_what_the_gate_accepts() {
        let spec = volume_for(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Some(30.0),
            Some(100.0),
        )
        .unwrap();
        assert!(spec.contains(spec.axis * 50.0));
        assert!(!spec.contains(spec.axis * 101.0), "past the range");
        // Rotated about an axis perpendicular to the look, so the angle to it
        // is exactly the rotation.
        let perpendicular = spec.axis.any_orthonormal_vector();
        let off = Quat::from_axis_angle(perpendicular, 31f32.to_radians()) * spec.axis;
        assert!(!spec.contains(off * 50.0), "outside the half-angle");
        let inside = Quat::from_axis_angle(perpendicular, 29f32.to_radians()) * spec.axis;
        assert!(spec.contains(inside * 50.0));
    }

    /// The unit mesh is the sector: every vertex within the unit ball, the
    /// cap exactly on the sphere out to the half-angle, and outward normals.
    #[test]
    fn the_sector_mesh_is_the_unit_sector() {
        for half in [10.0f32, 55.0, 120.0, 180.0] {
            let mesh = sector_mesh(half);
            let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("positions");
            };
            let mut widest = 0.0f32;
            for p in positions {
                let v = Vec3::from_array(*p);
                assert!(v.length() <= 1.0 + 1e-5, "{half}: {v}");
                if v.length() > 0.999 {
                    widest = widest.max(v.angle_between(Vec3::Z).to_degrees());
                }
            }
            assert!((widest - half).abs() < 0.01, "{half}: cap reaches {widest}");
            let has_apex = positions
                .iter()
                .any(|p| Vec3::from_array(*p).length() < 1e-6);
            assert_eq!(
                has_apex,
                half < 180.0,
                "{half}: the cone closes to the apex"
            );
        }
    }

    /// A volume around the eye is hidden, including a concave one (a sector
    /// past 90 degrees) that culling alone would still partly draw.
    #[test]
    fn a_volume_around_the_camera_is_hidden() {
        let wide = volume_for(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Some(150.0),
            Some(100.0),
        )
        .unwrap();
        let inside = Quat::from_axis_angle(wide.axis.any_orthonormal_vector(), 120f32.to_radians())
            * wide.axis
            * 50.0;
        assert!(
            !volume_shown(&wide, inside, true),
            "the camera is inside the sector"
        );
        assert!(
            volume_shown(&wide, Vec3::new(500.0, 0.0, 0.0), true),
            "outside, shown"
        );
        assert!(
            !volume_shown(&wide, Vec3::new(500.0, 0.0, 0.0), false),
            "toggled off"
        );
    }

    #[test]
    fn volumes_stop_at_the_terrains_lowest_point() {
        let scene = crate::schema::ScenePackage::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
        )
        .unwrap();
        let floor = volume_floor_z(&scene);
        assert!((floor - 0.657_108_6).abs() < 1e-4, "{floor}");
        assert_eq!(
            HologramMaterial::new(Vec4::ONE).floor.x,
            HologramMaterial::NO_FLOOR
        );
    }

    fn square(half: f32) -> crate::schema::Bounds2d {
        crate::schema::Bounds2d {
            x_min_m: -half,
            x_max_m: half,
            y_min_m: -half,
            y_max_m: half,
        }
    }

    /// The ring is the range where it meets the ground, draped above the
    /// terrain, and clipped to the scene.
    #[test]
    fn a_range_ring_is_draped_at_the_range_and_clipped() {
        let centre = Vec3::new(100.0, -50.0, 20.0);
        let ground = |x: f32, _y: f32| Some(x * 0.01);
        let ring = range_ring_vertices(centre, 400.0, &square(5000.0), &ground);
        assert_eq!(ring.len(), 2 * RING_SEGMENTS);
        for v in &ring {
            let r = Vec2::new(v[0] - centre.x, v[1] - centre.y).length();
            assert!((r - 400.0).abs() < 1e-2, "radius {r}");
            assert!(
                (v[2] - (v[0] * 0.01 + RING_DRAPE_M)).abs() < 1e-3,
                "not draped: {v:?}"
            );
        }
        // Off the terrain: the station's own height.
        let off = range_ring_vertices(centre, 400.0, &square(5000.0), &|_, _| None);
        assert!(off
            .iter()
            .all(|v| (v[2] - (centre.z + RING_DRAPE_M)).abs() < 1e-3));
        // A scene smaller than the ring: every vertex on or inside it.
        let clipped = range_ring_vertices(centre, 400.0, &square(300.0), &ground);
        assert!(!clipped.is_empty());
        assert!(clipped
            .iter()
            .all(|v| v[0].abs() <= 300.0 + 1e-3 && v[1].abs() <= 300.0 + 1e-3));
        assert!(
            clipped.iter().any(|v| (v[0] - 300.0).abs() < 1e-3),
            "clipped to the edge, not dropped"
        );
    }

    #[test]
    fn half_angles_share_a_mesh_at_half_degree_steps() {
        let a = volume_for(Vec3::ZERO, Vec3::X * 10.0, Some(55.1), Some(1.0))
            .unwrap()
            .shape;
        let b = volume_for(Vec3::ZERO, Vec3::X * 10.0, Some(54.9), Some(1.0))
            .unwrap()
            .shape;
        let c = volume_for(Vec3::ZERO, Vec3::X * 10.0, Some(55.4), Some(1.0))
            .unwrap()
            .shape;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
