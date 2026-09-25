//! Mission zones as extruded boundary walls (P2.2).
//!
//! Each overlap group's merged outer contour (`mission_zones`' marching-squares
//! output, clipped to the scene exactly as the gizmo outline clips it) is
//! extruded into a wall: from the terrain under each vertex up by a height
//! that scales with the scene, taller for exclusion zones. One mesh per
//! group, rebuilt only when the overlap model's `generation` changes; the
//! materials are restyled, not rebuilt, when the focus changes. Walls use the
//! additive hologram material, so they never hide what is behind them, and
//! carry [`crate::sensor_volumes::PresentationOnly`]. The gizmo outline now
//! draws only with the diagnostic glyphs on; the centre glyphs and badges,
//! which have no mesh equivalent, stay as they were.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::marker_meshes::HologramMaterial;
use crate::mission_zones::{
    average_zone_height, clip_segments_to_bounds, zone_color_rgba, ZONE_CONTOUR_MIN_CELL_SIZE_M,
};
use crate::schema::{Bounds2d, ScenePackage};
use crate::sensor_volumes::PresentationOnly;
use crate::state::{
    LoadedMissionZones, RuntimeOverlayVisibility, ZoneFocus, ZoneOverlapGroup, ZoneOverlapModel,
};

/// Wall height as a fraction of the scene's diagonal: about 70 m on the
/// bundled 6 km site.
pub const WALL_HEIGHT_FRACTION: f32 = 0.012;
/// Exclusion walls are this much taller.
pub const EXCLUSION_HEIGHT_FACTOR: f32 = 1.5;
/// Brightness of a wall outside the focus, relative to a focused one.
pub const UNFOCUSED_DIM: f32 = 0.3;

/// The dominant zone type of a group: most zones, ties by name.
pub fn group_type(group: &ZoneOverlapGroup) -> &str {
    group
        .type_counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(zone_type, _)| zone_type.as_str())
        .unwrap_or("unknown")
}

/// A group's wall height on a scene: a fraction of the diagonal, taller for
/// an exclusion group.
pub fn wall_height(bounds: &Bounds2d, group: &ZoneOverlapGroup) -> f32 {
    let span = bounds.span_xy();
    let factor = if group_type(group) == "exclusion" {
        EXCLUSION_HEIGHT_FACTOR
    } else {
        1.0
    };
    span[0].hypot(span[1]) * WALL_HEIGHT_FRACTION * factor
}

/// The wall along a contour: one vertical quad per segment, from `ground`
/// under each endpoint to `height_m` above it.
pub fn wall_mesh(
    segments: &[[[f32; 2]; 2]],
    height_m: f32,
    ground: &dyn Fn(f32, f32) -> f32,
) -> Option<Mesh> {
    if segments.is_empty() {
        return None;
    }
    let mut positions = Vec::with_capacity(segments.len() * 4);
    let mut normals = Vec::with_capacity(segments.len() * 4);
    let mut indices = Vec::with_capacity(segments.len() * 6);
    for segment in segments {
        let [a, b] = *segment;
        let (za, zb) = (ground(a[0], a[1]), ground(b[0], b[1]));
        let along = Vec2::new(b[0] - a[0], b[1] - a[1]).normalize_or(Vec2::X);
        let normal = [along.y, -along.x, 0.0];
        // Both faces, so the wall reads from inside the zone and out: the
        // hologram material culls back faces.
        for (facing, flip) in [(normal, false), ([-normal[0], -normal[1], 0.0], true)] {
            let base = positions.len() as u32;
            positions.extend_from_slice(&[
                [a[0], a[1], za],
                [b[0], b[1], zb],
                [a[0], a[1], za + height_m],
                [b[0], b[1], zb + height_m],
            ]);
            normals.extend_from_slice(&[facing; 4]);
            if flip {
                indices.extend_from_slice(&[
                    base,
                    base + 2,
                    base + 1,
                    base + 1,
                    base + 2,
                    base + 3,
                ]);
            } else {
                indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base + 1,
                    base + 3,
                    base + 2,
                ]);
            }
        }
    }
    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

/// The linear colour a group's wall draws with: its type's colour, dimmed
/// for the additive material, and further outside the focus.
pub fn wall_color(group: &ZoneOverlapGroup, focus: &ZoneFocus) -> Vec4 {
    let [r, g, b, _] = zone_color_rgba(group_type(group), 1.0);
    let focused = match focus {
        ZoneFocus::None => true,
        ZoneFocus::Group(id) => *id == group.group_id,
        ZoneFocus::Zone(zone) => group.zone_indices.contains(zone),
    };
    // The hologram lights a surface faced head-on at half its colour and at
    // four times it edge-on, so a wall facing the camera needs this much to
    // read at all.
    let dim = if focused { 0.35 } else { 0.35 * UNFOCUSED_DIM };
    let linear = LinearRgba::from(Color::srgb(r, g, b));
    Vec4::new(linear.red * dim, linear.green * dim, linear.blue * dim, 0.9)
}

#[derive(Component, Debug, Clone, Copy)]
pub struct ZoneWall {
    pub group_id: usize,
}

/// The wall entities and what they were built from.
#[derive(Resource, Default)]
pub struct ZoneWalls {
    /// `group_id -> (entity, material)`.
    entities: HashMap<usize, (Entity, Handle<HologramMaterial>)>,
    generation: Option<u64>,
    focus: Option<ZoneFocus>,
    builds: u64,
}

impl ZoneWalls {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Wall geometry rebuilds so far: one per overlap-model generation.
    pub fn builds(&self) -> u64 {
        self.builds
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sync_zone_walls(
    mut commands: Commands,
    zones: Res<LoadedMissionZones>,
    model: Res<ZoneOverlapModel>,
    visibility: Res<RuntimeOverlayVisibility>,
    scene: Res<ScenePackage>,
    mut walls: ResMut<ZoneWalls>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HologramMaterial>>,
    mut shown: Query<&mut Visibility, With<ZoneWall>>,
) {
    let visible = if visibility.zones {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut shown {
        v.set_if_neq(visible);
    }

    if walls.generation != Some(model.generation) {
        for (_, (entity, _)) in walls.entities.drain() {
            commands.entity(entity).despawn();
        }
        walls.generation = Some(model.generation);
        walls.focus = Some(model.focus.clone());
        walls.builds += 1;
        let bounds: &Bounds2d = &scene.environment.bounds_xy_m;
        let inset = bounds.inset(ZONE_CONTOUR_MIN_CELL_SIZE_M);
        for group in &model.groups {
            let clipped = clip_segments_to_bounds(&group.contour_segments, &inset);
            let fallback = average_zone_height(&zones.zones, &group.zone_indices);
            let ground = |x: f32, y: f32| {
                zones
                    .terrain_mesh
                    .as_ref()
                    .and_then(|m| m.height_at(x, y))
                    .unwrap_or(fallback)
            };
            let Some(mesh) = wall_mesh(&clipped, wall_height(bounds, group), &ground) else {
                continue;
            };
            let material = materials.add(HologramMaterial::new(wall_color(group, &model.focus)));
            let entity = commands
                .spawn((
                    ZoneWall {
                        group_id: group.group_id,
                    },
                    PresentationOnly,
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    visible,
                ))
                .id();
            walls.entities.insert(group.group_id, (entity, material));
        }
        return;
    }

    // Same geometry; the focus may have moved.
    if walls.focus.as_ref() != Some(&model.focus) {
        walls.focus = Some(model.focus.clone());
        for group in &model.groups {
            if let Some((_, handle)) = walls.entities.get(&group.group_id) {
                if let Some(mut material) = materials.get_mut(handle) {
                    material.color = wall_color(group, &model.focus);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wall_rises_from_the_ground_under_each_vertex() {
        let segments = [[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 10.0]]];
        let mesh = wall_mesh(&segments, 50.0, &|x, _| x).unwrap();
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        assert_eq!(p.len(), 16, "two segments, both faces");
        assert_eq!(p[0], [0.0, 0.0, 0.0]);
        assert_eq!(p[1], [10.0, 0.0, 10.0], "ground follows the terrain");
        assert_eq!(p[3], [10.0, 0.0, 60.0], "height above that ground");
        assert!(wall_mesh(&[], 50.0, &|_, _| 0.0).is_none());
    }

    fn group(id: usize, zone_type: &str) -> ZoneOverlapGroup {
        ZoneOverlapGroup {
            group_id: id,
            zone_indices: vec![id],
            centroid_xy: [0.0, 0.0],
            contour_segments: Vec::new(),
            type_counts: [(zone_type.to_string(), 1)].into_iter().collect(),
            anchor_xy: [0.0, 0.0],
            bounds_xy: [0.0; 4],
        }
    }

    #[test]
    fn exclusion_walls_are_taller_and_height_follows_the_scene() {
        let bounds = Bounds2d {
            x_min_m: -3000.0,
            x_max_m: 3000.0,
            y_min_m: -4000.0,
            y_max_m: 4000.0,
        };
        let plain = wall_height(&bounds, &group(0, "objective"));
        assert!((plain - 10_000.0 * WALL_HEIGHT_FRACTION).abs() < 1e-3);
        let exclusion = wall_height(&bounds, &group(1, "exclusion"));
        assert!((exclusion - plain * EXCLUSION_HEIGHT_FACTOR).abs() < 1e-3);
    }

    fn zone(id: &str, zone_type: &str, x: f32, y: f32) -> crate::replay::MissionZone {
        crate::replay::MissionZone {
            zone_id: id.into(),
            zone_type: zone_type.into(),
            center: [x, y, 5.0],
            radius_m: 200.0,
            priority: 1,
            label: id.into(),
        }
    }

    fn wall_app(zones: Vec<crate::replay::MissionZone>) -> App {
        let model = crate::mission_zones::build_zone_overlap_model(&zones, ZoneFocus::None);
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<HologramMaterial>>()
            .init_resource::<ZoneWalls>()
            .insert_resource(LoadedMissionZones {
                zones,
                terrain_mesh: None,
            })
            .insert_resource(model)
            .insert_resource(RuntimeOverlayVisibility::default())
            .insert_resource(
                ScenePackage::load(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
                )
                .unwrap(),
            )
            .add_systems(Update, sync_zone_walls);
        app
    }

    /// Geometry is rebuilt when the overlap model's generation moves, and
    /// only then; a focus change restyles the existing materials.
    #[test]
    fn walls_rebuild_per_generation_and_restyle_on_focus() {
        let mut app = wall_app(vec![
            zone("a", "exclusion", 0.0, 0.0),
            zone("b", "objective", 1500.0, 0.0),
        ]);
        app.update();
        app.update();
        assert_eq!(app.world().resource::<ZoneWalls>().builds(), 1);
        assert_eq!(app.world().resource::<ZoneWalls>().len(), 2);

        // Focus on the first group: no rebuild, the second dims.
        let (handle_b, before) = {
            let walls = app.world().resource::<ZoneWalls>();
            let handle = walls.entities[&1].1.clone();
            let color = app
                .world()
                .resource::<Assets<HologramMaterial>>()
                .get(&handle)
                .unwrap()
                .color;
            (handle, color)
        };
        app.world_mut().resource_mut::<ZoneOverlapModel>().focus = ZoneFocus::Group(0);
        app.update();
        assert_eq!(
            app.world().resource::<ZoneWalls>().builds(),
            1,
            "a focus change rebuilt geometry"
        );
        let after = app
            .world()
            .resource::<Assets<HologramMaterial>>()
            .get(&handle_b)
            .unwrap()
            .color;
        assert!(after.y < before.y, "the unfocused wall was not dimmed");

        // A new model generation, now with three zones: rebuilt.
        let zones = vec![
            zone("a", "exclusion", 0.0, 0.0),
            zone("b", "objective", 1500.0, 0.0),
            zone("c", "objective", -1500.0, 0.0),
        ];
        let mut model = crate::mission_zones::build_zone_overlap_model(&zones, ZoneFocus::None);
        model.generation = app.world().resource::<ZoneOverlapModel>().generation + 1;
        app.world_mut().resource_mut::<LoadedMissionZones>().zones = zones;
        app.insert_resource(model);
        app.update();
        let walls = app.world().resource::<ZoneWalls>();
        assert_eq!(walls.builds(), 2);
        assert_eq!(walls.len(), 3);
    }

    #[test]
    fn the_focus_dims_the_other_groups_only() {
        let (a, b) = (group(0, "exclusion"), group(1, "objective"));
        let open = wall_color(&a, &ZoneFocus::None);
        let focused = wall_color(&a, &ZoneFocus::Group(0));
        let other = wall_color(&b, &ZoneFocus::Group(0));
        assert_eq!(open, focused);
        assert!(other.y < wall_color(&b, &ZoneFocus::None).y);
        assert!(open.x > open.y, "exclusion is red");
    }
}
