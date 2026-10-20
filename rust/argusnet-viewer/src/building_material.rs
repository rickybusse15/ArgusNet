//! Building facades (R1): window rows from the data, and material classes.
//!
//! The scene builder colours each roofed building's walls and roof by its
//! material class (`scene_build::facade_colour`, `roof_colour`; the layer's
//! own colour for `generic`), and gives its walls window coordinates: the
//! height above the building's ground in storeys, the distance along the
//! wall, and the level count. [`BuildingMaterial`] is the `StandardMaterial`
//! the GLB carried, extended to draw window rows from those, so lighting,
//! shadows, fog and the layer's translucency stay what they were.
//!
//! It is applied only to parts the scene builder marked as facades (glTF
//! mesh `extras` `{"argusnet_facade": 1}`) that also carry both coordinate
//! sets: texture coordinates in some other file are never read as storeys.
//! Every other
//! building, wall and part keeps its baked material untouched, and no
//! windows are drawn on a building whose storeys the data does not give
//! (VIEWER_UI §18: no plausible lies).

use bevy::asset::embedded_asset;
use bevy::gltf::GltfMeshExtras;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::world_serialization::WorldAssetRoot;

use crate::schema::ScenePackage;

/// The `semantic_kind` of the layers whose meshes may carry facades.
const BUILDINGS_SEMANTIC_KIND: &str = "buildings";

/// Glass: a dark blue-grey, and how strongly it replaces the wall colour.
pub const WINDOW_COLOR: Vec4 = Vec4::new(0.16, 0.22, 0.30, 0.85);
/// Metres between window columns.
pub const WINDOW_SPACING_M: f32 = 3.0;
/// A window's width as a share of its column, its height and its sill as
/// shares of a storey.
pub const WINDOW_WIDTH: f32 = 0.55;
pub const WINDOW_HEIGHT: f32 = 0.45;
pub const WINDOW_SILL: f32 = 0.30;

#[derive(Clone, Debug, ShaderType, Reflect)]
pub struct FacadeSettings {
    pub window_color: Vec4,
    pub window: Vec4,
}

impl Default for FacadeSettings {
    fn default() -> Self {
        Self {
            window_color: WINDOW_COLOR,
            window: Vec4::new(WINDOW_SPACING_M, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_SILL),
        }
    }
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub struct FacadeExtension {
    #[uniform(100)]
    pub settings: FacadeSettings,
}

impl MaterialExtension for FacadeExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://argusnet_viewer/shaders/building.wesl".into()
    }
}

pub type BuildingMaterial = ExtendedMaterial<StandardMaterial, FacadeExtension>;

/// Marks a spawned layer root whose style declares it buildings.
#[derive(Component)]
pub struct BuildingLayerRoot;

/// Marks a layer root whose facades are applied.
#[derive(Component)]
struct FacadesApplied;

pub fn is_building_layer(scene_package: &ScenePackage, style_id: &str) -> bool {
    scene_package
        .style_for(style_id)
        .is_some_and(|style| style.semantic_kind == BUILDINGS_SEMANTIC_KIND)
}

/// Whether a mesh carries the scene builder's window coordinates.
pub fn has_facade_coordinates(mesh: &Mesh) -> bool {
    mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some() && mesh.attribute(Mesh::ATTRIBUTE_UV_1).is_some()
}

/// Whether a glTF mesh's `extras` mark it as the scene builder's facade, in
/// the version this viewer reads.
pub fn is_marked_facade(extras: Option<&GltfMeshExtras>) -> bool {
    extras
        .and_then(|extras| serde_json::from_str::<serde_json::Value>(&extras.value).ok())
        .and_then(|value| {
            value
                .get(argusnet_sim::scene_export::FACADE_EXTRAS_KEY)?
                .as_u64()
        })
        == Some(u64::from(argusnet_sim::scene_export::FACADE_EXTRAS_VERSION))
}

pub struct BuildingMaterialPlugin;

impl Plugin for BuildingMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/building.wesl");
        app.add_plugins(MaterialPlugin::<BuildingMaterial>::default())
            .add_systems(Update, apply_facades_system);
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn apply_facades_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    roots: Query<(Entity, &WorldAssetRoot), (With<BuildingLayerRoot>, Without<FacadesApplied>)>,
    children: Query<&Children>,
    parts: Query<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        Option<&GltfMeshExtras>,
    )>,
    meshes: Res<Assets<Mesh>>,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut building_materials: ResMut<Assets<BuildingMaterial>>,
) {
    for (root, asset) in &roots {
        if !asset_server.is_loaded_with_dependencies(&asset.0) {
            continue;
        }
        let mut seen = 0usize;
        let mut converted = 0usize;
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Ok(node_children) = children.get(entity) {
                stack.extend(node_children.iter());
            }
            let Ok((mesh, material, extras)) = parts.get(entity) else {
                continue;
            };
            seen += 1;
            if !is_marked_facade(extras) || !meshes.get(&mesh.0).is_some_and(has_facade_coordinates)
            {
                continue;
            }
            let base = standard_materials
                .get(&material.0)
                .cloned()
                .unwrap_or_default();
            let facade = building_materials.add(BuildingMaterial {
                base,
                extension: FacadeExtension::default(),
            });
            commands
                .entity(entity)
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .insert(MeshMaterial3d(facade));
            converted += 1;
        }
        // The glTF's entities appear a frame or more after it loads: wait
        // until there are some, then mark the root done.
        if seen == 0 {
            continue;
        }
        debug!("facades on {converted} of {seen} building mesh(es)");
        commands.entity(root).insert(FacadesApplied);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::PrimitiveTopology;

    #[test]
    fn only_meshes_with_both_coordinate_sets_get_facades() {
        let base = || {
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3])
        };
        assert!(!has_facade_coordinates(&base()));
        let one = base().with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; 3]);
        assert!(!has_facade_coordinates(&one));
        let both = one.with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0f32; 2]; 3]);
        assert!(has_facade_coordinates(&both));
    }

    /// Only the scene builder's own marker, in this version, makes a facade:
    /// a lightmapped file's two coordinate sets alone do not.
    #[test]
    fn only_marked_parts_are_facades() {
        let extras = |value: &str| GltfMeshExtras {
            value: value.to_string(),
        };
        assert!(!is_marked_facade(None));
        assert!(is_marked_facade(Some(&extras(r#"{"argusnet_facade": 1}"#))));
        assert!(!is_marked_facade(Some(&extras(
            r#"{"argusnet_facade": 2}"#
        ))));
        assert!(!is_marked_facade(Some(&extras(r#"{"lightmap": true}"#))));
        assert!(!is_marked_facade(Some(&extras("not json"))));
    }
}
