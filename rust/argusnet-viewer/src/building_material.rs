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

/// Entries in the surface table: material codes 0..`SURFACES`.
pub const SURFACES: usize = 24;

/// How each material class's surface responds to light (R4), by its code
/// (`argusnet_sim::scene_build::material_code`): perceptual roughness and
/// metalness, and whether the class sets them at all. Generic (0) and any
/// unused code keep the baked material's own. Metalness stays low: the
/// scene has no environment map, and a fully metallic surface with nothing
/// to reflect renders black, which would misstate the material.
pub fn surface(code: u32) -> Option<(f32, f32)> {
    Some(match code {
        1 => (0.85, 0.0),   // concrete
        2 => (0.90, 0.0),   // brick
        3 => (0.08, 0.0),   // glass: smooth, a dielectric
        4 => (0.40, 0.30),  // metal cladding
        5 => (0.80, 0.0),   // wood
        6 => (0.85, 0.0),   // stone
        7 => (0.90, 0.0),   // plaster
        11 => (0.80, 0.0),  // roof tiles
        12 => (0.50, 0.0),  // slate
        13 => (0.35, 0.30), // metal roof
        14 => (0.85, 0.0),  // concrete roof
        15 => (0.95, 0.0),  // membrane
        16 => (0.08, 0.0),  // glass roof
        17 => (1.00, 0.0),  // green roof
        _ => return None,
    })
}

#[derive(Clone, Debug, ShaderType, Reflect)]
pub struct FacadeSettings {
    pub window_color: Vec4,
    pub window: Vec4,
    /// Per material code: x roughness, y metalness, z 1 when the class sets
    /// them (see [`surface`]).
    pub surfaces: [Vec4; SURFACES],
}

impl Default for FacadeSettings {
    fn default() -> Self {
        Self {
            window_color: WINDOW_COLOR,
            window: Vec4::new(WINDOW_SPACING_M, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_SILL),
            surfaces: std::array::from_fn(|code| {
                surface(code as u32).map_or(Vec4::ZERO, |(roughness, metallic)| {
                    Vec4::new(roughness, metallic, 1.0, 0.0)
                })
            }),
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
            .add_systems(
                Update,
                (apply_facades_system, sync_building_opacity).chain(),
            );
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

/// The alpha mode a building part was given by its scene (its styled
/// translucency), kept the first time the opacity is set.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
struct StyledAlphaMode(AlphaMode);

/// Which opacity a building part's material was last set to, and which
/// material that was (the facade swaps it).
#[derive(Component, Debug, Clone, Copy, PartialEq)]
struct OpacitySet {
    see_through: bool,
    material: AssetId<StandardMaterial>,
    facade: Option<AssetId<BuildingMaterial>>,
}

/// The alpha mode a building is drawn with: the scene's styled one when
/// see-through is asked for, otherwise opaque (solid by default).
pub fn building_alpha_mode(styled: AlphaMode, see_through: bool) -> AlphaMode {
    if see_through {
        styled
    } else {
        AlphaMode::Opaque
    }
}

/// Buildings are solid unless Runtime Overlays asks for see-through
/// buildings: then each part is drawn at its scene's styled translucency
/// (the overlay GLB's `BLEND` and its vertex alpha), so what is behind a
/// building shows. Set once per part and change, never per frame.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync_building_opacity(
    mut commands: Commands,
    visibility: Res<crate::state::RuntimeOverlayVisibility>,
    roots: Query<Entity, With<BuildingLayerRoot>>,
    children: Query<&Children>,
    parts: Query<(
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&MeshMaterial3d<BuildingMaterial>>,
        Option<&StyledAlphaMode>,
        Option<&OpacitySet>,
    )>,
    mut standard: ResMut<Assets<StandardMaterial>>,
    mut facades: ResMut<Assets<BuildingMaterial>>,
) {
    let see_through = visibility.see_through_buildings;
    for root in &roots {
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Ok(node_children) = children.get(entity) {
                stack.extend(node_children.iter());
            }
            let Ok((plain, facade, styled, set)) = parts.get(entity) else {
                continue;
            };
            let current = OpacitySet {
                see_through,
                material: plain.map_or(AssetId::default(), |m| m.0.id()),
                facade: facade.map(|m| m.0.id()),
            };
            if plain.is_none() && facade.is_none() || set == Some(&current) {
                continue;
            }
            // The scene's own mode, kept from the first time this part is
            // seen (a facade copies the baked material, so it carries it).
            let baked = facade
                .and_then(|m| facades.get(&m.0))
                .map(|m| m.base.alpha_mode)
                .or_else(|| plain.and_then(|m| standard.get(&m.0)).map(|m| m.alpha_mode));
            let styled = match (styled, baked) {
                (Some(styled), _) => styled.0,
                (None, Some(mode)) => {
                    commands.entity(entity).insert(StyledAlphaMode(mode));
                    mode
                }
                (None, None) => continue,
            };
            let mode = building_alpha_mode(styled, see_through);
            if let Some(mut material) = facade.and_then(|m| facades.get_mut(&m.0)) {
                material.base.alpha_mode = mode;
            }
            if let Some(mut material) = plain.and_then(|m| standard.get_mut(&m.0)) {
                material.alpha_mode = mode;
            }
            commands.entity(entity).insert(current);
        }
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

    /// Every class the scene builder can write has a surface, within the
    /// table; generic keeps the baked one.
    #[test]
    fn every_material_class_has_a_surface_in_the_table() {
        use argusnet_sim::scene_build::{material_code, MaterialPart};
        let settings = FacadeSettings::default();
        for (part, classes) in [
            (
                MaterialPart::Facade,
                &[
                    "concrete", "brick", "glass", "metal", "wood", "stone", "plaster",
                ][..],
            ),
            (
                MaterialPart::Roof,
                &[
                    "tiles", "slate", "metal", "concrete", "membrane", "glass", "green",
                ][..],
            ),
        ] {
            for class in classes {
                let code = material_code(part, class) as usize;
                assert!(code < SURFACES, "{class}");
                assert_eq!(settings.surfaces[code].z, 1.0, "{class}");
            }
        }
        assert_eq!(settings.surfaces[0].z, 0.0, "generic keeps its own");
        // Glass is smooth and metal metallic; masonry is rough and not.
        assert!(surface(3).unwrap().0 < 0.2 && surface(4).unwrap().1 > 0.0);
        // Never so metallic that, with no environment to reflect, it goes black.
        assert!((0..SURFACES as u32)
            .filter_map(surface)
            .all(|(_, m)| m <= 0.5));
        assert!(surface(2).unwrap().0 > 0.8 && surface(2).unwrap().1 == 0.0);
    }

    #[test]
    fn buildings_are_solid_unless_see_through_is_asked_for() {
        assert_eq!(
            building_alpha_mode(AlphaMode::Blend, false),
            AlphaMode::Opaque
        );
        assert_eq!(
            building_alpha_mode(AlphaMode::Blend, true),
            AlphaMode::Blend
        );
        assert_eq!(
            building_alpha_mode(AlphaMode::Opaque, true),
            AlphaMode::Opaque
        );
        assert!(!crate::state::RuntimeOverlayVisibility::default().see_through_buildings);
    }

    /// A building part without the facade (a GLB the scene builder did not
    /// mark) keeps its plain material, which is set solid too, put back to
    /// the scene's mode when see-through is asked for, and left alone after.
    #[test]
    fn a_plain_building_part_follows_the_toggle() {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<BuildingMaterial>>()
            .init_resource::<crate::state::RuntimeOverlayVisibility>()
            .add_systems(Update, sync_building_opacity);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            });
        let part = app.world_mut().spawn(MeshMaterial3d(handle.clone())).id();
        app.world_mut()
            .spawn(BuildingLayerRoot)
            .add_children(&[part]);
        let mode = |app: &App| {
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap()
                .alpha_mode
        };
        app.update();
        assert_eq!(mode(&app), AlphaMode::Opaque);
        app.world_mut()
            .resource_mut::<crate::state::RuntimeOverlayVisibility>()
            .see_through_buildings = true;
        app.update();
        assert_eq!(mode(&app), AlphaMode::Blend);
        // Set once: a later change to the material is not overwritten.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&handle)
            .unwrap()
            .alpha_mode = AlphaMode::Mask(0.5);
        app.update();
        assert_eq!(mode(&app), AlphaMode::Mask(0.5));
        app.world_mut()
            .resource_mut::<crate::state::RuntimeOverlayVisibility>()
            .see_through_buildings = false;
        app.update();
        assert_eq!(mode(&app), AlphaMode::Opaque);
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
