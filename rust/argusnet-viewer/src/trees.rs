//! Trees (R2): the run's trees, from the replay's `meta.trees`, the same
//! trees the simulation sensed through and flew round.
//!
//! # One shape, the model's
//!
//! Each tree is drawn as the model holds it: a trunk, a cylinder from the
//! ground to the middle of the crown, and the crown, an ellipsoid of the
//! crown's radius between its base and the top. Species differ in colour and
//! faceting, never in volume (VIEWER_UI §18: no plausible lies), so what a
//! drone can be seen to pass through is what it passed through.
//!
//! # Batched and never tier-culled
//!
//! Every crown is one unit mesh per species scaled by its transform, every
//! trunk one shared unit cylinder, and the materials are shared per species
//! and season (leaf density in [`LEAF_LEVELS`] steps), so the renderer draws
//! all the trees of a kind in one batch. Beyond [`IMPOSTOR_DISTANCE_M`] from
//! the camera each tree is drawn as an impostor instead: two crossed
//! quads the size of the tree in the crown's colour. The switch is the
//! renderer's ([`VisibilityRange`]), so nothing is recomputed as the camera
//! moves, and no quality tier hides a tree: the near or the far form is
//! always drawn.
//!
//! # Seasons
//!
//! A deciduous crown's leaf density already carries the season's
//! `foliage_density_factor` (the builder applied it), so the crown's colour
//! runs from bare branches at density 0 to full leaf at 1.
//!
//! Trees are world structures: no layer preset owns them, and the Runtime
//! Overlays panel's Trees toggle hides and shows them. Picking ignores them
//! ([`PresentationOnly`]). Each tree in `meta.trees` is read on its own: one
//! the model refuses is not drawn and its reason is logged, and every other
//! tree still is.

use std::collections::HashMap;

use argusnet_world::trees::{Species, Tree};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::VisibilityRange;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::replay::ReplayState;
use crate::sensor_volumes::PresentationOnly;
use crate::state::RuntimeOverlayVisibility;

/// From this far from the camera, metres, a tree is drawn as its impostor.
pub const IMPOSTOR_DISTANCE_M: f32 = 2000.0;
/// How many steps of leaf density get a material of their own.
pub const LEAF_LEVELS: usize = 4;
/// Bare branches: what a crown of leaf density 0 looks like.
pub const BARE_COLOR: [f32; 3] = [0.42, 0.36, 0.30];
/// Bark.
pub const TRUNK_COLOR: [f32; 3] = [0.34, 0.25, 0.18];

/// A species' crown in full leaf.
pub fn leaf_color(species: Species) -> [f32; 3] {
    match species {
        Species::Conifer => [0.11, 0.26, 0.15],
        Species::Broadleaf => [0.22, 0.40, 0.16],
        Species::Palm => [0.38, 0.50, 0.19],
        Species::Shrub => [0.31, 0.40, 0.21],
    }
}

/// The step of leaf density a crown is drawn at: the nearest of
/// `0, 1/3, 2/3, 1`.
pub fn leaf_level(leaf_density: f64) -> usize {
    ((leaf_density.clamp(0.0, 1.0) * (LEAF_LEVELS - 1) as f64).round() as usize)
        .min(LEAF_LEVELS - 1)
}

/// The crown's colour at a step of leaf density, from bare to full leaf.
pub fn crown_color(species: Species, level: usize) -> [f32; 3] {
    let t = level as f32 / (LEAF_LEVELS - 1) as f32;
    let leaf = leaf_color(species);
    std::array::from_fn(|i| BARE_COLOR[i] + (leaf[i] - BARE_COLOR[i]) * t)
}

/// A species' unit crown: the unit sphere (its vertices on it, its poles on
/// the z axis and four of its meridians through the x and y axes, so it
/// spans exactly `[-1, 1]` each way), faceted by species (a conifer's, a
/// palm's and a shrub's coarsest, a broadleaf's finest), scaled per tree
/// to its crown's ellipsoid.
pub fn crown_mesh(species: Species) -> Mesh {
    let (sectors, stacks) = match species {
        Species::Conifer | Species::Shrub | Species::Palm => (8, 6),
        Species::Broadleaf => (12, 8),
    };
    Sphere::new(1.0)
        .mesh()
        .uv(sectors, stacks)
        .transformed_by(Transform::from_rotation(Quat::from_rotation_x(
            std::f32::consts::FRAC_PI_2,
        )))
}

/// The unit trunk: radius 1, from z = 0 to z = 1 (the viewer is Z-up),
/// eight-sided, scaled per tree.
pub fn trunk_mesh() -> Mesh {
    Mesh::from(Cylinder::new(1.0, 1.0).mesh().resolution(8)).transformed_by(Transform {
        translation: Vec3::new(0.0, 0.0, 0.5),
        rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        scale: Vec3::ONE,
    })
}

/// The unit impostor: two crossed quads, x and y in `[-1, 1]`, z in
/// `[0, 1]`, scaled per tree to its crown's width and its height.
pub fn impostor_mesh() -> Mesh {
    let positions: Vec<[f32; 3]> = vec![
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
    ];
    let normals: Vec<[f32; 3]> = vec![
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]))
}

/// Where a tree's three forms stand: the trunk (ground to the crown's
/// middle), the crown (its ellipsoid) and the impostor (ground to top,
/// the crown's width).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeTransforms {
    pub trunk: Transform,
    pub crown: Transform,
    pub impostor: Transform,
}

pub fn tree_transforms(tree: &Tree) -> TreeTransforms {
    let (x, y, base) = (
        tree.xy_m[0] as f32,
        tree.xy_m[1] as f32,
        tree.base_z_m as f32,
    );
    let centre = tree.crown_centre_m().map(|v| v as f32);
    let radius = tree.crown_radius_m as f32;
    let trunk = tree.trunk_radius_m as f32;
    TreeTransforms {
        trunk: Transform {
            translation: Vec3::new(x, y, base),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(trunk, trunk, (tree.trunk_top_z_m() - tree.base_z_m) as f32),
        },
        crown: Transform {
            translation: Vec3::from_array(centre),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(radius, radius, tree.crown_half_height_m() as f32),
        },
        impostor: Transform {
            translation: Vec3::new(x, y, base),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(radius, radius, tree.height_m as f32),
        },
    }
}

/// Which form of a tree an entity draws.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreePart {
    Trunk,
    Crown,
    Impostor,
}

pub struct TreePlugin;
impl Plugin for TreePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Trees>();
    }
}

/// The trees drawn for the current run, built once per run.
#[derive(Resource, Default)]
pub struct Trees {
    /// The replay epoch, and whether it had a meta, the trees were built for.
    built_for: Option<(u64, bool)>,
    entities: Vec<Entity>,
    /// The shared meshes: one crown and one impostor per species, one trunk.
    meshes: Vec<Handle<Mesh>>,
    /// Shared materials, kept across runs: per (species, leaf level) for
    /// crowns and impostors, and the bark.
    crown_materials: HashMap<(Species, usize), Handle<StandardMaterial>>,
    trunk_material: Option<Handle<StandardMaterial>>,
    trees: usize,
    builds: u64,
    refused: Vec<String>,
}

impl Trees {
    pub fn tree_count(&self) -> usize {
        self.trees
    }
    /// How many times the trees were built: once per run, never per frame.
    pub fn builds(&self) -> u64 {
        self.builds
    }
    /// How many crown materials exist: one per species and leaf level used.
    pub fn crown_material_count(&self) -> usize {
        self.crown_materials.len()
    }
    pub fn refused(&self) -> &[String] {
        &self.refused
    }
}

/// Build the run's trees when the replay changes, and show or hide them
/// with the Trees toggle.
#[allow(clippy::too_many_arguments)]
pub fn sync_trees(
    mut commands: Commands,
    replay: Res<ReplayState>,
    visibility: Res<RuntimeOverlayVisibility>,
    mut drawn: ResMut<Trees>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shown: Query<&mut Visibility, With<TreePart>>,
) {
    let visible = if visibility.trees {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut shown {
        v.set_if_neq(visible);
    }
    let run = (
        replay.epoch(),
        replay.document.as_ref().is_some_and(|d| d.meta.is_some()),
    );
    if drawn.built_for == Some(run) {
        return;
    }
    let drawn = drawn.as_mut();
    drawn.built_for = Some(run);
    drawn.builds += 1;
    for entity in drawn.entities.drain(..) {
        commands.entity(entity).despawn();
    }
    for handle in drawn.meshes.drain(..) {
        meshes.remove(&handle);
    }
    drawn.trees = 0;
    let read = replay
        .document
        .as_ref()
        .map(|d| d.trees())
        .unwrap_or_default();
    for reason in &read.refused {
        warn!("tree not drawn: {reason}");
    }
    drawn.refused = read.refused;
    if read.trees.is_empty() {
        return;
    }
    let trunk_mesh = meshes.add(trunk_mesh());
    drawn.meshes.push(trunk_mesh.clone());
    let trunk_material = drawn
        .trunk_material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::srgb(TRUNK_COLOR[0], TRUNK_COLOR[1], TRUNK_COLOR[2]),
                perceptual_roughness: 0.95,
                ..default()
            })
        })
        .clone();
    let mut crown_meshes: HashMap<Species, (Handle<Mesh>, Handle<Mesh>)> = HashMap::new();
    let near = VisibilityRange::abrupt(0.0, IMPOSTOR_DISTANCE_M);
    let far = VisibilityRange::abrupt(IMPOSTOR_DISTANCE_M, f32::MAX);
    for tree in &read.trees {
        let (crown, impostor) = crown_meshes
            .entry(tree.species)
            .or_insert_with(|| {
                let crown = meshes.add(crown_mesh(tree.species));
                let impostor = meshes.add(impostor_mesh());
                drawn.meshes.push(crown.clone());
                drawn.meshes.push(impostor.clone());
                (crown, impostor)
            })
            .clone();
        let level = leaf_level(tree.leaf_density);
        let material = drawn
            .crown_materials
            .entry((tree.species, level))
            .or_insert_with(|| {
                let [r, g, b] = crown_color(tree.species, level);
                materials.add(StandardMaterial {
                    base_color: Color::srgb(r, g, b),
                    perceptual_roughness: 0.9,
                    // The impostor's quads are seen from both sides.
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                })
            })
            .clone();
        let at = tree_transforms(tree);
        if tree.trunk_radius_m > 0.0 {
            drawn.entities.push(
                commands
                    .spawn((
                        TreePart::Trunk,
                        PresentationOnly,
                        Mesh3d(trunk_mesh.clone()),
                        MeshMaterial3d(trunk_material.clone()),
                        at.trunk,
                        near.clone(),
                        visible,
                    ))
                    .id(),
            );
        }
        drawn.entities.push(
            commands
                .spawn((
                    TreePart::Crown,
                    PresentationOnly,
                    Mesh3d(crown),
                    MeshMaterial3d(material.clone()),
                    at.crown,
                    near.clone(),
                    visible,
                ))
                .id(),
        );
        drawn.entities.push(
            commands
                .spawn((
                    TreePart::Impostor,
                    PresentationOnly,
                    Mesh3d(impostor),
                    MeshMaterial3d(material),
                    at.impostor,
                    far.clone(),
                    visible,
                ))
                .id(),
        );
        drawn.trees += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn tree() -> Tree {
        Tree {
            id: "t".into(),
            xy_m: [10.0, -20.0],
            base_z_m: 100.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 5.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: 0.8,
        }
        .validated()
        .unwrap()
    }

    fn positions(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(values) => values.clone(),
            other => panic!("{other:?}"),
        }
    }

    fn extent(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in points {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        (lo, hi)
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    /// Each form, placed by its transform, fills the model's volume: the
    /// crown its ellipsoid, the trunk ground to the crown's middle, the
    /// impostor ground to top across the crown.
    #[test]
    fn each_form_stands_where_the_model_puts_the_tree() {
        let tree = tree();
        let at = tree_transforms(&tree);
        let placed =
            |mesh: Mesh, transform: Transform| extent(&positions(&mesh.transformed_by(transform)));
        // The crown: x 5..15, y -25..-15, z 108..120.
        let (lo, hi) = placed(crown_mesh(tree.species), at.crown);
        assert!(
            close(lo, [5.0, -25.0, 108.0]) && close(hi, [15.0, -15.0, 120.0]),
            "{lo:?} {hi:?}"
        );
        // The trunk: radius 0.3, z 100..114.
        let (lo, hi) = placed(trunk_mesh(), at.trunk);
        assert!(
            close(lo, [9.7, -20.3, 100.0]) && close(hi, [10.3, -19.7, 114.0]),
            "{lo:?} {hi:?}"
        );
        // The impostor: the crown's width, ground to top.
        let (lo, hi) = placed(impostor_mesh(), at.impostor);
        assert!(
            close(lo, [5.0, -25.0, 100.0]) && close(hi, [15.0, -15.0, 120.0]),
            "{lo:?} {hi:?}"
        );
    }

    /// Every species' crown is the same unit sphere in volume, only its
    /// faceting differs; the unit trunk and impostor are what they say.
    #[test]
    fn the_unit_meshes_are_unit() {
        for species in Species::ALL {
            let (lo, hi) = extent(&positions(&crown_mesh(species)));
            assert!(close(lo, [-1.0; 3]) && close(hi, [1.0; 3]), "{species:?}");
            for p in positions(&crown_mesh(species)) {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!((r - 1.0).abs() < 1e-4, "{species:?} {p:?}");
            }
        }
        let (lo, hi) = extent(&positions(&trunk_mesh()));
        assert!(
            close(lo, [-1.0, -1.0, 0.0]) && close(hi, [1.0, 1.0, 1.0]),
            "{lo:?} {hi:?}"
        );
        let (lo, hi) = extent(&positions(&impostor_mesh()));
        assert!(
            close(lo, [-1.0, -1.0, 0.0]) && close(hi, [1.0, 1.0, 1.0]),
            "{lo:?} {hi:?}"
        );
        assert!(
            positions(&crown_mesh(Species::Broadleaf)).len()
                > positions(&crown_mesh(Species::Conifer)).len(),
            "a broadleaf is faceted finer"
        );
    }

    /// Leaf density in four steps, from bare branches to the species' full
    /// leaf.
    #[test]
    fn crowns_run_from_bare_to_full_leaf() {
        assert_eq!(leaf_level(0.0), 0);
        assert_eq!(leaf_level(0.15), 0);
        assert_eq!(leaf_level(0.2), 1);
        assert_eq!(leaf_level(0.6), 2);
        assert_eq!(leaf_level(0.9), 3);
        assert_eq!(leaf_level(1.0), 3);
        for species in Species::ALL {
            assert!(close(crown_color(species, 0), BARE_COLOR));
            assert!(close(
                crown_color(species, LEAF_LEVELS - 1),
                leaf_color(species)
            ));
        }
        assert_ne!(leaf_color(Species::Conifer), leaf_color(Species::Palm));
    }
}
