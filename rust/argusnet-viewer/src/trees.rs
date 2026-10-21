//! Trees (R2): the run's trees, from the replay's `meta.trees`, the same
//! trees the simulation sensed through and flew round.
//!
//! # Shaped by species, inside the model's crown
//!
//! Each tree is drawn within the volume the model holds: a tapering trunk
//! from the ground to the middle of the crown, and a crown inside the
//! crown's ellipsoid, filling most of it. Within it, species differ in
//! shape: lobed broadleaf clusters, tiered and serrated conifers, palms'
//! arching fronds, low shrub mounds. Each tree takes one of
//! [`CROWN_VARIANTS`] meshes, one of [`CROWN_TINTS`] tints and its own turn
//! from its id, so a forest is not one tree repeated, and the same every
//! run. Faces are flat-shaded, lighter towards the top, and jittered, so
//! the foliage has texture without an image. Nothing is drawn beyond the
//! model's crown (VIEWER_UI §18: no plausible lies).
//!
//! # Batched and never tier-culled
//!
//! Every crown is one unit mesh per species scaled by its transform, every
//! trunk one shared unit cylinder, and the materials are shared per species
//! and season (leaf density in [`LEAF_LEVELS`] steps), so the renderer draws
//! all the trees of a kind in one batch. Beyond [`IMPOSTOR_DISTANCE_M`] from
//! the camera each tree is drawn as an impostor instead: two crossed
//! planes cut to the species' silhouette, the size of the tree, in the
//! crown's colour. The switch is the
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

/// Mesh variants per species: each tree takes one by its id, so a forest
/// is not one tree repeated.
pub const CROWN_VARIANTS: u32 = 4;
/// Tints per species and leaf level: each tree takes one by its id.
pub const CROWN_TINTS: usize = 3;
/// The tints' value multipliers.
pub const TINT_VALUES: [f32; CROWN_TINTS] = [0.88, 1.0, 1.1];

/// A stable number from a tree's id (FNV-1a), for its variant, tint and
/// turn: the same every run, different between trees.
pub fn tree_hash(id: &str) -> u64 {
    id.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A tree's crown variant (`0..CROWN_VARIANTS`), tint (`0..CROWN_TINTS`)
/// and turn about its axis, radians.
pub fn tree_look(id: &str) -> (u32, usize, f32) {
    let h = tree_hash(id);
    (
        (h % u64::from(CROWN_VARIANTS)) as u32,
        ((h >> 8) % CROWN_TINTS as u64) as usize,
        ((h >> 16) % 3600) as f32 / 3600.0 * std::f32::consts::TAU,
    )
}

/// Smooth, deterministic noise in about [-1, 1] over a direction.
fn lumps(p: Vec3, seed: f32) -> f32 {
    0.5 * (p.x * 5.3 + p.y * 3.1 + seed).sin() * (p.y * 4.7 - p.z * 2.9 + 1.7 * seed).cos()
        + 0.5 * (p.z * 6.1 + p.x * 2.3 - seed).sin()
}

/// Triangles with flat normals and per-face colour, into a mesh.
#[derive(Default)]
struct Facets {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colours: Vec<[f32; 4]>,
}

impl Facets {
    /// Add `[a, b, c]`, held inside the unit sphere, facing out from
    /// `inside` (flipped if need be), shaded
    /// lighter towards the top of the unit crown and jittered per face, so
    /// the foliage reads as clumps of leaves in light and shade.
    fn add(&mut self, tri: [Vec3; 3], inside: Vec3, seed: f32) {
        // Every vertex inside the unit sphere: nothing drawn beyond the
        // model's crown, whatever the shape.
        let [mut a, b, mut c] = tri.map(|p| if p.length() > 1.0 { p.normalize() } else { p });
        let mut n = (b - a).cross(c - a);
        if n.length_squared() < 1e-12 {
            return;
        }
        let centre = (a + b + c) / 3.0;
        if n.dot(centre - inside) < 0.0 {
            std::mem::swap(&mut a, &mut c);
            n = -n;
        }
        let n = n.normalize();
        let height = ((centre.z + 1.0) * 0.5).clamp(0.0, 1.0);
        let jitter = 0.94 + 0.12 * lumps(centre * 3.0, seed);
        let shade = (0.62 + 0.38 * height) * jitter;
        for p in [a, b, c] {
            self.positions.push(p.to_array());
            self.normals.push(n.to_array());
            self.colours.push([shade, shade, shade, 1.0]);
        }
    }

    /// A lumpy lobe: an icosphere of `radius` at `centre`, displaced by
    /// noise, every vertex held inside the unit sphere (the model's crown).
    fn lobe(&mut self, centre: Vec3, radius: f32, squash: f32, seed: f32) {
        let ico = Sphere::new(1.0).mesh().ico(1).expect("a small icosphere");
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(points)) =
            ico.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            return;
        };
        let place = |p: &[f32; 3]| {
            let dir = Vec3::from_array(*p);
            let r = radius * (1.0 + 0.16 * lumps(dir * 2.0, seed));
            let q = centre + Vec3::new(dir.x * r, dir.y * r, dir.z * r * squash);
            if q.length() > 1.0 {
                q.normalize()
            } else {
                q
            }
        };
        let placed: Vec<Vec3> = points.iter().map(place).collect();
        let indices: Vec<usize> = match ico.indices() {
            Some(Indices::U32(i)) => i.iter().map(|&i| i as usize).collect(),
            Some(Indices::U16(i)) => i.iter().map(|&i| i as usize).collect(),
            None => (0..placed.len()).collect(),
        };
        for tri in indices.chunks(3) {
            self.add(
                [placed[tri[0]], placed[tri[1]], placed[tri[2]]],
                centre,
                seed,
            );
        }
    }

    fn mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colours)
    }
}

/// A species' unit crown, variant `variant`: drawn inside the unit sphere
/// (the crown's ellipsoid once scaled per tree) and filling most of it, so
/// what is drawn is never more than the model's crown.
///
/// - Broadleaf: a cluster of lumpy lobes round a large central one.
/// - Conifer: stacked, jagged cone tiers narrowing to a point.
/// - Palm: drooping fronds round a small heart, held high on its trunk.
/// - Shrub: a few low, wide mounds.
///
/// Every face is flat-shaded, lighter towards the top and jittered, so the
/// foliage carries texture without an image.
pub fn crown_mesh(species: Species, variant: u32) -> Mesh {
    let seed = 1.3 + variant as f32 * 2.17;
    let mut f = Facets::default();
    let angle = |k: usize, n: usize| seed + k as f32 / n as f32 * std::f32::consts::TAU;
    match species {
        Species::Broadleaf => {
            f.lobe(Vec3::new(0.0, 0.0, 0.0), 0.68, 1.0, seed);
            f.lobe(Vec3::new(0.0, 0.0, 0.5), 0.48, 1.0, seed + 0.4);
            for k in 0..6 {
                let a = angle(k, 6);
                let z = if k % 2 == 0 { 0.25 } else { -0.3 };
                f.lobe(
                    Vec3::new(0.5 * a.cos(), 0.5 * a.sin(), z),
                    0.47,
                    0.95,
                    seed + k as f32,
                );
            }
        }
        Species::Conifer => {
            // Five tiers of drooping, serrated skirts, each overhanging the
            // one above, narrowing to the leader at the top.
            let tiers = 5;
            let sectors = 14;
            for t in 0..tiers {
                // The lowest tier where the crown's ellipsoid is still wide.
                let base = -0.55 + t as f32 * 0.33;
                let apex = if t + 1 == tiers {
                    1.0
                } else {
                    (base + 0.62).min(1.0)
                };
                let radius = 1.0 - t as f32 * 0.19;
                let top = Vec3::new(0.0, 0.0, apex);
                let under = Vec3::new(0.0, 0.0, base + 0.2);
                let ring: Vec<Vec3> = (0..sectors)
                    .map(|k| {
                        let a = angle(k, sectors) + 0.7 * t as f32;
                        // Serrated: alternate points reach further and hang lower.
                        let reach = if k % 2 == 0 { 1.0 } else { 0.72 };
                        let r = radius * reach * (0.9 + 0.1 * lumps(Vec3::splat(a), seed));
                        let z = base - if k % 2 == 0 { 0.1 } else { 0.0 };
                        let q = Vec3::new(r * a.cos(), r * a.sin(), z);
                        if q.length() > 1.0 {
                            q.normalize()
                        } else {
                            q
                        }
                    })
                    .collect();
                for k in 0..sectors {
                    let (p, q) = (ring[k], ring[(k + 1) % sectors]);
                    f.add([p, q, top], Vec3::new(0.0, 0.0, base), seed + k as f32);
                    f.add([p, q, under], Vec3::new(0.0, 0.0, base + 0.6), seed);
                }
            }
        }
        Species::Palm => {
            f.lobe(Vec3::new(0.0, 0.0, 0.3), 0.22, 1.0, seed);
            // Fronds that rise from the heart, arch and hang, with leaflets
            // along both edges; alternate fronds lie higher.
            let fronds = 11;
            let steps = 8;
            for k in 0..fronds {
                let a = angle(k, fronds);
                let dir = Vec3::new(a.cos(), a.sin(), 0.0);
                let side = Vec3::new(-a.sin(), a.cos(), 0.0);
                let lift = if k % 2 == 0 { 0.2 } else { 0.0 };
                let along = |t: f32| {
                    let q = dir * (0.12 + 0.88 * t)
                        + Vec3::new(0.0, 0.0, 0.3 + lift + 0.9 * t - 2.1 * t * t);
                    if q.length() > 1.0 {
                        q.normalize()
                    } else {
                        q
                    }
                };
                for step in 0..steps {
                    let (t0, t1) = (step as f32 / steps as f32, (step + 1) as f32 / steps as f32);
                    let (p0, p1) = (along(t0), along(t1));
                    // Leaflets: each segment a pair of pointed blades, hanging.
                    let w = 0.22 * (std::f32::consts::PI * (0.15 + 0.85 * t0)).sin();
                    let hang = Vec3::new(0.0, 0.0, -0.08 - 0.1 * t0);
                    for sign in [1.0, -1.0] {
                        let tip = (p0 + p1) * 0.5 + side * (sign * w) + hang;
                        f.add([p0, p1, tip], p0 - Vec3::Z, seed + k as f32);
                    }
                }
            }
        }
        Species::Shrub => {
            f.lobe(Vec3::new(0.0, 0.0, -0.15), 0.72, 0.8, seed);
            for k in 0..4 {
                let a = angle(k, 4);
                f.lobe(
                    Vec3::new(0.45 * a.cos(), 0.45 * a.sin(), -0.25),
                    0.52,
                    0.75,
                    seed + k as f32,
                );
            }
        }
    }
    f.mesh()
}

/// The unit trunk: radius 1 at the ground narrowing to 0.55 at the top,
/// from z = 0 to z = 1 (the viewer is Z-up), seven-sided, scaled per tree.
pub fn trunk_mesh() -> Mesh {
    Mesh::from(
        ConicalFrustum {
            radius_top: 0.55,
            radius_bottom: 1.0,
            height: 1.0,
        }
        .mesh()
        .resolution(7),
    )
    .transformed_by(Transform {
        translation: Vec3::new(0.0, 0.0, 0.5),
        rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        scale: Vec3::ONE,
    })
}

/// Where a species' crown starts, as a share of the tree's height, on its
/// impostor (a typical tree of the kind).
fn impostor_crown_base(species: Species) -> f32 {
    match species {
        Species::Conifer => 0.12,
        Species::Broadleaf => 0.35,
        Species::Palm => 0.78,
        Species::Shrub => 0.0,
    }
}

/// A species' unit impostor: its silhouette (trunk and crown outline) on
/// two crossed planes, x and y in `[-1, 1]`, z in `[0, 1]`, scaled per tree
/// to its crown's width and its height; the trunk part darker.
pub fn impostor_mesh(species: Species) -> Mesh {
    let base = impostor_crown_base(species);
    let n = 16;
    // The crown's outline in (u, z), u across in [-1, 1].
    let outline: Vec<[f32; 2]> = (0..n)
        .map(|k| {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            let (u, v) = (a.cos(), a.sin());
            let half = 0.5 * (1.0 - base);
            let mid = base + half;
            match species {
                Species::Conifer => {
                    // A pointed outline: wide at the bottom, sharp at the top.
                    let w = (1.0 - (v * 0.5 + 0.5)).powf(0.9);
                    [u * w, mid + v * half]
                }
                Species::Palm => {
                    let w = 1.0 - 0.35 * ((k % 2) as f32);
                    [u * w, mid + v * half * 0.6]
                }
                _ => [u * (0.92 + 0.08 * (3.0 * a).cos()), mid + v * half],
            }
        })
        .collect();
    let trunk_top = base + 0.5 * (1.0 - base) * if species == Species::Palm { 0.4 } else { 0.0 };
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colours: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for (axis, normal) in [(Vec3::X, [0.0, -1.0, 0.0]), (Vec3::Y, [1.0, 0.0, 0.0])] {
        let at = |u: f32, z: f32| (axis * u + Vec3::Z * z).to_array();
        let mut push = |p: [f32; 3], shade: f32| {
            positions.push(p);
            normals.push(normal);
            colours.push([shade, shade, shade, 1.0]);
            (positions.len() - 1) as u32
        };
        // The trunk, a narrow dark strip.
        if trunk_top > 0.0 {
            let q = [
                at(-0.07, 0.0),
                at(0.07, 0.0),
                at(0.05, trunk_top),
                at(-0.05, trunk_top),
            ];
            let i: Vec<u32> = q.iter().map(|p| push(*p, 0.35)).collect();
            indices.extend_from_slice(&[i[0], i[1], i[2], i[0], i[2], i[3]]);
        }
        // The crown, a fan round its middle.
        let centre = push(at(0.0, base + 0.5 * (1.0 - base)), 0.9);
        let ring: Vec<u32> = outline
            .iter()
            .map(|[u, z]| push(at(*u, z.clamp(0.0, 1.0)), 0.62 + 0.38 * z))
            .collect();
        for k in 0..ring.len() {
            indices.extend_from_slice(&[centre, ring[k], ring[(k + 1) % ring.len()]]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colours)
    .with_inserted_indices(Indices::U32(indices))
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
    // Each tree turned its own way about its axis: the crown's extent from
    // above is a circle, so the turn changes how it looks, never its size.
    let turn = Quat::from_rotation_z(tree_look(&tree.id).2);
    TreeTransforms {
        trunk: Transform {
            translation: Vec3::new(x, y, base),
            rotation: turn,
            scale: Vec3::new(trunk, trunk, (tree.trunk_top_z_m() - tree.base_z_m) as f32),
        },
        crown: Transform {
            translation: Vec3::from_array(centre),
            rotation: turn,
            scale: Vec3::new(radius, radius, tree.crown_half_height_m() as f32),
        },
        impostor: Transform {
            translation: Vec3::new(x, y, base),
            rotation: turn,
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
    crown_materials: HashMap<(Species, usize, usize), Handle<StandardMaterial>>,
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
    let mut crown_meshes: HashMap<(Species, u32), Handle<Mesh>> = HashMap::new();
    let mut impostor_meshes: HashMap<Species, Handle<Mesh>> = HashMap::new();
    let near = VisibilityRange::abrupt(0.0, IMPOSTOR_DISTANCE_M);
    let far = VisibilityRange::abrupt(IMPOSTOR_DISTANCE_M, f32::MAX);
    for tree in &read.trees {
        let (variant, tint, _) = tree_look(&tree.id);
        let crown = crown_meshes
            .entry((tree.species, variant))
            .or_insert_with(|| {
                let mesh = meshes.add(crown_mesh(tree.species, variant));
                drawn.meshes.push(mesh.clone());
                mesh
            })
            .clone();
        let impostor = impostor_meshes
            .entry(tree.species)
            .or_insert_with(|| {
                let mesh = meshes.add(impostor_mesh(tree.species));
                drawn.meshes.push(mesh.clone());
                mesh
            })
            .clone();
        let level = leaf_level(tree.leaf_density);
        let material = drawn
            .crown_materials
            .entry((tree.species, level, tint))
            .or_insert_with(|| {
                let [r, g, b] = crown_color(tree.species, level).map(|c| c * TINT_VALUES[tint]);
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

    /// Each form, placed by its transform and turn, stands inside the
    /// model's volume and fills most of it: the crown in its ellipsoid, the
    /// trunk from the ground to the crown's middle, the impostor from the
    /// ground to the top across the crown.
    #[test]
    fn each_form_stands_where_the_model_puts_the_tree() {
        let tree = tree();
        let at = tree_transforms(&tree);
        let placed = |mesh: Mesh, transform: Transform| positions(&mesh.transformed_by(transform));
        let axis = |p: &[f32; 3]| (p[0] - 10.0).hypot(p[1] + 20.0);
        let (variant, _, _) = tree_look(&tree.id);
        // The crown: inside the ellipsoid round (10, -20, 114), radius 5,
        // half-height 6, reaching most of the way to its edge each way.
        let crown = placed(crown_mesh(tree.species, variant), at.crown);
        for p in &crown {
            let e = (axis(p) / 5.0).powi(2) + ((p[2] - 114.0) / 6.0).powi(2);
            assert!(e <= 1.0 + 1e-3, "{p:?} outside the crown ({e})");
        }
        let (lo, hi) = extent(&crown);
        assert!(
            crown.iter().map(axis).fold(0.0, f32::max) > 4.0,
            "fills its width"
        );
        assert!(
            lo[2] < 108.0 + 1.5 && hi[2] > 120.0 - 1.5,
            "fills its height {lo:?} {hi:?}"
        );
        // The trunk: radius 0.3 at the ground, z 100..114.
        let trunk = placed(trunk_mesh(), at.trunk);
        let (lo, hi) = extent(&trunk);
        assert!((lo[2] - 100.0).abs() < 1e-3 && (hi[2] - 114.0).abs() < 1e-3);
        assert!(trunk.iter().all(|p| axis(p) <= 0.3 + 1e-4));
        // The impostor: within the crown's width, ground to top.
        let impostor = placed(impostor_mesh(tree.species), at.impostor);
        let (lo, hi) = extent(&impostor);
        assert!(impostor.iter().all(|p| axis(p) <= 5.0 + 1e-3));
        assert!(
            (lo[2] - 100.0).abs() < 1e-3 && (hi[2] - 120.0).abs() < 0.5,
            "{lo:?} {hi:?}"
        );
    }

    /// Every species' crown, in every variant, stays inside the unit sphere
    /// (the model's crown once scaled) and fills most of it; species differ
    /// in shape, variants within a species differ, and the foliage carries
    /// shading in its vertex colours.
    #[test]
    fn the_unit_meshes_stay_in_the_crown_and_differ() {
        for species in Species::ALL {
            let mut seen = Vec::new();
            for variant in 0..CROWN_VARIANTS {
                let mesh = crown_mesh(species, variant);
                let points = positions(&mesh);
                assert!(
                    points
                        .iter()
                        .all(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 1.0 + 1e-4),
                    "{species:?} {variant}: outside the crown"
                );
                let reach = points.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f32::max);
                let (lo, hi) = extent(&points);
                assert!(reach > 0.8, "{species:?} {variant}: reaches {reach}");
                assert!(hi[2] - lo[2] > 1.0, "{species:?} {variant}: {lo:?} {hi:?}");
                assert!(
                    mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some(),
                    "{species:?}: shaded"
                );
                seen.push(points);
            }
            for a in 0..seen.len() {
                for b in a + 1..seen.len() {
                    assert_ne!(seen[a], seen[b], "{species:?}: variants {a} and {b} alike");
                }
            }
        }
        let count = |s| positions(&crown_mesh(s, 0)).len();
        assert_ne!(count(Species::Broadleaf), count(Species::Conifer));
        assert_ne!(count(Species::Palm), count(Species::Shrub));
        // The trunk: within radius 1, from the ground to the top, tapering.
        let trunk = positions(&trunk_mesh());
        let (lo, hi) = extent(&trunk);
        assert!(
            lo[2].abs() < 1e-6 && (hi[2] - 1.0).abs() < 1e-6,
            "{lo:?} {hi:?}"
        );
        assert!(trunk.iter().all(|p| p[0].hypot(p[1]) <= 1.0 + 1e-4));
        assert!(trunk
            .iter()
            .filter(|p| p[2] > 0.99)
            .all(|p| p[0].hypot(p[1]) <= 0.56));
        for species in Species::ALL {
            let (lo, hi) = extent(&positions(&impostor_mesh(species)));
            assert!(
                lo[2].abs() < 1e-6 && (hi[2] - 1.0).abs() < 0.05,
                "{species:?}"
            );
            assert!(lo[0] >= -1.0 - 1e-6 && hi[0] <= 1.0 + 1e-6, "{species:?}");
        }
    }

    /// A tree's look comes from its id: the same every run, spread across
    /// the variants, tints and turns.
    #[test]
    fn each_tree_looks_its_own_way_and_always_the_same() {
        assert_eq!(tree_look("oak-17"), tree_look("oak-17"));
        let looks: Vec<_> = (0..200).map(|i| tree_look(&format!("tree-{i}"))).collect();
        for v in 0..CROWN_VARIANTS {
            assert!(looks.iter().any(|l| l.0 == v), "variant {v} unused");
        }
        for t in 0..CROWN_TINTS {
            assert!(looks.iter().any(|l| l.1 == t), "tint {t} unused");
        }
        let turns: std::collections::BTreeSet<u32> =
            looks.iter().map(|l| (l.2 * 10.0) as u32).collect();
        assert!(turns.len() > 30, "turns spread out");
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

    /// Up close a tree is its trunk and crown meshes, past
    /// [`IMPOSTOR_DISTANCE_M`] its impostor, and never both: each part's
    /// range is the renderer's to apply, so it is checked here.
    #[test]
    fn meshes_draw_near_and_impostors_far() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Trees>()
            .insert_resource(RuntimeOverlayVisibility::default())
            .insert_resource(ReplayState::new(Some(
                serde_json::from_value(serde_json::json!({
                    "meta": {"trees": [serde_json::to_value(tree()).unwrap()]},
                    "frames": [],
                }))
                .unwrap(),
            )))
            .add_systems(Update, sync_trees);
        app.update();
        let mut query = app.world_mut().query::<(&TreePart, &VisibilityRange)>();
        let parts: Vec<(TreePart, VisibilityRange)> = query
            .iter(app.world())
            .map(|(part, range)| (*part, range.clone()))
            .collect();
        assert_eq!(parts.len(), 3, "a trunk, a crown and an impostor");
        for (part, range) in parts {
            match part {
                TreePart::Trunk | TreePart::Crown => {
                    assert_eq!(range.start_margin.end, 0.0, "{part:?}");
                    assert_eq!(range.end_margin.start, IMPOSTOR_DISTANCE_M, "{part:?}");
                }
                TreePart::Impostor => {
                    assert_eq!(range.start_margin.start, IMPOSTOR_DISTANCE_M);
                    assert_eq!(range.end_margin.start, f32::MAX);
                }
            }
        }
    }
}
