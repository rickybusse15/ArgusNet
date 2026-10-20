//! Power lines (R3): the run's supports and conductors, from the replay's
//! `meta.power_lines`, the same lines the safety stage judged flights
//! against.
//!
//! # Wires that never vanish
//!
//! A conductor is a centimetre or two thick, so drawn at its true size it is
//! under a pixel beyond a few hundred metres: the wire disappears at exactly
//! the range a pilot needs to see it. Each line's conductors are one mesh of
//! quads, one per straight piece of [`PowerLine::pieces`] (at most 5 m, so
//! the sag reads as a curve), and [`WireMaterial`]'s vertex shader expands
//! every quad in screen space to at least [`WIRE_MIN_WIDTH_PX`] pixels, or to
//! the conductor's true thickness when the camera is close enough for that to
//! be wider. The mesh is built once per run: nothing is recomputed as the
//! camera moves. Wires are opaque, depth-tested, cast no shadow, and are
//! never frustum-culled, at any quality tier.
//!
//! # Supports
//!
//! Each support is a lattice tower: four legs tapering from its base to its
//! top, braced at three levels, with a cross-arm across the line at the top
//! carrying the conductors (`PowerLine::cross_arm`: square to the span at an
//! end, bisecting the turn between two spans). The tower stands at the
//! support's own ground height, the one the simulation used; it is lit and
//! casts shadows like a building. Towers of one height share one mesh, so
//! the renderer batches them.
//!
//! Both are world structures, not overlays: no layer preset owns them, and
//! the Runtime Overlays panel's Power Lines toggle hides and shows them.
//!
//! # Clearance corridors
//!
//! An optional overlay, off by default (the Clearance Corridors toggle,
//! beside Power Lines; no preset owns it either), shows each span's
//! clearance volume: everywhere within the line's clearance of a conductor's
//! surface, which the safety engine's `structure_clearance` keeps flights
//! out of and efficient_v1's routes keep out of. Across the line a span's
//! conductors lie on one level segment, the cross-arm's, so the volume's
//! cross-section is that segment widened by the conductor radius plus the
//! clearance: a stadium. [`clearance_mesh`] sweeps it along the span at
//! every piece joint, square to the span, with the segment where the
//! conductors hang there; it is closed at the line's two ends. It is drawn
//! translucent and unlit, both faces, casting no shadow, built once per run
//! with the wires.
//! Picking ignores them ([`PresentationOnly`]). Each line in
//! `meta.power_lines` is read on its own: one the model refuses is not
//! drawn and its reason is logged once per run, and every other line is
//! still drawn.

use std::collections::HashMap;

use argusnet_world::powerline::PowerLine;
use bevy::asset::{embedded_asset, RenderAssetUsages};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{
    Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat,
};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::replay::ReplayState;
use crate::sensor_volumes::PresentationOnly;
use crate::state::RuntimeOverlayVisibility;

/// The least width a conductor is drawn at, screen pixels.
pub const WIRE_MIN_WIDTH_PX: f32 = 1.6;
/// Conductors: a pale steel, readable over the dark ground and the sky.
pub const WIRE_COLOR: [f32; 4] = [0.78, 0.80, 0.80, 1.0];
/// Towers: galvanised steel.
pub const TOWER_COLOR: [f32; 3] = [0.56, 0.58, 0.58];
/// Clearance corridors: a translucent amber, the caution colour.
pub const CLEARANCE_COLOR: [f32; 4] = [1.0, 0.55, 0.05, 0.35];
/// Segments in each half-circle end of a corridor's cross-section.
pub const CLEARANCE_HALF_SEGMENTS: usize = 8;

/// A wire vertex's far end: the other end of its piece.
pub const ATTRIBUTE_WIRE_OTHER: MeshVertexAttribute =
    MeshVertexAttribute::new("WireOther", 0x0a26_5e7a_0001, VertexFormat::Float32x3);
/// A wire vertex's side (+1 left, -1 right), end (-1 start, +1 end) and the
/// conductor's radius, metres.
pub const ATTRIBUTE_WIRE_SHAPE: MeshVertexAttribute =
    MeshVertexAttribute::new("WireShape", 0x0a26_5e7a_0002, VertexFormat::Float32x3);

/// Conductors drawn as screen-space lines (see the module docs).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct WireMaterial {
    #[uniform(0)]
    pub color: Vec4,
    /// `x`: the least width a conductor is drawn at, screen pixels.
    #[uniform(1)]
    pub width: Vec4,
}

impl Default for WireMaterial {
    fn default() -> Self {
        Self {
            color: Vec4::from_array(WIRE_COLOR),
            width: Vec4::new(WIRE_MIN_WIDTH_PX, 0.0, 0.0, 0.0),
        }
    }
}

impl Material for WireMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://argusnet_viewer/shaders/power_line.wesl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://argusnet_viewer/shaders/power_line.wesl".into()
    }
    fn enable_shadows() -> bool {
        false
    }
    fn enable_prepass() -> bool {
        false
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_WIRE_OTHER.at_shader_location(1),
            ATTRIBUTE_WIRE_SHAPE.at_shader_location(2),
        ])?];
        // A quad faces whichever way its expansion turned it.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub struct PowerLinePlugin;
impl Plugin for PowerLinePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/power_line.wesl");
        app.add_plugins(MaterialPlugin::<WireMaterial>::default())
            .init_resource::<PowerLines>();
    }
}

/// Every conductor of `lines` as one mesh of screen-space quads: four
/// vertices per straight piece, at its two ends, each carrying the other
/// end. `None` for no lines.
pub fn wire_mesh(lines: &[PowerLine]) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut others: Vec<[f32; 3]> = Vec::new();
    let mut shapes: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for line in lines {
        let radius = line.conductor_radius_m as f32;
        for piece in line.pieces() {
            let start = piece.start_m.map(|v| v as f32);
            let end = piece.end_m.map(|v| v as f32);
            let base = positions.len() as u32;
            for (at, other, end_sign) in [(start, end, -1.0), (end, start, 1.0)] {
                for side in [1.0, -1.0] {
                    positions.push(at);
                    others.push(other);
                    shapes.push([side, end_sign, radius]);
                }
            }
            // Start left, start right, end left, end right.
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
        }
    }
    if indices.is_empty() {
        return None;
    }
    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(ATTRIBUTE_WIRE_OTHER, others)
        .with_inserted_attribute(ATTRIBUTE_WIRE_SHAPE, shapes)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

/// The level segment `line`'s conductors hang on at fraction `t` of span
/// `span`, across the line: from the lowest offset's conductor to the
/// highest's.
pub fn conductor_segment(line: &PowerLine, span: usize, t: f64) -> ([f64; 3], [f64; 3]) {
    let (low, high) = line
        .conductor_offsets_m
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &o| {
            (lo.min(o), hi.max(o))
        });
    (
        line.conductor_at(span, low, t),
        line.conductor_at(span, high, t),
    )
}

/// The cross-section of `line`'s clearance volume at fraction `t` of span
/// `span`: a closed loop of `2 * (CLEARANCE_HALF_SEGMENTS + 1)` points, each
/// the conductor radius plus the clearance from [`conductor_segment`] there,
/// in the plane square to the span (a half-circle round each end of the
/// segment, joined by its two sides), and the outward normal at each.
pub fn clearance_ring(line: &PowerLine, span: usize, t: f64) -> Vec<(Vec3, Vec3)> {
    let reach = (line.conductor_radius_m + line.clearance_m) as f32;
    let (low, high) = conductor_segment(line, span, t);
    let (low, high) = (
        Vec3::from_array(low.map(|v| v as f32)),
        Vec3::from_array(high.map(|v| v as f32)),
    );
    // Across the line, level: the cross-arm's direction at t.
    let mut across = Vec3::from_array(line.conductor_at(span, 1.0, t).map(|v| v as f32))
        - Vec3::from_array(line.conductor_at(span, 0.0, t).map(|v| v as f32));
    across.z = 0.0;
    let across = across.normalize_or(Vec3::Y);
    // Along the span, and square to both: the section's "up".
    let (a, b) = (
        Vec3::from_array(
            line.conductor_at(span, 0.0, (t - 1e-3).max(0.0))
                .map(|v| v as f32),
        ),
        Vec3::from_array(
            line.conductor_at(span, 0.0, (t + 1e-3).min(1.0))
                .map(|v| v as f32),
        ),
    );
    let along = (b - a).normalize_or(Vec3::X);
    let mut up = along.cross(across).normalize_or(Vec3::Z);
    if up.z < 0.0 {
        up = -up;
    }
    let half = CLEARANCE_HALF_SEGMENTS;
    let mut ring = Vec::with_capacity(2 * (half + 1));
    for (centre, start) in [
        (high, -std::f32::consts::FRAC_PI_2),
        (low, std::f32::consts::FRAC_PI_2),
    ] {
        for j in 0..=half {
            let angle = start + std::f32::consts::PI * j as f32 / half as f32;
            let normal = across * angle.cos() + up * angle.sin();
            ring.push((centre + normal * reach, normal));
        }
    }
    ring
}

/// Every span of `lines` as its clearance volume (see the module docs): a
/// tube of [`clearance_ring`]s at each piece joint, closed at each line's
/// ends. `None` for no lines.
pub fn clearance_mesh(lines: &[PowerLine]) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let ring_len = 2 * (CLEARANCE_HALF_SEGMENTS + 1);
    for line in lines {
        let last = line.supports.len() - 2;
        for span in 0..=last {
            let (from, to) = (&line.supports[span], &line.supports[span + 1]);
            let span_m = (to.xy_m[0] - from.xy_m[0]).hypot(to.xy_m[1] - from.xy_m[1]);
            let count = (span_m / argusnet_world::powerline::MAX_PIECE_M)
                .ceil()
                .max(1.0) as usize;
            let first_ring = positions.len() as u32;
            for k in 0..=count {
                for (point, normal) in clearance_ring(line, span, k as f64 / count as f64) {
                    positions.push(point.to_array());
                    normals.push(normal.to_array());
                }
            }
            for k in 0..count as u32 {
                let (a, b) = (
                    first_ring + k * ring_len as u32,
                    first_ring + (k + 1) * ring_len as u32,
                );
                for j in 0..ring_len as u32 {
                    let next = (j + 1) % ring_len as u32;
                    indices.extend_from_slice(&[a + j, b + j, a + next, a + next, b + j, b + next]);
                }
            }
            // Closed at the line's two ends: a fan from the section's
            // centre.
            let ends = [(span == 0).then_some(0.0), (span == last).then_some(1.0)];
            for t in ends.into_iter().flatten() {
                let ring = clearance_ring(line, span, t);
                let (low, high) = conductor_segment(line, span, t);
                let centre = Vec3::from_array(low.map(|v| v as f32))
                    .lerp(Vec3::from_array(high.map(|v| v as f32)), 0.5);
                // Out of the line's end, along its span.
                let along = Vec3::from_array(line.conductor_at(span, 0.0, 1.0).map(|v| v as f32))
                    - Vec3::from_array(line.conductor_at(span, 0.0, 0.0).map(|v| v as f32));
                let along = along.normalize_or(Vec3::X);
                let normal = if t == 0.0 { -along } else { along };
                let base = positions.len() as u32;
                positions.push(centre.to_array());
                normals.push(normal.to_array());
                for (point, _) in &ring {
                    positions.push(point.to_array());
                    normals.push(normal.to_array());
                }
                for j in 0..ring_len as u32 {
                    let next = (j + 1) % ring_len as u32;
                    indices.extend_from_slice(&[base, base + 1 + j, base + 1 + next]);
                }
            }
        }
    }
    if indices.is_empty() {
        return None;
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

/// A straight beam `thickness_m` square from `from` to `to`.
fn beam(from: Vec3, to: Vec3, thickness_m: f32) -> Mesh {
    let along = to - from;
    let length = along.length().max(1e-3);
    Mesh::from(Cuboid::new(thickness_m, thickness_m, length)).transformed_by(Transform {
        translation: (from + to) * 0.5,
        rotation: Quat::from_rotation_arc(Vec3::Z, along / length),
        scale: Vec3::ONE,
    })
}

/// How far each side of the tower's centre its cross-arm reaches, for
/// conductors hanging `offsets_m` along it: a metre past the outermost.
pub fn arm_reach_m(offsets_m: &[f64]) -> f32 {
    offsets_m.iter().fold(0.0_f64, |m, o| m.max(o.abs())) as f32 + 1.0
}

/// A lattice tower `height_m` tall in its own frame: standing on the origin,
/// up +z, its cross-arm along x reaching `arm_reach_m` each side at the top,
/// where the conductors are attached.
pub fn tower_mesh(height_m: f32, arm_reach_m: f32) -> Mesh {
    let h = height_m.max(1.0);
    let base = (0.12 * h).clamp(1.0, 5.0);
    let top = (0.3 * base).max(0.5);
    let half = |z: f32| base + (top - base) * (z / h);
    let corners = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
    let at = |(sx, sy): (f32, f32), z: f32| Vec3::new(sx * half(z), sy * half(z), z);
    let leg = (0.02 * h).clamp(0.15, 0.4);
    let mut mesh = beam(at(corners[0], 0.0), at(corners[0], h), leg);
    for &corner in &corners[1..] {
        mesh.merge(&beam(at(corner, 0.0), at(corner, h), leg))
            .expect("cuboids share vertex attributes");
    }
    for level in [0.25, 0.5, 0.75, 1.0] {
        let z = h * level;
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            mesh.merge(&beam(at(a, z), at(b, z), 0.6 * leg))
                .expect("cuboids share vertex attributes");
        }
    }
    // The cross-arm, its top flush with the attachment height.
    let arm = 1.5 * leg;
    let z = h - 0.5 * arm;
    mesh.merge(&beam(
        Vec3::new(-arm_reach_m, 0.0, z),
        Vec3::new(arm_reach_m, 0.0, z),
        arm,
    ))
    .expect("cuboids share vertex attributes");
    mesh
}

/// Where each of `line`'s supports stands: on its ground at its position,
/// turned so its cross-arm lies along [`PowerLine::cross_arm`].
pub fn tower_transforms(line: &PowerLine) -> Vec<Transform> {
    line.supports
        .iter()
        .enumerate()
        .map(|(index, support)| {
            let arm = line.cross_arm(index);
            Transform {
                translation: Vec3::new(
                    support.xy_m[0] as f32,
                    support.xy_m[1] as f32,
                    support.base_z_m as f32,
                ),
                rotation: Quat::from_rotation_z((arm[1] as f32).atan2(arm[0] as f32)),
                scale: Vec3::ONE,
            }
        })
        .collect()
}

/// A drawn part of a power line.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerLinePart {
    Wires,
    Tower,
    /// The clearance-corridor overlay.
    Clearance,
}

/// The power lines drawn for the run in the replay.
#[derive(Resource, Default)]
pub struct PowerLines {
    /// The run (epoch, and whether it had metadata) they were built for.
    built_for: Option<(u64, bool)>,
    entities: Vec<Entity>,
    meshes: Vec<Handle<Mesh>>,
    wire_material: Option<Handle<WireMaterial>>,
    tower_material: Option<Handle<StandardMaterial>>,
    clearance_material: Option<Handle<StandardMaterial>>,
    lines: usize,
    towers: usize,
    pieces: usize,
    corridors: usize,
    builds: u64,
    refused: Vec<String>,
}

impl PowerLines {
    /// Lines drawn.
    pub fn line_count(&self) -> usize {
        self.lines
    }
    /// Towers drawn.
    pub fn tower_count(&self) -> usize {
        self.towers
    }
    /// Straight wire pieces drawn, over every conductor.
    pub fn piece_count(&self) -> usize {
        self.pieces
    }
    /// Spans whose clearance corridor is built (drawn when the overlay is
    /// on).
    pub fn corridor_count(&self) -> usize {
        self.corridors
    }
    /// How many times the geometry was built.
    pub fn builds(&self) -> u64 {
        self.builds
    }
    /// Why each line the model refused was not drawn, one per line.
    pub fn refused(&self) -> &[String] {
        &self.refused
    }
}

/// Build the run's power lines once per run (a new epoch, or metadata
/// arriving), and show or hide them with the Power Lines toggle, and their
/// clearance corridors with the Clearance Corridors toggle.
#[allow(clippy::too_many_arguments)]
pub fn sync_power_lines(
    mut commands: Commands,
    replay: Res<ReplayState>,
    visibility: Res<RuntimeOverlayVisibility>,
    mut drawn: ResMut<PowerLines>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut wire_materials: ResMut<Assets<WireMaterial>>,
    mut tower_materials: ResMut<Assets<StandardMaterial>>,
    mut shown: Query<(&PowerLinePart, &mut Visibility)>,
) {
    let shown_if = |on: bool| {
        if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    };
    let visible = shown_if(visibility.power_lines);
    let corridors_visible = shown_if(visibility.power_line_clearance);
    for (part, mut v) in &mut shown {
        v.set_if_neq(match part {
            PowerLinePart::Clearance => corridors_visible,
            _ => visible,
        });
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
    (drawn.lines, drawn.towers, drawn.pieces, drawn.corridors) = (0, 0, 0, 0);
    // Each line is read on its own: a refused one is reported and the rest
    // are still drawn.
    let read = replay
        .document
        .as_ref()
        .map(|d| d.power_lines())
        .unwrap_or_default();
    for reason in &read.refused {
        warn!("power line not drawn: {reason}");
    }
    drawn.refused = read.refused;
    let lines = read.lines;
    let Some(wires) = wire_mesh(&lines) else {
        return;
    };
    let wire_material = drawn
        .wire_material
        .get_or_insert_with(|| wire_materials.add(WireMaterial::default()))
        .clone();
    let tower_material = drawn
        .tower_material
        .get_or_insert_with(|| {
            tower_materials.add(StandardMaterial {
                base_color: Color::srgb(TOWER_COLOR[0], TOWER_COLOR[1], TOWER_COLOR[2]),
                perceptual_roughness: 0.65,
                metallic: 0.5,
                ..default()
            })
        })
        .clone();
    drawn.lines = lines.len();
    drawn.pieces = lines.iter().map(|line| line.pieces().len()).sum();
    let wire_handle = meshes.add(wires);
    drawn.meshes.push(wire_handle.clone());
    drawn.entities.push(
        commands
            .spawn((
                PowerLinePart::Wires,
                PresentationOnly,
                NoFrustumCulling,
                Mesh3d(wire_handle),
                MeshMaterial3d(wire_material),
                // The shader projects the mesh with the view alone: it is
                // in world coordinates and must stay untransformed.
                Transform::IDENTITY,
                visible,
            ))
            .id(),
    );
    if let Some(corridors) = clearance_mesh(&lines) {
        let material = drawn
            .clearance_material
            .get_or_insert_with(|| {
                tower_materials.add(StandardMaterial {
                    base_color: Color::srgba(
                        CLEARANCE_COLOR[0],
                        CLEARANCE_COLOR[1],
                        CLEARANCE_COLOR[2],
                        CLEARANCE_COLOR[3],
                    ),
                    unlit: true,
                    alpha_mode: AlphaMode::Blend,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                })
            })
            .clone();
        let handle = meshes.add(corridors);
        drawn.meshes.push(handle.clone());
        drawn.corridors = lines.iter().map(|line| line.supports.len() - 1).sum();
        drawn.entities.push(
            commands
                .spawn((
                    PowerLinePart::Clearance,
                    PresentationOnly,
                    bevy::light::NotShadowCaster,
                    Mesh3d(handle),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                    corridors_visible,
                ))
                .id(),
        );
    }
    // One mesh per tower height and arm, shared by every tower that has it.
    let mut shared: HashMap<(u32, u32), Handle<Mesh>> = HashMap::new();
    for line in &lines {
        let reach = arm_reach_m(&line.conductor_offsets_m);
        for (support, transform) in line.supports.iter().zip(tower_transforms(line)) {
            let height = support.height_m as f32;
            let mesh = shared
                .entry((height.to_bits(), reach.to_bits()))
                .or_insert_with(|| {
                    let handle = meshes.add(tower_mesh(height, reach));
                    drawn.meshes.push(handle.clone());
                    handle
                })
                .clone();
            drawn.entities.push(
                commands
                    .spawn((
                        PowerLinePart::Tower,
                        PresentationOnly,
                        Mesh3d(mesh),
                        MeshMaterial3d(tower_material.clone()),
                        transform,
                        visible,
                    ))
                    .id(),
            );
            drawn.towers += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_world::powerline::Support;
    use bevy::mesh::VertexAttributeValues;

    /// Two 30 m supports 200 m apart along x, then a turn north: three
    /// conductors 4 m apart.
    fn line() -> PowerLine {
        PowerLine {
            id: "line-a".into(),
            supports: vec![
                Support {
                    xy_m: [0.0, 0.0],
                    base_z_m: 10.0,
                    height_m: 30.0,
                },
                Support {
                    xy_m: [200.0, 0.0],
                    base_z_m: 12.0,
                    height_m: 30.0,
                },
                Support {
                    xy_m: [200.0, 150.0],
                    base_z_m: 14.0,
                    height_m: 34.0,
                },
            ],
            conductor_offsets_m: vec![-4.0, 0.0, 4.0],
            sag_ratio: 0.03,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap()
    }

    fn float3(mesh: &Mesh, attribute: MeshVertexAttribute) -> Vec<[f32; 3]> {
        match mesh.attribute(attribute).unwrap() {
            VertexAttributeValues::Float32x3(values) => values.clone(),
            other => panic!("{other:?}"),
        }
    }

    /// Every piece of every conductor is one quad: its four vertices sit at
    /// the piece's two ends, each knowing the other end, on both sides.
    #[test]
    fn every_piece_is_one_quad_between_its_ends() {
        let line = line();
        let pieces = line.pieces();
        assert_eq!(
            pieces.len(),
            3 * (40 + 30),
            "200 m and 150 m in 5 m pieces, 3 wires"
        );
        let mesh = wire_mesh(std::slice::from_ref(&line)).unwrap();
        let positions = float3(&mesh, Mesh::ATTRIBUTE_POSITION);
        let others = float3(&mesh, ATTRIBUTE_WIRE_OTHER);
        let shapes = float3(&mesh, ATTRIBUTE_WIRE_SHAPE);
        assert_eq!(positions.len(), 4 * pieces.len());
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("u32 indices");
        };
        assert_eq!(indices.len(), 6 * pieces.len());
        for (k, piece) in pieces.iter().enumerate() {
            let start = piece.start_m.map(|v| v as f32);
            let end = piece.end_m.map(|v| v as f32);
            for v in 0..4 {
                let i = 4 * k + v;
                let (at, other, end_sign) = if v < 2 {
                    (start, end, -1.0)
                } else {
                    (end, start, 1.0)
                };
                assert_eq!(positions[i], at);
                assert_eq!(others[i], other);
                assert_eq!(
                    shapes[i],
                    [if v % 2 == 0 { 1.0 } else { -1.0 }, end_sign, 0.015]
                );
            }
        }
        // Consecutive pieces of a conductor meet: the wire is continuous.
        assert_eq!(positions[2], positions[4]);
        assert!(wire_mesh(&[]).is_none());
    }

    /// A tower stands on its support's ground, with its cross-arm across
    /// the line: square to the span at an end, on the bisector at a turn.
    #[test]
    fn towers_stand_on_their_ground_with_the_arm_across_the_line() {
        let line = line();
        let transforms = tower_transforms(&line);
        assert_eq!(transforms.len(), 3);
        assert_eq!(transforms[0].translation, Vec3::new(0.0, 0.0, 10.0));
        assert_eq!(transforms[2].translation, Vec3::new(200.0, 150.0, 14.0));
        for (index, transform) in transforms.iter().enumerate() {
            let arm = line.cross_arm(index);
            let drawn = transform.rotation * Vec3::X;
            let expected = Vec3::new(arm[0] as f32, arm[1] as f32, 0.0);
            assert!(
                (drawn - expected).length() < 1e-6,
                "{index}: {drawn} vs {expected}"
            );
        }
        // Each conductor's attachment is on the drawn arm at the top.
        let piece = line.pieces()[0];
        let attach = Vec3::from_array(piece.start_m.map(|v| v as f32));
        let local = transforms[0]
            .compute_affine()
            .inverse()
            .transform_point3(attach);
        assert!(
            local.y.abs() < 1e-4 && (local.z - 30.0).abs() < 1e-4,
            "{local}"
        );
        assert!(local.x.abs() <= arm_reach_m(&line.conductor_offsets_m));
    }

    /// The tower reaches the attachment height and no higher, and its arm
    /// a metre past the outer conductors.
    #[test]
    fn a_tower_reaches_its_attachments() {
        let mesh = tower_mesh(30.0, arm_reach_m(&[-4.0, 0.0, 4.0]));
        let positions = float3(&mesh, Mesh::ATTRIBUTE_POSITION);
        let (lo, hi) = positions.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(Vec3::from_array(*p)), hi.max(Vec3::from_array(*p))),
        );
        assert!((hi.z - 30.0).abs() < 0.5, "top {hi}");
        assert!(lo.z > -0.5, "bottom {lo}");
        assert!(
            (hi.x - 5.0).abs() < 0.5 && (lo.x + 5.0).abs() < 0.5,
            "arm {lo} {hi}"
        );
    }

    fn app(document: serde_json::Value) -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<WireMaterial>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<PowerLines>()
            .insert_resource(RuntimeOverlayVisibility::default())
            .insert_resource(ReplayState::new(Some(
                serde_json::from_value(document).unwrap(),
            )))
            .add_systems(Update, sync_power_lines);
        app
    }

    fn parts(app: &mut App) -> Vec<(PowerLinePart, Visibility)> {
        let mut query = app.world_mut().query::<(&PowerLinePart, &Visibility)>();
        query.iter(app.world()).map(|(p, v)| (*p, *v)).collect()
    }

    /// Built once per run: one wire mesh and a tower per support, hidden and
    /// shown by the toggle without a rebuild, replaced by the next run's.
    #[test]
    fn lines_are_built_once_per_run_and_follow_the_toggle() {
        let lines = serde_json::to_value(vec![line()]).unwrap();
        let mut app = app(serde_json::json!({"meta": {"power_lines": lines}, "frames": []}));
        app.update();
        app.update();
        let drawn = app.world().resource::<PowerLines>();
        assert_eq!(drawn.builds(), 1);
        assert_eq!((drawn.line_count(), drawn.tower_count()), (1, 3));
        assert_eq!(drawn.piece_count(), 210);
        let all = parts(&mut app);
        assert_eq!(
            all.iter()
                .filter(|(p, _)| *p == PowerLinePart::Wires)
                .count(),
            1
        );
        assert_eq!(
            all.iter()
                .filter(|(p, _)| *p == PowerLinePart::Tower)
                .count(),
            3
        );
        // Two tower heights, two shared meshes, plus the wires and the
        // corridors.
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), 4);
        // The corridors are built, two spans of them, and hidden: the
        // overlay is off by default.
        assert_eq!(app.world().resource::<PowerLines>().corridor_count(), 2);
        assert_eq!(
            all.iter()
                .filter(|(p, v)| *p == PowerLinePart::Clearance && *v == Visibility::Hidden)
                .count(),
            1
        );
        // Its toggle shows the corridors and nothing else; Power Lines
        // leaves them alone.
        app.world_mut()
            .resource_mut::<RuntimeOverlayVisibility>()
            .power_line_clearance = true;
        app.update();
        for (part, visibility) in parts(&mut app) {
            assert_eq!(visibility, Visibility::Inherited, "{part:?}");
        }
        app.world_mut()
            .resource_mut::<RuntimeOverlayVisibility>()
            .power_line_clearance = false;

        app.world_mut()
            .resource_mut::<RuntimeOverlayVisibility>()
            .power_lines = false;
        app.update();
        assert!(parts(&mut app)
            .iter()
            .all(|(_, v)| *v == Visibility::Hidden));
        assert_eq!(app.world().resource::<PowerLines>().builds(), 1);

        // The next run has none: everything goes, meshes included.
        let previous = app.world().resource::<ReplayState>().epoch();
        let mut next = ReplayState::new(Some(
            serde_json::from_value(serde_json::json!({"meta": {}, "frames": []})).unwrap(),
        ));
        next.succeed(&ReplayState::default());
        assert_ne!(next.epoch(), previous);
        app.insert_resource(next);
        app.update();
        assert!(parts(&mut app).is_empty());
        assert_eq!(app.world().resource::<PowerLines>().builds(), 2);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), 0);
    }

    /// A line the model refuses is not drawn and its reason is kept; the
    /// valid line beside it is still drawn in full.
    #[test]
    fn a_refused_line_is_reported_and_hides_nothing_else() {
        let good = serde_json::to_value(line()).unwrap();
        let mut bad = good.clone();
        bad["id"] = serde_json::json!("line-bare");
        bad["conductor_offsets_m"] = serde_json::json!([]);
        let mut app = app(serde_json::json!({"meta": {"power_lines": [bad, good]}, "frames": []}));
        app.update();
        let drawn = app.world().resource::<PowerLines>();
        assert_eq!((drawn.line_count(), drawn.tower_count()), (1, 3));
        assert_eq!(drawn.refused().len(), 1);
        assert!(
            drawn.refused()[0].contains("line-bare"),
            "{:?}",
            drawn.refused()
        );
        assert_eq!(
            parts(&mut app).len(),
            5,
            "the good line's wires, three towers and its corridors"
        );
    }

    /// Every point of a corridor's cross-section is exactly the conductor
    /// radius plus the clearance from the segment the conductors hang on
    /// there, all round, in one flat section, at every piece joint of both
    /// spans; and both spans' sections meet at the support between them.
    #[test]
    fn a_corridor_is_the_clearance_round_the_conductors_everywhere() {
        use argusnet_world::powerline::segment_distance;
        let line = line();
        let reach = 0.015 + 10.0;
        for (span, count) in [(0, 40), (1, 30)] {
            for k in 0..=count {
                let t = k as f64 / count as f64;
                let (low, high) = conductor_segment(&line, span, t);
                // The segment holds every conductor there.
                for &offset in &line.conductor_offsets_m {
                    let c = line.conductor_at(span, offset, t);
                    assert!(segment_distance(c, c, low, high) < 1e-9);
                }
                let ring = clearance_ring(&line, span, t);
                assert_eq!(ring.len(), 2 * (CLEARANCE_HALF_SEGMENTS + 1));
                for (point, normal) in &ring {
                    let p = point.to_array().map(f64::from);
                    let d = segment_distance(p, p, low, high);
                    assert!((d - reach).abs() < 1e-3, "{span} {t}: {d}");
                    assert!((normal.length() - 1.0).abs() < 1e-5);
                }
                // One flat section holding the conductors' segment: every
                // point, and both ends of the segment, on one plane.
                let (p0, p1, p2) = (
                    ring[0].0,
                    ring[CLEARANCE_HALF_SEGMENTS / 2].0,
                    ring[CLEARANCE_HALF_SEGMENTS + 3].0,
                );
                let plane = (p1 - p0).cross(p2 - p0).normalize();
                for point in ring.iter().map(|(p, _)| *p).chain([
                    Vec3::from_array(low.map(|v| v as f32)),
                    Vec3::from_array(high.map(|v| v as f32)),
                ]) {
                    let off = (point - p0).dot(plane);
                    assert!(off.abs() < 1e-2, "{span} {t}: {off}");
                }
            }
        }
        // The two spans' sections at the support between them hold the same
        // conductors.
        let (a0, a1) = conductor_segment(&line, 0, 1.0);
        let (b0, b1) = conductor_segment(&line, 1, 0.0);
        for (p, q) in [(a0, b0), (a1, b1)] {
            assert!(segment_distance(p, p, q, q) < 1e-9, "{p:?} {q:?}");
        }
    }

    /// The tube: one ring per piece joint per span, quads between rings, and
    /// a fan closing each of the line's two ends.
    #[test]
    fn a_corridor_mesh_is_closed_rings_along_each_span() {
        let line = line();
        let mesh = clearance_mesh(std::slice::from_ref(&line)).unwrap();
        let ring = 2 * (CLEARANCE_HALF_SEGMENTS + 1);
        let rings = (40 + 1) + (30 + 1);
        let positions = float3(&mesh, Mesh::ATTRIBUTE_POSITION);
        assert_eq!(positions.len(), rings * ring + 2 * (1 + ring));
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("u32 indices");
        };
        assert_eq!(indices.len(), (40 + 30) * ring * 6 + 2 * ring * 3);
        assert!(indices.iter().all(|&i| (i as usize) < positions.len()));
        assert!(clearance_mesh(&[]).is_none());
    }
}
