//! Persistent domain markers with shared geometry and holographic materials.
use crate::{
    app::{is_marker_visible, marker_appearance, MainCamera},
    glyphs::{GlyphProjection, NODE_GLYPH_PX, TRACK_GLYPH_PX, TRUTH_GLYPH_PX},
    replay::{MarkerKind, ReplayState, RuntimeMarker},
    schema::ScenePackage,
    state::{CurrentRuntimeMarkers, RuntimeOverlayVisibility},
};
use bevy::asset::embedded_asset;
use bevy::ecs::query::QueryData;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use std::collections::HashMap;

#[derive(Component, Debug, PartialEq, Eq)]
pub enum MarkerIdentity {
    Platform(String),
    Track(String),
    Truth(String),
}
impl MarkerIdentity {
    fn new(marker: &RuntimeMarker) -> Self {
        match marker.kind {
            MarkerKind::Node => Self::Platform(marker.label.clone()),
            MarkerKind::Track => Self::Track(marker.label.clone()),
            MarkerKind::Truth => Self::Truth(marker.label.clone()),
        }
    }
    fn key(&self) -> (MarkerKind, &str) {
        match self {
            Self::Platform(id) => (MarkerKind::Node, id),
            Self::Track(id) => (MarkerKind::Track, id),
            Self::Truth(id) => (MarkerKind::Truth, id),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct HologramMaterial {
    #[uniform(0)]
    pub color: Vec4,
}
impl Material for HologramMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://argusnet_viewer/shaders/hologram.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
    fn enable_shadows() -> bool {
        false
    }
    fn enable_prepass() -> bool {
        false
    }
}

#[derive(Resource)]
pub struct MarkerAssets {
    drone: Handle<Mesh>,
    station: Handle<Mesh>,
    track: Handle<Mesh>,
    truth: Handle<Mesh>,
    // Colors are shared across entities, including the four track quality bands.
    materials: HashMap<[u32; 4], Handle<HologramMaterial>>,
}

pub struct MarkerMeshPlugin;
impl Plugin for MarkerMeshPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/hologram.wgsl");
        app.add_plugins(MaterialPlugin::<HologramMaterial>::default())
            .add_systems(Startup, setup_assets);
    }
}

fn drone_mesh() -> Mesh {
    let mut mesh = Mesh::from(Cuboid::new(0.55, 0.85, 0.3));
    for angle in [-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4] {
        let arm = Mesh::from(Cuboid::new(2.0, 0.16, 0.12))
            .transformed_by(Transform::from_rotation(Quat::from_rotation_z(angle)));
        mesh.merge(&arm).expect("cuboids share vertex attributes");
    }
    mesh
}
fn setup_assets(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands.insert_resource(MarkerAssets {
        drone: meshes.add(drone_mesh()),
        station: meshes.add(Cuboid::new(1.2, 1.2, 1.5)),
        track: meshes.add(Sphere::new(1.0).mesh().ico(0).unwrap()),
        truth: meshes.add(Sphere::new(0.85).mesh().uv(12, 8)),
        materials: HashMap::new(),
    });
}

#[derive(QueryData)]
#[query_data(mutable)]
pub struct MarkerEntity {
    entity: Entity,
    identity: &'static MarkerIdentity,
    transform: &'static mut Transform,
    visible: &'static mut Visibility,
    mesh: &'static mut Mesh3d,
    material: &'static mut MeshMaterial3d<HologramMaterial>,
}

// Runs after marker interpolation and before overlays. Hidden markers remain
// entities; missing IDs are removed, so seeking/replay replacement cannot leak.
#[allow(clippy::too_many_arguments)]
pub fn sync_marker_meshes(
    mut commands: Commands,
    markers: Res<CurrentRuntimeMarkers>,
    replay: Res<ReplayState>,
    package: Res<ScenePackage>,
    visibility: Res<RuntimeOverlayVisibility>,
    camera: Query<(&GlobalTransform, &Projection, &Camera), With<MainCamera>>,
    mut assets: ResMut<MarkerAssets>,
    mut materials: ResMut<Assets<HologramMaterial>>,
    mut entities: Query<MarkerEntity>,
) {
    let projection = camera
        .single()
        .map(|(t, p, c)| GlyphProjection::from_camera(t, p, c.logical_viewport_size().map(|s| s.y)))
        .unwrap_or_default();
    let mut remaining: HashMap<_, _> = markers
        .markers
        .iter()
        .map(|m| ((m.kind, m.label.as_str()), m))
        .collect();
    for mut entity in &mut entities {
        let Some(marker) = remaining.remove(&entity.identity.key()) else {
            commands.entity(entity.entity).despawn();
            continue;
        };
        let (t, v, m, mat) = presentation(
            marker,
            &projection,
            &replay,
            &package,
            &visibility,
            &mut assets,
            &mut materials,
        );
        *entity.transform = t;
        *entity.visible = v;
        entity.mesh.0 = m;
        entity.material.0 = mat;
    }
    // Iterate the source order, not HashMap order, for deterministic spawn order.
    for marker in &markers.markers {
        if remaining
            .remove(&(marker.kind, marker.label.as_str()))
            .is_none()
        {
            continue;
        }
        let (t, v, m, mat) = presentation(
            marker,
            &projection,
            &replay,
            &package,
            &visibility,
            &mut assets,
            &mut materials,
        );
        commands.spawn((
            MarkerIdentity::new(marker),
            Mesh3d(m),
            MeshMaterial3d(mat),
            t,
            v,
        ));
    }
}

fn presentation(
    marker: &RuntimeMarker,
    projection: &GlyphProjection,
    replay: &ReplayState,
    package: &ScenePackage,
    visibility: &RuntimeOverlayVisibility,
    assets: &mut MarkerAssets,
    materials: &mut Assets<HologramMaterial>,
) -> (
    Transform,
    Visibility,
    Handle<Mesh>,
    Handle<HologramMaterial>,
) {
    let (color, confidence_scale) = marker_appearance(marker, replay, package);
    let linear = LinearRgba::from(color).to_vec4();
    let key = linear.to_array().map(f32::to_bits);
    let material = assets
        .materials
        .entry(key)
        .or_insert_with(|| materials.add(HologramMaterial { color: linear }))
        .clone();
    let (mesh, pixels) = match marker.kind {
        MarkerKind::Node if marker.is_mobile == Some(true) => (assets.drone.clone(), NODE_GLYPH_PX),
        MarkerKind::Node => (assets.station.clone(), NODE_GLYPH_PX),
        MarkerKind::Track => (assets.track.clone(), TRACK_GLYPH_PX),
        MarkerKind::Truth => (assets.truth.clone(), TRUTH_GLYPH_PX),
    };
    let yaw = if marker.velocity.truncate().length_squared() > 0.01 {
        marker.velocity.y.atan2(marker.velocity.x) - std::f32::consts::FRAC_PI_2
    } else {
        0.0
    };
    let transform = Transform::from_translation(marker.position)
        .with_rotation(Quat::from_rotation_z(yaw))
        .with_scale(Vec3::splat(
            projection.radius(marker.position, pixels) * confidence_scale,
        ));
    let visible = if is_marker_visible(marker, visibility) {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    (transform, visible, mesh, material)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<HologramMaterial>>()
            .insert_resource(ReplayState::new(None))
            .insert_resource(
                ScenePackage::load(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
                )
                .unwrap(),
            )
            .insert_resource(RuntimeOverlayVisibility::default())
            .insert_resource(CurrentRuntimeMarkers::default())
            .add_systems(Startup, setup_assets)
            .add_systems(Update, sync_marker_meshes);
        app
    }
    fn marker(kind: MarkerKind, x: f32) -> RuntimeMarker {
        RuntimeMarker {
            kind,
            label: "same-id".into(),
            position: Vec3::new(x, 0.0, 20.0),
            velocity: Vec3::ZERO,
            is_mobile: Some(true),
        }
    }
    #[test]
    fn identity_survives_motion_and_hiding_but_not_removal() {
        let mut app = test_app();
        app.world_mut()
            .resource_mut::<CurrentRuntimeMarkers>()
            .markers = vec![marker(MarkerKind::Node, 1.0)];
        app.update();
        let mut query = app
            .world_mut()
            .query_filtered::<Entity, With<MarkerIdentity>>();
        let id = query.single(app.world()).unwrap();
        app.world_mut()
            .resource_mut::<CurrentRuntimeMarkers>()
            .markers[0]
            .position
            .x = 8.0;
        app.world_mut()
            .resource_mut::<RuntimeOverlayVisibility>()
            .nodes = false;
        app.update();
        assert_eq!(query.single(app.world()).unwrap(), id);
        assert_eq!(app.world().get::<Transform>(id).unwrap().translation.x, 8.0);
        assert_eq!(
            *app.world().get::<Visibility>(id).unwrap(),
            Visibility::Hidden
        );
        app.world_mut()
            .resource_mut::<CurrentRuntimeMarkers>()
            .markers
            .clear();
        app.update();
        assert_eq!(query.iter(app.world()).count(), 0);
    }
    #[test]
    fn different_kinds_with_same_label_remain_distinct_and_assets_are_reused() {
        let mut app = test_app();
        app.world_mut()
            .resource_mut::<CurrentRuntimeMarkers>()
            .markers = vec![
            marker(MarkerKind::Node, 1.0),
            marker(MarkerKind::Track, 2.0),
        ];
        app.update();
        let meshes = app.world().resource::<Assets<Mesh>>().len();
        let materials = app.world().resource::<Assets<HologramMaterial>>().len();
        for _ in 0..10 {
            app.update();
        }
        let mut query = app.world_mut().query::<&MarkerIdentity>();
        assert_eq!(query.iter(app.world()).count(), 2);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), meshes);
        assert_eq!(
            app.world().resource::<Assets<HologramMaterial>>().len(),
            materials
        );
        assert!(
            !app.world()
                .resource::<RuntimeOverlayVisibility>()
                .diagnostic_glyphs
        );
    }
}
