//! Persistent domain markers with shared geometry and holographic materials.
//!
//! One entity per presented [`EntityKey`], kept across frames (P2.0). Entities
//! are spawned and despawned only when membership changes, looked up through
//! [`MarkerEntityIndex`] rather than by a scan, and their transforms and
//! looks are recomputed only when a pose, the view or the overlay settings
//! changed, and written only where they differ, so an unchanged marker does
//! not register as changed. A new epoch despawns them all: keys from another
//! run name nothing here.
use crate::{
    app::{is_marker_visible, marker_appearance, MainCamera},
    glyphs::{GlyphProjection, NODE_GLYPH_PX, TRACK_GLYPH_PX, TRUTH_GLYPH_PX},
    presentation::EntityKey,
    replay::{MarkerKind, RuntimeMarker},
    schema::ScenePackage,
    state::{CurrentRuntimeMarkers, RuntimeOverlayVisibility},
};
use bevy::asset::embedded_asset;
use bevy::ecs::query::QueryData;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use std::collections::HashMap;

/// Which entity presents which key, and what its presentation was last
/// computed from.
#[derive(Resource, Debug, Default)]
pub struct MarkerEntityIndex {
    epoch: Option<u64>,
    entities: HashMap<EntityKey, Entity>,
    membership_revision: Option<u64>,
    pose_revision: Option<u64>,
    view: Option<GlyphProjection>,
    recomputes: u64,
}

impl MarkerEntityIndex {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn entity(&self, key: EntityKey) -> Option<Entity> {
        self.entities.get(&key).copied()
    }

    /// How many times every marker's presentation has been recomputed. The
    /// work is gated on dirty revisions, the view and the settings, so a
    /// still frame adds nothing here.
    pub fn recomputes(&self) -> u64 {
        self.recomputes
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
            .init_resource::<MarkerEntityIndex>()
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
pub fn setup_assets(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
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
    transform: &'static mut Transform,
    visible: &'static mut Visibility,
    mesh: &'static mut Mesh3d,
    material: &'static mut MeshMaterial3d<HologramMaterial>,
}

/// The presentation stage for markers. See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn sync_marker_meshes(
    mut commands: Commands,
    markers: Res<CurrentRuntimeMarkers>,
    package: Res<ScenePackage>,
    visibility: Res<RuntimeOverlayVisibility>,
    camera: Query<(&GlobalTransform, &Projection, &Camera), With<MainCamera>>,
    mut index: ResMut<MarkerEntityIndex>,
    mut assets: ResMut<MarkerAssets>,
    mut materials: ResMut<Assets<HologramMaterial>>,
    mut entities: Query<MarkerEntity>,
) {
    let projection = camera
        .single()
        .map(|(t, p, c)| GlyphProjection::from_camera(t, p, c.logical_viewport_size().map(|s| s.y)))
        .unwrap_or_default();

    // A new epoch: every key from the old one is void.
    if index.epoch != Some(markers.epoch()) {
        for (_, entity) in index.entities.drain() {
            commands.entity(entity).despawn();
        }
        index.epoch = Some(markers.epoch());
        index.membership_revision = None;
    }

    let membership_changed = index.membership_revision != Some(markers.membership_revision());
    if membership_changed {
        index.entities.retain(|key, entity| {
            let present = markers.index_of(*key).is_some();
            if !present {
                commands.entity(*entity).despawn();
            }
            present
        });
        // In marker order, so spawn order is deterministic.
        for marker in &markers.markers {
            if index.entities.contains_key(&marker.key) {
                continue;
            }
            let (t, v, m, mat) = presentation(
                marker,
                &projection,
                &package,
                &visibility,
                &mut assets,
                &mut materials,
            );
            let entity = commands
                .spawn((marker.key, Mesh3d(m), MeshMaterial3d(mat), t, v))
                .id();
            index.entities.insert(marker.key, entity);
        }
        index.membership_revision = Some(markers.membership_revision());
    }

    let stale = membership_changed
        || index.pose_revision != Some(markers.pose_revision())
        || index.view != Some(projection)
        || visibility.is_changed()
        || package.is_changed();
    if !stale {
        return;
    }
    index.recomputes += 1;
    for marker in &markers.markers {
        let Some(&entity) = index.entities.get(&marker.key) else {
            continue;
        };
        // Spawned this frame: its components land when commands apply.
        let Ok(mut current) = entities.get_mut(entity) else {
            continue;
        };
        let (t, v, m, mat) = presentation(
            marker,
            &projection,
            &package,
            &visibility,
            &mut assets,
            &mut materials,
        );
        current.transform.set_if_neq(t);
        current.visible.set_if_neq(v);
        if current.mesh.0 != m {
            current.mesh.0 = m;
        }
        if current.material.0 != mat {
            current.material.0 = mat;
        }
    }
    index.pose_revision = Some(markers.pose_revision());
    index.view = Some(projection);
}

fn presentation(
    marker: &RuntimeMarker,
    projection: &GlyphProjection,
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
    let (color, confidence_scale) = marker_appearance(marker, package);
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
    use crate::app::sync_current_markers_system;
    use crate::replay::{ReplayDocument, ReplayState};
    use crate::state::{CurrentFrameMetrics, SelectionState};

    fn document(frames: Vec<serde_json::Value>) -> ReplayDocument {
        serde_json::from_value(serde_json::json!({ "frames": frames })).expect("fixture")
    }

    /// `n` frames; a drone that moves, a track and a truth sharing the label
    /// `"same-id"`, and a second drone present on even frames only.
    fn walking(n: usize) -> ReplayDocument {
        document(
            (0..n)
                .map(|f| {
                    let x = f as f64;
                    let mut nodes = vec![serde_json::json!({"node_id": "drone-a",
                        "position": [x, 0.0, 20.0], "velocity": [1.0, 0.0, 0.0], "is_mobile": true})];
                    if f % 2 == 0 {
                        nodes.push(serde_json::json!({"node_id": "drone-b",
                            "position": [0.0, x, 20.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}));
                    }
                    serde_json::json!({"timestamp_s": x, "nodes": nodes,
                        "tracks": [{"track_id": "same-id", "position": [x, 5.0, 20.0], "velocity": [0.0, 0.0, 0.0],
                                    "measurement_std_m": (f % 60) as f64}],
                        "truths": [{"target_id": "same-id", "position": [x, 6.0, 20.0], "velocity": [0.0, 0.0, 0.0]}]})
                })
                .collect(),
        )
    }

    fn test_app(replay: ReplayState) -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<HologramMaterial>>()
            .init_resource::<MarkerEntityIndex>()
            .insert_resource(replay)
            .insert_resource(
                ScenePackage::load(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
                )
                .unwrap(),
            )
            .insert_resource(RuntimeOverlayVisibility::default())
            .insert_resource(CurrentRuntimeMarkers::default())
            .insert_resource(CurrentFrameMetrics::default())
            .insert_resource(SelectionState::default())
            .add_systems(Startup, setup_assets)
            .add_systems(
                Update,
                (sync_current_markers_system, sync_marker_meshes).chain(),
            );
        app
    }

    fn marker_entities(app: &mut App) -> Vec<(EntityKey, Entity)> {
        let mut query = app.world_mut().query::<(&EntityKey, Entity)>();
        let mut found: Vec<_> = query.iter(app.world()).map(|(k, e)| (*k, e)).collect();
        found.sort();
        found
    }

    fn seek(app: &mut App, frame: usize) {
        app.world_mut().resource_mut::<ReplayState>().step_to(frame);
        app.update();
    }

    #[test]
    fn identity_survives_motion_and_hiding_but_not_removal() {
        let mut app = test_app(ReplayState::new(Some(walking(10))));
        app.update();
        let drone = app
            .world()
            .resource::<CurrentRuntimeMarkers>()
            .key_of(MarkerKind::Node, "drone-a")
            .unwrap();
        let entity = app
            .world()
            .resource::<MarkerEntityIndex>()
            .entity(drone)
            .unwrap();

        app.world_mut()
            .resource_mut::<RuntimeOverlayVisibility>()
            .nodes = false;
        seek(&mut app, 8);
        assert_eq!(
            app.world().resource::<MarkerEntityIndex>().entity(drone),
            Some(entity)
        );
        assert_eq!(
            app.world().get::<Transform>(entity).unwrap().translation.x,
            8.0
        );
        assert_eq!(
            *app.world().get::<Visibility>(entity).unwrap(),
            Visibility::Hidden
        );

        // drone-b exists on even frames only: it comes and goes as membership.
        let b = app
            .world()
            .resource::<CurrentRuntimeMarkers>()
            .key_of(MarkerKind::Node, "drone-b")
            .unwrap();
        assert!(app
            .world()
            .resource::<MarkerEntityIndex>()
            .entity(b)
            .is_some());
        seek(&mut app, 7);
        assert!(app
            .world()
            .resource::<MarkerEntityIndex>()
            .entity(b)
            .is_none());
        assert_eq!(marker_entities(&mut app).len(), 3);
    }

    #[test]
    fn different_kinds_with_same_label_remain_distinct_and_assets_are_reused() {
        let mut app = test_app(ReplayState::new(Some(walking(4))));
        app.update();
        let kinds: Vec<MarkerKind> = marker_entities(&mut app)
            .iter()
            .map(|(k, _)| k.kind())
            .collect();
        assert!(kinds.contains(&MarkerKind::Track) && kinds.contains(&MarkerKind::Truth));
        let meshes = app.world().resource::<Assets<Mesh>>().len();
        let materials = app.world().resource::<Assets<HologramMaterial>>().len();
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), meshes);
        assert_eq!(
            app.world().resource::<Assets<HologramMaterial>>().len(),
            materials
        );
    }

    /// P2.0's leak check: a long loop of seeks and whole-run reloads leaves
    /// exactly one entity per presented marker, and the shared assets
    /// bounded by what is distinct, not by how many frames were shown.
    #[test]
    fn long_seek_and_reload_loops_leak_nothing() {
        let mut app = test_app(ReplayState::new(Some(walking(120))));
        app.update();
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as usize
        };
        let mut max_materials = 0;
        for round in 0..2_000 {
            if round % 97 == 0 {
                // A reload: a different run, so a new epoch.
                let previous = app.world().resource::<ReplayState>().clone();
                let mut fresh = ReplayState::new(Some(walking(60 + round % 50)));
                fresh.succeed(&previous);
                app.insert_resource(fresh);
            }
            let frames = app.world().resource::<ReplayState>().frame_count();
            seek(&mut app, next() % frames);
            let presented = app
                .world()
                .resource::<CurrentRuntimeMarkers>()
                .markers
                .len();
            assert_eq!(marker_entities(&mut app).len(), presented, "round {round}");
            assert_eq!(app.world().resource::<MarkerEntityIndex>().len(), presented);
            max_materials =
                max_materials.max(app.world().resource::<Assets<HologramMaterial>>().len());
        }
        // Four track-quality bands and the three kind colours, at most.
        assert!(max_materials <= 7, "{max_materials} materials");
        assert!(app.world().resource::<Assets<Mesh>>().len() <= 4);
    }

    /// A paused frame, with the view still, recomputes nothing and writes
    /// nothing: no marker's transform registers as changed.
    #[test]
    fn an_unchanged_frame_changes_no_marker() {
        let mut app = test_app(ReplayState::new(Some(walking(10))));
        app.update();
        app.update();
        let recomputes = app.world().resource::<MarkerEntityIndex>().recomputes();
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<MarkerEntityIndex>().recomputes(),
            recomputes
        );
        seek(&mut app, 4);
        assert_eq!(
            app.world().resource::<MarkerEntityIndex>().recomputes(),
            recomputes + 1
        );
        let before = app.world().read_change_tick();
        app.update();
        let now = app.world().read_change_tick();
        let mut query = app.world_mut().query::<(Ref<Transform>, &EntityKey)>();
        let written = query
            .iter(app.world())
            .filter(|(transform, _)| transform.last_changed().is_newer_than(before, now))
            .count();
        assert_eq!(written, 0, "an unchanged marker was written");
    }

    /// No entity outlives its epoch: a reload respawns every marker, even
    /// where the new run's ids happen to intern to the same numbers.
    #[test]
    fn a_new_epoch_respawns_every_marker() {
        let mut app = test_app(ReplayState::new(Some(walking(10))));
        app.update();
        let before: Vec<Entity> = marker_entities(&mut app)
            .into_iter()
            .map(|(_, e)| e)
            .collect();
        let previous = app.world().resource::<ReplayState>().clone();
        // The same run's shape again: identical ids, identical interning.
        let mut fresh = ReplayState::new(Some(walking(10)));
        fresh.succeed(&previous);
        app.insert_resource(fresh);
        app.update();
        let after: Vec<Entity> = marker_entities(&mut app)
            .into_iter()
            .map(|(_, e)| e)
            .collect();
        assert_eq!(after.len(), before.len());
        assert!(
            after.iter().all(|e| !before.contains(e)),
            "an entity survived its epoch"
        );
    }

    /// Every frame has its own metrics: they follow the cursor even when the
    /// same entities persist from frame to frame.
    #[test]
    fn frame_metrics_follow_the_frame_not_the_membership() {
        let frames = (0..4)
            .map(|f| {
                serde_json::json!({"timestamp_s": f as f64, "tracks": [], "truths": [],
                    "nodes": [{"node_id": "tower", "position": [0.0, 0.0, 0.0],
                               "velocity": [0.0, 0.0, 0.0], "is_mobile": false}],
                    "metrics": {"mean_error_m": f as f64 * 10.0}})
            })
            .collect();
        let mut app = test_app(ReplayState::new(Some(document(frames))));
        for frame in [0, 3, 1] {
            seek(&mut app, frame);
            let metrics = app
                .world()
                .resource::<CurrentFrameMetrics>()
                .metrics
                .clone();
            assert_eq!(
                metrics.and_then(|m| m.mean_error_m),
                Some(frame as f32 * 10.0),
                "frame {frame}"
            );
        }
    }

    /// Selecting a truth that shares its track's label stays on the truth.
    #[test]
    fn a_selected_truth_is_not_resolved_to_its_track() {
        let mut app = test_app(ReplayState::new(Some(walking(10))));
        app.update();
        let (truth_key, epoch) = {
            let markers = app.world().resource::<CurrentRuntimeMarkers>();
            (
                markers.key_of(MarkerKind::Truth, "same-id").unwrap(),
                markers.epoch(),
            )
        };
        app.world_mut().resource_mut::<SelectionState>().selected = Some(crate::state::Selection {
            key: truth_key,
            epoch,
            label: "same-id".into(),
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            details: crate::state::SelectionDetails::Truth(Default::default()),
        });
        seek(&mut app, 3);
        let selected = app
            .world()
            .resource::<SelectionState>()
            .selected
            .clone()
            .unwrap();
        assert_eq!(selected.key, truth_key);
        assert_eq!(selected.details.kind_label(), "Truth");
        assert_eq!(
            selected.position.y, 6.0,
            "the truth's pose, not the track's"
        );

        // A new epoch voids it.
        let previous = app.world().resource::<ReplayState>().clone();
        let mut fresh = ReplayState::new(Some(walking(10)));
        fresh.succeed(&previous);
        app.insert_resource(fresh);
        app.update();
        assert!(app.world().resource::<SelectionState>().selected.is_none());
    }
}
