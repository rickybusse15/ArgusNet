//! World picking, hover and keyboard selection (P2.3).
//!
//! # A ray, not a radius
//!
//! The pointer becomes a ray through the camera whose viewport holds it, and
//! the ray is cast against meshes with Bevy's `MeshRayCast`. The nearest hit
//! wins, among exactly two kinds of entity:
//!
//! * **markers** whose kind is shown, and which are visible;
//! * **occluders**: the meshes of the scene's base layers (terrain,
//!   buildings), tagged [`PickOccluder`] when they spawn.
//!
//! Occluders block from both faces. The committed terrain is wound facing
//! down while its normals face up, and draws correctly only because its
//! material is double-sided; with the ray cast's default back-face culling,
//! a ray from above passed straight through it. Casting both faces matches
//! how the layers are drawn.
//!
//! So a platform behind a hill is not picked through it, and the nearer of two
//! platforms on one ray is the one picked. Everything else is invisible to the
//! ray: sensor volumes, trails, zone walls, rings and the sky, all of them
//! [`crate::sensor_volumes::PresentationOnly`], never steal a pick, since no
//! interaction mode selects them yet.
//!
//! # Pick proxies
//!
//! A drone glyph is a body and two thin arms. The ray is cast against its
//! bounding box instead (a `SimplifiedMesh`): the glyph's own footprint, gaps
//! between the arms included, and nothing beyond it. The other glyphs are
//! compact and are cast against as drawn. Glyphs are screen-sized, so the
//! proxy is too, whatever the zoom.
//!
//! # Viewports and the UI
//!
//! In split view only the main camera's viewport shows markers; the pointer
//! over the reconstruction viewport picks nothing and clicks there change
//! nothing. The old proximity picker compared window coordinates against
//! projections that could fall outside the main viewport. A pointer the UI
//! owns does not reach the world at all.
//!
//! # Hover
//!
//! Hover is recomputed only when something it depends on moved: the pointer,
//! the camera, the markers, the kind toggles or the occluders. A still pointer
//! over a paused frame casts nothing.
//!
//! # Keyboard and list
//!
//! Every pointer path has a keyboard one: Tab and Shift+Tab step through the
//! shown markers in presentation order, and Escape clears. The inspector's
//! entity list selects by key through [`SelectRequest`].

use std::collections::HashSet;

use bevy::picking::mesh_picking::ray_cast::{
    MeshRayCast, MeshRayCastSettings, RayCastBackfaces, RayCastVisibility,
};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;

use crate::app::{is_marker_visible, MainCamera};
use crate::presentation::{CurrentRuntimeMarkers, EntityKey};
use crate::replay::{MarkerKind, RuntimeMarker};
use crate::state::{
    LayerEntityIndex, LayerVisibilityState, NodeDetails, RuntimeOverlayVisibility, Selection,
    SelectionDetails, SelectionState, TrackDetails, TruthDetails, ViewerUiState,
};

/// The pointer, as the world sees it. The window fills it in the
/// interactive viewer; the headless renderer fills it for probe clicks.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct PointerInput {
    /// Window position in logical pixels; `None` when off the window.
    pub position: Option<Vec2>,
    /// The UI owns the pointer: the world does not see it.
    pub over_ui: bool,
    /// The select button went down this frame. Consumed by the picker.
    pub pressed: bool,
}

/// A base-layer mesh that blocks picks behind it.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct PickOccluder;

/// Bumped whenever the set of occluders, or their visibility, may have
/// changed, so a cached hover is recomputed.
#[derive(Resource, Debug, Default)]
pub struct PickGeneration(pub u64);

/// What the pointer is over.
#[derive(Resource, Debug, Default)]
pub struct HoverState {
    hovered: Option<EntityKey>,
    inputs: Option<HoverInputs>,
    casts: u64,
}

impl HoverState {
    pub fn hovered(&self) -> Option<EntityKey> {
        self.hovered
    }

    /// Ray casts made for hover and clicks so far.
    pub fn casts(&self) -> u64 {
        self.casts
    }
}

/// Everything a hover result depends on. Equal inputs, equal result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverInputs {
    pub pointer: Vec2,
    pub camera: Mat4,
    pub viewport: Rect,
    pub epoch: u64,
    pub membership_revision: u64,
    pub pose_revision: u64,
    pub shown_kinds: [bool; 3],
    pub occluders: u64,
}

/// Every click the picker has consumed, and what the last one did: the
/// headless renderer reports these for its probe clicks.
#[derive(Resource, Debug, Default)]
pub struct PickLog {
    pub clicks: u64,
    /// What the last click selected; `None` when it cleared or was ignored.
    pub last: Option<EntityKey>,
    /// The last click fell on the UI, off the window, or outside the main
    /// viewport, and changed nothing.
    pub last_ignored: bool,
}

/// A selection the UI asks for; the UI only reads the selection itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectRequest {
    Select(EntityKey),
    Clear,
}

/// The ray under a window position, through `camera`, or `None` when the
/// camera is off or the position is outside its viewport.
pub fn cursor_ray(camera: &Camera, transform: &GlobalTransform, cursor: Vec2) -> Option<Ray3d> {
    if !camera.is_active {
        return None;
    }
    let viewport = camera.logical_viewport_rect()?;
    if !viewport_contains(viewport, cursor) {
        return None;
    }
    // `viewport_to_world` takes window coordinates and subtracts the
    // viewport's own origin.
    camera.viewport_to_world(transform, cursor).ok()
}

/// Whether a window position lies in a viewport: half-open, so two viewports
/// that share an edge never both claim a pixel.
pub fn viewport_contains(viewport: Rect, position: Vec2) -> bool {
    position.x >= viewport.min.x
        && position.x < viewport.max.x
        && position.y >= viewport.min.y
        && position.y < viewport.max.y
}

/// The kinds shown, in [`MarkerKind`] order: tracks, truths, nodes.
fn shown_kinds(visibility: &RuntimeOverlayVisibility) -> [bool; 3] {
    [visibility.tracks, visibility.truths, visibility.nodes]
}

/// The marker the ray meets first, or `None` when it meets an occluder, or
/// nothing, first.
pub fn pick_along(
    ray: Ray3d,
    ray_cast: &mut MeshRayCast,
    keys: &Query<&EntityKey>,
    occluders: &Query<(), With<PickOccluder>>,
    markers: &CurrentRuntimeMarkers,
    visibility: &RuntimeOverlayVisibility,
) -> Option<EntityKey> {
    // The kind check as well as the entity's visibility: a toggle flipped this
    // frame has not yet propagated to `InheritedVisibility`.
    let filter = |entity: Entity| {
        occluders.contains(entity)
            || keys.get(entity).is_ok_and(|key| {
                markers
                    .get(*key)
                    .is_some_and(|marker| is_marker_visible(marker, visibility))
            })
    };
    let settings = MeshRayCastSettings::default()
        .with_visibility(RayCastVisibility::Visible)
        .with_filter(&filter);
    let (entity, _) = ray_cast.cast_ray(ray, &settings).first()?;
    keys.get(*entity).ok().copied()
}

/// Where the ray first meets the ground or a base-layer mesh: the point a
/// click on empty world inspects (the map-cell inspector, C3.4).
pub fn ground_point_along(
    ray: Ray3d,
    ray_cast: &mut MeshRayCast,
    occluders: &Query<(), With<PickOccluder>>,
) -> Option<Vec3> {
    let filter = |entity: Entity| occluders.contains(entity);
    let settings = MeshRayCastSettings::default()
        .with_visibility(RayCastVisibility::Visible)
        .with_filter(&filter);
    ray_cast
        .cast_ray(ray, &settings)
        .first()
        .map(|(_, hit)| hit.point)
}

/// A new selection of `marker`. Its details are filled from the current frame
/// by `sync_current_markers_system`, which re-resolves a changed selection.
pub fn selection_of(marker: &RuntimeMarker, epoch: u64) -> Selection {
    Selection {
        key: marker.key,
        epoch,
        label: marker.label.to_string(),
        position: marker.position,
        velocity: marker.velocity,
        details: match marker.kind {
            MarkerKind::Track => SelectionDetails::Track(TrackDetails::default()),
            MarkerKind::Truth => SelectionDetails::Truth(TruthDetails::default()),
            MarkerKind::Node => SelectionDetails::Node(NodeDetails::default()),
        },
    }
}

/// Select `key`, or clear when it names nothing in this epoch.
pub fn select_key(
    selection: &mut SelectionState,
    markers: &CurrentRuntimeMarkers,
    key: Option<EntityKey>,
) {
    match key.and_then(|key| markers.get(key)) {
        Some(marker) => {
            if selection.selected.as_ref().map(|s| (s.key, s.epoch))
                != Some((marker.key, markers.epoch()))
            {
                selection.selected = Some(selection_of(marker, markers.epoch()));
            }
        }
        None => {
            if selection.selected.is_some() {
                selection.clear();
            }
        }
    }
}

/// The next shown marker after `current`, in presentation order, wrapping;
/// backwards with `forward == false`. From nothing, the first (or last).
pub fn cycle_selection(
    markers: &CurrentRuntimeMarkers,
    visibility: &RuntimeOverlayVisibility,
    current: Option<EntityKey>,
    forward: bool,
) -> Option<EntityKey> {
    let shown: Vec<EntityKey> = markers
        .markers
        .iter()
        .filter(|marker| is_marker_visible(marker, visibility))
        .map(|marker| marker.key)
        .collect();
    if shown.is_empty() {
        return None;
    }
    let n = shown.len();
    let at = current.and_then(|key| shown.iter().position(|k| *k == key));
    let index = match (at, forward) {
        (Some(i), true) => (i + 1) % n,
        (Some(i), false) => (i + n - 1) % n,
        (None, true) => 0,
        (None, false) => n - 1,
    };
    Some(shown[index])
}

/// Copies the window's pointer into [`PointerInput`]. Without a primary
/// window (headless) it leaves the resource to whoever drives it.
pub fn pointer_from_window_system(
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
    contexts: Option<EguiContexts>,
    ui_state: Option<Res<ViewerUiState>>,
    mut pointer: ResMut<PointerInput>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let position = window.cursor_position();
    // egui reports its own areas and popups. The shell's panels are drawn on
    // its root `Ui`, on the background layer, which egui does not count as
    // "over egui": without the canvas test a click on the rail or inspector
    // also reached the world, and on empty terrain cleared the selection.
    let off_canvas = match (&ui_state, position) {
        (Some(ui_state), Some(position)) => !ui_state.on_canvas([position.x, position.y]),
        _ => false,
    };
    let over_ui = off_canvas
        || contexts
            .and_then(|mut contexts| {
                contexts
                    .ctx_mut()
                    .ok()
                    .map(|ctx| ctx.is_pointer_over_egui() || ctx.egui_wants_pointer_input())
            })
            .unwrap_or(false);
    let next = PointerInput {
        position,
        over_ui,
        pressed: buttons.just_pressed(MouseButton::Left),
    };
    pointer.set_if_neq(next);
}

/// Tags base-layer meshes as occluders as they spawn, and marks the picking
/// inputs stale when a layer is shown or hidden.
pub fn tag_pick_occluders_system(
    mut commands: Commands,
    added: Query<Entity, Added<Mesh3d>>,
    parents: Query<&ChildOf>,
    layers: Res<LayerEntityIndex>,
    layer_visibility: Res<LayerVisibilityState>,
    mut generation: ResMut<PickGeneration>,
) {
    if layer_visibility.is_changed() {
        generation.0 += 1;
    }
    if added.is_empty() {
        return;
    }
    let roots: HashSet<Entity> = layers.entities_by_layer.values().copied().collect();
    let mut tagged = false;
    for entity in &added {
        if parents
            .iter_ancestors(entity)
            .any(|ancestor| roots.contains(&ancestor))
        {
            commands
                .entity(entity)
                .insert((PickOccluder, RayCastBackfaces));
            tagged = true;
        }
    }
    if tagged {
        generation.0 += 1;
    }
}

/// What the picker writes besides the hover: the selection, its log, and the
/// inspected map cell.
type PickOutputs<'w> = (
    Res<'w, PickGeneration>,
    ResMut<'w, HoverState>,
    ResMut<'w, SelectionState>,
    ResMut<'w, PickLog>,
    ResMut<'w, ViewerUiState>,
);

/// Hover, and selection on click. See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn pick_world_system(
    mut pointer: ResMut<PointerInput>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut ray_cast: MeshRayCast,
    keys: Query<&EntityKey>,
    occluders: Query<(), With<PickOccluder>>,
    markers: Res<CurrentRuntimeMarkers>,
    visibility: Res<RuntimeOverlayVisibility>,
    (generation, mut hover, mut selection, mut log, mut ui_state): PickOutputs,
) {
    let pressed = std::mem::take(&mut pointer.bypass_change_detection().pressed);
    let target = if pointer.over_ui {
        None
    } else {
        pointer.position.and_then(|position| {
            let (camera, transform) = cameras.single().ok()?;
            let ray = cursor_ray(camera, transform, position)?;
            let inputs = HoverInputs {
                pointer: position,
                camera: transform.to_matrix(),
                viewport: camera.logical_viewport_rect()?,
                epoch: markers.epoch(),
                membership_revision: markers.membership_revision(),
                pose_revision: markers.pose_revision(),
                shown_kinds: shown_kinds(&visibility),
                occluders: generation.0,
            };
            Some((ray, inputs))
        })
    };

    let inputs = target.map(|(_, inputs)| inputs);
    if hover.inputs != inputs {
        let hovered = target.and_then(|(ray, _)| {
            pick_along(ray, &mut ray_cast, &keys, &occluders, &markers, &visibility)
        });
        let hover = hover.as_mut();
        hover.inputs = inputs;
        hover.casts += u64::from(target.is_some());
        hover.hovered = hovered;
    }

    // A click selects what is under it, or clears on empty world. A click on
    // the UI, off the window, or in another viewport changes nothing.
    if pressed {
        log.clicks += 1;
        log.last_ignored = target.is_none();
        log.last = None;
        if let Some((ray, _)) = target {
            let picked = hover.hovered;
            select_key(&mut selection, &markers, picked);
            log.last = picked;
            // Empty world: inspect the ground there instead (§7.3).
            let cell = match picked {
                Some(_) => None,
                None => ground_point_along(ray, &mut ray_cast, &occluders)
                    .map(|point| [point.x, point.y]),
            };
            if ui_state.inspected_cell != cell {
                ui_state.inspected_cell = cell;
            }
        }
    }
}

/// Tab / Shift+Tab / Escape, and the inspector list's requests.
#[allow(clippy::too_many_arguments)]
pub fn keyboard_selection_system(
    keys: Res<ButtonInput<KeyCode>>,
    gate: Res<crate::bindings::KeyboardGate>,
    bindings: Res<crate::bindings::Bindings>,
    escape: Res<crate::camera_control::EscapeUsed>,
    markers: Res<CurrentRuntimeMarkers>,
    visibility: Res<RuntimeOverlayVisibility>,
    mut ui_state: ResMut<ViewerUiState>,
    mut selection: ResMut<SelectionState>,
) {
    if let Some(request) = ui_state.bypass_change_detection().select_request.take() {
        match request {
            SelectRequest::Select(key) => select_key(&mut selection, &markers, Some(key)),
            SelectRequest::Clear => select_key(&mut selection, &markers, None),
        }
    }
    use crate::bindings::Action;
    // Escape clears the selection only when the camera did not use it to
    // leave a transient mode (VIEWER_UI §12's order).
    if bindings.just_pressed(&gate, &keys, Action::Escape) && !escape.0 {
        select_key(&mut selection, &markers, None);
        ui_state.inspected_cell = None;
    } else if bindings.just_pressed(&gate, &keys, Action::NextSelection) {
        let back = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        let current = selection.selected.as_ref().map(|s| s.key);
        let next = cycle_selection(&markers, &visibility, current, !back);
        select_key(&mut selection, &markers, next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{ReplayDocument, ReplayState};
    use bevy::camera::primitives::{Aabb, MeshAabb};
    use bevy::camera::visibility::{InheritedVisibility, ViewVisibility};
    use bevy::ecs::system::RunSystemOnce;

    fn markers_for(frame: serde_json::Value) -> CurrentRuntimeMarkers {
        let document: ReplayDocument =
            serde_json::from_value(serde_json::json!({ "frames": [frame] })).unwrap();
        let replay = ReplayState::new(Some(document));
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(&replay);
        markers
    }

    /// A drone and a station at `x`, a track and a truth that share the id
    /// `"x"`: the fixture every test picks from.
    fn fixture() -> CurrentRuntimeMarkers {
        markers_for(serde_json::json!({
            "timestamp_s": 0.0,
            "tracks": [{"track_id": "x", "position": [0.0, 0.0, 50.0], "velocity": [0.0, 0.0, 0.0],
                        "measurement_std_m": 5.0, "update_count": 1, "stale_steps": 0}],
            "truths": [{"target_id": "x", "position": [0.0, 0.0, 30.0], "velocity": [0.0, 0.0, 0.0]}],
            "nodes": [
                {"node_id": "near", "position": [0.0, 0.0, 80.0], "velocity": [1.0, 0.0, 0.0], "is_mobile": true},
                {"node_id": "far", "position": [0.0, 0.0, 10.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": false}
            ],
        }))
    }

    fn key(markers: &CurrentRuntimeMarkers, kind: MarkerKind, id: &str) -> EntityKey {
        markers.key_of(kind, id).unwrap()
    }

    /// A world with a unit box per marker at its position (scaled by `size`),
    /// an optional ground slab, and the components `MeshRayCast` reads, set
    /// directly: this tests the cast and its filters, not Bevy's propagation.
    struct World3 {
        app: App,
        entities: Vec<(EntityKey, Entity)>,
    }

    fn world(
        markers: &CurrentRuntimeMarkers,
        ground_z: Option<f32>,
        hidden: &[EntityKey],
    ) -> World3 {
        // `MeshRayCast` culls in parallel.
        bevy::tasks::ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .insert_resource(fixture());
        let cube = Mesh::from(Cuboid::new(2.0, 2.0, 2.0));
        let aabb = cube.compute_aabb().unwrap();
        let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(cube);
        let mut entities = Vec::new();
        for marker in &markers.markers {
            let visible = !hidden.contains(&marker.key);
            let entity = app
                .world_mut()
                .spawn((
                    marker.key,
                    Mesh3d(handle.clone()),
                    aabb,
                    GlobalTransform::from(Transform::from_translation(marker.position)),
                    if visible {
                        InheritedVisibility::VISIBLE
                    } else {
                        InheritedVisibility::HIDDEN
                    },
                    ViewVisibility::default(),
                ))
                .id();
            entities.push((marker.key, entity));
        }
        if let Some(z) = ground_z {
            let slab = Mesh::from(Cuboid::new(1000.0, 1000.0, 1.0));
            let aabb: Aabb = slab.compute_aabb().unwrap();
            let slab = app.world_mut().resource_mut::<Assets<Mesh>>().add(slab);
            app.world_mut().spawn((
                PickOccluder,
                Mesh3d(slab),
                aabb,
                GlobalTransform::from(Transform::from_xyz(0.0, 0.0, z)),
                InheritedVisibility::VISIBLE,
                ViewVisibility::default(),
            ));
        }
        World3 { app, entities }
    }

    fn cast(
        world: &mut World3,
        visibility: RuntimeOverlayVisibility,
        ray: Ray3d,
    ) -> Option<EntityKey> {
        world.app.insert_resource(visibility);
        world
            .app
            .world_mut()
            .run_system_once(
                move |mut ray_cast: MeshRayCast,
                      keys: Query<&EntityKey>,
                      occluders: Query<(), With<PickOccluder>>,
                      markers: Res<CurrentRuntimeMarkers>,
                      visibility: Res<RuntimeOverlayVisibility>| {
                    pick_along(ray, &mut ray_cast, &keys, &occluders, &markers, &visibility)
                },
            )
            .unwrap()
    }

    fn all_shown() -> RuntimeOverlayVisibility {
        RuntimeOverlayVisibility {
            tracks: true,
            truths: true,
            nodes: true,
            ..Default::default()
        }
    }

    /// Straight down through the stack at the origin, from above.
    fn down() -> Ray3d {
        Ray3d::new(Vec3::new(0.0, 0.0, 500.0), Dir3::NEG_Z)
    }

    #[test]
    fn the_nearest_marker_on_the_ray_is_picked() {
        let markers = fixture();
        let mut w = world(&markers, None, &[]);
        assert_eq!(
            cast(&mut w, all_shown(), down()),
            Some(key(&markers, MarkerKind::Node, "near"))
        );
        // From below, the order reverses.
        let up = Ray3d::new(Vec3::new(0.0, 0.0, -500.0), Dir3::Z);
        assert_eq!(
            cast(&mut w, all_shown(), up),
            Some(key(&markers, MarkerKind::Node, "far"))
        );
    }

    #[test]
    fn an_occluder_in_front_blocks_the_pick() {
        let markers = fixture();
        // A slab between the camera and every marker.
        let mut w = world(&markers, Some(200.0), &[]);
        assert_eq!(cast(&mut w, all_shown(), down()), None);
        // A slab below them all blocks nothing from above.
        let mut w = world(&markers, Some(0.0), &[]);
        assert_eq!(
            cast(&mut w, all_shown(), down()),
            Some(key(&markers, MarkerKind::Node, "near"))
        );
    }

    /// A hidden marker is not there for the ray: the pick falls through to
    /// the next one, whether hidden by its kind's toggle or by its entity.
    #[test]
    fn hidden_markers_are_passed_through() {
        let markers = fixture();
        let near = key(&markers, MarkerKind::Node, "near");
        let track = key(&markers, MarkerKind::Track, "x");
        let truth = key(&markers, MarkerKind::Truth, "x");
        // Nodes off: the track at 50 m is next.
        let mut w = world(&markers, None, &[]);
        let no_nodes = RuntimeOverlayVisibility {
            nodes: false,
            ..all_shown()
        };
        assert_eq!(cast(&mut w, no_nodes, down()), Some(track));
        // Nodes off and tracks hidden as entities: the truth, by its own key.
        let mut w = world(&markers, None, &[track, near]);
        assert_eq!(cast(&mut w, all_shown(), down()), Some(truth));
        // Truths off too (the default): nothing but the far station.
        let defaults = RuntimeOverlayVisibility::default();
        assert!(!defaults.truths);
        let mut w = world(&markers, None, &[track, near]);
        assert_eq!(
            cast(&mut w, defaults, down()),
            Some(key(&markers, MarkerKind::Node, "far"))
        );
    }

    /// A base-layer mesh is tagged an occluder that blocks from both faces:
    /// a ground plane wound facing down, as the committed terrain is, still
    /// hides what is under it from a ray coming down.
    #[test]
    fn base_layer_meshes_occlude_from_either_face() {
        let markers = fixture();
        let mut w = world(&markers, None, &[]);
        let root = w.app.world_mut().spawn_empty().id();
        let mut layers = LayerEntityIndex::default();
        layers.entities_by_layer.insert("terrain".into(), root);
        // Wound facing down: its front face points away from a ray from above.
        let plane = Mesh::from(Plane3d::new(Vec3::NEG_Z, Vec2::splat(500.0)));
        let aabb = plane.compute_aabb().unwrap();
        let plane = w.app.world_mut().resource_mut::<Assets<Mesh>>().add(plane);
        let ground = w
            .app
            .world_mut()
            .spawn((
                Mesh3d(plane),
                aabb,
                GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 200.0)),
                InheritedVisibility::VISIBLE,
                ViewVisibility::default(),
                ChildOf(root),
            ))
            .id();
        w.app
            .insert_resource(layers)
            .init_resource::<LayerVisibilityState>()
            .init_resource::<PickGeneration>()
            .add_systems(Update, tag_pick_occluders_system);
        w.app.update();
        assert!(
            w.app.world().get::<PickOccluder>(ground).is_some(),
            "not tagged"
        );
        assert!(w.app.world().resource::<PickGeneration>().0 > 0);
        assert_eq!(
            cast(&mut w, all_shown(), down()),
            None,
            "the plane did not occlude"
        );
        // A mesh outside the base layers is not an occluder.
        let loose = w.app.world_mut().spawn(Mesh3d::default()).id();
        w.app.update();
        assert!(w.app.world().get::<PickOccluder>(loose).is_none());
    }

    /// Geometry that is not a marker or an occluder never takes a pick:
    /// a presentation-only volume in front of the drone is ignored.
    #[test]
    fn presentation_geometry_never_steals_a_pick() {
        let markers = fixture();
        let mut w = world(&markers, None, &[]);
        let big = Mesh::from(Sphere::new(100.0));
        let aabb = big.compute_aabb().unwrap();
        let big = w.app.world_mut().resource_mut::<Assets<Mesh>>().add(big);
        w.app.world_mut().spawn((
            crate::sensor_volumes::PresentationOnly,
            Mesh3d(big),
            aabb,
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 150.0)),
            InheritedVisibility::VISIBLE,
            ViewVisibility::default(),
        ));
        assert_eq!(
            cast(&mut w, all_shown(), down()),
            Some(key(&markers, MarkerKind::Node, "near"))
        );
    }

    /// A despawned marker cannot be picked, even with its key still live.
    #[test]
    fn a_despawned_marker_is_gone_from_the_ray() {
        let markers = fixture();
        let mut w = world(&markers, None, &[]);
        let near = key(&markers, MarkerKind::Node, "near");
        let entity = w.entities.iter().find(|(k, _)| *k == near).unwrap().1;
        w.app.world_mut().despawn(entity);
        assert_eq!(
            cast(&mut w, all_shown(), down()),
            Some(key(&markers, MarkerKind::Track, "x"))
        );
    }

    /// A track and a truth share the id `"x"`; each is selected as itself.
    #[test]
    fn same_label_different_kind_are_different_selections() {
        let markers = fixture();
        let track = key(&markers, MarkerKind::Track, "x");
        let truth = key(&markers, MarkerKind::Truth, "x");
        assert_ne!(track, truth);
        let mut selection = SelectionState::default();
        select_key(&mut selection, &markers, Some(truth));
        let selected = selection.selected.clone().unwrap();
        assert_eq!(selected.key, truth);
        assert!(matches!(selected.details, SelectionDetails::Truth(_)));
        select_key(&mut selection, &markers, Some(track));
        assert_eq!(selection.selected.as_ref().unwrap().key, track);
        assert!(matches!(
            selection.selected.as_ref().unwrap().details,
            SelectionDetails::Track(_)
        ));
    }

    #[test]
    fn tab_cycles_the_shown_markers_in_order_and_wraps() {
        let markers = fixture();
        let shown = RuntimeOverlayVisibility::default(); // truths hidden
        let order: Vec<EntityKey> = markers
            .markers
            .iter()
            .filter(|m| is_marker_visible(m, &shown))
            .map(|m| m.key)
            .collect();
        assert_eq!(order.len(), 3, "track and two nodes");
        let mut at = None;
        for expected in order.iter().chain(order.first()) {
            at = cycle_selection(&markers, &shown, at, true);
            assert_eq!(at, Some(*expected));
        }
        assert_eq!(
            cycle_selection(&markers, &shown, None, false),
            order.last().copied()
        );
        assert_eq!(
            cycle_selection(&markers, &shown, Some(order[0]), false),
            order.last().copied()
        );
        // A hidden current selection starts over from the first shown.
        let truth = key(&markers, MarkerKind::Truth, "x");
        assert_eq!(
            cycle_selection(&markers, &shown, Some(truth), true),
            Some(order[0])
        );
        let none = RuntimeOverlayVisibility {
            tracks: false,
            nodes: false,
            ..shown
        };
        assert_eq!(cycle_selection(&markers, &none, None, true), None);
    }

    /// A 640 x 480 camera at `eye`, looking straight down.
    fn camera(
        eye: Vec3,
        viewport: Option<(UVec2, UVec2)>,
    ) -> (MainCamera, Camera, GlobalTransform) {
        use bevy::camera::{ComputedCameraValues, RenderTargetInfo, Viewport};
        let size = viewport.map_or(UVec2::new(640, 480), |(_, size)| size);
        let camera = Camera {
            viewport: viewport.map(|(position, size)| Viewport {
                physical_position: position,
                physical_size: size,
                ..default()
            }),
            computed: ComputedCameraValues {
                clip_from_view: Mat4::perspective_infinite_reverse_rh(
                    std::f32::consts::FRAC_PI_4,
                    size.x as f32 / size.y as f32,
                    0.1,
                ),
                target_info: Some(RenderTargetInfo {
                    physical_size: UVec2::new(640, 480),
                    scale_factor: 1.0,
                }),
                ..default()
            },
            ..default()
        };
        let transform = Transform::from_translation(eye).looking_at(eye.with_z(0.0), Vec3::Y);
        (MainCamera, camera, GlobalTransform::from(transform))
    }

    /// The fixture's world, a camera over it, and the picker and keyboard
    /// systems with every resource they read.
    fn pick_app(viewport: Option<(UVec2, UVec2)>) -> (App, Entity) {
        let markers = fixture();
        let mut w = world(&markers, None, &[]);
        let cam = w
            .app
            .world_mut()
            .spawn(camera(Vec3::new(0.0, 0.0, 500.0), viewport))
            .id();
        w.app
            .insert_resource(all_shown())
            .init_resource::<PointerInput>()
            .init_resource::<PickGeneration>()
            .init_resource::<HoverState>()
            .init_resource::<SelectionState>()
            .init_resource::<PickLog>()
            .init_resource::<ViewerUiState>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<crate::bindings::Bindings>()
            .init_resource::<crate::bindings::KeyboardGate>()
            .init_resource::<crate::camera_control::EscapeUsed>()
            .add_systems(
                Update,
                (pick_world_system, keyboard_selection_system).chain(),
            );
        (w.app, cam)
    }

    fn point(app: &mut App, at: Option<Vec2>, pressed: bool, over_ui: bool) {
        *app.world_mut().resource_mut::<PointerInput>() = PointerInput {
            position: at,
            over_ui,
            pressed,
        };
        app.update();
    }

    fn hovered(app: &App) -> Option<EntityKey> {
        app.world().resource::<HoverState>().hovered()
    }

    fn selected(app: &App) -> Option<EntityKey> {
        app.world()
            .resource::<SelectionState>()
            .selected
            .as_ref()
            .map(|s| s.key)
    }

    const CENTRE: Vec2 = Vec2::new(320.0, 240.0);

    /// Pointer to selection, through the camera: hover follows the pointer
    /// and the camera, is cast only when an input moved, and a click selects
    /// what is hovered or clears on empty world.
    #[test]
    fn hover_and_click_follow_the_pointer_and_the_camera() {
        let (mut app, cam) = pick_app(None);
        let near = key(&fixture(), MarkerKind::Node, "near");
        point(&mut app, Some(CENTRE), false, false);
        assert_eq!(hovered(&app), Some(near));
        let casts = app.world().resource::<HoverState>().casts();
        // Nothing moved: nothing cast.
        app.update();
        app.update();
        assert_eq!(app.world().resource::<HoverState>().casts(), casts);
        // A click selects the hovered marker.
        point(&mut app, Some(CENTRE), true, false);
        assert_eq!(selected(&app), Some(near));
        // The camera moves off the stack: the same pointer hovers nothing,
        // and a click there clears.
        *app.world_mut().get_mut::<GlobalTransform>(cam).unwrap() = GlobalTransform::from(
            Transform::from_xyz(2000.0, 0.0, 500.0)
                .looking_at(Vec3::new(2000.0, 0.0, 0.0), Vec3::Y),
        );
        app.update();
        assert_eq!(hovered(&app), None, "hover did not follow the camera");
        assert!(app.world().resource::<HoverState>().casts() > casts);
        point(&mut app, Some(CENTRE), true, false);
        assert_eq!(selected(&app), None);
        let log = app.world().resource::<PickLog>();
        assert_eq!((log.clicks, log.last_ignored), (2, false));
    }

    /// A pointer the UI owns reaches nothing: no hover, and a click there is
    /// consumed without touching the selection.
    #[test]
    fn the_ui_keeps_the_pointer_from_the_world() {
        let (mut app, _) = pick_app(None);
        let near = key(&fixture(), MarkerKind::Node, "near");
        point(&mut app, Some(CENTRE), true, false);
        assert_eq!(selected(&app), Some(near));
        point(&mut app, Some(CENTRE), true, true);
        assert_eq!(hovered(&app), None);
        assert_eq!(
            selected(&app),
            Some(near),
            "a click on the UI changed the selection"
        );
        assert!(app.world().resource::<PickLog>().last_ignored);
    }

    /// With the camera in the left half of the window, its own centre picks
    /// through it and the right half is not its to pick.
    #[test]
    fn a_split_viewport_picks_only_inside_itself() {
        let (mut app, _) = pick_app(Some((UVec2::ZERO, UVec2::new(320, 480))));
        let near = key(&fixture(), MarkerKind::Node, "near");
        point(&mut app, Some(Vec2::new(160.0, 240.0)), true, false);
        assert_eq!(
            selected(&app),
            Some(near),
            "the viewport's centre did not pick"
        );
        point(&mut app, Some(CENTRE + Vec2::new(160.0, 0.0)), true, false);
        assert!(app.world().resource::<PickLog>().last_ignored);
        assert_eq!(selected(&app), Some(near));
        assert_eq!(hovered(&app), None);
    }

    /// Tab steps, Shift+Tab steps back, Escape clears, and the entity list's
    /// requests select by key.
    #[test]
    fn keyboard_and_list_select_without_the_pointer() {
        let (mut app, _) = pick_app(None);
        let markers = fixture();
        let order: Vec<EntityKey> = markers.markers.iter().map(|m| m.key).collect();
        let press = |app: &mut App, key: KeyCode, shift: bool| {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.reset_all();
            if shift {
                keys.press(KeyCode::ShiftLeft);
            }
            keys.press(key);
            app.update();
            app.world_mut()
                .resource_mut::<ButtonInput<KeyCode>>()
                .clear();
        };
        press(&mut app, KeyCode::Tab, false);
        assert_eq!(selected(&app), Some(order[0]));
        press(&mut app, KeyCode::Tab, false);
        assert_eq!(selected(&app), Some(order[1]));
        press(&mut app, KeyCode::Tab, true);
        assert_eq!(selected(&app), Some(order[0]));
        // An Escape the camera used (leaving free-fly, follow or chase)
        // keeps the selection; the next one clears it.
        app.world_mut()
            .resource_mut::<crate::camera_control::EscapeUsed>()
            .0 = true;
        press(&mut app, KeyCode::Escape, false);
        assert_eq!(
            selected(&app),
            Some(order[0]),
            "the camera's Escape cleared the selection"
        );
        app.world_mut()
            .resource_mut::<crate::camera_control::EscapeUsed>()
            .0 = false;
        press(&mut app, KeyCode::Escape, false);
        assert_eq!(selected(&app), None);
        let truth = key(&markers, MarkerKind::Truth, "x");
        app.world_mut()
            .resource_mut::<ViewerUiState>()
            .select_request = Some(SelectRequest::Select(truth));
        app.update();
        assert_eq!(selected(&app), Some(truth));
        assert_eq!(
            app.world().resource::<ViewerUiState>().select_request,
            None,
            "the request was not consumed"
        );
        app.world_mut()
            .resource_mut::<ViewerUiState>()
            .select_request = Some(SelectRequest::Clear);
        app.update();
        assert_eq!(selected(&app), None);
    }

    /// Two viewports sharing an edge never both claim a pixel.
    #[test]
    fn split_viewports_do_not_overlap() {
        let left = Rect::new(0.0, 0.0, 325.0, 480.0);
        let right = Rect::new(325.0, 0.0, 650.0, 480.0);
        for x in [0.0, 100.0, 324.9, 325.0, 400.0, 649.9] {
            let p = Vec2::new(x, 240.0);
            assert!(
                !(viewport_contains(left, p) && viewport_contains(right, p)),
                "{x}"
            );
            assert!(
                viewport_contains(left, p) || viewport_contains(right, p),
                "{x}"
            );
        }
        assert!(!viewport_contains(left, Vec2::new(325.0, 10.0)));
        assert!(!viewport_contains(left, Vec2::new(10.0, 480.0)));
    }
}
