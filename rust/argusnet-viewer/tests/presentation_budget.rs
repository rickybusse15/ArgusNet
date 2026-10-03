//! Per-frame presentation cost, counted in allocations (P2.0).
//!
//! A counting global allocator wraps the system one. A synthetic replay (80
//! platforms, tracks and truths over 240 frames, a quarter second apart) is
//! played back at 60 rendered frames a second through the snapshot adapter,
//! the marker entities and a reader of every trail (what the overlays do),
//! and the allocations per rendered frame are counted after a warm-up.
//!
//! Before P2.0 the same work (the marker list rebuilt from the frame, the
//! trails rescanned, the entities resynced) measured **1,108 allocations a
//! frame playing and 1,210 paused** on this harness. The bounds below hold
//! the new path to a small fraction of that, and to none at all when paused.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use argusnet_viewer::app::sync_current_markers_system;
use argusnet_viewer::marker_meshes::{
    setup_assets, sync_marker_meshes, HologramMaterial, MarkerEntityIndex,
};
use argusnet_viewer::presentation::TRAIL_MAX_FRAMES;
use argusnet_viewer::replay::{ReplayDocument, ReplayState};
use argusnet_viewer::schema::ScenePackage;
use argusnet_viewer::state::{
    CurrentFrameMetrics, CurrentRuntimeMarkers, RuntimeOverlayVisibility, SelectionState,
};
use bevy::prelude::*;

struct Counting;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocator is process-wide: tests that count must not overlap.
static COUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

const FRAMES: usize = 240;
const PER_KIND: usize = 80;

fn document() -> ReplayDocument {
    let frames: Vec<serde_json::Value> = (0..FRAMES)
        .map(|f| {
            let t = f as f64 * 0.25;
            let at = |i: usize| serde_json::json!([i as f64 * 10.0 + t, t, 100.0]);
            serde_json::json!({
                "timestamp_s": t,
                "nodes": (0..PER_KIND).map(|i| serde_json::json!({
                    "node_id": format!("drone-{i:03}"), "position": at(i),
                    "velocity": [4.0, 0.0, 0.0], "is_mobile": true})).collect::<Vec<_>>(),
                "tracks": (0..PER_KIND).map(|i| serde_json::json!({
                    "track_id": format!("asset-{i:03}"), "position": at(i),
                    "velocity": [4.0, 0.0, 0.0]})).collect::<Vec<_>>(),
                "truths": (0..PER_KIND).map(|i| serde_json::json!({
                    "target_id": format!("asset-{i:03}"), "position": at(i),
                    "velocity": [4.0, 0.0, 0.0]})).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::from_value(serde_json::json!({ "frames": frames })).expect("synthetic replay")
}

fn app() -> App {
    let mut app = App::new();
    let mut replay = ReplayState::new(Some(document()));
    replay.playing = true;
    app.init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<HologramMaterial>>()
        .insert_resource(replay)
        .insert_resource(
            ScenePackage::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
            )
            .expect("scene package"),
        )
        .insert_resource(RuntimeOverlayVisibility::default())
        .insert_resource(CurrentRuntimeMarkers::default())
        .insert_resource(CurrentFrameMetrics::default())
        .insert_resource(SelectionState::default())
        .init_resource::<MarkerEntityIndex>()
        .add_systems(Startup, setup_assets)
        .add_systems(
            Update,
            (sync_current_markers_system, sync_marker_meshes, read_trails).chain(),
        );
    app
}

/// What the overlays do with trails: walk every marker's.
fn read_trails(markers: Res<CurrentRuntimeMarkers>) {
    for index in 0..markers.markers.len() {
        std::hint::black_box(markers.trail(index, TRAIL_MAX_FRAMES).count());
    }
}

/// Allocations per rendered frame over `frames` frames, each advancing
/// playback by `dt_s`, after a warm-up.
fn allocations_per_frame(app: &mut App, dt_s: f32, frames: u32) -> f64 {
    for _ in 0..30 {
        app.world_mut().resource_mut::<ReplayState>().advance(dt_s);
        app.update();
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..frames {
        app.world_mut().resource_mut::<ReplayState>().advance(dt_s);
        app.update();
    }
    (ALLOCATIONS.load(Ordering::Relaxed) - before) as f64 / f64::from(frames)
}

/// The same app with three systems that do nothing: what Bevy's scheduler
/// allocates per update on its own (about four, plus two per system).
fn skeleton() -> App {
    let mut app = App::new();
    let mut replay = ReplayState::new(Some(document()));
    replay.playing = true;
    app.insert_resource(replay)
        .add_systems(Update, ((|| {}), (|| {}), (|| {})).chain());
    app
}

#[test]
fn presenting_a_frame_costs_almost_nothing_and_a_paused_one_nothing() {
    let _serial = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let mut bare = skeleton();
    let net = |app: &mut App, bare: &mut App, dt_s: f32| {
        allocations_per_frame(app, dt_s, 240) - allocations_per_frame(bare, dt_s, 240)
    };
    let playing = net(&mut app, &mut bare, 1.0 / 60.0);
    for world in [app.world_mut(), bare.world_mut()] {
        world.resource_mut::<ReplayState>().playing = false;
    }
    let paused = net(&mut app, &mut bare, 0.0);
    println!("P2.0 path, net of the scheduler: {playing:.2} allocations/frame playing, {paused:.2} paused");
    // Paused: the presentation code allocates nothing. What remains, about
    // two, is Bevy fetching the marker system's parameters: returning at the
    // very top of `sync_marker_meshes` measured the same 2.03. Before: 1,210.
    assert!(paused < 3.0, "a paused frame allocated {paused}");
    // Playing: that, plus the membership rebuild when the source frame
    // changes (every 15 rendered frames here). Before: 1,108.
    assert!(playing < 8.0, "{playing} allocations/frame playing");
    assert_eq!(
        app.world().resource::<MarkerEntityIndex>().len(),
        3 * PER_KIND,
        "one entity per presented marker"
    );
}

/// The same markers, plus the main camera and everything its behaviour
/// reads (P2.4b). With `camera` false the camera system is left out, so the
/// difference is what it costs.
fn camera_app(camera: bool) -> App {
    use argusnet_viewer::bindings::{Bindings, KeyboardGate};
    use argusnet_viewer::camera_control::{
        drive_main_camera_system, CameraControl, EscapeUsed, OrbitPose,
    };
    use argusnet_viewer::orbit_camera::OrbitCamera;
    use argusnet_viewer::picking::PointerInput;
    use argusnet_viewer::prefs::ViewerPrefs;
    use argusnet_viewer::state::LoadedMissionZones;
    use bevy::input::mouse::{MouseMotion, MouseWheel};

    let mut app = App::new();
    let replay = ReplayState::new(Some(document()));
    let orbit = OrbitCamera {
        focus: Vec3::ZERO,
        radius: 3000.0,
        yaw: 0.8,
        pitch: 1.0,
        min_radius: 40.0,
        max_radius: 20_000.0,
    };
    app.insert_resource(replay)
        .insert_resource(
            ScenePackage::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
            )
            .expect("scene package"),
        )
        .insert_resource(CurrentRuntimeMarkers::default())
        .insert_resource(CurrentFrameMetrics::default())
        .insert_resource(SelectionState::default())
        .insert_resource(LoadedMissionZones::default())
        .insert_resource(Time::<bevy::time::Real>::default())
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<ButtonInput<MouseButton>>()
        .init_resource::<KeyboardGate>()
        .init_resource::<Bindings>()
        .init_resource::<PointerInput>()
        .init_resource::<ViewerPrefs>()
        .init_resource::<EscapeUsed>()
        .insert_resource(CameraControl::new(OrbitPose::of(&orbit)))
        .add_message::<MouseMotion>()
        .add_message::<MouseWheel>();
    app.world_mut().spawn((
        argusnet_viewer::app::MainCamera,
        orbit,
        Transform::default(),
        Camera::default(),
    ));
    if camera {
        app.add_systems(
            Update,
            (sync_current_markers_system, drive_main_camera_system).chain(),
        );
    } else {
        app.add_systems(Update, (sync_current_markers_system, || {}).chain());
    }
    app
}

/// One press of `key` for one frame, as Bevy's input systems would do it:
/// down, a frame, then this frame's presses cleared and the key up.
fn tap(app: &mut App, key: KeyCode) {
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(key);
    app.update();
    let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
    keys.clear();
    keys.release(key);
    keys.clear();
}

/// The camera costs nothing a frame when idle, and nothing while following
/// an entity: its actions are a constant list and the followed label is
/// shared, not copied.
#[test]
fn the_camera_allocates_nothing_idle_or_following() {
    use argusnet_viewer::bindings::Action;
    use argusnet_viewer::camera_control::{CameraControl, CameraMode};
    let _serial = COUNTING.lock().unwrap_or_else(|e| e.into_inner());

    let mut with = camera_app(true);
    let mut without = camera_app(false);
    let net = |with: &mut App, without: &mut App| {
        allocations_per_frame(with, 0.0, 240) - allocations_per_frame(without, 0.0, 240)
    };
    let idle = net(&mut with, &mut without);

    // Select a drone and press L: the camera follows it.
    let key = {
        let markers = with.world().resource::<CurrentRuntimeMarkers>();
        let marker = &markers.markers[0];
        (marker.key, markers.epoch(), marker.clone())
    };
    // Both apps select it, so only the camera differs.
    for app in [&mut with, &mut without] {
        app.world_mut().resource_mut::<SelectionState>().selected =
            Some(argusnet_viewer::picking::selection_of(&key.2, key.1));
    }
    let follow = with
        .world()
        .resource::<argusnet_viewer::bindings::Bindings>()
        .key(Action::Follow);
    tap(&mut with, follow);
    assert_eq!(
        with.world().resource::<CameraControl>().mode,
        CameraMode::Follow(key.0)
    );
    let following = net(&mut with, &mut without);

    // Free-fly, forward held the whole time: keys held every frame.
    let bindings = with
        .world()
        .resource::<argusnet_viewer::bindings::Bindings>()
        .clone();
    tap(&mut with, bindings.key(Action::Escape));
    tap(&mut with, bindings.key(Action::FreeFly));
    assert_eq!(
        with.world().resource::<CameraControl>().mode,
        CameraMode::FreeFly
    );
    with.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(bindings.key(Action::MoveForward));
    let flying = net(&mut with, &mut without);
    println!(
        "P2.4b camera, net: {idle:.2} allocations/frame idle, {following:.2} following, {flying:.2} flying"
    );
    assert!(idle < 0.5, "an idle camera allocated {idle} a frame");
    assert!(
        following < 0.5,
        "a following camera allocated {following} a frame"
    );
    assert!(flying < 0.5, "a flying camera allocated {flying} a frame");
}
