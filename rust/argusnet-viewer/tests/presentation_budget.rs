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
