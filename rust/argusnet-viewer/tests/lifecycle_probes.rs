//! Resource growth across the presentation lifecycle (C8, P2.6;
//! VERIFICATION_AND_ROLLOUT.md §5): the real headless app, after its still,
//! runs rotations of an epoch reload of the same run, a seek to the start,
//! layer toggles off and on, a quality-tier step, a Section round trip and a
//! seek to the end, and records what the world holds after each (on the last
//! frame, trails full). Monotonic growth is a failure: once every tier has
//! been visited, each rotation must leave exactly what the rotation one
//! tier-cycle earlier left.
//!
//! Needs a GPU (or lavapipe): runs only with `ARGUSNET_GPU_TESTS=1`.

use std::path::{Path, PathBuf};

use argusnet_viewer::headless::{CameraPreset, HeadlessRenderOptions};
use argusnet_viewer::quality::QualityTier;
use argusnet_viewer::{render_headless, ViewMode};

fn gpu_tests() -> bool {
    std::env::var("ARGUSNET_GPU_TESTS").is_ok_and(|v| v == "1")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("argusnet-lifecycle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The committed scene with a moving replay: drones, tracks, truths, a radar
/// station and a zone, so markers, trails, sensor volumes, rings and walls
/// all exist to be toggled and reloaded.
fn scene() -> PathBuf {
    let root = scratch("scene");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
        &root,
    );
    let frames: Vec<serde_json::Value> = (0..40)
        .map(|f| {
            let t = f as f64 * 0.5;
            let at = |i: usize| serde_json::json!([-900.0 + i as f64 * 300.0 + 10.0 * t, -600.0, 300.0]);
            serde_json::json!({
                "timestamp_s": t,
                "nodes": (0..6).map(|i| serde_json::json!({
                    "node_id": format!("drone-{i}"), "position": at(i), "velocity": [10.0, 0.0, 0.0],
                    "is_mobile": true, "sensor_type": "optical", "fov_half_angle_deg": 30.0,
                    "max_range_m": 600.0})).chain(std::iter::once(serde_json::json!({
                    "node_id": "station-1", "position": [800.0, 900.0, 5.0], "velocity": [0.0, 0.0, 0.0],
                    "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0,
                    "max_range_m": 1200.0}))).collect::<Vec<_>>(),
                "tracks": (0..6).map(|i| serde_json::json!({
                    "track_id": format!("asset-{i}"), "position": at(i), "velocity": [10.0, 0.0, 0.0],
                    "measurement_std_m": 5.0, "update_count": 1, "stale_steps": 0})).collect::<Vec<_>>(),
                "truths": (0..6).map(|i| serde_json::json!({
                    "target_id": format!("asset-{i}"), "position": at(i),
                    "velocity": [10.0, 0.0, 0.0]})).collect::<Vec<_>>(),
            })
        })
        .collect();
    let replay = serde_json::json!({
        "meta": {"dt_s": 0.5, "zones": [
            {"zone_id": "keep-out", "zone_type": "exclusion", "center": [-800.0, 1000.0, 5.0], "radius_m": 350.0}
        ]},
        "frames": frames,
    });
    std::fs::create_dir_all(root.join("replay")).unwrap();
    std::fs::write(root.join("replay/replay.json"), replay.to_string()).unwrap();
    let manifest_path = root.join("scene_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["replay"] = serde_json::json!({"path": "replay/replay.json"});
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();
    root
}

const ROTATIONS: u32 = 12;
/// Rotations per full cycle of the quality tiers (one step a rotation).
const TIER_CYCLE: usize = 3;

#[test]
fn repeated_seek_reload_toggle_and_tier_changes_do_not_grow_the_world() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene();
    let out = scratch("out");
    let still = out.join("lifecycle.png");
    render_headless(
        &scene,
        HeadlessRenderOptions {
            output: Some(still.clone()),
            record_dir: None,
            camera: CameraPreset::Isometric,
            target_id: None,
            width: 320,
            height: 240,
            fps: 30,
            view_mode: ViewMode::RealWorld,
            probe: None,
            quality: QualityTier::default(),
            frame_stats: 0,
            lifecycle: ROTATIONS,
        },
    )
    .expect("render");
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(still.with_extension("lifecycle.json"))
            .expect("lifecycle written"),
    )
    .unwrap();
    let rotations = report["rotations"].as_array().unwrap();
    assert_eq!(rotations.len(), ROTATIONS as usize);
    for rotation in rotations {
        eprintln!("{rotation}");
    }
    let counted = |r: &serde_json::Value| {
        let mut r = r.clone();
        r.as_object_mut().unwrap().remove("rotation");
        r
    };
    // Something was presented and counted.
    assert!(rotations[0]["entities"].as_u64().unwrap() > 0);
    assert!(rotations[0]["trail_ribbons"].as_u64().unwrap() > 0);
    // After the first full tier cycle, each rotation leaves what the one a
    // tier cycle earlier left: nothing accumulates.
    for r in 2 * TIER_CYCLE..rotations.len() {
        assert_eq!(
            counted(&rotations[r]),
            counted(&rotations[r - TIER_CYCLE]),
            "rotation {} grew against rotation {}",
            r + 1,
            r + 1 - TIER_CYCLE
        );
    }
}
