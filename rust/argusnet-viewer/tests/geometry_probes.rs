//! Pixel probes (P2.2): sensor volumes, zone walls and trail ribbons, read
//! back from real headless renders.
//!
//! Each case renders the same scene twice, with its overlay on and off, and
//! compares the pixels at world points: where the geometry is, they must
//! differ; well away from it, they must not. Needs a GPU (or lavapipe), so it
//! runs only with `ARGUSNET_GPU_TESTS=1`.

use std::path::{Path, PathBuf};

use argusnet_sensing::look::{sensor_look_direction, DEFAULT_LOOK_DOWN_ANGLE_DEG};
use argusnet_viewer::headless::{CameraPreset, HeadlessRenderOptions, ProbeRequest};
use argusnet_viewer::state::RuntimeOverlayVisibility;
use argusnet_viewer::{render_headless, ViewMode};

fn gpu_tests() -> bool {
    std::env::var("ARGUSNET_GPU_TESTS").is_ok_and(|v| v == "1")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argusnet-geometry-{name}-{}", std::process::id()));
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

const DT: f64 = 0.5;
const FRAMES: usize = 60;
const DRONE_SPEED: f64 = 20.0;

/// The drone's position at frame `f`: east along y = -600, clear of the
/// buildings at the centre, 300 m up.
fn drone_at(f: usize) -> [f64; 3] {
    [-900.0 + DRONE_SPEED * DT * f as f64, -600.0, 300.0]
}

/// The committed scene with a replay: one optical drone flying east, one
/// radar station with the given range, and one exclusion zone. With
/// `flat_terrain_m`, the replay also carries a flat terrain viewer mesh at
/// that height, as a simulated replay carries the real one.
fn scene(name: &str, station_range_m: f64, flat_terrain_m: Option<f64>) -> PathBuf {
    let root = scratch(name);
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
        &root,
    );
    let frames: Vec<serde_json::Value> = (0..FRAMES)
        .map(|f| {
            serde_json::json!({
                "timestamp_s": f as f64 * DT,
                "tracks": [], "truths": [],
                "nodes": [
                    {"node_id": "drone-1", "position": drone_at(f), "velocity": [DRONE_SPEED, 0.0, 0.0],
                     "is_mobile": true, "sensor_type": "optical", "fov_half_angle_deg": 30.0, "max_range_m": 900.0},
                    {"node_id": "station-1", "position": [800.0, 900.0, 5.0], "velocity": [0.0, 0.0, 0.0],
                     "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0,
                     "max_range_m": station_range_m}
                ],
            })
        })
        .collect();
    let mut replay = serde_json::json!({
        "meta": {"dt_s": DT, "zones": [
            {"zone_id": "keep-out", "zone_type": "exclusion", "center": [-800.0, 1000.0, 5.0], "radius_m": 350.0}
        ]},
        "frames": frames,
    });
    if let Some(z) = flat_terrain_m {
        replay["meta"]["terrain"] = serde_json::json!({"viewer_mesh": {
            "x_min_m": -2160.0, "x_max_m": 2160.0, "y_min_m": -1980.0, "y_max_m": 2160.0,
            "rows": 2, "cols": 2, "heights_m": [[z, z], [z, z]],
        }});
    }
    std::fs::create_dir_all(root.join("replay")).unwrap();
    std::fs::write(root.join("replay/replay.json"), replay.to_string()).unwrap();
    let manifest_path = root.join("scene_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["replay"] = serde_json::json!({"path": "replay/replay.json"});
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();
    root
}

/// Mean colour of the 9 x 9 window at each point: first the `ground`
/// points (on the terrain), then the elevated `points`.
fn render(
    name: &str,
    scene: &Path,
    camera: CameraPreset,
    target_id: Option<&str>,
    overlays: RuntimeOverlayVisibility,
    ground: &[[f64; 2]],
    points: &[[f64; 3]],
) -> Vec<[f32; 3]> {
    let dir = scratch(&format!("{name}-out"));
    let probes = dir.join("probes.json");
    render_headless(
        scene,
        HeadlessRenderOptions {
            output: Some(dir.join(format!("{name}.png"))),
            record_dir: None,
            camera,
            target_id: target_id.map(str::to_string),
            width: 650,
            height: 480,
            fps: 30,
            view_mode: ViewMode::RealWorld,
            quality: Default::default(),
            probe: Some(ProbeRequest {
                belief: None,
                belief_overlay: false,
                late_patches: Vec::new(),
                overlays: Some(overlays),
                points: ground.to_vec(),
                elevated: points.to_vec(),
                output: probes.clone(),
                ..Default::default()
            }),
        },
    )
    .expect("render");
    if let Ok(keep) = std::env::var("ARGUSNET_PROBE_KEEP") {
        let _ = std::fs::create_dir_all(&keep);
        std::fs::copy(
            dir.join(format!("{name}.png")),
            Path::new(&keep).join(format!("{name}.png")),
        )
        .unwrap();
    }
    let document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(probes).unwrap()).unwrap();
    document["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            let window = point["window"].as_array().unwrap();
            assert_eq!(window.len(), 81, "probe off the frame: {point}");
            let n = window.len() as f32;
            [0, 1, 2].map(|k| {
                window
                    .iter()
                    .map(|p| p.as_array().unwrap()[k].as_f64().unwrap() as f32 / 255.0)
                    .sum::<f32>()
                    / n
            })
        })
        .collect()
}

fn difference(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|k| (a[k] - b[k]).abs()).sum()
}

fn overlays(fov: bool, zones: bool, trails: bool) -> RuntimeOverlayVisibility {
    RuntimeOverlayVisibility {
        fov_cones: fov,
        zones,
        trails,
        tracks: false,
        truths: false,
        nodes: false,
        radar_rings: false,
        inspection_events: false,
        launch_lines: false,
        ..Default::default()
    }
}

/// The field-of-view sector is drawn where the runtime's look puts it, and
/// nowhere else; the zone wall and the trail are drawn where they belong.
#[test]
fn volumes_walls_and_trails_are_drawn_where_the_runtime_says() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("main", 400.0, None);
    let last = drone_at(FRAMES - 1);
    let look = sensor_look_direction([DRONE_SPEED, 0.0, 0.0], DEFAULT_LOOK_DOWN_ANGLE_DEG).unwrap();
    // Halfway along the axis, at the axis's own height.
    let on_axis = [0, 1, 2].map(|k| last[k] + 450.0 * look[k]);
    // Behind the drone and off to the side, on the ground: outside the sector.
    let behind = [last[0] - 700.0, last[1] - 900.0, 5.0];
    // Halfway up the exclusion wall's southern face, and the trail 20 s back.
    let wall = [-800.0, 1000.0 - 350.0, 45.0];
    let trail = drone_at(FRAMES - 41);
    // The wall's northern and eastern faces: it goes all the way round.
    let wall_north = [-800.0, 1000.0 + 350.0, 45.0];
    let wall_east = [-800.0 + 350.0, 1000.0, 45.0];
    let points = [on_axis, behind, wall, trail, wall_north, wall_east];

    let off = render(
        "off",
        &scene,
        CameraPreset::Isometric,
        None,
        overlays(false, false, false),
        &[],
        &points,
    );
    let fov = render(
        "fov",
        &scene,
        CameraPreset::Isometric,
        None,
        overlays(true, false, false),
        &[],
        &points,
    );
    let zones = render(
        "zones",
        &scene,
        CameraPreset::Isometric,
        None,
        overlays(false, true, false),
        &[],
        &points,
    );
    let trails = render(
        "trails",
        &scene,
        CameraPreset::Isometric,
        None,
        overlays(false, false, true),
        &[],
        &points,
    );
    for (label, i) in [
        ("on-axis", 0),
        ("behind", 1),
        ("wall", 2),
        ("trail", 3),
        ("wall-n", 4),
        ("wall-e", 5),
    ] {
        eprintln!(
            "{label:>8}: off {:?} fov {:.3} zones {:.3} trails {:.3}",
            off[i],
            difference(fov[i], off[i]),
            difference(zones[i], off[i]),
            difference(trails[i], off[i])
        );
    }
    assert!(
        difference(fov[0], off[0]) > 0.03,
        "the sector is not drawn on its axis"
    );
    assert!(fov[0][0] >= off[0][0], "an additive volume only adds light");
    assert!(
        difference(fov[1], off[1]) < 0.01,
        "the sector leaks behind the drone"
    );
    assert!(
        difference(zones[2], off[2]) > 0.03,
        "the exclusion wall is missing"
    );
    assert!(zones[2][0] > zones[2][1], "the exclusion wall is red");
    for (i, face) in [(4, "north"), (5, "east")] {
        assert!(
            difference(zones[i], off[i]) > 0.03,
            "the wall's {face} face is missing"
        );
    }
    assert!(
        difference(zones[1], off[1]) < 0.01,
        "a wall where there is no zone"
    );
    assert!(difference(trails[3], off[3]) > 0.02, "the trail is missing");
    assert!(
        difference(trails[1], off[1]) < 0.01,
        "a trail where the drone never flew"
    );
}

/// A volume that encloses the camera is not drawn: a view from inside a
/// radar's range sphere looks exactly like the view without it.
#[test]
fn no_volume_washes_out_a_view_from_inside_it() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    // A range that swallows the whole scene, camera included.
    let scene = scene("inside", 20_000.0, None);
    let points = [
        [800.0, 900.0, 5.0],
        [650.0, 900.0, 5.0],
        [800.0, 750.0, 40.0],
    ];
    let off = render(
        "inside-off",
        &scene,
        CameraPreset::FollowTarget,
        Some("station-1"),
        overlays(false, false, false),
        &[],
        &points,
    );
    let on = render(
        "inside-on",
        &scene,
        CameraPreset::FollowTarget,
        Some("station-1"),
        overlays(true, false, false),
        &[],
        &points,
    );
    for (a, b) in off.iter().zip(&on) {
        assert!(
            difference(*a, *b) < 0.01,
            "the enclosing volume changed the view: {a:?} -> {b:?}"
        );
    }
}

/// A radar's range ring is drawn on the ground at its range, only with its
/// toggle on.
#[test]
fn a_range_ring_is_drawn_at_the_range() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    // The ring drapes on the replay's terrain mesh. This one is flat, above
    // the committed terrain's 18 m peak, so the ring is in view wherever it
    // is; the draping itself is unit-tested.
    let scene = scene("rings", 400.0, Some(25.0));
    // On the ring, south and east of the station; and at the station itself.
    let ground = [[800.0, 500.0], [1200.0, 900.0], [800.0, 900.0]];
    let rings = RuntimeOverlayVisibility {
        radar_rings: true,
        ..overlays(false, false, false)
    };
    let off = render(
        "rings-off",
        &scene,
        CameraPreset::TopDown,
        None,
        overlays(false, false, false),
        &ground,
        &[],
    );
    let on = render(
        "rings-on",
        &scene,
        CameraPreset::TopDown,
        None,
        rings,
        &ground,
        &[],
    );
    for (label, i) in [("south", 0), ("east", 1), ("centre", 2)] {
        eprintln!(
            "{label:>7}: off {:?} on {:?} diff {:.4}",
            off[i],
            on[i],
            difference(on[i], off[i])
        );
    }
    for (i, side) in [(0, "south"), (1, "east")] {
        assert!(difference(on[i], off[i]) > 0.02, "no ring on the {side}");
        assert!(on[i][2] > off[i][2], "the {side} ring is not blue");
    }
    assert!(
        difference(on[2], off[2]) < 0.001,
        "the ring is drawn at the station"
    );
}

/// The zone outline gizmo is a diagnostic now that the wall draws the
/// boundary: it appears with the diagnostic glyphs, and only then.
#[test]
fn the_zone_outline_is_a_diagnostic() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("outline", 400.0, None);
    // On the exclusion zone's boundary, at the wall's foot.
    let ground = [
        [-800.0, 1000.0 - 350.0],
        [-800.0 - 350.0, 1000.0],
        [-800.0 + 350.0, 1000.0],
    ];
    let walls = overlays(false, true, false);
    let diagnostic = RuntimeOverlayVisibility {
        diagnostic_glyphs: true,
        ..walls
    };
    let plain = render(
        "outline-off",
        &scene,
        CameraPreset::Isometric,
        None,
        walls,
        &ground,
        &[],
    );
    let with = render(
        "outline-on",
        &scene,
        CameraPreset::Isometric,
        None,
        diagnostic,
        &ground,
        &[],
    );
    let diffs: Vec<f32> = plain
        .iter()
        .zip(&with)
        .map(|(a, b)| difference(*a, *b))
        .collect();
    eprintln!("outline diffs {diffs:?}");
    assert!(
        diffs.iter().any(|d| *d > 0.01),
        "the outline did not appear with the diagnostic glyphs: {diffs:?}"
    );
}
