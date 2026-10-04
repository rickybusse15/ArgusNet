//! Picking and selection probes (P2.3), on real headless renders.
//!
//! The clicks go through the real picker: a world point is projected through
//! the live camera to a pixel, and that pixel becomes the pointer, which the
//! picker turns back into a ray and casts against the scene's real meshes,
//! the committed terrain included. Needs a GPU (or lavapipe), so it runs only
//! with `ARGUSNET_GPU_TESTS=1`.

use std::path::{Path, PathBuf};

use argusnet_viewer::headless::{CameraPreset, HeadlessRenderOptions, ProbePointer, ProbeRequest};
use argusnet_viewer::state::RuntimeOverlayVisibility;
use argusnet_viewer::{render_headless, ViewMode};

fn gpu_tests() -> bool {
    std::env::var("ARGUSNET_GPU_TESTS").is_ok_and(|v| v == "1")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argusnet-picking-{name}-{}", std::process::id()));
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

const WIDTH: u32 = 650;
const HEIGHT: u32 = 480;

/// Where each marker is on the last frame, which is the one captured.
const DRONE: [f64; 3] = [-310.0, -600.0, 300.0];
const STATION: [f64; 3] = [800.0, 900.0, 5.0];
/// Far below the committed terrain (0.7 to 18 m), deeper than its glyph is
/// wide, so the terrain hides all of it.
const BURIED: [f64; 3] = [400.0, -200.0, -250.0];
const TRACK: [f64; 3] = [500.0, 300.0, 250.0];
const TRUTH: [f64; 3] = [500.0, 300.0, 60.0];
/// Open ground, clear of every marker and building.
const GROUND: [f64; 3] = [1500.0, -1400.0, 5.0];

/// The committed scene with a two-frame replay: the drone inside a radar's
/// range sphere, a node buried under the terrain, and a track and a truth
/// that share the id `t-1`.
fn scene(name: &str) -> PathBuf {
    let root = scratch(name);
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
        &root,
    );
    let frames: Vec<serde_json::Value> = (0..2)
        .map(|f| {
            let drone = [DRONE[0] - 10.0 * (1 - f) as f64, DRONE[1], DRONE[2]];
            serde_json::json!({
                "timestamp_s": f as f64 * 0.5,
                "tracks": [{"track_id": "t-1", "position": TRACK, "velocity": [0.0, 0.0, 0.0],
                            "measurement_std_m": 5.0, "update_count": 3, "stale_steps": 0}],
                "truths": [{"target_id": "t-1", "position": TRUTH, "velocity": [0.0, 0.0, 0.0]}],
                "nodes": [
                    {"node_id": "drone-1", "position": drone, "velocity": [20.0, 0.0, 0.0],
                     "is_mobile": true, "sensor_type": "optical", "fov_half_angle_deg": 30.0, "max_range_m": 900.0},
                    {"node_id": "station-1", "position": STATION, "velocity": [0.0, 0.0, 0.0],
                     "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0, "max_range_m": 400.0},
                    {"node_id": "station-2", "position": [0.0, -600.0, 5.0], "velocity": [0.0, 0.0, 0.0],
                     "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0, "max_range_m": 800.0},
                    {"node_id": "buried", "position": BURIED, "velocity": [0.0, 0.0, 0.0],
                     "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0, "max_range_m": 0.0}
                ],
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

fn at(world: [f64; 3]) -> ProbePointer {
    ProbePointer {
        world,
        offset_px: [0.0, 0.0],
    }
}

/// Every overlay that could steal a pick, on.
fn busy() -> RuntimeOverlayVisibility {
    RuntimeOverlayVisibility {
        fov_cones: true,
        trails: true,
        zones: true,
        radar_rings: true,
        ..Default::default()
    }
}

struct Render {
    probes: serde_json::Value,
    image: image::RgbaImage,
}

fn render(name: &str, scene: &Path, view_mode: ViewMode, probe: ProbeRequest) -> Render {
    let dir = scratch(&format!("{name}-out"));
    let png = dir.join(format!("{name}.png"));
    let output = dir.join("probes.json");
    render_headless(
        scene,
        HeadlessRenderOptions {
            output: Some(png.clone()),
            record_dir: None,
            camera: CameraPreset::Isometric,
            target_id: None,
            width: WIDTH,
            height: HEIGHT,
            fps: 30,
            view_mode,
            quality: Default::default(),
            frame_stats: 0,
            lifecycle: 0,
            probe: Some(ProbeRequest {
                output: output.clone(),
                ..probe
            }),
        },
    )
    .expect("render");
    if let Ok(keep) = std::env::var("ARGUSNET_PROBE_KEEP") {
        let _ = std::fs::create_dir_all(&keep);
        std::fs::copy(&png, Path::new(&keep).join(format!("{name}.png"))).unwrap();
    }
    Render {
        probes: serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap(),
        image: image::open(&png).unwrap().to_rgba8(),
    }
}

fn selected(value: &serde_json::Value) -> Option<(String, String)> {
    (!value.is_null()).then(|| {
        (
            value["kind"].as_str().unwrap().to_string(),
            value["id"].as_str().unwrap().to_string(),
        )
    })
}

fn node(id: &str) -> Option<(String, String)> {
    Some(("node".into(), id.into()))
}

/// The pixels that differ between two renders.
fn differing(a: &image::RgbaImage, b: &image::RgbaImage) -> Vec<[i64; 2]> {
    a.enumerate_pixels()
        .filter(|(x, y, p)| b.get_pixel(*x, *y) != *p)
        .map(|(x, y, _)| [i64::from(x), i64::from(y)])
        .collect()
}

/// The pixel a world point was drawn at, from a render's probe points.
fn pixel_of(render: &Render, index: usize) -> [i64; 2] {
    let pixel = &render.probes["points"][index]["pixel"];
    [pixel[0].as_i64().unwrap(), pixel[1].as_i64().unwrap()]
}

/// Clicks through the busy scene: the drone is picked through its sensor
/// volumes, its trail and a radar sphere around it; bare ground clears; the
/// buried node is behind the terrain, so clicking where it is picks nothing.
#[test]
fn clicks_pick_what_is_visible_under_the_pointer() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("clicks");
    let r = render(
        "clicks",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            overlays: Some(busy()),
            clicks: vec![at(DRONE), at(GROUND), at(STATION), at(BURIED), at(DRONE)],
            ..Default::default()
        },
    );
    let clicks = r.probes["clicks"].as_array().unwrap();
    eprintln!("{clicks:#?}");
    let outcomes: Vec<_> = clicks.iter().map(|c| selected(&c["selected"])).collect();
    assert_eq!(
        outcomes,
        vec![
            node("drone-1"),
            None,
            node("station-1"),
            None,
            node("drone-1")
        ],
        "drone, ground, station, buried (behind terrain), drone"
    );
    assert!(clicks.iter().all(|c| c["ignored"] == false));
    assert_eq!(selected(&r.probes["selected"]), node("drone-1"));
}

/// A track and a truth share `t-1`: each click selects its own kind. With
/// truths hidden (the default), clicking the truth selects nothing.
#[test]
fn same_id_different_kinds_and_hidden_truth() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("kinds");
    let shown = RuntimeOverlayVisibility {
        truths: true,
        ..Default::default()
    };
    let r = render(
        "kinds",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            overlays: Some(shown),
            clicks: vec![at(TRUTH), at(TRACK)],
            ..Default::default()
        },
    );
    let outcomes: Vec<_> = r.probes["clicks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| selected(&c["selected"]))
        .collect();
    assert_eq!(
        outcomes,
        vec![
            Some(("truth".into(), "t-1".into())),
            Some(("track".into(), "t-1".into()))
        ]
    );

    let r = render(
        "kinds-hidden",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            clicks: vec![at(TRUTH)],
            ..Default::default()
        },
    );
    assert_eq!(
        selected(&r.probes["clicks"][0]["selected"]),
        None,
        "a hidden truth was picked"
    );
}

/// In split view the main camera has the left half. A click there picks
/// through it; the same click moved into the right half is ignored and
/// leaves the selection alone.
#[test]
fn split_view_picks_only_in_the_main_viewport() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("split");
    let r = render(
        "split",
        &scene,
        ViewMode::Split,
        ProbeRequest {
            elevated: vec![DRONE],
            clicks: vec![
                at(DRONE),
                ProbePointer {
                    world: DRONE,
                    offset_px: [WIDTH as f32 / 2.0, 0.0],
                },
            ],
            ..Default::default()
        },
    );
    let clicks = r.probes["clicks"].as_array().unwrap();
    eprintln!("{clicks:#?}");
    let drone_px = pixel_of(&r, 0);
    assert!(
        drone_px[0] < i64::from(WIDTH / 2),
        "the drone is in the left viewport: {drone_px:?}"
    );
    assert_eq!(selected(&clicks[0]["selected"]), node("drone-1"));
    assert_eq!(
        clicks[1]["ignored"], true,
        "a click in the reconstruction viewport picked"
    );
    assert_eq!(
        selected(&r.probes["selected"]),
        node("drone-1"),
        "the ignored click changed the selection"
    );
}

/// The default overlays without trails. The drone's trail in this two-frame
/// scene is a ribbon about a pixel long ending under the glyph's centre, and
/// in roughly one render in twelve with probe interactions it is not drawn
/// (open; `docs/migration/PICKING_AND_SELECTION.md`). Rings have nothing to do with
/// trails, so their pixel-exact comparisons leave them out.
fn rings_only() -> RuntimeOverlayVisibility {
    RuntimeOverlayVisibility {
        trails: false,
        ..Default::default()
    }
}

/// The selection ring is drawn around the selected marker and nowhere else;
/// around a marker the terrain hides, it is hidden too. The hover ring is
/// drawn around the hovered marker, in its own colour.
#[test]
fn rings_are_drawn_around_their_marker_and_not_through_terrain() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let scene = scene("rings");
    let plain = render(
        "rings-plain",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            ..Default::default()
        },
    );
    let selected_drone = render(
        "rings-selected",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            select: Some(("node".into(), "drone-1".into())),
            ..Default::default()
        },
    );
    let selected_buried = render(
        "rings-buried",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            select: Some(("node".into(), "buried".into())),
            ..Default::default()
        },
    );
    let hovered_drone = render(
        "rings-hover",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            hover: Some(at(DRONE)),
            ..Default::default()
        },
    );
    // A selected truth while truths are hidden (the default): no ring.
    let selected_hidden = render(
        "rings-hidden",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            select: Some(("truth".into(), "t-1".into())),
            ..Default::default()
        },
    );
    // Selected and hovered at once: the selection ring alone.
    let both = render(
        "rings-both",
        &scene,
        ViewMode::RealWorld,
        ProbeRequest {
            elevated: vec![DRONE],
            overlays: Some(rings_only()),
            select: Some(("node".into(), "drone-1".into())),
            hover: Some(at(DRONE)),
            ..Default::default()
        },
    );
    assert_eq!(
        selected(&selected_drone.probes["selected"]),
        node("drone-1")
    );
    assert_eq!(
        selected(&selected_buried.probes["selected"]),
        node("buried")
    );
    assert_eq!(selected(&hovered_drone.probes["hovered"]), node("drone-1"));

    let drone = pixel_of(&plain, 0);
    let near = |p: &[i64; 2]| (p[0] - drone[0]).pow(2) + (p[1] - drone[1]).pow(2) <= 20 * 20;

    let ring = differing(&plain.image, &selected_drone.image);
    eprintln!("selection ring: {} pixels", ring.len());
    assert!(
        ring.len() > 40,
        "no selection ring: {} pixels changed",
        ring.len()
    );
    let stray: Vec<_> = ring.iter().filter(|p| !near(p)).collect();
    assert!(
        stray.is_empty(),
        "the ring drew away from the drone: {stray:?}"
    );
    // Yellow: more red than blue.
    let [r, _, b] = mean_rgb(&selected_drone.image, &ring);
    assert!(r > b, "the selection ring is not yellow: {r} {b}");

    let buried = differing(&plain.image, &selected_buried.image);
    assert!(
        buried.is_empty(),
        "a ring drew through the terrain: {} pixels",
        buried.len()
    );

    let hidden = differing(&plain.image, &selected_hidden.image);
    assert!(
        hidden.is_empty(),
        "a ring on a hidden marker: {} pixels",
        hidden.len()
    );
    assert_eq!(selected(&both.probes["hovered"]), node("drone-1"));
    let doubled = differing(&selected_drone.image, &both.image);
    assert!(
        doubled.is_empty(),
        "a hover ring on the selected marker: {} pixels",
        doubled.len()
    );

    let hover = differing(&plain.image, &hovered_drone.image);
    eprintln!("hover ring: {} pixels", hover.len());
    assert!(hover.len() > 20, "no hover ring");
    assert!(
        hover.iter().all(near),
        "the hover ring drew away from the drone"
    );
    let [r, _, b] = mean_rgb(&hovered_drone.image, &hover);
    assert!(b > r, "the hover ring is not blue: {r} {b}");
}

fn mean_rgb(image: &image::RgbaImage, pixels: &[[i64; 2]]) -> [f32; 3] {
    let n = pixels.len().max(1) as f32;
    [0, 1, 2].map(|k| {
        pixels
            .iter()
            .map(|[x, y]| f32::from(image.get_pixel(*x as u32, *y as u32)[k]))
            .sum::<f32>()
            / n
    })
}
