//! Pixel probes (P2.1): what the terrain material draws, read back from real
//! headless renders and held to the CPU's belief and land-cover queries.
//!
//! Each probe projects a world point through the live camera after capture and
//! records the 9 x 9 pixels around it. Lighting and tonemapping change absolute
//! colours, so the checks are on which channel dominates and by how much, and
//! on the stripes of the unknown hatch, never on exact values.
//!
//! Needs a GPU (or a software Vulkan such as lavapipe), so it runs only with
//! `ARGUSNET_GPU_TESTS=1`; CI's viewer job sets it.

use std::path::{Path, PathBuf};

use argusnet_mapping::GridBounds;
use argusnet_viewer::belief_render::{
    probe_belief_layout, BeliefClass, BeliefRenderSnapshot, ProbeExpectation,
};
use argusnet_viewer::headless::{CameraPreset, HeadlessRenderOptions, ProbeRequest};
use argusnet_viewer::land_cover_raster::{LandCoverDocument, LandCoverRaster};
use argusnet_viewer::{render_headless, ViewMode};

fn gpu_tests() -> bool {
    std::env::var("ARGUSNET_GPU_TESTS").is_ok_and(|v| v == "1")
}

/// A fresh directory for one test's outputs.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argusnet-probes-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn committed_scene() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene")
}

/// The committed scene's extent.
const EXTENT: GridBounds = GridBounds {
    x_min_m: -2160.0,
    x_max_m: 2160.0,
    y_min_m: -1980.0,
    y_max_m: 2160.0,
    resolution_m: 1.0,
};

#[derive(Debug, Clone, Copy)]
struct Probe {
    r: f32,
    g: f32,
    b: f32,
    /// Luminance spread across the window: the hatch's stripes.
    spread: f32,
}

impl Probe {
    fn luma(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
    fn saturation(&self) -> f32 {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        (max - min) / max.max(1e-6)
    }
}

fn render(name: &str, scene: &Path, probe: ProbeRequest) -> Vec<Probe> {
    render_with_report(name, scene, probe).0
}

fn render_with_report(
    name: &str,
    scene: &Path,
    probe: ProbeRequest,
) -> (Vec<Probe>, serde_json::Value) {
    let dir = scratch(name);
    let probes_path = probe.output.clone();
    let still = dir.join(format!("{name}.png"));
    render_headless(
        scene,
        HeadlessRenderOptions {
            output: Some(still.clone()),
            record_dir: None,
            camera: CameraPreset::TopDown,
            target_id: None,
            width: 650,
            height: 480,
            fps: 30,
            view_mode: ViewMode::RealWorld,
            probe: Some(probe),
        },
    )
    .expect("render");
    if let Ok(keep) = std::env::var("ARGUSNET_PROBE_KEEP") {
        let _ = std::fs::create_dir_all(&keep);
        std::fs::copy(&still, Path::new(&keep).join(format!("{name}.png"))).unwrap();
    }
    let document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(probes_path).expect("probes written"))
            .unwrap();
    let probes = document["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            let window: Vec<[f32; 3]> = point["window"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let c = p.as_array().unwrap();
                    [0, 1, 2].map(|k| c[k].as_f64().unwrap() as f32 / 255.0)
                })
                .collect();
            assert_eq!(
                window.len(),
                81,
                "the probe window fell off the frame: {point}"
            );
            let mean = [0, 1, 2].map(|k| window.iter().map(|p| p[k]).sum::<f32>() / 81.0);
            let luma = |p: &[f32; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            let lumas: Vec<f32> = window.iter().map(luma).collect();
            Probe {
                r: mean[0],
                g: mean[1],
                b: mean[2],
                spread: lumas.iter().fold(f32::MIN, |a, &b| a.max(b))
                    - lumas.iter().fold(f32::MAX, |a, &b| a.min(b)),
            }
        })
        .collect();
    (probes, document["rasters"].clone())
}

/// The unknown hatch: blue-violet over the terrain, and striped.
fn is_unknown_hatch(p: &Probe) -> bool {
    p.b > 1.1 * p.r && p.b > 1.1 * p.g && p.spread > 0.02
}

#[test]
fn the_rendered_belief_is_the_cpu_belief() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let (store, targets) = probe_belief_layout(EXTENT);
    let snapshot = BeliefRenderSnapshot::from_store(&store).unwrap();
    let probes = render(
        "belief",
        &committed_scene(),
        ProbeRequest {
            belief: Some(store),
            belief_overlay: true,
            late_patches: Vec::new(),
            overlays: None,
            elevated: Vec::new(),
            points: targets.iter().map(|t| t.xy).collect(),
            output: scratch("belief-json").join("probes.json"),
            ..Default::default()
        },
    );
    let by_label = |label: &str| {
        let index = targets.iter().position(|t| t.label == label).unwrap();
        probes[index]
    };
    for (target, probe) in targets.iter().zip(&probes) {
        eprintln!("{:>9}: {probe:?}", target.label);
        // What the CPU says is there is what the pixels must show.
        let cpu = snapshot.sample(target.xy[0], target.xy[1]);
        let expected = match target.expect {
            ProbeExpectation::Class(class) => class,
            ProbeExpectation::StaleSafe => BeliefClass::KnownSafe,
        };
        assert_eq!(cpu.class, expected, "{}", target.label);
        let ok = match target.expect {
            ProbeExpectation::Class(BeliefClass::KnownSafe) => {
                probe.g > 1.3 * probe.r && probe.g > 1.1 * probe.b
            }
            ProbeExpectation::Class(BeliefClass::KnownObstacle) => {
                probe.r > 1.5 * probe.g && probe.r > 1.5 * probe.b
            }
            ProbeExpectation::Class(BeliefClass::Uncertain) => {
                probe.r > 1.4 * probe.b && probe.g > 1.4 * probe.b
            }
            ProbeExpectation::Class(BeliefClass::Unknown) => is_unknown_hatch(probe),
            ProbeExpectation::Class(BeliefClass::OutsideGeofence) => {
                probe.luma() < 0.85 * by_label("safe").luma() && probe.saturation() < 0.3
            }
            ProbeExpectation::StaleSafe => {
                probe.g >= probe.r && probe.saturation() < 0.75 * by_label("safe").saturation()
            }
            ProbeExpectation::Class(other) => panic!("no probe rule for {other:?}"),
        };
        assert!(ok, "{} drew {probe:?}", target.label);
    }
    // Everything but unknown is drawn solid: only unknown carries stripes.
    for label in ["safe", "obstacle", "uncertain"] {
        assert!(
            by_label(label).spread < by_label("unknown").spread,
            "{label}"
        );
    }
}

/// A revision that arrives after the belief textures exist reaches the GPU
/// as tile writes, only for the tiles it touched, and the pixels follow it.
#[test]
fn a_late_revision_is_drawn_through_tile_writes_alone() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let (mut store, targets) = probe_belief_layout(EXTENT);
    let initial = store.clone();
    // Drive the safe block to obstacle: three occupied marks per cell.
    let cells = (0..12)
        .flat_map(|i| (0..15).map(move |j| (i, j)))
        .flat_map(|ij| {
            std::iter::repeat_n(
                argusnet_mapping::CellObservation {
                    ij,
                    height_m: None,
                    occupancy: Some(argusnet_mapping::OccupancyEvidence::Occupied {
                        height_m: 5.0,
                    }),
                },
                3,
            )
        })
        .collect();
    store.apply(argusnet_mapping::Observation {
        source_id: "late".into(),
        timestamp_s: 1.0,
        cells,
    });
    let revision = store.commit();
    let mut cpu = BeliefRenderSnapshot::from_store(&initial).unwrap();
    cpu.apply_revision(&revision);
    let safe = targets.iter().position(|t| t.label == "safe").unwrap();
    let uncertain = targets.iter().position(|t| t.label == "uncertain").unwrap();
    let at = targets[safe].xy;
    assert_eq!(cpu.sample(at[0], at[1]).class, BeliefClass::KnownObstacle);

    // Cell (10, 7), in tile (1, 0): obstacle now. Tile (0, 1) holds, at the
    // same local position, cell (2, 15), which is unknown; so a tile written
    // with its origin's x and y swapped would draw the hatch here.
    let extra = {
        let resolution = (EXTENT.x_max_m - EXTENT.x_min_m) / 38.0;
        [
            EXTENT.x_min_m + 10.5 * resolution,
            EXTENT.y_min_m + 7.5 * resolution,
        ]
    };
    assert_eq!(
        cpu.sample(extra[0], extra[1]).class,
        BeliefClass::KnownObstacle
    );
    let mut points: Vec<[f64; 2]> = targets.iter().map(|t| t.xy).collect();
    points.push(extra);
    let (probes, rasters) = render_with_report(
        "late",
        &committed_scene(),
        ProbeRequest {
            belief: Some(initial),
            belief_overlay: true,
            late_patches: revision.patches.clone(),
            overlays: None,
            elevated: Vec::new(),
            points,
            output: scratch("late-json").join("probes.json"),
            ..Default::default()
        },
    );
    eprintln!("rasters {rasters}");
    let now_obstacle = probes[safe];
    assert!(
        now_obstacle.r > 1.5 * now_obstacle.g && now_obstacle.r > 1.5 * now_obstacle.b,
        "the late revision did not reach the GPU: {now_obstacle:?}"
    );
    let off_diagonal = probes[targets.len()];
    assert!(
        off_diagonal.r > 1.5 * off_diagonal.g && off_diagonal.r > 1.5 * off_diagonal.b,
        "tile (1, 0) was not written where it belongs: {off_diagonal:?}"
    );
    let still_uncertain = probes[uncertain];
    assert!(
        still_uncertain.r > 1.4 * still_uncertain.b && still_uncertain.g > 1.4 * still_uncertain.b
    );
    assert_eq!(rasters["uploaded_tiles"], revision.patches.len() as u64);
    assert_eq!(rasters["refused_tiles"], 0);
    assert_eq!(rasters["waiting_tiles"], 0);
}

/// Overlay on, no belief at all: the explicit unavailable state, everywhere.
#[test]
fn no_belief_renders_as_unavailable_not_as_a_guess() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    let (_, targets) = probe_belief_layout(EXTENT);
    let probes = render(
        "unavailable",
        &committed_scene(),
        ProbeRequest {
            belief: None,
            belief_overlay: true,
            late_patches: Vec::new(),
            overlays: None,
            elevated: Vec::new(),
            points: targets.iter().map(|t| t.xy).collect(),
            output: scratch("unavailable-json").join("probes.json"),
            ..Default::default()
        },
    );
    for (target, probe) in targets.iter().zip(&probes) {
        assert!(is_unknown_hatch(probe), "{} drew {probe:?}", target.label);
    }
}

/// A scene with land cover tints by the runtime layer's class at each point.
#[test]
fn the_land_cover_tint_is_the_runtime_layers_class() {
    if !gpu_tests() {
        eprintln!("skipped: set ARGUSNET_GPU_TESTS=1 to render");
        return;
    }
    // The committed scene, plus a land-cover file: forest west of x = 0,
    // water east of it, and an unpaletted open band across the middle.
    let scene = scratch("land-cover-scene");
    copy_dir(&committed_scene(), &scene);
    let resolution = 60.0;
    let cols = 72;
    let rows = 69;
    let class_at = |c: usize, r: usize| -> u8 {
        if (30..38).contains(&r) {
            0
        } else if c < cols / 2 {
            2
        } else {
            3
        }
    };
    let tiles: Vec<serde_json::Value> = (0..3)
        .flat_map(|ty| (0..3).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            let classes: Vec<Vec<u8>> = (0..32)
                .map(|r| {
                    (0..32)
                        .map(|c| {
                            let (gc, gr) = (tx * 32 + c, ty * 32 + r);
                            if gc < cols && gr < rows {
                                class_at(gc, gr)
                            } else {
                                0
                            }
                        })
                        .collect()
                })
                .collect();
            serde_json::json!({
                "tx": tx, "ty": ty,
                "x_min_m": -2160.0 + tx as f64 * 32.0 * resolution,
                "y_min_m": -1980.0 + ty as f64 * 32.0 * resolution,
                "cell_size_m": resolution, "classes": classes, "density": null,
            })
        })
        .collect();
    let document = serde_json::json!({
        "format": "argusnet.land-cover", "version": 1,
        "bounds": {"x_min_m": -2160.0, "x_max_m": 2160.0, "y_min_m": -1980.0, "y_max_m": 2160.0},
        "tile_size_cells": 32, "resolution": resolution, "tiles": tiles,
        "legend": {"open": 0, "forest": 2, "water": 3},
        "palette": {"forest": [0.10, 0.70, 0.15, 1.0], "water": [0.10, 0.30, 0.95, 1.0]},
    });
    std::fs::write(scene.join("metadata/land_cover.json"), document.to_string()).unwrap();
    let manifest_path = scene.join("scene_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["metadata"]["land_cover"] = serde_json::json!("metadata/land_cover.json");
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();

    let raster = LandCoverRaster::from_document(
        serde_json::from_value::<LandCoverDocument>(document).unwrap(),
    )
    .unwrap();
    // Clear of the scene's buildings, which sit near the corners and centre.
    let points = [[-1200.0, -1200.0], [1200.0, -1200.0], [-1200.0, 105.0]];
    let expected: Vec<u8> = points
        .iter()
        .map(|p| raster.class_at(p[0], p[1]).unwrap())
        .collect();
    assert_eq!(expected, vec![2, 3, 0]);
    let probes = render(
        "land-cover",
        &scene,
        ProbeRequest {
            belief: None,
            belief_overlay: false,
            late_patches: Vec::new(),
            overlays: None,
            elevated: Vec::new(),
            points: points.to_vec(),
            output: scratch("land-cover-json").join("probes.json"),
            ..Default::default()
        },
    );
    for (p, probe) in points.iter().zip(&probes) {
        eprintln!("{p:?}: {probe:?}");
    }
    // The tint is presentation, deliberately subtle, and multiplies the baked
    // terrain colour; so each class is judged against the untinted open
    // ground, not by absolute channel ratios.
    let (forest, water, open) = (probes[0], probes[1], probes[2]);
    let green = |p: &Probe| p.g / p.b;
    let blue = |p: &Probe| p.b / p.g;
    assert!(
        green(&forest) > 1.05 * green(&open),
        "forest {forest:?} open {open:?}"
    );
    assert!(
        blue(&water) > 1.08 * blue(&open),
        "water {water:?} open {open:?}"
    );
    assert!(
        blue(&water) > 1.15 * blue(&forest),
        "water {water:?} forest {forest:?}"
    );
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
