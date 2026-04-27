use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use image::{ImageBuffer, Rgba, RgbaImage};

use crate::replay::{ReplayDocument, ReplayFrame};
use crate::schema::{Bounds2d, ScenePackage};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum CameraPreset {
    TopDown,
    Isometric,
    FollowTarget,
}

#[derive(Clone, Debug)]
pub struct HeadlessRenderOptions {
    pub output: Option<PathBuf>,
    pub record_dir: Option<PathBuf>,
    pub camera: CameraPreset,
    pub target_id: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

pub fn render_headless(scene_path: impl AsRef<Path>, options: HeadlessRenderOptions) -> Result<()> {
    if options.output.is_none() && options.record_dir.is_none() {
        bail!("headless rendering requires --output or --record-dir");
    }

    let scene_package = ScenePackage::load(scene_path)?;
    let replay = scene_package
        .replay
        .clone()
        .map(ReplayDocument::try_from)
        .transpose()
        .context("failed to parse packaged replay.json")?;

    if let Some(output) = options.output.as_ref() {
        let image = render_frame(
            &scene_package,
            replay.as_ref(),
            replay.as_ref().and_then(|doc| doc.frames.first()),
            &options,
            0,
        );
        save_image(output, &image)?;
    }

    if let Some(record_dir) = options.record_dir.as_ref() {
        let replay = replay
            .as_ref()
            .context("record mode requires a packaged replay.json")?;
        fs::create_dir_all(record_dir)
            .with_context(|| format!("failed to create {}", record_dir.display()))?;
        for (frame_index, frame) in replay.frames.iter().enumerate() {
            let image = render_frame(
                &scene_package,
                Some(replay),
                Some(frame),
                &options,
                frame_index,
            );
            let path = record_dir.join(format!("frame_{frame_index:05}.png"));
            save_image(&path, &image)?;
        }
    }

    let _ = options.fps;
    Ok(())
}

fn save_image(path: &Path, image: &RgbaImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    image
        .save(path)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn render_frame(
    scene_package: &ScenePackage,
    replay: Option<&ReplayDocument>,
    frame: Option<&ReplayFrame>,
    options: &HeadlessRenderOptions,
    frame_index: usize,
) -> RgbaImage {
    let mut image: RgbaImage =
        ImageBuffer::from_pixel(options.width, options.height, Rgba([245, 242, 234, 255]));
    draw_background(&mut image);
    let view = view_bounds(
        &scene_package.environment.bounds_xy_m,
        frame,
        options.camera,
        options.target_id.as_deref(),
    );

    draw_border(
        &mut image,
        options.width,
        options.height,
        Rgba([48, 61, 72, 255]),
    );
    draw_rect_outline(
        &mut image,
        24,
        24,
        options.width as i32 - 24,
        options.height as i32 - 24,
        Rgba([48, 61, 72, 255]),
    );

    if let Some(replay) = replay {
        draw_track_trails(
            &mut image,
            replay,
            frame_index,
            &view,
            options.camera,
            &scene_package.environment.bounds_xy_m,
            options.width,
            options.height,
        );
    }

    if let Some(frame) = frame {
        for truth in &frame.truths {
            let (x, y) = project_point(
                truth.position,
                &view,
                options.camera,
                &scene_package.environment.bounds_xy_m,
                options.width,
                options.height,
            );
            draw_circle(&mut image, x, y, 6, Rgba([66, 148, 93, 255]));
        }
        for node in &frame.nodes {
            let (x, y) = project_point(
                node.position,
                &view,
                options.camera,
                &scene_package.environment.bounds_xy_m,
                options.width,
                options.height,
            );
            draw_square(&mut image, x, y, 7, Rgba([43, 112, 187, 255]));
        }
        for track in &frame.tracks {
            let color = if options.target_id.as_deref() == Some(track.track_id.as_str()) {
                Rgba([198, 44, 44, 255])
            } else {
                Rgba([227, 123, 48, 255])
            };
            let (x, y) = project_point(
                track.position,
                &view,
                options.camera,
                &scene_package.environment.bounds_xy_m,
                options.width,
                options.height,
            );
            draw_circle(&mut image, x, y, 7, color);
        }
    }

    image
}

fn view_bounds(
    scene_bounds: &Bounds2d,
    frame: Option<&ReplayFrame>,
    camera: CameraPreset,
    target_id: Option<&str>,
) -> Bounds2d {
    if camera != CameraPreset::FollowTarget {
        return scene_bounds.clone();
    }
    let Some(frame) = frame else {
        return scene_bounds.clone();
    };
    let focus = target_id
        .and_then(|id| {
            frame
                .tracks
                .iter()
                .find(|track| track.track_id == id)
                .map(|track| [track.position[0], track.position[1]])
                .or_else(|| {
                    frame
                        .truths
                        .iter()
                        .find(|truth| truth.target_id == id)
                        .map(|truth| [truth.position[0], truth.position[1]])
                })
        })
        .or_else(|| {
            frame
                .tracks
                .first()
                .map(|track| [track.position[0], track.position[1]])
        });
    let Some([cx, cy]) = focus else {
        return scene_bounds.clone();
    };
    let span = scene_bounds
        .span_xy()
        .into_iter()
        .fold(0.0_f32, f32::max)
        .max(200.0)
        * 0.35;
    Bounds2d {
        x_min_m: cx - span,
        x_max_m: cx + span,
        y_min_m: cy - span,
        y_max_m: cy + span,
    }
}

fn draw_background(image: &mut RgbaImage) {
    let height = image.height().max(1);
    for y in 0..image.height() {
        let blend = y as f32 / height as f32;
        let r = (245.0 - 18.0 * blend) as u8;
        let g = (242.0 - 10.0 * blend) as u8;
        let b = (234.0 - 2.0 * blend) as u8;
        for x in 0..image.width() {
            image.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_track_trails(
    image: &mut RgbaImage,
    replay: &ReplayDocument,
    frame_index: usize,
    view: &Bounds2d,
    camera: CameraPreset,
    scene_bounds: &Bounds2d,
    width: u32,
    height: u32,
) {
    let mut positions: std::collections::HashMap<String, Vec<[f32; 3]>> =
        std::collections::HashMap::new();
    for frame in replay.frames.iter().take(frame_index + 1) {
        for track in &frame.tracks {
            positions
                .entry(track.track_id.clone())
                .or_default()
                .push(track.position);
        }
    }
    for history in positions.values() {
        for segment in history.windows(2) {
            let (x0, y0) = project_point(segment[0], view, camera, scene_bounds, width, height);
            let (x1, y1) = project_point(segment[1], view, camera, scene_bounds, width, height);
            draw_line(image, x0, y0, x1, y1, Rgba([227, 123, 48, 180]));
        }
    }
}

fn project_point(
    point: [f32; 3],
    view: &Bounds2d,
    camera: CameraPreset,
    scene_bounds: &Bounds2d,
    width: u32,
    height: u32,
) -> (i32, i32) {
    let margin = 40.0_f32;
    let inner_width = (width as f32 - margin * 2.0).max(1.0);
    let inner_height = (height as f32 - margin * 2.0).max(1.0);
    let tx = ((point[0] - view.x_min_m) / (view.x_max_m - view.x_min_m).max(1.0)).clamp(0.0, 1.0);
    let ty = ((point[1] - view.y_min_m) / (view.y_max_m - view.y_min_m).max(1.0)).clamp(0.0, 1.0);
    let mut sx = margin + tx * inner_width;
    let mut sy = height as f32 - margin - ty * inner_height;

    if camera == CameraPreset::Isometric {
        let center = scene_bounds.center_xy();
        sx += (point[1] - center[1]) * 0.08;
        sy -= point[2] * 0.35;
        sy -= (point[0] - center[0]) * 0.04;
    }

    (sx.round() as i32, sy.round() as i32)
}

fn draw_border(image: &mut RgbaImage, width: u32, height: u32, color: Rgba<u8>) {
    draw_rect_outline(image, 0, 0, width as i32 - 1, height as i32 - 1, color);
}

fn draw_rect_outline(image: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    draw_line(image, x0, y0, x1, y0, color);
    draw_line(image, x1, y0, x1, y1, color);
    draw_line(image, x1, y1, x0, y1, color);
    draw_line(image, x0, y1, x0, y0, color);
}

fn draw_square(image: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: Rgba<u8>) {
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            put_pixel(image, x, y, color);
        }
    }
}

fn draw_circle(image: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: Rgba<u8>) {
    let radius_sq = radius * radius;
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius_sq {
                put_pixel(image, x, y, color);
            }
        }
    }
}

fn draw_line(image: &mut RgbaImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        put_pixel(image, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err * 2;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put_pixel(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    if x < image.width() && y < image.height() {
        image.put_pixel(x, y, color);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    #[test]
    fn headless_renderer_writes_png_outputs() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("argusnet-headless-{suffix}"));
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(
            root.join("scene_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "format_version": "smartscene-v1",
                "scene_id": "demo",
                "bounds_xy_m": {
                    "x_min_m": -100.0,
                    "x_max_m": 100.0,
                    "y_min_m": -100.0,
                    "y_max_m": 100.0
                },
                "runtime_crs": {},
                "source_crs_id": "local",
                "layers": [],
                "replay": {"path": "replay.json"},
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json"
                },
                "build": {"source_kind": "synthetic"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/style.json"),
            serde_json::to_string_pretty(&json!({
                "style_version": "smartstyle-v1",
                "layers": []
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/environment.json"),
            serde_json::to_string_pretty(&json!({
                "runtime_crs": {},
                "bounds_xy_m": {
                    "x_min_m": -100.0,
                    "x_max_m": 100.0,
                    "y_min_m": -100.0,
                    "y_max_m": 100.0
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("replay.json"),
            serde_json::to_string_pretty(&json!({
                "frames": [
                    {
                        "timestamp_s": 0.0,
                        "tracks": [{"track_id": "t1", "position": [0.0, 0.0, 10.0], "velocity": [0.0, 0.0, 0.0]}],
                        "truths": [{"target_id": "truth-1", "position": [10.0, 10.0, 12.0], "velocity": [0.0, 0.0, 0.0]}],
                        "nodes": [{"node_id": "n1", "position": [-20.0, -20.0, 2.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": false}]
                    },
                    {
                        "timestamp_s": 1.0,
                        "tracks": [{"track_id": "t1", "position": [20.0, 0.0, 10.0], "velocity": [0.0, 0.0, 0.0]}],
                        "truths": [],
                        "nodes": []
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();

        let still = root.join("still.png");
        let record_dir = root.join("frames");
        render_headless(
            &root,
            HeadlessRenderOptions {
                output: Some(still.clone()),
                record_dir: Some(record_dir.clone()),
                camera: CameraPreset::TopDown,
                target_id: None,
                width: 640,
                height: 360,
                fps: 30,
            },
        )
        .unwrap();

        assert!(still.exists());
        assert!(record_dir.join("frame_00000.png").exists());
        assert!(record_dir.join("frame_00001.png").exists());

        let _ = fs::remove_dir_all(root);
    }
}
