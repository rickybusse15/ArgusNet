use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracker_viewer::{render_headless, CameraPreset, HeadlessRenderOptions};

#[derive(Debug, Parser)]
#[command(
    name = "smart-tracker-viewer",
    about = "Open a smartscene-v1 package in the native viewer."
)]
struct Args {
    #[arg(
        long,
        help = "Path to a smartscene-v1 directory or scene_manifest.json file."
    )]
    scene: PathBuf,
    #[arg(long, default_value_t = false, help = "Render without opening the interactive viewer.")]
    headless: bool,
    #[arg(long, help = "Write one still PNG after scene load.")]
    output: Option<PathBuf>,
    #[arg(long, help = "Write a deterministic PNG sequence for the replay timeline.")]
    record_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = CameraPreset::TopDown)]
    camera: CameraPreset,
    #[arg(long, help = "Track/truth ID to focus when using follow-target camera.")]
    target_id: Option<String>,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.headless {
        return render_headless(
            args.scene,
            HeadlessRenderOptions {
                output: args.output,
                record_dir: args.record_dir,
                camera: args.camera,
                target_id: args.target_id,
                width: args.width,
                height: args.height,
                fps: args.fps,
            },
        );
    }
    tracker_viewer::run(args.scene)
}
