use std::path::PathBuf;

use anyhow::Result;
use argusnet_viewer::{render_headless, CameraPreset, HeadlessRenderOptions, ViewMode};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "argusnet-viewer",
    about = "Open a smartscene-v1 package in the native viewer."
)]
struct Args {
    #[arg(
        long,
        help = "Path to a smartscene-v1 directory or scene_manifest.json file."
    )]
    scene: PathBuf,
    #[arg(
        long,
        default_value_t = false,
        help = "Render without opening the interactive viewer."
    )]
    headless: bool,
    #[arg(long, help = "Write one still PNG after scene load.")]
    output: Option<PathBuf>,
    #[arg(
        long,
        help = "Write a deterministic PNG sequence for the replay timeline."
    )]
    record_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = CameraPreset::TopDown)]
    camera: CameraPreset,
    #[arg(
        long,
        help = "Track/truth ID to focus when using follow-target camera."
    )]
    target_id: Option<String>,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    #[arg(
        long,
        value_enum,
        default_value_t = ViewMode::RealWorld,
        help = "Initial view for the interactive viewer: real-world terrain, scan-map \
                reconstruction, or split (terrain left / reconstruction right)."
    )]
    view_mode: ViewMode,
    #[arg(
        long,
        default_value_t = false,
        help = "Start the replay timeline playing immediately (fills the scan-map \
                reconstruction without pressing Space)."
    )]
    autoplay: bool,
    #[cfg(feature = "live-stream")]
    #[arg(long, help = "Subscribe to live frames from this argusnetd endpoint.")]
    live: Option<String>,
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
                view_mode: args.view_mode,
            },
        );
    }
    #[cfg(feature = "live-stream")]
    if let Some(endpoint) = args.live {
        return argusnet_viewer::run_live(args.scene, endpoint, args.view_mode, args.autoplay);
    }
    argusnet_viewer::run(args.scene, args.view_mode, args.autoplay)
}
