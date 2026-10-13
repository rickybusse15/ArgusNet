use std::path::PathBuf;

use anyhow::Result;
use argusnet_viewer::quality::QualityTier;
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
        help = "Path to a smartscene-v1 directory or scene_manifest.json file. Optional \
                with --run: without it the viewer opens on the run's map bounds alone \
                and draws only what the run reveals (with --world-knowledge none, the \
                map the drones build)."
    )]
    scene: Option<PathBuf>,
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
    #[arg(
        long,
        value_enum,
        help = "Rendering quality tier (low, standard, high); overrides the saved \
                preference. Headless renders default to standard."
    )]
    quality: Option<QualityTier>,
    #[arg(
        long,
        default_value_t = 0,
        help = "Headless: after the still, time this many unpaced frames and write \
                <output>.frames.json (G2 frame statistics)."
    )]
    frame_stats: u32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Headless: after the still, run this many lifecycle rotations (seek, toggle, \
                reload, tier, projection) and write <output>.lifecycle.json (C8)."
    )]
    lifecycle: u32,
    #[arg(
        long,
        help = "Run this scenario (an argusnet.scenario-config or scenario bundle) in process, \
                in the scene: it starts paused; Run, Step, Speed and Reset drive it."
    )]
    run: Option<PathBuf>,
    #[command(flatten)]
    run_options: RunOptionArgs,
    #[cfg(feature = "live-stream")]
    #[arg(
        long,
        conflicts_with = "run",
        help = "Subscribe to live frames from this argusnetd endpoint."
    )]
    live: Option<String>,
    /// CA (PEM) that signed the daemon certificate. Required, unless the endpoint
    /// is loopback, to stream from `--live`.
    #[cfg(feature = "live-stream")]
    #[arg(long, env = "ARGUSNET_TLS_CLIENT_CA")]
    live_tls_ca: Option<PathBuf>,
    /// Client certificate (PEM) for mTLS, paired with --live-tls-key.
    #[cfg(feature = "live-stream")]
    #[arg(long, env = "ARGUSNET_TLS_CERT")]
    live_tls_cert: Option<PathBuf>,
    /// Client private key (PEM), matching --live-tls-cert.
    #[cfg(feature = "live-stream")]
    #[arg(long, env = "ARGUSNET_TLS_KEY")]
    live_tls_key: Option<PathBuf>,
    /// Name to verify against the daemon certificate, when it differs from the
    /// endpoint host.
    #[cfg(feature = "live-stream")]
    #[arg(long)]
    live_tls_domain: Option<String>,
}

/// The options `argusnet-rs sim` takes for a run, with the same meanings,
/// for `--run`: the energy profile (D-3), the planning policy (P1),
/// timestamped delivery (B4) and scheduled faults (F2).
#[derive(clap::Args, Debug)]
struct RunOptionArgs {
    /// `legacy` charges motion only; `physical_v1` also charges hover power.
    #[arg(long, value_parser = ["legacy", "physical_v1"], default_value = "legacy", requires = "run")]
    energy_profile: String,
    /// Who allocates the mission's POIs (P1): `reference` or `efficient_v1`.
    #[arg(long, value_parser = ["reference", "efficient_v1"], default_value = "reference", requires = "run")]
    planning_policy: String,
    /// Fuse the evidence acquired this many seconds ago (B4). Omit for the
    /// legacy same-tick path.
    #[arg(long, requires = "run")]
    fusion_lag_s: Option<f64>,
    /// Every source's delivery delay, unless `--source-delay` names it.
    #[arg(long, default_value_t = 0.0, requires = "fusion_lag_s")]
    delivery_delay_s: f64,
    /// A source's own delay, as `NODE_ID=SECONDS`; repeat per source.
    #[arg(long, value_parser = parse_source_delay, requires = "fusion_lag_s")]
    source_delay: Vec<(String, f64)>,
    /// Extra delay drawn uniformly per envelope (seeded).
    #[arg(long, default_value_t = 0.0, requires = "fusion_lag_s")]
    delivery_jitter_s: f64,
    /// Envelopes one source may have queued before new ones overflow.
    #[arg(long, default_value_t = argusnet_sim::delivery::DEFAULT_QUEUE_CAPACITY, requires = "fusion_lag_s")]
    delivery_queue_capacity: usize,
    /// Scheduled sensor and link faults: an `argusnet.fault-manifest` (F2).
    #[arg(long, requires = "run")]
    faults: Option<PathBuf>,
    /// `prior` (the true terrain and obstacles) or `none`: the platforms map
    /// the world themselves and the view shows the map they build.
    #[arg(long, value_parser = ["prior", "none"], default_value = "prior", requires = "run")]
    world_knowledge: String,
    /// Run the mission lane at `NUM/DEN` Hz instead of every tick (B2.4,
    /// `scheduled_v1`), as `argusnet-rs sim --executor lanes --mission-rate`.
    #[arg(long, value_parser = argusnet_sim::scheduler::parse_rate, requires = "run")]
    mission_rate: Option<argusnet_sim::scheduler::Schedule>,
    /// Run the mapping lane at `NUM/DEN` Hz (B2.5), as `argusnet-rs sim
    /// --executor lanes --mapping-rate`.
    #[arg(long, value_parser = argusnet_sim::scheduler::parse_rate, requires = "run")]
    mapping_rate: Option<argusnet_sim::scheduler::Schedule>,
}

impl RunOptionArgs {
    fn options(&self) -> Result<argusnet_sim::run_options::RunOptions> {
        use argusnet_sim::run_options::{DeliverySeconds, RunOptions};
        Ok(RunOptions {
            energy_profile: match self.energy_profile.as_str() {
                "physical_v1" => argusnet_sim::run_options::EnergyProfile::PhysicalV1,
                _ => argusnet_sim::run_options::EnergyProfile::Legacy,
            },
            planning_policy: match self.planning_policy.as_str() {
                "efficient_v1" => argusnet_sim::run_options::PlanningPolicy::EfficientV1,
                _ => argusnet_sim::run_options::PlanningPolicy::Reference,
            },
            delivery: self.fusion_lag_s.map(|fusion_lag_s| DeliverySeconds {
                fusion_lag_s,
                delay_s: self.delivery_delay_s,
                source_delay_s: self.source_delay.iter().cloned().collect(),
                jitter_s: self.delivery_jitter_s,
                queue_capacity: self.delivery_queue_capacity,
            }),
            faults: self
                .faults
                .as_deref()
                .map(RunOptions::read_faults)
                .transpose()
                .map_err(|error| anyhow::anyhow!(error))?,
            world_knowledge: match self.world_knowledge.as_str() {
                "none" => argusnet_sim::belief_world::WorldKnowledge::NoPrior,
                _ => argusnet_sim::belief_world::WorldKnowledge::Prior,
            },
            exploration: None,
            lane_rates: argusnet_sim::scheduler::LaneRates {
                mapping: self.mapping_rate,
                mission: self.mission_rate,
            },
        })
    }
}

/// `NODE_ID=SECONDS` for `--source-delay`.
fn parse_source_delay(text: &str) -> std::result::Result<(String, f64), String> {
    let (id, seconds) = text
        .split_once('=')
        .ok_or_else(|| format!("expected NODE_ID=SECONDS, got {text:?}"))?;
    let seconds: f64 = seconds
        .parse()
        .map_err(|_| format!("{seconds:?} is not a number of seconds"))?;
    if id.is_empty() {
        return Err("the node id is empty".into());
    }
    Ok((id.to_string(), seconds))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.headless {
        let Some(scene) = args.scene.clone() else {
            anyhow::bail!("--scene is required for a headless render");
        };
        return render_headless(
            scene,
            HeadlessRenderOptions {
                output: args.output,
                record_dir: args.record_dir,
                camera: args.camera,
                target_id: args.target_id,
                width: args.width,
                height: args.height,
                fps: args.fps,
                view_mode: args.view_mode,
                probe: None,
                quality: args.quality.unwrap_or_default(),
                frame_stats: args.frame_stats,
                lifecycle: args.lifecycle,
            },
        );
    }
    if let Some(run) = args.run {
        let options = args.run_options.options()?;
        return argusnet_viewer::run_sim(args.scene, run, options, args.view_mode, args.quality);
    }
    let Some(scene) = args.scene else {
        anyhow::bail!("--scene is required, except with --run");
    };
    #[cfg(feature = "live-stream")]
    if let Some(endpoint) = args.live {
        let tls = argusnet_viewer::live::LiveTlsConfig {
            ca_cert: args.live_tls_ca,
            client_cert: args.live_tls_cert,
            client_key: args.live_tls_key,
            domain_name: args.live_tls_domain,
        };
        return argusnet_viewer::run_live(
            scene,
            endpoint,
            tls,
            args.view_mode,
            args.autoplay,
            args.quality,
        );
    }
    argusnet_viewer::run(scene, args.view_mode, args.autoplay, args.quality)
}
