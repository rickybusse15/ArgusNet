//! `argusnet-rs`: run a scenario in the Rust tick, from a command line.
//!
//! The first Rust entry point for the simulation (DELIVERY_PLAN P4.9). It runs
//! an `argusnet.scenario-bundle/v1` -- written by `argusnet sim
//! --export-bundle` -- through the same assembly the tests use, and writes a
//! replay the Python evaluation and the viewer read. No Python is involved at
//! run time.
//!
//! ```text
//! argusnet-rs build --config config.json --bundle scenario.json
//! argusnet-rs sim --config config.json --replay replay.json
//! argusnet-rs sim --bundle scenario.json --replay replay.json
//! argusnet-rs sim --bundle scenario.json --checkpoint-at 120 --checkpoint ck.json
//! argusnet-rs sim --bundle scenario.json --resume ck.json --replay rest.json
//! ```
//!
//! # Exit codes
//!
//! `0` success; `2` the input could not be read or is invalid; `3` the bundle
//! uses something this build does not assemble; `1` the run faulted.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use argusnet_sim::builder::{self, BuildError, ScenarioConfig};
use argusnet_sim::scenario::{assemble_with, BundleError, RunError, ScenarioBundle};
use argusnet_sim::scheduler::{
    parse_rate, ExecutorError, LegacyLaneExecutor, PhasedLaneExecutor, Schedule,
};

mod rundir;
use argusnet_sim::simulation::TickError;
use argusnet_sim::snapshot::SimulationCheckpoint;
use clap::{Parser, Subcommand};
use rundir::{Recorder, RunDir};

#[derive(Parser)]
#[command(
    name = "argusnet-rs",
    about = "Run an ArgusNet scenario bundle in the Rust simulation runtime.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
// Parsed once per process; boxing the largest variant buys nothing.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run a scenario bundle (or a scenario config) to completion and write a
    /// replay.
    Sim(SimArgs),
    /// Build the scenario bundle an `argusnet.scenario-config` describes.
    Build(BuildArgs),
    /// Build a scene package (`smartscene-v1`) from a replay, as
    /// `argusnet build-scene` does, with no Python.
    BuildScene(BuildSceneArgs),
    /// Run captured `IngestFrameRequest`s through the in-process fusion engine
    /// and write its `IngestFrameResponse`s -- the direct-fusion parity check
    /// against `argusnetd` (P4.9c).
    #[command(hide = true)]
    Fuse(FuseArgs),
    /// Print the runtime, bundle and checkpoint format versions.
    Version,
}

#[derive(clap::Args)]
struct BuildSceneArgs {
    /// The replay whose `meta` describes the world (terrain, obstacles).
    #[arg(long)]
    replay: PathBuf,
    /// A new or empty directory for the package.
    #[arg(long)]
    output: PathBuf,
}

#[derive(clap::Args)]
struct FuseArgs {
    /// `{"tracker": hex(TrackerConfig), "requests": [hex(IngestFrameRequest)]}`.
    #[arg(long)]
    capture: PathBuf,
    /// Where to write `{"responses": [hex(IngestFrameResponse)]}`.
    #[arg(long)]
    output: PathBuf,
}

#[derive(clap::Args)]
struct BuildArgs {
    /// The `argusnet.scenario-config` document.
    #[arg(long)]
    config: PathBuf,
    /// Where to write the bundle.
    #[arg(long)]
    bundle: PathBuf,
}

#[derive(clap::Args)]
#[command(group(clap::ArgGroup::new("input").required(true).args(["bundle", "config", "recover"])))]
struct SimArgs {
    /// The scenario bundle (`argusnet sim --export-bundle`, or `argusnet-rs
    /// build`).
    #[arg(long)]
    bundle: Option<PathBuf>,
    /// An `argusnet.scenario-config`, built in process -- no bundle file, and
    /// no Python.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Where to write the replay. Omit to run without one.
    #[arg(long)]
    replay: Option<PathBuf>,
    /// Write a checkpoint after this many ticks have run.
    #[arg(long, requires = "checkpoint")]
    checkpoint_at: Option<u64>,
    /// Where `--checkpoint-at` writes.
    #[arg(long, requires = "checkpoint_at")]
    checkpoint: Option<PathBuf>,
    /// Resume from a checkpoint of this same bundle. The replay then holds
    /// the ticks from the resume point on.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Reset the run after this many ticks, then run it to completion (A2.1).
    /// The replay holds only the run after the reset, so it must equal a
    /// straight run's byte for byte.
    #[arg(long, conflicts_with = "resume")]
    reset_at: Option<u64>,
    /// `legacy` steps the eleven-stage tick directly; `scheduled` drives the
    /// same tick from the scheduler's integer-time queue as one compound lane
    /// (B2.2); `lanes` runs fusion, mapping, mission and publication as their
    /// own phased tasks after the compound lane (B2.3). The replay is
    /// identical every way.
    #[arg(long, value_enum, default_value = "legacy")]
    executor: Executor,
    /// Run the mission lane (stage 9) at its own rate, `NUM/DEN` Hz, instead
    /// of every tick (B2.4): `1/1` is once a second, `2/5` every 2.5 s. It
    /// runs at the first tick at or after each of its deadlines and sits out
    /// the rest, and its next run is given the time since it last ran. A
    /// declared behaviour change, recorded in the replay as
    /// `execution_model: scheduled_v1`. Needs `--executor lanes`.
    #[arg(long, value_parser = parse_rate)]
    mission_rate: Option<Schedule>,
    /// Run the mapping lane (stage 8) at its own rate, `NUM/DEN` Hz (B2.5).
    /// It keeps what every tick it sits out would have integrated -- the
    /// platforms' positions, the range scans -- and integrates them in order
    /// when it next runs, so coverage and the self-built map catch up exactly;
    /// between its runs the rest of the tick reads the map as it last left
    /// it. Localization runs at this rate. Recorded like `--mission-rate`.
    /// Needs `--executor lanes`.
    #[arg(long, value_parser = parse_rate)]
    mapping_rate: Option<Schedule>,
    /// Workers for the stages that fork work inside a phase (B3): currently
    /// observation generation. The replay is identical at every count; 1 is
    /// the serial path.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Scheduled sensor and link faults (F2): an `argusnet.fault-manifest`.
    /// Recorded in the replay's `meta.fault_manifest`. Link faults need
    /// `--fusion-lag-s`.
    #[arg(long)]
    faults: Option<PathBuf>,
    /// What the platforms know of the world before they sense it: `prior`
    /// (the true terrain and obstacles, as every run before) or `none` (they
    /// map it themselves with range scanners and plan only on that map).
    #[arg(long, value_enum, default_value = "prior")]
    world_knowledge: WorldKnowledgeArg,
    /// In a `--world-knowledge none` run, where follow controllers learn a
    /// target's position: `tracks` (the run's own fused tracks, the default)
    /// or `truth` (the legacy oracle, to measure what truth was worth).
    #[arg(long, value_enum)]
    pursuit: Option<PursuitArg>,
    /// In a `--world-knowledge none` run, what search drones fly: `frontier`
    /// (to the edge of their own map, the default) or `lanes` (the lanes
    /// planned before the run).
    #[arg(long, value_enum)]
    explore: Option<ExploreArg>,
    /// Save the map the platforms built as a scene package in this (new or
    /// empty) directory, when the run ends: the measured ground and the
    /// believed structures as GLB, and the map itself. `argusnet-viewer
    /// --scene DIR` opens it. Needs `--world-knowledge none`.
    #[arg(long)]
    export_scene: Option<PathBuf>,
    /// How battery energy is charged (D-3): `legacy` charges motion only, as
    /// the reference does; `physical_v1` also charges hover power for the part
    /// of each tick not spent moving.
    #[arg(long, value_enum, default_value = "legacy")]
    energy_profile: EnergyProfileArg,
    /// Model timestamped delivery (B4): fusion ingests the evidence acquired
    /// this many seconds ago. Delays plus jitter must not exceed it. Omit for
    /// the legacy same-tick path.
    #[arg(long)]
    fusion_lag_s: Option<f64>,
    /// Every source's delivery delay, unless `--source-delay` names it.
    #[arg(long, default_value_t = 0.0, requires = "fusion_lag_s")]
    delivery_delay_s: f64,
    /// A source's own delay, as `NODE_ID=SECONDS`; repeat per source.
    #[arg(long, value_parser = parse_source_delay, requires = "fusion_lag_s")]
    source_delay: Vec<(String, f64)>,
    /// Extra delay drawn uniformly per envelope (seeded), which reorders
    /// arrivals.
    #[arg(long, default_value_t = 0.0, requires = "fusion_lag_s")]
    delivery_jitter_s: f64,
    /// Envelopes one source may have queued before new ones overflow.
    #[arg(long, default_value_t = argusnet_sim::delivery::DEFAULT_QUEUE_CAPACITY, requires = "fusion_lag_s")]
    delivery_queue_capacity: usize,
    /// Record the run in this new directory: its immutable configuration, a
    /// durable journal of every committed tick, checkpoints, and a completion
    /// marker (A4). A crashed run is finished with `--recover`.
    #[arg(long, conflicts_with_all = ["resume", "reset_at", "recover"])]
    run_dir: Option<PathBuf>,
    /// With `--run-dir` or `--recover`: checkpoint every this many ticks.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    checkpoint_every: Option<u64>,
    /// Finish a crashed run from its directory: restore its latest
    /// checkpoint, cut the journal back to it, and run to completion.
    #[arg(long, conflicts_with_all = ["resume", "reset_at"])]
    recover: Option<PathBuf>,
    /// Testing aid: exit as a crash would after this many recorded ticks.
    #[arg(long, hide = true, requires = "run_dir")]
    crash_after: Option<u64>,
    /// Print nothing but errors.
    #[arg(long)]
    quiet: bool,
    /// Write the run's timing -- startup, and the tick-time distribution --
    /// as JSON. Timing is kept out of the replay so its bytes stay stable.
    #[arg(long)]
    timing_json: Option<PathBuf>,
    /// Record each stage's wall time and print a per-stage table (P4.10-pre).
    /// `--timing-json` then carries it too. Wall time stays out of the replay.
    #[arg(long)]
    stage_timing: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Executor {
    Legacy,
    Scheduled,
    Lanes,
}

/// The scheduler driving the run, if any.
enum Driver {
    Compound(LegacyLaneExecutor),
    Lanes(PhasedLaneExecutor),
}

const EXIT_FAULT: u8 = 1;
/// `--crash-after`'s exit: distinct from every real outcome.
pub(crate) const EXIT_CRASHED: i32 = 4;
const EXIT_INVALID: u8 = 2;
const EXIT_UNSUPPORTED: u8 = 3;

/// `--explore`'s values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ExploreArg {
    Frontier,
    Lanes,
}

/// `--pursuit`'s values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum PursuitArg {
    Tracks,
    Truth,
}

/// `--world-knowledge`'s values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum WorldKnowledgeArg {
    Prior,
    None,
}

/// `--energy-profile`'s values (D-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum EnergyProfileArg {
    Legacy,
    #[value(name = "physical_v1")]
    PhysicalV1,
}

/// The energy profile, delivery and faults the flags ask for, applied by
/// the same code as the viewer's in-process run.
fn run_options(args: &SimArgs) -> Result<argusnet_sim::run_options::RunOptions, String> {
    use argusnet_sim::run_options::{DeliverySeconds, RunOptions};
    Ok(RunOptions {
        energy_profile: match args.energy_profile {
            EnergyProfileArg::Legacy => argusnet_planning::battery::EnergyProfile::Legacy,
            EnergyProfileArg::PhysicalV1 => argusnet_planning::battery::EnergyProfile::PhysicalV1,
        },
        delivery: args.fusion_lag_s.map(|fusion_lag_s| DeliverySeconds {
            fusion_lag_s,
            delay_s: args.delivery_delay_s,
            source_delay_s: args.source_delay.iter().cloned().collect(),
            jitter_s: args.delivery_jitter_s,
            queue_capacity: args.delivery_queue_capacity,
        }),
        faults: args
            .faults
            .as_deref()
            .map(RunOptions::read_faults)
            .transpose()?,
        world_knowledge: match args.world_knowledge {
            WorldKnowledgeArg::Prior => argusnet_sim::belief_world::WorldKnowledge::Prior,
            WorldKnowledgeArg::None => argusnet_sim::belief_world::WorldKnowledge::NoPrior,
        },
        exploration: (args.pursuit.is_some() || args.explore.is_some()).then(|| {
            let defaults = argusnet_sim::belief_world::ExplorationConfig::default();
            argusnet_sim::belief_world::ExplorationConfig {
                pursuit: match args.pursuit {
                    Some(PursuitArg::Tracks) => argusnet_sim::belief_world::PursuitSource::Tracks,
                    Some(PursuitArg::Truth) => argusnet_sim::belief_world::PursuitSource::Truth,
                    None => defaults.pursuit,
                },
                explore: match args.explore {
                    Some(ExploreArg::Frontier) => true,
                    Some(ExploreArg::Lanes) => false,
                    None => defaults.explore,
                },
                ..defaults
            }
        }),
        // Resolved by `sim`: a recovered run takes it from its run directory.
        lane_rates: argusnet_sim::scheduler::LaneRates::default(),
    })
}

/// `NODE_ID=SECONDS` for `--source-delay`.
fn parse_source_delay(text: &str) -> Result<(String, f64), String> {
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

/// A tick's own error as itself; the executor's as a broken invariant.
fn executor_error(error: ExecutorError<RunError>) -> RunError {
    match error {
        ExecutorError::Tick(error) => error,
        other => RunError::Invariant(other.to_string()),
    }
}

fn fail(code: u8, message: impl std::fmt::Display) -> ExitCode {
    eprintln!("argusnet-rs: {message}");
    ExitCode::from(code)
}

fn percentile(sorted_ms: &[f64], fraction: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let index = ((sorted_ms.len() - 1) as f64 * fraction).round() as usize;
    sorted_ms[index]
}

/// The `tick_ms` block of `--timing-json`, from ascending tick times.
fn tick_summary(sorted_ms: &[f64]) -> serde_json::Value {
    let mean = if sorted_ms.is_empty() {
        0.0
    } else {
        sorted_ms.iter().sum::<f64>() / sorted_ms.len() as f64
    };
    serde_json::json!({
        "mean": mean,
        "p50": percentile(sorted_ms, 0.5),
        "p95": percentile(sorted_ms, 0.95),
        "p99": percentile(sorted_ms, 0.99),
        "max": sorted_ms.last().copied().unwrap_or(0.0),
    })
}

/// Read and build a config, mapping the builder's errors to exit codes.
fn built_bundle(path: &std::path::Path) -> Result<serde_json::Value, ExitCode> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| fail(EXIT_INVALID, format!("{}: {error}", path.display())))?;
    let config = ScenarioConfig::from_json(&text).map_err(|error| fail(EXIT_INVALID, error))?;
    builder::build_bundle(&config).map_err(|error| match error {
        BuildError::Unsupported(_) => fail(EXIT_UNSUPPORTED, error),
        BuildError::Invalid(_) => fail(EXIT_INVALID, error),
    })
}

fn hex_decode(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("odd-length hex".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `argusnetd`'s ingest path, in process: the same proto conversion as its
/// `convert_request`, the same `TrackingEngine::ingest_frame`, the same
/// response message -- minus the transport.
fn fuse(args: FuseArgs) -> ExitCode {
    use argusnet_proto::pb;
    use prost::Message;

    let run = || -> Result<Vec<String>, String> {
        let text = std::fs::read_to_string(&args.capture).map_err(|e| e.to_string())?;
        let capture: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let tracker = pb::TrackerConfig::decode(
            hex_decode(capture["tracker"].as_str().ok_or("no tracker")?)?.as_slice(),
        )
        .map_err(|e| e.to_string())?;
        let mut engine =
            argusnet_core::TrackingEngine::new(argusnet_proto::tracker_config_from_pb(tracker)?)?;
        let mut responses = Vec::new();
        for request in capture["requests"].as_array().ok_or("no requests")? {
            let request = pb::IngestFrameRequest::decode(
                hex_decode(request.as_str().ok_or("request is not hex")?)?.as_slice(),
            )
            .map_err(|e| e.to_string())?;
            let target_metadata = request
                .target_metadata
                .iter()
                .cloned()
                .map(argusnet_proto::target_metadata_from_pb)
                .collect();
            let safety_events = request
                .safety_events
                .iter()
                .cloned()
                .map(argusnet_proto::safety_event_from_pb)
                .collect();
            let nodes = request
                .node_states
                .into_iter()
                .map(argusnet_proto::node_state_from_pb)
                .collect::<Result<Vec<_>, _>>()?;
            let observations = request
                .observations
                .into_iter()
                .map(argusnet_proto::observation_from_pb)
                .collect::<Result<Vec<_>, _>>()?;
            let truths = request
                .truths
                .into_iter()
                .map(argusnet_proto::truth_from_pb)
                .collect::<Result<Vec<_>, _>>()?;
            let frame = engine.ingest_frame(
                request.timestamp_s,
                nodes,
                observations,
                truths,
                target_metadata,
                safety_events,
            );
            let response = pb::IngestFrameResponse {
                frame: Some(argusnet_proto::frame_to_pb_ref(&frame)),
            };
            responses.push(hex_encode(&response.encode_to_vec()));
        }
        Ok(responses)
    };
    match run() {
        Ok(responses) => {
            let document = serde_json::json!({ "responses": responses });
            match std::fs::write(&args.output, document.to_string()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => fail(EXIT_FAULT, format!("{}: {error}", args.output.display())),
            }
        }
        Err(error) => fail(EXIT_INVALID, error),
    }
}

fn build(args: BuildArgs) -> ExitCode {
    let bundle = match built_bundle(&args.config) {
        Ok(bundle) => bundle,
        Err(code) => return code,
    };
    // Compact, as `write_bundle` writes it.
    let json = match serde_json::to_string(&bundle) {
        Ok(json) => json,
        Err(error) => return fail(EXIT_FAULT, format!("serialising bundle: {error}")),
    };
    if let Err(error) = std::fs::write(&args.bundle, json) {
        return fail(EXIT_FAULT, format!("{}: {error}", args.bundle.display()));
    }
    ExitCode::SUCCESS
}

fn sim(args: SimArgs) -> ExitCode {
    let started = Instant::now();
    let flag_rates = argusnet_sim::scheduler::LaneRates {
        mapping: args.mapping_rate,
        mission: args.mission_rate,
    };
    if !flag_rates.is_empty() && !matches!(args.executor, Executor::Lanes) {
        return fail(
            EXIT_INVALID,
            "--mission-rate and --mapping-rate run a lane at its own rate; they need --executor lanes",
        );
    }
    // A recovered run takes the scheduler options its run directory recorded.
    let mut executor_kind = args.executor;
    let mut lane_rates = flag_rates;
    if args.pursuit.is_some() && matches!(args.world_knowledge, WorldKnowledgeArg::Prior) {
        return fail(
            EXIT_INVALID,
            "--pursuit chooses how a --world-knowledge none run pursues targets; \
             a prior run always pursues the true targets",
        );
    }
    if args.explore.is_some() && matches!(args.world_knowledge, WorldKnowledgeArg::Prior) {
        return fail(
            EXIT_INVALID,
            "--explore chooses what search drones fly in a --world-knowledge none run; \
             a prior run always flies its lanes",
        );
    }
    if let Some(dir) = &args.export_scene {
        if matches!(args.world_knowledge, WorldKnowledgeArg::Prior) {
            return fail(
                EXIT_INVALID,
                "--export-scene saves the map the platforms built; it needs --world-knowledge none",
            );
        }
        if std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some()) {
            return fail(
                EXIT_INVALID,
                format!(
                    "{} is not empty; export into a new directory",
                    dir.display()
                ),
            );
        }
    }
    let mut resolved_config = None;
    let text = match (&args.bundle, &args.config) {
        (Some(path), _) => match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => return fail(EXIT_INVALID, format!("{}: {error}", path.display())),
        },
        (None, Some(path)) => match built_bundle(path) {
            Ok(bundle) => {
                // Every option, defaults filled in: what was actually built.
                resolved_config = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|text| ScenarioConfig::from_json(&text).ok())
                    .and_then(|config| serde_json::to_value(config).ok());
                bundle.to_string()
            }
            Err(code) => return code,
        },
        (None, None) => match &args.recover {
            Some(path) => {
                let dir = RunDir::new(path);
                if dir.is_complete() {
                    println!("{}: already complete; nothing to recover", path.display());
                    return ExitCode::SUCCESS;
                }
                match dir.recorded() {
                    Ok((text, recorded)) => {
                        let differs = |flag: Option<Schedule>, recorded: Option<Schedule>| {
                            flag.is_some() && flag != recorded
                        };
                        if differs(args.mission_rate, recorded.mission)
                            || differs(args.mapping_rate, recorded.mapping)
                        {
                            return fail(
                                EXIT_INVALID,
                                "--mission-rate or --mapping-rate differs from the rates this \
                                 run was recorded with",
                            );
                        }
                        if !recorded.is_empty() {
                            executor_kind = Executor::Lanes;
                        }
                        lane_rates = recorded;
                        text
                    }
                    Err(error) => return fail(EXIT_INVALID, error),
                }
            }
            None => unreachable!("clap requires one of --bundle, --config and --recover"),
        },
    };
    let bundle = match ScenarioBundle::from_json(&text) {
        Ok(bundle) => bundle,
        Err(error) => return fail(EXIT_INVALID, error),
    };
    let compute = match argusnet_sim::compute::ComputePool::new(args.workers) {
        Ok(pool) => std::sync::Arc::new(pool),
        Err(error) => return fail(EXIT_INVALID, error),
    };
    let mut run = match assemble_with(&bundle, compute) {
        Ok(run) => run,
        Err(error @ BundleError::Unsupported(_)) => return fail(EXIT_UNSUPPORTED, error),
        Err(error) => return fail(EXIT_INVALID, error),
    };
    let options = match run_options(&args) {
        Ok(options) => argusnet_sim::run_options::RunOptions {
            lane_rates,
            ..options
        },
        Err(error) => return fail(EXIT_UNSUPPORTED, error),
    };
    // Refused before anything runs (B4, F2: an unsupported delivery profile
    // or an impossible fault manifest is rejected before the run).
    if let Err(error) = options.apply(&mut run, &bundle) {
        return fail(EXIT_UNSUPPORTED, error);
    }
    // `apply` declared the mission rate, before any restore: a checkpoint
    // resumes only under the rates it was taken with.
    // P4.9: "emit resolved config/version/provenance". No clock and no host,
    // so a replay's bytes depend only on what was run.
    run.extend_replay_meta([(
        "provenance".to_string(),
        serde_json::json!({
            "runtime": "argusnet-rs",
            "version": env!("CARGO_PKG_VERSION"),
            "bundle_format": bundle.format,
            "bundle_version": bundle.version,
            // The format this run's checkpoints are written in: a run with a
            // lane rate carries it in every checkpoint (format 5).
            "checkpoint_format": if !lane_rates.is_empty() {
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION_STAGE_LAG
            } else {
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION
            },
            "runtime_algorithm": argusnet_sim::snapshot::CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
            "input": if args.recover.is_some() {
                "recovered run directory"
            } else if args.config.is_some() {
                "config"
            } else {
                "bundle"
            },
            "config": resolved_config,
        }),
    )]);
    if let Some(path) = &args.resume {
        let checkpoint: SimulationCheckpoint = match std::fs::read_to_string(path)
            .map_err(|error| error.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))
        {
            Ok(checkpoint) => checkpoint,
            Err(error) => return fail(EXIT_INVALID, format!("{}: {error}", path.display())),
        };
        if let Err(error) = run.restore(&checkpoint) {
            return fail(EXIT_INVALID, format!("cannot resume: {error}"));
        }
    }
    let mut recovered_from_tick = None;
    let mut recorder = match (&args.run_dir, &args.recover) {
        (Some(path), _) => {
            let dir = RunDir::new(path);
            if let Err(error) = dir.create_with(&text, &lane_rates) {
                return fail(EXIT_INVALID, error);
            }
            match Recorder::start(
                dir,
                bundle.content_hash(),
                args.checkpoint_every,
                args.crash_after,
            ) {
                Ok(recorder) => Some(recorder),
                Err(error) => return fail(EXIT_FAULT, error),
            }
        }
        (None, Some(path)) => {
            let dir = RunDir::new(path);
            let keep = match dir.checkpoint() {
                Ok(Some(checkpoint)) => {
                    if let Err(error) = run.restore(&checkpoint) {
                        return fail(EXIT_INVALID, format!("cannot recover: {error}"));
                    }
                    recovered_from_tick = Some(checkpoint.tick);
                    checkpoint.header.journal_cursor
                }
                Ok(None) => {
                    recovered_from_tick = Some(0);
                    0
                }
                Err(error) => return fail(EXIT_INVALID, format!("cannot recover: {error}")),
            };
            match Recorder::resume(dir, bundle.content_hash(), keep, args.checkpoint_every) {
                Ok(recorder) => Some(recorder),
                Err(error) => return fail(EXIT_INVALID, error),
            }
        }
        (None, None) => None,
    };
    if let Some(tick) = recovered_from_tick {
        run.extend_replay_meta([(
            "recovery".to_string(),
            serde_json::json!({
                "recovered_from_tick": tick,
                "note": "frames before this tick were lost with the crash; the journal is complete",
            }),
        )]);
    }
    if args.stage_timing {
        run.sim.enable_stage_timing();
    }
    let startup_ms = started.elapsed().as_secs_f64() * 1e3;
    if !args.quiet {
        println!(
            "argusnet-rs {} | bundle v{} | {} | seed {} | {} ticks of {} s | from tick {}",
            env!("CARGO_PKG_VERSION"),
            bundle.version,
            bundle.scenario_name,
            bundle.seed,
            bundle.steps,
            bundle.dt_s,
            run.sim.tick()
        );
    }

    let mut tick_ms: Vec<f64> = Vec::new();
    let mut reset_pending = args.reset_at;
    let mut executor = match executor_kind {
        Executor::Legacy => None,
        Executor::Scheduled => {
            match LegacyLaneExecutor::new::<RunError>(bundle.dt_s, run.sim.tick()) {
                Ok(executor) => Some(Driver::Compound(executor)),
                Err(error) => return fail(EXIT_INVALID, error),
            }
        }
        Executor::Lanes => match PhasedLaneExecutor::new::<RunError>(bundle.dt_s, run.sim.tick()) {
            Ok(executor) => match lane_rates.apply(executor) {
                Ok(executor) => Some(Driver::Lanes(executor)),
                Err(error) => return fail(EXIT_INVALID, error),
            },
            Err(error) => return fail(EXIT_INVALID, error),
        },
    };
    loop {
        if reset_pending == Some(run.sim.tick()) {
            reset_pending = None;
            run.reset();
            match executor.as_mut() {
                Some(Driver::Compound(executor)) => {
                    executor.reset();
                }
                Some(Driver::Lanes(executor)) => {
                    executor.reset();
                }
                None => {}
            }
            tick_ms.clear();
            if !args.quiet {
                println!("reset; running again from tick 0");
            }
        }
        if let (Some(at), Some(path)) = (args.checkpoint_at, &args.checkpoint) {
            if run.sim.tick() == at {
                let written = run
                    .sim
                    .checkpoint()
                    .map_err(|error| error.to_string())
                    .and_then(|checkpoint| {
                        serde_json::to_string(&checkpoint).map_err(|error| error.to_string())
                    })
                    .and_then(|json| std::fs::write(path, json).map_err(|error| error.to_string()));
                if let Err(error) = written {
                    return fail(EXIT_FAULT, format!("writing checkpoint: {error}"));
                }
                if !args.quiet {
                    println!("checkpoint at tick {at} -> {}", path.display());
                }
            }
        }
        let tick_started = Instant::now();
        let stepped = match executor.as_mut() {
            None => run.step(),
            Some(Driver::Compound(executor)) => executor
                .step(|| {
                    let tick = run.sim.tick();
                    run.step().map(|snapshot| (tick, snapshot))
                })
                .map_err(executor_error),
            Some(Driver::Lanes(executor)) => executor.step(&mut run).map_err(executor_error),
        };
        match stepped {
            Ok(_) => {
                tick_ms.push(tick_started.elapsed().as_secs_f64() * 1e3);
                if let Some(recorder) = recorder.as_mut() {
                    if let Err(error) = recorder.after_tick(&mut run) {
                        return fail(EXIT_FAULT, error);
                    }
                }
            }
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            // A tick error names its tick and stage; truth and invariant
            // faults do not, so the tick is added for them.
            Err(error @ RunError::Tick(_)) => return fail(EXIT_FAULT, error),
            Err(error) => return fail(EXIT_FAULT, format!("tick {}: {error}", run.sim.tick())),
        }
    }

    let phase = run.mission().map(|mission| mission.phase().to_string());
    // Read before `run_to_replay` consumes the run.
    let stage_summaries = run
        .sim
        .stage_timings()
        .map(|timings| timings.summaries())
        .unwrap_or_default();
    let replay_started = Instant::now();
    let replay = match run.run_to_replay() {
        Ok(replay) => replay,
        Err(error) => return fail(EXIT_FAULT, error),
    };
    if let Some(path) = &args.replay {
        let json = match argusnet_sim::replay::to_json(&replay) {
            Ok(json) => json,
            Err(error) => return fail(EXIT_FAULT, format!("serialising replay: {error}")),
        };
        if let Err(error) = std::fs::write(path, json) {
            return fail(EXIT_FAULT, format!("{}: {error}", path.display()));
        }
    }
    let replay_write_ms = replay_started.elapsed().as_secs_f64() * 1e3;
    if let Some(dir) = &args.export_scene {
        let provenance = serde_json::json!({
            "producer": "argusnet-rs sim",
            "scenario_name": &bundle.scenario_name,
            "seed": bundle.seed,
            "ticks": replay.frames.len(),
            "replay": args.replay.as_ref().map(|path| path.display().to_string()),
        });
        let scene_id = format!("{}-self-built", bundle.scenario_name);
        match argusnet_sim::scene_export::export_replay_scene(dir, &scene_id, &replay, provenance) {
            Ok(export) => {
                if !args.quiet {
                    println!(
                        "map ({} cells known, {} structure) -> {}",
                        export.known_cells,
                        export.structure_cells,
                        export.root.display()
                    );
                }
            }
            Err(error) => return fail(EXIT_FAULT, format!("exporting the map: {error}")),
        }
    }
    if let Some(recorder) = &recorder {
        let summary = serde_json::json!({
            "journal_records": recorder.cursor(),
            "recovered_from_tick": recovered_from_tick,
            "replay": args.replay.as_ref().map(|path| path.display().to_string()),
        });
        if let Err(error) = recorder.dir.mark_complete(&summary) {
            return fail(EXIT_FAULT, format!("marking the run complete: {error}"));
        }
    }
    let mut sorted = tick_ms.clone();
    sorted.sort_by(f64::total_cmp);
    if let Some(path) = &args.timing_json {
        let mut timing = serde_json::json!({
            "startup_ms": startup_ms,
            "ticks": sorted.len(),
            "tick_ms": tick_summary(&sorted),
            "replay_write_ms": replay_write_ms,
            "total_ms": started.elapsed().as_secs_f64() * 1e3,
            "build": if cfg!(debug_assertions) { "debug" } else { "release" },
        });
        if args.stage_timing {
            timing["stages"] = stage_summaries
                .iter()
                .map(|(stage, summary)| (stage.to_string(), serde_json::json!(summary)))
                .collect::<serde_json::Map<_, _>>()
                .into();
        }
        if let Err(error) = std::fs::write(path, timing.to_string()) {
            return fail(EXIT_FAULT, format!("{}: {error}", path.display()));
        }
    }
    if !args.quiet {
        println!(
            "{} ticks | startup {:.1} ms | tick p50 {:.3} ms, p95 {:.3} ms, max {:.3} ms{}",
            tick_ms.len(),
            startup_ms,
            percentile(&sorted, 0.5),
            percentile(&sorted, 0.95),
            sorted.last().copied().unwrap_or(0.0),
            phase.map_or(String::new(), |phase| format!(" | mission {phase}"))
        );
        if let Some(path) = &args.replay {
            println!(
                "replay ({} frames) -> {}",
                replay.frames.len(),
                path.display()
            );
        }
        if args.stage_timing {
            print!("{}", stage_table(&stage_summaries, replay_write_ms));
        }
    }
    ExitCode::SUCCESS
}

/// The `--stage-timing` table: each stage's share of the summed stage time.
fn stage_table(
    summaries: &[(
        argusnet_sim::stage::Stage,
        argusnet_sim::timing::TimingSummary,
    )],
    replay_write_ms: f64,
) -> String {
    let total: f64 = summaries.iter().map(|(_, summary)| summary.total_ms).sum();
    let mut table = format!(
        "{:<32} {:>7} {:>10} {:>10} {:>10} {:>10} {:>7}\n",
        "stage", "ticks", "p50 ms", "p95 ms", "p99 ms", "total ms", "share"
    );
    for (stage, summary) in summaries {
        let share = if total > 0.0 {
            100.0 * summary.total_ms / total
        } else {
            0.0
        };
        table.push_str(&format!(
            "{:<32} {:>7} {:>10.4} {:>10.4} {:>10.4} {:>10.2} {:>6.1}%\n",
            stage.to_string(),
            summary.count,
            summary.p50_ms,
            summary.p95_ms,
            summary.p99_ms,
            summary.total_ms,
            share
        ));
    }
    table.push_str(&format!(
        "replay build and write: {replay_write_ms:.2} ms\n"
    ));
    table
}

fn build_scene(args: BuildSceneArgs) -> ExitCode {
    let text = match std::fs::read_to_string(&args.replay) {
        Ok(text) => text,
        Err(error) => return fail(EXIT_INVALID, format!("{}: {error}", args.replay.display())),
    };
    let replay: serde_json::Value = match serde_json::from_str(&text) {
        Ok(replay) => replay,
        Err(error) => return fail(EXIT_INVALID, format!("{}: {error}", args.replay.display())),
    };
    let meta = replay.get("meta").cloned().unwrap_or_default();
    let provenance = serde_json::json!({
        "replay_source": args.replay.display().to_string(),
        "environment_bundle": null,
    });
    match argusnet_sim::scene_build::build_scene(&args.output, &meta, Some(&text), provenance) {
        Ok(built) => {
            println!(
                "scene ({}; {} triangles) -> {}",
                built.layers.join(", "),
                built.triangles,
                built.root.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => fail(EXIT_INVALID, format!("building the scene: {error}")),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => sim(args),
        Command::Build(args) => build(args),
        Command::BuildScene(args) => build_scene(args),
        Command::Fuse(args) => fuse(args),
        Command::Version => {
            println!(
                "argusnet-rs {}\nscenario config format {} v{}\nscenario bundle format {} v{}\ncheckpoint format {} ({} with lane rates), runtime algorithm {}, seed version {}",
                env!("CARGO_PKG_VERSION"),
                builder::CONFIG_FORMAT,
                builder::CONFIG_VERSION,
                argusnet_sim::scenario::BUNDLE_FORMAT,
                argusnet_sim::scenario::BUNDLE_VERSION,
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION,
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION_STAGE_LAG,
                argusnet_sim::snapshot::CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
                argusnet_sim::snapshot::CheckpointHeader::SEED_VERSION,
            );
            ExitCode::SUCCESS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tick_summary_reads_its_percentiles_off_the_sorted_times() {
        // 1..=101 ms: index = round(100 * fraction), so each percentile is
        // one more than its own number.
        let sorted: Vec<f64> = (1..=101).map(f64::from).collect();
        let summary = tick_summary(&sorted);
        assert_eq!(summary["p50"], 51.0);
        assert_eq!(summary["p95"], 96.0);
        assert_eq!(summary["p99"], 100.0);
        assert_eq!(summary["max"], 101.0);
        assert_eq!(summary["mean"], 51.0);
    }

    #[test]
    fn an_empty_run_summarises_to_zeros() {
        let summary = tick_summary(&[]);
        for key in ["mean", "p50", "p95", "p99", "max"] {
            assert_eq!(summary[key], 0.0, "{key}");
        }
    }

    #[test]
    fn hex_decoding_refuses_what_is_not_bytes() {
        assert_eq!(hex_decode("0aff").unwrap(), vec![0x0a, 0xff]);
        assert!(hex_decode("abc").is_err(), "odd length");
        assert!(hex_decode("zz").is_err(), "not hex");
        assert_eq!(
            hex_decode(&hex_encode(&[0, 1, 254, 255])).unwrap(),
            vec![0, 1, 254, 255]
        );
    }
}
