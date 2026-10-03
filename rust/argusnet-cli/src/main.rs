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
use argusnet_sim::scenario::{assemble, BundleError, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;
use argusnet_sim::snapshot::SimulationCheckpoint;
use clap::{Parser, Subcommand};

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
enum Command {
    /// Run a scenario bundle (or a scenario config) to completion and write a
    /// replay.
    Sim(SimArgs),
    /// Build the scenario bundle an `argusnet.scenario-config` describes.
    Build(BuildArgs),
    /// Run captured `IngestFrameRequest`s through the in-process fusion engine
    /// and write its `IngestFrameResponse`s -- the direct-fusion parity check
    /// against `argusnetd` (P4.9c).
    #[command(hide = true)]
    Fuse(FuseArgs),
    /// Print the runtime, bundle and checkpoint format versions.
    Version,
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
#[command(group(clap::ArgGroup::new("input").required(true).args(["bundle", "config"])))]
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

const EXIT_FAULT: u8 = 1;
const EXIT_INVALID: u8 = 2;
const EXIT_UNSUPPORTED: u8 = 3;

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
        (None, None) => unreachable!("clap requires one of --bundle and --config"),
    };
    let bundle = match ScenarioBundle::from_json(&text) {
        Ok(bundle) => bundle,
        Err(error) => return fail(EXIT_INVALID, error),
    };
    let mut run = match assemble(&bundle) {
        Ok(run) => run,
        Err(error @ BundleError::Unsupported(_)) => return fail(EXIT_UNSUPPORTED, error),
        Err(error) => return fail(EXIT_INVALID, error),
    };
    // P4.9: "emit resolved config/version/provenance". No clock and no host,
    // so a replay's bytes depend only on what was run.
    run.extend_replay_meta([(
        "provenance".to_string(),
        serde_json::json!({
            "runtime": "argusnet-rs",
            "version": env!("CARGO_PKG_VERSION"),
            "bundle_format": bundle.format,
            "bundle_version": bundle.version,
            "checkpoint_format": argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION,
            "runtime_algorithm": argusnet_sim::snapshot::CheckpointHeader::RUNTIME_ALGORITHM_VERSION,
            "input": if args.config.is_some() { "config" } else { "bundle" },
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
        if let Err(error) = run.sim.restore(&checkpoint) {
            return fail(EXIT_INVALID, format!("cannot resume: {error}"));
        }
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
    loop {
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
        match run.step() {
            Ok(_) => tick_ms.push(tick_started.elapsed().as_secs_f64() * 1e3),
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            // A tick error names its tick and stage; truth and invariant
            // faults do not, so the tick is added for them.
            Err(error @ RunError::Tick(_)) => return fail(EXIT_FAULT, error),
            Err(error) => return fail(EXIT_FAULT, format!("tick {}: {error}", run.sim.tick())),
        }
    }

    let phase = run
        .mission
        .as_ref()
        .map(|mission| mission.phase().to_string());
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => sim(args),
        Command::Build(args) => build(args),
        Command::Fuse(args) => fuse(args),
        Command::Version => {
            println!(
                "argusnet-rs {}\nscenario config format {} v{}\nscenario bundle format {} v{}\ncheckpoint format {}, runtime algorithm {}, seed version {}",
                env!("CARGO_PKG_VERSION"),
                builder::CONFIG_FORMAT,
                builder::CONFIG_VERSION,
                argusnet_sim::scenario::BUNDLE_FORMAT,
                argusnet_sim::scenario::BUNDLE_VERSION,
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION,
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
