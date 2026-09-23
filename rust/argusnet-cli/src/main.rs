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
    /// Print the runtime, bundle and checkpoint format versions.
    Version,
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
    let text = match (&args.bundle, &args.config) {
        (Some(path), _) => match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => return fail(EXIT_INVALID, format!("{}: {error}", path.display())),
        },
        (None, Some(path)) => match built_bundle(path) {
            Ok(bundle) => bundle.to_string(),
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
    if !args.quiet {
        let mut sorted = tick_ms.clone();
        sorted.sort_by(f64::total_cmp);
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
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => sim(args),
        Command::Build(args) => build(args),
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
