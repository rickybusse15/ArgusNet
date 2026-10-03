//! The crash-recoverable run directory (A4).
//!
//! ```text
//! run.json         immutable run configuration: the bundle, its hash, versions
//! journal.jsonl    one durable line per committed tick (argusnet_sim::journal)
//! checkpoint.json  the latest checkpoint, replaced atomically; its header's
//!                  journal_cursor says how much of the journal it covers
//! complete.json    written last; without it the run is incomplete
//! ```
//!
//! A run that crashes leaves no `complete.json`. `argusnet-rs sim --recover`
//! restores the latest checkpoint, cuts the journal back to that checkpoint's
//! cursor (dropping a torn tail and any ticks after it), and runs on. The
//! journal then equals an uninterrupted run's. The replay only holds the ticks
//! after the checkpoint, and says so: a crash never turns an incomplete
//! artifact into a complete one.

use std::path::{Path, PathBuf};

use argusnet_sim::config::fnv1a;
use argusnet_sim::journal::{JournalError, JournalWriter};
use argusnet_sim::scenario::AssembledRun;
use argusnet_sim::snapshot::SimulationCheckpoint;

pub const RUN_FORMAT: &str = "argusnet-run/1";

/// What `run.json` holds. Written once, never modified.
#[derive(serde::Serialize, serde::Deserialize)]
struct RunConfig {
    format: String,
    runtime_version: String,
    checkpoint_format: u32,
    /// FNV-1a of `bundle`'s bytes: a damaged file is refused, not run.
    bundle_fnv1a: String,
    /// The bundle exactly as it was run.
    bundle: String,
}

pub struct RunDir {
    path: PathBuf,
}

/// Replace `path` with `contents` atomically: a crash leaves the old file or
/// the new one, never half of either.
fn write_atomic(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, contents)?;
    std::fs::File::open(&temporary)?.sync_all()?;
    std::fs::rename(&temporary, path)
}

impl RunDir {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    pub fn journal_path(&self) -> PathBuf {
        self.file("journal.jsonl")
    }

    /// Start a run here: the directory must be new or empty.
    pub fn create(&self, bundle_text: &str) -> Result<(), String> {
        std::fs::create_dir_all(&self.path).map_err(|e| e.to_string())?;
        let occupied = std::fs::read_dir(&self.path)
            .map_err(|e| e.to_string())?
            .next()
            .is_some();
        if occupied {
            return Err(format!(
                "{} is not empty: a run directory is never reused (recover it with --recover)",
                self.path.display()
            ));
        }
        let config = RunConfig {
            format: RUN_FORMAT.to_string(),
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            checkpoint_format: argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION,
            bundle_fnv1a: format!("{:016x}", fnv1a(bundle_text.as_bytes())),
            bundle: bundle_text.to_string(),
        };
        let json = serde_json::to_vec(&config).map_err(|e| e.to_string())?;
        write_atomic(&self.file("run.json"), &json).map_err(|e| e.to_string())
    }

    /// The bundle text the run was started with, checked.
    pub fn bundle_text(&self) -> Result<String, String> {
        let incomplete = |detail: String| {
            format!(
                "{}: incomplete run artifact, cannot recover: {detail}",
                self.path.display()
            )
        };
        let text = std::fs::read_to_string(self.file("run.json"))
            .map_err(|e| incomplete(format!("run.json: {e}")))?;
        let config: RunConfig = serde_json::from_str(&text)
            .map_err(|e| incomplete(format!("run.json is damaged: {e}")))?;
        if config.format != RUN_FORMAT {
            return Err(incomplete(format!(
                "run.json is format {}, this build reads {RUN_FORMAT}",
                config.format
            )));
        }
        if config.bundle_fnv1a != format!("{:016x}", fnv1a(config.bundle.as_bytes())) {
            return Err(incomplete(
                "run.json's bundle does not match its hash".to_string(),
            ));
        }
        Ok(config.bundle)
    }

    pub fn is_complete(&self) -> bool {
        self.file("complete.json").exists()
    }

    pub fn checkpoint(&self) -> Result<Option<SimulationCheckpoint>, String> {
        let path = self.file("checkpoint.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("checkpoint.json is damaged: {e}"))
    }

    pub fn write_checkpoint(&self, checkpoint: &SimulationCheckpoint) -> Result<(), String> {
        let json = serde_json::to_vec(checkpoint).map_err(|e| e.to_string())?;
        write_atomic(&self.file("checkpoint.json"), &json).map_err(|e| e.to_string())
    }

    pub fn mark_complete(&self, summary: &serde_json::Value) -> Result<(), String> {
        write_atomic(&self.file("complete.json"), summary.to_string().as_bytes())
            .map_err(|e| e.to_string())
    }
}

/// Journals every committed tick, checkpoints on a cadence.
pub struct Recorder {
    pub dir: RunDir,
    journal: JournalWriter,
    every: Option<u64>,
    crash_after: Option<u64>,
    recorded: u64,
}

impl Recorder {
    pub fn start(
        dir: RunDir,
        run: u64,
        every: Option<u64>,
        crash_after: Option<u64>,
    ) -> Result<Self, String> {
        let journal = JournalWriter::create(&dir.journal_path(), run).map_err(|e| e.to_string())?;
        Ok(Self {
            dir,
            journal,
            every,
            crash_after,
            recorded: 0,
        })
    }

    /// Continue a crashed run's journal after the first `keep` records.
    pub fn resume(dir: RunDir, run: u64, keep: u64, every: Option<u64>) -> Result<Self, String> {
        let journal = JournalWriter::resume(&dir.journal_path(), run, keep)
            .map_err(|error: JournalError| format!("cannot recover: {error}"))?;
        Ok(Self {
            dir,
            journal,
            every,
            crash_after: None,
            recorded: 0,
        })
    }

    pub fn cursor(&self) -> u64 {
        self.journal.cursor()
    }

    /// After a committed tick: make it durable, then maybe checkpoint. The
    /// run must not advance if this fails.
    pub fn after_tick(&mut self, run: &mut AssembledRun) -> Result<(), String> {
        let tick = run.sim.tick() - 1;
        let timestamp_s = tick as f64 * run.dt_s();
        self.journal
            .append(run.sim.epoch(), tick, timestamp_s, run.journal_body())
            .map_err(|error| {
                format!("journal write failed after tick {tick}; stopped at that boundary: {error}")
            })?;
        run.sim.set_journal_cursor(self.journal.cursor());
        self.recorded += 1;
        if self
            .every
            .is_some_and(|every| run.sim.tick().is_multiple_of(every))
        {
            let checkpoint = run.sim.checkpoint().map_err(|e| e.to_string())?;
            self.dir.write_checkpoint(&checkpoint)?;
        }
        if self.crash_after == Some(self.recorded) {
            // Testing aid: die as a crash would, mid-run, with no completion.
            std::process::exit(crate::EXIT_CRASHED);
        }
        Ok(())
    }
}
