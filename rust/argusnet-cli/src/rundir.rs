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
use argusnet_sim::run_options::PlanningPolicy;
use argusnet_sim::scenario::AssembledRun;
use argusnet_sim::scheduler::{parse_rate, rate_text, LaneRates};
use argusnet_sim::snapshot::SimulationCheckpoint;

pub const RUN_FORMAT: &str = "argusnet-run/1";
/// The format of a run under a planning policy other than the reference
/// (P1). A build that predates the policy reads only `argusnet-run/1`, so it
/// refuses such a run by name instead of recovering it under the reference
/// planner. A reference run keeps `argusnet-run/1`.
pub const RUN_FORMAT_POLICY: &str = "argusnet-run/2";

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
    /// `NUM/DEN` Hz when the mission lane ran at its own rate
    /// (`--mission-rate`, B2.4), so recovery continues the same run. Such a
    /// run records checkpoint format 5, which a build without the field
    /// refuses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mission_rate: Option<String>,
    /// The same for the mapping lane (`--mapping-rate`, B2.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mapping_rate: Option<String>,
    /// How the run allocates POIs (`--planning-policy`, P1). Absent for the
    /// reference, so a reference run's `run.json` is unchanged.
    #[serde(default, skip_serializing_if = "PlanningPolicy::is_reference")]
    planning_policy: PlanningPolicy,
}

/// What a run directory says the run was started with.
pub struct Recorded {
    pub bundle: String,
    pub rates: LaneRates,
    pub policy: PlanningPolicy,
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
    #[cfg(test)]
    pub fn create(&self, bundle_text: &str) -> Result<(), String> {
        self.create_with(
            bundle_text,
            &LaneRates::default(),
            PlanningPolicy::Reference,
        )
    }

    /// [`Self::create`], recording the lanes' own rates if they have any and
    /// a planning policy that is not the reference.
    pub fn create_with(
        &self,
        bundle_text: &str,
        rates: &LaneRates,
        policy: PlanningPolicy,
    ) -> Result<(), String> {
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
            format: if policy.is_reference() {
                RUN_FORMAT
            } else {
                RUN_FORMAT_POLICY
            }
            .to_string(),
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            checkpoint_format: if !rates.is_empty() {
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION_STAGE_LAG
            } else {
                argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION
            },
            bundle_fnv1a: format!("{:016x}", fnv1a(bundle_text.as_bytes())),
            bundle: bundle_text.to_string(),
            mission_rate: rates.mission.map(rate_text),
            mapping_rate: rates.mapping.map(rate_text),
            planning_policy: policy,
        };
        let json = serde_json::to_vec(&config).map_err(|e| e.to_string())?;
        write_atomic(&self.file("run.json"), &json).map_err(|e| e.to_string())
    }

    /// The bundle text the run was started with, checked.
    #[cfg(test)]
    pub fn bundle_text(&self) -> Result<String, String> {
        self.recorded().map(|recorded| recorded.bundle)
    }

    /// The bundle text, the lanes' own rates and the planning policy the run
    /// was started with, checked.
    pub fn recorded(&self) -> Result<Recorded, String> {
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
        let expected = if config.planning_policy.is_reference() {
            RUN_FORMAT
        } else {
            RUN_FORMAT_POLICY
        };
        if config.format != RUN_FORMAT && config.format != RUN_FORMAT_POLICY {
            return Err(incomplete(format!(
                "run.json is format {}, this build reads {RUN_FORMAT} and {RUN_FORMAT_POLICY}",
                config.format
            )));
        }
        if config.format != expected {
            return Err(incomplete(format!(
                "run.json is format {} but records planning policy {}, which is format \
                 {expected}",
                config.format,
                config.planning_policy.as_str()
            )));
        }
        // Recorded for the recovery that reads it back: a run whose checkpoints
        // this build cannot read is refused here, by name, rather than failing
        // later on the checkpoint's own header.
        let checkpoint_format = if config.mission_rate.is_some() || config.mapping_rate.is_some() {
            argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION_STAGE_LAG
        } else {
            argusnet_sim::snapshot::CheckpointHeader::FORMAT_VERSION
        };
        if config.checkpoint_format != checkpoint_format {
            return Err(incomplete(format!(
                "run.json records checkpoint format {}, this build reads {checkpoint_format}",
                config.checkpoint_format
            )));
        }
        if config.bundle_fnv1a != format!("{:016x}", fnv1a(config.bundle.as_bytes())) {
            return Err(incomplete(
                "run.json's bundle does not match its hash".to_string(),
            ));
        }
        let rate = |text: Option<String>, flag: &str| {
            text.as_deref()
                .map(parse_rate)
                .transpose()
                .map_err(|error| incomplete(format!("run.json's {flag}: {error}")))
        };
        let rates = LaneRates {
            mapping: rate(config.mapping_rate, "mapping rate")?,
            mission: rate(config.mission_rate, "mission rate")?,
        };
        Ok(Recorded {
            bundle: config.bundle,
            rates,
            policy: config.planning_policy,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A run directory whose run.json is `config` with `edit` applied.
    fn run_dir(name: &str, edit: impl FnOnce(&mut serde_json::Value)) -> RunDir {
        let path =
            std::env::temp_dir().join(format!("argusnet-rundir-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let dir = RunDir::new(&path);
        dir.create("{\"bundle\": true}").unwrap();
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.file("run.json")).unwrap()).unwrap();
        edit(&mut config);
        std::fs::write(dir.file("run.json"), config.to_string()).unwrap();
        dir
    }

    #[test]
    fn a_run_as_written_reads_back() {
        let dir = run_dir("ok", |_| {});
        assert_eq!(dir.bundle_text().unwrap(), "{\"bundle\": true}");
    }

    #[test]
    fn an_unsupported_run_format_is_refused_by_name() {
        let dir = run_dir("format", |config| {
            config["format"] = "argusnet-run/9".into()
        });
        let error = dir.bundle_text().unwrap_err();
        assert!(error.contains("format argusnet-run/9"), "{error}");
    }

    /// P1: a run under another planning policy is format 2, which a build
    /// predating the policy refuses; the format and the policy must agree.
    #[test]
    fn a_policy_run_is_format_2_and_must_say_so() {
        let path =
            std::env::temp_dir().join(format!("argusnet-rundir-policy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let dir = RunDir::new(&path);
        dir.create_with("{}", &LaneRates::default(), PlanningPolicy::EfficientV1)
            .unwrap();
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.file("run.json")).unwrap()).unwrap();
        assert_eq!(config["format"], RUN_FORMAT_POLICY);
        assert_eq!(dir.recorded().unwrap().policy, PlanningPolicy::EfficientV1);

        let mismatched = run_dir("policy-format", |config| {
            config["planning_policy"] = "efficient_v1".into()
        });
        let error = mismatched.bundle_text().unwrap_err();
        assert!(
            error.contains("records planning policy efficient_v1"),
            "{error}"
        );
        let mismatched = run_dir("reference-format", |config| {
            config["format"] = RUN_FORMAT_POLICY.into()
        });
        let error = mismatched.bundle_text().unwrap_err();
        assert!(error.contains("planning policy reference"), "{error}");
    }

    #[test]
    fn a_checkpoint_format_this_build_cannot_read_is_refused() {
        let dir = run_dir("checkpoint", |config| {
            config["checkpoint_format"] = 99.into()
        });
        let error = dir.bundle_text().unwrap_err();
        assert!(error.contains("checkpoint format 99"), "{error}");
    }
}
