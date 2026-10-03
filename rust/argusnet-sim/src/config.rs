//! Run configuration and its validation.
//!
//! Validation happens once, at construction, and returns every problem it
//! finds rather than the first. A run rejected here has mutated nothing, which
//! is the property §4 asks for: *"Invalid commands are rejected before
//! mutation."* The same reasoning applies to the config that starts the run.

use std::fmt;

use crate::clock::{Timestep, TimestepError};

/// Everything a run needs before its first tick.
///
/// Deliberately small for P4.1. It grows as stages move across; what matters
/// now is that it is validated, hashable into a checkpoint header, and carries
/// the seed the determinism contract is written against.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationConfig {
    /// Base seed for every derived stream. See `deterministic::stable_seed`.
    pub seed: u32,
    /// Fixed timestep, validated into [`Timestep`].
    pub dt_s: f64,
    /// Number of ticks the run should execute. `None` runs until stopped,
    /// which is what interactive mode wants.
    pub steps: Option<u64>,
    /// Names this run in acknowledgements and checkpoint headers.
    pub run_id: String,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            dt_s: 0.1,
            steps: None,
            run_id: "run".to_string(),
        }
    }
}

/// A single reason a configuration was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigProblem {
    Timestep(TimestepError),
    /// A run of zero ticks is almost always a mistake in the caller, and it
    /// produces an artifact indistinguishable from a run that crashed before
    /// its first tick.
    ZeroSteps,
    EmptyRunId,
}

impl fmt::Display for ConfigProblem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigProblem::Timestep(error) => write!(formatter, "dt_s: {error}"),
            ConfigProblem::ZeroSteps => write!(formatter, "steps: a run must have at least 1 tick"),
            ConfigProblem::EmptyRunId => write!(formatter, "run_id: must not be empty"),
        }
    }
}

/// Every problem found, not just the first.
///
/// Reporting one at a time turns fixing a configuration into a guessing game,
/// and this is the first thing a new caller hits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub problems: Vec<ConfigProblem>,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid simulation config: ")?;
        for (index, problem) in self.problems.iter().enumerate() {
            if index > 0 {
                write!(formatter, "; ")?;
            }
            write!(formatter, "{problem}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigError {}

/// A configuration that has passed validation.
///
/// The type is the evidence: a stage holding one of these does not re-check
/// the timestep, and there is no way to construct it except through
/// [`ValidatedConfig::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedConfig {
    config: SimulationConfig,
    timestep: Timestep,
}

impl ValidatedConfig {
    pub fn new(config: SimulationConfig) -> Result<Self, ConfigError> {
        let mut problems = Vec::new();

        let timestep = match Timestep::new(config.dt_s) {
            Ok(timestep) => Some(timestep),
            Err(error) => {
                problems.push(ConfigProblem::Timestep(error));
                None
            }
        };
        if config.steps == Some(0) {
            problems.push(ConfigProblem::ZeroSteps);
        }
        if config.run_id.trim().is_empty() {
            problems.push(ConfigProblem::EmptyRunId);
        }

        match (problems.is_empty(), timestep) {
            (true, Some(timestep)) => Ok(Self { config, timestep }),
            _ => Err(ConfigError { problems }),
        }
    }

    pub fn timestep(&self) -> Timestep {
        self.timestep
    }

    pub fn seed(&self) -> u32 {
        self.config.seed
    }

    pub fn steps(&self) -> Option<u64> {
        self.config.steps
    }

    pub fn run_id(&self) -> &str {
        &self.config.run_id
    }

    pub fn raw(&self) -> &SimulationConfig {
        &self.config
    }

    /// Content hash for the checkpoint header.
    ///
    /// §6 requires a config hash so a checkpoint cannot be restored into a run
    /// configured differently. FNV-1a over the field renderings: not a
    /// cryptographic commitment, and it does not need to be — it detects
    /// accidental mismatch, which is the actual failure. A checkpoint file is
    /// not a trust boundary; if it becomes one, this needs to be a real digest
    /// and this comment is the place that says so.
    pub fn content_hash(&self) -> u64 {
        fnv1a(
            format!(
                "seed={};dt_s={:?};steps={:?};run_id={}",
                self.config.seed, self.config.dt_s, self.config.steps, self.config.run_id
            )
            .as_bytes(),
        )
    }
}

/// 64-bit FNV-1a: the checkpoint header's accidental-mismatch hash. See
/// [`ValidatedConfig::content_hash`] for why it is not a digest.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> SimulationConfig {
        SimulationConfig {
            seed: 7,
            dt_s: 0.1,
            steps: Some(100),
            run_id: "scan".to_string(),
        }
    }

    #[test]
    fn a_valid_config_is_accepted() {
        let validated = ValidatedConfig::new(valid()).expect("valid");
        assert_eq!(validated.seed(), 7);
        assert_eq!(validated.timestep().seconds(), 0.1);
        assert_eq!(validated.steps(), Some(100));
    }

    #[test]
    fn every_problem_is_reported_not_just_the_first() {
        let error = ValidatedConfig::new(SimulationConfig {
            seed: 0,
            dt_s: -1.0,
            steps: Some(0),
            run_id: "   ".to_string(),
        })
        .expect_err("three problems");
        assert_eq!(error.problems.len(), 3, "{error}");
        assert!(error.problems.contains(&ConfigProblem::ZeroSteps));
        assert!(error.problems.contains(&ConfigProblem::EmptyRunId));
    }

    #[test]
    fn an_unbounded_run_is_allowed_but_a_zero_length_one_is_not() {
        assert!(ValidatedConfig::new(SimulationConfig {
            steps: None,
            ..valid()
        })
        .is_ok());
        assert!(ValidatedConfig::new(SimulationConfig {
            steps: Some(0),
            ..valid()
        })
        .is_err());
    }

    #[test]
    fn the_content_hash_moves_when_any_field_does() {
        let base = ValidatedConfig::new(valid()).expect("valid").content_hash();
        let cases = [
            SimulationConfig { seed: 8, ..valid() },
            SimulationConfig {
                dt_s: 0.2,
                ..valid()
            },
            SimulationConfig {
                steps: Some(101),
                ..valid()
            },
            SimulationConfig {
                run_id: "scan2".to_string(),
                ..valid()
            },
        ];
        for case in cases {
            let hash = ValidatedConfig::new(case.clone())
                .expect("valid")
                .content_hash();
            assert_ne!(hash, base, "hash did not move for {case:?}");
        }
    }

    #[test]
    fn the_content_hash_is_stable_for_the_same_config() {
        let first = ValidatedConfig::new(valid()).expect("valid").content_hash();
        let second = ValidatedConfig::new(valid()).expect("valid").content_hash();
        assert_eq!(first, second);
    }
}
