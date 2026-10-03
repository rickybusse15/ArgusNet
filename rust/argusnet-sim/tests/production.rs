//! A1.3: a production run refuses to build with a required stage missing.
//!
//! `Simulation::production` is the only constructor the CLI uses (through
//! `scenario::assemble`). These pin that a missing required stage is refused
//! by name, that stage 5 (VIO) is not required, and that a mode's extra
//! requirements are enforced.

use argusnet_sim::config::SimulationConfig;
use argusnet_sim::simulation::{BuildError, Simulation, StageHandler, TickContext, UnportedStage};
use argusnet_sim::stage::{Stage, StageError, StageOutcome};

fn config() -> SimulationConfig {
    SimulationConfig {
        seed: 5,
        dt_s: 1.0,
        steps: Some(10),
        run_id: "production".to_string(),
    }
}

/// A stage that does something, as far as the builder can tell.
#[derive(Debug)]
struct Ported;

impl StageHandler for Ported {
    fn run(&mut self, _context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        Ok(StageOutcome::NoWork)
    }
}

fn with_core_except(missing: Option<Stage>) -> argusnet_sim::simulation::ProductionBuilder {
    let mut builder = Simulation::production(config()).expect("config is valid");
    for stage in Stage::PRODUCTION_CORE {
        if Some(stage) != missing {
            builder.install(stage, Box::new(Ported)).expect("installs");
        }
    }
    builder
}

#[test]
fn each_missing_core_stage_is_refused_by_name() {
    for stage in Stage::PRODUCTION_CORE {
        let error = with_core_except(Some(stage))
            .build()
            .expect_err("a core stage is missing");
        assert_eq!(error, BuildError::MissingStage { stage });
        assert!(
            error.to_string().contains(&stage.to_string()),
            "{error} does not name {stage}"
        );
    }
}

#[test]
fn an_explicitly_unported_core_stage_is_refused() {
    let mut builder = with_core_except(None);
    builder
        .install(Stage::UpdateFusion, Box::new(UnportedStage))
        .expect("installs");
    assert_eq!(
        builder.build().expect_err("stage 7 is unported"),
        BuildError::MissingStage {
            stage: Stage::UpdateFusion
        }
    );
}

#[test]
fn the_core_builds_without_vio() {
    let builder = with_core_except(None);
    assert!(!builder
        .required()
        .any(|stage| stage == Stage::UpdateVioAndPose));
    let sim = builder.build().expect("the core is installed");
    assert_eq!(sim.tick(), 0);
}

#[test]
fn a_modes_extra_requirement_is_enforced() {
    let mut builder = with_core_except(None);
    builder.require(Stage::SafetyMonitor);
    assert_eq!(
        builder.build().expect_err("safety was required"),
        BuildError::MissingStage {
            stage: Stage::SafetyMonitor
        }
    );
    let mut builder = with_core_except(None);
    builder.require(Stage::SafetyMonitor);
    builder
        .install(Stage::SafetyMonitor, Box::new(Ported))
        .expect("installs");
    builder.build().expect("safety is installed");
}
