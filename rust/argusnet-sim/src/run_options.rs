//! The run options shared by every front end that assembles a run: the
//! energy profile (D-3), the planning policy (P1), timestamped delivery (B4), scheduled faults (F2),
//! what the platforms know of the world before sensing it, and the lanes'
//! own rates (B2.4 mission, B2.5 mapping: `scheduled_v1`).
//!
//! `argusnet-rs sim` and the viewer's in-process run (`--run`) both apply them
//! through [`RunOptions::apply`], so a run commanded from the viewer is the
//! same run, byte for byte, as the CLI's with the same options. Each option is
//! validated against the assembled run before any tick; on a refusal the
//! caller discards the run rather than starting it.

use std::collections::BTreeMap;

pub use argusnet_planning::allocation::PlanningPolicy;
pub use argusnet_planning::battery::EnergyProfile;
use serde::{Deserialize, Serialize};

use crate::belief_world::{ExplorationConfig, WorldKnowledge};
use crate::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
use crate::faults::FaultManifest;
use crate::scenario::{AssembledRun, ScenarioBundle};
use crate::scheduler::{describe, rate_text, LaneRates};

/// Timestamped delivery in seconds, as a user states it. Converted to ticks
/// against the run's timestep by [`DeliveryProfile::from_seconds`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliverySeconds {
    pub fusion_lag_s: f64,
    #[serde(default)]
    pub delay_s: f64,
    #[serde(default)]
    pub source_delay_s: BTreeMap<String, f64>,
    #[serde(default)]
    pub jitter_s: f64,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

fn default_queue_capacity() -> usize {
    DEFAULT_QUEUE_CAPACITY
}

impl DeliverySeconds {
    pub fn with_lag(fusion_lag_s: f64) -> Self {
        Self {
            fusion_lag_s,
            delay_s: 0.0,
            source_delay_s: BTreeMap::new(),
            jitter_s: 0.0,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
        }
    }

    /// The profile in ticks for `bundle`, seeded by the bundle's seed.
    pub fn profile(&self, bundle: &ScenarioBundle) -> Result<DeliveryProfile, String> {
        DeliveryProfile::from_seconds(
            bundle.dt_s,
            self.fusion_lag_s,
            self.delay_s,
            &self.source_delay_s,
            self.jitter_s,
            self.queue_capacity,
            bundle.seed,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunOptions {
    pub energy_profile: EnergyProfile,
    /// Who allocates the mission's POIs (P1); the reference by default.
    pub planning_policy: PlanningPolicy,
    /// `None` is the legacy same-tick path.
    pub delivery: Option<DeliverySeconds>,
    pub faults: Option<FaultManifest>,
    /// What the platforms know of the world before sensing it.
    pub world_knowledge: WorldKnowledge,
    /// How a `none` run senses and maps; the defaults when `None`.
    pub exploration: Option<ExplorationConfig>,
    /// The lanes' own rates (B2.4, B2.5); empty runs every lane every tick.
    /// The driver must step the run with a
    /// [`crate::scheduler::PhasedLaneExecutor`] at these rates
    /// ([`LaneRates::apply`]): `apply` only declares them on the simulation
    /// and records them.
    pub lane_rates: LaneRates,
    /// Sensors that capture at their own rates (P4.10b), by node; empty is
    /// every sensor every tick. Independent of the executor: stage 6 decides
    /// at each tick which sensors are due.
    pub sensor_rates: std::collections::BTreeMap<String, crate::scheduler::Schedule>,
}

impl RunOptions {
    /// Whether these are the legacy options: nothing to apply, nothing
    /// recorded.
    pub fn is_legacy(&self) -> bool {
        self.energy_profile.is_legacy()
            && self.planning_policy.is_reference()
            && self.delivery.is_none()
            && self.faults.is_none()
            && self.world_knowledge == WorldKnowledge::Prior
            && self.lane_rates.is_empty()
            && self.sensor_rates.is_empty()
    }

    /// Install the options on `run`, assembled from `bundle`, and record what
    /// was installed in the replay's metadata (`energy_profile`,
    /// `delivery_profile`; the fault manifest records itself). Faults go last:
    /// link faults are validated against the delivery profile.
    pub fn apply(&self, run: &mut AssembledRun, bundle: &ScenarioBundle) -> Result<(), String> {
        if !self.energy_profile.is_legacy() {
            run.battery_mut().set_energy_profile(self.energy_profile);
            run.extend_replay_meta([(
                "energy_profile".to_string(),
                serde_json::json!(self.energy_profile.as_str()),
            )]);
        }
        if !self.planning_policy.is_reference() {
            // A run with no mission driver has no POIs to allocate; it is
            // still recorded, so the replay says what was asked for.
            run.set_planning_policy(self.planning_policy)?;
            run.extend_replay_meta([(
                "planning_policy".to_string(),
                serde_json::json!(self.planning_policy.as_str()),
            )]);
        }
        if let Some(delivery) = &self.delivery {
            let profile = delivery.profile(bundle)?;
            run.set_delivery(profile.clone())?;
            run.extend_replay_meta([(
                "delivery_profile".to_string(),
                serde_json::to_value(&profile).expect("profile serialises"),
            )]);
        }
        if !self.sensor_rates.is_empty() {
            let capture =
                crate::observe::CaptureRates::new(self.sensor_rates.clone(), bundle.dt_s)?;
            run.set_sensor_rates(capture)?;
            // Additive: a run with every sensor at the tick carries no key.
            run.extend_replay_meta([(
                "sensor_rates".to_string(),
                serde_json::Value::Object(
                    self.sensor_rates
                        .iter()
                        .map(|(id, rate)| (id.clone(), serde_json::json!(describe(*rate))))
                        .collect(),
                ),
            )]);
        }
        if let Some(manifest) = &self.faults {
            run.set_faults(manifest.clone())?;
        }
        run.set_world_knowledge(
            self.world_knowledge,
            self.exploration.clone().unwrap_or_default(),
        )?;
        if !self.lane_rates.is_empty() {
            let mut recorded = serde_json::Map::new();
            for (stage, name, rate) in self.lane_rates.rated() {
                // Declared before any restore: a checkpoint resumes only
                // under the rates it was taken with.
                run.sim.declare_lane_rate(stage, rate_text(rate));
                recorded.insert(name.to_string(), serde_json::json!(describe(rate)));
            }
            // Additive: a run at the legacy rates carries neither key, so its
            // replay is byte for byte what it was.
            run.extend_replay_meta([
                (
                    "execution_model".to_string(),
                    serde_json::json!("scheduled_v1"),
                ),
                (
                    "lane_rates".to_string(),
                    serde_json::Value::Object(recorded),
                ),
            ]);
        }
        Ok(())
    }

    /// `--sensor-rate` values as a table, refusing a sensor named twice.
    pub fn sensor_rate_table(
        pairs: &[(String, crate::scheduler::Schedule)],
    ) -> Result<std::collections::BTreeMap<String, crate::scheduler::Schedule>, String> {
        let mut table = std::collections::BTreeMap::new();
        for (node_id, rate) in pairs {
            if table.insert(node_id.clone(), *rate).is_some() {
                return Err(format!("--sensor-rate names {node_id} more than once"));
            }
        }
        Ok(table)
    }

    /// A fault manifest from a file, checked for format and version.
    pub fn read_faults(path: &std::path::Path) -> Result<FaultManifest, String> {
        const MAX_FAULT_BYTES: u64 = 256 * 1024 * 1024;
        let len = std::fs::metadata(path)
            .map_err(|error| format!("{}: {error}", path.display()))?
            .len();
        if len > MAX_FAULT_BYTES {
            return Err(format!(
                "{} is {len} bytes, over the {MAX_FAULT_BYTES} byte limit",
                path.display()
            ));
        }
        std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))
            .and_then(|text| FaultManifest::from_json(&text))
    }
}

/// `NODE_ID=NUM/DEN` for `--sensor-rate`: that sensor captures at `NUM/DEN`
/// Hz (P4.10b).
pub fn parse_sensor_rate(text: &str) -> Result<(String, crate::scheduler::Schedule), String> {
    let (node_id, rate) = text
        .split_once('=')
        .ok_or_else(|| format!("expected NODE_ID=NUM/DEN, got {text:?}"))?;
    if node_id.is_empty() {
        return Err("the node id is empty".into());
    }
    Ok((node_id.to_string(), crate::scheduler::parse_rate(rate)?))
}
