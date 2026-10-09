//! The run options shared by every front end that assembles a run: the
//! energy profile (D-3), timestamped delivery (B4) and scheduled faults (F2).
//!
//! `argusnet-rs sim` and the viewer's in-process run (`--run`) both apply them
//! through [`RunOptions::apply`], so a run commanded from the viewer is the
//! same run, byte for byte, as the CLI's with the same options. Each option is
//! validated against the assembled run before any tick; on a refusal the
//! caller discards the run rather than starting it.

use std::collections::BTreeMap;

pub use argusnet_planning::battery::EnergyProfile;
use serde::{Deserialize, Serialize};

use crate::delivery::{DeliveryProfile, DEFAULT_QUEUE_CAPACITY};
use crate::faults::FaultManifest;
use crate::scenario::{AssembledRun, ScenarioBundle};

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
    /// `None` is the legacy same-tick path.
    pub delivery: Option<DeliverySeconds>,
    pub faults: Option<FaultManifest>,
}

impl RunOptions {
    /// Whether these are the legacy options: nothing to apply, nothing
    /// recorded.
    pub fn is_legacy(&self) -> bool {
        self.energy_profile.is_legacy() && self.delivery.is_none() && self.faults.is_none()
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
        if let Some(delivery) = &self.delivery {
            let profile = delivery.profile(bundle)?;
            run.set_delivery(profile.clone())?;
            run.extend_replay_meta([(
                "delivery_profile".to_string(),
                serde_json::to_value(&profile).expect("profile serialises"),
            )]);
        }
        if let Some(manifest) = &self.faults {
            run.set_faults(manifest.clone())?;
        }
        Ok(())
    }

    /// A fault manifest from a file, checked for format and version.
    pub fn read_faults(path: &std::path::Path) -> Result<FaultManifest, String> {
        std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))
            .and_then(|text| FaultManifest::from_json(&text))
    }
}
