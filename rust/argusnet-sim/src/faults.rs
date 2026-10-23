//! Scheduled sensor and link faults (F2).
//!
//! A [`FaultManifest`] lists faults by simulated time. It is recorded in the
//! replay (`meta.fault_manifest`), so a run is reproducible from its seed,
//! configuration and manifest:
//!
//! - **`sensor_outage`**: a node's sensor produces nothing over `[start_s,
//!   end_s)`, then comes back with a fresh bias state (it restarted).
//! - **`link_delay`**: a node's evidence takes `delay_s` to deliver over
//!   `[start_s, end_s)` instead of its profile delay. Needs timestamped
//!   delivery (B4), and must fit within the fusion lag, like any profile.
//! - **`source_restart`**: at `at_s` the node's transport restarts: a new
//!   source epoch, so evidence it sent before is dropped as stale.
//! - **`calibration_drift`**: a node's bearing calibration drifts at
//!   `rate_rad_per_s` over `[start_s, end_s)` and stays where it got to:
//!   nothing recalibrates it. Added to the sensor's bias walk.
//!
//! Outages, delays and drift offsets are a pure function of the tick and are re-applied
//! idempotently before every tick, so a checkpoint needs no extra state to
//! resume them. A restart happens once, at its tick; its effect (the epoch)
//! lives in the delivery network's own checkpoint.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::observe::SensorSpec;

pub const MANIFEST_FORMAT: &str = "argusnet.fault-manifest";
pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
    SensorOutage {
        node_id: String,
        start_s: f64,
        end_s: f64,
    },
    LinkDelay {
        node_id: String,
        start_s: f64,
        end_s: f64,
        delay_s: f64,
    },
    SourceRestart {
        node_id: String,
        at_s: f64,
    },
    CalibrationDrift {
        node_id: String,
        start_s: f64,
        end_s: f64,
        rate_rad_per_s: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultManifest {
    pub format: String,
    pub version: u32,
    pub faults: Vec<Fault>,
}

impl FaultManifest {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let manifest: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if manifest.format != MANIFEST_FORMAT || manifest.version != MANIFEST_VERSION {
            return Err(format!(
                "expected {MANIFEST_FORMAT} version {MANIFEST_VERSION}, found {} version {}",
                manifest.format, manifest.version
            ));
        }
        Ok(manifest)
    }
}

/// Seconds to the tick that contains them: `[k*dt, (k+1)*dt)` is tick `k`.
fn tick_of(seconds: f64, dt_s: f64) -> u64 {
    ((seconds / dt_s) + 1e-9).floor().max(0.0) as u64
}

/// A validated manifest, in ticks, with what each outage restores.
#[derive(Debug, Clone)]
pub struct FaultSchedule {
    manifest: FaultManifest,
    /// `(node, start tick, end tick)`, end exclusive.
    outages: Vec<(String, u64, u64)>,
    /// `(node, start tick, end tick, delay ticks)`.
    delays: Vec<(String, u64, u64, u64)>,
    restarts: Vec<(String, u64)>,
    /// `(node, start tick, end tick, radians per tick)`.
    drifts: Vec<(String, u64, u64, f64)>,
    /// Each outage node's sensor as installed, to put back.
    specs: BTreeMap<String, SensorSpec>,
}

impl FaultSchedule {
    /// Validate `manifest` against the run: every node must have a sensor,
    /// every interval must be ordered, link faults need delivery modelling,
    /// and a delay must fit within the fusion lag (`delay_fits`).
    pub fn new(
        manifest: FaultManifest,
        dt_s: f64,
        sensors: &BTreeMap<String, SensorSpec>,
        delivery: Option<&dyn Fn(u64) -> Result<(), String>>,
    ) -> Result<Self, String> {
        let known = |node: &str| -> Result<(), String> {
            if sensors.contains_key(node) {
                Ok(())
            } else {
                Err(format!(
                    "fault names {node:?}, which has no sensor in this run"
                ))
            }
        };
        let interval = |node: &str, start: f64, end: f64| -> Result<(u64, u64), String> {
            if !(start.is_finite() && end.is_finite() && start >= 0.0 && end > start) {
                return Err(format!(
                    "{node}: the interval [{start}, {end}) is empty or invalid"
                ));
            }
            Ok((tick_of(start, dt_s), tick_of(end, dt_s)))
        };
        let ticks = |seconds: f64| -> Result<u64, String> {
            if !(seconds.is_finite() && seconds >= 0.0) {
                return Err(format!("{seconds} is not a non-negative number of seconds"));
            }
            Ok((seconds / dt_s - 1e-9).ceil().max(0.0) as u64)
        };
        let mut schedule = Self {
            manifest: manifest.clone(),
            outages: Vec::new(),
            delays: Vec::new(),
            restarts: Vec::new(),
            drifts: Vec::new(),
            specs: BTreeMap::new(),
        };
        for fault in &manifest.faults {
            match fault {
                Fault::SensorOutage {
                    node_id,
                    start_s,
                    end_s,
                } => {
                    known(node_id)?;
                    let (start, end) = interval(node_id, *start_s, *end_s)?;
                    schedule.outages.push((node_id.clone(), start, end));
                    schedule
                        .specs
                        .insert(node_id.clone(), sensors[node_id.as_str()]);
                }
                Fault::LinkDelay {
                    node_id,
                    start_s,
                    end_s,
                    delay_s,
                } => {
                    known(node_id)?;
                    let fits = delivery.ok_or_else(|| {
                        format!("link_delay on {node_id} needs timestamped delivery (a fusion lag)")
                    })?;
                    let (start, end) = interval(node_id, *start_s, *end_s)?;
                    let delay = ticks(*delay_s)?;
                    fits(delay).map_err(|error| format!("link_delay on {node_id}: {error}"))?;
                    schedule.delays.push((node_id.clone(), start, end, delay));
                }
                Fault::SourceRestart { node_id, at_s } => {
                    known(node_id)?;
                    if delivery.is_none() {
                        return Err(format!(
                            "source_restart on {node_id} needs timestamped delivery (a fusion lag)"
                        ));
                    }
                    if !(at_s.is_finite() && *at_s >= 0.0) {
                        return Err(format!("{node_id}: restart time {at_s} is invalid"));
                    }
                    schedule
                        .restarts
                        .push((node_id.clone(), tick_of(*at_s, dt_s)));
                }
                Fault::CalibrationDrift {
                    node_id,
                    start_s,
                    end_s,
                    rate_rad_per_s,
                } => {
                    known(node_id)?;
                    let (start, end) = interval(node_id, *start_s, *end_s)?;
                    if !(rate_rad_per_s.is_finite() && *rate_rad_per_s != 0.0) {
                        return Err(format!(
                            "calibration_drift on {node_id}: the rate {rate_rad_per_s} rad/s \
                             is not a finite non-zero number"
                        ));
                    }
                    schedule
                        .drifts
                        .push((node_id.clone(), start, end, rate_rad_per_s * dt_s));
                }
            }
        }
        Ok(schedule)
    }

    pub fn manifest(&self) -> &FaultManifest {
        &self.manifest
    }

    /// Every drifted node's calibration offset at `tick`, radians: the drift
    /// accumulated over the ticks of its intervals before `tick`, held after
    /// an interval ends. Zero before the first. Several faults on one node
    /// add.
    pub fn bias_offsets(&self, tick: u64) -> BTreeMap<&str, f64> {
        let mut out: BTreeMap<&str, f64> = BTreeMap::new();
        for (node, start, end, per_tick) in &self.drifts {
            let elapsed = tick.clamp(*start, *end) - start;
            *out.entry(node.as_str()).or_insert(0.0) += per_tick * elapsed as f64;
        }
        out
    }

    /// The faults in force at `tick`, in manifest order: an interval fault
    /// over its interval, a restart at its tick. A drift is listed while it
    /// drifts; what it left behind is in [`Self::bias_offsets`].
    pub fn active_at(&self, tick: u64, dt_s: f64) -> Vec<&Fault> {
        let within =
            |start: f64, end: f64| (tick_of(start, dt_s)..tick_of(end, dt_s)).contains(&tick);
        self.manifest
            .faults
            .iter()
            .filter(|fault| match fault {
                Fault::SensorOutage { start_s, end_s, .. }
                | Fault::LinkDelay { start_s, end_s, .. }
                | Fault::CalibrationDrift { start_s, end_s, .. } => within(*start_s, *end_s),
                Fault::SourceRestart { at_s, .. } => tick_of(*at_s, dt_s) == tick,
            })
            .collect()
    }

    /// Whether `node`'s sensor is out at `tick`.
    pub fn sensor_out(&self, node: &str, tick: u64) -> bool {
        self.outages
            .iter()
            .any(|(n, start, end)| n == node && (*start..*end).contains(&tick))
    }

    /// The outage nodes, each with its sensor and whether it is out at `tick`.
    pub fn sensor_states(&self, tick: u64) -> Vec<(&str, SensorSpec, bool)> {
        self.specs
            .iter()
            .map(|(node, spec)| (node.as_str(), *spec, self.sensor_out(node, tick)))
            .collect()
    }

    /// Every delayed node's override at `tick`: `Some` inside a fault,
    /// `None` outside (the profile delay applies).
    pub fn delay_overrides(&self, tick: u64) -> BTreeMap<&str, Option<u64>> {
        let mut out: BTreeMap<&str, Option<u64>> = BTreeMap::new();
        for (node, start, end, delay) in &self.delays {
            let entry = out.entry(node.as_str()).or_insert(None);
            if (*start..*end).contains(&tick) {
                *entry = Some(entry.map_or(*delay, |d| d.max(*delay)));
            }
        }
        out
    }

    /// The nodes whose transport restarts at `tick`.
    pub fn restarts_at(&self, tick: u64) -> Vec<&str> {
        self.restarts
            .iter()
            .filter(|(_, at)| *at == tick)
            .map(|(node, _)| node.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(faults: Vec<Fault>) -> FaultManifest {
        FaultManifest {
            format: MANIFEST_FORMAT.into(),
            version: MANIFEST_VERSION,
            faults,
        }
    }

    fn sensors() -> BTreeMap<String, SensorSpec> {
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), SensorSpec::default());
        map
    }

    #[test]
    fn outages_are_half_open_intervals_in_ticks() {
        let s = FaultSchedule::new(
            manifest(vec![Fault::SensorOutage {
                node_id: "a".into(),
                start_s: 1.0,
                end_s: 2.0,
            }]),
            0.5,
            &sensors(),
            None,
        )
        .unwrap();
        let out: Vec<bool> = (0..6).map(|t| s.sensor_out("a", t)).collect();
        assert_eq!(out, [false, false, true, true, false, false]);
    }

    #[test]
    fn a_manifest_naming_an_unknown_node_or_needing_delivery_is_refused() {
        let unknown = manifest(vec![Fault::SensorOutage {
            node_id: "zz".into(),
            start_s: 0.0,
            end_s: 1.0,
        }]);
        assert!(FaultSchedule::new(unknown, 0.5, &sensors(), None).is_err());
        let delay = manifest(vec![Fault::LinkDelay {
            node_id: "a".into(),
            start_s: 0.0,
            end_s: 1.0,
            delay_s: 0.5,
        }]);
        let error = FaultSchedule::new(delay.clone(), 0.5, &sensors(), None).unwrap_err();
        assert!(error.contains("needs timestamped delivery"), "{error}");
        let too_slow = |ticks: u64| {
            if ticks > 0 {
                Err("exceeds the fusion lag".to_string())
            } else {
                Ok(())
            }
        };
        let error = FaultSchedule::new(delay, 0.5, &sensors(), Some(&too_slow)).unwrap_err();
        assert!(error.contains("exceeds the fusion lag"), "{error}");
        let backwards = manifest(vec![Fault::SensorOutage {
            node_id: "a".into(),
            start_s: 2.0,
            end_s: 1.0,
        }]);
        assert!(FaultSchedule::new(backwards, 0.5, &sensors(), None).is_err());
    }

    #[test]
    fn the_manifest_round_trips_and_checks_its_format() {
        let m = manifest(vec![Fault::SourceRestart {
            node_id: "a".into(),
            at_s: 3.0,
        }]);
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains("\"kind\":\"source_restart\""));
        assert_eq!(FaultManifest::from_json(&text).unwrap(), m);
        let wrong = text.replace(MANIFEST_FORMAT, "something-else");
        assert!(FaultManifest::from_json(&wrong).is_err());
    }
}
