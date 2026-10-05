//! Timestamped sensor delivery (B4, P4.10b).
//!
//! In the legacy tick an observation is captured, delivered and fused in the
//! same tick. A real sensor's evidence arrives later than it was captured, by
//! an amount that depends on the source, and sometimes out of order. This
//! module models that between stage 6 (capture) and stage 7 (fusion), without
//! changing the estimator:
//!
//! - Every observation is wrapped in a [`SensorEnvelope`]: its **source**,
//!   per-source **sequence**, the source's **epoch** (a restart begins a new
//!   one), the tick it was **acquired**, the tick it becomes **available**, and
//!   the tick it was **received**. Capture and delivery stay distinct.
//! - A [`DeliveryProfile`] sets each source's delay, plus a seeded jitter.
//!   Jitter reorders arrivals; the draws are keyed per `(source, epoch,
//!   sequence)`, so a run is deterministic for a fixed seed and profile.
//! - **Fusion runs at a fixed lag.** The tracker is a forward-only filter: it
//!   cannot take a measurement older than its last update. So at tick `t`
//!   fusion ingests exactly the evidence **acquired** at `t - L`, where `L` is
//!   the profile's fusion lag, and the evidence keeps its acquisition time.
//!   Every acquisition tick is ingested once, with or without evidence, so
//!   tracks age by elapsed acquisition time through empty intervals, and
//!   nothing is fused out of order. A profile whose delays can exceed `L` is
//!   refused before the run, rather than dropping evidence by construction.
//! - **Nothing is lost silently.** An envelope that cannot be fused is dropped
//!   with a [`DropReason`]: `Late` (it arrived after its acquisition tick
//!   passed the fusion horizon), `Duplicate` (a repeated source sequence),
//!   `StaleEpoch` (sent before its source restarted) or `Overflow` (the
//!   source's bounded queue was full). Fusion appends each drop to the frame's
//!   rejected observations, so the replay records it.
//!
//! With no profile the fusion stage keeps the legacy path untouched. A profile
//! of all zeros goes through this network and reproduces the legacy run
//! exactly, which the tests use as the proof that the network itself changes
//! nothing it should not.

use std::collections::{BTreeMap, BTreeSet};

use argusnet_types::{BearingObservation, ObservationRejection, Vector3};
use serde::{Deserialize, Serialize};

use crate::deterministic::{DeterministicRng, SeedPart};

/// Sequences remembered per source for duplicate detection.
const DEDUP_WINDOW: u64 = 4096;

/// Default bound on one source's envelopes in flight or awaiting fusion.
pub const DEFAULT_QUEUE_CAPACITY: usize = 4096;

/// How each source's evidence is delayed, in ticks of the run's timestep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryProfile {
    /// Fusion ingests the evidence acquired this many ticks ago.
    pub fusion_lag_ticks: u64,
    /// Delay for a source with no entry of its own.
    pub default_delay_ticks: u64,
    pub per_source_delay_ticks: BTreeMap<String, u64>,
    /// Each envelope's extra delay is drawn uniformly from `0..=jitter_ticks`.
    pub jitter_ticks: u64,
    pub queue_capacity: usize,
    pub seed: u32,
}

impl DeliveryProfile {
    /// A profile from seconds, rounded up to whole ticks of `dt_s`. Refused
    /// when evidence could arrive after fusion has passed its acquisition
    /// time: that profile would drop evidence by construction.
    pub fn from_seconds(
        dt_s: f64,
        fusion_lag_s: f64,
        default_delay_s: f64,
        per_source_delay_s: &BTreeMap<String, f64>,
        jitter_s: f64,
        queue_capacity: usize,
        seed: u32,
    ) -> Result<Self, String> {
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return Err(format!("the timestep must be positive, got {dt_s}"));
        }
        let ticks = |what: &str, seconds: f64| -> Result<u64, String> {
            if !(seconds.is_finite() && seconds >= 0.0) {
                return Err(format!(
                    "{what} must be a non-negative number of seconds, got {seconds}"
                ));
            }
            // A hair under a whole tick still needs that tick.
            Ok((seconds / dt_s - 1e-9).ceil().max(0.0) as u64)
        };
        let profile = Self {
            fusion_lag_ticks: ticks("the fusion lag", fusion_lag_s)?,
            default_delay_ticks: ticks("the delivery delay", default_delay_s)?,
            per_source_delay_ticks: per_source_delay_s
                .iter()
                .map(|(source, s)| Ok((source.clone(), ticks(&format!("{source}'s delay"), *s)?)))
                .collect::<Result<_, String>>()?,
            jitter_ticks: ticks("the delivery jitter", jitter_s)?,
            queue_capacity: queue_capacity.max(1),
            seed,
        };
        profile.validate(dt_s)?;
        Ok(profile)
    }

    /// The longest any envelope can take to arrive.
    pub fn max_delay_ticks(&self) -> u64 {
        self.per_source_delay_ticks
            .values()
            .copied()
            .chain(std::iter::once(self.default_delay_ticks))
            .max()
            .unwrap_or(0)
            + self.jitter_ticks
    }

    pub fn validate(&self, dt_s: f64) -> Result<(), String> {
        if self.max_delay_ticks() > self.fusion_lag_ticks {
            return Err(format!(
                "unsupported delivery profile: evidence can arrive up to {} tick(s) ({:.3} s) \
                 after capture, but fusion waits only {} tick(s) ({:.3} s); raise the fusion \
                 lag to at least the largest delay plus jitter",
                self.max_delay_ticks(),
                self.max_delay_ticks() as f64 * dt_s,
                self.fusion_lag_ticks,
                self.fusion_lag_ticks as f64 * dt_s,
            ));
        }
        Ok(())
    }

    fn delay_for(&self, source: &str) -> u64 {
        self.per_source_delay_ticks
            .get(source)
            .copied()
            .unwrap_or(self.default_delay_ticks)
    }
}

/// An observation as it travels, serialisable for checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredObservation {
    pub node_id: String,
    pub target_id: String,
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub bearing_std_rad: f64,
    pub timestamp_s: f64,
    pub confidence: f64,
}

impl From<&BearingObservation> for StoredObservation {
    fn from(o: &BearingObservation) -> Self {
        Self {
            node_id: o.node_id.clone(),
            target_id: o.target_id.clone(),
            origin: [o.origin.x, o.origin.y, o.origin.z],
            direction: [o.direction.x, o.direction.y, o.direction.z],
            bearing_std_rad: o.bearing_std_rad,
            timestamp_s: o.timestamp_s,
            confidence: o.confidence,
        }
    }
}

impl From<&StoredObservation> for BearingObservation {
    fn from(o: &StoredObservation) -> Self {
        Self {
            node_id: o.node_id.clone(),
            target_id: o.target_id.clone(),
            origin: Vector3::new(o.origin[0], o.origin[1], o.origin[2]),
            direction: Vector3::new(o.direction[0], o.direction[1], o.direction[2]),
            bearing_std_rad: o.bearing_std_rad,
            timestamp_s: o.timestamp_s,
            confidence: o.confidence,
        }
    }
}

/// One observation in transit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorEnvelope {
    pub source_id: String,
    pub sequence: u64,
    pub source_epoch: u32,
    pub acquisition_tick: u64,
    pub availability_tick: u64,
    /// Set when it reaches the fusion side.
    pub receipt_tick: Option<u64>,
    /// Capture order across all sources: the tie-break that keeps a tick's
    /// evidence in the order it was produced.
    pub capture_index: u64,
    pub observation: StoredObservation,
}

/// Why an envelope was not fused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum DropReason {
    Late { horizon_tick: u64 },
    Duplicate,
    StaleEpoch { current_epoch: u32 },
    Overflow { capacity: usize },
}

impl DropReason {
    pub fn code(&self) -> &'static str {
        match self {
            DropReason::Late { .. } => "delivery_late",
            DropReason::Duplicate => "delivery_duplicate",
            DropReason::StaleEpoch { .. } => "delivery_stale_epoch",
            DropReason::Overflow { .. } => "delivery_overflow",
        }
    }
}

/// A dropped envelope and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dropped {
    pub envelope: SensorEnvelope,
    pub at_tick: u64,
    pub reason: DropReason,
}

impl Dropped {
    /// The drop as the replay's rejected-observation record.
    pub fn rejection(&self, dt_s: f64) -> ObservationRejection {
        let e = &self.envelope;
        let detail = match &self.reason {
            DropReason::Late { horizon_tick } => format!(
                "acquired at tick {}, received at tick {}, after fusion had passed tick {}",
                e.acquisition_tick, self.at_tick, horizon_tick
            ),
            DropReason::Duplicate => format!(
                "sequence {} of epoch {} from {} was already delivered",
                e.sequence, e.source_epoch, e.source_id
            ),
            DropReason::StaleEpoch { current_epoch } => format!(
                "sent in epoch {} but {} has restarted (epoch {current_epoch})",
                e.source_epoch, e.source_id
            ),
            DropReason::Overflow { capacity } => {
                format!("{} already had {capacity} envelope(s) queued", e.source_id)
            }
        };
        ObservationRejection {
            node_id: e.observation.node_id.clone(),
            target_id: e.observation.target_id.clone(),
            timestamp_s: self.at_tick as f64 * dt_s,
            reason: self.reason.code().to_string(),
            detail,
            origin: Some(Vector3::new(
                e.observation.origin[0],
                e.observation.origin[1],
                e.observation.origin[2],
            )),
            attempted_point: None,
            closest_point: None,
            blocker_type: String::new(),
            first_hit_range_m: None,
        }
    }
}

/// Totals over the run, by outcome.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeliveryTotals {
    pub captured: u64,
    pub fused: u64,
    pub dropped: BTreeMap<String, u64>,
}

/// The network between capture and fusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryNetwork {
    profile: DeliveryProfile,
    /// Not yet available, keyed `(availability, capture index)`.
    #[serde(with = "pairs")]
    in_flight: BTreeMap<(u64, u64), SensorEnvelope>,
    /// Received, waiting for fusion to reach their acquisition tick, keyed
    /// `(acquisition, capture index)`.
    #[serde(with = "pairs")]
    received: BTreeMap<(u64, u64), SensorEnvelope>,
    next_sequence: BTreeMap<String, u64>,
    epochs: BTreeMap<String, u32>,
    /// Delivered `(epoch, sequence)` per source, bounded by [`DEDUP_WINDOW`].
    delivered: BTreeMap<String, BTreeSet<(u32, u64)>>,
    queued: BTreeMap<String, usize>,
    next_capture_index: u64,
    /// Each tick's timestamp, kept until fusion has passed it.
    tick_times: BTreeMap<u64, f64>,
    /// Temporary per-source delays (F2's link faults), outside the profile so
    /// a checkpoint's profile still matches the run's. Set every tick from
    /// the fault schedule.
    #[serde(default)]
    delay_override: BTreeMap<String, u64>,
    /// This tick's drops; taken by fusion.
    drops: Vec<Dropped>,
    totals: DeliveryTotals,
}

impl DeliveryNetwork {
    pub fn new(profile: DeliveryProfile) -> Self {
        Self {
            profile,
            in_flight: BTreeMap::new(),
            received: BTreeMap::new(),
            next_sequence: BTreeMap::new(),
            epochs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            queued: BTreeMap::new(),
            next_capture_index: 0,
            tick_times: BTreeMap::new(),
            delay_override: BTreeMap::new(),
            drops: Vec::new(),
            totals: DeliveryTotals::default(),
        }
    }

    pub fn profile(&self) -> &DeliveryProfile {
        &self.profile
    }

    pub fn totals(&self) -> &DeliveryTotals {
        &self.totals
    }

    /// Back to the start of a run, with the same profile.
    pub fn reset(&mut self) {
        *self = Self::new(self.profile.clone());
    }

    /// Delay `source` by `ticks` instead of its profile delay, until set back
    /// to `None`. Refused when the delay plus jitter would exceed the fusion
    /// lag: that evidence would be late by construction.
    pub fn set_delay_override(&mut self, source: &str, ticks: Option<u64>) -> Result<(), String> {
        match ticks {
            Some(ticks) => {
                if ticks + self.profile.jitter_ticks > self.profile.fusion_lag_ticks {
                    return Err(format!(
                        "a delay of {ticks} tick(s) for {source} plus {} of jitter exceeds the \
                         fusion lag of {} tick(s)",
                        self.profile.jitter_ticks, self.profile.fusion_lag_ticks
                    ));
                }
                self.delay_override.insert(source.to_string(), ticks);
            }
            None => {
                self.delay_override.remove(source);
            }
        }
        Ok(())
    }

    /// The source restarted: its sequence starts again in a new epoch, and
    /// anything it sent before is stale.
    pub fn restart_source(&mut self, source: &str) {
        *self.epochs.entry(source.to_string()).or_insert(0) += 1;
        self.next_sequence.insert(source.to_string(), 0);
    }

    /// Wrap this tick's observations and send them on their way.
    pub fn capture(&mut self, tick: u64, timestamp_s: f64, observations: &[BearingObservation]) {
        self.tick_times.insert(tick, timestamp_s);
        for observation in observations {
            let source = observation.node_id.clone();
            let sequence = {
                let next = self.next_sequence.entry(source.clone()).or_insert(0);
                let sequence = *next;
                *next += 1;
                sequence
            };
            let epoch = self.epochs.get(&source).copied().unwrap_or(0);
            let jitter = if self.profile.jitter_ticks == 0 {
                0
            } else {
                let mut rng = DeterministicRng::keyed(
                    self.profile.seed,
                    &[
                        SeedPart::Str("delivery"),
                        SeedPart::Str(&source),
                        SeedPart::Int(i64::from(epoch)),
                        SeedPart::Int(sequence as i64),
                    ],
                );
                rng.next_u64() % (self.profile.jitter_ticks + 1)
            };
            let envelope = SensorEnvelope {
                availability_tick: tick
                    + self
                        .delay_override
                        .get(&source)
                        .copied()
                        .unwrap_or_else(|| self.profile.delay_for(&source))
                    + jitter,
                source_id: source,
                sequence,
                source_epoch: epoch,
                acquisition_tick: tick,
                receipt_tick: None,
                capture_index: self.next_capture_index,
                observation: StoredObservation::from(observation),
            };
            self.next_capture_index += 1;
            self.totals.captured += 1;
            self.send(envelope, tick);
        }
    }

    /// Put an envelope in flight, as a source's transport would. `capture`
    /// uses it; tests use it to inject duplicates and stale epochs.
    pub fn send(&mut self, envelope: SensorEnvelope, now_tick: u64) {
        let queued = self.queued.entry(envelope.source_id.clone()).or_insert(0);
        if *queued >= self.profile.queue_capacity {
            self.drop_envelope(
                envelope,
                now_tick,
                DropReason::Overflow {
                    capacity: self.profile.queue_capacity,
                },
            );
            return;
        }
        *queued += 1;
        self.in_flight.insert(
            (envelope.availability_tick, envelope.capture_index),
            envelope,
        );
    }

    fn drop_envelope(&mut self, envelope: SensorEnvelope, at_tick: u64, reason: DropReason) {
        *self
            .totals
            .dropped
            .entry(reason.code().to_string())
            .or_insert(0) += 1;
        self.drops.push(Dropped {
            envelope,
            at_tick,
            reason,
        });
    }

    fn dequeue(&mut self, source: &str) {
        if let Some(queued) = self.queued.get_mut(source) {
            *queued = queued.saturating_sub(1);
        }
    }

    /// The acquisition tick fusion ingests at `now`, once there is one.
    pub fn horizon(&self, now_tick: u64) -> Option<u64> {
        now_tick.checked_sub(self.profile.fusion_lag_ticks)
    }

    /// Receive what has arrived by `now`, and hand fusion the evidence
    /// acquired at the horizon, in capture order, with the horizon's time.
    /// `None` until the first horizon tick.
    pub fn deliver(&mut self, now_tick: u64) -> Option<(f64, Vec<BearingObservation>)> {
        let horizon = self.horizon(now_tick);
        // Arrivals, in availability then capture order.
        let due: Vec<(u64, u64)> = self
            .in_flight
            .range(..=(now_tick, u64::MAX))
            .map(|(key, _)| *key)
            .collect();
        for key in due {
            let mut envelope = self.in_flight.remove(&key).expect("listed above");
            envelope.receipt_tick = Some(now_tick);
            let current_epoch = self.epochs.get(&envelope.source_id).copied().unwrap_or(0);
            let seen = self
                .delivered
                .entry(envelope.source_id.clone())
                .or_default();
            let reason = if envelope.source_epoch < current_epoch {
                Some(DropReason::StaleEpoch { current_epoch })
            } else if seen.contains(&(envelope.source_epoch, envelope.sequence)) {
                Some(DropReason::Duplicate)
            } else if horizon.is_some_and(|h| envelope.acquisition_tick < h) {
                Some(DropReason::Late {
                    horizon_tick: horizon.expect("checked"),
                })
            } else {
                None
            };
            if let Some(reason) = reason {
                let source = envelope.source_id.clone();
                self.dequeue(&source);
                self.drop_envelope(envelope, now_tick, reason);
                continue;
            }
            seen.insert((envelope.source_epoch, envelope.sequence));
            if seen.len() as u64 > DEDUP_WINDOW {
                let oldest = *seen.iter().next().expect("non-empty");
                seen.remove(&oldest);
            }
            self.received.insert(
                (envelope.acquisition_tick, envelope.capture_index),
                envelope,
            );
        }

        let horizon = horizon?;
        let ready: Vec<(u64, u64)> = self
            .received
            .range(..=(horizon, u64::MAX))
            .map(|(key, _)| *key)
            .collect();
        let mut batch = Vec::with_capacity(ready.len());
        for key in ready {
            let envelope = self.received.remove(&key).expect("listed above");
            self.dequeue(&envelope.source_id);
            if envelope.acquisition_tick < horizon {
                // Received in time but stranded below the horizon: cannot
                // happen for a validated profile, and is dropped rather than
                // fused out of order if it ever does.
                self.drop_envelope(
                    envelope,
                    now_tick,
                    DropReason::Late {
                        horizon_tick: horizon,
                    },
                );
                continue;
            }
            self.totals.fused += 1;
            batch.push(BearingObservation::from(&envelope.observation));
        }
        let timestamp_s = self.tick_times.get(&horizon).copied();
        // Times below the horizon are never needed again.
        self.tick_times = self.tick_times.split_off(&horizon);
        Some((timestamp_s?, batch))
    }

    /// This tick's drops, for the frame's rejection records.
    pub fn take_drops(&mut self) -> Vec<Dropped> {
        std::mem::take(&mut self.drops)
    }

    /// Envelopes not yet fused: in flight or awaiting the horizon.
    pub fn pending(&self) -> usize {
        self.in_flight.len() + self.received.len()
    }
}

/// A map with composite keys as a list of `[key, value]` pairs: JSON object
/// keys must be strings.
mod pairs {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize,
        V: Serialize,
        S: Serializer,
    {
        serializer.collect_seq(map.iter())
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Vec::<(K, V)>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(node: &str, target: &str, t: f64) -> BearingObservation {
        BearingObservation {
            node_id: node.into(),
            target_id: target.into(),
            origin: Vector3::new(1.0, 2.0, 3.0),
            direction: Vector3::new(0.0, 1.0, 0.0),
            bearing_std_rad: 0.01,
            timestamp_s: t,
            confidence: 0.9,
        }
    }

    fn profile(lag: u64, delay: u64, jitter: u64) -> DeliveryProfile {
        DeliveryProfile {
            fusion_lag_ticks: lag,
            default_delay_ticks: delay,
            per_source_delay_ticks: BTreeMap::new(),
            jitter_ticks: jitter,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            seed: 7,
        }
    }

    #[test]
    fn a_zero_profile_delivers_each_tick_in_capture_order_at_once() {
        let mut net = DeliveryNetwork::new(profile(0, 0, 0));
        let batch = vec![obs("b", "x", 0.5), obs("a", "x", 0.5), obs("b", "y", 0.5)];
        net.capture(1, 0.5, &batch);
        let (t, out) = net.deliver(1).expect("horizon 1");
        assert_eq!(t, 0.5);
        assert_eq!(out, batch, "same observations, same order");
        assert!(net.take_drops().is_empty());
        assert_eq!(net.pending(), 0);
    }

    /// Capture and delivery stay distinct: evidence captured at tick 3 with a
    /// delay of 2 is received at tick 5 and fused when the horizon (lag 3)
    /// reaches tick 3, at tick 6, keeping its acquisition time.
    #[test]
    fn capture_and_delivery_times_are_kept_apart() {
        let mut net = DeliveryNetwork::new(profile(3, 2, 0));
        for tick in 0..=6 {
            let observations = if tick == 3 {
                vec![obs("a", "x", 1.5)]
            } else {
                vec![]
            };
            net.capture(tick, tick as f64 * 0.5, &observations);
            let delivered = net.deliver(tick);
            match tick {
                0..=2 => assert!(delivered.is_none(), "no horizon yet at tick {tick}"),
                3..=5 => assert!(
                    delivered.unwrap().1.is_empty(),
                    "nothing acquired at the horizon"
                ),
                6 => {
                    let (t, batch) = delivered.unwrap();
                    assert_eq!(t, 1.5, "the horizon's own time");
                    assert_eq!(batch.len(), 1);
                    assert_eq!(batch[0].timestamp_s, 1.5, "acquisition time, not receipt");
                }
                _ => unreachable!(),
            }
            if tick == 5 {
                let (_, env) = net.received.iter().next().expect("received at 5");
                assert_eq!(
                    (
                        env.acquisition_tick,
                        env.availability_tick,
                        env.receipt_tick
                    ),
                    (3, 5, Some(5))
                );
            }
        }
        assert_eq!(net.totals().fused, 1);
    }

    #[test]
    fn a_repeated_sequence_is_dropped_as_a_duplicate() {
        let mut net = DeliveryNetwork::new(profile(1, 0, 0));
        net.capture(0, 0.0, &[obs("a", "x", 0.0)]);
        let copy = net.in_flight.values().next().unwrap().clone();
        net.deliver(0);
        let mut again = copy;
        again.capture_index = 99;
        net.send(again, 0);
        net.deliver(1);
        let drops = net.take_drops();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].reason, DropReason::Duplicate);
        assert_eq!(drops[0].rejection(0.5).reason, "delivery_duplicate");
    }

    #[test]
    fn a_source_restart_makes_its_earlier_envelopes_stale() {
        let mut net = DeliveryNetwork::new(profile(2, 2, 0));
        net.capture(0, 0.0, &[obs("a", "x", 0.0)]);
        net.restart_source("a");
        net.capture(1, 0.5, &[obs("a", "x", 0.5)]);
        for tick in 0..=3 {
            net.deliver(tick);
        }
        let drops = net.take_drops();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].reason, DropReason::StaleEpoch { current_epoch: 1 });
        // The new epoch's first envelope starts its sequence again.
        assert_eq!(net.totals().fused, 1);
    }

    #[test]
    fn evidence_arriving_after_the_horizon_is_late_not_fused() {
        let mut net = DeliveryNetwork::new(profile(1, 0, 0));
        net.capture(0, 0.0, &[]);
        net.capture(1, 0.5, &[]);
        net.deliver(1);
        // A transport that delayed tick 0's evidence past the lag.
        let mut env = SensorEnvelope {
            source_id: "a".into(),
            sequence: 0,
            source_epoch: 0,
            acquisition_tick: 0,
            availability_tick: 3,
            receipt_tick: None,
            capture_index: 0,
            observation: StoredObservation::from(&obs("a", "x", 0.0)),
        };
        net.send(env.clone(), 1);
        for tick in 2..=3 {
            net.capture(tick, tick as f64 * 0.5, &[]);
            net.deliver(tick);
        }
        let drops = net.take_drops();
        env.receipt_tick = Some(3);
        assert_eq!(
            drops,
            vec![Dropped {
                envelope: env,
                at_tick: 3,
                reason: DropReason::Late { horizon_tick: 2 }
            }]
        );
        assert_eq!(net.totals().dropped.get("delivery_late"), Some(&1));
    }

    #[test]
    fn a_full_queue_overflows_with_a_reason() {
        let mut p = profile(5, 5, 0);
        p.queue_capacity = 2;
        let mut net = DeliveryNetwork::new(p);
        net.capture(
            0,
            0.0,
            &[obs("a", "x", 0.0), obs("a", "y", 0.0), obs("a", "z", 0.0)],
        );
        let drops = net.take_drops();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].reason, DropReason::Overflow { capacity: 2 });
        assert_eq!(net.pending(), 2);
    }

    /// Jitter reorders arrivals, but fusion still sees each acquisition tick
    /// whole and in capture order, and the same seed gives the same arrivals.
    #[test]
    fn jitter_reorders_arrivals_but_not_what_fusion_sees() {
        let run = |seed: u32| {
            let mut p = profile(4, 1, 3);
            p.seed = seed;
            let mut net = DeliveryNetwork::new(p);
            let mut arrivals = Vec::new();
            let mut fused = Vec::new();
            for tick in 0..20u64 {
                let batch: Vec<_> = ["a", "b", "c"]
                    .iter()
                    .map(|n| obs(n, "x", tick as f64))
                    .collect();
                net.capture(tick, tick as f64, &batch);
                arrivals.push(net.in_flight.keys().copied().collect::<Vec<_>>());
                if let Some((t, batch)) = net.deliver(tick) {
                    fused.push((
                        t,
                        batch.iter().map(|o| o.node_id.clone()).collect::<Vec<_>>(),
                    ));
                }
            }
            assert!(net.take_drops().is_empty());
            (arrivals, fused)
        };
        let (arrivals, fused) = run(7);
        assert_eq!(
            run(7),
            (arrivals.clone(), fused.clone()),
            "deterministic for a seed"
        );
        assert_ne!(run(8).0, arrivals, "another seed draws other delays");
        for (t, nodes) in &fused {
            assert_eq!(
                nodes,
                &["a", "b", "c"],
                "tick {t} complete, in capture order"
            );
        }
        assert_eq!(fused.len(), 16);
    }

    #[test]
    fn a_profile_slower_than_its_lag_is_refused_before_the_run() {
        let error = DeliveryProfile::from_seconds(0.5, 1.0, 0.75, &BTreeMap::new(), 0.5, 64, 1)
            .unwrap_err();
        assert!(error.contains("unsupported delivery profile"), "{error}");
        let ok =
            DeliveryProfile::from_seconds(0.5, 1.0, 0.5, &BTreeMap::new(), 0.5, 64, 1).unwrap();
        assert_eq!(
            (ok.fusion_lag_ticks, ok.default_delay_ticks, ok.jitter_ticks),
            (2, 1, 1)
        );
        let mut slow = BTreeMap::new();
        slow.insert("drone-1".to_string(), 2.0);
        assert!(DeliveryProfile::from_seconds(0.5, 1.0, 0.0, &slow, 0.0, 64, 1).is_err());
        assert!(
            DeliveryProfile::from_seconds(0.5, -1.0, 0.0, &BTreeMap::new(), 0.0, 64, 1).is_err()
        );
    }

    #[test]
    fn the_network_round_trips_through_a_checkpoint() {
        let mut net = DeliveryNetwork::new(profile(3, 1, 2));
        for tick in 0..5 {
            net.capture(tick, tick as f64, &[obs("a", "x", tick as f64)]);
            net.deliver(tick);
        }
        let saved = serde_json::to_value(&net).unwrap();
        let restored: DeliveryNetwork = serde_json::from_value(saved).unwrap();
        assert_eq!(restored, net);
    }
}
