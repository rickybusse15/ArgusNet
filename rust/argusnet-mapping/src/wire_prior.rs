//! Uncertain wire priors (W5): a map of power lines that may be wrong.
//!
//! A utility's or OpenStreetMap's line is useful long before a sensor can see
//! a wire, and it is not the truth. It may be metres off, a span may be
//! missing, and a line may have been taken down. So a prior span here is a
//! hypothesis with a declared error, and flight changes it three ways:
//!
//! - **Confirmed.** A fitted track or pole lying on the span, within its
//!   error, fixes the offset and shrinks the error to what was measured.
//! - **Refuted.** Flying near enough that a wire there could not be missed,
//!   and getting no return, counts against each stretch. Enough stretches
//!   with nothing, and no evidence for it, drops the span.
//! - **Open.** Everything else stays a hazard with a wide clearance.
//!
//! A wire missing from the prior is found as any unknown wire is. The prior
//! changes nothing about measuring.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::wires::WireEstimate;

/// One span of a prior map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorSpan {
    pub id: String,
    /// The wire's ends, at attachment height, metres.
    pub from_m: [f64; 3],
    pub to_m: [f64; 3],
    /// Declared one-sigma error across the line and in height, metres.
    pub sigma_lateral_m: f64,
    pub sigma_vertical_m: f64,
}

/// What is done with a prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WirePriorConfig {
    /// A wire's evidence supports a span within this many sigma.
    pub gate_sigmas: f64,
    /// A track supports a span when this parallel to it (cosine).
    pub min_alignment: f64,
    /// Observed length of track a span needs to be confirmed, metres.
    pub confirm_length_m: f64,
    /// A stretch of span, metres.
    pub stretch_m: f64,
    /// A stretch this close to a sensor would have shown a wire, metres.
    pub refute_range_m: f64,
    /// Scans seeing nothing a stretch needs to count against the span.
    pub refute_scans: u32,
    /// The share of a span's stretches that must count against it.
    pub refute_share: f64,
    /// The error left on a confirmed span, metres: how well a prior's shape
    /// follows the wire away from where it was measured.
    pub confirmed_sigma_m: f64,
}

impl Default for WirePriorConfig {
    fn default() -> Self {
        Self {
            gate_sigmas: 3.0,
            min_alignment: 0.97,
            confirm_length_m: 20.0,
            stretch_m: 20.0,
            refute_range_m: 60.0,
            refute_scans: 3,
            refute_share: 0.5,
            confirmed_sigma_m: 1.0,
        }
    }
}

impl WirePriorConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("gate_sigmas", self.gate_sigmas),
            ("confirm_length_m", self.confirm_length_m),
            ("stretch_m", self.stretch_m),
            ("refute_range_m", self.refute_range_m),
            ("confirmed_sigma_m", self.confirmed_sigma_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("wire prior {name} must be positive, got {value}"));
            }
        }
        if !(self.min_alignment > 0.0 && self.min_alignment <= 1.0) {
            return Err("wire prior min_alignment must be in (0, 1]".into());
        }
        if !(self.refute_share > 0.0 && self.refute_share <= 1.0) {
            return Err("wire prior refute_share must be in (0, 1]".into());
        }
        if self.refute_scans == 0 {
            return Err("wire prior refute_scans must be at least 1".into());
        }
        Ok(())
    }
}

/// What flight has made of a prior span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorStatus {
    Open,
    Confirmed,
    Refuted,
}

/// A prior span as flight has left it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorState {
    pub span: PriorSpan,
    pub status: PriorStatus,
    /// Scans that saw nothing at each stretch.
    misses: BTreeMap<i32, u32>,
    /// Stretches with a return near them.
    hits: BTreeSet<i32>,
}

/// The prior map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorMap {
    config: WirePriorConfig,
    spans: Vec<PriorState>,
}

fn flat_length(span: &PriorSpan) -> f64 {
    (span.to_m[0] - span.from_m[0]).hypot(span.to_m[1] - span.from_m[1])
}

/// Along and across `span` for a point on the ground.
fn frame(span: &PriorSpan, x: f64, y: f64) -> (f64, f64) {
    let length = flat_length(span).max(1e-9);
    let (ux, uy) = (
        (span.to_m[0] - span.from_m[0]) / length,
        (span.to_m[1] - span.from_m[1]) / length,
    );
    let (dx, dy) = (x - span.from_m[0], y - span.from_m[1]);
    (dx * ux + dy * uy, dy * ux - dx * uy)
}

impl PriorMap {
    pub fn new(config: WirePriorConfig, spans: Vec<PriorSpan>) -> Self {
        Self {
            config,
            spans: spans
                .into_iter()
                .map(|span| PriorState {
                    span,
                    status: PriorStatus::Open,
                    misses: BTreeMap::new(),
                    hits: BTreeSet::new(),
                })
                .collect(),
        }
    }

    pub fn spans(&self) -> &[PriorState] {
        &self.spans
    }

    pub fn count(&self, status: PriorStatus) -> usize {
        self.spans.iter().filter(|s| s.status == status).count()
    }

    /// Back to what the map said.
    pub fn clear(&mut self) {
        for state in &mut self.spans {
            state.status = PriorStatus::Open;
            state.misses.clear();
            state.hits.clear();
        }
    }

    /// One scan from `origin_m`: its returns count for the stretches they
    /// are near, and the stretches near enough to show a wire with nothing
    /// count against them. A scan looks below the horizon, so a wire above
    /// the sensor says nothing.
    pub fn observe(&mut self, origin_m: [f64; 3], points: &[[f64; 3]]) {
        let config = &self.config;
        for state in &mut self.spans {
            if state.status != PriorStatus::Open {
                continue;
            }
            let span = &state.span;
            let gate = config.gate_sigmas * span.sigma_lateral_m;
            let vertical = config.gate_sigmas * span.sigma_vertical_m;
            let length = flat_length(span);
            let high = span.from_m[2].max(span.to_m[2]);
            let mut hit_now = BTreeSet::new();
            for point in points {
                let (s, d) = frame(span, point[0], point[1]);
                if s >= -gate
                    && s <= length + gate
                    && d.abs() <= gate
                    && (point[2] - high).abs() <= vertical + 0.05 * length
                {
                    hit_now.insert((s / config.stretch_m).floor() as i32);
                }
            }
            state.hits.extend(hit_now.iter().copied());
            let stretches = (length / config.stretch_m).ceil() as i32;
            for k in 0..stretches {
                if hit_now.contains(&k) {
                    continue;
                }
                let t = ((f64::from(k) + 0.5) * config.stretch_m / length).min(1.0);
                let at: [f64; 3] =
                    std::array::from_fn(|i| span.from_m[i] + (span.to_m[i] - span.from_m[i]) * t);
                let range = ((at[0] - origin_m[0]).powi(2)
                    + (at[1] - origin_m[1]).powi(2)
                    + (at[2] - origin_m[2]).powi(2))
                .sqrt();
                if range <= config.refute_range_m && at[2] < origin_m[2] {
                    *state.misses.entry(k).or_insert(0) += 1;
                }
            }
            let against = state
                .misses
                .iter()
                .filter(|(k, n)| **n >= config.refute_scans && !state.hits.contains(k))
                .count();
            if state.hits.is_empty()
                && against as f64 >= config.refute_share * f64::from(stretches.max(1))
            {
                state.status = PriorStatus::Refuted;
            }
        }
    }

    /// Hold the prior against what was fitted: a confirmed track on a span
    /// moves it by the track's offset and shrinks its error. A span is only
    /// raised: the prior's height is at the attachments, a track's is lower
    /// by its sag.
    pub fn reconcile(&mut self, estimates: &[WireEstimate]) {
        let config = &self.config;
        for state in &mut self.spans {
            if state.status != PriorStatus::Open {
                continue;
            }
            let span = &state.span;
            let gate = config.gate_sigmas * span.sigma_lateral_m;
            let length = flat_length(span).max(1e-9);
            let u = [
                (span.to_m[0] - span.from_m[0]) / length,
                (span.to_m[1] - span.from_m[1]) / length,
            ];
            let mut support: Option<(f64, f64, f64)> = None;
            for track in estimates.iter().filter(|track| track.confirmed) {
                if track.observed_length_m < config.confirm_length_m {
                    continue;
                }
                let dx = track.to_m[0] - track.from_m[0];
                let dy = track.to_m[1] - track.from_m[1];
                let along = dx.hypot(dy).max(1e-9);
                if ((dx * u[0] + dy * u[1]) / along).abs() < config.min_alignment {
                    continue;
                }
                let middle = [
                    (track.from_m[0] + track.to_m[0]) / 2.0,
                    (track.from_m[1] + track.to_m[1]) / 2.0,
                    (track.from_m[2] + track.to_m[2]) / 2.0,
                ];
                let (s, d) = frame(span, middle[0], middle[1]);
                if s < -gate || s > length + gate || d.abs() > gate {
                    continue;
                }
                let z =
                    span.from_m[2] + (span.to_m[2] - span.from_m[2]) * (s / length).clamp(0.0, 1.0);
                let dz = middle[2] - z;
                if support.is_none_or(|(_, best, _)| track.observed_length_m > best) {
                    support = Some((d, track.observed_length_m, dz));
                }
            }
            let Some((d, _, dz)) = support else {
                continue;
            };
            let (nx, ny) = (-u[1], u[0]);
            for end in [&mut state.span.from_m, &mut state.span.to_m] {
                end[0] += nx * d;
                end[1] += ny * d;
                end[2] += dz.max(0.0);
            }
            state.span.sigma_lateral_m = config.confirmed_sigma_m;
            state.span.sigma_vertical_m = config.confirmed_sigma_m;
            state.status = PriorStatus::Confirmed;
        }
    }
}

/// How a true line is spoiled into a prior, for evaluation: what a map could
/// be wrong by. The errors are drawn from `seed`, so a run repeats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorDegradation {
    /// One-sigma shift of a whole line across the ground, metres.
    pub line_offset_m: f64,
    /// One-sigma error of each support's height, metres.
    pub height_error_m: f64,
    /// The share of spans missing from the map.
    pub drop_share: f64,
    /// Spans on the map with no wire: a line since taken down.
    pub phantom_spans: u32,
    /// How far a phantom lies from the line, metres.
    pub phantom_offset_m: f64,
    /// The error the map declares for itself, metres.
    pub declared_sigma_m: f64,
    pub seed: u64,
}

impl Default for PriorDegradation {
    fn default() -> Self {
        Self {
            line_offset_m: 6.0,
            height_error_m: 3.0,
            drop_share: 0.25,
            phantom_spans: 1,
            phantom_offset_m: 180.0,
            declared_sigma_m: 8.0,
            seed: 0x5EED,
        }
    }
}

impl PriorDegradation {
    /// A map of the spans given as `(id, from, to)`, as wrong as declared.
    pub fn apply(&self, spans: &[(String, [f64; 3], [f64; 3])]) -> Vec<PriorSpan> {
        let mut state = self.seed | 1;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1_000_000) as f64 / 1_000_000.0
        };
        // Roughly normal: the sum of twelve uniforms.
        let normal = |sigma: f64, draw: &mut dyn FnMut() -> f64| {
            sigma * ((0..12).map(|_| draw()).sum::<f64>() - 6.0)
        };
        let shift = [
            normal(self.line_offset_m, &mut uniform),
            normal(self.line_offset_m, &mut uniform),
        ];
        let mut out = Vec::new();
        for (id, from, to) in spans {
            let dropped = uniform() < self.drop_share;
            let (dz0, dz1) = (
                normal(self.height_error_m, &mut uniform),
                normal(self.height_error_m, &mut uniform),
            );
            if dropped {
                continue;
            }
            out.push(PriorSpan {
                id: id.clone(),
                from_m: [from[0] + shift[0], from[1] + shift[1], from[2] + dz0],
                to_m: [to[0] + shift[0], to[1] + shift[1], to[2] + dz1],
                sigma_lateral_m: self.declared_sigma_m,
                sigma_vertical_m: self.declared_sigma_m,
            });
        }
        for (k, (_, from, to)) in spans.iter().take(self.phantom_spans as usize).enumerate() {
            let length = (to[0] - from[0]).hypot(to[1] - from[1]).max(1e-9);
            let (nx, ny) = (-(to[1] - from[1]) / length, (to[0] - from[0]) / length);
            let offset = self.phantom_offset_m;
            out.push(PriorSpan {
                id: format!("phantom-{k}"),
                from_m: [from[0] + nx * offset, from[1] + ny * offset, from[2]],
                to_m: [to[0] + nx * offset, to[1] + ny * offset, to[2]],
                sigma_lateral_m: self.declared_sigma_m,
                sigma_vertical_m: self.declared_sigma_m,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> PriorSpan {
        PriorSpan {
            id: "s".into(),
            from_m: [0.0, 0.0, 50.0],
            to_m: [200.0, 0.0, 50.0],
            sigma_lateral_m: 8.0,
            sigma_vertical_m: 8.0,
        }
    }

    fn map() -> PriorMap {
        PriorMap::new(WirePriorConfig::default(), vec![span()])
    }

    fn track(offset: f64, length: f64) -> WireEstimate {
        WireEstimate {
            id: 0,
            from_m: [20.0, offset, 50.0],
            to_m: [20.0 + length, offset, 50.0],
            sag_m: 0.0,
            sigma_lateral_m: 0.2,
            sigma_vertical_m: 0.2,
            observed_length_m: length,
            points: 100,
            camera_support: 0,
            triangulated: false,
            confirmed: true,
            pole_ends: [None, None],
        }
    }

    #[test]
    fn a_track_on_a_span_confirms_and_moves_it() {
        let mut map = map();
        map.reconcile(&[track(5.0, 60.0)]);
        let state = &map.spans()[0];
        assert_eq!(state.status, PriorStatus::Confirmed);
        assert!((state.span.from_m[1] - 5.0).abs() < 1e-9);
        assert_eq!(state.span.sigma_lateral_m, 1.0);
    }

    #[test]
    fn a_track_outside_the_error_or_across_the_span_confirms_nothing() {
        let mut map = map();
        map.reconcile(&[track(40.0, 60.0)]);
        let mut across = track(0.0, 60.0);
        across.to_m = [20.0, 60.0, 50.0];
        map.reconcile(&[across]);
        assert_eq!(map.spans()[0].status, PriorStatus::Open);
    }

    /// Flying close over a stretch with nothing there counts against it,
    /// and enough stretches drop the span. A distant pass does not.
    #[test]
    fn nothing_seen_from_near_refutes_a_span() {
        let mut map = map();
        for _ in 0..10 {
            map.observe([100.0, 0.0, 300.0], &[]);
        }
        assert_eq!(map.spans()[0].status, PriorStatus::Open, "too far to tell");
        for x in [20.0, 50.0, 80.0, 110.0, 140.0, 170.0] {
            for _ in 0..3 {
                map.observe([x, 0.0, 80.0], &[]);
            }
        }
        assert_eq!(map.spans()[0].status, PriorStatus::Refuted);
    }

    #[test]
    fn a_return_on_a_span_keeps_it_from_being_refuted() {
        let mut map = map();
        map.observe([100.0, 0.0, 80.0], &[[100.0, 3.0, 50.0]]);
        for x in [20.0, 50.0, 80.0, 110.0, 140.0, 170.0] {
            for _ in 0..4 {
                map.observe([x, 0.0, 80.0], &[]);
            }
        }
        assert_eq!(map.spans()[0].status, PriorStatus::Open);
    }

    #[test]
    fn a_wire_above_the_sensor_says_nothing() {
        let mut map = map();
        for x in [20.0, 50.0, 80.0, 110.0, 140.0, 170.0] {
            for _ in 0..4 {
                map.observe([x, 0.0, 30.0], &[]);
            }
        }
        assert_eq!(map.spans()[0].status, PriorStatus::Open);
    }

    #[test]
    fn degrading_repeats_and_holds_what_it_declares() {
        let spans: Vec<(String, [f64; 3], [f64; 3])> = (0..8)
            .map(|k| {
                let y = f64::from(k) * 150.0;
                (format!("s{k}"), [0.0, y, 50.0], [0.0, y + 150.0, 50.0])
            })
            .collect();
        let degradation = PriorDegradation::default();
        let map = degradation.apply(&spans);
        assert_eq!(map, degradation.apply(&spans));
        assert!(map.len() < spans.len() + 1, "some were dropped");
        assert!(map.iter().any(|s| s.id == "phantom-0"));
        let moved = map.iter().find(|s| s.id != "phantom-0").unwrap();
        assert!(moved.from_m[0].abs() > 1e-6 || moved.from_m[1].fract() != 0.0);
        assert_eq!(moved.sigma_lateral_m, 8.0);
    }

    #[test]
    fn a_map_survives_a_checkpoint_and_a_reset() {
        let mut map = map();
        map.observe([100.0, 0.0, 80.0], &[[100.0, 3.0, 50.0]]);
        let text = serde_json::to_string(&map).unwrap();
        let restored: PriorMap = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, map);
        map.clear();
        assert_eq!(map.spans()[0].status, PriorStatus::Open);
    }

    #[test]
    fn nonsense_is_refused() {
        let bad = WirePriorConfig {
            refute_share: 0.0,
            ..WirePriorConfig::default()
        };
        assert!(bad.validate().is_err());
    }
}
