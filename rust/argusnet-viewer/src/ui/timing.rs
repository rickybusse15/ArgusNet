//! Timing in Diagnostics (console parity V4, P4.10): how the run was
//! scheduled and how fast the view is advancing. Every row reads what the
//! replay recorded, or what the viewer measured; what neither has reads as
//! unavailable with the reason, never as a default.

use bevy::prelude::*;
use bevy_egui::egui;
use serde_json::Value;

use crate::replay::ReplayState;

/// Source seconds advanced per wall second, measured over the last second
/// or so the view moved: what the operator actually gets, against the speed
/// asked for.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct AchievedRate {
    /// Smoothed rate; `None` until the view has advanced once.
    pub rate: Option<f32>,
    last_source_s: Option<f32>,
    /// Wall seconds since the view last advanced.
    idle_s: f32,
}

/// How quickly the smoothed rate follows a change, seconds.
const RATE_TAU_S: f32 = 0.75;
/// Still for this long reads as stopped.
const STOPPED_AFTER_S: f32 = 1.0;

impl AchievedRate {
    /// Fold in one wall-clock step of `wall_dt_s` in which the cursor moved
    /// to `source_s`. A jump backwards (a seek, a new run) restarts it.
    pub fn observe(&mut self, source_s: f32, wall_dt_s: f32) {
        let Some(last) = self.last_source_s.replace(source_s) else {
            return;
        };
        if wall_dt_s <= 0.0 {
            return;
        }
        let advanced = source_s - last;
        if advanced < 0.0 {
            self.rate = None;
            self.idle_s = 0.0;
            return;
        }
        if advanced == 0.0 {
            self.idle_s += wall_dt_s;
            if self.idle_s >= STOPPED_AFTER_S {
                self.rate = Some(0.0);
            }
            return;
        }
        // Ticks arrive in bursts (one source tick per several wall frames),
        // so the step is spread over the idle time it ended.
        let span = wall_dt_s + self.idle_s;
        self.idle_s = 0.0;
        let instant = advanced / span;
        let weight = (span / RATE_TAU_S).min(1.0);
        self.rate = Some(match self.rate {
            Some(rate) => rate + (instant - rate) * weight,
            None => instant,
        });
    }
}

/// Measured on the real clock, not Bevy's virtual one, which caps each
/// step at 250 ms and would hide a slow frame; and on the cursor itself,
/// which playback advances continuously between recorded frames, not the
/// frame it sits in, which would read normal slow playback as stalls.
pub fn measure_achieved_rate_system(
    time: Res<Time<Real>>,
    replay: Res<ReplayState>,
    mut rate: ResMut<AchievedRate>,
) {
    if replay.frame_count() == 0 {
        *rate = AchievedRate::default();
        return;
    }
    rate.observe(replay.time_s(), time.delta_secs());
}

/// One timing readout: a value, or why there is none.
pub type TimingRow = (&'static str, Result<String, &'static str>);

fn rate_table(value: Option<&Value>) -> Option<String> {
    let table = value?.as_object()?;
    (!table.is_empty()).then(|| {
        table
            .iter()
            .map(|(name, rate)| format!("{name} {}", rate.as_str().unwrap_or("?")))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The rows for a replay's `meta`, the requested speed (`None` where the
/// view does not set it, such as a run with its own Speed control) and the
/// achieved rate.
pub fn timing_rows(
    meta: Option<&Value>,
    requested: Option<f32>,
    achieved: Option<f32>,
) -> Vec<TimingRow> {
    let get = |key: &str| meta.and_then(|meta| meta.get(key));
    let mut rows: Vec<TimingRow> = Vec::new();
    rows.push((
        "Execution",
        Ok(match get("execution_model").and_then(Value::as_str) {
            Some(model) => model.to_string(),
            None => "legacy: every stage every tick".to_string(),
        }),
    ));
    rows.push((
        "Lane rates",
        match get("lane_rates") {
            Some(rates) => rate_table(Some(rates)).ok_or("the run records an empty table"),
            None => Ok("every lane every tick".to_string()),
        },
    ));
    rows.push((
        "Sensor rates",
        match get("sensor_rates") {
            Some(rates) => rate_table(Some(rates)).ok_or("the run records an empty table"),
            None => Ok("every sensor every tick".to_string()),
        },
    ));
    rows.push((
        "Delivery",
        match get("delivery_profile") {
            Some(profile) => Ok(format!(
                "fusion lag {} tick(s), jitter {}",
                profile
                    .get("fusion_lag_ticks")
                    .and_then(Value::as_u64)
                    .map_or("?".to_string(), |v| v.to_string()),
                profile
                    .get("jitter_ticks")
                    .and_then(Value::as_u64)
                    .map_or("?".to_string(), |v| format!("{v} tick(s)")),
            )),
            None => Err("same-tick fusion: no delivery modelled"),
        },
    ));
    rows.push((
        "Delivered",
        match get("delivery_totals") {
            Some(totals) => {
                let n = |key: &str| totals.get(key).and_then(Value::as_u64).unwrap_or(0);
                Ok(format!(
                    "{} captured, {} fused, {} dropped, {} in transit at the end",
                    n("captured"),
                    n("fused"),
                    n("dropped"),
                    n("in_transit_at_end")
                ))
            }
            None => Err("no delivery totals in this replay"),
        },
    ));
    rows.push((
        "Speed asked",
        requested
            .map(|speed| format!("{speed:.1}\u{00d7}"))
            .ok_or("set by the run's Speed control"),
    ));
    rows.push((
        "Speed achieved",
        match achieved {
            Some(0.0) => Ok("stopped".to_string()),
            Some(rate) => Ok(format!("{rate:.2}\u{00d7} real time")),
            None => Err("the view has not advanced yet"),
        },
    ));
    rows
}

/// The Diagnostics section.
pub fn section_timing(ui: &mut egui::Ui, replay: &ReplayState, achieved: &AchievedRate) {
    let meta = replay.document.as_ref().and_then(|d| d.meta.as_ref());
    let requested = (!replay.live).then_some(replay.playback_speed);
    ui.collapsing("Timing", |ui| {
        egui::Grid::new("timing_rows")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (label, value) in timing_rows(meta, requested, achieved.rate) {
                    ui.label(label);
                    match value {
                        Ok(text) => ui.label(text),
                        Err(why) => ui.weak(format!("unavailable: {why}")),
                    };
                    ui.end_row();
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value<'a>(rows: &'a [TimingRow], label: &str) -> &'a Result<String, &'static str> {
        &rows.iter().find(|(l, _)| *l == label).unwrap().1
    }

    /// A legacy replay says so plainly; what it never recorded is
    /// unavailable, never a default.
    #[test]
    fn a_legacy_replay_reads_as_legacy_and_unavailable() {
        let rows = timing_rows(Some(&serde_json::json!({})), Some(1.0), None);
        assert_eq!(
            value(&rows, "Execution"),
            &Ok("legacy: every stage every tick".into())
        );
        assert_eq!(
            value(&rows, "Lane rates"),
            &Ok("every lane every tick".into())
        );
        assert!(value(&rows, "Delivery").is_err());
        assert!(value(&rows, "Delivered").is_err());
        assert_eq!(value(&rows, "Speed asked"), &Ok("1.0\u{00d7}".into()));
        assert!(value(&rows, "Speed achieved").is_err());
    }

    /// A scheduled, delivered run shows its rates and totals.
    #[test]
    fn a_scheduled_run_shows_its_rates_and_delivery() {
        let meta = serde_json::json!({
            "execution_model": "scheduled_v1",
            "lane_rates": {"mission": "1/2"},
            "sensor_rates": {"ground-alpha": "1/3"},
            "delivery_profile": {"fusion_lag_ticks": 3, "jitter_ticks": 1},
            "delivery_totals": {"captured": 10, "fused": 8, "dropped": 1, "in_transit_at_end": 1}
        });
        let rows = timing_rows(Some(&meta), None, Some(0.0));
        assert_eq!(value(&rows, "Execution"), &Ok("scheduled_v1".into()));
        assert_eq!(value(&rows, "Lane rates"), &Ok("mission 1/2".into()));
        assert_eq!(value(&rows, "Sensor rates"), &Ok("ground-alpha 1/3".into()));
        assert_eq!(
            value(&rows, "Delivery"),
            &Ok("fusion lag 3 tick(s), jitter 1 tick(s)".into())
        );
        assert_eq!(
            value(&rows, "Delivered"),
            &Ok("10 captured, 8 fused, 1 dropped, 1 in transit at the end".into())
        );
        assert!(value(&rows, "Speed asked").is_err(), "a run sets its own");
        assert_eq!(value(&rows, "Speed achieved"), &Ok("stopped".into()));
    }

    /// The section draws in a frame, with nothing loaded and with a replay.
    #[test]
    fn the_section_draws() {
        let context = egui::Context::default();
        let empty = ReplayState::new(None);
        let replay = ReplayState::new(Some(
            serde_json::from_value(serde_json::json!({
                "meta": {"execution_model": "scheduled_v1"},
                "frames": [{"timestamp_s": 0.0}]
            }))
            .unwrap(),
        ));
        for state in [&empty, &replay] {
            context
                .run_ui(egui::RawInput::default(), |ui| {
                    section_timing(ui, state, &AchievedRate::default());
                })
                .textures_delta
                .clear();
        }
    }

    /// The achieved rate follows the source over wall time, spreads a
    /// burst over the idle time it ends, restarts on a jump back, and reads
    /// as stopped after a second still.
    #[test]
    fn the_achieved_rate_tracks_the_source_over_wall_time() {
        let mut rate = AchievedRate::default();
        rate.observe(0.0, 0.1);
        assert_eq!(rate.rate, None);
        // 4x: one 0.4 s source tick every 0.1 s.
        for k in 1..=40 {
            rate.observe(0.4 * k as f32, 0.1);
        }
        assert!((rate.rate.unwrap() - 4.0).abs() < 1e-3, "{:?}", rate.rate);
        // 1x in bursts: a 1 s tick after nine still 0.1 s frames.
        let mut source = 16.0;
        for _ in 0..30 {
            for _ in 0..9 {
                rate.observe(source, 0.1);
            }
            source += 1.0;
            rate.observe(source, 0.1);
        }
        assert!((rate.rate.unwrap() - 1.0).abs() < 0.05, "{:?}", rate.rate);
        rate.observe(2.0, 0.1);
        assert_eq!(rate.rate, None, "a seek back restarts");
        // Slow playback, the cursor moving a little every frame: 0.25x,
        // never "stopped".
        let mut slow = AchievedRate::default();
        for k in 0..300 {
            slow.observe(0.0025 * k as f32, 0.01);
        }
        assert!((slow.rate.unwrap() - 0.25).abs() < 1e-3, "{:?}", slow.rate);
        for _ in 0..11 {
            rate.observe(2.0, 0.1);
        }
        assert_eq!(rate.rate, Some(0.0));
    }
}
