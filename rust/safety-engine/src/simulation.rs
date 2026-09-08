//! Safety policy for simulation telemetry (no fabricated comms/gimbal signals).
//! Owns the former Python policy, using the same physical limits as flight validation.
use crate::limits::DronePhysicalLimits;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub constraint: String,
    pub commanded_value: f64,
    pub limit_value: f64,
    pub description: String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct State {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub terrain_height_m: f64,
    pub acceleration: Option<[f64; 3]>,
    pub previous_velocity: Option<[f64; 3]>,
    pub dt_s: Option<f64>,
    #[serde(default)]
    pub other_drone_positions: Vec<[f64; 3]>,
    pub energy_fraction: Option<f64>,
}
fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|v| v * v).sum::<f64>().sqrt()
}
pub fn check(l: &DronePhysicalLimits, s: &State) -> Vec<Violation> {
    let mut out = vec![];
    let mut add = |name: &str, value: f64, limit: f64, below: bool| {
        out.push(Violation {
            constraint: name.into(),
            commanded_value: value,
            limit_value: limit,
            description: format!(
                "{} {:.2} {} {:.2}",
                name.replace('_', " "),
                value,
                if below { "is below" } else { "exceeds" },
                limit
            ),
        });
    };
    let speed = norm(s.velocity);
    let agl = s.position[2] - s.terrain_height_m;
    for (name, value, limit, below) in [
        ("max_speed", speed, l.max_speed_mps, false),
        ("min_speed", speed, l.min_speed_mps, true),
        ("min_agl", agl, l.min_agl_m, true),
        ("max_agl", agl, l.max_agl_m, false),
        ("max_climb_rate", s.velocity[2], l.max_climb_rate_mps, false),
        (
            "max_descent_rate",
            -s.velocity[2],
            l.max_descent_rate_mps,
            false,
        ),
    ] {
        if (if below { value < limit } else { value > limit })
            && !(name == "min_speed" && speed == 0.0)
        {
            add(name, value, limit, below);
        }
    }
    let acceleration = s.acceleration.or_else(|| {
        s.previous_velocity
            .zip(s.dt_s)
            .filter(|(_, dt)| *dt > 0.0)
            .map(|(v, dt)| std::array::from_fn(|i| (s.velocity[i] - v[i]) / dt))
    });
    if let Some(a) = acceleration {
        for (name, v, limit) in [
            (
                "max_horizontal_accel",
                a[0].hypot(a[1]),
                l.max_horizontal_accel_mps2,
            ),
            ("max_vertical_accel", a[2].abs(), l.max_vertical_accel_mps2),
        ] {
            if v > limit {
                add(name, v, limit, false);
            }
        }
    }
    if let Some(e) = s.energy_fraction {
        if e < l.min_energy_reserve_fraction {
            add("min_energy_reserve", e, l.min_energy_reserve_fraction, true);
        }
    }
    for p in &s.other_drone_positions {
        let h = (p[0] - s.position[0]).hypot(p[1] - s.position[1]);
        if h < l.min_drone_separation_m
            && (p[2] - s.position[2]).abs() < l.min_drone_vertical_separation_m
        {
            add("min_drone_separation", h, l.min_drone_separation_m, true);
        }
    }
    out
}
#[derive(Debug, Serialize)]
pub struct Decision {
    pub state: String,
    pub violations: Vec<Violation>,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}
#[derive(Debug, Clone)]
pub struct SimulationSafetyMonitor {
    pub limits: DronePhysicalLimits,
    pub abort_frames: u32,
    pub caution_fraction: f64,
    records: BTreeMap<String, (u32, String)>,
}
impl SimulationSafetyMonitor {
    pub fn new(
        limits: DronePhysicalLimits,
        abort_frames: u32,
        caution_fraction: f64,
    ) -> Result<Self, String> {
        if abort_frames == 0
            || !(0.0 < caution_fraction && caution_fraction <= 1.0)
            || limits.min_agl_m < 0.0
            || limits.max_agl_m < limits.min_agl_m
        {
            return Err("invalid safety configuration".into());
        }
        Ok(Self {
            limits,
            abort_frames,
            caution_fraction,
            records: BTreeMap::new(),
        })
    }
    pub fn state(&self, id: &str) -> &str {
        self.records.get(id).map_or("nominal", |r| r.1.as_str())
    }
    pub fn clear(&mut self, id: &str) {
        self.records.remove(id);
    }
    pub fn process(&mut self, id: &str, s: &State) -> Decision {
        let violations = check(&self.limits, s);
        let l = &self.limits;
        let speed = norm(s.velocity);
        let agl = s.position[2] - s.terrain_height_m;
        let count = if violations.is_empty() {
            0
        } else {
            self.records.get(id).map_or(1, |r| r.0 + 1)
        };
        let state = if !violations.is_empty() {
            if count >= self.abort_frames || violations.iter().any(|v| v.constraint == "min_agl") {
                "abort"
            } else {
                "warning"
            }
        } else if speed >= l.max_speed_mps * self.caution_fraction
            || agl <= l.min_agl_m / self.caution_fraction
            || s.energy_fraction
                .is_some_and(|e| e <= l.min_energy_reserve_fraction / self.caution_fraction)
        {
            "caution"
        } else {
            "nominal"
        };
        let mut position = s.position;
        let mut velocity = s.velocity;
        position[2] = position[2].clamp(
            s.terrain_height_m + l.min_agl_m,
            s.terrain_height_m + l.max_agl_m,
        );
        if speed > l.max_speed_mps {
            for v in &mut velocity {
                *v *= l.max_speed_mps / speed;
            }
        }
        velocity[2] = velocity[2].clamp(-l.max_descent_rate_mps, l.max_climb_rate_mps);
        if state == "abort" {
            velocity = [0.0; 3];
        }
        self.records.insert(id.into(), (count, state.into()));
        Decision {
            state: state.into(),
            violations,
            position,
            velocity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> State {
        State {
            position: [0.0, 0.0, 100.0],
            velocity: [50.0, 0.0, 0.0],
            terrain_height_m: 0.0,
            acceleration: None,
            previous_velocity: None,
            dt_s: None,
            other_drone_positions: vec![],
            energy_fraction: None,
        }
    }
    #[test]
    fn escalation_and_reset_are_per_drone() {
        let mut m =
            SimulationSafetyMonitor::new(DronePhysicalLimits::tracker_default(), 3, 0.9).unwrap();
        assert_eq!(m.process("a", &state()).state, "warning");
        assert_eq!(m.process("a", &state()).state, "warning");
        let d = m.process("a", &state());
        assert_eq!(d.state, "abort");
        assert_eq!(d.velocity, [0.0; 3]);
        assert_eq!(m.state("b"), "nominal");
        m.clear("a");
        assert_eq!(m.state("a"), "nominal");
    }
    #[test]
    fn simultaneous_violations_keep_terrain_floor() {
        let mut m =
            SimulationSafetyMonitor::new(DronePhysicalLimits::tracker_default(), 3, 0.9).unwrap();
        let mut s = state();
        s.terrain_height_m = 150.0;
        s.acceleration = Some([10.0, 0.0, 5.0]);
        s.energy_fraction = Some(0.1);
        let d = m.process("a", &s);
        assert_eq!(d.state, "abort");
        assert_eq!(d.position[2], 200.0);
        assert_eq!(d.velocity, [0.0; 3]);
        assert_eq!(d.violations.len(), 5);
    }
    #[test]
    fn finite_difference_acceleration_and_separation_are_checked() {
        let mut s = state();
        s.velocity = [0.0, 0.0, 0.0];
        s.previous_velocity = Some([0.0, 0.0, -4.0]);
        s.dt_s = Some(1.0);
        s.other_drone_positions = vec![[0.0, 0.0, 101.0]];
        let v = check(&DronePhysicalLimits::tracker_default(), &s);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].constraint, "max_vertical_accel");
        assert_eq!(v[1].constraint, "min_drone_separation");
    }
}
