//! Height governor (W4): the last word on every flight near a believed wire.
//!
//! W2 had explorers pass fitted wires by height, and left mission flights to
//! route planning, which can only plan round what is mapped when a flight
//! starts. A wire mapped mid-flight waits for a replan, and a hover beside a
//! POI has no plan at all. This runs after mission overrides, on what every
//! mobile platform is about to fly, whoever is flying it:
//!
//! - It looks ahead along the flight for what the belief holds: fitted wires,
//!   corridors, poles, and evidence no track has confirmed yet.
//! - A flight at or above a wire's height goes over, one clear below stays
//!   below, and one already under stays under ([`BeliefSnapshot::hazard_bounds_ahead`]).
//! - Height is changed at a set rate, from where the platform last was, so a
//!   step through a wire is never commanded. Away from any wire the offset is
//!   flown off at the same rate, and a flight with no offset is untouched.
//!
//! It reads the belief only. Truth stays with the physical guard, which
//! still has the final say on where a platform can be.
//!
//! [`BeliefSnapshot::hazard_bounds_ahead`]: crate::belief_world::BeliefSnapshot::hazard_bounds_ahead

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::belief_world::{BeliefHandle, PhysicalGuard, WirePlanningConfig};
use crate::simulation::TickContext;

/// Where a platform was left, and how far from what it was commanded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Governed {
    pub position_m: [f64; 3],
    /// Flown height less commanded height, metres.
    pub offset_m: f64,
}

/// Holds flights clear of the wires a map believes in.
#[derive(Debug, Clone)]
pub struct WireGovernor {
    config: WirePlanningConfig,
    map: BeliefHandle,
    state: BTreeMap<String, Governed>,
    /// Mission flights the guard held this tick. Transient: a mission flight
    /// whose target lies inside a wire nobody mapped has no plan that helps,
    /// and physics has just said something is there.
    held: BTreeSet<String>,
}

impl WireGovernor {
    pub fn new(config: WirePlanningConfig, map: BeliefHandle) -> Self {
        Self {
            config,
            map,
            state: BTreeMap::new(),
            held: BTreeSet::new(),
        }
    }

    /// The guard held `node_id`'s mission flight this tick.
    pub fn held_by_guard(&mut self, node_id: &str) {
        self.held.insert(node_id.to_string());
    }

    pub fn state(&self) -> &BTreeMap<String, Governed> {
        &self.state
    }

    pub fn restore(&mut self, state: BTreeMap<String, Governed>) {
        self.state = state;
    }

    pub fn reset(&mut self) {
        self.state.clear();
        self.held.clear();
    }

    /// The height to fly from `flown_m` toward `commanded_m`, and how far
    /// that is from the command.
    fn height(&self, flown: Governed, commanded_m: [f64; 3], dt_s: f64, held: bool) -> f64 {
        let map = self.map.current();
        let [fx, fy, fz] = flown.position_m;
        let (dx, dy) = (commanded_m[0] - fx, commanded_m[1] - fy);
        let step = dx.hypot(dy);
        let direction = if step > 1e-6 {
            [dx / step, dy / step]
        } else {
            [0.0, 0.0]
        };
        let reach = (step / dt_s.max(1e-6) * self.config.lookahead_s).max(self.config.min_reach_m);
        let bounds =
            map.hazard_bounds_ahead([fx, fy], direction, reach, fz, self.config.clearance_m);
        let mut wanted = commanded_m[2];
        if let Some(ceiling) = bounds.ceiling_m {
            let lowest = map.ground_at(fx, fy) + self.config.min_agl_m;
            let room = commanded_m[2].min(ceiling) >= lowest && commanded_m[2] <= ceiling + 15.0;
            if room {
                wanted = wanted.min(ceiling);
            } else if let Some(over) = bounds.over_m {
                // No room under it: a wire over a hilltop is gone over.
                wanted = wanted.max(over);
            }
        }
        if let Some(floor) = bounds.floor_m {
            wanted = wanted.max(floor);
        }
        // Held by physics at something unmapped: up and off it, and no
        // further than the clearance kept from what is known.
        if held {
            wanted = wanted.max(commanded_m[2] + 2.0 * self.config.clearance_m);
        }
        let bound = held || bounds.ceiling_m.is_some() || bounds.floor_m.is_some();
        if !bound && flown.offset_m == 0.0 {
            return commanded_m[2];
        }
        let reach_z = self.config.climb_mps * dt_s;
        wanted.clamp(fz - reach_z, fz + reach_z)
    }

    /// Govern every mobile platform for this tick, and say how many were
    /// moved.
    pub fn govern(
        &mut self,
        context: &mut TickContext<'_>,
        guard: Option<&PhysicalGuard>,
    ) -> usize {
        let dt_s = context.dt_s;
        let timestamp_s = context.timestamp_s;
        let mobile: Vec<(String, [f64; 3])> = context
            .world
            .mobile_platforms()
            .map(|platform| {
                (
                    platform.node_id.clone(),
                    [
                        platform.position.x,
                        platform.position.y,
                        platform.position.z,
                    ],
                )
            })
            .collect();
        let mut held = 0;
        for (node_id, commanded_m) in mobile {
            let flown = self.state.get(&node_id).copied().unwrap_or(Governed {
                position_m: commanded_m,
                offset_m: 0.0,
            });
            let z = self.height(flown, commanded_m, dt_s, self.held.contains(&node_id));
            let mut position = [commanded_m[0], commanded_m[1], z];
            if let Some(guard) = guard.filter(|guard| guard.violates(position)) {
                position = guard.enforce(&node_id, timestamp_s, position);
            }
            let offset_m = position[2] - commanded_m[2];
            if position != commanded_m {
                held += 1;
                if let Some(platform) = context.world.platform_mut(&node_id) {
                    platform.position.x = position[0];
                    platform.position.y = position[1];
                    platform.position.z = position[2];
                }
            }
            self.state.insert(
                node_id,
                Governed {
                    position_m: position,
                    offset_m: if offset_m.abs() < 1e-9 { 0.0 } else { offset_m },
                },
            );
        }
        self.held.clear();
        held
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_world::{BeliefGrid, ExplorationConfig};
    use argusnet_types::Bounds2d;
    use argusnet_world::powerline::{PowerLine, Support};

    /// A fitted wire along y at x = 0, 100 m up on ground at 0, 10 m clear.
    fn governor() -> WireGovernor {
        let support = |y: f64| Support {
            xy_m: [0.0, y],
            base_z_m: 99.0,
            height_m: 1.0,
        };
        let wire = PowerLine {
            id: "wire-track-0".into(),
            supports: vec![support(-400.0), support(400.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.0,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap();
        let config = ExplorationConfig::default();
        let mut grid =
            BeliefGrid::new(Bounds2d::new(-500.0, 500.0, -500.0, 500.0), 10.0, 0.0).unwrap();
        grid.know_wires(vec![wire]);
        WireGovernor::new(
            WirePlanningConfig::default(),
            BeliefHandle::new(grid.snapshot(&config)),
        )
    }

    fn flown(x: f64, z: f64) -> Governed {
        Governed {
            position_m: [x, 0.0, z],
            offset_m: 0.0,
        }
    }

    #[test]
    fn a_flight_away_from_wires_is_untouched() {
        let z = governor().height(flown(-400.0, 95.0), [-380.0, 0.0, 80.0], 1.0, false);
        assert_eq!(z, 80.0);
    }

    #[test]
    fn a_flight_at_wire_height_climbs_over_at_the_set_rate() {
        let z = governor().height(flown(-100.0, 95.0), [-80.0, 0.0, 95.0], 1.0, false);
        assert_eq!(z, 110.0, "15 m/s up, not a jump to the floor");
    }

    #[test]
    fn a_flight_clear_below_stays_below() {
        let z = governor().height(flown(-100.0, 60.0), [-80.0, 0.0, 95.0], 1.0, false);
        assert_eq!(z, 75.0, "toward the 90 m ceiling at the set rate");
        let z = governor().height(flown(-100.0, 88.0), [-80.0, 0.0, 95.0], 1.0, false);
        assert_eq!(z, 90.0, "no higher than the ceiling");
    }

    #[test]
    fn an_offset_is_flown_off_once_clear() {
        let mut held = flown(-400.0, 120.0);
        held.offset_m = 40.0;
        let z = governor().height(held, [-380.0, 0.0, 80.0], 1.0, false);
        assert_eq!(z, 105.0, "back down at the set rate, not at once");
    }

    /// A flight held at something unmapped climbs off it, however its
    /// command stays put, and stops at twice the clearance over the command.
    #[test]
    fn a_flight_the_guard_holds_climbs_off_what_stopped_it() {
        let z = governor().height(flown(-400.0, 80.0), [-400.0, 0.0, 80.0], 1.0, true);
        assert_eq!(z, 95.0);
        let mut high = flown(-400.0, 100.0);
        high.offset_m = 20.0;
        let z = governor().height(high, [-400.0, 0.0, 80.0], 1.0, true);
        assert_eq!(z, 100.0);
    }
}
