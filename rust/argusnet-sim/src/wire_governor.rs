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

use crate::belief_world::{BeliefHandle, MissionFlightConfig, PhysicalGuard, WirePlanningConfig};
use crate::simulation::TickContext;

/// Where a platform was left, and how far from what it was commanded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Governed {
    pub position_m: [f64; 3],
    /// Flown height less commanded height, metres.
    pub offset_m: f64,
    /// A mission flight physics stopped at something the map does not hold
    /// climbs to this height before it tries the way again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub climb_to_m: Option<f64>,
    /// A mission flight, or the flight back from one, flew this platform
    /// last tick: whoever flies it next takes it from this height at the
    /// set rate, not at once.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub handed_over: bool,
}

/// Holds flights clear of the wires a map believes in.
#[derive(Debug, Clone)]
pub struct WireGovernor {
    /// Wire planning (W4); `None` governs for the surface alone.
    config: Option<WirePlanningConfig>,
    /// Keeps mission flights above the believed surface; `None` leaves
    /// them to the wires alone.
    surface: Option<MissionFlightConfig>,
    map: BeliefHandle,
    state: BTreeMap<String, Governed>,
    /// Mission flights the guard held this tick. Transient: a mission flight
    /// whose target lies inside a wire nobody mapped has no plan that helps,
    /// and physics has just said something is there.
    held: BTreeSet<String>,
}

impl WireGovernor {
    pub fn new(config: WirePlanningConfig, map: BeliefHandle) -> Self {
        Self::of(Some(config), None, map)
    }

    /// A governor for whichever of wire planning and mission flights a run
    /// asks for.
    pub fn of(
        config: Option<WirePlanningConfig>,
        surface: Option<MissionFlightConfig>,
        map: BeliefHandle,
    ) -> Self {
        Self {
            config,
            surface,
            map,
            state: BTreeMap::new(),
            held: BTreeSet::new(),
        }
    }

    /// Whether mission flights are kept above the believed surface.
    pub fn governs_mission_flights(&self) -> bool {
        self.surface.is_some()
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

    /// The height to fly from `flown_m` toward `commanded_m` for the wires
    /// believed, and whether a wire bounds it.
    fn wire_height(&self, flown: Governed, commanded_m: [f64; 3], dt_s: f64, held: bool) -> f64 {
        let Some(config) = &self.config else {
            return commanded_m[2];
        };
        let map = self.map.current();
        let [fx, fy, fz] = flown.position_m;
        let (dx, dy) = (commanded_m[0] - fx, commanded_m[1] - fy);
        let step = dx.hypot(dy);
        let direction = if step > 1e-6 {
            [dx / step, dy / step]
        } else {
            [0.0, 0.0]
        };
        let reach = (step / dt_s.max(1e-6) * config.lookahead_s).max(config.min_reach_m);
        let bounds = map.hazard_bounds_ahead([fx, fy], direction, reach, fz, config.clearance_m);
        let mut wanted = commanded_m[2];
        if let Some(ceiling) = bounds.ceiling_m {
            let lowest = map.ground_at(fx, fy) + config.min_agl_m;
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
            wanted = wanted.max(commanded_m[2] + 2.0 * config.clearance_m);
        }
        let bound = held || bounds.ceiling_m.is_some() || bounds.floor_m.is_some();
        if !bound && flown.offset_m == 0.0 {
            return commanded_m[2];
        }
        let reach_z = config.climb_mps * dt_s;
        wanted.clamp(fz - reach_z, fz + reach_z)
    }

    /// [`Self::wire_height`], under its old name for the tests.
    #[cfg(test)]
    fn height(&self, flown: Governed, commanded_m: [f64; 3], dt_s: f64, held: bool) -> f64 {
        self.wire_height(flown, commanded_m, dt_s, held)
    }

    /// Where a mission flight bound for `target_m` goes this tick, from
    /// `flown` toward `commanded_m` at `wire_z`: the height the believed
    /// surface ahead asks for, reached at the set rate, and whether it must
    /// wait where it is because the step itself crosses something it is not
    /// yet above.
    fn surface_step(
        &self,
        flown: Governed,
        commanded_m: [f64; 3],
        wire_z: f64,
        target_m: [f64; 2],
        dt_s: f64,
    ) -> ([f64; 3], bool) {
        let surface = self.surface.as_ref().expect("only called with a surface");
        let map = self.map.current();
        let [fx, fy, fz] = flown.position_m;
        let (dx, dy) = (commanded_m[0] - fx, commanded_m[1] - fy);
        let step = dx.hypot(dy);
        let direction = if step > 1e-6 {
            [dx / step, dy / step]
        } else {
            [0.0, 0.0]
        };
        let to_target = (target_m[0] - fx).hypot(target_m[1] - fy);
        let approaching = to_target <= surface.approach_radius_m;
        let reach_z = surface.climb_mps * dt_s;
        // What the surface asks of a segment from where the platform is.
        let needed = |length_m: f64| -> Option<f64> {
            let to = [fx + direction[0] * length_m, fy + direction[1] * length_m];
            let top = if approaching {
                // Coming down to the target: only structures on the way in.
                map.surface_top_along([fx, fy], to, 0.0, false, Some(target_m))
            } else {
                map.surface_top_along([fx, fy], to, surface.corridor_m, true, None)
            };
            top.map(|top| top + surface.clearance_m)
        };
        let reach = if approaching {
            step
        } else {
            // Not past where the approach begins: the target's own
            // surroundings are the approach's to judge.
            (step / dt_s.max(1e-6) * surface.lookahead_s)
                .max(surface.min_reach_m)
                .min((to_target - surface.approach_radius_m).max(step))
        };
        // The step's own strip as well as the look ahead: the two are
        // sampled apart, and a cell only the step clips must still be
        // climbed for, or the flight waits for a height nothing asks of it.
        let step_floor = needed(step);
        let mut wanted = wire_z;
        for floor in [needed(reach), step_floor].into_iter().flatten() {
            wanted = wanted.max(floor);
        }
        // Physics stopped it at something the map does not hold: up, where
        // it is, before trying the way again.
        let climbing = flown.climb_to_m.filter(|to| fz + 1e-9 < *to);
        if let Some(to) = climbing {
            wanted = wanted.max(to);
        }
        let z = wanted.clamp(fz - reach_z, fz + reach_z);
        // The step itself: what lies between here and the commanded place
        // must already be under the platform.
        let blocked = climbing.is_some() || step_floor.is_some_and(|floor| z + 1e-9 < floor);
        if blocked {
            ([fx, fy, z], true)
        } else {
            ([commanded_m[0], commanded_m[1], z], false)
        }
    }

    /// One step of a flight back to a platform's own path at `path_m`,
    /// from `from_m` toward `commanded_m`, kept above the believed surface
    /// as a mission flight is: the step itself when nothing is in the way.
    pub fn rejoin_step(
        &self,
        from_m: [f64; 3],
        commanded_m: [f64; 3],
        path_m: [f64; 2],
        dt_s: f64,
    ) -> [f64; 3] {
        if self.surface.is_none() {
            return commanded_m;
        }
        let flown = Governed {
            position_m: from_m,
            offset_m: 0.0,
            climb_to_m: None,
            handed_over: false,
        };
        self.surface_step(flown, commanded_m, commanded_m[2], path_m, dt_s)
            .0
    }

    /// Govern every mobile platform for this tick. `missions` names the
    /// platforms a mission flight is flying and where each is bound. Says
    /// how many were moved, and which mission flights waited where they
    /// were this tick: their flights are a tick later than planned.
    pub fn govern(
        &mut self,
        context: &mut TickContext<'_>,
        guard: Option<&PhysicalGuard>,
        missions: &BTreeMap<String, [f64; 2]>,
        rejoining: &BTreeSet<String>,
    ) -> (usize, Vec<String>) {
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
        let mut moved = 0;
        let mut waited = Vec::new();
        for (node_id, commanded_m) in mobile {
            let flown = self.state.get(&node_id).copied().unwrap_or(Governed {
                position_m: commanded_m,
                offset_m: 0.0,
                climb_to_m: None,
                handed_over: false,
            });
            let held = self.held.contains(&node_id);
            let wire_z = self.wire_height(flown, commanded_m, dt_s, held);
            let mut position = [commanded_m[0], commanded_m[1], wire_z];
            let mission = missions.get(&node_id).filter(|_| self.surface.is_some());
            let mut climb_to_m = flown.climb_to_m.filter(|_| mission.is_some());
            if let Some(target_m) = mission {
                let (stepped, wait) =
                    self.surface_step(flown, commanded_m, wire_z, *target_m, dt_s);
                position = stepped;
                if wait {
                    waited.push(node_id.clone());
                }
            } else if let (Some(surface), false) = (&self.surface, rejoining.contains(&node_id)) {
                // Handed back from a mission flight, it is taken to the
                // height it is now commanded at the set rate. A wire's
                // bound, already in `wire_z`, stands.
                // So is any other change of height: a controller that asks
                // for a new height at once (an explorer over newly seen
                // ground, a lane handed back) is flown there, not put there.
                // The first tick places a platform.
                if self.state.contains_key(&node_id) {
                    let reach_z = surface.climb_mps * dt_s;
                    let fz = flown.position_m[2];
                    position[2] = wire_z.clamp(fz - reach_z, fz + reach_z);
                }
            }
            if let Some(guard) = guard {
                if mission.is_some() {
                    // Swept: a mission flight never ends a tick beyond a
                    // wall it flew at.
                    if guard
                        .first_structure_hit(flown.position_m, position)
                        .is_some()
                        || guard.violates(position)
                    {
                        position =
                            guard.enforce_from(&node_id, timestamp_s, flown.position_m, position);
                        let clearance_m = self
                            .surface
                            .as_ref()
                            .map_or(0.0, |surface| surface.clearance_m);
                        climb_to_m = Some(position[2] + 2.0 * clearance_m);
                        if !waited.contains(&node_id) {
                            waited.push(node_id.clone());
                        }
                    }
                } else if guard.violates(position) {
                    position = guard.enforce(&node_id, timestamp_s, position);
                }
            }
            let offset_m = position[2] - commanded_m[2];
            if position != commanded_m {
                moved += 1;
                if let Some(platform) = context.world.platform_mut(&node_id) {
                    platform.position.x = position[0];
                    platform.position.y = position[1];
                    platform.position.z = position[2];
                }
            }
            let handed_over =
                self.surface.is_some() && (mission.is_some() || rejoining.contains(&node_id));
            self.state.insert(
                node_id,
                Governed {
                    position_m: position,
                    offset_m: if offset_m.abs() < 1e-9 { 0.0 } else { offset_m },
                    climb_to_m: climb_to_m.filter(|to| position[2] + 1e-9 < *to),
                    handed_over,
                },
            );
        }
        self.held.clear();
        (moved, waited)
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
            climb_to_m: None,
            handed_over: false,
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

    // --- mission flights over the believed surface ---

    use crate::belief_world::MissionFlightConfig;
    use crate::range_scan::{RangeReturn, RangeScan, ReturnClass};

    /// Flat measured ground at 0 m over a 400 m square, with a 30 m
    /// structure in the cell at x 100..110, y 0..10.
    fn surface_governor() -> WireGovernor {
        let mut grid =
            BeliefGrid::new(Bounds2d::new(-200.0, 200.0, -200.0, 200.0), 10.0, 0.0).unwrap();
        let hit = |x: f64, y: f64, z: f64, class| RangeReturn {
            point_m: [x, y, z],
            class,
            acquired_offset_s: None,
            acquired_from_m: None,
        };
        let mut returns = Vec::new();
        for j in 0..40 {
            for i in 0..40 {
                let (x, y) = (-195.0 + 10.0 * i as f64, -195.0 + 10.0 * j as f64);
                returns.push(hit(x, y, 0.0, ReturnClass::Ground));
            }
        }
        for _ in 0..12 {
            returns.push(hit(105.0, 5.0, 30.0, ReturnClass::Structure));
        }
        grid.integrate(
            &RangeScan {
                node_id: "d".into(),
                timestamp_s: 0.0,
                origin_m: [0.0, 0.0, 100.0],
                returns,
                misses: 0,
                wires: Vec::new(),
                sensor: Default::default(),
                pod: None,
                wire_points: Vec::new(),
                wire_lines: Vec::new(),
            },
            1,
        );
        let config = ExplorationConfig::default();
        let snapshot = grid.snapshot(&config);
        assert_eq!(snapshot.structure_top_at(105.0, 5.0), Some(30.0));
        WireGovernor::of(
            None,
            Some(MissionFlightConfig::default()),
            BeliefHandle::new(snapshot),
        )
    }

    fn at(x: f64, y: f64, z: f64) -> Governed {
        Governed {
            position_m: [x, y, z],
            offset_m: 0.0,
            climb_to_m: None,
            handed_over: false,
        }
    }

    const FAR: [f64; 2] = [190.0, 5.0];

    /// Over open measured ground, well above it, a mission flight is flown
    /// exactly as commanded.
    #[test]
    fn a_mission_flight_over_open_ground_is_untouched() {
        let governor = surface_governor();
        let (position, waits) = governor.surface_step(
            at(-150.0, -100.0, 40.0),
            [-143.0, -100.0, 40.0],
            40.0,
            [150.0, -100.0],
            0.25,
        );
        assert_eq!(position, [-143.0, -100.0, 40.0]);
        assert!(!waits);
    }

    /// A believed structure ahead is climbed for before it is reached, at
    /// the set rate, while the flight goes on.
    #[test]
    fn a_mission_flight_climbs_for_a_structure_ahead() {
        let governor = surface_governor();
        let (position, waits) =
            governor.surface_step(at(20.0, 5.0, 20.0), [27.0, 5.0, 20.0], 20.0, FAR, 0.25);
        assert_eq!(position[0], 27.0, "still far from it: the flight goes on");
        assert_eq!(position[2], 20.0 + 15.0 * 0.25, "up at the set rate");
        assert!(!waits);
    }

    /// One whose next step would cross the structure before it is above it
    /// waits where it is and climbs, and goes on once it is above.
    #[test]
    fn a_mission_flight_waits_until_it_is_above_what_its_step_crosses() {
        let governor = surface_governor();
        let (position, waits) =
            governor.surface_step(at(93.0, 5.0, 20.0), [100.5, 5.0, 20.0], 20.0, FAR, 0.25);
        assert_eq!(position[0], 93.0, "it does not fly into the cell");
        assert_eq!(position[2], 23.75);
        assert!(waits, "its plan is a tick later");

        let (position, waits) =
            governor.surface_step(at(93.0, 5.0, 38.0), [100.5, 5.0, 20.0], 20.0, FAR, 0.25);
        assert_eq!(position, [100.5, 5.0, 38.0], "30 m top plus 8 m clearance");
        assert!(!waits);
    }

    /// Unseen ground is not free: a flight toward a part of the map nobody
    /// has measured is held to the stand-in.
    #[test]
    fn unseen_ground_is_flown_over_at_its_stand_in() {
        let config = ExplorationConfig::default();
        let mut grid =
            BeliefGrid::new(Bounds2d::new(-200.0, 200.0, -200.0, 200.0), 10.0, 0.0).unwrap();
        let governor = WireGovernor::of(
            None,
            Some(MissionFlightConfig::default()),
            BeliefHandle::new(grid.snapshot(&config)),
        );
        let stand_in = governor.map.current().ground_at(50.0, 50.0);
        let (position, _) = governor.surface_step(
            at(0.0, 50.0, 5.0),
            [7.0, 50.0, 5.0],
            5.0,
            [150.0, 50.0],
            0.25,
        );
        assert!(stand_in >= config.unknown_margin_m);
        assert_eq!(
            position[2],
            5.0 + 15.0 * 0.25,
            "climbing toward {stand_in} m"
        );
    }

    /// Within the approach radius a flight comes down to its target: the
    /// ground and the target's own cell do not hold it up, a structure on
    /// the way in still does.
    #[test]
    fn a_mission_flight_comes_down_to_its_target() {
        let governor = surface_governor();
        // Landing on open ground, from above the clearance.
        let (position, waits) = governor.surface_step(
            at(-100.0, -100.0, 6.0),
            [-100.0, -100.0, 1.0],
            1.0,
            [-100.0, -100.0],
            0.25,
        );
        assert_eq!(position, [-100.0, -100.0, 6.0 - 3.75]);
        assert!(!waits);
        // A target in the structure's own cell: approached, not refused.
        let (position, waits) = governor.surface_step(
            at(98.0, 5.0, 45.0),
            [104.0, 5.0, 40.0],
            40.0,
            [105.0, 5.0],
            0.25,
        );
        assert_eq!(position[0], 104.0);
        assert!(!waits);
        // A target just past the structure: the structure is still cleared.
        let (position, waits) = governor.surface_step(
            at(98.0, 5.0, 20.0),
            [104.0, 5.0, 20.0],
            20.0,
            [115.0, 5.0],
            0.25,
        );
        assert_eq!(position[0], 98.0);
        assert!(waits);
    }

    /// After physics stops a flight at something the map does not hold, it
    /// climbs twice the clearance where it is before trying again.
    #[test]
    fn a_flight_physics_stopped_climbs_before_it_tries_again() {
        let governor = surface_governor();
        let mut stopped = at(-150.0, -100.0, 40.0);
        stopped.climb_to_m = Some(56.0);
        let (position, waits) =
            governor.surface_step(stopped, [-143.0, -100.0, 40.0], 40.0, [150.0, -100.0], 0.25);
        assert_eq!(position, [-150.0, -100.0, 43.75]);
        assert!(waits);
    }

    /// The state a checkpoint carries reads back, with and without a climb.
    #[test]
    fn governed_state_round_trips_and_reads_what_was_written_before() {
        let mut state = at(1.0, 2.0, 3.0);
        assert_eq!(
            serde_json::to_string(&state).unwrap(),
            r#"{"position_m":[1.0,2.0,3.0],"offset_m":0.0}"#,
            "no climb, nothing added to what was written before"
        );
        state.climb_to_m = Some(9.0);
        let back: Governed = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(back, state);
    }
}
