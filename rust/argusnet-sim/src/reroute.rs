//! Mission flights planned again when the map they were planned on changes under
//! them (P4.10c's affected routes, as P4.10d's jobs).
//!
//! A flight whose way was open when it was set is watched: every `check_s` the
//! way it has left is tested against the map as it is now. One that something
//! newly mapped lies across is planned again, on the map as it was then, as a
//! [`JobBoard`] job due `latency_s` later: planning takes time, and the run says
//! how long, so that it does not depend on the machine. What comes back is flown
//! only if the drone is still on that flight and the way is open on the map as it
//! is *then*. Until then, and if not, the height governor has the flight, as ever.
//!
//! A flight set anew, or dropped, cancels its job.

use std::collections::BTreeMap;
use std::sync::Mutex;

use argusnet_planning::battery::BatteryModel;
use argusnet_planning::route3d::{way_is_open, Plan3d, ReplanConfig};
use serde::{Deserialize, Serialize};

use crate::overrides::{GotoHover, OverrideCommand};
use crate::route_air::RouteAir;
use crate::scheduler::{JobBoard, JobState};

/// Corners this near a drone are behind it.
const PASSED_M: f64 = 15.0;
/// A flight this near its end is left to it.
const ARRIVING_M: f64 = 60.0;

/// Which table flies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlightKind {
    Poi,
    Egress,
}

/// A mission flight under way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatchedFlight {
    pub kind: FlightKind,
    /// The way left, its end last.
    pub way_m: Vec<[f64; 3]>,
    /// Whether the way was open when it was set. One that was not is the
    /// governor's.
    pub open: bool,
    pub set_tick: u64,
    pub checked_tick: u64,
}

/// What a rerouter checkpoints. Empty until a flight is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerouteState {
    pub flights: BTreeMap<String, WatchedFlight>,
    /// What each pending job found: the corners, or that there is no way.
    pub jobs: JobState<Option<Vec<[f64; 3]>>>,
}

/// What became of a job.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Fly this way from here.
    Reroute {
        kind: FlightKind,
        corners_m: Vec<[f64; 3]>,
        target_m: [f64; 3],
    },
    /// No way was found worth flying.
    NoWay,
    /// The way found is closed on the map as it is now.
    Closed,
}

/// A job that came due, and for whom.
#[derive(Debug, Clone, PartialEq)]
pub struct Landed {
    pub drone: String,
    pub submitted_tick: u64,
    pub outcome: Outcome,
}

/// Watches mission flights. Owned by the mission driver.
#[derive(Debug)]
pub struct Rerouter {
    air: RouteAir,
    replan: ReplanConfig,
    flights: BTreeMap<String, WatchedFlight>,
    jobs: Mutex<JobBoard<Option<Vec<[f64; 3]>>>>,
}

impl Rerouter {
    /// `None` when the configuration asks for no replanning.
    pub fn new(air: &RouteAir, workers: usize) -> Result<Option<Self>, String> {
        let Some(replan) = air.config.replan.clone() else {
            return Ok(None);
        };
        Ok(Some(Self {
            jobs: Mutex::new(JobBoard::new(replan.capacity).with_workers(workers)?),
            air: air.clone(),
            replan,
            flights: BTreeMap::new(),
        }))
    }

    /// The jobs' own threads; one works them out on the calling thread.
    pub fn set_workers(&mut self, workers: usize) -> Result<(), String> {
        *self.board() = JobBoard::new(self.replan.capacity).with_workers(workers)?;
        Ok(())
    }

    fn board(&mut self) -> &mut JobBoard<Option<Vec<[f64; 3]>>> {
        self.jobs.get_mut().expect("jobs lock")
    }

    pub fn pending(&mut self) -> usize {
        self.board().len()
    }

    pub fn reset(&mut self) {
        self.flights.clear();
        self.board().clear();
    }

    pub fn state(&self) -> Option<RerouteState> {
        let jobs = self.jobs.lock().expect("jobs lock").state();
        (!self.flights.is_empty() || !jobs.is_new()).then(|| RerouteState {
            flights: self.flights.clone(),
            jobs,
        })
    }

    pub fn check(&self, state: &RerouteState) -> Result<(), String> {
        self.jobs.lock().expect("jobs lock").check(&state.jobs)
    }

    pub fn restore(&mut self, state: RerouteState) {
        self.flights = state.flights;
        self.board().restore(state.jobs);
    }

    /// The mission has written `command` at `tick`.
    pub fn note(&mut self, command: &OverrideCommand, tick: u64) {
        let (drone, flight) = match command {
            OverrideCommand::SetPoi(id, flight) => (id, Some((FlightKind::Poi, flight))),
            OverrideCommand::SetEgress(id, flight) => (id, Some((FlightKind::Egress, flight))),
            OverrideCommand::ClearPoi(id) => {
                // A flight home outlives the POI it was sent from.
                if self
                    .flights
                    .get(id)
                    .is_some_and(|flight| flight.kind == FlightKind::Egress)
                {
                    return;
                }
                (id, None)
            }
            OverrideCommand::ClearAllEgress => {
                let homeward: Vec<String> = self
                    .flights
                    .iter()
                    .filter(|(_, flight)| flight.kind == FlightKind::Egress)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in homeward {
                    self.flights.remove(&id);
                    self.board().cancel(&id);
                }
                return;
            }
            OverrideCommand::SetFrontier(..) | OverrideCommand::ClearFrontier(_) => return,
        };
        self.board().cancel(drone);
        match flight {
            None => {
                self.flights.remove(drone);
            }
            Some((kind, flight)) => {
                let way_m = flight.way_m();
                let open = way_is_open(
                    &self.air.columns(&self.air.map.current()),
                    flight.start(),
                    &way_m,
                    &self.air.config,
                );
                self.flights.insert(
                    drone.clone(),
                    WatchedFlight {
                        kind,
                        way_m,
                        open,
                        set_tick: tick,
                        checked_tick: tick,
                    },
                );
            }
        }
    }

    /// Take what is due at `tick`, then look at what is flying.
    pub fn run(
        &mut self,
        tick: u64,
        dt_s: f64,
        positions: &[(String, [f64; 3])],
        battery: &BatteryModel,
    ) -> Vec<Landed> {
        let map = self.air.map.current();
        let mut landed = Vec::new();
        for done in self.board().take_due(tick) {
            let Some(at) = positions
                .iter()
                .find(|(id, _)| *id == done.key)
                .map(|(_, at)| *at)
            else {
                continue;
            };
            let Some(flight) = self.flights.get_mut(&done.key) else {
                continue;
            };
            let target_m = *flight.way_m.last().expect("a way has an end");
            let outcome = match done.value {
                None => {
                    // Nothing to be had: the governor's from here.
                    flight.open = false;
                    Outcome::NoWay
                }
                Some(corners) => {
                    let mut corners_m = corners;
                    while corners_m
                        .first()
                        .is_some_and(|c| (c[0] - at[0]).hypot(c[1] - at[1]) < PASSED_M)
                    {
                        corners_m.remove(0);
                    }
                    let mut way = corners_m.clone();
                    way.push(target_m);
                    if way_is_open(&self.air.columns(&map), at, &way, &self.air.config) {
                        Outcome::Reroute {
                            kind: flight.kind,
                            corners_m,
                            target_m,
                        }
                    } else {
                        Outcome::Closed
                    }
                }
            };
            landed.push(Landed {
                drone: done.key,
                submitted_tick: done.submitted_tick,
                outcome,
            });
        }
        let ticks = |seconds: f64| (seconds / dt_s).round().max(1.0) as u64;
        let (check, latency) = (ticks(self.replan.check_s), ticks(self.replan.latency_s));
        for (drone, at) in positions {
            let Some(flight) = self.flights.get_mut(drone) else {
                continue;
            };
            let end = *flight.way_m.last().expect("a way has an end");
            if (end[0] - at[0]).hypot(end[1] - at[1]) < ARRIVING_M {
                continue;
            }
            // The corners behind it.
            while flight.way_m.len() > 1 {
                let next = flight.way_m[0];
                let after = flight.way_m[1];
                let near = (next[0] - at[0]).hypot(next[1] - at[1]) < PASSED_M;
                // Past it, on the leg after it.
                let past = (after[0] - next[0]) * (at[0] - next[0])
                    + (after[1] - next[1]) * (at[1] - next[1])
                    > 0.0
                    && (after[0] - at[0]).hypot(after[1] - at[1])
                        < (after[0] - next[0]).hypot(after[1] - next[1]);
                if near || past {
                    flight.way_m.remove(0);
                } else {
                    break;
                }
            }
            if !flight.open || tick < flight.checked_tick + check {
                continue;
            }
            flight.checked_tick = tick;
            if way_is_open(
                &self.air.columns(&map),
                *at,
                &flight.way_m,
                &self.air.config,
            ) {
                continue;
            }
            let (air, map, battery, start) = (self.air.clone(), map.clone(), *battery, *at);
            // Refused when one is pending or the board is full: it is looked at
            // again at the next check.
            let _ = self.jobs.get_mut().expect("jobs lock").submit(
                drone,
                tick,
                tick + latency,
                tick,
                move || match air.plan_on(&map, &battery, start, end) {
                    Plan3d::Straight => Some(Vec::new()),
                    Plan3d::Route(route) => Some(route.corners_m),
                    Plan3d::NotWorth | Plan3d::None => None,
                },
            );
        }
        landed
    }
}

impl Outcome {
    /// Its name in a decision.
    pub fn reason(&self) -> &'static str {
        match self {
            Outcome::Reroute { .. } => "map_changed",
            Outcome::NoWay => "no_way_worth_flying",
            Outcome::Closed => "closed_since",
        }
    }

    /// The flight it asks for from `at`.
    pub fn flight(&self, at: [f64; 3], t_start: f64, speed_mps: f64) -> Option<GotoHover> {
        let Outcome::Reroute {
            corners_m,
            target_m,
            ..
        } = self
        else {
            return None;
        };
        Some(GotoHover::routed_over(
            at,
            corners_m.iter().map(|c| [c[0], c[1]]).collect(),
            corners_m.iter().map(|c| c[2]).collect(),
            [target_m[0], target_m[1]],
            target_m[2],
            t_start,
            speed_mps,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_world::{BeliefGrid, BeliefHandle, ExplorationConfig};
    use crate::range_scan::{RangeReturn, RangeScan, ReturnClass};
    use argusnet_planning::route3d::Route3dConfig;

    /// Flat ground, with structures `top` m high in the cells about `at`.
    fn map(at: &[[f64; 2]], top: f64) -> crate::belief_world::BeliefSnapshot {
        let bounds = argusnet_types::Bounds2d::new(-500.0, 500.0, -500.0, 500.0);
        let mut grid = BeliefGrid::new(bounds, 10.0, 0.0).unwrap();
        let hit = |x: f64, y: f64, z: f64, class| RangeReturn {
            point_m: [x, y, z],
            class,
            acquired_offset_s: None,
            acquired_from_m: None,
        };
        let mut returns: Vec<RangeReturn> = (0..10_000)
            .map(|k| {
                let (i, j) = (f64::from(k % 100), f64::from(k / 100));
                let (x, y) = (-495.0 + 10.0 * i, -495.0 + 10.0 * j);
                hit(x, y, 0.0, ReturnClass::Ground)
            })
            .collect();
        for [x, y] in at {
            returns.extend((0..12).map(|_| hit(*x, *y, top, ReturnClass::Structure)));
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
                pose: None,
                wire_points: Vec::new(),
                wire_lines: Vec::new(),
            },
            0,
        );
        grid.snapshot(&ExplorationConfig::default())
    }

    const START: [f64; 3] = [-300.0, 5.0, 50.0];
    const GOAL: [f64; 3] = [300.0, 5.0, 50.0];
    const TOWER: [f64; 2] = [55.0, 5.0];

    fn rerouter(workers: usize) -> (Rerouter, BeliefHandle) {
        let handle = BeliefHandle::new(map(&[], 0.0));
        let air = RouteAir {
            config: Route3dConfig {
                replan: Some(ReplanConfig {
                    check_s: 1.0,
                    latency_s: 2.0,
                    capacity: 4,
                }),
                ..Route3dConfig::default()
            },
            map: handle.clone(),
            surface: Default::default(),
            wires: None,
        };
        (Rerouter::new(&air, workers).unwrap().unwrap(), handle)
    }

    fn set(rerouter: &mut Rerouter, tick: u64) {
        let flight = GotoHover::redirect(START, [GOAL[0], GOAL[1]], GOAL[2], 0.0, 20.0);
        rerouter.note(&OverrideCommand::SetPoi("a".into(), flight), tick);
    }

    fn run(rerouter: &mut Rerouter, tick: u64, x: f64) -> Vec<Landed> {
        let at = vec![("a".to_string(), [x, 5.0, 50.0])];
        rerouter.run(tick, 0.5, &at, &BatteryModel::default())
    }

    /// Nothing until something is mapped across the way; then a way round it,
    /// two seconds after it was asked for, whatever the workers.
    #[test]
    fn a_tower_mapped_on_the_way_is_gone_round() {
        for workers in [1, 3] {
            let (mut rerouter, handle) = rerouter(workers);
            set(&mut rerouter, 0);
            for tick in 1..10 {
                assert_eq!(run(&mut rerouter, tick, -300.0 + 10.0 * tick as f64), []);
            }
            assert_eq!(rerouter.pending(), 0);
            handle.publish(map(&[TOWER], 150.0));
            let mut landed = Vec::new();
            for tick in 10..20 {
                for found in run(&mut rerouter, tick, -200.0) {
                    // Set, as the mission sets it, it is open and left alone.
                    let at = [-200.0, 5.0, 50.0];
                    let flight = found.outcome.flight(at, 0.0, 20.0).expect("a flight");
                    rerouter.note(&OverrideCommand::SetPoi("a".into(), flight), tick);
                    landed.push((tick, found));
                }
            }
            assert_eq!(landed.len(), 1, "{landed:?}");
            let (tick, found) = &landed[0];
            assert_eq!(found.submitted_tick, 10);
            assert_eq!(*tick, 14, "4 ticks of half a second");
            let Outcome::Reroute {
                kind,
                corners_m,
                target_m,
            } = &found.outcome
            else {
                panic!("{found:?}")
            };
            assert_eq!((*kind, *target_m), (FlightKind::Poi, GOAL));
            assert!(corners_m.iter().any(|c| (c[1] - 5.0).abs() > 15.0));
            assert!(corners_m.iter().all(|c| c[2] < 90.0), "{corners_m:?}");
        }
    }

    /// What is found is not flown if the map has closed it meanwhile.
    #[test]
    fn a_way_closed_since_is_not_flown() {
        let (mut rerouter, handle) = rerouter(1);
        set(&mut rerouter, 0);
        handle.publish(map(&[TOWER], 150.0));
        assert_eq!(run(&mut rerouter, 2, -200.0), []);
        assert_eq!(rerouter.pending(), 1);
        // A wall of them, both ways round.
        let wall: Vec<[f64; 2]> = (0..40).map(|k| [55.0, -195.0 + 10.0 * k as f64]).collect();
        handle.publish(map(&wall, 150.0));
        let landed = run(&mut rerouter, 6, -200.0);
        assert_eq!(landed.len(), 1);
        assert_eq!(landed[0].outcome, Outcome::Closed);
    }

    #[test]
    fn a_flight_set_anew_or_dropped_cancels_its_job() {
        let (mut rerouter, handle) = rerouter(1);
        set(&mut rerouter, 0);
        handle.publish(map(&[TOWER], 150.0));
        run(&mut rerouter, 2, -200.0);
        assert_eq!(rerouter.pending(), 1);
        rerouter.note(&OverrideCommand::ClearPoi("a".into()), 3);
        assert_eq!(rerouter.pending(), 0);
        assert_eq!(run(&mut rerouter, 6, -200.0), []);
        // Set across what the map already holds, it is the governor's.
        set(&mut rerouter, 7);
        for tick in 8..20 {
            assert_eq!(run(&mut rerouter, tick, -200.0), []);
        }
    }

    /// Stopped with a job pending, it lands as it would have; the plan is the
    /// one made on the map the job was asked on, not on a later one.
    #[test]
    fn it_resumes_with_its_jobs_pending() {
        let (mut straight, handle) = rerouter(2);
        set(&mut straight, 0);
        handle.publish(map(&[TOWER], 150.0));
        run(&mut straight, 2, -200.0);
        let text = serde_json::to_string(&straight.state().unwrap()).unwrap();
        let state: RerouteState = serde_json::from_str(&text).unwrap();
        let (mut resumed, other) = rerouter(1);
        other.publish(map(&[TOWER, [55.0, 305.0]], 150.0));
        handle.publish(map(&[TOWER, [55.0, 305.0]], 150.0));
        resumed.check(&state).unwrap();
        resumed.restore(state);
        for tick in 3..9 {
            let (a, b) = (
                run(&mut straight, tick, -200.0),
                run(&mut resumed, tick, -200.0),
            );
            assert_eq!(a, b, "tick {tick}");
        }
        assert_eq!(straight.state(), resumed.state());
    }

    #[test]
    fn without_replanning_there_is_no_rerouter() {
        let air = RouteAir {
            config: Route3dConfig::default(),
            map: BeliefHandle::new(map(&[], 0.0)),
            surface: Default::default(),
            wires: None,
        };
        assert!(Rerouter::new(&air, 1).unwrap().is_none());
    }
}
