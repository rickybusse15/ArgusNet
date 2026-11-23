//! Whether a point to inspect was seen well enough, and going back when it was not
//! (F7).
//!
//! A point was inspected when a drone had hovered over it for its dwell. That says
//! nothing of what was seen: one look straight down, or at a wall. With a
//! requirement, a point also asks for views of it: so many, from so far round it, from
//! near enough and neither level nor straight down, with nothing in the way. Once every
//! dwell is done, drones with energy go round the points still short, one viewpoint at a
//! time, until each is **met**, or is **unmet** with the reason: no viewpoint is left,
//! its tries or the time are spent, or every drone is going home. A point is never
//! called met on fewer views, and never holds the mission past its bounds.
//!
//! A viewpoint is chosen on the map the drones built. Whether the view from it shows the
//! point is asked of the image taken there, which here is the world.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// What a point's views must be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceConfig {
    pub min_views: usize,
    /// How far apart, round the point, any two of them are.
    pub min_separation_deg: f64,
    pub max_range_m: f64,
    /// The elevations a view is taken from, above level at the point.
    pub elevation_deg: [f64; 2],
    /// Where viewpoints are: this far from the point and this high over it.
    pub standoff_m: f64,
    pub height_m: f64,
    /// How many places round the point are considered.
    pub places: usize,
    /// How long a drone holds at one to take its image, seconds.
    pub hold_s: f64,
    /// The most viewpoints flown to for one point.
    pub max_tries: usize,
    /// The most time given to all of them, seconds.
    pub max_s: f64,
}

impl Default for EvidenceConfig {
    fn default() -> Self {
        Self {
            min_views: 3,
            min_separation_deg: 80.0,
            max_range_m: 60.0,
            elevation_deg: [15.0, 75.0],
            standoff_m: 35.0,
            height_m: 30.0,
            places: 12,
            hold_s: 2.0,
            max_tries: 8,
            max_s: 120.0,
        }
    }
}

impl EvidenceConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("min_separation_deg", self.min_separation_deg),
            ("max_range_m", self.max_range_m),
            ("standoff_m", self.standoff_m),
            ("height_m", self.height_m),
            ("hold_s", self.hold_s),
            ("max_s", self.max_s),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("inspection evidence {name} cannot be {value}"));
            }
        }
        let [low, high] = self.elevation_deg;
        if !(0.0..90.0).contains(&low) || !(high > low && high <= 90.0) {
            return Err("inspection evidence elevations must rise from 0 to 90 degrees".into());
        }
        if self.min_views == 0 || self.places == 0 || self.max_tries == 0 {
            return Err("inspection evidence needs a view, a place and a try".into());
        }
        if self.min_views as f64 * self.min_separation_deg > 360.0 + 1e-9 {
            return Err(format!(
                "{} views cannot be {} degrees apart round a point",
                self.min_views, self.min_separation_deg
            ));
        }
        let range = self.standoff_m.hypot(self.height_m);
        let elevation = self.height_m.atan2(self.standoff_m).to_degrees();
        if range > self.max_range_m || elevation < low || elevation > high {
            return Err(format!(
                "a viewpoint {range:.0} m off at {elevation:.0} degrees is not one it asks for"
            ));
        }
        Ok(())
    }
}

/// A view that shows the point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct View {
    pub drone_id: String,
    pub timestamp_s: f64,
    /// From the point, degrees counterclockwise of east, 0 to 360.
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub range_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Met,
    /// Nowhere left to see it from.
    NoViewpoint,
    TriesSpent,
    TimeSpent,
    /// No drone is left to go.
    FleetHomeward,
}

/// One point's evidence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PointEvidence {
    /// What its views are of, as the map had it when it was first asked.
    pub subject_m: [f64; 3],
    pub views: Vec<View>,
    /// Places flown to whose view did not show it, degrees.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_deg: Vec<f64>,
    pub tries: usize,
    /// `None` while it is still open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
}

/// A drone on its way to a viewpoint, or at it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tour {
    pub poi_id: String,
    pub azimuth_deg: f64,
    pub viewpoint_m: [f64; 3],
    /// When it got there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrived_s: Option<f64>,
}

/// What is checkpointed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceState {
    pub points: BTreeMap<String, PointEvidence>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tours: BTreeMap<String, Tour>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_s: Option<f64>,
}

/// What the drones' own map says, and what an image would show.
pub trait Eyes {
    /// Height to fly at over a place at least.
    fn floor_m(&self, xy_m: [f64; 2]) -> f64;
    /// Whether, as the map has it, a point can be seen from a place.
    fn expects_to_see(&self, from_m: [f64; 3], point_m: [f64; 3]) -> bool;
    /// Whether the image `drone_id` takes where it believes it is shows it.
    fn shows(&self, drone_id: &str, from_m: [f64; 3], point_m: [f64; 3]) -> bool;
}

/// The drones' own map, and the world for what an image would show.
#[derive(Debug, Clone)]
pub struct MapEyes {
    pub map: crate::belief_world::BeliefHandle,
    /// Height kept over the believed surface.
    pub clearance_m: f64,
    pub world: crate::belief_world::PhysicalGuard,
}

impl MapEyes {
    /// What of a place is inspected: the top of what the map has there.
    pub fn subject_m(&self, xy_m: [f64; 2]) -> [f64; 3] {
        let map = self.map.current();
        let ground = map.ground_at(xy_m[0], xy_m[1]);
        let top = map
            .structure_top_at(xy_m[0], xy_m[1])
            .map_or(ground, |top| top.max(ground));
        [xy_m[0], xy_m[1], top + 0.5]
    }
}

impl Eyes for MapEyes {
    fn floor_m(&self, xy_m: [f64; 2]) -> f64 {
        let map = self.map.current();
        map.surface_top_along(xy_m, xy_m, self.clearance_m, true, None)
            .unwrap_or(f64::INFINITY)
            + self.clearance_m
    }

    fn expects_to_see(&self, from_m: [f64; 3], point_m: [f64; 3]) -> bool {
        use crate::follow_path::VisibilitySampler;
        crate::belief_world::BeliefVisibility::new(self.map.clone(), 0.0)
            .is_visible(from_m, point_m)
    }

    fn shows(&self, drone_id: &str, from_m: [f64; 3], point_m: [f64; 3]) -> bool {
        self.world.sees(drone_id, from_m, point_m)
    }
}

/// What a run of it asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Fly `drone_id` to a viewpoint of `poi_id`.
    Go {
        drone_id: String,
        poi_id: String,
        viewpoint_m: [f64; 3],
        azimuth_deg: f64,
    },
    /// The drone is done.
    Release { drone_id: String },
    /// A view was taken there: shown or not.
    Viewed {
        drone_id: String,
        poi_id: String,
        azimuth_deg: f64,
        shown: bool,
    },
    /// The point is settled.
    Settled {
        poi_id: String,
        verdict: Verdict,
        views: usize,
    },
}

fn apart_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// Within this of a viewpoint a drone is at it, metres in plan.
const ARRIVED_M: f64 = 6.0;

impl EvidenceState {
    /// Whether `azimuth` is as far from every view kept as is asked.
    fn adds(config: &EvidenceConfig, point: &PointEvidence, azimuth: f64) -> bool {
        point
            .views
            .iter()
            .all(|view| apart_deg(view.azimuth_deg, azimuth) >= config.min_separation_deg - 1e-9)
    }

    /// One run, every dwell done. `points` are those asked about, where each is;
    /// `working` the drones that may go, and where they are. Returns what to do;
    /// nothing is open once every point is settled ([`Self::is_open`]).
    pub fn run(
        &mut self,
        config: &EvidenceConfig,
        timestamp_s: f64,
        points: &[(String, [f64; 3])],
        working: &[(String, [f64; 3])],
        eyes: &impl Eyes,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        let started = *self.started_at_s.get_or_insert(timestamp_s);
        for (poi_id, subject_m) in points {
            self.points
                .entry(poi_id.clone())
                .or_insert_with(|| PointEvidence {
                    subject_m: *subject_m,
                    ..PointEvidence::default()
                });
        }
        let where_is: BTreeMap<&str, [f64; 3]> =
            points.iter().map(|(id, at)| (id.as_str(), *at)).collect();
        // Drones that may no longer go give up their viewpoints.
        let may: BTreeSet<&str> = working.iter().map(|(id, _)| id.as_str()).collect();
        let gone: Vec<String> = self
            .tours
            .keys()
            .filter(|drone| !may.contains(drone.as_str()))
            .cloned()
            .collect();
        for drone_id in gone {
            self.tours.remove(&drone_id);
            actions.push(Action::Release { drone_id });
        }
        // Views taken.
        for (drone, at) in working {
            let Some(tour) = self.tours.get_mut(drone) else {
                continue;
            };
            let off = (tour.viewpoint_m[0] - at[0]).hypot(tour.viewpoint_m[1] - at[1]);
            if off > ARRIVED_M {
                continue;
            }
            let arrived = *tour.arrived_s.get_or_insert(timestamp_s);
            if timestamp_s < arrived + config.hold_s - 1e-9 {
                continue;
            }
            let tour = self.tours.remove(drone).expect("just found");
            let point_m = where_is[tour.poi_id.as_str()];
            let point = self.points.get_mut(&tour.poi_id).expect("listed");
            let d: [f64; 3] = std::array::from_fn(|k| at[k] - point_m[k]);
            let plan = d[0].hypot(d[1]);
            let (range, elevation) = (plan.hypot(d[2]), d[2].atan2(plan).to_degrees());
            let azimuth = d[1].atan2(d[0]).to_degrees().rem_euclid(360.0);
            let [low, high] = config.elevation_deg;
            let shown = range <= config.max_range_m
                && (low..=high).contains(&elevation)
                && Self::adds(config, point, azimuth)
                && eyes.shows(drone, *at, point_m);
            if shown {
                point.views.push(View {
                    drone_id: drone.clone(),
                    timestamp_s,
                    azimuth_deg: azimuth,
                    elevation_deg: elevation,
                    range_m: range,
                });
            } else {
                point.failed_deg.push(tour.azimuth_deg);
            }
            actions.push(Action::Viewed {
                drone_id: drone.clone(),
                poi_id: tour.poi_id.clone(),
                azimuth_deg: azimuth,
                shown,
            });
            if point.views.len() >= config.min_views {
                point.verdict = Some(Verdict::Met);
                actions.push(Action::Settled {
                    poi_id: tour.poi_id,
                    verdict: Verdict::Met,
                    views: point.views.len(),
                });
            }
            actions.push(Action::Release {
                drone_id: drone.clone(),
            });
        }
        let spent = timestamp_s >= started + config.max_s - 1e-9;
        // Each free drone to the nearest viewpoint worth flying to, one drone a
        // point at a time.
        if !spent {
            for (drone, at) in working {
                if self.tours.contains_key(drone) {
                    continue;
                }
                let held: BTreeSet<&str> = self.tours.values().map(|t| t.poi_id.as_str()).collect();
                let mut best: Option<(f64, &str, f64, [f64; 3])> = None;
                for (poi_id, point_m) in points {
                    let point = &self.points[poi_id];
                    if point.verdict.is_some()
                        || point.tries >= config.max_tries
                        || held.contains(poi_id.as_str())
                    {
                        continue;
                    }
                    for (azimuth, viewpoint) in viewpoints(config, point, *point_m, eyes) {
                        let far = (viewpoint[0] - at[0]).hypot(viewpoint[1] - at[1]);
                        if best.is_none_or(|(least, ..)| far < least - 1e-9) {
                            best = Some((far, poi_id, azimuth, viewpoint));
                        }
                    }
                }
                let Some((_, poi_id, azimuth_deg, viewpoint_m)) = best else {
                    continue;
                };
                let poi_id = poi_id.to_string();
                self.points.get_mut(&poi_id).expect("listed").tries += 1;
                self.tours.insert(
                    drone.clone(),
                    Tour {
                        poi_id: poi_id.clone(),
                        azimuth_deg,
                        viewpoint_m,
                        arrived_s: None,
                    },
                );
                actions.push(Action::Go {
                    drone_id: drone.clone(),
                    poi_id,
                    viewpoint_m,
                    azimuth_deg,
                });
            }
        }
        // What is settled short.
        let toured: BTreeSet<String> = self.tours.values().map(|t| t.poi_id.clone()).collect();
        for (poi_id, point_m) in points {
            let point = self.points.get_mut(poi_id).expect("listed");
            if point.verdict.is_some() {
                continue;
            }
            let verdict = if working.is_empty() {
                Verdict::FleetHomeward
            } else if spent {
                Verdict::TimeSpent
            } else if toured.contains(poi_id) {
                continue;
            } else if point.tries >= config.max_tries {
                Verdict::TriesSpent
            } else if viewpoints(config, point, *point_m, eyes).is_empty() {
                Verdict::NoViewpoint
            } else {
                continue;
            };
            point.verdict = Some(verdict);
            actions.push(Action::Settled {
                poi_id: poi_id.clone(),
                verdict,
                views: point.views.len(),
            });
        }
        // Nobody flies on to a point that is settled.
        let settled: Vec<String> = self
            .tours
            .iter()
            .filter(|(_, tour)| self.points[&tour.poi_id].verdict.is_some())
            .map(|(drone, _)| drone.clone())
            .collect();
        for drone_id in settled {
            self.tours.remove(&drone_id);
            actions.push(Action::Release { drone_id });
        }
        actions
    }

    /// Whether any point is still to be settled.
    pub fn is_open(&self) -> bool {
        self.points.values().any(|point| point.verdict.is_none())
    }
}

/// The places round `point_m` a view would add from, as the map has it.
fn viewpoints(
    config: &EvidenceConfig,
    point: &PointEvidence,
    point_m: [f64; 3],
    eyes: &impl Eyes,
) -> Vec<(f64, [f64; 3])> {
    let step = 360.0 / config.places as f64;
    (0..config.places)
        .map(|k| k as f64 * step)
        .filter(|&azimuth| {
            EvidenceState::adds(config, point, azimuth)
                && point
                    .failed_deg
                    .iter()
                    .all(|&failed| apart_deg(failed, azimuth) > 0.5 * step)
        })
        .filter_map(|azimuth| {
            let (sin, cos) = azimuth.to_radians().sin_cos();
            let xy = [
                point_m[0] + config.standoff_m * cos,
                point_m[1] + config.standoff_m * sin,
            ];
            let z = (point_m[2] + config.height_m).max(eyes.floor_m(xy));
            let up = z - point_m[2];
            let elevation = up.atan2(config.standoff_m).to_degrees();
            let at = [xy[0], xy[1], z];
            (config.standoff_m.hypot(up) <= config.max_range_m
                && (config.elevation_deg[0]..=config.elevation_deg[1]).contains(&elevation)
                && eyes.expects_to_see(at, point_m))
            .then_some((azimuth, at))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sees, and expects to see, from the azimuths listed; all when none are.
    #[derive(Default)]
    struct Where {
        hidden_deg: Vec<(f64, f64)>,
        expects_hidden: bool,
        floor_m: f64,
    }

    impl Where {
        fn open(&self, from: [f64; 3], point: [f64; 3]) -> bool {
            let azimuth = (from[1] - point[1])
                .atan2(from[0] - point[0])
                .to_degrees()
                .rem_euclid(360.0);
            !self
                .hidden_deg
                .iter()
                .any(|&(low, high)| azimuth > low && azimuth < high)
        }
    }

    impl Eyes for Where {
        fn floor_m(&self, _xy_m: [f64; 2]) -> f64 {
            self.floor_m
        }
        fn expects_to_see(&self, from: [f64; 3], point: [f64; 3]) -> bool {
            !self.expects_hidden || self.open(from, point)
        }
        fn shows(&self, _drone: &str, from: [f64; 3], point: [f64; 3]) -> bool {
            self.open(from, point)
        }
    }

    /// Fly every drone straight to what it is asked, a second a run.
    fn fly(
        state: &mut EvidenceState,
        config: &EvidenceConfig,
        eyes: &Where,
        points: &[(String, [f64; 3])],
        drones: usize,
    ) -> (f64, Vec<Action>) {
        let mut at: BTreeMap<String, [f64; 3]> = (0..drones)
            .map(|k| (format!("d{k}"), [200.0 + 50.0 * k as f64, 0.0, 60.0]))
            .collect();
        let mut going: BTreeMap<String, [f64; 3]> = BTreeMap::new();
        let mut all = Vec::new();
        for tick in 0..2000 {
            let t = f64::from(tick);
            let working: Vec<(String, [f64; 3])> =
                at.iter().map(|(id, p)| (id.clone(), *p)).collect();
            let actions = state.run(config, t, points, &working, eyes);
            for action in &actions {
                match action {
                    Action::Go {
                        drone_id,
                        viewpoint_m,
                        ..
                    } => {
                        going.insert(drone_id.clone(), *viewpoint_m);
                    }
                    Action::Release { drone_id } => {
                        going.remove(drone_id);
                    }
                    _ => {}
                }
            }
            all.extend(actions);
            if !state.is_open() {
                return (t, all);
            }
            for (id, to) in &going {
                let p = at.get_mut(id).unwrap();
                let d: [f64; 3] = std::array::from_fn(|k| to[k] - p[k]);
                let length = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                let share = if length > 15.0 { 15.0 / length } else { 1.0 };
                *p = std::array::from_fn(|k| p[k] + d[k] * share);
            }
        }
        panic!("never settled");
    }

    fn points(n: usize) -> Vec<(String, [f64; 3])> {
        (0..n)
            .map(|k| (format!("poi-{k}"), [150.0 * k as f64, 0.0, 10.0]))
            .collect()
    }

    #[test]
    fn an_open_point_is_seen_from_all_round() {
        let config = EvidenceConfig::default();
        let mut state = EvidenceState::default();
        fly(&mut state, &config, &Where::default(), &points(1), 1);
        let point = &state.points["poi-0"];
        assert_eq!(point.verdict, Some(Verdict::Met));
        assert_eq!((point.views.len(), point.tries), (3, 3));
        for (k, a) in point.views.iter().enumerate() {
            for b in &point.views[k + 1..] {
                assert!(apart_deg(a.azimuth_deg, b.azimuth_deg) >= 80.0 - 1.0);
            }
            assert!(a.range_m <= config.max_range_m);
            assert!((15.0..=75.0).contains(&a.elevation_deg), "{a:?}");
        }
        assert!(state.tours.is_empty());
    }

    /// A wall the map does not hold: the view from behind it shows nothing, and
    /// is not counted; the point is seen from elsewhere.
    #[test]
    fn a_view_that_does_not_show_it_is_not_one() {
        let eyes = Where {
            hidden_deg: vec![(-1.0, 40.0), (330.0, 361.0)],
            ..Where::default()
        };
        let mut state = EvidenceState::default();
        let (_, actions) = fly(&mut state, &EvidenceConfig::default(), &eyes, &points(1), 1);
        let point = &state.points["poi-0"];
        assert_eq!(point.verdict, Some(Verdict::Met));
        assert!(!point.failed_deg.is_empty() && point.tries > 3, "{point:?}");
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::Viewed { shown: false, .. })));
        // The map knowing of it saves the try.
        let knowing = Where {
            expects_hidden: true,
            ..eyes
        };
        let mut state = EvidenceState::default();
        fly(
            &mut state,
            &EvidenceConfig::default(),
            &knowing,
            &points(1),
            1,
        );
        assert_eq!(state.points["poi-0"].tries, 3);
    }

    /// Seen from one side only, it is unmet and says why; it is not called met.
    #[test]
    fn a_point_that_cannot_be_seen_enough_is_unmet() {
        let one_side = vec![(30.0, 330.0)];
        for (expects_hidden, why) in [(true, Verdict::NoViewpoint), (false, Verdict::TriesSpent)] {
            let eyes = Where {
                hidden_deg: one_side.clone(),
                expects_hidden,
                floor_m: 0.0,
            };
            let config = EvidenceConfig {
                max_tries: 5,
                ..EvidenceConfig::default()
            };
            let mut state = EvidenceState::default();
            fly(&mut state, &config, &eyes, &points(1), 1);
            let point = &state.points["poi-0"];
            assert_eq!(point.verdict, Some(why), "{point:?}");
            assert!(point.views.len() < config.min_views);
            assert!(point.tries <= config.max_tries);
        }
        // Too high to see it from near enough: nowhere to see it from.
        let eyes = Where {
            floor_m: 200.0,
            ..Where::default()
        };
        let mut state = EvidenceState::default();
        fly(&mut state, &EvidenceConfig::default(), &eyes, &points(1), 1);
        assert_eq!(state.points["poi-0"].verdict, Some(Verdict::NoViewpoint));
    }

    #[test]
    fn it_ends_when_time_or_the_fleet_does() {
        let config = EvidenceConfig {
            max_s: 30.0,
            ..EvidenceConfig::default()
        };
        let mut state = EvidenceState::default();
        let (t, _) = fly(&mut state, &config, &Where::default(), &points(6), 1);
        assert!((t - 30.0).abs() < 1e-9, "{t}");
        let verdicts: Vec<_> = state.points.values().map(|p| p.verdict).collect();
        assert!(verdicts.contains(&Some(Verdict::TimeSpent)));
        assert!(state.tours.is_empty(), "nobody flies on");
        let mut alone = EvidenceState::default();
        alone.run(&config, 0.0, &points(2), &[], &Where::default());
        assert!(!alone.is_open());
        assert_eq!(alone.points["poi-1"].verdict, Some(Verdict::FleetHomeward));
    }

    /// Two drones share the points, never one viewpoint, and are the quicker.
    #[test]
    fn drones_share_the_points() {
        let config = EvidenceConfig {
            max_s: 1000.0,
            ..EvidenceConfig::default()
        };
        let time = |drones: usize| {
            let mut state = EvidenceState::default();
            let (t, _) = fly(&mut state, &config, &Where::default(), &points(4), drones);
            assert!(state
                .points
                .values()
                .all(|p| p.verdict == Some(Verdict::Met)));
            t
        };
        assert!(time(2) < 0.75 * time(1));
    }

    #[test]
    fn a_drone_sent_home_gives_up_its_viewpoint() {
        let config = EvidenceConfig::default();
        let mut state = EvidenceState::default();
        let at = |id: &str| (id.to_string(), [100.0, 0.0, 60.0]);
        state.run(&config, 0.0, &points(1), &[at("a")], &Where::default());
        assert!(state.tours.contains_key("a"));
        let actions = state.run(&config, 1.0, &points(1), &[at("b")], &Where::default());
        assert!(!state.tours.contains_key("a"));
        assert!(matches!(&actions[0], Action::Release { drone_id } if drone_id == "a"));
        assert!(matches!(&actions[1], Action::Go { drone_id, .. } if drone_id == "b"));
    }

    #[test]
    fn a_requirement_nothing_could_meet_is_refused() {
        let good = EvidenceConfig::default();
        assert!(good.validate().is_ok());
        for bad in [
            EvidenceConfig {
                min_views: 5,
                ..good.clone()
            },
            EvidenceConfig {
                max_range_m: 20.0,
                ..good.clone()
            },
            EvidenceConfig {
                elevation_deg: [60.0, 75.0],
                ..good.clone()
            },
            EvidenceConfig {
                hold_s: f64::NAN,
                ..good.clone()
            },
            EvidenceConfig { places: 0, ..good },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
