//! View planning (SR5d; SENSING_REALISM_IMPLEMENTATION_GUIDE §8.3): where an
//! aircraft cruises next and where its pod looks, chosen from what the map
//! still lacks and what getting there costs.
//!
//! Two choices, both pure functions over plain views so they test without a
//! simulation:
//!
//! - **Where to go** ([`choose_goal`]): the block whose improvable deficit
//!   is worth the most per watt-hour, not the nearest one. A block's worth is
//!   its cells' deficit weights ([`DeficitWeights`]); its cost is the cruise
//!   energy to reach it plus the energy of the turn onto it. A lift-and-cruise
//!   aircraft pays for turns, and it never hovers to look, so every view is
//!   taken on the way through. The leg's energy comes from the caller, who
//!   knows the aircraft's energy model and the ground it will climb over;
//!   [`ViewPlanningConfig::cruise_wh_per_km`] is the flat fallback.
//! - **Where to look** ([`choose_roll`]): the pod's roll across the track,
//!   from a fixed set of candidates, by the deficit cells its predicted
//!   footprint would improve. Only rays the quality requirement would accept
//!   count, so the pod never looks somewhere it cannot improve. A change of
//!   roll must win by a margin, because slewing costs settled captures.
//!
//! The energy model is declared, not calibrated: watt-hours per kilometre of
//! cruise and per radian of turn, both tagged assumed in the run's metadata.
//! Earlier endurance estimates are not an aircraft model (guide §7).

use serde::{Deserialize, Serialize};

/// How much one cell of each improvable deficit is worth fixing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeficitWeights {
    pub unseen: f64,
    pub too_oblique: f64,
    pub sparse: f64,
    pub stale: f64,
}

impl Default for DeficitWeights {
    fn default() -> Self {
        Self {
            unseen: 1.0,
            too_oblique: 0.8,
            sparse: 0.5,
            stale: 0.7,
        }
    }
}

/// The view planner's declared parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewPlanningConfig {
    #[serde(default)]
    pub weights: DeficitWeights,
    /// Cruise energy per kilometre over flat ground, watt-hours: the leg cost
    /// when the caller has no energy model of its own.
    pub cruise_wh_per_km: f64,
    /// Energy per radian of heading change, watt-hours.
    pub turn_wh_per_rad: f64,
    /// A goal nearer than this is not chosen: the view is taken on the way
    /// through, and a nearer goal sits under the aircraft.
    pub min_goal_distance_m: f64,
    /// A goal within this of another aircraft's goal is someone else's.
    pub claim_radius_m: f64,
    /// How much of its worth a claimed block keeps, in `[0, 1]`: 0 never
    /// shares a goal, 1 ignores claims. A crowded fleet that never shares
    /// sends aircraft far for little.
    #[serde(default = "default_claim_factor")]
    pub claimed_worth: f64,
    /// How much of what a leg's swath sweeps on the way counts toward its
    /// worth, beside its goal block's deficit. Off by default: measured on
    /// the regional scenario it did no better than the goal block alone
    /// (docs/migration/VIEW_PLANNING.md).
    #[serde(default)]
    pub en_route_weight: f64,
    /// The largest cross-track roll the pod is pointed at, degrees.
    pub max_roll_deg: f64,
    /// Candidate rolls either side of nadir: `max_roll_deg` split into this
    /// many steps.
    pub roll_steps: usize,
    /// How much more a new roll must be worth than the held one to change,
    /// on top of paying for the captures the slew loses.
    pub roll_hysteresis: f64,
    /// How many captures a chosen roll is expected to be held for: the
    /// horizon over which a better view repays the captures a slew loses.
    #[serde(default = "default_roll_horizon")]
    pub roll_horizon_captures: f64,
    /// How long a chosen roll is held before it is planned again, seconds,
    /// unless the aircraft turns onto a new leg first: a roll is chosen per
    /// leg, over the ground the leg will sweep, not per capture.
    #[serde(default = "default_roll_replan")]
    pub roll_replan_s: f64,
    /// A heading change of more than this starts a new leg, degrees.
    #[serde(default = "default_leg_turn")]
    pub leg_turn_deg: f64,
    /// Whether goals are chosen by worth per watt-hour (else the nearest,
    /// as without planning): for measuring each half apart.
    #[serde(default = "yes")]
    pub plan_goals: bool,
    /// Whether the pod's roll is planned (else nadir). Off by default:
    /// measured on the regional scenario it did not repay its slews and
    /// settles (docs/migration/VIEW_PLANNING.md).
    #[serde(default)]
    pub plan_pointing: bool,
}

fn yes() -> bool {
    true
}

fn default_claim_factor() -> f64 {
    0.25
}

fn default_roll_horizon() -> f64 {
    10.0
}

fn default_roll_replan() -> f64 {
    8.0
}

fn default_leg_turn() -> f64 {
    15.0
}

impl Default for ViewPlanningConfig {
    fn default() -> Self {
        Self {
            weights: DeficitWeights::default(),
            // Assumed: a ~25 kg lift-and-cruise aircraft at cruise.
            cruise_wh_per_km: 18.0,
            turn_wh_per_rad: 0.5,
            min_goal_distance_m: 250.0,
            claim_radius_m: 400.0,
            claimed_worth: default_claim_factor(),
            en_route_weight: 0.0,
            max_roll_deg: 30.0,
            roll_steps: 3,
            roll_hysteresis: 0.15,
            roll_horizon_captures: default_roll_horizon(),
            roll_replan_s: default_roll_replan(),
            leg_turn_deg: default_leg_turn(),
            plan_goals: true,
            plan_pointing: false,
        }
    }
}

impl ViewPlanningConfig {
    pub fn validate(&self) -> Result<(), String> {
        let w = &self.weights;
        if [w.unseen, w.too_oblique, w.sparse, w.stale]
            .iter()
            .any(|v| !(v.is_finite() && *v >= 0.0))
        {
            return Err("view planning deficit weights must not be negative".into());
        }
        for (name, value) in [
            ("cruise_wh_per_km", self.cruise_wh_per_km),
            ("min_goal_distance_m", self.min_goal_distance_m),
            ("claim_radius_m", self.claim_radius_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "view planning {name} must be positive, got {value}"
                ));
            }
        }
        if !(self.turn_wh_per_rad.is_finite() && self.turn_wh_per_rad >= 0.0) {
            return Err("view planning turn_wh_per_rad must not be negative".into());
        }
        if !(self.max_roll_deg.is_finite() && (0.0..=80.0).contains(&self.max_roll_deg)) {
            return Err("view planning max_roll_deg must be within [0, 80]".into());
        }
        if !(self.claimed_worth.is_finite() && (0.0..=1.0).contains(&self.claimed_worth)) {
            return Err("view planning claimed_worth must be within [0, 1]".into());
        }
        if !(self.en_route_weight.is_finite() && self.en_route_weight >= 0.0) {
            return Err("view planning en_route_weight must not be negative".into());
        }
        if self.roll_steps == 0 {
            return Err("view planning needs at least one roll step".into());
        }
        if !(self.roll_hysteresis.is_finite() && self.roll_hysteresis >= 0.0) {
            return Err("view planning roll_hysteresis must not be negative".into());
        }
        if !(self.roll_horizon_captures.is_finite() && self.roll_horizon_captures > 0.0) {
            return Err("view planning roll_horizon_captures must be positive".into());
        }
        if !(self.roll_replan_s.is_finite() && self.roll_replan_s > 0.0) {
            return Err("view planning roll_replan_s must be positive".into());
        }
        if !(self.leg_turn_deg.is_finite() && self.leg_turn_deg > 0.0) {
            return Err("view planning leg_turn_deg must be positive".into());
        }
        Ok(())
    }

    /// The candidate rolls, radians, nadir first, then outward in pairs.
    pub fn roll_candidates(&self) -> Vec<f64> {
        let step = self.max_roll_deg.to_radians() / self.roll_steps as f64;
        let mut rolls = vec![0.0];
        for k in 1..=self.roll_steps {
            let roll = step * k as f64;
            rolls.push(roll);
            rolls.push(-roll);
        }
        rolls
    }

    /// The flat fallback leg cost from `from_m`: cruise energy for the
    /// straight-line distance.
    pub fn flat_leg_cost(&self, from_m: [f64; 2]) -> impl Fn([f64; 2]) -> f64 + '_ {
        move |to: [f64; 2]| {
            self.cruise_wh_per_km * (to[0] - from_m[0]).hypot(to[1] - from_m[1]) / 1000.0
        }
    }
}

/// One block of the map that still needs work: its centre and the weighted
/// deficit flying over it would fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkBlock {
    pub centre_m: [f64; 2],
    pub gain: f64,
}

/// The chosen goal, and why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalChoice {
    pub centre_m: [f64; 2],
    pub gain: f64,
    pub cost_wh: f64,
}

/// The smallest angle from `a` to `b`, radians, in `[0, π]`.
fn turn_between(a: f64, b: f64) -> f64 {
    let d = (b - a).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

/// The leg worth the most per watt-hour from `from_m`, heading
/// `heading_rad` (`None` when the aircraft has no heading yet: no turn is
/// charged). A leg is worth its goal block's deficit plus `en_route_gain`,
/// the deficit the swath sweeps on the way there: a cruise-only aircraft
/// maps every metre it flies, so a long leg over unmapped ground is worth
/// more than its end. `leg_cost_wh` is the energy to cruise there. A block
/// within the claim radius of another aircraft's goal keeps only
/// [`ViewPlanningConfig::claimed_worth`] of its worth. Ties go to the
/// lowest index, so the choice is deterministic. `None` when no leg has any
/// worth.
pub fn choose_goal(
    blocks: &[WorkBlock],
    from_m: [f64; 2],
    heading_rad: Option<f64>,
    claims: &[[f64; 2]],
    config: &ViewPlanningConfig,
    en_route_gain: impl Fn([f64; 2]) -> f64,
    leg_cost_wh: impl Fn([f64; 2]) -> f64,
) -> Option<GoalChoice> {
    let mut best: Option<(f64, GoalChoice)> = None;
    for block in blocks {
        let (dx, dy) = (block.centre_m[0] - from_m[0], block.centre_m[1] - from_m[1]);
        let distance = dx.hypot(dy);
        if distance < config.min_goal_distance_m || block.gain <= 0.0 {
            continue;
        }
        let claimed = claims.iter().any(|c| {
            (c[0] - block.centre_m[0]).hypot(c[1] - block.centre_m[1]) < config.claim_radius_m
        });
        let gain = block.gain + config.en_route_weight * en_route_gain(block.centre_m);
        let worth = if claimed {
            gain * config.claimed_worth
        } else {
            gain
        };
        if worth <= 0.0 {
            continue;
        }
        let turn = heading_rad.map_or(0.0, |h| turn_between(h, dy.atan2(dx)));
        let cost_wh = leg_cost_wh(block.centre_m) + config.turn_wh_per_rad * turn;
        let score = worth / cost_wh.max(1e-9);
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((
                score,
                GoalChoice {
                    centre_m: block.centre_m,
                    gain,
                    cost_wh,
                },
            ));
        }
    }
    best.map(|(_, choice)| choice)
}

/// The roll to hold. Each candidate is worth its footprint's `gain_of` per
/// capture over [`ViewPlanningConfig::roll_horizon_captures`], less the
/// captures switching to it loses (`lost_captures_of`: slewing and
/// settling, at the held view's gain). The held roll `current_rad` is kept
/// unless the best beats it by the hysteresis as well. Ties go to the
/// candidate nearest nadir (the first in [`roll_candidates`] order), so a
/// map with nothing to gain points straight down.
///
/// [`roll_candidates`]: ViewPlanningConfig::roll_candidates
pub fn choose_roll(
    current_rad: f64,
    config: &ViewPlanningConfig,
    gain_of: impl Fn(f64) -> f64,
    lost_captures_of: impl Fn(f64) -> f64,
) -> f64 {
    let horizon = config.roll_horizon_captures;
    let held_gain = gain_of(current_rad);
    let mut best = (f64::NEG_INFINITY, 0.0, 0.0);
    for roll in config.roll_candidates() {
        let gain = gain_of(roll);
        let lost = if (roll - current_rad).abs() < 1e-9 {
            0.0
        } else {
            lost_captures_of(roll)
        };
        let value = gain * horizon - held_gain * lost;
        if value > best.0 {
            best = (value, roll, gain);
        }
    }
    if best.2 <= 0.0 && held_gain <= 0.0 {
        return 0.0;
    }
    if (best.1 - current_rad).abs() < 1e-9
        || best.0 <= held_gain * horizon * (1.0 + config.roll_hysteresis)
    {
        return current_rad;
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(x: f64, y: f64, gain: f64) -> WorkBlock {
        WorkBlock {
            centre_m: [x, y],
            gain,
        }
    }

    #[test]
    fn the_goal_is_the_best_gain_per_watt_hour_not_the_nearest() {
        let config = ViewPlanningConfig::default();
        // A small deficit at 300 m against a large one at 600 m.
        let blocks = [block(300.0, 0.0, 5.0), block(600.0, 0.0, 25.0)];
        let choice = choose_goal(
            &blocks,
            [0.0, 0.0],
            None,
            &[],
            &config,
            |_| 0.0,
            config.flat_leg_cost([0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(choice.centre_m, [600.0, 0.0]);
        // With equal deficits the nearer is cheaper.
        let blocks = [block(300.0, 0.0, 10.0), block(600.0, 0.0, 10.0)];
        let choice = choose_goal(
            &blocks,
            [0.0, 0.0],
            None,
            &[],
            &config,
            |_| 0.0,
            config.flat_leg_cost([0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(choice.centre_m, [300.0, 0.0]);
        assert!((choice.cost_wh - 18.0 * 0.3).abs() < 1e-9);
    }

    #[test]
    fn turning_costs_so_ahead_beats_behind_at_equal_distance() {
        let config = ViewPlanningConfig {
            turn_wh_per_rad: 5.0,
            ..ViewPlanningConfig::default()
        };
        let blocks = [block(-400.0, 0.0, 10.0), block(400.0, 0.0, 10.0)];
        // Flying east: the block ahead wins though it is listed second.
        let choice = choose_goal(
            &blocks,
            [0.0, 0.0],
            Some(0.0),
            &[],
            &config,
            |_| 0.0,
            config.flat_leg_cost([0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(choice.centre_m, [400.0, 0.0]);
        // Flying west: the other.
        let choice = choose_goal(
            &blocks,
            [0.0, 0.0],
            Some(std::f64::consts::PI),
            &[],
            &config,
            |_| 0.0,
            config.flat_leg_cost([0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(choice.centre_m, [-400.0, 0.0]);
    }

    /// The leg cost is the caller's: a block behind a ridge costs its climb.
    #[test]
    fn a_dear_leg_loses_to_a_cheap_one() {
        let config = ViewPlanningConfig::default();
        let blocks = [block(400.0, 0.0, 10.0), block(0.0, 400.0, 10.0)];
        let over_a_ridge = |to: [f64; 2]| if to[0] > 0.0 { 50.0 } else { 5.0 };
        let choice = choose_goal(
            &blocks,
            [0.0, 0.0],
            None,
            &[],
            &config,
            |_| 0.0,
            over_a_ridge,
        )
        .unwrap();
        assert_eq!(choice.centre_m, [0.0, 400.0]);
        assert_eq!(choice.cost_wh, 5.0);
    }

    /// A block too near, or worth nothing, is never a goal. A claimed one
    /// keeps only part of its worth, so it loses to an unclaimed one and
    /// is chosen only when nothing else is left; with no worth kept, never.
    #[test]
    fn near_worthless_and_claimed_blocks() {
        let config = ViewPlanningConfig::default();
        let near_and_worthless = [block(100.0, 0.0, 50.0), block(900.0, 0.0, 0.0)];
        let flat = config.flat_leg_cost([0.0, 0.0]);
        let none = |_: [f64; 2]| 0.0;
        assert_eq!(
            choose_goal(
                &near_and_worthless,
                [0.0, 0.0],
                None,
                &[],
                &config,
                none,
                &flat
            ),
            None
        );
        let claims = [[520.0, 0.0]];
        let blocks = [block(500.0, 0.0, 50.0), block(0.0, 900.0, 50.0)];
        let choice = choose_goal(&blocks, [0.0, 0.0], None, &claims, &config, none, &flat).unwrap();
        assert_eq!(
            choice.centre_m,
            [0.0, 900.0],
            "the unclaimed one, though further"
        );
        let choice = choose_goal(
            &blocks[..1],
            [0.0, 0.0],
            None,
            &claims,
            &config,
            none,
            &flat,
        )
        .unwrap();
        assert_eq!(
            choice.centre_m,
            [500.0, 0.0],
            "a claimed goal beats no goal"
        );
        let exclusive = ViewPlanningConfig {
            claimed_worth: 0.0,
            ..ViewPlanningConfig::default()
        };
        assert_eq!(
            choose_goal(
                &blocks[..1],
                [0.0, 0.0],
                None,
                &claims,
                &exclusive,
                none,
                &flat
            ),
            None
        );
    }

    /// A leg is worth what it sweeps on the way: a further goal across
    /// unmapped ground beats a nearer one over mapped ground.
    #[test]
    fn what_a_leg_sweeps_counts() {
        let config = ViewPlanningConfig {
            en_route_weight: 1.0,
            ..ViewPlanningConfig::default()
        };
        let blocks = [block(300.0, 0.0, 10.0), block(0.0, 600.0, 10.0)];
        let flat = config.flat_leg_cost([0.0, 0.0]);
        let sweep = |to: [f64; 2]| if to[1] > 0.0 { 200.0 } else { 0.0 };
        let choice = choose_goal(&blocks, [0.0, 0.0], None, &[], &config, sweep, &flat).unwrap();
        assert_eq!(choice.centre_m, [0.0, 600.0]);
        assert_eq!(choice.gain, 210.0);
    }

    #[test]
    fn the_roll_follows_the_deficit_with_hysteresis() {
        let config = ViewPlanningConfig::default();
        let rolls = config.roll_candidates();
        assert_eq!(rolls.len(), 7);
        assert_eq!(rolls[0], 0.0);
        // The deficit is to the left (positive roll).
        let left = |roll: f64| if roll > 0.0 { 10.0 + roll } else { 1.0 };
        let max = 30f64.to_radians();
        let free = |_: f64| 0.0;
        assert!((choose_roll(0.0, &config, left, free) - max).abs() < 1e-9);
        // Nothing to gain anywhere: nadir.
        assert_eq!(choose_roll(0.2, &config, |_| 0.0, free), 0.0);
        // A held roll keeps unless beaten by more than the hysteresis.
        let step = max / 3.0;
        let close = |roll: f64| {
            if (roll - step).abs() < 1e-9 {
                10.0
            } else {
                11.0
            }
        };
        assert_eq!(choose_roll(step, &config, close, free), step);
        let far = |roll: f64| {
            if (roll - step).abs() < 1e-9 {
                10.0
            } else {
                12.0
            }
        };
        assert_ne!(choose_roll(step, &config, far, free), step);
        // The same gain, but each switch loses three captures of ten: kept.
        let dear = |_: f64| 3.0;
        assert_eq!(choose_roll(step, &config, far, dear), step);
    }

    #[test]
    fn invalid_configs_are_refused() {
        assert!(ViewPlanningConfig::default().validate().is_ok());
        let bad = [
            ViewPlanningConfig {
                cruise_wh_per_km: 0.0,
                ..ViewPlanningConfig::default()
            },
            ViewPlanningConfig {
                roll_steps: 0,
                ..ViewPlanningConfig::default()
            },
            ViewPlanningConfig {
                max_roll_deg: 90.0,
                ..ViewPlanningConfig::default()
            },
        ];
        for config in bad {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }
}
