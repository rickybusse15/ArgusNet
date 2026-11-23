//! The air over a self-built map, for [`argusnet_planning::route3d`]: what a
//! mission flight is kept above and off ([`crate::wire_governor`]), as columns
//! a route is searched through. The governor still has the last word on every
//! flight; a planned route gives it less to do.

use argusnet_planning::battery::BatteryModel;
use argusnet_planning::route3d::{plan, Airspace, Column, Plan3d, Route3dConfig};

use crate::belief_world::{BeliefHandle, BeliefSnapshot, MissionFlightConfig, WirePlanningConfig};

/// Below anything a wire could hang at: asked from there, its bounds are the
/// whole band it closes.
const FAR_BELOW_M: f64 = -1.0e6;

/// Routes over the map as it is when asked.
#[derive(Debug, Clone)]
pub struct RouteAir {
    pub config: Route3dConfig,
    pub map: BeliefHandle,
    pub surface: MissionFlightConfig,
    pub wires: Option<WirePlanningConfig>,
}

impl RouteAir {
    pub fn plan(&self, battery: &BatteryModel, start: [f64; 3], goal: [f64; 3]) -> Plan3d {
        let map = self.map.current();
        let air = Columns {
            map: &map,
            surface: &self.surface,
            wires: self.wires.as_ref(),
            wire_margin_m: self.config.wire_margin_m,
        };
        plan(&air, battery, start, goal, &self.config)
    }
}

struct Columns<'a> {
    map: &'a BeliefSnapshot,
    surface: &'a MissionFlightConfig,
    wires: Option<&'a WirePlanningConfig>,
    wire_margin_m: f64,
}

impl Airspace for Columns<'_> {
    fn column(&self, xy_m: [f64; 2], half_m: f64) -> Option<Column> {
        let [x, y] = xy_m;
        if !self.map.contains(xy_m) {
            return None;
        }
        let (from, to) = ([x - half_m, y], [x + half_m, y]);
        let top =
            self.map
                .surface_top_along(from, to, half_m + self.surface.corridor_m, true, None)?;
        let mut floor_m = top + self.surface.clearance_m;
        let mut closed_m = None;
        if let Some(wires) = self.wires {
            let bounds = self.map.hazard_bounds_within(
                from,
                [1.0, 0.0],
                2.0 * half_m,
                FAR_BELOW_M,
                wires.clearance_m,
                [half_m + self.wire_margin_m, self.wire_margin_m],
            );
            // A pole goes to the ground: nothing passes under it.
            if let Some(floor) = bounds.floor_m {
                floor_m = floor_m.max(floor);
            }
            if let Some(low) = bounds.ceiling_m {
                let high = bounds.over_m.unwrap_or(f64::INFINITY).max(low);
                closed_m = Some([low, high]);
                // Under a wire no lower than the governor goes.
                floor_m = floor_m.max(self.map.ground_at(x, y) + wires.min_agl_m);
            }
        }
        Some(Column { floor_m, closed_m })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_world::{BeliefGrid, ExplorationConfig};
    use crate::range_scan::{RangeReturn, RangeScan, ReturnClass};
    use argusnet_world::powerline::{PowerLine, Support};

    /// Flat ground, a 60 m structure at the origin and a wire along y at
    /// x = 200, 46 m up.
    fn air(min_saving: f64) -> RouteAir {
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
                hit(
                    -495.0 + 10.0 * i,
                    -495.0 + 10.0 * j,
                    0.0,
                    ReturnClass::Ground,
                )
            })
            .collect();
        returns.extend((0..12).map(|_| hit(5.0, 5.0, 60.0, ReturnClass::Structure)));
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
        let support = |y: f64| Support {
            xy_m: [200.0, y],
            base_z_m: 45.0,
            height_m: 1.0,
        };
        grid.know_wires(vec![PowerLine {
            id: "wire-track-0".into(),
            supports: vec![support(-400.0), support(400.0)],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.0,
            conductor_radius_m: 0.015,
            clearance_m: 10.0,
        }
        .validated()
        .unwrap()]);
        RouteAir {
            config: Route3dConfig {
                min_saving,
                ..Route3dConfig::default()
            },
            map: BeliefHandle::new(grid.snapshot(&ExplorationConfig::default())),
            surface: MissionFlightConfig::default(),
            wires: Some(WirePlanningConfig::default()),
        }
    }

    #[test]
    fn a_structure_is_gone_round_and_a_wire_kept_off() {
        let air = air(-10.0);
        let battery = BatteryModel::default();
        let Plan3d::Route(round) = air.plan(&battery, [-200.0, 5.0, 40.0], [120.0, 5.0, 40.0])
        else {
            panic!("a route")
        };
        assert!(round.corners_m.iter().all(|c| c[2] < 55.0), "{round:?}");
        assert!(
            round.corners_m.iter().any(|c| (c[1] - 5.0).abs() > 10.0),
            "{round:?}"
        );
        // At the wire's height: under it or over it, never through.
        let Plan3d::Route(across) = air.plan(&battery, [100.0, 200.0, 46.0], [300.0, 200.0, 46.0])
        else {
            panic!("a route")
        };
        let mut points = vec![[100.0, 200.0, 46.0]];
        points.extend(&across.corners_m);
        points.push([300.0, 200.0, 46.0]);
        for leg in points.windows(2) {
            for k in 0..=100 {
                let f = f64::from(k) / 100.0;
                let at: [f64; 3] = std::array::from_fn(|a| leg[0][a] + f * (leg[1][a] - leg[0][a]));
                let apart = (at[0] - 200.0).hypot(at[2] - 46.0);
                assert!(apart >= 20.0, "{at:?} is {apart} m from the wire");
            }
        }
        // Clear of both, the straight line is left alone.
        let plan = air.plan(&battery, [-300.0, 300.0, 60.0], [-300.0, -300.0, 60.0]);
        assert_eq!(plan, Plan3d::Straight);
        assert_eq!(
            air.plan(&battery, [0.0, 300.0, 60.0], [900.0, 300.0, 20.0]),
            Plan3d::None,
            "off the map there is nothing to plan over"
        );
    }
}
