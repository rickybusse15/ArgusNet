//! Power lines for a built world (R3, `powerline-v1`): opt in with the
//! `power_lines` option. Laid out from the map alone, with no random draw,
//! so a world's line is the same for every seed, and a world without the
//! option is exactly what it was before lines existed.

use argusnet_world::obstacles::Obstacle;
use argusnet_world::powerline::{PowerLine, Support};

use super::environment::MapBounds;
use argusnet_world::landscape::BuiltTerrain;

/// Supports this far apart along the line, metres (scaled with the map).
const SPAN_M: f64 = 200.0;
/// Each support's height above the ground, metres.
const SUPPORT_HEIGHT_M: f64 = 30.0;
/// Three conductors on a cross-arm: one each side, one on the line.
const CONDUCTOR_OFFSETS_M: [f64; 3] = [-4.0, 0.0, 4.0];
/// A transmission line's sag at mid-span, as a fraction of the span.
const SAG_RATIO: f64 = 0.03;
const CONDUCTOR_RADIUS_M: f64 = 0.015;
/// How far a flight must keep from a conductor of a transmission line.
const CLEARANCE_M: f64 = 10.0;

/// One transmission line crossing the map from its south-west quarter to its
/// north-east quarter. A support that would stand inside a building or wall
/// is left out (its neighbours' span covers it); the end supports always
/// stand.
pub fn build_power_lines(
    terrain: &BuiltTerrain,
    bounds: MapBounds,
    scale: f64,
    obstacles: &[Obstacle],
) -> Vec<PowerLine> {
    let width = bounds.x_max_m - bounds.x_min_m;
    let depth = bounds.y_max_m - bounds.y_min_m;
    let from = [bounds.x_min_m + 0.1 * width, bounds.y_min_m + 0.25 * depth];
    let to = [bounds.x_max_m - 0.1 * width, bounds.y_min_m + 0.75 * depth];
    let length = (to[0] - from[0]).hypot(to[1] - from[1]);
    let spans = (length / (SPAN_M * scale)).ceil().max(1.0) as usize;
    let stands_in_structure = |xy: [f64; 2]| {
        obstacles.iter().any(|obstacle| {
            obstacle.blocker_type.is_hard()
                && obstacle.point_inside(xy[0], xy[1], 0.5 * (obstacle.base_z_m + obstacle.top_z_m))
        })
    };
    let supports: Vec<Support> = (0..=spans)
        .map(|k| {
            let t = k as f64 / spans as f64;
            [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            ]
        })
        .enumerate()
        .filter(|(k, xy)| *k == 0 || *k == spans || !stands_in_structure(*xy))
        .map(|(_, xy)| Support {
            xy_m: xy,
            base_z_m: terrain.height_at(xy[0], xy[1]),
            height_m: SUPPORT_HEIGHT_M,
        })
        .collect();
    let line = PowerLine {
        id: "powerline-transmission".to_string(),
        supports,
        conductor_offsets_m: CONDUCTOR_OFFSETS_M.to_vec(),
        sag_ratio: SAG_RATIO,
        conductor_radius_m: CONDUCTOR_RADIUS_M,
        clearance_m: CLEARANCE_M,
    };
    vec![line
        .validated()
        .expect("a line laid across the map is a valid line")]
}

#[cfg(test)]
mod tests {
    use crate::builder::{build_bundle, ScenarioConfig};
    use crate::scenario::{build_world, ScenarioBundle};
    use serde_json::{json, Value};

    fn bundle(power_lines: bool) -> Value {
        let config: ScenarioConfig = serde_json::from_value(json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "duration_s": 5.0,
            "options": {"power_lines": power_lines},
        }))
        .expect("config");
        build_bundle(&config).expect("bundle builds")
    }

    /// The option strings one line across the map, seated on the ground,
    /// carried by the bundle into the run and into the replay's scene. A
    /// world without it is the world it was: no key anywhere, and the same
    /// `Debug` rendering, so the same checkpoint binding.
    #[test]
    fn a_world_has_a_power_line_only_when_asked() {
        let without = bundle(false);
        let with = bundle(true);
        assert!(without["world"].get("power_lines").is_none());
        assert!(without["presentation"].get("power_lines").is_none());
        assert!(without["presentation"]["scenario_options"]
            .get("power_lines")
            .is_none());

        // Everything else in the world is the same.
        let mut stripped = with["world"].clone();
        stripped.as_object_mut().unwrap().remove("power_lines");
        assert_eq!(stripped, without["world"]);
        assert_eq!(
            with["presentation"]["power_lines"],
            with["world"]["power_lines"]
        );
        assert_eq!(
            with["presentation"]["scenario_options"]["power_lines"],
            json!(true)
        );

        let parsed = ScenarioBundle::from_json(&with.to_string()).expect("parses");
        let world = build_world(&parsed.world).expect("builds");
        assert_eq!(world.power_lines.len(), 1);
        let line = &world.power_lines[0];
        assert!(line.supports.len() >= 2);
        for support in &line.supports {
            let ground = crate::observe::TerrainHeight::height_at(
                &world.terrain,
                support.xy_m[0],
                support.xy_m[1],
            );
            assert!(
                (support.base_z_m - ground).abs() < 1e-6,
                "seated on the ground"
            );
        }

        let plain = ScenarioBundle::from_json(&without.to_string()).expect("parses");
        assert!(!format!("{:?}", plain.world).contains("power_lines"));
        assert!(format!("{:?}", parsed.world).contains("power_lines"));
    }
}
