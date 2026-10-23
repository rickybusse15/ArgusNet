//! The primary belief/truth switch (console parity V3, VIEWER_UI 2.2): the
//! canvas shows either the world as the platforms know it, or the
//! simulation's truth for evaluation. One control, ahead of the layer
//! chips, sets every truth-bearing layer together, so a view is never half
//! one and half the other.
//!
//! The view is read from the overlays, not held beside them: the true world
//! is on exactly when truth markers are, which is also what the header reads
//! as `EVALUATION / TRUTH ENABLED`. Secondary chips stay free to change
//! afterwards; the switch sets them as a group.

use std::collections::HashMap;

use crate::state::RuntimeOverlayVisibility;

/// Which world the canvas shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldView {
    /// What the platforms know: their own map in a run that builds one, the
    /// prior world otherwise, and fused tracks, never target truth.
    DronesMap,
    /// The simulation's truth, for evaluation: target truth, and in a run
    /// that builds its own map, the true terrain, structures, lines and trees.
    TrueWorld,
}

impl WorldView {
    pub const ALL: [WorldView; 2] = [WorldView::DronesMap, WorldView::TrueWorld];

    /// The view the overlays show now.
    pub fn of(overlays: &RuntimeOverlayVisibility) -> Self {
        if overlays.truths {
            WorldView::TrueWorld
        } else {
            WorldView::DronesMap
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WorldView::DronesMap => "Drones' map",
            WorldView::TrueWorld => "True world",
        }
    }

    pub fn explanation(self) -> &'static str {
        match self {
            WorldView::DronesMap => "What the platforms know: their map and fused tracks.",
            WorldView::TrueWorld => {
                "Simulation truth, for evaluation. Never available in live operation."
            }
        }
    }
}

/// Why truth may not be shown, if it may not: never in live operation
/// (VIEWER_UI 3), where there is no truth to show.
pub fn truth_unavailable(live: bool) -> Option<&'static str> {
    live.then_some("live operation has no ground truth")
}

/// Set every truth-bearing layer for `view`. `self_built` is a run that maps
/// the world for itself: there the scene's layers, the run's power lines and
/// trees are truth, and the drones' map is its own layer (`belief_layer`).
/// In a run with a prior map those are what the platforms were given, so
/// they stay as they are, and only target truth changes.
pub fn apply(
    view: WorldView,
    self_built: bool,
    scene_layers: &[String],
    belief_layer: &str,
    overlays: &mut RuntimeOverlayVisibility,
    base_layers: &mut HashMap<String, bool>,
) {
    let truth = view == WorldView::TrueWorld;
    overlays.truths = truth;
    if !self_built {
        return;
    }
    overlays.power_lines = truth;
    overlays.trees = truth;
    if !truth {
        overlays.power_line_clearance = false;
    }
    for layer in scene_layers {
        base_layers.insert(layer.clone(), truth);
    }
    base_layers.insert(belief_layer.to_string(), !truth);
}

#[cfg(test)]
mod tests {
    use super::*;

    const BELIEF: &str = "drones-map";

    fn layers() -> Vec<String> {
        vec!["terrain".into(), "buildings".into()]
    }

    /// In a self-built run the switch moves every truth-bearing layer
    /// together, and the drones' map with it the other way.
    #[test]
    fn a_self_built_run_switches_every_truth_layer_together() {
        let mut overlays = RuntimeOverlayVisibility::default();
        let mut base = HashMap::new();
        apply(
            WorldView::TrueWorld,
            true,
            &layers(),
            BELIEF,
            &mut overlays,
            &mut base,
        );
        assert_eq!(WorldView::of(&overlays), WorldView::TrueWorld);
        assert!(overlays.power_lines && overlays.trees);
        assert!(base["terrain"]);
        assert!(base["buildings"]);
        assert!(!base[BELIEF]);

        overlays.power_line_clearance = true;
        apply(
            WorldView::DronesMap,
            true,
            &layers(),
            BELIEF,
            &mut overlays,
            &mut base,
        );
        assert_eq!(WorldView::of(&overlays), WorldView::DronesMap);
        assert!(!overlays.truths);
        assert!(!overlays.power_lines && !overlays.trees && !overlays.power_line_clearance);
        assert!(!base["terrain"]);
        assert!(!base["buildings"]);
        assert!(base[BELIEF]);
    }

    /// With a prior map the scene, lines and trees are what the platforms
    /// were given: the switch moves target truth only.
    #[test]
    fn a_prior_map_run_switches_target_truth_only() {
        let mut overlays = RuntimeOverlayVisibility {
            power_lines: true,
            trees: false,
            ..RuntimeOverlayVisibility::default()
        };
        let mut base = HashMap::from([("terrain".to_string(), true)]);
        apply(
            WorldView::TrueWorld,
            false,
            &layers(),
            BELIEF,
            &mut overlays,
            &mut base,
        );
        assert!(overlays.truths);
        assert!(overlays.power_lines && !overlays.trees, "left as they were");
        assert_eq!(base, HashMap::from([("terrain".to_string(), true)]));
        apply(
            WorldView::DronesMap,
            false,
            &layers(),
            BELIEF,
            &mut overlays,
            &mut base,
        );
        assert!(!overlays.truths);
        assert!(overlays.power_lines);
    }

    /// A replay starts on the drones' map, and live operation has no truth.
    #[test]
    fn replays_start_on_the_drones_map_and_live_has_no_truth() {
        assert_eq!(
            WorldView::of(&RuntimeOverlayVisibility::default()),
            WorldView::DronesMap
        );
        assert!(truth_unavailable(true).is_some());
        assert!(truth_unavailable(false).is_none());
    }
}
