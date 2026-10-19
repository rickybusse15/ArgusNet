//! Layer presets (C6, VIEWER_UI §6.2): what each preset shows, as one table.
//! Presets name meaning, not renderer toggles; the individual toggles stay
//! available in the tool panel for advanced use.
//!
//! | Preset | Shows |
//! | --- | --- |
//! | Operational | terrain, platforms, fused tracks, active task (POIs), route (trails, egress), zones |
//! | Reconstruction | belief overlay (observed mask, confidence, staleness), coverage, localization ellipses, reconstruction, scan grid, inspection events (provenance) |
//! | Safety | zones, platforms, tracks, trajectories (trails), deconfliction (launch lines), sensor clearance (rings), fallbacks (egress) |
//!
//! The scan-map reconstruction stays on in every preset: it draws only in the
//! Scan and Split views, which the operator chooses explicitly.
//!
//! Every preset keeps the zones (safety geometry) and the platforms
//! (selection identity), and none shows truth: truth is a capability, not a
//! layer. The startup layers are the Operational preset's.

use crate::state::{MapLayerPreset, MissionOverlaySettings, RuntimeOverlayVisibility};

/// The layers a preset sets. Toggles a preset does not own (diagnostic
/// glyphs, the land-cover tint, power lines) are left as they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetLayers {
    pub nodes: bool,
    pub tracks: bool,
    pub zones: bool,
    pub radar_rings: bool,
    pub fov_cones: bool,
    pub observations: bool,
    pub rejection_markers: bool,
    pub coverage_overlay: bool,
    pub inspection_events: bool,
    pub launch_lines: bool,
    pub covariance_ellipsoids: bool,
    pub belief: bool,
    pub trails: bool,
    pub scan_grid: bool,
    pub poi_markers: bool,
    pub loc_ellipses: bool,
    pub reconstruction: bool,
    pub coord_frame: bool,
    pub egress_paths: bool,
}

/// What `preset` shows.
pub fn preset_layers(preset: MapLayerPreset) -> PresetLayers {
    match preset {
        MapLayerPreset::Operator => PresetLayers {
            nodes: true,
            tracks: true,
            zones: true,
            radar_rings: false,
            fov_cones: false,
            observations: false,
            rejection_markers: false,
            coverage_overlay: false,
            inspection_events: false,
            launch_lines: false,
            covariance_ellipsoids: false,
            belief: false,
            trails: true,
            scan_grid: false,
            poi_markers: true,
            loc_ellipses: false,
            reconstruction: true,
            coord_frame: false,
            egress_paths: true,
        },
        MapLayerPreset::Reconstruction => PresetLayers {
            nodes: true,
            tracks: false,
            zones: true,
            radar_rings: false,
            fov_cones: false,
            observations: false,
            rejection_markers: false,
            coverage_overlay: true,
            inspection_events: true,
            launch_lines: false,
            covariance_ellipsoids: true,
            belief: true,
            trails: false,
            scan_grid: true,
            poi_markers: true,
            loc_ellipses: true,
            reconstruction: true,
            coord_frame: true,
            egress_paths: false,
        },
        MapLayerPreset::Safety => PresetLayers {
            nodes: true,
            tracks: true,
            zones: true,
            radar_rings: true,
            fov_cones: false,
            observations: false,
            rejection_markers: false,
            coverage_overlay: false,
            inspection_events: false,
            launch_lines: true,
            covariance_ellipsoids: false,
            belief: false,
            trails: true,
            scan_grid: false,
            poi_markers: false,
            loc_ellipses: false,
            reconstruction: true,
            coord_frame: false,
            egress_paths: true,
        },
    }
}

/// Set the toggles `preset` owns. Truth is always hidden: a preset never
/// shows it, and switching presets withdraws it.
pub fn apply_layer_preset(
    preset: MapLayerPreset,
    runtime: &mut RuntimeOverlayVisibility,
    mission: &mut MissionOverlaySettings,
) {
    let layers = preset_layers(preset);
    runtime.nodes = layers.nodes;
    runtime.tracks = layers.tracks;
    runtime.truths = false;
    runtime.zones = layers.zones;
    runtime.radar_rings = layers.radar_rings;
    runtime.fov_cones = layers.fov_cones;
    runtime.observations = layers.observations;
    runtime.rejection_markers = layers.rejection_markers;
    runtime.coverage_overlay = layers.coverage_overlay;
    runtime.inspection_events = layers.inspection_events;
    runtime.launch_lines = layers.launch_lines;
    runtime.show_covariance_ellipsoids = layers.covariance_ellipsoids;
    runtime.belief = layers.belief;
    runtime.trails = layers.trails;
    mission.show_scan_grid = layers.scan_grid;
    mission.show_poi_markers = layers.poi_markers;
    mission.show_loc_ellipses = layers.loc_ellipses;
    mission.show_reconstruction = layers.reconstruction;
    mission.show_coord_frame = layers.coord_frame;
    mission.show_egress_paths = layers.egress_paths;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_keeps_safety_geometry_and_identity_and_hides_truth() {
        for preset in MapLayerPreset::ALL {
            let mut runtime = RuntimeOverlayVisibility {
                truths: true,
                ..Default::default()
            };
            let mut mission = MissionOverlaySettings::default();
            apply_layer_preset(preset, &mut runtime, &mut mission);
            assert!(runtime.zones, "{preset:?} keeps the zones");
            assert!(runtime.nodes, "{preset:?} keeps the platforms");
            assert!(!runtime.truths, "{preset:?} withdraws truth");
        }
    }

    #[test]
    fn presets_say_different_things() {
        let operational = preset_layers(MapLayerPreset::Operator);
        let reconstruction = preset_layers(MapLayerPreset::Reconstruction);
        let safety = preset_layers(MapLayerPreset::Safety);
        assert_ne!(operational, reconstruction);
        assert_ne!(operational, safety);
        assert_ne!(reconstruction, safety);
        // §6.2's distinguishing content.
        assert!(
            reconstruction.belief && reconstruction.reconstruction && reconstruction.loc_ellipses
        );
        assert!(!operational.belief && !safety.belief);
        assert!(safety.launch_lines && safety.egress_paths && safety.trails);
        assert!(operational.tracks && operational.poi_markers && operational.trails);
    }

    #[test]
    fn the_startup_layers_are_the_operational_preset() {
        let mut runtime = RuntimeOverlayVisibility::default();
        let mut mission = MissionOverlaySettings::default();
        let (before_runtime, before_mission) = (runtime.clone(), mission.clone());
        apply_layer_preset(MapLayerPreset::default(), &mut runtime, &mut mission);
        assert_eq!(runtime, before_runtime);
        assert_eq!(mission, before_mission);
    }
}
