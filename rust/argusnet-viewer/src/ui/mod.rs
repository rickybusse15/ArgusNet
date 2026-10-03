use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_extras::{Column, TableBuilder};
use egui_plot::{HLine, Line, Plot, PlotPoints, Points, Polygon, VLine};

use crate::mission_zones::{zone_color_rgba, ProjectedZoneBadges};
use crate::replay::{EgressProgress, MarkerKind, ReplayState};
use crate::schema::ScenePackage;
use crate::state::{
    CurrentFrameMetrics, CurrentRuntimeMarkers, LayerVisibilityState, LoadedMissionZones,
    MapLayerPreset, MissionOverlaySettings, ReconstructionCloud, RuntimeOverlayVisibility,
    SelectionDetails, SelectionState, SimulationRunner, ViewMode, ViewerMode, ViewerUiState,
    ZoneFocus, ZoneOverlapModel,
};

mod commands;
mod inspectors;
mod shell;
mod timeline;
mod widgets;

use commands::*;
pub use inspectors::terrain_raster_status;
use inspectors::*;
use shell::*;
use timeline::*;
use widgets::*;
pub use widgets::{covariance_diagonal, health_color};

const MISSION_MODES: &[&str] = &["scan_map_inspect", "target_tracking"];

/// Distance (m) below which an egress drone is counted as "returned to home".
///
/// Must match the simulator's egress→complete transition threshold in
/// `src/argusnet/simulation/sim.py` so the HUD doesn't claim full RTH while
/// the mission is still in egress.
pub const EGRESS_ARRIVAL_THRESHOLD_M: f32 = 15.0;
const MAP_PRESETS: &[&str] = &[
    "small",
    "medium",
    "large",
    "xlarge",
    "regional",
    "theater",
    "operational",
];
const TERRAIN_PRESETS: &[&str] = &[
    "default",
    "alpine",
    "coastal",
    "urban_flat",
    "desert_canyon",
    "rolling_highlands",
    "lake_district",
    "jungle_canopy",
    "arctic_tundra",
    "military_compound",
    "river_valley",
    "mountain_pass",
];
const PLATFORM_PRESETS: &[&str] = &["baseline", "wide_area"];
const MOTION_PRESETS: &[&str] = &["sinusoid", "racetrack", "waypoint_patrol", "mixed"];
const DRONE_MODES: &[&str] = &["inspect", "search", "mixed"];

/// Node health fraction below which a node is flagged in the alert panel.
/// Health is computed in `argusnet_grpc.py` as `1.0 - dropout_probability * 0.5`.
const HEALTH_ALERT_THRESHOLD: f32 = 0.3;

const TOP_STATUS_HEIGHT: f32 = 52.0;
const BOTTOM_TIMELINE_HEIGHT: f32 = 126.0;
const DRAWER_WIDTH: f32 = 390.0;

/// Read-only resources the UI shows, grouped: the system is at Bevy's
/// sixteen-parameter limit.
#[derive(bevy::ecs::system::SystemParam)]
pub struct UiReads<'w> {
    selection: Res<'w, SelectionState>,
    markers: Res<'w, CurrentRuntimeMarkers>,
    rasters: Res<'w, crate::terrain_rasters::TerrainRasters>,
    belief: Res<'w, crate::belief_render::BeliefLayer>,
    camera: Res<'w, crate::camera_control::CameraControl>,
    prefs: Res<'w, crate::prefs::ViewerPrefs>,
}

#[allow(clippy::too_many_arguments)]
pub fn viewer_ui_system(
    mut contexts: EguiContexts,
    scene_package: Res<ScenePackage>,
    mut replay_state: ResMut<ReplayState>,
    mut layer_visibility: ResMut<LayerVisibilityState>,
    mut runtime_visibility: ResMut<RuntimeOverlayVisibility>,
    mut mission_overlay: ResMut<MissionOverlaySettings>,
    mut view_mode: ResMut<ViewMode>,
    reads: UiReads,
    frame_metrics: Res<CurrentFrameMetrics>,
    mission_zones: Res<LoadedMissionZones>,
    mut zone_overlap_model: ResMut<ZoneOverlapModel>,
    mut sim_runner: ResMut<SimulationRunner>,
    projected_badges: Res<ProjectedZoneBadges>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut ui_state: ResMut<ViewerUiState>,
    reconstruction: Res<ReconstructionCloud>,
) {
    let UiReads {
        selection,
        markers,
        rasters,
        belief,
        camera,
        prefs,
    } = reads;
    // `ctx_mut` is fallible in bevy_egui 0.42: no primary context means there is
    // no frame to draw into yet.
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    let context = context.clone();
    let supports_hot_swap = scene_package.is_synthetic_source();
    ui_state.focused_entity_label = selection.label().map(str::to_owned);
    let raster_status =
        terrain_raster_status(&rasters.report(), &belief, &scene_package.land_cover);

    // egui 0.36 unified `TopBottomPanel`/`SidePanel` into `Panel`, which attaches
    // to a `Ui` rather than to the `Context`. The shell therefore opens one
    // screen-sized root `Ui` and nests the operator panels inside it; panel order
    // still determines nesting, outermost first.
    let mut root = egui::Ui::new(
        context.clone(),
        egui::Id::new("argusnet_operator_shell"),
        egui::UiBuilder::new().max_rect(context.viewport_rect()),
    );
    let root = &mut root;

    egui::Panel::top("operator_status_bar")
        .exact_size(TOP_STATUS_HEIGHT)
        .show(root, |ui| {
            top_status_bar(
                ui,
                &scene_package,
                &mut replay_state,
                &mut view_mode,
                &mut ui_state,
                &camera,
                &prefs,
            );
        });

    egui::Panel::bottom("operator_timeline")
        .exact_size(BOTTOM_TIMELINE_HEIGHT)
        .show(root, |ui| {
            section_operator_timeline(ui, &mut replay_state, &mut ui_state);
        });

    if ui_state.drawer_open {
        egui::Panel::right("operator_drawer")
            .resizable(true)
            .default_size(DRAWER_WIDTH)
            .size_range(320.0..=540.0)
            .show(root, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| match ui_state.mode {
                        ViewerMode::Mission => drawer_mission_content(
                            ui,
                            &scene_package,
                            &replay_state,
                            &mission_zones,
                            &reconstruction,
                            &mut view_mode,
                            &mut runtime_visibility,
                            &mut mission_overlay,
                            &selection,
                            &markers,
                            &raster_status,
                            &mut layer_visibility,
                            &mut zone_overlap_model,
                            &mut ui_state,
                        ),
                        ViewerMode::Replay => {
                            drawer_replay_content(ui, &mut replay_state, &mut ui_state)
                        }
                        ViewerMode::Diagnostics => drawer_diagnostics_content(
                            ui,
                            &replay_state,
                            &frame_metrics,
                            &mission_zones,
                            &diagnostics,
                        ),
                        ViewerMode::Scenario => drawer_scenario_content(
                            ui,
                            &scene_package,
                            &replay_state,
                            &mission_zones,
                            &mut zone_overlap_model,
                            supports_hot_swap,
                            &mut sim_runner,
                        ),
                    });
            });
    }

    draw_zone_badges(&context, &projected_badges);
    draw_mission_phase_hud(&context, &replay_state);
}

#[cfg(test)]
mod tests {
    /// The operator is told why belief is missing, not shown a blank.
    #[test]
    fn the_raster_status_says_why_belief_is_unavailable() {
        use crate::belief_render::{BeliefLayer, BeliefRenderSnapshot, BeliefUnavailable};
        use crate::terrain_material::BeliefMode;
        use crate::terrain_rasters::RasterReport;
        let report = RasterReport {
            belief_mode: BeliefMode::Unavailable,
            land_cover_bound: false,
            uploaded_tiles: 3,
            waiting_tiles: 1,
            refused_tiles: 2,
        };
        let absent = crate::schema::LandCoverState::Absent;
        let text = terrain_raster_status(&report, &BeliefLayer::default(), &absent);
        assert!(text.contains("no belief data"), "{text}");
        assert!(text.contains("No land cover"), "{text}");
        let too_large = BeliefLayer::Unavailable(BeliefUnavailable::TooLarge { nx: 5000, ny: 3 });
        assert!(terrain_raster_status(&report, &too_large, &absent).contains("5000 x 3"));
        let ready = BeliefLayer::Ready(Box::new(
            BeliefRenderSnapshot::from_store(&crate::belief_render::synthetic_belief_store())
                .unwrap(),
        ));
        let text = terrain_raster_status(&report, &ready, &absent);
        assert!(
            text.contains("3 tile(s) uploaded, 1 waiting, 2 refused"),
            "{text}"
        );
    }

    use super::*;

    #[test]
    fn covariance_diagonal_extracts_4x4() {
        let flat_4x4: Vec<f64> = vec![
            1.0, 0.1, 0.0, 0.0, 0.1, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 16.0,
        ];
        let diag = covariance_diagonal(&flat_4x4);
        assert!((diag[0] - 1.0).abs() < 1e-9);
        assert!((diag[1] - 4.0).abs() < 1e-9);
        assert!((diag[2] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_diagonal_extracts_3x3() {
        let flat_3x3: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 8.0];
        let diag = covariance_diagonal(&flat_3x3);
        assert!((diag[0] - 2.0).abs() < 1e-9);
        assert!((diag[1] - 5.0).abs() < 1e-9);
        assert!((diag[2] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_diagonal_handles_empty() {
        let diag = covariance_diagonal(&[]);
        assert_eq!(diag, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn health_color_returns_expected_colors() {
        assert_eq!(health_color(0.9), egui::Color32::GREEN);
        assert_eq!(health_color(0.5), egui::Color32::YELLOW);
        assert_eq!(health_color(0.1), egui::Color32::RED);
    }

    #[test]
    fn egress_arrived_count_uses_shared_threshold() {
        let progress = vec![
            EgressProgress {
                drone_id: "arrived".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M - 0.1,
                home_position: [0.0, 0.0, 0.0],
            },
            EgressProgress {
                drone_id: "at-threshold".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M,
                home_position: [0.0, 0.0, 0.0],
            },
            EgressProgress {
                drone_id: "en-route".into(),
                distance_to_home_m: EGRESS_ARRIVAL_THRESHOLD_M + 5.0,
                home_position: [0.0, 0.0, 0.0],
            },
        ];

        assert_eq!(egress_arrived_count(&progress), 1);
    }
}
