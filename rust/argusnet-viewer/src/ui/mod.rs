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

mod cell;
mod commands;
mod event;
mod inspectors;
mod layout;
mod platform;
mod shell;
mod source;
mod time;
mod timeline;
mod widgets;
mod workspace;

use cell::*;
use commands::*;
pub use event::EventRef;
use event::*;
pub use inspectors::terrain_raster_status;
use inspectors::*;
pub use layout::ShellLayout;
use layout::*;
use platform::*;
use shell::*;
pub use source::{SourceContext, SourceInputs};
use time::*;
pub use time::{apply_playback, PlaybackCommand};
use timeline::*;
use widgets::*;
pub use widgets::{covariance_diagonal, health_color};
use workspace::*;
pub use workspace::{
    apply_shell_command, close_event_on_escape_system, shell_keyboard_system, ShellCommand,
};

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
    // The theme's faces bind a pass after they are installed; the shell's
    // type roles need them, so the first frame draws nothing.
    if !crate::theme::fonts_ready(&context) {
        return;
    }
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

    let viewport = context.viewport_rect();
    let layout =
        ShellLayout::for_viewport(viewport.width(), viewport.height(), ui_state.drawer_open);
    let alerts = collect_alerts(&replay_state, &mission_zones);
    let frame_count = replay_state.frame_count();
    let source = SourceInputs {
        has_frames: frame_count > 0,
        live: replay_state.live,
        at_newest: frame_count > 0 && replay_state.frame_index + 1 >= frame_count,
        // The in-process worker and SIL/HIL adapters arrive with D1.
        sim: None,
        in_loop: None,
        truth_shown: runtime_visibility.truths,
    }
    .context();
    let canvas = show_shell(root, &layout, |region, ui| match region {
        Region::Header => {
            if let Some(command) =
                header_bar(ui, source, &scene_package, &mut view_mode, &mut ui_state)
            {
                apply_shell_command(
                    command,
                    &mut ui_state,
                    &mut runtime_visibility,
                    &mut mission_overlay,
                );
            }
        }
        Region::StatusLine => status_line(
            ui,
            &replay_state,
            alerts.as_deref(),
            &camera,
            &prefs,
            &layout,
        ),
        Region::Timeline => section_operator_timeline(ui, &mut replay_state, &mut ui_state),
        Region::Rail => {
            let workspace = ui_state.workspace;
            mission_rail(
                ui,
                workspace,
                &replay_state,
                &mission_zones,
                &markers,
                &selection,
                &mut ui_state,
            )
        }
        Region::Drawer => match ui_state.mode {
            ViewerMode::Mission => drawer_mission_content(
                ui,
                &scene_package,
                &replay_state,
                &mission_zones,
                &reconstruction,
                &mut view_mode,
                &mut runtime_visibility,
                &mut mission_overlay,
                &markers,
                &raster_status,
                &mut layer_visibility,
                &mut zone_overlap_model,
            ),
            ViewerMode::Replay => drawer_replay_content(ui, &replay_state),
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
        },
        Region::Inspector => selection_inspector(
            ui,
            &belief,
            &mut replay_state,
            &selection,
            &markers,
            &runtime_visibility,
            &mut ui_state,
            layout.inspector_essential,
        ),
    });

    ui_state.canvas_rect = Some([canvas.min.x, canvas.min.y, canvas.max.x, canvas.max.y]);
    draw_zone_badges(&context, &projected_badges);
    draw_mission_phase_hud(&context, &replay_state, canvas.min);
}

/// A region of the operate shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Region {
    Header,
    StatusLine,
    Timeline,
    Rail,
    Drawer,
    Inspector,
}

/// Lay the shell's regions out in `root` as `layout` says, drawing each with
/// `draw`, and return what is left: the world canvas. Flat, opaque regions
/// separated by hairlines (VIEWER_UI §4.3, §5.4). Panel order is nesting
/// order, outermost first: the header, status line and timeline span the
/// window; the rail, drawer and inspector sit between them.
fn show_shell(
    root: &mut egui::Ui,
    layout: &ShellLayout,
    mut draw: impl FnMut(Region, &mut egui::Ui),
) -> egui::Rect {
    let frame = |ui: &egui::Ui| {
        egui::Frame::NONE
            .fill(ui.visuals().panel_fill)
            .inner_margin(egui::Margin::symmetric(8, 4))
    };
    egui::Panel::top("operator_header")
        .exact_size(HEADER_HEIGHT)
        .resizable(false)
        .frame(frame(root))
        .show(root, |ui| draw(Region::Header, ui));
    egui::Panel::top("operator_status_line")
        .exact_size(STATUS_LINE_HEIGHT)
        .resizable(false)
        .frame(frame(root))
        .show(root, |ui| draw(Region::StatusLine, ui));
    egui::Panel::bottom("operator_timeline")
        .exact_size(TIMELINE_HEIGHT)
        .resizable(false)
        .frame(frame(root))
        .show(root, |ui| draw(Region::Timeline, ui));
    let scrolled =
        |region: Region, ui: &mut egui::Ui, draw: &mut dyn FnMut(Region, &mut egui::Ui)| {
            egui::ScrollArea::vertical()
                .id_salt(region)
                .auto_shrink([false; 2])
                .show(ui, |ui| draw(region, ui));
        };
    if let Some(width) = layout.rail {
        egui::Panel::left("operator_mission_rail")
            .exact_size(width)
            .resizable(false)
            .frame(frame(root))
            .show(root, |ui| scrolled(Region::Rail, ui, &mut draw));
    }
    if let Some(width) = layout.drawer {
        egui::Panel::right("operator_drawer")
            .exact_size(width)
            .resizable(false)
            .frame(frame(root))
            .show(root, |ui| scrolled(Region::Drawer, ui, &mut draw));
    }
    egui::Panel::right("operator_inspector")
        .exact_size(layout.inspector)
        .resizable(false)
        .frame(frame(root))
        .show(root, |ui| scrolled(Region::Inspector, ui, &mut draw));
    root.available_rect_before_wrap()
}

#[cfg(test)]
mod tests {
    /// The shell's real panels, laid out in a real egui context at the
    /// reference viewport and at reduced widths (C2.1).
    fn lay_out(
        width: f32,
        height: f32,
        drawer_open: bool,
    ) -> (ShellLayout, Vec<(Region, egui::Rect)>, egui::Rect) {
        use super::*;
        let context = egui::Context::default();
        let layout = ShellLayout::for_viewport(width, height, drawer_open);
        let mut regions = Vec::new();
        let mut canvas = egui::Rect::NOTHING;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, height),
            )),
            ..Default::default()
        };
        // Two passes: egui settles panel sizes on the second.
        for _ in 0..2 {
            regions.clear();
            // No renderer takes the font atlas uploads, so they are
            // discarded explicitly (egui asserts none are dropped unseen).
            context
                .run_ui(input.clone(), |ui| {
                    canvas = show_shell(ui, &layout, |region, ui| {
                        regions.push((region, ui.max_rect()));
                    });
                })
                .textures_delta
                .clear();
        }
        (layout, regions, canvas)
    }

    fn outer(regions: &[(super::Region, egui::Rect)], region: super::Region) -> Option<egui::Rect> {
        regions
            .iter()
            .find(|(r, _)| *r == region)
            .map(|(_, rect)| *rect)
    }

    #[test]
    fn the_shell_lays_out_at_the_reference_viewport() {
        use super::*;
        let (_, regions, canvas) = lay_out(1440.0, 900.0, false);
        // The canvas is exactly what the reference proportions leave.
        assert_eq!(canvas.min, egui::pos2(194.0, 92.0), "{canvas:?}");
        assert_eq!(canvas.size(), egui::vec2(994.0, 708.0), "{canvas:?}");
        for region in [
            Region::Header,
            Region::StatusLine,
            Region::Timeline,
            Region::Rail,
            Region::Inspector,
        ] {
            assert!(outer(&regions, region).is_some(), "{region:?} drawn");
        }
        assert!(
            outer(&regions, Region::Drawer).is_none(),
            "the drawer starts closed"
        );
        let header = outer(&regions, Region::Header).unwrap();
        let timeline = outer(&regions, Region::Timeline).unwrap();
        assert!(header.min.y < 8.0 && timeline.max.y > 892.0);
        let rail = outer(&regions, Region::Rail).unwrap();
        let inspector = outer(&regions, Region::Inspector).unwrap();
        assert!(rail.max.x <= canvas.min.x && inspector.min.x >= canvas.max.x);
    }

    #[test]
    fn a_narrow_shell_collapses_the_rail_but_keeps_status_timeline_and_inspector() {
        use super::*;
        let (layout, regions, canvas) = lay_out(1000.0, 900.0, false);
        assert_eq!(layout.rail, None);
        assert!(outer(&regions, Region::Rail).is_none());
        for region in [
            Region::Header,
            Region::StatusLine,
            Region::Timeline,
            Region::Inspector,
        ] {
            assert!(outer(&regions, region).is_some(), "{region:?} kept");
        }
        assert_eq!(canvas.min.x, 0.0);
        assert_eq!(canvas.width(), 748.0);
    }

    /// The source indicator lives in the header, and no width or drawer
    /// state removes the header (VIEWER_UI §3, C2.2).
    #[test]
    fn the_source_indicator_region_survives_every_width_and_drawer_state() {
        use super::*;
        for width in [1920.0, 1440.0, 1100.0, 1000.0, 780.0, 600.0, 320.0] {
            for drawer_open in [false, true] {
                let (_, regions, _) = lay_out(width, 900.0, drawer_open);
                let header = outer(&regions, Region::Header)
                    .unwrap_or_else(|| panic!("no header at {width} (drawer {drawer_open})"));
                // Inner rect: the window less the 8 px margin each side.
                assert!(
                    header.width() >= width - 17.0,
                    "the header spans the window"
                );
                assert!(outer(&regions, Region::StatusLine).is_some());
            }
        }
    }

    /// The platform inspector draws, full and reduced, and the reduced one
    /// is shorter.
    #[test]
    fn the_platform_inspector_draws_full_and_essential() {
        use super::*;
        let frame: crate::replay::ReplayFrame = serde_json::from_value(serde_json::json!({
            "timestamp_s": 1.0,
            "nodes": [{"node_id": "drone-1", "position": [0.0, 0.0, 10.0],
                       "velocity": [1.0, 0.0, 0.0], "is_mobile": true}]
        }))
        .unwrap();
        let sections = inspect_platform(&frame, "drone-1").unwrap();
        let context = egui::Context::default();
        // The type roles use the bundled faces, as the app installs them.
        crate::theme::install(&context);
        let mut heights = Vec::new();
        for essential in [false, true] {
            let mut height = 0.0;
            for _ in 0..2 {
                context
                    .run_ui(egui::RawInput::default(), |ui| {
                        let response =
                            ui.scope(|ui| platform_inspector(ui, "drone-1", &sections, essential));
                        height = response.response.rect.height();
                    })
                    .textures_delta
                    .clear();
            }
            heights.push(height);
        }
        assert!(heights[1] < heights[0], "essential is shorter: {heights:?}");
    }

    /// The fleet index lists platforms by the typed key picking resolves
    /// to; a track sharing a platform's label is not in it (C3).
    #[test]
    fn the_fleet_index_is_typed_like_picking() {
        use super::*;
        use crate::presentation::EntityKey;
        let document: crate::replay::ReplayDocument = serde_json::from_value(serde_json::json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "drone-1", "position": [0.0, 0.0, 50.0],
                            "velocity": [0.0, 0.0, 0.0]}],
                "nodes": [
                    {"node_id": "drone-1", "position": [0.0, 0.0, 80.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": true},
                    {"node_id": "ground-1", "position": [5.0, 0.0, 1.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": false}
                ]
            }]
        }))
        .unwrap();
        let mut markers = CurrentRuntimeMarkers::default();
        markers.apply(&ReplayState::new(Some(document)));
        let fleet = fleet_index(&markers);
        let labels: Vec<&str> = fleet.iter().map(|(_, label)| &**label).collect();
        assert_eq!(labels, ["drone-1", "ground-1"]);
        for (key, label) in &fleet {
            assert_eq!(Some(*key), markers.key_of(MarkerKind::Node, label));
            assert!(matches!(key, EntityKey::Platform(_)));
        }
        let track = markers.key_of(MarkerKind::Track, "drone-1").unwrap();
        assert!(
            fleet.iter().all(|(key, _)| *key != track),
            "same label, different kind"
        );
    }

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
