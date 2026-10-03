//! Commands: the controls that ask the simulation runner to do something.

use super::*;

pub(super) fn section_scenario(
    ui: &mut egui::Ui,
    supports_hot_swap: bool,
    sim_runner: &mut SimulationRunner,
) {
    ui.collapsing("Scenario", |ui| {
        if !supports_hot_swap {
            ui.label("Re-simulation only for synthetic scenes.");
        }

        ui.add_enabled_ui(supports_hot_swap && !sim_runner.is_busy(), |ui| {
            combo_box(
                ui,
                "Mission Mode",
                &sim_runner.mission_mode.clone(),
                MISSION_MODES,
                |v| {
                    sim_runner.mission_mode = v.to_string();
                },
            );
            ui.separator();
            combo_box(
                ui,
                "Map",
                &sim_runner.map_preset.clone(),
                MAP_PRESETS,
                |v| {
                    sim_runner.map_preset = v.to_string();
                },
            );
            combo_box(
                ui,
                "Terrain",
                &sim_runner.terrain_preset.clone(),
                TERRAIN_PRESETS,
                |v| {
                    sim_runner.terrain_preset = v.to_string();
                },
            );
            ui.checkbox(
                &mut sim_runner.clean_terrain,
                "Clean Terrain (terrain-only blockers off)",
            );
            combo_box(
                ui,
                "Platform",
                &sim_runner.platform_preset.clone(),
                PLATFORM_PRESETS,
                |v| {
                    sim_runner.platform_preset = v.to_string();
                },
            );
            ui.add(egui::Slider::new(&mut sim_runner.drone_count, 1..=8).text("Drones"));
            ui.add(
                egui::Slider::new(&mut sim_runner.ground_stations, 1..=12).text("Ground Stations"),
            );
            ui.add(
                egui::Slider::new(&mut sim_runner.duration_s, 30.0..=600.0)
                    .text("Duration (s)")
                    .step_by(10.0),
            );
            ui.add(egui::Slider::new(&mut sim_runner.seed, 0..=999).text("Seed"));

            // Mission-mode-specific parameters
            if sim_runner.mission_mode == "scan_map_inspect" {
                ui.separator();
                ui.label("Scan / Map / Inspect");
                ui.add(
                    egui::Slider::new(&mut sim_runner.scan_coverage_threshold, 0.3..=0.95)
                        .text("Scan threshold")
                        .step_by(0.05),
                );
                ui.add(egui::Slider::new(&mut sim_runner.poi_count, 1..=10).text("POI count"));
            } else {
                // target_tracking mode — show target-centric controls
                ui.separator();
                ui.label("Target Tracking");
                combo_box(
                    ui,
                    "Target Motion",
                    &sim_runner.target_motion.clone(),
                    MOTION_PRESETS,
                    |v| {
                        sim_runner.target_motion = v.to_string();
                    },
                );
                combo_box(
                    ui,
                    "Drone Mode",
                    &sim_runner.drone_mode.clone(),
                    DRONE_MODES,
                    |v| {
                        sim_runner.drone_mode = v.to_string();
                    },
                );
                ui.add(egui::Slider::new(&mut sim_runner.target_count, 1..=8).text("Targets"));
            }
        });

        if sim_runner.is_busy() {
            ui.add(
                egui::ProgressBar::new(sim_runner.progress)
                    .text(sim_runner.status_label())
                    .animate(true),
            );
        } else if supports_hot_swap && ui.button("\u{25b6} Run Simulation").clicked() {
            sim_runner.start();
        } else if !supports_hot_swap {
            ui.add_enabled(false, egui::Button::new("\u{25b6} Run Simulation"));
        }

        if let Some(error) = &sim_runner.error {
            ui.colored_label(egui::Color32::RED, error);
        }
    });
}

// ---------------------------------------------------------------------------
// Section: Mission Progress
// ---------------------------------------------------------------------------
