use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::prelude::{Component, Entity, Resource, Vec3};

use crate::replay::{FrameMetrics, MissionZone, RuntimeMarker, TerrainViewerMesh};

#[derive(Debug, Clone, Resource, Default)]
pub struct LayerEntityIndex {
    pub entities_by_layer: HashMap<String, Entity>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct LayerVisibilityState {
    pub base_layers: HashMap<String, bool>,
}

#[derive(Debug, Clone, Resource)]
pub struct RuntimeOverlayVisibility {
    pub tracks: bool,
    pub truths: bool,
    pub nodes: bool,
    pub observations: bool,
    pub rejection_markers: bool,
    pub zones: bool,
    pub fov_cones: bool,
    pub radar_rings: bool,
    pub coverage_overlay: bool,
    pub inspection_events: bool,
    pub launch_lines: bool,
    pub show_covariance_ellipsoids: bool,
}

impl Default for RuntimeOverlayVisibility {
    fn default() -> Self {
        Self {
            tracks: true,
            truths: true,
            nodes: true,
            observations: false,
            rejection_markers: false,
            zones: true,
            fov_cones: false,
            radar_rings: true,
            coverage_overlay: false,
            inspection_events: true,
            launch_lines: true,
            show_covariance_ellipsoids: false,
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct MissionOverlaySettings {
    pub show_scan_grid: bool,
    pub show_poi_markers: bool,
    pub show_loc_ellipses: bool,
    pub show_reconstruction: bool,
    pub show_coord_frame: bool,
    pub show_egress_paths: bool,
}

impl Default for MissionOverlaySettings {
    fn default() -> Self {
        Self {
            show_scan_grid: true,
            show_poi_markers: true,
            show_loc_ellipses: true,
            show_reconstruction: true,
            show_coord_frame: true,
            show_egress_paths: true,
        }
    }
}

/// Persistent reconstruction cloud — accumulated by the viewer as the
/// replay plays back.  Reset when the replay is restarted or reloaded.
#[derive(Debug, Clone, Resource)]
pub struct ReconstructionCloud {
    /// Accumulated scan points: [x_m, y_m, terrain_height_m] in world coords.
    pub points: Vec<[f32; 3]>,
    /// Last replay frame index processed (used to reset on rewind).
    pub last_frame_index: usize,
    /// True when points changed since last GPU mesh upload.
    pub dirty: bool,
    /// Maximum terrain height seen across all scan points (for height-hue mapping).
    pub max_z: f32,
    /// Minimum terrain height seen across all scan points.
    pub min_z: f32,
}

impl Default for ReconstructionCloud {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            last_frame_index: 0,
            dirty: false,
            max_z: 1.0,
            min_z: 0.0,
        }
    }
}

impl ReconstructionCloud {
    pub fn reset(&mut self) {
        self.points.clear();
        self.last_frame_index = 0;
        self.dirty = true;
        self.max_z = 1.0;
        self.min_z = 0.0;
    }
}

/// Active tab in the dashboard tab bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource, Default)]
pub enum ActiveTab {
    /// 2-D top-down map + per-drone status cards.
    #[default]
    Mission,
    /// 3-D scene controls: overlays, layers, zones, selection, alerts.
    Scene,
    /// Tracking metrics, node table, track table.
    Tracks,
    /// Frame events, inspection events, deconfliction log.
    Events,
}

/// Marker component on the top-down orthographic camera used for the Split-mode
/// right-half reconstruction viewport.
#[derive(Component)]
pub struct ReconstructionCamera;

/// Which rendering layer the viewer shows.
#[derive(Debug, Clone, PartialEq, Eq, Resource, Default)]
pub enum ViewMode {
    /// Show the real terrain GLB (default).
    #[default]
    RealWorld,
    /// Hide terrain, show the accumulated LiDAR reconstruction only.
    ScanMap,
    /// Show terrain + reconstruction overlay simultaneously.
    Split,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct CurrentRuntimeMarkers {
    pub markers: Vec<RuntimeMarker>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct CurrentFrameMetrics {
    pub metrics: Option<FrameMetrics>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct SelectionState {
    pub selected_kind: Option<String>,
    pub selected_label: Option<String>,
    pub selected_position: Option<Vec3>,
    pub selected_velocity: Option<Vec3>,
    pub selected_health: Option<f32>,
    pub selected_measurement_std_m: Option<f32>,
    pub selected_update_count: Option<u32>,
    pub selected_stale_steps: Option<u32>,
    pub selected_track_error_m: Option<f32>,
    // Node-specific fields
    pub selected_sensor_type: Option<String>,
    pub selected_fov_half_angle_deg: Option<f32>,
    pub selected_max_range_m: Option<f32>,
    pub selected_is_mobile: Option<bool>,
    // Track-specific fields
    pub selected_covariance_diag: Option<[f64; 3]>,
    /// IMM CV-model probability for the selected track.
    pub selected_mode_probability_cv: Option<f32>,
    /// Node IDs that contributed to the selected track's last update.
    pub selected_contributing_nodes: Vec<String>,
    // Truth-specific fields
    pub selected_nearest_track_dist_m: Option<f32>,
}

impl SelectionState {
    pub fn clear(&mut self) {
        self.selected_kind = None;
        self.selected_label = None;
        self.selected_position = None;
        self.selected_velocity = None;
        self.selected_health = None;
        self.selected_measurement_std_m = None;
        self.selected_update_count = None;
        self.selected_stale_steps = None;
        self.selected_track_error_m = None;
        self.selected_sensor_type = None;
        self.selected_fov_half_angle_deg = None;
        self.selected_max_range_m = None;
        self.selected_is_mobile = None;
        self.selected_covariance_diag = None;
        self.selected_mode_probability_cv = None;
        self.selected_contributing_nodes.clear();
        self.selected_nearest_track_dist_m = None;
    }
}

#[derive(Debug, Clone, Resource, Default)]
pub struct LoadedMissionZones {
    pub zones: Vec<MissionZone>,
    pub terrain_mesh: Option<TerrainViewerMesh>,
}

// ---------------------------------------------------------------------------
// Simulation Runner with explicit phases
// ---------------------------------------------------------------------------

/// Phase of the simulation pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SimPhase {
    #[default]
    Idle,
    Simulating,
    BuildingScene,
    ReloadingScene,
    Error,
}

#[derive(Debug, Clone, Resource)]
pub struct SimulationRunner {
    pub phase: SimPhase,
    pub progress: f32,
    pub error: Option<String>,
    pub map_preset: String,
    pub terrain_preset: String,
    pub clean_terrain: bool,
    pub platform_preset: String,
    pub mission_mode: String,
    pub target_motion: String,
    pub drone_mode: String,
    pub target_count: u32,
    pub drone_count: u32,
    pub ground_stations: u32,
    pub duration_s: f32,
    pub seed: i64,
    pub scan_coverage_threshold: f32,
    pub poi_count: u32,
}

impl Default for SimulationRunner {
    fn default() -> Self {
        Self {
            phase: SimPhase::Idle,
            progress: 0.0,
            error: None,
            map_preset: "regional".into(),
            terrain_preset: "military_compound".into(),
            clean_terrain: false,
            platform_preset: "baseline".into(),
            mission_mode: "scan_map_inspect".into(),
            target_motion: "mixed".into(),
            drone_mode: "inspect".into(),
            target_count: 0,
            drone_count: 4,
            ground_stations: 7,
            duration_s: 180.0,
            seed: 7,
            scan_coverage_threshold: 0.70,
            poi_count: 3,
        }
    }
}

impl SimulationRunner {
    pub fn is_busy(&self) -> bool {
        matches!(
            self.phase,
            SimPhase::Simulating | SimPhase::BuildingScene | SimPhase::ReloadingScene
        )
    }

    pub fn start(&mut self) {
        self.phase = SimPhase::Simulating;
        self.progress = 0.0;
        self.error = None;
    }

    pub fn set_phase(&mut self, phase: SimPhase, progress: f32) {
        self.phase = phase;
        self.progress = progress.clamp(0.0, 1.0);
    }

    pub fn finish(&mut self) {
        self.phase = SimPhase::Idle;
        self.progress = 0.0;
        self.error = None;
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.phase = SimPhase::Error;
        self.error = Some(message.into());
    }

    pub fn status_label(&self) -> &'static str {
        match self.phase {
            SimPhase::Idle => "Idle",
            SimPhase::Simulating => "Simulating replay...",
            SimPhase::BuildingScene => "Building scene...",
            SimPhase::ReloadingScene => "Reloading scene...",
            SimPhase::Error => "Simulation failed",
        }
    }
}

// ---------------------------------------------------------------------------
// Zone Overlap Model
// ---------------------------------------------------------------------------

/// Focus selection for mission zones.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ZoneFocus {
    #[default]
    None,
    Zone(usize),
    Group(usize),
}

/// A group of overlapping zones whose circles intersect or nearly touch.
#[derive(Debug, Clone)]
pub struct ZoneOverlapGroup {
    pub group_id: usize,
    pub zone_indices: Vec<usize>,
    pub centroid_xy: [f32; 2],
    /// Line segments forming the merged outer contour (marching-squares output).
    pub contour_segments: Vec<[[f32; 2]; 2]>,
    /// Count of zones per zone_type within this group.
    pub type_counts: HashMap<String, usize>,
    /// Badge anchor point in world XY (defaults to centroid_xy).
    pub anchor_xy: [f32; 2],
    /// Group axis-aligned bounding box `[min_x, max_x, min_y, max_y]` for clipping.
    pub bounds_xy: [f32; 4],
}

/// Precomputed zone overlap model -- viewer-only presentation state.
#[derive(Debug, Clone, Resource, Default)]
pub struct ZoneOverlapModel {
    pub groups: Vec<ZoneOverlapGroup>,
    /// Maps zone index (into `LoadedMissionZones.zones`) to its group index.
    pub zone_to_group: HashMap<usize, usize>,
    pub focus: ZoneFocus,
    /// Bumped each time the model is recomputed so the UI can detect staleness.
    pub generation: u64,
}

impl ZoneOverlapModel {
    pub fn focused_zone_indices(&self) -> Vec<usize> {
        match self.focus {
            ZoneFocus::None => Vec::new(),
            ZoneFocus::Zone(zone_index) => vec![zone_index],
            ZoneFocus::Group(group_index) => self
                .groups
                .get(group_index)
                .map(|group| group.zone_indices.clone())
                .unwrap_or_default(),
        }
    }

    pub fn group_for_zone(&self, zone_index: usize) -> Option<usize> {
        self.zone_to_group.get(&zone_index).copied()
    }
}

// ---------------------------------------------------------------------------
// Working Scene Root (terrain hot-swap)
// ---------------------------------------------------------------------------

/// Stable working directory for scene hot-swap.
#[derive(Debug, Clone, Resource)]
pub struct WorkingSceneRoot {
    pub original_scene_root: PathBuf,
    pub asset_root: PathBuf,
    pub current_scene_root: PathBuf,
    pub revision: u32,
    pub is_synthetic: bool,
}

impl WorkingSceneRoot {
    pub fn supports_hot_swap(&self) -> bool {
        self.is_synthetic
    }

    pub fn next_scene_root(&self) -> PathBuf {
        if self.is_synthetic {
            self.asset_root
                .join(format!("scene-{:04}", self.revision.saturating_add(1)))
        } else {
            self.current_scene_root.clone()
        }
    }

    pub fn advance_to(&mut self, scene_root: PathBuf) {
        self.current_scene_root = scene_root;
        if self.is_synthetic {
            self.revision = self.revision.saturating_add(1);
        }
    }

    pub fn asset_path_for(&self, scene_root: &Path, relative_path: &str) -> String {
        let rooted = scene_root.join(relative_path);
        let Ok(relative) = rooted.strip_prefix(&self.asset_root) else {
            return relative_path.replace('\\', "/");
        };
        relative.to_string_lossy().replace('\\', "/")
    }
}
