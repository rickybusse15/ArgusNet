use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::prelude::{Component, Entity, Resource, Vec3};

use crate::replay::{FrameMetrics, MissionZone, TerrainViewerMesh};

#[derive(Debug, Clone, Resource, Default)]
pub struct LayerEntityIndex {
    pub entities_by_layer: HashMap<String, Entity>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct LayerVisibilityState {
    pub base_layers: HashMap<String, bool>,
}

#[derive(Debug, Clone, PartialEq, Resource)]
pub struct RuntimeOverlayVisibility {
    pub diagnostic_glyphs: bool,
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
    /// The terrain material's belief overlay (P2.1). Off by default: with no
    /// belief data it draws the explicit unavailable state over everything.
    pub belief: bool,
    /// The terrain material's land-cover tint, where the scene carries one.
    pub land_cover_tint: bool,
    /// Trail ribbons behind every presented entity (P2.2).
    pub trails: bool,
    /// The run's power lines, towers and conductors (R3). A world
    /// structure, not an overlay: no layer preset owns it.
    pub power_lines: bool,
    /// The run's trees (R2). A world structure, not an overlay: no layer
    /// preset owns it.
    pub trees: bool,
    /// Each power-line span's clearance volume (R3): an overlay, off by
    /// default.
    pub power_line_clearance: bool,
    /// The power-line spans the platforms' own map holds, in a run that maps
    /// the world for itself (R3): what the drones know, not the true lines.
    /// On by default; nothing to draw in a run with a prior map.
    pub mapped_wires: bool,
    /// Buildings drawn at the scene's styled translucency, so drones and
    /// tracks behind them show through. Off by default: buildings are
    /// solid. No layer preset owns it.
    pub see_through_buildings: bool,
}

/// The startup layers are the Operational preset's (`crate::layers`).
impl Default for RuntimeOverlayVisibility {
    fn default() -> Self {
        Self {
            diagnostic_glyphs: false,
            tracks: true,
            truths: false,
            nodes: true,
            observations: false,
            rejection_markers: false,
            zones: true,
            fov_cones: false,
            radar_rings: false,
            coverage_overlay: false,
            inspection_events: false,
            launch_lines: false,
            show_covariance_ellipsoids: false,
            belief: false,
            land_cover_tint: true,
            trails: true,
            power_lines: true,
            trees: true,
            power_line_clearance: false,
            mapped_wires: true,
            see_through_buildings: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Resource)]
pub struct MissionOverlaySettings {
    pub show_scan_grid: bool,
    pub show_poi_markers: bool,
    pub show_loc_ellipses: bool,
    pub show_reconstruction: bool,
    pub show_coord_frame: bool,
    pub show_egress_paths: bool,
}

/// The startup layers are the Operational preset's (`crate::layers`).
impl Default for MissionOverlaySettings {
    fn default() -> Self {
        Self {
            show_scan_grid: false,
            show_poi_markers: true,
            show_loc_ellipses: false,
            show_reconstruction: true,
            show_coord_frame: false,
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

/// Primary operator mode for the viewer shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewerMode {
    /// Mission operations, drone health, map layers, selection, and alerts.
    #[default]
    Mission,
    /// Replay timeline, frame/event navigation, and playback inspection.
    Replay,
    /// Tracking, node, event, performance, and deconfliction diagnostics.
    Diagnostics,
    /// Scenario regeneration and synthetic mission controls.
    Scenario,
}

/// Preset grouping for high-level map and overlay controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapLayerPreset {
    /// Operator-focused default overlays.
    #[default]
    Operator,
    /// Reconstruction and localization-heavy overlays.
    Reconstruction,
    /// Safety, zones, and rejection-heavy overlays.
    Safety,
}

impl MapLayerPreset {
    pub const ALL: [MapLayerPreset; 3] = [
        MapLayerPreset::Operator,
        MapLayerPreset::Reconstruction,
        MapLayerPreset::Safety,
    ];

    /// VIEWER_UI §4's preset names.
    pub fn label(self) -> &'static str {
        match self {
            MapLayerPreset::Operator => "Operational",
            MapLayerPreset::Reconstruction => "Reconstruction",
            MapLayerPreset::Safety => "Safety",
        }
    }
}

/// The three primary operator verbs (VIEWER_UI §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Workspace {
    /// Supervise current mission execution.
    #[default]
    Operate,
    /// Define or revise bounded mission intent.
    Plan,
    /// Reconstruct and assess what happened.
    Review,
}

impl Workspace {
    pub const ALL: [Workspace; 3] = [Workspace::Operate, Workspace::Plan, Workspace::Review];

    pub fn label(self) -> &'static str {
        match self {
            Workspace::Operate => "Operate",
            Workspace::Plan => "Plan",
            Workspace::Review => "Review",
        }
    }

    /// Why the workspace has nothing to show yet: the contract it waits for.
    /// Shown instead of mock data (C2.3).
    pub fn unavailable_until(self) -> Option<&'static str> {
        match self {
            Workspace::Operate => None,
            Workspace::Plan => Some(
                "Plan is unavailable until the canonical site, mission and trajectory \
                 contracts exist (S1).",
            ),
            // Review reads the assurance report a client package carries (S1).
            Workspace::Review => None,
        }
    }
}

/// How the world is projected onto the canvas (VIEWER_UI §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Projection {
    #[default]
    Perspective,
    TopDown,
    Section,
}

impl Projection {
    pub const ALL: [Projection; 3] = [
        Projection::Perspective,
        Projection::TopDown,
        Projection::Section,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Projection::Perspective => "Perspective",
            Projection::TopDown => "Top-down",
            Projection::Section => "Section",
        }
    }

    /// Why the projection cannot be chosen yet, if it cannot.
    pub fn unavailable_until(self) -> Option<&'static str> {
        match self {
            Projection::Perspective => None,
            Projection::TopDown | Projection::Section => None,
        }
    }
}

/// Which vertical plane a Section cuts (C5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectionAxis {
    /// The plane runs east-west; the view looks north along +Y, east to the
    /// right.
    #[default]
    LookingNorth,
    /// The plane runs north-south; the view looks east along +X, north to the
    /// left.
    LookingEast,
}

impl SectionAxis {
    pub fn normal(self) -> Vec3 {
        match self {
            SectionAxis::LookingNorth => Vec3::Y,
            SectionAxis::LookingEast => Vec3::X,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SectionAxis::LookingNorth => "looking north",
            SectionAxis::LookingEast => "looking east",
        }
    }

    pub fn other(self) -> Self {
        match self {
            SectionAxis::LookingNorth => SectionAxis::LookingEast,
            SectionAxis::LookingEast => SectionAxis::LookingNorth,
        }
    }
}

/// A vertical slab through the world (VIEWER_UI §6.1: plane/frame, offset,
/// thickness). Its centre plane passes `offset_m` along the axis's normal
/// from the orbit focus; everything outside `thickness_m` is clipped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionPlane {
    pub axis: SectionAxis,
    pub offset_m: f32,
    pub thickness_m: f32,
}

impl Default for SectionPlane {
    fn default() -> Self {
        Self {
            axis: SectionAxis::LookingNorth,
            offset_m: 0.0,
            thickness_m: 50.0,
        }
    }
}

impl SectionPlane {
    pub const MIN_THICKNESS_M: f32 = 1.0;
    pub const MAX_THICKNESS_M: f32 = 2000.0;

    /// Whether `point` lies in the slab around `focus`. A point outside is
    /// clipped from view and from picking -- clipped, not observed-free.
    pub fn contains(&self, focus: Vec3, point: Vec3) -> bool {
        let along = (point - focus).dot(self.axis.normal()) - self.offset_m;
        along.abs() <= self.thickness_m * 0.5
    }

    /// Scale the thickness by `factor`, within the limits.
    pub fn scale_thickness(&mut self, factor: f32) {
        self.thickness_m =
            (self.thickness_m * factor).clamp(Self::MIN_THICKNESS_M, Self::MAX_THICKNESS_M);
    }

    /// Move the slab by `steps` thicknesses along its normal.
    pub fn step(&mut self, steps: f32) {
        self.offset_m += steps * self.thickness_m;
    }
}

/// UI-only presentation state for the operator shell.
#[derive(Debug, Clone, Resource)]
pub struct ViewerUiState {
    pub mode: ViewerMode,
    pub drawer_open: bool,
    pub selected_timeline_frame: Option<usize>,
    pub focused_entity_label: Option<String>,
    pub map_layer_preset: MapLayerPreset,
    /// A selection the inspector's entity list asks for, applied by
    /// `picking::keyboard_selection_system` (the UI only reads the selection).
    pub select_request: Option<crate::picking::SelectRequest>,
    /// The world canvas in logical pixels, `[min_x, min_y, max_x, max_y]`, as
    /// the shell last laid it out; `None` before the first layout (and in
    /// headless runs, which have no shell). Canvas overlays (the phase HUD,
    /// the live badge) anchor to it, and the picker treats a pointer outside
    /// it as over the UI.
    pub canvas_rect: Option<[f32; 4]>,
    /// The primary workspace (§4.1). Independent of the projection and the
    /// layer preset: three distinct controls, none touching the selection.
    pub workspace: Workspace,
    pub projection: Projection,
    /// The slab Section shows (C5.2).
    pub section: SectionPlane,
    /// The quality tier the operator chose (C6); `crate::quality` carries it
    /// to the renderer and the preferences.
    pub quality: crate::quality::QualityTier,
    /// UI scale (C7.2): egui's zoom, independent of the world's scale.
    /// `canvas_rect` is in UI points; window pixels are points times this.
    pub ui_scale: f32,
    /// Nonessential transitions off (C7.2).
    pub reduced_motion: bool,
    /// The preferences window is open.
    pub show_preferences: bool,
    /// The preferences window is waiting for a key to bind; the viewer's own
    /// keys are held until it has one.
    pub capturing_key: bool,
    /// The safety event the inspector shows, when one is open (C3.3). It
    /// takes the inspector over from the selection until closed; the
    /// selection itself is untouched.
    pub inspected_event: Option<crate::ui::EventRef>,
    /// The ground point a click on empty world asked about, in scene metres
    /// (C3.4). The map-cell inspector shows it when nothing is selected.
    pub inspected_cell: Option<[f32; 2]>,
}

impl ViewerUiState {
    /// The canvas's top-left corner, or the window's before any layout.
    pub fn canvas_origin(&self) -> [f32; 2] {
        self.canvas_rect.map_or([0.0, 0.0], |[x, y, _, _]| [x, y])
    }

    /// Whether a window position is on the world canvas. With no shell
    /// laid out, everything is.
    pub fn on_canvas(&self, position: [f32; 2]) -> bool {
        // The pointer comes in window pixels, the canvas in UI points.
        let scale = if self.ui_scale > 0.0 {
            self.ui_scale
        } else {
            1.0
        };
        let position = [position[0] / scale, position[1] / scale];
        self.canvas_rect.is_none_or(|[min_x, min_y, max_x, max_y]| {
            position[0] >= min_x
                && position[0] < max_x
                && position[1] >= min_y
                && position[1] < max_y
        })
    }
}

impl Default for ViewerUiState {
    fn default() -> Self {
        Self {
            mode: ViewerMode::Mission,
            // A secondary destination (VIEWER_UI §4.1): closed until asked
            // for, so the canvas stays the dominant region.
            drawer_open: false,
            selected_timeline_frame: None,
            focused_entity_label: None,
            map_layer_preset: MapLayerPreset::Operator,
            select_request: None,
            canvas_rect: None,
            workspace: Workspace::Operate,
            projection: Projection::Perspective,
            section: SectionPlane::default(),
            quality: crate::quality::QualityTier::default(),
            ui_scale: 1.0,
            reduced_motion: false,
            show_preferences: false,
            capturing_key: false,
            inspected_event: None,
            inspected_cell: None,
        }
    }
}

/// Marker component on the top-down orthographic camera used for the Split-mode
/// right-half reconstruction viewport.
#[derive(Component)]
pub struct ReconstructionCamera;

/// Which rendering layer the viewer shows.
#[derive(Debug, Clone, PartialEq, Eq, Resource, Default, clap::ValueEnum)]
pub enum ViewMode {
    /// Show the real terrain GLB (default).
    #[default]
    RealWorld,
    /// Hide terrain, show the accumulated LiDAR reconstruction only.
    ScanMap,
    /// Left half real terrain, right half the accumulated reconstruction.
    Split,
}

/// The presented domain entities. Built by the snapshot adapter.
pub use crate::presentation::CurrentRuntimeMarkers;

#[derive(Debug, Clone, Resource, Default)]
pub struct CurrentFrameMetrics {
    pub metrics: Option<FrameMetrics>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct SelectionState {
    pub selected: Option<Selection>,
}

/// One selected entity: what every kind has, plus the part that differs.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// What is selected. Resolved by key, never by label: a track and its
    /// target's truth share a label.
    pub key: crate::presentation::EntityKey,
    /// The presentation epoch `key` belongs to. A selection from another
    /// epoch is void, since its key may name a different entity there.
    pub epoch: u64,
    pub label: String,
    pub position: Vec3,
    pub velocity: Vec3,
    pub details: SelectionDetails,
}

/// The per-kind half of a selection.
///
/// This was previously one flat struct of seventeen `Option` fields covering
/// every kind at once, which meant each match arm had to remember to null the
/// other kinds' entries by hand — and two of them, `mode_probability_cv` and
/// `contributing_nodes`, were in fact never cleared, so a track's values
/// outlived it in the resource. Nothing displayed them, because rendering is
/// gated on the selected kind, but the state was wrong and the next reader of
/// those fields would have inherited the bug. As an enum the mistake is not
/// expressible: a selection carries its own kind's data and no other's.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionDetails {
    Track(TrackDetails),
    Truth(TruthDetails),
    Node(NodeDetails),
}

impl SelectionDetails {
    /// Human-readable kind, used as the inspector's heading.
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Track(_) => "Track",
            Self::Truth(_) => "Truth",
            Self::Node(_) => "Node",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackDetails {
    pub measurement_std_m: Option<f32>,
    pub update_count: Option<u32>,
    pub stale_steps: Option<u32>,
    pub track_error_m: Option<f32>,
    pub covariance_diag: Option<[f64; 3]>,
    /// IMM CV-model probability for the selected track.
    pub mode_probability_cv: Option<f32>,
    /// Node IDs that contributed to the track's last update.
    pub contributing_nodes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TruthDetails {
    pub nearest_track_dist_m: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeDetails {
    pub health: Option<f32>,
    pub sensor_type: Option<String>,
    pub fov_half_angle_deg: Option<f32>,
    pub max_range_m: Option<f32>,
    pub is_mobile: Option<bool>,
}

impl SelectionState {
    pub fn clear(&mut self) {
        self.selected = None;
    }

    /// Label of the current selection, if any.
    pub fn label(&self) -> Option<&str> {
        self.selected.as_ref().map(|s| s.label.as_str())
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

    /// Resolves `relative_path` against `scene_root`, returning `None` if the
    /// result would escape `asset_root` (an absolute path, or `..`
    /// components that break out of the scene package). Scene manifests can
    /// come from an untrusted, shared package, so a `None` here must be
    /// treated as a rejection, not silently passed through to the caller.
    ///
    /// `..` components are rejected explicitly rather than left to
    /// `strip_prefix` below: `Path` methods are purely lexical and never
    /// resolve `..`, so e.g. `scene-0001/../../../etc/passwd` still has
    /// `asset_root` as a literal component prefix and would otherwise pass.
    pub fn asset_path_for(&self, scene_root: &Path, relative_path: &str) -> Option<String> {
        if Path::new(relative_path)
            .components()
            .any(|component| component == std::path::Component::ParentDir)
        {
            return None;
        }
        let rooted = scene_root.join(relative_path);
        let relative = rooted.strip_prefix(&self.asset_root).ok()?;
        Some(relative.to_string_lossy().replace('\\', "/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn working_root() -> WorkingSceneRoot {
        WorkingSceneRoot {
            original_scene_root: PathBuf::from("/assets/scene-0001"),
            asset_root: PathBuf::from("/assets"),
            current_scene_root: PathBuf::from("/assets/scene-0001"),
            revision: 0,
            is_synthetic: false,
        }
    }

    #[test]
    fn a_section_slab_holds_what_lies_within_half_its_thickness() {
        let focus = Vec3::new(10.0, 20.0, 0.0);
        let mut plane = SectionPlane::default();
        assert!(plane.contains(focus, focus + Vec3::new(900.0, 25.0, 300.0)));
        assert!(!plane.contains(focus, focus + Vec3::new(0.0, 25.5, 0.0)));
        plane.step(2.0);
        assert_eq!(plane.offset_m, 100.0);
        assert!(plane.contains(focus, focus + Vec3::new(0.0, 110.0, 0.0)));
        assert!(!plane.contains(focus, focus));
        plane.axis = plane.axis.other();
        assert!(plane.contains(focus, focus + Vec3::new(100.0, 0.0, 0.0)));
        plane.scale_thickness(1e6);
        assert_eq!(plane.thickness_m, SectionPlane::MAX_THICKNESS_M);
        plane.scale_thickness(1e-9);
        assert_eq!(plane.thickness_m, SectionPlane::MIN_THICKNESS_M);
    }

    #[test]
    fn asset_path_for_resolves_contained_relative_path() {
        let root = working_root();
        let scene_root = PathBuf::from("/assets/scene-0001");
        assert_eq!(
            root.asset_path_for(&scene_root, "terrain/mesh.glb")
                .as_deref(),
            Some("scene-0001/terrain/mesh.glb")
        );
    }

    #[test]
    fn asset_path_for_rejects_absolute_path() {
        let root = working_root();
        let scene_root = PathBuf::from("/assets/scene-0001");
        assert_eq!(root.asset_path_for(&scene_root, "/etc/passwd"), None);
    }

    #[test]
    fn asset_path_for_rejects_dotdot_escape() {
        let root = working_root();
        let scene_root = PathBuf::from("/assets/scene-0001");
        assert_eq!(
            root.asset_path_for(&scene_root, "../../../../etc/passwd"),
            None
        );
    }

    #[test]
    fn asset_path_for_rejects_dotdot_within_an_otherwise_contained_path() {
        // Regression test: `strip_prefix` is purely lexical and does not
        // resolve `..`, so `scene-0001/../../../../etc/passwd` still has
        // `/assets` as a literal component prefix. Without the explicit
        // `..`-component check this would be silently accepted.
        let root = working_root();
        let scene_root = PathBuf::from("/assets/scene-0001");
        assert_eq!(
            root.asset_path_for(&scene_root, "terrain/../../../../etc/passwd"),
            None
        );
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    fn key(kind: crate::replay::MarkerKind, id: &str) -> crate::presentation::EntityKey {
        let mut ids = crate::presentation::IdTable::default();
        crate::presentation::EntityKey::new(kind, ids.intern(id))
    }

    fn track_selection() -> Selection {
        Selection {
            key: key(crate::replay::MarkerKind::Track, "t1"),
            epoch: 0,
            label: "t1".into(),
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            details: SelectionDetails::Track(TrackDetails {
                mode_probability_cv: Some(0.8),
                contributing_nodes: vec!["n1".into(), "n2".into()],
                ..Default::default()
            }),
        }
    }

    #[test]
    fn selecting_another_kind_cannot_inherit_the_previous_kind_s_data() {
        let mut state = SelectionState {
            selected: Some(track_selection()),
        };

        // The flat-struct version required each arm to null the other kinds'
        // fields by hand, and missed `mode_probability_cv` and
        // `contributing_nodes`, so a track's values survived into the next
        // selection. Replacing the whole variant makes that unrepresentable.
        state.selected = Some(Selection {
            key: key(crate::replay::MarkerKind::Node, "drone-a"),
            epoch: 0,
            label: "drone-a".into(),
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            details: SelectionDetails::Node(NodeDetails {
                health: Some(0.5),
                ..Default::default()
            }),
        });

        match &state.selected.as_ref().unwrap().details {
            SelectionDetails::Node(node) => assert_eq!(node.health, Some(0.5)),
            other => panic!("expected node details, got {other:?}"),
        }
    }

    #[test]
    fn kind_label_tracks_the_variant() {
        assert_eq!(
            SelectionDetails::Track(TrackDetails::default()).kind_label(),
            "Track"
        );
        assert_eq!(
            SelectionDetails::Truth(TruthDetails::default()).kind_label(),
            "Truth"
        );
        assert_eq!(
            SelectionDetails::Node(NodeDetails::default()).kind_label(),
            "Node"
        );
    }

    #[test]
    fn clearing_drops_the_selection_entirely() {
        let mut state = SelectionState {
            selected: Some(track_selection()),
        };
        assert_eq!(state.label(), Some("t1"));
        state.clear();
        assert!(state.selected.is_none());
        assert_eq!(state.label(), None);
    }
}
