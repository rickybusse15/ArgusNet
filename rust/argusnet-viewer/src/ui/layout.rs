//! The operate-workspace shell's regions (VIEWER_UI §4.3) and how they give
//! way at reduced sizes (§4.4), as a pure function of the viewport so the
//! rules are testable without rendering.

/// Application identity, mission, workspace and source.
pub const HEADER_HEIGHT: f32 = 64.0;
/// System status and freshness.
pub const STATUS_LINE_HEIGHT: f32 = 28.0;
/// Mission clock, event scale and selected event.
pub const TIMELINE_HEIGHT: f32 = 100.0;
/// The compact, index-like mission rail.
pub const RAIL_WIDTH: f32 = 194.0;
/// The selection inspector.
pub const INSPECTOR_WIDTH: f32 = 252.0;
/// The inspector reduced to its essential fields (§4.4 rule 3).
pub const INSPECTOR_ESSENTIAL_WIDTH: f32 = 200.0;
/// The secondary-destination drawer (mission tools, replay, diagnostics,
/// scenario), when open.
pub const DRAWER_WIDTH: f32 = 390.0;

/// Below this the world canvas stops being useful, so the shell gives way
/// around it rather than compress it further.
pub const MIN_CANVAS_WIDTH: f32 = 560.0;
pub const MIN_CANVAS_HEIGHT: f32 = 360.0;

/// Where everything goes for one viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellLayout {
    /// `None`: collapsed to give the canvas room (§4.4 rule 2).
    pub rail: Option<f32>,
    pub inspector: f32,
    /// Only the inspector's essential fields fit (§4.4 rule 3).
    pub inspector_essential: bool,
    /// `None`: closed, or no room for it.
    pub drawer: Option<f32>,
    /// The drawer was asked for but there is no room.
    pub drawer_suppressed: bool,
    /// Below the supported operational viewport: a monitoring layout only.
    pub compact: bool,
    /// What is left for the world.
    pub canvas: [f32; 2],
}

impl ShellLayout {
    /// The header, status line and timeline are never given up: they carry
    /// the source indicator, safety state and time (§4.4 rule 1). Width is
    /// recovered in order: the mission rail, then the drawer, then the
    /// inspector's non-essential fields. What still does not fit is compact.
    pub fn for_viewport(width: f32, height: f32, drawer_open: bool) -> Self {
        Self::for_workspace(width, height, drawer_open, false)
    }

    /// As [`Self::for_viewport`], but when `rail_holds_workspace` (Review, whose
    /// report is the rail) the rail is never given up: the drawer and the
    /// inspector's non-essential fields give way instead, and what still does
    /// not fit is compact.
    pub fn for_workspace(
        width: f32,
        height: f32,
        drawer_open: bool,
        rail_holds_workspace: bool,
    ) -> Self {
        let mut layout = ShellLayout {
            rail: Some(RAIL_WIDTH),
            inspector: INSPECTOR_WIDTH,
            inspector_essential: false,
            drawer: drawer_open.then_some(DRAWER_WIDTH),
            drawer_suppressed: false,
            compact: false,
            canvas: [0.0, 0.0],
        };
        let canvas_width = |layout: &ShellLayout| {
            width - layout.rail.unwrap_or(0.0) - layout.inspector - layout.drawer.unwrap_or(0.0)
        };
        if canvas_width(&layout) < MIN_CANVAS_WIDTH && !rail_holds_workspace {
            layout.rail = None;
        }
        if canvas_width(&layout) < MIN_CANVAS_WIDTH && layout.drawer.is_some() {
            layout.drawer = None;
            layout.drawer_suppressed = true;
        }
        if canvas_width(&layout) < MIN_CANVAS_WIDTH {
            layout.inspector = INSPECTOR_ESSENTIAL_WIDTH;
            layout.inspector_essential = true;
        }
        let canvas_height = height - HEADER_HEIGHT - STATUS_LINE_HEIGHT - TIMELINE_HEIGHT;
        layout.canvas = [canvas_width(&layout).max(0.0), canvas_height.max(0.0)];
        layout.compact =
            layout.canvas[0] < MIN_CANVAS_WIDTH || layout.canvas[1] < MIN_CANVAS_HEIGHT;
        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_viewport_has_every_region_at_its_reference_size() {
        let layout = ShellLayout::for_viewport(1440.0, 900.0, false);
        assert_eq!(layout.rail, Some(194.0));
        assert_eq!(layout.inspector, 252.0);
        assert!(!layout.inspector_essential && !layout.compact);
        assert_eq!(layout.drawer, None);
        // The canvas is the remainder, and dominant.
        assert_eq!(layout.canvas, [994.0, 708.0]);
        assert!(layout.canvas[0] * layout.canvas[1] > 0.5 * 1440.0 * 900.0);
    }

    #[test]
    fn the_reference_viewport_fits_the_drawer_too() {
        let layout = ShellLayout::for_viewport(1440.0, 900.0, true);
        assert_eq!(layout.rail, Some(194.0));
        assert_eq!(layout.drawer, Some(DRAWER_WIDTH));
        assert_eq!(layout.canvas[0], 1440.0 - 194.0 - 252.0 - 390.0);
    }

    #[test]
    fn the_rail_collapses_before_the_canvas_is_compressed() {
        let layout = ShellLayout::for_viewport(1000.0, 900.0, false);
        assert_eq!(layout.rail, None);
        assert_eq!(
            layout.inspector, INSPECTOR_WIDTH,
            "the inspector keeps its fields"
        );
        assert_eq!(layout.canvas[0], 748.0);
        assert!(!layout.compact);
    }

    /// In Review the rail is the workspace: at the same 1000 px it stays,
    /// and the inspector gives up its non-essential fields instead.
    #[test]
    fn the_rail_stays_when_it_holds_the_workspace() {
        let layout = ShellLayout::for_workspace(1000.0, 900.0, false, true);
        assert_eq!(layout.rail, Some(RAIL_WIDTH));
        assert!(layout.inspector_essential);
        let narrow = ShellLayout::for_workspace(780.0, 900.0, true, true);
        assert_eq!(narrow.rail, Some(RAIL_WIDTH));
        assert!(narrow.drawer.is_none() && narrow.drawer_suppressed);
    }

    #[test]
    fn the_drawer_gives_way_after_the_rail() {
        let layout = ShellLayout::for_viewport(1280.0, 900.0, true);
        assert_eq!(layout.rail, None);
        assert_eq!(layout.drawer, Some(DRAWER_WIDTH));
        let layout = ShellLayout::for_viewport(1100.0, 900.0, true);
        assert_eq!(layout.drawer, None);
        assert!(
            layout.drawer_suppressed,
            "the shell says why it is not shown"
        );
        assert!(!layout.inspector_essential);
    }

    #[test]
    fn the_inspector_drops_to_essential_fields_last() {
        let layout = ShellLayout::for_viewport(780.0, 900.0, false);
        assert_eq!(layout.rail, None);
        assert!(layout.inspector_essential);
        assert_eq!(layout.inspector, INSPECTOR_ESSENTIAL_WIDTH);
        assert!(!layout.compact);
    }

    #[test]
    fn an_undersized_viewport_is_compact_but_keeps_header_status_and_timeline() {
        for (width, height) in [(700.0, 900.0), (1440.0, 500.0)] {
            let layout = ShellLayout::for_viewport(width, height, false);
            assert!(layout.compact, "{width}x{height}");
            assert!(layout.canvas[0] >= 0.0 && layout.canvas[1] >= 0.0);
        }
    }
}
