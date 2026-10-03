//! The operator UI's visual tokens (P2.5a, VIEWER_UI §5).
//!
//! One place for every UI colour, spacing step and type role, so the shell
//! (P2.5b onward) draws from the specification instead of from literals.
//! Scene colours (terrain, sensors, trails) are world presentation and live
//! with their renderers; these are the 2D instrument's.
//!
//! **Type.** VIEWER_UI §5.2 names Helvetica / Neue Haas Grotesk, both
//! commercial. Decision D-2 bundles an open, metric-compatible face instead:
//! Arimo 1.341 (SIL OFL 1.1, `assets/fonts/OFL.txt`), whose advance widths
//! match Helvetica/Arial and whose digits are tabular. It is compiled into the
//! binary, so it cannot fail to load at run time and the UI never falls back
//! to egui's default face silently. A licensed face can replace it by
//! swapping the two `include_bytes!` below. Arimo has no 300 weight; roles the
//! specification sets at 300 use Regular (400). egui's default fonts stay
//! behind Arimo only as glyph fallback for characters Arimo lacks.

use bevy_egui::egui;
use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, RichText, Shadow, Stroke,
};

// --------------------------------------------------------------------------
// Colour (VIEWER_UI §5.3)
// --------------------------------------------------------------------------

/// Deep neutral application ground.
pub const WINDOW: Color32 = Color32::from_rgb(0x0A, 0x0C, 0x0D);
/// Rails, header and timeline.
pub const CHROME: Color32 = Color32::from_rgb(0x10, 0x13, 0x15);
/// World viewport ground.
pub const CANVAS: Color32 = Color32::from_rgb(0x0C, 0x10, 0x12);
/// Structural boundaries.
pub const RULE_STRONG: Color32 = Color32::from_rgb(0x36, 0x3B, 0x3E);
/// Local grouping and table rows.
pub const RULE_QUIET: Color32 = Color32::from_rgb(0x28, 0x2E, 0x31);
/// Primary identities and values.
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xE2, 0xE2, 0xDF);
/// Normal readouts.
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xA5, 0xAA, 0xAC);
/// Labels and passive metadata. §5.3's reference is `#687175`, which reaches
/// only 3.7-3.9:1 on the three grounds, below the WCAG AA that §13 requires
/// for all UI text. This is the same hue lifted uniformly (+14 per channel)
/// to the first value that clears 4.5:1 on every ground (worst 4.56:1, on
/// chrome), still well below secondary (7.9:1), so the hierarchy holds.
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x76, 0x7F, 0x83);
/// Focus, selection, active index, actionable caution.
pub const ACCENT: Color32 = Color32::from_rgb(0xD2, 0xAD, 0x62);
/// Blocked execution, fault or immediate safety action, and nothing else
/// (§5.3). Not a specification value: chosen to meet WCAG AA on every
/// ground, which `tests::text_on_every_ground_meets_wcag_aa` checks.
pub const CRITICAL: Color32 = Color32::from_rgb(0xE8, 0x6A, 0x5C);

/// Grounds text may sit on.
pub const GROUNDS: [(&str, Color32); 3] =
    [("window", WINDOW), ("chrome", CHROME), ("canvas", CANVAS)];

// --------------------------------------------------------------------------
// Spacing (VIEWER_UI §5.4): a 4 px base, primary intervals 8-32
// --------------------------------------------------------------------------

pub const SPACE_BASE: f32 = 4.0;
pub const SPACE_8: f32 = 8.0;
pub const SPACE_12: f32 = 12.0;
pub const SPACE_16: f32 = 16.0;
pub const SPACE_20: f32 = 20.0;
pub const SPACE_24: f32 = 24.0;
pub const SPACE_32: f32 = 32.0;
/// Hierarchy rules are one pixel.
pub const RULE_WIDTH: f32 = 1.0;

// --------------------------------------------------------------------------
// Type (VIEWER_UI §5.2)
// --------------------------------------------------------------------------

/// The named egui family holding Arimo Medium (500).
pub const MEDIUM: &str = "arimo-medium";
const ARIMO_REGULAR: &[u8] = include_bytes!("../assets/fonts/Arimo-Regular.ttf");
const ARIMO_MEDIUM: &[u8] = include_bytes!("../assets/fonts/Arimo-Medium.ttf");

/// The six type roles of §5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeRole {
    /// Selected platform or track ID. 28-32 px, 300, +0.05 em.
    ObjectIdentity,
    /// Mission or review context. 16-18 px, 400, -0.01 em.
    WorkspaceTitle,
    /// Named content group. 13-15 px, 400-500, 0.
    SectionTitle,
    /// Values and concise explanation. 10-12 px, 400, +0.01 em.
    Body,
    /// Uppercase metadata and axes. 8-9 px, 400-500, +0.12 to +0.18 em.
    Utility,
    /// Time, coordinates, measurements. 9-21 px, 300-400, +0.03 to +0.08 em.
    Numeric,
}

impl TypeRole {
    /// Size in points, inside the specification's range.
    pub fn size(self) -> f32 {
        match self {
            TypeRole::ObjectIdentity => 28.0,
            TypeRole::WorkspaceTitle => 16.0,
            TypeRole::SectionTitle => 13.0,
            TypeRole::Body => 12.0,
            TypeRole::Utility => 9.0,
            TypeRole::Numeric => 12.0,
        }
    }

    /// Tracking in em.
    pub fn tracking_em(self) -> f32 {
        match self {
            TypeRole::ObjectIdentity => 0.05,
            TypeRole::WorkspaceTitle => -0.01,
            TypeRole::SectionTitle => 0.0,
            TypeRole::Body => 0.01,
            TypeRole::Utility => 0.14,
            TypeRole::Numeric => 0.04,
        }
    }

    /// Medium (500) where the specification allows it and it helps; Regular
    /// otherwise, including the 300 roles Arimo cannot set.
    pub fn family(self) -> FontFamily {
        match self {
            TypeRole::SectionTitle | TypeRole::Utility => FontFamily::Name(MEDIUM.into()),
            _ => FontFamily::Proportional,
        }
    }

    pub fn color(self) -> Color32 {
        match self {
            TypeRole::ObjectIdentity | TypeRole::WorkspaceTitle => TEXT_PRIMARY,
            TypeRole::SectionTitle | TypeRole::Body | TypeRole::Numeric => TEXT_SECONDARY,
            TypeRole::Utility => TEXT_MUTED,
        }
    }

    /// `text` set in this role. Utility labels are uppercase (§5.2).
    pub fn text(self, text: impl Into<String>) -> RichText {
        let text: String = text.into();
        let text = if self == TypeRole::Utility {
            text.to_uppercase()
        } else {
            text
        };
        RichText::new(text)
            .family(self.family())
            .size(self.size())
            .extra_letter_spacing(self.tracking_em() * self.size())
            .color(self.color())
    }
}

/// Arimo first in every family, egui's defaults behind it for glyphs Arimo
/// lacks. Arimo's digits are tabular, so it serves the monospace family's
/// numerals too.
pub fn font_definitions() -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("arimo".into(), FontData::from_static(ARIMO_REGULAR).into());
    fonts
        .font_data
        .insert(MEDIUM.into(), FontData::from_static(ARIMO_MEDIUM).into());
    let fallbacks = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "arimo".into());
    let mut medium = vec![MEDIUM.to_string()];
    medium.extend(fallbacks);
    fonts
        .families
        .insert(FontFamily::Name(MEDIUM.into()), medium);
    fonts
}

/// Square, flat, opaque, 1 px rules (§5.4); the palette above (§5.3).
pub fn apply_visuals(style: &mut egui::Style) {
    let visuals = &mut style.visuals;
    *visuals = egui::Visuals::dark();
    visuals.panel_fill = CHROME;
    visuals.window_fill = CHROME;
    visuals.extreme_bg_color = WINDOW;
    visuals.faint_bg_color = CANVAS;
    visuals.code_bg_color = WINDOW;
    visuals.window_stroke = Stroke::new(RULE_WIDTH, RULE_STRONG);
    visuals.window_corner_radius = CornerRadius::ZERO;
    visuals.menu_corner_radius = CornerRadius::ZERO;
    // Structural regions are flat. Only temporary overlays may lift (§5.4).
    visuals.window_shadow = Shadow::NONE;
    visuals.hyperlink_color = ACCENT;
    visuals.warn_fg_color = ACCENT;
    visuals.error_fg_color = CRITICAL;
    visuals.selection.bg_fill = ACCENT.linear_multiply(0.35);
    visuals.selection.stroke = Stroke::new(RULE_WIDTH, ACCENT);
    visuals.text_cursor.stroke = Stroke::new(RULE_WIDTH, ACCENT);

    let widgets = &mut visuals.widgets;
    // idle: muted label, quiet rule; hover: one contrast step brighter;
    // active: the accent (§5.6).
    let states = [
        (
            &mut widgets.noninteractive,
            CHROME,
            RULE_QUIET,
            TEXT_SECONDARY,
        ),
        (&mut widgets.inactive, CHROME, RULE_QUIET, TEXT_SECONDARY),
        (&mut widgets.hovered, CHROME, RULE_STRONG, TEXT_PRIMARY),
        (&mut widgets.active, CHROME, ACCENT, TEXT_PRIMARY),
        (&mut widgets.open, CHROME, RULE_STRONG, TEXT_PRIMARY),
    ];
    for (state, fill, rule, text) in states {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = Stroke::new(RULE_WIDTH, rule);
        state.fg_stroke = Stroke::new(RULE_WIDTH, text);
        state.corner_radius = CornerRadius::ZERO;
        state.expansion = 0.0;
    }

    style.spacing.item_spacing = egui::vec2(SPACE_8, SPACE_BASE);
    style.spacing.button_padding = egui::vec2(SPACE_8, SPACE_BASE);
    style.spacing.window_margin = egui::Margin::same(SPACE_12 as i8);
}

/// Install fonts and visuals on a context. Fonts replace the whole set, so
/// call this once, not every frame.
pub fn install(context: &egui::Context) {
    context.set_fonts(font_definitions());
    // egui keeps a style per theme; theme every one so a system switch to
    // light mode does not undo this.
    context.all_styles_mut(apply_visuals);
}

/// Whether the bundled faces are bound yet. Fonts set during a pass take
/// effect at the start of the next, and drawing a type role whose family is
/// not bound panics; the UI waits for this before drawing.
pub fn fonts_ready(context: &egui::Context) -> bool {
    context.fonts(|fonts| {
        fonts
            .definitions()
            .families
            .contains_key(&FontFamily::Name(MEDIUM.into()))
    })
}

/// Installs the theme on the first frame that has an egui context.
pub fn install_theme_system(
    mut contexts: bevy_egui::EguiContexts,
    mut installed: bevy::prelude::Local<bool>,
) {
    if *installed {
        return;
    }
    if let Ok(context) = contexts.ctx_mut() {
        install(context);
        *installed = true;
    }
}

// --------------------------------------------------------------------------
// Status vocabulary (VIEWER_UI §5.3, §5.6)
// --------------------------------------------------------------------------

/// Every state a readout can be in. Each has a label and a shape, so colour
/// is never the only cue (§5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    Nominal,
    Unknown,
    Stale,
    Degraded,
    Critical,
    Unavailable,
    Pending,
}

impl Status {
    pub const ALL: [Status; 7] = [
        Status::Nominal,
        Status::Unknown,
        Status::Stale,
        Status::Degraded,
        Status::Critical,
        Status::Unavailable,
        Status::Pending,
    ];

    /// Uppercase status vocabulary (§5.2).
    pub fn label(self) -> &'static str {
        match self {
            Status::Nominal => "NOMINAL",
            Status::Unknown => "UNKNOWN",
            Status::Stale => "STALE",
            Status::Degraded => "DEGRADED",
            Status::Critical => "CRITICAL",
            Status::Unavailable => "UNAVAILABLE",
            Status::Pending => "PENDING",
        }
    }

    /// A shape distinct per state, each present in Arimo.
    pub fn glyph(self) -> char {
        match self {
            Status::Nominal => '\u{25CF}',     // ● filled
            Status::Unknown => '\u{25CB}',     // ○ open
            Status::Stale => '\u{25CC}',       // ◌ dotted
            Status::Degraded => '\u{25B2}',    // ▲
            Status::Critical => '\u{25A0}',    // ■
            Status::Unavailable => '\u{2014}', // — no value
            Status::Pending => '\u{2026}',     // … awaiting
        }
    }

    /// Monochrome plus the accent; red only for critical (§5.3).
    pub fn color(self) -> Color32 {
        match self {
            Status::Nominal => TEXT_SECONDARY,
            Status::Unknown | Status::Unavailable => TEXT_MUTED,
            Status::Stale | Status::Pending => TEXT_SECONDARY,
            Status::Degraded => ACCENT,
            Status::Critical => CRITICAL,
        }
    }

    /// Glyph and label, in the utility role and the status colour.
    pub fn rich_text(self) -> RichText {
        TypeRole::Utility
            .text(format!("{} {}", self.glyph(), self.label()))
            .color(self.color())
    }
}

// --------------------------------------------------------------------------
// Fields: a readout that cannot pass missing off as zero
// --------------------------------------------------------------------------

/// A readout's value, or why there is none. Every inspector field is one of
/// these, so a missing value cannot render as `0`, empty or confident
/// (VIEWER_UI §12, agent plan rule 7).
#[derive(Debug, Clone, PartialEq)]
pub enum Field<T> {
    Value(T),
    /// No value, and why: "not in this replay", "no fix", ...
    Unavailable(&'static str),
    /// The last value, and how old it is.
    Stale {
        value: T,
        age_s: f64,
    },
}

impl<T> Field<T> {
    pub fn status(&self) -> Status {
        match self {
            Field::Value(_) => Status::Nominal,
            Field::Unavailable(_) => Status::Unavailable,
            Field::Stale { .. } => Status::Stale,
        }
    }

    /// `Some(value)` for a value; `None` otherwise, stale included.
    pub fn current(&self) -> Option<&T> {
        match self {
            Field::Value(value) => Some(value),
            _ => None,
        }
    }

    /// The text a readout shows: the formatted value, the stale value with
    /// its age, or the unavailable mark and reason. Never a default number.
    pub fn display(&self, format: impl Fn(&T) -> String) -> String {
        match self {
            Field::Value(value) => format(value),
            Field::Stale { value, age_s } => format!("{} ({age_s:.1} s old)", format(value)),
            Field::Unavailable(reason) => format!("{} {reason}", Status::Unavailable.glyph()),
        }
    }
}

impl<T> From<Option<T>> for Field<T> {
    /// `None` is "not reported", never zero.
    fn from(value: Option<T>) -> Self {
        value.map_or(Field::Unavailable("not reported"), Field::Value)
    }
}

// --------------------------------------------------------------------------
// Contrast (WCAG 2.x)
// --------------------------------------------------------------------------

/// WCAG relative luminance of an sRGB colour.
pub fn relative_luminance(color: Color32) -> f64 {
    let channel = |c: u8| {
        let c = f64::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
}

/// WCAG contrast ratio, from 1 to 21.
pub fn contrast_ratio(a: Color32, b: Color32) -> f64 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = if la > lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG AA: 4.5:1 for normal text.
    const AA: f64 = 4.5;

    #[test]
    fn text_on_every_ground_meets_wcag_aa() {
        // Muted is for labels and passive metadata; §5.3 still has it read,
        // so it is held to the same bar.
        let texts = [
            ("primary", TEXT_PRIMARY),
            ("secondary", TEXT_SECONDARY),
            ("muted", TEXT_MUTED),
            ("accent", ACCENT),
            ("critical", CRITICAL),
        ];
        let mut failures = Vec::new();
        for (ground, background) in GROUNDS {
            for (name, text) in texts {
                let ratio = contrast_ratio(text, background);
                if ratio < AA {
                    failures.push(format!("{name} on {ground}: {ratio:.2}"));
                }
            }
        }
        assert!(failures.is_empty(), "below WCAG AA: {failures:?}");
    }

    #[test]
    fn contrast_matches_the_wcag_reference_points() {
        assert!((contrast_ratio(Color32::BLACK, Color32::WHITE) - 21.0).abs() < 1e-9);
        assert!((contrast_ratio(ACCENT, ACCENT) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn every_status_differs_by_label_and_shape_not_only_colour() {
        let labels: std::collections::HashSet<_> = Status::ALL.iter().map(|s| s.label()).collect();
        let glyphs: std::collections::HashSet<_> = Status::ALL.iter().map(|s| s.glyph()).collect();
        assert_eq!(labels.len(), Status::ALL.len());
        assert_eq!(glyphs.len(), Status::ALL.len());
        // Red is reserved for critical (§5.3).
        for status in Status::ALL {
            assert_eq!(
                status.color() == CRITICAL,
                status == Status::Critical,
                "{status:?}"
            );
        }
    }

    #[test]
    fn a_missing_field_never_reads_as_a_number() {
        let missing: Field<f64> = None.into();
        assert_eq!(missing.status(), Status::Unavailable);
        assert_eq!(missing.current(), None);
        let shown = missing.display(|v| format!("{v:.1} m"));
        assert!(!shown.chars().any(|c| c.is_ascii_digit()), "{shown}");
        assert!(shown.contains("not reported"));

        let stale = Field::Stale {
            value: 12.0,
            age_s: 3.25,
        };
        assert_eq!(stale.current(), None, "stale is not current");
        assert_eq!(stale.display(|v| format!("{v:.1} m")), "12.0 m (3.2 s old)");
        assert_eq!(Field::Value(4.0).display(|v| format!("{v:.1}")), "4.0");
    }

    /// The UI's guard: the faces are not bound in the pass that installs
    /// them, and are in the next.
    #[test]
    fn fonts_are_ready_only_from_the_pass_after_install() {
        let context = egui::Context::default();
        let mut during = None;
        context
            .run_ui(egui::RawInput::default(), |ui| {
                install(ui.ctx());
                during = Some(fonts_ready(ui.ctx()));
            })
            .textures_delta
            .clear();
        assert_eq!(during, Some(false), "drawing a type role here would panic");
        let mut after = None;
        context
            .run_ui(egui::RawInput::default(), |ui| {
                after = Some(fonts_ready(ui.ctx()));
                // And drawing every role now works.
                for role in [
                    TypeRole::Utility,
                    TypeRole::SectionTitle,
                    TypeRole::ObjectIdentity,
                ] {
                    ui.label(role.text("ready"));
                }
            })
            .textures_delta
            .clear();
        assert_eq!(after, Some(true));
    }

    /// The bundled faces parse, Arimo is the primary proportional face, its
    /// digits are tabular, and every status glyph is in Arimo itself.
    #[test]
    fn the_bundled_face_is_installed_and_its_digits_are_tabular() {
        let context = egui::Context::default();
        install(&context);
        // Fonts take effect at the start of the next pass.
        // A test has no renderer to take the font atlas uploads, so they are
        // discarded explicitly (egui asserts that none are dropped unseen).
        for _ in 0..2 {
            context
                .run_ui(egui::RawInput::default(), |_| {})
                .textures_delta
                .clear();
        }

        let definitions = font_definitions();
        assert_eq!(
            definitions.families[&FontFamily::Proportional]
                .first()
                .map(String::as_str),
            Some("arimo")
        );
        assert_eq!(
            definitions.families[&FontFamily::Name(MEDIUM.into())]
                .first()
                .map(String::as_str),
            Some(MEDIUM)
        );

        let width = |text: &str, family: FontFamily| {
            context.fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(text.into(), egui::FontId::new(12.0, family), Color32::WHITE)
                    .size()
                    .x
            })
        };
        let digit_widths: Vec<f32> = "0123456789"
            .chars()
            .map(|d| width(&d.to_string(), FontFamily::Proportional))
            .collect();
        assert!(
            digit_widths.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-3),
            "digits are not tabular: {digit_widths:?}"
        );
        assert!(width("W", FontFamily::Proportional) > width("i", FontFamily::Proportional));

        // From Arimo's own character map: egui's `has_glyph` reports a false
        // negative when the owning face is also its replacement face.
        let face = ttf_parser::Face::parse(ARIMO_REGULAR, 0).expect("Arimo parses");
        for status in Status::ALL {
            assert!(
                face.glyph_index(status.glyph()).is_some(),
                "{status:?}'s glyph {:?} is not in Arimo",
                status.glyph()
            );
        }
    }

    #[test]
    fn visuals_are_square_flat_and_from_the_palette() {
        let mut style = egui::Style::default();
        apply_visuals(&mut style);
        let v = &style.visuals;
        for w in [
            &v.widgets.noninteractive,
            &v.widgets.inactive,
            &v.widgets.hovered,
            &v.widgets.active,
            &v.widgets.open,
        ] {
            assert_eq!(w.corner_radius, CornerRadius::ZERO);
            assert_eq!(w.bg_stroke.width, RULE_WIDTH);
        }
        assert_eq!(v.window_corner_radius, CornerRadius::ZERO);
        assert_eq!(v.window_shadow, Shadow::NONE);
        assert_eq!(v.panel_fill, CHROME);
        assert_eq!(v.widgets.active.bg_stroke.color, ACCENT);
    }
}
