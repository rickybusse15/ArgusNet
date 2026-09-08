//! How marker symbology is sized and oriented.
//!
//! Two defects this replaces, both of which made the glyph layer close to
//! useless at the framing the viewer actually opens on.
//!
//! **Sizing.** Glyph radii were fixed world distances — `Track => 12.0` metres
//! and so on. At the default framing the camera sits about 6.3 km out with a
//! 45° vertical field of view, which puts roughly 5.7 metres in every logical
//! pixel: a 12 m radius is *two pixels*. Zoom in and the same glyph swells to
//! fill the screen. A symbol layer wants the opposite of both — constant on
//! screen, wherever the camera is.
//!
//! **Orientation.** Glyphs were built in the XY plane, using `Vec3::Y` for
//! "top". This world is Z-up, so `Vec3::Y` is *north*: the glyphs were flat
//! plates lying on the ground, read edge-on from any camera near the horizon,
//! and their "top" pointed sideways. They are now built in the camera's own
//! right/up basis, so they always face the viewer.
//!
//! The vertical spines stay on world Z deliberately. They mean altitude, so
//! they should stand up in the world rather than on the screen — and from
//! directly overhead, foreshortening them to a point is the correct reading.

use bevy::prelude::*;

/// Target on-screen radii, in logical pixels.
pub const TRACK_GLYPH_PX: f32 = 10.0;
pub const TRUTH_GLYPH_PX: f32 = 9.0;
pub const NODE_GLYPH_PX: f32 = 8.0;
// Rejection glyphs carry their own per-reason pixel radii in `RejectionMarkerStyle`.

/// Metres per pixel used when the camera cannot be inspected. Chosen so a
/// `TRACK_GLYPH_PX` glyph comes out at the 12 m radius this code used before,
/// i.e. the fallback reproduces the historical (wrong, but familiar) sizing
/// rather than collapsing the overlay to nothing.
const FALLBACK_METRES_PER_PIXEL: f32 = 1.2;

/// View depth below which the perspective scaling stops shrinking glyphs. A
/// marker at or behind the camera would otherwise get a zero or negative
/// radius.
const MIN_VIEW_DEPTH_M: f32 = 1.0;

/// The camera-facing basis and depth scaling used to draw glyphs.
#[derive(Clone, Copy, Debug)]
pub struct GlyphProjection {
    /// World direction that is "right" on screen.
    pub right: Vec3,
    /// World direction that is "up" on screen.
    pub up: Vec3,
    scale: GlyphScale,
}

#[derive(Clone, Copy, Debug)]
enum GlyphScale {
    /// Constant on screen: the world radius grows with view depth.
    Screen {
        eye: Vec3,
        forward: Vec3,
        /// World metres per logical pixel, per metre of view depth.
        metres_per_pixel_per_metre: f32,
    },
    /// Constant in the world, regardless of where the camera is.
    World { metres_per_pixel: f32 },
}

impl Default for GlyphProjection {
    /// The world-space fallback: the historical XY basis and fixed sizes.
    fn default() -> Self {
        Self {
            right: Vec3::X,
            up: Vec3::Y,
            scale: GlyphScale::World {
                metres_per_pixel: FALLBACK_METRES_PER_PIXEL,
            },
        }
    }
}

impl GlyphProjection {
    /// Derives the basis and scale from a perspective camera.
    ///
    /// Returns the world-space fallback for an orthographic or custom
    /// projection, or a camera whose render target has no size yet — those are
    /// states the overlay should survive, not disappear in.
    pub fn from_camera(
        transform: &GlobalTransform,
        projection: &Projection,
        viewport_height_px: Option<f32>,
    ) -> Self {
        let Projection::Perspective(perspective) = projection else {
            return Self::default();
        };
        let Some(height) = viewport_height_px.filter(|height| *height > 0.0) else {
            return Self::default();
        };

        // A perspective camera shows `2 * depth * tan(fov/2)` metres of height
        // at `depth` metres, spread over `height` pixels.
        let metres_per_pixel_per_metre = 2.0 * (perspective.fov * 0.5).tan() / height;

        Self {
            right: transform.right().into(),
            up: transform.up().into(),
            scale: GlyphScale::Screen {
                eye: transform.translation(),
                forward: transform.forward().into(),
                metres_per_pixel_per_metre,
            },
        }
    }

    /// World radius that draws a glyph at `pixels` logical pixels from `position`.
    pub fn radius(&self, position: Vec3, pixels: f32) -> f32 {
        match self.scale {
            GlyphScale::Screen {
                eye,
                forward,
                metres_per_pixel_per_metre,
            } => {
                // Depth along the view axis, not euclidean distance: that is
                // what perspective divide uses, so a glyph at the edge of the
                // frame keeps the same size as one at the centre.
                let depth = (position - eye).dot(forward).max(MIN_VIEW_DEPTH_M);
                pixels * metres_per_pixel_per_metre * depth
            }
            GlyphScale::World { metres_per_pixel } => pixels * metres_per_pixel,
        }
    }
}

#[cfg(test)]
mod glyph_tests {
    use super::*;

    /// The default framing of the standard test scene: 6.3 km out, 45° vertical
    /// field of view, in a 1440x920 window.
    const FOV: f32 = std::f32::consts::FRAC_PI_4;
    const HEIGHT_PX: f32 = 920.0;

    fn looking_down_from(eye: Vec3) -> GlobalTransform {
        GlobalTransform::from(Transform::from_translation(eye).looking_at(Vec3::ZERO, Vec3::Z))
    }

    fn projection() -> Projection {
        Projection::Perspective(PerspectiveProjection {
            fov: FOV,
            near: 1.0,
            far: 20_000.0,
            ..default()
        })
    }

    /// The property the whole change exists for: doubling the view depth
    /// doubles the world radius, so the on-screen size does not move.
    #[test]
    fn radius_is_constant_on_screen() {
        let eye = Vec3::new(0.0, 0.0, 1000.0);
        let glyphs =
            GlyphProjection::from_camera(&looking_down_from(eye), &projection(), Some(HEIGHT_PX));

        let near = glyphs.radius(Vec3::new(0.0, 0.0, 0.0), TRACK_GLYPH_PX);
        let far = glyphs.radius(Vec3::new(0.0, 0.0, -1000.0), TRACK_GLYPH_PX);

        assert!(
            (far / near - 2.0).abs() < 1e-3,
            "radius at 2000 m ({far}) should be twice the radius at 1000 m ({near})"
        );
    }

    /// And the size it holds is the size asked for. Projecting the radius back
    /// through the perspective divide must return the requested pixel count.
    #[test]
    fn radius_projects_back_to_the_requested_pixel_size() {
        let eye = Vec3::new(0.0, 0.0, 6300.0);
        let glyphs =
            GlyphProjection::from_camera(&looking_down_from(eye), &projection(), Some(HEIGHT_PX));

        let depth = 6300.0_f32;
        let radius = glyphs.radius(Vec3::ZERO, TRACK_GLYPH_PX);
        let visible_height_m = 2.0 * depth * (FOV * 0.5).tan();
        let pixels = radius / visible_height_m * HEIGHT_PX;

        assert!(
            (pixels - TRACK_GLYPH_PX).abs() < 1e-2,
            "asked for {TRACK_GLYPH_PX} px, got {pixels} px"
        );
    }

    /// The bug this replaces, stated as a test: at the default framing the old
    /// fixed 12 m radius was about two pixels across.
    #[test]
    fn fixed_world_radii_were_invisible_at_default_framing() {
        let depth = 6300.0_f32;
        let visible_height_m = 2.0 * depth * (FOV * 0.5).tan();
        let old_pixels = 12.0 / visible_height_m * HEIGHT_PX;

        assert!(
            old_pixels < 3.0,
            "the 12 m track glyph should be under 3 px at 6.3 km, was {old_pixels}"
        );
    }

    /// Camera-facing means the glyph plane contains no component along the view
    /// axis — the failure it replaces is a glyph lying flat in XY and seen
    /// edge-on.
    #[test]
    fn basis_faces_the_camera() {
        let eye = Vec3::new(1500.0, -2000.0, 900.0);
        let transform = looking_down_from(eye);
        let glyphs = GlyphProjection::from_camera(&transform, &projection(), Some(HEIGHT_PX));

        let forward: Vec3 = transform.forward().into();
        assert!(glyphs.right.dot(forward).abs() < 1e-5);
        assert!(glyphs.up.dot(forward).abs() < 1e-5);
        assert!(glyphs.right.cross(glyphs.up).dot(forward).abs() > 0.99);
    }

    /// A low camera is exactly where the old XY-plane glyphs vanished, so the
    /// basis has to stay usable there too.
    #[test]
    fn basis_faces_the_camera_from_near_the_horizon() {
        let eye = Vec3::new(6000.0, 0.0, 300.0);
        let transform = looking_down_from(eye);
        let glyphs = GlyphProjection::from_camera(&transform, &projection(), Some(HEIGHT_PX));

        let forward: Vec3 = transform.forward().into();
        assert!(glyphs.right.dot(forward).abs() < 1e-5);
        assert!(glyphs.up.dot(forward).abs() < 1e-5);
        // The old code's "top" was Vec3::Y, which from here is almost entirely
        // across the view rather than up it.
        assert!(
            glyphs.up.z > 0.9,
            "screen-up should be close to world-up from a low camera, was {}",
            glyphs.up
        );
    }

    /// An orthographic camera, or one whose target has no size yet, falls back
    /// to the previous fixed world sizes rather than to zero.
    #[test]
    fn falls_back_to_fixed_world_sizes() {
        let transform = looking_down_from(Vec3::new(0.0, 0.0, 6300.0));

        let orthographic = GlyphProjection::from_camera(
            &transform,
            &Projection::Orthographic(OrthographicProjection::default_3d()),
            Some(HEIGHT_PX),
        );
        let unsized_target = GlyphProjection::from_camera(&transform, &projection(), None);
        let zero_height = GlyphProjection::from_camera(&transform, &projection(), Some(0.0));

        for glyphs in [orthographic, unsized_target, zero_height] {
            assert_eq!(glyphs.radius(Vec3::ZERO, TRACK_GLYPH_PX), 12.0);
            assert_eq!(
                glyphs.radius(Vec3::new(0.0, 0.0, -5000.0), TRACK_GLYPH_PX),
                12.0
            );
        }
    }

    /// A marker level with or behind the camera must not produce a zero or
    /// negative radius.
    #[test]
    fn radius_stays_positive_behind_the_camera() {
        let eye = Vec3::new(0.0, 0.0, 1000.0);
        let glyphs =
            GlyphProjection::from_camera(&looking_down_from(eye), &projection(), Some(HEIGHT_PX));

        for position in [eye, eye + Vec3::Z * 500.0, Vec3::new(0.0, 0.0, 5000.0)] {
            assert!(glyphs.radius(position, TRACK_GLYPH_PX) > 0.0);
        }
    }
}
