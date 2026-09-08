//! Terrain material: elevation and slope banding with contour lines.
//!
//! The terrain is the largest surface in the frame and, until now, carried no
//! information at all — it was drawn with whatever `StandardMaterial` its baked
//! GLB happened to hold. This is the first half of `DELIVERY_PLAN.md`'s P2.1,
//! and it implements the roadmap principle that the visual world should reflect
//! the analytic one.
//!
//! # An extension, not a replacement
//!
//! This is an [`ExtendedMaterial`] over `StandardMaterial` rather than a
//! material of its own, and that is load-bearing. A standalone material would
//! silently drop everything the base PBR pass does: shadows, and — more
//! visibly — `DistanceFog`, which is what makes distant terrain fade into the
//! gradient sky instead of ending on a hard line. The shader calls
//! `main_pass_post_lighting_processing` for exactly that reason.
//!
//! # What is not here yet
//!
//! **Land-cover tint** needs `argusnet_world::land_cover::LandCoverLayer`
//! sampled on the CPU into a texture, with nearest filtering because the class
//! channel is categorical. **Belief** has no data to draw: `belief_cells` is in
//! the replay schema and `mapping/belief.py` builds the objects, but nothing
//! serialises them, so every replay carries an empty array. Both are follow-ups
//! rather than half-implementations here.

use bevy::asset::embedded_asset;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::world_serialization::WorldAssetRoot;

use crate::schema::ScenePackage;

/// The `semantic_kind` a style layer uses to declare itself terrain. Matching on
/// this rather than on a layer id keeps the material attached to the *meaning*
/// of a layer, so a scene that names its ground something else still gets it.
const TERRAIN_SEMANTIC_KIND: &str = "terrain";

/// Elevation ramp endpoints, as multipliers on the layer's own colour.
///
/// Multiplicative rather than absolute so the baked scene colour still decides
/// the palette and this only shapes it — and so the values do not have to be
/// authored around the tonemapper, which the sky's colours do.
const LOW_TINT: Vec3 = Vec3::new(0.66, 0.82, 1.10);
const HIGH_TINT: Vec3 = Vec3::new(1.28, 1.14, 0.80);

/// Tint for ground steeper than [`SLOPE_THRESHOLD_RAD`], and how strongly it is
/// mixed in. Steep ground reads as exposed rock rather than as vegetation.
const SLOPE_TINT: Vec3 = Vec3::new(0.82, 0.78, 0.74);
const SLOPE_TINT_STRENGTH: f32 = 0.55;
/// Deliberately shallow — about 7 degrees. The bundled site spans 17.5 m of
/// height over 4.3 km, so a threshold picked for mountains (26 degrees, the
/// first value here) never fires on it and the slope channel silently does
/// nothing. Gentle ground is the common case for this product, and what matters
/// there is telling flats from banks.
const SLOPE_THRESHOLD_RAD: f32 = 0.12;
const SLOPE_SOFTNESS_RAD: f32 = 0.18;

/// Contour line colour multiplier and strength.
const CONTOUR_TINT: Vec3 = Vec3::new(0.55, 0.60, 0.62);
const CONTOUR_STRENGTH: f32 = 0.65;

/// Contour half-width in screen pixels. Screen-space, not world-space, so the
/// lines stay legible at any zoom — the same reasoning as the marker glyphs.
const CONTOUR_WIDTH_PX: f32 = 1.1;

/// How many elevation bands to divide the terrain's own height range into, and
/// how many bands per contour line. Derived from the scene rather than fixed in
/// metres, so a 40 m site and a 4 km one both get a readable number of them.
const ELEVATION_BANDS: f32 = 24.0;
const BANDS_PER_CONTOUR: f32 = 2.0;

/// Depth of the band shading. Subtle: the bands are there to make height
/// readable as steps, not to repaint the terrain.
const BAND_STRENGTH: f32 = 0.8;

/// Fallback height range for a scene whose environment declares no terrain
/// summary. Only affects how the ramp is spread, never whether it draws.
const FALLBACK_ELEVATION_SPAN_M: f32 = 250.0;

#[derive(Clone, Debug, ShaderType)]
pub struct TerrainSettings {
    pub low_color: Vec4,
    pub high_color: Vec4,
    pub slope_color: Vec4,
    pub contour_color: Vec4,
    pub min_elevation_m: f32,
    pub max_elevation_m: f32,
    pub band_interval_m: f32,
    pub contour_interval_m: f32,
    pub contour_width_px: f32,
    pub slope_threshold_rad: f32,
    pub slope_softness_rad: f32,
    pub band_strength: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct TerrainExtension {
    #[uniform(100)]
    pub settings: TerrainSettings,
}

impl MaterialExtension for TerrainExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://argusnet_viewer/shaders/terrain.wgsl".into()
    }
}

/// The material actually applied to terrain meshes.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// Marks a spawned layer root whose style declares it terrain.
#[derive(Component)]
pub struct TerrainLayerRoot;

/// Marks a layer root whose meshes have had the material applied, so the walk
/// runs once rather than every frame.
#[derive(Component)]
struct TerrainMaterialApplied;

/// Terrain layer roots still awaiting their material.
type PendingTerrainRoots<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static WorldAssetRoot),
    (With<TerrainLayerRoot>, Without<TerrainMaterialApplied>),
>;

impl TerrainSettings {
    /// Derives the settings from a scene's own height range.
    pub fn for_elevation_range(min_elevation_m: f32, max_elevation_m: f32) -> Self {
        let span = (max_elevation_m - min_elevation_m).max(1.0);
        let band_interval_m = span / ELEVATION_BANDS;
        Self {
            low_color: LOW_TINT.extend(1.0),
            high_color: HIGH_TINT.extend(1.0),
            slope_color: SLOPE_TINT.extend(SLOPE_TINT_STRENGTH),
            contour_color: CONTOUR_TINT.extend(CONTOUR_STRENGTH),
            min_elevation_m,
            max_elevation_m,
            band_interval_m,
            contour_interval_m: band_interval_m * BANDS_PER_CONTOUR,
            contour_width_px: CONTOUR_WIDTH_PX,
            slope_threshold_rad: SLOPE_THRESHOLD_RAD,
            slope_softness_rad: SLOPE_SOFTNESS_RAD,
            band_strength: BAND_STRENGTH,
        }
    }
}

pub struct TerrainMaterialPlugin;

impl Plugin for TerrainMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/terrain.wgsl");
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            .add_systems(Update, apply_terrain_material_system);
    }
}

/// Whether a layer's style declares it terrain.
pub fn is_terrain_layer(scene_package: &ScenePackage, style_id: &str) -> bool {
    scene_package
        .style_for(style_id)
        .is_some_and(|style| style.semantic_kind == TERRAIN_SEMANTIC_KIND)
}

/// Swaps the baked `StandardMaterial` on terrain meshes for the extended one.
///
/// The meshes are children of the layer root, and the timing here is easy to
/// get wrong: `is_loaded_with_dependencies` reports that the *asset* finished
/// loading, which happens a frame or more before the glTF is instantiated into
/// entities. Gating on it alone finds an empty subtree and silently applies
/// nothing. So the root is marked done only once at least one mesh has actually
/// been converted, and the walk simply retries until then.
// Eight system parameters: the scene resources it reads, the subtree it walks,
// and the two asset collections it converts between.
#[allow(clippy::too_many_arguments)]
fn apply_terrain_material_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    scene_package: Res<ScenePackage>,
    roots: PendingTerrainRoots,
    children: Query<&Children>,
    meshes: Query<&MeshMaterial3d<StandardMaterial>>,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
) {
    let summary = scene_package.environment.terrain_summary.as_ref();
    let min_elevation_m = summary.and_then(|s| s.min_height_m).unwrap_or(0.0);
    let max_elevation_m = summary
        .and_then(|s| s.max_height_m)
        .unwrap_or(min_elevation_m + FALLBACK_ELEVATION_SPAN_M);
    let settings = TerrainSettings::for_elevation_range(min_elevation_m, max_elevation_m);

    for (root, asset) in &roots {
        if !asset_server.is_loaded_with_dependencies(&asset.0) {
            continue;
        }

        let mut descendants = 0usize;
        let mut converted = 0usize;
        // Iterative rather than recursive: glTF nesting depth is not bounded by
        // anything this code controls.
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Ok(node_children) = children.get(entity) {
                descendants += node_children.len();
                stack.extend(node_children.iter());
            }
            let Ok(existing) = meshes.get(entity) else {
                continue;
            };
            // Carry the baked material through as the base, so the scene's own
            // colour, roughness and alpha still apply and the extension only
            // shapes them.
            let base = standard_materials
                .get(&existing.0)
                .cloned()
                .unwrap_or_default();
            let material = terrain_materials.add(TerrainMaterial {
                base,
                extension: TerrainExtension {
                    settings: settings.clone(),
                },
            });
            commands
                .entity(entity)
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .insert(MeshMaterial3d(material));
            converted += 1;
        }

        if converted == 0 {
            // Not an error: the glTF has loaded but its entities are not in the
            // world yet. Leave the root unmarked and try again next frame.
            trace!("terrain layer has {descendants} descendant(s) and no meshes yet");
            continue;
        }
        debug!("applied the terrain material to {converted} mesh(es)");
        commands.entity(root).insert(TerrainMaterialApplied);
    }
}

#[cfg(test)]
mod terrain_settings_tests {
    use super::*;

    #[test]
    fn bands_and_contours_scale_with_the_scene() {
        let small = TerrainSettings::for_elevation_range(0.0, 40.0);
        let large = TerrainSettings::for_elevation_range(0.0, 4000.0);

        // A hundredfold height range gives a hundredfold band interval, so both
        // scenes show the same number of bands rather than one showing a single
        // flat band and the other a moire.
        assert!((large.band_interval_m / small.band_interval_m - 100.0).abs() < 1e-3);
        assert!((small.band_interval_m - 40.0 / ELEVATION_BANDS).abs() < 1e-6);
    }

    #[test]
    fn contours_are_a_whole_number_of_bands_apart() {
        let settings = TerrainSettings::for_elevation_range(12.0, 237.0);
        let ratio = settings.contour_interval_m / settings.band_interval_m;
        assert!((ratio - BANDS_PER_CONTOUR).abs() < 1e-6);
    }

    /// A flat scene must not divide by zero and must not produce a zero band
    /// interval, which the shader would clamp but which would look like a
    /// solid colour rather than terrain.
    #[test]
    fn a_flat_scene_still_produces_a_usable_interval() {
        let settings = TerrainSettings::for_elevation_range(50.0, 50.0);
        assert!(settings.band_interval_m > 0.0);
        assert!(settings.contour_interval_m > 0.0);
        assert!(settings.band_interval_m.is_finite());
    }

    /// An inverted or nonsensical range must not yield a negative interval.
    #[test]
    fn an_inverted_range_is_clamped_rather_than_negated() {
        let settings = TerrainSettings::for_elevation_range(100.0, 20.0);
        assert!(settings.band_interval_m > 0.0);
    }
}
