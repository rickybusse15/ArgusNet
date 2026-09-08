//! Gradient sky.
//!
//! The viewer's first shader, and the one that establishes how shaders reach the
//! GPU here at all.
//!
//! **Why the shader is embedded in the binary.** `AssetPlugin::file_path` is set
//! to the *scene package* root (see `initialize_scene_package`) — for synthetic
//! scenes, a freshly created temp directory. There is no `assets/` directory to
//! put a `.wgsl` file in, and any relative path the asset server resolved would
//! change with the scene. `embedded_asset!` sidesteps that: the shader is
//! `include_bytes!`d at compile time and served from the `embedded://` asset
//! source, which is independent of `file_path`. That is also the honest model —
//! shaders are code, not scene content, and the viewer should render the same
//! way whatever package it opened.
//!
//! **Why a sphere and not `Skybox` or `Atmosphere`.** `Skybox` wants a cubemap
//! image, which would mean shipping (or baking) a texture for what is three
//! colours and a smoothstep. Bevy 0.19's procedural `Atmosphere` assumes a
//! **Y-up** planet and costs several LUT passes per frame — wrong for this Z-up
//! world, and wrong for a CI render that runs on a software rasteriser.

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::camera::visibility::RenderLayers;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::orbit_camera::OrbitCamera;

// The three colours below are *authored*, not displayed. Everything the main
// camera draws passes through `Tonemapping::AcesFitted`, which compresses hard
// at the bottom of the range: an authored 0.03 lands on a displayed 0. So each
// is written as the value that renders as the intended colour, and the intended
// colour is given alongside it. Re-tune by rendering, not by reading these.

/// The colour terrain fades into at the far edge of the fog band. Renders as
/// roughly `#38424f` — a slate that reads as haze, not as a wall.
///
/// Shared with `DistanceFog` so the two agree: if the fog resolved to anything
/// else, distant ridges would dissolve into a band of that colour standing in
/// front of the sky rather than merging with it.
pub const HORIZON_COLOR: Color = Color::srgb(0.341, 0.378, 0.431);

/// Straight up: renders as roughly `#0a0e1a`. Darker than the horizon, which is
/// what gives the frame a top.
const ZENITH_COLOR: Color = Color::srgb(0.150, 0.171, 0.234);

/// Below the horizon: renders as roughly `#101116`. Dark, but deliberately not
/// black — the first pass at this crushed a fifth of the frame to pure 0, which
/// loses the horizon glow behind the terrain plate entirely.
const NADIR_COLOR: Color = Color::srgb(0.181, 0.186, 0.212);

/// Half-width of the horizon haze above the horizon, as a fraction of the
/// hemisphere.
const SKY_SOFTNESS: f32 = 0.55;

/// The same below the horizon, and tuned to where the camera actually looks.
///
/// At the default isometric framing the horizon is not in shot at all: the
/// visible band runs from elevation -0.47 at the top of the frame to -0.89 at
/// the bottom, entirely below it. So this is what decides the backdrop in the
/// default view, and both ways of getting it wrong show up there — 0.55 crushed
/// the lower half to black, 1.6 flattened the whole frame to one slate tone.
const GROUND_SOFTNESS: f32 = 0.8;

/// Depth of the scan banding near the horizon, as a fraction of local brightness.
const BAND_STRENGTH: f32 = 0.06;

/// Bands over the full hemisphere.
const BAND_COUNT: f32 = 14.0;

/// Marks the sky sphere so the follow system can find it.
#[derive(Component)]
pub struct SkyDome;

#[derive(Clone, Debug, ShaderType)]
pub struct SkySettings {
    pub zenith: Vec4,
    pub horizon: Vec4,
    pub nadir: Vec4,
    pub sky_softness: f32,
    pub ground_softness: f32,
    pub band_strength: f32,
    pub band_count: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct SkyMaterial {
    #[uniform(0)]
    pub settings: SkySettings,
}

impl Default for SkyMaterial {
    fn default() -> Self {
        Self {
            settings: SkySettings {
                zenith: LinearRgba::from(ZENITH_COLOR).to_vec4(),
                horizon: LinearRgba::from(HORIZON_COLOR).to_vec4(),
                nadir: LinearRgba::from(NADIR_COLOR).to_vec4(),
                sky_softness: SKY_SOFTNESS,
                ground_softness: GROUND_SOFTNESS,
                band_strength: BAND_STRENGTH,
                band_count: BAND_COUNT,
            },
        }
    }
}

impl Material for SkyMaterial {
    fn fragment_shader() -> ShaderRef {
        // Matches the path `embedded_asset!` derives below: the crate name, then
        // the shader's path relative to `src/`.
        "embedded://argusnet_viewer/shaders/sky.wgsl".into()
    }

    /// The camera is *inside* the sphere, so the faces pointing at it are the
    /// back ones. Culling the front faces is what makes it a dome rather than an
    /// invisible ball.
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }

    /// A dome around the camera would otherwise enclose the whole scene in the
    /// directional light's cascades and shadow everything.
    fn enable_shadows() -> bool {
        false
    }

    fn enable_prepass() -> bool {
        false
    }
}

pub struct GradientSkyPlugin;

impl Plugin for GradientSkyPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/sky.wgsl");
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default())
            .add_systems(Update, sync_sky_to_camera_system);
    }
}

/// Spawns the sky dome. Called from `setup_world` once the scene bounds are known.
///
/// `render_layers` keeps it off the reconstruction pane, which draws the
/// *believed* world against a deliberately flat dark clear colour.
pub fn spawn_sky(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<SkyMaterial>,
    asset_server: &AssetServer,
    initial_radius: f32,
) {
    // Touch the embedded handle so the shader is loaded even before the first
    // material is specialized; without a live handle the embedded source is
    // never read and the pipeline compiles against a missing asset.
    let _shader: Handle<Shader> = load_embedded_asset!(asset_server, "shaders/sky.wgsl");

    // A unit sphere scaled per-frame, so the radius is one number in the follow
    // system rather than a mesh regeneration.
    let mesh = meshes.add(Sphere::new(1.0).mesh().ico(3).expect("ico sphere"));
    commands.spawn((
        SkyDome,
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(SkyMaterial::default())),
        Transform::from_scale(Vec3::splat(initial_radius.max(1.0))),
        RenderLayers::layer(0),
    ));
}

/// Keeps the dome centred on the camera and sized to stay behind the terrain.
///
/// The dome is ordinary geometry with ordinary depth testing — there is no
/// depth trickery here — so "behind everything" has to be literally true. The
/// farthest terrain from a camera `radius` out from the scene centre is at most
/// `radius + span/2 * sqrt(2)` away, so `radius + span` clears it with margin,
/// and stays inside `OrbitCamera::far_plane` (`(max_radius + span) * 1.3`) at
/// every zoom level.
pub fn sync_sky_to_camera_system(
    camera: Query<(&GlobalTransform, &OrbitCamera), With<crate::app::MainCamera>>,
    mut dome: Query<&mut Transform, With<SkyDome>>,
    scene_package: Res<crate::schema::ScenePackage>,
) {
    let Ok((camera_transform, orbit)) = camera.single() else {
        return;
    };
    let Ok(mut transform) = dome.single_mut() else {
        return;
    };
    let bounds = &scene_package.environment.bounds_xy_m;
    let span = (bounds.x_max_m - bounds.x_min_m)
        .max(bounds.y_max_m - bounds.y_min_m)
        .max(200.0);

    transform.translation = camera_transform.translation();
    transform.scale = Vec3::splat(sky_radius(orbit.radius, span));
}

/// Radius of the dome for a camera `radius` from the scene centre.
fn sky_radius(radius: f32, span: f32) -> f32 {
    (radius + span).max(1.0)
}

#[cfg(test)]
mod sky_tests {
    use super::*;

    // The scene used by the viewer's other framing tests: 4320 m x 4140 m.
    const SPAN: f32 = 4320.0;

    /// The dome is depth-tested like any other mesh, so anything it fails to
    /// enclose punches a hole in the sky. The worst case is a corner of the
    /// terrain, `span/2 * sqrt(2)` from the centre, seen from a camera on the
    /// opposite side.
    #[test]
    fn dome_encloses_the_terrain_at_every_zoom() {
        let farthest_terrain = |radius: f32| radius + SPAN * 0.5 * std::f32::consts::SQRT_2;
        for radius in [50.0, 500.0, 3000.0, 6300.0, 20000.0] {
            let dome = sky_radius(radius, SPAN);
            assert!(
                dome > farthest_terrain(radius),
                "dome radius {dome} at camera radius {radius} does not clear terrain at \
                 {}",
                farthest_terrain(radius)
            );
        }
    }

    /// ...and the other way it can be wrong: a dome past the far plane is
    /// clipped away, leaving the clear colour behind the terrain instead.
    #[test]
    fn dome_stays_inside_the_far_plane() {
        use crate::schema::Bounds2d;

        let bounds = Bounds2d {
            x_min_m: -2160.0,
            x_max_m: 2160.0,
            y_min_m: -1980.0,
            y_max_m: 2160.0,
        };
        let camera = OrbitCamera::from_bounds(&bounds, Some(0.0), Some(237.0));
        let far = camera.far_plane(SPAN);

        for radius in [50.0, camera.radius, camera.max_radius] {
            let dome = sky_radius(radius, SPAN);
            assert!(
                dome < far,
                "dome radius {dome} at camera radius {radius} is past the far plane {far}"
            );
        }
    }

    /// The fog fades terrain to `HORIZON_COLOR`; the sky renders that same
    /// colour at the horizon. If the two drift apart the horizon gains a seam.
    #[test]
    fn fog_and_sky_share_the_horizon_colour() {
        let material = SkyMaterial::default();
        assert_eq!(
            material.settings.horizon,
            LinearRgba::from(HORIZON_COLOR).to_vec4()
        );
    }
}
