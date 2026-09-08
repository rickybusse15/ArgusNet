// Gradient sky for the mission-control look.
//
// Drawn on the inside of a sphere centred on the camera, so the only thing that
// matters is the *direction* from the camera to the fragment. This world is
// Z-up (Python ENU, the proto, and all three cameras agree), so elevation is the
// z component of that direction, not y.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

// Mirrors `SkySettings` in sky.rs, field for field.
struct SkySettings {
    // Straight up.
    zenith: vec4<f32>,
    // At the horizon, and the colour `DistanceFog` fades terrain into.
    horizon: vec4<f32>,
    // Below the horizon: the sphere continues under the camera, and at the low
    // pitches the orbit camera allows, a slice of it is visible past the edge of
    // the terrain.
    nadir: vec4<f32>,
    // Half-width of the horizon band above the horizon, as a fraction of the
    // hemisphere. Small values give a hard line, large ones a soft haze.
    sky_softness: f32,
    // The same below the horizon, and deliberately much wider: this camera
    // looks *down*, so the lower hemisphere is most of the frame, and a narrow
    // falloff there crushes a quarter of the image to black.
    ground_softness: f32,
    // Strength of the horizontal scan banding, 0 to disable. The banding is what
    // keeps the sky from reading as a plain vertical ramp and ties it to the
    // holographic overlays.
    band_strength: f32,
    // Number of bands over the full hemisphere.
    band_count: f32,
}

// 3 is `bevy_pbr::MATERIAL_BIND_GROUP_INDEX`: view is 0, mesh is 2.
@group(3) @binding(0) var<uniform> material: SkySettings;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let direction = normalize(mesh.world_position.xyz - view.world_position);
    // -1 straight down, 0 at the horizon, +1 straight up.
    let elevation = clamp(direction.z, -1.0, 1.0);

    // smoothstep away from the horizon in each direction, so the transition is
    // a haze rather than a seam. The two halves get their own widths.
    let up_mix = smoothstep(0.0, max(material.sky_softness, 1e-4), elevation);
    let sky = mix(material.horizon.rgb, material.zenith.rgb, up_mix);

    let down_mix = smoothstep(0.0, max(material.ground_softness, 1e-4), -elevation);
    let ground = mix(material.horizon.rgb, material.nadir.rgb, down_mix);

    var color = select(ground, sky, elevation >= 0.0);

    // Faint horizontal banding, strongest at the horizon and gone by the zenith,
    // so it reads as atmospheric depth rather than as a pattern on a dome.
    if material.band_strength > 0.0 {
        let phase = elevation * material.band_count * 6.2831853;
        let band = sin(phase) * 0.5 + 0.5;
        // Fades out towards either pole. The range is the whole hemisphere, not
        // a narrow band around the horizon: this camera looks down, so a band
        // that stopped at 0.6 would put the banding entirely out of shot.
        let falloff = 1.0 - smoothstep(0.0, 1.0, abs(elevation));
        color = color * (1.0 + material.band_strength * falloff * (band - 0.5));
    }

    // Opaque: the sphere is the backdrop, nothing is behind it.
    return vec4<f32>(color, 1.0);
}
