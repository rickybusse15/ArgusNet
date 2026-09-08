// Terrain material: elevation and slope banding, contour lines, land-cover tint.
//
// An *extension* to StandardMaterial rather than a replacement, so the base PBR
// pass survives. That matters concretely: replacing the material would drop
// shadows and `DistanceFog`, and the fog is what makes distant terrain fade into
// the gradient sky instead of ending on a hard line.
//
// The world is Z-up (Python ENU, the proto and all three cameras agree), so
// elevation is `world_position.z` and the ground normal is close to +Z. Reading
// y here would give north, which is the mistake the marker glyphs used to make.

#import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing}

struct TerrainSettings {
    /// Elevation ramp endpoints, metres. Set from the scene's terrain summary.
    low_color: vec4<f32>,
    high_color: vec4<f32>,
    /// Tint applied where the ground is steeper than `slope_threshold_rad`.
    slope_color: vec4<f32>,
    contour_color: vec4<f32>,
    min_elevation_m: f32,
    max_elevation_m: f32,
    /// Spacing of the elevation bands, metres.
    band_interval_m: f32,
    /// Spacing of the contour lines, metres. Usually a multiple of the band.
    contour_interval_m: f32,
    /// Contour half-width, in screen pixels, so lines stay legible at any zoom.
    contour_width_px: f32,
    slope_threshold_rad: f32,
    slope_softness_rad: f32,
    /// 0 disables the banding, 1 applies it fully.
    band_strength: f32,
}

@group(3) @binding(100) var<uniform> terrain: TerrainSettings;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let elevation = in.world_position.z;
    let span = max(terrain.max_elevation_m - terrain.min_elevation_m, 1.0);
    let height_t = clamp((elevation - terrain.min_elevation_m) / span, 0.0, 1.0);

    // Elevation ramp.
    var tint = mix(terrain.low_color.rgb, terrain.high_color.rgb, height_t);

    // Discrete elevation bands, so height is readable as steps rather than as a
    // gradient the eye cannot measure. Alternating, not additive, so the ramp
    // above still carries the absolute height.
    let band_interval = max(terrain.band_interval_m, 0.01);
    let band_index = floor(elevation / band_interval);
    let band_parity = band_index - 2.0 * floor(band_index * 0.5);
    tint = tint * (1.0 + terrain.band_strength * (band_parity - 0.5) * 0.12);

    // Slope tint. `world_normal` is unit length, so its z component is the
    // cosine of the angle from vertical.
    let normal = normalize(in.world_normal);
    let slope_rad = acos(clamp(abs(normal.z), 0.0, 1.0));
    let steepness = smoothstep(
        terrain.slope_threshold_rad,
        terrain.slope_threshold_rad + max(terrain.slope_softness_rad, 1e-4),
        slope_rad,
    );
    tint = mix(tint, terrain.slope_color.rgb, steepness * terrain.slope_color.a);

    // Contour lines, anti-aliased in screen space. `fwidth` gives how much
    // elevation this pixel spans, so dividing by it converts the distance to
    // the nearest contour into pixels and holds the line a constant width at
    // any zoom or grazing angle -- the same derivative-filtering idiom
    // hologram.wgsl uses for its scan bands.
    let contour_interval = max(terrain.contour_interval_m, 0.01);
    let contour_phase = elevation / contour_interval;
    let distance_to_line = abs(fract(contour_phase) - 0.5);
    let phase_width = max(fwidth(contour_phase), 1e-5);
    let line_px = distance_to_line / phase_width;
    // 0.5 offset because `distance_to_line` peaks at 0.5 between lines.
    let contour = 1.0 - smoothstep(0.0, terrain.contour_width_px, line_px);
    tint = mix(tint, terrain.contour_color.rgb, contour * terrain.contour_color.a);

    pbr_input.material.base_color = vec4<f32>(
        pbr_input.material.base_color.rgb * tint,
        pbr_input.material.base_color.a,
    );

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    // Fog and tonemapping live here. Dropping this line is what makes a custom
    // terrain material stop fading into the sky.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
