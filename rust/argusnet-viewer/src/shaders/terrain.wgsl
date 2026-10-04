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
    /// x_min, y_min, cell size, tint strength.
    land_cover_grid: vec4<f32>,
    /// columns, rows, 1 when bound.
    land_cover_dims: vec4<u32>,
    land_cover_palette: array<vec4<f32>, 16>,
    /// x_min, y_min, cell size, overlay strength.
    belief_grid: vec4<f32>,
    /// columns, rows, mode (0 off, 1 unavailable, 2 ready).
    belief_dims: vec4<u32>,
    /// simulation time now, stale-after age.
    belief_time: vec4<f32>,
    belief_safe_color: vec4<f32>,
    belief_uncertain_color: vec4<f32>,
    belief_obstacle_color: vec4<f32>,
    belief_unknown_color: vec4<f32>,
    belief_outside_color: vec4<f32>,
}

@group(3) @binding(100) var<uniform> terrain: TerrainSettings;
// Belief (see belief_render.rs): class/observed exact, confidence/time numeric.
// Every read is a textureLoad: nothing categorical is ever filtered.
@group(3) @binding(101) var belief_class: texture_2d<u32>;
@group(3) @binding(102) var belief_numeric: texture_2d<f32>;
// Land-cover class per cell (see land_cover_raster.rs).
@group(3) @binding(103) var land_cover: texture_2d<u32>;

// Belief class codes, as BeliefClass in belief_render.rs.
const CLASS_OUTSIDE: u32 = 1u;
const CLASS_UNKNOWN: u32 = 2u;
const CLASS_OBSTACLE: u32 = 3u;

// The belief overlay's own shading: a fixed sun (the scene light's direction)
// and ambient floor, so relief still reads through it, and a brightness that
// puts its classes in the lit terrain's range before tonemapping.
const BELIEF_SUN: vec3<f32> = vec3<f32>(0.5726, 0.3436, 0.7444);
const BELIEF_AMBIENT: f32 = 0.55;
const BELIEF_EXPOSURE: f32 = 0.6;
const CLASS_UNCERTAIN: u32 = 4u;
const CLASS_SAFE: u32 = 5u;

// The cell containing a world point, or (-1, -1) off the grid. `floor`, not
// truncation: a point below the origin is outside, not in cell 0.
fn cell_of(xy: vec2<f32>, grid: vec4<f32>, dims: vec4<u32>) -> vec2<i32> {
    let cell = floor((xy - grid.xy) / max(grid.z, 1e-6));
    if cell.x < 0.0 || cell.y < 0.0 || cell.x >= f32(dims.x) || cell.y >= f32(dims.y) {
        return vec2<i32>(-1, -1);
    }
    return vec2<i32>(cell);
}

// Confidence, bilinear between the four nearest cell centres among observed
// cells only, renormalised: BeliefRenderSnapshot::sample, texel for texel.
fn belief_confidence(xy: vec2<f32>) -> f32 {
    let f = (xy - terrain.belief_grid.xy) / max(terrain.belief_grid.z, 1e-6) - vec2<f32>(0.5);
    let base = floor(f);
    let t = f - base;
    var weighted = 0.0;
    var weight = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let offset = vec2<f32>(f32(k & 1), f32(k >> 1u));
        let cell = base + offset;
        if cell.x < 0.0 || cell.y < 0.0
            || cell.x >= f32(terrain.belief_dims.x) || cell.y >= f32(terrain.belief_dims.y) {
            continue;
        }
        let texel = vec2<i32>(cell);
        if textureLoad(belief_class, texel, 0).g == 0u {
            continue;
        }
        let w = mix(1.0 - t.x, t.x, offset.x) * mix(1.0 - t.y, t.y, offset.y);
        weighted = weighted + w * textureLoad(belief_numeric, texel, 0).r;
        weight = weight + w;
    }
    return select(0.0, weighted / weight, weight > 0.0);
}

// Diagonal stripes in screen space, anti-aliased, 0..1. Unknown and
// unavailable are drawn as this pattern so they can never be mistaken for a
// measured value.
fn hatch(frag: vec2<f32>) -> f32 {
    let phase = fract((frag.x + frag.y) / 12.0);
    let edge = 1.0 / 12.0;
    return smoothstep(0.5 - edge, 0.5 + edge, phase) * (1.0 - smoothstep(1.0 - edge, 1.0, phase));
}

// The belief overlay's colour and weight at a point: vec4(rgb, weight).
fn belief_overlay(xy: vec2<f32>, frag: vec2<f32>) -> vec4<f32> {
    let mode = terrain.belief_dims.z;
    let strength = terrain.belief_grid.w;
    if mode == 0u {
        return vec4<f32>(0.0);
    }
    let stripes = hatch(frag);
    if mode == 1u {
        // Unavailable: no belief exists to draw, so everywhere is unknown.
        return vec4<f32>(terrain.belief_unknown_color.rgb * (0.6 + 0.4 * stripes), strength * (0.55 + 0.45 * stripes));
    }
    let cell = cell_of(xy, terrain.belief_grid, terrain.belief_dims);
    if cell.x < 0 {
        return vec4<f32>(0.0);
    }
    let class_code = textureLoad(belief_class, cell, 0).r;
    if class_code == CLASS_UNKNOWN {
        return vec4<f32>(terrain.belief_unknown_color.rgb * (0.6 + 0.4 * stripes), strength * (0.55 + 0.45 * stripes));
    }
    if class_code == CLASS_OUTSIDE {
        return vec4<f32>(terrain.belief_outside_color.rgb, strength);
    }
    var color = terrain.belief_safe_color.rgb;
    if class_code == CLASS_OBSTACLE {
        color = terrain.belief_obstacle_color.rgb;
    } else if class_code == CLASS_UNCERTAIN {
        color = terrain.belief_uncertain_color.rgb;
    } else {
        // Known safe: brighter the more confident.
        color = color * (0.45 + 0.55 * belief_confidence(xy));
    }
    // Stale: observed long enough ago to have faded toward grey.
    let last = textureLoad(belief_numeric, cell, 0).g;
    let age = terrain.belief_time.x - last;
    if age > terrain.belief_time.y {
        color = mix(color, vec3<f32>(dot(color, vec3<f32>(0.333))), 0.6);
    }
    return vec4<f32>(color, strength);
}

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

    let xy = in.world_position.xy;

    // Land cover: the class's colour, where the scene carries a raster.
    if terrain.land_cover_dims.z == 1u && terrain.land_cover_grid.w > 0.0 {
        let cell = cell_of(xy, terrain.land_cover_grid, terrain.land_cover_dims);
        if cell.x >= 0 {
            let code = min(textureLoad(land_cover, cell, 0).r, 15u);
            let class_color = terrain.land_cover_palette[code];
            tint = mix(tint, class_color.rgb, class_color.a * terrain.land_cover_grid.w);
        }
    }

    let base_rgb = pbr_input.material.base_color.rgb * tint;
    pbr_input.material.base_color = vec4<f32>(base_rgb, pbr_input.material.base_color.a);

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    // Fog and tonemapping live here. Dropping this line is what makes a custom
    // terrain material stop fading into the sky.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // Belief is information, not scenery: it is composited after lighting and
    // fog, with its own fixed relief shading, so no quality tier's shadows,
    // fog or bloom can shift a class's colour (VIEWER_UI §15). It replaces
    // rather than multiplies, so its classes stay legible over whatever colour
    // the terrain was baked with.
    let overlay = belief_overlay(xy, in.position.xy);
    if overlay.a > 0.0 {
        let relief = BELIEF_AMBIENT + (1.0 - BELIEF_AMBIENT) * clamp(dot(normal, BELIEF_SUN), 0.0, 1.0);
        let shaded = overlay.rgb * relief * BELIEF_EXPOSURE;
        out.color = vec4<f32>(mix(out.color.rgb, shaded, overlay.a), out.color.a);
    }
    return out;
}
