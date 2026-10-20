// Building facade (R1): window rows from the data's storeys.
//
// An extension to StandardMaterial, like the terrain's, so lighting,
// shadows, fog and the layer's translucency all stay. Only meshes that carry
// the scene builder's window coordinates get it (building_material.rs):
// TEXCOORD_0 is (storey above the building's ground, metres along the
// wall), TEXCOORD_1.x the level count, or -1 where the data gave no storeys
// and on roofs. Windows are drawn only between the ground and the eave of a
// building whose storeys are known: none are invented.

import bevy_pbr::render::forward_io::{VertexOutput, FragmentOutput};
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
import bevy_pbr::render::pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing};

struct FacadeSettings {
    /// rgb: the glass; a: how strongly it replaces the wall.
    window_color: vec4<f32>,
    /// x: metres between window columns; y: window width as a share of a
    /// column; z: window height as a share of a storey; w: sill height as a
    /// share of a storey.
    window: vec4<f32>,
}

@group(3) @binding(100) var<uniform> facade: FacadeSettings;

// 1 inside [lo, hi] of a repeating unit, anti-aliased by the pixel's span.
fn band(phase: f32, lo: f32, hi: f32) -> f32 {
    let f = fract(phase);
    let w = max(fwidth(phase), 1e-4);
    return smoothstep(lo - w, lo + w, f) * (1.0 - smoothstep(hi - w, hi + w, f));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let storey = in.uv.x;
    let levels = in.uv_b.x;
    if levels > 0.0 && storey >= 0.0 && storey < levels {
        let column = in.uv.y / max(facade.window.x, 0.5);
        let half = 0.5 * facade.window.y;
        let window = band(storey, facade.window.w, facade.window.w + facade.window.z)
            * band(column, 0.5 - half, 0.5 + half)
            * facade.window_color.a;
        let base = pbr_input.material.base_color;
        pbr_input.material.base_color = vec4<f32>(
            mix(base.rgb, facade.window_color.rgb, window),
            base.a,
        );
        pbr_input.material.perceptual_roughness =
            mix(pbr_input.material.perceptual_roughness, 0.15, window);
    }

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
