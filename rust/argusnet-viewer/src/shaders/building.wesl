// Building facades (R1, R4): material, weathering, storeys and roofs.
//
// An extension to StandardMaterial, like the terrain's, so lighting,
// shadows, fog and the layer's translucency all stay. Only parts the scene
// builder marked as facades get it (building_material.rs). It reads:
// TEXCOORD_0 = (height above the building's own ground, metres along the
// wall); TEXCOORD_1 = (storey height in metres, 0 when the data gives no
// storeys and -1 on a roof; material class + 32 * level count).
//
// Every surface: the class's roughness and metalness, and a soft mottle.
// Walls: darker toward the ground; with known storeys, floor slabs, a
// cornice at the eave and windows in panes, taller on the ground floor.
// Roofs: tile and slate courses, staggered tiles, metal standing seams.
// Windows are drawn only between the ground and the eave of a building
// whose storeys the data gives: none are invented.

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
    /// Per material code (R4): x roughness, y metalness, z 1 when the class
    /// sets them.
    surfaces: array<vec4<f32>, 24>,
}

@group(3) @binding(100) var<uniform> facade: FacadeSettings;

// Smooth value noise in about [-1, 1].
fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i);
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return 2.0 * mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) - 1.0;
}

// 1 inside [lo, hi] of a repeating unit, anti-aliased by the pixel's span.
fn band(phase: f32, lo: f32, hi: f32) -> f32 {
    let f = fract(phase);
    let w = max(fwidth(phase), 1e-4);
    return smoothstep(lo - w, lo + w, f) * (1.0 - smoothstep(hi - w, hi + w, f));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // What the scene builder wrote (scene_build::roofed_mesh): the height
    // above the building's own ground and the distance along the wall
    // (TEXCOORD_0), the storey height (0 unknown, -1 on a roof) and the
    // material class and level count packed as class + 32 * levels
    // (TEXCOORD_1).
    let height = in.uv.x;
    let along = in.uv.y;
    let storey_m = in.uv_b.x;
    let on_roof = storey_m < -0.5;
    let packed = max(round(in.uv_b.y), 0.0);
    let levels = floor(packed / 32.0);
    let code = u32(clamp(packed - 32.0 * levels, 0.0, 23.0));

    // The material class's surface (R4), for walls and roofs alike.
    let surface = facade.surfaces[code];
    if surface.z > 0.5 {
        pbr_input.material.perceptual_roughness = surface.x;
        pbr_input.material.metallic = surface.y;
    }

    let world = in.world_position.xyz;
    var shade = 1.0;
    // Weathering: a soft mottle over every surface, at two scales, so no
    // wall or roof is one flat colour.
    shade *= 0.93 + 0.07 * value_noise(world * 0.35) + 0.04 * value_noise(world * 1.7);

    if on_roof {
        let n = normalize(in.world_normal);
        let slope = length(n.xy);
        if slope > 0.05 {
            let down = normalize(n.xy);
            let across = vec2<f32>(-down.y, down.x);
            if code == 11u || code == 12u {
                // Tiles (11) and slate (12): courses of constant height,
                // each a shade darker at its lower edge.
                let course = select(0.32, 0.22, code == 12u);
                shade *= 0.82 + 0.18 * band(world.z / course, 0.18, 1.0);
                // Tiles also break into staggered rows of tiles.
                if code == 11u {
                    let row = floor(world.z / course);
                    let s = dot(world.xy, across) / 0.3 + 0.5 * row;
                    shade *= 0.9 + 0.1 * band(s, 0.1, 1.0);
                }
            } else if code == 13u {
                // Metal: standing seams running down the slope.
                let s = dot(world.xy, across) / 0.55;
                shade *= 0.85 + 0.25 * (1.0 - band(s, 0.06, 0.94));
            }
        }
    } else {
        // A wall: darker toward the ground (grime, the street's shade).
        shade *= 0.78 + 0.22 * smoothstep(0.0, 4.0, height);
        if storey_m > 0.0 && levels > 0.0 {
            let storey = height / storey_m;
            let eave = levels;
            if storey >= 0.0 && storey < eave {
                // Floor slabs between storeys, and the cornice at the eave.
                shade *= 0.86 + 0.14 * band(storey, 0.04, 1.0);
                // Windows; taller on the ground floor (a shopfront),
                // each split into two panes by a mullion.
                let ground = storey < 1.0;
                let sill = select(facade.window.w, 0.12, ground);
                let tall = select(facade.window.z, 0.68, ground);
                let column = along / max(facade.window.x, 0.5);
                let half = 0.5 * facade.window.y;
                let window = band(storey, sill, sill + tall)
                    * band(column, 0.5 - half, 0.5 + half)
                    * (1.0 - band(column * 2.0, 0.47, 0.53))
                    * facade.window_color.a;
                let base = pbr_input.material.base_color;
                pbr_input.material.base_color = vec4<f32>(
                    mix(base.rgb, facade.window_color.rgb, window),
                    base.a,
                );
                pbr_input.material.perceptual_roughness =
                    mix(pbr_input.material.perceptual_roughness, 0.12, window);
            } else if storey >= eave && storey < eave + 0.15 {
                shade *= 0.8;
            }
        }
    }
    let base = pbr_input.material.base_color;
    pbr_input.material.base_color = vec4<f32>(base.rgb * shade, base.a);

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
