import bevy_pbr::render::forward_io::VertexOutput;
import bevy_pbr::render::mesh_view_bindings::view;

@group(3) @binding(0) var<uniform> color: vec4<f32>;
// x: the world height below which nothing is drawn. y: 1 for a marker glyph.
@group(3) @binding(1) var<uniform> clip_floor: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    if mesh.world_position.z < clip_floor.x {
        discard;
    }
    let eye = normalize(view.world_position - mesh.world_position.xyz);
    let rim = pow(1.0 - abs(dot(normalize(mesh.world_normal), eye)), 2.0);
    // Z-up world scan bands. Derivative filtering avoids distant band aliasing.
    let phase = mesh.world_position.z * 0.6;
    let band = sin(phase) * (1.0 - smoothstep(0.5, 3.0, fwidth(phase)));
    let emission = 0.5 + 3.5 * rim + 0.12 * band;
    // `AlphaMode::Add` blends premultiplied (one, one-minus-source-alpha).
    // A volume or wall is light added to the scene: its colour times its
    // weight, alpha 0, as Bevy's own shaders write additive output. A marker
    // glyph is information: it also covers 85% of what is behind it, so it
    // keeps its colour over sunlit ground.
    let weight = color.a * 0.85;
    if clip_floor.y > 0.5 {
        return vec4<f32>(color.rgb * emission, weight);
    }
    return vec4<f32>(color.rgb * emission * weight, 0.0);
}
