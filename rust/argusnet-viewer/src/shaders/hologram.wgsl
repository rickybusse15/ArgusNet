#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var<uniform> color: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let eye = normalize(view.world_position - mesh.world_position.xyz);
    let rim = pow(1.0 - abs(dot(normalize(mesh.world_normal), eye)), 2.0);
    // Z-up world scan bands. Derivative filtering avoids distant band aliasing.
    let phase = mesh.world_position.z * 0.6;
    let band = sin(phase) * (1.0 - smoothstep(0.5, 3.0, fwidth(phase)));
    let emission = 0.5 + 3.5 * rim + 0.12 * band;
    return vec4<f32>(color.rgb * emission, color.a * 0.85);
}
