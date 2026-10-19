// Power-line conductors as screen-space lines (R3, see power_lines.rs).
//
// A conductor is 1.5 cm thick: drawn at its true size it vanishes beyond a
// few hundred metres, which is exactly the hazard. So each straight piece of
// wire is a quad expanded here, in screen space, to at least `width.x`
// pixels, or to its true thickness when the camera is close enough for that
// to be wider. Every vertex carries the piece's two ends: `position` is its
// own end, `other` the far one, and `shape` says which side of the wire it
// lies on (x, +1 left or -1 right), which end it is (y, -1 the start, +1 the
// end) and the conductor's radius in metres (z). Ends are pushed out along
// the wire by the half-width too, so consecutive pieces overlap at a bend
// instead of leaving a notch.
//
// The mesh is in world coordinates (its entity has the identity transform),
// so it is projected with the view alone: no mesh-transform imports, which
// also keeps this shader free of the libraries that load after it.

import bevy_pbr::render::mesh_view_bindings::view;

@group(3) @binding(0) var<uniform> color: vec4<f32>;
// x: the least width a conductor is drawn at, screen pixels.
@group(3) @binding(1) var<uniform> width: vec4<f32>;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) other: vec3<f32>,
    @location(2) shape: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// How far in front of the eye an end behind it is pulled to, in clip w.
const NEAR_W: f32 = 1.0e-3;

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var here = view.clip_from_world * vec4<f32>(v.position, 1.0);
    var there = view.clip_from_world * vec4<f32>(v.other, 1.0);
    if here.w < NEAR_W && there.w < NEAR_W {
        // The whole piece is behind the eye: outside the depth range.
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    // Clip space is linear along the piece, so an end behind the eye moves
    // along it to just in front.
    if here.w < NEAR_W {
        here = mix(here, there, (NEAR_W - here.w) / (there.w - here.w));
    } else if there.w < NEAR_W {
        there = mix(there, here, (NEAR_W - there.w) / (here.w - there.w));
    }
    let half_px = 0.5 * view.viewport.zw;
    let screen_here = here.xy / here.w * half_px;
    let screen_there = there.xy / there.w * half_px;
    // Along the wire from its start to its end, whichever end this is.
    let along = (screen_there - screen_here) * (-v.shape.y);
    let length_px = length(along);
    var forward = vec2<f32>(1.0, 0.0);
    if length_px > 1.0e-6 {
        forward = along / length_px;
    }
    let left = vec2<f32>(-forward.y, forward.x);
    // The conductor's true half-width in pixels here: the projection's
    // vertical focal length in pixels over the depth (w is 1 for an
    // orthographic view, where clip_from_view[1][1] is 2 / height).
    let focal_px = view.clip_from_view[1][1] * half_px.y;
    let half_width_px = max(0.5 * width.x, v.shape.z * focal_px / here.w);
    let offset_px = (left * v.shape.x + forward * v.shape.y) * half_width_px;
    out.clip_position = here + vec4<f32>(offset_px / half_px * here.w, 0.0, 0.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return color;
}
