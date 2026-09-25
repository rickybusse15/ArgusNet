//! The direction a directional sensor looks, and the angle a sight line makes
//! with it.
//!
//! Shared by the runtime's stage 6, which gates observations on it, and by the
//! viewer, which draws each sensor's field of view along the same axis
//! (P2.2), so the drawn volume is the gated one.

/// Below this **horizontal** speed a directional sensor has no look
/// direction, and the FOV gate is skipped rather than applied against a
/// degenerate heading. The reference's `speed_xy > 0.5`.
///
/// P4.8c used the full 3-D speed against 1e-6 m/s. A drone climbing
/// vertically, or hovering in a light drift, then had a look direction the
/// reference does not give it.
pub const MIN_LOOK_SPEED_XY_MPS: f64 = 0.5;

/// `drone_look_down_angle_deg`: how far below the heading a directional
/// sensor looks. The reference applies it to every directional sensor.
pub const DEFAULT_LOOK_DOWN_ANGLE_DEG: f64 = 30.0;

/// `sensor_look_direction`: along the velocity, tilted down, as a unit
/// vector; `None` below [`MIN_LOOK_SPEED_XY_MPS`] of horizontal speed.
///
/// The vertical component is **replaced**, not adjusted: the look is
/// `[vx, vy, -tan(tilt) * speed_xy]`, so a climbing drone still looks down.
/// Norms are the plain square root of the sum of squares, as numpy computes
/// them -- see `follow_path::norm`.
pub fn sensor_look_direction(velocity: [f64; 3], look_down_angle_deg: f64) -> Option<[f64; 3]> {
    let speed_xy = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
    if speed_xy <= MIN_LOOK_SPEED_XY_MPS {
        return None;
    }
    let look = [
        velocity[0],
        velocity[1],
        -look_down_angle_deg.to_radians().tan() * speed_xy,
    ];
    let norm = (look[0] * look[0] + look[1] * look[1] + look[2] * look[2]).sqrt();
    Some([look[0] / norm, look[1] / norm, look[2] / norm])
}

/// `fov_angle_deg`: the angle between the sight line and the look
/// direction. The sight line is normalised **first**, component by
/// component, then dotted -- the reference's order, which differs in the last
/// bit from dividing the dot product by the range.
pub fn fov_angle_deg(line_of_sight: [f64; 3], range_m: f64, look: [f64; 3]) -> f64 {
    let unit = [
        line_of_sight[0] / range_m,
        line_of_sight[1] / range_m,
        line_of_sight[2] / range_m,
    ];
    let cosine = unit[0] * look[0] + unit[1] * look[1] + unit[2] * look[2];
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}
