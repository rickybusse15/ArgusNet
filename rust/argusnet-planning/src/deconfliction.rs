//! Per-step separation enforcement: corridor windows, then pairwise yields.
//!
//! Runs after the controllers have proposed a position and velocity for each
//! platform. Corridor holds are applied first by zeroing a held platform's
//! velocity; pairwise separation then resolves close approaches with a lateral
//! offset, its mirror, or a vertical one.
//!
//! # Order decides the outcome three times over
//!
//! **The lexicographic tie-break yields the LARGER id.** When two platforms
//! share a role priority, `_yielder` returns `(a, b) if a > b else (b, a)` —
//! so `bravo` yields to `alpha`, not the other way round. A port that yielded
//! the smaller id would pass every case where the roles differ and quietly
//! swap the two whenever they do not.
//!
//! **The pairwise pass writes back in place.** It walks `sorted(ids)` and
//! stores each adjustment immediately, so a later pair sees an earlier pair's
//! result. Three platforms in a row do not resolve to what three independent
//! pairs would give.
//!
//! **Corridor membership follows the PROPOSED order, not sorted order**, and a
//! `held` set short-circuits later pairs — so which platform ends up holding
//! depends on the order the caller supplied. Reproduced with an order-
//! preserving input rather than a map.

use std::collections::BTreeSet;

pub type Vec3 = [f64; 3];

/// Role priorities. Anything not listed scores [`DEFAULT_ROLE_PRIORITY`].
pub const ROLE_PRIORITY: &[(&str, i64)] = &[
    ("primary_observer", 5),
    ("secondary_baseline", 4),
    ("corridor_watcher", 3),
    ("relay", 2),
    ("reserve", 1),
];

pub const DEFAULT_ROLE_PRIORITY: i64 = 0;

pub fn role_priority(role: &str) -> i64 {
    ROLE_PRIORITY
        .iter()
        .find(|(name, _)| *name == role)
        .map(|(_, value)| *value)
        .unwrap_or(DEFAULT_ROLE_PRIORITY)
}

/// Which of two platforms gives way. Returns `(yielder, peer)`.
///
/// Lower priority yields. On a tie the **larger** id yields — see the module
/// docs; this is the reference's rule and it is not the obvious one.
pub fn yielder<'a>(a_id: &'a str, a_role: &str, b_id: &'a str, b_role: &str) -> (&'a str, &'a str) {
    let (pa, pb) = (role_priority(a_role), role_priority(b_role));
    if pa < pb {
        return (a_id, b_id);
    }
    if pb < pa {
        return (b_id, a_id);
    }
    if a_id > b_id {
        (a_id, b_id)
    } else {
        (b_id, a_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeconflictionConfig {
    pub min_separation_m: f64,
    pub look_ahead_s: f64,
    pub vertical_separation_m: f64,
    pub min_agl_m: f64,
    pub corridor_window_default_s: f64,
    pub enabled: bool,
}

impl Default for DeconflictionConfig {
    fn default() -> Self {
        Self {
            min_separation_m: 16.0,
            look_ahead_s: 1.0,
            vertical_separation_m: 20.0,
            min_agl_m: 18.0,
            corridor_window_default_s: 5.0,
            enabled: true,
        }
    }
}

/// A bidirectional lane platforms are routed through.
#[derive(Debug, Clone, PartialEq)]
pub struct FlightCorridor {
    pub corridor_id: String,
    pub waypoints_xy_m: Vec<[f64; 2]>,
    /// Half-width either side of the centreline.
    pub width_m: f64,
    /// `"inbound"`, `"outbound"` or `"bidirectional"`. Only bidirectional
    /// corridors are window-managed; the others are left alone entirely.
    pub direction: String,
    /// Empty means any platform may use it.
    pub assigned_drone_ids: Vec<String>,
    pub active_window: Vec<f64>,
}

/// A circular no-fly area.
#[derive(Debug, Clone, PartialEq)]
pub struct ExclusionZone {
    pub zone_id: String,
    pub center: Vec3,
    pub radius_m: f64,
}

impl ExclusionZone {
    /// Inclusive at the rim, matching `MissionZone.contains_point`.
    pub fn contains(&self, xy: [f64; 2]) -> bool {
        let dx = xy[0] - self.center[0];
        let dy = xy[1] - self.center[1];
        dx * dx + dy * dy <= self.radius_m * self.radius_m
    }
}

/// How a conflict was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    CorridorHold,
    LateralOffset,
    VerticalOffset,
}

impl Resolution {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CorridorHold => "corridor_hold",
            Self::LateralOffset => "lateral_offset",
            Self::VerticalOffset => "vertical_offset",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeconflictionEvent {
    pub yielding_drone_id: String,
    pub conflicting_drone_id: String,
    pub predicted_separation_m: f64,
    pub resolution: Resolution,
    pub timestamp_s: f64,
}

/// One platform's proposed motion.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub drone_id: String,
    pub position: Vec3,
    pub velocity: Vec3,
}

/// `np.linalg.norm` of a 2-vector: `sqrt(x*x + y*y)`, **not** `hypot`.
///
/// `hypot` is the more careful function and it rounds differently: the
/// reference's predicted separation came out one ulp apart from this port's
/// on a real resolution (P4.9a.2). That value sizes the lateral push, so the
/// difference is not cosmetic -- it can move a platform.
fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// The nearest point on the corridor centreline, and the direction there.
struct CorridorFrame {
    direction_xy: [f64; 2],
    distance_m: f64,
}

fn corridor_frame(corridor: &FlightCorridor, xy: [f64; 2]) -> Option<CorridorFrame> {
    if corridor.waypoints_xy_m.len() < 2 {
        return None;
    }
    let mut best: Option<CorridorFrame> = None;
    for pair in corridor.waypoints_xy_m.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let segment = [end[0] - start[0], end[1] - start[1]];
        let length = norm2(segment);
        if length <= 1.0e-9 {
            continue;
        }
        let direction = [segment[0] / length, segment[1] / length];
        let along = dot2([xy[0] - start[0], xy[1] - start[1]], direction);
        let clamped = along.clamp(0.0, length);
        let closest = [
            start[0] + direction[0] * clamped,
            start[1] + direction[1] * clamped,
        ];
        let distance_m = norm2([xy[0] - closest[0], xy[1] - closest[1]]);
        // Strictly less: the FIRST nearest segment wins a tie.
        if best.as_ref().is_none_or(|b| distance_m < b.distance_m) {
            best = Some(CorridorFrame {
                direction_xy: direction,
                distance_m,
            });
        }
    }
    best
}

fn corridor_length_m(corridor: &FlightCorridor) -> f64 {
    corridor
        .waypoints_xy_m
        .windows(2)
        .map(|p| norm2([p[1][0] - p[0][0], p[1][1] - p[0][1]]))
        .sum()
}

/// A corridor with fewer than two window bounds is always active.
fn corridor_is_active(corridor: &FlightCorridor, timestamp_s: f64) -> bool {
    if corridor.active_window.len() < 2 {
        return true;
    }
    corridor.active_window[0] <= timestamp_s && timestamp_s <= corridor.active_window[1]
}

/// Which way the corridor is open, alternating every window.
fn open_direction_sign(corridor: &FlightCorridor, timestamp_s: f64, window_s: f64) -> i32 {
    let start_s = corridor.active_window.first().copied().unwrap_or(0.0);
    let elapsed = (timestamp_s - start_s).max(0.0);
    let index = (elapsed / window_s.max(1.0e-6)).floor() as i64;
    if index % 2 == 0 {
        1
    } else {
        -1
    }
}

struct Member {
    drone_id: String,
    sign: i32,
    speed_along: f64,
    position: Vec3,
}

/// Platforms inside the corridor and moving along it, in PROPOSED order.
fn corridor_members(corridor: &FlightCorridor, proposals: &[Proposal]) -> Vec<Member> {
    let mut members = Vec::new();
    for proposal in proposals {
        if !corridor.assigned_drone_ids.is_empty()
            && !corridor.assigned_drone_ids.contains(&proposal.drone_id)
        {
            continue;
        }
        let Some(frame) = corridor_frame(corridor, [proposal.position[0], proposal.position[1]])
        else {
            continue;
        };
        if frame.distance_m > corridor.width_m {
            continue;
        }
        let speed_along = dot2(
            [proposal.velocity[0], proposal.velocity[1]],
            frame.direction_xy,
        );
        if speed_along.abs() <= 1.0e-6 {
            continue;
        }
        members.push(Member {
            drone_id: proposal.drone_id.clone(),
            sign: if speed_along > 0.0 { 1 } else { -1 },
            speed_along: speed_along.abs(),
            position: proposal.position,
        });
    }
    members
}

fn window_duration_s(corridor: &FlightCorridor, members: &[Member], default_s: f64) -> f64 {
    let length_m = corridor_length_m(corridor);
    let speeds: Vec<f64> = members
        .iter()
        .map(|m| m.speed_along)
        .filter(|s| *s > 1.0e-6)
        .collect();
    if length_m > 0.0 && !speeds.is_empty() {
        let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
        (length_m / mean).max(1.0e-6)
    } else {
        default_s
    }
}

/// Minimum XY distance over `[0, look_ahead_s]`.
fn closest_approach_xy(
    a_pos: Vec3,
    a_vel: Vec3,
    b_pos: Vec3,
    b_vel: Vec3,
    look_ahead_s: f64,
) -> f64 {
    let r0 = [a_pos[0] - b_pos[0], a_pos[1] - b_pos[1]];
    let vr = [a_vel[0] - b_vel[0], a_vel[1] - b_vel[1]];
    let speed_sq = dot2(vr, vr);
    let t_star = if speed_sq <= 1e-12 {
        0.0
    } else {
        (-dot2(r0, vr) / speed_sq).clamp(0.0, look_ahead_s)
    };
    norm2([r0[0] + vr[0] * t_star, r0[1] + vr[1] * t_star])
}

/// A perpendicular-to-heading offset directed away from the peer.
///
/// With no heading to work from, the direction away from the peer is used
/// instead; with no separation either — the platforms exactly coincident — it
/// falls back to `(1, 0)`.
fn lateral_push(y_pos: Vec3, y_vel: Vec3, peer_pos: Vec3, min_sep: f64, pred_sep: f64) -> [f64; 2] {
    let margin = 1.0;
    let push_mag = (min_sep - pred_sep + margin).max(margin);

    let speed_xy = norm2([y_vel[0], y_vel[1]]);
    let mut perp = if speed_xy > 1e-6 {
        let heading = [y_vel[0] / speed_xy, y_vel[1] / speed_xy];
        [-heading[1], heading[0]]
    } else {
        let away = [y_pos[0] - peer_pos[0], y_pos[1] - peer_pos[1]];
        let n = norm2(away);
        if n > 1e-6 {
            [away[0] / n, away[1] / n]
        } else {
            [1.0, 0.0]
        }
    };
    let peer_dir = [peer_pos[0] - y_pos[0], peer_pos[1] - y_pos[1]];
    if dot2(perp, peer_dir) > 0.0 {
        perp = [-perp[0], -perp[1]];
    }
    [perp[0] * push_mag, perp[1] * push_mag]
}

/// Resolves a step's proposals.
///
/// `role_of` maps a platform id to its role; `terrain_floor` returns the
/// lowest altitude a platform may hold at that XY, or `None` when there is no
/// terrain to clamp against.
pub fn resolve_step<R, T>(
    config: &DeconflictionConfig,
    proposals: &[Proposal],
    timestamp_s: f64,
    mut role_of: R,
    mut terrain_floor: T,
    zones: &[ExclusionZone],
    corridors: &[FlightCorridor],
) -> (Vec<Proposal>, Vec<DeconflictionEvent>)
where
    R: FnMut(&str) -> String,
    T: FnMut([f64; 2], f64, f64) -> Option<f64>,
{
    let mut adjusted: Vec<Proposal> = proposals.to_vec();
    let mut events: Vec<DeconflictionEvent> = Vec::new();
    if !config.enabled || adjusted.len() < 2 {
        return (adjusted, events);
    }

    let index_of = |adjusted: &[Proposal], id: &str| {
        adjusted
            .iter()
            .position(|p| p.drone_id == id)
            .expect("proposal id")
    };
    let in_zone = |xy: [f64; 2]| zones.iter().any(|z| z.contains(xy));

    // --- corridor windows ---------------------------------------------------
    for corridor in corridors {
        if corridor.direction != "bidirectional" || !corridor_is_active(corridor, timestamp_s) {
            continue;
        }
        let members = corridor_members(corridor, &adjusted);
        if members.len() < 2 {
            continue;
        }
        let window_s = window_duration_s(corridor, &members, config.corridor_window_default_s);
        let open_sign = open_direction_sign(corridor, timestamp_s, window_s);

        // `held` is per corridor, and it short-circuits later pairs — so the
        // proposed order decides who ends up holding.
        let mut held: BTreeSet<String> = BTreeSet::new();
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let (a, b) = (&members[i], &members[j]);
                if held.contains(&a.drone_id) || held.contains(&b.drone_id) || a.sign == b.sign {
                    continue;
                }
                let (holding, peer) = if a.sign != open_sign && b.sign == open_sign {
                    (a.drone_id.clone(), b.drone_id.clone())
                } else if b.sign != open_sign && a.sign == open_sign {
                    (b.drone_id.clone(), a.drone_id.clone())
                } else {
                    let (y, p) = yielder(
                        &a.drone_id,
                        &role_of(&a.drone_id),
                        &b.drone_id,
                        &role_of(&b.drone_id),
                    );
                    (y.to_string(), p.to_string())
                };
                held.insert(holding.clone());
                let separation =
                    norm2([a.position[0] - b.position[0], a.position[1] - b.position[1]]);
                let index = index_of(&adjusted, &holding);
                adjusted[index].velocity = [0.0, 0.0, 0.0];
                events.push(DeconflictionEvent {
                    yielding_drone_id: holding,
                    conflicting_drone_id: peer,
                    predicted_separation_m: separation,
                    resolution: Resolution::CorridorHold,
                    timestamp_s,
                });
            }
        }
    }

    // --- pairwise separation ------------------------------------------------
    let mut ids: Vec<String> = adjusted.iter().map(|p| p.drone_id.clone()).collect();
    ids.sort();
    let look = config.look_ahead_s;
    let min_sep = config.min_separation_m;
    let vert_sep = config.vertical_separation_m;

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let (a_id, b_id) = (ids[i].clone(), ids[j].clone());
            // Re-read every time: an earlier pair may have moved one of these.
            let a = adjusted[index_of(&adjusted, &a_id)].clone();
            let b = adjusted[index_of(&adjusted, &b_id)].clone();

            let pred_sep =
                closest_approach_xy(a.position, a.velocity, b.position, b.velocity, look);
            if pred_sep >= min_sep {
                continue;
            }

            let (yielder_id, peer_id) = yielder(&a_id, &role_of(&a_id), &b_id, &role_of(&b_id));
            let (yielder_id, peer_id) = (yielder_id.to_string(), peer_id.to_string());
            let y = adjusted[index_of(&adjusted, &yielder_id)].clone();
            let peer_pos = adjusted[index_of(&adjusted, &peer_id)].position;

            let push = lateral_push(y.position, y.velocity, peer_pos, min_sep, pred_sep);
            let lateral = [
                y.position[0] + push[0],
                y.position[1] + push[1],
                y.position[2],
            ];
            // Note the comparison is against the peer's CURRENT position, not
            // its predicted one, while the yielder's is projected forward.
            let lateral_pred = [
                lateral[0] + y.velocity[0] * look,
                lateral[1] + y.velocity[1] * look,
            ];
            let still_close =
                norm2([lateral_pred[0] - peer_pos[0], lateral_pred[1] - peer_pos[1]]) < min_sep;

            let (mut new_pos, resolution) = if !still_close && !in_zone([lateral[0], lateral[1]]) {
                (lateral, Resolution::LateralOffset)
            } else {
                let opposite = [
                    y.position[0] - push[0],
                    y.position[1] - push[1],
                    y.position[2],
                ];
                let opposite_pred = [
                    opposite[0] + y.velocity[0] * look,
                    opposite[1] + y.velocity[1] * look,
                ];
                let opposite_close = norm2([
                    opposite_pred[0] - peer_pos[0],
                    opposite_pred[1] - peer_pos[1],
                ]) < min_sep;
                if !opposite_close && !in_zone([opposite[0], opposite[1]]) {
                    (opposite, Resolution::LateralOffset)
                } else {
                    // Inclusive: level platforms both climb.
                    let direction = if y.position[2] >= peer_pos[2] {
                        1.0
                    } else {
                        -1.0
                    };
                    (
                        [
                            y.position[0],
                            y.position[1],
                            y.position[2] + direction * vert_sep,
                        ],
                        Resolution::VerticalOffset,
                    )
                }
            };

            if let Some(floor) =
                terrain_floor([new_pos[0], new_pos[1]], new_pos[2], config.min_agl_m)
            {
                new_pos[2] = floor;
            }

            let index = index_of(&adjusted, &yielder_id);
            adjusted[index].position = new_pos;
            events.push(DeconflictionEvent {
                yielding_drone_id: yielder_id,
                conflicting_drone_id: peer_id,
                predicted_separation_m: pred_sep,
                resolution,
                timestamp_s,
            });
        }
    }

    (adjusted, events)
}
