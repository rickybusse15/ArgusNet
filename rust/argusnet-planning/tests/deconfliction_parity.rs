//! Parity for `DeconflictionLayer.resolve_step`.
//!
//! Every case compares the full adjusted state AND the event list in order.
//! Events are ordered by the sorted-id pairwise walk, and comparing them as a
//! set would miss a port that resolved the right conflicts in the wrong
//! sequence — which matters, because the pass writes back in place and a later
//! pair sees an earlier pair's adjustment.

use argusnet_planning::deconfliction::{
    resolve_step, role_priority, yielder, DeconflictionConfig, ExclusionZone, FlightCorridor,
    Proposal, DEFAULT_ROLE_PRIORITY,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    role_priority: BTreeMap<String, i64>,
    default_role_priority: i64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ConfigSpec {
    min_separation_m: String,
    look_ahead_s: String,
    vertical_separation_m: String,
    min_agl_m: String,
    corridor_window_default_s: String,
    enabled: bool,
}

#[derive(Deserialize)]
struct ZoneSpec {
    zone_id: String,
    zone_type: String,
    center: [String; 3],
    radius_m: String,
}

#[derive(Deserialize)]
struct CorridorSpec {
    corridor_id: String,
    waypoints_xy_m: Vec<Vec<String>>,
    width_m: String,
    direction: String,
    assigned_drone_ids: Vec<String>,
    active_window: Vec<String>,
}

#[derive(Deserialize)]
struct MotionSpec {
    drone_id: String,
    position: [String; 3],
    velocity: [String; 3],
}

#[derive(Deserialize)]
struct EventSpec {
    yielding_drone_id: String,
    conflicting_drone_id: String,
    predicted_separation_m: String,
    resolution: String,
    timestamp_s: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    timestamp_s: String,
    config: ConfigSpec,
    terrain_height_m: Option<String>,
    zones: Vec<ZoneSpec>,
    corridors: Vec<CorridorSpec>,
    roles: BTreeMap<String, String>,
    proposed: Vec<MotionSpec>,
    adjusted: Vec<MotionSpec>,
    events: Vec<EventSpec>,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn triple(values: &[String; 3]) -> [f64; 3] {
    [f(&values[0]), f(&values[1]), f(&values[2])]
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/deconfliction.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1.0e-9 * scale,
        "{what}: {actual:?} != {expected:?}"
    );
}

fn run(
    case: &Case,
) -> (
    Vec<Proposal>,
    Vec<argusnet_planning::deconfliction::DeconflictionEvent>,
) {
    let config = DeconflictionConfig {
        min_separation_m: f(&case.config.min_separation_m),
        look_ahead_s: f(&case.config.look_ahead_s),
        vertical_separation_m: f(&case.config.vertical_separation_m),
        min_agl_m: f(&case.config.min_agl_m),
        corridor_window_default_s: f(&case.config.corridor_window_default_s),
        enabled: case.config.enabled,
    };
    // Proposed ORDER is preserved: corridor membership reads it.
    let proposals: Vec<Proposal> = case
        .proposed
        .iter()
        .map(|m| Proposal {
            drone_id: m.drone_id.clone(),
            position: triple(&m.position),
            velocity: triple(&m.velocity),
        })
        .collect();
    let zones: Vec<ExclusionZone> = case
        .zones
        .iter()
        .map(|z| {
            assert_eq!(z.zone_type, "exclusion", "only exclusion zones are kept");
            ExclusionZone {
                zone_id: z.zone_id.clone(),
                center: triple(&z.center),
                radius_m: f(&z.radius_m),
            }
        })
        .collect();
    let corridors: Vec<FlightCorridor> = case
        .corridors
        .iter()
        .map(|c| FlightCorridor {
            corridor_id: c.corridor_id.clone(),
            waypoints_xy_m: c
                .waypoints_xy_m
                .iter()
                .map(|p| [f(&p[0]), f(&p[1])])
                .collect(),
            width_m: f(&c.width_m),
            direction: c.direction.clone(),
            assigned_drone_ids: c.assigned_drone_ids.clone(),
            active_window: c.active_window.iter().map(|v| f(v)).collect(),
        })
        .collect();
    let terrain = case.terrain_height_m.as_ref().map(|h| f(h));

    resolve_step(
        &config,
        &proposals,
        f(&case.timestamp_s),
        |id| case.roles.get(id).cloned().unwrap_or_default(),
        |_xy, z, min_agl| terrain.map(|height| z.max(height + min_agl)),
        &zones,
        &corridors,
    )
}

#[test]
fn the_role_table_matches_python() {
    let fixture = load();
    assert_eq!(fixture.default_role_priority, DEFAULT_ROLE_PRIORITY);
    for (role, expected) in &fixture.role_priority {
        assert_eq!(role_priority(role), *expected, "priority of {role}");
    }
    assert_eq!(
        role_priority("not_a_role"),
        DEFAULT_ROLE_PRIORITY,
        "an unknown role takes the default"
    );
}

#[test]
fn every_case_matches_python() {
    for case in load().cases.iter() {
        let (adjusted, events) = run(case);
        assert_eq!(
            adjusted.len(),
            case.adjusted.len(),
            "{}: platform count",
            case.name
        );
        for (actual, expected) in adjusted.iter().zip(&case.adjusted) {
            // Positional: the adjusted list keeps the proposed order.
            assert_eq!(actual.drone_id, expected.drone_id, "{}: order", case.name);
            let e_pos = triple(&expected.position);
            let e_vel = triple(&expected.velocity);
            for axis in 0..3 {
                assert_close(
                    actual.position[axis],
                    e_pos[axis],
                    &format!("{}/{} position[{axis}]", case.name, actual.drone_id),
                );
                assert_close(
                    actual.velocity[axis],
                    e_vel[axis],
                    &format!("{}/{} velocity[{axis}]", case.name, actual.drone_id),
                );
            }
        }
        assert_eq!(
            events.len(),
            case.events.len(),
            "{}: event count ({:?} vs expected)",
            case.name,
            events
                .iter()
                .map(|e| e.resolution.as_str())
                .collect::<Vec<_>>()
        );
        // Compared IN ORDER: events follow the sorted-id pairwise walk, and a
        // set comparison would accept the right conflicts resolved in the
        // wrong sequence.
        for (index, (actual, expected)) in events.iter().zip(&case.events).enumerate() {
            let what = format!("{}/event[{index}]", case.name);
            assert_eq!(
                actual.yielding_drone_id, expected.yielding_drone_id,
                "{what}: yielder"
            );
            assert_eq!(
                actual.conflicting_drone_id, expected.conflicting_drone_id,
                "{what}: peer"
            );
            assert_eq!(
                actual.resolution.as_str(),
                expected.resolution,
                "{what}: resolution"
            );
            assert_close(
                actual.predicted_separation_m,
                f(&expected.predicted_separation_m),
                &format!("{what}: separation"),
            );
            assert_close(
                actual.timestamp_s,
                f(&expected.timestamp_s),
                &format!("{what}: timestamp"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

/// The tie-break yields the LARGER id — the opposite of the obvious guess.
#[test]
fn a_priority_tie_is_broken_by_yielding_the_larger_id() {
    assert_eq!(
        yielder("alpha", "relay", "bravo", "relay"),
        ("bravo", "alpha")
    );
    assert_eq!(
        yielder("bravo", "relay", "alpha", "relay"),
        ("bravo", "alpha")
    );
    // Unknown roles both score the default, so they tie the same way.
    assert_eq!(yielder("alpha", "nope", "bravo", "nah"), ("bravo", "alpha"));
    // And priority still beats the id when it differs.
    assert_eq!(
        yielder("alpha", "reserve", "bravo", "primary_observer"),
        ("alpha", "bravo")
    );

    let fixture = load();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.name == "equal_roles_larger_id_yields")
        .expect("the tie case must exist");
    assert_eq!(
        case.events[0].yielding_drone_id, "bravo",
        "the larger id yields"
    );
}

/// Every resolution must appear, and a clean case with no events must too.
#[test]
fn the_fixture_reaches_every_resolution() {
    let fixture = load();
    let mut seen: Vec<&str> = fixture
        .cases
        .iter()
        .flat_map(|c| &c.events)
        .map(|e| e.resolution.as_str())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    for expected in ["corridor_hold", "lateral_offset", "vertical_offset"] {
        assert!(seen.contains(&expected), "no case produces {expected}");
    }
    assert!(
        fixture.cases.iter().any(|c| c.events.is_empty()),
        "no case resolves cleanly"
    );
    assert!(
        fixture.cases.iter().any(|c| c.events.len() > 1),
        "no case emits more than one event, so the event order is unobservable"
    );
}

/// The pairwise pass writes back in place, so a later pair sees an earlier
/// pair's adjustment. Three platforms in a row must not resolve to what three
/// independent pairs would give.
#[test]
fn the_pairwise_pass_is_sequential_not_independent() {
    let fixture = load();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.name == "three_in_a_row_sequential")
        .expect("the three-platform case must exist");
    assert_eq!(case.events.len(), 2, "two conflicts must resolve");
    // The second event's yielder was already moved by the first.
    assert_eq!(case.events[0].yielding_drone_id, "b");
    assert_eq!(case.events[1].conflicting_drone_id, "b");
    // And `b`'s recorded separation in the second event reflects its NEW
    // position, not the proposed one.
    let (_, events) = run(case);
    assert_close(
        events[1].predicted_separation_m,
        f(&case.events[1].predicted_separation_m),
        "the second pair sees the first pair's adjustment",
    );
}

/// The corridor's open direction alternates, so the same head-on pair yields
/// a different holder one window later.
#[test]
fn the_corridor_window_alternates() {
    let fixture = load();
    let first = fixture
        .cases
        .iter()
        .find(|c| c.name == "corridor_head_on_hold")
        .expect("case");
    let later = fixture
        .cases
        .iter()
        .find(|c| c.name == "corridor_window_flips")
        .expect("case");
    assert_eq!(first.proposed.len(), later.proposed.len());
    assert_ne!(
        first.events[0].yielding_drone_id, later.events[0].yielding_drone_id,
        "the same platform holds in both windows, so the alternation is unobservable"
    );
    assert_ne!(f(&first.timestamp_s), f(&later.timestamp_s));
}

/// The opposite-side flip must actually fire, and land on the mirror rather
/// than falling through to a vertical offset.
///
/// This takes deliberate geometry: the first push already points away from the
/// peer, so its mirror points toward it and is usually still too close. An
/// earlier version of this case produced a vertical offset and never reached
/// the branch at all.
#[test]
fn a_blocked_lateral_flips_to_the_other_side() {
    let fixture = load();
    let blocked = fixture
        .cases
        .iter()
        .find(|c| c.name == "lateral_blocked_flips_side")
        .expect("case");
    let clear = fixture
        .cases
        .iter()
        .find(|c| c.name == "lateral_first_side_when_clear")
        .expect("case");
    assert_eq!(blocked.events[0].resolution, "lateral_offset");
    assert_eq!(clear.events[0].resolution, "lateral_offset");
    let blocked_x = f(&blocked.adjusted[1].position[0]);
    let clear_x = f(&clear.adjusted[1].position[0]);
    assert_eq!(
        blocked_x, -clear_x,
        "the blocked side must mirror the clear one"
    );
    assert!(clear_x != 0.0, "the push must be non-zero");
}

/// Corridors that are one-way, inactive, or too far off the centreline are
/// left entirely alone, and same-direction traffic is not a conflict.
#[test]
fn the_fixture_covers_every_reason_a_corridor_does_nothing() {
    let fixture = load();
    for name in [
        "corridor_same_direction_ignored",
        "corridor_one_way_ignored",
        "corridor_outside_active_window",
        "corridor_too_far_off_centreline",
    ] {
        let case = fixture
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert!(
            case.events.is_empty(),
            "{name}: the corridor must produce no hold"
        );
        assert!(
            !case.corridors.is_empty(),
            "{name}: the case must actually have a corridor"
        );
    }
}

/// Terrain clamps a vertical yield back up. Without the pair, a port that
/// ignored terrain entirely would pass.
#[test]
fn terrain_clamps_a_downward_yield() {
    let fixture = load();
    let unclamped = fixture
        .cases
        .iter()
        .find(|c| c.name == "vertical_offset_down")
        .expect("case");
    let clamped = fixture
        .cases
        .iter()
        .find(|c| c.name == "vertical_offset_clamped_by_terrain")
        .expect("case");
    assert!(clamped.terrain_height_m.is_some());
    assert!(unclamped.terrain_height_m.is_none());
    let low = f(&unclamped.adjusted[1].position[2]);
    let high = f(&clamped.adjusted[1].position[2]);
    assert!(
        high > low,
        "terrain must raise the yield: {high} is not above {low}"
    );
}

/// **Four equivalent mutants, recorded rather than tested around.**
///
/// 1. **`t_star` for a stationary pair.** When the relative velocity is zero
///    the separation is `r0 + vr * t` for any `t`, so whether `t_star` is 0 or
///    the full look-ahead makes no difference.
/// 2. **The `max(margin)` floor on the push magnitude.** The code only reaches
///    it when `pred_sep < min_sep`, so `min_sep - pred_sep` is already
///    positive and `+ 1.0` puts the magnitude above the floor. It can never
///    bind.
/// 3. **The direction of the stationary fallback.** With no heading, the push
///    starts from `y - peer`; starting from `peer - y` instead is corrected by
///    the very next step, which flips any perpendicular pointing at the peer.
///    The two arrive at the same vector.
/// 4. **The `len() < 2` early return.** With one platform the pairwise loop
///    makes no pairs and a corridor needs two members, so the function
///    already does nothing.
#[test]
fn four_recorded_equivalent_mutants_in_the_deconfliction_layer() {
    // 1. Zero relative velocity: separation is independent of t.
    let r0 = [30.0_f64, -40.0];
    for t in [0.0_f64, 0.5, 1.0, 100.0] {
        let separation = (r0[0] + 0.0 * t).hypot(r0[1] + 0.0 * t);
        assert_eq!(
            separation, 50.0,
            "t = {t} must not change a stationary pair"
        );
    }

    // 2. The floor cannot bind below the minimum separation.
    let margin = 1.0_f64;
    for min_sep in [1.0_f64, 16.0, 100.0] {
        for fraction in [0.0_f64, 0.5, 0.999] {
            let pred_sep = min_sep * fraction;
            let raw = min_sep - pred_sep + margin;
            assert!(
                raw >= margin,
                "min_sep {min_sep}, pred_sep {pred_sep}: raw {raw} fell below the floor"
            );
        }
    }

    // 3. Either starting direction ends up the same after the flip-away step.
    let flip_away = |mut perp: [f64; 2], peer_dir: [f64; 2]| {
        if perp[0] * peer_dir[0] + perp[1] * peer_dir[1] > 0.0 {
            perp = [-perp[0], -perp[1]];
        }
        perp
    };
    for away in [[3.0_f64, 4.0], [-1.0, 0.0], [0.0, -7.0]] {
        let n = away[0].hypot(away[1]);
        let peer_dir = [-away[0], -away[1]];
        let from_away = flip_away([away[0] / n, away[1] / n], peer_dir);
        let from_toward = flip_away([-away[0] / n, -away[1] / n], peer_dir);
        assert_eq!(
            from_away, from_toward,
            "the flip-away step erases the choice"
        );
    }

    // 4. A single platform is a no-op with or without the guard.
    let config = DeconflictionConfig::default();
    let single = vec![Proposal {
        drone_id: "solo".into(),
        position: [0.0, 0.0, 100.0],
        velocity: [1.0, 0.0, 0.0],
    }];
    let (adjusted, events) = resolve_step(
        &config,
        &single,
        0.0,
        |_| "relay".to_string(),
        |_, _, _| None,
        &[],
        &[],
    );
    assert_eq!(adjusted, single);
    assert!(events.is_empty());
}

/// Supplying ids out of sorted order must not change the result — the
/// pairwise walk sorts them. Every earlier case happened to be already
/// sorted, so this was indistinguishable from walking the input order.
#[test]
fn the_pairwise_walk_sorts_the_ids() {
    let fixture = load();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.name == "proposed_order_is_not_sorted_order")
        .expect("case");
    let supplied: Vec<&str> = case.proposed.iter().map(|m| m.drone_id.as_str()).collect();
    let mut sorted = supplied.clone();
    sorted.sort_unstable();
    assert_ne!(
        supplied, sorted,
        "the case must supply ids out of order, or it proves nothing"
    );
    // The first event's yielder is decided by the sorted walk, not the input.
    assert!(!case.events.is_empty(), "the case must produce a conflict");
}

/// The edges the earlier cases do not reach, each closing a survivor: the
/// separation guard exactly at the minimum, a peer on the left of the
/// heading, a zone on the mirror side, a single-element active window, a
/// platform that would be held twice, a member exactly at the corridor's
/// half-width, crossing traffic with no along-axis speed, and a window
/// derived from corridor length rather than the default.
#[test]
fn the_fixture_reaches_the_boundary_cases() {
    let fixture = load();
    let by_name = |name: &str| {
        fixture
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("missing case {name}"))
    };

    // `>=` means exactly the minimum is NOT a conflict.
    let at_minimum = by_name("separation_exactly_at_the_minimum");
    assert!(
        at_minimum.events.is_empty(),
        "separation exactly at the minimum must not trigger a yield"
    );
    let separation =
        f(&at_minimum.proposed[1].position[0]) - f(&at_minimum.proposed[0].position[0]);
    assert_eq!(separation, f(&at_minimum.config.min_separation_m));

    // Exactly at the corridor half-width IS a member.
    let at_width = by_name("corridor_member_exactly_at_the_width");
    assert!(
        !at_width.events.is_empty(),
        "a platform exactly at the half-width is inside the corridor"
    );
    assert_eq!(
        f(&at_width.proposed[1].position[1]),
        f(&at_width.corridors[0].width_m)
    );

    // Crossing traffic has no along-axis speed and is not a member.
    let crossing = by_name("corridor_crossing_traffic_is_not_a_member");
    assert!(
        crossing.events.is_empty(),
        "crossing traffic is not corridor traffic"
    );

    // A one-element window means always active.
    let short_window = by_name("corridor_with_a_short_window_is_active");
    assert_eq!(short_window.corridors[0].active_window.len(), 1);
    assert!(
        !short_window.events.is_empty(),
        "a short window is active, not inactive"
    );

    // The held set stops a second hold on the same platform.
    let twice = by_name("held_set_prevents_a_second_hold");
    assert_eq!(
        twice.events.len(),
        1,
        "the held set must short-circuit the second pair"
    );

    // A zone on the mirror side forces the vertical fallback.
    let both_blocked = by_name("opposite_side_blocked_by_a_zone");
    assert_eq!(
        both_blocked.events[0].resolution, "vertical_offset",
        "with both sides in a zone the layer must climb"
    );
    assert_eq!(both_blocked.zones.len(), 2);

    // The flip-away step, which is harder to make observable than it looks.
    // Dropping it is usually absorbed: the un-flipped push heads toward the
    // peer, the still-close check rejects it, and the opposite-side fallback
    // lands on exactly the side the flip would have chosen. It only shows
    // when the toward-the-peer side is ALSO viable.
    let left = by_name("peer_on_the_left_forces_the_flip_away");
    assert!(!left.events.is_empty());
    let flip = by_name("flip_away_when_both_sides_are_viable");
    assert_eq!(flip.events[0].resolution, "lateral_offset");
    let yielded_x = f(&flip.adjusted[1].position[0]);
    let peer_x = f(&flip.proposed[0].position[0]);
    assert!(
        yielded_x < 0.0 && peer_x > 0.0,
        "the yielder moved to x={yielded_x} with the peer at x={peer_x}; the \
         push must go AWAY from the peer"
    );
}
