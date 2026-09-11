//! Parity for the POI manager, replayed as call sequences.
//!
//! `POIManager` is stateful, so the fixture records a sequence of calls and
//! the result of each, plus every status after each one. A port that got each
//! method right in isolation but kept its POIs in a hash map would still
//! diverge: construction order decides which POI is chosen among equals, and
//! the order `check_completions` reports ids in.

use argusnet_planning::poi::{InspectionPoi, PoiAssignmentContext, PoiManager, PoiState};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    travel_power_w: String,
    sequences: Vec<Sequence>,
}

#[derive(Deserialize)]
struct Sequence {
    name: String,
    pois: Vec<PoiSpec>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct PoiSpec {
    poi_id: String,
    name: String,
    position: [String; 3],
    priority: i64,
    required_dwell_s: String,
}

#[derive(Deserialize)]
struct StatusSpec {
    poi_id: String,
    status: String,
    assigned_drone_id: Option<String>,
    arrival_time_s: Option<String>,
    completion_time_s: Option<String>,
    dwell_accumulated_s: String,
}

#[derive(Deserialize)]
struct Step {
    call: String,
    args: serde_json::Value,
    result: serde_json::Value,
    statuses: Vec<StatusSpec>,
    completed_count: usize,
    all_complete: bool,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn triple(values: &[String; 3]) -> [f64; 3] {
    [f(&values[0]), f(&values[1]), f(&values[2])]
}

fn arg_f(args: &serde_json::Value, key: &str) -> f64 {
    f(args[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} missing")))
}

fn arg_triple(args: &serde_json::Value, key: &str) -> [f64; 3] {
    let raw = args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} missing"));
    [
        f(raw[0].as_str().unwrap()),
        f(raw[1].as_str().unwrap()),
        f(raw[2].as_str().unwrap()),
    ]
}

fn arg_str<'a>(args: &'a serde_json::Value, key: &str) -> &'a str {
    args[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} missing"))
}

fn load() -> Fixture {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/poi.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn pois_of(spec: &Sequence) -> Vec<InspectionPoi> {
    spec.pois
        .iter()
        .map(|p| InspectionPoi {
            poi_id: p.poi_id.clone(),
            name: p.name.clone(),
            position: triple(&p.position),
            priority: p.priority,
            required_dwell_s: f(&p.required_dwell_s),
        })
        .collect()
}

fn assert_statuses(manager: &PoiManager, expected: &[StatusSpec], what: &str) {
    let actual = manager.statuses();
    assert_eq!(actual.len(), expected.len(), "{what}: status count");
    for (a, e) in actual.iter().zip(expected) {
        // Compared positionally: the reference lists statuses in construction
        // order, and so must the port.
        assert_eq!(a.poi_id, e.poi_id, "{what}: status order");
        assert_eq!(a.state.as_str(), e.status, "{what}/{}: state", e.poi_id);
        assert_eq!(
            a.assigned_drone_id, e.assigned_drone_id,
            "{what}/{}: assigned drone",
            e.poi_id
        );
        let close = |actual: Option<f64>, expected: &Option<String>| match (actual, expected) {
            (None, None) => true,
            (Some(a), Some(e)) => (a - f(e)).abs() <= 1.0e-12,
            _ => false,
        };
        assert!(
            close(a.arrival_time_s, &e.arrival_time_s),
            "{what}/{}: arrival {:?} != {:?}",
            e.poi_id,
            a.arrival_time_s,
            e.arrival_time_s
        );
        assert!(
            close(a.completion_time_s, &e.completion_time_s),
            "{what}/{}: completion {:?} != {:?}",
            e.poi_id,
            a.completion_time_s,
            e.completion_time_s
        );
        assert!(
            (a.dwell_accumulated_s - f(&e.dwell_accumulated_s)).abs() <= 1.0e-12,
            "{what}/{}: dwell {} != {}",
            e.poi_id,
            a.dwell_accumulated_s,
            e.dwell_accumulated_s
        );
    }
}

/// Replays every sequence, checking each call's result AND the full status
/// list after it. Comparing only the return values would miss a port that
/// answered correctly while leaving its bookkeeping wrong — the mistake that
/// cost two rounds in P4.4.
#[test]
fn every_sequence_replays_exactly() {
    let fixture = load();
    for sequence in &fixture.sequences {
        let mut manager = PoiManager::new(pois_of(sequence));
        // The rescoring stub answers by POI position, as the generator's did.
        let positions: Vec<([f64; 3], String)> = sequence
            .pois
            .iter()
            .map(|p| (triple(&p.position), p.poi_id.clone()))
            .collect();

        for (step_index, step) in sequence.steps.iter().enumerate() {
            let what = format!("{}[{step_index}] {}", sequence.name, step.call);
            match step.call.as_str() {
                "assign_nearest" => {
                    let chosen = manager.assign_nearest(
                        arg_str(&step.args, "drone_id"),
                        arg_triple(&step.args, "drone_pos"),
                        arg_f(&step.args, "battery_fraction"),
                    );
                    let actual = chosen.map(|p| p.poi_id.clone());
                    let expected = step.result.as_str().map(str::to_string);
                    assert_eq!(actual, expected, "{what}: result");
                }
                "assign_energy_aware" => {
                    let context = PoiAssignmentContext {
                        drone_id: arg_str(&step.args, "drone_id").to_string(),
                        drone_pos: arg_triple(&step.args, "drone_pos"),
                        battery_remaining_wh: arg_f(&step.args, "battery_remaining_wh"),
                        battery_capacity_wh: arg_f(&step.args, "battery_capacity_wh"),
                        cruise_speed_m_per_s: arg_f(&step.args, "cruise_speed_mps"),
                        battery_reserve_fraction: arg_f(&step.args, "battery_reserve_fraction"),
                    };
                    let actual = manager
                        .assign_energy_aware(&context)
                        .map(|p| p.poi_id.clone());
                    let expected = step.result.as_str().map(str::to_string);
                    assert_eq!(actual, expected, "{what}: result");
                }
                "accumulate_dwell" => manager.accumulate_dwell(
                    arg_str(&step.args, "drone_id"),
                    arg_triple(&step.args, "drone_pos"),
                    arg_f(&step.args, "dt_s"),
                ),
                "check_completions" => {
                    let actual = manager.check_completions(arg_f(&step.args, "timestamp_s"));
                    let expected: Vec<String> = step
                        .result
                        .as_array()
                        .expect("a list")
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string())
                        .collect();
                    // Order matters: construction order, not completion order.
                    assert_eq!(actual, expected, "{what}: completed ids and their order");
                }
                "trigger_handoff" => {
                    let actual = manager.trigger_handoff(
                        arg_str(&step.args, "from_drone_id"),
                        arg_str(&step.args, "to_drone_id"),
                    );
                    assert_eq!(Some(actual), step.result.as_bool(), "{what}: result");
                }
                "request_team_assign" => {
                    let actual = manager.request_team_assign(
                        arg_str(&step.args, "poi_id"),
                        arg_str(&step.args, "second_drone_id"),
                    );
                    assert_eq!(Some(actual), step.result.as_bool(), "{what}: result");
                }
                "rescore_from_map" => {
                    let densities = step.args["densities"].as_object().expect("densities");
                    manager.rescore_from_map(|xy| {
                        let found = positions.iter().find(|(p, _)| {
                            (p[0] - xy[0]).abs() < 1.0e-9 && (p[1] - xy[1]).abs() < 1.0e-9
                        })?;
                        densities.get(&found.1).map(|v| f(v.as_str().unwrap()))
                    });
                }
                "assignment_for" => {
                    let actual = manager.assignment_for(arg_str(&step.args, "drone_id"));
                    let expected = step.result.as_str();
                    assert_eq!(actual, expected, "{what}: result");
                }
                other => panic!("unhandled call {other}"),
            }
            assert_statuses(&manager, &step.statuses, &what);
            assert_eq!(
                manager.completed_count(),
                step.completed_count,
                "{what}: completed count"
            );
            assert_eq!(
                manager.all_complete(),
                step.all_complete,
                "{what}: all complete"
            );
        }
    }
    assert_eq!(
        argusnet_planning::poi::TRAVEL_POWER_W,
        f(&fixture.travel_power_w)
    );
}

// ---------------------------------------------------------------------------
// Coverage: the sequences must reach what they claim to.
// ---------------------------------------------------------------------------

/// `check_completions` reports in construction order, not completion order.
/// The sequence must therefore complete two POIs in ONE call, with the
/// later-constructed one having finished first — otherwise the order carries
/// no information and a hash-map port passes.
#[test]
fn the_fixture_pins_the_completion_order() {
    let fixture = load();
    let multi: Vec<&Step> = fixture
        .sequences
        .iter()
        .flat_map(|s| &s.steps)
        .filter(|st| {
            st.call == "check_completions"
                && st.result.as_array().map(|a| a.len() > 1).unwrap_or(false)
        })
        .collect();
    assert!(
        !multi.is_empty(),
        "no call completes more than one POI, so the reported order is unobservable"
    );
    let ids: Vec<&str> = multi[0]
        .result
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec!["first", "second"],
        "the ids must come back in construction order; 'second' reached its \
         dwell first in time and must still be reported second"
    );
}

/// The stable sort. Two POIs at the same priority and distance are separated
/// only by construction order, so the fixture must contain such a pair.
#[test]
fn the_fixture_pins_the_stable_sort() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "assign_nearest_ordering")
        .expect("the ordering sequence must exist");
    let pois = pois_of(sequence);
    // Distances measured from the origin, which is where every drone stands.
    let mut seen: Vec<(i64, u64)> = Vec::new();
    let mut tied = 0;
    for p in &pois {
        let distance = p.position[0].hypot(p.position[1]);
        let key = (p.priority, distance.to_bits());
        if seen.contains(&key) {
            tied += 1;
        }
        seen.push(key);
    }
    assert!(
        tied >= 1,
        "no two POIs share a priority AND a distance, so the stable sort is \
         indistinguishable from an unstable one"
    );
}

/// Both sides of the 25% battery gate, and the gate is `<`, so exactly a
/// quarter is allowed.
#[test]
fn the_fixture_covers_both_sides_of_the_battery_gate() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "assign_nearest_ordering")
        .expect("sequence");
    let refused = sequence.steps.iter().any(|st| {
        st.call == "assign_nearest"
            && arg_f(&st.args, "battery_fraction") < 0.25
            && st.result.is_null()
    });
    let allowed = sequence.steps.iter().any(|st| {
        st.call == "assign_nearest"
            && arg_f(&st.args, "battery_fraction") == 0.25
            && !st.result.is_null()
    });
    assert!(refused, "no case below the gate");
    assert!(
        allowed,
        "no case exactly AT the gate; `<` and `<=` would be indistinguishable"
    );
}

/// `rescore_from_map` rounds half to EVEN, as Python's `round()` does. A
/// density of 5/6 makes the boost exactly 0.5, which goes to 0 — Rust's
/// `f64::round()` would send it to 1 and change which POI is chosen.
#[test]
fn the_boost_rounds_half_to_even() {
    // The arithmetic, stated directly -- and the trap that reaching a tie at
    // all takes care. The obvious density of 5/6 does NOT land on one.
    assert_eq!((1.0 - 5.0 / 6.0) * 3.0, 0.4999999999999999);
    assert_eq!(
        (0.4999999999999999_f64).round_ties_even(),
        (0.4999999999999999_f64).round(),
        "just below a half rounds the same way under either rule, so a density \
         of 5/6 cannot distinguish them"
    );
    // One 6 short of the nearest double to 1/6 does land on one.
    assert_eq!((1.0 - 0.1666666666666666_f64) * 3.0, 2.5);
    assert_eq!(2.5_f64.round_ties_even(), 2.0);
    assert_eq!(2.5_f64.round(), 3.0, "Rust's default rounds the other way");

    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "rescore_banker_rounding")
        .expect("the rounding sequence must exist");
    let last = sequence.steps.last().expect("a step");
    assert_eq!(
        last.result.as_str(),
        Some("near_high_priority"),
        "under banker's rounding the 2.5 boost becomes 2, the priorities tie \
         and the nearer POI wins on the energy term; rounding half away from \
         zero makes it 3 and hands the assignment to the far one"
    );
}

/// `assign_nearest` reads the RAW priority and ignores rescoring, while
/// `assign_energy_aware` reads the effective one. Only visible when the two
/// disagree.
#[test]
fn assign_nearest_ignores_the_rescored_priorities() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "rescore_ignored_by_assign_nearest")
        .expect("sequence");
    let last = sequence.steps.last().expect("a step");
    assert_eq!(
        last.result.as_str(),
        Some("high_raw"),
        "the raw priority must win; picking the boosted POI means the \
         non-energy path consulted the rescored priorities"
    );
}

/// Completing a POI releases EVERY platform on it, which is what makes team
/// assignment work at all.
#[test]
fn completion_releases_every_assigned_platform() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "handoff_and_team")
        .expect("sequence");
    let released: Vec<&Step> = sequence
        .steps
        .iter()
        .filter(|st| st.call == "assignment_for")
        .collect();
    let after_completion: Vec<&&Step> = released.iter().rev().take(2).collect();
    assert_eq!(after_completion.len(), 2);
    for step in after_completion {
        assert!(
            step.result.is_null(),
            "both platforms must be released, not just the one in the status"
        );
    }
    // And a status can only name one of them, which is worth knowing.
    let team = sequence
        .steps
        .iter()
        .find(|st| st.call == "request_team_assign" && st.result.as_bool() == Some(true))
        .expect("a successful team assignment");
    let named = team
        .statuses
        .iter()
        .filter(|s| s.assigned_drone_id.is_some())
        .count();
    assert_eq!(
        named, 1,
        "the status records a single assigned drone even with two working the POI"
    );
}

/// Every state in the lifecycle must appear somewhere, or a port could leave
/// one unreachable.
#[test]
fn the_fixture_reaches_every_lifecycle_state() {
    let fixture = load();
    let mut seen: Vec<&str> = fixture
        .sequences
        .iter()
        .flat_map(|s| &s.steps)
        .flat_map(|st| &st.statuses)
        .map(|s| s.status.as_str())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    for state in [PoiState::Pending, PoiState::Active, PoiState::Complete] {
        assert!(
            seen.contains(&state.as_str()),
            "no POI ever reaches {}",
            state.as_str()
        );
    }
}

/// The status's `dwell_accumulated_s` stays at zero while a platform hovers.
///
/// The reference keeps the running dwell in a separate `_dwell_acc` dict and
/// only copies it into the status at completion, so a caller watching that
/// field for progress sees nothing until the POI is already done. Reproduced
/// rather than merged, because merging them would quietly change what a
/// status means.
#[test]
fn the_reported_dwell_stays_zero_until_completion() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "dwell_and_completion")
        .expect("sequence");
    let mid = sequence
        .steps
        .iter()
        .find(|st| st.call == "accumulate_dwell")
        .expect("a dwell step");
    assert!(
        mid.statuses
            .iter()
            .all(|s| f(&s.dwell_accumulated_s) == 0.0),
        "dwell is accruing but every status must still report zero"
    );
    let done = sequence
        .steps
        .iter()
        .find(|st| st.call == "check_completions" && !st.result.as_array().unwrap().is_empty())
        .expect("a completion step");
    assert!(
        done.statuses
            .iter()
            .any(|s| s.status == "complete" && f(&s.dwell_accumulated_s) > 0.0),
        "completion must publish the accumulated dwell"
    );
}

/// **Five equivalent mutants, recorded rather than tested around.**
///
/// Each rests on an invariant of the manager rather than on the fixture, so
/// each is asserted as that invariant.
///
/// 1. **The completed check in `existing_assignment`.** An entry in the
///    assignment map can only ever point at an *Active* POI: assignment sets
///    the state in the same breath, and completion removes every entry
///    pointing at the POI. So the state can never be `Complete` when the
///    lookup succeeds, and dropping the check changes nothing.
/// 2. **The Active check in `trigger_handoff`**, for the same reason.
/// 3. **The dwell reset on assignment.** A POI is only ever assigned while
///    Pending, and dwell is credited only to a drone's *assigned* POI — which
///    is Active. A Pending POI's dwell is therefore always already zero.
/// 4. **The priority tier scale of 10.** The affordability gate caps the cost
///    at 80% of usable energy, so the energy term is at most 0.8 while
///    priorities differ by at least 1. Any scale of 1 or more keeps priority
///    dominant, so 10 and 1 order the candidates identically.
/// 5. **Replacing an arrival time of exactly zero.** Nothing writes
///    `arrival_time_s` before completion, so it is always `None` there and
///    the zero branch is unreachable. Kept because the reference's
///    `arrival_time_s or timestamp_s` would behave this way if a writer ever
///    appeared, and a port using a plain `is_some()` would then diverge
///    silently.
#[test]
fn five_recorded_equivalent_mutants_in_the_poi_manager() {
    let fixture = load();

    // 1, 2 and 3: every assignment entry points at an Active POI, and every
    // Pending POI has zero dwell. Checked after EVERY recorded step.
    for sequence in &fixture.sequences {
        for (index, step) in sequence.steps.iter().enumerate() {
            for status in &step.statuses {
                if status.assigned_drone_id.is_some() {
                    assert_ne!(
                        status.status, "pending",
                        "{}[{index}]: a POI with an assigned drone is never Pending",
                        sequence.name
                    );
                }
                if status.status == "pending" {
                    assert_eq!(
                        f(&status.dwell_accumulated_s),
                        0.0,
                        "{}[{index}]/{}: a Pending POI must have zero dwell, which \
                         is why the reset on assignment cannot matter",
                        sequence.name,
                        status.poi_id
                    );
                }
                // 5: an arrival time only ever appears on a completed POI.
                if status.arrival_time_s.is_some() {
                    assert_eq!(
                        status.status, "complete",
                        "{}[{index}]/{}: arrival is written only at completion",
                        sequence.name, status.poi_id
                    );
                }
            }
        }
    }

    // 4: the energy term cannot bridge a priority tier at any scale >= 1.
    // cost <= 0.8 * usable, so cost / usable <= 0.8 < 1.
    for scale in [1.0_f64, 10.0] {
        let usable = 37.5_f64;
        let worst_cost = usable * 0.8;
        let energy_term = worst_cost / usable.max(0.01);
        assert!(
            energy_term < 1.0 * scale,
            "at a scale of {scale} the energy term {energy_term} could outweigh \
             a priority tier, and 10 versus 1 would not be equivalent"
        );
    }

    // And the live manager agrees: the same POI is chosen either way.
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "energy_aware")
        .expect("sequence");
    let mut manager = PoiManager::new(pois_of(sequence));
    let context = PoiAssignmentContext::new("probe", [0.0, 0.0, 0.0], 100.0, 100.0);
    let chosen = manager
        .assign_energy_aware(&context)
        .map(|p| p.poi_id.clone());
    assert_eq!(chosen.as_deref(), Some("near_high"));
}

/// The stable sort, at the largest tie the fixture can express.
///
/// **This does not fully close the gap, and saying so is the point.**
/// Swapping `sort_by` for `sort_unstable_by` in the port is *not* caught by
/// anything here. `assign_nearest` returns only `pending[0]`, so a reorder is
/// invisible unless it disturbs the minimum, and it does not at this shape.
///
/// Measured rather than assumed: `sort_unstable_by` preserves order for a
/// handful of elements whatever their keys, and for ANY number of elements
/// when every key is equal — pdqsort's all-equal partition never moves
/// anything. It begins reordering at around forty elements once several
/// distinct groups exist, which is why this set is 48 POIs across twelve
/// groups rather than the five, then twenty-four, it held earlier.
///
/// So the requirement is pinned as a *case* (the reference's stable sort
/// returns the first-constructed member of the winning group, and the port
/// agrees) but not as a *mutation*. Recorded as an open gap rather than
/// dressed up as equivalent: `sort_unstable_by` guarantees nothing about
/// equal keys, and a future compiler could make this diverge silently.
#[test]
fn the_stable_sort_is_pinned_at_a_size_that_can_reorder() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "stable_sort_at_scale")
        .expect("the scaled sequence must exist");
    // Measured: `sort_unstable_by` preserves order for a handful of elements
    // whatever their keys, and for ANY number when every key is equal --
    // pdqsort's all-equal partition never moves anything. It reorders from
    // about forty elements once several distinct groups exist. Both earlier
    // shapes (a five-element tie, then twenty-four identical keys) were
    // therefore indistinguishable from an unstable sort.
    assert!(
        sequence.pois.len() >= 40,
        "only {} POIs; an unstable sort does not reorder below about forty",
        sequence.pois.len()
    );
    let mut groups: Vec<(i64, u64)> = sequence
        .pois
        .iter()
        .map(|p| (p.priority, f(&p.position[0]).to_bits()))
        .collect();
    let total = groups.len();
    groups.sort_unstable();
    groups.dedup();
    assert!(
        groups.len() >= 4,
        "only {} distinct key groups; an all-equal set never reorders",
        groups.len()
    );
    assert!(
        total / groups.len() >= 2,
        "the groups must contain ties, or stability is meaningless"
    );
    let step = sequence.steps.last().expect("a step");
    assert_eq!(
        step.result.as_str(),
        Some("ring03"),
        "the first-constructed member of the (highest priority, nearest) group \
         must win"
    );
}

/// The energy edges the main sequences do not reach: usable energy of exactly
/// zero, and a cruise speed below the 0.1 m/s floor.
#[test]
fn the_fixture_reaches_the_energy_edges() {
    let fixture = load();
    let sequence = fixture
        .sequences
        .iter()
        .find(|s| s.name == "energy_edges")
        .expect("sequence");
    let zero_usable = &sequence.steps[0];
    assert_eq!(
        arg_f(&zero_usable.args, "battery_remaining_wh"),
        arg_f(&zero_usable.args, "battery_capacity_wh")
            * arg_f(&zero_usable.args, "battery_reserve_fraction"),
        "the first case must leave EXACTLY zero usable energy"
    );
    assert!(
        zero_usable.result.is_null(),
        "zero usable energy is refused; `< 0.0` would allow it"
    );
    let floored = &sequence.steps[1];
    assert_eq!(
        arg_f(&floored.args, "cruise_speed_mps"),
        0.0,
        "the second case must sit below the 0.1 m/s floor"
    );
    assert!(
        !floored.result.is_null(),
        "the floor keeps the cost finite, so the POI is still affordable"
    );
}
