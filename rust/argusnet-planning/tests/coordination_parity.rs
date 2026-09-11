//! Parity for `CoordinationManager`, replayed from the captured reference.
//!
//! Three groups, matching the module's three concerns. The message group is
//! replayed as call sequences with the whole shared state compared after every
//! call, because `update_claimed` and `flush_messages` are the only stateful
//! pair here and their divergences are all order divergences: which of several
//! due messages wins, and whether a drained queue leaves a key behind.
//!
//! Floats arrive as `repr()` strings and are compared bit-for-bit, so a port
//! that grouped the formation arithmetic differently fails rather than passing
//! inside a tolerance.

use argusnet_planning::battery::BatteryState;
use argusnet_planning::coordination::{
    CoordinationManager, CoordinationPolicy, FormationMode, SharedMissionState,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Fixture {
    elections: Vec<ElectionCase>,
    message_sequences: Vec<MessageSequence>,
    formations: Vec<FormationCase>,
}

#[derive(Deserialize)]
struct ElectionCase {
    name: String,
    drone_ids: Vec<String>,
    battery_states: Vec<BatterySpec>,
    battery_capacity_wh: String,
    elected: Option<String>,
}

#[derive(Deserialize)]
struct BatterySpec {
    drone_id: String,
    remaining_wh: String,
}

#[derive(Deserialize)]
struct MessageSequence {
    name: String,
    message_latency_steps: i64,
    steps: Vec<MessageStep>,
}

#[derive(Deserialize)]
struct MessageStep {
    call: String,
    drone_id: Option<String>,
    cell: Option<[i64; 2]>,
    current_step: i64,
    state: StateSpec,
}

#[derive(Deserialize)]
struct StateSpec {
    claimed_cells: Vec<ClaimSpec>,
    pending: Vec<QueueSpec>,
}

#[derive(Deserialize)]
struct ClaimSpec {
    drone_id: String,
    cell: [i64; 2],
}

#[derive(Deserialize)]
struct QueueSpec {
    drone_id: String,
    messages: Vec<MessageSpec>,
}

#[derive(Deserialize)]
struct MessageSpec {
    cell: [i64; 2],
    deliver_at_step: i64,
}

#[derive(Deserialize)]
struct FormationCase {
    name: String,
    formation_mode: String,
    formation_spacing_m: String,
    drone_ids: Vec<String>,
    lead_heading_rad: String,
    offsets: Vec<OffsetSpec>,
}

#[derive(Deserialize)]
struct OffsetSpec {
    drone_id: String,
    offset: [String; 2],
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/coordination.json"
    );
    let text = std::fs::read_to_string(path).expect("coordination fixture");
    serde_json::from_str(&text).expect("coordination fixture parses")
}

/// `(drone_id, [(ci, cj, deliver_at_step)])` -- one pending queue, flattened
/// so the comparison covers identity, order and contents at once.
type FlatQueue = (String, Vec<(i64, i64, i64)>);

/// Compare the whole shared state, not just the field the call touched.
fn assert_state(case: &str, step: usize, shared: &SharedMissionState, expected: &StateSpec) {
    let mut actual_claims: Vec<(String, (i64, i64))> = shared
        .claimed_cells
        .iter()
        .map(|(id, cell)| (id.clone(), *cell))
        .collect();
    actual_claims.sort();
    let expected_claims: Vec<(String, (i64, i64))> = expected
        .claimed_cells
        .iter()
        .map(|c| (c.drone_id.clone(), (c.cell[0], c.cell[1])))
        .collect();
    assert_eq!(
        actual_claims, expected_claims,
        "{case} step {step}: claimed cells"
    );

    // Queue identity, order and contents all matter: a port that dropped a
    // drained queue, or that kept its queues in a hash map, differs here and
    // nowhere else.
    let actual_queues: Vec<FlatQueue> = shared
        .pending_queues()
        .map(|(id, queue)| {
            (
                id.to_string(),
                queue
                    .iter()
                    .map(|m| (m.cell.0, m.cell.1, m.deliver_at_step))
                    .collect(),
            )
        })
        .collect();
    let expected_queues: Vec<FlatQueue> = expected
        .pending
        .iter()
        .map(|q| {
            (
                q.drone_id.clone(),
                q.messages
                    .iter()
                    .map(|m| (m.cell[0], m.cell[1], m.deliver_at_step))
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        actual_queues, expected_queues,
        "{case} step {step}: pending queues"
    );
}

#[test]
fn elections_match_the_reference() {
    let fixture = load();
    assert!(!fixture.elections.is_empty(), "no election cases");
    let manager = CoordinationManager::default();

    for case in &fixture.elections {
        let states: HashMap<String, BatteryState> = case
            .battery_states
            .iter()
            .map(|spec| {
                (
                    spec.drone_id.clone(),
                    BatteryState {
                        remaining_wh: f(&spec.remaining_wh),
                        timestamp_s: 0.0,
                    },
                )
            })
            .collect();
        let elected =
            manager.elect_coordinator(&case.drone_ids, &states, f(&case.battery_capacity_wh));
        assert_eq!(
            elected.as_deref(),
            case.elected.as_deref(),
            "{}: elected coordinator",
            case.name
        );
    }
}

#[test]
fn message_sequences_match_the_reference() {
    let fixture = load();
    assert!(
        !fixture.message_sequences.is_empty(),
        "no message sequences"
    );
    let manager = CoordinationManager::default();

    for sequence in &fixture.message_sequences {
        let mut shared = SharedMissionState::with_latency(sequence.message_latency_steps);
        for (index, step) in sequence.steps.iter().enumerate() {
            match step.call.as_str() {
                "update_claimed" => {
                    let drone_id = step.drone_id.as_deref().expect("update needs a drone id");
                    let cell = step.cell.map(|c| (c[0], c[1]));
                    manager.update_claimed(drone_id, cell, &mut shared, step.current_step);
                }
                "flush_messages" => manager.flush_messages(&mut shared, step.current_step),
                other => panic!("{}: unknown call {other:?}", sequence.name),
            }
            assert_state(&sequence.name, index, &shared, &step.state);
        }
    }
}

#[test]
fn formation_offsets_match_the_reference_bit_for_bit() {
    let fixture = load();
    assert!(!fixture.formations.is_empty(), "no formation cases");

    for case in &fixture.formations {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::from_name(&case.formation_mode),
            formation_spacing_m: f(&case.formation_spacing_m),
            ..CoordinationPolicy::default()
        });
        let offsets = manager.formation_offsets(&case.drone_ids, f(&case.lead_heading_rad));

        assert_eq!(
            offsets.len(),
            case.offsets.len(),
            "{}: offset count",
            case.name
        );
        for (index, (actual, expected)) in offsets.iter().zip(&case.offsets).enumerate() {
            assert_eq!(
                actual.0, expected.drone_id,
                "{}: drone at position {index}",
                case.name
            );
            // Bit-for-bit: these are the same arithmetic on the same inputs,
            // so anything but equality is a regrouped expression.
            assert_eq!(
                actual.1[0].to_bits(),
                f(&expected.offset[0]).to_bits(),
                "{}: {} offset x ({} vs {})",
                case.name,
                expected.drone_id,
                actual.1[0],
                expected.offset[0]
            );
            assert_eq!(
                actual.1[1].to_bits(),
                f(&expected.offset[1]).to_bits(),
                "{}: {} offset y ({} vs {})",
                case.name,
                expected.drone_id,
                actual.1[1],
                expected.offset[1]
            );
        }
    }
}

/// The fixture is only worth what its hardest cases pin. If the formation set
/// ever loses its rank-3 entries it silently stops testing the arithmetic, so
/// assert here that at least one non-lead offset disagrees with the plausible
/// wrong grouping -- the same guard the capture script applies on the way in.
#[test]
fn the_formation_fixture_still_discriminates_the_grouping() {
    let fixture = load();
    let mut discriminating = 0usize;

    for case in &fixture.formations {
        let mode = FormationMode::from_name(&case.formation_mode);
        if mode == FormationMode::Off || matches!(mode, FormationMode::Unrecognized(_)) {
            continue;
        }
        let heading = f(&case.lead_heading_rad);
        let spacing = f(&case.formation_spacing_m);
        let perp = [-heading.sin(), heading.cos()];
        let fwd = [heading.cos(), heading.sin()];

        for (index, expected) in case.offsets.iter().enumerate().skip(1) {
            let sign = if index % 2 == 1 { 1.0 } else { -1.0 };
            let rank = index.div_ceil(2) as f64;
            // Fold the factors in one at a time instead of forming the scalar.
            let wrong: [f64; 2] = match mode {
                FormationMode::LineAbreast => [
                    ((perp[0] * sign) * rank) * spacing,
                    ((perp[1] * sign) * rank) * spacing,
                ],
                _ => [
                    -(((fwd[0] * rank) * spacing) * 0.5)
                        + (((perp[0] * sign) * rank) * spacing) * 0.5,
                    -(((fwd[1] * rank) * spacing) * 0.5)
                        + (((perp[1] * sign) * rank) * spacing) * 0.5,
                ],
            };
            if wrong[0].to_bits() != f(&expected.offset[0]).to_bits()
                || wrong[1].to_bits() != f(&expected.offset[1]).to_bits()
            {
                discriminating += 1;
            }
        }
    }

    assert!(
        discriminating > 0,
        "no formation offset distinguishes the vector-scaling grouping; \
         ranks 1 and 2 scale exactly, so the set needs a formation of at \
         least seven drones"
    );
}
