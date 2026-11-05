//! The wire evaluation: every wire mode over a set of placed lines and seeds,
//! judged against truth. Truth is read here and nowhere in the flying.
//!
//! ```text
//! cargo run --release -p argusnet-sim --example wire_eval -- [--quick] [--out FILE]
//! ```
//!
//! Each row is one run. The report is JSON lines on stdout or in `--out`, then
//! one summary line per mode on stderr.

use std::collections::BTreeMap;
use std::io::Write;

use argusnet_mapping::poles::PoleConfig;
use argusnet_mapping::wire_prior::PriorDegradation;
use argusnet_mapping::wires::WireTrackerConfig;
use argusnet_sim::belief_world::{
    ExplorationConfig, UncertainPrior, WirePlanningConfig, WorldKnowledge,
};
use argusnet_sim::rig::RigProfile;
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, build_world, ScenarioBundle};
use argusnet_sim::wire_sensing::WireSensorModel;
use argusnet_world::powerline::{closest_on_segment, PowerLine};

/// How the wires are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
    Oracle,
    Band,
    Tracks,
    Poles,
    Planning,
    Prior,
}

impl Mode {
    const ALL: [Mode; 6] = [
        Mode::Oracle,
        Mode::Band,
        Mode::Tracks,
        Mode::Poles,
        Mode::Planning,
        Mode::Prior,
    ];

    fn name(self) -> &'static str {
        match self {
            Mode::Oracle => "oracle",
            Mode::Band => "w1_band",
            Mode::Tracks => "w2_tracks",
            Mode::Poles => "w3_poles",
            Mode::Planning => "w4_planning",
            Mode::Prior => "w5_prior",
        }
    }

    fn exploration(self, seed: u64) -> ExplorationConfig {
        let measured = self != Mode::Oracle;
        ExplorationConfig {
            rig: Some(RigProfile::provisional()),
            measured_wires: measured.then(WireSensorModel::default),
            wire_tracker: (self >= Mode::Tracks).then(WireTrackerConfig::default),
            poles: (self >= Mode::Poles).then(PoleConfig::default),
            wire_planning: (self >= Mode::Planning).then(WirePlanningConfig::default),
            uncertain_prior: (self >= Mode::Prior).then(|| UncertainPrior {
                config: Default::default(),
                spans: Vec::new(),
                degraded_truth: Some(PriorDegradation {
                    seed,
                    ..PriorDegradation::default()
                }),
            }),
            ..ExplorationConfig::default()
        }
    }
}

/// The scan fixture with a double conductor line along y at `x_m`, its top
/// `top_m` above the datum, on five poles.
fn bundle(x_m: f64, top_m: f64, seed: u64) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    let support = |y: f64| serde_json::json!({ "xy_m": [x_m, y], "base_z_m": 100.0, "height_m": top_m - 100.0 });
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "across",
        "supports": [support(-300.0), support(-150.0), support(0.0), support(150.0), support(300.0)],
        "conductor_offsets_m": [-2.0, 2.0],
        "sag_ratio": 0.02,
        "conductor_radius_m": 0.015,
        "clearance_m": 10.0
    }]);
    document["seed"] = serde_json::json!(seed);
    ScenarioBundle::from_json(&document.to_string()).unwrap()
}

fn off_conductor(lines: &[PowerLine], point_m: [f64; 3]) -> f64 {
    lines
        .iter()
        .flat_map(|line| line.pieces())
        .map(|piece| {
            let on = closest_on_segment(point_m, piece.start_m, piece.end_m);
            (0..3)
                .map(|k| (on[k] - point_m[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// One run's judgement.
#[derive(Debug, Default, Clone, serde::Serialize)]
struct Row {
    mode: &'static str,
    line_x_m: f64,
    line_top_m: f64,
    seed: u64,
    ticks_within_10_m: u64,
    closest_m: f64,
    contacts: u64,
    known_share: f64,
    poles: u64,
    poles_true: u64,
    worst_pole_error_m: f64,
    tracks: u64,
    corridors: u64,
    prior_open: u64,
    prior_confirmed: u64,
    prior_refuted: u64,
}

fn run_one(mode: Mode, x_m: f64, top_m: f64, seed: u64) -> Row {
    let bundle = bundle(x_m, top_m, seed);
    let lines = build_world(&bundle.world).unwrap().power_lines.to_vec();
    let mut run = assemble(&bundle).unwrap();
    RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        world_knowledge: WorldKnowledge::NoPrior,
        exploration: Some(mode.exploration(seed)),
        ..Default::default()
    }
    .apply(&mut run, &bundle)
    .expect("assembles");
    let mut row = Row {
        mode: mode.name(),
        line_x_m: x_m,
        line_top_m: top_m,
        seed,
        closest_m: f64::INFINITY,
        poles_true: lines[0].supports.len() as u64,
        ..Row::default()
    };
    let mut last = BTreeMap::new();
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        row.contacts += frame["world_contacts"]
            .as_array()
            .map_or(0, |contacts| contacts.len() as u64);
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|k| node["position"][k].as_f64().unwrap());
            let id = node["node_id"].as_str().unwrap().to_string();
            let from = last.insert(id, p).unwrap_or(p);
            if lines
                .iter()
                .any(|line| line.clearance_along(from, p) < 10.0)
            {
                row.ticks_within_10_m += 1;
            }
            row.closest_m = row.closest_m.min(off_conductor(&lines, p));
        }
        if let Some(belief) = frame["mapping_state"].get("belief") {
            let count = |key: &str| belief.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
            row.tracks = count("wire_tracks");
            row.poles = count("wire_poles");
            row.corridors = count("wire_corridors");
            row.prior_open = count("wire_prior_open");
            row.prior_confirmed = count("wire_prior_confirmed");
            row.prior_refuted = count("wire_prior_refuted");
            if let (Some(known), Some(total)) = (
                belief.get("known_cells").and_then(|v| v.as_f64()),
                belief.get("total_cells").and_then(|v| v.as_f64()),
            ) {
                row.known_share = known / total;
            }
        }
    }
    if let Some(detector) = run
        .mapping()
        .and_then(|m| m.belief_grid())
        .and_then(|g| g.pole_detector())
    {
        for pole in detector.poles() {
            let [x, y] = pole.xy_m();
            let off = lines[0]
                .supports
                .iter()
                .map(|s| (s.xy_m[0] - x).hypot(s.xy_m[1] - y))
                .fold(f64::INFINITY, f64::min);
            row.worst_pole_error_m = row.worst_pole_error_m.max(off);
        }
    }
    row
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|k| args.get(k + 1));
    let placements: &[(f64, f64)] = if quick {
        &[(0.0, 260.0), (0.0, 340.0), (-120.0, 300.0)]
    } else {
        &[
            (0.0, 220.0),
            (0.0, 260.0),
            (0.0, 300.0),
            (0.0, 340.0),
            (-120.0, 260.0),
            (-120.0, 300.0),
            (-120.0, 340.0),
            (120.0, 260.0),
            (120.0, 300.0),
            (120.0, 340.0),
            (60.0, 260.0),
            (60.0, 300.0),
            (60.0, 340.0),
        ]
    };
    let seeds: &[u64] = if quick { &[7] } else { &[7, 42, 1234] };
    let mut rows = Vec::new();
    for &(x, top) in placements {
        for &seed in seeds {
            for mode in Mode::ALL {
                rows.push(run_one(mode, x, top, seed));
            }
        }
    }
    let text: String = rows
        .iter()
        .map(|row| serde_json::to_string(row).unwrap() + "\n")
        .collect();
    match out {
        Some(path) => std::fs::File::create(path)
            .and_then(|mut file| file.write_all(text.as_bytes()))
            .expect("writes the report"),
        None => print!("{text}"),
    }
    for mode in Mode::ALL {
        let of: Vec<&Row> = rows.iter().filter(|r| r.mode == mode.name()).collect();
        let n = of.len() as f64;
        eprintln!(
            "{:10} near {:7.1}  closest {:5.2} (worst {:5.2})  contacts {:6.1}  known {:.3}  poles {:.1}/{}  pole err {:.2}  tracks {:.1}",
            mode.name(),
            of.iter().map(|r| r.ticks_within_10_m as f64).sum::<f64>() / n,
            of.iter().map(|r| r.closest_m).sum::<f64>() / n,
            of.iter().map(|r| r.closest_m).fold(f64::INFINITY, f64::min),
            of.iter().map(|r| r.contacts as f64).sum::<f64>() / n,
            of.iter().map(|r| r.known_share).sum::<f64>() / n,
            of.iter().map(|r| r.poles as f64).sum::<f64>() / n,
            of.first().map_or(0, |r| r.poles_true),
            of.iter().map(|r| r.worst_pole_error_m).fold(0.0, f64::max),
            of.iter().map(|r| r.tracks as f64).sum::<f64>() / n,
        );
    }
}
