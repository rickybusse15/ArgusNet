//! `argusnet-rs` as a user runs it: the real binary, real files, exit codes.

use std::path::{Path, PathBuf};
use std::process::Command;

use argusnet_sim::replay::ReplayDocument;
use argusnet_sim::scenario::{assemble, ScenarioBundle};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_argusnet-rs"))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argusnet-rs-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join(name)
}

fn read_replay(path: &Path) -> ReplayDocument {
    serde_json::from_str(&std::fs::read_to_string(path).expect("replay written"))
        .expect("replay parses")
}

/// The replay the binary writes is exactly the replay the runtime computes.
#[test]
fn sim_writes_the_runtimes_replay() {
    let replay = scratch("tt_replay.json");
    let status = binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture("scenario_bundle_tt.json"))
        .arg("--replay")
        .arg(&replay)
        .status()
        .expect("runs");
    assert!(status.success(), "exit {status}");

    let bundle = ScenarioBundle::from_json(
        &std::fs::read_to_string(fixture("scenario_bundle_tt.json")).unwrap(),
    )
    .unwrap();
    let direct = assemble(&bundle).unwrap().run_to_replay().unwrap();
    let mut written = read_replay(&replay);
    // The binary adds its provenance; the run itself is the runtime's.
    let provenance = written
        .meta
        .scenario
        .remove("provenance")
        .expect("provenance");
    assert_eq!(provenance["runtime"], "argusnet-rs");
    assert_eq!(provenance["input"], "bundle");
    assert_eq!(provenance["checkpoint_format"], 4);
    assert_eq!(written.frames.len() as u64, bundle.steps);
    assert_eq!(written, direct, "the binary's replay is not the runtime's");
}

/// Checkpoint then resume, through the command line, reproduces the tail of
/// the uninterrupted run exactly -- and the scan run's mission is mid-flight
/// at the checkpoint.
#[test]
fn a_checkpoint_resumed_from_the_command_line_continues_the_same_run() {
    let bundle = fixture("scenario_bundle_scan.json");
    let whole = scratch("scan_whole.json");
    let checkpoint = scratch("scan_ck.json");
    let rest = scratch("scan_rest.json");
    let at = 150;

    let run = |extra: &[&str], out: &Path| {
        let status = binary()
            .args(["sim", "--quiet", "--bundle"])
            .arg(&bundle)
            .arg("--replay")
            .arg(out)
            .args(extra)
            .status()
            .expect("runs");
        assert!(status.success(), "exit {status} for {extra:?}");
    };
    run(&[], &whole);
    let ck = checkpoint.to_str().unwrap().to_string();
    let at_text = at.to_string();
    run(
        &["--checkpoint-at", &at_text, "--checkpoint", &ck],
        &scratch("scan_first.json"),
    );
    run(&["--resume", &ck], &rest);

    let whole = read_replay(&whole);
    let rest = read_replay(&rest);
    assert_eq!(rest.frames.len(), whole.frames.len() - at);
    for (resumed, original) in rest.frames.iter().zip(&whole.frames[at..]) {
        assert_eq!(resumed, original, "diverged at t={}", original.timestamp_s);
    }
}

#[test]
fn bad_input_and_unsupported_bundles_exit_distinctly() {
    let missing = binary()
        .args(["sim", "--bundle", "/definitely/not/here.json"])
        .status()
        .unwrap();
    assert_eq!(missing.code(), Some(2));

    let garbage = scratch("garbage.json");
    std::fs::write(&garbage, "{ not json").unwrap();
    let invalid = binary()
        .args(["sim", "--bundle"])
        .arg(&garbage)
        .status()
        .unwrap();
    assert_eq!(invalid.code(), Some(2));

    let text = std::fs::read_to_string(fixture("scenario_bundle_tt.json")).unwrap();
    let unknown = scratch("unknown_mode.json");
    std::fs::write(
        &unknown,
        text.replacen(
            "\"mission_mode\":\"target_tracking\"",
            "\"mission_mode\":\"patrol\"",
            1,
        ),
    )
    .unwrap();
    let unsupported = binary()
        .args(["sim", "--bundle"])
        .arg(&unknown)
        .status()
        .unwrap();
    assert_eq!(unsupported.code(), Some(3));
}

#[test]
fn version_names_every_format() {
    let output = binary().arg("version").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    for needle in [
        "argusnet-rs",
        "scenario bundle",
        "checkpoint format",
        "with lane rates",
    ] {
        assert!(text.contains(needle), "{needle} missing from: {text}");
    }
}

/// A run that faults mid-way exits 1, names the fault and its tick once, and
/// writes no replay, rather than succeeding with a truncated one. A hole in
/// the target's truth reaches both fault paths: at t=3 the truth table itself
/// has nothing to prescribe; at t=10 the pursuing drone's oracle, which looks
/// ahead, faults in stage 1 first.
#[test]
fn a_run_that_faults_exits_1_and_writes_no_replay() {
    for (hole_s, expected) in [
        (3.0, "tick 6: no truth for asset-lynx at t=3"),
        (10.0, "stage sample_controllers_and_battery"),
    ] {
        let mut bundle: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture("scenario_bundle_tt.json")).unwrap(),
        )
        .unwrap();
        let samples = bundle["targets"][0]["samples"].as_array_mut().unwrap();
        let before = samples.len();
        samples.retain(|s| s["t"].as_f64() != Some(hole_s));
        assert_eq!(samples.len(), before - 1, "no sample at t={hole_s}");
        let broken = scratch(&format!("broken_truth_{hole_s}.json"));
        std::fs::write(&broken, serde_json::to_string(&bundle).unwrap()).unwrap();
        let replay = scratch(&format!("broken_replay_{hole_s}.json"));
        let _ = std::fs::remove_file(&replay);

        let output = binary()
            .args(["sim", "--bundle"])
            .arg(&broken)
            .arg("--replay")
            .arg(&replay)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(),
            Some(1),
            "hole at t={hole_s}: {stderr}"
        );
        assert!(stderr.contains(expected), "hole at t={hole_s}: {stderr}");
        assert_eq!(
            stderr.matches("tick ").count(),
            1,
            "tick named twice: {stderr}"
        );
        assert!(!replay.exists(), "a faulted run wrote a replay");
    }
}

/// What the viewer shows is what it can read. An `argusnet-rs` replay, parsed
/// by the viewer's own `argusnet-replay` types, carries the terrain mesh, the
/// zones and the tracker's stale threshold in `meta`, and every panel's
/// payload in its frames (P4.9a.2).
#[test]
fn the_viewer_reads_everything_an_argusnet_rs_replay_carries() {
    let read_with_viewer = |case: &str| -> argusnet_replay::ReplayDocument {
        let replay = scratch(&format!("viewer_{case}.json"));
        let status = binary()
            .args(["sim", "--quiet", "--bundle"])
            .arg(fixture(&format!("scenario_bundle_{case}.json")))
            .arg("--replay")
            .arg(&replay)
            .status()
            .expect("runs");
        assert!(status.success(), "{case}: exit {status}");
        serde_json::from_str(&std::fs::read_to_string(&replay).unwrap())
            .unwrap_or_else(|error| panic!("{case}: the viewer cannot parse it: {error}"))
    };

    let scan = read_with_viewer("scan");
    assert!(scan.terrain_viewer_mesh().is_some(), "no terrain mesh");
    assert!(!scan.zones().is_empty(), "no zones");
    assert!(scan.max_stale_steps().is_some(), "no tracker threshold");
    assert!(scan
        .frames
        .iter()
        .all(|f| f.mapping_state.is_some() && f.scan_mission_state.is_some()));
    let states = || {
        scan.frames
            .iter()
            .filter_map(|f| f.scan_mission_state.as_ref())
    };
    assert!(
        states().any(|s| !s.newly_scanned_cells.is_empty()),
        "no scanned cells"
    );
    assert!(
        states().any(|s| !s.egress_progress.is_empty()),
        "no egress progress"
    );
    assert!(
        states().any(|s| !s.localization_estimates.is_empty()),
        "no estimates"
    );
    assert!(states()
        .flat_map(|s| &s.poi_statuses)
        .all(|poi| poi.position.is_some()));
    assert!(states().all(|s| s.coordinator_drone_id.is_some()));
    assert!(scan.frames.iter().any(|f| !f.inspection_events.is_empty()));

    let follow = read_with_viewer("tt_follow");
    assert!(follow.frames.iter().all(|f| f.target_metadata.len() == 1));
    assert!(follow
        .frames
        .iter()
        .any(|f| !f.deconfliction_events.is_empty()));
    assert!(follow
        .frames
        .iter()
        .any(|f| !f.generation_rejections.is_empty()));
    assert!(follow.frames.iter().any(|f| f.localization_state.is_some()));
}

fn write_config(name: &str, options: serde_json::Value) -> PathBuf {
    let path = scratch(name);
    let config = serde_json::json!({
        "format": "argusnet.scenario-config",
        "version": 1,
        "seed": 5,
        "steps": 30,
        "dt_s": 0.5,
        "options": options,
    });
    std::fs::write(&path, config.to_string()).unwrap();
    path
}

/// `build` writes the library's bundle, and `sim --config` runs exactly what
/// `build` then `sim --bundle` runs: no Python, no difference.
#[test]
fn a_config_builds_and_runs_without_python() {
    let config = write_config(
        "config.json",
        serde_json::json!({"map_preset": "small", "mission_mode": "target_tracking",
                           "target_count": 2, "drone_count": 3, "ground_station_count": 3}),
    );
    let bundle_path = scratch("built_bundle.json");
    let status = binary()
        .args(["build", "--config"])
        .arg(&config)
        .arg("--bundle")
        .arg(&bundle_path)
        .status()
        .unwrap();
    assert!(status.success(), "build: {status}");
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&bundle_path).unwrap()).unwrap();
    let parsed = argusnet_sim::builder::ScenarioConfig::from_json(
        &std::fs::read_to_string(&config).unwrap(),
    )
    .unwrap();
    assert_eq!(
        written,
        argusnet_sim::builder::build_bundle(&parsed).unwrap()
    );

    let via_config = scratch("via_config.json");
    let via_bundle = scratch("via_bundle.json");
    for (flag, input, out) in [
        ("--config", &config, &via_config),
        ("--bundle", &bundle_path, &via_bundle),
    ] {
        let status = binary()
            .args(["sim", "--quiet", flag])
            .arg(input)
            .arg("--replay")
            .arg(out)
            .status()
            .unwrap();
        assert!(status.success(), "sim {flag}: {status}");
    }
    let mut config_replay = read_replay(&via_config);
    let mut bundle_replay = read_replay(&via_bundle);
    assert_eq!(config_replay.frames.len(), 30);
    // The provenance says which input each run came from; the runs are one.
    let from_config = config_replay.meta.scenario.remove("provenance").unwrap();
    let from_bundle = bundle_replay.meta.scenario.remove("provenance").unwrap();
    assert_eq!(
        (from_config["input"].as_str(), from_bundle["input"].as_str()),
        (Some("config"), Some("bundle"))
    );
    assert!(from_bundle["config"].is_null());
    assert_eq!(config_replay, bundle_replay);
}

#[test]
fn a_config_the_builder_refuses_exits_by_kind() {
    let unsupported = write_config(
        "unsupported.json",
        serde_json::json!({"adaptive_search": true}),
    );
    let unknown = write_config("unknown.json", serde_json::json!({"map_presett": "small"}));
    for (config, code) in [(&unsupported, 3), (&unknown, 2)] {
        for command in ["build", "sim"] {
            let mut run = binary();
            run.args([command, "--config"]).arg(config);
            if command == "build" {
                run.arg("--bundle").arg(scratch("never.json"));
            }
            assert_eq!(
                run.status().unwrap().code(),
                Some(code),
                "{command} {config:?}"
            );
        }
    }
    let neither = binary().args(["sim"]).status().unwrap();
    assert_eq!(neither.code(), Some(2), "sim needs --bundle or --config");
}

/// A run from a config records the resolved config -- every option, defaults
/// filled in -- and `--timing-json` writes the tick distribution, which stays
/// out of the replay.
#[test]
fn a_config_run_records_its_resolved_config_and_timing() {
    let config = write_config(
        "provenance.json",
        serde_json::json!({"map_preset": "small", "mission_mode": "target_tracking",
                           "target_count": 1, "drone_count": 2, "ground_station_count": 2}),
    );
    let replay = scratch("provenance_replay.json");
    let timing = scratch("timing.json");
    let status = binary()
        .args(["sim", "--quiet", "--config"])
        .arg(&config)
        .arg("--replay")
        .arg(&replay)
        .arg("--timing-json")
        .arg(&timing)
        .status()
        .unwrap();
    assert!(status.success());
    let document = read_replay(&replay);
    let provenance = &document.meta.scenario["provenance"];
    assert_eq!(provenance["input"], "config");
    let resolved = &provenance["config"]["options"];
    assert_eq!(resolved["map_preset"], "small");
    assert_eq!(
        resolved["weather_preset"], "clear",
        "defaults are filled in"
    );
    assert_eq!(provenance["config"]["steps"], 30);
    let timing: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&timing).unwrap()).unwrap();
    assert_eq!(provenance["runtime"], "argusnet-rs");
    assert_eq!(provenance["bundle_format"], "argusnet.scenario-bundle");
    assert_eq!(timing["ticks"], 30);
    let ms = |key: &str| timing["tick_ms"][key].as_f64().unwrap();
    assert!(ms("p50") > 0.0);
    assert!(ms("p50") <= ms("p95") && ms("p95") <= ms("p99") && ms("p99") <= ms("max"));
    assert!(ms("mean") > 0.0 && ms("mean") <= ms("max"));
    let startup = timing["startup_ms"].as_f64().unwrap();
    let total = timing["total_ms"].as_f64().unwrap();
    assert!(startup > 0.0, "startup includes the world build");
    // The ticks and the startup both happen inside the total.
    assert!(total >= startup + ms("mean") * 30.0 * 0.999);
    // This test and the binary it runs share a profile.
    let build = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    assert_eq!(timing["build"], build);
    assert!(!document.meta.scenario.contains_key("performance"));
}

/// `--stage-timing` (P4.10-pre) reports every stage's wall time and changes
/// nothing a run produces: the replay is byte-identical with it on.
#[test]
fn stage_timing_reports_every_stage_and_leaves_the_replay_unchanged() {
    let run = |name: &str, timed: bool| {
        let replay = scratch(&format!("{name}_replay.json"));
        let timing = scratch(&format!("{name}_timing.json"));
        let mut command = binary();
        command
            .args(["sim", "--bundle"])
            .arg(fixture("scenario_bundle_tt.json"))
            .arg("--replay")
            .arg(&replay)
            .arg("--timing-json")
            .arg(&timing);
        if timed {
            command.arg("--stage-timing");
        } else {
            command.arg("--quiet");
        }
        let output = command.output().expect("runs");
        assert!(output.status.success(), "exit {}", output.status);
        let timing: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&timing).unwrap()).unwrap();
        (
            std::fs::read(&replay).unwrap(),
            timing,
            String::from_utf8(output.stdout).unwrap(),
        )
    };
    let (plain, plain_timing, _) = run("untimed", false);
    let (timed, timing, stdout) = run("timed", true);
    assert!(plain == timed, "stage timing changed the replay bytes");
    assert!(plain_timing.get("stages").is_none(), "off unless asked for");

    let ticks = timing["ticks"].as_u64().unwrap();
    let stages = timing["stages"].as_object().expect("a stages block");
    assert_eq!(stages.len(), 11, "every stage: {:?}", stages.keys());
    for (name, summary) in stages {
        // Stage 0 also runs on the call that finds the run complete: commands
        // are admitted before the completion check (see `Simulation::step`).
        let expected = if name == "apply_commands" {
            ticks + 1
        } else {
            ticks
        };
        assert_eq!(
            summary["count"].as_u64(),
            Some(expected),
            "{name} once a tick"
        );
        let ms = |key: &str| summary[key].as_f64().unwrap();
        assert!(ms("p50_ms") <= ms("p95_ms") && ms("p95_ms") <= ms("p99_ms"));
        assert!(ms("p99_ms") <= ms("max_ms") && ms("total_ms") >= 0.0);
        assert!(stdout.contains(name.as_str()), "{name} in the table");
    }
    assert!(timing["replay_write_ms"].as_f64().unwrap() > 0.0);
    // The stages run inside the ticks they are part of.
    let stage_total: f64 = stages
        .iter()
        .filter(|(name, _)| name.as_str() != "apply_commands")
        .map(|(_, s)| s["total_ms"].as_f64().unwrap())
        .sum();
    let tick_total = timing["tick_ms"]["mean"].as_f64().unwrap() * ticks as f64;
    assert!(
        stage_total <= tick_total * 1.001,
        "{stage_total} > {tick_total}"
    );
}

// ---------------------------------------------------------------------------
// A4: the crash-recoverable run directory
// ---------------------------------------------------------------------------

fn run_dir(name: &str) -> PathBuf {
    let dir = scratch(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn record(bundle: &str, dir: &Path, extra: &[&str]) -> std::process::Output {
    binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture(bundle))
        .arg("--run-dir")
        .arg(dir)
        .args(["--checkpoint-every", "25"])
        .args(extra)
        .output()
        .expect("runs")
}

fn recover(dir: &Path) -> std::process::Output {
    binary()
        .args(["sim", "--quiet", "--recover"])
        .arg(dir)
        .args(["--checkpoint-every", "25"])
        .output()
        .expect("runs")
}

/// A run that crashes mid-way, with a write torn by the crash, recovers from
/// its last checkpoint to exactly the journal an uninterrupted run writes.
#[test]
fn a_crashed_run_recovers_to_the_uninterrupted_journal() {
    let whole = run_dir("journal-whole");
    let output = record("scenario_bundle_scan.json", &whole, &[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(whole.join("complete.json").exists());

    let crashed = run_dir("journal-crashed");
    let output = record(
        "scenario_bundle_scan.json",
        &crashed,
        &["--crash-after", "60"],
    );
    assert_eq!(output.status.code(), Some(4), "the crash aid exits 4");
    assert!(
        !crashed.join("complete.json").exists(),
        "a crashed run is incomplete"
    );
    // The crash also tore the write in flight.
    let mut journal = std::fs::OpenOptions::new()
        .append(true)
        .open(crashed.join("journal.jsonl"))
        .unwrap();
    std::io::Write::write_all(&mut journal, b"00ff00ff00ff00ff {\"seq\":60,").unwrap();
    drop(journal);

    let output = recover(&crashed);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let complete: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("complete.json")).unwrap())
            .unwrap();
    assert_eq!(
        complete["recovered_from_tick"], 50,
        "the last checkpoint before the crash"
    );
    assert_eq!(
        std::fs::read(crashed.join("journal.jsonl")).unwrap(),
        std::fs::read(whole.join("journal.jsonl")).unwrap(),
        "the recovered journal is not the uninterrupted one"
    );

    // Recovering a complete run does nothing.
    let output = recover(&crashed);
    assert!(output.status.success());
}

/// P1: an `efficient_v1` run recovers as that run. Its run directory
/// records the policy, so recovery needs no flag, and a flag naming another
/// policy is refused.
#[test]
fn a_crashed_efficient_run_recovers_with_its_planning_policy() {
    let policy = ["--planning-policy", "efficient_v1"];
    let whole = run_dir("journal-efficient-whole");
    let output = record("scenario_bundle_scan.json", &whole, &policy);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let crashed = run_dir("journal-efficient-crashed");
    let with_crash: Vec<&str> = policy
        .iter()
        .copied()
        .chain(["--crash-after", "60"])
        .collect();
    let output = record("scenario_bundle_scan.json", &crashed, &with_crash);
    assert_eq!(output.status.code(), Some(4), "the crash aid exits 4");
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["planning_policy"], "efficient_v1");
    assert_eq!(run["format"], "argusnet-run/2", "refused by a pre-P1 build");

    let named_default = binary()
        .args(["sim", "--quiet", "--recover"])
        .arg(&crashed)
        .args(["--checkpoint-every", "25", "--planning-policy", "reference"])
        .output()
        .expect("runs");
    // The reference is the flag's default, so naming it changes nothing:
    // the run directory's policy wins. (Only a different non-reference
    // policy is refused, and there is no second one yet.)
    assert!(
        named_default.status.success(),
        "{}",
        String::from_utf8_lossy(&named_default.stderr)
    );
    assert_eq!(
        std::fs::read(crashed.join("journal.jsonl")).unwrap(),
        std::fs::read(whole.join("journal.jsonl")).unwrap(),
        "the recovered journal is not the uninterrupted efficient_v1 one"
    );

    // A reference run's run directory does not mention a policy.
    let reference = run_dir("journal-efficient-reference");
    let output = record("scenario_bundle_scan.json", &reference, &[]);
    assert!(output.status.success());
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference.join("run.json")).unwrap())
            .unwrap();
    assert!(run.get("planning_policy").is_none());
    assert_eq!(run["format"], "argusnet-run/1");
}

/// B2.4: a run with the mission lane at its own rate recovers as that run,
/// not as the legacy one: its run directory records the rate.
#[test]
fn a_crashed_scheduled_run_recovers_with_its_mission_rate() {
    let rate = ["--executor", "lanes", "--mission-rate", "3/7"];
    let whole = run_dir("journal-scheduled-whole");
    let output = record("scenario_bundle_scan.json", &whole, &rate);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let crashed = run_dir("journal-scheduled-crashed");
    let with_crash: Vec<&str> = rate
        .iter()
        .copied()
        .chain(["--crash-after", "60"])
        .collect();
    let output = record("scenario_bundle_scan.json", &crashed, &with_crash);
    assert_eq!(output.status.code(), Some(4), "the crash aid exits 4");
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["mission_rate"], "3/7");
    assert_eq!(run["checkpoint_format"], 5);

    // Recovered with no scheduler flags at all: the run directory supplies them.
    let output = recover(&crashed);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(crashed.join("journal.jsonl")).unwrap(),
        std::fs::read(whole.join("journal.jsonl")).unwrap(),
        "the recovered scheduled journal is not the uninterrupted one"
    );

    // A different rate is refused rather than mixed in.
    let other = run_dir("journal-scheduled-other");
    let with_crash: Vec<&str> = rate
        .iter()
        .copied()
        .chain(["--crash-after", "60"])
        .collect();
    assert_eq!(
        record("scenario_bundle_scan.json", &other, &with_crash)
            .status
            .code(),
        Some(4)
    );
    let output = binary()
        .args(["sim", "--quiet", "--recover"])
        .arg(&other)
        .args(["--executor", "lanes", "--mission-rate", "1/1"])
        .output()
        .expect("runs");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("differs"));
}

/// B2.5: a run with its mapping lane at its own rate records the rate and,
/// crashed, recovers to the uninterrupted journal byte for byte -- including
/// when its last checkpoint holds boundaries the mapping lane had not yet
/// integrated.
#[test]
fn a_crashed_run_with_a_mapping_rate_recovers_with_it() {
    // Every 3 s against 1 s ticks: the checkpoint at tick 50 comes after the
    // lane ran at tick 48, with tick 49 still to integrate.
    let rates = [
        "--executor",
        "lanes",
        "--mapping-rate",
        "1/3",
        "--mission-rate",
        "3/7",
    ];
    let whole = run_dir("journal-mapping-whole");
    let replay = scratch("journal-mapping-whole-replay.json");
    let with_replay: Vec<&str> = rates
        .iter()
        .copied()
        .chain(["--replay", replay.to_str().unwrap()])
        .collect();
    let output = record("scenario_bundle_scan.json", &whole, &with_replay);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let meta = serde_json::to_value(&read_replay(&replay).meta).unwrap();
    assert_eq!(meta["lane_rates"]["mapping"], "1/3 Hz");
    assert_eq!(meta["lane_rates"]["mission"], "3/7 Hz");

    let crashed = run_dir("journal-mapping-crashed");
    let with_crash: Vec<&str> = rates
        .iter()
        .copied()
        .chain(["--crash-after", "61"])
        .collect();
    let output = record("scenario_bundle_scan.json", &crashed, &with_crash);
    assert_eq!(output.status.code(), Some(4), "the crash aid exits 4");
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["mapping_rate"], "1/3");
    let checkpoint: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("checkpoint.json")).unwrap())
            .unwrap();
    assert!(
        checkpoint["stage_states"]
            .as_object()
            .unwrap()
            .values()
            .any(|state| state.get("pending").is_some()),
        "the last checkpoint holds boundaries the mapping lane had not integrated"
    );

    // A different rate is refused before anything is touched.
    let output = binary()
        .args(["sim", "--quiet", "--recover"])
        .arg(&crashed)
        .args(["--executor", "lanes", "--mapping-rate", "1/1"])
        .output()
        .expect("runs");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("differs"));

    let output = recover(&crashed);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(crashed.join("journal.jsonl")).unwrap(),
        std::fs::read(whole.join("journal.jsonl")).unwrap(),
        "the recovered journal is not the uninterrupted one"
    );
}

/// B2.6: a run with its fusion lane at its own rate records the rate and,
/// crashed, recovers to the uninterrupted journal byte for byte -- including
/// when its last checkpoint holds boundaries the fusion lane had not yet
/// fused.
#[test]
fn a_crashed_run_with_a_fusion_rate_recovers_with_it() {
    // Every 3 s against 1 s ticks: the checkpoint at tick 50 comes after the
    // lane ran at tick 48, with ticks 49 and 50 still to fuse.
    let rates = ["--executor", "lanes", "--fusion-rate", "1/3"];
    let whole = run_dir("journal-fusion-whole");
    let replay = scratch("journal-fusion-whole-replay.json");
    let with_replay: Vec<&str> = rates
        .iter()
        .copied()
        .chain(["--replay", replay.to_str().unwrap()])
        .collect();
    let output = record("scenario_bundle_scan.json", &whole, &with_replay);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let meta = serde_json::to_value(&read_replay(&replay).meta).unwrap();
    assert_eq!(meta["lane_rates"]["fusion"], "1/3 Hz");

    let crashed = run_dir("journal-fusion-crashed");
    let with_crash: Vec<&str> = rates
        .iter()
        .copied()
        .chain(["--crash-after", "61"])
        .collect();
    let output = record("scenario_bundle_scan.json", &crashed, &with_crash);
    assert_eq!(output.status.code(), Some(4), "the crash aid exits 4");
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["fusion_rate"], "1/3");
    let checkpoint: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(crashed.join("checkpoint.json")).unwrap())
            .unwrap();
    assert!(
        checkpoint["stage_states"]
            .as_object()
            .unwrap()
            .values()
            .any(|state| state.get("engine").is_some() && state.get("pending").is_some()),
        "the last checkpoint holds boundaries the fusion lane had not fused"
    );

    // A different rate is refused before anything is touched.
    let output = binary()
        .args(["sim", "--quiet", "--recover"])
        .arg(&crashed)
        .args(["--executor", "lanes", "--fusion-rate", "1/1"])
        .output()
        .expect("runs");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("differs"));

    let output = recover(&crashed);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(crashed.join("journal.jsonl")).unwrap(),
        std::fs::read(whole.join("journal.jsonl")).unwrap(),
        "the recovered journal is not the uninterrupted one"
    );

    // A rate that would buffer without bound is refused, as for mapping.
    let refused = run_dir("journal-fusion-refused");
    let output = record(
        "scenario_bundle_scan.json",
        &refused,
        &["--executor", "lanes", "--fusion-rate", "1/2000"],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("at most 1024"));
    assert!(!refused.exists());
}

/// A mapping rate that would buffer more than the allowed run of skipped
/// ticks is refused before anything is written, so the run directory is not
/// left behind half made and the corrected command can use the same path.
#[test]
fn a_refused_mapping_rate_leaves_no_run_directory() {
    let dir = run_dir("journal-mapping-refused");
    let output = record(
        "scenario_bundle_scan.json",
        &dir,
        &["--executor", "lanes", "--mapping-rate", "1/2000"],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("at most 1024"));
    assert!(!dir.exists(), "a refused rate created {}", dir.display());
}

/// A checkpoint taken while the mission lane was sitting out boundaries
/// carries that lag; resuming it without the rate is refused.
#[test]
fn a_checkpoint_with_lane_lag_needs_its_rate_to_resume() {
    let checkpoint = scratch("lag-checkpoint.json");
    let replay = scratch("lag-replay.json");
    let output = binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture("scenario_bundle_scan.json"))
        .args(["--executor", "lanes", "--mission-rate", "1/7"])
        .args(["--checkpoint-at", "10", "--checkpoint"])
        .arg(&checkpoint)
        .arg("--replay")
        .arg(&replay)
        .output()
        .expect("runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
    assert_eq!(written["header"]["format_version"], 5);
    // The replay names the format its checkpoints are written in.
    let mut meta = read_replay(&replay).meta.scenario;
    assert_eq!(
        meta.remove("provenance").expect("provenance")["checkpoint_format"],
        5
    );
    assert!(
        written["stage_lag"]["evaluate_mission_transitions"]
            .as_u64()
            .unwrap()
            > 0
    );

    let output = binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture("scenario_bundle_scan.json"))
        .arg("--resume")
        .arg(&checkpoint)
        .output()
        .expect("runs");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--mission-rate"));

    // Nor with another rate.
    let output = binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture("scenario_bundle_scan.json"))
        .args(["--executor", "lanes", "--mission-rate", "3/7", "--resume"])
        .arg(&checkpoint)
        .output()
        .expect("runs");
    assert_eq!(output.status.code(), Some(2));

    let output = binary()
        .args(["sim", "--quiet", "--bundle"])
        .arg(fixture("scenario_bundle_scan.json"))
        .args(["--executor", "lanes", "--mission-rate", "1/7", "--resume"])
        .arg(&checkpoint)
        .output()
        .expect("runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn a_run_directory_is_never_reused() {
    let dir = run_dir("journal-reuse");
    assert!(record("scenario_bundle_tt.json", &dir, &[])
        .status
        .success());
    let output = record("scenario_bundle_tt.json", &dir, &[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("never reused"));
}

/// A damaged immutable configuration is reported as an incomplete artifact,
/// not run.
#[test]
fn a_damaged_run_configuration_is_an_incomplete_artifact() {
    let dir = run_dir("journal-damaged-config");
    let output = record("scenario_bundle_tt.json", &dir, &["--crash-after", "10"]);
    assert_eq!(output.status.code(), Some(4));
    let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
    std::fs::write(
        dir.join("run.json"),
        text.replacen("\\\"seed\\\":", "\\\"seed\\\":1", 1),
    )
    .unwrap();
    let output = recover(&dir);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("incomplete run artifact"), "{stderr}");
}

/// Damage in the middle of the journal, inside what the checkpoint covers, is
/// refused rather than truncated away.
#[test]
fn a_corrupt_journal_is_refused() {
    let dir = run_dir("journal-corrupt");
    let output = record("scenario_bundle_tt.json", &dir, &["--crash-after", "40"]);
    assert_eq!(output.status.code(), Some(4));
    let text = std::fs::read_to_string(dir.join("journal.jsonl")).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    let damaged = lines[5].replacen("\"tick\":5", "\"tick\":6", 1);
    assert_ne!(damaged, lines[5]);
    lines[5] = &damaged;
    std::fs::write(dir.join("journal.jsonl"), lines.join("\n") + "\n").unwrap();
    let output = recover(&dir);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("corrupt"), "{stderr}");
}

/// A checkpoint from another scenario, dropped into a run directory, is
/// refused before anything runs.
#[test]
fn a_checkpoint_from_another_run_is_refused() {
    let tt = run_dir("journal-tt");
    assert_eq!(
        record("scenario_bundle_tt.json", &tt, &["--crash-after", "30"])
            .status
            .code(),
        Some(4)
    );
    let scan = run_dir("journal-scan");
    assert_eq!(
        record("scenario_bundle_scan.json", &scan, &["--crash-after", "30"])
            .status
            .code(),
        Some(4)
    );
    std::fs::copy(tt.join("checkpoint.json"), scan.join("checkpoint.json")).unwrap();
    let output = recover(&scan);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot recover"), "{stderr}");
}
