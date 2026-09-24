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
    for needle in ["argusnet-rs", "scenario bundle", "checkpoint format"] {
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
