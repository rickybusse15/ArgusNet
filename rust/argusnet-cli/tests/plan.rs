//! `argusnet-rs plan`: what a mission's declarations say of it, before it is flown.

use std::path::PathBuf;
use std::process::{Command, Output};

use argusnet_sim::plan_report::{CheckStatus, MissionPlan};

fn plan(bundle: &str, output: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_argusnet-rs"))
        .args(["plan", "--bundle", bundle, "--profile", "field_v3"])
        .args(["--output", output])
        .output()
        .expect("runs")
}

#[test]
fn a_mission_is_planned_and_one_too_short_is_refused() {
    let root = std::env::temp_dir().join(format!("argusnet-cli-plan-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = |name: &str| root.join(name).display().to_string();
    let fixture: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "../../tests/fixtures/scenario_bundle_scan.json",
    ]
    .iter()
    .collect();
    let read = |name: &str| -> MissionPlan {
        serde_json::from_str(&std::fs::read_to_string(path(name)).unwrap()).unwrap()
    };
    let good = plan(&fixture.display().to_string(), &path("good.json"));
    let said = String::from_utf8_lossy(&good.stdout).to_string();
    assert!(good.status.success(), "{said}");
    let planned = read("good.json");
    assert!(planned.check_format().is_ok());
    assert_eq!(planned.run_profile.as_deref(), Some("field_v3"));
    assert!(!planned.points.is_empty() && planned.scan.is_some());
    assert!(said.contains("points_within_reach"), "{said}");

    // The same mission, given twenty seconds.
    let mut bundle: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
    bundle["steps"] = serde_json::json!(20);
    std::fs::write(root.join("short.json"), bundle.to_string()).unwrap();
    let short = plan(&path("short.json"), &path("refused.json"));
    assert_eq!(short.status.code(), Some(2));
    let refused = read("refused.json");
    assert_eq!(refused.status, CheckStatus::Failed);
    assert!(refused
        .checks
        .iter()
        .any(|check| check.name == "time" && check.status == CheckStatus::Failed));
    std::fs::remove_dir_all(&root).unwrap();
}
