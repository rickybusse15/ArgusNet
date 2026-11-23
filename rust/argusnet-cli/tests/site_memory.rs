//! `sim --site-memory` and `site`: a site remembered between missions (F6).

use std::path::PathBuf;
use std::process::{Command, Output};

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_argusnet-rs"))
        .args(args)
        .output()
        .expect("runs")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn two_missions_of_one_site_are_kept_and_compared() {
    let root = std::env::temp_dir().join(format!("argusnet-cli-site-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = |name: &str| root.join(name).display().to_string();
    let bundle: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "../../tests/fixtures/scenario_bundle_scan.json",
    ]
    .iter()
    .collect();
    let bundle = bundle.display().to_string();
    let (memory, replay) = (path("memory"), path("replay.json"));
    let sim = |profile: &[&str]| {
        let mut args = vec!["sim", "--bundle", &bundle, "--replay", &replay];
        args.extend_from_slice(profile);
        args.extend_from_slice(&["--site-memory", &memory, "--site", "hills"]);
        cli(&args)
    };
    let empty = cli(&["site", "--memory", &memory]);
    assert!(text(&empty).contains("no sites"), "{}", text(&empty));

    let first = sim(&["--profile", "field_v3"]);
    assert!(first.status.success(), "{}", text(&first));
    assert!(text(&first).contains("first revision"), "{}", text(&first));
    // The same mission again: the same map, and nothing changed.
    let second = sim(&["--profile", "field_v3"]);
    assert!(second.status.success(), "{}", text(&second));
    assert!(
        text(&second).contains(" 0 changes in "),
        "{}",
        text(&second)
    );
    let listed = cli(&["site", "--memory", &memory]);
    assert!(text(&listed).contains("hills: 2 revisions"));
    let changes = path("changes.json");
    let compared = cli(&[
        "site", "--memory", &memory, "--site", "hills", "--output", &changes,
    ]);
    assert!(compared.status.success(), "{}", text(&compared));
    let document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&changes).unwrap()).unwrap();
    assert_eq!(document["format"], "argusnet.site-changes");
    assert_eq!(document["changes"], serde_json::json!([]));
    assert!(document["compared_cells"].as_u64().unwrap() > 100);

    // The same grid about another place on the earth is another frame: a world
    // whose buildings or features, and not terrain, say where it is, too.
    let mut moved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&bundle).unwrap()).unwrap();
    let origin = &mut moved["presentation"]["enu_origin"];
    assert!(origin.is_object(), "the bundle says where it is");
    origin["latitude_deg"] = serde_json::json!(12.5);
    assert!(moved.get("terrain_import").is_none());
    std::fs::write(root.join("moved.json"), moved.to_string()).unwrap();
    let elsewhere = cli(&[
        "sim",
        "--bundle",
        &path("moved.json"),
        "--replay",
        &replay,
        "--profile",
        "field_v3",
        "--site-memory",
        &memory,
        "--site",
        "hills",
    ]);
    assert!(!elsewhere.status.success());
    assert!(text(&elsewhere).contains("two sites are not compared"));

    // A run that keeps no such map remembers nothing, and says so.
    let refused = sim(&[]);
    assert!(!refused.status.success());
    assert!(text(&refused).contains("quality requirement"));
    let nameless = cli(&["site", "--memory", &memory, "--site", "../up"]);
    assert!(!nameless.status.success());
    std::fs::remove_dir_all(&root).unwrap();
}
