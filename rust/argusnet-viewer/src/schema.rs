use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bevy::prelude::Resource;
use serde::Deserialize;
use serde_json::Value;

pub const SCENE_FORMAT_VERSION: &str = "smartscene-v1";
pub const STYLE_FORMAT_VERSION: &str = "smartstyle-v1";

#[derive(Debug, Clone, Deserialize)]
pub struct Bounds2d {
    pub x_min_m: f32,
    pub x_max_m: f32,
    pub y_min_m: f32,
    pub y_max_m: f32,
}

impl Bounds2d {
    pub fn center_xy(&self) -> [f32; 2] {
        [
            (self.x_min_m + self.x_max_m) * 0.5,
            (self.y_min_m + self.y_max_m) * 0.5,
        ]
    }

    pub fn span_xy(&self) -> [f32; 2] {
        [self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m]
    }

    pub fn inset(&self, margin: f32) -> Self {
        Self {
            x_min_m: self.x_min_m + margin,
            x_max_m: self.x_max_m - margin,
            y_min_m: self.y_min_m + margin,
            y_max_m: self.y_max_m - margin,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneManifestLayer {
    pub id: String,
    pub kind: String,
    pub semantic_kind: String,
    pub asset_path: String,
    pub style_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneManifestReplay {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneManifestMetadata {
    pub environment: String,
    pub style: String,
    /// Optional (P2.1): the land-cover classes the terrain material tints by.
    #[serde(default)]
    pub land_cover: Option<String>,
}

/// The scene's land cover, as the terrain material can use it.
#[derive(Debug, Clone, Default)]
pub enum LandCoverState {
    /// The scene declares none: built without an environment. No tint.
    #[default]
    Absent,
    Ready(std::sync::Arc<crate::land_cover_raster::LandCoverRaster>),
    /// Declared and readable, but not something this viewer can draw.
    Unavailable(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneManifest {
    pub format_version: String,
    pub scene_id: String,
    pub bounds_xy_m: Bounds2d,
    pub runtime_crs: Value,
    pub source_crs_id: String,
    pub layers: Vec<SceneManifestLayer>,
    pub replay: Option<SceneManifestReplay>,
    pub metadata: SceneManifestMetadata,
    pub build: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StyleLayer {
    pub id: String,
    pub semantic_kind: String,
    pub color_rgba: [f32; 4],
    pub opacity: f32,
    pub elevation_mode: String,
    pub draw_order: i32,
    pub default_visibility: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StyleDocument {
    pub style_version: String,
    pub layers: Vec<StyleLayer>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TerrainSummary {
    pub min_height_m: Option<f32>,
    pub max_height_m: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvironmentDocument {
    pub source_kind: Option<String>,
    pub scene_id: Option<String>,
    pub source_crs_id: Option<String>,
    pub runtime_crs: Value,
    pub bounds_xy_m: Bounds2d,
    pub terrain_summary: Option<TerrainSummary>,
}

#[derive(Debug, Clone, Resource)]
pub struct ScenePackage {
    pub root: PathBuf,
    pub manifest: SceneManifest,
    pub style: StyleDocument,
    pub environment: EnvironmentDocument,
    pub replay: Option<Value>,
    pub land_cover: LandCoverState,
    style_lookup: HashMap<String, StyleLayer>,
}

impl ScenePackage {
    pub fn load(scene_path: impl AsRef<Path>) -> Result<Self> {
        let root = normalize_scene_root(scene_path.as_ref())?;
        let manifest: SceneManifest = read_json(root.join("scene_manifest.json"))?;
        if manifest.format_version != SCENE_FORMAT_VERSION {
            anyhow::bail!(
                "unsupported scene format {:?}; expected {}",
                manifest.format_version,
                SCENE_FORMAT_VERSION
            );
        }

        let style_path = safe_manifest_path(&root, &manifest.metadata.style, "metadata.style")?;
        let style: StyleDocument = read_json(style_path)?;
        if style.style_version != STYLE_FORMAT_VERSION {
            anyhow::bail!(
                "unsupported style format {:?}; expected {}",
                style.style_version,
                STYLE_FORMAT_VERSION
            );
        }

        let environment_path = safe_manifest_path(
            &root,
            &manifest.metadata.environment,
            "metadata.environment",
        )?;
        let environment: EnvironmentDocument = read_json(environment_path)?;
        // A manifest that declares `replay.path` is asserting the file is there.
        // Silently falling back to `None` turned a packaging mistake into a viewer
        // that opens on an empty timeline with no indication why -- both bundled
        // sample scenes shipped that way. A scene with no replay is legitimate
        // (live mode streams its frames); a scene that *claims* one is not.
        let replay = match manifest.replay.as_ref() {
            Some(entry) => {
                let replay_path = safe_manifest_path(&root, &entry.path, "replay.path")?;
                if !replay_path.exists() {
                    anyhow::bail!(
                        "scene manifest declares replay.path {:?}, but {} does not exist; \
                         omit the `replay` key for a scene that carries no replay",
                        entry.path,
                        replay_path.display()
                    );
                }
                Some(read_json::<Value>(replay_path)?)
            }
            None => None,
        };

        // Like `replay.path`: a declared file that is missing or unreadable is
        // a packaging mistake and fails the load. One that is readable but
        // beyond what the viewer draws is reported, not fatal.
        let land_cover = match manifest.metadata.land_cover.as_ref() {
            Some(path) => {
                let path = safe_manifest_path(&root, path, "metadata.land_cover")?;
                let document: crate::land_cover_raster::LandCoverDocument = read_json(path)?;
                match crate::land_cover_raster::LandCoverRaster::from_document(document) {
                    Ok(raster) => LandCoverState::Ready(std::sync::Arc::new(raster)),
                    Err(reason) => LandCoverState::Unavailable(reason),
                }
            }
            None => LandCoverState::Absent,
        };

        let style_lookup = style
            .layers
            .iter()
            .cloned()
            .map(|layer| (layer.id.clone(), layer))
            .collect();

        Ok(Self {
            root,
            manifest,
            style,
            environment,
            replay,
            land_cover,
            style_lookup,
        })
    }

    pub fn style_for(&self, style_id: &str) -> Option<&StyleLayer> {
        self.style_lookup.get(style_id)
    }

    pub fn source_kind(&self) -> Option<&str> {
        self.environment.source_kind.as_deref().or_else(|| {
            self.manifest
                .build
                .get("source_kind")
                .and_then(|value| value.as_str())
        })
    }

    pub fn is_synthetic_source(&self) -> bool {
        matches!(self.source_kind(), Some("synthetic"))
    }
}

/// Write a scene package with no layers at all into `dir`: the frame for a
/// run whose world is not known in advance (`world_knowledge: none`). It
/// carries the map's bounds, so camera, fog, sky and shadows frame the right
/// area; a rough height span from the launch pads; and the marker styles. The
/// terrain and buildings are what the drones build.
pub fn write_unmapped_scene(
    dir: &Path,
    scene_id: &str,
    bounds: [f64; 4],
    height_span_m: (f64, f64),
) -> Result<PathBuf> {
    let [x_min, x_max, y_min, y_max] = bounds;
    let bounds = serde_json::json!({
        "x_min_m": x_min, "x_max_m": x_max, "y_min_m": y_min, "y_max_m": y_max,
    });
    let crs = serde_json::json!({
        "source_crs_id": "local-enu",
        "runtime_crs_id": "local-enu",
        "origin_geodetic": {"lat_deg": null, "lon_deg": null, "h_m": null},
        "xy_units": "meters",
        "z_datum": "local",
    });
    let manifest = serde_json::json!({
        "format_version": "smartscene-v1",
        "scene_id": scene_id,
        "bounds_xy_m": bounds,
        "runtime_crs": crs,
        "source_crs_id": "local-enu",
        "layers": [],
        "replay": null,
        "metadata": {"environment": "metadata/environment.json", "style": "metadata/style.json"},
        "build": {"source_kind": "self-built", "tool": "argusnet-viewer --run --world-knowledge none"},
    });
    let environment = serde_json::json!({
        "source_kind": "self-built",
        "scene_id": scene_id,
        "source_crs_id": "local-enu",
        "runtime_crs": crs,
        "bounds_xy_m": bounds,
        "terrain_summary": {
            "kind": "unknown-until-sensed",
            "min_height_m": height_span_m.0,
            "max_height_m": height_span_m.1,
        },
    });
    let style_layer = |id: &str, rgba: [f64; 4]| {
        serde_json::json!({
            "id": id, "semantic_kind": id, "color_rgba": rgba, "opacity": rgba[3],
            "elevation_mode": "absolute", "draw_order": 100, "default_visibility": true,
        })
    };
    let style = serde_json::json!({
        "style_version": "smartstyle-v1",
        "layers": [
            style_layer("tracks", [1.0, 0.604, 0.302, 1.0]),
            style_layer("truths", [0.302, 0.835, 0.906, 1.0]),
            style_layer("nodes", [0.953, 0.941, 0.647, 1.0]),
        ],
    });
    fs::create_dir_all(dir.join("metadata"))
        .with_context(|| format!("failed to create {}", dir.display()))?;
    for (path, value) in [
        ("scene_manifest.json", &manifest),
        ("metadata/environment.json", &environment),
        ("metadata/style.json", &style),
    ] {
        fs::write(dir.join(path), serde_json::to_vec_pretty(value)?)
            .with_context(|| format!("failed to write {}", dir.join(path).display()))?;
    }
    Ok(dir.to_path_buf())
}

pub fn normalize_scene_root(scene_path: &Path) -> Result<PathBuf> {
    let resolved = scene_path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", scene_path.display()))?;
    if resolved.is_file() {
        if resolved.file_name().and_then(|value| value.to_str()) != Some("scene_manifest.json") {
            anyhow::bail!(
                "expected a scene package directory or scene_manifest.json, got {}",
                resolved.display()
            );
        }
        return Ok(resolved
            .parent()
            .context("scene_manifest.json is missing a parent directory")?
            .to_path_buf());
    }
    if !resolved.join("scene_manifest.json").exists() {
        anyhow::bail!(
            "scene package at {} does not contain scene_manifest.json",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T> {
    let raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

/// Joins `relative` onto `root`, rejecting an absolute path or any `..`
/// component. Manifest path fields (`metadata.style`, `metadata.environment`,
/// `replay.path`) can come from an untrusted, shared scene package and are
/// read directly via `fs::read_to_string` in [`read_json`] — an unescaped
/// path here is an arbitrary file read, not just an asset-load path
/// traversal, so this is checked before the path is ever joined and read.
fn safe_manifest_path(root: &Path, relative: &str, field_name: &str) -> Result<PathBuf> {
    let candidate = Path::new(relative);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        anyhow::bail!(
            "scene manifest {field_name} must be a relative path with no '..' components, got {relative:?}"
        );
    }
    Ok(root.join(candidate))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{safe_manifest_path, ScenePackage};

    /// The committed scene, copied to a temporary directory for editing.
    fn scene_copy(name: &str) -> std::path::PathBuf {
        fn copy(from: &Path, to: &Path) {
            fs::create_dir_all(to).unwrap();
            for entry in fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                let target = to.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy(&entry.path(), &target);
                } else {
                    fs::copy(entry.path(), target).unwrap();
                }
            }
        }
        let root =
            std::env::temp_dir().join(format!("argusnet-schema-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        copy(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scene.smartscene"),
            &root,
        );
        root
    }

    fn edit_json(path: &Path, key: &str, value: &str) {
        let mut doc: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        doc[key] = json!(value);
        fs::write(path, doc.to_string()).unwrap();
    }

    #[test]
    fn a_scene_or_style_of_another_version_is_refused_by_name() {
        let scene = scene_copy("scene-version");
        assert!(
            ScenePackage::load(&scene).is_ok(),
            "the committed scene loads"
        );
        edit_json(
            &scene.join("scene_manifest.json"),
            "format_version",
            "smartscene-v2",
        );
        let error = ScenePackage::load(&scene).unwrap_err().to_string();
        assert!(error.contains("smartscene-v2"), "{error}");

        let scene = scene_copy("style-version");
        edit_json(
            &scene.join("metadata/style.json"),
            "style_version",
            "smartstyle-v9",
        );
        let error = format!("{:#}", ScenePackage::load(&scene).unwrap_err());
        assert!(error.contains("smartstyle-v9"), "{error}");
    }

    #[test]
    fn safe_manifest_path_resolves_contained_relative_path() {
        let root = Path::new("/scene-root");
        let resolved = safe_manifest_path(root, "metadata/style.json", "metadata.style").unwrap();
        assert_eq!(resolved, root.join("metadata/style.json"));
    }

    #[test]
    fn safe_manifest_path_rejects_absolute_path() {
        let root = Path::new("/scene-root");
        assert!(safe_manifest_path(root, "/etc/passwd", "metadata.style").is_err());
    }

    #[test]
    fn safe_manifest_path_rejects_dotdot_escape() {
        let root = Path::new("/scene-root");
        assert!(
            safe_manifest_path(root, "../../../../etc/passwd", "metadata.environment").is_err()
        );
    }

    #[test]
    fn load_scene_package_reads_manifest_style_and_environment() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("argusnet-schema-{suffix}"));
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(
            root.join("scene_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "format_version": "smartscene-v1",
                "scene_id": "demo",
                "bounds_xy_m": {
                    "x_min_m": -10.0,
                    "x_max_m": 10.0,
                    "y_min_m": -5.0,
                    "y_max_m": 5.0
                },
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "source_crs_id": "EPSG:32611",
                "layers": [],
                "replay": null,
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json"
                },
                "build": {"source_kind": "synthetic"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/style.json"),
            serde_json::to_string_pretty(&json!({
                "style_version": "smartstyle-v1",
                "layers": [{
                    "id": "terrain-base",
                    "semantic_kind": "terrain",
                    "color_rgba": [1.0, 1.0, 1.0, 1.0],
                    "opacity": 1.0,
                    "elevation_mode": "terrain-draped",
                    "draw_order": 0,
                    "default_visibility": true
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/environment.json"),
            serde_json::to_string_pretty(&json!({
                "bounds_xy_m": {
                    "x_min_m": -10.0,
                    "x_max_m": 10.0,
                    "y_min_m": -5.0,
                    "y_max_m": 5.0
                },
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "terrain_summary": {"min_height_m": 10.0, "max_height_m": 42.0}
            }))
            .unwrap(),
        )
        .unwrap();

        let package = ScenePackage::load(&root).unwrap();
        assert_eq!(package.manifest.scene_id, "demo");
        assert_eq!(package.style.layers.len(), 1);
        assert!(package.replay.is_none());
        assert_eq!(
            package
                .environment
                .terrain_summary
                .as_ref()
                .unwrap()
                .max_height_m,
            Some(42.0)
        );

        let _ = fs::remove_dir_all(root);
    }

    /// Regression: both bundled sample scenes declared `replay.path` pointing at a
    /// `replay/replay.json` that was never packaged. `load` swallowed the miss and
    /// returned `replay: None`, so `argusnet live --scene ...` -- the command in
    /// `docs/LIVE_OPERATIONS.md` -- opened on an empty timeline with no diagnostic.
    #[test]
    fn load_scene_package_rejects_a_declared_but_missing_replay() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("argusnet-schema-missing-replay-{suffix}"));
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(
            root.join("scene_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "format_version": "smartscene-v1",
                "scene_id": "demo",
                "bounds_xy_m": {"x_min_m": -10.0, "x_max_m": 10.0, "y_min_m": -5.0, "y_max_m": 5.0},
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "source_crs_id": "local-enu",
                "layers": [],
                "replay": {"path": "replay/replay.json"},
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json"
                },
                "build": {"source_kind": "synthetic"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/style.json"),
            serde_json::to_string_pretty(&json!({"style_version": "smartstyle-v1", "layers": []}))
                .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/environment.json"),
            serde_json::to_string_pretty(&json!({
                "bounds_xy_m": {"x_min_m": -10.0, "x_max_m": 10.0, "y_min_m": -5.0, "y_max_m": 5.0},
                "runtime_crs": {"runtime_crs_id": "local-enu"}
            }))
            .unwrap(),
        )
        .unwrap();

        let error = ScenePackage::load(&root).expect_err("missing replay must not load silently");
        let message = format!("{error:#}");
        assert!(
            message.contains("replay.path"),
            "error should name the offending manifest key, got: {message}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_scene_package_rejects_environment_path_traversal() {
        // Regression test for a scene package whose metadata.environment
        // escapes the scene root: ScenePackage::load must reject it before
        // ever reading the escaping path, not just when a layer is spawned.
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("argusnet-schema-traversal-{suffix}"));
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(
            root.join("scene_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "format_version": "smartscene-v1",
                "scene_id": "evil",
                "bounds_xy_m": {
                    "x_min_m": -10.0,
                    "x_max_m": 10.0,
                    "y_min_m": -5.0,
                    "y_max_m": 5.0
                },
                "runtime_crs": {"runtime_crs_id": "local-enu"},
                "source_crs_id": "EPSG:32611",
                "layers": [],
                "replay": null,
                "metadata": {
                    "environment": "../../../../etc/passwd",
                    "style": "metadata/style.json"
                },
                "build": {"source_kind": "synthetic"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("metadata/style.json"),
            serde_json::to_string_pretty(&json!({
                "style_version": "smartstyle-v1",
                "layers": []
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(ScenePackage::load(&root).is_err());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn an_unmapped_scene_loads_with_bounds_and_no_layers() {
        let dir =
            std::env::temp_dir().join(format!("argusnet-unmapped-test-{}", std::process::id()));
        let path =
            super::write_unmapped_scene(&dir, "demo", [-100.0, 200.0, -50.0, 150.0], (10.0, 160.0))
                .expect("written");
        let package = ScenePackage::load(&path).expect("loads");
        assert!(package.manifest.layers.is_empty());
        assert_eq!(package.manifest.bounds_xy_m.x_max_m, 200.0);
        assert!(package.style_for("tracks").is_some());
        assert!(
            !package.is_synthetic_source(),
            "no hot-swap for a self-built world"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
