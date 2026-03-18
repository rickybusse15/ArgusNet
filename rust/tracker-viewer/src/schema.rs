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

        let style: StyleDocument = read_json(root.join(&manifest.metadata.style))?;
        if style.style_version != STYLE_FORMAT_VERSION {
            anyhow::bail!(
                "unsupported style format {:?}; expected {}",
                style.style_version,
                STYLE_FORMAT_VERSION
            );
        }

        let environment: EnvironmentDocument =
            read_json(root.join(&manifest.metadata.environment))?;
        let replay = manifest
            .replay
            .as_ref()
            .map(|entry| read_json::<Value>(root.join(&entry.path)))
            .transpose()?;

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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::ScenePackage;

    #[test]
    fn load_scene_package_reads_manifest_style_and_environment() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tracker-viewer-schema-{suffix}"));
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
}
