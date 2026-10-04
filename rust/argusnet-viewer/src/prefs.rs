//! Viewer presentation preferences (P2.4b): camera bookmarks and key remaps.
//!
//! These are the operator's, not the run's. They never enter a replay, a
//! bundle or a checkpoint, and nothing deterministic reads them. They live in
//! one JSON file: `$ARGUSNET_VIEWER_PREFS` if set, else
//! `~/.config/argusnet/viewer.json`. Bookmarks are kept per scene id, so a
//! bookmark from one site never lands the camera in the wrong place on
//! another. The headless renderer reads and writes none of it, so a render
//! does not depend on who ran it.
//!
//! A missing file is the defaults. A file that cannot be read or parsed is
//! reported and ignored, never fatal, and left untouched: bookmarks saved in
//! that session are kept in memory only. Writes go to a temporary file first
//! and are then renamed into place, so a crash cannot leave half a file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// A saved orbit pose.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bookmark {
    pub focus: [f32; 3],
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefsFile {
    /// `scene id -> slot ("1".."9") -> bookmark`.
    #[serde(default)]
    pub bookmarks: BTreeMap<String, BTreeMap<String, Bookmark>>,
    /// `action name -> key name`; see [`crate::bindings`].
    #[serde(default)]
    pub bindings: BTreeMap<String, String>,
    /// The quality tier's name (`low`, `standard`, `high`); see
    /// [`crate::quality`]. Absent means standard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
}

/// The loaded preferences, where they came from, and what went wrong.
#[derive(Resource, Debug, Clone, Default)]
pub struct ViewerPrefs {
    pub file: PrefsFile,
    /// Where to save; `None` means do not persist (headless, tests).
    pub path: Option<PathBuf>,
    pub scene_id: String,
    /// Problems loading the file, or applying its bindings, for the UI.
    pub problems: Vec<String>,
}

/// Where the preferences live, if anywhere.
pub fn default_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ARGUSNET_VIEWER_PREFS") {
        return Some(PathBuf::from(path));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config/argusnet/viewer.json"))
}

impl ViewerPrefs {
    /// Load from `path`. Missing is empty; unreadable or malformed is empty
    /// with the problem recorded.
    pub fn load(path: Option<PathBuf>, scene_id: &str) -> Self {
        let mut prefs = Self {
            path: path.clone(),
            scene_id: scene_id.to_string(),
            ..Self::default()
        };
        let Some(path) = path else {
            return prefs;
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<PrefsFile>(&text) {
                Ok(file) => prefs.file = file,
                Err(error) => {
                    prefs.problems.push(format!("{}: {error}", path.display()));
                    // Keep a malformed file intact until something is saved.
                    prefs.path = None;
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                prefs.problems.push(format!("{}: {error}", path.display()));
                prefs.path = None;
            }
        }
        prefs
    }

    /// The saved quality tier; an unknown name is standard, with the problem
    /// recorded.
    pub fn quality_tier(&mut self) -> crate::quality::QualityTier {
        let Some(name) = self.file.quality.as_deref() else {
            return crate::quality::QualityTier::default();
        };
        match crate::quality::QualityTier::from_name(name) {
            Some(tier) => tier,
            None => {
                self.problems
                    .push(format!("unknown quality tier {name:?}; using standard"));
                crate::quality::QualityTier::default()
            }
        }
    }

    /// Remember the quality tier.
    pub fn set_quality_tier(&mut self, tier: crate::quality::QualityTier) -> Result<(), String> {
        self.file.quality = Some(tier.name().to_string());
        self.save()
    }

    pub fn bookmark(&self, slot: u8) -> Option<Bookmark> {
        self.file
            .bookmarks
            .get(&self.scene_id)
            .and_then(|slots| slots.get(&slot.to_string()))
            .copied()
    }

    /// Store a bookmark for this scene, and write the file if there is one.
    pub fn set_bookmark(&mut self, slot: u8, bookmark: Bookmark) -> Result<(), String> {
        self.file
            .bookmarks
            .entry(self.scene_id.clone())
            .or_default()
            .insert(slot.to_string(), bookmark);
        self.save()
    }

    fn save(&self) -> Result<(), String> {
        let Some(path) = self.path.as_ref() else {
            return Ok(());
        };
        write_atomically(path, &self.file).map_err(|error| format!("{}: {error}", path.display()))
    }
}

fn write_atomically(path: &Path, file: &PrefsFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_string_pretty(file)?)?;
    std::fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("argusnet-prefs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("nested/viewer.json")
    }

    fn mark(x: f32) -> Bookmark {
        Bookmark {
            focus: [x, 2.0, 3.0],
            radius: 400.0,
            yaw: 0.5,
            pitch: 0.9,
        }
    }

    /// Bookmarks persist per scene: saved on one site, recalled there, and
    /// absent on another.
    #[test]
    fn bookmarks_persist_per_scene() {
        let path = scratch("persist");
        let mut site_a = ViewerPrefs::load(Some(path.clone()), "site-a");
        assert_eq!(site_a.bookmark(3), None);
        site_a.set_bookmark(3, mark(1.0)).unwrap();
        let reloaded = ViewerPrefs::load(Some(path.clone()), "site-a");
        assert_eq!(reloaded.bookmark(3), Some(mark(1.0)));
        let site_b = ViewerPrefs::load(Some(path.clone()), "site-b");
        assert_eq!(site_b.bookmark(3), None, "a bookmark leaked across scenes");
        assert!(
            !path.with_extension("json.tmp").exists(),
            "the temporary file was left"
        );
    }

    /// A malformed file is reported and ignored, and not overwritten.
    #[test]
    fn a_malformed_file_is_reported_and_left_alone() {
        let path = scratch("malformed");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ not json").unwrap();
        let mut prefs = ViewerPrefs::load(Some(path.clone()), "site");
        assert_eq!(prefs.problems.len(), 1);
        prefs.set_bookmark(1, mark(0.0)).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
    }

    #[test]
    fn no_path_persists_nothing_and_bindings_round_trip() {
        let mut prefs = ViewerPrefs::load(None, "site");
        prefs.set_bookmark(1, mark(0.0)).unwrap();
        assert_eq!(prefs.bookmark(1), Some(mark(0.0)));
        let file: PrefsFile = serde_json::from_str(r#"{"bindings": {"focus": "KeyZ"}}"#).unwrap();
        assert_eq!(file.bindings["focus"], "KeyZ");
        assert!(file.bookmarks.is_empty());
    }
}
