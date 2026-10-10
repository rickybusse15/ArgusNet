//! The self-built map saved as a scene package (`smartscene-v1`), so a map
//! the drones made is a world the viewer can open again like any other.
//!
//! The package is the layout `argusnet build-scene` writes, read by the same
//! loader:
//!
//! - `terrain/terrain-base.glb`: the measured ground, as one surface with a
//!   skirt round its edge ([`crate::belief_mesh::ground_mesh`]).
//! - `overlays/buildings.glb`: the believed structures, when there are any.
//! - `metadata/style.json`, `metadata/environment.json`: the styles and the
//!   bounds and height range.
//! - `metadata/belief_map.json`: the map itself (grid layout and cell rows),
//!   so the belief behind the meshes is kept, not only its picture.
//! - `replay/replay.json`: the run that built the map, when the export is
//!   made from a replay ([`export_replay_scene`]).
//! - `scene_manifest.json`, with `build.source_kind = "self-built"` and the
//!   run that built it.
//!
//! GLB files follow `argusnet/world/_glb.py`: one mesh, one primitive with
//! `POSITION`, `NORMAL` and `COLOR_0` (linear RGBA floats), `u32` indices,
//! one white double-sided material; metres, ENU, Z up.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::belief_mesh::{
    ground_mesh, structures_mesh, BeliefCells, GridLayout, GroundColours, MeshData,
};
use crate::belief_world::BeliefCellUpdate;

/// A map as its replay rows leave it: the last row of every cell.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeliefRows(pub BTreeMap<(u32, u32), BeliefCellUpdate>);

impl BeliefRows {
    /// Apply rows in order: each replaces its cell's last.
    pub fn apply<'a>(&mut self, rows: impl IntoIterator<Item = &'a BeliefCellUpdate>) {
        for row in rows {
            self.0.insert((row.0, row.1), row.clone());
        }
    }

    fn get(&self, i: i64, j: i64) -> Option<&BeliefCellUpdate> {
        if i < 0 || j < 0 {
            return None;
        }
        self.0.get(&(i as u32, j as u32))
    }

    pub fn known_cells(&self) -> usize {
        self.0.len()
    }
}

impl BeliefCells for BeliefRows {
    fn ground(&self, i: i64, j: i64) -> Option<f32> {
        self.get(i, j)
            .and_then(|row| row.2)
            .map(|cm| cm as f32 / 100.0)
    }

    fn structure_top(&self, i: i64, j: i64) -> Option<f32> {
        self.get(i, j)
            .and_then(|row| row.3)
            .map(|cm| cm as f32 / 100.0)
    }

    fn structure_cells(&self) -> Vec<(u32, u32)> {
        self.0
            .iter()
            .filter(|(_, row)| row.3.is_some())
            .map(|(ij, _)| *ij)
            .collect()
    }
}

/// The colour a scene's terrain layer is styled with (`terrain-base`).
pub const TERRAIN_SRGB: [f32; 3] = [0.655, 0.710, 0.545];
/// The colour believed structures are drawn in, roof then wall.
pub const STRUCTURE_ROOF_SRGB: [f32; 3] = [0.90, 0.74, 0.86];
pub const STRUCTURE_WALL_SRGB: [f32; 3] = [0.80, 0.66, 0.78];
/// How far the skirt drops below the edge of the known ground, metres.
pub const SKIRT_M: f32 = 6.0;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// What was written, for the caller to report.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneExport {
    pub root: PathBuf,
    pub known_cells: usize,
    pub structure_cells: usize,
    pub ground_triangles: usize,
    pub structure_triangles: usize,
}

/// Write the map in `rows` as a scene package in `dir` (created; must be
/// empty or absent). `provenance` is recorded under `build`.
pub fn export_belief_scene(
    dir: &Path,
    scene_id: &str,
    layout: GridLayout,
    rows: &BeliefRows,
    provenance: Value,
) -> Result<SceneExport, String> {
    if rows.0.values().all(|row| row.2.is_none()) {
        return Err("the map has no measured ground yet: nothing to export".into());
    }
    if dir.exists()
        && std::fs::read_dir(dir)
            .map_err(|error| format!("{}: {error}", dir.display()))?
            .next()
            .is_some()
    {
        return Err(format!(
            "{} is not empty; export into a new directory",
            dir.display()
        ));
    }
    let heights: Vec<f32> = rows
        .0
        .values()
        .filter_map(|row| row.2)
        .map(|cm| cm as f32 / 100.0)
        .collect();
    let (low, high) = heights
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), h| {
            (lo.min(*h), hi.max(*h))
        });

    // Linear vertex colours, as glTF's COLOR_0 is. Cells with few returns are
    // washed toward slate, as the viewer draws them.
    let lin = |c: [f32; 3]| {
        [
            srgb_to_linear(c[0]),
            srgb_to_linear(c[1]),
            srgb_to_linear(c[2]),
            1.0,
        ]
    };
    let corner = |ci: i64, cj: i64| -> [f32; 4] {
        let mut confidence = 0.0;
        let mut n = 0.0;
        for (di, dj) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            if let Some(row) = rows.get(ci + di, cj + dj).filter(|row| row.2.is_some()) {
                confidence += (row.5.clamp(1, 4) as f32 - 1.0) / 3.0;
                n += 1.0;
            }
        }
        let confidence = if n > 0.0 { confidence / n } else { 1.0 };
        let washed = [0.52, 0.58, 0.64];
        lin(std::array::from_fn(|k| {
            washed[k] + (TERRAIN_SRGB[k] - washed[k]) * (0.35 + 0.65 * confidence)
        }))
    };
    let ground = ground_mesh(
        rows,
        layout,
        0..layout.dimensions[0],
        0..layout.dimensions[1],
        low,
        SKIRT_M,
        &GroundColours {
            corner: &corner,
            skirt_top: lin([0.20, 0.22, 0.24]),
            skirt_bottom: lin([0.07, 0.08, 0.09]),
        },
    );
    let structure_colour = |_: i64, _: i64| (lin(STRUCTURE_ROOF_SRGB), lin(STRUCTURE_WALL_SRGB));
    let structures = structures_mesh(rows, layout, &structure_colour);

    let mkdir = |path: PathBuf| {
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))
    };
    mkdir(dir.join("terrain"))?;
    mkdir(dir.join("metadata"))?;
    write_glb(
        &dir.join("terrain/terrain-base.glb"),
        &ground,
        "terrain-base",
    )?;
    let mut layers = vec![json!({
        "id": "terrain-base", "kind": "terrain", "semantic_kind": "terrain",
        "asset_path": "terrain/terrain-base.glb", "style_id": "terrain-base",
    })];
    if !structures.is_empty() {
        mkdir(dir.join("overlays"))?;
        write_glb(
            &dir.join("overlays/buildings.glb"),
            &structures,
            "buildings",
        )?;
        layers.push(json!({
            "id": "buildings", "kind": "overlay", "semantic_kind": "buildings",
            "asset_path": "overlays/buildings.glb", "style_id": "buildings",
        }));
    }

    let [x0, y0] = layout.origin_m;
    let bounds = json!({
        "x_min_m": x0,
        "x_max_m": x0 + layout.dimensions[0] as f32 * layout.resolution_m,
        "y_min_m": y0,
        "y_max_m": y0 + layout.dimensions[1] as f32 * layout.resolution_m,
    });
    let crs = json!({
        "source_crs_id": "local-enu", "runtime_crs_id": "local-enu",
        "origin_geodetic": {"lat_deg": null, "lon_deg": null, "h_m": null},
        "xy_units": "meters", "z_datum": "local",
    });
    let style = |id: &str, kind: &str, rgba: [f32; 4], order: i32| {
        json!({
            "id": id, "semantic_kind": kind, "color_rgba": rgba, "opacity": rgba[3],
            "elevation_mode": "absolute", "draw_order": order, "default_visibility": true,
        })
    };
    let [tr, tg, tb] = TERRAIN_SRGB;
    let [sr, sg, sb] = STRUCTURE_ROOF_SRGB;
    let documents = [
        (
            "metadata/style.json",
            json!({
                "style_version": "smartstyle-v1",
                "layers": [
                    style("terrain-base", "terrain", [tr, tg, tb, 1.0], 0),
                    style("buildings", "buildings", [sr, sg, sb, 1.0], 10),
                    style("tracks", "tracks", [1.0, 0.604, 0.302, 1.0], 100),
                    style("truths", "truths", [0.302, 0.835, 0.906, 1.0], 100),
                    style("nodes", "nodes", [0.953, 0.941, 0.647, 1.0], 100),
                ],
            }),
        ),
        (
            "metadata/environment.json",
            json!({
                "source_kind": "self-built",
                "scene_id": scene_id,
                "source_crs_id": "local-enu",
                "runtime_crs": crs,
                "bounds_xy_m": bounds,
                "terrain_summary": {
                    "kind": "self-built-belief",
                    "base_resolution_m": layout.resolution_m,
                    "min_height_m": low,
                    "max_height_m": high,
                },
            }),
        ),
        (
            "metadata/belief_map.json",
            json!({
                "format": "argusnet.belief-map",
                "version": 1,
                "belief_grid": {
                    "origin_m": layout.origin_m,
                    "resolution_m": layout.resolution_m,
                    "dimensions": layout.dimensions,
                },
                "row_fields": ["i", "j", "ground_cm", "structure_top_cm", "occupancy_pct", "confidence"],
                "rows": rows.0.values().collect::<Vec<_>>(),
            }),
        ),
        (
            "scene_manifest.json",
            json!({
                "format_version": "smartscene-v1",
                "scene_id": scene_id,
                "bounds_xy_m": bounds,
                "runtime_crs": crs,
                "source_crs_id": "local-enu",
                "layers": layers,
                "replay": null,
                "metadata": {
                    "environment": "metadata/environment.json",
                    "style": "metadata/style.json",
                    "belief_map": "metadata/belief_map.json",
                },
                "build": {
                    "source_kind": "self-built",
                    "tool": "argusnet_sim::scene_export",
                    "provenance": provenance,
                },
            }),
        ),
    ];
    for (path, value) in documents {
        // The map's rows are data, one per known cell: written compact.
        let text = if path == "metadata/belief_map.json" {
            serde_json::to_vec(&value)
        } else {
            serde_json::to_vec_pretty(&value)
        }
        .map_err(|error| error.to_string())?;
        std::fs::write(dir.join(path), text)
            .map_err(|error| format!("{}: {error}", dir.join(path).display()))?;
    }
    Ok(SceneExport {
        root: dir.to_path_buf(),
        known_cells: rows.known_cells(),
        structure_cells: rows.structure_cells().len(),
        ground_triangles: ground.indices.len() / 3,
        structure_triangles: structures.indices.len() / 3,
    })
}

/// The grid layout a no-prior replay records in `meta.world_knowledge`.
pub fn replay_layout(replay: &crate::replay::ReplayDocument) -> Result<GridLayout, String> {
    let grid = replay
        .meta
        .scenario
        .get("world_knowledge")
        .and_then(|knowledge| knowledge.get("belief_grid"))
        .ok_or("the run built no map (it was not a --world-knowledge none run)")?;
    let pair = |key: &str| -> Option<[f64; 2]> {
        let v = grid.get(key)?.as_array()?;
        Some([v.first()?.as_f64()?, v.get(1)?.as_f64()?])
    };
    match (
        pair("origin_m"),
        grid.get("resolution_m").and_then(Value::as_f64),
        pair("dimensions"),
    ) {
        (Some(origin), Some(resolution), Some(dimensions)) => Ok(GridLayout {
            origin_m: [origin[0] as f32, origin[1] as f32],
            resolution_m: resolution as f32,
            dimensions: [dimensions[0] as u32, dimensions[1] as u32],
        }),
        _ => Err("meta.world_knowledge.belief_grid is malformed".into()),
    }
}

/// Save the map a no-prior replay ends with: its frames' rows folded in
/// order, on the grid its meta records. The replay goes into the package as
/// well (`replay/replay.json`, as `argusnet build-scene` packages one), so
/// the saved world opens with the run that built it on the timeline.
pub fn export_replay_scene(
    dir: &Path,
    scene_id: &str,
    replay: &crate::replay::ReplayDocument,
    provenance: Value,
) -> Result<SceneExport, String> {
    let layout = replay_layout(replay)?;
    let mut rows = BeliefRows::default();
    for frame in &replay.frames {
        rows.apply(&frame.belief_updates);
    }
    let export = export_belief_scene(dir, scene_id, layout, &rows, provenance)?;
    let json = crate::replay::to_json(replay).map_err(|error| error.to_string())?;
    let replay_dir = dir.join("replay");
    std::fs::create_dir_all(&replay_dir)
        .map_err(|error| format!("{}: {error}", replay_dir.display()))?;
    std::fs::write(replay_dir.join("replay.json"), json)
        .map_err(|error| format!("{}: {error}", replay_dir.display()))?;
    let manifest_path = dir.join("scene_manifest.json");
    let mut manifest: Value = std::fs::read(&manifest_path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))?;
    manifest["replay"] = json!({ "path": "replay/replay.json" });
    let text = serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
    std::fs::write(&manifest_path, text)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    Ok(export)
}

/// Write `mesh` as a binary glTF 2.0 file, laid out as `_glb.py` writes it.
pub fn write_glb(path: &Path, mesh: &MeshData, name: &str) -> Result<(), String> {
    fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values.into_iter().flat_map(f32::to_le_bytes).collect()
    }
    let count = mesh.positions.len();
    let payloads: [(Vec<u8>, u32); 4] = [
        (floats(mesh.positions.iter().flatten().copied()), 34962),
        (floats(mesh.normals.iter().flatten().copied()), 34962),
        (
            floats(mesh.colours.iter().flatten().map(|c| c.clamp(0.0, 1.0))),
            34962,
        ),
        (
            mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
            34963,
        ),
    ];
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    for (payload, target) in &payloads {
        views.push(json!({
            "buffer": 0, "byteOffset": bin.len(), "byteLength": payload.len(), "target": target,
        }));
        bin.extend_from_slice(payload);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &mesh.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let gltf = json!({
        "asset": {"version": "2.0", "generator": "argusnet_sim::scene_export"},
        "scene": 0,
        "scenes": [{"nodes": [0], "name": name}],
        "nodes": [{"mesh": 0, "name": name}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": count, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": mesh.indices.len(), "type": "SCALAR"},
        ],
        "materials": [{
            "name": format!("{name}-material"),
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "doubleSided": true,
        }],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                "indices": 3, "material": 0, "mode": 4,
            }],
        }],
    });
    let mut json_chunk = serde_json::to_vec(&gltf).map_err(|error| error.to_string())?;
    while json_chunk.len() % 4 != 0 {
        json_chunk.push(b' ');
    }
    let total = 12 + 8 + json_chunk.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_chunk);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    std::fs::write(path, out).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> GridLayout {
        GridLayout {
            origin_m: [-50.0, -50.0],
            resolution_m: 10.0,
            dimensions: [10, 10],
        }
    }

    fn rows() -> BeliefRows {
        let mut rows = BeliefRows::default();
        let mut all = Vec::new();
        for i in 2..6 {
            for j in 2..6 {
                all.push(BeliefCellUpdate(
                    i,
                    j,
                    Some(1000 + 100 * i as i32),
                    None,
                    0,
                    3,
                ));
            }
        }
        all.push(BeliefCellUpdate(3, 3, Some(1300), Some(3000), 90, 3));
        rows.apply(&all);
        rows
    }

    #[test]
    fn an_exported_map_is_a_loadable_scene_package() {
        let dir = std::env::temp_dir().join(format!("argusnet-export-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let summary =
            export_belief_scene(&dir, "demo-map", layout(), &rows(), json!({"run": "test"}))
                .expect("exports");
        assert_eq!(summary.known_cells, 16);
        assert_eq!(summary.structure_cells, 1);
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("scene_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["format_version"], "smartscene-v1");
        assert_eq!(manifest["layers"].as_array().unwrap().len(), 2);
        let glb = std::fs::read(dir.join("terrain/terrain-base.glb")).unwrap();
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(
            u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
            glb.len()
        );
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let gltf: Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        assert_eq!(gltf["accessors"][0]["count"], gltf["accessors"][1]["count"]);
        assert_eq!(
            gltf["accessors"][3]["count"].as_u64().unwrap() as usize,
            summary.ground_triangles * 3
        );
        let belief: Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata/belief_map.json")).unwrap())
                .unwrap();
        assert_eq!(belief["rows"].as_array().unwrap().len(), 16);
        // A second export into the same directory is refused, not mixed in.
        assert!(export_belief_scene(&dir, "demo-map", layout(), &rows(), json!({})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_map_with_no_ground_is_not_exported() {
        let mut rows = BeliefRows::default();
        rows.apply(&[BeliefCellUpdate(1, 1, None, Some(2000), 90, 2)]);
        let dir = std::env::temp_dir().join("argusnet-export-empty-test");
        assert!(export_belief_scene(&dir, "x", layout(), &rows, json!({})).is_err());
    }
}
