//! What is remembered of a site between missions (F6), and what has changed.
//!
//! A mission's map is kept as a **revision**: an immutable document, named by the
//! hash of its content, holding each cell's ground, structure, what the mission
//! claims to know of them, and when. Revisions are only ever added. A later
//! mission compares its own map with the last one and says what differs, cell by
//! cell, only where both missions knew the cell and the difference is more than
//! both claims allow.
//!
//! Three answers, not two. A difference within the claims is **no change**. One
//! past them is a **change**. Where the missions' own uncertainty of *where* they
//! were is more than half a cell, the maps cannot be laid on each other to say
//! which, and the place is **undetermined**: never reported as changed, nor as
//! unchanged.

use std::path::{Path, PathBuf};

use blake2::Digest;
use serde::{Deserialize, Serialize};

pub const REVISION_FORMAT: &str = "argusnet.site-revision";
pub const REVISION_VERSION: u32 = 1;
pub const CHANGES_FORMAT: &str = "argusnet.site-changes";

/// One cell as a mission left it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteCell {
    /// Ground height, and its claimed one sigma, metres. `None` unmeasured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_m: Option<f64>,
    /// Where its returns are, one sigma, metres; `None` when they were placed
    /// by the true position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placed_sigma_m: Option<f64>,
    /// The top of a structure believed there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_top_m: Option<f64>,
    /// Ground returns behind it.
    #[serde(default)]
    pub returns: u32,
}

/// The grid a revision is laid on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteGrid {
    pub origin_m: [f64; 2],
    pub resolution_m: f64,
    pub nx: usize,
    pub ny: usize,
}

/// One mission's map of a site. Immutable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteRevision {
    pub format: String,
    pub version: u32,
    pub site_id: String,
    /// The frame its coordinates are in, where the world has one, so a map is
    /// laid only on maps in the same frame. Not the world's own identity: what
    /// is there changes, and that is what is looked for.
    pub frame_id: Option<String>,
    /// What made it: the profile, seed and scenario.
    pub mission: serde_json::Value,
    /// The revision this one followed.
    pub parent: Option<String>,
    pub grid: SiteGrid,
    pub cells: Vec<SiteCell>,
}

impl SiteRevision {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != REVISION_FORMAT || self.version != REVISION_VERSION {
            return Err(format!(
                "not a site revision this build reads: {} version {}",
                self.format, self.version
            ));
        }
        check_id(&self.site_id)?;
        let grid = &self.grid;
        if !(grid.resolution_m.is_finite() && grid.resolution_m > 0.0)
            || !grid.origin_m.iter().all(|v| v.is_finite())
        {
            return Err("a site revision's grid is malformed".into());
        }
        if grid.nx.checked_mul(grid.ny) != Some(self.cells.len()) {
            return Err(format!(
                "a {} by {} grid with {} cells",
                grid.nx,
                grid.ny,
                self.cells.len()
            ));
        }
        for cell in &self.cells {
            let heights = [cell.ground_m, cell.structure_top_m];
            let sigmas = [cell.sigma_m, cell.placed_sigma_m];
            if heights.iter().flatten().any(|v| !v.is_finite())
                || sigmas
                    .iter()
                    .flatten()
                    .any(|v| !(v.is_finite() && *v >= 0.0))
            {
                return Err("a site cell holds a height or sigma that is not one".into());
            }
        }
        Ok(())
    }

    /// Its content's name.
    pub fn hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a revision serialises");
        blake2::Blake2b::<blake2::digest::consts::U32>::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    pub fn known_cells(&self) -> usize {
        self.cells.iter().filter(|c| c.ground_m.is_some()).count()
    }
}

/// A site's id is a file name and nothing else.
fn check_id(id: &str) -> Result<(), String> {
    let fits = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if fits {
        Ok(())
    } else {
        Err(format!(
            "a site id is up to 64 letters, digits, - and _, not {id:?}"
        ))
    }
}

/// What a site's index says of one revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    pub hash: String,
    pub parent: Option<String>,
    pub frame_id: Option<String>,
    pub mission: serde_json::Value,
    pub known_cells: usize,
}

/// A directory of sites.
#[derive(Debug, Clone)]
pub struct SiteMemory {
    root: PathBuf,
}

impl SiteMemory {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn site_dir(&self, site_id: &str) -> Result<PathBuf, String> {
        check_id(site_id)?;
        Ok(self.root.join(site_id))
    }

    /// The sites held, by id.
    pub fn sites(&self) -> Result<Vec<String>, String> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(&self.root).map_err(|e| at(&self.root, e))? {
            let entry = entry.map_err(|e| at(&self.root, e))?;
            if entry.path().join("index.json").is_file() {
                if let Some(id) = entry.file_name().to_str() {
                    ids.push(id.to_string());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }

    /// A site's revisions, oldest first. None for a site not held.
    pub fn index(&self, site_id: &str) -> Result<Vec<IndexEntry>, String> {
        let path = self.site_dir(site_id)?.join("index.json");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&path).map_err(|e| at(&path, e))?;
        serde_json::from_str(&text).map_err(|e| at(&path, e))
    }

    /// The revision of that hash, checked against it.
    pub fn read(&self, site_id: &str, hash: &str) -> Result<SiteRevision, String> {
        if hash.is_empty() || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("{hash:?} is not a revision's hash"));
        }
        let path = self.site_dir(site_id)?.join(format!("{hash}.json"));
        let text = std::fs::read_to_string(&path).map_err(|e| at(&path, e))?;
        let revision: SiteRevision = serde_json::from_str(&text).map_err(|e| at(&path, e))?;
        revision.validate().map_err(|e| at(&path, e))?;
        if revision.hash() != hash || revision.site_id != site_id {
            return Err(format!(
                "{}: its content is not what its name says",
                path.display()
            ));
        }
        Ok(revision)
    }

    /// The last revision, if there is one.
    pub fn latest(&self, site_id: &str) -> Result<Option<SiteRevision>, String> {
        match self.index(site_id)?.last() {
            Some(entry) => self.read(site_id, &entry.hash).map(Some),
            None => Ok(None),
        }
    }

    /// Keep `revision`, which must follow the last one. Its file is written
    /// whole before the index names it, so a stop between leaves the site as it
    /// was.
    pub fn add(&self, revision: &SiteRevision) -> Result<String, String> {
        revision.validate()?;
        let dir = self.site_dir(&revision.site_id)?;
        let mut index = self.index(&revision.site_id)?;
        let last = index.last().map(|entry| entry.hash.clone());
        if revision.parent != last {
            return Err(format!(
                "the revision follows {:?}, and the site's last is {last:?}",
                revision.parent
            ));
        }
        if let Some(entry) = index.last() {
            if entry.frame_id != revision.frame_id {
                return Err(format!(
                    "site {} is in frame {:?}, not {:?}",
                    revision.site_id, entry.frame_id, revision.frame_id
                ));
            }
        }
        let hash = revision.hash();
        std::fs::create_dir_all(&dir).map_err(|e| at(&dir, e))?;
        let text = serde_json::to_string(revision).map_err(|e| e.to_string())?;
        write_whole(&dir.join(format!("{hash}.json")), &text)?;
        index.push(IndexEntry {
            hash: hash.clone(),
            parent: revision.parent.clone(),
            frame_id: revision.frame_id.clone(),
            mission: revision.mission.clone(),
            known_cells: revision.known_cells(),
        });
        let text = serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?;
        write_whole(&dir.join("index.json"), &(text + "\n"))?;
        Ok(hash)
    }
}

impl SiteMemory {
    /// Keep what a revision changed beside it, by its hash.
    pub fn write_changes(&self, site_id: &str, changes: &Changes) -> Result<PathBuf, String> {
        let path = self
            .site_dir(site_id)?
            .join(format!("{}.changes.json", changes.after));
        let text = serde_json::to_string_pretty(changes).map_err(|e| e.to_string())?;
        write_whole(&path, &(text + "\n"))?;
        Ok(path)
    }
}

fn at(path: &Path, error: impl std::fmt::Display) -> String {
    format!("{}: {error}", path.display())
}

fn write_whole(path: &Path, text: &str) -> Result<(), String> {
    let part = path.with_extension("part");
    std::fs::write(&part, text).map_err(|e| at(&part, e))?;
    std::fs::rename(&part, path).map_err(|e| at(path, e))
}

/// How two revisions are compared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompareConfig {
    /// How many sigma of both claims together a difference must pass.
    pub sigmas: f64,
    /// And by how much at least, metres, for ground.
    pub min_ground_m: f64,
    /// For a structure's height.
    pub min_structure_m: f64,
    /// The fewest cells, touching, worth reporting.
    pub min_cells: usize,
    /// The fewest ground returns a cell is known by.
    pub min_returns: u32,
    /// The fewest that say no structure stands in a cell: one can fill part of
    /// it and be missed by a thin scatter over the rest.
    pub min_absent_returns: u32,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            sigmas: 4.0,
            min_ground_m: 1.0,
            min_structure_m: 2.0,
            min_cells: 2,
            min_returns: 4,
            min_absent_returns: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    StructureAppeared,
    StructureRemoved,
    StructureRaised,
    StructureLowered,
    GroundRaised,
    GroundLowered,
}

/// Touching cells that changed the same way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    pub change_type: ChangeType,
    pub cells: usize,
    /// The middle of it, and its bounds `[x0, y0, x1, y1]`, metres.
    pub location_m: [f64; 3],
    pub bounds_m: [f64; 4],
    /// The mean difference of height, metres: new less old. For a structure
    /// that appeared, how far it stands over the old ground; removed, less how
    /// far it stood.
    pub difference_m: f64,
    /// The least, over its cells, of the difference in sigmas of both claims.
    pub least_sigmas: f64,
}

/// What differs between two revisions, and what could not be said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Changes {
    pub format: String,
    pub site_id: String,
    pub before: String,
    pub after: String,
    pub config: CompareConfig,
    /// Cells known by both.
    pub compared_cells: usize,
    /// Cells both saw, and could not be laid on each other, or where one saw a
    /// structure the other saw too little to say was not there.
    pub undetermined_cells: usize,
    /// Cells only one of them knows.
    pub one_sided_cells: usize,
    pub changes: Vec<Change>,
}

/// What differs from `before` in `after`.
pub fn compare(
    before: &SiteRevision,
    after: &SiteRevision,
    config: &CompareConfig,
) -> Result<Changes, String> {
    if before.site_id != after.site_id || before.frame_id != after.frame_id {
        return Err("two sites are not compared".into());
    }
    if before.grid != after.grid {
        return Err(format!(
            "the maps are laid on different grids: {:?} and {:?}",
            before.grid, after.grid
        ));
    }
    let grid = &before.grid;
    let (nx, ny) = (grid.nx, grid.ny);
    let known = |cell: &SiteCell| cell.ground_m.is_some() && cell.returns >= config.min_returns;
    let placed = |cell: &SiteCell| cell.placed_sigma_m.unwrap_or(0.0);
    let near = |k: usize| {
        let (i, j) = ((k % nx) as i64, (k / nx) as i64);
        (-1..=1)
            .flat_map(move |dj| (-1..=1).map(move |di| (i + di, j + dj)))
            .filter(move |&(i, j)| i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny)
            .map(move |(i, j)| j as usize * nx + i as usize)
    };
    // A structure is there if it is in the cell or beside it: the two missions
    // saw its edge from different sides.
    let structure_by = |map: &SiteRevision, k: usize| {
        near(k)
            .filter_map(|n| map.cells[n].structure_top_m)
            .fold(None, |top: Option<f64>, t| {
                Some(top.map_or(t, |b| b.max(t)))
            })
    };
    let mut kind: Vec<Option<(ChangeType, f64, f64)>> = vec![None; nx * ny];
    let (mut compared, mut undetermined, mut one_sided) = (0, 0, 0);
    for (k, slot) in kind.iter_mut().enumerate() {
        let (a, b) = (&before.cells[k], &after.cells[k]);
        // A structure with no ground seen in its cell is known by it.
        let seen = |cell: &SiteCell| known(cell) || cell.structure_top_m.is_some();
        match (seen(a), seen(b)) {
            (true, true) => {}
            (false, false) => continue,
            _ => {
                one_sided += 1;
                continue;
            }
        }
        if placed(a).hypot(placed(b)) > 0.5 * grid.resolution_m {
            undetermined += 1;
            continue;
        }
        compared += 1;
        let sigma = a
            .sigma_m
            .unwrap_or(0.0)
            .hypot(b.sigma_m.unwrap_or(0.0))
            .max(1e-3);
        let judged = |difference: f64, least: f64| {
            let sigmas = difference.abs() / sigma;
            (difference.abs() >= least && sigmas >= config.sigmas).then_some((difference, sigmas))
        };
        let (was, is) = (a.structure_top_m, b.structure_top_m);
        // What stands in the cell or beside it: a roof seen in part, or from
        // the other side, is the roof.
        let (was_by, is_by) = (structure_by(before, k), structure_by(after, k));
        let absent = |cell: &SiteCell| known(cell) && cell.returns >= config.min_absent_returns;
        *slot = match (was, is) {
            (Some(was), Some(is)) => {
                let raised = is - was_by.expect("it stands in the cell").max(was);
                let lowered = is_by.expect("it stands in the cell").max(is) - was;
                if raised > 0.0 {
                    judged(raised, config.min_structure_m)
                        .map(|(d, s)| (ChangeType::StructureRaised, d, s))
                } else if lowered < 0.0 {
                    judged(lowered, config.min_structure_m)
                        .map(|(d, s)| (ChangeType::StructureLowered, d, s))
                } else {
                    None
                }
            }
            (None, Some(is)) if was_by.is_none() => {
                if absent(a) {
                    judged(is - a.ground_m.expect("known"), config.min_structure_m)
                        .filter(|(d, _)| *d > 0.0)
                        .map(|(d, s)| (ChangeType::StructureAppeared, d, s))
                } else {
                    undetermined += 1;
                    compared -= 1;
                    None
                }
            }
            (Some(was), None) if is_by.is_none() => {
                if absent(b) {
                    judged(b.ground_m.expect("known") - was, config.min_structure_m)
                        .filter(|(d, _)| *d < 0.0)
                        .map(|(d, s)| (ChangeType::StructureRemoved, d, s))
                } else {
                    undetermined += 1;
                    compared -= 1;
                    None
                }
            }
            (None, None) if known(a) && known(b) => {
                let difference = b.ground_m.expect("known") - a.ground_m.expect("known");
                judged(difference, config.min_ground_m).map(|(d, s)| {
                    let up = if d > 0.0 {
                        ChangeType::GroundRaised
                    } else {
                        ChangeType::GroundLowered
                    };
                    (up, d, s)
                })
            }
            _ => None,
        };
    }
    // Beside a structure either mission knows, ground is part wall: its
    // difference says which side was seen.
    for (k, slot) in kind.iter_mut().enumerate() {
        if matches!(
            *slot,
            Some((ChangeType::GroundRaised | ChangeType::GroundLowered, ..))
        ) && (structure_by(before, k).is_some() || structure_by(after, k).is_some())
        {
            *slot = None;
        }
    }
    let mut taken = vec![false; nx * ny];
    let mut changes = Vec::new();
    for first in 0..nx * ny {
        let Some((change_type, ..)) = kind[first] else {
            continue;
        };
        if taken[first] {
            continue;
        }
        let mut cells = vec![first];
        taken[first] = true;
        let mut next = 0;
        while next < cells.len() {
            for n in near(cells[next]) {
                if !taken[n] && kind[n].is_some_and(|(t, ..)| t == change_type) {
                    taken[n] = true;
                    cells.push(n);
                }
            }
            next += 1;
        }
        if cells.len() < config.min_cells {
            continue;
        }
        let centre = |k: usize| {
            [
                grid.origin_m[0] + ((k % nx) as f64 + 0.5) * grid.resolution_m,
                grid.origin_m[1] + ((k / nx) as f64 + 0.5) * grid.resolution_m,
            ]
        };
        let count = cells.len() as f64;
        let mean = |f: &dyn Fn(usize) -> f64| cells.iter().map(|&k| f(k)).sum::<f64>() / count;
        let half = 0.5 * grid.resolution_m;
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for &k in &cells {
            let [x, y] = centre(k);
            bounds = [
                bounds[0].min(x - half),
                bounds[1].min(y - half),
                bounds[2].max(x + half),
                bounds[3].max(y + half),
            ];
        }
        let height = |k: usize| {
            let cell = &after.cells[k];
            cell.structure_top_m.or(cell.ground_m).unwrap_or_default()
        };
        changes.push(Change {
            change_type,
            cells: cells.len(),
            location_m: [
                mean(&|k| centre(k)[0]),
                mean(&|k| centre(k)[1]),
                mean(&height),
            ],
            bounds_m: bounds,
            difference_m: mean(&|k| kind[k].expect("in the region").1),
            least_sigmas: cells
                .iter()
                .map(|&k| kind[k].expect("in the region").2)
                .fold(f64::INFINITY, f64::min),
        });
    }
    changes.sort_by(|a, b| {
        b.cells
            .cmp(&a.cells)
            .then(a.change_type.cmp(&b.change_type))
            .then(a.location_m[0].total_cmp(&b.location_m[0]))
            .then(a.location_m[1].total_cmp(&b.location_m[1]))
    });
    Ok(Changes {
        format: CHANGES_FORMAT.into(),
        site_id: after.site_id.clone(),
        before: before.hash(),
        after: after.hash(),
        config: config.clone(),
        compared_cells: compared,
        undetermined_cells: undetermined,
        one_sided_cells: one_sided,
        changes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 20;

    fn flat(parent: Option<String>) -> SiteRevision {
        SiteRevision {
            format: REVISION_FORMAT.into(),
            version: REVISION_VERSION,
            site_id: "site".into(),
            frame_id: Some("frame".into()),
            mission: serde_json::json!({"seed": 7}),
            parent,
            grid: SiteGrid {
                origin_m: [-100.0, -100.0],
                resolution_m: 10.0,
                nx: N,
                ny: N,
            },
            cells: vec![
                SiteCell {
                    ground_m: Some(50.0),
                    sigma_m: Some(0.1),
                    placed_sigma_m: Some(0.1),
                    structure_top_m: None,
                    returns: 20,
                };
                N * N
            ],
        }
    }

    fn cell(i: usize, j: usize) -> usize {
        j * N + i
    }

    fn built(map: &mut SiteRevision, cells: &[(usize, usize)], top: f64) {
        for &(i, j) in cells {
            map.cells[cell(i, j)].structure_top_m = Some(top);
        }
    }

    fn found(before: &SiteRevision, after: &SiteRevision) -> Vec<(ChangeType, usize)> {
        compare(before, after, &CompareConfig::default())
            .unwrap()
            .changes
            .iter()
            .map(|change| (change.change_type, change.cells))
            .collect()
    }

    #[test]
    fn what_both_claims_allow_is_no_change() {
        let before = flat(None);
        let mut after = flat(None);
        for (k, cell) in after.cells.iter_mut().enumerate() {
            cell.ground_m = Some(50.0 + if k % 2 == 0 { 0.4 } else { -0.4 });
        }
        assert_eq!(found(&before, &after), []);
        // A metre and a half, but known to a metre: allowed.
        let mut unsure = flat(None);
        for cell in &mut unsure.cells {
            cell.ground_m = Some(51.5);
            cell.sigma_m = Some(1.0);
        }
        assert_eq!(found(&before, &unsure), []);
    }

    #[test]
    fn a_building_put_up_and_one_taken_down_are_found_where_they_are() {
        let (mut before, mut after) = (flat(None), flat(None));
        built(&mut before, &[(3, 3), (4, 3), (3, 4), (4, 4)], 58.0);
        built(&mut after, &[(12, 10), (13, 10)], 60.0);
        let changes = compare(&before, &after, &CompareConfig::default()).unwrap();
        assert_eq!(changes.changes.len(), 2, "{changes:?}");
        let gone = &changes.changes[0];
        assert_eq!(
            (gone.change_type, gone.cells),
            (ChangeType::StructureRemoved, 4)
        );
        assert_eq!(gone.bounds_m, [-70.0, -70.0, -50.0, -50.0]);
        assert!((gone.difference_m + 8.0).abs() < 1e-9);
        let new = &changes.changes[1];
        assert_eq!(new.change_type, ChangeType::StructureAppeared);
        assert_eq!(new.location_m, [30.0, 5.0, 60.0]);
        assert!((new.difference_m - 10.0).abs() < 1e-9);
        assert!(new.least_sigmas > 50.0);
    }

    /// The same building, its edge seen a cell over, is the same building.
    #[test]
    fn an_edge_seen_from_the_other_side_is_no_change() {
        let (mut before, mut after) = (flat(None), flat(None));
        built(&mut before, &[(3, 3), (4, 3), (3, 4), (4, 4)], 58.0);
        built(&mut after, &[(4, 3), (5, 3), (4, 4), (5, 4)], 58.3);
        for &(i, j) in &[(3, 3), (5, 5), (2, 3)] {
            after.cells[cell(i, j)].ground_m = Some(53.0);
        }
        assert_eq!(found(&before, &after), []);
        // Two storeys more is a change to it.
        let mut raised = flat(None);
        built(&mut raised, &[(3, 3), (4, 3), (3, 4), (4, 4)], 64.0);
        assert_eq!(found(&before, &raised), [(ChangeType::StructureRaised, 4)]);
    }

    /// West Pleasant View, 7 against 42: a house in the corner of two cells,
    /// and 8 and 12 ground returns over the rest of them the second time.
    #[test]
    fn a_structure_a_thin_scatter_missed_is_undetermined() {
        let mut before = flat(None);
        built(&mut before, &[(3, 3), (4, 3)], 60.0);
        let mut after = flat(None);
        after.cells[cell(3, 3)].returns = 8;
        after.cells[cell(4, 3)].returns = 12;
        for (a, b) in [(&before, &after), (&after, &before)] {
            let changes = compare(a, b, &CompareConfig::default()).unwrap();
            assert_eq!(changes.changes, []);
            assert_eq!(changes.undetermined_cells, 2);
        }
        // Seen well, it is gone.
        assert_eq!(
            found(&before, &flat(None)),
            [(ChangeType::StructureRemoved, 2)]
        );
    }

    /// A roof seen in part is the roof: lower in one cell, as high beside it.
    #[test]
    fn a_roof_seen_in_part_is_no_lower() {
        let (mut before, mut after) = (flat(None), flat(None));
        built(&mut before, &[(3, 3), (4, 3), (3, 4), (4, 4)], 60.0);
        built(&mut after, &[(3, 3), (4, 3)], 53.0);
        built(&mut after, &[(3, 4), (4, 4)], 60.0);
        assert_eq!(found(&before, &after), []);
        assert_eq!(found(&after, &before), []);
    }

    #[test]
    fn earth_moved_is_found_and_one_cell_is_not_enough() {
        let before = flat(None);
        let mut after = flat(None);
        for i in 5..9 {
            after.cells[cell(i, 8)].ground_m = Some(47.0);
        }
        after.cells[cell(15, 15)].ground_m = Some(58.0);
        assert_eq!(found(&before, &after), [(ChangeType::GroundLowered, 4)]);
    }

    /// A mission that did not know where it was says neither changed nor not.
    #[test]
    fn what_cannot_be_laid_on_the_other_is_undetermined() {
        let before = flat(None);
        let mut lost = flat(None);
        built(&mut lost, &[(12, 10), (13, 10)], 60.0);
        for cell in &mut lost.cells {
            cell.placed_sigma_m = Some(6.0);
        }
        let changes = compare(&before, &lost, &CompareConfig::default()).unwrap();
        assert_eq!(changes.changes, []);
        assert_eq!(changes.compared_cells, 0);
        assert_eq!(changes.undetermined_cells, N * N);
    }

    #[test]
    fn a_cell_only_one_knows_is_not_compared() {
        let mut before = flat(None);
        for i in 0..N {
            before.cells[cell(i, 0)] = SiteCell::default();
            before.cells[cell(i, 1)].returns = 1;
        }
        let mut after = flat(None);
        built(&mut after, &[(5, 0), (6, 0)], 70.0);
        let changes = compare(&before, &after, &CompareConfig::default()).unwrap();
        assert_eq!(changes.changes, []);
        assert_eq!(changes.one_sided_cells, 2 * N);
    }

    #[test]
    fn different_grids_and_sites_are_refused() {
        let before = flat(None);
        let mut moved = flat(None);
        moved.grid.origin_m[0] += 1.0;
        assert!(compare(&before, &moved, &CompareConfig::default()).is_err());
        let mut other = flat(None);
        other.frame_id = None;
        assert!(compare(&before, &other, &CompareConfig::default()).is_err());
    }

    fn memory(name: &str) -> SiteMemory {
        let root =
            std::env::temp_dir().join(format!("argusnet-site-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        SiteMemory::new(root)
    }

    #[test]
    fn a_site_is_remembered_across_reopening() {
        let memory = memory("kept");
        assert_eq!(memory.sites().unwrap(), Vec::<String>::new());
        assert_eq!(memory.latest("site").unwrap(), None);
        let first = flat(None);
        let hash = memory.add(&first).unwrap();
        let mut second = flat(Some(hash.clone()));
        built(&mut second, &[(12, 10), (13, 10)], 60.0);
        memory.add(&second).unwrap();
        let reopened = SiteMemory::new(memory.root.clone());
        assert_eq!(reopened.sites().unwrap(), ["site"]);
        assert_eq!(reopened.latest("site").unwrap(), Some(second));
        assert_eq!(reopened.read("site", &hash).unwrap(), first);
        assert_eq!(reopened.index("site").unwrap().len(), 2);
        std::fs::remove_dir_all(&memory.root).unwrap();
    }

    #[test]
    fn a_site_s_history_is_only_added_to_in_order() {
        let memory = memory("order");
        let hash = memory.add(&flat(None)).unwrap();
        assert!(memory.add(&flat(None)).is_err(), "follows nothing");
        let mut other = flat(Some(hash.clone()));
        other.frame_id = Some("another".into());
        assert!(memory.add(&other).is_err(), "another frame");
        // A file changed under its name is not read.
        let path = memory.root.join("site").join(format!("{hash}.json"));
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replacen("50.0", "51.0", 1)).unwrap();
        assert!(memory
            .read("site", &hash)
            .unwrap_err()
            .contains("name says"));
        for id in ["", "../up", "a/b", "."] {
            assert!(memory.latest(id).is_err(), "{id:?}");
        }
        assert!(memory.read("site", "../index").is_err());
        std::fs::remove_dir_all(&memory.root).unwrap();
    }

    #[test]
    fn a_malformed_revision_is_refused() {
        let break_it = |f: &dyn Fn(&mut SiteRevision)| {
            let mut revision = flat(None);
            f(&mut revision);
            revision.validate()
        };
        assert!(break_it(&|_| {}).is_ok());
        assert!(break_it(&|r| r.version = 2).is_err());
        assert!(break_it(&|r| r.cells.truncate(3)).is_err());
        assert!(break_it(&|r| r.grid.resolution_m = 0.0).is_err());
        assert!(break_it(&|r| r.cells[0].ground_m = Some(f64::NAN)).is_err());
        assert!(break_it(&|r| r.cells[0].sigma_m = Some(-1.0)).is_err());
    }
}
