//! The self-built map as geometry: one place that turns belief cells into
//! triangles, shared by the viewer (which draws it live) and the scene
//! export (which writes it to GLB). Both read the same cells through
//! [`BeliefCells`], so what is saved is what was on screen.
//!
//! - **Ground**: one continuous surface. Its vertices are cell corners, shared
//!   by every known cell that touches them, each at the mean height of those
//!   cells, with a normal from the neighbouring corners. A skirt drops from
//!   every edge where the known map meets the unknown, so the map reads as a
//!   cut-out of the world rather than a sheet with ragged holes.
//! - **Structures**: one solid. Each occupied cell gets a roof at its own top
//!   and a wall on each side whose neighbour is lower or empty, so adjacent
//!   cells merge into a single building outline.
//!
//! Coordinates are the run's: metres, ENU, Z up.

/// The cells a mesh is built from, by grid index. Indices may be negative or
/// past the grid; such a cell is simply unknown.
pub trait BeliefCells {
    /// Measured ground height, if the cell has had a ground return.
    fn ground(&self, i: i64, j: i64) -> Option<f32>;
    /// Highest structure return, if the cell is believed occupied.
    fn structure_top(&self, i: i64, j: i64) -> Option<f32>;
    /// Every believed structure cell, in a fixed order.
    fn structure_cells(&self) -> Vec<(u32, u32)>;

    /// A cell corner's height: the mean of the known cells that share it.
    fn corner(&self, ci: i64, cj: i64) -> Option<f32> {
        let mut sum = 0.0;
        let mut count = 0;
        for (di, dj) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            if let Some(h) = self.ground(ci + di, cj + dj) {
                sum += h;
                count += 1;
            }
        }
        (count > 0).then(|| sum / count as f32)
    }
}

/// Where the grid is: its low corner, cell size and cell counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridLayout {
    pub origin_m: [f32; 2],
    pub resolution_m: f32,
    pub dimensions: [u32; 2],
}

/// Triangles with per-vertex normal and colour. Colours are sRGB.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colours: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn quad(&mut self, corners: [[f32; 3]; 4], normal: [f32; 3], colour: [f32; 4]) {
        let base = self.positions.len() as u32;
        for c in corners {
            self.positions.push(c);
            self.normals.push(normal);
            self.colours.push(colour);
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// How the ground is coloured: each surface corner, and the skirt's top and
/// bottom.
pub struct GroundColours<'a> {
    pub corner: &'a dyn Fn(i64, i64) -> [f32; 4],
    pub skirt_top: [f32; 4],
    pub skirt_bottom: [f32; 4],
}

/// The ground over cells `i_range` x `j_range` (half open). `fallback_m` is
/// the height of a corner nothing known touches, which only a skirt can
/// reach.
pub fn ground_mesh(
    cells: &dyn BeliefCells,
    layout: GridLayout,
    i_range: std::ops::Range<u32>,
    j_range: std::ops::Range<u32>,
    fallback_m: f32,
    skirt_m: f32,
    colours: &GroundColours<'_>,
) -> MeshData {
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    let known = |i: i64, j: i64| cells.ground(i, j).is_some();
    let mut mesh = MeshData::default();
    let mut corner_index: std::collections::HashMap<(i64, i64), u32> =
        std::collections::HashMap::new();
    let mut vertex = |mesh: &mut MeshData, ci: i64, cj: i64| -> u32 {
        *corner_index.entry((ci, cj)).or_insert_with(|| {
            let h = cells.corner(ci, cj).unwrap_or(fallback_m);
            let around = |a: i64, b: i64| cells.corner(a, b).unwrap_or(h);
            let dzdx = (around(ci + 1, cj) - around(ci - 1, cj)) / (2.0 * res);
            let dzdy = (around(ci, cj + 1) - around(ci, cj - 1)) / (2.0 * res);
            mesh.positions
                .push([x0 + ci as f32 * res, y0 + cj as f32 * res, h]);
            mesh.normals.push(normalize([-dzdx, -dzdy, 1.0]));
            mesh.colours.push((colours.corner)(ci, cj));
            (mesh.positions.len() - 1) as u32
        })
    };
    for j in j_range.clone() {
        for i in i_range.clone() {
            let (ii, jj) = (i as i64, j as i64);
            if !known(ii, jj) {
                continue;
            }
            let a = vertex(&mut mesh, ii, jj);
            let b = vertex(&mut mesh, ii + 1, jj);
            let c = vertex(&mut mesh, ii + 1, jj + 1);
            let d = vertex(&mut mesh, ii, jj + 1);
            mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Skirts, after the surface so they do not share its smoothed vertices.
    type Side = ((i64, i64), [(i64, i64); 2], [f32; 3]);
    for j in j_range {
        for i in i_range.clone() {
            let (ii, jj) = (i as i64, j as i64);
            if !known(ii, jj) {
                continue;
            }
            let sides: [Side; 4] = [
                ((0, -1), [(ii + 1, jj), (ii, jj)], [0.0, -1.0, 0.0]),
                ((1, 0), [(ii + 1, jj + 1), (ii + 1, jj)], [1.0, 0.0, 0.0]),
                ((0, 1), [(ii, jj + 1), (ii + 1, jj + 1)], [0.0, 1.0, 0.0]),
                ((-1, 0), [(ii, jj), (ii, jj + 1)], [-1.0, 0.0, 0.0]),
            ];
            for ((di, dj), corners, normal) in sides {
                if known(ii + di, jj + dj) {
                    continue;
                }
                let base = mesh.positions.len() as u32;
                for &(ci, cj) in &corners {
                    let h = cells.corner(ci, cj).unwrap_or(fallback_m);
                    let p = [x0 + ci as f32 * res, y0 + cj as f32 * res];
                    mesh.positions.push([p[0], p[1], h]);
                    mesh.positions.push([p[0], p[1], h - skirt_m]);
                    mesh.normals.push(normal);
                    mesh.normals.push(normal);
                    mesh.colours.push(colours.skirt_top);
                    mesh.colours.push(colours.skirt_bottom);
                }
                mesh.indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 3,
                    base,
                    base + 3,
                    base + 2,
                ]);
            }
        }
    }
    mesh
}

/// Every believed structure as one solid. `colour` gives a cell's roof and
/// wall colours.
pub fn structures_mesh(
    cells: &dyn BeliefCells,
    layout: GridLayout,
    colour: &dyn Fn(i64, i64) -> ([f32; 4], [f32; 4]),
) -> MeshData {
    let res = layout.resolution_m;
    let [x0, y0] = layout.origin_m;
    let mut mesh = MeshData::default();
    type Wall = ((i64, i64), [[f32; 2]; 2], [f32; 3]);
    for (i, j) in cells.structure_cells() {
        let (ii, jj) = (i as i64, j as i64);
        let Some(roof) = cells.structure_top(ii, jj) else {
            continue;
        };
        // The ground the walls stand on: this cell's, a neighbour's, or a
        // storey below the roof when nothing under it has been seen.
        let footing = cells
            .ground(ii, jj)
            .or_else(|| {
                [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter_map(|(di, dj)| cells.ground(ii + di, jj + dj))
                    .reduce(f32::min)
            })
            .unwrap_or(roof - 12.0)
            .min(roof - 1.0);
        let (roof_colour, wall_colour) = colour(ii, jj);
        let (ax, ay) = (x0 + i as f32 * res, y0 + j as f32 * res);
        let (bx, by) = (ax + res, ay + res);
        mesh.quad(
            [
                [ax, ay, roof],
                [bx, ay, roof],
                [bx, by, roof],
                [ax, by, roof],
            ],
            [0.0, 0.0, 1.0],
            roof_colour,
        );
        let walls: [Wall; 4] = [
            ((0, -1), [[bx, ay], [ax, ay]], [0.0, -1.0, 0.0]),
            ((1, 0), [[bx, by], [bx, ay]], [1.0, 0.0, 0.0]),
            ((0, 1), [[ax, by], [bx, by]], [0.0, 1.0, 0.0]),
            ((-1, 0), [[ax, ay], [ax, by]], [-1.0, 0.0, 0.0]),
        ];
        for ((di, dj), [p, q], normal) in walls {
            let foot = cells
                .structure_top(ii + di, jj + dj)
                .unwrap_or(footing)
                .max(footing);
            if foot >= roof {
                continue;
            }
            mesh.quad(
                [
                    [p[0], p[1], foot],
                    [q[0], q[1], foot],
                    [q[0], q[1], roof],
                    [p[0], p[1], roof],
                ],
                normal,
                wall_colour,
            );
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Cells(BTreeMap<(i64, i64), (Option<f32>, Option<f32>)>);

    impl BeliefCells for Cells {
        fn ground(&self, i: i64, j: i64) -> Option<f32> {
            self.0.get(&(i, j)).and_then(|c| c.0)
        }
        fn structure_top(&self, i: i64, j: i64) -> Option<f32> {
            self.0.get(&(i, j)).and_then(|c| c.1)
        }
        fn structure_cells(&self) -> Vec<(u32, u32)> {
            self.0
                .iter()
                .filter(|(_, c)| c.1.is_some())
                .map(|((i, j), _)| (*i as u32, *j as u32))
                .collect()
        }
    }

    const LAYOUT: GridLayout = GridLayout {
        origin_m: [0.0, 0.0],
        resolution_m: 10.0,
        dimensions: [64, 64],
    };

    fn plain(_: i64, _: i64) -> [f32; 4] {
        [1.0; 4]
    }

    #[test]
    fn neighbouring_cells_share_corners_and_the_outline_gets_a_skirt() {
        let mut cells = Cells::default();
        cells.0.insert((1, 1), (Some(12.0), None));
        cells.0.insert((2, 1), (Some(14.0), None));
        let colours = GroundColours {
            corner: &plain,
            skirt_top: [0.5; 4],
            skirt_bottom: [0.1; 4],
        };
        let mesh = ground_mesh(&cells, LAYOUT, 0..32, 0..32, 0.0, 6.0, &colours);
        assert_eq!(mesh.positions.len(), 6 + 6 * 4);
        // The shared corner sits at the mean of both cells.
        assert!(mesh.positions.contains(&[20.0, 10.0, 13.0]));
        assert!(ground_mesh(&cells, LAYOUT, 32..64, 32..64, 0.0, 6.0, &colours).is_empty());
    }

    #[test]
    fn adjacent_structure_cells_merge_into_one_outline() {
        let mut cells = Cells::default();
        for i in 3..5 {
            cells.0.insert((i, 4), (Some(10.0), Some(40.0)));
        }
        let colour = |_: i64, _: i64| ([1.0; 4], [0.5; 4]);
        let mesh = structures_mesh(&cells, LAYOUT, &colour);
        assert_eq!(mesh.positions.len(), 4 * (2 + 6));
        cells.0.insert((5, 4), (Some(10.0), Some(25.0)));
        let mesh = structures_mesh(&cells, LAYOUT, &colour);
        assert_eq!(mesh.positions.len(), 4 * (3 + 6 + 3));
    }
}
