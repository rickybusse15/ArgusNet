//! Deterministic tile index shared by collision and visibility queries.
use crate::obstacles::{Bounds2d, Obstacle};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct ObstacleLayer {
    pub bounds: Bounds2d,
    pub tile_size_m: f64,
    pub primitives: Vec<Obstacle>,
    tiles: BTreeMap<(i64, i64), Vec<usize>>,
}
impl ObstacleLayer {
    pub fn new(
        bounds: Bounds2d,
        tile_size_m: f64,
        mut primitives: Vec<Obstacle>,
    ) -> Result<Self, String> {
        if !tile_size_m.is_finite() || tile_size_m <= 0.0 {
            return Err("tile size must be positive".into());
        }
        primitives.sort_by(|a, b| a.id.cmp(&b.id));
        if primitives.windows(2).any(|p| p[0].id == p[1].id) {
            return Err("duplicate obstacle id".into());
        }
        let mut layer = Self {
            bounds,
            tile_size_m,
            primitives,
            tiles: BTreeMap::new(),
        };
        for i in 0..layer.primitives.len() {
            let (x0, x1, y0, y1) = layer.tile_range(layer.primitives[i].bounds_xy_m());
            for y in y0..=y1 {
                for x in x0..=x1 {
                    layer.tiles.entry((x, y)).or_default().push(i);
                }
            }
        }
        Ok(layer)
    }
    fn tile_range(&self, b: Bounds2d) -> (i64, i64, i64, i64) {
        let x = |v: f64| ((v - self.bounds.x_min_m) / self.tile_size_m).floor() as i64;
        let y = |v: f64| ((v - self.bounds.y_min_m) / self.tile_size_m).floor() as i64;
        (x(b.x_min_m), x(b.x_max_m), y(b.y_min_m), y(b.y_max_m))
    }
    pub fn query(&self, b: Bounds2d) -> Vec<&Obstacle> {
        let (x0, x1, y0, y1) = self.tile_range(b);
        let mut ids = BTreeSet::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                if let Some(v) = self.tiles.get(&(x, y)) {
                    ids.extend(v);
                }
            }
        }
        ids.into_iter()
            .map(|i: &usize| &self.primitives[*i])
            .collect()
    }
    pub fn point_collides(&self, p: [f64; 3]) -> Option<&Obstacle> {
        self.query(Bounds2d::new(p[0], p[0], p[1], p[1]).expanded(1e-6))
            .into_iter()
            .find(|o| o.blocker_type.is_hard() && o.point_inside(p[0], p[1], p[2]))
    }
}
