//! One owned world, lent to every stage that reads it.
//!
//! [`crate::motion_world`] adapts `argusnet-world` by *borrowing*, which suits
//! a test that holds the world on its stack. An assembled run does not: stage
//! handlers are owned by the `Simulation`, must be `Send`, and outlive any one
//! tick. So the world is built once, held behind `Arc`s, and each stage gets a
//! cheap clone of the capability it needs. Every stage reads the **same**
//! terrain rather than its own copy, and that is what keeps "the platform is
//! above the ground" one fact across stages 1, 3 and 6.
//!
//! The arithmetic is not here. Each adapter builds the borrowed adapter from
//! `motion_world` per call, which costs a pointer and a float, so there is one
//! implementation of the default gradient step and of the blocker filter.

use std::fmt;
use std::sync::Arc;

use argusnet_sensing::cascade::{Blocker, Visibility};
use argusnet_world::index::ObstacleLayer;
use argusnet_world::land_cover::LandCoverLayer;
use argusnet_world::tiled::TiledTerrain;
use argusnet_world::trees::{TreeLayer, CROWN_EXTINCTION_PER_M};
use argusnet_world::visibility::{self, VisibilityProfile, VisibilityResult};

use crate::follow_path::VisibilitySampler;
use crate::motion::{ObstacleSampler, TerrainSampler};
use crate::motion_world::{WorldObstacles, WorldTerrain};
use crate::observe::{Detection, SightLine, TerrainHeight, VisibilityOracle};
use crate::weather::MotionWeather;

/// The terrain, shared.
#[derive(Clone)]
pub struct SharedTerrain {
    terrain: Arc<TiledTerrain>,
    base_resolution_m: f64,
}

impl fmt::Debug for SharedTerrain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedTerrain")
            .field("base_resolution_m", &self.base_resolution_m)
            .finish()
    }
}

impl SharedTerrain {
    pub fn new(terrain: Arc<TiledTerrain>, base_resolution_m: f64) -> Self {
        Self {
            terrain,
            base_resolution_m,
        }
    }

    fn borrowed(&self) -> WorldTerrain<'_> {
        WorldTerrain::new(self.terrain.as_ref(), self.base_resolution_m)
    }
}

impl TerrainSampler for SharedTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        self.borrowed().height_at(x_m, y_m)
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2] {
        self.borrowed().gradient_at(x_m, y_m, delta_m)
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        self.borrowed().clamp_altitude(x_m, y_m, z_m, min_agl_m)
    }
}

impl TerrainHeight for SharedTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        TerrainSampler::height_at(self, x_m, y_m)
    }

    fn height_along(
        &self,
        from: [f64; 2],
        to: [f64; 2],
    ) -> Option<(Vec<argusnet_world::tiled::HeightPiece>, f64)> {
        Some((self.terrain.height_along(from, to), self.terrain.floor_m))
    }
}

/// The obstacle layer, shared.
#[derive(Clone)]
pub struct SharedObstacles {
    layer: Arc<ObstacleLayer>,
}

impl SharedObstacles {
    pub fn new(layer: Arc<ObstacleLayer>) -> Self {
        Self { layer }
    }

    /// The layer itself, for a stage that plans routes over it.
    pub fn layer(&self) -> Arc<ObstacleLayer> {
        Arc::clone(&self.layer)
    }
}

impl ObstacleSampler for SharedObstacles {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        WorldObstacles::new(&self.layer).push_clear(point_m, margin_m)
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        WorldObstacles::new(&self.layer).collides(point_m)
    }
}

/// Line of sight over the shared world, as `EnvironmentQuery.los` computes it.
///
/// Land cover is sampled at the sight line's **midpoint**, which is the
/// reference's choice (one class for the whole line), and class 0 is used
/// when the world has no land-cover layer.
#[derive(Clone)]
pub struct SharedVisibility {
    terrain: Arc<TiledTerrain>,
    obstacles: Arc<ObstacleLayer>,
    land_cover: Option<Arc<LandCoverLayer>>,
    profile: Arc<VisibilityProfile>,
    clearance_m: f64,
    /// Weather for the detection model; `None` is clear conditions (factor
    /// 1, noise scale 1), as `compute_weather_factor(range, None)` is.
    weather: Option<Arc<MotionWeather>>,
    /// The world's trees (R2), `None` when it has none: then every answer
    /// is the one a world before trees gave.
    trees: Option<Arc<TreeLayer>>,
}

impl fmt::Debug for SharedVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedVisibility")
            .field("clearance_m", &self.clearance_m)
            .field("has_land_cover", &self.land_cover.is_some())
            .finish()
    }
}

impl SharedVisibility {
    pub fn new(
        terrain: Arc<TiledTerrain>,
        obstacles: Arc<ObstacleLayer>,
        land_cover: Option<Arc<LandCoverLayer>>,
        profile: Arc<VisibilityProfile>,
        clearance_m: f64,
    ) -> Self {
        Self {
            terrain,
            obstacles,
            land_cover,
            profile,
            clearance_m,
            weather: None,
            trees: None,
        }
    }

    /// The same world with its trees. An empty layer is no trees.
    pub fn with_trees(&self, trees: Arc<TreeLayer>) -> Self {
        Self {
            trees: (!trees.is_empty()).then_some(trees),
            ..self.clone()
        }
    }

    /// The same world, with weather for the detection model.
    pub fn with_weather(&self, weather: Option<Arc<MotionWeather>>) -> Self {
        Self {
            weather,
            ..self.clone()
        }
    }

    /// The same world at a different terrain clearance.
    pub fn with_clearance(&self, clearance_m: f64) -> Self {
        Self {
            clearance_m,
            ..self.clone()
        }
    }

    pub fn los(&self, from: [f64; 3], to: [f64; 3]) -> VisibilityResult {
        let land_cover = self.land_cover.as_ref().map_or(0, |layer| {
            let midpoint = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
            i32::from(layer.sample(midpoint[0], midpoint[1]).0)
        });
        visibility::los_with_trees(
            &self.terrain,
            &self.obstacles,
            self.trees.as_deref(),
            from,
            to,
            &self.profile,
            land_cover,
            self.clearance_m,
        )
    }
}

impl SharedVisibility {
    /// The first surface a ray from `from` toward `to` meets: `terrain`,
    /// `building` or `wall`, and the point. The nearer of the terrain crossing
    /// and the first structure face (wall or roof, in 3D) wins. (`los` answers a different question,
    /// whether anything blocks, and checks terrain first.) Vegetation lets the
    /// ray through. The ray is cut at the edge of the terrain first, so a ray
    /// that would leave the world still reports what it meets before the
    /// edge. `None` when it meets nothing.
    pub fn first_hit(&self, from: [f64; 3], to: [f64; 3]) -> Option<(String, [f64; 3])> {
        let bounds = self.terrain.bounds;
        if !bounds.contains_xy(from[0], from[1]) {
            return None;
        }
        let (_, t_end) = visibility::clip([from[0], from[1]], [to[0], to[1]], bounds)?;
        let t_end = (t_end - 1e-6).max(0.0);
        let end: [f64; 3] = std::array::from_fn(|i| from[i] + (to[i] - from[i]) * t_end);
        let terrain = visibility::terrain_intersection(&self.terrain, from, end, 0.0).and_then(
            |(kind, t, point)| match (kind.as_str(), t) {
                ("terrain", Some(t)) => Some((t, "terrain".to_string(), point)),
                _ => None,
            },
        );
        let box_ = argusnet_world::obstacles::Bounds2d::new(
            from[0].min(end[0]),
            from[0].max(end[0]) + 1e-6,
            from[1].min(end[1]),
            from[1].max(end[1]) + 1e-6,
        );
        let structure = self
            .obstacles
            .query(box_)
            .into_iter()
            .filter(|o| o.blocker_type.is_hard())
            .filter_map(|o| {
                o.surface_hit_t(from, end)
                    .map(|t| (t, o.blocker_type.as_str().to_string()))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(t, kind)| {
                let point: [f64; 3] = std::array::from_fn(|i| from[i] + (end[i] - from[i]) * t);
                (t, kind, point)
            });
        match (terrain, structure) {
            (Some(t), Some(o)) => Some(if o.0 < t.0 { o } else { t }),
            (t, o) => t.or(o),
        }
        .map(|(_, kind, point)| (kind, point))
    }
}

impl SharedVisibility {
    /// What a range scanner's ray from `from` toward `to` returns from:
    /// [`SharedVisibility::first_hit`], or before it a tree. A trunk
    /// returns like a structure (`"trunk"`). A crown is leaves with gaps:
    /// at each crown the ray enters, in order, `draw` gives a uniform `u`
    /// and the ray is stopped `-ln(1 - u) / (k · d)` metres in (Beer–Lambert
    /// with the crown's leaf density `d`), returning `"canopy"` there if
    /// that is inside the crown and short of anything harder; otherwise it
    /// goes on through. So a crown returns with probability one minus its
    /// transmittance along the ray, the same light [`SharedVisibility::los`]
    /// lets through. `draw` is called once per crown entered, and never in
    /// a world without trees.
    pub fn first_return(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        draw: &mut dyn FnMut() -> f64,
    ) -> Option<(String, [f64; 3])> {
        let hard = self.first_hit(from, to);
        let Some(trees) = self.trees.as_ref() else {
            return hard;
        };
        if !self.terrain.bounds.contains_xy(from[0], from[1]) {
            return hard;
        }
        // Where the ray stops: at what it hit, or where it leaves the
        // terrain, as `first_hit` cut it; a tree past the world's edge is
        // never met.
        let end = match &hard {
            Some((_, point)) => *point,
            None => {
                let (_, t_end) =
                    visibility::clip([from[0], from[1]], [to[0], to[1]], self.terrain.bounds)?;
                let t_end = (t_end - 1e-6).max(0.0);
                std::array::from_fn(|i| from[i] + (to[i] - from[i]) * t_end)
            }
        };
        let along =
            |t: f64| -> [f64; 3] { std::array::from_fn(|i| from[i] + (end[i] - from[i]) * t) };
        let mut stop = 1.0;
        let mut best = hard.clone();
        if let Some(hit) = trees.first_trunk_hit(from, end) {
            stop = hit.t;
            best = Some(("trunk".to_string(), along(hit.t)));
        }
        let length = (0..3)
            .map(|i| (end[i] - from[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        if length <= 0.0 {
            return best;
        }
        for (t0, t1, tree) in trees.crowns_along(from, end) {
            if t0 >= stop {
                break;
            }
            let u = draw();
            let extinction = CROWN_EXTINCTION_PER_M * tree.leaf_density;
            if extinction <= 0.0 {
                continue;
            }
            let t = t0 + (-(1.0 - u).ln() / extinction) / length;
            if t <= t1.min(stop) {
                return Some(("canopy".to_string(), along(t)));
            }
        }
        best
    }
}

impl VisibilitySampler for SharedVisibility {
    fn is_visible(&self, observer_m: [f64; 3], target_m: [f64; 3]) -> bool {
        self.los(observer_m, target_m).visible
    }
}

impl VisibilityOracle for SharedVisibility {
    fn sight_line(&self, from: [f64; 3], to: [f64; 3]) -> SightLine {
        sight_line_of(&self.los(from, to))
    }

    fn sight_and_detection(
        &self,
        from: [f64; 3],
        to: [f64; 3],
        max_range_m: f64,
    ) -> (SightLine, Option<Detection>) {
        let result = self.los(from, to);
        // `float(np.linalg.norm(target - origin))`.
        let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let range_m = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let (weather_factor, weather_noise, cloud_factor) = match &self.weather {
            Some(weather) => (
                weather.visibility_at_range(range_m).max(0.0),
                weather.bearing_noise_scale(),
                if weather.cloud_layers.is_empty() {
                    1.0
                } else {
                    weather.cloud_factor(from[2], to[2])
                },
            ),
            None => (1.0, 1.0, 1.0),
        };
        let detection =
            visibility::detection(&result, range_m, max_range_m, weather_factor, weather_noise);
        (
            sight_line_of(&result),
            Some(Detection {
                p_d: detection.p_d,
                effective_noise_multiplier: detection.effective_noise_multiplier,
                dominant_loss_factor: dominant_loss_name(&detection.dominant_loss_factor),
                cloud_factor,
            }),
        )
    }
}

/// `visibility::dominant`'s four answers, as the `'static` names `Detection`
/// carries.
fn dominant_loss_name(name: &str) -> &'static str {
    match name {
        "blocked" => "blocked",
        "range" => "range",
        "transmittance" => "transmittance",
        "weather" => "weather",
        other => unreachable!("visibility::dominant has no answer {other:?}"),
    }
}

fn sight_line_of(result: &VisibilityResult) -> SightLine {
    {
        SightLine {
            visibility: if result.visible {
                Visibility::Visible
            } else {
                Visibility::Blocked(match result.blocker_type.as_str() {
                    "terrain" => Blocker::Terrain,
                    "building" => Blocker::Building,
                    "wall" => Blocker::Wall,
                    "vegetation" => Blocker::Vegetation,
                    "out_of_coverage" => Blocker::OutOfCoverage,
                    _ => Blocker::Other,
                })
            },
            noise_multiplier: result.noise_multiplier,
            blocker_type: result.blocker_type.clone(),
            first_hit_range_m: result.first_hit_range_m,
            closest_point: result.closest_point,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_world::obstacles::Bounds2d;
    use argusnet_world::trees::{Species, Tree};

    /// A flat 200 m square at 0 m, and one 20 m broadleaf at (100, 100):
    /// crown 8–20 m up (centre 14 m, half-height 6 m), radius 5 m, a 0.3 m
    /// trunk to 14 m.
    fn world(leaf_density: f64) -> (SharedVisibility, SharedVisibility) {
        let bounds = Bounds2d::new(0.0, 200.0, 0.0, 200.0);
        let terrain = Arc::new(
            TiledTerrain::from_grid(bounds, &vec![vec![0.0; 21]; 21], 10.0, 4, 0.0, &[]).unwrap(),
        );
        let obstacles = Arc::new(ObstacleLayer::new(bounds, 50.0, vec![]).unwrap());
        let plain = SharedVisibility::new(
            terrain,
            obstacles,
            None,
            Arc::new(VisibilityProfile::default()),
            0.0,
        );
        let tree = Tree {
            id: "t".into(),
            xy_m: [100.0, 100.0],
            base_z_m: 0.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 5.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density,
        };
        let with = plain.with_trees(Arc::new(TreeLayer::new(vec![tree]).unwrap()));
        (plain, with)
    }

    #[test]
    fn without_trees_a_scan_ray_returns_what_it_did_and_draws_nothing() {
        let (plain, _) = world(0.8);
        let empty = plain.with_trees(Arc::new(TreeLayer::new(vec![]).unwrap()));
        let mut draws = 0;
        for vis in [&plain, &empty] {
            let ray = ([100.0, 100.0, 60.0], [100.0, 100.0, -200.0]);
            let hit = vis.first_return(ray.0, ray.1, &mut || {
                draws += 1;
                0.5
            });
            assert_eq!(hit, plain.first_hit(ray.0, ray.1));
            assert_eq!(hit.map(|h| h.0), Some("terrain".to_string()));
        }
        assert_eq!(draws, 0);
    }

    /// Straight down 2 m off the trunk the crown's top is at
    /// `14 + 6·√(1 − 0.16)` and its chord 11 m; at density 0.5 it stops
    /// `1 - exp(-0.35 · 0.5 · 11)` = 85 % of rays. A uniform `u` stops the
    /// ray `-ln(1 - u) / (0.35 · 0.5)` metres into the crown.
    #[test]
    fn a_crown_stops_a_ray_as_often_as_it_dims_light() {
        let (_, with) = world(0.5);
        let ray = ([102.0, 100.0, 60.0], [102.0, 100.0, -200.0]);
        let top = 14.0 + 6.0 * (1.0_f64 - 0.16).sqrt();
        // u = 0.5: ln 2 / 0.175 = 3.96 m below the crown's top.
        let (kind, point) = with
            .first_return(ray.0, ray.1, &mut || 0.5)
            .expect("returns");
        assert_eq!(kind, "canopy");
        assert!(
            (point[2] - (top - 2f64.ln() / 0.175)).abs() < 1e-9,
            "{point:?}"
        );
        // A u past the crown's whole depth passes through to the ground.
        let (kind, _) = with
            .first_return(ray.0, ray.1, &mut || 0.95)
            .expect("returns");
        assert_eq!(kind, "terrain");
        // Over many draws, the share stopped is one minus the transmittance.
        let mut rng = crate::deterministic::DeterministicRng::from_seed(3);
        let n = 20_000;
        let stopped = (0..n)
            .filter(|_| {
                with.first_return(ray.0, ray.1, &mut || rng.uniform())
                    .is_some_and(|(kind, _)| kind == "canopy")
            })
            .count() as f64
            / n as f64;
        let expected = 1.0 - with.los(ray.0, [102.0, 100.0, 0.5]).transmittance;
        assert!((stopped - expected).abs() < 0.01, "{stopped} vs {expected}");
        // A bare crown never stops one, but still takes its draw.
        let (_, bare) = world(0.0);
        let mut draws = 0;
        let hit = bare.first_return(ray.0, ray.1, &mut || {
            draws += 1;
            0.0
        });
        assert_eq!(hit.map(|h| h.0), Some("terrain".to_string()));
        assert_eq!(draws, 1);
    }

    #[test]
    fn a_trunk_returns_like_a_structure_and_hides_what_is_behind() {
        let (_, with) = world(0.8);
        // Level at 5 m, under the crown: the trunk's face at x = 99.7.
        let (kind, point) = with
            .first_return([20.0, 100.0, 5.0], [180.0, 100.0, 5.0], &mut || 0.99)
            .expect("returns");
        assert_eq!(kind, "trunk");
        assert!((point[0] - 99.7).abs() < 1e-9, "{point:?}");
        // At 12 m the ray meets the crown before the trunk inside it: with
        // a u that stops it, a canopy return short of the trunk.
        let (kind, point) = with
            .first_return([20.0, 100.0, 12.0], [180.0, 100.0, 12.0], &mut || 0.3)
            .expect("returns");
        assert_eq!(kind, "canopy");
        assert!(point[0] < 99.7, "{point:?}");
        // With one that does not, the trunk.
        let (kind, _) = with
            .first_return([20.0, 100.0, 12.0], [180.0, 100.0, 12.0], &mut || 0.999)
            .expect("returns");
        assert_eq!(kind, "trunk");
        // Line of sight: the trunk is a hard blocker.
        let sight = with.los([20.0, 100.0, 5.0], [180.0, 100.0, 5.0]);
        assert!(!sight.visible);
        assert_eq!(sight.blocker_type, "trunk");
    }

    /// A crown behind a trunk is never reached: the ray stops at the trunk
    /// and takes no draw for the crown past it.
    #[test]
    fn nothing_past_a_trunk_returns_or_draws() {
        let (plain, _) = world(0.8);
        let near = Tree {
            id: "near".into(),
            xy_m: [100.0, 100.0],
            base_z_m: 0.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 5.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: 0.8,
        };
        // A low, wide crown behind the near trunk, crossed at 5 m.
        let behind = Tree {
            id: "behind".into(),
            xy_m: [130.0, 100.0],
            crown_base_m: 0.5,
            crown_radius_m: 8.0,
            trunk_radius_m: 0.0,
            ..near.clone()
        };
        let with = plain.with_trees(Arc::new(TreeLayer::new(vec![near, behind]).unwrap()));
        let mut draws = 0;
        let (kind, point) = with
            .first_return([20.0, 100.0, 5.0], [180.0, 100.0, 5.0], &mut || {
                draws += 1;
                0.0
            })
            .expect("returns");
        assert_eq!(kind, "trunk");
        assert!((point[0] - 99.7).abs() < 1e-9, "{point:?}");
        assert_eq!(draws, 0, "the crown past the trunk takes no draw");
    }

    /// A ray that leaves the world meets no tree past its edge.
    #[test]
    fn no_tree_past_the_worlds_edge_returns() {
        let (plain, _) = world(0.8);
        let outside = Tree {
            id: "outside".into(),
            xy_m: [205.0, 100.0],
            base_z_m: 0.0,
            height_m: 20.0,
            crown_base_m: 2.0,
            crown_radius_m: 8.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: 1.0,
        };
        let with = plain.with_trees(Arc::new(TreeLayer::new(vec![outside]).unwrap()));
        // Level at 10 m eastward from x = 150: the crown reaches back to
        // x = 197, inside; past x = 200 the ray has left the world.
        let mut draws = 0;
        let hit = with.first_return([150.0, 100.0, 10.0], [260.0, 100.0, 10.0], &mut || {
            draws += 1;
            0.0
        });
        assert_eq!(
            hit.as_ref().map(|h| h.0.as_str()),
            Some("canopy"),
            "{hit:?}"
        );
        assert!(hit.unwrap().1[0] < 200.0);
        // A draw that would stop it 13 m in, past the edge at x = 200: it
        // has left the world, and returns nothing.
        let hit = with.first_return([150.0, 100.0, 10.0], [260.0, 100.0, 10.0], &mut || 0.99);
        assert_eq!(hit, None, "{hit:?}");
        assert_eq!(draws, 1);
    }
}
