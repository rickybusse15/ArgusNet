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
        visibility::los(
            &self.terrain,
            &self.obstacles,
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
