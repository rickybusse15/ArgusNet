//! Wire the motion substrate to the real world.
//!
//! [`crate::motion`] defines [`TerrainSampler`] and [`ObstacleSampler`] as the
//! narrowest contracts the kinematic helpers need, and deliberately shipped
//! with no implementation over `argusnet-world`: nothing called the substrate
//! yet, and an adapter nobody exercises is an untested seam pretending to be
//! progress. P4.8d.2's controllers are its first caller, so the adapter
//! arrives now, ahead of them, with tests of its own.
//!
//! # This module is where the world dependency lives
//!
//! `motion.rs` still has no idea `argusnet-world` exists, and that is worth
//! preserving rather than merging the two files: the substrate's tests drive
//! it with a recorded oracle, and they can only do that because the contract
//! is a trait rather than a concrete terrain. Keeping the adapter separate
//! means the fixture tests stay honest about *what* they test, and this
//! module's tests are free to be about the wiring instead.

use argusnet_world::index::ObstacleLayer;
use argusnet_world::TerrainQuery;

use crate::motion::{ObstacleSampler, TerrainSampler};

/// The gradient step [`TerrainSampler::gradient_at`] means by `None`.
///
/// The reference resolves its default as
/// `max(terrain.base_resolution_m * 0.5, 0.25)`, and `argusnet-world`'s
/// `TerrainQuery::gradient_at` takes the step as a required argument — so
/// somebody has to supply it, and the choice belongs here rather than in
/// `motion.rs`, which has no terrain to ask.
///
/// Three different steps are in use across this codebase (`gradient_at`
/// defaults to `base_resolution * 0.5`, `slope_rad_at` uses `1.0`,
/// `slope_rad_at_many` uses `max(base_resolution * 0.5, 0.25)`), which is
/// exactly why the substrate refused to pick one silently.
fn default_gradient_delta_m(base_resolution_m: f64) -> f64 {
    (base_resolution_m * 0.5).max(0.25)
}

/// [`TerrainSampler`] over any `argusnet-world` terrain.
///
/// Borrowed rather than owned: a tick's terrain outlives the stage that reads
/// it, and copying a tiled height field per stage per tick would be the kind
/// of cost that only shows up at map scale.
pub struct WorldTerrain<'a> {
    terrain: &'a dyn TerrainQuery,
    base_resolution_m: f64,
}

impl<'a> WorldTerrain<'a> {
    /// Adapt `terrain`, resolving the default gradient step from
    /// `base_resolution_m`.
    ///
    /// The resolution is passed in rather than read off the terrain because
    /// [`TerrainQuery`] does not expose one: a tiled field, an analytic
    /// surface and an imported raster do not agree on what "the resolution"
    /// is, and guessing on their behalf is how the wrong finite-difference
    /// step gets picked.
    pub fn new(terrain: &'a dyn TerrainQuery, base_resolution_m: f64) -> Self {
        Self {
            terrain,
            base_resolution_m,
        }
    }

    /// The step this adapter passes when the caller does not name one.
    pub fn default_delta_m(&self) -> f64 {
        default_gradient_delta_m(self.base_resolution_m)
    }
}

impl TerrainSampler for WorldTerrain<'_> {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        self.terrain.height_at(x_m, y_m)
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2] {
        self.terrain
            .gradient_at(x_m, y_m, delta_m.unwrap_or_else(|| self.default_delta_m()))
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        self.terrain.clamp_altitude(x_m, y_m, z_m, min_agl_m)
    }
}

/// [`ObstacleSampler`] over an `argusnet-world` obstacle index.
///
/// # The `{building, wall}` filter is not this adapter's
///
/// `ObstacleLayer::point_collides` already tests
/// `matches!(blocker_type.as_str(), "building" | "wall")` before containment,
/// which is the same rule the Python `ObstacleLayer` applies — so a platform
/// inside a forest stand does not collide, and this adapter does not need to
/// re-implement that. It is stated here because the filter is surprising, it
/// lives two crates away from the caller, and a future reader tempted to
/// "fix" the adapter by adding a containment test would silently start
/// colliding with vegetation.
pub struct WorldObstacles<'a> {
    index: &'a ObstacleLayer,
}

impl<'a> WorldObstacles<'a> {
    pub fn new(index: &'a ObstacleLayer) -> Self {
        Self { index }
    }
}

impl ObstacleSampler for WorldObstacles<'_> {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        self.index
            .point_collides(point_m)
            .map(|obstacle| obstacle.push_outside_xy(point_m[0], point_m[1], margin_m))
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        self.index.point_collides(point_m).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::{collision_aware_position, compose_terrain_state};
    use argusnet_world::obstacles::{BlockerType, Obstacle};

    /// A tilted plane, so the gradient is a known non-zero constant and a
    /// wrong finite-difference step is still a *correct* answer — which is
    /// the point: these tests are about wiring, not about arithmetic the
    /// substrate's own fixture already pins.
    struct Plane;

    impl TerrainQuery for Plane {
        fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
            10.0 + 0.2 * x_m + 0.1 * y_m
        }
        fn analytic_height_at(&self, x_m: f64, y_m: f64) -> f64 {
            self.height_at(x_m, y_m)
        }
        fn normal_at(&self, _x_m: f64, _y_m: f64) -> [f64; 3] {
            [0.0, 0.0, 1.0]
        }
        fn gradient_at(&self, _x_m: f64, _y_m: f64, delta_m: f64) -> [f64; 2] {
            // Reports the step it was given, so a test can assert which one
            // the adapter chose.
            [0.2, delta_m]
        }
        fn curvature_at(&self, _x_m: f64, _y_m: f64, _delta_m: f64) -> f64 {
            0.0
        }
        fn slope_rad_at(&self, _x_m: f64, _y_m: f64) -> f64 {
            0.0
        }
        fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
            z_m.max(self.height_at(x_m, y_m) + min_agl_m.max(0.0))
        }
        fn los_raycast(
            &self,
            _a: [f64; 3],
            _b: [f64; 3],
            _c: f64,
        ) -> Option<argusnet_world::TerrainHit> {
            None
        }
        fn comms_shadow(&self, _a: [f64; 3], _b: [f64; 3]) -> bool {
            false
        }
        fn ground_plane_m(&self) -> f64 {
            0.0
        }
        fn bounds(&self) -> argusnet_world::TerrainBounds {
            argusnet_world::TerrainBounds {
                x_min_m: -1000.0,
                x_max_m: 1000.0,
                y_min_m: -1000.0,
                y_max_m: 1000.0,
                z_min_m: 0.0,
                z_max_m: 500.0,
            }
        }
    }

    #[test]
    fn the_default_gradient_step_is_half_the_resolution_with_a_floor() {
        // The ordinary case: half the base resolution.
        let terrain = WorldTerrain::new(&Plane, 25.0);
        assert_eq!(terrain.gradient_at(0.0, 0.0, None)[1], 12.5);

        // The floor, which bites on a fine grid. A 0.1 m resolution would
        // otherwise ask for a 0.05 m step and difference two samples of the
        // same cell.
        let fine = WorldTerrain::new(&Plane, 0.1);
        assert_eq!(fine.gradient_at(0.0, 0.0, None)[1], 0.25);
    }

    #[test]
    fn an_explicit_step_is_passed_through_unchanged() {
        let terrain = WorldTerrain::new(&Plane, 25.0);
        assert_eq!(terrain.gradient_at(0.0, 0.0, Some(1.0))[1], 1.0);
        // Including one below the floor: the floor is the *default's*, not a
        // clamp on what a caller may ask for. `slope_rad_at` legitimately
        // uses 1.0 and a caller may legitimately want something smaller.
        assert_eq!(terrain.gradient_at(0.0, 0.0, Some(0.05))[1], 0.05);
    }

    #[test]
    fn height_and_clamp_reach_the_terrain() {
        let terrain = WorldTerrain::new(&Plane, 10.0);
        assert_eq!(terrain.height_at(100.0, 0.0), 30.0);
        // Below the floor: raised to ground + agl.
        assert_eq!(terrain.clamp_altitude(100.0, 0.0, 0.0, 20.0), 50.0);
        // Above it: left alone.
        assert_eq!(terrain.clamp_altitude(100.0, 0.0, 500.0, 20.0), 500.0);
    }

    fn tower(blocker_type: BlockerType) -> ObstacleLayer {
        let obstacle = Obstacle::cylinder("tower", blocker_type, 0.0, 0.0, 60.0, 0.0, 300.0)
            .expect("valid cylinder");
        ObstacleLayer::new(
            argusnet_world::obstacles::Bounds2d::new(-500.0, 500.0, -500.0, 500.0),
            250.0,
            vec![obstacle],
        )
        .expect("valid layer")
    }

    #[test]
    fn a_building_pushes_and_vegetation_does_not() {
        let building = tower(BlockerType::Building);
        let sampler = WorldObstacles::new(&building);
        let pushed = sampler
            .push_clear([10.0, 10.0, 150.0], 1.0)
            .expect("inside a building");
        let cleared = pushed[0].hypot(pushed[1]);
        assert!(
            cleared > 60.0,
            "the push did not clear the radius: {pushed:?}"
        );

        // The margin is forwarded, not swallowed. Asserting only "outside the
        // radius" passed with a margin of zero, because clearing to exactly
        // the radius lands a hair outside it in floating point -- so the
        // margin has to be measured against a second, different one.
        // Found by a mutation run: replacing `margin_m` with `0.0` survived.
        let further = sampler
            .push_clear([10.0, 10.0, 150.0], 5.0)
            .expect("inside a building");
        let cleared_further = further[0].hypot(further[1]);
        assert!(
            (cleared_further - cleared - 4.0).abs() < 1e-9,
            "a 4 m larger margin moved the point by {} m",
            cleared_further - cleared
        );

        // Same geometry, same point, different blocker type. The filter is
        // the index's; this asserts the adapter does not defeat it by adding
        // a containment test of its own.
        let stand = tower(BlockerType::Vegetation);
        assert_eq!(
            WorldObstacles::new(&stand).push_clear([10.0, 10.0, 150.0], 1.0),
            None,
            "a platform collided with a forest stand"
        );
    }

    #[test]
    fn a_point_above_the_top_does_not_collide() {
        let index = tower(BlockerType::Building);
        assert_eq!(
            WorldObstacles::new(&index).push_clear([0.0, 0.0, 400.0], 1.0),
            None
        );
    }

    /// The substrate and the adapter compose: this is the call a controller
    /// will actually make, and it is the thing neither file's own tests can
    /// check on its own.
    #[test]
    fn the_substrate_runs_against_a_real_world() {
        let terrain = WorldTerrain::new(&Plane, 25.0);
        let index = tower(BlockerType::Building);
        let obstacles = WorldObstacles::new(&index);

        let (position, velocity) =
            compose_terrain_state(&terrain, [10.0, 10.0], [12.0, 0.0], 80.0, 0.0, 20.0, false);
        // Ground is 10 + 2 + 1 = 13, so 80 m AGL puts it at 93.
        assert_eq!(position[2], 93.0);
        // vz is the gradient dotted with the horizontal velocity: flying up
        // a 0.2 slope at 12 m/s is a 2.4 m/s climb.
        assert!((velocity[2] - 2.4).abs() < 1e-12, "vz was {}", velocity[2]);

        let resolved = collision_aware_position(&terrain, Some(&obstacles), position, 20.0);
        assert!(!resolved.unresolved);
        assert!(
            resolved.position_m[0].hypot(resolved.position_m[1]) > 60.0,
            "the platform was left inside the tower"
        );
        // And the re-clamp kept it above the ground it was pushed onto.
        let ground = terrain.height_at(resolved.position_m[0], resolved.position_m[1]);
        assert!(
            resolved.position_m[2] >= ground + 20.0,
            "pushed below its minimum AGL"
        );
    }
}
