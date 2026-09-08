//! Reads the obstacle metadata of the repository's own scene package.
//!
//! The unit tests use hand-written fragments; this one proves the reader
//! handles what `argusnet build-scene` actually writes, including the flat
//! footprint encoding that every packaged scene currently carries.

use std::path::PathBuf;

use serde::Deserialize;
use terrain_engine::obstacles::{Obstacle, ObstacleMetadata, ObstacleShape};

#[derive(Deserialize)]
struct EnvironmentMetadata {
    occluding_objects: Vec<ObstacleMetadata>,
}

fn shipped_scene_environment() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scene.smartscene/metadata/environment.json")
}

fn load() -> Vec<Obstacle> {
    let path = shipped_scene_environment();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let metadata: EnvironmentMetadata =
        serde_json::from_str(&raw).expect("environment.json parses");
    metadata
        .occluding_objects
        .into_iter()
        .map(|entry| entry.into_obstacle().expect("every shipped entry converts"))
        .collect()
}

#[test]
fn every_shipped_obstacle_converts() {
    let obstacles = load();
    assert!(
        !obstacles.is_empty(),
        "the shipped scene should declare obstacles"
    );
    for obstacle in &obstacles {
        assert!(
            obstacle.top_z_m > obstacle.base_z_m,
            "{} has no height",
            obstacle.id
        );
        let bounds = obstacle.bounds_xy_m();
        assert!(
            bounds.x_max_m > bounds.x_min_m && bounds.y_max_m > bounds.y_min_m,
            "{} has an empty footprint",
            obstacle.id
        );
    }
}

/// The property that the flat-footprint encoding could silently break: a
/// mis-paired footprint still parses, but produces a polygon that no longer
/// contains the building it describes.
#[test]
fn every_shipped_obstacle_contains_its_own_centre() {
    for obstacle in load() {
        let bounds = obstacle.bounds_xy_m();
        let mid_z = (obstacle.base_z_m + obstacle.top_z_m) * 0.5;
        let (centre_x, centre_y) = match &obstacle.shape {
            // A polygon's bounding-box centre is only guaranteed inside for a
            // convex footprint; every shipped footprint is a quadrilateral.
            ObstacleShape::Polygon { .. } => (
                (bounds.x_min_m + bounds.x_max_m) * 0.5,
                (bounds.y_min_m + bounds.y_max_m) * 0.5,
            ),
            ObstacleShape::Cylinder { center_xy_m, .. } => (center_xy_m[0], center_xy_m[1]),
        };
        assert!(
            obstacle.point_inside(centre_x, centre_y, mid_z),
            "{} ({}) does not contain its own centre ({centre_x}, {centre_y})",
            obstacle.id,
            obstacle.kind
        );
    }
}

/// A ray fired horizontally through each obstacle's centre must register a
/// chord, and one fired well above its roof must not.
#[test]
fn shipped_obstacles_occlude_a_ray_through_them() {
    for obstacle in load() {
        let bounds = obstacle.bounds_xy_m();
        let mid_y = (bounds.y_min_m + bounds.y_max_m) * 0.5;
        let mid_z = (obstacle.base_z_m + obstacle.top_z_m) * 0.5;
        let span = bounds.x_max_m - bounds.x_min_m;

        let origin = [bounds.x_min_m - span, mid_y, mid_z];
        let target = [bounds.x_max_m + span, mid_y, mid_z];
        assert!(
            obstacle.path_length_inside(origin, target) > 0.0,
            "{} did not occlude a ray through its middle",
            obstacle.id
        );
        assert!(
            obstacle.first_hit_t(origin, target).is_some(),
            "{} reported a chord but no entry point",
            obstacle.id
        );

        let above = obstacle.top_z_m + 1_000.0;
        assert_eq!(
            obstacle.path_length_inside([origin[0], mid_y, above], [target[0], mid_y, above]),
            0.0,
            "{} occluded a ray 1 km above its roof",
            obstacle.id
        );
    }
}

#[test]
fn pushing_out_of_a_shipped_obstacle_escapes_it() {
    for obstacle in load() {
        let bounds = obstacle.bounds_xy_m();
        let mid_z = (obstacle.base_z_m + obstacle.top_z_m) * 0.5;
        let (centre_x, centre_y) = match &obstacle.shape {
            ObstacleShape::Polygon { .. } => (
                (bounds.x_min_m + bounds.x_max_m) * 0.5,
                (bounds.y_min_m + bounds.y_max_m) * 0.5,
            ),
            ObstacleShape::Cylinder { center_xy_m, .. } => (center_xy_m[0], center_xy_m[1]),
        };
        let [x, y] = obstacle.push_outside_xy(centre_x, centre_y, 2.0);
        assert!(
            !obstacle.point_inside(x, y, mid_z),
            "{} left the pushed point inside",
            obstacle.id
        );
    }
}
