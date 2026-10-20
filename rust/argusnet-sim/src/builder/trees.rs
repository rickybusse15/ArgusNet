//! Trees for a built world (R2, `tree-v1`): opt in with the `trees` option.
//!
//! Trees are scattered over the land cover's Forest and Scrub cells from a
//! seeded, fixed-order Poisson-disk sample ([`poisson_disk`]): the sample
//! covers the whole map, and a point is kept where the ground is forest
//! (every point) or scrub (about one in three, as a shrub mostly). A tree is
//! part of the world, like the land cover, so it is seeded from the terrain
//! seed and the presets, not the run's seed: seeds 7 and 42 fly over the
//! same forest. A world without the option is exactly what it was before
//! trees existed.
//!
//! Not every forest cell gets trees:
//!
//! - **Forest stands** (`forest-stand-v1`) already stand for the trees
//!   inside them; they remain the cheap aggregate, and no tree is placed in
//!   one, so a crown is never dimmed twice.
//! - No trunk stands inside a building or wall.
//! - At most [`MAX_TREES`]: a large map's forest is thinned evenly to it.

use argusnet_world::land_cover_build::{CoverGrid, LandCoverClass, SeasonalVariation};
use argusnet_world::landscape::BuiltTerrain;
use argusnet_world::numpy_rng::NumpyRng;
use argusnet_world::obstacles::{point_in_polygon, Bounds2d, Obstacle, ObstacleShape};
use argusnet_world::trees::{poisson_disk, Species, Tree};

use super::environment::MapBounds;

/// No two trunks closer than this, metres. With crowns 2.5–8 m across in
/// radius, a forest at this spacing is closed canopy with gaps.
const SPACING_M: f64 = 8.0;
/// Candidates a sampler cell draws.
const ATTEMPTS: usize = 10;
/// The share of sample points kept over scrub.
const SCRUB_KEEP: f64 = 0.3;
/// The most trees a built world holds. A medium map's forest (55 ha on the
/// default terrain) is about 6,500; a regional map's (810 ha) would be
/// ten times that, more than the bundle and the replay should carry, so it
/// is thinned evenly to this.
pub const MAX_TREES: usize = 15_000;
/// Uniform draws each sample point takes, kept or not, so one point's fate
/// never shifts another's shape.
const DRAWS_PER_POINT: usize = 8;

/// The ranges a species' trees are drawn from: height, crown base and
/// crown radius as fractions of the height, trunk radius as a fraction of
/// the height (clamped), and leaf density in full leaf.
struct Shape {
    height_m: (f64, f64),
    crown_base: (f64, f64),
    crown_radius: (f64, f64),
    trunk_radius: f64,
    trunk_radius_m: (f64, f64),
    leaf_density: (f64, f64),
    /// Whether the crown thins with the season.
    deciduous: bool,
}

fn shape(species: Species) -> Shape {
    match species {
        Species::Conifer => Shape {
            height_m: (14.0, 28.0),
            crown_base: (0.15, 0.35),
            crown_radius: (0.15, 0.22),
            trunk_radius: 0.012,
            trunk_radius_m: (0.12, 0.4),
            leaf_density: (0.75, 0.9),
            deciduous: false,
        },
        Species::Broadleaf => Shape {
            height_m: (10.0, 22.0),
            crown_base: (0.3, 0.45),
            crown_radius: (0.28, 0.38),
            trunk_radius: 0.015,
            trunk_radius_m: (0.12, 0.45),
            leaf_density: (0.8, 0.95),
            deciduous: true,
        },
        Species::Palm => Shape {
            height_m: (8.0, 18.0),
            crown_base: (0.75, 0.85),
            crown_radius: (0.2, 0.28),
            trunk_radius: 0.015,
            trunk_radius_m: (0.12, 0.25),
            leaf_density: (0.45, 0.6),
            deciduous: false,
        },
        Species::Shrub => Shape {
            height_m: (1.5, 4.0),
            crown_base: (0.05, 0.15),
            crown_radius: (0.4, 0.6),
            trunk_radius: 0.02,
            trunk_radius_m: (0.03, 0.08),
            leaf_density: (0.6, 0.8),
            deciduous: true,
        },
    }
}

/// The species mix over forest for a terrain preset, as cumulative shares
/// of conifer, broadleaf and palm (the rest).
fn forest_mix(terrain_preset: &str) -> [f64; 2] {
    match terrain_preset {
        "alpine" | "mountain_pass" | "arctic_tundra" => [0.85, 1.0],
        "jungle_canopy" => [0.0, 0.7],
        "coastal" | "desert_canyon" => [0.3, 0.75],
        _ => [0.4, 1.0],
    }
}

/// The share of shrubs among what is kept over scrub; the rest are small
/// broadleaves.
const SCRUB_SHRUBS: f64 = 0.85;

/// What a tree draws from the world it stands in.
pub struct TreeGround<'a> {
    pub terrain: &'a BuiltTerrain,
    pub bounds: MapBounds,
    pub land_cover: &'a CoverGrid,
    pub land_cover_resolution_m: f64,
    pub obstacles: &'a [Obstacle],
    pub season: SeasonalVariation,
    pub terrain_preset: &'a str,
}

impl TreeGround<'_> {
    /// The land-cover class at `(x, y)`, as `LandCoverLayer::sample` reads
    /// the same grid.
    fn class_at(&self, xy: [f64; 2]) -> u8 {
        let grid = self.land_cover;
        let column = ((xy[0] - self.bounds.x_min_m) / self.land_cover_resolution_m).floor();
        let row = ((xy[1] - self.bounds.y_min_m) / self.land_cover_resolution_m).floor();
        let column = (column.max(0.0) as usize).min(grid.cols - 1);
        let row = (row.max(0.0) as usize).min(grid.rows - 1);
        grid.classes[row * grid.cols + column]
    }

    /// Whether `xy` is inside a forest stand's footprint, or a trunk there
    /// would stand inside a building or wall.
    fn taken(&self, xy: [f64; 2]) -> bool {
        self.obstacles.iter().any(|obstacle| {
            if !obstacle.bounds_xy_m().contains_xy(xy[0], xy[1]) {
                return false;
            }
            if obstacle.kind == "forest-stand-v1" {
                return match &obstacle.shape {
                    ObstacleShape::Polygon { footprint_xy_m } => {
                        point_in_polygon(xy, footprint_xy_m)
                    }
                    ObstacleShape::Cylinder {
                        center_xy_m,
                        radius_m,
                    } => (xy[0] - center_xy_m[0]).hypot(xy[1] - center_xy_m[1]) <= *radius_m,
                };
            }
            obstacle.blocker_type.is_hard()
                && obstacle.point_inside(xy[0], xy[1], 0.5 * (obstacle.base_z_m + obstacle.top_z_m))
        })
    }
}

/// The world's trees, scattered over its forest and scrub, seeded by
/// `seed`, in id order (`tree-00000`, …, south to north and west to east
/// within a row of the sample).
pub fn build_trees(ground: &TreeGround<'_>, seed: u32) -> Vec<Tree> {
    let bounds = Bounds2d::new(
        ground.bounds.x_min_m,
        ground.bounds.x_max_m,
        ground.bounds.y_min_m,
        ground.bounds.y_max_m,
    );
    let points = poisson_disk(bounds, SPACING_M, ATTEMPTS, seed);
    // A second stream for each point's fate and shape, a fixed number of
    // draws per point, so the sample and the shapes are independent.
    let mut rng = NumpyRng::new(seed ^ 0x5eed_7ee5);
    let mut kept: Vec<([f64; 2], [f64; DRAWS_PER_POINT])> = Vec::new();
    for xy in points {
        let draws: [f64; DRAWS_PER_POINT] = std::array::from_fn(|_| rng.random());
        // Rounded first, so the class is read where the tree will stand.
        let xy = [cm(xy[0]), cm(xy[1])];
        let class = ground.class_at(xy);
        let keep = if class == LandCoverClass::Forest as u8 {
            true
        } else if class == LandCoverClass::Scrub as u8 {
            draws[0] < SCRUB_KEEP
        } else {
            false
        };
        if keep && !ground.taken(xy) {
            kept.push((xy, draws));
        }
    }
    let kept = thin_evenly(kept, MAX_TREES, |(_, draws)| draws[7]);
    let mix = forest_mix(ground.terrain_preset);
    kept.into_iter()
        .enumerate()
        .map(|(index, (xy, d))| {
            let over_forest = ground.class_at(xy) == LandCoverClass::Forest as u8;
            let species = if over_forest {
                if d[1] < mix[0] {
                    Species::Conifer
                } else if d[1] < mix[1] {
                    Species::Broadleaf
                } else {
                    Species::Palm
                }
            } else if d[1] < SCRUB_SHRUBS {
                Species::Shrub
            } else {
                Species::Broadleaf
            };
            let s = shape(species);
            let lerp = |(lo, hi): (f64, f64), t: f64| lo + (hi - lo) * t;
            let height_m = cm(lerp(s.height_m, d[2]));
            let full_leaf = lerp(s.leaf_density, d[6]);
            let leaf_density = if s.deciduous {
                full_leaf * ground.season.foliage_density_factor
            } else {
                full_leaf
            };
            Tree {
                id: format!("tree-{index:05}"),
                xy_m: xy,
                base_z_m: mm(ground.terrain.height_at(xy[0], xy[1])),
                height_m,
                crown_base_m: cm(height_m * lerp(s.crown_base, d[3])),
                crown_radius_m: cm(height_m * lerp(s.crown_radius, d[4])),
                trunk_radius_m: cm((height_m * s.trunk_radius * (0.8 + 0.4 * d[5]))
                    .clamp(s.trunk_radius_m.0, s.trunk_radius_m.1)),
                species,
                leaf_density: cm(leaf_density).clamp(0.0, 1.0),
            }
        })
        .collect()
}

/// At most `cap` of `items`, chosen evenly: the `cap` with the lowest
/// `draw` (ties by position), kept in their order.
fn thin_evenly<T>(items: Vec<T>, cap: usize, draw: impl Fn(&T) -> f64) -> Vec<T> {
    if items.len() <= cap {
        return items;
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| draw(&items[a]).total_cmp(&draw(&items[b])).then(a.cmp(&b)));
    let mut keep = vec![false; items.len()];
    for &index in &order[..cap] {
        keep[index] = true;
    }
    items
        .into_iter()
        .zip(keep)
        .filter_map(|(item, kept)| kept.then_some(item))
        .collect()
}

/// To the centimetre, so the bundle and the replay carry short numbers.
fn cm(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// To the millimetre.
fn mm(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{build_world, ScenarioOptions};
    use argusnet_world::trees::TreeLayer;
    use serde_json::json;

    fn world(extra: serde_json::Value) -> crate::builder::BuiltWorld {
        let mut options =
            json!({"map_preset": "small", "terrain_preset": "default", "trees": true});
        for (key, value) in extra.as_object().expect("object") {
            options[key] = value.clone();
        }
        build_world(&serde_json::from_value::<ScenarioOptions>(options).expect("options"))
            .expect("world")
    }

    fn class_at(world: &crate::builder::BuiltWorld, xy: [f64; 2]) -> u8 {
        let grid = &world.land_cover;
        let res = world.land_cover_resolution_m;
        let column =
            (((xy[0] - world.bounds.x_min_m) / res).floor().max(0.0) as usize).min(grid.cols - 1);
        let row =
            (((xy[1] - world.bounds.y_min_m) / res).floor().max(0.0) as usize).min(grid.rows - 1);
        grid.classes[row * grid.cols + column]
    }

    #[test]
    fn trees_stand_on_forest_and_scrub_on_the_ground_and_apart() {
        let world = world(json!({}));
        let trees = &world.trees;
        // The small default map has 21 ha of forest and 15 of scrub.
        assert!(trees.len() > 1500, "{}", trees.len());
        let layer = TreeLayer::new(trees.clone()).expect("every tree is valid");
        assert_eq!(layer.len(), trees.len());
        let mut shrubs = 0;
        for tree in trees {
            let class = class_at(&world, tree.xy_m);
            assert!(
                class == LandCoverClass::Forest as u8 || class == LandCoverClass::Scrub as u8,
                "{} on class {class}",
                tree.id
            );
            let ground = world.terrain.height_at(tree.xy_m[0], tree.xy_m[1]);
            assert!((tree.base_z_m - ground).abs() < 0.05, "{}", tree.id);
            shrubs += usize::from(tree.species == Species::Shrub);
        }
        assert!(shrubs > 0);
        // Over scrub about one point in three is kept: the density there is
        // about 0.3 of the forest's.
        let res2 = world.land_cover_resolution_m.powi(2);
        let area = |class: LandCoverClass| {
            world
                .land_cover
                .classes
                .iter()
                .filter(|&&c| c == class as u8)
                .count() as f64
                * res2
        };
        let on = |class: LandCoverClass| {
            trees
                .iter()
                .filter(|t| class_at(&world, t.xy_m) == class as u8)
                .count() as f64
        };
        let ratio = (on(LandCoverClass::Scrub) / area(LandCoverClass::Scrub))
            / (on(LandCoverClass::Forest) / area(LandCoverClass::Forest));
        assert!((0.2..0.4).contains(&ratio), "{ratio}");
        // To the centimetre.
        for tree in trees {
            for v in [
                tree.xy_m[0],
                tree.xy_m[1],
                tree.height_m,
                tree.crown_radius_m,
            ] {
                assert_eq!((v * 100.0).round() / 100.0, v, "{}", tree.id);
            }
        }
        // Spacing survives the rounding to centimetres.
        for tree in trees {
            let near = layer.near(Bounds2d::new(
                tree.xy_m[0] - 7.9,
                tree.xy_m[0] + 7.9,
                tree.xy_m[1] - 7.9,
                tree.xy_m[1] + 7.9,
            ));
            for other in near {
                if other.id != tree.id {
                    let gap = (other.xy_m[0] - tree.xy_m[0]).hypot(other.xy_m[1] - tree.xy_m[1]);
                    assert!(gap >= SPACING_M - 0.015, "{} {} {gap}", tree.id, other.id);
                }
            }
        }
    }

    #[test]
    fn the_forest_is_the_worlds_not_the_runs_and_off_by_default() {
        let one = world(json!({}));
        let two = world(json!({}));
        assert_eq!(one.trees, two.trees);
        let other_terrain = world(json!({"terrain_seed": 3}));
        assert_ne!(one.trees, other_terrain.trees);
        let off = build_world(
            &serde_json::from_value::<ScenarioOptions>(
                json!({"map_preset": "small", "terrain_preset": "default"}),
            )
            .expect("options"),
        )
        .expect("world");
        assert!(off.trees.is_empty());
    }

    #[test]
    fn deciduous_crowns_thin_with_the_season_and_evergreens_do_not() {
        let summer = world(json!({"season_month": 7}));
        let winter = world(json!({"season_month": 1}));
        assert_eq!(summer.trees.len(), winter.trees.len());
        for (s, w) in summer.trees.iter().zip(&winter.trees) {
            assert_eq!((s.id.as_str(), s.species), (w.id.as_str(), w.species));
            match s.species {
                Species::Broadleaf | Species::Shrub => {
                    assert!(w.leaf_density < 0.2 * s.leaf_density + 0.01, "{}", s.id)
                }
                Species::Conifer | Species::Palm => assert_eq!(w.leaf_density, s.leaf_density),
            }
        }
    }

    #[test]
    fn no_tree_stands_in_a_forest_stand_or_a_building() {
        // The small alpine map has a forest stand; urban_dense has buildings.
        for terrain in ["alpine", "urban_dense"] {
            let world = world(json!({"terrain_preset": terrain}));
            let obstacles: Vec<Obstacle> = world
                .obstacles
                .iter()
                .map(|built| {
                    serde_json::from_value::<argusnet_world::obstacles::ObstacleMetadata>(
                        built.metadata.clone(),
                    )
                    .expect("metadata")
                    .into_obstacle()
                    .expect("obstacle")
                })
                .collect();
            assert!(!world.trees.is_empty(), "{terrain}");
            for tree in &world.trees {
                for obstacle in &obstacles {
                    let inside = obstacle.point_inside(
                        tree.xy_m[0],
                        tree.xy_m[1],
                        0.5 * (obstacle.base_z_m + obstacle.top_z_m),
                    );
                    assert!(
                        !(inside
                            && (obstacle.blocker_type.is_hard()
                                || obstacle.kind == "forest-stand-v1")),
                        "{terrain}: {} in {}",
                        tree.id,
                        obstacle.id
                    );
                }
            }
        }
    }

    #[test]
    fn thinning_keeps_the_lowest_draws_in_their_order() {
        let items: Vec<(usize, f64)> = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2]
            .into_iter()
            .enumerate()
            .collect();
        let kept = thin_evenly(items.clone(), 3, |(_, d)| *d);
        assert_eq!(kept, vec![(1, 0.1), (3, 0.3), (5, 0.2)]);
        assert_eq!(thin_evenly(items.clone(), 6, |(_, d)| *d), items);
        assert_eq!(thin_evenly(items.clone(), 7, |(_, d)| *d), items);
    }

    #[test]
    fn a_large_forest_is_thinned_evenly_to_the_cap() {
        let world = world(json!({"map_preset": "regional", "terrain_preset": "jungle_canopy"}));
        assert_eq!(world.trees.len(), MAX_TREES);
        // Evenly: each half of the map keeps a share near its forest's.
        // Evenly: the south half keeps a share near its forest's, not all of
        // the first trees sampled (the sample runs south to north).
        let mid = 0.5 * (world.bounds.y_min_m + world.bounds.y_max_m);
        let south = world.trees.iter().filter(|t| t.xy_m[1] < mid).count() as f64;
        let share = south / MAX_TREES as f64;
        assert!((0.3..0.7).contains(&share), "{share}");
        // Still in the sample's order, row by row (a row is 5.7 m deep).
        for pair in world.trees.windows(2) {
            assert!(
                pair[1].xy_m[1] > pair[0].xy_m[1] - 5.7,
                "{} {}",
                pair[0].id,
                pair[1].id
            );
        }
        assert_eq!(
            world.trees.last().map(|t| t.id.as_str()),
            Some("tree-14999")
        );
    }

    fn bundle(options: serde_json::Value) -> serde_json::Value {
        let mut all = json!({"map_preset": "small", "terrain_preset": "default"});
        for (key, value) in options.as_object().expect("object") {
            all[key] = value.clone();
        }
        let config: crate::builder::ScenarioConfig = serde_json::from_value(json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "duration_s": 5.0,
            "options": all,
        }))
        .expect("config");
        crate::builder::build_bundle(&config).expect("bundle builds")
    }

    /// The option puts trees in the bundle's world, into the run and into
    /// the replay's scene. A world without it is the world it was: no key
    /// anywhere, and the same `Debug` rendering, so the same checkpoint
    /// binding.
    #[test]
    fn a_world_has_trees_only_when_asked() {
        use crate::scenario::{build_world, ScenarioBundle};
        let without = bundle(json!({}));
        let with = bundle(json!({"trees": true}));
        assert!(without["world"].get("trees").is_none());
        assert!(without["presentation"].get("trees").is_none());
        assert!(without["presentation"]["scenario_options"]
            .get("trees")
            .is_none());

        let mut stripped = with["world"].clone();
        stripped.as_object_mut().unwrap().remove("trees");
        assert_eq!(stripped, without["world"]);
        assert_eq!(with["presentation"]["trees"], with["world"]["trees"]);
        assert_eq!(
            with["presentation"]["scenario_options"]["trees"],
            json!(true)
        );

        let parsed = ScenarioBundle::from_json(&with.to_string()).expect("parses");
        let world = build_world(&parsed.world).expect("builds");
        assert!(world.trees.len() > 1500, "{}", world.trees.len());
        let plain = ScenarioBundle::from_json(&without.to_string()).expect("parses");
        assert!(build_world(&plain.world).expect("builds").trees.is_empty());
        assert!(!format!("{:?}", plain.world).contains("trees"));
        assert!(format!("{:?}", parsed.world).contains("trees"));
    }

    /// A tree from data is read as the model reads it: one it refuses, or
    /// two with one id, is a bundle error, not a tree that blocks nothing.
    #[test]
    fn a_bad_tree_in_a_bundle_is_an_error() {
        use crate::scenario::{build_world, ScenarioBundle};
        let mut with = bundle(json!({"trees": true}));
        with["world"]["trees"][0]["crown_base_m"] = json!(1000.0);
        assert!(ScenarioBundle::from_json(&with.to_string()).is_err());

        let mut twice = bundle(json!({"trees": true}));
        let first = twice["world"]["trees"][0].clone();
        twice["world"]["trees"][1]["id"] = first["id"].clone();
        let parsed = ScenarioBundle::from_json(&twice.to_string()).expect("each tree reads");
        let error = build_world(&parsed.world)
            .err()
            .expect("refused")
            .to_string();
        assert!(error.contains("two trees"), "{error}");
    }

    /// `tree_inspection` stands its POIs beside tall forest trees, inside
    /// the canopy, in no crown; it needs the trees and a plain
    /// scan-map-inspect mission.
    #[test]
    fn tree_inspection_pois_stand_in_the_canopy_beside_a_crown() {
        let built = bundle(json!({"trees": true, "tree_inspection": true, "poi_count": 6}));
        let layer = TreeLayer::new(
            serde_json::from_value::<Vec<Tree>>(built["world"]["trees"].clone()).expect("trees"),
        )
        .expect("layer");
        let pois = built["mission"]["pois"].as_array().expect("pois");
        assert_eq!(pois.len(), 6);
        let world = world(json!({}));
        for poi in pois {
            let p: [f64; 3] = serde_json::from_value(poi["position"].clone()).expect("position");
            let agl = p[2] - world.terrain.height_at(p[0], p[1]);
            assert!((agl - 16.0).abs() < 1e-6, "{agl}");
            assert!(layer.crown_at(p).is_none(), "{poi}");
            // Beside the crown of a forest tree taller than the POI by 4 m.
            let id = poi["name"]
                .as_str()
                .expect("name")
                .trim_start_matches("Tree ");
            let tree = layer.trees().iter().find(|t| t.id == id).expect("its tree");
            assert!(
                tree.height_m > 20.0 && tree.species != Species::Shrub,
                "{poi}"
            );
            let gap = (p[0] - tree.xy_m[0]).hypot(p[1] - tree.xy_m[1]) - tree.crown_radius_m;
            assert!((gap - 1.5).abs() < 1e-6, "{poi}: {gap}");
        }
        assert_eq!(
            built["presentation"]["scenario_options"]["tree_inspection"],
            json!(true)
        );
        let config = |options: serde_json::Value| -> crate::builder::ScenarioConfig {
            serde_json::from_value(json!({
                "format": "argusnet.scenario-config",
                "version": 1,
                "duration_s": 5.0,
                "options": options,
            }))
            .expect("config")
        };
        for refused in [
            json!({"map_preset": "small", "tree_inspection": true}),
            json!({"map_preset": "small", "trees": true, "tree_inspection": true, "mission_mode": "target_tracking", "target_count": 1}),
        ] {
            assert!(
                crate::builder::build_bundle(&config(refused.clone())).is_err(),
                "{refused}"
            );
        }
    }
}
