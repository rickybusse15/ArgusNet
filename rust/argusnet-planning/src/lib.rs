//! Planning: battery accounting, and later the frontier, POI and
//! deconfliction work P4.7 names.

pub mod battery;
pub mod frontier;
pub mod planner;
pub mod route;

pub use battery::{BatteryModel, BatteryState};
pub use route::{astar_path, build_visibility_graph, smooth_path, RouteConfig};
