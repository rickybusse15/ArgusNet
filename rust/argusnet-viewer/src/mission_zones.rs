//! Mission-zone presentation logic (viewer-only).
//!
//! This module owns overlap recomputation, contour generation and simplification,
//! render-tier resolution, bounds clipping, and all draw helpers for mission zones.
//! It is intentionally decoupled from replay/protobuf/schema concerns.
//!
//! # Tiering
//!
//! [`ZoneRenderTier`] controls visual density based on camera distance:
//! - **Overview** (`ratio >= 1.15`): one merged contour + one badge per overlap group;
//!   isolated zones draw only a centroid dot; member glyphs suppressed for multi-zone groups.
//! - **Mid** (`0.65 <= ratio < 1.15`): center glyphs shown, sized from tier constants.
//! - **Focused** (`ratio < 0.65` or explicit `ZoneFocus`): selected zone/group rings
//!   render solid; non-focused contours fade to alpha 0.25.
//!
//! # Clipping
//!
//! All group contours and focused rings are clipped to `scene_bounds` before terrain
//! sampling. Segments whose endpoints still miss the terrain mesh are skipped when a
//! mesh is present.
//!
//! # Badge anchoring
//!
//! [`ProjectedZoneBadge`] stores the screen-space pill data for each overlap group.
//! Badges are drawn by the egui layer in `ui.rs`, not as world-space gizmo rectangles.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::orbit_camera::OrbitCamera;
use crate::replay::{MissionZone, TerrainViewerMesh};
use crate::schema::Bounds2d;
use crate::state::{
    LoadedMissionZones, RuntimeOverlayVisibility, ZoneFocus, ZoneOverlapGroup, ZoneOverlapModel,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ZONE_CONNECT_EPSILON_M: f32 = 4.0;
const ZONE_CONTOUR_PADDING_M: f32 = 2.5;
const ZONE_CONTOUR_MIN_CELL_SIZE_M: f32 = 2.0;
const ZONE_CONTOUR_TARGET_CELLS: f32 = 96.0;
const ZONE_GLYPH_LIFT_M: f32 = 1.6;
const ZONE_BADGE_LIFT_M: f32 = 2.2;

/// Minimum screen_x for badges to avoid overlapping the side panel.
const SIDE_PANEL_WIDTH_PX: f32 = 310.0;

/// Badge pill height in screen pixels (must match ui.rs badge rendering).
const BADGE_PILL_HEIGHT: f32 = 20.0;

/// Glyph half-size used in Mid tier (independent of zone `radius_m`).
const MID_TIER_GLYPH_HALF: f32 = 2.8;

/// Centroid dot half-size drawn for isolated zones in Overview tier.
const OVERVIEW_CENTROID_DOT_HALF: f32 = 1.2;

// ---------------------------------------------------------------------------
// Render tier
// ---------------------------------------------------------------------------

/// Camera-distance tier that controls mission-zone visual density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneRenderTier {
    /// Far zoom: merged contour + badge only.
    Overview,
    /// Medium zoom: center glyphs sized from constants.
    Mid,
    /// Close zoom or explicit focus: full detail, non-focused faded.
    Focused,
}

/// Resolve the render tier from the current camera radius and the scene diagonal.
///
/// If the zone focus is not `None`, the tier is always `Focused`.
pub fn resolve_render_tier(
    camera_radius: f32,
    scene_diagonal: f32,
    focus: &ZoneFocus,
) -> ZoneRenderTier {
    if !matches!(focus, ZoneFocus::None) {
        return ZoneRenderTier::Focused;
    }
    let ratio = if scene_diagonal > f32::EPSILON {
        camera_radius / scene_diagonal
    } else {
        1.0
    };
    if ratio >= 1.15 {
        ZoneRenderTier::Overview
    } else if ratio >= 0.65 {
        ZoneRenderTier::Mid
    } else {
        ZoneRenderTier::Focused
    }
}

// ---------------------------------------------------------------------------
// Projected badge (screen-space, viewer-only)
// ---------------------------------------------------------------------------

/// Per-type chip inside a projected badge.
#[derive(Debug, Clone)]
pub struct BadgeChip {
    pub zone_type: String,
    pub count: usize,
    pub color: [f32; 4],
}

/// Screen-space badge data for one overlap group.
///
/// Created each frame from the group's `anchor_xy` projected through the camera.
/// Consumed by `ui.rs` to draw egui pills instead of world-space gizmo rectangles.
#[derive(Debug, Clone, Resource, Default)]
pub struct ProjectedZoneBadges {
    pub badges: Vec<ProjectedZoneBadge>,
}

/// One badge for one overlap group.
#[derive(Debug, Clone)]
pub struct ProjectedZoneBadge {
    pub group_id: usize,
    pub screen_x: f32,
    pub screen_y: f32,
    pub chips: Vec<BadgeChip>,
}

// ---------------------------------------------------------------------------
// Badge layout helpers
// ---------------------------------------------------------------------------

/// Estimate pill width from chip data (mirrors `badge_pill_width` in ui.rs).
fn badge_pill_width_from_chips(chips: &[BadgeChip]) -> f32 {
    let chip_widths: f32 = chips
        .iter()
        .map(|chip| {
            let letter = match chip.zone_type.as_str() {
                "surveillance" => "S",
                "exclusion" => "X",
                "patrol" => "P",
                "objective" => "O",
                _ => "?",
            };
            let text_len = letter.len() + format!("{}", chip.count).len();
            text_len as f32 * 7.0 + 4.0
        })
        .sum();
    chip_widths + 8.0
}

/// Single-pass greedy declutter: push overlapping badges downward.
fn declutter_badges(badges: &mut [ProjectedZoneBadge]) {
    for i in 1..badges.len() {
        let w_i = badge_pill_width_from_chips(&badges[i].chips);
        for j in 0..i {
            let w_j = badge_pill_width_from_chips(&badges[j].chips);

            let overlap_x = badges[i].screen_x < badges[j].screen_x + w_j
                && badges[i].screen_x + w_i > badges[j].screen_x;
            let overlap_y = badges[i].screen_y < badges[j].screen_y + BADGE_PILL_HEIGHT
                && badges[i].screen_y + BADGE_PILL_HEIGHT > badges[j].screen_y;

            if overlap_x && overlap_y {
                badges[i].screen_y = badges[j].screen_y + BADGE_PILL_HEIGHT + 2.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Overlap model construction
// ---------------------------------------------------------------------------

pub fn build_zone_overlap_model(zones: &[MissionZone], focus: ZoneFocus) -> ZoneOverlapModel {
    let components = compute_zone_overlap_components(zones);
    let mut zone_to_group = HashMap::new();
    let mut groups = Vec::with_capacity(components.len());

    for (group_index, zone_indices) in components.into_iter().enumerate() {
        for &zone_index in &zone_indices {
            zone_to_group.insert(zone_index, group_index);
        }

        let centroid_xy = group_centroid_xy(zones, &zone_indices);
        let contour_segments = build_group_contour_segments(zones, &zone_indices);
        let mut type_counts = HashMap::new();
        for &zone_index in &zone_indices {
            if let Some(zone) = zones.get(zone_index) {
                *type_counts.entry(zone.zone_type.clone()).or_insert(0) += 1;
            }
        }

        let anchor_xy = centroid_xy;
        let bounds_xy = compute_group_bounds_xy(zones, &zone_indices);

        groups.push(ZoneOverlapGroup {
            group_id: group_index,
            zone_indices,
            centroid_xy,
            contour_segments,
            type_counts,
            anchor_xy,
            bounds_xy,
        });
    }

    let focus = match focus {
        ZoneFocus::Zone(zone_index) if zone_index < zones.len() => ZoneFocus::Zone(zone_index),
        ZoneFocus::Group(group_index) if group_index < groups.len() => {
            ZoneFocus::Group(group_index)
        }
        _ => ZoneFocus::None,
    };

    ZoneOverlapModel {
        groups,
        zone_to_group,
        focus,
        generation: 0,
    }
}

fn compute_group_bounds_xy(zones: &[MissionZone], zone_indices: &[usize]) -> [f32; 4] {
    let Some((&first, rest)) = zone_indices.split_first() else {
        return [0.0; 4];
    };
    let z = &zones[first];
    let mut min_x = z.center[0] - z.radius_m;
    let mut max_x = z.center[0] + z.radius_m;
    let mut min_y = z.center[1] - z.radius_m;
    let mut max_y = z.center[1] + z.radius_m;
    for &idx in rest {
        let z = &zones[idx];
        min_x = min_x.min(z.center[0] - z.radius_m);
        max_x = max_x.max(z.center[0] + z.radius_m);
        min_y = min_y.min(z.center[1] - z.radius_m);
        max_y = max_y.max(z.center[1] + z.radius_m);
    }
    [min_x, max_x, min_y, max_y]
}

pub fn compute_zone_overlap_components(zones: &[MissionZone]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; zones.len()];
    let mut groups = Vec::new();

    for start in 0..zones.len() {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(index) = stack.pop() {
            component.push(index);
            for candidate in 0..zones.len() {
                if visited[candidate] || candidate == index {
                    continue;
                }
                if zones_are_connected(&zones[index], &zones[candidate]) {
                    visited[candidate] = true;
                    stack.push(candidate);
                }
            }
        }
        component.sort_unstable();
        groups.push(component);
    }

    groups
}

fn zones_are_connected(a: &MissionZone, b: &MissionZone) -> bool {
    let dx = a.center[0] - b.center[0];
    let dy = a.center[1] - b.center[1];
    let limit = a.radius_m + b.radius_m + ZONE_CONNECT_EPSILON_M;
    (dx * dx) + (dy * dy) <= limit * limit
}

fn group_centroid_xy(zones: &[MissionZone], zone_indices: &[usize]) -> [f32; 2] {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;
    for &zone_index in zone_indices {
        let Some(zone) = zones.get(zone_index) else {
            continue;
        };
        sum_x += zone.center[0];
        sum_y += zone.center[1];
        count += 1.0;
    }
    if count == 0.0 {
        [0.0, 0.0]
    } else {
        [sum_x / count, sum_y / count]
    }
}

// ---------------------------------------------------------------------------
// Contour generation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ContourKey(i32, i32);

#[derive(Clone, Copy, Debug)]
struct ContourSegment {
    start: ContourKey,
    end: ContourKey,
}

pub fn build_group_contour_segments(
    zones: &[MissionZone],
    zone_indices: &[usize],
) -> Vec<[[f32; 2]; 2]> {
    let Some((&first_index, rest)) = zone_indices.split_first() else {
        return Vec::new();
    };

    let mut min_x =
        zones[first_index].center[0] - zones[first_index].radius_m - ZONE_CONTOUR_PADDING_M;
    let mut max_x =
        zones[first_index].center[0] + zones[first_index].radius_m + ZONE_CONTOUR_PADDING_M;
    let mut min_y =
        zones[first_index].center[1] - zones[first_index].radius_m - ZONE_CONTOUR_PADDING_M;
    let mut max_y =
        zones[first_index].center[1] + zones[first_index].radius_m + ZONE_CONTOUR_PADDING_M;

    for &zone_index in rest {
        let zone = &zones[zone_index];
        min_x = min_x.min(zone.center[0] - zone.radius_m - ZONE_CONTOUR_PADDING_M);
        max_x = max_x.max(zone.center[0] + zone.radius_m + ZONE_CONTOUR_PADDING_M);
        min_y = min_y.min(zone.center[1] - zone.radius_m - ZONE_CONTOUR_PADDING_M);
        max_y = max_y.max(zone.center[1] + zone.radius_m + ZONE_CONTOUR_PADDING_M);
    }

    let span_x = (max_x - min_x).max(ZONE_CONTOUR_MIN_CELL_SIZE_M * 4.0);
    let span_y = (max_y - min_y).max(ZONE_CONTOUR_MIN_CELL_SIZE_M * 4.0);
    let cell_size =
        (span_x.max(span_y) / ZONE_CONTOUR_TARGET_CELLS).max(ZONE_CONTOUR_MIN_CELL_SIZE_M);
    let cols = ((span_x / cell_size).ceil() as usize).max(4) + 2;
    let rows = ((span_y / cell_size).ceil() as usize).max(4) + 2;

    let origin_x = min_x - cell_size;
    let origin_y = min_y - cell_size;
    let mut occupied = vec![false; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let x = origin_x + col as f32 * cell_size;
            let y = origin_y + row as f32 * cell_size;
            occupied[row * cols + col] = zone_indices.iter().any(|&zone_index| {
                let zone = &zones[zone_index];
                let dx = x - zone.center[0];
                let dy = y - zone.center[1];
                let effective_radius = zone.radius_m + ZONE_CONTOUR_PADDING_M;
                (dx * dx) + (dy * dy) <= effective_radius * effective_radius
            });
        }
    }

    let mut segments = Vec::new();
    for row in 0..rows.saturating_sub(1) {
        for col in 0..cols.saturating_sub(1) {
            let bl = occupied[row * cols + col];
            let br = occupied[row * cols + col + 1];
            let tl = occupied[(row + 1) * cols + col];
            let tr = occupied[(row + 1) * cols + col + 1];
            let mask = (bl as u8) | ((br as u8) << 1) | ((tr as u8) << 2) | ((tl as u8) << 3);

            let left = ContourKey((col as i32) * 2, (row as i32) * 2 + 1);
            let bottom = ContourKey((col as i32) * 2 + 1, (row as i32) * 2);
            let right = ContourKey((col as i32) * 2 + 2, (row as i32) * 2 + 1);
            let top = ContourKey((col as i32) * 2 + 1, (row as i32) * 2 + 2);

            match mask {
                0 | 15 => {}
                1 | 14 => segments.push(ContourSegment {
                    start: left,
                    end: bottom,
                }),
                2 | 13 => segments.push(ContourSegment {
                    start: bottom,
                    end: right,
                }),
                3 | 12 => segments.push(ContourSegment {
                    start: left,
                    end: right,
                }),
                4 | 11 => segments.push(ContourSegment {
                    start: right,
                    end: top,
                }),
                5 => {
                    segments.push(ContourSegment {
                        start: left,
                        end: top,
                    });
                    segments.push(ContourSegment {
                        start: bottom,
                        end: right,
                    });
                }
                6 | 9 => segments.push(ContourSegment {
                    start: bottom,
                    end: top,
                }),
                7 | 8 => segments.push(ContourSegment {
                    start: left,
                    end: top,
                }),
                10 => {
                    segments.push(ContourSegment {
                        start: left,
                        end: bottom,
                    });
                    segments.push(ContourSegment {
                        start: top,
                        end: right,
                    });
                }
                _ => {}
            }
        }
    }

    let contour = extract_largest_contour(&segments, origin_x, origin_y, cell_size);
    if contour.len() >= 3 {
        let simplified = simplify_contour(&contour, cell_size * 0.35);
        return simplified
            .iter()
            .enumerate()
            .map(|(index, point)| [*point, simplified[(index + 1) % simplified.len()]])
            .collect();
    }

    fallback_circle_segments(&zones[first_index])
}

fn extract_largest_contour(
    segments: &[ContourSegment],
    origin_x: f32,
    origin_y: f32,
    cell_size: f32,
) -> Vec<[f32; 2]> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut adjacency: HashMap<ContourKey, Vec<usize>> = HashMap::new();
    for (index, segment) in segments.iter().enumerate() {
        adjacency.entry(segment.start).or_default().push(index);
        adjacency.entry(segment.end).or_default().push(index);
    }

    let mut used = vec![false; segments.len()];
    let mut best_contour = Vec::new();
    let mut best_area = 0.0_f32;

    for start_index in 0..segments.len() {
        if used[start_index] {
            continue;
        }

        used[start_index] = true;
        let first = segments[start_index];
        let mut keys = vec![first.start, first.end];
        let mut current = first.end;

        while let Some(candidate_indices) = adjacency.get(&current) {
            let next_index = candidate_indices
                .iter()
                .copied()
                .find(|&index| !used[index]);
            let Some(next_index) = next_index else {
                break;
            };

            used[next_index] = true;
            let next_segment = segments[next_index];
            let next_key = if next_segment.start == current {
                next_segment.end
            } else {
                next_segment.start
            };
            if next_key == keys[0] {
                break;
            }
            keys.push(next_key);
            current = next_key;
        }

        if keys.len() < 3 {
            continue;
        }

        let contour: Vec<[f32; 2]> = keys
            .into_iter()
            .map(|key| {
                [
                    origin_x + key.0 as f32 * cell_size * 0.5,
                    origin_y + key.1 as f32 * cell_size * 0.5,
                ]
            })
            .collect();
        let area = polygon_area(&contour).abs();
        if area > best_area {
            best_area = area;
            best_contour = contour;
        }
    }

    best_contour
}

/// Ramer–Douglas–Peucker simplification for closed contours.
fn simplify_contour(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if points.len() <= 4 || epsilon <= 0.0 {
        return points.to_vec();
    }
    let mut keep = vec![true; points.len()];
    rdp_range(points, 0, points.len() - 1, epsilon, &mut keep);
    points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
}

fn rdp_range(points: &[[f32; 2]], start: usize, end: usize, epsilon: f32, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let mut max_dist = 0.0_f32;
    let mut max_index = start;
    let sx = points[start][0];
    let sy = points[start][1];
    let ex = points[end][0];
    let ey = points[end][1];
    let dx = ex - sx;
    let dy = ey - sy;
    let len_sq = dx * dx + dy * dy;
    for (index, point) in points.iter().enumerate().take(end).skip(start + 1) {
        let px = point[0] - sx;
        let py = point[1] - sy;
        let dist = if len_sq < f32::EPSILON {
            (px * px + py * py).sqrt()
        } else {
            ((px * dy - py * dx).abs()) / len_sq.sqrt()
        };
        if dist > max_dist {
            max_dist = dist;
            max_index = index;
        }
    }
    if max_dist > epsilon {
        rdp_range(points, start, max_index, epsilon, keep);
        rdp_range(points, max_index, end, epsilon, keep);
    } else {
        for item in keep.iter_mut().take(end).skip(start + 1) {
            *item = false;
        }
    }
}

pub fn polygon_area(points: &[[f32; 2]]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for (index, point) in points.iter().enumerate() {
        let next = points[(index + 1) % points.len()];
        area += point[0] * next[1] - next[0] * point[1];
    }
    area * 0.5
}

fn fallback_circle_segments(zone: &MissionZone) -> Vec<[[f32; 2]; 2]> {
    let mut points = Vec::with_capacity(40);
    for index in 0..40 {
        let angle = std::f32::consts::TAU * index as f32 / 40.0;
        points.push([
            zone.center[0] + angle.cos() * zone.radius_m,
            zone.center[1] + angle.sin() * zone.radius_m,
        ]);
    }

    points
        .iter()
        .enumerate()
        .map(|(index, point)| [*point, points[(index + 1) % points.len()]])
        .collect()
}

// ---------------------------------------------------------------------------
// Clipping
// ---------------------------------------------------------------------------

/// Clip a single line segment to an axis-aligned rectangle using
/// Liang-Barsky parametric clipping.  Returns `None` when the segment
/// lies entirely outside the bounds.
pub fn clip_segment_to_bounds(
    p0: [f32; 2],
    p1: [f32; 2],
    bounds: &Bounds2d,
) -> Option<[[f32; 2]; 2]> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let p_vals = [-dx, dx, -dy, dy];
    let q_vals = [
        p0[0] - bounds.x_min_m,
        bounds.x_max_m - p0[0],
        p0[1] - bounds.y_min_m,
        bounds.y_max_m - p0[1],
    ];

    let mut t0: f32 = 0.0;
    let mut t1: f32 = 1.0;

    for i in 0..4 {
        let p = p_vals[i];
        let q = q_vals[i];
        if p.abs() <= 1.0e-12 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            t0 = t0.max(r);
        } else {
            t1 = t1.min(r);
        }
        if t0 > t1 {
            return None;
        }
    }

    Some([
        [p0[0] + t0 * dx, p0[1] + t0 * dy],
        [p0[0] + t1 * dx, p0[1] + t1 * dy],
    ])
}

/// Clip contour segments to scene bounds using Liang-Barsky line clipping.
/// Segments crossing the boundary are clipped to the edge rather than dropped.
pub fn clip_segments_to_bounds(
    segments: &[[[f32; 2]; 2]],
    bounds: &Bounds2d,
) -> Vec<[[f32; 2]; 2]> {
    segments
        .iter()
        .filter_map(|seg| clip_segment_to_bounds(seg[0], seg[1], bounds))
        .collect()
}

// ---------------------------------------------------------------------------
// Terrain sampling
// ---------------------------------------------------------------------------

pub fn sample_terrain_height(
    terrain_mesh: &Option<TerrainViewerMesh>,
    fallback_height: f32,
    x_m: f32,
    y_m: f32,
) -> f32 {
    terrain_mesh
        .as_ref()
        .and_then(|mesh| mesh.sample_height(x_m, y_m))
        .unwrap_or(fallback_height)
}

/// Try to sample terrain height; returns `None` when a mesh exists but the
/// point is outside its coverage. Used to skip segments that leak off-map.
fn try_terrain_height(terrain_mesh: &Option<TerrainViewerMesh>, x_m: f32, y_m: f32) -> Option<f32> {
    terrain_mesh.as_ref()?.sample_height(x_m, y_m)
}

pub fn average_zone_height(zones: &[MissionZone], zone_indices: &[usize]) -> f32 {
    let mut total = 0.0;
    let mut count = 0.0;
    for &zone_index in zone_indices {
        let Some(zone) = zones.get(zone_index) else {
            continue;
        };
        total += zone.center[2];
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

pub fn zone_color(zone_type: &str, alpha: f32) -> Color {
    match zone_type {
        "surveillance" => Color::srgba(0.2, 0.45, 0.95, alpha),
        "exclusion" => Color::srgba(0.95, 0.2, 0.2, alpha),
        "patrol" => Color::srgba(0.96, 0.82, 0.18, alpha),
        "objective" => Color::srgba(0.18, 0.84, 0.34, alpha),
        _ => Color::srgba(0.7, 0.72, 0.74, alpha),
    }
}

pub fn zone_color_rgba(zone_type: &str, alpha: f32) -> [f32; 4] {
    match zone_type {
        "surveillance" => [0.2, 0.45, 0.95, alpha],
        "exclusion" => [0.95, 0.2, 0.2, alpha],
        "patrol" => [0.96, 0.82, 0.18, alpha],
        "objective" => [0.18, 0.84, 0.34, alpha],
        _ => [0.7, 0.72, 0.74, alpha],
    }
}

/// Return the zone type with the highest count in a group.
fn dominant_zone_type(group: &ZoneOverlapGroup) -> &str {
    group
        .type_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(t, _)| t.as_str())
        .unwrap_or("unknown")
}

fn group_outline_color(group: &ZoneOverlapGroup) -> Color {
    if group.type_counts.len() == 1 {
        let zone_type = group
            .type_counts
            .keys()
            .next()
            .map(String::as_str)
            .unwrap_or("unknown");
        return zone_color(zone_type, 0.72);
    }
    zone_color(dominant_zone_type(group), 0.78)
}

// ---------------------------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------------------------

fn draw_xy_rect(
    gizmos: &mut Gizmos,
    center: Vec3,
    half_width: f32,
    half_height: f32,
    color: Color,
) {
    let a = center + Vec3::new(-half_width, -half_height, 0.0);
    let b = center + Vec3::new(half_width, -half_height, 0.0);
    let c = center + Vec3::new(half_width, half_height, 0.0);
    let d = center + Vec3::new(-half_width, half_height, 0.0);
    gizmos.line(a, b, color);
    gizmos.line(b, c, color);
    gizmos.line(c, d, color);
    gizmos.line(d, a, color);
}

fn draw_zone_center_glyph(
    gizmos: &mut Gizmos,
    zone: &MissionZone,
    terrain_mesh: &Option<TerrainViewerMesh>,
    half: f32,
) {
    let color = zone_color(&zone.zone_type, 0.92);
    let center = Vec3::from_array(zone.center);
    let center_z =
        sample_terrain_height(terrain_mesh, zone.center[2], center.x, center.y) + ZONE_GLYPH_LIFT_M;
    let center = Vec3::new(center.x, center.y, center_z);

    match zone.zone_type.as_str() {
        "exclusion" => {
            gizmos.line(
                center - Vec3::new(half, half, 0.0),
                center + Vec3::new(half, half, 0.0),
                color,
            );
            gizmos.line(
                center + Vec3::new(half, -half, 0.0),
                center - Vec3::new(half, -half, 0.0),
                color,
            );
        }
        "patrol" => {
            let top = center + Vec3::Y * half;
            let right = center + Vec3::X * half;
            let bottom = center - Vec3::Y * half;
            let left = center - Vec3::X * half;
            gizmos.line(top, right, color);
            gizmos.line(right, bottom, color);
            gizmos.line(bottom, left, color);
            gizmos.line(left, top, color);
        }
        "objective" => {
            draw_xy_rect(gizmos, center, half * 1.2, half * 1.2, color);
        }
        _ => {
            gizmos.line(center - Vec3::X * half, center + Vec3::X * half, color);
            gizmos.line(center - Vec3::Y * half, center + Vec3::Y * half, color);
        }
    }
}

/// Draw a centroid dot for an isolated zone in Overview tier.
fn draw_centroid_dot(
    gizmos: &mut Gizmos,
    zone: &MissionZone,
    terrain_mesh: &Option<TerrainViewerMesh>,
) {
    let color = zone_color(&zone.zone_type, 0.7);
    let center = Vec3::from_array(zone.center);
    let z =
        sample_terrain_height(terrain_mesh, zone.center[2], center.x, center.y) + ZONE_GLYPH_LIFT_M;
    let c = Vec3::new(center.x, center.y, z);
    let h = OVERVIEW_CENTROID_DOT_HALF;
    gizmos.line(c - Vec3::X * h, c + Vec3::X * h, color);
    gizmos.line(c - Vec3::Y * h, c + Vec3::Y * h, color);
}

fn draw_zone_ring(
    gizmos: &mut Gizmos,
    zone: &MissionZone,
    terrain_mesh: &Option<TerrainViewerMesh>,
    alpha: f32,
    scene_bounds: &Bounds2d,
) {
    let center = Vec3::from_array(zone.center);
    let color = zone_color(&zone.zone_type, alpha);
    let seg_count = 48;
    let arc_step_m = std::f32::consts::TAU / seg_count as f32 * zone.radius_m;
    let inset = scene_bounds.inset(arc_step_m);
    for i in 0..seg_count {
        let a0 = std::f32::consts::TAU * i as f32 / seg_count as f32;
        let a1 = std::f32::consts::TAU * (i + 1) as f32 / seg_count as f32;
        let raw_p0 = [
            center.x + a0.cos() * zone.radius_m,
            center.y + a0.sin() * zone.radius_m,
        ];
        let raw_p1 = [
            center.x + a1.cos() * zone.radius_m,
            center.y + a1.sin() * zone.radius_m,
        ];
        // Clip ring segments to inset scene bounds.
        let Some([p0_xy, p1_xy]) = clip_segment_to_bounds(raw_p0, raw_p1, &inset) else {
            continue;
        };
        // When terrain mesh exists, skip segments that can't be sampled.
        if terrain_mesh.is_some() {
            let h0 = try_terrain_height(terrain_mesh, p0_xy[0], p0_xy[1]);
            let h1 = try_terrain_height(terrain_mesh, p1_xy[0], p1_xy[1]);
            if h0.is_none() || h1.is_none() {
                continue;
            }
            let p0 = Vec3::new(p0_xy[0], p0_xy[1], h0.unwrap() + ZONE_GLYPH_LIFT_M);
            let p1 = Vec3::new(p1_xy[0], p1_xy[1], h1.unwrap() + ZONE_GLYPH_LIFT_M);
            gizmos.line(p0, p1, color);
        } else {
            let p0 = Vec3::new(p0_xy[0], p0_xy[1], zone.center[2] + ZONE_GLYPH_LIFT_M);
            let p1 = Vec3::new(p1_xy[0], p1_xy[1], zone.center[2] + ZONE_GLYPH_LIFT_M);
            gizmos.line(p0, p1, color);
        }
    }
}

fn draw_zone_group_outline(
    gizmos: &mut Gizmos,
    group: &ZoneOverlapGroup,
    zones: &[MissionZone],
    terrain_mesh: &Option<TerrainViewerMesh>,
    scene_bounds: &Bounds2d,
    alpha: f32,
) {
    if group.contour_segments.is_empty() {
        return;
    }

    let inset = scene_bounds.inset(ZONE_CONTOUR_MIN_CELL_SIZE_M);
    let clipped = clip_segments_to_bounds(&group.contour_segments, &inset);
    let fallback_height = average_zone_height(zones, &group.zone_indices);
    let color = {
        let c = group_outline_color(group);
        if alpha < 0.99 {
            // Re-create with adjusted alpha.
            if group.type_counts.len() == 1 {
                let zone_type = group
                    .type_counts
                    .keys()
                    .next()
                    .map(String::as_str)
                    .unwrap_or("unknown");
                zone_color(zone_type, alpha)
            } else {
                zone_color(dominant_zone_type(group), alpha)
            }
        } else {
            c
        }
    };

    for (index, segment) in clipped.iter().enumerate() {
        if index % 2 == 1 {
            continue;
        }
        // When terrain mesh exists, skip segments whose endpoints can't be sampled.
        if terrain_mesh.is_some() {
            let h0 = try_terrain_height(terrain_mesh, segment[0][0], segment[0][1]);
            let h1 = try_terrain_height(terrain_mesh, segment[1][0], segment[1][1]);
            if h0.is_none() || h1.is_none() {
                continue;
            }
            let start = Vec3::new(
                segment[0][0],
                segment[0][1],
                h0.unwrap() + ZONE_GLYPH_LIFT_M,
            );
            let end = Vec3::new(
                segment[1][0],
                segment[1][1],
                h1.unwrap() + ZONE_GLYPH_LIFT_M,
            );
            gizmos.line(start, end, color);
        } else {
            let start = Vec3::new(
                segment[0][0],
                segment[0][1],
                fallback_height + ZONE_GLYPH_LIFT_M,
            );
            let end = Vec3::new(
                segment[1][0],
                segment[1][1],
                fallback_height + ZONE_GLYPH_LIFT_M,
            );
            gizmos.line(start, end, color);
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Recompute the overlap model when zones change.
pub fn refresh_zone_overlap_model_system(
    zones: Res<LoadedMissionZones>,
    mut overlap_model: ResMut<ZoneOverlapModel>,
) {
    if !zones.is_changed() {
        return;
    }

    let mut next_model = build_zone_overlap_model(&zones.zones, ZoneFocus::None);
    next_model.generation = overlap_model.generation.saturating_add(1).max(1);
    *overlap_model = next_model;
}

/// Main draw system: renders mission zones with tiered visual density.
pub fn draw_mission_zones_system(
    mut gizmos: Gizmos,
    zones: Res<LoadedMissionZones>,
    overlap_model: Res<ZoneOverlapModel>,
    runtime_visibility: Res<RuntimeOverlayVisibility>,
    scene_package: Res<crate::schema::ScenePackage>,
    camera_query: Query<&OrbitCamera>,
) {
    if !runtime_visibility.zones || zones.zones.is_empty() {
        return;
    }

    let scene_bounds = &scene_package.environment.bounds_xy_m;
    let scene_span = scene_bounds.span_xy();
    let scene_diagonal = scene_span[0].hypot(scene_span[1]).max(1.0);

    let camera_radius = camera_query
        .iter()
        .next()
        .map(|cam| cam.radius)
        .unwrap_or(scene_diagonal * 1.2);

    let tier = resolve_render_tier(camera_radius, scene_diagonal, &overlap_model.focus);

    match tier {
        ZoneRenderTier::Overview => {
            // One merged contour + one badge per group; isolated zones get a centroid dot.
            // Multi-zone groups suppress member glyphs.
            for group in &overlap_model.groups {
                draw_zone_group_outline(
                    &mut gizmos,
                    group,
                    &zones.zones,
                    &zones.terrain_mesh,
                    scene_bounds,
                    0.72,
                );
                // Isolated zones: just a dot, no full glyph.
                if group.zone_indices.len() == 1 {
                    if let Some(zone) = zones.zones.get(group.zone_indices[0]) {
                        draw_centroid_dot(&mut gizmos, zone, &zones.terrain_mesh);
                    }
                }
                // Badges are drawn in screen space by the egui layer, not here.
            }
        }
        ZoneRenderTier::Mid => {
            // Contours + center glyphs sized from tier constants.
            for group in &overlap_model.groups {
                draw_zone_group_outline(
                    &mut gizmos,
                    group,
                    &zones.zones,
                    &zones.terrain_mesh,
                    scene_bounds,
                    0.72,
                );
            }
            for zone in &zones.zones {
                draw_zone_center_glyph(&mut gizmos, zone, &zones.terrain_mesh, MID_TIER_GLYPH_HALF);
            }
        }
        ZoneRenderTier::Focused => {
            let focused_indices = overlap_model.focused_zone_indices();
            let has_focus = !focused_indices.is_empty();

            // Non-focused contours fade.
            for group in &overlap_model.groups {
                let is_focused = match &overlap_model.focus {
                    ZoneFocus::Group(g) => *g == group.group_id,
                    ZoneFocus::Zone(z) => group.zone_indices.contains(z),
                    ZoneFocus::None => true,
                };
                let alpha = if has_focus && !is_focused { 0.25 } else { 0.72 };
                draw_zone_group_outline(
                    &mut gizmos,
                    group,
                    &zones.zones,
                    &zones.terrain_mesh,
                    scene_bounds,
                    alpha,
                );
            }

            // All zones get glyphs.
            for zone in &zones.zones {
                let half = zone.radius_m.clamp(6.0, 18.0) * 0.22;
                draw_zone_center_glyph(&mut gizmos, zone, &zones.terrain_mesh, half);
            }

            // Focused zone rings at full alpha.
            for zone_index in focused_indices {
                let Some(zone) = zones.zones.get(zone_index) else {
                    continue;
                };
                draw_zone_ring(&mut gizmos, zone, &zones.terrain_mesh, 0.92, scene_bounds);
            }
        }
    }
}

/// Build projected badge data for the egui layer.
pub fn build_projected_badges_system(
    zones: Res<LoadedMissionZones>,
    overlap_model: Res<ZoneOverlapModel>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    scene_package: Res<crate::schema::ScenePackage>,
    runtime_visibility: Res<RuntimeOverlayVisibility>,
    mut badges_res: ResMut<ProjectedZoneBadges>,
) {
    badges_res.badges.clear();
    if !runtime_visibility.zones || zones.zones.is_empty() {
        return;
    }

    let Ok((camera, camera_transform)) = camera_query.get_single() else {
        return;
    };

    let scene_bounds = &scene_package.environment.bounds_xy_m;
    let scene_span = scene_bounds.span_xy();
    let scene_diagonal = scene_span[0].hypot(scene_span[1]).max(1.0);

    // Only show badges in Overview and Mid tiers.
    let orbit_radius = {
        // We don't have OrbitCamera in this query, approximate from camera position.
        let eye = camera_transform.translation();
        let center = scene_bounds.center_xy();
        let focus = Vec3::new(center[0], center[1], eye.z * 0.3);
        eye.distance(focus)
    };
    let tier = resolve_render_tier(orbit_radius, scene_diagonal, &overlap_model.focus);
    if tier == ZoneRenderTier::Focused && !matches!(overlap_model.focus, ZoneFocus::None) {
        // In focused mode with active focus, don't show badges.
        return;
    }

    for group in &overlap_model.groups {
        let fallback_height = average_zone_height(&zones.zones, &group.zone_indices);
        let world_z = sample_terrain_height(
            &zones.terrain_mesh,
            fallback_height,
            group.anchor_xy[0],
            group.anchor_xy[1],
        ) + ZONE_BADGE_LIFT_M;
        let world_pos = Vec3::new(group.anchor_xy[0], group.anchor_xy[1], world_z);

        let Some(ndc) = camera.world_to_ndc(camera_transform, world_pos) else {
            continue;
        };
        // Skip if behind camera.
        if ndc.z < 0.0 || ndc.z > 1.0 {
            continue;
        }

        let Some(viewport) = camera.logical_viewport_size() else {
            continue;
        };

        let screen_x = ((ndc.x + 1.0) * 0.5 * viewport.x + 18.0).max(SIDE_PANEL_WIDTH_PX);
        let screen_y = (1.0 - ndc.y) * 0.5 * viewport.y - 18.0;

        let mut chips = Vec::new();
        let mut counts: Vec<_> = group.type_counts.iter().collect();
        counts.sort_by(|a, b| a.0.cmp(b.0));
        for (zone_type, count) in counts {
            chips.push(BadgeChip {
                zone_type: zone_type.clone(),
                count: *count,
                color: zone_color_rgba(zone_type, 0.9),
            });
        }

        badges_res.badges.push(ProjectedZoneBadge {
            group_id: group.group_id,
            screen_x,
            screen_y,
            chips,
        });
    }

    declutter_badges(&mut badges_res.badges);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::MissionZone;
    use crate::state::ZoneFocus;

    fn zone(zone_id: &str, zone_type: &str, x: f32, y: f32, radius_m: f32) -> MissionZone {
        MissionZone {
            zone_id: zone_id.into(),
            zone_type: zone_type.into(),
            center: [x, y, 12.0],
            radius_m,
            priority: 1,
            label: zone_id.into(),
        }
    }

    // -- Tier resolution tests --

    #[test]
    fn tier_overview_at_high_ratio() {
        let tier = resolve_render_tier(1200.0, 1000.0, &ZoneFocus::None);
        assert_eq!(tier, ZoneRenderTier::Overview);
    }

    #[test]
    fn tier_overview_at_boundary() {
        // ratio = 1.15 exactly → Overview
        let tier = resolve_render_tier(1150.0, 1000.0, &ZoneFocus::None);
        assert_eq!(tier, ZoneRenderTier::Overview);
    }

    #[test]
    fn tier_mid_between_boundaries() {
        let tier = resolve_render_tier(900.0, 1000.0, &ZoneFocus::None);
        assert_eq!(tier, ZoneRenderTier::Mid);
    }

    #[test]
    fn tier_mid_at_lower_boundary() {
        // ratio = 0.65 exactly → Mid
        let tier = resolve_render_tier(650.0, 1000.0, &ZoneFocus::None);
        assert_eq!(tier, ZoneRenderTier::Mid);
    }

    #[test]
    fn tier_focused_below_lower_boundary() {
        let tier = resolve_render_tier(600.0, 1000.0, &ZoneFocus::None);
        assert_eq!(tier, ZoneRenderTier::Focused);
    }

    #[test]
    fn tier_focused_when_zone_focus_set() {
        // Even at overview distance, explicit focus forces Focused tier.
        let tier = resolve_render_tier(1200.0, 1000.0, &ZoneFocus::Zone(0));
        assert_eq!(tier, ZoneRenderTier::Focused);
    }

    #[test]
    fn tier_focused_when_group_focus_set() {
        let tier = resolve_render_tier(1200.0, 1000.0, &ZoneFocus::Group(0));
        assert_eq!(tier, ZoneRenderTier::Focused);
    }

    // -- Contour clipping tests --

    #[test]
    fn clip_fully_inside_segments() {
        let bounds = Bounds2d {
            x_min_m: -100.0,
            x_max_m: 100.0,
            y_min_m: -100.0,
            y_max_m: 100.0,
        };
        let segments = vec![[[0.0, 0.0], [50.0, 50.0]], [[-20.0, -20.0], [20.0, 20.0]]];
        let clipped = clip_segments_to_bounds(&segments, &bounds);
        assert_eq!(clipped.len(), 2);
    }

    #[test]
    fn clip_fully_outside_segments() {
        let bounds = Bounds2d {
            x_min_m: -100.0,
            x_max_m: 100.0,
            y_min_m: -100.0,
            y_max_m: 100.0,
        };
        let segments = vec![[[200.0, 200.0], [300.0, 300.0]]];
        let clipped = clip_segments_to_bounds(&segments, &bounds);
        assert_eq!(clipped.len(), 0);
    }

    #[test]
    fn clip_crossing_segment_is_clipped() {
        let bounds = Bounds2d {
            x_min_m: -100.0,
            x_max_m: 100.0,
            y_min_m: -100.0,
            y_max_m: 100.0,
        };
        // One endpoint inside, one outside → clipped to boundary.
        let segments = vec![[[-50.0, 0.0], [150.0, 0.0]]];
        let clipped = clip_segments_to_bounds(&segments, &bounds);
        assert_eq!(clipped.len(), 1);
        assert!((clipped[0][0][0] - (-50.0)).abs() < 0.01);
        assert!((clipped[0][1][0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn clip_both_outside_crossing() {
        let bounds = Bounds2d {
            x_min_m: -100.0,
            x_max_m: 100.0,
            y_min_m: -100.0,
            y_max_m: 100.0,
        };
        // Both endpoints outside but segment crosses through bounds.
        let segments = vec![[[-150.0, 0.0], [150.0, 0.0]]];
        let clipped = clip_segments_to_bounds(&segments, &bounds);
        assert_eq!(clipped.len(), 1);
        assert!((clipped[0][0][0] - (-100.0)).abs() < 0.01);
        assert!((clipped[0][1][0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn inset_bounds_shrinks_correctly() {
        let bounds = Bounds2d {
            x_min_m: -100.0,
            x_max_m: 100.0,
            y_min_m: -100.0,
            y_max_m: 100.0,
        };
        let inset = bounds.inset(5.0);
        assert!((inset.x_min_m - (-95.0)).abs() < 0.01);
        assert!((inset.x_max_m - 95.0).abs() < 0.01);
        assert!((inset.y_min_m - (-95.0)).abs() < 0.01);
        assert!((inset.y_max_m - 95.0).abs() < 0.01);
    }

    // -- Badge model generation tests --

    #[test]
    fn badge_model_isolated_group() {
        let zones = vec![zone("a", "surveillance", 0.0, 0.0, 20.0)];
        let model = build_zone_overlap_model(&zones, ZoneFocus::None);
        assert_eq!(model.groups.len(), 1);
        assert_eq!(model.groups[0].type_counts.len(), 1);
        assert_eq!(model.groups[0].type_counts["surveillance"], 1);
        assert_eq!(model.groups[0].anchor_xy, model.groups[0].centroid_xy);
    }

    #[test]
    fn badge_model_single_type_overlap() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "surveillance", 32.0, 0.0, 20.0),
        ];
        let model = build_zone_overlap_model(&zones, ZoneFocus::None);
        assert_eq!(model.groups.len(), 1);
        assert_eq!(model.groups[0].type_counts.len(), 1);
        assert_eq!(model.groups[0].type_counts["surveillance"], 2);
    }

    #[test]
    fn badge_model_mixed_type_overlap() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 32.0, 0.0, 20.0),
            zone("c", "patrol", 60.0, 0.0, 20.0),
        ];
        let model = build_zone_overlap_model(&zones, ZoneFocus::None);
        assert_eq!(model.groups.len(), 1);
        let g = &model.groups[0];
        assert_eq!(g.type_counts.len(), 3);
        assert_eq!(g.type_counts["surveillance"], 1);
        assert_eq!(g.type_counts["exclusion"], 1);
        assert_eq!(g.type_counts["patrol"], 1);
        // bounds_xy should cover all zones.
        assert!(g.bounds_xy[0] < -15.0); // min_x
        assert!(g.bounds_xy[1] > 75.0); // max_x
    }

    // -- Overlap grouping tests (previously in app.rs tests) --

    #[test]
    fn overlap_grouping_splits_isolated_and_merges_connected() {
        let isolated = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 80.0, 0.0, 20.0),
        ];
        let isolated_groups = compute_zone_overlap_components(&isolated);
        assert_eq!(isolated_groups.len(), 2);

        let pair = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 32.0, 0.0, 20.0),
        ];
        let pair_groups = compute_zone_overlap_components(&pair);
        assert_eq!(pair_groups, vec![vec![0, 1]]);
    }

    #[test]
    fn contour_generation_returns_geometry() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 35.0, 0.0, 20.0),
            zone("c", "objective", 70.0, 0.0, 20.0),
        ];
        let groups = compute_zone_overlap_components(&zones);
        assert_eq!(groups.len(), 1);

        let contour = build_group_contour_segments(&zones, &groups[0]);
        assert!(contour.len() > 8);
    }

    #[test]
    fn focused_zone_indices_expand_correctly() {
        let zones = vec![
            zone("a", "surveillance", 0.0, 0.0, 20.0),
            zone("b", "exclusion", 32.0, 0.0, 20.0),
            zone("c", "objective", 120.0, 0.0, 20.0),
        ];

        let zone_focus = build_zone_overlap_model(&zones, ZoneFocus::Zone(1));
        assert_eq!(zone_focus.focused_zone_indices(), vec![1]);

        let group_focus = build_zone_overlap_model(&zones, ZoneFocus::Group(0));
        assert_eq!(group_focus.focused_zone_indices(), vec![0, 1]);
    }

    #[test]
    fn terrain_sampling_with_and_without_mesh() {
        use crate::replay::TerrainViewerMesh;

        let mesh = TerrainViewerMesh {
            x_min_m: -10.0,
            x_max_m: 10.0,
            y_min_m: -10.0,
            y_max_m: 10.0,
            rows: 2,
            cols: 2,
            heights_m: vec![vec![10.0, 14.0], vec![18.0, 22.0]],
        };

        let sampled = sample_terrain_height(&Some(mesh), 42.0, 0.0, 0.0);
        assert!((sampled - 16.0).abs() < 1.0e-6);
        assert_eq!(sample_terrain_height(&None, 42.0, 0.0, 0.0), 42.0);
    }
}
