from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._glb import SimpleMesh, merge_meshes, solid_color_array, triangle_normals, write_glb
from .environment import LandCoverClass


@dataclass(frozen=True)
class SceneLayerMesh:
    layer_id: str
    semantic_kind: str
    mesh: SimpleMesh
    style_id: Optional[str] = None


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / length).astype(np.float32)


def grid_normals(heights_m: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    rows, cols = heights_m.shape
    normals = np.zeros((rows, cols, 3), dtype=np.float32)
    for row in range(rows):
        y0 = y_values[max(row - 1, 0)]
        y1 = y_values[min(row + 1, rows - 1)]
        for col in range(cols):
            x0 = x_values[max(col - 1, 0)]
            x1 = x_values[min(col + 1, cols - 1)]
            dz_dx = (heights_m[row, min(col + 1, cols - 1)] - heights_m[row, max(col - 1, 0)]) / max(x1 - x0, 1.0e-6)
            dz_dy = (heights_m[min(row + 1, rows - 1), col] - heights_m[max(row - 1, 0), col]) / max(y1 - y0, 1.0e-6)
            normals[row, col] = normalize(np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float32))
    return normals.reshape((-1, 3))


def _hash_2d(ix: int, iy: int) -> float:
    """Deterministic hash producing a float in [-1, 1]."""
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFF) / 32768.0 - 1.0


def _value_noise_2d(x: float, y: float) -> float:
    """Smooth 2D value noise, output in [-1, 1]."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    c00 = _hash_2d(ix, iy)
    c10 = _hash_2d(ix + 1, iy)
    c01 = _hash_2d(ix, iy + 1)
    c11 = _hash_2d(ix + 1, iy + 1)
    a = c00 + (c10 - c00) * fx
    b = c01 + (c11 - c01) * fx
    return a + (b - a) * fy


def shade_terrain_colors(
    heights_m: np.ndarray,
    normals: np.ndarray,
    *,
    positions_xy: Optional[np.ndarray] = None,
    landcover: Optional[np.ndarray] = None,
) -> np.ndarray:
    light = normalize(np.array([0.45, 0.25, 0.86], dtype=np.float32))
    fill_light = normalize(np.array([-0.3, 0.4, 0.55], dtype=np.float32))
    heights = heights_m.reshape((-1,))
    if heights.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    low = np.array([0.45, 0.65, 0.32], dtype=np.float32)
    mid = np.array([0.72, 0.68, 0.45], dtype=np.float32)
    high = np.array([0.55, 0.50, 0.44], dtype=np.float32)
    rock = np.array([0.52, 0.48, 0.42], dtype=np.float32)

    min_height = float(np.min(heights))
    max_height = float(np.max(heights))
    span = max(max_height - min_height, 1.0)
    colors = np.zeros((heights.size, 4), dtype=np.float32)
    for index, height in enumerate(heights):
        t = (float(height) - min_height) / span
        # Three-stop gradient: low → mid → high.
        if t < 0.5:
            s = t * 2.0
            color = low + (mid - low) * s
        else:
            s = (t - 0.5) * 2.0
            color = mid + (high - mid) * s

        # Noise-based perturbation from vertex position.
        if positions_xy is not None:
            px = float(positions_xy[index, 0])
            py = float(positions_xy[index, 1])
            n1 = _value_noise_2d(px / 41.7, py / 41.7) * 0.08
            n2 = _value_noise_2d(px / 12.3, py / 12.3) * 0.04
            n3 = _value_noise_2d(px / 24.0 + 7.3, py / 24.0 + 3.1) * 0.05
            color = color + (n1 + n2 + n3)

        # Slope-based rock coloring from normal z-component.
        slope = 1.0 - float(normals[index][2])
        slope_blend = min(slope * 3.0, 1.0) * 0.55
        color = color * (1.0 - slope_blend) + rock * slope_blend

        contour_interval = max(span / 6.0, 2.0)
        contour_dist = abs((float(height) - min_height) % contour_interval - contour_interval * 0.5)
        contour_factor = 1.0 - 0.06 * max(0.0, 1.0 - contour_dist / (contour_interval * 0.08))
        color = color * contour_factor

        if landcover is not None:
            cover = int(landcover.reshape((-1,))[index])
            if cover == int(LandCoverClass.WATER):
                color = np.array([0.78, 0.86, 0.92], dtype=np.float32)
            elif cover == int(LandCoverClass.URBAN):
                color = np.array([0.74, 0.78, 0.70], dtype=np.float32)
            elif cover == int(LandCoverClass.FOREST):
                color = np.array([0.62, 0.72, 0.52], dtype=np.float32)
        main_shade = max(float(np.dot(normals[index], light)), 0.0)
        fill_shade = max(float(np.dot(normals[index], fill_light)), 0.0) * 0.15
        shade = 0.65 + (0.25 * main_shade) + fill_shade
        colors[index, :3] = np.clip(color * shade, 0.0, 1.0)
        colors[index, 3] = 1.0
    return colors


def mesh_from_height_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights_m: np.ndarray,
    *,
    color_override: Optional[Sequence[float]] = None,
    landcover: Optional[np.ndarray] = None,
) -> SimpleMesh:
    rows, cols = heights_m.shape
    xx, yy = np.meshgrid(x_values, y_values)
    positions = np.stack([xx, yy, heights_m], axis=-1).reshape((-1, 3)).astype(np.float32)
    normals = grid_normals(heights_m, x_values, y_values)
    colors = (
        solid_color_array(positions.shape[0], color_override)
        if color_override is not None
        else shade_terrain_colors(heights_m, normals, positions_xy=positions[:, :2], landcover=landcover)
    )

    indices = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = (row + 1) * cols + col
            d = c + 1
            indices.extend([a, c, b, b, c, d])
    return SimpleMesh(
        positions=positions,
        normals=normals,
        colors=colors,
        indices=np.asarray(indices, dtype=np.uint32),
    )


def polygon_area(points_xy: np.ndarray) -> float:
    return 0.5 * float(np.sum(points_xy[:, 0] * np.roll(points_xy[:, 1], -1) - points_xy[:, 1] * np.roll(points_xy[:, 0], -1)))


def point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = (dot00 * dot11) - (dot01 * dot01)
    if abs(denom) <= 1.0e-12:
        return False
    inv = 1.0 / denom
    u = ((dot11 * dot02) - (dot01 * dot12)) * inv
    v = ((dot00 * dot12) - (dot01 * dot02)) * inv
    return u >= -1.0e-6 and v >= -1.0e-6 and (u + v) <= 1.0 + 1.0e-6


def cross_z(a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] * b[1]) - (a[1] * b[0]))


def triangulate_polygon(points_xy: np.ndarray) -> np.ndarray:
    polygon = np.asarray(points_xy, dtype=np.float32)
    if polygon.shape[0] < 3:
        return np.zeros((0,), dtype=np.uint32)
    vertices = list(range(polygon.shape[0]))
    clockwise = polygon_area(polygon) < 0.0
    triangles: List[int] = []
    guard = 0
    while len(vertices) > 2 and guard < polygon.shape[0] * polygon.shape[0]:
        ear_found = False
        for index in range(len(vertices)):
            prev_index = vertices[(index - 1) % len(vertices)]
            curr_index = vertices[index]
            next_index = vertices[(index + 1) % len(vertices)]
            a = polygon[prev_index]
            b = polygon[curr_index]
            c = polygon[next_index]
            cross = cross_z(b - a, c - b)
            if clockwise:
                if cross >= -1.0e-9:
                    continue
            elif cross <= 1.0e-9:
                continue
            if any(
                point_in_triangle(polygon[other], a, b, c)
                for other in vertices
                if other not in {prev_index, curr_index, next_index}
            ):
                continue
            triangles.extend([prev_index, curr_index, next_index] if not clockwise else [prev_index, next_index, curr_index])
            del vertices[index]
            ear_found = True
            break
        if not ear_found:
            break
        guard += 1
    return np.asarray(triangles, dtype=np.uint32)


def mesh_from_polygon(
    points_xy: np.ndarray,
    *,
    height_at: Callable[[float, float], float],
    color_rgba: Sequence[float],
    z_offset_m: float = 0.35,
) -> SimpleMesh:
    if len(points_xy) < 3:
        return merge_meshes(())
    positions = np.array(
        [
            [float(point[0]), float(point[1]), float(height_at(float(point[0]), float(point[1]))) + z_offset_m]
            for point in points_xy
        ],
        dtype=np.float32,
    )
    indices = triangulate_polygon(points_xy)
    if indices.size == 0:
        return merge_meshes(())
    return SimpleMesh(
        positions=positions,
        normals=triangle_normals(positions, indices),
        colors=solid_color_array(positions.shape[0], color_rgba),
        indices=indices,
    )


def mesh_from_polyline(
    points_xy: np.ndarray,
    *,
    height_at: Callable[[float, float], float],
    color_rgba: Sequence[float],
    width_m: float,
    z_offset_m: float = 0.5,
) -> SimpleMesh:
    if len(points_xy) < 2:
        return merge_meshes(())
    positions: List[List[float]] = []
    indices: List[int] = []
    for segment_index in range(len(points_xy) - 1):
        start = np.asarray(points_xy[segment_index], dtype=np.float32)
        end = np.asarray(points_xy[segment_index + 1], dtype=np.float32)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-6:
            continue
        normal = np.array([-direction[1], direction[0]], dtype=np.float32) / length
        offset = normal * (float(width_m) * 0.5)
        quad_xy = [start - offset, start + offset, end + offset, end - offset]
        base = len(positions)
        for point in quad_xy:
            positions.append([
                float(point[0]),
                float(point[1]),
                float(height_at(float(point[0]), float(point[1]))) + z_offset_m,
            ])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    if not indices:
        return merge_meshes(())
    position_array = np.asarray(positions, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    return SimpleMesh(
        positions=position_array,
        normals=triangle_normals(position_array, index_array),
        colors=solid_color_array(position_array.shape[0], color_rgba),
        indices=index_array,
    )


def mesh_from_extrusion(
    points_xy: np.ndarray,
    *,
    base_z_m: float,
    top_z_m: float,
    color_rgba: Sequence[float],
) -> SimpleMesh:
    if len(points_xy) < 3 or top_z_m <= base_z_m:
        return merge_meshes(())
    footprint = np.asarray(points_xy, dtype=np.float32)
    roof_positions = np.column_stack([footprint, np.full((len(footprint),), top_z_m, dtype=np.float32)])
    roof_indices = triangulate_polygon(footprint)
    roof_mesh = merge_meshes(())
    if roof_indices.size:
        roof_mesh = SimpleMesh(
            positions=roof_positions,
            normals=np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), len(roof_positions), axis=0),
            colors=solid_color_array(len(roof_positions), color_rgba),
            indices=roof_indices,
        )

    wall_positions: List[List[float]] = []
    wall_indices: List[int] = []
    for index in range(len(footprint)):
        next_index = (index + 1) % len(footprint)
        start = footprint[index]
        end = footprint[next_index]
        base = len(wall_positions)
        wall_positions.extend(
            [
                [float(start[0]), float(start[1]), float(base_z_m)],
                [float(end[0]), float(end[1]), float(base_z_m)],
                [float(end[0]), float(end[1]), float(top_z_m)],
                [float(start[0]), float(start[1]), float(top_z_m)],
            ]
        )
        wall_indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    if not wall_indices:
        return roof_mesh
    wall_positions_array = np.asarray(wall_positions, dtype=np.float32)
    wall_indices_array = np.asarray(wall_indices, dtype=np.uint32)
    wall_mesh = SimpleMesh(
        positions=wall_positions_array,
        normals=triangle_normals(wall_positions_array, wall_indices_array),
        colors=solid_color_array(wall_positions_array.shape[0], color_rgba),
        indices=wall_indices_array,
    )
    return merge_meshes([roof_mesh, wall_mesh])


def height_function_from_mesh(mesh: Mapping[str, object]) -> Callable[[float, float], float]:
    rows = int(mesh["rows"])
    cols = int(mesh["cols"])
    heights = np.asarray(mesh["heights_m"], dtype=np.float32)
    x_values = np.linspace(float(mesh["x_min_m"]), float(mesh["x_max_m"]), num=cols, dtype=np.float32)
    y_values = np.linspace(float(mesh["y_min_m"]), float(mesh["y_max_m"]), num=rows, dtype=np.float32)

    def height_at(x_m: float, y_m: float) -> float:
        tx = np.clip(np.interp(float(x_m), x_values, np.arange(cols, dtype=np.float32)), 0.0, cols - 1)
        ty = np.clip(np.interp(float(y_m), y_values, np.arange(rows, dtype=np.float32)), 0.0, rows - 1)
        col = min(int(math.floor(float(tx))), cols - 2)
        row = min(int(math.floor(float(ty))), rows - 2)
        ax = float(tx) - col
        ay = float(ty) - row
        z00 = float(heights[row, col])
        z10 = float(heights[row, col + 1])
        z01 = float(heights[row + 1, col])
        z11 = float(heights[row + 1, col + 1])
        z0 = z00 + ((z10 - z00) * ax)
        z1 = z01 + ((z11 - z01) * ax)
        return z0 + ((z1 - z0) * ay)

    return height_at


def terrain_mesh_from_viewer_mesh(mesh: Mapping[str, object], *, landcover: Optional[np.ndarray] = None) -> SimpleMesh:
    x_values = np.linspace(float(mesh["x_min_m"]), float(mesh["x_max_m"]), num=int(mesh["cols"]), dtype=np.float32)
    y_values = np.linspace(float(mesh["y_min_m"]), float(mesh["y_max_m"]), num=int(mesh["rows"]), dtype=np.float32)
    heights = np.asarray(mesh["heights_m"], dtype=np.float32)
    return mesh_from_height_grid(x_values, y_values, heights, landcover=landcover)


def chunked_height_grid_layers(
    *,
    layer_id: str,
    style_id: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights_m: np.ndarray,
    max_chunk_vertices: int = 65,
) -> List[SceneLayerMesh]:
    rows, cols = heights_m.shape
    if rows < 2 or cols < 2:
        return []

    if rows <= max_chunk_vertices and cols <= max_chunk_vertices:
        return [
            SceneLayerMesh(
                layer_id=layer_id,
                semantic_kind="terrain",
                mesh=mesh_from_height_grid(x_values, y_values, heights_m),
                style_id=style_id,
            )
        ]

    layers: List[SceneLayerMesh] = []
    step = max(max_chunk_vertices - 1, 1)
    chunk_row = 0
    for row_start in range(0, rows - 1, step):
        row_end = min(row_start + max_chunk_vertices, rows)
        if row_end - row_start < 2:
            continue
        chunk_col = 0
        for col_start in range(0, cols - 1, step):
            col_end = min(col_start + max_chunk_vertices, cols)
            if col_end - col_start < 2:
                continue
            layers.append(
                SceneLayerMesh(
                    layer_id=f"{layer_id}-r{chunk_row:02d}-c{chunk_col:02d}",
                    semantic_kind="terrain",
                    mesh=mesh_from_height_grid(
                        x_values[col_start:col_end],
                        y_values[row_start:row_end],
                        heights_m[row_start:row_end, col_start:col_end],
                    ),
                    style_id=style_id,
                )
            )
            chunk_col += 1
        chunk_row += 1
    return layers


def write_scene_layers(
    output_dir: Path,
    layers: Sequence[SceneLayerMesh],
) -> List[Dict[str, object]]:
    manifest_layers: List[Dict[str, object]] = []
    for layer in layers:
        layer_id = layer.layer_id
        semantic_kind = layer.semantic_kind
        mesh = layer.mesh
        if mesh.vertex_count == 0 or mesh.indices.size == 0:
            continue
        relative_path = (Path("terrain") if semantic_kind == "terrain" else Path("overlays")) / f"{layer_id}.glb"
        write_glb(output_dir / relative_path, mesh, name=layer_id)
        manifest_layers.append(
            {
                "id": layer_id,
                "kind": "terrain" if semantic_kind == "terrain" else "overlay",
                "semantic_kind": semantic_kind,
                "asset_path": str(relative_path),
                "style_id": layer.style_id or layer_id,
            }
        )
    return manifest_layers


__all__ = [
    "SceneLayerMesh",
    "chunked_height_grid_layers",
    "height_function_from_mesh",
    "mesh_from_extrusion",
    "mesh_from_height_grid",
    "mesh_from_polygon",
    "mesh_from_polyline",
    "terrain_mesh_from_viewer_mesh",
    "write_scene_layers",
]
