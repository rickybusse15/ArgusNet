from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._glb import SimpleMesh, merge_meshes, solid_color_array, triangle_normals, write_glb
from .environment import LandCoverClass


@dataclass(frozen=True)
class SceneLayerMesh:
    layer_id: str
    semantic_kind: str
    mesh: SimpleMesh
    style_id: str | None = None
    # One mesh per object (building, wall, stand), written as separate glTF
    # nodes so the viewer depth-sorts them; empty writes `mesh` whole.
    parts: tuple[SimpleMesh, ...] = ()


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
            dz_dx = (
                heights_m[row, min(col + 1, cols - 1)] - heights_m[row, max(col - 1, 0)]
            ) / max(x1 - x0, 1.0e-6)
            dz_dy = (
                heights_m[min(row + 1, rows - 1), col] - heights_m[max(row - 1, 0), col]
            ) / max(y1 - y0, 1.0e-6)
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
    positions_xy: np.ndarray | None = None,
    landcover: np.ndarray | None = None,
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
            elif cover == int(LandCoverClass.SCRUB):
                color = np.array([0.68, 0.68, 0.42], dtype=np.float32)
            elif cover == int(LandCoverClass.WETLAND):
                color = np.array([0.52, 0.67, 0.58], dtype=np.float32)
            elif cover == int(LandCoverClass.ROCKY):
                color = np.array([0.58, 0.55, 0.50], dtype=np.float32)
            elif cover == int(LandCoverClass.SNOW):
                color = np.array([0.90, 0.94, 0.96], dtype=np.float32)
            elif cover == int(LandCoverClass.ROAD):
                color = np.array([0.64, 0.62, 0.57], dtype=np.float32)
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
    color_override: Sequence[float] | None = None,
    landcover: np.ndarray | None = None,
) -> SimpleMesh:
    rows, cols = heights_m.shape
    xx, yy = np.meshgrid(x_values, y_values)
    positions = np.stack([xx, yy, heights_m], axis=-1).reshape((-1, 3)).astype(np.float32)
    normals = grid_normals(heights_m, x_values, y_values)
    colors = (
        solid_color_array(positions.shape[0], color_override)
        if color_override is not None
        else shade_terrain_colors(
            heights_m, normals, positions_xy=positions[:, :2], landcover=landcover
        )
    )

    indices = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = (row + 1) * cols + col
            d = c + 1
            # Counter-clockwise seen from above: each triangle faces up, the
            # way its normals point.
            indices.extend([a, b, c, b, d, c])
    return SimpleMesh(
        positions=positions,
        normals=normals,
        colors=colors,
        indices=np.asarray(indices, dtype=np.uint32),
    )


def polygon_area(points_xy: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points_xy[:, 0] * np.roll(points_xy[:, 1], -1)
            - points_xy[:, 1] * np.roll(points_xy[:, 0], -1)
        )
    )


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
    triangles: list[int] = []
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
            triangles.extend(
                [prev_index, curr_index, next_index]
                if not clockwise
                else [prev_index, next_index, curr_index]
            )
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
            [
                float(point[0]),
                float(point[1]),
                float(height_at(float(point[0]), float(point[1]))) + z_offset_m,
            ]
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


_Polygon = list[tuple[float, float]]


def _polygon_area(polygon: _Polygon) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )


def _split_by_line(
    polygon: _Polygon, a: tuple[float, float], b: tuple[float, float]
) -> tuple[_Polygon, _Polygon]:
    """Split a convex polygon by the line ``a -> b``: (right of it, left of it)."""

    def side(point: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])

    left: _Polygon = []
    right: _Polygon = []
    for current, following in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        s0, s1 = side(current), side(following)
        # A vertex on the line bounds both sides.
        if s0 >= 0.0:
            left.append(current)
        if s0 <= 0.0:
            right.append(current)
        if (s0 > 0.0 and s1 < 0.0) or (s0 < 0.0 and s1 > 0.0):
            t = s0 / (s0 - s1)
            crossing = (
                current[0] + t * (following[0] - current[0]),
                current[1] + t * (following[1] - current[1]),
            )
            left.append(crossing)
            right.append(crossing)
    return right, left


def _overlaps(first: _Polygon, second: _Polygon, tolerance_m: float = 1.0e-7) -> bool:
    """Whether two convex polygons share interior (separating-axis test)."""
    for polygon in (first, second):
        for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            axis = (a[1] - b[1], b[0] - a[0])
            length = math.hypot(*axis)
            if length <= 1.0e-12:
                continue
            p1 = [(x * axis[0] + y * axis[1]) / length for x, y in first]
            p2 = [(x * axis[0] + y * axis[1]) / length for x, y in second]
            if max(p1) <= min(p2) + tolerance_m or max(p2) <= min(p1) + tolerance_m:
                return False
    return True


def _convex_minus(polygon: _Polygon, cutter: _Polygon) -> list[_Polygon]:
    """``polygon`` minus ``cutter`` (both convex, counter-clockwise), as convex pieces."""
    pieces: list[_Polygon] = []
    rest = polygon
    for index, a in enumerate(cutter):
        outside, inside = _split_by_line(rest, a, cutter[(index + 1) % len(cutter)])
        if len(outside) >= 3 and _polygon_area(outside) > 1.0e-9:
            pieces.append(outside)
        if len(inside) < 3 or _polygon_area(inside) <= 1.0e-9:
            return pieces
        rest = inside
    return pieces


def mesh_from_polyline(
    points_xy: np.ndarray,
    *,
    height_at: Callable[[float, float], float],
    color_rgba: Sequence[float],
    width_m: float,
    z_offset_m: float = 0.5,
) -> SimpleMesh:
    """A line as a flat stroke ``width_m`` wide, facing up, covering each point once.

    A translucent layer draws any overlap twice, and segment quads overlap at
    every bend -- and wholesale where a line retraces or crosses itself. So the
    stroke is built as the exact union of convex pieces: a quad per segment
    and a bevel wedge on the outside of each bend, each piece minus every
    earlier one (convex minus convex is a few convex pieces), fanned into
    counter-clockwise triangles. Every point is kept, so the stroke samples
    the terrain at each one.
    """
    if len(points_xy) < 2:
        return merge_meshes(())
    half_width = float(width_m) * 0.5
    points = [np.asarray(points_xy[0], dtype=np.float64)]
    for point in points_xy[1:]:
        point = np.asarray(point, dtype=np.float64)
        if float(np.linalg.norm(point - points[-1])) > 1.0e-6:
            points.append(point)
    if len(points) < 2:
        return merge_meshes(())

    def xy(point: np.ndarray) -> tuple[float, float]:
        return (float(point[0]), float(point[1]))

    lefts = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        direction = (end - start) / float(np.linalg.norm(end - start))
        lefts.append(np.array([-direction[1], direction[0]]))
    pieces: list[_Polygon] = []
    for index, left in enumerate(lefts):
        start, end = points[index], points[index + 1]
        offset = left * half_width
        pieces.append([xy(start - offset), xy(end - offset), xy(end + offset), xy(start + offset)])
        if index + 1 < len(lefts):
            following = lefts[index + 1]
            # The outside of the bend: right of a left turn, left of a right one.
            turn = following[0] * left[1] - following[1] * left[0]
            if abs(turn) > 1.0e-9:
                side = 1.0 if turn > 0.0 else -1.0
                wedge = [
                    xy(end),
                    xy(end + side * left * half_width),
                    xy(end + side * following * half_width),
                ]
                if _polygon_area(wedge) < 0.0:
                    wedge.reverse()
                pieces.append(wedge)
    boxes = [
        (min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p))
        for p in pieces
    ]
    # The union so far is kept as disjoint convex fragments, indexed in a
    # uniform grid: each new piece is cut only by fragments near it, and a
    # stretch the line retraces adds no fragments, so buckets stay bounded by
    # the stroke's area rather than its length. Cells the size of a typical
    # piece keep a long diagonal segment from spanning a square of tiny cells;
    # the rare fragment spanning many cells sits on a short list instead.
    extents = sorted(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes)
    cell_m = max(extents[len(extents) // 2], float(width_m), 1.0e-3)
    grid: dict[tuple[int, int], list[int]] = {}
    large: list[int] = []

    def bounds(polygon: _Polygon) -> tuple[float, float, float, float]:
        xs, ys = [x for x, _ in polygon], [y for _, y in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def cells(box: tuple[float, float, float, float]) -> list[tuple[int, int]] | None:
        x0, y0, x1, y1 = (math.floor(v / cell_m) for v in box)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 64:
            return None
        return [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1)]

    union: list[_Polygon] = []
    union_boxes: list[tuple[float, float, float, float]] = []
    for index, piece in enumerate(pieces):
        fragments = [piece]
        x0, y0, x1, y1 = boxes[index]
        piece_cells = cells(boxes[index])
        if piece_cells is None:
            nearby: list[int] = list(range(len(union)))
        else:
            nearby = sorted(
                {other for cell in piece_cells for other in grid.get(cell, ())}.union(large)
            )
        for other in nearby:
            ex0, ey0, ex1, ey1 = union_boxes[other]
            # Most pieces only touch their neighbours; cutting by one that
            # merely touches would only shatter this piece into slivers.
            if ex0 >= x1 or ex1 <= x0 or ey0 >= y1 or ey1 <= y0:
                continue
            cutter = union[other]
            fragments = [
                part
                for fragment in fragments
                for part in (
                    _convex_minus(fragment, cutter) if _overlaps(fragment, cutter) else [fragment]
                )
            ]
            if not fragments:
                break  # already covered
        for fragment in fragments:
            fragment_box = bounds(fragment)
            fragment_cells = cells(fragment_box)
            if fragment_cells is None:
                large.append(len(union))
            else:
                for cell in fragment_cells:
                    grid.setdefault(cell, []).append(len(union))
            union.append(fragment)
            union_boxes.append(fragment_box)

    positions: list[list[float]] = []
    indices: list[int] = []
    for fragment in union:
        # Splits can leave vertices a hair apart; float32 would merge them.
        polygon: _Polygon = []
        for point in fragment:
            if not polygon or math.dist(point, polygon[-1]) > 1.0e-4:
                polygon.append(point)
        if len(polygon) > 1 and math.dist(polygon[0], polygon[-1]) <= 1.0e-4:
            polygon.pop()
        if len(polygon) < 3:
            continue
        base = len(positions)
        for x, y in polygon:
            positions.append([x, y, float(height_at(x, y)) + z_offset_m])
        for corner in range(1, len(polygon) - 1):
            # Collinear points from a split leave slivers that round to zero
            # area (or flip) in float32; skip any under a square millimetre.
            triangle = [polygon[0], polygon[corner], polygon[corner + 1]]
            if _polygon_area(triangle) > 1.0e-6:
                indices.extend([base, base + corner, base + corner + 1])
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
    roof_positions = np.column_stack(
        [footprint, np.full((len(footprint),), top_z_m, dtype=np.float32)]
    )
    roof_indices = triangulate_polygon(footprint)
    roof_mesh = merge_meshes(())
    if roof_indices.size:
        roof_mesh = SimpleMesh(
            positions=roof_positions,
            normals=np.repeat(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32), len(roof_positions), axis=0
            ),
            colors=solid_color_array(len(roof_positions), color_rgba),
            indices=roof_indices,
        )

    # Outward for a counter-clockwise footprint, so a clockwise one is walked
    # backwards: outward walls let a translucent layer cull back faces.
    ccw = footprint[::-1] if polygon_area(footprint) < 0.0 else footprint
    wall_positions: list[list[float]] = []
    wall_indices: list[int] = []
    for index in range(len(ccw)):
        next_index = (index + 1) % len(ccw)
        start = ccw[index]
        end = ccw[next_index]
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
    x_values = np.linspace(
        float(mesh["x_min_m"]), float(mesh["x_max_m"]), num=cols, dtype=np.float32
    )
    y_values = np.linspace(
        float(mesh["y_min_m"]), float(mesh["y_max_m"]), num=rows, dtype=np.float32
    )

    def height_at(x_m: float, y_m: float) -> float:
        tx = np.clip(
            np.interp(float(x_m), x_values, np.arange(cols, dtype=np.float32)), 0.0, cols - 1
        )
        ty = np.clip(
            np.interp(float(y_m), y_values, np.arange(rows, dtype=np.float32)), 0.0, rows - 1
        )
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


def terrain_mesh_from_viewer_mesh(
    mesh: Mapping[str, object], *, landcover: np.ndarray | None = None
) -> SimpleMesh:
    x_values = np.linspace(
        float(mesh["x_min_m"]), float(mesh["x_max_m"]), num=int(mesh["cols"]), dtype=np.float32
    )
    y_values = np.linspace(
        float(mesh["y_min_m"]), float(mesh["y_max_m"]), num=int(mesh["rows"]), dtype=np.float32
    )
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
) -> list[SceneLayerMesh]:
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

    layers: list[SceneLayerMesh] = []
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
) -> list[dict[str, object]]:
    manifest_layers: list[dict[str, object]] = []
    for layer in layers:
        layer_id = layer.layer_id
        semantic_kind = layer.semantic_kind
        mesh = layer.mesh
        if mesh.vertex_count == 0 or mesh.indices.size == 0:
            continue
        relative_path = (
            Path("terrain") if semantic_kind == "terrain" else Path("overlays")
        ) / f"{layer_id}.glb"
        parts = [part for part in layer.parts if part.vertex_count and part.indices.size]
        write_glb(output_dir / relative_path, parts or mesh, name=layer_id)
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
