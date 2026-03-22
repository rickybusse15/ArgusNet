from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from argusnet.world.environment import Bounds2D, ObstacleLayer
from argusnet.world.obstacles import (
    BuildingPrism,
    CylinderObstacle,
    ObstaclePrimitive,
    OrientedBox,
    PolygonPrism,
    WallSegment,
    _edge_outward_normal,
    _nearest_point_on_segment,
    _point_in_polygon,
    _point_on_segment,
    _segment_polygon_intervals,
    _signed_polygon_area,
)

HARD_BLOCKER_TYPES = frozenset({"building", "wall"})


@dataclass(frozen=True)
class PlannerConfig:
    drone_clearance_m: float = 8.0
    target_clearance_m: float = 4.0
    snap_m: float = 10.0
    cylinder_vertex_count: int = 12
    turn_penalty_m: float = 3.0
    max_cache_entries: int = 512


@dataclass(frozen=True)
class PlannerRoute:
    points_xy_m: np.ndarray
    length_m: float
    vertex_count: int
    cache_hit: bool = False
    skipped_segments: int = 0

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("PlannerRoute.points_xy_m must have shape (n, 2) with n >= 2.")
        object.__setattr__(self, "points_xy_m", points)


def _segment_bounds(start_xy: np.ndarray, end_xy: np.ndarray) -> Bounds2D:
    x_min_m = min(float(start_xy[0]), float(end_xy[0]))
    x_max_m = max(float(start_xy[0]), float(end_xy[0]))
    y_min_m = min(float(start_xy[1]), float(end_xy[1]))
    y_max_m = max(float(start_xy[1]), float(end_xy[1]))
    epsilon_m = 1.0e-6
    return Bounds2D(
        x_min_m=x_min_m - epsilon_m,
        x_max_m=x_max_m + epsilon_m,
        y_min_m=y_min_m - epsilon_m,
        y_max_m=y_max_m + epsilon_m,
    )


def _path_length(points_xy_m: np.ndarray) -> float:
    if len(points_xy_m) < 2:
        return 0.0
    deltas = points_xy_m[1:] - points_xy_m[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _expand_polygon(polygon_xy_m: np.ndarray, clearance_m: float) -> np.ndarray:
    clearance = max(float(clearance_m), 0.0)
    polygon = np.asarray(polygon_xy_m, dtype=float)
    if clearance <= 1.0e-9:
        return polygon.copy()
    is_ccw = _signed_polygon_area(polygon) >= 0.0
    expanded = np.empty_like(polygon)
    for index in range(len(polygon)):
        previous_xy = polygon[index - 1]
        current_xy = polygon[index]
        next_xy = polygon[(index + 1) % len(polygon)]
        incoming_normal = _edge_outward_normal(previous_xy, current_xy, is_ccw=is_ccw)
        outgoing_normal = _edge_outward_normal(current_xy, next_xy, is_ccw=is_ccw)
        offset_direction = incoming_normal + outgoing_normal
        norm = float(np.linalg.norm(offset_direction))
        if norm <= 1.0e-9:
            offset_direction = incoming_normal
        else:
            offset_direction = offset_direction / norm
        scale = max(float(np.dot(offset_direction, incoming_normal)), 0.2)
        expanded[index] = current_xy + offset_direction * (clearance / scale)
    return expanded


def _push_outside_polygon(point_xy: np.ndarray, polygon_xy_m: np.ndarray, margin_m: float) -> np.ndarray:
    if not _point_in_polygon(point_xy, polygon_xy_m):
        return point_xy.copy()

    is_ccw = _signed_polygon_area(polygon_xy_m) >= 0.0
    best_candidate = point_xy.copy()
    best_distance = float("inf")
    for index in range(len(polygon_xy_m)):
        edge_start = polygon_xy_m[index]
        edge_end = polygon_xy_m[(index + 1) % len(polygon_xy_m)]
        boundary_point = _nearest_point_on_segment(point_xy, edge_start, edge_end)
        outward_normal = _edge_outward_normal(edge_start, edge_end, is_ccw=is_ccw)
        candidate = boundary_point + outward_normal * max(margin_m, 0.5)
        for _ in range(8):
            if not _point_in_polygon(candidate, polygon_xy_m):
                break
            candidate = candidate + outward_normal * max(margin_m, 0.5)
        distance = float(np.linalg.norm(candidate - point_xy))
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate
    return best_candidate


def _point_on_polygon_boundary(point_xy: np.ndarray, polygon_xy_m: np.ndarray) -> bool:
    for index in range(len(polygon_xy_m)):
        edge_start = polygon_xy_m[index]
        edge_end = polygon_xy_m[(index + 1) % len(polygon_xy_m)]
        if _point_on_segment(point_xy, edge_start, edge_end):
            return True
    return False


class PathPlanner2D:
    def __init__(
        self,
        *,
        bounds_xy_m: Bounds2D,
        obstacle_layer: ObstacleLayer,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.bounds_xy_m = bounds_xy_m
        self.obstacle_layer = obstacle_layer
        self.config = config or PlannerConfig()
        self._hard_primitives = tuple(
            primitive for primitive in obstacle_layer.primitives if primitive.blocker_type in HARD_BLOCKER_TYPES
        )
        self._polygon_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._route_cache: "OrderedDict[Tuple[int, int, int, int, float], PlannerRoute]" = OrderedDict()

    def clear_cache(self) -> None:
        self._route_cache.clear()

    def expanded_polygon_for_primitive(self, primitive: ObstaclePrimitive, clearance_m: float) -> np.ndarray:
        cache_key = (primitive.primitive_id, round(float(clearance_m), 3))
        cached = self._polygon_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if isinstance(primitive, CylinderObstacle):
            radius_m = primitive.radius_m + max(float(clearance_m), 0.0)
            angles = np.linspace(0.0, math.tau, num=max(self.config.cylinder_vertex_count, 6), endpoint=False)
            polygon = np.column_stack(
                [
                    primitive.center_x_m + radius_m * np.cos(angles),
                    primitive.center_y_m + radius_m * np.sin(angles),
                ]
            )
        elif isinstance(primitive, (OrientedBox, WallSegment)):
            polygon = _expand_polygon(primitive.footprint_xy_m(), clearance_m)
        elif isinstance(primitive, (PolygonPrism, BuildingPrism)):
            polygon = _expand_polygon(primitive.footprint_xy_m, clearance_m)
        else:
            raise TypeError(f"Unsupported obstacle primitive {type(primitive)!r}.")

        self._polygon_cache[cache_key] = polygon.copy()
        return polygon

    def nearest_free_point(self, point_xy: Sequence[float], clearance_m: float) -> np.ndarray:
        margin_m = max(float(clearance_m), 0.0)
        candidate = np.array(
            [
                np.clip(float(point_xy[0]), self.bounds_xy_m.x_min_m + margin_m, self.bounds_xy_m.x_max_m - margin_m),
                np.clip(float(point_xy[1]), self.bounds_xy_m.y_min_m + margin_m, self.bounds_xy_m.y_max_m - margin_m),
            ],
            dtype=float,
        )
        for _ in range(10):
            blocker = self._blocking_primitive(candidate, clearance_m)
            if blocker is None:
                return candidate
            polygon = self.expanded_polygon_for_primitive(blocker, clearance_m)
            candidate = _push_outside_polygon(candidate, polygon, margin_m=max(margin_m, 0.5))
            candidate[0] = np.clip(candidate[0], self.bounds_xy_m.x_min_m + margin_m, self.bounds_xy_m.x_max_m - margin_m)
            candidate[1] = np.clip(candidate[1], self.bounds_xy_m.y_min_m + margin_m, self.bounds_xy_m.y_max_m - margin_m)
        return candidate

    def plan_route(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        clearance_m: float,
    ) -> Optional[PlannerRoute]:
        start = np.asarray(start_xy, dtype=float).reshape(2)
        goal = np.asarray(goal_xy, dtype=float).reshape(2)
        if not self._point_within_bounds(start, clearance_m) or not self._point_within_bounds(goal, clearance_m):
            return None
        if self._blocking_primitive(start, clearance_m) is not None:
            return None
        if self._blocking_primitive(goal, clearance_m) is not None:
            return None

        cache_key = self._route_cache_key(start, goal, clearance_m)
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            self._route_cache.move_to_end(cache_key)
            return replace(cached, cache_hit=True)

        if self._segment_is_free(start, goal, clearance_m):
            route = PlannerRoute(
                points_xy_m=np.vstack([start, goal]),
                length_m=float(np.linalg.norm(goal - start)),
                vertex_count=2,
            )
            self._remember_route(cache_key, route)
            return route

        primitives = self._candidate_primitives(start, goal)
        polygons = [(primitive.primitive_id, self.expanded_polygon_for_primitive(primitive, clearance_m)) for primitive in primitives]
        nodes: List[np.ndarray] = [start, goal]
        polygon_node_indices: List[List[int]] = []
        for _, polygon in polygons:
            indices: List[int] = []
            for vertex in polygon:
                nodes.append(np.asarray(vertex, dtype=float))
                indices.append(len(nodes) - 1)
            polygon_node_indices.append(indices)
        adjacency = self._build_visibility_graph(nodes, polygon_node_indices, clearance_m)
        path_points = self._astar_path(nodes, adjacency)
        if path_points is None:
            return None
        smoothed = self._smooth_path(path_points, clearance_m)
        route = PlannerRoute(
            points_xy_m=smoothed,
            length_m=_path_length(smoothed),
            vertex_count=len(smoothed),
        )
        self._remember_route(cache_key, route)
        return route

    def route_waypoints(
        self,
        anchor_points_xy: Sequence[Sequence[float]],
        *,
        clearance_m: float,
        closed: bool = True,
    ) -> Optional[PlannerRoute]:
        anchors = np.asarray(anchor_points_xy, dtype=float)
        if anchors.ndim != 2 or anchors.shape[0] < 2 or anchors.shape[1] != 2:
            raise ValueError("anchor_points_xy must have shape (n, 2) with n >= 2.")

        adjusted = np.vstack([self.nearest_free_point(point_xy, clearance_m) for point_xy in anchors])
        segments: List[np.ndarray] = []
        skipped_segments = 0
        segment_count = len(adjusted) if closed else len(adjusted) - 1
        for index in range(segment_count):
            start = adjusted[index]
            end = adjusted[(index + 1) % len(adjusted)]
            route = self.plan_route(start, end, clearance_m=clearance_m)
            if route is None:
                skipped_segments += 1
                continue
            if not segments:
                segments.append(route.points_xy_m)
            else:
                segments.append(route.points_xy_m[1:])

        if not segments:
            return None
        if skipped_segments > max(1, segment_count // 2):
            return None

        points = np.vstack(segments)
        if closed and len(points) >= 2 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 2:
            return None
        smoothed = self._smooth_path(points, clearance_m)
        if len(smoothed) < 2 or _path_length(smoothed) <= 1.0e-6:
            return None
        return PlannerRoute(
            points_xy_m=smoothed,
            length_m=_path_length(smoothed),
            vertex_count=len(smoothed),
            skipped_segments=skipped_segments,
        )

    def _candidate_primitives(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Tuple[ObstaclePrimitive, ...]:
        segment_bounds = _segment_bounds(start_xy, goal_xy).padded(max(self.config.drone_clearance_m * 4.0, 80.0))
        candidates = self.obstacle_layer.query_obstacles(segment_bounds)
        hard_candidates = tuple(
            primitive for primitive in candidates if primitive.blocker_type in HARD_BLOCKER_TYPES
        )
        return hard_candidates if hard_candidates else self._hard_primitives

    def _remember_route(self, cache_key: Tuple[int, int, int, int, float], route: PlannerRoute) -> None:
        self._route_cache[cache_key] = route
        self._route_cache.move_to_end(cache_key)
        while len(self._route_cache) > self.config.max_cache_entries:
            self._route_cache.popitem(last=False)

    def _route_cache_key(self, start_xy: np.ndarray, goal_xy: np.ndarray, clearance_m: float) -> Tuple[int, int, int, int, float]:
        snap_m = max(self.config.snap_m, 1.0)
        return (
            int(round(float(start_xy[0]) / snap_m)),
            int(round(float(start_xy[1]) / snap_m)),
            int(round(float(goal_xy[0]) / snap_m)),
            int(round(float(goal_xy[1]) / snap_m)),
            round(float(clearance_m), 3),
        )

    def _point_within_bounds(self, point_xy: np.ndarray, clearance_m: float) -> bool:
        margin_m = max(float(clearance_m), 0.0)
        return (
            self.bounds_xy_m.x_min_m + margin_m <= point_xy[0] <= self.bounds_xy_m.x_max_m - margin_m
            and self.bounds_xy_m.y_min_m + margin_m <= point_xy[1] <= self.bounds_xy_m.y_max_m - margin_m
        )

    def _blocking_primitive(self, point_xy: np.ndarray, clearance_m: float) -> Optional[ObstaclePrimitive]:
        for primitive in self._hard_primitives:
            polygon = self.expanded_polygon_for_primitive(primitive, clearance_m)
            if _point_in_polygon(point_xy, polygon):
                return primitive
        return None

    def _segment_is_free(self, start_xy: np.ndarray, end_xy: np.ndarray, clearance_m: float) -> bool:
        if not self._point_within_bounds(start_xy, clearance_m) or not self._point_within_bounds(end_xy, clearance_m):
            return False

        segment_bounds = _segment_bounds(start_xy, end_xy).padded(max(float(clearance_m), 0.5))
        for primitive in self.obstacle_layer.query_obstacles(segment_bounds):
            if primitive.blocker_type not in HARD_BLOCKER_TYPES:
                continue
            polygon = self.expanded_polygon_for_primitive(primitive, clearance_m)
            if _point_in_polygon(start_xy, polygon) and not _point_on_polygon_boundary(start_xy, polygon):
                return False
            if _point_in_polygon(end_xy, polygon) and not _point_on_polygon_boundary(end_xy, polygon):
                return False
            intervals = _segment_polygon_intervals(start_xy, end_xy, polygon)
            if any((end_t - start_t) > 1.0e-6 for start_t, end_t in intervals):
                return False
        return True

    def _build_visibility_graph(
        self,
        nodes: Sequence[np.ndarray],
        polygon_node_indices: Sequence[Sequence[int]],
        clearance_m: float,
    ) -> Dict[int, List[int]]:
        adjacency: Dict[int, List[int]] = {index: [] for index in range(len(nodes))}
        for left_index in range(len(nodes)):
            for right_index in range(left_index + 1, len(nodes)):
                start = nodes[left_index]
                end = nodes[right_index]
                if not self._segment_is_free(start, end, clearance_m):
                    continue
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
        for indices in polygon_node_indices:
            if len(indices) < 2:
                continue
            for index, current_node in enumerate(indices):
                next_node = indices[(index + 1) % len(indices)]
                if next_node not in adjacency[current_node]:
                    adjacency[current_node].append(next_node)
                if current_node not in adjacency[next_node]:
                    adjacency[next_node].append(current_node)
        return adjacency

    def _astar_path(
        self,
        nodes: Sequence[np.ndarray],
        adjacency: Dict[int, List[int]],
    ) -> Optional[np.ndarray]:
        goal_index = 1
        queue: List[Tuple[float, float, int, int]] = []
        start_state = (0, -1)
        best_cost: Dict[Tuple[int, int], float] = {start_state: 0.0}
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_state: None}
        heapq.heappush(queue, (float(np.linalg.norm(nodes[goal_index] - nodes[0])), 0.0, 0, -1))

        best_goal_state: Optional[Tuple[int, int]] = None
        best_goal_cost = float("inf")
        while queue:
            _, current_cost, current_index, previous_index = heapq.heappop(queue)
            state = (current_index, previous_index)
            if current_cost > best_cost.get(state, float("inf")) + 1.0e-9:
                continue
            if current_index == goal_index:
                best_goal_state = state
                best_goal_cost = current_cost
                break
            for neighbor_index in adjacency.get(current_index, []):
                edge_length = float(np.linalg.norm(nodes[neighbor_index] - nodes[current_index]))
                turn_penalty = 0.0
                if previous_index >= 0:
                    previous_vector = nodes[current_index] - nodes[previous_index]
                    next_vector = nodes[neighbor_index] - nodes[current_index]
                    denom = max(float(np.linalg.norm(previous_vector) * np.linalg.norm(next_vector)), 1.0e-9)
                    cosine = float(np.clip(np.dot(previous_vector, next_vector) / denom, -1.0, 1.0))
                    turn_penalty = self.config.turn_penalty_m * math.acos(cosine)
                next_cost = current_cost + edge_length + turn_penalty
                next_state = (neighbor_index, current_index)
                if next_cost + 1.0e-9 >= best_cost.get(next_state, float("inf")):
                    continue
                best_cost[next_state] = next_cost
                parents[next_state] = state
                heuristic = float(np.linalg.norm(nodes[goal_index] - nodes[neighbor_index]))
                heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbor_index, current_index))

        if best_goal_state is None or best_goal_cost == float("inf"):
            return None

        indices: List[int] = []
        current_state: Optional[Tuple[int, int]] = best_goal_state
        while current_state is not None:
            indices.append(current_state[0])
            current_state = parents.get(current_state)
        indices.reverse()
        return np.vstack([nodes[index] for index in indices])

    def _smooth_path(self, points_xy_m: np.ndarray, clearance_m: float) -> np.ndarray:
        if len(points_xy_m) <= 2:
            return points_xy_m
        smoothed: List[np.ndarray] = [points_xy_m[0]]
        index = 0
        while index < len(points_xy_m) - 1:
            next_index = len(points_xy_m) - 1
            while next_index > index + 1:
                if self._segment_is_free(points_xy_m[index], points_xy_m[next_index], clearance_m):
                    break
                next_index -= 1
            smoothed.append(points_xy_m[next_index])
            index = next_index
        return np.vstack(smoothed)


__all__ = [
    "PathPlanner2D",
    "PlannerConfig",
    "PlannerRoute",
]
