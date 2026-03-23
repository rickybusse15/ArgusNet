from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .environment import Bounds2D


def _as_float_array(values: Sequence[Sequence[float]], *, dims: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != dims:
        raise ValueError(f"Expected an array of shape (n, {dims}).")
    return array


def _unique_sorted(values: Iterable[float], tolerance: float = 1.0e-9) -> List[float]:
    deduped: List[float] = []
    for value in sorted(values):
        if not deduped or abs(value - deduped[-1]) > tolerance:
            deduped.append(value)
    return deduped


def _point_on_segment(
    point_xy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    tolerance: float = 1.0e-6,
) -> bool:
    segment = end_xy - start_xy
    point = point_xy - start_xy
    cross = abs((segment[0] * point[1]) - (segment[1] * point[0]))
    if cross > tolerance:
        return False
    dot = float(np.dot(point, segment))
    if dot < -tolerance:
        return False
    if dot > float(np.dot(segment, segment)) + tolerance:
        return False
    return True


def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    inside = False
    for index in range(len(polygon_xy)):
        a = polygon_xy[index]
        b = polygon_xy[(index + 1) % len(polygon_xy)]
        if _point_on_segment(point_xy, a, b):
            return True
        intersects = ((a[1] > point_xy[1]) != (b[1] > point_xy[1]))
        if intersects:
            x_cross = (b[0] - a[0]) * (point_xy[1] - a[1]) / max((b[1] - a[1]), 1.0e-12) + a[0]
            if point_xy[0] < x_cross:
                inside = not inside
    return inside


def _segment_intersection_parameters(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> Optional[Tuple[float, float]]:
    da = a1 - a0
    db = b1 - b0
    denominator = (da[0] * db[1]) - (da[1] * db[0])
    if abs(denominator) <= 1.0e-9:
        return None
    delta = b0 - a0
    ta = ((delta[0] * db[1]) - (delta[1] * db[0])) / denominator
    tb = ((delta[0] * da[1]) - (delta[1] * da[0])) / denominator
    if -1.0e-9 <= ta <= 1.0 + 1.0e-9 and -1.0e-9 <= tb <= 1.0 + 1.0e-9:
        return (float(ta), float(tb))
    return None


def _segment_polygon_intervals(
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
    polygon_xy: np.ndarray,
) -> List[Tuple[float, float]]:
    t_values: List[float] = [0.0, 1.0]
    for index in range(len(polygon_xy)):
        edge_start = polygon_xy[index]
        edge_end = polygon_xy[(index + 1) % len(polygon_xy)]
        parameters = _segment_intersection_parameters(origin_xy, target_xy, edge_start, edge_end)
        if parameters is not None:
            t_values.append(parameters[0])
    t_values = _unique_sorted(value for value in t_values if -1.0e-9 <= value <= 1.0 + 1.0e-9)

    intervals: List[Tuple[float, float]] = []
    for start, end in zip(t_values[:-1], t_values[1:]):
        if end - start <= 1.0e-9:
            continue
        midpoint = (start + end) * 0.5
        point_xy = origin_xy + (target_xy - origin_xy) * midpoint
        if _point_in_polygon(point_xy, polygon_xy):
            intervals.append((start, end))
    return intervals


def _nearest_point_on_segment(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    segment = end_xy - start_xy
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-12:
        return start_xy.copy()
    t_value = float(np.clip(np.dot(point_xy - start_xy, segment) / length_sq, 0.0, 1.0))
    return start_xy + segment * t_value


def _signed_polygon_area(polygon_xy: np.ndarray) -> float:
    x_values = polygon_xy[:, 0]
    y_values = polygon_xy[:, 1]
    return 0.5 * float(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))


def _edge_outward_normal(start_xy: np.ndarray, end_xy: np.ndarray, *, is_ccw: bool) -> np.ndarray:
    edge = end_xy - start_xy
    length = max(float(np.linalg.norm(edge)), 1.0e-9)
    if is_ccw:
        return np.array([edge[1], -edge[0]], dtype=float) / length
    return np.array([-edge[1], edge[0]], dtype=float) / length


@dataclass(frozen=True)
class ObstaclePrimitive:
    primitive_id: str
    blocker_type: str

    def bounds_xy_m(self) -> Bounds2D:
        raise NotImplementedError

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        raise NotImplementedError

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def to_metadata(self) -> Dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class PolygonPrism(ObstaclePrimitive):
    footprint_xy_m: np.ndarray
    base_z_m: float
    top_z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "footprint_xy_m", _as_float_array(self.footprint_xy_m, dims=2))
        if len(self.footprint_xy_m) < 3:
            raise ValueError("PolygonPrism requires at least three footprint vertices.")
        if self.top_z_m <= self.base_z_m:
            raise ValueError("top_z_m must be greater than base_z_m.")

    def bounds_xy_m(self) -> Bounds2D:
        return Bounds2D(
            x_min_m=float(np.min(self.footprint_xy_m[:, 0])),
            x_max_m=float(np.max(self.footprint_xy_m[:, 0])),
            y_min_m=float(np.min(self.footprint_xy_m[:, 1])),
            y_max_m=float(np.max(self.footprint_xy_m[:, 1])),
        )

    def _inside_intervals(self, origin: np.ndarray, target: np.ndarray) -> List[Tuple[float, float]]:
        intervals = _segment_polygon_intervals(origin[:2], target[:2], self.footprint_xy_m)
        filtered: List[Tuple[float, float]] = []
        for start, end in intervals:
            z_start = float(origin[2] + ((target[2] - origin[2]) * start))
            z_end = float(origin[2] + ((target[2] - origin[2]) * end))
            if max(z_start, z_end) >= self.base_z_m and min(z_start, z_end) <= self.top_z_m:
                filtered.append((start, end))
        return filtered

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        distance_m = float(np.linalg.norm(target - origin))
        return float(sum((end - start) * distance_m for start, end in self._inside_intervals(origin, target)))

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> Optional[float]:
        intervals = self._inside_intervals(origin, target)
        return intervals[0][0] if intervals else None

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        if z_m < self.base_z_m - 1.0e-9 or z_m > self.top_z_m + 1.0e-9:
            return False
        return _point_in_polygon(np.array([x_m, y_m], dtype=float), self.footprint_xy_m)

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        point_xy = np.array([x_m, y_m], dtype=float)
        if not _point_in_polygon(point_xy, self.footprint_xy_m):
            return point_xy

        is_ccw = _signed_polygon_area(self.footprint_xy_m) >= 0.0
        best_candidate = point_xy.copy()
        best_distance = float("inf")
        for index in range(len(self.footprint_xy_m)):
            edge_start = self.footprint_xy_m[index]
            edge_end = self.footprint_xy_m[(index + 1) % len(self.footprint_xy_m)]
            boundary_point = _nearest_point_on_segment(point_xy, edge_start, edge_end)
            outward_normal = _edge_outward_normal(edge_start, edge_end, is_ccw=is_ccw)
            candidate = boundary_point + outward_normal * max(margin_m, 1.0)
            for _ in range(6):
                if not _point_in_polygon(candidate, self.footprint_xy_m):
                    break
                candidate = candidate + outward_normal * max(margin_m, 1.0)
            distance = float(np.linalg.norm(candidate - point_xy))
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
        return best_candidate

    def to_metadata(self) -> Dict[str, object]:
        return {
            "object_id": self.primitive_id,
            "kind": "polygon-prism-v1",
            "blocker_type": self.blocker_type,
            "footprint_xy_m": self.footprint_xy_m.tolist(),
            "base_elevation_m": self.base_z_m,
            "top_elevation_m": self.top_z_m,
        }


@dataclass(frozen=True)
class BuildingPrism(PolygonPrism):
    def __init__(
        self,
        primitive_id: str,
        footprint_xy_m: Sequence[Sequence[float]],
        base_z_m: float,
        top_z_m: float,
    ) -> None:
        super().__init__(
            primitive_id=primitive_id,
            blocker_type="building",
            footprint_xy_m=np.asarray(footprint_xy_m, dtype=float),
            base_z_m=base_z_m,
            top_z_m=top_z_m,
        )


@dataclass(frozen=True)
class ForestStand(PolygonPrism):
    density: float = 0.35

    def __init__(
        self,
        primitive_id: str,
        footprint_xy_m: Sequence[Sequence[float]],
        canopy_base_z_m: float,
        canopy_top_z_m: float,
        density: float = 0.35,
    ) -> None:
        super().__init__(
            primitive_id=primitive_id,
            blocker_type="vegetation",
            footprint_xy_m=np.asarray(footprint_xy_m, dtype=float),
            base_z_m=canopy_base_z_m,
            top_z_m=canopy_top_z_m,
        )
        object.__setattr__(self, "density", float(max(density, 0.0)))

    def to_metadata(self) -> Dict[str, object]:
        metadata = super().to_metadata()
        metadata["kind"] = "forest-stand-v1"
        metadata["density"] = self.density
        return metadata


@dataclass(frozen=True)
class OrientedBox(ObstaclePrimitive):
    center_x_m: float
    center_y_m: float
    length_m: float
    width_m: float
    yaw_rad: float
    base_z_m: float
    top_z_m: float

    def __post_init__(self) -> None:
        if self.length_m <= 0.0 or self.width_m <= 0.0 or self.top_z_m <= self.base_z_m:
            raise ValueError("Invalid OrientedBox dimensions.")

    def footprint_xy_m(self) -> np.ndarray:
        half_length = self.length_m * 0.5
        half_width = self.width_m * 0.5
        corners = np.array(
            [
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
            ],
            dtype=float,
        )
        rotation = np.array(
            [
                [math.cos(self.yaw_rad), -math.sin(self.yaw_rad)],
                [math.sin(self.yaw_rad), math.cos(self.yaw_rad)],
            ],
            dtype=float,
        )
        return (corners @ rotation.T) + np.array([self.center_x_m, self.center_y_m], dtype=float)

    def _as_prism(self) -> PolygonPrism:
        return PolygonPrism(
            primitive_id=self.primitive_id,
            blocker_type=self.blocker_type,
            footprint_xy_m=self.footprint_xy_m(),
            base_z_m=self.base_z_m,
            top_z_m=self.top_z_m,
        )

    def bounds_xy_m(self) -> Bounds2D:
        return self._as_prism().bounds_xy_m()

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return self._as_prism().path_length_inside(origin, target)

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> Optional[float]:
        return self._as_prism().first_hit_t(origin, target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return self._as_prism().point_inside(x_m, y_m, z_m)

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return self._as_prism().push_outside_xy(x_m, y_m, margin_m=margin_m)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "object_id": self.primitive_id,
            "kind": "box-v1",
            "blocker_type": self.blocker_type,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "yaw_rad": self.yaw_rad,
            "base_elevation_m": self.base_z_m,
            "top_elevation_m": self.top_z_m,
            "footprint_xy_m": self.footprint_xy_m().tolist(),
        }


@dataclass(frozen=True)
class WallSegment(ObstaclePrimitive):
    start_xy_m: np.ndarray
    end_xy_m: np.ndarray
    thickness_m: float
    base_z_m: float
    top_z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_xy_m", np.asarray(self.start_xy_m, dtype=float).reshape(2))
        object.__setattr__(self, "end_xy_m", np.asarray(self.end_xy_m, dtype=float).reshape(2))
        if self.thickness_m <= 0.0 or self.top_z_m <= self.base_z_m:
            raise ValueError("Invalid WallSegment dimensions.")

    def footprint_xy_m(self) -> np.ndarray:
        segment = self.end_xy_m - self.start_xy_m
        length = max(float(np.linalg.norm(segment)), 1.0e-9)
        normal = np.array([-segment[1], segment[0]], dtype=float) / length
        offset = normal * (self.thickness_m * 0.5)
        return np.vstack(
            [
                self.start_xy_m - offset,
                self.end_xy_m - offset,
                self.end_xy_m + offset,
                self.start_xy_m + offset,
            ]
        )

    def _as_prism(self) -> PolygonPrism:
        return PolygonPrism(
            primitive_id=self.primitive_id,
            blocker_type=self.blocker_type,
            footprint_xy_m=self.footprint_xy_m(),
            base_z_m=self.base_z_m,
            top_z_m=self.top_z_m,
        )

    def bounds_xy_m(self) -> Bounds2D:
        return self._as_prism().bounds_xy_m()

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return self._as_prism().path_length_inside(origin, target)

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> Optional[float]:
        return self._as_prism().first_hit_t(origin, target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return self._as_prism().point_inside(x_m, y_m, z_m)

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return self._as_prism().push_outside_xy(x_m, y_m, margin_m=margin_m)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "object_id": self.primitive_id,
            "kind": "wall-v1",
            "blocker_type": self.blocker_type,
            "start_xy_m": self.start_xy_m.tolist(),
            "end_xy_m": self.end_xy_m.tolist(),
            "thickness_m": self.thickness_m,
            "base_elevation_m": self.base_z_m,
            "top_elevation_m": self.top_z_m,
            "footprint_xy_m": self.footprint_xy_m().tolist(),
        }


@dataclass(frozen=True)
class CylinderObstacle(ObstaclePrimitive):
    center_x_m: float
    center_y_m: float
    radius_m: float
    base_z_m: float
    top_z_m: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0 or self.top_z_m <= self.base_z_m:
            raise ValueError("Invalid CylinderObstacle dimensions.")

    def bounds_xy_m(self) -> Bounds2D:
        return Bounds2D(
            self.center_x_m - self.radius_m,
            self.center_x_m + self.radius_m,
            self.center_y_m - self.radius_m,
            self.center_y_m + self.radius_m,
        )

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        interval = self._t_interval(origin, target)
        if interval is None:
            return 0.0
        return float((interval[1] - interval[0]) * np.linalg.norm(target - origin))

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> Optional[float]:
        interval = self._t_interval(origin, target)
        return None if interval is None else interval[0]

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        if z_m < self.base_z_m - 1.0e-9 or z_m > self.top_z_m + 1.0e-9:
            return False
        dx = x_m - self.center_x_m
        dy = y_m - self.center_y_m
        return (dx * dx + dy * dy) <= (self.radius_m * self.radius_m + 1.0e-9)

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        delta = np.array([x_m - self.center_x_m, y_m - self.center_y_m], dtype=float)
        radius = float(np.linalg.norm(delta))
        if radius <= 1.0e-9:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = delta / radius
        return np.array(
            [
                self.center_x_m + direction[0] * (self.radius_m + max(margin_m, 1.0)),
                self.center_y_m + direction[1] * (self.radius_m + max(margin_m, 1.0)),
            ],
            dtype=float,
        )

    def _t_interval(self, origin: np.ndarray, target: np.ndarray) -> Optional[Tuple[float, float]]:
        origin_xy = origin[:2]
        target_xy = target[:2]
        segment_xy = target_xy - origin_xy
        relative_xy = origin_xy - np.array([self.center_x_m, self.center_y_m], dtype=float)
        a = float(np.dot(segment_xy, segment_xy))
        if a <= 1.0e-9:
            if float(np.dot(relative_xy, relative_xy)) > self.radius_m * self.radius_m:
                return None
            if max(float(origin[2]), float(target[2])) < self.base_z_m or min(float(origin[2]), float(target[2])) > self.top_z_m:
                return None
            return (0.0, 1.0)
        b = 2.0 * float(np.dot(relative_xy, segment_xy))
        c = float(np.dot(relative_xy, relative_xy) - (self.radius_m * self.radius_m))
        discriminant = b * b - (4.0 * a * c)
        if discriminant < 0.0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        start = max(0.0, min(t0, t1))
        end = min(1.0, max(t0, t1))
        if start > end:
            return None
        z_start = float(origin[2] + ((target[2] - origin[2]) * start))
        z_end = float(origin[2] + ((target[2] - origin[2]) * end))
        if max(z_start, z_end) < self.base_z_m or min(z_start, z_end) > self.top_z_m:
            return None
        return (start, end)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "object_id": self.primitive_id,
            "kind": "cylinder-v1",
            "blocker_type": self.blocker_type,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "radius_m": self.radius_m,
            "base_elevation_m": self.base_z_m,
            "top_elevation_m": self.top_z_m,
        }


__all__ = [
    "_as_float_array",
    "_point_in_polygon",
    "_point_on_segment",
    "_segment_intersection_parameters",
    "_segment_polygon_intervals",
    "_unique_sorted",
    "BuildingPrism",
    "CylinderObstacle",
    "ForestStand",
    "ObstaclePrimitive",
    "OrientedBox",
    "PolygonPrism",
    "WallSegment",
]
