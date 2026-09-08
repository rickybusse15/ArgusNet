from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from argusnet._native_world import NativeHandle, call

from .environment import Bounds2D


def _as_float_array(values: Sequence[Sequence[float]], *, dims: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != dims:
        raise ValueError(f"Expected an array of shape (n, {dims}).")
    return array


def _unique_sorted(values: Iterable[float], tolerance: float = 1.0e-9) -> list[float]:
    deduped: list[float] = []
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
    return not dot > float(np.dot(segment, segment)) + tolerance


def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    return bool(call("point_in_polygon", point=point_xy, polygon=polygon_xy))


def _segment_intersection_parameters(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[float, float] | None:
    result = call("segment_hit", a0=a0, a1=a1, b0=b0, b1=b1)
    return None if result is None else tuple(result)


def _segment_polygon_intervals(
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
    polygon_xy: np.ndarray,
) -> list[tuple[float, float]]:
    return [
        tuple(v)
        for v in call("polygon_intervals", origin=origin_xy, target=target_xy, polygon=polygon_xy)
    ]


def _nearest_point_on_segment(
    point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray
) -> np.ndarray:
    return np.asarray(
        call("nearest_point", point=point_xy, start=start_xy, end=end_xy), dtype=float
    )


def _signed_polygon_area(polygon_xy: np.ndarray) -> float:
    return float(call("polygon_area", polygon=polygon_xy))


def _edge_outward_normal(start_xy: np.ndarray, end_xy: np.ndarray, *, is_ccw: bool) -> np.ndarray:
    return np.asarray(call("outward_normal", start=start_xy, end=end_xy, ccw=is_ccw), dtype=float)


@dataclass(frozen=True)
class ObstaclePrimitive:
    primitive_id: str
    blocker_type: str

    @cached_property
    def _native(self):
        return NativeHandle("obstacle_new", metadata=self.to_metadata())

    def bounds_xy_m(self) -> Bounds2D:
        raise NotImplementedError

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> float | None:
        raise NotImplementedError

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        raise NotImplementedError

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def to_metadata(self) -> dict[str, object]:
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
        return Bounds2D(**call("obstacle", handle=self._native.id, method="bounds"))

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return float(
            call("obstacle", handle=self._native.id, method="length", origin=origin, target=target)
        )

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> float | None:
        return call("obstacle", handle=self._native.id, method="hit", origin=origin, target=target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return bool(
            call("obstacle", handle=self._native.id, method="inside", point=[x_m, y_m, z_m])
        )

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return np.asarray(
            call(
                "obstacle", handle=self._native.id, method="push", point=[x_m, y_m], margin=margin_m
            )
        )

    def to_metadata(self) -> dict[str, object]:
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

    def to_metadata(self) -> dict[str, object]:
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
        return Bounds2D(**call("obstacle", handle=self._native.id, method="bounds"))

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return float(
            call("obstacle", handle=self._native.id, method="length", origin=origin, target=target)
        )

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> float | None:
        return call("obstacle", handle=self._native.id, method="hit", origin=origin, target=target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return bool(
            call("obstacle", handle=self._native.id, method="inside", point=[x_m, y_m, z_m])
        )

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return np.asarray(
            call(
                "obstacle", handle=self._native.id, method="push", point=[x_m, y_m], margin=margin_m
            )
        )

    def to_metadata(self) -> dict[str, object]:
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
        return Bounds2D(**call("obstacle", handle=self._native.id, method="bounds"))

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return float(
            call("obstacle", handle=self._native.id, method="length", origin=origin, target=target)
        )

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> float | None:
        return call("obstacle", handle=self._native.id, method="hit", origin=origin, target=target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return bool(
            call("obstacle", handle=self._native.id, method="inside", point=[x_m, y_m, z_m])
        )

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return np.asarray(
            call(
                "obstacle", handle=self._native.id, method="push", point=[x_m, y_m], margin=margin_m
            )
        )

    def to_metadata(self) -> dict[str, object]:
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
        return Bounds2D(**call("obstacle", handle=self._native.id, method="bounds"))

    def path_length_inside(self, origin: np.ndarray, target: np.ndarray) -> float:
        return float(
            call("obstacle", handle=self._native.id, method="length", origin=origin, target=target)
        )

    def first_hit_t(self, origin: np.ndarray, target: np.ndarray) -> float | None:
        return call("obstacle", handle=self._native.id, method="hit", origin=origin, target=target)

    def point_inside(self, x_m: float, y_m: float, z_m: float) -> bool:
        return bool(
            call("obstacle", handle=self._native.id, method="inside", point=[x_m, y_m, z_m])
        )

    def push_outside_xy(self, x_m: float, y_m: float, margin_m: float = 1.0) -> np.ndarray:
        return np.asarray(
            call(
                "obstacle", handle=self._native.id, method="push", point=[x_m, y_m], margin=margin_m
            )
        )

    def to_metadata(self) -> dict[str, object]:
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
