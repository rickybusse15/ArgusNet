from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .environment import (
    Bounds2D,
    BuildingPrism,
    CylinderObstacle,
    EnvironmentCRS,
    EnvironmentModel,
    ForestStand,
    LandCoverClass,
    LandCoverLayer,
    OrientedBox,
    PolygonPrism,
    SensorVisibilityModel,
    VisibilityResult,
    WallSegment,
    load_environment_bundle,
    write_environment_bundle,
)


def _as_xy(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(2)
    return array


def _segment_frame(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> Tuple[float, float, float]:
    segment = end_xy - start_xy
    length = max(float(np.linalg.norm(segment)), 1.0e-9)
    direction = segment / length
    relative = point_xy - start_xy
    along = float(np.dot(relative, direction))
    clamped_along = float(np.clip(along, 0.0, length))
    closest = start_xy + direction * clamped_along
    cross = float(np.linalg.norm(point_xy - closest))
    return along, clamped_along, cross


@dataclass(frozen=True)
class TerrainFeature:
    def height_contribution(self, x_m: float, y_m: float) -> float:
        raise NotImplementedError

    def scaled(self, scale: float) -> "TerrainFeature":
        raise NotImplementedError

    def to_metadata(self) -> Dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class MountainRange(TerrainFeature):
    start_xy: np.ndarray
    end_xy: np.ndarray
    peak_elevation_m: float
    width_m: float
    peak_count: int
    roughness: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_xy", _as_xy(self.start_xy))
        object.__setattr__(self, "end_xy", _as_xy(self.end_xy))

    def height_contribution(self, x_m: float, y_m: float) -> float:
        point_xy = np.array([x_m, y_m], dtype=float)
        segment = self.end_xy - self.start_xy
        length = max(float(np.linalg.norm(segment)), self.width_m, 1.0)
        _, clamped_along, cross = _segment_frame(point_xy, self.start_xy, self.end_xy)
        spacing = length / max(self.peak_count + 1, 1)
        ridge_envelope = math.exp(-0.5 * (cross / max(self.width_m, 1.0)) ** 2)
        contribution = 0.0
        for peak_index in range(max(self.peak_count, 1)):
            peak_along = spacing * (peak_index + 1)
            peak_span = max(spacing * 0.9, self.width_m * 0.75, 1.0)
            roughness_phase = (peak_index + 1) * 1.618
            amplitude = self.peak_elevation_m * (0.8 + self.roughness * 0.2 * math.sin(roughness_phase))
            along_delta = (clamped_along - peak_along) / peak_span
            contribution += amplitude * math.exp(-0.5 * along_delta * along_delta)
        return contribution * ridge_envelope

    def scaled(self, scale: float) -> "MountainRange":
        return MountainRange(
            start_xy=self.start_xy * scale,
            end_xy=self.end_xy * scale,
            peak_elevation_m=self.peak_elevation_m,
            width_m=self.width_m * scale,
            peak_count=self.peak_count,
            roughness=self.roughness,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "mountain-range-v1",
            "start_xy": self.start_xy.tolist(),
            "end_xy": self.end_xy.tolist(),
            "peak_elevation_m": self.peak_elevation_m,
            "width_m": self.width_m,
            "peak_count": self.peak_count,
            "roughness": self.roughness,
        }


@dataclass(frozen=True)
class Valley(TerrainFeature):
    start_xy: np.ndarray
    end_xy: np.ndarray
    depth_m: float
    width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_xy", _as_xy(self.start_xy))
        object.__setattr__(self, "end_xy", _as_xy(self.end_xy))

    def height_contribution(self, x_m: float, y_m: float) -> float:
        point_xy = np.array([x_m, y_m], dtype=float)
        _, _, cross = _segment_frame(point_xy, self.start_xy, self.end_xy)
        normalized = cross / max(self.width_m, 1.0)
        return -self.depth_m * math.exp(-0.5 * normalized * normalized)

    def scaled(self, scale: float) -> "Valley":
        return Valley(
            start_xy=self.start_xy * scale,
            end_xy=self.end_xy * scale,
            depth_m=self.depth_m,
            width_m=self.width_m * scale,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "valley-v1",
            "start_xy": self.start_xy.tolist(),
            "end_xy": self.end_xy.tolist(),
            "depth_m": self.depth_m,
            "width_m": self.width_m,
        }


@dataclass(frozen=True)
class Plateau(TerrainFeature):
    center_xy: np.ndarray
    radius_m: float
    elevation_m: float
    edge_sharpness: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_xy", _as_xy(self.center_xy))

    def height_contribution(self, x_m: float, y_m: float) -> float:
        radius = max(self.radius_m, 1.0)
        distance = float(np.linalg.norm(np.array([x_m, y_m], dtype=float) - self.center_xy))
        normalized = distance / radius
        sharpness = max(self.edge_sharpness, 1.0)
        return self.elevation_m / (1.0 + normalized ** sharpness)

    def scaled(self, scale: float) -> "Plateau":
        return Plateau(
            center_xy=self.center_xy * scale,
            radius_m=self.radius_m * scale,
            elevation_m=self.elevation_m,
            edge_sharpness=self.edge_sharpness,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "plateau-v1",
            "center_xy": self.center_xy.tolist(),
            "radius_m": self.radius_m,
            "elevation_m": self.elevation_m,
            "edge_sharpness": self.edge_sharpness,
        }


@dataclass(frozen=True)
class RidgeLine(TerrainFeature):
    start_xy: np.ndarray
    end_xy: np.ndarray
    peak_elevation_m: float
    width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_xy", _as_xy(self.start_xy))
        object.__setattr__(self, "end_xy", _as_xy(self.end_xy))

    def height_contribution(self, x_m: float, y_m: float) -> float:
        point_xy = np.array([x_m, y_m], dtype=float)
        _, _, cross = _segment_frame(point_xy, self.start_xy, self.end_xy)
        normalized = cross / max(self.width_m, 1.0)
        return self.peak_elevation_m * math.exp(-0.5 * normalized * normalized)

    def scaled(self, scale: float) -> "RidgeLine":
        return RidgeLine(
            start_xy=self.start_xy * scale,
            end_xy=self.end_xy * scale,
            peak_elevation_m=self.peak_elevation_m,
            width_m=self.width_m * scale,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "ridgeline-v1",
            "start_xy": self.start_xy.tolist(),
            "end_xy": self.end_xy.tolist(),
            "peak_elevation_m": self.peak_elevation_m,
            "width_m": self.width_m,
        }


def _noise_hash2d(ix: int, iy: int, seed: int) -> float:
    """Deterministic hash of grid coordinates to a float in [-1, 1]."""
    h = (ix * 73856093) ^ (iy * 19349663) ^ seed
    h = ((h >> 13) ^ h) * 0x45D9F3B & 0xFFFFFFFF
    h = ((h >> 13) ^ h) * 0x45D9F3B & 0xFFFFFFFF
    h = (h >> 13) ^ h
    return (h & 0xFFFF) / 32767.5 - 1.0


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(x: float, y: float, seed: int) -> float:
    """Bilinearly interpolated value noise at (x, y)."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = _smoothstep(x - ix)
    fy = _smoothstep(y - iy)
    n00 = _noise_hash2d(ix, iy, seed)
    n10 = _noise_hash2d(ix + 1, iy, seed)
    n01 = _noise_hash2d(ix, iy + 1, seed)
    n11 = _noise_hash2d(ix + 1, iy + 1, seed)
    nx0 = n00 + fx * (n10 - n00)
    nx1 = n01 + fx * (n11 - n01)
    return nx0 + fy * (nx1 - nx0)


@dataclass(frozen=True)
class NoiseLayer(TerrainFeature):
    """Fractal Brownian motion noise for natural terrain roughness."""

    seed: int
    amplitude_m: float
    base_wavelength_m: float
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0

    def height_contribution(self, x_m: float, y_m: float) -> float:
        total = 0.0
        wavelength = max(self.base_wavelength_m, 1.0)
        amp = 1.0
        for octave in range(self.octaves):
            total += amp * _value_noise_2d(
                x_m / wavelength, y_m / wavelength, self.seed + octave
            )
            wavelength /= self.lacunarity
            amp *= self.persistence
        return total * self.amplitude_m

    def scaled(self, scale: float) -> "NoiseLayer":
        return NoiseLayer(
            seed=self.seed,
            amplitude_m=self.amplitude_m,
            base_wavelength_m=self.base_wavelength_m * scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "noise-layer-v1",
            "seed": self.seed,
            "amplitude_m": self.amplitude_m,
            "base_wavelength_m": self.base_wavelength_m,
            "octaves": self.octaves,
            "persistence": self.persistence,
            "lacunarity": self.lacunarity,
        }


@dataclass(frozen=True)
class River(TerrainFeature):
    """Sinuous water channel carved along a polyline centerline."""

    control_points: Tuple[np.ndarray, ...]
    width_m: float
    depth_m: float
    bank_slope: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_points",
            tuple(_as_xy(p) for p in self.control_points),
        )

    def height_contribution(self, x_m: float, y_m: float) -> float:
        point_xy = np.array([x_m, y_m], dtype=float)
        min_cross = float("inf")
        for i in range(len(self.control_points) - 1):
            _, _, cross = _segment_frame(
                point_xy, self.control_points[i], self.control_points[i + 1]
            )
            if cross < min_cross:
                min_cross = cross
        half_w = max(self.width_m * 0.5, 1.0)
        normalized = min_cross / half_w
        return -self.depth_m * math.exp(-0.5 * normalized ** self.bank_slope)

    def scaled(self, scale: float) -> "River":
        return River(
            control_points=tuple(p * scale for p in self.control_points),
            width_m=self.width_m * scale,
            depth_m=self.depth_m,
            bank_slope=self.bank_slope,
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "river-v1",
            "control_points": [p.tolist() for p in self.control_points],
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "bank_slope": self.bank_slope,
        }


@dataclass(frozen=True)
class TerrainModel:
    ground_plane_m: float = 0.0
    base_elevation_m: float = 0.0
    slope_x_m_per_m: float = 0.005
    slope_y_m_per_m: float = -0.003
    wave_amplitude_m: float = 7.5
    wave_length_x_m: float = 280.0
    wave_length_y_m: float = 210.0
    wave_phase_x_rad: float = 0.35
    wave_phase_y_rad: float = -0.55
    ridge_amplitude_m: float = 22.0
    ridge_center_x_m: float = 95.0
    ridge_center_y_m: float = -70.0
    ridge_radius_m: float = 180.0
    basin_depth_m: float = 14.0
    basin_center_x_m: float = -165.0
    basin_center_y_m: float = 145.0
    basin_radius_m: float = 220.0
    features: Tuple[TerrainFeature, ...] = ()

    @classmethod
    def default(cls) -> "TerrainModel":
        return cls()

    def analytic_height_at(self, x_m: float, y_m: float) -> float:
        ridge_r2 = max(self.ridge_radius_m ** 2, 1.0)
        basin_r2 = max(self.basin_radius_m ** 2, 1.0)

        x_freq = (2.0 * np.pi) / max(self.wave_length_x_m, 1.0)
        y_freq = (2.0 * np.pi) / max(self.wave_length_y_m, 1.0)
        wave = self.wave_amplitude_m * np.sin(x_m * x_freq + self.wave_phase_x_rad) * np.cos(
            y_m * y_freq + self.wave_phase_y_rad
        )

        ridge_dx = x_m - self.ridge_center_x_m
        ridge_dy = y_m - self.ridge_center_y_m
        ridge = self.ridge_amplitude_m * np.exp(-(ridge_dx * ridge_dx + ridge_dy * ridge_dy) / (2.0 * ridge_r2))

        basin_dx = x_m - self.basin_center_x_m
        basin_dy = y_m - self.basin_center_y_m
        basin = self.basin_depth_m * np.exp(-(basin_dx * basin_dx + basin_dy * basin_dy) / (2.0 * basin_r2))

        slope = x_m * self.slope_x_m_per_m + y_m * self.slope_y_m_per_m
        feature_height = sum(feature.height_contribution(x_m, y_m) for feature in self.features)
        return float(self.base_elevation_m + slope + wave + ridge - basin + feature_height)

    def height_at(self, x_m: float, y_m: float) -> float:
        return max(self.analytic_height_at(x_m, y_m), self.ground_plane_m)

    def clamp_altitude(self, xy_m: Sequence[float], z_m: float, min_agl_m: float) -> float:
        ground_m = self.height_at(float(xy_m[0]), float(xy_m[1]))
        return max(z_m, ground_m + max(min_agl_m, 0.0))

    def gradient_at(self, x_m: float, y_m: float, delta_m: float = 0.5) -> np.ndarray:
        delta = max(delta_m, 0.05)
        dz_dx = (self.height_at(x_m + delta, y_m) - self.height_at(x_m - delta, y_m)) / (2.0 * delta)
        dz_dy = (self.height_at(x_m, y_m + delta) - self.height_at(x_m, y_m - delta)) / (2.0 * delta)
        return np.array([dz_dx, dz_dy], dtype=float)

    def curvature_at(self, x_m: float, y_m: float, delta_m: float = 1.0) -> float:
        """Scalar surface curvature (Laplacian approximation).

        Positive = convex (hilltop), negative = concave (valley).
        """
        d = max(delta_m, 0.1)
        return (
            self.height_at(x_m + d, y_m)
            + self.height_at(x_m - d, y_m)
            + self.height_at(x_m, y_m + d)
            + self.height_at(x_m, y_m - d)
            - 4.0 * self.height_at(x_m, y_m)
        ) / (d * d)

    def slope_rad_at(self, x_m: float, y_m: float, delta_m: float = 1.0) -> float:
        """Slope magnitude in radians at the given point."""
        gx, gy = self.gradient_at(x_m, y_m, delta_m)
        return math.atan(math.sqrt(gx * gx + gy * gy))

    def scaled(self, scale: float) -> "TerrainModel":
        return TerrainModel(
            ground_plane_m=self.ground_plane_m,
            base_elevation_m=self.base_elevation_m,
            slope_x_m_per_m=self.slope_x_m_per_m,
            slope_y_m_per_m=self.slope_y_m_per_m,
            wave_amplitude_m=self.wave_amplitude_m,
            wave_length_x_m=self.wave_length_x_m * scale,
            wave_length_y_m=self.wave_length_y_m * scale,
            wave_phase_x_rad=self.wave_phase_x_rad,
            wave_phase_y_rad=self.wave_phase_y_rad,
            ridge_amplitude_m=self.ridge_amplitude_m,
            ridge_center_x_m=self.ridge_center_x_m * scale,
            ridge_center_y_m=self.ridge_center_y_m * scale,
            ridge_radius_m=self.ridge_radius_m * scale,
            basin_depth_m=self.basin_depth_m,
            basin_center_x_m=self.basin_center_x_m * scale,
            basin_center_y_m=self.basin_center_y_m * scale,
            basin_radius_m=self.basin_radius_m * scale,
            features=tuple(feature.scaled(scale) for feature in self.features),
        )

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "analytic-v2",
            "parameters": {
                "ground_plane_m": self.ground_plane_m,
                "base_elevation_m": self.base_elevation_m,
                "slope_x_m_per_m": self.slope_x_m_per_m,
                "slope_y_m_per_m": self.slope_y_m_per_m,
                "wave_amplitude_m": self.wave_amplitude_m,
                "wave_length_x_m": self.wave_length_x_m,
                "wave_length_y_m": self.wave_length_y_m,
                "wave_phase_x_rad": self.wave_phase_x_rad,
                "wave_phase_y_rad": self.wave_phase_y_rad,
                "ridge_amplitude_m": self.ridge_amplitude_m,
                "ridge_center_x_m": self.ridge_center_x_m,
                "ridge_center_y_m": self.ridge_center_y_m,
                "ridge_radius_m": self.ridge_radius_m,
                "basin_depth_m": self.basin_depth_m,
                "basin_center_x_m": self.basin_center_x_m,
                "basin_center_y_m": self.basin_center_y_m,
                "basin_radius_m": self.basin_radius_m,
                "features": [feature.to_metadata() for feature in self.features],
            },
        }


def alpine_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=52.0,
        slope_x_m_per_m=0.024,
        slope_y_m_per_m=-0.018,
        wave_amplitude_m=22.0,
        wave_length_x_m=180.0 * scale,
        wave_length_y_m=145.0 * scale,
        ridge_amplitude_m=38.0,
        ridge_center_x_m=-90.0 * scale,
        ridge_center_y_m=55.0 * scale,
        ridge_radius_m=115.0 * scale,
        basin_depth_m=18.0,
        basin_center_x_m=120.0 * scale,
        basin_center_y_m=-70.0 * scale,
        basin_radius_m=145.0 * scale,
        features=(
            MountainRange(
                start_xy=np.array([-320.0, -120.0], dtype=float) * scale,
                end_xy=np.array([280.0, 180.0], dtype=float) * scale,
                peak_elevation_m=110.0,
                width_m=78.0 * scale,
                peak_count=5,
                roughness=0.9,
            ),
            Valley(
                start_xy=np.array([-280.0, 200.0], dtype=float) * scale,
                end_xy=np.array([240.0, -160.0], dtype=float) * scale,
                depth_m=38.0,
                width_m=56.0 * scale,
            ),
            RidgeLine(
                start_xy=np.array([-180.0, -260.0], dtype=float) * scale,
                end_xy=np.array([150.0, -80.0], dtype=float) * scale,
                peak_elevation_m=52.0,
                width_m=44.0 * scale,
            ),
            NoiseLayer(seed=101, amplitude_m=12.0, base_wavelength_m=80.0 * scale),
        ),
    )


def coastal_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=9.0,
        slope_x_m_per_m=0.001,
        slope_y_m_per_m=-0.02,
        wave_amplitude_m=4.5,
        wave_length_x_m=380.0 * scale,
        wave_length_y_m=320.0 * scale,
        ridge_amplitude_m=0.0,
        basin_depth_m=0.0,
        features=(
            Plateau(
                center_xy=np.array([0.0, 180.0], dtype=float) * scale,
                radius_m=155.0 * scale,
                elevation_m=18.0,
                edge_sharpness=6.0,
            ),
            RidgeLine(
                start_xy=np.array([-260.0, 110.0], dtype=float) * scale,
                end_xy=np.array([260.0, 70.0], dtype=float) * scale,
                peak_elevation_m=12.0,
                width_m=75.0 * scale,
            ),
        ),
    )


def urban_flat_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=6.0,
        slope_x_m_per_m=0.0015,
        slope_y_m_per_m=-0.001,
        wave_amplitude_m=1.0,
        wave_length_x_m=500.0 * scale,
        wave_length_y_m=450.0 * scale,
        ridge_amplitude_m=0.0,
        basin_depth_m=0.0,
        features=(
            Plateau(
                center_xy=np.array([-110.0, 90.0], dtype=float) * scale,
                radius_m=90.0 * scale,
                elevation_m=5.5,
                edge_sharpness=8.0,
            ),
        ),
    )


def desert_canyon_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=18.0,
        slope_x_m_per_m=0.004,
        slope_y_m_per_m=-0.002,
        wave_amplitude_m=5.5,
        wave_length_x_m=260.0 * scale,
        wave_length_y_m=240.0 * scale,
        ridge_amplitude_m=10.0,
        ridge_center_x_m=160.0 * scale,
        ridge_center_y_m=-120.0 * scale,
        ridge_radius_m=170.0 * scale,
        basin_depth_m=0.0,
        features=(
            Valley(
                start_xy=np.array([-320.0, -40.0], dtype=float) * scale,
                end_xy=np.array([260.0, 120.0], dtype=float) * scale,
                depth_m=36.0,
                width_m=58.0 * scale,
            ),
            Plateau(
                center_xy=np.array([-150.0, 170.0], dtype=float) * scale,
                radius_m=88.0 * scale,
                elevation_m=24.0,
                edge_sharpness=10.0,
            ),
            Plateau(
                center_xy=np.array([170.0, -150.0], dtype=float) * scale,
                radius_m=102.0 * scale,
                elevation_m=19.0,
                edge_sharpness=9.0,
            ),
        ),
    )


def rolling_highlands_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=28.0,
        slope_x_m_per_m=0.006,
        slope_y_m_per_m=-0.004,
        wave_amplitude_m=9.0,
        wave_length_x_m=420.0 * scale,
        wave_length_y_m=360.0 * scale,
        ridge_amplitude_m=14.0,
        ridge_center_x_m=-210.0 * scale,
        ridge_center_y_m=140.0 * scale,
        ridge_radius_m=260.0 * scale,
        basin_depth_m=6.0,
        basin_center_x_m=240.0 * scale,
        basin_center_y_m=-160.0 * scale,
        basin_radius_m=280.0 * scale,
        features=(
            MountainRange(
                start_xy=np.array([-460.0, -210.0], dtype=float) * scale,
                end_xy=np.array([410.0, 260.0], dtype=float) * scale,
                peak_elevation_m=44.0,
                width_m=120.0 * scale,
                peak_count=5,
                roughness=0.55,
            ),
            RidgeLine(
                start_xy=np.array([-420.0, 230.0], dtype=float) * scale,
                end_xy=np.array([380.0, 150.0], dtype=float) * scale,
                peak_elevation_m=18.0,
                width_m=95.0 * scale,
            ),
            Valley(
                start_xy=np.array([-360.0, 120.0], dtype=float) * scale,
                end_xy=np.array([300.0, -240.0], dtype=float) * scale,
                depth_m=20.0,
                width_m=78.0 * scale,
            ),
        ),
    )


def lake_district_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=12.0,
        slope_x_m_per_m=-0.002,
        slope_y_m_per_m=0.004,
        wave_amplitude_m=6.0,
        wave_length_x_m=500.0 * scale,
        wave_length_y_m=460.0 * scale,
        ridge_amplitude_m=8.0,
        ridge_center_x_m=180.0 * scale,
        ridge_center_y_m=60.0 * scale,
        ridge_radius_m=240.0 * scale,
        basin_depth_m=0.0,
        features=(
            Plateau(
                center_xy=np.array([-260.0, 190.0], dtype=float) * scale,
                radius_m=130.0 * scale,
                elevation_m=20.0,
                edge_sharpness=7.0,
            ),
            Plateau(
                center_xy=np.array([260.0, -180.0], dtype=float) * scale,
                radius_m=118.0 * scale,
                elevation_m=16.0,
                edge_sharpness=8.0,
            ),
            Valley(
                start_xy=np.array([-420.0, -80.0], dtype=float) * scale,
                end_xy=np.array([420.0, -20.0], dtype=float) * scale,
                depth_m=18.0,
                width_m=92.0 * scale,
            ),
            Valley(
                start_xy=np.array([-340.0, 140.0], dtype=float) * scale,
                end_xy=np.array([280.0, 220.0], dtype=float) * scale,
                depth_m=14.0,
                width_m=76.0 * scale,
            ),
        ),
    )


def jungle_canopy_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=14.0,
        slope_x_m_per_m=0.003,
        slope_y_m_per_m=-0.002,
        wave_amplitude_m=4.0,
        wave_length_x_m=320.0 * scale,
        wave_length_y_m=280.0 * scale,
        ridge_amplitude_m=6.0,
        ridge_center_x_m=80.0 * scale,
        ridge_center_y_m=-60.0 * scale,
        ridge_radius_m=200.0 * scale,
        basin_depth_m=8.0,
        basin_center_x_m=-180.0 * scale,
        basin_center_y_m=120.0 * scale,
        basin_radius_m=180.0 * scale,
        features=(
            Plateau(
                center_xy=np.array([0.0, 0.0], dtype=float) * scale,
                radius_m=220.0 * scale,
                elevation_m=10.0,
                edge_sharpness=4.0,
            ),
            Plateau(
                center_xy=np.array([-280.0, -140.0], dtype=float) * scale,
                radius_m=160.0 * scale,
                elevation_m=8.0,
                edge_sharpness=5.0,
            ),
            Valley(
                start_xy=np.array([-380.0, 80.0], dtype=float) * scale,
                end_xy=np.array([360.0, -60.0], dtype=float) * scale,
                depth_m=12.0,
                width_m=65.0 * scale,
            ),
        ),
    )


def arctic_tundra_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=4.0,
        slope_x_m_per_m=0.001,
        slope_y_m_per_m=0.001,
        wave_amplitude_m=2.0,
        wave_length_x_m=600.0 * scale,
        wave_length_y_m=550.0 * scale,
        ridge_amplitude_m=0.0,
        basin_depth_m=6.0,
        basin_center_x_m=0.0,
        basin_center_y_m=0.0,
        basin_radius_m=300.0 * scale,
        features=(
            RidgeLine(
                start_xy=np.array([-500.0, -200.0], dtype=float) * scale,
                end_xy=np.array([500.0, -120.0], dtype=float) * scale,
                peak_elevation_m=14.0,
                width_m=110.0 * scale,
            ),
            Valley(
                start_xy=np.array([-300.0, 200.0], dtype=float) * scale,
                end_xy=np.array([350.0, 250.0], dtype=float) * scale,
                depth_m=8.0,
                width_m=140.0 * scale,
            ),
        ),
    )


def military_compound_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=8.0,
        slope_x_m_per_m=0.002,
        slope_y_m_per_m=-0.001,
        wave_amplitude_m=1.5,
        wave_length_x_m=480.0 * scale,
        wave_length_y_m=440.0 * scale,
        ridge_amplitude_m=0.0,
        basin_depth_m=0.0,
        features=(
            Plateau(
                center_xy=np.array([0.0, 0.0], dtype=float) * scale,
                radius_m=180.0 * scale,
                elevation_m=4.0,
                edge_sharpness=10.0,
            ),
            Plateau(
                center_xy=np.array([300.0, 200.0], dtype=float) * scale,
                radius_m=100.0 * scale,
                elevation_m=6.0,
                edge_sharpness=12.0,
            ),
            Plateau(
                center_xy=np.array([-250.0, -180.0], dtype=float) * scale,
                radius_m=120.0 * scale,
                elevation_m=5.0,
                edge_sharpness=10.0,
            ),
        ),
    )


def river_valley_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=20.0,
        slope_x_m_per_m=0.003,
        slope_y_m_per_m=-0.005,
        wave_amplitude_m=6.0,
        wave_length_x_m=350.0 * scale,
        wave_length_y_m=300.0 * scale,
        ridge_amplitude_m=18.0,
        ridge_center_x_m=-200.0 * scale,
        ridge_center_y_m=0.0,
        ridge_radius_m=200.0 * scale,
        basin_depth_m=0.0,
        features=(
            Valley(
                start_xy=np.array([-500.0, 0.0], dtype=float) * scale,
                end_xy=np.array([500.0, 20.0], dtype=float) * scale,
                depth_m=32.0,
                width_m=80.0 * scale,
            ),
            MountainRange(
                start_xy=np.array([-400.0, -200.0], dtype=float) * scale,
                end_xy=np.array([400.0, -160.0], dtype=float) * scale,
                peak_elevation_m=55.0,
                width_m=90.0 * scale,
                peak_count=4,
                roughness=0.7,
            ),
            MountainRange(
                start_xy=np.array([-400.0, 200.0], dtype=float) * scale,
                end_xy=np.array([400.0, 180.0], dtype=float) * scale,
                peak_elevation_m=48.0,
                width_m=85.0 * scale,
                peak_count=3,
                roughness=0.6,
            ),
            RidgeLine(
                start_xy=np.array([300.0, -120.0], dtype=float) * scale,
                end_xy=np.array([400.0, 120.0], dtype=float) * scale,
                peak_elevation_m=30.0,
                width_m=60.0 * scale,
            ),
        ),
    )


def mountain_pass_terrain(scale: float) -> TerrainModel:
    return TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=80.0,
        slope_x_m_per_m=0.008,
        slope_y_m_per_m=-0.004,
        wave_amplitude_m=25.0,
        wave_length_x_m=220.0 * scale,
        wave_length_y_m=190.0 * scale,
        ridge_amplitude_m=40.0,
        ridge_center_x_m=0.0,
        ridge_center_y_m=0.0,
        ridge_radius_m=200.0 * scale,
        basin_depth_m=20.0,
        basin_center_x_m=100.0 * scale,
        basin_center_y_m=-50.0 * scale,
        basin_radius_m=160.0 * scale,
        features=(
            # Northern mountain range
            MountainRange(
                start_xy=np.array([-380.0, 140.0], dtype=float) * scale,
                end_xy=np.array([350.0, 200.0], dtype=float) * scale,
                peak_elevation_m=550.0,
                width_m=100.0 * scale,
                peak_count=6,
                roughness=0.85,
            ),
            # Southern mountain range
            MountainRange(
                start_xy=np.array([-350.0, -200.0], dtype=float) * scale,
                end_xy=np.array([380.0, -140.0], dtype=float) * scale,
                peak_elevation_m=550.0,
                width_m=100.0 * scale,
                peak_count=6,
                roughness=0.80,
            ),
            # Central valley (the pass)
            Valley(
                start_xy=np.array([-400.0, 0.0], dtype=float) * scale,
                end_xy=np.array([400.0, 0.0], dtype=float) * scale,
                depth_m=100.0,
                width_m=70.0 * scale,
            ),
            # River snaking through the valley
            River(
                control_points=(
                    np.array([-380.0, 10.0], dtype=float) * scale,
                    np.array([-180.0, -25.0], dtype=float) * scale,
                    np.array([0.0, 20.0], dtype=float) * scale,
                    np.array([200.0, -15.0], dtype=float) * scale,
                    np.array([380.0, 5.0], dtype=float) * scale,
                ),
                width_m=35.0 * scale,
                depth_m=18.0,
                bank_slope=4.0,
            ),
            # Saddle ridge crossing the pass
            RidgeLine(
                start_xy=np.array([-40.0, -120.0], dtype=float) * scale,
                end_xy=np.array([40.0, 120.0], dtype=float) * scale,
                peak_elevation_m=60.0,
                width_m=50.0 * scale,
            ),
            # Fractal roughness
            NoiseLayer(seed=202, amplitude_m=40.0, base_wavelength_m=60.0 * scale),
        ),
    )


TERRAIN_PRESET_BUILDERS = {
    "alpine": alpine_terrain,
    "coastal": coastal_terrain,
    "urban_flat": urban_flat_terrain,
    "desert_canyon": desert_canyon_terrain,
    "rolling_highlands": rolling_highlands_terrain,
    "lake_district": lake_district_terrain,
    "jungle_canopy": jungle_canopy_terrain,
    "arctic_tundra": arctic_tundra_terrain,
    "military_compound": military_compound_terrain,
    "river_valley": river_valley_terrain,
    "mountain_pass": mountain_pass_terrain,
}
KNOWN_TERRAIN_PRESETS = frozenset(TERRAIN_PRESET_BUILDERS)


def terrain_model_from_preset(preset_name: str, scale: float) -> TerrainModel:
    try:
        builder = TERRAIN_PRESET_BUILDERS[preset_name]
    except KeyError as error:
        raise ValueError(f"Unknown terrain preset {preset_name!r}.") from error
    return builder(scale)


@dataclass(frozen=True)
class OccludingObject:
    object_id: str
    center_x_m: float
    center_y_m: float
    radius_m: float
    height_agl_m: float

    def base_elevation_m(self, terrain: TerrainModel) -> float:
        return terrain.height_at(self.center_x_m, self.center_y_m)

    def top_elevation_m(self, terrain: TerrainModel) -> float:
        return self.base_elevation_m(terrain) + max(self.height_agl_m, 0.0)

    def segment_intersects(self, origin: Sequence[float], target: Sequence[float], terrain: TerrainModel) -> bool:
        origin_xy = np.asarray(origin[:2], dtype=float)
        target_xy = np.asarray(target[:2], dtype=float)
        segment_xy = target_xy - origin_xy
        center_xy = np.array([self.center_x_m, self.center_y_m], dtype=float)
        relative_xy = origin_xy - center_xy
        radius_m = max(self.radius_m, 0.0)

        a = float(np.dot(segment_xy, segment_xy))
        if a <= 1.0e-4:
            if float(np.dot(relative_xy, relative_xy)) > radius_m * radius_m:
                return False
            base_z = self.base_elevation_m(terrain)
            top_z = self.top_elevation_m(terrain)
            low_z = min(float(origin[2]), float(target[2]))
            high_z = max(float(origin[2]), float(target[2]))
            return high_z >= base_z and low_z <= top_z

        b = 2.0 * float(np.dot(relative_xy, segment_xy))
        c = float(np.dot(relative_xy, relative_xy) - radius_m * radius_m)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return False

        sqrt_discriminant = float(np.sqrt(discriminant))
        t_entry = (-b - sqrt_discriminant) / (2.0 * a)
        t_exit = (-b + sqrt_discriminant) / (2.0 * a)
        segment_start = max(0.0, min(t_entry, t_exit))
        segment_end = min(1.0, max(t_entry, t_exit))
        if segment_start > segment_end:
            return False

        origin_z = float(origin[2])
        target_z = float(target[2])
        z_start = origin_z + (target_z - origin_z) * segment_start
        z_end = origin_z + (target_z - origin_z) * segment_end
        low_z = min(z_start, z_end)
        high_z = max(z_start, z_end)
        base_z = self.base_elevation_m(terrain)
        top_z = self.top_elevation_m(terrain)
        return high_z >= base_z and low_z <= top_z

    def to_metadata(self, terrain: TerrainModel) -> Dict[str, object]:
        base_elevation_m = self.base_elevation_m(terrain)
        return {
            "object_id": self.object_id,
            "kind": "cylinder-v1",
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "radius_m": self.radius_m,
            "height_agl_m": self.height_agl_m,
            "base_elevation_m": base_elevation_m,
            "top_elevation_m": base_elevation_m + max(self.height_agl_m, 0.0),
        }


def xy_bounds(points: Iterable[Sequence[float]], padding_m: float = 100.0) -> Dict[str, float]:
    collected = [np.asarray(point, dtype=float) for point in points]
    if not collected:
        return {"x_min_m": -300.0, "x_max_m": 300.0, "y_min_m": -300.0, "y_max_m": 300.0}

    values = np.vstack(collected)
    x_min = float(values[:, 0].min() - padding_m)
    x_max = float(values[:, 0].max() + padding_m)
    y_min = float(values[:, 1].min() - padding_m)
    y_max = float(values[:, 1].max() + padding_m)
    return {"x_min_m": x_min, "x_max_m": x_max, "y_min_m": y_min, "y_max_m": y_max}


__all__ = [
    "Bounds2D",
    "BuildingPrism",
    "CylinderObstacle",
    "EnvironmentCRS",
    "EnvironmentModel",
    "ForestStand",
    "KNOWN_TERRAIN_PRESETS",
    "LandCoverClass",
    "LandCoverLayer",
    "MountainRange",
    "NoiseLayer",
    "OccludingObject",
    "OrientedBox",
    "Plateau",
    "PolygonPrism",
    "RidgeLine",
    "River",
    "SensorVisibilityModel",
    "TerrainFeature",
    "TerrainModel",
    "Valley",
    "VisibilityResult",
    "WallSegment",
    "load_environment_bundle",
    "terrain_model_from_preset",
    "write_environment_bundle",
    "xy_bounds",
]
