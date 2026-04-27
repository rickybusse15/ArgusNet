"""LiDAR / ToF depth sensor model for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3

__all__ = [
    "DepthModel",
    "DepthMeasurement",
    "sample_depth",
]


@dataclass(frozen=True)
class DepthModel:
    """Range noise model for a LiDAR or time-of-flight depth sensor.

    Noise grows with range according to a polynomial:
        sigma(R) = sigma_0 + sigma_slope * R

    Beam divergence causes a footprint at range R:
        footprint_radius_m = R * tan(beam_divergence_half_angle_rad)
    """

    sigma_0_m: float = 0.02
    """Range noise standard deviation at zero range (metres)."""

    sigma_slope: float = 0.001
    """Additional noise per metre of range (m/m)."""

    max_range_m: float = 100.0
    """Maximum measurable range (metres). Returns None beyond this."""

    min_range_m: float = 0.1
    """Minimum measurable range (metres). Returns None below this."""

    beam_divergence_half_angle_rad: float = 0.0005
    """Half-angle of the beam divergence cone (radians)."""

    def range_sigma(self, range_m: float) -> float:
        """1-sigma range noise at a given range."""
        return self.sigma_0_m + self.sigma_slope * range_m

    def footprint_radius(self, range_m: float) -> float:
        """Beam footprint radius at *range_m* (metres)."""
        return range_m * np.tan(self.beam_divergence_half_angle_rad)

    def in_range(self, range_m: float) -> bool:
        return self.min_range_m <= range_m <= self.max_range_m


@dataclass(frozen=True)
class DepthMeasurement:
    """A single noisy depth/range measurement."""

    range_m: float
    """Measured range (metres)."""

    sigma_m: float
    """1-sigma range uncertainty estimate (metres)."""

    direction: Vector3
    """Unit vector in the measurement direction (sensor frame)."""

    timestamp_s: float


def sample_depth(
    true_range_m: float,
    direction: Vector3,
    model: DepthModel,
    timestamp_s: float,
    rng: np.random.Generator | None = None,
) -> DepthMeasurement | None:
    """Generate a noisy depth measurement from ground-truth range.

    Returns None if the true range is outside [min_range_m, max_range_m].
    """
    if not model.in_range(true_range_m):
        return None
    if rng is None:
        rng = np.random.default_rng()
    sigma = model.range_sigma(true_range_m)
    noisy_range = float(true_range_m + rng.normal(0.0, sigma))
    noisy_range = max(model.min_range_m, noisy_range)
    return DepthMeasurement(
        range_m=noisy_range,
        sigma_m=sigma,
        direction=direction,
        timestamp_s=timestamp_s,
    )
