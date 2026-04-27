"""GNSS position noise model for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3, vec3

__all__ = [
    "GNSSModel",
    "GNSSMeasurement",
    "sample_gnss_position",
]


@dataclass(frozen=True)
class GNSSModel:
    """GNSS receiver noise model based on DOP (dilution of precision).

    Horizontal and vertical 1-sigma errors are derived from a base CEP
    (circular error probable) value multiplied by HDOP and VDOP.
    """

    base_cep_m: float = 2.5
    """Base circular error probable at HDOP=1 (metres, 50th percentile)."""

    hdop: float = 1.0
    """Horizontal dilution of precision."""

    vdop: float = 1.5
    """Vertical dilution of precision."""

    # CEP → 1-sigma conversion factor for 2-D Gaussian: CEP ≈ 1.177 * σ
    _CEP_TO_SIGMA: float = 1.0 / 1.177

    @property
    def horizontal_sigma_m(self) -> float:
        """1-sigma horizontal position error (metres)."""
        return self.base_cep_m * self.hdop * self._CEP_TO_SIGMA

    @property
    def vertical_sigma_m(self) -> float:
        """1-sigma vertical position error (metres)."""
        return self.base_cep_m * self.vdop * self._CEP_TO_SIGMA

    def sample_error(self, rng: np.random.Generator | None = None) -> Vector3:
        """Sample a 3-D position error vector (dx, dy, dz) in metres."""
        if rng is None:
            rng = np.random.default_rng()
        h_sigma = self.horizontal_sigma_m
        v_sigma = self.vertical_sigma_m
        dx = float(rng.normal(0.0, h_sigma))
        dy = float(rng.normal(0.0, h_sigma))
        dz = float(rng.normal(0.0, v_sigma))
        return vec3(dx, dy, dz)


@dataclass(frozen=True)
class GNSSMeasurement:
    """A noisy GNSS position fix."""

    position: Vector3
    """Reported position in ENU metres."""

    timestamp_s: float
    """Time of fix (simulation seconds)."""

    h_sigma_m: float
    """1-sigma horizontal accuracy estimate (metres)."""

    v_sigma_m: float
    """1-sigma vertical accuracy estimate (metres)."""


def sample_gnss_position(
    true_position: Vector3,
    model: GNSSModel,
    timestamp_s: float,
    rng: np.random.Generator | None = None,
) -> GNSSMeasurement:
    """Generate a noisy GNSS fix from a ground-truth position."""
    error = model.sample_error(rng)
    noisy_pos = np.asarray(true_position, dtype=float) + error
    return GNSSMeasurement(
        position=noisy_pos,
        timestamp_s=timestamp_s,
        h_sigma_m=model.horizontal_sigma_m,
        v_sigma_m=model.vertical_sigma_m,
    )
