"""Thermal (IR) camera sensor model for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.sensing.sensor_base import SensorBase

__all__ = ["ThermalCameraModel"]

# Radian equivalents of the degree-based FOV defaults (pre-computed as module
# constants for clarity; the dataclass fields store degrees so callers use
# the familiar unit).
_DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True)
class ThermalCameraModel(SensorBase):
    """Thermal infrared camera sensor model.

    Uses a sigmoid detection-probability curve driven by SNR, where SNR is
    computed from the thermal contrast divided by the NETD grown by
    Beer-Lambert atmospheric absorption.

    The FOV check is performed in 3-D: a bearing vector is inside the FOV
    when *both* its horizontal and vertical off-boresight angles are within
    the respective half-FOV limits.

    Parameters
    ----------
    sensor_id:
        Unique string identifier (default ``"thermal"``).
    netd_mk:
        Noise Equivalent Temperature Difference in milli-Kelvin.
    fov_h_deg:
        Horizontal full field-of-view in degrees.
    fov_v_deg:
        Vertical full field-of-view in degrees.
    max_range_m:
        Maximum operating range in metres.
    wavelength_band:
        ``"MWIR"`` (3–5 µm) or ``"LWIR"`` (8–12 µm).
    absorption_coeff_per_m:
        One-way atmospheric IR absorption coefficient (m⁻¹).
    """

    sensor_id: str = "thermal"
    netd_mk: float = 0.05
    fov_h_deg: float = 24.0
    fov_v_deg: float = 18.0
    max_range_m: float = 2000.0
    wavelength_band: str = "MWIR"
    absorption_coeff_per_m: float = 0.0008

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snr(self, range_m: float, thermal_contrast_k: float) -> float:
        """Signal-to-noise ratio for a target with given thermal contrast."""
        denominator = self.netd_mk * np.exp(self.absorption_coeff_per_m * range_m)
        # Guard against division by zero from extreme parameter combinations.
        if denominator <= 0.0:
            return 0.0
        return thermal_contrast_k / denominator

    def _fov_h_half_rad(self) -> float:
        return (self.fov_h_deg / 2.0) * _DEG_TO_RAD

    def _fov_v_half_rad(self) -> float:
        return (self.fov_v_deg / 2.0) * _DEG_TO_RAD

    # ------------------------------------------------------------------
    # SensorBase abstract method implementations
    # ------------------------------------------------------------------

    def detection_probability(  # type: ignore[override]
        self,
        range_m: float,
        thermal_contrast_k: float = 5.0,
    ) -> float:
        """Sigmoid detection probability driven by IR SNR.

        Args:
            range_m:            Slant range to target (metres).
            thermal_contrast_k: Target-background temperature difference (K).

        Returns:
            Detection probability in [0, 1].
        """
        if range_m > self.max_range_m or range_m < 0.0:
            return 0.0
        snr = self._snr(range_m, thermal_contrast_k)
        return float(1.0 / (1.0 + np.exp(-3.0 * (snr - 1.5))))

    def effective_noise_std(self, range_m: float) -> float:
        """Position-equivalent noise standard deviation (metres).

        Derived from NETD grown by Beer-Lambert absorption, scaled by a
        factor of 0.3 that converts temperature noise to range-equivalent
        position noise for a representative target geometry.

        Args:
            range_m: Slant range to target (metres).

        Returns:
            1-sigma noise in metres.
        """
        return float(self.netd_mk * np.exp(self.absorption_coeff_per_m * range_m) * 0.3)

    def in_fov(self, bearing_vec: np.ndarray) -> bool:
        """Check whether *bearing_vec* lies inside the rectangular FOV.

        The bearing vector is projected onto the boresight plane to extract
        independent horizontal and vertical off-axis angles, then compared
        against the half-FOV limits.

        The boresight is assumed to be along the +X axis; the horizontal
        plane is XY and the vertical plane is XZ.

        Args:
            bearing_vec: Unit vector from sensor to target, shape (3,).

        Returns:
            True if inside FOV, False otherwise.
        """
        bv = np.asarray(bearing_vec, dtype=float)
        norm = np.linalg.norm(bv)
        if norm < 1e-12:
            return False
        bv = bv / norm

        # Horizontal angle: angle between bearing projected onto XY plane
        # and the +X boresight direction.
        bv_h = np.array([bv[0], bv[1], 0.0])
        norm_h = np.linalg.norm(bv_h)
        if norm_h < 1e-12:
            # Bearing is straight up or down — inside horizontal FOV.
            h_angle = 0.0
        else:
            cos_h = np.clip(bv_h[0] / norm_h, -1.0, 1.0)
            h_angle = float(np.arccos(cos_h))

        # Vertical angle: elevation from the XY plane.
        np.clip(bv[0], -1.0, 1.0)
        # Use arcsin of the Z component relative to the bearing magnitude.
        v_angle = float(abs(np.arcsin(np.clip(bv[2], -1.0, 1.0))))

        return h_angle <= self._fov_h_half_rad() and v_angle <= self._fov_v_half_rad()

    def generate_observation(
        self,
        platform_pos: np.ndarray,
        target_pos: np.ndarray,
        timestamp_s: float,
    ) -> dict | None:
        """Generate a thermal observation of the target.

        Steps:
        1. Compute displacement vector, range, bearing, and elevation.
        2. Check FOV.
        3. Compute detection probability (with daytime 15 % penalty).
        4. Stochastic detection draw.
        5. If detected, build and return measurement dict.

        Args:
            platform_pos: Sensor world position (metres, shape (3,)).
            target_pos:   True target world position (metres, shape (3,)).
            timestamp_s:  Simulation time (seconds).

        Returns:
            Measurement dict on detection, ``None`` otherwise.
        """
        p = np.asarray(platform_pos, dtype=float)
        t = np.asarray(target_pos, dtype=float)

        delta = t - p
        range_m = float(np.linalg.norm(delta))

        if range_m < 1e-6 or range_m > self.max_range_m:
            return None

        bearing_unit = delta / range_m

        if not self.in_fov(bearing_unit):
            return None

        # Compute raw detection probability.
        pd = self.detection_probability(range_m)

        # Daytime penalty: thermal contrast is lower during daylight hours
        # because background temperature approaches target temperature.
        hour_of_day = (timestamp_s % 86400.0) / 3600.0
        if 6.0 < hour_of_day < 18.0:
            pd *= 0.85

        # Stochastic detection gate.
        if np.random.random() > pd:
            return None

        # Measurement angles (ENU convention: bearing from +X toward +Y,
        # elevation above the horizontal plane).
        bearing_rad = float(np.arctan2(delta[1], delta[0]))
        elevation_rad = float(np.arctan2(delta[2], np.hypot(delta[0], delta[1])))

        noise_std_m = self.effective_noise_std(range_m)

        return {
            "range_m": range_m,
            "bearing_rad": bearing_rad,
            "elevation_rad": elevation_rad,
            "noise_std_m": noise_std_m,
            "sensor_type": "thermal",
        }
