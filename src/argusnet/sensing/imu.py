"""IMU noise model and dead-reckoning integrator for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3

__all__ = [
    "IMUModel",
    "IMUMeasurement",
    "IMUIntegrator",
    "sample_imu",
]


@dataclass(frozen=True)
class IMUModel:
    """Stochastic IMU error model (gyro + accelerometer).

    All noise values are power spectral densities expressed in SI units
    per sqrt(Hz); multiply by sqrt(sample_rate_hz) to get per-sample sigma.
    """

    # --- Gyroscope ---
    gyro_noise_rad_per_s_sqrt_hz: float = 1e-3
    """Gyro white noise PSD (rad/s/√Hz)."""

    gyro_bias_rad_per_s: float = 5e-4
    """Constant gyro bias offset magnitude (rad/s)."""

    gyro_bias_drift_rad_per_s2_sqrt_hz: float = 1e-5
    """Gyro bias random walk PSD (rad/s²/√Hz)."""

    # --- Accelerometer ---
    accel_noise_m_per_s2_sqrt_hz: float = 5e-3
    """Accel white noise PSD (m/s²/√Hz)."""

    accel_bias_m_per_s2: float = 1e-2
    """Constant accel bias offset magnitude (m/s²)."""

    accel_bias_drift_m_per_s3_sqrt_hz: float = 1e-4
    """Accel bias random walk PSD (m/s³/√Hz)."""

    sample_rate_hz: float = 100.0
    """IMU sample rate (Hz)."""

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    def gyro_sigma(self) -> float:
        return self.gyro_noise_rad_per_s_sqrt_hz * np.sqrt(self.sample_rate_hz)

    def accel_sigma(self) -> float:
        return self.accel_noise_m_per_s2_sqrt_hz * np.sqrt(self.sample_rate_hz)


@dataclass
class IMUMeasurement:
    """Raw IMU measurement (gyro + accel)."""

    angular_velocity: Vector3
    """Measured angular velocity (rad/s) in body frame."""

    linear_acceleration: Vector3
    """Measured specific force (m/s²) in body frame."""

    timestamp_s: float


def sample_imu(
    true_accel: Vector3,
    true_angular_vel: Vector3,
    model: IMUModel,
    timestamp_s: float,
    rng: np.random.Generator | None = None,
) -> IMUMeasurement:
    """Generate a noisy IMU measurement from ground-truth kinematics."""
    if rng is None:
        rng = np.random.default_rng()
    gyro_sigma = model.gyro_sigma()
    accel_sigma = model.accel_sigma()

    noisy_gyro = (
        np.asarray(true_angular_vel, dtype=float)
        + model.gyro_bias_rad_per_s
        + rng.normal(0.0, gyro_sigma, 3)
    )
    noisy_accel = (
        np.asarray(true_accel, dtype=float)
        + model.accel_bias_m_per_s2
        + rng.normal(0.0, accel_sigma, 3)
    )
    return IMUMeasurement(
        angular_velocity=noisy_gyro,
        linear_acceleration=noisy_accel,
        timestamp_s=timestamp_s,
    )


class IMUIntegrator:
    """Simple dead-reckoning integrator from IMU measurements.

    Integrates accelerometer readings (in world frame) to maintain
    position and velocity estimates.  Assumes small rotation angles
    between samples and a flat world (no gravity compensation here —
    the caller must subtract gravity from accel before passing it).
    """

    def __init__(self, initial_position: Vector3, initial_velocity: Vector3) -> None:
        self.position = np.asarray(initial_position, dtype=float).copy()
        self.velocity = np.asarray(initial_velocity, dtype=float).copy()
        self._last_t: float | None = None

    def update(self, accel_world: Vector3, timestamp_s: float) -> None:
        """Integrate one accelerometer sample."""
        if self._last_t is None:
            self._last_t = timestamp_s
            return
        dt = timestamp_s - self._last_t
        if dt <= 0:
            return
        a = np.asarray(accel_world, dtype=float)
        self.position += self.velocity * dt + 0.5 * a * dt * dt
        self.velocity += a * dt
        self._last_t = timestamp_s

    @property
    def pose(self) -> tuple:
        return self.position.copy(), self.velocity.copy()
