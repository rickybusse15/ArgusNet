"""Dead-reckoning odometry integrator for ArgusNet.

Maintains a position and velocity estimate by integrating noisy IMU
accelerometer readings.  Suitable as a short-horizon prediction prior
before a GNSS update corrects accumulated drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3
from argusnet.sensing.imu import IMUMeasurement

__all__ = [
    "OdometryState",
    "DeadReckoningOdometry",
]

_GRAVITY = np.array([0.0, 0.0, -9.80665])  # m/s² in ENU (pointing down)


@dataclass
class OdometryState:
    """Current dead-reckoning pose estimate."""

    position: np.ndarray
    """ENU position estimate (metres)."""

    velocity: np.ndarray
    """ENU velocity estimate (m/s)."""

    timestamp_s: float

    position_cov: np.ndarray = None  # 3×3 covariance, or None
    velocity_cov: np.ndarray = None  # 3×3 covariance, or None

    def __post_init__(self) -> None:
        if self.position_cov is None:
            object.__setattr__(self, "position_cov", np.zeros((3, 3)))
        if self.velocity_cov is None:
            object.__setattr__(self, "velocity_cov", np.zeros((3, 3)))


class DeadReckoningOdometry:
    """Simple constant-acceleration dead-reckoning integrator.

    Gravity is subtracted from accelerometer readings before integration.
    The body-to-world rotation must be supplied externally (e.g. from a
    magnetometer or attitude estimator).  If not supplied, identity rotation
    is assumed (body frame = world frame, suitable when the drone is
    near-level).
    """

    def __init__(
        self,
        initial_position: Vector3,
        initial_velocity: Vector3,
        initial_timestamp_s: float,
        accel_noise_sigma_m_per_s2: float = 0.05,
    ) -> None:
        self._pos = np.asarray(initial_position, dtype=float).copy()
        self._vel = np.asarray(initial_velocity, dtype=float).copy()
        self._t = initial_timestamp_s
        self._accel_sigma = accel_noise_sigma_m_per_s2

        # Simple diagonal covariance; grows with integration time
        self._pos_cov = np.diag([0.01, 0.01, 0.01])
        self._vel_cov = np.diag([0.001, 0.001, 0.001])

    def update(
        self,
        measurement: IMUMeasurement,
        body_to_world_R: np.ndarray | None = None,
    ) -> OdometryState:
        """Integrate one IMU measurement and return the new state."""
        dt = measurement.timestamp_s - self._t
        if dt <= 0.0:
            return self.state

        R = body_to_world_R if body_to_world_R is not None else np.eye(3)
        # Rotate accel to world frame and subtract gravity
        accel_world = R @ measurement.linear_acceleration - _GRAVITY

        # Kinematics integration (constant acceleration model)
        self._pos += self._vel * dt + 0.5 * accel_world * dt * dt
        self._vel += accel_world * dt
        self._t = measurement.timestamp_s

        # Grow covariance with time (simple linear growth heuristic)
        s2 = self._accel_sigma**2
        self._pos_cov += np.eye(3) * (s2 * dt * dt)
        self._vel_cov += np.eye(3) * (s2 * dt)

        return self.state

    def reset(
        self,
        position: Vector3,
        velocity: Vector3,
        timestamp_s: float,
        pos_cov: np.ndarray | None = None,
        vel_cov: np.ndarray | None = None,
    ) -> None:
        """Reset state, typically called after a GNSS correction."""
        self._pos = np.asarray(position, dtype=float).copy()
        self._vel = np.asarray(velocity, dtype=float).copy()
        self._t = timestamp_s
        self._pos_cov = pos_cov if pos_cov is not None else np.diag([0.01, 0.01, 0.01])
        self._vel_cov = vel_cov if vel_cov is not None else np.diag([0.001, 0.001, 0.001])

    @property
    def state(self) -> OdometryState:
        return OdometryState(
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            timestamp_s=self._t,
            position_cov=self._pos_cov.copy(),
            velocity_cov=self._vel_cov.copy(),
        )
