"""GNSS/INS EKF fusion for ArgusNet.

Fuses GNSS position fixes with dead-reckoning odometry estimates using a
linear Kalman filter over the 6-D state [position, velocity].

This is a loosely-coupled architecture: GNSS provides position measurements;
odometry provides the prediction step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from argusnet.core.types import Vector3, vec3
from argusnet.sensing.gnss import GNSSMeasurement
from argusnet.localization.odometry import OdometryState

__all__ = [
    "GNSSFusionState",
    "GNSSINSFusion",
]


@dataclass
class GNSSFusionState:
    """Fused position/velocity estimate with full 6×6 covariance."""

    position: np.ndarray  # (3,) ENU metres
    velocity: np.ndarray  # (3,) m/s
    covariance: np.ndarray  # (6, 6)
    timestamp_s: float


class GNSSINSFusion:
    """Loosely-coupled EKF for GNSS + dead-reckoning fusion.

    State vector: x = [px, py, pz, vx, vy, vz]  (6-D)

    Prediction step uses constant-velocity model between updates.
    Measurement step uses GNSS position with diagonal noise covariance.
    """

    def __init__(
        self,
        initial_position: Vector3,
        initial_velocity: Vector3,
        initial_timestamp_s: float,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
    ) -> None:
        self._x = np.concatenate([
            np.asarray(initial_position, dtype=float),
            np.asarray(initial_velocity, dtype=float),
        ])
        # Initial covariance: moderate uncertainty
        self._P = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self._t = initial_timestamp_s

        self._q_pos = process_noise_pos
        self._q_vel = process_noise_vel
        self._last_innovation_mag: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Advance the state with a constant-velocity model over *dt* seconds."""
        if dt <= 0:
            return
        # Transition matrix F: position += velocity * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self._x = F @ self._x

        # Process noise (simple additive) with adaptive scaling from last innovation
        adaptive_scale = 1.0 + self._last_innovation_mag * 0.5
        Q = np.zeros((6, 6))
        q4 = self._q_pos * dt * dt
        q_v = self._q_vel * dt
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q4
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q_v
        Q *= adaptive_scale
        self._P = F @ self._P @ F.T + Q

    def update_gnss(self, measurement: GNSSMeasurement) -> None:
        """Incorporate a GNSS position fix."""
        dt = measurement.timestamp_s - self._t
        if dt > 0:
            self.predict(dt)
            self._t = measurement.timestamp_s

        # Measurement model: H maps state to position (3×6)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        h_sigma = max(measurement.h_sigma_m, 0.1)
        v_sigma = max(measurement.v_sigma_m, 0.1)
        R = np.diag([h_sigma ** 2, h_sigma ** 2, v_sigma ** 2])

        z = np.asarray(measurement.position, dtype=float)
        y = z - H @ self._x  # innovation
        self._last_innovation_mag = float(np.linalg.norm(y))
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ H) @ self._P

    def update_odometry(self, state: OdometryState) -> None:
        """Soft update using odometry velocity estimate as a measurement."""
        dt = state.timestamp_s - self._t
        if dt > 0:
            self.predict(dt)
            self._t = state.timestamp_s

        # Measurement: velocity (3×6)
        H = np.zeros((3, 6))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0

        R = state.velocity_cov if state.velocity_cov is not None else np.eye(3) * 0.25
        z = state.velocity
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ H) @ self._P

    @property
    def state(self) -> GNSSFusionState:
        return GNSSFusionState(
            position=self._x[:3].copy(),
            velocity=self._x[3:].copy(),
            covariance=self._P.copy(),
            timestamp_s=self._t,
        )
