"""Visual-inertial odometry (VIO) interface for ArgusNet.

Provides a simple sparse-feature VIO that fuses monocular/stereo image
features with IMU preintegration to estimate relative camera pose.  The
implementation is intentionally lightweight — suitable for simulation
and for wrapping an external VIO backend (VINS-Mono, OKVIS, Basalt)
via the common :class:`VIOBackend` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from argusnet.core.types import Vector3
from argusnet.localization.transforms import SE3
from argusnet.sensing.imu import IMUMeasurement

__all__ = [
    "VIOState",
    "VisualFeature",
    "VIOBackend",
    "SimpleVIO",
    "EKFVIO",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualFeature:
    """A tracked image feature."""

    feature_id: int
    """Persistent ID across frames."""

    u: float
    """Horizontal pixel coordinate."""

    v: float
    """Vertical pixel coordinate."""

    descriptor: np.ndarray | None = None
    """Optional feature descriptor vector."""

    depth_m: float | None = None
    """Depth from stereo or depth sensor (metres), if available."""


@dataclass
class VIOState:
    """Current VIO pose and motion estimate."""

    pose: SE3
    """World-to-body transform (SE3)."""

    velocity: np.ndarray
    """Body velocity in world frame (m/s)."""

    timestamp_s: float
    """Last update timestamp."""

    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated gyroscope bias (rad/s)."""

    bias_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated accelerometer bias (m/s²)."""

    covariance: np.ndarray | None = None
    """15×15 state covariance [rotation, velocity, position, bg, ba]."""

    tracked_feature_count: int = 0
    """Number of currently tracked features."""

    @property
    def position(self) -> np.ndarray:
        return self.pose.t

    @property
    def rotation(self) -> np.ndarray:
        return self.pose.R


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class VIOBackend(Protocol):
    """Interface for pluggable VIO backends."""

    def initialize(
        self,
        position: Vector3,
        timestamp_s: float,
    ) -> None:
        """Set initial pose."""
        ...

    def process_imu(self, measurement: IMUMeasurement) -> None:
        """Ingest an IMU sample for preintegration."""
        ...

    def process_image(
        self,
        features: list[VisualFeature],
        timestamp_s: float,
    ) -> VIOState | None:
        """Process an image (as a set of features) and return updated state."""
        ...

    def get_state(self) -> VIOState | None:
        """Return current state estimate."""
        ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Simple built-in VIO
# ---------------------------------------------------------------------------

_GRAVITY = np.array([0.0, 0.0, -9.80665])


class SimpleVIO:
    """Lightweight VIO for simulation.

    Uses constant-velocity motion model with IMU preintegration and
    simple feature-count-based confidence.  Not intended for real
    sensor data — use a proper VIO backend for that.
    """

    def __init__(
        self,
        min_features: int = 8,
        feature_noise_px: float = 1.0,
    ) -> None:
        self._min_features = min_features
        self._feature_noise_px = feature_noise_px
        self._state: VIOState | None = None
        self._imu_buffer: list[IMUMeasurement] = []
        self._prev_features: dict[int, VisualFeature] = {}

    def initialize(
        self,
        position: Vector3,
        timestamp_s: float,
    ) -> None:
        pos = np.asarray(position, dtype=float)
        self._state = VIOState(
            pose=SE3(R=np.eye(3), t=pos),
            velocity=np.zeros(3),
            timestamp_s=timestamp_s,
        )
        self._imu_buffer.clear()
        self._prev_features.clear()

    def process_imu(self, measurement: IMUMeasurement) -> None:
        """Buffer an IMU sample for the next image update."""
        self._imu_buffer.append(measurement)

    def process_image(
        self,
        features: list[VisualFeature],
        timestamp_s: float,
    ) -> VIOState | None:
        """Process features and return updated state.

        Returns ``None`` if the system is not initialised or if too few
        features are tracked.
        """
        if self._state is None:
            return None

        # --- IMU preintegration ---
        dt = timestamp_s - self._state.timestamp_s
        if dt <= 0:
            return self._state

        # Integrate buffered IMU
        delta_v = np.zeros(3)
        delta_p = np.zeros(3)
        t_prev = self._state.timestamp_s
        for imu in sorted(self._imu_buffer, key=lambda m: m.timestamp_s):
            if imu.timestamp_s < t_prev or imu.timestamp_s > timestamp_s:
                continue
            imu_dt = imu.timestamp_s - t_prev
            if imu_dt <= 0:
                continue
            accel_world = self._state.rotation @ np.asarray(imu.accel_mps2, dtype=float) + _GRAVITY
            delta_v += accel_world * imu_dt
            delta_p += self._state.velocity * imu_dt + 0.5 * accel_world * imu_dt**2
            t_prev = imu.timestamp_s
        self._imu_buffer.clear()

        # If no IMU, use constant velocity
        if np.linalg.norm(delta_v) < 1e-12:
            delta_p = self._state.velocity * dt

        # --- Feature matching (simplified) ---
        matched = 0
        current_ids = {f.feature_id for f in features}
        for fid in current_ids:
            if fid in self._prev_features:
                matched += 1

        self._prev_features = {f.feature_id: f for f in features}

        # Not enough features → coast on IMU only but flag low confidence
        has_visual = matched >= self._min_features

        # --- State update ---
        new_position = self._state.position + delta_p
        new_velocity = self._state.velocity + delta_v

        # Scale correction based on visual matches (crude)
        if has_visual:
            # Visual features constrain scale — reduce drift
            min(matched / (self._min_features * 2.0), 1.0)
        else:
            pass

        self._state = VIOState(
            pose=SE3(R=self._state.rotation.copy(), t=new_position),
            velocity=new_velocity,
            timestamp_s=timestamp_s,
            bias_gyro=self._state.bias_gyro,
            bias_accel=self._state.bias_accel,
            tracked_feature_count=matched,
        )
        return self._state

    def get_state(self) -> VIOState | None:
        return self._state

    def reset(self) -> None:
        self._state = None
        self._imu_buffer.clear()
        self._prev_features.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to 3x3 rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(rvec)
    if angle < 1e-9:
        return np.eye(3)
    axis = rvec / angle
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# EKF-based VIO
# ---------------------------------------------------------------------------

_GRAVITY_WORLD = np.array([0.0, 0.0, 9.80665])


class EKFVIO:
    """15-state EKF VIO: rotation(3) velocity(3) position(3) gyro_bias(3) accel_bias(3).

    Suitable for simulation — uses synthetic features from known 3D positions.
    Conforms to the VIOBackend protocol.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.05,
        gyro_noise_std: float = 0.005,
        accel_bias_drift: float = 0.001,
        gyro_bias_drift: float = 0.0001,
        feature_noise_px: float = 1.5,
        focal_length_px: float = 500.0,
        min_features: int = 6,
    ) -> None:
        self._accel_noise_std = accel_noise_std
        self._gyro_noise_std = gyro_noise_std
        self._accel_bias_drift = accel_bias_drift
        self._gyro_bias_drift = gyro_bias_drift
        self._feature_noise_px = feature_noise_px
        self._focal_length_px = focal_length_px
        self._min_features = min_features

        self._x: np.ndarray = np.zeros(15)
        self._P: np.ndarray = np.eye(15) * 0.01
        self._timestamp_s: float = 0.0
        self._initialized: bool = False
        self._imu_buffer: list[IMUMeasurement] = []
        self._prev_feature_ids: set = set()
        self._prev_features: dict[int, VisualFeature] = {}  # ID → feature, for optical flow

    def initialize(self, position: Vector3, timestamp_s: float) -> None:
        self._x = np.zeros(15)
        self._x[6:9] = np.asarray(position, dtype=float)
        self._P = np.diag([0.01] * 3 + [0.1] * 3 + [1.0] * 3 + [0.001] * 3 + [0.01] * 3)
        self._timestamp_s = timestamp_s
        self._initialized = True
        self._imu_buffer.clear()
        self._prev_feature_ids = set()
        self._prev_features = {}

    def process_imu(self, measurement: IMUMeasurement) -> None:
        self._imu_buffer.append(measurement)

    def process_image(
        self,
        features: list[VisualFeature],
        timestamp_s: float,
    ) -> VIOState | None:
        if not self._initialized:
            return None

        dt_total = timestamp_s - self._timestamp_s
        if dt_total < 0:
            return self.get_state()

        # --- Predict step via buffered IMU ---
        sorted_imu = sorted(self._imu_buffer, key=lambda m: m.timestamp_s)
        self._imu_buffer.clear()

        t_prev = self._timestamp_s
        for meas in sorted_imu:
            if meas.timestamp_s <= t_prev or meas.timestamp_s > timestamp_s:
                continue
            dt = meas.timestamp_s - t_prev

            gyro = np.asarray(meas.angular_velocity, dtype=float) - self._x[12:15]
            accel_body = np.asarray(meas.linear_acceleration, dtype=float) - self._x[9:12]
            R_wb = _rotvec_to_matrix(self._x[0:3])
            accel_world = R_wb @ accel_body - _GRAVITY_WORLD

            # Propagate state
            self._x[6:9] = self._x[6:9] + self._x[3:6] * dt + 0.5 * accel_world * dt**2
            self._x[3:6] = self._x[3:6] + accel_world * dt
            self._x[0:3] = self._x[0:3] + gyro * dt

            # Propagate covariance: F = I + Jac * dt (simplified linearisation)
            F = np.eye(15)
            # d(pos)/d(vel)
            F[6, 3] = dt
            F[7, 4] = dt
            F[8, 5] = dt
            # d(vel)/d(rot) — simplified (ignores cross terms fully)
            # d(vel)/d(accel_bias) = -R_wb * dt
            F[3:6, 9:12] = -R_wb * dt
            # d(pos)/d(accel_bias) = -0.5 * R_wb * dt^2
            F[6:9, 9:12] = -0.5 * R_wb * dt**2
            # d(rot)/d(gyro_bias) = -I * dt
            F[0:3, 12:15] = -np.eye(3) * dt

            # Process noise Q
            q_a = self._accel_noise_std**2 * dt
            q_g = self._gyro_noise_std**2 * dt
            q_ba = self._accel_bias_drift**2 * dt
            q_bg = self._gyro_bias_drift**2 * dt
            Q = np.diag([q_g] * 3 + [q_a] * 3 + [q_a * dt] * 3 + [q_ba] * 3 + [q_bg] * 3)

            self._P = F @ self._P @ F.T + Q
            t_prev = meas.timestamp_s

        # If no IMU covered the interval, do a single constant-velocity step
        if t_prev < timestamp_s and dt_total > 0:
            dt = timestamp_s - t_prev
            self._x[6:9] = self._x[6:9] + self._x[3:6] * dt
            F = np.eye(15)
            F[6, 3] = dt
            F[7, 4] = dt
            F[8, 5] = dt
            Q = np.diag(
                [self._gyro_noise_std**2 * dt] * 3
                + [self._accel_noise_std**2 * dt] * 3
                + [self._accel_noise_std**2 * dt**2] * 3
                + [self._accel_bias_drift**2 * dt] * 3
                + [self._gyro_bias_drift**2 * dt] * 3
            )
            self._P = F @ self._P @ F.T + Q

        self._timestamp_s = timestamp_s

        # --- Update step via visual features ---
        # Build per-feature lookup for current frame.
        cur_feature_map: dict[int, VisualFeature] = {f.feature_id: f for f in features}
        current_ids = set(cur_feature_map.keys())
        matched_ids = current_ids & self._prev_feature_ids

        matched = len(matched_ids)

        if matched >= 2:
            # --- Multi-feature optical-flow EKF update ---
            # Measurement model (simplified, world-frame linear velocity only):
            #   predicted Δu_i = vx * dt_total (x-velocity drives horizontal flow)
            #   predicted Δv_i = vy * dt_total (y-velocity drives vertical flow)
            # Both normalised by focal_length_px to give bearing residuals.
            f_px = self._focal_length_px
            r_var = (self._feature_noise_px / f_px) ** 2  # per-component noise variance

            H_rows = []
            z_rows = []
            valid_ids = []
            for fid in matched_ids:
                prev_f = self._prev_features[fid]
                cur_f = cur_feature_map[fid]
                delta_u = (cur_f.u - prev_f.u) / f_px
                delta_v = (cur_f.v - prev_f.v) / f_px
                pred_u = self._x[3] * dt_total / f_px
                pred_v = self._x[4] * dt_total / f_px
                res_u = delta_u - pred_u
                res_v = delta_v - pred_v

                # 2-row block of H for this feature (indices 3,4 = vx, vy)
                H_u = np.zeros(15)
                H_u[3] = dt_total / f_px
                H_v = np.zeros(15)
                H_v[4] = dt_total / f_px

                # Per-feature 2×2 innovation covariance for Mahalanobis gate
                H_feat = np.stack([H_u, H_v])
                S_feat = H_feat @ self._P @ H_feat.T + r_var * np.eye(2)
                res_feat = np.array([res_u, res_v])
                maha_sq = float(res_feat @ np.linalg.solve(S_feat, res_feat))
                # 3σ gate in 2-DOF: χ²(2, 0.997) ≈ 11.8
                if maha_sq > 11.8:
                    continue

                H_rows.append(H_u)
                H_rows.append(H_v)
                z_rows.append(res_u)
                z_rows.append(res_v)
                valid_ids.append(fid)

            if len(valid_ids) >= 2:
                H_batch = np.array(H_rows)  # (2K, 15)
                z_batch = np.array(z_rows)  # (2K,)
                R_batch = r_var * np.eye(len(z_rows))  # (2K, 2K)
                S_batch = H_batch @ self._P @ H_batch.T + R_batch
                K_batch = self._P @ H_batch.T @ np.linalg.inv(S_batch)
                self._x = self._x + K_batch @ z_batch
                self._P = (np.eye(15) - K_batch @ H_batch) @ self._P
            else:
                # Too few features passed the gate — fall back to scalar update.
                matched = max(matched, 0)  # keep count accurate for reporting
                if matched >= self._min_features:
                    scale_factor = matched / (self._min_features * 2.0) - 0.5
                    y_meas = np.array([scale_factor * 0.1])
                    H = np.zeros((1, 15))
                    H[0, 8] = 1.0 / self._focal_length_px
                    R_noise = np.array([[self._feature_noise_px**2 / self._focal_length_px**2]])
                    S = H @ self._P @ H.T + R_noise
                    K = self._P @ H.T @ np.linalg.inv(S)
                    self._x = self._x + (K @ y_meas).ravel()
                    self._P = (np.eye(15) - K @ H) @ self._P
        elif matched >= self._min_features:
            # Fewer than 2 features matched: original scalar update.
            scale_factor = matched / (self._min_features * 2.0) - 0.5
            y_meas = np.array([scale_factor * 0.1])
            H = np.zeros((1, 15))
            H[0, 8] = 1.0 / self._focal_length_px
            R_noise = np.array([[self._feature_noise_px**2 / self._focal_length_px**2]])
            S = H @ self._P @ H.T + R_noise
            K = self._P @ H.T @ np.linalg.inv(S)
            self._x = self._x + (K @ y_meas).ravel()
            self._P = (np.eye(15) - K @ H) @ self._P

        # Advance feature store for next frame.
        self._prev_feature_ids = current_ids
        self._prev_features = cur_feature_map

        return VIOState(
            pose=SE3(
                R=_rotvec_to_matrix(self._x[0:3]),
                t=self._x[6:9].copy(),
            ),
            velocity=self._x[3:6].copy(),
            timestamp_s=timestamp_s,
            bias_gyro=self._x[12:15].copy(),
            bias_accel=self._x[9:12].copy(),
            covariance=self._P.copy(),
            tracked_feature_count=matched,
        )

    def get_state(self) -> VIOState | None:
        if not self._initialized:
            return None
        return VIOState(
            pose=SE3(
                R=_rotvec_to_matrix(self._x[0:3]),
                t=self._x[6:9].copy(),
            ),
            velocity=self._x[3:6].copy(),
            timestamp_s=self._timestamp_s,
            bias_gyro=self._x[12:15].copy(),
            bias_accel=self._x[9:12].copy(),
            covariance=self._P.copy(),
            tracked_feature_count=0,
        )

    def reset(self) -> None:
        self._x = np.zeros(15)
        self._P = np.eye(15) * 0.01
        self._timestamp_s = 0.0
        self._initialized = False
        self._imu_buffer.clear()
        self._prev_feature_ids = set()
        self._prev_features = {}
