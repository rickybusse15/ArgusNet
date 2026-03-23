from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from argusnet.core.types import BearingObservation, TrackState


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


@dataclass(frozen=True)
class TriangulatedEstimate:
    position: np.ndarray
    measurement_std_m: float


def triangulate_bearings(observations: Iterable[BearingObservation]) -> np.ndarray:
    """Estimate the point closest to all observation rays."""
    lhs = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)

    used = 0
    for observation in observations:
        direction = _normalize(observation.direction)
        projector = np.eye(3) - np.outer(direction, direction)
        weight = max(observation.confidence, 0.05) / max(observation.bearing_std_rad**2, 1e-6)
        lhs += weight * projector
        rhs += weight * (projector @ observation.origin)
        used += 1

    if used < 2:
        raise ValueError("At least two observations are required for triangulation.")

    solution, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return solution


@dataclass
class KalmanTrack3D:
    timestamp_s: float
    state: np.ndarray
    covariance: np.ndarray
    process_accel_std: float = 3.0

    @classmethod
    def initialize(
        cls,
        timestamp_s: float,
        position: np.ndarray,
        position_std_m: float = 30.0,
        velocity_std_mps: float = 15.0,
    ) -> "KalmanTrack3D":
        state = np.zeros(6, dtype=float)
        state[:3] = position

        covariance = np.diag(
            [
                position_std_m**2,
                position_std_m**2,
                position_std_m**2,
                velocity_std_mps**2,
                velocity_std_mps**2,
                velocity_std_mps**2,
            ]
        )
        return cls(timestamp_s=timestamp_s, state=state, covariance=covariance)

    def predict(self, timestamp_s: float) -> None:
        dt = timestamp_s - self.timestamp_s
        if dt <= 0.0:
            return

        transition = np.eye(6, dtype=float)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt

        accel_var = self.process_accel_std**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        process_block = np.array(
            [
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ],
            dtype=float,
        ) * accel_var

        process_noise = np.zeros((6, 6), dtype=float)
        for axis in range(3):
            start = axis
            velocity = axis + 3
            process_noise[np.ix_([start, velocity], [start, velocity])] = process_block

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.timestamp_s = timestamp_s

    def update_position(self, position: np.ndarray, measurement_std_m: float) -> None:
        measurement = np.asarray(position, dtype=float)
        measurement_matrix = np.zeros((3, 6), dtype=float)
        measurement_matrix[0, 0] = 1.0
        measurement_matrix[1, 1] = 1.0
        measurement_matrix[2, 2] = 1.0

        measurement_cov = np.eye(3, dtype=float) * (measurement_std_m**2)
        innovation = measurement - (measurement_matrix @ self.state)
        innovation_cov = (
            measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_cov
        )
        kalman_gain = np.linalg.solve(innovation_cov.T, (self.covariance @ measurement_matrix.T).T).T

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(6, dtype=float)
        ikh = identity - kalman_gain @ measurement_matrix
        self.covariance = (
            ikh @ self.covariance @ ikh.T
            + kalman_gain @ measurement_cov @ kalman_gain.T
        )

    def snapshot(
        self,
        track_id: str,
        measurement_std_m: float,
        update_count: int,
        stale_steps: int,
    ) -> TrackState:
        return TrackState(
            track_id=track_id,
            timestamp_s=self.timestamp_s,
            position=self.state[:3].copy(),
            velocity=self.state[3:].copy(),
            covariance=self.covariance.copy(),
            measurement_std_m=measurement_std_m,
            update_count=update_count,
            stale_steps=stale_steps,
        )


def infer_measurement_std(observations: Iterable[BearingObservation], estimate: np.ndarray) -> float:
    observations = list(observations)
    if not observations:
        raise ValueError("At least one observation is required.")

    per_node_std = []
    for observation in observations:
        range_m = np.linalg.norm(estimate - observation.origin)
        per_node_std.append(max(range_m * observation.bearing_std_rad, 1.0))

    return float(np.mean(per_node_std))


def fuse_bearing_cluster(observations: Iterable[BearingObservation]) -> TriangulatedEstimate:
    observations = list(observations)
    if len(observations) < 2:
        raise ValueError("At least two observations are required for fusion.")

    triangulated_position = triangulate_bearings(observations)
    measurement_std_m = infer_measurement_std(observations, triangulated_position)
    return TriangulatedEstimate(
        position=triangulated_position,
        measurement_std_m=measurement_std_m,
    )


# ---------------------------------------------------------------------------
# Adaptive Kalman filter configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveFilterConfig:
    """Configuration for the adaptive / IMM Kalman filter."""

    # Constant-velocity model process noise (m/s²)
    cv_accel_std: float = 3.0

    # Coordinated-turn model process noise (m/s²)
    ct_accel_std: float = 8.0

    # Coordinated-turn model turn rate std (rad/s)
    ct_turn_rate_std: float = 0.1

    # Initial position uncertainty (m)
    init_position_std_m: float = 30.0

    # Initial velocity uncertainty (m/s)
    init_velocity_std_mps: float = 15.0

    # IMM model transition probabilities
    # cv_to_ct: probability of switching from CV to CT per step
    cv_to_ct_prob: float = 0.05
    # ct_to_cv: probability of switching from CT to CV per step
    ct_to_cv_prob: float = 0.10

    # Innovation-based adaptive Q scaling
    innovation_window: int = 5
    """Number of recent innovations to use for adaptive process noise."""

    innovation_scale_factor: float = 1.5
    """Scale Q when normalized innovation exceeds this threshold."""

    innovation_max_scale: float = 4.0
    """Maximum multiplier on process noise."""

    # Measurement noise adaptation
    adaptive_measurement_noise: bool = False
    """If True, scale measurement noise based on recent innovation consistency."""


# ---------------------------------------------------------------------------
# Coordinated-turn Kalman filter model
# ---------------------------------------------------------------------------

@dataclass
class CoordinatedTurnTrack3D:
    """7-state coordinated-turn model: [x, y, z, vx, vy, vz, omega].

    omega is the turn rate in the XY plane (rad/s).
    """
    timestamp_s: float
    state: np.ndarray       # [x, y, z, vx, vy, vz, omega]
    covariance: np.ndarray  # 7×7
    accel_std: float = 8.0
    turn_rate_std: float = 0.1

    @classmethod
    def initialize(
        cls,
        timestamp_s: float,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        position_std_m: float = 30.0,
        velocity_std_mps: float = 15.0,
        turn_rate_std: float = 0.1,
        accel_std: float = 8.0,
    ) -> "CoordinatedTurnTrack3D":
        state = np.zeros(7, dtype=float)
        state[:3] = position
        if velocity is not None:
            state[3:6] = velocity

        diag = [
            position_std_m**2,
            position_std_m**2,
            position_std_m**2,
            velocity_std_mps**2,
            velocity_std_mps**2,
            velocity_std_mps**2,
            turn_rate_std**2,
        ]
        covariance = np.diag(diag)
        return cls(
            timestamp_s=timestamp_s,
            state=state,
            covariance=covariance,
            accel_std=accel_std,
            turn_rate_std=turn_rate_std,
        )

    @classmethod
    def from_cv_track(
        cls,
        cv: KalmanTrack3D,
        turn_rate_std: float = 0.1,
        accel_std: float = 8.0,
    ) -> "CoordinatedTurnTrack3D":
        """Convert a 6-state CV track to a 7-state CT track."""
        state = np.zeros(7, dtype=float)
        state[:6] = cv.state
        state[6] = 0.0  # initial turn rate
        cov = np.zeros((7, 7), dtype=float)
        cov[:6, :6] = cv.covariance
        cov[6, 6] = turn_rate_std**2
        return cls(
            timestamp_s=cv.timestamp_s,
            state=state,
            covariance=cov,
            accel_std=accel_std,
            turn_rate_std=turn_rate_std,
        )

    def predict(self, timestamp_s: float) -> None:
        dt = timestamp_s - self.timestamp_s
        if dt <= 0.0:
            return

        omega = self.state[6]
        vx, vy = self.state[3], self.state[4]
        dt2 = dt * dt

        if abs(omega) < 1e-6:
            # Degenerate to constant-velocity in XY
            self.state[0] += vx * dt
            self.state[1] += vy * dt
        else:
            sin_wt = math.sin(omega * dt)
            cos_wt = math.cos(omega * dt)
            self.state[0] += (vx * sin_wt - vy * (1.0 - cos_wt)) / omega
            self.state[1] += (vx * (1.0 - cos_wt) + vy * sin_wt) / omega
            new_vx = vx * cos_wt - vy * sin_wt
            new_vy = vx * sin_wt + vy * cos_wt
            self.state[3] = new_vx
            self.state[4] = new_vy

        # Z is constant-velocity
        self.state[2] += self.state[5] * dt
        # omega stays constant (random walk in process noise)

        # Approximate Jacobian (linearized around current state)
        F = np.eye(7, dtype=float)
        if abs(omega) < 1e-6:
            F[0, 3] = dt
            F[1, 4] = dt
        else:
            sin_wt = math.sin(omega * dt)
            cos_wt = math.cos(omega * dt)
            F[0, 3] = sin_wt / omega
            F[0, 4] = -(1.0 - cos_wt) / omega
            F[1, 3] = (1.0 - cos_wt) / omega
            F[1, 4] = sin_wt / omega
            F[3, 3] = cos_wt
            F[3, 4] = -sin_wt
            F[4, 3] = sin_wt
            F[4, 4] = cos_wt
        F[2, 5] = dt

        # Process noise
        accel_var = self.accel_std**2
        Q = np.zeros((7, 7), dtype=float)
        # XY coupled via turn model
        for axis in range(3):
            p_idx = axis
            v_idx = axis + 3
            Q[p_idx, p_idx] = dt2 * dt2 / 4.0 * accel_var
            Q[p_idx, v_idx] = dt2 * dt / 2.0 * accel_var
            Q[v_idx, p_idx] = dt2 * dt / 2.0 * accel_var
            Q[v_idx, v_idx] = dt2 * accel_var
        Q[6, 6] = self.turn_rate_std**2 * dt

        self.covariance = F @ self.covariance @ F.T + Q
        self.timestamp_s = timestamp_s

    def update_position(self, position: np.ndarray, measurement_std_m: float) -> float:
        """Update with a 3D position measurement. Returns NIS (normalized innovation squared)."""
        measurement = np.asarray(position, dtype=float)
        H = np.zeros((3, 7), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.eye(3, dtype=float) * (measurement_std_m**2)
        innovation = measurement - (H @ self.state)
        S = H @ self.covariance @ H.T + R
        K = np.linalg.solve(S.T, (self.covariance @ H.T).T).T

        self.state = self.state + K @ innovation
        I_KH = np.eye(7, dtype=float) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

        # Normalized Innovation Squared (NIS)
        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            nis = float(np.sum(innovation**2))
        return nis

    def to_cv_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 6-state (pos+vel) and 6×6 covariance for output."""
        return self.state[:6].copy(), self.covariance[:6, :6].copy()

    def snapshot(
        self,
        track_id: str,
        measurement_std_m: float,
        update_count: int,
        stale_steps: int,
    ) -> TrackState:
        return TrackState(
            track_id=track_id,
            timestamp_s=self.timestamp_s,
            position=self.state[:3].copy(),
            velocity=self.state[3:6].copy(),
            covariance=self.covariance[:6, :6].copy(),
            measurement_std_m=measurement_std_m,
            update_count=update_count,
            stale_steps=stale_steps,
        )


# ---------------------------------------------------------------------------
# IMM (Interacting Multiple Model) filter
# ---------------------------------------------------------------------------

@dataclass
class IMMTrack3D:
    """IMM filter combining constant-velocity and coordinated-turn models.

    Automatically adapts between CV (straight flight) and CT (turning)
    based on observation consistency.
    """
    cv_track: KalmanTrack3D
    ct_track: CoordinatedTurnTrack3D
    mode_probabilities: np.ndarray   # [P(CV), P(CT)]
    config: AdaptiveFilterConfig
    _innovation_history: List[float] = field(default_factory=list, init=False, repr=False)
    _q_scale: float = field(default=1.0, init=False, repr=False)

    @classmethod
    def initialize(
        cls,
        timestamp_s: float,
        position: np.ndarray,
        config: Optional[AdaptiveFilterConfig] = None,
    ) -> "IMMTrack3D":
        if config is None:
            config = AdaptiveFilterConfig()

        cv = KalmanTrack3D.initialize(
            timestamp_s=timestamp_s,
            position=position,
            position_std_m=config.init_position_std_m,
            velocity_std_mps=config.init_velocity_std_mps,
        )
        cv.process_accel_std = config.cv_accel_std

        ct = CoordinatedTurnTrack3D.initialize(
            timestamp_s=timestamp_s,
            position=position,
            position_std_m=config.init_position_std_m,
            velocity_std_mps=config.init_velocity_std_mps,
            turn_rate_std=config.ct_turn_rate_std,
            accel_std=config.ct_accel_std,
        )

        return cls(
            cv_track=cv,
            ct_track=ct,
            mode_probabilities=np.array([0.8, 0.2]),
            config=config,
        )

    @property
    def timestamp_s(self) -> float:
        return self.cv_track.timestamp_s

    @property
    def state(self) -> np.ndarray:
        """Combined state estimate (6-state: pos + vel)."""
        p_cv, p_ct = self.mode_probabilities
        cv_state = self.cv_track.state
        ct_state = self.ct_track.state[:6]
        return p_cv * cv_state + p_ct * ct_state

    @property
    def covariance(self) -> np.ndarray:
        """Combined covariance (6×6)."""
        p_cv, p_ct = self.mode_probabilities
        cv_state = self.cv_track.state
        ct_state = self.ct_track.state[:6]
        combined_state = self.state

        cv_diff = cv_state - combined_state
        ct_diff = ct_state - combined_state

        return (
            p_cv * (self.cv_track.covariance + np.outer(cv_diff, cv_diff))
            + p_ct * (self.ct_track.covariance[:6, :6] + np.outer(ct_diff, ct_diff))
        )

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:]

    def predict(self, timestamp_s: float) -> None:
        """Predict both models forward, including mode mixing."""
        # Mode transition matrix
        T = np.array([
            [1.0 - self.config.cv_to_ct_prob, self.config.cv_to_ct_prob],
            [self.config.ct_to_cv_prob, 1.0 - self.config.ct_to_cv_prob],
        ])

        # Predicted mode probabilities
        c_bar = T.T @ self.mode_probabilities
        c_bar = np.maximum(c_bar, 1e-10)
        c_bar /= c_bar.sum()

        # Mixing probabilities
        mix = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                mix[i, j] = T[i, j] * self.mode_probabilities[i] / c_bar[j]

        # Mix states into each model
        cv_state_mixed = mix[0, 0] * self.cv_track.state + mix[1, 0] * self.ct_track.state[:6]
        ct_state_mixed = np.zeros(7)
        ct_state_mixed[:6] = mix[0, 1] * self.cv_track.state + mix[1, 1] * self.ct_track.state[:6]
        ct_state_mixed[6] = mix[1, 1] * self.ct_track.state[6]

        self.cv_track.state = cv_state_mixed
        self.ct_track.state = ct_state_mixed

        # Apply adaptive Q scaling
        original_cv_std = self.cv_track.process_accel_std
        original_ct_std = self.ct_track.accel_std
        self.cv_track.process_accel_std *= self._q_scale
        self.ct_track.accel_std *= self._q_scale

        # Predict each model
        self.cv_track.predict(timestamp_s)
        self.ct_track.predict(timestamp_s)

        # Restore original process noise settings
        self.cv_track.process_accel_std = original_cv_std
        self.ct_track.accel_std = original_ct_std

        self.mode_probabilities = c_bar

    def update_position(self, position: np.ndarray, measurement_std_m: float) -> None:
        """Update both models and recompute mode probabilities."""
        measurement = np.asarray(position, dtype=float)

        # --- CV model update ---
        H_cv = np.zeros((3, 6), dtype=float)
        np.fill_diagonal(H_cv[:3, :3], 1.0)
        R = np.eye(3, dtype=float) * (measurement_std_m**2)

        innov_cv = measurement - H_cv @ self.cv_track.state
        S_cv = H_cv @ self.cv_track.covariance @ H_cv.T + R

        try:
            cv_likelihood = _gaussian_likelihood(innov_cv, S_cv)
        except np.linalg.LinAlgError:
            cv_likelihood = 1e-30

        self.cv_track.update_position(position, measurement_std_m)

        # --- CT model update ---
        nis_ct = self.ct_track.update_position(position, measurement_std_m)

        H_ct = np.zeros((3, 7), dtype=float)
        np.fill_diagonal(H_ct[:3, :3], 1.0)
        innov_ct = measurement - H_ct @ self.ct_track.state
        # Recompute S for likelihood (pre-update S approximated)
        S_ct = H_ct @ self.ct_track.covariance @ H_ct.T + R
        try:
            ct_likelihood = _gaussian_likelihood(innov_ct, S_ct)
        except np.linalg.LinAlgError:
            ct_likelihood = 1e-30

        # Update mode probabilities
        c = np.array([
            self.mode_probabilities[0] * max(cv_likelihood, 1e-30),
            self.mode_probabilities[1] * max(ct_likelihood, 1e-30),
        ])
        total = c.sum()
        if total > 0:
            self.mode_probabilities = c / total
        else:
            self.mode_probabilities = np.array([0.5, 0.5])

        # Track innovation for adaptive Q
        combined_innov = measurement - self.state[:3]
        nis = float(np.sum(combined_innov**2) / max(measurement_std_m**2, 1e-6))
        self._innovation_history.append(nis)
        if len(self._innovation_history) > self.config.innovation_window:
            self._innovation_history = self._innovation_history[-self.config.innovation_window:]

        # Adaptive Q scaling
        if len(self._innovation_history) >= 3:
            mean_nis = np.mean(self._innovation_history)
            # For 3D position, expected NIS ≈ 3.0
            if mean_nis > self.config.innovation_scale_factor * 3.0:
                self._q_scale = min(
                    mean_nis / 3.0,
                    self.config.innovation_max_scale,
                )
            else:
                self._q_scale = max(1.0, self._q_scale * 0.9)

    def snapshot(
        self,
        track_id: str,
        measurement_std_m: float,
        update_count: int,
        stale_steps: int,
    ) -> TrackState:
        return TrackState(
            track_id=track_id,
            timestamp_s=self.timestamp_s,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            covariance=self.covariance.copy(),
            measurement_std_m=measurement_std_m,
            update_count=update_count,
            stale_steps=stale_steps,
        )


def _gaussian_likelihood(innovation: np.ndarray, S: np.ndarray) -> float:
    """Compute Gaussian likelihood of an innovation given covariance S."""
    n = len(innovation)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return 1e-30
    maha = float(innovation @ np.linalg.solve(S, innovation))
    log_likelihood = -0.5 * (n * math.log(2.0 * math.pi) + logdet + maha)
    return max(math.exp(log_likelihood), 1e-30)


# ---------------------------------------------------------------------------
# Track lifecycle management
# ---------------------------------------------------------------------------

TRACK_STATE_TENTATIVE = "tentative"
TRACK_STATE_CONFIRMED = "confirmed"
TRACK_STATE_COASTING = "coasting"
TRACK_STATE_DELETED = "deleted"


@dataclass(frozen=True)
class TrackLifecycleConfig:
    """Configuration for M-of-N track confirmation and lifecycle."""

    # M-of-N confirmation gate
    confirmation_m: int = 3
    """Number of updates required within confirmation_n frames."""

    confirmation_n: int = 5
    """Window size for M-of-N confirmation test."""

    # Coasting / deletion
    max_coast_frames: int = 10
    """Maximum frames without update before track is deleted."""

    max_coast_seconds: float = 5.0
    """Maximum time without update before track is deleted."""

    # Quality scoring
    min_quality_score: float = 0.1
    """Minimum quality score before track is deleted."""


@dataclass
class ManagedTrack:
    """A track with lifecycle management (tentative/confirmed/coasting/deleted)."""

    track_id: str
    filter: IMMTrack3D
    lifecycle_state: str = TRACK_STATE_TENTATIVE
    config: TrackLifecycleConfig = field(default_factory=TrackLifecycleConfig)
    update_count: int = 0
    stale_steps: int = 0
    last_update_time_s: float = 0.0
    measurement_std_m: float = 30.0
    _update_history: List[bool] = field(default_factory=list, init=False, repr=False)
    _quality_score: float = field(default=0.5, init=False, repr=False)

    @property
    def quality_score(self) -> float:
        return self._quality_score

    @property
    def is_alive(self) -> bool:
        return self.lifecycle_state != TRACK_STATE_DELETED

    def predict(self, timestamp_s: float) -> None:
        self.filter.predict(timestamp_s)

    def update(self, position: np.ndarray, measurement_std_m: float, timestamp_s: float) -> None:
        self.filter.update_position(position, measurement_std_m)
        self.update_count += 1
        self.stale_steps = 0
        self.last_update_time_s = timestamp_s
        self.measurement_std_m = measurement_std_m
        self._update_history.append(True)
        if len(self._update_history) > self.config.confirmation_n:
            self._update_history = self._update_history[-self.config.confirmation_n:]
        self._update_lifecycle()

    def mark_missed(self, timestamp_s: float) -> None:
        """Mark that no observation was associated this frame."""
        self.stale_steps += 1
        self._update_history.append(False)
        if len(self._update_history) > self.config.confirmation_n:
            self._update_history = self._update_history[-self.config.confirmation_n:]
        self._update_lifecycle()

        # Check coasting timeout
        time_since_update = timestamp_s - self.last_update_time_s
        if (
            self.stale_steps >= self.config.max_coast_frames
            or time_since_update >= self.config.max_coast_seconds
        ):
            self.lifecycle_state = TRACK_STATE_DELETED

    def _update_lifecycle(self) -> None:
        """Update lifecycle state based on update history."""
        if self.lifecycle_state == TRACK_STATE_DELETED:
            return

        recent_updates = sum(self._update_history[-self.config.confirmation_n:])

        # Quality score: fraction of recent frames with updates
        if self._update_history:
            self._quality_score = recent_updates / min(len(self._update_history), self.config.confirmation_n)
        else:
            self._quality_score = 0.5

        if self.lifecycle_state == TRACK_STATE_TENTATIVE:
            if recent_updates >= self.config.confirmation_m:
                self.lifecycle_state = TRACK_STATE_CONFIRMED
            elif len(self._update_history) >= self.config.confirmation_n and recent_updates < 2:
                self.lifecycle_state = TRACK_STATE_DELETED
        elif self.lifecycle_state == TRACK_STATE_CONFIRMED:
            if self.stale_steps > 0:
                self.lifecycle_state = TRACK_STATE_COASTING
        elif self.lifecycle_state == TRACK_STATE_COASTING:
            if self.stale_steps == 0:
                self.lifecycle_state = TRACK_STATE_CONFIRMED
            elif self._quality_score < self.config.min_quality_score:
                self.lifecycle_state = TRACK_STATE_DELETED

    def snapshot(self) -> TrackState:
        return self.filter.snapshot(
            track_id=self.track_id,
            measurement_std_m=self.measurement_std_m,
            update_count=self.update_count,
            stale_steps=self.stale_steps,
        )
