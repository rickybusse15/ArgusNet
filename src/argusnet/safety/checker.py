"""Drone safety validation and opt-in blocking command monitor.

The defaults mirror ``rust/safety-engine/src/limits.rs``.  The checker remains
useful as a shadow validator; :class:`SafetyMonitor` adds deterministic
escalation and best-effort command clamping for real-time callers.

Comms and gimbal constraints are intentionally not evaluated here because the
Python simulator does not yet provide those signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "ConstraintViolation",
    "DroneConstraintChecker",
    "DronePhysicalLimits",
    "DroneSafetyState",
    "SafetyDecision",
    "SafetyMonitor",
]


@dataclass(frozen=True)
class DronePhysicalLimits:
    """Physical envelope for a drone platform."""

    max_speed_mps: float = 35.0
    min_speed_mps: float = 0.0
    max_bank_angle_rad: float = 0.436
    max_yaw_rate_rad_s: float = 0.5
    max_climb_rate_mps: float = 6.0
    max_descent_rate_mps: float = 4.0
    max_horizontal_accel_mps2: float = 6.0
    max_vertical_accel_mps2: float = 3.0
    min_agl_m: float = 50.0
    max_agl_m: float = 600.0
    min_energy_reserve_fraction: float = 0.25
    min_drone_separation_m: float = 12.0
    min_drone_vertical_separation_m: float = 8.0

    @classmethod
    def interceptor_default(cls) -> DronePhysicalLimits:
        return cls(
            max_speed_mps=42.0,
            max_bank_angle_rad=0.524,
            max_yaw_rate_rad_s=0.8,
            max_climb_rate_mps=8.0,
            max_descent_rate_mps=5.0,
            max_horizontal_accel_mps2=8.0,
            max_vertical_accel_mps2=4.0,
            min_agl_m=30.0,
            max_agl_m=500.0,
            min_energy_reserve_fraction=0.20,
            min_drone_separation_m=8.0,
            min_drone_vertical_separation_m=5.0,
        )

    @classmethod
    def tracker_default(cls) -> DronePhysicalLimits:
        return cls()

    def min_turn_radius_at(self, speed_mps: float) -> float:
        bank = max(self.max_bank_angle_rad, math.radians(0.01))
        return speed_mps**2 / (9.80665 * math.tan(bank))


@dataclass(frozen=True)
class ConstraintViolation:
    constraint: str
    commanded_value: float
    limit_value: float
    description: str


class DroneConstraintChecker:
    """Collect every modeled violation for a commanded drone state."""

    def __init__(self, limits: DronePhysicalLimits | None = None) -> None:
        self.limits = limits or DronePhysicalLimits.tracker_default()
        self._violations: list[ConstraintViolation] = []

    @property
    def violations(self) -> list[ConstraintViolation]:
        return list(self._violations)

    def clear(self) -> None:
        self._violations.clear()

    def check_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        agl_m: float,
        acceleration: np.ndarray | None = None,
        other_drone_positions: list[np.ndarray] | None = None,
        energy_fraction: float | None = None,
        previous_velocity: np.ndarray | None = None,
        dt_s: float | None = None,
    ) -> list[ConstraintViolation]:
        """Validate state, returning all violations without changing the command."""
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        violations: list[ConstraintViolation] = []

        def add(name: str, value: float, limit: float, relation: str) -> None:
            violations.append(
                ConstraintViolation(
                    name,
                    value,
                    limit,
                    f"{name.replace('_', ' ')} {value:.2f} {relation} {limit:.2f}",
                )
            )

        speed = float(np.linalg.norm(velocity))
        if speed > self.limits.max_speed_mps:
            add("max_speed", speed, self.limits.max_speed_mps, "exceeds")
        if 0.0 < speed < self.limits.min_speed_mps:
            add("min_speed", speed, self.limits.min_speed_mps, "is below")
        if agl_m < self.limits.min_agl_m:
            add("min_agl", agl_m, self.limits.min_agl_m, "is below")
        if agl_m > self.limits.max_agl_m:
            add("max_agl", agl_m, self.limits.max_agl_m, "exceeds")

        vz = float(velocity[2])
        if vz > self.limits.max_climb_rate_mps:
            add("max_climb_rate", vz, self.limits.max_climb_rate_mps, "exceeds")
        if vz < -self.limits.max_descent_rate_mps:
            add("max_descent_rate", abs(vz), self.limits.max_descent_rate_mps, "exceeds")

        if acceleration is None and previous_velocity is not None and dt_s and dt_s > 0.0:
            acceleration = (velocity - np.asarray(previous_velocity, dtype=float)) / dt_s
        if acceleration is not None:
            acceleration = np.asarray(acceleration, dtype=float)
            horizontal = float(np.linalg.norm(acceleration[:2]))
            vertical = abs(float(acceleration[2]))
            if horizontal > self.limits.max_horizontal_accel_mps2:
                add(
                    "max_horizontal_accel",
                    horizontal,
                    self.limits.max_horizontal_accel_mps2,
                    "exceeds",
                )
            if vertical > self.limits.max_vertical_accel_mps2:
                add(
                    "max_vertical_accel",
                    vertical,
                    self.limits.max_vertical_accel_mps2,
                    "exceeds",
                )

        if (
            energy_fraction is not None
            and energy_fraction < self.limits.min_energy_reserve_fraction
        ):
            add(
                "min_energy_reserve",
                energy_fraction,
                self.limits.min_energy_reserve_fraction,
                "is below",
            )

        for other in other_drone_positions or []:
            delta = np.asarray(other, dtype=float) - np.asarray(position, dtype=float)
            horizontal = float(np.linalg.norm(delta[:2]))
            vertical = abs(float(delta[2]))
            # A conflict requires both horizontal and vertical minima to be breached.
            if horizontal < self.limits.min_drone_separation_m and vertical < (
                self.limits.min_drone_vertical_separation_m
            ):
                add(
                    "min_drone_separation",
                    horizontal,
                    self.limits.min_drone_separation_m,
                    "is below",
                )

        self._violations.extend(violations)
        return violations


class DroneSafetyState(str, Enum):
    NOMINAL = "nominal"
    CAUTION = "caution"
    WARNING = "warning"
    ABORT = "abort"


@dataclass(frozen=True)
class SafetyDecision:
    state: DroneSafetyState
    violations: tuple[ConstraintViolation, ...]
    position: np.ndarray
    velocity: np.ndarray

    @property
    def blocked(self) -> bool:
        return self.state is DroneSafetyState.ABORT


class SafetyMonitor:
    """Stateful blocking gate mirroring the Rust monitor's core escalation."""

    def __init__(
        self,
        limits: DronePhysicalLimits | None = None,
        *,
        abort_repeated_violation_frames: int = 3,
        caution_fraction: float = 0.9,
    ) -> None:
        self.limits = limits or DronePhysicalLimits.tracker_default()
        self.abort_repeated_violation_frames = abort_repeated_violation_frames
        self.caution_fraction = caution_fraction
        self._states: dict[str, DroneSafetyState] = {}
        self._consecutive: dict[str, int] = {}

    def state(self, drone_id: str) -> DroneSafetyState:
        return self._states.get(drone_id, DroneSafetyState.NOMINAL)

    def clear(self, drone_id: str) -> None:
        self._states.pop(drone_id, None)
        self._consecutive.pop(drone_id, None)

    def process(
        self,
        drone_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        terrain_height_m: float,
        acceleration: np.ndarray | None = None,
        previous_velocity: np.ndarray | None = None,
        dt_s: float | None = None,
        other_drone_positions: list[np.ndarray] | None = None,
        energy_fraction: float | None = None,
    ) -> SafetyDecision:
        position = np.asarray(position, dtype=float).copy()
        velocity = np.asarray(velocity, dtype=float).copy()
        agl_m = float(position[2] - terrain_height_m)
        violations = DroneConstraintChecker(self.limits).check_state(
            position,
            velocity,
            agl_m,
            acceleration,
            other_drone_positions,
            energy_fraction,
            previous_velocity,
            dt_s,
        )

        if violations:
            count = self._consecutive.get(drone_id, 0) + 1
            self._consecutive[drone_id] = count
            terrain_violation = any(v.constraint == "min_agl" for v in violations)
            state = (
                DroneSafetyState.ABORT
                if terrain_violation or count >= self.abort_repeated_violation_frames
                else DroneSafetyState.WARNING
            )
        else:
            self._consecutive[drone_id] = 0
            near_limit = (
                np.linalg.norm(velocity) >= self.limits.max_speed_mps * self.caution_fraction
                or agl_m <= self.limits.min_agl_m / self.caution_fraction
                or (
                    energy_fraction is not None
                    and energy_fraction
                    <= self.limits.min_energy_reserve_fraction / self.caution_fraction
                )
            )
            state = DroneSafetyState.CAUTION if near_limit else DroneSafetyState.NOMINAL

        # Clamp independently of escalation so Warning decisions are safe to apply.
        position[2] = float(
            np.clip(
                position[2],
                terrain_height_m + self.limits.min_agl_m,
                terrain_height_m + self.limits.max_agl_m,
            )
        )
        speed = float(np.linalg.norm(velocity))
        if speed > self.limits.max_speed_mps:
            velocity *= self.limits.max_speed_mps / speed
        velocity[2] = float(
            np.clip(
                velocity[2],
                -self.limits.max_descent_rate_mps,
                self.limits.max_climb_rate_mps,
            )
        )
        if state is DroneSafetyState.ABORT:
            velocity[:] = 0.0

        self._states[drone_id] = state
        return SafetyDecision(state, tuple(violations), position, velocity)
