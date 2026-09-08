"""Bindings to safety-engine; physical defaults and decisions are owned by Rust."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from argusnet._native_world import NativeHandle, call

_DEFAULTS = call("limits", profile="tracker")


@dataclass(frozen=True)
class DronePhysicalLimits:
    """Physical envelope for a drone platform."""

    max_speed_mps: float = _DEFAULTS["max_speed_mps"]
    min_speed_mps: float = _DEFAULTS["min_speed_mps"]
    max_bank_angle_rad: float = _DEFAULTS["max_bank_angle_rad"]
    max_yaw_rate_rad_s: float = _DEFAULTS["max_yaw_rate_rad_s"]
    max_climb_rate_mps: float = _DEFAULTS["max_climb_rate_mps"]
    max_descent_rate_mps: float = _DEFAULTS["max_descent_rate_mps"]
    max_horizontal_accel_mps2: float = _DEFAULTS["max_horizontal_accel_mps2"]
    max_vertical_accel_mps2: float = _DEFAULTS["max_vertical_accel_mps2"]
    min_agl_m: float = _DEFAULTS["min_agl_m"]
    max_agl_m: float = _DEFAULTS["max_agl_m"]
    min_energy_reserve_fraction: float = _DEFAULTS["min_energy_reserve_fraction"]
    min_drone_separation_m: float = _DEFAULTS["min_drone_separation_m"]
    min_drone_vertical_separation_m: float = _DEFAULTS["min_drone_vertical_separation_m"]

    @classmethod
    def interceptor_default(cls) -> DronePhysicalLimits:
        values = call("limits", profile="interceptor")
        return cls(**{k: values[k] for k in cls.__dataclass_fields__})

    @classmethod
    def tracker_default(cls) -> DronePhysicalLimits:
        return cls()

    def min_turn_radius_at(self, speed_mps: float) -> float:
        return float(call("turn_radius", limits=asdict(self), speed=speed_mps))


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
        violations = [
            ConstraintViolation(**v)
            for v in call(
                "safety_check",
                limits=asdict(self.limits),
                state={
                    "position": position,
                    "velocity": velocity,
                    "terrain_height_m": float(position[2]) - agl_m,
                    "acceleration": acceleration,
                    "previous_velocity": previous_velocity,
                    "dt_s": dt_s,
                    "other_drone_positions": other_drone_positions or [],
                    "energy_fraction": energy_fraction,
                },
            )
        ]
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
    """Stateful binding to the simulation safety policy in safety-engine."""

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
        self._native = NativeHandle(
            "safety_new",
            limits=asdict(self.limits),
            abort_frames=abort_repeated_violation_frames,
            caution_fraction=caution_fraction,
        )

    def state(self, drone_id: str) -> DroneSafetyState:
        return DroneSafetyState(call("safety_state", handle=self._native.id, id=drone_id))

    def clear(self, drone_id: str) -> None:
        call("safety_clear", handle=self._native.id, id=drone_id)

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
        result = call(
            "safety_process",
            handle=self._native.id,
            id=drone_id,
            state={
                "position": position,
                "velocity": velocity,
                "terrain_height_m": terrain_height_m,
                "acceleration": acceleration,
                "previous_velocity": previous_velocity,
                "dt_s": dt_s,
                "other_drone_positions": other_drone_positions or [],
                "energy_fraction": energy_fraction,
            },
        )
        return SafetyDecision(
            DroneSafetyState(result["state"]),
            tuple(ConstraintViolation(**v) for v in result["violations"]),
            np.asarray(result["position"], dtype=float),
            np.asarray(result["velocity"], dtype=float),
        )
