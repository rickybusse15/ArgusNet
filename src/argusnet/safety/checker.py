"""Shadow safety constraint checker for ArgusNet simulation.

Mirrors the physical limits from rust/safety-engine/src/limits.rs in Python.
This is a NON-BLOCKING validator — it logs violations but does not stop motion
(Posture A from architectural decisions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["DronePhysicalLimits", "DroneConstraintChecker", "ConstraintViolation"]


@dataclass(frozen=True)
class DronePhysicalLimits:
    """Physical envelope for a drone platform (mirrors safety-engine Rust struct)."""

    max_speed_mps: float = 35.0
    min_speed_mps: float = 0.0
    max_bank_angle_rad: float = math.radians(25.0)
    max_yaw_rate_rad_s: float = 0.5
    max_climb_rate_mps: float = 8.0
    max_descent_rate_mps: float = 5.0
    max_horizontal_accel_mps2: float = 6.0
    max_vertical_accel_mps2: float = 4.0
    min_agl_m: float = 50.0
    max_agl_m: float = 600.0
    min_drone_separation_m: float = 12.0
    min_drone_vertical_separation_m: float = 8.0

    @classmethod
    def interceptor_default(cls) -> "DronePhysicalLimits":
        return cls(
            max_speed_mps=42.0,
            max_bank_angle_rad=math.radians(30.0),
            max_yaw_rate_rad_s=0.8,
            max_horizontal_accel_mps2=8.0,
            min_agl_m=30.0,
            max_agl_m=500.0,
            min_drone_separation_m=8.0,
            min_drone_vertical_separation_m=5.0,
        )

    @classmethod
    def tracker_default(cls) -> "DronePhysicalLimits":
        return cls()


@dataclass(frozen=True)
class ConstraintViolation:
    constraint: str
    commanded_value: float
    limit_value: float
    description: str


class DroneConstraintChecker:
    """Non-blocking shadow validator: logs violations without stopping motion."""

    def __init__(self, limits: Optional[DronePhysicalLimits] = None) -> None:
        self.limits = limits or DronePhysicalLimits.tracker_default()
        self._violations: List[ConstraintViolation] = []

    @property
    def violations(self) -> List[ConstraintViolation]:
        return list(self._violations)

    def clear(self) -> None:
        self._violations.clear()

    def check_state(
        self,
        position: np.ndarray,         # (3,) ENU
        velocity: np.ndarray,          # (3,) m/s
        agl_m: float,
        acceleration: Optional[np.ndarray] = None,  # (3,) m/s^2
        other_drone_positions: Optional[List[np.ndarray]] = None,
    ) -> List[ConstraintViolation]:
        """Validate a drone commanded state. Returns list of violations (non-blocking)."""
        violations = []
        speed = float(np.linalg.norm(velocity))

        # Speed check
        if speed > self.limits.max_speed_mps:
            violations.append(ConstraintViolation(
                constraint="max_speed",
                commanded_value=speed,
                limit_value=self.limits.max_speed_mps,
                description=f"Speed {speed:.1f} m/s exceeds limit {self.limits.max_speed_mps:.1f} m/s",
            ))

        # AGL floor
        if agl_m < self.limits.min_agl_m:
            violations.append(ConstraintViolation(
                constraint="min_agl",
                commanded_value=agl_m,
                limit_value=self.limits.min_agl_m,
                description=f"AGL {agl_m:.1f} m below floor {self.limits.min_agl_m:.1f} m",
            ))

        # AGL ceiling
        if agl_m > self.limits.max_agl_m:
            violations.append(ConstraintViolation(
                constraint="max_agl",
                commanded_value=agl_m,
                limit_value=self.limits.max_agl_m,
                description=f"AGL {agl_m:.1f} m above ceiling {self.limits.max_agl_m:.1f} m",
            ))

        # Vertical rate
        vz = float(velocity[2]) if len(velocity) > 2 else 0.0
        if vz > self.limits.max_climb_rate_mps:
            violations.append(ConstraintViolation(
                constraint="max_climb_rate",
                commanded_value=vz,
                limit_value=self.limits.max_climb_rate_mps,
                description=f"Climb rate {vz:.1f} m/s exceeds {self.limits.max_climb_rate_mps:.1f} m/s",
            ))
        if vz < -self.limits.max_descent_rate_mps:
            violations.append(ConstraintViolation(
                constraint="max_descent_rate",
                commanded_value=abs(vz),
                limit_value=self.limits.max_descent_rate_mps,
                description=f"Descent rate {abs(vz):.1f} m/s exceeds {self.limits.max_descent_rate_mps:.1f} m/s",
            ))

        # Acceleration
        if acceleration is not None:
            h_accel = float(np.linalg.norm(acceleration[:2]))
            v_accel = float(abs(acceleration[2])) if len(acceleration) > 2 else 0.0
            if h_accel > self.limits.max_horizontal_accel_mps2:
                violations.append(ConstraintViolation(
                    constraint="max_horizontal_accel",
                    commanded_value=h_accel,
                    limit_value=self.limits.max_horizontal_accel_mps2,
                    description=f"H accel {h_accel:.1f} m/s² exceeds {self.limits.max_horizontal_accel_mps2:.1f} m/s²",
                ))

        # Drone separation
        if other_drone_positions:
            pos = np.asarray(position)
            for other in other_drone_positions:
                other_arr = np.asarray(other)
                h_dist = float(np.linalg.norm(pos[:2] - other_arr[:2]))
                v_dist = float(abs(pos[2] - other_arr[2])) if len(pos) > 2 and len(other_arr) > 2 else 0.0
                if h_dist < self.limits.min_drone_separation_m and h_dist > 0.1:
                    violations.append(ConstraintViolation(
                        constraint="min_drone_separation",
                        commanded_value=h_dist,
                        limit_value=self.limits.min_drone_separation_m,
                        description=f"Drone separation {h_dist:.1f} m below min {self.limits.min_drone_separation_m:.1f} m",
                    ))

        self._violations.extend(violations)
        return violations
