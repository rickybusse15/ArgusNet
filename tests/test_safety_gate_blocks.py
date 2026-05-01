"""Verify the safety gate blocks hard violations and lets soft violations pass.

These tests exercise the gate logic directly (without spinning up a full sim).
The gate is the closure ``_validate_command`` in ``run_simulation`` that
combines DroneConstraintChecker output with a hard/soft severity policy.
The same combination is reproduced here against the same checker.
"""

from __future__ import annotations

import numpy as np

from argusnet.mission.execution import (
    DroneRuntimeState,
    ExecutableCommand,
    MissionTaskType,
    SafetyDecision,
    SafetyStatus,
)
from argusnet.safety.checker import DroneConstraintChecker, DronePhysicalLimits


def _gate(blocking: bool):
    """Replicate sim.py's _validate_command closure for testing."""
    checker = DroneConstraintChecker(DronePhysicalLimits.tracker_default())

    def _validate(cmd: ExecutableCommand, st: DroneRuntimeState) -> SafetyDecision:
        violations = checker.check_state(
            position=np.asarray(st.position_m, dtype=float),
            velocity=np.asarray(st.velocity_mps, dtype=float),
            agl_m=st.agl_m,
            other_drone_positions=[np.asarray(p, dtype=float) for p in st.others_position_m],
        )
        if not violations or not blocking:
            return SafetyDecision(status=SafetyStatus.OK)
        hard = [v for v in violations if v.severity == "hard"]
        if not hard:
            return SafetyDecision(status=SafetyStatus.OK)
        return SafetyDecision(
            status=SafetyStatus.REJECTED,
            reason="; ".join(v.description for v in hard),
            violations=tuple(v.constraint for v in hard),
        )

    return _validate


def _cmd() -> ExecutableCommand:
    return ExecutableCommand(
        drone_id="drone-0",
        target_xy_m=(0.0, 0.0),
        target_z_m=100.0,
        task_type=MissionTaskType.MAP_FRONTIER,
    )


def test_low_agl_is_hard_blocked() -> None:
    gate = _gate(blocking=True)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 30.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=10.0,  # below DronePhysicalLimits.tracker_default().min_agl_m (50)
        battery_fraction=1.0,
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.REJECTED
    assert "min_agl" in decision.violations


def test_high_agl_is_hard_blocked() -> None:
    gate = _gate(blocking=True)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 800.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=800.0,  # above max_agl_m (600)
        battery_fraction=1.0,
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.REJECTED
    assert "max_agl" in decision.violations


def test_drone_separation_is_hard_blocked() -> None:
    gate = _gate(blocking=True)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=100.0,
        battery_fraction=1.0,
        others_position_m=((5.0, 0.0, 100.0),),  # 5m away, < min_drone_separation (12)
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.REJECTED
    assert "min_drone_separation" in decision.violations


def test_excess_speed_is_soft_only_logs() -> None:
    """Speed violations are SOFT — they log via the checker but do not reject."""
    gate = _gate(blocking=True)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(50.0, 0.0, 0.0),  # > max_speed_mps (35)
        agl_m=100.0,
        battery_fraction=1.0,
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.OK


def test_blocking_disabled_lets_hard_violations_through() -> None:
    """Phase A posture: even hard violations return OK when blocking is off."""
    gate = _gate(blocking=False)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 30.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=10.0,
        battery_fraction=1.0,
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.OK
