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


def _gate_with_target():
    """Replicate sim.py's _validate_command including the target-state check.

    The real closure consults ``scenario.terrain`` for the target's terrain
    height; here we assume flat ground (terrain_h = 0) so the target's z
    coordinate equals its AGL.
    """
    checker = DroneConstraintChecker(DronePhysicalLimits.tracker_default())

    def _validate(cmd: ExecutableCommand, st: DroneRuntimeState) -> SafetyDecision:
        others = [np.asarray(p, dtype=float) for p in st.others_position_m]
        cur = checker.check_state(
            position=np.asarray(st.position_m, dtype=float),
            velocity=np.asarray(st.velocity_mps, dtype=float),
            agl_m=st.agl_m,
            other_drone_positions=others,
        )
        target_xyz = np.array([cmd.target_xy_m[0], cmd.target_xy_m[1], cmd.target_z_m], dtype=float)
        tgt = checker.check_state(
            position=target_xyz,
            velocity=np.asarray(st.velocity_mps, dtype=float),
            agl_m=float(target_xyz[2]),  # flat ground assumption
            other_drone_positions=others,
        )
        seen: dict[str, object] = {}
        for v in list(cur) + list(tgt):
            existing = seen.get(v.constraint)
            if existing is None or (
                getattr(v, "severity", "soft") == "hard"
                and getattr(existing, "severity", "soft") != "hard"
            ):
                seen[v.constraint] = v
        violations = list(seen.values())
        hard = [v for v in violations if v.severity == "hard"]
        if not hard:
            return SafetyDecision(status=SafetyStatus.OK)
        return SafetyDecision(
            status=SafetyStatus.REJECTED,
            reason="; ".join(v.description for v in hard),
            violations=tuple(v.constraint for v in hard),
        )

    return _validate


def test_target_state_low_agl_is_blocked_when_current_is_legal() -> None:
    """A drone at a legal altitude commanded to a too-low target altitude
    must be rejected by the gate's target-state check (P1 fix)."""
    gate = _gate_with_target()
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=100.0,  # legal current altitude
        battery_fraction=1.0,
    )
    cmd = ExecutableCommand(
        drone_id="drone-0",
        target_xy_m=(50.0, 0.0),
        target_z_m=10.0,  # below min_agl_m=50
        task_type=MissionTaskType.INSPECT_TARGET,
    )
    decision = gate(cmd, state)
    assert decision.status == SafetyStatus.REJECTED
    assert "min_agl" in decision.violations


def test_target_state_high_agl_is_blocked_when_current_is_legal() -> None:
    gate = _gate_with_target()
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=100.0,
        battery_fraction=1.0,
    )
    cmd = ExecutableCommand(
        drone_id="drone-0",
        target_xy_m=(50.0, 0.0),
        target_z_m=900.0,  # above max_agl_m=600
        task_type=MissionTaskType.INSPECT_TARGET,
    )
    decision = gate(cmd, state)
    assert decision.status == SafetyStatus.REJECTED
    assert "max_agl" in decision.violations


def test_drone_state_for_populates_peer_positions() -> None:
    """Sanity: the safety gate sees other drones when DroneRuntimeState
    carries them (P1 fix). Without peers, separation can never trigger."""
    gate = _gate(blocking=True)
    state = DroneRuntimeState(
        drone_id="drone-0",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=100.0,
        battery_fraction=1.0,
        others_position_m=((6.0, 0.0, 100.0),),  # 6 m away → < 12 m gate
    )
    decision = gate(_cmd(), state)
    assert decision.status == SafetyStatus.REJECTED
    assert "min_drone_separation" in decision.violations
