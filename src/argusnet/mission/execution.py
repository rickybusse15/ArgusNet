from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple


# -----------------------------
# Core enums and data classes
# -----------------------------


class MissionStatus(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class MissionTaskType(str, Enum):
    MAP_FRONTIER = "map_frontier"
    INSPECT_TARGET = "inspect_target"
    TRACK_TARGET = "track_target"
    RELOCALIZE = "relocalize"
    RETURN_HOME = "return_home"
    HOLD = "hold"


class MissionTaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


class SafetyStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"


@dataclass
class MissionConstraints:
    geofence_radius_m: float
    battery_reserve_fraction: float = 0.2


@dataclass
class MissionTask:
    task_id: str
    task_type: MissionTaskType
    priority: int = 1
    status: MissionTaskStatus = MissionTaskStatus.PENDING
    reason: Optional[str] = None


@dataclass(frozen=True)
class ExecutableCommand:
    """A per-drone motion command produced by the executor's plan stage."""

    drone_id: str
    target_xy_m: Tuple[float, float]
    target_z_m: float
    task_type: MissionTaskType
    reason: str = ""


@dataclass(frozen=True)
class DroneRuntimeState:
    """Snapshot of drone state passed to the executor for one dispatch."""

    drone_id: str
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    agl_m: float
    battery_fraction: float
    others_position_m: Tuple[Tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class SafetyEvent:
    """Recorded safety-gate decision; emitted on rejection."""

    timestamp_s: float
    drone_id: str
    task_type: MissionTaskType
    target_xy_m: Tuple[float, float]
    target_z_m: float
    reason: str
    violations: Tuple[str, ...] = ()


@dataclass
class SafetyDecision:
    status: SafetyStatus
    reason: Optional[str] = None
    violations: Tuple[str, ...] = ()


@dataclass
class MissionState:
    mission_id: str
    status: MissionStatus = MissionStatus.CREATED
    tasks: List[MissionTask] = field(default_factory=list)
    active_task: Optional[MissionTask] = None
    safety_events: List[SafetyEvent] = field(default_factory=list)


# -----------------------------
# Execution context (dependency injection)
# -----------------------------


@dataclass
class MissionExecutionContext:
    get_localization_confidence: Callable[[], float]
    get_battery_fraction: Callable[[], float]
    validate_command: Callable[[ExecutableCommand, DroneRuntimeState], SafetyDecision]
    execute_command: Callable[[ExecutableCommand, DroneRuntimeState], None]
    is_task_complete: Callable[[], bool] = field(default_factory=lambda: lambda: True)
    now_s: Callable[[], float] = field(default_factory=lambda: lambda: 0.0)


# -----------------------------
# Mission executor
# -----------------------------


class MissionExecutor:
    """
    Closed-loop mission execution skeleton.

    The executor is the single choke-point for motion intent: every per-drone
    command in scan_map_inspect mode flows through ``dispatch()``, which runs
    the validate → execute pipeline and records safety events on rejection.

    ``step()`` advances mission-level lifecycle state (status transitions,
    high-level task selection). It does not by itself produce commands; the
    sim drives per-drone motion through ``dispatch()`` each tick.
    """

    HOLD_PRIORITY = 90

    def __init__(self, state: MissionState, constraints: MissionConstraints, ctx: MissionExecutionContext):
        self.state = state
        self.constraints = constraints
        self.ctx = ctx

    # -------------------------
    # Per-drone command dispatch
    # -------------------------

    def dispatch(
        self, command: ExecutableCommand, drone_state: DroneRuntimeState
    ) -> SafetyDecision:
        """Validate and (if approved) execute one drone command.

        Rejection records a ``SafetyEvent`` on the mission state and inserts a
        single-tick ``HOLD`` task at the front of the queue so the drone holds
        position rather than continuing on the unsafe waypoint.
        """
        decision = self.ctx.validate_command(command, drone_state)
        if decision.status == SafetyStatus.OK:
            self.ctx.execute_command(command, drone_state)
            return decision

        self.state.safety_events.append(
            SafetyEvent(
                timestamp_s=float(self.ctx.now_s()),
                drone_id=command.drone_id,
                task_type=command.task_type,
                target_xy_m=command.target_xy_m,
                target_z_m=command.target_z_m,
                reason=decision.reason or "safety rejected",
                violations=decision.violations,
            )
        )
        if self.state.active_task is not None:
            self.state.active_task.status = MissionTaskStatus.BLOCKED
            self.state.active_task.reason = decision.reason
        self._insert_hold_task(reason=decision.reason or "safety rejected")
        return decision

    # -------------------------
    # Mission-level loop step
    # -------------------------

    def step(self) -> None:
        """Advance mission-level state and select the active task."""

        if self.state.status in {MissionStatus.COMPLETED, MissionStatus.ABORTED, MissionStatus.FAILED}:
            return

        if self.state.status == MissionStatus.CREATED:
            self.state.status = MissionStatus.INITIALIZING

        if self.state.status == MissionStatus.INITIALIZING:
            self.state.status = MissionStatus.ACTIVE

        battery = self.ctx.get_battery_fraction()
        if battery < self.constraints.battery_reserve_fraction:
            self._force_return_home("battery low")
            return

        localization_conf = self.ctx.get_localization_confidence()
        if localization_conf < 0.5:
            self._activate_relocalization()
            return

        task = self._select_task()
        if task is None:
            self.state.status = MissionStatus.COMPLETED
            return

        if self.state.active_task is not task:
            self.state.active_task = task
            task.status = MissionTaskStatus.ACTIVE

        if self.ctx.is_task_complete():
            task.status = MissionTaskStatus.COMPLETE
            self.state.active_task = None

    # -------------------------
    # Helpers
    # -------------------------

    def _select_task(self) -> Optional[MissionTask]:
        candidates = [
            t for t in self.state.tasks
            if t.status in (MissionTaskStatus.PENDING, MissionTaskStatus.ACTIVE)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda t: -t.priority)[0]

    def _force_return_home(self, reason: str) -> None:
        self.state.status = MissionStatus.RETURNING_HOME
        self.state.tasks = [
            MissionTask(
                task_id="return_home",
                task_type=MissionTaskType.RETURN_HOME,
                priority=100,
                reason=reason,
            )
        ]
        self.state.active_task = None

    def _activate_relocalization(self) -> None:
        if any(
            t.task_type == MissionTaskType.RELOCALIZE
            and t.status in (MissionTaskStatus.PENDING, MissionTaskStatus.ACTIVE)
            for t in self.state.tasks
        ):
            return
        self.state.tasks.insert(
            0,
            MissionTask(task_id="relocalize", task_type=MissionTaskType.RELOCALIZE, priority=100),
        )

    def _insert_hold_task(self, *, reason: str) -> None:
        if any(
            t.task_type == MissionTaskType.HOLD
            and t.status in (MissionTaskStatus.PENDING, MissionTaskStatus.ACTIVE)
            for t in self.state.tasks
        ):
            return
        self.state.tasks.insert(
            0,
            MissionTask(
                task_id=f"hold-{len(self.state.safety_events)}",
                task_type=MissionTaskType.HOLD,
                priority=self.HOLD_PRIORITY,
                reason=reason,
            ),
        )
