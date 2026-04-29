from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional


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


@dataclass
class ExecutableCommand:
    description: str


@dataclass
class SafetyDecision:
    status: SafetyStatus
    reason: Optional[str] = None


@dataclass
class MissionState:
    mission_id: str
    status: MissionStatus = MissionStatus.CREATED
    tasks: List[MissionTask] = field(default_factory=list)
    active_task: Optional[MissionTask] = None


# -----------------------------
# Execution context (dependency injection)
# -----------------------------


@dataclass
class MissionExecutionContext:
    get_localization_confidence: Callable[[], float]
    get_battery_fraction: Callable[[], float]
    plan_task: Callable[[MissionTask], ExecutableCommand]
    validate_command: Callable[[ExecutableCommand], SafetyDecision]
    execute_command: Callable[[ExecutableCommand], None]


# -----------------------------
# Mission executor
# -----------------------------


class MissionExecutor:
    """
    Lightweight closed-loop mission execution skeleton.

    This class does not implement mapping, localization, or planning logic.
    It orchestrates them via injected callables.
    """

    def __init__(self, state: MissionState, constraints: MissionConstraints, ctx: MissionExecutionContext):
        self.state = state
        self.constraints = constraints
        self.ctx = ctx

    # -------------------------
    # Core loop step
    # -------------------------

    def step(self) -> None:
        """Execute one iteration of the mission loop."""

        if self.state.status in {MissionStatus.COMPLETED, MissionStatus.ABORTED, MissionStatus.FAILED}:
            return

        # 1. Ensure active state
        if self.state.status == MissionStatus.CREATED:
            self.state.status = MissionStatus.INITIALIZING

        if self.state.status == MissionStatus.INITIALIZING:
            self.state.status = MissionStatus.ACTIVE

        # 2. Safety: battery check
        battery = self.ctx.get_battery_fraction()
        if battery < self.constraints.battery_reserve_fraction:
            self._force_return_home("battery low")
            return

        # 3. Localization gate
        localization_conf = self.ctx.get_localization_confidence()
        if localization_conf < 0.5:
            self._activate_relocalization()
            return

        # 4. Select task
        task = self._select_task()
        if task is None:
            self.state.status = MissionStatus.COMPLETED
            return

        self.state.active_task = task
        task.status = MissionTaskStatus.ACTIVE

        # 5. Plan
        command = self.ctx.plan_task(task)

        # 6. Safety validation
        decision = self.ctx.validate_command(command)
        if decision.status == SafetyStatus.REJECTED:
            task.status = MissionTaskStatus.BLOCKED
            task.reason = decision.reason
            return

        # 7. Execute
        self.ctx.execute_command(command)

        # 8. Mark complete (stub behavior)
        task.status = MissionTaskStatus.COMPLETE

    # -------------------------
    # Helpers
    # -------------------------

    def _select_task(self) -> Optional[MissionTask]:
        pending = [t for t in self.state.tasks if t.status == MissionTaskStatus.PENDING]
        if not pending:
            return None
        return sorted(pending, key=lambda t: -t.priority)[0]

    def _force_return_home(self, reason: str) -> None:
        self.state.status = MissionStatus.RETURNING_HOME
        self.state.tasks = [
            MissionTask(task_id="return_home", task_type=MissionTaskType.RETURN_HOME, priority=100)
        ]

    def _activate_relocalization(self) -> None:
        self.state.tasks.insert(
            0,
            MissionTask(task_id="relocalize", task_type=MissionTaskType.RELOCALIZE, priority=100),
        )
