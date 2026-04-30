from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argusnet.evaluation.metrics import EvaluationReport, report_to_dict

SCHEMA_VERSION = "argusnet-performance-v1"


def _command_output(command: Sequence[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.check_output(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.strip() or None


def _memory_total_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().total)
    except (OSError, AttributeError):
        return None


def capture_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "memory_total_bytes": _memory_total_bytes(),
        "rustc_version": _command_output(("rustc", "--version"), cwd),
        "cargo_version": _command_output(("cargo", "--version"), cwd),
    }


def current_commit_sha(cwd: str | Path | None = None) -> str | None:
    return _command_output(("git", "rev-parse", "HEAD"), cwd)


def normalize_command(command: Sequence[object] | str | None) -> list[str]:
    if command is None:
        return [str(part) for part in sys.argv]
    if isinstance(command, str):
        return [command]
    return [str(part) for part in command]


@dataclass(frozen=True)
class PerformanceSummary:
    commit_sha: str | None
    command: list[str]
    environment: Mapping[str, Any]
    metrics: Mapping[str, Any]
    scenario_name: str | None = None
    seed: int | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "commit_sha": self.commit_sha,
            "created_at_utc": self.created_at_utc,
            "command": list(self.command),
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
        }
        if self.scenario_name is not None:
            payload["scenario_name"] = self.scenario_name
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        return payload


def metrics_from_report(report: EvaluationReport) -> dict[str, Any]:
    report_dict = report_to_dict(report)
    return {
        "report": report_dict,
        "mission": {
            "passed": report.passed,
            "mission_completion_rate": report.mission_completion_rate,
            "localisation_rmse_m": report.localisation_rmse_m,
            "track_continuity_mean": report.track_continuity_mean,
        },
        "performance": {
            "wall_clock_s": report.wall_clock_s,
            "frame_time_mean_ms": report.frame_time_mean_ms,
            "frame_time_p95_ms": report.frame_time_p95_ms,
            "frame_time_p99_ms": report.frame_time_p99_ms,
            "peak_rss_mb": report.peak_rss_mb,
            "replay_size_mb": report.replay_size_mb,
            "terrain_queries_per_sec": report.terrain_queries_per_sec,
            "observations_per_sec": report.observations_per_sec,
        },
    }


def build_summary(
    *,
    report: EvaluationReport,
    env: Mapping[str, Any] | None = None,
    command: Sequence[object] | str | None = None,
    seed: int | None = None,
    commit_sha: str | None = None,
    cwd: str | Path | None = None,
) -> PerformanceSummary:
    return PerformanceSummary(
        commit_sha=commit_sha if commit_sha is not None else current_commit_sha(cwd),
        command=normalize_command(command),
        environment=dict(env) if env is not None else capture_environment(cwd),
        metrics=metrics_from_report(report),
        scenario_name=report.scenario_name,
        seed=report.seed if seed is None else int(seed),
    )


def write_summary(
    path: str | Path,
    report: EvaluationReport,
    env: Mapping[str, Any] | None = None,
    command: Sequence[object] | str | None = None,
    seed: int | None = None,
    commit_sha: str | None = None,
    cwd: str | Path | None = None,
) -> Path:
    output_path = Path(path)
    if output_path.suffix == "" or output_path.is_dir():
        output_path = output_path / "performance_summary.json"
    summary = build_summary(
        report=report,
        env=env,
        command=command,
        seed=seed,
        commit_sha=commit_sha,
        cwd=cwd,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path

