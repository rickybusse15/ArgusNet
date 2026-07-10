"""System scorecard summaries for ArgusNet benchmarks and replays."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from argusnet.evaluation.metrics import EvaluationReport, evaluate_replay


@dataclass(frozen=True)
class SystemScorecard:
    scenario_name: str
    seed: int
    generated_at_utc: str
    mission: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    data_health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "generated_at_utc": self.generated_at_utc,
            "mission": dict(self.mission),
            "performance": dict(self.performance),
            "reliability": dict(self.reliability),
            "data_health": dict(self.data_health),
        }


def scorecard_from_report(report: EvaluationReport) -> SystemScorecard:
    return SystemScorecard(
        scenario_name=report.scenario_name,
        seed=report.seed,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mission={
            "mission_type": report.mission_type,
            "duration_s": report.duration_s,
            "completion_rate": report.mission_completion_rate,
            "required_objectives_met": report.required_objectives_met,
            "required_objectives_total": report.required_objectives_total,
            "energy_reserve_min": report.energy_reserve_min,
            "localisation_rmse_m": report.localisation_rmse_m,
            "track_continuity_mean": report.track_continuity_mean,
        },
        performance={
            "wall_clock_s": report.wall_clock_s,
            "frame_time_mean_ms": report.frame_time_mean_ms,
            "frame_time_p95_ms": report.frame_time_p95_ms,
            "frame_time_p99_ms": report.frame_time_p99_ms,
            "peak_rss_mb": report.peak_rss_mb,
            "replay_size_mb": report.replay_size_mb,
            "terrain_queries_per_sec": report.terrain_queries_per_sec,
            "observations_per_sec": report.observations_per_sec,
        },
        reliability={
            "safety_override_count": report.safety_override_count,
            "infeasible_path_rejection_count": report.infeasible_path_rejection_count,
            "time_to_reacquire_mean_s": report.time_to_reacquire_mean_s,
            "time_to_reacquire_p95_s": report.time_to_reacquire_p95_s,
            "comms_dropout_count": report.comms_dropout_count,
            "comms_dropout_duration_s": report.comms_dropout_duration_s,
        },
        data_health={
            "passed": report.passed,
            "failure_reasons": list(report.failure_reasons),
            "tags": list(report.tags),
        },
    )


def scorecard_from_replay(
    replay_doc: dict[str, Any],
    *,
    scenario_name: str | None = None,
    mission_type: str = "unknown",
    difficulty: float = 0.0,
    seed: int | None = None,
) -> SystemScorecard:
    meta = replay_doc.get("meta") or {}
    report = evaluate_replay(
        replay_doc,
        scenario_name=scenario_name or str(meta.get("scenario_name", "unknown")),
        mission_type=mission_type,
        difficulty=difficulty,
        seed=int(seed if seed is not None else meta.get("seed", 0)),
    )
    return scorecard_from_report(report)
