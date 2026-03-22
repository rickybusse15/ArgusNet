"""Evaluation report generation for ArgusNet.

Renders :class:`EvaluationReport` and :class:`AggregatedResult` objects
into human-readable formats: plain text, Markdown, and JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

from argusnet.evaluation.metrics import EvaluationReport
from argusnet.evaluation.benchmarks import AggregatedResult

__all__ = [
    "report_to_dict",
    "report_to_json",
    "report_to_markdown",
    "aggregate_to_markdown",
    "print_report",
    "print_aggregate",
]


def report_to_dict(report: EvaluationReport) -> dict:
    """Convert an EvaluationReport to a plain Python dict."""
    return asdict(report)


def report_to_json(report: EvaluationReport, indent: int = 2) -> str:
    """Serialise an EvaluationReport to a JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, default=str)


def report_to_markdown(report: EvaluationReport) -> str:
    """Render a single EvaluationReport as a Markdown section."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"## Evaluation: {report.scenario_name}  [{status}]",
        "",
        f"- **Mission type**: {report.mission_type}",
        f"- **Difficulty**: {report.difficulty:.2f}",
        f"- **Seed**: {report.seed}",
        f"- **Duration**: {report.duration_s:.1f} s",
        "",
        "### Track Performance",
        "",
    ]

    def _fmt(v: Optional[float], unit: str = "") -> str:
        return f"{v:.3f}{unit}" if v is not None else "N/A"

    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Localisation RMSE | {_fmt(report.localisation_rmse_m, ' m')} |",
        f"| Track continuity | {_fmt(report.track_continuity_mean)} |",
        f"| Time to reacquire (mean) | {_fmt(report.time_to_reacquire_mean_s, ' s')} |",
        f"| Covariance reduction | {_fmt(report.covariance_reduction_mean)} |",
        "",
        "### Reliability",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| False handoff rate | {report.false_handoff_rate:.4f} |",
        f"| Safety overrides | {report.safety_override_count} |",
        f"| Comms dropout duration | {report.comms_dropout_duration_s:.2f} s |",
        "",
        "### Mission Outcome",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Completion rate | {report.mission_completion_rate:.2%} |",
        f"| Energy reserve (min) | {report.energy_reserve_min:.2%} |",
        "",
    ]

    if report.failure_reasons:
        lines += ["### Failure Reasons", ""]
        for reason in report.failure_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    return "\n".join(lines)


def aggregate_to_markdown(result: AggregatedResult) -> str:
    """Render an AggregatedResult as a Markdown section."""
    def _fmt(v: Optional[float], unit: str = "") -> str:
        return f"{v:.3f}{unit}" if v is not None else "N/A"

    def _fmt_pm(mean: Optional[float], std: Optional[float], unit: str = "") -> str:
        if mean is None:
            return "N/A"
        if std is not None:
            return f"{mean:.3f} ± {std:.3f}{unit}"
        return f"{mean:.3f}{unit}"

    lines = [
        f"## Benchmark: {result.name}",
        "",
        f"- **Runs**: {result.successful_runs}/{result.total_runs}",
        f"- **Pass rate**: {result.pass_rate:.0%}",
        "",
        "| Metric | Mean ± Std |",
        "|--------|-----------|",
        f"| Localisation RMSE | {_fmt_pm(result.localisation_rmse_mean, result.localisation_rmse_std, ' m')} |",
        f"| Track continuity | {_fmt_pm(result.track_continuity_mean, result.track_continuity_std)} |",
        f"| Mission completion | {_fmt(result.mission_completion_mean)} |",
        "",
    ]

    if result.failed_seeds:
        lines.append(f"**Failed seeds**: {result.failed_seeds}")
        lines.append("")
    if result.errors:
        lines += ["**Errors**:", ""]
        for err in result.errors:
            lines.append(f"- `{err}`")
        lines.append("")

    return "\n".join(lines)


def print_report(report: EvaluationReport) -> None:
    """Print a compact summary of an EvaluationReport to stdout."""
    status = "PASS" if report.passed else "FAIL"
    print(f"[{status}] {report.scenario_name} (seed={report.seed})")
    if report.localisation_rmse_m is not None:
        print(f"  RMSE: {report.localisation_rmse_m:.2f} m")
    print(f"  Track continuity: {report.track_continuity_mean:.3f}")
    print(f"  Mission completion: {report.mission_completion_rate:.1%}")
    if report.failure_reasons:
        for reason in report.failure_reasons:
            print(f"  ! {reason}")


def print_aggregate(result: AggregatedResult) -> None:
    """Print a compact summary of an AggregatedResult to stdout."""
    print(f"[{result.name}] {result.successful_runs}/{result.total_runs} runs, "
          f"pass={result.pass_rate:.0%}")
    if result.localisation_rmse_mean is not None:
        print(f"  RMSE: {result.localisation_rmse_mean:.2f} ± "
              f"{result.localisation_rmse_std or 0:.2f} m")
    if result.track_continuity_mean is not None:
        print(f"  Continuity: {result.track_continuity_mean:.3f} ± "
              f"{result.track_continuity_std or 0:.3f}")
