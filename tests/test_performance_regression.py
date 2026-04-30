from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from argusnet.evaluation.scenarios import benchmark_fast, run_config_with_sim

GOLDEN_DIR = Path(__file__).parent / "golden" / "performance"
PERF_FIELDS = (
    "wall_clock_s",
    "frame_time_mean_ms",
    "frame_time_p95_ms",
    "frame_time_p99_ms",
    "peak_rss_mb",
)


pytestmark = pytest.mark.benchmark_fast


def _golden_path(scenario_name: str, seed: int) -> Path:
    return GOLDEN_DIR / f"benchmark_fast_seed{seed}_{scenario_name}.json"


def _metric(summary: dict, field: str) -> float | None:
    value = summary.get("metrics", {}).get("performance", {}).get(field)
    return None if value is None else float(value)


def _classify_ratio(actual: float, baseline: float) -> str:
    if baseline <= 0.0:
        return "green"
    ratio = (actual - baseline) / baseline
    if ratio > 0.20:
        return "red"
    if ratio > 0.05:
        return "yellow"
    return "green"


def test_fast_benchmark_regression(tmp_path) -> None:
    if os.environ.get("ARGUSNET_RUN_PERF_REGRESSION") != "1":
        pytest.skip("performance regression run requires ARGUSNET_RUN_PERF_REGRESSION=1")

    failures: list[str] = []
    for config in benchmark_fast(duration_s=1.0):
        seed = config.seeds[0]
        golden = json.loads(_golden_path(config.name, seed).read_text(encoding="utf-8"))
        report = run_config_with_sim(config, seed, tmp_path / config.name)
        actual = {
            "metrics": {
                "performance": {
                    "wall_clock_s": report.wall_clock_s,
                    "frame_time_mean_ms": report.frame_time_mean_ms,
                    "frame_time_p95_ms": report.frame_time_p95_ms,
                    "frame_time_p99_ms": report.frame_time_p99_ms,
                    "peak_rss_mb": report.peak_rss_mb,
                }
            }
        }
        for field in PERF_FIELDS:
            actual_value = _metric(actual, field)
            baseline_value = _metric(golden, field)
            if actual_value is None or baseline_value is None:
                continue
            severity = _classify_ratio(actual_value, baseline_value)
            if severity == "red":
                failures.append(
                    f"{config.name} seed={seed} {field}: "
                    f"{actual_value:.3f} > {baseline_value:.3f} by >20%"
                )
    if failures and os.environ.get("ARGUSNET_PERF_OVERRIDE") != "1":
        pytest.fail("\n".join(failures))
