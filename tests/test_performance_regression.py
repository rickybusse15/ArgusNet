"""Performance regression gate for the canonical fast benchmark scenarios.

This gate has two halves, because the two things worth guarding have very
different reproducibility.

**Work counters** are deterministic for a fixed seed and configuration: how many
terrain heights the simulation looked up, how many line-of-sight rays it cast,
how many observations it generated and rejected. They do not depend on how fast
the machine is, so they can be compared exactly and are enforced on every run.
They are also what actually regresses when someone makes the simulation do more
work per tick, which is the failure this suite exists to catch.

**Timing** (wall clock, frame time, peak RSS) is machine-dependent. Comparing it
against a baseline recorded on a different machine measures the runner, not the
change. It is recorded in each golden file for reference and only compared when
``ARGUSNET_RUN_PERF_TIMING=1`` marks the current machine as the calibrated one.

Regenerate the golden files after an intentional change:

    ARGUSNET_UPDATE_PERF_GOLDEN=1 python -m pytest tests/test_performance_regression.py

and commit the result with an explanation of why the work changed.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import pytest

from argusnet.evaluation.scenarios import benchmark_fast, run_config_with_sim

GOLDEN_DIR = Path(__file__).parent / "golden" / "performance"

BENCHMARK_DURATION_S = 1.0

TIMING_FIELDS = (
    "wall_clock_s",
    "frame_time_mean_ms",
    "frame_time_p95_ms",
    "frame_time_p99_ms",
    "peak_rss_mb",
)

#: Timing tolerance, used only on a calibrated machine. Deliberately generous:
#: even on one machine, frame time over a handful of frames is noisy.
TIMING_RED_RATIO = 0.50

pytestmark = pytest.mark.benchmark_fast


def _golden_path(scenario_name: str, seed: int) -> Path:
    return GOLDEN_DIR / f"benchmark_fast_seed{seed}_{scenario_name}.json"


def _flatten_cache_counters(cache_metrics: Any) -> dict[str, dict[str, int]]:
    """Reduce the nested cache-metrics tree to ``group.cache -> counters``.

    Only the counters that are a function of the work done are kept. ``size`` and
    ``capacity`` describe the cache rather than the workload, so they are dropped.
    """
    flattened: dict[str, dict[str, int]] = {}
    if not isinstance(cache_metrics, dict):
        return flattened
    for group_name, group in sorted(cache_metrics.items()):
        if not isinstance(group, dict):
            continue
        for cache_name, counters in sorted(group.items()):
            if not isinstance(counters, dict):
                continue
            flattened[f"{group_name}.{cache_name}"] = {
                key: int(counters.get(key, 0)) for key in ("hits", "misses", "evictions")
            }
    return flattened


def _work_snapshot(replay: dict) -> dict[str, Any]:
    """Machine-independent description of how much work a run performed."""
    meta = replay.get("meta", {})
    summary = replay.get("summary", {})
    performance = meta.get("performance", {})
    return {
        "frame_count": int(meta.get("frame_count", 0)),
        "total_accepted_observations": int(summary.get("total_accepted_observations", 0)),
        "total_rejected_observations": int(summary.get("total_rejected_observations", 0)),
        "cache": _flatten_cache_counters(performance.get("cache_metrics")),
    }


def _timing_snapshot(report: Any) -> dict[str, float | None]:
    return {field: getattr(report, field, None) for field in TIMING_FIELDS}


def _run(config: Any, seed: int, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = run_config_with_sim(config, seed, tmp_path / config.name)
    replay = json.loads((tmp_path / config.name / "replay.json").read_text(encoding="utf-8"))
    return _work_snapshot(replay), _timing_snapshot(report)


def _diff_work(actual: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    """Human-readable differences between two work snapshots."""
    failures: list[str] = []
    for field in ("frame_count", "total_accepted_observations", "total_rejected_observations"):
        if actual.get(field) != golden.get(field):
            failures.append(f"  {field}: {golden.get(field)} -> {actual.get(field)}")

    actual_cache = actual.get("cache", {})
    golden_cache = golden.get("cache", {})
    for name in sorted(set(actual_cache) | set(golden_cache)):
        if name not in golden_cache:
            failures.append(f"  {name}: new cache, now {actual_cache[name]}")
            continue
        if name not in actual_cache:
            failures.append(f"  {name}: cache disappeared, was {golden_cache[name]}")
            continue
        for counter in ("hits", "misses", "evictions"):
            before = golden_cache[name].get(counter, 0)
            after = actual_cache[name].get(counter, 0)
            if before != after:
                delta = after - before
                failures.append(f"  {name}.{counter}: {before} -> {after} ({delta:+d})")
    return failures


def test_fast_benchmark_work_is_unchanged(tmp_path) -> None:
    """The simulation must not silently start doing more work per run.

    Unlike the timing check below this is machine-independent, so it runs
    everywhere and compares exactly.
    """
    updating = os.environ.get("ARGUSNET_UPDATE_PERF_GOLDEN") == "1"
    failures: list[str] = []

    for config in benchmark_fast(duration_s=BENCHMARK_DURATION_S):
        seed = config.seeds[0]
        work, timing = _run(config, seed, tmp_path)
        path = _golden_path(config.name, seed)

        if updating:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "scenario": config.name,
                        "seed": seed,
                        "duration_s": BENCHMARK_DURATION_S,
                        "work": work,
                        "timing_reference": {
                            "note": (
                                "Recorded for reference. Only compared when "
                                "ARGUSNET_RUN_PERF_TIMING=1 marks this machine as calibrated."
                            ),
                            "machine": f"{platform.system()} {platform.machine()}",
                            "python": platform.python_version(),
                            **timing,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            continue

        golden = json.loads(path.read_text(encoding="utf-8"))
        differences = _diff_work(work, golden.get("work", {}))
        if differences:
            failures.append(
                f"{config.name} seed={seed} performs different work than the baseline:\n"
                + "\n".join(differences)
            )

    if updating:
        pytest.skip("golden performance baselines rewritten")
    if failures:
        pytest.fail(
            "\n\n".join(failures) + "\n\nIf the change is intentional, regenerate with "
            "ARGUSNET_UPDATE_PERF_GOLDEN=1 and explain the delta in the commit message."
        )


def test_fast_benchmark_timing_on_a_calibrated_machine(tmp_path) -> None:
    """Compare wall-clock timing, but only where the baseline was recorded.

    Absolute timings do not transfer between machines, so this is opt-in. The
    work-counter test above is the gate that runs everywhere.
    """
    if os.environ.get("ARGUSNET_RUN_PERF_TIMING") != "1":
        pytest.skip(
            "timing comparison requires ARGUSNET_RUN_PERF_TIMING=1 on the machine "
            "the golden baselines were recorded on"
        )

    failures: list[str] = []
    for config in benchmark_fast(duration_s=BENCHMARK_DURATION_S):
        seed = config.seeds[0]
        _, timing = _run(config, seed, tmp_path)
        golden = json.loads(_golden_path(config.name, seed).read_text(encoding="utf-8"))
        reference = golden.get("timing_reference", {})

        for field in TIMING_FIELDS:
            actual = timing.get(field)
            baseline = reference.get(field)
            if actual is None or baseline is None or baseline <= 0.0:
                continue
            ratio = (float(actual) - float(baseline)) / float(baseline)
            if ratio > TIMING_RED_RATIO:
                failures.append(
                    f"{config.name} seed={seed} {field}: "
                    f"{float(actual):.3f} exceeds baseline {float(baseline):.3f} "
                    f"by {ratio:.0%}"
                )

    if failures and os.environ.get("ARGUSNET_PERF_OVERRIDE") != "1":
        pytest.fail("\n".join(failures))
