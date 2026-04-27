"""Benchmark runner for ArgusNet evaluation.

Provides a harness for running simulation scenarios at multiple seeds,
collecting EvaluationReport objects, and aggregating results across runs.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from argusnet.evaluation.metrics import EvaluationReport

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRun",
    "BenchmarkSuite",
    "AggregatedResult",
    "aggregate_reports",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""

    name: str
    """Unique benchmark name."""

    mission_type: str = "surveillance"
    """Mission type passed to scenario builder."""

    difficulty: float = 0.5
    """Difficulty level [0, 1]."""

    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    """Random seeds to run."""

    duration_s: float = 120.0
    """Simulation duration per seed (seconds)."""

    tags: list[str] = field(default_factory=list)
    """Optional categorisation tags."""


@dataclass
class BenchmarkRun:
    """A single executed benchmark (one seed)."""

    config: BenchmarkConfig
    seed: int
    report: EvaluationReport | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.report is not None and self.error is None


@dataclass(frozen=True)
class AggregatedResult:
    """Aggregated statistics across multiple seeds for a benchmark."""

    name: str
    total_runs: int
    successful_runs: int
    pass_rate: float

    localisation_rmse_mean: float | None
    localisation_rmse_std: float | None
    track_continuity_mean: float | None
    track_continuity_std: float | None
    mission_completion_mean: float | None

    failed_seeds: list[int]
    errors: list[str]


def aggregate_reports(
    name: str,
    runs: list[BenchmarkRun],
) -> AggregatedResult:
    """Aggregate a list of :class:`BenchmarkRun` objects into statistics."""
    successful = [r for r in runs if r.succeeded]
    total = len(runs)
    n_ok = len(successful)

    reports = [r.report for r in successful if r.report is not None]

    def _mean_std(vals: list[float]):
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rmse_vals = [r.localisation_rmse_m for r in reports if r.localisation_rmse_m is not None]
    cont_vals = [r.track_continuity_mean for r in reports]
    comp_vals = [r.mission_completion_rate for r in reports]

    rmse_mean, rmse_std = _mean_std(rmse_vals)
    cont_mean, cont_std = _mean_std(cont_vals)
    comp_mean, _ = _mean_std(comp_vals)

    pass_rate = sum(1 for r in reports if r.passed) / max(n_ok, 1)

    failed_seeds = [r.seed for r in runs if not r.succeeded]
    errors = [r.error for r in runs if r.error]

    return AggregatedResult(
        name=name,
        total_runs=total,
        successful_runs=n_ok,
        pass_rate=pass_rate,
        localisation_rmse_mean=rmse_mean,
        localisation_rmse_std=rmse_std,
        track_continuity_mean=cont_mean,
        track_continuity_std=cont_std,
        mission_completion_mean=comp_mean,
        failed_seeds=failed_seeds,
        errors=errors,
    )


RunFn = Callable[[BenchmarkConfig, int], EvaluationReport]


class BenchmarkSuite:
    """Collection of benchmark configurations with a shared run function.

    Usage::

        suite = BenchmarkSuite(run_fn=my_run_function)
        suite.add(BenchmarkConfig("baseline", seeds=[42, 43, 44]))
        results = suite.run_all()
    """

    def __init__(self, run_fn: RunFn) -> None:
        self._run_fn = run_fn
        self._configs: list[BenchmarkConfig] = []

    def add(self, config: BenchmarkConfig) -> None:
        self._configs.append(config)

    def run_all(self, verbose: bool = False) -> dict[str, AggregatedResult]:
        results: dict[str, AggregatedResult] = {}
        for config in self._configs:
            runs: list[BenchmarkRun] = []
            for seed in config.seeds:
                run = BenchmarkRun(config=config, seed=seed)
                try:
                    run.report = self._run_fn(config, seed)
                except Exception as exc:  # noqa: BLE001
                    run.error = str(exc)
                runs.append(run)
                if verbose:
                    status = (
                        "PASS" if run.succeeded and run.report and run.report.passed else "FAIL"
                    )
                    print(f"  [{status}] {config.name} seed={seed}")
            results[config.name] = aggregate_reports(config.name, runs)
        return results
