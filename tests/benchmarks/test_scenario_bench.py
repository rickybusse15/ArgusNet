from __future__ import annotations

import os

import pytest

pytest.importorskip("pytest_benchmark")

from argusnet.evaluation.scenarios import benchmark_fast, run_config_with_sim


def _require_benchmark_only(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--benchmark-only", default=False):
        pytest.skip("benchmark tests run only with --benchmark-only")


@pytest.mark.benchmark_slow
def test_benchmark_fast_suite_end_to_end(
    benchmark,
    request: pytest.FixtureRequest,
    tmp_path,
) -> None:
    _require_benchmark_only(request)
    if os.environ.get("ARGUSNET_RUN_SCENARIO_BENCH") != "1":
        pytest.skip("end-to-end scenario benchmark requires ARGUSNET_RUN_SCENARIO_BENCH=1")
    configs = benchmark_fast(duration_s=1.0)

    def run() -> int:
        passed = 0
        for config in configs:
            report = run_config_with_sim(config, config.seeds[0], tmp_path / config.name)
            passed += int(report.passed)
        return passed

    benchmark(run)
