from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from argusnet.evaluation.benchmarks import BenchmarkRun, aggregate_reports
from argusnet.evaluation.perf_summary import capture_environment, current_commit_sha, write_summary
from argusnet.evaluation.scenarios import (
    configs_for_suite,
    iter_config_runs,
    list_scenarios,
    run_config_with_sim,
)


def _parse_seeds(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--suite",
        default="fast",
        help="Benchmark suite: fast, slow, or one canonical scenario name.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed override, for example 7,42,137.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for per-run summaries and suite_summary.json.",
    )
    parser.add_argument(
        "--verbose-runs",
        action="store_true",
        default=False,
        help="Let simulator progress output stream to stdout instead of per-run sim.log files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List canonical benchmark scenarios and exit.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    if getattr(args, "list", False):
        for name in list_scenarios():
            print(name)
        return

    seeds = _parse_seeds(getattr(args, "seeds", None))
    configs = configs_for_suite(args.suite, seeds=seeds)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output or Path("runs") / "benchmarks" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    commit_sha = current_commit_sha()
    aggregate_inputs: dict[str, list[BenchmarkRun]] = {}
    run_summaries: list[dict] = []

    for config, seed in iter_config_runs(configs):
        run_dir = output_dir / config.name / f"seed-{seed}"
        run = BenchmarkRun(config=config, seed=seed)
        try:
            report = run_config_with_sim(
                config,
                seed,
                run_dir,
                quiet=not getattr(args, "verbose_runs", False),
            )
            run.report = report
            summary_path = write_summary(
                run_dir,
                report,
                env=env,
                command=["argusnet", "benchmark", "--suite", args.suite],
                seed=seed,
                commit_sha=commit_sha,
            )
            run_summaries.append(
                {
                    "scenario_name": config.name,
                    "seed": seed,
                    "summary_path": str(summary_path),
                    "passed": report.passed,
                    "frame_time_p95_ms": report.frame_time_p95_ms,
                    "wall_clock_s": report.wall_clock_s,
                }
            )
            print(f"[PASS] {config.name} seed={seed} -> {summary_path}")
        except Exception as exc:  # noqa: BLE001
            run.error = str(exc)
            run_summaries.append(
                {
                    "scenario_name": config.name,
                    "seed": seed,
                    "error": str(exc),
                }
            )
            print(f"[FAIL] {config.name} seed={seed}: {exc}")
        aggregate_inputs.setdefault(config.name, []).append(run)

    aggregates = {
        scenario_name: asdict(aggregate_reports(scenario_name, runs))
        for scenario_name, runs in aggregate_inputs.items()
    }
    suite_summary = {
        "schema_version": "argusnet-performance-suite-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "commit_sha": commit_sha,
        "environment": env,
        "runs": run_summaries,
        "aggregates": aggregates,
    }
    suite_path = output_dir / "suite_summary.json"
    with suite_path.open("w", encoding="utf-8") as handle:
        json.dump(suite_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Suite summary written to {suite_path}")
