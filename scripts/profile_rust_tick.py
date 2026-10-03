#!/usr/bin/env python3
"""P4.10-pre: per-stage profile of the Rust tick over the canonical scenarios.

For every canonical scenario (`argusnet.evaluation.scenarios`) and seed, the
scenario's flags go through `argusnet sim --export-config` (as
`run_phase_acceptance.py` does), then a release `argusnet-rs sim --config
--stage-timing` runs it `--runs` times. Each run's per-stage wall times come
from `--timing-json`; nothing is read from the replay, and the replay is the
same with timing on or off (`argusnet-cli` tests that).

The report gives, per scenario, each stage's share of summed stage time and
its p50/p95/p99 over every tick of every run, plus run-to-run spread of the
per-tick mean. The Amdahl table ranks stages by share: the upper bound on the
whole-tick speed-up from making one stage free is `1 / (1 - share)`.

Wall time depends on the machine and its load. The report records the
machine; numbers from different machines are not comparable.

Usage:
    cargo build --release -p argusnet-cli
    python3 scripts/profile_rust_tick.py [--scenarios a,b] [--seeds 7] [--runs 5] \\
        [--out docs/performance/rust-tick-profile.json]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.evaluation.scenarios import CANONICAL_SCENARIOS, _sim_argv  # noqa: E402
from argusnet.simulation import sim  # noqa: E402

RELEASE_BINARY = REPO_ROOT / "target" / "release" / "argusnet-rs"
OUTPUT_ONLY_FLAGS = {"--streaming": 0, "--max-frames-in-memory": 1}


def export_config(config, seed: int, work: Path) -> Path:
    argv = _sim_argv(config, seed, work / "unused.json")
    kept: list[str] = []
    skip = 0
    for token in argv:
        if skip:
            skip -= 1
            continue
        if token in OUTPUT_ONLY_FLAGS:
            skip = OUTPUT_ONLY_FLAGS[token]
            continue
        kept.append(token)
    exported = work / "config.json"
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run_from_args(sim.parse_args([*kept, "--export-config", str(exported)]))
    return exported


def run_once(binary: Path, config_path: Path, work: Path) -> dict | None:
    timing_path = work / "timing.json"
    completed = subprocess.run(
        [
            str(binary),
            "sim",
            "--quiet",
            "--stage-timing",
            "--config",
            str(config_path),
            "--timing-json",
            str(timing_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 3:
        return None
    if completed.returncode != 0:
        raise RuntimeError(f"argusnet-rs failed ({completed.returncode}): {completed.stderr}")
    return json.loads(timing_path.read_text())


def summarize(name: str, runs: list[dict]) -> dict:
    stages = list(runs[0]["stages"])
    totals = {stage: sum(r["stages"][stage]["total_ms"] for r in runs) for stage in stages}
    grand = sum(totals.values())
    per_stage = {}
    for stage in stages:
        share = totals[stage] / grand if grand > 0 else 0.0
        means = [r["stages"][stage]["mean_ms"] for r in runs]
        per_stage[stage] = {
            "share": share,
            "amdahl_bound": (1.0 / (1.0 - share)) if share < 1.0 else None,
            "mean_ms": statistics.fmean(means),
            "mean_ms_stdev": statistics.stdev(means) if len(means) > 1 else 0.0,
            # Medians of per-run percentiles: robust to one noisy run.
            "p50_ms": statistics.median(r["stages"][stage]["p50_ms"] for r in runs),
            "p95_ms": statistics.median(r["stages"][stage]["p95_ms"] for r in runs),
            "p99_ms": statistics.median(r["stages"][stage]["p99_ms"] for r in runs),
            "max_ms": max(r["stages"][stage]["max_ms"] for r in runs),
        }
    tick_means = [r["tick_ms"]["mean"] for r in runs]
    return {
        "scenario": name,
        "runs": len(runs),
        "ticks_per_run": runs[0]["ticks"],
        "tick_ms": {
            "mean": statistics.fmean(tick_means),
            "mean_stdev": statistics.stdev(tick_means) if len(tick_means) > 1 else 0.0,
            "p95_median": statistics.median(r["tick_ms"]["p95"] for r in runs),
            "p99_median": statistics.median(r["tick_ms"]["p99"] for r in runs),
        },
        "startup_ms_median": statistics.median(r["startup_ms"] for r in runs),
        "replay_write_ms_median": statistics.median(r["replay_write_ms"] for r in runs),
        "stages": per_stage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="", help="comma-separated names (default: all)")
    parser.add_argument("--seeds", default="7", help="comma-separated seeds (default: 7)")
    parser.add_argument("--runs", type=int, default=5, help="runs per scenario and seed")
    parser.add_argument("--binary", type=Path, default=RELEASE_BINARY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.exists():
        print(f"{args.binary} missing: cargo build --release -p argusnet-cli")
        return 2
    wanted = {s for s in args.scenarios.split(",") if s}
    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for config in CANONICAL_SCENARIOS:
            if wanted and config.name not in wanted:
                continue
            runs: list[dict] = []
            unsupported = False
            for seed in seeds:
                work = Path(scratch) / config.name / str(seed)
                work.mkdir(parents=True)
                exported = export_config(config, seed, work)
                for _ in range(args.runs):
                    timing = run_once(args.binary, exported, work)
                    if timing is None:
                        unsupported = True
                        break
                    runs.append(timing)
            if unsupported:
                results.append({"scenario": config.name, "unsupported": True})
                continue
            summary = summarize(config.name, runs)
            results.append(summary)
            top = sorted(summary["stages"].items(), key=lambda kv: -kv[1]["share"])[:3]
            shares = ", ".join(f"{stage} {100 * s['share']:.1f}%" for stage, s in top)
            print(f"{config.name}: tick mean {summary['tick_ms']['mean']:.3f} ms | {shares}")
    document = {
        "_source": "scripts/profile_rust_tick.py",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "seeds": seeds,
        "runs_per_seed": args.runs,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
