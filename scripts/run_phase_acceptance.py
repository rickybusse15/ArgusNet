#!/usr/bin/env python3
"""P4.9c phase acceptance: the canonical suite, reference against argusnet-rs.

For every canonical scenario (`argusnet.evaluation.scenarios`) and seed:

1. the reference -- `run_config_with_sim`, `argusnet sim` with the
   scenario's flags -- scored by `evaluate_replay` and `check_pass_fail`;
2. the same flags through `argusnet sim --export-config` (the P4.9 adapter)
   and `argusnet-rs sim --config` on a release build, scored the same way.

Then the envelopes declared in `docs/migration/PHASE_ACCEPTANCE.md` are
applied: exact checks per seed, statistical checks over the seeds, verdict
agreement, and the performance gates. The report goes to
`docs/performance/p49c-acceptance.json`; the exit status is non-zero if any
envelope fails.

Usage:
    cargo build --release -p argusnet-cli
    python3 scripts/run_phase_acceptance.py [--scenarios a,b] [--seeds 7,42] [--duration-s 20]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.evaluation.metrics import check_pass_fail, evaluate_replay  # noqa: E402
from argusnet.evaluation.replay import load_replay_document  # noqa: E402
from argusnet.evaluation.scenarios import (  # noqa: E402
    CANONICAL_SCENARIOS,
    CANONICAL_SEEDS,
    _sim_argv,
    run_config_with_sim,
)
from argusnet.simulation import sim  # noqa: E402

RELEASE_BINARY = REPO_ROOT / "target" / "release" / "argusnet-rs"
REPORT = REPO_ROOT / "docs" / "performance" / "p49c-acceptance.json"

# Flags that shape the output, not the scenario; dropped from the Rust side.
OUTPUT_ONLY_FLAGS = {"--streaming": 0, "--max-frames-in-memory": 1}

# The declared envelopes (PHASE_ACCEPTANCE.md). metric -> (relative, absolute):
# the allowed |mean(rust) - mean(reference)| is max(relative * |reference|,
# absolute), or just `absolute` where `relative` is None.
STATISTICAL = {
    "localisation_rmse_m": (0.25, 2.0),
    "track_continuity_mean": (None, 0.10),
    "time_to_reacquire_mean_s": (0.50, 2.0),
    "covariance_reduction_mean": (None, 0.10),
    "mission_completion_rate": (None, 0.20),
    "energy_reserve_min": (None, 0.02),
}
VERDICT_AGREEMENT_MIN = 4
# `covariance_reduction_mean` is a mean over tracks. A side's mean over the
# seeds is scored only when that many tracks qualified across all its seeds:
# below it, one track decides the number (decision D-1, SAFETY_AND_PROFILES.md).
# Neither side reaching it makes the check unavailable, not passed; only one
# side reaching it is a divergence and fails.
MIN_QUALIFYING_TRACKS = {"covariance_reduction_mean": ("covariance_reduction_tracks", 5)}
TICK_P95_MAX_MS = 5.0
STARTUP_MAX_MS = 2000.0
SPEEDUP_MIN = 5.0


def rust_argv(config, seed: int, replay: Path, exported: Path) -> list[str]:
    argv = _sim_argv(config, seed, replay)
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
    return [*kept, "--export-config", str(exported)]


def identity(replay: dict) -> dict:
    frames = replay.get("frames") or []
    targets = sorted({t["target_id"] for f in frames for t in f.get("truths") or []})
    return {
        "frames": len(frames),
        "node_ids": sorted((replay.get("meta") or {}).get("node_ids") or []),
        "target_ids": targets,
    }


def report_dict(report) -> dict:
    return {
        k: v
        for k, v in dataclasses.asdict(report).items()
        if not isinstance(v, dict) and not k.endswith("_per_target")
    }


def run_seed(config, seed: int, work: Path, duration_s: float | None) -> dict:
    if duration_s is not None:
        config = dataclasses.replace(config, duration_s=duration_s)
    reference_dir = work / "reference"
    started = time.perf_counter()
    reference = run_config_with_sim(config, seed, reference_dir)
    reference_elapsed = time.perf_counter() - started
    reference_replay = load_replay_document(str(reference_dir / "replay.json"))

    exported = work / "config.json"
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run_from_args(sim.parse_args(rust_argv(config, seed, work / "unused.json", exported)))
    rust_replay_path = work / "rust_replay.json"
    timing_path = work / "timing.json"
    completed = subprocess.run(
        [
            str(RELEASE_BINARY),
            "sim",
            "--quiet",
            "--config",
            str(exported),
            "--replay",
            str(rust_replay_path),
            "--timing-json",
            str(timing_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 3:
        return {"seed": seed, "unsupported": completed.stderr.strip()}
    if completed.returncode != 0:
        raise RuntimeError(f"argusnet-rs failed ({completed.returncode}): {completed.stderr}")
    rust_replay = load_replay_document(str(rust_replay_path))
    rust = check_pass_fail(
        evaluate_replay(
            rust_replay,
            scenario_name=config.name,
            mission_type=config.mission_type,
            difficulty=config.difficulty,
            seed=seed,
        )
    )
    timing = json.loads(timing_path.read_text())
    return {
        "seed": seed,
        "reference": report_dict(reference),
        "rust": report_dict(rust),
        "reference_identity": identity(reference_replay),
        "rust_identity": identity(rust_replay),
        "reference_wall_s": reference.wall_clock_s,
        "reference_process_s": reference_elapsed,
        "rust_timing": timing,
    }


def mean_of(runs: list[dict], side: str, metric: str) -> float | None:
    values = [r[side][metric] for r in runs if r[side].get(metric) is not None]
    return statistics.fmean(values) if values else None


def tracks_of(runs: list[dict], side: str, count_key: str) -> int:
    return sum(int(r[side].get(count_key) or 0) for r in runs)


def judge(name: str, runs: list[dict]) -> dict:
    checks: list[dict] = []

    def check(kind: str, what: str, ok: bool, detail: str) -> None:
        checks.append({"kind": kind, "check": what, "passed": bool(ok), "detail": detail})

    def unavailable(kind: str, what: str, detail: str) -> None:
        # Neither passed nor failed: there is nothing to compare. Kept out of
        # the verdict and listed beside it, never counted as a pass.
        checks.append({"kind": kind, "check": what, "passed": None, "detail": detail})

    for run in runs:
        ref, rust = run["reference_identity"], run["rust_identity"]
        for key in ("frames", "node_ids", "target_ids"):
            check(
                "exact",
                f"seed {run['seed']} {key}",
                ref[key] == rust[key],
                f"{ref[key]} / {rust[key]}",
            )
        ref_total = run["reference"]["required_objectives_total"]
        rust_total = run["rust"]["required_objectives_total"]
        check(
            "exact",
            f"seed {run['seed']} required_objectives_total",
            ref_total == rust_total,
            f"{ref_total} / {rust_total}",
        )
    for metric, (relative, absolute) in STATISTICAL.items():
        ref_mean = mean_of(runs, "reference", metric)
        rust_mean = mean_of(runs, "rust", metric)
        if metric in MIN_QUALIFYING_TRACKS:
            count_key, minimum = MIN_QUALIFYING_TRACKS[metric]
            ref_tracks = tracks_of(runs, "reference", count_key)
            rust_tracks = tracks_of(runs, "rust", count_key)
            counts = f"{ref_tracks} / {rust_tracks} qualifying tracks, {minimum} needed"
            if ref_tracks < minimum and rust_tracks < minimum:
                unavailable("statistical", metric, f"{counts}; {ref_mean} / {rust_mean}")
                continue
            if ref_tracks < minimum or rust_tracks < minimum:
                check("statistical", metric, False, counts)
                continue
        if ref_mean is None and rust_mean is None:
            unavailable("statistical", metric, "neither side reports it")
            continue
        if ref_mean is None or rust_mean is None:
            check("statistical", metric, False, f"{ref_mean} / {rust_mean}")
            continue
        allowed = absolute if relative is None else max(relative * abs(ref_mean), absolute)
        delta = abs(rust_mean - ref_mean)
        check(
            "statistical",
            metric,
            delta <= allowed,
            f"reference {ref_mean:.4g}, rust {rust_mean:.4g}, |delta| {delta:.4g} <= {allowed:.4g}",
        )
    agreeing = sum(1 for r in runs if r["reference"]["passed"] == r["rust"]["passed"])
    check(
        "verdict",
        "check_pass_fail agreement",
        agreeing >= min(VERDICT_AGREEMENT_MIN, len(runs)),
        f"{agreeing}/{len(runs)} seeds agree",
    )
    p95 = max(r["rust_timing"]["tick_ms"]["p95"] for r in runs)
    startup = max(r["rust_timing"]["startup_ms"] for r in runs)
    ref_wall = [r["reference_wall_s"] for r in runs if r["reference_wall_s"]]
    # Like for like: the reference's `wall_clock_s` times its simulation
    # loop (scenario building happens before it), so the Rust side is its
    # tick loop -- total less startup. Startup has its own gate.
    rust_wall = [
        (r["rust_timing"]["total_ms"] - r["rust_timing"]["startup_ms"]) / 1e3 for r in runs
    ]
    speedup = statistics.fmean(ref_wall) / statistics.fmean(rust_wall) if ref_wall else None
    check(
        "performance",
        "tick p95",
        p95 <= TICK_P95_MAX_MS,
        f"worst p95 {p95:.3f} ms <= {TICK_P95_MAX_MS} ms",
    )
    check(
        "performance",
        "startup",
        startup <= STARTUP_MAX_MS,
        f"worst {startup:.1f} ms <= {STARTUP_MAX_MS} ms",
    )
    check(
        "performance",
        "speed-up",
        speedup is not None and speedup >= SPEEDUP_MIN,
        f"{speedup:.1f}x >= {SPEEDUP_MIN}x" if speedup else "no reference wall clock",
    )
    return {
        "scenario": name,
        "passed": all(c["passed"] is not False for c in checks),
        "unavailable": [c["check"] for c in checks if c["passed"] is None],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="", help="comma-separated names (default: all)")
    parser.add_argument("--seeds", default="", help="comma-separated seeds (default: canonical)")
    parser.add_argument("--duration-s", type=float, default=None, help="override durations")
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    if not RELEASE_BINARY.exists():
        print(f"{RELEASE_BINARY} missing: cargo build --release -p argusnet-cli")
        return 2
    wanted = {s for s in args.scenarios.split(",") if s}
    seeds = [int(s) for s in args.seeds.split(",") if s] or list(CANONICAL_SEEDS)
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for config in CANONICAL_SCENARIOS:
            if wanted and config.name not in wanted:
                continue
            runs = []
            for seed in seeds:
                work = Path(scratch) / config.name / str(seed)
                work.mkdir(parents=True)
                print(f"{config.name} seed {seed} ...", flush=True)
                runs.append(run_seed(config, seed, work, args.duration_s))
            unsupported = [r for r in runs if "unsupported" in r]
            if unsupported:
                results.append(
                    {"scenario": config.name, "unsupported": unsupported[0]["unsupported"]}
                )
                print(f"  unsupported: {unsupported[0]['unsupported']}")
                continue
            verdict = judge(config.name, runs)
            verdict["runs"] = runs
            results.append(verdict)
            print(f"  {'PASS' if verdict['passed'] else 'FAIL'}")
            if verdict["unavailable"]:
                print(f"    unavailable: {', '.join(verdict['unavailable'])}")
            for failed in [c for c in verdict["checks"] if c["passed"] is False]:
                print(f"    {failed['check']}: {failed['detail']}")
    document = {
        "_source": "scripts/run_phase_acceptance.py",
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "seeds": seeds,
        "duration_override_s": args.duration_s,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n")
    scored = [r for r in results if "passed" in r]
    print(f"{sum(r['passed'] for r in scored)}/{len(scored)} scored scenarios pass")
    return 0 if all(r["passed"] for r in scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
