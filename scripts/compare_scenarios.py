#!/usr/bin/env python3
"""Compare variants of a scenario over paired seeds (F4).

Every variant runs on the same seeds through a release ``argusnet-rs``, and
is scored the way acceptance scores (``evaluate_replay``). A variant is the
canonical scenario plus extra ``argusnet sim`` arguments; the last value of a
repeated flag wins, so a variant overrides the scenario's own setting.

    cargo build --release -p argusnet-cli
    python3 scripts/compare_scenarios.py --scenario baseline_coverage \\
        --variant "baseline=" --variant "six-drones=--drone-count 6" \\
        --seed-set qualification --out comparison.json --markdown comparison.md

The report (JSON, and Markdown with ``--markdown``) records the identities that
make it reproducible: the scenario, each variant's arguments, the sha256 of
every exported run configuration, the world, the git commit and the runtime
binary's sha256. See ``argusnet.evaluation.comparison`` for the statistics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import shlex

# Runs the project's own release binary and fixed git metadata commands,
# always as argument lists without a shell.
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_phase_acceptance import RELEASE_BINARY, SEED_SETS, rust_argv  # noqa: E402

from argusnet.evaluation.comparison import RunResult, compare, markdown  # noqa: E402
from argusnet.evaluation.metrics import check_pass_fail, evaluate_replay  # noqa: E402
from argusnet.evaluation.replay import load_replay_document  # noqa: E402
from argusnet.evaluation.scenarios import CANONICAL_SCENARIOS  # noqa: E402
from argusnet.simulation import sim  # noqa: E402


def parse_variant(text: str) -> tuple[str, list[str]]:
    """``NAME=ARGS``: ``argusnet sim`` arguments, then optionally ``--rs``
    followed by ``argusnet-rs sim`` arguments (for example
    ``--energy-profile physical_v1``). Kept as one token list; see
    :func:`split_args`."""
    name, sep, args = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"--variant takes NAME=ARGS, got {text!r}")
    return name.strip(), shlex.split(args)


def split_args(tokens: list[str]) -> tuple[list[str], list[str]]:
    """A variant's tokens as (scenario arguments, runtime arguments)."""
    if "--rs" not in tokens:
        return tokens, []
    at = tokens.index("--rs")
    return tokens[:at], tokens[at + 1 :]


def export(config, extra: list[str], seed: int, work: Path) -> tuple[Path | None, str | None]:
    """Write the run's scenario config. Serial: the export prints, and
    redirecting stdout is process-wide, so it must not share threads."""
    work.mkdir(parents=True, exist_ok=True)
    exported = work / "config.json"
    argv = rust_argv(config, seed, work / "unused.json", exported)
    # Overrides go before --export-config, which rust_argv puts last.
    argv = [*argv[:-2], *extra, *argv[-2:]]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sim.run_from_args(sim.parse_args(argv))
    except SystemExit as exc:  # argparse rejects a bad override
        return None, f"invalid arguments: {extra} (exit {exc.code})"
    return exported, None


def run_one(
    config,
    variant: str,
    seed: int,
    exported: Path,
    binary: Path,
    work: Path,
    runtime_args: list[str] | None = None,
) -> RunResult:
    result = RunResult(variant=variant, seed=seed)
    result.config_sha256 = hashlib.sha256(exported.read_bytes()).hexdigest()
    replay_path, timing_path = work / "replay.json", work / "timing.json"
    started = time.perf_counter()
    # The project's own release binary, as an argument list without a shell.
    completed = subprocess.run(  # nosec B603
        [
            str(binary),
            "sim",
            "--quiet",
            "--config",
            str(exported),
            "--replay",
            str(replay_path),
            "--timing-json",
            str(timing_path),
            *(runtime_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result.wall_s = time.perf_counter() - started
    if completed.returncode == 3:
        result.error = f"unsupported: {completed.stderr.strip()}"
        return result
    if completed.returncode != 0:
        result.error = (
            f"argusnet-rs exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
        return result
    replay = load_replay_document(str(replay_path))
    report = check_pass_fail(
        evaluate_replay(
            replay,
            scenario_name=config.name,
            mission_type=config.mission_type,
            difficulty=config.difficulty,
            seed=seed,
        )
    )
    result.passed_envelope = bool(report.passed)
    result.metrics = {
        k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
        for k, v in report.__dict__.items()
    }
    timing = json.loads(timing_path.read_text())
    result.tick_mean_ms = (timing.get("tick_ms") or {}).get("mean")
    meta = replay.get("meta") or {}
    terrain = (meta.get("terrain") or {}).get("source") or {}
    result.world = {
        "environment_id": meta.get("environment_id"),
        "terrain": {k: terrain.get(k) for k in ("source", "preset", "seed", "generator_version")},
    }
    return result


def git(*args: str) -> str | None:
    try:
        # A fixed git metadata command, as an argument list without a shell.
        out = subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", required=True, help="a canonical scenario name")
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        help='NAME=ARGS, e.g. "six-drones=--drone-count 6"; the first is the baseline',
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed-set", choices=tuple(SEED_SETS), default="canonical")
    seeds.add_argument("--seeds", help="comma-separated seeds instead of a named set")
    parser.add_argument("--duration-s", type=float, default=None, help="override durations")
    parser.add_argument("--binary", type=Path, default=RELEASE_BINARY)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    configs = {c.name: c for c in CANONICAL_SCENARIOS}
    if args.scenario not in configs:
        parser.error(f"unknown scenario {args.scenario!r}; one of {', '.join(configs)}")
    config = configs[args.scenario]
    if args.duration_s is not None:
        import dataclasses

        config = dataclasses.replace(config, duration_s=args.duration_s)
    names = [name for name, _ in args.variant]
    if len(set(names)) != len(names):
        parser.error("variant names must be distinct")
    if not args.binary.exists():
        print(f"{args.binary} missing: cargo build --release -p argusnet-cli", file=sys.stderr)
        return 2
    seed_list = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else list(SEED_SETS[args.seed_set])
    )

    runs: list[RunResult] = []
    with tempfile.TemporaryDirectory() as scratch:
        jobs = []
        for name, tokens in args.variant:
            extra, runtime_args = split_args(tokens)
            for seed in seed_list:
                work = Path(scratch) / name / str(seed)
                exported, error = export(config, extra, seed, work)
                if error is not None:
                    runs.append(RunResult(variant=name, seed=seed, error=error))
                    print(f"{name} / {seed}: {error}", flush=True)
                else:
                    jobs.append((name, seed, exported, work, runtime_args))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(
                    run_one, config, name, seed, exported, args.binary, work, runtime_args
                ): (name, seed)
                for name, seed, exported, work, runtime_args in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                name, seed = futures[future]
                try:
                    run = future.result()
                except Exception as exc:  # noqa: BLE001 -- recorded as a failed run
                    run = RunResult(variant=name, seed=seed, error=f"{type(exc).__name__}: {exc}")
                runs.append(run)
                state = run.error or ("ok" if run.passed_envelope else "outside envelope")
                print(f"{name} / {seed}: {state}", flush=True)

    body = compare(runs, names)
    worlds = {json.dumps(r.world, sort_keys=True) for r in runs if r.error is None}
    report = {
        "format": "argusnet.scenario-comparison",
        "version": 1,
        "scenario": config.name,
        "seed_set": None if args.seeds else args.seed_set,
        "duration_s": config.duration_s,
        "variant_args": {name: extra for name, extra in args.variant},
        **body,
        "identity": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "binary": str(args.binary),
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "world": [json.loads(w) for w in sorted(worlds)],
            "configs": {
                name: {
                    str(r.seed): r.config_sha256
                    for r in sorted(runs, key=lambda r: r.seed)
                    if r.variant == name
                }
                for name in names
            },
        },
    }
    report["runs"] = [
        {
            "variant": r.variant,
            "seed": r.seed,
            "error": r.error,
            "passed_envelope": r.passed_envelope,
            "wall_s": r.wall_s,
            "tick_mean_ms": r.tick_mean_ms,
            "config_sha256": r.config_sha256,
            "metrics": r.metrics,
        }
        for r in sorted(runs, key=lambda r: (names.index(r.variant), r.seed))
    ]
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report) + "\n")
    print(f"wrote {args.out}" + (f" and {args.markdown}" if args.markdown else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
