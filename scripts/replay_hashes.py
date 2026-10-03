#!/usr/bin/env python3
"""Replay hashes of the canonical scenarios: the behaviour-neutral proof.

A refactor that claims to change nothing a run produces (agent execution
plan §12.3, the A1.2 ownership slices) proves it by running every canonical
scenario through a release `argusnet-rs` before and after, and comparing the
replay bytes. This records them, or checks a build against a record:

    cargo build --release -p argusnet-cli
    python3 scripts/replay_hashes.py --out before.json        # on the base
    python3 scripts/replay_hashes.py --check before.json      # on the change

`--check` exits non-zero and names every scenario/seed whose replay differs.

`--lifecycle` checks A2 on every canonical scenario and seed instead: a run
reset at a seeded tick K replays byte-identical to a straight run (A2.1), and
a run resumed from a checkpoint taken at K (serialised to disk, read by a
freshly assembled process) produces exactly the straight run's frames from K
on (A2.2):

    python3 scripts/replay_hashes.py --lifecycle
Replays are deterministic for a fixed seed and config on one platform; hashes
from different platforms are not comparable (see PARITY_PORTABILITY.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profile_rust_tick import export_config  # noqa: E402

from argusnet.evaluation.scenarios import CANONICAL_SCENARIOS  # noqa: E402

RELEASE_BINARY = REPO_ROOT / "target" / "release" / "argusnet-rs"


EXECUTOR_ARGS: list[str] = []


def replay_hash(binary: Path, config_path: Path, replay: Path) -> str | None:
    completed = subprocess.run(
        [
            str(binary),
            "sim",
            "--quiet",
            "--config",
            str(config_path),
            "--replay",
            str(replay),
            *EXECUTOR_ARGS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 3:
        return None
    if completed.returncode != 0:
        raise RuntimeError(f"argusnet-rs failed ({completed.returncode}): {completed.stderr}")
    return hashlib.sha256(replay.read_bytes()).hexdigest()


def run_sim(binary: Path, config_path: Path, *extra: str) -> int:
    completed = subprocess.run(
        [str(binary), "sim", "--quiet", "--config", str(config_path), *extra, *EXECUTOR_ARGS],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 3):
        raise RuntimeError(f"argusnet-rs failed ({completed.returncode}): {completed.stderr}")
    return completed.returncode


def lifecycle(binary: Path, seeds: list[int]) -> list[str]:
    """A2 over the canonical scenarios: the failures, named."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        for config in CANONICAL_SCENARIOS:
            for seed in seeds:
                key = f"{config.name}/{seed}"
                work = Path(scratch) / config.name / str(seed)
                work.mkdir(parents=True)
                exported = export_config(config, seed, work)
                straight = work / "straight.json"
                if run_sim(binary, exported, "--replay", str(straight)) == 3:
                    print(f"{key}: unsupported", flush=True)
                    continue
                frames = json.loads(straight.read_text())["frames"]
                # A predetermined tick per scenario and seed, away from both ends.
                k = int(len(frames) * random.Random(key).uniform(0.2, 0.8))

                reset = work / "reset.json"
                run_sim(binary, exported, "--reset-at", str(k), "--replay", str(reset))
                reset_ok = reset.read_bytes() == straight.read_bytes()

                checkpoint = work / "checkpoint.json"
                run_sim(
                    binary, exported, "--checkpoint-at", str(k), "--checkpoint", str(checkpoint)
                )
                resumed = work / "resumed.json"
                run_sim(binary, exported, "--resume", str(checkpoint), "--replay", str(resumed))
                resumed_frames = json.loads(resumed.read_text())["frames"]
                resume_ok = resumed_frames == frames[k:]

                print(
                    f"{key}: k={k} reset {'equal' if reset_ok else 'DIFFERS'}, "
                    f"resume {'equal' if resume_ok else 'DIFFERS'}",
                    flush=True,
                )
                if not reset_ok:
                    failures.append(f"{key}: reset at {k} differs from a fresh run")
                if not resume_ok:
                    pairs = zip(resumed_frames, frames[k:], strict=False)
                    first = next(
                        (i for i, (a, b) in enumerate(pairs) if a != b),
                        min(len(resumed_frames), len(frames) - k),
                    )
                    failures.append(f"{key}: resumed from {k}, first differs at tick {k + first}")
    return failures


def collect(binary: Path, seeds: list[int]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for config in CANONICAL_SCENARIOS:
            for seed in seeds:
                work = Path(scratch) / config.name / str(seed)
                work.mkdir(parents=True)
                exported = export_config(config, seed, work)
                key = f"{config.name}/{seed}"
                hashes[key] = replay_hash(binary, exported, work / "replay.json")
                print(f"{key}: {hashes[key] or 'unsupported'}", flush=True)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,42", help="comma-separated seeds")
    parser.add_argument("--binary", type=Path, default=RELEASE_BINARY)
    parser.add_argument(
        "--executor",
        choices=("legacy", "scheduled", "lanes"),
        default="legacy",
        help="argusnet-rs's executor; scheduled (B2.2) and lanes (B2.3) must reproduce "
        "legacy's hashes",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="record hashes here")
    mode.add_argument("--check", type=Path, help="compare against a record")
    mode.add_argument(
        "--lifecycle", action="store_true", help="reset == fresh and resume == uninterrupted"
    )
    args = parser.parse_args()
    if not args.binary.exists():
        print(f"{args.binary} missing: cargo build --release -p argusnet-cli")
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s]
    EXECUTOR_ARGS[:] = ["--executor", args.executor]
    if args.lifecycle:
        failures = lifecycle(args.binary, seeds)
        for failure in failures:
            print(f"FAIL {failure}")
        print("lifecycle: " + ("all equal" if not failures else f"{len(failures)} failure(s)"))
        return 1 if failures else 0
    if args.check:
        record = json.loads(args.check.read_text())
        seeds = record["seeds"]
    hashes = collect(args.binary, seeds)
    if args.out:
        args.out.write_text(
            json.dumps(
                {"platform": platform.platform(), "seeds": seeds, "hashes": hashes}, indent=1
            )
            + "\n"
        )
        return 0
    if record["platform"] != platform.platform():
        print(f"warning: recorded on {record['platform']}, checking on {platform.platform()}")
    differ = [k for k, v in record["hashes"].items() if hashes.get(k) != v]
    missing = [k for k in hashes if k not in record["hashes"]]
    for key in differ:
        print(f"DIFFERS {key}: {record['hashes'][key]} -> {hashes.get(key)}")
    for key in missing:
        print(f"NOT IN RECORD {key}")
    print(f"{len(record['hashes']) - len(differ)}/{len(record['hashes'])} replays identical")
    return 1 if differ or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
