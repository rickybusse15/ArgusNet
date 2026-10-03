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
Replays are deterministic for a fixed seed and config on one platform; hashes
from different platforms are not comparable (see PARITY_PORTABILITY.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


def replay_hash(binary: Path, config_path: Path, replay: Path) -> str | None:
    completed = subprocess.run(
        [str(binary), "sim", "--quiet", "--config", str(config_path), "--replay", str(replay)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 3:
        return None
    if completed.returncode != 0:
        raise RuntimeError(f"argusnet-rs failed ({completed.returncode}): {completed.stderr}")
    return hashlib.sha256(replay.read_bytes()).hexdigest()


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="record hashes here")
    mode.add_argument("--check", type=Path, help="compare against a record")
    args = parser.parse_args()
    if not args.binary.exists():
        print(f"{args.binary} missing: cargo build --release -p argusnet-cli")
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s]
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
