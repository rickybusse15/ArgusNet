#!/usr/bin/env python3
"""Qualify a run profile on a real site: many seeds, frozen thresholds.

Imports the site, runs the profile twice per seed through a release
``argusnet-rs`` (the second run proves the first is reproducible), scores
every replay against the thresholds, and writes ``results.json`` and
``report.md``. Exits 0 only when the campaign qualifies: every qualification
and held-out seed passes every gate.

    cargo build --release --locked -p argusnet-cli
    python scripts/qualify_field_profile.py --output runs/field-v1

Tuning seeds (those the profile was composed on) are run and reported but
never counted. Use a new output directory for each campaign.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argusnet.core.io_limits import read_text_capped, sha256_file  # noqa: E402
from argusnet.evaluation.field_qualification import (  # noqa: E402
    REPORT_FORMAT,
    load_thresholds,
    markdown,
    score_replay,
    summarize,
)
from argusnet.world.world_import import import_world  # noqa: E402

#: The second set, chosen before its campaign ran (2026-11-13). The first
#: (101..1010 by 101, held out 9001..9005) failed the step gate on four
#: seeds; the fix was made on those results, so none of that set counts
#: again. Its four failing seeds are re-run as tuning seeds, with the two
#: the profile was composed on.
QUALIFICATION_SEEDS = [1117, 2239, 3343, 4451, 5563, 6673, 7789, 8807, 9923, 11003]
HELDOUT_SEEDS = [9101, 9102, 9103, 9104, 9105]
TUNING_SEEDS = [7, 42, 707, 1010, 9002, 9005]


def run_seed(binary: Path, config: dict, directory: Path, seed: int, profile: str) -> dict:
    """Run one seed twice; keep the replay only when asked to, its hash always."""
    directory.mkdir(parents=True)
    scenario = directory / "scenario.json"
    scenario.write_text(json.dumps({**config, "seed": seed}, indent=2) + "\n", encoding="utf-8")
    replays = []
    for name in ("replay.json", "repeat.json"):
        replay = directory / name
        completed = subprocess.run(
            [
                str(binary),
                "sim",
                "--quiet",
                "--config",
                str(scenario),
                "--profile",
                profile,
                "--replay",
                str(replay),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return {"seed": seed, "error": completed.stderr.strip() or "run failed"}
        replays.append(replay)
    hashes = [sha256_file(replay) for replay in replays]
    replays[1].unlink()
    return {"seed": seed, "replay": str(replays[0]), "hashes": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="a new output directory")
    parser.add_argument(
        "--site",
        type=Path,
        default=ROOT / "tests/fixtures/real/west_pleasant_view",
        help="a site directory holding canonical.json and its sources",
    )
    parser.add_argument("--profile", default="field_v1")
    parser.add_argument(
        "--thresholds", type=Path, default=ROOT / "docs/contracts/field_v1.thresholds.json"
    )
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/argusnet-rs")
    parser.add_argument("--seeds", type=int, nargs="+", default=QUALIFICATION_SEEDS)
    parser.add_argument("--heldout-seeds", type=int, nargs="*", default=HELDOUT_SEEDS)
    parser.add_argument("--tuning-seeds", type=int, nargs="*", default=TUNING_SEEDS)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--keep-replays",
        action="store_true",
        help="keep every replay (about 100 MB each); otherwise only their hashes",
    )
    args = parser.parse_args(argv)
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory; preserve earlier evidence")
    seeds = [*args.seeds, *args.heldout_seeds, *args.tuning_seeds]
    if len(set(seeds)) != len(seeds):
        parser.error("a seed appears in more than one of the three sets")
    thresholds = load_thresholds(args.thresholds)
    if thresholds["profile"] != args.profile:
        parser.error(f"{args.thresholds} is for {thresholds['profile']}, not {args.profile}")

    site = args.site.resolve(strict=True)
    canonical = json.loads(read_text_capped(site / "canonical.json"))
    world = import_world(
        site / canonical["dem"],
        site / canonical["osm"],
        output / "world",
        map_preset=canonical["map_preset"],
        origin=tuple(canonical["origin"]),
        terrain_preset=canonical["terrain_preset"],
        duration_s=args.duration_s,
    )
    config = json.loads(read_text_capped(output / "world/scenario.json"))

    heldout = set(args.heldout_seeds)
    tuning = set(args.tuning_seeds)
    runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(run_seed, binary, config, output / f"seed-{seed}", seed, args.profile)
            for seed in seeds
        ]
        for future in futures:
            ran = future.result()
            seed = ran["seed"]
            role = "tuning" if seed in tuning else "heldout" if seed in heldout else "qualification"
            if "error" in ran:
                runs.append(
                    {
                        "seed": seed,
                        "role": role,
                        "passed": False,
                        "failed": ["run"],
                        "measured": {},
                        "gates": {},
                        "error": ran["error"],
                    }
                )
                continue
            replay = Path(ran["replay"])
            scored = score_replay(
                json.loads(read_text_capped(replay)),
                thresholds,
                repeat_identical=ran["hashes"][0] == ran["hashes"][1],
            )
            if not args.keep_replays:
                replay.unlink()
            runs.append({"seed": seed, "role": role, "replay_sha256": ran["hashes"][0], **scored})
            print(
                f"seed {seed} ({role}): "
                + ("pass" if scored["passed"] else "FAIL " + ", ".join(scored["failed"])),
                flush=True,
            )

    report = {
        "format": REPORT_FORMAT,
        "profile": args.profile,
        "site": canonical["name"],
        "world_id": world["world_id"],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "binary_sha256": sha256_file(binary),
        "workflow_sha256": sha256_file(Path(__file__)),
        "thresholds_sha256": sha256_file(args.thresholds),
        "thresholds": thresholds,
        "duration_s": args.duration_s,
        "seeds": {
            "qualification": args.seeds,
            "heldout": args.heldout_seeds,
            "tuning": args.tuning_seeds,
        },
        "runs": runs,
        "summary": summarize(runs, thresholds, heldout_seeds=heldout),
        "limits": [
            "A simulated qualification: the rig, LiDAR and camera models are assumed, "
            "not calibrated against hardware.",
            "Truth is read to score the map and wire clearance, never by the run.",
            "Station siting and the POIs' horizontal places come from the scenario generator.",
            "Localization starts from the true pose plus noise; landing is a hover at the station.",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(
        json.dumps(
            {
                "results": str(output / "results.json"),
                "qualified": summary["qualified"],
                "passed": summary["passed"],
                "runs": summary["runs"],
                "reasons": summary["reasons"],
            }
        )
    )
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
