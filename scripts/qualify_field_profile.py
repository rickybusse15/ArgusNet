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

#: ``field_v1``: the fifth set, chosen before its campaign ran (2026-11-14). The fourth
#: set (23131..33149, held out 9301..9305) qualified, but on sensing that
#: put every pod LiDAR ground return about 0.7 m low; terrain returns were
#: put on the ray after it (found by analysis, not from its results), so
#: its evidence describes code that no longer runs. Seeds that informed a
#: fix never count again:
#: - the first set (101..1010 by 101, held out 9001..9005) failed the step
#:   gate on four seeds;
#: - the second (1117..11003, held out 9101..9105) failed the separation
#:   gate on one;
#: - the third (12007..22073, held out 9201..9205) failed it on another.
#: Each failing seed is re-run as a tuning seed, with the two the profile
#: was composed on.
FIELD_V1_SEEDS = {
    "qualification": [34253, 35363, 36467, 37571, 38693, 39799, 40903, 42013, 43117, 44221],
    "heldout": [9401, 9402, 9403, 9404, 9405],
    "tuning": [7, 42, 707, 1010, 9002, 9005, 4451, 13127],
}
#: ``field_v2``: the fourth set, chosen before its campaign ran (2026-11-23).
#: The first (45329..55291 by about 1100, held out 9501..9505) was stopped with
#: five seeds scored, all passing, when review of the imaging code found two
#: defects. The second (56393..66343, held out 9601..9605) qualified, on a build
#: that was then changed for what a second site showed: evidence for a build
#: that no longer exists counts for nothing. The third (78437..88321, held out
#: 9801..9805) was stopped before any seed was scored, when that site failed.
FIELD_V2_SEEDS = {
    "qualification": [89417, 90511, 91621, 92717, 93811, 94907, 96001, 97103, 98207, 99317],
    "heldout": [9901, 9902, 9903, 9904, 9905],
    "tuning": [7, 42, 707, 1010, 9002, 9005, 4451, 13127],
}
#: ``field_v3``: the fourth set, chosen before its campaign ran (2026-11-23).
#: The first (111409..121309, held out 10101..10105) failed the step gate on one
#: seed, 119107: a drone handed to its explorer at the top of its climb was put
#: where the explorer had run on to. The second (133403..143303, held out
#: 10301..10305) qualified, on a build whose routes were then changed for what
#: review found: a steep leg could step over a wire. The third (155401..165311,
#: held out 10501..10505) failed the home gate on one seed, 158701: a route's last
#: corners went by or round the pad. Those two are tuning seeds now.
FIELD_V3_SEEDS = {
    "qualification": [
        177409,
        178501,
        179603,
        180701,
        181813,
        182901,
        184003,
        185107,
        186211,
        187303,
    ],
    "heldout": [10701, 10702, 10703, 10704, 10705],
    "tuning": [7, 42, 707, 1010, 9002, 4451, 13127, 119107, 158701],
}
PROFILE_SEEDS = {
    "field_v1": FIELD_V1_SEEDS,
    "field_v2": FIELD_V2_SEEDS,
    "field_v3": FIELD_V3_SEEDS,
}
#: A site that has its own seeds, as ``(profile, site name)``. Montague City's
#: second set, chosen before any of it was run (2026-11-23). The first
#: (67421..77339, held out 9701..9705) failed on one seed, 74027, whose scan was
#: held open by a strip between two lines; it is a tuning seed now. The
#: profile's own tuning seeds were run on the site first, and never count.
SITE_SEEDS = {
    ("field_v2", "montague_city"): {
        "qualification": [
            100403,
            101503,
            102607,
            103703,
            104801,
            105907,
            107003,
            108107,
            109201,
            110311,
        ],
        "heldout": [10001, 10002, 10003, 10004, 10005],
        "tuning": [*FIELD_V2_SEEDS["tuning"], 74027],
    },
    # Its fourth set, chosen before any of it was run (2026-11-23). The first
    # three (122401..132301, 144409..154303, 166403..176303, held out 10201..,
    # 10401.., 10601..) passed on builds that no longer exist.
    ("field_v3", "montague_city"): {
        "qualification": [
            188401,
            189503,
            190607,
            191701,
            192811,
            193903,
            195001,
            196107,
            197203,
            198301,
        ],
        "heldout": [10801, 10802, 10803, 10804, 10805],
        "tuning": FIELD_V3_SEEDS["tuning"],
    },
}


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
        "--thresholds",
        type=Path,
        default=None,
        help="the profile's own in docs/contracts when not given",
    )
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/argusnet-rs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--heldout-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--tuning-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--keep-replays",
        action="store_true",
        help="keep every replay (about 100 MB each); otherwise only their hashes",
    )
    args = parser.parse_args(argv)
    if args.thresholds is None:
        args.thresholds = ROOT / f"docs/contracts/{args.profile}.thresholds.json"
    site = args.site.resolve(strict=True)
    canonical = json.loads(read_text_capped(site / "canonical.json"))
    chosen = SITE_SEEDS.get((args.profile, canonical["name"]), PROFILE_SEEDS.get(args.profile))
    if chosen is None and None in (args.seeds, args.heldout_seeds, args.tuning_seeds):
        parser.error(f"no seeds chosen for {args.profile}: give all three sets")
    if args.seeds is None:
        args.seeds = chosen["qualification"]
    if args.heldout_seeds is None:
        args.heldout_seeds = chosen["heldout"]
    if args.tuning_seeds is None:
        args.tuning_seeds = chosen["tuning"]
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
