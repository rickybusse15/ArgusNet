#!/usr/bin/env python3
"""Run the checked-in real site through import, simulation, client report and Review scene.

Produces a reproducibility record, not a flight or mission-quality certification.
Run with the project's Python after building argusnet-cli.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argusnet.core.io_limits import read_text_capped, sha256_file  # noqa: E402
from argusnet.evaluation.assurance import build_client_package  # noqa: E402
from argusnet.evaluation.replay import validate_replay_with_schema  # noqa: E402
from argusnet.world.world_import import import_world  # noqa: E402


def run(binary: Path, *args: str) -> None:
    subprocess.run([str(binary), *args], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="a new output directory")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/argusnet-rs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42])
    parser.add_argument("--duration-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory; preserve earlier evidence")
    if not args.duration_s > 0 or args.duration_s == float("inf"):
        parser.error("--duration-s must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")

    site = ROOT / "tests/fixtures/real/west_pleasant_view"
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
    config_path = output / "world/scenario.json"
    config = json.loads(read_text_capped(config_path))
    result: dict = {
        "format": "argusnet.real-site-inspection/v1",
        "site": canonical["name"],
        "world_id": world["world_id"],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "binary_sha256": sha256_file(binary),
        "workflow_sha256": sha256_file(Path(__file__)),
        "duration_s": args.duration_s,
        "checks": ["replay schema", "same-seed replay equality", "package artifact hashes"],
        "runs": [],
        "limits": [
            "Reproducibility checks do not certify mission completion or map quality.",
            "Scenario setup still uses truth for some routes and inspection points.",
            "Landing, localization integrity and evidence sufficiency remain unavailable.",
        ],
    }
    for seed in args.seeds:
        directory = output / f"seed-{seed}"
        directory.mkdir()
        config["seed"] = seed
        scenario = directory / "scenario.json"
        scenario.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        common = (
            "sim",
            "--quiet",
            "--config",
            str(scenario),
            "--world-knowledge",
            "none",
            "--planning-policy",
            "contingency_v1",
            "--energy-profile",
            "flown_v1",
        )
        replay = directory / "replay.json"
        repeat = directory / "repeat.json"
        run(binary, *common, "--replay", str(replay))
        run(binary, *common, "--replay", str(repeat))
        if sha256_file(replay) != sha256_file(repeat):
            raise RuntimeError(f"seed {seed}: repeated run differs; both replays preserved")
        document = json.loads(read_text_capped(replay))
        errors = validate_replay_with_schema(document)
        if errors:
            raise RuntimeError(f"seed {seed}: replay schema errors: {errors[:5]}")
        package = build_client_package(document)
        package_dir = directory / "client-package"
        package.write(package_dir)
        manifest = json.loads(package.manifest)
        for artifact in manifest["artifacts"]:
            path = package_dir / artifact["path"]
            if sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"package artifact hash mismatch: {path}")
        run(
            binary,
            "build-scene",
            "--replay",
            str(package_dir / "replay.json"),
            "--output",
            str(directory / "review.smartscene"),
        )
        report = json.loads(package.assurance)
        last = document["frames"][-1]
        belief = (last.get("mapping_state") or {}).get("belief") or {}
        result["runs"].append(
            {
                "seed": seed,
                "replay_sha256": sha256_file(replay),
                "package_manifest_sha256": hashlib.sha256(package.manifest).hexdigest(),
                "repeat_identical": True,
                "frames": len(document["frames"]),
                "mission_status": report["status"],
                "outcome": report.get("outcome"),
                "completion_rate": report.get("completion_rate"),
                "energy_reserve_min": report.get("energy_reserve_min"),
                "belief": belief,
                "unavailable_fields": report["unavailable_fields"],
                "review_scene": str((directory / "review.smartscene").relative_to(output)),
            }
        )
        (output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"results": str(output / "results.json"), "runs": len(result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
