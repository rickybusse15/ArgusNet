"""G2 renderer fixtures and frame statistics (E2).

Renders the five G2 fixtures (overview, close view, low pitch, split mode,
high entity count) in each quality tier through the real headless renderer,
times frames after warm-up, and writes one manifest: per run the fixture,
tier, camera, view mode, viewport, scene hash, image hash, frame summary,
adapter, build profile and renderer class (software or hardware).

A software renderer (lavapipe/llvmpipe) is a shader-correctness and relative
cost check, not the 60 Hz acceptance budget: that is calibrated on named
reference hardware (VERIFICATION_AND_ROLLOUT.md §5, decision D-5). The
manifest says which it was.

    python scripts/render_fixtures.py OUT_DIR --viewer target/release/argusnet-viewer

The realism fixtures (R4) render worlds the native builder makes, with
buildings, power lines and trees, and need `--argusnet-rs`:

    python scripts/render_fixtures.py OUT --viewer target/release/argusnet-viewer \
        --argusnet-rs target/release/argusnet-rs --fixtures realism_dense realism_large \
        --width 1280 --height 720 --repeats 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_viewer_smoke_scene import prepare  # noqa: E402

# name -> (scene, camera, view mode)
FIXTURES = {
    "overview": ("base", "top-down", "real-world"),
    "close": ("base", "close", "real-world"),
    "low_pitch": ("base", "low-pitch", "real-world"),
    "split": ("base", "isometric", "split"),
    "high_entity": ("load", "isometric", "real-world"),
    # R4: a dense block grid with a power line and scattered trees, from
    # three cameras; and a regional map with 15,000 trees (the builder's cap).
    "realism_dense": ("dense", "isometric", "real-world"),
    "realism_dense_close": ("dense", "close", "real-world"),
    "realism_dense_low_pitch": ("dense", "low-pitch", "real-world"),
    "realism_large": ("large", "isometric", "real-world"),
    "realism_large_close": ("large", "close", "real-world"),
    # R5: the canonical real place, West Pleasant View, Colorado (a USGS
    # DEM and an OSM extract, tests/fixtures/real/west_pleasant_view).
    "realism_real": ("real", "isometric", "real-world"),
    "realism_real_close": ("real", "close", "real-world"),
    "realism_real_low_pitch": ("real", "low-pitch", "real-world"),
}
DEFAULT_FIXTURES = ("overview", "close", "low_pitch", "split", "high_entity")
# The worlds the realism fixtures render: scenario-config options.
REALISM_WORLDS = {
    "dense": {
        "map_preset": "medium",
        "terrain_preset": "urban_dense",
        "trees": True,
        "power_lines": True,
        "drone_count": 6,
        "poi_count": 8,
    },
    "large": {
        "map_preset": "regional",
        "terrain_preset": "alpine",
        "trees": True,
        "power_lines": True,
        "drone_count": 16,
        "poi_count": 8,
    },
}
# The real places the realism fixtures render: a fixture folder under
# tests/fixtures/real with a canonical.json.
REAL_WORLDS = {"real": "west_pleasant_view"}
TIERS = ("low", "standard", "high")
HIGH_ENTITY_COUNT = 400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_hash(scene: Path) -> str:
    digest = hashlib.sha256()
    for name in ("scene_manifest.json", "replay/replay.json"):
        digest.update((scene / name).read_bytes())
    return digest.hexdigest()


def build_realism_scene(argusnet_rs: Path, world: str, out: Path) -> None:
    """Run a 30 s scenario of `world` and package its scene in `out`."""
    config = out.with_suffix(".config.json")
    replay = out.with_suffix(".replay.json")
    config.write_text(
        json.dumps(
            {
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 7,
                "duration_s": 30.0,
                "options": REALISM_WORLDS[world],
            }
        )
    )
    for command in (
        [str(argusnet_rs), "sim", "--quiet", "--config", str(config), "--replay", str(replay)],
        [str(argusnet_rs), "build-scene", "--replay", str(replay), "--output", str(out)],
    ):
        subprocess.run(command, check=True, capture_output=True)


def build_real_scene(argusnet_rs: Path, world: str, out: Path) -> None:
    """Import the canonical real place for `world`, run it for 30 s and
    package its scene in `out`."""
    sys.path.insert(0, str(ROOT / "src"))
    from argusnet.world.world_import import import_world

    fixture = ROOT / "tests" / "fixtures" / "real" / REAL_WORLDS[world]
    canonical = json.loads((fixture / "canonical.json").read_text(encoding="utf-8"))
    folder = out.with_name(f"{out.name}-world")
    import_world(
        fixture / canonical["dem"],
        fixture / canonical["osm"],
        folder,
        map_preset=canonical["map_preset"],
        origin=tuple(canonical["origin"]),
        terrain_preset=canonical["terrain_preset"],
        seed=canonical["seed"],
        duration_s=30.0,
    )
    replay = folder / "replay.json"
    for command in (
        [
            str(argusnet_rs),
            "sim",
            "--quiet",
            "--config",
            str(folder / "scenario.json"),
            "--replay",
            str(replay),
        ],
        [str(argusnet_rs), "build-scene", "--replay", str(replay), "--output", str(out)],
    ):
        subprocess.run(command, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("--viewer", type=Path, default=ROOT / "rust/target/release/argusnet-viewer")
    parser.add_argument(
        "--frames", type=int, default=120, help="frames timed per run after warm-up"
    )
    parser.add_argument("--width", type=int, default=650)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    parser.add_argument(
        "--fixtures", nargs="+", default=list(DEFAULT_FIXTURES), choices=list(FIXTURES)
    )
    parser.add_argument(
        "--argusnet-rs", type=Path, help="the native CLI, to build the realism fixtures' worlds"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="runs per fixture and tier (protocol: 5)"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = {"base": args.out / "scene-base", "load": args.out / "scene-load"}
    needed = {FIXTURES[f][0] for f in args.fixtures}
    if "base" in needed:
        prepare(scenes["base"])
    if "load" in needed:
        prepare(scenes["load"], HIGH_ENTITY_COUNT)
    for world in REALISM_WORLDS:
        if world in needed:
            if args.argusnet_rs is None:
                parser.error(f"the {world} realism fixtures need --argusnet-rs")
            scenes[world] = args.out / f"scene-{world}"
            build_realism_scene(args.argusnet_rs, world, scenes[world])
    for world in REAL_WORLDS:
        if world in needed:
            if args.argusnet_rs is None:
                parser.error(f"the {world} realism fixtures need --argusnet-rs")
            scenes[world] = args.out / f"scene-{world}"
            build_real_scene(args.argusnet_rs, world, scenes[world])

    runs = []
    for fixture in args.fixtures:
        scene_key, camera, view_mode = FIXTURES[fixture]
        scene = scenes[scene_key]
        for tier, repeat in [(t, r) for t in args.tiers for r in range(args.repeats)]:
            suffix = f"-{repeat}" if args.repeats > 1 else ""
            still = args.out / f"{fixture}-{tier}{suffix}.png"
            command = [
                str(args.viewer),
                "--scene",
                str(scene),
                "--headless",
                "--output",
                str(still),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--camera",
                camera,
                "--view-mode",
                view_mode,
                "--quality",
                tier,
                "--frame-stats",
                str(args.frames),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                sys.stderr.write(result.stderr[-4000:])
                print(f"{fixture}/{tier}: render failed ({result.returncode})", file=sys.stderr)
                return 1
            stats = json.loads(still.with_suffix(".frames.json").read_text())
            summary = stats["summary"]
            runs.append(
                {
                    "fixture": fixture,
                    "tier": tier,
                    "repeat": repeat,
                    "camera": camera,
                    "view_mode": view_mode,
                    "viewport": stats["viewport"],
                    "scene_sha256": scene_hash(scene),
                    "image_sha256": sha256(still),
                    "summary": summary,
                    "warmup": stats["warmup"],
                    "profile": stats["profile"],
                    "renderer_class": stats["renderer_class"],
                    "adapter": stats["adapter"],
                }
            )
            print(
                f"{fixture:12} {tier:8} p50 {summary['p50_ms']:7.1f} ms"
                f"  p95 {summary['p95_ms']:7.1f}  p99 {summary['p99_ms']:7.1f}"
                f"  max {summary['max_ms']:7.1f}"
                f"  stalls {summary['stalls_over_100ms']}"
            )

    manifest = {
        "kind": "argusnet.g2-fixtures",
        "version": 1,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "frames_per_run": args.frames,
        "repeats": args.repeats,
        "high_entity_count": HIGH_ENTITY_COUNT,
        "runs": runs,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
