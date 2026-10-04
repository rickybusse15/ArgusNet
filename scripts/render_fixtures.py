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
}
TIERS = ("low", "standard", "high")
HIGH_ENTITY_COUNT = 400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_hash(scene: Path) -> str:
    digest = hashlib.sha256()
    for name in ("scene_manifest.json", "replay/replay.json"):
        digest.update((scene / name).read_bytes())
    return digest.hexdigest()


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
    parser.add_argument("--fixtures", nargs="+", default=list(FIXTURES), choices=list(FIXTURES))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = {"base": args.out / "scene-base", "load": args.out / "scene-load"}
    prepare(scenes["base"])
    prepare(scenes["load"], HIGH_ENTITY_COUNT)

    runs = []
    for fixture in args.fixtures:
        scene_key, camera, view_mode = FIXTURES[fixture]
        scene = scenes[scene_key]
        for tier in args.tiers:
            still = args.out / f"{fixture}-{tier}.png"
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
        "frames_per_run": args.frames,
        "high_entity_count": HIGH_ENTITY_COUNT,
        "runs": runs,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
