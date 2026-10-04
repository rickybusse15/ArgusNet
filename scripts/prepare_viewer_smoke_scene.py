"""Create a deterministic rendering fixture with separated, airborne markers.

The layout is synthetic display-test data, not a physical simulation result.
It exercises the shader that a terrain-only package never loads.
"""

import argparse
import json
import shutil
from pathlib import Path


def add_entities(replay: dict, count: int) -> None:
    """Add `count` stationary tracks on a grid, each with its own id: G2's
    high-entity-count fixture. Display-test data, like the rest."""
    side = max(1, int(count**0.5))
    for frame in replay["frames"]:
        template = dict(frame["tracks"][0]) if frame["tracks"] else None
        if template is None:
            continue
        for i in range(count):
            track = dict(template)
            track["track_id"] = f"load-{i:04d}"
            track["position"] = [
                -1800 + (i % side) * 3600 / side,
                -1800 + (i // side) * 3600 / side,
                400,
            ]
            track["velocity"] = [0, 0, 0]
            frame["tracks"].append(track)


def prepare(output: Path, entities: int = 0) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "scene.smartscene", output, dirs_exist_ok=True)
    replay = json.loads((root / "tests/fixtures/replay_contract_fixture.json").read_text())
    for frame in replay["frames"]:
        for i, node in enumerate(frame["nodes"]):
            node["position"] = [-900 + i * 450, 0, 600]
            node["velocity"] = [0, 0, 0]
        for track in frame["tracks"]:
            track["position"] = [0, 800, 600]
            track["velocity"] = [0, 0, 0]
            track["measurement_std_m"] = 2.0
    if entities:
        add_entities(replay, entities)
    (output / "replay").mkdir(exist_ok=True)
    (output / "replay/replay.json").write_text(json.dumps(replay))
    manifest = output / "scene_manifest.json"
    data = json.loads(manifest.read_text())
    data["replay"] = {"path": "replay/replay.json"}
    manifest.write_text(json.dumps(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--entities", type=int, default=0, help="extra stationary tracks (G2 high entity count)"
    )
    args = parser.parse_args()
    prepare(args.output, args.entities)
