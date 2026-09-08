"""Create a deterministic rendering fixture with separated, airborne markers.

The layout is synthetic display-test data, not a physical simulation result.
It exercises the shader that a terrain-only package never loads.
"""

import argparse
import json
import shutil
from pathlib import Path


def prepare(output: Path) -> None:
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
    (output / "replay").mkdir(exist_ok=True)
    (output / "replay/replay.json").write_text(json.dumps(replay))
    manifest = output / "scene_manifest.json"
    data = json.loads(manifest.read_text())
    data["replay"] = {"path": "replay/replay.json"}
    manifest.write_text(json.dumps(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
