"""`argusnet-rs build-scene` builds the package `argusnet build-scene` does (D2).

The viewer builds scene packages in Rust (`argusnet_sim::scene_build`), so a
prior-world run needs no Python interpreter. This checks the Rust package
against the Python one for the same replay: the same manifest, styles and
environment, and per layer the same vertices, normals and colours (to float
rounding) and the same triangles in the same order and winding. The winding
includes the terrain's downward-facing triangles (`docs/KNOWN_GAPS.md`): the
viewer's lighting is tuned against them, so the Rust builder keeps them.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from argusnet.world.scene_loader import build_scene_from_replay, load_scene_manifest
from tests.test_scene import _sample_replay_document


def _argusnet_rs() -> Path:
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    return binary


def _glb(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = path.read_bytes()
    (json_length,) = struct.unpack_from("<I", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    binary = data[20 + json_length + 8 :]
    primitive = document["meshes"][0]["primitives"][0]

    def accessor(index: int) -> np.ndarray:
        acc = document["accessors"][index]
        view = document["bufferViews"][acc["bufferView"]]
        width = {"VEC3": 3, "VEC4": 4, "SCALAR": 1}[acc["type"]]
        dtype = {5126: np.float32, 5125: np.uint32}[acc["componentType"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return np.frombuffer(
            binary, dtype=dtype, count=acc["count"] * width, offset=offset
        ).reshape(-1, width)

    attributes = {name: accessor(index) for name, index in primitive["attributes"].items()}
    return attributes, accessor(primitive["indices"]).reshape(-1)


def _replay_with_obstacles() -> dict:
    replay = _sample_replay_document()
    replay["meta"]["occluding_objects"] = [
        {
            "object_id": "b1",
            "kind": "polygon-prism-v1",
            "blocker_type": "building",
            "footprint_xy_m": [[-40.0, -30.0], [10.0, -30.0], [10.0, 20.0], [-40.0, 20.0]],
            "base_elevation_m": 100.0,
            "top_elevation_m": 135.0,
        },
        {
            "object_id": "w1",
            "kind": "polygon-prism-v1",
            "blocker_type": "wall",
            # The flat form older packages carry.
            "footprint_xy_m": [30.0, -50.0, 32.0, -50.0, 32.0, 40.0, 30.0, 40.0],
            "height_agl_m": 4.0,
        },
        {
            "object_id": "v1",
            "kind": "cylinder-v1",
            "blocker_type": "vegetation",
            "center_x_m": -60.0,
            "center_y_m": 40.0,
            "radius_m": 12.0,
            "height_agl_m": 9.0,
        },
    ]
    return replay


def test_the_rust_scene_package_matches_the_python_one(tmp_path: Path) -> None:
    binary = _argusnet_rs()
    replay = _replay_with_obstacles()
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    python_root, rust_root = tmp_path / "python", tmp_path / "rust"
    build_scene_from_replay(replay, python_root)
    subprocess.run(
        [str(binary), "build-scene", "--replay", str(replay_path), "--output", str(rust_root)],
        check=True,
        capture_output=True,
    )

    python_manifest = load_scene_manifest(python_root / "scene_manifest.json")
    rust_manifest = load_scene_manifest(rust_root / "scene_manifest.json")
    for key in (
        "format_version",
        "scene_id",
        "bounds_xy_m",
        "runtime_crs",
        "source_crs_id",
        "layers",
        "replay",
        "metadata",
    ):
        assert rust_manifest[key] == python_manifest[key], key
    assert rust_manifest["build"]["source_kind"] == "synthetic"

    for name in ("metadata/style.json", "metadata/environment.json"):
        python_doc = json.loads((python_root / name).read_text(encoding="utf-8"))
        rust_doc = json.loads((rust_root / name).read_text(encoding="utf-8"))
        assert rust_doc.keys() == python_doc.keys(), name
        for key in python_doc:
            if key == "replay_meta":
                # Python's copy passes through `_json_safe`; the values agree.
                assert rust_doc[key]["occluding_objects"] == python_doc[key]["occluding_objects"]
                continue
            assert rust_doc[key] == python_doc[key], (name, key)

    for layer in python_manifest["layers"]:
        python_mesh, python_indices = _glb(python_root / layer["asset_path"])
        rust_mesh, rust_indices = _glb(rust_root / layer["asset_path"])
        for attribute in ("POSITION", "NORMAL", "COLOR_0"):
            np.testing.assert_allclose(
                rust_mesh[attribute],
                python_mesh[attribute],
                atol=1.0e-4,
                err_msg=f"{layer['id']} {attribute}",
            )
        np.testing.assert_array_equal(rust_indices, python_indices, err_msg=layer["id"])


def test_build_scene_refuses_a_replay_without_terrain(tmp_path: Path) -> None:
    binary = _argusnet_rs()
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps({"meta": {"dt_s": 0.5}, "frames": [{}]}), encoding="utf-8")
    result = subprocess.run(
        [str(binary), "build-scene", "--replay", str(replay_path), "--output", str(tmp_path / "o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "terrain.viewer_mesh" in result.stderr


@pytest.mark.parametrize("frame_count", ["wrong", "2", 1.5, None])
def test_build_scene_refuses_a_replay_python_would_reject(
    tmp_path: Path, frame_count: object
) -> None:
    """The replay contract is checked before packaging, as Python checks it."""
    binary = _argusnet_rs()
    replay = _replay_with_obstacles()
    replay["meta"]["frame_count"] = (
        len(replay["frames"]) + 1 if frame_count == "wrong" else frame_count
    )
    with pytest.raises(ValueError, match="frame_count"):
        build_scene_from_replay(replay, tmp_path / "python")
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    output = tmp_path / "rust"
    result = subprocess.run(
        [str(binary), "build-scene", "--replay", str(replay_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "frame_count" in result.stderr
    assert not output.exists()
