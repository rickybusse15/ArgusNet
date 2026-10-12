"""R0: a translucent overlay mesh blends; an opaque one stays OPAQUE.

glTF ignores vertex alpha unless the material's ``alphaMode`` is ``BLEND``,
so without it a layer's styled opacity never reached the viewer.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from argusnet.world._glb import SimpleMesh, write_glb


def _material(path: Path) -> dict:
    data = path.read_bytes()
    (json_length,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20 : 20 + json_length])["materials"][0]


def _triangle(alpha: float) -> SimpleMesh:
    return SimpleMesh(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        colors=np.array([[0.9, 0.9, 0.9, alpha]] * 3, dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
    )


def test_a_translucent_mesh_blends_and_an_opaque_one_does_not(tmp_path: Path) -> None:
    write_glb(tmp_path / "blend.glb", _triangle(0.45), name="buildings")
    write_glb(tmp_path / "opaque.glb", _triangle(1.0), name="terrain-base")
    assert _material(tmp_path / "blend.glb")["alphaMode"] == "BLEND"
    assert "alphaMode" not in _material(tmp_path / "opaque.glb")
