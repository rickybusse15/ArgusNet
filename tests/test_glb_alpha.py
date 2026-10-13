"""R0: a translucent overlay mesh blends; an opaque one stays OPAQUE.

glTF ignores vertex alpha unless the material's ``alphaMode`` is ``BLEND``,
so without it a layer's styled opacity never reached the viewer.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from argusnet.world._glb import SimpleMesh, write_glb
from argusnet.world._scene_geometry import mesh_from_polyline


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
    # Blended faces are culled from behind; opaque meshes keep both faces.
    assert _material(tmp_path / "blend.glb")["doubleSided"] is False
    assert _material(tmp_path / "opaque.glb")["doubleSided"] is True


def test_each_part_is_its_own_node(tmp_path: Path) -> None:
    write_glb(tmp_path / "parts.glb", [_triangle(0.45), _triangle(0.45)], name="buildings")
    data = (tmp_path / "parts.glb").read_bytes()
    (json_length,) = struct.unpack_from("<I", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    assert len(document["meshes"]) == 2
    assert [node["name"] for node in document["nodes"]] == ["buildings-0", "buildings-1"]
    assert document["scenes"][0]["nodes"] == [0, 1]


def test_a_line_strip_faces_up() -> None:
    # Translucent layers are single-sided, so a strip wound downward (as
    # water and roads once were) would be culled from every camera above it.
    strip = mesh_from_polyline(
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]),
        height_at=lambda x, y: 0.0,
        color_rgba=(0.2, 0.4, 0.9, 0.85),
        width_m=4.0,
    )
    corners = strip.positions.astype(np.float64)[strip.indices.reshape(-1, 3)]
    up = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]
    assert (up > 0.0).all()
    assert (strip.normals[:, 2] > 0.0).all()


def test_a_bent_line_strip_does_not_overlap_itself() -> None:
    # Translucent fragments do not write depth, so overlapping segment quads
    # would draw a darker seam at every bend. The joined strip's triangles
    # cover exactly its outline.
    points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 12.0]])
    strip = mesh_from_polyline(
        points, height_at=lambda x, y: 0.0, color_rgba=(0.2, 0.4, 0.9, 0.85), width_m=4.0
    )
    assert strip.positions.shape[0] == 2 * len(points)
    xy = strip.positions[:, :2].astype(np.float64)
    corners = xy[strip.indices.reshape(-1, 3)]
    u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    covered = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    outline = np.concatenate([xy[0::2], xy[1::2][::-1]])
    x, y = outline[:, 0], outline[:, 1]
    outline_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert covered == pytest.approx(outline_area, rel=1e-5)


def test_a_line_that_doubles_back_keeps_every_triangle_facing_up() -> None:
    # A mitre at a reversal would swap the strip's sides and wind triangles
    # downward, which a single-sided translucent layer culls.
    for points in (
        [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [10.0, 0.0], [5.0, 3.0], [5.0, 20.0]],
    ):
        strip = mesh_from_polyline(
            np.array(points),
            height_at=lambda x, y: 0.0,
            color_rgba=(0.2, 0.4, 0.9, 0.85),
            width_m=4.0,
        )
        corners = strip.positions.astype(np.float64)[strip.indices.reshape(-1, 3)]
        u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        assert ((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) > 0.0).all(), points


def test_random_lines_never_wind_a_triangle_downward() -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        points = rng.uniform(-30.0, 30.0, size=(int(rng.integers(2, 8)), 2))
        strip = mesh_from_polyline(
            points,
            height_at=lambda x, y: 0.0,
            color_rgba=(0.2, 0.4, 0.9, 0.85),
            width_m=float(rng.uniform(1.0, 10.0)),
        )
        corners = strip.positions.astype(np.float64)[strip.indices.reshape(-1, 3)]
        u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        assert ((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) > 0.0).all(), points.tolist()
