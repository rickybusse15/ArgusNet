from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SimpleMesh:
    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float32)
        normals = np.asarray(self.normals, dtype=np.float32)
        colors = np.asarray(self.colors, dtype=np.float32)
        indices = np.asarray(self.indices, dtype=np.uint32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n, 3)")
        if normals.shape != positions.shape:
            raise ValueError("normals must have the same shape as positions")
        if colors.ndim != 2 or colors.shape[0] != positions.shape[0] or colors.shape[1] not in (3, 4):
            raise ValueError("colors must have shape (n, 3) or (n, 4)")
        if indices.ndim != 1:
            raise ValueError("indices must be a flat array")
        if len(indices) % 3 != 0:
            raise ValueError("indices must describe triangles")

        if colors.shape[1] == 3:
            alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
            colors = np.concatenate([colors, alpha], axis=1)

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "indices", indices)

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


def merge_meshes(meshes: Sequence[SimpleMesh]) -> SimpleMesh:
    active = [mesh for mesh in meshes if mesh.vertex_count > 0 and mesh.indices.size > 0]
    if not active:
        return SimpleMesh(
            positions=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )

    positions = []
    normals = []
    colors = []
    indices = []
    vertex_offset = 0
    for mesh in active:
        positions.append(mesh.positions)
        normals.append(mesh.normals)
        colors.append(mesh.colors)
        indices.append(mesh.indices + vertex_offset)
        vertex_offset += mesh.vertex_count

    return SimpleMesh(
        positions=np.concatenate(positions, axis=0),
        normals=np.concatenate(normals, axis=0),
        colors=np.concatenate(colors, axis=0),
        indices=np.concatenate(indices, axis=0),
    )


def _byte_align(payload: bytes, alignment: int = 4, *, pad: bytes = b"\x00") -> bytes:
    remainder = len(payload) % alignment
    if remainder == 0:
        return payload
    return payload + (pad * (alignment - remainder))


def _accessor_bounds(array: np.ndarray) -> tuple[list[float], list[float]]:
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    return minimum.astype(float).tolist(), maximum.astype(float).tolist()


def write_glb(path: str | Path, mesh: SimpleMesh, *, name: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    position_bytes = np.ascontiguousarray(mesh.positions.astype(np.float32)).tobytes(order="C")
    normal_bytes = np.ascontiguousarray(mesh.normals.astype(np.float32)).tobytes(order="C")
    color_bytes = np.ascontiguousarray(np.clip(mesh.colors, 0.0, 1.0).astype(np.float32)).tobytes(order="C")
    index_bytes = np.ascontiguousarray(mesh.indices.astype(np.uint32)).tobytes(order="C")

    chunks: list[bytes] = []
    buffer_views = []
    accessors = []
    offset = 0

    def add_view(payload: bytes, *, target: int | None = None) -> tuple[int, int]:
        nonlocal offset
        padded = _byte_align(payload)
        chunk_offset = offset
        chunks.append(padded)
        offset += len(padded)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": chunk_offset,
                "byteLength": len(payload),
                **({"target": target} if target is not None else {}),
            }
        )
        return len(buffer_views) - 1, len(payload)

    position_view, _ = add_view(position_bytes, target=34962)
    position_min, position_max = _accessor_bounds(mesh.positions)
    accessors.append(
        {
            "bufferView": position_view,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC3",
            "min": position_min,
            "max": position_max,
        }
    )
    position_accessor = len(accessors) - 1

    normal_view, _ = add_view(normal_bytes, target=34962)
    accessors.append(
        {
            "bufferView": normal_view,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC3",
        }
    )
    normal_accessor = len(accessors) - 1

    color_view, _ = add_view(color_bytes, target=34962)
    accessors.append(
        {
            "bufferView": color_view,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC4",
        }
    )
    color_accessor = len(accessors) - 1

    index_view, _ = add_view(index_bytes, target=34963)
    accessors.append(
        {
            "bufferView": index_view,
            "componentType": 5125,
            "count": int(mesh.indices.shape[0]),
            "type": "SCALAR",
        }
    )
    index_accessor = len(accessors) - 1

    gltf = {
        "asset": {"version": "2.0", "generator": "smart_tracker.scene"},
        "scene": 0,
        "scenes": [{"nodes": [0], "name": name}],
        "nodes": [{"mesh": 0, "name": name}],
        "buffers": [{"byteLength": offset}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": [
            {
                "name": f"{name}-material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "COLOR_0": color_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
    }

    json_chunk = _byte_align(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad=b" ")
    bin_chunk = b"".join(chunks)

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total_length)
    json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
    output_path.write_bytes(header + json_header + json_chunk + bin_header + bin_chunk)


def triangle_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    for tri in indices.reshape(-1, 3):
        a = positions[int(tri[0])]
        b = positions[int(tri[1])]
        c = positions[int(tri[2])]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length > 1.0e-9:
            normal = normal / length
        normals[int(tri[0])] += normal
        normals[int(tri[1])] += normal
        normals[int(tri[2])] += normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1.0e-9)
    return normals / lengths


def solid_color_array(count: int, color_rgba: Iterable[float]) -> np.ndarray:
    color = np.asarray(list(color_rgba), dtype=np.float32)
    if color.shape != (4,):
        raise ValueError("color_rgba must have four channels")
    return np.repeat(color.reshape(1, 4), count, axis=0)


__all__ = [
    "SimpleMesh",
    "merge_meshes",
    "solid_color_array",
    "triangle_normals",
    "write_glb",
]
