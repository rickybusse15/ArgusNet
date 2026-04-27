"""SE(3) transform tree for ArgusNet.

Provides rigid-body transforms (rotation + translation) that can be
composed and inverted, wrapping the WGS84/ENU frame machinery in
``argusnet.core.frames``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from argusnet.core.geometry import rotation_matrix_z
from argusnet.core.types import Vector3

__all__ = [
    "SE3",
    "TransformTree",
    "identity_se3",
    "compose",
    "invert",
]


@dataclass
class SE3:
    """A rigid-body transform: rotation R (3×3) and translation t (3,).

    Represents the transform from frame A to frame B:
        p_B = R @ p_A + t
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    """3×3 rotation matrix."""

    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Translation vector (metres)."""

    def apply(self, point: Vector3) -> np.ndarray:
        """Transform a point from the source frame to the target frame."""
        return self.R @ np.asarray(point, dtype=float) + self.t

    def apply_direction(self, direction: Vector3) -> np.ndarray:
        """Rotate a direction vector (no translation)."""
        return self.R @ np.asarray(direction, dtype=float)

    def __matmul__(self, other: SE3) -> SE3:
        """Compose two transforms: self ∘ other."""
        return compose(self, other)


def identity_se3() -> SE3:
    """Return the identity transform."""
    return SE3(R=np.eye(3), t=np.zeros(3))


def compose(a: SE3, b: SE3) -> SE3:
    """Return the composed transform a ∘ b (apply b first, then a)."""
    return SE3(R=a.R @ b.R, t=a.R @ b.t + a.t)


def invert(tf: SE3) -> SE3:
    """Return the inverse of *tf*."""
    R_inv = tf.R.T
    return SE3(R=R_inv, t=-(R_inv @ tf.t))


def from_xyz_yaw(x: float, y: float, z: float, yaw_rad: float) -> SE3:
    """Build an SE3 from XYZ translation and a yaw rotation about Z."""
    return SE3(R=rotation_matrix_z(yaw_rad), t=np.array([x, y, z], dtype=float))


class TransformTree:
    """Named transform graph supporting arbitrary frame hierarchies.

    Frames are identified by string names.  The root frame (world) is
    implicitly at the identity transform.  Only parent→child edges are
    stored; world→frame lookups walk up to the root.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._tf: dict[str, SE3] = {}  # child → transform from parent to child

    def register(self, frame: str, parent: str, tf: SE3) -> None:
        """Register *frame* as a child of *parent* with transform *tf*."""
        self._parent[frame] = parent
        self._tf[frame] = tf

    def lookup(self, target_frame: str, source_frame: str) -> SE3:
        """Return the transform from *source_frame* to *target_frame*.

        Walks up to "world" for both frames and composes the path.
        Only supports tree-structured (acyclic) graphs.
        """

        def chain_to_world(frame: str) -> list[SE3]:
            tfs: list[SE3] = []
            while frame in self._parent:
                tfs.append(self._tf[frame])
                frame = self._parent[frame]
            return tfs

        # source → world chain (need to invert each)
        s_chain = chain_to_world(source_frame)
        # target → world chain
        t_chain = chain_to_world(target_frame)

        # source→world = invert(s_chain[-1]) ∘ ... ∘ invert(s_chain[0])
        tf = identity_se3()
        for t in s_chain:
            tf = compose(tf, invert(t))
        # world→target = t_chain[-1] ∘ ... ∘ t_chain[0]
        for t in reversed(t_chain):
            tf = compose(tf, t)
        return tf

    def transform_point(self, point: Vector3, source_frame: str, target_frame: str) -> np.ndarray:
        return self.lookup(target_frame, source_frame).apply(point)
