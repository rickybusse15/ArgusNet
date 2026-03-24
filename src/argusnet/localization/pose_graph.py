"""Pose graph for ArgusNet localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["PoseGraph", "PoseNode", "PoseEdge"]


@dataclass
class PoseNode:
    keyframe_id: str
    position: np.ndarray      # (3,) ENU metres  [x, y, z]
    yaw_rad: float            # heading (yaw only; SE2.5 representation)
    covariance: np.ndarray    # (4, 4) [x, y, z, yaw]


@dataclass
class PoseEdge:
    from_id: str
    to_id: str
    delta_position: np.ndarray  # (3,) relative translation [dx, dy, dz]
    delta_yaw: float
    information: np.ndarray     # (4, 4) inverse covariance [x, y, z, yaw]


class PoseGraph:
    """Gauss-Newton pose graph optimizer — SE2.5 (x, y, z, yaw).

    Altitude (z) is tracked as a fourth dimension alongside the standard
    SE2 (x, y, yaw) state.  Full 6-DOF SO(3) rotation is not modelled;
    the yaw-only heading approximation remains appropriate for near-level
    UAV flight.

    Uses scipy if available, otherwise falls back to numpy lstsq.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, PoseNode] = {}
        self._edges: List[PoseEdge] = []

    def add_node(self, node: PoseNode) -> None:
        self._nodes[node.keyframe_id] = node

    def add_edge(self, edge: PoseEdge) -> None:
        self._edges.append(edge)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def optimize(self, iterations: int = 5) -> Dict[str, np.ndarray]:
        """Run Gauss-Newton optimization. Returns {keyframe_id: [x, y, z, yaw]}."""
        if len(self._nodes) < 2 or not self._edges:
            return {k: np.append(v.position, v.yaw_rad) for k, v in self._nodes.items()}

        # Build index
        ids = list(self._nodes.keys())
        id_to_idx = {k: i for i, k in enumerate(ids)}
        n = len(ids)
        DOF = 4  # x, y, z, yaw

        # Initial state
        x = np.zeros(n * DOF)
        for i, kid in enumerate(ids):
            node = self._nodes[kid]
            x[i * DOF:(i + 1) * DOF] = np.append(node.position, node.yaw_rad)

        # Fix first node (anchor)
        fixed = 0

        for _ in range(iterations):
            # Build H (approximate hessian) and b (gradient) via GN
            H = np.zeros((n * DOF, n * DOF))
            b = np.zeros(n * DOF)

            for edge in self._edges:
                if edge.from_id not in id_to_idx or edge.to_id not in id_to_idx:
                    continue
                i = id_to_idx[edge.from_id]
                j = id_to_idx[edge.to_id]
                xi = x[i * DOF:(i + 1) * DOF]
                xj = x[j * DOF:(j + 1) * DOF]

                # Residual: measured - predicted relative pose
                pred_delta_pos = xj[:3] - xi[:3]
                pred_delta_yaw = xj[3] - xi[3]

                res = np.array([
                    edge.delta_position[0] - pred_delta_pos[0],
                    edge.delta_position[1] - pred_delta_pos[1],
                    edge.delta_position[2] - pred_delta_pos[2],
                    edge.delta_yaw - pred_delta_yaw,
                ])

                # Jacobian (simple linear case)
                J_i = -np.eye(DOF)
                J_j = np.eye(DOF)

                # Use diagonal of information matrix (simplified).
                # information is (4, 4); fall back to identity if shape mismatches
                # legacy data from before the SE2 → SE2.5 upgrade.
                if edge.information.shape == (4, 4):
                    info_diag = np.diag(edge.information)
                elif edge.information.shape == (3, 3):
                    # Legacy 3×3 edge: promote to 4-DOF by appending a unit weight for z.
                    info_diag = np.append(np.diag(edge.information), 1.0)
                else:
                    info_diag = np.ones(DOF)
                Omega = np.diag(info_diag)

                H[i * DOF:(i + 1) * DOF, i * DOF:(i + 1) * DOF] += J_i.T @ Omega @ J_i
                H[i * DOF:(i + 1) * DOF, j * DOF:(j + 1) * DOF] += J_i.T @ Omega @ J_j
                H[j * DOF:(j + 1) * DOF, i * DOF:(i + 1) * DOF] += J_j.T @ Omega @ J_i
                H[j * DOF:(j + 1) * DOF, j * DOF:(j + 1) * DOF] += J_j.T @ Omega @ J_j
                b[i * DOF:(i + 1) * DOF] += J_i.T @ Omega @ res
                b[j * DOF:(j + 1) * DOF] += J_j.T @ Omega @ res

            # Fix anchor node (set its rows/cols to identity)
            H[fixed * DOF:(fixed + 1) * DOF, :] = 0
            H[:, fixed * DOF:(fixed + 1) * DOF] = 0
            H[fixed * DOF:(fixed + 1) * DOF, fixed * DOF:(fixed + 1) * DOF] = np.eye(DOF)
            b[fixed * DOF:(fixed + 1) * DOF] = 0

            # Solve: H dx = b
            try:
                dx = np.linalg.solve(H + np.eye(n * DOF) * 1e-6, b)
            except np.linalg.LinAlgError:
                break
            x = x + dx
            if np.linalg.norm(dx) < 1e-6:
                break

        # Update nodes
        result = {}
        for i, kid in enumerate(ids):
            result[kid] = x[i * DOF:(i + 1) * DOF].copy()
        return result
