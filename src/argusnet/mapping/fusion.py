"""Bearing triangulation and position fusion for ArgusNet mapping.

Re-exports the triangulation functions from localization.state so that the
mapping subsystem has a stable import path independent of the filter internals.
"""

from __future__ import annotations

from argusnet.localization.state import (
    TriangulatedEstimate,
    fuse_bearing_cluster,
    infer_measurement_std,
    triangulate_bearings,
)

__all__ = [
    "TriangulatedEstimate",
    "fuse_bearing_cluster",
    "infer_measurement_std",
    "triangulate_bearings",
]
