"""Backward-compatibility shim — imports from argusnet.world.obstacles and environment."""
from argusnet.world.obstacles import *  # noqa: F401, F403
from argusnet.world.obstacles import (
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    ObstaclePrimitive,
    OrientedBox,
    PolygonPrism,
    WallSegment,
)
# ObstacleLayer is defined in environment.py, not obstacles.py
from argusnet.world.environment import ObstacleLayer
