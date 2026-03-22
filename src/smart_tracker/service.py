"""Backward-compatibility shim — imports from argusnet.adapters.argusnet_grpc."""
from argusnet.adapters.argusnet_grpc import *  # noqa: F401, F403
from argusnet.adapters.argusnet_grpc import (
    REJECT_DUPLICATE_NODE,
    REJECT_EXCESS_BEARING_STD,
    REJECT_FUSION_FAILURE,
    REJECT_INSUFFICIENT_CLUSTER,
    REJECT_INVALID_BEARING_STD,
    REJECT_INVALID_DIRECTION,
    REJECT_INVALID_TARGET,
    REJECT_LOW_CONFIDENCE,
    REJECT_TIMESTAMP_SKEW,
    REJECT_UNKNOWN_NODE,
    REJECT_WEAK_GEOMETRY,
    TrackerConfig,
    TrackingService,
)
