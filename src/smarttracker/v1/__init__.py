"""Backward-compatibility shim: re-exports argusnet.v1 proto bindings as smarttracker.v1."""
from argusnet.v1 import world_model_pb2 as tracker_pb2  # noqa: F401
from argusnet.v1 import world_model_pb2_grpc as tracker_pb2_grpc  # noqa: F401
