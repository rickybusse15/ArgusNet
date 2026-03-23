"""Backward-compatibility shim — imports from argusnet.sensing.ingestion.frame_stream."""
from argusnet.sensing.ingestion.frame_stream import *  # noqa: F401, F403
from argusnet.sensing.ingestion.frame_stream import (
    FileReplayIngestionAdapter,
    IngestionAdapter,
    LiveIngestionRunner,
    MQTTIngestionAdapter,
)
